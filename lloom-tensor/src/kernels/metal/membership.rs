//! Allocation-local residency lifetime. Records are pre-registered on the
//! shared host retirement service; token Drop publishes only metadata. Native
//! submission adapters still have to retain the token through terminality.
//!
//! Host metadata growth is presently explicit, but not yet charged to a model.
//! This source tranche does not enable family admission or claim free Drop.
use super::{Buffer, ResidencySet};
use crate::allocation_owner::AllocationLease;
use crate::completion_wake::Wake;
use crate::host_anchor::{AnchorError, HostAnchorPermit, HostAnchorRegistry};
use crate::retirement_owners::FailureSignal;
use crate::retirement_registry::RegistryError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLAllocation;
use parking_lot::Mutex;
use std::sync::Arc;

type Raw = Retained<ProtocolObject<dyn MTLAllocation>>;
type Lease = AllocationLease<Raw>;
type Registry = HostAnchorRegistry<Cleanup, Anchors>;
type Permit = HostAnchorPermit<Cleanup, Anchors>;

/// Inline ordinary-allocation record; retained groups grow only at allocation
/// birth. prepare_push is fallible and never consumes the incoming owner.
enum Members<T> {
    Empty,
    One(T),
    Many(Vec<T>),
}
impl<T> Members<T> {
    fn prepare_push(&mut self) -> Result<(), RegistryError> {
        match self {
            Self::Empty => Ok(()),
            Self::One(_) => {
                let mut items = Vec::new();
                items
                    .try_reserve_exact(2)
                    .map_err(|_| RegistryError::MetadataAllocation)?;
                if let Self::One(old) = std::mem::replace(self, Self::Empty) {
                    items.push(old);
                }
                *self = Self::Many(items);
                Ok(())
            }
            Self::Many(items) => items
                .try_reserve(1)
                .map_err(|_| RegistryError::MetadataAllocation),
        }
    }
    fn push_prepared(&mut self, item: T) {
        match self {
            Self::Empty => *self = Self::One(item),
            Self::Many(items) => items.push(item),
            Self::One(_) => unreachable!("prepare_push precedes owner transfer"),
        }
    }
    fn iter(&self) -> impl Iterator<Item = &T> {
        let (one, many) = match self {
            Self::Empty => (None, &[][..]),
            Self::One(item) => (Some(item), &[][..]),
            Self::Many(items) => (None, items.as_slice()),
        };
        one.into_iter().chain(many)
    }
}
struct Cleanup {
    residency: Arc<ResidencySet>,
    keys: Members<Raw>,
}
struct Anchors(Members<Lease>);
// SAFETY: these are retained, immutable engine-created Metal allocations. The
// residency mutation itself is serialized by ResidencySet's existing mutex.
unsafe impl Send for Cleanup {}
unsafe impl Sync for Cleanup {}
unsafe impl Send for Anchors {}
unsafe impl Sync for Anchors {}
impl Drop for Cleanup {
    fn drop(&mut self) {
        // This C runs on host retirement while A still retains all raw/charges.
        // Extraction keeps final releases outside the membership mutex.
        drop(
            self.residency
                .remove_allocations_owned(self.keys.iter().map(|raw| &**raw)),
        );
    }
}

#[derive(Debug)]
pub(crate) enum MembershipError {
    Registry(RegistryError),
    Anchor(AnchorError),
    Edit,
}
impl std::fmt::Display for MembershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "membership retirement: {self:?}")
    }
}

/// Arc keeps Buffer Clone/view infallible. Its final metadata deallocation is
/// still on the dropping thread; native/raw/charge cleanup is not.
#[derive(Clone)]
pub(crate) struct Membership(Arc<Token>);
struct Token {
    permit: Permit,
}
impl std::fmt::Debug for Membership {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Membership")
            .field(&Arc::as_ptr(&self.0))
            .finish()
    }
}

/// Namespace-local retention group, never a namespace/Commands backedge. Legacy
/// backing constructors get an independent group with the same policy; an
/// AllocationNamespace shares one group across all its backings.
pub(crate) struct MembershipDomain {
    residency: Arc<ResidencySet>,
    retain: bool,
    registry: Mutex<Option<Arc<Registry>>>,
    group: Mutex<Option<Membership>>,
}
impl MembershipDomain {
    pub(crate) fn new(residency: Arc<ResidencySet>, retain: bool) -> Arc<Self> {
        Arc::new(Self {
            residency,
            retain,
            registry: Mutex::new(None),
            group: Mutex::new(None),
        })
    }
    fn registry(&self) -> Result<Arc<Registry>, MembershipError> {
        if let Some(registry) = self.registry.lock().as_ref() {
            return Ok(registry.clone());
        }
        // Service setup/growth is outside our namespace mutex. An unused
        // competing candidate contains no resource records.
        let candidate = Arc::new(
            Registry::new(
                Arc::new(FailureSignal::default()),
                Arc::new(Wake::default()),
            )
            .map_err(MembershipError::Registry)?,
        );
        let selected = {
            let mut slot = self.registry.lock();
            slot.get_or_insert_with(|| candidate.clone()).clone()
        };
        drop(candidate);
        Ok(selected)
    }
    fn empty_token(&self) -> Result<Membership, MembershipError> {
        let registry = self.registry()?;
        // TODO #952: retain a typed admitted host metadata permit in each slab.
        // This is NOT model memory admission; allocation charges remain separate.
        match registry.reserve(
            Cleanup {
                residency: self.residency.clone(),
                keys: Members::Empty,
            },
            Anchors(Members::Empty),
            |_| Ok(()),
        ) {
            Ok(permit) => Ok(Membership(Arc::new(Token { permit }))),
            Err(rejected) => {
                // Empty C cannot unregister anything; raw owners have not moved.
                drop(rejected.owner);
                Err(MembershipError::Anchor(rejected.error))
            }
        }
    }
    fn token(&self) -> Result<Membership, MembershipError> {
        if !self.retain {
            return self.empty_token();
        }
        if let Some(group) = self.group.lock().as_ref() {
            return Ok(group.clone());
        }
        let candidate = self.empty_token()?;
        let selected = {
            let mut slot = self.group.lock();
            slot.get_or_insert_with(|| candidate.clone()).clone()
        };
        drop(candidate); // Empty losing record retires outside namespace lock.
        Ok(selected)
    }
    pub(crate) fn for_buffer(&self, buffer: &Buffer) -> Result<Membership, MembershipError> {
        // SAFETY: MTLBuffer implements MTLAllocation; protocol upcast only.
        let allocation = unsafe {
            &*(buffer.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.for_allocation(allocation, buffer.allocation_owner().cloned())
    }
    pub(crate) fn for_allocation(
        &self,
        allocation: &ProtocolObject<dyn MTLAllocation>,
        owner: Option<crate::allocation_owner::AllocationOwner>,
    ) -> Result<Membership, MembershipError> {
        let token = self.token()?;
        // Both protocol retains exist before entering the prepared record edit.
        let raw: Raw = unsafe { Retained::retain(allocation as *const _ as *mut _) }
            .expect("borrowed live native allocation");
        let input = (raw.clone(), AllocationLease::new(raw, owner));
        match token.0.permit.edit(input, |cleanup, anchors, input| {
            cleanup.keys.prepare_push()?;
            anchors.0.prepare_push()?;
            let (key, lease) = input.take().expect("prepared owner input");
            cleanup.keys.push_prepared(key);
            anchors.0.push_prepared(lease);
            Ok::<_, RegistryError>(())
        }) {
            Ok((Ok(()), None)) => Ok(token),
            Ok((Err(error), input)) => {
                drop(input);
                Err(MembershipError::Registry(error))
            }
            Ok((Ok(()), Some(input))) => {
                drop(input);
                Err(MembershipError::Edit)
            }
            Err(error) => {
                drop(error);
                Err(MembershipError::Edit)
            }
        }
    }
}
