//! `tuning.instrument.late_bind` (`LLOOM_LATE_BIND`, read once): does anything
//! bind a pooled buffer after its last handle dropped? (§15.1m)
//!
//! The pool stamps a released buffer with the epoch open at the drop and
//! offers it again once that epoch has retired (§6.3b). That is sound iff
//! every GPU use of the buffer was encoded before the drop: a use encoded
//! later sits in a later command buffer, which the stamp does not cover, and
//! if the buffer has meanwhile gone to a new tenant — a host upload writes its
//! bytes the instant it is acquired — that use reads the new tenant's bytes.
//!
//! This keeps the set of released buffers (parked or free) and checks every
//! compute bind and blit against it. A hit is a use after release, counted
//! and, for the first few, printed with the kernel label and the epochs.
//! `report()` gives the totals.
//!
//! Off, every entry point is one relaxed load. Never the shipping path.

use crate::retire::Epoch;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::Buffer;

#[cfg(test)]
thread_local! {
    /// Armed by a test on its own thread, where the tuning knob cannot be set
    /// and a process-wide arm would reach every other test.
    static ARMED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn armed() -> bool {
    #[cfg(test)]
    if ARMED_HERE.with(|a| a.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().instrument.late_bind)
}

/// Allocation-local managed diagnostic state, NOT reuse/native authority.
/// Independent Arc has no physical/namespace/pool/root edge. Cold registration
/// and final native-baseline removal own the map lock; managed acquire/release
/// publish only atoms. Global map readers never clone/drop an Arc under their
/// lock: they copy an acquired stamp while its map owner remains rooted.
pub(super) struct ManagedRelease {
    released: AtomicBool,
    epoch: AtomicU64,
}
impl ManagedRelease {
    fn stamp(&self, epoch: Epoch) {
        self.epoch.store(epoch.0, Ordering::Relaxed);
        self.released.store(true, Ordering::Release);
    }
    pub(super) fn acquired(&self) {
        self.released.store(false, Ordering::Release);
    }
    fn stamp_if_released(&self) -> Option<Epoch> {
        self.released
            .load(Ordering::Acquire)
            .then(|| Epoch(self.epoch.load(Ordering::Relaxed)))
    }
}
/// Sole allocation-local requested-byte receipt. State's Arc frees before its
/// receipt, after exact baseline removal of the map's extra state owner.
/// Published node independently retains the same receipt if C unwinds before
/// exact removal; that unreachable diagnostic stays charged, not early-refunded.
pub(super) struct ManagedRegistration {
    state: Arc<ManagedRelease>,
    _charge: crate::memory_admission::PhysicalCharge,
}
impl ManagedRegistration {
    pub(super) fn state(&self) -> &ManagedRelease {
        &self.state
    }
}
/// Sole preallocated map-node credit. It is constructed once beside the
/// registration receipt and consumed once by cold publication.
pub(super) struct ManagedNodeCredit(Option<Box<ManagedNode>>);
impl Drop for ManagedNodeCredit {
    fn drop(&mut self) {
        if let Some(mut node) = self.0.take() {
            let charge = node._charge.take();
            drop(node);
            drop(charge);
        }
    }
}
struct ManagedNode {
    native_key: usize,
    state: Option<Arc<ManagedRelease>>,
    left: Option<Box<ManagedNode>>,
    right: Option<Box<ManagedNode>>,
    height: usize,
    // Independently retained across PhysicalRecord C unwind before removal.
    // Normal removal still has the allocation-local receipt through Box free.
    _charge: Option<crate::memory_admission::PhysicalCharge>,
}
#[derive(Default)]
struct ReleasedRegistry {
    // Explicit legacy diagnostic allocations remain outside rich admission.
    legacy: HashMap<usize, Epoch>,
    managed: Option<Box<ManagedNode>>,
}
impl ManagedNode {
    fn height(node: &Option<Box<Self>>) -> usize {
        node.as_ref().map_or(0, |node| node.height)
    }
    fn update(&mut self) {
        self.height = 1 + Self::height(&self.left).max(Self::height(&self.right));
    }
    fn rotate_left(mut node: Box<Self>) -> Box<Self> {
        let mut head = node.right.take().expect("right-heavy node");
        node.right = head.left.take();
        node.update();
        head.left = Some(node);
        head.update();
        head
    }
    fn rotate_right(mut node: Box<Self>) -> Box<Self> {
        let mut head = node.left.take().expect("left-heavy node");
        node.left = head.right.take();
        node.update();
        head.right = Some(node);
        head.update();
        head
    }
    fn balance(mut node: Box<Self>) -> Box<Self> {
        node.update();
        let left = Self::height(&node.left);
        let right = Self::height(&node.right);
        if left > right && left - right > 1 {
            let child = node.left.as_ref().expect("left-heavy node");
            if Self::height(&child.right) > Self::height(&child.left) {
                node.left = node.left.take().map(Self::rotate_left);
            }
            Self::rotate_right(node)
        } else if right > left && right - left > 1 {
            let child = node.right.as_ref().expect("right-heavy node");
            if Self::height(&child.left) > Self::height(&child.right) {
                node.right = node.right.take().map(Self::rotate_right);
            }
            Self::rotate_left(node)
        } else {
            node
        }
    }
    fn insert(tree: Option<Box<Self>>, node: Box<Self>) -> (Option<Box<Self>>, Option<Box<Self>>) {
        let Some(mut root) = tree else {
            return (Some(node), None);
        };
        let rejected = if node.native_key < root.native_key {
            let (subtree, rejected) = Self::insert(root.left.take(), node);
            root.left = subtree;
            rejected
        } else if node.native_key > root.native_key {
            let (subtree, rejected) = Self::insert(root.right.take(), node);
            root.right = subtree;
            rejected
        } else {
            return (Some(root), Some(node));
        };
        (Some(Self::balance(root)), rejected)
    }
    fn pop_min(mut tree: Box<Self>) -> (Option<Box<Self>>, Box<Self>) {
        if let Some(left) = tree.left.take() {
            let (subtree, minimum) = Self::pop_min(left);
            tree.left = subtree;
            (Some(Self::balance(tree)), minimum)
        } else {
            let right = tree.right.take();
            tree.update();
            (right, tree)
        }
    }
    fn remove(tree: Option<Box<Self>>, key: usize) -> (Option<Box<Self>>, Option<Box<Self>>) {
        let Some(mut root) = tree else {
            return (None, None);
        };
        if key < root.native_key {
            let (subtree, removed) = Self::remove(root.left.take(), key);
            root.left = subtree;
            (Some(Self::balance(root)), removed)
        } else if key > root.native_key {
            let (subtree, removed) = Self::remove(root.right.take(), key);
            root.right = subtree;
            (Some(Self::balance(root)), removed)
        } else {
            let left = root.left.take();
            let right = root.right.take();
            root.update();
            let replacement = match right {
                None => left,
                Some(right) => {
                    let (subtree, mut successor) = Self::pop_min(right);
                    successor.left = left;
                    successor.right = subtree;
                    Some(Self::balance(successor))
                }
            };
            (replacement, Some(root))
        }
    }
}
impl ReleasedRegistry {
    fn managed(&self, key: usize) -> Option<&Arc<ManagedRelease>> {
        let mut cursor = self.managed.as_deref();
        while let Some(node) = cursor {
            if key < node.native_key {
                cursor = node.left.as_deref();
            } else if key > node.native_key {
                cursor = node.right.as_deref();
            } else {
                return node.state.as_ref();
            }
        }
        None
    }
}
/// Owned requested Arc stamp + exact Box map node + one charge-control term.
/// No table overcapacity or implicit managed HashMap growth is hidden here.
pub(super) fn requested_managed_bytes() -> Result<usize, crate::memory_admission::AllocationError> {
    use crate::memory_admission::{AllocationError, AllocationTicket};
    let control = AllocationTicket::requested_charge_control_bytes()?;
    crate::managed_layout::arc_requested_layout::<ManagedRelease>()
        .map_err(AllocationError::from)?
        .size()
        .checked_add(std::alloc::Layout::new::<ManagedNode>().size())
        .and_then(|bytes| bytes.checked_add(control))
        .ok_or(AllocationError::Overflow)
}
/// Cold enrollment, same explicit promise BEFORE state/control/node allocation.
/// Caller installs this receipt in physical C before map publication. No raw or
/// usable Primary alias escapes between construction and that installation.
pub(super) fn prepare_managed(
    context: &crate::memory_admission::AllocationContext,
) -> Result<(ManagedRegistration, ManagedNodeCredit), crate::memory_admission::AllocationError> {
    use crate::memory_admission::AllocationTicket;
    let quote = requested_managed_bytes()?;
    let payload = quote - AllocationTicket::requested_charge_control_bytes()?;
    let mut ticket = context.take(quote)?;
    let state = Arc::new(ManagedRelease {
        released: AtomicBool::new(false),
        epoch: AtomicU64::new(0),
    });
    let mut node = match crate::cold_box::try_new(ManagedNode {
        native_key: 0,
        state: Some(state.clone()),
        left: None,
        right: None,
        height: 1,
        _charge: None,
    }) {
        Ok(node) => node,
        Err(unallocated) => {
            drop(unallocated); // extra stamp Arc, no node allocation
            drop(state); // actual stamp storage freed before ticket cancellation
            return Err(crate::memory_admission::AllocationError::HostAllocation);
        }
    };
    let charge = match ticket.try_commit_with_control(payload) {
        Ok(charge) => charge,
        Err(error) => {
            drop(node);
            drop(state);
            return Err(error);
        }
    };
    node._charge = Some(charge.clone());
    Ok((
        ManagedRegistration {
            state,
            _charge: charge,
        },
        ManagedNodeCredit(Some(node)),
    ))
}
/// AFTER same record's OnceLock installation. Consumes its unique admitted node;
/// publication cannot allocate a duplicate candidate against one node receipt.
/// The credit is dropped before its registration receipt on all cold refusals.
pub(super) fn register_managed(
    buffer: &Buffer,
    owner: &ManagedRegistration,
    mut credit: ManagedNodeCredit,
) -> Result<(), &'static str> {
    if !credit
        .0
        .as_ref()
        .expect("unpublished node credit")
        .state
        .as_ref()
        .is_some_and(|state| Arc::ptr_eq(state, &owner.state))
    {
        return Err("managed diagnostic credit identity mismatch");
    }
    let mut node = credit.0.take().expect("validated node credit");
    node.native_key = buffer.raw_ptr() as usize;
    let mut candidate = Some(node);
    let result = {
        let mut map = released().lock();
        // Closed cold enrollment forbids a second live physical identity at
        // this native key. Preserve an existing identity if violated, not an
        // overwrite/final owner Drop under the map guard.
        match map.managed(buffer.raw_ptr() as usize) {
            Some(existing) if Arc::ptr_eq(existing, &owner.state) => Ok(()),
            Some(_) => Err("managed diagnostic identity already registered"),
            None => {
                let (tree, rejected) =
                    ManagedNode::insert(map.managed.take(), candidate.take().expect("candidate"));
                map.managed = tree;
                candidate = rejected;
                if candidate.is_none() {
                    Ok(())
                } else {
                    Err("managed diagnostic duplicate key")
                }
            }
        }
    };
    drop(ManagedNodeCredit(candidate)); // Box free precedes even final receipt
    result
}
/// Final baseline service/cold owner only, regardless of instrumentation OFF.
/// Moves state out before node deallocation; receipt remains in physical C.
pub(super) fn unregister_managed(buffer: &Buffer, owner: &ManagedRegistration) {
    let removed = {
        let mut map = released().lock();
        let key = buffer.raw_ptr() as usize;
        if map
            .managed(key)
            .is_some_and(|state| Arc::ptr_eq(state, &owner.state))
        {
            let (tree, removed) = ManagedNode::remove(map.managed.take(), key);
            map.managed = tree;
            removed.map(|mut node| {
                let state = node.state.take();
                let charge = node._charge.take();
                (node, state, charge)
            })
        } else {
            None
        }
    };
    if let Some((node, state, charge)) = removed {
        drop(node); // exact Box allocation freed offguard, receipt still live
        drop(state); // independent C owner survives this metadata Arc decrement
        drop(charge); // both node and its former state storage precede refund
    }
}
fn released() -> &'static parking_lot::Mutex<ReleasedRegistry> {
    static MAP: OnceLock<parking_lot::Mutex<ReleasedRegistry>> = OnceLock::new();
    MAP.get_or_init(|| parking_lot::Mutex::new(ReleasedRegistry::default()))
}

static RELEASES: AtomicU64 = AtomicU64::new(0);
static BINDS: AtomicU64 = AtomicU64::new(0);
static LATE: AtomicU64 = AtomicU64::new(0);
static LATE_WRITES: AtomicU64 = AtomicU64::new(0);

/// What the instrument saw, since the process started.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LateBindReport {
    /// Whether the instrument ran at all.
    pub armed: bool,
    /// Buffers released to a pool.
    pub releases: u64,
    /// Compute binds and blit operands checked.
    pub binds: u64,
    /// Of those, uses of a buffer released and not yet handed out again.
    pub late: u64,
    /// Of `late`, the ones that write.
    pub late_writes: u64,
}

/// The totals.
pub fn report() -> LateBindReport {
    LateBindReport {
        armed: armed(),
        releases: RELEASES.load(Ordering::Relaxed),
        binds: BINDS.load(Ordering::Relaxed),
        late: LATE.load(Ordering::Relaxed),
        late_writes: LATE_WRITES.load(Ordering::Relaxed),
    }
}

/// A pool took `buffer` back, stamped with `epoch`.
pub fn on_release(buffer: &Buffer, epoch: Epoch) {
    if !armed() {
        return;
    }
    RELEASES.fetch_add(1, Ordering::Relaxed);
    if let Some(state) = buffer.managed_late_bind() {
        state.stamp(epoch);
    } else {
        let mut map = released().lock();
        let key = buffer.raw_ptr() as usize;
        if let Some(state) = map.managed(key) {
            // Explicit Native diagnostic interoperability cannot overwrite the
            // closed allocation's identity; this is diagnostic state, not a pin.
            state.stamp(epoch);
        } else {
            map.legacy.insert(key, epoch);
        }
    }
}

/// A pool handed `buffer` to a new tenant, or destroyed it.
pub fn on_leave(buffer: &Buffer) {
    if !armed() {
        return;
    }
    if let Some(state) = buffer.managed_late_bind() {
        state.acquired();
    } else {
        // Native interoperability removes only legacy diagnostics, never the
        // closed physical registration and its independently held receipt.
        released()
            .lock()
            .legacy
            .remove(&(buffer.raw_ptr() as usize));
    }
}

/// `ptr` (a buffer's object address) is being bound, or used by a blit.
pub fn on_bind(ptr: usize, write: bool, label: &str) {
    if !armed() {
        return;
    }
    BINDS.fetch_add(1, Ordering::Relaxed);
    let stamp = {
        let map = released().lock();
        match map.managed(ptr) {
            Some(state) => state.stamp_if_released(),
            None => map.legacy.get(&ptr).copied(),
        }
    };
    let Some(stamp) = stamp else {
        return;
    };
    let n = LATE.fetch_add(1, Ordering::Relaxed) + 1;
    if write {
        LATE_WRITES.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 20 {
        eprintln!(
            "[late bind] #{n} ptr=0x{ptr:x} {} by `{label}`; released at epoch {stamp}",
            if write { "written" } else { "read" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::metal::{BufferPool, CommandBuffer, Device, GpuClock};
    use objc2_metal::MTLCommandQueue;
    use std::sync::Arc;

    /// The instrument's positive control: a bind of a buffer the pool has
    /// taken back counts, and the same buffer bound after it is handed out
    /// again does not. Without it, `late=0` on a real run could mean the
    /// lookup never matches (a key that is not the bind's), not that nothing
    /// binds late.
    #[test]
    fn a_bind_of_a_released_buffer_counts_and_one_of_a_reacquired_buffer_does_not() {
        ARMED_HERE.with(|a| a.set(true));
        let dev = Device::system_default().expect("no Metal device");
        let clock = Arc::new(GpuClock::new());
        let pool = BufferPool::with_clock(Arc::clone(&clock));
        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .unwrap();
        let handle = pool.adopt(raw, 1024);
        let stale: Buffer = (*handle).clone();
        drop(handle);

        let queue = dev.new_command_queue().expect("queue");
        // SAFETY: fresh Apple queue buffer, one importing family, host-only
        // binding fixture. Bound resources stay owned through end_encoding;
        // this test never submits GPU work or uses a native completion shortcut.
        let cb = unsafe { CommandBuffer::new(queue.commandBuffer().unwrap()) };
        let enc = cb.compute_command_encoder_no_fence().unwrap();
        let before_stale = report().late;
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(0, Some(&stale), 0) };
        let after_stale = report().late;

        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();
        let again = pool
            .acquire(1024)
            .expect("the released buffer is free again");
        let before_live = report().late;
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_output_buffer(1, Some(&again), 0) };
        let after_live = report().late;
        // Ended before asserting: an encoder dropped open aborts the process
        // and hides which assertion failed.
        enc.end_encoding().unwrap();

        assert_eq!(
            after_stale,
            before_stale + 1,
            "a bind of a released buffer was not seen"
        );
        assert_eq!(
            again.raw_ptr(),
            stale.raw_ptr(),
            "a different buffer came back"
        );
        assert_eq!(
            after_live, before_live,
            "a bind of a live buffer was called late"
        );
    }
}

#[cfg(test)]
mod managed_avl_controls {
    use super::*;
    fn node(key: usize) -> Box<ManagedNode> {
        crate::cold_box::try_new(ManagedNode {
            native_key: key,
            state: Some(Arc::new(ManagedRelease {
                released: AtomicBool::new(false),
                epoch: AtomicU64::new(0),
            })),
            left: None,
            right: None,
            height: 1,
            _charge: None,
        })
        .unwrap_or_else(|_| panic!("bounded node birth refused"))
    }
    fn audit(
        tree: &Option<Box<ManagedNode>>,
        lower: Option<usize>,
        upper: Option<usize>,
    ) -> (usize, usize) {
        let Some(root) = tree else { return (0, 0) };
        assert!(lower.is_none_or(|key| key < root.native_key));
        assert!(upper.is_none_or(|key| root.native_key < key));
        let (left, nleft) = audit(&root.left, lower, Some(root.native_key));
        let (right, nright) = audit(&root.right, Some(root.native_key), upper);
        assert!(left.abs_diff(right) <= 1, "AVL imbalance");
        assert_eq!(root.height, 1 + left.max(right), "cached height drift");
        (root.height, 1 + nleft + nright)
    }
    fn exercise(keys: &[usize], removals: &[usize]) {
        let mut tree = None;
        for (index, key) in keys.iter().enumerate() {
            let (next, rejected) = ManagedNode::insert(tree.take(), node(*key));
            assert!(rejected.is_none());
            tree = next;
            assert_eq!(audit(&tree, None, None).1, index + 1);
        }
        for (index, key) in removals.iter().enumerate() {
            let (next, removed) = ManagedNode::remove(tree.take(), *key);
            tree = next;
            let removed = removed.expect("existing key");
            assert_eq!(removed.native_key, *key);
            assert!(
                removed.left.is_none() && removed.right.is_none(),
                "removed node retained children"
            );
            assert_eq!(audit(&tree, None, None).1, keys.len() - index - 1);
            drop(removed); // off any diagnostic guard; no native allocation/use
        }
        assert!(tree.is_none());
    }
    #[test]
    fn managed_avl_ordered_and_mixed_insert_remove_preserve_all_invariants() {
        exercise(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[5, 1, 9, 2, 8, 4, 6, 3, 7]);
        exercise(&[9, 8, 7, 6, 5, 4, 3, 2, 1], &[1, 9, 5, 8, 2, 7, 3, 6, 4]);
        exercise(&[4, 8, 2, 9, 1, 6, 3, 7, 5], &[4, 6, 3, 8, 1, 9, 2, 7, 5]);
    }
    #[test]
    fn managed_avl_duplicate_returns_candidate_without_identity_overwrite() {
        let original = node(7);
        let old_state = original.state.as_ref().unwrap().clone();
        let candidate = node(7);
        let rejected_address = &*candidate as *const ManagedNode;
        let (tree, rejected) = ManagedNode::insert(Some(original), candidate);
        let rejected = rejected.expect("duplicate must return its candidate");
        assert_eq!(&*rejected as *const ManagedNode, rejected_address);
        let registry = ReleasedRegistry {
            legacy: HashMap::new(),
            managed: tree,
        };
        assert!(Arc::ptr_eq(registry.managed(7).unwrap(), &old_state));
        assert!(!Arc::ptr_eq(
            registry.managed(7).unwrap(),
            rejected.state.as_ref().unwrap()
        ));
        assert_eq!(audit(&registry.managed, None, None).1, 1);
        drop(rejected);
        assert!(Arc::ptr_eq(registry.managed(7).unwrap(), &old_state));
    }
}
