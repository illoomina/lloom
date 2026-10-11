//! Owner-directed bridge between numeric domain intents and machine holdings.
//!
//! Internal and not wired to loaders/backings yet. One non-clone owner is moved
//! into one domain; its promises and physical charges retain that domain. Never
//! grow the first member of a weak aggregate list on another model's behalf.

use crate::machine::{Admitted, Verdict};

/// Explicit policy outcome; absence of a registered holding is not "covered".
#[derive(Clone, Debug)]
pub(crate) enum MachineAdmission {
    Registered {
        reserved: u64,
        forced: bool,
        growth: Option<Verdict>,
    },
    Off,
    ForcedUnregistered {
        initial: Verdict,
    },
}

/// Intentionally not Clone. Domain creation consumes it, preventing accidental
/// assignment of one holding's target to two independently counted domains.
/// Last destruction is host-only: Holding::drop performs registry operations.
pub(crate) struct MachineOwner {
    admitted: Admitted,
}

impl MachineOwner {
    pub fn new(admitted: Admitted) -> crate::Result<Self> {
        if admitted.holding.is_none() && admitted.verdict.as_ref().is_some_and(|v| !v.forced) {
            crate::bail!("machine owner has no holding and no explicit force/off override");
        }
        Ok(Self { admitted })
    }

    /// Called AFTER publishing numeric pending demand and releasing all account
    /// locks, BEFORE returning an allocatable promise. grow_to serializes and
    /// rechecks under the registry lock; older targets never shrink high-water.
    pub fn ensure_reserved(&self, target: u64) -> crate::Result<MachineAdmission> {
        if let Some(holding) = &self.admitted.holding {
            let growth = holding.grow_to(target)?;
            return Ok(MachineAdmission::Registered {
                reserved: holding.reserved(),
                forced: self.admitted.verdict.as_ref().is_some_and(|v| v.forced),
                growth,
            });
        }
        match &self.admitted.verdict {
            None => Ok(MachineAdmission::Off),
            Some(initial) if initial.forced => Ok(MachineAdmission::ForcedUnregistered {
                initial: initial.clone(),
            }),
            Some(_) => crate::bail!("unregistered machine owner without an explicit override"),
        }
    }
}
