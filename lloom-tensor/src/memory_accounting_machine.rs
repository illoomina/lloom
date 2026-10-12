//! Owner-directed bridge between numeric domain intents and machine holdings.
//!
//! Internal and not wired to loaders/backings yet. One non-clone owner is moved
//! into one domain; its promises and physical charges retain that domain. Never
//! grow the first member of a weak aggregate list on another model's behalf.

use crate::machine::{Admitted, GrowRefusal, HoldingDomainBinding, Verdict};

/// Why a machine reservation could not widen; `Copy` so a refusal carries no
/// text. The gate's arithmetic stays on the holding (`Holding::last_refusal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MachineRefusal {
    /// The room could not hold `need` more bytes.
    Room { need: u64, room: i128 },
    /// The registry or the host's memory could not be read.
    Unavailable,
    /// No holding without an explicit override, or one bound to another domain.
    Binding,
}

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

/// Intentionally not Clone, and holds an exclusive binding on the Holding.
/// Non-Clone alone is insufficient: callers can reconstruct public Admitted
/// values sharing one Arc<Holding>. The binding rejects that aliasing route.
/// Last destruction is host-only: Holding::drop performs registry operations.
pub(crate) struct MachineOwner {
    admitted: Admitted,
    _binding: Option<HoldingDomainBinding>,
}

impl MachineOwner {
    pub fn new(admitted: Admitted) -> Result<Self, MachineRefusal> {
        if admitted.holding.is_none() && admitted.verdict.as_ref().is_some_and(|v| !v.forced) {
            return Err(MachineRefusal::Binding);
        }
        let binding = admitted
            .holding
            .as_ref()
            .map(|h| h.try_bind_domain())
            .transpose()
            .map_err(|_| MachineRefusal::Binding)?;
        Ok(Self {
            admitted,
            _binding: binding,
        })
    }

    pub fn status(&self) -> MachineAdmission {
        if let Some(holding) = &self.admitted.holding {
            MachineAdmission::Registered {
                reserved: holding.reserved(),
                forced: self.admitted.verdict.as_ref().is_some_and(|v| v.forced),
                growth: None,
            }
        } else if let Some(initial) = &self.admitted.verdict {
            // Constructor rejects an unregistered, non-forced verdict.
            MachineAdmission::ForcedUnregistered {
                initial: initial.clone(),
            }
        } else {
            MachineAdmission::Off
        }
    }

    /// Called AFTER publishing numeric pending demand and releasing all account
    /// locks, BEFORE returning an allocatable promise. grow_to serializes and
    /// rechecks under the registry lock; older targets never shrink high-water.
    pub fn ensure_reserved(&self, target: u64) -> Result<MachineAdmission, MachineRefusal> {
        if let Some(holding) = &self.admitted.holding {
            let growth = holding.try_grow_to(target).map_err(|e| match e {
                GrowRefusal::Room(v) => MachineRefusal::Room {
                    need: v.need,
                    room: v.room(),
                },
                GrowRefusal::Unavailable(_) => MachineRefusal::Unavailable,
            })?;
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
            Some(_) => Err(MachineRefusal::Binding),
        }
    }
}
