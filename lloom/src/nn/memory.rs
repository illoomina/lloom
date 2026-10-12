//! The process's memory, read from outside the allocator: the device's
//! allocated bytes, the pool's live / free / pending, the process's
//! `phys_footprint` and Metal's `recommendedMaxWorkingSetSize` — the gauge
//! `DESIGN.md` §9.5m puts on a serving binary before a cell is trusted
//! (`/health`, `--mem-sample`), here so every harness can put it on its
//! lines (`qwen4exp-chat`'s `MEM`, issue #749 P18).
//!
//! §6.3e: an accounting instrument cannot find memory its own accounting
//! does not name; `device_allocated` and `phys_footprint` are the two
//! readings from outside it, and the ones that found the retention.

use lloom_tensor::machine::{Admitted, Gate, Holding, Peak};
use lloom_tensor::memory_admission::{
    process_account, AllocationContext, ModelDomain, PassReservation,
};
use lloom_tensor::{Device, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// One reading of the process's memory: the device's allocated bytes, the
/// pool's live / free / pending (CPU-released, awaiting the GPU — §9.5o's
/// class) bytes, and the process's `phys_footprint`, against Metal's
/// `recommendedMaxWorkingSetSize`. Every field is 0 off Metal.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemSnapshot {
    pub device_allocated: u64,
    pub pool_live: u64,
    pub pool_free: u64,
    pub pool_pending: u64,
    pub phys_footprint: u64,
    pub working_set: u64,
    /// The ledger's quota and committed bytes (`ledger.rs`): what the
    /// process holds under reservations, beside what the device reports.
    pub ledger_quota: u64,
    pub ledger_committed: u64,
}

impl MemSnapshot {
    pub fn read(device: &Device) -> Self {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            let (shared, private) = m.pool_occupancy();
            return Self {
                device_allocated: m.device().current_allocated_size() as u64,
                pool_live: (shared.live_bytes + private.live_bytes) as u64,
                pool_free: (shared.free_bytes + private.free_bytes) as u64,
                pool_pending: (shared.pending_bytes + private.pending_bytes) as u64,
                phys_footprint: proc_footprint().map(|(p, _)| p).unwrap_or(0),
                working_set: m.device().recommended_max_working_set_size() as u64,
                ledger_quota: m.ledger().quota() as u64,
                ledger_committed: m.ledger().committed() as u64,
            };
        }
        let _ = device;
        Self::default()
    }

    pub fn json(&self) -> Value {
        json!({
            "device_allocated_bytes": self.device_allocated,
            "pool_live_bytes": self.pool_live,
            "pool_free_bytes": self.pool_free,
            "pool_pending_bytes": self.pool_pending,
            "phys_footprint_bytes": self.phys_footprint,
            "recommended_max_working_set_bytes": self.working_set,
            "ledger_quota_bytes": self.ledger_quota,
            "ledger_committed_bytes": self.ledger_committed,
        })
    }
}

/// What a cell's memory gauge reports: the readings before and after, and —
/// under `--mem-sample` — the peaks over the cell.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemReport {
    pub before: MemSnapshot,
    pub after: MemSnapshot,
    /// Peaks over the cell; `None` when the sampler was off.
    pub peak: Option<MemSnapshot>,
    pub samples: u64,
}

impl MemReport {
    pub fn json(&self) -> Value {
        let mut v = json!({
            "before": self.before.json(),
            "after": self.after.json(),
            "sampled": self.peak.is_some(),
            "samples": self.samples,
        });
        if let Some(p) = &self.peak {
            v["peak"] = p.json();
        }
        v
    }

    /// One line for the log: the peak (or the after) device bytes and the
    /// footprint, and what the cell left behind.
    pub fn line(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        match &self.peak {
            Some(p) => format!(
                "mem: device {:.2} -> peak {:.2} -> {:.2} GB, pending peak {:.2} GB, phys peak {:.2} GB ({} samples)",
                gb(self.before.device_allocated),
                gb(p.device_allocated),
                gb(self.after.device_allocated),
                gb(p.pool_pending),
                gb(p.phys_footprint),
                self.samples
            ),
            None => format!(
                "mem: device {:.2} -> {:.2} GB, phys {:.2} -> {:.2} GB, ledger committed {:.2} -> {:.2} GB of quota {:.2} GB",
                gb(self.before.device_allocated),
                gb(self.after.device_allocated),
                gb(self.before.phys_footprint),
                gb(self.after.phys_footprint),
                gb(self.before.ledger_committed),
                gb(self.after.ledger_committed),
                gb(self.after.ledger_quota)
            ),
        }
    }
}

/// The ledger's line for a report: the quota, the committed bytes and every
/// tenant's claim and holding, beside the device's own figure. `None` off
/// Metal.
pub fn ledger_line(device: &Device) -> Option<String> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        let gb = |b: usize| b as f64 / 1e9;
        return Some(format!(
            "[ledger: {} device_allocated={:.2} GB]",
            m.ledger().snapshot().line(),
            gb(m.device().current_allocated_size())
        ));
    }
    let _ = device;
    None
}

/// The 10 ms sampler's stop flag and thread, joined for `(peak, samples)`.
type PeakSampler = (Arc<AtomicBool>, std::thread::JoinHandle<(MemSnapshot, u64)>);

/// The gauge around one cell: a reading before, a reading after, and under
/// `--mem-sample` a 10 ms thread holding the peaks in between.
pub struct MemGauge {
    before: MemSnapshot,
    sampler: Option<PeakSampler>,
}

impl MemGauge {
    pub fn start(device: &Device, sample: bool) -> Self {
        let before = MemSnapshot::read(device);
        let sampler = if sample && !device.is_cpu() {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let device = device.clone();
            let handle = std::thread::spawn(move || {
                let mut peak = MemSnapshot::default();
                let mut samples = 0u64;
                loop {
                    let s = MemSnapshot::read(&device);
                    peak.device_allocated = peak.device_allocated.max(s.device_allocated);
                    peak.pool_live = peak.pool_live.max(s.pool_live);
                    peak.pool_free = peak.pool_free.max(s.pool_free);
                    peak.pool_pending = peak.pool_pending.max(s.pool_pending);
                    peak.phys_footprint = peak.phys_footprint.max(s.phys_footprint);
                    peak.working_set = s.working_set;
                    peak.ledger_quota = s.ledger_quota;
                    peak.ledger_committed = peak.ledger_committed.max(s.ledger_committed);
                    samples += 1;
                    if flag.load(Ordering::Acquire) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                (peak, samples)
            });
            Some((stop, handle))
        } else {
            None
        };
        Self { before, sampler }
    }

    pub fn finish(self, device: &Device) -> MemReport {
        let (peak, samples) = match self.sampler {
            Some((stop, handle)) => {
                stop.store(true, Ordering::Release);
                match handle.join() {
                    Ok((p, n)) => (Some(p), n),
                    Err(_) => (None, 0),
                }
            }
            None => (None, 0),
        };
        MemReport {
            before: self.before,
            after: MemSnapshot::read(device),
            peak,
            samples,
        }
    }
}

/// The process's `phys_footprint` and resident size, from `proc_pid_rusage`
/// (`RUSAGE_INFO_V6`; the two words at their struct offsets).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn proc_footprint() -> Option<(u64, u64)> {
    const RUSAGE_INFO_V6: i32 = 6;
    const W_RESIDENT: usize = 8;
    const W_PHYS: usize = 9;
    const WORDS: usize = 64;
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    let mut buf = [0u64; WORDS];
    // SAFETY: `proc_pid_rusage` writes at most `sizeof(rusage_info_v6)` bytes
    // (~472) into the buffer, and `WORDS * 8 = 512` is larger, so the write is
    // in bounds; the buffer outlives the call.
    let rc = unsafe {
        proc_pid_rusage(
            std::process::id() as i32,
            RUSAGE_INFO_V6,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    if rc != 0 {
        return None;
    }
    Some((buf[W_PHYS], buf[W_RESIDENT]))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[allow(dead_code)]
fn proc_footprint() -> Option<(u64, u64)> {
    None
}

/// The budget a caller declares for admission (`DESIGN.md` §9.5d):
/// what the device must hold beside the weights, predicted once before any
/// allocation. Carried on a family's config as an `Option` so a caller that
/// declares none keeps the unadmitted path.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    /// Total weight bytes, from the checkpoint header. A number rather
    /// than a reading because the loader's backing store is not visible
    /// here, and a text-only load legitimately drops a vision tower; the
    /// §9.5m reconciliation is the check that it is right, bounded by
    /// `weight_tolerance`.
    pub weight_bytes: usize,
    /// Concurrent sequences.
    pub batch: usize,
    /// Fraction of `recommendedMaxWorkingSetSize` this process may claim
    /// (§9.5k's table value, a settable default rather than a measurement).
    pub fraction: f64,
    /// How far `weight_bytes` may diverge from what was allocated before
    /// the §9.5m reconciliation reports it: a bound on the unaccounted,
    /// cleared ~6× by the measured non-weight residual and exceeded ~3× by
    /// the vision tower it exists to catch.
    pub weight_tolerance: usize,
    /// The longest single forward pass this process will run: `1` for
    /// decode, the prompt length for a single-shot prefill, the chunk size
    /// for a chunked one (§13.2b). Admission runs before any pass, so the
    /// figure cannot be inferred; a caller that prefills without stating it
    /// gets the unchecked verdict.
    pub max_seq_len: usize,
    /// The rows the activation class is computed for: `None` = `batch`
    /// (every row its own forward); `Some(1)` for the batched serving loop,
    /// whose step is one ragged forward over every row's tokens. KV and
    /// conv state are per row either way.
    pub activation_rows: Option<usize>,
}

impl MemoryBudget {
    /// §9.5k's fraction. A copy of `admission::DEFAULT_BUDGET_FRACTION`,
    /// which is Metal-only where this type exists on every backend; the two
    /// are asserted equal under the `metal` feature.
    pub const DEFAULT_FRACTION: f64 = 0.65;

    /// §9.5m's default tolerance for the weight reconciliation. A copy of
    /// `admission::WEIGHT_RECONCILE_TOLERANCE` for `DEFAULT_FRACTION`'s
    /// reason, asserted equal the same way.
    pub const DEFAULT_WEIGHT_TOLERANCE: usize = 256 * 1_000_000;

    /// A B=1 decode budget at §9.5k's fraction for a model whose weights are
    /// `weight_bytes`.
    pub fn new(weight_bytes: usize) -> Self {
        Self {
            weight_bytes,
            batch: 1,
            fraction: Self::DEFAULT_FRACTION,
            weight_tolerance: Self::DEFAULT_WEIGHT_TOLERANCE,
            max_seq_len: 1,
            activation_rows: None,
        }
    }

    /// The same budget, told the longest forward pass it will be asked for:
    /// a single-shot prefill's length, or a chunked prefill's chunk size.
    pub fn with_prefill(mut self, seq_len: usize) -> Self {
        self.max_seq_len = seq_len;
        self
    }
}

/// What a family's loader admits on the machine before its first byte
/// (`lloom_tensor::machine`, `memory.md` §9.10): the gate, the label the
/// reservation is listed under, and the run's shape the loader cannot read
/// from the checkpoint (the rows, the KV each row is sized for, the prefill
/// tile).
#[derive(Clone, Debug)]
pub struct MachineLoad {
    pub gate: Gate,
    pub label: String,
    pub rows: usize,
    pub kv_tokens: usize,
    /// The longest forward the run takes; 0 = the family's own chunk.
    pub prefill_tile: usize,
}

/// One admitted load's domain and whole-load promise. No producer is implicitly
/// enrolled: factories/workers must consume `allocations()` explicitly. Issued
/// tickets outlive finish; untouched demand cancels when the load closes.
#[must_use = "dropping the load cancels unissued demand"]
pub struct LoadContext {
    pass: PassReservation,
    domain: ModelDomain,
}

impl LoadContext {
    fn from_admitted(admitted: Admitted, bytes: usize, quota: Option<usize>) -> Result<Self> {
        let domain = process_account().domain(quota, admitted)?;
        let pass = domain.reserve(bytes)?;
        Ok(Self { pass, domain })
    }

    pub fn domain(&self) -> &ModelDomain {
        &self.domain
    }
    pub fn allocations(&self) -> AllocationContext {
        self.pass.context()
    }

    /// Close future issuance and cancel unissued demand. Already-issued tickets
    /// and physical charges retain the same domain independently.
    pub fn finish(self) -> ModelDomain {
        let Self { pass, domain } = self;
        drop(pass);
        domain
    }
}

impl MachineLoad {
    /// Shared gate/log path. Add the exact pool term ONCE and retain its checked
    /// total for the load promise; neither public route reconstructs that sum.
    /// The gate this load is admitted through on `device`: the reserve's
    /// basis is the device's working set where it is a Metal device.
    pub fn gate_for(&self, device: &Device) -> Gate {
        self.gate.with_device_working_set(working_set_bytes(device))
    }

    fn admit_unattached(
        &self,
        gate: &Gate,
        peak: Peak,
        pool_bound: usize,
    ) -> Result<(Admitted, usize)> {
        let pool_bound = u64::try_from(pool_bound).map_err(lloom_tensor::Error::wrap)?;
        let peak = peak.with("pool free lists", pool_bound);
        let total = peak.checked_total().ok_or_else(|| {
            lloom_tensor::Error::Msg(format!(
                "machine load peak arithmetic overflow: {}; reduce requested sizes",
                peak.describe()
            ))
        })?;
        let bytes = usize::try_from(total).map_err(lloom_tensor::Error::wrap)?;
        // Even gate-off must not make an unrepresentable allocation possible.
        let t = std::time::Instant::now();
        let admitted = gate.admit(&self.label, &peak)?;
        eprintln!(
            "[machine: peak {} for {}; gate {:.0} us]",
            peak.describe(),
            self.label,
            t.elapsed().as_secs_f64() * 1e6
        );
        eprintln!("{}", admitted.line());
        if let (Some(_), Some(v)) = (&admitted.holding, &admitted.verdict) {
            if v.forced {
                eprintln!("[machine: {}]", v.describe().replace('\n', "\n  "));
            }
        }
        Ok((admitted, bytes))
    }

    /// Legacy route: preserves device-ledger attachment and visible policy
    /// logging. Checked augmented geometry precedes the gate, including off.
    pub fn admit(&self, device: &Device, peak: Peak) -> Result<Admitted> {
        let (admitted, _) =
            self.admit_unattached(&self.gate_for(device), peak, pool_bound_bytes(device))?;
        if let (Some(h), Some(_)) = (&admitted.holding, &admitted.verdict) {
            attach(device, h);
        }
        Ok(admitted)
    }

    /// Additive admitted ownership route: one domain, one load promise, no
    /// legacy weak-list attachment. Configure process policy before first use.
    /// Native/family/worker allocation paths are NOT implicitly migrated.
    pub fn admit_context(
        &self,
        device: &Device,
        peak: Peak,
        quota: Option<usize>,
    ) -> Result<LoadContext> {
        let (admitted, bytes) =
            self.admit_unattached(&self.gate_for(device), peak, pool_bound_bytes(device))?;
        LoadContext::from_admitted(admitted, bytes, quota)
    }
}

/// Count `holding` on `device`'s ledger, so a pass past it is admitted on
/// the machine (`MetalDevice::admit_bytes`). Nothing to do off Metal.
pub fn attach(device: &Device, holding: &std::sync::Arc<Holding>) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.ledger().attach_machine(holding);
    }
    let _ = (device, holding);
}

/// Metal's `recommendedMaxWorkingSetSize` for `device`; `None` off Metal.
pub fn working_set_bytes(device: &Device) -> Option<u64> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Some(m.device().recommended_max_working_set_size() as u64);
    }
    let _ = device;
    None
}

/// The pools' free lists at their caps: bytes the process may keep that no
/// tenant claims. 0 off Metal.
pub fn pool_bound_bytes(device: &Device) -> usize {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.free_budgets();
    }
    let _ = device;
    0
}

#[cfg(test)]
mod load_context_tests {
    use super::*;
    use lloom_tensor::machine::GateConfig;
    use lloom_tensor::memory_admission::{AccountSnapshot, AllocationError, MachineStatus};

    fn off() -> MachineLoad {
        MachineLoad {
            gate: Gate::new(GateConfig {
                off: true,
                ..Default::default()
            }),
            label: "numeric-load-context".into(),
            rows: 1,
            kv_tokens: 1,
            prefill_tile: 1,
        }
    }

    #[test]
    fn numeric_gate_off_still_refuses_overflow_before_admission() {
        let peak = Peak::new().with("a", u64::MAX).with("b", 1);
        assert_eq!(peak.checked_total(), None);
        assert!(off()
            .admit_context(&Device::Cpu, peak.clone(), None)
            .is_err());
        assert!(off().admit(&Device::Cpu, peak).is_err());
        assert!(off()
            .admit_unattached(&off().gate, Peak::new().with("a", u64::MAX), 1)
            .is_err());
        #[cfg(target_pointer_width = "32")]
        assert!(off()
            .admit_context(&Device::Cpu, Peak::new().with("a", u64::MAX), None)
            .is_err());
    }

    /// A load whose promise its quota cannot hold is refused as the typed
    /// allocation error, not as text (`memory_admission::AllocationError`).
    #[test]
    fn a_load_past_its_quota_is_refused_typed() {
        let refused = off()
            .admit_context(&Device::Cpu, Peak::new().with("weights", 64), Some(16))
            .err()
            .expect("a 64-byte load in a 16-byte quota is refused");
        assert!(
            matches!(
                refused,
                lloom_tensor::Error::Allocation(AllocationError::Refused {
                    requested: 64,
                    available: 16,
                    domain: lloom_tensor::memory_admission::RefusalDomain::Model,
                })
            ),
            "{refused:?}"
        );
    }

    #[test]
    fn numeric_pool_inclusive_promise_finish_and_issued_ticket() {
        // The actual helper used by both public routes; injected pool=5 keeps
        // this CPU-only while checking that the returned total includes it once.
        let (admitted, bytes) = off()
            .admit_unattached(&off().gate, Peak::new().with("weights", 7), 5)
            .unwrap();
        assert_eq!(bytes, 12);
        let load = LoadContext::from_admitted(admitted, bytes, Some(16)).unwrap();
        assert!(matches!(load.domain().machine_status(), MachineStatus::Off));
        assert_eq!(
            load.domain().snapshot(),
            AccountSnapshot {
                pending: 12,
                committed: 0
            }
        );
        let context = load.allocations();
        let ticket = context.take(8).unwrap();
        let domain = load.finish();
        assert!(matches!(context.take(1), Err(AllocationError::Cancelled)));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 8,
                committed: 0
            }
        );
        let charge = ticket.commit(3).unwrap();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 3
            }
        );
        drop(charge);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
        let load = off()
            .admit_context(&Device::Cpu, Peak::new().with("weights", 4), Some(4))
            .unwrap();
        let context = load.allocations();
        let domain = load.domain().clone();
        drop(load);
        assert!(matches!(context.take(1), Err(AllocationError::Cancelled)));
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
}
