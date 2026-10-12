//! The EOS gate (lloom #735, step 4): forwards queued past an EOS write
//! nothing.
//!
//! Under the adaptive commit (`DecodeOptions::adaptive_commit`) the host
//! encodes up to two run-ahead windows of forwards before the reader learns
//! a token was EOS, and every one of them ran: on a 58-token reply, 11–14
//! forwards (≈16 ms each) of work the sequence discards — and, on a cache
//! whose rewind horizon is zero (the crown's `Shuffle` conv state, in-place
//! KV), an overrun that cannot be reconciled, so the next turn re-prefills.
//!
//! The gate makes those forwards nearly free and keeps them out of the cache,
//! at the kernel: the decode kernels that carry the time and every persistent
//! write (`gemv*`, `rmsnorm`, `bmul`, `shortconv_step`, `copy2d` — 314 of a
//! crown step's 341 dispatches) have a variant, selected by function constant
//! 777 (`Kernels::set_eos_gated`), that takes the gate's flag word at buffer
//! index 30 and returns before doing anything when it is set — except the fused
//! `Shuffle` conv step, whose output state its caller swaps into the cache
//! every step, and which therefore carries the state through unchanged
//! instead (`docs/dispatch.md` §11.5z-x). One tiny kernel
//! ([`crate::kernels::call_eos_gate`]) runs at the head of every forward,
//! after the previous step's argmax, and raises the flag if the sampled token
//! is a stop token. The flag is bound once per encoder (`Commands`) while the
//! gate is armed, so a gated dispatch costs what it cost: no per-dispatch
//! binding, no indirect argument fetch.
//!
//! Why not indirect dispatch: the first cut of this gate routed every
//! dispatch through `dispatchThreadgroupsWithIndirectBuffer` with a per-step
//! table the gate kernel zeroed. Correct and byte-identical, and +2.3 % GPU
//! time per step on both ring arms (5-rep rotation, `issue-735-eos-gate.md`)
//! — about 1 µs per indirect dispatch, more than the adaptive commit gains.
//!
//! The 27 dispatches a step that are not gated (`rope`, `sdpa`, `argmax`,
//! `cast`, `is`) read garbage past the EOS and write scratch; none writes a
//! cache. The host's position bookkeeping is rewound as "unwritten" at the
//! end of the run (`CacheRewind::rewind_unwritten`).

use crate::kernels::metal::buffer::Buffer;
use crate::kernels::metal::device::Device;
use crate::kernels::{MetalKernelError, RESOURCE_OPTIONS};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

pub use crate::device::EosGateReport;

/// The buffer index the gated kernel variants read the flag from
/// (`EOS_GATE_ARG` in `utils.metal`).
pub const EOS_GATE_FLAG_INDEX: usize = 30;
/// The function constant that selects the gated variant (`eos_gated` in
/// `utils.metal`).
pub const EOS_GATE_FUNCTION_CONSTANT: usize = 777;

/// The sticky flag and the armed state. One per [`super::Commands`]; every
/// encoder it opens while the gate is armed binds the flag at
/// [`EOS_GATE_FLAG_INDEX`].
pub struct DispatchGate {
    flag: OnceLock<Buffer>,
    armed: AtomicBool,
    steps: AtomicU64,
}

impl Default for DispatchGate {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchGate {
    /// Disarmed; nothing allocated until [`Self::begin`].
    pub fn new() -> Self {
        Self {
            flag: OnceLock::new(),
            armed: AtomicBool::new(false),
            steps: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// The flag to bind while armed; `None` when the gate is off.
    #[inline]
    pub fn armed_flag(&self) -> Option<&Buffer> {
        if self.is_armed() {
            self.flag.get()
        } else {
            None
        }
    }

    /// Allocate the flag (once), clear it, arm. From here every encoder binds
    /// the flag; the caller switches the kernels to their gated variants
    /// (`Kernels::set_eos_gated`) AFTER this, so no gated dispatch runs on an
    /// encoder without the binding.
    ///
    /// Precondition: no gated work in flight — the previous run drained. The
    /// flag is cleared by a host store into shared memory.
    pub fn begin(&self, device: &Device) -> Result<(), MetalKernelError> {
        if self.flag.get().is_none() {
            let flag = device.new_buffer(16, RESOURCE_OPTIONS)?;
            unsafe { std::ptr::write_bytes(flag.contents(), 0, 16) };
            let _ = self.flag.set(flag);
        }
        let flag = self.flag.get().expect("set above");
        unsafe { (flag.contents() as *mut u32).write_volatile(0) };
        self.steps.store(0, Ordering::Relaxed);
        self.armed.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Between a step's argmax and the next forward: encode the gate kernel
    /// on `token` (what the coming forward consumes), then a buffers-scope
    /// barrier so every later dispatch reads the flag the kernel left.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        device: &Device,
        ep: impl crate::kernels::utils::EncoderProvider,
        kernels: &crate::kernels::Kernels,
        token: &Buffer,
        token_offset: usize,
        eos_ids: &Buffer,
        eos_offset: usize,
        n_eos: usize,
    ) -> Result<(), MetalKernelError> {
        let flag = self.flag.get().ok_or_else(|| {
            MetalKernelError::FailedToCreateResource("eos gate: `step` before `begin`".to_string())
        })?;
        let encoder = ep.encoder();
        let encoder: &super::ComputeCommandEncoder = encoder.as_ref();
        // SAFETY: `token` and `eos_ids` are the caller's storages (binding
        // authority, `metal_backend` module head); the flag is this gate's own
        // buffer, which the host writes only in `begin`.
        unsafe {
            crate::kernels::call_eos_gate(
                device,
                encoder,
                kernels,
                token,
                token_offset,
                eos_ids,
                eos_offset,
                n_eos,
                flag,
            )
        }?;
        encoder.insert_memory_barrier();
        // The gate kernel bound the flag as its output at index 3; the gated
        // variants read it at index 30 — rebind there in case a `set_params!`
        // of a wide kernel touched it (none binds 31 buffers, but cheap).
        // SAFETY: as above.
        unsafe { encoder.set_input_buffer(EOS_GATE_FLAG_INDEX, Some(flag), 0) };
        self.steps.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The fused publish (`call_argmax_publish`) did this step's EOS test:
    /// count it as a gate step.
    pub fn note_step(&self) {
        self.steps.fetch_add(1, Ordering::Relaxed);
    }

    /// Disarm and report. Precondition as `begin`'s: the run drained, so the
    /// flag read here is the last gate kernel's.
    pub fn end(&self) -> EosGateReport {
        self.armed.store(false, Ordering::Relaxed);
        let fired = self
            .flag
            .get()
            .map(|b| unsafe { (b.contents() as *const u32).read_volatile() } != 0)
            .unwrap_or(false);
        EosGateReport {
            fired,
            steps: self.steps.load(Ordering::Relaxed),
        }
    }
}
