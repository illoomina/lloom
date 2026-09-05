//! GPU-resident state for the arena's bump allocator (`DESIGN.md` §9.2d, #70).
//!
//! # What this is, in one paragraph
//!
//! #69 computes every arena offset on the CPU: `Arena::acquire` maps an
//! allocation ordinal to a slot through a `Vec` lookup, on the host, once per
//! activation. This module computes the same offsets with a kernel instead --
//! one `atomic_uint` cursor in device memory, bumped in ordinal order, reset
//! once per decode step. The CPU path stays and stays selectable
//! ([`ArenaOffsets`]), the same discipline `ParamStyle` follows for binding
//! styles (§11.3b) and [`ArenaLayout`](super::ArenaLayout) for layouts: keeping
//! both live is what makes the A/B free and the parity check a comparison
//! between two paths that exist.
//!
//! # Why the offsets must be *equal*, not merely valid
//!
//! The acceptance bar for #70 is bit-identical activations, and the mechanism
//! that delivers it is stronger than a numerical tolerance: the GPU-computed
//! offset table must be **element-wise equal** to the CPU-computed one. If the
//! two agree exactly then every kernel binds the same bytes it bound before, so
//! the activations cannot differ -- bit-identity follows by construction rather
//! than by measurement. That is why [`ArenaCursor::offsets`] exists and why the
//! parity test compares tables rather than model outputs alone.
//!
//! # What this deliberately does not do
//!
//! **It does not choose a buffer.** Everything is `arena_base + offset` against
//! one allocation, so residency stays a CPU-side fact established once
//! (§9.2d case 2). The GPU never selects an identity, and the design is shaped
//! so the question cannot arise.
//!
//! **It is not a free list.** §9.3: MSL `device` atomics accept only
//! `memory_order_relaxed`, so a lock-free free list has no standard correctness
//! argument available (§9.2d case 3). A bump allocator with a per-step reset is
//! sufficient because live ranges within a step are planned offline (#68) and
//! everything resets between steps.
//!
//! **It does not remove the CPU from binding.** `setBuffer_offset_atIndex` is a
//! host call, so a GPU-computed offset reaches a *classical* dispatch only by
//! being read back. That is not a regression to be apologised for, it is the
//! shape of the boundary: the offset a kernel computes can be consumed without
//! a round-trip only inside an ICB command written by an encoding kernel
//! (§11.3c, verified), and the ICB executor is explicitly out of scope for this
//! issue. What lands here is the allocator itself, GPU-resident and proven
//! equal to the CPU's, which is the mechanism a GPU-driven token loop needs
//! (§11.5) and cannot be built without.

use super::{Buffer, Device};
use crate::kernels::MetalKernelError;
use objc2_metal::MTLResourceOptions;

/// The offset value meaning "this ordinal is not the arena's".
///
/// Must equal the sentinel `arena_alloc.metal` writes. Checked across the
/// language boundary by `arena_alloc_reports_alignment` rather than asserted
/// twice -- §11.3d's argument: a constant asserted on each side proves only
/// that each side agrees with itself.
pub const ARENA_DECLINED: u32 = u32::MAX;

/// Where an arena offset is computed.
///
/// Both are compiled and either is selectable at run time. `Cpu` is the default
/// and is #69's path exactly, so an unconfigured process is byte-for-byte what
/// shipped -- the same property `HazardKey::Pointer` and `ArenaLayout::Packed`
/// preserve for their axes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArenaOffsets {
    /// #69's path: the offset comes from `StepPlan::by_ordinal`, on the host.
    #[default]
    Cpu,
    /// Issue #70's path: the offset comes from a kernel bump-allocating over an
    /// `atomic_uint` in device memory.
    ///
    /// Equal to `Cpu`'s output by construction and by test -- see
    /// [`ArenaCursor::verify_against`]. If the two ever disagree the arena
    /// declines to install, rather than binding an offset nobody planned.
    Gpu,
}

impl ArenaOffsets {
    pub fn is_gpu(self) -> bool {
        self == ArenaOffsets::Gpu
    }

    /// The wire form for [`Arena`](super::Arena)'s `AtomicU8` (issue #562).
    ///
    /// The arena records this field with an atomic rather than under a mutex,
    /// because it is a single value with no invariant tying it to anything
    /// else. These two functions are the whole of what that costs, and both are
    /// a `match` so a third variant is a **compile error** rather than a value
    /// that silently round-trips to `Cpu` -- §7.1a-i's *presence is not
    /// correctness*, where a two-armed conditional over a three-armed axis
    /// rendered `--attn flash` as `AttnImpl=Generic` on every `RESULT` line
    /// since #116, and a wrong arm compares equal to every run of that arm
    /// where an absent one compares equal to nothing.
    pub(crate) fn to_bits(self) -> u8 {
        match self {
            ArenaOffsets::Cpu => 0,
            ArenaOffsets::Gpu => 1,
        }
    }

    /// Inverse of [`to_bits`](Self::to_bits).
    ///
    /// Panics on an unknown value rather than defaulting to `Cpu`: the only
    /// writer is `to_bits`, so an unknown bit pattern means memory corruption
    /// rather than an input to be tolerated. Defaulting would report the CPU
    /// path for a run that took the GPU one, which is the engagement failure
    /// §2.4 exists to prevent -- #69's determinism gate was vacuous precisely
    /// because a flag was believed rather than checked against a quantity it
    /// should have moved.
    pub(crate) fn from_bits(bits: u8) -> Self {
        match bits {
            0 => ArenaOffsets::Cpu,
            1 => ArenaOffsets::Gpu,
            other => unreachable!("ArenaOffsets has no bit pattern {other}"),
        }
    }
}

/// Storage for the bump allocator: the cursor, the request sizes, the results.
///
/// # Storage modes, and why they differ
///
/// The arena itself is `StorageModePrivate` -- it holds activations, which no
/// CPU ever reads, and Private is the faster mapping. These three buffers are
/// `StorageModeShared` instead, because they are **allocator state rather than
/// activation data**: the sizes are written by the host each step, and the
/// offsets are read back by the host to bind through. A Private cursor would
/// have `contents() == null` and the parity check could not exist at all.
///
/// That readback is the honest cost of this path on a *classical* dispatch and
/// is stated rather than hidden -- see the module docs. It is not what a
/// GPU-driven loop would pay, because there the offsets are consumed by an
/// encoding kernel and never cross to the host.
pub struct ArenaCursor {
    /// The allocator's only mutable state: one `atomic_uint`, 4 bytes.
    cursor: Buffer,
    /// One `u32` byte-size per allocation ordinal. 0 marks an ordinal the arena
    /// does not serve, which keeps its place so that excluding it does not
    /// renumber the ordinals after it (§9.1, and `StepPlan::by_ordinal`).
    sizes: Buffer,
    /// One `u32` offset per ordinal, or [`ARENA_DECLINED`].
    offsets: Buffer,
    /// Ordinals the plan covers.
    n: u32,
    /// The arena's byte length. An allocation that would run past it is
    /// declined rather than wrapped: wrapping would hand out an offset
    /// addressing another slot's bytes, which under
    /// `HazardTrackingModeUntracked` is silent corruption (§3.5).
    capacity: u32,
}

/// Shared storage, so the host can write the request sizes and read the results.
const CURSOR_OPTIONS: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

impl ArenaCursor {
    /// Allocate cursor state for a plan of `sizes`, over an arena of
    /// `capacity` bytes.
    ///
    /// `sizes[i]` is ordinal `i`'s byte request, 0 where the arena declines it.
    pub fn new(device: &Device, sizes: &[u32], capacity: usize) -> Result<Self, MetalKernelError> {
        let n = sizes.len();
        let cursor = device.new_buffer(4, CURSOR_OPTIONS)?;
        // `new_buffer_with_data` rejects a null pointer, and an empty plan is a
        // real case (a recording that saw nothing), so the buffers are sized to
        // at least one element rather than zero.
        let size_bytes = (n.max(1)) * 4;
        let sizes_buf = device.new_buffer(size_bytes, CURSOR_OPTIONS)?;
        let offsets = device.new_buffer(size_bytes, CURSOR_OPTIONS)?;

        let this = Self {
            cursor,
            sizes: sizes_buf,
            offsets,
            n: n as u32,
            capacity: u32::try_from(capacity).map_err(|_| {
                MetalKernelError::InvalidInput(format!(
                    "arena is {capacity} B; the cursor is a uint and cannot address it"
                ))
            })?,
        };
        this.write_sizes(sizes);
        this.store_cursor(0);
        Ok(this)
    }

    /// Write the request sizes the allocator will walk.
    fn write_sizes(&self, sizes: &[u32]) {
        let dst = self.sizes.contents() as *mut u32;
        debug_assert!(!dst.is_null(), "cursor sizes buffer has no CPU mapping");
        for (i, &s) in sizes.iter().enumerate() {
            // SAFETY: `sizes` buffer holds `max(n, 1)` u32 and `i < n`.
            unsafe { dst.add(i).write(s) };
        }
    }

    pub fn cursor_buffer(&self) -> &Buffer {
        &self.cursor
    }

    pub fn sizes_buffer(&self) -> &Buffer {
        &self.sizes
    }

    pub fn offsets_buffer(&self) -> &Buffer {
        &self.offsets
    }

    pub fn len(&self) -> u32 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The cursor's current value, as the host sees it.
    ///
    /// Only meaningful after the command buffer carrying the allocator has
    /// completed. Reading it mid-flight is a race the host cannot detect, which
    /// is why every caller in this crate reads it after
    /// `wait_until_completed`.
    pub fn cursor_value(&self) -> u32 {
        let p = self.cursor.contents() as *const u32;
        if p.is_null() {
            return 0;
        }
        // SAFETY: shared storage, 4 bytes, and the caller has synchronised.
        unsafe { p.read() }
    }

    /// Set the cursor from the host.
    ///
    /// Used to initialise it and by tests; the decode path resets it with
    /// `arena_reset_cursor` instead, because a host store cannot be ordered
    /// against GPU work still reading the arena (see `call_arena_reset`).
    pub fn store_cursor(&self, v: u32) {
        let p = self.cursor.contents() as *mut u32;
        if p.is_null() {
            return;
        }
        // SAFETY: shared storage, 4 bytes, host-side initialisation only.
        unsafe { p.write(v) };
    }

    /// The offsets the allocator produced, one per ordinal.
    ///
    /// Only meaningful after the command buffer has completed, as
    /// [`Self::cursor_value`].
    pub fn offsets(&self) -> Vec<u32> {
        let p = self.offsets.contents() as *const u32;
        if p.is_null() {
            return Vec::new();
        }
        // SAFETY: shared storage holding `max(n, 1)` u32, and the caller has
        // synchronised.
        unsafe { std::slice::from_raw_parts(p, self.n as usize) }.to_vec()
    }

    /// Compare the GPU's offsets against what the CPU plan assigned.
    ///
    /// `expected[i]` is the offset ordinal `i` should receive, or `None` where
    /// the arena declines it. Returns the first disagreement, so a failure
    /// names the ordinal rather than only reporting that one exists.
    ///
    /// **This is the whole correctness argument for #70**, and it is an
    /// equality rather than a tolerance. If every ordinal resolves to the byte
    /// the CPU path would have chosen, then every kernel binds what it bound
    /// before and the activations are bit-identical by construction. A
    /// comparison of model outputs alone could pass by luck on a step where a
    /// mis-assigned slot happened not to be read; this cannot.
    pub fn verify_against(&self, expected: &[Option<usize>]) -> Result<(), String> {
        let got = self.offsets();
        if got.len() != expected.len() {
            return Err(format!(
                "GPU produced {} offsets, the plan has {} ordinals",
                got.len(),
                expected.len()
            ));
        }
        for (i, (&g, e)) in got.iter().zip(expected.iter()).enumerate() {
            match e {
                Some(want) => {
                    if g as usize != *want {
                        return Err(format!("ordinal {i}: GPU offset {g}, plan offset {want}"));
                    }
                }
                None => {
                    if g != ARENA_DECLINED {
                        return Err(format!("ordinal {i}: plan declines it, GPU offset {g}"));
                    }
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for ArenaCursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArenaCursor")
            .field("ordinals", &self.n)
            .field("capacity", &self.capacity)
            .field("cursor", &self.cursor_value())
            .finish()
    }
}
