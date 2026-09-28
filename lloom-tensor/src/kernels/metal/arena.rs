//! The activation arena: one allocation, slots at fixed offsets.
//!
//! `DESIGN.md` §9.2. Decode activation shapes are compile-time constants (§4.2),
//! so which buffer serves a given dispatch is a question that can be answered
//! once, offline, instead of by an allocator on every token.
//!
//! # What this is for, and it is not memory
//!
//! Issue #68 measured the packing: 517 activation values fit in **5 slots,
//! 68 KB**, against a 5.2 GB pool. That is 0.0013 % — packing activations saves
//! nothing worth having.
//!
//! The payoff is **dispatch stability** (§9.2c). Buffer identity varies at all
//! 674 decode dispatch positions today (§11.1a.1), because the pool hands out a
//! different allocation for the same logical slot on every token, and ICB replay
//! needs that identity fixed. The arena makes every activation a region of one
//! buffer, so the identity a dispatch binds stops depending on allocation
//! history. **Its acceptance criterion is "674 varying → 0", not bytes saved.**
//!
//! # Layer 3, and why that is not a detail
//!
//! §9.2a: the pool has three layers, and each is optional to the one above.
//!
//! 1. automatic `Drop` reclamation — the default, and it stays
//! 2. explicit control (`trim`, unpooled handles) — opt-in, on top
//! 3. **the arena** — this module
//!
//! A slot is a `PooledBuffer` handle held by the plan for the plan's lifetime,
//! so the free list simply never sees it. That is exactly how resident weights
//! are already handled: a weight's handle is never dropped, so it is never in a
//! free list and no lookup ever looks at it (§6.3a). **No new mechanism, no
//! manual `free()`, and no exemption from the reuse path** — the reuse path is
//! defined by whether anyone lets go, and the arena simply does not.
//!
//! Do not invert this. A design that required callers to release arena buffers
//! by hand would push manual lifetime management onto every caller permanently,
//! which is the property §9.2a says any replacement must preserve.
//!
//! # Liveness keys on the value, never on the buffer
//!
//! §9.2c, and it is the trap #68 hit and self-caught. Candle's pool recycles one
//! allocation across unrelated values *within a single token*: in #68's trace
//! `buf#328` is written and fully consumed 13 times per token, and `buf#47`
//! carries 60 distinct values. Keying liveness on `(buffer, offset)` merges
//! those into one interval and invents lifetimes no activation has — #68's first
//! planner did exactly that and reported **3.55 MB against the true 68 KB**, a
//! 52× overstatement.
//!
//! In candle the buffer identity is right there in the allocation call, so the
//! wrong key is the *easier* one to reach for. This module does not key on it at
//! all: a slot is chosen by **allocation ordinal within the step**, which is a
//! property of where the allocation sits in the sequence, not of what the pool
//! happened to hand out last time. See [`ArenaPlan::acquire`].
//!
//! # Session state must not enter the arena
//!
//! §9.1 and #68's finding 4. 169 dispatch positions write values that grow
//! 128 B per token — the KV cache and the `Tensor::cat` copies it moves through.
//! Those are *session state*: per-sequence, live across steps, sized by
//! `kv_len`. A slot whose occupant grows every token cannot hold a fixed offset,
//! so everything packed after it drifts; with KV left in, **969 bindings fail
//! the 674-stability check**.
//!
//! The plan enforces this by construction rather than by classification: a
//! request whose size does not match the slot recorded for that ordinal is
//! declined and falls through to the pool (see [`ArenaPlan::acquire`]). A
//! growing KV allocation changes size every token, so it declines every token.
//! `arena_size_mismatch_declines_rather_than_drifting` pins it.
//!
//! # The hazard
//!
//! §9.3: under `HazardTrackingModeUntracked` there is **no safety net**, and
//! aliasing correctness rests entirely on this module's offset arithmetic. Every
//! layout must be validated by a parity test against a non-aliasing reference
//! layout — one where every value gets its own slot and aliasing is impossible
//! by construction. [`ArenaLayout::NonAliasing`] is that reference, and it is an
//! *execution* comparison rather than a size comparison: both layouts run the
//! real model and their outputs are compared.

// No `BufferPool` here: the arena does not touch the pool at all (issue #489).
// Its slots are unpooled handles over its own base, which is what §9.2a's
// layer 3 means -- the free list never sees them. The test module below imports
// one of its own, to assert that installing an arena leaves it untouched.
use super::{ArenaOffsets, Buffer, PooledBuffer};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// The recorder currently observing binds, if any.
///
/// A process-global slot rather than a field, because binds are observed in the
/// encoder (`candle-metal-kernels`) while the recorder is driven from the device
/// (`candle-core`), and threading a handle from one to the other would mean a
/// parameter on `set_input_buffer`/`set_output_buffer` -- the two functions
/// `DESIGN.md` §6.4a measures as the hottest per-bind path in the backend.
///
/// The cost when nothing is recording is one relaxed atomic load, the same shape
/// the dispatch trace uses for the same reason. Recording happens once, over two
/// decode steps, before the arena exists; it is not a steady-state path.
static BIND_OBSERVER: OnceLock<Mutex<Option<Arc<Mutex<ArenaRecorder>>>>> = OnceLock::new();
static OBSERVING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn observer_slot() -> &'static Mutex<Option<Arc<Mutex<ArenaRecorder>>>> {
    BIND_OBSERVER.get_or_init(|| Mutex::new(None))
}

/// Install `recorder` as the bind observer, or clear it with `None`.
pub fn set_bind_observer(recorder: Option<Arc<Mutex<ArenaRecorder>>>) {
    {
        let mut slot = observer_slot().lock();
        OBSERVING.store(recorder.is_some(), Ordering::Relaxed);
        *slot = recorder;
    }
}

/// Whether anything is observing binds. One relaxed load.
#[inline(always)]
pub fn observing_binds() -> bool {
    OBSERVING.load(Ordering::Relaxed)
}

/// Report that a dispatch bound the buffer at `addr`.
#[inline]
pub fn note_bind(addr: usize) {
    if !observing_binds() {
        return;
    }
    {
        let slot = observer_slot().lock();
        if let Some(rec) = slot.as_ref() {
            {
                let mut r = rec.lock();
                r.record_bind(addr);
            }
        }
    }
}

/// Report that a new `MTLBuffer` object exists at `addr`.
///
/// The recorder names values by the address of the buffer object serving
/// them, and an address outlives its object: once the pool destroys a buffer
/// (eviction, a trim), the allocator is free to hand the same address to the
/// next object created on any path. Most of those paths never tell the
/// recorder about their allocation (the Shared pool, host uploads, heap
/// placements), so without this the new object's binds are charged to
/// whatever value the old one held — a phantom escape when that value was
/// the previous pass's, a stretched interval when it was this pass's. Called
/// from `Buffer::new`, the one constructor of an object of our own; one
/// relaxed load when nothing records (`DESIGN.md` §9.2aa).
#[inline]
pub fn note_new_buffer(addr: usize) {
    if !observing_binds() {
        return;
    }
    let slot = observer_slot().lock();
    if let Some(rec) = slot.as_ref() {
        rec.lock().forget_addr(addr);
    }
}

/// Dispatches encoded since the installed arena's step began -- the clock the
/// plan's allocation positions are checked against (`Arena::acquire_attributed`).
static STEP_DISPATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Whether an arena step is open, so the counter above advances only while
/// something reads it. One relaxed load on the classical path.
static DISPATCH_COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Open [`EscapeGuard`]s. While one is open, every allocation is DECLARED to
/// outlive its pass (`DESIGN.md` §9.2aa, "escapes are declared"): the
/// recorder marks it excluded so no plan ever serves its ordinal, and an
/// installed arena declines it, so it takes the pool (or a region tenant)
/// whichever pass it is made in. A value a later pass of a *different* key
/// reads — a drafter's tap, a probe's output — cannot be found by the
/// cross-pass detector, which only sees the pass right after; the guard is
/// how its producer says so up front.
static ESCAPING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether an [`EscapeGuard`] is open. One relaxed load.
#[inline(always)]
pub fn escaping() -> bool {
    ESCAPING.load(Ordering::Relaxed) > 0
}

/// Declares the allocations made while it is alive as escaping their pass.
/// Nestable; process-wide, like the dispatch counter it composes with (the
/// hot thread is the one encoding).
#[must_use = "the guard declares escapes only while it is alive"]
pub struct EscapeGuard(());

impl EscapeGuard {
    pub fn new() -> Self {
        ESCAPING.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

impl Default for EscapeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EscapeGuard {
    fn drop(&mut self) {
        ESCAPING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Report that a dispatch was encoded, advancing the position intervals live in.
#[inline]
pub fn note_dispatch() {
    // A dispatch under an escape guard is a declared outsider's (a streamed
    // expert's landing copy, whose count is the pool's state, not the pass
    // shape's): it takes no position, so a pass whose outsiders vary is
    // not a drift (#749 P28).
    if ESCAPING.load(Ordering::Relaxed) > 0 {
        return;
    }
    if DISPATCH_COUNTING.load(Ordering::Relaxed) {
        STEP_DISPATCH.fetch_add(1, Ordering::Relaxed);
    }
    if !observing_binds() {
        return;
    }
    {
        let slot = observer_slot().lock();
        if let Some(rec) = slot.as_ref() {
            {
                let mut r = rec.lock();
                r.record_dispatch();
            }
        }
    }
}

/// The shape of one forward pass, as the key a cached plan is valid for
/// (`DESIGN.md` §9.2aa). A plan is a slot assignment over an allocation
/// sequence; the sequence is a function of what the pass computes, so two
/// passes of one key allocate the same ordinals in the same positions and
/// the drift guard refuses the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// One decode step over `b` lockstep rows.
    Decode { b: u16 },
    /// One prefill forward of `w` tokens (a tile, a ramp unit, or the whole
    /// prompt). `fresh` is the first pass on an empty cache, which allocates
    /// the session state (KV slab, conv state) that every later pass finds
    /// in place — a different allocation sequence, so a different key.
    Prefill { w: u32, fresh: bool },
    /// One speculative verify pass of `k` positions.
    Verify { k: u16 },
    /// One batched step of `live` decode rows with `pieces` prompt pieces
    /// riding it as extra ragged rows (#728). `layout` hashes the pieces'
    /// widths in order: the dispatch sequence is a function of every width,
    /// so each distinct layout is its own key. The caller bounds the set —
    /// the `ladder` arm quantises widths, the `lru` arm bounds the cache.
    Ragged { live: u16, pieces: u16, layout: u64 },
}

/// Who opened a pass. A driver opens its pass before calling the model, so
/// the pass holds everything `forward` allocates; a model that brackets a
/// pass no driver holds opens it inside `forward`, after its embedding. One
/// shape therefore names two allocation sequences, and a shift of whole
/// allocate-and-dispatch operations moves an ordinal and its position
/// together, which the drift guard cannot see. Keying on the opener keeps
/// each plan to the sequence it was recorded from (`DESIGN.md` §9.2aa).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PassOpener {
    #[default]
    Driver,
    Model,
}

/// A pass kind under the axes that change its dispatch sequence. `arms` is
/// the caller's hash of those axes; the harnesses run one configuration per
/// process and pass 0. `opener` is who bracketed the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassKey {
    pub kind: PassKind,
    pub arms: u64,
    pub opener: PassOpener,
}

impl PassKey {
    /// A driver's pass of `kind`.
    pub fn new(kind: PassKind) -> Self {
        Self {
            kind,
            arms: 0,
            opener: PassOpener::Driver,
        }
    }

    /// This key, for a pass `opener` bracketed.
    pub fn opened_by(self, opener: PassOpener) -> Self {
        Self { opener, ..self }
    }
}

/// What the pass-keyed plan cache did over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassPlanCounters {
    /// Passes that ran under a cached plan.
    pub served: u64,
    /// Passes that ran on the pool while a key was recorded (its first sight).
    pub recorded: u64,
    /// Passes that ran on the pool because the previous pass's key was still
    /// awaiting the cross-pass liveness check.
    pub observed: u64,
    /// Plans built.
    pub plans: u64,
    /// Plans dropped after the drift guard fired under them.
    pub evicted: u64,
    /// Ordinals that outlived their pass UNDECLARED, summed over the plans
    /// built: values the pass after a key's first sight read and no
    /// [`EscapeGuard`] had claimed. Each is served from the pool (correct,
    /// as under `--arena`) and each is a hole in the no-allocation invariant
    /// — on the ring conv arm with the logits hold this is 0.
    pub escapes: u64,
    /// Times the shared base buffer was (re)allocated.
    pub base_grows: u64,
    /// `Ragged` plans dropped because the cache holds at most
    /// `MetalDevice::set_pass_plan_ragged_cap` of them (the `lru` arm's
    /// bound); a dropped layout is recorded again on its next sight.
    pub capped: u64,
}

/// Alignment of every slot, in bytes.
///
/// 128 B, per §9.2 and our fork's PR #8. Two independent reasons land on the
/// same number and both must hold:
///
/// - it covers every Metal dtype (the kernels use `float4`/`half4` at 16 B and
///   quantized block structures are wider, all well under 128)
/// - `hw.cachelinesize` on this machine is **also** 128 B, so a slot boundary is
///   a CPU cache-line boundary
///
/// §9.2 is explicit that the coincidence is load-bearing in one direction:
/// **lowering this to 64 would silently introduce false sharing between adjacent
/// slots** on any CPU-side access, and the Metal-dtype reasoning alone would not
/// reveal that. Do not change it without measuring both.
pub const ARENA_ALIGNMENT: usize = 128;

/// Which layout an arena hands out.
///
/// Both are compiled and either is selectable at run time, the same discipline
/// `ParamStyle` follows for binding styles (§11.3b): keeping both live makes the
/// A/B free, and makes the parity check in §9.3 a comparison between two paths
/// that actually exist rather than between one path and an argument.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArenaLayout {
    /// The packed plan: values whose liveness intervals are disjoint share a
    /// slot. This is the layout §9.2 specifies and the one that saves the 68 KB.
    #[default]
    Packed,

    /// **The reference layout §9.3 requires.** Every value gets its own slot, so
    /// no two values ever share bytes and aliasing is impossible by
    /// construction.
    ///
    /// This is not a debugging aid, it is the oracle. The packed layout is
    /// correct only if it computes what this one computes; any offset-arithmetic
    /// error shows up as a difference between them, and under
    /// `HazardTrackingModeUntracked` there is no other detector (§9.3).
    NonAliasing,
}

/// One slot: a fixed region of the arena buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// Byte offset of the slot within the arena buffer. 128 B aligned.
    pub offset: usize,
    /// Bytes reserved for the slot -- its largest occupant.
    pub size: usize,
}

/// What a decode step's allocations map to.
///
/// Index is the **allocation ordinal within the step**: the Nth activation
/// allocation of every token gets entry N. That is the positional keying #68's
/// finding 1 asks for — "the assignment is a property of where a dispatch sits
/// in the sequence" — and it is what makes the plan independent of which buffer
/// the pool handed out.
#[derive(Clone, Debug, Default)]
pub struct StepPlan {
    slots: Vec<Slot>,
    /// Slot index per allocation ordinal, and the size that ordinal expects.
    ///
    /// `None` means the ordinal is **not the arena's** -- session state, whose
    /// size grows with `kv_len` (§9.1, #68 finding 4). It keeps its place in the
    /// sequence so excluding it does not renumber later ordinals, and it is
    /// declined at `acquire` so it allocates from the pool as before.
    ///
    /// The size is carried so a request that does not match can also be
    /// declined, rather than silently served a slot of the wrong extent.
    by_ordinal: Vec<Option<(usize, usize)>>,
    /// The dispatch position each ordinal was allocated at when the plan was
    /// recorded, or empty for a plan built without a recording.
    ///
    /// The drift guard's second key. A size matches by coincidence often --
    /// most activations of a decode step are the hidden size -- so a sequence
    /// that gained or lost a dispatch before ordinal N can hand N a request of
    /// the planned size for a *different* value, with a different lifetime,
    /// and the packed layout would alias it (§9.3). The position it was
    /// allocated at cannot coincide across such a shift. It can across a
    /// shift of whole allocate-and-dispatch operations — one allocation and
    /// one dispatch move the ordinal and the position together — so two
    /// sequences that differ that way must not share a key (§9.2aa).
    alloc_at: Vec<usize>,
}

impl StepPlan {
    /// Build a plan from the per-ordinal `(slot, size)` assignment.
    pub fn new(slots: Vec<Slot>, by_ordinal: Vec<Option<(usize, usize)>>) -> Self {
        Self {
            slots,
            by_ordinal,
            alloc_at: Vec::new(),
        }
    }

    /// Attach the dispatch position each ordinal was allocated at, one per
    /// ordinal, so `Arena::acquire_attributed` can refuse a request that
    /// arrives at a different position than the plan recorded.
    pub fn with_alloc_positions(mut self, alloc_at: Vec<usize>) -> Self {
        assert_eq!(
            alloc_at.len(),
            self.by_ordinal.len(),
            "one allocation position per ordinal"
        );
        self.alloc_at = alloc_at;
        self
    }

    /// The dispatch position ordinal `ordinal` was allocated at, when known.
    pub fn alloc_position(&self, ordinal: usize) -> Option<usize> {
        self.alloc_at.get(ordinal).copied()
    }

    /// Drop the allocation positions, for a plan whose intervals are a
    /// fixture rather than a recording: the drift guard then has nothing to
    /// check, and every ordinal is served on size alone.
    pub fn without_alloc_positions(mut self) -> Self {
        self.alloc_at.clear();
        self
    }

    /// Ordinals the arena serves, against the total recorded.
    pub fn covered(&self) -> (usize, usize) {
        (
            self.by_ordinal.iter().filter(|e| e.is_some()).count(),
            self.by_ordinal.len(),
        )
    }

    /// Slot serving each ordinal, `None` where the ordinal is not the arena's.
    pub fn slot_of(&self, ordinal: usize) -> Option<usize> {
        self.by_ordinal
            .get(ordinal)
            .copied()
            .flatten()
            .map(|(s, _)| s)
    }

    /// Total bytes the arena must reserve.
    pub fn arena_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.offset + s.size)
            .max()
            .unwrap_or(0)
    }

    /// Bytes a **bump allocator** must be given to reproduce this plan.
    ///
    /// Not the same number as [`Self::arena_bytes`], and the difference is a
    /// real one rather than a rounding preference (issue #70).
    ///
    /// `arena_bytes` reports `max(offset + size)` -- where the last *value*
    /// ends. A cursor rounds **every** request up to [`ARENA_ALIGNMENT`],
    /// including the final one, so it ends where the last *slot* ends. For a
    /// plan whose last size is not a multiple of 128 those differ, and the
    /// cursor's figure is the larger:
    ///
    /// ```text
    /// sizes [100, 300, 5000]  ->  slots at 0, 128, 512
    ///   arena_bytes    = 512 + 5000 = 5512   (where the value ends)
    ///   bump_capacity  = 512 + 5120 = 5632   (where the slot ends)
    /// ```
    ///
    /// Handing a bump allocator `arena_bytes` would make it decline the last
    /// ordinal -- an allocation that fits the plan perfectly, refused because
    /// the capacity it was checked against measured something else. That is a
    /// silent loss of arena coverage rather than a corruption, since a declined
    /// ordinal falls through to the pool, but it would have shown up as "the
    /// GPU path serves fewer ordinals than the CPU path" with no obvious cause.
    ///
    /// `arena_bytes` is deliberately left alone: it decides how large an
    /// `MTLBuffer` #69 allocates, and changing that for every existing arena is
    /// a wider blast radius than this issue's evidence covers. The extra bytes
    /// are tail padding on one slot.
    pub fn bump_capacity(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.offset + align_up(s.size, ARENA_ALIGNMENT))
            .max()
            .unwrap_or(0)
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn allocations(&self) -> usize {
        self.by_ordinal.len()
    }

    /// The request sizes a GPU bump allocator must walk to reproduce this plan.
    ///
    /// One entry per allocation ordinal, in ordinal order: the byte size that
    /// ordinal requests, or **0** where the arena declines it. Zero is the
    /// spelling `arena_alloc.metal` reads as "not mine", and it consumes no
    /// bytes -- so a declined ordinal keeps its position without shifting the
    /// offsets of those after it, which is the same property `by_ordinal`'s
    /// `None` gives the CPU path (§9.1, issue #70).
    ///
    /// # Why the sizes and not the offsets
    ///
    /// Handing the kernel the *offsets* would make the comparison against the
    /// CPU plan vacuous -- it would check that the GPU can copy an array.
    /// Handing it the sizes makes it re-derive the layout, so agreement is
    /// evidence that the two allocators compute the same thing.
    pub fn request_sizes(&self) -> Vec<u32> {
        self.by_ordinal
            .iter()
            .map(|e| e.map_or(0, |(_, size)| size as u32))
            .collect()
    }

    /// The offset each ordinal must receive, or `None` where it is declined.
    ///
    /// The oracle `ArenaCursor::verify_against` compares a GPU run against.
    /// Derived from the same `by_ordinal` the CPU path binds through, so the
    /// two cannot drift apart.
    ///
    /// **This equals a bump allocator's output only when the plan is itself a
    /// bump layout** -- one slot per ordinal, laid out in ordinal order. A
    /// *packed* plan reuses slots, so ordinal 7 may resolve to an offset
    /// earlier than ordinal 3's, which no forward-only cursor can produce. That
    /// is not a defect in either allocator; it is why the GPU path serves
    /// [`ArenaLayout::NonAliasing`] and declines `Packed` rather than silently
    /// disagreeing. See [`Self::is_bump_reproducible`].
    pub fn expected_offsets(&self) -> Vec<Option<usize>> {
        self.by_ordinal
            .iter()
            .map(|e| e.map(|(slot, _)| self.slots[slot].offset))
            .collect()
    }

    /// Whether these offsets are reproducible by a forward-only bump allocator.
    ///
    /// True when each served ordinal's offset is strictly greater than the
    /// previous served ordinal's, which is all a cursor that only increases can
    /// produce. A packed plan is not, by construction: reusing a slot means
    /// revisiting an offset.
    ///
    /// Checked rather than inferred from the layout enum, because the property
    /// that matters is the *shape of the offsets*, not the name of the layout
    /// that produced them. A future layout that happened to be monotone would
    /// be served correctly, and a `NonAliasing` plan that somehow was not would
    /// be caught here rather than at a wrong bind -- which under
    /// `HazardTrackingModeUntracked` is silent (§3.5).
    pub fn is_bump_reproducible(&self) -> bool {
        let mut previous: Option<usize> = None;
        for entry in self.by_ordinal.iter().flatten() {
            let offset = self.slots[entry.0].offset;
            if let Some(p) = previous {
                if offset <= p {
                    return false;
                }
            }
            previous = Some(offset);
        }
        true
    }

    /// Every slot is 128 B aligned and no two slots overlap.
    ///
    /// The overlap half is the one that matters: two slots sharing bytes would
    /// alias two values that the plan believes are disjoint, and §9.3 says
    /// nothing in the driver would catch it. Checked exhaustively rather than
    /// sampled, for the reason #68 gives -- a sampled check on an aliasing
    /// invariant is worth very little.
    pub fn check_disjoint(&self) -> Result<(), String> {
        for (i, s) in self.slots.iter().enumerate() {
            if !s.offset.is_multiple_of(ARENA_ALIGNMENT) {
                return Err(format!(
                    "slot {i} at offset {} is not {ARENA_ALIGNMENT} B aligned",
                    s.offset
                ));
            }
        }
        for (i, a) in self.slots.iter().enumerate() {
            for (j, b) in self.slots.iter().enumerate().skip(i + 1) {
                let overlap = a.offset < b.offset + b.size && b.offset < a.offset + a.size;
                if overlap {
                    return Err(format!(
                        "slots {i} ({}..{}) and {j} ({}..{}) overlap",
                        a.offset,
                        a.offset + a.size,
                        b.offset,
                        b.offset + b.size
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Counters, so "arena buffers never enter the free list" is checkable rather
/// than asserted.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArenaCounters {
    /// Allocation requests offered to the arena.
    pub offers: u64,
    /// Requests served from a slot.
    pub hits: u64,
    /// Requests declined because the ordinal is session state.
    ///
    /// Excluded when the plan was built, because the allocation's size moved
    /// between the two recorded steps -- the KV cache and its `Tensor::cat`
    /// copies, which grow 128 B per token (§9.1, #68 finding 4). A zero here on
    /// a real LFM2 decode would mean the exclusion never engaged, and the drift
    /// it prevents would be reachable.
    pub declined_session: u64,

    /// Requests declined because the size did not match the ordinal's slot.
    ///
    /// The second line of defence, behind the exclusion above: it catches an
    /// allocation whose size moved in a way two recorded steps did not reveal.
    /// Serving it would either overrun the slot or pin a growing value at a
    /// fixed offset.
    pub declined_size: u64,
    /// Requests past the end of the plan.
    pub declined_exhausted: u64,
    /// Steps on which a request arrived at a dispatch position other than the
    /// one its ordinal was recorded at, so the sequence had drifted from the
    /// plan. The arena stops serving for the rest of that step and reports
    /// the drift for the loop to re-record (`Arena::take_drifted`).
    pub declined_drift: u64,
    /// Requests made under an [`EscapeGuard`], declined as declared escapes.
    pub declined_escaped: u64,
    /// Decode steps begun.
    pub steps: u64,
    /// Times a GPU-computed offset table was verified equal to the plan and
    /// adopted (issue #70).
    ///
    /// Zero on a run that asked for GPU offsets means the switch never engaged,
    /// which §2.4 says must be distinguishable from a run where it did: #69's
    /// first determinism gate was vacuous precisely because a flag was believed
    /// rather than checked against a quantity it should have moved.
    pub gpu_verified: u64,
}

/// The live counters, one relaxed atomic per field (issue #562).
///
/// # Why these are atomics and not a `Mutex<ArenaCounters>`
///
/// A lock is required when state is **mutated**, by **more than one thread**,
/// under a **compound invariant** -- and these fail the third test and the
/// second. Every field is an independent monotone counter: nothing reads two of
/// them and requires that they were sampled at one instant, because the only
/// consumers are `Arena::counters()` at a step boundary
/// (`lfm2-dispatch-trace/main.rs:807`) and this file's own tests. So there is no
/// invariant *between* fields for a lock to preserve, which is what the
/// `PoolInner::state` mutex genuinely does have (`pending`/`free`/`free_bytes`/
/// `free_order` move together or the free list is wrong).
///
/// **What the lock was FORCING is the point rather than what it cost.** §9.2o
/// measures an uncontended `pthread_mutex` at 12.49 ns, so this is not a
/// throughput change and no speedup is claimed for it. What the mutex forced is
/// that a *counter increment* and an *immutable plan lookup* were the same
/// critical section, so the hot path could not read the plan without also
/// serialising against a bump of `offers` -- and the plan is `&'static`-shaped
/// data that needs no serialisation at all.
///
/// `Relaxed` throughout, for the reason §9.2p gives for `ordinal`: nothing
/// downstream derives an ordering from a counter's value. They are read once,
/// after the step, by a harness that wants a total rather than a sequence.
#[derive(Debug, Default)]
struct LiveCounters {
    offers: AtomicU64,
    hits: AtomicU64,
    declined_session: AtomicU64,
    declined_size: AtomicU64,
    declined_exhausted: AtomicU64,
    declined_drift: AtomicU64,
    declined_escaped: AtomicU64,
    steps: AtomicU64,
    gpu_verified: AtomicU64,
}

impl LiveCounters {
    fn snapshot(&self) -> ArenaCounters {
        ArenaCounters {
            offers: self.offers.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            declined_session: self.declined_session.load(Ordering::Relaxed),
            declined_size: self.declined_size.load(Ordering::Relaxed),
            declined_exhausted: self.declined_exhausted.load(Ordering::Relaxed),
            declined_drift: self.declined_drift.load(Ordering::Relaxed),
            declined_escaped: self.declined_escaped.load(Ordering::Relaxed),
            steps: self.steps.load(Ordering::Relaxed),
            gpu_verified: self.gpu_verified.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.offers.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.declined_session.store(0, Ordering::Relaxed);
        self.declined_size.store(0, Ordering::Relaxed);
        self.declined_exhausted.store(0, Ordering::Relaxed);
        self.declined_drift.store(0, Ordering::Relaxed);
        self.declined_escaped.store(0, Ordering::Relaxed);
        self.steps.store(0, Ordering::Relaxed);
        self.gpu_verified.store(0, Ordering::Relaxed);
    }
}

/// The plan and its slot handles: written once, read for the arena's life.
///
/// # Why this is not behind a lock (issue #562)
///
/// Every field here is established in [`Arena::new`] and never mutated
/// afterwards -- verified by `the_plan_is_never_mutated_after_construction`,
/// which is a source-level census in #507's and §9.2p's shape, because
/// immutability is a property of the **write set** and a runtime test can only
/// observe that no sequence it happened to try disturbed it.
///
/// So the hot path's lookups (`by_ordinal.get(ordinal)`, `slot_handles[slot]`)
/// are reads of immutable data reached through the `Arc`, which is exactly the
/// shape that needs no synchronisation: the `Arc` already publishes them, and
/// there is no later write for a reader to race.
struct ArenaPlanState {
    plan: StepPlan,
    layout: ArenaLayout,
    /// One handle per slot, created once and held for the plan's lifetime.
    ///
    /// **This is what makes the arena layer 3.** The handles never drop while
    /// the plan lives, so the free list never sees them and no lookup ever
    /// considers them -- exactly how resident weights already behave (§6.3a).
    /// Nothing here calls `free`.
    slot_handles: Vec<Arc<PooledBuffer>>,
}

/// Why the arena declined an allocation.
///
/// Every variant is a **normal outcome**, not an error: the allocation falls
/// through to the pool exactly as it did before the arena existed. They are
/// distinguished because they want different responses, and reporting one number
/// that could mean any of them is the shape §3.1c warns about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaDecline {
    /// Outside a decode step: prefill, model setup, sampling.
    Inactive,
    /// The step allocated more values than the plan recorded.
    PastPlan,
    /// **Session state**, excluded when the plan was built (§9.1).
    ///
    /// This is the population the session region serves (#47): per-sequence
    /// state whose size grows with `kv_len`, which a fixed-offset slot cannot
    /// hold. Measured on LFM2 decode this is every decline the arena makes —
    /// `lfm2-dispatch-trace --arena` reports `declined(size) 0` and
    /// `declined(past plan) 0`.
    Session,
    /// The request's size is not what this ordinal was planned for.
    SizeMismatch,
    /// The request arrived at a different dispatch position than the ordinal
    /// was recorded at: the step's sequence has drifted from the plan, and
    /// the arena serves nothing more this step.
    Drift,
    /// Made under an [`EscapeGuard`]: declared to outlive the pass, so no
    /// slot may hold it whatever the plan says about the ordinal.
    Escaped,
}

/// What the arena did with one offered allocation.
#[derive(Clone)]
pub struct ArenaOffer {
    /// The buffer, when the arena served it.
    pub served: Option<Arc<PooledBuffer>>,
    /// The allocation ordinal this offer consumed, when a step was open.
    ///
    /// **The key a downstream region must use**, and the reason this struct
    /// exists: the ordinal is consumed by the `fetch_add` inside `acquire`, so a
    /// caller cannot recompute it without double-counting and walking the plan
    /// out of alignment.
    pub ordinal: Option<usize>,
    /// Why it declined, or `None` when it served.
    pub decline: Option<ArenaDecline>,
}

/// The activation arena.
///
/// Cheap to clone; every clone refers to the same arena.
#[derive(Clone)]
pub struct Arena {
    inner: Arc<ArenaInner>,
}

struct ArenaInner {
    /// The one allocation. §9.2's "the arena is one allocation": if a future
    /// class needs more, it gets its own arena rather than turning this into an
    /// array of buffers.
    base: Buffer,
    /// The plan and its handles: immutable after construction, so no lock
    /// (issue #562). See [`ArenaPlanState`] for why that is sound here and is
    /// not sound for `PoolInner::state`.
    plan_state: ArenaPlanState,
    /// Where the offsets this arena binds through came from (issue #70).
    ///
    /// `Cpu` until [`Arena::adopt_gpu_offsets`] verifies a GPU-computed table
    /// against the plan. It is a *recorded* fact rather than a behavioural
    /// switch, because adoption only happens when the two agree element-wise --
    /// so the bytes bound are identical either way, and what this names is
    /// which allocator computed them.
    ///
    /// An `AtomicU8` rather than a mutex field: it is a single value with no
    /// invariant tying it to anything else, written once by whoever adopts and
    /// read by a reporting path. `Relaxed` because no reader derives an
    /// ordering from it -- adoption has already *verified* the offsets are
    /// element-wise equal to the plan's, so both arms bind identical bytes and
    /// what this records is which allocator computed them (issue #562).
    offsets: AtomicU8,
    /// Live counters, one relaxed atomic each (issue #562).
    counters: LiveCounters,
    /// Set when a step's sequence drifted from the plan; read and cleared by
    /// the loop at the step's close (`Arena::take_drifted`).
    drifted: std::sync::atomic::AtomicBool,
    /// Allocation ordinal within the current decode step.
    ///
    /// Relaxed because a decode step is single-threaded through the model's
    /// forward pass; this counter orders allocations against each other on one
    /// thread, and nothing reads it from another.
    ///
    /// # Deliberately NOT `CachePadded`, and this is the recorded reason (#347)
    ///
    /// `GpuClock`'s `submitted`/`completed` pair is padded (`buffer_pool.rs`,
    /// §9.2l) because it is a **concurrent write/write**: the host increments
    /// one in `commit_swap_locked` while Metal's completion handler increments
    /// the other on a driver thread, ~11-14 times per token, which §9.2l models
    /// at **5.42x on one line against two**. The question for this field is
    /// whether it has that shape. **It does not, and the discriminator is a
    /// second writer rather than a hot one.**
    ///
    /// Enumerated rather than assumed -- this field has exactly three access
    /// sites in the tree, and all three are the decode-loop thread:
    ///
    /// | site | what | thread |
    /// |---|---|---|
    /// | `Arena::new` | `AtomicUsize::new(0)` | whoever installs the arena |
    /// | `begin_step` | `store(0)` | the decode loop, once per token |
    /// | `acquire` | `fetch_add(1)` | the decode loop, per allocation |
    ///
    /// `begin_step` is reached only from `MetalDevice::begin_decode_step`,
    /// whose only callers are the three LFM2 harnesses' decode loops, and
    /// `acquire` only from `MetalDevice::new_buffer`, which a `Tensor` op calls
    /// on the thread running the forward pass. **The one genuinely concurrent
    /// path in this backend is `Commands::on_command_buffer_complete`, and its
    /// subscribers are `shared.drain_completed()` and
    /// `private.drain_completed()` (`metal_backend/mod.rs:2294-2297`) -- the
    /// pool's free list, never the arena.** So there is no second thread to
    /// share a line with, and padding would buy nothing at any contention
    /// level.
    ///
    /// **This is `kernel.rs`'s `pipelines` shape rather than `GpuClock`'s**
    /// (§9.2o): a single-threaded hot access, where the mechanism a
    /// concurrency remedy addresses is absent rather than small. And the
    /// single-threading is a **determinism requirement** rather than an
    /// accident -- §9.2g records that a concurrent bump allocator's
    /// ordinal->offset map varies run to run, and §2.3.2 makes a
    /// nondeterministic layout indistinguishable from a missing fence. A second
    /// thread here would be a correctness defect before it was a false-sharing
    /// one.
    ///
    /// `the_ordinal_has_one_writing_thread` pins the enumeration, so a future
    /// second writer fails a test rather than silently making this comment
    /// false. **What would change this: B > 1**, where §9.2l's own registered
    /// prediction is a null at B=1 for the same reason (#303: the decode CPU
    /// side is nearly all one thread), and where `Commands`'s already-padded
    /// pair becomes live rather than structural (§11.3o).
    ordinal: AtomicUsize,
    /// Whether a step is currently open. Outside a step the arena declines
    /// everything, so prefill and setup allocate from the pool as before.
    active: AtomicU64,
}

impl Arena {
    /// Allocate the arena and build its slot handles.
    ///
    /// One `MTLBuffer` of `plan.arena_bytes()`, allocated once here and never
    /// again. The slot handles are created now and held until the arena drops.
    ///
    /// # This takes no pool, and that is the point (issue #489)
    ///
    /// It used to take a `&BufferPool` and `adopt` each slot into it, which is
    /// what let a cleared arena's slots re-enter the free list. The parameter is
    /// removed rather than ignored so the property is enforced by the signature:
    /// **the arena cannot hand the pool a view of its base, because it is not
    /// given a pool to hand it to.** §9.2a's layer 3 is defined by the free list
    /// never seeing these buffers, and a comment is a weaker guarantee than an
    /// argument that does not exist (§6.7 L2).
    pub fn new(base: Buffer, plan: StepPlan, layout: ArenaLayout) -> Result<Self, String> {
        plan.check_disjoint()?;
        let needed = plan.arena_bytes();
        if base.length() < needed {
            return Err(format!(
                "arena buffer is {} B, plan needs {needed} B",
                base.length()
            ));
        }

        // One handle per slot, made once, **unpooled**. Every consumer still
        // takes the same `Arc<PooledBuffer>` it always did; what an unpooled
        // handle changes is what happens when the last one drops.
        //
        // # Why not `adopt` (lloom issue #489)
        //
        // `adopt` installs a back-reference to the pool, so dropping a slot
        // hands it *back* to the pool -- and a slot is a `Buffer::view` over the
        // arena's base, not an allocation of its own. This code previously
        // adopted, on the reasoning that "the handles are held here for the
        // arena's lifetime, so none is ever dropped and none reaches a free
        // list". That is true exactly while nothing drops the arena, and
        // `MetalDevice::clear_arena` does: measured, clearing put all five
        // slots of §9.2c's 69 632 B arena into `private_buffers`' free list,
        // from which `acquire` -- which keys on **size alone** and knows
        // nothing of provenance -- handed every one of them straight back out.
        // Five "separate" activation buffers, all views of one dropped base at
        // offsets 0/21504/43008/64512/68608, under
        // `HazardTrackingModeUntracked` where §9.3 says there is no safety net.
        //
        // §6.7 L4 in its silent-corruption form, and the failure would be
        // invisible: nothing downstream can tell a recycled arena view from an
        // allocation, which is the whole reason the pool must never be offered
        // one. Unpooled is `PooledBuffer::unpooled`'s stated purpose --
        // "allocations that are deliberately outside the pool, so that callers
        // which must not participate in reuse still get the same handle type"
        // -- and the arena is §9.2a layer 3, which is defined by the free list
        // never seeing its buffers.
        //
        // The base itself is unaffected: it is owned by `ArenaInner` and freed
        // by Metal when the last `Arena` clone drops, exactly as before.
        let slot_handles = plan
            .slots()
            .iter()
            .map(|s| Arc::new(PooledBuffer::unpooled(base.view(s.offset, s.size), s.size)))
            .collect();

        Ok(Self {
            inner: Arc::new(ArenaInner {
                base,
                plan_state: ArenaPlanState {
                    plan,
                    layout,
                    slot_handles,
                },
                // #69's path until a GPU table is verified against the plan.
                offsets: AtomicU8::new(ArenaOffsets::Cpu.to_bits()),
                counters: LiveCounters::default(),
                drifted: std::sync::atomic::AtomicBool::new(false),
                ordinal: AtomicUsize::new(0),
                active: AtomicU64::new(0),
            }),
        })
    }

    /// Adopt GPU-computed offsets for this arena (`DESIGN.md` §9.2d, issue #70).
    ///
    /// `gpu_offsets[i]` is the offset a kernel bump-allocated for ordinal `i`,
    /// or [`ARENA_DECLINED`](super::ARENA_DECLINED) where the plan declines it.
    ///
    /// # This verifies before it adopts, and that ordering is the whole point
    ///
    /// The offsets are checked element-wise against the plan **first**, and the
    /// arena refuses to switch if they disagree. So a GPU offset can never reach
    /// a bind unless it is the byte the CPU path would have chosen -- which
    /// makes bit-identical activations a structural property rather than
    /// something measured afterwards and hoped for.
    ///
    /// That matters more here than the usual "validate your inputs", because
    /// §9.3 is explicit that aliasing correctness rests entirely on this
    /// module's offset arithmetic and that `HazardTrackingModeUntracked` gives
    /// no safety net: an offset that is wrong by 128 bytes does not fail, it
    /// silently overlaps another value and corrupts intermittently. Refusing is
    /// the only safe response to a disagreement, and falling back to the CPU
    /// path costs nothing -- it is the default anyway.
    ///
    /// Returns the number of ordinals now served from GPU-computed offsets.
    pub fn adopt_gpu_offsets(&self, gpu_offsets: &[u32]) -> Result<usize, String> {
        let expected = self.inner.plan_state.plan.expected_offsets();
        if gpu_offsets.len() != expected.len() {
            return Err(format!(
                "GPU produced {} offsets, the plan has {} ordinals",
                gpu_offsets.len(),
                expected.len()
            ));
        }
        for (i, (&got, want)) in gpu_offsets.iter().zip(expected.iter()).enumerate() {
            match want {
                Some(w) if got as usize == *w => {}
                Some(w) => {
                    return Err(format!("ordinal {i}: GPU offset {got}, plan offset {w}"));
                }
                None if got == super::ARENA_DECLINED => {}
                None => {
                    return Err(format!("ordinal {i}: plan declines it, GPU offset {got}"));
                }
            }
        }

        // Equality held, so adopting changes no byte any dispatch will bind.
        // The switch is recorded rather than being a no-op with a nice comment:
        // `offsets_are_gpu` is what lets a harness report that the GPU path was
        // actually engaged, and §2.4's lesson from #69's vacuous determinism run
        // is that an instrument which cannot be shown to have engaged has not
        // measured anything.
        self.inner
            .offsets
            .store(ArenaOffsets::Gpu.to_bits(), Ordering::Relaxed);
        self.inner
            .counters
            .gpu_verified
            .fetch_add(1, Ordering::Relaxed);
        Ok(expected.iter().filter(|e| e.is_some()).count())
    }

    /// A copy of the plan this arena serves.
    ///
    /// Still cloned rather than borrowed, and the reason has changed (issue
    /// #562): the plan is no longer behind a mutex, so the old *"must not hold
    /// that lock across a dispatch"* justification is gone. What remains is
    /// that the return type is owned, and narrowing it to a borrow would tie
    /// every caller's lifetime to the arena for no benefit -- callers derive
    /// the GPU allocator's request sizes from it and outlive the borrow.
    pub fn plan_snapshot(&self) -> StepPlan {
        self.inner.plan_state.plan.clone()
    }

    /// Where this arena's offsets come from.
    pub fn offsets(&self) -> ArenaOffsets {
        ArenaOffsets::from_bits(self.inner.offsets.load(Ordering::Relaxed))
    }

    /// The single underlying allocation, for residency registration.
    ///
    /// §9.2's "residency is a CPU-side fact established once": one buffer in the
    /// residency set rather than 674.
    pub fn base(&self) -> &Buffer {
        &self.inner.base
    }

    pub fn layout(&self) -> ArenaLayout {
        self.inner.plan_state.layout
    }

    /// Open a decode step, resetting the allocation ordinal.
    ///
    /// The ordinal is what ties an allocation to a slot, so it must restart at
    /// the same point every token or the assignment would walk. The caller marks
    /// the boundary because only the caller knows where a step begins -- the same
    /// signal the dispatch-trace harness already takes for its per-step regions.
    pub fn begin_step(&self) {
        self.inner.ordinal.store(0, Ordering::Relaxed);
        STEP_DISPATCH.store(0, Ordering::Relaxed);
        DISPATCH_COUNTING.store(true, Ordering::Relaxed);
        self.inner.active.store(1, Ordering::Relaxed);
        self.inner.counters.steps.fetch_add(1, Ordering::Relaxed);
    }

    /// Close the current step. Allocations outside a step go to the pool.
    pub fn end_step(&self) {
        self.inner.active.store(0, Ordering::Relaxed);
        DISPATCH_COUNTING.store(false, Ordering::Relaxed);
    }

    /// Whether a step drifted from the plan since this was last asked, and
    /// clear it.
    ///
    /// A drifted plan describes a stream that no longer runs: every later
    /// step would decline at the same position and serve the ordinals before
    /// it, which is correct and covers less each time the sequence moves
    /// again. The loop that installed the arena re-records it instead
    /// (`MetalDevice::finish_decode_step`).
    pub fn take_drifted(&self) -> bool {
        self.inner.drifted.swap(false, Ordering::Relaxed)
    }

    pub fn is_active(&self) -> bool {
        self.inner.active.load(Ordering::Relaxed) != 0
    }

    /// Offer an allocation of `size` bytes to the arena.
    ///
    /// `None` means the caller should allocate from the pool as before, and it
    /// is a normal outcome rather than a failure: everything outside a decode
    /// step, everything past the end of the plan, and every session-state
    /// allocation lands here.
    ///
    /// # Why the ordinal, and not the buffer
    ///
    /// The slot is chosen by *where this allocation falls in the step*, which is
    /// stable across tokens because the op sequence is (§11.1a.1: kernel name
    /// varies at 0 of 674 positions). Choosing by buffer identity instead would
    /// key on exactly the thing the arena exists to remove, which is the 52×
    /// error §9.2c records.
    ///
    /// # The size gate
    ///
    /// A request is served only if its size matches what the plan recorded for
    /// that ordinal. This is what keeps session state out (§9.1, #68 finding 4):
    /// the KV cache grows 128 B per token, so it matches on no token and takes
    /// the pool path every time. Serving it a fixed slot would either overflow
    /// the slot or pin a growing value at a fixed offset, and everything packed
    /// after it would drift.
    ///
    /// # The `Relaxed` on the `fetch_add`, audited (#347)
    ///
    /// **`Relaxed` is correct because nothing downstream derives ORDERING from
    /// this value -- it is a unique-ticket source and a table index, and both
    /// uses are satisfied by uniqueness alone.** Traced from the returned
    /// `ordinal` to every consumer:
    ///
    /// - `state.plan.by_ordinal.get(ordinal)` -- a **pure index** into a `Vec`
    ///   built once at `install_arena` and never mutated afterwards. It reads
    ///   position `n`; it does not read "the entry the previous call read", so
    ///   no happens-before edge is required against any other `fetch_add`.
    /// - `state.slot_handles[slot]` -- indexed by the *plan's* slot, not by the
    ///   ordinal, and likewise immutable for the plan's lifetime.
    ///
    /// **And there is no ordering to supply, which is a stronger statement than
    /// the one this paragraph used to make** (issue #562). It previously read
    /// *"every read of the ordinal happens under `state.lock()` ... so the plan
    /// and the handles are already behind a `Mutex`'s acquire/release"* -- true
    /// when written, and it rested the `Relaxed` on an edge the mutex happened
    /// to supply. **That was the wrong reason to be right**: the plan and the
    /// handles are *immutable after construction*, published once by the `Arc`
    /// that owns them, so no reader needs an edge against any writer because
    /// there is no writer to race. #562 removed the mutex and the `Relaxed`
    /// stands on its own terms.
    ///
    /// Strengthening this `fetch_add` to `AcqRel` would order it against *other
    /// `fetch_add`s* -- which, per the field's own docs, are all on this one
    /// thread and therefore already ordered by program order. It would emit a
    /// barrier for an edge that exists for free.
    ///
    /// **The one property that IS load-bearing is uniqueness, and `Relaxed`
    /// gives it.** Two allocations must never receive the same ordinal, or two
    /// live values would resolve to one slot and alias under
    /// `HazardTrackingModeUntracked`, where §9.3 says there is no safety net and
    /// §2.3.2 says the corruption is indistinguishable from benign float noise.
    /// An RMW is atomic at every ordering, including `Relaxed`, and atomicity is
    /// what uniqueness rests on -- so this is not a case where the weaker
    /// ordering weakens the guarantee.
    ///
    /// **This differs from the GPU-side allocator's constraint and the two must
    /// not be conflated** (§9.2g, `arena_alloc.metal:33-42`). There, a
    /// *concurrent* `atomic_fetch_add` is refused because the hardware chooses
    /// the return order, so the ordinal->offset map would vary run to run --
    /// which is a **determinism** argument about who gets which ticket, not a
    /// memory-ordering one. Here there is one thread, so the ticket sequence is
    /// program order and the question does not arise.
    ///
    /// `nothing_downstream_derives_ordering_from_the_ordinal` pins the two
    /// consumers above, so a future reader that *does* need an edge fails a test
    /// rather than inheriting a `Relaxed` that no longer suffices.
    pub fn acquire(&self, size: usize) -> Option<Arc<PooledBuffer>> {
        self.acquire_attributed(size).served
    }

    /// As [`Self::acquire`], and it also reports **which ordinal** was consumed
    /// and **why** a decline happened.
    ///
    /// # Why this exists rather than a second counter read (#47)
    ///
    /// The session region (`session_region.rs`) serves the population this arena
    /// declines as session state, and it keys on the same allocation ordinal for
    /// the same reason this does: the kernel sequence is byte-identical across
    /// tokens (§11.1a.1), so the Nth allocation of one step is the Nth of the
    /// next. That key is consumed *inside* this function — the `fetch_add` is
    /// here — so a caller that wanted to offer the declined allocation onwards
    /// had no way to name it.
    ///
    /// Returning it is strictly cheaper than the alternatives and does not touch
    /// the ordinal's single-writer property: the `fetch_add` still happens
    /// exactly once, on the same thread, and this only reports what it returned.
    /// A second `fetch_add` in the caller would **double-count the ordinal** and
    /// walk the plan out of alignment, which is the defect this shape prevents
    /// rather than risks.
    ///
    /// `acquire` above is kept as the one-line wrapper so every existing caller
    /// is untouched — §3.6's *add a variant behind a seam, never rip out the
    /// block*.
    pub fn acquire_attributed(&self, size: usize) -> ArenaOffer {
        if !self.is_active() {
            return ArenaOffer {
                served: None,
                ordinal: None,
                decline: Some(ArenaDecline::Inactive),
            };
        }
        let ordinal = self.inner.ordinal.fetch_add(1, Ordering::Relaxed);
        let counters = &self.inner.counters;
        let plan_state = &self.inner.plan_state;
        counters.offers.fetch_add(1, Ordering::Relaxed);
        // Declared to escape: the ordinal is consumed (the sequence is what
        // it is) and the value is never a slot's, whatever the plan planned.
        if escaping() {
            counters.declined_escaped.fetch_add(1, Ordering::Relaxed);
            return ArenaOffer {
                served: None,
                ordinal: Some(ordinal),
                decline: Some(ArenaDecline::Escaped),
            };
        }

        let entry = match plan_state.plan.by_ordinal.get(ordinal) {
            Some(e) => *e,
            None => {
                counters.declined_exhausted.fetch_add(1, Ordering::Relaxed);
                return ArenaOffer {
                    served: None,
                    ordinal: Some(ordinal),
                    decline: Some(ArenaDecline::PastPlan),
                };
            }
        };
        // Not the arena's: session state, excluded when the plan was built
        // because its size moved between the two recorded steps (§9.1).
        let Some((slot, want)) = entry else {
            counters.declined_session.fetch_add(1, Ordering::Relaxed);
            return ArenaOffer {
                served: None,
                ordinal: Some(ordinal),
                decline: Some(ArenaDecline::Session),
            };
        };
        // Size must match what was planned for this ordinal. A larger request
        // would run past the slot; a smaller one means the sequence has drifted
        // from the plan and the ordinal no longer names what it did.
        if size != want {
            counters.declined_size.fetch_add(1, Ordering::Relaxed);
            return ArenaOffer {
                served: None,
                ordinal: Some(ordinal),
                decline: Some(ArenaDecline::SizeMismatch),
            };
        }
        // The position must match too. A size coincides across a shifted
        // sequence -- the hidden size is most of a step's allocations -- and a
        // request served at the planned slot for a *different* value has a
        // lifetime the packing never modelled, which is §9.3's aliasing with
        // no safety net. A mismatch here is the sequence having moved: this
        // step is served no further, and the loop re-records. A shift of whole
        // allocate-and-dispatch operations moves the ordinal and the position
        // together and passes (`StepPlan::alloc_at`).
        if let Some(at) = plan_state.plan.alloc_position(ordinal) {
            let now = STEP_DISPATCH.load(Ordering::Relaxed);
            if at != now {
                counters.declined_drift.fetch_add(1, Ordering::Relaxed);
                self.inner.drifted.store(true, Ordering::Relaxed);
                self.inner.active.store(0, Ordering::Relaxed);
                return ArenaOffer {
                    served: None,
                    ordinal: Some(ordinal),
                    decline: Some(ArenaDecline::Drift),
                };
            }
        }
        counters.hits.fetch_add(1, Ordering::Relaxed);
        ArenaOffer {
            served: Some(Arc::clone(&plan_state.slot_handles[slot])),
            ordinal: Some(ordinal),
            decline: None,
        }
    }

    /// A snapshot of the counters.
    ///
    /// **Not an atomic snapshot, and it does not need to be** (issue #562).
    /// Each field is loaded independently, so a caller reading during a step
    /// could in principle see `offers` from after a `hits` -- and no consumer
    /// does: `lfm2-dispatch-trace` reads them at a step boundary, and this
    /// file's tests read them with the decode loop stopped. The counters carry
    /// no invariant *between* fields, which is the property that made the mutex
    /// removable in the first place; a reader that needed one would be asking
    /// for a guarantee the old mutex gave only by accident of scope.
    pub fn counters(&self) -> ArenaCounters {
        self.inner.counters.snapshot()
    }

    pub fn reset_counters(&self) {
        self.inner.counters.reset();
    }

    /// Slots, for reporting and for the layout checks.
    pub fn slots(&self) -> Vec<Slot> {
        self.inner.plan_state.plan.slots().to_vec()
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plan = &self.inner.plan_state.plan;
        f.debug_struct("Arena")
            .field("slots", &plan.slots().len())
            .field("bytes", &plan.arena_bytes())
            .finish()
    }
}

/// Build a plan from a recorded sequence of activation allocation sizes.
///
/// `sizes[i]` is the byte size of the i-th activation allocation in a decode
/// step, and `last_use[i]` is the ordinal of the last allocation that may still
/// be reading it. Together those are the liveness interval #68 computes by
/// refcount (§9.2b), expressed positionally.
///
/// # Ordering is size-major, and that is measured rather than inherited
///
/// §9.2 and issue #43. Slots are assigned in **size-descending** order, not in
/// start order. Linear-scan register allocation sweeps in start order and gets
/// away with it because registers are uniform in size; arena slots are not, and
/// that is the whole difference.
///
/// #68 measured both on our shapes: size-major packs to **68.00 KB** and
/// start-major to **85.00 KB**, a 1.25× difference, and the mechanism is visible
/// in the slot sizes -- start-major opens a fourth MLP-width slot because a
/// 4096 B value takes a fresh slot first and a 21504 B value later first-fits
/// into it and widens it 5.25×. Luminal hit the same mechanism at 27× on ~100
/// slots; LFM2 decode has 5 slots and two size classes, which bounds the damage
/// at about one extra wide slot.
pub fn plan_from_sizes(sizes: &[usize], last_use: &[usize], layout: ArenaLayout) -> StepPlan {
    let first_use: Vec<usize> = (0..sizes.len()).collect();
    // A fixture's `first_use` is a liveness coordinate, not an observed
    // dispatch position, so the plan carries no positions for the drift guard
    // to compare against.
    plan_from_intervals(
        sizes,
        &first_use,
        last_use,
        &vec![false; sizes.len()],
        layout,
    )
    .without_alloc_positions()
}

/// As [`plan_from_sizes`], but with explicit liveness intervals and exclusions.
///
/// An interval is `first_use[i] ..= last_use[i]`, in **dispatch positions**.
/// That coordinate matters: liveness has to be measured on the clock that orders
/// execution, and for the GPU that is the dispatch stream, not the order in
/// which the CPU happened to drop handles (see [`ArenaRecorder::record_bind`]).
///
/// An excluded ordinal keeps its position in the sequence and is marked so that
/// [`Arena::acquire`] declines it, which is how session state stays out (§9.1).
/// Removing it from the sequence instead would renumber every later allocation
/// and break the positional keying the whole plan rests on.
pub fn plan_from_intervals(
    sizes: &[usize],
    first_use: &[usize],
    last_use: &[usize],
    excluded: &[bool],
    layout: ArenaLayout,
) -> StepPlan {
    assert_eq!(
        sizes.len(),
        last_use.len(),
        "every allocation needs a last-use position"
    );
    assert_eq!(
        sizes.len(),
        first_use.len(),
        "every allocation needs a first-use position"
    );
    assert_eq!(
        sizes.len(),
        excluded.len(),
        "every allocation needs an exclusion flag"
    );

    if layout == ArenaLayout::NonAliasing {
        // The §9.3 reference: one slot per value, so no two values can share
        // bytes whatever the liveness intervals say. Deliberately not packed --
        // its whole purpose is to be the layout that cannot alias.
        let mut slots = Vec::with_capacity(sizes.len());
        let mut by_ordinal = Vec::with_capacity(sizes.len());
        let mut cursor = 0usize;
        for (i, &size) in sizes.iter().enumerate() {
            if excluded[i] {
                // Session state is out of the arena under either layout, so the
                // two layouts differ *only* in whether activations share bytes.
                // That is what makes the comparison between them a test of the
                // offset arithmetic rather than of two different populations.
                by_ordinal.push(None);
                continue;
            }
            slots.push(Slot {
                offset: cursor,
                size,
            });
            by_ordinal.push(Some((slots.len() - 1, size)));
            cursor = align_up(cursor + size, ARENA_ALIGNMENT);
        }
        return StepPlan::new(slots, by_ordinal).with_alloc_positions(first_use.to_vec());
    }

    // Size-major first-fit. Each slot holds values whose intervals are pairwise
    // disjoint; a slot widens to its largest occupant. Excluded ordinals are not
    // placed at all -- they take the pool path.
    let mut order: Vec<usize> = (0..sizes.len()).filter(|&i| !excluded[i]).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(sizes[i]),
            first_use[i],
            std::cmp::Reverse(last_use[i].saturating_sub(first_use[i])),
            i,
        )
    });

    // Per slot: the intervals already placed in it, and its current size.
    let mut slot_intervals: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut slot_sizes: Vec<usize> = Vec::new();
    let mut assigned = vec![usize::MAX; sizes.len()];

    for &i in &order {
        let (start, end) = (first_use[i], last_use[i]);
        let mut placed = None;
        for (si, intervals) in slot_intervals.iter_mut().enumerate() {
            // Overlap is half-open in the direction this problem needs: a value
            // dies after its last read, so a value first written at exactly
            // another's last use may not share its bytes -- the producing
            // dispatch of the second reads the first.
            let clash = intervals.iter().any(|&(s, e)| !(end < s || e < start));
            if clash {
                continue;
            }
            intervals.push((start, end));
            slot_sizes[si] = slot_sizes[si].max(sizes[i]);
            placed = Some(si);
            break;
        }
        let si = match placed {
            Some(si) => si,
            None => {
                slot_intervals.push(vec![(start, end)]);
                slot_sizes.push(sizes[i]);
                slot_intervals.len() - 1
            }
        };
        assigned[i] = si;
    }

    // Slot offsets are assigned only once every size is final, since a slot
    // widens after it is opened.
    let mut slots = Vec::with_capacity(slot_sizes.len());
    let mut cursor = 0usize;
    for &size in &slot_sizes {
        slots.push(Slot {
            offset: cursor,
            size,
        });
        cursor = align_up(cursor + size, ARENA_ALIGNMENT);
    }

    let by_ordinal = assigned
        .iter()
        .enumerate()
        .map(|(i, &slot)| {
            if slot == usize::MAX {
                // Excluded: session state, or an ordinal the plan does not
                // cover. `None` makes `acquire` decline it, so it allocates
                // from the pool exactly as it did before.
                None
            } else {
                Some((slot, sizes[i]))
            }
        })
        .collect();

    StepPlan::new(slots, by_ordinal).with_alloc_positions(first_use.to_vec())
}

/// Round `x` up to a multiple of `a`.
///
/// Shared with [`scratch`](super::scratch) rather than copied into it. A second
/// spelling of the rounding rule would be the hand-sync §8.1b exists to remove,
/// one scale down -- and it is the function #70 showed a parity test can delete
/// without any LFM2-shaped fixture noticing (§9.2c), so two copies would be two
/// places for that to happen unobserved.
pub(crate) fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Observes one decode step and derives the plan from it.
///
/// # Why a recorder rather than a table
///
/// The plan needs each activation's size and the interval it is live over.
/// Candle is eager (§11.1a), so nothing declares that up front -- but the
/// dispatch sequence *is* stable (§11.1a.1: kernel name varies at 0 of 674
/// positions), so one observed step describes every later one. That is the same
/// record-then-replay shape §11.1a calls "the honest first version", applied to
/// allocation rather than to dispatch.
///
/// # Liveness is keyed on the value, never on the buffer
///
/// §9.2c, and this is where the 52× error lives if it is going to. A value here
/// is **one allocation event** -- one `record_alloc`, ending at the matching
/// `record_free`. When the pool hands the same `MTLBuffer` back for a later,
/// unrelated value, that is a *new* value with its own interval, because it is a
/// new allocation event. Nothing in this recorder consults buffer identity at
/// all, which is why it cannot make the mistake: in #68's trace `buf#328` is
/// written and consumed 13 times per token, and keying on it would merge those
/// into one 3.55 MB interval against the true 68 KB.
///
/// The generation counter is implicit in the ordinal: allocation N and
/// allocation M are different values whenever `N != M`, whatever buffer either
/// one happened to get.
#[derive(Debug, Default)]
pub struct ArenaRecorder {
    /// Size of each allocation, in ordinal order.
    sizes: Vec<usize>,
    /// Dispatch position each value was allocated at.
    first_bind: Vec<usize>,
    /// The previous step's intervals, kept so a plan can be built for the
    /// previous step once this one has said which of its values survived.
    previous_first_bind: Vec<usize>,
    previous_last_bind: Vec<usize>,
    /// Dispatch position each value was last bound at -- **where its interval
    /// ends**, because binding is the clock the GPU orders execution by.
    last_bind: Vec<usize>,
    /// Which value currently owns each buffer address.
    ///
    /// Reassigned when a buffer is handed to a later value, which is how one
    /// pooled buffer serving many values yields many intervals rather than one
    /// merged one (§9.2c).
    by_addr: std::collections::HashMap<usize, usize>,
    /// Dispatches encoded so far this step -- the coordinate intervals live in.
    dispatch: usize,
    /// Allocations seen so far this step.
    ordinal: usize,
    /// Sizes seen at each ordinal on a *previous* step, when one was recorded.
    ///
    /// Empty until [`Self::next_step`] is called. Comparing two steps is what
    /// separates an activation from session state, and it has to be measured
    /// rather than assumed -- see [`Self::kv_dependent`].
    previous: Vec<usize>,
    /// Which value owned each buffer address at the end of the *previous* step.
    ///
    /// Carried across the step boundary so that a bind in this step against a
    /// buffer allocated in the last one can still be attributed. Without it the
    /// address is simply absent from `by_addr` and the bind is dropped, which is
    /// exactly how a value that outlives its step passes for an activation --
    /// see [`Self::outlives_step`].
    previous_by_addr: std::collections::HashMap<usize, usize>,
    /// Ordinals of the *previous* step still bound during this one.
    ///
    /// The direct measurement of "this value is session state" (§9.1): it is
    /// still being read a step after it was produced.
    survived: std::collections::HashMap<usize, usize>,
    /// Ordinals of the CURRENT step allocated under an [`EscapeGuard`].
    declared: std::collections::HashSet<usize>,
    /// The previous step's `declared`, kept for [`Self::plan_previous`].
    previous_declared: std::collections::HashSet<usize>,
}

impl ArenaRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Close the step just recorded and begin another, keeping the first step's
    /// sizes for comparison.
    ///
    /// Two steps are the minimum that can distinguish an activation from session
    /// state, which is why this exists rather than the recorder planning from
    /// one step. #68's planner needs `>= 2` decode steps for the same reason and
    /// raises rather than silently returning an empty set.
    pub fn next_step(&mut self) {
        self.previous = std::mem::take(&mut self.sizes);
        self.previous_first_bind = std::mem::take(&mut self.first_bind);
        self.previous_last_bind = std::mem::take(&mut self.last_bind);
        // The address map is *carried*, not cleared. A value produced last step
        // and read in this one is session state, and dropping the map is what
        // made that invisible -- the bind found no owner and was discarded, so
        // the interval stopped at the step boundary and the value looked like an
        // activation that died where it was produced. See `outlives_step`.
        self.previous_by_addr = std::mem::take(&mut self.by_addr);
        self.previous_declared = std::mem::take(&mut self.declared);
        self.survived.clear();
        self.dispatch = 0;
        self.ordinal = 0;
    }

    /// Ordinals whose allocation size differs between the two recorded steps.
    ///
    /// **This is how session state is kept out of the arena** (`DESIGN.md` §9.1,
    /// #68 finding 4), and it is detected empirically rather than by kernel
    /// name, because `copy2d_f16` serves both the KV reallocation and ordinary
    /// activation movement -- the name does not separate them, and the size
    /// growing with `kv_len` does.
    ///
    /// Measured on LFM2 decode, the excluded population is the KV cache and the
    /// `Tensor::cat` copies it moves through, whose values grow by exactly 128 B
    /// per token. A slot whose occupant grows cannot hold a fixed offset, so
    /// everything packed after it drifts: with them left in, 969 bindings fail
    /// the 674-position stability check.
    ///
    /// An ordinal present in one step and not the other counts as differing --
    /// the sequence itself moved there, so nothing about it is fixed.
    pub fn kv_dependent(&self) -> Vec<bool> {
        let n = self.sizes.len();
        (0..n)
            .map(|i| match self.previous.get(i) {
                Some(&prev) => prev != self.sizes[i],
                // No previous step to compare against: treated as varying, so
                // an unpaired recording excludes everything rather than
                // admitting session state by default. Failing toward the pool is
                // the safe direction -- it costs arena coverage, never
                // correctness.
                None => true,
            })
            .collect()
    }

    /// Ordinals whose value was still bound a step after it was allocated.
    ///
    /// **This is the second way session state gets in, and size cannot see it.**
    /// [`Self::kv_dependent`] separates activations from session state by the
    /// size moving with `kv_len`, which is right for the KV cache and blind to
    /// anything session-scoped at a *fixed* size. LFM2's conv state is exactly
    /// that: `[B, 2048, 3]` = 12288 B at every `kv_len` (§5.7), so it passes the
    /// size test and enters the arena.
    ///
    /// What it cannot pass is this one, because the property being tested is the
    /// definition §9.1 gives rather than a proxy for it: session state is state
    /// that lives across steps, so measure whether it is read in the next one.
    ///
    /// Measured on LFM2 decode (`measurements/issue-69-raw/trace-after-packed.txt`):
    ///
    /// ```text
    /// decode[1] pos 4 copy2d out @0x9edae5a40+0   <- conv-state cat destination
    /// decode[1] pos 5 copy2d out @0x9edae5a40+4
    /// decode[1] pos 6 bmul   in  @0x9edae5a40+0   <- consumed, same token
    /// decode[2] pos 4 copy2d in  @0x9edae5a40+2   <- read again, next token
    /// ```
    ///
    /// The interval a within-step recorder sees ends at pos 6; the true one runs
    /// ~670 dispatches further, into the following step. Packed on the short
    /// interval, slot 1 serves five values per layer and the conv state is
    /// overwritten by the `fast_sum` at pos 7 long before pos 4 of the next
    /// token reads it.
    ///
    /// Note what this is *not*. The `narrow` in §6.1's shuffle is a red herring
    /// for classification: `narrow` shares storage and shifts only the layout,
    /// so the tail read and the `cat` destination are genuinely separate values
    /// with disjoint intervals and the packer is right to unify them. The defect
    /// is that one of those values outlives the step, not that either is
    /// addressed at an offset.
    ///
    /// Ordinals are compared across steps positionally, which is sound for the
    /// same reason [`Self::kv_dependent`] compares sizes that way: the kernel
    /// sequence is byte-identical across tokens (§11.1a.1), so the Nth
    /// allocation of one step is the Nth of the next.
    pub fn outlives_step(&self) -> Vec<bool> {
        let n = self.sizes.len();
        (0..n).map(|i| self.survived.contains_key(&i)).collect()
    }

    /// Every ordinal the arena must decline: session state by either test.
    ///
    /// Two independent detectors, unioned, because they see different
    /// populations and neither subsumes the other -- size growth finds the KV
    /// cache, cross-step liveness finds the conv state.
    pub fn excluded_flags(&self) -> Vec<bool> {
        let size = self.kv_dependent();
        let cross = self.outlives_step();
        size.iter()
            .zip(cross.iter())
            .enumerate()
            // Declared in either recorded step: a declaration that moved
            // between two sightings of one shape fails toward the pool.
            .map(|(i, (a, b))| {
                *a || *b || self.declared.contains(&i) || self.previous_declared.contains(&i)
            })
            .collect()
    }

    /// Record an allocation of `size` bytes served by the buffer at `addr`.
    ///
    /// `addr` identifies the *buffer*, and the ordinal identifies the *value*.
    /// Both are needed and they are not the same thing (§9.2c): the pool hands
    /// one buffer to many unrelated values within a token, so binding is
    /// observed by address while liveness is keyed by allocation event. A later
    /// allocation reusing the same address opens a new value and takes the
    /// address over from the old one, which is what stops the two from merging.
    pub fn record_alloc(&mut self, addr: usize, size: usize) {
        let ordinal = self.ordinal;
        self.ordinal += 1;
        self.sizes.push(size);
        // A value is born at the dispatch about to be encoded and, until a bind
        // says otherwise, dies there too.
        self.first_bind.push(self.dispatch);
        self.last_bind.push(self.dispatch);
        if escaping() {
            self.declared.insert(ordinal);
        }
        // The address now names this value. Any earlier value that had it has
        // been reused, and its own binds were recorded while it held it.
        //
        // **The previous step's claim on this address is dropped, not just
        // shadowed.** `record_bind` consults `by_addr` first, so shadowing
        // covers every bind AFTER this allocation — but the pool recycles an
        // address into a value the recorder has not reached yet, and a bind
        // of that value BEFORE its `record_alloc` runs falls through to
        // `previous_by_addr` and is charged to the previous step's ordinal as
        // an escape that never happened
        // (`a_recycled_address_stops_naming_the_previous_steps_value`). An
        // address reused by a new object that never reaches `record_alloc`
        // is the same phantom by another path; `note_new_buffer` covers it.
        self.previous_by_addr.remove(&addr);
        self.by_addr.insert(addr, ordinal);
    }

    /// Record that a dispatch bound the buffer at `addr`.
    ///
    /// Called once per binding, in encode order, and it is what closes an
    /// interval -- see the type docs. The dispatch counter is the recorder's
    /// own, so it advances with [`Self::record_dispatch`].
    ///
    /// A bind that matches no value of *this* step is checked against the
    /// previous one before being discarded. That is the whole of the
    /// cross-step detection: an address still being read a step after it was
    /// allocated names a value that outlived its step, and §9.1 calls such a
    /// value session state.
    pub fn record_bind(&mut self, addr: usize) {
        if let Some(&ordinal) = self.by_addr.get(&addr) {
            self.last_bind[ordinal] = self.dispatch;
        } else if let Some(&ordinal) = self.previous_by_addr.get(&addr) {
            // The first bind's position in THIS step, for naming the escape.
            self.survived.entry(ordinal).or_insert(self.dispatch);
        }
    }

    /// Forget the value at `addr`: something that is NOT part of the recorded
    /// sequence now occupies it — an allocation made between two passes (a
    /// sampler's ids, a staged input, a cat of the step's rows), or a new
    /// buffer object created at an address a destroyed one had
    /// ([`note_new_buffer`]). It takes no ordinal, and the address must stop
    /// naming whatever held it, or a bind of the NEW occupant is read as a
    /// bind of the old value — reported as escaping its pass when the value
    /// was the previous pass's: a phantom escape, and a hole punched in the
    /// plan for a value that never outlived anything.
    pub fn forget_addr(&mut self, addr: usize) {
        self.by_addr.remove(&addr);
        self.previous_by_addr.remove(&addr);
    }

    /// Advance the dispatch counter. Called once per dispatch, in encode order.
    pub fn record_dispatch(&mut self) {
        self.dispatch += 1;
    }

    /// Allocations recorded so far.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Build a plan from what was observed, excluding session state.
    ///
    /// Session state is left out of the arena entirely and takes the pool path,
    /// by **both** tests -- [`Self::kv_dependent`] for values whose size moves
    /// with `kv_len`, and [`Self::outlives_step`] for values still bound a step
    /// after they were produced. An excluded ordinal keeps its place in the
    /// sequence, so excluding one does not shift the slot every later
    /// allocation resolves to.
    ///
    /// A value never bound again is treated as dying where it was produced,
    /// which is only safe *because* the cross-step test runs: a value whose
    /// last reader is in the next step would otherwise look exactly like one
    /// that was never read at all.
    pub fn plan(&self, layout: ArenaLayout) -> StepPlan {
        plan_from_intervals(
            &self.sizes,
            &self.first_bind,
            &self.last_bind,
            &self.excluded_flags(),
            layout,
        )
    }

    /// A plan for the PREVIOUS step, from its own intervals and the values of
    /// it this step read (`DESIGN.md` §9.2aa). The two steps need not share a
    /// shape: only the cross-step liveness test applies, since a value of the
    /// previous step that this step bound is session state whatever this
    /// step computes, and a size comparison across different shapes says
    /// nothing. `None` before `next_step` has been called.
    pub fn plan_previous(&self, layout: ArenaLayout) -> Option<StepPlan> {
        let n = self.previous.len();
        if n == 0 || self.previous_first_bind.len() != n || self.previous_last_bind.len() != n {
            return None;
        }
        let excluded: Vec<bool> = (0..n)
            .map(|i| self.survived.contains_key(&i) || self.previous_declared.contains(&i))
            .collect();
        Some(plan_from_intervals(
            &self.previous,
            &self.previous_first_bind,
            &self.previous_last_bind,
            &excluded,
            layout,
        ))
    }

    /// Ordinals of the previous step this step bound — the values that
    /// escaped it — and how many the previous step allocated.
    pub fn previous_escapes(&self) -> (usize, usize) {
        (self.survived.len(), self.previous.len())
    }

    /// Ordinals of the previous step this step bound that no [`EscapeGuard`]
    /// had declared — the escapes the plan cache counts as holes.
    pub fn previous_undeclared(&self) -> usize {
        self.previous_undeclared_ordinals().len()
    }

    /// The same escapes as `(ordinal, bytes, allocated at dispatch, first
    /// read at dispatch of the following step)`, sorted by ordinal — what
    /// names an escape when one shows up.
    pub fn previous_undeclared_ordinals(&self) -> Vec<(usize, usize, usize, usize)> {
        let mut v: Vec<(usize, usize, usize, usize)> = self
            .survived
            .iter()
            .filter(|(i, _)| !self.previous_declared.contains(i))
            .map(|(&i, &at)| {
                (
                    i,
                    self.previous.get(i).copied().unwrap_or(0),
                    self.previous_first_bind.get(i).copied().unwrap_or(0),
                    at,
                )
            })
            .collect();
        v.sort_unstable();
        v
    }

    /// The dispatch position ordinal `i` of the CURRENT step was last bound
    /// at, for tests that check a bind was observed.
    pub fn last_bind_of(&self, i: usize) -> Option<usize> {
        self.last_bind.get(i).copied()
    }

    /// Dispatches this step has counted so far — the position the next bind
    /// would be recorded at.
    pub fn dispatch_position(&self) -> usize {
        self.dispatch
    }

    /// How many ordinals were excluded as session state, and how many recorded.
    pub fn excluded(&self) -> (usize, usize) {
        let flags = self.excluded_flags();
        (flags.iter().filter(|&&x| x).count(), flags.len())
    }

    /// Excluded ordinals split by which test caught them: `(size, cross_step)`.
    ///
    /// Reported separately because the two populations are the evidence that
    /// neither test subsumes the other -- a run where `cross_step` is 0 would
    /// mean this detector is not earning its place, and one where it is nonzero
    /// names values the size test admitted.
    pub fn excluded_by_test(&self) -> (usize, usize) {
        let size = self.kv_dependent();
        let cross = self.outlives_step();
        (
            size.iter().filter(|&&x| x).count(),
            cross.iter().filter(|&&x| x).count(),
        )
    }

    /// Reservations for the session region: `(ordinal, bytes at capacity)`
    /// (#47, `DESIGN.md` §9.2x).
    ///
    /// The population is exactly the ordinals [`Self::kv_dependent`] excluded —
    /// the ones whose allocation size **moved between the two recorded steps** —
    /// and the reservation is that growth extrapolated to `capacity_tokens`.
    ///
    /// # The stride is measured, never assumed
    ///
    /// Two recorded steps one token apart give `stride = this - previous` at
    /// each ordinal directly. Under `KvAppend::Cat` that is
    /// `b × kv_heads × head_dim × dtype` per token, but this does not compute it
    /// from the geometry: the recorder already holds both figures, and a derived
    /// constant would be a second source of truth to drift (§8.1b's argument for
    /// a checked registry, applied to a size).
    ///
    /// `tokens_recorded` is how many tokens of growth the observed step already
    /// contains — the `kv_len` at recording time — so the reservation covers the
    /// tenant from where it is now to `capacity_tokens`:
    ///
    /// ```text
    /// reserved = this_step_size + stride × (capacity_tokens − tokens_recorded)
    /// ```
    ///
    /// # Which ordinals are excluded, and why only one of the two tests
    ///
    /// [`Self::excluded_flags`] unions two detectors, and only `kv_dependent`'s
    /// population belongs here. The other — [`Self::outlives_step`] — catches
    /// state that lives across steps at a **fixed** size, LFM2's conv state
    /// being the worked case at 12288 B. A fixed-size tenant does not grow, so
    /// it has no stride to extrapolate and reserving for it would hand the
    /// region a tenant it cannot size; those keep falling to the pool, which is
    /// what they do today.
    ///
    /// An ordinal that **shrank** between the two steps is skipped rather than
    /// reserved: a negative stride means the sequence drifted rather than grew,
    /// and extrapolating it would produce a reservation smaller than the tenant
    /// already needs — a decline every token, which costs the lookup and buys
    /// nothing.
    pub fn session_reservations(
        &self,
        tokens_recorded: usize,
        capacity_tokens: usize,
    ) -> Vec<(usize, usize)> {
        let growing = self.kv_dependent();
        let fixed = self.outlives_step();
        let remaining = capacity_tokens.saturating_sub(tokens_recorded);
        (0..self.sizes.len())
            .filter(|&i| growing[i] && !fixed[i])
            .filter_map(|i| {
                let now = self.sizes[i];
                // `kv_dependent` marks an ordinal absent from the previous step
                // as varying, so `previous` may be short here. No previous size
                // means no measurable stride, and a guessed one is exactly what
                // this doc comment refuses.
                let prev = *self.previous.get(i)?;
                let stride = now.checked_sub(prev)?;
                if stride == 0 {
                    return None;
                }
                Some((i, now + stride * remaining))
            })
            .collect()
    }

    /// Forget everything, including the previous step.
    pub fn reset(&mut self) {
        self.sizes.clear();
        self.first_bind.clear();
        self.last_bind.clear();
        self.by_addr.clear();
        self.previous.clear();
        self.previous_by_addr.clear();
        self.survived.clear();
        self.dispatch = 0;
        self.ordinal = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Imported here rather than at module scope: the arena itself must not be
    // able to reach a pool (issue #489), and the tests are the only thing that
    // legitimately holds one.
    use crate::kernels::metal::BufferPool;
    use crate::kernels::metal::Device;

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
    }

    fn arena_of(sizes: &[usize], last_use: &[usize], layout: ArenaLayout) -> (Arena, BufferPool) {
        let dev = device();
        let plan = plan_from_sizes(sizes, last_use, layout);
        let base = dev
            .new_buffer(plan.arena_bytes().max(1), crate::kernels::RESOURCE_OPTIONS)
            .expect("arena allocation");
        let pool = BufferPool::new();
        let arena = Arena::new(base, plan, layout).expect("plan is valid");
        (arena, pool)
    }

    /// Two values whose lifetimes do not overlap share a slot; two that do
    /// overlap do not. This is the packing property the 68 KB comes from.
    #[test]
    fn disjoint_values_share_a_slot_and_overlapping_ones_do_not() {
        // 0 and 2 are disjoint (0 dies at 1); 1 overlaps both.
        let sizes = [4096, 4096, 4096];
        let last_use = [1, 2, 2];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        let s: Vec<usize> = (0..plan.allocations())
            .map(|i| plan.slot_of(i).expect("ordinal is served"))
            .collect();
        assert_ne!(s[0], s[1], "overlapping values shared a slot");
        assert_ne!(s[1], s[2], "overlapping values shared a slot");
        assert_eq!(s[0], s[2], "disjoint values did not share a slot");
        assert_eq!(plan.slots().len(), 2);
    }

    /// The §9.3 reference layout gives every value its own slot, so aliasing is
    /// impossible by construction. It must be strictly larger than the packed
    /// one on any input where packing does anything -- if they ever tie, the
    /// reference has stopped being a reference.
    #[test]
    fn non_aliasing_reference_never_shares_bytes() {
        let sizes = [4096, 4096, 4096];
        let last_use = [1, 2, 2];
        let packed = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        let reference = plan_from_sizes(&sizes, &last_use, ArenaLayout::NonAliasing);

        assert_eq!(reference.slots().len(), 3, "a value shared a slot");
        let s: Vec<usize> = (0..reference.allocations())
            .map(|i| reference.slot_of(i).expect("ordinal is served"))
            .collect();
        assert_eq!(s, vec![0, 1, 2]);
        assert!(
            reference.arena_bytes() > packed.arena_bytes(),
            "reference {} is not larger than packed {}",
            reference.arena_bytes(),
            packed.arena_bytes()
        );
        reference.check_disjoint().expect("reference slots overlap");
    }

    /// Size-major beats start-major on the shape that produces #68's 1.25×: a
    /// small value opening a slot that a large one later widens.
    ///
    /// This is issue #43's mechanism, reproduced at the smallest size that shows
    /// it. Pinned because the sort key is three lines and reverting it to start
    /// order would look like a simplification.
    #[test]
    fn size_major_does_not_widen_a_slot_a_small_value_opened() {
        // A small value, then a large one whose interval is disjoint from it.
        // Start order places the small one first, so the large one first-fits
        // into its slot and widens it 5.25x; size-major places the large one
        // first and the small one fills in behind it.
        let sizes = [4096, 21504, 21504];
        let last_use = [0, 2, 2];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);

        // The 4096 must not have widened a slot to 21504 that then needs a
        // second 21504 beside it: two MLP-width slots is right, three is the
        // start-major pathology.
        let mlp_slots = plan.slots().iter().filter(|s| s.size == 21504).count();
        assert_eq!(
            mlp_slots,
            2,
            "expected 2 MLP-width slots, got {mlp_slots}: {:?}",
            plan.slots()
        );
    }

    /// Every slot is 128 B aligned, and the alignment is the cache line.
    ///
    /// §9.2 says lowering this to 64 would introduce false sharing between
    /// adjacent slots without the Metal-dtype reasoning revealing it, so the
    /// constant is asserted rather than left to a comment.
    #[test]
    fn slots_are_cache_line_aligned() {
        assert_eq!(ARENA_ALIGNMENT, 128);
        let sizes = [4096, 21504, 1024, 32];
        let last_use = [3, 3, 3, 3];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        for s in plan.slots() {
            assert_eq!(
                s.offset % ARENA_ALIGNMENT,
                0,
                "slot at {} is not {ARENA_ALIGNMENT} B aligned",
                s.offset
            );
        }
        plan.check_disjoint().expect("slots overlap");
    }

    /// The check that guards the aliasing invariant must be able to fail.
    /// `CONTRIBUTING.md` §3.1: a test that cannot fail is not a test, and this
    /// one guards the case §9.3 says has no safety net.
    #[test]
    fn overlap_check_catches_overlapping_slots() {
        let good = StepPlan::new(
            vec![
                Slot {
                    offset: 0,
                    size: 128,
                },
                Slot {
                    offset: 128,
                    size: 128,
                },
            ],
            vec![Some((0, 128)), Some((1, 128))],
        );
        good.check_disjoint().expect("disjoint slots rejected");

        // Mutation: the second slot starts inside the first.
        let bad = StepPlan::new(
            vec![
                Slot {
                    offset: 0,
                    size: 256,
                },
                Slot {
                    offset: 128,
                    size: 128,
                },
            ],
            vec![Some((0, 256)), Some((1, 128))],
        );
        let err = bad
            .check_disjoint()
            .expect_err("overlapping slots were accepted");
        assert!(err.contains("overlap"), "unexpected error: {err}");

        // Mutation: a slot that is not cache-line aligned.
        let misaligned = StepPlan::new(
            vec![Slot {
                offset: 64,
                size: 128,
            }],
            vec![Some((0, 128))],
        );
        let err = misaligned
            .check_disjoint()
            .expect_err("a 64 B aligned slot was accepted");
        assert!(err.contains("aligned"), "unexpected error: {err}");
    }

    /// The same ordinal gets the same slot on every step. This is the property
    /// the whole issue exists for -- 674 varying buffer identities becoming 0 --
    /// asserted at the level where it is decided.
    #[test]
    fn the_same_ordinal_gets_the_same_slot_every_step() {
        let sizes = [4096, 21504, 4096];
        let last_use = [2, 2, 2];
        let (arena, _pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        let mut per_step = Vec::new();
        for _ in 0..8 {
            arena.begin_step();
            let step: Vec<usize> = sizes
                .iter()
                .map(|&s| {
                    let b = arena.acquire(s).expect("plan covers this ordinal");
                    // Identity of the region, which is what a dispatch binds.
                    (b.base_offset(), b.length()).0
                })
                .collect();
            arena.end_step();
            per_step.push(step);
        }

        let first = &per_step[0];
        for (i, step) in per_step.iter().enumerate() {
            assert_eq!(step, first, "step {i} bound different offsets: {step:?}");
        }
    }

    /// A request that arrives at a different dispatch position than its
    /// ordinal was recorded at is the sequence having drifted: it is declined,
    /// the step is served no further, and the drift is reported once.
    ///
    /// The size gate cannot see this case -- the sizes here are identical --
    /// and it is the case the model produces: attention routing to its
    /// two-pass kernel at `kv_len` 1024 inserts a dispatch and three
    /// allocations at every attention layer, so every later ordinal names a
    /// different value than the plan packed a slot for.
    #[test]
    fn a_request_at_the_wrong_dispatch_position_is_a_drift_and_ends_the_step() {
        let dev = device();
        let sizes = [4096, 4096, 4096];
        let first_use = [0, 1, 2];
        let last_use = [1, 2, 3];
        let plan = plan_from_intervals(
            &sizes,
            &first_use,
            &last_use,
            &[false; 3],
            ArenaLayout::Packed,
        );
        assert_eq!(
            plan.alloc_position(1),
            Some(1),
            "a recording carries its positions"
        );
        let base = dev
            .new_buffer(plan.arena_bytes().max(1), crate::kernels::RESOURCE_OPTIONS)
            .expect("arena allocation");
        let arena = Arena::new(base, plan, ArenaLayout::Packed).expect("plan is valid");

        // The recorded sequence: allocate, dispatch, allocate, dispatch, ...
        arena.begin_step();
        for &s in &sizes {
            assert!(
                arena.acquire(s).is_some(),
                "the recorded sequence is served"
            );
            note_dispatch();
        }
        arena.end_step();
        assert!(!arena.take_drifted(), "a matching step does not drift");
        assert_eq!(arena.counters().hits, 3);

        // The drifted sequence: one dispatch inserted before ordinal 1.
        arena.begin_step();
        assert!(
            arena.acquire(sizes[0]).is_some(),
            "ordinal 0 is where it was"
        );
        note_dispatch();
        note_dispatch();
        let offer = arena.acquire_attributed(sizes[1]);
        assert!(offer.served.is_none());
        assert_eq!(offer.decline, Some(ArenaDecline::Drift));
        note_dispatch();
        let after = arena.acquire_attributed(sizes[2]);
        assert!(after.served.is_none(), "nothing is served after a drift");
        assert_eq!(after.decline, Some(ArenaDecline::Inactive));
        arena.end_step();
        let c = arena.counters();
        assert_eq!(
            c.declined_drift, 1,
            "the drift is counted once, at the position it was seen"
        );
        assert_eq!(c.hits, 4);
        assert!(
            arena.take_drifted(),
            "the loop learns of it at the step's close"
        );
        assert!(!arena.take_drifted(), "and it is cleared by being taken");
    }

    /// A slot handle is never dropped while the arena lives, so it never enters
    /// a free list. That is what makes the arena layer 3 (§9.2a) rather than a
    /// change to how the pool decides a buffer is free.
    #[test]
    fn arena_buffers_never_enter_the_free_list() {
        let sizes = [4096, 21504];
        let last_use = [1, 1];
        let (arena, pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        for _ in 0..4 {
            arena.begin_step();
            for &s in &sizes {
                let b = arena.acquire(s).expect("plan covers this ordinal");
                // The caller drops its handle, exactly as a tensor would.
                drop(b);
            }
            arena.end_step();
        }

        assert_eq!(
            pool.counters().releases,
            0,
            "an arena buffer was released to the pool"
        );
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "an arena buffer reached a free list"
        );
    }

    /// Session state must not enter the arena (§9.1, #68 finding 4). A value
    /// that grows every token does not match its ordinal's planned size, so it
    /// is declined and falls through to the pool.
    ///
    /// The failure this prevents is not a size overflow but a *drift*: a slot
    /// whose occupant grows cannot hold a fixed offset, and with KV left in,
    /// 969 bindings fail the 674-stability check.
    #[test]
    fn arena_size_mismatch_declines_rather_than_drifting() {
        let sizes = [4096, 21504];
        let last_use = [1, 1];
        let (arena, _pool) = arena_of(&sizes, &last_use, ArenaLayout::Packed);

        for token in 0..4usize {
            arena.begin_step();
            assert!(arena.acquire(4096).is_some(), "fixed-size value declined");
            // A KV-shaped allocation: 128 B larger every token.
            assert!(
                arena.acquire(21504 + 128 * (token + 1)).is_none(),
                "a growing value was given a fixed slot"
            );
            arena.end_step();
        }

        let c = arena.counters();
        assert_eq!(c.declined_size, 4, "the size gate never engaged");
        assert_eq!(c.hits, 4);
    }

    /// An address the pool RECYCLES stops naming the previous step's value.
    ///
    /// The cross-step detector reads a bind that matches nothing in this step
    /// against the previous step's address map, and calls a match an escape.
    /// That is right for a value still being read — and wrong for an address
    /// the pool has handed to a NEW value whose own `record_alloc` has not
    /// run yet: the new value's first bind is then charged to the old
    /// ordinal. Measured on the real model, that phantom count moved between
    /// identical runs (8 escapes, then 7, different ordinals), which is the
    /// signature — a real escape is the same ordinal every run.
    #[test]
    fn a_recycled_address_stops_naming_the_previous_steps_value() {
        let mut rec = ArenaRecorder::new();
        // Step 1: two values, neither read again.
        rec.record_alloc(0x1000, 4096);
        rec.record_alloc(0x2000, 4096);
        rec.next_step();

        // Step 2: the pool recycles 0x1000 into a new value. Its `record_alloc`
        // runs, and a LATER bind of it must belong to this step.
        rec.record_alloc(0x1000, 4096);
        rec.record_dispatch();
        rec.record_bind(0x1000);
        assert_eq!(
            rec.previous_escapes(),
            (0, 2),
            "a recycled address must not name the previous step's value"
        );

        // And the detector still works: 0x2000 was NOT reallocated, so a bind
        // of it here is a genuine cross-step read.
        rec.record_bind(0x2000);
        assert_eq!(
            rec.previous_escapes(),
            (1, 2),
            "a value still read a step later IS an escape; the fix must not \
             blind the detector"
        );
    }

    /// An allocation made under an [`EscapeGuard`] is DECLARED to outlive its
    /// pass (§9.2aa): the recorder excludes its ordinal from the plan it
    /// builds — for the current step and, through `plan_previous`, for the
    /// step just closed — and an installed arena declines it even when the
    /// plan has a slot for that ordinal.
    #[test]
    fn an_allocation_under_an_escape_guard_is_declared_and_never_served() {
        let mut rec = ArenaRecorder::new();
        rec.record_alloc(0x1000, 4096);
        {
            let _g = EscapeGuard::new();
            rec.record_alloc(0x2000, 4096);
        }
        rec.record_alloc(0x3000, 4096);
        assert!(!escaping(), "the guard outlived its scope");
        assert_eq!(
            rec.excluded_flags(),
            vec![true, true, true],
            "one step excludes all"
        );
        rec.next_step();
        rec.record_alloc(0x1000, 4096);
        rec.record_alloc(0x2000, 4096);
        rec.record_alloc(0x3000, 4096);
        // Across two steps of the same shape only the declared one is out.
        assert_eq!(rec.excluded_flags(), vec![false, true, false]);
        let prev = rec
            .plan_previous(ArenaLayout::Packed)
            .expect("two steps recorded");
        assert_eq!(prev.request_sizes(), vec![4096, 0, 4096], "{prev:?}");
        assert_eq!(rec.previous_undeclared(), 0);

        let (arena, _pool) = arena_of(&[4096, 4096], &[1, 1], ArenaLayout::Packed);
        arena.begin_step();
        assert!(arena.acquire(4096).is_some());
        {
            let _g = EscapeGuard::new();
            let offer = arena.acquire_attributed(4096);
            assert!(offer.served.is_none(), "a declared escape was given a slot");
            assert_eq!(offer.decline, Some(ArenaDecline::Escaped));
            assert_eq!(offer.ordinal, Some(1), "the ordinal is still consumed");
        }
        arena.end_step();
        assert_eq!(arena.counters().declined_escaped, 1);
        assert_eq!(arena.counters().hits, 1);
    }

    /// Outside a decode step the arena declines everything, so prefill and
    /// model setup allocate from the pool exactly as before.
    #[test]
    fn nothing_is_served_outside_a_step() {
        let (arena, _pool) = arena_of(&[4096], &[0], ArenaLayout::Packed);
        assert!(arena.acquire(4096).is_none(), "served outside a step");
        arena.begin_step();
        assert!(arena.acquire(4096).is_some());
        arena.end_step();
        assert!(arena.acquire(4096).is_none(), "served after the step ended");
    }

    /// A step longer than the plan falls through to the pool rather than
    /// wrapping around and re-serving slot 0, which would alias two live values.
    #[test]
    fn allocations_past_the_plan_are_declined() {
        let (arena, _pool) = arena_of(&[4096], &[0], ArenaLayout::Packed);
        arena.begin_step();
        assert!(arena.acquire(4096).is_some());
        assert!(arena.acquire(4096).is_none(), "wrapped past the plan");
        assert_eq!(arena.counters().declined_exhausted, 1);
    }

    /// A view addresses its own region: its length is the slot's, not the
    /// arena's, and its CPU pointer starts at the slot.
    ///
    /// Both halves are silent-corruption guards. `fill_buffer` takes
    /// `buffer.length()` as a range, so a view reporting the parent's length
    /// would blit across every slot after it; `read_to_vec` takes `contents()`
    /// with no offset, so a view reporting the parent's pointer would read the
    /// wrong slot's bytes.
    #[test]
    fn a_view_addresses_only_its_own_region() {
        let dev = device();
        let base = dev
            .new_buffer(4096, crate::kernels::RESOURCE_OPTIONS)
            .expect("allocation");
        let parent_ptr = base.contents() as usize;

        let view = base.view(128, 256);
        assert_eq!(view.length(), 256, "view reported the parent's length");
        assert_eq!(view.base_offset(), 128);
        assert_eq!(
            view.contents() as usize,
            parent_ptr + 128,
            "view's CPU pointer did not start at its slot"
        );

        // A view of a view composes rather than resetting.
        let inner = view.view(64, 64);
        assert_eq!(inner.base_offset(), 192);
        assert_eq!(inner.length(), 64);
        assert_eq!(inner.contents() as usize, parent_ptr + 192);

        // An ordinary buffer is unaffected: offset 0, full length.
        assert_eq!(base.base_offset(), 0);
        assert_eq!(base.length(), 4096);
    }

    /// **The trap #68 hit and self-caught** (§9.2c), pinned so this port cannot
    /// re-make it.
    ///
    /// One buffer reused for three unrelated values, one after another. Keyed on
    /// the *value* -- one allocation event each -- their intervals are disjoint
    /// and they need one slot. Keyed on the *buffer* they would merge into one
    /// interval spanning the whole step, which invents a lifetime no activation
    /// has; #68's first planner did that and reported 3.55 MB against the true
    /// 68 KB, a 52x overstatement.
    ///
    /// The recorder cannot make the mistake because it never sees a buffer: the
    /// token identifies the allocation event. This test states that as a
    /// property rather than trusting the absence.
    ///
    /// Record `f` as two consecutive decode steps.
    ///
    /// Two steps are what the recorder needs to tell an activation from session
    /// state, and replaying identical steps is the fixed-shape case: nothing
    /// grows, so nothing is excluded. A test that recorded one step would get an
    /// empty plan, because an unpaired recording excludes everything rather than
    /// admitting session state by default.
    fn record_twice(f: impl Fn(&mut ArenaRecorder)) -> ArenaRecorder {
        let mut rec = ArenaRecorder::new();
        f(&mut rec);
        rec.next_step();
        f(&mut rec);
        rec
    }

    #[test]
    fn liveness_keys_on_the_value_not_on_the_recycled_buffer() {
        // Three values that happen to land in the same pooled buffer, each
        // fully consumed before the next is allocated -- the `buf#328` pattern,
        // which #68 observed 13 times per token.
        // Each is allocated, bound by the dispatch that produces it and the one
        // that consumes it, and then the buffer is handed to the next value --
        // the same address, a new allocation event.
        const ADDR: usize = 0x328;
        let rec = record_twice(|rec| {
            for _ in 0..3 {
                rec.record_alloc(ADDR, 21504);
                rec.record_bind(ADDR); // produced
                rec.record_dispatch();
                rec.record_bind(ADDR); // consumed
                rec.record_dispatch();
            }
        });

        let plan = rec.plan(ArenaLayout::Packed);
        assert_eq!(
            plan.slots().len(),
            1,
            "three sequentially-dead values did not share one slot: {:?}",
            plan.slots()
        );
        assert_eq!(
            plan.arena_bytes(),
            21504,
            "peak is {} B; keying on the buffer would inflate it",
            plan.arena_bytes()
        );
        // All three ordinals resolve to the same slot, which is the positional
        // stability the arena exists for.
        let slots: Vec<usize> = (0..plan.allocations())
            .map(|i| plan.slot_of(i).expect("ordinal is served"))
            .collect();
        assert_eq!(slots, vec![0, 0, 0]);
    }

    /// **A fixed-size value read in the next step is session state, and the
    /// size test cannot see it** (§9.1, §9.2c).
    ///
    /// This is LFM2's conv state, reduced to its shape. `[B, 2048, 3]` is
    /// 12288 B at every `kv_len` (§5.7), so it never grows and
    /// [`ArenaRecorder::kv_dependent`] admits it. It is produced by the §6.1
    /// shuffle, consumed once in the same token, and then read again by the
    /// *next* token's shuffle -- which is the pattern the committed trace shows:
    ///
    /// ```text
    /// decode[1] pos 4 copy2d out @0x9edae5a40+0
    /// decode[1] pos 6 bmul   in  @0x9edae5a40+0
    /// decode[2] pos 4 copy2d in  @0x9edae5a40+2   <- one token later
    /// ```
    ///
    /// Packed on the within-step interval its slot is reused immediately, and
    /// the value is gone before the next token reads it.
    ///
    /// The mutation this is proof against: drop the `previous_by_addr` carry in
    /// `next_step` and the cross-step bind matches nothing, `outlives_step` is
    /// all-false, and the conv-state stand-in packs into a shared slot.
    #[test]
    fn a_fixed_size_value_read_next_step_is_excluded() {
        const CONV: usize = 0xc0;
        const ACT: usize = 0xa0;
        const CONV_BYTES: usize = 12288;

        let mut rec = ArenaRecorder::new();
        let mut step = |rec: &mut ArenaRecorder| {
            // The shuffle reads *last* step's conv state first. On the opening
            // step there is none, which is why the read comes before the alloc.
            rec.record_bind(CONV);
            rec.record_dispatch();
            // This step's conv state, at a size that never moves.
            rec.record_alloc(CONV, CONV_BYTES);
            rec.record_bind(CONV);
            rec.record_dispatch();
            // Consumed within the token, as `bmul` does at pos 6.
            rec.record_bind(CONV);
            rec.record_dispatch();
            // An ordinary activation of the same size, to prove the exclusion
            // is about liveness and not about the size.
            rec.record_alloc(ACT, CONV_BYTES);
            rec.record_bind(ACT);
            rec.record_dispatch();
            rec.record_bind(ACT);
            rec.record_dispatch();
        };
        step(&mut rec);
        rec.next_step();
        step(&mut rec);

        assert_eq!(
            rec.kv_dependent(),
            vec![false, false],
            "the size test sees nothing here -- that is the point"
        );
        assert_eq!(
            rec.outlives_step(),
            vec![true, false],
            "the conv-state stand-in was not detected as living across the step"
        );

        let plan = rec.plan(ArenaLayout::Packed);
        assert!(
            plan.slot_of(0).is_none(),
            "session state entered the arena; it must fall through to the pool"
        );
        assert!(
            plan.slot_of(1).is_some(),
            "the ordinary activation was excluded too -- the test is over-broad"
        );
        assert_eq!(
            rec.excluded_by_test(),
            (0, 1),
            "exclusion was not attributed to the cross-step test"
        );
    }

    /// A growing tenant's reservation is its measured stride extrapolated to
    /// capacity, by absolute byte count (#47, §9.2x).
    ///
    /// **Kills**: any change to `session_reservations`' arithmetic — dropping
    /// the `tokens_recorded` subtraction, using `prev` instead of `now` as the
    /// base, or multiplying by `capacity` rather than by the remaining tokens.
    /// A test asserting only *"the reservation is at least the current size"*
    /// survives all three (§3.1b rule 1: an absolute reference, not a shape).
    #[test]
    fn a_growing_tenant_reserves_its_measured_stride_to_capacity() {
        const KV: usize = 0x11;
        const ACT: usize = 0x22;
        // 128 B of growth per token: the `Tensor::cat` stride under `Cat`.
        const STRIDE: usize = 128;

        let mut rec = ArenaRecorder::new();
        let mut step = |rec: &mut ArenaRecorder, kv_bytes: usize| {
            rec.record_alloc(KV, kv_bytes);
            rec.record_bind(KV);
            rec.record_dispatch();
            rec.record_alloc(ACT, 4096);
            rec.record_bind(ACT);
            rec.record_dispatch();
        };
        // Two steps one token apart: 40 tokens of KV, then 41.
        step(&mut rec, STRIDE * 40);
        rec.next_step();
        step(&mut rec, STRIDE * 41);

        // Recorded at 41 tokens, reserving to a capacity of 100.
        let res = rec.session_reservations(41, 100);
        assert_eq!(
            res,
            vec![(0, STRIDE * 41 + STRIDE * 59)],
            "the reservation must be this step's size plus the stride over the \
             remaining 59 tokens, and must not include the activation"
        );
        // Which is the closed form, stated as the number a reader can check:
        assert_eq!(res[0].1, 12800, "100 tokens x 128 B");
    }

    /// A fixed-size tenant that outlives its step gets no reservation.
    ///
    /// It is session state and the arena declines it, but it has no stride to
    /// extrapolate — reserving for it would hand the region a tenant it cannot
    /// size. LFM2's conv state is the worked case at a flat 12288 B.
    ///
    /// **Kills**: dropping the `!fixed[i]` filter, which would reserve for the
    /// conv state using a stride of zero and produce a tenant whose reservation
    /// never grows.
    #[test]
    fn a_fixed_size_session_tenant_gets_no_reservation() {
        const CONV: usize = 0xc0;
        let mut rec = ArenaRecorder::new();
        let mut step = |rec: &mut ArenaRecorder| {
            rec.record_bind(CONV);
            rec.record_dispatch();
            rec.record_alloc(CONV, 12288);
            rec.record_bind(CONV);
            rec.record_dispatch();
        };
        step(&mut rec);
        rec.next_step();
        step(&mut rec);

        assert_eq!(
            rec.outlives_step(),
            vec![true],
            "the fixture must produce a cross-step tenant, or this asserts nothing"
        );
        assert!(
            rec.session_reservations(41, 100).is_empty(),
            "a fixed-size session tenant must not be reserved: it has no stride"
        );
    }

    /// An ordinary activation — same size both steps — is never reserved.
    ///
    /// It belongs to the arena, and reserving for it in the region as well would
    /// mean two owned allocations for one value.
    #[test]
    fn a_stable_activation_is_never_reserved() {
        const ACT: usize = 0xa0;
        let mut rec = ArenaRecorder::new();
        let mut step = |rec: &mut ArenaRecorder| {
            rec.record_alloc(ACT, 21504);
            rec.record_bind(ACT);
            rec.record_dispatch();
        };
        step(&mut rec);
        rec.next_step();
        step(&mut rec);
        assert!(
            rec.session_reservations(41, 100).is_empty(),
            "a stable activation must stay the arena's, not the region's"
        );
    }

    /// The counterpart: carrying the address map across the step boundary must
    /// not make ordinary recycled activations look like survivors.
    ///
    /// The failure this guards against is one-sided and quiet. Over-detection
    /// costs arena coverage rather than correctness, so it breaks nothing
    /// visible and would not be found by the digests -- which is exactly the
    /// shape that needs a test rather than a run.
    ///
    /// **The over-detection is structurally impossible, and the test says so
    /// rather than the comment alone.** `record_bind` consults `by_addr` before
    /// the carried map, and `record_alloc` writes `by_addr` for every value, so
    /// a previous-step entry is shadowed the moment the address is handed on.
    /// An earlier draft added a `remove` in `record_alloc` to enforce that; it
    /// was deleted when no mutation of it could fail this test, which is §15.2
    /// #11 applied to a guard rather than to state.
    ///
    /// The pool recycles an address several times per token (§9.2c's `buf#328`,
    /// 13 times), so the recycled case here is the common one and the survivor
    /// beside it is the rare one. Asserting both in one test is what shows they
    /// are told apart, rather than everything being admitted or everything
    /// excluded.
    #[test]
    fn a_recycled_address_is_not_mistaken_for_a_survivor() {
        const ADDR: usize = 0xa0;
        const KEPT: usize = 0xb0;
        let mut rec = ArenaRecorder::new();
        let mut step = |rec: &mut ArenaRecorder| {
            // A genuine survivor, read before anything this step allocates --
            // the conv-state shape, kept here so the test also shows the two
            // cases being told apart rather than everything being admitted.
            rec.record_bind(KEPT);
            rec.record_dispatch();
            // Two unrelated values sharing one address, one after the other.
            // The second's binds are the ones a stale entry would capture.
            rec.record_alloc(ADDR, 4096);
            rec.record_bind(ADDR);
            rec.record_dispatch();
            rec.record_alloc(ADDR, 4096);
            rec.record_bind(ADDR);
            rec.record_dispatch();
            rec.record_bind(ADDR);
            rec.record_dispatch();
            rec.record_alloc(KEPT, 4096);
            rec.record_bind(KEPT);
            rec.record_dispatch();
        };
        step(&mut rec);
        rec.next_step();
        step(&mut rec);

        assert_eq!(
            rec.outlives_step(),
            vec![false, false, true],
            "a value re-allocated at the same address was read as a survivor"
        );
        let plan = rec.plan(ArenaLayout::Packed);
        assert!(
            plan.slot_of(0).is_some() && plan.slot_of(1).is_some(),
            "an ordinary recycled activation was excluded from the arena"
        );
        assert!(
            plan.slot_of(2).is_none(),
            "the genuine survivor was admitted to the arena"
        );
    }

    /// Values that are live at the same time must not share a slot, however
    /// they were allocated. The counterpart to the test above: it would pass
    /// trivially if the recorder simply gave everything its own slot.
    #[test]
    fn simultaneously_live_values_do_not_share() {
        let rec = record_twice(|rec| {
            for addr in [0xa0, 0xb0, 0xc0] {
                rec.record_alloc(addr, 21504);
                rec.record_bind(addr);
                rec.record_dispatch();
            }
            // All three read together -- the SwiGLU moment #68 identifies as the
            // peak: silu(gate), up, and their product.
            for addr in [0xa0, 0xb0, 0xc0] {
                rec.record_bind(addr);
            }
            rec.record_dispatch();
        });

        let plan = rec.plan(ArenaLayout::Packed);
        assert_eq!(plan.slots().len(), 3, "live values shared a slot");
        assert_eq!(plan.arena_bytes(), 3 * 21504);
    }

    /// A value never freed is live to the end of the step, so it shares with
    /// nothing. Conservative in the safe direction: a larger arena, never two
    /// live values on one slot. #68 counts 39 of these.
    #[test]
    fn a_value_never_freed_shares_with_nothing() {
        let rec = record_twice(|rec| {
            rec.record_alloc(0xa0, 4096); // bound throughout
            rec.record_bind(0xa0);
            rec.record_dispatch();
            rec.record_alloc(0xb0, 4096);
            rec.record_bind(0xb0);
            rec.record_dispatch();
            rec.record_alloc(0xc0, 4096);
            rec.record_bind(0xc0);
            rec.record_dispatch();
            // The long-lived value is read once more at the end, so it spans
            // both of the others.
            rec.record_bind(0xa0);
            rec.record_dispatch();
        });

        let plan = rec.plan(ArenaLayout::Packed);
        let slots: Vec<usize> = (0..plan.allocations())
            .map(|i| plan.slot_of(i).expect("ordinal is served"))
            .collect();
        assert_ne!(slots[0], slots[1], "a live value shared with a later one");
        assert_ne!(slots[0], slots[2], "a live value shared with a later one");
        assert_eq!(slots[1], slots[2], "two dead values did not share");
        assert_eq!(plan.slots().len(), 2);
    }

    /// **Session state is excluded because its size moves between two recorded
    /// steps** (§9.1, #68 finding 4), and this is the check that makes the
    /// exclusion empirical rather than a guess about kernel names.
    ///
    /// The modelling error it prevents is not an overflow but a *drift*: a slot
    /// whose occupant grows cannot hold a fixed offset, so every value packed
    /// after it moves. With KV left in, 969 bindings fail #68's 674-position
    /// stability check.
    #[test]
    fn a_value_that_grows_between_steps_is_kept_out_of_the_arena() {
        let mut step = |kv: usize, rec: &mut ArenaRecorder| {
            rec.record_alloc(0xa0, 4096);
            rec.record_bind(0xa0);
            rec.record_dispatch();
            rec.record_alloc(0xb0, kv * 1024);
            rec.record_bind(0xb0);
            rec.record_dispatch();
        };
        let mut rec = ArenaRecorder::new();
        step(35, &mut rec); // step one
        rec.next_step();
        step(36, &mut rec); // step two: the kv value is one token larger

        let (excluded, total) = rec.excluded();
        assert_eq!(total, 2);
        assert_eq!(excluded, 1, "the growing value was not excluded");

        let plan = rec.plan(ArenaLayout::Packed);
        assert!(
            plan.slot_of(0).is_some(),
            "the fixed activation was excluded"
        );
        assert_eq!(
            plan.slot_of(1),
            None,
            "the growing value was given a fixed slot"
        );
        // And the arena is sized for the activation alone, not for the kv value.
        assert_eq!(plan.arena_bytes(), 4096);
    }

    /// The mutation for the test above: if the two steps are *identical*, the
    /// same value is an ordinary activation and must be served.
    ///
    /// Without this, `a_value_that_grows_between_steps_is_kept_out_of_the_arena`
    /// would also pass if the recorder excluded everything unconditionally --
    /// which it does, deliberately, when there is only one step to look at.
    #[test]
    fn a_value_that_does_not_grow_is_served() {
        let rec = record_twice(|rec| {
            rec.record_alloc(0xa0, 4096);
            rec.record_bind(0xa0);
            rec.record_dispatch();
            rec.record_alloc(0xb0, 35 * 1024);
            rec.record_bind(0xb0);
            rec.record_dispatch();
        });

        let (excluded, total) = rec.excluded();
        assert_eq!((excluded, total), (0, 2), "a fixed value was excluded");
        let plan = rec.plan(ArenaLayout::Packed);
        assert!(plan.slot_of(0).is_some());
        assert!(plan.slot_of(1).is_some());
    }

    /// An unpaired recording excludes everything rather than admitting session
    /// state by default.
    ///
    /// Failing toward the pool costs arena coverage and never correctness, which
    /// is the right direction for a mechanism whose failure mode under
    /// `HazardTrackingModeUntracked` is silent corruption (§9.3).
    #[test]
    fn a_single_recorded_step_plans_nothing() {
        let mut rec = ArenaRecorder::new();
        rec.record_alloc(0xa0, 4096);
        rec.record_bind(0xa0);
        rec.record_dispatch();

        let (excluded, total) = rec.excluded();
        assert_eq!((excluded, total), (1, 1));
        assert_eq!(rec.plan(ArenaLayout::Packed).arena_bytes(), 0);
    }

    /// The plan's byte total is the peak, and it is what the arena allocates.
    #[test]
    fn arena_bytes_is_the_packed_peak() {
        // #68's shape: three MLP-width values live together at the SwiGLU, plus
        // the residual stream and a kv-head value.
        let sizes = [21504, 21504, 21504, 4096, 1024];
        let last_use = [4, 4, 4, 4, 4];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        // All five are live together, so none can share: 3*21504 + 4096 + 1024,
        // each slot 128 B aligned. 21504 and 4096 and 1024 are all multiples of
        // 128, so the total is exact.
        assert_eq!(plan.arena_bytes(), 3 * 21504 + 4096 + 1024);
        assert_eq!(plan.arena_bytes(), 69632, "not #68's measured 68.00 KB");
        assert_eq!(plan.slots().len(), 5);
    }

    /// `ordinal` has one writing thread, so it is not `GpuClock`'s shape and is
    /// deliberately unpadded (#347, §9.2l's unit).
    ///
    /// **This pins the ENUMERATION, which is the load-bearing half.** The
    /// padding decision rests on there being no second writer, and that is a
    /// claim about the call graph rather than about this file -- so what the
    /// test asserts is the source-level fact a future change would break: the
    /// field is written at exactly the sites the docs name, and the one
    /// genuinely concurrent path in the backend (the completion handler) does
    /// not reach the arena.
    ///
    /// A runtime test cannot do this. Padding emits no instruction (§9.2l), so
    /// there is no behaviour to observe -- and a second writer appearing would
    /// be a *correctness* defect (§9.2g: a concurrent bump allocator's
    /// ordinal->offset map varies run to run) long before it was a false-sharing
    /// one. The check is therefore on the write shape, in the same spirit as
    /// #507's source-level guard for `PoolCounters`.
    #[test]
    fn the_ordinal_has_one_writing_thread() {
        let src = include_str!("arena.rs");

        // Every mutation of the field, enumerated. `fetch_add` is `acquire`'s
        // and `store` is `begin_step`'s; the struct-literal field is
        // construction. A fourth would mean a writer the padding decision did
        // not consider.
        //
        // # The needles are ASSEMBLED, because a literal one finds itself
        //
        // This took two tries and both failures are the same shape. Counting
        // `"ordinal.fetch_add"` reads **6**: the three real sites and the three
        // this assertion spells to name them. Anchoring on the receiver path
        // and skipping comment lines *also* reads 6 -- the needles are string
        // literals in this test's own code, which no comment filter excludes.
        //
        // **A pattern written to find a thing is a thing the pattern finds.**
        // #347's own prior run shouted the smaller version (`repr(align(128))`
        // counting 2 in a 148-line file: the attribute, and the sentence
        // quoting it), and `CONTRIBUTING.md` §3.1a rule 3 is the general form.
        // What is new here is that *anchoring is not the repair* -- a better
        // needle is still a needle, spelled in the file it searches.
        //
        // The repair is that the literal never appears: each needle is built by
        // concatenation, so the source contains its halves and not the string.
        // Verified by the arms below, which fail if either half stops being
        // unique to the site it names.
        let count = |needle: &str| src.matches(needle).count();
        let field = "ordinal";
        let recv = format!("self.inner.{field}.");
        let writes = count(&format!("{recv}fetch{}add", '_'))
            + count(&format!("{recv}sto{}", "re("))
            + count(&format!("{field}: AtomicUsize::{}", "new(0)"));
        assert_eq!(
            writes, 3,
            "ArenaInner::ordinal has {writes} write sites, not the 3 the \
             unpadded decision was made against (#347). A new writer needs its \
             thread established before this field's docs stay true: if it is \
             the decode loop the decision is unchanged, and if it is a second \
             thread this is GpuClock's shape and wants CachePadded -- and \
             §9.2g's determinism argument first."
        );

        // The discriminator against `GpuClock`: that pair is written by the
        // host AND by Metal's completion handler. Nothing here is.
        //
        // Assembled AND scoped to code, because this one is named in prose.
        // The field's docs cite it to explain why it is absent, so a literal
        // `contains` matches the explanation and reports the opposite of the
        // truth. Unlike the write census above, a comment filter *is* the right
        // repair here: the mention is a `///` line rather than a needle in this
        // test's own code, so the two halves of §3.1a rule 3 want different
        // fixes and applying either alone leaves one hole.
        let code_only = |needle: &str| {
            src.lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !t.starts_with("//") && t.contains(needle)
                })
                .count()
        };
        assert_eq!(
            code_only(&format!("on_command_buffer{}", "_complete")),
            0,
            "arena.rs subscribes to the completion handler, which is the one \
             concurrent path in this backend (metal_backend/mod.rs:2294). That \
             would make `ordinal` a two-thread field and this file's unpadded \
             decision would need re-taking against §9.2l."
        );
    }

    /// The plan is never mutated after construction, which is what makes it
    /// safe to read without a lock (issue #562).
    ///
    /// # Why this is a source-level check and not a runtime one
    ///
    /// Immutability is a property of the **write set**, and a runtime test can
    /// only observe that no sequence it happened to try disturbed it -- which
    /// is #507's argument for `PoolCounters`' gauge verbatim: *"a runtime test
    /// catches only a gauge some sequence disturbs, where 'no field here is
    /// assigned' is visible in the text and invisible to any pool state."*
    ///
    /// The claim it pins is the one the removal rests on: `plan`,
    /// `slot_handles` and `layout` are established in `Arena::new` and never
    /// assigned again, so every reader on the decode path reads data the `Arc`
    /// already published and no reader can race a writer.
    ///
    /// # The needles are assembled, for §9.2p's reason
    ///
    /// A literal needle finds itself -- #347's own run counted
    /// `"ordinal.fetch_add"` at **6** against a true 3, three of them being the
    /// assertion's own spellings, and found that *anchoring is not the repair*
    /// because a better needle is still a needle spelled in the file it
    /// searches. Built by concatenation, the source holds the halves and never
    /// the string.
    #[test]
    fn the_plan_is_never_mutated_after_construction() {
        let src = include_str!("arena.rs");

        // Assignments to the plan's fields through the shared handle. The only
        // legitimate initialisation is the struct literal in `Arena::new`,
        // which does not go through `self.inner`.
        let code_only = |needle: &str| {
            src.lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !t.starts_with("//") && !t.starts_with("///") && t.contains(needle)
                })
                .count()
        };

        for field in ["plan", "slot_handles", "layout"] {
            let assign = format!("plan_state.{field} {}", "=");
            assert_eq!(
                code_only(&assign),
                0,
                "`plan_state.{field}` is assigned after construction. The \
                 lock-free read in `acquire` rests on this field being \
                 immutable once the `Arc` publishes it (#562); a writer makes \
                 it a data race that `HazardTrackingModeUntracked` reports as \
                 nothing (§3.5, §9.3). Either revert to a mutex or establish \
                 that the writer runs before any reader exists."
            );
        }

        // `plan_state` itself must not be replaced wholesale either -- that
        // would be the same defect one level up, and a per-field scan would
        // miss it.
        assert_eq!(
            code_only(&format!("inner.plan_state {}", "=")),
            0,
            "`ArenaInner::plan_state` is assigned after construction, which \
             defeats the per-field check above (#562)."
        );

        // Non-vacuity: the fields must actually be READ through the handle, or
        // the zeros above are equally consistent with the struct having been
        // renamed out from under this test (§15.1a's VACUOUS class, #457's
        // rule that a zero is not evidence unless something nonzero came from
        // the same command).
        let reads = code_only("plan_state.plan") + code_only("plan_state.slot_handles");
        assert!(
            reads > 0,
            "no read of `plan_state.plan` or `plan_state.slot_handles` -- this \
             test is asserting zeros about a field that no longer exists, \
             which reads exactly like a passing check."
        );
    }

    /// The counters carry no invariant between fields, which is what makes them
    /// atomics rather than a mutex (issue #562).
    ///
    /// **Asserted by observation rather than by restating the derivation**
    /// (`CONTRIBUTING.md` §3.1a rule 3): the test drives the arena through
    /// every disposition `acquire` can take and checks that each counter moved
    /// by the amount that disposition implies. A counter wired to the wrong
    /// branch is caught by the value rather than by the shape.
    ///
    /// It is an ABSOLUTE reference, not a shape (§3.1b rule 1): `[1, 1, 1]` is
    /// a stronger assertion than *all equal*, and the difference matters here
    /// because a defect that bumped every counter on every call would leave a
    /// flat sequence flat.
    #[test]
    fn each_counter_moves_only_on_its_own_disposition() {
        // #68's shape, non-aliasing so each ordinal has its own slot and the
        // dispositions below are reachable one at a time.
        let sizes = [21504usize, 4096, 1024];
        let last_use = [2usize, 2, 2];
        let (arena, _pool) = arena_of(&sizes, &last_use, ArenaLayout::NonAliasing);

        let before = arena.counters();
        assert_eq!(before.offers, 0, "a fresh arena has offered nothing");

        arena.begin_step();
        assert_eq!(arena.counters().steps, before.steps + 1, "begin_step");

        // A hit: the first ordinal at exactly its planned size.
        let served = arena.acquire(sizes[0]);
        assert!(served.is_some(), "ordinal 0 at its planned size must serve");
        let c = arena.counters();
        assert_eq!(c.hits, 1, "one hit");
        assert_eq!(c.offers, 1, "one offer");
        assert_eq!(c.declined_size, 0, "a hit is not a size decline");
        assert_eq!(c.declined_exhausted, 0, "a hit is not an exhaustion");

        // A size mismatch: the next ordinal at the wrong size.
        assert!(
            arena.acquire(sizes[1] + 128).is_none(),
            "wrong size declines"
        );
        let c = arena.counters();
        assert_eq!(c.declined_size, 1, "one size decline");
        assert_eq!(c.hits, 1, "a size decline is not a hit");
        assert_eq!(c.offers, 2, "every acquire is an offer");

        // Past the end of the plan: exhaustion, not a size decline.
        for _ in 0..sizes.len() {
            let _ = arena.acquire(4096);
        }
        let c = arena.counters();
        assert!(
            c.declined_exhausted > 0,
            "walking past the plan exhausts it"
        );
        assert_eq!(c.hits, 1, "exhaustion never serves");

        arena.reset_counters();
        let c = arena.counters();
        assert_eq!(c.offers, 0, "reset clears offers");
        assert_eq!(c.hits, 0, "reset clears hits");
        assert_eq!(c.declined_size, 0, "reset clears declined_size");
        assert_eq!(c.steps, 0, "reset clears steps");
    }

    /// Nothing downstream of the `fetch_add` derives ordering from the ordinal,
    /// which is what makes `Relaxed` sufficient (#347).
    ///
    /// **The consumers are a `Vec` index and an immutable table lookup**, and
    /// this asserts the property those rest on: `by_ordinal` is a pure function
    /// of position. Reading entry `n` must not depend on entry `n-1` having been
    /// read, because that is exactly the edge `Relaxed` does not provide.
    ///
    /// Asserted by *observation* rather than by restating the derivation
    /// (`CONTRIBUTING.md` §3.1a rule 3): the same plan is read in forward order
    /// and in reverse, and every entry must agree. A `by_ordinal` that carried
    /// state between reads would disagree in one direction.
    #[test]
    fn nothing_downstream_derives_ordering_from_the_ordinal() {
        // #68's shape, with a declined ordinal in the middle so the test covers
        // the `None` arm as well as the served one.
        let sizes = [21504, 4096, 1024];
        let last_use = [2, 2, 2];
        let plan = plan_from_sizes(&sizes, &last_use, ArenaLayout::Packed);
        let n = plan.allocations();
        assert!(n >= 3, "fixture is too small to show order-independence");

        let forward: Vec<_> = (0..n).map(|i| plan.slot_of(i)).collect();
        let reverse: Vec<_> = {
            let mut v: Vec<_> = (0..n).rev().map(|i| plan.slot_of(i)).collect();
            v.reverse();
            v
        };
        assert_eq!(
            forward, reverse,
            "`by_ordinal` is not a pure function of position, so reading entry \
             n depends on entry n-1 having been read. That is a happens-before \
             edge `Ordering::Relaxed` on the fetch_add does not supply, and the \
             audit at `acquire` is no longer true."
        );

        // Reading one entry many times must also agree -- the lookup is a read
        // of immutable state, so it cannot consume anything.
        for i in 0..n {
            assert_eq!(
                plan.slot_of(i),
                forward[i],
                "ordinal {i} resolves differently on a second read"
            );
        }

        // And the property the `Relaxed` RMW genuinely provides: uniqueness of
        // the ticket. Distinct ordinals that are both served must name distinct
        // *values*; they may share a slot only when the packer decided their
        // live ranges are disjoint, which `check_disjoint` already guards.
        assert!(
            plan.check_disjoint().is_ok(),
            "the fixture's slots overlap, so this test cannot speak to \
             uniqueness"
        );
    }
}
