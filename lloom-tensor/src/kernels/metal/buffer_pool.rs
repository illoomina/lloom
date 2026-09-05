//! An event-driven buffer pool for the Metal backend.
//!
//! # Why this exists
//!
//! The pool this replaces stored `Arc<Buffer>` in size buckets and decided a
//! buffer was reusable by testing `Arc::strong_count(b) == 1`. Because the pool
//! itself held a strong reference, that predicate was only ever true *between*
//! uses -- and nothing announced the transition. So the count had to be
//! **polled**, at two unrelated moments: a full sweep in `drop_unused_buffers`,
//! and again at every lookup in `find_available_buffer`.
//!
//! That is the actual defect. The linear scan is a consequence: with no event
//! at the instant a buffer becomes free, the only way to find a free one is to
//! look at all of them.
//!
//! # What changes
//!
//! The pool holds `Weak`, the caller holds the only strong reference, and
//! `PooledBuffer::drop` **pushes** the buffer back at the exact moment the last
//! user releases it. Lookup then pops from a free list instead of searching for
//! a free buffer.
//!
//! Reclamation stays automatic and `Drop`-driven. **No caller is required to
//! release a buffer by hand**, which is the property that makes candle's
//! allocate-and-forget ergonomics work, and the reason this is not a manual
//! `free()` API. `PooledBuffer` derefs to `Buffer`, so every existing consumer
//! -- all of which take `&Buffer` -- is unaffected.
//!
//! # Structure
//!
//! ```text
//!   free:    BTreeMap<usize, Vec<Buffer>>  size -> buffers ready to hand out
//!   pending: VecDeque<PendingBuffer>       released, waiting on the GPU
//!   live_buffers / live_bytes: usize       what callers currently hold
//! ```
//!
//! `BTreeMap` rather than `HashMap` because lookup wants the *smallest bucket
//! at least as large as the request*, which is `range(size..).next()`: one
//! ordered probe, not a scan for the minimum. Buckets that hold nothing are
//! never visited, which is what removes the pathology -- in a 400-token
//! generation the old pool accumulated 1345 bucket keys of which 1324 were
//! empty, and walked all of them on every allocation.
//!
//! # Reuse is decided on the GPU clock
//!
//! A buffer becomes reusable when **the GPU is finished with it**, not when the
//! CPU drops its last handle. Those are different instants, and the difference
//! is a correctness bug (issue #19): the CPU routinely runs a long way ahead of
//! the GPU, so a dropped buffer usually still has work outstanding on it. The
//! pool used to hand it straight back, aliasing two unrelated tensors onto one
//! allocation while the first was still being written. Measured, ~90 % of the
//! wrong values that produced were exactly another operation's correct output.
//!
//! No fence can fix that, which is why the fix is here rather than in the
//! encoder. At the instant the pool aliases two tensors, no encoder has bound
//! the buffer, so there is nothing to fence against; by the time one binds it,
//! it looks like an ordinary fresh allocation. The dependency is real and
//! invisible at every point where a fence could be emitted.
//!
//! So the release condition is GPU completion, and the CPU drop is not a second
//! predicate beside it -- it is only the signal that the buffer has no *future*
//! CPU user. One clock decides reuse. See [`GpuClock`] and `PoolInner::release`.

use super::cache_pad::CachePadded;
use super::Buffer;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Tracks how far the GPU has got through the command buffers submitted to it.
///
/// Command buffers on one queue complete **in order**, so a single monotone
/// counter is enough: if buffer N has completed, every buffer before it has
/// too. `Commands` already relies on this to wait only on the last in-flight
/// buffer.
///
/// Two counters:
///
/// - `submitted` names the command buffer work is *currently being encoded
///   into*. It is incremented when that buffer is committed and a fresh one
///   takes its place.
/// - `completed` is the highest epoch the GPU has finished.
///
/// A buffer dropped now may have been bound into any command buffer up to and
/// including `submitted`, so `submitted` is the epoch it must outlive. That is
/// conservative -- the buffer may have last been touched much earlier, or never
/// touched at all -- and deliberately so: being wrong in this direction costs a
/// little reuse latency, while being wrong in the other direction corrupts.
///
/// # The two counters are on different cache lines, and that is not tidiness
///
/// **They are written by different threads, concurrently** (issue #442).
/// `submitted` is a `fetch_add` on the **host** thread, from
/// `Commands::commit_swap_locked`; `completed` is a `fetch_max` from the
/// **Metal completion handler**, which Metal delivers on a driver thread --
/// the doc comment on [`Self::mark_completed`] says so in its own terms, since
/// handlers for different command buffers may arrive on different threads.
/// Nothing serialises the pair: the host commits the next buffer while the GPU
/// is still retiring the last, which is the *normal* case rather than a race
/// (§6.4: "whenever the CPU runs ahead of the GPU").
///
/// Unpadded they are adjacent `AtomicU64` in one 128 B line, so every
/// increment by either invalidates the other's copy. It fires **~11-14 times
/// per token** at §9.2f's command-buffer count -- not per dispatch, but not
/// rare either.
///
/// Modelled on this hardware, two threads doing `atomic_fetch_add` at
/// `memory_order_relaxed`: **164.60 Matomics/s on one line against 892.63 on
/// two, a 5.42x penalty**. The padding emits no instruction and allocates
/// nothing, so it cannot change what is computed -- which is why the digest
/// gate for it is an *unmoved* pair rather than a new one.
///
/// **This is the write/write case and it is the one that pays.** A read-mostly
/// flag measures 0.96x -- a wash -- and is deliberately not padded anywhere;
/// [`super::cache_pad`] carries that measurement and the unit the rule needs.
#[derive(Debug)]
pub struct GpuClock {
    /// Host thread, per committed command buffer.
    submitted: CachePadded<AtomicU64>,
    /// Metal completion handler, on a driver thread. Its own line -- see the
    /// struct docs.
    completed: CachePadded<AtomicU64>,
    /// Distinct per clock for the life of the process, so a reader that stored
    /// an epoch can tell whether it is still comparing against the same clock.
    /// An address cannot do that: a dropped clock's allocation is reused.
    id: u64,
}

impl Default for GpuClock {
    fn default() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        GpuClock {
            submitted: CachePadded::default(),
            completed: CachePadded::default(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl GpuClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// This clock's process-unique identity. Never zero.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The epoch a buffer dropped right now must outlive.
    pub fn current_epoch(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// Opens a new epoch, because the command buffer holding the old one has
    /// been committed. Returns the epoch just closed -- the one whose
    /// completion handler should report back to [`Self::mark_completed`].
    pub fn commit_epoch(&self) -> u64 {
        self.submitted.fetch_add(1, Ordering::AcqRel)
    }

    /// Records that the GPU has finished epoch `epoch`.
    ///
    /// Stored as `epoch + 1` -- a count of finished epochs -- so that the
    /// initial zero means "nothing has completed" rather than "epoch 0 has",
    /// which are different states and would otherwise be indistinguishable.
    ///
    /// Takes a max rather than a store because completion handlers for
    /// different command buffers may be delivered on different threads;
    /// in-order execution guarantees the ordering of the *work*, not of the
    /// notification.
    pub fn mark_completed(&self, epoch: u64) {
        self.completed.fetch_max(epoch + 1, Ordering::AcqRel);
    }

    /// How many epochs have finished. Zero means none have.
    pub fn completed_count(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Whether work encoded in `epoch` is known to have finished.
    pub fn is_complete(&self, epoch: u64) -> bool {
        self.completed_count() > epoch
    }
}

/// A buffer that returns itself to its pool when the last handle drops.
///
/// Derefs to `Buffer`, so it substitutes for the `Arc<Buffer>` this replaces
/// everywhere the buffer is read. Cloning the `Arc<PooledBuffer>` shares the
/// handle; the return happens when the final clone drops.
pub struct PooledBuffer {
    /// `Option` only so `drop` can take ownership and hand the buffer back
    /// rather than destroying it. Always `Some` for a live handle.
    buffer: Option<Buffer>,

    /// The bucket this belongs to -- the *allocated* size, not the requested
    /// one, so it goes back where a future request of that size will find it.
    size: usize,

    /// `Weak` so a leaked buffer cannot keep the whole device alive, and so a
    /// buffer outliving its pool drops cleanly instead of resurrecting it.
    pool: Weak<PoolInner>,
}

impl PooledBuffer {
    /// A handle with no pool behind it: dropping it frees the buffer instead
    /// of offering it for reuse.
    ///
    /// For allocations that are deliberately outside the pool, so that callers
    /// which must not participate in reuse still get the same handle type.
    pub fn unpooled(buffer: Buffer, size: usize) -> Self {
        Self {
            buffer: Some(buffer),
            size,
            pool: Weak::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn buffer(&self) -> &Buffer {
        // Only `None` during `drop`, which does not call this.
        self.buffer
            .as_ref()
            .expect("PooledBuffer used after its buffer was taken in drop")
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Buffer {
        self.buffer()
    }
}

impl AsRef<Buffer> for PooledBuffer {
    fn as_ref(&self) -> &Buffer {
        self.buffer()
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for PooledBuffer {
    /// The event the old design lacked.
    ///
    /// This runs when the last `Arc<PooledBuffer>` goes away, which is the
    /// same instant `Arc::strong_count == 1` used to become true of the pooled
    /// `Arc<Buffer>` -- except that now it is a notification rather than a
    /// state to be discovered later by a scan.
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        // Pool gone: the buffer drops here and Metal frees it. Correct, and the
        // reason the back-reference is `Weak`.
        if let Some(pool) = self.pool.upgrade() {
            pool.release(buffer, self.size);
        } else {
            // Destroyed rather than returned, so any mark must come off before
            // the allocation does (#475).
            //
            // **This branch is why the poison discipline is stated over the
            // mapping rather than over the buffer.** Metal SUB-ALLOCATES: a
            // later `newBufferWithLength:` is served from
            // `allocBufferSubDataWithLength`, which can hand back an address
            // range a destroyed buffer occupied and then `bzero` it. A mark
            // outliving its mapping therefore reports Metal's own zeroing of
            // an unrelated fresh allocation -- a FALSE POSITIVE, and the
            // direction that discredits the instrument rather than merely
            // missing a defect. Measured while building this: the pool's own
            // test suite tripped it, in `__bzero` under
            // `-[IOGPUMetalDevice allocBufferSubDataWithLength:...]`.
            //
            // SAFETY: nothing holds this buffer -- this is its last handle, and
            // the pool that could have handed out another is gone. Inert
            // without `asan-poison`.
            unsafe { super::poison::unpoison_buffer(&buffer) };
        }
    }
}

/// Statistics, maintained unconditionally because they are counter increments
/// on a path that already takes a lock. Read via `MetalDevice::pool_stats`.
///
/// # Every field here is an ACCUMULATOR, and that is now a property rather than
/// an observation (issue #507)
///
/// Each one counts events since the last [`BufferPool::reset_counters`], so
/// zeroing the struct is the whole meaning of resetting it. A **gauge** -- a
/// figure describing the pool's state *now* rather than its history -- cannot
/// live here, because `reset_counters` has no way to tell the two apart and
/// would zero a quantity nothing had changed.
///
/// This carried one until #507: `pending` was written as an assignment of
/// `state.pending.len()` where every other field is `+=`, so `reset_counters`
/// left `counters().pending` reading 0 while `occupancy().pending_buffers` read
/// 1 -- **two public accessors of one pool disagreeing about how many buffers
/// await the GPU, at one instant, with nothing having changed about the GPU.**
/// It now lives only in [`PoolOccupancySnapshot::pending_buffers`], where every
/// other live figure already lived, derived from `state.pending` at the moment
/// it is asked for rather than mirrored.
///
/// **The repair removes the class, not the instance.** Re-deriving the gauge
/// inside `reset_counters` would have fixed this field and left the next one
/// added here free to reintroduce it; a struct whose every member is an
/// accumulator cannot express the defect. That is `DESIGN.md` §6.7 **L2** --
/// state duplicated and then not kept -- and a duplicate that can disagree is
/// worse than one that cannot.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolCounters {
    /// Calls into `acquire`.
    pub lookups: u64,
    /// Calls that were served from a free list.
    pub hits: u64,
    /// Free-list buckets probed, summed over all lookups.
    ///
    /// The scan-length figure comparable to the old pool's "buckets walked".
    /// Bounded by 2 per lookup here: the exact-size probe, plus at most one
    /// larger bucket from the ordered range query.
    pub buckets_probed: u64,
    /// Buffers returned by `Drop`.
    pub releases: u64,
    /// Fresh `MTLBuffer` allocations.
    pub allocations: u64,
    /// Bytes handed to `newBufferWithLength`.
    pub allocated_bytes: u64,
    /// Buffers destroyed by `trim`.
    pub trimmed: u64,
    /// Calls into `trim`.
    pub trims: u64,
    /// Releases that had to wait for the GPU before becoming reusable.
    ///
    /// In steady-state decode this is essentially every release: the CPU runs
    /// far enough ahead that a dropped buffer almost always still has work
    /// outstanding. A number near zero here would mean the deferral is not
    /// doing anything, and the corruption it prevents would still be reachable.
    pub deferred: u64,
    /// Buffers moved from the pending list into a free list on GPU completion.
    pub drained: u64,
    /// Free buffers destroyed to keep the free list inside its byte budget.
    ///
    /// Nonzero means the workload is freeing sizes it does not ask for again --
    /// a growing KV cache does this every token. Zero means every freed buffer
    /// found a later use, which is the steady state a fixed-shape workload
    /// reaches.
    pub evicted: u64,

    /// Bytes callers actually asked for, summed over every served `acquire`.
    ///
    /// The denominator of the overhang below. Counted at the same instant as
    /// [`Self::served_bytes`] and over exactly the same population, because a
    /// ratio between two counters incremented on different conditions is the
    /// shape `CONTRIBUTING.md` §3.1b names -- two numbers that look like a
    /// comparison and are not.
    pub requested_bytes: u64,

    /// Bytes the pool actually handed over, summed over the same acquisitions.
    ///
    /// Not the same as [`Self::allocated_bytes`], and the difference is the
    /// point. `allocated_bytes` counts what went to `newBufferWithLength` --
    /// fresh allocations only, so a pool HIT contributes nothing to it. This
    /// counts what every caller was *given*, hit or miss, which is the only
    /// population against which "what did we ask for" is a fair denominator.
    pub served_bytes: u64,

    /// [`Self::requested_bytes`] and [`Self::served_bytes`], split by the size
    /// band the *request* fell in (`DESIGN.md` §9.2u, #47).
    ///
    /// # Why the request and not the bucket
    ///
    /// The band is chosen by what the caller **asked for**, so a request that
    /// is over-served stays in its own band rather than being filed under
    /// whatever oversized bucket happened to answer it. Keying on the served
    /// size would put the overhang in the *large* band by construction, and the
    /// split would be an artifact of the keying -- the shape §3.1b names, a
    /// comparison that is not one.
    ///
    /// # Why bands rather than a tenant tag
    ///
    /// A tenant tag would have to be threaded from `Tensor::cat` and
    /// `KvSlot::append` through `new_buffer`'s call layers, and the pool has no
    /// argument for it (§9.2s: *"the request is discarded at the pool
    /// boundary"*). A band is derivable from what `acquire` already holds. It
    /// is a **proxy** and is reported as one: §9.2u states which LFM2 class
    /// occupies which size range and where they could overlap, so a reader can
    /// see which rows are clean rather than trusting a label the pool cannot
    /// verify.
    pub bands: [SizeBand; SIZE_BAND_COUNT],

    /// Range-query candidates rejected by the reuse-oversize bound
    /// (`DESIGN.md` §9.2v, #47).
    ///
    /// **The arm's engagement proof, and it is a third counter rather than a
    /// derivation from the two above for #500's reason**: a bound that is
    /// installed but never binds and a bound that is not installed at all
    /// produce the *same* `requested`/`served` sums, so nothing else in this
    /// struct separates them. A run reporting `bound=4.0` with
    /// `oversize_declined = 0` has either found a free list that never offers a
    /// candidate past the bound -- a real and interesting outcome -- or has not
    /// engaged the mechanism at all, and only this counter can say which.
    ///
    /// Counted at the rejection, so it is a count of *candidates declined*,
    /// never of allocations caused: a decline is followed by the caller
    /// allocating fresh, but a miss with an empty free list allocates fresh
    /// too and does not increment this.
    pub oversize_declined: u64,
}

/// One row of [`PoolCounters::bands`]: the same two sums over one size range.
///
/// Both fields are accumulators over the same population and the same instant
/// as their whole-pool counterparts, so the bands sum to
/// [`PoolCounters::requested_bytes`] and [`PoolCounters::served_bytes`]
/// exactly -- asserted by `bands_sum_to_the_whole_pool_counters`, which is what
/// makes a per-band figure comparable *with* the headline rather than merely
/// adjacent to it.
#[derive(Debug, Default, Clone, Copy)]
pub struct SizeBand {
    /// Acquisitions whose request fell in this band.
    pub acquisitions: u64,
    /// Bytes asked for, summed over those acquisitions.
    pub requested_bytes: u64,
    /// Bytes handed over, summed over the same acquisitions.
    pub served_bytes: u64,
}

impl SizeBand {
    /// Bytes served beyond what was asked for, in this band.
    pub fn overhang_bytes(&self) -> u64 {
        self.served_bytes.saturating_sub(self.requested_bytes)
    }
}

/// Upper bounds (exclusive) of the size bands, in bytes.
///
/// Chosen against LFM2's measured decode geometry (`DESIGN.md` §9.2u) rather
/// than as round numbers. The B=1 activation shapes are 1 KB -- 21.5 KB
/// (§9.2c's 68 KB arena over five slots) with the `Cat` arm's per-layer
/// temporaries just above; one K or V tensor is 3.83 MB at `kv_len` 234 and
/// grows linearly (§9.2s). **256 KB is what separates the two at every `kv_len`
/// this project has run** -- and because that separation is `kv_len`-dependent
/// rather than structural, §9.2u states the bound explicitly instead of
/// presenting the split as exact.
pub const SIZE_BAND_LIMITS: [usize; SIZE_BAND_COUNT - 1] = [4 * 1024, 256 * 1024, 4 * 1024 * 1024];

/// Number of bands: the limits above, plus the unbounded tail.
pub const SIZE_BAND_COUNT: usize = 4;

/// Human-readable band names, index-aligned with [`PoolCounters::bands`].
pub const SIZE_BAND_NAMES: [&str; SIZE_BAND_COUNT] = ["<4KB", "4KB-256KB", "256KB-4MB", ">=4MB"];

/// The reuse-oversize bound in force, as a multiple of the request
/// (`DESIGN.md` §9.2v, #47). `None` is the shipped behaviour: unbounded.
///
/// # What this axis is, and why it is the one the measurement pointed at
///
/// §9.2u measured the pool's overhang and found **84.1 % of it in the
/// 4KB-256KB activation band at a 16.01x served/requested ratio**, against KV's
/// 15.9 % at 1.35x. The mechanism is not that the activation band asks for too
/// much: it is that [`BufferPool::acquire`]'s second probe returns the smallest
/// bucket *strictly larger* than the request **with no ceiling on how much
/// larger**, and by token 90 the only thing left free is the `Cat` KV arm's
/// discarded multi-MB tensors -- `evict_over_budget` having destroyed the small
/// buckets to stay inside `free_budget`, oldest-first and **size-blind**. So a
/// 76 KB activation request meets a 3.83 MB bucket and takes it.
///
/// §9.2t went looking for a borrow for this and found none: *"neither engine
/// bounds how much larger a reused buffer may be than the request"* -- ggml
/// searches best-fit over an offset-sorted free-block list, we take the
/// smallest bucket strictly larger, and **no prior art bounds the ratio**. That
/// section's disposition was that it is *"ours to settle"*. This is the
/// instrument that settles it.
///
/// # Declining is not refusing to reuse
///
/// When the bound rejects a candidate, `acquire` returns `None` and the caller
/// allocates fresh **at its own size**, which then returns to the free list as a
/// bucket of that size -- so the mechanism *creates* the small buckets whose
/// absence is the pathology, rather than merely refusing the large ones. That
/// is why the arm is expected to move the band-1 ratio and not merely trade one
/// waste for another.
///
/// **`Drop` reclamation is untouched** (§9.2a layer 1, non-negotiable). This
/// changes only *which bucket answers a request*, never who releases a buffer
/// or when. No caller is asked to free anything.
///
/// # Read once, not per acquisition
///
/// A `OnceLock`, exactly as §9.2r's `LLOOM_586_BUFFER_ALIGNMENT`: `acquire` is
/// the hot path at §6.3b's ~11.6 destroys/token, and an env read per call would
/// put a syscall-shaped cost on it. Reading once also makes the arm a property
/// of the process, which is what lets a harness report it back from the
/// mechanism rather than from the flag (§7.1a-i).
pub fn reuse_oversize_bound() -> Option<f64> {
    static BOUND: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *BOUND.get_or_init(|| {
        let Ok(raw) = std::env::var("LLOOM_47_REUSE_OVERSIZE_BOUND") else {
            return None;
        };
        let raw = raw.trim();
        // The explicit spelling of the shipped arm, so a sweep can name it
        // rather than expressing it by unsetting the variable -- an arm you
        // cannot write down is one a harness renders as absent, which is
        // §7.1a-i's present-and-wrong case from the other side.
        if raw.eq_ignore_ascii_case("off") || raw.eq_ignore_ascii_case("none") {
            return None;
        }
        let Ok(v) = raw.parse::<f64>() else {
            panic!(
                "LLOOM_47_REUSE_OVERSIZE_BOUND={raw:?} is not a number. \
                 Give a multiple of the request >= 1.0, or `off` for the \
                 unbounded behaviour that ships (DESIGN.md §9.2v)."
            )
        };
        // Refused rather than clamped, and loudly -- a silently-corrected arm
        // renders a bound the process did not take and compares equal to real
        // runs of that arm (§9.2r's reasoning, same failure).
        if !v.is_finite() || v < 1.0 {
            panic!(
                "LLOOM_47_REUSE_OVERSIZE_BOUND={v} is below 1.0. A bound under \
                 1.0 would reject every candidate including an exact match, \
                 which is not a reuse policy but a way to disable the pool \
                 (DESIGN.md §9.2v)."
            )
        }
        Some(v)
    })
}

/// The eviction order in force (`DESIGN.md` §9.2w, #47). `Age` is the shipped
/// behaviour.
///
/// # Why this axis exists, and why it is the lever §9.2v pointed at rather than
/// the one §9.2v built
///
/// §9.2v bounded [`BufferPool::acquire`]'s second probe and took the activation
/// band's served/requested ratio **16.12x -> 1.17x with the acquisition count
/// identical** -- and paid **3.03x the `newBufferWithLength` calls**, hit rate
/// 87.5 % -> 61.8 %. Its own reading of that result is what this axis tests:
///
/// > **the bound does not shrink the free list -- it keeps small buckets
/// > alive** (431 free buffers against 71, 28 buckets against 6) [...] **the
/// > lever this points at is not this bound**: `evict_over_budget` is
/// > oldest-first and **size-blind**, so **evicting by SIZE rather than by AGE
/// > would reach the same place without declining a single reuse and without
/// > paying the 3.03x.**
///
/// That is a **prediction with a mechanism**, and it is falsifiable in the
/// direction that matters: if band 1's ratio falls only because reuse was
/// declined, then an arm that declines nothing cannot move it.
///
/// # Why the largest, and not the smallest
///
/// `free_budget` caps the *bytes* held free, so evicting one 3.83 MB KV discard
/// releases as much budget as **fifty-two** 76 KB activation buckets. Under
/// `Age` the small buckets are destroyed first purely because they were freed
/// first, and what survives is the multi-MB tail that then over-serves band 1.
/// Taking the largest first reaches the same `free_bytes` in **fewer**
/// evictions and leaves the small buckets in place -- which is the state
/// §9.2v's `4.0` arm reached by refusing reuse, arrived at by retention
/// instead.
///
/// **The two arms are not the same policy under different names**, and the
/// discriminator is `oversize_declined`: this axis must leave it at **0**,
/// because it never rejects a candidate. An arm that moves the ratio *and*
/// declines nothing separates retention from refusal.
///
/// # What this does not change
///
/// **`Drop` reclamation is untouched** (§9.2a layer 1, non-negotiable). This
/// decides only *which free buffer is destroyed when the budget is exceeded*,
/// never who releases a buffer or when. No caller is asked to free anything,
/// and every buffer evicted under either arm was already free.
///
/// **`acquire` is untouched**, which is what keeps this off §3.3's no-O(N)
/// list from the other side: the cost moves to the eviction path, which already
/// runs inside the per-command-buffer completion handler (§6.3f) rather than on
/// the encode path.
pub fn eviction_order() -> EvictionOrder {
    static ORDER: std::sync::OnceLock<EvictionOrder> = std::sync::OnceLock::new();
    *ORDER.get_or_init(|| {
        let Ok(raw) = std::env::var("LLOOM_47_EVICTION_ORDER") else {
            return EvictionOrder::Age;
        };
        let raw = raw.trim();
        match raw.to_ascii_lowercase().as_str() {
            // The shipped arm spelled explicitly, so a sweep can name it rather
            // than express it by unsetting the variable (§7.1a-i).
            "age" | "oldest" | "off" => EvictionOrder::Age,
            "size" | "largest" => EvictionOrder::Size,
            _ => panic!(
                "LLOOM_47_EVICTION_ORDER={raw:?} is not an eviction order. \
                 Give `age` for the oldest-first behaviour that ships, or \
                 `size` for largest-first (DESIGN.md §9.2w)."
            ),
        }
    })
}

/// Which free buffer `evict_over_budget` destroys first (`DESIGN.md` §9.2w).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionOrder {
    /// Oldest-first, size-blind. What ships.
    Age,
    /// Largest-first. Releases the same budget in fewer evictions and leaves
    /// the small buckets alive.
    Size,
}

impl EvictionOrder {
    /// The arm name, for a `RESULT` line.
    ///
    /// Read back from the value rather than from the flag, so a harness reports
    /// the arm the process **took** (§7.1a-i). Spelled out rather than derived
    /// from `Debug` so the rendering cannot drift with a type rename -- and
    /// because §9.2v's own float arm rendered `4` for `4.0` and two runs of one
    /// arm would not have pooled.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Age => "age",
            Self::Size => "size",
        }
    }
}

/// Whether a free bucket of `bucket` bytes may answer a request for `size`,
/// under `bound` (`DESIGN.md` §9.2v, #47).
///
/// # Why this is a pure function and not a comparison inlined at the call site
///
/// [`reuse_oversize_bound`] is a process-wide `OnceLock`, so a test cannot
/// select an arm without deciding it for every other test in the binary. §9.2r's
/// `LLOOM_586_BUFFER_ALIGNMENT` has exactly that shape and is consequently
/// **exercised only end-to-end** -- 18 runs of a whole model to check an
/// arithmetic property. Splitting the *policy* out from the *arm selection*
/// makes the policy unit-testable at every bound including the ones no run
/// takes, and leaves the `OnceLock` doing only what it must: supplying the
/// default.
///
/// This is §3.1b rule 1 -- prefer an absolute reference to a shape. The
/// predicate can be asserted against exact byte pairs; an end-to-end run can
/// only be asserted against itself.
#[inline]
pub fn bucket_admissible(size: usize, bucket: usize, bound: Option<f64>) -> bool {
    match bound {
        // `size` is never 0 on the allocation path -- `buf_size` returns
        // `max(1)` -- but this is a public predicate, and at `size == 0` the
        // product is 0 while every bucket is larger, so an unguarded
        // comparison would reject every candidate and read as "the bound is
        // very tight" rather than as "the question is malformed". Admitting is
        // the direction that matches the unbounded arm.
        Some(_) if size == 0 => true,
        Some(limit) => (bucket as f64) <= (size as f64) * limit,
        None => true,
    }
}

/// Which band a request of `size` bytes falls in.
///
/// A free function rather than a method so the counting site and the tests
/// share one definition -- a second copy is how a band boundary drifts between
/// the instrument and the analysis that reads it.
#[inline]
pub fn size_band(size: usize) -> usize {
    // Linear over three constants and monotone by construction. A binary
    // search would be slower at this length, and a match on ranges would spell
    // the boundaries twice.
    let mut band = 0;
    while band < SIZE_BAND_LIMITS.len() && size >= SIZE_BAND_LIMITS[band] {
        band += 1;
    }
    band
}

impl PoolCounters {
    /// `DESIGN.md` §9.2's residual overhang: bytes held minus bytes wanted.
    ///
    /// The quantity #47's expanded acceptance list asks for a number on before
    /// any rewrite, and which nothing counted -- `buf_size` rounds at
    /// `device.rs`, discards the request, and [`PooledBuffer::size`] records
    /// only *"the allocated size, not the requested one"*.
    ///
    /// # Two sources, deliberately summed
    ///
    /// A caller is over-served on a **miss** when `buf_size` rounds the request
    /// up to [`crate::kernels::metal::arena::ARENA_ALIGNMENT`]-granularity, and on a
    /// **hit** when the ordered range query returns a bucket strictly larger
    /// than the request (`acquire`'s second probe). Both are the same
    /// phenomenon -- bytes resident and unwanted -- and separating them would
    /// invite reporting whichever is smaller.
    ///
    /// # Why saturating
    ///
    /// It cannot go negative: every served buffer is at least as large as its
    /// request, asserted by `served_is_never_below_requested`. The saturation
    /// is so a counter overflow degrades to zero rather than to a
    /// near-`u64::MAX` figure that would read as catastrophic waste.
    pub fn overhang_bytes(&self) -> u64 {
        self.served_bytes.saturating_sub(self.requested_bytes)
    }

    /// The overhang as a fraction of what was served, or `None` if nothing was.
    ///
    /// `None` rather than `0.0` for an empty pool: a zero ratio and "no
    /// acquisitions happened" are different facts, and §3.1c's rule is that a
    /// zero is not evidence unless something nonzero came from the same
    /// command.
    pub fn overhang_fraction(&self) -> Option<f64> {
        if self.served_bytes == 0 {
            return None;
        }
        Some(self.overhang_bytes() as f64 / self.served_bytes as f64)
    }

    /// Record one acquisition into its size band (§9.2u).
    ///
    /// Both call sites -- `acquire`'s hit and `adopt_requesting`'s miss -- go
    /// through here, so the band key is chosen in exactly one place. A second
    /// copy is how the two halves of one population come to disagree about
    /// where a boundary is.
    ///
    /// A method rather than an inline `&mut` borrow at each site for a second
    /// reason, and it is a real one: `pool_counters_holds_accumulators_only`
    /// reads this file's *text* for `counters.<field> =` because a gauge is a
    /// write shape, and `let band = &mut state.counters.bands[..]` is
    /// lexically that pattern while being a borrow. Keeping the borrow inside
    /// `impl PoolCounters` means that guard sees `+=` at every `counters.`
    /// site, and the borrow is covered by
    /// `no_counter_field_is_mutably_borrowed_out_of_the_struct` instead --
    /// which closes a hole the older guard could not see.
    #[inline]
    fn record_band(&mut self, requested: usize, served: usize) {
        let band = &mut self.bands[size_band(requested)];
        band.acquisitions += 1;
        band.requested_bytes += requested as u64;
        band.served_bytes += served as u64;
    }
}

struct PoolState {
    /// Buffers available for immediate reuse, keyed by allocated size.
    ///
    /// Ordered so lookup can ask for "smallest size >= n" directly. A bucket
    /// is removed from the map when it empties, so the map's length is the
    /// number of sizes that actually have something free -- never a graveyard
    /// of emptied keys.
    free: BTreeMap<usize, Vec<Buffer>>,

    /// Buffers currently held by a caller, and their total allocated bytes.
    ///
    /// Two counters rather than a registry of the buffers themselves. A
    /// registry would need an entry inserted on every acquire and removed on
    /// every release -- hot-path work whose only consumer is a diagnostic --
    /// and its dead entries would accumulate between trims, which is the
    /// accumulate-forever shape this change exists to remove. Incrementing a
    /// pair of counters under a lock that is already held costs nothing.
    live_buffers: usize,
    live_bytes: usize,

    /// Buffers the CPU has released but the GPU may still be using, with the
    /// epoch each must outlive.
    ///
    /// Ordered by epoch, because that is the order they become reusable in:
    /// draining is a prefix walk that stops at the first epoch still
    /// outstanding, never a scan of the whole list. Epochs are assigned from a
    /// monotone counter at push time, so the list is sorted by construction and
    /// nothing has to sort it.
    pending: std::collections::VecDeque<PendingBuffer>,

    /// Bytes currently sitting in `free`, maintained incrementally so the
    /// budget check never has to walk the buckets.
    free_bytes: usize,

    /// Sizes in the order they entered `free`, so eviction can take the oldest
    /// without sorting or scanning. An entry whose buffer has since been handed
    /// out is stale and simply skipped; that is cheaper than removing it
    /// eagerly, and it keeps `acquire` off this structure entirely.
    free_order: std::collections::VecDeque<usize>,

    /// Cap on `free_bytes`. See `evict_over_budget` for why there has to be one.
    free_budget: usize,

    /// How much larger than a request a reused bucket may be (`DESIGN.md`
    /// §9.2v, #47). `None` is the shipped behaviour: unbounded.
    ///
    /// # Why this is state rather than a read of the axis at each `acquire`
    ///
    /// It is initialised from [`reuse_oversize_bound`] once, at pool
    /// construction, so a process still takes exactly one arm and `acquire`
    /// still reads no environment. What it buys is that a **test** can select
    /// an arm for one pool, which a process-wide `OnceLock` cannot offer -- and
    /// #586's axis, which does not have this, is exercised only by running a
    /// whole model 18 times.
    ///
    /// The field is written once at construction and by
    /// [`BufferPool::set_reuse_oversize_bound`]; `acquire` only reads it.
    oversize_bound: Option<f64>,

    /// Which free buffer eviction destroys first (`DESIGN.md` §9.2w, #47).
    /// `Age` is the shipped behaviour.
    ///
    /// State rather than a read of the axis inside `evict_over_budget`, for the
    /// same reason as `oversize_bound` above: a test can then select an arm for
    /// **one** pool, where a process-wide `OnceLock` decides it for every test
    /// in the binary.
    eviction_order: EvictionOrder,

    /// Cap on bytes the pools hold that **no planner owns** — `DESIGN.md`
    /// §9.5k's derived residual, installed by admission.
    ///
    /// `None` until a caller installs one, which is every pool in the tests and
    /// the default `BufferPool::new`. **Absent means unbounded, which is the
    /// behaviour that shipped**, so a process that never runs admission is
    /// byte-for-byte what it was.
    ///
    /// See `residual_cap` for what it is, why the quantity it bounds is
    /// derived rather than `live_bytes`, and why this is one branch rather
    /// than a general gate.
    residual: Option<ResidualCap>,

    counters: PoolCounters,
}

/// The derived cap on unplanned bytes, and the planned figure it is derived
/// from (`DESIGN.md` §9.5k).
///
/// # Why a cap here at all, when §9.5e declines per-allocation checking
///
/// §9.5e declines a per-allocation gate on a **correctness** argument rather
/// than a cost one: the allocation site is deep inside a forward pass, its
/// caller is a `Tensor` op with no policy to apply, and the only available
/// response is an error unwinding mid-token. **That argument is right, and it
/// covers the classes admission KNOWS.** Checking those again at allocation
/// time can only re-discover what was already decided, later and worse.
///
/// It does **not** cover the one class admission *derives*, because there is
/// nothing for admission to have decided about it. So this is not a second
/// gate doing admission's job worse — it is the backstop for the single term
/// that can exceed its share at runtime, and **it cannot fire at all if
/// admission was honest**.
///
/// # Why it is affordable
///
/// §9.5e's premise, from #167's ablation (§6.3d): **accounting is cheap,
/// calling into Metal is not.** Eager residency notification cost
/// +0.093 ms/token and the ablation with the bookkeeping kept and the Metal
/// call skipped read baseline — so the cost was Metal's. This check calls into
/// Metal **nowhere**: `live_bytes` and `free_bytes` are already maintained
/// incrementally, `pending_bytes` is a sum over a list the same lock already
/// holds, and the comparison is one branch on integers.
#[derive(Clone, Copy, Debug)]
struct ResidualCap {
    /// `budget − predicted`: what is left for everything the five classes do
    /// not name.
    limit: usize,
    /// The classes this pool serves — weights, KV, scratch, conv — which are
    /// **already inside `live_bytes`** and must be subtracted before the
    /// comparison.
    ///
    /// **Comparing `live_bytes` against `limit` would refuse the weights
    /// themselves**: `to_dtype` puts them in `private_buffers` and
    /// `KvSlot::append` puts KV in `buffers` (§9.5k). The arena is the one
    /// class outside the pools, since `install_arena` calls the raw device.
    planned: usize,
}

/// A fresh allocation would take the pool past its derived residual
/// (`DESIGN.md` §9.5k).
///
/// **The failure stays ugly and becomes ugly at a known boundary rather than
/// as a kernel panic** — which is the whole justification for the check. The
/// two panics this design exists to prevent (§9.5a) gave a machine-wide
/// `IOGPUGroupMemory` assertion with `memoryPressure` reading False; this gives
/// an error naming the quantity that overran and the figure admission derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualExceeded {
    /// Bytes already held that no planner owns.
    pub unplanned: usize,
    /// The allocation that would have crossed the limit.
    pub requested: usize,
    /// `budget − predicted`, from admission.
    pub limit: usize,
    /// The predicted classes this pool serves, subtracted to get `unplanned`.
    pub planned: usize,
}

impl std::fmt::Display for ResidualExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gb = |b: usize| b as f64 / 1e9;
        write!(
            f,
            "memory residual exhausted: {:.2} GB unplanned + {:.2} GB requested \
             > {:.2} GB residual (DESIGN.md §9.5k). {:.2} GB of predicted \
             classes are excluded from the figure. Either an allocation path is \
             growing without bound (§9.5f candidate 2 -- §6.3b's stranding is \
             the measured instance) or admission was given a configuration the \
             run then left behind.",
            gb(self.unplanned),
            gb(self.requested),
            gb(self.limit),
            gb(self.planned),
        )
    }
}

impl std::error::Error for ResidualExceeded {}

/// A buffer waiting for the GPU to finish with it.
struct PendingBuffer {
    buffer: Buffer,
    /// Bucket to return to -- the allocated size, as for a live handle.
    size: usize,
    /// The epoch this must outlive before it can be handed out again.
    epoch: u64,
}

/// Told when the pool destroys a free buffer, so a holder of a second reference
/// to it can let go first.
///
/// Exists for the residency set (`DESIGN.md` §6.3c): the set retains every
/// buffer the pool allocated, and eviction *destroys* buffers to keep the free
/// list inside its budget. Without a notification the set goes on listing an
/// allocation whose handle is gone, which is the state that makes a later
/// `removeAllocation` a machine panic rather than an error.
///
/// This is §6.7 L4a's rule applied to the one transition the pool did not
/// announce: eviction was a state change with no event, so the only other way
/// for the set to learn of it would be to poll -- and the same section says a
/// poll that exists because an event is missing cannot be optimized into
/// correctness.
pub trait BufferEvictionObserver: Send + Sync {
    /// Called with **all** the buffers one eviction round is about to destroy.
    ///
    /// # Why a batch rather than one call per buffer
    ///
    /// Measured, and the difference is the whole cost of this mechanism. A
    /// residency-set removal ends in `commit()`, which is the expensive part;
    /// decode evicts ~11.6 buffers per token (§6.3b, because the KV cache asks
    /// for a slightly larger size every step and never asks for the one it just
    /// freed), so a per-buffer call pays ~11.6 commits per token where a batch
    /// pays one. Per-buffer measured **+0.087 ms/token of non-GPU time — 9 % of
    /// §11.2's 6.1 % budget**; batched is inside the noise.
    ///
    /// Called with the pool's lock **released**, and before the buffers are
    /// dropped, so the observer may take its own locks and is looking at
    /// allocations that still exist.
    fn on_evict(&self, buffers: &[Buffer]);
}

pub struct PoolInner {
    state: Mutex<PoolState>,
    /// How far the GPU has got. Shared with `Commands`, which advances it.
    clock: Arc<GpuClock>,
    /// Told when a free buffer is destroyed. See [`BufferEvictionObserver`].
    ///
    /// `None` for a pool with no second referent, which is every pool in the
    /// tests and the default `BufferPool::new`. The device installs one.
    on_evict: Mutex<Option<Arc<dyn BufferEvictionObserver>>>,
}

/// Default cap on bytes held in a free list.
///
/// Sized from measurement rather than taste. Issue #21 recorded 43.6 MB of free
/// buffers in LFM2 decode steady state under CPU-drop release; deferring to GPU
/// completion holds roughly the in-flight window's worth on top of that. 256 MB
/// leaves several times that headroom while capping the pathological case at a
/// bounded overshoot instead of the 13.6 GB an unbounded free list reached at
/// 400 decode tokens.
///
/// It is a ceiling, not a target. A workload that reuses what it frees never
/// approaches it and never runs the eviction path at all; what it bounds is the
/// case where freed sizes are never asked for again.
const DEFAULT_FREE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

impl PoolInner {
    fn new(clock: Arc<GpuClock>) -> Self {
        Self {
            state: Mutex::new(PoolState {
                free: BTreeMap::new(),
                live_buffers: 0,
                live_bytes: 0,
                pending: std::collections::VecDeque::new(),
                free_bytes: 0,
                free_order: std::collections::VecDeque::new(),
                free_budget: DEFAULT_FREE_BUDGET_BYTES,
                // Read once here rather than per `acquire`, so the process
                // takes one arm and the hot path reads no environment (§9.2v).
                oversize_bound: reuse_oversize_bound(),
                // Same, for the eviction order (§9.2w).
                eviction_order: eviction_order(),
                // Absent = unbounded, which is what shipped. Admission installs
                // one; a process that never runs admission is unchanged.
                residual: None,
                counters: PoolCounters::default(),
            }),
            clock,
            on_evict: Mutex::new(None),
        }
    }

    /// The eviction observer, if one is installed. Cheap: one lock and a clone
    /// of an `Option<Arc>`, taken only when an eviction actually happens.
    fn eviction_observer(&self) -> Option<Arc<dyn BufferEvictionObserver>> {
        self.on_evict.lock().ok().and_then(|o| o.clone())
    }

    /// Hands a buffer back. Called only from `PooledBuffer::drop`.
    ///
    /// **This is where issue #19 is fixed.** The CPU dropping its last handle
    /// says the buffer has no future CPU user; it says nothing about whether
    /// the GPU has finished with it, and in decode it usually has not. So the
    /// buffer does not go into the free list here unless the GPU is known to be
    /// done -- otherwise it is parked until the epoch it was dropped in
    /// completes.
    ///
    /// Lookup is untouched. This is a change to *when* a buffer is offered, not
    /// to how one is found.
    fn release(&self, buffer: Buffer, size: usize) {
        let epoch = self.clock.current_epoch();

        let Ok(mut state) = self.state.lock() else {
            // Poisoned: another thread panicked holding the lock. Dropping the
            // buffer here is safe -- Metal frees it -- and is better than
            // propagating a panic out of a destructor.
            return;
        };
        state.counters.releases += 1;
        state.live_buffers = state.live_buffers.saturating_sub(1);
        state.live_bytes = state.live_bytes.saturating_sub(size);

        // Always parks. There is deliberately no "the GPU is idle, hand it back
        // now" fast path: the epoch a release is stamped with is the one still
        // open, and an open epoch has by definition not completed, so such a
        // branch would be unreachable. Anything that would have taken it is
        // instead swept up by the next drain, which is O(1) amortized -- a
        // `VecDeque` push here against a `BTreeMap` entry push there, so the
        // branch would not have bought anything measurable either.
        state.counters.deferred += 1;
        state.pending.push_back(PendingBuffer {
            buffer,
            size,
            epoch,
        });
        // No gauge to update: how many buffers are parked is `state.pending`
        // itself, which `occupancy()` reports (issue #507). Mirroring it into
        // `PoolCounters` is what let `reset_counters` zero it.
    }

    /// Moves everything whose epoch has completed into the free lists.
    ///
    /// Called from a command buffer's completion handler -- **once per command
    /// buffer, not once per buffer**. A handler per pooled buffer would put
    /// thousands of block allocations on the decode path; this puts one, and it
    /// releases every buffer that command buffer was holding up.
    ///
    /// `pending` is ordered by epoch, so this stops at the first entry still
    /// outstanding rather than walking the rest.
    fn drain_completed(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let mut drained = 0u64;
        while let Some(front) = state.pending.front() {
            if !self.clock.is_complete(front.epoch) {
                break;
            }
            let entry = state
                .pending
                .pop_front()
                .expect("front() just returned Some");
            // The buffer is now in the free list: no caller holds it and the
            // GPU epoch it was released in has completed, so *nothing* may
            // touch it until `acquire` hands it out again. Until #475 that was
            // a convention no tool could check; poisoning makes a host read of
            // a free buffer a reported error instead of a legal read of a live
            // allocation.
            //
            // SAFETY: this is the strongest point in the buffer's life for the
            // claim -- the last handle dropped (`PooledBuffer::drop` is what
            // put it in `pending`) AND `is_complete` just said the GPU is done
            // with it. Inert without the `asan-poison` feature.
            unsafe { super::poison::poison_buffer(&entry.buffer) };
            state.free_bytes += entry.size;
            state.free_order.push_back(entry.size);
            state.free.entry(entry.size).or_default().push(entry.buffer);
            drained += 1;
        }
        if drained > 0 {
            state.counters.drained += drained;
            let evicted = state.evict_over_budget();
            // Drop the lock before telling the observer, and tell it before the
            // buffers are destroyed: `evicted` still owns them here, so the
            // residency set is removing an allocation that provably still
            // exists (`DESIGN.md` §6.3c).
            drop(state);
            self.notify_evicted(&evicted);
        }
    }

    /// Tells the eviction observer about buffers the pool is destroying.
    ///
    /// (See `impl Drop for PoolInner` for the fourth destruction route, which
    /// is the pool itself going away with buffers still in its free list.)
    ///
    /// Takes them by reference and lets the caller drop them afterwards, so the
    /// `MTLBuffer` is still alive while `removeAllocation` runs. Removing an
    /// allocation that has already been freed is the exact operation
    /// `IOGPUGroupMemory::remove_memory_object()` aborts the machine over, so
    /// the order here is the point rather than a detail.
    fn notify_evicted(&self, evicted: &[Buffer]) {
        if evicted.is_empty() {
            return;
        }
        if let Some(observer) = self.eviction_observer() {
            observer.on_evict(evicted);
        }
    }
}

/// Clears every poison mark the pool still holds before its buffers are freed
/// (#475).
///
/// **The fourth destruction route, and the one that made the discipline
/// structural rather than a list of call sites.** A pool going away takes its
/// free list and its pending list with it, and both can hold marked buffers.
///
/// Metal SUB-ALLOCATES — `newBufferWithLength:` is served by
/// `allocBufferSubDataWithLength`, which hands back address ranges a destroyed
/// buffer occupied and `bzero`s them — so a mark that outlives its mapping
/// reports Metal zeroing an unrelated *fresh* allocation. That is a **false
/// positive**, which discredits the instrument rather than merely missing a
/// defect, and it is what the pool's own test suite hit while this was being
/// built.
///
/// Inert without `asan-poison`: the loops run over lists that are about to be
/// dropped anyway and the calls compile to nothing, so this costs the shipping
/// build one empty `Drop`.
impl Drop for PoolInner {
    fn drop(&mut self) {
        if !super::poison::ENABLED {
            return;
        }
        let Ok(state) = self.state.lock() else {
            // Poisoned lock: another thread panicked holding it. Leaving marks
            // behind is the safe direction here — a false positive later is
            // worse than none, but reaching into a `Mutex` whose invariants may
            // be broken is worse than both.
            return;
        };
        for bucket in state.free.values() {
            for buffer in bucket {
                // SAFETY: the pool is being destroyed, so nothing can acquire
                // these; they are freed as `state` drops.
                unsafe { super::poison::unpoison_buffer(buffer) };
            }
        }
        for entry in &state.pending {
            // Pending buffers are not poisoned today (they are marked only on
            // entering the free list), but clearing is idempotent and this is
            // the site that must not be re-derived if that changes.
            unsafe { super::poison::unpoison_buffer(&entry.buffer) };
        }
    }
}

impl PoolState {
    /// Bytes this pool holds that no planner owns (`DESIGN.md` §9.5k).
    ///
    /// `live + free + pending − planned`. All three terms are bytes the process
    /// holds and the OS cannot have, and **`pending_bytes` is the one that
    /// matters**: `free_bytes` is capped by `free_budget`, and `pending_bytes`
    /// is capped by nothing — it is §6.3b's stranding at ~21 MB/token, 8.398 GB
    /// over 400 tokens, the largest unbounded term this project has observed.
    ///
    /// The subtraction **saturates**, and that is not defensive tidiness: early
    /// in a run the pool holds far less than the planned set because the
    /// weights are still loading, so a wrapping subtraction would report a
    /// colossal figure and refuse a run that is fine.
    ///
    /// `pending` is summed rather than maintained incrementally. It is a walk
    /// of a list the lock already holds, it happens only on a genuine
    /// allocation (never on a pool hit), and `occupancy()` already sums it the
    /// same way — maintaining a fifth counter to avoid it would be hot-path
    /// work whose only consumer is this branch.
    fn unplanned_bytes(&self, cap: ResidualCap) -> usize {
        let held = self.live_bytes
            + self.free_bytes
            + self
                .pending
                .iter()
                .map(|p| p.buffer.length())
                .sum::<usize>();
        held.saturating_sub(cap.planned)
    }

    /// Drops the oldest free buffers until the free list is back inside its
    /// byte budget.
    ///
    /// **Why a bound is needed at all**, when the pool did not have one before:
    /// under CPU-drop release a buffer re-entered the free list *within* the
    /// operation that freed it, so a workload asking for a slightly larger
    /// buffer each step -- a growing KV cache, which is exactly LFM2 decode --
    /// reused each size before moving on to the next. Deferring the return to
    /// GPU completion moves it past that point, so the size just freed is never
    /// the size next requested and every one of those buffers is stranded.
    /// Measured on LFM2 without this: 11.6 stranded buffers per token, taking
    /// the pool from 5231 MB to 13629 MB at 400 tokens and still climbing.
    ///
    /// That is the same unbounded-growth shape issue #21 removed from the
    /// bucket keys, so it must not be reintroduced in the buffers themselves.
    ///
    /// Note what this is *not*: it is not the old sweep. That existed to
    /// **discover** which buffers were free, by testing `strong_count` on every
    /// one of them, and bounded the pool only as a side effect. Discovery is an
    /// event here. This only enforces the bound; it never searches for anything,
    /// and it never touches a buffer that is live or still pending.
    ///
    /// Under [`EvictionOrder::Age`] -- what ships -- oldest-first, because a
    /// size that has not been asked for since it was freed is the least likely
    /// to be asked for again; in the growing case, provably never.
    ///
    /// Under [`EvictionOrder::Size`] (§9.2w, #47) largest-first, because
    /// `free_budget` caps *bytes*: one 3.83 MB KV discard releases as much
    /// budget as fifty-two 76 KB activation buckets, so taking the largest
    /// reaches the same `free_bytes` in fewer evictions and leaves the small
    /// buckets alive. That is the state §9.2v's bound reached by refusing
    /// reuse, arrived at by retention instead -- and `oversize_declined` stays
    /// at 0, which is what tells the two apart.
    ///
    /// Returns the buffers it removed rather than dropping them here, so the
    /// caller can hand them to the eviction observer **before** they are
    /// destroyed (`DESIGN.md` §6.3c). Dropping them inside this function would
    /// free the `MTLBuffer` while the residency set still lists it, which is the
    /// same defect the teardown guard exists to prevent, one clock earlier.
    #[must_use]
    fn evict_over_budget(&mut self) -> Vec<Buffer> {
        let mut evicted = Vec::new();
        while self.free_bytes > self.free_budget {
            let Some((size, buffer)) = self.take_eviction_victim() else {
                break;
            };
            self.free_bytes = self.free_bytes.saturating_sub(size);
            self.counters.evicted += 1;
            // Leaving the free list to be DESTROYED rather than reused, so the
            // mark must come off before the `MTLBuffer` is freed. A stale
            // poison would outlive the allocation and sit over address space
            // Metal is free to hand back for something else -- turning a later
            // correct access into a reported error, which is a false positive
            // and the direction that discredits the whole instrument (#475).
            //
            // SAFETY: `take_eviction_victim` just removed it from the free list
            // under the pool lock and nothing has handed it out. Inert without
            // `asan-poison`.
            unsafe { super::poison::unpoison_buffer(&buffer) };
            evicted.push(buffer);
        }
        evicted
    }

    /// Removes one buffer from the free list under the arm in force, returning
    /// its bucket size (`DESIGN.md` §9.2w, #47).
    ///
    /// Returns `None` only when the free list holds nothing evictable, which
    /// stops `evict_over_budget`'s loop. **`free_bytes` is not adjusted here** —
    /// the caller does that, so the two arms cannot disagree about the
    /// accounting.
    ///
    /// # Neither arm scans
    ///
    /// `Age` pops `free_order`'s front and skips stale entries, as it always
    /// has. `Size` reads `free`'s **last key** — a `BTreeMap`, so the largest
    /// bucket is `O(log n)` and there is no walk. Both stay off §3.3's
    /// no-O(N)-on-the-per-token-path list, and this runs in the completion
    /// handler rather than on the encode path either way (§6.3f).
    fn take_eviction_victim(&mut self) -> Option<(usize, Buffer)> {
        match self.eviction_order {
            EvictionOrder::Age => loop {
                let size = self.free_order.pop_front()?;
                let bucket = match self.free.get_mut(&size) {
                    Some(b) => b,
                    // The key is gone: its bucket emptied and was removed.
                    None => continue,
                };
                // Stale entry: a lookup already took this buffer, so there is
                // nothing here to evict and the queue entry is just noise.
                let Some(buffer) = bucket.pop() else {
                    continue;
                };
                if bucket.is_empty() {
                    self.free.remove(&size);
                }
                return Some((size, buffer));
            },
            EvictionOrder::Size => {
                // `last_key_value` is the largest bucket. A bucket present in
                // `free` always holds at least one buffer -- the key is removed
                // when it empties, here and in `acquire` -- so the `pop` below
                // cannot observe an empty `Vec`, and there is no stale-entry
                // case to skip because this arm reads the map rather than the
                // queue.
                let size = *self.free.last_key_value()?.0;
                let bucket = self.free.get_mut(&size)?;
                let buffer = bucket.pop()?;
                if bucket.is_empty() {
                    self.free.remove(&size);
                }
                // `free_order`'s entry for this buffer is left behind and
                // skipped if the arm ever changes -- the same treatment
                // `acquire` already gives an entry whose buffer it took, and
                // the reason removing it would be a linear search.
                Some((size, buffer))
            }
        }
    }
}

/// The device-side buffer pool.
///
/// Cheap to clone; every clone refers to the same pool.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// A pool with a private clock that nothing advances.
    ///
    /// Every release then parks, because no epoch ever completes. That is the
    /// safe direction to fail, but it is not useful for a real device: use
    /// [`Self::with_clock`] with the clock `Commands` advances.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(GpuClock::new()))
    }

    /// A pool that decides reuse against `clock`.
    pub fn with_clock(clock: Arc<GpuClock>) -> Self {
        Self {
            inner: Arc::new(PoolInner::new(clock)),
        }
    }

    /// Installs the observer told when this pool destroys a free buffer.
    ///
    /// Call during device setup, before any allocation: an observer installed
    /// later would miss buffers already evicted, and for the residency set that
    /// means an allocation it still lists whose handle is gone
    /// (`DESIGN.md` §6.3c). Replaces any previous observer.
    pub fn set_eviction_observer(&self, observer: Arc<dyn BufferEvictionObserver>) {
        if let Ok(mut slot) = self.inner.on_evict.lock() {
            *slot = Some(observer);
        }
    }

    /// Returns every buffer whose epoch has completed to its free list.
    ///
    /// Call from a command buffer completion handler, once per command buffer.
    pub fn drain_completed(&self) {
        self.inner.drain_completed();
    }

    /// Selects the reuse-oversize bound for this pool (`DESIGN.md` §9.2v, #47).
    ///
    /// `None` restores the unbounded behaviour that ships. Every pool is
    /// constructed at [`reuse_oversize_bound`]'s arm already, so **production
    /// does not call this** -- it exists so a test can exercise a bound the
    /// process-wide `OnceLock` cannot be made to yield, which is the gap that
    /// leaves §9.2r's alignment axis testable only by running a whole model.
    ///
    /// Retroactive by construction and deliberately so: the bound is consulted
    /// at `acquire`, never recorded on a buffer, so tightening it cannot
    /// invalidate a handle already handed out. Nothing is evicted -- a bucket
    /// past the bound is still the right answer for a larger request, and
    /// `evict_over_budget` remains the only thing that destroys a free buffer.
    pub fn set_reuse_oversize_bound(&self, bound: Option<f64>) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.oversize_bound = bound;
        }
    }

    /// Selects the eviction order for this pool (`DESIGN.md` §9.2w, #47).
    ///
    /// [`EvictionOrder::Age`] restores the behaviour that ships. As with
    /// [`Self::set_reuse_oversize_bound`], **production does not call this** —
    /// every pool is constructed at [`eviction_order`]'s arm already, and this
    /// exists so a test can exercise an arm the process-wide `OnceLock` cannot
    /// be made to yield.
    ///
    /// Does **not** evict: the order decides which buffer goes when the budget
    /// is next exceeded, and changing it is not itself a budget event. A caller
    /// wanting both calls [`Self::set_free_budget`] after this.
    pub fn set_eviction_order(&self, order: EvictionOrder) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.eviction_order = order;
        }
    }

    /// The eviction order in force for this pool, for a `RESULT` line.
    ///
    /// Read from the state the eviction path consults, never from the flag, so
    /// a harness reports the arm the process **took** (§7.1a-i).
    pub fn eviction_order(&self) -> EvictionOrder {
        self.inner
            .state
            .lock()
            .map(|s| s.eviction_order)
            .unwrap_or(EvictionOrder::Age)
    }

    /// Sets the cap on bytes retained in the free list, evicting immediately if
    /// the new cap is already exceeded. See `DEFAULT_FREE_BUDGET_BYTES`.
    pub fn set_free_budget(&self, bytes: usize) {
        let evicted = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.free_budget = bytes;
            state.evict_over_budget()
        };
        // Outside the lock, and before `evicted` is dropped -- see
        // `PoolInner::notify_evicted`.
        self.inner.notify_evicted(&evicted);
    }

    /// Installs `DESIGN.md` §9.5k's derived cap on **unplanned** bytes.
    ///
    /// `limit` is `budget − predicted` and `planned` is the part of the
    /// predicted set this pool serves — weights, KV, scratch and conv, all of
    /// which are already inside `live_bytes`. Both come from admission, which
    /// computed them before anything was allocated.
    ///
    /// **Call once, at configuration time.** Installing a cap mid-run is
    /// permitted and is what a test does, but in a real process the numbers are
    /// admission's and admission runs once.
    ///
    /// # This is one branch on one class, deliberately
    ///
    /// §9.5e declines per-allocation checking **as a general gate** on a
    /// correctness argument, and that argument stands for the classes admission
    /// knows. This bounds only the class admission *derives* — see
    /// [`ResidualCap`]. It calls into Metal nowhere and it cannot fire if
    /// admission was honest.
    pub fn set_residual_cap(&self, limit: usize, planned: usize) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.residual = Some(ResidualCap { limit, planned });
        }
    }

    /// Removes the residual cap, restoring the unbounded behaviour that
    /// shipped.
    pub fn clear_residual_cap(&self) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.residual = None;
        }
    }

    /// Bytes this pool holds that no planner owns, or `None` with no cap
    /// installed.
    ///
    /// `live + free + pending − planned`. See [`ResidualCap::planned`] for why
    /// the subtraction is there and what comparing `live_bytes` alone would do.
    pub fn unplanned_bytes(&self) -> Option<usize> {
        let state = self.inner.state.lock().ok()?;
        let cap = state.residual?;
        Some(state.unplanned_bytes(cap))
    }

    /// Takes a reusable buffer of at least `size` bytes, if one is free.
    ///
    /// Probes the exact-size bucket first, since an exact match cannot be
    /// improved on, then the smallest larger bucket. At most two ordered map
    /// probes, and buckets with nothing free are never looked at.
    ///
    /// `None` means the caller should allocate; it does **not** mean the pool
    /// holds nothing of that size, only that nothing of that size is free.
    pub fn acquire(&self, size: usize) -> Option<Arc<PooledBuffer>> {
        let mut state = self.inner.state.lock().ok()?;
        state.counters.lookups += 1;

        // Exact hit. Checked separately from the range query below because it
        // is the common case in decode -- every token asks for the same shapes
        // -- and because an exact match is optimal, so there is nothing to gain
        // by looking further.
        state.counters.buckets_probed += 1;
        let found = if let Some(bucket) = state.free.get_mut(&size) {
            bucket.pop().map(|b| (b, size))
        } else {
            None
        };

        // Otherwise the smallest bucket that can satisfy the request. One
        // ordered probe: `range` starts at the first key > size and stops.
        let found = match found {
            Some(hit) => Some(hit),
            None => {
                state.counters.buckets_probed += 1;
                // The reuse-oversize bound (§9.2v, #47), applied to the
                // candidate *before* it is popped. `range` yields the smallest
                // key above the request, so if that one is past the bound then
                // every other candidate is further past it -- the ordering is
                // what makes one comparison sufficient and keeps this off
                // §3.3's no-O(N) list.
                //
                // Declining leaves the buffer in the free list rather than
                // destroying it: it is still the right answer for a larger
                // request, and evicting on this path would make a lookup
                // policy into a retention policy. `evict_over_budget` remains
                // the only thing that removes a free buffer.
                let bound = state.oversize_bound;
                let next = state
                    .free
                    .range_mut((std::ops::Bound::Excluded(size), std::ops::Bound::Unbounded))
                    .next()
                    .filter(|(k, _)| bucket_admissible(size, **k, bound))
                    .and_then(|(k, v)| v.pop().map(|b| (b, *k)));
                if next.is_none() && bound.is_some() {
                    // Only when a candidate existed and the bound is what
                    // rejected it. An empty range yields `None` too, and
                    // counting that would make this a miss counter rather than
                    // the arm's engagement proof (#500).
                    let had_candidate = state
                        .free
                        .range((std::ops::Bound::Excluded(size), std::ops::Bound::Unbounded))
                        .next()
                        .is_some();
                    if had_candidate {
                        state.counters.oversize_declined += 1;
                    }
                }
                next
            }
        };

        let (buffer, bucket_size) = found?;

        // The buffer has left the free list. Its `free_order` entry is left
        // behind and skipped when eviction reaches it -- removing it here would
        // mean a linear search, which is exactly what must not be on this path.
        state.free_bytes = state.free_bytes.saturating_sub(bucket_size);

        // Drop the key when its bucket empties. This is what stops the map
        // from becoming a graveyard: the old pool's `retain` cleared a bucket's
        // Vec but left the key, so emptied keys accumulated at ~3.2 per decode
        // token and were walked forever.
        if state
            .free
            .get(&bucket_size)
            .is_some_and(|bucket| bucket.is_empty())
        {
            state.free.remove(&bucket_size);
        }

        // Handed to a caller, so it is theirs to touch again. This must happen
        // before the handle can be observed, or the caller's first legitimate
        // access is what gets reported (#475).
        //
        // SAFETY: the buffer has just left the free list under the pool lock,
        // so no other handle to it exists. Inert without `asan-poison`.
        unsafe { super::poison::unpoison_buffer(&buffer) };

        state.counters.hits += 1;
        // The overhang's HIT source (§9.2). `bucket_size` is what the range
        // query found and `size` is what was asked for; when the exact probe
        // above missed, the second probe deliberately returns a *larger*
        // bucket, and the difference is bytes this caller holds and does not
        // want. Recorded here rather than at `adopt` because on a hit no
        // allocation happens and `adopt` is never reached.
        state.counters.requested_bytes += size as u64;
        state.counters.served_bytes += bucket_size as u64;
        // Banded by the REQUEST (§9.2u). Keying on `bucket_size` would file
        // every over-served acquisition under the band of the bucket that
        // answered it -- which is the band the overhang would then appear to
        // belong to, true by construction and therefore evidence of nothing.
        state.counters.record_band(size, bucket_size);
        state.live_buffers += 1;
        state.live_bytes += bucket_size;
        Some(Arc::new(PooledBuffer {
            buffer: Some(buffer),
            size: bucket_size,
            pool: Arc::downgrade(&self.inner),
        }))
    }

    /// Whether allocating `size` fresh bytes would take this pool past
    /// `DESIGN.md` §9.5k's derived residual.
    ///
    /// **This is the one branch, and it is asked before the buffer is
    /// created** — checking after `newBufferWithLength` would report an
    /// overrun having already committed it, which is the shape that makes an
    /// error useless. `Ok(())` with no cap installed, which is what shipped.
    ///
    /// # It cannot fire if admission was honest
    ///
    /// `unplanned` counts bytes the pool holds that no planner owns, so a
    /// process whose predicted classes are what it actually allocates never
    /// approaches the limit. What it bounds is the case admission cannot see:
    /// an allocation path growing without bound at runtime (§9.5f's second
    /// candidate explanation), of which §6.3b's stranding is the one instance
    /// that has been measured — 11.6 buffers and ~21 MB per token, capped by
    /// nothing before this.
    ///
    /// # What it does NOT do
    ///
    /// It calls into Metal nowhere. `live_bytes` and `free_bytes` are already
    /// maintained incrementally (*"so the budget check never has to walk the
    /// buckets"*), and the comparison is one branch on integers. #167's
    /// ablation is the premise: **accounting is cheap, calling into Metal is
    /// not** (§6.3d) — an eager `removeAllocation` per buffer cost
    /// +0.093 ms/token and the same bookkeeping with the Metal call skipped
    /// read baseline.
    pub fn check_residual(&self, size: usize) -> Result<(), ResidualExceeded> {
        let Ok(state) = self.inner.state.lock() else {
            // Poisoned. Refusing here would turn one thread's panic into a
            // failure to allocate for every other, which is a worse outcome
            // than the unbounded behaviour that shipped.
            return Ok(());
        };
        let Some(cap) = state.residual else {
            return Ok(());
        };
        let unplanned = state.unplanned_bytes(cap);
        let would_be = unplanned.saturating_add(size);
        if would_be > cap.limit {
            return Err(ResidualExceeded {
                unplanned,
                requested: size,
                limit: cap.limit,
                planned: cap.planned,
            });
        }
        Ok(())
    }

    /// Wraps a freshly created `Buffer` in a pool handle.
    ///
    /// `size` must be the buffer's allocated length, since that is the bucket
    /// it returns to.
    ///
    /// # The pool takes allocations, never views (issue #507)
    ///
    /// `acquire` keys on **size alone**. Two free buffers of one size are
    /// interchangeable *because each owns its whole `MTLBuffer`*; that premise
    /// is what makes an O(1) bucket lookup correct, and §6.3a chose it
    /// deliberately. A [view](Buffer::view) breaks it: N views of one allocation
    /// are N distinct handles at **one base address**, so handing two of them
    /// out aliases two unrelated tensors onto one region -- the #23 corruption
    /// class, under `HazardTrackingModeUntracked` where §9.3 says there is no
    /// safety net and §2.3.2 says the result is indistinguishable from benign
    /// float noise.
    ///
    /// **This is checked here rather than at `acquire`, and that is the design
    /// decision rather than an implementation detail.** The two are not
    /// equivalent placements of one check:
    ///
    /// - `acquire` is the **hot** path -- §6.3b measures ~11.6 buffers destroyed
    ///   per token -- so a check there pays on every lookup, forever, to
    ///   re-discover a property that is decided once, when the buffer enters.
    /// - `adopt` is **rare**: it is reached only on a pool *miss*, where the
    ///   caller has just called `newBufferWithLength`, which costs orders of
    ///   magnitude more than a bool test on a field already in cache.
    /// - and a check at `acquire` fires at the **wrong instant**. It reports the
    ///   aliasing when the second handle is handed out, with nothing to say who
    ///   put the buffer in the free list; here it fires at the site that
    ///   violated the invariant, which is the one that has to change.
    ///
    /// **The invariant holds by construction today**, audited for #507: both
    /// production callers (`MetalDevice::new_buffer` and
    /// `MetalDevice::allocate_buffer`) pass the direct result of
    /// `device.new_buffer(...)`, and the only `Buffer::view` producer in the
    /// tree is `Arena::new`, which builds [`PooledBuffer::unpooled`] and is no
    /// longer handed a pool at all (§9.2m). So this guards a property that is
    /// currently true rather than repairing one that is not -- #505 fixed the
    /// single path that violated it, and what it left behind was that **nothing
    /// prevented the next one**, which is that section's own closing question.
    ///
    /// A `debug_assert!` because the property is already enforced by a
    /// *signature* at the only site that could break it, so this is the
    /// backstop rather than the mechanism; and because a violation is a
    /// programming error rather than a resource one, where §9.5e's argument
    /// against panicking deep inside a forward pass applies.
    pub fn adopt(&self, buffer: Buffer, size: usize) -> Arc<PooledBuffer> {
        debug_assert!(
            !buffer.is_view(),
            "BufferPool::adopt was given a VIEW ({} bytes at offset {} of a \
             larger allocation). `acquire` keys on size alone, so this handle \
             and every other view of the same allocation would be handed out as \
             though they were independent buffers -- lloom #23's corruption \
             class, and the defect #489 measured on the arena: five \"separate\" \
             activation buffers at one base address. A handle that must not \
             participate in reuse wants `PooledBuffer::unpooled`, which exists \
             for exactly this case (DESIGN.md §9.2a layer 3, §9.2m).",
            buffer.length(),
            buffer.base_offset(),
        );
        if let Ok(mut state) = self.inner.state.lock() {
            state.counters.allocations += 1;
            state.counters.allocated_bytes += size as u64;
            state.live_buffers += 1;
            state.live_bytes += size;
        }
        Arc::new(PooledBuffer {
            buffer: Some(buffer),
            size,
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// [`Self::adopt`], also recording what the caller originally asked for.
    ///
    /// The overhang's MISS source (`DESIGN.md` §9.2, #47). `buf_size` rounds a
    /// request up to the buffer alignment and then **shadows the variable**, so
    /// by the time `adopt` is called the original figure is gone at both
    /// production sites (`device.rs:1050`, `device.rs:1183`). This takes both
    /// numbers so the difference can be counted at the one instant it is still
    /// known.
    ///
    /// # Why a sibling rather than a wider `adopt`
    ///
    /// `adopt` has two production callers and eleven in tests, probes and
    /// fuzzers. Widening its signature would edit thirteen sites to instrument
    /// two, and every test call would have to invent a "requested" figure it
    /// does not model -- which is how a counter acquires a population that is
    /// not the one it claims to measure. `adopt` keeps its meaning (*take
    /// ownership of this allocation*) and this adds the accounting the decode
    /// path needs.
    ///
    /// `requested` is the pre-rounding byte count; `allocated` is what was
    /// handed to `newBufferWithLength` and what the bucket is keyed on.
    pub fn adopt_requesting(
        &self,
        buffer: Buffer,
        requested: usize,
        allocated: usize,
    ) -> Arc<PooledBuffer> {
        debug_assert!(
            allocated >= requested,
            "adopt_requesting({requested} -> {allocated}) shrank the request; \
             `buf_size` only ever rounds up, so a smaller allocation than the \
             request means the two arguments were passed in the wrong order \
             and the overhang would underflow (DESIGN.md §9.2)."
        );
        let handle = self.adopt(buffer, allocated);
        if let Ok(mut state) = self.inner.state.lock() {
            state.counters.requested_bytes += requested as u64;
            state.counters.served_bytes += allocated as u64;
            // The MISS half of the same split, banded on the same key as the
            // hit half so the two are one population (§9.2u).
            state.counters.record_band(requested, allocated);
        }
        handle
    }

    /// Drops every currently-free buffer, returning them so the caller can
    /// unregister them from the residency set.
    ///
    /// This is the sweep, and it is now a **rare trim under memory pressure**
    /// rather than a step on the allocation path. It cannot race a live buffer:
    /// the free list holds only buffers whose last handle has already dropped.
    ///
    /// **Pending buffers are deliberately left alone.** They have been released
    /// by the CPU but the GPU may still be reading or writing them, so freeing
    /// one here would be the same use-after-free the deferral exists to
    /// prevent -- with destruction in place of aliasing. Anything genuinely
    /// finished has already been drained into a free list and is taken by the
    /// loop below; the rest is reclaimed by a later trim.
    pub fn trim(&self) -> Vec<Buffer> {
        // Sweep in anything the GPU has since finished, so a trim after a
        // synchronize reclaims what that synchronize made safe.
        self.inner.drain_completed();

        let mut freed = Vec::new();
        if let Ok(mut state) = self.inner.state.lock() {
            for (_, mut bucket) in std::mem::take(&mut state.free) {
                // Destroyed rather than reused, as in `evict_over_budget`: the
                // mark comes off before the allocation goes, or it outlives the
                // buffer and reports a later correct access (#475).
                //
                // SAFETY: these have just been taken out of the free list under
                // the pool lock, so nothing holds them. Inert without
                // `asan-poison`.
                for buffer in &bucket {
                    unsafe { super::poison::unpoison_buffer(buffer) };
                }
                freed.append(&mut bucket);
            }
            state.free_bytes = 0;
            state.free_order.clear();
            state.counters.trims += 1;
            state.counters.trimmed += freed.len() as u64;
        }
        freed
    }

    pub fn counters(&self) -> PoolCounters {
        self.inner
            .state
            .lock()
            .map(|s| s.counters)
            .unwrap_or_default()
    }

    /// Zeroes the event counters, so a caller can measure decode rather than
    /// decode-plus-load.
    ///
    /// **Every field of [`PoolCounters`] is an accumulator**, so zeroing the
    /// struct is exactly what resetting it means and this cannot disturb any
    /// figure describing the pool's current state. That was not true until
    /// #507: `pending` was a gauge living among the accumulators, and this call
    /// zeroed it while the buffers it counted were still parked. The live
    /// figures are [`Self::occupancy`]'s, and this does not touch them.
    pub fn reset_counters(&self) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.counters = PoolCounters::default();
        }
    }

    /// `(live buffers, free buffers, free bytes, non-empty free buckets)`.
    ///
    /// "live" counts handles still held by a caller; "free" counts buffers
    /// sitting in the free list ready to be handed out.
    pub fn occupancy(&self) -> PoolOccupancySnapshot {
        let Ok(state) = self.inner.state.lock() else {
            return PoolOccupancySnapshot::default();
        };
        let free_buffers = state.free.values().map(|b| b.len()).sum();
        PoolOccupancySnapshot {
            live_buffers: state.live_buffers,
            live_bytes: state.live_bytes,
            free_buffers,
            // The maintained figure, not a re-walk of the buckets: it is what
            // the budget is enforced against, so reporting anything else would
            // let the two disagree silently.
            free_bytes: state.free_bytes,
            free_buckets: state.free.len(),
            pending_buffers: state.pending.len(),
            pending_bytes: state.pending.iter().map(|p| p.buffer.length()).sum(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PoolOccupancySnapshot {
    pub live_buffers: usize,
    pub live_bytes: usize,
    pub free_buffers: usize,
    pub free_bytes: usize,
    /// Free-list buckets that hold at least one buffer.
    ///
    /// By construction there are no others: a bucket is removed when it
    /// empties. This is the number that grew without bound before.
    pub free_buckets: usize,
    /// Released by the CPU, still waiting on the GPU. Not yet reusable.
    ///
    /// This is the memory cost of deciding reuse on the GPU clock: a buffer
    /// counted here would have been immediately reusable under the old
    /// predicate, and immediately *corruptible* with it.
    pub pending_buffers: usize,
    pub pending_bytes: usize,
}

impl PoolOccupancySnapshot {
    pub fn total_buffers(&self) -> usize {
        self.live_buffers + self.free_buffers + self.pending_buffers
    }

    pub fn total_bytes(&self) -> usize {
        self.live_bytes + self.free_bytes + self.pending_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::metal::Device;

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
    }

    /// A pool plus the clock driving it, so a test can say when the GPU
    /// finishes rather than depending on a real command queue.
    fn pool_with_clock() -> (BufferPool, Arc<GpuClock>) {
        let clock = Arc::new(GpuClock::new());
        (BufferPool::with_clock(Arc::clone(&clock)), clock)
    }

    /// Stands in for `Commands`: closes the open epoch, reports it finished,
    /// and sweeps -- what a command buffer commit and its completion handler do
    /// between them.
    fn gpu_completes(pool: &BufferPool, clock: &GpuClock) {
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();
    }

    /// Allocates through the pool the way `MetalDevice` does: try to reuse,
    /// otherwise create and adopt.
    fn alloc(pool: &BufferPool, dev: &Device, size: usize) -> Arc<PooledBuffer> {
        if let Some(b) = pool.acquire(size) {
            return b;
        }
        let raw = dev
            .new_buffer(size, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        pool.adopt(raw, size)
    }

    /// Drops `b` and lets the GPU finish, so it is available for reuse. The
    /// two-step is the point of this change: a drop alone no longer suffices.
    fn release_and_complete(pool: &BufferPool, clock: &GpuClock, b: Arc<PooledBuffer>) {
        drop(b);
        gpu_completes(pool, clock);
    }

    /// Allocates the way `MetalDevice` does **with the residual check in
    /// place**: reuse if possible, else check, else create and adopt.
    ///
    /// The ordering mirrors `device.rs` exactly -- the check sits on the pool
    /// miss and before the allocation -- so a test exercising this exercises
    /// the shape that ships.
    fn alloc_checked(
        pool: &BufferPool,
        dev: &Device,
        size: usize,
    ) -> std::result::Result<Arc<PooledBuffer>, ResidualExceeded> {
        if let Some(b) = pool.acquire(size) {
            return Ok(b);
        }
        pool.check_residual(size)?;
        let raw = dev
            .new_buffer(size, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        Ok(pool.adopt(raw, size))
    }

    /// **Both bounds** (`DESIGN.md` §8.1g, and #184's precedent): a check that
    /// refuses everything is not a check.
    ///
    /// Asserted in one test so neither arm can be dropped without the other
    /// going red. The admitted arm is not incidental -- it is half the result.
    #[test]
    fn residual_admits_inside_the_cap_and_refuses_past_it() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();
        // 1 MiB of residual, and nothing planned in this pool.
        pool.set_residual_cap(1024 * 1024, 0);

        // ADMITTED: comfortably inside.
        let a = alloc_checked(&pool, &dev, 256 * 1024).expect("inside the cap must be admitted");
        assert_eq!(pool.unplanned_bytes(), Some(256 * 1024));

        // Still admitted: exactly at the cap, since the test is `>` not `>=`.
        let b = alloc_checked(&pool, &dev, 768 * 1024).expect("exactly at the cap fits");
        assert_eq!(pool.unplanned_bytes(), Some(1024 * 1024));

        // REFUSED: one byte past it.
        let err = alloc_checked(&pool, &dev, 1).expect_err("past the cap must be refused");
        assert_eq!(err.limit, 1024 * 1024);
        assert_eq!(err.unplanned, 1024 * 1024);
        assert_eq!(err.requested, 1);
        drop((a, b));
    }

    /// **The check is inert with no cap installed**, which is what shipped.
    ///
    /// This is the off-arm, and it is what makes the change safe to land: a
    /// process that never runs admission is byte-for-byte what it was.
    #[test]
    fn no_cap_installed_means_no_refusal() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();
        assert_eq!(pool.unplanned_bytes(), None, "no cap installed");
        for _ in 0..8 {
            alloc_checked(&pool, &dev, 1024 * 1024).expect("uncapped pool never refuses");
        }
        assert!(pool.check_residual(usize::MAX / 2).is_ok());
    }

    /// **The planned classes are subtracted, so the weights are not refused.**
    ///
    /// §9.5k's load-bearing correction: weights land in `private_buffers` and
    /// KV in `buffers`, so both are already inside `live_bytes`. A check
    /// comparing `live_bytes` against the residual would refuse them.
    ///
    /// The mutation this kills is the *obvious* form of the check, and it is
    /// the one the issue calls out by name.
    #[test]
    fn planned_bytes_are_subtracted_so_the_weights_are_not_refused() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();
        let planned = 4 * 1024 * 1024;
        // A residual much smaller than the planned set -- exactly the real
        // shape, where 5.4 GB of weights sit under a 3.5 GB residual.
        pool.set_residual_cap(1024 * 1024, planned);

        // Allocate the whole planned set. Under a `live_bytes >= limit` check
        // every one of these would be refused.
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(
                alloc_checked(&pool, &dev, 1024 * 1024)
                    .expect("a planned allocation must never be refused"),
            );
        }
        assert_eq!(
            pool.unplanned_bytes(),
            Some(0),
            "the planned set contributes ZERO unplanned bytes"
        );
        // And the residual still bounds what comes after it.
        alloc_checked(&pool, &dev, 1024 * 1024).expect("the first unplanned MiB fits");
        alloc_checked(&pool, &dev, 1).expect_err("the residual still bounds the rest");
    }

    /// **`pending_bytes` counts**, and it is the term that matters.
    ///
    /// §6.3b's stranding is `pending`: 11.6 buffers/token, ~21 MB/token,
    /// capped by nothing. `free_bytes` is capped by `set_free_budget`. A check
    /// counting only `live_bytes` would miss the one unbounded term, so this
    /// releases a buffer *without* completing the GPU -- which parks it in
    /// `pending` -- and asserts it is still counted.
    #[test]
    fn pending_and_free_bytes_are_counted_not_only_live() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_residual_cap(4 * 1024 * 1024, 0);

        let b = alloc_checked(&pool, &dev, 2 * 1024 * 1024).expect("first fits");
        // Release without completing: the buffer parks in `pending`, which is
        // where §6.3b's stranding lives.
        drop(b);
        assert_eq!(
            pool.unplanned_bytes(),
            Some(2 * 1024 * 1024),
            "a stranded buffer is still bytes the OS cannot have"
        );

        // Now let the GPU finish, moving it to the free list. Still counted.
        gpu_completes(&pool, &clock);
        assert_eq!(
            pool.unplanned_bytes(),
            Some(2 * 1024 * 1024),
            "a free buffer is still held by the process"
        );
    }

    /// A pool hit allocates nothing, so it is not bounded and pays no check.
    #[test]
    fn a_pool_hit_is_not_subject_to_the_cap() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_residual_cap(2 * 1024 * 1024, 0);

        let b = alloc_checked(&pool, &dev, 1024 * 1024).expect("first fits");
        release_and_complete(&pool, &clock, b);

        // Shrink the cap below what the pool already holds. A fresh allocation
        // would now be refused...
        pool.set_residual_cap(0, 0);
        assert!(pool.check_residual(1).is_err());
        // ...but the reuse path never asks, because it allocates nothing.
        assert!(
            alloc_checked(&pool, &dev, 1024 * 1024).is_ok(),
            "reuse must not be refused: it allocates nothing"
        );
    }

    /// The refusal names the quantity that overran and the derived figure.
    ///
    /// §9.5k: the failure *"stays ugly and becomes ugly at a known boundary
    /// rather than as a kernel panic"* -- so the message has to say which
    /// boundary.
    #[test]
    fn refusal_message_names_the_residual_and_the_overrun() {
        let (pool, _clock) = pool_with_clock();
        pool.set_residual_cap(1000, 0);
        let err = pool.check_residual(2000).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("residual exhausted"), "{msg}");
        assert!(msg.contains("§9.5k"), "cite the section: {msg}");
        assert!(msg.contains("§6.3b"), "name the measured instance: {msg}");
    }

    /// **Why the readback arm is not a call-site property** (#536, §11.5c,
    /// §15.2 #12a).
    ///
    /// §11.5c measures `to_cpu`'s blit-free path firing only below 16384 B, and
    /// #350 read that as *"allocate the reduce output under 16 KiB"* -- a
    /// call-site constant. **The reduce output is already 128 B**: the argmax
    /// asks for `dst_el = 1` `U32` (`metal_backend/mod.rs`'s contiguous reduce),
    /// and `buf_size` rounds 4 B to one `BUFFER_ALIGNMENT`. So there is no
    /// call-site constant left to lower, and the sizes §15.2 #12a records --
    /// 12288 B by default, 731136 B under `--sample-in-buffer` -- cannot have
    /// come from the request.
    ///
    /// They come from **here**. `acquire` serves the smallest free bucket that
    /// can satisfy the request, so a 128 B ask is answered by whatever oversized
    /// buffer happens to be free, and the handed-back size is a property of what
    /// the *rest of the step* freed rather than of what the caller asked for.
    /// That is why #454 measures the arm drifting **blit 12 / 1 / 2 / 5 over
    /// four consecutive runs of one binary**: not a configuration, an outcome.
    ///
    /// **Both bounds**, because only the pair distinguishes *serves oversized*
    /// from *never reuses*: an exact-size buffer must be preferred when one is
    /// free (or the test passes on a pool that always allocates fresh).
    #[test]
    fn acquire_serves_an_oversized_bucket_when_no_exact_size_is_free() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // The size a 4-byte argmax output actually asks for: `buf_size(4)`.
        const REDUCE_OUTPUT: usize = 128;
        // Above §15.2 #12a's 16384 B mapping threshold, so a readback served by
        // this bucket takes the blit arm.
        const OVERSIZED: usize = 32768;

        // Free an oversized buffer and nothing else.
        let big = alloc(&pool, &dev, OVERSIZED);
        release_and_complete(&pool, &clock, big);

        let served = pool.acquire(REDUCE_OUTPUT).expect("the pool must reuse");
        assert_eq!(
            served.size(),
            OVERSIZED,
            "a {REDUCE_OUTPUT} B request is served by the smallest free bucket \
             above it, so the readback's arm is decided by what the step freed \
             -- not by the call site's size (#536, §11.5c)"
        );
        assert!(
            served.size() >= 16384,
            "and that is what puts this readback on the blit arm"
        );
        release_and_complete(&pool, &clock, served);

        // The other bound: an exact bucket is preferred when one exists, so the
        // assertion above is about which bucket is chosen and not about the
        // pool refusing to reuse at all.
        //
        // The exact-size buffer is adopted directly rather than through
        // `alloc`, because `alloc` would be served the oversized bucket freed
        // above -- which is this test's own first finding, and would otherwise
        // make the second arm construct a 32768 B buffer and assert about it.
        let raw = dev
            .new_buffer(REDUCE_OUTPUT, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let exact = pool.adopt(raw, REDUCE_OUTPUT);
        release_and_complete(&pool, &clock, exact);
        let served = pool.acquire(REDUCE_OUTPUT).expect("the pool must reuse");
        assert_eq!(
            served.size(),
            REDUCE_OUTPUT,
            "an exact hit is taken in preference to the oversized bucket, which \
             is what makes the assertion above a statement about WHICH bucket \
             is chosen rather than about the pool declining to reuse"
        );
    }

    /// The reuse-oversize bound rejects the candidate the test above accepts
    /// (`DESIGN.md` §9.2v, #47).
    ///
    /// **This is the same 128 B request and the same 32768 B bucket as
    /// `acquire_serves_an_oversized_bucket_when_no_exact_size_is_free`
    /// directly above**, so the pair is one A/B on one population: that test
    /// pins the shipped arm's behaviour, this one pins the bounded arm's, and
    /// the only thing that differs is the bound. Read them together -- a
    /// change that broke the range query entirely would take this test green
    /// and its sibling red, which is why the sibling's assertion is what makes
    /// this one meaningful (§3.1b rule 4, recording the dependency in a
    /// comment where it cannot be made intrinsic).
    ///
    /// The measured motivation is §9.2u's: a 76 KB activation request meeting
    /// a 3.83 MB KV leftover is a **256x** over-service, and 32768/128 is
    /// **256x** exactly -- so this fixture is that pathology at the same ratio
    /// rather than at an arbitrary one.
    #[test]
    fn the_oversize_bound_declines_a_bucket_past_the_ceiling() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        const REQUEST: usize = 128;
        const OVERSIZED: usize = 32768;
        // 256x, the ratio §9.2u measured between a 76 KB activation request
        // and the 3.83 MB KV bucket that answers it.
        assert_eq!(
            OVERSIZED / REQUEST,
            256,
            "the fixture is the measured ratio"
        );

        pool.set_reuse_oversize_bound(Some(4.0));

        let big = alloc(&pool, &dev, OVERSIZED);
        release_and_complete(&pool, &clock, big);

        assert!(
            pool.acquire(REQUEST).is_none(),
            "a {OVERSIZED} B bucket is 256x a {REQUEST} B request, so a 4.0x \
             bound must decline it -- the caller then allocates at its own \
             size, which is what puts a small bucket back in a free list whose \
             small buckets eviction destroyed (§9.2u)"
        );
        assert_eq!(
            pool.counters().oversize_declined,
            1,
            "and the decline is counted, because a bound that never binds and \
             a bound that is not installed produce identical requested/served \
             sums -- this counter is the only thing that separates them (#500)"
        );

        // The bucket is still free: declining is a lookup policy, not a
        // retention policy. A larger request must still be served from it.
        let served = pool
            .acquire(OVERSIZED)
            .expect("the declined bucket is still free for a request it fits");
        assert_eq!(
            served.size(),
            OVERSIZED,
            "declining for a small request must not have destroyed or resized \
             the bucket -- `evict_over_budget` is the only thing that removes a \
             free buffer"
        );
    }

    /// The bound admits what it should: an exact hit and a candidate inside the
    /// ceiling are both unaffected (§9.2v).
    ///
    /// **The non-vacuity arm for the test above.** A bound that rejected
    /// everything would take that test green while destroying reuse entirely,
    /// and the two assertions there cannot see it -- so the ceiling has to be
    /// shown to admit as well as to decline. This is §3.1b rule 2: a shape
    /// needs a second arm producing a value the shape can be checked against.
    #[test]
    fn the_oversize_bound_admits_an_exact_hit_and_a_candidate_inside_it() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        pool.set_reuse_oversize_bound(Some(4.0));

        // An exact match is never affected: `acquire`'s first probe returns
        // before the bound is consulted, and 1.0x is inside any bound >= 1.0.
        const EXACT: usize = 4096;
        let b = alloc(&pool, &dev, EXACT);
        release_and_complete(&pool, &clock, b);
        let served = pool.acquire(EXACT).expect("an exact hit must be served");
        assert_eq!(served.size(), EXACT, "an exact match is not over-service");

        // And a candidate strictly inside the ceiling is served: 8192 is 2.0x
        // a 4096 B request, under the 4.0x bound.
        //
        // The exact-size handle above is deliberately **still held** here
        // rather than released, so the only thing free is the 8192 bucket and
        // the range query is what has to answer. Releasing it first would put
        // a 4096 B buffer back in the free list, the exact probe would return
        // it, and this arm would assert about a path that never consults the
        // bound at all -- which is how a test comes to pass without exercising
        // its subject. (Caught by this test failing on its first version.)
        const INSIDE: usize = 8192;
        let b = alloc(&pool, &dev, INSIDE);
        release_and_complete(&pool, &clock, b);
        let from_range = pool
            .acquire(EXACT)
            .expect("2.0x is inside a 4.0x bound and must be served");
        assert_eq!(
            from_range.size(),
            INSIDE,
            "the bound is a ceiling on the ratio, not a requirement of an \
             exact match -- a change that made it the latter would pass the \
             decline test and destroy every hit on the decode path"
        );
        assert_eq!(
            pool.counters().oversize_declined,
            0,
            "and nothing was declined, which is what makes this the \
             non-vacuity arm rather than a second decline test"
        );
    }

    /// The bound's boundary is inclusive, at the exact byte pair (§9.2v).
    ///
    /// **An absolute reference rather than a shape** (§3.1b rule 1): the
    /// assertion is about two specific sizes whose ratio is exactly the bound,
    /// so an off-by-one in the comparison direction fails here and nowhere
    /// else. `<=` admits; `<` would reject 4x at a 4x bound, which is a
    /// different policy silently.
    #[test]
    fn the_bound_is_inclusive_at_exactly_the_ratio() {
        // The predicate directly, not through a pool: this is arithmetic, and
        // routing it through an allocation would make a comparison test depend
        // on a Metal device.
        assert!(
            bucket_admissible(1024, 4096, Some(4.0)),
            "4096 is exactly 4.0x 1024, and the bound is a ceiling the \
             candidate may reach"
        );
        assert!(
            !bucket_admissible(1024, 4097, Some(4.0)),
            "one byte past it is declined"
        );
        assert!(
            bucket_admissible(1024, 4096, None),
            "and with no bound installed the same pair is admitted, which is \
             the arm that ships"
        );
        // The degenerate input the public predicate has to answer for.
        assert!(
            bucket_admissible(0, 4096, Some(4.0)),
            "a zero-byte request admits rather than rejecting every bucket: \
             `buf_size` never produces one, so this is a malformed question \
             and the safe answer is the unbounded arm's"
        );
    }

    /// An empty free list is a MISS, never a decline (§9.2v).
    ///
    /// **This is what keeps `oversize_declined` an engagement proof rather
    /// than a miss counter.** `range(..).next()` yields `None` both when a
    /// candidate was rejected by the bound and when there was no candidate at
    /// all, and those mean opposite things: the first says the mechanism fired,
    /// the second says the pool was empty. Counting both would make the
    /// counter read nonzero on a run where the bound never bound, which is
    /// exactly the #500 confusion the counter exists to resolve -- so the
    /// re-probe that distinguishes them is load-bearing and this test is the
    /// only thing that can see it removed.
    #[test]
    fn an_empty_free_list_is_a_miss_and_not_a_decline() {
        let (pool, _clock) = pool_with_clock();
        pool.set_reuse_oversize_bound(Some(4.0));

        // Nothing has ever been freed, so the range query finds no candidate.
        assert!(
            pool.acquire(4096).is_none(),
            "an empty pool has nothing to serve"
        );
        assert_eq!(
            pool.counters().oversize_declined,
            0,
            "an empty free list must not increment the decline counter -- \
             otherwise a run whose pool was simply cold would report the bound \
             as having engaged, and the counter would say nothing about the \
             mechanism it exists to prove (#500)"
        );
        // And the miss itself is still counted where misses belong, so this is
        // a statement about WHICH counter moves rather than about nothing
        // happening (§3.1c: a zero is not evidence unless something nonzero
        // came from the same command).
        assert_eq!(
            pool.counters().lookups,
            1,
            "the lookup happened; it was a miss, not a decline"
        );
        assert_eq!(pool.counters().hits, 0, "and it was not a hit");
    }

    /// The axis parses its arms, including the spelled-out default (§9.2v).
    ///
    /// Reads the parse rules through the same predicate the pool uses rather
    /// than through the `OnceLock`, which cannot be re-selected in a test. What
    /// this pins is that `off` and an absent variable mean the same thing --
    /// the property that lets a sweep name the shipped arm instead of
    /// expressing it by unsetting a variable, which a harness renders as
    /// absent (§7.1a-i).
    #[test]
    fn the_unbounded_arm_and_a_generous_bound_are_different_facts() {
        // 512x admits the 256x pathology; the arm exists so a sweep can show a
        // bound that is installed and does not bind, which is a different
        // finding from no bound at all -- and `oversize_declined` is what
        // reports the difference.
        assert!(
            bucket_admissible(128, 32768, Some(512.0)),
            "a bound above the observed ratio admits, so a null result at a \
             generous bound is evidence about the free list rather than about \
             the mechanism"
        );
        assert!(
            !bucket_admissible(128, 32768, Some(255.0)),
            "and just below it declines, which brackets the measured 256x"
        );
    }

    /// **The mechanism §9.2w exists for**: under `Age`, the small bucket is
    /// destroyed and the large one survives to over-serve (`DESIGN.md` §9.2v).
    ///
    /// This is the pathology at its smallest reproducible size, with an
    /// **absolute reference** rather than a shape (§3.1b rule 1): the assertion
    /// names the exact bucket that must survive and the exact one that must
    /// not, so a failure says which way the order went rather than "they were
    /// not equal".
    #[test]
    fn age_eviction_destroys_the_small_bucket_and_keeps_the_kv_sized_one() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_eviction_order(EvictionOrder::Age);

        // §9.2u's two tenants at their measured sizes: a 76 KB activation
        // request and one K tensor at `kv_len` 234. Freed activation-first,
        // which is the order decode produces them in.
        const ACT: usize = 76 * 1024;
        const KV: usize = 3_833_856;

        // Held live together, then released together, so both are in the free
        // list at once: allocating one after freeing the other would simply
        // reuse it and there would be no pair to choose between. `free_order`
        // records activation-first, which is the order decode frees them in.
        let held = vec![alloc(&pool, &dev, ACT), alloc(&pool, &dev, KV)];
        drop(held);
        gpu_completes(&pool, &clock);

        // Room for the KV tensor alone, so exactly one of the two must go.
        pool.set_free_budget(KV);

        // The activation bucket was freed first, so Age destroyed it -- and the
        // request does NOT miss: it is answered by the KV discard that
        // survived. That is the pathology in one assertion, and it is why the
        // reference here is the served SIZE rather than hit-or-miss. A
        // hit/miss assertion would pass on a pool that reused correctly.
        let served = pool
            .acquire(ACT)
            .expect("the request is served -- from the wrong bucket");
        assert_eq!(
            served.size(),
            KV,
            "under Age the {ACT} B bucket is destroyed as the oldest and the \
             {KV} B KV discard survives, so an activation request is served at \
             {}x its size. This is §9.2u's 'one class served out of another \
             class's garbage', at its smallest reproducible size",
            KV / ACT
        );
    }

    /// The same fixture under `Size`, which is the falsifiable half (§9.2w).
    ///
    /// Largest-first destroys the KV discard and **leaves the activation bucket
    /// alive**, so the next activation request is served at its own size. The
    /// two tests share a fixture and differ only in the arm, which is what
    /// makes the arm the attributable cause.
    ///
    /// **`oversize_declined` must stay 0**: that is the discriminator against
    /// §9.2v's bound, which reaches the same ratio by refusing reuse. An arm
    /// that moves the ratio while declining nothing is retention, not refusal.
    #[test]
    fn size_eviction_keeps_the_small_bucket_and_destroys_the_kv_sized_one() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_eviction_order(EvictionOrder::Size);

        const ACT: usize = 76 * 1024;
        const KV: usize = 3_833_856;

        // Identical fixture to the `Age` test above; only the arm differs,
        // which is what makes the arm the attributable cause.
        let held = vec![alloc(&pool, &dev, ACT), alloc(&pool, &dev, KV)];
        drop(held);
        gpu_completes(&pool, &clock);

        pool.set_free_budget(KV);

        let served = pool
            .acquire(ACT)
            .expect("under Size the activation bucket survives");
        assert_eq!(
            served.size(),
            ACT,
            "and it is served at its own size -- ratio 1.00x, against Age's \
             {}x on the identical fixture",
            KV / ACT
        );
        assert_eq!(
            pool.counters().oversize_declined,
            0,
            "and nothing was declined. This is what separates this axis from \
             §9.2v's bound, which reaches the same ratio by refusing reuse and \
             pays 3.03x the allocations for it"
        );
    }

    /// Evicting the largest first reaches the budget in **fewer** evictions,
    /// which is the arithmetic the axis rests on (`DESIGN.md` §9.2w).
    ///
    /// An absolute reference on both arms: 52 small buckets against 1 large one
    /// for the same released bytes. A shape invariant ("Size evicts fewer")
    /// would pass on any pair of numbers with that ordering; these are the
    /// numbers the geometry predicts.
    #[test]
    fn size_eviction_reaches_the_budget_in_fewer_evictions_than_age() {
        let dev = device();
        const ACT: usize = 73_728;
        // Exactly 52 activation buckets, so the arithmetic is exact rather than
        // approximate and the assertion can name a count.
        const KV: usize = ACT * 52;

        for (order, expected) in [(EvictionOrder::Age, 52), (EvictionOrder::Size, 1)] {
            let (pool, clock) = pool_with_clock();
            pool.set_eviction_order(order);

            // All 53 held live at once, so every one reaches the free list --
            // allocating and freeing in turn would reuse the same buffer 52
            // times and there would be nothing to choose between. The 52 small
            // ones are allocated first, so `free_order` records them first.
            let mut held: Vec<_> = (0..52).map(|_| alloc(&pool, &dev, ACT)).collect();
            held.push(alloc(&pool, &dev, KV));
            drop(held);
            gpu_completes(&pool, &clock);

            // free_bytes is 2 x KV (52 x ACT of small, plus KV of large). A
            // budget of KV therefore requires exactly KV bytes released, which
            // is 52 small buckets or 1 large one.
            assert_eq!(
                pool.occupancy().free_bytes,
                2 * KV,
                "{order:?}: the fixture must hold both halves before the budget \
                 is applied, or the count below measures the fixture"
            );
            pool.set_free_budget(KV);

            assert_eq!(
                pool.counters().evicted,
                expected,
                "{order:?} must release {KV} bytes in {expected} eviction(s): \
                 Age takes them {ACT} B at a time because that is the order \
                 they were freed in, Size takes the one bucket that is the \
                 whole amount"
            );
        }
    }

    /// The arm renders as it is declared, read from the pool rather than the
    /// flag (§7.1a-i, and §9.2v's `Display`-drops-a-trailing-zero defect).
    ///
    /// §9.2v's float arm rendered `4` for a `.bench/configurations.md` arm
    /// declared `4.0`, so two runs of one arm would not have pooled. These arms
    /// are strings, which removes that failure by construction -- and this
    /// pins the spelling so a type rename cannot silently move it.
    #[test]
    fn the_eviction_arm_renders_as_the_registry_declares_it() {
        assert_eq!(EvictionOrder::Age.as_str(), "age");
        assert_eq!(EvictionOrder::Size.as_str(), "size");

        let (pool, _clock) = pool_with_clock();
        pool.set_eviction_order(EvictionOrder::Size);
        assert_eq!(
            pool.eviction_order().as_str(),
            "size",
            "the harness must read the arm back from the state the eviction \
             path consults, never from the flag it was set with"
        );
    }

    /// Emptying a bucket must remove its key, or eviction stops making
    /// progress while the budget is still exceeded (§9.2w, and issue #21's
    /// graveyard defect).
    ///
    /// **Written because a mutation survived without it.** The `Size` arm
    /// reads `free`'s last key and pops from that bucket; if the emptied key
    /// is left behind, the next call finds the same key, pops an **empty**
    /// `Vec`, and `take_eviction_victim` returns `None` -- so
    /// `evict_over_budget` breaks out with `free_bytes` still over budget and
    /// **retains the memory silently**. Every other test here evicts at most
    /// one buffer per size, so none of them reaches the second pop.
    ///
    /// The reference is absolute (§3.1b rule 1): two buffers of ONE size, a
    /// budget requiring BOTH to go, and an assertion on the resulting
    /// `free_bytes` -- which a "some eviction happened" shape would pass.
    #[test]
    fn emptying_a_bucket_removes_its_key_so_eviction_can_continue() {
        let dev = device();
        const SIZE: usize = 1024 * 1024;

        for order in [EvictionOrder::Age, EvictionOrder::Size] {
            let (pool, clock) = pool_with_clock();
            pool.set_eviction_order(order);

            // Two buffers of the SAME size, so one bucket holds both and
            // draining it needs two pops from one key.
            let held = vec![alloc(&pool, &dev, SIZE), alloc(&pool, &dev, SIZE)];
            drop(held);
            gpu_completes(&pool, &clock);
            assert_eq!(
                pool.occupancy().free_bytes,
                2 * SIZE,
                "{order:?}: both must be free before the budget is applied"
            );

            // A budget of zero requires the whole bucket to go: two pops, one
            // key.
            pool.set_free_budget(0);

            assert_eq!(
                pool.occupancy().free_bytes,
                0,
                "{order:?}: eviction must reach the budget. A non-zero residue \
                 means the loop stopped early -- which is what happens when an \
                 emptied bucket's key is left in `free` and the next pop finds \
                 an empty Vec"
            );
            assert_eq!(
                pool.counters().evicted,
                2,
                "{order:?}: and it took exactly two evictions to get there"
            );
            // **The assertion the byte count cannot make.** Draining the last
            // buffer must take the KEY with it. `free_bytes` reaches 0 either
            // way -- the budget is met by the second pop, so a left-behind key
            // is never popped a third time and never stops progress *here* --
            // so the residue is visible only as a bucket that still exists
            // holding an empty `Vec`. That is issue #21's graveyard: keys
            // accumulating at ~3.2 per token, 412 291 walked per token to
            // manage 329 live buffers, an O(N) whose N is work-done-so-far.
            assert_eq!(
                pool.occupancy().free_buckets,
                0,
                "{order:?}: the emptied bucket's KEY must go too. `free_bytes` \
                 is 0 either way, so a key left behind is invisible to every \
                 byte-valued assertion and shows up only here -- and it is \
                 what `acquire`'s range query then walks forever (#21)"
            );
        }
    }

    /// An empty free list stops the loop rather than spinning (§9.2w).
    ///
    /// Both arms return `None` when there is nothing evictable, which is what
    /// terminates `evict_over_budget`. Worth pinning because the `Size` arm
    /// reads the map rather than the queue, so it has a different exhaustion
    /// condition from the one `Age` has always had.
    #[test]
    fn eviction_terminates_on_an_empty_free_list_under_both_arms() {
        for order in [EvictionOrder::Age, EvictionOrder::Size] {
            let (pool, _clock) = pool_with_clock();
            pool.set_eviction_order(order);
            // A budget of 0 with nothing free: the loop must not spin.
            pool.set_free_budget(0);
            assert_eq!(
                pool.counters().evicted,
                0,
                "{order:?}: nothing was free, so nothing can be evicted -- and \
                 the call returned, which is the half a hang would fail"
            );
        }
    }

    /// **The invariant `acquire` rests on**: a view must never enter the pool
    /// (#507).
    ///
    /// `acquire` keys on size alone, which is correct exactly while two buffers
    /// of one size are interchangeable -- true for allocations, false for views,
    /// which are N handles onto one allocation. This drives the guard from the
    /// aliasing side: a `Buffer::view` is exactly what `Arena::new` used to hand
    /// to `adopt` before #505, and it is what produced five "separate"
    /// activation buffers at one base address (§9.2m).
    ///
    /// **The check is at `adopt` rather than `acquire` because `acquire` is
    /// hot** (§6.3b: ~11.6 destroys/token) and the property is decided once, on
    /// entry. This test is `debug_assertions`-only for the same reason the
    /// assertion is.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "was given a VIEW")]
    fn adopting_a_view_is_refused() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();

        // An arena-shaped base, and a slot carved out of it. §9.2c's shipping
        // arena is [21504, 21504, 21504, 4096, 1024]; the first slot is the one
        // an offset-based check would MISS, so it is the one used here.
        let base = dev
            .new_buffer(69632, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let slot = base.view(0, 21504);
        assert_eq!(slot.base_offset(), 0, "the offset-0 slot is the hard case");
        assert!(
            slot.is_view(),
            "a carved slot is a view whatever its offset"
        );

        let _ = pool.adopt(slot, 21504);
    }

    /// The guard admits what it must: an ordinary allocation.
    ///
    /// **Both bounds** (§1.1, #184's precedent). A refusal that refuses
    /// everything is not a check, and this is the arm that would catch an
    /// `is_view` inverted or an assertion written `!` the wrong way -- which
    /// would otherwise refuse every allocation the engine makes and be caught
    /// only by the whole suite going red for an unrelated-looking reason.
    #[test]
    fn adopting_a_whole_allocation_is_admitted() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();

        let raw = dev
            .new_buffer(4096, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        assert!(!raw.is_view(), "a fresh allocation is not a view");

        let b = pool.adopt(raw, 4096);
        assert_eq!(pool.counters().allocations, 1);
        assert_eq!(pool.occupancy().live_buffers, 1);
        drop(b);
    }

    /// **#494's finding, repaired by #507**: `reset_counters` must not disturb
    /// how many buffers are awaiting the GPU.
    ///
    /// The defect was that `PoolCounters::pending` was a **gauge** among eleven
    /// accumulators -- its only two write sites assigned `state.pending.len()`,
    /// where every other field is `+=` -- so `reset_counters`, which zeroes the
    /// whole struct, zeroed a figure describing **live occupancy** rather than
    /// history. The pool then contradicted itself: `counters().pending` read 0
    /// while `occupancy().pending_buffers` read 1, for the same pool at the same
    /// instant, with nothing having changed about the GPU.
    ///
    /// Nothing in the 30 hand-written tests called `reset_counters` while a
    /// buffer was parked, which is why it survived: the two that call it both
    /// have `gpu_completes` on the line above, so each resets a *settled* pool.
    /// **The defect needs a release without a completing drain, then a reset --
    /// a sequence, not a state**, which is why a stateful fuzzer found it in
    /// three operations and thirty hand-written tests did not.
    ///
    /// **The repair removed the field rather than re-deriving it** (#507), so
    /// this test pins the *absence of the mirror*: it asserts the live figure
    /// survives a reset, which is the property that was violated, entirely
    /// through the public API. The companion guard against a *future* gauge
    /// being added to `PoolCounters` is
    /// `buffer_pool_fuzz::pool_counters_holds_accumulators_only`, which has to
    /// read the source because that defect is a write shape rather than a state.
    #[test]
    fn reset_counters_does_not_disturb_the_pending_occupancy() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let b = alloc(&pool, &dev, 512);
        drop(b);

        // One buffer is genuinely parked awaiting the GPU. This is the state the
        // defect needed and that no pre-#494 test constructed: released, and
        // *not* drained by a completion.
        assert_eq!(pool.occupancy().pending_buffers, 1);
        assert_eq!(
            pool.counters().releases,
            1,
            "the release must have happened, or this fixture proves nothing"
        );

        pool.reset_counters();

        // The accumulators ARE zeroed -- that is what the call is for, and
        // asserting it stops a `reset_counters` that does nothing at all from
        // passing this test (§15.1a's vacuous class: both bounds, §1.1).
        assert_eq!(pool.counters().releases, 0, "the accumulator did not reset");
        assert_eq!(pool.counters().deferred, 0, "the accumulator did not reset");

        // And the live figure is untouched: nothing about the GPU changed, so
        // nothing about how many buffers await it may change either.
        assert_eq!(
            pool.occupancy().pending_buffers,
            1,
            "reset_counters disturbed a live occupancy figure -- #494's defect, \
             repaired in #507 by removing the mirrored gauge from PoolCounters"
        );

        // The parked buffer is still a real buffer: completing the GPU drains it
        // to the free list exactly as it would have without the reset. Without
        // this the test would also pass on a pool that had lost the buffer.
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().pending_buffers, 0);
        assert_eq!(
            pool.occupancy().free_buffers,
            1,
            "the parked buffer did not survive the reset as a real buffer"
        );
    }

    #[test]
    fn empty_pool_misses() {
        let pool = BufferPool::new();
        assert!(pool.acquire(128).is_none());
        let c = pool.counters();
        assert_eq!(c.lookups, 1);
        assert_eq!(c.hits, 0);
        // Two probes: exact bucket, then the range query. Both find nothing.
        assert_eq!(c.buckets_probed, 2);
    }

    #[test]
    fn occupancy_of_empty_pool_is_zero() {
        let pool = BufferPool::new();
        let occ = pool.occupancy();
        assert_eq!(occ.free_buckets, 0);
        assert_eq!(occ.total_buffers(), 0);
    }

    /// Releasing is still an *event* -- nothing is swept, polled or asked -- but
    /// the event now says "the CPU is done", which is not the same as "reusable".
    /// The buffer is parked, and it is the GPU finishing that frees it.
    #[test]
    fn drop_parks_the_buffer_until_the_gpu_is_done() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let b = alloc(&pool, &dev, 1024);
        assert_eq!(pool.occupancy().free_buffers, 0, "live buffer is not free");
        assert_eq!(pool.counters().releases, 0);

        drop(b);

        assert_eq!(pool.counters().releases, 1, "Drop did not push");
        assert_eq!(pool.counters().deferred, 1, "release was not deferred");
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "buffer offered for reuse while the GPU may still be using it"
        );
        assert_eq!(pool.occupancy().pending_buffers, 1);
        assert_eq!(pool.occupancy().live_buffers, 0);

        gpu_completes(&pool, &clock);

        assert_eq!(
            pool.occupancy().free_buffers,
            1,
            "GPU completion did not free it"
        );
        assert_eq!(pool.occupancy().pending_buffers, 0);
        assert_eq!(pool.counters().drained, 1);
    }

    /// The bug, stated as a test. A dropped buffer whose GPU work is still
    /// outstanding must not be handed to a second caller -- that is the aliasing
    /// that corrupted grouped convolutions (issue #19).
    #[test]
    fn in_flight_buffer_is_not_handed_out_again() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let first = alloc(&pool, &dev, 4096);
        // The CPU is done with it; the GPU, which has not advanced, is not.
        drop(first);

        assert!(
            pool.acquire(4096).is_none(),
            "handed back a buffer with GPU work outstanding -- this is issue #19"
        );

        gpu_completes(&pool, &clock);
        assert!(
            pool.acquire(4096).is_some(),
            "buffer stayed unavailable after its work completed"
        );
    }

    /// Every release is deferred, including one made while the GPU is idle.
    ///
    /// This looks like a missed optimization and is not one. A release is
    /// stamped with the epoch that is still *open*, and an open epoch cannot
    /// have completed, so a "hand it straight back" branch would be
    /// unreachable. Pinned as a test because the branch is the obvious thing to
    /// add back, and adding it would either do nothing or -- if the condition
    /// were loosened until it did something -- reintroduce the bug.
    #[test]
    fn release_is_deferred_even_when_the_gpu_is_idle() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Everything submitted so far is finished.
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        assert!(clock.is_complete(epoch));

        let b = alloc(&pool, &dev, 1024);
        drop(b);

        assert_eq!(pool.counters().deferred, 1);
        assert_eq!(
            pool.occupancy().free_buffers,
            0,
            "released against an epoch that is still open"
        );

        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// A released buffer is handed straight back rather than reallocated, and
    /// it is the same Metal object.
    #[test]
    fn released_buffer_is_reused() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Identity of the underlying Metal object, not of the wrapper: the
        // handle is necessarily a different allocation each time, and what
        // matters is that the same MTLBuffer came back.
        fn metal_id(b: &PooledBuffer) -> usize {
            let raw: &Buffer = b;
            raw.as_ref() as *const _ as *const () as usize
        }

        let first = alloc(&pool, &dev, 2048);
        let addr = metal_id(&first);
        release_and_complete(&pool, &clock, first);

        let second = alloc(&pool, &dev, 2048);
        assert_eq!(metal_id(&second), addr, "did not reuse the released buffer");
        assert_eq!(pool.counters().allocations, 1, "allocated a second time");
        assert_eq!(pool.counters().hits, 1);
    }

    /// Epochs retire in order, so a buffer parked in an earlier epoch is freed
    /// by a later completion. The drain walks a prefix and stops; it must not
    /// leave an early buffer behind because a later one is still outstanding.
    #[test]
    fn earlier_epochs_are_freed_by_later_completions() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Parked in epoch 0.
        drop(alloc(&pool, &dev, 1024));
        let e0 = clock.commit_epoch();

        // Parked in epoch 1.
        drop(alloc(&pool, &dev, 2048));
        let e1 = clock.commit_epoch();

        assert_eq!(pool.occupancy().pending_buffers, 2);

        // Epoch 0 finishes: only the first buffer is safe.
        clock.mark_completed(e0);
        pool.drain_completed();
        assert_eq!(pool.occupancy().free_buffers, 1);
        assert_eq!(pool.occupancy().pending_buffers, 1);

        clock.mark_completed(e1);
        pool.drain_completed();
        assert_eq!(pool.occupancy().free_buffers, 2);
        assert_eq!(pool.occupancy().pending_buffers, 0);
    }

    /// Reuse must not require the caller to say anything. A buffer moved into
    /// a struct, cloned, and passed around still returns itself when the last
    /// reference dies -- which is what keeps candle's allocate-and-forget
    /// ergonomics intact.
    #[test]
    fn reclamation_needs_no_caller_cooperation() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        struct Holder {
            _buffer: Arc<PooledBuffer>,
        }

        let b = alloc(&pool, &dev, 512);
        let clone = b.clone();
        let holder = Holder { _buffer: b };

        drop(clone);
        assert_eq!(
            pool.counters().releases,
            0,
            "returned while a reference was still held"
        );

        drop(holder);
        assert_eq!(
            pool.counters().releases,
            1,
            "last reference did not return it"
        );

        // Still nothing the caller has to do: the GPU finishing is what makes
        // it reusable, and that is the device's business, not the caller's.
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// A request is served by a larger free buffer when no exact match exists,
    /// which is what the old scan's "best fit" achieved -- but by one ordered
    /// probe rather than a walk over every bucket.
    #[test]
    fn larger_buffer_satisfies_smaller_request() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        release_and_complete(&pool, &clock, alloc(&pool, &dev, 4096));
        let got = pool.acquire(1024).expect("larger buffer should satisfy");
        assert_eq!(got.size(), 4096);
        assert_eq!(pool.counters().allocations, 1);
    }

    /// The smallest sufficient buffer wins, so a large one is not consumed to
    /// serve a small request while a closer fit sits free.
    ///
    /// Built by allocating all three concurrently and releasing them together:
    /// going through `alloc` one at a time would hand the 8192 back to satisfy
    /// the 2048 request, which is correct reuse but not the state under test.
    #[test]
    fn smallest_sufficient_buffer_wins() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let held: Vec<_> = [8192, 2048, 4096]
            .into_iter()
            .map(|s| alloc(&pool, &dev, s))
            .collect();
        drop(held);
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buckets, 3);

        let got = pool.acquire(1024).expect("something should satisfy");
        assert_eq!(got.size(), 2048, "did not pick the closest fit");
    }

    /// An exact match is taken without consulting any larger bucket. This is
    /// the decode-steady-state case, and it is why the scan is bounded: one
    /// probe, regardless of how many other sizes the pool holds.
    #[test]
    fn exact_match_costs_one_probe() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        for size in [512, 1024, 2048, 4096, 8192] {
            drop(alloc(&pool, &dev, size));
        }
        gpu_completes(&pool, &clock);
        pool.reset_counters();

        let got = pool.acquire(2048).expect("exact match exists");
        assert_eq!(got.size(), 2048);
        assert_eq!(
            pool.counters().buckets_probed,
            1,
            "an exact hit must not consult a second bucket"
        );
    }

    /// The pathology this change removes. The old pool's `retain` emptied a
    /// bucket's `Vec` but left the key in the map, so emptied keys accumulated
    /// -- 1324 of 1345 buckets were empty after a 400-token generation, and
    /// every allocation walked all of them. Here a bucket that empties is
    /// removed, so the map only ever holds sizes that have something free.
    #[test]
    fn emptied_buckets_do_not_accumulate() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Distinct sizes, as a growing KV cache produces.
        for i in 0..64 {
            let size = 1024 + i * 128;
            let b = alloc(&pool, &dev, size);
            release_and_complete(&pool, &clock, b);
            // Take it straight back out, emptying the bucket again.
            let b = pool.acquire(size).expect("just released");
            std::mem::forget(b);
        }

        assert_eq!(
            pool.occupancy().free_buckets,
            0,
            "emptied buckets were left behind"
        );
    }

    /// Scan length is bounded by a constant, not by pool size. This is the
    /// invariant `CONTRIBUTING.md` §3.3 and `DESIGN.md` §15.2 #10 demand of
    /// anything on the per-token path.
    #[test]
    fn lookup_cost_is_bounded_regardless_of_pool_size() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let mut held = Vec::new();
        for i in 0..256 {
            held.push(alloc(&pool, &dev, 1024 + i * 128));
        }
        drop(held);
        gpu_completes(&pool, &clock);
        assert_eq!(pool.occupancy().free_buckets, 256);

        pool.reset_counters();
        for i in 0..256 {
            let b = pool.acquire(1024 + i * 128).expect("exact match exists");
            std::mem::forget(b);
        }
        let c = pool.counters();
        assert_eq!(c.lookups, 256);
        assert_eq!(
            c.buckets_probed, 256,
            "expected exactly one probe per exact-match lookup, got {}",
            c.buckets_probed
        );
    }

    /// Trim destroys what is free and leaves what is live, so it can run under
    /// memory pressure without disturbing work in progress.
    #[test]
    fn trim_frees_only_what_is_free() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let live = alloc(&pool, &dev, 1024);
        release_and_complete(&pool, &clock, alloc(&pool, &dev, 2048));
        assert_eq!(pool.occupancy().free_buffers, 1);

        let freed = pool.trim();
        assert_eq!(freed.len(), 1, "trim should return the one free buffer");
        assert_eq!(pool.occupancy().free_buffers, 0);
        assert_eq!(pool.occupancy().live_buffers, 1);

        // The live one still returns itself afterwards.
        release_and_complete(&pool, &clock, live);
        assert_eq!(pool.occupancy().free_buffers, 1);
    }

    /// Trim must not destroy a buffer the GPU may still be reading. Freeing one
    /// here would be the same defect the deferral exists to prevent, with
    /// use-after-free in place of aliasing.
    #[test]
    fn trim_leaves_buffers_the_gpu_may_still_be_using() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        drop(alloc(&pool, &dev, 2048));
        assert_eq!(pool.occupancy().pending_buffers, 1);

        let freed = pool.trim();
        assert!(
            freed.is_empty(),
            "trim destroyed a buffer with GPU work outstanding"
        );
        assert_eq!(pool.occupancy().pending_buffers, 1);

        // Once the work completes, a trim reclaims it as normal.
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        assert_eq!(pool.trim().len(), 1);
    }

    /// An unpooled handle frees its buffer outright. `new_private_buffer`
    /// depends on this: it exists for persistent allocations that must not
    /// re-enter the reuse path.
    #[test]
    fn unpooled_handle_does_not_return_to_any_pool() {
        let dev = device();
        let pool = BufferPool::new();

        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = Arc::new(PooledBuffer::unpooled(raw, 1024));
        drop(b);

        assert_eq!(pool.occupancy().free_buffers, 0);
        assert_eq!(pool.counters().releases, 0);
    }

    /// The regression that deferring release introduced, and the bound that
    /// stops it.
    ///
    /// A workload asking for a slightly larger buffer each step never asks
    /// again for the size it just freed. Under CPU-drop release that did not
    /// matter, because the buffer was reused within the same operation that
    /// freed it. Deferring to GPU completion moves the return past that point,
    /// so every one of those buffers is stranded -- 11.6 per token on LFM2,
    /// taking the pool to 13.6 GB at 400 tokens before this bound existed.
    #[test]
    fn a_workload_that_never_reuses_a_size_does_not_grow_the_pool() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_free_budget(4 * 1024 * 1024);

        // Ever-growing sizes, as a KV cache produces.
        for i in 0..512usize {
            let size = 64 * 1024 + i * 1024;
            let b = alloc(&pool, &dev, size);
            release_and_complete(&pool, &clock, b);
        }

        let occ = pool.occupancy();
        assert!(
            occ.free_bytes <= 4 * 1024 * 1024,
            "free list grew to {} bytes, past its {} byte budget",
            occ.free_bytes,
            4 * 1024 * 1024
        );
        assert!(
            pool.counters().evicted > 0,
            "nothing was evicted, so the bound never engaged"
        );
    }

    /// The bound must not cost anything when the workload does reuse what it
    /// frees, which is the fixed-shape steady state. Nothing should be evicted,
    /// and every allocation after the first should be a hit.
    #[test]
    fn a_reusing_workload_never_evicts() {
        let dev = device();
        let (pool, clock) = pool_with_clock();
        pool.set_free_budget(1024 * 1024);

        for _ in 0..256 {
            let b = alloc(&pool, &dev, 4096);
            release_and_complete(&pool, &clock, b);
        }

        let c = pool.counters();
        assert_eq!(
            c.evicted, 0,
            "evicted from a workload that reuses its sizes"
        );
        assert_eq!(c.allocations, 1, "allocated more than once for one size");
        assert_eq!(c.hits, 255);
    }

    /// Fixed-shape work converges: the working set stops growing.
    ///
    /// This is the property the device-level
    /// `repeated_identical_work_converges_to_bounded_reuse` is after, pinned
    /// here where the pool is owned exclusively and the numbers are exact. At
    /// device level several tests share one process-wide pool and trim each
    /// other's free lists, so the same assertion can only be made qualitatively
    /// there.
    ///
    /// The in-flight window is modelled directly: `depth` iterations are in
    /// flight before any of them completes, which is what a command buffer
    /// batching many dispatches does. So the pool must allocate `depth` buffers
    /// and then **never allocate again**, however long the workload runs -- the
    /// working set is a function of the window, not of the iteration count.
    #[test]
    fn fixed_shape_work_allocates_the_window_then_stops() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        const DEPTH: usize = 8;

        // Steady state: `DEPTH` iterations in flight at any moment.
        let mut in_flight: std::collections::VecDeque<Arc<PooledBuffer>> =
            std::collections::VecDeque::new();
        for _ in 0..DEPTH {
            in_flight.push_back(alloc(&pool, &dev, 4096));
        }

        for _ in 0..1000 {
            // Oldest completes and is released.
            drop(in_flight.pop_front().expect("window is full"));
            gpu_completes(&pool, &clock);
            // A new iteration starts, which must reuse what just came back.
            in_flight.push_back(alloc(&pool, &dev, 4096));
        }

        let c = pool.counters();
        assert_eq!(
            c.allocations, DEPTH as u64,
            "allocated {} buffers for a window {DEPTH} deep; the working set is \
             growing with the iteration count rather than with the window",
            c.allocations
        );
        assert_eq!(c.hits, 1000, "every steady-state iteration should be a hit");
    }

    /// Eviction must take the oldest free buffer, not whichever is convenient.
    /// The oldest is the one whose size has gone longest without being asked
    /// for, which in the growing case is the one that will never be asked for
    /// again.
    #[test]
    fn eviction_takes_the_oldest_free_buffer_first() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // All three allocated and released before anything drains, so they
        // enter the free list together and the budget applies to the set rather
        // than to each in turn.
        let held = vec![
            alloc(&pool, &dev, 4096),
            alloc(&pool, &dev, 8192),
            alloc(&pool, &dev, 12288),
        ];
        drop(held);

        // Room for the newest only: 12288 fits in 16384, and adding either
        // older one would not.
        pool.set_free_budget(16384);
        gpu_completes(&pool, &clock);

        assert!(pool.occupancy().free_bytes <= 16384);
        assert!(
            pool.acquire(12288).is_some(),
            "evicted the newest buffer instead of the oldest"
        );
        assert!(
            pool.counters().evicted > 0,
            "nothing was evicted, so the ordering was never exercised"
        );
    }

    /// A poison mark must not outlive the mapping it was applied to (#475).
    ///
    /// **This pins a FALSE POSITIVE, which is the failure direction that
    /// discredits an instrument rather than merely blunting it**, and it is
    /// not hypothetical: the first version of the poisoning marked buffers on
    /// entry to the free list and cleared them on eviction, trim and acquire —
    /// and the suite tripped ASan anyway, inside `__bzero` under
    /// `-[IOGPUMetalDevice allocBufferSubDataWithLength:...]`.
    ///
    /// The mechanism is that **Metal sub-allocates**: a later
    /// `newBufferWithLength:` can be served from an address range a destroyed
    /// buffer occupied, and Metal zeroes it on the way out. A mark left behind
    /// therefore reports Metal's own zeroing of an unrelated *fresh*
    /// allocation.
    ///
    /// So the discipline is over the **mapping**, not the buffer, and every
    /// route out of the pool must clear: acquire, evict, trim, a handle
    /// outliving its pool, and the pool itself dropping.
    ///
    /// Under `asan-poison` this test aborts the process if a route is missed,
    /// which is a stronger signal than an assertion; without the feature it is
    /// an ordinary allocate-free-allocate exercise of the same routes.
    #[test]
    fn a_poison_mark_does_not_outlive_its_mapping() {
        let dev = device();

        // Route 1: the pool drops with buffers still in its free list.
        {
            let (pool, clock) = pool_with_clock();
            let b = alloc(&pool, &dev, 1024);
            release_and_complete(&pool, &clock, b);
            assert!(
                pool.counters().drained > 0,
                "nothing reached the free list, so no mark was applied and \
                 this test could not have caught a stale one"
            );
        }

        // Route 2: a handle outliving its pool, where `Weak::upgrade` fails
        // and the buffer is destroyed rather than returned.
        {
            let (pool, _clock) = pool_with_clock();
            let b = alloc(&pool, &dev, 1024);
            drop(pool);
            drop(b);
        }

        // Route 3: trim, which destroys the whole free list at once.
        {
            let (pool, clock) = pool_with_clock();
            let b = alloc(&pool, &dev, 1024);
            release_and_complete(&pool, &clock, b);
            let freed = pool.trim();
            assert!(!freed.is_empty(), "trim freed nothing to clear a mark on");
            drop(freed);
        }

        // If any route above left a mark behind, Metal's zeroing of these
        // fresh allocations is what reports it -- the exact shape that was
        // observed. Several, because sub-allocation reuse is not guaranteed to
        // land on the first try.
        for _ in 0..16 {
            let _ = dev.new_buffer(1024, crate::kernels::RESOURCE_OPTIONS);
        }
    }

    /// A buffer outliving its pool must drop cleanly rather than resurrect it.
    /// This is why the back-reference is `Weak`.
    #[test]
    fn buffer_outliving_its_pool_drops_cleanly() {
        let dev = device();
        let (pool, _clock) = pool_with_clock();
        let b = alloc(&pool, &dev, 1024);

        drop(pool);
        // Must not panic, and must not try to push into a dead pool.
        drop(b);
    }

    /// Zero must mean "nothing has completed", not "epoch 0 has". They are
    /// different states, and conflating them would free every buffer parked in
    /// the first epoch before the GPU had run anything at all -- the exact bug
    /// this change exists to fix, reintroduced by an off-by-one.
    #[test]
    fn a_fresh_clock_has_completed_nothing() {
        let clock = GpuClock::new();
        assert_eq!(clock.current_epoch(), 0);
        assert_eq!(clock.completed_count(), 0);
        assert!(
            !clock.is_complete(0),
            "epoch 0 reported finished before anything ran"
        );

        clock.mark_completed(0);
        assert!(clock.is_complete(0));
    }

    /// The two counters are written by **different threads concurrently** --
    /// `submitted` by the host at commit, `completed` by Metal's completion
    /// handler -- so they must not share a cache line (issue #442).
    ///
    /// Asserted on the **production struct** rather than on `CachePadded`
    /// alone: `super::cache_pad`'s own tests prove the wrapper aligns, and this
    /// proves `GpuClock` *uses* it. Reverting both fields to bare `AtomicU64`
    /// passes every one of those tests and fails this one at **8 bytes apart**,
    /// which is the whole difference between a mechanism existing and a
    /// mechanism being applied.
    ///
    /// **One mutation SURVIVES and it is recorded rather than engineered
    /// away**: unpadding `submitted` *alone* leaves this green, because
    /// `completed`'s own 128 B alignment pushes it onto the next line anyway.
    /// That is not a hole in the test -- the struct really is correct in that
    /// arm, since what matters is that the pair does not share a line, and it
    /// does not. **Padding one field of an adjacent pair is sufficient**; both
    /// carry it here because which field needs padding is a property of who
    /// writes it, and a third field inserted between them would re-share a line
    /// with only one padded. **The test asserts the property, not the
    /// spelling**, which is why it admits that arm.
    #[test]
    fn the_clocks_two_counters_do_not_share_a_cache_line() {
        use crate::kernels::metal::cache_pad::CACHE_LINE;

        let clock = GpuClock::new();
        let submitted = &clock.submitted as *const _ as usize;
        let completed = &clock.completed as *const _ as usize;
        assert!(
            completed.abs_diff(submitted) >= CACHE_LINE,
            "GpuClock::submitted ({submitted:#x}) and ::completed ({completed:#x}) are within \
             one {CACHE_LINE} B line -- the host thread and the completion handler would \
             invalidate each other ~11-14 times per token"
        );
    }

    /// Completion notifications for different command buffers may arrive on
    /// different threads, so the clock must not go backwards when an older one
    /// is delivered late.
    #[test]
    fn completion_never_goes_backwards() {
        let clock = GpuClock::new();
        clock.mark_completed(5);
        clock.mark_completed(2);
        assert_eq!(
            clock.completed_count(),
            6,
            "a late report moved the clock back"
        );
        assert!(clock.is_complete(5));
    }

    /// The overhang counts BOTH sources, with an absolute reference for each.
    ///
    /// `CONTRIBUTING.md` §3.1b rule 1: an absolute figure over a shape. *"The
    /// overhang is nonzero"* would pass on a counter that double-counts, on one
    /// that counts only misses, and on one that adds a constant -- so every
    /// number here is derived by hand from the two inputs and asserted exactly.
    ///
    /// The two sources are deliberately in one test: they are summed into one
    /// figure, so a test covering only one leaves the other free to drift while
    /// the suite stays green (§3.1b rule 4 -- incidental coverage is a distinct
    /// risk from absence).
    #[test]
    fn the_overhang_counts_the_miss_rounding_and_the_hit_oversize() {
        let dev = device();
        // A pool with a clock, because `release` always parks into `pending`
        // (§6.3b's deliberate absence of an idle fast path) -- a drop alone
        // leaves the free list empty and every reuse assertion below vacuous.
        // That is §9.2m's measured VACUOUS case, in the test written to catch
        // this defect.
        let (pool, clock) = pool_with_clock();

        // --- Source 1: the MISS. A caller wants 1000 B; `buf_size`-style
        // rounding to 1024 hands it 24 B it did not ask for.
        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = pool.adopt_requesting(raw, 1000, 1024);
        let c = pool.counters();
        assert_eq!(c.requested_bytes, 1000);
        assert_eq!(c.served_bytes, 1024);
        assert_eq!(c.overhang_bytes(), 24, "the miss rounding is 1024 - 1000");

        // The handle must report the BUCKET, not the request: reuse keys on it,
        // and a handle claiming 1000 would return to the wrong free list.
        assert_eq!(b.size(), 1024);

        // --- Source 2: the HIT. Return the 1024 B buffer, then ask for 900.
        // The exact probe misses and the range query serves the 1024 bucket,
        // so this caller is over-served by 124 B without any allocation.
        release_and_complete(&pool, &clock, b);
        assert_eq!(
            pool.occupancy().free_buffers,
            1,
            "the buffer must actually be free, or the acquire below is vacuous"
        );
        let d = pool.acquire(900).expect("the 1024 bucket must satisfy 900");
        assert_eq!(d.size(), 1024, "served from the larger bucket");

        let c = pool.counters();
        assert_eq!(c.requested_bytes, 1000 + 900);
        assert_eq!(c.served_bytes, 1024 + 1024);
        assert_eq!(
            c.overhang_bytes(),
            24 + 124,
            "both sources must be summed: 24 B of miss rounding + 124 B of \
             hit oversize"
        );

        // The fraction is against what was SERVED, not what was requested.
        let f = c.overhang_fraction().expect("something was served");
        assert!(
            (f - (148.0 / 2048.0)).abs() < 1e-12,
            "overhang_fraction must be overhang/served, got {f}"
        );
    }

    /// An exact hit contributes nothing, which is what makes a nonzero
    /// overhang attributable.
    ///
    /// The non-vacuity arm for the test above: if every acquisition inflated
    /// the counter, a nonzero total would say nothing about rounding. Decode
    /// asks for the same shapes every token (`acquire`'s exact-probe comment),
    /// so this is the *common* case and it must read zero.
    #[test]
    fn an_exact_hit_adds_no_overhang_and_the_counter_still_moves() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        let raw = dev
            .new_buffer(2048, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = pool.adopt_requesting(raw, 2048, 2048);
        release_and_complete(&pool, &clock, b);
        assert_eq!(
            pool.occupancy().free_buffers,
            1,
            "the buffer must actually be free, or the acquire below is vacuous"
        );

        let before = pool.counters();
        assert_eq!(
            before.overhang_bytes(),
            0,
            "an exact request wastes nothing"
        );

        let b = pool.acquire(2048).expect("exact bucket");
        assert_eq!(b.size(), 2048);

        let after = pool.counters();
        // The zero above is evidence only because these moved (§3.1c: a zero
        // is not evidence unless something nonzero came from the same command).
        assert_eq!(
            after.served_bytes,
            before.served_bytes + 2048,
            "the counter must have moved, or the zero overhang is vacuous"
        );
        assert_eq!(after.requested_bytes, before.requested_bytes + 2048);
        assert_eq!(after.overhang_bytes(), 0, "an exact hit is not overhang");
    }

    /// Nothing served means no ratio, rather than a ratio of zero.
    ///
    /// `0.0` would be indistinguishable from a pool that served petabytes with
    /// perfect fit, and the two want opposite readings.
    #[test]
    fn overhang_fraction_is_none_on_an_untouched_pool() {
        let pool = BufferPool::new();
        assert_eq!(pool.counters().served_bytes, 0);
        assert!(pool.counters().overhang_fraction().is_none());
    }

    /// The invariant the subtraction rests on, over every path that records.
    ///
    /// `overhang_bytes` saturates, so a served-below-requested bug would read
    /// as a comfortable zero rather than as a failure. This asserts the
    /// property directly instead of trusting the subtraction to reveal it.
    #[test]
    fn served_is_never_below_requested() {
        let dev = device();
        let pool = BufferPool::new();

        for (requested, allocated) in [(1usize, 128usize), (128, 128), (129, 256), (5000, 5120)] {
            let raw = dev
                .new_buffer(allocated, crate::kernels::RESOURCE_OPTIONS)
                .expect("buffer allocation");
            let _ = pool.adopt_requesting(raw, requested, allocated);
        }

        let c = pool.counters();
        assert!(
            c.served_bytes >= c.requested_bytes,
            "served {} < requested {}, so the overhang would underflow",
            c.served_bytes,
            c.requested_bytes
        );
        // 127 + 0 + 127 + 120, by hand.
        assert_eq!(c.overhang_bytes(), 374);
    }

    /// The three band constants describe one partition, and cannot drift apart.
    ///
    /// `SIZE_BAND_COUNT` is spelled a second time in `lfm2-decode-profile`'s
    /// `MemProbe`, which is compiled without the `metal` feature and so cannot
    /// import this one. That copy is why this exists: a band added here without
    /// widening the probe's arrays would silently drop the new row from every
    /// committed series, and a series missing a row still parses.
    #[test]
    fn the_band_constants_describe_one_partition() {
        assert_eq!(
            SIZE_BAND_COUNT, 4,
            "lfm2-decode-profile's MemProbe spells this 4"
        );
        assert_eq!(
            SIZE_BAND_LIMITS.len(),
            SIZE_BAND_COUNT - 1,
            "N bands need N-1 interior boundaries"
        );
        assert_eq!(SIZE_BAND_NAMES.len(), SIZE_BAND_COUNT);
        assert!(
            SIZE_BAND_LIMITS.windows(2).all(|w| w[0] < w[1]),
            "the limits must be strictly increasing or `size_band`'s walk is \
             not a partition: {SIZE_BAND_LIMITS:?}"
        );
        // Each limit is the first size in the band above it, which is what
        // `size_band`'s `>=` implements.
        for (i, limit) in SIZE_BAND_LIMITS.iter().enumerate() {
            assert_eq!(size_band(limit - 1), i, "{limit} - 1 belongs below");
            assert_eq!(size_band(*limit), i + 1, "{limit} itself belongs above");
        }
    }

    /// The band boundaries, at every edge and one byte either side.
    ///
    /// An absolute reference per §3.1b rule 1: each case names the band it must
    /// land in, so a moved boundary fails with the size that moved rather than
    /// with "the bands were not monotone".
    #[test]
    fn size_band_puts_each_size_in_its_named_band() {
        for (size, want) in [
            (0usize, 0usize),
            (1, 0),
            (4095, 0),
            (4096, 1),
            (4097, 1),
            (21504, 1), // §9.2c's largest activation slot
            (262_143, 1),
            (262_144, 2),
            (3_833_856, 2), // one K or V at kv_len 234 (§9.2s)
            (4_194_303, 2),
            (4_194_304, 3),
            (1 << 30, 3),
        ] {
            assert_eq!(
                size_band(size),
                want,
                "size {size} landed in band {} ({}), expected {want} ({})",
                size_band(size),
                SIZE_BAND_NAMES[size_band(size)],
                SIZE_BAND_NAMES[want]
            );
        }
    }

    /// The bands are a partition of the whole-pool counters, not a sample.
    ///
    /// This is what licenses reading a band figure against the headline
    /// (§9.2u). Without it a band could silently miss a path -- `acquire`'s hit
    /// or `adopt_requesting`'s miss -- and every per-band share would be over a
    /// population smaller than the one it is compared to, which is
    /// `CONTRIBUTING.md` §3.1b's *"looks like a comparison and is not"*.
    ///
    /// Both paths are exercised deliberately, and the hit is an **over-served**
    /// one (3000 B answered by the 4096 B bucket) so the hit half contributes
    /// overhang rather than only bytes.
    #[test]
    fn bands_sum_to_the_whole_pool_counters() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Misses, in two different bands.
        for (requested, allocated) in [(1000usize, 1024usize), (300_000, 300_032)] {
            let raw = dev
                .new_buffer(allocated, crate::kernels::RESOURCE_OPTIONS)
                .expect("buffer allocation");
            let _ = pool.adopt_requesting(raw, requested, allocated);
        }

        // A third miss, whose buffer is then freed to set up an over-served hit.
        let raw = dev
            .new_buffer(4096, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = pool.adopt_requesting(raw, 4096, 4096);
        release_and_complete(&pool, &clock, b);
        let hit = pool.acquire(3000).expect("the 4096 bucket answers 3000");
        assert_eq!(
            hit.size(),
            4096,
            "the hit must be over-served, or the overhang contributed by the \
             hit path is zero and this test is vacuous on half its population"
        );

        let c = pool.counters();
        let band_req: u64 = c.bands.iter().map(|b| b.requested_bytes).sum();
        let band_srv: u64 = c.bands.iter().map(|b| b.served_bytes).sum();
        let band_acq: u64 = c.bands.iter().map(|b| b.acquisitions).sum();

        assert_eq!(
            band_req, c.requested_bytes,
            "bands requested {band_req} != pool requested {}",
            c.requested_bytes
        );
        assert_eq!(
            band_srv, c.served_bytes,
            "bands served {band_srv} != pool served {}",
            c.served_bytes
        );
        assert_eq!(band_acq, 4, "three misses and one hit were recorded");

        // Absolute references rather than a shape (§3.1b rule 1), so a
        // mis-banded acquisition fails naming the band that received it.
        // Band 0 (<4KB): the 1000 B miss and the 3000 B hit.
        assert_eq!(
            c.bands[0].acquisitions, 2,
            "1000 B and 3000 B are both <4KB"
        );
        assert_eq!(c.bands[0].requested_bytes, 1000 + 3000);
        assert_eq!(c.bands[0].served_bytes, 1024 + 4096);
        assert_eq!(c.bands[0].overhang_bytes(), 24 + 1096);
        // Band 1 (4KB-256KB): the 4096 B miss that set up the hit.
        assert_eq!(c.bands[1].acquisitions, 1, "4096 B is 4KB-256KB");
        assert_eq!(c.bands[1].overhang_bytes(), 0, "it was served exactly");
        // Band 2 (256KB-4MB): the 300 000 B miss.
        assert_eq!(c.bands[2].acquisitions, 1, "300000 B is 256KB-4MB");
        assert_eq!(c.bands[2].overhang_bytes(), 32);
        assert_eq!(c.bands[3].acquisitions, 0, "nothing here reached 4MB");
    }

    /// A request over-served from a larger bucket stays in the REQUEST's band.
    ///
    /// The keying decision §9.2u rests on, and the assertion that dies if the
    /// two keys are swapped. Banding on the served size would move this
    /// acquisition into the bucket's band, where the overhang would then appear
    /// in the large band **by construction** -- which is exactly the reading
    /// the split exists to refute.
    #[test]
    fn an_over_served_request_is_banded_by_what_was_asked_for() {
        let dev = device();
        let (pool, clock) = pool_with_clock();

        // Free a 1 MB buffer (band 2), then ask for 2 KB (band 0).
        let raw = dev
            .new_buffer(1 << 20, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let big = pool.adopt_requesting(raw, 1 << 20, 1 << 20);
        release_and_complete(&pool, &clock, big);

        let before = pool.counters();
        let served = pool.acquire(2048).expect("the 1 MB bucket answers 2 KB");
        assert_eq!(
            served.size(),
            1 << 20,
            "the range query must over-serve here"
        );

        let c = pool.counters();
        let d0 = c.bands[0].acquisitions - before.bands[0].acquisitions;
        let d2 = c.bands[2].acquisitions - before.bands[2].acquisitions;
        assert_eq!(d0, 1, "the 2 KB REQUEST belongs to band 0 (<4KB)");
        assert_eq!(d2, 0, "banding on the 1 MB BUCKET would have put it here");

        let oh = c.bands[0].overhang_bytes() - before.bands[0].overhang_bytes();
        assert_eq!(
            oh,
            (1 << 20) - 2048,
            "the whole over-service is attributed to the asking band"
        );
    }
}
