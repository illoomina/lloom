//! The scratch class: kernel scratch sized by `kv_len` (`DESIGN.md` §9.1, #71).
//!
//! # The class §9.1 was missing, and why it is the one worth getting right
//!
//! §9.1 names four memory classes -- weights, conv state, KV cache, activations.
//! FlashDecoding partials are none of them. §9.4 counts their *fences* (two per
//! attention layer per step) and nothing in the document sized them. At B=1,
//! F32 accumulation, page size 256 (§10.4):
//!
//! ```text
//! kv_len   2,720 (the largest ever measured) ->   11 chunks -> 0.09 MB
//! kv_len  32,768                             ->  128 chunks -> 1.03 MB
//! kv_len 131,072                             ->  512 chunks -> 4.12 MB
//! kv_len 262,144                             -> 1024 chunks -> 8.25 MB
//! ```
//!
//! Against a **68 KB** activation arena (§9.2c, measured by #68), that is ~62x
//! at 128k. It is also the **only class whose size depends on `kv_len`**, which
//! is the axis this project has never entered -- every measurement in
//! `measurements/` is at `kv_len` < 2720.
//!
//! It cannot be shrunk to f16. Partials accumulate in F32 per §8.1 principle 4,
//! and the combine merges in index order per §10.4's determinism constraint --
//! see [`CombineOrder`].
//!
//! # Sizing is a variant axis, not a decision
//!
//! Three policies are plausible and **which one wins is unmeasured**, because
//! the regime that distinguishes them is long context. So none is picked:
//! [`Sizing`] is a compile-tier policy in §7.1's sense, all three variants are
//! compiled, and the choice is made at construction. That is exactly the
//! discipline `ParamStyle` follows for binding styles (§11.3b) and
//! [`ArenaLayout`](super::ArenaLayout) for layouts -- keeping every variant live
//! is what makes the A/B free and keeps the comparison between paths that exist
//! rather than between one path and an argument.
//!
//! §7.2 assigns the tier: sizing changes addressing and loop bounds, so it is
//! **compile tier**, pre-built and selected at construction. The Metal-side
//! precedent is already in the tree -- `reduce.metal` carries
//! `template<typename OP, ushort BLOCKSIZE, typename T>`, a non-dtype policy
//! parameter -- and this is the Rust-side mirror of the same move.
//!
//! # Why this lands here rather than inside a later KV issue
//!
//! **This is the API the KV cache work will build on.** Getting the policy seam
//! right once, statically, means the KV allocator inherits a shape rather than
//! inventing one -- and static means deterministic, which §2.3 makes an
//! invariant rather than a preference.
//!
//! # What this deliberately does not do
//!
//! **It does not implement FlashDecoding.** The kernel is Phase 4/5 (§17). This
//! sizes and allocates its scratch, and a stub that writes the right *shape*
//! drives the allocation, because the memory behaviour is what is being
//! designed. Nothing here computes attention.

use super::{align_up, ARENA_ALIGNMENT};

/// Bytes per F32 accumulator element.
///
/// F32 and not f16, and that is §8.1 principle 4 rather than a default: partials
/// accumulate in F32 and are stored in it, because the combine merges them and a
/// merge in half precision loses what the accumulation bought. §9.1 states the
/// consequence -- **this class cannot be shrunk to f16** -- and this constant is
/// where it is spelled.
pub const PARTIAL_ELEM_BYTES: usize = 4;

/// Extra F32s carried per (head, chunk) beside the accumulator: the online
/// softmax running maximum `m` and denominator `l` (§10.4, glossary).
///
/// Two, and they are what make the chunks independently computable and
/// associatively mergeable. Named rather than inlined as `2` because the count
/// is a property of the online-softmax formulation, not an arbitrary padding.
pub const PARTIAL_STATS: usize = 2;

/// How a combine kernel consumes the partials.
///
/// # This is the single most likely place for nondeterminism to enter
///
/// §10.4 says so in those words, and §2.3.3 #1 is the rule: **every reduction
/// merges in fixed index order, never completion order.** Online softmax merging
/// is associative *mathematically* and **not** in floating point, so a
/// completion-ordered merge gives different bits run to run depending on which
/// chunk finished first.
///
/// The symptom is a generation that diverges from a previous run after a few
/// hundred tokens -- which §2.3.2 says is **indistinguishable from a missing
/// fence**. It is free to prevent and expensive to diagnose later.
///
/// So the order is a value that can be asserted rather than a property hoped
/// for. [`ScratchPlan::combine_order`] returns it and
/// [`ScratchPlan::check_index_ordered`] refuses anything else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CombineOrder {
    /// Partials are merged by ascending chunk index. **The only admissible
    /// order**, and the default.
    #[default]
    Index,
    /// Merged as chunks complete. Kept named rather than merely absent so the
    /// rule has something to exclude: a test that asserts "the order is Index"
    /// with no other spelling in the type system asserts very little.
    ///
    /// Nothing constructs this outside the test that shows
    /// [`ScratchPlan::check_index_ordered`] rejects it.
    Completion,
    /// Merged in the order the **page table** lists the chunks.
    ///
    /// # The second wrong order, and the one a B=1 gate cannot see
    ///
    /// §10.3h: a page table threatens §10.4's determinism constraint *"not by
    /// changing the order, but by making the order a property of a data
    /// structure rather than of a loop bound"*. Both this and [`Index`] visit
    /// every chunk:
    ///
    /// ```text
    /// for (uint c = 0; c < n_chunks; ++c) { merge(partial[c]); }   // Index -- correct
    /// for (page in page_list)             { merge(partial_for(page)); }  // TableOrder
    /// ```
    ///
    /// The second is not bit-stable, because a table's *contents* depend on
    /// allocation history — which under B>1 depends on **what other sequences
    /// did**. That would make a sequence's logits depend on its batch
    /// neighbours, violating §2.3.3 #7 directly.
    ///
    /// **It is strictly sharper than [`Completion`]**, and that is why it needed
    /// its own spelling rather than being folded into it: a completion-ordered
    /// merge varies run to run on one sequence, where this one is perfectly
    /// *reproducible* for a given allocation history and wrong only across
    /// histories. §10.3h: *"a B=1 determinism gate structurally cannot detect
    /// it"* — at B=1 with a fresh pool the table is `page_table[c] == c` and the
    /// two orders coincide exactly.
    ///
    /// # Why it exists now
    ///
    /// §10.3h asks for it *"as a second negative control"*, and #181's survey
    /// recorded it as **the only one of six deliberately-wrong arms this
    /// document names that was never built** — re-counted at zero by #568, four
    /// days and ~40 commits later. It was deferred on the ground that *"the arm
    /// it controls for is indistinguishable from `Index` at B=1"*, which is
    /// exactly the condition [`KvPagePool`](super::kv_pages::KvPagePool) removes:
    /// a recycled pool produces a non-identity table from an ordinary
    /// allocation history, so the arm is now distinguishable and the control has
    /// something to reject.
    ///
    /// Nothing constructs this outside the tests that show
    /// [`ScratchPlan::check_index_ordered`] rejects it.
    TableOrder,
}

/// Where a scratch region's byte extent comes from.
///
/// Three policies, all compiled, none chosen here. §7.2 puts the axis at
/// **compile tier** because sizing changes addressing and loop bounds.
///
/// | policy | shape | bets on |
/// |---|---|---|
/// | [`Reserve`](Sizing::Reserve) | `MAX_CHUNKS` from the configured max context, allocated once | fixed offsets; wastes MB at short context |
/// | [`Grow`](Sizing::Grow) | sized to the current `kv_len`, grown at chunk boundaries | minimal footprint; a realloc changes buffer identity |
/// | [`Bucket`](Sizing::Bucket) | N pre-planned sizes, picked per step | composes with the B-bucketing batching needs anyway (§13.6) |
///
/// **`Grow`'s cost is the one that is easy to miss.** A realloc changes buffer
/// identity, which is the precise thing #69 exists to prevent (§9.2c: the
/// arena's acceptance criterion is "674 varying identities -> 0"). So it needs a
/// rebind path, and [`ScratchPlan::rebinds_on_growth`] reports that rather than
/// leaving it as a comment -- a policy whose cost is invisible in the type is a
/// policy that gets chosen by accident.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sizing {
    /// Reserve for the configured maximum context, once.
    ///
    /// The default, and it is the *conservative* choice rather than the measured
    /// one: it is the only policy whose buffer identity is fixed for the
    /// process, so it is the one that cannot regress #69's stability property.
    /// It is **not** a claim that it wins -- see the module docs.
    #[default]
    Reserve,
    /// Size to the current `kv_len`, growing at chunk boundaries.
    Grow,
    /// Pick from a fixed ladder of pre-planned sizes.
    Bucket,
}

impl Sizing {
    /// Every policy, for the A/B harness and for exhaustive tests.
    ///
    /// An array rather than a derive so that adding a policy without extending
    /// the harness is visible: the length is asserted against the number of
    /// variants by `every_sizing_policy_is_in_all`, which is written as an
    /// exhaustive `match` so it cannot fall behind the enum (#58's mechanism,
    /// §11.3i).
    pub const ALL: [Sizing; 3] = [Sizing::Reserve, Sizing::Grow, Sizing::Bucket];

    /// The name segment this policy contributes to a `[[host_name]]`.
    ///
    /// The Metal side instantiates one kernel per policy from one body -- a
    /// non-dtype template parameter, exactly as `reduce.metal` carries
    /// `ushort BLOCKSIZE` (§8.1d) -- so the Rust side needs the same spelling
    /// the instantiation macro uses. Checked against the compiled library by
    /// `scratch_names_resolve` rather than against a second copy of the list,
    /// which is §8.1b's argument and what caught 48 absent `reduce` variants.
    pub fn suffix(self) -> &'static str {
        match self {
            Sizing::Reserve => "reserve",
            Sizing::Grow => "grow",
            Sizing::Bucket => "bucket",
        }
    }

    /// Whether a `kv_len` change under this policy can move the region's base
    /// buffer, requiring consumers to rebind.
    ///
    /// True only for [`Grow`](Sizing::Grow), and that is the property #69's
    /// stability criterion cares about (§9.2c). Reported as a value so a caller
    /// can refuse the policy where rebinding is not available, rather than
    /// discovering it as a wrong bind -- which under
    /// `HazardTrackingModeUntracked` is silent (§3.5).
    pub fn rebinds_on_growth(self) -> bool {
        matches!(self, Sizing::Grow)
    }

    /// Chunks a region reserves under this policy, given what the step needs.
    ///
    /// # Why this is a function and not four lines inside `plan_scratch`
    ///
    /// It **was** four lines inside [`plan_scratch`] until #234, and that is a
    /// large part of why the axis had three compiled arms and no consumer.
    /// `plan_scratch` owns *the whole step's* regions and lays them out at
    /// offsets; a caller that allocates one region per call needs none of that
    /// and needs exactly this one number. §9.1a records the wiring as deferred
    /// and §7.1b as the cheapest missing prerequisite, and the mismatch both
    /// name is between a **plan** and an **op** — the part of the plan an op
    /// actually needs is the sizing arithmetic alone.
    ///
    /// So this is the seam. [`plan_scratch`] calls it and so does
    /// FlashDecoding's allocation, which is what stops the two disagreeing
    /// about what a policy means — §8.1b's hand-sync argument, applied to a
    /// policy rather than to a kernel name.
    ///
    /// `live_chunks` is `ceil(kv_len / chunk_size)`. `max_chunks` bounds
    /// [`Reserve`](Sizing::Reserve) and is ignored by the others;
    /// `bucket_rungs` is [`BUCKET_LADDER`] converted to chunks by the caller,
    /// because a rung is a `kv_len` and a chunk count depends on the chunk size
    /// the caller chose.
    ///
    /// # Errors
    ///
    /// [`Reserve`](Sizing::Reserve) refuses a `live_chunks` past `max_chunks`
    /// and [`Bucket`](Sizing::Bucket) refuses one past its last rung, rather
    /// than rounding down. A region sized for fewer chunks than the step writes
    /// is an overrun, and §3.5 says nothing catches it.
    pub fn sized_chunks(
        self,
        live_chunks: usize,
        max_chunks: usize,
        bucket_rungs: &[usize],
    ) -> Result<usize, String> {
        match self {
            Sizing::Reserve => {
                if live_chunks > max_chunks {
                    // Phrased in chunks *and* named for what bounds it. The
                    // caller here holds only chunk counts — that is the whole
                    // reason this function takes them — but the number it
                    // converted is a **max context**, and an error a reader
                    // cannot trace back to the knob that set it is an error
                    // that costs a debugging session.
                    return Err(format!(
                        "live chunks {live_chunks} exceed the {max_chunks} the configured \
                         max context reserves"
                    ));
                }
                Ok(max_chunks)
            }
            Sizing::Grow => Ok(live_chunks),
            Sizing::Bucket => bucket_rungs
                .iter()
                .copied()
                .find(|&rung| live_chunks <= rung)
                .ok_or_else(|| {
                    format!(
                        "live chunks {live_chunks} exceed the largest bucket {:?}; refused \
                         rather than rounded down, which would size a region for fewer chunks \
                         than the step writes",
                        bucket_rungs.last()
                    )
                }),
        }
    }
}

/// The geometry one attention layer's partials occupy.
///
/// Plain numbers, and deliberately so: §15.2 #8 is *policy on the CPU, numbers
/// on the GPU*, and this is the CPU-side half. Swapping the sizing policy
/// changes what the numbers are, never what a kernel does with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialsGeometry {
    /// Query heads whose partials are held. 32 for LFM2 (§5.2).
    ///
    /// Query heads and not KV heads: a partial is per *query* head, because
    /// GQA broadcasts 8 KV heads to 32 query heads in registers (§8.3 item 2)
    /// and the accumulator is downstream of that broadcast.
    pub n_heads: usize,
    /// Accumulator width per head. 64 for LFM2 (§5.2).
    pub head_dim: usize,
    /// KV tokens per chunk. 256 proposed in §10.4 and **UNVERIFIED** there --
    /// carried as a field rather than a constant for that reason.
    pub page_size: usize,
    /// Batch size. Every measurement in this project is B=1 (§13.2).
    pub batch: usize,
}

impl Default for PartialsGeometry {
    /// LFM2.5-VL-3B at B=1, from §5.2's verified config.
    fn default() -> Self {
        Self {
            n_heads: 32,
            head_dim: 64,
            page_size: 256,
            batch: 1,
        }
    }
}

impl PartialsGeometry {
    /// Chunks a `kv_len` splits into: `ceil(kv_len / page_size)`.
    ///
    /// §10.6: contiguous is paged with one page, and single-pass attention is
    /// FlashDecoding with `n_chunks == 1`. A `kv_len` of 0 gives 0 chunks rather
    /// than 1, so an empty cache reserves nothing -- the degenerate case falls
    /// out instead of needing a branch.
    pub fn chunks(&self, kv_len: usize) -> usize {
        kv_len.div_ceil(self.page_size.max(1))
    }

    /// Bytes the accumulator plane needs for `chunks` chunks.
    ///
    /// `batch * n_heads * chunks * head_dim * 4`.
    pub fn accumulator_bytes(&self, chunks: usize) -> usize {
        self.batch * self.n_heads * chunks * self.head_dim * PARTIAL_ELEM_BYTES
    }

    /// Bytes the online-softmax statistics plane needs for `chunks` chunks.
    ///
    /// `batch * n_heads * chunks * 2 * 4` -- the running maximum `m` and the
    /// denominator `l` per (head, chunk).
    pub fn stats_bytes(&self, chunks: usize) -> usize {
        self.batch * self.n_heads * chunks * PARTIAL_STATS * PARTIAL_ELEM_BYTES
    }

    /// Total partial bytes for one attention layer at `chunks` chunks.
    ///
    /// This is the figure §9.1's table reports, and it reconciles exactly:
    /// `32 * 512 * (64 + 2) * 4 = 4325376 B = 4.125 MB` at `kv_len` 131072.
    pub fn partials_bytes(&self, chunks: usize) -> usize {
        self.accumulator_bytes(chunks) + self.stats_bytes(chunks)
    }

    /// Bytes for one (head, chunk) record when accumulator and statistics are
    /// **interleaved** rather than held in separate planes.
    ///
    /// `(head_dim + 2) * 4` = **264 B for LFM2**, which is *not* a multiple of
    /// [`ARENA_ALIGNMENT`]. That is the whole reason this function exists and is
    /// named -- see [`ScratchLayout::Interleaved`] and #70's warning, restated
    /// in §9.2c: **LFM2's own shapes cannot expose an alignment defect**, and
    /// this is the first shape in the project that can.
    pub fn interleaved_record_bytes(&self) -> usize {
        (self.head_dim + PARTIAL_STATS) * PARTIAL_ELEM_BYTES
    }
}

/// How a layer's partials are arranged within its region.
///
/// # Why both, and why the interleaved one is not merely an option
///
/// #70 earned a warning that §9.2c now carries: **every LFM2 decode activation
/// is a multiple of 128 B, so `align_up` is a no-op on every shape our own model
/// produces.** #70 deleted `align_up` and its acceptance test still *passed*,
/// until a deliberately unaligned size was added to the fixture.
///
/// The scratch class is the first consumer where this bites, and only under one
/// of these two layouts:
///
/// - [`Planes`](ScratchLayout::Planes): the accumulator plane is
///   `n_heads * chunks * 64 * 4` and the statistics plane is
///   `n_heads * chunks * 2 * 4`. Both are 128-multiples for LFM2's shapes, so
///   **this layout is blind to an alignment defect exactly as the activations
///   were.**
/// - [`Interleaved`](ScratchLayout::Interleaved): each (head, chunk) record is
///   `(64 + 2) * 4` = **264 B**, which is not. Under this layout `align_up`
///   is load-bearing on our own model's shapes.
///
/// Both are carried, and the fixtures use both, because a fixture built only
/// from the blind one is testing the identity function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScratchLayout {
    /// Accumulator and statistics in separate planes. 128 B-aligned on LFM2's
    /// shapes.
    #[default]
    Planes,
    /// One `(head_dim + 2)` record per (head, chunk). **264 B on LFM2** -- the
    /// shape that can expose an alignment defect.
    Interleaved,
}

/// One attention layer's scratch region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchRegion {
    /// Byte offset within the scratch arena. [`ARENA_ALIGNMENT`]-aligned.
    pub offset: usize,
    /// Bytes reserved. Under [`Sizing::Reserve`] this is the maximum-context
    /// figure; under the others it is what the current step needs.
    pub size: usize,
    /// Chunks the region is sized for.
    pub chunks: usize,
}

impl ScratchRegion {
    /// The region's end, as a bump cursor would compute it.
    ///
    /// `offset + align_up(size, 128)`, not `offset + size`. The distinction is
    /// #70's `bump_capacity` finding and it is a real one -- see
    /// [`ScratchPlan::bump_capacity`].
    pub fn slot_end(&self) -> usize {
        self.offset + align_up(self.size, ARENA_ALIGNMENT)
    }

    /// Where the region's last *value* ends.
    pub fn value_end(&self) -> usize {
        self.offset + self.size
    }
}

/// A sized, laid-out scratch arena for one decode step.
///
/// Produced by [`plan_scratch`], which is a pure function: it allocates nothing
/// and touches no device, so the arithmetic is testable without a GPU and the
/// fixtures can contain shapes LFM2 never produces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScratchPlan {
    regions: Vec<ScratchRegion>,
    sizing: Sizing,
    layout: ScratchLayout,
    order: CombineOrder,
    /// `kv_len` the plan was built for.
    kv_len: usize,
    /// Chunks the current `kv_len` needs, whatever the regions are sized for.
    live_chunks: usize,
}

impl ScratchPlan {
    /// Assemble a plan from parts, without running [`plan_scratch`]'s
    /// arithmetic or its checks.
    ///
    /// Exists so a test can build a **deliberately broken** plan -- overlapping
    /// regions, a 64 B-aligned offset, a completion-ordered combine -- and show
    /// that the checks reject it. `CONTRIBUTING.md` §3.1: a test that cannot
    /// fail is not a test, and a checker with nothing invalid to reject is in
    /// exactly that position.
    ///
    /// Not `pub(crate)`: the A/B harness lives in `candle-examples` and builds
    /// the same mutations to demonstrate them outside the test binary.
    pub fn from_parts(
        regions: Vec<ScratchRegion>,
        sizing: Sizing,
        layout: ScratchLayout,
        order: CombineOrder,
        kv_len: usize,
        live_chunks: usize,
    ) -> Self {
        Self {
            regions,
            sizing,
            layout,
            order,
            kv_len,
            live_chunks,
        }
    }

    pub fn regions(&self) -> &[ScratchRegion] {
        &self.regions
    }

    pub fn sizing(&self) -> Sizing {
        self.sizing
    }

    pub fn layout(&self) -> ScratchLayout {
        self.layout
    }

    pub fn kv_len(&self) -> usize {
        self.kv_len
    }

    /// Chunks the *current* `kv_len` requires.
    ///
    /// Under [`Sizing::Reserve`] and [`Sizing::Bucket`] a region is sized for
    /// more than this, and the difference is the reservation waste
    /// [`Self::reserved_waste`] reports.
    pub fn live_chunks(&self) -> usize {
        self.live_chunks
    }

    /// The merge order a combine kernel must use.
    ///
    /// Always [`CombineOrder::Index`] from [`plan_scratch`]. Exposed so a
    /// consumer asserts it rather than assuming it -- §10.4, and the issue's
    /// "assert it" is this function plus [`Self::check_index_ordered`].
    pub fn combine_order(&self) -> CombineOrder {
        self.order
    }

    /// Bytes where the last *value* ends.
    pub fn arena_bytes(&self) -> usize {
        self.regions
            .iter()
            .map(ScratchRegion::value_end)
            .max()
            .unwrap_or(0)
    }

    /// Bytes a **bump allocator** must be given to reproduce this plan.
    ///
    /// **Not the same number as [`Self::arena_bytes`]**, and #70 records why:
    /// `arena_bytes` is where the last value ends, while a cursor rounds *every*
    /// request including the last, so it ends where the last *slot* ends. On
    /// `[100, 300, 5000]` that is 5512 against 5632.
    ///
    /// Handing a bump allocator the smaller figure makes it decline an ordinal
    /// that fits the plan perfectly -- a quiet loss of coverage rather than a
    /// corruption, so nothing goes red.
    ///
    /// **On LFM2's own shapes under [`ScratchLayout::Planes`] the two are
    /// equal**, because every plane size is a 128-multiple. They differ under
    /// [`ScratchLayout::Interleaved`], whose 264 B record is not. That is the
    /// blindness §9.2c warns about, made reachable.
    pub fn bump_capacity(&self) -> usize {
        self.regions
            .iter()
            .map(ScratchRegion::slot_end)
            .max()
            .unwrap_or(0)
    }

    /// Bytes reserved beyond what the current `kv_len` needs.
    ///
    /// Zero under [`Sizing::Grow`] by construction. This is the quantity the
    /// three policies trade against buffer stability, and it is the one an A/B
    /// at long context would resolve -- see the crate's scratch A/B example.
    pub fn reserved_waste(&self, geometry: &PartialsGeometry) -> usize {
        let live = self.per_region_bytes(geometry, self.live_chunks);
        self.regions
            .iter()
            .map(|r| r.size.saturating_sub(live))
            .sum()
    }

    fn per_region_bytes(&self, geometry: &PartialsGeometry, chunks: usize) -> usize {
        region_bytes(geometry, chunks, self.layout)
    }

    /// Every region is [`ARENA_ALIGNMENT`]-aligned and no two overlap.
    ///
    /// The overlap half is what matters. Two regions sharing bytes would alias
    /// two layers' partials, and §9.3 says nothing in the driver would catch it
    /// -- under `HazardTrackingModeUntracked` a wrong offset is silent
    /// corruption, not an error. Checked exhaustively rather than sampled, for
    /// the reason #68 gives: a sampled check on an aliasing invariant is worth
    /// very little.
    pub fn check_disjoint(&self) -> Result<(), String> {
        for (i, r) in self.regions.iter().enumerate() {
            if !r.offset.is_multiple_of(ARENA_ALIGNMENT) {
                return Err(format!(
                    "region {i} at offset {} is not {ARENA_ALIGNMENT} B aligned",
                    r.offset
                ));
            }
        }
        for (i, a) in self.regions.iter().enumerate() {
            for (j, b) in self.regions.iter().enumerate().skip(i + 1) {
                if a.offset < b.value_end() && b.offset < a.value_end() {
                    return Err(format!(
                        "regions {i} ({}..{}) and {j} ({}..{}) overlap",
                        a.offset,
                        a.value_end(),
                        b.offset,
                        b.value_end()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Refuse any combine order but [`CombineOrder::Index`].
    ///
    /// §2.3.3 #1 and §10.4. This is the "assert it, do not assume it" the issue
    /// asks for, and it is a function rather than a comment because the failure
    /// it guards is invisible: a completion-ordered merge computes a *plausible*
    /// answer that differs run to run, which §2.3.2 says looks exactly like a
    /// missing fence.
    pub fn check_index_ordered(&self) -> Result<(), String> {
        match self.order {
            CombineOrder::Index => Ok(()),
            CombineOrder::Completion => Err(
                "combine order is Completion; online softmax is associative in R but not in \
                 floating point, so a completion-ordered merge is nondeterministic \
                 (DESIGN.md 2.3.3 #1, 10.4)"
                    .to_string(),
            ),
            CombineOrder::TableOrder => Err(
                "combine order is TableOrder; the page table must be INDEXED by chunk number \
                 and never WALKED, because its contents depend on allocation history and under \
                 B>1 that depends on what other sequences did -- which would make a sequence's \
                 logits depend on its batch neighbours (DESIGN.md 2.3.3 #7, 10.3h). Note this \
                 is reproducible for a given history and therefore invisible to a run-to-run \
                 digest gate, and identical to Index at B=1"
                    .to_string(),
            ),
        }
    }

    /// The chunk indices a combine kernel must read, in the order it must read
    /// them.
    ///
    /// Ascending, always. Returned as a sequence rather than as a promise so a
    /// test can compare against it -- an assertion that a kernel "merges in
    /// index order" with nothing to compare against asserts nothing.
    ///
    /// **Independent of the page table by construction**, which is the property
    /// §10.3h asks for: the combine iterates chunk index `0..n_chunks` and the
    /// table is *indexed* by that, never *walked*. See
    /// [`Self::merge_sequence_under`] for what the rejected orders would produce
    /// — a control that cannot be evaluated is not a control.
    pub fn merge_sequence(&self) -> Vec<usize> {
        (0..self.live_chunks).collect()
    }

    /// What a combine would read under `order`, given the page table `table`.
    ///
    /// # Why a rejected arm needs to be computable
    ///
    /// [`Self::check_index_ordered`] refuses everything but
    /// [`CombineOrder::Index`], and a refusal alone cannot demonstrate that the
    /// refused thing is *different*. §10.3h's hazard is precisely that
    /// [`CombineOrder::TableOrder`] and `Index` **coincide** at
    /// `page_table[c] == c` — so an arm that is only ever rejected leaves the
    /// reader to take on faith that it would diverge elsewhere.
    ///
    /// This computes both, so a test can show they agree on a fresh pool's
    /// identity table and disagree on a recycled pool's — which is the whole of
    /// why §10.3i sequences the allocator before the kernel.
    ///
    /// [`CombineOrder::Completion`] has no deterministic sequence to return by
    /// its own definition; it is reported as `None` rather than given an
    /// arbitrary one, since inventing an order for the arm that is defined by
    /// not having one would be the vacuity §15.1a names.
    pub fn merge_sequence_under(&self, order: CombineOrder, table: &[u32]) -> Option<Vec<usize>> {
        match order {
            CombineOrder::Index => Some(self.merge_sequence()),
            CombineOrder::Completion => None,
            // The table's entries are page indices; walking it means visiting
            // chunks in the order their pages happen to be listed.
            CombineOrder::TableOrder => {
                let mut idx: Vec<usize> = (0..self.live_chunks.min(table.len())).collect();
                idx.sort_by_key(|&c| table[c]);
                Some(idx)
            }
        }
    }

    /// The byte sizes a bump allocator walks to reproduce this plan.
    ///
    /// One entry per region, in order. Shaped to match
    /// [`StepPlan::request_sizes`](super::StepPlan::request_sizes) so the scratch
    /// arena can be served by the **same** GPU bump allocator #70 built, rather
    /// than by a second one that would have to be verified separately.
    pub fn request_sizes(&self) -> Vec<u32> {
        self.regions.iter().map(|r| r.size as u32).collect()
    }

    /// The offset each region must receive, for
    /// [`ArenaCursor::verify_against`](super::ArenaCursor::verify_against).
    pub fn expected_offsets(&self) -> Vec<Option<usize>> {
        self.regions.iter().map(|r| Some(r.offset)).collect()
    }
}

/// Bytes one region occupies at `chunks` chunks under `layout`.
///
/// The two layouts differ in **alignment behaviour**, not in total data: an
/// interleaved region pads each (head, chunk) record up to 128 B, so it is
/// larger and its size is the one that exercises `align_up`.
fn region_bytes(geometry: &PartialsGeometry, chunks: usize, layout: ScratchLayout) -> usize {
    match layout {
        ScratchLayout::Planes => geometry.partials_bytes(chunks),
        ScratchLayout::Interleaved => {
            // Each record is padded to the alignment so a record boundary is a
            // cache-line boundary, which is the same reasoning §9.2 gives for
            // slots. 264 -> 384 on LFM2's shapes.
            let record = align_up(geometry.interleaved_record_bytes(), ARENA_ALIGNMENT);
            geometry.batch * geometry.n_heads * chunks * record
        }
    }
}

/// The bucket ladder [`Sizing::Bucket`] picks from, in `kv_len` tokens.
///
/// 8k / 32k / 128k, as the issue proposes. A ladder rather than a formula
/// because it **composes with the B-bucketing batching needs anyway** (§13.6):
/// `B` is in the dispatch grid for every op, so it is already a bucketed axis,
/// and a second bucketed axis alongside it is one mechanism rather than two.
///
/// The last rung is the ceiling: a `kv_len` past it is refused rather than
/// silently rounded down, because rounding down would size a region for fewer
/// chunks than the step will write -- an overrun, and §3.5 says nothing catches
/// it.
pub const BUCKET_LADDER: [usize; 3] = [8_192, 32_768, 131_072];

/// Build a scratch plan.
///
/// `layers` is the number of attention layers needing partials -- **8 for LFM2**
/// (§5.3: attention at indices 2, 5, 9, 13, 17, 21, 24, 27), not 30.
///
/// `max_context` bounds [`Sizing::Reserve`] and is ignored by the others.
///
/// # Regions do not share bytes, and that is not a packing failure
///
/// The activation arena packs values whose live ranges are disjoint (§9.2). The
/// partials of two attention layers within one decode step are **not**
/// obviously disjoint -- each layer's combine reads its own partials, and
/// whether layer *l+1*'s partials may reuse layer *l*'s bytes depends on the
/// fence placement §9.4 specifies rather than on liveness alone. Packing them
/// would be a claim about ordering this issue has no evidence for, and §9.3 says
/// a wrong such claim is silent.
///
/// So each layer gets its own region. That is the conservative direction: it
/// costs bytes and never correctness, and it leaves the packing question open
/// for the issue that builds the kernel and can measure it.
pub fn plan_scratch(
    geometry: &PartialsGeometry,
    kv_len: usize,
    layers: usize,
    sizing: Sizing,
    layout: ScratchLayout,
    max_context: usize,
) -> Result<ScratchPlan, String> {
    let live_chunks = geometry.chunks(kv_len);

    // Routed through [`Sizing::sized_chunks`] rather than inlined here as of
    // #234. It was inlined, and a second consumer -- FlashDecoding's per-call
    // allocation -- could then only have re-implemented it, which is the
    // hand-sync §8.1b exists to remove. The refusals move with it: `Reserve`
    // compared `kv_len > max_context` and now compares the chunk counts those
    // map to, which is the same verdict because `chunks()` is monotone, and it
    // is the comparison a caller holding only chunk counts can make.
    let rungs: Vec<usize> = BUCKET_LADDER.iter().map(|&b| geometry.chunks(b)).collect();
    let sized_chunks = sizing.sized_chunks(live_chunks, geometry.chunks(max_context), &rungs)?;

    let size = region_bytes(geometry, sized_chunks, layout);
    let mut regions = Vec::with_capacity(layers);
    let mut cursor = 0usize;
    for _ in 0..layers {
        regions.push(ScratchRegion {
            offset: cursor,
            size,
            chunks: sized_chunks,
        });
        // Every region is rounded, including the last -- which is what makes
        // `bump_capacity` differ from `arena_bytes` under a layout whose size is
        // not a 128-multiple (#70).
        cursor = align_up(cursor + size, ARENA_ALIGNMENT);
    }

    let plan = ScratchPlan {
        regions,
        sizing,
        layout,
        // §2.3.3 #1: there is no other admissible value, and `plan_scratch`
        // never produces one.
        order: CombineOrder::Index,
        kv_len,
        live_chunks,
    };
    plan.check_disjoint()?;
    Ok(plan)
}
