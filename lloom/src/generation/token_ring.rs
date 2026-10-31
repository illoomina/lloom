//! The device-resident token slot the host polls.
//!
//! `DESIGN.md` §11.5g's non-blocking token path, built. The design's own
//! finding is what makes this small: §11.5f established that the sampled token
//! makes a **round trip** -- down through `to_scalar` and back up through
//! `Tensor::new` as the next step's input -- and registered that a design
//! removing the read still owes the write. §11.5g read the source and found it
//! does not: `argmax()` returns a device tensor in `U32`, so
//!
//! > removing the read removes the **reason** for the write.
//!
//! **So there is one ring, not two.** The write leg is *deleted rather than
//! relocated*: the token is written in place into a persistent input buffer
//! with `slice_set` (`candle-core/src/tensor_cat.rs`), which is the mechanism
//! #141's conv ring already uses in this very model. Nothing is uploaded
//! because nothing came down.
//!
//! # The two legs are in different storage classes
//!
//! §11.5g's table, and the reason two issues' worth of mapping findings apply
//! to one leg only:
//!
//! | | read leg | write leg |
//! |---|---|---|
//! | storage | `StorageModePrivate` | `StorageModeShared` |
//! | mapped? | iff < 16384 B (§15.2 #12a) | **always**, at every size |
//! | blocks? | **yes** -- completion wait 0.190-0.205 ms | no |
//!
//! §15.2 #12a's 16384 B threshold, #454's arm-instability census and #366's
//! tested-pointer defect are properties of the **read** leg.
//!
//! # What is NOT claimed
//!
//! **No timing.** The prize is bounded at 0.175-0.188 ms per readback against
//! an 18.1625 ms step -- ~1.0 % -- and §11.5c's own end-to-end read of exactly
//! this seam was **-3.74 % against a null of +4.08 %, `unresolved`**. §11.5g
//! stacks three discounts on the 0.19 ms before anyone measures: there is no
//! cheap park (§3.7h), 0.175-0.188 is the *pre-drained best case*, and #454
//! measures which `to_cpu` arm fires as unstable run to run (blit 1->12 on one
//! binary, 8.7x in the corpus), so the figure is itself partly allocator-state
//! variance.
//!
//! **This is a `--temperature 0`, B=1 mechanism.** §6.6c records the readback
//! as `B` round trips per step at B>1, which makes the slot `B` slots and is
//! #263's change.

use lloom_tensor::{DType, Device, Result, Tensor};

/// Reads the poll made through a resolved address (#578).
///
/// **The fast path moves no other counter, and that is why this exists.**
/// `peek_hit` is incremented inside `MetalStorage::peek_cpu`; a poll that reads
/// through an address resolved once never calls it, so the engagement figure
/// for the fast path would otherwise be **zero on a path that ran every time**
/// -- indistinguishable from a path that never ran (§3.1c: *a zero is not
/// evidence unless something nonzero came from the same command*). An
/// optimisation whose engagement nobody counts is one whose null reads as a
/// result (§2.4).
static POLL_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reads made through the resolved address since process start (#578).
///
/// Pairs with `lloom_tensor::metal_backend::peek_counts` the way that pair is used on
/// a `RESULT` line: together they say which arm of the poll carried the run.
pub fn poll_read_count() -> u64 {
    POLL_READS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A persistent device-resident input token, written by the GPU.
///
/// Holds the state that survives a decode step:
///
/// * `input` -- the `[1, 1]` `U32` tensor the model reads as the next step's
///   input. Written **in place** by [`Self::publish`], never reallocated.
/// * `eos` -- a single-element `U8` tensor, set GPU-side when the sampled token
///   matched one of the EOS ids.
/// * `eos_ids` -- the EOS set, uploaded once.
///
/// # Why the flag is narrow
///
/// §3.7g measured a 64-bit publish tearing: MSL `device` atomics are 32-bit, so
/// a wider store is two stores and **11 of 20 reader-arms latched a torn pair**,
/// always exactly one generation apart. Anything above 4 bytes needs a version
/// protocol. §11.5g's *"the flag is one 32-bit word"* is what skips it; a `U8`
/// comparison result is narrower still and is what `broadcast_eq` produces
/// natively, so no cast is added to widen it back.
///
/// # Lifetime
///
/// These are **session state, not activation state**, and are held for the
/// length of the generation. §9.1 requires session state to stay out of the
/// activation arena, and §9.2c measures what happens when it does not: the conv
/// state entered the arena and §6.1's shuffle then *read a slot it also wrote*.
/// Holding the handles is also what keeps them away from the pool's reuse scan
/// (§9.2a layer 3), so §6.7 L4's two-clocks defect -- `Arc::strong_count == 1`
/// says nothing about GPU liveness -- cannot reach them.
pub struct TokenRing {
    /// The INPUT staging ring — a [`SegmentedRing`] whose tiles are input CHUNKS
    /// (each `quantum = max_width` tokens wide), sized to `depth + 1` tiles so a
    /// host-staged chunk never overwrites a buffer an in-flight forward is reading.
    /// Reuses the history ring's segmented-mapped-buffer mechanism, so it stays
    /// zero-copy at ANY chunk width (segments ≤ the 16384 B window) — unlike a flat
    /// `[1, max_width]` buffer, which de-maps and copies per stage past 4092 slots.
    /// The decode path (device-side `publish`, queue-ordered) does not strictly need
    /// `depth + 1`, but cycling tiles is harmless there. The input-side twin of the
    /// history's "don't overwrite what's in flight" discipline (§10.3e).
    inputs: super::readback::SegmentedRing,
    /// The tile the NEXT stage/publish writes and the next forward reads. Advances
    /// once per input write, wrapping at the ring's tile count.
    slot: std::cell::Cell<usize>,
    /// The per-tile token capacity (`= inputs.quantum()`). A stage wider than this
    /// is refused rather than silently truncated.
    max_width: usize,
    /// Pre-allocated CYCLED SOURCE ring for the generation stamp off Metal — the
    /// whole of why [`Self::publish_marker`] does no hot-path allocation there.
    /// On Metal the stamp is a kernel writing a constant and this is not read.
    ///
    /// Writing the stamp is `step.slice_set(src)`, and `slice_set` reads `src` at
    /// GPU-EXECUTE time (not encode time). A single reused source would tear: the
    /// host staging stamp N+1 into it could race the still-pending `slice_set` of
    /// stamp N. The cure is the input ring's own discipline — cycle `depth + 1`
    /// sources so the one being staged is never the one an in-flight `slice_set`
    /// reads (the pacer bounds in-flight writes to `depth`). Reuses [`SegmentedRing`]
    /// (quantum 1, `depth + 1` one-slot tiles), the same battle-tested
    /// zero-copy-staged, mapped, pre-allocated mechanism as `inputs` — so the stamp
    /// value is staged with `write_host_slice` (no alloc) and copied to `step` with
    /// one queue-ordered `slice_set`, and NOTHING is allocated per publish.
    marker_src: super::readback::SegmentedRing,
    /// The marker-source tile the next stamp stages into; cycles over
    /// `marker_src`'s `depth + 1` tiles. Single-thread (the decode loop owns it),
    /// so a `Cell` rather than an atomic — exactly like `slot`.
    marker_slot: std::cell::Cell<usize>,
    eos: Tensor,
    eos_ids: Tensor,
    /// The event a host waiter parks on instead of polling the stamp, made on
    /// first arming and signalled after each stamp with its generation while
    /// armed ([`Self::arm_publish_event`]). Unarmed, a stamp encodes nothing
    /// more (`expert-streaming.md` §20.14b).
    #[cfg(feature = "metal")]
    publish_event:
        std::cell::RefCell<Option<std::sync::Arc<lloom_tensor::metal_backend::PublishEvent>>>,
    publish_armed: std::cell::Cell<bool>,
    /// The published-generation counter, and the reason it exists is #535.
    ///
    /// A **flag** cannot be polled non-blockingly and a **counter** can, which
    /// is a property of the reader rather than of the writer. Reading the flag
    /// through a blocking `to_vec1` is unambiguous because the wait is what
    /// establishes that the GPU has finished: whatever the flag says afterwards
    /// is this step's answer. Remove the wait -- which is the whole point of
    /// §11.5 -- and `eos == 0` becomes two states the host cannot tell apart:
    ///
    /// * the GPU has published and the token was **not** EOS, or
    /// * the GPU **has not published yet**.
    ///
    /// Both read zero. A host that treats the first `Some(0)` as "not EOS"
    /// races, and a host that waits for a `1` that will never come for a
    /// non-EOS token hangs. This is not a hazard the peek introduces; it is one
    /// the *blocking* read was silently discharging.
    ///
    /// So the ring publishes a **monotone counter** the host compares against a
    /// value it already knows. `step` is written with the step's own ordinal at
    /// the end of `publish`, so the host polls until it observes its own
    /// expected value and only then reads the flag beside it -- which by then
    /// has been written, because the counter is written after it.
    ///
    /// **The ordering that makes this work is the only unproven link and it is
    /// stated rather than assumed.** MSL documents no ordering for a plain
    /// store and `device` atomics accept only `memory_order_relaxed` (§9.3), so
    /// *counter written after flag* is a source-order claim, not a
    /// visibility-order guarantee. #535 measured payload-behind-flag at
    /// **0 of 200** tears with a `mem_device` barrier between them, and records
    /// that as a **bound rather than a guarantee**. The consequence is drawn in
    /// [`Self::poll_eos`]: the counter is what is polled, the flag is read
    /// after it, and a caller that cannot accept that bound has
    /// [`Self::eos_flag`]'s blocking read unchanged beside it.
    step: Tensor,
    /// The mapped addresses of `step[0]` and `eos[0]`, resolved once (#578).
    ///
    /// **This is the whole of #578's first defect.** `poll_eos` used to reach
    /// `peek_vec1`, which acquires the storage `RwLock`, materialises a `Vec`
    /// of the whole storage and then a second `Vec` for the slice -- two heap
    /// allocations and a lock to read one `u32`, at ~283 000 poll iterations
    /// per token. The non-blocking path existed to stop touching the allocator
    /// on the decode path (§11.5n: `to_cpu_fast` 65 -> 1) and was replacing a
    /// blocking readback with an allocator storm.
    ///
    /// Resolving the address once is sound here for the reason §10.3e names:
    /// these allocations are **stable by design** -- `publish` writes in place
    /// with `slice_set` and never reallocates, and the tensors are held by this
    /// struct for the length of the generation, which is the same property that
    /// keeps them out of the pool's reuse scan (§9.2a layer 3).
    ///
    /// `None` when either allocation is unmapped -- an allocator property
    /// (§15.2 #12a's 16384 B threshold) that #454 measures as unstable run to
    /// run -- in which case [`Self::poll_eos`] falls back to the locking peek
    /// rather than failing, so the fast path is an optimisation and not a new
    /// precondition.
    mapped: Option<Mapped>,
}

/// The two addresses [`TokenRing::poll_eos`] reads, and the `Tensor`s they
/// belong to are owned by the same struct.
///
/// Raw pointers are not `Send`/`Sync` and the ring must stay both, so they are
/// held as `usize` and rebuilt at the read. That is a lint accommodation rather
/// than a semantic one: the addresses are valid for exactly as long as the
/// `TokenRing` that owns the tensors, which is what
/// `Tensor::peek_scalar_ptr`'s safety contract requires.
#[cfg(feature = "metal")]
struct Mapped {
    step: usize,
    eos: usize,
}

#[cfg(not(feature = "metal"))]
struct Mapped {
    _never: std::convert::Infallible,
}

impl TokenRing {
    /// Allocates the slot and uploads the EOS ids once.
    ///
    /// `eos_ids` is a loop invariant: `lfm2-release` builds it before the loop
    /// from config plus tokenizer, so it is a device constant uploaded once and
    /// the comparison is a handful of `u32`s (§11.5g). Sampling EOS from the
    /// logits instead is ruled out on the corpus's own number -- §11.2a prices
    /// the host EOS test at **0.0002 ms/token**, so it would remove something
    /// already off the critical path.
    ///
    /// An empty `eos_ids` is rejected rather than accepted as "never
    /// terminates": the harness obligation §11.5f identifies (the model never
    /// checks EOS -- `eos_token_id` appears three times in `lfm2.rs` and all
    /// three are plumbing) has no model-side counterpart to fall back on.
    pub fn new(first_input: u32, eos_ids: &[u32], device: &Device) -> Result<Self> {
        Self::with_input_ring(first_input, eos_ids, device, 1, 1)
    }

    /// As [`Self::new`], with the input buffer sized to hold up to `max_width`
    /// tokens rather than one. One slot — for a caller with no concurrent forwards
    /// staging host tokens (the incumbent single-input path).
    pub fn with_input_width(
        first_input: u32,
        eos_ids: &[u32],
        device: &Device,
        max_width: usize,
    ) -> Result<Self> {
        Self::with_input_ring(first_input, eos_ids, device, max_width, 1)
    }

    /// As [`Self::with_input_width`], with an input RING of `n_slots` buffers so a
    /// host-staged chunk does not overwrite an in-flight forward's input (see the
    /// `inputs` field). `n_slots` = the pacer's in-flight depth + 1; `1` reduces to
    /// the single-buffer path.
    ///
    /// Each buffer holds up to `max_width` tokens and **decouples what is STAGED
    /// from what one forward CONSUMES** — the state machine steps over a prompt in
    /// `ubatch`-wide units (streamed prefill) or a drafted block in `K+1`-wide
    /// units, writing them into the current slot (zero-copy
    /// [`Tensor::write_host_slice`] for a host source) and consuming a
    /// [`Self::input_for`] narrow of the width that step eats. Buffers are
    /// `Tensor::empty` (no fill, no blit, no host upload — the content is written
    /// before it is read), allocated ONCE.
    pub fn with_input_ring(
        first_input: u32,
        eos_ids: &[u32],
        device: &Device,
        max_width: usize,
        n_slots: usize,
    ) -> Result<Self> {
        if eos_ids.is_empty() {
            lloom_tensor::bail!("TokenRing: eos_ids is empty, so the loop would never terminate");
        }
        let w = max_width.max(1);
        let n = n_slots.max(1);
        // The input ring is a `SegmentedRing` whose TILE is one chunk (`quantum = w`
        // tokens) and which holds `n` tiles (= `depth + 1`). `cap = n · w` slots;
        // the segmentation keeps each buffer ≤ the 16384 B mapping window, so a
        // chunk of ANY width up to a segment stages zero-copy — the property a flat
        // `[1, w]` buffer loses past 4092 slots. (`SegmentedRing::new` zeros the
        // segments once at construction; the content is overwritten before it is
        // read, so the fill value is irrelevant — the seed below just makes slot 0's
        // first token defined.)
        let inputs = super::readback::SegmentedRing::new(n * w, w, device)?;
        // Seed tile 0's first slot (the first forward's input), zero-copy through the
        // mapped pointer.
        inputs.stage_at_tile(0, &[first_input])?;
        // The marker-source ring: `n` (= depth + 1) one-slot tiles, quantum 1, so a
        // stamp staged into the current tile is never the one an in-flight
        // `slice_set` still reads (the pacer bounds in-flight to `depth`). Same
        // pre-allocated, mapped, zero-copy-staged mechanism as `inputs` — no
        // per-publish allocation. `n * 1` slots.
        let marker_src = super::readback::SegmentedRing::new(n, 1, device)?;
        let eos_ids = Tensor::from_vec(eos_ids.to_vec(), eos_ids.len(), device)?;
        // Materialised on the device so the first `publish` overwrites rather
        // than allocates. `from_vec` of a `0u8` is the same shape `publish`
        // writes, which is what makes the two comparable.
        let eos = Tensor::from_vec(vec![0u8], 1, device)?;
        // The generation counter starts at 0 and `publish` writes 1, 2, 3, ...
        // so the host's "has step N landed" test is `observed >= N` against a
        // value it chose, never against a value it read earlier. U32 is the
        // width §3.7g's tear finding leaves safe: MSL `device` atomics are
        // 32-bit, so one word is one store and 11-of-20 torn pairs is a
        // property of publishing *two*.
        let step = Tensor::from_vec(vec![0u32], 1, device)?;
        // Resolve the poll's two addresses once (#578). Both must be mapped for
        // the fast path to be usable -- a partial resolution would leave
        // `poll_eos` reading one word through a pointer and the other through
        // the locking peek, which is two contracts in one function for no gain.
        let mapped = Self::resolve_mapped(&step, &eos);
        Ok(Self {
            inputs,
            slot: std::cell::Cell::new(0),
            max_width: w,
            marker_src,
            marker_slot: std::cell::Cell::new(0),
            eos,
            eos_ids,
            #[cfg(feature = "metal")]
            publish_event: std::cell::RefCell::new(None),
            publish_armed: std::cell::Cell::new(false),
            step,
            mapped,
        })
    }

    /// Arm (or disarm) the publish event for the stamps encoded from here on.
    /// Arming makes it on first use; off Metal there is none and arming does
    /// nothing.
    pub fn arm_publish_event(&self, on: bool) -> Result<()> {
        #[cfg(feature = "metal")]
        if on {
            if let lloom_tensor::Device::Metal(m) = self.step.device() {
                let mut slot = self.publish_event.borrow_mut();
                if slot.is_none() {
                    *slot = Some(std::sync::Arc::new(
                        lloom_tensor::metal_backend::PublishEvent::new(m)?,
                    ));
                }
                self.publish_armed.set(true);
                return Ok(());
            }
        }
        let _ = on;
        self.publish_armed.set(false);
        Ok(())
    }

    /// The publish event, while armed.
    #[cfg(feature = "metal")]
    pub fn publish_event(
        &self,
    ) -> Option<std::sync::Arc<lloom_tensor::metal_backend::PublishEvent>> {
        if !self.publish_armed.get() {
            return None;
        }
        self.publish_event.borrow().clone()
    }

    /// The mapped addresses, or `None` if either allocation is unmapped.
    #[cfg(feature = "metal")]
    fn resolve_mapped(step: &Tensor, eos: &Tensor) -> Option<Mapped> {
        // SAFETY: the pointers are stored beside the tensors that own them in
        // the same struct, so they cannot outlive the allocations; `publish`
        // writes in place and never reallocates (§10.3e), so they stay valid
        // for the ring's lifetime. Nothing is read here.
        unsafe {
            let s = step.peek_scalar_ptr::<u32>(0)?;
            let e = eos.peek_scalar_ptr::<u8>(0)?;
            Some(Mapped {
                step: s as usize,
                eos: e as usize,
            })
        }
    }

    #[cfg(not(feature = "metal"))]
    fn resolve_mapped(_step: &Tensor, _eos: &Tensor) -> Option<Mapped> {
        None
    }

    /// Writes the sampled token into the input slot **in place** and computes
    /// the EOS flag, both on the device.
    ///
    /// `token` is the rank-0 `U32` device tensor
    /// `LogitsProcessor::sample_argmax_on_device` returns. Nothing here reads
    /// it host-side, which is the whole point: after this call the next
    /// `forward` can be encoded without the host ever having learned which
    /// token was sampled.
    ///
    /// # The `slice_set` contract, and why the reshape is not cosmetic
    ///
    /// `slice_set` refuses a non-contiguous side on either end
    /// (`Error::RequiresContiguous`) **and** refuses `self` and `src` sharing
    /// storage. §10.3e records the first constraint biting from the other
    /// direction -- `narrow` produces a strided view, which is why the two
    /// `Tensor`-level append mechanisms could not survive paging. Here both
    /// sides are freshly shaped `[1, 1]` contiguous `U32` over distinct
    /// allocations, so both hold; the reshape is what establishes the first.
    pub fn publish(&self, token: &Tensor) -> Result<()> {
        self.publish_step(token, None)
    }

    /// [`Self::publish`], additionally stamping the generation counter so the
    /// host can poll for *this* step without a synchronization point (#535).
    ///
    /// `generation` is the step's own ordinal, chosen by the caller and starting
    /// at 1 -- it is a value the host already holds, which is what makes the
    /// comparison sound: the host waits for a number it decided rather than for
    /// a change in a number it read.
    ///
    /// **The stamp is written last, and that ordering is the protocol.** The
    /// input slot, then the EOS flag, then the counter, which certifies the
    /// flag ([`Self::publish_marker`]): the GPU writes the counter after the
    /// flag. Whether the host then sees the two stores in that order is a
    /// measured bound rather than a documented one -- §9.3, #535's 0 of 200
    /// tears with a device barrier -- and [`Self::poll_eos`] states what a
    /// caller is accepting.
    pub fn publish_step(&self, token: &Tensor, generation: Option<u32>) -> Result<()> {
        self.publish_step_certifying(token, generation, None)
    }

    /// [`Self::publish_step`], with the stamp also certifying `slot` — a
    /// buffer the caller wrote for this step before calling (the history slot
    /// a drain reads up to the stamp). See [`Self::publish_marker`].
    pub fn publish_step_certifying(
        &self,
        token: &Tensor,
        generation: Option<u32>,
        slot: Option<&Tensor>,
    ) -> Result<()> {
        if token.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "TokenRing::publish expects the U32 argmax output, got {:?}",
                token.dtype()
            );
        }
        if generation == Some(0) {
            // 0 is the counter's initial value, so a step stamped 0 is
            // indistinguishable from one that never published -- exactly the
            // ambiguity this counter exists to remove. Rejected rather than
            // silently accepted (§2.4: make an invalid setting fail loudly).
            lloom_tensor::bail!(
                "TokenRing: generation 0 is the unpublished sentinel, ordinals start at 1"
            );
        }
        // `argmax` over the vocabulary dimension returns rank 0 (it squeezes);
        // `reshape` to `[1]` makes it a single contiguous slot value.
        let token = token.reshape(1)?;
        // Advance to a fresh tile, then device-write the token into its first slot
        // (queue-ordered after the prior forward's read); the forward reads
        // `input_for(1)` on this same tile.
        let tile = self.next_slot();
        self.inputs.write_device_at_tile(tile, &token)?;
        // The EOS decision, device-side and against a loop-invariant constant.
        // `broadcast_eq` against the uploaded id set then `max_all` reduces to
        // one element: nonzero iff the token matched any id. This is the
        // comparison §11.5g describes as "a handful of `u32`s".
        let hit = token
            .reshape(1)?
            .broadcast_as(self.eos_ids.shape())?
            .eq(&self.eos_ids)?
            .max_all()?
            .reshape(1)?;
        self.eos.slice_set(&hit, 0, 0)?;
        // The stamp, LAST — but only when the caller stamps at THIS granularity.
        // `Some(g)` is the per-step arm (the completion marker and the EOS flag
        // coincide, `live_every == 1`); `None` is the batched arm, where the loop
        // stamps once per commit window via [`Self::publish_marker`] instead, so
        // the per-step path writes no stamp at all.
        if let Some(g) = generation {
            match slot {
                Some(slot) => self.publish_marker(g, &[&self.eos, slot])?,
                None => self.publish_marker(g, &[&self.eos])?,
            }
        }
        Ok(())
    }

    /// Reuse the existing event notification after a guarded stamp. The event
    /// is a wakeup, not token-validity proof: readers still inspect the stamp,
    /// and the producer independently checks request errors when it is absent.
    #[cfg(feature = "metal")]
    pub(crate) fn signal_publication(&self, generation: u32) -> Result<()> {
        if self.publish_armed.get() {
            if let Some(ev) = &*self.publish_event.borrow() {
                ev.encode_signal(generation as u64)?;
            }
        }
        Ok(())
    }

    /// Stamp the generation counter after the writes it certifies, WITHOUT a
    /// per-call allocation — the completion marker the host WFE-waits on and
    /// the drain clamps to ([`Self::published_generation`]).
    ///
    /// `certified` (one or two tensors) are the buffers whose writes a host
    /// that observes the counter at `>= generation` will read: the history
    /// segment the slots went to, the EOS flag. Encoding the stamp after them
    /// is not enough on Metal. The compute encoder is concurrent and the
    /// command stream orders a dispatch only after the last writers of the
    /// buffers it binds (`DESIGN.md` §6.4), so a stamp that binds only `step`
    /// can run before the slots whenever they are written by compute kernels
    /// (`copy2d` under a pass-plan base). There the stamp is `ring_stamp`,
    /// which binds `certified` and writes a constant. Off Metal the queue is
    /// in order, and the value stages into the current [`Self::marker_src`]
    /// tile through its mapped pointer and one `slice_set` copies it to
    /// `step`; cycling `depth + 1` tiles keeps the host stage off a source a
    /// pending copy still reads (§10.3e).
    ///
    /// Called either per step (the marker-coincides-with-EOS arm) or once per
    /// commit window (the batched arm), so `generation` is whatever the host
    /// holds as `fed` at that point.
    pub fn publish_marker(&self, generation: u32, certified: &[&Tensor]) -> Result<()> {
        if generation == 0 {
            lloom_tensor::bail!(
                "TokenRing: generation 0 is the unpublished sentinel, ordinals start at 1"
            );
        }
        let device = self.step.device();
        if device.is_metal() {
            device.ring_stamp(&self.step, generation, certified)?;
            #[cfg(feature = "metal")]
            self.signal_publication(generation)?;
            return Ok(());
        }
        // Cycle to a fresh source tile so the stage does not race an in-flight
        // `slice_set` reading the previous one.
        let n = self.marker_src.total_tiles();
        let src_tile = (self.marker_slot.get() + 1) % n;
        self.marker_slot.set(src_tile);
        // Stage the value zero-copy (mapped write, no alloc); the fallback inside
        // `stage_at_tile` covers an unmapped source. Then one queue-ordered copy to
        // the persistent `step` word.
        self.marker_src.stage_at_tile(src_tile, &[generation])?;
        let src = self.marker_src.tile_view(src_tile, 1)?.reshape(1)?;
        self.step.slice_set(&src, 0, 0)?;
        Ok(())
    }

    /// Write the sampled token into the input slot ONLY — no device EOS decision
    /// and no per-step stamp.
    ///
    /// This is the per-step path for the arms that do NOT stamp per token: the
    /// batched arm learns EOS host-side or in a GPU window check, and the
    /// completion marker is written once per commit window
    /// ([`Self::publish_marker`]) rather than once per token. So the per-step path
    /// is just the input slot — the next forward's input, device-written and
    /// queue-ordered after the prior forward's read — and nothing the host reads.
    ///
    /// Skips both the `broadcast_eq → max_all → slice_set(eos)` (~5 dispatches,
    /// `DESIGN.md` §11.5g) AND the per-step stamp (`from_vec` + `slice_set`) the
    /// incumbent wrote every token. The `eos` flag tensor is left at whatever it
    /// last held; nothing reads it on this arm.
    pub fn publish_input(&self, token: &Tensor) -> Result<()> {
        if token.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "TokenRing::publish_input expects the U32 argmax output, got {:?}",
                token.dtype()
            );
        }
        let token = token.reshape(1)?;
        let tile = self.next_slot();
        self.inputs.write_device_at_tile(tile, &token)?;
        Ok(())
    }

    /// The EOS id set this ring compares against, for the host-side EOS check on
    /// the [`crate::generation::EosRead::HostOnDrain`] arm. Read back once at
    /// ring construction and cached by the caller, so this is `to_vec1`'d off
    /// the hot path (it is a loop invariant — the same handful of ids every
    /// step).
    pub fn eos_ids(&self) -> Result<Vec<u32>> {
        self.eos_ids.to_vec1::<u32>()
    }

    /// The EOS id set as the device tensor, for the `GpuBatched` arm's `eos_index`
    /// dispatch (which compares the window against it on the GPU). A loop-invariant
    /// device constant, uploaded once at construction.
    pub fn eos_ids_tensor(&self) -> &Tensor {
        &self.eos_ids
    }

    /// Non-blocking: has step `generation` published, and was it EOS? (#535)
    ///
    /// Returns `None` when the answer is not yet available *or* not available
    /// without blocking -- the caller polls, and falls back to
    /// [`Self::eos_flag`] if it must have an answer now.
    ///
    /// | returns | meaning |
    /// |---|---|
    /// | `None` | this step has not published yet, **or** no non-blocking read is available here |
    /// | `Some(false)` | published, not EOS |
    /// | `Some(true)` | published, EOS |
    ///
    /// # Why the counter is read first and the flag second
    ///
    /// The publish order is input, flag, counter; the poll order is the reverse
    /// of the dependency, counter then flag. Observing the counter at this
    /// step's ordinal is what licenses reading the flag at all -- without it,
    /// `eos == 0` conflates *not EOS* with *not yet written* and the poll is a
    /// race (see the [`Self::step`] field comment).
    ///
    /// # What the caller is accepting, stated because it is not free
    ///
    /// **A measured bound, not a documented guarantee.** MSL promises no
    /// ordering between two plain stores, so *counter visible ⇒ flag visible* is
    /// supported by #535's 0-of-200 tear arm and by nothing in the spec.
    /// `DESIGN.md` §Epistemics: absence of an observed tear is evidence of
    /// absence only to the extent of the trials. The failure this would produce
    /// is a mis-read EOS on the step it occurred -- a generation that stops one
    /// token late or early -- not corruption, because the token itself travels
    /// through `input` device-side and is never read here.
    ///
    /// A caller unwilling to take that bound has the blocking read unchanged:
    /// `eos_flag().to_vec1::<u8>()`.
    /// # Allocation, and why this is not `peek_vec1` any more (#578)
    ///
    /// This is polled ~283 000 times per token, so it does **zero** heap
    /// allocations and takes **no lock** on the fast path: both words are read
    /// through addresses resolved once at construction (see [`Self::mapped`]).
    /// The previous spelling reached `peek_vec1` twice, each acquiring the
    /// storage lock and materialising two `Vec`s to deliver one scalar.
    ///
    /// The locking peek remains the fallback for an unmapped allocation, so the
    /// behaviour is unchanged and only the cost differs.
    pub fn poll_eos(&self, generation: u32) -> Option<bool> {
        let seen = self.peek_step()?;
        // `>=` rather than `==`: a poller that missed an intermediate value
        // must not spin forever. The counter is monotone, so a larger value
        // means this step published and a later one did too -- which is a
        // caller error (it should have polled each step) but must not hang.
        if seen < generation {
            return None;
        }
        Some(self.peek_eos_flag()? != 0)
    }

    /// The generation counter, through the resolved address when there is one.
    #[inline]
    fn peek_step(&self) -> Option<u32> {
        #[cfg(feature = "metal")]
        if let Some(m) = &self.mapped {
            POLL_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: `m.step` was resolved from `self.step`, which this struct
            // owns and which `publish` writes in place without reallocating
            // (§10.3e), so the address is live for this borrow. The GPU may be
            // writing it concurrently -- that is the point, and the contract on
            // `MetalStorage::peek_scalar` states the value may be stale or
            // torn. `read_volatile` is what stops the load being hoisted out of
            // a caller's poll loop.
            return Some(unsafe { (m.step as *const u32).read_volatile() });
        }
        self.step.peek_scalar::<u32>(0)
    }

    /// The step ordinal the GPU has published so far, read non-blocking through
    /// the mapped counter — or `None` if the counter is not CPU-mapped.
    ///
    /// This is the visibility signal a streaming consumer of a *second* buffer
    /// needs: the history slot and the EOS flag are written, then this counter
    /// **last**, by a stamp that certifies them ([`Self::publish_marker`]), so
    /// an observed value `>= N` means every write of steps `1..=N` — including
    /// the history slot — has landed (§9.3). A consumer draining a device-side history therefore reads
    /// only up to this ordinal, never a slot the GPU has not yet written. Only
    /// meaningful on the stamped (`publish_step`) arm; the blocking arm's own
    /// per-step readback already synced, so it does not need this.
    pub fn published_generation(&self) -> Option<u32> {
        self.peek_step()
    }

    /// The EOS flag, through the resolved address when there is one.
    #[inline]
    fn peek_eos_flag(&self) -> Option<u8> {
        #[cfg(feature = "metal")]
        if let Some(m) = &self.mapped {
            // SAFETY: as `peek_step` above; `m.eos` was resolved from
            // `self.eos`, owned by this struct and written in place.
            return Some(unsafe { (m.eos as *const u8).read_volatile() });
        }
        self.eos.peek_scalar::<u8>(0)
    }

    /// The address of the generation counter, for a poll loop that wants to
    /// wait **on** it rather than merely between reads of it (#578).
    ///
    /// `PollMechanism::Wfe` arms the exclusive monitor on the watched word, so
    /// it needs the address; every other arm ignores it. `None` when the
    /// allocation is unmapped, in which case the caller uses the address-free
    /// `wait` and loses only the monitor.
    ///
    /// # Safety
    ///
    /// The pointer is valid while this `TokenRing` is alive. The caller must
    /// not read through it after dropping the ring, and must treat the value as
    /// possibly stale or concurrently written -- it is the same racy-by-design
    /// word [`Self::poll_eos`] reads.
    pub unsafe fn step_addr(&self) -> Option<*const u32> {
        #[cfg(feature = "metal")]
        if let Some(m) = &self.mapped {
            return Some(m.step as *const u32);
        }
        None
    }

    /// Whether the poll is reading through resolved addresses (#578).
    ///
    /// The fast path takes no lock and allocates nothing, so it moves **no**
    /// counter -- there is no call to count. A test asserting the poll is
    /// allocation-free needs to know which path ran, and a run's `RESULT` line
    /// needs it for the same reason the peek counters are on it: an
    /// optimisation whose engagement nobody counts is one whose null reads as a
    /// result (§2.4).
    pub fn poll_is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    /// The generation counter, for a caller that wants to observe it directly.
    pub fn step_counter(&self) -> &Tensor {
        &self.step
    }

    /// Advance to the next input tile and return it as a `[1]` U32 view — the
    /// slot the fused publish (`Device::argmax_publish`, lloom #735 step 5)
    /// writes the sampled token into on the GPU, in place of
    /// [`Self::publish_input`]'s blit. `input_for(1)` then reads the same tile.
    pub(crate) fn next_input_slot_view(&self) -> Result<Tensor> {
        let tile = self.next_slot();
        self.inputs.tile_view(tile, 1)?.reshape(1)
    }

    /// The tensor the model reads as its next input. Stable across steps, and
    /// the same allocation every step -- §10.3e names buffer-identity stability
    /// as one of the three properties a replacement append must not regress.
    ///
    /// The input ring's device — for a caller that needs it (e.g. the strategy's
    /// `device()`), replacing the old whole-buffer `input()` accessor.
    pub fn input_device(&self) -> &Tensor {
        self.inputs.segment0()
    }

    /// The input tile's token capacity — the largest chunk `stage_prompt` accepts.
    /// A caller staging a wider unit (the one-shot whole prompt) takes a
    /// whole-tensor path instead.
    pub fn input_width(&self) -> usize {
        self.max_width
    }

    /// Advance the cursor to the next input TILE and return its index. Every write
    /// (stage or publish) calls this FIRST so it targets a tile no in-flight forward
    /// is reading; `input_for` then views this same tile for the forward. With one
    /// tile it is the same tile every time (the incumbent).
    fn next_slot(&self) -> usize {
        let n = self.inputs.total_tiles();
        self.slot.set((self.slot.get() + 1) % n);
        self.slot.get()
    }

    /// The current input tile viewed as the `width` tokens THIS forward consumes —
    /// a contiguous `[1, width]` prefix of the tile within one mapped segment, **no
    /// copy**. This is the seam that decouples staging capacity from consumption
    /// width: `stage_prompt` (or `publish`) writes up to `max_width` tokens into the
    /// tile, and the forward reads exactly the `width` it steps over — 1 for decode,
    /// `K+1` for a drafted block, `ubatch` for a streamed prefill chunk.
    pub fn input_for(&self, width: usize) -> Result<Tensor> {
        if width == 0 || width > self.max_width {
            lloom_tensor::bail!(
                "TokenRing::input_for: width {width} outside [1, {}]",
                self.max_width
            );
        }
        // The CURRENT tile — the one the last stage/publish wrote (`next_slot`
        // advanced to it), which is what the forward now reads.
        self.inputs.tile_view(self.slot.get(), width)
    }

    /// Stage `tokens` into the next input tile for the forward to consume — the
    /// host-sourced feed (a streamed-prefill chunk, a host-provided block).
    /// **Zero-copy** through the segment's mapped pointer, at ANY chunk width up to
    /// a segment (the `SegmentedRing` keeps every segment mapped); an unmapped arm
    /// falls back to one `slice_set` upload inside `stage_at_tile`.
    ///
    /// Advances to a fresh tile FIRST so this stage does not overwrite a tile an
    /// in-flight forward is still reading; the forward then reads `input_for` on
    /// this same (now-current) tile.
    pub fn stage_prompt(&self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.max_width {
            lloom_tensor::bail!(
                "TokenRing::stage_prompt: {} tokens outside [1, {}]",
                tokens.len(),
                self.max_width
            );
        }
        let tile = self.next_slot();
        self.inputs.stage_at_tile(tile, tokens)
    }

    /// The current input TILE index (the one the last stage/publish wrote and the
    /// forward reads). For a fuzz oracle that tracks per-tile contents.
    pub fn current_input_slot(&self) -> usize {
        self.slot.get()
    }

    /// The number of input tiles (`depth + 1`) — the ring's cycle length. For a
    /// fuzz oracle mirroring the cursor.
    pub fn input_slot_count(&self) -> usize {
        self.inputs.total_tiles()
    }

    /// Read the CURRENT input tile's first `width` slots back through the mapped
    /// pointer — the same zero-copy access the GPU forward makes. For fuzzing the
    /// input ring on the real path (not a `to_cpu` copy). `None` if unmapped.
    /// Without Metal the segment is host memory and the read is a plain copy.
    pub fn peek_current_input(&mut self, width: usize) -> Option<Vec<u32>> {
        let tile = self.slot.get();
        self.inputs.peek_tile(tile, width)
    }

    /// Poison every input tile with a sentinel — a fuzz then asserts a read of a
    /// never-staged slot returns it, catching a wrong-tile or over-width read.
    pub fn poison_input(&self, sentinel: u32) -> Result<()> {
        self.inputs.poison_all(sentinel)
    }

    /// The device-resident EOS flag. Nonzero iff the last published token was
    /// an EOS id.
    ///
    /// Reading this host-side is a readback and reintroduces exactly the round
    /// trip this module exists to remove -- so it is what the *poll* reads, once
    /// the GPU has published, and not something to call inside the step.
    pub fn eos_flag(&self) -> &Tensor {
        &self.eos
    }
}

#[cfg(all(test, feature = "metal"))]
mod terminal_trace_tests {
    use super::*;
    use crate::nn::terminal_trace;

    #[test]
    fn disabled_trace_never_polls_the_mapped_generation() {
        const CHILD: &str = "LLOOM_TRACE_POLL_GATE_CHILD";
        let Ok(case) = std::env::var(CHILD) else {
            // Each arm owns the actual cached tuning/profile state and poll
            // counter; neither mutates the test process's environment.
            for enabled in [false, true] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "generation::token_ring::terminal_trace_tests::disabled_trace_never_polls_the_mapped_generation",
                        "--nocapture",
                    ])
                    .env(CHILD, if enabled { "on" } else { "off" })
                    .output()
                    .unwrap();
                eprint!("{}", String::from_utf8_lossy(&output.stdout));
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
                assert!(
                    output.status.success(),
                    "isolated trace arm {enabled} failed: {}",
                    output.status
                );
                let compiled = cfg!(any(debug_assertions, feature = "profile"));
                let expected = u64::from(compiled && enabled);
                let marker = format!(
                    "TRACE-POLL compiled={compiled} enabled={enabled} direct=1 diagnostic={} submitted_snapshots={expected}",
                    2 * expected
                );
                assert!(
                    String::from_utf8_lossy(&output.stderr).contains(&marker),
                    "isolated child did not execute the expected actual-poll arm: {marker}"
                );
            }
            return;
        };
        let enabled = match case.as_str() {
            "on" => true,
            "off" => false,
            _ => panic!("unknown trace fixture arm {case}"),
        };
        let mut tuning = lloom_tensor::tuning::Tuning::default();
        tuning.instrument.profile = enabled;
        assert!(lloom_tensor::tuning::install(tuning).is_ok());
        assert_eq!(lloom_tensor::kernels::metal::profile::enabled(), enabled);

        // Host-only storage: no Metal Device, queue, command buffer or native
        // work. These immutable words stay alive longer than the ring below.
        // The fixture installs their addresses solely to exercise the ACTUAL
        // mapped peek_step/POLL_READS path, not a fake polling closure.
        let step = 7u32;
        let eos = 0u8;
        let mut ring = TokenRing::new(1, &[99], &Device::Cpu).unwrap();
        ring.mapped = Some(Mapped {
            step: std::ptr::from_ref(&step) as usize,
            eos: std::ptr::from_ref(&eos) as usize,
        });
        assert!(ring.poll_is_mapped());
        let before = poll_read_count();
        assert_eq!(ring.published_generation(), Some(7));
        assert_eq!(
            poll_read_count() - before,
            1,
            "the actual poll counter is engaged"
        );
        let before = poll_read_count();
        let submitted_snapshots = std::cell::Cell::new(0);
        let run = terminal_trace::run_begin(
            || ring.published_generation().map(|p| p as usize),
            Default::default(),
            Default::default(),
        );
        terminal_trace::producer_stop(&run, terminal_trace::StopBy::Length, || {
            submitted_snapshots.set(submitted_snapshots.get() + 1);
            (3, ring.published_generation().map(|p| p as usize))
        });
        let compiled = cfg!(any(debug_assertions, feature = "profile"));
        let expected = u64::from(compiled && enabled);
        assert_eq!(poll_read_count() - before, 2 * expected);
        assert_eq!(submitted_snapshots.get(), expected);
        eprintln!(
            "TRACE-POLL compiled={compiled} enabled={enabled} direct=1 diagnostic={} submitted_snapshots={}",
            poll_read_count() - before, submitted_snapshots.get()
        );
    }
}
