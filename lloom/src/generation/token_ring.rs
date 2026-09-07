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
    input: Tensor,
    /// The input buffer's capacity in tokens (`input` is `[1, max_width]`). A
    /// stage wider than this is refused rather than silently truncated.
    max_width: usize,
    eos: Tensor,
    eos_ids: Tensor,
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
        Self::with_input_width(first_input, eos_ids, device, 1)
    }

    /// As [`Self::new`], with the input buffer sized to hold up to `max_width`
    /// tokens rather than one.
    ///
    /// The input is the buffer the model reads as its next forward's tokens. At
    /// width 1 it is the single sampled token a decode step feeds; a wider buffer
    /// **decouples what is STAGED from what one forward CONSUMES** — the state
    /// machine can step over a prompt in `ubatch`-wide units (streamed prefill) or
    /// a drafted block in `K+1`-wide units, writing them into this one reused
    /// buffer (zero-copy [`Tensor::write_host_slice`] for a host source) and
    /// consuming a [`Self::input_for`] narrow of the width that step eats. Sized
    /// with `from_vec` (not `zeros`) so there is no pending blit-fill to race the
    /// host write, and allocated ONCE — the buffer identity §10.3e requires is
    /// preserved (every step reads the same allocation, just a different-width
    /// prefix of it).
    pub fn with_input_width(
        first_input: u32,
        eos_ids: &[u32],
        device: &Device,
        max_width: usize,
    ) -> Result<Self> {
        if eos_ids.is_empty() {
            lloom_tensor::bail!("TokenRing: eos_ids is empty, so the loop would never terminate");
        }
        let w = max_width.max(1);
        // A bare allocation — **no fill, no blit, no host upload**. The input's
        // content is irrelevant at construction: every forward's input is WRITTEN
        // (a stage or a `publish`) before it is read (`sample_and_feed` writes the
        // slot, then hands `input()` to the next forward — write-then-return), so
        // `first_input` is never actually consumed. `Tensor::zeros` would enqueue
        // a blit-fill (an alloc + a blit whose result is discarded, the cost
        // removed from hot paths), and `from_vec` would upload a host vec; neither
        // is needed. `empty` gives the buffer identity §10.3e wants with no cost.
        //
        // SAFETY: uninitialised memory, made defined before any read by the
        // write-then-return discipline above; `first_input` is written into slot 0
        // below to keep the "seed" contract explicit and to define the slot even
        // on an arm that might read before writing.
        let input = unsafe { Tensor::empty((1, w), DType::U32, device)? };
        // Define slot 0 with the seed, zero-copy through the mapped pointer (the
        // buffer is small → mapped; the fallback path is a no-op seed on an
        // unmapped arm, which the write-then-return discipline makes safe anyway).
        let _ = first_input;
        let seed = [first_input];
        // SAFETY: no GPU work is in flight on this just-allocated buffer.
        unsafe {
            input.write_host_slice::<u32>(0, &seed);
        }
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
            input,
            max_width: w,
            eos,
            eos_ids,
            step,
            mapped,
        })
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
    /// input slot, then the EOS flag, then the counter. A host that observes the
    /// counter has, in source order, a written flag behind it. §9.3 records that
    /// MSL gives no ordering guarantee for plain stores, so this rests on a
    /// measured bound rather than on a documented one -- #535 observed 0 of 200
    /// tears with a device barrier -- and [`Self::poll_eos`] states what a
    /// caller is accepting.
    pub fn publish_step(&self, token: &Tensor, generation: Option<u32>) -> Result<()> {
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
        // `reshape` to the slot's own shape makes it contiguous and
        // rank-compatible in one step.
        let token = token.reshape((1, 1))?;
        self.input.slice_set(&token, 1, 0)?; // column 0: one token into the [1, max_width] input
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
        // The stamp, LAST. Everything a poller will read once it sees this
        // value has been encoded before it.
        if let Some(g) = generation {
            let stamp = Tensor::from_vec(vec![g], 1, self.step.device())?;
            self.step.slice_set(&stamp, 0, 0)?;
        }
        Ok(())
    }

    /// [`Self::publish_step`] **without the device EOS decision** — writes the
    /// input slot and the generation stamp only, skipping the
    /// `broadcast_eq → max_all → slice_set(eos)` that costs ~5 dispatches every
    /// token in the command stream (`DESIGN.md` §11.5g's comparison).
    ///
    /// For the [`crate::generation::EosRead::HostOnDrain`] arm, where EOS is not
    /// learned per token from the device flag but host-side from the drained ids
    /// (a plain Rust compare on the batch the drain already reads), so the flag
    /// is never read and the dispatches that compute it are pure cost. The stamp
    /// is still written last and is still the visibility signal the zero-copy
    /// drain rides ([`Self::published_generation`]) — that protocol is unchanged,
    /// only the EOS flag is gone.
    ///
    /// The `eos` flag tensor is left at whatever it last held; nothing reads it
    /// on this arm. A `generation` of 0 is rejected for [`Self::publish_step`]'s
    /// reason.
    pub fn publish_input_and_stamp(&self, token: &Tensor, generation: u32) -> Result<()> {
        if token.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "TokenRing::publish_input_and_stamp expects the U32 argmax output, got {:?}",
                token.dtype()
            );
        }
        if generation == 0 {
            lloom_tensor::bail!(
                "TokenRing: generation 0 is the unpublished sentinel, ordinals start at 1"
            );
        }
        let token = token.reshape((1, 1))?;
        self.input.slice_set(&token, 1, 0)?; // column 0: one token into the [1, max_width] input
        // The stamp, LAST — the drain's visibility signal, same ordering as
        // `publish_step`. No EOS flag is written: this arm never reads it.
        let stamp = Tensor::from_vec(vec![generation], 1, self.step.device())?;
        self.step.slice_set(&stamp, 0, 0)?;
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
    /// needs: `publish_step` writes the history/input slot, then the EOS flag,
    /// then this counter **last**, in queue order, so an observed value `>= N`
    /// means every write of steps `1..=N` — including the history slot — is
    /// visible (§9.3). A consumer draining a device-side history therefore reads
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

    /// The tensor the model reads as its next input. Stable across steps, and
    /// the same allocation every step -- §10.3e names buffer-identity stability
    /// as one of the three properties a replacement append must not regress.
    ///
    /// This is the FULL `[1, max_width]` buffer; a step that consumes fewer than
    /// `max_width` tokens takes an [`Self::input_for`] narrow. The width-1 decode
    /// step could take `input_for(1)`, but returns the whole buffer here for the
    /// `[1, 1]`-max case where they coincide.
    pub fn input(&self) -> &Tensor {
        &self.input
    }

    /// The input buffer's token capacity — the largest chunk `stage_prompt`
    /// accepts. A caller staging a wider unit (the one-shot whole prompt) takes a
    /// whole-tensor path instead.
    pub fn input_width(&self) -> usize {
        self.max_width
    }

    /// The input narrowed to the `width` tokens THIS forward consumes — a
    /// contiguous prefix view of the `[1, max_width]` buffer, **no copy** (start 0,
    /// so the narrow is a plain offset+length view of already-contiguous memory).
    /// This is the seam that decouples staging capacity from consumption width:
    /// `stage_prompt` (or `publish`) writes up to `max_width` tokens, and the
    /// forward reads exactly the `width` it steps over — 1 for decode, `K+1` for a
    /// drafted block, `ubatch` for a streamed prefill chunk.
    pub fn input_for(&self, width: usize) -> Result<Tensor> {
        if width == 0 || width > self.max_width {
            lloom_tensor::bail!(
                "TokenRing::input_for: width {width} outside [1, {}]",
                self.max_width
            );
        }
        self.input.narrow(1, 0, width)
    }

    /// Stage `tokens` into the input buffer at offset 0 for the next forward to
    /// consume — the host-sourced feed (a streamed-prefill chunk, a host-provided
    /// block). **Zero-copy** through the mapped pointer
    /// ([`Tensor::write_host_slice`]) on a shared-memory arm; falls back to a
    /// `slice_set` upload only when the buffer is not CPU-mapped (a discrete-GPU /
    /// large-buffer arm), so correctness holds everywhere and the copy is paid
    /// only where the architecture forces it.
    ///
    /// The tokens overwrite slots `[0, tokens.len())`; the forward then reads
    /// `input_for(tokens.len())`. `tokens.len()` must not exceed `max_width`.
    ///
    /// SAFETY of the fast path: the caller guarantees no GPU work is reading the
    /// input buffer at stage time — the pacer/commit ordering holds this, exactly
    /// as it does for the sampled-token `publish` (a stage is the same "write the
    /// slot before the forward that reads it" shape).
    pub fn stage_prompt(&self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.max_width {
            lloom_tensor::bail!(
                "TokenRing::stage_prompt: {} tokens outside [1, {}]",
                tokens.len(),
                self.max_width
            );
        }
        // SAFETY: the input is contiguous rank-2 U32; no GPU work reads it while a
        // stage runs (the caller's pacer ordering).
        let wrote = unsafe { self.input.write_host_slice::<u32>(0, tokens) };
        if wrote {
            return Ok(());
        }
        // Unmapped fallback (discrete-GPU / large buffer): upload + slice_set. One
        // alloc, taken only on the arm where the mapped write is unavailable.
        let src = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), self.input.device())?;
        self.input.slice_set(&src, 1, 0)?;
        Ok(())
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
