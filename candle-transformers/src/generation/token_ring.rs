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

use candle::{DType, Device, Result, Tensor};

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
    eos: Tensor,
    eos_ids: Tensor,
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
        if eos_ids.is_empty() {
            candle::bail!("TokenRing: eos_ids is empty, so the loop would never terminate");
        }
        let input = Tensor::from_vec(vec![first_input], (1, 1), device)?;
        let eos_ids = Tensor::from_vec(eos_ids.to_vec(), eos_ids.len(), device)?;
        // Materialised on the device so the first `publish` overwrites rather
        // than allocates. `from_vec` of a `0u8` is the same shape `publish`
        // writes, which is what makes the two comparable.
        let eos = Tensor::from_vec(vec![0u8], 1, device)?;
        Ok(Self {
            input,
            eos,
            eos_ids,
        })
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
        if token.dtype() != DType::U32 {
            candle::bail!(
                "TokenRing::publish expects the U32 argmax output, got {:?}",
                token.dtype()
            );
        }
        // `argmax` over the vocabulary dimension returns rank 0 (it squeezes);
        // `reshape` to the slot's own shape makes it contiguous and
        // rank-compatible in one step.
        let token = token.reshape((1, 1))?;
        self.input.slice_set(&token, 0, 0)?;
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
        Ok(())
    }

    /// The tensor the model reads as its next input. Stable across steps, and
    /// the same allocation every step -- §10.3e names buffer-identity stability
    /// as one of the three properties a replacement append must not regress.
    pub fn input(&self) -> &Tensor {
        &self.input
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
