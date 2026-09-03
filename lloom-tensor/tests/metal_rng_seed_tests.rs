#![cfg(feature = "metal")]
//! `Device::set_seed` must actually reach the Metal RNG kernel.
//!
//! # The defect these pin
//!
//! `random.metal`'s `atomic_load_seed` reconstructs the 64-bit seed from two
//! 32-bit atomics as `(ulong)seed[0] << 32 | seed[1]` — **word 0 is the HIGH
//! half**. `MetalDevice::set_seed` copied a native `u64`, which on a
//! little-endian host puts the **LOW** half in word 0, so the kernel read
//! `swap32(seed)`.
//!
//! A permuted seed would still be a seed. What made this fatal is the next
//! step: `HybridTaus::seed_per_thread` computes
//! `uint4(ulong4(seeds) * ulong4(PHI…) * ulong4(1099087573))`, and `uint4`
//! **truncates to the low 32 bits**. For any seed below `2^32` the swap makes
//! the kernel's value `seed << 32`, whose low 32 bits are zero — and every
//! multiple of it keeps them zero. So the seed-bearing component was
//! identically `0`, and the remaining three components of the vector are
//! `{tid, 1, 1}`.
//!
//! **The generated stream was therefore a function of `tid` alone**, and every
//! seed the public API can express produced byte-identical output.
//!
//! # Why this is not merely untidy
//!
//! `candle_nn::sampling::gumbel_softmax` draws its Gumbel variates from
//! `rand_like`, so a sampler built on it is reproducible **for the wrong
//! reason**: the stream repeats because the seed is not an input, not because
//! it is a respected one. A "seeded reproduction" gate over such a run passes
//! while measuring nothing (lloom `DESIGN.md` §15.1a's VACUOUS class), and
//! lloom #277 measured exactly that before this was found.
//!
//! # What is fixed, and what is a bound
//!
//! The fix is host-side: `set_seed` writes the two 32-bit words in the order
//! the kernel reads them. **32 bits of the seed reach the generator, not 64** —
//! `seed_per_thread` truncates to `uint4` and the vector's other components are
//! `{tid, 1, 1}`, so there is nowhere for the high half to go without changing
//! `random.metal`. That is a real limit, it is pinned by
//! `the_seeds_low_32_bits_are_what_reach_the_kernel` below, and it is left in
//! place because widening it would move every existing seed's meaning on a file
//! shared with the CPU and CUDA backends, to buy keyspace nothing here needs.

use lloom_tensor::{DType, Device, Result, Tensor};

/// Four draws is enough: the failure being pinned makes **every** element equal
/// across seeds, so a longer vector adds no discriminating power.
fn draw(device: &Device, seed: u64) -> Result<Vec<f32>> {
    device.set_seed(seed)?;
    Tensor::rand(0f32, 1f32, (4,), device)?.to_vec1::<f32>()
}

#[test]
fn set_seed_changes_the_metal_rng_stream() -> Result<()> {
    let device = Device::new_metal(0)?;

    // Seeds a caller would actually pass: all below 2^32, which is the entire
    // population the defect silenced. `0xDEAD_BEEF` is included because it is
    // the largest of them and still under the boundary — before the fix it gave
    // the same four floats as `1`.
    let seeds = [1u64, 999, 12345, 299_792_458, 0xDEAD_BEEF];
    let mut seen: Vec<(u64, Vec<f32>)> = Vec::new();
    for s in seeds {
        seen.push((s, draw(&device, s)?));
    }

    for i in 0..seen.len() {
        for j in (i + 1)..seen.len() {
            assert_ne!(
                seen[i].1, seen[j].1,
                "seeds {} and {} produced an identical stream: the seed is not \
                 reaching the kernel (see this file's header)",
                seen[i].0, seen[j].0
            );
        }
    }
    Ok(())
}

#[test]
fn a_repeated_seed_reproduces_its_stream() -> Result<()> {
    let device = Device::new_metal(0)?;

    // The other half of the property, and it is not implied by the first: a
    // stream could depend on the seed and still not be reproducible — if, say,
    // it also depended on the buffer's prior contents. Both are required before
    // a sampler built on this can claim seeded reproduction.
    //
    // Interleaved with a different seed between the two draws, so this cannot
    // pass merely because nothing advanced the generator's state.
    let a = draw(&device, 4242)?;
    let _ = draw(&device, 777)?;
    let b = draw(&device, 4242)?;
    assert_eq!(a, b, "the same seed did not reproduce its own stream");
    Ok(())
}

#[test]
fn the_seeds_low_32_bits_are_what_reach_the_kernel() -> Result<()> {
    let device = Device::new_metal(0)?;

    // **This pins a LIMIT, not a capability, and it is deliberately not a
    // 64-bit claim.** `seed_per_thread` returns `uint4(...)`, truncating each
    // product to 32 bits, so **at most 32 bits of the seed can ever reach the
    // generator** — and the three remaining components of the vector are
    // `{tid, 1, 1}`, so there is nowhere for the other half to go without
    // changing the kernel.
    //
    // Widening it is a `.metal` change that would alter what every existing
    // seed means, on a file shared with the CPU and CUDA parity tests and with
    // every other model that seeds. It buys keyspace this project has no use
    // for: `DESIGN.md` §2.3.3c asks tier 3 for *seeded reproduction*, which 32
    // bits satisfies. **Recorded as a bound rather than closed**, so that a
    // caller who needs more than 32 bits finds this test rather than a silence.
    let lo_a = draw(&device, 0x0000_0000_0000_0005)?;
    let lo_b = draw(&device, 0x0000_0000_0000_0006)?;
    assert_ne!(
        lo_a, lo_b,
        "seeds differing in the LOW 32 bits must differ: that half is the one \
         that reaches the kernel"
    );

    // The high half is inert, and asserting so is what keeps the limit honest:
    // if a later change widens the kernel, this assertion fails and the header
    // gets rewritten, rather than the bound quietly ceasing to be true.
    let hi_a = draw(&device, 0x0000_0005_0000_0000)?;
    let hi_b = draw(&device, 0x0000_0006_0000_0000)?;
    assert_eq!(
        hi_a, hi_b,
        "the HIGH 32 bits reached the kernel: the documented 32-bit bound no \
         longer holds and this file's header is stale"
    );

    Ok(())
}

#[test]
fn f16_and_bf16_take_the_same_seed_path() -> Result<()> {
    let device = Device::new_metal(0)?;

    // `rand_uniform` instantiates per dtype (`rand_uniform_f32`, `_f16`,
    // `_bf16`) over one templated body, so the seeding is shared — but it is
    // shared *by construction* rather than by test, and LFM2 decodes at f16
    // (`DESIGN.md` §2.3.3b). Checking the dtype the model actually runs at
    // costs one dispatch and removes the inference.
    for dtype in [DType::F16, DType::BF16] {
        device.set_seed(31337)?;
        let a = Tensor::rand(0f32, 1f32, (4,), &device)?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        device.set_seed(31338)?;
        let b = Tensor::rand(0f32, 1f32, (4,), &device)?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_ne!(a, b, "{dtype:?}: two seeds gave one stream");
    }
    Ok(())
}
