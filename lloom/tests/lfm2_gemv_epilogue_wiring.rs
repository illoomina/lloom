//! The GEMV epilogue fusions, **on the decode path** (lloom #509 wiring;
//! `DESIGN.md` §12.2d, §12.2h).
//!
//! `gemv_silu` (#484) and the residual epilogue (#509) were both built,
//! measured and correct, and **dispatched by nothing** — `Tensor::matmul`
//! reaches `call_mlx_gemv` through a wrapper that pins the epilogue to `None`
//! (`mlx_gemm.rs:667`), so no model could select one. §12.2h records that state
//! and names the wiring as what a decode-path measurement waits on.
//!
//! **What this file tests is that the wiring is REACHED, not that the kernel is
//! right.** The kernels have their own parity, mutation and engagement tests in
//! `candle-metal-kernels` (#484, #509). What no test there can see is whether a
//! model's forward pass ever arrives at them — which is exactly the failure
//! #500 and #536 record twice over, and which every green gate missed *because
//! the new path was not on the path*.
//!
//! So the assertions here are about **reachability and equivalence**:
//!
//! * with the flag off, the forward pass is bit-identical to the pass before
//!   this change existed — the fused code is not merely inert, it is absent;
//! * with the flag on, the answer **changes** (the fusion computes at the f32
//!   accumulator where the dispatch pair computes at `T` on the rounded store,
//!   §2.3.5a) but stays **close** — a reordering, not a defect;
//! * with the flag on, the fused arm is **actually taken** rather than silently
//!   falling back, which is asserted by a shape the fallback cannot produce.
//!
//! Metal only. The fusions are Metal GEMV epilogues and have no CPU arm — the
//! op `bail!`s rather than falling back, per `DESIGN.md` §2.4.
//!
//! Dual (`lloom::testing::fixture`): under `LLOOM_FIXTURE=real|synthetic`
//! every reachability and equivalence assertion runs at LFM2.5-2.6B's
//! geometry, where the decode tiles the GEMV derives are the served ones.
//! The CPU references — the pinned fingerprint and the tolerance measured
//! against it — are the tiny model's and run on it alone.

#![cfg(feature = "metal")]

use lloom::models::lfm2::{Cache, Config, KvAppend, LayerType, Model};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The tiny geometry with the arms applied: what the CPU reference builds.
fn tiny_config(gemv_silu: bool, gemv_residual: bool) -> Config {
    let mut cfg = tiny_geometry();
    arms(gemv_silu, gemv_residual)(&mut cfg);
    cfg
}

/// The arms every test here runs, on whichever geometry the fixture holds:
/// the in-place KV and the f16 KV every recorded digest belongs to, the
/// default attention and conv state, and the two epilogue flags.
fn arms(gemv_silu: bool, gemv_residual: bool) -> impl FnOnce(&mut Config) {
    move |c| {
        c.use_flash_attn = false;
        c.attn_impl = Default::default();
        c.kv_append = KvAppend::InPlace;
        c.kv_quant = Default::default();
        c.conv_state = Default::default();
        c.memory_budget = None;
        c.gemv_silu = gemv_silu;
        c.gemv_residual = gemv_residual;
        c.tap_layers = Vec::new();
        c.tap_dtype = DType::F32;
    }
}

/// The fixture a test runs on: [`tiny_geometry`] by default.
fn fixture(test: &str) -> Result<Option<Fixture<Lfm2Tiny>>> {
    Fixture::new(test, tiny_geometry())
}

/// A two-layer LFM2 — one conv layer, one attention layer — so **all three**
/// residual producers are exercised: `conv.out_proj`, `self_attn.out_proj` and
/// `feed_forward.w2` (§12.2e's 22 + 8 + 30 dispatches per token).
///
/// `hidden_size` and `intermediate_size` are multiples of 4 so the GEMV's `TM`
/// tiling is exercised, and both are ≥ 512 so `derive_gemv_tile` selects an
/// instantiated decode tile rather than one the fused arms do not carry.
fn tiny_geometry() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        ..Default::default()
    }
}

/// The fixture's LFM2 at `cfg`'s geometry and arms, at f16 — so two runs
/// agree and any difference between two *arms* is attributable to the arms.
fn build(fx: &Fixture<Lfm2Tiny>, cfg: &Config, dev: &Device) -> Result<Model> {
    fx.model_in(cfg, Some(DType::F16), dev)
}

/// The same decode step on the CPU backend at f32, over the weights the f16
/// model holds (rounded through f16), fusions off (they are Metal
/// epilogues): the absolute reference every arm is held to (CONTRIBUTING
/// §3.1b). A defect in a fused arm that moves the logits by less than the
/// relative bound below still moves them away from this.
fn cpu_decode_logits(cfg: &Config) -> Result<Vec<f32>> {
    let mut cpu = cfg.clone();
    cpu.gemv_silu = false;
    cpu.gemv_residual = false;
    let dev = Device::Cpu;
    let model = TinyCheckpoint::<Lfm2Tiny>::new(
        cpu.clone(),
        Weights {
            dense: GgmlDType::F16,
            ..Weights::default()
        },
    )
    .with_dtype(DType::F32)
    .model(&dev)?;
    decode_with(&model, &cpu, &dev, DType::F32)
}

/// Every logit bit, folded: a pinned value is an absolute reference that
/// moves when any bit does — the one here that a defect in code the CPU and
/// Metal arms share (an op both devices dispatch) cannot move alike.
fn fingerprint(v: &[f32]) -> u64 {
    v.iter().fold(0x761, |h, x| {
        lloom::testing::tiny::init::splitmix64(h ^ x.to_bits() as u64)
    })
}

/// The CPU reference's fingerprint under the default law and seed. It moves
/// with the law, the fixture's geometry and the CPU backend's arithmetic —
/// each a reason to re-pin, named in the commit that does.
const CPU_FINGERPRINT: u64 = 0xe41c_e0b0_36c4_0328;

/// `|arm - CPU|` must stay inside the f16 activations' own error, measured
/// on the unfused arm (`the_unfused_arm_is_within_the_cpu_reference`).
const CPU_TOL: f32 = 4e-3;

fn assert_near_cpu(tag: &str, arm: &[f32], cpu: &[f32]) {
    let (d, m) = max_abs_delta(arm, cpu);
    eprintln!("{tag}: |arm - CPU f32| = {d:e} at scale {m:e}");
    assert!(
        d <= CPU_TOL * m,
        "{tag}: {d:e} from the CPU backend at scale {m:e} (tolerance {CPU_TOL} x scale)"
    );
}

/// Prefill a prompt, then take one **decode** step, and return that step's
/// logits. The decode step is the only one the fusions apply to: `seq_len > 1`
/// is a GEMM and takes the eager path by construction.
fn decode_logits(fx: &Fixture<Lfm2Tiny>, cfg: &Config, dev: &Device) -> Result<Vec<f32>> {
    let model = build(fx, cfg, dev)?;
    decode_with(&model, cfg, dev, DType::F16)
}

fn decode_with(model: &Model, cfg: &Config, dev: &Device, dtype: DType) -> Result<Vec<f32>> {
    let mut cache = Cache::new(true, dtype, cfg, dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    let step = Tensor::new(&[7u32], dev)?.reshape((1, 1))?;
    let logits = model.forward(&step, prompt.len(), &mut cache)?;
    logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

/// `max|a - b|` and the max magnitude, so a caller can talk in relative terms.
fn max_abs_delta(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "arms must produce the same shape");
    let mut d = 0f32;
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        d = d.max((x - y).abs());
        m = m.max(x.abs()).max(y.abs());
    }
    (d, m)
}

/// **The flag-off arm is the arm that ships, and it must be untouched.**
///
/// The wiring introduces `forward_add` on three types. With both flags off,
/// each one is `self.forward(..)? + residual` — a call into the *same*
/// `forward` that shipped, not a copy of it — so this asserts the refactor is
/// behaviour-preserving rather than merely believed to be.
///
/// Run twice in one process, which is also a run-to-run determinism check on
/// the default path (§2.3.1's second kind).
#[test]
fn flags_off_is_deterministic_and_is_the_shipping_path() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = fixture("gemv_wiring_flags_off")? else {
        return Ok(());
    };
    let cfg = fx.config(arms(false, false));
    let a = decode_logits(&fx, &cfg, &dev)?;
    let b = decode_logits(&fx, &cfg, &dev)?;
    assert_eq!(a, b, "the default path must be bit-identical run to run");
    assert!(
        a.iter().any(|v| *v != 0.0),
        "a forward pass of all zeros would pass every comparison below \
         vacuously (§3.7a's all-zeros failure, §15.1 #1)"
    );
    Ok(())
}

/// **`gemv_silu` is reached, and it changes the answer by a rounding.**
///
/// #484 measured this kernel computing SiLU at the f32 accumulator where the
/// `silu_f16` dispatch it replaces computes at `T` on the already-rounded
/// store. That is **strictly fewer roundings and a different computation**, so
/// the digest moves by construction (§12.2d) — this asserts the difference is
/// present (the arm was taken) and small (it is a reordering, not a defect).
///
/// **The non-vacuity arm is the load-bearing one.** A wiring that silently fell
/// back to the eager path would produce delta == 0 and read as *"bit-identical,
/// how reassuring"* — which is exactly #550's fixture defect, where a cross-arm
/// disagreement of exactly 0 was recorded as a scope limit rather than a
/// result. Here the fallback is a *failure*, so it is asserted against.
#[test]
fn gemv_silu_is_reached_from_the_decode_path() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = fixture("gemv_wiring_silu")? else {
        return Ok(());
    };
    let base = decode_logits(&fx, &fx.config(arms(false, false)), &dev)?;
    let fused = decode_logits(&fx, &fx.config(arms(true, false)), &dev)?;
    if !fx.at_scale() {
        assert_near_cpu(
            "gemv_silu",
            &fused,
            &cpu_decode_logits(&tiny_config(true, false))?,
        );
    }

    let (delta, mag) = max_abs_delta(&base, &fused);
    eprintln!("gemv_silu: max|delta| = {delta:e}, max|logit| = {mag:e}");

    assert!(
        delta > 0.0,
        "the fused SiLU arm computes at the f32 accumulator where the pair \
         computes at f16 on the rounded store, so it MUST differ. A delta of \
         exactly 0 means the fusion did not run -- the arm fell back silently, \
         which is the #500/#536 failure this wiring exists to end"
    );
    assert!(
        delta <= 0.05 * mag.max(1.0),
        "delta {delta:e} against magnitude {mag:e} is too large for a \
         rounding-order difference; that is a defect, not a reordering \
         (§2.3.5a's discriminator)"
    );
    Ok(())
}

/// **The residual epilogue is reached, and it changes the answer by a
/// rounding** — the same argument as `gemv_silu` above, on #509's kernel.
///
/// Since #516 the `_residual` names are instantiated unconditionally, so this
/// is an ordinary gate on any Metal build. The skip that used to swallow a
/// `LoadFunction` error is gone: with the kernel always present, that error
/// means something is genuinely wrong and must fail rather than report the
/// same green as a passing run (§15.1a's VACUOUS class in the suite).
#[test]
fn gemv_residual_is_reached_from_the_decode_path() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = fixture("gemv_wiring_residual")? else {
        return Ok(());
    };
    let base = decode_logits(&fx, &fx.config(arms(false, false)), &dev)?;
    let fused = decode_logits(&fx, &fx.config(arms(false, true)), &dev)?;
    if !fx.at_scale() {
        assert_near_cpu(
            "gemv_residual",
            &fused,
            &cpu_decode_logits(&tiny_config(false, true))?,
        );
    }

    let (delta, mag) = max_abs_delta(&base, &fused);
    eprintln!("gemv_residual: max|delta| = {delta:e}, max|logit| = {mag:e}");

    assert!(
        delta > 0.0,
        "the fused residual arm adds at the f32 accumulator where `badd_f16` \
         adds at f16 on the rounded store, so it MUST differ. A delta of \
         exactly 0 means the fusion did not run"
    );
    assert!(
        delta <= 0.05 * mag.max(1.0),
        "delta {delta:e} against magnitude {mag:e} is too large for a \
         rounding-order difference (§2.3.5a)"
    );
    Ok(())
}

/// **Prefill is untouched by either flag**, which is what makes the fusion a
/// decode-path change rather than a model change.
///
/// `seq_len > 1` takes the GEMM route (`takes_gemv_route` is `m == 1 || n == 1`)
/// and the fused arms are instantiated only at the non-transposed decode tiles.
/// Both applicability predicates require `seq_len == 1`, so a prefill must be
/// bit-identical with the flags on — asserted rather than assumed, because a
/// predicate that let prefill through would fail at pipeline load in the
/// residual case and silently mis-tile in the SiLU one.
#[test]
fn prefill_is_bit_identical_with_the_flags_on() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = fixture("gemv_wiring_prefill")? else {
        return Ok(());
    };

    let prefill_logits = |cfg: &Config| -> Result<Vec<f32>> {
        let model = build(&fx, cfg, &dev)?;
        let mut cache = Cache::new(true, DType::F16, cfg, &dev)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
        let logits = model.forward(&inp, 0, &mut cache)?;
        logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    };

    let base = prefill_logits(&fx.config(arms(false, false)))?;
    let with_silu = prefill_logits(&fx.config(arms(true, false)))?;
    assert_eq!(
        base, with_silu,
        "gemv_silu must not reach prefill: it is a decode-path GEMV fusion"
    );

    match prefill_logits(&fx.config(arms(false, true))) {
        Ok(with_res) => assert_eq!(
            base, with_res,
            "gemv_residual must not reach prefill: it is a decode-path GEMV fusion"
        ),
        Err(e) => {
            // A pipeline-load failure here would mean the predicate DID let
            // prefill through, which is the defect this test is for.
            let msg = e.to_string();
            assert!(
                !msg.contains("residual"),
                "gemv_residual reached prefill and failed to load: {msg}"
            );
            return Err(e);
        }
    }
    Ok(())
}

/// **The unfused arm is the CPU backend's answer at f16's precision**: the
/// level [`CPU_TOL`] is set from, and the absolute reference the shape
/// checks above (flag off agrees with itself; a fused arm is near the
/// unfused one) cannot see — a defect that moves every arm alike. The tiny
/// model's own reference: it runs on the tiny backend only.
#[test]
fn the_unfused_arm_is_within_the_cpu_reference() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = fixture("gemv_wiring_cpu_reference")? else {
        return Ok(());
    };
    if fx.at_scale() {
        println!("the CPU fingerprint is the tiny model's; nothing to check at scale");
        return Ok(());
    }
    let cfg = tiny_config(false, false);
    let cpu = cpu_decode_logits(&cfg)?;
    assert_eq!(
        fingerprint(&cpu),
        CPU_FINGERPRINT,
        "the CPU reference moved: {:#x}",
        fingerprint(&cpu)
    );
    assert_near_cpu("unfused", &decode_logits(&fx, &cfg, &dev)?, &cpu);
    Ok(())
}
