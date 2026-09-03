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

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};
use lloom::nn::VarBuilder;
use lloom::models::lfm2::{Cache, Config, KvAppend, LayerType, Model};

/// A two-layer LFM2 — one conv layer, one attention layer — so **all three**
/// residual producers are exercised: `conv.out_proj`, `self_attn.out_proj` and
/// `feed_forward.w2` (§12.2e's 22 + 8 + 30 dispatches per token).
///
/// `hidden_size` and `intermediate_size` are multiples of 4 so the GEMV's `TM`
/// tiling is exercised, and both are ≥ 512 so `derive_gemv_tile` selects an
/// instantiated decode tile rather than one the fused arms do not carry.
fn tiny_config(gemv_silu: bool, gemv_residual: bool) -> Config {
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
        use_flash_attn: false,
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        // `Off`, the f16 arm every recorded digest belongs to. Added by #218
        // after this file was written, which is why it stopped compiling --
        // `flash_scratch_sizing` above carries the same note from #116.
        kv_quant: Default::default(),
        conv_state: Default::default(),
        memory_budget: None,
        gemv_silu,
        gemv_residual,
        // #363's tap: off, as every pre-#363 caller had it.
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
    }
}

/// Deterministic weights: a fixed ramp, so two runs agree and any difference
/// between two *arms* is attributable to the arms. Same shape as
/// `lfm2_spec_conv_rollback.rs`'s builder.
fn build(cfg: &Config, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F16, dev);
    Model::new(cfg, vb)
}

/// Prefill a prompt, then take one **decode** step, and return that step's
/// logits. The decode step is the only one the fusions apply to: `seq_len > 1`
/// is a GEMM and takes the eager path by construction.
fn decode_logits(cfg: &Config, dev: &Device) -> Result<Vec<f32>> {
    let model = build(cfg, dev)?;
    let mut cache = Cache::new(true, DType::F16, cfg, dev)?;
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
    let cfg = tiny_config(false, false);
    let a = decode_logits(&cfg, &dev)?;
    let b = decode_logits(&cfg, &dev)?;
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
    let base = decode_logits(&tiny_config(false, false), &dev)?;
    let fused = decode_logits(&tiny_config(true, false), &dev)?;

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
    let base = decode_logits(&tiny_config(false, false), &dev)?;
    let fused = decode_logits(&tiny_config(false, true), &dev)?;

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

    let prefill_logits = |cfg: &Config| -> Result<Vec<f32>> {
        let model = build(cfg, &dev)?;
        let mut cache = Cache::new(true, DType::F16, cfg, &dev)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
        let logits = model.forward(&inp, 0, &mut cache)?;
        logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    };

    let base = prefill_logits(&tiny_config(false, false))?;
    let with_silu = prefill_logits(&tiny_config(true, false))?;
    assert_eq!(
        base, with_silu,
        "gemv_silu must not reach prefill: it is a decode-path GEMV fusion"
    );

    match prefill_logits(&tiny_config(false, true)) {
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
