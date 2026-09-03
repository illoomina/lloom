//! **The deliverable: the dispatch census shows the fused kernel present and
//! the unfused one absent** (lloom #509 wiring).
//!
//! `lfm2_gemv_epilogue_wiring.rs` proves the *answer changes*, which is
//! evidence the fusion ran. This file proves it **by name**, from the
//! instrument that counts what the GPU was actually asked to do —
//! `record_dispatch`/`by_label` behind `CANDLE_METAL_PROFILE=1`, the census
//! #517 found already existing.
//!
//! **Why both.** A numeric delta says *something* changed; it cannot say the
//! change was the kernel you meant. The census names the pipeline. And the
//! census alone is not sufficient either: a dispatch recorded is not a
//! dispatch that computed the right thing, which is what the wiring file's
//! magnitude bound is for. Together they say the named kernel ran *and* the
//! arithmetic moved by a rounding.
//!
//! **The assertion that matters is the ABSENCE.** #500, #536 and #550 all
//! record a mechanism that was built, correct, and dispatched by nothing --
//! and in every case the fused kernel *could* have been present while the
//! unfused one was *also* still there, which is a fusion that added work
//! rather than replacing it. So each arm below asserts both halves: the fused
//! name appears, and the dispatch it replaces does not.

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};
use lloom::nn::VarBuilder;
use lloom::models::lfm2::{Cache, Config, KvAppend, LayerType, Model};

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

/// The per-label dispatch census **of one decode step**.
///
/// The prompt is prefilled first and the counters reset afterwards, so what is
/// counted is a decode token and not a prefill — which matters because prefill
/// is a GEMM and would put `gemv` names in the census that no decode step
/// dispatched.
fn decode_census(cfg: &Config, dev: &Device) -> Result<Vec<(String, u64)>> {
    let model = build(cfg, dev)?;
    let mut cache = Cache::new(true, DType::F16, cfg, dev)?;

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    dev.synchronize()?;

    // Reset AFTER the synchronize, per `profile::reset`'s own doc: an in-flight
    // command buffer's completion handler fires later and would otherwise land
    // in this window.
    lloom_tensor::metal_backend::profile::reset();

    let step = Tensor::new(&[7u32], dev)?.reshape((1, 1))?;
    let logits = model.forward(&step, prompt.len(), &mut cache)?;
    // Force the work to be encoded and completed before snapshotting.
    logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    dev.synchronize()?;

    Ok(lloom_tensor::metal_backend::profile::snapshot().by_label)
}

fn count_matching(census: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    census
        .iter()
        .filter(|(l, _)| pred(l))
        .map(|(_, n)| *n)
        .sum()
}

fn dump(tag: &str, census: &[(String, u64)]) {
    eprintln!("--- census: {tag}");
    for (label, n) in census.iter().filter(|(l, _)| {
        l.starts_with("gemv")
            || l.starts_with("silu")
            || l.starts_with("badd")
            || l.starts_with("add")
    }) {
        eprintln!("    {n:>4}  {label}");
    }
}

/// **`gemv_silu`: the fused name appears and `silu_f16` disappears.**
///
/// Both halves are asserted. A fusion that added the fused kernel *beside* the
/// dispatch it was meant to replace would be strictly worse than not fusing,
/// and a delta-only test cannot see it.
#[test]
fn the_census_shows_gemv_silu_replacing_the_silu_dispatch() -> Result<()> {
    if !lloom_tensor::metal_backend::profile::enabled() {
        eprintln!(
            "SKIPPED: this test reads the dispatch census, which is only \
             recorded under CANDLE_METAL_PROFILE=1. Re-run with it set."
        );
        return Ok(());
    }
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };

    let off = decode_census(&tiny_config(false, false), &dev)?;
    dump("gemv_silu OFF", &off);
    let on = decode_census(&tiny_config(true, false), &dev)?;
    dump("gemv_silu ON", &on);

    // Non-vacuity: the census must contain SOMETHING, or every absence
    // assertion below passes trivially (§3.7a's all-zeros failure).
    assert!(
        !off.is_empty() && off.iter().map(|(_, n)| *n).sum::<u64>() > 0,
        "the census recorded no dispatches at all -- the instrument is blind, \
         and an absence read from a blind instrument is not evidence (§2.4)"
    );

    let silu_off = count_matching(&off, |l| l.starts_with("silu"));
    let fused_off = count_matching(&off, |l| l.contains("_silu") && l.starts_with("gemv"));
    let silu_on = count_matching(&on, |l| l.starts_with("silu"));
    let fused_on = count_matching(&on, |l| l.contains("_silu") && l.starts_with("gemv"));

    eprintln!("silu_f16: {silu_off} -> {silu_on}   gemv_*_silu: {fused_off} -> {fused_on}");

    assert!(
        silu_off > 0,
        "the unfused arm must dispatch a standalone silu -- if it does not, \
         this test is comparing against the wrong baseline"
    );
    assert_eq!(
        fused_off, 0,
        "the fused gemv_silu kernel must NOT be dispatched with the flag off; \
         the default arm is the path that ships (§7.1a)"
    );
    assert!(
        fused_on > 0,
        "PRESENT: with the flag on, the fused gemv_*_silu kernel must appear \
         in the census. This is the assertion #500, #536 and #550 could not \
         make -- the mechanism has production callers now, or it does not"
    );
    assert!(
        silu_on < silu_off,
        "ABSENT: the standalone silu dispatches must FALL ({silu_off} -> \
         {silu_on}). A fused kernel added beside the dispatch it replaces is \
         strictly worse than no fusion"
    );
    Ok(())
}

/// **The residual epilogue: the fused name appears and the `add` disappears.**
///
/// Since #516 the `_residual` names are instantiated unconditionally, so the
/// fused arm is reachable on any Metal build and a `LoadFunction` error here is
/// a REAL failure rather than an unset compile flag. The skip that used to
/// swallow it is gone; the only skip left is the census instrument itself,
/// which is recorded only under `CANDLE_METAL_PROFILE=1`.
#[test]
fn the_census_shows_gemv_residual_replacing_the_add_dispatch() -> Result<()> {
    if !lloom_tensor::metal_backend::profile::enabled() {
        eprintln!(
            "SKIPPED: this test reads the dispatch census, which is only \
             recorded under CANDLE_METAL_PROFILE=1. Re-run with it set."
        );
        return Ok(());
    }
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };

    let off = decode_census(&tiny_config(false, false), &dev)?;
    dump("gemv_residual OFF", &off);

    // Propagates rather than skipping: the kernel is always instantiated now,
    // so failing to load it means something is genuinely wrong.
    let on = decode_census(&tiny_config(false, true), &dev)?;
    dump("gemv_residual ON", &on);

    assert!(
        !off.is_empty() && off.iter().map(|(_, n)| *n).sum::<u64>() > 0,
        "the census recorded no dispatches at all -- the instrument is blind"
    );

    // The residual add is a binary `add` over f16. Counted by prefix so a
    // rename of the kernel family does not silently zero this assertion --
    // which would make the ABSENCE half pass for the wrong reason.
    let add_off = count_matching(&off, |l| l.starts_with("badd") || l.starts_with("add"));
    let fused_off = count_matching(&off, |l| l.contains("_residual"));
    let add_on = count_matching(&on, |l| l.starts_with("badd") || l.starts_with("add"));
    let fused_on = count_matching(&on, |l| l.contains("_residual"));

    eprintln!("add/badd: {add_off} -> {add_on}   gemv_*_residual: {fused_off} -> {fused_on}");

    assert!(
        add_off > 0,
        "the unfused arm must dispatch standalone adds for the residual -- if \
         it does not, this test compares against the wrong baseline"
    );
    assert_eq!(
        fused_off, 0,
        "the fused residual kernel must NOT be dispatched with the flag off"
    );
    assert!(
        fused_on > 0,
        "PRESENT: with the flag on, the fused gemv_*_residual kernel must \
         appear in the census"
    );
    assert!(
        add_on < add_off,
        "ABSENT: the standalone add dispatches must FALL ({add_off} -> \
         {add_on}); a fusion that does not remove the dispatch it replaces has \
         added work"
    );
    Ok(())
}
