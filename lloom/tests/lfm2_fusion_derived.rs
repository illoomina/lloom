//! `Fusion=Derived` on Metal (issue #732, P1): a model whose epilogues the
//! optimizer derived computes the bits the flag-selected model computes, and
//! the dispatch census shows the derived plan actually taken.
//!
//! `optimizer_derivation.rs` proves the derivation lands on the hand-written
//! composition, symbolically. This file proves the plan REACHES the forward
//! pass (`CONTRIBUTING.md` §7.1 F4: built ≠ reachable): the derived arm's
//! logits are bit-identical to `gemv_silu = gemv_residual = true`, differ
//! from the unfused default by a rounding, and the census under `Derived`
//! names the fused kernels with the dispatches they replace at zero.
//!
//! The default (`Manual`, flags off) is asserted untouched: bit-identical to
//! itself across the arms, which is the half that proves the shipping path
//! was not disturbed (§7.1a).

#![cfg(feature = "metal")]

use lloom::models::lfm2::{Cache, Config, FusionPolicy, KvAppend, LayerType, Model};
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The two-layer fixture `lfm2_gemv_epilogue_wiring.rs` uses: one conv layer,
/// one attention layer, so every epilogue site kind is exercised, and sizes
/// the fused GEMV tiles are instantiated for.
fn tiny_config(fusion: FusionPolicy, gemv_silu: bool, gemv_residual: bool) -> Config {
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
        kv_append: KvAppend::InPlace,
        fusion,
        gemv_silu,
        gemv_residual,
        ..Default::default()
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

/// A prefill and one decode step; the decode step's logits, and the per-label
/// dispatch census of that step (recorded only under `CANDLE_METAL_PROFILE=1`).
fn decode(cfg: &Config, dev: &Device) -> Result<(Vec<f32>, Vec<(String, u64)>)> {
    let model = build(cfg, dev)?;
    let mut cache = Cache::new(true, DType::F16, cfg, dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    dev.synchronize()?;
    lloom_tensor::metal_backend::profile::reset();
    let step = Tensor::new(&[7u32], dev)?.reshape((1, 1))?;
    let logits = model.forward(&step, prompt.len(), &mut cache)?;
    let logits = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    dev.synchronize()?;
    Ok((
        logits,
        lloom_tensor::metal_backend::profile::snapshot().by_label,
    ))
}

fn count(census: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    census
        .iter()
        .filter(|(l, _)| pred(l))
        .map(|(_, n)| *n)
        .sum()
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// **The derived arm computes the flag-selected arm's bits.** And it is not
/// the unfused arm's: the two epilogues move the logits by a rounding, so a
/// derivation that silently fell back to `Manual` would read as a defect here.
#[test]
fn derived_matches_the_hand_selected_fusions_bit_for_bit() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let (derived, _) = decode(&tiny_config(FusionPolicy::Derived, false, false), &dev)?;
    let (manual_on, _) = decode(&tiny_config(FusionPolicy::Manual, true, true), &dev)?;
    let (manual_off, _) = decode(&tiny_config(FusionPolicy::Manual, false, false), &dev)?;
    assert_eq!(derived.len(), manual_on.len());
    assert_eq!(
        derived, manual_on,
        "Fusion=Derived must compute exactly what gemv_silu=gemv_residual=true computes"
    );
    let delta = max_abs_delta(&derived, &manual_off);
    assert!(
        delta > 0.0,
        "the derived arm is bit-identical to the UNFUSED default: the plan was not applied"
    );
    // A rounding, not a defect: the same bound the wiring test uses for the
    // flag-selected arm (§2.3.5a's magnitude row).
    let scale = manual_off.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        delta < scale * 1e-2,
        "max |Δ| {delta} against a logit scale of {scale}"
    );
    eprintln!("derived == manual(on); max |Δ| vs unfused {delta:.3e} (scale {scale:.3e})");
    Ok(())
}

/// **The census under `Derived` names the fused kernels and shows the
/// dispatches they replace at zero** — the plan reached the GPU, and it did
/// not add the fused kernel beside the pair it was to replace.
#[test]
fn the_census_shows_the_derived_plan_taken() -> Result<()> {
    if !lloom_tensor::metal_backend::profile::enabled() {
        eprintln!("SKIPPED: the dispatch census is recorded only under CANDLE_METAL_PROFILE=1.");
        return Ok(());
    }
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let (_, off) = decode(&tiny_config(FusionPolicy::Manual, false, false), &dev)?;
    let (_, derived) = decode(&tiny_config(FusionPolicy::Derived, false, false), &dev)?;
    assert!(
        off.iter().map(|(_, n)| *n).sum::<u64>() > 0,
        "the census recorded nothing"
    );

    let silu_off = count(&off, |l| l.starts_with("silu"));
    let badd_off = count(&off, |l| l.starts_with("badd") || l.starts_with("add"));
    let silu_on = count(&derived, |l| l.starts_with("silu"));
    let badd_on = count(&derived, |l| l.starts_with("badd") || l.starts_with("add"));
    let fused_silu = count(&derived, |l| l.starts_with("gemv") && l.ends_with("_silu"));
    let fused_res = count(&derived, |l| {
        l.starts_with("gemv") && l.ends_with("_residual")
    });
    eprintln!(
        "silu {silu_off} -> {silu_on}; badd {badd_off} -> {badd_on}; \
         gemv_*_silu {fused_silu}; gemv_*_residual {fused_res}"
    );
    // The fixture's decode step: 2 MLPs (2 silu, 2 w2 residuals) + 1 conv
    // out_proj + 1 attention out_proj residual — the absolute references.
    assert_eq!((silu_off, badd_off), (2, 4), "the unfused arm's counts");
    assert_eq!(
        (silu_on, badd_on),
        (0, 0),
        "ABSENT: the replaced dispatches"
    );
    assert_eq!(
        (fused_silu, fused_res),
        (2, 4),
        "PRESENT: the fused kernels"
    );
    assert_eq!(
        count(&off, |l| l.ends_with("_silu") || l.ends_with("_residual")),
        0
    );
    Ok(())
}

/// The default is untouched: `Manual` with the flags off computes what it
/// computed, run to run.
#[test]
fn manual_off_is_the_shipping_path_and_deterministic() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let cfg = tiny_config(FusionPolicy::Manual, false, false);
    assert_eq!(Config::default().fusion, FusionPolicy::Manual);
    let (a, _) = decode(&cfg, &dev)?;
    let (b, _) = decode(&cfg, &dev)?;
    assert_eq!(a, b);
    Ok(())
}
