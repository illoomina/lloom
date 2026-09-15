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

use lloom::models::lfm2::{
    Cache, Composition, Config, ConvState, FusionPolicy, GateFusion, KvAppend, LayerType, Model,
};
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The two-layer fixture `lfm2_gemv_epilogue_wiring.rs` uses: one conv layer,
/// one attention layer, so every epilogue site kind is exercised, and sizes
/// the fused GEMV tiles are instantiated for.
fn tiny_config(fusion: FusionPolicy, gemv_silu: bool, gemv_residual: bool) -> Config {
    // `Manual` with the flags on means ALL the hand flags the derivation
    // reproduces: the residual and, for the gate, the two-weight kernel —
    // under which the SiLU flag is moot (`w1` is never dispatched alone).
    let gemv_gate = if gemv_silu && gemv_residual {
        GateFusion::Swiglu
    } else {
        GateFusion::Off
    };
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
        gemv_gate,
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
    let bmul_off = count(&off, |l| l.starts_with("bmul") || l.starts_with("mul"));
    let silu_on = count(&derived, |l| l.starts_with("silu"));
    let badd_on = count(&derived, |l| l.starts_with("badd") || l.starts_with("add"));
    let bmul_on = count(&derived, |l| l.starts_with("bmul") || l.starts_with("mul"));
    let fused_silu = count(&derived, |l| l.starts_with("gemv") && l.ends_with("_silu"));
    let fused_res = count(&derived, |l| {
        l.starts_with("gemv") && l.ends_with("_residual")
    });
    let fused_mul = count(&derived, |l| l.starts_with("gemv") && l.ends_with("_mul"));
    let fused_swiglu = count(&derived, |l| {
        l.starts_with("gemv") && l.ends_with("_swiglu")
    });
    eprintln!(
        "silu {silu_off} -> {silu_on}; badd {badd_off} -> {badd_on}; bmul {bmul_off} -> {bmul_on}; \
         gemv_*_silu {fused_silu}; gemv_*_residual {fused_res}; gemv_*_mul {fused_mul}; \
         gemv_*_swiglu {fused_swiglu}"
    );
    // The fixture's decode step: 2 MLPs (2 silu, 2 w2 residuals, 2 gate
    // products) + 1 conv out_proj + 1 attention out_proj residual; the conv
    // layer's three `bmul`s (`b*x`, `window*w`, `c*conv`) stay — the
    // absolute references.
    assert_eq!(
        (silu_off, badd_off, bmul_off),
        (2, 4, 5),
        "the unfused arm's counts"
    );
    assert_eq!(
        (silu_on, badd_on, bmul_on),
        (0, 0, 3),
        "ABSENT: the replaced dispatches"
    );
    // The two-weight kernel takes the gate: no `gemv_*_silu` and no
    // `gemv_*_mul` beside it — one dispatch per MLP for the four.
    assert_eq!(
        (fused_silu, fused_res, fused_mul, fused_swiglu),
        (0, 4, 0, 2),
        "PRESENT: the fused kernels"
    );
    assert_eq!(
        count(&off, |l| {
            l.ends_with("_silu")
                || l.ends_with("_residual")
                || l.ends_with("_mul")
                || l.ends_with("_swiglu")
        }),
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

/// P2: the exclusion set is consulted for the arms the config selects, not
/// only for the fragments the derivation composes. `silu_impl=relu` is the
/// transform §12.2l measured degenerate (2307 f16 ulp, `ARAARAARA…`); under
/// `Derived` a model selecting it is refused at construction, by name and
/// with its numbers. Under `Manual` nothing changes — the arm stays selectable
/// for the ladder it exists to measure.
#[test]
fn derived_refuses_an_excluded_silu_arm_at_construction() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let mut cfg = tiny_config(FusionPolicy::Derived, false, false);
    cfg.silu_impl = lloom_tensor::kernels::SiluImpl::Relu;
    let err = match build(&cfg, &dev) {
        Ok(_) => panic!("Fusion=Derived must refuse the excluded `silu_impl=relu` arm"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("silu_impl=relu") && err.contains("excluded"),
        "{err}"
    );
    eprintln!("{err}");
    // Manual is untouched: the arm builds.
    let mut manual = tiny_config(FusionPolicy::Manual, false, false);
    manual.silu_impl = lloom_tensor::kernels::SiluImpl::Relu;
    build(&manual, &dev)?;
    Ok(())
}

/// P2, the model key and the regions (`optimizer-design.md` §5a, §10c): what
/// the report says for the crown's conv arms. `conv_fused` on the rotating
/// ring is the `shortconv_ring` kernel — ONE transform of the recurrence, not
/// `conv_fused` and `conv_state=rotating` composed — and the set validates it
/// jointly, under `tree=serial-rot`; the sliding ring under `conv_fused` runs
/// `conv1d_depthwise`, which the set has no joint entry for, and the report
/// says so instead of reusing the two separate verdicts. The tiny fixture's
/// regions: one conv ring, one KV, no fused site in either.
#[test]
fn the_report_names_the_model_the_regions_and_the_joint_validation() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    // The crown's conv arms: the fused rotating ring.
    let mut crown = tiny_config(FusionPolicy::Derived, false, false);
    crown.conv_fused = true;
    crown.conv_state = ConvState::RotatingRing { k: 0 };
    let model = build(&crown, &dev)?;
    let r = model.fusion_report().expect("Derived keeps its report");
    eprintln!("{r}");
    assert!(!r.gemv_silu && r.gemv_residual && r.gemv_gate == GateFusion::Swiglu);
    assert_eq!(
        r.compositions,
        vec![Composition {
            region: "conv-ring".to_string(),
            transforms: vec!["conv_fused".to_string(), "conv_state=rotating".to_string()],
            joint: Some("conv_fused+conv_state=rotating".to_string()),
        }]
    );
    assert_eq!(
        r.validated,
        vec![(
            "conv_fused+conv_state=rotating".to_string(),
            "shortconv_ring".to_string()
        )]
    );
    assert!(r.unvalidated.is_empty(), "{:?}", r.unvalidated);
    assert!(r.fused_in_stateful.is_empty(), "{:?}", r.fused_in_stateful);
    // On the fixture's 512-wide weights the barrier term admits the gate (a
    // 0.5 MB sweep costs less than a dispatch), so all three ride; the cost
    // line is on the plan.
    assert!(
        r.plan.contains("gemv_swiglu=2")
            && r.plan
                .contains("regions[conv-ring=1 kv=1 nodes=7 fused-in-stateful=0]")
            && r.plan.contains(" cost[before: "),
        "{}",
        r.plan
    );

    // The sliding ring under `conv_fused` runs `conv1d_depthwise`: no joint
    // entry, so the composition is reported as NOT validated jointly, and the
    // pair is unvalidated under that kernel rather than admitted from the
    // two separate entries.
    let mut sliding = tiny_config(FusionPolicy::Derived, false, false);
    sliding.conv_fused = true;
    sliding.conv_state = ConvState::SlidingRing { k: 2, slack: 4 };
    let model = build(&sliding, &dev)?;
    let r = model.fusion_report().unwrap();
    eprintln!("{r}");
    assert_eq!(
        r.compositions,
        vec![Composition {
            region: "conv-ring".to_string(),
            transforms: vec!["conv_fused".to_string(), "conv_state=ring".to_string()],
            joint: None,
        }]
    );
    assert_eq!(
        r.unvalidated,
        vec![(
            "conv_fused+conv_state=ring".to_string(),
            "no entry".to_string()
        )]
    );
    assert!(r.validated.is_empty());

    // `conv_fused` alone on the shuffle is `shortconv_step`, validated under
    // its serial tree; `conv_state=rotating` alone is the unfused fold
    // (`fast_sum`, store-width, simd tree), validated under that.
    let mut step = tiny_config(FusionPolicy::Derived, false, false);
    step.conv_fused = true;
    let r = build(&step, &dev)?.fusion_report().unwrap().clone();
    assert_eq!(
        r.validated,
        vec![("conv_fused".to_string(), "shortconv_step".to_string())]
    );
    assert!(r.compositions.is_empty());
    let mut rot = tiny_config(FusionPolicy::Derived, false, false);
    rot.conv_state = ConvState::RotatingRing { k: 0 };
    let r = build(&rot, &dev)?.fusion_report().unwrap().clone();
    assert_eq!(
        r.validated,
        vec![("conv_state=rotating".to_string(), "fast_sum".to_string())]
    );
    Ok(())
}

/// The closed-form gain §5a names, read from the loaded weights: the fold's
/// ∞-norm per channel, `Σ_j |w[c, j]|`. The fixture fills
/// `conv.conv.weight` (`[512, 1, 3]`) with `((i % 17) − 8) / 32` over the
/// flattened index, so the gain is computable by hand — asserted as the
/// absolute reference (`CONTRIBUTING.md` §3.1b rule 1), max, min and the
/// count of expansive channels alike — and the horizon is `taps − 1 = 2`.
#[test]
fn the_conv_fold_gain_is_the_weights_norm_and_the_horizon_is_the_window() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let cfg = tiny_config(FusionPolicy::Derived, false, false);
    let model = build(&cfg, &dev)?;
    let g = model
        .fusion_report()
        .unwrap()
        .conv_fold_gain
        .clone()
        .expect("one conv layer");
    // By hand, at f16 (the fixture's dtype): every value ((i % 17) - 8) / 32
    // is exact in f16, so the sum of three is exact too.
    let h = cfg.hidden_size;
    let k = cfg.conv_l_cache;
    let mut max = f32::MIN;
    let mut min = f32::MAX;
    let mut over_one = 0;
    for c in 0..h {
        let l: f32 = (0..k)
            .map(|j| ((((c * k + j) % 17) as f32) - 8.0) / 32.0)
            .map(f32::abs)
            .sum();
        max = max.max(l);
        min = min.min(l);
        if l > 1.0 {
            over_one += 1;
        }
    }
    assert_eq!((g.layers, g.taps, g.channels), (1, 3, h));
    assert_eq!((g.max, g.min, g.over_one), (max, min, over_one));
    // The absolute values for this fill: the largest window is where the
    // 17-cycle wraps, {7, 8, −8}/32 → 23/32; the smallest {−1, 0, 1}/32 →
    // 2/32; no channel is expansive.
    assert_eq!(g.max, 23.0 / 32.0);
    assert_eq!(g.min, 2.0 / 32.0);
    assert_eq!(g.over_one, 0);
    assert!(model
        .fusion_report()
        .unwrap()
        .to_string()
        .contains("window write L=1 (permutation), horizon 2 tokens"));
    Ok(())
}
