//! The walk at a position against what the decode step at that position
//! dispatches (`optimizer::walk_at`, `design/docs/optimizer-design.md`
//! §10i): Qwen4-exp at tiny geometry, its full-attention layers carrying the
//! QSA indexer, decoded from an empty cache one position at a time — steady
//! steps, the steps whose key completes an indexer block, the steps past the
//! dense range, and (the 2-pass length set low, `LLOOM_SDPA_TWO_PASS_K`)
//! the steps whose attention takes the two-launch kernel — each step's
//! census from the device's own window (`CANDLE_METAL_PROFILE`), in the IR's
//! families
//! (`optimizer::kernel_family`), held to the derived walk at that position.
//! The stems are f16, the real checkpoint's dense-stem kind.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::model::fusion_policy;
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::ops::{KvPrecision, KvQuant};
use lloom::optimizer::{census_of_labels, derive, walk_at, Census, Policy};
use lloom::profile::CrownExact;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The 2-pass length this binary runs at, so that the steps from
/// `TWO_PASS - 1` on (and every step past the dense range) take the
/// two-launch kernel.
const TWO_PASS: usize = 12;

fn device() -> Option<Device> {
    // Both switches are read once, on first use, by this process.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    std::env::set_var("LLOOM_SDPA_TWO_PASS_K", TWO_PASS.to_string());
    // A mismatched step's dispatches in order, for its report.
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    Device::new_metal(0).ok()
}

fn config() -> Config {
    config_at(32)
}

/// The tiny geometry at attention head size `head_dim` (the quantized KV
/// store's kernels take 64, 128 and 256), the indexer's heads wide enough
/// for the partial rope's rotated dims and wider than them, as the
/// checkpoint's are.
fn config_at(head_dim: usize) -> Config {
    let mut cfg = Config::tiny();
    cfg.moe_intermediate_size = 32;
    cfg.head_dim = head_dim;
    let n_rot = (head_dim as f64 * cfg.partial_rotary_factor) as usize;
    cfg.indexer_head_dim = cfg.indexer_head_dim.max(2 * n_rot);
    cfg
}

fn model(cfg: &Config, dev: &Device) -> Qwen4Exp<CrownExact, Resident> {
    testing::set_f16_stems(true);
    testing::model_with_indexer(cfg, dev, |l| Ok(testing::resident_experts(cfg, l, dev)))
        .expect("tiny indexed build")
}

/// [`model`] with the routed experts stored as `kind` (an f16 routed
/// activation needs a kind with the `(f16, f32, f16)` arm).
fn model_with_experts(
    cfg: &Config,
    dev: &Device,
    kind: GgmlDType,
) -> Qwen4Exp<CrownExact, Resident> {
    use std::sync::Arc;
    testing::set_f16_stems(true);
    testing::model_with_indexer(cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(cfg, l, dev, kind);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })
    .expect("tiny indexed build")
}

/// The families on which `a` and `b` differ, with both counts.
fn differences(a: &Census, b: &Census) -> Vec<(String, u64, u64)> {
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|k| {
            let (x, y) = (
                a.get(k).copied().unwrap_or(0),
                b.get(k).copied().unwrap_or(0),
            );
            (x != y).then(|| (k.clone(), x, y))
        })
        .collect()
}

/// A position whose step dispatches other than its walk, and the families
/// that differ (family, walked, ran).
type Wrong = (usize, Vec<(String, u64, u64)>);

/// The HyperConnection brackets' grouped-norm arm (`QWEN4EXP_HC_NORM_GEMV`,
/// set per bracket): off; on; on with every bracket's epilogue switches off
/// (as a plan that refuses them sets them) — the arm keeps them.
#[derive(Clone, Copy, PartialEq)]
enum Hc {
    Off,
    NormGemv,
    NormGemvSwitchesOff,
}

/// The arms the walk describes on top of the default (an f32 KV store, the
/// routed experts at f32): the store at f16 and quantized to 8 and 4 bits,
/// the routed experts' activations at f16 with the SwiGLU and the combine's
/// read fused and composed, the brackets' grouped norm in their products.
fn arms() -> Vec<(&'static str, Option<KvPrecision>, Option<bool>, Hc)> {
    vec![
        ("default", None, None, Hc::Off),
        ("kv-f16", Some(KvPrecision::dense(DType::F16)), None, Hc::Off),
        (
            "kv-int8",
            Some(KvPrecision::quantized(KvQuant::Int8).unwrap()),
            None,
            Hc::Off,
        ),
        (
            "kv-int4",
            Some(KvPrecision::quantized(KvQuant::Int4).unwrap()),
            None,
            Hc::Off,
        ),
        ("act-f16-fused", None, Some(true), Hc::Off),
        ("act-f16-composed", None, Some(false), Hc::Off),
        ("hc-norm-gemv", None, None, Hc::NormGemv),
        ("hc-norm-gemv-switches-off", None, None, Hc::NormGemvSwitchesOff),
    ]
}

/// The composition an arm runs, as the policy the walk is derived under —
/// the model's own (`fusion_policy`: the bracket epilogues on the brackets'
/// products only); with the f16 routed step composed, its two kernel
/// fragments refused; with the brackets' epilogues off, theirs.
fn policy(act_f16_fuse: Option<bool>, hc: Hc) -> Policy {
    let mut excluded = Vec::new();
    if act_f16_fuse == Some(false) {
        excluded.extend(["silu_mul_f16".to_string(), "combine_cast".to_string()]);
    }
    if hc == Hc::NormGemvSwitchesOff {
        excluded.extend(["mixed_silu".to_string(), "mixed_sigmoid".to_string()]);
    }
    fusion_policy(&Policy {
        excluded,
        ..Policy::default()
    })
}

/// Decode `arm`'s model from an empty cache over every kind of position,
/// each step's census against the derived walk at it.
fn check(
    name: &str,
    kv: Option<KvPrecision>,
    act_f16_fuse: Option<bool>,
    hc: Hc,
    dev: &Device,
) -> Result<(Vec<Wrong>, [usize; 3])> {
    let quant = kv.is_some_and(|p| p.quant().is_on());
    let mut cfg = if quant { config_at(64) } else { config() };
    if hc != Hc::Off {
        // Four streams: the injection's product (`[n, n·d]`) reaches the
        // grouped-norm kernel's tile (`n_out >= 4`, `K >= 16 n_out`).
        cfg.hc_count = 4;
    }
    let mut m = if act_f16_fuse.is_some() {
        model_with_experts(&cfg, dev, GgmlDType::Q8_0)
    } else {
        model(&cfg, dev)
    };
    if let Some(p) = kv {
        m.set_kv_precision(p)?;
    }
    if let Some(fuse) = act_f16_fuse {
        m.set_routed_act(DType::F16);
        m.set_f16_fuse(fuse);
    }
    let policy = policy(act_f16_fuse, hc);
    if hc == Hc::NormGemvSwitchesOff {
        // A plan refusing the brackets' epilogues sets every switch off.
        m.derive_fusion_with(&policy)?;
        assert!(m.bracket_epilogues().iter().all(|e| *e == (false, false)));
    }
    if hc != Hc::Off {
        m.set_hc_norm_gemv(true);
    }
    let ratio = cfg.indexer_compress_ratio;
    let dense = cfg.indexer_budget + ratio - 1;
    // The cache the model's own configuration asks for: the store at the
    // precision just set.
    let mut cache = m.new_cache(&m.config().clone(), dev)?;
    let mut wrong = Vec::new();
    let (mut blocks, mut past, mut two_pass) = (0, 0, 0);
    for pos in 0..dense + 2 * ratio {
        let tok = Tensor::new(&[[(pos % 97) as u32 + 3]], dev)?;
        dev.synchronize()?;
        let census = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("the census window");
        let _ = lloom_tensor::kernels::metal::trace::take_dispatches();
        lloom_tensor::kernels::metal::trace::set_recording(true);
        m.decode(&tok, &mut cache, &mut ())?;
        dev.synchronize()?;
        lloom_tensor::kernels::metal::trace::set_recording(false);
        let order: Vec<String> = lloom_tensor::kernels::metal::trace::take_dispatches()
            .iter()
            .map(|d| d.pipeline.clone())
            .collect();
        let ran = census_of_labels(census.close().iter().map(|(l, n)| (l.as_str(), *n)));
        let walked = derive(&walk_at(&m, pos), &policy)?.graph.census();
        let d = differences(&walked, &ran);
        if hc != Hc::Off && pos == 0 {
            assert!(
                walked.contains_key("hc_collapse_scaled") && !walked.contains_key("hc_norm"),
                "{name}: the arm serves the tiny brackets: {walked:?}"
            );
        }
        eprintln!(
            "{name} pos {pos}: ran {} walked {} {}",
            ran.values().sum::<u64>(),
            walked.values().sum::<u64>(),
            if d.is_empty() {
                "=".to_string()
            } else {
                format!("{d:?}")
            }
        );
        if !d.is_empty() && wrong.is_empty() {
            eprintln!("{name} pos {pos} order: {order:?}");
        }
        blocks += usize::from((pos + 1) % ratio == 0);
        two_pass += usize::from(walked.contains_key("sdpa_vector_2pass_1"));
        past += usize::from(pos + 1 > dense);
        if !d.is_empty() {
            wrong.push((pos, d));
        }
    }
    Ok((wrong, [blocks, past, two_pass]))
}

#[test]
fn every_arm_dispatches_what_its_walk_names_at_every_position() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let mut wrong = Vec::new();
    for (name, kv, act, hc) in arms() {
        let (w, [blocks, past, two_pass]) = check(name, kv, act, hc, &dev)?;
        assert!(
            blocks >= 2 && past >= 2,
            "{name}: the positions cover the block and the past-dense steps"
        );
        if kv.is_none_or(|p| !p.quant().is_on()) {
            assert!(
                two_pass >= 2,
                "{name}: the positions reach the 2-pass length"
            );
        }
        if !w.is_empty() {
            wrong.push((name, w));
        }
    }
    assert!(
        wrong.is_empty(),
        "arms whose steps dispatch other than their walk (position, (family, walked, ran)): {wrong:?}"
    );
    Ok(())
}

/// The MTP head's one-row pass (`MtpHead::forward` at `a = 1`, a draft
/// step's) against the head's walk at its lane's position, from an empty
/// lane cache up to the indexer's dense range — the block-completing and
/// the 2-pass positions among them.
#[test]
fn every_head_pass_dispatches_what_its_walk_names() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let cfg = config();
    testing::set_f16_stems(true);
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let mut cache = lloom::models::qwen4exp::State::<CrownExact>::for_layers(&cfg, 1);
    let (n, d) = (cfg.hc_count, cfg.hidden_size);
    let dense = cfg.indexer_budget + cfg.indexer_compress_ratio - 1;
    let mut wrong = Vec::new();
    for pos in 0..dense {
        let hidden = Tensor::from_vec(
            (0..n * d)
                .map(|i| ((i + pos) as f32 * 0.37).sin())
                .collect::<Vec<f32>>(),
            (1, 1, n * d),
            &dev,
        )?;
        let embeds = Tensor::from_vec(
            (0..d)
                .map(|i| ((i * 3 + pos) as f32 * 0.21).cos())
                .collect::<Vec<f32>>(),
            (1, 1, d),
            &dev,
        )?;
        dev.synchronize()?;
        let census = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("the census window");
        head.forward(&hidden, &embeds, &mut cache)?;
        dev.synchronize()?;
        let labels = census.close();
        if pos == 0 {
            eprintln!("head pos 0 labels: {labels:?}");
        }
        let ran = census_of_labels(labels.iter().map(|(l, n)| (l.as_str(), *n)));
        let walked = derive(&walk_at(&head, pos), &fusion_policy(&Policy::default()))?
            .graph
            .census();
        let d = differences(&walked, &ran);
        eprintln!(
            "head pos {pos}: ran {} walked {} {}",
            ran.values().sum::<u64>(),
            walked.values().sum::<u64>(),
            if d.is_empty() {
                "=".to_string()
            } else {
                format!("{d:?}")
            }
        );
        if !d.is_empty() {
            wrong.push((pos, d));
        }
    }
    assert!(
        wrong.is_empty(),
        "head passes that dispatch other than the head's walk (family, walked, ran): {wrong:?}"
    );
    Ok(())
}
