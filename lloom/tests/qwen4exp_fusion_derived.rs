//! `qwen4exp.fusion = derived` on Metal (issue #732): the plan the
//! optimizer derives over the model's walk REACHES the forward
//! (`CONTRIBUTING.md` §7.1 F4) — a decode step under the derived plan runs
//! the kernels the plan names, and a plan with the brackets' fragments
//! excluded runs the composed chain the walk named in their place.
//!
//! The tiny model is built with f16 stems (`testing::set_f16_stems`), the
//! real checkpoint's dense-stem kind, so the brackets' projections take the
//! mixed GEMV and its epilogues as they do on the real model. The census is
//! the Metal backend's own recorder (`CANDLE_METAL_PROFILE`).
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::optimizer::Policy;
use lloom::profile::CrownExact;
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Option<Device> {
    // The recorder reads its switch once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    Device::new_metal(0).ok()
}

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.moe_intermediate_size = 32;
    cfg.head_dim = 32;
    cfg
}

fn model(dev: &Device) -> Qwen4Exp<CrownExact, Resident> {
    testing::set_f16_stems(true);
    testing::resident_model::<CrownExact>(&config(), dev)
}

/// A step's logits and its census, every label with its count.
type Step = (Vec<f32>, Vec<(String, u64)>);

/// Two decode steps from an empty cache; the second step's [`Step`].
fn step(m: &Qwen4Exp<CrownExact, Resident>, dev: &Device) -> Result<Step> {
    let cfg = config();
    let mut cache = m.new_cache(&cfg, dev)?;
    let first = Tensor::new(&[[3u32]], dev)?;
    m.decode(&first, &mut cache, &mut ())?;
    dev.synchronize()?;
    profile::reset();
    let next = Tensor::new(&[[5u32]], dev)?;
    let logits = m.decode(&next, &mut cache, &mut ())?;
    let logits = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    dev.synchronize()?;
    let s = profile::snapshot();
    assert!(s.dispatches > 0, "the census recorded nothing");
    Ok((logits, s.by_label.clone()))
}

fn count(census: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    census
        .iter()
        .filter(|(l, _)| pred(l))
        .map(|(_, n)| *n)
        .sum()
}

/// `(mixed GEMV + its SiLU, mixed GEMV + its sigmoid, standalone affine,
/// silu, sigmoid)` launches of a step. The standalone activations are
/// `silu_f32` / `sigmoid_f32`, not the routed `silu_mul_f32` or the
/// attention gate's `sigmoid_mul_f32`.
fn arms(census: &[(String, u64)]) -> (u64, u64, u64, u64, u64) {
    (
        count(census, |l| {
            l.starts_with("gemv_mixed") && l.ends_with("_silu")
        }),
        count(census, |l| {
            l.starts_with("gemv_mixed") && l.ends_with("_sigmoid")
        }),
        count(census, |l| l.starts_with("affine_f")),
        count(census, |l| l.starts_with("silu_f")),
        count(census, |l| l.starts_with("sigmoid_f")),
    )
}

#[test]
fn the_derived_plan_reaches_the_brackets() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let mut m = model(&dev);
    let brackets = m.bracket_epilogues().len() as u64;

    // Manual: every bracket takes both epilogues.
    let (manual, c_manual) = step(&m, &dev)?;
    // Derived, default policy: the plan is the hand-wired composition.
    let line = m.derive_fusion()?;
    eprintln!("{line}");
    let (derived, c_derived) = step(&m, &dev)?;
    // Derived with both bracket fragments excluded: the composed chain.
    let policy = Policy {
        excluded: vec!["mixed_silu".to_string(), "mixed_sigmoid".to_string()],
        ..Default::default()
    };
    let line = m.derive_fusion_with(&policy)?;
    eprintln!("{line}");
    let (composed, c_composed) = step(&m, &dev)?;

    let (a, b, c) = (arms(&c_manual), arms(&c_derived), arms(&c_composed));
    eprintln!("brackets {brackets}: manual {a:?} derived {b:?} composed {c:?}");
    assert_eq!(a, (brackets, brackets, 0, 0, 0), "manual");
    assert_eq!(
        b, a,
        "derived = the hand-wired composition, launch for launch"
    );
    assert_eq!(
        c,
        (0, 0, brackets, brackets, brackets),
        "excluded: the walk's composed chain, one affine, silu and sigmoid a bracket"
    );
    // The same kernels, the same bits; the composed chain rounds once more.
    assert_eq!(
        manual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        derived.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "derived computes the manual bits"
    );
    // The mixed arm accumulates and stores at f32, so the composed chain
    // rounds nothing the epilogue does not: measured bit-identical here (the
    // GEMV's f16 epilogues skip a store rounding; these do not). Pinned, so a
    // change to either side's arithmetic says so.
    let delta = manual
        .iter()
        .zip(&composed)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |manual - composed| = {delta}");
    assert_eq!(delta, 0.0, "the composed chain's bits, on this fixture");
    Ok(())
}

/// The brackets' grouped-norm arm (`QWEN4EXP_HC_NORM_GEMV`, set per
/// bracket) is a composition the catalogue cannot express — the collapse
/// reads the stream scales `down` writes — so `fusion = derived` refuses it
/// by name rather than derive a plan the arm would then override.
#[test]
fn the_derived_plan_refuses_the_grouped_norm_arm_by_name() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let mut m = model(&dev);
    m.set_hc_norm_gemv(true);
    let err = m
        .derive_fusion_with(&Policy::default())
        .expect_err("the arm cannot be derived");
    let msg = err.to_string();
    eprintln!("{msg}");
    assert!(
        msg.contains("qwen4exp.hc_norm_gemv") && msg.contains("not expressible"),
        "{msg}"
    );
    m.set_hc_norm_gemv(false);
    m.derive_fusion_with(&Policy::default())?;
    Ok(())
}
