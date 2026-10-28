//! The walk at a position against what LFM2's decode step at that position
//! dispatches (`optimizer::walk_at`, `design/docs/optimizer-design.md`
//! §10i), on every arm the walk describes: the two-layer tiny model built
//! at each arm, decoded past a prefill one position at a time, with the
//! 2-pass attention's length set low (`LLOOM_SDPA_TWO_PASS_K`) so that the
//! later positions take the two-launch kernel — each step's census from the
//! device's own window, in the IR's families, held to the walk at that
//! position (derived, for an arm whose epilogues the plan decides, or whose
//! MLP `kernels.fusion.silu_mul` composes). Both arms of that switch are
//! held, each in its own process: the switch is read once.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{
    AttnImpl, Config, ConvState, FusionPolicy, GateFusion, KvAppend, LayerType, Model, QkvLayout,
    Shape as Lfm2Shape, State,
};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::optimizer::{census_of_labels, derive, walk_at, Census, NumericClass, Policy};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};

/// The 2-pass length this binary runs at: positions from `TWO_PASS - 1` on
/// attend that many keys or more.
const TWO_PASS: usize = 8;

/// The positions decoded after the 5-token prefill.
const STEPS: usize = 8;

fn base() -> Config {
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
        attn_impl: AttnImpl::Sdpa,
        ..Default::default()
    }
}

/// How an arm's model is set after it is built: as configured, with the
/// norm arm's switch on, or derived at the `reorder` budget.
#[derive(Clone, Copy, PartialEq)]
enum Set {
    Built,
    NormGemv,
    DerivedReorder,
}

/// Every arm the walk describes, by name.
fn arms() -> Vec<(&'static str, Config, Set)> {
    let arm = |f: &dyn Fn(&mut Config)| {
        let mut c = base();
        f(&mut c);
        c
    };
    let built = |(name, cfg): (&'static str, Config)| (name, cfg, Set::Built);
    let mut arms: Vec<_> = vec![
        ("default", base()),
        ("conv-fused", arm(&|c| c.conv_fused = true)),
        (
            "rotating",
            arm(&|c| c.conv_state = ConvState::RotatingRing { k: 1 }),
        ),
        (
            "rotating-fused",
            arm(&|c| {
                c.conv_state = ConvState::RotatingRing { k: 1 };
                c.conv_fused = true;
            }),
        ),
        (
            "sliding",
            arm(&|c| c.conv_state = ConvState::SlidingRing { k: 1, slack: 2 }),
        ),
        ("qkv-strided", arm(&|c| c.qkv_layout = QkvLayout::Strided)),
        ("qkv-fused", arm(&|c| c.qkv_layout = QkvLayout::Fused)),
        ("kv-cat", arm(&|c| c.kv_append = KvAppend::Cat)),
        ("generic", arm(&|c| c.attn_impl = AttnImpl::Generic)),
        ("derived", arm(&|c| c.fusion = FusionPolicy::Derived)),
        ("crown", arm(&lloom::models::lfm2_presets::crown)),
    ]
    .into_iter()
    .map(built)
    .collect();
    // The norm inside its consumers' GEMVs (`LLOOM_LFM2_NORM_GEMV`'s
    // switch), alone and under the two-weight kernel; and the plan derived
    // at the budget that admits it, from the default arms.
    arms.push(("norm-gemv", base(), Set::NormGemv));
    arms.push((
        "norm-gemv-swiglu",
        arm(&|c| {
            c.gemv_gate = GateFusion::Swiglu;
            c.gemv_residual = true;
        }),
        Set::NormGemv,
    ));
    arms.push(("derived-reorder", base(), Set::DerivedReorder));
    arms
}

/// The kernel fragments an arm does not run: every one but `silu_mul`'s,
/// which `kernels.fusion.silu_mul` selects when `silu_mul` is set.
fn unavailable(silu_mul: bool) -> Vec<(String, String)> {
    lloom::optimizer::fragments::kernels()
        .into_iter()
        .filter(|k| !(silu_mul && k.family == "silu_mul"))
        .map(|k| (k.name.to_string(), "no LFM2 switch".to_string()))
        .collect()
}

/// The composition an arm runs: the derived plan under `Derived` (whose
/// derivation sets the epilogues alone, every kernel fragment refused);
/// under `Manual` the fragments its flags select, every other refused,
/// nothing re-priced (a hand composition is what it is); `None`, the
/// source. With `silu_mul` (the switch on) the MLP's SiLU and product are
/// `silu_mul` wherever no epilogue takes them, which the derivation's costs
/// decide as the forward's order does; the norm arm's MLP composes the pair.
fn policy(cfg: &Config, set: Set, silu_mul: bool) -> Option<Policy> {
    if set == Set::DerivedReorder {
        return Some(Policy {
            budget: NumericClass::ReorderEquiv,
            unavailable: unavailable(false),
            ..Policy::default()
        });
    }
    if cfg.fusion == FusionPolicy::Derived {
        return Some(Policy {
            unavailable: unavailable(false),
            ..Policy::default()
        });
    }
    let norm = set == Set::NormGemv;
    let mut excluded = Vec::new();
    for (on, name) in [
        (cfg.gemv_silu, "silu"),
        (cfg.gemv_residual, "residual"),
        (cfg.gemv_gate == GateFusion::Mul, "mul"),
        (cfg.gemv_gate == GateFusion::Swiglu, "swiglu"),
        (norm, "norm"),
    ] {
        if !on {
            excluded.push(name.to_string());
        }
    }
    let kernel = silu_mul && !norm;
    (excluded.len() < 5 || kernel).then(|| Policy {
        excluded,
        barrier_model: None,
        budget: if norm {
            NumericClass::ReorderEquiv
        } else {
            NumericClass::Rounding
        },
        unavailable: unavailable(kernel),
        ..Policy::default()
    })
}

/// The launches a sliding ring's compaction adds to a step (the live
/// window gathered, `Tensor::cat` with the pad), which the walk does not
/// describe: it is the ring's write phase, set by the steps since the pass
/// that seeded it, not by the position.
fn compaction() -> Vec<(String, u64, u64)> {
    vec![
        ("copy2d".to_string(), 0, 2),
        ("copy_strided".to_string(), 0, 1),
    ]
}

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

/// The positions of `cfg`'s arm whose step dispatches other than its walk,
/// how many of its steps took the 2-pass attention, and how many
/// `silu_mul` launches its steps ran.
fn check(
    name: &str,
    cfg: &Config,
    set: Set,
    silu_mul: bool,
    dev: &Device,
) -> Result<(Vec<Wrong>, usize, u64)> {
    let key = cfg.exact_key();
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => check_at::<P>(name, cfg, set, silu_mul, dev))
}

/// `check` at the profile the config names.
fn check_at<P: Lfm2Shape>(
    name: &str,
    cfg: &Config,
    set: Set,
    silu_mul: bool,
    dev: &Device,
) -> Result<(Vec<Wrong>, usize, u64)> {
    let mut model: Model<P> = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<P>(dev)?;
    match set {
        Set::Built => {}
        Set::NormGemv => model.set_norm_gemv(true),
        Set::DerivedReorder => {
            let r = model.derive_fusion_at(cfg, NumericClass::ReorderEquiv)?;
            eprintln!("{name}: {r}");
            assert!(
                r.norm_gemv,
                "{name}: the plan at `reorder` takes the norm prologue"
            );
        }
    }
    let mut cache = State::<P>::new(true, DType::F16, cfg, dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    dev.synchronize()?;
    let policy = policy(cfg, set, silu_mul);
    let sliding = matches!(cfg.conv_state, ConvState::SlidingRing { .. });
    let mut wrong = Vec::new();
    let mut two_pass = 0;
    let mut compactions = 0;
    let mut silu_mul_launches = 0;
    for pos in prompt.len()..prompt.len() + STEPS {
        let census = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("the census window");
        let tok = Tensor::new(&[(pos % 60) as u32 + 1], dev)?.reshape((1, 1))?;
        let _ = lloom_tensor::kernels::metal::trace::take_dispatches();
        lloom_tensor::kernels::metal::trace::set_recording(true)?;
        model.forward(&tok, pos, &mut cache)?;
        dev.synchronize()?;
        lloom_tensor::kernels::metal::trace::set_recording(false)?;
        let order: Vec<String> = lloom_tensor::kernels::metal::trace::take_dispatches()
            .iter()
            .map(|d| d.pipeline.clone())
            .collect();
        let labels = census.close();
        let ran = census_of_labels(labels.iter().map(|(l, n)| (l.as_str(), *n)));
        silu_mul_launches += ran.get("silu_mul").copied().unwrap_or(0);
        let g = walk_at(&model, pos);
        let walked = match &policy {
            Some(p) => derive(&g, p)?.graph.census(),
            None => g.census(),
        };
        let mut d = differences(&walked, &ran);
        // A sliding ring's compaction step: the walk's step and the
        // compaction's launches exactly, nothing else.
        if sliding && differences_net(&d) == compaction() {
            compactions += 1;
            d.clear();
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
        if !d.is_empty() {
            eprintln!("{name} pos {pos} order: {order:?}");
            wrong.push((pos, d));
        }
        two_pass += usize::from(walked.contains_key("sdpa_vector_2pass_1"));
    }
    if sliding {
        assert!(
            compactions >= 2,
            "{name}: the positions reach the ring's compaction"
        );
    }
    Ok((wrong, two_pass, silu_mul_launches))
}

/// `d` as (family, 0, ran − walked): what a step ran beyond its walk.
fn differences_net(d: &[(String, u64, u64)]) -> Vec<(String, u64, u64)> {
    d.iter()
        .map(|(f, walked, ran)| (f.clone(), 0, ran.saturating_sub(*walked)))
        .collect()
}

#[test]
fn every_arm_dispatches_what_its_walk_names_at_every_position() -> Result<()> {
    every_arm_at(true)
}

#[test]
fn every_arm_dispatches_what_its_walk_names_with_silu_mul_off() -> Result<()> {
    every_arm_at(false)
}

/// Every arm, with `kernels.fusion.silu_mul` at `silu_mul`; the steps of the
/// MLPs that run it dispatch `silu_mul` and neither `silu` nor that `bmul`.
fn every_arm_at(silu_mul: bool) -> Result<()> {
    // The switches are read once, on first use, by this process: the
    // environment is set before the tuning reads it.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    std::env::set_var("LLOOM_SDPA_TWO_PASS_K", TWO_PASS.to_string());
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    let mut t = lloom::tuning::Tuning::default();
    t.from_env().expect("the tuning's environment");
    t.lfm2.skip_silu_mul = !silu_mul;
    assert!(
        lloom::tuning::install(t).is_ok(),
        "the tuning was read before the arm was installed"
    );
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let mut wrong = Vec::new();
    let mut ran_silu_mul = 0;
    for (name, cfg, set) in arms() {
        let (w, two_pass, n) = check(name, &cfg, set, silu_mul, &dev)?;
        ran_silu_mul += n;
        if cfg.attn_impl == AttnImpl::Sdpa {
            assert!(
                two_pass >= 2,
                "{name}: the positions reach the 2-pass length"
            );
        }
        if !w.is_empty() {
            wrong.push((name, w));
        }
    }
    // The arm is the one installed: the default arms' MLPs run `silu_mul`
    // under the switch and never without it.
    eprintln!("silu_mul={silu_mul}: {ran_silu_mul} silu_mul launches over every arm");
    assert_eq!(
        ran_silu_mul > 0,
        silu_mul,
        "silu_mul={silu_mul}: {ran_silu_mul} silu_mul launches"
    );
    assert!(
        wrong.is_empty(),
        "arms whose steps dispatch other than their walk (position, (family, walked, ran)): {wrong:?}"
    );
    Ok(())
}
