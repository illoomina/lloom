//! How far the derived walk is from a declaration of LFM2's decode step,
//! dispatch by dispatch (`design/docs/planned-step-design.md` §2.3).
//!
//! A plan's command list is emitted in its graph's order, so the walk must
//! name the forward's dispatches in the forward's order, not only the right
//! count of each family at each position (`lfm2_walk_census`). This holds the
//! census as that test does, so the composition compared is the one the step
//! ran, and reports the order: the walked families against the families the
//! dispatch trace recorded, the first position where they part, and the
//! longest common subsequence; and the walk before the derivation, in the
//! order the blocks emitted it. It asserts the census and reports the order.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{
    AttnImpl, Config, FusionPolicy, KvAppend, LayerType, Model, Shape as Lfm2Shape, State,
};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::optimizer::ir::family;
use lloom::optimizer::{derive, kernel_family, walk_at, NumericClass, Policy};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};

const PROMPT: usize = 5;
const STEPS: usize = 2;

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

/// The composition the step runs, as `lfm2_walk_census`'s `policy` states
/// it for these two arms: the default arm composes `silu_mul` (the switch is
/// on by default) and no epilogue; the derived arm's plan sets the epilogues
/// and refuses every kernel fragment.
fn policy(cfg: &Config) -> Policy {
    let unavailable = |silu_mul: bool| {
        lloom::optimizer::fragments::kernels()
            .into_iter()
            .filter(|k| !(silu_mul && k.family == "silu_mul"))
            .map(|k| (k.name.to_string(), "no LFM2 switch".to_string()))
            .collect()
    };
    if cfg.fusion == FusionPolicy::Derived {
        return Policy {
            unavailable: unavailable(false),
            ..Policy::default()
        };
    }
    Policy {
        excluded: ["silu", "residual", "mul", "swiglu", "norm"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        barrier_model: None,
        budget: NumericClass::Rounding,
        unavailable: unavailable(true),
        ..Policy::default()
    }
}

/// Longest common subsequence of two family sequences.
fn lcs(a: &[String], b: &[String]) -> usize {
    let mut row = vec![0usize; b.len() + 1];
    for x in a {
        let mut prev = 0;
        for (j, y) in b.iter().enumerate() {
            let keep = row[j + 1];
            row[j + 1] = if x == y {
                prev + 1
            } else {
                row[j + 1].max(row[j])
            };
            prev = keep;
        }
    }
    row[b.len()]
}

fn sorted(v: &[String]) -> Vec<String> {
    let mut v = v.to_vec();
    v.sort();
    v
}

fn report<P: Lfm2Shape>(name: &str, cfg: &Config, dev: &Device) -> Result<()> {
    let model: Model<P> = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<P>(dev)?;
    let mut cache = State::<P>::new(true, DType::F16, cfg, dev)?;
    let prompt: Vec<u32> = (1..=PROMPT as u32).collect();
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, PROMPT))?;
    model.forward(&inp, 0, &mut cache)?;
    dev.synchronize()?;
    let policy = policy(cfg);
    for pos in PROMPT..PROMPT + STEPS {
        let tok = Tensor::new(&[pos as u32 + 1], dev)?.reshape((1, 1))?;
        let _ = lloom_tensor::kernels::metal::trace::take_dispatches();
        lloom_tensor::kernels::metal::trace::set_recording(true)?;
        model.forward(&tok, pos, &mut cache)?;
        dev.synchronize()?;
        lloom_tensor::kernels::metal::trace::set_recording(false)?;
        let ran: Vec<String> = lloom_tensor::kernels::metal::trace::take_dispatches()
            .iter()
            .map(|d| kernel_family(&d.pipeline))
            .collect();
        let source = walk_at(&model, pos);
        let g = derive(&source, &policy)?.graph;
        let walked: Vec<String> = g.nodes().iter().filter_map(family).collect();
        let unfused: Vec<String> = source.nodes().iter().filter_map(family).collect();
        assert_eq!(
            sorted(&walked),
            sorted(&ran),
            "{name} pos {pos}: the census differs, so the order below would compare \
             two compositions"
        );
        let first = walked.iter().zip(&ran).position(|(a, b)| a != b);
        let common = lcs(&walked, &ran);
        eprintln!(
            "ORDER {name} pos {pos}: dispatches={} in_order={} lcs={common} first_part={}",
            ran.len(),
            first.is_none(),
            first.map_or("-".to_string(), |i| format!(
                "{i} walked={} ran={}",
                walked[i], ran[i]
            )),
        );
        eprintln!("ORDER {name} pos {pos} ran:    {ran:?}");
        eprintln!("ORDER {name} pos {pos} walked: {walked:?}");
        eprintln!("ORDER {name} pos {pos} source: {unfused:?}");
    }
    Ok(())
}

#[test]
fn the_derived_walks_order_against_the_forwards() -> Result<()> {
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    let mut t = lloom::tuning::Tuning::default();
    t.from_env().expect("the tuning's environment");
    assert!(
        lloom::tuning::install(t).is_ok(),
        "the tuning was read before the test installed it"
    );
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    for (name, cfg) in [
        ("default", base()),
        (
            "derived",
            Config {
                fusion: FusionPolicy::Derived,
                ..base()
            },
        ),
    ] {
        let key = cfg.exact_key();
        lloom::profile::dispatch!(key, lloom::profile::LFM2, P => report::<P>(name, &cfg, &dev))?;
    }
    Ok(())
}
