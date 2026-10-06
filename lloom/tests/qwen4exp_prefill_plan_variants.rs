//! Qwen4-exp's prefill chunks under the pass-plan cache (`DESIGN.md` §9.2aa):
//! one `Prefill { w, fresh }` kind, three allocation sequences past the QSA
//! indexer's dense range, each named by the model's variant
//! (`Qwen4Exp::prefill_variant`, `PassKey::variant`) so each is served its
//! own plan: a chunk inside the range attends densely; one crossing it runs a
//! dense part and then the selection; one wholly past it runs the selection
//! alone, as wide as the context's blocks (`pass-plan-design.md` "what would
//! move this"; the chunk path past the range is #897's).
//!
//! The chunks here are all of one width, so without the variant one key
//! names all three sequences and a crossing chunk is served the plan an
//! inside chunk recorded — the drift guard drops it (`counters.evicted`).
//! With the variant no plan drifts, the three kinds each hold a plan, the
//! repeat of the prompt is served, and its logits are the plans-off run's.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident, State};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKind, PassOpener};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// The recorder's bind observer and the dispatch clock are process globals.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// The chunk width; the tiny indexer (budget 16, ratio 4) puts the dense
/// range at 19 positions, so the chunks at 0 and 8 are inside it, the one
/// at 16 crosses it (16..24) and the ones at 24 and 32 are past it.
const W: usize = 8;
const CHUNKS: usize = 5;

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

fn resident(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<CrownExact, Resident>> {
    testing::model_with_indexer::<CrownExact, _>(cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })
}

/// The prompt as `CHUNKS` forwards of `W` tokens each on a fresh cache — the
/// model brackets every chunk as a pass of its own — and the last chunk's
/// logits.
fn prefill_in_chunks(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
) -> Result<Vec<f32>> {
    let mut cache: State<CrownExact> = model.new_cache(cfg, dev)?;
    let mut last = None;
    for c in 0..CHUNKS {
        let toks: Vec<u32> = (0..W).map(|i| ((c * W + i) * 7 + 3) as u32 % 120).collect();
        let input = Tensor::from_vec(toks, (1, W), dev)?;
        last = Some(model.forward(&input, cache.pos, &mut cache)?);
    }
    last.expect("a chunk").flatten_all()?.to_vec1::<f32>()
}

/// The model's prefill keys with a plan: `(fresh, variant, ordinals served,
/// ordinals recorded, undeclared escapes)`.
fn prefill_plans(m: &MetalDevice) -> Vec<(bool, u32, usize, usize, usize)> {
    let (keys, _) = m.pass_plan_report().expect("the cache is on");
    let mut v: Vec<_> = keys
        .into_iter()
        .filter_map(|(k, served, recorded, escapes)| match k.kind {
            PassKind::Prefill { w, fresh } if w as usize == W && k.opener == PassOpener::Model => {
                Some((fresh, k.variant, served, recorded, escapes))
            }
            _ => None,
        })
        .collect();
    v.sort();
    v
}

/// Each of the three chunk kinds past the dense range is served its own
/// plan: no plan is dropped for drifting, the keys name the inside,
/// crossing and past sequences, the prompt's repeat is served, and the
/// logits are the plans-off run's.
#[test]
fn each_chunk_kind_across_the_dense_range_is_served_its_own_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = tiny();
    let limit = {
        let dev = Device::new_metal(0)?;
        resident(&cfg, &dev)?
            .dense_limit()
            .expect("the tiny indexer")
    };
    assert!(
        2 * W <= limit && limit < 3 * W && 4 * W > limit,
        "the chunks must straddle the range: limit {limit}, width {W}"
    );
    let want = {
        let dev = Device::new_metal(0)?;
        prefill_in_chunks(&resident(&cfg, &dev)?, &cfg, &dev)?
    };
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let model = resident(&cfg, &dev)?;
    let first = prefill_in_chunks(&model, &cfg, &dev)?;
    let c1 = m.pass_plan_counters().expect("the cache is on");
    let again = prefill_in_chunks(&model, &cfg, &dev)?;
    let c2 = m.pass_plan_counters().expect("the cache is on");
    let plans = prefill_plans(m);
    eprintln!(
        "[prefill variants] limit {limit}, first {c1:?}, repeat {c2:?}, plans (fresh, variant, served, recorded, escapes) {plans:?}"
    );
    assert_eq!(
        first, want,
        "a planned chunk moved the logits (first sight)"
    );
    assert_eq!(again, want, "a served chunk moved the logits");
    assert_eq!(
        c2.evicted, 0,
        "a prefill plan drifted and was dropped: one key served a chunk of another kind"
    );
    let kinds: Vec<(bool, u32)> = plans.iter().map(|p| (p.0, p.1)).collect();
    assert_eq!(
        kinds,
        vec![(false, 0), (false, 1), (false, 2), (true, 0)],
        "one plan per (fresh, kind): inside, crossing and past the range"
    );
    // The repeat: every chunk whose key had a plan was served from it (the
    // first sight recorded four keys and observed the passes after them).
    assert!(
        c2.served >= c1.served + CHUNKS as u64 - 1,
        "the repeat was not served: {c1:?} then {c2:?}"
    );
    Ok(())
}
