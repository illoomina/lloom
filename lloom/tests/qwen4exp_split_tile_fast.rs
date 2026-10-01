//! Under the `fast` contract each group of a prefill tile the streamed pool
//! splits takes the routed arm of its own rows (`MoeArm::for_group`), so the
//! split tile's logits depend on the pool — the relaxation of
//! `expert-streaming.md` §20.15a that `fast` licenses. The deterministic
//! twin is `qwen4exp_split_tile` (every group on the tile's arm, bit for bit
//! the resident model).
//!
//! The fixture is that test's: the tiny model widened until a 100-row tile
//! takes the GEMM on the resident model, split at 40 slots into groups far
//! under `LLOOM_MOE_MM_FROM`. Here the small groups take the matmul-vec, so
//! the logits leave the resident model's (the arm is on the path), stay
//! within a reordering of them, and a second run of the same pool gives the
//! same bits as
//! a fresh pool of the same size (the split is a function of the pool's
//! state, so the run is reproducible from the same state).
//!
//! Its own binary: the contract is read once per process.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Cache, Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::testing::fixture::Fixture;
use lloom_tensor::quantized::metal::moe_mm_from;
use lloom_tensor::{Device, Result, Tensor};

fn install_fast() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(
        lloom_tensor::tuning::contract().is_fast(),
        "the contract was read before this binary installed it"
    );
}

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
    }
}

fn prefill_logits<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
) -> Result<Vec<f32>> {
    let mut cache: Cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    model.forward(&input, 0, &mut cache)?.flatten_all()?.to_vec1()
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn under_fast_a_split_tiles_groups_take_their_own_arm() -> Result<()> {
    install_fast();
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_split_tile_fast", Config::tiny())? else {
        return Ok(());
    };
    if fx.at_scale() {
        // The tiny model's geometry is what puts a split tile's groups
        // under the GEMM's threshold; at scale `qwen4exp_split_tile` holds.
        return Ok(());
    }
    let cfg = fx.config(|c| {
        c.moe_intermediate_size = 32;
        c.indexer_budget = 128;
        c.n_experts = 64;
    });
    let opts = LoadOptions::default();
    let rows = 100usize;
    let prompt: Vec<u32> = (0..rows as u32).map(|i| (i * 37 + 5) % 128).collect();
    assert!(rows >= moe_mm_from() && moe_mm_from() > 0);

    let resident = fx.resident(&cfg, &dev, &opts)?;
    let want = prefill_logits(&resident, &cfg, &dev, &prompt)?;
    let streamed = fx.streamed(&cfg, &dev, &opts, &stream(40))?;
    let got = prefill_logits(&streamed, &cfg, &dev, &prompt)?;
    let st = streamed.stream_stats();
    let again = prefill_logits(&fx.streamed(&cfg, &dev, &opts, &stream(40))?, &cfg, &dev, &prompt)?;
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| {
        let d = (a - b).abs();
        if d.is_nan() || m.is_nan() {
            f32::NAN
        } else {
            m.max(d)
        }
    });
    println!(
        "SPLITFAST rows={rows} slots=40 plans={} groups={} max_abs={worst:e} scale={scale:e}",
        st.plans, st.groups
    );
    assert!(st.groups > st.plans, "the pool did not split the tile");
    assert!(
        bits(&got) != bits(&want),
        "the split tile's logits are the resident model's bit for bit: the groups did not take their own arm"
    );
    assert!(worst.is_finite() && worst <= 1e-3 * scale.max(1.0), "max |Δ| {worst} against a logit scale {scale}");
    assert_eq!(bits(&got), bits(&again), "two fresh pools of one size disagree");
    Ok(())
}
