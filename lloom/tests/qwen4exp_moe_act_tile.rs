//! The f16 routed activations (`--moe-act f16`, the `fast` contract's
//! `moe-act-f16` arm) reach the matmul-vec rows and leave a routed-GEMM tile
//! on f32: the GEMM has no f16 activation arm, so an f16 tile would fall back
//! to the matmul-vec planes for the whole prefill (`architecture.md` §2.3.5b).
//!
//! On `qwen4exp_split_tile`'s fixture (the tiny model widened until a 100-row
//! tile takes the GEMM on the resident model), its experts stored Q8_0 (an
//! f32 expert has no f16 activation arm at all): the prefill's logits under f16
//! are the f32 run's bit for bit and the census shows the GEMM ran; the
//! decode step after it (one row, the matmul-vec) differs, so the f16 arm is
//! on the path where it should be.
//!
//! Its own binary: the census switch and the contract are read once.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{Cache, Config, LoadOptions};
use lloom::nn::model::CausalLM;
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::Weights;
use lloom_tensor::quantized::metal::moe_mm_from;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::kernels::metal::profile::enabled());
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn an_f16_routed_tile_stays_on_the_gemm_and_the_decode_row_takes_f16() -> Result<()> {
    install();
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_moe_act_tile", Config::tiny())? else {
        return Ok(());
    };
    let fx = fx.with_weights(Weights {
        experts: GgmlDType::Q8_0,
        ..Weights::default()
    });
    if fx.at_scale() {
        return Ok(());
    }
    let cfg = fx.config(|c| {
        c.moe_intermediate_size = 32;
        c.indexer_budget = 128;
        c.n_experts = 64;
    });
    let rows = 100usize;
    assert!(rows >= moe_mm_from() && moe_mm_from() > 0);
    let prompt: Vec<u32> = (0..rows as u32).map(|i| (i * 37 + 5) % 128).collect();

    let mut runs = Vec::new();
    for act in [DType::F32, DType::F16] {
        let mut model = fx.resident(&cfg, &dev, &LoadOptions::default())?;
        model.set_routed_act(act);
        let mut cache: Cache = model.new_cache(&cfg, &dev)?;
        dev.synchronize()?;
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("a census window on this test's device");
        let input = Tensor::from_vec(prompt.clone(), (1, rows), &dev)?;
        let prefill: Vec<f32> = model
            .forward(&input, 0, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        dev.synchronize()?;
        let census = window.close();
        let gemm: u64 = census
            .iter()
            .filter(|(l, _)| l.contains("mul_mm_id"))
            .map(|(_, n)| *n)
            .sum();
        let step = Tensor::from_vec(vec![7u32], (1, 1), &dev)?;
        let decode: Vec<f32> = model
            .forward(&step, cache.pos, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        println!("MOEACT act={act:?} prefill_mul_mm_id={gemm}");
        runs.push((act, prefill, decode, gemm));
    }
    let (f32_run, f16_run) = (&runs[0], &runs[1]);
    assert!(f32_run.3 > 0, "the f32 tile did not take the routed GEMM");
    assert_eq!(
        f16_run.3, f32_run.3,
        "the f16 tile left the routed GEMM (the matmul-vec planes ran it)"
    );
    assert_eq!(
        bits(&f16_run.1),
        bits(&f32_run.1),
        "the f16 tile's prefill logits are not the f32 tile's"
    );
    assert_ne!(
        bits(&f16_run.2),
        bits(&f32_run.2),
        "the decode row computed the same bits on f16: the f16 arm is not on the matmul-vec path"
    );
    assert!(f16_run.2.iter().all(|v| v.is_finite()));
    Ok(())
}
