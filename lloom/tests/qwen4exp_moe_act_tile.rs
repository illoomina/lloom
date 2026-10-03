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
//! on the path where it should be. The decode step's census shows gate|up as
//! one paired launch a layer on either dtype (`_f16_f16` under f16), so the
//! f16 step pays the f32 step's routed launch count. With `f16_fuse` the f16
//! step's SwiGLU is one `silu_mul_f16` and its combine reads the f16 routed
//! output (`moe_combine_f16`), so it pays one launch a layer over f32 (the
//! cast in); without it, three (`silu_f16` + `bmul_f16`, the cast out). With
//! `f16_fold` as well the cast in is gone too: gate|up read the f32
//! activation and round it on load (`_f32h_f16`), the f32 step's launch count.
//! All three f16 steps compute the same bits.
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
    let mut steps = Vec::new();
    for (act, fuse, fold) in [
        (DType::F32, true, false),
        (DType::F16, false, false),
        (DType::F16, true, false),
        (DType::F16, true, true),
    ] {
        let mut model = fx.resident(&cfg, &dev, &LoadOptions::default())?;
        model.set_routed_act(act);
        model.set_f16_fuse(fuse);
        model.set_f16_fold(fold);
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
        dev.synchronize()?;
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("a census window on this test's device");
        let decode: Vec<f32> = model
            .forward(&step, cache.pos, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        dev.synchronize()?;
        let step_census = window.close();
        let launches = |label: &str| -> u64 {
            step_census
                .iter()
                .filter(|(l, _)| l.contains(label))
                .map(|(_, n)| *n)
                .sum()
        };
        let suffix = if act == DType::F16 {
            "_f16_f16"
        } else {
            "_f32"
        };
        let pair_suffix = if fold { "_f32h_f16" } else { suffix };
        let (pair, single) = (
            launches(&format!("kernel_mul_mv_id2_q8_0{pair_suffix}")),
            launches(&format!("kernel_mul_mv_id_q8_0{suffix}")),
        );
        let named = |name: &str| -> u64 {
            step_census
                .iter()
                .filter(|(l, _)| l == name)
                .map(|(_, n)| *n)
                .sum()
        };
        let swiglu = [
            "silu_mul_f32",
            "silu_mul_f16",
            "silu_f16",
            "bmul_f16",
            "cast_f32_f16",
            "cast_f16_f32",
            "moe_combine_f32",
            "moe_combine_f16",
        ]
        .map(|k| (k, named(k)));
        println!(
            "MOEACT act={act:?} f16_fuse={fuse} f16_fold={fold} prefill_mul_mm_id={gemm} decode_mul_mv_id2={pair} decode_mul_mv_id={single} decode_routed={} {}",
            launches("mul_mv_id"),
            swiglu
                .iter()
                .map(|(k, n)| format!("{k}={n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        steps.push((act, fuse, fold, swiglu));
        let layers = cfg.n_layers as u64;
        assert_eq!(
            (pair, single, launches("mul_mv_id")),
            (layers, layers, 2 * layers),
            "{act:?}: a decode step's routed launches are not one gate|up pair and one down a layer ({step_census:?})"
        );
        runs.push((act, prefill, decode, gemm));
    }
    // Each f16 step's launches against the f32 step's, so the census reads
    // what the dtype adds whatever else the step casts.
    let layers = cfg.n_layers as i64;
    let n = |i: usize, k: &str| -> i64 {
        steps[i]
            .3
            .iter()
            .find(|(name, _)| *name == k)
            .expect("a counted kernel")
            .1 as i64
    };
    let (f32_i, off_i, on_i, fold_i) = (0, 1, 2, 3);
    for (i, label) in [(off_i, "f16_fuse off"), (on_i, "f16_fuse on")] {
        assert_eq!(
            n(i, "cast_f32_f16"),
            n(f32_i, "cast_f32_f16") + layers,
            "{label}: the routed activation's cast in is not one a layer"
        );
    }
    let off = (
        n(off_i, "silu_f16"),
        n(off_i, "bmul_f16"),
        n(off_i, "silu_mul_f16"),
        n(off_i, "cast_f16_f32") - n(f32_i, "cast_f16_f32"),
        n(off_i, "moe_combine_f16"),
    );
    assert_eq!(
        off,
        (layers, layers, 0, layers, 0),
        "f16_fuse off: the f16 step is not silu + mul and a cast out a layer"
    );
    let on = (
        n(on_i, "silu_f16"),
        n(on_i, "bmul_f16"),
        n(on_i, "silu_mul_f16"),
        n(on_i, "cast_f16_f32") - n(f32_i, "cast_f16_f32"),
        n(on_i, "moe_combine_f16"),
        n(on_i, "moe_combine_f32"),
    );
    assert_eq!(
        on,
        (
            0,
            0,
            layers,
            0,
            layers,
            n(f32_i, "moe_combine_f32") - layers
        ),
        "f16_fuse on: the f16 step's SwiGLU is not one launch, or its combine still reads a cast"
    );
    assert_eq!(
        n(fold_i, "cast_f32_f16"),
        n(f32_i, "cast_f32_f16"),
        "f16_fold on: the routed activation is still cast in a launch of its own"
    );
    for k in ["silu_mul_f16", "moe_combine_f16", "cast_f16_f32"] {
        assert_eq!(n(fold_i, k), n(on_i, k), "f16_fold on: {k} moved");
    }
    let (f32_run, off_run, f16_run, fold_run) = (&runs[0], &runs[1], &runs[2], &runs[3]);
    assert_eq!(
        bits(&f16_run.2),
        bits(&off_run.2),
        "the fused f16 step's logits are not the unfused f16 step's"
    );
    assert_eq!(
        bits(&fold_run.2),
        bits(&f16_run.2),
        "the folded f16 step's logits are not the cast-then-launch step's"
    );
    assert_eq!(bits(&fold_run.1), bits(&f16_run.1));
    assert_eq!(bits(&off_run.1), bits(&f16_run.1));
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
