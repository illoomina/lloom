//! The annotated shape checks (`lloom_tensor::shape_probe`,
//! `docs/porting-models.md` §21.8) at their production sites: each probe
//! reports the operand and the outcome the site acted on, and the dispatch
//! census agrees with the outcome — the kernel the check admits runs, and
//! the one it turns away does not.
#![cfg(feature = "metal")]

use lloom_tensor::kernels::metal::profile;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::shape_probe::{arm, Event, Kind};
use lloom_tensor::{DType, Device, Module, Result, Tensor};

fn device() -> Option<Device> {
    // The recorder reads its switch once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    Device::new_metal(0).ok()
}

fn site<'a>(events: &'a [Event], site: &str) -> Vec<&'a Event> {
    events.iter().filter(|e| e.site == site).collect()
}

fn labels(census: &[(String, u64)]) -> Vec<String> {
    census.iter().map(|(l, _)| l.clone()).collect()
}

/// `QMatMul::TensorF16` at a prefill's rows: K = 64 takes the f16-weight
/// GEMM, K = 40 (off its 32-wide block) the classical route; the direct
/// call refuses K = 40.
#[test]
fn the_f16_weight_gemm_check_routes_and_refuses_by_k() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    assert!(profile::enabled(), "the census must be on");
    let rows = 7;
    for (k, admitted) in [(64usize, true), (40, false)] {
        let w = Tensor::ones((96, k), DType::F16, &dev)?;
        let x = Tensor::ones((1, rows, k), DType::F32, &dev)?;
        let mm = QMatMul::TensorF16(w.clone());
        let a = arm();
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("one census window on this test's device");
        let y = mm.forward(&x)?;
        dev.synchronize()?;
        let census = window.close();
        let got = site(&a.take(), "f16_weight_gemm")
            .into_iter()
            .map(|e| (e.cond, e.kind, e.ok, e.operands.clone()))
            .collect::<Vec<_>>();
        // Admitted, the route reaches the GEMM, whose own refusal then
        // passes; turned away, the GEMM (and its refusal) is never reached.
        let mut want = vec![(
            "w.dims()[1] % 32 == 0",
            Kind::Route,
            admitted,
            vec![("k", k as i64)],
        )];
        if admitted {
            want.push(("k % 32 == 0", Kind::Refuse, true, vec![("k", k as i64)]));
        }
        assert_eq!(got, want, "K = {k}");
        let gemm = labels(&census).iter().any(|l| l.contains("mul_mm_f16_f32"));
        assert_eq!(gemm, admitted, "K = {k}: census {:?}", labels(&census));
        let v: Vec<f32> = y.flatten_all()?.to_vec1()?;
        assert!(
            v.iter().all(|&e| e == k as f32),
            "K = {k}: every output is K ones"
        );
        drop(a);

        let a = arm();
        let direct = lloom_tensor::quantized::metal::matmul_f16_weight_f32(&x, &w);
        let got = site(&a.take(), "f16_weight_gemm")
            .into_iter()
            .map(|e| (e.kind, e.ok))
            .collect::<Vec<_>>();
        assert_eq!(got, vec![(Kind::Refuse, admitted)], "K = {k}");
        assert_eq!(direct.is_ok(), admitted, "K = {k}");
    }
    Ok(())
}

/// `ops::sdpa` refuses a head dim outside the instantiated set, and the
/// probe carries the head dim it compared.
#[test]
fn the_sdpa_head_dim_check_names_the_head_dim() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    for (hd, admitted) in [(64usize, true), (16, false)] {
        let q = Tensor::ones((1, 2, 1, hd), DType::F32, &dev)?;
        let kv = Tensor::ones((1, 2, 5, hd), DType::F32, &dev)?;
        let a = arm();
        let out = lloom::nn::ops::sdpa(&q, &kv, &kv, None, false, 1.0, 1.0);
        let events = a.take();
        let head = site(&events, "sdpa_head_dim");
        assert_eq!(
            head.len(),
            2,
            "the set and the f32 × 512 exclusion: {head:?}"
        );
        assert_eq!(head[0].ok, admitted, "hd = {hd}");
        assert_eq!(head[0].operands, vec![("q_head", hd as i64)]);
        assert!(head[1].ok, "hd = {hd} is not 512");
        assert_eq!(out.is_ok(), admitted, "hd = {hd}");
    }
    Ok(())
}
