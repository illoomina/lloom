//! `ops::sdpa`'s one-query arm (`sdpa_vector`) on Metal against a CPU
//! `softmax(q·kᵀ·scale)·v`, per head dim and dtype, with GQA: every head dim
//! the arm admits matches, and f32 at head dim 512 — where the kernel's
//! output is not the attention — is refused by name rather than returned,
//! as the multi-query arm's already was.
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Result, Tensor, D};

fn reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let (b, nh, _, hd) = q.dims4()?;
    let nkv = k.dim(1)?;
    let g = nh / nkv;
    let expand = |t: &Tensor| -> Result<Tensor> {
        let l = t.dim(2)?;
        t.unsqueeze(2)?
            .expand((b, nkv, g, l, hd))?
            .reshape((b, nh, l, hd))
    };
    let s = (q.matmul(&expand(k)?.t()?)? * scale)?;
    lloom::nn::ops::softmax_last_dim(&s)?.matmul(&expand(v)?)
}

/// `(max |metal - reference|, max |reference|)`, or the refusal.
fn one(dt: DType, hd: usize, kv_len: usize) -> Result<std::result::Result<(f32, f32), String>> {
    let gpu = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let n = |len: usize, seed: f32| -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    };
    let (nh, nkv) = (4, 2);
    let q = Tensor::from_vec(n(nh * hd, 0.1), (1, nh, 1, hd), &cpu)?;
    let k = Tensor::from_vec(n(nkv * kv_len * hd, 1.3), (1, nkv, kv_len, hd), &cpu)?;
    let v = Tensor::from_vec(n(nkv * kv_len * hd, 2.7), (1, nkv, kv_len, hd), &cpu)?;
    let scale = 1.0 / (hd as f64).sqrt();
    let want = reference(&q, &k, &v, scale)?;
    let on = |t: &Tensor| t.to_dtype(dt)?.to_device(&gpu);
    match lloom::nn::ops::sdpa(&on(&q)?, &on(&k)?, &on(&v)?, None, false, scale as f32, 1.0) {
        Err(e) => Ok(Err(e.to_string())),
        Ok(got) => {
            let got = got.to_dtype(DType::F32)?.to_device(&cpu)?;
            let d = (got - &want)?.abs()?.flatten_all()?.max(D::Minus1)?;
            let s = want.abs()?.flatten_all()?.max(D::Minus1)?;
            Ok(Ok((d.to_scalar::<f32>()?, s.to_scalar::<f32>()?)))
        }
    }
}

#[test]
fn the_one_query_arm_matches_or_refuses_by_head_dim() -> Result<()> {
    if Device::new_metal(0).is_err() {
        return Ok(());
    }
    for dt in [DType::F32, DType::F16] {
        let tol = if dt == DType::F32 { 1e-5 } else { 2e-3 };
        for hd in [64usize, 128, 256, 512] {
            for kv_len in [1usize, 5, 33] {
                let r = one(dt, hd, kv_len)?;
                println!("SDPA dt={dt:?} hd={hd} kv={kv_len}: {r:?}");
                if dt == DType::F32 && hd == 512 {
                    let e = r.expect_err("f32 at head dim 512 must be refused, not computed");
                    assert!(
                        e.contains("does not support"),
                        "refused for another reason: {e}"
                    );
                } else {
                    let (d, s) = r.map_err(lloom_tensor::Error::msg)?;
                    assert!(
                        d <= tol * s,
                        "dt={dt:?} hd={hd} kv={kv_len}: Metal differs by {d} at scale {s}"
                    );
                }
            }
        }
    }
    Ok(())
}
