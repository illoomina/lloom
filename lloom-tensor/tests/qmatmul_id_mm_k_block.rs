//! The routed GEMM (`kernel_mul_mm_id_mapped`) reads K in 32-wide blocks, as
//! the f16-weight GEMM does (`qmm_f16w_tiny_shapes`): a stacked weight whose
//! K is off the block must take the routed matmul-vec instead, at every row
//! count. The reference is a dense `x · Wᵀ` per selected expert on the CPU
//! (CONTRIBUTING §3.1b), and the fixture varies the axis the tile is
//! parameterised on — K — where `qmatmul_id_mm` holds it at 256 (§3.1b rule 5).
//!
//! Its own file: `set_moe_mm_from` is process-wide.
#![cfg(feature = "metal")]
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::quantized::metal::set_moe_mm_from;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};

/// `(max |got - want|, max |want|, whether the grouped GEMM ran)` for one
/// routed call at `(k, batch)`.
fn routed(dtype: GgmlDType, k: usize, batch: usize, from: usize) -> Result<(f32, f32, bool)> {
    let dev = Device::new_metal(0)?;
    let (e, n, topk) = (8usize, 32usize, 2usize);
    let w: Vec<f32> = (0..e * n * k)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.2)
        .collect();
    let w_cpu = Tensor::from_vec(w, (e, n, k), &Device::Cpu)?;
    let q = QTensor::quantize(&w_cpu.to_device(&dev)?, dtype)?;
    let w_ref = q.dequantize(&Device::Cpu)?;
    let x_cpu = Tensor::from_vec(
        (0..batch * k)
            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
            .collect::<Vec<f32>>(),
        (batch, 1, k),
        &Device::Cpu,
    )?;
    let ids: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 5 + i / topk) % e) as u32)
        .collect();
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &Device::Cpu)?;
    let (x, ids_d) = (x_cpu.to_device(&dev)?, ids_t.to_device(&dev)?);
    set_moe_mm_from(from);
    profile::reset();
    let got = q.indexed_moe_forward(&x, &ids_d)?.to_device(&Device::Cpu)?;
    let gemm = profile::snapshot()
        .by_label
        .iter()
        .any(|(l, _)| l.contains("mul_mm_id"));
    let mut maxdiff = 0f32;
    let mut scale = 0f32;
    for b in 0..batch {
        for t in 0..topk {
            let we = w_ref.get(ids[b * topk + t] as usize)?; // [n, k]
            let want: Vec<f32> = x_cpu.get(b)?.matmul(&we.t()?)?.flatten_all()?.to_vec1()?;
            let g: Vec<f32> = got.get(b)?.get(t)?.to_vec1()?;
            for (a, c) in g.iter().zip(&want) {
                maxdiff = maxdiff.max((a - c).abs());
                scale = scale.max(c.abs());
            }
        }
    }
    Ok((maxdiff, scale, gemm))
}

/// K ∈ {16, 48} (off the block) and {32, 64} (on it), with the grouped GEMM
/// enabled from one row, over enough rows that every expert takes a tile.
#[test]
fn a_k_off_the_32_block_never_reaches_the_grouped_gemm() -> Result<()> {
    // The recorder reads its switch once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    if Device::new_metal(0).is_err() {
        return Ok(());
    }
    assert!(profile::enabled(), "the census decides which arm ran");
    for dtype in [GgmlDType::F32, GgmlDType::F16] {
        for k in [16usize, 48, 32, 64] {
            for batch in [64usize, 65] {
                let (d, s, gemm) = routed(dtype, k, batch, 1)?;
                println!(
                    "ROUTED {dtype:?} k={k} batch={batch}: grouped GEMM {gemm} maxdiff {d:e} scale {s:e}"
                );
                assert_eq!(
                    gemm,
                    k % 32 == 0,
                    "{dtype:?} k={k} batch={batch}: the grouped GEMM ran = {gemm}"
                );
                assert!(
                    d <= 2e-3 * s.max(1.0),
                    "{dtype:?} k={k} batch={batch}: the routed arm differs from the dense \
                     reference by {d} at scale {s}"
                );
            }
        }
    }
    set_moe_mm_from(lloom_tensor::quantized::metal::DEFAULT_MOE_MM_FROM);
    Ok(())
}
