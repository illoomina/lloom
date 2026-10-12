//! #942: half rows use the same one-row reduction under deterministic.
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Result<Device> {
    static PIN: std::sync::Once = std::sync::Once::new();
    PIN.call_once(|| {
        lloom_tensor::tuning::pin_contract(lloom_tensor::tuning::Contract::Deterministic)
            .expect("isolated deterministic test process");
    });
    assert!(!lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
fn f16_rows_keep_the_one_row_bits() -> Result<()> {
    let dev = device()?;
    let mut checked = 0;
    for (n, k) in [(1, 64), (35, 64), (65, 256), (128, 2048)] {
        let w = Tensor::from_vec(
            (0..(n + 2) * k)
                .map(|i| (i as f32 * 0.61).sin() * 0.05)
                .collect::<Vec<_>>(),
            (n + 2, k),
            &dev,
        )?
        .to_dtype(DType::F16)?
        .narrow(0, 1, n)?;
        for transposed_weights in [false, true] {
            let rhs = if transposed_weights {
                w.t()?
            } else {
                w.t()?.contiguous()?
            };
            for m in 1..=8 {
                let x = Tensor::from_vec(
                    (0..(m + 2) * k)
                        .map(|i| (i as f32 * 0.013).sin())
                        .collect::<Vec<_>>(),
                    (m + 2, k),
                    &dev,
                )?
                .to_dtype(DType::F16)?
                .narrow(0, 1, m)?;
                for transposed_input in [false, true] {
                    let x = if transposed_input {
                        x.t()?.contiguous()?.t()?
                    } else {
                        x.clone()
                    };
                    let y = x.matmul(&rhs)?;
                    assert_eq!(bits(&y)?, bits(&x.matmul(&rhs)?)?, "repeat identity");
                    // The CPU is a numerical reference, not the bit-identity contract.
                    let cpu = x
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .contiguous()?
                        .matmul(&rhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?;
                    let actual = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let expected = cpu.flatten_all()?.to_vec1::<f32>()?;
                    for (a, b) in actual.iter().zip(expected) {
                        assert!((a - b).abs() < 0.002, "CPU parity: {a} vs {b}");
                    }
                    for row in 0..m {
                        let one = x.narrow(0, row, 1)?.contiguous()?.matmul(&rhs)?;
                        assert_eq!(bits(&y.narrow(0, row, 1)?)?, bits(&one)?,
                            "m={m} row={row} n={n} k={k} transposed input={transposed_input} weights={transposed_weights}");
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("F16_ROWS_IDENTITY rows_checked={checked} bit_equal={checked}");
    Ok(())
}

#[test]
fn f16_rows_keep_batched_weight_offsets() -> Result<()> {
    let dev = device()?;
    let (b, n, k) = (3, 35, 64);
    let w = Tensor::from_vec(
        (0..b * n * k)
            .map(|i| (i as f32 * 0.61).sin() * 0.05)
            .collect::<Vec<_>>(),
        (b, n, k),
        &dev,
    )?
    .to_dtype(DType::F16)?;
    for m in [1, 2, 3, 4, 8, 33] {
        let x = Tensor::from_vec(
            (0..b * m * k)
                .map(|i| (i as f32 * 0.013).sin())
                .collect::<Vec<_>>(),
            (b, m, k),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let y = x.matmul(&w.transpose(1, 2)?)?;
        for batch in 0..b {
            for row in 0..m {
                let one = x
                    .get(batch)?
                    .narrow(0, row, 1)?
                    .matmul(&w.get(batch)?.t()?)?;
                assert_eq!(
                    bits(&y.get(batch)?.narrow(0, row, 1)?)?,
                    bits(&one)?,
                    "batch={batch} m={m} row={row}"
                );
            }
        }
    }
    println!("F16_BATCHED_IDENTITY batches=3 rows=1,2,3,4,8,33 PASS");
    Ok(())
}
