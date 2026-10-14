//! Half rows use the one-row reduction under `deterministic` at every row
//! count, and a speculative verify window takes the GEMM instead
//! (`architecture.md` §2.3.5b).
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

/// Inside a verify window (`tuning::VerifyWindow`) a half matmul of two rows
/// or more takes the GEMM: each launch is counted once, some row differs
/// from its one-row reduction (another arithmetic ran), the window repeats
/// bit for bit at its shape and matches the CPU. Outside the window, and with
/// the window's arm off, the same call is the one-row reduction and counts
/// nothing.
#[test]
fn f16_window_rows_take_the_gemm() -> Result<()> {
    use lloom_tensor::tuning::{set_verify_window_arm, verify_window_gemms, VerifyWindow};
    let dev = device()?;
    let (mut rows, mut differing) = (0usize, 0usize);
    for (n, k) in [(256, 512), (512, 2048)] {
        let w = Tensor::from_vec(
            (0..n * k)
                .map(|i| (i as f32 * 0.61).sin() * 0.05)
                .collect::<Vec<_>>(),
            (n, k),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let rhs = w.t()?;
        for m in 2..=10 {
            let x = Tensor::from_vec(
                (0..m * k)
                    .map(|i| (i as f32 * 0.013).sin())
                    .collect::<Vec<_>>(),
                (m, k),
                &dev,
            )?
            .to_dtype(DType::F16)?;
            set_verify_window_arm(true);
            let before = verify_window_gemms();
            let (win, again) = {
                let _window = VerifyWindow::open();
                (x.matmul(&rhs)?, x.matmul(&rhs)?)
            };
            assert_eq!(
                verify_window_gemms() - before,
                2,
                "m={m} n={n} k={k}: each window launch counted once"
            );
            assert_eq!(
                bits(&win)?,
                bits(&again)?,
                "m={m}: the window repeats at its shape"
            );
            let before = verify_window_gemms();
            let outside = x.matmul(&rhs)?;
            set_verify_window_arm(false);
            let arm_off = {
                let _window = VerifyWindow::open();
                x.matmul(&rhs)?
            };
            set_verify_window_arm(true);
            assert_eq!(
                verify_window_gemms(),
                before,
                "m={m}: outside a window, or with its arm off, nothing takes the window's GEMM"
            );
            assert_eq!(
                bits(&arm_off)?,
                bits(&outside)?,
                "m={m}: the arm off is the one-row reduction"
            );
            let cpu = x
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .matmul(&rhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?;
            let actual = win.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for (a, b) in actual.iter().zip(cpu.flatten_all()?.to_vec1::<f32>()?) {
                assert!((a - b).abs() < 0.002, "CPU parity: {a} vs {b}");
            }
            for row in 0..m {
                let one = x.narrow(0, row, 1)?.contiguous()?.matmul(&rhs)?;
                assert_eq!(
                    bits(&outside.narrow(0, row, 1)?)?,
                    bits(&one)?,
                    "m={m} row={row}: outside the window, the one-row bits"
                );
                rows += 1;
                if bits(&win.narrow(0, row, 1)?)? != bits(&one)? {
                    differing += 1;
                }
            }
        }
    }
    println!("F16_WINDOW_GEMM rows={rows} differing_from_one_row={differing}");
    assert!(
        differing > 0,
        "no window row of {rows} differs from its one-row reduction: the window did not take the GEMM"
    );
    Ok(())
}
