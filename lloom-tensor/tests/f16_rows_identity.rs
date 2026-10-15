//! Half rows use the one-row reduction under `deterministic` at every row
//! count, and a pass off the K=1 decode path — a speculative verify window,
//! a prompt's prefill tile — takes the GEMM instead (`architecture.md`
//! §2.3.5b).
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

/// `|got − Σx·w|` beyond the store's half-ulp rounding, over `Σ|x·w|`: the
/// worst over every output of `y = x · wᵀ`, the reference summed in f64 over
/// the same half operands (`x [m, k]`, `w [n, k]`, `y [m, n]`).
fn worst_reduction_error(x: &Tensor, w: &Tensor, y: &Tensor) -> Result<f64> {
    let x = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let w = w.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let y = y.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let half_ulp = |v: f64| -> f64 {
        let v = v.abs();
        if v < 2f64.powi(-14) {
            2f64.powi(-25)
        } else {
            2f64.powi(v.log2().floor() as i32 - 11)
        }
    };
    let mut worst = 0f64;
    for (xr, yr) in x.iter().zip(&y) {
        for (wr, &got) in w.iter().zip(yr) {
            let (mut exact, mut scale) = (0f64, 0f64);
            for (&a, &b) in xr.iter().zip(wr) {
                exact += a as f64 * b as f64;
                scale += (a as f64 * b as f64).abs();
            }
            let got = got as f64;
            let err = ((got - exact).abs() - half_ulp(got.abs().max(exact.abs()))).max(0.0);
            worst = worst.max(err / scale.max(f64::MIN_POSITIVE));
        }
    }
    Ok(worst)
}

/// Bound on [`worst_reduction_error`]: four times the largest measured worst
/// (`7.6e-9` over every tile below) rounded up to a power of two; one wrong
/// weight costs about `1/k` of the scale, 2¹⁵ times more at `k = 512`.
const REDUCTION_BOUND: f64 = 1.0 / (1u64 << 24) as f64;

/// A prompt's prefill tile (`tuning::VerifyWindow::prefill`) takes the GEMM
/// at every row count from two, across the row tiles `select_tile_config`
/// picks (16 rows below 33, 64 above): each launch counted once on the
/// prefill window and on no other, the tile repeating bit for bit at its
/// shape, the same bits as a verify window's GEMM (the route `fast` takes),
/// and within the reduction bound of an f64 reference over the same half
/// operands. Outside the window and with its arm off, the same call is the
/// one-row reduction and counts nothing; some tile row differs from its
/// one-row bits, so another arithmetic ran.
#[test]
fn f16_prefill_tiles_take_the_gemm() -> Result<()> {
    use lloom_tensor::tuning::{VerifyWindow, Window};
    let dev = device()?;
    let (mut rows, mut differing, mut worst) = (0usize, 0usize, 0f64);
    for (n, k, ms) in [
        (256, 512, &[2usize, 7, 32, 33, 64, 100, 512][..]),
        (512, 2048, &[2, 33, 100][..]),
    ] {
        let w = Tensor::from_vec(
            (0..n * k)
                .map(|i| (i as f32 * 0.61).sin() * 0.05)
                .collect::<Vec<_>>(),
            (n, k),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let rhs = w.t()?;
        for &m in ms {
            let x = Tensor::from_vec(
                (0..m * k)
                    .map(|i| (i as f32 * 0.013).sin())
                    .collect::<Vec<_>>(),
                (m, k),
                &dev,
            )?
            .to_dtype(DType::F16)?;
            Window::Prefill.set_arm(true);
            let counts = || Window::ALL.map(|w| w.gemms());
            let before = counts();
            let (tile, again) = {
                let _prefill = VerifyWindow::prefill();
                (x.matmul(&rhs)?, x.matmul(&rhs)?)
            };
            let after = counts();
            assert_eq!(
                [
                    after[0] - before[0],
                    after[1] - before[1],
                    after[2] - before[2]
                ],
                [0, 0, 2],
                "m={m} n={n} k={k}: each prefill launch counted once, on the prefill window only"
            );
            assert_eq!(
                bits(&tile)?,
                bits(&again)?,
                "m={m}: the tile repeats at its shape"
            );
            let verify = {
                let _verify = VerifyWindow::open();
                x.matmul(&rhs)?
            };
            assert_eq!(
                bits(&tile)?,
                bits(&verify)?,
                "m={m}: the prefill tile takes the verify window's route"
            );
            let before = Window::Prefill.gemms();
            let outside = x.matmul(&rhs)?;
            Window::Prefill.set_arm(false);
            let arm_off = {
                let _prefill = VerifyWindow::prefill();
                x.matmul(&rhs)?
            };
            Window::Prefill.set_arm(true);
            assert_eq!(
                Window::Prefill.gemms(),
                before,
                "m={m}: outside a window, or with its arm off, nothing takes the prefill's GEMM"
            );
            assert_eq!(
                bits(&arm_off)?,
                bits(&outside)?,
                "m={m}: the arm off is the one-row reduction"
            );
            worst = worst.max(worst_reduction_error(&x, &w, &tile)?);
            for row in 0..m {
                let one = x.narrow(0, row, 1)?.contiguous()?.matmul(&rhs)?;
                assert_eq!(
                    bits(&outside.narrow(0, row, 1)?)?,
                    bits(&one)?,
                    "m={m} row={row}: outside the window, the one-row bits"
                );
                rows += 1;
                if bits(&tile.narrow(0, row, 1)?)? != bits(&one)? {
                    differing += 1;
                }
            }
        }
    }
    println!("F16_PREFILL_GEMM rows={rows} differing_from_one_row={differing} worst_reduction_error={worst:.3e}");
    assert!(
        worst <= REDUCTION_BOUND,
        "a prefill tile's GEMM is {worst:.3e} of its scale from the f64 reference (bound {REDUCTION_BOUND:.3e})"
    );
    assert!(
        differing > 0,
        "no prefill row of {rows} differs from its one-row reduction: the tile did not take the GEMM"
    );
    Ok(())
}

/// Two owners at once: one thread holds a prefill window open and keeps
/// launching its tile while another thread encodes batched K=1 decode rows
/// (two to eight rows, no window). Every decode row is its one-row
/// reduction's bits, every launch the prefill thread made is counted and no
/// decode launch is; the same rows under a window of their own differ, so the
/// comparison can see a leaked window.
#[test]
fn a_decode_beside_a_prefill_keeps_the_one_row_bits() -> Result<()> {
    use lloom_tensor::tuning::{VerifyWindow, Window};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    let dev = device()?;
    Window::Prefill.set_arm(true);
    let (n, k) = (256, 512);
    let rhs = Tensor::from_vec(
        (0..n * k)
            .map(|i| (i as f32 * 0.61).sin() * 0.05)
            .collect::<Vec<_>>(),
        (n, k),
        &dev,
    )?
    .to_dtype(DType::F16)?
    .t()?;
    let input = |m: usize, phase: f32| -> Result<Tensor> {
        Tensor::from_vec(
            (0..m * k)
                .map(|i| (i as f32 * 0.013 + phase).sin())
                .collect::<Vec<_>>(),
            (m, k),
            &dev,
        )?
        .to_dtype(DType::F16)
    };
    let decode: Vec<Tensor> = (2..=8).map(|m| input(m, 0.5)).collect::<Result<_>>()?;
    let mut one_row = Vec::new();
    let mut seen_by_a_window = 0usize;
    for x in &decode {
        let windowed = {
            let _prefill = VerifyWindow::prefill();
            x.matmul(&rhs)?
        };
        let mut rows = Vec::new();
        for row in 0..x.dim(0)? {
            let one = bits(&x.narrow(0, row, 1)?.contiguous()?.matmul(&rhs)?)?;
            if bits(&windowed.narrow(0, row, 1)?)? != one {
                seen_by_a_window += 1;
            }
            rows.push(one);
        }
        one_row.push(rows);
    }
    assert!(
        seen_by_a_window > 0,
        "a window moves none of the decode rows: this comparison could not see a leak"
    );

    let started = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));
    let before = Window::Prefill.gemms();
    let prefill = {
        let (started, done, rhs) = (started.clone(), done.clone(), rhs.clone());
        let tile = input(64, 0.0)?;
        std::thread::spawn(move || -> Result<u64> {
            let _prefill = VerifyWindow::prefill();
            let mut launched = 0u64;
            let _ = tile.matmul(&rhs)?;
            launched += 1;
            started.wait();
            while !done.load(Ordering::Acquire) {
                let _ = tile.matmul(&rhs)?;
                launched += 1;
            }
            Ok(launched)
        })
    };
    started.wait();
    let mut decode_rows = 0usize;
    for _ in 0..4 {
        for (x, expected) in decode.iter().zip(&one_row) {
            assert!(
                !Window::Prefill.is_open(),
                "the decode thread holds no window"
            );
            let y = x.matmul(&rhs)?;
            for (row, want) in expected.iter().enumerate() {
                assert_eq!(
                    &bits(&y.narrow(0, row, 1)?)?,
                    want,
                    "m={} row={row}: a decode row beside another thread's prefill keeps its one-row bits",
                    x.dim(0)?
                );
                decode_rows += 1;
            }
        }
    }
    done.store(true, Ordering::Release);
    let launched = prefill.join().expect("the prefill thread panicked")?;
    assert_eq!(
        Window::Prefill.gemms() - before,
        launched,
        "the prefill thread's launches counted, and no decode launch"
    );
    println!("F16_DECODE_BESIDE_PREFILL decode_rows={decode_rows} prefill_launches={launched} rows_a_window_moves={seen_by_a_window}");
    Ok(())
}
