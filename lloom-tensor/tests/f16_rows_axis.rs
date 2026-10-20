//! The GEMV's row axis on the dense f16 route (`docs/tuning.md` §13.5y):
//! under `fast`, inside a speculative verify window and with
//! `kernels.gemm.f16_rows` on, a half matmul of 2 to `f16_rows_max` rows
//! sweeps its weight once for every row, each row the one-row GEMV's bits.
//! Everywhere else the route is the one it was.
#![cfg(feature = "metal")]
use lloom_tensor::tuning::{
    f16_rows_launches, set_f16_rows, verify_window_gemms, Contract, VerifyWindow, Window,
};
use lloom_tensor::{DType, Device, Result, Tensor};

fn device(contract: Contract) -> Result<Device> {
    lloom_tensor::tuning::pin_contract(contract).expect("an isolated test process");
    assert_eq!(lloom_tensor::tuning::contract(), contract);
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

fn weight(n: usize, k: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..(n + 2) * k)
            .map(|i| (i as f32 * 0.61).sin() * 0.05)
            .collect::<Vec<_>>(),
        (n + 2, k),
        dev,
    )?
    .to_dtype(DType::F16)?
    .narrow(0, 1, n)
}

fn rows(m: usize, k: usize, phase: f32, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..(m + 2) * k)
            .map(|i| (i as f32 * 0.013 + phase).sin())
            .collect::<Vec<_>>(),
        (m + 2, k),
        dev,
    )?
    .to_dtype(DType::F16)?
    .narrow(0, 1, m)
}

/// Inside a verify window with the arm on, every `m` from 2 to the ceiling
/// takes the row axis — one launch counted per matmul, nothing on the
/// window's GEMM count — and each row is the one-row GEMV's bits (`x[r] · Wᵀ`
/// alone, the `m = 1` route), at both decode tiles, an `n` that is not a
/// multiple of either, offset operands, and a batch folded into the rows.
/// With the arm off the same window takes the GEMM, and some row of it
/// differs from its one-row bits: the comparison can see which route ran.
#[test]
fn a_verify_windows_f16_rows_take_the_row_axis_with_the_one_row_bits() -> Result<()> {
    let dev = device(Contract::Fast)?;
    let (mut checked, mut gemm_rows_differing, mut gemm_rows) = (0usize, 0usize, 0usize);
    for (n, k) in [(35usize, 512usize), (4100, 512), (2048, 2048)] {
        let rhs = weight(n, k, &dev)?.t()?;
        for m in 2..=16 {
            let x = rows(m, k, m as f32, &dev)?;
            set_f16_rows(Some(16));
            let (launches, gemms) = (f16_rows_launches(), verify_window_gemms());
            let (y, again) = {
                let _window = VerifyWindow::open();
                (x.matmul(&rhs)?, x.matmul(&rhs)?)
            };
            assert_eq!(
                f16_rows_launches() - launches,
                2,
                "[{n}, {k}] m={m}: each window matmul one row-axis launch"
            );
            assert_eq!(verify_window_gemms(), gemms, "no GEMM counted under fast");
            assert_eq!(bits(&y)?, bits(&again)?, "m={m}: repeat identity");
            set_f16_rows(None);
            let gemm = {
                let _window = VerifyWindow::open();
                x.matmul(&rhs)?
            };
            assert_eq!(
                f16_rows_launches() - launches,
                2,
                "m={m}: with the arm off nothing takes the row axis"
            );
            for r in 0..m {
                let one = bits(&x.narrow(0, r, 1)?.contiguous()?.matmul(&rhs)?)?;
                assert_eq!(
                    bits(&y.narrow(0, r, 1)?)?,
                    one,
                    "[{n}, {k}] m={m} row={r}: the row axis is not the one-row GEMV's bits"
                );
                checked += 1;
                gemm_rows += 1;
                if bits(&gemm.narrow(0, r, 1)?)? != one {
                    gemm_rows_differing += 1;
                }
            }
        }
    }
    // A batch the dispatcher folds into the rows: `[2, 3, k] · Wᵀ` is six rows.
    let (n, k) = (2048, 2048);
    let rhs = weight(n, k, &dev)?.t()?;
    let x = rows(6, k, 0.25, &dev)?.reshape((2, 3, k))?;
    set_f16_rows(Some(16));
    let launches = f16_rows_launches();
    let y = {
        let _window = VerifyWindow::open();
        x.broadcast_matmul(&rhs)?
    };
    assert_eq!(
        f16_rows_launches() - launches,
        1,
        "a folded batch is one launch"
    );
    for b in 0..2 {
        for r in 0..3 {
            let one = x.get(b)?.narrow(0, r, 1)?.contiguous()?.matmul(&rhs)?;
            assert_eq!(
                bits(&y.get(b)?.narrow(0, r, 1)?)?,
                bits(&one)?,
                "batch={b} row={r}"
            );
            checked += 1;
        }
    }
    println!(
        "F16_ROWS_AXIS rows_checked={checked} bit_equal={checked} \
         gemm_rows_differing_from_one_row={gemm_rows_differing}/{gemm_rows}"
    );
    assert!(
        gemm_rows_differing > 0,
        "no GEMM row differs from its one-row bits: this comparison cannot tell the routes apart"
    );
    Ok(())
}

/// The arm reaches only what it names: outside a window, in the drafter's or
/// the prefill's window, at one row, past the ceiling, with per-batch
/// weights or a weight laid `[k, n]` (the transposed GEMV), nothing takes the
/// row axis, and each of those is the bits it was with the arm off.
#[test]
fn the_row_axis_is_taken_in_a_verify_window_only() -> Result<()> {
    let dev = device(Contract::Fast)?;
    let (n, k) = (512usize, 1024usize);
    let w = weight(n, k, &dev)?;
    let rhs = w.t()?;
    let x = rows(10, k, 0.5, &dev)?;
    let x17 = rows(17, k, 0.75, &dev)?;
    let x1 = rows(1, k, 1.0, &dev)?;
    type Case<'a> = (&'a str, Box<dyn Fn() -> Result<Tensor> + 'a>);
    let cases: Vec<Case> = vec![
        ("outside a window", Box::new(|| x.matmul(&rhs))),
        (
            "the drafter's window",
            Box::new(|| {
                let _w = VerifyWindow::draft();
                x.matmul(&rhs)
            }),
        ),
        (
            "the prefill's window",
            Box::new(|| {
                let _w = VerifyWindow::prefill();
                x.matmul(&rhs)
            }),
        ),
        (
            "one row",
            Box::new(|| {
                let _w = VerifyWindow::open();
                x1.matmul(&rhs)
            }),
        ),
        (
            "seventeen rows (past the ceiling)",
            Box::new(|| {
                let _w = VerifyWindow::open();
                x17.matmul(&rhs)
            }),
        ),
        (
            "per-batch weights",
            Box::new(|| {
                let _w = VerifyWindow::open();
                let wb = Tensor::stack(&[&w, &w], 0)?;
                Tensor::stack(&[&x, &x], 0)?.matmul(&wb.transpose(1, 2)?)
            }),
        ),
        (
            "a [k, n] weight",
            Box::new(|| {
                let _w = VerifyWindow::open();
                x.matmul(&rhs.contiguous()?)
            }),
        ),
    ];
    let mut checked = 0;
    for (what, run) in &cases {
        set_f16_rows(None);
        let off = run()?;
        set_f16_rows(Some(16));
        let before = f16_rows_launches();
        let on = run()?;
        assert_eq!(
            f16_rows_launches(),
            before,
            "{what}: the row axis is taken where the arm does not reach"
        );
        assert_eq!(
            bits(&on)?,
            bits(&off)?,
            "{what}: the arm moved bits it does not reach"
        );
        checked += 1;
    }
    // The positive control: the same rows in a verify window do take it.
    let before = f16_rows_launches();
    {
        let _w = VerifyWindow::open();
        x.matmul(&rhs)?;
    }
    assert_eq!(
        f16_rows_launches() - before,
        1,
        "the control: a verify window takes it"
    );
    assert!(!Window::Verify.is_open());
    println!("F16_ROWS_SCOPE cases={checked} control=1");
    Ok(())
}

/// `deterministic` never takes the row axis: with the arm on, a verify
/// window's half rows take the window's GEMM (decision 20) as they did.
#[test]
fn deterministic_never_takes_the_row_axis() -> Result<()> {
    let dev = device(Contract::Deterministic)?;
    let rhs = weight(512, 1024, &dev)?.t()?;
    let x = rows(10, 1024, 0.5, &dev)?;
    set_f16_rows(None);
    let off = {
        let _w = VerifyWindow::open();
        x.matmul(&rhs)?
    };
    set_f16_rows(Some(16));
    let (launches, gemms) = (f16_rows_launches(), verify_window_gemms());
    let on = {
        let _w = VerifyWindow::open();
        x.matmul(&rhs)?
    };
    assert_eq!(
        f16_rows_launches(),
        launches,
        "deterministic took the row axis"
    );
    assert_eq!(verify_window_gemms() - gemms, 1, "the window's GEMM ran");
    assert_eq!(bits(&on)?, bits(&off)?);
    assert!(
        !lloom_tensor::tuning::window_report().contains("f16_rows=rows"),
        "the receipt names the arm under deterministic: {}",
        lloom_tensor::tuning::window_report()
    );
    println!("F16_ROWS_DETERMINISTIC launches=0 window_gemm=1");
    Ok(())
}

/// The receipts name the arm while it is on, under `fast`: the contract line
/// (`arms=…,f16-rows`) and `VERIFY_WINDOW`'s `f16_rows=rows-<max>` with its
/// launch count.
#[test]
fn the_receipts_name_the_row_axis() -> Result<()> {
    let dev = device(Contract::Fast)?;
    set_f16_rows(None);
    let off = (
        Contract::Fast.report(),
        lloom_tensor::tuning::window_report(),
    );
    assert!(!off.0.contains("f16-rows"), "{}", off.0);
    assert!(off.1.contains("f16_rows=off"), "{}", off.1);
    set_f16_rows(Some(12));
    let rhs = weight(256, 512, &dev)?.t()?;
    let x = rows(8, 512, 0.0, &dev)?;
    {
        let _w = VerifyWindow::open();
        x.matmul(&rhs)?;
    }
    let on = (
        Contract::Fast.report(),
        lloom_tensor::tuning::window_report(),
    );
    assert!(on.0.contains("f16-rows"), "{}", on.0);
    assert!(on.1.contains("f16_rows=rows-12"), "{}", on.1);
    assert!(
        on.1.contains(&format!("f16_rows_launches={}", f16_rows_launches())),
        "{}",
        on.1
    );
    assert!(f16_rows_launches() >= 1);
    println!("F16_ROWS_RECEIPT off=[{}] on=[{}]", off.1, on.1);
    Ok(())
}
