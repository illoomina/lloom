//! The f16 row axis's capacity rounding (`docs/tuning.md` §13.5zb): under
//! `fast`, with `kernels.gemm.f16_rows_round` on, a row-axis launch of an
//! odd number of rows takes the next even capacity — a batched decode
//! step's, a ragged step's live rows and a verify window's alike — and every
//! row's bits are the ones it has at capacity `m`, the one-row GEMV's. One
//! row keeps the one-row GEMV. The dispatch census names the capacity each
//! launch ran at, so the comparison is between kernels shown to differ.
//! Run with process isolation: the census switch is read once per process.
#![cfg(feature = "metal")]
use lloom_tensor::tuning::{
    f16_rows_rounded_launches, set_f16_rows, set_f16_rows_decode, set_f16_rows_round, DecodeRows,
    RaggedDecodeRows, VerifyWindow,
};
use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::kernels::metal::profile::enabled());
    assert!(lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).expect("a Metal device")
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

/// `f`'s result and the GEMV/GEMM pipelines it dispatched, by label.
fn kernels(dev: &Device, f: impl FnOnce() -> Result<Tensor>) -> Result<(Tensor, Vec<String>)> {
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    let y = f()?;
    dev.synchronize()?;
    let mut labels: Vec<String> = window
        .close()
        .into_iter()
        .map(|(l, _)| l)
        .filter(|l| l.starts_with("gemm") || l.starts_with("gemv"))
        .collect();
    labels.sort();
    Ok((y, labels))
}

/// The row axis's launch among `labels` at capacity `cap`: exactly one
/// label carries the `_r<cap>` name.
fn one_launch_at(labels: &[String], cap: usize, what: &str) {
    let suffix = format!("_r{cap}");
    let at: Vec<&String> = labels.iter().filter(|l| l.ends_with(&suffix)).collect();
    assert_eq!(
        at.len(),
        1,
        "{what}: expected one launch at capacity {cap}, got {labels:?}"
    );
}

/// The capacity the rounding must give `m` rows: the next even count for an
/// odd `m` from 3, `m` otherwise. Written out, not read from the route's
/// table, so a moved table fails here.
fn rounded(m: usize) -> usize {
    if m >= 3 && m % 2 == 1 {
        m + 1
    } else {
        m
    }
}

/// A batched decode step of every `m` from 2 to 16 under a ceiling of 16,
/// and a verify window of the same rows: with the rounding off each launch
/// runs at capacity `m`, with it on at [`rounded`]`(m)` — one rounded launch
/// counted for an odd `m`, none for an even one — and the two outputs are the
/// same bits, each row the one-row GEMV's, at LFM2.5's decode stems and an
/// `n` that is neither tile's multiple.
#[test]
fn an_odd_launch_takes_the_next_even_capacity_with_the_same_bits() -> Result<()> {
    let dev = device();
    set_f16_rows(Some(16));
    set_f16_rows_decode(true);
    let (mut checked, mut rounded_launches) = (0usize, 0usize);
    for (n, k) in [
        (35usize, 512usize),
        (2048, 2048),
        (6144, 2048),
        (2048, 10752),
    ] {
        let rhs = weight(n, k, &dev)?.t()?;
        for m in 2..=16 {
            let x = rows(m, k, m as f32 * 0.3, &dev)?;
            let one: Vec<Vec<u32>> = (0..m)
                .map(|r| bits(&x.narrow(0, r, 1)?.contiguous()?.matmul(&rhs)?))
                .collect::<Result<_>>()?;
            type Scope = fn() -> Box<dyn std::any::Any>;
            let scopes: [(&str, Scope); 2] = [
                ("decode step", || Box::new(DecodeRows::open())),
                ("verify window", || Box::new(VerifyWindow::open())),
            ];
            for (what, open) in scopes {
                set_f16_rows_round(false);
                let (off, off_labels) = kernels(&dev, || {
                    let _s = open();
                    x.matmul(&rhs)
                })?;
                one_launch_at(
                    &off_labels,
                    m,
                    &format!("[{n}, {k}] m={m} {what}, rounding off"),
                );
                set_f16_rows_round(true);
                let before = f16_rows_rounded_launches();
                let (on, on_labels) = kernels(&dev, || {
                    let _s = open();
                    x.matmul(&rhs)
                })?;
                let cap = rounded(m);
                one_launch_at(
                    &on_labels,
                    cap,
                    &format!("[{n}, {k}] m={m} {what}, rounding on"),
                );
                assert_eq!(
                    f16_rows_rounded_launches() - before,
                    u64::from(cap > m),
                    "[{n}, {k}] m={m} {what}: rounded launches counted"
                );
                rounded_launches += usize::from(cap > m);
                assert_eq!(
                    bits(&on)?,
                    bits(&off)?,
                    "[{n}, {k}] m={m} {what}: capacity {cap} moved a bit against capacity {m}"
                );
                for (r, want) in one.iter().enumerate() {
                    assert_eq!(
                        &bits(&on.narrow(0, r, 1)?)?,
                        want,
                        "[{n}, {k}] m={m} {what} row={r}: not the one-row GEMV's bits"
                    );
                    checked += 1;
                }
            }
        }
    }
    set_f16_rows_round(false);
    println!(
        "F16_ROWS_ROUND rows_checked={checked} bit_equal={checked} rounded_launches={rounded_launches}"
    );
    assert_eq!(
        rounded_launches,
        4 * 2 * 7,
        "every odd m from 3 to 15, at four shapes and two scopes, took a rounded capacity"
    );
    Ok(())
}

/// A ragged step's live rows round as a decode step's do — 3, 7 and 9 live
/// rows ahead of a five-row piece take capacities 4, 8 and 10 — and one live
/// row keeps the one-row GEMV under either setting. The whole output (live
/// rows and piece rows) is the same bits at either capacity, so a padded
/// lane stores nothing into the pieces' rows.
#[test]
fn a_ragged_steps_live_rows_round_and_one_live_row_does_not() -> Result<()> {
    let dev = device();
    set_f16_rows(Some(10));
    set_f16_rows_decode(true);
    let mut cases = 0usize;
    for (n, k) in [(2048usize, 2048usize), (2048, 10752)] {
        let rhs = weight(n, k, &dev)?.t()?;
        for live in [1usize, 2, 3, 7, 9] {
            let piece = 5usize;
            let m = live + piece;
            let x = rows(m, k, live as f32 * 0.7, &dev)?;
            set_f16_rows_round(false);
            let (off, off_labels) = kernels(&dev, || {
                let _s = RaggedDecodeRows::open(live, m, live + 1);
                x.matmul(&rhs)
            })?;
            set_f16_rows_round(true);
            let before = f16_rows_rounded_launches();
            let (on, on_labels) = kernels(&dev, || {
                let _s = RaggedDecodeRows::open(live, m, live + 1);
                x.matmul(&rhs)
            })?;
            let what = format!("[{n}, {k}] live={live} piece={piece}");
            if live == 1 {
                assert!(
                    !on_labels.iter().any(|l| l.contains("_r")),
                    "{what}: one live row took a row-axis capacity: {on_labels:?}"
                );
                assert_eq!(on_labels, off_labels, "{what}: one live row's launch moved");
            } else {
                one_launch_at(&off_labels, live, &format!("{what}, rounding off"));
                one_launch_at(&on_labels, rounded(live), &format!("{what}, rounding on"));
            }
            assert_eq!(
                f16_rows_rounded_launches() - before,
                u64::from(rounded(live) > live),
                "{what}: rounded launches counted"
            );
            assert_eq!(bits(&on)?, bits(&off)?, "{what}: the rounding moved a bit");
            cases += 1;
        }
    }
    set_f16_rows_round(false);
    println!("F16_ROWS_ROUND_RAGGED cases={cases}");
    Ok(())
}

/// The rounding reaches only a row-axis launch: outside a scope, with the
/// row axis off and past the ceiling, an odd matmul's kernels and bits are
/// the same with the switch on and off and nothing is counted; the same rows
/// in a decode step are rounded (the positive control).
#[test]
fn the_rounding_reaches_a_row_axis_launch_only() -> Result<()> {
    let dev = device();
    set_f16_rows_decode(true);
    let (n, k) = (512usize, 1024usize);
    let rhs = weight(n, k, &dev)?.t()?;
    let x7 = rows(7, k, 0.5, &dev)?;
    let x11 = rows(11, k, 0.25, &dev)?;
    type Case<'a> = (&'a str, Option<usize>, Box<dyn Fn() -> Result<Tensor> + 'a>);
    let cases: Vec<Case> = vec![
        ("outside a scope", Some(10), Box::new(|| x7.matmul(&rhs))),
        (
            "the row axis off",
            None,
            Box::new(|| {
                let _s = DecodeRows::open();
                x7.matmul(&rhs)
            }),
        ),
        (
            "eleven rows past a ceiling of ten",
            Some(10),
            Box::new(|| {
                let _s = DecodeRows::open();
                x11.matmul(&rhs)
            }),
        ),
    ];
    for (what, rows_max, run) in &cases {
        set_f16_rows(*rows_max);
        set_f16_rows_round(false);
        let (off, off_labels) = kernels(&dev, run)?;
        set_f16_rows_round(true);
        let before = f16_rows_rounded_launches();
        let (on, on_labels) = kernels(&dev, run)?;
        assert_eq!(
            f16_rows_rounded_launches(),
            before,
            "{what}: the rounding reached past the row axis"
        );
        assert_eq!(on_labels, off_labels, "{what}: the kernels moved");
        assert_eq!(bits(&on)?, bits(&off)?, "{what}: bits moved");
    }
    set_f16_rows(Some(10));
    set_f16_rows_round(true);
    let before = f16_rows_rounded_launches();
    let (_, labels) = kernels(&dev, || {
        let _s = DecodeRows::open();
        x7.matmul(&rhs)
    })?;
    one_launch_at(&labels, 8, "the control: seven rows in a decode step");
    assert_eq!(f16_rows_rounded_launches() - before, 1, "the control");
    set_f16_rows_round(false);
    println!("F16_ROWS_ROUND_SCOPE cases={} control=1", cases.len());
    Ok(())
}
