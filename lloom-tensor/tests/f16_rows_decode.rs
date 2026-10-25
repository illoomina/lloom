//! The GEMV's row axis for a batched decode step (`docs/tuning.md` §13.5zb):
//! under `fast`, with `kernels.gemm.f16_rows` and
//! `kernels.gemm.f16_rows_decode` on, inside a batched decode step's forward
//! (`tuning::DecodeRows`), a half matmul of 2 to `f16_rows_max` rows sweeps
//! its weight once for every row, each row the one-row GEMV's bits; in a
//! step that carries prompt pieces (`tuning::RaggedDecodeRows`) its live
//! rows do the same and the pieces keep the GEMM's bits. Everywhere else the
//! route is the one it was.
#![cfg(feature = "metal")]
use lloom_tensor::tuning::{
    f16_rows_decode_launches, f16_rows_launches, set_f16_rows, set_f16_rows_decode, Contract,
    DecodeRows, RaggedDecodeRows, VerifyWindow,
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

/// Inside a decode step with both switches on, every `m` from 2 to the
/// ceiling takes the row axis — one decode launch counted per matmul, none
/// on the verify window's count — and each row is the one-row GEMV's bits,
/// at LFM2.5's decode shapes and an `n` that is neither tile's multiple.
/// With the decode switch off the same step takes the GEMM, and some row of
/// it differs from its one-row bits, so the comparison sees which route ran.
#[test]
fn a_decode_steps_f16_rows_take_the_row_axis_with_the_one_row_bits() -> Result<()> {
    let dev = device(Contract::Fast)?;
    set_f16_rows(Some(10));
    let (mut checked, mut gemm_differing, mut gemm_rows) = (0usize, 0usize, 0usize);
    for (n, k) in [
        (35usize, 512usize),
        (2048, 2048),
        (6144, 2048),
        (2048, 10752),
    ] {
        let rhs = weight(n, k, &dev)?.t()?;
        for m in 2..=10 {
            let x = rows(m, k, m as f32, &dev)?;
            set_f16_rows_decode(true);
            let (decode, verify) = (f16_rows_decode_launches(), f16_rows_launches());
            let y = {
                let _step = DecodeRows::open();
                x.matmul(&rhs)?
            };
            assert_eq!(
                f16_rows_decode_launches() - decode,
                1,
                "[{n}, {k}] m={m}: one decode launch"
            );
            assert_eq!(f16_rows_launches(), verify, "nothing on the verify count");
            set_f16_rows_decode(false);
            let gemm = {
                let _step = DecodeRows::open();
                x.matmul(&rhs)?
            };
            assert_eq!(
                f16_rows_decode_launches() - decode,
                1,
                "m={m}: with the decode switch off nothing takes the row axis"
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
                    gemm_differing += 1;
                }
            }
        }
    }
    println!(
        "F16_ROWS_DECODE rows_checked={checked} bit_equal={checked} \
         gemm_rows_differing_from_one_row={gemm_differing}/{gemm_rows}"
    );
    assert!(
        gemm_differing > 0,
        "no GEMM row differs from its one-row bits: the comparison cannot tell the routes apart"
    );
    Ok(())
}

/// A ceiling anywhere up to the axis's last capacity changes which rows take
/// the axis, never what a row computes there: with the ceiling at 16, every
/// `m` from 11 to 16 takes it and each row is the one-row GEMV's bits, at
/// LFM2.5's decode stems and at a `k = 5120` model's down projection. With
/// the decode switch off some GEMM row differs, so the comparison sees the
/// route.
#[test]
fn every_ceiling_to_sixteen_rows_keeps_the_one_row_bits() -> Result<()> {
    let dev = device(Contract::Fast)?;
    set_f16_rows(Some(16));
    let (mut checked, mut gemm_differing) = (0usize, 0usize);
    for (n, k) in [
        (35usize, 512usize),
        (2048, 2048),
        (6144, 2048),
        (2048, 10752),
        (5120, 17408),
    ] {
        let rhs = weight(n, k, &dev)?.t()?;
        for m in 11..=16 {
            let x = rows(m, k, m as f32 * 0.5, &dev)?;
            set_f16_rows_decode(true);
            let before = f16_rows_decode_launches();
            let y = {
                let _step = DecodeRows::open();
                x.matmul(&rhs)?
            };
            assert_eq!(
                f16_rows_decode_launches() - before,
                1,
                "[{n}, {k}] m={m}: one decode launch under a ceiling of 16"
            );
            set_f16_rows_decode(false);
            let gemm = {
                let _step = DecodeRows::open();
                x.matmul(&rhs)?
            };
            for r in 0..m {
                let one = bits(&x.narrow(0, r, 1)?.contiguous()?.matmul(&rhs)?)?;
                assert_eq!(
                    bits(&y.narrow(0, r, 1)?)?,
                    one,
                    "[{n}, {k}] m={m} row={r}: the row axis is not the one-row GEMV's bits"
                );
                checked += 1;
                if bits(&gemm.narrow(0, r, 1)?)? != one {
                    gemm_differing += 1;
                }
            }
        }
    }
    println!(
        "F16_ROWS_DECODE_CEILING16 rows_checked={checked} bit_equal={checked} \
         gemm_rows_differing_from_one_row={gemm_differing}/{checked}"
    );
    assert!(
        gemm_differing > 0,
        "no GEMM row differs from its one-row bits: the comparison cannot tell the routes apart"
    );
    Ok(())
}

/// The decode switch reaches only a decode step: outside the scope, with
/// the row axis itself off, past the ceiling and at one row, nothing takes
/// it and the bits are the switch-off bits; a verify window opened inside
/// a decode step is the verify window's (counted there, not here).
#[test]
fn the_decode_switch_reaches_a_decode_step_only() -> Result<()> {
    let dev = device(Contract::Fast)?;
    let (n, k) = (512usize, 1024usize);
    let rhs = weight(n, k, &dev)?.t()?;
    let x = rows(8, k, 0.5, &dev)?;
    let x11 = rows(11, k, 0.75, &dev)?;
    let x1 = rows(1, k, 1.0, &dev)?;
    type Case<'a> = (&'a str, Option<usize>, Box<dyn Fn() -> Result<Tensor> + 'a>);
    let cases: Vec<Case> = vec![
        (
            "outside a decode step",
            Some(10),
            Box::new(|| x.matmul(&rhs)),
        ),
        (
            "the row axis off",
            None,
            Box::new(|| {
                let _s = DecodeRows::open();
                x.matmul(&rhs)
            }),
        ),
        (
            "eleven rows (past the ceiling)",
            Some(10),
            Box::new(|| {
                let _s = DecodeRows::open();
                x11.matmul(&rhs)
            }),
        ),
        (
            "one row",
            Some(10),
            Box::new(|| {
                let _s = DecodeRows::open();
                x1.matmul(&rhs)
            }),
        ),
    ];
    for (what, rows_max, run) in &cases {
        set_f16_rows(*rows_max);
        set_f16_rows_decode(false);
        let off = run()?;
        set_f16_rows_decode(true);
        let before = f16_rows_decode_launches();
        let on = run()?;
        assert_eq!(
            f16_rows_decode_launches(),
            before,
            "{what}: the decode switch reached past its scope"
        );
        assert_eq!(bits(&on)?, bits(&off)?, "{what}: bits moved");
    }
    // A verify window inside a decode step is the window's.
    set_f16_rows(Some(10));
    set_f16_rows_decode(true);
    let (decode, verify) = (f16_rows_decode_launches(), f16_rows_launches());
    {
        let _s = DecodeRows::open();
        let _w = VerifyWindow::open();
        x.matmul(&rhs)?;
    }
    assert_eq!(
        f16_rows_decode_launches(),
        decode,
        "counted as a decode launch"
    );
    assert_eq!(
        f16_rows_launches() - verify,
        1,
        "the verify window's launch"
    );
    // The positive control: the same rows in a decode step do take it.
    {
        let _s = DecodeRows::open();
        x.matmul(&rhs)?;
    }
    assert_eq!(f16_rows_decode_launches() - decode, 1, "the control");
    assert!(
        !lloom_tensor::tuning::in_decode_rows(),
        "the scope outlived its guard"
    );
    println!("F16_ROWS_DECODE_SCOPE cases={} control=1", cases.len());
    Ok(())
}

/// A step that carries prompt pieces puts its live rows on the decode
/// route: inside `RaggedDecodeRows` with both switches on, a half matmul
/// over the step's input rows or its head's rows takes the row axis for its
/// first `live` rows — each the one-row GEMV's bits, from one live row to
/// the ceiling — and the GEMM for the pieces after them, each piece row the
/// bits the whole matmul's GEMM gives it; one decode launch counted. With
/// the decode switch off the scope moves no bit, and some live row of the
/// GEMM differs from its one-row bits, so the comparison sees the route.
#[test]
fn a_ragged_steps_live_rows_take_the_decode_route_and_its_pieces_keep_the_gemm() -> Result<()> {
    let dev = device(Contract::Fast)?;
    set_f16_rows(Some(10));
    let (mut live_rows, mut piece_rows, mut gemm_differing) = (0usize, 0usize, 0usize);
    for (n, k) in [
        (35usize, 512usize),
        (2048, 2048),
        (6144, 2048),
        (2048, 10752),
    ] {
        let rhs = weight(n, k, &dev)?.t()?;
        for live in [1usize, 2, 3, 4, 7, 10] {
            for piece in [1usize, 5, 111] {
                let m = live + piece;
                let x = rows(m, k, (live * 7 + piece) as f32 * 0.1, &dev)?;
                // The input's rows, then a head of one row a sequence.
                for (scope_rows, heads) in [(m, live + 1), (live + 2 * piece, m)] {
                    set_f16_rows_decode(false);
                    let whole = x.matmul(&rhs)?;
                    let off = {
                        let _s = RaggedDecodeRows::open(live, scope_rows, heads);
                        x.matmul(&rhs)?
                    };
                    assert_eq!(
                        bits(&off)?,
                        bits(&whole)?,
                        "[{n}, {k}] live={live} piece={piece}: the switch off, the scope moved bits"
                    );
                    set_f16_rows_decode(true);
                    let before = f16_rows_decode_launches();
                    let y = {
                        let _s = RaggedDecodeRows::open(live, scope_rows, heads);
                        x.matmul(&rhs)?
                    };
                    assert_eq!(
                        f16_rows_decode_launches() - before,
                        1,
                        "[{n}, {k}] live={live} piece={piece}: one decode launch"
                    );
                    for r in 0..live {
                        let one = bits(&x.narrow(0, r, 1)?.contiguous()?.matmul(&rhs)?)?;
                        assert_eq!(
                            bits(&y.narrow(0, r, 1)?)?,
                            one,
                            "[{n}, {k}] live={live} piece={piece} row={r}: a live row is not the \
                             one-row GEMV's bits"
                        );
                        live_rows += 1;
                        if bits(&whole.narrow(0, r, 1)?)? != one {
                            gemm_differing += 1;
                        }
                    }
                    for r in live..m {
                        assert_eq!(
                            bits(&y.narrow(0, r, 1)?)?,
                            bits(&whole.narrow(0, r, 1)?)?,
                            "[{n}, {k}] live={live} piece={piece} row={r}: a piece row is not \
                             the whole GEMM's bits"
                        );
                        piece_rows += 1;
                    }
                }
            }
        }
    }
    println!(
        "F16_ROWS_DECODE_RAGGED live_rows={live_rows} one_row_bit_equal={live_rows} \
         piece_rows={piece_rows} gemm_bit_equal={piece_rows} \
         gemm_live_rows_differing_from_one_row={gemm_differing}/{live_rows}"
    );
    assert!(
        gemm_differing > 0,
        "no GEMM live row differs from its one-row bits: the comparison cannot tell the routes apart"
    );
    Ok(())
}

/// The ragged split reaches only the step's own rows: a matmul of another
/// row count inside the scope, more live rows than the ceiling, the row axis
/// off, a verify window inside the scope, and the scope closed all keep the
/// switch-off bits and count nothing; the same matmul over the step's rows
/// is split (the positive control).
#[test]
fn the_ragged_split_reaches_the_steps_rows_only() -> Result<()> {
    let dev = device(Contract::Fast)?;
    let (n, k) = (512usize, 1024usize);
    let rhs = weight(n, k, &dev)?.t()?;
    let x = rows(9, k, 0.5, &dev)?;
    let x14 = rows(14, k, 0.25, &dev)?;
    type Case<'a> = (&'a str, Option<usize>, Box<dyn Fn() -> Result<Tensor> + 'a>);
    let cases: Vec<Case> = vec![
        (
            "another row count",
            Some(10),
            Box::new(|| {
                let _s = RaggedDecodeRows::open(3, 12, 5);
                x.matmul(&rhs)
            }),
        ),
        (
            "eleven live rows (past the ceiling)",
            Some(10),
            Box::new(|| {
                let _s = RaggedDecodeRows::open(11, 14, 13);
                x14.matmul(&rhs)
            }),
        ),
        (
            "the row axis off",
            None,
            Box::new(|| {
                let _s = RaggedDecodeRows::open(3, 9, 5);
                x.matmul(&rhs)
            }),
        ),
        (
            "a verify window inside the scope",
            Some(10),
            Box::new(|| {
                let _s = RaggedDecodeRows::open(3, 9, 5);
                let _w = VerifyWindow::open();
                x.matmul(&rhs)
            }),
        ),
        (
            "the scope closed",
            Some(10),
            Box::new(|| {
                drop(RaggedDecodeRows::open(3, 9, 5));
                x.matmul(&rhs)
            }),
        ),
    ];
    for (what, rows_max, run) in &cases {
        set_f16_rows(*rows_max);
        set_f16_rows_decode(false);
        let off = run()?;
        set_f16_rows_decode(true);
        let before = f16_rows_decode_launches();
        let on = run()?;
        assert_eq!(
            f16_rows_decode_launches(),
            before,
            "{what}: the ragged split reached past the step's rows"
        );
        assert_eq!(bits(&on)?, bits(&off)?, "{what}: bits moved");
    }
    set_f16_rows(Some(10));
    set_f16_rows_decode(true);
    let before = f16_rows_decode_launches();
    {
        let _s = RaggedDecodeRows::open(3, 9, 5);
        x.matmul(&rhs)?;
    }
    assert_eq!(f16_rows_decode_launches() - before, 1, "the control");
    assert_eq!(
        lloom_tensor::tuning::ragged_decode_lead(9),
        None,
        "the scope outlived its guard"
    );
    println!(
        "F16_ROWS_DECODE_RAGGED_SCOPE cases={} control=1",
        cases.len()
    );
    Ok(())
}

/// `deterministic` never takes it: a decode step's half rows keep the
/// one-row reduction per row with both switches on.
#[test]
fn deterministic_decode_steps_never_take_the_row_axis() -> Result<()> {
    let dev = device(Contract::Deterministic)?;
    let rhs = weight(512, 1024, &dev)?.t()?;
    let x = rows(4, 1024, 0.5, &dev)?;
    set_f16_rows(Some(10));
    set_f16_rows_decode(false);
    let off = {
        let _s = DecodeRows::open();
        x.matmul(&rhs)?
    };
    set_f16_rows_decode(true);
    let before = f16_rows_decode_launches();
    let on = {
        let _s = DecodeRows::open();
        x.matmul(&rhs)?
    };
    assert_eq!(f16_rows_decode_launches(), before, "deterministic took it");
    assert_eq!(bits(&on)?, bits(&off)?);
    let ragged = {
        let _s = RaggedDecodeRows::open(2, 4, 3);
        x.matmul(&rhs)?
    };
    assert_eq!(
        f16_rows_decode_launches(),
        before,
        "deterministic split a ragged step"
    );
    assert_eq!(bits(&ragged)?, bits(&off)?);
    assert!(
        !Contract::Deterministic.report().contains("f16-rows-decode"),
        "{}",
        Contract::Deterministic.report()
    );
    Ok(())
}

/// The contract line names the arm while both switches are on under
/// `fast`, and not otherwise.
#[test]
fn the_contract_line_names_the_decode_arm() -> Result<()> {
    let _dev = device(Contract::Fast)?;
    set_f16_rows(Some(10));
    set_f16_rows_decode(false);
    assert!(!Contract::Fast.report().contains("f16-rows-decode"));
    set_f16_rows_decode(true);
    let on = Contract::Fast.report();
    assert!(on.contains("f16-rows,f16-rows-decode"), "{on}");
    set_f16_rows(None);
    let rows_off = Contract::Fast.report();
    assert!(!rows_off.contains("f16-rows"), "{rows_off}");
    println!("F16_ROWS_DECODE_RECEIPT on=[{on}] rows_off=[{rows_off}]");
    Ok(())
}
