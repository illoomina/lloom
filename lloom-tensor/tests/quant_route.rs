//! The Metal projection route over packed weights (`quantized::route`):
//! every cell of the table below resolves to the route written beside it, or
//! is refused naming its cell. The table is the reference, written out, not
//! derived here; a change to the resolver that moves any cell fails it by
//! name.
#![cfg(feature = "metal")]
use lloom_tensor::kernels::{GemvTriple, GgmlDType};
use lloom_tensor::quantized::route::{resolve, Policy, Route, RowsPolicy, Spec, Weight};
use lloom_tensor::DType;

const SHIPPED: Policy = Policy {
    mv_rows_max: 8,
    dequant_from: 64,
    dequant_steel: false,
    rows: RowsPolicy::Fixed {
        axis: true,
        rows_max: None,
    },
};

#[derive(Debug, PartialEq)]
enum Want {
    /// The matmul-vec at this triple and row capacity.
    Mv(GemvTriple, usize),
    Gemm(GemvTriple),
    DequantGemm(GemvTriple),
    DequantSteel,
    /// Refused, the message carrying this text.
    No(&'static str),
}

fn check(
    kind: GgmlDType,
    act: DType,
    rows: usize,
    row_per_batch: bool,
    rank: usize,
    k: usize,
    p: Policy,
    want: Want,
) {
    let got = resolve(
        Weight { kind, rank, k },
        Spec {
            act,
            rows,
            row_per_batch,
        },
        p,
    );
    let ok = match (&got, &want) {
        (Ok(Route::Matvec { triple, rows }), Want::Mv(w, r)) => triple == w && rows == r,
        (Ok(Route::Gemm(t)), Want::Gemm(w)) => t == w,
        (Ok(Route::DequantOnceGemm(t)), Want::DequantGemm(w)) => t == w,
        (Ok(Route::DequantOnceSteel), Want::DequantSteel) => true,
        (Err(r), Want::No(text)) => r.to_string().contains(text),
        _ => false,
    };
    assert!(
        ok,
        "{kind:?} × {act:?} × {rows} rows (row per batch {row_per_batch}, rank {rank}, k {k}) under {p:?}: got {got:?}, want {want:?}"
    );
}

#[test]
fn every_cell_resolves_as_tabled() {
    use GemvTriple::{F16F32F16 as H, F32F32F32 as F};
    use Want::*;
    let p = SHIPPED;
    // The decode step: one row a batch takes the matmul-vec at any batch.
    check(GgmlDType::Q4K, DType::F32, 1, true, 2, 2560, p, Mv(F, 1));
    check(GgmlDType::Q4K, DType::F16, 1, true, 2, 2560, p, Mv(H, 1));
    check(GgmlDType::Q4K, DType::F16, 512, true, 2, 2560, p, Mv(H, 1));
    check(GgmlDType::Q8_0, DType::F16, 1, true, 2, 2048, p, Mv(H, 1));
    check(GgmlDType::Q6K, DType::F32, 4, false, 2, 5120, p, Mv(F, 1));
    // The matmul-vec's last row and the GEMM's first, at both triples.
    check(GgmlDType::Q4K, DType::F32, 8, false, 2, 2560, p, Mv(F, 1));
    check(GgmlDType::Q4K, DType::F16, 8, false, 2, 2560, p, Mv(H, 1));
    check(GgmlDType::Q4K, DType::F32, 9, false, 2, 2560, p, Gemm(F));
    check(GgmlDType::Q4K, DType::F16, 9, false, 2, 2560, p, Gemm(H));
    check(GgmlDType::Q4K, DType::F32, 63, false, 2, 2560, p, Gemm(F));
    check(GgmlDType::Q6K, DType::F16, 63, false, 2, 5120, p, Gemm(H));
    // Dequantise-once from 64 rows, a rank-2 weight, k a multiple of 16; the
    // scratch's GEMM at the request's triple.
    check(
        GgmlDType::Q4K,
        DType::F32,
        64,
        false,
        2,
        2560,
        p,
        DequantGemm(F),
    );
    check(
        GgmlDType::Q4K,
        DType::F16,
        64,
        false,
        2,
        2560,
        p,
        DequantGemm(H),
    );
    check(
        GgmlDType::Q8_0,
        DType::F32,
        512,
        false,
        2,
        2048,
        p,
        DequantGemm(F),
    );
    check(
        GgmlDType::Q8_0,
        DType::F16,
        512,
        false,
        2,
        2048,
        p,
        DequantGemm(H),
    );
    check(GgmlDType::Q4K, DType::F32, 64, false, 3, 2560, p, Gemm(F));
    check(GgmlDType::Q4_0, DType::F32, 64, false, 2, 2568, p, Gemm(F));
    check(GgmlDType::Q4_0, DType::F16, 64, false, 2, 2568, p, Gemm(H));
    // The float kinds have nothing to dequantise; the dense f16 weight has
    // the GEMM's f16 arm, the f32 weight has none.
    check(GgmlDType::F16, DType::F32, 64, false, 2, 2560, p, Gemm(F));
    check(GgmlDType::F16, DType::F16, 64, false, 2, 2560, p, Gemm(H));
    check(GgmlDType::F32, DType::F32, 9, false, 2, 2560, p, Gemm(F));
    // Refused, by name.
    check(
        GgmlDType::Q8_1,
        DType::F32,
        9,
        false,
        2,
        2560,
        p,
        No("q8_1 weight × F32 activations × 9 rows; it needs a GEMM arm for this weight kind"),
    );
    check(
        GgmlDType::F32,
        DType::F16,
        9,
        false,
        2,
        2560,
        p,
        No("f32 weight × F16 activations × 9 rows; it needs a GEMM arm"),
    );
    check(
        GgmlDType::BF16,
        DType::F16,
        9,
        false,
        2,
        2560,
        p,
        No("bf16 weight × F16 activations × 9 rows; it needs a GEMM arm"),
    );
    check(
        GgmlDType::Q4K,
        DType::BF16,
        9,
        false,
        2,
        2560,
        p,
        No("q4_K weight × BF16 activations × 9 rows; it needs a GEMM arm"),
    );
    check(
        GgmlDType::Q4K,
        DType::BF16,
        1,
        true,
        2,
        2560,
        p,
        No("q4_K weight × BF16 activations × 1 rows; it needs a matmul-vec arm"),
    );
    check(
        GgmlDType::F16,
        DType::F16,
        1,
        true,
        2,
        2560,
        p,
        No("f16 weight × F16 activations × 1 rows; it needs a matmul-vec arm"),
    );
}

#[test]
fn the_policy_moves_the_cells_it_names() {
    use GemvTriple::{F16F32F16 as H, F32F32F32 as F};
    use Want::*;
    let steel = Policy {
        dequant_steel: true,
        ..SHIPPED
    };
    check(
        GgmlDType::Q4K,
        DType::F32,
        512,
        false,
        2,
        2560,
        steel,
        DequantSteel,
    );
    check(
        GgmlDType::Q4K,
        DType::F16,
        512,
        false,
        2,
        2560,
        steel,
        No("it needs the f32 activations the steel"),
    );
    check(
        GgmlDType::Q4K,
        DType::F16,
        63,
        false,
        2,
        2560,
        steel,
        Gemm(H),
    );
    let never = Policy {
        dequant_from: 0,
        ..SHIPPED
    };
    check(
        GgmlDType::Q4K,
        DType::F32,
        512,
        false,
        2,
        2560,
        never,
        Gemm(F),
    );
    check(
        GgmlDType::Q4K,
        DType::F16,
        512,
        false,
        2,
        2560,
        never,
        Gemm(H),
    );
    let mm_from_2 = Policy {
        mv_rows_max: 1,
        ..SHIPPED
    };
    check(
        GgmlDType::Q4K,
        DType::F32,
        2,
        false,
        2,
        2560,
        mm_from_2,
        Gemm(F),
    );
    check(
        GgmlDType::Q4K,
        DType::F32,
        2,
        true,
        2,
        2560,
        mm_from_2,
        Mv(F, 1),
    );
}

/// Every kind at every activation dtype and rows band resolves or is refused;
/// none panics. And the f16 cells follow the kind table: the decode cell its
/// `f16_arm`, the GEMM cell `mul_mm_f16_arm` (the table cells above pin both
/// on Q4_K, Q8_0, F16, F32 and BF16).
#[test]
fn kinds_without_a_matvec_and_bad_geometry_are_refused_by_name() {
    for kind in [GgmlDType::BF16, GgmlDType::Q8_1, GgmlDType::Q8K] {
        let r = resolve(
            Weight { kind, rank: 2, k: 4096 },
            Spec { act: DType::F32, rows: 1, row_per_batch: true },
            SHIPPED,
        );
        let e = r.expect_err("no matvec kernel: refused, not certified");
        assert_eq!(e.need, "a matmul-vec arm for this weight kind", "{kind:?}");
        assert!(e.to_string().contains("no route for"), "{e}");
    }
    let q = GgmlDType::Q4K;
    let rank3_mv = resolve(
        Weight { kind: q, rank: 3, k: 4096 },
        Spec { act: DType::F32, rows: 1, row_per_batch: true },
        SHIPPED,
    );
    assert_eq!(
        rank3_mv.unwrap_err().need,
        "a rank-2 weight (the matmul-vec takes one matrix)"
    );
    for (rows, k, rank) in [(0, 4096, 2), (1, 0, 2), (1, 4096, 1)] {
        let r = resolve(
            Weight { kind: q, rank, k },
            Spec { act: DType::F32, rows, row_per_batch: false },
            SHIPPED,
        );
        assert_eq!(
            r.unwrap_err().need,
            "a non-empty request over a weight of rank >= 2",
            "rows={rows} k={k} rank={rank}"
        );
    }
}

#[test]
fn every_kind_resolves_or_refuses() {
    for kind in GgmlDType::ALL {
        for act in [DType::F32, DType::F16, DType::BF16] {
            for (rows, rpb) in [(1, true), (4, false), (9, false), (512, false), (512, true)] {
                let got = resolve(
                    Weight {
                        kind,
                        rank: 2,
                        k: 4096,
                    },
                    Spec {
                        act,
                        rows,
                        row_per_batch: rpb,
                    },
                    SHIPPED,
                );
                // Every cell's outcome is the kind table's capability, not
                // merely "some route" (Codex review on PR1161).
                let mv = rpb || rows <= SHIPPED.mv_rows_max;
                let has_mv = !matches!(
                    kind.kind().mul_mv,
                    lloom_tensor::kernels::kernels::quantized::MulMv::None
                );
                let expect = if mv {
                    has_mv
                        && match act {
                            DType::F32 => true,
                            DType::F16 => kind.has_gemv_triple(),
                            _ => false,
                        }
                } else {
                    kind.kind().mul_mm.is_some()
                        && match act {
                            DType::F32 => true,
                            DType::F16 => kind.kind().mul_mm_f16_arm(),
                            _ => false,
                        }
                };
                assert_eq!(got.is_ok(), expect, "{kind:?} {act:?} rows={rows} rpb={rpb}: {got:?}");
                if mv && !has_mv {
                    let e = got.unwrap_err();
                    assert_eq!(e.need, "a matmul-vec arm for this weight kind", "{kind:?}");
                }
            }
        }
        let decode = resolve(
            Weight {
                kind,
                rank: 2,
                k: 4096,
            },
            Spec {
                act: DType::F16,
                rows: 1,
                row_per_batch: true,
            },
            SHIPPED,
        );
        assert_eq!(decode.is_ok(), kind.has_gemv_triple(), "{kind:?}");
        let gemm = resolve(
            Weight {
                kind,
                rank: 3,
                k: 4096,
            },
            Spec {
                act: DType::F16,
                rows: 9,
                row_per_batch: false,
            },
            SHIPPED,
        );
        assert_eq!(
            gemm.is_ok(),
            kind.kind().mul_mm_f16_arm() && kind.kind().mul_mm.is_some(),
            "{kind:?}"
        );
    }
}

/// The matmul-vec's row capacity, per cell: the kind table's measured
/// ceilings (`docs/tuning.md` §13.5w: Q2_K 8, Q8_0 2, the K-quants 1), the
/// smallest instantiated capacity (2, 4, 8) at or above the rows, at both
/// triples; one row a weight read with the axis off (`deterministic`), on a
/// dense kind, and under an override of 1; every kind's capacities under an
/// override of 8.
#[test]
fn the_row_capacity_resolves_as_tabled() {
    use GemvTriple::{F16F32F16 as H, F32F32F32 as F};
    use Want::*;
    let p = SHIPPED;
    check(GgmlDType::Q2K, DType::F32, 1, true, 2, 2560, p, Mv(F, 1));
    check(GgmlDType::Q2K, DType::F32, 2, false, 2, 2560, p, Mv(F, 2));
    check(GgmlDType::Q2K, DType::F32, 3, false, 2, 2560, p, Mv(F, 4));
    check(GgmlDType::Q2K, DType::F16, 4, false, 2, 2560, p, Mv(H, 4));
    check(GgmlDType::Q2K, DType::F16, 5, false, 2, 2560, p, Mv(H, 8));
    check(GgmlDType::Q2K, DType::F32, 8, false, 2, 2560, p, Mv(F, 8));
    check(GgmlDType::Q2K, DType::F32, 512, true, 2, 2560, p, Mv(F, 8));
    check(GgmlDType::Q8_0, DType::F32, 2, false, 2, 2048, p, Mv(F, 2));
    check(GgmlDType::Q8_0, DType::F16, 6, false, 2, 2048, p, Mv(H, 2));
    check(GgmlDType::Q4K, DType::F32, 2, true, 2, 2560, p, Mv(F, 1));
    check(GgmlDType::Q5K, DType::F16, 8, false, 2, 5120, p, Mv(H, 1));
    check(GgmlDType::IQ4XS, DType::F32, 4, false, 2, 4096, p, Mv(F, 1));
    let off = Policy {
        rows: RowsPolicy::Fixed {
            axis: false,
            rows_max: None,
        },
        ..SHIPPED
    };
    check(GgmlDType::Q2K, DType::F32, 4, false, 2, 2560, off, Mv(F, 1));
    check(
        GgmlDType::Q8_0,
        DType::F16,
        2,
        false,
        2,
        2048,
        off,
        Mv(H, 1),
    );
    let all8 = Policy {
        rows: RowsPolicy::Fixed {
            axis: true,
            rows_max: Some(8),
        },
        ..SHIPPED
    };
    check(
        GgmlDType::Q4K,
        DType::F32,
        2,
        false,
        2,
        2560,
        all8,
        Mv(F, 2),
    );
    check(
        GgmlDType::Q4K,
        DType::F32,
        3,
        false,
        2,
        2560,
        all8,
        Mv(F, 4),
    );
    check(
        GgmlDType::Q6K,
        DType::F16,
        6,
        false,
        2,
        5120,
        all8,
        Mv(H, 8),
    );
    check(
        GgmlDType::IQ3XXS,
        DType::F16,
        8,
        true,
        2,
        4096,
        all8,
        Mv(H, 8),
    );
    check(
        GgmlDType::F16,
        DType::F32,
        4,
        false,
        2,
        2560,
        all8,
        Mv(F, 1),
    );
    let one = Policy {
        rows: RowsPolicy::Fixed {
            axis: true,
            rows_max: Some(1),
        },
        ..SHIPPED
    };
    check(GgmlDType::Q2K, DType::F32, 8, false, 2, 2560, one, Mv(F, 1));
}

/// The process's policy resolves the capacity the dispatch takes:
/// [`RowsPolicy::Process`] agrees, cell by cell, with the same policy stated
/// from this process's switches, for every kind with a matmul-vec, rows
/// 1–9 and both triples. The dispatch outside the route
/// (`call_quantized_matmul_mv_t`) reads `mv_rows_capacity`, which this
/// pins to the route's answer.
#[test]
fn the_process_policy_is_the_dispatchs() {
    let process = Policy::current();
    assert_eq!(process.rows, RowsPolicy::Process);
    let stated = Policy {
        rows: RowsPolicy::Fixed {
            axis: lloom_tensor::kernels::mv_rows_axis_on(),
            rows_max: lloom_tensor::kernels::mv_rows_max_override(),
        },
        ..process
    };
    let mut cells = 0;
    let mut axis_cells = 0;
    for kind in GgmlDType::ALL {
        for act in [DType::F32, DType::F16] {
            for rows in 1..=9 {
                let w = Weight {
                    kind,
                    rank: 2,
                    k: 4096,
                };
                let s = Spec {
                    act,
                    rows,
                    row_per_batch: true,
                };
                let (a, b) = (resolve(w, s, process), resolve(w, s, stated));
                assert_eq!(a, b, "{kind:?} × {act:?} × {rows} rows");
                if let Ok(Route::Matvec { rows: cap, .. }) = a {
                    assert_eq!(
                        cap,
                        lloom_tensor::kernels::mv_rows_capacity(kind, rows, false),
                        "{kind:?} × {act:?} × {rows} rows"
                    );
                    cells += 1;
                    if cap > 1 {
                        axis_cells += 1;
                    }
                }
            }
        }
    }
    // Non-vacuity: the cells compared include matmul-vec cells, and on the
    // default tuning (`fast`, `qmv.rows` on) some of them take the axis.
    assert!(cells > 0);
    assert!(
        axis_cells > 0,
        "no cell took the row axis: the comparison never reached a capacity above 1"
    );
}
