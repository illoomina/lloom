//! The Metal projection route over packed weights (`quantized::route`):
//! every cell of the table below resolves to the route written beside it, or
//! is refused naming its cell. The table is the reference, written out, not
//! derived here; a change to the resolver that moves any cell fails it by
//! name.
#![cfg(feature = "metal")]
use lloom_tensor::kernels::{GemvTriple, GgmlDType};
use lloom_tensor::quantized::route::{resolve, Policy, Route, Spec, Weight};
use lloom_tensor::DType;

const SHIPPED: Policy = Policy {
    mv_rows_max: 8,
    dequant_from: 64,
    dequant_steel: false,
};

#[derive(Debug, PartialEq)]
enum Want {
    Mv(GemvTriple),
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
        (Ok(Route::Matvec(t)), Want::Mv(w)) => t == w,
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
    check(GgmlDType::Q4K, DType::F32, 1, true, 2, 2560, p, Mv(F));
    check(GgmlDType::Q4K, DType::F16, 1, true, 2, 2560, p, Mv(H));
    check(GgmlDType::Q4K, DType::F16, 512, true, 2, 2560, p, Mv(H));
    check(GgmlDType::Q8_0, DType::F16, 1, true, 2, 2048, p, Mv(H));
    check(GgmlDType::Q6K, DType::F32, 4, false, 2, 5120, p, Mv(F));
    // The matmul-vec's last row and the GEMM's first, at both triples.
    check(GgmlDType::Q4K, DType::F32, 8, false, 2, 2560, p, Mv(F));
    check(GgmlDType::Q4K, DType::F16, 8, false, 2, 2560, p, Mv(H));
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
        Mv(F),
    );
}

/// Every kind at every activation dtype and rows band resolves or is refused;
/// none panics. And the f16 cells follow the kind table: the decode cell its
/// `f16_arm`, the GEMM cell `mul_mm_f16_arm` (the table cells above pin both
/// on Q4_K, Q8_0, F16, F32 and BF16).
#[test]
fn every_kind_resolves_or_refuses() {
    for kind in GgmlDType::ALL {
        for act in [DType::F32, DType::F16, DType::BF16] {
            for (rows, rpb) in [(1, true), (4, false), (9, false), (512, false), (512, true)] {
                let _ = resolve(
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
