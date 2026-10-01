//! The routed matmul's arm as the caller's decision (`MoeArm`,
//! `docs/expert-streaming.md` §20.15): the arm a caller hands in is the
//! kernel that runs, whatever the row count, and on either arm a row's
//! bits do not depend on the rows computed beside it. The second property
//! is what lets a tile split into groups compute each row as the unsplit
//! tile does, once every group runs the tile's arm.
//!
//! No test here sets the process-wide `LLOOM_MOE_MM_FROM` / block overrides:
//! the default entry point is checked against `MoeArm::for_rows`.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::{GgmlDType, MoeArm, QTensor};
use lloom_tensor::{Device, Tensor};

const E: usize = 8;
const N: usize = 128;
const K: usize = 256;
const TOPK: usize = 4;
const ROWS: usize = 70;

fn weight(dev: &Device, dtype: GgmlDType, seed: f32) -> QTensor {
    if matches!(dtype, GgmlDType::Q2_0 | GgmlDType::IQ4NL) {
        return packed(dev, dtype, seed);
    }
    let w = Tensor::from_vec(
        (0..E * N * K)
            .map(|i| ((i as f32) * 0.0173 + seed).sin() * 0.2)
            .collect::<Vec<f32>>(),
        (E, N, K),
        dev,
    )
    .unwrap();
    QTensor::quantize(&w, dtype).unwrap()
}

/// Q2_0 and IQ4_NL have no quantizer in the tree (their blocks are read,
/// not written): the packed blocks directly — an f16 scale and 16 bytes of
/// codes each (64 two-bit codes, 32 four-bit grid indices).
fn packed(dev: &Device, dtype: GgmlDType, seed: f32) -> QTensor {
    use lloom_tensor::quantized::QStorage;
    let blocks = E * N * K / dtype.block_size();
    let mut bytes = Vec::with_capacity(blocks * dtype.type_size());
    for b in 0..blocks {
        let d = 0.05 + 0.04 * ((b as f32) * 0.37 + seed).sin();
        bytes.extend(half::f16::from_f32(d).to_le_bytes());
        bytes.extend((0..16).map(|j| ((b * 131 + j * 29) % 256) as u8));
    }
    assert_eq!(
        bytes.len(),
        blocks * dtype.type_size(),
        "{dtype:?}: not 2 + 16 bytes a block"
    );
    let storage = QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, dtype).unwrap();
    QTensor::new(storage, (E, N, K)).unwrap()
}

fn input(dev: &Device, rows: usize, in_dim1: usize) -> Tensor {
    Tensor::from_vec(
        (0..rows * in_dim1 * K)
            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
            .collect::<Vec<f32>>(),
        (rows, in_dim1, K),
        dev,
    )
    .unwrap()
}

fn ids(dev: &Device, rows: usize) -> Tensor {
    Tensor::from_vec(
        (0..rows * TOPK)
            .map(|i| ((i * 5 + i / TOPK) % E) as u32)
            .collect::<Vec<u32>>(),
        (rows, TOPK),
        dev,
    )
    .unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The kinds the real checkpoint's `down` stacks carry (Q2_0, IQ4_NL), a
/// K-quant, a legacy block and the float stack.
const KINDS: [GgmlDType; 5] = [
    GgmlDType::F32,
    GgmlDType::Q8_0,
    GgmlDType::Q4K,
    GgmlDType::Q2_0,
    GgmlDType::IQ4NL,
];

/// `rows` of the tile, computed over the groups `cuts` splits it into, on
/// `arm`, concatenated.
fn grouped(q: &QTensor, x: &Tensor, ids: &Tensor, cuts: &[usize], arm: MoeArm) -> Vec<u32> {
    let rows = x.dim(0).unwrap();
    let mut out = Vec::new();
    let mut start = 0;
    for &end in cuts.iter().chain(std::iter::once(&rows)) {
        let xg = x.narrow(0, start, end - start).unwrap();
        let ig = ids.narrow(0, start, end - start).unwrap();
        out.extend(bits(&q.indexed_moe_forward_with(&xg, &ig, arm).unwrap()));
        start = end;
    }
    out
}

/// The arm handed in is the kernel that runs: the GEMM on 8 rows (under the
/// default threshold) and the matmul-vec on 70 (over it), each told apart
/// from the other arm; and the entry point without an arm is `for_rows`'.
#[test]
fn the_callers_arm_is_the_kernel_that_runs() {
    let Ok(dev) = Device::new_metal(0) else {
        return;
    };
    for dtype in KINDS {
        let q = weight(&dev, dtype, 0.0);
        for rows in [8usize, ROWS] {
            let (x, ids) = (input(&dev, rows, 1), ids(&dev, rows));
            let mv = bits(
                &q.indexed_moe_forward_with(&x, &ids, MoeArm::MatVec)
                    .unwrap(),
            );
            let mm = bits(&q.indexed_moe_forward_with(&x, &ids, MoeArm::Gemm).unwrap());
            assert_ne!(
                mv, mm,
                "{dtype:?} rows={rows}: the two arms agree bit for bit — did the GEMM run?"
            );
            let default = bits(&q.indexed_moe_forward(&x, &ids).unwrap());
            let want = match MoeArm::for_rows(rows) {
                MoeArm::MatVec => &mv,
                MoeArm::Gemm => &mm,
            };
            assert_eq!(
                &default, want,
                "{dtype:?} rows={rows}: the entry point without an arm is not for_rows' arm"
            );
        }
    }
}

/// On either arm a row's bits are the same whether it is computed with the
/// whole tile or in a group of it — groups of 1, 29 and 40 rows beside the
/// 70-row tile, so the GEMM's row block (16 columns for a few rows an
/// expert, 32 for many) differs between the tile and its groups — for both
/// activation layouts.
#[test]
fn a_row_does_not_depend_on_the_group_it_is_computed_in() {
    let Ok(dev) = Device::new_metal(0) else {
        return;
    };
    for dtype in KINDS {
        let q = weight(&dev, dtype, 0.0);
        for in_dim1 in [1usize, TOPK] {
            let (x, ids) = (input(&dev, ROWS, in_dim1), ids(&dev, ROWS));
            for arm in [MoeArm::MatVec, MoeArm::Gemm] {
                let whole = grouped(&q, &x, &ids, &[], arm);
                for cuts in [&[1usize, 30][..], &[29], &[40, 63]] {
                    assert_eq!(
                        grouped(&q, &x, &ids, cuts, arm),
                        whole,
                        "{dtype:?} in_dim1={in_dim1} {arm:?}: split at {cuts:?} moved a row"
                    );
                }
            }
        }
    }
}

/// The gate|up pair: one launch, the matmul-vec, whatever the arm (the GEMM
/// has no paired form) — so its choice is the weights', never the rows'; and
/// where it falls back to two launches (the kinds differ), each is the
/// caller's arm.
#[test]
fn the_pair_is_the_matvec_and_its_fallback_takes_the_arm() {
    let Ok(dev) = Device::new_metal(0) else {
        return;
    };
    let (x, ids) = (input(&dev, ROWS, 1), ids(&dev, ROWS));
    let g = weight(&dev, GgmlDType::Q4K, 0.0);
    let u = weight(&dev, GgmlDType::Q4K, 1.0);
    for arm in [MoeArm::MatVec, MoeArm::Gemm] {
        let (pg, pu) = g.indexed_moe_forward_pair_with(&u, &x, &ids, arm).unwrap();
        assert_eq!(
            (bits(&pg), bits(&pu)),
            (
                bits(
                    &g.indexed_moe_forward_with(&x, &ids, MoeArm::MatVec)
                        .unwrap()
                ),
                bits(
                    &u.indexed_moe_forward_with(&x, &ids, MoeArm::MatVec)
                        .unwrap()
                )
            ),
            "{arm:?}: the one-launch pair is not the matmul-vec"
        );
    }
    let u8 = weight(&dev, GgmlDType::Q8_0, 1.0);
    for arm in [MoeArm::MatVec, MoeArm::Gemm] {
        let (pg, pu) = g.indexed_moe_forward_pair_with(&u8, &x, &ids, arm).unwrap();
        assert_eq!(
            (bits(&pg), bits(&pu)),
            (
                bits(&g.indexed_moe_forward_with(&x, &ids, arm).unwrap()),
                bits(&u8.indexed_moe_forward_with(&x, &ids, arm).unwrap())
            ),
            "{arm:?}: the two-launch pair did not take the caller's arm"
        );
    }
}
