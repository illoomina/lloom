//! The routed matmul's arm as the caller's decision (`MoeArm`,
//! `docs/expert-streaming.md` §20.15): the arm a caller hands in is the
//! kernel that runs, whatever the row count, and on either arm a row's
//! bits do not depend on the rows computed beside it. The second property
//! is what lets a tile split into groups compute each row as the unsplit
//! tile does, once every group runs the tile's arm.
//!
//! No test here sets the process-wide `LLOOM_MOE_MM_FROM` / block overrides:
//! the default entry point is checked against `MoeArm::for_rows`. The
//! dispatch census is on in this binary (`device`), so the gate|up pair's
//! test can read which kernels ran.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::{GgmlDType, MoeArm, QTensor};
use lloom_tensor::{Device, Tensor};

const E: usize = 8;
const N: usize = 128;
const K: usize = 256;
const TOPK: usize = 4;
const ROWS: usize = 70;

/// The Metal device, with the dispatch census on. The profiler reads its
/// switch once per process, so every test sets it before its first Metal
/// call; counting changes no bits.
fn device() -> Option<Device> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    Device::new_metal(0).ok()
}

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
    let Some(dev) = device() else {
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
    let Some(dev) = device() else {
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

/// Each dispatch encoded on `dev` while `f` runs, counted by kernel label
/// (the device's own census window, so the other tests' devices do not
/// count).
fn launches(dev: &Device, f: impl FnOnce()) -> Vec<(String, u64)> {
    dev.synchronize().unwrap();
    let window = dev
        .as_metal_device()
        .unwrap()
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    f();
    dev.synchronize().unwrap();
    window.close()
}

fn count(census: &[(String, u64)], label: &str) -> u64 {
    census
        .iter()
        .filter(|(l, _)| l.contains(label))
        .map(|(_, n)| *n)
        .sum()
}

/// The gate|up pair takes the caller's arm. On the matmul-vec it is ONE
/// launch (`kernel_mul_mv_id2`) whose halves are the two matmul-vec calls
/// bit for bit; on the GEMM it is two GEMM calls — the GEMM has no paired
/// form, so a tile never leaves it for the pair — and the two arms' bits
/// differ. The entry point without an arm is `for_rows`' arm (8 rows and
/// 70); where the kinds differ, two calls on either arm.
#[test]
fn the_pair_takes_the_callers_arm() {
    let Some(dev) = device() else {
        return;
    };
    assert!(
        lloom_tensor::kernels::metal::profile::enabled(),
        "the census decides which kernels ran"
    );
    let g = weight(&dev, GgmlDType::Q4K, 0.0);
    let u = weight(&dev, GgmlDType::Q4K, 1.0);
    for in_dim1 in [1usize, TOPK] {
        let (x, ids) = (input(&dev, ROWS, in_dim1), ids(&dev, ROWS));
        let mut pairs = Vec::new();
        for arm in [MoeArm::MatVec, MoeArm::Gemm] {
            let mut got = None;
            let census = launches(&dev, || {
                let (pg, pu) = g.indexed_moe_forward_pair_with(&u, &x, &ids, arm).unwrap();
                got = Some((bits(&pg), bits(&pu)));
            });
            let got = got.unwrap();
            let want = (
                bits(&g.indexed_moe_forward_with(&x, &ids, arm).unwrap()),
                bits(&u.indexed_moe_forward_with(&x, &ids, arm).unwrap()),
            );
            assert_eq!(
                got, want,
                "in_dim1={in_dim1} {arm:?}: the pair's halves are not the two calls on its arm"
            );
            let (pair, mv, mm) = (
                count(&census, "mul_mv_id2_"),
                count(&census, "mul_mv_id_"),
                count(&census, "mul_mm_id"),
            );
            println!(
                "PAIR in_dim1={in_dim1} {arm:?}: mul_mv_id2={pair} mul_mv_id={mv} mul_mm_id={mm}"
            );
            match arm {
                MoeArm::MatVec => assert_eq!(
                    (pair, mv, mm),
                    (1, 0, 0),
                    "in_dim1={in_dim1}: the matmul-vec pair is not one paired launch ({census:?})"
                ),
                MoeArm::Gemm => assert_eq!(
                    (pair, mv, mm),
                    (0, 0, 2),
                    "in_dim1={in_dim1}: the GEMM pair is not two GEMM launches ({census:?})"
                ),
            }
            pairs.push(got);
        }
        assert_ne!(
            pairs[0], pairs[1],
            "in_dim1={in_dim1}: the pair's two arms agree bit for bit — did the GEMM run?"
        );
        for rows in [8usize, ROWS] {
            let (xr, ir) = (input(&dev, rows, in_dim1), self::ids(&dev, rows));
            let (dg, du) = g.indexed_moe_forward_pair(&u, &xr, &ir).unwrap();
            let (wg, wu) = g
                .indexed_moe_forward_pair_with(&u, &xr, &ir, MoeArm::for_rows(rows))
                .unwrap();
            assert_eq!(
                (bits(&dg), bits(&du)),
                (bits(&wg), bits(&wu)),
                "in_dim1={in_dim1} rows={rows}: the pair without an arm is not for_rows' arm"
            );
        }
    }
    let (x, ids) = (input(&dev, ROWS, 1), ids(&dev, ROWS));
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
