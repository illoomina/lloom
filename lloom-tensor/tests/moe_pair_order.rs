//! The routed gate|up pair's grid order (`PairOrder`, `LLOOM_MOE_PAIR_ORDER`;
//! `docs/expert-streaming.md` §20.15e) as the device records it: each order
//! dispatches its own grid, the split order the pair's kernel once a half,
//! every order the same bits, and the order a process runs is the one its
//! environment names. The bits alone cannot tell the orders apart — that is
//! the point of them — so without the grids an order that silently ran
//! another would pass every parity test (`moe_routed_metal.rs` holds the bits
//! at every kind).
//!
//! This binary runs with `LLOOM_MOE_PAIR_ORDER=tile` and the dispatch trace
//! on, both read once per process, so every test sets them before its first
//! Metal call.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::{GgmlDType, MoeArm, PairOrder, QTensor};
use lloom_tensor::{DType, Device, Tensor};

const E: usize = 8;
const N: usize = 128;
const K: usize = 256;
const TOPK: usize = 4;
const ROWS: usize = 2;

fn device() -> Option<Device> {
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    std::env::set_var("LLOOM_MOE_PAIR_ORDER", "tile");
    Device::new_metal(0).ok()
}

fn weight(dev: &Device, dtype: GgmlDType, seed: f32) -> QTensor {
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

fn inputs(dev: &Device, act: DType) -> (Tensor, Tensor) {
    let x = Tensor::from_vec(
        (0..ROWS * K)
            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
            .collect::<Vec<f32>>(),
        (ROWS, 1, K),
        dev,
    )
    .unwrap()
    .to_dtype(act)
    .unwrap();
    let ids = Tensor::from_vec(vec![3u32, 0, 7, 3, 5, 1, 6, 2], (ROWS, TOPK), dev).unwrap();
    (x, ids)
}

fn bits(t: &Tensor) -> Vec<u32> {
    let t = t.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    t.to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The `(pipeline, grid)` of every dispatch encoded on `dev` while `f` runs.
fn grids(dev: &Device, f: impl FnOnce()) -> Vec<(String, (usize, usize, usize))> {
    dev.synchronize().unwrap();
    let window = dev
        .as_metal_device()
        .unwrap()
        .dispatch_trace()
        .open()
        .expect("a trace window on this test's device");
    f();
    dev.synchronize().unwrap();
    window
        .close()
        .into_iter()
        .map(|d| (d.pipeline, d.grid))
        .collect()
}

/// Each order's dispatches, against the incumbent's absolute grid — Q8_0 at
/// `n = 128` is 16 threadgroups of 8 rows, over `2 · rows · topk` planes —
/// and each order's halves equal to the incumbent's bit for bit.
#[test]
fn each_order_dispatches_its_own_grid() {
    let Some(dev) = device() else {
        return;
    };
    let planes = ROWS * TOPK;
    for act in [DType::F32, DType::F16] {
        let (g, u) = (
            weight(&dev, GgmlDType::Q8_0, 0.0),
            weight(&dev, GgmlDType::Q8_0, 1.0),
        );
        let (x, ids) = inputs(&dev, act);
        let mut halves = Vec::new();
        for order in PairOrder::ALL {
            let mut got = None;
            let seen = grids(&dev, || {
                got = Some(
                    g.indexed_moe_forward_pair_in(&u, &x, &ids, MoeArm::MatVec, act, order)
                        .unwrap(),
                );
            });
            let (pg, pu) = got.unwrap();
            println!("ORDER {act:?} {order:?}: {seen:?}");
            let suffix = if act == DType::F16 {
                "_f16_f16"
            } else {
                "_f32"
            };
            let name = format!("kernel_mul_mv_id2_q8_0{suffix}");
            let tiles = 16;
            let want: Vec<(String, (usize, usize, usize))> = match order {
                PairOrder::Half => vec![(name, (tiles, 1, 2 * planes))],
                PairOrder::Expert => vec![(name, (tiles, 2, planes))],
                PairOrder::Tile => vec![(name, (2 * tiles, 1, planes))],
                PairOrder::Split => vec![
                    (name.clone(), (tiles, 1, planes)),
                    (name, (tiles, 1, planes)),
                ],
            };
            assert_eq!(seen, want, "{act:?} {order:?}: not this order's dispatches");
            halves.push((bits(&pg), bits(&pu)));
        }
        for (order, h) in PairOrder::ALL.iter().zip(&halves) {
            assert_eq!(h, &halves[0], "{act:?} {order:?}: not the incumbent's bits");
        }
    }
}

/// The order this process runs is the one its environment names: the entry
/// point without an order dispatches the tile order's grid, and the parser
/// names every order it accepts back.
#[test]
fn the_configured_order_is_the_environments() {
    let Some(dev) = device() else {
        return;
    };
    assert_eq!(PairOrder::configured(), PairOrder::Tile);
    for order in PairOrder::ALL {
        assert_eq!(PairOrder::parse(order.name()), Ok(order));
    }
    assert!(PairOrder::parse("diagonal").is_err());
    let (g, u) = (
        weight(&dev, GgmlDType::Q8_0, 0.0),
        weight(&dev, GgmlDType::Q8_0, 1.0),
    );
    let (x, ids) = inputs(&dev, DType::F16);
    let configured = grids(&dev, || {
        g.indexed_moe_forward_pair_with(&u, &x, &ids, MoeArm::MatVec)
            .unwrap();
    });
    let tile = grids(&dev, || {
        g.indexed_moe_forward_pair_in(&u, &x, &ids, MoeArm::MatVec, DType::F16, PairOrder::Tile)
            .unwrap();
    });
    assert_eq!(
        configured, tile,
        "the configured entry point did not take LLOOM_MOE_PAIR_ORDER=tile"
    );
    assert_eq!(configured.len(), 1);
}

/// The f16 arm over an f32 activation (the cast folded into the launch): the
/// pair is one `_f32h_f16` launch and nothing else — no cast — where the
/// cast-then-launch path is a `cast_f32_f16` and the `_f16_f16` pair.
#[test]
fn the_folded_pair_is_one_launch_and_no_cast() {
    let Some(dev) = device() else {
        return;
    };
    let (g, u) = (
        weight(&dev, GgmlDType::Q8_0, 0.0),
        weight(&dev, GgmlDType::Q8_0, 1.0),
    );
    let (x, ids) = inputs(&dev, DType::F32);
    let order = PairOrder::Half;
    let folded = grids(&dev, || {
        g.indexed_moe_forward_pair_in(&u, &x, &ids, MoeArm::MatVec, DType::F16, order)
            .unwrap();
    });
    let cast = grids(&dev, || {
        let x16 = x.to_dtype(DType::F16).unwrap();
        g.indexed_moe_forward_pair_in(&u, &x16, &ids, MoeArm::MatVec, DType::F16, order)
            .unwrap();
    });
    println!("FOLD folded={folded:?}\nFOLD cast={cast:?}");
    let names = |v: &[(String, (usize, usize, usize))]| {
        v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
    };
    assert_eq!(
        names(&folded),
        vec!["kernel_mul_mv_id2_q8_0_f32h_f16".to_string()]
    );
    assert_eq!(
        names(&cast),
        vec![
            "cast_f32_f16".to_string(),
            "kernel_mul_mv_id2_q8_0_f16_f16".to_string()
        ]
    );
    assert_eq!(
        folded[0].1, cast[1].1,
        "the folded launch's grid is the f16 pair's"
    );
}
