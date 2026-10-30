//! A ragged FlashDecoding batch over a page pool computes each row's
//! attention from that row's own keys alone, at the partial width `serve`
//! takes for LFM2 (`kernels.attention.flash.partial_block = 8`).
//!
//! The single-chunk pass reduces a chunk at 32 simdgroups and the partial
//! pass at the configured width, so the two passes give a one-chunk row
//! different bits at width 8. A batch whose longest row spans two chunks used
//! to put every row on the partial pass, so a row of 256 keys read different
//! bits beside a row of 257 than alone, and a served batch's streams parted
//! run to run with the step at which their rows crossed a chunk boundary.
//! The reference is each row through the same op alone (`CONTRIBUTING.md`
//! §3.1b: an absolute reference, not a shape); the op is the one the model's
//! pool rows call, so the decision at its call site is what is tested.
#![cfg(feature = "metal")]

use half::f16;
use lloom::nn::ops::{flash_decoding_paged_ragged, FlashScratchSizing};
use lloom_tensor::{DType, Device, Result, Tensor};

/// The partial width serve takes for LFM2, set before anything reads the
/// tuning (the op reads it once per process).
fn served_partial_width() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::env::set_var("LLOOM_FLASH_PARTIAL_BN", "8"));
    assert_eq!(
        lloom::tuning::tuning().lfm2.flash_partial_bn,
        8,
        "the partial width was read before it was set"
    );
}

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

const H: usize = 32;
const H_KV: usize = 8;
const D: usize = 64;
const PAGE: usize = 256;
const POOL_PAGES: usize = 6;

/// `rows[b] = (live keys, table)` against one pool: the batch's output and
/// each row's output alone, as f16 bits.
fn together_and_alone(
    device: &Device,
    rows: &[(usize, Vec<u32>)],
) -> Result<(Vec<Vec<u16>>, Vec<Vec<u16>>)> {
    let scale = 1.0 / (D as f32).sqrt();
    let stride = rows.iter().map(|r| r.1.len()).max().unwrap_or(1);
    let pool = |seed| -> Result<Tensor> {
        Tensor::from_vec(
            fill(H_KV * POOL_PAGES * PAGE * D, seed),
            (1, H_KV, POOL_PAGES * PAGE, D),
            device,
        )?
        .to_dtype(DType::F16)
    };
    let (k, v) = (pool(11)?, pool(13)?);
    let q = Tensor::from_vec(fill(rows.len() * H * D, 7), (rows.len(), H, 1, D), device)?
        .to_dtype(DType::F16)?;
    let run = |q: &Tensor, rows: &[(usize, Vec<u32>)]| -> Result<Vec<Vec<u16>>> {
        let mut table = Vec::new();
        for (_, t) in rows {
            let mut t = t.clone();
            t.resize(stride, t[0]);
            table.extend(t);
        }
        let n_keys: Vec<u32> = rows.iter().map(|r| r.0 as u32).collect();
        let longest = rows.iter().map(|r| r.0).max().unwrap_or(1);
        let out = flash_decoding_paged_ragged(
            q,
            &k,
            &v,
            scale,
            1.0,
            PAGE,
            1,
            FlashScratchSizing::Grow,
            longest.div_ceil(PAGE),
            table,
            stride,
            n_keys,
        )?;
        let bits: Vec<u16> = out
            .flatten_all()?
            .to_vec1::<f16>()?
            .iter()
            .map(|x| x.to_bits())
            .collect();
        Ok(bits.chunks(H * D).map(<[u16]>::to_vec).collect())
    };
    let together = run(&q, rows)?;
    let mut alone = Vec::new();
    for (b, row) in rows.iter().enumerate() {
        alone.push(run(&q.narrow(0, b, 1)?, std::slice::from_ref(row))?.remove(0));
    }
    Ok((together, alone))
}

/// A row of one chunk beside a row of two, and the controls on either side
/// of the boundary: every row's bits are its own.
#[test]
fn every_row_of_a_ragged_flash_batch_is_its_own_at_the_served_width() -> Result<()> {
    served_partial_width();
    let device = Device::new_metal(0)?;
    let cases: [Vec<(usize, Vec<u32>)>; 4] = [
        vec![(256, vec![1]), (257, vec![3, 0])],
        vec![(100, vec![2]), (300, vec![4, 5]), (256, vec![1])],
        vec![(255, vec![1]), (256, vec![3])],
        vec![(257, vec![1, 2]), (300, vec![3, 0])],
    ];
    let mut differing = Vec::new();
    for case in &cases {
        let (together, alone) = together_and_alone(&device, case)?;
        for (b, (t, a)) in together.iter().zip(&alone).enumerate() {
            let n = t.iter().zip(a).filter(|(x, y)| x != y).count();
            let keys: Vec<usize> = case.iter().map(|r| r.0).collect();
            eprintln!("FLASH_PAGED_ROW keys={keys:?} row={b} differing={n}/{}", a.len());
            assert!(a.iter().any(|&x| x != 0), "row {b}'s reference is all zero");
            if n > 0 {
                differing.push((keys, b, n));
            }
        }
    }
    assert!(
        differing.is_empty(),
        "a row's attention moved with the rows beside it at the served partial width: \
         {differing:?} (the batch's live keys, the row, elements differing)"
    );
    Ok(())
}
