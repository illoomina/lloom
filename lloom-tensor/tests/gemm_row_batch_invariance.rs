//! Whether a row of a dense f16 `x · Wᵀ` keeps its bits when the number of
//! rows beside it changes — the property a batched decode's row needs for its
//! logits not to depend on who else is in the batch (§2.3.3 #7), checked
//! against the tile the live `m` selects (`mlx_gemm::select_tile_config`:
//! the GEMV at `m = 1`, the 16-row tile below `skinny_max`, the 64-row tile
//! from it).
//!
//! Prints, for each `m`, whether row 0..2 match the `m = 2` result bit for
//! bit, at LFM2's decode shapes (hidden 2048 → 8192 and back), in the two
//! layouts a batched step takes (`[m, k]` and the rows-as-batch `[m, 1, k]`).
//! The assertion is the partition measured on this device: the GEMV differs
//! from the tiles, and every `m` on one tile agrees with every other.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};

fn rows_bits(t: &Tensor, rows: usize) -> Result<Vec<Vec<u32>>> {
    let t = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let n = t.dim(t.rank() - 1)?;
    let t = t.reshape(((), n))?;
    (0..rows)
        .map(|r| {
            Ok(t.get(r)?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        })
        .collect()
}

#[test]
fn a_rows_bits_by_the_tile_its_batch_selects() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let ms = [1usize, 2, 3, 8, 16, 31, 32, 33, 40, 64, 128];
    let mut verdicts = Vec::new();
    for (n, k) in [(8192usize, 2048usize), (2048, 8192)] {
        let w = Tensor::randn(0f32, 0.02, (n, k), &dev)?.to_dtype(DType::F16)?;
        let x = Tensor::randn(0f32, 1.0, (128, k), &dev)?.to_dtype(DType::F16)?;
        for rank3 in [false, true] {
            let mut by_m = Vec::new();
            for &m in &ms {
                let xm = x.narrow(0, 0, m)?;
                let y = if rank3 {
                    xm.reshape((m, 1, k))?.broadcast_matmul(&w.t()?)?
                } else {
                    xm.matmul(&w.t()?)?
                };
                by_m.push((m, rows_bits(&y, 1)?));
            }
            let reference = by_m.iter().find(|(m, _)| *m == 2).unwrap().1.clone();
            let line: Vec<String> = by_m
                .iter()
                .map(|(m, b)| format!("m={m}:{}", if *b == reference { "=" } else { "≠" }))
                .collect();
            println!(
                "ROWINV n={n} k={k} rank3={rank3} row0 vs m=2: {}",
                line.join(" ")
            );
            verdicts.push((n, k, rank3, by_m, reference));
        }
    }
    // The measured partition (see the module docs), asserted.
    let skinny = lloom_tensor::tuning::tuning().gemm.skinny_max;
    for (n, k, rank3, by_m, reference) in &verdicts {
        for (m, b) in by_m {
            let same = b == reference;
            let on_16_tile = *m >= 2 && *m < skinny;
            if *m == 1 {
                assert!(
                    !same,
                    "n={n} k={k} rank3={rank3}: the GEMV row matched the GEMM's — the seam moved"
                );
            } else if on_16_tile {
                assert!(
                    same,
                    "n={n} k={k} rank3={rank3} m={m}: a row moved within the 16-row tile"
                );
            }
        }
    }
    Ok(())
}
