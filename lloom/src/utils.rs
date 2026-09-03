//! Shared utilities: repeat_kv, repeat_penalty, causal mask, position-grid resample.

use lloom_tensor::{DType, Device, Result, Tensor};

/// One output position's resample taps: the first source index, and the weights
/// applied to the run of source samples starting there.
type Taps = Vec<(usize, Vec<f32>)>;

/// Separable antialiased-bilinear taps for one axis, `src_n -> dst_n`.
///
/// Matches `torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
/// antialias=True)` along a single axis. Separability is what makes that legal:
/// a 2-D triangle filter is the outer product of two 1-D ones, so resampling
/// rows then columns is the same operation as one 2-D pass, at `O(n)` taps
/// instead of `O(n^2)`.
///
/// # What the antialias term is, mechanically
///
/// `scale < 1` is a **downsample**, and torch widens the filter support by
/// `1/scale` in exactly that case. That widening *is* the antialiasing: it makes
/// each output sample average over every source sample that maps onto it, rather
/// than over the two nearest. Without it a downsample aliases by construction —
/// source samples between the two taps are simply never read.
///
/// On an **upsample** (`scale >= 1`) `filter_scale` is 1.0 and this reduces to
/// ordinary bilinear, which is why `antialias` is a no-op there and why a test
/// suite built on upsamples cannot see the difference (lloom #609).
fn antialias_bilinear_taps(src_n: usize, dst_n: usize) -> Taps {
    let scale = dst_n as f64 / src_n as f64;
    let filter_scale = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    // A bilinear (triangle) filter has support 1.0, widened on a downsample.
    let support = filter_scale;
    (0..dst_n)
        .map(|i| {
            // `align_corners=false`: sample at pixel centres.
            let centre = (i as f64 + 0.5) / scale;
            let left = ((centre - support) + 0.5).floor().max(0.0) as usize;
            let right = (((centre + support) + 0.5).floor() as usize).min(src_n);
            let mut weights = Vec::with_capacity(right.saturating_sub(left));
            let mut total = 0.0f64;
            for j in left..right {
                // Triangle filter evaluated in source space, scaled.
                let t = ((j as f64 + 0.5) - centre).abs() / filter_scale;
                let w = if t < 1.0 { 1.0 - t } else { 0.0 };
                weights.push(w);
                total += w;
            }
            // Normalising here rather than folding a constant in is what keeps
            // the edges right: near a boundary the support is clipped, so the
            // surviving taps must still sum to one or the output darkens there.
            if total > 0.0 {
                for w in weights.iter_mut() {
                    *w /= total;
                }
            }
            (left, weights.into_iter().map(|w| w as f32).collect())
        })
        .collect()
}

/// Antialiased bilinear resample of a **host-side** `[side, side, dim]` position
/// table to `[rows, cols, dim]`, returned flat.
///
/// Equivalent to `F.interpolate(mode="bilinear", align_corners=False,
/// antialias=True)` on the `[1, dim, side, side]` view of the same data.
///
/// # Why this is not `Tensor::interpolate2d`
///
/// **`Tensor::interpolate2d` is `upsample_nearest2d`** — the name says bilinear
/// and the implementation is nearest-neighbour (`candle-core/src/tensor.rs`
/// constructs `Op::UpsampleNearest2D`). `Tensor::upsample_bilinear2d` *does*
/// exist and is correct, but it has **no antialias term**, so on a downsample it
/// is the wrong one of the two. Measured on LFM2-VL's own 16x16 position table
/// resampled to 8x8, against `bilinear+antialias` as the reference:
///
/// | interpolation | max abs difference |
/// |---|---|
/// | bilinear, `align_corners=false`, **antialias on** | reference |
/// | bilinear, `align_corners=false`, antialias off | 5.10e0 |
/// | bicubic | 7.77e0 |
/// | nearest (what `interpolate2d` does) | 9.15e0 |
/// | bilinear, `align_corners=true` | 1.87e1 |
///
/// This runs **once per image**, never per token, so a host-side implementation
/// costs nothing on the decode path.
pub fn resize_pos_embed_antialias(
    table: &[f32],
    side: usize,
    dim: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let row_taps = antialias_bilinear_taps(side, rows);
    let col_taps = antialias_bilinear_taps(side, cols);

    // Pass 1: [side, side, dim] -> [rows, side, dim]
    let mut mid = vec![0f32; rows * side * dim];
    for (r, (r0, rw)) in row_taps.iter().enumerate() {
        for (k, w) in rw.iter().enumerate() {
            let sr = r0 + k;
            for c in 0..side {
                let src = (sr * side + c) * dim;
                let dst = (r * side + c) * dim;
                for d in 0..dim {
                    mid[dst + d] += w * table[src + d];
                }
            }
        }
    }
    // Pass 2: [rows, side, dim] -> [rows, cols, dim]
    let mut out = vec![0f32; rows * cols * dim];
    for r in 0..rows {
        for (c, (c0, cw)) in col_taps.iter().enumerate() {
            for (k, w) in cw.iter().enumerate() {
                let sc = c0 + k;
                let src = (r * side + sc) * dim;
                let dst = (r * cols + c) * dim;
                for d in 0..dim {
                    out[dst + d] += w * mid[src + d];
                }
            }
        }
    }
    out
}

/// Antialiased bilinear resample of a `[b, c, h, w]` tensor to `[b, c, rows, cols]`.
///
/// The tensor-native counterpart to [`resize_pos_embed_antialias`], for callers
/// whose position grid is already channels-first on a device. Both go through
/// [`antialias_bilinear_taps`], so the two cannot drift apart — the weights are
/// computed once, in one place.
///
/// The taps are applied as two `matmul`s (one per axis), which keeps the work on
/// whatever device the input is on and avoids a host round trip. Row taps
/// contract `h`, column taps contract `w`:
///
/// ```text
///   [b, c, rows, h] @ [b, c, h, w]  ->  [b, c, rows, w]
///   [b, c, rows, w] @ [b, c, w, cols] -> [b, c, rows, cols]
/// ```
///
/// See [`resize_pos_embed_antialias`] for why this exists rather than
/// `Tensor::interpolate2d` or `Tensor::upsample_bilinear2d`.
pub fn interpolate2d_bilinear_antialias(xs: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let (_b, _c, h, w) = xs.dims4()?;
    if h == rows && w == cols {
        return Ok(xs.clone());
    }
    let device = xs.device();
    // The taps are computed in f32 and the matmul runs in the input's dtype, so
    // a f16 grid stays f16 rather than being silently widened.
    let dtype = xs.dtype();

    // `[rows, h]`: dense weight matrix, zero outside each output's support.
    let dense = |taps: &Taps, src_n: usize, dst_n: usize| -> Result<Tensor> {
        let mut m = vec![0f32; dst_n * src_n];
        for (i, (start, weights)) in taps.iter().enumerate() {
            for (k, weight) in weights.iter().enumerate() {
                m[i * src_n + start + k] = *weight;
            }
        }
        Tensor::from_vec(m, (dst_n, src_n), device)?.to_dtype(dtype)
    };

    let row_m = dense(&antialias_bilinear_taps(h, rows), h, rows)?;
    let col_m = dense(&antialias_bilinear_taps(w, cols), w, cols)?;

    // Contract h: broadcast_matmul lifts [rows, h] over the leading [b, c].
    let out = row_m.broadcast_matmul(xs)?;
    // Contract w: transpose the column taps so `w` is the contracted axis.
    out.broadcast_matmul(&col_m.t()?.contiguous()?)
}

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

/// Build an *additive* causal attention mask of shape `(1, 1, tgt, tgt + offset)`,
/// where masked positions are `-inf` and visible ones are `0`.
///
/// This is the counterpart to [`build_causal_mask`] for models that apply the mask
/// with `scores.broadcast_add(&mask)` rather than by selecting on a `u8` mask.
///
/// The buffer holds `tgt * (tgt + offset)` values and does not depend on the batch
/// size, so the leading dim stays `1` and is broadcast over the batch by the caller.
/// Shaping it `(b, 1, ..)` instead claims `b` times more elements than the buffer
/// holds, which `Tensor::from_slice` does not reject; see
/// https://github.com/huggingface/candle/issues/3582.
///
/// `sw` optionally restricts each query to a sliding window of `sw` preceding keys.
pub fn build_additive_causal_mask(
    tgt: usize,
    offset: usize,
    sw: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<_> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                let past_ok = j <= i + offset;
                // Within the window iff `(i + offset) - j <= w`, rearranged to
                // `j + w >= i + offset` to stay in `usize` (no signed casts, no
                // subtraction underflow when `j > i + offset`).
                let sw_ok = match sw {
                    Some(w) => j + w >= i + offset,
                    None => true,
                };
                if past_ok && sw_ok {
                    0.
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), device)?.to_dtype(dtype)
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression test for https://github.com/huggingface/candle/issues/3582
    #[test]
    fn additive_causal_mask_is_batch_independent_and_broadcasts() {
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(3, 0, None, &device, DType::F32).unwrap();
        assert_eq!(
            mask.dims(),
            &[1, 1, 3, 3],
            "mask must carry a leading batch dim of 1 so broadcast_add applies it to every row",
        );

        // Broadcasting onto a 2-sequence batch must mask both rows identically and causally.
        let scores = Tensor::zeros((2, 1, 3, 3), DType::F32, &device).unwrap();
        let masked = scores
            .broadcast_add(&mask)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(
            masked[0], masked[1],
            "both batch rows must receive the same causal mask",
        );
        assert_eq!(masked[0][0], vec![0.0, neg, neg]);
        assert_eq!(masked[0][1], vec![0.0, 0.0, neg]);
        assert_eq!(masked[0][2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn additive_causal_mask_handles_kv_cache_offset() {
        // With `offset` cached keys the mask is rectangular: every query attends to
        // all cached prefix keys, then the usual causal triangle over the new tokens.
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(2, 2, None, &device, DType::F32)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(mask[0], vec![0.0, 0.0, 0.0, neg]);
        assert_eq!(mask[1], vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn additive_causal_mask_respects_sliding_window() {
        // With a sliding window of `w`, query i may attend to key j only when it is
        // both causal (j <= i + offset) and within the window (i + offset - j <= w).
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(4, 0, Some(1), &device, DType::F32)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // window = 1: each query sees itself and the single preceding position.
        assert_eq!(mask[0], vec![0.0, neg, neg, neg]);
        assert_eq!(mask[1], vec![0.0, 0.0, neg, neg]);
        assert_eq!(mask[2], vec![neg, 0.0, 0.0, neg]);
        assert_eq!(mask[3], vec![neg, neg, 0.0, 0.0]);
    }
}
