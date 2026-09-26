//! Shared utilities: repeat_kv, repeat_penalty, causal mask, position-grid resample.

use lloom_tensor::{DType, Device, Result, Tensor};

/// One output position's resample taps: the first source index, and the weights
/// applied to the run of source samples starting there.
type Taps = Vec<(usize, Vec<f32>)>;

/// The kernel an antialiased separable resample convolves with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AaFilter {
    /// The triangle filter, support 1: `mode="bilinear"`.
    Bilinear,
    /// Keys' cubic with `a = -0.5`, support 2: the antialiased `mode="bicubic"`
    /// of torch and Pillow. torch's non-antialiased bicubic uses `a = -0.75`;
    /// the two are different filters, not one filter at two settings.
    Bicubic,
}

impl AaFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
        }
    }

    fn weight(self, t: f64) -> f64 {
        let t = t.abs();
        match self {
            Self::Bilinear => {
                if t < 1.0 {
                    1.0 - t
                } else {
                    0.0
                }
            }
            Self::Bicubic => {
                const A: f64 = -0.5;
                if t < 1.0 {
                    ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0
                } else if t < 2.0 {
                    ((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// Separable antialiased taps for one axis, `src_n -> dst_n`, as `f64`.
///
/// Matches `torch.nn.functional.interpolate(mode=<filter>, align_corners=False,
/// antialias=True)` along a single axis. Separability is what makes that legal:
/// a 2-D kernel of this kind is the outer product of two 1-D ones, so resampling
/// rows then columns is the same operation as one 2-D pass, at `O(n)` taps
/// instead of `O(n^2)`.
///
/// # What the antialias term is, mechanically
///
/// `scale < 1` is a **downsample**, and torch widens the filter support by
/// `1/scale` in exactly that case. That widening *is* the antialiasing: it makes
/// each output sample average over every source sample that maps onto it, rather
/// than over the nearest few. Without it a downsample aliases by construction —
/// source samples between the taps are simply never read.
///
/// On an **upsample** (`scale >= 1`) `filter_scale` is 1.0 and this reduces to
/// the plain filter, which is why `antialias` is a no-op there and why a test
/// suite built on upsamples cannot see the difference (lloom #609).
pub fn antialias_taps(src_n: usize, dst_n: usize, filter: AaFilter) -> Vec<(usize, Vec<f64>)> {
    let scale = dst_n as f64 / src_n as f64;
    let filter_scale = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = filter.support() * filter_scale;
    (0..dst_n)
        .map(|i| {
            // `align_corners=false`: sample at pixel centres.
            let centre = (i as f64 + 0.5) / scale;
            let left = ((centre - support) + 0.5).floor().max(0.0) as usize;
            let right = (((centre + support) + 0.5).floor() as usize).min(src_n);
            let mut weights = Vec::with_capacity(right.saturating_sub(left));
            let mut total = 0.0f64;
            for j in left..right {
                let w = filter.weight(((j as f64 + 0.5) - centre) / filter_scale);
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
            (left, weights)
        })
        .collect()
}

/// [`antialias_taps`] for the triangle filter, narrowed to `f32`.
fn antialias_bilinear_taps(src_n: usize, dst_n: usize) -> Taps {
    antialias_taps(src_n, dst_n, AaFilter::Bilinear)
        .into_iter()
        .map(|(left, w)| (left, w.into_iter().map(|w| w as f32).collect()))
        .collect()
}

/// One axis of [`resize_rgb8_antialias`]: the taps in 16-bit fixed point, and
/// the fraction bits they carry.
///
/// The precision is the largest that keeps the largest weight's doubling below
/// `2^15`, capped at 22 bits, and each weight is rounded half away from zero —
/// torch's `_compute_index_ranges_int16_weights`, which is what makes a uint8
/// resize reproduce torch's to the byte.
fn fixed_point_taps(taps: Vec<(usize, Vec<f64>)>) -> (Vec<(usize, Vec<i16>)>, u32) {
    let w_max = taps
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0.0f64, f64::max);
    let mut bits = 0u32;
    while bits < 22 {
        let next = (0.5 + w_max * (1u64 << (bits + 1)) as f64) as i64;
        if next >= 1 << 15 {
            break;
        }
        bits += 1;
    }
    let one = (1u64 << bits) as f64;
    let taps = taps
        .into_iter()
        .map(|(left, w)| {
            let q = w
                .into_iter()
                .map(|w| {
                    let v = w * one;
                    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i16
                })
                .collect();
            (left, q)
        })
        .collect();
    (taps, bits.max(1))
}

/// One fixed-point pass over axis `stride`-separated samples: `src` holds
/// `outer` lines of `src_n` samples of `channels` bytes, `dst` receives `outer`
/// lines of `taps.len()` samples. `line` / `step` are the element distances
/// between consecutive lines and consecutive samples on the resampled axis.
#[allow(clippy::too_many_arguments)]
fn fixed_point_pass(
    src: &[u8],
    dst: &mut [u8],
    taps: &[(usize, Vec<i16>)],
    bits: u32,
    outer: usize,
    src_line: usize,
    src_step: usize,
    dst_line: usize,
    dst_step: usize,
    channels: usize,
) {
    let round = 1i32 << (bits - 1);
    for o in 0..outer {
        for (i, (left, w)) in taps.iter().enumerate() {
            for c in 0..channels {
                let mut acc = round;
                for (k, &wk) in w.iter().enumerate() {
                    let s = src[o * src_line + (left + k) * src_step + c] as i32;
                    acc += s * wk as i32;
                }
                dst[o * dst_line + i * dst_step + c] = (acc >> bits).clamp(0, 255) as u8;
            }
        }
    }
}

/// Resize an interleaved 8-bit image `[height, width, channels]` to
/// `[new_height, new_width, channels]`, antialiased, in fixed point.
///
/// Reproduces `torchvision.transforms.v2.functional.resize` on a CPU `uint8`
/// tensor (`antialias=True`, `align_corners=False`) **byte for byte**: torch
/// resizes `uint8` natively for bilinear and bicubic, with 16-bit fixed-point
/// weights ([`fixed_point_taps`]), the horizontal pass first, and the
/// intermediate rounded to `uint8` between the passes. A float resample
/// rounded once at the end differs from it by one level on a fraction of the
/// pixels, which a patch-level comparison against the reference sees as a
/// preprocessing defect. An axis whose size does not change is not resampled,
/// and a resize to the same size returns the input.
pub fn resize_rgb8_antialias(
    src: &[u8],
    height: usize,
    width: usize,
    channels: usize,
    new_height: usize,
    new_width: usize,
    filter: AaFilter,
) -> Result<Vec<u8>> {
    if src.len() != height * width * channels {
        lloom_tensor::bail!(
            "resize: {} bytes for a {height}x{width}x{channels} image",
            src.len()
        );
    }
    if new_height == 0 || new_width == 0 || height == 0 || width == 0 {
        lloom_tensor::bail!(
            "resize: {height}x{width} -> {new_height}x{new_width} has an empty side"
        );
    }
    let mut cur = src.to_vec();
    let mut w_now = width;
    if new_width != width {
        let (taps, bits) = fixed_point_taps(antialias_taps(width, new_width, filter));
        let mut out = vec![0u8; height * new_width * channels];
        fixed_point_pass(
            &cur,
            &mut out,
            &taps,
            bits,
            height,
            width * channels,
            channels,
            new_width * channels,
            channels,
            channels,
        );
        cur = out;
        w_now = new_width;
    }
    if new_height != height {
        let (taps, bits) = fixed_point_taps(antialias_taps(height, new_height, filter));
        let mut out = vec![0u8; new_height * w_now * channels];
        // Each column is a line; consecutive samples are a row apart.
        for x in 0..w_now {
            fixed_point_pass(
                &cur[x * channels..],
                &mut out[x * channels..],
                &taps,
                bits,
                1,
                0,
                w_now * channels,
                0,
                w_now * channels,
                channels,
            );
        }
        cur = out;
    }
    Ok(cur)
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
    /// A 7x5 RGB image whose bytes are `(i * 53 + 17) % 256`.
    fn probe_image() -> Vec<u8> {
        (0..7 * 5 * 3)
            .map(|i| ((i * 53 + 17) % 256) as u8)
            .collect()
    }

    /// `torchvision.transforms.v2.functional.resize` on the probe image as a
    /// CPU uint8 tensor, `antialias=True` (torch 2.14.0, torchvision 0.29.0),
    /// pinned byte for byte: a downscale, an upscale, and one axis alone.
    #[test]
    fn the_uint8_resize_is_torchvisions_byte_for_byte() {
        let src = probe_image();
        let cases: &[(usize, usize, AaFilter, &[u8])] = &[
            (
                4,
                3,
                AaFilter::Bicubic,
                &[
                    100, 102, 92, 127, 114, 143, 142, 142, 126, 114, 80, 141, 115, 129, 146, 127,
                    145, 84, 76, 134, 162, 121, 159, 84, 158, 80, 139, 126, 179, 81, 176, 80, 143,
                    77, 131, 184,
                ],
            ),
            (
                4,
                3,
                AaFilter::Bilinear,
                &[
                    99, 100, 95, 123, 119, 142, 145, 140, 124, 110, 84, 137, 119, 131, 144, 129,
                    141, 89, 81, 134, 158, 128, 152, 92, 150, 86, 139, 123, 176, 90, 170, 88, 134,
                    82, 128, 181,
                ],
            ),
            (
                11,
                9,
                AaFilter::Bicubic,
                &[
                    2, 54, 129, 62, 121, 89, 166, 236, 25, 133, 213, 91, 77, 130, 183, 181, 58,
                    125, 252, 41, 96, 177, 145, 211, 130, 205, 255, 14, 75, 141, 74, 94, 101, 178,
                    128, 37, 152, 145, 103, 89, 142, 195, 113, 77, 145, 144, 53, 97, 150, 157, 133,
                    151, 217, 152, 33, 108, 160, 94, 64, 120, 198, 0, 56, 181, 63, 123, 108, 161,
                    215, 31, 105, 176, 10, 73, 105, 120, 177, 36, 184, 237, 2, 49, 125, 178, 116,
                    82, 137, 232, 14, 73, 209, 82, 149, 126, 179, 250, 49, 121, 201, 30, 91, 121,
                    138, 207, 55, 201, 255, 22, 75, 142, 195, 102, 102, 158, 150, 38, 94, 160, 104,
                    116, 143, 196, 164, 76, 144, 168, 54, 100, 143, 158, 149, 81, 218, 174, 47,
                    106, 159, 212, 66, 119, 181, 2, 55, 119, 68, 121, 50, 160, 213, 10, 103, 173,
                    102, 71, 104, 168, 175, 42, 104, 235, 11, 64, 124, 176, 249, 80, 136, 212, 9,
                    71, 138, 78, 147, 61, 177, 249, 16, 120, 200, 114, 90, 120, 186, 206, 51, 122,
                    255, 17, 81, 140, 193, 186, 100, 155, 173, 36, 92, 149, 102, 127, 88, 194, 189,
                    44, 141, 179, 136, 101, 141, 202, 167, 78, 138, 200, 45, 98, 158, 211, 13, 118,
                    179, 57, 54, 117, 152, 120, 56, 118, 212, 23, 62, 171, 108, 154, 104, 166, 220,
                    51, 102, 156, 25, 63, 116, 177, 230, 0, 137, 200, 56, 73, 138, 169, 139, 62,
                    139, 231, 15, 81, 192, 114, 173, 121, 187, 239, 50, 123, 175, 15, 82, 135, 189,
                    242, 17, 149, 211, 78, 85, 149, 182, 151, 80, 150, 243, 40, 93, 203, 132, 185,
                    134, 198, 251, 72, 134, 187, 41, 94, 147,
                ],
            ),
            (
                11,
                9,
                AaFilter::Bilinear,
                &[
                    17, 70, 123, 70, 123, 91, 158, 211, 37, 133, 186, 97, 79, 132, 185, 167, 78,
                    131, 227, 53, 106, 173, 141, 194, 141, 194, 247, 29, 82, 135, 82, 97, 103, 170,
                    120, 49, 145, 134, 109, 91, 144, 197, 115, 90, 143, 136, 65, 105, 147, 153,
                    129, 153, 206, 143, 46, 99, 152, 99, 67, 120, 187, 13, 66, 162, 73, 126, 108,
                    161, 214, 54, 107, 160, 29, 82, 106, 117, 170, 52, 170, 223, 20, 64, 117, 170,
                    117, 85, 138, 205, 31, 84, 180, 91, 144, 126, 179, 232, 72, 125, 178, 47, 100,
                    124, 135, 188, 70, 188, 241, 38, 81, 134, 187, 103, 102, 155, 139, 48, 101,
                    145, 108, 119, 143, 196, 156, 89, 142, 153, 64, 106, 141, 152, 143, 87, 205,
                    165, 55, 98, 151, 204, 66, 119, 172, 12, 65, 118, 72, 125, 64, 160, 213, 10,
                    106, 159, 98, 81, 105, 158, 169, 51, 104, 222, 19, 72, 115, 168, 221, 83, 136,
                    189, 29, 82, 135, 89, 142, 81, 177, 230, 27, 123, 176, 115, 98, 122, 175, 186,
                    68, 121, 239, 36, 89, 132, 185, 169, 100, 153, 160, 46, 99, 144, 106, 128, 98,
                    194, 178, 44, 140, 162, 132, 107, 139, 192, 157, 85, 138, 187, 53, 106, 150,
                    203, 23, 118, 171, 68, 64, 117, 143, 124, 73, 116, 212, 32, 62, 158, 107, 150,
                    106, 157, 210, 65, 103, 156, 41, 71, 124, 167, 220, 17, 135, 188, 70, 81, 134,
                    158, 141, 80, 133, 229, 26, 79, 175, 114, 167, 121, 174, 227, 67, 120, 173, 35,
                    88, 141, 179, 232, 29, 147, 200, 82, 93, 146, 170, 153, 92, 145, 241, 38, 91,
                    187, 126, 179, 133, 186, 239, 79, 132, 185, 47, 100, 153,
                ],
            ),
            (
                7,
                2,
                AaFilter::Bicubic,
                &[
                    100, 146, 93, 171, 118, 164, 120, 67, 128, 92, 138, 102, 147, 94, 155, 119,
                    165, 129, 68, 129, 120, 139, 103, 94, 95, 156, 147, 166, 130, 121, 130, 121,
                    85, 104, 95, 156, 157, 148, 112, 131, 122, 183,
                ],
            ),
            (
                7,
                2,
                AaFilter::Bilinear,
                &[
                    106, 148, 96, 168, 116, 158, 122, 70, 123, 90, 132, 103, 149, 97, 150, 117,
                    159, 130, 71, 124, 119, 133, 104, 99, 98, 151, 146, 160, 131, 126, 125, 120,
                    91, 105, 100, 153, 152, 147, 118, 132, 127, 180,
                ],
            ),
        ];
        for &(nh, nw, filter, want) in cases {
            let got = resize_rgb8_antialias(&src, 7, 5, 3, nh, nw, filter).unwrap();
            assert_eq!(got, want, "{filter:?} 7x5 -> {nh}x{nw}");
        }
        // A resize to the same size is the input, as torchvision returns it.
        assert_eq!(
            resize_rgb8_antialias(&src, 7, 5, 3, 7, 5, AaFilter::Bicubic).unwrap(),
            src
        );
    }

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
