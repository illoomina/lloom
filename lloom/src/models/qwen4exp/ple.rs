//! N-gram PLE (Predictive Lookup Embedding): the hashed n-gram table injected
//! into the n-stream carry at one layer (`ple_layers`, GGUF `blk.1`).
//!
//! Three pieces, in dataflow order:
//!
//! 1. [`NgramHash`] — the multi-head hash from the token window to table rows,
//!    the engine's (`nn::ngram_memory`), with its constants from the GGUF.
//!    Bit-exact to the reference (vLLM `qwen4_exp/nvidia/ops/ple.py`, llama.cpp
//!    `qwen4exp.cpp` `llm_graph_input_ple::set_input`): per n-gram order
//!    `n ∈ 2..=ngram_size`, `mixed_n = tok[t]·m[0] ^ tok[t-1]·m[1] ^ … ^
//!    tok[t-n+1]·m[n-1]` in wrapping `i64`, and head `h` of that order reads
//!    row `offset[h] + rem(mixed_n, size[h])` with a non-negative remainder.
//!    An EOS (or a missing predecessor) cuts the window: it and everything
//!    older read as EOS; the current token being EOS does not cut its own
//!    context. A wrong hash changes no shape — it is a text-gate-only trap
//!    (`design/docs/expert-streaming.md` §20.1a), so the tests here pin it
//!    against hand-computed rows and the checkpoint's own hash constants.
//!    On a ring arm a producer thread hashes and gathers (the n-gram memory's
//!    host fill, §20.13) and the GPU waits for its rows before this block.
//! 2. [`PleTable`] — the row gather, the engine's `nn::ngram_memory`. The
//!    table is 320 M rows × 160 (28.8 GB at IQ4_NL, the second GGUF shard), so
//!    it is never loaded: a token's 16 rows (90 B each) are read from a memory
//!    map — the cold ones concurrently — and dequantized on the CPU.
//! 3. [`PleLayer`] — key/value projections of the gathered embedding, grouped
//!    RMS norms, the per-stream signed-sqrt gate, and the dilated depthwise
//!    conv over the gated value (kernel 4, dilation `ngram_size`), whose
//!    9-position history is a [`RecurrentState`] channel. Output is the delta
//!    added to the carry: `gated + silu(conv)`.

use super::hc::grouped_rms_norm;
use crate::nn::{ops, RecurrentState};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor};

/// The table's storage is the engine's n-gram memory (`nn::ngram_memory`),
/// under the names this model's code has always used.
pub use crate::nn::ngram_memory::{
    CachedRows as CachedPleTable, MmapRows as MmapPleTable, NgramHash, NgramWindow,
    RowTable as PleTable,
};

/// The PLE block's weights for its one layer.
#[derive(Debug, Clone)]
pub struct PleLayer {
    /// `ple_key.weight` `[hc_dim, ple_dim]`.
    key: QMatMul,
    /// `ple_value.weight` `[hidden, ple_dim]`.
    value: QMatMul,
    /// Grouped-norm γ `[hc_dim]` (converter-folded `1 + w`), per §hc.
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// Depthwise conv kernel `[hc_dim, k]`, tap `k−1` newest.
    conv_weight: Tensor,
    unit_d: Tensor,
    n: usize,
    d: usize,
    k: usize,
    dilation: usize,
    eps: f32,
    /// The gate's scalars `(1e-6, f32::MAX, 0)` on the device at the gate's
    /// dtype, made on first use: a scalar operand is an upload per call
    /// (`TensorOrScalar`), and these are the same three every step.
    gate_scalars: std::sync::OnceLock<(Tensor, Tensor, Tensor)>,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: QMatMul,
        value: QMatMul,
        norm_key: Tensor,
        norm_query: Tensor,
        norm_conv: Tensor,
        conv_weight: Tensor,
        n: usize,
        d: usize,
        dilation: usize,
        eps: f32,
    ) -> Result<Self> {
        let (ch, k) = conv_weight.dims2()?;
        if ch != n * d {
            lloom_tensor::bail!("ple conv weight [{ch}, {k}] vs hc_dim {}", n * d)
        }
        let unit_d = Tensor::ones(d, norm_key.dtype(), norm_key.device())?;
        Ok(Self {
            key,
            value,
            norm_key,
            norm_query,
            norm_conv,
            conv_weight,
            unit_d,
            n,
            d,
            k,
            dilation,
            eps,
            gate_scalars: std::sync::OnceLock::new(),
        })
    }

    /// See [`Self::gate_scalars`]. Made outside any pass plan: they outlive
    /// every pass.
    fn gate_scalars(&self, like: &Tensor) -> Result<&(Tensor, Tensor, Tensor)> {
        if let Some(t) = self.gate_scalars.get() {
            if t.0.dtype() == like.dtype() && t.0.device().same_device(like.device()) {
                return Ok(t);
            }
        }
        #[cfg(feature = "metal")]
        let _declared = lloom_tensor::EscapeGuard::new();
        let (dt, dev) = (like.dtype(), like.device());
        // Converted on the host and moved, as the scalar operand path does.
        let cpu = lloom_tensor::Device::Cpu;
        let made = (
            Tensor::new(1e-6f32, &cpu)?.to_dtype(dt)?.to_device(dev)?,
            Tensor::new(f32::MAX, &cpu)?.to_dtype(dt)?.to_device(dev)?,
            Tensor::new(0f32, &cpu)?.to_dtype(dt)?.to_device(dev)?,
        );
        let _ = self.gate_scalars.set(made);
        match self.gate_scalars.get() {
            Some(t) if t.0.dtype() == dt && t.0.device().same_device(dev) => Ok(t),
            _ => lloom_tensor::bail!("ple gate scalars: made for another dtype or device"),
        }
    }

    /// The device the block's weights live on.
    pub fn device(&self) -> &lloom_tensor::Device {
        self.norm_key.device()
    }

    /// The conv history the state channel carries per position: `(k−1)·dilation`.
    pub fn history_len(&self) -> usize {
        (self.k - 1) * self.dilation
    }

    /// The delta to add to the carry `hidden [1, seq, n·d]` for the gathered
    /// embeddings `emb [seq, ple_dim]`, advancing the conv history channel `l`.
    pub fn forward(
        &self,
        hidden: &Tensor,
        emb: &Tensor,
        conv_state: &mut RecurrentState,
        l: usize,
    ) -> Result<Tensor> {
        let (b, seq, hc_dim) = hidden.dims3()?;
        let (n, d) = (self.n, self.d);
        if b != 1 || hc_dim != n * d || emb.dim(0)? != seq {
            lloom_tensor::bail!(
                "ple: hidden {:?} / emb {:?} do not fit n={n} d={d}",
                hidden.shape(),
                emb.shape()
            )
        }
        let dt = hidden.dtype();
        let emb = emb.to_dtype(dt)?.reshape((1, seq, emb.dim(1)?))?;
        let key = self.key.forward(&emb)?; // [1, seq, hc_dim]
        let value = self.value.forward(&emb)?; // [1, seq, d]

        let key_n = grouped_rms_norm(&key, &self.unit_d, &self.norm_key, n, d, self.eps)?;
        let query_n = grouped_rms_norm(hidden, &self.unit_d, &self.norm_query, n, d, self.eps)?;

        // Per-stream dot, signed square root, sigmoid.
        let prod = (key_n.reshape((1, seq, n, d))? * query_n.reshape((1, seq, n, d))?)?;
        let s = (prod.sum(3)? * (1.0 / (d as f64).sqrt()))?; // [1, seq, n]
        let (lo, hi, zero) = self.gate_scalars(&s)?;
        let shape = s.shape();
        let (lo, hi, zero) = (
            lo.broadcast_as(shape)?,
            hi.broadcast_as(shape)?,
            zero.broadcast_as(shape)?,
        );
        let mag = s.abs()?.clamp(&lo, &hi)?.sqrt()?;
        let sgn = (s.gt(&zero)?.to_dtype(dt)? - s.lt(&zero)?.to_dtype(dt)?)?;
        let gate = ops::sigmoid(&(sgn * mag)?)?; // [1, seq, n]

        // value broadcast over the streams, scaled by the gate.
        let gated = value
            .reshape((1, seq, 1, d))?
            .broadcast_mul(&gate.reshape((1, seq, n, 1))?)?
            .reshape((1, seq, n * d))?;

        let normed = grouped_rms_norm(&gated, &self.unit_d, &self.norm_conv, n, d, self.eps)?;
        let conv = self.dilated_conv(&normed, conv_state, l)?; // [1, seq, n*d]
        let conv = ops::silu(&conv)?;
        gated + conv
    }

    /// Depthwise causal conv, dilated by `ngram_size`, over `x [1, seq, hc_dim]`
    /// with the channel's `[hc_dim, hist]` history prepended (oldest first);
    /// writes the new history back.
    fn dilated_conv(
        &self,
        x: &Tensor,
        conv_state: &mut RecurrentState,
        l: usize,
    ) -> Result<Tensor> {
        let (_, seq, ch) = x.dims3()?;
        let hist = self.history_len();
        let dt = x.dtype();
        let x_t = x.reshape((seq, ch))?.t()?.contiguous()?; // [ch, seq]
        let history = match conv_state.get(l) {
            Some(h) => h.to_dtype(dt)?,
            None => Tensor::zeros((ch, hist), dt, x.device())?,
        };
        let padded = Tensor::cat(&[&history, &x_t], 1)?; // [ch, hist + seq]
        let w = self.conv_weight.to_dtype(dt)?; // [ch, k]
        let mut acc: Option<Tensor> = None;
        for tap in 0..self.k {
            // tap k reads (k−1−tap)·dilation positions back: column t + tap·dilation.
            let cols = padded.narrow(1, tap * self.dilation, seq)?; // [ch, seq]
            let term = cols.broadcast_mul(&w.narrow(1, tap, 1)?)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let out = acc.expect("k >= 1");
        // New history: the last `hist` columns of the padded input.
        let new_hist = padded.narrow(1, seq, hist)?.contiguous()?;
        conv_state.write(l, new_hist)?;
        out.t()?.reshape((1, seq, ch))
    }
}

/// The PLE block's decode step ([`PleLayer::forward`] at one position,
/// f32): the key and value projections of the gathered rows, the grouped
/// norms, the per-stream signed-sqrt gate, the gated value, its norm, the
/// dilated conv over the history channel, the SiLU and the sum — the delta
/// the model adds to the carry. The history read, its two copies and the
/// next history are layer `l`'s conv recurrence. `In` is the carry and the
/// layer; the position's rows, gathered on the host, are the step's input.
#[cfg(not(target_arch = "wasm32"))]
impl crate::optimizer::Visit for PleLayer {
    type In = (crate::optimizer::NodeId, usize);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (hidden, l): Self::In) -> Self::Out {
        use crate::optimizer::{Region, Shape};
        let f32 = lloom_tensor::DType::F32;
        let (n, d) = (self.n, self.d);
        let nd = n * d;
        let (_, key_shape, _) = crate::optimizer::walk::qmat_operand(&self.key);
        let width = key_shape.0.last().map_or(0, |k| match k {
            crate::optimizer::Dim::Const(k) => *k,
            crate::optimizer::Dim::Sym(_) => 0,
        });
        let emb = w.input("ple.rows", Shape::of(&[1, 1, width]), f32);
        let unit = w.weight("ple_unit_d", Shape::of(&[d]), self.unit_d.dtype());
        let norm = |w: &mut crate::optimizer::Walker, x, gamma: &Tensor, leaf: &str| {
            let g = w.weight(leaf, Shape::of(gamma.dims()), gamma.dtype());
            let xs = w.view("reshape", x, Shape::of(&[1, 1, n, d]));
            let normed = w.rms_norm(unit, xs);
            let normed = w.view("reshape", normed, Shape::of(&[1, 1, nd]));
            w.mul(normed, g)
        };
        let key = w.project_qmat("ple_key.weight", &self.key, emb);
        let value = w.project_qmat("ple_value.weight", &self.value, emb);
        let key_n = norm(w, key, &self.norm_key, "ple_norm_key.weight");
        let query_n = norm(w, hidden, &self.norm_query, "ple_norm_query.weight");
        let a = w.view("reshape", key_n, Shape::of(&[1, 1, n, d]));
        let b = w.view("reshape", query_n, Shape::of(&[1, 1, n, d]));
        let prod = w.mul(a, b);
        let s3 = Shape::of(&[1, 1, n]);
        let s = w.opaque("fast_sum", &[prod], s3.clone(), f32);
        let s = w.scale(s, (1.0 / (d as f64).sqrt()) as f32);
        // The gate's scalars, made once and kept (`gate_scalars`).
        let lo = w.input("ple.gate.lo", Shape::of(&[]), f32);
        let hi = w.input("ple.gate.hi", Shape::of(&[]), f32);
        let zero = w.input("ple.gate.zero", Shape::of(&[]), f32);
        let mag = w.opaque("abs", &[s], s3.clone(), f32);
        let mag = w.opaque("bmaximum", &[mag, lo], s3.clone(), f32);
        let mag = w.opaque("bminimum", &[mag, hi], s3.clone(), f32);
        let mag = w.opaque("sqrt", &[mag], s3.clone(), f32);
        let gt = w.opaque("gt", &[s, zero], s3.clone(), lloom_tensor::DType::U8);
        let gt = w.opaque("cast", &[gt], s3.clone(), f32);
        let lt = w.opaque("lt", &[s, zero], s3.clone(), lloom_tensor::DType::U8);
        let lt = w.opaque("cast", &[lt], s3.clone(), f32);
        let sgn = w.opaque("bsub", &[gt, lt], s3, f32);
        let gate = w.mul(sgn, mag);
        let gate = w.sigmoid(gate);
        let v4 = w.view("reshape", value, Shape::of(&[1, 1, 1, d]));
        let g4 = w.view("reshape", gate, Shape::of(&[1, 1, n, 1]));
        let gated = w.mul(v4, g4);
        let gated = w.view("reshape", gated, Shape::of(&[1, 1, nd]));
        let normed = norm(w, gated, &self.norm_conv, "ple_norm_conv.weight");
        // The dilated conv: the history and this position as one padded
        // window (a `cat`: two copies), the taps' products summed, the next
        // history copied out.
        let hist = self.history_len();
        let x_t = w.view("transpose", normed, Shape::of(&[nd, 1]));
        let history = w.input(&format!("ple.conv@{l}"), Shape::of(&[nd, hist]), f32);
        let h_copy = w.opaque("copy2d", &[history], Shape::of(&[nd, hist]), f32);
        let x_copy = w.opaque(
            "copy2d",
            &[x_t, h_copy],
            Shape::of(&[nd, hist + 1]),
            f32,
        );
        let cw = w.weight(
            "ple_conv1d.weight",
            Shape::of(self.conv_weight.dims()),
            self.conv_weight.dtype(),
        );
        let mut acc = None;
        for _ in 0..self.k {
            let cols = w.view("narrow", x_copy, Shape::of(&[nd, 1]));
            let wt = w.view("narrow", cw, Shape::of(&[nd, 1]));
            let term = w.mul(cols, wt);
            acc = Some(match acc {
                Some(a) => w.add(a, term),
                None => term,
            });
        }
        let next = w.opaque("copy2d", &[x_copy], Shape::of(&[nd, hist]), f32);
        // The next step's history: written here, read there.
        w.state_out(next);
        w.mark(&[history, h_copy, x_copy, next], Region::ConvRing(l));
        let conv = acc.expect("k >= 1");
        let conv = w.view("transpose", conv, Shape::of(&[1, 1, nd]));
        let conv = w.silu(conv);
        w.add(gated, conv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{DType, Device};

    /// The checkpoint's recorded hash constants (`qwen4exp.ple.*`, read from the
    /// IQ3_XXS GSQ-RCO GGUF header, 2026-09-21) — the ground truth the
    /// derivation is checked against.
    const MULTIPLIERS: [i64; 3] = [23703573157769, 20109073645365, 8052911324071];
    const SIZES: [i64; 16] = [
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
        20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ];
    const OFFSETS: [i64; 16] = [
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
        180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
    ];
    const EOS: u32 = 248044;

    fn real_hash() -> NgramHash {
        NgramHash::new(
            MULTIPLIERS.to_vec(),
            SIZES.to_vec(),
            OFFSETS.to_vec(),
            8,
            EOS,
        )
        .unwrap()
    }

    /// The reference derivation reproduces every recorded constant: seed 1234
    /// (the config default), vocab 248320, base 20 000 000, layer id 0. So the
    /// hash constants in the file are the ones the reference computes, and
    /// `derive` is a faithful port of the reference's splitmix64 / nth-prime.
    #[test]
    fn derivation_reproduces_checkpoint_constants() {
        let h = NgramHash::derive(1234, 248320, 3, 8, 20_000_000, 0, EOS).unwrap();
        assert_eq!(h.multipliers, MULTIPLIERS);
        assert_eq!(h.sizes, SIZES);
        assert_eq!(h.offsets, OFFSETS);
        assert_eq!(h.total_rows(), 320001446);
    }

    /// Hand-computed rows for a 3-token window, both orders, every head:
    /// `mixed_2 = t0·m0 ^ t1·m1`, `mixed_3 = mixed_2 ^ t2·m2`, row = offset +
    /// mixed mod size. Computed here with i128 arithmetic (no wrapping can
    /// occur — the products stay below 2^63 for in-vocab ids) as the
    /// independent reference.
    #[test]
    fn rows_match_hand_computation() {
        let h = real_hash();
        let (t0, t1, t2) = (846u32, 25u32, 1234u32);
        let mut out = vec![0i64; 16];
        h.row_ids(&[Some(t0), Some(t1), Some(t2)], &mut out);
        let m2 = (t0 as i128 * MULTIPLIERS[0] as i128) ^ (t1 as i128 * MULTIPLIERS[1] as i128);
        let m3 = m2 ^ (t2 as i128 * MULTIPLIERS[2] as i128);
        assert!(m2 >= 0 && m2 < (1i128 << 63));
        assert!(m3 >= 0 && m3 < (1i128 << 63));
        for hd in 0..8 {
            let want = OFFSETS[hd] as i128 + m2 % SIZES[hd] as i128;
            assert_eq!(out[hd] as i128, want, "bigram head {hd}");
            let want3 = OFFSETS[8 + hd] as i128 + m3 % SIZES[8 + hd] as i128;
            assert_eq!(out[8 + hd] as i128, want3, "trigram head {hd}");
        }
        // The reference's exact numbers for this window (vLLM semantics),
        // pinned so a change in operator order is caught by value:
        assert_eq!(out[0], (m2 % SIZES[0] as i128) as i64);
        assert_eq!(out[15], OFFSETS[15] + (m3 % SIZES[15] as i128) as i64);
    }

    /// The remainder is non-negative even when the XOR has its sign bit set
    /// (`torch.remainder` semantics), never the C-style negative remainder and
    /// never the unsigned reinterpretation.
    #[test]
    fn remainder_is_python_style_on_negative_mix() {
        // A multiplier set whose products overflow into the sign bit.
        let h = NgramHash::new(
            vec![i64::MAX / 3, i64::MAX / 5, 7],
            vec![101; 16],
            vec![0; 16],
            8,
            EOS,
        )
        .unwrap();
        let mut out = vec![0i64; 16];
        h.row_ids(&[Some(7), Some(9), Some(11)], &mut out);
        let mixed = (7i64.wrapping_mul(i64::MAX / 3)) ^ (9i64.wrapping_mul(i64::MAX / 5));
        assert!(mixed < 0, "the fixture must exercise a negative mix");
        let want = mixed.rem_euclid(101);
        assert_eq!(out[0], want);
        assert!(out.iter().all(|&r| (0..101).contains(&r)));
    }

    /// EOS scoping: an EOS one position back cuts the trigram head's older
    /// predecessor to EOS too; an EOS two back only affects the trigram heads;
    /// the current token being EOS cuts nothing; a missing predecessor
    /// (sequence start) reads as EOS. Regression: a hash that ignores the
    /// reset gives a different row for the cut case.
    #[test]
    fn eos_resets_the_window_by_scope() {
        let h = real_hash();
        let ids = |w: &[Option<u32>]| {
            let mut o = vec![0i64; 16];
            h.row_ids(w, &mut o);
            o
        };
        // EOS one back: bigram heads see (t0, EOS); trigram heads see (t0, EOS, EOS)
        // regardless of the real two-back token.
        let cut = ids(&[Some(5), Some(EOS), Some(77)]);
        let cut_other = ids(&[Some(5), Some(EOS), Some(78)]);
        assert_eq!(
            cut, cut_other,
            "the token behind an EOS must not reach the hash"
        );
        let explicit = ids(&[Some(5), Some(EOS), Some(EOS)]);
        assert_eq!(cut, explicit);
        // Not resetting would be a different row: prove the reset is live.
        let uncut = ids(&[Some(5), Some(6), Some(77)]);
        assert_ne!(cut[8..], uncut[8..]);
        // EOS two back: bigram heads unchanged vs a non-EOS two-back.
        let a = ids(&[Some(5), Some(6), Some(EOS)]);
        let b = ids(&[Some(5), Some(6), Some(77)]);
        assert_eq!(a[..8], b[..8]);
        assert_ne!(a[8..], b[8..]);
        // The current token being EOS does not cut its predecessors.
        let c = ids(&[Some(EOS), Some(6), Some(77)]);
        let c2 = ids(&[Some(EOS), Some(6), Some(78)]);
        assert_ne!(c[8..], c2[8..]);
        // Missing predecessors are EOS.
        let start = ids(&[Some(5), None, None]);
        assert_eq!(start, ids(&[Some(5), Some(EOS), Some(EOS)]));
        let one = ids(&[Some(5), Some(6), None]);
        assert_eq!(one, ids(&[Some(5), Some(6), Some(EOS)]));
    }

    /// The window rotates newest-last and reads newest-first.
    #[test]
    fn ngram_window_rotates() {
        let mut w = NgramWindow::new(2);
        assert_eq!(w.window(Some(1)), vec![Some(1), None, None]);
        w.push(Some(1));
        assert_eq!(w.window(Some(2)), vec![Some(2), Some(1), None]);
        w.push(Some(2));
        w.push(Some(3));
        assert_eq!(w.window(Some(4)), vec![Some(4), Some(3), Some(2)]);
        w.reset();
        assert_eq!(w.window(Some(9)), vec![Some(9), None, None]);
    }

    /// The dilated conv reads taps `t − (k−1−j)·dilation` and carries a
    /// `(k−1)·dilation` history across calls: two single-position calls equal
    /// one two-position call, and a hand-computed value checks the taps.
    #[test]
    fn dilated_conv_taps_and_history() {
        let dev = Device::Cpu;
        let (n, d, k, dil) = (1usize, 2usize, 4usize, 3usize);
        let ch = n * d;
        // weight [ch, k]: channel 0 taps [1, 2, 3, 4], channel 1 taps [0,0,0,1].
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4., 0., 0., 0., 1.], (ch, k), &dev).unwrap();
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let layer = PleLayer::new(
            zero(ch, 4),
            zero(d, 4),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            w,
            n,
            d,
            dil,
            1e-6,
        )
        .unwrap();
        assert_eq!(layer.history_len(), 9);
        // x over 12 positions: channel 0 = position index, channel 1 = 10·index.
        let seq = 12;
        let mut xv = vec![0f32; seq * ch];
        for t in 0..seq {
            xv[t * ch] = t as f32;
            xv[t * ch + 1] = 10.0 * t as f32;
        }
        let x = Tensor::from_vec(xv, (1, seq, ch), &dev).unwrap();
        let mut st = RecurrentState::new(1);
        let y = layer.dilated_conv(&x, &mut st, 0).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t = 11, channel 0: 1·x[2] + 2·x[5] + 3·x[8] + 4·x[11] = 2 + 10 + 24 + 44 = 80.
        assert_eq!(yv[11 * ch], 80.0);
        // t = 11, channel 1: newest tap only = 10·11 = 110.
        assert_eq!(yv[11 * ch + 1], 110.0);
        // t = 1, channel 0: taps at -8, -5, -2 are zero history; 4·x[1] = 4.
        assert_eq!(yv[ch], 4.0);
        // Position by position from a fresh state equals the one-shot call.
        let mut st2 = RecurrentState::new(1);
        for t in 0..seq {
            let xt = x.narrow(1, t, 1).unwrap();
            let yt = layer.dilated_conv(&xt, &mut st2, 0).unwrap();
            let v = yt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(v, yv[t * ch..(t + 1) * ch].to_vec(), "position {t}");
        }
    }

    /// The PLE block on Metal equals the CPU block at the real widths (n=4,
    /// d=2560, 16 heads × 160): same weights, same 3-position chunk with a
    /// carried conv history, delta within f32 noise.
    #[cfg(feature = "metal")]
    #[test]
    fn ple_block_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, ple_dim) = (4usize, 2560usize, 2560usize);
        let ch = n * d;
        let build = |dev: &Device| {
            let lin = |o: usize, i: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..o * i)
                            .map(|x| ((x as f32 * 0.0037 + seed).sin()) * 0.02)
                            .collect::<Vec<_>>(),
                        (o, i),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let vecw = |k: usize, f: &dyn Fn(usize) -> f32| {
                Tensor::from_vec((0..k).map(f).collect::<Vec<f32>>(), k, dev).unwrap()
            };
            PleLayer::new(
                lin(ch, ple_dim, 1.0),
                lin(d, ple_dim, 2.0),
                vecw(ch, &|i| 1.0 + 0.001 * (i % 9) as f32),
                vecw(ch, &|i| 1.0 - 0.001 * (i % 7) as f32),
                vecw(ch, &|i| 1.0 + 0.002 * (i % 5) as f32),
                Tensor::from_vec(
                    (0..ch * 4)
                        .map(|x| (x as f32 * 0.1).cos() * 0.3)
                        .collect::<Vec<_>>(),
                    (ch, 4),
                    dev,
                )
                .unwrap(),
                n,
                d,
                3,
                1e-6,
            )
            .unwrap()
        };
        let (b_cpu, b_gpu) = (build(&cpu), build(&gpu));
        let seq = 3;
        let hidden: Vec<f32> = (0..seq * ch).map(|x| (x as f32 * 0.002).cos()).collect();
        let emb: Vec<f32> = (0..seq * ple_dim)
            .map(|x| (x as f32 * 0.011).sin())
            .collect();
        let h_cpu = Tensor::from_vec(hidden.clone(), (1, seq, ch), &cpu).unwrap();
        let h_gpu = Tensor::from_vec(hidden, (1, seq, ch), &gpu).unwrap();
        let e_cpu = Tensor::from_vec(emb.clone(), (seq, ple_dim), &cpu).unwrap();
        let e_gpu = Tensor::from_vec(emb, (seq, ple_dim), &gpu).unwrap();
        let (mut sc, mut sg) = (RecurrentState::new(1), RecurrentState::new(1));
        let dc = b_cpu.forward(&h_cpu, &e_cpu, &mut sc, 0).unwrap();
        let dg = b_gpu.forward(&h_gpu, &e_gpu, &mut sg, 0).unwrap();
        // and once more, so the carried history is exercised on both
        let dc2 = b_cpu.forward(&h_cpu, &e_cpu, &mut sc, 0).unwrap();
        let dg2 = b_gpu.forward(&h_gpu, &e_gpu, &mut sg, 0).unwrap();
        for (name, a, b) in [("first", &dc, &dg), ("second", &dc2, &dg2)] {
            let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vb = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                vb.iter().filter(|x| !x.is_finite()).count(),
                0,
                "{name}: non-finite on Metal"
            );
            let scale = va.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 2e-3 * scale,
                "{name}: Metal/CPU differ by {md} at scale {scale}"
            );
        }
    }

    /// Whole block on a tiny dense table: shapes thread, output finite, and the
    /// table row actually matters (two different tokens → different deltas)
    /// while a zero-weight block is a zero delta.
    #[test]
    fn ple_layer_threads_end_to_end() {
        let dev = Device::Cpu;
        let (n, d, ple_dim, heads) = (2usize, 4usize, 8usize, 2usize);
        let head_dim = ple_dim / heads;
        let ch = n * d;
        let lin = |o: usize, i: usize, seed: f32| {
            QMatMul::Tensor(
                Tensor::from_vec(
                    (0..o * i)
                        .map(|x| ((x as f32 * 0.37 + seed).sin()) * 0.5)
                        .collect::<Vec<_>>(),
                    (o, i),
                    &dev,
                )
                .unwrap(),
            )
        };
        let layer = PleLayer::new(
            lin(ch, ple_dim, 1.0),
            lin(d, ple_dim, 2.0),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::from_vec(
                (0..ch * 4)
                    .map(|x| (x as f32 * 0.1).cos() * 0.3)
                    .collect::<Vec<_>>(),
                (ch, 4),
                &dev,
            )
            .unwrap(),
            n,
            d,
            3,
            1e-6,
        )
        .unwrap();
        let rows = 50usize;
        let table = PleTable::Dense(
            Tensor::from_vec(
                (0..rows * head_dim)
                    .map(|x| (x as f32 * 0.11).sin())
                    .collect::<Vec<_>>(),
                (rows, head_dim),
                &dev,
            )
            .unwrap(),
        );
        let hidden = Tensor::from_vec(
            (0..ch).map(|x| (x as f32 * 0.2).cos()).collect::<Vec<_>>(),
            (1, 1, ch),
            &dev,
        )
        .unwrap();
        let mut st = RecurrentState::new(1);
        let emb_a = table
            .gather(&[3, 17], &dev)
            .unwrap()
            .reshape((1, ple_dim))
            .unwrap();
        let emb_b = table
            .gather(&[4, 17], &dev)
            .unwrap()
            .reshape((1, ple_dim))
            .unwrap();
        let da = layer.forward(&hidden, &emb_a, &mut st, 0).unwrap();
        assert_eq!(da.dims(), &[1, 1, ch]);
        let mut st_b = RecurrentState::new(1);
        let db = layer.forward(&hidden, &emb_b, &mut st_b, 0).unwrap();
        let (va, vb) = (
            da.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            db.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(va.iter().all(|x| x.is_finite()));
        assert!(
            va.iter().zip(&vb).any(|(a, b)| (a - b).abs() > 1e-6),
            "the gathered row must reach the output"
        );
        assert!(st.get(0).is_some(), "the conv history advanced");
    }
}
