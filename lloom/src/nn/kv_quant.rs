//! The quantized arm of an attention layer's KV block (`DESIGN.md` §10.7).
//!
//! One sequence's K and V as integer codes plus one (scale, bias) pair per head
//! vector, at 8 or 4 bits ([`KvQuant`]), in planes allocated at a capacity and
//! written at a moving offset. The two reads are methods of the block: the
//! attention over every live position, and over a selection of cells (QSA's
//! indexer), both straight from the codes; [`QuantKvSlab::dequantize`] is the
//! dense read a multi-token pass attends through.
//!
//! Metal only: the append and both reads are Metal kernels
//! (`quantize_kv`, `sdpa_vector_quant[_sel]`, `dequantize_kv`).

use crate::nn::ops::{self, KvQuant};
use lloom_tensor::{DType, Result, Tensor};

/// A quantized KV slab for one sequence: `n_heads` KV heads of `head_dim`.
///
/// The planes are flat — codes `[heads · capacity · code_bytes(head_dim)]`
/// u8, scales and biases `[heads · capacity]` at the activation dtype — so a
/// head's stride is its capacity and the kernels' strides come from it. They
/// are allocated at the first append (the dtype and the head geometry are the
/// source's) and widened by `grow_by` tokens when full.
#[derive(Debug, Clone)]
pub struct QuantKvSlab {
    quant: KvQuant,
    k_codes: Option<Tensor>,
    v_codes: Option<Tensor>,
    k_scales: Option<Tensor>,
    k_biases: Option<Tensor>,
    v_scales: Option<Tensor>,
    v_biases: Option<Tensor>,
    n_heads: usize,
    head_dim: usize,
    dtype: DType,
    capacity: usize,
    grow_by: usize,
    len: usize,
}

impl QuantKvSlab {
    /// An empty slab of `capacity` tokens, widened by `grow_by` when full.
    pub fn new(quant: KvQuant, capacity: usize, grow_by: usize) -> Result<Self> {
        if !quant.is_on() {
            lloom_tensor::bail!("QuantKvSlab on the Off arm: the dense cache is that arm")
        }
        Ok(Self {
            quant,
            k_codes: None,
            v_codes: None,
            k_scales: None,
            k_biases: None,
            v_scales: None,
            v_biases: None,
            n_heads: 0,
            head_dim: 0,
            dtype: DType::F32,
            capacity: capacity.max(1),
            grow_by: grow_by.max(1),
            len: 0,
        })
    }

    pub fn quant(&self) -> KvQuant {
        self.quant
    }

    /// Live positions.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Token capacity of the planes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes the six planes hold (0 before the first append).
    pub fn bytes(&self) -> usize {
        self.planes()
            .into_iter()
            .flatten()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    fn planes(&self) -> [&Option<Tensor>; 6] {
        [
            &self.k_codes,
            &self.v_codes,
            &self.k_scales,
            &self.k_biases,
            &self.v_scales,
            &self.v_biases,
        ]
    }

    /// Bytes an append of `n` tokens would allocate: the first allocation, or
    /// the widening (the old planes are live beside the new ones while the
    /// copy runs — counted). `per_token` is [`ops::KvPrecision::bytes_per_token`]'s.
    pub fn grow_bytes_for(&self, n: usize, per_token: usize) -> usize {
        if self.len + n <= self.capacity {
            return if self.k_codes.is_none() {
                self.capacity * per_token
            } else {
                0
            };
        }
        let steps = (self.len + n - self.capacity).div_ceil(self.grow_by);
        let wider = self.capacity + steps * self.grow_by;
        if self.k_codes.is_none() {
            wider * per_token
        } else {
            (wider + self.capacity) * per_token
        }
    }

    /// Keep the first `len` positions (a rejected speculative tail): a length
    /// decrement, the codes above it unreachable.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for p in [
            &mut self.k_codes,
            &mut self.v_codes,
            &mut self.k_scales,
            &mut self.k_biases,
            &mut self.v_scales,
            &mut self.v_biases,
        ] {
            *p = None;
        }
    }

    fn code_row(&self, capacity: usize) -> Result<usize> {
        self.quant
            .code_bytes(capacity * self.head_dim)
            .ok_or_else(|| lloom_tensor::Error::msg("QuantKvSlab on the Off arm"))
    }

    /// Allocate the planes for `n_heads × head_dim` at `dtype`.
    fn ensure(
        &mut self,
        n_heads: usize,
        head_dim: usize,
        dtype: DType,
        dev: &lloom_tensor::Device,
    ) -> Result<()> {
        if self.k_codes.is_some() {
            if (n_heads, head_dim, dtype) != (self.n_heads, self.head_dim, self.dtype) {
                lloom_tensor::bail!(
                    "QuantKvSlab: an append of {n_heads}×{head_dim} {dtype:?} into planes of {}×{} {:?}",
                    self.n_heads,
                    self.head_dim,
                    self.dtype
                )
            }
            return Ok(());
        }
        self.n_heads = n_heads;
        self.head_dim = head_dim;
        self.dtype = dtype;
        let codes = n_heads * self.code_row(self.capacity)?;
        let scales = n_heads * self.capacity;
        // Two allocations per plane: a `clone` of a tensor shares its storage.
        self.k_codes = Some(Tensor::zeros(codes, DType::U8, dev)?);
        self.v_codes = Some(Tensor::zeros(codes, DType::U8, dev)?);
        self.k_scales = Some(Tensor::zeros(scales, dtype, dev)?);
        self.k_biases = Some(Tensor::zeros(scales, dtype, dev)?);
        self.v_scales = Some(Tensor::zeros(scales, dtype, dev)?);
        self.v_biases = Some(Tensor::zeros(scales, dtype, dev)?);
        Ok(())
    }

    /// Widen the planes to `capacity` tokens, every head's positions kept at
    /// the head of its wider row (one strided copy per plane).
    fn grow_to(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        let (old_row, new_row) = (self.code_row(self.capacity)?, self.code_row(capacity)?);
        let heads = self.n_heads;
        let old_cap = self.capacity;
        let widen = |plane: &mut Option<Tensor>, old: usize, new: usize| -> Result<()> {
            if let Some(p) = plane.as_ref() {
                let src = p.reshape((heads, old))?;
                let grown = Tensor::zeros((heads, new), p.dtype(), p.device())?;
                grown.slice_set(&src, 1, 0)?;
                *plane = Some(grown.reshape(heads * new)?);
            }
            Ok(())
        };
        widen(&mut self.k_codes, old_row, new_row)?;
        widen(&mut self.v_codes, old_row, new_row)?;
        for p in [
            &mut self.k_scales,
            &mut self.k_biases,
            &mut self.v_scales,
            &mut self.v_biases,
        ] {
            widen(p, old_cap, capacity)?;
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Quantize `k`, `v` `[n_heads, n, head_dim]` and append them.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (h, n, d) = k.dims3()?;
        if v.dims3()? != (h, n, d) || v.dtype() != k.dtype() {
            lloom_tensor::bail!("QuantKvSlab::append: k {:?} v {:?}", k.shape(), v.shape())
        }
        self.ensure(h, d, k.dtype(), k.device())?;
        if self.len + n > self.capacity {
            let steps = (self.len + n - self.capacity).div_ceil(self.grow_by);
            self.grow_to(self.capacity + steps * self.grow_by)?;
        }
        for (src, codes, scales, biases) in [
            (k, &self.k_codes, &self.k_scales, &self.k_biases),
            (v, &self.v_codes, &self.v_scales, &self.v_biases),
        ] {
            ops::quantize_kv_into(
                &src.reshape((1, h, n, d))?.contiguous()?,
                codes.as_ref().expect("allocated by ensure"),
                scales.as_ref().expect("allocated by ensure"),
                biases.as_ref().expect("allocated by ensure"),
                self.capacity,
                self.len,
                self.quant,
            )?;
        }
        self.len += n;
        Ok(())
    }

    fn planes_or_bail(&self) -> Result<[&Tensor; 6]> {
        match (
            &self.k_codes,
            &self.k_scales,
            &self.k_biases,
            &self.v_codes,
            &self.v_scales,
            &self.v_biases,
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) if self.len > 0 => {
                Ok([a, b, c, d, e, f])
            }
            _ => lloom_tensor::bail!("QuantKvSlab: a read of an empty slab"),
        }
    }

    /// Attention of `q [1, n_q_heads, 1, head_dim]` over the live positions,
    /// or over the cells `sel` (`u32 [n_sel]`, each `< len`) when given —
    /// read from the codes, no dequantized copy. → `[1, n_q_heads, 1, head_dim]`.
    pub fn attend(&self, q: &Tensor, sel: Option<&Tensor>, scale: f32) -> Result<Tensor> {
        let [kc, ks, kb, vc, vs, vb] = self.planes_or_bail()?;
        let n_q = q.dim(1)?;
        if self.n_heads == 0 || n_q % self.n_heads != 0 {
            lloom_tensor::bail!(
                "QuantKvSlab::attend: {n_q} query heads over {} KV heads",
                self.n_heads
            )
        }
        let gqa = n_q / self.n_heads;
        match sel {
            None => ops::sdpa_quant(
                q,
                kc,
                ks,
                kb,
                vc,
                vs,
                vb,
                gqa,
                self.len,
                self.capacity,
                scale,
                1.0,
                self.quant,
            ),
            Some(sel) => ops::sdpa_quant_selected(
                q,
                kc,
                ks,
                kb,
                vc,
                vs,
                vb,
                sel,
                gqa,
                self.capacity,
                scale,
                1.0,
                self.quant,
            ),
        }
    }

    /// The live K and V dequantized, `[n_heads, len, head_dim]` each at the
    /// planes' dtype (the scale dtype is the output's: the kernel reads both as
    /// one `T`) — the read a multi-token pass attends through, so it sees the
    /// cache the decode steps will see.
    pub fn dequantize(&self) -> Result<(Tensor, Tensor)> {
        let dtype = self.dtype;
        let [kc, ks, kb, vc, vs, vb] = self.planes_or_bail()?;
        let (h, d) = (self.n_heads, self.head_dim);
        let read = |c: &Tensor, s: &Tensor, b: &Tensor| -> Result<Tensor> {
            ops::dequantize_kv(c, s, b, 1, h, self.len, d, self.capacity, dtype, self.quant)?
                .reshape((h, self.len, d))
        };
        Ok((read(kc, ks, kb)?, read(vc, vs, vb)?))
    }
}
