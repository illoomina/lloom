//! The quantized arm of an attention layer's KV block (`DESIGN.md` §10.7).
//!
//! One sequence's — or a batch's, in lockstep — K and V as integer codes plus
//! one (scale, bias) pair per head vector, at 8 or 4 bits ([`KvQuant`]), in
//! planes allocated at a capacity and written at a moving offset. The two
//! reads are methods of the block: the attention over every live position,
//! and over a selection of cells (QSA's indexer), both straight from the
//! codes; [`QuantKvSlab::dequantize`] is the dense read a multi-token pass
//! attends through, so a prefill sees exactly the cache decode will see.
//!
//! Whether the planes widen on their own ([`Grow::By`], Qwen4-exp's) or only
//! when the owner widens them before the pass ([`Grow::Fixed`], LFM2's
//! bucket, admitted first) is the rule, not the type — `nn::slab::Slab`'s
//! split. The scales' dtype is the source's unless pinned
//! ([`QuantKvSlab::scales_at`]): a dtype outside the per-token reduction,
//! computed once at write time (§10.7a), so it needs no determinism variant.
//!
//! Metal only: the append and both reads are Metal kernels
//! (`quantize_kv`, `sdpa_vector_quant[_sel]`, `dequantize_kv`).

use crate::nn::ops::{self, KvQuant};
use crate::nn::slab::Grow;
use crate::nn::zeros as zs;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The zeros site the planes take when the owner names none
/// (`nn::zeros`): never poisoned, read as zeros by every owner.
pub const NN_KV_QUANT: &str = "nn.kv_quant.planes";

/// A quantized KV slab: `b_sz × n_heads` KV heads of `head_dim`, the batch
/// known at the first append.
///
/// The planes are flat — codes `[b · heads · capacity · code_bytes(head_dim)]`
/// u8, scales and biases `[b · heads · capacity]` — so a head's stride is the
/// capacity and the kernels' strides come from it. A parallel scale plane
/// rather than a scale beside each vector keeps `sdpa_vector`'s contiguous
/// walk (§9.1d).
#[derive(Debug, Clone)]
pub struct QuantKvSlab {
    quant: KvQuant,
    k_codes: Option<Tensor>,
    v_codes: Option<Tensor>,
    k_scales: Option<Tensor>,
    k_biases: Option<Tensor>,
    v_scales: Option<Tensor>,
    v_biases: Option<Tensor>,
    b_sz: usize,
    n_heads: usize,
    head_dim: usize,
    /// The scales' dtype once allocated.
    dtype: DType,
    /// The scales' dtype when pinned; otherwise the source's.
    scale_dtype: Option<DType>,
    capacity: usize,
    grow: Grow,
    len: usize,
    site: &'static str,
}

impl QuantKvSlab {
    /// An empty slab of `capacity` tokens, widened by `grow_by` when full.
    pub fn new(quant: KvQuant, capacity: usize, grow_by: usize) -> Result<Self> {
        if !quant.is_on() {
            lloom_tensor::bail!("QuantKvSlab on the Off arm: the dense cache is that arm")
        }
        Ok(Self::with_grow(quant, capacity, Grow::By(grow_by.max(1))))
    }

    /// An empty slab of `capacity` tokens that the owner widens
    /// ([`Self::grow_to`]) before a pass appends; an append past the capacity
    /// is refused. Constructible on the `Off` arm (it allocates nothing);
    /// an append there is refused.
    pub fn fixed(quant: KvQuant, capacity: usize) -> Self {
        Self::with_grow(quant, capacity, Grow::Fixed)
    }

    fn with_grow(quant: KvQuant, capacity: usize, grow: Grow) -> Self {
        Self {
            quant,
            k_codes: None,
            v_codes: None,
            k_scales: None,
            k_biases: None,
            v_scales: None,
            v_biases: None,
            b_sz: 0,
            n_heads: 0,
            head_dim: 0,
            dtype: DType::F32,
            scale_dtype: None,
            capacity: capacity.max(1),
            grow,
            len: 0,
            site: NN_KV_QUANT,
        }
    }

    /// Pin the scales' and biases' dtype, whatever the source's.
    pub fn scales_at(mut self, dtype: DType) -> Self {
        self.scale_dtype = Some(dtype);
        self
    }

    /// The zeros site the planes are allocated at (`nn::zeros`).
    pub fn at_site(mut self, site: &'static str) -> Self {
        self.site = site;
        self
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

    /// Whether the planes exist (allocated at the first append).
    pub fn is_allocated(&self) -> bool {
        self.k_codes.is_some()
    }

    /// Token capacity of the planes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Set the capacity before the planes exist; a slab that is already
    /// allocated keeps its width (widen it with [`Self::grow_to`]).
    pub fn set_capacity(&mut self, capacity: usize) {
        if !self.is_allocated() {
            self.capacity = capacity.max(1);
        }
    }

    /// Bytes the six planes hold (0 before the first append).
    pub fn bytes(&self) -> usize {
        self.planes()
            .into_iter()
            .flatten()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    /// Bytes one token costs across K and V at the allocated geometry: the
    /// codes plus a scale and a bias per head vector, per batch row. 0
    /// before the first append.
    pub fn bytes_per_token(&self) -> usize {
        if !self.is_allocated() {
            return 0;
        }
        let rows = self.b_sz * self.n_heads;
        let codes = self.quant.code_bytes(self.head_dim).unwrap_or(0);
        2 * (rows * codes + 2 * rows * self.dtype.size_in_bytes())
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
        let wider = match self.grow {
            Grow::By(step) => {
                let steps = (self.len + n - self.capacity).div_ceil(step);
                self.capacity + steps * step
            }
            Grow::Fixed => self.capacity,
        };
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

    /// Discard the last `count` positions (§10.2a): a length decrement and
    /// nothing else, refused past the start.
    pub fn rewind(&mut self, count: usize) -> Result<()> {
        if count > self.len {
            lloom_tensor::bail!(
                "quantized KV rewind past the start: {count} > len {}",
                self.len
            )
        }
        self.len -= count;
        Ok(())
    }

    /// Drop the planes: the next append allocates them at its own batch.
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

    /// Record the geometry and allocate one side's planes, at `dtype` unless
    /// the scales are pinned. The first side records the geometry; the other
    /// must match.
    fn ensure(
        &mut self,
        side: Side,
        b_sz: usize,
        n_heads: usize,
        head_dim: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<()> {
        let dtype = self.scale_dtype.unwrap_or(dtype);
        if self.k_codes.is_some() || self.v_codes.is_some() {
            if (b_sz, n_heads, head_dim, dtype)
                != (self.b_sz, self.n_heads, self.head_dim, self.dtype)
            {
                lloom_tensor::bail!(
                    "QuantKvSlab: an append of {b_sz}×{n_heads}×{head_dim} {dtype:?} into planes of \
                     {}×{}×{} {:?}",
                    self.b_sz,
                    self.n_heads,
                    self.head_dim,
                    self.dtype
                )
            }
        } else {
            self.b_sz = b_sz;
            self.n_heads = n_heads;
            self.head_dim = head_dim;
            self.dtype = dtype;
        }
        let rows = b_sz * n_heads;
        let code_els = rows * self.code_row(self.capacity)?;
        let scale_els = rows * self.capacity;
        let site = self.site;
        let (codes, scales, biases) = match side {
            Side::K => (&mut self.k_codes, &mut self.k_scales, &mut self.k_biases),
            Side::V => (&mut self.v_codes, &mut self.v_scales, &mut self.v_biases),
        };
        if codes.is_some() {
            return Ok(());
        }
        // A u8 plane rather than a typed one: the packing is the kernel's, and
        // a `Tensor` of the code width does not exist at int4.
        *codes = Some(zs::zeros(site, code_els, DType::U8, dev)?);
        *scales = Some(zs::zeros(site, scale_els, dtype, dev)?);
        *biases = Some(zs::zeros(site, scale_els, dtype, dev)?);
        Ok(())
    }

    /// Widen the planes to `capacity` tokens, every row's positions kept at
    /// the head of its wider row (one strided copy per plane). Positions
    /// above `len` are carried too; they are unreachable either way.
    pub fn grow_to(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        if self.is_allocated() {
            let (old_row, new_row) = (self.code_row(self.capacity)?, self.code_row(capacity)?);
            let rows = self.b_sz * self.n_heads;
            let old_cap = self.capacity;
            let site = self.site;
            let widen = |plane: &mut Option<Tensor>, old: usize, new: usize| -> Result<()> {
                if let Some(p) = plane.as_ref() {
                    let src = p.reshape((rows, old))?;
                    let grown = zs::zeros(site, (rows, new), p.dtype(), p.device())?;
                    grown.slice_set(&src, 1, 0)?;
                    *plane = Some(grown.reshape(rows * new)?);
                }
                Ok(())
            };
            widen(&mut self.k_codes, old_row, new_row)?;
            for p in [&mut self.k_scales, &mut self.k_biases] {
                widen(p, old_cap, capacity)?;
            }
            widen(&mut self.v_codes, old_row, new_row)?;
            for p in [&mut self.v_scales, &mut self.v_biases] {
                widen(p, old_cap, capacity)?;
            }
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Quantize `k`, `v` — `[n_heads, n, head_dim]`, or `[b, n_heads, n,
    /// head_dim]` for a batch in lockstep — and append them. Past the
    /// capacity the slab widens ([`Grow::By`]) or refuses ([`Grow::Fixed`]).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (b, h, n, d) = match k.rank() {
            3 => {
                let (h, n, d) = k.dims3()?;
                (1, h, n, d)
            }
            _ => k.dims4()?,
        };
        if v.dims() != k.dims() || v.dtype() != k.dtype() {
            lloom_tensor::bail!("QuantKvSlab::append: k {:?} v {:?}", k.shape(), v.shape())
        }
        if self.len + n > self.capacity {
            match self.grow {
                Grow::By(step) => {
                    let steps = (self.len + n - self.capacity).div_ceil(step);
                    self.grow_to(self.capacity + steps * step)?;
                }
                Grow::Fixed => lloom_tensor::bail!(
                    "quantized KV cache exhausted: {} + {n} > capacity {}",
                    self.len,
                    self.capacity
                ),
            }
        }
        let k4 = k.reshape((b, h, n, d))?.contiguous()?;
        let v4 = v.reshape((b, h, n, d))?.contiguous()?;
        // K's planes then K's write, then V's: the first append allocates in
        // the order a per-side slot pair did.
        self.ensure(Side::K, b, h, d, k.dtype(), k.device())?;
        ops::quantize_kv_into(
            &k4,
            self.k_codes.as_ref().expect("allocated by ensure"),
            self.k_scales.as_ref().expect("allocated by ensure"),
            self.k_biases.as_ref().expect("allocated by ensure"),
            self.capacity,
            self.len,
            self.quant,
        )?;
        self.ensure(Side::V, b, h, d, v.dtype(), v.device())?;
        ops::quantize_kv_into(
            &v4,
            self.v_codes.as_ref().expect("allocated by ensure"),
            self.v_scales.as_ref().expect("allocated by ensure"),
            self.v_biases.as_ref().expect("allocated by ensure"),
            self.capacity,
            self.len,
            self.quant,
        )?;
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

    /// Attention of `q [b, n_q_heads, 1, head_dim]` over the live positions,
    /// or over the cells `sel` (`u32 [n_sel]`, each `< len`) when given —
    /// read from the codes, no dequantized copy. → `[b, n_q_heads, 1, head_dim]`.
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

    /// The live K and V dequantized to `dtype`, `[b, n_heads, len, head_dim]`
    /// each — the read a multi-token pass attends through, so it sees the
    /// cache the decode steps will see. The values are the codes', not the
    /// originals.
    pub fn dequantize_to(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        let [kc, ks, kb, vc, vs, vb] = self.planes_or_bail()?;
        let (b, h, d) = (self.b_sz, self.n_heads, self.head_dim);
        let read = |c: &Tensor, s: &Tensor, bi: &Tensor| -> Result<Tensor> {
            ops::dequantize_kv(
                c,
                s,
                bi,
                b,
                h,
                self.len,
                d,
                self.capacity,
                dtype,
                self.quant,
            )
        };
        Ok((read(kc, ks, kb)?, read(vc, vs, vb)?))
    }

    /// One sequence's live K and V dequantized at the planes' dtype (the
    /// scale dtype is the output's: the kernel reads both as one `T`),
    /// `[n_heads, len, head_dim]` each.
    pub fn dequantize(&self) -> Result<(Tensor, Tensor)> {
        if self.b_sz != 1 {
            lloom_tensor::bail!(
                "QuantKvSlab::dequantize: a batch of {}; `dequantize_to` keeps the batch",
                self.b_sz
            )
        }
        let (k, v) = self.dequantize_to(self.dtype)?;
        let (h, d) = (self.n_heads, self.head_dim);
        Ok((k.reshape((h, self.len, d))?, v.reshape((h, self.len, d))?))
    }
}

#[derive(Clone, Copy)]
enum Side {
    K,
    V,
}
