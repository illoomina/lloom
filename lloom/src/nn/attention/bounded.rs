//! Additive compact-key-mask policy. Existing callers retain their policies.
//! Query tiling bounds an individual dispatch, not outstanding asynchronous
//! work. The plan reports conservative whole-call allocation VOLUME; execution
//! still requires the caller's model-domain/pass admission and retirement rules.
use super::{flash_attn, AttnMask};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueWorkspace {
    /// Rayon/allocator metadata is outside numeric-payload accounting.
    CpuRuntime,
    /// Pipeline/compiler/driver and threadgroup working sets are unmeasured.
    MetalDriver,
    /// cuBLAS/runtime workspace is not represented by Tensor allocations here.
    CudaBlas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionScratch {
    /// Sum of possible temporary/output Tensor allocation bytes for ONE call.
    /// Includes all query tiles and concatenations, not only the largest tile.
    /// Excludes existing inputs; it is conservative, not a measured peak.
    pub tensor_allocation_volume: usize,
    /// Conservative CPU online-softmax accumulator allocation volume.
    pub host_accumulator_volume: usize,
    /// Unknown extra obligations are explicit, never encoded as zero bytes.
    pub opaque_workspace: OpaqueWorkspace,
}

/// Borrows immutable input geometry. Repeated `execute` calls are NEW work and
/// each needs outer pass coverage; this validated plan is not an entitlement.
/// Mask is additive `[B,1,1,K]`; full K/V remain present, including finite-masked
/// keys. At least one usable key per row is the caller's numerical contract.
/// Tensor values are not scanned/read back to validate that contract.
pub struct MaskedAttentionPlan<'a> {
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    mask: &'a Tensor,
    scale: f32,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    queries: usize,
    keys: usize,
    tile: usize,
    scratch: AttentionScratch,
    /// The fused route on Metal, checked against every tile at preparation.
    route: Option<super::route::PreparedAttention>,
}

fn mul(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("masked attention product/byte overflow".into()))
}
fn sum(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(0usize, |n, &d| n.checked_add(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            lloom_tensor::Error::Msg("masked attention allocation-volume overflow".into())
        })
}
fn i32_dim(value: usize) -> Result<()> {
    i32::try_from(value).map(|_| ()).map_err(|_| {
        lloom_tensor::Error::Msg("masked attention dimension exceeds I32 backend range".into())
    })
}
fn u32_count(value: usize) -> Result<()> {
    u32::try_from(value).map(|_| ()).map_err(|_| {
        lloom_tensor::Error::Msg("masked attention launch count exceeds U32 backend range".into())
    })
}

impl<'a> MaskedAttentionPlan<'a> {
    pub fn new(
        q: &'a Tensor,
        k: &'a Tensor,
        v: &'a Tensor,
        key_mask: &'a Tensor,
        scale: f64,
        query_tile: NonZeroUsize,
    ) -> Result<Self> {
        let (b, h, t, d) = q.dims4()?;
        let (kb, kh, k_len, kd) = k.dims4()?;
        let (vb, vh, v_len, vd) = v.dims4()?;
        if [b, h, t, d, kb, kh, k_len, kd, vb, vh, v_len, vd].contains(&0)
            || b != kb
            || b != vb
            || d != kd
            || d != vd
            || kh != vh
            || k_len != v_len
            || h % kh != 0
        {
            lloom_tensor::bail!("masked attention requires positive compatible B/H/T/K/D geometry and divisible GQA heads")
        }
        if !matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32)
            || [k, v, key_mask]
                .iter()
                .any(|x| x.dtype() != q.dtype() || !x.device().same_device(q.device()))
            || key_mask.dims() != [b, 1, 1, k_len]
        {
            lloom_tensor::bail!(
                "masked attention requires common F16/BF16/F32 dtype/device and [B,1,1,K] mask"
            )
        }
        if !scale.is_finite() || !(scale as f32).is_finite() {
            lloom_tensor::bail!("masked attention scale must be finite and representable in F32")
        }
        // All native paths below use at least one I32 dimension. CPU also
        // computes a U32 power-of-two head bucket, even with ALiBi disabled.
        for n in [b, h, kh, t, k_len, d] {
            i32_dim(n)?;
        }
        let e = q.dtype().size_in_bytes();
        for x in [q, k, v, key_mask] {
            // Shape::elem_count uses an unchecked product; broadcast views can
            // have enormous logical shapes over one scalar of actual storage.
            let elements = mul(x.dims())?;
            mul(&[elements, e])?;
            let mut last = x.layout().start_offset();
            for (&extent, &stride) in x.dims().iter().zip(x.stride()) {
                i64::try_from(stride).map_err(|_| {
                    lloom_tensor::Error::Msg("masked attention stride exceeds I64".into())
                })?;
                last = last
                    .checked_add(mul(&[extent - 1, stride])?)
                    .ok_or_else(|| {
                        lloom_tensor::Error::Msg("masked attention view offset overflow".into())
                    })?;
            }
            mul(&[
                last.checked_add(1).ok_or_else(|| {
                    lloom_tensor::Error::Msg("masked attention view extent overflow".into())
                })?,
                e,
            ])?;
        }
        // Metal full-mask dispatch requires q_tile<=K. This changes only tile
        // size, not key membership or the attention formula, for T>K callers.
        let tile = query_tile.get().min(t).min(k_len);
        let output = mul(&[b, h, t, d, e])?;
        let output_f32 = mul(&[b, h, t, d, 4])?;
        let concat = mul(&[2, output])?;
        let half_cast = if q.dtype() == DType::F32 { 0 } else { output };
        let mut route = None;
        let scratch = match q.device() {
            Device::Cpu => {
                if q.stride()[3] != 1 || k.stride()[3] != 1 {
                    lloom_tensor::bail!("CPU masked attention requires unit Q/K channel stride; normalize under caller admission")
                }
                AttentionScratch {
                    tensor_allocation_volume: sum(&[
                        mul(&[b, t, k_len, e])?,
                        output_f32,
                        half_cast,
                        concat,
                    ])?,
                    host_accumulator_volume: output_f32,
                    opaque_workspace: OpaqueWorkspace::CpuRuntime,
                }
            }
            Device::Metal(_) => {
                route = Some(metal_route(q, h, kh, d, tile, k_len)?);
                if [q, k, v].iter().any(|x| x.stride()[3] != 1) {
                    lloom_tensor::bail!(
                        "masked Metal full SDPA requires unit Q/K/V channel strides"
                    )
                }
                AttentionScratch {
                    tensor_allocation_volume: sum(&[output, concat])?,
                    host_accumulator_volume: 0,
                    opaque_workspace: OpaqueWorkspace::MetalDriver,
                }
            }
            Device::Cuda(_) => {
                // Covers cast/affine/copy launches, cuBLAS batch counts and
                // softmax rows/columns before any conversion allocates.
                for count in [
                    mul(&[b, h, t, d])?,
                    mul(&[b, h, k_len, d])?,
                    mul(&[h, tile, k_len])?,
                    mul(&[h, tile])?,
                    mul(&[b, k_len])?,
                ] {
                    u32_count(count)?;
                }
                let kv_normalize = mul(&[4, b, kh, k_len, d, 4])?;
                let kv_repeat = if h == kh {
                    0
                } else {
                    mul(&[2, b, h, k_len, d, 4])?
                };
                let q_normalize = mul(&[2, b, h, t, d, 4])?;
                let mask_cast = mul(&[b, k_len, 4])?;
                // Score, scaled score, additive-mask score, softmax output.
                // Count ALL tiles, even if native completion retires some early.
                let scores = mul(&[4, b, h, t, k_len, 4])?;
                AttentionScratch {
                    tensor_allocation_volume: sum(&[
                        kv_normalize,
                        kv_repeat,
                        q_normalize,
                        mask_cast,
                        scores,
                        output_f32,
                        half_cast,
                        concat,
                    ])?,
                    host_accumulator_volume: 0,
                    opaque_workspace: OpaqueWorkspace::CudaBlas,
                }
            }
        };
        Ok(Self {
            q,
            k,
            v,
            mask: key_mask,
            scale: scale as f32,
            batch: b,
            heads: h,
            kv_heads: kh,
            queries: t,
            keys: k_len,
            tile,
            scratch,
            route,
        })
    }

    pub fn scratch(&self) -> AttentionScratch {
        self.scratch
    }

    pub fn execute(&self) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            let k = self.k.narrow(0, row, 1)?;
            let v = self.v.narrow(0, row, 1)?;
            let mask = self.mask.narrow(0, row, 1)?;
            let cuda = matches!(self.q.device(), Device::Cuda(_));
            // CUDA explicit-math policy uses F32 accumulation for all accepted
            // dtypes, then explicitly casts the output. Other backends retain
            // native streaming/fused mechanisms; parity is separately gated.
            let (k, v, mask) = if cuda {
                (
                    crate::utils::repeat_kv(
                        k.to_dtype(DType::F32)?.contiguous()?,
                        self.heads / self.kv_heads,
                    )?,
                    crate::utils::repeat_kv(
                        v.to_dtype(DType::F32)?.contiguous()?,
                        self.heads / self.kv_heads,
                    )?,
                    mask.to_dtype(DType::F32)?,
                )
            } else {
                (k, v, mask)
            };
            let mut chunks = Vec::with_capacity(self.queries.div_ceil(self.tile));
            for start in (0..self.queries).step_by(self.tile) {
                let n = self.tile.min(self.queries - start);
                let q = self.q.narrow(0, row, 1)?.narrow(2, start, n)?;
                let output = match self.q.device() {
                    Device::Cpu => {
                        let q = q.transpose(1, 2)?;
                        let k = k.transpose(1, 2)?;
                        let v = v.transpose(1, 2)?;
                        // Kernel flattens T*K, not H*T*K. Contiguous
                        // materialization occurs inside the existing CPU op.
                        let m = AttnMask::Mask(mask.broadcast_as((1, 1, n, self.keys))?);
                        let out = match q.dtype() {
                            DType::F32 => flash_attn::<f32>(&q, &k, &v, self.scale, m, None, None)?,
                            DType::F16 => {
                                flash_attn::<half::f16>(&q, &k, &v, self.scale, m, None, None)?
                            }
                            DType::BF16 => {
                                flash_attn::<half::bf16>(&q, &k, &v, self.scale, m, None, None)?
                            }
                            _ => unreachable!("validated dtype"),
                        };
                        out.to_dtype(self.q.dtype())?
                    }
                    Device::Metal(_) => crate::nn::ops::attention(
                        &q,
                        &k,
                        &v,
                        Some(&mask.broadcast_as((1, self.heads, n, self.keys))?),
                        self.scale,
                        self.route
                            .as_ref()
                            .expect("a Metal plan prepares its route"),
                    )?,
                    Device::Cuda(_) => {
                        let q = q.to_dtype(DType::F32)?.contiguous()?;
                        let scores = q.matmul(&k.transpose(2, 3)?)?;
                        let scaled = (scores * self.scale as f64)?;
                        let masked = scaled.broadcast_add(&mask)?;
                        let probabilities = crate::nn::ops::softmax_last_dim(&masked)?;
                        probabilities.matmul(&v)?.to_dtype(self.q.dtype())?
                    }
                };
                chunks.push(output);
            }
            rows.push(Tensor::cat(&chunks, 2)?);
        }
        Tensor::cat(&rows, 0)
    }
}

/// The tiled kernel under an additive mask, one sequence a dispatch, up to
/// `tile` queries over `keys`: a head dim or dtype it does not instantiate is
/// refused here, before anything is allocated.
fn metal_route(
    q: &Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tile: usize,
    keys: usize,
) -> Result<super::route::PreparedAttention> {
    #[cfg(feature = "metal")]
    {
        use super::route::{
            AttentionBounds, AttentionSpec, KvLayout, MaskKind, MetalKernelSet, PreparedAttention,
        };
        let spec = AttentionSpec {
            q_heads: heads,
            kv_heads,
            head_dim,
            v_head_dim: head_dim,
            dtype: q.dtype(),
            kv_dtype: q.dtype(),
            mask: MaskKind::Additive,
            softcap: 1.0,
            layout: KvLayout::Strided,
            bounds: AttentionBounds {
                batch: 1,
                min_queries: 1,
                queries: tile,
                keys,
            },
        };
        PreparedAttention::prepare(
            spec,
            &MetalKernelSet {
                device: q.device().as_metal_device()?,
            },
        )
        .map_err(|u| lloom_tensor::Error::Msg(format!("masked Metal attention: {u}")))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (q, heads, kv_heads, head_dim, tile, keys);
        lloom_tensor::bail!("masked Metal attention without the metal feature")
    }
}

#[cfg(test)]
mod tests;
