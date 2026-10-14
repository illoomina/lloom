//! Attention routing: one prepared description of which fused kernel serves
//! an attention layer and what memory each call takes, read by the dispatch
//! and by admission alike (`docs/dependency-ordered-reuse-design.md` §8).
//!
//! A route is prepared once, at load, from the layer's geometry and the
//! device's kernel set. A shape no fused kernel serves is refused there, with
//! the missing capability named; nothing falls back to the composed scores
//! path. A request is validated against the route before anything is
//! allocated.

use lloom_tensor::{DType, Result};

/// How a layer masks its keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    /// Every query attends to every key.
    None,
    /// The queries are the last `queries` positions of the key range and each
    /// attends to keys up to its own position.
    Causal,
    /// An additive mask `[batch, q_heads, queries, keys]` in the layer dtype.
    Additive,
}

/// Where keys and values are read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLayout {
    /// `[batch, kv_heads, keys, head_dim]` through strides.
    Strided,
    /// One shared pool read through per-sequence tables of `page_size` keys.
    Paged { page_size: usize },
}

/// The requests a prepared route serves. `usize::MAX` leaves an axis
/// unbounded by the route: requests are still checked, and the worst case is
/// admission's to bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBounds {
    pub batch: usize,
    /// The fewest queries a request carries: 1 when the layer decodes, more
    /// when it only ever runs windows (a paged verify).
    pub min_queries: usize,
    pub queries: usize,
    pub keys: usize,
}

/// The attention a layer computes, fixed at load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSpec {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    /// The dtype the kernel computes in: the queries' and the output's.
    pub dtype: DType,
    /// The dtype keys and values are stored in; converted to `dtype` per call
    /// when it differs.
    pub kv_dtype: DType,
    pub mask: MaskKind,
    /// The `tanh` soft cap on the scores; 1.0 is none.
    pub softcap: f32,
    pub layout: KvLayout,
    pub bounds: AttentionBounds,
}

/// A fused kernel family that serves attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionKernel {
    /// `sdpa_vector`: one query, no mask tensor, single pass.
    Vector,
    /// `sdpa_vector_2pass`: as `Vector`, split over key blocks.
    VectorTwoPass,
    /// `steel_attention`: any query count, causal or additive mask in-kernel.
    Full,
}

impl AttentionKernel {
    pub fn name(self) -> &'static str {
        match self {
            AttentionKernel::Vector => "sdpa_vector",
            AttentionKernel::VectorTwoPass => "sdpa_vector_2pass",
            AttentionKernel::Full => "steel_attention",
        }
    }
}

/// What the device's kernels instantiate and can run, asked at load only.
pub trait KernelSet {
    /// Whether the one-query kernels run `head_dim` at `dtype` on this device,
    /// both passes included.
    fn vector(&self, head_dim: usize, dtype: DType) -> bool;
    /// The key block of the tiled kernel at `head_dim` and `dtype`, or `None`
    /// when it is not instantiated there.
    fn full_key_block(&self, head_dim: usize, dtype: DType) -> Option<usize>;
    /// The key count from which a one-query call takes the two-pass kernel.
    fn two_pass_from(&self, head_dim: usize, dtype: DType) -> usize;
    /// The key blocks the two-pass kernel splits a query's keys into.
    fn two_pass_blocks(&self) -> usize;
}

/// A capability a shape needs and no kernel provides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Uncovered {
    pub phase: &'static str,
    pub need: String,
}

impl std::fmt::Display for Uncovered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} attention: {}", self.phase, self.need)
    }
}

/// The memory one call takes beyond its inputs, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionWorkspace {
    /// The output `[batch, q_heads, queries, v_head_dim]`.
    pub output: usize,
    /// The two-pass kernel's partial outputs, maxima and sums, in f32.
    pub scratch: usize,
    /// Keys and values converted to the compute dtype.
    pub conversion: usize,
    /// The additive mask at its full logical extent, an upper bound on what
    /// a caller materializes for it.
    pub mask: usize,
    /// The paged layout's page table and live-key counts.
    pub metadata: usize,
}

impl AttentionWorkspace {
    pub fn total(&self) -> Option<usize> {
        [self.scratch, self.conversion, self.mask, self.metadata]
            .into_iter()
            .try_fold(self.output, usize::checked_add)
    }

    fn max(self, other: AttentionWorkspace) -> AttentionWorkspace {
        AttentionWorkspace {
            output: self.output.max(other.output),
            scratch: self.scratch.max(other.scratch),
            conversion: self.conversion.max(other.conversion),
            mask: self.mask.max(other.mask),
            metadata: self.metadata.max(other.metadata),
        }
    }
}

/// A route prepared for one layer: the kernel each request shape takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedAttention {
    spec: AttentionSpec,
    /// What a one-query request takes, keys below `two_pass_from`, when the
    /// bounds admit one.
    single: Option<AttentionKernel>,
    /// Keys from which an unmasked one-query request takes the two-pass
    /// kernel, and the key blocks it splits them into.
    two_pass_from: Option<usize>,
    two_pass_blocks: usize,
    /// What a request of more than one query takes, when the bounds admit one.
    multi: Option<AttentionKernel>,
}

fn checked(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("attention workspace overflows".into()))
}

impl PreparedAttention {
    /// Route `spec` against `kernels`, refusing any request shape within the
    /// bounds that no fused kernel serves.
    pub fn prepare(
        spec: AttentionSpec,
        kernels: &impl KernelSet,
    ) -> std::result::Result<Self, Uncovered> {
        let b = spec.bounds;
        if b.batch == 0 || b.min_queries == 0 || b.queries < b.min_queries || b.keys == 0 {
            return Err(Uncovered {
                phase: "any",
                need: format!("non-zero bounds, got {b:?}"),
            });
        }
        if spec.q_heads == 0 || spec.kv_heads == 0 || !spec.q_heads.is_multiple_of(spec.kv_heads) {
            return Err(Uncovered {
                phase: "any",
                need: format!(
                    "q heads ({}) a multiple of kv heads ({})",
                    spec.q_heads, spec.kv_heads
                ),
            });
        }
        if !matches!(spec.dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(Uncovered {
                phase: "any",
                need: format!("a fused kernel at {:?}", spec.dtype),
            });
        }
        if spec.head_dim != spec.v_head_dim {
            return Err(Uncovered {
                phase: "any",
                need: format!(
                    "value head dim {} equal to the key head dim {}",
                    spec.v_head_dim, spec.head_dim
                ),
            });
        }
        let full = kernels.full_key_block(spec.head_dim, spec.dtype);
        let full_ok = |phase: &'static str| -> std::result::Result<AttentionKernel, Uncovered> {
            let Some(bk) = full else {
                return Err(Uncovered {
                    phase,
                    need: format!(
                        "steel_attention at head dim {} and {:?}",
                        spec.head_dim, spec.dtype
                    ),
                });
            };
            if spec.softcap != 1.0 {
                return Err(Uncovered {
                    phase,
                    need: format!("score soft cap {} on steel_attention", spec.softcap),
                });
            }
            if let KvLayout::Paged { page_size } = spec.layout {
                if spec.mask == MaskKind::Additive {
                    return Err(Uncovered {
                        phase,
                        need: "an additive mask over a page pool".into(),
                    });
                }
                if page_size == 0 || !page_size.is_multiple_of(bk) {
                    return Err(Uncovered {
                        phase,
                        need: format!(
                            "page size {page_size} a multiple of steel_attention's key block {bk}"
                        ),
                    });
                }
            }
            Ok(AttentionKernel::Full)
        };

        let vector = spec.mask != MaskKind::Additive
            && spec.layout == KvLayout::Strided
            && kernels.vector(spec.head_dim, spec.dtype);
        let (single, two_pass_from) = if b.min_queries > 1 {
            (None, None)
        } else if vector {
            (
                Some(AttentionKernel::Vector),
                Some(kernels.two_pass_from(spec.head_dim, spec.dtype).max(1)),
            )
        } else if matches!(spec.layout, KvLayout::Paged { .. }) {
            return Err(Uncovered {
                phase: "single-query",
                need: "one query over a page pool (flash_decoding_paged serves it)".into(),
            });
        } else {
            (Some(full_ok("single-query")?), None)
        };
        let multi = if b.queries > 1 {
            Some(full_ok("multi-query")?)
        } else {
            None
        };
        Ok(PreparedAttention {
            spec,
            single,
            two_pass_from,
            two_pass_blocks: kernels.two_pass_blocks(),
            multi,
        })
    }

    pub fn spec(&self) -> &AttentionSpec {
        &self.spec
    }

    /// The kernel a request of `queries` over `keys` takes, after checking
    /// it against the bounds. Call before allocating anything for it.
    pub fn kernel_for(&self, batch: usize, queries: usize, keys: usize) -> Result<AttentionKernel> {
        let b = self.spec.bounds;
        if batch == 0 || queries == 0 || keys == 0 {
            lloom_tensor::bail!("attention request with an empty axis: {batch}x{queries}x{keys}");
        }
        if batch > b.batch || queries < b.min_queries || queries > b.queries || keys > b.keys {
            lloom_tensor::bail!(
                "attention request {batch}x{queries} over {keys} keys is outside the prepared \
                 bounds {}x{}..={} over {} keys",
                b.batch,
                b.min_queries,
                b.queries,
                b.keys
            );
        }
        if self.spec.mask != MaskKind::None && queries > keys {
            lloom_tensor::bail!("masked attention needs a key per query: {queries} over {keys}");
        }
        if queries == 1 {
            return match (self.single, self.two_pass_from) {
                (Some(AttentionKernel::Vector), Some(from)) if keys >= from => {
                    Ok(AttentionKernel::VectorTwoPass)
                }
                (Some(k), _) => Ok(k),
                (None, _) => lloom_tensor::bail!("one-query request on a window-only route"),
            };
        }
        self.multi.ok_or_else(|| {
            lloom_tensor::Error::Msg("multi-query request on a one-query route".into())
        })
    }

    /// The memory a request of `batch` sequences of `queries` over `keys`
    /// takes, from the kernel that request takes.
    pub fn workspace(
        &self,
        batch: usize,
        queries: usize,
        keys: usize,
    ) -> Result<AttentionWorkspace> {
        let kernel = self.kernel_for(batch, queries, keys)?;
        let s = &self.spec;
        let size = s.dtype.size_in_bytes();
        let output = checked(&[batch, s.q_heads, queries, s.v_head_dim, size])?;
        let scratch = match kernel {
            AttentionKernel::VectorTwoPass => {
                checked(&[4, batch, s.q_heads, self.two_pass_blocks, s.v_head_dim + 2])?
            }
            _ => 0,
        };
        let conversion = if s.kv_dtype == s.dtype {
            0
        } else {
            let kv_rows = match s.layout {
                KvLayout::Strided => batch,
                KvLayout::Paged { .. } => 1,
            };
            checked(&[2, kv_rows, s.kv_heads, keys, s.head_dim, size])?
        };
        let mask = match s.mask {
            MaskKind::Additive => checked(&[batch, s.q_heads, queries, keys, size])?,
            _ => 0,
        };
        let metadata = match s.layout {
            KvLayout::Strided => 0,
            KvLayout::Paged { page_size } => {
                let pages = keys.div_ceil(page_size);
                checked(&[batch, pages, 4])?
                    .checked_add(checked(&[batch, 4])?)
                    .ok_or_else(|| {
                        lloom_tensor::Error::Msg("attention workspace overflows".into())
                    })?
            }
        };
        Ok(AttentionWorkspace {
            output,
            scratch,
            conversion,
            mask,
            metadata,
        })
    }

    /// The largest workspace any request within the bounds takes, field by
    /// field: what admission charges for one call of this layer. An axis left
    /// at `usize::MAX` is unbounded by the route, and admission has to bound
    /// the request instead.
    pub fn worst_case_workspace(&self) -> Result<AttentionWorkspace> {
        let b = self.spec.bounds;
        for (axis, n) in [("batch", b.batch), ("queries", b.queries), ("keys", b.keys)] {
            if n == usize::MAX {
                lloom_tensor::bail!("attention route has no {axis} bound to size a worst case at");
            }
        }
        let mut worst = AttentionWorkspace::default();
        if self.single.is_some() {
            worst = worst.max(self.workspace(b.batch, 1, b.keys)?);
            if let Some(from) = self.two_pass_from {
                if from > 1 && from - 1 <= b.keys {
                    worst = worst.max(self.workspace(b.batch, 1, from - 1)?);
                }
            }
        }
        if b.queries > 1 {
            let queries = if self.spec.mask == MaskKind::None {
                b.queries
            } else {
                b.queries.min(b.keys)
            };
            worst = worst.max(self.workspace(b.batch, queries, b.keys)?);
        }
        Ok(worst)
    }

    /// The route as one line, for the load log.
    pub fn describe(&self) -> String {
        let s = &self.spec;
        let single = match (self.single, self.two_pass_from) {
            (Some(AttentionKernel::Vector), Some(from)) => {
                format!("sdpa_vector, sdpa_vector_2pass from {from} keys")
            }
            (Some(k), _) => k.name().to_string(),
            (None, _) => "none (windows only)".to_string(),
        };
        let multi = self.multi.map_or("none (one query)", AttentionKernel::name);
        format!(
            "attention route: head_dim {} {:?} kv {:?} heads {}/{} mask {:?} layout {:?} \
             bounds {}x{}..={} over {} keys: one query -> {single}; more -> {multi}",
            s.head_dim,
            s.dtype,
            s.kv_dtype,
            s.q_heads,
            s.kv_heads,
            s.mask,
            s.layout,
            s.bounds.batch,
            s.bounds.min_queries,
            s.bounds.queries,
            s.bounds.keys
        )
    }
}

/// The Metal kernel set, from the instantiated head-dim tables and the
/// compiled pipelines' admitted threadgroup widths.
#[cfg(feature = "metal")]
pub struct MetalKernelSet<'a> {
    pub device: &'a lloom_tensor::MetalDevice,
}

#[cfg(feature = "metal")]
impl KernelSet for MetalKernelSet<'_> {
    fn vector(&self, head_dim: usize, dtype: DType) -> bool {
        use lloom_tensor::kernels::{sdpa_vector_admits_its_widths, SDPA_VECTOR_HEAD_DIMS};
        let Some(itype) = sdpa_dtype(dtype) else {
            return false;
        };
        let is_f32 = dtype == DType::F32;
        lloom_tensor::shape_req!(
            sdpa_vector_head_dim,
            SDPA_VECTOR_HEAD_DIMS.contains(&head_dim) && !(head_dim == 512 && is_f32);
            head_dim,
            is_f32
        ) && sdpa_vector_admits_its_widths(
            self.device.device(),
            self.device.kernels(),
            head_dim,
            itype,
        )
        .unwrap_or(false)
    }

    fn full_key_block(&self, head_dim: usize, dtype: DType) -> Option<usize> {
        use lloom_tensor::kernels::{sdpa_full_tile, SDPA_FULL_HEAD_DIMS};
        let itype = sdpa_dtype(dtype)?;
        let is_f32 = dtype == DType::F32;
        if !lloom_tensor::shape_req!(
            sdpa_head_dim,
            SDPA_FULL_HEAD_DIMS.contains(&head_dim) && !(head_dim == 512 && is_f32);
            head_dim,
            is_f32
        ) {
            return None;
        }
        sdpa_full_tile(head_dim, itype).ok().map(|(_, bk, _, _)| bk)
    }

    fn two_pass_from(&self, head_dim: usize, dtype: DType) -> usize {
        crate::nn::ops::sdpa_vector_two_pass_threshold(head_dim, dtype)
    }

    fn two_pass_blocks(&self) -> usize {
        lloom_tensor::kernels::SDPA_2PASS_BLOCKS
    }
}

#[cfg(feature = "metal")]
fn sdpa_dtype(dtype: DType) -> Option<lloom_tensor::kernels::SdpaDType> {
    use lloom_tensor::kernels::SdpaDType;
    match dtype {
        DType::F16 => Some(SdpaDType::F16),
        DType::BF16 => Some(SdpaDType::BF16),
        DType::F32 => Some(SdpaDType::F32),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
