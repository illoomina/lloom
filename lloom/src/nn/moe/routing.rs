use crate::nn::ops;
use lloom_tensor::{bail, DType, Result, Tensor, D};

/// Scoring and mixture policy, independent of expert residency and execution.
#[derive(Debug, Clone, Default)]
pub enum Routing {
    /// Softmax over all experts, then normalize the selected weights.
    #[default]
    Softmax,
    /// Sigmoid in the router's dtype. F32 bias changes selection only; selected
    /// unbiased weights are normalized/scaled in the router's dtype.
    Sigmoid {
        bias: Option<Tensor>,
        normalize: bool,
        scale: f64,
    },
}

impl Routing {
    pub fn is_softmax(&self) -> bool {
        matches!(self, Self::Softmax)
    }

    /// Composed routing for all backends. Ties follow the backend's argsort;
    /// like the reference top-k, no cross-backend tie ordering is promised.
    /// The returned weights are F32 for the shared routed combine.
    pub fn route(&self, logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        let (_, experts) = logits.dims2()?;
        if k == 0 || k > experts || experts == 0 {
            bail!("MoE routing: top-k {k} outside 1..={experts}")
        }
        if !matches!(logits.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            bail!(
                "MoE routing: expected float logits, got {:?}",
                logits.dtype()
            )
        }
        let (probs, scores) = match self {
            Self::Softmax => {
                let p = ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
                (p.clone(), p)
            }
            Self::Sigmoid { bias, scale, .. } => {
                if !scale.is_finite() || *scale <= 0.0 {
                    bail!("MoE routing: scale must be finite and positive")
                }
                let p = ops::sigmoid(&logits.to_dtype(DType::F32)?)?.to_dtype(logits.dtype())?;
                let scores = match bias {
                    Some(b) => {
                        if b.dtype() != DType::F32 || b.dims() != [experts] {
                            bail!("MoE routing: bias must be F32 [{experts}]")
                        }
                        p.to_dtype(DType::F32)?.broadcast_add(b)?
                    }
                    None => p.to_dtype(DType::F32)?,
                };
                (p, scores)
            }
        };
        let ids = scores
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?;
        let weights = probs.gather(&ids, D::Minus1)?;
        let weights = match self {
            Self::Softmax => weights.broadcast_div(&weights.sum_keepdim(D::Minus1)?)?,
            Self::Sigmoid {
                normalize, scale, ..
            } => {
                let weights = if *normalize {
                    let sum = weights
                        .to_dtype(DType::F32)?
                        .sum_keepdim(D::Minus1)?
                        .to_dtype(logits.dtype())?;
                    let denominator =
                        (sum.to_dtype(DType::F32)? + 1e-6)?.to_dtype(logits.dtype())?;
                    weights
                        .to_dtype(DType::F32)?
                        .broadcast_div(&denominator.to_dtype(DType::F32)?)?
                        .to_dtype(logits.dtype())?
                } else {
                    weights
                };
                (weights.to_dtype(DType::F32)? * *scale)?.to_dtype(logits.dtype())?
            }
        };
        Ok((ids, weights.to_dtype(DType::F32)?))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Routing {
    /// The composed route, with separate ids and mixture-weight values. The
    /// softmax caller retains its existing fused node when that route is used.
    pub(crate) fn visit_composed(
        &self,
        w: &mut crate::optimizer::Walker,
        logits: crate::optimizer::NodeId,
        k: usize,
    ) -> (crate::optimizer::NodeId, crate::optimizer::NodeId) {
        use crate::optimizer::{NodeId, Shape};
        let shape = w.node(logits).shape.clone();
        let dtype = w.node(logits).dtype;
        let cast = |w: &mut crate::optimizer::Walker, x: NodeId, dtype: DType| {
            if w.node(x).dtype == dtype {
                x
            } else {
                let shape = w.node(x).shape.clone();
                w.opaque("cast", &[x], shape, dtype)
            }
        };
        let widened = cast(w, logits, DType::F32);
        let (probs, scores) = match self {
            Self::Softmax => {
                let p = w.opaque("softmax", &[widened], shape.clone(), DType::F32);
                (p, p)
            }
            Self::Sigmoid { bias, .. } => {
                let p = w.opaque("sigmoid", &[widened], shape.clone(), DType::F32);
                let p = cast(w, p, dtype);
                let scores = cast(w, p, DType::F32);
                let scores = match bias {
                    Some(b) => {
                        let bias = w.weight("expert_bias", Shape::of(b.dims()), DType::F32);
                        w.opaque("badd", &[scores, bias], shape.clone(), DType::F32)
                    }
                    None => scores,
                };
                (p, scores)
            }
        };
        let order = w.opaque("arg_sort", &[scores], shape, DType::U32);
        let ids = w.view("narrow", order, Shape::of(&[1, k]));
        // This walk is one decode row: narrowing the last dimension keeps
        // it contiguous, so runtime contiguous() encodes no copy.
        let probs_dtype = w.node(probs).dtype;
        let weights = w.opaque("gather_u32", &[probs, ids], Shape::of(&[1, k]), probs_dtype);
        let (normalize, epsilon, scale) = match self {
            Self::Softmax => (true, false, false),
            Self::Sigmoid { normalize, .. } => (*normalize, true, true),
        };
        let mut weights = weights;
        if normalize {
            let wide = cast(w, weights, DType::F32);
            let sum = w.opaque("fast_sum", &[wide], Shape::of(&[1, 1]), DType::F32);
            let sum = cast(w, sum, probs_dtype);
            let wide_sum = cast(w, sum, DType::F32);
            let denominator = if epsilon {
                w.opaque("affine", &[wide_sum], Shape::of(&[1, 1]), DType::F32)
            } else {
                wide_sum
            };
            let denominator = cast(w, denominator, probs_dtype);
            // Runtime widens selected weights separately for the reduction
            // and division. Preserve both values/lifetimes, not a fictitious
            // common-subexpression elimination of the F16 cast. The receiver
            // is evaluated before broadcast_div's denominator argument.
            let numerator = cast(w, weights, DType::F32);
            let denominator = cast(w, denominator, DType::F32);
            weights = w.opaque(
                "bdiv",
                &[numerator, denominator],
                Shape::of(&[1, k]),
                DType::F32,
            );
            weights = cast(w, weights, probs_dtype);
        }
        if scale {
            weights = cast(w, weights, DType::F32);
            weights = w.opaque("affine", &[weights], Shape::of(&[1, k]), DType::F32);
            weights = cast(w, weights, probs_dtype);
        }
        (ids, cast(w, weights, DType::F32))
    }
}
