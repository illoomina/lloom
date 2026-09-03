use crate::nn::VarBuilder;
use lloom_tensor::{Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: crate::nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = crate::nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = crate::nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: crate::nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = crate::nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }

    /// The `[out, in]` weight, for a caller that fuses this projection into a
    /// single kernel rather than calling [`Module::forward`] (issue #509).
    ///
    /// Forwards to [`crate::nn::Linear::weight`], which is already public — the
    /// wrapper was simply opaque. A fused GEMV epilogue needs the operands, not
    /// the op, and this is the accessor that hands them over.
    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }

    /// The optional bias, so a fusing caller can **verify there is none**
    /// rather than assume it.
    ///
    /// `gemv_silu` and `gemv_residual` compute `x @ w.t()` with no bias term.
    /// LFM2 builds every projection with `linear_no_bias`, so this is `None` at
    /// every site that fuses — but a caller that assumed rather than checked
    /// would silently drop a bias if that ever changed.
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }
}

pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = crate::nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = crate::nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = crate::nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: crate::nn::Conv2d,
    span: tracing::Span,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: crate::nn::Conv2dConfig,
    vs: crate::nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = crate::nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: lloom_tensor::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = lloom_tensor::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    pub fn from_weights(ws: std::sync::Arc<lloom_tensor::quantized::QTensor>) -> Result<Self> {
        let inner = lloom_tensor::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    inner: crate::nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = crate::nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self { inner, span }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

pub fn layer_norm<C: Into<crate::nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = crate::nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: crate::nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = crate::nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    pub fn forward_diff(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward_diff(x)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}
