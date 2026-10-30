//! The engine's weight representation, per tensor, whichever container the
//! checkpoint came in (`docs/porting-models.md` §21.14).
//!
//! A container adapter — a `VarBuilder` backend — answers each tensor a model
//! asks for with a [`WeightRepr`]: a dense tensor at the model's dtype, or the
//! packed blocks the quantized cores execute directly. One model
//! implementation consumes both; nothing above the adapter knows the format.
use crate::nn::var_builder::{LoadRecord, Recorder, SimpleBackend, VarBuilder};
use crate::nn::Init;
use lloom_tensor::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Module, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// One tensor as the engine holds it.
#[derive(Debug, Clone)]
pub enum WeightRepr {
    /// Dense, at the dtype the model computes in.
    Dense(Tensor),
    /// Packed quantization blocks (any packed `GgmlDType`), executed by the
    /// quantized cores without a dense copy.
    Blocks(Arc<QTensor>),
}

impl WeightRepr {
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Dense(t) => t.dims(),
            Self::Blocks(q) => q.shape().dims(),
        }
    }

    /// The storage's name: the dense dtype, or the block kind.
    pub fn kind(&self) -> String {
        match self {
            Self::Dense(t) => t.dtype().as_str().to_string(),
            Self::Blocks(q) => format!("{:?}", q.dtype()).to_lowercase(),
        }
    }
}

/// `y = x · Wᵀ (+ b)` over a [`WeightRepr`]. Dense runs `nn::Linear`'s own
/// forward, so a dense model computes exactly what it did before the seam;
/// blocks run the quantized matmul. The GEMV epilogue kernels read a dense
/// weight, so they reach it only through [`Self::dense`].
#[derive(Debug, Clone)]
pub struct ReprLinear {
    inner: Projection,
    span: tracing::Span,
}

#[derive(Debug, Clone)]
enum Projection {
    Dense(crate::nn::Linear),
    Blocks(QMatMul),
}

impl ReprLinear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        Self::of(Projection::Dense(crate::nn::Linear::new(weights, bias)))
    }

    /// A bias-free projection over `w`. Packed blocks go through
    /// `QMatMul::from_arc`, so the quantized tuning (`qmm.dequantize_all`)
    /// applies as it does to every other GGUF consumer.
    pub fn from_repr(w: WeightRepr) -> Result<Self> {
        Ok(Self::of(match w {
            WeightRepr::Dense(t) => Projection::Dense(crate::nn::Linear::new(t, None)),
            WeightRepr::Blocks(q) => Projection::Blocks(QMatMul::from_arc(q)?),
        }))
    }

    fn of(inner: Projection) -> Self {
        Self {
            inner,
            span: tracing::span!(tracing::Level::TRACE, "linear"),
        }
    }

    /// The `[out, in]` dense weight, for a caller that fuses this projection
    /// into one kernel; `None` for packed blocks, which no GEMV epilogue reads.
    pub fn dense(&self) -> Option<&Tensor> {
        match &self.inner {
            Projection::Dense(l) => Some(l.weight()),
            Projection::Blocks(_) => None,
        }
    }

    /// [`Self::dense`] where a fused arm already established it: a packed
    /// weight reaching a dense-only kernel is an error, not a panic.
    pub fn dense_weight(&self) -> Result<&Tensor> {
        self.dense().ok_or_else(|| {
            lloom_tensor::Error::msg(format!(
                "a dense-only fused projection was handed packed {} blocks",
                self.kind()
            ))
        })
    }

    /// The quantized matmul, for the walk's core attribution.
    pub fn blocks(&self) -> Option<&QMatMul> {
        match &self.inner {
            Projection::Blocks(q) => Some(q),
            Projection::Dense(_) => None,
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        match &self.inner {
            Projection::Dense(l) => l.bias(),
            Projection::Blocks(_) => None,
        }
    }

    /// `(out, in)`.
    pub fn dims(&self) -> (usize, usize) {
        let d = match &self.inner {
            Projection::Dense(l) => l.weight().dims(),
            Projection::Blocks(QMatMul::QTensor(q)) => q.shape().dims(),
            Projection::Blocks(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.dims(),
        };
        (d[0], d[d.len() - 1])
    }

    pub fn device(&self) -> Device {
        match &self.inner {
            Projection::Dense(l) => l.weight().device().clone(),
            Projection::Blocks(QMatMul::QTensor(q)) => q.device(),
            Projection::Blocks(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.device().clone(),
        }
    }

    /// The storage's name (`f16`, `q8_0`, …).
    pub fn kind(&self) -> String {
        match &self.inner {
            Projection::Dense(l) => l.weight().dtype().as_str().to_string(),
            Projection::Blocks(QMatMul::QTensor(q)) => format!("{:?}", q.dtype()).to_lowercase(),
            Projection::Blocks(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => {
                format!("{} (dequantized)", t.dtype().as_str())
            }
        }
    }
}

impl Module for ReprLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match &self.inner {
            Projection::Dense(l) => l.forward(x),
            // The quantized cores read a packed row; a strided view is copied
            // first (a no-op on a contiguous input).
            Projection::Blocks(q) => q.forward(&x.contiguous()?),
        }
    }
}

impl crate::nn::linear::Projection for ReprLinear {
    fn dimensions(&self) -> Result<(usize, usize)> {
        Ok(self.dims())
    }
    fn projection_bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

/// A bias-free projection loaded as the container stores it, with the hints
/// `nn::linear_no_bias` gives an initializing backend.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<ReprLinear> {
    let w = vb.get_repr_with_hints(
        (out_dim, in_dim),
        "weight",
        crate::nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    ReprLinear::from_repr(w)
}

/// A token table over a [`WeightRepr`]: dense rows by `index_select`, or
/// packed rows dequantized per lookup (the values a dense copy would hold).
#[derive(Debug, Clone)]
pub struct ReprEmbedding {
    inner: Table,
    hidden: usize,
    span: tracing::Span,
}

#[derive(Debug, Clone)]
enum Table {
    Dense(crate::nn::Embedding),
    Blocks(Arc<QTensor>),
}

impl ReprEmbedding {
    pub fn new(vocab: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get_repr_with_hints(
            (vocab, hidden),
            "weight",
            Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
        )?;
        Ok(Self {
            inner: match w {
                WeightRepr::Dense(t) => Table::Dense(crate::nn::Embedding::new(t, hidden)),
                WeightRepr::Blocks(q) => Table::Blocks(q),
            },
            hidden,
            span: tracing::span!(tracing::Level::TRACE, "embedding"),
        })
    }

    /// The dense table; `None` when it is packed.
    pub fn embeddings(&self) -> Option<&Tensor> {
        match &self.inner {
            Table::Dense(e) => Some(e.embeddings()),
            Table::Blocks(_) => None,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The tied head: a projection over this same storage, no copy.
    pub fn tied_head(&self) -> Result<ReprLinear> {
        match &self.inner {
            Table::Dense(e) => Ok(ReprLinear::from_weights(e.embeddings().clone(), None)),
            Table::Blocks(q) => ReprLinear::from_repr(WeightRepr::Blocks(q.clone())),
        }
    }

    /// Whether `head` projects through this table's storage.
    pub fn ties(&self, head: &ReprLinear) -> bool {
        match (&self.inner, &head.inner) {
            (Table::Dense(e), Projection::Dense(l)) => l.weight().id() == e.embeddings().id(),
            (Table::Blocks(t), Projection::Blocks(QMatMul::QTensor(q))) => Arc::ptr_eq(t, q),
            _ => false,
        }
    }

    pub fn kind(&self) -> String {
        match &self.inner {
            Table::Dense(e) => e.embeddings().dtype().as_str().to_string(),
            Table::Blocks(q) => format!("{:?}", q.dtype()).to_lowercase(),
        }
    }

    /// The dtype a lookup returns: the dense table's, f32 for packed rows.
    pub fn compute_dtype(&self) -> DType {
        match &self.inner {
            Table::Dense(e) => e.embeddings().dtype(),
            Table::Blocks(_) => DType::F32,
        }
    }

    /// The table's dtype as the walk names it: the dense dtype, `U8` packed.
    pub fn walk_dtype(&self) -> DType {
        match &self.inner {
            Table::Dense(e) => e.embeddings().dtype(),
            Table::Blocks(_) => DType::U8,
        }
    }
}

impl Module for ReprEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match &self.inner {
            Table::Dense(e) => e.forward(ids),
            Table::Blocks(q) => q.embedding(ids),
        }
    }
}

/// A GGUF file as a `VarBuilder` backend: engine tensor names mapped onto
/// the file's (`names`, built from the family's GGUF schema), each tensor
/// read on demand. Packed kinds come back as [`WeightRepr::Blocks`]; dense
/// kinds (F32/F16/BF16), and any tensor asked for through `get`, come back
/// dequantized at the requested dtype. A requested shape equal to the
/// stored one up to singleton axes (`[w, k]` stored, `[w, 1, k]` asked) is a
/// reshape; any other difference is refused with both shapes.
pub struct GgufBackend<R> {
    content: gguf_file::Content,
    reader: parking_lot::Mutex<R>,
    names: HashMap<String, String>,
    recorder: Recorder,
}

impl<R: std::io::Read + std::io::Seek + Send> GgufBackend<R> {
    pub fn new(content: gguf_file::Content, reader: R, names: HashMap<String, String>) -> Self {
        let entries = content
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), stored_bytes(info)))
            .collect();
        Self {
            content,
            reader: parking_lot::Mutex::new(reader),
            names,
            recorder: Recorder::new(entries),
        }
    }

    /// Which of the file's tensors the model has read, with their bytes in
    /// the file (`LoadRecord`, the safetensors backend's record).
    pub fn record(&self) -> LoadRecord {
        self.recorder.record()
    }

    fn stored(&self, name: &str) -> Result<&str> {
        self.names.get(name).map(String::as_str).ok_or_else(|| {
            lloom_tensor::Error::msg(format!("GGUF adapter: no tensor maps to `{name}`"))
        })
    }

    fn read(&self, name: &str, s: &Shape, dev: &Device) -> Result<QTensor> {
        let stored = self.stored(name)?;
        self.recorder.mark(stored);
        let q = self.content.tensor(&mut *self.reader.lock(), stored, dev)?;
        let squeeze = |d: &[usize]| d.iter().copied().filter(|&x| x != 1).collect::<Vec<_>>();
        if q.shape() != s && squeeze(q.shape().dims()) != squeeze(s.dims()) {
            lloom_tensor::bail!(
                "GGUF adapter: `{name}` (stored as `{stored}`) is {:?}, the model asks for {:?}",
                q.shape(),
                s
            )
        }
        Ok(q)
    }

    fn dense(&self, q: &QTensor, s: &Shape, dtype: DType, dev: &Device) -> Result<Tensor> {
        q.dequantize(dev)?.reshape(s)?.to_dtype(dtype)
    }
}

impl<R: std::io::Read + std::io::Seek + Send> SimpleBackend for GgufBackend<R> {
    fn get(&self, s: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let q = self.read(name, &s, dev)?;
        self.dense(&q, &s, dtype, dev)
    }

    fn get_repr(
        &self,
        s: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<WeightRepr> {
        let q = self.read(name, &s, dev)?;
        match q.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                Ok(WeightRepr::Dense(self.dense(&q, &s, dtype, dev)?))
            }
            _ if q.shape() == &s => Ok(WeightRepr::Blocks(Arc::new(q))),
            _ => lloom_tensor::bail!(
                "GGUF adapter: packed `{name}` is {:?}, the model asks for {:?}: blocks are not reshaped",
                q.shape(),
                s
            ),
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let stored = self.stored(name)?;
        self.recorder.mark(stored);
        let q = self.content.tensor(&mut *self.reader.lock(), stored, dev)?;
        q.dequantize(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.names
            .get(name)
            .is_some_and(|s| self.content.tensor_infos.contains_key(s))
    }
}

/// A tensor's bytes in the file: its blocks at the kind's size.
pub fn stored_bytes(info: &gguf_file::TensorInfo) -> usize {
    let kind = info.ggml_dtype;
    info.shape.elem_count() / kind.block_size() * kind.type_size()
}

/// What the tensors of `content` hold once loaded for a model computing in
/// `dense`: packed kinds at their stored size, dense kinds (F32/F16/BF16)
/// at `dense`'s size — the admission's weight term.
pub fn loaded_bytes(content: &gguf_file::Content, dense: DType) -> usize {
    content
        .tensor_infos
        .values()
        .map(|info| match info.ggml_dtype {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                info.shape.elem_count() * dense.size_in_bytes()
            }
            _ => stored_bytes(info),
        })
        .sum()
}
