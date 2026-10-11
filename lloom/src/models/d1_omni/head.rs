use super::weights::{LinearWeights, NormWeights, Weight, WeightPlan};
use crate::models::siglip::{
    Attention, Encoder, EncoderLayer, Mlp, PackedQkv, QkvProjection, SeparateQkv,
};
use crate::nn::linear::Projection;
use crate::nn::{
    embedding, layer_norm, linear, Activation, Embedding, LayerNorm, Linear, VarBuilder,
};
use crate::quantized_nn::Linear as QuantizedLinear;
use lloom_tensor::{DType, Module, Result, Tensor};

/// Prepared, right-padded text rows and option markers for one head invocation.
/// Mask backing is B*L, not B*H*L*L. Marker indices are B*options, not B*options*D.
#[derive(Debug)]
pub struct DecisionHeadBatch {
    batch: usize,
    length: usize,
    width: usize,
    options: usize,
    qtypes: Tensor,
    key_mask: Tensor,
    marker_rows: Tensor,
    marker_valid: Tensor,
    missing_logits: Tensor,
}

/// Question-type embedding, pre-norm transformer stack and shared marker scorer.
/// Dense checkpoint facade; the private implementation also serves packed GGUF.
#[derive(Debug, Clone)]
pub struct DecisionHead {
    inner: HeadInner<Linear, SeparateQkv<Linear>>,
}

#[derive(Debug, Clone)]
struct HeadInner<P, Q> {
    width: usize,
    type_embedding: Embedding,
    encoder: Encoder<P, Q>,
    scorer_norm: LayerNorm,
    scorer_hidden: P,
    scorer_output: P,
}

impl DecisionHead {
    pub(crate) fn validate_config(
        width: usize,
        layers: usize,
        dtype: DType,
    ) -> Result<(usize, usize, usize)> {
        HeadInner::<Linear, SeparateQkv<Linear>>::validate_config(width, layers, dtype)
    }
    pub fn load(width: usize, layers: usize, vb: VarBuilder) -> Result<Self> {
        let (heads, packed, intermediate) = Self::validate_config(width, layers, vb.dtype())?;
        let type_embedding = embedding(3, width, vb.pp("type_emb"))?;
        let mut stack = Vec::with_capacity(layers);
        for i in 0..layers {
            let layer = vb.pp("head.layers").pp(i);
            let attn = layer.pp("self_attn");
            let w = attn.get((packed, width), "in_proj_weight")?;
            let b = attn.get(packed, "in_proj_bias")?;
            let projection = |part| -> Result<Linear> {
                Ok(Linear::new(
                    w.narrow(0, part * width, width)?,
                    Some(b.narrow(0, part * width, width)?),
                ))
            };
            let attention = Attention::from_projections(
                projection(0)?,
                projection(1)?,
                projection(2)?,
                linear(width, width, attn.pp("out_proj"))?,
                heads,
            )?;
            stack.push(EncoderLayer::from_parts(
                attention,
                layer_norm(width, 1e-5, layer.pp("norm1"))?,
                Mlp::from_parts(
                    linear(width, intermediate, layer.pp("linear1"))?,
                    linear(intermediate, width, layer.pp("linear2"))?,
                    Activation::Relu,
                ),
                layer_norm(width, 1e-5, layer.pp("norm2"))?,
            ));
        }
        Ok(Self {
            inner: HeadInner {
                width,
                type_embedding,
                encoder: Encoder::from_layers(stack),
                scorer_norm: layer_norm(width, 1e-5, vb.pp("scorer.0"))?,
                scorer_hidden: linear(width, width, vb.pp("scorer.1"))?,
                scorer_output: linear(width, 1, vb.pp("scorer.3"))?,
            },
        })
    }

    /// Validate right-padded rows and option markers before device indexing.
    pub fn prepare_batch(
        &self,
        length: usize,
        lengths: &[usize],
        markers: &[&[usize]],
        qtypes: &[u32],
    ) -> Result<DecisionHeadBatch> {
        self.inner.prepare_batch(length, lengths, markers, qtypes)
    }

    /// Score [B,L,D] into F32 [B,options]. Callback applies the compact additive
    /// key mask without causality; there is no hidden attention fallback.
    pub fn forward_with_attention<F>(
        &self,
        hidden: &Tensor,
        batch: &DecisionHeadBatch,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.forward_with_attention(hidden, batch, attend)
    }
}

impl<P: Projection, Q: QkvProjection> HeadInner<P, Q> {
    pub(crate) fn validate_config(
        width: usize,
        layers: usize,
        dtype: DType,
    ) -> Result<(usize, usize, usize)> {
        let heads = width / 64;
        if heads == 0 || width % heads != 0 {
            lloom_tensor::bail!("decision head width must be divisible by its width/64 heads")
        }
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            lloom_tensor::bail!("decision head requires floating-point weights")
        }
        let packed = width
            .checked_mul(3)
            .ok_or_else(|| lloom_tensor::Error::Msg("decision head width overflow".into()))?;
        let intermediate = width
            .checked_mul(4)
            .ok_or_else(|| lloom_tensor::Error::Msg("decision head width overflow".into()))?;
        // Representability of actual weight arrays and layer metadata, not
        // a weight-load/pass reservation or physical memory-fit claim.
        let checked = |dims: &[usize]| -> Result<()> {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| {
                    lloom_tensor::Error::Msg("decision head shape/byte overflow".into())
                })?;
            Ok(())
        };
        let e = dtype.size_in_bytes();
        checked(&[3, width, e])?;
        checked(&[width, width, e])?; // scorer exists even with zero layers
        checked(&[layers, std::mem::size_of::<EncoderLayer<P, Q>>()])?;
        if layers != 0 {
            checked(&[packed, width, e])?;
            checked(&[intermediate, width, e])?;
        }
        Ok((heads, packed, intermediate))
    }

    /// Validate host metadata before device indexing. Each row has at least one
    /// real token and marker; marker positions refer only to real text tokens.
    /// Question types use the checkpoint's 0/1/2 embedding order, not labels.
    pub fn prepare_batch(
        &self,
        length: usize,
        lengths: &[usize],
        markers: &[&[usize]],
        qtypes: &[u32],
    ) -> Result<DecisionHeadBatch> {
        let batch = lengths.len();
        if batch == 0 || length == 0 || markers.len() != batch || qtypes.len() != batch {
            lloom_tensor::bail!("decision head requires matching nonempty batch metadata")
        }
        let rows = batch
            .checked_mul(length)
            .ok_or_else(|| lloom_tensor::Error::Msg("decision head batch overflow".into()))?;
        if rows > u32::MAX as usize
            || lengths.iter().any(|&n| n == 0 || n > length)
            || qtypes.iter().any(|&q| q >= 3)
        {
            lloom_tensor::bail!("decision head invalid lengths, question types or row-index range")
        }
        let options = markers.iter().map(|row| row.len()).max().unwrap_or(0);
        if markers
            .iter()
            .zip(lengths)
            .any(|(row, &n)| row.is_empty() || row.iter().any(|&p| p >= n))
        {
            lloom_tensor::bail!("decision head markers must index real tokens in every row")
        }
        let count = batch.checked_mul(options).ok_or_else(|| {
            lloom_tensor::Error::Msg("decision head marker count overflow".into())
        })?;
        let mut mask = Vec::with_capacity(rows);
        let mut indices = Vec::with_capacity(count);
        let mut valid = Vec::with_capacity(count);
        for (row, (&n, positions)) in lengths.iter().zip(markers).enumerate() {
            mask.extend((0..length).map(|p| if p < n { 0f32 } else { f32::NEG_INFINITY }));
            for option in 0..options {
                let position = positions.get(option);
                indices.push((row * length + position.copied().unwrap_or(0)) as u32);
                valid.push(u8::from(position.is_some()));
            }
        }
        let weights = self.type_embedding.embeddings();
        let device = weights.device();
        Ok(DecisionHeadBatch {
            batch,
            length,
            width: self.width,
            options,
            qtypes: Tensor::from_slice(qtypes, batch, device)?,
            key_mask: Tensor::from_vec(mask, (batch, 1, 1, length), device)?
                .to_dtype(weights.dtype())?,
            marker_rows: Tensor::from_vec(indices, count, device)?,
            marker_valid: Tensor::from_vec(valid, (batch, options), device)?,
            missing_logits: Tensor::full(-1e4f32, (batch, options), device)?,
        })
    }

    /// Score text hidden states [B,L,D], returning F32 [B,options] logits.
    /// Missing padded options are -1e4, after scoring and F32 conversion.
    /// The callback receives the compact additive key mask [B,1,1,L]; it must
    /// apply that mask without introducing causality. No attention fallback is
    /// selected here. Calibration and semantic option ordering belong to readout.
    pub fn forward_with_attention<F>(
        &self,
        hidden: &Tensor,
        batch: &DecisionHeadBatch,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        let weights = self.type_embedding.embeddings();
        if hidden.dims() != [batch.batch, batch.length, self.width]
            || batch.width != self.width
            || hidden.dtype() != weights.dtype()
            || batch.key_mask.dtype() != weights.dtype()
            || !hidden.device().same_device(weights.device())
            || !batch.key_mask.device().same_device(weights.device())
        {
            lloom_tensor::bail!(
                "decision head input and prepared batch must match weight shape, dtype and device"
            )
        }
        let typed =
            hidden.broadcast_add(&self.type_embedding.forward(&batch.qtypes)?.unsqueeze(1)?)?;
        let encoded = self
            .encoder
            .forward_with_attention(&typed, Some(&batch.key_mask), attend)?;
        let selected = encoded
            .reshape((batch.batch * batch.length, self.width))?
            .index_select(&batch.marker_rows, 0)?
            .reshape((batch.batch, batch.options, self.width))?;
        let logits = selected
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_hidden)?
            .apply(&Activation::Gelu)?
            .apply(&self.scorer_output)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;
        batch
            .marker_valid
            .where_cond(&logits, &batch.missing_logits)
    }
}

/// Official GGUF head namespace, using packed QKV and quantized projections.
/// The caller supplies an already-loaded weight view and owns load/pass admission;
/// constructing this component does not load a second checkpoint or create quota.
#[derive(Debug, Clone)]
pub(crate) struct QuantizedDecisionHead {
    inner: HeadInner<QuantizedLinear, PackedQkv<QuantizedLinear>>,
}
#[derive(Debug)]
struct HeadBlockWeights {
    attention: [LinearWeights; 2],
    norms: [NormWeights; 2],
    mlp: [LinearWeights; 2],
}
impl WeightPlan for HeadBlockWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.attention.as_slice().visit(f)?;
        self.norms[0].visit(f)?;
        self.mlp.as_slice().visit(f)?;
        self.norms[1].visit(f)
    }
}
#[derive(Debug)]
pub(super) struct HeadWeights {
    type_embedding: Weight,
    blocks: Vec<HeadBlockWeights>,
    scorer_norm: NormWeights,
    scorer: [LinearWeights; 2],
}
impl HeadWeights {
    pub(super) fn new(width: usize, layers: usize, trunk_layers: usize) -> Result<Self> {
        let (_, packed, intermediate) = QuantizedDecisionHead::validate_config(width, layers)?;
        trunk_layers
            .checked_add(layers)
            .ok_or_else(|| lloom_tensor::Error::Msg("GGUF head layer index overflow".into()))?;
        let mut blocks = super::host::prepared_vec(layers)?;
        for i in 0..layers {
            let p = format!("blk.{}", trunk_layers + i);
            blocks.push(HeadBlockWeights {
                attention: [
                    LinearWeights::new(&format!("{p}.attn_qkv"), width, packed, true),
                    LinearWeights::new(&format!("{p}.attn_output"), width, width, true),
                ],
                norms: [
                    NormWeights::new(&format!("{p}.attn_norm"), width),
                    NormWeights::new(&format!("{p}.ffn_norm"), width),
                ],
                mlp: [
                    LinearWeights::new(&format!("{p}.ffn_up"), width, intermediate, true),
                    LinearWeights::new(&format!("{p}.ffn_down"), intermediate, width, true),
                ],
            });
        }
        Ok(Self {
            type_embedding: Weight::new("token_types.weight", (3, width)),
            blocks,
            scorer_norm: NormWeights::new("cls.norm", width),
            scorer: [
                LinearWeights::new("cls", width, width, true),
                LinearWeights::new("cls.output", width, 1, true),
            ],
        })
    }
}
impl WeightPlan for HeadWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.type_embedding.visit(f)?;
        self.blocks.as_slice().visit(f)?;
        self.scorer_norm.visit(f)?;
        self.scorer.as_slice().visit(f)
    }
}

impl QuantizedDecisionHead {
    pub(crate) fn validate_config(width: usize, layers: usize) -> Result<(usize, usize, usize)> {
        HeadInner::<QuantizedLinear, PackedQkv<QuantizedLinear>>::validate_config(
            width,
            layers,
            DType::F32,
        )
    }
    pub(crate) fn load(
        width: usize,
        layers: usize,
        trunk_layers: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let weights = HeadWeights::new(width, layers, trunk_layers)?;
        let heads = width / 64;
        let type_embedding = Embedding::new(weights.type_embedding.dequantize(&vb)?, width);
        let mut stack = Vec::with_capacity(layers);
        for block in &weights.blocks {
            let attention = Attention::from_qkv(
                PackedQkv::new(block.attention[0].load(&vb)?)?,
                block.attention[1].load(&vb)?,
                heads,
            )?;
            stack.push(EncoderLayer::from_parts(
                attention,
                block.norms[0].load(&vb, 1e-5)?,
                Mlp::from_parts(
                    block.mlp[0].load(&vb)?,
                    block.mlp[1].load(&vb)?,
                    Activation::Relu,
                ),
                block.norms[1].load(&vb, 1e-5)?,
            ));
        }
        Ok(Self {
            inner: HeadInner {
                width,
                type_embedding,
                encoder: Encoder::from_layers(stack),
                scorer_norm: weights.scorer_norm.load(&vb, 1e-5)?,
                scorer_hidden: weights.scorer[0].load(&vb)?,
                scorer_output: weights.scorer[1].load(&vb)?,
            },
        })
    }
    /// Validate right-padded rows and option markers before device indexing.
    pub fn prepare_batch(
        &self,
        length: usize,
        lengths: &[usize],
        markers: &[&[usize]],
        qtypes: &[u32],
    ) -> Result<DecisionHeadBatch> {
        self.inner.prepare_batch(length, lengths, markers, qtypes)
    }

    /// Score [B,L,D] into F32 [B,options]. Callback applies the compact additive
    /// key mask without causality; there is no hidden attention fallback.
    pub fn forward_with_attention<F>(
        &self,
        hidden: &Tensor,
        batch: &DecisionHeadBatch,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.forward_with_attention(hidden, batch, attend)
    }
}

#[cfg(test)]
mod tests;
