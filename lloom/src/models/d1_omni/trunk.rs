//! Bidirectional LFM2 encoder composition, following LiquidAI/d1-omni-600M
//! encoder.py at 02b55d7076f15129e59ab3f94783f32c4b088674.
//! Backend attention and request admission are supplied by the loaded owner.
use crate::nn::blocks::{Neox, Rope, SwiGlu};
use crate::nn::short_conv::{Centered, ShortConv};
use crate::nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Module, Result, Tensor};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub layer_types: Vec<String>,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub block_ffn_dim_multiplier: f64,
    pub block_multiple_of: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| lloom_tensor::Error::Msg("omni trunk shape/byte overflow".into()))
}

impl TextConfig {
    /// Reference MLP conversion, not LFM2's automatic-width policy. The official
    /// 6656 intermediate becomes int(2*6656/3)=4437, then rounds to 4608.
    pub fn validated_intermediate(&self) -> Result<usize> {
        let d = self.hidden_size;
        let h = self.num_attention_heads;
        let k = self.num_key_value_heads;
        if d == 0
            || h == 0
            || k == 0
            || d % h != 0
            || h % k != 0
            || (d / h) % 2 != 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.layer_types.len() != self.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|s| s != "conv" && s != "full_attention")
            || self.conv_l_cache != 3
            || self.intermediate_size == 0
            || self.block_multiple_of == 0
            || self.max_position_embeddings == 0
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
            || !self.block_ffn_dim_multiplier.is_finite()
            || self.block_ffn_dim_multiplier <= 0.
        {
            lloom_tensor::bail!(
                "invalid omni text geometry, layer types or normalization/rope configuration"
            )
        }
        let twice = product(&[2, self.intermediate_size])?;
        // Exact integer quotient agrees with the reference's int(2*I/3) for
        // supported checkpoint dimensions; reject inexact float-size metadata.
        if twice as u128 > (1u128 << 53) {
            lloom_tensor::bail!("omni intermediate exceeds exact configuration arithmetic")
        }
        let scaled = ((twice / 3) as f64 * self.block_ffn_dim_multiplier).trunc();
        if scaled < 1. || scaled >= usize::MAX as f64 || scaled > (1u64 << 53) as f64 {
            lloom_tensor::bail!("omni intermediate multiplier overflows or gives zero width")
        }
        let units = (scaled as usize)
            .checked_add(self.block_multiple_of - 1)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("omni intermediate rounding overflow".into())
            })?
            / self.block_multiple_of;
        let intermediate = product(&[units, self.block_multiple_of])?;
        product(&[d, intermediate])?;
        product(&[d, d, 3])?;
        product(&[self.vocab_size, d])?;
        Ok(intermediate)
    }
}

/// One contiguous query segment uses a single additive key row. Keys are NEVER
/// truncated: the reference's finite negative value is not negative infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuerySegment {
    pub batch_row: usize,
    pub start: usize,
    pub length: usize,
    mask_row: usize,
}

#[derive(Debug)]
pub struct TrunkBatch {
    batch: usize,
    length: usize,
    width: usize,
    pad: Tensor,
    keep_right: Tensor,
    key_rows: Tensor,
    segments: Vec<QuerySegment>,
}

impl TrunkBatch {
    pub fn segments(&self) -> &[QuerySegment] {
        &self.segments
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.batch, self.length)
    }

    /// [1,1,1,L] backing row for a segment returned by `segments`. A backend
    /// can broadcast this over a bounded query tile without first making L².
    pub fn key_mask(&self, segment_index: usize) -> Result<Tensor> {
        let segment = self
            .segments
            .get(segment_index)
            .ok_or_else(|| lloom_tensor::Error::Msg("omni query segment out of range".into()))?;
        self.key_rows.narrow(0, segment.mask_row, 1)
    }
}

#[derive(Debug, Clone)]
struct Gqa {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Gqa {
    fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let hd = d / cfg.num_attention_heads;
        let kv = product(&[cfg.num_key_value_heads, hd])?;
        Ok(Self {
            q: linear_no_bias(d, d, vb.pp("q_proj"))?,
            k: linear_no_bias(d, kv, vb.pp("k_proj"))?,
            v: linear_no_bias(d, kv, vb.pp("v_proj"))?,
            out: linear_no_bias(d, d, vb.pp("out_proj"))?,
            q_norm: rms_norm(hd, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: rms_norm(hd, cfg.norm_eps, vb.pp("k_layernorm"))?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
        })
    }

    fn forward<F>(
        &self,
        x: &Tensor,
        batch: &TrunkBatch,
        cos: &Tensor,
        sin: &Tensor,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
    {
        let (b, l, d) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((b, l, self.heads, self.head_dim))?
            .apply(&self.q_norm)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, l, self.kv_heads, self.head_dim))?
            .apply(&self.k_norm)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, l, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = Neox.execute(&q, cos, sin)?;
        let k = Neox.execute(&k, cos, sin)?;
        let groups = self.heads / self.kv_heads;
        let k = crate::utils::repeat_kv(k, groups)?;
        let v = crate::utils::repeat_kv(v, groups)?;
        let y = attend(&q, &k, &v, batch, (self.head_dim as f64).powf(-0.5))?;
        if y.dims() != q.dims() || y.dtype() != x.dtype() || !y.device().same_device(x.device()) {
            lloom_tensor::bail!(
                "omni attention callback must return [B,H,L,Dh] on input dtype/device"
            )
        }
        self.out.forward(&y.transpose(1, 2)?.reshape((b, l, d))?)
    }
}

#[derive(Debug, Clone)]
enum Mixer {
    Conv {
        input: Linear,
        conv: ShortConv<Centered>,
        output: Linear,
    },
    Attention(Gqa),
}

#[derive(Debug, Clone)]
struct Layer {
    mixer: Mixer,
    ffn: SwiGlu,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

#[derive(Debug, Clone)]
pub struct Trunk {
    cfg: TextConfig,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

impl Trunk {
    pub fn load(cfg: TextConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate = cfg.validated_intermediate()?;
        if !matches!(
            vb.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            lloom_tensor::bail!("omni trunk requires floating-point weights")
        }
        let d = cfg.hidden_size;
        product(&[cfg.vocab_size, d, vb.dtype().size_in_bytes()])?;
        product(&[d, intermediate, vb.dtype().size_in_bytes()])?;
        let embedding = embedding(cfg.vocab_size, d, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg.layer_types.iter().enumerate() {
            let layer = vb.pp("layers").pp(i);
            let mixer = if kind == "full_attention" {
                Mixer::Attention(Gqa::load(&cfg, layer.pp("self_attn"))?)
            } else {
                let conv = layer.pp("conv");
                Mixer::Conv {
                    input: linear_no_bias(d, product(&[3, d])?, conv.pp("in_proj"))?,
                    conv: ShortConv::<Centered>::new(
                        conv.get((d, 1, 3), "conv.weight")?.squeeze(1)?,
                    )?,
                    output: linear_no_bias(d, d, conv.pp("out_proj"))?,
                }
            };
            let ff = layer.pp("feed_forward");
            layers.push(Layer {
                mixer,
                ffn: SwiGlu::new(
                    QMatMul::Tensor(ff.get((intermediate, d), "w1.weight")?),
                    QMatMul::Tensor(ff.get((intermediate, d), "w3.weight")?),
                    QMatMul::Tensor(ff.get((d, intermediate), "w2.weight")?),
                    d,
                    intermediate,
                ),
                operator_norm: rms_norm(d, cfg.norm_eps, layer.pp("operator_norm"))?,
                ffn_norm: rms_norm(d, cfg.norm_eps, layer.pp("ffn_norm"))?,
            });
        }
        let norm = rms_norm(d, cfg.norm_eps, vb.pp("embedding_norm"))?;
        Ok(Self {
            cfg,
            embedding,
            layers,
            norm,
        })
    }

    /// Prepare default NeoX tables on the trunk's dtype/device through the
    /// shared LFM2 generator. A loaded owner can retain a maximum-length table
    /// and pass contiguous prefix views to `forward_with_attention`.
    /// The reference's F32 frequency arithmetic rounds the JSON theta to F32;
    /// a cast that becomes zero or nonfinite is rejected by the shared helper.
    pub fn rope_tables(&self, length: usize) -> Result<(Tensor, Tensor)> {
        if length > self.cfg.max_position_embeddings {
            lloom_tensor::bail!("omni rotary table exceeds configured maximum positions")
        }
        let weights = self.embedding.embeddings();
        crate::nn::rotary_emb::default_rope_tables(
            self.cfg.hidden_size / self.cfg.num_attention_heads,
            self.cfg.rope_theta as f32,
            length,
            weights.dtype(),
            weights.device(),
        )
    }

    /// Host token validation precedes native gather. Media embeddings join the
    /// returned text embeddings in the request owner, before trunk execution.
    pub fn embed(&self, tokens: &[u32], batch: usize, length: usize) -> Result<Tensor> {
        if batch == 0
            || length == 0
            || product(&[batch, length])? != tokens.len()
            || tokens.iter().any(|&t| t as usize >= self.cfg.vocab_size)
        {
            lloom_tensor::bail!("omni embedding invalid token geometry or vocabulary index")
        }
        self.embedding.forward(&Tensor::from_slice(
            tokens,
            (batch, length),
            self.embedding.embeddings().device(),
        )?)
    }

    /// Right-padded rows, with media prefix <= real row length. Two compact
    /// mask rows per batch; zero-length query segments are omitted, not keys.
    /// Padding query positions still execute, as in the reference.
    pub fn prepare_batch(
        &self,
        length: usize,
        lengths: &[usize],
        prefixes: &[usize],
    ) -> Result<TrunkBatch> {
        let b = lengths.len();
        if b == 0
            || length == 0
            || length > self.cfg.max_position_embeddings
            || prefixes.len() != b
            || lengths
                .iter()
                .zip(prefixes)
                .any(|(&n, &p)| n == 0 || n > length || p > n)
        {
            lloom_tensor::bail!("omni trunk invalid sequence lengths or media prefixes")
        }
        let weights = self.embedding.embeddings();
        let n = product(&[b, length])?;
        product(&[
            b,
            length,
            self.cfg.hidden_size,
            weights.dtype().size_in_bytes(),
        ])?;
        product(&[n, 2, std::mem::size_of::<f32>()])?;
        let mask_rows = product(&[b, 2])?;
        let neg = if weights.dtype() == DType::F16 {
            -65504f32
        } else {
            -1e9f32
        };
        let mut pad = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        let mut keys = Vec::with_capacity(n * 2);
        let mut segments = Vec::with_capacity(mask_rows);
        for (row, (&real, &prefix)) in lengths.iter().zip(prefixes).enumerate() {
            pad.extend((0..length).map(|p| if p < real { 1f32 } else { 0. }));
            right.extend((0..length).map(|p| {
                if prefix != 0 && p == prefix - 1 {
                    0f32
                } else {
                    1.
                }
            }));
            for allowed in [prefix, real] {
                keys.extend((0..length).map(|k| if k < allowed { 0f32 } else { neg }));
            }
            if prefix != 0 {
                segments.push(QuerySegment {
                    batch_row: row,
                    start: 0,
                    length: prefix,
                    mask_row: 2 * row,
                });
            }
            if prefix < length {
                segments.push(QuerySegment {
                    batch_row: row,
                    start: prefix,
                    length: length - prefix,
                    mask_row: 2 * row + 1,
                });
            }
        }
        let dev = weights.device();
        Ok(TrunkBatch {
            batch: b,
            length,
            width: self.cfg.hidden_size,
            pad: Tensor::from_vec(pad, (b, length, 1), dev)?.to_dtype(weights.dtype())?,
            keep_right: Tensor::from_vec(right, (b, 1, length), dev)?.to_dtype(weights.dtype())?,
            key_rows: Tensor::from_vec(keys, (mask_rows, 1, 1, length), dev)?
                .to_dtype(weights.dtype())?,
            segments,
        })
    }

    /// Full bidirectional trunk over [B,L,D]. `cos/sin` are contiguous [L,Dh/2]
    /// NeoX tables on the weight dtype/device, prepared once by the caller.
    /// `rope_tables` prepares them through the shared default generator.
    /// The callback sees full K/V (already GQA-expanded), compact segment masks,
    /// and no causal flag. It owns bounded backend dispatch, not this block.
    pub fn forward_with_attention<F>(
        &self,
        hidden: &Tensor,
        batch: &TrunkBatch,
        cos: &Tensor,
        sin: &Tensor,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
    {
        let weights = self.embedding.embeddings();
        let d = self.cfg.hidden_size;
        if batch.length > self.cfg.max_position_embeddings
            || hidden.dims() != [batch.batch, batch.length, d]
            || batch.width != d
            || [
                hidden,
                &batch.pad,
                &batch.keep_right,
                &batch.key_rows,
                cos,
                sin,
            ]
            .iter()
            .any(|t| t.dtype() != weights.dtype() || !t.device().same_device(weights.device()))
            || cos.dims() != [batch.length, d / self.cfg.num_attention_heads / 2]
            || sin.dims() != cos.dims()
            || !cos.is_contiguous()
            || !sin.is_contiguous()
        {
            lloom_tensor::bail!("omni trunk input, prepared metadata and rotary tables must match shape/dtype/device")
        }
        let mut h = hidden.clone();
        for layer in &self.layers {
            let x = layer.operator_norm.forward(&h)?;
            let x = match &layer.mixer {
                Mixer::Attention(attention) => attention.forward(&x, batch, cos, sin, attend)?,
                Mixer::Conv {
                    input,
                    conv,
                    output,
                } => {
                    let projected = input
                        .forward(&x.broadcast_mul(&batch.pad)?)?
                        .transpose(1, 2)?;
                    let b = projected.narrow(1, 0, d)?;
                    let c = projected.narrow(1, d, d)?;
                    let u = projected.narrow(1, 2 * d, d)?;
                    output.forward(&conv.gated_sequence(&b, &u, &c, &batch.keep_right)?)?
                }
            };
            h = (&h + x)?;
            h = (&h + layer.ffn.forward(&layer.ffn_norm.forward(&h)?)?)?;
        }
        self.norm.forward(&h)
    }
}

#[cfg(test)]
mod tests;
