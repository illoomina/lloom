//! Siglip model implementation.
//!
//! Siglip architecture combining vision and language for zero-shot tasks.
//!
//! References:
//! - 🤗 [Model Card](https://huggingface.co/google/siglip-base-patch16-224)
//!

use crate::models::clip::div_l2_norm;
use crate::nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};
use lloom_tensor::{IndexOp, Module, Result, Tensor, D};

fn default_text_vocab_size() -> usize {
    32000
}

fn default_text_hidden_size() -> usize {
    768
}

fn default_text_intermediate_size() -> usize {
    3072
}

fn default_text_num_hidden_layers() -> usize {
    12
}

fn default_text_num_attention_heads() -> usize {
    12
}

fn default_text_max_position_embeddings() -> usize {
    64
}

fn default_text_layer_norm_eps() -> f64 {
    1e-6
}

fn default_text_pad_token_id() -> u32 {
    1
}

fn default_text_bos_token_id() -> u32 {
    49406
}

fn default_text_eos_token_id() -> u32 {
    49407
}

fn default_text_hidden_act() -> crate::nn::Activation {
    crate::nn::Activation::GeluPytorchTanh
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L27
#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: crate::nn::Activation,
    #[serde(default = "default_text_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_text_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_text_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_text_eos_token_id")]
    pub eos_token_id: u32,
}

fn default_vision_hidden_size() -> usize {
    768
}

fn default_vision_intermediate_size() -> usize {
    3072
}

fn default_vision_num_hidden_layers() -> usize {
    12
}

fn default_vision_num_attention_heads() -> usize {
    12
}

fn default_vision_num_channels() -> usize {
    3
}

fn default_vision_image_size() -> usize {
    224
}

fn default_vision_batch_size() -> usize {
    16
}

fn default_vision_layer_norm_eps() -> f64 {
    1e-6
}

fn default_vision_hidden_act() -> crate::nn::Activation {
    crate::nn::Activation::GeluPytorchTanh
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L132
#[derive(serde::Deserialize, Clone, Debug)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,
    #[serde(default = "default_vision_batch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: crate::nn::Activation,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

pub(crate) trait TransformerConfig {
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn layer_norm_eps(&self) -> f64;
    fn hidden_act(&self) -> crate::nn::Activation;
}

impl TransformerConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> crate::nn::Activation {
        self.hidden_act
    }
}

impl TransformerConfig for VisionConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> crate::nn::Activation {
        self.hidden_act
    }
}

impl VisionConfig {
    pub fn paligemma_3b_224() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 224, // num_image_tokens: (224 / 14)^2 = 256
            // Default values.
            num_channels: 3,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn paligemma_3b_448() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-448/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 448, // num_image_tokens: (448 / 14)^2 = 1024
            // Default values.
            num_channels: 3,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn paligemma_3b_896() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-448/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 896, // num_image_tokens: (896 / 14)^2 = 4096
            // Default values.
            num_channels: 3,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L228
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

impl Config {
    pub fn base_patch16_224() -> Self {
        let text_config = TextConfig {
            // https://huggingface.co/google/siglip-base-patch16-224/blob/main/config.json
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            vocab_size: 32000,
            // Default values.
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
            layer_norm_eps: 1e-6,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
            max_position_embeddings: 64,
            num_hidden_layers: 12,
        };
        let vision_config = VisionConfig {
            patch_size: 16,
            // Default values.
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        };
        Self {
            text_config,
            vision_config,
        }
    }
}

#[derive(Clone, Debug)]
struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl MultiheadAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let w_in_proj = vb.get((3 * h, h), "in_proj_weight")?.chunk(3, 0)?;
        let b_in_proj = vb.get(3 * h, "in_proj_bias")?.chunk(3, 0)?;
        let q_proj = Linear::new(w_in_proj[0].clone(), Some(b_in_proj[0].clone()));
        let k_proj = Linear::new(w_in_proj[1].clone(), Some(b_in_proj[1].clone()));
        let v_proj = Linear::new(w_in_proj[2].clone(), Some(b_in_proj[2].clone()));
        let out_proj = linear(h, h, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(&q.contiguous()?)?;
        let k = self.k_proj.forward(&k.contiguous()?)?;
        let v = self.v_proj.forward(&v.contiguous()?)?;

        let q = self.separate_heads(&q)?;
        let k = self.separate_heads(&k)?;
        let v = self.separate_heads(&v)?;

        let (_, _, _, c_per_head) = q.dims4()?;
        let attn = (q.matmul(&k.t()?)? / (c_per_head as f64).sqrt())?;
        let attn = crate::nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        self.recombine_heads(&out)?.apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct MultiheadAttentionPoolingHead {
    probe: Tensor,
    attention: MultiheadAttention,
    layernorm: LayerNorm,
    mlp: Mlp,
}

impl MultiheadAttentionPoolingHead {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        let probe = vb.get((1, 1, cfg.hidden_size), "probe")?;
        let attention = MultiheadAttention::new(cfg, vb.pp("attention"))?;
        Ok(Self {
            probe,
            attention,
            layernorm,
            mlp,
        })
    }
}

impl Module for MultiheadAttentionPoolingHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let batch_size = xs.dim(0)?;
        let probe = self.probe.repeat((batch_size, 1, 1))?;
        let xs = self.attention.forward(&probe, xs, xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layernorm)?.apply(&self.mlp)?;
        (xs + residual)?.i((.., 0))
    }
}

/// Existing SigLIP default: preserve operation order and dtype.
fn dense_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let weights = (q.matmul(&k.t()?)? * scale)?;
    let weights = match attention_mask {
        None => weights,
        Some(mask) => weights.broadcast_add(mask)?,
    };
    crate::nn::ops::softmax_last_dim(&weights)?.matmul(v)
}

#[derive(Debug, Clone)]
pub(crate) struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    /// Compose the same attention block from already loaded projections.
    /// Packed-QKV checkpoints split their weights once at load, not per pass.
    /// This does not select a new attention backend or change `forward`.
    pub(crate) fn from_projections(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        num_heads: usize,
    ) -> Result<Self> {
        let (embed_dim, input_dim) = q_proj.weight().dims2()?;
        if embed_dim == 0 || input_dim != embed_dim || num_heads == 0 {
            lloom_tensor::bail!(
                "attention components require square nonempty projections and positive heads"
            )
        }
        if embed_dim % num_heads != 0 {
            lloom_tensor::bail!("attention width {embed_dim} is not divisible by {num_heads} heads")
        }
        for (name, projection) in [
            ("query", &q_proj),
            ("key", &k_proj),
            ("value", &v_proj),
            ("output", &out_proj),
        ] {
            if projection.weight().dims2()? != (embed_dim, embed_dim) {
                lloom_tensor::bail!(
                    "attention {name} projection must be {embed_dim} by {embed_dim}"
                )
            }
            if let Some(bias) = projection.bias() {
                if bias.dims1()? != embed_dim {
                    lloom_tensor::bail!("attention {name} bias must have {embed_dim} elements")
                }
            }
        }
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    pub(crate) fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size();
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let num_heads = cfg.num_attention_heads();
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_attention(xs, attention_mask, &mut dense_attention)
    }

    /// Reuse projections with caller-owned, monomorphized attention dispatch.
    /// The callback receives contiguous [B,H,L,D] Q/K/V, the unchanged mask
    /// and scale, and returns [B,H,L,D] on the same device and in the same dtype.
    /// Compact-mask policies may live in the callback rather than allocating
    /// a full L-by-L mask before this call. This seam itself sets no budget.
    pub(crate) fn forward_with_attention<F>(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attended = attend(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            self.scale,
        )?;
        if attended.dims() != [batch_size, self.num_heads, q_len, self.head_dim]
            || attended.dtype() != query_states.dtype()
            || !attended.device().same_device(query_states.device())
        {
            lloom_tensor::bail!(
                "attention callback must preserve [B,H,L,D] shape, dtype and device"
            )
        }
        let attn_outputs = attended
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?
            .apply(&self.out_proj)?;
        Ok(attn_outputs)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L599
#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: crate::nn::Activation,
}

impl Mlp {
    /// Compose existing feed-forward execution with another checkpoint spelling.
    pub(crate) fn from_parts(
        fc1: Linear,
        fc2: Linear,
        activation_fn: crate::nn::Activation,
    ) -> Self {
        Self {
            fc1,
            fc2,
            activation_fn,
        }
    }

    pub(crate) fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let intermediate_size = cfg.intermediate_size();
        let fc1 = crate::nn::linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = crate::nn::linear(intermediate_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation_fn: cfg.hidden_act(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &lloom_tensor::Tensor) -> Result<lloom_tensor::Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L614
#[derive(Debug, Clone)]
pub(crate) struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    /// Assemble pre-norm residual blocks without changing their execution order.
    pub(crate) fn from_parts(
        self_attn: Attention,
        layer_norm1: LayerNorm,
        mlp: Mlp,
        layer_norm2: LayerNorm,
    ) -> Self {
        Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        }
    }

    pub(crate) fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let layer_norm_eps = cfg.layer_norm_eps();
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_attention(xs, attention_mask, &mut dense_attention)
    }

    pub(crate) fn forward_with_attention<F>(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self
            .self_attn
            .forward_with_attention(&xs, attention_mask, attend)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    /// A loader may assemble the same stack from differently named components.
    pub(crate) fn from_layers(layers: Vec<EncoderLayer>) -> Self {
        Self { layers }
    }

    pub(crate) fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers() {
            let layer = EncoderLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    pub(crate) fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_attention(xs, attention_mask, &mut dense_attention)
    }

    pub(crate) fn forward_with_attention<F>(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_with_attention(&xs, attention_mask, attend)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: crate::nn::Conv2d,
    position_embedding: Tensor,
    patch_size: usize,
    base_num_patches_per_side: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = crate::nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = crate::nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_patches_per_side = cfg.image_size / cfg.patch_size;
        let embedder = crate::nn::embedding(
            num_patches_per_side.pow(2),
            cfg.hidden_size(),
            vb.pp("position_embedding"),
        )?;
        let position_embedding = embedder.embeddings();
        let position_embedding = position_embedding
            .reshape((
                1,
                num_patches_per_side,
                num_patches_per_side,
                cfg.hidden_size(),
            ))?
            .permute((0, 3, 1, 2))?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            patch_size: cfg.patch_size,
            base_num_patches_per_side: num_patches_per_side,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //embed tokens
        let (_batch, _channels, _height, _width) = xs.dims4()?;
        let embeddings = xs.apply(&self.patch_embedding)?;
        // interpolate position embeddings for the current image size (if needed)
        let num_patches_h = _height / self.patch_size;
        let num_patches_w = _width / self.patch_size;
        let resized_position_embedding = if num_patches_w == self.base_num_patches_per_side
            && num_patches_h == self.base_num_patches_per_side
        {
            self.position_embedding.clone()
        } else {
            // Antialiased bilinear, matching the reference's
            // `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`.
            //
            // This called `interpolate2d` until lloom #609, which is
            // `upsample_nearest2d` despite the name — a silent-wrong that changes
            // no shape, no dtype and no token count, so the model ran and emitted
            // fluent output about subtly-wrong features at any non-base image
            // size. Measured on LFM2-VL's 16x16 grid resampled to 8x8, nearest
            // reads 9.15e0 against the reference where this reads ~4.77e-7.
            crate::utils::interpolate2d_bilinear_antialias(
                &self.position_embedding,
                num_patches_h,
                num_patches_w,
            )?
        };
        // Add position embeddings to tokens and flatten from 2D patches to 1D sequence
        let embeddings = embeddings
            .broadcast_add(&resized_position_embedding)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    head: Option<MultiheadAttentionPoolingHead>,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, use_head: bool, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        let head = if use_head {
            Some(MultiheadAttentionPoolingHead::new(cfg, vb.pp("head"))?)
        } else {
            None
        };
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            head,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.embeddings)?;
        let xs = self.encoder.forward(&xs, None)?;
        let xs = xs.apply(&self.post_layernorm)?;
        match self.head.as_ref() {
            None => Ok(xs),
            Some(h) => xs.apply(h),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    vision_model: VisionTransformer,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, use_head: bool, vb: VarBuilder) -> Result<Self> {
        let vision_model = VisionTransformer::new(cfg, use_head, vb)?;
        Ok(Self { vision_model })
    }
}

impl Module for VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.vision_model)
    }
}

#[derive(Debug, Clone)]
struct TextEmbeddings {
    token_embedding: crate::nn::Embedding,
    position_embedding: crate::nn::Embedding,
    position_ids: Tensor,
}

impl TextEmbeddings {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding =
            crate::nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embedding"))?;
        let position_embedding = crate::nn::embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for TextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

#[derive(Debug, Clone)]
pub struct TextTransformer {
    embeddings: TextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
    pub head: Linear,
}

impl TextTransformer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;
        let head = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            head,
        })
    }
}
impl Module for TextTransformer {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsz, seq_len) = input_ids.dims2()?;
        let input_ids = self.embeddings.forward(input_ids)?;
        let input_ids = self.encoder.forward(&input_ids, None)?;
        let last_hidden_state = self.final_layer_norm.forward(&input_ids)?;
        last_hidden_state
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.head)
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    pub text_model: TextTransformer,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let text_model = TextTransformer::new(cfg, vb)?;
        Ok(Self { text_model })
    }
}

impl Module for TextModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.text_model)
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    text_model: TextModel,
    vision_model: VisionModel,
    logit_bias: Tensor,
    logit_scale: Tensor,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_model = TextModel::new(&cfg.text_config, vb.pp("text_model"))?;
        let vision_model = VisionModel::new(&cfg.vision_config, true, vb.pp("vision_model"))?;
        let logit_scale = vb.get(&[1], "logit_scale")?;
        let logit_bias = vb.get(&[1], "logit_bias")?;
        Ok(Self {
            text_model,
            vision_model,
            logit_bias,
            logit_scale,
        })
    }

    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        input_ids.apply(&self.text_model)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values.apply(&self.vision_model)
    }

    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids)?;
        let image_features_normalized = div_l2_norm(&image_features)?;
        let text_features_normalized = div_l2_norm(&text_features)?;
        let logits_per_text = text_features_normalized.matmul(&image_features_normalized.t()?)?;
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text
            .broadcast_mul(&logit_scale)?
            .broadcast_add(&self.logit_bias)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}

#[cfg(test)]
mod component_tests {
    use super::*;
    use lloom_tensor::{DType, Device};
    use std::collections::HashMap;

    fn config() -> VisionConfig {
        VisionConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_channels: 3,
            image_size: 2,
            patch_size: 1,
            hidden_act: crate::nn::Activation::Relu,
            layer_norm_eps: 1e-5,
        }
    }

    fn weights(cfg: &VisionConfig) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for l in 0..cfg.num_hidden_layers {
            for (name, rows, cols) in [
                ("self_attn.q_proj", 4, 4),
                ("self_attn.k_proj", 4, 4),
                ("self_attn.v_proj", 4, 4),
                ("self_attn.out_proj", 4, 4),
                ("mlp.fc1", 8, 4),
                ("mlp.fc2", 4, 8),
            ] {
                let values: Vec<f32> = (0..rows * cols)
                    .map(|i| ((i + 3 * l) % 11) as f32 / 32. - 0.125)
                    .collect();
                tensors.insert(
                    format!("layers.{l}.{name}.weight"),
                    Tensor::from_vec(values, (rows, cols), &Device::Cpu)?,
                );
                tensors.insert(
                    format!("layers.{l}.{name}.bias"),
                    Tensor::from_vec(
                        (0..rows).map(|i| i as f32 / 64.).collect::<Vec<_>>(),
                        rows,
                        &Device::Cpu,
                    )?,
                );
            }
            for name in ["layer_norm1", "layer_norm2"] {
                tensors.insert(
                    format!("layers.{l}.{name}.weight"),
                    Tensor::from_slice(&[0.75f32, 1., 1.25, 1.5], 4, &Device::Cpu)?,
                );
                tensors.insert(
                    format!("layers.{l}.{name}.bias"),
                    Tensor::from_slice(&[-0.125f32, 0., 0.125, 0.25], 4, &Device::Cpu)?,
                );
            }
        }
        Ok(tensors)
    }

    #[test]
    fn component_stack_matches_existing_loader_with_and_without_additive_mask() -> Result<()> {
        let cfg = config();
        let vb = VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu);
        let original = Encoder::new(&cfg, vb.clone())?;
        let mut layers = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            let v = vb.pp("layers").pp(l);
            let a = v.pp("self_attn");
            let attention = Attention::from_projections(
                linear(4, 4, a.pp("q_proj"))?,
                linear(4, 4, a.pp("k_proj"))?,
                linear(4, 4, a.pp("v_proj"))?,
                linear(4, 4, a.pp("out_proj"))?,
                2,
            )?;
            let mlp = Mlp::from_parts(
                linear(4, 8, v.pp("mlp.fc1"))?,
                linear(8, 4, v.pp("mlp.fc2"))?,
                cfg.hidden_act,
            );
            layers.push(EncoderLayer::from_parts(
                attention,
                layer_norm(4, cfg.layer_norm_eps, v.pp("layer_norm1"))?,
                mlp,
                layer_norm(4, cfg.layer_norm_eps, v.pp("layer_norm2"))?,
            ));
        }
        let composed = Encoder::from_layers(layers);
        let x = Tensor::from_vec(
            (0..24)
                .map(|i| (i % 13) as f32 / 8. - 0.75)
                .collect::<Vec<_>>(),
            (2, 3, 4),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_slice(
            &[0f32, -2., -1e4, -1e4, 0.5, 0., 0., -1e4, -0.5],
            (1, 1, 3, 3),
            &Device::Cpu,
        )?;
        // Captured from the original production forward before callback factoring.
        // The raw pre-factor log records source hash and exact machine/command.
        // Portable tolerance avoids asserting cross-architecture reduction bits.
        let golden_bits: [[u32; 24]; 2] = [
            [
                3206781994, 3203708153, 3201896840, 1018045364, 3188617388, 1018474628, 1033615838,
                1057321246, 1052297129, 1057334660, 1058240316, 1065531535, 1062202343, 3206063862,
                3205099284, 3196998335, 3196307605, 3189721012, 3161067640, 1053239214, 1048080086,
                1051745319, 1056485244, 1063490520,
            ],
            [
                3206781995, 3203708154, 3201896840, 1018045352, 3188617388, 1018474636, 1033615836,
                1057321246, 1052297129, 1057334660, 1058240316, 1065531535, 1062208295, 3207627715,
                3203783499, 3197448061, 3196532651, 3184704685, 3175653896, 1053484064, 1048436732,
                1050127430, 1057267118, 1063364322,
            ],
        ];
        for (case, mask) in [None, Some(&mask)].into_iter().enumerate() {
            let expected = original
                .forward(&x, mask)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = composed
                .forward(&x, mask)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(expected.iter().all(|x| x.is_finite()));
            for (&value, &bits) in expected.iter().zip(&golden_bits[case]) {
                assert!((value - f32::from_bits(bits)).abs() <= 1e-6);
            }
            let mut calls = 0;
            let via_callback = composed
                .forward_with_attention(&x, mask, &mut |q, k, v, seen_mask, scale| {
                    calls += 1;
                    assert_eq!(q.dims(), [2, 2, 3, 2]);
                    assert_eq!(q.dims(), k.dims());
                    assert_eq!(q.dims(), v.dims());
                    assert!(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
                    assert_eq!(scale.to_bits(), (2f64).powf(-0.5).to_bits());
                    assert_eq!(
                        seen_mask.map(|m| m as *const Tensor),
                        mask.map(|m| m as *const Tensor)
                    );
                    dense_attention(q, k, v, seen_mask, scale)
                })?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(calls, 2);
            assert_eq!(
                via_callback.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            println!(
                "SIGLIP_GOLDEN_MASK={}; BITS={:?}",
                mask.is_some(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn projection_constructor_rejects_bad_geometry_before_forward() -> Result<()> {
        let make = |rows, cols, bias: Option<usize>| -> Result<Linear> {
            Ok(Linear::new(
                Tensor::zeros((rows, cols), DType::F32, &Device::Cpu)?,
                bias.map(|n| Tensor::zeros(n, DType::F32, &Device::Cpu))
                    .transpose()?,
            ))
        };
        let valid = make(4, 4, Some(4))?;
        for heads in [0, 3, 5] {
            assert!(Attention::from_projections(
                valid.clone(),
                valid.clone(),
                valid.clone(),
                valid.clone(),
                heads
            )
            .is_err());
        }
        for bad in [make(0, 0, None)?, make(4, 3, None)?, make(4, 4, Some(3))?] {
            assert!(Attention::from_projections(
                bad.clone(),
                valid.clone(),
                valid.clone(),
                valid.clone(),
                2
            )
            .is_err());
            assert!(Attention::from_projections(
                valid.clone(),
                bad.clone(),
                valid.clone(),
                valid.clone(),
                2
            )
            .is_err());
            assert!(Attention::from_projections(
                valid.clone(),
                valid.clone(),
                bad.clone(),
                valid.clone(),
                2
            )
            .is_err());
            assert!(Attention::from_projections(
                valid.clone(),
                valid.clone(),
                valid.clone(),
                bad,
                2
            )
            .is_err());
        }
        assert!(Attention::from_projections(
            make(4, 4, None)?,
            valid.clone(),
            valid.clone(),
            valid,
            2
        )
        .is_ok());
        Ok(())
    }

    #[test]
    fn callback_output_contract_rejects_shape_and_dtype_and_propagates_errors() -> Result<()> {
        let cfg = config();
        let vb = VarBuilder::from_tensors(weights(&cfg)?, DType::F32, &Device::Cpu);
        let attention = Attention::new(&cfg, vb.pp("layers.0.self_attn"))?;
        let x = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
        let original = attention
            .forward(&x, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let alternate = attention
            .forward_with_attention(&x, None, &mut dense_attention)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(original, alternate);
        for dtype in [DType::F64, DType::U32] {
            let error = attention
                .forward_with_attention(&x, None, &mut |q, _, _, _, _| {
                    Tensor::zeros(q.shape(), dtype, &Device::Cpu)
                })
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("attention callback must preserve"));
        }
        let error = attention
            .forward_with_attention(&x, None, &mut |_, _, _, _, _| {
                Tensor::zeros((2, 3, 2, 2), DType::F32, &Device::Cpu)
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("attention callback must preserve"));
        let error = attention
            .forward_with_attention(&x, None, &mut |_, _, _, _, _| {
                lloom_tensor::bail!("caller scratch admission refused")
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("caller scratch admission refused"));
        let layer = EncoderLayer::new(&cfg, vb.pp("layers.0"))?;
        let old = layer.forward(&x, None)?.flatten_all()?.to_vec1::<f32>()?;
        let new = layer
            .forward_with_attention(&x, None, &mut dense_attention)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(old, new);
        Ok(())
    }
}
