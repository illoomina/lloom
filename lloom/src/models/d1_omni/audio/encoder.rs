//! Pinned omni audio.py Conformer/adapter composition. The frontend supplies
//! normalized mel; the loaded family supplies admission and attention policy.
use super::relative::RelativeInputs;
use crate::models::siglip::Mlp;
use crate::nn::{
    batch_norm, conv1d, conv2d, layer_norm, linear, linear_no_bias, Activation, BatchNorm, Conv1d,
    Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};
use lloom_tensor::{DType, Device, Module, ModuleT, Result, Tensor};

/// The pinned 30-second,16kHz/hop160 frontend has3001 STFT rows,3000 valid.
const MAX_MEL_FRAMES: usize = 3001;
const MAX_ENCODED_FRAMES: usize = 376;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AudioConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling_conv_channels: usize,
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub residual_width: usize,
}
fn product(xs: &[usize]) -> Result<usize> {
    xs.iter()
        .try_fold(1usize, |n, &x| n.checked_mul(x))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni audio shape/byte overflow".into()))
}
fn ceil_half(n: usize) -> usize {
    n / 2 + n % 2
}
impl AudioConfig {
    fn validate(&self, out: usize, dtype: DType) -> Result<()> {
        let d = self.d_model;
        let c = self.subsampling_conv_channels;
        if [
            self.feat_in,
            self.n_layers,
            d,
            c,
            self.ff_expansion_factor,
            self.n_heads,
            self.conv_kernel_size,
            self.residual_width,
            out,
        ]
        .contains(&0)
            || d % 2 != 0
            || d % self.n_heads != 0
            || self.conv_kernel_size % 2 != 1
        {
            lloom_tensor::bail!("invalid omni audio dimensions, heads or odd convolution kernel")
        }
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            lloom_tensor::bail!("omni audio encoder supports F16/BF16/F32 explicitly")
        }
        let ff = product(&[d, self.ff_expansion_factor])?;
        let freq = ceil_half(ceil_half(ceil_half(self.feat_in)));
        let e = dtype.size_in_bytes();
        // Weight-class and worst-geometry numeric representability only.
        // This is not a total weight-load or whole-pass memory reservation.
        for dims in [
            &[c, 1, 3, 3][..],
            &[c, c, 1, 1],
            &[d, c, freq],
            &[self.n_layers, 4, d, ff],
            &[self.n_layers, 5, d, d],
            &[self.n_layers, 3, d, d],
            &[self.n_layers, d, self.conv_kernel_size],
            &[self.n_layers, 24, d],
            &[out, d],
            &[out, out],
            &[self.residual_width, out],
            &[out, self.residual_width],
            &[c, MAX_MEL_FRAMES, self.feat_in, 9],
            &[MAX_ENCODED_FRAMES, d, 2],
            &[MAX_ENCODED_FRAMES, ff],
            &[MAX_ENCODED_FRAMES, out],
            &[MAX_ENCODED_FRAMES, self.residual_width],
        ] {
            product(&[product(dims)?, e])?;
        }
        product(&[2 * MAX_ENCODED_FRAMES - 1, d, 4])?;
        let padded = MAX_ENCODED_FRAMES
            .checked_add(self.conv_kernel_size - 1)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("audio padded convolution length overflow".into())
            })?;
        product(&[padded, d, e])?;
        u32::try_from(d).map_err(|_| {
            lloom_tensor::Error::Msg("audio width exceeds U32 position generator".into())
        })?;
        product(&[self.n_layers, std::mem::size_of::<ConformerLayer>()])?;
        product(&[2, d])?;
        product(&[MAX_ENCODED_FRAMES, self.conv_kernel_size])?;
        Ok(())
    }
}
fn mlp(
    input: usize,
    hidden: usize,
    output: usize,
    a: &str,
    b: &str,
    act: Activation,
    vb: VarBuilder,
) -> Result<Mlp> {
    Ok(Mlp::from_parts(
        linear(input, hidden, vb.pp(a))?,
        linear(hidden, output, vb.pp(b))?,
        act,
    ))
}

#[derive(Debug)]
enum SubsampleOp {
    Conv(Conv2d),
    Relu,
}
#[derive(Debug)]
struct Subsampling {
    ops: Vec<SubsampleOp>,
    output: Linear,
}
impl Subsampling {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let c = cfg.subsampling_conv_channels;
        let strided = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let depth = Conv2dConfig {
            groups: c,
            ..strided
        };
        let point = Conv2dConfig::default();
        let ops = vec![
            SubsampleOp::Conv(conv2d(1, c, 3, strided, vb.pp("conv.0"))?),
            SubsampleOp::Relu,
            SubsampleOp::Conv(conv2d(c, c, 3, depth, vb.pp("conv.2"))?),
            SubsampleOp::Conv(conv2d(c, c, 1, point, vb.pp("conv.3"))?),
            SubsampleOp::Relu,
            SubsampleOp::Conv(conv2d(c, c, 3, depth, vb.pp("conv.5"))?),
            SubsampleOp::Conv(conv2d(c, c, 1, point, vb.pp("conv.6"))?),
            SubsampleOp::Relu,
        ];
        let freq = ceil_half(ceil_half(ceil_half(cfg.feat_in)));
        Ok(Self {
            ops,
            output: linear(product(&[c, freq])?, cfg.d_model, vb.pp("out"))?,
        })
    }
    fn forward(&self, mel: &Tensor, valid: usize) -> Result<(Tensor, usize)> {
        let mut t = mel.dim(2)?;
        let mut n = valid;
        let mut masks = Vec::with_capacity(4);
        // All four stage masks are compact, reused before each pinned op.
        for _ in 0..4 {
            let mask = Tensor::from_vec(
                (0..t)
                    .map(|i| if i < n { 1f32 } else { 0. })
                    .collect::<Vec<_>>(),
                (1, 1, t, 1),
                mel.device(),
            )?
            .to_dtype(mel.dtype())?;
            masks.push(mask);
            t = ceil_half(t);
            n = ceil_half(n);
        }
        let mut x = mel.transpose(1, 2)?.unsqueeze(1)?;
        let mut stage = 0;
        let mut valid = valid;
        for op in &self.ops {
            x = x.broadcast_mul(&masks[stage])?;
            x = match op {
                SubsampleOp::Conv(conv) => {
                    let out = conv.forward(&x)?;
                    if conv.config().stride != 1 {
                        stage += 1;
                        valid = ceil_half(valid);
                    }
                    out
                }
                SubsampleOp::Relu => x.relu()?,
            };
        }
        let x = x.broadcast_mul(&masks[stage])?;
        let (b, c, t, f) = x.dims4()?;
        Ok((
            self.output
                .forward(&x.transpose(1, 2)?.reshape((b, t, product(&[c, f])?))?)?,
            valid,
        ))
    }
}

#[derive(Debug)]
struct RelativeProjection {
    q: Linear,
    k: Linear,
    v: Linear,
    p: Linear,
    out: Linear,
    u: Tensor,
    v_bias: Tensor,
    heads: usize,
    width: usize,
}
impl RelativeProjection {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let h = cfg.n_heads;
        let width = d / h;
        Ok(Self {
            q: linear(d, d, vb.pp("linear_q"))?,
            k: linear(d, d, vb.pp("linear_k"))?,
            v: linear(d, d, vb.pp("linear_v"))?,
            p: linear_no_bias(d, d, vb.pp("linear_pos"))?,
            out: linear(d, d, vb.pp("linear_out"))?,
            u: vb.get((h, width), "pos_bias_u")?,
            v_bias: vb.get((h, width), "pos_bias_v")?,
            heads: h,
            width,
        })
    }
    fn forward<F>(&self, x: &Tensor, pos: &Tensor, valid: usize, attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let (b, t, d) = x.dims3()?;
        let shape = (b, t, self.heads, self.width);
        let q = self.q.forward(x)?.reshape(shape)?;
        let qc = q.broadcast_add(&self.u)?.transpose(1, 2)?.contiguous()?;
        let qp = q
            .broadcast_add(&self.v_bias)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let p = self
            .p
            .forward(pos)?
            .reshape((1, pos.dim(1)?, self.heads, self.width))?
            .transpose(1, 2)?
            .contiguous()?;
        let out = attend(RelativeInputs {
            content_query: &qc,
            position_query: &qp,
            key: &k,
            value: &v,
            positions: &p,
            valid,
        })?;
        if out.dims() != [b, self.heads, t, self.width]
            || out.dtype() != x.dtype()
            || !out.device().same_device(x.device())
        {
            lloom_tensor::bail!("audio attention callback must preserve shape, dtype and device")
        }
        self.out.forward(&out.transpose(1, 2)?.reshape((b, t, d))?)
    }
}

#[derive(Debug)]
struct ConvModule {
    input: Conv1d,
    depth: Conv1d,
    norm: BatchNorm,
    output: Conv1d,
    pad: usize,
    width: usize,
}
impl ConvModule {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let k = cfg.conv_kernel_size;
        Ok(Self {
            input: conv1d(d, 2 * d, 1, Default::default(), vb.pp("pointwise_conv1"))?,
            depth: conv1d(
                d,
                d,
                k,
                Conv1dConfig {
                    groups: d,
                    ..Default::default()
                },
                vb.pp("depthwise_conv"),
            )?,
            norm: batch_norm(d, 1e-5, vb.pp("batch_norm"))?,
            output: conv1d(d, d, 1, Default::default(), vb.pp("pointwise_conv2"))?,
            pad: (k - 1) / 2,
            width: d,
        })
    }
    fn forward(&self, x: &Tensor, valid_mask: &Tensor, zero: &Tensor) -> Result<Tensor> {
        let x = self.input.forward(&x.transpose(1, 2)?)?;
        let left = x.narrow(1, 0, self.width)?;
        let right = x.narrow(1, self.width, self.width)?;
        let x = left.mul(&crate::nn::ops::sigmoid(&right)?)?;
        let mask = valid_mask.broadcast_as(x.shape())?;
        let x = mask
            .where_cond(&x, &zero.broadcast_as(x.shape())?)?
            .pad_with_zeros(2, self.pad, self.pad)?;
        let x = self.depth.forward(&x)?;
        let x = self.norm.forward_t(&x, false)?.silu()?;
        self.output.forward(&x)?.transpose(1, 2)
    }
}
#[derive(Debug)]
struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: Mlp,
    norm_att: LayerNorm,
    att: RelativeProjection,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: Mlp,
    norm_out: LayerNorm,
}
impl ConformerLayer {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let ff = product(&[d, cfg.ff_expansion_factor])?;
        Ok(Self {
            norm_ff1: layer_norm(d, 1e-5, vb.pp("norm_feed_forward1"))?,
            ff1: mlp(
                d,
                ff,
                d,
                "linear1",
                "linear2",
                Activation::Silu,
                vb.pp("feed_forward1"),
            )?,
            norm_att: layer_norm(d, 1e-5, vb.pp("norm_self_att"))?,
            att: RelativeProjection::load(cfg, vb.pp("self_attn"))?,
            norm_conv: layer_norm(d, 1e-5, vb.pp("norm_conv"))?,
            conv: ConvModule::load(cfg, vb.pp("conv"))?,
            norm_ff2: layer_norm(d, 1e-5, vb.pp("norm_feed_forward2"))?,
            ff2: mlp(
                d,
                ff,
                d,
                "linear1",
                "linear2",
                Activation::Silu,
                vb.pp("feed_forward2"),
            )?,
            norm_out: layer_norm(d, 1e-5, vb.pp("norm_out"))?,
        })
    }
    fn forward<F>(
        &self,
        x: &Tensor,
        pos: &Tensor,
        valid: usize,
        mask: &Tensor,
        zero: &Tensor,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let x = (x + (self.ff1.forward(&self.norm_ff1.forward(x)?)? * 0.5)?)?;
        let x = (&x
            + self
                .att
                .forward(&self.norm_att.forward(&x)?, pos, valid, attend)?)?;
        let x = (&x
            + self
                .conv
                .forward(&self.norm_conv.forward(&x)?, mask, zero)?)?;
        let x = (&x + (self.ff2.forward(&self.norm_ff2.forward(&x)?)? * 0.5)?)?;
        self.norm_out.forward(&x)
    }
}

/// Loaded learned audio encoder, adapter and residual. No waveform decode or
/// host/device fallback is hidden here. VarBuilder supplies dtype/device, not
/// a ModelDomain; the eventual factory must establish shared load/pass scopes.
#[derive(Debug)]
pub struct AudioEncoder {
    cfg: AudioConfig,
    out: usize,
    subsample: Subsampling,
    layers: Vec<ConformerLayer>,
    positions: Tensor,
    adapter_norm: LayerNorm,
    adapter: Mlp,
    residual_norm: LayerNorm,
    residual: Mlp,
    dtype: DType,
    device: Device,
}
impl AudioEncoder {
    /// vb is rooted at the checkpoint's `audio` prefix.
    pub fn load(cfg: AudioConfig, out: usize, vb: VarBuilder) -> Result<Self> {
        cfg.validate(out, vb.dtype())?;
        let d = cfg.d_model;
        let subsample = Subsampling::load(&cfg, vb.pp("encoder.pre_encode"))?;
        let layers = (0..cfg.n_layers)
            .map(|i| ConformerLayer::load(&cfg, vb.pp(format!("encoder.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        // Source specifically builds F32 on CPU, even for a device/half model.
        // Frequencies use exp, not the pow-based default RoPE generator, and
        // layout is interleaved, not Whisper/Gemma's concatenated halves.
        let index =
            Tensor::arange_step(0u32, d as u32, 2u32, &Device::Cpu)?.to_dtype(DType::F32)?;
        let inv = index.affine(-(10000f64).ln() / d as f64, 0.)?.exp()?;
        let pos = Tensor::arange_step(
            (MAX_ENCODED_FRAMES - 1) as f32,
            -(MAX_ENCODED_FRAMES as f32),
            -1f32,
            &Device::Cpu,
        )?;
        let angles = pos.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)?;
        let positions = Tensor::stack(&[angles.sin()?, angles.cos()?], 2)?
            .flatten_from(1)?
            .unsqueeze(0)?
            .to_device(vb.device())?
            .to_dtype(vb.dtype())?;
        Ok(Self {
            adapter_norm: layer_norm(d, 1e-5, vb.pp("adapter.norm"))?,
            adapter: mlp(
                d,
                out,
                out,
                "linear_1",
                "linear_2",
                Activation::Gelu,
                vb.pp("adapter"),
            )?,
            residual_norm: layer_norm(out, 1e-5, vb.pp("residual.ln"))?,
            residual: mlp(
                out,
                cfg.residual_width,
                out,
                "down",
                "up",
                Activation::Gelu,
                vb.pp("residual"),
            )?,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            cfg,
            out,
            subsample,
            layers,
            positions,
        })
    }
    /// Input is one already-normalized mel clip[1,F,T] in the model dtype.
    /// `valid` is the frontend's floor(samples/hop), not its extra STFT endpoint.
    /// Caller reserves the complete load/pass and asynchronous retirement, not
    /// merely each relative-attention row's scratch. This API is not admission.
    pub fn forward_with_attention<F>(
        &self,
        mel: &Tensor,
        valid: usize,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let (b, f, t) = mel.dims3()?;
        if b != 1
            || f != self.cfg.feat_in
            || t == 0
            || t > MAX_MEL_FRAMES
            || valid == 0
            || valid > t
            || mel.dtype() != self.dtype
            || !mel.device().same_device(&self.device)
        {
            lloom_tensor::bail!(
                "omni audio requires model-typed [1,F,T<=3001] mel and positive valid<=T"
            )
        }
        product(&[f, t, self.dtype.size_in_bytes()])?;
        let (mut x, valid) = self.subsample.forward(mel, valid)?;
        let time = x.dim(1)?;
        if time > MAX_ENCODED_FRAMES {
            lloom_tensor::bail!("audio subsampling exceeded relative table bound")
        }
        let pos = self
            .positions
            .narrow(1, MAX_ENCODED_FRAMES - time, 2 * time - 1)?
            .contiguous()?;
        let mask = Tensor::from_vec(
            (0..time).map(|i| u8::from(i < valid)).collect::<Vec<_>>(),
            (1, 1, time),
            &self.device,
        )?;
        let zero = Tensor::zeros((1, 1, time), self.dtype, &self.device)?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos, valid, &mask, &zero, attend)?;
        }
        let x = x.narrow(1, 0, valid)?;
        let x = self.adapter.forward(&self.adapter_norm.forward(&x)?)?;
        let output = (&x + self.residual.forward(&self.residual_norm.forward(&x)?)?)?;
        if output.dims() != [1, valid, self.out] {
            lloom_tensor::bail!("omni audio output shape mismatch")
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
