use super::*;
use crate::models::d1_omni::weights::{
    LinearWeights, NormWeights, Weight, WeightDType, WeightPlan,
};

fn f32_weight(name: impl Into<String>, shape: impl Into<lloom_tensor::Shape>) -> Weight {
    Weight::new(name, shape).dtype(WeightDType::F32)
}

#[derive(Debug)]
pub(super) struct ConvWeights {
    pub input: [Weight; 2],
    pub depth: [Weight; 2],
    pub norm: [Weight; 2],
    pub output: [Weight; 2],
}
impl ConvWeights {
    pub fn new(cfg: &AudioConfig, prefix: &str) -> Result<Self> {
        cfg.validate(cfg.d_model, DType::F32)?;
        let d = cfg.d_model;
        let twice = product(&[2, d])?;
        let key = |name: &str| {
            if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            }
        };
        Ok(Self {
            input: [
                f32_weight(key("conv_pw1.weight"), (twice, d)),
                f32_weight(key("conv_pw1.bias"), twice),
            ],
            depth: [
                f32_weight(key("conv_dw.weight"), (d, cfg.conv_kernel_size)),
                f32_weight(key("conv_dw.bias"), d),
            ],
            norm: [
                f32_weight(key("conv_norm.weight"), d),
                f32_weight(key("conv_norm.bias"), d),
            ],
            output: [
                f32_weight(key("conv_pw2.weight"), (d, d)),
                f32_weight(key("conv_pw2.bias"), d),
            ],
        })
    }
}
impl WeightPlan for ConvWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.input.as_slice().visit(f)?;
        self.depth.as_slice().visit(f)?;
        self.norm.as_slice().visit(f)?;
        self.output.as_slice().visit(f)
    }
}

#[derive(Debug)]
pub(super) struct SubsampleConvWeights {
    pub weight: Weight,
    pub bias: Weight,
    pub kernel: usize,
    pub stride: usize,
    pub groups: usize,
    pub relu: bool,
}
impl WeightPlan for SubsampleConvWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.weight.visit(f)?;
        self.bias.visit(f)
    }
}
#[derive(Debug)]
pub(super) struct RelativeWeights {
    pub projections: [LinearWeights; 5],
    pub u: Weight,
    pub v: Weight,
}
impl RelativeWeights {
    fn new(cfg: &AudioConfig, prefix: &str) -> Self {
        let d = cfg.d_model;
        Self {
            projections: ["attn_q", "attn_k", "attn_v", "linear_pos", "attn_out"].map(|name| {
                LinearWeights::new(&format!("{prefix}.{name}"), d, d, name != "linear_pos")
            }),
            u: Weight::new(
                format!("{prefix}.pos_bias_u"),
                (cfg.n_heads, d / cfg.n_heads),
            ),
            v: Weight::new(
                format!("{prefix}.pos_bias_v"),
                (cfg.n_heads, d / cfg.n_heads),
            ),
        }
    }
}
impl WeightPlan for RelativeWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.projections.as_slice().visit(f)?;
        self.u.visit(f)?;
        self.v.visit(f)
    }
}
#[derive(Debug)]
pub(super) struct BlockWeights {
    pub norms: [NormWeights; 5],
    pub ff1: [LinearWeights; 2],
    pub attention: RelativeWeights,
    pub conv: ConvWeights,
    pub ff2: [LinearWeights; 2],
}
impl WeightPlan for BlockWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.norms[0].visit(f)?;
        self.ff1.as_slice().visit(f)?;
        self.norms[1].visit(f)?;
        self.attention.visit(f)?;
        self.norms[2].visit(f)?;
        self.conv.visit(f)?;
        self.norms[3].visit(f)?;
        self.ff2.as_slice().visit(f)?;
        self.norms[4].visit(f)
    }
}

#[derive(Debug)]
pub(in crate::models::d1_omni) struct AudioWeights {
    pub(super) subsample: Vec<SubsampleConvWeights>,
    pub(super) subsample_output: LinearWeights,
    pub(super) blocks: Vec<BlockWeights>,
    pub(super) adapter_norm: NormWeights,
    pub(super) adapter: [LinearWeights; 2],
    pub(super) residual_norm: NormWeights,
    pub(super) residual: [LinearWeights; 2],
}
impl AudioWeights {
    pub(in crate::models::d1_omni) fn new(cfg: &AudioConfig, out: usize) -> Result<Self> {
        QuantizedAudioEncoder::validate_config(cfg, out)?;
        let d = cfg.d_model;
        let c = cfg.subsampling_conv_channels;
        let ff = product(&[d, cfg.ff_expansion_factor])?;
        let mut subsample = crate::models::d1_omni::host::prepared_vec(5)?;
        for (index, input, kernel, stride, groups, relu) in [
            (0, 1, 3, 2, 1, true),
            (2, c, 3, 2, c, false),
            (3, c, 1, 1, 1, true),
            (5, c, 3, 2, c, false),
            (6, c, 1, 1, 1, true),
        ] {
            subsample.push(SubsampleConvWeights {
                weight: f32_weight(
                    format!("a.conv1d.{index}.weight"),
                    (c, input / groups, kernel, kernel),
                ),
                bias: f32_weight(format!("a.conv1d.{index}.bias"), (c, 1, 1)),
                kernel,
                stride,
                groups,
                relu,
            });
        }
        let mut blocks = crate::models::d1_omni::host::prepared_vec(cfg.n_layers)?;
        for i in 0..cfg.n_layers {
            let p = format!("a.blk.{i}");
            blocks.push(BlockWeights {
                norms: ["ffn_norm", "ln1", "norm_conv", "ffn_norm_1", "ln2"]
                    .map(|name| NormWeights::new(&format!("{p}.{name}"), d)),
                ff1: [
                    LinearWeights::new(&format!("{p}.ffn_up"), d, ff, true),
                    LinearWeights::new(&format!("{p}.ffn_down"), ff, d, true),
                ],
                attention: RelativeWeights::new(cfg, &p),
                conv: ConvWeights::new(cfg, &p)?,
                ff2: [
                    LinearWeights::new(&format!("{p}.ffn_up_1"), d, ff, true),
                    LinearWeights::new(&format!("{p}.ffn_down_1"), ff, d, true),
                ],
            });
        }
        Ok(Self {
            subsample,
            subsample_output: LinearWeights::new(
                "a.pre_encode.out",
                product(&[c, ceil_half(ceil_half(ceil_half(cfg.feat_in)))])?,
                d,
                true,
            ),
            blocks,
            adapter_norm: NormWeights::new("mm.a.mlp.0", d),
            adapter: [
                LinearWeights::new("mm.a.mlp.1", d, out, true),
                LinearWeights::new("mm.a.mlp.3", out, out, true),
            ],
            residual_norm: NormWeights::new("mm.a.mlp.4", out),
            residual: [
                LinearWeights::new("mm.a.mlp.5", out, cfg.residual_width, true),
                LinearWeights::new("mm.a.mlp.6", cfg.residual_width, out, true),
            ],
        })
    }
}
impl WeightPlan for AudioWeights {
    fn visit(&self, f: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.subsample.as_slice().visit(f)?;
        self.subsample_output.visit(f)?;
        self.blocks.as_slice().visit(f)?;
        self.adapter_norm.visit(f)?;
        self.adapter.as_slice().visit(f)?;
        self.residual_norm.visit(f)?;
        self.residual.as_slice().visit(f)
    }
}
