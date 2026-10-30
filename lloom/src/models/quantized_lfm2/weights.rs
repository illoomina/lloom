//! Causal LFM2 GGUF demands: resolved aliases are shared by preflight and load.
use super::{read_usize_list, value_to_usize};
use crate::nn::gguf_weights::{Weight, WeightDType, WeightPlan};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::{bail, Result, Shape};

pub(crate) struct AttentionWeights {
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub output: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub kv_heads: usize,
}
pub(crate) struct ConvWeights {
    pub input: Weight,
    pub output: Weight,
    pub conv: Weight,
}
pub(crate) enum MixerWeights {
    Attention(AttentionWeights),
    Conv(ConvWeights),
}
pub(crate) struct LayerWeights {
    pub operator_norm: Weight,
    pub ffn_norm: Weight,
    pub gate: Weight,
    pub down: Weight,
    pub up: Weight,
    pub mixer: MixerWeights,
}
pub(crate) struct Lfm2Weights {
    pub heads: usize,
    pub width: usize,
    pub context: usize,
    pub epsilon: f64,
    pub rope_base: f32,
    pub kernel: usize,
    pub vocab: usize,
    pub embedding: Weight,
    pub norm: Weight,
    pub output: Option<Weight>,
    pub layers: Vec<LayerWeights>,
}
fn info<'a>(
    ct: &'a Content,
    name: &str,
) -> Result<&'a lloom_tensor::quantized::gguf_file::TensorInfo> {
    ct.tensor_infos
        .get(name)
        .ok_or_else(|| lloom_tensor::Error::msg(format!("cannot find tensor info for {name}")))
}
fn named(ct: &Content, names: &[String], shape: impl Into<Shape>) -> Result<Weight> {
    chosen(ct, Weight::first_present(ct, names)?, shape)
}
fn chosen(ct: &Content, name: String, shape: impl Into<Shape>) -> Result<Weight> {
    let dtype = info(ct, &name)?.ggml_dtype;
    Ok(Weight::new(name, shape)
        .family("lfm2")
        .dtype(WeightDType::Exact(dtype)))
}
fn aliases(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_owned()).collect()
}
impl Lfm2Weights {
    pub(crate) fn resolve(ct: &Content) -> Result<Self> {
        let md = |s: &str| {
            ct.metadata
                .get(s)
                .ok_or_else(|| lloom_tensor::Error::msg(format!("cannot find {s} in metadata")))
        };
        let heads = value_to_usize(md("lfm2.attention.head_count")?)?;
        let width = value_to_usize(md("lfm2.embedding_length")?)?;
        let context = value_to_usize(md("lfm2.context_length")?)?;
        let blocks = value_to_usize(md("lfm2.block_count")?)?;
        let epsilon = md("lfm2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_base = ct
            .metadata
            .get("lfm2.rope.freq_base")
            .map(|v| v.to_f32())
            .transpose()?
            .unwrap_or(1_000_000f32);
        let kernel = value_to_usize(md("lfm2.shortconv.l_cache")?)?;
        if heads == 0
            || width == 0
            || blocks == 0
            || context == 0
            || kernel == 0
            || width % heads != 0
            || !epsilon.is_finite()
            || epsilon <= 0.
            || !rope_base.is_finite()
            || rope_base <= 0.
        {
            bail!("invalid quantized lfm2 geometry or normalization epsilon")
        }
        if blocks > ct.tensor_infos.len() {
            bail!("quantized lfm2 layer count exceeds available tensor headers")
        }
        crate::nn::rotary_emb::validate_default_rope_tables(
            width / heads,
            rope_base,
            context,
            lloom_tensor::DType::F32,
        )?;
        let kv = read_usize_list(md("lfm2.attention.head_count_kv")?, blocks)?;
        if kv
            .iter()
            .any(|&kv| kv != 0 && (kv > heads || heads % kv != 0))
        {
            bail!("invalid quantized lfm2 KV head count")
        }
        let emb_aliases = aliases(&[
            "token_embd.weight",
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
        ]);
        let emb_name = Weight::first_present(ct, &emb_aliases)?;
        let (vocab, emb_width) = info(ct, &emb_name)?.shape.dims2()?;
        if vocab == 0 || emb_width != width {
            bail!("lfm2 GGUF embedding geometry mismatch")
        }
        let embedding = chosen(ct, emb_name, (vocab, width))?;
        let norm = named(
            ct,
            &aliases(&[
                "output_norm.weight",
                "embedding_norm.weight",
                "model.embedding_norm.weight",
                "model.embedding_norm",
                "token_embd_norm.weight",
            ]),
            width,
        )?;
        let output_aliases = aliases(&[
            "output.weight",
            "lm_head.weight",
            "model.output.weight",
            "model.lm_head.weight",
        ]);
        let output = if output_aliases
            .iter()
            .any(|name| ct.tensor_infos.contains_key(name))
        {
            Some(named(ct, &output_aliases, (vocab, width))?)
        } else {
            None
        };
        let mut layers = crate::nn::branch::retained_vec(blocks)?;
        for (index, &kv_heads) in kv.iter().enumerate() {
            let p = format!("blk.{index}");
            let operator_norm = named(
                ct,
                &[
                    format!("{p}.attn_norm.weight"),
                    format!("{p}.operator_norm.weight"),
                    format!("{p}.attention_norm.weight"),
                ],
                width,
            )?;
            let ffn_norm = named(
                ct,
                &[format!("{p}.ffn_norm.weight"), format!("{p}.ffn_norm")],
                width,
            )?;
            let gate_aliases = [
                format!("{p}.ffn_gate.weight"),
                format!("{p}.feed_forward.w1.weight"),
                format!("{p}.mlp.gate_proj.weight"),
            ];
            let gate_name = Weight::first_present(ct, &gate_aliases)?;
            let (ffn, input) = info(ct, &gate_name)?.shape.dims2()?;
            if ffn == 0 || input != width {
                bail!("lfm2 GGUF FFN geometry mismatch")
            }
            let gate = chosen(ct, gate_name, (ffn, width))?;
            let down = named(
                ct,
                &[
                    format!("{p}.ffn_down.weight"),
                    format!("{p}.feed_forward.w2.weight"),
                    format!("{p}.mlp.down_proj.weight"),
                ],
                (width, ffn),
            )?;
            let up = named(
                ct,
                &[
                    format!("{p}.ffn_up.weight"),
                    format!("{p}.feed_forward.w3.weight"),
                    format!("{p}.mlp.up_proj.weight"),
                ],
                (ffn, width),
            )?;
            let mixer = if kv_heads > 0 {
                let head_dim = width / heads;
                let kv_width = kv_heads
                    .checked_mul(head_dim)
                    .ok_or_else(|| lloom_tensor::Error::msg("lfm2 GGUF KV width overflow"))?;
                MixerWeights::Attention(AttentionWeights {
                    q: named(
                        ct,
                        &[
                            format!("{p}.attn_q.weight"),
                            format!("{p}.self_attn.q_proj.weight"),
                        ],
                        (width, width),
                    )?,
                    k: named(
                        ct,
                        &[
                            format!("{p}.attn_k.weight"),
                            format!("{p}.self_attn.k_proj.weight"),
                        ],
                        (kv_width, width),
                    )?,
                    v: named(
                        ct,
                        &[
                            format!("{p}.attn_v.weight"),
                            format!("{p}.self_attn.v_proj.weight"),
                        ],
                        (kv_width, width),
                    )?,
                    output: named(
                        ct,
                        &[
                            format!("{p}.attn_output.weight"),
                            format!("{p}.self_attn.out_proj.weight"),
                        ],
                        (width, width),
                    )?,
                    q_norm: named(
                        ct,
                        &[
                            format!("{p}.attn_q_norm.weight"),
                            format!("{p}.self_attn.q_layernorm.weight"),
                            format!("{p}.attention.q_norm.weight"),
                        ],
                        head_dim,
                    )?,
                    k_norm: named(
                        ct,
                        &[
                            format!("{p}.attn_k_norm.weight"),
                            format!("{p}.self_attn.k_layernorm.weight"),
                            format!("{p}.attention.k_norm.weight"),
                        ],
                        head_dim,
                    )?,
                    kv_heads,
                })
            } else {
                let triple = width.checked_mul(3).ok_or_else(|| {
                    lloom_tensor::Error::msg("lfm2 GGUF convolution width overflow")
                })?;
                let conv_aliases = [
                    format!("{p}.shortconv.conv.weight"),
                    format!("{p}.conv.conv.weight"),
                    format!("{p}.shortconv.conv"),
                ];
                let conv_name = Weight::first_present(ct, &conv_aliases)?;
                let shape = &info(ct, &conv_name)?.shape;
                if shape.dims() != [width, kernel]
                    && shape.dims() != [kernel, width]
                    && shape.dims() != [width, 1, kernel]
                {
                    bail!("lfm2 GGUF convolution geometry mismatch")
                }
                MixerWeights::Conv(ConvWeights {
                    input: named(
                        ct,
                        &[
                            format!("{p}.shortconv.in_proj.weight"),
                            format!("{p}.conv.in_proj.weight"),
                        ],
                        (triple, width),
                    )?,
                    output: named(
                        ct,
                        &[
                            format!("{p}.shortconv.out_proj.weight"),
                            format!("{p}.conv.out_proj.weight"),
                        ],
                        (width, width),
                    )?,
                    conv: chosen(ct, conv_name.clone(), shape.clone())?,
                })
            };
            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                gate,
                down,
                up,
                mixer,
            });
        }
        Ok(Self {
            heads,
            width,
            context,
            epsilon,
            rope_base,
            kernel,
            vocab,
            embedding,
            norm,
            output,
            layers,
        })
    }
}
impl WeightPlan for Lfm2Weights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.embedding.visit(visitor)?;
        self.norm.visit(visitor)?;
        if let Some(output) = &self.output {
            output.visit(visitor)?;
        }
        self.layers.as_slice().visit(visitor)
    }
}
impl WeightPlan for LayerWeights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        for w in [
            &self.operator_norm,
            &self.ffn_norm,
            &self.gate,
            &self.down,
            &self.up,
        ] {
            w.visit(visitor)?;
        }
        match &self.mixer {
            MixerWeights::Attention(a) => {
                for w in [&a.q, &a.k, &a.v, &a.output, &a.q_norm, &a.k_norm] {
                    w.visit(visitor)?;
                }
            }
            MixerWeights::Conv(c) => {
                for w in [&c.input, &c.output, &c.conv] {
                    w.visit(visitor)?;
                }
            }
        }
        Ok(())
    }
}
