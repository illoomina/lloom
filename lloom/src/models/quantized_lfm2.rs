use crate::nn::{Conv1d, Conv1dConfig, Embedding, Module};
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{bail, DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn get_qtensor<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<lloom_tensor::quantized::QTensor> {
    for name in names {
        if ct.tensor_infos.contains_key(name) {
            // An existing malformed/unreadable alias is not a missing alias.
            return ct.tensor(reader, name, device);
        }
    }
    bail!("cannot find tensor info for {}", names.join(" | "))
}

fn get_dequantized<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<Tensor> {
    get_qtensor(ct, reader, device, names)?.dequantize(device)
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(xs)?;
        let w3 = self.w3.forward(xs)?;
        self.w2.forward(&(crate::nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
}

#[derive(Debug, Clone)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    conv: Tensor,
    l_cache: usize,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    crate::nn::rotary_emb::default_rope_tables(
        head_dim,
        freq_base,
        context_length,
        DType::F32,
        device,
    )
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b, _n, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        crate::nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = crate::nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&y)
    }
}

impl ShortConvLayer {
    fn forward(
        &self,
        xs: &Tensor,
        cache: &mut Option<Tensor>,
        continuation: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, hidden)?;
        let c = bcx.narrow(1, hidden, hidden)?;
        let x = bcx.narrow(1, 2 * hidden, hidden)?;
        let bx = (b * &x)?.contiguous()?;

        // conv_weight shape -> [hidden, l_cache]
        let mut conv_weight = self.conv.clone();
        if conv_weight.dims().len() == 3 {
            conv_weight = conv_weight.squeeze(1)?;
        } else if conv_weight.dims().len() == 2 && conv_weight.dims2()? == (self.l_cache, hidden) {
            conv_weight = conv_weight.t()?.contiguous()?;
        }
        let conv_weight = conv_weight.contiguous()?;

        let mut conv_out = if continuation && seq_len > 1 {
            // The common request loop uses one private sequence. Reuse the
            // carried-window block rather than reinitializing each prefill tile.
            if b_sz != 1 {
                bail!("quantized lfm2 continuation currently requires batch one")
            }
            let history = cache.as_ref().map(|t| t.squeeze(0)).transpose()?;
            let (out, next) = crate::nn::short_conv::shift_prefill_reference(
                &bx.squeeze(0)?.t()?,
                &conv_weight,
                history.as_ref(),
                hidden,
                self.l_cache,
            )?;
            *cache = Some(next.unsqueeze(0)?);
            out.t()?.unsqueeze(0)?
        } else if seq_len == 1 {
            let mut state = if let Some(cache) = &*cache {
                cache.clone()
            } else {
                Tensor::zeros((b_sz, hidden, self.l_cache), bx.dtype(), bx.device())?
            };

            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }
            *cache = Some(state.clone());

            (state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                conv_weight
                    .reshape((hidden, 1, self.l_cache))?
                    .contiguous()?,
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: hidden,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx.contiguous()?)?;
            out = out.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let (_, _, cur_len) = bx.dims3()?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache_src = bx.narrow(2, start, cur_len - start)?;
                if cache_src.dims3()?.2 < self.l_cache {
                    let pad = self.l_cache - cache_src.dims3()?.2;
                    let zeros =
                        Tensor::zeros((b_sz, hidden, pad), cache_src.dtype(), cache_src.device())?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                *cache = Some(cache_src);
            }

            out
        };

        conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

pub struct Model {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    config: Config,
    owner: Arc<()>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    // Dimensions are unsigned GGUF integers, not lossy signed/float casts.
    if !matches!(
        v,
        gguf_file::Value::U8(_)
            | gguf_file::Value::U16(_)
            | gguf_file::Value::U32(_)
            | gguf_file::Value::U64(_)
    ) {
        bail!("quantized lfm2 dimension must be an unsigned integer")
    }
    usize::try_from(v.to_u64()?).map_err(lloom_tensor::Error::wrap)
}

fn read_usize_list(v: &gguf_file::Value, len: usize) -> Result<Vec<usize>> {
    use gguf_file::Value::Array;
    match v {
        Array(arr) => {
            let mut out = crate::nn::branch::retained_vec(arr.len())?;
            for item in arr {
                out.push(value_to_usize(item)?);
            }
            if out.len() == len {
                Ok(out)
            } else if out.len() == 1 {
                Ok(vec![out[0]; len])
            } else {
                bail!(
                    "unexpected array length in metadata, expected {len} got {}",
                    out.len()
                )
            }
        }
        _ => Ok(vec![value_to_usize(v)?; len]),
    }
}

impl Model {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("lfm2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv_meta = md_get("lfm2.attention.head_count_kv")?;
        let embedding_length = md_get("lfm2.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("lfm2.context_length")?.to_u32()? as usize;
        let block_count = md_get("lfm2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("lfm2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("lfm2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(1_000_000f32);
        let l_cache = md_get("lfm2.shortconv.l_cache")?.to_u32()? as usize;

        if head_count == 0
            || embedding_length == 0
            || block_count == 0
            || context_length == 0
            || l_cache == 0
            || embedding_length % head_count != 0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
        {
            bail!("invalid quantized lfm2 geometry or normalization epsilon")
        }
        if block_count > ct.tensor_infos.len() {
            bail!("quantized lfm2 layer count exceeds available tensor headers")
        }
        let head_count_kv = read_usize_list(head_count_kv_meta, block_count)?;
        if head_count_kv
            .iter()
            .any(|&kv| kv != 0 && (kv > head_count || head_count % kv != 0))
        {
            bail!("invalid quantized lfm2 KV head count")
        }
        let head_dim = embedding_length / head_count;
        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = get_qtensor(
            &ct,
            reader,
            device,
            &[
                "token_embd.weight",
                "tok_embeddings.weight",
                "model.embed_tokens.weight",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        )?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        tracing::debug!(
            tok_embd_shape = ?tok_embeddings.shape().dims(),
            "loaded lfm2 token embeddings"
        );

        let norm = RmsNorm::from_qtensor(
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "output_norm.weight",
                    "embedding_norm.weight",
                    "model.embedding_norm.weight",
                    "model.embedding_norm",
                    "token_embd_norm.weight",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            )?,
            rms_norm_eps,
        )?;
        let output_names = [
            "output.weight",
            "lm_head.weight",
            "model.output.weight",
            "model.lm_head.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let output_q = if output_names
            .iter()
            .any(|name| ct.tensor_infos.contains_key(name))
        {
            get_qtensor(&ct, reader, device, &output_names)?
        } else {
            tok_embeddings_q
        };
        tracing::debug!(
            output_shape = ?output_q.shape().dims(),
            "loaded lfm2 output weight (using tok_embd if missing)"
        );

        let mut layers = crate::nn::branch::retained_vec(block_count)?;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let is_attention = head_count_kv.get(layer_idx).copied().unwrap_or(head_count) > 0;

            let operator_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    format!("{prefix}.attn_norm.weight"),
                    format!("{prefix}.operator_norm.weight"),
                    format!("{prefix}.attention_norm.weight"),
                ],
            )?;
            let ffn_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    format!("{prefix}.ffn_norm.weight"),
                    format!("{prefix}.ffn_norm"),
                ],
            )?;
            let mlp = {
                let w1 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_gate.weight"),
                        format!("{prefix}.feed_forward.w1.weight"),
                        format!("{prefix}.mlp.gate_proj.weight"),
                    ],
                )?;
                let w2 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_down.weight"),
                        format!("{prefix}.feed_forward.w2.weight"),
                        format!("{prefix}.mlp.down_proj.weight"),
                    ],
                )?;
                let w3 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_up.weight"),
                        format!("{prefix}.feed_forward.w3.weight"),
                        format!("{prefix}.mlp.up_proj.weight"),
                    ],
                )?;
                Mlp {
                    w1: QMatMul::from_qtensor(w1)?,
                    w2: QMatMul::from_qtensor(w2)?,
                    w3: QMatMul::from_qtensor(w3)?,
                }
            };

            let kind = if is_attention {
                let n_kv_head = head_count_kv[layer_idx];
                let wq = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_q.weight"),
                        format!("{prefix}.self_attn.q_proj.weight"),
                    ],
                )?;
                let wk = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_k.weight"),
                        format!("{prefix}.self_attn.k_proj.weight"),
                    ],
                )?;
                let wv = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_v.weight"),
                        format!("{prefix}.self_attn.v_proj.weight"),
                    ],
                )?;
                let wo = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_output.weight"),
                        format!("{prefix}.self_attn.out_proj.weight"),
                    ],
                )?;
                let q_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_q_norm.weight"),
                        format!("{prefix}.self_attn.q_layernorm.weight"),
                        format!("{prefix}.attention.q_norm.weight"),
                    ],
                )?;
                let k_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_k_norm.weight"),
                        format!("{prefix}.self_attn.k_layernorm.weight"),
                        format!("{prefix}.attention.k_norm.weight"),
                    ],
                )?;

                LayerKind::Attention(AttentionLayer {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                    wo: QMatMul::from_qtensor(wo)?,
                    q_norm: RmsNorm::from_qtensor(q_norm, rms_norm_eps)?,
                    k_norm: RmsNorm::from_qtensor(k_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                })
            } else {
                let in_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.in_proj.weight"),
                        format!("{prefix}.conv.in_proj.weight"),
                    ],
                )?;
                let out_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.out_proj.weight"),
                        format!("{prefix}.conv.out_proj.weight"),
                    ],
                )?;
                let conv = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.conv.weight"),
                        format!("{prefix}.conv.conv.weight"),
                        format!("{prefix}.shortconv.conv"),
                    ],
                )?;
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: QMatMul::from_qtensor(in_proj)?,
                    out_proj: QMatMul::from_qtensor(out_proj)?,
                    conv,
                    l_cache,
                })
            };

            layers.push(LayerWeights {
                operator_norm: RmsNorm::from_qtensor(operator_norm, rms_norm_eps)?,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                mlp,
                kind,
                span_mlp: tracing::span!(tracing::Level::TRACE, "ffn"),
            });
        }

        let vocab = tok_embeddings.dim(0)?;
        Ok(Self {
            config: Config {
                context_length,
                hidden_size: embedding_length,
                vocab_size: vocab,
            },
            owner: Arc::new(()),
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output_q)?,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn new_cache(&self) -> Result<Cache> {
        self.cache_with_continuation(true)
    }

    fn cache_with_continuation(&self, continuation: bool) -> Result<Cache> {
        let mut layers = crate::nn::branch::retained_vec(self.layers.len())?;
        for layer in &self.layers {
            layers.push(match layer.kind {
                LayerKind::Attention(_) => LayerState::Attention(None),
                LayerKind::ShortConv(_) => LayerState::ShortConv(None),
            });
        }
        Ok(Cache {
            layers,
            masks: HashMap::new(),
            owner: self.owner.clone(),
            epoch: Arc::new(()),
            pos: 0,
            continuation,
            failed: false,
        })
    }

    fn forward_hidden(
        &self,
        mut hidden: Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (batch, seq_len, width) = hidden.dims3()?;
        let end = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| lloom_tensor::Error::msg("quantized lfm2 position overflow"))?;
        if seq_len == 0
            || batch == 0
            || (cache.continuation && batch != 1)
            || width != self.config.hidden_size
            || end > self.config.context_length
            || !Arc::ptr_eq(&cache.owner, &self.owner)
            || cache.layers.len() != self.layers.len()
            || (cache.continuation && (cache.failed || index_pos != cache.pos))
        {
            bail!("quantized lfm2 invalid sequence geometry, owner or cache position")
        }
        let mask = if seq_len == 1 {
            None
        } else {
            let key = (seq_len, end);
            if !cache.masks.contains_key(&key) {
                cache.masks.insert(
                    key,
                    crate::utils::build_causal_mask(seq_len, index_pos, hidden.device())?,
                );
            }
            Some(cache.masks[&key].clone())
        };
        cache.failed = true;
        let _enter = self.span.enter();
        for (layer, state) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match (&layer.kind, state) {
                (LayerKind::Attention(attn), LayerState::Attention(kv)) => {
                    attn.forward(&normed, mask.as_ref(), index_pos, kv)?
                }
                (LayerKind::ShortConv(conv), LayerState::ShortConv(history)) => {
                    conv.forward(&normed, history, cache.continuation)?
                }
                _ => bail!("quantized lfm2 cache layer kind mismatch"),
            };
            hidden = (hidden + residual)?;
            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let _enter = layer.span_mlp.enter();
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        let logits = self.output.forward(&hidden)?;
        cache.pos = end;
        cache.failed = false;
        Ok(logits)
    }
}

/// Runtime limits resolved from the actual GGUF header; no inverse FFN guess.
#[derive(Clone, Debug)]
pub struct Config {
    pub context_length: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

enum LayerState {
    Attention(Option<(Tensor, Tensor)>),
    ShortConv(Option<Tensor>),
}

/// Sequence state belongs to one loaded immutable model. No cache aliases escape.
pub struct Cache {
    layers: Vec<LayerState>,
    masks: HashMap<(usize, usize), Tensor>,
    owner: Arc<()>,
    epoch: Arc<()>,
    pos: usize,
    continuation: bool,
    failed: bool,
}

/// Incumbent stateful API, preserved over the same weights and forward body.
pub struct ModelWeights {
    model: Model,
    cache: Cache,
}
impl ModelWeights {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let model = Model::from_gguf(ct, reader, device)?;
        let cache = model.cache_with_continuation(false)?;
        Ok(Self { model, cache })
    }
    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.model.tok_embeddings.forward(tokens)?;
        self.model
            .forward_hidden(hidden, index_pos, &mut self.cache)
    }
}

impl crate::nn::Model for Model {
    type Config = Config;
    fn from_weights(_: crate::nn::VarBuilder, _: &Config) -> Result<Self> {
        bail!("quantized lfm2 requires its GGUF loader")
    }
}
impl crate::nn::CausalLM for Model {
    type Cache = Cache;
    fn new_cache(&self, cfg: &Config, device: &Device) -> Result<Cache> {
        if cfg.context_length != self.config.context_length
            || cfg.hidden_size != self.config.hidden_size
            || cfg.vocab_size != self.config.vocab_size
            || !device.same_device(self.tok_embeddings.embeddings().device())
        {
            bail!("quantized lfm2 cache config/device differs from loaded model")
        }
        let mut cache = Model::new_cache(self)?;
        cache.continuation = true;
        Ok(cache)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(tokens)
    }
    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        self.forward_hidden(embeds.clone(), pos, cache)
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
impl crate::nn::CacheReset for Cache {
    fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention(kv) => *kv = None,
                LayerState::ShortConv(history) => *history = None,
            }
        }
        self.pos = 0;
        self.failed = false;
        self.epoch = Arc::new(());
        self.masks.clear();
    }
}

/// KV handles refer to immutable Cat operands; recurrent state is an owned copy.
/// This is request-scoped, not a shallow cache clone or a model copy.
pub struct Retained {
    layers: Vec<LayerState>,
    owner: Arc<()>,
    epoch: Arc<()>,
    pos: usize,
}
impl crate::nn::branch::RetainState for Cache {
    type Retained = Retained;
    fn prepare_branching(&mut self) -> Result<()> {
        if self.pos != 0 || self.failed {
            bail!("quantized lfm2 branching needs a fresh cache")
        }
        self.continuation = true;
        Ok(())
    }
    fn retain_state(&mut self) -> Result<Retained> {
        if self.failed {
            bail!("quantized lfm2 failed cache cannot be retained")
        }
        let mut layers = crate::nn::branch::retained_vec(self.layers.len())?;
        for layer in &self.layers {
            layers.push(match layer {
                LayerState::Attention(kv) => LayerState::Attention(kv.clone()),
                LayerState::ShortConv(history) => LayerState::ShortConv(
                    history
                        .as_ref()
                        .map(crate::nn::recurrent::own_copy)
                        .transpose()?,
                ),
            });
        }
        Ok(Retained {
            layers,
            owner: self.owner.clone(),
            epoch: self.epoch.clone(),
            pos: self.pos,
        })
    }
    fn restore_retained(&mut self, saved: &Retained) -> Result<()> {
        if self.failed
            || !Arc::ptr_eq(&self.owner, &saved.owner)
            || !Arc::ptr_eq(&self.epoch, &saved.epoch)
            || saved.pos > self.pos
            || saved.layers.len() != self.layers.len()
        {
            bail!("quantized lfm2 foreign, reset or failed retained cache")
        }
        // All writes replace their handles; none mutates retained tensors.
        for (target, source) in self.layers.iter_mut().zip(&saved.layers) {
            *target = match source {
                LayerState::Attention(kv) => LayerState::Attention(kv.clone()),
                LayerState::ShortConv(history) => LayerState::ShortConv(history.clone()),
            };
        }
        self.pos = saved.pos;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
