use crate::nn::{Conv1d, Conv1dConfig, Module};
use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{bail, DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

mod weights;

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
    kv_dtype: DType,
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
        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;
        let y = self.attend(&q, &k, &v, mask, index_pos, kv_cache)?;
        self.wo.forward(&y)
    }

    /// One sequence's attention over its projected `q`, `k`, `v`
    /// (`[b, seq, heads · head_dim]`) against its KV, before `wo`.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = q.dims3()?;
        let n_embd = self.n_head * self.head_dim;
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

        // Store only KV in the requested precision; attention arithmetic stays f32.
        let k = k.to_dtype(self.kv_dtype)?;
        let v = v.to_dtype(self.kv_dtype)?;
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

        let k = repeat_kv(k.to_dtype(q.dtype())?, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v.to_dtype(q.dtype())?, self.n_head / self.n_kv_head)?;

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

        y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])
    }
}

impl ShortConvLayer {
    fn forward(
        &self,
        xs: &Tensor,
        cache: &mut Option<Tensor>,
        continuation: bool,
    ) -> Result<Tensor> {
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let conv_out = self.convolve(&bcx, cache, continuation)?;
        self.out_proj.forward(&conv_out)
    }

    /// One sequence's gated short convolution over its projected `bcx`
    /// (`[b, 3 · hidden, seq]`) against its history, before `out_proj`;
    /// returns `[b, seq, hidden]`.
    fn convolve(
        &self,
        bcx: &Tensor,
        cache: &mut Option<Tensor>,
        continuation: bool,
    ) -> Result<Tensor> {
        let (b_sz, channels, seq_len) = bcx.dims3()?;
        let hidden = channels / 3;
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
        conv_out.transpose(1, 2)?.contiguous()
    }
}

pub struct Model {
    /// The token table as the file stores it: a block-quantized table stays
    /// packed and its rows are dequantized per lookup (the same values a
    /// dequantized copy holds, §15.1p); a tied head projects through it.
    tok_embeddings: QMatMul,
    device: Device,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    config: Config,
    owner: Arc<()>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    usize::try_from(v.to_count()?).map_err(lloom_tensor::Error::wrap)
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

/// Bytes the loaded model holds for its token table beyond the file's own:
/// the dense copy `QMatMul::from_arc` makes (f32 for an F32/F16/BF16 table
/// or under `dequantize_all`, f16 under `dequantize_all_f16`), none for a
/// block-quantized table kept packed.
pub fn dense_token_table_bytes(ct: &gguf_file::Content) -> Result<u64> {
    use lloom_tensor::quantized::GgmlDType;
    let plan = weights::Lfm2Weights::resolve(ct)?;
    let info = ct
        .tensor_infos
        .get(plan.embedding.name())
        .ok_or_else(|| lloom_tensor::Error::msg("lfm2 GGUF has no token table"))?;
    let qmm = &lloom_tensor::tuning::tuning().qmm;
    let per_element = match info.ggml_dtype {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => 4,
        _ if qmm.dequantize_all => 4,
        _ if qmm.dequantize_all_f16 => 2,
        _ => 0,
    };
    Ok((plan.vocab as u64) * (plan.width as u64) * per_element)
}

impl Model {
    /// Additive factory route over the caller's existing load promise. Header
    /// parsing and complete demand resolution occur inside that supplied scope.
    /// Host header/plan storage and transfers still need producer enrollment;
    /// this entry point does not enable the default served family yet.
    pub fn load_gguf_admitted(
        path: &std::path::Path,
        device: &Device,
        load: crate::nn::memory::LoadContext,
    ) -> Result<crate::nn::memory::Loaded<Self>> {
        crate::nn::memory::Loaded::new(device, load, |device, _context| {
            let mut file = std::fs::File::open(path)?;
            let content = gguf_file::Content::read(&mut file)?;
            Self::from_gguf(content, &mut file, device)
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_content(&ct, reader, device)
    }

    /// Borrow the already-read header so an app factory can resolve its text
    /// metadata after complete model preflight/materialization, without a
    /// second parse or a second weight schema. The incumbent owned entry stays.
    pub fn from_gguf_content<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        Self::preflight_gguf_content(ct, reader)?.materialize(reader, device)
    }

    /// Resolve complete demands and file extents once, without model tensors.
    /// An app factory can then validate tokenizer/policy before consuming this
    /// exact plan. Header/plan host storage still requires caller admission.
    /// The caller must materialize from the same unchanged backing content;
    /// this plan binds demands/header, not reader identity or a file snapshot.
    pub fn preflight_gguf_content<'a, R: std::io::Seek>(
        ct: &'a gguf_file::Content,
        reader: &mut R,
    ) -> Result<GgufPlan<'a>> {
        use crate::nn::gguf_weights::WeightPlan;
        let weights = weights::Lfm2Weights::resolve(ct)?;
        weights.validate_headers(ct)?;
        ct.validate_payloads(reader)?;
        Ok(GgufPlan {
            content: ct,
            weights,
        })
    }

    fn from_gguf_plan<R: std::io::Seek + std::io::Read>(
        plan: weights::Lfm2Weights,
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = plan.width / plan.heads;
        let (cos, sin) = precomput_freqs_cis(head_dim, plan.rope_base, plan.context, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let tok_embeddings = QMatMul::from_qtensor(plan.embedding.read(ct, reader, device)?)?;
        let norm = RmsNorm::from_qtensor(plan.norm.read(ct, reader, device)?, plan.epsilon)?;
        let output = match &plan.output {
            Some(output) => QMatMul::from_qtensor(output.read(ct, reader, device)?)?,
            None => tok_embeddings.clone(),
        };
        let mut layers = crate::nn::branch::retained_vec(plan.layers.len())?;
        for layer in &plan.layers {
            let operator_norm =
                RmsNorm::from_qtensor(layer.operator_norm.read(ct, reader, device)?, plan.epsilon)?;
            let ffn_norm =
                RmsNorm::from_qtensor(layer.ffn_norm.read(ct, reader, device)?, plan.epsilon)?;
            let mlp = Mlp {
                w1: QMatMul::from_qtensor(layer.gate.read(ct, reader, device)?)?,
                w2: QMatMul::from_qtensor(layer.down.read(ct, reader, device)?)?,
                w3: QMatMul::from_qtensor(layer.up.read(ct, reader, device)?)?,
            };
            let kind = match &layer.mixer {
                weights::MixerWeights::Attention(a) => LayerKind::Attention(AttentionLayer {
                    wq: QMatMul::from_qtensor(a.q.read(ct, reader, device)?)?,
                    wk: QMatMul::from_qtensor(a.k.read(ct, reader, device)?)?,
                    wv: QMatMul::from_qtensor(a.v.read(ct, reader, device)?)?,
                    wo: QMatMul::from_qtensor(a.output.read(ct, reader, device)?)?,
                    q_norm: RmsNorm::from_qtensor(
                        a.q_norm.read(ct, reader, device)?,
                        plan.epsilon,
                    )?,
                    k_norm: RmsNorm::from_qtensor(
                        a.k_norm.read(ct, reader, device)?,
                        plan.epsilon,
                    )?,
                    n_head: plan.heads,
                    n_kv_head: a.kv_heads,
                    head_dim,
                    kv_dtype: DType::F32,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                }),
                weights::MixerWeights::Conv(c) => LayerKind::ShortConv(ShortConvLayer {
                    in_proj: QMatMul::from_qtensor(c.input.read(ct, reader, device)?)?,
                    out_proj: QMatMul::from_qtensor(c.output.read(ct, reader, device)?)?,
                    conv: c.conv.read(ct, reader, device)?.dequantize(device)?,
                    l_cache: plan.kernel,
                }),
            };
            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                mlp,
                kind,
                span_mlp: tracing::span!(tracing::Level::TRACE, "ffn"),
            });
        }
        Ok(Self {
            config: Config {
                context_length: plan.context,
                hidden_size: plan.width,
                vocab_size: plan.vocab,
            },
            owner: Arc::new(()),
            tok_embeddings,
            device: device.clone(),
            layers,
            norm,
            output,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    /// Select the KV storage precision before creating any sequence. The default
    /// remains f32. f16 rounds at each append; projections, RoPE and attention
    /// still compute in f32, including CPU targets without native half matmul.
    pub fn with_kv_dtype(mut self, dtype: DType) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("quantized lfm2 KV storage requires f32 or f16")
        }
        // A cache created before this consuming configuration change must not
        // be accepted with a different store precision.
        self.owner = Arc::new(());
        for layer in &mut self.layers {
            if let LayerKind::Attention(attn) = &mut layer.kind {
                attn.kv_dtype = dtype;
            }
        }
        Ok(self)
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
        let mask = cache.causal_mask(seq_len, index_pos, hidden.device())?;
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

    /// [`BatchedCausalLM::forward_ragged`]: `hidden` is `[1, T, width]`, the
    /// rows' runs end to end. The norms, every projection, the MLP and the
    /// head run once over all `T` tokens (one weight sweep for every row);
    /// attention and the convolution run per row, over that row's tokens
    /// against its own cache, through the bodies [`Self::forward_hidden`]
    /// runs. Returns `[rows, vocab]`, each row's logits at its last token.
    fn forward_ragged_hidden(
        &self,
        mut hidden: Tensor,
        rows: &mut [crate::nn::RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let (batch, total, width) = hidden.dims3()?;
        let lens: usize = rows.iter().map(|r| r.len).sum();
        if batch != 1 || rows.is_empty() || lens != total || width != self.config.hidden_size {
            bail!("quantized lfm2 ragged: {total} tokens for rows {:?}", {
                rows.iter().map(|r| r.len).collect::<Vec<_>>()
            })
        }
        let mut masks = crate::nn::branch::retained_vec(rows.len())?;
        for row in rows.iter_mut() {
            let cache = &mut *row.cache;
            let end = row
                .pos
                .checked_add(row.len)
                .ok_or_else(|| lloom_tensor::Error::msg("quantized lfm2 position overflow"))?;
            if row.len == 0
                || end > self.config.context_length
                || !Arc::ptr_eq(&cache.owner, &self.owner)
                || cache.layers.len() != self.layers.len()
                || cache.failed
                || row.pos != cache.pos
            {
                bail!("quantized lfm2 ragged: invalid row geometry, owner or cache position")
            }
            masks.push(cache.causal_mask(row.len, row.pos, hidden.device())?);
        }
        for row in rows.iter_mut() {
            row.cache.failed = true;
        }
        let _enter = self.span.enter();
        for (l, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            let mut outs = crate::nn::branch::retained_vec(rows.len())?;
            let mut off = 0;
            hidden = match &layer.kind {
                LayerKind::Attention(attn) => {
                    let _enter = attn.span_attn.enter();
                    let q = attn.wq.forward(&normed)?;
                    let k = attn.wk.forward(&normed)?;
                    let v = attn.wv.forward(&normed)?;
                    for (row, mask) in rows.iter_mut().zip(&masks) {
                        let LayerState::Attention(kv) = &mut row.cache.layers[l] else {
                            bail!("quantized lfm2 cache layer kind mismatch")
                        };
                        outs.push(attn.attend(
                            &q.narrow(1, off, row.len)?,
                            &k.narrow(1, off, row.len)?,
                            &v.narrow(1, off, row.len)?,
                            mask.as_ref(),
                            row.pos,
                            kv,
                        )?);
                        off += row.len;
                    }
                    attn.wo.forward(&Tensor::cat(&outs, 1)?)?
                }
                LayerKind::ShortConv(conv) => {
                    let bcx = conv.in_proj.forward(&normed)?.transpose(1, 2)?;
                    for row in rows.iter_mut() {
                        let continuation = row.cache.continuation;
                        let LayerState::ShortConv(history) = &mut row.cache.layers[l] else {
                            bail!("quantized lfm2 cache layer kind mismatch")
                        };
                        outs.push(conv.convolve(
                            &bcx.narrow(2, off, row.len)?,
                            history,
                            continuation,
                        )?);
                        off += row.len;
                    }
                    conv.out_proj.forward(&Tensor::cat(&outs, 1)?)?
                }
            };
            hidden = (hidden + residual)?;
            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let _enter = layer.span_mlp.enter();
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let mut last = crate::nn::branch::retained_vec(rows.len())?;
        let mut off = 0u32;
        for row in rows.iter() {
            off += row.len as u32;
            last.push(off - 1);
        }
        let idx = Tensor::new(last.as_slice(), hidden.device())?;
        let hidden = hidden.squeeze(0)?.index_select(&idx, 0)?.contiguous()?;
        let _enter = self.span_output.enter();
        let logits = self.output.forward(&hidden)?;
        for row in rows.iter_mut() {
            row.cache.pos = row.pos + row.len;
            row.cache.failed = false;
        }
        Ok(logits)
    }
}

impl Cache {
    /// The additive causal mask of a pass of `seq_len` tokens at `index_pos`,
    /// kept per shape; `None` for one token.
    fn causal_mask(
        &mut self,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if seq_len == 1 {
            return Ok(None);
        }
        let key = (seq_len, index_pos + seq_len);
        if !self.masks.contains_key(&key) {
            self.masks.insert(
                key,
                crate::utils::build_causal_mask(seq_len, index_pos, device)?,
            );
        }
        Ok(Some(self.masks[&key].clone()))
    }
}

impl crate::nn::BatchedCausalLM for Model {
    /// A decode step for every row: a ragged pass of one token each.
    fn forward_rows(
        &self,
        tokens: &Tensor,
        rows: &mut [crate::nn::Row<'_, Cache>],
    ) -> Result<Tensor> {
        let ids = tokens.flatten_all()?.unsqueeze(0)?;
        let mut ragged = crate::nn::branch::retained_vec(rows.len())?;
        for row in rows.iter_mut() {
            ragged.push(crate::nn::RaggedRow {
                pos: row.pos,
                len: 1,
                cache: &mut *row.cache,
            });
        }
        self.forward_ragged(&ids, &mut ragged)
    }

    fn forward_ragged(
        &self,
        tokens: &Tensor,
        rows: &mut [crate::nn::RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let hidden = self.tok_embeddings.embedding(tokens)?;
        self.forward_ragged_hidden(hidden, rows)
    }
}

/// Resolved, validated demand plan over one immutably borrowed header.
/// Fields are private: a consumer cannot swap the schema/header after preflight.
/// Materialization consumes it once; no duplicate factory weight schema exists.
/// The caller supplies the same unchanged backing content during materialization;
/// reader identity and file contents are not snapshotted by this demand plan.
pub struct GgufPlan<'a> {
    content: &'a gguf_file::Content,
    weights: weights::Lfm2Weights,
}
impl GgufPlan<'_> {
    /// Resolve the requested runtime context before preflight and RoPE allocation.
    /// A caller may shorten, but not extend, the checkpoint's supported context.
    pub fn with_context_limit(mut self, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 || max_tokens > self.weights.context {
            bail!("quantized lfm2 requested context {max_tokens} must be in 1..={} (checkpoint context)", self.weights.context)
        }
        self.weights.context = max_tokens;
        Ok(self)
    }

    pub fn config(&self) -> Config {
        Config {
            context_length: self.weights.context,
            hidden_size: self.weights.width,
            vocab_size: self.weights.vocab,
        }
    }
    pub fn materialize<R: std::io::Read + std::io::Seek>(
        self,
        reader: &mut R,
        device: &Device,
    ) -> Result<Model> {
        lloom_tensor::memory_admission::begin_model_load();
        Model::from_gguf_plan(self.weights, self.content, reader, device)
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
        let hidden = self.model.tok_embeddings.embedding(tokens)?;
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
            || !device.same_device(&self.device)
        {
            bail!("quantized lfm2 cache config/device differs from loaded model")
        }
        let mut cache = Model::new_cache(self)?;
        cache.continuation = true;
        Ok(cache)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.embedding(tokens)
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
        self.take_layers(saved);
        Ok(())
    }
}

impl Cache {
    /// All writes replace their handles; none mutates retained tensors.
    fn take_layers(&mut self, saved: &Retained) {
        for (target, source) in self.layers.iter_mut().zip(&saved.layers) {
            *target = match source {
                LayerState::Attention(kv) => LayerState::Attention(kv.clone()),
                LayerState::ShortConv(history) => LayerState::ShortConv(history.clone()),
            };
        }
        self.pos = saved.pos;
    }
}

/// Another fresh cache of the same model continues a retained trunk: its
/// layers take the trunk's handles, which no later write mutates.
impl crate::nn::branch::AdoptRetained for Cache {
    fn adopt_retained(&mut self, saved: &Retained) -> Result<()> {
        if self.failed
            || self.pos != 0
            || !Arc::ptr_eq(&self.owner, &saved.owner)
            || saved.layers.len() != self.layers.len()
        {
            bail!("quantized lfm2: only a fresh cache of the same model adopts a retained trunk")
        }
        self.take_layers(saved);
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod weight_plan_tests;

// The generic generation driver may recover by resetting this sequence, but
// cannot roll back its recurrent channels. RetainState remains the separate
// decision-branch mechanism; declaring horizon zero does not remove it.
impl crate::nn::CacheRewind for Cache {
    fn rewind_horizon(&self) -> usize {
        0
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n != 0 {
            bail!("quantized lfm2 has no standing rewind window")
        }
        Ok(())
    }
}
