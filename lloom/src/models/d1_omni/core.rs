//! The pinned `modeling_d1.py::_forward` composition. Media encoders supply a
//! shared prefix; request rendering/readout remain in the engine. This is not
//! an independent allocator/admission policy or a complete family loader.
use super::{DecisionHead, DecisionHeadBatch, TextConfig, Trunk, TrunkBatch};
use crate::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecisionCoreConfig {
    pub text_config: TextConfig,
    pub head_layers: usize,
    /// Outer model limit, distinct from text_config.max_position_embeddings.
    pub max_length: usize,
}

/// One question's already-rendered text and optional shared media embeddings.
/// Marker positions index text only, never the media prefix.
pub struct DecisionRow<'a> {
    pub tokens: &'a [u32],
    pub markers: &'a [usize],
    pub question_type: u32,
    pub prefix: Option<&'a Tensor>,
}

/// Numeric backing is owned until the request releases it. The final engine
/// owner must account it and asynchronous retirement in its whole-pass promise.
#[derive(Debug)]
pub struct PreparedRows {
    owner: Arc<()>,
    hidden: Tensor,
    trunk: TrunkBatch,
    head: DecisionHeadBatch,
    text_lengths: Vec<usize>,
    prefix_lengths: Vec<usize>,
    option_counts: Vec<usize>,
    max_text: usize,
    input_tokens: usize,
}

impl PreparedRows {
    pub fn option_counts(&self) -> &[usize] {
        &self.option_counts
    }
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (
            self.text_lengths.len(),
            self.trunk.dimensions().1,
            self.max_text,
        )
    }
}

/// Loaded trunk/head composition. Vision/audio and safetensors/GGUF source
/// selection join this through the complete family owner, not a second model.
#[derive(Debug, Clone)]
pub struct DecisionCore {
    owner: Arc<()>,
    cfg: DecisionCoreConfig,
    trunk: Trunk,
    head: DecisionHead,
    cos: Tensor,
    sin: Tensor,
    dtype: DType,
    device: Device,
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni decision rows shape/byte overflow".into()))
}

impl DecisionCore {
    pub(crate) fn validate_config(cfg: &DecisionCoreConfig, dtype: DType) -> Result<()> {
        cfg.text_config.validated_intermediate()?;
        if cfg.max_length == 0 || cfg.max_length > cfg.text_config.max_position_embeddings {
            lloom_tensor::bail!("omni outer maximum must fit positive text position limit")
        }
        Trunk::validate_config(&cfg.text_config, dtype)?;
        DecisionHead::validate_config(cfg.text_config.hidden_size, cfg.head_layers, dtype)?;
        crate::nn::rotary_emb::validate_default_rope_tables(
            cfg.text_config.hidden_size / cfg.text_config.num_attention_heads,
            cfg.text_config.rope_theta as f32,
            cfg.max_length,
            dtype,
        )?;
        Ok(())
    }

    /// `vb` is rooted at the omni checkpoint: namespaces `encoder` and `head`.
    /// Caller supplies the admitted origin device/domain. Tables are retained
    /// once, at the OUTER maximum length, not text_config's larger position cap.
    pub fn load(cfg: DecisionCoreConfig, vb: VarBuilder) -> Result<Self> {
        Self::validate_config(&cfg, vb.dtype())?;
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let trunk = Trunk::load(cfg.text_config.clone(), vb.pp("encoder"))?;
        let head = DecisionHead::load(cfg.text_config.hidden_size, cfg.head_layers, vb.pp("head"))?;
        let (cos, sin) = trunk.rope_tables(cfg.max_length)?;
        Ok(Self {
            owner: Arc::new(()),
            cfg,
            trunk,
            head,
            cos,
            sin,
            dtype,
            device,
        })
    }

    /// Validate host metadata and all prefix geometries before creating any
    /// Tensor. Prefix values are not read back for validation. Input usage counts
    /// the repeated prefix per question, matching the reference's batched API.
    pub fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        if rows.is_empty() {
            lloom_tensor::bail!("omni decision batch is empty")
        }
        let width = self.cfg.text_config.hidden_size;
        let mut lengths = Vec::with_capacity(rows.len());
        let mut prefixes = Vec::with_capacity(rows.len());
        let mut totals = Vec::with_capacity(rows.len());
        let mut input_tokens = 0usize;
        for row in rows {
            let n = row.tokens.len();
            if n == 0
                || row.question_type >= 3
                || row.markers.is_empty()
                || row.markers.iter().any(|&p| p >= n)
                || row
                    .tokens
                    .iter()
                    .any(|&t| t as usize >= self.cfg.text_config.vocab_size)
            {
                lloom_tensor::bail!(
                    "omni decision row has invalid tokens, markers or question type"
                )
            }
            let p = match row.prefix {
                None => 0,
                Some(prefix) => {
                    let (b, p, d) = prefix.dims3()?;
                    if b != 1
                        || d != width
                        || !prefix.device().same_device(&self.device)
                        || !matches!(
                            prefix.dtype(),
                            DType::F16 | DType::BF16 | DType::F32 | DType::F64
                        )
                    {
                        lloom_tensor::bail!(
                            "omni prefix must be floating [1,P,D] on the core device"
                        )
                    }
                    p
                }
            };
            let total = p.checked_add(n).ok_or_else(|| {
                lloom_tensor::Error::Msg("omni prefix/text length overflow".into())
            })?;
            if total > self.cfg.max_length {
                lloom_tensor::bail!("omni decision row exceeds the outer model maximum")
            }
            input_tokens = input_tokens.checked_add(total).ok_or_else(|| {
                lloom_tensor::Error::Msg("omni decision input-token count overflow".into())
            })?;
            lengths.push(n);
            prefixes.push(p);
            totals.push(total);
        }
        let longest = *totals.iter().max().expect("nonempty rows");
        let max_text = *lengths.iter().max().expect("nonempty rows");
        checked_product(&[rows.len(), longest, width, self.dtype.size_in_bytes()])?;
        checked_product(&[rows.len(), max_text, width, self.dtype.size_in_bytes()])?;
        // Head flattened gather uses U32 row indices. Validate before embeddings,
        // not only later when the head constructs its prepared metadata.
        if checked_product(&[rows.len(), max_text])? > u32::MAX as usize {
            lloom_tensor::bail!("omni decision text rows exceed U32 marker indexing")
        }
        let max_options = rows.iter().map(|r| r.markers.len()).max().unwrap();
        checked_product(&[rows.len(), max_options, std::mem::size_of::<u32>()])?;
        let mut embedded = Vec::with_capacity(rows.len());
        for (row, &total) in rows.iter().zip(&totals) {
            let text = self.trunk.embed(row.tokens, 1, row.tokens.len())?;
            let h = match row.prefix {
                Some(prefix) if prefix.dim(1)? != 0 => {
                    Tensor::cat(&[prefix.to_dtype(self.dtype)?, text], 1)?
                }
                _ => text,
            };
            embedded.push(h.pad_with_zeros(1, 0, longest - total)?);
        }
        let hidden = Tensor::cat(&embedded, 0)?;
        let trunk = self.trunk.prepare_batch(longest, &totals, &prefixes)?;
        let markers: Vec<&[usize]> = rows.iter().map(|r| r.markers).collect();
        let qtypes: Vec<u32> = rows.iter().map(|r| r.question_type).collect();
        let head = self
            .head
            .prepare_batch(max_text, &lengths, &markers, &qtypes)?;
        Ok(PreparedRows {
            owner: Arc::clone(&self.owner),
            hidden,
            trunk,
            head,
            text_lengths: lengths,
            prefix_lengths: prefixes,
            option_counts: rows.iter().map(|r| r.markers.len()).collect(),
            max_text,
            input_tokens,
        })
    }

    fn text_rows(&self, hidden: &Tensor, rows: &PreparedRows) -> Result<Tensor> {
        let mut text = Vec::with_capacity(rows.text_lengths.len());
        for (row, (&prefix, &length)) in rows
            .prefix_lengths
            .iter()
            .zip(&rows.text_lengths)
            .enumerate()
        {
            text.push(
                hidden
                    .narrow(0, row, 1)?
                    .narrow(1, prefix, length)?
                    .pad_with_zeros(1, 0, rows.max_text - length)?,
            );
        }
        // Computed trunk padding is deliberately discarded. Reusing it here
        // would differ from Python pad_sequence's new ZERO text padding.
        Tensor::cat(&text, 0)
    }

    /// Full trunk → real-text extraction/zero-padding → decision head. Returns
    /// F32 [B,max_options]; use option_counts to discard missing marker slots.
    /// Both policies are explicit and monomorphized; neither selects a hidden
    /// dense fallback, changes causality, or materializes an L² mask here.
    pub fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk_attend: &mut T,
        head_attend: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        if !Arc::ptr_eq(&self.owner, &rows.owner) {
            lloom_tensor::bail!("omni prepared rows belong to another loaded core")
        }
        let length = rows.trunk.dimensions().1;
        let hidden = self.trunk.forward_with_attention(
            &rows.hidden,
            &rows.trunk,
            &self.cos.narrow(0, 0, length)?,
            &self.sin.narrow(0, 0, length)?,
            trunk_attend,
        )?;
        self.head
            .forward_with_attention(&self.text_rows(&hidden, rows)?, &rows.head, head_attend)
    }
}

#[cfg(test)]
mod tests;
