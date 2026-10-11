//! Owned aggregate over existing omni components. The engine factory and
//! shared Domain/load/pass integration are not implemented by this assembler.
use super::{
    AudioConfig, AudioEncoder, DecisionCore, DecisionCoreConfig, DecisionRow, FrontendScratch,
    MelFeatures, MelFrontend, PreparedRows, RelativeInputs, TrunkBatch, Vision, VisionImagePlan,
};
use crate::models::lfm2_vl::VisionConfig;
use crate::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};
use std::collections::BTreeMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct OmniConfig {
    pub model_type: String,
    #[serde(flatten)]
    pub core: DecisionCoreConfig,
    pub vision_config: VisionConfig,
    pub audio_config: AudioConfig,
    pub projector_hidden_size: usize,
    pub image_text_length: usize,
    pub audio_text_length: usize,
    #[serde(default)]
    pub temperatures: BTreeMap<String, f64>,
}
impl OmniConfig {
    fn validate_outer(&self) -> Result<()> {
        if self.model_type != "d1_omni"
            || self.image_text_length == 0
            || self.audio_text_length == 0
            || self.projector_hidden_size == 0
            || self
                .temperatures
                .values()
                .any(|t| !t.is_finite() || *t <= 0.)
        {
            lloom_tensor::bail!("invalid omni family, text limits, projector width or temperatures")
        }
        Ok(())
    }
    /// Pinned request allowance min(configured text limit, outer max-prefix).
    /// Renderer applies the media-specific limit; it never substitutes the
    /// text encoder's larger128000 position limit for outer16384.
    pub fn text_allowance(&self, prefix_tokens: usize, configured_limit: usize) -> Result<usize> {
        let left = self
            .core
            .max_length
            .checked_sub(prefix_tokens)
            .ok_or_else(|| lloom_tensor::Error::Msg("omni media exceed outer context".into()))?;
        let limit = left.min(configured_limit);
        if limit < 64 {
            lloom_tensor::bail!("omni media leave fewer than64 text positions")
        }
        Ok(limit)
    }
}

/// Borrowed decoded RGB8. Decoder/resampler selection belongs to the front end;
/// the borrowed allocation keeps its originating owner's charge.
pub struct OmniImage<'a> {
    pub rgb: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// One loaded non-generating family: core, vision and audio share one checkpoint.
/// Component revision guards are NOT domain identities or allocation promises.
/// The public engine factory is intentionally absent until actual shared
/// Domain and load/pass-context wiring can cover host and device producers.
#[derive(Debug)]
pub struct OmniModel {
    cfg: OmniConfig,
    core: DecisionCore,
    vision: Vision,
    audio: AudioEncoder,
    frontend: MelFrontend,
    dtype: DType,
    device: Device,
}
impl OmniModel {
    /// Intermediate assembler only. The future factory must create and retain
    /// the model Domain and establish its shared load scope BEFORE this call.
    /// Host frontend tables need real tickets too; vb.device is insufficient.
    /// No per-component reload, separate quota or new checkpoint source exists.
    pub(crate) fn load_components(cfg: OmniConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate_outer()?;
        DecisionCore::validate_config(&cfg.core, vb.dtype())?;
        Vision::validate_config(
            &cfg.vision_config,
            cfg.projector_hidden_size,
            cfg.core.text_config.hidden_size,
            vb.dtype(),
        )?;
        cfg.audio_config
            .validate(cfg.core.text_config.hidden_size, vb.dtype())?;
        MelFrontend::table_bytes(cfg.audio_config.feat_in)?;
        let core = DecisionCore::load(cfg.core.clone(), vb.clone())?;
        let width = cfg.core.text_config.hidden_size;
        let vision = Vision::load(
            &cfg.vision_config,
            cfg.projector_hidden_size,
            width,
            vb.pp("vision"),
        )?;
        let audio = AudioEncoder::load(cfg.audio_config.clone(), width, vb.pp("audio"))?;
        let frontend = MelFrontend::new(cfg.audio_config.feat_in)?;
        Ok(Self {
            cfg,
            core,
            vision,
            audio,
            frontend,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }
    pub fn config(&self) -> &OmniConfig {
        &self.cfg
    }
    pub fn audio_frontend_scratch(&self, input_samples: usize) -> Result<FrontendScratch> {
        self.frontend.scratch(input_samples)
    }
    /// One media prefix is produced once and can be borrowed by many question
    /// rows. An enclosing shared pass must cover frontend+transfer+cast+encoder,
    /// all asynchronous retirement, and opaque backend workspaces together.
    pub fn audio_prefix_f32<F>(&self, samples: &[f32], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.audio_allowance(samples.len())?;
        self.audio_prefix(self.frontend.prepare_f32(samples)?, attend)
    }
    pub fn audio_prefix_i16<F>(&self, samples: &[i16], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.audio_allowance(samples.len())?;
        self.audio_prefix(self.frontend.prepare_i16(samples)?, attend)
    }
    fn audio_allowance(&self, samples: usize) -> Result<usize> {
        let n = self.frontend.scratch(samples)?.valid_frames;
        self.cfg
            .text_allowance(n.div_ceil(8), self.cfg.audio_text_length)
    }
    fn audio_prefix<F>(&self, mel: MelFeatures, attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let n = mel.valid_frames;
        let rows = n.div_ceil(8);
        self.cfg.text_allowance(rows, self.cfg.audio_text_length)?;
        // The frontend is always host F32. Transfer occurs as F32, then cast on
        // the model device, matching the reference stage boundary explicitly.
        let mel = Tensor::from_vec(
            mel.values,
            (1, mel.features, mel.computed_frames),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        self.audio.forward_with_attention(&mel, n, attend)
    }
    pub fn image_prefix<F>(&self, images: &[OmniImage<'_>], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        if images.is_empty() {
            lloom_tensor::bail!("omni image prefix needs nonempty images")
        }
        // Validate all geometries/context allowance BEFORE pixel buffers,
        // preparation tensors or model execution. This does not admit bytes.
        let mut total = 0usize;
        for image in images {
            let plan = VisionImagePlan::new(image.height, image.width, image.rgb.len())?;
            total = total.checked_add(plan.prefix_tokens()).ok_or_else(|| {
                lloom_tensor::Error::Msg("omni image prefix count overflow".into())
            })?;
        }
        self.cfg.text_allowance(total, self.cfg.image_text_length)?;
        let prepared = images
            .iter()
            .map(|i| self.vision.prepare_rgb8(i.rgb, i.height, i.width))
            .collect::<Result<Vec<_>>>()?;
        self.vision
            .forward_with_attention(&prepared.iter().collect::<Vec<_>>(), attend)
    }
    pub fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        self.core.prepare_rows(rows)
    }
    pub fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk: &mut T,
        head: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.core.forward_with_attention(rows, trunk, head)
    }
}

#[cfg(test)]
mod tests;
