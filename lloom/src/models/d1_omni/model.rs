//! Owned aggregate over existing omni components. The engine factory and
//! shared Domain/load/pass integration are not implemented by this assembler.
use super::audio::QuantizedAudioEncoder;
use super::core::{QuantizedDecisionCore, ResolvedDecisionCoreConfig};
use super::vision::QuantizedVision;
use super::PreparedVisionImage;
use super::{
    AudioConfig, AudioEncoder, DecisionCore, DecisionCoreConfig, DecisionRow, FrontendScratch,
    MelFeatures, MelFrontend, PreparedRows, RelativeInputs, TrunkBatch, Vision, VisionImagePlan,
};
use crate::models::lfm2_vl::VisionConfig;
use crate::nn::VarBuilder;
use crate::quantized_var_builder::VarBuilder as QuantizedVarBuilder;
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
        if self.model_type != "d1_omni" {
            lloom_tensor::bail!("invalid omni family, text limits, projector width or temperatures")
        }
        validate_outer(
            self.image_text_length,
            self.audio_text_length,
            self.projector_hidden_size,
            &self.temperatures,
        )
    }
    /// Pinned request allowance min(configured text limit, outer max-prefix).
    /// Renderer applies the media-specific limit; it never substitutes the
    /// text encoder's larger128000 position limit for outer16384.
    pub fn text_allowance(&self, prefix_tokens: usize, configured_limit: usize) -> Result<usize> {
        text_allowance(self.core.max_length, prefix_tokens, configured_limit)
    }
}

fn validate_outer(
    image_text_length: usize,
    audio_text_length: usize,
    projector_hidden_size: usize,
    temperatures: &BTreeMap<String, f64>,
) -> Result<()> {
    if image_text_length == 0
        || audio_text_length == 0
        || projector_hidden_size == 0
        || temperatures.values().any(|t| !t.is_finite() || *t <= 0.)
    {
        lloom_tensor::bail!("invalid omni family, text limits, projector width or temperatures")
    }
    Ok(())
}
fn text_allowance(
    max_length: usize,
    prefix_tokens: usize,
    configured_limit: usize,
) -> Result<usize> {
    let left = max_length
        .checked_sub(prefix_tokens)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni media exceed outer context".into()))?;
    let limit = left.min(configured_limit);
    if limit < 64 {
        lloom_tensor::bail!("omni media leave fewer than64 text positions")
    }
    Ok(limit)
}
#[derive(Debug, Clone)]
struct RequestLimits {
    max_length: usize,
    image_text_length: usize,
    audio_text_length: usize,
}
impl RequestLimits {
    fn text_allowance(&self, prefix_tokens: usize, configured_limit: usize) -> Result<usize> {
        text_allowance(self.max_length, prefix_tokens, configured_limit)
    }
}

/// Header-resolved GGUF geometry. The factory supplies resolved FFN dimensions
/// and calibrated names without constructing an inverse Python configuration.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedOmniConfig {
    pub(crate) core: ResolvedDecisionCoreConfig,
    pub(crate) vision_config: VisionConfig,
    pub(crate) audio_config: AudioConfig,
    pub(crate) projector_hidden_size: usize,
    pub(crate) image_text_length: usize,
    pub(crate) audio_text_length: usize,
    pub(crate) temperatures: BTreeMap<String, f64>,
}
impl ResolvedOmniConfig {
    fn validate(&self) -> Result<()> {
        validate_outer(
            self.image_text_length,
            self.audio_text_length,
            self.projector_hidden_size,
            &self.temperatures,
        )?;
        self.core.validate()?;
        QuantizedVision::validate_config(
            &self.vision_config,
            self.projector_hidden_size,
            self.core.text.hidden_size,
        )?;
        QuantizedAudioEncoder::validate_config(&self.audio_config, self.core.text.hidden_size)?;
        MelFrontend::table_bytes(self.audio_config.feat_in)?;
        Ok(())
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
    inner: OmniInner<DecisionCore, Vision, AudioEncoder>,
}
#[derive(Debug)]
pub(crate) struct QuantizedOmniModel {
    cfg: ResolvedOmniConfig,
    inner: OmniInner<QuantizedDecisionCore, QuantizedVision, QuantizedAudioEncoder>,
}
#[derive(Debug)]
struct OmniInner<C, V, A> {
    limits: RequestLimits,
    core: C,
    vision: V,
    audio: A,
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
        let limits = RequestLimits {
            max_length: cfg.core.max_length,
            image_text_length: cfg.image_text_length,
            audio_text_length: cfg.audio_text_length,
        };
        Ok(Self {
            cfg,
            inner: OmniInner {
                limits,
                core,
                vision,
                audio,
                frontend,
                dtype: vb.dtype(),
                device: vb.device().clone(),
            },
        })
    }
    pub fn config(&self) -> &OmniConfig {
        &self.cfg
    }
    pub fn audio_frontend_scratch(&self, input_samples: usize) -> Result<FrontendScratch> {
        self.inner.audio_frontend_scratch(input_samples)
    }
    /// One media prefix is produced once and can be borrowed by many question
    /// rows. An enclosing shared pass must cover frontend+transfer+cast+encoder,
    /// all asynchronous retirement, and opaque backend workspaces together.
    pub fn audio_prefix_f32<F>(&self, samples: &[f32], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.inner.audio_prefix_f32(samples, attend)
    }
    pub fn audio_prefix_i16<F>(&self, samples: &[i16], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.inner.audio_prefix_i16(samples, attend)
    }
    pub fn image_prefix<F>(&self, images: &[OmniImage<'_>], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.image_prefix(images, attend)
    }
    pub fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        self.inner.prepare_rows(rows)
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
        self.inner.forward_with_attention(rows, trunk, head)
    }
}

impl QuantizedOmniModel {
    /// The main GGUF and mmproj map share one enclosing model/load promise.
    /// Device equality only rejects incompatible physical backends; it does
    /// not establish Domain identity or charge these eager maps automatically.
    pub(crate) fn load_components(
        cfg: ResolvedOmniConfig,
        core_vb: QuantizedVarBuilder,
        media_vb: QuantizedVarBuilder,
    ) -> Result<Self> {
        cfg.validate()?;
        if !core_vb.device().same_device(media_vb.device()) {
            lloom_tensor::bail!("omni core and media builders must use the same device")
        }
        let width = cfg.core.text.hidden_size;
        let core = QuantizedDecisionCore::load(cfg.core.clone(), core_vb.clone())?;
        let vision = QuantizedVision::load(
            &cfg.vision_config,
            cfg.projector_hidden_size,
            width,
            media_vb.clone(),
        )?;
        let audio = QuantizedAudioEncoder::load(cfg.audio_config.clone(), width, media_vb)?;
        let frontend = MelFrontend::new(cfg.audio_config.feat_in)?;
        let limits = RequestLimits {
            max_length: cfg.core.max_length,
            image_text_length: cfg.image_text_length,
            audio_text_length: cfg.audio_text_length,
        };
        Ok(Self {
            cfg,
            inner: OmniInner {
                limits,
                core,
                vision,
                audio,
                frontend,
                dtype: DType::F32,
                device: core_vb.device().clone(),
            },
        })
    }
    pub(crate) fn config(&self) -> &ResolvedOmniConfig {
        &self.cfg
    }
    pub(crate) fn audio_frontend_scratch(&self, input_samples: usize) -> Result<FrontendScratch> {
        self.inner.audio_frontend_scratch(input_samples)
    }
    /// One media prefix is produced once and can be borrowed by many question
    /// rows. An enclosing shared pass must cover frontend+transfer+cast+encoder,
    /// all asynchronous retirement, and opaque backend workspaces together.
    pub(crate) fn audio_prefix_f32<F>(&self, samples: &[f32], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.inner.audio_prefix_f32(samples, attend)
    }
    pub(crate) fn audio_prefix_i16<F>(&self, samples: &[i16], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        self.inner.audio_prefix_i16(samples, attend)
    }
    pub(crate) fn image_prefix<F>(&self, images: &[OmniImage<'_>], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.image_prefix(images, attend)
    }
    pub(crate) fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        self.inner.prepare_rows(rows)
    }
    pub(crate) fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk: &mut T,
        head: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.forward_with_attention(rows, trunk, head)
    }
}

trait CoreExecution {
    fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows>;
    fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk: &mut T,
        head: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>;
}
trait VisionExecution {
    fn prepare_rgb8(&self, rgb: &[u8], height: usize, width: usize) -> Result<PreparedVisionImage>;
    fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>;
}
trait AudioExecution {
    fn forward_with_attention<F>(
        &self,
        mel: &Tensor,
        valid: usize,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>;
}
impl CoreExecution for DecisionCore {
    fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        DecisionCore::prepare_rows(self, rows)
    }
    fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk: &mut T,
        head: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        DecisionCore::forward_with_attention(self, rows, trunk, head)
    }
}
impl CoreExecution for QuantizedDecisionCore {
    fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        QuantizedDecisionCore::prepare_rows(self, rows)
    }
    fn forward_with_attention<T, H>(
        &self,
        rows: &PreparedRows,
        trunk: &mut T,
        head: &mut H,
    ) -> Result<Tensor>
    where
        T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor>,
        H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        QuantizedDecisionCore::forward_with_attention(self, rows, trunk, head)
    }
}
impl VisionExecution for Vision {
    fn prepare_rgb8(&self, rgb: &[u8], height: usize, width: usize) -> Result<PreparedVisionImage> {
        Vision::prepare_rgb8(self, rgb, height, width)
    }
    fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        Vision::forward_with_attention(self, images, attend)
    }
}
impl VisionExecution for QuantizedVision {
    fn prepare_rgb8(&self, rgb: &[u8], height: usize, width: usize) -> Result<PreparedVisionImage> {
        QuantizedVision::prepare_rgb8(self, rgb, height, width)
    }
    fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        QuantizedVision::forward_with_attention(self, images, attend)
    }
}
impl AudioExecution for AudioEncoder {
    fn forward_with_attention<F>(
        &self,
        mel: &Tensor,
        valid: usize,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        AudioEncoder::forward_with_attention(self, mel, valid, attend)
    }
}
impl AudioExecution for QuantizedAudioEncoder {
    fn forward_with_attention<F>(
        &self,
        mel: &Tensor,
        valid: usize,
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        QuantizedAudioEncoder::forward_with_attention(self, mel, valid, attend)
    }
}

impl<C: CoreExecution, V: VisionExecution, A: AudioExecution> OmniInner<C, V, A> {
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
        self.limits
            .text_allowance(n.div_ceil(8), self.limits.audio_text_length)
    }
    fn audio_prefix<F>(&self, mel: MelFeatures, attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        let n = mel.valid_frames;
        let rows = n.div_ceil(8);
        self.limits
            .text_allowance(rows, self.limits.audio_text_length)?;
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
        self.limits
            .text_allowance(total, self.limits.image_text_length)?;
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
