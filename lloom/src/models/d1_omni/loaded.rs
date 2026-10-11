//! Factory ownership integration over the shared memory domain.
//!
//! Source-only intermediate: engine registration is intentionally not enabled
//! until load prediction, mapped/temporary host producers and request admission
//! are complete. No per-component domain or uncharged backend fallback.
use super::{MelFrontend, OmniConfig, OmniModel};
use crate::nn::memory::{LoadContext, Loaded};
use lloom_tensor::{DType, Device, Result};
use std::path::Path;

pub(crate) enum LoadedModel {
    Dense(OmniModel),
    Quantized(super::model::QuantizedOmniModel),
}

pub(crate) struct LoadedOmni {
    loaded: Loaded<LoadedModel>,
}
impl LoadedOmni {
    /// The embedding/app factory creates LoadContext through MachineLoad's
    /// existing admission path, before calling this constructor. Namespace and
    /// host tables receive THAT identity, not an identity inferred from Device.
    ///
    /// # Safety
    /// Safetensor files must not be modified for the mapping's lifetime, as
    /// required by VarBuilder::from_mmaped_safetensors. This method does not
    /// yet account mmap/metadata and host transfer temporaries; not enabled.
    pub(crate) unsafe fn from_safetensors<P: AsRef<Path>>(
        config: OmniConfig,
        paths: &[P],
        dtype: DType,
        device: &Device,
        load: LoadContext,
    ) -> Result<Self> {
        config.validate(dtype)?;
        let loaded = Loaded::new(device, load, |device, context| {
            // Builder creation remains inside the supplied load scope.
            let vb =
                unsafe { crate::nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
            let frontend = MelFrontend::new_in(config.audio_config.feat_in, context)?;
            OmniModel::load_components_with_frontend(config, vb, frontend).map(LoadedModel::Dense)
        })?;
        Ok(Self { loaded })
    }

    /// Main and media headers are inspected inside the supplied load scope;
    /// both eager maps use those same readers, after all-component preflight.
    /// No companion filename is guessed and no second model domain is made.
    pub(crate) fn from_gguf(
        main: &Path,
        media: &Path,
        device: &Device,
        load: LoadContext,
    ) -> Result<Self> {
        let loaded = Loaded::new(device, load, |device, context| {
            use crate::quantized_var_builder::VarBuilder;
            use lloom_tensor::quantized::gguf_file::Content;
            let mut main_file = std::fs::File::open(main)?;
            let mut media_file = std::fs::File::open(media)?;
            let main_header = Content::read(&mut main_file)?;
            let media_header = Content::read(&mut media_file)?;
            let cfg = super::gguf::resolve(&main_header, &media_header)?;
            super::gguf::validate_demands(&cfg, &main_header, &media_header)?;
            main_header.validate_payloads(&mut main_file)?;
            media_header.validate_payloads(&mut media_file)?;
            let main_vb = VarBuilder::from_gguf_content(&main_header, &mut main_file, device)?;
            let media_vb = VarBuilder::from_gguf_content(&media_header, &mut media_file, device)?;
            let frontend = MelFrontend::new_in(cfg.audio_config.feat_in, context)?;
            super::model::QuantizedOmniModel::load_components_with_frontend(
                cfg, main_vb, media_vb, frontend,
            )
            .map(LoadedModel::Quantized)
        })?;
        Ok(Self { loaded })
    }

    /// Whole-pass demand comes from the shared predictor/caller, NOT a second
    /// quota. This is an intermediate seam: host request producers still need
    /// tickets and backend workspace obligations before app activation.
    pub(crate) fn with_pass<T>(
        &self,
        predicted_extra: usize,
        execute: impl FnOnce(&LoadedModel) -> Result<T>,
    ) -> Result<T> {
        self.loaded
            .with_pass(predicted_extra, |model, _context| execute(model))
    }
}
