//! One existing dense/GGUF omni owner and caller metadata in the shared domain.
//!
//! Explicit embedding seam, not default registration or complete admission.
//! Header/mmap/request/returned host producers and native error custody remain
//! coverage obligations; no per-component domain or uncharged backend fallback.
use super::{
    DecisionRow, FrontendScratch, MelFrontend, OmniConfig, OmniImage, OmniModel, PreparedRows,
    RelativeInputs, TrunkBatch,
};
use crate::nn::memory::{LoadContext, Loaded};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

enum LoadedModel {
    Dense(OmniModel),
    Quantized(super::model::QuantizedOmniModel),
}
struct Owned<T> {
    model: LoadedModel,
    metadata: T,
}

/// Borrowed format-independent facts from the actual resolved checkpoint.
/// This is not a second configuration schema, allocation quote or model copy.
#[derive(Clone, Copy)]
pub struct OmniMetadata<'a> {
    pub vocab_size: usize,
    pub max_length: usize,
    pub image_text_length: usize,
    pub audio_text_length: usize,
    pub temperatures: &'a BTreeMap<String, f64>,
}
impl OmniMetadata<'_> {
    /// Preserve the incumbent minimum64 media text allowance.
    pub fn text_allowance(self, prefix_tokens: usize, configured_limit: usize) -> Result<usize> {
        super::model::text_allowance(self.max_length, prefix_tokens, configured_limit)
    }
}
fn dense_metadata(config: &OmniConfig) -> OmniMetadata<'_> {
    OmniMetadata {
        vocab_size: config.core.text_config.vocab_size,
        max_length: config.core.max_length,
        image_text_length: config.image_text_length,
        audio_text_length: config.audio_text_length,
        temperatures: &config.temperatures,
    }
}
fn quantized_metadata(config: &super::model::ResolvedOmniConfig) -> OmniMetadata<'_> {
    OmniMetadata {
        vocab_size: config.core.text.vocab_size,
        max_length: config.core.max_length,
        image_text_length: config.image_text_length,
        audio_text_length: config.audio_text_length,
        temperatures: &config.temperatures,
    }
}

/// Opaque borrowed facade over the same existing model. Format selection occurs
/// once per component operation, not per layer; quantized internals stay private.
/// All callbacks remain explicit and monomorphized. Returned tensors and callback
/// work require the enclosing pass's real lifetime/queue coverage.
pub struct OmniPass<'a> {
    model: &'a LoadedModel,
}
macro_rules! dispatch {
    ($owner:expr, $name:ident $(, $argument:expr)*) => {
        match $owner.model {
            LoadedModel::Dense(model) => model.$name($($argument),*),
            LoadedModel::Quantized(model) => model.$name($($argument),*),
        }
    };
}
impl OmniPass<'_> {
    pub fn metadata(&self) -> OmniMetadata<'_> {
        match self.model {
            LoadedModel::Dense(model) => dense_metadata(model.config()),
            LoadedModel::Quantized(model) => quantized_metadata(model.config()),
        }
    }
    pub fn audio_frontend_scratch(&self, samples: usize) -> Result<FrontendScratch> {
        dispatch!(self, audio_frontend_scratch, samples)
    }
    pub fn audio_prefix_f32<F>(&self, samples: &[f32], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        dispatch!(self, audio_prefix_f32, samples, attend)
    }
    pub fn audio_prefix_i16<F>(&self, samples: &[i16], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(RelativeInputs<'_>) -> Result<Tensor>,
    {
        dispatch!(self, audio_prefix_i16, samples, attend)
    }
    pub fn image_prefix<F>(&self, images: &[OmniImage<'_>], attend: &mut F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        dispatch!(self, image_prefix, images, attend)
    }
    pub fn prepare_rows(&self, rows: &[DecisionRow<'_>]) -> Result<PreparedRows> {
        dispatch!(self, prepare_rows, rows)
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
        dispatch!(self, forward_with_attention, rows, trunk, head)
    }
}

/// Who scopes the owner's request work. `Domain`: one model domain, and each
/// pass reserves the caller's prediction in it. `Machine`: admitted on the
/// machine at load (`lloom serve`'s default route, as the causal families'
/// is), each pass run directly with no per-pass reservation.
enum Custody<T> {
    Domain(Loaded<Owned<T>>),
    Machine {
        owned: Owned<T>,
        /// The machine reservation the load took, held for the model's life.
        _machine: lloom_tensor::machine::Admitted,
    },
}

/// SAME existing Loaded namespace holds the model and once-produced metadata.
/// T need not be HTTP Sync. No unscoped model/tokenizer getter is exposed.
pub struct LoadedOmni<T = ()> {
    custody: Custody<T>,
}

/// The safetensors model; `paths` must stay unchanged while mapped.
unsafe fn produce_dense<P: AsRef<Path>, T>(
    config: OmniConfig,
    paths: &[P],
    dtype: DType,
    device: &Device,
    frontend: MelFrontend,
    preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
) -> Result<Owned<T>> {
    let metadata = preflight(dense_metadata(&config))?;
    let vb = unsafe { crate::nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    let model = OmniModel::load_components_with_frontend(config, vb, frontend)?;
    Ok(Owned {
        model: LoadedModel::Dense(model),
        metadata,
    })
}

/// The main GGUF and its mmproj: every component demand and payload extent is
/// checked, and `preflight` run, before either file's tensors are read.
fn produce_gguf<T>(
    main: &Path,
    media: &Path,
    device: &Device,
    frontend: impl FnOnce(usize) -> Result<MelFrontend>,
    preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
) -> Result<Owned<T>> {
    use crate::quantized_var_builder::VarBuilder;
    use lloom_tensor::quantized::gguf_file::Content;
    let mut main_file = std::fs::File::open(main)?;
    let mut media_file = std::fs::File::open(media)?;
    let main_header = Content::read(&mut main_file)?;
    let media_header = Content::read(&mut media_file)?;
    let config = super::gguf::resolve(&main_header, &media_header)?;
    super::gguf::validate_demands(&config, &main_header, &media_header)?;
    main_header.validate_payloads(&mut main_file)?;
    media_header.validate_payloads(&mut media_file)?;
    let metadata = preflight(quantized_metadata(&config))?;
    let main_vb = VarBuilder::from_gguf_content(&main_header, &mut main_file, device)?;
    let media_vb = VarBuilder::from_gguf_content(&media_header, &mut media_file, device)?;
    let frontend = frontend(config.audio_config.feat_in)?;
    let model = super::model::QuantizedOmniModel::load_components_with_frontend(
        config, main_vb, media_vb, frontend,
    )?;
    Ok(Owned {
        model: LoadedModel::Quantized(model),
        metadata,
    })
}

impl<T> LoadedOmni<T> {
    /// Metadata preflight runs in the supplied load scope BEFORE mapping/weight
    /// lookup. Its allocation charges are not implied by this call's name.
    ///
    /// # Safety
    /// Safetensor files must remain unchanged for the mapping's lifetime, as
    /// required by VarBuilder::from_mmaped_safetensors. Mmap/metadata and host
    /// transfers/native partial-construction custody are not fully enrolled.
    pub unsafe fn from_safetensors_with<P: AsRef<Path>>(
        config: OmniConfig,
        paths: &[P],
        dtype: DType,
        device: &Device,
        load: LoadContext,
        preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        config.validate(dtype)?;
        let loaded = Loaded::new(device, load, |device, context| {
            let frontend = MelFrontend::new_in(config.audio_config.feat_in, context)?;
            unsafe { produce_dense(config, paths, dtype, device, frontend, preflight) }
        })?;
        Ok(Self {
            custody: Custody::Domain(loaded),
        })
    }
    /// Explicit main+media files; all component demands/payload extents and
    /// metadata preflight precede eager tensor maps on those SAME readers.
    pub fn from_gguf_with(
        main: &Path,
        media: &Path,
        device: &Device,
        load: LoadContext,
        preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        let loaded = Loaded::new(device, load, |device, context| {
            produce_gguf(
                main,
                media,
                device,
                |features| MelFrontend::new_in(features, context),
                preflight,
            )
        })?;
        Ok(Self {
            custody: Custody::Domain(loaded),
        })
    }
    /// [`Self::from_safetensors_with`], admitted on the machine at load.
    ///
    /// # Safety
    /// As [`Self::from_safetensors_with`]: the files must not change while mapped.
    pub unsafe fn from_safetensors_on_machine<P: AsRef<Path>>(
        config: OmniConfig,
        paths: &[P],
        dtype: DType,
        device: &Device,
        machine: lloom_tensor::machine::Admitted,
        preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        config.validate(dtype)?;
        let frontend = MelFrontend::new(config.audio_config.feat_in)?;
        let owned = unsafe { produce_dense(config, paths, dtype, device, frontend, preflight)? };
        Ok(Self {
            custody: Custody::Machine {
                owned,
                _machine: machine,
            },
        })
    }
    /// [`Self::from_gguf_with`], admitted on the machine at load.
    pub fn from_gguf_on_machine(
        main: &Path,
        media: &Path,
        device: &Device,
        machine: lloom_tensor::machine::Admitted,
        preflight: impl FnOnce(OmniMetadata<'_>) -> Result<T>,
    ) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        let owned = produce_gguf(main, media, device, MelFrontend::new, preflight)?;
        Ok(Self {
            custody: Custody::Machine {
                owned,
                _machine: machine,
            },
        })
    }
    /// Caller aggregate coverage reserves BEFORE invoking request producers
    /// (a machine-admitted owner reserves nothing per pass). Complete charging
    /// and returned/asynchronous lifetime enrollment remain separate
    /// obligations; this is not a new quota or native retirement proof.
    pub fn with_pass<R>(
        &self,
        predicted_extra: usize,
        execute: impl FnOnce(OmniPass<'_>, &T) -> Result<R>,
    ) -> Result<R> {
        let run = |owned: &Owned<T>| {
            execute(
                OmniPass {
                    model: &owned.model,
                },
                &owned.metadata,
            )
        };
        match &self.custody {
            Custody::Domain(loaded) => {
                loaded.with_pass(predicted_extra, |owned, _context| run(owned))
            }
            Custody::Machine { owned, .. } => run(owned),
        }
    }
}
impl LoadedOmni<()> {
    /// Preserve the model-only constructor; see from_safetensors_with's safety contract.
    /// # Safety
    /// Files must not change for the lifetime of their safetensor mappings.
    pub unsafe fn from_safetensors<P: AsRef<Path>>(
        config: OmniConfig,
        paths: &[P],
        dtype: DType,
        device: &Device,
        load: LoadContext,
    ) -> Result<Self> {
        unsafe { Self::from_safetensors_with(config, paths, dtype, device, load, |_| Ok(())) }
    }
    pub fn from_gguf(
        main: &Path,
        media: &Path,
        device: &Device,
        load: LoadContext,
    ) -> Result<Self> {
        Self::from_gguf_with(main, media, device, load, |_| Ok(()))
    }
}
