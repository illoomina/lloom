//! Factory ownership integration over the shared memory domain.
//!
//! Source-only intermediate: engine registration is intentionally not enabled
//! until load prediction, mapped/temporary host producers and request admission
//! are complete. No per-component domain or uncharged backend fallback.
use super::{MelFrontend, OmniConfig, OmniModel};
use crate::nn::memory::LoadContext;
use lloom_tensor::{DType, Device, Result};
use std::path::Path;

pub(crate) struct LoadedOmni {
    // Drop model-owned host tables/tensors before the execution namespace.
    model: OmniModel,
    #[cfg(feature = "metal")]
    allocations: lloom_tensor::metal_backend::MetalAllocationDomain,
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
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = device {
            let allocations = lloom_tensor::metal_backend::MetalAllocationDomain::new(
                device,
                load.domain().clone(),
            )?;
            let context = load.allocations();
            let scope = allocations.scope(context.clone())?;
            let model = scope.run(|resolved| {
                let device = Device::Metal(resolved.clone());
                // The actual builder is created inside the scope, not passed
                // in after eager allocations have already escaped enrollment.
                let vb = unsafe {
                    crate::nn::VarBuilder::from_mmaped_safetensors(paths, dtype, &device)?
                };
                let frontend = MelFrontend::new_in(config.audio_config.feat_in, &context)?;
                OmniModel::load_components_with_frontend(config, vb, frontend)
            })?;
            // Close unused load issuance only after the synchronous producer
            // scope returns; issued backing charges retain the same domain.
            let domain = load.finish();
            debug_assert_eq!(allocations.domain().key(), domain.key());
            return Ok(Self { model, allocations });
        }
        let _ = (config, paths, dtype, device, load);
        lloom_tensor::bail!(
            "admitted omni loading requires enrolled allocator producers; CPU/CUDA enrollment is pending"
        )
    }

    /// Whole-pass demand comes from the shared predictor/caller, NOT a second
    /// quota. This is an intermediate seam: host request producers still need
    /// tickets and backend workspace obligations before app activation.
    pub(crate) fn with_pass<T>(
        &self,
        predicted_extra: usize,
        execute: impl FnOnce(&OmniModel) -> Result<T>,
    ) -> Result<T> {
        #[cfg(feature = "metal")]
        {
            let pass = self
                .allocations
                .domain()
                .reserve(predicted_extra)
                .map_err(lloom_tensor::Error::wrap)?;
            let scope = self.allocations.scope(pass.context())?;
            let output = scope.run(|_| execute(&self.model));
            // Cancellation closes future issuance; physical outputs/native
            // retirement retain issued charges rather than this pass handle.
            drop(pass);
            output
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (predicted_extra, execute, &self.model);
            lloom_tensor::bail!("admitted omni execution backend enrollment is pending")
        }
    }
}
