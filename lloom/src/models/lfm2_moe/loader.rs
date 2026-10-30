use super::{config::MoeConfig, Config};
use crate::models::lfm2;
use crate::models::qwen4exp::moe::{MoeGeometry, MoeLayer, Resident};
use crate::nn::moe::Routing;
use crate::nn::VarBuilder;
use crate::profile::Profile;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{bail, DType, Result, Tensor};
use std::sync::Arc;

impl Config {
    /// Build on the existing LFM decoder/state/driver. `runtime` may select LFM
    /// execution arms, but may not reinterpret the checkpoint's geometry.
    pub fn build<P: Profile>(
        &self,
        runtime: &lfm2::Config,
        vb: VarBuilder,
    ) -> Result<lfm2::Model<P>> {
        let expected = self.lfm_config()?;
        if runtime.hidden_size != expected.hidden_size
            || runtime.vocab_size != expected.vocab_size
            || runtime.intermediate_size != expected.intermediate_size
            || runtime.num_hidden_layers != expected.num_hidden_layers
            || runtime.num_attention_heads != expected.num_attention_heads
            || runtime.num_key_value_heads != expected.num_key_value_heads
            || runtime.layer_types != expected.layer_types
            || runtime.tie_embedding != expected.tie_embedding
            || runtime.conv_l_cache != expected.conv_l_cache
            || runtime.conv_bias != expected.conv_bias
            || runtime.rope_theta != expected.rope_theta
            || runtime.norm_eps != expected.norm_eps
            || runtime.moe != expected.moe
        {
            bail!("lfm2_moe: runtime geometry differs from the checkpoint schema")
        }
        lfm2::Model::new(runtime, vb)
    }
}

impl MoeConfig {
    pub(crate) fn load_ffn<P: Profile>(
        &self,
        h: usize,
        vb: VarBuilder,
    ) -> Result<MoeLayer<P, Resident>> {
        let (i, e) = (self.moe_intermediate_size, self.num_experts);
        let kind = match vb.dtype() {
            DType::F32 => GgmlDType::F32,
            DType::F16 => GgmlDType::F16,
            DType::BF16 => GgmlDType::BF16,
            dtype => bail!("lfm2_moe: unsupported compute dtype {dtype:?}"),
        };
        // Pack once at load, never in the token path. The existing residency
        // and routed matmuls consume these stacks without per-expert dispatch.
        let cpu = vb.clone().set_device(lloom_tensor::Device::Cpu);
        let stack = |name: &str, out: usize, inp: usize| -> Result<Arc<QTensor>> {
            let planes = (0..e)
                .map(|expert| {
                    cpu.pp("experts")
                        .pp(expert)
                        .pp(name)
                        .get((out, inp), "weight")
                })
                .collect::<Result<Vec<_>>>()?;
            let packed = Tensor::stack(&planes, 0)?;
            drop(planes);
            Ok(Arc::new(QTensor::quantize_onto(
                &packed,
                kind,
                vb.device(),
            )?))
        };
        let experts = Resident::new(stack("w1", i, h)?, stack("w3", i, h)?, stack("w2", h, i)?);
        let router = QMatMul::Tensor(vb.pp("gate").get((e, h), "weight")?);
        let bias = if self.use_expert_bias {
            Some(vb.to_dtype(DType::F32).get(e, "expert_bias")?)
        } else {
            None
        };
        Ok(MoeLayer::routed_only(
            router,
            experts,
            &MoeGeometry {
                n_experts: e,
                n_experts_per_tok: self.num_experts_per_tok,
                moe_intermediate_size: i,
            },
            Routing::Sigmoid {
                bias,
                normalize: self.norm_topk_prob,
                scale: self.routed_scaling_factor,
            },
        )?
        .with_weight_names(
            (
                "gate.weight",
                "experts.w1.packed",
                "experts.w3.packed",
                "experts.w2.packed",
            ),
            vb.dtype(),
        ))
    }
}

/// Header-derived resident payload at the compute dtype and the separate
/// conservative load-time packing allowance. Neither certifies a model run.
#[derive(Debug, Clone, Copy)]
pub struct Preflight {
    pub resident_bytes: usize,
    pub packing_bytes: usize,
}

/// Validate every required tensor before materialization, reusing the shared
/// safetensors reader and the same schema used by synthetic fixtures.
///
/// # Safety
/// The files must not be modified while their read-only mappings are alive.
pub unsafe fn preflight(
    config: &Config,
    files: &[std::path::PathBuf],
    compute: DType,
) -> Result<Preflight> {
    if !matches!(compute, DType::F32 | DType::F16 | DType::BF16) {
        bail!("lfm2_moe: unsupported compute dtype {compute:?}")
    }
    let expected = super::schema::tensor_schema(config, GgmlDType::F32)?;
    let mapped = files
        .iter()
        .map(|file| {
            // SAFETY: inherited from this function's mapping contract. Header
            // validation must not advise the weight payload into the page cache.
            unsafe {
                lloom_tensor::safetensors::MmapedSafetensors::with_advice(
                    file,
                    lloom_tensor::safetensors::MmapAdvice::None,
                )
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let headers: Vec<_> = mapped.iter().flat_map(|m| m.tensors()).collect();
    let names: std::collections::HashSet<_> = headers.iter().map(|(n, _)| n.as_str()).collect();
    if names.len() != headers.len() {
        bail!("lfm2_moe: duplicate tensor names across shards")
    }
    expected.validate(headers.iter().map(|(name, t)| (name.as_str(), t.shape())))?;
    let mut resident_bytes = 0usize;
    for spec in &expected.tensors {
        let t = headers
            .iter()
            .find(|(name, _)| name == &spec.name)
            .map(|(_, t)| t)
            .expect("schema validation established every required name");
        if !matches!(
            t.dtype(),
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32
        ) {
            bail!(
                "{}: unsupported safetensors dtype {:?}",
                spec.name,
                t.dtype()
            )
        }
        let size = if spec.name.ends_with(".expert_bias") {
            if t.dtype() != safetensors::Dtype::F32 {
                bail!("{}: expert bias must be F32", spec.name)
            }
            4
        } else {
            compute.size_in_bytes()
        };
        let bytes = spec
            .shape
            .iter()
            .try_fold(size, |n, d| n.checked_mul(*d))
            .ok_or_else(|| lloom_tensor::Error::Msg(format!("{}: size overflow", spec.name)))?;
        resident_bytes = resident_bytes.checked_add(bytes).ok_or_else(|| {
            lloom_tensor::Error::Msg("lfm2_moe: resident payload overflow".into())
        })?;
    }
    let runtime = config.lfm_config()?;
    Ok(Preflight {
        resident_bytes,
        packing_bytes: runtime
            .moe
            .expect("MoE config supplies its FFN")
            .packing_bound(runtime.hidden_size, compute),
    })
}
