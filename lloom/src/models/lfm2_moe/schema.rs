use super::Config;
use crate::models::lfm2::{layer_group_parts, LayerType};
use crate::testing::tiny::{Role, Schema, TensorSpec};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{bail, Result};

/// Original safetensors names, not the stacked-expert layout of the reference
/// implementation's in-memory tensors. Bias is F32 even in a BF16 checkpoint.
pub fn tensor_schema(config: &Config, store: GgmlDType) -> Result<Schema> {
    if !matches!(store, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
        bail!("lfm2_moe safetensors cannot store {store:?}")
    }
    let cfg = config.lfm_config()?;
    let mut tensors = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, role: Role, dtype: GgmlDType| {
        tensors.push(TensorSpec {
            name,
            shape,
            role,
            dtype,
        });
    };
    let h = cfg.hidden_size;
    push(
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, h],
        Role::Matrix { fan_in: h },
        store,
    );
    push(
        "model.embedding_norm.weight".into(),
        vec![h],
        Role::GAIN,
        store,
    );
    if !cfg.tie_embedding {
        push(
            "lm_head.weight".into(),
            vec![cfg.vocab_size, h],
            Role::Matrix { fan_in: h },
            store,
        );
    }
    for (l, kind) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.layers.{l}");
        for name in ["operator_norm", "ffn_norm"] {
            push(format!("{p}.{name}.weight"), vec![h], Role::GAIN, store);
        }
        // The existing LFM schema supplies the mixer's projection geometry.
        for (name, shape) in layer_group_parts(&cfg, l).into_iter().skip(3) {
            let shape = shape.dims().to_vec();
            let fan_in = shape[1];
            push(name, shape, Role::Matrix { fan_in }, store);
        }
        match kind {
            LayerType::Conv => {
                push(
                    format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                    Role::Matrix {
                        fan_in: cfg.conv_l_cache,
                    },
                    store,
                );
                if cfg.conv_bias {
                    push(format!("{p}.conv.conv.bias"), vec![h], Role::BIAS, store);
                    push(
                        format!("{p}.conv.in_proj.bias"),
                        vec![3 * h],
                        Role::BIAS,
                        store,
                    );
                    push(
                        format!("{p}.conv.out_proj.bias"),
                        vec![h],
                        Role::BIAS,
                        store,
                    );
                }
            }
            LayerType::FullAttention => {
                for name in ["q_layernorm", "k_layernorm"] {
                    push(
                        format!("{p}.self_attn.{name}.weight"),
                        vec![cfg.head_dim()],
                        Role::GAIN,
                        store,
                    );
                }
            }
        }
        if config.is_dense(l) {
            for (name, shape) in layer_group_parts(&cfg, l).into_iter().take(3) {
                let shape = shape.dims().to_vec();
                let fan_in = shape[1];
                push(name, shape, Role::Matrix { fan_in }, store);
            }
        } else {
            let p = format!("{p}.feed_forward");
            push(
                format!("{p}.gate.weight"),
                vec![config.num_experts, h],
                Role::Matrix { fan_in: h },
                store,
            );
            if config.use_expert_bias {
                push(
                    format!("{p}.expert_bias"),
                    vec![config.num_experts],
                    Role::BIAS,
                    GgmlDType::F32,
                );
            }
            let i = config.moe_intermediate_size;
            for e in 0..config.num_experts {
                for (name, shape, fan_in) in [
                    ("w1", vec![i, h], h),
                    ("w3", vec![i, h], h),
                    ("w2", vec![h, i], i),
                ] {
                    push(
                        format!("{p}.experts.{e}.{name}.weight"),
                        shape,
                        Role::Matrix { fan_in },
                        store,
                    );
                }
            }
        }
    }
    Ok(Schema { tensors })
}

/// Resident tensor payload only. Loader staging/packing, state and activations
/// must be admitted separately; this is not a peak-memory prediction.
pub fn payload_bytes(schema: &Schema) -> Result<usize> {
    schema.tensors.iter().try_fold(0usize, |total, t| {
        let bytes = t
            .shape
            .iter()
            .try_fold(t.dtype.type_size(), |n, &d| n.checked_mul(d))
            .ok_or_else(|| {
                lloom_tensor::Error::Msg(format!("{}: payload size overflow", t.name))
            })?;
        total
            .checked_add(bytes)
            .ok_or_else(|| lloom_tensor::Error::Msg("lfm2_moe: payload size overflow".into()))
    })
}
