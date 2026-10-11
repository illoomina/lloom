use super::*;
use std::collections::HashMap;

fn config() -> OmniConfig {
    serde_json::from_str(include_str!("config_fixture.json")).unwrap()
}
#[test]
fn all_component_preflight_refuses_before_first_weight_lookup() {
    // No model weights/storage. A missed preflight would produce the backend's
    // missing-tensor error, not these specific metadata-validation errors.
    let empty = || VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    for (arm, message) in [
        (0, "audio dimensions"),
        (1, "omni vision"),
        (2, "head width"),
        (3, "RoPE theta"),
    ] {
        let mut cfg = config();
        match arm {
            0 => cfg.audio_config.n_heads = 0,
            1 => cfg.vision_config.num_attention_heads = 0,
            2 => {
                cfg.core.text_config.hidden_size = 32;
                cfg.core.text_config.num_attention_heads = 8;
                cfg.core.text_config.num_key_value_heads = 4;
            }
            _ => cfg.core.text_config.rope_theta = f64::MAX,
        }
        let error = OmniModel::load_components(cfg, empty())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(message),
            "expected preflight {message}, got {error}"
        );
    }
}
#[test]
fn pure_head_and_context_metadata_never_allocate_large_tensors() -> Result<()> {
    let cfg = config();
    assert_eq!(cfg.text_allowance(0, cfg.core.max_length)?, 16384);
    assert_eq!(cfg.text_allowance(256, cfg.image_text_length)?, 896);
    assert_eq!(cfg.text_allowance(375, cfg.audio_text_length)?, 15360);
    assert_eq!(cfg.text_allowance(16320, 16384)?, 64);
    assert!(cfg.text_allowance(16321, 16384).is_err());
    assert!(cfg.text_allowance(usize::MAX, 16384).is_err());
    use super::super::DecisionHead;
    assert_eq!(
        DecisionHead::validate_config(64, 0, DType::F32)?,
        (1, 192, 256)
    );
    assert!(DecisionHead::validate_config(64, usize::MAX, DType::F32).is_err());
    let wide = 1usize << (usize::BITS / 2);
    assert!(DecisionHead::validate_config(wide, 0, DType::F32).is_err());
    assert!(DecisionHead::validate_config(wide, 1, DType::F32).is_err());
    Ok(())
}

#[test]
fn gguf_aggregate_preflight_refuses_before_first_weight_lookup() -> Result<()> {
    use super::super::trunk::ResolvedTextConfig;
    use lloom_tensor::quantized::gguf_file;
    let cfg = config();
    let t = &cfg.core.text_config;
    let resolved = ResolvedOmniConfig {
        core: ResolvedDecisionCoreConfig {
            text: ResolvedTextConfig {
                vocab_size: t.vocab_size,
                hidden_size: t.hidden_size,
                intermediate_size: t.validated_intermediate()?,
                num_hidden_layers: t.num_hidden_layers,
                num_attention_heads: t.num_attention_heads,
                num_key_value_heads: t.num_key_value_heads,
                layer_types: t.layer_types.clone(),
                norm_eps: t.norm_eps,
                conv_l_cache: t.conv_l_cache,
                max_position_embeddings: t.max_position_embeddings,
                rope_theta: t.rope_theta,
            },
            head_layers: cfg.core.head_layers,
            max_length: cfg.core.max_length,
        },
        vision_config: cfg.vision_config,
        audio_config: cfg.audio_config,
        projector_hidden_size: cfg.projector_hidden_size,
        image_text_length: cfg.image_text_length,
        audio_text_length: cfg.audio_text_length,
        temperatures: cfg.temperatures,
    };
    resolved.validate()?;
    let mut empty = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut empty, &[], &[])?;
    assert!(empty.get_ref().len() <= 64);
    let vb = QuantizedVarBuilder::from_gguf_buffer(empty.get_ref(), &Device::Cpu)?;
    // The valid metadata reaches a missing lookup, not any tensor allocation.
    let missing = QuantizedOmniModel::load_components(resolved.clone(), vb.clone(), vb.clone())
        .unwrap_err()
        .to_string();
    assert!(missing.contains("cannot find tensor"), "{missing}");
    for (arm, message) in [
        (0, "audio dimensions"),
        (1, "omni vision"),
        (2, "head width"),
        (3, "RoPE theta"),
        (4, "audio shape/byte overflow"),
        (5, "capacity"),
    ] {
        let mut invalid = resolved.clone();
        match arm {
            0 => invalid.audio_config.n_heads = 0,
            1 => invalid.vision_config.num_attention_heads = 0,
            2 => {
                invalid.core.text.hidden_size = 32;
                invalid.core.text.num_attention_heads = 8;
                invalid.core.text.num_key_value_heads = 4;
            }
            3 => invalid.core.text.rope_theta = f64::MAX,
            4 => invalid.audio_config.n_layers = usize::MAX,
            _ => invalid.vision_config.num_hidden_layers = usize::MAX,
        }
        let error = QuantizedOmniModel::load_components(invalid, vb.clone(), vb.clone())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(message),
            "expected preflight {message}, got {error}"
        );
        assert!(
            !error.contains("cannot find tensor"),
            "late validation: {error}"
        );
    }
    Ok(())
}
