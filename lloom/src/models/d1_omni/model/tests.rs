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
