use super::*;
use lloom_tensor::quantized::gguf_file::{TensorInfo, VersionedMagic};
use lloom_tensor::quantized::GgmlDType;

fn header(entries: Vec<(&str, Value)>, tensors: Vec<(&str, &[usize])>) -> Content {
    Content {
        magic: VersionedMagic::GgufV3,
        metadata: entries.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        tensor_infos: tensors
            .into_iter()
            .map(|(name, dims)| {
                (
                    name.into(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: Shape::from(dims),
                        offset: 0,
                    },
                )
            })
            .collect(),
        tensor_data_offset: 0,
    }
}
fn fixture() -> (Content, Content) {
    use Value::{Array as A, Bool as B, String as S, F32 as F, U32 as U};
    // Header-only source-schema fixture: no tensor payload exists or is read.
    // Tiny widths preserve canonical converter names and resolved layer axes.
    let main = header(
        vec![
            ("general.architecture", S("lfm2".into())),
            ("lfm2.decision.type", S("lfm2-d1-omni".into())),
            ("lfm2.attention.causal", B(false)),
            ("tokenizer.ggml.token_type_count", U(3)),
            ("lfm2.attention.layer_norm_epsilon", F(1e-5)),
            ("lfm2.block_count", U(3)),
            ("lfm2.decision.block_count", U(1)),
            ("lfm2.embedding_length", U(64)),
            ("lfm2.attention.head_count", U(1)),
            ("lfm2.feed_forward_length", A(vec![U(32), U(32), U(256)])),
            ("lfm2.attention.head_count_kv", A(vec![U(0), U(1), U(1)])),
            ("lfm2.vocab_size", U(8)),
            ("lfm2.context_length", U(16384)),
            ("lfm2.attention.layer_norm_rms_epsilon", F(1e-5)),
            ("lfm2.shortconv.l_cache", U(3)),
            ("lfm2.rope.freq_base", F(1_000_000.)),
            ("lfm2.decision.temperature.choice.3_5", F(2.)),
            ("lfm2.decision.temperature.score.11", F(3.)),
            ("lfm2.decision.temperature.noul", F(4.)),
        ],
        vec![
            ("token_embd.weight", &[8, 64]),
            ("blk.0.shortconv.conv.weight", &[64, 3]),
            ("blk.1.attn_q.weight", &[64, 64]),
        ],
    );
    let media = header(
        vec![
            ("clip.vision.projector_type", S("d1omni_v".into())),
            ("clip.audio.projector_type", S("d1omni_a".into())),
            ("clip.has_vision_encoder", B(true)),
            ("clip.has_audio_encoder", B(true)),
            ("clip.use_gelu", B(true)),
            ("clip.vision.projection_dim", U(64)),
            ("clip.audio.projection_dim", U(64)),
            ("clip.vision.projector.scale_factor", U(2)),
            ("clip.vision.image_mean", A(vec![F(0.5); 3])),
            ("clip.vision.image_std", A(vec![F(0.5); 3])),
            ("clip.vision.embedding_length", U(4)),
            ("clip.vision.feed_forward_length", U(8)),
            ("clip.vision.block_count", U(1)),
            ("clip.vision.attention.head_count", U(1)),
            ("clip.vision.patch_size", U(16)),
            ("clip.vision.attention.layer_norm_epsilon", F(1e-6)),
            ("clip.audio.attention.layer_norm_epsilon", F(1e-5)),
            ("clip.audio.embedding_length", U(4)),
            ("clip.audio.feed_forward_length", U(8)),
            ("clip.audio.num_mel_bins", U(8)),
            ("clip.audio.block_count", U(1)),
            ("clip.audio.attention.head_count", U(2)),
        ],
        vec![
            ("v.position_embd.weight", &[4, 4]),
            ("mm.1.weight", &[8, 16]),
            ("a.conv1d.0.weight", &[2, 1, 3, 3]),
            ("a.blk.0.conv_dw.weight", &[4, 3]),
            ("mm.a.mlp.5.weight", &[3, 64]),
        ],
    );
    (main, media)
}

#[test]
fn canonical_header_resolves_dimensions_without_payload_or_inverse_ffn() -> Result<()> {
    let (main, media) = fixture();
    let cfg = resolve(&main, &media)?;
    assert_eq!(cfg.core.text.intermediate_size, 32); // not Python's2/3 transform
    assert_eq!(cfg.core.text.layer_types, ["conv", "full_attention"]);
    assert_eq!(cfg.core.head_layers, 1);
    assert_eq!(cfg.core.max_length, 16384);
    assert_eq!((cfg.image_text_length, cfg.audio_text_length), (896, 15360));
    assert_eq!(
        (cfg.vision_config.num_patches, cfg.projector_hidden_size),
        (4, 8)
    );
    assert_eq!(
        (
            cfg.audio_config.subsampling_conv_channels,
            cfg.audio_config.conv_kernel_size,
            cfg.audio_config.ff_expansion_factor,
            cfg.audio_config.residual_width
        ),
        (2, 3, 2, 3)
    );
    assert_eq!(
        cfg.temperatures,
        BTreeMap::from([
            ("choice:3-5".into(), 2.),
            ("score:11+".into(), 3.),
            ("noul".into(), 4.),
        ])
    );
    Ok(())
}

#[test]
fn malformed_headers_refuse_before_any_eager_builder() {
    for (key, value, message) in [
        ("lfm2.block_count", Value::U64(u64::MAX), "block count"),
        (
            "lfm2.embedding_length",
            Value::Bool(true),
            "unsigned integer",
        ),
        (
            "lfm2.feed_forward_length",
            Value::Array(vec![]),
            "layer count mismatch",
        ),
        ("lfm2.attention.causal", Value::Bool(true), "must be false"),
        (
            "lfm2.decision.temperature.noul",
            Value::F32(f32::NAN),
            "temperatures",
        ),
    ] {
        let (mut main, media) = fixture();
        main.metadata.insert(key.into(), value);
        let error = resolve(&main, &media).unwrap_err().to_string();
        // U64MAX is nonrepresentable on32bit before the header-count guard.
        if key == "lfm2.block_count" && usize::BITS == 32 {
            assert!(error.contains("out of range"), "{error}");
        } else {
            assert!(error.contains(message), "{key}: {error}");
        }
    }
    for key in ["clip.vision.block_count", "clip.audio.block_count"] {
        let (main, mut media) = fixture();
        media.metadata.insert(key.into(), Value::U32(u32::MAX));
        assert!(resolve(&main, &media)
            .unwrap_err()
            .to_string()
            .contains("available tensor headers"));
    }
    let (main, mut media) = fixture();
    media
        .tensor_infos
        .get_mut("a.conv1d.0.weight")
        .unwrap()
        .shape = Shape::from((2, 3));
    assert!(resolve(&main, &media).is_err());
    let (main, mut media) = fixture();
    media.metadata.insert(
        "clip.vision.image_std".into(),
        Value::Array(vec![Value::F32(1.); 3]),
    );
    assert!(resolve(&main, &media)
        .unwrap_err()
        .to_string()
        .contains("normalization"));
}

#[test]
fn incomplete_weight_headers_refuse_before_payload_materialization() -> Result<()> {
    let (main, media) = fixture();
    let cfg = resolve(&main, &media)?;
    assert!(validate_demands(&cfg, &main, &media)
        .unwrap_err()
        .to_string()
        .contains("tensor header"));
    Ok(())
}
