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

// Integration-only generated weights. The component namespace/schema and
// independent numerical references are tested separately; this is not a
// checkpoint conformance oracle.
struct BoundedWeights {
    elements: std::sync::atomic::AtomicUsize,
}
impl crate::nn::var_builder::SimpleBackend for BoundedWeights {
    fn get(
        &self,
        shape: lloom_tensor::Shape,
        name: &str,
        _hint: crate::nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        use std::sync::atomic::Ordering;
        assert_eq!(dtype, DType::F32);
        assert!(matches!(device, Device::Cpu));
        assert!(name.starts_with("fixture."));
        let n = shape
            .dims()
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .unwrap();
        assert!(n <= 16_384, "unexpected weight {name}: {shape:?}");
        let old = self.elements.fetch_add(n, Ordering::Relaxed);
        assert!(
            old.checked_add(n).unwrap() <= 262_144,
            "fixture weight budget"
        );
        let seed = name
            .bytes()
            .fold(0usize, |sum, byte| (sum + byte as usize) % 17);
        let values = (0..n)
            .map(|i| {
                if name.ends_with("running_var") {
                    1.
                } else if name.ends_with("running_mean") {
                    0.
                } else if shape.rank() == 1 && name.ends_with(".weight") {
                    1. + ((i + seed) % 7) as f32 / 128.
                } else {
                    (((i * 7 + seed * 3) % 17) as f32 - 8.) / 128.
                }
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)
    }
    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> Result<Tensor> {
        lloom_tensor::bail!("unchecked fixture tensor lookup {name}")
    }
    fn contains_tensor(&self, name: &str) -> bool {
        name.starts_with("fixture.")
    }
}
fn aggregate_fixture_config() -> OmniConfig {
    let mut c = config();
    let t = &mut c.core.text_config;
    t.vocab_size = 8;
    t.hidden_size = 64;
    t.intermediate_size = 96;
    t.num_hidden_layers = 2;
    t.num_attention_heads = 1;
    t.num_key_value_heads = 1;
    t.layer_types = vec!["conv".into(), "full_attention".into()];
    t.block_multiple_of = 2;
    t.max_position_embeddings = 128;
    c.core.max_length = 128;
    c.core.head_layers = 1;
    c.vision_config.hidden_size = 4;
    c.vision_config.intermediate_size = 8;
    c.vision_config.num_hidden_layers = 1;
    c.vision_config.num_attention_heads = 1;
    c.vision_config.num_patches = 4;
    c.projector_hidden_size = 8;
    c.audio_config = AudioConfig {
        feat_in: 4,
        n_layers: 1,
        d_model: 4,
        subsampling_conv_channels: 2,
        ff_expansion_factor: 2,
        n_heads: 2,
        conv_kernel_size: 3,
        residual_width: 8,
    };
    c.image_text_length = 64;
    c.audio_text_length = 64;
    c
}
fn bounded_key_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    use crate::nn::attention::bounded::MaskedAttentionPlan;
    let mask = mask.expect("fixture callers supply compact key masks");
    assert!(q.dim(2)? <= 16 && k.dim(2)? <= 16);
    let q = q.contiguous()?;
    MaskedAttentionPlan::new(
        &q,
        k,
        v,
        mask,
        scale,
        std::num::NonZeroUsize::new(2).unwrap(),
    )?
    .execute()
}
fn bounded_trunk_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: &TrunkBatch,
    scale: f64,
) -> Result<Tensor> {
    let mut rows = Vec::new();
    for row in 0..batch.dimensions().0 {
        let mut pieces = Vec::new();
        for (i, segment) in batch
            .segments()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.batch_row == row)
        {
            pieces.push(bounded_key_attention(
                &q.narrow(0, row, 1)?
                    .narrow(2, segment.start, segment.length)?,
                &k.narrow(0, row, 1)?,
                &v.narrow(0, row, 1)?,
                Some(&batch.key_mask(i)?),
                scale,
            )?);
        }
        rows.push(Tensor::cat(&pieces, 2)?);
    }
    Tensor::cat(&rows, 0)
}
#[test]
fn assembled_audio_prefix_and_decision_rows_match_explicit_components_cpu() -> Result<()> {
    if std::env::var("CANDLE_NUM_THREADS").as_deref() == Ok("1") {
        let workers = lloom_tensor::utils::barrier_pool().n_workers();
        eprintln!("BOUNDED_BARRIER_WORKERS={workers}");
        assert_eq!(workers, 0);
    }
    let cfg = aggregate_fixture_config();
    let vb = VarBuilder::from_backend(
        Box::new(BoundedWeights {
            elements: std::sync::atomic::AtomicUsize::new(0),
        }),
        DType::F32,
        Device::Cpu,
    )
    .pp("fixture");
    let model = OmniModel::load_components(cfg.clone(), vb.clone())?;
    let audio = AudioEncoder::load(cfg.audio_config.clone(), 64, vb.pp("audio"))?;
    let core = DecisionCore::load(cfg.core, vb)?;
    let samples = (0..8_000)
        .map(|i| ((i % 31) as f32 - 15.) / 64.)
        .collect::<Vec<_>>();
    let frontend = MelFrontend::new(4)?;
    let mel = frontend.prepare_f32(&samples)?;
    assert_eq!(
        (mel.features, mel.computed_frames, mel.valid_frames),
        (4, 51, 50)
    );
    let mel_tensor = Tensor::from_vec(mel.values, (1, 4, 51), &Device::Cpu)?;
    let mut calls = 0;
    let mut attend = |input: RelativeInputs<'_>| {
        calls += 1;
        assert_eq!(input.content_query.dims(), &[1, 2, 7, 2]);
        assert_eq!(input.valid, 7);
        super::super::RelativeAttentionPlan::new(input)?.execute()
    };
    let expected_prefix = audio.forward_with_attention(&mel_tensor, 50, &mut attend)?;
    let prefix = model.audio_prefix_f32(&samples, &mut attend)?;
    assert_eq!(calls, 2);
    assert_eq!(prefix.dims(), &[1, 7, 64]);
    let a = prefix.flatten_all()?.to_vec1::<f32>()?;
    let b = expected_prefix.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), 448);
    assert_eq!(b.len(), 448);
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(
            a.is_finite() && (*a - *b).abs() < 1e-5,
            "aggregate prefix index{i}: {a} != {b}"
        );
    }
    assert!(a.iter().any(|x| x.abs() > 1e-3), "nonzero media control");
    let tokens = [&[1, 2, 3][..], &[4, 5][..]];
    let markers = [&[2, 0][..], &[1][..]];
    let run = |prefix: &Tensor, aggregate: bool| -> Result<Vec<f32>> {
        let rows = [
            DecisionRow {
                tokens: tokens[0],
                markers: markers[0],
                question_type: 0,
                prefix: Some(prefix),
            },
            DecisionRow {
                tokens: tokens[1],
                markers: markers[1],
                question_type: 2,
                prefix: Some(prefix),
            },
        ];
        let prepared = if aggregate {
            model.prepare_rows(&rows)?
        } else {
            core.prepare_rows(&rows)?
        };
        assert_eq!(prepared.dimensions(), (2, 10, 3));
        assert_eq!(prepared.option_counts(), &[2, 1]);
        let result = if aggregate {
            model.forward_with_attention(
                &prepared,
                &mut bounded_trunk_attention,
                &mut bounded_key_attention,
            )?
        } else {
            core.forward_with_attention(
                &prepared,
                &mut bounded_trunk_attention,
                &mut bounded_key_attention,
            )?
        };
        assert_eq!(result.dims(), &[2, 2]);
        result.flatten_all()?.to_vec1::<f32>()
    };
    let expected = run(&expected_prefix, false)?;
    let actual = run(&prefix, true)?;
    assert_eq!(expected.len(), 4);
    assert_eq!(actual.len(), 4);
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.is_finite() && (*a - *b).abs() < 1e-5,
            "aggregate score index{i}: {a} != {b}"
        );
    }
    assert_eq!(actual[3], -1e4);
    let foreign = core.prepare_rows(&[DecisionRow {
        tokens: &[1],
        markers: &[0],
        question_type: 0,
        prefix: None,
    }])?;
    assert!(model
        .forward_with_attention(
            &foreign,
            &mut bounded_trunk_attention,
            &mut bounded_key_attention
        )
        .is_err());
    Ok(())
}
