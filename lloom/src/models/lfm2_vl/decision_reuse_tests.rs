//! Tiny compatibility capture first; independent new-contract tests added next.
use super::*;
use std::collections::HashMap;

fn fixture() -> Result<(VisionTower, Tensor)> {
    let cfg = VisionConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_channels: 3,
        patch_size: 1,
        num_patches: 4,
        layer_norm_eps: 1e-6,
        hidden_act: crate::nn::Activation::GeluPytorchTanh,
    };
    let mut weights = HashMap::new();
    let mut insert = |name: &str, dims: &[usize], seed: usize, norm: bool| -> Result<()> {
        let data = (0..dims.iter().product::<usize>())
            .map(|i| {
                let value = (((i * 3 + seed) % 11) as f32 - 5.) / 32.;
                if norm {
                    1. + value
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        weights.insert(
            format!("vision_model.{name}"),
            Tensor::from_vec(data, dims, &Device::Cpu)?,
        );
        Ok(())
    };
    insert("embeddings.patch_embedding.weight", &[4, 3], 1, false)?;
    insert("embeddings.patch_embedding.bias", &[4], 2, false)?;
    insert("embeddings.position_embedding.weight", &[4, 4], 3, false)?;
    for (i, p) in ["q_proj", "k_proj", "v_proj", "out_proj"]
        .iter()
        .enumerate()
    {
        insert(
            &format!("encoder.layers.0.self_attn.{p}.weight"),
            &[4, 4],
            i + 4,
            false,
        )?;
        insert(
            &format!("encoder.layers.0.self_attn.{p}.bias"),
            &[4],
            i + 5,
            false,
        )?;
    }
    insert("encoder.layers.0.mlp.fc1.weight", &[8, 4], 8, false)?;
    insert("encoder.layers.0.mlp.fc1.bias", &[8], 9, false)?;
    insert("encoder.layers.0.mlp.fc2.weight", &[4, 8], 10, false)?;
    insert("encoder.layers.0.mlp.fc2.bias", &[4], 11, false)?;
    for (i, p) in [
        "encoder.layers.0.layer_norm1",
        "encoder.layers.0.layer_norm2",
        "post_layernorm",
    ]
    .iter()
    .enumerate()
    {
        insert(&format!("{p}.weight"), &[4], i + 12, true)?;
        insert(&format!("{p}.bias"), &[4], i + 13, false)?;
    }
    let tower = VisionTower::new(
        &cfg,
        VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu),
    )?;
    let patches = Tensor::from_vec(
        (0..12).map(|i| (i as f32 - 4.) / 8.).collect::<Vec<_>>(),
        (1, 4, 3),
        &Device::Cpu,
    )?;
    Ok((tower, patches))
}
fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
fn legacy_vision_entry_compatibility_capture() -> Result<()> {
    let (tower, patches) = fixture()?;
    // Capture both direct position-table and actual resampling branches BEFORE
    // factoring production code. Captured outputs prove compatibility, not an
    // independent mathematical/Python reference.
    for (rows, cols) in [(2, 2), (1, 4)] {
        let embeddings = bits(&tower.embeddings_only(&patches, rows, cols)?)?;
        let output = bits(&tower.forward(&patches, rows, cols)?)?;
        println!("LEGACY_EMBED_{rows}x{cols}={embeddings:?}");
        println!("LEGACY_TOWER_{rows}x{cols}={output:?}");
        for _ in 0..2 {
            assert_eq!(
                bits(&tower.embeddings_only(&patches, rows, cols)?)?,
                embeddings
            );
            assert_eq!(bits(&tower.forward(&patches, rows, cols)?)?, output);
        }
    }
    let processor = ImageProcessor {
        patch_size: 1,
        ..ImageProcessor::lfm2_vl_3b()
    };
    let rgb = (0..=255u8).flat_map(|v| [v, v, v]).collect::<Vec<_>>();
    let got = processor.patchify_rgb8(&rgb, 1, 256, &Device::Cpu, DType::F32)?;
    let expected = rgb
        .iter()
        .map(|&v| ((v as f32 / 255. - 0.5) / 0.5).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits(&got.patches)?, expected);
    println!("LEGACY_UNIT_RANGE_BITS={expected:?}");
    Ok(())
}

#[test]
fn repeat_first_padding_and_attention_callback_preserve_explicit_contract() -> Result<()> {
    use crate::nn::attention::bounded::MaskedAttentionPlan;
    use std::num::NonZeroUsize;
    let (tower, real) = fixture()?;
    let patches = Tensor::cat(
        &[real, Tensor::zeros((1, 2, 3), DType::F32, &Device::Cpu)?],
        1,
    )?;
    let projected = tower
        .patch_embed_only(&patches)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let embedded =
        tower.embeddings_with_padding(&patches, 2, 2, VisionPositionPadding::RepeatFirst)?;
    let actual = embedded.flatten_all()?.to_vec1::<f32>()?;
    for row in 0..6 {
        for channel in 0..4 {
            // Independent index oracle: positions0..3, then repeat row0 twice.
            let position_row = if row < 4 { row } else { 0 };
            let pos = (((position_row * 4 + channel) * 3 + 3) % 11) as f32 / 32. - 5. / 32.;
            assert_eq!(
                actual[row * 4 + channel].to_bits(),
                (projected[row * 4 + channel] + pos).to_bits()
            );
        }
    }
    let omitted = tower
        .embeddings_only(&patches, 2, 2)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        actual[16..]
            .iter()
            .zip(&omitted[16..])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "padding-policy discriminator vacuous"
    );
    let mask = Tensor::from_vec(
        vec![0f32, 0., 0., 0., f32::NEG_INFINITY, f32::NEG_INFINITY],
        (1, 1, 1, 6),
        &Device::Cpu,
    )?;
    let dense = tower
        .inner
        .encoder
        .forward(&embedded, Some(&mask.broadcast_as((1, 1, 6, 6))?))?
        .apply(&tower.inner.post_layernorm)?;
    let mut calls = 0;
    let mut previous = None;
    for _ in 0..3 {
        let output = tower.forward_with_attention(
            &patches,
            2,
            2,
            VisionPositionPadding::RepeatFirst,
            Some(&mask),
            &mut |q, k, v, seen, scale| {
                calls += 1;
                assert_eq!(q.dims(), [1, 1, 6, 4]);
                assert_eq!(scale, 0.5);
                assert_eq!(bits(seen.unwrap())?, bits(&mask)?);
                MaskedAttentionPlan::new(
                    q,
                    k,
                    v,
                    seen.unwrap(),
                    scale,
                    NonZeroUsize::new(2).unwrap(),
                )?
                .execute()
            },
        )?;
        for (a, b) in output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(dense.flatten_all()?.to_vec1::<f32>()?)
        {
            assert!((a - b).abs() < 2e-6, "callback vs dense {a} != {b}");
        }
        let result = bits(&output)?;
        if let Some(previous) = &previous {
            assert_eq!(&result, previous);
        }
        previous = Some(result);
    }
    assert_eq!(calls, 3);
    for (rows, cols) in [(0, 2), (2, 0), (3, 3), (usize::MAX, 2)] {
        assert!(tower
            .embeddings_with_padding(&patches, rows, cols, VisionPositionPadding::RepeatFirst)
            .is_err());
    }
    let wrong = Tensor::zeros((1, 4, 2), DType::F32, &Device::Cpu)?;
    assert!(tower
        .embeddings_with_padding(&wrong, 2, 2, VisionPositionPadding::RepeatFirst)
        .is_err());
    let huge = Tensor::new(0f32, &Device::Cpu)?.broadcast_as((usize::MAX / 4, 6, 3))?;
    assert!(tower
        .embeddings_with_padding(&huge, 2, 2, VisionPositionPadding::RepeatFirst)
        .is_err());
    Ok(())
}

#[test]
fn byte_range_matches_reference_arithmetic_and_invalid_patch_geometry_is_rejected() -> Result<()> {
    let processor = ImageProcessor {
        patch_size: 1,
        ..ImageProcessor::lfm2_vl_3b()
    };
    let rgb = (0..=255u8).flat_map(|v| [v, v, v]).collect::<Vec<_>>();
    let normalization = RgbNormalization::ByteRange {
        mean: [127.5; 3],
        std: [127.5; 3],
    };
    let output = processor.patchify_rgb8_with_normalization(
        &rgb,
        1,
        256,
        &Device::Cpu,
        DType::F32,
        normalization,
    )?;
    let expected = rgb
        .iter()
        .map(|&v| ((v as f32 - 127.5) / 127.5).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits(&output.patches)?, expected);
    let old = bits(
        &processor
            .patchify_rgb8(&rgb, 1, 256, &Device::Cpu, DType::F32)?
            .patches,
    )?;
    assert!(
        old.iter().zip(&expected).any(|(a, b)| a != b),
        "operation-order discriminator vacuous"
    );
    // P=2 distinguishes P,P,C from channels-first independently of normalization.
    let p2 = ImageProcessor {
        patch_size: 2,
        ..processor.clone()
    };
    let rgb = (0..24).map(|i| i as u8).collect::<Vec<_>>();
    let got =
        p2.patchify_rgb8_with_normalization(&rgb, 1, 2, &Device::Cpu, DType::F32, normalization)?;
    let order = [
        0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
    ];
    let want = order.map(|i| ((rgb[i] as f32 - 127.5) / 127.5).to_bits());
    assert_eq!(bits(&got.patches)?, want);
    for (rows, cols) in [(0, 1), (1, 0), (usize::MAX, 2)] {
        assert!(processor
            .patchify_rgb8_with_normalization(
                &[],
                rows,
                cols,
                &Device::Cpu,
                DType::F32,
                normalization
            )
            .is_err());
    }
    for bad in [
        ImageProcessor {
            patch_size: 0,
            ..processor.clone()
        },
        ImageProcessor {
            num_channels: 4,
            ..processor.clone()
        },
    ] {
        assert!(bad
            .patchify_rgb8_with_normalization(&[], 1, 1, &Device::Cpu, DType::F32, normalization)
            .is_err());
    }
    for std in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(processor
            .patchify_rgb8_with_normalization(
                &[0, 0, 0],
                1,
                1,
                &Device::Cpu,
                DType::F32,
                RgbNormalization::ByteRange {
                    mean: [127.5; 3],
                    std: [std; 3]
                }
            )
            .is_err());
    }
    Ok(())
}
