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
