use super::*;
use crate::nn::attention::bounded::MaskedAttentionPlan;
use std::collections::HashMap;
use std::num::NonZeroUsize;

fn fixture() -> Result<Vision> {
    let cfg = VisionConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_channels: 3,
        patch_size: 16,
        num_patches: 4,
        layer_norm_eps: 1e-6,
        hidden_act: Activation::GeluPytorchTanh,
    };
    let mut weights = HashMap::new();
    let mut put = |name: &str, dims: &[usize], seed: usize, norm: bool| -> Result<()> {
        let v = (0..dims.iter().product::<usize>())
            .map(|i| {
                let v = (((i * 3 + seed) % 11) as f32 - 5.) / 32.;
                if norm {
                    v + 1.
                } else {
                    v
                }
            })
            .collect::<Vec<_>>();
        weights.insert(
            format!("wrapper.vision.{name}"),
            Tensor::from_vec(v, dims, &Device::Cpu)?,
        );
        Ok(())
    };
    put(
        "tower.embeddings.patch_embedding.weight",
        &[4, 768],
        1,
        false,
    )?;
    put("tower.embeddings.patch_embedding.bias", &[4], 2, false)?;
    put(
        "tower.embeddings.position_embedding.weight",
        &[4, 4],
        3,
        false,
    )?;
    for (i, p) in ["q_proj", "k_proj", "v_proj", "out_proj"]
        .iter()
        .enumerate()
    {
        put(
            &format!("tower.encoder.layers.0.self_attn.{p}.weight"),
            &[4, 4],
            i + 4,
            false,
        )?;
        put(
            &format!("tower.encoder.layers.0.self_attn.{p}.bias"),
            &[4],
            i + 5,
            false,
        )?;
    }
    put("tower.encoder.layers.0.mlp.fc1.weight", &[8, 4], 8, false)?;
    put("tower.encoder.layers.0.mlp.fc1.bias", &[8], 9, false)?;
    put("tower.encoder.layers.0.mlp.fc2.weight", &[4, 8], 10, false)?;
    put("tower.encoder.layers.0.mlp.fc2.bias", &[4], 11, false)?;
    for (i, p) in [
        "tower.encoder.layers.0.layer_norm1",
        "tower.encoder.layers.0.layer_norm2",
        "tower.post_layernorm",
    ]
    .iter()
    .enumerate()
    {
        put(&format!("{p}.weight"), &[4], i + 12, true)?;
        put(&format!("{p}.bias"), &[4], i + 13, false)?;
    }
    put("projector.linear_1.weight", &[8, 16], 29, false)?;
    put("projector.linear_1.bias", &[8], 30, false)?;
    put("projector.linear_2.weight", &[4, 8], 31, false)?;
    put("projector.linear_2.bias", &[4], 32, false)?;
    assert_eq!(weights.len(), 25);
    // No vision_model spelling exists in this source map. Incorrect stripping
    // or double-applying wrapper.vision fails in the actual loader.
    Vision::load(
        &cfg,
        8,
        4,
        VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu).pp("wrapper.vision"),
    )
}
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, m: Option<&Tensor>, scale: f64) -> Result<Tensor> {
    MaskedAttentionPlan::new(q, k, v, m.unwrap(), scale, NonZeroUsize::new(2).unwrap())?.execute()
}
fn flat(x: &Tensor) -> Result<Vec<f32>> {
    x.flatten_all()?.to_vec1::<f32>()
}
fn tile(seed: usize) -> Result<VisionTile> {
    // Synthetic component fixture ONLY: six padded rows, not the production
    // image processor's1024. This isolates loader/tower/projector/order without
    // executing prepare_rgb8 or allocating an image/1024-row patch array.
    let patches = Tensor::from_vec(
        (0..6 * 768)
            .map(|i| (((i * 7 + seed) % 19) as f32 - 9.) / 16.)
            .collect::<Vec<_>>(),
        (1, 6, 768),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![0f32, 0., 0., 0., f32::NEG_INFINITY, f32::NEG_INFINITY],
        (1, 1, 1, 6),
        &Device::Cpu,
    )?;
    Ok(VisionTile {
        rows: 2,
        cols: 2,
        patches,
        mask,
    })
}
#[test]
fn loaded_namespace_and_ordered_prefix_composition_cpu() -> Result<()> {
    let vision = fixture()?;
    let first = PreparedVisionImage {
        owner: vision.owner.clone(),
        tiles: vec![tile(1)?, tile(5)?],
        prefix_tokens: 2,
    };
    let second = PreparedVisionImage {
        owner: vision.owner.clone(),
        tiles: vec![tile(9)?],
        prefix_tokens: 1,
    };
    let mut expected = Vec::new();
    for t in first.tiles.iter().chain(&second.tiles) {
        // Bypass Vision.forward iteration/cat with explicit per-tile calls.
        // Projector arithmetic has a separate independent scalar fixture below;
        // this control proves ordering/composition, not independent tower math.
        let embeddings = vision.tower.embeddings_with_padding(
            &t.patches,
            2,
            2,
            VisionPositionPadding::RepeatFirst,
        )?;
        let output = vision.tower.forward_with_attention(
            &t.patches,
            2,
            2,
            VisionPositionPadding::RepeatFirst,
            Some(&t.mask),
            &mut attend,
        )?;
        assert_eq!(embeddings.dims(), [1, 6, 4]);
        expected.extend(flat(
            &vision
                .projector
                .forward(&output.narrow(1, 0, 4)?.reshape((1, 2, 2, 4))?)?,
        )?);
    }
    assert!(
        expected[..4]
            .iter()
            .zip(&expected[8..12])
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "tile-order discriminator vacuous"
    );
    let mut prior = None;
    for _ in 0..3 {
        let got = vision.forward_with_attention(&[&first, &second], &mut attend)?;
        assert_eq!(got.dims(), [1, 3, 4]);
        let values = flat(&got)?;
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        if let Some(prior) = &prior {
            assert_eq!(&values, prior);
        }
        prior = Some(values);
    }
    let foreign = PreparedVisionImage {
        owner: Arc::new(()),
        tiles: vec![tile(1)?],
        prefix_tokens: 1,
    };
    assert!(vision
        .forward_with_attention(&[&foreign], &mut attend)
        .is_err());
    assert!(vision.forward_with_attention(&[], &mut attend).is_err());
    Ok(())
}
#[test]
fn projector_matches_independent_scalar_nontrivial_unshuffle_cpu() -> Result<()> {
    let vision = fixture()?;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("projector_fixture.json")).unwrap();
    let input = fixture["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(input, (1, 2, 4, 4), &Device::Cpu)?;
    let expected = fixture["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 8);
    let output = vision.projector.forward(&input)?;
    assert_eq!(output.dims(), [2, 4]);
    for (i, (a, b)) in flat(&output)?.iter().zip(expected).enumerate() {
        assert!(
            (*a as f64 - b).abs() < 2e-6,
            "projector scalar index{i}: {a} != {b}"
        );
    }
    let swapped = input.transpose(1, 2)?.contiguous()?;
    let wrong = vision.projector.forward(&swapped)?;
    assert!(
        flat(&output)?
            .iter()
            .zip(flat(&wrong)?)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "unshuffle-axis discriminator vacuous"
    );
    Ok(())
}
