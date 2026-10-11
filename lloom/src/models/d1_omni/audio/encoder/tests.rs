use super::*;
use crate::models::d1_omni::RelativeAttentionPlan;
use serde_json::Value;
use std::collections::HashMap;

fn fixture() -> Result<(AudioEncoder, Value)> {
    let data: Value = serde_json::from_str(include_str!("scalar_fixture.json")).unwrap();
    let cfg: AudioConfig = serde_json::from_value(data["config"].clone()).unwrap();
    let mut weights = HashMap::new();
    let mut elements = 0;
    for (name, value) in data["weights"].as_object().unwrap() {
        assert!(name.starts_with("fixture.audio."));
        let shape = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let values = floats(&value["data"]);
        elements += values.len();
        weights.insert(
            name.clone(),
            Tensor::from_vec(values, shape.as_slice(), &Device::Cpu)?,
        );
    }
    assert_eq!(weights.len(), 63);
    assert_eq!(elements, 559);
    // Only the pinned fixture.audio namespace exists: load must resolve every
    // learned block through the supplied prefix, not an invented second root.
    let encoder = AudioEncoder::load(
        cfg,
        4,
        VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu).pp("fixture.audio"),
    )?;
    Ok((encoder, data))
}
fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn compare(t: &Tensor, expected: &Value, n: usize, tolerance: f64, label: &str) -> Result<()> {
    let got = t.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.as_array().unwrap();
    assert_eq!(got.len(), n);
    assert_eq!(expected.len(), n);
    for (i, (&a, b)) in got.iter().zip(expected).enumerate() {
        let b = b.as_f64().unwrap();
        assert!(
            (a as f64 - b).abs() < tolerance,
            "audio {label} index{i}: {a} != {b}"
        );
    }
    Ok(())
}

#[test]
fn learned_audio_matches_scalar_padded_stages_and_cropped_prefix() -> Result<()> {
    let (encoder, data) = fixture()?;
    assert_eq!(encoder.positions.dims(), [1, 751, 4]);
    let positions = encoder.positions.flatten_all()?.to_vec1::<f32>()?;
    for row in [0usize, 375, 750] {
        let position = 375f64 - row as f64;
        for pair in 0..2 {
            let phase = position * (-((pair * 2) as f64) * (10000f64).ln() / 4.).exp();
            for (offset, expected) in [phase.sin(), phase.cos()].into_iter().enumerate() {
                let got = positions[row * 4 + pair * 2 + offset] as f64;
                assert!((got-expected).abs()<1e-5,"retained audio position row{row} pair{pair} offset{offset}: {got} != {expected}");
            }
        }
    }
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let time = case["time"].as_u64().unwrap() as usize;
        let valid = case["valid"].as_u64().unwrap() as usize;
        let computed = ceil_half(ceil_half(ceil_half(time)));
        let real = ceil_half(ceil_half(ceil_half(valid)));
        assert_eq!(computed, real + 1);
        let mel = Tensor::from_vec(floats(&case["mel"]), (1, 8, time), &Device::Cpu)?;
        let (sub, n) = encoder.subsample.forward(&mel, valid)?;
        assert_eq!(n, real);
        assert_eq!(sub.dims(), [1, computed, 4]);
        compare(
            &sub,
            &case["subsample"],
            computed * 4,
            3e-6,
            "subsample padded",
        )?;
        let reference = floats(&case["subsample"]);
        let wrong = floats(&case["unmasked_subsample"]);
        assert_eq!(wrong.len(), computed * 4);
        assert!(
            reference
                .iter()
                .zip(wrong)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "padded subsample mask discriminator vacuous"
        );
        assert!(
            reference[real * 4..].iter().any(|x| x.abs() > 1e-3),
            "computed padding must retain output bias"
        );
        // Full encoder intermediate is compared to an independent scalar
        // oracle, including padded rows; this is not new-vs-new equivalence.
        let pos = encoder
            .positions
            .narrow(1, MAX_ENCODED_FRAMES - computed, 2 * computed - 1)?;
        let mask = Tensor::from_vec(
            (0..computed)
                .map(|i| u8::from(i < real))
                .collect::<Vec<_>>(),
            (1, 1, computed),
            &Device::Cpu,
        )?;
        let zero = Tensor::zeros((1, 1, computed), DType::F32, &Device::Cpu)?;
        let mut hidden = sub;
        for layer in &encoder.layers {
            hidden = layer.forward(&hidden, &pos, real, &mask, &zero, &mut |input| {
                RelativeAttentionPlan::new(input)?.execute()
            })?;
        }
        compare(
            &hidden,
            &case["encoded_full"],
            computed * 4,
            2e-5,
            "full Conformer",
        )?;
        let mut prior = None;
        for _ in 0..3 {
            let mut calls = 0;
            let got = encoder.forward_with_attention(&mel, valid, &mut |input| {
                calls += 1;
                assert_eq!(input.content_query.dims(), [1, 2, computed, 2]);
                assert_eq!(input.position_query.dims(), [1, 2, computed, 2]);
                assert_eq!(input.positions.dims(), [1, 2, 2 * computed - 1, 2]);
                assert_eq!(input.valid, real); // both query and key masking
                RelativeAttentionPlan::new(input)?.execute()
            })?;
            assert_eq!(calls, 1);
            assert_eq!(got.dims(), [1, real, 4]);
            compare(
                &got,
                &case["output"],
                real * 4,
                2e-5,
                "cropped adapter residual",
            )?;
            let bits = got
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>();
            if let Some(ref last) = prior {
                assert_eq!(&bits, last);
            }
            prior = Some(bits);
        }
    }
    Ok(())
}

#[test]
fn learned_audio_rejects_malformed_metadata_and_callback() -> Result<()> {
    let (encoder, data) = fixture()?;
    let cfg: AudioConfig = serde_json::from_value(data["config"].clone()).unwrap();
    let empty = || VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
    for field in [0, 1, 2, 3] {
        let mut bad = cfg.clone();
        match field {
            0 => bad.n_heads = 0,
            1 => bad.d_model = 3,
            2 => bad.conv_kernel_size = 2,
            _ => bad.n_layers = 0,
        }
        assert!(AudioEncoder::load(bad, 4, empty()).is_err());
    }
    assert!(AudioEncoder::load(cfg, 0, empty()).is_err());
    let mel = Tensor::zeros((1, 8, 9), DType::F32, &Device::Cpu)?;
    for valid in [0usize, 10] {
        assert!(encoder
            .forward_with_attention(&mel, valid, &mut |_| panic!(
                "metadata refusal precedes callback"
            ))
            .is_err());
    }
    let wrong = Tensor::zeros((1, 7, 9), DType::F32, &Device::Cpu)?;
    assert!(encoder
        .forward_with_attention(&wrong, 8, &mut |_| panic!(
            "shape refusal precedes callback"
        ))
        .is_err());
    assert!(encoder
        .forward_with_attention(&mel, 8, &mut |_| Tensor::zeros(
            (1, 1, 1, 1),
            DType::F32,
            &Device::Cpu
        ))
        .is_err());
    Ok(())
}
