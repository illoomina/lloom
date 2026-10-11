use super::*;
use lloom_tensor::Device;
use std::collections::HashMap;

const WIDTH: usize = 64;

fn weights() -> Result<HashMap<String, Tensor>> {
    let mut values = HashMap::new();
    let mut type_embedding = vec![0f32; 3 * WIDTH];
    for kind in 0..3 {
        type_embedding[kind * WIDTH] = kind as f32 * 0.002;
        type_embedding[kind * WIDTH + 1] = -(kind as f32) * 0.002;
    }
    values.insert(
        "type_emb.weight".into(),
        Tensor::from_vec(type_embedding, (3, WIDTH), &Device::Cpu)?,
    );
    for (name, output, input) in [
        ("head.layers.0.self_attn.in_proj", 3 * WIDTH, WIDTH),
        ("head.layers.0.self_attn.out_proj", WIDTH, WIDTH),
        ("head.layers.0.linear1", 4 * WIDTH, WIDTH),
        ("head.layers.0.linear2", WIDTH, 4 * WIDTH),
        ("scorer.1", WIDTH, WIDTH),
        ("scorer.3", 1, WIDTH),
    ] {
        let mut weight = vec![0f32; output * input];
        let mut bias = vec![0f32; output];
        if name == "scorer.1" {
            for i in 0..WIDTH {
                weight[i * WIDTH + i] = 1.;
            }
        } else if name == "scorer.3" {
            weight[0] = 1.;
            weight[1] = -1.;
            bias[0] = 0.25;
        } else if name.ends_with("in_proj") {
            for (i, b) in bias.iter_mut().enumerate() {
                *b = (i / WIDTH + 1) as f32 * 0.125;
            }
        }
        let separator = if name.ends_with("in_proj") { "_" } else { "." };
        values.insert(
            format!("{name}{separator}weight"),
            Tensor::from_vec(weight, (output, input), &Device::Cpu)?,
        );
        values.insert(
            format!("{name}{separator}bias"),
            Tensor::from_vec(bias, output, &Device::Cpu)?,
        );
    }
    for name in ["head.layers.0.norm1", "head.layers.0.norm2", "scorer.0"] {
        values.insert(
            format!("{name}.weight"),
            Tensor::ones(WIDTH, DType::F32, &Device::Cpu)?,
        );
        values.insert(
            format!("{name}.bias"),
            Tensor::zeros(WIDTH, DType::F32, &Device::Cpu)?,
        );
    }
    Ok(values)
}

fn head() -> Result<DecisionHead> {
    DecisionHead::load(
        WIDTH,
        1,
        VarBuilder::from_tensors(weights()?, DType::F32, &Device::Cpu),
    )
}

fn hidden() -> Result<Tensor> {
    let mut data = vec![0f32; 2 * 5 * WIDTH];
    for row in 0..10 {
        let x = (row as f32 + 1.) * 0.003;
        data[row * WIDTH] = x;
        data[row * WIDTH + 1] = -x;
    }
    Tensor::from_vec(data, (2, 5, WIDTH), &Device::Cpu)
}

#[test]
fn checkpoint_head_matches_absolute_scorer_reference_and_compact_mask_contract() -> Result<()> {
    let head = head()?;
    let markers: [&[usize]; 2] = [&[4, 0, 2], &[1, 0]];
    let batch = head.prepare_batch(5, &[5, 2], &markers, &[0, 2])?;
    assert_eq!(batch.key_mask.elem_count(), 10);
    assert_eq!(batch.marker_rows.elem_count(), 6);
    assert_eq!(batch.marker_rows.to_vec1::<u32>()?, [4, 0, 2, 6, 5, 5]);
    let hidden = hidden()?;
    let mut previous = None;
    for _ in 0..3 {
        let mut calls = 0;
        let logits =
            head.forward_with_attention(&hidden, &batch, &mut |q, k, v, mask, scale| {
                calls += 1;
                assert_eq!(q.dims(), [2, 1, 5, WIDTH]);
                assert_eq!(scale, 0.125);
                for (tensor, expected) in [(q, 0.125f32), (k, 0.25), (v, 0.375)] {
                    assert!(tensor
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|&x| x == expected));
                }
                let mask = mask.expect("key padding must reach attention");
                assert_eq!(mask.dims(), [2, 1, 1, 5]);
                assert_eq!(
                    mask.flatten_all()?.to_vec1::<f32>()?,
                    [
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY
                    ]
                );
                // Dense only in this fixed tiny oracle, never selected by the loader.
                let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(mask)?;
                crate::nn::ops::softmax_last_dim(&scores)?.matmul(v)
            })?;
        assert_eq!(calls, 1);
        assert_eq!(logits.dtype(), DType::F32);
        let actual = logits.to_vec2::<f32>()?;
        for row in 0..2 {
            for (option, &position) in markers[row].iter().enumerate() {
                // Zero output projections leave h+type_emb. Unit LayerNorm and
                // GELU(z)-GELU(-z)=z make an independent absolute scalar oracle.
                let x =
                    ((row * 5 + position) as f32 + 1.) * 0.003 + if row == 0 { 0. } else { 0.004 };
                let x = x as f64;
                let expected = x / (x * x / 32. + 1e-5).sqrt() + 0.25;
                assert!(
                    (actual[row][option] as f64 - expected).abs() < 2e-5,
                    "row={row} option={option} got={} expected={expected}",
                    actual[row][option]
                );
            }
        }
        assert_eq!(actual[1][2], -1e4);
        let bits: Vec<_> = actual.iter().flatten().map(|x| x.to_bits()).collect();
        if let Some(prior) = &previous {
            assert_eq!(&bits, prior);
        }
        previous = Some(bits);
    }
    Ok(())
}

#[test]
fn metadata_errors_are_rejected_before_device_indexing() -> Result<()> {
    let head = head()?;
    let marker: [&[usize]; 1] = [&[0]];
    for (length, lengths, types) in [
        (0, vec![1], vec![0]),
        (1, vec![], vec![]),
        (2, vec![0], vec![0]),
        (2, vec![3], vec![0]),
        (2, vec![1], vec![3]),
        (2, vec![1], vec![]),
    ] {
        assert!(head
            .prepare_batch(length, &lengths, &marker, &types)
            .is_err());
    }
    assert!(head.prepare_batch(2, &[1], &[&[]], &[0]).is_err());
    assert!(head.prepare_batch(2, &[1], &[&[1]], &[0]).is_err());
    assert!(head.prepare_batch(2, &[1], &[], &[0]).is_err());
    assert!(head
        .prepare_batch(u32::MAX as usize + 1, &[1], &marker, &[0])
        .is_err());
    let batch = head.prepare_batch(5, &[5, 2], &[&[0], &[1]], &[1, 2])?;
    let hidden = hidden()?;
    for invalid in [
        hidden.to_dtype(DType::F64)?,
        hidden.to_dtype(DType::U32)?,
        hidden.reshape((10, WIDTH))?,
        hidden.narrow(2, 0, WIDTH - 1)?,
    ] {
        let mut calls = 0;
        assert!(head
            .forward_with_attention(&invalid, &batch, &mut |q, _, _, _, _| {
                calls += 1;
                Ok(q.clone())
            })
            .is_err());
        assert_eq!(calls, 0);
    }
    Ok(())
}

#[test]
fn loader_checks_geometry_dtype_and_packed_weight_shape() -> Result<()> {
    for width in [0, 1, 63, 129, usize::MAX] {
        let builder = VarBuilder::from_tensors(weights()?, DType::F32, &Device::Cpu);
        assert!(DecisionHead::load(width, 1, builder).is_err());
    }
    let builder = VarBuilder::from_tensors(weights()?, DType::U32, &Device::Cpu);
    assert!(DecisionHead::load(WIDTH, 1, builder).is_err());
    let mut tensors = weights()?;
    tensors.insert(
        "head.layers.0.self_attn.in_proj_weight".into(),
        Tensor::zeros((3 * WIDTH - 1, WIDTH), DType::F32, &Device::Cpu)?,
    );
    assert!(DecisionHead::load(
        WIDTH,
        1,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    )
    .is_err());
    let mut tensors = weights()?;
    tensors.remove("head.layers.0.self_attn.in_proj_bias");
    assert!(DecisionHead::load(
        WIDTH,
        1,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    )
    .is_err());
    Ok(())
}

#[derive(serde::Deserialize)]
struct ScalarFixture {
    hidden: Vec<Vec<Vec<f32>>>,
    lengths: Vec<usize>,
    types: Vec<u32>,
    markers: Vec<Vec<usize>>,
    expected: Vec<Vec<f64>>,
    ignored_mask: Vec<Vec<f64>>,
    wrong_causal: Vec<Vec<f64>>,
    exact_gelu_one: f64,
    tanh_gelu_one: f64,
}

fn scalar_fixture() -> ScalarFixture {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/d1_omni/head_scalar.json"
    )))
    .unwrap()
}

#[test]
fn asymmetric_scorer_pins_exact_erf_gelu_not_tanh() -> Result<()> {
    let fixture = scalar_fixture();
    assert!((fixture.exact_gelu_one - fixture.tanh_gelu_one).abs() > 1e-4);
    let mut tensors = weights()?;
    tensors.insert(
        "scorer.1.weight".into(),
        Tensor::zeros((WIDTH, WIDTH), DType::F32, &Device::Cpu)?,
    );
    let mut bias = vec![0f32; WIDTH];
    bias[0] = 1.;
    tensors.insert(
        "scorer.1.bias".into(),
        Tensor::from_vec(bias, WIDTH, &Device::Cpu)?,
    );
    let head = DecisionHead::load(
        WIDTH,
        1,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
    )?;
    let batch = head.prepare_batch(1, &[1], &[&[0]], &[0])?;
    let hidden = Tensor::zeros((1, 1, WIDTH), DType::F32, &Device::Cpu)?;
    let output = head
        .forward_with_attention(&hidden, &batch, &mut |_, _, v, _, _| Ok(v.clone()))?
        .to_vec2::<f32>()?[0][0] as f64;
    assert!(
        (output - (fixture.exact_gelu_one + 0.25)).abs() < 2e-6,
        "exact GELU scorer got={output} expected={}",
        fixture.exact_gelu_one + 0.25
    );
    assert!((output - (fixture.tanh_gelu_one + 0.25)).abs() > 1e-4);
    Ok(())
}

#[test]
fn nonzero_transformer_matches_scalar_reference_and_distinguishes_padding_and_causality(
) -> Result<()> {
    let fixture = scalar_fixture();
    assert!((fixture.expected[1][0] - fixture.ignored_mask[1][0]).abs() > 1.);
    assert!((fixture.expected[0][1] - fixture.wrong_causal[0][1]).abs() > 0.01);
    let mut tensors = weights()?;
    let mut packed = vec![0f32; 3 * WIDTH * WIDTH];
    for (part, scale) in [0.125, 0.25, 0.5].into_iter().enumerate() {
        for i in 0..WIDTH {
            packed[(part * WIDTH + i) * WIDTH + i] = scale;
        }
    }
    tensors.insert(
        "head.layers.0.self_attn.in_proj_weight".into(),
        Tensor::from_vec(packed, (3 * WIDTH, WIDTH), &Device::Cpu)?,
    );
    tensors.insert(
        "head.layers.0.self_attn.in_proj_bias".into(),
        Tensor::zeros(3 * WIDTH, DType::F32, &Device::Cpu)?,
    );
    for (name, rows, cols, scale) in [
        ("head.layers.0.self_attn.out_proj", WIDTH, WIDTH, 0.125f32),
        ("head.layers.0.linear1", 4 * WIDTH, WIDTH, 0.5),
        ("head.layers.0.linear2", WIDTH, 4 * WIDTH, 0.25),
    ] {
        let mut data = vec![0f32; rows * cols];
        for i in 0..WIDTH {
            data[i * cols + i] = scale;
        }
        tensors.insert(
            format!("{name}.weight"),
            Tensor::from_vec(data, (rows, cols), &Device::Cpu)?,
        );
    }
    let mut bias = vec![0f32; 4 * WIDTH];
    for (i, value) in bias[..WIDTH].iter_mut().enumerate() {
        *value = if i % 2 == 0 { 0.02 } else { -0.02 };
    }
    tensors.insert(
        "head.layers.0.linear1.bias".into(),
        Tensor::from_vec(bias, 4 * WIDTH, &Device::Cpu)?,
    );
    let head = DecisionHead::load(
        WIDTH,
        1,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
    )?;
    let marker_refs: Vec<_> = fixture.markers.iter().map(Vec::as_slice).collect();
    let batch = head.prepare_batch(3, &fixture.lengths, &marker_refs, &fixture.types)?;
    let hidden = Tensor::from_vec(
        fixture
            .hidden
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<_>>(),
        (2, 3, WIDTH),
        &Device::Cpu,
    )?;
    let mut calls = 0;
    let output = head
        .forward_with_attention(&hidden, &batch, &mut |q, k, v, mask, scale| {
            calls += 1;
            let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(mask.unwrap())?;
            crate::nn::ops::softmax_last_dim(&scores)?.matmul(v)
        })?
        .to_vec2::<f32>()?;
    assert_eq!(calls, 1);
    for (row, expected) in fixture.expected.iter().enumerate() {
        for (option, &expected) in expected.iter().enumerate() {
            assert!(
                (output[row][option] as f64 - expected).abs() < 3e-5,
                "nonzero transformer row={row} option={option} got={} expected={expected}",
                output[row][option]
            );
        }
    }
    assert_eq!(output[1][1], -1e4);
    Ok(())
}
