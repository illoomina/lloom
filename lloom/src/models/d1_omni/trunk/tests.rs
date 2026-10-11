use super::*;
use lloom_tensor::Device;
use std::collections::HashMap;

fn config() -> TextConfig {
    TextConfig {
        vocab_size: 8,
        hidden_size: 4,
        intermediate_size: 6,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        layer_types: vec!["conv".into(), "full_attention".into()],
        norm_eps: 1e-5,
        conv_l_cache: 3,
        block_ffn_dim_multiplier: 1.,
        block_multiple_of: 2,
        max_position_embeddings: 32,
        rope_theta: 1e6,
    }
}
fn matrix(out: usize, input: usize, seed: usize) -> Result<Tensor> {
    Tensor::from_vec(
        (0..out * input)
            .map(|i| ((i * 7 + seed * 3) % 17) as f32 / 32. - 0.25)
            .collect::<Vec<_>>(),
        (out, input),
        &Device::Cpu,
    )
}
fn trunk() -> Result<Trunk> {
    let mut w = HashMap::new();
    w.insert("embed_tokens.weight".into(), matrix(8, 4, 0)?);
    for (name, out, input, seed) in [
        ("layers.0.conv.in_proj", 12, 4, 1),
        ("layers.0.conv.out_proj", 4, 4, 3),
        ("layers.0.feed_forward.w1", 4, 4, 4),
        ("layers.0.feed_forward.w3", 4, 4, 5),
        ("layers.0.feed_forward.w2", 4, 4, 6),
        ("layers.1.self_attn.q_proj", 4, 4, 7),
        ("layers.1.self_attn.k_proj", 2, 4, 8),
        ("layers.1.self_attn.v_proj", 2, 4, 9),
        ("layers.1.self_attn.out_proj", 4, 4, 10),
        ("layers.1.feed_forward.w1", 4, 4, 11),
        ("layers.1.feed_forward.w3", 4, 4, 12),
        ("layers.1.feed_forward.w2", 4, 4, 13),
    ] {
        w.insert(format!("{name}.weight"), matrix(out, input, seed)?);
    }
    w.insert(
        "layers.0.conv.conv.weight".into(),
        matrix(4, 3, 2)?.reshape((4, 1, 3))?,
    );
    for name in [
        "layers.0.operator_norm",
        "layers.0.ffn_norm",
        "layers.1.operator_norm",
        "layers.1.ffn_norm",
        "embedding_norm",
    ] {
        w.insert(
            format!("{name}.weight"),
            Tensor::ones(4, DType::F32, &Device::Cpu)?,
        );
    }
    for name in [
        "layers.1.self_attn.q_layernorm",
        "layers.1.self_attn.k_layernorm",
    ] {
        w.insert(
            format!("{name}.weight"),
            Tensor::ones(2, DType::F32, &Device::Cpu)?,
        );
    }
    Trunk::load(
        config(),
        VarBuilder::from_tensors(w, DType::F32, &Device::Cpu),
    )
}
fn inputs() -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        Tensor::from_vec(
            (0..32)
                .map(|i| ((i * 5) % 19) as f32 / 8. - 9. / 8.)
                .collect::<Vec<_>>(),
            (2, 4, 4),
            &Device::Cpu,
        )?,
        Tensor::new(&[[1f32], [0.6], [0.], [-0.8]], &Device::Cpu)?,
        Tensor::new(&[[0f32], [0.8], [1.], [0.6]], &Device::Cpu)?,
    ))
}
// Tiny test oracle callback only: no production dense fallback. Consume each
// compact segment separately, retaining all keys, including finite-masked ones.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, batch: &TrunkBatch, scale: f64) -> Result<Tensor> {
    assert_eq!(q.dims(), [2, 2, 4, 2]);
    assert_eq!(k.dims(), q.dims());
    assert_eq!(v.dims(), q.dims());
    assert_eq!(scale, 2f64.powf(-0.5));
    let mut rows = Vec::new();
    for row in 0..2 {
        let mut parts = Vec::new();
        for (index, segment) in batch
            .segments()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.batch_row == row)
        {
            let query = q
                .narrow(0, row, 1)?
                .narrow(2, segment.start, segment.length)?
                .contiguous()?;
            let keys = k.narrow(0, row, 1)?.transpose(2, 3)?.contiguous()?;
            let values = v.narrow(0, row, 1)?.contiguous()?;
            let scores = (query.matmul(&keys)? * scale)?.broadcast_add(&batch.key_mask(index)?)?;
            parts.push(crate::nn::ops::softmax_last_dim(&scores)?.matmul(&values)?);
        }
        rows.push(Tensor::cat(&parts, 2)?);
    }
    Tensor::cat(&rows, 0)
}
#[test]
fn nonzero_bidirectional_trunk_matches_scalar_reference_and_discriminators() -> Result<()> {
    let trunk = trunk()?;
    let batch = trunk.prepare_batch(4, &[4, 3], &[2, 1])?;
    assert_eq!(batch.key_rows.elem_count(), 16);
    assert_eq!(
        batch
            .segments()
            .iter()
            .map(|s| s.length)
            .collect::<Vec<_>>(),
        [2, 2, 1, 3]
    );
    let (h, cos, sin) = inputs()?;
    let fixture: HashMap<String, Vec<Vec<Vec<f64>>>> = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/d1_omni/trunk_scalar.json"
    ))
    .unwrap();
    let expected = &fixture["expected"];
    let layer = &trunk.layers[0];
    let x = layer.operator_norm.forward(&h)?;
    if let Mixer::Conv {
        input,
        conv,
        output,
    } = &layer.mixer
    {
        let p = input
            .forward(&x.broadcast_mul(&batch.pad)?)?
            .transpose(1, 2)?;
        let y = output.forward(&conv.gated_sequence(
            &p.narrow(1, 0, 4)?,
            &p.narrow(1, 8, 4)?,
            &p.narrow(1, 4, 4)?,
            &batch.keep_right,
        )?)?;
        let res = (&h + y)?;
        let mixed =
            (&res + layer.ffn.forward(&layer.ffn_norm.forward(&res)?)?)?.to_vec3::<f32>()?;
        for (actual, expected) in mixed
            .iter()
            .flatten()
            .flatten()
            .zip(fixture["after_conv"].iter().flatten().flatten())
        {
            assert!(
                (*actual as f64 - expected).abs() < 2e-5,
                "conv/FFN scalar mismatch: {actual} vs {expected}"
            );
        }
    }

    for name in ["ignored_mask", "wrong_causal", "ignored_right", "no_rope"] {
        let max = expected
            .iter()
            .flatten()
            .flatten()
            .zip(fixture[name].iter().flatten().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        assert!(max > 1e-3, "{name} is not discriminated: {max}");
    }
    let mut prior = None;
    for _ in 0..3 {
        let actual = trunk
            .forward_with_attention(&h, &batch, &cos, &sin, &mut attention)?
            .to_vec3::<f32>()?;
        for (a, b) in actual
            .iter()
            .flatten()
            .flatten()
            .zip(expected.iter().flatten().flatten())
        {
            assert!(
                (*a as f64 - b).abs() < 2e-4,
                "trunk scalar mismatch: {a} vs {b}"
            );
        }
        if let Some(previous) = &prior {
            assert_eq!(&actual, previous);
        }
        prior = Some(actual);
    }
    Ok(())
}
#[test]
fn geometry_and_callback_contracts_fail_before_invalid_native_use() -> Result<()> {
    let mut cfg = config();
    cfg.intermediate_size = 6656;
    cfg.block_multiple_of = 256;
    assert_eq!(cfg.validated_intermediate()?, 4608);
    for corrupt in 0..10 {
        let mut c = config();
        match corrupt {
            0 => c.num_attention_heads = 0,
            1 => c.num_key_value_heads = 3,
            2 => c.hidden_size = 6,
            3 => c.layer_types.pop().map(|_| ()).unwrap(),
            4 => c.layer_types[0] = "unknown".into(),
            5 => c.conv_l_cache = 4,
            6 => c.block_multiple_of = 0,
            7 => c.block_ffn_dim_multiplier = f64::NAN,
            8 => c.intermediate_size = usize::MAX,
            _ => c.norm_eps = 0.,
        }
        assert!(
            c.validated_intermediate().is_err(),
            "invalid config {corrupt}"
        );
    }
    let trunk = trunk()?;
    assert!(trunk.prepare_batch(0, &[1], &[0]).is_err());
    assert!(trunk.prepare_batch(4, &[3], &[4]).is_err());
    assert!(trunk.prepare_batch(4, &[3], &[]).is_err());
    assert!(trunk.prepare_batch(4, &[0], &[0]).is_err());
    assert!(trunk.prepare_batch(33, &[3], &[0]).is_err());
    assert!(trunk.embed(&[8], 1, 1).is_err());
    assert!(trunk.embed(&[0], 2, 1).is_err());
    assert_eq!(trunk.embed(&[0, 7], 1, 2)?.dims(), [1, 2, 4]);
    let batch = trunk.prepare_batch(4, &[4, 3], &[2, 1])?;
    assert!(batch.key_mask(4).is_err());
    let mut constrained = trunk.clone();
    constrained.cfg.max_position_embeddings = 3;
    let (h, cos, sin) = inputs()?;
    let mut never =
        |_: &Tensor, _: &Tensor, _: &Tensor, _: &TrunkBatch, _: f64| -> Result<Tensor> {
            panic!("invalid input reached callback")
        };
    assert!(trunk
        .forward_with_attention(&h.to_dtype(DType::F64)?, &batch, &cos, &sin, &mut never)
        .is_err());
    assert!(trunk
        .forward_with_attention(&h, &batch, &cos.narrow(0, 0, 3)?, &sin, &mut never)
        .is_err());
    assert!(constrained
        .forward_with_attention(&h, &batch, &cos, &sin, &mut never)
        .is_err());
    let mut wrong =
        |q: &Tensor, _: &Tensor, _: &Tensor, _: &TrunkBatch, _: f64| q.to_dtype(DType::F64);
    assert!(trunk
        .forward_with_attention(&h, &batch, &cos, &sin, &mut wrong)
        .is_err());
    Ok(())
}

// Regression discriminator for the native CPU primitive reached by a centered
// block's transposed output followed by a broadcast-weight Linear. Keep the
// absolute scalar reference independent of any matmul implementation.
#[test]
fn transposed_batch_linear_matches_scalar_rows() -> Result<()> {
    let x = Tensor::from_vec(
        (0..12).map(|x| x as f32).collect::<Vec<_>>(),
        (2, 2, 3),
        &Device::Cpu,
    )?
    .transpose(1, 2)?;
    let w = Tensor::new(&[[1f32, 10.]], &Device::Cpu)?;
    let actual = Linear::new(w, None).forward(&x)?.to_vec3::<f32>()?;
    assert_eq!(
        actual,
        vec![
            vec![vec![30.], vec![41.], vec![52.]],
            vec![vec![96.], vec![107.], vec![118.]]
        ]
    );
    Ok(())
}

#[test]
fn owned_rotary_preparation_reuses_shared_generator_and_config_limits() -> Result<()> {
    let trunk = trunk()?;
    let (cos, sin) = trunk.rope_tables(4)?;
    assert_eq!(cos.dims(), [4, 1]);
    assert_eq!(sin.dims(), [4, 1]);
    let actual_cos = cos.to_vec2::<f32>()?;
    let actual_sin = sin.to_vec2::<f32>()?;
    for position in 0..4 {
        assert!((actual_cos[position][0] - (position as f32).cos()).abs() < 1e-7);
        assert!((actual_sin[position][0] - (position as f32).sin()).abs() < 1e-7);
    }
    let batch = trunk.prepare_batch(4, &[4, 3], &[2, 1])?;
    let (h, _, _) = inputs()?;
    let output = trunk.forward_with_attention(&h, &batch, &cos, &sin, &mut attention)?;
    assert_eq!(output.dims(), [2, 4, 4]);
    assert!(trunk.rope_tables(0).is_err());
    assert!(trunk.rope_tables(33).is_err());
    for theta in [f64::MAX, 1e-300] {
        let mut invalid = trunk.clone();
        invalid.cfg.rope_theta = theta;
        assert!(invalid.rope_tables(4).is_err());
    }
    Ok(())
}
