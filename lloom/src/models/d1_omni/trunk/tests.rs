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
fn weights() -> Result<HashMap<String, Tensor>> {
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
    Ok(w)
}
fn trunk() -> Result<Trunk> {
    Trunk::load(
        config(),
        VarBuilder::from_tensors(weights()?, DType::F32, &Device::Cpu),
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
    let layer = &trunk.inner.layers[0];
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
    constrained.inner.cfg.max_position_embeddings = 3;
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
        invalid.inner.cfg.rope_theta = theta;
        assert!(invalid.rope_tables(4).is_err());
    }
    Ok(())
}

#[test]
fn resolved_gguf_width_does_not_repeat_python_conversion() -> Result<()> {
    let mut raw = config();
    raw.intermediate_size = 6656;
    raw.block_multiple_of = 256;
    let resolved = raw.resolved()?;
    assert_eq!(resolved.intermediate_size, 4608);
    assert_eq!(resolved.validate(DType::F32)?, 4608);
    // GGUF provides4608 directly. No fabricated raw intermediate/multiplier is
    // part of this format-independent constructor contract.
    let gguf = ResolvedTextConfig {
        intermediate_size: 4608,
        ..resolved
    };
    assert_eq!(gguf.validate(DType::F32)?, 4608);
    // Negative control documents the actual double-conversion failure.
    raw.intermediate_size = 4608;
    assert_eq!(raw.validated_intermediate()?, 3072);
    Ok(())
}

#[test]
fn resolved_geometry_rejects_metadata_before_weight_work() -> Result<()> {
    let resolved = config().resolved()?;
    assert!(ResolvedTextConfig {
        intermediate_size: 0,
        ..resolved.clone()
    }
    .validate(DType::F32)
    .is_err());
    assert!(ResolvedTextConfig {
        intermediate_size: usize::MAX,
        ..resolved.clone()
    }
    .validate(DType::F32)
    .is_err());
    assert!(ResolvedTextConfig {
        num_attention_heads: 0,
        ..resolved.clone()
    }
    .validate(DType::F32)
    .is_err());
    assert!(ResolvedTextConfig {
        layer_types: vec!["conv".into()],
        ..resolved.clone()
    }
    .validate(DType::F32)
    .is_err());
    assert!(resolved.validate(DType::U32).is_err());
    Ok(())
}

// Canonical names from the pinned converter tensor map. F32 GGUF preserves
// the complete nonzero scalar fixture without introducing quantization error.
#[test]
fn gguf_f32_trunk_matches_independent_nonzero_scalar_reference() -> Result<()> {
    use lloom_tensor::quantized::{gguf_file, GgmlDType, QTensor};
    use std::io::Cursor;
    let mappings = [
        ("embed_tokens", "token_embd"),
        ("embedding_norm", "token_embd_norm"),
        ("layers.0.conv.in_proj", "blk.0.shortconv.in_proj"),
        ("layers.0.conv.out_proj", "blk.0.shortconv.out_proj"),
        ("layers.0.conv.conv", "blk.0.shortconv.conv"),
        ("layers.1.self_attn.q_proj", "blk.1.attn_q"),
        ("layers.1.self_attn.k_proj", "blk.1.attn_k"),
        ("layers.1.self_attn.v_proj", "blk.1.attn_v"),
        ("layers.1.self_attn.out_proj", "blk.1.attn_output"),
        ("layers.1.self_attn.q_layernorm", "blk.1.attn_q_norm"),
        ("layers.1.self_attn.k_layernorm", "blk.1.attn_k_norm"),
        ("layers.0.operator_norm", "blk.0.attn_norm"),
        ("layers.1.operator_norm", "blk.1.attn_norm"),
        ("layers.0.ffn_norm", "blk.0.ffn_norm"),
        ("layers.1.ffn_norm", "blk.1.ffn_norm"),
        ("layers.0.feed_forward.w1", "blk.0.ffn_gate"),
        ("layers.0.feed_forward.w3", "blk.0.ffn_up"),
        ("layers.0.feed_forward.w2", "blk.0.ffn_down"),
        ("layers.1.feed_forward.w1", "blk.1.ffn_gate"),
        ("layers.1.feed_forward.w3", "blk.1.ffn_up"),
        ("layers.1.feed_forward.w2", "blk.1.ffn_down"),
    ];
    let mut tensors = Vec::new();
    for (name, tensor) in weights()? {
        let stem = name.strip_suffix(".weight").unwrap();
        let mapped = mappings.iter().find(|(old, _)| *old == stem).unwrap().1;
        let tensor = if mapped == "blk.0.shortconv.conv" {
            tensor.squeeze(1)?
        } else {
            tensor
        };
        tensors.push((
            format!("{mapped}.weight"),
            QTensor::quantize(&tensor, GgmlDType::F32)?,
        ));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(tensors.len(), 21);
    let entries: Vec<_> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, &[], &entries)?;
    assert!(bytes.get_ref().len() < 16 * 1024);
    let vb =
        crate::quantized_var_builder::VarBuilder::from_gguf_buffer(bytes.get_ref(), &Device::Cpu)?;
    let trunk = QuantizedTrunk::load(config().resolved()?, vb)?;
    let batch = trunk.prepare_batch(4, &[4, 3], &[2, 1])?;
    let (h, cos, sin) = inputs()?;
    let fixture: HashMap<String, Vec<Vec<Vec<f64>>>> = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/d1_omni/trunk_scalar.json"
    ))
    .unwrap();
    let expected: Vec<_> = fixture["expected"]
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect();
    assert_eq!(expected.len(), 32);
    for _ in 0..2 {
        let actual = trunk
            .forward_with_attention(&h, &batch, &cos, &sin, &mut attention)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a as f64 - e).abs() < 2e-4,
                "GGUF scalar trunk[{i}]: {a} != {e}"
            );
        }
    }
    assert_eq!(trunk.embed(&[0, 7], 1, 2)?.dims(), [1, 2, 4]);
    assert_eq!(trunk.rope_tables(4)?.0.dims(), [4, 1]);
    Ok(())
}

// Quantized projection arithmetic has a separate scalar oracle; the callback
// returns V deliberately, so this tests trunk composition, not softmax parity.
#[test]
fn gguf_q8_trunk_multirow_gqa_rope_matches_scalar_projection_reference() -> Result<()> {
    use lloom_tensor::quantized::{gguf_file, GgmlDType, QTensor};
    use std::io::Cursor;
    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning.cpu.num_threads = 1;
    tuning.qmm.dequantize_all = false;
    tuning.qmm.dequantize_all_f16 = false;
    lloom_tensor::tuning::install(tuning)
        .expect("isolated Q8 trunk fixture (nextest or exact cargo filter)");
    assert_eq!(lloom_tensor::utils::barrier_pool().n_workers(), 0);
    eprintln!("BOUNDED_BARRIER_WORKERS=0");
    fn vals(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + salt) % 29) as f32 / 64. - 0.21875)
            .collect()
    }
    fn block(row: &[f32]) -> (Vec<i32>, f32) {
        assert_eq!(row.len(), 32);
        let d = row.iter().fold(0f32, |a, x| a.max(x.abs())) / 127.;
        let inv = if d == 0. { 0. } else { 1. / d };
        (
            row.iter().map(|x| (x * inv).round() as i32).collect(),
            half::f16::from_f32(d).to_f32(),
        )
    }
    fn project(x: &[f32], w: &[f32]) -> Vec<f32> {
        x.chunks_exact(32)
            .flat_map(|row| {
                let (q, d) = block(row);
                w.chunks_exact(32).map(move |w| {
                    let (w, wd) = block(w);
                    let dot: i32 = q.iter().zip(w).map(|(&a, b)| a * b).sum();
                    dot as f32 * wd * d
                })
            })
            .collect()
    }
    fn norm(x: &[f32], width: usize) -> Vec<f32> {
        x.chunks_exact(width)
            .flat_map(|r| {
                let inv = (r.iter().map(|x| x * x).sum::<f32>() / width as f32 + 1e-5)
                    .sqrt()
                    .recip();
                r.iter().map(move |x| x * inv)
            })
            .collect()
    }
    fn close(a: &[f32], e: &[f32], name: &str) {
        assert_eq!(a.len(), e.len());
        for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
            assert!((a - e).abs() < 2e-4, "{name}[{i}]: {a} != {e}");
        }
    }
    let mut cfg = config().resolved()?;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 32;
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec!["full_attention".into()];
    let mut tensors = Vec::new();
    for (name, rows, cols, salt) in [
        ("attn_q", 32, 32, 1),
        ("attn_k", 16, 32, 2),
        ("attn_v", 16, 32, 3),
        ("attn_output", 32, 32, 4),
        ("ffn_gate", 32, 32, 0),
        ("ffn_up", 32, 32, 0),
        ("ffn_down", 32, 32, 0),
    ] {
        let data = if salt == 0 {
            vec![0f32; rows * cols]
        } else {
            vals(rows * cols, salt)
        };
        tensors.push((
            format!("blk.0.{name}.weight"),
            QTensor::quantize(
                &Tensor::from_vec(data, (rows, cols), &Device::Cpu)?,
                GgmlDType::Q8_0,
            )?,
        ));
    }
    for (name, n) in [
        ("token_embd_norm", 32),
        ("blk.0.attn_norm", 32),
        ("blk.0.ffn_norm", 32),
        ("blk.0.attn_q_norm", 16),
        ("blk.0.attn_k_norm", 16),
    ] {
        tensors.push((
            format!("{name}.weight"),
            QTensor::quantize(&Tensor::ones(n, DType::F32, &Device::Cpu)?, GgmlDType::F32)?,
        ));
    }
    tensors.push((
        "token_embd.weight".into(),
        QTensor::quantize(
            &Tensor::from_vec(vals(8 * 32, 8), (8, 32), &Device::Cpu)?,
            GgmlDType::F32,
        )?,
    ));
    let entries: Vec<_> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, &[], &entries)?;
    assert!(bytes.get_ref().len() < 32 * 1024);
    let vb =
        crate::quantized_var_builder::VarBuilder::from_gguf_buffer(bytes.get_ref(), &Device::Cpu)?;
    assert_eq!(
        vb.get((32, 32), "blk.0.attn_q.weight")?.dtype(),
        GgmlDType::Q8_0
    );
    let trunk = QuantizedTrunk::load(cfg, vb)?;
    let batch = trunk.prepare_batch(4, &[4, 3], &[2, 1])?;
    let h = vals(2 * 4 * 32, 9);
    let x = norm(&h, 32);
    let q = norm(&project(&x, &vals(32 * 32, 1)), 16);
    let k = norm(&project(&x, &vals(16 * 32, 2)), 16);
    let v = project(&x, &vals(16 * 32, 3));
    let cs = [1f32, 0.6, 0., -0.8];
    let sn = [0f32, 0.8, 1., 0.6];
    let cos = Tensor::from_vec(
        cs.into_iter().flat_map(|x| [x; 8]).collect::<Vec<_>>(),
        (4, 8),
        &Device::Cpu,
    )?;
    let sin = Tensor::from_vec(
        sn.into_iter().flat_map(|x| [x; 8]).collect::<Vec<_>>(),
        (4, 8),
        &Device::Cpu,
    )?;
    let mut refs: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    let mut merged_v = vec![0f32; 256];
    for b in 0..2 {
        for head in 0..2 {
            for t in 0..4 {
                for c in 0..16 {
                    let qi = (b * 4 + t) * 32 + head * 16 + c;
                    let ki = (b * 4 + t) * 16 + c;
                    let sign = if c < 8 { -1. } else { 1. };
                    let other = (c + 8) % 16;
                    refs[0].push(
                        q[qi] * cs[t] + sign * q[(b * 4 + t) * 32 + head * 16 + other] * sn[t],
                    );
                    refs[1].push(k[ki] * cs[t] + sign * k[(b * 4 + t) * 16 + other] * sn[t]);
                    refs[2].push(v[ki]);
                    merged_v[qi] = v[ki];
                }
            }
        }
    }
    let mixed = project(&merged_v, &vals(32 * 32, 4));
    let expected = norm(
        &h.iter().zip(mixed).map(|(&a, b)| a + b).collect::<Vec<_>>(),
        32,
    );
    assert!(expected
        .iter()
        .zip(norm(&h, 32))
        .any(|(a, b)| (a - b).abs() > 1e-3));
    let mut calls = 0;
    let actual = trunk
        .forward_with_attention(
            &Tensor::from_slice(&h, (2, 4, 32), &Device::Cpu)?,
            &batch,
            &cos,
            &sin,
            &mut |q, k, v, meta, scale| {
                calls += 1;
                assert_eq!(scale, 0.25);
                assert_eq!(
                    meta.segments().iter().map(|s| s.length).collect::<Vec<_>>(),
                    [2, 2, 1, 3]
                );
                for (i, p) in [q, k, v].into_iter().enumerate() {
                    assert_eq!(p.dims(), [2, 2, 4, 16]);
                    close(
                        &p.flatten_all()?.to_vec1::<f32>()?,
                        &refs[i],
                        "Q8 trunk QKV",
                    );
                }
                Ok(v.clone())
            },
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(calls, 1);
    close(&actual, &expected, "Q8 trunk residual output");
    Ok(())
}
