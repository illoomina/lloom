use lloom::models::lfm2::{self, LayerType, State};
use lloom::models::lfm2_moe::{schema, testing::Lfm2MoeTiny, Config};
use lloom::nn::moe::Routing;
use lloom::profile::CrownExact;
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

fn checkpoint() -> Config {
    serde_json::from_str(include_str!("fixtures/lfm2_moe/config.json")).unwrap()
}

#[test]
fn pinned_config_does_not_import_dense_auto_adjust() -> Result<()> {
    let cfg = checkpoint();
    let runtime = cfg.lfm_config()?;
    assert_eq!(runtime.intermediate_size, 11776);
    assert_eq!(runtime.hidden_size, 2048);
    assert_eq!(runtime.rope_theta, 10_000_000.0);
    assert_eq!(runtime.max_position_embeddings, 131072);
    assert!(runtime.tie_embedding);
    assert_eq!(
        runtime
            .layer_types
            .iter()
            .filter(|&&x| x == LayerType::Conv)
            .count(),
        30
    );
    assert_eq!(runtime.moe.unwrap().num_dense_layers, 2);
    let legacy: lfm2::Lfm2Config =
        serde_json::from_str(include_str!("fixtures/lfm2_moe/config.json")).unwrap();
    // The adapter does not change the dense parser's legacy default.
    assert_eq!(legacy.into_config(false).intermediate_size, 8192);
    Ok(())
}

#[test]
fn pinned_index_names_and_payload_match_the_schema() -> Result<()> {
    let schema = schema::tensor_schema(&checkpoint(), GgmlDType::BF16)?;
    let mut names: Vec<_> = schema.tensors.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    let original: Vec<_> = include_str!("fixtures/lfm2_moe/tensor-names.txt")
        .lines()
        .collect();
    assert_eq!(names, original);
    assert_eq!(names.len(), 7610);
    assert_eq!(schema::payload_bytes(&schema)?, 47_943_180_288);
    assert_eq!(
        schema
            .get("model.layers.0.feed_forward.w1.weight")
            .unwrap()
            .shape,
        [11776, 2048]
    );
    assert_eq!(
        schema
            .get("model.layers.2.feed_forward.experts.63.w2.weight")
            .unwrap()
            .shape,
        [2048, 1536]
    );
    assert_eq!(
        schema
            .get("model.layers.2.feed_forward.expert_bias")
            .unwrap()
            .dtype,
        GgmlDType::F32
    );
    assert!(schema.get("lm_head.weight").is_none());
    assert!(schema
        .get("model.layers.2.feed_forward.w1.weight")
        .is_none());
    Ok(())
}

#[test]
fn malformed_geometry_and_unimplemented_rope_are_refused() {
    let base = checkpoint();
    for change in 0..8 {
        let mut cfg = base.clone();
        match change {
            0 => cfg.num_experts_per_tok = 0,
            1 => cfg.num_experts_per_tok = cfg.num_experts + 1,
            2 => cfg.num_dense_layers = 41,
            3 => cfg.base.layer_types.pop().map(|_| ()).unwrap(),
            4 => cfg.base.num_attention_heads = 0,
            5 => cfg.base.intermediate_size = None,
            6 => cfg.rope_parameters.rope_type = "dynamic".into(),
            _ => cfg.routed_scaling_factor = f64::NAN,
        }
        assert!(cfg.lfm_config().is_err(), "invalid case {change}");
    }
}

fn values(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn close(got: &[f32], expected: &[f32], bound: f32) {
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= bound,
            "value {i}: {a} != {b} (bound {bound})"
        );
    }
}

#[test]
fn sigmoid_bias_selects_but_never_becomes_a_mixture_weight() -> Result<()> {
    let dev = Device::Cpu;
    // Every selection score is negative. A softmax router's -1 sentinel
    // cannot be reused as a picked-expert marker for this policy.
    let logits = Tensor::from_vec(vec![0f32, 2., -2., 1.], (1, 4), &dev)?;
    let bias = Tensor::from_vec(vec![-2f32, -4., -1., -3.], 4, &dev)?;
    let route = Routing::Sigmoid {
        bias: Some(bias),
        normalize: true,
        scale: 1.7,
    };
    let (ids, w) = route.route(&logits, 2)?;
    assert_eq!(ids.to_vec2::<u32>()?, vec![vec![2, 0]]);
    let p = 1.0f64 / (1.0 + 2.0f64.exp());
    let want = [1.7 * p / (p + 0.5 + 1e-6), 1.7 * 0.5 / (p + 0.5 + 1e-6)];
    close(&values(&w), &want.map(|v| v as f32), 2e-7);
    Ok(())
}

#[test]
fn epsilon_normalization_scale_and_unnormalized_policy_are_value_checked() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::from_vec(vec![-20f32, -21., -22.], (1, 3), &dev)?;
    for normalize in [false, true] {
        let policy = Routing::Sigmoid {
            bias: None,
            normalize,
            scale: 2.0,
        };
        let (ids, w) = policy.route(&logits, 2)?;
        assert_eq!(ids.to_vec2::<u32>()?, vec![vec![0, 1]]);
        let p = [20f64, 21.].map(|x| 1.0 / (1.0 + x.exp()));
        let denominator = if normalize { p[0] + p[1] + 1e-6 } else { 1.0 };
        close(
            &values(&w),
            &p.map(|x| (2.0 * x / denominator) as f32),
            1e-9,
        );
    }
    Ok(())
}

#[test]
fn ties_are_unique_topk_and_repeatable_without_cross_backend_order_claim() -> Result<()> {
    let logits = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;
    let policy = Routing::Sigmoid {
        bias: None,
        normalize: true,
        scale: 1.0,
    };
    let (ids, w) = policy.route(&logits, 3)?;
    let first = ids.to_vec2::<u32>()?;
    for row in &first {
        let unique: std::collections::BTreeSet<_> = row.iter().collect();
        assert_eq!(unique.len(), 3);
        assert!(row.iter().all(|&x| x < 4));
    }
    close(&values(&w), &[0.5 / (1.5 + 1e-6); 9], 1e-7);
    for _ in 0..3 {
        assert_eq!(policy.route(&logits, 3)?.0.to_vec2::<u32>()?, first);
    }
    assert!(policy.route(&logits, 0).is_err());
    assert!(policy.route(&logits, 5).is_err());
    Ok(())
}

#[test]
fn tiny_constructor_requests_the_same_schema_and_preserves_mixed_blocks() -> Result<()> {
    let cfg = Lfm2MoeTiny::template();
    let recorded = TinyCheckpoint::<Lfm2MoeTiny>::template().schema()?;
    let parsed: Config =
        serde_json::from_value(lloom::models::lfm2_testing::geometry_json(&cfg)).unwrap();
    let declared = schema::tensor_schema(&parsed, GgmlDType::F32)?;
    let table = |s: &lloom::testing::tiny::Schema| {
        s.tensors
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(table(&recorded), table(&declared));
    assert!(declared
        .get("model.layers.0.feed_forward.w1.weight")
        .is_some());
    assert!(declared
        .get("model.layers.1.feed_forward.experts.0.w1.weight")
        .is_some());
    assert!(declared
        .get("model.layers.2.self_attn.q_proj.weight")
        .is_some());
    Ok(())
}

#[test]
fn mixed_model_reset_and_continuation_use_existing_lfm_state() -> Result<()> {
    let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_state", Lfm2MoeTiny::template())? else {
        return Ok(());
    };
    let Some(dev) = fx.device(Device::Cpu) else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let model = fx.model::<CrownExact>(&cfg, &dev)?;
    let mut state = State::<CrownExact>::new(
        true,
        lloom::models::lfm2_testing::run_dtype(&dev),
        &cfg,
        &dev,
    )?;
    let tokens = Tensor::new(&[[1u32, 2, 3]], &dev)?;
    let all = model.forward(&tokens, 0, &mut state)?;
    state.clear();
    let again = model.forward(&tokens, 0, &mut state)?;
    close(&values(&again), &values(&all), 0.0);
    state.clear();
    let _ = model.forward(&tokens.narrow(1, 0, 2)?, 0, &mut state)?;
    let split = model.forward(&tokens.narrow(1, 2, 1)?, 2, &mut state)?;
    close(
        &values(&split),
        &values(&all),
        if dev.is_cpu() { 1e-4 } else { 1e-2 },
    );
    assert!(values(&all).iter().any(|x| x.abs() > 1e-4));
    Ok(())
}

#[test]
fn routed_only_block_matches_scalar_experts_and_batched_positions() -> Result<()> {
    use lloom::models::qwen4exp::moe::{MoeGeometry, MoeLayer, Resident};
    use lloom_tensor::quantized::{QMatMul, QTensor};
    use std::sync::Arc;
    let dev = Device::Cpu;
    let (e, h, i, k) = (4usize, 4usize, 3usize, 2usize);
    let matrix = |count: usize, phase: f32| {
        (0..count)
            .map(|j| ((j as f32 * 0.31 + phase).sin()) * 0.3)
            .collect::<Vec<_>>()
    };
    let gate = matrix(e * i * h, 0.1);
    let up = matrix(e * i * h, 0.7);
    let down = matrix(e * h * i, 1.3);
    let router = matrix(e * h, 2.1);
    let bias = vec![-2f32, -4., -1., -3.];
    let pack = |v: &[f32], shape: (usize, usize, usize)| -> Result<Arc<QTensor>> {
        Ok(Arc::new(QTensor::quantize(
            &Tensor::from_vec(v.to_vec(), shape, &dev)?,
            GgmlDType::F32,
        )?))
    };
    let layer = MoeLayer::<CrownExact, _>::routed_only(
        QMatMul::Tensor(Tensor::from_vec(router.clone(), (e, h), &dev)?),
        Resident::new(
            pack(&gate, (e, i, h))?,
            pack(&up, (e, i, h))?,
            pack(&down, (e, h, i))?,
        ),
        &MoeGeometry {
            n_experts: e,
            n_experts_per_tok: k,
            moe_intermediate_size: i,
        },
        Routing::Sigmoid {
            bias: Some(Tensor::from_vec(bias.clone(), e, &dev)?),
            normalize: true,
            scale: 1.3,
        },
    )?;
    let xs = [vec![0.2f32, -0.7, 1.1, 0.4], vec![-0.3, 0.8, 0.1, -0.2]];
    let mut expected = Vec::new();
    let mut expected_ids = Vec::new();
    for x in &xs {
        let probs: Vec<_> = (0..e)
            .map(|expert| {
                let z: f64 = (0..h)
                    .map(|n| x[n] as f64 * router[expert * h + n] as f64)
                    .sum();
                1.0 / (1.0 + (-z).exp())
            })
            .collect();
        let mut order: Vec<_> = (0..e).collect();
        order.sort_by(|&a, &b| (probs[b] + bias[b] as f64).total_cmp(&(probs[a] + bias[a] as f64)));
        let selected = &order[..k];
        expected_ids.push(selected.iter().map(|&n| n as u32).collect::<Vec<_>>());
        let sum: f64 = selected.iter().map(|&expert| probs[expert]).sum::<f64>() + 1e-6;
        let mut y = vec![0f64; h];
        for &expert in selected {
            let mut mid = vec![0f64; i];
            for row in 0..i {
                let dot = |w: &[f32]| {
                    (0..h)
                        .map(|n| x[n] as f64 * w[(expert * i + row) * h + n] as f64)
                        .sum::<f64>()
                };
                let g = dot(&gate);
                mid[row] = g / (1.0 + (-g).exp()) * dot(&up);
            }
            for row in 0..h {
                let v: f64 = (0..i)
                    .map(|n| mid[n] * down[(expert * h + row) * i + n] as f64)
                    .sum();
                y[row] += v * probs[expert] / sum * 1.3;
            }
        }
        expected.extend(y.into_iter().map(|x| x as f32));
    }
    let x = Tensor::from_vec(xs.concat(), (1, 2, h), &dev)?;
    let got = layer.forward(&x)?;
    close(&values(&got), &expected, 2e-7);
    assert!(
        expected.iter().any(|v| v.abs() > 1e-5),
        "non-vacuous scalar oracle"
    );
    assert_eq!(layer.route_ids(&x)?.to_vec2::<u32>()?, expected_ids);
    let rows = layer.forward(&x.reshape((2, 1, h))?)?;
    close(&values(&rows), &expected, 2e-7);
    for _ in 0..3 {
        close(&values(&layer.forward(&x)?), &values(&got), 0.0);
    }
    Ok(())
}

#[test]
fn softmax_policy_keeps_qwens_renormalized_weights_without_epsilon() -> Result<()> {
    let logits = Tensor::from_vec(vec![0f32, 1., 2., -1.], (1, 4), &Device::Cpu)?;
    let (ids, w) = Routing::Softmax.route(&logits, 2)?;
    assert_eq!(ids.to_vec2::<u32>()?, vec![vec![2, 1]]);
    let p = 1.0 / (1.0 + (-1f64).exp());
    close(&values(&w), &[p as f32, (1.0 - p) as f32], 1e-7);
    Ok(())
}

#[test]
fn mixed_fixture_roundtrips_through_the_actual_safetensors_loader() -> Result<()> {
    let mut tiny = Lfm2MoeTiny::template();
    tiny.vocab_size = Lfm2MoeTiny::tokenizer().vocab_size();
    let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_files", tiny)? else {
        return Ok(());
    };
    // Scale runs use the checkpoint fixture; they do not synthesize a second
    // 48 GB file merely to exercise the tiny writer.
    if fx.at_scale() {
        return Ok(());
    }
    let cfg = fx.config(|_| {});
    let path = fx.scratch("checkpoint");
    std::fs::create_dir_all(&path)?;
    let ckpt = TinyCheckpoint::<Lfm2MoeTiny>::new(cfg.clone(), Default::default());
    Lfm2MoeTiny::write(&ckpt, &path)?;
    let stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?).unwrap();
    let parsed = lloom::models::lfm2_moe::config::parse_lfm_config(stored)?;
    assert_eq!(parsed.moe, cfg.moe);
    let a = fx.model::<CrownExact>(&cfg, &Device::Cpu)?;
    let b = Lfm2MoeTiny::load::<CrownExact>(&path, &Device::Cpu)?;
    let mut sa = State::<CrownExact>::new(true, DType::F32, &cfg, &Device::Cpu)?;
    let mut sb = State::<CrownExact>::new(true, DType::F32, &parsed, &Device::Cpu)?;
    let ids = Tensor::new(&[[2u32, 3, 4]], &Device::Cpu)?;
    close(
        &values(&a.forward(&ids, 0, &mut sa)?),
        &values(&b.forward(&ids, 0, &mut sb)?),
        0.0,
    );
    // The public programmatic consumer uses the same model trait and local
    // checkpoint path. No Hub fetch or alternative decode driver is involved.
    use lloom::hub::{Checkpoint, Pretrained};
    use lloom::nn::CausalLM;
    let checkpoint = Checkpoint::from_safetensors_dir(&path)?;
    let runtime = lloom::models::lfm2_moe::config::parse_lfm_config(checkpoint.config.clone())?;
    let wire: Config = serde_json::from_value(checkpoint.config.clone()).unwrap();
    // SAFETY: this fixture's files remain immutable until every mapping drops.
    unsafe {
        lloom::models::lfm2_moe::preflight(&wire, &checkpoint.weights.paths(), DType::F32)?;
    }
    let pretrained: Pretrained<lfm2::Model<CrownExact>> =
        unsafe { Pretrained::from_checkpoint(checkpoint, &runtime, DType::F32, &Device::Cpu)? };
    let tokenized = pretrained
        .tokenizer
        .as_ref()
        .unwrap()
        .encode("Hi", true)
        .map_err(lloom_tensor::Error::wrap)?;
    assert!(!tokenized.get_ids().is_empty());
    let ids = Tensor::from_vec(
        tokenized.get_ids().to_vec(),
        (1, tokenized.len()),
        &Device::Cpu,
    )?;
    let mut state = pretrained.model.new_cache(&runtime, &Device::Cpu)?;
    sa.clear();
    close(
        &values(&pretrained.model.forward(&ids, 0, &mut state)?),
        &values(&a.forward(&ids, 0, &mut sa)?),
        0.0,
    );
    Ok(())
}

#[test]
fn bf16_route_preserves_reference_rounding_between_stages() -> Result<()> {
    use half::bf16;
    let round = |x: f32| bf16::from_f32(x).to_f32();
    let raw = [0.003f32, 0.021, 2.125, -2.25];
    let bias = Tensor::from_vec(vec![0.05f32, 0.03, -0.6, 1.0], 4, &Device::Cpu)?;
    let logits = Tensor::from_vec(raw.to_vec(), (1, 4), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let (ids, w) = Routing::Sigmoid {
        bias: Some(bias),
        normalize: true,
        scale: 1.7,
    }
    .route(&logits, 3)?;
    let p = raw.map(|x| round(1.0 / (1.0 + (-round(x)).exp())));
    let b = [0.05f32, 0.03, -0.6, 1.0];
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &bidx| (p[bidx] + b[bidx]).total_cmp(&(p[a] + b[a])));
    let selected = &order[..3];
    assert_eq!(
        ids.to_vec2::<u32>()?,
        vec![selected.iter().map(|&x| x as u32).collect::<Vec<_>>()]
    );
    let denominator = round(round(selected.iter().map(|&n| p[n]).sum::<f32>()) + 1e-6);
    let expected: Vec<_> = selected
        .iter()
        .map(|&n| round(round(p[n] / denominator) * 1.7))
        .collect();
    close(&values(&w), &expected, 0.0);
    Ok(())
}

#[test]
fn header_preflight_checks_shape_dtype_bias_and_duplicate_shards() -> Result<()> {
    use lloom::models::lfm2_moe::preflight;
    let cfg: Config = serde_json::from_value(lloom::models::lfm2_testing::geometry_json(
        &Lfm2MoeTiny::template(),
    ))
    .unwrap();
    let schema = schema::tensor_schema(&cfg, GgmlDType::F32)?;
    let dir = std::env::temp_dir().join(format!("lloom-moe-headers-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let file = dir.join("model.safetensors");
    let mut tensors = std::collections::HashMap::new();
    for t in &schema.tensors {
        tensors.insert(
            t.name.clone(),
            Tensor::zeros(t.shape.clone(), DType::F32, &Device::Cpu)?,
        );
    }
    let bias = "model.layers.1.feed_forward.expert_bias";
    let gate = "model.layers.1.feed_forward.gate.weight";
    let check = |tensors: &std::collections::HashMap<String, Tensor>| {
        lloom_tensor::safetensors::save(tensors, &file)?;
        // SAFETY: this test never changes a file while its mapping is alive.
        unsafe { preflight(&cfg, std::slice::from_ref(&file), DType::F16) }
    };
    let good = check(&tensors)?;
    let elements = schema.elements();
    let bias_elements = cfg.num_experts * (cfg.base.num_hidden_layers - cfg.num_dense_layers);
    assert_eq!(good.resident_bytes, elements * 2 + bias_elements * 2);
    assert!(good.packing_bytes > 0);
    let original = tensors.remove(bias).unwrap();
    assert!(check(&tensors).unwrap_err().to_string().contains("missing"));
    tensors.insert(bias.into(), original.to_dtype(DType::F16)?);
    assert!(check(&tensors)
        .unwrap_err()
        .to_string()
        .contains("bias must be F32"));
    tensors.insert(bias.into(), original);
    let original = tensors.remove(gate).unwrap();
    tensors.insert(gate.into(), original.to_dtype(DType::U8)?);
    assert!(check(&tensors)
        .unwrap_err()
        .to_string()
        .contains("unsupported safetensors dtype"));
    tensors.insert(
        gate.into(),
        Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
    );
    assert!(check(&tensors).unwrap_err().to_string().contains("shape"));
    tensors.insert(gate.into(), original);
    check(&tensors)?;
    // SAFETY: the previous check's mappings have been dropped.
    let duplicate = unsafe { preflight(&cfg, &[file.clone(), file], DType::F32) };
    assert!(duplicate
        .unwrap_err()
        .to_string()
        .contains("duplicate tensor"));
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn admission_bounds_do_not_wrap_and_unsupported_conv_bias_is_explicit() -> Result<()> {
    let mut cfg = Lfm2MoeTiny::template();
    let mut moe = cfg.moe.unwrap();
    assert!(moe.activation_bound(cfg.hidden_size) > 3 * cfg.intermediate_size * 2);
    assert!(
        moe.packing_bound(cfg.hidden_size, DType::F32)
            >= moe.packing_bound(cfg.hidden_size, DType::F16)
    );
    moe.num_experts = usize::MAX;
    assert_eq!(moe.packing_bound(cfg.hidden_size, DType::F32), usize::MAX);
    assert_eq!(moe.activation_bound(cfg.hidden_size), usize::MAX);
    cfg.moe = Some(moe);
    assert!(moe.validate(&cfg).is_err());
    let mut wire = checkpoint();
    wire.base.conv_bias = true;
    assert!(wire
        .lfm_config()
        .unwrap_err()
        .to_string()
        .contains("conv_bias"));
    Ok(())
}

#[test]
fn existing_qwen_constructor_keeps_the_nonzero_gated_shared_expert() -> Result<()> {
    use lloom::models::qwen4exp::moe::{MoeGeometry, MoeLayer, Resident};
    use lloom_tensor::quantized::{QMatMul, QTensor};
    use std::sync::Arc;
    let dev = Device::Cpu;
    let identity = vec![1f32, 0., 0., 1.];
    let matrix = || -> Result<QMatMul> {
        Ok(QMatMul::Tensor(Tensor::from_vec(
            identity.clone(),
            (2, 2),
            &dev,
        )?))
    };
    let stack = || -> Result<Arc<QTensor>> {
        Ok(Arc::new(QTensor::quantize(
            &Tensor::from_vec(
                [identity.clone(), identity.clone()].concat(),
                (2, 2, 2),
                &dev,
            )?,
            GgmlDType::F32,
        )?))
    };
    let layer = MoeLayer::<CrownExact, _>::with_geometry(
        matrix()?,
        Resident::new(stack()?, stack()?, stack()?),
        matrix()?,
        matrix()?,
        matrix()?,
        QMatMul::Tensor(Tensor::from_vec(vec![0.7f32, -0.4], (1, 2), &dev)?),
        &MoeGeometry {
            n_experts: 2,
            n_experts_per_tok: 1,
            moe_intermediate_size: 2,
        },
    );
    let x = Tensor::new(&[[[0.2f32, -0.3]]], &dev)?;
    assert_eq!(layer.route_ids(&x)?.to_vec2::<u32>()?, vec![vec![0]]);
    let shared_gate = 1.0 / (1.0 + (-0.26f64).exp());
    let expected: Vec<_> = [0.2f64, -0.3]
        .map(|v| ((1.0 + shared_gate) * v * v / (1.0 + (-v).exp())) as f32)
        .to_vec();
    close(&values(&layer.forward(&x)?), &expected, 1e-7);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn sigmoid_route_native_matches_cpu_reference_and_repeats() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let raw = vec![0.3f32, -2.4, 1.9, -0.7, -1.2, 2.7, 0.8, -3.0];
    for dtype in [DType::F32, DType::F16] {
        let cpu = Tensor::from_vec(raw.clone(), (2, 4), &Device::Cpu)?.to_dtype(dtype)?;
        let gpu = cpu.to_device(&dev)?;
        let bias = Tensor::from_vec(vec![-2f32, -3., -1., -4.], 4, &Device::Cpu)?;
        let route = |bias| Routing::Sigmoid {
            bias: Some(bias),
            normalize: true,
            scale: 1.3,
        };
        let (ids, reference) = route(bias.clone()).route(&cpu, 2)?;
        let native = route(bias.to_device(&dev)?);
        let (selected, weights) = native.route(&gpu, 2)?;
        assert_eq!(selected.to_vec2::<u32>()?, ids.to_vec2::<u32>()?);
        close(
            &values(&weights),
            &values(&reference),
            if dtype == DType::F32 { 2e-6 } else { 0.002 },
        );
        for _ in 0..4 {
            let (again_ids, again_weights) = native.route(&gpu, 2)?;
            assert_eq!(again_ids.to_vec2::<u32>()?, selected.to_vec2::<u32>()?);
            close(&values(&again_weights), &values(&weights), 0.0);
        }
    }
    Ok(())
}

#[test]
fn mixed_model_walk_names_actual_ffns_and_separate_ids_and_weights() -> Result<()> {
    let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_walk", Lfm2MoeTiny::template())? else {
        return Ok(());
    };
    let Some(dev) = fx.device(Device::Cpu) else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let model = fx.model::<CrownExact>(&cfg, &dev)?;
    let graph = lloom::optimizer::walk(&model);
    let names: Vec<_> = graph.nodes().iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"model.layers.0.feed_forward.w1.weight"));
    assert!(names.contains(&"model.layers.1.feed_forward.gate.weight"));
    assert!(names.contains(&"model.layers.1.feed_forward.expert_bias"));
    assert!(!names.iter().any(|name| name.contains("shexp")));
    assert!(
        !names.contains(&"moe_topk"),
        "sigmoid is not the fused softmax route"
    );
    assert!(graph.nodes().iter().any(|n| n.name == "gather_u32"
        && n.inputs.len() == 2
        && graph.node(n.inputs[1]).dtype == DType::U32
        && n.dtype == DType::F32));
    assert!(names.contains(&"bdiv"));
    assert!(names.contains(&"bmul"));
    assert!(names.contains(&"silu_mul"));
    assert!(names.contains(&"fast_sum"));
    // Topology/value dependencies only: native dispatch-census reconciliation
    // is a separate acceptance gate, not established by these assertions.
    Ok(())
}

#[test]
fn mixed_state_clear_recovers_after_a_late_attention_error() -> Result<()> {
    let mut tiny = Lfm2MoeTiny::template();
    tiny.max_position_embeddings = 2;
    let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_error_reset", tiny)? else {
        return Ok(());
    };
    if fx.at_scale() {
        return Ok(());
    }
    let cfg = fx.config(|_| {});
    let model = fx.model::<CrownExact>(&cfg, &Device::Cpu)?;
    let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, &Device::Cpu)?;
    let input = Tensor::new(&[[1u32]], &Device::Cpu)?;
    // A fresh cache has room for one token, but position 2 lies outside its
    // RoPE table. The two conv mixers/FFNs precede the attention in this
    // fixture, so this is a failure after earlier layers have written state.
    let error = model
        .forward(&input, 2, &mut cache)
        .unwrap_err()
        .to_string();
    assert!(error.contains("narrow"), "unexpected failure: {error}");
    cache.clear();
    let after = model.forward(&input, 0, &mut cache)?;
    let mut fresh = State::<CrownExact>::new(true, DType::F32, &cfg, &Device::Cpu)?;
    close(
        &values(&after),
        &values(&model.forward(&input, 0, &mut fresh)?),
        0.0,
    );
    // This pins model-state reset, not the engine's callback/unwind custody.
    Ok(())
}

#[test]
fn mixed_derived_fusion_keeps_the_router_unfused_at_every_budget() -> Result<()> {
    use lloom::optimizer::NumericClass;
    // Four experts exercise the skinny router tile. Sixteen makes the router
    // norm-eligible in the generic graph, but the routed executor has no
    // norm-prologue switch: its norm must remain a dispatch at either shape.
    for experts in [4, 16] {
        let mut cfg = Lfm2MoeTiny::template();
        cfg.moe.as_mut().unwrap().num_experts = experts;
        let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_fusion", cfg)? else {
            return Ok(());
        };
        if fx.at_scale() {
            return Ok(());
        }
        let cfg = fx.config(|_| {});
        for budget in [
            NumericClass::BitExact,
            NumericClass::Rounding,
            NumericClass::ReorderEquiv,
        ] {
            let mut model = fx.model::<CrownExact>(&cfg, &Device::Cpu)?;
            let report = model.derive_fusion_at(&cfg, budget)?;
            eprintln!("experts={experts} budget={budget:?}: {report}");
            // derive_fusion_at rejects any fused site without an execution
            // switch (including the router). A successful reorder derivation
            // must still retain the existing mixer/dense/head norm arm.
            assert_eq!(report.norm_gemv, budget == NumericClass::ReorderEquiv);
            let input = Tensor::new(&[[1u32, 2]], &Device::Cpu)?;
            let mut state = State::<CrownExact>::new(true, DType::F32, &cfg, &Device::Cpu)?;
            let actual = model.forward(&input, 0, &mut state)?;
            let plain = fx.model::<CrownExact>(&cfg, &Device::Cpu)?;
            state.clear();
            close(
                &values(&actual),
                &values(&plain.forward(&input, 0, &mut state)?),
                0.0,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn mixed_native_model_matches_rounded_cpu_weights_and_repeats() -> Result<()> {
    use lloom::testing::tiny;
    let dev = Device::new_metal(0)?;
    let ckpt = TinyCheckpoint::<Lfm2MoeTiny>::template().with_dtype(DType::F16);
    let reference = Lfm2MoeTiny::reference::<CrownExact>(&ckpt, &Device::Cpu)?;
    let model = ckpt.model::<CrownExact>(&dev)?;
    // Same normalized max-logit metric as the incumbent tiny family suite.
    // This is a synthetic backend gate, not a Transformers/real-model result.
    for workload in Lfm2MoeTiny::workloads(&ckpt.cfg) {
        let want = tiny::run(
            &reference,
            &ckpt.cfg,
            ckpt.cfg.vocab_size,
            &Device::Cpu,
            &workload,
        )?;
        let got = tiny::run(&model, &ckpt.cfg, ckpt.cfg.vocab_size, &dev, &workload)?;
        assert_eq!(want.len(), got.len());
        let mut relative = 0f32;
        for (cpu, gpu) in want.iter().zip(&got) {
            assert_eq!(cpu.len(), gpu.len());
            let scale = cpu.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            for (&a, &b) in cpu.iter().zip(gpu) {
                assert!(a.is_finite() && b.is_finite());
                relative = relative.max((a - b).abs() / scale);
            }
        }
        eprintln!("lfm2_moe native {workload:?}: max normalized logit error={relative:e}");
        assert!(relative <= 0.05, "{workload:?}: {relative:e}");
        for _ in 0..3 {
            let again = tiny::run(&model, &ckpt.cfg, ckpt.cfg.vocab_size, &dev, &workload)?;
            assert_eq!(again.len(), got.len());
            for (a, b) in again.iter().zip(&got) {
                assert_eq!(
                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn mixed_native_step_dispatches_the_families_in_its_walk() -> Result<()> {
    use lloom::optimizer::{census_of_labels, walk_at};
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let ckpt = TinyCheckpoint::<Lfm2MoeTiny>::template().with_dtype(DType::F16);
    let model = ckpt.model::<CrownExact>(&dev)?;
    let mut state = State::<CrownExact>::new(true, DType::F16, &ckpt.cfg, &dev)?;
    let _ = model.forward(&Tensor::new(&[[1u32, 2, 3]], &dev)?, 0, &mut state)?;
    dev.synchronize()?;
    for pos in 3..6 {
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("census window");
        let _ = model.forward(&Tensor::new(&[[pos as u32]], &dev)?, pos, &mut state)?;
        dev.synchronize()?;
        let labels = window.close();
        let ran = census_of_labels(labels.iter().map(|(label, count)| (label.as_str(), *count)));
        let walked = walk_at(&model, pos).census();
        eprintln!("lfm2_moe position={pos} walked={walked:?} ran={ran:?} labels={labels:?}");
        assert_eq!(walked, ran, "position {pos}: actual family census != Visit");
    }
    Ok(())
}

/// Original-name loading and wire policy are exercised end to end against an
/// analytic nonzero model, independently of the shared routed-block oracle.
#[test]
fn original_name_loader_and_wire_policy_match_an_analytic_model() -> Result<()> {
    use lloom::nn::VarBuilder;
    use std::collections::HashMap;
    let dev = Device::Cpu;
    let mut runtime = Lfm2MoeTiny::template();
    runtime.hidden_size = 4;
    runtime.vocab_size = 4;
    runtime.intermediate_size = 4;
    runtime.num_hidden_layers = 2;
    runtime.num_attention_heads = 1;
    runtime.num_key_value_heads = 1;
    runtime.max_position_embeddings = 8;
    runtime.layer_types = vec![LayerType::Conv; 2];
    let moe = runtime.moe.as_mut().unwrap();
    moe.num_experts = 2;
    moe.num_experts_per_tok = 1;
    moe.moe_intermediate_size = 4;
    moe.routed_scaling_factor = 1.7;
    let embeddings = vec![
        0.1f32, -0.2, 0.3, 0.4, 0.2, -0.7, 1.1, 0.4, -0.3, 0.1, 0.8, -0.5, 0.6, 0.4, -0.1, 0.2,
    ];
    let x: Vec<_> = embeddings[4..8].iter().map(|&v| v as f64).collect();
    let norm = |v: &[f64]| {
        let rms = (v.iter().map(|x| x * x).sum::<f64>() / 4.0 + runtime.norm_eps).sqrt();
        v.iter().map(|x| x / rms).collect::<Vec<_>>()
    };
    let n = norm(&x);
    let coefficient = (2.0 / n[0]) as f32;
    for use_bias in [false, true] {
        for normalize in [false, true] {
            let m = runtime.moe.as_mut().unwrap();
            m.use_expert_bias = use_bias;
            m.norm_topk_prob = normalize;
            let wire: Config =
                serde_json::from_value(lloom::models::lfm2_testing::geometry_json(&runtime))
                    .unwrap();
            let mut tensors = HashMap::new();
            for spec in schema::tensor_schema(&wire, GgmlDType::F32)?.tensors {
                let tensor = if spec.name.ends_with("norm.weight") {
                    Tensor::ones(spec.shape.as_slice(), DType::F32, &dev)?
                } else {
                    Tensor::zeros(spec.shape.as_slice(), DType::F32, &dev)?
                };
                tensors.insert(spec.name, tensor);
            }
            tensors.insert(
                "model.embed_tokens.weight".into(),
                Tensor::from_vec(embeddings.clone(), (4, 4), &dev)?,
            );
            tensors.insert(
                "model.layers.1.feed_forward.gate.weight".into(),
                Tensor::from_vec(
                    vec![coefficient, 0., 0., 0., -coefficient, 0., 0., 0.],
                    (2, 4),
                    &dev,
                )?,
            );
            if use_bias {
                tensors.insert(
                    "model.layers.1.feed_forward.expert_bias".into(),
                    Tensor::new(&[0f32, 2.], &dev)?,
                );
            }
            for expert in 0..2 {
                for (name, factor) in [
                    ("w1", expert as f32 + 1.),
                    ("w3", expert as f32 + 2.),
                    ("w2", expert as f32 + 3.),
                ] {
                    let diagonal: Vec<_> = (0..16)
                        .map(|j| if j / 4 == j % 4 { factor } else { 0. })
                        .collect();
                    tensors.insert(
                        format!("model.layers.1.feed_forward.experts.{expert}.{name}.weight"),
                        Tensor::from_vec(diagonal, (4, 4), &dev)?,
                    );
                }
            }
            let model = wire.build::<CrownExact>(
                &runtime,
                VarBuilder::from_tensors(tensors, DType::F32, &dev),
            )?;
            let mut state = State::<CrownExact>::new(true, DType::F32, &runtime, &dev)?;
            let actual = model.forward(&Tensor::new(&[[1u32]], &dev)?, 0, &mut state)?;
            let expert = usize::from(use_bias);
            let logit = coefficient as f64 * n[0] * if use_bias { -1.0 } else { 1.0 };
            let probability = 1.0 / (1.0 + (-logit).exp());
            let weight = 1.7
                * if normalize {
                    probability / (probability + 1e-6)
                } else {
                    probability
                };
            let residual: Vec<_> = x
                .iter()
                .zip(&n)
                .map(|(&x, &v)| {
                    let gate = (expert + 1) as f64 * v;
                    x + weight * gate / (1.0 + (-gate).exp())
                        * ((expert + 2) as f64 * v)
                        * (expert + 3) as f64
                })
                .collect();
            let final_norm = norm(&residual);
            let expected: Vec<_> = embeddings
                .chunks_exact(4)
                .map(|row| {
                    row.iter()
                        .zip(&final_norm)
                        .map(|(&w, &v)| w as f64 * v)
                        .sum::<f64>() as f32
                })
                .collect();
            eprintln!(
                "analytic original-name load bias={use_bias} normalize={normalize} expert={expert}"
            );
            close(&values(&actual), &expected, 1e-5);
        }
    }
    Ok(())
}

#[test]
fn mixed_batched_and_ragged_rows_keep_independent_logits_and_state() -> Result<()> {
    use lloom::nn::{RaggedRow, Row};
    let Some(fx) = Fixture::<Lfm2MoeTiny>::new("lfm2_moe_rows", Lfm2MoeTiny::template())? else {
        return Ok(());
    };
    if fx.at_scale() {
        return Ok(());
    }
    let cfg = fx.config(|_| {});
    let dev = Device::Cpu;
    let model = fx.model::<CrownExact>(&cfg, &dev)?;
    let prefixes = [vec![1u32, 2, 3], vec![4u32, 5, 6, 7, 8]];
    for ragged in [false, true] {
        let fresh = || {
            (0..2)
                .map(|_| State::<CrownExact>::new(true, DType::F32, &cfg, &dev))
                .collect::<Result<Vec<_>>>()
        };
        let mut batched = fresh()?;
        let mut single = fresh()?;
        for (row, prefix) in prefixes.iter().enumerate() {
            let input = Tensor::from_vec(prefix.clone(), (1, prefix.len()), &dev)?;
            let _ = model.forward(&input, 0, &mut batched[row])?;
            let _ = model.forward(&input, 0, &mut single[row])?;
        }
        let tails = if ragged {
            [vec![9u32, 10], vec![11u32]]
        } else {
            [vec![9u32], vec![11u32]]
        };
        let mut expected = Vec::new();
        for row in 0..2 {
            let input = Tensor::from_vec(tails[row].clone(), (1, tails[row].len()), &dev)?;
            expected.push(values(&model.forward(
                &input,
                prefixes[row].len(),
                &mut single[row],
            )?));
        }
        let flat = tails.concat();
        let actual = if ragged {
            let input = Tensor::from_vec(flat.clone(), (1, flat.len()), &dev)?;
            let mut rows: Vec<_> = batched
                .iter_mut()
                .enumerate()
                .map(|(row, cache)| RaggedRow {
                    pos: prefixes[row].len(),
                    len: tails[row].len(),
                    cache,
                })
                .collect();
            model.forward_ragged(&input, &mut rows)?
        } else {
            let input = Tensor::from_vec(flat, (2, 1), &dev)?;
            let mut rows: Vec<_> = batched
                .iter_mut()
                .enumerate()
                .map(|(row, cache)| Row {
                    pos: prefixes[row].len(),
                    cache,
                })
                .collect();
            model.forward_rows(&input, &mut rows)?
        }
        .to_vec2::<f32>()?;
        assert_eq!(actual.len(), 2);
        for row in 0..2 {
            close(&actual[row], &expected[row], 1e-4);
            // A subsequent solo step checks what the row path wrote, not
            // only the logits it returned for this call.
            let next = Tensor::new(&[[12u32]], &dev)?;
            let pos = prefixes[row].len() + tails[row].len();
            close(
                &values(&model.forward(&next, pos, &mut batched[row])?),
                &values(&model.forward(&next, pos, &mut single[row])?),
                1e-4,
            );
        }
        assert!(
            actual[0]
                .iter()
                .zip(&actual[1])
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "rows must differ"
        );
    }
    Ok(())
}
