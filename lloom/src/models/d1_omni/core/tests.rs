use super::*;
use std::collections::HashMap;

const D: usize = 64;
fn cfg() -> DecisionCoreConfig {
    DecisionCoreConfig {
        text_config: TextConfig {
            vocab_size: 8,
            hidden_size: D,
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            layer_types: vec!["conv".into(), "full_attention".into()],
            norm_eps: 1e-5,
            conv_l_cache: 3,
            block_ffn_dim_multiplier: 1.,
            block_multiple_of: 2,
            max_position_embeddings: 32,
            rope_theta: 1e6,
        },
        head_layers: 1,
        max_length: 8,
    }
}
fn matrix(out: usize, input: usize, seed: usize) -> Result<Tensor> {
    Tensor::from_vec(
        (0..out * input)
            .map(|i| (((i * 7 + seed * 3) % 17) as f32 - 8.) / 256.)
            .collect::<Vec<_>>(),
        (out, input),
        &Device::Cpu,
    )
}
fn core() -> Result<DecisionCore> {
    let mut w = HashMap::new();
    w.insert("encoder.embed_tokens.weight".into(), matrix(8, D, 0)?);
    for (name, out, input, seed) in [
        ("encoder.layers.0.conv.in_proj", 3 * D, D, 1),
        ("encoder.layers.0.conv.out_proj", D, D, 2),
        ("encoder.layers.0.feed_forward.w1", D, D, 3),
        ("encoder.layers.0.feed_forward.w3", D, D, 4),
        ("encoder.layers.0.feed_forward.w2", D, D, 5),
        ("encoder.layers.1.self_attn.q_proj", D, D, 6),
        ("encoder.layers.1.self_attn.k_proj", D, D, 7),
        ("encoder.layers.1.self_attn.v_proj", D, D, 8),
        ("encoder.layers.1.self_attn.out_proj", D, D, 9),
        ("encoder.layers.1.feed_forward.w1", D, D, 10),
        ("encoder.layers.1.feed_forward.w3", D, D, 11),
        ("encoder.layers.1.feed_forward.w2", D, D, 12),
    ] {
        w.insert(format!("{name}.weight"), matrix(out, input, seed)?);
    }
    w.insert(
        "encoder.layers.0.conv.conv.weight".into(),
        matrix(D, 3, 13)?.reshape((D, 1, 3))?,
    );
    for name in [
        "encoder.layers.0.operator_norm",
        "encoder.layers.0.ffn_norm",
        "encoder.layers.1.operator_norm",
        "encoder.layers.1.ffn_norm",
        "encoder.embedding_norm",
        "encoder.layers.1.self_attn.q_layernorm",
        "encoder.layers.1.self_attn.k_layernorm",
    ] {
        w.insert(
            format!("{name}.weight"),
            Tensor::ones(D, DType::F32, &Device::Cpu)?,
        );
    }
    w.insert("head.type_emb.weight".into(), matrix(3, D, 14)?);
    for (name, out, input, seed) in [
        ("head.head.layers.0.self_attn.in_proj", 3 * D, D, 15),
        ("head.head.layers.0.self_attn.out_proj", D, D, 16),
        ("head.head.layers.0.linear1", 4 * D, D, 17),
        ("head.head.layers.0.linear2", D, 4 * D, 18),
        ("head.scorer.1", D, D, 19),
        ("head.scorer.3", 1, D, 20),
    ] {
        let separator = if name.ends_with("in_proj") { "_" } else { "." };
        w.insert(
            format!("{name}{separator}weight"),
            matrix(out, input, seed)?,
        );
        w.insert(
            format!("{name}{separator}bias"),
            matrix(out, 1, seed + 1)?.flatten_all()?,
        );
    }
    for name in [
        "head.head.layers.0.norm1",
        "head.head.layers.0.norm2",
        "head.scorer.0",
    ] {
        w.insert(
            format!("{name}.weight"),
            Tensor::ones(D, DType::F32, &Device::Cpu)?,
        );
        w.insert(
            format!("{name}.bias"),
            Tensor::zeros(D, DType::F32, &Device::Cpu)?,
        );
    }
    DecisionCore::load(cfg(), VarBuilder::from_tensors(w, DType::F32, &Device::Cpu))
}
// Dense math is confined to this fixed tiny fixture; never a production policy.
fn head_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
    crate::nn::ops::softmax_last_dim(&scores)?.matmul(&v.contiguous()?)
}
fn trunk_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: &TrunkBatch,
    scale: f64,
) -> Result<Tensor> {
    let mut rows = Vec::new();
    for row in 0..batch.dimensions().0 {
        let mut pieces = Vec::new();
        for (index, segment) in batch
            .segments()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.batch_row == row)
        {
            pieces.push(head_attention(
                &q.narrow(0, row, 1)?
                    .narrow(2, segment.start, segment.length)?
                    .contiguous()?,
                &k.narrow(0, row, 1)?,
                &v.narrow(0, row, 1)?,
                Some(&batch.key_mask(index)?),
                scale,
            )?);
        }
        rows.push(Tensor::cat(&pieces, 2)?);
    }
    Tensor::cat(&rows, 0)
}
// Independently assembled one-row reference from the pinned _forward order.
// Does NOT call DecisionCore prepare/forward/text_rows, and has no padding.
fn per_row(core: &DecisionCore, row: &DecisionRow<'_>) -> Result<Vec<f32>> {
    let n = row.tokens.len();
    let text = core.inner.trunk.embed(row.tokens, 1, n)?;
    let (hidden, p) = if let Some(prefix) = row.prefix {
        (
            Tensor::cat(&[prefix.to_dtype(DType::F32)?, text], 1)?,
            prefix.dim(1)?,
        )
    } else {
        (text, 0)
    };
    let total = n + p;
    let trunk = core.inner.trunk.prepare_batch(total, &[total], &[p])?;
    let output = core.inner.trunk.forward_with_attention(
        &hidden,
        &trunk,
        &core.inner.cos.narrow(0, 0, total)?,
        &core.inner.sin.narrow(0, 0, total)?,
        &mut trunk_attention,
    )?;
    let text = output.narrow(1, p, n)?;
    let head = core
        .inner
        .head
        .prepare_batch(n, &[n], &[row.markers], &[row.question_type])?;
    Ok(core
        .inner
        .head
        .forward_with_attention(&text, &head, &mut head_attention)?
        .to_vec2::<f32>()?
        .remove(0))
}

#[test]
fn batching_matches_independent_rows_and_discards_nonzero_trunk_padding() -> Result<()> {
    let core = core()?;
    let prefix = matrix(1, D, 21)?.reshape((1, 1, D))?.to_dtype(DType::F64)?;
    let rows = [
        DecisionRow {
            tokens: &[1, 2, 3],
            markers: &[2, 0, 1],
            question_type: 0,
            prefix: Some(&prefix),
        },
        DecisionRow {
            tokens: &[4, 5],
            markers: &[1, 0],
            question_type: 2,
            prefix: None,
        },
    ];
    let prepared = core.prepare_rows(&rows)?;
    assert_eq!(prepared.dimensions(), (2, 4, 3));
    assert_eq!(prepared.option_counts(), [3, 2]);
    assert_eq!(prepared.input_tokens(), 6);
    assert_eq!(core.inner.cos.dims(), [8, 32]); // outer max8, NOT text max32
    let hidden = core.inner.trunk.forward_with_attention(
        &prepared.hidden,
        &prepared.trunk,
        &core.inner.cos.narrow(0, 0, 4)?,
        &core.inner.sin.narrow(0, 0, 4)?,
        &mut trunk_attention,
    )?;
    let computed_padding = hidden
        .narrow(0, 1, 1)?
        .narrow(1, 2, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        computed_padding.iter().any(|x| x.abs() > 1e-3),
        "padding negative control is vacuous"
    );
    let text = core.inner.text_rows(&hidden, &prepared)?;
    assert!(
        text.narrow(0, 1, 1)?
            .narrow(1, 2, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&x| x == 0.),
        "head must receive new zero padding, not computed trunk padding"
    );
    let expected: Vec<_> = rows
        .iter()
        .map(|r| per_row(&core, r))
        .collect::<Result<_>>()?;
    let mut previous = None;
    for _ in 0..3 {
        let actual = core
            .forward_with_attention(&prepared, &mut trunk_attention, &mut head_attention)?
            .to_vec2::<f32>()?;
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            for (option, (&a, &e)) in a.iter().zip(e).enumerate() {
                assert!(
                    (a - e).abs() < 2e-5,
                    "batched row{row} option{option}: {a} != {e}"
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
fn invalid_rows_and_foreign_prepared_embeddings_are_rejected() -> Result<()> {
    let core = core()?;
    assert!(core.prepare_rows(&[]).is_err());
    for (tokens, markers, question_type) in [
        (&[][..], &[0][..], 0),
        (&[8][..], &[0][..], 0),
        (&[1][..], &[][..], 0),
        (&[1][..], &[1][..], 0),
        (&[1][..], &[0][..], 3),
        (&[1; 9][..], &[0][..], 0),
    ] {
        assert!(core
            .prepare_rows(&[DecisionRow {
                tokens,
                markers,
                question_type,
                prefix: None
            }])
            .is_err());
    }
    for prefix in [
        Tensor::zeros((2, 1, D), DType::F32, &Device::Cpu)?,
        Tensor::zeros((1, 1, D - 1), DType::F32, &Device::Cpu)?,
        Tensor::zeros((1, 1, D), DType::U32, &Device::Cpu)?,
        Tensor::zeros((1, 8, D), DType::F32, &Device::Cpu)?,
    ] {
        assert!(core
            .prepare_rows(&[DecisionRow {
                tokens: &[0],
                markers: &[0],
                question_type: 0,
                prefix: Some(&prefix)
            }])
            .is_err());
    }
    let prepared = core.prepare_rows(&[DecisionRow {
        tokens: &[0],
        markers: &[0],
        question_type: 0,
        prefix: None,
    }])?;
    let mut foreign = core.clone();
    foreign.inner.owner = Arc::new(());
    assert!(foreign
        .forward_with_attention(&prepared, &mut trunk_attention, &mut head_attention)
        .is_err());
    assert!(core
        .clone()
        .forward_with_attention(&prepared, &mut trunk_attention, &mut head_attention)
        .is_ok());
    Ok(())
}

#[test]
fn outer_limit_rejects_before_weight_loading() {
    for limit in [0, 33] {
        let mut config = cfg();
        config.max_length = limit;
        let err = DecisionCore::load(
            config,
            VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu),
        )
        .unwrap_err();
        assert!(err.to_string().contains("outer maximum"));
    }
}
