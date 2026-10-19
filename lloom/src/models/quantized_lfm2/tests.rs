use super::*;
use crate::nn::branch::RetainState;
use crate::nn::{CacheReset, CausalLM};
use std::io::Cursor;

#[path = "fixture.rs"]
mod synthetic;
pub(super) fn fixture() -> Result<Vec<u8>> {
    synthetic::fixture(None)
}

fn load(bytes: &[u8]) -> Result<Model> {
    let mut reader = Cursor::new(bytes);
    Model::from_gguf(
        gguf_file::Content::read(&mut reader)?,
        &mut reader,
        &Device::Cpu,
    )
}
fn run(model: &Model, cache: &mut Cache, tokens: &[u32], tile: usize) -> Result<Vec<f32>> {
    let mut result = None;
    for chunk in tokens.chunks(tile) {
        let x = Tensor::new(chunk, &Device::Cpu)?.unsqueeze(0)?;
        result = Some(CausalLM::forward(model, &x, cache.pos, cache)?);
    }
    result.expect("fixture nonempty").flatten_all()?.to_vec1()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), 8);
    assert_eq!(b.len(), 8);
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 3e-5, "{a} != {b}");
    }
}

#[test]
fn explicit_cache_continues_tiles_and_retains_long_suffixes() -> Result<()> {
    let bytes = fixture()?;
    let model = load(&bytes)?;
    for prefix in [&[][..], &[1][..], &[1, 2, 3][..]] {
        let mut cache = model.new_cache()?; // Deliberately inherent factory.
        if !prefix.is_empty() {
            run(&model, &mut cache, prefix, 2)?;
        }
        let saved = cache.retain_state()?;
        for suffix in [
            &[4, 5, 1, 2, 3, 4, 5, 6, 7][..],
            &[7, 2, 6, 1, 5, 3, 4, 0, 2][..],
            &[4, 5, 1, 2, 3, 4, 5, 6][..],
            &[7, 2, 6, 1, 5, 3, 4, 0][..],
        ] {
            cache.restore_retained(&saved)?;
            let mut consumed = 0;
            for chunk in suffix.chunks(3) {
                let actual = run(&model, &mut cache, chunk, 3)?;
                consumed += chunk.len();
                let all = prefix
                    .iter()
                    .chain(&suffix[..consumed])
                    .copied()
                    .collect::<Vec<_>>();
                let mut reference = model.new_cache()?;
                let expected = run(&model, &mut reference, &all, 1)?;
                // A full final tile can hide lost history in the logits.
                // Pin the carried window independently at every tile boundary.
                for (layer, (actual, expected)) in
                    cache.layers.iter().zip(&reference.layers).enumerate()
                {
                    if let LayerState::ShortConv(Some(b)) = expected {
                        let LayerState::ShortConv(Some(a)) = actual else {
                            panic!("carried convolution state missing layer={layer}");
                        };
                        assert_eq!(a.dims(), &[1, 4, 3]);
                        assert_eq!(a.dims(), b.dims());
                        for (slot, (a, b)) in a
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .iter()
                            .zip(b.flatten_all()?.to_vec1::<f32>()?)
                            .enumerate()
                        {
                            assert!(
                                (a - b).abs() < 3e-5,
                                "carried convolution mismatch layer={layer} slot={slot} prefix={} suffix={consumed}: {a} != {b}",
                                prefix.len(),
                            );
                        }
                    }
                }
                close(&actual, &expected);
            }
            assert_eq!(consumed, suffix.len());
        }
        cache.reset();
        assert!(cache.restore_retained(&saved).is_err());
    }
    // Legacy facade still handles a full initial prompt through Conv1d;
    // this is same-backend compatibility, not independent checkpoint parity.
    let tokens = [1, 2, 3, 4, 5];
    let mut reader = Cursor::new(&bytes);
    let mut legacy = ModelWeights::from_gguf(
        gguf_file::Content::read(&mut reader)?,
        &mut reader,
        &Device::Cpu,
    )?;
    let got = legacy
        .forward(&Tensor::new(&tokens, &Device::Cpu)?.unsqueeze(0)?, 0)?
        .flatten_all()?
        .to_vec1()?;
    close(&got, &run(&model, &mut model.new_cache()?, &tokens, 1)?);
    Ok(())
}

#[test]
fn public_cache_refuses_before_writes_and_invalidates_failed_state() -> Result<()> {
    let bytes = fixture()?;
    let model = load(&bytes)?;
    let mut cache = model.new_cache()?;
    let batch = Tensor::zeros((2, 2, 4), DType::F32, &Device::Cpu)?;
    assert!(model.forward_embeds(&batch, 0, &mut cache).is_err());
    assert_eq!(cache.pos, 0);
    assert!(!cache.failed);
    assert!(cache.masks.is_empty());
    for layer in &cache.layers {
        match layer {
            LayerState::Attention(kv) => assert!(kv.is_none()),
            LayerState::ShortConv(h) => assert!(h.is_none()),
        }
    }
    run(&model, &mut cache, &[1, 2], 1)?;
    let saved = cache.retain_state()?;
    let pos = cache.pos;
    let one = model.embed(&Tensor::new(&[[3u32]], &Device::Cpu)?)?;
    assert!(model.forward_embeds(&one, pos + 1, &mut cache).is_err());
    assert_eq!(cache.pos, pos);
    assert!(!cache.failed);
    let other = load(&bytes)?;
    let mut foreign = other.new_cache()?;
    run(&other, &mut foreign, &[1, 2], 1)?;
    assert_eq!(
        foreign.pos, saved.pos,
        "owner control must not fail on position"
    );
    assert!(foreign.restore_retained(&saved).is_err());
    assert!(
        model
            .forward_embeds(&one, foreign.pos, &mut foreign)
            .is_err(),
        "foreign model cache was accepted by forward"
    );
    assert_eq!(
        foreign.pos, saved.pos,
        "foreign refusal must precede writes"
    );
    // Simulate a partial-write failure at the cache boundary; poison refuses
    // further model work until reset, it is not transactional state rollback.
    cache.failed = true;
    assert!(model.forward_embeds(&one, pos, &mut cache).is_err());
    assert!(cache.retain_state().is_err());
    cache.reset();
    run(&model, &mut cache, &[1, 2], 1)?;
    assert_eq!(
        cache.pos, saved.pos,
        "epoch control must not fail on position"
    );
    assert!(cache.restore_retained(&saved).is_err());
    run(&model, &mut cache, &[3], 1)?;
    Ok(())
}

/// A ragged pass computes, for every row, what that row computes alone: rows
/// of different lengths, each continuing a trunk its cache adopted, in one
/// weight sweep. The logits at each row's last token match one pass over the
/// row's whole sequence, the step after the pass matches too (the row's KV
/// and conv history are its own), and the trunk stays as retained: a later
/// adopter reads it unchanged.
#[test]
fn a_ragged_pass_computes_each_rows_own_logits_and_state() -> Result<()> {
    use crate::nn::branch::AdoptRetained;
    use crate::nn::{BatchedCausalLM, RaggedRow};
    let model = load(&fixture()?)?;
    let trunk = [1u32, 2, 3];
    let mut source = model.new_cache()?;
    run(&model, &mut source, &trunk, 2)?;
    let saved = source.retain_state()?;
    let suffixes: [&[u32]; 3] = [&[4, 5], &[6], &[7, 2, 6, 1, 5]];
    let adopt = || -> Result<Cache> {
        let mut cache = model.new_cache()?;
        cache.adopt_retained(&saved)?;
        Ok(cache)
    };
    let mut caches = vec![adopt()?, adopt()?, adopt()?];
    let ids = suffixes.concat();
    let tokens = Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let mut rows = caches
        .iter_mut()
        .zip(&suffixes)
        .map(|(cache, s)| RaggedRow {
            pos: trunk.len(),
            len: s.len(),
            cache,
        })
        .collect::<Vec<_>>();
    let logits = BatchedCausalLM::forward_ragged(&model, &tokens, &mut rows)?;
    drop(rows);
    assert_eq!(logits.dims(), &[3, 8]);
    let mut spread = Vec::new();
    for (r, s) in suffixes.iter().enumerate() {
        let all = trunk.iter().chain(s.iter()).copied().collect::<Vec<_>>();
        let mut alone = model.new_cache()?;
        let expected = run(&model, &mut alone, &all, all.len())?;
        let got = logits.get(r)?.to_vec1::<f32>()?;
        close(&got, &expected);
        spread.push(got[0]);
        assert_eq!(caches[r].pos, all.len());
        close(
            &run(&model, &mut caches[r], &[3], 1)?,
            &run(&model, &mut alone, &[3], 1)?,
        );
    }
    let (lo, hi) = spread
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    assert!(
        hi - lo > 1e-3,
        "the rows' logits are too close ({spread:?}) for a row reading another's to show"
    );
    let mut late = adopt()?;
    close(
        &run(&model, &mut late, &[4, 5], 2)?,
        &logits.get(0)?.to_vec1::<f32>()?,
    );
    // Only a fresh cache of the same model adopts.
    let mut used = model.new_cache()?;
    run(&model, &mut used, &[1], 1)?;
    assert!(used.adopt_retained(&saved).is_err());
    let other = load(&fixture()?)?;
    assert!(other.new_cache()?.adopt_retained(&saved).is_err());
    Ok(())
}

#[test]
fn malformed_metadata_and_existing_alias_errors_do_not_fall_back() -> Result<()> {
    for value in [
        gguf_file::Value::I32(-1),
        gguf_file::Value::F32(2.),
        gguf_file::Value::Bool(true),
    ] {
        assert!(value_to_usize(&value).is_err());
    }
    // llama.cpp's converter writes head_count_kv as i32 (d1-3B's GGUFs).
    assert_eq!(value_to_usize(&gguf_file::Value::I32(8))?, 8);
    let bytes = fixture()?;
    let mut reader = Cursor::new(&bytes);
    let mut header = gguf_file::Content::read(&mut reader)?;
    let signed = match &header.metadata["lfm2.attention.head_count_kv"] {
        gguf_file::Value::Array(values) => gguf_file::Value::Array(
            values
                .iter()
                .map(|v| Ok(gguf_file::Value::I32(i32::try_from(v.to_u64()?).unwrap())))
                .collect::<Result<Vec<_>>>()?,
        ),
        other => gguf_file::Value::I32(i32::try_from(other.to_u64()?).unwrap()),
    };
    header
        .metadata
        .insert("lfm2.attention.head_count_kv".into(), signed);
    let model = Model::from_gguf(header, &mut reader, &Device::Cpu)?;
    let reference = load(&bytes)?;
    let mut a = model.new_cache()?;
    let mut b = reference.new_cache()?;
    close(
        &run(&model, &mut a, &[1, 2, 3], 3)?,
        &run(&reference, &mut b, &[1, 2, 3], 3)?,
    );
    let bytes = fixture()?;
    let mut reader = Cursor::new(&bytes);
    let mut header = gguf_file::Content::read(&mut reader)?;
    header
        .metadata
        .insert("lfm2.attention.head_count".into(), gguf_file::Value::U32(0));
    assert!(Model::from_gguf(header, &mut reader, &Device::Cpu)
        .err()
        .unwrap()
        .to_string()
        .contains("geometry"));
    let mut reader = Cursor::new(&bytes);
    let mut header = gguf_file::Content::read(&mut reader)?;
    let original = &header.tensor_infos["token_embd.weight"];
    let invalid = gguf_file::TensorInfo {
        shape: original.shape.clone(),
        ggml_dtype: original.ggml_dtype,
        offset: u64::MAX,
    };
    header.tensor_infos.insert("output.weight".into(), invalid);
    assert!(
        Model::from_gguf(header, &mut reader, &Device::Cpu).is_err(),
        "present malformed output was replaced with tied embeddings"
    );
    Ok(())
}

/// A block-quantized token table stays packed, is the tied head's weight,
/// and gathers exactly the rows its dequantized copy holds (§15.1p's
/// `get_rows` identity, through the model). The price follows the format.
fn packed_table_gathers_dense_rows(device: &Device) -> Result<()> {
    use lloom_tensor::quantized::GgmlDType;
    let dense_bytes = fixture()?;
    let dense = gguf_file::Content::read(&mut Cursor::new(&dense_bytes[..]))?;
    assert_eq!(dense_token_table_bytes(&dense)?, 8 * 4 * 4);

    let bytes = synthetic::fixture_table(GgmlDType::Q8_0, 32)?;
    let mut reader = Cursor::new(&bytes[..]);
    let content = gguf_file::Content::read(&mut reader)?;
    assert_eq!(dense_token_table_bytes(&content)?, 0);
    let table = content
        .tensor(&mut reader, "token_embd.weight", &Device::Cpu)?
        .dequantize(&Device::Cpu)?;
    let model = Model::from_gguf(content, &mut reader, device)?;
    match (&model.tok_embeddings, &model.output) {
        (QMatMul::QTensor(table), QMatMul::QTensor(head)) => {
            assert_eq!(table.dtype(), GgmlDType::Q8_0);
            assert!(
                Arc::ptr_eq(table, head),
                "the tied head holds a second table"
            );
        }
        _ => panic!("the Q8_0 token table was not kept packed"),
    }
    let ids = [7u32, 0, 3, 7];
    let got = CausalLM::embed(&model, &Tensor::new(&[ids], device)?)?;
    assert_eq!(got.dims(), &[1, 4, 32]);
    assert_eq!(got.dtype(), DType::F32);
    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let rows = table.to_vec2::<f32>()?;
    for (i, &id) in ids.iter().enumerate() {
        for (j, want) in rows[id as usize].iter().enumerate() {
            let have = got[i * 32 + j];
            assert_eq!(
                have.to_bits(),
                want.to_bits(),
                "token {id} column {j}: gathered {have:e}, dequantized table {want:e}"
            );
        }
    }
    Ok(())
}

#[test]
fn a_packed_token_table_gathers_the_rows_its_dense_copy_holds() -> Result<()> {
    packed_table_gathers_dense_rows(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn a_packed_token_table_gathers_the_rows_its_dense_copy_holds_on_metal() -> Result<()> {
    packed_table_gathers_dense_rows(&Device::new_metal(0)?)
}

#[test]
fn f16_kv_is_actually_half_sized_and_survives_tiles_retention_and_reset() -> Result<()> {
    let bytes = fixture()?;
    let reference = load(&bytes)?;
    let model = load(&bytes)?.with_kv_dtype(DType::F16)?;
    let mut half_first = model.new_cache()?;
    let mut full_first = reference.new_cache()?;
    run(&model, &mut half_first, &[1], 1)?;
    run(&reference, &mut full_first, &[1], 1)?;
    for (half, full) in half_first.layers.iter().zip(&full_first.layers) {
        if let (LayerState::Attention(Some((hk, hv))), LayerState::Attention(Some((fk, fv)))) =
            (half, full)
        {
            for (h, f) in [(hk, fk), (hv, fv)] {
                assert_eq!(
                    h.flatten_all()?.to_vec1::<half::f16>()?,
                    f.to_dtype(DType::F16)?
                        .flatten_all()?
                        .to_vec1::<half::f16>()?,
                    "stored values must be exactly f16-rounded projections"
                );
            }
        }
    }
    let tokens = [1, 2, 3, 4, 5];
    let mut cache = model.new_cache()?;
    let got = run(&model, &mut cache, &tokens, 2)?;
    let mut full = reference.new_cache()?;
    let expected = run(&reference, &mut full, &tokens, 2)?;
    assert!(
        got.iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 0.005),
        "{got:?} != {expected:?}"
    );
    let mut stored = 0;
    for (half, full) in cache.layers.iter().zip(&full.layers) {
        if let (LayerState::Attention(Some((hk, hv))), LayerState::Attention(Some((fk, fv)))) =
            (half, full)
        {
            for (h, f) in [(hk, fk), (hv, fv)] {
                assert_eq!(h.dtype(), DType::F16);
                assert_eq!(h.dims(), f.dims());
                assert_eq!(
                    2 * h.elem_count() * h.dtype().size_in_bytes(),
                    f.elem_count() * f.dtype().size_in_bytes()
                );
                stored += h.elem_count();
            }
        }
    }
    assert!(stored > 0, "fixture must exercise real attention stores");
    let saved = cache.retain_state()?;
    let first = run(&model, &mut cache, &[6, 7], 2)?;
    cache.restore_retained(&saved)?;
    assert_eq!(first, run(&model, &mut cache, &[6, 7], 2)?);
    cache.reset();
    assert_eq!(got, run(&model, &mut cache, &tokens, 2)?);
    assert!(load(&bytes)?.with_kv_dtype(DType::BF16).is_err());
    Ok(())
}

#[test]
fn context_limit_controls_actual_rope_and_sequence_bounds() -> Result<()> {
    let bytes = fixture()?;
    let mut file = Cursor::new(&bytes);
    let header = gguf_file::Content::read(&mut file)?;
    let plan = Model::preflight_gguf_content(&header, &mut file)?.with_context_limit(3)?;
    assert_eq!(plan.config().context_length, 3);
    let model = plan.materialize(&mut file, &Device::Cpu)?;
    for layer in &model.layers {
        if let LayerKind::Attention(attn) = &layer.kind {
            assert_eq!(attn.cos.dims()[0], 3);
        }
    }
    let mut cache = model.new_cache()?;
    let actual = run(&model, &mut cache, &[1, 2, 3], 2)?;
    let native = load(&bytes)?;
    assert_eq!(
        actual,
        run(&native, &mut native.new_cache()?, &[1, 2, 3], 2)?,
        "shortening the context must preserve prefix logits"
    );
    assert!(run(&model, &mut cache, &[4], 1).is_err());
    assert_eq!(cache.pos, 3);
    assert!(!cache.failed, "overflow must refuse before writes");
    assert!(Model::preflight_gguf_content(&header, &mut file)?
        .with_context_limit(0)
        .is_err());
    assert!(Model::preflight_gguf_content(&header, &mut file)?
        .with_context_limit(usize::MAX)
        .is_err());
    Ok(())
}

/// The optional store changes no projection/attention dtype on Metal either.
/// Compare both stores to the same CPU reference at a small real GGUF layout.
#[test]
#[cfg(feature = "metal")]
fn metal_dense_kv_stores_match_the_cpu_reference() -> Result<()> {
    let bytes = synthetic::fixture_table(lloom_tensor::quantized::GgmlDType::F32, 32)?;
    let gpu = Device::new_metal(0)?;
    let make = |device: &Device, dtype| -> Result<Model> {
        let mut file = Cursor::new(&bytes);
        Model::from_gguf(gguf_file::Content::read(&mut file)?, &mut file, device)?
            .with_kv_dtype(dtype)
    };
    let drive = |model: &Model| -> Result<(Vec<f32>, Cache)> {
        let mut cache = model.new_cache()?;
        let mut logits = None;
        for chunk in [&[1u32, 2][..], &[3, 4][..], &[5][..]] {
            let input = Tensor::new(chunk, &model.device)?.unsqueeze(0)?;
            logits = Some(CausalLM::forward(model, &input, cache.pos, &mut cache)?);
        }
        model.device.synchronize()?;
        Ok((logits.unwrap().flatten_all()?.to_vec1()?, cache))
    };
    for dtype in [DType::F32, DType::F16] {
        let cpu = make(&Device::Cpu, dtype)?;
        let metal = make(&gpu, dtype)?;
        let (expected, _) = drive(&cpu)?;
        let (got, cache) = drive(&metal)?;
        for (a, b) in got.iter().zip(&expected) {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() < 1e-3,
                "{dtype:?}: {got:?} != {expected:?}"
            );
        }
        assert_eq!(got, drive(&metal)?.0, "run-to-run identity {dtype:?}");
        let mut stores = 0;
        for layer in &cache.layers {
            if let LayerState::Attention(Some((k, v))) = layer {
                assert_eq!(k.dtype(), dtype);
                assert_eq!(v.dtype(), dtype);
                stores += 1;
            }
        }
        assert!(stores > 0);
    }
    Ok(())
}
