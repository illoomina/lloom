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

#[test]
fn malformed_metadata_and_existing_alias_errors_do_not_fall_back() -> Result<()> {
    for value in [
        gguf_file::Value::I32(-1),
        gguf_file::Value::F32(2.),
        gguf_file::Value::Bool(true),
    ] {
        assert!(value_to_usize(&value).is_err());
    }
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
