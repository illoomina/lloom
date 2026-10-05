//! KV growth on the arms only a Metal device has (lloom #696): the paged pool
//! and the quantized planes. Each is compared against the same model on the
//! same arm with a bucket wide enough that no grow fires, so the only thing
//! that differs is the number of reallocations — and the stream must not.
//!
//! The memory refusal is `lfm2_kv_growth_refusal_metal.rs`, in its own
//! process, because it reads the process's device allocation.

#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::ops::KvQuant;
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

fn device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// A tiny LFM2 with LFM2's head geometry — `head_dim = 64`, one KV head —
/// because the fused kernels instantiate that width and nothing narrower.
fn config(kv_append: KvAppend, attn_impl: AttnImpl, kv_quant: KvQuant) -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 128,
        intermediate_size: 32,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 2048,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        attn_impl,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        kv_append,
        kv_quant,
        conv_state: ConvState::Shuffle,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

fn build(cfg: &Config, dtype: DType, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(v, shape, dev)?.to_dtype(dtype)?,
        );
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, dtype, dev);
    Model::new(cfg, vb)
}

const PROMPT: [u32; 5] = [3, 11, 20, 5, 9];

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn decode(model: &Model, cache: &mut Cache, steps: usize, dev: &Device) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(steps + 1);
    let input = Tensor::new(&PROMPT, dev)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0, cache)?.i(0)?.to_vec1::<f32>()?;
    let mut pos = PROMPT.len();
    let mut next = argmax(&logits);
    out.push(logits);
    for _ in 0..steps {
        let input = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos, cache)?.i(0)?.to_vec1::<f32>()?;
        pos += 1;
        next = argmax(&logits);
        out.push(logits);
    }
    Ok(out)
}

fn assert_same_stream(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: step counts differ");
    for (step, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x == y,
            "{what}: logits differ at step {step} (first differing index {:?})",
            x.iter().zip(y).position(|(p, q)| p != q)
        );
    }
    assert!(
        !a.is_empty() && a[0].iter().any(|v| *v != 0.0),
        "{what}: an empty stream"
    );
}

/// The paged pool, grown a page-bucket at a time, against one sized to the
/// ceiling from the start: bit-identical logits over two grows.
#[test]
fn a_paged_pool_grown_by_pages_matches_one_sized_upfront() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let cfg = config(KvAppend::Paged, AttnImpl::FlashDecoding, KvQuant::Off);
    let model = build(&cfg, DType::F32, &dev)?;
    let steps = 600;
    let mut wide = Cache::new(true, DType::F32, &cfg, &dev)?;
    let mut narrow = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(256)?;
    let wide_stream = decode(&model, &mut wide, steps, &dev)?;
    let narrow_stream = decode(&model, &mut narrow, steps, &dev)?;
    assert_same_stream(
        &wide_stream,
        &narrow_stream,
        "paged, bucket 256 against a pool sized upfront",
    );
    let (w, n) = (wide.kv_report(), narrow.kv_report());
    assert_eq!(
        w.grows, 0,
        "the wide pool must never grow, or the reference is not one"
    );
    assert_eq!(
        n.grows,
        2,
        "256 -> 512 -> 768 over {} positions",
        PROMPT.len() + steps
    );
    assert_eq!(n.kv_len, PROMPT.len() + steps);
    assert_eq!(n.slab, 768);
    assert_eq!(w.slab, 2048);
    assert!(
        n.bytes < w.bytes,
        "a grown pool holds what it uses: {} < {}",
        n.bytes,
        w.bytes
    );
    let (tokens, in_use, _) = narrow.paged_stats().expect("the paged arm");
    assert_eq!(tokens, PROMPT.len() + steps);
    assert_eq!(in_use, 3, "three pages hold 605 positions at 256 per page");
    Ok(())
}

/// The quantized planes — codes, scales and biases, each re-laid at a wider
/// stride — grown against planes that never grew: bit-identical logits.
#[test]
fn quantized_planes_grown_by_buckets_match_planes_sized_upfront() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    for quant in [KvQuant::Int8, KvQuant::Int4] {
        let cfg = config(KvAppend::InPlace, AttnImpl::Sdpa, quant);
        let model = build(&cfg, DType::F16, &dev)?;
        let steps = 300;
        let mut wide = Cache::new(true, DType::F16, &cfg, &dev)?;
        let mut narrow = Cache::new(true, DType::F16, &cfg, &dev)?.with_kv_bucket(64)?;
        let wide_stream = decode(&model, &mut wide, steps, &dev)?;
        let narrow_stream = decode(&model, &mut narrow, steps, &dev)?;
        assert_same_stream(
            &wide_stream,
            &narrow_stream,
            &format!("{quant:?}, bucket 64 against planes sized upfront"),
        );
        let (w, n) = (wide.kv_report(), narrow.kv_report());
        assert_eq!(w.grows, 0);
        assert_eq!(
            n.grows, 4,
            "{quant:?}: 64 -> 128 -> 192 -> 256 -> 320 over 305 positions"
        );
        assert_eq!(n.slab, 320);
        assert_eq!(n.kv_len, PROMPT.len() + steps);
        assert!(n.bytes < w.bytes, "{quant:?}: {} < {}", n.bytes, w.bytes);
    }
    Ok(())
}

/// A pass of `len` tokens from position 0, as a turn's prefill.
fn prefill(model: &Model, cache: &mut Cache, len: usize, dev: &Device) -> Result<Vec<f32>> {
    let ids: Vec<u32> = (0..len as u32).map(|i| (i * 7 + 3) % 48).collect();
    let input = Tensor::new(ids.as_slice(), dev)?.unsqueeze(0)?;
    model.forward(&input, 0, cache)?.i(0)?.to_vec1::<f32>()
}

/// A cleared paged cache refilled before the GPU released the previous turn's
/// pages grows the pool rather than refusing (lloom #892 A1, #859 finding 3).
///
/// `Cache::clear` quarantines the turn's pages until the epoch open at the
/// clear completes, and nothing commits between a chat's turns: the next
/// turn's first pass finds them held. A pool sized by positions alone (two
/// pages, one turn's worth) refuses that pass with "pool exhausted" although
/// the sequence is empty. The plan has to count the pages the pass can take
/// — free, or drainable — and widen the pool by the shortfall.
///
/// The reference is a fresh cache on the same model (a second mechanism, not
/// the grown pool against itself): the refilled turn's logits must equal a
/// first turn's, so the grow moved bytes and not math. Non-vacuity: the pages
/// are asserted held at the refill, and the refill is asserted to have grown.
#[test]
fn a_cleared_paged_pool_refilled_under_quarantine_grows_instead_of_refusing() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let cfg = config(KvAppend::Paged, AttnImpl::FlashDecoding, KvQuant::Off);
    let model = build(&cfg, DType::F32, &dev)?;
    // Two pages of 256: one turn of 300 positions holds both.
    let turn = 300;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(512)?;
    let first = prefill(&model, &mut cache, turn, &dev)?;
    let (_, in_use, _) = cache.paged_stats().expect("the paged arm");
    assert_eq!(in_use, 2, "300 positions take both pages of 256");
    cache.clear();
    let (len, _, held) = cache.paged_stats().expect("the paged arm");
    assert_eq!(len, 0, "cleared");
    assert_eq!(
        held, 2,
        "the turn's pages are quarantined until the open epoch completes; nothing has \
         committed since the clear, so the refill finds them held (the premise of this test)"
    );
    let second = prefill(&model, &mut cache, turn, &dev).map_err(|e| {
        lloom_tensor::Error::msg(format!("the refill after a clear was refused: {e}"))
    })?;
    let mut fresh = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(512)?;
    let reference = prefill(&model, &mut fresh, turn, &dev)?;
    assert!(
        first == reference,
        "two first turns on fresh caches disagree: the model is not deterministic here"
    );
    assert!(
        second == reference,
        "the refilled turn's logits differ from a fresh cache's (first differing index {:?})",
        second.iter().zip(&reference).position(|(p, q)| p != q)
    );
    let r = cache.kv_report();
    assert_eq!(
        r.grows, 1,
        "the refill grew the pool once, by the pages the quarantine held"
    );
    assert_eq!(r.kv_len, turn);
    let (_, in_use, _) = cache.paged_stats().expect("the paged arm");
    assert_eq!(in_use, 2, "the refilled turn holds two pages");
    Ok(())
}
