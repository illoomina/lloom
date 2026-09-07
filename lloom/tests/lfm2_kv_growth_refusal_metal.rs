//! The positive control for the KV grow's memory refusal (lloom #696): a grow
//! that would take the process past its fraction of
//! `recommendedMaxWorkingSetSize` is refused — an `Err` naming the numbers,
//! not a panic and not a truncation — and it moves nothing, so the same call
//! succeeds once the memory is back and produces the stream an unrefused run
//! does.
//!
//! The refusal is forced honestly rather than by a stub: a raw 256 MiB device
//! buffer is held across the pass while the fraction is set so the budget lies
//! between the process's allocation with and without it. That is the case the
//! check exists for — something the cache cannot see ate the memory — and the
//! control (the same pass with nothing held) is what shows the refusal was the
//! hog's and not the arithmetic's. The buffer is a raw device allocation, not
//! a pooled one, because the pool retains what it frees and the process's
//! allocation would not fall when the hog is dropped.
//!
//! Own test binary: `currentAllocatedSize` is process-wide, and a sibling test
//! allocating in another thread would move the figure this reads.

#![cfg(feature = "metal")]

use lloom::models::lfm2::{kv_grow_budget, Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

fn tiny_config() -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
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
        kv_append: KvAppend::InPlace,
        // The rotating ring: its phase advances once per pass, so a refusal
        // that came after the advance is visible where the shuffle's would not be.
        conv_state: ConvState::RotatingRing { k: 2 },
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
const HOG_BYTES: usize = 256 * 1024 * 1024;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn step(
    model: &Model,
    cache: &mut Cache,
    token: u32,
    pos: usize,
    dev: &Device,
) -> Result<Vec<f32>> {
    let input = Tensor::new(&[token], dev)?.unsqueeze(0)?;
    model.forward(&input, pos, cache)?.i(0)?.to_vec1::<f32>()
}

#[test]
fn a_grow_past_the_fraction_is_refused_without_moving_state_and_admitted_once_memory_returns(
) -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Device::Metal(metal) = &dev else {
        unreachable!()
    };
    let raw = metal.device().clone();
    let cfg = tiny_config();
    let model = build(&cfg, DType::F32, &dev)?;

    // The budget: halfway between the process's allocation now and that plus
    // the hog. A grow of this model is a few hundred bytes, so the hog decides.
    let working_set = raw.recommended_max_working_set_size();
    let baseline = raw.current_allocated_size();
    let fraction = (baseline + HOG_BYTES / 2) as f64 / working_set as f64;
    let budget = kv_grow_budget(working_set, fraction);
    assert!(budget > baseline && budget < baseline + HOG_BYTES);

    // The subject: bucket 2, so the pass after the prefill's grow has to grow
    // again — the one this test holds the memory against.
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?
        .with_kv_bucket(2)?
        .with_kv_grow_fraction(fraction)?;
    let input = Tensor::new(&PROMPT, &dev)?.unsqueeze(0)?;
    let first = model
        .forward(&input, 0, &mut cache)?
        .i(0)?
        .to_vec1::<f32>()?;
    let next = argmax(&first);
    let prefilled = cache.kv_report();
    assert_eq!(prefilled.slab, 6, "the prefill grew the slab to 6");
    let mut pos = PROMPT.len();

    // Step 1 fits the slab (6 = 5 + 1): admitted with nothing checked.
    let step1 = step(&model, &mut cache, next, pos, &dev)?;
    pos += 1;
    let next2 = argmax(&step1);
    let before = cache.kv_report();
    let phase = cache.conv_phase();

    // Step 2 needs a grow (7 > 6). Hold the memory and ask.
    let hog = raw
        .new_buffer(HOG_BYTES, lloom_tensor::kernels::RESOURCE_OPTIONS)
        .map_err(lloom_tensor::Error::wrap)?;
    let held = raw.current_allocated_size();
    assert!(
        held >= baseline + HOG_BYTES,
        "the device does not account the hog: {held} < {baseline} + {HOG_BYTES}"
    );
    let err = step(&model, &mut cache, next2, pos, &dev)
        .err()
        .expect("a grow past the fraction was admitted with 256 MiB held")
        .to_string();
    assert!(err.contains("KV grow refused"), "{err}");
    assert!(
        err.contains("from 6 to 8 tokens"),
        "the message names the slab it could not widen: {err}"
    );
    assert_eq!(
        cache.conv_phase(),
        phase,
        "the conv phase moved on a refusal"
    );
    assert_eq!(cache.kv_report(), before, "the slab moved on a refusal");
    // And again: the refusal is idempotent while the memory is held.
    let err = step(&model, &mut cache, next2, pos, &dev)
        .err()
        .expect("a second attempt was admitted with the memory still held")
        .to_string();
    assert!(err.contains("KV grow refused"), "{err}");
    assert_eq!(cache.kv_report(), before);

    // Free the memory: the same call is admitted, and the slab is 8.
    drop(hog);
    let released = raw.current_allocated_size();
    assert!(
        released < budget,
        "the hog did not release: {released} against a budget of {budget}"
    );
    let step2 = step(&model, &mut cache, next2, pos, &dev)?;
    assert_eq!(cache.kv_report().slab, 8);
    assert_eq!(cache.kv_report().grows, before.grows + 1);

    // The control: the same model and tokens with nothing ever held. The
    // refused-then-admitted run emitted exactly its stream — the refusal left
    // no trace in the cache.
    let mut control = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(2)?;
    let c_first = model
        .forward(&input, 0, &mut control)?
        .i(0)?
        .to_vec1::<f32>()?;
    let c_step1 = step(&model, &mut control, next, PROMPT.len(), &dev)?;
    let c_step2 = step(&model, &mut control, next2, PROMPT.len() + 1, &dev)?;
    assert!(c_first == first && c_step1 == step1 && c_step2 == step2);
    let mut tok = argmax(&step2);
    let mut pos = PROMPT.len() + 2;
    for _ in 0..20 {
        let a = step(&model, &mut cache, tok, pos, &dev)?;
        let b = step(&model, &mut control, tok, pos, &dev)?;
        assert!(
            a == b,
            "the streams diverge after the refusal at position {pos}"
        );
        tok = argmax(&a);
        pos += 1;
    }
    assert_eq!(cache.kv_report(), control.kv_report());
    Ok(())
}
