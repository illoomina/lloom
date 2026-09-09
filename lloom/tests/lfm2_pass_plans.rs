//! The pass-keyed plan cache (`DESIGN.md` §9.2aa) on the tiny LFM2 fixture:
//! with `--pass-plans` on, a prompt's prefill and its decode steps compute
//! the same logits as the pool path, bit for bit; and once every pass shape
//! the run takes has been seen twice, a further prompt of the same shape
//! makes **zero** pool allocations — an absolute reference, not "fewer".
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::VarBuilder;
use lloom_tensor::metal_backend::{ArenaLayout, PassKind};
use lloom_tensor::{DType, Device, Result, Tensor};

fn tiny_config(conv_state: ConvState) -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: 3,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: AttnImpl::Sdpa,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state,
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

fn build(cfg: &Config, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
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
    let vb = VarBuilder::from_tensors(tensors, DType::F16, dev);
    Model::new(cfg, vb)
}

/// One prompt through the driver's own brackets — `Session::decode`, which
/// prefills `prompt` in `tile`-wide forwards and then decodes `steps` tokens,
/// exactly as `lfm2-chat` does — returning the ids it produced.
fn run_prompt(
    model: &Model,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    tile: usize,
    steps: usize,
) -> Result<Vec<u32>> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, IdentityProposer, Session};
    let cache = Cache::new(true, DType::F16, cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&[u32::MAX], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions {
        prefill_tile: tile,
        ..DecodeOptions::new(steps)
    };
    let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
    let out = session.decode(&mut spec, &mut rb, &mut sampler, prompt, &opts, &mut sink)?;
    Ok(out.ids)
}

#[test]
fn plans_serve_every_pass_shape_after_two_sightings_with_the_same_ids() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    // A multi-forward prefill onto a warm cache carries the conv context.
    cfg.conv_prefill_ctx = true;
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = (1..=10).collect();
    let other: Vec<u32> = (11..=20).collect();

    // Reference: the pool path.
    let want = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
    let want_other = run_prompt(&model, &cfg, &dev, &other, 4, 6)?;
    assert_eq!(want.len(), 6);
    assert_ne!(want, want_other, "the two prompts must decode differently");

    m.enable_pass_plans(ArenaLayout::Packed);
    let snapshot = |tag: &str| {
        let (shared, private) = m.pool_counters();
        eprintln!(
            "[{tag}] allocations={} {:?}",
            shared.allocations + private.allocations,
            m.pass_plan_counters().unwrap()
        );
    };
    // Two sightings of every shape: Prefill{4} ×2 + Prefill{2} + Decode{1} ×6 per run.
    snapshot("before run 1");
    let got1 = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
    snapshot("after run 1");
    let got2 = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
    snapshot("after run 2");
    assert_eq!(got1, want, "first planned run differs from the pool path");
    assert_eq!(got2, want, "second planned run differs from the pool path");

    let (keys, base) = m.pass_plan_report().expect("cache on");
    let kinds: Vec<PassKind> = keys.iter().map(|(k, _, _)| k.kind).collect();
    assert!(
        kinds.contains(&PassKind::Prefill { w: 4, fresh: true }),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&PassKind::Prefill { w: 4, fresh: false }),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&PassKind::Prefill { w: 2, fresh: false }),
        "{kinds:?}"
    );
    assert!(kinds.contains(&PassKind::Decode { b: 1 }), "{kinds:?}");
    assert_eq!(kinds.len(), 4, "{kinds:?}");
    assert!(base > 0);
    // Run 1 recorded the four shapes (each sighting closes the previous
    // key); run 2 was served entirely.
    let c = m.pass_plan_counters().unwrap();
    assert_eq!(c.evicted, 0, "a key drifted under its own plan: {c:?}");
    assert!(c.plans >= 4, "{c:?}");
    // Run 1: the fresh tile, the second tile, the tail and the first decode
    // step each record (the next pass closing the last), the fifth pass
    // closes the decode key, and the remaining 4 decode steps are served;
    // run 2 is served entirely.
    assert_eq!(c.served, 4 + 9, "{c:?}");

    // Served runs: the pool sees no allocation, and a different prompt of
    // the same shape reads its own values, not the last run's.
    let (shared, private) = m.pool_counters();
    let allocs_before = shared.allocations + private.allocations;
    let served_before = m.pass_plan_counters().unwrap().served;
    let got3 = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
    snapshot("after run 3");
    let got_other = run_prompt(&model, &cfg, &dev, &other, 4, 6)?;
    snapshot("after run 4");
    assert_eq!(got3, want, "served run differs from the pool path");
    assert_eq!(
        got_other, want_other,
        "served run on another prompt differs"
    );
    let (shared, private) = m.pool_counters();
    let allocs_after = shared.allocations + private.allocations;
    let c = m.pass_plan_counters().unwrap();
    eprintln!("pass plans: {c:?} keys={keys:?} base={base}");
    assert_eq!(c.served - served_before, 2 * (3 + 6), "served {c:?}");
    assert_eq!(c.evicted, 0, "{c:?}");
    assert_eq!(
        allocs_after - allocs_before,
        0,
        "two fully served prompts must allocate nothing from the pool ({c:?})"
    );
    Ok(())
}
