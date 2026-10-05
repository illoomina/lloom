//! The pass-keyed plan cache (`DESIGN.md` §9.2aa) on the tiny LFM2 fixture:
//! with `--pass-plans` on, a prompt's prefill and its decode steps compute
//! the same logits as the pool path, bit for bit; and once every pass shape
//! the run takes has been seen twice, the passes of a further prompt of the
//! same shape take every activation from their plans, asking the pools only
//! for what no plan serves — an exact count, not "fewer".
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::VarBuilder;
use lloom_tensor::metal_backend::{ArenaLayout, PassKind};
use lloom_tensor::{DType, Device, Result, Tensor};

/// The recorder's bind observer and the dispatch clock are process globals
/// (`arena.rs`), so two tests planning passes at once record each other's
/// dispatches: every test here holds this for its whole body.
///
/// `parking_lot`, as every lock in this workspace is: infallible, so there is
/// no poison `Result` to unwrap, and no `pthread_rwlock_t` behind it
/// (`lloom-tensor/Cargo.toml`).
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

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
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    // A multi-forward prefill onto a warm cache carries the conv context.
    cfg.conv_prefill_ctx = true;
    let tile = 4;
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = (1..=10).collect();
    let other: Vec<u32> = (11..=20).collect();

    // Reference: the pool path.
    let want = run_prompt(&model, &cfg, &dev, &prompt, tile, 6)?;
    let want_other = run_prompt(&model, &cfg, &dev, &other, tile, 6)?;
    assert_eq!(want.len(), 6);
    assert_ne!(want, want_other, "the two prompts must decode differently");
    eprintln!("[pool reference] want={want:?} want_other={want_other:?}");

    m.enable_pass_plans(ArenaLayout::Packed);
    let snapshot = |tag: &str| {
        let (shared, private) = m.pool_counters();
        eprintln!(
            "[{tag}] allocations={} served-pass requests={:?} {:?} keys={:?}",
            shared.allocations + private.allocations,
            m.served_pass_pool_requests(),
            m.pass_plan_counters().unwrap(),
            m.pass_plan_report().map(|(k, _)| k).unwrap_or_default()
        );
    };
    // Two sightings of every shape: Prefill{4} ×2 + Prefill{2} + Decode{1} ×6 per run.
    snapshot("before run 1");
    let got1 = run_prompt(&model, &cfg, &dev, &prompt, tile, 6)?;
    snapshot("after run 1");
    let got2 = run_prompt(&model, &cfg, &dev, &prompt, tile, 6)?;
    snapshot("after run 2");
    assert_eq!(got1, want, "first planned run differs from the pool path");
    assert_eq!(got2, want, "second planned run differs from the pool path");

    let (keys, base) = m.pass_plan_report().expect("cache on");
    let kinds: Vec<PassKind> = keys.iter().map(|(k, ..)| k.kind).collect();
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
    // The ring conv arm writes its state in place and the held logits are
    // copied out: nothing a pass produced was read by the next one undeclared.
    assert_eq!(
        c.escapes, 0,
        "an activation outlived its pass undeclared: {c:?}"
    );
    // Run 1: the fresh tile, the second tile, the tail and the first decode
    // step each record (the next pass closing the last), the fifth pass
    // closes the decode key, and the remaining 4 decode steps are served;
    // run 2 is served entirely.
    assert_eq!(c.served, 4 + 9, "{c:?}");

    // Served runs: every activation of their passes comes from the plans, and
    // a different prompt of the same shape reads its own values, not the last
    // run's.
    //
    // The pool's own `allocations` is not that claim. A request made between
    // passes (the staged ids, the next prompt's cache) hits or misses
    // according to whether the completion handler has drained its same-size
    // predecessor yet: the GPU's timing, not the plan's coverage (§6.3b). The
    // trim empties both free lists so those requests allocate on every run;
    // the served-pass count must not see them.
    //
    // What a served pass does ask the pool for is exact, and all of it is
    // `Tensor::zeros`, which the Shared pool serves and no plan does: the
    // sequence's session state on the fresh tile (each conv layer's carry,
    // each attention layer's K and V, and the turn-checkpoint ring — one
    // allocation for every slot and layer, `nn::ring::Ring`), and nothing on
    // a later tile, since the carry's padding is set in place in the state
    // (`pass-plan-design.md` §18) and the ring's slots are written in place.
    // The private pool, which plans serve, is asked nothing; nor is any pool
    // by a decode step.
    let conv = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::Conv))
        .count() as u64;
    let attn = cfg.layer_types.len() as u64 - conv;
    let turn_ring = 1;
    let per_prompt = (conv + 2 * attn + turn_ring, 0);
    let served_before = m.pass_plan_counters().unwrap().served;
    let requests_before = m.served_pass_pool_requests();
    m.trim_unused_buffers();
    let (shared, private) = m.pool_counters();
    let allocs_before = shared.allocations + private.allocations;
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
    let requests = m.served_pass_pool_requests();
    let requests = (
        requests.0 - requests_before.0,
        requests.1 - requests_before.1,
    );
    let c = m.pass_plan_counters().unwrap();
    eprintln!(
        "pass plans: {c:?} keys={keys:?} base={base} pool allocations after the trim={} \
         served-pass pool requests={requests:?}",
        allocs_after - allocs_before,
    );
    assert_eq!(c.served - served_before, 2 * (3 + 6), "served {c:?}");
    assert_eq!(c.evicted, 0, "{c:?}");
    assert_eq!(
        requests,
        (2 * per_prompt.0, 2 * per_prompt.1),
        "(shared, private) pool requests made inside the two prompts' served passes: \
         beyond the session state, a served pass allocates from its plan alone ({c:?})"
    );
    assert!(
        allocs_after > allocs_before,
        "nothing allocated after the trim: the runs' between-pass requests never reached \
         the pool, so the count above was not tested against any traffic"
    );
    Ok(())
}

/// The served-pass pool count `plans_serve_every_pass_shape_…` pins counts
/// exactly the requests a SERVED pass makes: one host upload and one
/// op past the plan's last ordinal, made inside a served pass, read `(1, 1)`;
/// the same two made between passes, and every request of the recording
/// passes, read nothing — though the pool saw them, which the lookups show.
#[test]
fn a_served_pass_is_charged_its_own_pool_requests_and_nothing_else() -> Result<()> {
    use lloom::nn::CausalLM;
    use lloom_tensor::metal_backend::PassKey;
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    cfg.conv_prefill_ctx = true;
    let model = build(&cfg, &dev)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    let lookups = || {
        let (shared, private) = m.pool_counters();
        (shared.lookups, private.lookups)
    };
    // One of each pool's requests: a host upload (shared) and an op on a
    // pass's output (private; inside a served pass, past the plan's end).
    let extras = |out: &Tensor| -> Result<()> {
        let _ids = Tensor::new(&[1u32, 2, 3], &dev)?;
        let _scaled = out.affine(2.0, 0.0)?;
        Ok(())
    };

    let mut cache = Cache::new(true, DType::F16, &cfg, &dev)?;
    let prompt = Tensor::new(&[1u32, 2, 3, 4], &dev)?.unsqueeze(0)?;
    model.prepare_pass(&mut cache, 4)?;
    m.begin_pass(PassKey::new(PassKind::Prefill { w: 4, fresh: true }))?;
    model.forward(&prompt, 0, &mut cache)?;
    m.finish_pass()?;
    // `(shared, private)`, as `served_pass_pool_requests` and `pool_counters`.
    type Pools = (u64, u64);
    let mut pos = 4usize;
    let mut decode = |extra_inside: bool| -> Result<(bool, Pools, Pools)> {
        let next = Tensor::new(&[5u32], &dev)?.unsqueeze(0)?;
        model.prepare_pass(&mut cache, 1)?;
        let served = m.pass_plan_counters().unwrap().served;
        let (req, look) = (m.served_pass_pool_requests(), lookups());
        m.begin_pass(PassKey::new(PassKind::Decode { b: 1 }))?;
        let out = model.forward(&next, pos, &mut cache)?;
        if extra_inside {
            extras(&out)?;
        }
        m.finish_pass()?;
        pos += 1;
        let was_served = m.pass_plan_counters().unwrap().served > served;
        let (r, l) = (m.served_pass_pool_requests(), lookups());
        Ok((
            was_served,
            (r.0 - req.0, r.1 - req.1),
            (l.0 - look.0, l.1 - look.1),
        ))
    };

    // Decode{1}'s first sight records and its second closes the plan: both
    // run on the pool, and neither is counted.
    for sight in 0..2 {
        let (served, counted, looked) = decode(false)?;
        assert!(!served, "sight {sight} was served before its plan existed");
        assert_eq!(
            counted,
            (0, 0),
            "sight {sight}: a recording pass was counted"
        );
        assert!(
            looked.1 > 0,
            "sight {sight}: a recording pass asked the private pool nothing"
        );
    }
    let (served, counted, _) = decode(false)?;
    assert!(served, "the third sight was not served");
    assert_eq!(counted, (0, 0), "a served pass with nothing extra");
    let (served, counted, _) = decode(true)?;
    assert!(served, "the fourth sight was not served");
    assert_eq!(
        counted,
        (1, 1),
        "one upload and one op past the plan, inside a served pass, are (shared, private)"
    );

    // The same two requests between passes reach the pools and are not counted.
    let out = Tensor::zeros((1, 4), DType::F16, &dev)?;
    let (req, look) = (m.served_pass_pool_requests(), lookups());
    extras(&out)?;
    let (r, l) = (m.served_pass_pool_requests(), lookups());
    assert_eq!(
        (l.0 - look.0, l.1 - look.1),
        (1, 1),
        "the extras' own requests"
    );
    assert_eq!(r, req, "a request between passes was counted");
    assert_eq!(m.pass_plan_counters().unwrap().evicted, 0);
    Ok(())
}

/// Decode `prompt` on `session` — an empty prompt resumes from its held
/// logits — and return the ids.
fn decode_on(
    session: &mut lloom::nn::Session<'_, Model>,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<u32>> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, IdentityProposer};
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&[u32::MAX], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(steps);
    let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
    let out = session.decode(&mut spec, &mut rb, &mut sampler, prompt, &opts, &mut sink)?;
    Ok(out.ids)
}

/// Session A decodes and holds its last logits; session B runs its own
/// passes on the same device; A resumes from what it held. Returns A's
/// ids over both calls and B's.
fn interleaved(
    model: &Model,
    cfg: &Config,
    dev: &Device,
    a_prompt: &[u32],
    b_prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    use lloom::nn::Session;
    let mut a = Session::new(model, Cache::new(true, DType::F16, cfg, dev)?);
    let mut b = Session::new(model, Cache::new(true, DType::F16, cfg, dev)?);
    let mut ids_a = decode_on(&mut a, dev, a_prompt, steps)?;
    let ids_b = decode_on(&mut b, dev, b_prompt, steps)?;
    ids_a.extend(decode_on(&mut a, dev, &[], steps)?);
    Ok((ids_a, ids_b))
}

/// Logits held across passes are read after their slot was reused: A's
/// resume follows B's passes, which under the plan cache run over the same
/// base A's last decode step did. The hold (`Session::hold_logits`) copies
/// them into memory the sequence owns, so A resumes with the ids the pool
/// path produces — on the recording passes and on the served ones.
#[test]
fn held_logits_survive_another_sessions_passes() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;
    let a_prompt: Vec<u32> = (1..=10).collect();
    let b_prompt: Vec<u32> = (11..=20).collect();

    let (want_a, want_b) = interleaved(&model, &cfg, &dev, &a_prompt, &b_prompt, 6)?;
    assert_eq!(want_a.len(), 12);
    assert_ne!(
        want_a[..6],
        want_b[..6],
        "the two prompts must decode differently"
    );

    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..3 {
        let (got_a, got_b) = interleaved(&model, &cfg, &dev, &a_prompt, &b_prompt, 6)?;
        assert_eq!(got_b, want_b, "round {round}: B differs from the pool path");
        assert_eq!(
            got_a, want_a,
            "round {round}: A's resume read logits another session's passes overwrote"
        );
    }
    let c = m.pass_plan_counters().unwrap();
    assert!(c.served > 0, "{c:?}");
    assert_eq!(c.evicted, 0, "{c:?}");
    assert_eq!(c.escapes, 0, "{c:?}");
    Ok(())
}

/// Every pass of a tiled prompt and its decode computes the same logits
/// under the plan cache as on the pool, bit for bit, on the recording
/// round and on two served rounds. The served rounds are the gate: a
/// prefill tile's conv carry is read by the next tile, and read through a
/// blit `copy2d` from a served slot it was wrong from the second tile on
/// (every later pass's logits differed — the blit encoder is outside the
/// compute barrier chain), which is why `MetalDevice::base_mode` routes
/// copies and fills through compute kernels.
#[test]
fn served_tiles_compute_the_pool_paths_logits_bit_for_bit() -> Result<()> {
    use lloom::nn::CausalLM;
    use lloom_tensor::metal_backend::PassKey;
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    cfg.conv_prefill_ctx = true;
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = (1..=10).collect();
    let widths: Vec<usize> = vec![4, 4, 2];
    let run = |bracket: bool| -> Result<Vec<Vec<f32>>> {
        let mut cache = Cache::new(true, DType::F16, &cfg, &dev)?;
        let mut outs = Vec::new();
        let mut pos = 0usize;
        for &w in &widths {
            let input = Tensor::new(&prompt[pos..pos + w], &dev)?.unsqueeze(0)?;
            model.prepare_pass(&mut cache, w)?;
            if bracket {
                m.begin_pass(PassKey::new(PassKind::Prefill {
                    w: w as u32,
                    fresh: pos == 0,
                }))?;
            }
            let out = model.forward(&input, pos, &mut cache)?;
            if bracket {
                m.finish_pass()?;
            }
            outs.push(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
            pos += w;
        }
        for t in 0..3u32 {
            let next = Tensor::new(&[7u32 + t], &dev)?.unsqueeze(0)?;
            model.prepare_pass(&mut cache, 1)?;
            if bracket {
                m.begin_pass(PassKey::new(PassKind::Decode { b: 1 }))?;
            }
            let out = model.forward(&next, pos, &mut cache)?;
            if bracket {
                m.finish_pass()?;
            }
            outs.push(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
            pos += 1;
        }
        Ok(outs)
    };
    let want = run(false)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..3 {
        let got = run(true)?;
        let diffs: Vec<usize> = want
            .iter()
            .zip(&got)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .collect();
        let c = m.pass_plan_counters().unwrap();
        eprintln!("[round {round}] per-pass differing logits {diffs:?} {c:?}");
        assert!(
            diffs.iter().all(|&d| d == 0),
            "round {round}: a pass under the plan cache computed different logits: {diffs:?}"
        );
        assert_eq!(c.evicted, 0, "{c:?}");
        assert_eq!(c.escapes, 0, "{c:?}");
    }
    let c = m.pass_plan_counters().unwrap();
    assert!(c.served >= 12, "the last two rounds are served: {c:?}");
    Ok(())
}

/// Three prompts of different lengths arriving one step apart, prefilled in
/// pieces of at most `chunk` tokens that ride the batched step as ragged
/// rows (#728). Every stream equals the pool path's under both width arms
/// (the ladder only re-cuts the pieces), no key drifts, nothing escapes;
/// with the ragged cache capped at 2 the excess layouts are dropped and
/// re-recorded, and the streams still match.
#[test]
fn ragged_steps_are_planned_under_both_width_arms() -> Result<()> {
    use lloom::generation::{LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, PassWidths, Session};
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    cfg.conv_prefill_ctx = true;
    let model = build(&cfg, &dev)?;
    let prompts: Vec<Vec<u32>> = vec![(1..=10).collect(), (11..=17).collect(), (21..=33).collect()];
    let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
    let arrivals = [0u64, 1, 2];
    let run = |widths: PassWidths| -> Result<Vec<Vec<u32>>> {
        let mut sessions: Vec<Session<'_, Model>> = prompts
            .iter()
            .map(|_| {
                Ok(Session::new(
                    &model,
                    Cache::new(true, DType::F16, &cfg, &dev)?,
                ))
            })
            .collect::<Result<_>>()?;
        let opts = DecodeOptions {
            lookahead: 1,
            prefill_chunk: 4,
            pass_widths: widths,
            ..DecodeOptions::new(5)
        };
        let outs = Session::decode_batched_admitted(
            &mut sessions,
            &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
            &refs,
            &arrivals,
            &opts,
            &[u32::MAX],
            &dev,
            &mut |_, _| Ok(()),
        )?;
        Ok(outs.into_iter().map(|o| o.ids).collect())
    };
    let want = run(PassWidths::Lru)?;
    assert_eq!(
        run(PassWidths::Ladder)?,
        want,
        "the ladder re-cuts pieces, not streams"
    );
    assert!(want.iter().all(|ids| ids.len() == 5));

    m.enable_pass_plans(ArenaLayout::Packed);
    for widths in [PassWidths::Lru, PassWidths::Ladder] {
        for round in 0..3 {
            let got = run(widths)?;
            assert_eq!(got, want, "{widths:?} round {round}");
        }
        let c = m.pass_plan_counters().unwrap();
        let (keys, _) = m.pass_plan_report().unwrap();
        let ragged: Vec<_> = keys
            .iter()
            .filter(|(k, ..)| matches!(k.kind, PassKind::Ragged { .. }))
            .collect();
        eprintln!(
            "[{widths:?}] {c:?} ragged keys={} of {}",
            ragged.len(),
            keys.len()
        );
        for (k, covered, total, escapes) in &keys {
            eprintln!(
                "  {:?} {covered}/{total} escapes={escapes} {:?}",
                k.kind,
                m.pass_plan_escapes(k)
            );
        }
        assert_eq!(c.evicted, 0, "{widths:?}: {c:?}");
        assert_eq!(c.escapes, 0, "{widths:?}: {c:?}");
        assert!(!ragged.is_empty(), "{widths:?}: no ragged pass was keyed");
        assert!(
            ragged.iter().all(|(_, covered, total, _)| covered == total),
            "{widths:?}: a ragged plan declines ordinals: {ragged:?}"
        );
    }
    // The `lru` bound: at most two ragged plans are kept — the older ones
    // are dropped when the cap is lowered and as later layouts are planned,
    // and a run whose layouts are being re-recorded still streams the same
    // ids (a miss is a pool pass, never a wrong pass).
    let before = m.pass_plan_counters().unwrap();
    m.set_pass_plan_ragged_cap(2);
    for round in 0..2 {
        assert_eq!(run(PassWidths::Lru)?, want, "capped round {round}");
    }
    let c = m.pass_plan_counters().unwrap();
    assert!(
        c.capped > before.capped,
        "the cap never dropped a plan: {c:?}"
    );
    assert!(c.escapes == 0, "{c:?}");
    assert_eq!(c.evicted, 0, "{c:?}");
    let (keys, _) = m.pass_plan_report().unwrap();
    let ragged = keys
        .iter()
        .filter(|(k, ..)| matches!(k.kind, PassKind::Ragged { .. }))
        .count();
    assert!(ragged <= 2, "{ragged} ragged plans kept over a cap of 2");
    Ok(())
}

#[test]
fn the_ladder_rounds_down_to_a_rung() {
    use lloom::nn::ladder_width;
    assert_eq!(ladder_width(4, 4), 4);
    assert_eq!(ladder_width(3, 4), 2);
    assert_eq!(ladder_width(1, 4), 1);
    assert_eq!(ladder_width(5, 6), 3);
    assert_eq!(ladder_width(7, 0), 7);
    assert_eq!(ladder_width(0, 8), 0);
    assert_eq!(ladder_width(100, 8), 8);
}

/// A drafter that proposes a constant block — almost always rejected, so the
/// target's own correction advances the stream and the accept/rollback path is
/// exercised every round (`dspark_decode_identity.rs`'s argument). It learns
/// nothing from the probe, which is what makes the identity hold regardless.
struct GarbageDrafter {
    k: usize,
    token: u32,
    /// Every probe row count the loop handed back, to prove the taps the
    /// drafter reads AFTER a planned verify are still its own memory.
    seen: Vec<Vec<f32>>,
}

impl lloom::nn::speculator::Speculator for GarbageDrafter {
    type Probe = lloom::nn::speculator::LayerTaps<5>;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: lloom::nn::speculator::SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(
        &mut self,
        active: &[lloom::nn::speculator::SeqId],
    ) -> Result<Vec<lloom::nn::speculator::Draft>> {
        Ok(active
            .iter()
            .map(|_| lloom::nn::speculator::Draft::host(vec![self.token; self.k]))
            .collect())
    }

    fn observe(
        &mut self,
        _seq: lloom::nn::speculator::SeqId,
        probe: &<Self::Probe as lloom::nn::speculator::TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        // Read the taps HERE — after the verify pass that produced them has
        // been closed. Under a plan they live in a slot the next verify
        // reuses unless they are declared escapes, so this read is the gate.
        let mut row = Vec::new();
        for t in probe {
            row.extend(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
        }
        self.seen.push(row);
        Ok(())
    }
}

/// A speculative run under the plan cache: the verify pass is keyed
/// `Verify { k: K + 1 }`, the drafter's taps are declared escapes, and the
/// stream plus every probe the drafter read is identical to the pool path's
/// over a recording round and two served rounds.
///
/// The probe comparison is what pins step 4's declaration on this path: the
/// taps are produced INSIDE a verify and read AFTER it, which the cross-pass
/// detector cannot see (the next pass is another verify), so without the
/// `EscapeGuard` at `tap_layer_input` a served verify hands the drafter a
/// slot the following verify overwrites.
#[test]
fn a_planned_verify_keeps_the_drafters_taps_and_the_stream() -> Result<()> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, Session};
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let k = 3usize;
    let mut cfg = tiny_config(ConvState::RotatingRing { k: k + 1 });
    // The taps `LayerTaps<5>` needs, on layers this fixture has.
    cfg.tap_layers = vec![0, 1, 2, 0, 1];
    cfg.tap_dtype = DType::F32;
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = (1..=10).collect();
    let eos: Vec<u32> = vec![u32::MAX];

    let run = || -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let mut session = Session::new(&model, Cache::new(true, DType::F16, &cfg, &dev)?);
        let mut spec = GarbageDrafter {
            k,
            token: 0,
            seen: Vec::new(),
        };
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(8),
            &eos,
            &mut |_: &[u32]| Ok(()),
        )?;
        Ok((out.ids, spec.seen))
    };

    let (want_ids, want_probes) = run()?;
    assert!(
        want_ids.len() >= 4,
        "the run must span rounds: {want_ids:?}"
    );
    assert!(want_probes.len() >= 2, "the drafter must observe rounds");

    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..3 {
        let (ids, probes) = run()?;
        assert_eq!(ids, want_ids, "round {round}: the stream moved under plans");
        assert_eq!(
            probes.len(),
            want_probes.len(),
            "round {round}: a different number of observes"
        );
        for (i, (got, want)) in probes.iter().zip(&want_probes).enumerate() {
            assert_eq!(
                got, want,
                "round {round}: the drafter's probe {i} was overwritten after its verify"
            );
        }
    }
    let (keys, _) = m.pass_plan_report().expect("cache on");
    let verify: Vec<_> = keys
        .iter()
        .filter(|(key, ..)| matches!(key.kind, PassKind::Verify { .. }))
        .collect();
    let c = m.pass_plan_counters().unwrap();
    eprintln!("[dspark] {c:?} verify keys={verify:?}");
    assert_eq!(
        verify.len(),
        1,
        "one verify window, one key: {:?}",
        keys.iter().map(|(k, ..)| k.kind).collect::<Vec<_>>()
    );
    assert!(
        matches!(verify[0].0.kind, PassKind::Verify { k: w } if w as usize == k + 1),
        "the key is the K+1 window: {:?}",
        verify[0].0.kind
    );
    assert_eq!(c.evicted, 0, "a verify drifted under its own plan: {c:?}");
    assert_eq!(c.escapes, 0, "the taps were not declared: {c:?}");
    Ok(())
}

/// A base grow the machine cannot take is REFUSED before it is allocated
/// (`DESIGN.md` §9.5, the KV grow's gate): the run stops with a message
/// naming the key, nothing was allocated, and no plan was cached — so the
/// pass would have run on the pool as it did before the cache existed.
///
/// Driven by setting the admitted fraction to a value the process is already
/// over, which is the same shape as a real machine being out of room.
#[test]
fn a_base_grow_over_the_budget_is_refused_before_it_allocates() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = (1..=8).collect();

    m.enable_pass_plans(ArenaLayout::Packed);
    // A fraction the process is certainly already past, so the FIRST base
    // grow — the one closing the first key — is the one refused.
    m.set_pass_plan_admit_fraction(1e-9);
    let err = run_prompt(&model, &cfg, &dev, &prompt, 0, 3).expect_err("the grow must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("pass plan base grow refused"),
        "the refusal must name itself: {msg}"
    );
    assert!(
        msg.contains("Nothing was allocated"),
        "the refusal must say the run is intact: {msg}"
    );
    let c = m.pass_plan_counters().unwrap();
    assert_eq!(c.plans, 0, "a plan was cached over a refused base: {c:?}");
    assert_eq!(c.base_grows, 0, "the base grew after the refusal: {c:?}");

    // With the check off the same run completes and caches its plans.
    m.set_pass_plan_admit_fraction(0.0);
    let ids = run_prompt(&model, &cfg, &dev, &prompt, 0, 3)?;
    assert_eq!(ids.len(), 3);
    let c = m.pass_plan_counters().unwrap();
    assert!(c.plans > 0, "{c:?}");
    assert!(c.base_grows > 0, "{c:?}");
    Ok(())
}

/// The backend seam (`DESIGN.md` §9.2aa step 8): a device that has no plan
/// cache DECLINES the bracket and allocates as it always did, so a driver
/// brackets unconditionally and no call site carries a `#[cfg]`. Metal
/// declines too until `enable_pass_plans` turns the cache on — which is what
/// makes the brackets inert on every arm that did not ask for them.
#[test]
fn the_backend_hook_declines_by_default_and_metal_takes_it_up() -> Result<()> {
    use lloom_tensor::backend::PassShapeKey;
    let _serial = SERIAL.lock();
    let shape = PassShapeKey::Decode { b: 1 };

    // The CPU has no plan cache: it declines, and closing is a no-op.
    let cpu = Device::Cpu;
    assert!(!cpu.begin_pass(shape)?, "the CPU claimed to plan a pass");
    cpu.finish_pass()?;

    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    // A fresh device has no cache enabled yet — but this process may have
    // enabled it in an earlier test on the same device, so only the
    // enabled-state implication is asserted, both ways.
    let before = m.pass_plans_enabled();
    assert_eq!(
        dev.begin_pass(shape)?,
        before,
        "the hook must answer whether the cache is on"
    );
    dev.finish_pass()?;

    m.enable_pass_plans(ArenaLayout::Packed);
    assert!(
        dev.begin_pass(shape)?,
        "Metal declined the hook with the cache on"
    );
    dev.finish_pass()?;
    Ok(())
}

/// A RING conv arm reaches the no-allocation invariant — `escapes == 0` on
/// every key — **fused or not**.
///
/// It did not, until the fused kernel took its state as an in-out buffer:
/// `shortconv_ring` returned `(out, new_state)` and allocated a fresh ring
/// every pass, so the crown's own conv arm escaped every plan (22 ordinals
/// per token, one per conv layer). Both arms are asserted, because the two
/// must agree: the fusion is a dispatch-count change, never a change to
/// where the ring lives.
#[test]
fn a_ring_conv_arm_has_no_escapes_fused_or_not() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let prompt: Vec<u32> = (1..=10).collect();
    let escapes_with = |conv_fused: bool| -> Result<u64> {
        let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
        cfg.conv_prefill_ctx = true;
        cfg.conv_fused = conv_fused;
        let model = build(&cfg, &dev)?;
        m.enable_pass_plans(ArenaLayout::Packed);
        let before = m.pass_plan_counters().unwrap().escapes;
        for _ in 0..3 {
            run_prompt(&model, &cfg, &dev, &prompt, 0, 6)?;
        }
        Ok(m.pass_plan_counters().unwrap().escapes - before)
    };

    assert_eq!(
        escapes_with(false)?,
        0,
        "the unfused ring writes its state in place; nothing should outlive a pass"
    );
    assert_eq!(
        escapes_with(true)?,
        0,
        "the FUSED ring allocated a state per pass again — the in-out buffer regressed"
    );
    Ok(())
}

/// A buffer object created at the address a destroyed one had is a new
/// occupant, and its binds are not charged to the value the old object held.
///
/// The recorder names a value by the address of the buffer object serving
/// it, and the address outlives the object: once the pool destroys a buffer
/// (eviction, a trim) the next object created on ANY path may land there —
/// the Shared pool, a host upload, a heap placement, none of which reports an
/// allocation to the recorder. Before `Buffer::new` told the recorder
/// (`arena::note_new_buffer`), such an object's first bind in the pass after
/// was read as the previous pass's value still being read: a phantom escape,
/// counted in a number that moved run to run with which buffers the
/// completion handler had evicted by then (`DESIGN.md` §9.2aa).
///
/// The control arm pins the detector itself: a value whose object is still
/// alive and bound in the next pass IS an escape, so the fixed arm's zero is
/// not a detector that stopped seeing anything.
///
/// Whether a new object lands on a freed address is the allocator's choice:
/// in a fresh process the first freed object's address rarely comes back,
/// and once a round of objects has been created and freed it comes back on
/// the first attempt. So the scenario repeats, each time with a fresh
/// recorder, until one lands.
#[test]
fn a_new_buffer_at_a_destroyed_buffers_address_is_not_the_old_value() -> Result<()> {
    use lloom_tensor::kernels::metal::arena::{set_bind_observer, ArenaRecorder};
    use lloom_tensor::kernels::metal::buffer::MTLResourceOptions;
    use lloom_tensor::kernels::metal::device::Device as RawDevice;
    use std::sync::Arc;
    let _serial = SERIAL.lock();
    let raw = RawDevice::system_default().expect("a Metal device");
    let new_buffer = || {
        raw.new_buffer(4096, MTLResourceOptions::empty())
            .expect("a 4 KiB buffer")
    };
    // `Some((attempt, escaped ordinals))` once a new object took the dead
    // object's address.
    let scenario = || -> Option<(usize, Vec<usize>)> {
        let rec = Arc::new(parking_lot::Mutex::new(ArenaRecorder::new()));
        set_bind_observer(Some(Arc::clone(&rec)));

        // Pass 1: two values, one whose object then dies and one that lives on.
        let dying = new_buffer();
        let living = new_buffer();
        let dead_addr = dying.raw_addr();
        {
            let mut r = rec.lock();
            r.record_alloc(dead_addr, 4096);
            r.record_alloc(living.raw_addr(), 4096);
        }
        drop(dying);
        rec.lock().next_step();

        // Pass 2: objects are created, none reported as an allocation, until
        // one lands on the dead object's address. Each is kept alive so no
        // attempt can hand back an address an earlier attempt freed.
        let mut attempts = Vec::new();
        let reused = loop {
            let b = new_buffer();
            let hit = b.raw_addr() == dead_addr;
            attempts.push(b);
            if hit || attempts.len() == 16 {
                break hit;
            }
        };
        if reused {
            let mut r = rec.lock();
            r.record_bind(dead_addr);
            r.record_bind(living.raw_addr());
        }
        set_bind_observer(None);
        let escaped: Vec<usize> = rec
            .lock()
            .previous_undeclared_ordinals()
            .iter()
            .map(|&(ordinal, ..)| ordinal)
            .collect();
        reused.then_some((attempts.len(), escaped))
    };
    let mut scenarios = 0;
    let (attempt, escaped) = loop {
        scenarios += 1;
        if let Some(found) = scenario() {
            break found;
        }
        assert!(
            scenarios < 256,
            "no new buffer object took a destroyed one's address in {scenarios} scenarios: \
             the fixture cannot exercise the path it exists for"
        );
    };
    eprintln!(
        "the destroyed object's address came back on attempt {attempt} of scenario {scenarios}"
    );
    assert_eq!(
        escaped,
        vec![1],
        "ordinal 0's object was destroyed and a new object bound at its address; ordinal 1's \
         object lived and was bound: only 1 escaped its pass"
    );
    Ok(())
}

/// A ramped prefill and its decode have no escapes on either attention arm
/// while the pool destroys every buffer it frees — the regime in which the
/// addresses of one pass's values are handed to the next pass's new objects.
///
/// The count this pins used to be nonzero and different in every run of one
/// configuration (flash and sdpa alike, once eviction ran), from the phantom
/// `a_new_buffer_at_a_destroyed_buffers_address_is_not_the_old_value`
/// isolates. The eviction count is asserted nonzero so the regime is known to
/// have been entered, and the flash arm's decode plan is asserted to hold more
/// ordinals than sdpa's — FlashDecoding's per-layer scratch — so its own
/// kernels ran rather than a fallback (the config line cannot say, §7.1a-i).
#[test]
fn a_ramped_prefill_has_no_escapes_under_eviction_on_either_attention_arm() -> Result<()> {
    let _serial = SERIAL.lock();
    let prompt: Vec<u32> = (1..=10).collect();
    let mut decode_ordinals = Vec::new();
    for attn in [AttnImpl::FlashDecoding, AttnImpl::Sdpa] {
        let dev = Device::new_metal(0)?;
        let Device::Metal(m) = &dev else {
            unreachable!()
        };
        let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
        cfg.conv_prefill_ctx = true;
        cfg.attn_impl = attn;
        let model = build(&cfg, &dev)?;
        let want = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
        m.set_free_budget(0);
        m.enable_pass_plans(ArenaLayout::Packed);
        let evicted = || {
            let (shared, private) = m.pool_counters();
            shared.evicted + private.evicted
        };
        let evicted_before = evicted();
        for round in 0..3 {
            let got = run_prompt(&model, &cfg, &dev, &prompt, 4, 6)?;
            assert_eq!(
                got, want,
                "{attn:?} round {round}: ids differ from the pool path"
            );
        }
        let c = m.pass_plan_counters().unwrap();
        let keys = m
            .pass_plan_report()
            .map(|(keys, _)| keys)
            .unwrap_or_default();
        decode_ordinals.push(
            keys.iter()
                .find(|(k, ..)| k.kind == PassKind::Decode { b: 1 })
                .map(|&(_, _, total, _)| total)
                .expect("a Decode{1} plan"),
        );
        let escapes: Vec<_> = keys
            .iter()
            .map(|(k, ..)| (k.kind, m.pass_plan_escapes(k)))
            .filter(|(_, e)| !e.is_empty())
            .collect();
        let evicted = evicted() - evicted_before;
        eprintln!("[{attn:?}] evicted={evicted} {c:?} escapes={escapes:?}");
        assert!(
            evicted > 0,
            "{attn:?}: the pool evicted nothing, so no destroyed buffer's address could come back"
        );
        assert!(c.plans >= 4, "{attn:?}: {c:?}");
        assert_eq!(
            c.escapes, 0,
            "{attn:?}: values reported as outliving their pass: {escapes:?}"
        );
    }
    assert!(
        decode_ordinals[0] > decode_ordinals[1],
        "the flash arm's decode step allocated no more than sdpa's ({decode_ordinals:?}): \
         FlashDecoding did not run"
    );
    Ok(())
}

/// The crown's conv arm — the SHUFFLED state decoded by the fused step
/// (`shortconv_step`) — reaches the no-allocation invariant too: the step
/// writes the next state into the layer's spare and the two swap, so a
/// served decode step asks the pools for nothing and no ordinal escapes.
///
/// The shuffle cannot write in place (it reads column `j + 1` while it writes
/// `j`), which is why its state was a fresh tensor per step: one escape per
/// conv layer per step, the 22 a token the real model's `--pass-plans` line
/// carried (`pass-plan-design.md` §17). The stream is the pool path's, and
/// the unfused shuffle arm — which still allocates its state per step — is
/// the control that the zero is the spare's, not a detector gone blind.
#[test]
fn the_fused_shuffle_step_writes_its_spare_and_asks_the_pools_for_nothing() -> Result<()> {
    let _serial = SERIAL.lock();
    let prompt: Vec<u32> = (1..=10).collect();
    let steps = 12;
    let conv_layers = 2u64;
    // (ids, escapes, served decode passes, served-pass requests (shared, private))
    let run = |conv_fused: bool, plans: bool| -> Result<(Vec<u32>, u64, u64, (u64, u64))> {
        let dev = Device::new_metal(0)?;
        let Device::Metal(m) = &dev else {
            unreachable!()
        };
        let mut cfg = tiny_config(ConvState::Shuffle);
        cfg.conv_prefill_ctx = true;
        cfg.conv_fused = conv_fused;
        let model = build(&cfg, &dev)?;
        if !plans {
            return Ok((
                run_prompt(&model, &cfg, &dev, &prompt, 0, steps)?,
                0,
                0,
                (0, 0),
            ));
        }
        m.enable_pass_plans(ArenaLayout::Packed);
        // Two prompts see every key twice; the third is the measured one.
        for _ in 0..2 {
            run_prompt(&model, &cfg, &dev, &prompt, 0, steps)?;
        }
        let c0 = m.pass_plan_counters().unwrap();
        let r0 = m.served_pass_pool_requests();
        let ids = run_prompt(&model, &cfg, &dev, &prompt, 0, steps)?;
        let c1 = m.pass_plan_counters().unwrap();
        let r1 = m.served_pass_pool_requests();
        Ok((
            ids,
            c1.escapes,
            c1.served - c0.served,
            (r1.0 - r0.0, r1.1 - r0.1),
        ))
    };
    let (want, ..) = run(true, false)?;
    let (ids, escapes, served, requests) = run(true, true)?;
    eprintln!("[fused shuffle] escapes={escapes} served={served} requests={requests:?}");
    assert_eq!(ids, want, "a served step moved the stream");
    assert!(
        served >= steps as u64 - 1,
        "the decode was not served: {served}"
    );
    assert_eq!(escapes, 0, "the fused shuffle's state escaped its pass");
    // What the measured sequence's served passes asked for is its session
    // state, once: the fresh prefill's K and V slab (2), its two conv states
    // (2) and its turn-checkpoint ring (1, every slot and layer in one
    // allocation), and each conv layer's spare on the first fused step (2),
    // all from the Shared pool — none per decode step.
    let turn_ring = 1;
    assert_eq!(
        requests,
        (4 + turn_ring + conv_layers, 0),
        "served passes asked the pools (shared, private) for more than the sequence's \
         session state: a per-step state allocation is back"
    );

    // The control: the unfused shuffle still builds its state per step
    // (`cat`), one Shared-pool request per conv layer per served step on top
    // of the same session state, so the counter sees what the spare removed.
    let (want_u, ..) = run(false, false)?;
    let (ids_u, escapes_u, served_u, requests_u) = run(false, true)?;
    eprintln!("[unfused shuffle] escapes={escapes_u} served={served_u} requests={requests_u:?}");
    assert_eq!(ids_u, want_u);
    assert_eq!(
        requests_u,
        (4 + turn_ring + conv_layers * steps as u64, 0),
        "the unfused shuffle's per-step state is the request the counter must see"
    );
    Ok(())
}
