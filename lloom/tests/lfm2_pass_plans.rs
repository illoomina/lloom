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

/// The recorder's bind observer and the dispatch clock are process globals
/// (`arena.rs`), so two tests planning passes at once record each other's
/// dispatches: every test here holds this for its whole body.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
            "[{tag}] allocations={} {:?} keys={:?}",
            shared.allocations + private.allocations,
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
