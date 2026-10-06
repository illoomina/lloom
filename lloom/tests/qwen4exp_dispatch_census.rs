//! The dispatch census of Qwen4-exp's decode and prefill paths, block by
//! block, from the Metal backend's own recorder (`CANDLE_METAL_PROFILE`): how
//! many kernels one decode step of each block launches, which ones, and what
//! a token and a prompt position cost in launches. The counts are a property
//! of the composition, not the geometry, so the tiny synthetic model gives
//! the real model's per-block numbers (the real model's dense weights are
//! `TensorF16`, which adds two casts per dense matmul the `Tensor` weights
//! here do not — counted separately below).
//!
//! The reason: the real model decodes at ~5.7k dispatches a token, 89 % of a
//! 116 ms step GPU-busy on launches of `[1, 1, 2560]` elementwise kernels
//! (`.dev/serving-catalogue.md` §0). This census is the ledger the fusion
//! program works from and the gate it is measured against: every fusion
//! must lower a block's row here, and the whole-token row is the sum.
//!
//! Metal only (the recorder is the Metal backend's). Prints `CENSUS` lines;
//! asserts each block launches something and that the per-token total is
//! the layer sum.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::RecurrentState;
use lloom::profile::CrownExact;
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result, Tensor};
use std::sync::Arc;

fn device() -> Option<Device> {
    // The recorder reads its switch once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    match Device::new_metal(0) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Run `f` twice to warm (pipelines, first-write allocations), then once
/// under a fresh recorder; print and return the census.
fn census(name: &str, mut f: impl FnMut() -> Result<()>) -> Result<(u64, Vec<(String, u64)>)> {
    f()?;
    f()?;
    profile::reset();
    f()?;
    let s = profile::snapshot();
    let mut by = s.by_label.clone();
    by.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let labels: Vec<String> = by.iter().map(|(l, n)| format!("{l}={n}")).collect();
    println!(
        "CENSUS block={name} dispatches={} encoders={} command_buffers={} labels=[{}]",
        s.dispatches,
        s.encoders,
        s.command_buffers,
        labels.join(" ")
    );
    assert!(profile::enabled(), "the recorder must be on for a census");
    assert!(s.dispatches > 0, "{name}: no dispatch recorded");
    Ok((s.dispatches, by))
}

fn x(cfg: &Config, seed: f32, dev: &Device) -> Tensor {
    let d = cfg.hidden_size;
    Tensor::from_vec(
        (0..d)
            .map(|i| ((i as f32 * 0.05 + seed).sin()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, 1, d),
        dev,
    )
    .unwrap()
}

fn carry(cfg: &Config, seed: f32, dev: &Device) -> Tensor {
    let nd = cfg.hc_dim();
    Tensor::from_vec(
        (0..nd)
            .map(|i| ((i as f32 * 0.07 + seed).cos()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, 1, nd),
        dev,
    )
    .unwrap()
}

#[test]
fn decode_blocks_and_the_token() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let mut cfg = Config::tiny();
    // Q8_0 expert stacks route through `mul_mv_id` as the real model does
    // (`moe_intermediate_size` a multiple of the block).
    cfg.moe_intermediate_size = 32;
    // The fused QSA step needs a head the sdpa vector kernel is built for.
    cfg.head_dim = 32;
    let n_layers = cfg.n_layers;

    // --- HC bracket: mix (prologue) and combine (epilogue) ---
    let hc = testing::hc(&cfg, 1.0, &dev);
    let h = carry(&cfg, 0.0, &dev);
    let (hc_mix, _) = census("hc.mix", || {
        let _ = hc.mix(&h)?;
        Ok(())
    })?;
    let (mixed, inject) = hc.mix(&h)?;
    let (hc_combine, _) = census("hc.combine", || {
        let _ = hc.combine(&h, &mixed, &inject)?;
        Ok(())
    })?;

    // --- DeltaNet decode step (steady state: the slabs exist) ---
    let dn = testing::deltanet::<CrownExact>(&cfg, &dev);
    let mut s_state = RecurrentState::new(n_layers);
    let mut conv_state = RecurrentState::new(n_layers);
    let xin = x(&cfg, 0.0, &dev);
    let (deltanet, _) = census("deltanet.decode", || {
        let _ = dn.decode(&xin, 0, &mut s_state, &mut conv_state)?;
        Ok(())
    })?;

    // --- QSA decode step (a few tokens in the cache) ---
    let qsa = testing::qsa::<CrownExact>(&cfg, &dev);
    let mut kv = lloom::models::qwen4exp::KvCache::default();
    for t in 0..3 {
        let _ = qsa.decode(&x(&cfg, t as f32, &dev), &mut kv)?;
    }
    let (qsa_n, _) = census("qsa.decode", || {
        let _ = qsa.decode(&xin, &mut kv)?;
        Ok(())
    })?;

    // --- MoE decode (router + routed dispatch + shared expert) ---
    let [g, u, d] = testing::expert_stacks(&cfg, 0, &dev, GgmlDType::Q8_0);
    let moe = testing::moe_with::<CrownExact, _>(
        &cfg,
        &dev,
        Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)),
    );
    let (moe_n, _) = census("moe.decode", || {
        let _ = moe.decode(&xin)?;
        Ok(())
    })?;

    // --- the head: final HC collapse + lm_head ---
    let head = testing::hc_with(&cfg, 999.0, &dev, false);
    let lm = testing::lin(cfg.vocab_size, cfg.hidden_size, 7.0, &dev);
    let (head_n, _) = census("head", || {
        let (c, _) = head.mix(&h)?;
        let _ = lloom_tensor::Module::forward(&lm, &c)?;
        Ok(())
    })?;

    // --- one whole layer, both kinds, and one whole token ---
    let model: Qwen4Exp<CrownExact, Resident> =
        testing::model_with::<CrownExact, _>(&cfg, &dev, |l| {
            let [g, u, d] = testing::expert_stacks(&cfg, l, &dev, GgmlDType::Q8_0);
            Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
        })?;
    let mut cache = model.new_cache(&cfg, &dev)?;
    let prompt = Tensor::from_vec(vec![3u32, 5, 2, 8, 8, 1, 9], (1, 7), &dev)?;
    let _ = model.forward(&prompt, 0, &mut cache)?;
    let mut tok = 4u32;
    let (linear_layer, _) = census("layer.linear", || {
        let _ = model.layers()[0].decode(&h, 0, &mut cache)?;
        Ok(())
    })?;
    let (full_layer, _) = census("layer.full", || {
        let _ = model.layers()[3].decode(&h, 3, &mut cache)?;
        Ok(())
    })?;
    let (token, _) = census("token.decode", || {
        let input = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
        let pos = cache.pos;
        let _ = model.forward(&input, pos, &mut cache)?;
        tok = (tok * 7 + 3) % cfg.vocab_size as u32;
        Ok(())
    })?;
    // --- the prefill blocks over a 16-position chunk ---
    let xs = Tensor::from_vec(
        (0..16 * cfg.hidden_size)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, 16, cfg.hidden_size),
        &dev,
    )?;
    let hs = Tensor::from_vec(
        (0..16 * cfg.hc_dim())
            .map(|i| ((i as f32 * 0.017).cos()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, 16, cfg.hc_dim()),
        &dev,
    )?;
    let (hc_mix16, _) = census("hc.mix16", || {
        let _ = hc.mix(&hs)?;
        Ok(())
    })?;
    let (mixed16, inject16) = hc.mix(&hs)?;
    let (hc_combine16, _) = census("hc.combine16", || {
        let _ = hc.combine(&hs, &mixed16, &inject16)?;
        Ok(())
    })?;
    let (deltanet16, _) = census("deltanet.prefill16", || {
        let mut s_state = RecurrentState::new(n_layers);
        let mut conv_state = RecurrentState::new(n_layers);
        let _ = dn.prefill(&xs, 0, &mut s_state, &mut conv_state)?;
        Ok(())
    })?;
    let (qsa16, _) = census("qsa.prefill16", || {
        let mut kv = lloom::models::qwen4exp::KvCache::default();
        let _ = qsa.prefill(&xs, &mut kv)?;
        Ok(())
    })?;
    let (moe16, _) = census("moe.forward16", || {
        let _ = moe.forward(&xs)?;
        Ok(())
    })?;
    println!(
        "CENSUS prefill16 summary hc.mix={hc_mix16} hc.combine={hc_combine16} deltanet={deltanet16} qsa={qsa16} moe={moe16}"
    );

    // --- a 16-position chunk prefill, per position ---
    let (chunk, chunk_by) = census("prefill.chunk16", || {
        let mut fresh = model.new_cache(&cfg, &dev)?;
        let p = Tensor::from_vec(
            (0..16u32).map(|i| (i * 5 + 1) % 128).collect(),
            (1, 16),
            &dev,
        )?;
        let _ = model.forward(&p, 0, &mut fresh)?;
        Ok(())
    })?;
    let copy2d: u64 = chunk_by
        .iter()
        .filter(|(l, _)| l.starts_with("copy2d"))
        .map(|(_, n)| n)
        .sum();

    println!(
        "CENSUS summary hc.mix={hc_mix} hc.combine={hc_combine} deltanet={deltanet} qsa={qsa_n} moe={moe_n} head={head_n} layer.linear={linear_layer} layer.full={full_layer} token={token} (layers {}: 3 linear + 1 full) prefill.chunk16={chunk} per_position={:.1} copy2d_per_position={:.1}",
        n_layers,
        chunk as f64 / 16.0,
        copy2d as f64 / 16.0
    );
    println!(
        "CENSUS real-geometry projection: 36 linear + 12 full layers + head = {} dispatches/token before the TensorF16 casts",
        36 * linear_layer + 12 * full_layer + head_n
    );
    // The token is its layers plus the head plus the embedding/argmax-free
    // logits cast — within a handful of the layer sum.
    let layers_sum = 3 * linear_layer + full_layer + head_n;
    assert!(
        token >= layers_sum && token <= layers_sum + 8,
        "token {token} vs layer sum {layers_sum} (+ head): the decode has a cost outside its layers"
    );
    Ok(())
}
