//! Does the multi-position conv path leave enough state to roll back?
//!
//! `DESIGN.md` §10.2a's rollback contract assumes the conv half's `resolve(n)`
//! is *"a pointer move"* — `phase += n`. That is true of the **decode** ring
//! writer, which `ShortConv::forward` takes at `seq_len == 1`. A speculative
//! verify pass runs at `seq_len == k`, and that is a different branch.
//!
//! This test exists to establish, by execution rather than by reading, what the
//! `seq_len > 1` branch leaves behind — because the answer decides whether the
//! verifier can use the model's existing conv path at all.
//!
//! It runs on the CPU: the question is about which bytes survive a call, not
//! about a kernel, and the CPU backend is bit-stable and needs no GPU
//! (`CONTRIBUTING.md` §3.1).

use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::nn::VarBuilder;
use lloom::profile::RewindableExact;
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

/// A two-layer LFM2 — one conv layer, one attention layer — small enough to
/// build from random weights and large enough to exercise both paths.
fn tiny_config(conv_state: ConvState) -> Config {
    Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        // §10.4's defaults, matching `into_config`. Added by #116 after this
        // file was written, which is why it stopped compiling.
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        // `Grow`, which is what #116's per-call allocation did before
        // the axis reached it (#234). Not a choice — see the field.
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        // `Off`, the f16 arm every recorded digest belongs to. Added by #218
        // after this file was written, which is why it stopped compiling --
        // `flash_scratch_sizing` above carries the same note from #116.
        kv_quant: Default::default(),
        conv_state,
        memory_budget: None,
        // Both epilogue fusions off, per DESIGN.md 7.1a (issue #509).
        gemv_silu: false,
        gemv_residual: false,
        // #363's tap: off, as every pre-#363 caller had it.
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

fn build(cfg: &Config, dev: &Device) -> Result<Model<RewindableExact>> {
    // Deterministic weights: a fixed ramp, so two runs of this test agree and a
    // difference between two *paths* is attributable to the paths.
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
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Model::<RewindableExact>::new(cfg, vb)
}

/// **The finding this test exists to pin.**
///
/// A `k`-position pass followed by `resolve(n)` must leave the model in the
/// state `n` separate decode steps would have left it in. If it does, a
/// speculative verifier can use the existing conv path; if it does not, the
/// verify pass needs a conv writer of its own.
///
/// Written as an equality against the **decode** path, which is the reference
/// the whole issue is gated on: greedy speculation is output-identical to
/// non-speculative decoding *by construction*, so anything else is a defect.
#[test]
fn a_k_position_pass_then_resolve_matches_n_decode_steps() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;

    // A prompt, then a known token sequence to walk.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let seq: Vec<u32> = vec![6, 7, 8, 9];

    // --- reference: prefill, then n=2 separate decode steps -----------------
    let mut ref_cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut ref_cache)?;
    let n = 2usize;
    let k = 4usize;
    // A **continuation token**, decoded after the n accepted ones. Comparing
    // the two paths on *this* step rather than on step n is what makes the
    // comparison a statement about the surviving state: step n's own logits are
    // produced by different code on the two arms (`forward` against
    // `forward_all`), where the step after it runs identical code over whatever
    // each arm left behind.
    let cont: u32 = 21;
    for (i, t) in seq.iter().take(n).enumerate() {
        let inp = Tensor::new(&[*t], &dev)?.reshape((1, 1))?;
        model.forward(&inp, prompt.len() + i, &mut ref_cache)?;
    }
    let inp = Tensor::new(&[cont], &dev)?.reshape((1, 1))?;
    let want = model
        .forward(&inp, prompt.len() + n, &mut ref_cache)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // --- speculative: prefill, one k-position pass, resolve(n) --------------
    let mut spec_cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut spec_cache)?;

    let tok = spec_cache.advance(k)?;
    let window = Tensor::new(&seq[..k], &dev)?.reshape((1, k))?;
    model.forward_all(&window, prompt.len(), &mut spec_cache)?;
    spec_cache.resolve(tok, n)?;

    // The rolled-back state must be indistinguishable from the one n decode
    // steps produced — including that the k − n discarded positions leave no
    // trace, which is exactly what §10.2a's "discarded bytes need not be
    // cleared" claims and what this asserts rather than assumes.
    let inp = Tensor::new(&[cont], &dev)?.reshape((1, 1))?;
    let got = model
        .forward(&inp, prompt.len() + n, &mut spec_cache)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-5,
        "a k-position pass + resolve({n}) must leave the state n decode steps would: \
         worst |Δ| = {worst}"
    );

    // **The logit comparison alone is not sufficient, measured rather than
    // assumed.** With the speculative conv branch disabled — so the pass
    // rebuilds the state and leaves the ring's window in the wrong place — the
    // assertion above reads 6.1e-6 and *passes*, against a 1e-5 threshold it
    // clears by 1.6×. A two-layer model at `hidden_size = 8` simply does not
    // amplify a wrong conv window into a logit difference reliably.
    //
    // That is `DESIGN.md` §11.3l's mutation lesson in a third place: a fixture
    // built only from what is convenient cannot see the defect it was written
    // for. So the contract is asserted where it is *stated* — on the surviving
    // state — and the logits are the corroboration rather than the check.
    let live_w = cfg.conv_l_cache;
    let width = live_w + 4; // RotatingRing { k: 4 }
    let ref_state = ref_cache.conv_state_for_test(0).unwrap().i(0)?;
    let spec_state = spec_cache.conv_state_for_test(0).unwrap().i(0)?;
    assert_eq!(ref_cache.conv_phase(), spec_cache.conv_phase());

    // Only the live window is compared: the discarded slots keep their bytes by
    // design (§10.2a — "discarded bytes need not be cleared"), so requiring the
    // whole buffer to match would assert the opposite of the contract.
    let phase = spec_cache.conv_phase();
    for j in 0..live_w {
        let slot = (phase + width - (live_w - 1 - j)) % width;
        let a = ref_state
            .narrow(1, slot, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let b = spec_state
            .narrow(1, slot, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            d < 1e-6,
            "live conv slot {slot} must survive the rollback unchanged: |Δ| = {d}"
        );
    }
    Ok(())
}

/// **Full acceptance is the transparency claim**, and it is the one #89's
/// acceptance list names first: with a proposer that always proposes correctly,
/// the mechanism must be invisible.
///
/// A `k`-position verify pass must produce, at every position, the logits the
/// corresponding decode step would have produced. Asserted per position rather
/// than on the last one alone — a pass that got only the final position right
/// would still be wrong for every rejection, which is the case that matters.
#[test]
fn every_position_of_a_verify_pass_matches_its_decode_step() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let seq: Vec<u32> = vec![6, 7, 8, 9];
    let k = seq.len();

    let mut ref_cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut ref_cache)?;
    let mut want = Vec::new();
    for (i, t) in seq.iter().enumerate() {
        let inp = Tensor::new(&[*t], &dev)?.reshape((1, 1))?;
        want.push(
            model
                .forward(&inp, prompt.len() + i, &mut ref_cache)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
    }

    let mut spec_cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut spec_cache)?;
    let tok = spec_cache.advance(k)?;
    let window = Tensor::new(seq.as_slice(), &dev)?.reshape((1, k))?;
    let got = model.forward_all(&window, prompt.len(), &mut spec_cache)?;
    spec_cache.resolve(tok, k)?;

    for i in 0..k {
        let g = got.i(0)?.i(i)?.flatten_all()?.to_vec1::<f32>()?;
        let worst = g
            .iter()
            .zip(want[i].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "verify position {i} must match decode step {i}: worst |Δ| = {worst}"
        );
    }
    Ok(())
}

/// **Consecutive windows, with a rejection in the first.**
///
/// The single-window tests above show a rollback lands in the right place. They
/// do not show the *next* window starts from it correctly, and that is a
/// separate claim: a verify pass reads the ring at a phase the previous
/// `resolve` set, so an off-by-one in the phase is invisible until a second
/// window runs against it.
///
/// Found by the real model diverging at step 3 — the first window that had a
/// rejection in it — while every single-window test passed. Two windows is the
/// smallest fixture that can see it.
#[test]
fn a_window_after_a_rejection_starts_from_the_rolled_back_state() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    // Two windows' worth of tokens.
    let seq: Vec<u32> = vec![6, 7, 8, 9, 10, 11, 12, 13];
    let k = 4usize;
    let n = 2usize; // the first window accepts 2 of 4

    // Reference: n + k decode steps, one at a time.
    let mut rc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut rc)?;
    let mut want = Vec::new();
    for (i, t) in seq.iter().take(n + k).enumerate() {
        let inp = Tensor::new(&[*t], &dev)?.reshape((1, 1))?;
        want.push(
            model
                .forward(&inp, prompt.len() + i, &mut rc)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
    }

    // Speculative: a k-window resolved to n, then a second k-window.
    let mut sc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut sc)?;

    let tok = sc.advance(k)?;
    let w1 = Tensor::new(&seq[..k], &dev)?.reshape((1, k))?;
    model.forward_all(&w1, prompt.len(), &mut sc)?;
    sc.resolve(tok, n)?;

    let tok = sc.advance(k)?;
    let w2 = Tensor::new(&seq[n..n + k], &dev)?.reshape((1, k))?;
    let got = model.forward_all(&w2, prompt.len() + n, &mut sc)?;
    sc.resolve(tok, k)?;

    // The second window's positions must match decode steps n..n+k.
    for j in 0..k {
        let g = got.i(0)?.i(j)?.flatten_all()?.to_vec1::<f32>()?;
        let worst = g
            .iter()
            .zip(want[n + j].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "second window position {j} must match decode step {}: worst |Δ| = {worst}",
            n + j
        );
    }
    Ok(())
}

/// Isolation: the same two-window sequence on a **conv-only** model.
///
/// If this passes while the mixed model fails, the defect is in the KV half; if
/// both fail it is the conv half. Kept because "which half" is the first
/// question any future regression here has to answer, and re-deriving it costs
/// a build.
#[test]
fn two_windows_on_a_conv_only_model() -> Result<()> {
    let dev = Device::Cpu;
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    cfg.layer_types = vec![LayerType::Conv, LayerType::Conv];
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let seq: Vec<u32> = vec![6, 7, 8, 9, 10, 11, 12, 13];
    let k = 4usize;
    let n = 2usize;

    let mut rc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut rc)?;
    let mut want = Vec::new();
    for (i, t) in seq.iter().take(n + k).enumerate() {
        let inp = Tensor::new(&[*t], &dev)?.reshape((1, 1))?;
        want.push(
            model
                .forward(&inp, prompt.len() + i, &mut rc)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
    }

    let mut sc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut sc)?;
    let tok = sc.advance(k)?;
    let w1 = Tensor::new(&seq[..k], &dev)?.reshape((1, k))?;
    model.forward_all(&w1, prompt.len(), &mut sc)?;
    sc.resolve(tok, n)?;
    let tok = sc.advance(k)?;
    let w2 = Tensor::new(&seq[n..n + k], &dev)?.reshape((1, k))?;
    let got = model.forward_all(&w2, prompt.len() + n, &mut sc)?;
    sc.resolve(tok, k)?;

    for j in 0..k {
        let g = got.i(0)?.i(j)?.flatten_all()?.to_vec1::<f32>()?;
        let worst = g
            .iter()
            .zip(want[n + j].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "conv-only second window position {j}: worst |Δ| = {worst}"
        );
    }
    Ok(())
}

/// Control for the two-window test: **two windows with no rejection**.
///
/// If this passes and the rejection case fails, the rollback is the suspect. If
/// both fail, the second window is wrong regardless of what the first did, and
/// the rollback is exonerated. Named a control because it is what makes the
/// other result attributable (§1.3).
#[test]
fn two_full_windows_with_no_rejection() -> Result<()> {
    let dev = Device::Cpu;
    let mut cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    cfg.layer_types = vec![LayerType::Conv, LayerType::Conv];
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let seq: Vec<u32> = vec![6, 7, 8, 9, 10, 11, 12, 13];
    let k = 4usize;

    let mut rc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut rc)?;
    let mut want = Vec::new();
    for (i, t) in seq.iter().take(2 * k).enumerate() {
        let inp = Tensor::new(&[*t], &dev)?.reshape((1, 1))?;
        want.push(
            model
                .forward(&inp, prompt.len() + i, &mut rc)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
    }

    let mut sc = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut sc)?;
    let tok = sc.advance(k)?;
    let w1 = Tensor::new(&seq[..k], &dev)?.reshape((1, k))?;
    model.forward_all(&w1, prompt.len(), &mut sc)?;
    sc.resolve(tok, k)?;
    let tok = sc.advance(k)?;
    let w2 = Tensor::new(&seq[k..2 * k], &dev)?.reshape((1, k))?;
    let got = model.forward_all(&w2, prompt.len() + k, &mut sc)?;
    sc.resolve(tok, k)?;

    for j in 0..k {
        let g = got.i(0)?.i(j)?.flatten_all()?.to_vec1::<f32>()?;
        let worst = g
            .iter()
            .zip(want[k + j].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "no-rejection second window position {j}: worst |Δ| = {worst}"
        );
    }
    Ok(())
}
