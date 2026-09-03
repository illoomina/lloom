//! Why the verify pass needed a conv writer of its own — the **prefill** branch,
//! pinned.
//!
//! `DESIGN.md` §10.2a's rollback contract costs the conv half's `resolve(n)` at
//! *"a pointer move"*, and §6.1 describes the decode conv path. Both are about
//! `ShortConv::forward`'s `seq_len == 1` branch. A K-position pass takes a
//! different branch, and **this file is the measurement that established the
//! ordinary `seq_len > 1` branch cannot serve a verify pass** — which is why
//! `Cache::advance` exists and why the speculative branch beside it was written.
//!
//! **These tests run without `advance()`, so they exercise the prefill path,
//! and that is deliberate**: prefill's rebuild is correct for prefill and must
//! not change. They assert the property that made a second writer necessary, so
//! that a future change collapsing the two paths turns them red rather than
//! silently making rollback wrong. `lfm2_spec_conv_rollback.rs` is the arm that
//! exercises the speculative writer.
//!
//! CPU only: the question is which bytes survive a call, not what a kernel
//! computes, and the CPU backend is bit-stable (`CONTRIBUTING.md` §3.1).

use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};
use lloom::nn::VarBuilder;
use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};

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
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Model::new(cfg, vb)
}

/// **The finding: a multi-position pass does not ring-write, it rebuilds.**
///
/// `ShortConv::forward`'s `seq_len > 1` branch runs `Conv1d` and then *replaces*
/// `conv_states[block]` with the last `l_cache` columns of this pass's own `bx`,
/// zero-padded to the ring's width. Two consequences, and both are
/// rollback-relevant:
///
/// * the write index is **seeded, not advanced** — `Model::forward` sets the
///   phase to `l_cache - 1` at any `seq_len > 1` rather than moving it by
///   `seq_len`;
/// * the pre-pass history is **gone** — the new state is built from this pass's
///   activations alone, so the columns a `resolve(n < k)` would need to expose
///   are not in the buffer.
///
/// This is `DESIGN.md` §10.2a's *destructive shuffle* argument, reappearing in
/// the branch that section does not discuss. §10.2a establishes that the decode
/// shuffle cannot rewind and that a ring fixes it; the ring fixes the **decode**
/// writer, and the multi-position writer was never converted.
///
/// Asserted as the *current* behaviour in #64's shape — a test that passes while
/// the limitation is present, so it turns red when the limitation is fixed
/// rather than sitting as a comment.
#[test]
fn the_multi_position_conv_path_seeds_the_phase_rather_than_advancing_it() -> Result<()> {
    let dev = Device::Cpu;
    let l_cache = 3usize;
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    // Prefill, then a k-position pass.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    let after_prefill = cache.conv_phase();
    assert_eq!(
        after_prefill,
        l_cache - 1,
        "prefill seeds the newest token at slot l_cache - 1"
    );

    let k = 4usize;
    let window = Tensor::new(&[6u32, 7, 8, 9][..], &dev)?.reshape((1, k))?;
    model.forward_all(&window, prompt.len(), &mut cache)?;

    // **The limitation, pinned.** A ring write of k positions would leave the
    // phase at `(after_prefill + k) % width`; the rebuild leaves it where a
    // prefill leaves it, whatever k was.
    assert_eq!(
        cache.conv_phase(),
        l_cache - 1,
        "the seq_len > 1 branch seeds the phase; it does not advance it by seq_len"
    );
    Ok(())
}

/// The same limitation from the value side, and this is the half that matters
/// for correctness rather than for bookkeeping.
///
/// If the multi-position path merely wrote `k` ring slots, the state after it
/// would contain the pre-pass history, and a `resolve(n)` could expose a shorter
/// window. It does not: the state after a `k`-position pass is a function of
/// that pass's own `k` activations only.
///
/// Demonstrated by running the same window from two *different* histories: if
/// the pre-pass history survived, the resulting states would differ.
#[test]
fn the_multi_position_conv_path_discards_the_pre_pass_history() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, &dev)?;

    let window: Vec<u32> = vec![6, 7, 8, 9];
    let mut states = Vec::new();
    // Two different prompts, so two different pre-pass conv histories.
    for prompt in [vec![1u32, 2, 3, 4, 5], vec![11u32, 12, 13, 14, 15]] {
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
        model.forward(&inp, 0, &mut cache)?;
        let w = Tensor::new(window.as_slice(), &dev)?.reshape((1, window.len()))?;
        model.forward_all(&w, prompt.len(), &mut cache)?;
        states.push(cache.conv_state_for_test(0).unwrap());
    }

    // The conv layer's state is layer 0. Under a ring write the two would
    // differ, because the surviving history would differ. They do not.
    let a = states[0].flatten_all()?.to_vec1::<f32>()?;
    let b = states[1].flatten_all()?.to_vec1::<f32>()?;
    let worst = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-6,
        "the seq_len > 1 branch rebuilds conv state from its own activations, \
         so two different histories must give the same state: worst |Δ| = {worst}"
    );

    // And the state is exactly the last `l_cache` columns of the window's own
    // activations followed by zeros -- the ring's history slots are untouched,
    // which is the direct statement of what a rollback has to work with.
    let st = states[0].i(0)?;
    let width = st.dim(1)?;
    let tail = st.narrow(1, cfg.conv_l_cache, width - cfg.conv_l_cache)?;
    let tail_max = tail
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.abs())
        .fold(0f32, f32::max);
    assert_eq!(
        tail_max, 0.0,
        "prefill zeroes every slot past the live window, so a k-position pass \
         leaves no speculative history to roll back into"
    );
    Ok(())
}

/// **The arms speculation refuses, and it refuses them at `advance`.**
///
/// `DESIGN.md` §10.2a establishes that the shuffle is destructive and cannot
/// rewind. #141 added two rings and §10.2a's cost table — written before either
/// existed — costs *"the ring"* as one mechanism at *"zero — eviction is a
/// pointer move"*. That is true of the rotating arm and **false of the sliding
/// one**: its window compacts when it runs out of slack (§10.2e), and a
/// compaction relocates the live window's bytes, so rewinding across one is not
/// a decrement — the bytes the rollback wants have moved and the slot they
/// occupied has been zeroed.
///
/// Both are refused at `advance`, before any state is written, because a partial
/// speculative pass is exactly what cannot be undone.
#[test]
fn only_the_rotating_ring_admits_a_speculative_window() -> Result<()> {
    let dev = Device::Cpu;
    for (arm, needle) in [
        (ConvState::Shuffle, "rotating"),
        (ConvState::SlidingRing { k: 4, slack: 16 }, "compacts"),
    ] {
        let cfg = tiny_config(arm);
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let err = cache.advance(2).unwrap_err().to_string();
        assert!(
            err.contains(needle),
            "{arm:?} must be refused for its own reason; got: {err}"
        );
    }
    Ok(())
}

/// **`k` is bounded by the ring's history depth, and the bound is checked.**
///
/// The rotating buffer is `l_cache + k_ring` wide. A `k`-position speculation
/// writes `k` slots, leaving `l_cache + k_ring − k` for a window that needs
/// `l_cache` — so `k > k_ring` overwrites the live window and the rollback has
/// nothing to return to.
///
/// This is the constraint §16 6b's *"K is a real memory decision"* acquires once
/// a verifier exists: **the ring's `k` and the verifier's `K` are the same
/// number**, and #141's default of `k = 0` therefore admits no speculation at
/// all. Refusing loudly is what stops that reading as a slow path.
#[test]
fn a_window_wider_than_the_rings_history_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 2 });
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    // Inside the reserve: admitted, and a real pass runs between the `advance`
    // and the `resolve`, because `resolve` checks that the phase moved by `k`.
    let model = build(&cfg, &dev)?;
    let prompt = Tensor::new(&[1u32, 2, 3][..], &dev)?.reshape((1, 3))?;
    model.forward(&prompt, 0, &mut cache)?;
    let tok = cache.advance(2)?;
    let window = Tensor::new(&[4u32, 5][..], &dev)?.reshape((1, 2))?;
    model.forward_all(&window, 3, &mut cache)?;
    cache.resolve(tok, 0)?;

    // **A window opened and closed without running its pass is refused**, which
    // is what stops a caller "rolling back" to an arbitrary slot. Checked here
    // because it is the same guard the lines above depend on.
    let stray = cache.advance(2)?;
    let err = cache.resolve(stray, 1).unwrap_err().to_string();
    assert!(
        err.contains("different number of passes"),
        "resolving a window whose pass never ran must be refused; got: {err}"
    );

    // Past the reserve: refused, and the message names the flag that fixes it.
    let err = cache.advance(3).unwrap_err().to_string();
    assert!(
        err.contains("rotating:3"),
        "the refusal must name the configuration that would admit it; got: {err}"
    );

    // And the default ring admits nothing, which is the case a caller is most
    // likely to hit first.
    let cfg0 = tiny_config(ConvState::RotatingRing { k: 0 });
    let mut cache0 = Cache::new(true, DType::F32, &cfg0, &dev)?;
    assert!(cache0.advance(1).is_err(), "k = 0 reserves no history");
    Ok(())
}
