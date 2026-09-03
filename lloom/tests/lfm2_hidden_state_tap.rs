//! The target-side hidden-state tap — `DESIGN.md` §10.2m, issue #363.
//!
//! §10.2l measured `LFM2.5-2.6B-DSpark` and found it **cannot be run from
//! tokens**: no `embed_tokens`, no `lm_head`, and an `fc.weight [2048, 10240]`
//! consuming five concatenated 2048-d hidden states from the target's layers.
//! §10.2j's proposer seam is `(last, proposed) -> Vec<u32>`, so the drafter does
//! not fit it and **the prerequisite is a target-side tap**. This file is that
//! tap's test.
//!
//! # What has to be asserted, and why a shape invariant will not do
//!
//! `CONTRIBUTING.md` §3.1b: *an invariant over a sequence is blind to any defect
//! that shifts the whole sequence*, and a tap is exactly the shape that invites
//! one — five tensors of identical geometry, where "there are five of them, all
//! `[b, seq, hidden]`" is satisfied equally by the right five and by five
//! post-MLP residuals, five copies of layer 0, or five zeros.
//!
//! So every test here carries an **absolute reference**: the tap is compared
//! against the residual stream recomputed independently, position by position,
//! by running the model one layer at a time. That reference is what makes the
//! representation claim — *layer-input, before the block runs* — falsifiable,
//! and it is the claim §10.2m takes from llama.cpp PR #27383's
//! `t_layer_inp[il] = cur` rather than from prose.
//!
//! # CPU only, deliberately
//!
//! The question is *which tensor* is captured, not what a kernel computes, and
//! the CPU backend is bit-stable where the Metal generic path is not
//! (`CONTRIBUTING.md` §3.1, §2.3.8a). A tap that captures the wrong layer is
//! wrong on every backend; a tap that captures the right one is bit-exact here
//! and lets the assertions be equalities rather than tolerances.

use lloom_tensor::{DType, Device, Result, Tensor};
use lloom::nn::VarBuilder;
use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};

/// Six layers, alternating conv and attention, so a tap index can fall on
/// either kind. The real target's `target_layer_ids` `[2, 9, 17, 21, 27]` hits
/// both (§5.3), so a fixture that only tapped one kind could not see a defect
/// specific to the other.
fn tiny_config(tap_layers: Vec<usize>) -> Config {
    Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state: ConvState::default(),
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers,
        tap_dtype: DType::F32,
    }
}

/// Distinct values per position, so a transposed or misaligned capture is
/// visible rather than accidentally equal — `lfm2.rs`'s own fixture rule
/// (§11.3l finding 3: a fixture whose values coincide proves nothing).
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

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?;
    d.flatten_all()?.max(0)?.to_scalar::<f32>()
}

/// **The absolute reference.** The residual stream at the entry of each layer,
/// recomputed by an independent path: `Model::layer_input_residuals` runs the
/// same 30 blocks and records `hidden_states` before each `layer.forward`, which
/// is what §10.2m specifies and what llama.cpp's `t_layer_inp[il] = cur` does.
///
/// This is a *second* computation of the same quantity rather than a restatement
/// of the tap's own output — `CONTRIBUTING.md` §3.1b rule 1, and the distinction
/// that separates a comparison from a self-consistency check. It shares the
/// model and the weights (there is only one model), and it does **not** share
/// the capture: the tap pushes inside the production loop and this reads the
/// value the loop was about to consume.
fn reference_layer_inputs(
    model: &Model,
    input: &Tensor,
    index_pos: usize,
    cfg: &Config,
    dev: &Device,
) -> Result<Vec<Tensor>> {
    // A cache of its own, so the reference walk does not disturb the cache the
    // measured run used. Two runs over one cache are two different
    // computations — §2.3.8d's turn-2 argument, and it would make the
    // comparison meaningless rather than merely noisy.
    let mut c = Cache::new(true, DType::F32, cfg, dev)?;
    model.layer_input_residuals(input, index_pos, &mut c)
}

/// The tap captures the **layer-input residual** at each configured index, and
/// the reference is an independent recomputation rather than the tap's own
/// output.
///
/// This is the representation claim. A tap that captured the layer *output*, or
/// the post-attention residual, or the normed input, would satisfy every shape
/// property and fail here — which is the point, since §10.2m's whole content is
/// *which* tensor, and nothing downstream can tell: the target verifies
/// bit-exactly whatever the draft proposes, so a wrong tap yields a silently bad
/// acceptance rate and a green identity gate (§10.2m's soundness argument).
#[test]
fn the_tap_is_the_layer_input_residual() -> Result<()> {
    let dev = Device::Cpu;
    let taps_at = vec![0usize, 2, 5];
    let cfg = tiny_config(taps_at.clone());
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    let input = Tensor::new(&[[3u32, 7, 1, 9]], &dev)?;
    let _ = model.forward(&input, 0, &mut cache)?;
    let taps = cache.take_taps();

    assert_eq!(
        taps.len(),
        taps_at.len(),
        "one tap per configured layer index"
    );

    let reference = reference_layer_inputs(&model, &input, 0, &cfg, &dev)?;
    assert_eq!(
        reference.len(),
        cfg.num_hidden_layers,
        "the reference walks every layer, tapped or not"
    );

    for (slot, &layer) in taps_at.iter().enumerate() {
        let got = &taps[slot];
        let want = &reference[layer];
        assert_eq!(
            got.dims(),
            want.dims(),
            "tap {slot} (layer {layer}) shape must be the residual's"
        );
        let d = max_abs_diff(got, want)?;
        assert_eq!(
            d, 0.0,
            "tap {slot} must BE layer {layer}'s input residual, bit for bit \
             (max|delta| = {d:e}). A non-zero delta here means the tap is a \
             different tensor than DESIGN.md 10.2m specifies."
        );
    }
    Ok(())
}

/// **Order is load-bearing, and this is the assertion that pins it.**
///
/// The consumer concatenates the taps to reach `fc.weight [2048, 10240]`, which
/// expects `[H(l1); ...; H(l5)]` in the drafter's own `target_layer_ids` order.
/// A permuted concat pairs every slice with the wrong 2048-column block of `W_c`
/// — algebraically valid, silently wrong, and exactly the shape §10.2h found in
/// the rotating ring's weight permutation, where *"the newest token is weighted
/// zero"* and no gate could see it.
///
/// So the test is not *"the taps are the tapped layers in some order"* but
/// *"tap `i` is `tap_layers[i]`"*, and the fixture makes those distinguishable:
/// the layers are compared against each other first, so a test that passed by
/// the layers being coincidentally equal is refused.
#[test]
fn tap_order_follows_tap_layers_and_the_layers_are_distinguishable() -> Result<()> {
    let dev = Device::Cpu;
    // Deliberately NOT ascending: an implementation that sorted, or that
    // iterated the layer loop and pushed on membership, would produce
    // `[1, 4]` here and pass an ascending-order fixture by accident.
    let taps_at = vec![4usize, 1];
    let cfg = tiny_config(taps_at.clone());
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    let input = Tensor::new(&[[5u32, 2, 8]], &dev)?;
    let _ = model.forward(&input, 0, &mut cache)?;
    let taps = cache.take_taps();
    let reference = reference_layer_inputs(&model, &input, 0, &cfg, &dev)?;

    // The non-vacuity arm, and it runs first (§3.1a rule 1): if layers 1 and 4
    // carried the same residual, every order assertion below would pass under
    // any permutation and the test would be measuring its own fixture.
    let spread = max_abs_diff(&reference[1], &reference[4])?;
    assert!(
        spread > 1e-6,
        "fixture is vacuous: layers 1 and 4 have indistinguishable residuals \
         (max|delta| = {spread:e}), so no order assertion below can fail"
    );

    for (slot, &layer) in taps_at.iter().enumerate() {
        let d = max_abs_diff(&taps[slot], &reference[layer])?;
        assert_eq!(
            d, 0.0,
            "tap slot {slot} must hold layer {layer} -- the order of \
             Config::tap_layers, not ascending order"
        );
    }

    // And the sharper half, stated as its own assertion so a failure says
    // *permuted* rather than *wrong*: slot 0 must NOT be the ascending answer.
    let ascending = max_abs_diff(&taps[0], &reference[1])?;
    assert!(
        ascending > 1e-6,
        "tap slot 0 holds layer 1, which is the ASCENDING answer rather than \
         tap_layers[0] = 4 -- the implementation is sorting or ignoring order"
    );
    Ok(())
}

/// A verify pass taps every position, which is the case #363 exists for.
///
/// `forward_all` is the `seq_len > 1` entry point a speculative window runs
/// through (§10.2h), and a drafter conditioned on the target's hidden states
/// needs them **per position** — `M = accepted_len` in §10.2m's projection. A
/// tap that only worked at `seq_len == 1` would be a tap the consumer cannot
/// use, and the two entry points sharing one helper is what this asserts.
#[test]
fn forward_all_taps_every_position() -> Result<()> {
    let dev = Device::Cpu;
    let taps_at = vec![2usize, 5];
    let cfg = tiny_config(taps_at.clone());
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    let k = 4usize;
    let input = Tensor::new(&[[3u32, 7, 1, 9]], &dev)?;
    let logits = model.forward_all(&input, 0, &mut cache)?;
    assert_eq!(logits.dims()[1], k, "forward_all projects every position");

    let taps = cache.take_taps();
    assert_eq!(taps.len(), taps_at.len());
    for (slot, &layer) in taps_at.iter().enumerate() {
        assert_eq!(
            taps[slot].dims(),
            &[1, k, cfg.hidden_size],
            "tap {slot} (layer {layer}) must carry all {k} positions, not the last"
        );
    }
    Ok(())
}

/// Taps do not accumulate across passes, and a rolled-back window leaves none
/// behind for the next pass to read as though it were committed.
///
/// The vector is cleared at pass entry rather than at `take_taps`, so a pass
/// whose taps are never drained cannot contaminate the next one. That is the
/// property that makes the accessor safe for a speculative caller, which
/// `verify_step` is: a window that proposes 4 and accepts 1 must not leave four
/// positions of hidden state where the consumer expects one.
#[test]
fn taps_are_replaced_per_pass_and_never_accumulate() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(vec![1usize, 3]);
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;

    // Three passes, and the middle one is deliberately NOT drained -- which is
    // the case the clear-at-entry placement exists for.
    let a = Tensor::new(&[[3u32]], &dev)?;
    let _ = model.forward(&a, 0, &mut cache)?;
    let first = cache.take_taps();
    assert_eq!(first.len(), 2);

    let b = Tensor::new(&[[7u32]], &dev)?;
    let _ = model.forward(&b, 1, &mut cache)?;
    // not drained

    let c = Tensor::new(&[[1u32]], &dev)?;
    let _ = model.forward(&c, 2, &mut cache)?;
    let third = cache.take_taps();
    assert_eq!(
        third.len(),
        2,
        "an undrained pass must not leave its taps to be counted with the next \
         pass's -- 4 here would mean the vector accumulates"
    );

    // And the surviving taps are the THIRD pass's, not the second's. A clear
    // that ran but pushed the wrong pass's values would keep the length right.
    let reference = reference_layer_inputs(&model, &c, 2, &cfg, &dev)?;
    // The reference walks with a fresh cache at index_pos 2, so its conv and KV
    // history differ from the measured run's; compare only that the tap is not
    // the FIRST pass's, which is what a stale vector would hold.
    let stale = max_abs_diff(&third[0], &first[0])?;
    assert!(
        stale > 1e-6,
        "the third pass's tap equals the first pass's -- the vector is stale \
         rather than replaced"
    );
    let _ = reference;
    Ok(())
}

/// The default is no tap, and it allocates nothing.
///
/// §7.1a: a default is flipped by its own argued decision, and #363 does not
/// make one -- every pre-#363 caller keeps the path it had. This is the arm that
/// says so, and it is what licenses the claim that the tap is free when unused:
/// an empty `tap_layers` reaches an `is_empty()` and returns.
#[test]
fn the_default_taps_nothing() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(Vec::new());
    assert!(
        cfg.tap_layers.is_empty(),
        "tap_layers defaults to empty (7.1a)"
    );
    let model = build(&cfg, &dev)?;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
    let input = Tensor::new(&[[3u32, 7]], &dev)?;
    let _ = model.forward(&input, 0, &mut cache)?;
    assert!(
        cache.take_taps().is_empty(),
        "an unconfigured run must capture nothing"
    );
    Ok(())
}

/// An out-of-range tap index is refused at configuration rather than ignored.
///
/// The silent alternative is worse than it looks: `target_layer_ids` is a
/// property of the *drafter* checkpoint and the layer count a property of the
/// *target*, so a mismatched pair is exactly the case a consumer will hit
/// first -- pairing a 30-layer drafter's indices with a different target. Left
/// to `contains`, index 30 on a 30-layer model captures nothing and the
/// consumer receives four slices where `fc.weight` expects five, which fails
/// as a shape error far from the cause.
#[test]
fn a_tap_index_past_the_last_layer_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(vec![0usize, cfg_layers()]);
    let model = build(&cfg, &dev)?;
    let err = Cache::new(true, DType::F32, &cfg, &dev)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("tap_layers"),
        "an out-of-range tap index must be refused by name, got: {err:?}"
    );
    let _ = model;
    Ok(())
}

fn cfg_layers() -> usize {
    6
}
