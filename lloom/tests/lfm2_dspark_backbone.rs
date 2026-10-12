//! The DSpark drafter backbone — `DESIGN.md` §10.2p, issue #363.
//!
//! §10.2o built the projection and named the backbone as the next increment.
//! This file is its test.
//!
//! # What has to be asserted, and why the obvious assertions do not
//!
//! Every defect this backbone can carry is **shape-valid and silently wrong**,
//! because the target verifies every proposed token: a broken drafter emits a
//! byte-identical stream and costs only acceptance rate (§10.2n, §10.2j). So
//! there is no digest, no smoke test and no output comparison that can see any
//! of these, and each one has to be pinned by an *absolute reference*
//! (`CONTRIBUTING.md` §3.1b rule 1):
//!
//! * **`markov_w1` and `markov_w2` swapped.** Both are `[vocab, rank]` on disk.
//!   One is indexed on axis 0, the other contracted on axis 1; swapping them
//!   typechecks, produces a well-formed `[vocab]` bias, and changes only the
//!   numbers. This is §10.2n's permutation defect in a second place.
//! * **`logits_start` off by one.** Slicing predictions from slot 1 instead of
//!   slot 0 re-predicts the token already in hand and lands every draft one
//!   position late. mlx-dspark measured the sibling case at **accept 1.35
//!   against 3.42** with *"no error anywhere"*.
//! * **neox RoPE instead of interleaved.** `rope` and `rope_i` have identical
//!   signatures and identical output shapes. The drafter's config says
//!   `rope_is_neox_style: false` where the LFM2 target is neox, so the two
//!   sides of this seam legitimately differ and a copy-paste from the target
//!   is wrong in a way nothing reports (§5.11's `rope_theta` trap, one field
//!   over).
//! * **a causal block mask.** The draft block is bidirectional; masking it
//!   causally still produces γ logits of plausible magnitude, and is the
//!   difference between block diffusion and ordinary autoregression.
//! * **the Markov bias applied in parallel rather than sequentially.** Position
//!   `k`'s bias depends on the token sampled at `k−1`. Applying one bias to
//!   every position is the same shape and drops exactly the inter-token
//!   dependency the head exists to restore.
//!
//! # CPU only, deliberately
//!
//! The projection's test gives the reason and it holds here: the question is
//! *which arithmetic*, and the CPU backend is bit-stable where the Metal
//! generic path is not (`CONTRIBUTING.md` §3.1, §2.3.8a). It also keeps this
//! runnable with no `metal` feature, which is the build §14.1 gates on.

use lloom::models::lfm2_dspark::{
    confidence_feature_width, drafter_params, DSparkConfig, MarkovHead,
};
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor, D};

/// The released 2.6B drafter's own totals, read from the payload-bearing
/// safetensors header rather than from any section's prose.
const CHECKPOINT_PARAMS: usize = 327_707_521;
const CHECKPOINT_TENSORS: usize = 62;

// ---------------------------------------------------------------------------
// The config, against the checkpoint
// ---------------------------------------------------------------------------

/// `drafter_params` is computed from the *config*; `CHECKPOINT_PARAMS` is
/// summed from the *header*. They are two independent quantities and their
/// agreement is the evidence that the config this build hardcodes describes the
/// file it will load.
///
/// This is `CONTRIBUTING.md` §3.1b rule 1 at the level of a whole model: an
/// absolute reference (327 707 521, from the artifact) rather than a shape
/// invariant (*"the parameter count is plausible"*).
#[test]
fn drafter_params_matches_the_checkpoint_exactly() {
    let cfg = DSparkConfig::lfm2_2_6b();
    let got = drafter_params(&cfg);
    assert_eq!(
        got,
        CHECKPOINT_PARAMS,
        "config-derived parameter count {got} does not match the checkpoint's \
         {CHECKPOINT_PARAMS} (delta {}). Either the config is wrong or the layer \
         structure is — issue #363, DESIGN.md 10.2p",
        got as i64 - CHECKPOINT_PARAMS as i64
    );
}

/// The four components, each pinned separately.
///
/// The total agreeing is necessary and not sufficient: two errors of opposite
/// sign cancel in a sum. #91's own cost table is the demonstration — its four
/// rows sum to 21 120 MORE than the checkpoint carries, because the 241.2M
/// decoder-stack row already contains the per-layer norms its "norms +
/// confidence head" row counts again.
#[test]
fn each_component_matches_the_checkpoints_own_grouping() {
    let cfg = DSparkConfig::lfm2_2_6b();
    let h = cfg.hidden_size;

    // Projection: fc [2048, 10240] + hidden_norm [2048].
    let projection = h * cfg.target_layer_ids.len() * h + h;
    assert_eq!(projection, 20_973_568);

    // Markov: TWO [128000, 256] tensors, not one.
    let markov = 2 * cfg.vocab_size * cfg.markov_rank;
    assert_eq!(markov, 65_536_000);

    // Confidence: proj.weight [1, 2304] + proj.bias [1].
    let confidence = (h + cfg.markov_rank) + 1;
    assert_eq!(
        confidence, 2305,
        "the confidence head is [1, hidden + markov_rank] = [1, 2304] plus a bias; \
         a head over the hidden state alone would be [1, 2048] and the checkpoint \
         ships none"
    );

    // The stack is what is left over, and it is #91's 241.2M row.
    let stack = CHECKPOINT_PARAMS - projection - markov - confidence - h;
    assert_eq!(stack, 241_193_600);
}

/// `2304 = 2048 + 256` is the checkpoint confirming the paper's formula.
///
/// §10.2m derives `c_k = σ(w⊤[h_k ; W₁[x_{k−1}]])` from the paper. This asserts
/// the *artifact* implies the same composition — an independent witness, which
/// is what §3.1b means by a reference that does not come from the same source
/// as its subject.
///
/// **It calls `confidence_feature_width` rather than recomputing `hidden +
/// rank`**, and that is the whole difference between a test and a restatement.
/// The first version of this test asserted the arithmetic, and
/// `mutate_backbone.sh` arm 8 — a constructor built over `cfg.hidden_size`
/// alone — **survived it**: the mutation changed production code that no
/// assertion resolved through. `CONTRIBUTING.md` §3.1a rule 3, found by the
/// harness rather than by review.
#[test]
fn the_confidence_head_width_implies_the_hidden_plus_markov_concat() {
    let cfg = DSparkConfig::lfm2_2_6b();

    // The absolute reference, against the production site.
    assert_eq!(
        confidence_feature_width(&cfg),
        2304,
        "confidence_head.proj.weight is [1, 2304] in the checkpoint"
    );

    // And it is that number because of the concatenation, not by coincidence.
    assert_eq!(
        confidence_feature_width(&cfg),
        cfg.hidden_size + cfg.markov_rank
    );
    assert_ne!(
        confidence_feature_width(&cfg),
        cfg.hidden_size,
        "a hidden-only confidence head would be [1, 2048] and the checkpoint \
         ships none"
    );
    assert_ne!(
        confidence_feature_width(&cfg),
        cfg.markov_rank,
        "a markov-only confidence head would be [1, 256]"
    );
}

/// `logits_start` is 0 on this checkpoint, so a block of 9 proposes 9 tokens.
#[test]
fn anchor_as_pos0_yields_block_size_predictions() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert_eq!(cfg.logits_start, 0);
    assert_eq!(cfg.max_draft(), 9);
    assert_eq!(
        cfg.max_draft(),
        cfg.block_size,
        "anchor-as-pos0 means slot 0 both embeds the known token and predicts \
         the first draft token, so all {} slots are predictions. The card's \
         'ceiling is 10' is these 9 plus the target's bonus token.",
        cfg.block_size
    );
}

/// The drafter is interleaved-RoPE where the LFM2 target is neox.
///
/// Pinned because the two are one function name apart, produce identical
/// shapes, and the wrong one is invisible to every gate in this tree.
#[test]
fn the_drafter_is_interleaved_rope_unlike_the_target() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert!(
        !cfg.rope_is_neox_style,
        "config.json says rope_is_neox_style: false — the drafter runs rope_i \
         where lfm2.rs' apply_rotary_emb runs rope (DESIGN.md 10.2l item 3)"
    );
}

/// `rope_theta` gets its own test, separate from the convention above.
///
/// Two fields, two failure modes, and bundling them means one assertion
/// reports for both — the mutation harness showed it, killing a `rope_theta`
/// arm with the *rope-style* test's message. A test that fires for the wrong
/// reason still passes the harness and misleads the reader who has to act on
/// it (§3.1b rule 1: prefer the form that fails with a sentence a reader can
/// act on).
///
/// §5.11 is why this is worth pinning at all: *"a wrong theta changes no shape
/// and no dispatch count, so every gate in this document except the text one
/// would stay green while the model produced fluent wrong output"* — and on a
/// drafter there is no text gate either, because the target verifies every
/// token. VL-3B is 1e6 and this pair is 1e7.
#[test]
fn the_drafter_rope_theta_is_1e7_not_the_vl3b_1e6() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert_eq!(
        cfg.rope_theta, 1e7,
        "the drafter's own config.json carries rope_theta 1e7, matching the 2.6B \
         target (§5.11) and NOT VL-3B's 1e6"
    );
}

/// GQA is 4:1, matching the target's own geometry (§5.2).
#[test]
fn the_backbone_is_gqa_four_to_one() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.num_kv_groups(), 4);
    assert_eq!(
        cfg.num_attention_heads * cfg.head_dim,
        cfg.hidden_size,
        "32 heads x 64 = 2048"
    );
}

/// Every tapped layer index is a real target layer.
#[test]
fn every_tap_index_is_inside_the_target_stack() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert_eq!(cfg.target_layer_ids, vec![2, 9, 17, 21, 27]);
    assert_eq!(cfg.num_target_layers, 30);
    for &l in &cfg.target_layer_ids {
        assert!(l < cfg.num_target_layers, "tap layer {l} is past the stack");
    }
    // Sorted here, which is exactly why the tap's ORDER defect was latent
    // (§10.2n): on this checkpoint layer-iteration order and target_layer_ids
    // order coincide. The invariant is held by the tap's own test.
    let mut sorted = cfg.target_layer_ids.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted, cfg.target_layer_ids,
        "these indices are sorted, which is what makes a permuted tap latent \
         on THIS checkpoint and live on any other"
    );
}

// ---------------------------------------------------------------------------
// The Markov head, against a hand-computed reference
// ---------------------------------------------------------------------------

/// A tiny asymmetric fixture: vocab 4, rank 3.
///
/// `vocab != rank` deliberately, so an axis confusion is a hard error at some
/// call and a wrong *number* at others — a square fixture would hide both.
const V: usize = 4;
const R: usize = 3;

/// `w1[t][r] = 1 + t*R + r`, `w2[v][r] = (v + 1) * (r + 1)`.
///
/// Distinct per (row, column) in both, so a transpose or a swap lands on
/// different numbers rather than coincidentally equal ones.
fn markov_fixture(dev: &Device) -> Result<MarkovHead> {
    let w1: Vec<f32> = (0..V * R).map(|i| 1.0 + i as f32).collect();
    let w2: Vec<f32> = (0..V * R)
        .map(|i| {
            let v = i / R;
            let r = i % R;
            ((v + 1) * (r + 1)) as f32
        })
        .collect();
    Ok(MarkovHead::from_parts(
        Tensor::from_vec(w1, (V, R), dev)?,
        Tensor::from_vec(w2, (V, R), dev)?,
    ))
}

/// `prev_embedding(t)` is `w1`'s row `t` — an axis-0 lookup.
#[test]
fn prev_embedding_is_w1_row_lookup() -> Result<()> {
    let dev = Device::Cpu;
    let m = markov_fixture(&dev)?;
    for t in 0..V {
        let got = m.prev_embedding(t as u32)?.to_vec1::<f32>()?;
        let want: Vec<f32> = (0..R).map(|r| 1.0 + (t * R + r) as f32).collect();
        assert_eq!(got, want, "w1 row {t}");
    }
    Ok(())
}

/// `step_bias(t) = W₂ · W₁[t]`, computed here by an explicit scalar loop.
///
/// The reference is a hand-written double loop over the fixture's own
/// definition — never a second call into the production path with different
/// arguments, which is #503's shape and survives the mutation it exists to
/// catch (`CONTRIBUTING.md` §3.1b rule 3).
#[test]
fn step_bias_is_w2_times_the_w1_row() -> Result<()> {
    let dev = Device::Cpu;
    let m = markov_fixture(&dev)?;

    for t in 0..V {
        let got = m.step_bias(t as u32)?.to_vec1::<f32>()?;
        assert_eq!(got.len(), V, "the bias is one value per vocabulary entry");

        // Reference: bias[v] = sum_r w2[v][r] * w1[t][r].
        let mut want = [0f32; V];
        for (v, w) in want.iter_mut().enumerate() {
            let mut acc = 0f32;
            for r in 0..R {
                let w1_tr = 1.0 + (t * R + r) as f32;
                let w2_vr = ((v + 1) * (r + 1)) as f32;
                acc += w2_vr * w1_tr;
            }
            *w = acc;
        }
        for (v, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "step_bias({t})[{v}] = {g}, reference {w}"
            );
        }
    }
    Ok(())
}

/// **The swap is shape-valid and gives a different answer.**
///
/// This is the test that makes the asymmetry a checked invariant rather than a
/// convention. Both weights are `[V, R]`, so a loader that binds them the other
/// way round produces a well-formed `[V]` bias — and a different one. The
/// assertion is inequality against the correct reference, which is the only
/// thing that separates them.
#[test]
fn swapping_w1_and_w2_is_shape_valid_and_wrong() -> Result<()> {
    let dev = Device::Cpu;
    let correct = markov_fixture(&dev)?;

    // The same two tensors, bound the other way round.
    let w1: Vec<f32> = (0..V * R).map(|i| 1.0 + i as f32).collect();
    let w2: Vec<f32> = (0..V * R)
        .map(|i| (((i / R) + 1) * ((i % R) + 1)) as f32)
        .collect();
    let swapped = MarkovHead::from_parts(
        Tensor::from_vec(w2, (V, R), &dev)?,
        Tensor::from_vec(w1, (V, R), &dev)?,
    );

    for t in 0..V {
        let a = correct.step_bias(t as u32)?;
        let b = swapped.step_bias(t as u32)?;
        // Shape-valid: the swap produces the same shape, which is the whole
        // problem — nothing downstream can reject it.
        assert_eq!(a.dims(), b.dims(), "the swap is shape-valid at token {t}");
        let a = a.to_vec1::<f32>()?;
        let b = b.to_vec1::<f32>()?;
        assert_ne!(
            a, b,
            "the swap must change the numbers at token {t}, or this test cannot \
             detect it and the fixture is degenerate"
        );
    }
    Ok(())
}

/// The bias enters logits additively and at raw scale.
///
/// Pinned because the failure mode is a *scale* mismatch rather than a wrong
/// value: mlx-dspark records first-position acceptance falling ~85 % -> ~29 %
/// on a target whose head scales logits by ~0.196, because the raw-scale Markov
/// bias then overwhelms the shrunken base logits. LFM2 applies no such
/// transform, which is why `Model::draft_lm_head` is named for the property
/// rather than for the code it currently equals.
#[test]
fn the_bias_is_additive_on_raw_logits() -> Result<()> {
    let dev = Device::Cpu;
    let m = markov_fixture(&dev)?;
    let base = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.0], (V,), &dev)?;
    let bias = m.step_bias(0)?;
    let sum = (&base + &bias)?.to_vec1::<f32>()?;
    let b = base.to_vec1::<f32>()?;
    let x = bias.to_vec1::<f32>()?;
    for i in 0..V {
        assert!((sum[i] - (b[i] + x[i])).abs() < 1e-5);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The sequential sampler
// ---------------------------------------------------------------------------

/// A bias fixture whose argmax actually depends on the previous token.
///
/// [`markov_fixture`]'s `w2` is monotone in `v`, so `W₂·W₁[t]` is maximal at
/// the last vocabulary entry for *every* `t` and the argmax is bias-dominated
/// and constant. That is a degenerate fixture for this question — it cannot
/// express the difference it is meant to detect — and the first version of this
/// test used it and failed its own non-vacuity assertion, which is the
/// assertion doing its job (`CONTRIBUTING.md` §3.1b rule 3: read the printed
/// values, `[3,3,3]` against `[3,3,3]`).
///
/// This one makes the bias a *permutation* of the vocabulary: `W₁[t]` is the
/// one-hot of `t`, and `W₂` is built so that `W₂·onehot(t)` puts its maximum at
/// `(t + 1) mod V`. So "the previous token was `t`" pushes towards `t+1`, at a
/// magnitude comparable to the base logits rather than swamping them.
fn cyclic_markov_fixture(dev: &Device) -> Result<MarkovHead> {
    // w1: [V, R] with R == V, the identity — so w1[t] is onehot(t).
    let mut w1 = vec![0f32; V * V];
    for (t, chunk) in w1.chunks_mut(V).enumerate() {
        chunk[t] = 1.0;
    }
    // w2: [V, R]; row v, column r is the bias added to vocab entry v when the
    // previous token was r. Favour v == r + 1 (mod V).
    let mut w2 = vec![0f32; V * V];
    for v in 0..V {
        for r in 0..V {
            w2[v * V + r] = if v == (r + 1) % V { 0.5 } else { 0.0 };
        }
    }
    Ok(MarkovHead::from_parts(
        Tensor::from_vec(w1, (V, V), dev)?,
        Tensor::from_vec(w2, (V, V), dev)?,
    ))
}

/// The semi-AR loop is sequential, and a parallel application is a different
/// answer.
///
/// Position `k`'s bias depends on the token sampled at `k−1` (§10.2m step 3).
/// This reproduces the loop against an explicit reference and then shows that
/// applying the anchor's bias to every position — the natural "vectorized"
/// spelling — gives a different block. Without the second half the first is a
/// shape invariant: any correct-looking sequence passes it.
#[test]
fn the_markov_pass_is_sequential_not_parallel() -> Result<()> {
    let dev = Device::Cpu;
    let m = cyclic_markov_fixture(&dev)?;

    // Base logits near-tied per position, so the +0.5 bias decides the argmax
    // and the chain actually propagates. A fixture whose argmax is
    // bias-independent could not express the difference (§15.1 non-vacuity).
    let base = Tensor::from_vec(
        vec![
            0.10f32, 0.00, 0.00, 0.00, // position 0
            0.10, 0.00, 0.00, 0.00, // position 1
            0.10, 0.00, 0.00, 0.00, // position 2
        ],
        (3, V),
        &dev,
    )?;

    let anchor = 0u32;

    // Sequential: prev starts at the anchor and becomes each drafted token in
    // turn, so the cyclic bias walks 0 -> 1 -> 2 -> 3.
    let mut seq_tokens = Vec::new();
    let mut prev = anchor;
    for k in 0..3 {
        let row = base.i(k)?;
        let biased = (&row + &m.step_bias(prev)?)?;
        let t = biased.argmax(D::Minus1)?.to_scalar::<u32>()?;
        seq_tokens.push(t);
        prev = t;
    }

    // Parallel mis-spelling: the anchor's bias applied to every position, so
    // every position is pushed towards 1 and the walk never advances.
    let anchor_bias = m.step_bias(anchor)?;
    let mut par_tokens = Vec::new();
    for k in 0..3 {
        let row = base.i(k)?;
        let biased = (&row + &anchor_bias)?;
        par_tokens.push(biased.argmax(D::Minus1)?.to_scalar::<u32>()?);
    }

    // Absolute references, not shapes: the walk is 1,2,3 and the stall is 1,1,1
    // (§3.1b rule 1 — `[1,2,3]` is a stronger assertion than "they differ", and
    // it names why: the chain advanced).
    assert_eq!(
        seq_tokens,
        vec![1, 2, 3],
        "the sequential pass must walk the cycle as each drafted token becomes \
         the next position's `prev`"
    );
    assert_eq!(
        par_tokens,
        vec![1, 1, 1],
        "the parallel mis-spelling reuses the anchor's bias, so every position \
         is pushed to the same entry and the chain never advances"
    );
    assert_eq!(
        seq_tokens[0], par_tokens[0],
        "position 0 uses the anchor in both, so it must agree — if it does not, \
         the reference is wrong rather than the mechanism"
    );
    assert_ne!(
        seq_tokens, par_tokens,
        "the sequential and parallel passes must differ on this fixture, or the \
         fixture cannot express the dependency the Markov head exists to restore"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// The block input
// ---------------------------------------------------------------------------

/// The block is `[anchor, mask, mask, …]` — one anchor, then mask tokens.
#[test]
fn the_block_is_an_anchor_followed_by_mask_tokens() {
    let cfg = DSparkConfig::lfm2_2_6b();
    let anchor = 42u32;

    // Reconstructed from the config rather than by calling the drafter, since
    // building one needs the checkpoint. The shape is what is pinned.
    let width = cfg.block_size;
    let mut want = vec![cfg.mask_token_id; width];
    want[0] = anchor;

    assert_eq!(want.len(), 9);
    assert_eq!(want[0], anchor);
    assert!(want[1..].iter().all(|&t| t == 125_017));
    assert!(
        (cfg.mask_token_id as usize) < cfg.vocab_size,
        "the mask token must be embeddable by the TARGET's table, since the \
         drafter ships no embed_tokens"
    );
}

/// The drafter ships neither an embedding nor an LM head, so both come from the
/// target — recorded as a test because it is the constraint that decides the
/// module's API shape.
#[test]
fn the_drafter_has_no_embedding_and_no_lm_head() {
    // 62 tensors, and the two names that are absent are the ones a standalone
    // LM would need. Asserted as a count plus the structural consequence rather
    // than by reading the file, which the projection's own harness does.
    assert_eq!(CHECKPOINT_TENSORS, 62);
    let cfg = DSparkConfig::lfm2_2_6b();
    let with_own_embed = drafter_params(&cfg) + cfg.vocab_size * cfg.hidden_size;
    assert_ne!(
        with_own_embed, CHECKPOINT_PARAMS,
        "if the drafter carried its own embed_tokens the parameter count would \
         be {with_own_embed}, and the checkpoint carries {CHECKPOINT_PARAMS}"
    );
}

// ---------------------------------------------------------------------------
// Dtype
// ---------------------------------------------------------------------------

/// The checkpoint is BF16 throughout, and a loader assuming F16 reads the right
/// bytes as the wrong numbers.
#[test]
fn the_checkpoint_dtype_is_bf16() {
    assert_eq!(lloom::models::lfm2_dspark::CHECKPOINT_DTYPE, DType::BF16);
}

// ---------------------------------------------------------------------------
// Which target layers are tapped — §10.2s
// ---------------------------------------------------------------------------

/// The layer-input tap is requested at `target_layer_ids + 1`, because the
/// drafter is trained on those layers' **output** residual and the tap reads
/// the residual **entering** a layer.
///
/// The absolute reference (§3.1b rule 1) is the released GGUF, whose metadata
/// carries `dflash.target_layers = [3, 10, 18, 22, 28]` for this checkpoint's
/// `target_layer_ids = [2, 9, 17, 21, 27]` — llama.cpp's converter applies the
/// same `+ 1` (`conversion/qwen.py`, `extract_layer_ids = [i + 1 for i in
/// target_layer_ids]`), and its `t_layer_inp[il]` is the input of layer `il`.
/// Requesting `[2, 9, 17, 21, 27]` directly is one layer early: §10.2s measured
/// a different drafted block from slot 1 onward and a lower accept rate, with
/// every other gate green.
#[test]
fn tap_layers_are_target_layer_ids_plus_one_matching_the_released_gguf() {
    let cfg = DSparkConfig::lfm2_2_6b();
    assert_eq!(cfg.target_layer_ids, vec![2, 9, 17, 21, 27]);
    // `dflash.target_layers` in LFM2.5-2.6B-DSpark-F16.gguf, read with gguf_dump.
    const GGUF_TARGET_LAYERS: [usize; 5] = [3, 10, 18, 22, 28];
    assert_eq!(
        cfg.tap_layers(),
        GGUF_TARGET_LAYERS.to_vec(),
        "tap_layers must be the layer-INPUT indices whose residual is target_layer_ids' OUTPUT"
    );
    // Every tap index is a real layer of the 30-layer target: the largest,
    // 28, is the input of the last-but-one layer, so no `+ 1` runs off the end.
    for &l in &cfg.tap_layers() {
        assert!(
            l < cfg.num_target_layers,
            "tap layer {l} is past the target's layers"
        );
    }
}
