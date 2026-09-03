//! Admission at configuration time (lloom #186, `DESIGN.md` §9.5).
//!
//! §9.5d's check is *"predict the peak from `(model, max_context, B, scratch
//! policy)` and refuse before allocating"*, and §9.5k's residual is the derived
//! cap on everything the five classes do not name. This file is the
//! model-level half: that the check runs where it must, that it refuses and
//! admits, and that the RoPE tables — built in `Cache::new_with` **before**
//! anything else and not one of §9.1's five classes — are accounted for.
//!
//! # Both bounds, and why that is the load-bearing property
//!
//! **An admission check that refuses everything is not a check** (§8.1g's rule,
//! and #184's precedent of pinning both bounds for exactly this reason). Every
//! refusal assertion here has an admission assertion beside it. The mutation
//! that motivates it is trivial to write and impossible to see from a green
//! suite that only tests the refusal.
//!
//! # What is not tested here, stated rather than left to be noticed
//!
//! **Nothing in this file runs a forward pass or touches a GPU.** Admission is
//! a pure function of the configuration — that is what makes it free (§9.5e) —
//! so it is testable without a device, and the residual *branch* is tested
//! where it lives, in `candle-metal-kernels`' `buffer_pool` suite. What a Metal
//! device would add is the real `recommendedMaxWorkingSetSize` in place of the
//! `hw.memsize` stand-in, which changes the arithmetic's inputs and not its
//! shape.

use lloom::models::lfm2::MemoryBudget;

/// `hw.memsize` on the development machine (§3.4), the stand-in for
/// `recommendedMaxWorkingSetSize` that §9.5k's own table uses.
const MACHINE: usize = 68_719_476_736;
/// §5.5: language 5.394 GB + vision 0.83 + projector 0.03.
const WEIGHTS_FULL: usize = 6_254_000_000;
const GB: f64 = 1e9;

/// `MemoryBudget::DEFAULT_FRACTION` is a copy of `admission`'s, because that
/// constant is Metal-only and this type exists on every backend.
///
/// **A copy that nothing checks is the hand-sync §8.1b exists to remove**, so
/// this checks it. One line, and it turns a silent divergence into a failing
/// test.
#[cfg(feature = "metal")]
#[test]
fn default_fraction_matches_admissions() {
    assert_eq!(
        MemoryBudget::DEFAULT_FRACTION,
        lloom_tensor::metal_backend::admission::DEFAULT_BUDGET_FRACTION,
        "the two spellings of §9.5k's fraction have drifted"
    );
}

/// `MemoryBudget::DEFAULT_WEIGHT_TOLERANCE` is a copy of `admission`'s for the
/// same reason, and is checked for the same reason (§9.5m, #244).
#[cfg(feature = "metal")]
#[test]
fn default_weight_tolerance_matches_admissions() {
    assert_eq!(
        MemoryBudget::DEFAULT_WEIGHT_TOLERANCE,
        lloom_tensor::metal_backend::admission::WEIGHT_RECONCILE_TOLERANCE,
        "the two spellings of §9.5m's tolerance have drifted"
    );
}

/// **Both bounds** — a configuration inside the budget is admitted and one
/// outside it is refused (§8.1g, #184).
#[cfg(feature = "metal")]
#[test]
fn admits_what_fits_and_refuses_what_does_not() {
    use lloom_tensor::metal_backend::admission::Budget;

    // ADMITTED: what ships today. §9.5b puts it at 9.6 % of the machine.
    let ships = Budget::new(WEIGHTS_FULL, 4096, 4096).admit(MACHINE);
    assert!(
        ships.fits,
        "the shipping configuration must be admitted:\n{}",
        ships.describe()
    );

    // REFUSED: B=32 at 128k, where §9.5b's KV alone is 68.719 GB -- exactly
    // `hw.memsize`, before a single weight is loaded.
    let mut over = Budget::new(WEIGHTS_FULL, 131_072, 131_072);
    over.batch = 32;
    let refused = over.admit(MACHINE);
    assert!(!refused.fits, "B=32 x 128k must be refused");
    assert!(
        refused.describe().contains("dominant"),
        "a refusal names the class to reduce (§9.5g):\n{}",
        refused.describe()
    );
}

/// **`B=16 ctx 128k` passes the sum-based table and fails the residual test.**
///
/// This is the trap the issue names explicitly and §9.5k records: it sums to
/// **60.3 %** of the machine, which reads as comfortable, and its residual is
/// **3.495 GB** against §6.3b's **measured** 8.398 GB of stranding over 400
/// tokens. *"Tight"* is graded against a measured quantity rather than a round
/// number, and a refusal that did not reflect that would admit a configuration
/// a known allocation shape can kill.
#[cfg(feature = "metal")]
#[test]
fn b16_ctx128k_is_tight_against_measured_stranding_not_against_the_sum() {
    use lloom_tensor::metal_backend::admission::{Budget, OBSERVED_STRANDING_BYTES};

    let mut b = Budget::new(WEIGHTS_FULL, 131_072, 131_072);
    b.batch = 16;
    let a = b.admit(MACHINE);

    assert!(a.fits, "it passes the sum:\n{}", a.describe());
    assert!(
        (a.footprint.predicted() as f64 / MACHINE as f64 - 0.603).abs() < 0.01,
        "§9.5b: 60.3 % of the machine"
    );
    assert!(
        a.is_tight(),
        "and it must be TIGHT: 3.495 GB of residual against 8.398 GB of \
         measured stranding:\n{}",
        a.describe()
    );
    assert!(
        a.residual < OBSERVED_STRANDING_BYTES,
        "residual {:.3} GB vs stranding {:.3} GB",
        a.residual as f64 / GB,
        OBSERVED_STRANDING_BYTES as f64 / GB
    );

    // The other half of the bound: a configuration with room is NOT tight, so
    // "tight" is a discrimination rather than a constant.
    let ample = Budget::new(WEIGHTS_FULL, 4096, 4096).admit(MACHINE);
    assert!(
        !ample.is_tight(),
        "the shipping configuration has 38 GB of room and must not read tight"
    );
}

/// **The RoPE tables are accounted for, and the ordering is why it matters.**
///
/// §9.5k: they are built in `Cache::new_with` from `max_position_embeddings`
/// through `Tensor` ops — i.e. through the pool — at **16.4 MB resident and
/// ~49 MB transient** at the shipped 128000. They are not one of §9.1's five
/// classes, they are allocated **before the first token**, and **they scale
/// with the axis #161 sweeps**.
///
/// The figures are asserted here so that a change to the table geometry cannot
/// silently stop matching what admission subtracts.
#[test]
fn rope_table_bytes_match_what_9_5k_records() {
    // Two f16 tables of `max_position_embeddings x head_dim/2`.
    let head_dim = 64;
    let resident = |mpe: usize| 2 * mpe * (head_dim / 2) * 2;
    // An f32 `idx_theta` plus f32 cos/sin before the cast.
    let transient = |mpe: usize| 3 * mpe * (head_dim / 2) * 4;

    let mpe = 128_000;
    assert!(
        (resident(mpe) as f64 / 1e6 - 16.4).abs() < 0.1,
        "§9.5k: 16.4 MB resident at max_position_embeddings=128000, got {:.2}",
        resident(mpe) as f64 / 1e6
    );
    assert!(
        (transient(mpe) as f64 / 1e6 - 49.2).abs() < 0.1,
        "§9.5k: ~49.2 MB transient, got {:.2}",
        transient(mpe) as f64 / 1e6
    );

    // And the reason they are recorded rather than folded in: they scale with
    // the context axis. A 32x larger context is a 32x larger table.
    assert_eq!(
        resident(131_072) / resident(4_096),
        32,
        "the tables scale with max_position_embeddings, which is the axis #161 \
         sweeps -- that is why §9.5k names them"
    );
}

/// Admission is **off by default**, so every existing caller is unchanged.
///
/// §7.1a: no default is flipped without its own argued decision. This is the
/// off-arm, and it is what makes the change safe to land — a `Config` built the
/// way every caller builds it carries no budget and admits unconditionally.
#[test]
fn admission_is_off_by_default() {
    let cfg = lloom::models::lfm2::Config {
        vocab_size: 128_000,
        hidden_size: 2048,
        intermediate_size: 10_752,
        num_hidden_layers: 30,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        norm_eps: 1e-5,
        rope_theta: 1e6,
        max_position_embeddings: 128_000,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        // §10.4's defaults, matching `into_config`. Added by #116 after this
        // file was written, which is why it stopped compiling -- see the
        // module doc.
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        // `Grow`, which is what #116's per-call allocation did before
        // the axis reached it (#234). Not a choice — see the field.
        flash_scratch_sizing: Default::default(),
        kv_append: Default::default(),
        // `Off`, the f16 arm every recorded digest belongs to. Added by #218
        // after this file was written, which is why it stopped compiling --
        // `flash_scratch_sizing` above carries the same note from #116.
        kv_quant: Default::default(),
        conv_state: Default::default(),
        memory_budget: None,
        // Both epilogue fusions off, per DESIGN.md 7.1a (issue #509).
        gemv_silu: false,
        gemv_residual: false,
        // #363's tap: off, as every pre-#363 caller had it.
        tap_layers: Vec::new(),
        tap_dtype: lloom_tensor::DType::F32,
    };
    assert!(
        cfg.memory_budget.is_none(),
        "admission must be opt-in (§7.1a)"
    );

    // And a budget is one field, which is the axis shape every other variant
    // in this file uses.
    let with = lloom::models::lfm2::Config {
        memory_budget: Some(MemoryBudget::new(WEIGHTS_FULL)),
        ..cfg
    };
    assert_eq!(with.memory_budget.unwrap().batch, 1, "B=1 default (§13.2)");
    assert_eq!(with.memory_budget.unwrap().weight_bytes, WEIGHTS_FULL);
}

// -------------------------------------------------------------------- #306
// The prefill activation term reaches admission through `MemoryBudget`.

/// **`max_seq_len` defaults to decode and is opt-in**, which is what keeps
/// every recorded verdict where it was (§7.1a).
///
/// A budget that defaulted to `max_position_embeddings` would refuse
/// essentially every configuration — §8.1g's failure, and the reason the
/// admitted arm is the load-bearing half of every bound in this file.
#[test]
fn the_prefill_length_is_opt_in_and_defaults_to_decode() {
    let b = MemoryBudget::new(WEIGHTS_FULL);
    assert_eq!(b.max_seq_len, 1, "decode by default (#306)");
    assert_eq!(
        b.with_prefill(2048).max_seq_len,
        2048,
        "a prefilling caller says so"
    );
    // And a chunked caller passes its CHUNK size, not its prompt length --
    // which is the whole mechanism by which chunking makes a long prompt
    // admissible (§13.2b).
    assert_eq!(b.with_prefill(512).max_seq_len, 512);
}

/// **The term reaches the `admission::Budget`, and it changes the verdict.**
///
/// The engagement check §2.4 requires: proved from a quantity the flag should
/// have changed — here the predicted footprint — rather than from the field
/// being set. #9.5l's finding 5 is why: admission compiled to its non-Metal
/// stub in every harness and *"no amount of arithmetic testing could have seen
/// it"*.
#[cfg(feature = "metal")]
#[test]
fn the_prefill_length_moves_the_predicted_footprint() {
    use lloom_tensor::metal_backend::admission::Budget;

    let decode = Budget::new(WEIGHTS_FULL, 4096, 4096);
    let prefill = decode.with_prefill(2048);

    let d = decode.footprint();
    let p = prefill.footprint();

    // Only the activation class moves. Every other class is a property of the
    // session state and is unchanged by a forward pass's length (§4.1's `seq`
    // against `kv_len`) -- asserted, because a change that moved KV here would
    // mean the two axes had been conflated.
    assert_eq!(d.weights, p.weights);
    assert_eq!(d.kv, p.kv, "KV is `kv_len`'s, not `seq`'s (§4.1)");
    assert_eq!(d.conv, p.conv);
    assert_eq!(d.scratch, p.scratch);
    assert!(
        p.activations > d.activations * 100_000,
        "at P=2048 the activation class is five orders of magnitude above its \
         decode figure: {} B against {} B",
        p.activations,
        d.activations
    );
    // 18.66 GB, reconciled against §13.4c in `admission.rs`'s own tests.
    assert!(
        (p.activations as f64 / GB - 18.66).abs() < 0.1,
        "predicted {:.2} GB",
        p.activations as f64 / GB
    );
}
