//! Admission against a real Metal device (lloom #186, `DESIGN.md` §9.5l).
//!
//! The unit tests in `lfm2_admission_tests.rs` are pure arithmetic against a
//! `hw.memsize` stand-in. **This one builds a `Cache` on the device**, which is
//! the only thing that exercises three properties the arithmetic cannot:
//!
//! * that `admit_memory_budget` is *reached* from `Cache::new_with` at all;
//! * that it runs **before** the RoPE tables, which are allocated in that same
//!   function and are the reason the ordering matters (§9.5k);
//! * that the real denominator is used, which is **not** `hw.memsize` —
//!   `recommendedMaxWorkingSetSize` reads **55.663 GB** on this machine against
//!   `hw.memsize`'s 68.719 (§9.5l).
//!
//! **This was worth writing, and the reason is recorded rather than implied.**
//! The unit tests were green while admission was compiling to its non-Metal
//! stub in every harness — `candle-examples`' `metal` feature did not propagate
//! `candle-transformers/metal`, so the `#[cfg(feature = "metal")]` arm was
//! never built. Nothing that tested the *arithmetic* could have seen that; only
//! building a `Cache` and watching it refuse can.

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device};
use lloom::models::lfm2::{Cache, Config, LayerType, MemoryBudget};

/// §5.5's language-model weights, which is what a text-only LFM2 load carries.
const WEIGHTS: usize = 5_394_397_184;

fn config(max_position_embeddings: usize) -> Config {
    Config {
        vocab_size: 128_000,
        hidden_size: 2048,
        intermediate_size: 10_752,
        num_hidden_layers: 30,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        norm_eps: 1e-5,
        rope_theta: 1e6,
        max_position_embeddings,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv; 30],
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
        tap_dtype: DType::F32,
    }
}

/// **Both bounds, on the device** (§8.1g, #184): a configuration inside the
/// budget builds a `Cache`, and one outside it fails to.
#[test]
fn admits_what_fits_and_refuses_what_does_not_on_the_device() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };

    // ADMITTED: B=1 at the shipped capacity. §9.5b puts this well inside.
    let mut ok = config(4096);
    ok.memory_budget = Some(MemoryBudget::new(WEIGHTS));
    Cache::new_with(true, DType::F16, &ok, &device, 4096)
        .expect("the shipping configuration must be admitted");

    // REFUSED: B=32 at 128k, where §9.5b's KV alone is 68.719 GB -- more than
    // this machine's whole working set, before a weight is loaded.
    let mut over = config(131_072);
    let mut budget = MemoryBudget::new(WEIGHTS);
    budget.batch = 32;
    over.memory_budget = Some(budget);
    let err = Cache::new_with(true, DType::F16, &over, &device, 131_072)
        .expect_err("B=32 x 128k must be refused");
    let msg = err.to_string();

    // §9.5g: the refusal carries the arithmetic that produced it, names the
    // dominant class, and lists the levers -- so the next configuration is a
    // decision rather than a guess.
    assert!(msg.contains("memory budget exceeded"), "{msg}");
    assert!(
        msg.contains("<- dominant"),
        "names the class to reduce: {msg}"
    );
    assert!(msg.contains("reductions that would fit"), "{msg}");
    // And the RoPE tables are named, since they are allocated in this very
    // function and are not one of §9.1's five classes (§9.5k).
    assert!(msg.contains("RoPE"), "{msg}");
}

/// **Admission off by default builds exactly as it did before.**
///
/// The off-arm on the device rather than in the abstract: a `Config` with no
/// budget must reach the same `Cache` it always did, including the RoPE tables
/// admission would otherwise have been asked about.
#[test]
fn no_budget_means_no_refusal_on_the_device() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // A configuration admission would refuse outright -- if it were consulted.
    let cfg = config(131_072);
    assert!(cfg.memory_budget.is_none());
    Cache::new_with(true, DType::F16, &cfg, &device, 131_072)
        .expect("with no budget set, nothing is refused however large");
}

// ---- #244: the weight term is caller-supplied, and nothing checked it ----

/// **A caller passing the WRONG weight constant is detected on the device**
/// (§9.5m, #244) — and this is the test the issue says is the whole point.
///
/// # Why this could not be an arithmetic test
///
/// `lfm2_admission_tests.rs` and this file's existing cases **both pass
/// `WEIGHTS` / `WEIGHTS_FULL` as constants**, so no test exercised a caller
/// getting it wrong, and the arithmetic is correct for whatever it is handed.
/// That is #240's shape one level up: the *arithmetic* was tested and the
/// *contract* was not.
///
/// So this test does not check arithmetic. It **allocates real bytes on the
/// device**, then tells admission a figure that does not match them, and checks
/// that the divergence is seen. `reconcile_weights` reads the pool's own
/// counters — from outside the model's arithmetic, per #162 — which is why an
/// allocation has to actually happen for this to mean anything.
#[test]
fn a_caller_passing_the_wrong_weight_constant_is_detected_on_the_device() {
    use lloom_tensor::metal_backend::admission::{reconcile_weights, WEIGHT_RECONCILE_TOLERANCE};
    use lloom_tensor::Tensor;

    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Ok(metal) = device.as_metal_device() else {
        eprintln!("not a Metal device; skipping");
        return;
    };

    // Allocate a known quantity into the pool the WEIGHTS are served from,
    // standing in for a weight load. 512 MB is above the 256 MB tolerance, so a
    // caller that claims zero is wrong by more than the slack -- which is the
    // discrimination under test rather than a magnitude that happens to be
    // large.
    //
    // **`to_dtype`, not `Tensor::zeros`, and the distinction is §9.5k's.**
    // `Tensor::zeros` goes `allocate_buffer` -> `self.buffers`, the SHARED pool
    // that serves the KV reserve. Weights go `to_dtype` -> `new_buffer_builder`
    // -> `private_buffers`. Writing this test the obvious way allocated into one
    // pool and read the other, and it read 0 bytes -- which is the same
    // conflation §9.5k had to read the allocation paths to get right, met from
    // the test side.
    const ALLOCATED: usize = 512 * 1_000_000;
    let elems = ALLOCATED / 2; // f16
    let held = Tensor::zeros(elems, DType::F32, &device)
        .expect("allocate")
        .to_dtype(DType::F16)
        .expect("into private_buffers");
    let (_, private) = metal.pool_occupancy();
    assert!(
        private.live_bytes >= ALLOCATED,
        "the private pool must actually hold the bytes this test reasons about, \
         got {} MB -- if this fails the allocation path has moved (§9.5k)",
        private.live_bytes / 1_000_000
    );

    // A caller that UNDER-predicts: it told admission there were no weights
    // while the process allocated 512 MB. This is the silent direction -- §3.5
    // reports no overrun, so without this check nothing anywhere says so.
    let under = reconcile_weights(0, &private, 0, WEIGHT_RECONCILE_TOLERANCE);
    assert!(
        under.under_predicted(),
        "a caller that under-predicts by 512 MB on a real device must be \
         detected -- this is the direction that fails silently:\n{}",
        under.describe()
    );
    assert!(
        under.describe().contains("UNDER-PREDICTED"),
        "{}",
        under.describe()
    );

    // **The other bound, without which the above is worthless** (§8.1g): a
    // caller that tells the truth about the same pool must be quiet.
    let honest = reconcile_weights(private.live_bytes, &private, 0, WEIGHT_RECONCILE_TOLERANCE);
    assert!(
        honest.agrees(),
        "a caller that got it right must not be reported wrong:\n{}",
        honest.describe()
    );

    // And over-prediction reads as over-prediction, not as the silent
    // direction: #162's own case, the full checkpoint passed for a text-only
    // load, is a real error and a LOUD one.
    let over = reconcile_weights(
        private.live_bytes + 825_299_424,
        &private,
        0,
        WEIGHT_RECONCILE_TOLERANCE,
    );
    assert!(over.over_predicted(), "{}", over.describe());
    assert!(
        !over.under_predicted(),
        "the two directions must not be conflated:\n{}",
        over.describe()
    );

    drop(held);
}

/// **The reconciliation is REACHED from `Cache::new_with`**, not merely
/// present — §9.5l finding 5's lesson, which is this file's whole reason to
/// exist (§9.5m, #244).
///
/// The unit tests were green while admission compiled to its non-Metal stub in
/// every harness. Nothing that tests arithmetic could have seen that; only
/// building a `Cache` on a device can. This builds one with a budget whose
/// weight figure is deliberately wrong and asserts the path runs to completion
/// — the check reports rather than refuses (see `report_weight_divergence` for
/// why), so what is asserted here is that a wrong figure does not break the
/// build, and the detection itself is asserted in the test above.
#[test]
fn a_wrong_weight_figure_reports_without_refusing_a_model_that_fits() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };

    // A budget claiming the full VL checkpoint for what is a text-only load.
    // #162's case: over-predicted by 0.825 GB, and it still fits, so the Cache
    // must still build -- refusing here would turn a bookkeeping error into a
    // refusal to run a model that has room.
    let mut cfg = config(4096);
    cfg.memory_budget = Some(MemoryBudget::new(WEIGHTS + 825_299_424));
    Cache::new_with(true, DType::F16, &cfg, &device, 4096)
        .expect("an over-predicted weight figure that still fits must not refuse");

    // And the under-predicting direction likewise builds: the bytes are already
    // spent by the time this runs, so the report is the deliverable.
    let mut under = config(4096);
    let mut budget = MemoryBudget::new(0);
    budget.weight_tolerance = MemoryBudget::DEFAULT_WEIGHT_TOLERANCE;
    under.memory_budget = Some(budget);
    Cache::new_with(true, DType::F16, &under, &device, 4096)
        .expect("an under-predicted weight figure reports; it does not refuse");
}
