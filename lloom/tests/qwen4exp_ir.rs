//! The optimizer on the Qwen4-exp decode step (lloom #749 P31 (6)): the IR
//! `qwen4exp_ir::decode_step` emits reconciles against the traced step of
//! the real model, and the derivation over it reports what it can and cannot
//! fuse. No GPU, no weights.
//!
//! Every count is read from the artifact it cites
//! (`design/measurements/qwen4exp-dispatches-raw/p31-trace-driver-pair.txt`,
//! run 1, step 10: `dispatches=1628`), the trace's kernel names collapsed to
//! the IR's families: `kernel_mul_mv_id2_*` → `mul_mv_id2`,
//! `kernel_mul_mv_id_*` → `mul_mv_id`, `kernel_mul_mv_*` → `mul_mv`,
//! `gemv_mixed_f16w_*[_silu|_sigmoid]` → `gemv_mixed[_silu|_sigmoid]`,
//! `gemv_float32_*` → `gemv_f32`, the dtype/strided suffixes dropped.

use lloom::models::qwen4exp::Config;
use lloom::models::qwen4exp_ir::decode_step;
use lloom::optimizer::{derive, plan, Policy};

#[test]
fn the_ir_reconciles_with_the_traced_step() {
    let cfg = Config::flash_next();
    let g = decode_step(&cfg);
    let census: Vec<(String, u64)> = g.census().into_iter().collect();
    let expected: Vec<(&str, u64)> = vec![
        ("abs", 1),
        ("affine", 1),
        ("badd", 5),
        ("bmaximum", 1),
        ("bminimum", 1),
        ("bmul", 10),
        ("bsub", 1),
        ("cast_u8", 2),
        ("copy2d_f32", 39),
        ("copy2d_u32", 48),
        ("copy_f32_strided", 1),
        ("fast_argmax", 1),
        ("fast_sum", 1),
        ("gdn_gated_norm", 36),
        ("gdn_prologue", 36),
        ("gdn_step", 36),
        ("gemv_f32", 48),
        ("gemv_mixed", 194),
        ("gemv_mixed_sigmoid", 97),
        ("gemv_mixed_silu", 97),
        ("get_rows", 1),
        ("gt", 1),
        ("hc_collapse", 97),
        ("hc_combine", 96),
        ("hc_norm", 97),
        ("lt", 1),
        ("moe_combine", 48),
        ("moe_slot_map", 48),
        ("moe_topk", 48),
        ("mul_mv", 301),
        ("mul_mv_id", 48),
        ("mul_mv_id2", 48),
        ("qsa_prep", 12),
        ("rmsnorm", 3),
        ("sdpa_vector", 12),
        ("sigmoid", 1),
        ("sigmoid_mul", 12),
        ("silu", 1),
        ("silu_mul", 96),
        ("sqrt", 1),
    ];
    let got: Vec<(&str, u64)> = census.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(got, expected, "the IR's census against the traced step's");
    assert_eq!(g.dispatches(), 1628, "`TRACE run=1 step=10 dispatches=1628`");
}

/// The derivation over the step: nothing to fuse — every chain of this model
/// is hand-fused or sits on a quantized core without an epilogue seam — and
/// the seam report is empty for the same reason (the routed chain is the
/// first seam a fragment for the quantized cores would open).
#[test]
fn the_derivation_reports_the_seams_and_fuses_nothing() {
    let cfg = Config::flash_next();
    let g = decode_step(&cfg);
    // `plan::seams` reads GEMV → elementwise edges; every producer here is
    // opaque (a quantized core, a mixed gemv with its epilogue), so the
    // report is empty by construction — the routed chain
    // `mul_mv_id2 > silu_mul > mul_mv_id` becomes a seam the day the
    // quantized cores are an `Op` with a fragment seam.
    let seams = plan::seams(&g);
    assert!(seams.is_empty(), "seams: {seams:?}");
    let d = derive(&g, &Policy::default()).expect("derivation");
    eprintln!("plan: {}", d.plan);
    eprintln!("egglog: {:.2} ms", d.elapsed.as_secs_f64() * 1e3);
    assert_eq!(d.plan.dispatches_before(), 1628);
    assert_eq!(
        d.plan.dispatches_after(),
        1628,
        "the catalogue's fragments bind to the f16 gemv; this step has none unfused"
    );
}

/// E2 of #749's closing plan: the step's dependency depth `D` — the longest
/// chain of data-dependent dispatches, the serial boundaries no reordering
/// removes — and its width. Every dispatch sits at exactly one depth, so the
/// widths sum to the census's 1 628.
#[test]
fn the_step_depth_and_width() {
    let cfg = Config::flash_next();
    let g = decode_step(&cfg);
    let p = g.depth_profile();
    eprintln!(
        "DEPTH qwen4exp decode_step D={} max_width={} dispatches={} widths={:?}",
        p.depth(),
        p.max_width(),
        g.dispatches(),
        p.widths
    );
    assert_eq!(p.widths.iter().map(|&w| w as u64).sum::<u64>(), 1628);
    // The values #749 P33 E2 recorded: an emitter change that moves them owes
    // the ledger a new E2.
    assert_eq!((p.depth(), p.max_width()), (1157, 4));
}
