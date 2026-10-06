//! The optimizer on the Qwen4-exp decode step, through the one walker
//! (issue #732): the IR is the walk of the committed checkpoint's model —
//! the loader's own assembly over ghost weights, every stem the kind the
//! header stores (`testing::ghost_model`) — and the derivation over it
//! decides the HyperConnection brackets' epilogues, which is the hand-wired
//! composition today's forward takes. No GPU, no weights.
//!
//! Every count is an ABSOLUTE reference read from the artifact it cites
//! (`CONTRIBUTING.md` §3.1b rule 1).

use lloom::models::qwen4exp::testing::{ghost_model, GhostExperts};
use lloom::models::qwen4exp::Qwen4Exp;
use lloom::optimizer::{derive, walk, Graph, NumericClass, Op, Policy};
use lloom_tensor::{DType, Device};

fn model() -> Qwen4Exp<GhostExperts> {
    ghost_model(&Device::Cpu).expect("the ghost of the committed manifest")
}

/// The blocking decode loop's two launches, the driver's and not the
/// model's (`generation/readback.rs` `sample_and_feed` →
/// `draw_and_read`): the on-device argmax and the id's copy into the slot
/// the host reads. Appended so a census compares with a trace of the step.
fn with_driver(mut g: Graph) -> Graph {
    use lloom::optimizer::Shape;
    // The logits: the step's last value (the other roots are its state
    // writes, emitted inside the layers).
    let root = *g.roots().iter().max().expect("a root");
    let next = g.opaque("fast_argmax", &[root], Shape::of(&[1]), DType::U32);
    let read = g.opaque("copy2d_u32", &[next], Shape::of(&[1]), DType::U32);
    g.mark_root(read);
    g
}

/// The decode step as dev traced it: `qwen4exp-chat` under
/// `CANDLE_METAL_TRACE=1`, streamed IQ3_XXS, `--slots auto`, the prose
/// prompt, run 1 step 9 — the device plan armed, inside the indexer's dense
/// range, no block completing
/// (`design/measurements/issue-732-raw/walker/qwen4exp-trace-dev-bfc70b8f4.txt`,
/// `TRACE run=1 step=9 dispatches=1581`). The trace's kernel names collapsed
/// to the IR's families: `kernel_mul_mv_id2_*` → `mul_mv_id2`,
/// `kernel_mul_mv_id_*` → `mul_mv_id`, `kernel_mul_mv_*` → `mul_mv`,
/// `gemv_mixed_f16w_*[_silu|_sigmoid]` → `gemv_mixed[_silu|_sigmoid]`,
/// `gemv_float32_*` → `gemv`, `gdn_step_r16` → `gdn_step`,
/// `kernel_get_rows_*` → `get_rows`, `sdpa_vector_float_256` →
/// `sdpa_vector`, `bmul_f32_strided` → `bmul`, the `_f32`/`_cs` suffixes
/// dropped where the family carries none in the IR.
const TRACED_STEP: &[(&str, u64)] = &[
    ("abs", 1),
    ("affine", 1),
    ("badd", 5),
    ("bmaximum", 1),
    ("bminimum", 1),
    ("bmul", 10),
    ("bsub", 1),
    ("cast_u8", 2),
    ("copy2d_f32", 39),
    ("copy2d_u32", 1),
    ("copy_f32_strided", 1),
    ("fast_argmax", 1),
    ("fast_sum", 1),
    ("gdn_gated_norm", 36),
    ("gdn_prologue", 36),
    ("gdn_step", 36),
    ("gemv", 48),
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

/// **The walk is today's step.** The derived walk, with the driver's two
/// launches, equals the traced step family by family, 1 581 dispatches;
/// the source walk is the same step with every bracket's epilogues
/// unfused — 97 `affine`, `silu` and `sigmoid` more, the 194 mixed
/// products bare.
#[test]
fn the_derived_walk_is_the_traced_step() {
    let source = walk(&model());
    let derived = derive(&source, &Policy::default()).expect("derive").graph;
    let got: Vec<(String, u64)> = with_driver(derived.clone()).census().into_iter().collect();
    let got: Vec<(&str, u64)> = got.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(
        got, TRACED_STEP,
        "the derived walk's census against the trace's"
    );
    assert_eq!(TRACED_STEP.iter().map(|(_, n)| n).sum::<u64>(), 1581);
    assert_eq!(
        derived.dispatches(),
        1579,
        "the model's step, the driver's two aside"
    );
    let src = source.census();
    assert_eq!(
        (
            src.get("gemv_mixed"),
            src.get("gemv_mixed_silu"),
            src.get("affine"),
            src.get("silu"),
            src.get("sigmoid")
        ),
        (Some(&388), None, Some(&98), Some(&98), Some(&98))
    );
    assert_eq!(source.dispatches(), 1579 + 3 * 97);
}

/// E2 of #749's ledger, re-read through the walk: the derived step's
/// dependency depth and widest level — `(1097, 5)`, where the emitter's
/// `(1157, 4)` described the P31 trace's step (a record copy per layer, the
/// PLE chain named from counts, the head's collapse called the PLE's).
#[test]
fn the_step_depth_and_width() {
    let derived = derive(&walk(&model()), &Policy::default())
        .expect("derive")
        .graph;
    let p = derived.depth_profile();
    eprintln!(
        "DEPTH qwen4exp walk D={} max_width={} dispatches={} widths={:?}",
        p.depth(),
        p.max_width(),
        derived.dispatches(),
        p.widths
    );
    assert_eq!(p.widths.iter().map(|&w| w as u64).sum::<u64>(), 1579);
    assert_eq!((p.depth(), p.max_width()), (1097, 5));
}

/// 48 layers × 2 brackets + the head's: every bracket's down projection
/// takes `mixed_silu` (its `1/n` scale absorbed) and its up projection
/// `mixed_sigmoid`, nothing else — the forward's hand-wired arms
/// (`hc.rs` `mix_parts`), derived. The two fragments have no offline
/// verdict and are reported as such.
#[test]
fn the_derivation_reproduces_the_hand_wired_brackets() {
    let g = walk(&model());
    let d = derive(&g, &Policy::default()).expect("derive");
    eprintln!("plan: {}", d.plan);
    let silu = d.plan.sites_of("mixed_silu");
    let sigmoid = d.plan.sites_of("mixed_sigmoid");
    assert_eq!((silu.len(), sigmoid.len()), (97, 97));
    assert_eq!(d.plan.sites.len(), 194);
    assert!(silu.iter().all(|s| s.ends_with("_down.weight")), "{silu:?}");
    assert!(
        sigmoid.iter().all(|s| s.ends_with("_up.weight")),
        "{sigmoid:?}"
    );
    assert!(silu.contains(&"output_hc_down.weight"));
    assert!(silu.contains(&"blk.47.hc_ffn_down.weight"));
    assert!(d.refused.is_empty(), "{:?}", d.refused);
    assert_eq!(d.plan.class, NumericClass::Rounding);
    // Every scale absorbed is the bracket's 1/n.
    for n in d.graph.nodes().iter().filter(|n| n.op == Op::Fused) {
        let want = if n.name == "mixed_silu" { 0.25 } else { 1.0 };
        assert_eq!(n.scalar, Some(want), "{}", n.name);
    }
    // The fused-in-stateful report: no bracket sits in a recurrence.
    assert!(d.plan.regions.fused_in_stateful.is_empty());
    assert_eq!(
        Policy::default().unvalidated_in(&g),
        vec![
            ("mixed_silu".to_string(), "no entry".to_string()),
            ("mixed_sigmoid".to_string(), "no entry".to_string())
        ]
    );
}

/// The plan reaches the model: `derive_fusion` sets every bracket from it
/// — all on under the default policy, the down projections off when
/// `mixed_silu` is excluded — and a model built `manual` has them all on.
#[test]
fn the_plan_is_set_on_the_brackets_it_decides() {
    let mut m = model();
    assert!(m.fusion_report().is_none());
    let brackets = |m: &Qwen4Exp<GhostExperts>| m.bracket_epilogues();
    assert!(brackets(&m).iter().all(|&e| e == (true, true)));
    let line = m.derive_fusion().expect("derive");
    assert!(line.starts_with("qwen4exp fusion=derived: "), "{line}");
    assert!(line.contains("gemv_mixed_silu=97"), "{line}");
    assert!(line.contains("2 unvalidated"), "{line}");
    assert!(brackets(&m).iter().all(|&e| e == (true, true)));
    assert_eq!(brackets(&m).len(), 97);

    let policy = Policy {
        excluded: vec!["mixed_silu".to_string()],
        ..Default::default()
    };
    let line = m.derive_fusion_with(&policy).expect("derive");
    assert!(
        line.contains("mixed_silu: excluded by the exclusion set"),
        "{line}"
    );
    assert!(brackets(&m).iter().all(|&e| e == (false, true)));
}

/// The walk's recurrences: 36 DeltaNet states, 12 KV caches, the PLE's
/// conv history at its layer — and no fused site inside one.
#[test]
fn the_recurrences_are_marked() {
    let g = walk(&model());
    let d = derive(&g, &Policy::default()).expect("derive");
    let r = &d.plan.regions;
    assert_eq!((r.ssm, r.kv, r.conv_ring), (36, 12, 1), "{}", d.plan);
}
