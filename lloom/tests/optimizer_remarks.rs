//! The optimization remarks (`optimizer::remarks`,
//! `design/docs/optimizer-design.md` §10i) over the two families' real
//! decode steps: every seam the derived step leaves, why, and its census
//! cost, ranked — held to the tables committed for them
//! (`tests/remarks/`), beside the derived step's listing the remarks were
//! read from, so the edge enumeration can be re-derived from the listing by
//! a second reader (`design/measurements/issue-732-remarks.md`).
//!
//! Every reason asserted below is the gate the seam fails in
//! `design/docs/fusion.md` §12.2e's terms; every count is the committed
//! walk's.

use lloom::optimizer::{
    derive, listing, remarks, walk_at, CostModel, Graph, Policy, Remarks, Side, Why,
};
use lloom_tensor::Device;

/// `name`'s committed table: the derived step's listing, then its remarks.
/// Rewritten under `LLOOM_BLESS_LISTINGS=1`, as the walk listings are.
fn remarks_gate(name: &str, derived: &Graph, r: &Remarks) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/remarks");
    for (suffix, now) in [
        ("derived.txt", listing(derived)),
        ("remarks.txt", r.to_string()),
    ] {
        let path = dir.join(format!("{name}.{suffix}"));
        if std::env::var_os("LLOOM_BLESS_LISTINGS").is_some() {
            std::fs::create_dir_all(&dir).expect("remarks dir");
            std::fs::write(&path, &now).expect("bless");
            continue;
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (bless with LLOOM_BLESS_LISTINGS=1)",
                path.display()
            )
        });
        if committed != now {
            let first = committed
                .lines()
                .zip(now.lines())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| committed.lines().count().min(now.lines().count()));
            panic!(
                "{}: differs from the committed table at line {}:\n  committed: {}\n  now:       {}",
                path.display(),
                first + 1,
                committed.lines().nth(first).unwrap_or("<end>"),
                now.lines().nth(first).unwrap_or("<end>")
            );
        }
    }
}

/// The rows of the class `producer > consumer`.
fn class<'a>(r: &'a Remarks, producer: &str, consumer: &str) -> Vec<(&'a Why, u64)> {
    r.rows
        .iter()
        .filter(|x| x.producer == producer && x.consumer == consumer)
        .map(|x| (&x.why, x.edges))
        .collect()
}

fn no_undeclared(r: &Remarks) {
    let undeclared: Vec<&Why> = r
        .rows
        .iter()
        .map(|x| &x.why)
        .filter(|w| matches!(w, Why::Undeclared(_)))
        .collect();
    assert!(
        undeclared.is_empty(),
        "kernels the remarks cannot classify: {undeclared:?}"
    );
}

/// Qwen4-exp's steady decode step (the walk `tests/qwen4exp_walk.rs`
/// reconciles with the trace), derived under the default policy.
#[test]
fn the_qwen4exp_step_remarks() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    let g = lloom::optimizer::walk(&model);
    let d = derive(&g, &Policy::default()).expect("derive");
    let r = remarks(&d, &CostModel::default());
    eprintln!("{r}");
    no_undeclared(&r);
    assert_eq!((r.dispatches, r.fused), (1579, 194));
    // The seams with an end in a recurrence: each of the 36 DeltaNet layers'
    // four (its two products into the prologue and the step, the prologue
    // into the step, the step into its gated norm), each of the 12 QSA
    // layers' seven (the prologue into the append and the attention, the two
    // stems' appends, the appends into the attention, the attention into its
    // gate), and the PLE's conv history's seven.
    assert_eq!(
        r.rows.iter().map(|x| x.stateful).sum::<u64>(),
        36 * 4 + 12 * 7 + 7
    );
    // The routed and shared SwiGLU products ride no store: the quantized
    // matvec takes no block (step (c)'s hooks).
    assert_eq!(
        class(&r, "mul_mv", "silu_mul"),
        [(&Why::NewBody("no store hook on mul_mv".to_string()), 96)]
    );
    // The bracket's grouped norm feeds the projections' load. The
    // catalogue's grouped-norm prologue computes it there at any tile (its
    // per-group sums in the threadgroup before the K-walk) and is refused by
    // name: its third reader, the collapse, is not a product.
    let gnorm = Why::Refused {
        fragment: "mixed_gnorm".to_string(),
        reason: lloom::optimizer::fragments::MIXED_GNORM_REFUSAL.to_string(),
    };
    assert_eq!(class(&r, "hc_norm", "gemv_mixed"), [(&gnorm, 96)]);
    assert_eq!(class(&r, "hc_norm", "gemv_mixed_silu"), [(&gnorm, 97)]);
    // The up projection's sigmoid store feeds the gated mean over streams:
    // an output element reads n stores a stream apart.
    assert_eq!(
        class(&r, "gemv_mixed_sigmoid", "hc_collapse"),
        [(&Why::CrossThreadgroup(Side::Store), 97)]
    );
}

/// LFM2-2.6B's decode step at the committed trace's arms (`Shuffle × Sdpa ×
/// InPlace × Contiguous`), derived under the default policy, and the crown
/// preset's — the arms that collect seams by hand.
#[cfg(feature = "metal")]
#[test]
fn the_lfm2_step_remarks() {
    use lloom::models::lfm2::{AttnImpl, ConvState, KvAppend, QkvLayout};
    use lloom::testing::ghost::{lfm2, lfm2_2_6b};
    use lloom_tensor::DType;
    let dev = Device::new_metal(0).expect("metal");
    let mut cfg = lfm2_2_6b();
    cfg.attn_impl = AttnImpl::Sdpa;
    cfg.kv_append = KvAppend::InPlace;
    cfg.conv_state = ConvState::Shuffle;
    cfg.qkv_layout = QkvLayout::Contiguous;
    let model = lfm2(&cfg, DType::F16, &dev).expect("ghost build");
    let g = lloom::optimizer::walk(&model);
    let d = derive(&g, &Policy::default()).expect("derive");
    let r = remarks(&d, &CostModel::default());
    eprintln!("{r}");
    no_undeclared(&r);
    remarks_gate("lfm2-2.6b-traced", &d.graph, &r);
    // Past the 2-pass length (1 024 keys at 64 × f16 by default) the
    // attention is two launches with a seam between them.
    let g = walk_at(&model, 1600);
    let d2 = derive(&g, &Policy::default()).expect("derive");
    let r2 = remarks(&d2, &CostModel::default());
    eprintln!("two-pass: {r2}");
    no_undeclared(&r2);
    remarks_gate("lfm2-2.6b-traced-two-pass", &d2.graph, &r2);
    assert_eq!(
        r2.dispatches,
        r.dispatches + 8,
        "one more launch a layer, eight layers"
    );
    // 388 before the in-place shift's strided gather (one a conv layer,
    // `docs/dispatch.md` §11.5z-x: 555 → 577 on the sdpa arm, all 22 of it
    // `copy_f16_strided`).
    assert_eq!((r.dispatches, r.fused), (410, 90));
    // The unfused conv chain: the window's gather into the gate product.
    assert_eq!(
        class(&r, "copy2d", "bmul"),
        [(&Why::NewBody("movement to elementwise".to_string()), 22)]
    );
    // The operator / FFN / final norms into their products' load: every
    // product's tile walks K in one simdgroup, so the inline norm applies —
    // the three attention projections, the conv's in-projection, the
    // two-weight kernel, the head — and the default budget refuses it by
    // class (its sum of squares takes another tree).
    let refused_norm = Why::Refused {
        fragment: "norm".to_string(),
        reason: "class reorder exceeds the budget rounding".to_string(),
    };
    let norm_into_gemv: u64 = r
        .rows
        .iter()
        .filter(|x| x.producer == "rmsnorm" && x.why == refused_norm)
        .map(|x| x.edges)
        .sum();
    assert_eq!(norm_into_gemv, 8 * 3 + 22 + 1 + 30, "{r}");
    assert!(
        !r.rows.iter().any(|x| x.producer == "rmsnorm" && x.why == Why::CrossThreadgroup(Side::Load)),
        "{r}"
    );

    let mut crown = lfm2_2_6b();
    lloom::models::lfm2_presets::crown(&mut crown);
    let g = lloom::optimizer::walk(&lfm2(&crown, DType::F16, &dev).expect("ghost build"));
    let d = derive(&g, &Policy::default()).expect("derive");
    let rc = remarks(&d, &CostModel::default());
    eprintln!("{rc}");
    no_undeclared(&rc);
    remarks_gate("lfm2-2.6b-crown", &d.graph, &rc);
    // The fused conv step collects the chain by hand: no copy2d>bmul left.
    assert!(class(&rc, "copy2d", "bmul").is_empty(), "{rc}");
}

/// The Qwen4-exp tables, committed beside LFM2's.
#[test]
fn the_qwen4exp_step_remarks_match_the_committed_table() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    let g = lloom::optimizer::walk(&model);
    let d = derive(&g, &Policy::default()).expect("derive");
    remarks_gate(
        "qwen4exp-iq3xxs",
        &d.graph,
        &remarks(&d, &CostModel::default()),
    );
}

/// The steps only some positions take, at the real geometry: one whose key
/// completes an indexer block (inside the dense range), one past the dense
/// range (the selection, the gather, and — the selection reaching the
/// 2-pass length — the two-launch attention). Each derived and remarked as
/// the steady step is, committed beside it.
#[test]
fn the_qwen4exp_position_remarks() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    let dense = model
        .dense_limit()
        .expect("the checkpoint's QSA layers carry the indexer");
    assert_eq!(dense, 2051);
    for (name, pos) in [("block", 47), ("past-dense", 4001)] {
        let g = walk_at(&model, pos);
        let d = derive(&g, &Policy::default()).expect("derive");
        let r = remarks(&d, &CostModel::default());
        eprintln!("{name} (position {pos}): {r}");
        no_undeclared(&r);
        remarks_gate(&format!("qwen4exp-iq3xxs-{name}"), &d.graph, &r);
    }
}

/// The MTP head's one-row pass at the sidecar's geometry
/// (`testing::ghost_model_with_mtp`), derived and remarked.
#[test]
fn the_qwen4exp_mtp_head_remarks() {
    let model = lloom::models::qwen4exp::testing::ghost_model_with_mtp(&Device::Cpu)
        .expect("ghost build with the head");
    let head = model.mtp_head().expect("the head was asked for");
    let g = lloom::optimizer::walk(head);
    let d = derive(&g, &Policy::default()).expect("derive");
    let r = remarks(&d, &CostModel::default());
    eprintln!("mtp head: {r}");
    eprintln!("plan: {}", d.plan);
    no_undeclared(&r);
    remarks_gate("qwen4exp-iq3xxs-mtp-head", &d.graph, &r);
}
