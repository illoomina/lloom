//! The optimizer on the real LFM2 decode step (issue #732, P1), on the CPU:
//! the IR `lfm2_ir::decode_step` emits reconciles against a committed
//! dispatch trace of the real model, and the derivation over it produces
//! exactly the composition that was hand-written — the same 90 sites, the
//! same two epilogues, nothing more.
//!
//! No GPU, no model weights: the IR is emitted from the configuration and the
//! derivation is symbolic. The Metal half — that a `Fusion=Derived` model
//! computes the bits the flag-selected model computes — is
//! `lfm2_fusion_derived.rs`.
//!
//! Every count below is an ABSOLUTE reference (`CONTRIBUTING.md` §3.1b rule
//! 1), read from the artifact it cites rather than from this crate's own
//! output.

use lloom::models::lfm2::{AttnImpl, Config, ConvState, KvAppend, LayerType, QkvLayout};
use lloom::models::lfm2_ir::decode_step;
use lloom::optimizer::{derive, NumericClass, Op, Policy, Region, Seam};
use lloom_tensor::DType;

/// `LFM2.5-2.6B`'s `config.json`
/// (`models--LiquidAI--LFM2.5-2.6B/snapshots/654f9463…/config.json`): 30
/// layers, 22 conv and 8 attention in the listed order, hidden 2048,
/// intermediate 10752, 32 heads over 8 KV heads, `conv_L_cache` 3, tied
/// embeddings over a 128000 vocabulary.
fn lfm2_2_6b() -> Config {
    use LayerType::{Conv as C, FullAttention as A};
    Config {
        vocab_size: 128000,
        hidden_size: 2048,
        intermediate_size: 10752,
        num_hidden_layers: 30,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            C, C, A, C, C, A, C, C, C, A, C, C, C, A, C, C, C, A, C, C, C, A, C, C, A, C, C, A, C,
            C,
        ],
        tie_embedding: true,
        ..Default::default()
    }
}

/// The arms of the committed trace
/// `design/measurements/issue-727-raw/pass-plan/runs/trace-arena-shuffle.txt`:
/// `Shuffle × Sdpa × InPlace × Contiguous`, no fusion.
fn traced_arms(cfg: &mut Config, conv_state: ConvState) {
    cfg.attn_impl = AttnImpl::Sdpa;
    cfg.kv_append = KvAppend::InPlace;
    cfg.conv_state = conv_state;
    cfg.qkv_layout = QkvLayout::Contiguous;
    cfg.gemv_silu = false;
    cfg.gemv_residual = false;
    cfg.conv_fused = false;
}

fn census_of(cfg: &Config) -> (u64, Vec<(String, u64)>) {
    let g = decode_step(cfg, DType::F16);
    let census: Vec<(String, u64)> = g.census().into_iter().collect();
    (g.dispatches(), census)
}

/// The IR's dispatch census on the Shuffle arm equals the trace's, family by
/// family: `== kernels per decode token (step 1) ==` of
/// `trace-arena-shuffle.txt`, 538 dispatches. The trace names the GEMV by its
/// tile (`gemv_float16_bm4…` 84 + `bm8…` 83), the norm by dtype, the cast and
/// the embedding gather by their kernels; the IR names families.
#[test]
fn the_shuffle_ir_reconciles_with_the_committed_trace() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::Shuffle);
    let (total, census) = census_of(&cfg);
    let expected: Vec<(&str, u64)> = vec![
        ("badd", 60),
        ("bmul", 96),
        ("cast", 1),
        ("copy2d", 60),
        ("fast_sum", 22),
        ("gemv", 84 + 83),
        ("is_u32", 1),
        ("rmsnorm", 77),
        ("rope", 16),
        ("sdpa_vector", 8),
        ("silu", 30),
    ];
    let got: Vec<(&str, u64)> = census.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(got, expected, "the IR's census against the trace's");
    assert_eq!(total, 538, "`dispatches per decode token: 538`");
}

/// The rotating-ring arm: `trace-arena-rotating:16.txt`, 516 dispatches —
/// one `copy2d` per conv layer instead of two (`slice_set` into the ring),
/// everything else the Shuffle count.
#[test]
fn the_rotating_ring_ir_reconciles_with_the_committed_trace() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::RotatingRing { k: 16 });
    let (total, census) = census_of(&cfg);
    let copy2d = census.iter().find(|(k, _)| k == "copy2d").map(|(_, v)| *v);
    assert_eq!(copy2d, Some(38), "`38  copy2d_f16`");
    assert_eq!(total, 516, "`dispatches per decode token: 516`");
}

/// The elementwise family is 186 of the 538: 96 `bmul` + 60 `badd` + 30
/// `silu`. The issue quotes "186 of 652"; the 186 is this count, the 652 is
/// a different harness's total (§20 of `issue-727-pass-plans.md`, not a
/// committed trace) — see the PR's premise check.
#[test]
fn the_elementwise_family_is_186_of_538() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::Shuffle);
    let g = decode_step(&cfg, DType::F16);
    let elementwise = g.nodes().iter().filter(|n| n.op.is_elementwise()).count();
    assert_eq!(elementwise, 186);
    assert_eq!(g.dispatches(), 538);
}

/// **The P1 gate.** The derivation over the unfused step independently
/// arrives at the hand-written composition: SiLU onto every `w1` (30 sites,
/// `Config::gemv_silu`) and the residual onto every `w2`, `out_proj` and
/// `conv.out_proj` (30 + 8 + 22 = 60 sites, `Config::gemv_residual`) — and
/// onto nothing else. 538 dispatches become 448: the 30 `silu_f16` and 60
/// `badd_f16` the fused arms replace (`design/docs/fusion.md` §12.2i's
/// census), in the `Rounding` class (§12.2i: 2.50 and 2.25 f16 ulp, flat).
#[test]
fn the_derivation_reproduces_the_hand_written_composition() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::Shuffle);
    let g = decode_step(&cfg, DType::F16);
    let d = derive(&g, &Policy::default()).expect("derivation");
    let plan = &d.plan;
    eprintln!("plan: {plan}");
    eprintln!("egglog: {:.2} ms", d.elapsed.as_secs_f64() * 1e3);

    let mut silu = plan.sites_of("silu");
    silu.sort();
    assert_eq!(silu.len(), 30, "gemv_silu sites: {silu:?}");
    assert!(
        silu.iter().all(|s| s.ends_with(".feed_forward.w1.weight")),
        "{silu:?}"
    );

    let mut residual = plan.sites_of("residual");
    residual.sort();
    assert_eq!(residual.len(), 60, "gemv_residual sites: {residual:?}");
    let w2 = residual
        .iter()
        .filter(|s| s.ends_with(".feed_forward.w2.weight"))
        .count();
    let attn = residual
        .iter()
        .filter(|s| s.ends_with(".self_attn.out_proj.weight"))
        .count();
    let conv = residual
        .iter()
        .filter(|s| s.ends_with(".conv.out_proj.weight"))
        .count();
    assert_eq!((w2, attn, conv), (30, 8, 22));
    assert_eq!(
        plan.sites.len(),
        90,
        "nothing else is fused: {:?}",
        plan.sites
    );

    assert_eq!(plan.dispatches_before(), 538);
    assert_eq!(plan.dispatches_after(), 448);
    assert_eq!(plan.after.get("silu"), None);
    assert_eq!(plan.after.get("badd"), None);
    assert_eq!(plan.after.get("gemv_silu"), Some(&30));
    assert_eq!(plan.after.get("gemv_residual"), Some(&60));
    assert_eq!(plan.after.get("gemv"), Some(&(167 - 90)));
    assert_eq!(plan.class, NumericClass::Rounding);
    assert!(d.refused.is_empty(), "{:?}", d.refused);

    // The derived graph carries 90 fused nodes and no standalone silu/add.
    let fused = d.graph.nodes().iter().filter(|n| n.op == Op::Fused).count();
    assert_eq!(fused, 90);
    assert_eq!(
        d.graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Silu | Op::Add))
            .count(),
        0
    );
}

/// What the derivation could NOT fuse, and it is the number the design puts on
/// the "bounded by hand-writing" gap (`optimizer-design.md` §9a item 3): the
/// seams left in the derived step, each an adjacent single-use producer
/// feeding an elementwise dispatch the catalogue has no fragment for.
///
/// Per conv layer (22): `bmul(state, w)` reads the second `copy2d` (the new
/// state) — `copy2d>bmul`; `bmul(c, conv)` reads `fast_sum` — `fast_sum>bmul`.
/// Per MLP (30): `bmul(gate, up)` reads the fused `gemv_silu` and the `w3`
/// GEMV — `gemv_silu>bmul` and `gemv>bmul`. `bmul(b, x)` reads two views of
/// `in_proj`'s output, which has three readers: not a seam.
#[test]
fn the_seams_left_are_the_gap_the_catalogue_cannot_express() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::Shuffle);
    let g = decode_step(&cfg, DType::F16);
    let d = derive(&g, &Policy::default()).expect("derivation");
    let seam = |producer: &str, consumer: &str, count: u64| Seam {
        producer: producer.into(),
        consumer: consumer.into(),
        count,
    };
    assert_eq!(
        d.plan.seams,
        vec![
            seam("gemv", "bmul", 30),
            seam("gemv_silu", "bmul", 30),
            seam("copy2d", "bmul", 22),
            seam("fast_sum", "bmul", 22),
        ]
    );
}

/// The budget refuses what it cannot admit: at `BitExact` neither shipped
/// epilogue is derivable (both compute on the accumulator), the step is left
/// at 538, and the refusal names the class. This is the premise correction
/// the PR records — the issue calls the elementwise family's class
/// `BitExact`; the shipped fusions are `Rounding`.
#[test]
fn a_bit_exact_budget_derives_nothing_on_the_real_step() {
    let mut cfg = lfm2_2_6b();
    traced_arms(&mut cfg, ConvState::Shuffle);
    let g = decode_step(&cfg, DType::F16);
    let policy = Policy {
        budget: NumericClass::BitExact,
        ..Default::default()
    };
    let d = derive(&g, &policy).expect("derivation");
    assert!(d.plan.sites.is_empty());
    assert_eq!(d.plan.dispatches_after(), 538);
    assert_eq!(d.plan.class, NumericClass::BitExact);
    assert_eq!(d.refused.len(), 2);
    assert!(
        d.refused
            .iter()
            .all(|(_, why)| why.contains("class rounding exceeds the budget bit-exact")),
        "{:?}",
        d.refused
    );
}

/// The fused arms (`conv_fused`, `QkvLayout::Fused`) change what the block
/// dispatches but not what the derivation finds: the same 90 sites.
#[test]
fn the_other_arms_derive_the_same_sites() {
    for (conv_state, conv_fused, qkv) in [
        (ConvState::Shuffle, true, QkvLayout::Contiguous),
        (ConvState::RotatingRing { k: 10 }, true, QkvLayout::Fused),
        (
            ConvState::SlidingRing { k: 4, slack: 8 },
            false,
            QkvLayout::Strided,
        ),
    ] {
        let mut cfg = lfm2_2_6b();
        traced_arms(&mut cfg, conv_state);
        cfg.conv_fused = conv_fused;
        cfg.qkv_layout = qkv;
        let g = decode_step(&cfg, DType::F16);
        let d = derive(&g, &Policy::default()).expect("derivation");
        assert_eq!(
            d.plan.sites.len(),
            90,
            "{conv_state:?} fused={conv_fused} {qkv:?}"
        );
        assert_eq!(d.plan.sites_of("silu").len(), 30);
        assert_eq!(d.plan.sites_of("residual").len(), 60);
        assert_eq!(d.plan.dispatches_before() - d.plan.dispatches_after(), 90);
    }
}

/// **The stateful regions are marked** (`optimizer-design.md` §5a, P2): the
/// IR of the real step carries 22 conv-ring regions and 8 KV regions — one
/// per layer of each kind in `LFM2.5-2.6B`'s `layer_types` — and none of the
/// 90 fused sites sits in one: the epilogues ride `w1`, `w2` and the two
/// `out_proj`s, all straight-line, and the fold's result never re-enters the
/// window. Absolute references, per arm — the nodes of one conv layer's
/// region: unfused Shuffle = the state, its `narrow`, the two copies, the
/// window·weight product and the `fast_sum` (6); fused Shuffle = the state
/// and `shortconv_step` (2); unfused rotating = the state, the ring write,
/// the product, the sum (4); fused rotating = the state and `shortconv_ring`
/// (2); unfused sliding = the rotating count plus the live `narrow` (5);
/// fused sliding = the state, the write, the `narrow`, `conv1d_depthwise`
/// (4). A KV region is the two caches, the two appends and the attention (5).
#[test]
fn the_stateful_regions_are_marked_and_no_fused_site_sits_in_one() {
    for (conv_state, conv_fused, per_conv) in [
        (ConvState::Shuffle, false, 6usize),
        (ConvState::Shuffle, true, 2),
        (ConvState::RotatingRing { k: 16 }, false, 4),
        (ConvState::RotatingRing { k: 16 }, true, 2),
        (ConvState::SlidingRing { k: 4, slack: 8 }, false, 5),
        (ConvState::SlidingRing { k: 4, slack: 8 }, true, 4),
    ] {
        let mut cfg = lfm2_2_6b();
        traced_arms(&mut cfg, conv_state);
        cfg.conv_fused = conv_fused;
        let g = decode_step(&cfg, DType::F16);
        let regions = g.regions();
        let conv: Vec<_> = regions
            .iter()
            .filter(|(r, _)| matches!(r, Region::ConvRing(_)))
            .collect();
        let kv: Vec<_> = regions
            .iter()
            .filter(|(r, _)| matches!(r, Region::Kv(_)))
            .collect();
        assert_eq!(conv.len(), 22, "{conv_state:?} fused={conv_fused}");
        assert_eq!(kv.len(), 8, "{conv_state:?} fused={conv_fused}");
        for (r, nodes) in &conv {
            assert_eq!(
                nodes.len(),
                per_conv,
                "{r} under {conv_state:?} fused={conv_fused}"
            );
        }
        for (r, nodes) in &kv {
            assert_eq!(nodes.len(), 5, "{r}");
        }
        // The layers marked are the conv and attention layers, by index.
        let conv_layers: Vec<usize> = conv
            .iter()
            .map(|(r, _)| match r {
                Region::ConvRing(l) => *l,
                _ => unreachable!(),
            })
            .collect();
        let expected: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::Conv)
            .map(|(l, _)| l)
            .collect();
        assert_eq!(conv_layers, expected);

        let d = derive(&g, &Policy::default()).expect("derivation");
        assert_eq!(d.plan.sites.len(), 90);
        assert_eq!(d.plan.regions.conv_ring, 22);
        assert_eq!(d.plan.regions.kv, 8);
        assert_eq!(d.plan.regions.nodes, 22 * per_conv + 8 * 5);
        assert!(
            d.plan.regions.fused_in_stateful.is_empty(),
            "{conv_state:?} fused={conv_fused}: {:?}",
            d.plan.regions.fused_in_stateful
        );
        assert!(d.plan.to_string().contains("regions[conv-ring=22 kv=8"));
    }
}
