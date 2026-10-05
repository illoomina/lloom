//! The one walker (`optimizer::walk`) against what it replaces: the IR a
//! model's blocks emit when walked, built through the model's own
//! constructor over ghost weights at the real checkpoint's geometry, diffed
//! against the per-model emitter's (`lfm2_ir::decode_step`) up to a
//! dependency-preserving permutation. Identical, or the first edge that
//! differs is named.
//!
//! On Metal: the attention kernel a layer takes is the route its
//! construction resolved on the device (`AttnRoute`), which is what a walk
//! reads; the emitter named it from the requested arm.

use lloom::models::lfm2::{AttnImpl, Config, ConvState, KvAppend, LayerType, QkvLayout};
use lloom::optimizer::{diff, walk, Graph};
use lloom::testing::ghost;
use lloom_tensor::{DType, Device};

/// `LFM2.5-2.6B`'s `config.json` (as `tests/optimizer_derivation.rs`).
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

/// The emitter's graph and the walk's, for one configuration.
fn both(cfg: &Config, dev: &Device) -> (Graph, Graph) {
    let emitted = lloom::models::lfm2_ir::decode_step(cfg, DType::F16);
    let model = ghost::lfm2(cfg, DType::F16, dev).expect("ghost build");
    (emitted, walk(&model))
}

/// Every arm the emitter describes, one axis at a time from the traced
/// configuration (`Shuffle × Sdpa × InPlace × Contiguous`), plus the fused
/// conv on every state arm.
fn arms() -> Vec<(String, Config)> {
    let mut out = Vec::new();
    let base = {
        let mut c = lfm2_2_6b();
        c.attn_impl = AttnImpl::Sdpa;
        c.kv_append = KvAppend::InPlace;
        c.conv_state = ConvState::Shuffle;
        c.qkv_layout = QkvLayout::Contiguous;
        c
    };
    out.push(("traced".to_string(), base.clone()));
    for (name, s) in [
        ("rotating", ConvState::RotatingRing { k: 16 }),
        ("sliding", ConvState::SlidingRing { k: 4, slack: 16 }),
        ("shuffle", ConvState::Shuffle),
    ] {
        for fused in [false, true] {
            let mut c = base.clone();
            c.conv_state = s;
            c.conv_fused = fused;
            out.push((format!("{name} fused={fused}"), c));
        }
    }
    for q in [QkvLayout::Strided, QkvLayout::Fused] {
        let mut c = base.clone();
        c.qkv_layout = q;
        out.push((format!("{q:?}"), c));
    }
    for k in [KvAppend::Cat, KvAppend::Paged] {
        let mut c = base.clone();
        c.kv_append = k;
        out.push((format!("{k:?}"), c));
    }
    for a in [AttnImpl::FlashDecoding, AttnImpl::Generic] {
        let mut c = base.clone();
        c.attn_impl = a;
        out.push((format!("{a:?}"), c));
    }
    // The fusion flags select arms the forward runs, not what it computes:
    // the walk is the unfused chain under any of them, as the emitter was.
    let mut c = base.clone();
    c.gemv_silu = true;
    c.gemv_residual = true;
    out.push(("flags on".to_string(), c));
    out
}

#[cfg(feature = "metal")]
#[test]
fn the_lfm2_walk_is_the_emitters_ir_on_every_arm() {
    let dev = Device::new_metal(0).expect("metal");
    let mut failures = Vec::new();
    for (name, cfg) in arms() {
        let (emitted, walked) = both(&cfg, &dev);
        let d = diff(&emitted, &walked);
        eprintln!(
            "WALK lfm2 {name}: emitter {} dispatches, walk {} — {d}",
            emitted.dispatches(),
            walked.dispatches()
        );
        if !d.is_empty() {
            failures.push(format!("{name}: {d}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Off Metal every attention arm resolves to the composed attention at
/// construction, and the walk says so where the emitter named the requested
/// kernel: the one family that moves, eight nodes, nothing else.
#[test]
fn off_metal_the_walk_reads_the_route_the_model_took() {
    let mut cfg = lfm2_2_6b();
    cfg.attn_impl = AttnImpl::Sdpa;
    let (emitted, walked) = both(&cfg, &Device::Cpu);
    let d = diff(&emitted, &walked);
    assert_eq!(walked.census().get("attention_generic"), Some(&8), "{d}");
    assert_eq!(walked.census().get("sdpa_vector"), None);
    assert_eq!(emitted.census().get("sdpa_vector"), Some(&8));
    let (l, r) = d.first();
    assert!(l.unwrap().describe.starts_with("sdpa_vector"), "{d}");
    assert!(r.unwrap().describe.starts_with("attention_generic"), "{d}");
    assert_eq!(walked.dispatches(), emitted.dispatches());
}

/// Qwen4-exp: the walk of the committed checkpoint's model (the loader's own
/// assembly over ghost weights) against the emitter it replaces, and both
/// derived. Printed for the record; the walk's own reconciliation with a
/// trace of today's step is `qwen4exp_walk.rs`.
#[test]
fn the_qwen4exp_walk_against_its_emitter() {
    use lloom::models::qwen4exp::{testing::ghost_model, Config};
    use lloom::optimizer::{derive, Policy};
    let model = ghost_model(&Device::Cpu).expect("ghost build");
    let walked = walk(&model);
    let emitted = lloom::models::qwen4exp_ir::decode_step(&Config::flash_next());
    let d = diff(&emitted, &walked);
    eprintln!(
        "WALK qwen4exp source: emitter {} dispatches, walk {} — {d}",
        emitted.dispatches(),
        walked.dispatches()
    );
    for u in d.only_left.iter().take(40) {
        eprintln!(
            "WALK qwen4exp only-emitter L{} x{} {}",
            u.level, u.count, u.describe
        );
    }
    for u in d.only_right.iter().take(40) {
        eprintln!(
            "WALK qwen4exp only-walk L{} x{} {}",
            u.level, u.count, u.describe
        );
    }
    let derived = derive(&walked, &Policy::default()).expect("derive").graph;
    eprintln!(
        "WALK qwen4exp derived: {} dispatches, emitter census {:?}",
        derived.dispatches(),
        emitted.census()
    );
    eprintln!("WALK qwen4exp derived census {:?}", derived.census());
    eprintln!("WALK qwen4exp source census {:?}", walked.census());
}

/// The IR-diff gate (`profiles-and-memory-design.md` §11 item 1): today's
/// walk of a family's real checkpoint against its committed listing
/// (`tests/walk_listings/`), compared up to a dependency-preserving
/// permutation. A change to a forward that changes what its step
/// dispatches changes the listing — through the `visit` beside the changed
/// forward — and fails here, naming the first edge, until the listing is
/// rewritten (`LLOOM_BLESS_LISTINGS=1`) and the rewrite reviewed.
fn listing_gate(name: &str, g: &Graph) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/walk_listings")
        .join(format!("{name}.txt"));
    let now = lloom::optimizer::listing(g);
    if std::env::var_os("LLOOM_BLESS_LISTINGS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &now).unwrap();
        eprintln!("WALK blessed {}", path.display());
        return;
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
            "{name}: the walk differs from its committed listing at line {first}:\n  committed: {}\n  walk:      {}",
            committed.lines().nth(first).unwrap_or("<end>"),
            now.lines().nth(first).unwrap_or("<end>")
        );
    }
}

#[test]
fn the_qwen4exp_walk_matches_its_committed_listing() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    listing_gate("qwen4exp-iq3xxs", &walk(&model));
}

#[cfg(feature = "metal")]
#[test]
fn the_lfm2_walk_matches_its_committed_listing() {
    let dev = Device::new_metal(0).expect("metal");
    let (_, cfg) = arms().into_iter().next().expect("the traced arm");
    let model = ghost::lfm2(&cfg, DType::F16, &dev).expect("ghost build");
    listing_gate("lfm2-2.6b-traced", &walk(&model));
}
