//! The one walker (`optimizer::walk`): the IR a model's blocks emit when
//! walked, built through the model's own constructor over ghost weights at
//! the real checkpoint's geometry (`testing::ghost`), held to the listing
//! committed for it — the IR-diff gate (`profiles-and-memory-design.md` §11
//! item 1). The walk replaced the per-model emitters after diffing equal to
//! LFM2's on every arm it described and reconciling with a fresh trace of
//! Qwen4-exp's step (`design/measurements/issue-732-walker.md`).
//!
//! On Metal the attention kernel a layer takes is the route its
//! construction resolved on the device, which is what a walk reads.

use lloom::models::lfm2::{AttnImpl, Config, ConvState, KvAppend, LayerType, QkvLayout};
use lloom::optimizer::{walk, Graph};
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

/// The configuration of the committed trace (`issue-727-raw/pass-plan/
/// runs/trace-arena-shuffle.txt`): `Shuffle × Sdpa × InPlace × Contiguous`.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn traced() -> Config {
    let mut c = lfm2_2_6b();
    c.attn_impl = AttnImpl::Sdpa;
    c.kv_append = KvAppend::InPlace;
    c.conv_state = ConvState::Shuffle;
    c.qkv_layout = QkvLayout::Contiguous;
    c
}

/// Off Metal every attention arm resolves to the composed attention at
/// construction, and the walk says so: the eight attention layers take
/// `attention_generic`, whatever was requested — the route the model took,
/// not the arm the config named.
#[test]
fn off_metal_the_walk_reads_the_route_the_model_took() {
    let mut cfg = lfm2_2_6b();
    cfg.attn_impl = AttnImpl::Sdpa;
    let walked = walk(&ghost::lfm2(&cfg, DType::F16, &Device::Cpu).expect("ghost build"));
    assert_eq!(walked.census().get("attention_generic"), Some(&8));
    assert_eq!(walked.census().get("sdpa_vector"), None);
    assert_eq!(walked.dispatches(), 538);
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
    let model = ghost::lfm2(&traced(), DType::F16, &dev).expect("ghost build");
    listing_gate("lfm2-2.6b-traced", &walk(&model));
}
