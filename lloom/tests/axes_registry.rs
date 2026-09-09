//! The within-crate axis registry (`lloom::axes::AXES`) and its manifest
//! (RESTRUCTURING_DESIGN.md §1.6).
//!
//! These tests live in `lloom` because `lloom-runs` is candle-free and cannot
//! read `AXES` directly (§14.6). This crate emits the committed manifest and
//! checks the slice against it; `lloom-runs/tests/registry_mirror.rs` then reads
//! that manifest and reconciles it against `REGISTRY` — the third leg.

use std::path::PathBuf;

/// The `#[axis]` fields on `Config` that the derive contributes, in the metal
/// build (the gate build — `SiluImpl` is metal-only). Off-metal, `SiluImpl` is
/// absent by construction (the field is `#[cfg(feature = "metal")]`).
///
/// This is an ABSOLUTE reference (`CONTRIBUTING.md` §3.1b rule 1): a shape check
/// like "AXES is non-empty" is blind to a field losing its `#[axis]` tag. The
/// list names exactly which axes the derive must produce, so dropping a tag
/// fails with the missing name rather than a silent shrink.
const EXPECTED_AXES: &[&str] = &[
    "AttnImpl",
    "ConvFused",
    "ConvPrefillCtx",
    "ConvState",
    "GemvResidual",
    "GemvSilu",
    "KvAppend",
    "KvQuant",
    "QkvLayout",
    "ScratchSizing",
    #[cfg(feature = "metal")]
    "SiluImpl",
];

#[test]
fn axes_slice_matches_the_expected_config_axes() {
    let got = lloom::axes::axis_names();
    let mut expected: Vec<&str> = EXPECTED_AXES.to_vec();
    expected.sort_unstable();
    assert_eq!(
        got, expected,
        "lloom::axes::AXES drifted from the expected Config #[axis] set.\n\
         got:      {got:?}\n\
         expected: {expected:?}\n\
         A field losing its #[axis] tag, or a new tagged field, shows here."
    );
}

/// The acceptance gate (issue #646, RESTRUCTURING §10.1 Wave A): `--attn` and
/// `--kv-quant` exist as clap flags emitted by the derive, and applying them
/// mutates the config.
///
/// `--draft` is deliberately NOT here: it is a speculative-window count owned by
/// the drafter surface (Wave C, §3.4/PR 5), not a `Config` axis, so it cannot
/// fall out of a Wave-A `Config` derive. See the PR's findings.
#[test]
fn derived_flags_exist_and_apply() {
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        axes: lloom::models::lfm2::ConfigArgs,
    }

    // The flags parse (they exist), and unknown values are rejected by the
    // parse route rather than silently ignored (§2.4).
    let h = Harness::try_parse_from(["t", "--attn", "sdpa", "--kv-quant", "int8"])
        .expect("--attn and --kv-quant must be derived clap flags");

    let mut cfg = lloom::models::lfm2::Config::default();
    h.axes
        .apply(&mut cfg)
        .expect("apply must accept valid arms");

    assert_eq!(cfg.attn_impl, lloom::models::lfm2::AttnImpl::Sdpa);
    assert_eq!(cfg.kv_quant, lloom::nn::ops::KvQuant::Int8);

    // A bad arm is a parse error, not a silent default (the failure §2.4 names).
    let mut bad = lloom::models::lfm2::Config::default();
    let h2 = Harness::try_parse_from(["t", "--attn", "nonsense"]).unwrap();
    assert!(
        h2.axes.apply(&mut bad).is_err(),
        "a bad --attn arm must error"
    );
}

/// A flag absent from the command line leaves the base config's value — the
/// `Option<String>` "unset means keep" contract the derive relies on.
#[test]
fn absent_flags_keep_the_base_value() {
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        axes: lloom::models::lfm2::ConfigArgs,
    }

    let h = Harness::try_parse_from(["t"]).unwrap();
    let mut cfg = lloom::models::lfm2::Config::default();
    cfg.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa; // a non-default base
    h.axes.apply(&mut cfg).unwrap();
    assert_eq!(
        cfg.attn_impl,
        lloom::models::lfm2::AttnImpl::Sdpa,
        "an unset flag must not overwrite the base config"
    );
}

#[test]
fn every_axis_has_a_default_that_is_one_of_its_variants() {
    for axis in lloom::axes::AXES {
        assert!(
            axis.variants.contains(&axis.default),
            "axis {:?} default {:?} is not in its variants {:?}",
            axis.name,
            axis.default,
            axis.variants
        );
    }
}

/// Regenerate/check the manifest `registry_mirror.rs`'s third leg reads.
///
/// The manifest is the bridge across the candle boundary: `lloom-runs` cannot
/// depend on this crate, so the collected `AXES` names are written to a
/// committed file and reconciled against `REGISTRY` there. This test keeps the
/// committed file in sync with the slice — a drift is a test failure, not a
/// silent hole.
///
/// `LLOOM_WRITE_AXES_MANIFEST=1` writes the file; the default asserts it matches
/// so CI catches a stale manifest.
#[test]
fn axes_manifest_is_in_sync() {
    let manifest = lloom::axes::manifest();
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../design/.bench/axes-manifest.txt");

    if std::env::var_os("LLOOM_WRITE_AXES_MANIFEST").is_some() {
        std::fs::write(&path, &manifest)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        return;
    }

    let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}). Regenerate with LLOOM_WRITE_AXES_MANIFEST=1.",
            path.display()
        )
    });

    // The metal build is the authority: it carries every axis including the
    // metal-only `SiluImpl`. Off-metal the slice is a subset, so only assert the
    // full match under metal; off-metal assert the committed manifest is a
    // superset of what this build sees.
    #[cfg(feature = "metal")]
    assert_eq!(
        manifest, committed,
        "axes-manifest.txt is stale. Regenerate with LLOOM_WRITE_AXES_MANIFEST=1 \
         under --features metal (the authoritative build)."
    );
    #[cfg(not(feature = "metal"))]
    {
        let committed_names: std::collections::BTreeSet<&str> = committed.lines().collect();
        for name in manifest.lines() {
            assert!(
                committed_names.contains(name),
                "axis {name:?} is in the off-metal slice but not the committed \
                 manifest; regenerate under --features metal."
            );
        }
    }
}
