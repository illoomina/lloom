//! The named LFM2 configurations the engine serves, as one definition.
//!
//! A preset is a set of `Config` axes applied over the checkpoint's config;
//! per-axis overrides are applied over the preset. The library's `Config`
//! ships the conservative arms (§7.1a) and a serving binary turns the
//! measured ones on here. `lloom chat|serve` resolve their `--cache` through
//! these functions and the LFM2 harnesses take `--preset` from them, so a
//! gate run on a preset decodes the arms the engine serves rather than a flag
//! list kept beside them (§2.3.8c).

use crate::models::lfm2::{AttnImpl, Config, ConvState, KvAppend, QkvLayout};
use crate::models::lfm2_dspark::DSparkConfig;
use lloom_tensor::DType;

/// The single-sequence serving configuration (`--cache crown`): fused Q/K/V
/// views, the sdpa read over an in-place KV, the shuffled conv state decoded
/// by the fused step (`shortconv_step`), and both GEMV epilogues. Every arm is
/// token-identical to the generic path (§13.5, §12.2d, §12.2h).
pub fn crown(config: &mut Config) {
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::Sdpa;
    config.kv_append = KvAppend::InPlace;
    config.conv_state = ConvState::Shuffle;
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
}

/// The rows' shared-pool configuration (`--rows pool`): the paged pool the
/// slot-indexed kernels read through FlashDecoding, and the conv state as a
/// one-slot rotating ring decoded by the fused ring step (`shortconv_ring`)
/// (`batched-decode-design.md` §12.2).
pub fn pool(config: &mut Config) {
    config.qkv_layout = QkvLayout::Fused;
    config.attn_impl = AttnImpl::FlashDecoding;
    config.kv_append = KvAppend::Paged;
    config.conv_state = ConvState::RotatingRing { k: 1 };
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
}

/// The configuration the speculative body needs: the cache rewinds only under
/// a rotating ring wide enough for a `k`-token verify window (the shuffle is
/// destructive) and only over an in-place KV (`Cat` has no length to rewind);
/// `taps` feed the drafter's projection at f32 (§10.2m).
pub fn rewindable(config: &mut Config, k: usize, taps: Vec<usize>) {
    crown(config);
    config.conv_state = ConvState::RotatingRing { k: k + 1 };
    config.tap_dtype = DType::F32;
    config.tap_layers = taps;
}

/// The pool under a drafter (§12.4): `pool` with a ring wide enough for the
/// drafter's verify window, and its taps.
pub fn pool_spec(config: &mut Config, dcfg: &DSparkConfig) {
    pool(config);
    config.conv_state = ConvState::RotatingRing {
        k: dcfg.max_draft() + 1,
    };
    config.tap_dtype = DType::F32;
    config.tap_layers = dcfg.tap_layers();
}

/// The axes, by flag, that every preset above sets whatever config it is
/// applied over. A front-end counts a named preset as explicit for these,
/// as it counts a named key: its own defaults leave them to the preset.
pub const AXES: &[&str] = &[
    "qkv-layout",
    "attn",
    "kv-append",
    "conv-state",
    "gemv-silu",
    "gemv-residual",
    "conv-fused",
];

/// The presets a harness can name without a drafter, in `--preset` spelling.
pub const NAMED: &[&str] = &["crown", "pool"];

/// The preset `name` spells, if it is one of [`NAMED`].
pub fn by_name(name: &str) -> Option<fn(&mut Config)> {
    match name {
        "crown" => Some(crown),
        "pool" => Some(pool),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rewindable preset's ring holds the verify window, and its taps are
    /// whatever the caller names (none on `rewindable-notap`).
    #[test]
    fn the_rewindable_preset_sizes_the_ring_for_k() {
        let mut c = Config::default();
        rewindable(&mut c, 9, vec![3, 10]);
        assert_eq!(c.conv_state, ConvState::RotatingRing { k: 10 });
        assert_eq!(
            (c.attn_impl, c.kv_append),
            (AttnImpl::Sdpa, KvAppend::InPlace)
        );
        assert_eq!(
            (c.tap_dtype, c.tap_layers.clone()),
            (DType::F32, vec![3, 10])
        );
        let mut c = Config::default();
        rewindable(&mut c, 3, Vec::new());
        assert_eq!(c.conv_state, ConvState::RotatingRing { k: 4 });
        assert!(c.tap_layers.is_empty());
    }

    /// `AXES` is exactly what each preset sets: applied over configs that
    /// differ in one axis alone, a preset that sets the axis gives one
    /// config, and a preset that leaves it gives as many as there were.
    #[test]
    fn every_preset_sets_exactly_its_axes() {
        type Preset = Box<dyn Fn(&mut Config)>;
        let dcfg = DSparkConfig::lfm2_2_6b();
        let presets: [(&str, Preset); 5] = [
            ("crown", Box::new(crown)),
            ("pool", Box::new(pool)),
            (
                "rewindable",
                Box::new(|c: &mut Config| rewindable(c, 9, vec![3, 10])),
            ),
            (
                "rewindable-notap",
                Box::new(|c: &mut Config| rewindable(c, 3, Vec::new())),
            ),
            (
                "pool-spec",
                Box::new(move |c: &mut Config| pool_spec(c, &dcfg)),
            ),
        ];
        let distinct = |cs: &[Config]| {
            let mut lines: Vec<String> = cs.iter().map(Config::axes_line).collect();
            lines.sort();
            lines.dedup();
            lines.len()
        };
        let mut probed = 0;
        for (name, preset) in &presets {
            let mut set = Vec::new();
            let mut varied = 0;
            for axis in Config::AXES_META {
                let starts: Vec<Config> = std::iter::once(Config::default())
                    .chain(
                        // An arm with parameters (`rotating[:<K>]`) by its
                        // bare name.
                        axis.variants
                            .iter()
                            .map(|v| v.split('[').next().unwrap_or(v))
                            .filter(|v| !v.contains('<'))
                            .filter_map(|v| {
                                let mut c = Config::default();
                                c.set_axis(axis.key, v).ok().map(|_| c)
                            }),
                    )
                    .collect();
                let before = distinct(&starts);
                if before < 2 {
                    assert!(
                        !AXES.contains(&axis.key),
                        "{} is in AXES and the probe cannot vary it",
                        axis.key
                    );
                    continue;
                }
                varied += 1;
                let after: Vec<Config> = starts
                    .into_iter()
                    .map(|mut c| {
                        preset(&mut c);
                        c
                    })
                    .collect();
                match distinct(&after) {
                    1 => set.push(axis.key),
                    n => assert_eq!(n, before, "{name} sets {} for some arms only", axis.key),
                }
            }
            set.sort();
            eprintln!("[{name}: sets {set:?}, of the {varied} axes the probe varied]");
            probed += varied;
            let mut want = AXES.to_vec();
            want.sort();
            assert_eq!(set, want, "the axes {name} sets");
        }
        assert!(probed >= 5 * AXES.len(), "{probed} axes probed");
    }

    /// Every named preset resolves, and to the function its name says: the
    /// crown decodes the shuffled state on the fused step, the pool the
    /// one-slot ring on the fused ring step.
    #[test]
    fn the_named_presets_resolve_to_their_conv_arms() {
        for name in NAMED {
            assert!(by_name(name).is_some(), "{name} is NAMED and unresolved");
        }
        assert!(by_name("rewindable").is_none());
        let mut c = Config::default();
        by_name("crown").unwrap()(&mut c);
        assert_eq!((c.conv_fused, c.conv_state), (true, ConvState::Shuffle));
        let mut c = Config::default();
        by_name("pool").unwrap()(&mut c);
        assert_eq!(
            (c.conv_fused, c.conv_state),
            (true, ConvState::RotatingRing { k: 1 })
        );
    }
}
