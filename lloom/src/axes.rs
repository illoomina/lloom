//! The within-crate axis registry (RESTRUCTURING_DESIGN.md §1.6).
//!
//! Before the mono-crate merge (§7.1) the four configuration surfaces — the
//! `Config` struct, the `SILU_IMPL` steer switch, the `config_line()` renderer
//! and `lloom-runs`' `axes::REGISTRY` — lived in four separate crates, so no one
//! `#[derive]` could reach them (N7). The merge collapsed the first three into
//! this crate, so a `linkme` distributed slice *within* the crate is now what a
//! derive can contribute to: every `#[derive(Config)]` emits one
//! `#[distributed_slice(AXES)]` static per `#[axis]` field, and the whole
//! engine's axis set is [`AXES`] at link time.
//!
//! The one surface that stays out is the external yardstick — `lloom-runs`'
//! `axes::REGISTRY` (§14.6, "the yardstick reads, never defines"). It is
//! reconciled against this slice by `lloom-runs/tests/registry_mirror.rs`, which
//! reads the [manifest](write_manifest) this module emits rather than depending
//! on this crate (`lloom-runs` is candle-free by construction).

use linkme::distributed_slice;

/// One configuration axis, contributed by a `#[axis]`-tagged field.
///
/// The three fields are exactly what `.bench/configurations.md` §1 argues in
/// prose (name, variant set, default) reduced to what a type can hold: the
/// evidence for the default stays in the markdown, the *names* live here, and
/// the mirror test binds the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisMeta {
    /// The registry name, e.g. `"AttnImpl"`. Must match a `REGISTRY` entry in
    /// `lloom-runs` or the mirror test's third leg is RED.
    pub name: &'static str,
    /// The arm names, e.g. `["Generic", "Sdpa", "FlashDecoding"]`.
    pub variants: &'static [&'static str],
    /// The default arm, e.g. `"Generic"`.
    pub default: &'static str,
}

/// The authoritative axis registry, collected at link time.
///
/// Every `#[derive(Config)]` contributes its `#[axis]` fields here. Reading it
/// is a total view of the engine's configuration surface — the thing the mirror
/// test reconciles against `lloom-runs`' `REGISTRY` and `.bench/configurations.md`
/// §1.
#[distributed_slice]
pub static AXES: [AxisMeta];

/// Every axis name in [`AXES`], sorted and deduplicated.
///
/// Link-time collection order is deterministic but unspecified across
/// declaration sites, so callers that compare against an ordered external list
/// sort first.
pub fn axis_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = AXES.iter().map(|a| a.name).collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// The manifest text the mirror test consumes: one axis name per line, sorted.
///
/// This is the bridge across the candle boundary (§14.6). `lloom-runs` must not
/// depend on this crate, so it cannot read [`AXES`] directly; instead this crate
/// writes the collected names to a committed file and `registry_mirror.rs` reads
/// that file. The file is regenerated and checked by
/// [`axes_manifest`](../../tests) so a drift between the slice and the committed
/// manifest is a test failure rather than a silent hole.
pub fn manifest() -> String {
    let mut out = String::new();
    for name in axis_names() {
        out.push_str(name);
        out.push('\n');
    }
    out
}
