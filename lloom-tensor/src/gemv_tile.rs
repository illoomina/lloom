//! The GEMV's tile as the Metal dispatcher selects it, as plain arithmetic
//! on the two vector lengths — so the dispatcher and a reader with no device
//! (the optimizer's legality facts, `lloom::optimizer`) read one definition.
//! No Metal type is named here; the dispatcher re-exports what it calls.

/// Which threads hold the whole `K` of one output element when the tile's
/// dot product is done: the fact a reduction over `K` riding the GEMV's load
/// needs (`docs/fusion.md` §12.2s). A GEMV partitions its OUTPUT across
/// threadgroups, so no tile ever holds `K` for a value another threadgroup
/// stores; inside one threadgroup the `K`-walk is split or it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KWalk {
    /// One simdgroup walks all of `K` (`bn == 1`, `sn == 32`): a sum over the
    /// loaded coefficients is complete in its registers after one shuffle
    /// tree — the inline norm's tile.
    Simdgroup,
    /// The threadgroup's simdgroups split `K` (`bn > 1`) and their partials
    /// meet in threadgroup memory: a sum over `K` needs that reduction too.
    Threadgroup,
}

/// The non-transposed GEMV's whole tile as the dispatcher selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemvTile {
    pub bm: usize,
    pub bn: usize,
    pub sm: usize,
    pub sn: usize,
    pub tm: usize,
    pub tn: usize,
}

impl GemvTile {
    /// Who holds `K` per output element at this tile.
    pub fn k_walk(&self) -> KWalk {
        if self.bn == 1 && self.sn == 32 {
            KWalk::Simdgroup
        } else {
            KWalk::Threadgroup
        }
    }
}

/// The tile the dispatcher selects for a non-transposed `[out_vec_size,
/// in_vec_size]` matrix ([`gemv_tile_dims`], #492's override included).
pub fn gemv_tile(out_vec_size: usize, in_vec_size: usize) -> GemvTile {
    let (bm, bn, sm, sn, tm, tn) = gemv_tile_dims(out_vec_size, in_vec_size);
    GemvTile {
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
    }
}

/// Issue #492's GEMV-tile arm: pin the `bm` the GEMV path derives from
/// `out_vec_size`, so **N and the tile stop moving together**.
///
/// **Unset is what ships** and selects byte-for-byte what shipped: the override
/// is read only where the derived `bm` would otherwise be used, and an unset
/// variable returns `None` before any branch changes.
///
/// It exists because `select_tile_config` is **never consulted on this path**
/// (`call_mlx_gemm` returns into `call_mlx_gemv` before reaching it), so
/// `LLOOM_383_TILE` — which forces the GEMM tile — cannot reach a GEMV. The
/// GEMV's tile is derived inline from `out_vec_size`:
///
/// ```text
/// out_vec_size >= 4096  ->  bm8      (256 threads/tgp, 32 outputs/tgp)
/// else                  ->  bm4      (128 threads/tgp, 16 outputs/tgp)
/// ```
///
/// **That boundary sits inside #488's sweep**, between its N=2048 and N=6144
/// points — which is exactly where its effect curve bends. So the two steep
/// small-N points are the only `bm4` points in that sweep, and *"the effect
/// falls with N"* and *"the effect is larger at `bm4`"* are the same two
/// observations. A sweep in N alone cannot separate them; this knob is what
/// makes N the variable with the tile held (`DESIGN.md` §12.2d, issue #492).
///
/// Only `bm` is overridable, and deliberately: `bn`/`sm`/`sn`/`tn` are fixed at
/// `1/1/32/4` for both instantiated decode tiles, and `tm` follows
/// `out_vec_size < 4`. `bm` is the whole difference between them, so it is the
/// whole axis.
///
/// An unparseable or unknown value **panics** rather than falling back:
/// `DESIGN.md` §2.4 — an arm that silently ran the baseline is #69's vacuous
/// determinism run, and the whole point of this knob is to be A/B'd. Engagement
/// is nonetheless proved from the **kernel name** rather than from the flag, per
/// that same rule: the tile is in the `[[host_name]]` the census reports.
fn forced_gemv_bm() -> Option<usize> {
    crate::tuning::tuning().gemm.gemv_bm
}

/// The non-transposed GEMV's tile, derived from the two vector lengths.
///
/// **Extracted so a test can call it rather than restate it.** Issue #492's
/// first version of `the_gemv_tile_boundary_is_at_four_thousand_and_ninety_six`
/// re-implemented this expression inside the test module, and a mutation moving
/// the production boundary from 4096 to 2048 **survived it** — the test was
/// checking its own copy. That is §15.1a's VACUOUS class in the check written to
/// pin the fact, and §15.2 #11b's *"does the fixture vary the axis the code is
/// parameterised on"* answered the wrong way. One definition, two callers.
///
/// Returns `(bm, bn, sm, sn)`. `tm` follows `out_vec_size` at the call site and
/// `tn` is a constant, so neither is part of the tile decision this names.
pub(crate) fn derive_gemv_tile(
    out_vec_size: usize,
    in_vec_size: usize,
) -> (usize, usize, usize, usize) {
    if in_vec_size <= 64 {
        (1, 1, 8, 4)
    } else if in_vec_size >= 16 * out_vec_size {
        (1, 8, 1, 32)
    } else if out_vec_size >= 4096 {
        (8, 1, 1, 32)
    } else {
        (4, 1, 1, 32)
    }
}

/// The non-transposed GEMV's whole tile `(bm, bn, sm, sn, tm, tn)` as
/// `gemv_dispatch` selects it: [`derive_gemv_tile`], #492's override and the
/// `tm`/`tn` that follow. One definition for the dispatcher and for
/// [`gemv_norm_serves`], so a caller's "does the fused kernel serve this
/// shape" cannot drift from the tile the dispatcher then picks.
pub(crate) fn gemv_tile_dims(
    out_vec_size: usize,
    in_vec_size: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let (bm, bn, sm, sn) = derive_gemv_tile(out_vec_size, in_vec_size);
    // #492's override, applied AFTER the shipping branch has chosen, so an
    // unset variable leaves this expression byte-for-byte what it was. It
    // reaches only the `(4|8, 1, 1, 32)` decode tiles: the `in_vec_size`
    // branches above pick `bm == 1` shapes whose `bn`/`sm`/`sn` differ, and
    // substituting `bm` alone there would name a tile that is not
    // instantiated. Guarded rather than trusted, because the failure mode is
    // a `LoadFunctionError` at a shape the caller cannot see.
    let bm = match forced_gemv_bm() {
        Some(forced) if bn == 1 && sm == 1 && sn == 32 => forced,
        _ => bm,
    };
    let tm: usize = if out_vec_size < 4 { 1 } else { 4 };
    (bm, bn, sm, sn, tm, 4usize)
}

/// Whether `call_mlx_gemv_norm` serves a non-transposed `[out, in]`
/// matrix: the tile the dispatcher selects for it is one of the two decode
/// tiles the inline norm is instantiated at (`bm4`/`bm8 bn1 sm1 sn32 tm4
/// tn4`). A caller with an unfused path checks this first and takes that
/// path where it is false, rather than meeting the refusal.
pub fn gemv_norm_serves(out_vec_size: usize, in_vec_size: usize) -> bool {
    let (bm, bn, sm, sn, tm, tn) = gemv_tile_dims(out_vec_size, in_vec_size);
    bn == 1 && sm == 1 && sn == 32 && tn == 4 && tm == 4 && (bm == 4 || bm == 8)
}

/// The most groups `PrologueGroupNorm`'s threadgroup scratch holds
/// (`kGroupNormMaxGroups`).
pub const GEMV_GROUP_NORM_MAX_GROUPS: usize = 8;

/// Whether `call_mlx_gemv_mixed_gnorm` serves an `[n_out, k]` matrix with
/// `groups` groups: the tile `derive_gemv_tile` picks is the one instantiated
/// (`bm1 bn8 sm1 sn32 tm4`, 256 threads), `k` splits into the groups, and the
/// groups fit the scratch. A caller checks this and composes otherwise.
pub fn gemv_mixed_gnorm_serves(n_out: usize, k: usize, groups: usize) -> bool {
    derive_gemv_tile(n_out, k) == (1, 8, 1, 32)
        && n_out >= 4
        && (1..=GEMV_GROUP_NORM_MAX_GROUPS).contains(&groups)
        && k.is_multiple_of(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode tiles walk `K` in one simdgroup; the long-`K` tile splits
    /// it across eight; the short-`K` tile (`sn8`) splits it within a
    /// simdgroup's lanes. Read off the dispatcher's own selection.
    #[test]
    fn the_k_walk_follows_the_selected_tile() {
        // LFM2's `w1` (N 10752, K 2048) and `out_proj` (2048, 2048).
        assert_eq!(gemv_tile(10752, 2048).k_walk(), KWalk::Simdgroup);
        assert_eq!(gemv_tile(2048, 2048).k_walk(), KWalk::Simdgroup);
        // Qwen4-exp's HyperConnection `down` (N 320, K 4·2560): `bn8`.
        assert_eq!(gemv_tile(320, 4 * 2560).bn, 8);
        assert_eq!(gemv_tile(320, 4 * 2560).k_walk(), KWalk::Threadgroup);
        // K <= 64: `sm8 sn4`, the lanes split K four ways.
        assert_eq!(gemv_tile(128, 64).sn, 4);
        assert_eq!(gemv_tile(128, 64).k_walk(), KWalk::Threadgroup);
        // The inline norm serves exactly the simdgroup tiles at bm4/bm8.
        for (n, k) in [(10752, 2048), (2048, 2048), (320, 4 * 2560), (128, 64)] {
            let t = gemv_tile(n, k);
            assert_eq!(
                gemv_norm_serves(n, k),
                t.k_walk() == KWalk::Simdgroup && t.tm == 4 && (t.bm == 4 || t.bm == 8),
                "[{n}, {k}]"
            );
        }
    }
}
