//! The conformance fixtures of the top-k / top-p contract
//! (`design/docs/dispatch.md` §11.2j). Every expectation is derived from the
//! contract by hand (in the comment beside it), not read off an
//! implementation, so an implementation and the reference are both judged
//! against it.

#![allow(dead_code)]

use super::oracle::{Params, Refusal};

/// What a case expects.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// The kept set, ids ascending.
    Kept(Vec<u32>),
    Refused(Refusal),
}

/// One row, one parameter set, what is kept and what a few uniforms draw.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub z: Vec<f32>,
    pub params: Params,
    pub expect: Expect,
    /// `(u, id)`: the uniform and the token it draws.
    pub draws: Vec<(f32, u32)>,
    /// Every score is an f16 value, so the case also holds at f16.
    pub f16_exact: bool,
}

/// The largest f32 below 1: the top of the uniform's range.
pub const U_MAX: f32 = 0.999_999_94;

/// A row of `vocab` at `base`, with `plants` written over it.
pub fn planted(vocab: usize, base: f32, plants: &[(usize, f32)]) -> Vec<f32> {
    let mut z = vec![base; vocab];
    for &(i, x) in plants {
        z[i] = x;
    }
    z
}

fn case(
    name: &'static str,
    z: Vec<f32>,
    params: Params,
    expect: Expect,
    draws: &[(f32, u32)],
    f16_exact: bool,
) -> Case {
    Case {
        name,
        z,
        params,
        expect,
        draws: draws.to_vec(),
        f16_exact,
    }
}

fn kept(ids: &[u32]) -> Expect {
    Expect::Kept(ids.to_vec())
}

const NEG: f32 = f32::NEG_INFINITY;
const T1: f64 = 1.0;

/// The small rows: ties, edges, masks, refusals.
pub fn small() -> Vec<Case> {
    let ln = |x: f64| x.ln() as f32;
    let flat4 = vec![0.0f32; 4];
    vec![
        // Eight equal scores, k = 2: the two lowest ids. Each weighs 1, the
        // mass is 2, and the first id's cumulative mass is 1: u·2 < 1 draws
        // id 0, u·2 >= 1 draws id 1.
        case(
            "all_equal_k2",
            vec![0.5; 8],
            Params::new(T1, 2, 1.0),
            kept(&[0, 1]),
            &[(0.0, 0), (0.25, 0), (0.5, 1), (0.75, 1), (U_MAX, 1)],
            true,
        ),
        // The k-th value is tied three ways (ids 2, 3, 4 at 2.0) and k = 3
        // keeps id 0 (3.0) and the two lowest of the tie. Weights 1, e^-1,
        // e^-1: the cumulative mass is 1 at id 0 and 1.368 at id 2, of 1.736.
        case(
            "kth_tie_lowest_ids",
            vec![3.0, 1.0, 2.0, 2.0, 2.0, 0.0],
            Params::new(T1, 3, 1.0),
            kept(&[0, 2, 3]),
            &[(0.0, 0), (0.5, 0), (0.6, 2), (0.9, 3)],
            true,
        ),
        // k = 1 over a tied maximum: the lower id, drawn at every u.
        case(
            "k1_lowest_max",
            vec![1.0, 5.0, 5.0, 0.0],
            Params::new(T1, 1, 1.0),
            kept(&[1]),
            &[(0.0, 1), (0.5, 1), (U_MAX, 1)],
            true,
        ),
        // k = 0, k = V and k > V keep every candidate. Weights e^-.375,
        // e^-.25, e^-.125, 1: cumulative .687, 1.466, 2.349, 3.349.
        case(
            "k0_is_off",
            vec![0.125, 0.25, 0.375, 0.5],
            Params::new(T1, 0, 1.0),
            kept(&[0, 1, 2, 3]),
            &[(0.1, 0), (0.3, 1), (0.6, 2), (0.9, 3)],
            true,
        ),
        case(
            "k_equal_v_is_off",
            vec![0.125, 0.25, 0.375, 0.5],
            Params::new(T1, 4, 1.0),
            kept(&[0, 1, 2, 3]),
            &[(0.1, 0), (0.3, 1), (0.6, 2), (0.9, 3)],
            true,
        ),
        case(
            "k_past_v_is_off",
            vec![0.125, 0.25, 0.375, 0.5],
            Params::new(T1, 5, 1.0),
            kept(&[0, 1, 2, 3]),
            &[(0.1, 0), (0.3, 1), (0.6, 2), (0.9, 3)],
            true,
        ),
        // Probabilities .4, .35, .25: k = 2 then p = .5. The
        // nucleus is conditional on the top-k set: weights 1, .875, mass
        // 1.875, target .9375, and id 0 alone reaches it. (The full row's
        // mass, the original-mass reading, keeps ids 0 and 1.)
        case(
            "k_then_p_is_conditional",
            vec![ln(0.4), ln(0.35), ln(0.25)],
            Params::new(T1, 2, 0.5),
            kept(&[0]),
            &[(0.0, 0), (U_MAX, 0)],
            false,
        ),
        // The same at p = .6: target 1.125, id 0 alone is 1, both 1.875.
        case(
            "k_then_p_crossing_kept",
            vec![ln(0.4), ln(0.35), ln(0.25)],
            Params::new(T1, 2, 0.6),
            kept(&[0, 1]),
            &[],
            false,
        ),
        // No k, p = .5: weights 1, .875, .625, mass 2.5, target 1.25.
        case(
            "p_without_k",
            vec![ln(0.4), ln(0.35), ln(0.25)],
            Params::new(T1, 0, 0.5),
            kept(&[0, 1]),
            &[],
            false,
        ),
        // Four equal weights, mass 4. p = .5 is a target of exactly 2, met
        // by the first two: equality stops there.
        case(
            "p_equality_stops",
            flat4.clone(),
            Params::new(T1, 0, 0.5),
            kept(&[0, 1]),
            &[(0.0, 0), (0.5, 1), (U_MAX, 1)],
            true,
        ),
        // One f32 step above .5: a target of 2.0000002, which two do not
        // reach.
        case(
            "p_one_step_above_equality",
            flat4.clone(),
            Params::new(T1, 0, 0.500_000_059_604_644_8),
            kept(&[0, 1, 2]),
            &[],
            true,
        ),
        // One f32 step below .5: a target of 1.9999999, reached by two.
        case(
            "p_one_step_below_equality",
            flat4.clone(),
            Params::new(T1, 0, 0.499_999_970_197_677_6),
            kept(&[0, 1]),
            &[],
            true,
        ),
        case(
            "p_quarter",
            flat4.clone(),
            Params::new(T1, 0, 0.25),
            kept(&[0]),
            &[],
            true,
        ),
        case(
            "p_three_quarters",
            flat4,
            Params::new(T1, 0, 0.75),
            kept(&[0, 1, 2]),
            &[],
            true,
        ),
        // p = 0 keeps the top candidate, the lower id of a tie.
        case(
            "p0_top_token",
            vec![0.125, 0.875, 0.875, 0.375],
            Params::new(T1, 0, 0.0),
            kept(&[1]),
            &[(0.0, 1), (U_MAX, 1)],
            true,
        ),
        case(
            "p1_is_off",
            vec![0.125, 0.875, 0.875, 0.375],
            Params::new(T1, 0, 1.0),
            kept(&[0, 1, 2, 3]),
            &[],
            true,
        ),
        // p = 1 keeps the whole set, a token too light to move its mass
        // included: id 0 weighs exp(-50) ≈ 2e-22, the mass is 2 with or
        // without it, and u = 0 (r = 0) draws it, its cumulative mass being
        // positive. (A cutoff at the full mass would drop it.)
        case(
            "p1_keeps_a_negligible_token",
            vec![-50.0, 0.0, 0.0],
            Params::new(T1, 0, 1.0),
            kept(&[0, 1, 2]),
            &[(0.0, 0), (U_MAX, 2)],
            true,
        ),
        case(
            "k2_p1",
            vec![0.125, 0.875, 0.875, 0.375],
            Params::new(T1, 2, 1.0),
            kept(&[1, 2]),
            &[],
            true,
        ),
        case(
            "k3_p0",
            vec![0.125, 0.875, 0.875, 0.375],
            Params::new(T1, 3, 0.0),
            kept(&[1]),
            &[],
            true,
        ),
        // -inf is a mask: never a candidate, never counted toward k.
        case(
            "masks_do_not_count_toward_k",
            vec![NEG, 1.0, NEG, 0.5, 0.25],
            Params::new(T1, 2, 1.0),
            kept(&[1, 3]),
            &[],
            true,
        ),
        case(
            "fewer_candidates_than_k",
            vec![NEG, 1.0, NEG, 0.5, 0.25],
            Params::new(T1, 5, 1.0),
            kept(&[1, 3, 4]),
            &[],
            true,
        ),
        // Weights 1, e^-.5, e^-.75 at ids 1, 3, 4: cumulative 0 at id 0,
        // 1 at ids 1 and 2, 1.607 at id 3, 2.079 at id 4. u = 0 draws id 1,
        // not the masked id 0.
        case(
            "a_mask_is_never_drawn",
            vec![NEG, 1.0, NEG, 0.5, 0.25],
            Params::new(T1, 0, 1.0),
            kept(&[1, 3, 4]),
            &[(0.0, 1), (0.5, 3), (0.95, 4), (U_MAX, 4)],
            true,
        ),
        case(
            "masked_p0",
            vec![NEG, 1.0, NEG, 0.5, 0.25],
            Params::new(T1, 0, 0.0),
            kept(&[1]),
            &[],
            true,
        ),
        // -0 and +0 are one score: the tie goes to the lower id.
        case(
            "signed_zero_is_one_score",
            vec![-0.0, 0.0, -1.0],
            Params::new(T1, 1, 1.0),
            kept(&[0]),
            &[(U_MAX, 0)],
            true,
        ),
        case(
            "signed_zero_k2",
            vec![-0.0, 0.0, -1.0],
            Params::new(T1, 2, 1.0),
            kept(&[0, 1]),
            &[],
            true,
        ),
        // Refusals: the row.
        case(
            "every_score_masked",
            vec![NEG; 3],
            Params::new(T1, 0, 1.0),
            Expect::Refused(Refusal::NoCandidate),
            &[],
            true,
        ),
        case(
            "nan_score",
            vec![0.0, 1.0, f32::NAN, 2.0],
            Params::new(T1, 2, 1.0),
            Expect::Refused(Refusal::NotANumber(2)),
            &[],
            true,
        ),
        case(
            "nan_beside_masks",
            vec![NEG, f32::NAN],
            Params::new(T1, 0, 1.0),
            Expect::Refused(Refusal::NotANumber(1)),
            &[],
            true,
        ),
        case(
            "positive_infinity_score",
            vec![0.0, f32::INFINITY, 1.0],
            Params::new(T1, 0, 1.0),
            Expect::Refused(Refusal::PositiveInfinity(1)),
            &[],
            true,
        ),
        // Refusals: the parameters.
        case(
            "p_above_one",
            vec![0.0, 1.0],
            Params::new(T1, 0, 1.5),
            Expect::Refused(Refusal::TopP),
            &[],
            true,
        ),
        case(
            "p_negative",
            vec![0.0, 1.0],
            Params::new(T1, 0, -0.25),
            Expect::Refused(Refusal::TopP),
            &[],
            true,
        ),
        case(
            "p_nan",
            vec![0.0, 1.0],
            Params::new(T1, 0, f64::NAN),
            Expect::Refused(Refusal::TopP),
            &[],
            true,
        ),
        case(
            "temperature_zero_is_not_a_sampled_row",
            vec![0.0, 1.0],
            Params::new(0.0, 0, 1.0),
            Expect::Refused(Refusal::Temperature),
            &[],
            true,
        ),
        case(
            "temperature_negative",
            vec![0.0, 1.0],
            Params::new(-1.0, 0, 1.0),
            Expect::Refused(Refusal::Temperature),
            &[],
            true,
        ),
        case(
            "temperature_infinite",
            vec![0.0, 1.0],
            Params::new(f64::INFINITY, 0, 1.0),
            Expect::Refused(Refusal::Temperature),
            &[],
            true,
        ),
        // 1/T is 1e45, past f32: no finite weight for the top token.
        case(
            "temperature_reciprocal_past_f32",
            vec![0.0, 1.0],
            Params::new(1e-45, 0, 1.0),
            Expect::Refused(Refusal::Temperature),
            &[],
            true,
        ),
    ]
}

/// Ids at 3.0 for the top-k fixtures at width `v`.
fn tops(v: usize) -> Vec<usize> {
    vec![0, 5, 9, 100, 1000, 64001, 77777, 99999, v - 1, v - 10]
}

/// Ids at 2.0 for the top-k fixtures at width `v`; the ten lowest are
/// 3, 6, 7, 8, 11, 13, 33, 250, 499, 1024.
fn seconds(v: usize) -> Vec<usize> {
    vec![
        v - 2,
        11,
        13,
        250,
        499,
        1024,
        1025,
        2047,
        4096,
        8191,
        10000,
        12345,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        110000,
        120000,
        127000,
        3,
        7,
        6,
        8,
        33,
        64000,
    ]
}

/// The flat-nucleus fixture's planted ids: `(1237·j + 11) mod v`, distinct
/// for `j < v` since 1237 is prime to both vocabularies.
fn flat_ids(v: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..100).map(|j| (1237 * j + 11) % v).collect();
    ids.sort_unstable();
    ids
}

fn u32s(ids: &[usize]) -> Vec<u32> {
    let mut v: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    v.sort_unstable();
    v
}

/// The vocabulary-scale rows, at LFM2.5's 128 000 and Qwen3.6's 248 320.
pub fn wide() -> Vec<Case> {
    let mut out = Vec::new();
    for &v in &[128_000usize, 248_320] {
        let (name20, name50, flat, qwen) = if v == 128_000 {
            (
                "v128000_top20_ties",
                "v128000_top50_tie",
                "v128000_flat_nucleus",
                "v128000_k20_p95",
            )
        } else {
            (
                "v248320_top20_ties",
                "v248320_top50_tie",
                "v248320_flat_nucleus",
                "v248320_k20_p95",
            )
        };

        // Ten ids at 3.0, thirty at 2.0, the rest at -4: k = 20 keeps the
        // ten and the ten lowest ids of the thirty. u = 0 draws the lowest
        // kept id (0); the top of the uniform the highest (v - 1).
        let plants: Vec<(usize, f32)> = tops(v)
            .into_iter()
            .map(|i| (i, 3.0))
            .chain(seconds(v).into_iter().map(|i| (i, 2.0)))
            .collect();
        let mut want = tops(v);
        let mut s = seconds(v);
        s.sort_unstable();
        want.extend_from_slice(&s[..10]);
        out.push(case(
            name20,
            planted(v, -4.0, &plants),
            Params::new(T1, 20, 1.0),
            Expect::Kept(u32s(&want)),
            &[(0.0, 0), (U_MAX, (v - 1) as u32)],
            true,
        ));

        // Forty ids at 1.5 and twenty at 1.0: k = 50 keeps the forty and the
        // ten lowest of the twenty.
        let firsts: Vec<usize> = (0..40).map(|j| (7919 * j + 3) % v).collect();
        let lasts: Vec<usize> = (0..20).map(|j| (104_729 * j + 17) % v).collect();
        let plants: Vec<(usize, f32)> = firsts
            .iter()
            .map(|&i| (i, 1.5))
            .chain(lasts.iter().map(|&i| (i, 1.0)))
            .collect();
        let mut want = firsts.clone();
        let mut l = lasts.clone();
        l.sort_unstable();
        want.extend_from_slice(&l[..10]);
        out.push(case(
            name50,
            planted(v, -4.0, &plants),
            Params::new(T1, 50, 1.0),
            Expect::Kept(u32s(&want)),
            &[],
            true,
        ));

        // A hundred ids at 0 and the rest at -40, no k, p = .5. Each planted
        // weight is 1 and the rest exp(-40) ≈ 4e-18, whose whole sum (< 1.1e-12)
        // vanishes against any count of ones, so the mass is exactly 100, the
        // target exactly 50, and the nucleus the fifty lowest planted ids. The
        // draw's cumulative mass then counts kept ids: u = .37 is r = 18.5,
        // the 19th kept id; u = 0 the first; the top of the uniform
        // (r = 49.999996) the 50th.
        let ids = flat_ids(v);
        let plants: Vec<(usize, f32)> = ids.iter().map(|&i| (i, 0.0)).collect();
        let nucleus = &ids[..50];
        out.push(case(
            flat,
            planted(v, -40.0, &plants),
            Params::new(T1, 0, 0.5),
            Expect::Kept(u32s(nucleus)),
            &[
                (0.0, nucleus[0] as u32),
                (0.37, nucleus[18] as u32),
                (U_MAX, nucleus[49] as u32),
            ],
            true,
        ));

        // Qwen's checkpoint shape, k = 20 then p = .95: ten ids at 0, ten at
        // -50 (weight 2e-22), the rest at -60. The top-k set is the twenty,
        // its mass exactly 10, the target 9.5 (.95 in f32 times 10, rounded):
        // nine ones fall short, ten reach it.
        let zeros: Vec<usize> = (0..10).map(|j| (24_989 * j + 101) % v).collect();
        let fifties: Vec<usize> = (0..10).map(|j| (31_337 * j + 29) % v).collect();
        let plants: Vec<(usize, f32)> = zeros
            .iter()
            .map(|&i| (i, 0.0))
            .chain(fifties.iter().map(|&i| (i, -50.0)))
            .collect();
        out.push(case(
            qwen,
            planted(v, -60.0, &plants),
            Params::new(T1, 20, 0.95),
            Expect::Kept(u32s(&zeros)),
            &[],
            true,
        ));
    }
    out
}

/// Every fixture.
pub fn all() -> Vec<Case> {
    let mut v = small();
    v.extend(wide());
    v
}
