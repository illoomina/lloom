//! The f32 reference of the top-k / top-p sampling contract
//! (`design/docs/dispatch.md` §11.2j): the definition transcribed for
//! reading, not for speed. Test-only; never a runtime arm.
//!
//! Every quantity is the f32 value the contract names, each operation one
//! IEEE round-to-nearest step in the order written there: the exponential is
//! [`exp32`], every sum is [`mass`]'s fold. An implementation conforms when
//! it returns what these functions return, bit for bit, on every row.
//!
//! The pipeline takes its exponential and its sum as an [`Arith`], so a test
//! can run it with a different one (a foil) and show that a fixture tells the
//! contract's arithmetic from another; [`CONTRACT`] is the contract's.

#![allow(dead_code)]
// `!(a >= b)` is written where a NaN must take the branch.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::cmp::Ordering;

/// The mass fold's lanes, and lanes per group.
pub const LANES: usize = 1024;
pub const GROUP: usize = 32;

/// [`exp32`]'s constants, as their f32 bit patterns.
pub const EXP_MIN: f32 = -87.0;
pub const LOG2E: u32 = 0x3fb8_aa3b;
pub const LN2_HI: u32 = 0x3f31_8000;
pub const LN2_LO: u32 = 0xb95e_8083;
pub const POLY: [u32; 6] = [
    0x3950_6967,
    0x3ab7_43ce,
    0x3c08_8908,
    0x3d2a_a9c1,
    0x3e2a_aaaa,
    0x3f00_0000,
];

/// The sampler's parameters at a positive temperature: `k = 0` is no top-k,
/// `p = 1.0` no top-p.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub temperature: f64,
    pub k: usize,
    pub p: f64,
}

impl Params {
    pub const fn new(temperature: f64, k: usize, p: f64) -> Self {
        Params { temperature, k, p }
    }
}

/// What the contract refuses for a row or a parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// A score is NaN, at this id.
    NotANumber(usize),
    /// A score is +inf, at this id.
    PositiveInfinity(usize),
    /// Every score is -inf.
    NoCandidate,
    /// The temperature is not finite and positive, or `1/T` is not a finite f32.
    Temperature,
    /// `p` is not in [0, 1].
    TopP,
    /// The uniform is not in [0, 1).
    Uniform,
}

/// The exponential and the sum the pipeline runs on.
#[derive(Clone, Copy)]
pub struct Arith {
    pub exp: fn(f32) -> f32,
    pub mass: fn(&[f32]) -> f32,
}

/// The contract's arithmetic.
pub const CONTRACT: Arith = Arith { exp: exp32, mass };

/// The contract's exponential, for `x <= 0`: zero below [`EXP_MIN`] (and for
/// -inf), otherwise a range reduction by `ln 2` in two parts and a degree-5
/// polynomial, each step one rounded operation or one fused multiply-add.
pub fn exp32(x: f32) -> f32 {
    assert!(!(x > 0.0), "exp32 is defined for x <= 0, got {x}");
    if !(x >= EXP_MIN) {
        return 0.0;
    }
    let f = f32::from_bits;
    let n = (x * f(LOG2E)).round_ties_even();
    let r = n.mul_add(-f(LN2_HI), x);
    let r = n.mul_add(-f(LN2_LO), r);
    let mut a = f(POLY[0]).mul_add(r, f(POLY[1]));
    for &c in &POLY[2..] {
        a = a.mul_add(r, f(c));
    }
    let r2 = r * r;
    let y = a.mul_add(r2, r) + 1.0;
    y * f((((n as i32) + 127) as u32) << 23)
}

/// The contract's sum of a non-negative row: [`LANES`] lanes of
/// `ceil(len / LANES)` consecutive ids, each folded left to right from +0;
/// the lanes in groups of [`GROUP`], each group folded the same way; the
/// groups folded the same way.
pub fn mass(a: &[f32]) -> f32 {
    let v = a.len();
    let c = v.div_ceil(LANES);
    let mut lanes = [0f32; LANES];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let lo = (l * c).min(v);
        let hi = (lo + c).min(v);
        let mut s = 0f32;
        for &x in &a[lo..hi] {
            s += x;
        }
        *lane = s;
    }
    let mut total = 0f32;
    for group in lanes.chunks(GROUP) {
        let mut s = 0f32;
        for &x in group {
            s += x;
        }
        total += s;
    }
    total
}

/// `1/T` rounded to f32, for a finite positive `T` whose reciprocal is a
/// finite f32.
pub fn inv_t(temperature: f64) -> Result<f32, Refusal> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(Refusal::Temperature);
    }
    let inv = (1.0 / temperature) as f32;
    if !inv.is_finite() {
        return Err(Refusal::Temperature);
    }
    Ok(inv)
}

/// The contract's order: `i` before `j` when its score is greater, or equal
/// and its id smaller. Scores compare as numbers, so -0 ties +0.
pub fn order(z: &[f32], i: usize, j: usize) -> Ordering {
    if z[i] > z[j] {
        Ordering::Less
    } else if z[i] < z[j] {
        Ordering::Greater
    } else {
        i.cmp(&j)
    }
}

/// Refuse a NaN or a +inf score, the lowest such id first.
pub fn check_row(z: &[f32]) -> Result<(), Refusal> {
    for (i, &x) in z.iter().enumerate() {
        if x.is_nan() {
            return Err(Refusal::NotANumber(i));
        }
        if x == f32::INFINITY {
            return Err(Refusal::PositiveInfinity(i));
        }
    }
    Ok(())
}

/// The candidates (every finite score) in the contract's order.
pub fn ranked(z: &[f32]) -> Result<Vec<usize>, Refusal> {
    check_row(z)?;
    let mut ids: Vec<usize> = (0..z.len()).filter(|&i| z[i].is_finite()).collect();
    if ids.is_empty() {
        return Err(Refusal::NoCandidate);
    }
    ids.sort_by(|&i, &j| order(z, i, j));
    Ok(ids)
}

/// The top-k set in order: the first `k` candidates, every candidate when
/// `k` is 0 or at least their number.
pub fn top_k(z: &[f32], k: usize) -> Result<Vec<usize>, Refusal> {
    let mut ids = ranked(z)?;
    if k != 0 && k < ids.len() {
        ids.truncate(k);
    }
    Ok(ids)
}

impl Arith {
    /// The weights of `ids` against the row's top score `m`, dense over the
    /// row and zero elsewhere: `exp((z - m) · inv_t)`, the subtraction and
    /// the product each rounded to f32.
    pub fn weights(&self, z: &[f32], ids: &[usize], m: f32, inv_t: f32) -> Vec<f32> {
        let mut w = vec![0f32; z.len()];
        for &i in ids {
            w[i] = (self.exp)((z[i] - m) * inv_t);
        }
        w
    }

    /// The sum of `a` with every entry past `i` taken as zero: the draw's
    /// cumulative mass at `i`.
    pub fn prefix_mass(&self, a: &[f32], i: usize) -> f32 {
        let mut b = a.to_vec();
        for x in &mut b[i + 1..] {
            *x = 0.0;
        }
        (self.mass)(&b)
    }

    /// The kept set in the contract's order: top-k, then the shortest prefix
    /// of it whose mass reaches `p` times the top-k set's mass. `p = 1` keeps
    /// the top-k set.
    pub fn nucleus(&self, z: &[f32], params: Params) -> Result<Vec<usize>, Refusal> {
        if !(0.0..=1.0).contains(&params.p) {
            return Err(Refusal::TopP);
        }
        let inv = inv_t(params.temperature)?;
        let kept = top_k(z, params.k)?;
        if params.p == 1.0 {
            return Ok(kept);
        }
        let m = z[kept[0]];
        let total = (self.mass)(&self.weights(z, &kept, m, inv));
        let target = (params.p as f32) * total;
        let reaches = |j: usize| (self.mass)(&self.weights(z, &kept[..j], m, inv)) >= target;
        // The prefix's mass is monotone in its length (§11.2j) and the whole
        // set reaches the target, so the least length that does is found by
        // bisection.
        let (mut lo, mut hi) = (1, kept.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if reaches(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(kept[..lo].to_vec())
    }

    /// The kept set's ids, ascending.
    pub fn kept_ids(&self, z: &[f32], params: Params) -> Result<Vec<u32>, Refusal> {
        let mut ids: Vec<u32> = self.nucleus(z, params)?.iter().map(|&i| i as u32).collect();
        ids.sort_unstable();
        Ok(ids)
    }

    /// The draw with the uniform `u`: the least id whose cumulative mass over
    /// the kept set exceeds `u` times the kept set's mass; when rounding
    /// leaves none, the greatest kept id with a nonzero weight.
    pub fn draw(&self, z: &[f32], params: Params, u: f32) -> Result<u32, Refusal> {
        if !(0.0..1.0).contains(&u) {
            return Err(Refusal::Uniform);
        }
        let kept = self.nucleus(z, params)?;
        let inv = inv_t(params.temperature)?;
        let m = z[kept[0]];
        let w = self.weights(z, &kept, m, inv);
        let total = (self.mass)(&w);
        let r = u * total;
        if !(total > r) {
            let last = (0..w.len()).rev().find(|&i| w[i] > 0.0);
            return Ok(last.expect("the top candidate's weight is 1") as u32);
        }
        // The cumulative mass is monotone in the id and is the total at the
        // last id, so the least id past `r` is found by bisection.
        let (mut lo, mut hi) = (0, w.len() - 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.prefix_mass(&w, mid) > r {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo as u32)
    }
}

/// The contract's kept set, ids ascending.
pub fn kept_ids(z: &[f32], params: Params) -> Result<Vec<u32>, Refusal> {
    CONTRACT.kept_ids(z, params)
}

/// The contract's draw.
pub fn draw(z: &[f32], params: Params, u: f32) -> Result<u32, Refusal> {
    CONTRACT.draw(z, params, u)
}
