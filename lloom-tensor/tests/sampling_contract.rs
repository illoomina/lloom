//! The top-k / top-p sampling contract (`design/docs/dispatch.md` §11.2j),
//! held by its f32 reference (`sampling_contract/oracle.rs`):
//!
//! - the fixtures (`sampling_contract/fixtures.rs`), each derived by hand;
//! - the reference's two bisections against the linear definitions they
//!   shortcut, and the monotone mass both rest on;
//! - the exponential against f64, and its bits pinned;
//! - pins that tell the contract's arithmetic from a sequential sum and from
//!   the platform's `exp`, so neither can stand in for it unnoticed;
//! - the draw against the distribution it defines (χ², with controls).
//!
//! Each implementation of the contract (the host arm, the Metal arms) is
//! held to the same fixtures and to this reference, bit for bit.

#[path = "sampling_contract/fixtures.rs"]
pub mod fixtures;
#[path = "sampling_contract/oracle.rs"]
pub mod oracle;

use fixtures::Expect;
use oracle::{exp32, mass, Arith, Params, CONTRACT};

/// A deterministic row: `vocab` scores in [-6, 6) from splitmix64, on a grid
/// of `grid` when it is nonzero (coarse grids make ties).
pub fn noise_row(seed: u64, vocab: usize, grid: f32) -> Vec<f32> {
    let mut s = seed;
    (0..vocab)
        .map(|_| {
            let x = (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32;
            let v = x * 12.0 - 6.0;
            if grid > 0.0 {
                (v / grid).round() * grid
            } else {
                v
            }
        })
        .collect()
}

pub fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(s: &mut u64) -> f32 {
    (splitmix(s) >> 40) as f32 / (1u64 << 24) as f32
}

#[test]
fn every_fixture_holds_on_the_reference() {
    let (mut kept, mut refused, mut draws) = (0, 0, 0);
    for c in fixtures::all() {
        let got = oracle::kept_ids(&c.z, c.params);
        match &c.expect {
            Expect::Kept(want) => {
                assert_eq!(got.as_ref(), Ok(want), "{}: the kept set", c.name);
                kept += 1;
            }
            Expect::Refused(why) => {
                assert_eq!(got, Err(why.clone()), "{}: the refusal", c.name);
                assert_eq!(
                    oracle::draw(&c.z, c.params, 0.5),
                    Err(why.clone()),
                    "{}: the draw refuses too",
                    c.name
                );
                refused += 1;
            }
        }
        for &(u, id) in &c.draws {
            assert_eq!(
                oracle::draw(&c.z, c.params, u),
                Ok(id),
                "{}: u = {u}",
                c.name
            );
            draws += 1;
        }
    }
    println!("FIXTURES kept={kept} refused={refused} draws={draws}");
    // By count, so a fixture dropped from the table fails here.
    assert_eq!((kept, refused, draws), (33, 11, 48));
}

/// Where a fixture says its scores are f16 values, they are: an
/// implementation fed the row at f16 sees the same scores.
#[test]
fn the_f16_fixtures_are_f16_values() {
    let mut n = 0;
    for c in fixtures::all().into_iter().filter(|c| c.f16_exact) {
        for (i, &x) in c.z.iter().enumerate() {
            let back = half::f16::from_f32(x).to_f32();
            assert!(
                back.to_bits() == x.to_bits() || (x.is_nan() && back.is_nan()),
                "{}: score {i} = {x} is not an f16 value",
                c.name
            );
        }
        n += 1;
    }
    assert_eq!(n, 41, "fixtures marked f16");
}

/// A wide fixture plants each id once: a collision would quietly change the
/// counts its expectation is derived from.
#[test]
fn the_wide_fixtures_plant_distinct_ids() {
    for c in fixtures::wide() {
        let Expect::Kept(want) = &c.expect else {
            panic!("{}: a wide fixture keeps a set", c.name)
        };
        let mut sorted = want.clone();
        sorted.dedup();
        assert_eq!(&sorted, want, "{}: the kept ids repeat", c.name);
        let distinct: std::collections::BTreeSet<u32> = c.z.iter().map(|x| x.to_bits()).collect();
        assert!(
            distinct.len() <= 4,
            "{}: more score levels than planted",
            c.name
        );
        let want_len = match c.params.k {
            0 => 50,
            k => k.min(if c.params.p < 1.0 { 10 } else { k }),
        };
        assert_eq!(want.len(), want_len, "{}: the kept count", c.name);
    }
}

/// The nucleus by its definition: the least prefix whose mass reaches the
/// target, scanned rather than bisected.
fn nucleus_linear(z: &[f32], params: Params) -> Vec<usize> {
    let inv = oracle::inv_t(params.temperature).unwrap();
    let kept = oracle::top_k(z, params.k).unwrap();
    if params.p == 1.0 {
        return kept;
    }
    let m = z[kept[0]];
    let total = mass(&CONTRACT.weights(z, &kept, m, inv));
    let target = (params.p as f32) * total;
    for j in 1..=kept.len() {
        if mass(&CONTRACT.weights(z, &kept[..j], m, inv)) >= target {
            return kept[..j].to_vec();
        }
    }
    unreachable!("the whole set reaches the target");
}

/// The draw by its definition: the least id whose cumulative mass passes
/// `u` times the total, scanned rather than bisected.
fn draw_linear(z: &[f32], params: Params, u: f32) -> u32 {
    let kept = CONTRACT.nucleus(z, params).unwrap();
    let inv = oracle::inv_t(params.temperature).unwrap();
    let w = CONTRACT.weights(z, &kept, z[kept[0]], inv);
    let total = mass(&w);
    let r = u * total;
    for i in 0..w.len() {
        if CONTRACT.prefix_mass(&w, i) > r {
            return i as u32;
        }
    }
    (0..w.len()).rev().find(|&i| w[i] > 0.0).unwrap() as u32
}

fn random_params(s: &mut u64, vocab: usize) -> Params {
    let t = [0.3, 0.7, 1.0, 2.5][(splitmix(s) % 4) as usize];
    let k = match splitmix(s) % 3 {
        0 => 0,
        _ => 1 + (splitmix(s) as usize % vocab),
    };
    let p = match splitmix(s) % 4 {
        0 => 1.0,
        1 => 0.0,
        _ => uniform(s) as f64,
    };
    Params::new(t, k, p)
}

/// The reference's bisections find what the linear definitions find.
#[test]
fn the_bisections_match_the_linear_definitions() {
    let mut s = 0x1036u64;
    let mut n = 0;
    for &vocab in &[1usize, 5, 37, 100, 1500, 2100] {
        for round in 0..12 {
            let grid = [0.0, 0.25, 1.0][round % 3];
            let mut z = noise_row(splitmix(&mut s), vocab, grid);
            if round % 4 == 3 && vocab > 1 {
                for i in (1..vocab).step_by(3) {
                    z[i] = f32::NEG_INFINITY;
                }
            }
            let params = random_params(&mut s, vocab);
            assert_eq!(
                CONTRACT.nucleus(&z, params).unwrap(),
                nucleus_linear(&z, params),
                "vocab {vocab} round {round} {params:?}"
            );
            for _ in 0..4 {
                let u = uniform(&mut s);
                assert_eq!(
                    oracle::draw(&z, params, u).unwrap(),
                    draw_linear(&z, params, u),
                    "vocab {vocab} round {round} {params:?} u {u}"
                );
                n += 1;
            }
        }
    }
    assert_eq!(n, 6 * 12 * 4);
}

/// The two monotonicities the bisections rest on, over magnitudes that span
/// the f32 range a weight takes: adding a non-negative entry never lowers the
/// mass, so a prefix's mass rises with its length and with its last id.
#[test]
fn the_mass_is_monotone_in_its_set() {
    let mut s = 0xfeedu64;
    let mut steps = 0;
    for &vocab in &[3usize, 64, 1025, 4096, 70_000] {
        let a: Vec<f32> = (0..vocab)
            .map(|_| {
                let e = (splitmix(&mut s) % 60) as i32 - 50;
                uniform(&mut s) * 2f32.powi(e)
            })
            .collect();
        let mut order: Vec<usize> = (0..vocab).collect();
        for i in (1..vocab).rev() {
            order.swap(i, splitmix(&mut s) as usize % (i + 1));
        }
        let mut b = vec![0f32; vocab];
        let mut last = 0f32;
        let stride = (vocab / 200).max(1);
        for (j, &i) in order.iter().enumerate() {
            b[i] = a[i];
            if j % stride == 0 || j == vocab - 1 {
                let m = mass(&b);
                assert!(m >= last, "vocab {vocab}: {m} after {last}");
                last = m;
                steps += 1;
            }
        }
        assert_eq!(last, mass(&a));
    }
    assert_eq!(steps, 680);
}

/// No token with zero weight is drawn, at uniforms placed on every kept
/// token's cumulative boundary and one f32 step either side of it.
#[test]
fn a_zero_weight_token_is_never_drawn() {
    let mut s = 0x2e40u64;
    let mut n = 0;
    for &vocab in &[17usize, 300, 3000] {
        let mut z = noise_row(splitmix(&mut s), vocab, 0.0);
        for (i, x) in z.iter_mut().enumerate() {
            // Every third score underflows its weight at T = 0.5.
            if i % 3 == 1 {
                *x -= 60.0;
            }
        }
        let params = Params::new(0.5, 0, 1.0);
        let kept = CONTRACT.nucleus(&z, params).unwrap();
        let w = CONTRACT.weights(&z, &kept, z[kept[0]], oracle::inv_t(0.5).unwrap());
        assert!(w.contains(&0.0), "no zero weight: vacuous");
        let total = mass(&w);
        for i in 0..vocab {
            let c = CONTRACT.prefix_mass(&w, i) / total;
            for u in [c.next_down(), c, c.next_up()] {
                if (0.0..1.0).contains(&u) {
                    let id = oracle::draw(&z, params, u).unwrap() as usize;
                    assert!(w[id] > 0.0, "vocab {vocab} u {u}: drew {id}, weight 0");
                    n += 1;
                }
            }
        }
    }
    assert!(n > 9000, "{n}");
}

/// ULPs of f32 between `x` and the f64 `y`.
fn ulps(x: f32, y: f64) -> f64 {
    let e = y.abs().log2().floor() as i32;
    (x as f64 - y).abs() / 2f64.powi(e - 23)
}

/// The contract's exponential is within one ULP of `exp` over its whole
/// domain (every 257th f32 in [-87, 0], and its ends), is 0 below it and 1
/// at 0.
#[test]
fn exp32_is_within_one_ulp_of_exp() {
    let lo = (-87.0f32).to_bits();
    let mut worst = (0.0f64, 0.0f32);
    let (mut n, mut rounded_off) = (0u64, 0u64);
    let mut bits = (-0.0f32).to_bits();
    while bits <= lo {
        let x = f32::from_bits(bits);
        let e = ulps(exp32(x), (x as f64).exp());
        if e > worst.0 {
            worst = (e, x);
        }
        rounded_off += u64::from(exp32(x) != (x as f64).exp() as f32);
        n += 1;
        bits += 257;
    }
    for x in [-87.0f32, -86.999_99, -1e-30, -0.0] {
        let e = ulps(exp32(x), (x as f64).exp());
        if e > worst.0 {
            worst = (e, x);
        }
    }
    println!(
        "EXP32 n={n} worst_ulps={:.4} at x={:e} not_correctly_rounded={rounded_off}",
        worst.0, worst.1
    );
    assert!(n > 4_000_000, "{n}");
    assert!(
        worst.0 < 1.0,
        "exp32 is {} ULPs off at {}",
        worst.0,
        worst.1
    );
    assert_eq!(exp32(0.0), 1.0);
    assert_eq!(exp32(-0.0), 1.0);
    assert_eq!(exp32((-87.0f32).next_down()), 0.0);
    assert_eq!(exp32(f32::NEG_INFINITY), 0.0);
    assert!(exp32(-87.0) >= f32::MIN_POSITIVE, "a subnormal weight");
}

/// The constants are the decimals `dispatch.md` §11.2j names, and the
/// function's value at a few points is pinned: a changed constant or step is
/// a changed contract, and every implementation moves with it.
#[test]
fn exp32_is_pinned() {
    let f = f32::from_bits;
    assert_eq!(f(oracle::LOG2E), std::f32::consts::LOG2_E);
    assert_eq!(f(oracle::LN2_HI) as f64, 0.693_359_375);
    assert_eq!(f(oracle::LN2_LO), -2.121_944_4e-4_f32);
    assert_eq!(
        oracle::POLY.map(f),
        [
            1.987_569_1e-4_f32,
            1.398_199_9e-3,
            8.333_452e-3,
            4.166_579_6e-2,
            1.666_666_5e-1,
            0.5
        ]
    );
    let moved: Vec<String> = EXP32_PINS
        .iter()
        .filter(|&&(x, want)| exp32(x).to_bits() != want)
        .map(|&(x, want)| format!("({x:?}, {:#010x}) pinned {want:#010x}", exp32(x).to_bits()))
        .collect();
    assert!(moved.is_empty(), "exp32 moved:\n{}", moved.join("\n"));
}

/// `exp32` at a few points, as computed by the definition when it was
/// written down (each within one ULP of `exp`, by the test above). The last
/// four are points where it is not the correctly rounded value, so a
/// correctly rounded `exp` cannot stand in for it.
const EXP32_PINS: [(f32, u32); 13] = [
    (0.0, 0x3f80_0000),
    (-1e-8, 0x3f80_0000),
    (-0.5, 0x3f1b_4598),
    (-1.0, 0x3ebc_5ab2),
    (-std::f32::consts::LN_2, 0x3f00_0000),
    (-10.0, 0x383e_6bce),
    (-50.0, 0x1b69_2beb),
    (-86.9, 0x00c6_0f89),
    (-87.0, 0x00b3_3687),
    (-0.500_202_9, 0x3f1b_3d88),
    (-0.503_848_7, 0x3f1a_ace8),
    (-0.508_781_2, 0x3f19_ea12),
    (-0.510_175_17, 0x3f19_b330),
];

/// The foils: the contract's pipeline with a plain left-to-right sum, and
/// with the platform's `exp`.
fn sequential(a: &[f32]) -> f32 {
    let mut s = 0f32;
    for &x in a {
        s += x;
    }
    s
}

fn platform_exp(x: f32) -> f32 {
    x.exp()
}

const FOLD_FOIL: Arith = Arith {
    exp: exp32,
    mass: sequential,
};
const EXP_FOIL: Arith = Arith {
    exp: platform_exp,
    mass,
};

#[derive(Clone, Copy, Debug)]
enum Foil {
    Fold,
    Exp,
}

impl Foil {
    fn arith(self) -> Arith {
        match self {
            Foil::Fold => FOLD_FOIL,
            Foil::Exp => EXP_FOIL,
        }
    }
}

/// A row on which the contract's arithmetic and a foil part: the kept set's
/// length (the set is that prefix of the ranked candidates), or a draw.
#[derive(Clone, Copy, Debug)]
struct Pin {
    seed: u64,
    vocab: usize,
    params: Params,
    /// `Some(u)`: a draw at `u`, pinned to `want`; `None`: the kept length.
    u: Option<f32>,
    want: usize,
    foil: Foil,
}

/// Found by `search_pins` below (`measurements/issue-1036-raw/step1/run2/
/// search-pins.out`), then fixed here.
const PINS: [Pin; 9] = [
    Pin {
        seed: 7166411029125059793,
        vocab: 3000,
        params: Params::new(2.0, 1181, 0.9696083068847656),
        u: None,
        want: 1056,
        foil: Foil::Fold,
    },
    Pin {
        seed: 14993398499489279759,
        vocab: 3000,
        params: Params::new(1.0, 455, 0.5804826617240906),
        u: None,
        want: 168,
        foil: Foil::Fold,
    },
    Pin {
        seed: 10752749459413695495,
        vocab: 3000,
        params: Params::new(0.5, 0, 1.0),
        u: Some(0.16891079),
        want: 490,
        foil: Foil::Fold,
    },
    Pin {
        seed: 8518066908502861862,
        vocab: 3000,
        params: Params::new(0.5, 480, 1.0),
        u: Some(0.46938762),
        want: 1505,
        foil: Foil::Fold,
    },
    Pin {
        seed: 5445465867046481102,
        vocab: 5000,
        params: Params::new(0.5, 0, 0.9999961853027344),
        u: None,
        want: 2548,
        foil: Foil::Exp,
    },
    Pin {
        seed: 9388319547254842595,
        vocab: 5000,
        params: Params::new(2.0, 1896, 0.298232764005661),
        u: None,
        want: 265,
        foil: Foil::Exp,
    },
    Pin {
        seed: 11469413305603673856,
        vocab: 5000,
        params: Params::new(1.0, 245, 1.0),
        u: Some(0.44293648),
        want: 2153,
        foil: Foil::Exp,
    },
    Pin {
        seed: 5445465867046481102,
        vocab: 5000,
        params: Params::new(0.5, 0, 1.0),
        u: Some(0.2516549),
        want: 1339,
        foil: Foil::Exp,
    },
    Pin {
        seed: 3232577526175246933,
        vocab: 5000,
        params: Params::new(1.0, 1306, 1.0),
        u: Some(0.009029534),
        want: 32,
        foil: Foil::Exp,
    },
];

fn pin_value(a: Arith, p: &Pin) -> usize {
    let z = noise_row(p.seed, p.vocab, 0.0);
    match p.u {
        None => a.nucleus(&z, p.params).unwrap().len(),
        Some(u) => a.draw(&z, p.params, u).unwrap() as usize,
    }
}

/// The contract's arithmetic decides each pin as pinned, and the foil the
/// pin names decides it otherwise: a sum taken in another order, or the
/// platform's `exp`, cannot stand in for the contract's unseen.
#[test]
fn the_pins_tell_the_contract_from_its_foils() {
    let (mut fold, mut exp) = (0, 0);
    for p in PINS.iter() {
        assert_eq!(pin_value(CONTRACT, p), p.want, "{p:?}: the contract");
        assert_ne!(
            pin_value(p.foil.arith(), p),
            p.want,
            "{p:?}: the foil agrees"
        );
        match p.foil {
            Foil::Fold => fold += 1,
            Foil::Exp => exp += 1,
        }
    }
    assert_eq!((fold, exp), (4, 5));
}

/// Finds rows the contract and a foil decide differently: `cargo test -p
/// lloom-tensor --test main sampling_contract::search_pins -- --ignored
/// --nocapture`.
#[test]
#[ignore = "a search, run by hand to make PINS"]
fn search_pins() {
    let mut s = 0x5ea4c4u64;
    let mut found = (0, 0, 0, 0);
    for attempt in 0..4000u64 {
        let seed = splitmix(&mut s);
        let vocab = [3000usize, 5000][(attempt % 2) as usize];
        let z = noise_row(seed, vocab, 0.0);
        let t = [0.5, 1.0, 2.0][(attempt % 3) as usize];
        let foil = if attempt % 2 == 0 {
            Foil::Fold
        } else {
            Foil::Exp
        };
        let k = if attempt % 5 == 0 {
            0
        } else {
            200 + (splitmix(&mut s) as usize % 2000)
        };
        let inv = oracle::inv_t(t).unwrap();
        let kept = oracle::top_k(&z, k).unwrap();
        let m = z[kept[0]];
        let total = mass(&CONTRACT.weights(&z, &kept, m, inv));
        // A nucleus pin: p at the ratio of a random prefix's mass, and the
        // f32 steps either side of it.
        let j = 1 + splitmix(&mut s) as usize % (kept.len() - 1);
        let ratio = mass(&CONTRACT.weights(&z, &kept[..j], m, inv)) / total;
        for p32 in [
            ratio.next_down().next_down(),
            ratio.next_down(),
            ratio,
            ratio.next_up(),
        ] {
            let params = Params::new(t, k, p32 as f64);
            let pin = Pin {
                seed,
                vocab,
                params,
                u: None,
                want: 0,
                foil,
            };
            let (a, b) = (pin_value(CONTRACT, &pin), pin_value(foil.arith(), &pin));
            if a != b {
                println!("NUCLEUS {:?}", Pin { want: a, ..pin });
                match foil {
                    Foil::Fold => found.0 += 1,
                    Foil::Exp => found.1 += 1,
                }
                break;
            }
        }
        // A draw pin: u at a random kept id's cumulative boundary.
        let params = Params::new(t, k, 1.0);
        let w = CONTRACT.weights(&z, &kept, m, inv);
        let i = kept[splitmix(&mut s) as usize % kept.len()];
        let c = CONTRACT.prefix_mass(&w, i) / total;
        for u in [c.next_down(), c, c.next_up()] {
            if !(0.0..1.0).contains(&u) {
                continue;
            }
            let pin = Pin {
                seed,
                vocab,
                params,
                u: Some(u),
                want: 0,
                foil,
            };
            let (a, b) = (pin_value(CONTRACT, &pin), pin_value(foil.arith(), &pin));
            if a != b {
                println!("DRAW {:?}", Pin { want: a, ..pin });
                match foil {
                    Foil::Fold => found.2 += 1,
                    Foil::Exp => found.3 += 1,
                }
                break;
            }
        }
        if found.0 >= 3 && found.1 >= 3 && found.2 >= 3 && found.3 >= 3 {
            break;
        }
    }
    println!(
        "FOUND nucleus fold={} exp={} draw fold={} exp={}",
        found.0, found.1, found.2, found.3
    );
}

/// Critical values of χ² at p = 0.001, by degrees of freedom.
const CHI2_999: [f64; 21] = [
    0.0, 10.828, 13.816, 16.266, 18.467, 20.515, 22.458, 24.322, 26.124, 27.877, 29.588, 31.264,
    32.909, 34.528, 36.123, 37.697, 39.252, 40.790, 42.312, 43.820, 45.315,
];

/// The draws of `sampler` over `n` uniforms, against `softmax(z / T)` over
/// `support` computed in f64: (χ², degrees of freedom, draws outside).
fn chi_square(
    z: &[f32],
    t: f64,
    support: &[usize],
    n: usize,
    seed: u64,
    sampler: impl Fn(f32) -> u32,
) -> (f64, usize, usize) {
    let m = support
        .iter()
        .map(|&i| z[i] as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = support
        .iter()
        .map(|&i| ((z[i] as f64 - m) / t).exp())
        .collect();
    let sum: f64 = e.iter().sum();
    let mut counts = vec![0usize; z.len()];
    let mut s = seed;
    for _ in 0..n {
        counts[sampler(uniform(&mut s)) as usize] += 1;
    }
    let outside = counts.iter().sum::<usize>() - support.iter().map(|&i| counts[i]).sum::<usize>();
    let chi2 = support
        .iter()
        .zip(&e)
        .map(|(&i, &w)| {
            let want = n as f64 * w / sum;
            (counts[i] as f64 - want).powi(2) / want
        })
        .sum();
    (chi2, support.len() - 1, outside)
}

/// The draw's distribution is the one the contract names — `softmax(z / T)`
/// over the kept set — at two temperatures, top-k and top-p. The controls:
/// the same draws judged against the unfiltered distribution, and a draw
/// that drops the temperature, both fail.
#[test]
fn the_draw_has_the_kept_distribution() {
    let n = 20_000;
    let peaked: Vec<f32> = (0..64).map(|i| 2.0 - 0.25 * i as f32).collect();
    let spread = noise_row(0xc415, 48, 0.0);
    let cells = [
        ("top-k", peaked.clone(), Params::new(0.7, 6, 1.0)),
        ("top-k", peaked.clone(), Params::new(1.3, 6, 1.0)),
        ("top-p", spread.clone(), Params::new(0.7, 0, 0.8)),
        ("top-p", spread.clone(), Params::new(1.3, 0, 0.8)),
        ("top-k-top-p", spread.clone(), Params::new(1.3, 6, 0.9)),
    ];
    for (what, z, params) in &cells {
        let support = CONTRACT.nucleus(z, *params).unwrap();
        assert!(
            (3..CHI2_999.len()).contains(&support.len()),
            "{what} {params:?}: support {}",
            support.len()
        );
        if params.k > 0 && params.p < 1.0 {
            let without_k = Params::new(params.temperature, 0, params.p);
            assert_ne!(
                CONTRACT.nucleus(z, without_k).unwrap(),
                support,
                "{what} {params:?}: k does not bind on this row"
            );
        }
        let (chi2, df, outside) = chi_square(z, params.temperature, &support, n, 0x77, |u| {
            oracle::draw(z, *params, u).unwrap()
        });
        println!("CHI2 {what} T={} k={} p={} support={} chi2={chi2:.2} df={df} critical={} outside={outside}",
            params.temperature, params.k, params.p, support.len(), CHI2_999[df]);
        assert_eq!(outside, 0, "{what} {params:?}: drew outside the kept set");
        assert!(
            chi2 < CHI2_999[df],
            "{what} {params:?}: chi2 {chi2} >= {}",
            CHI2_999[df]
        );

        // Control 1: the unfiltered draw judged against the kept set.
        let unfiltered = Params::new(params.temperature, 0, 1.0);
        let (_, _, outside) = chi_square(z, params.temperature, &support, n, 0x77, |u| {
            oracle::draw(z, unfiltered, u).unwrap()
        });
        assert!(
            outside > 100,
            "{what} {params:?}: an unfiltered draw stayed inside ({outside})"
        );

        // Control 2: the temperature dropped.
        let cold = Params::new(1.0, params.k, params.p);
        let (chi2, df, _) = chi_square(z, params.temperature, &support, n, 0x77, |u| {
            let id = oracle::draw(z, cold, u).unwrap();
            if support.contains(&(id as usize)) {
                id
            } else {
                support[0] as u32
            }
        });
        println!(
            "CHI2-CONTROL {what} T dropped: chi2={chi2:.2} critical={}",
            CHI2_999[df]
        );
        assert!(
            chi2 > CHI2_999[df],
            "{what} {params:?}: T = 1 passed at {chi2}"
        );
    }
}
