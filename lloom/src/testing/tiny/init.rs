//! Keyed weights (`docs/porting-models.md` §21.8, INV-3a): every value is a
//! function of the seed, the tensor's name and the element's index, and of
//! nothing else — not the device, not the order tensors are created in, not
//! whether the tensor was fabricated in memory or read back from a file.
//!
//! [`ValueLaw`] maps a keyed stream to values. The hash laws draw from the
//! key; the ramp laws reproduce the two smooth index ramps the hand-rolled
//! fixtures used (`qwen4exp::testing::lin`, the LFM2 census fixture), so an
//! experiment can run a mutation under both.

use lloom_tensor::{Device, Result, Shape, Tensor};

/// splitmix64's finalizer (Steele, Lea, Flood 2014): a bijection on `u64`
/// whose output bits each depend on every input bit.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a over the name's bytes: stable across builds and platforms, unlike
/// `std`'s `DefaultHasher`.
pub fn name_hash(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The stream of one tensor: `(seed, name)` folded once, then indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key(u64);

impl Key {
    pub fn new(seed: u64, name: &str) -> Self {
        Key(splitmix64(seed ^ splitmix64(name_hash(name))))
    }

    /// Element `i`'s 64 bits.
    pub fn bits(self, i: u64) -> u64 {
        splitmix64(self.0 ^ splitmix64(i))
    }

    /// Element `i` as a uniform value in `[-1, 1)`, from the top 24 bits so
    /// it is exact in an `f32`.
    pub fn unit(self, i: u64) -> f32 {
        let top = (self.bits(i) >> 40) as u32; // 24 bits
        (top as f32) * (2.0 / (1u32 << 24) as f32) - 1.0
    }

    /// A value in `[0, 1)` for this whole stream (a per-tensor phase).
    pub fn phase(self) -> f32 {
        ((self.0 >> 40) as u32) as f32 / (1u32 << 24) as f32
    }
}

/// What a tensor is to the value law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    /// A matrix (or a stack of them) read as `x · Wᵀ`, `fan_in` inputs per
    /// output: a projection, an embedding table, a conv filter.
    Matrix { fan_in: usize },
    /// Uniform in `[lo, hi)` whatever the law: a norm's gain near 1, a bias
    /// near 0, a decay that must stay negative.
    Range { lo: f32, hi: f32 },
    /// Exactly this value.
    Const(f32),
}

impl Role {
    /// A norm's gain: near one, never one, so a gain the kernel ignores or
    /// misindexes changes the output.
    pub const GAIN: Role = Role::Range { lo: 0.9, hi: 1.1 };
    /// A bias: near zero, never zero.
    pub const BIAS: Role = Role::Range {
        lo: -0.02,
        hi: 0.02,
    };

    /// The role a constructor's request implies: a matrix by rank; for a
    /// vector, the gain when the constructor's hint is the constant one (a
    /// norm), else a bias.
    pub fn of_request(shape: &Shape, hint: &crate::nn::Init) -> Role {
        let dims = shape.dims();
        if dims.len() >= 2 {
            Role::Matrix {
                fan_in: dims[1..].iter().product::<usize>().max(1),
            }
        } else if matches!(hint, crate::nn::Init::Const(c) if *c == 1.0) {
            Role::GAIN
        } else {
            Role::BIAS
        }
    }
}

/// How a keyed stream becomes values (INV-3b: which of these the fixtures
/// use is what spikes P-a and P-b decide).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueLaw {
    /// `gain · u / √fan_in` with `u` hash-uniform in `[-1, 1)`: every
    /// element independent, activations of order one at every depth.
    HashUniform { gain: f32 },
    /// Hash-uniform in `[-half, half)` for a matrix, whatever its fan-in
    /// (vLLM's `--load-format dummy`: ±1e-3).
    Uniform { half: f32 },
    /// `sin(i · 0.007 + φ) · 0.08` for a matrix, with φ from the key — the
    /// Qwen4-exp fixtures' ramp (`qwen4exp::testing::lin`), made
    /// order-independent by keying its phase.
    RampSin,
    /// `((i mod 17) − 8) / 32` for every tensor, the role ignored — the LFM2
    /// census fixture's ramp, exactly.
    RampMod17,
}

impl ValueLaw {
    /// The law the tiny models use unless a caller asks for another.
    pub const DEFAULT: ValueLaw = ValueLaw::HashUniform { gain: 1.0 };

    /// Element `i` of a tensor with stream `key` and role `role`.
    pub fn value(self, key: Key, role: Role, i: usize) -> f32 {
        if let ValueLaw::RampMod17 = self {
            return ((i % 17) as f32 - 8.0) / 32.0;
        }
        match role {
            Role::Const(c) => c,
            Role::Range { lo, hi } => lo + (key.unit(i as u64) + 1.0) * 0.5 * (hi - lo),
            Role::Matrix { fan_in } => match self {
                ValueLaw::HashUniform { gain } => {
                    gain * key.unit(i as u64) / (fan_in.max(1) as f32).sqrt()
                }
                ValueLaw::Uniform { half } => half * key.unit(i as u64),
                ValueLaw::RampSin => {
                    ((i as f32 * 0.007 + key.phase() * std::f32::consts::TAU).sin()) * 0.08
                }
                ValueLaw::RampMod17 => unreachable!("handled above"),
            },
        }
    }

    /// `n` values of one tensor.
    pub fn fill(self, seed: u64, name: &str, role: Role, n: usize) -> Vec<f32> {
        let key = Key::new(seed, name);
        (0..n).map(|i| self.value(key, role, i)).collect()
    }

    /// One tensor, f32, on `dev`.
    pub fn tensor(
        self,
        seed: u64,
        name: &str,
        role: Role,
        shape: &Shape,
        dev: &Device,
    ) -> Result<Tensor> {
        let v = self.fill(seed, name, role, shape.elem_count());
        Tensor::from_vec(v, shape.clone(), dev)
    }

    /// The law by its name on a command line: `hash[:gain]`,
    /// `uniform[:half]`, `ramp-sin`, `ramp-mod17`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let (name, arg) = match s.split_once(':') {
            Some((n, a)) => (n, Some(a)),
            None => (s, None),
        };
        let num = |d: f32| -> std::result::Result<f32, String> {
            arg.map_or(Ok(d), |a| a.parse::<f32>().map_err(|e| format!("{s}: {e}")))
        };
        match name {
            "hash" => Ok(ValueLaw::HashUniform { gain: num(1.0)? }),
            "uniform" => Ok(ValueLaw::Uniform { half: num(1e-3)? }),
            "ramp-sin" => Ok(ValueLaw::RampSin),
            "ramp-mod17" => Ok(ValueLaw::RampMod17),
            _ => Err(format!(
                "{s}: hash[:gain] | uniform[:half] | ramp-sin | ramp-mod17"
            )),
        }
    }
}

impl std::fmt::Display for ValueLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueLaw::HashUniform { gain } => write!(f, "hash:{gain}"),
            ValueLaw::Uniform { half } => write!(f, "uniform:{half}"),
            ValueLaw::RampSin => write!(f, "ramp-sin"),
            ValueLaw::RampMod17 => write!(f, "ramp-mod17"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference vectors pin the stream itself: splitmix64's published
    /// outputs for seed 0 (the generator advanced by the golden gamma), and
    /// FNV-1a's for "" and "a". A change to either moves every tiny model's
    /// weights, so it must fail here first.
    #[test]
    fn the_stream_matches_its_references() {
        // splitmix64 with state 0: the first three outputs.
        let mut s = 0u64;
        let mut next = || {
            let out = splitmix64(s);
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            out
        };
        assert_eq!(next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(next(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(next(), 0x06C4_5D18_8009_454F);
        assert_eq!(name_hash(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(name_hash("a"), 0xaf63_dc4c_8601_ec8c);
    }

    /// Same (seed, name, index) → same value; a different name, seed or
    /// index → a different one; the unit stream stays in `[-1, 1)`.
    #[test]
    fn values_are_keyed_by_seed_name_and_index_only() {
        let a = ValueLaw::DEFAULT.fill(
            7,
            "model.layers.0.w1.weight",
            Role::Matrix { fan_in: 16 },
            64,
        );
        let b = ValueLaw::DEFAULT.fill(
            7,
            "model.layers.0.w1.weight",
            Role::Matrix { fan_in: 16 },
            64,
        );
        assert_eq!(a, b);
        let other_name = ValueLaw::DEFAULT.fill(
            7,
            "model.layers.0.w3.weight",
            Role::Matrix { fan_in: 16 },
            64,
        );
        let other_seed = ValueLaw::DEFAULT.fill(
            8,
            "model.layers.0.w1.weight",
            Role::Matrix { fan_in: 16 },
            64,
        );
        assert_ne!(a, other_name);
        assert_ne!(a, other_seed);
        // A prefix of a longer tensor is the shorter tensor: the index, not
        // the length, keys an element.
        let long = ValueLaw::DEFAULT.fill(
            7,
            "model.layers.0.w1.weight",
            Role::Matrix { fan_in: 16 },
            128,
        );
        assert_eq!(&long[..64], &a[..]);
        let key = Key::new(1, "x");
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..100_000 {
            let u = key.unit(i);
            assert!((-1.0..1.0).contains(&u), "{u}");
            lo = lo.min(u);
            hi = hi.max(u);
        }
        assert!(
            lo < -0.999 && hi > 0.999,
            "the stream covers [-1, 1): {lo} {hi}"
        );
    }

    /// The laws' scales, which is what P-b's margins depend on: hash-uniform
    /// at `1/√fan_in`, the dummy law at its half-width, the gains near one.
    #[test]
    fn the_laws_hold_their_scales() {
        let m = ValueLaw::HashUniform { gain: 1.0 }.fill(3, "w", Role::Matrix { fan_in: 64 }, 4096);
        let amax = m.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(amax <= 1.0 / 8.0 && amax > 0.12, "{amax}");
        let u = ValueLaw::Uniform { half: 1e-3 }.fill(3, "w", Role::Matrix { fan_in: 64 }, 4096);
        assert!(u.iter().all(|v| v.abs() <= 1e-3));
        let g = ValueLaw::DEFAULT.fill(3, "norm", Role::GAIN, 4096);
        assert!(g.iter().all(|v| (0.9..1.1).contains(v)));
        assert!(g.iter().any(|v| *v != 1.0));
        // The LFM2 fixture's ramp, exactly, whatever the role.
        let r = ValueLaw::RampMod17.fill(3, "norm", Role::GAIN, 20);
        assert_eq!(r[0], -0.25);
        assert_eq!(r[16], 0.25);
        assert_eq!(r[17], -0.25);
        assert_eq!(
            ValueLaw::parse("hash:2").unwrap(),
            ValueLaw::HashUniform { gain: 2.0 }
        );
        assert_eq!(
            ValueLaw::parse("uniform").unwrap(),
            ValueLaw::Uniform { half: 1e-3 }
        );
        assert!(ValueLaw::parse("ramp").is_err());
    }
}
