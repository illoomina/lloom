//! The profile: two sealed shape axes and the regime, one type
//! (`profiles-and-memory-design.md` §3, #892 B1).
//!
//! `P<G, W, R>` names what a family is instantiated for — the determinism
//! regime (`Exact` opt-in, `Free` the default; RESTRUCTURING §2), whether
//! per-position state can be rolled back (`Rewind`: a speculative verify's
//! need) and whether one sequence or paged rows are served (`Rows`). The
//! preset names stay as aliases. The dtype triple is NOT here: it is the
//! family's default (`lloom_tensor::typed::Triple`, §5), a block overrides
//! it. Nor is memory policy (§4): configuration, not type.
//!
//! The axes are sealed so the set of profiles is a bound (`ProfileKey`,
//! what `Capabilities` lists), not an open list: a new shape adds a
//! variant to one axis rather than multiplying a hand-written table.
//! B1 is the foundation; the blocks and the families take `P` in B3.

use std::marker::PhantomData;

mod sealed {
    pub trait Regime {}
    pub trait Rewind {}
    pub trait Rows {}
    pub trait Profile {}
}

/// The determinism regime (RESTRUCTURING §2): `Exact` is run-to-run
/// bit-identical for a given build and §2.3.5a-attributed; `Free` is the
/// fastest arms with no cross-run claim. `Reproducible` is the sampler's,
/// not a compile-time regime.
pub trait Regime: sealed::Regime + Copy + Default + std::fmt::Debug + 'static {
    const NAME: &'static str;
    /// The CLI's `--profile` word.
    const PROFILE: &'static str;
}

/// Whether per-position state can be rolled back (a speculative verify's
/// requirement — `State<P>: CacheRewind` by construction when `Ring`).
pub trait Rewind: sealed::Rewind + Copy + Default + std::fmt::Debug + 'static {
    const NAME: &'static str;
    const REWINDABLE: bool;
}

/// One sequence, or paged KV with row pools.
pub trait Rows: sealed::Rows + Copy + Default + std::fmt::Debug + 'static {
    const NAME: &'static str;
    const PAGED: bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Exact;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Free;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NoRewind;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Ring;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct One;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Paged;

impl sealed::Regime for Exact {}
impl sealed::Regime for Free {}
impl Regime for Exact {
    const NAME: &'static str = "Exact";
    const PROFILE: &'static str = "deterministic";
}
impl Regime for Free {
    const NAME: &'static str = "Free";
    const PROFILE: &'static str = "fast";
}
impl sealed::Rewind for NoRewind {}
impl sealed::Rewind for Ring {}
impl Rewind for NoRewind {
    const NAME: &'static str = "No";
    const REWINDABLE: bool = false;
}
impl Rewind for Ring {
    const NAME: &'static str = "Ring";
    const REWINDABLE: bool = true;
}
impl sealed::Rows for One {}
impl sealed::Rows for Paged {}
impl Rows for One {
    const NAME: &'static str = "One";
    const PAGED: bool = false;
}
impl Rows for Paged {
    const NAME: &'static str = "Paged";
    const PAGED: bool = true;
}

/// A profile: the three axes as one type parameter.
pub trait Profile: sealed::Profile + Copy + Default + std::fmt::Debug + 'static {
    type Regime: Regime;
    type Rewind: Rewind;
    type Rows: Rows;
    /// The profile as a value — what a registry or a capability bound
    /// holds, and what `key()` compares.
    const KEY: ProfileKey = ProfileKey {
        regime: Self::Regime::NAME,
        rewind: Self::Rewind::NAME,
        rows: Self::Rows::NAME,
    };
    fn key() -> ProfileKey {
        Self::KEY
    }
}

/// `P<G, W, R>`: 2 × 2 × 2 = 8 profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct P<G: Regime, W: Rewind, R: Rows>(PhantomData<(G, W, R)>);

impl<G: Regime, W: Rewind, R: Rows> sealed::Profile for P<G, W, R> {}
impl<G: Regime, W: Rewind, R: Rows> Profile for P<G, W, R> {
    type Regime = G;
    type Rewind = W;
    type Rows = R;
}

/// The preset names stay; `Free` is the default (maintainer 2026-10-01).
pub type Crown = P<Free, NoRewind, One>;
pub type Rewindable = P<Free, Ring, One>;
pub type Pool = P<Free, NoRewind, Paged>;
pub type PoolSpec = P<Free, Ring, Paged>;
/// The `Exact` twins: the opt-in, and what every gate pins.
pub type CrownExact = P<Exact, NoRewind, One>;
pub type RewindableExact = P<Exact, Ring, One>;
pub type PoolExact = P<Exact, NoRewind, Paged>;
pub type PoolSpecExact = P<Exact, Ring, Paged>;

/// A profile as a value: the three axes' names. What a family's
/// `Capabilities` lists as its bound (§3 "which combinations a family
/// ships is a bound, not a list"), and what the boundary (§7.4) matches
/// on from `(--profile, the mode's rewind, the mode's rows)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProfileKey {
    pub regime: &'static str,
    pub rewind: &'static str,
    pub rows: &'static str,
}

impl ProfileKey {
    /// From the CLI's words: the `--profile` arm (`deterministic` |
    /// `fast`, or the regime's own name), whether the mode needs a
    /// rewindable state (`--dspark` / `--mtp`), whether it serves rows
    /// (`serve --batch`). `None` for a profile word that is no regime.
    pub fn from_mode(profile: &str, rewindable: bool, paged: bool) -> Option<Self> {
        let regime = match profile {
            "deterministic" | "Exact" | "exact" => Exact::NAME,
            "fast" | "Free" | "free" => Free::NAME,
            _ => return None,
        };
        Some(Self {
            regime,
            rewind: if rewindable {
                Ring::NAME
            } else {
                NoRewind::NAME
            },
            rows: if paged { Paged::NAME } else { One::NAME },
        })
    }

    /// Today's preset name for the shape (the regime aside).
    pub fn preset(&self) -> &'static str {
        match (self.rewind == Ring::NAME, self.rows == Paged::NAME) {
            (false, false) => "crown",
            (true, false) => "rewindable",
            (false, true) => "pool",
            (true, true) => "pool-spec",
        }
    }
}

impl std::fmt::Display for ProfileKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P<{}, {}, {}>", self.regime, self.rewind, self.rows)
    }
}

/// Every profile, in `P<G, W, R>` order: what a family's bound is drawn
/// from.
pub const ALL: [ProfileKey; 8] = [
    CrownExact::KEY,
    RewindableExact::KEY,
    PoolExact::KEY,
    PoolSpecExact::KEY,
    Crown::KEY,
    Rewindable::KEY,
    Pool::KEY,
    PoolSpec::KEY,
];

/// What a family ships (§3, from `Capabilities`): a bound over `ALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    pub family: &'static str,
    pub rewind: &'static [&'static str],
    pub rows: &'static [&'static str],
}

impl Bound {
    pub fn admits(&self, key: ProfileKey) -> bool {
        self.rewind.contains(&key.rewind) && self.rows.contains(&key.rows)
    }
    pub fn profiles(&self) -> Vec<ProfileKey> {
        ALL.iter().copied().filter(|k| self.admits(*k)).collect()
    }
}

/// LFM2 ships all four shapes (§3); Qwen4-exp `Ring × One` and `No × One`
/// until its rows work lands.
pub const LFM2: Bound = Bound {
    family: "lfm2",
    rewind: &["No", "Ring"],
    rows: &["One", "Paged"],
};
pub const QWEN4EXP: Bound = Bound {
    family: "qwen4exp",
    rewind: &["No", "Ring"],
    rows: &["One"],
};

/// The families' default dtype triples (§5): a block declares its own or
/// takes the family's. LFM2 computes in f16 with f32 accumulation and
/// stores f16; Qwen4-exp's dense path is f32 end to end (its KV, its
/// state; the routed experts override to f16 — #746's f16 arm).
pub mod triples {
    pub type Lfm2 = lloom_tensor::typed::F16F32F16;
    pub type Qwen4Exp = lloom_tensor::typed::F32F32F32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_presets_are_the_eight_keys_and_the_cli_words_reach_them() {
        assert_eq!(Crown::key().to_string(), "P<Free, No, One>");
        assert_eq!(PoolSpecExact::key().to_string(), "P<Exact, Ring, Paged>");
        assert_eq!(
            ProfileKey::from_mode("deterministic", false, false),
            Some(CrownExact::KEY)
        );
        assert_eq!(
            ProfileKey::from_mode("fast", true, true),
            Some(PoolSpec::KEY)
        );
        assert_eq!(ProfileKey::from_mode("balanced", false, false), None);
        assert_eq!(Rewindable::KEY.preset(), "rewindable");
        assert_eq!(Pool::KEY.preset(), "pool");
        let mut seen = std::collections::HashSet::new();
        for k in ALL {
            assert!(seen.insert(k), "{k} twice");
        }
    }

    #[test]
    fn a_familys_bound_is_drawn_from_the_eight_not_listed() {
        assert_eq!(LFM2.profiles().len(), 8);
        let q = QWEN4EXP.profiles();
        assert_eq!(q.len(), 4);
        assert!(q.iter().all(|k| k.rows == "One"));
        assert!(QWEN4EXP.admits(RewindableExact::KEY));
        assert!(!QWEN4EXP.admits(Pool::KEY));
    }

    fn needs_rewind<Q: Profile<Rewind = Ring>>() -> &'static str {
        Q::Rewind::NAME
    }

    #[test]
    fn a_speculative_driver_requires_a_rewindable_profile_by_construction() {
        assert_eq!(needs_rewind::<Rewindable>(), "Ring");
        assert_eq!(needs_rewind::<PoolSpecExact>(), "Ring");
        // `needs_rewind::<Crown>()` does not compile: `Rewind = No`.
    }
}
