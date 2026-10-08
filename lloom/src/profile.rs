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
pub trait Regime:
    sealed::Regime + Copy + Default + std::fmt::Debug + Send + Sync + 'static
{
    const NAME: &'static str;
    /// The CLI's `--profile` word.
    const PROFILE: &'static str;
    /// The run-time contract this regime is (`lloom_tensor::tuning`): what
    /// [`hold`] pins the process to when a family is built at this regime.
    const CONTRACT: lloom_tensor::tuning::Contract;
}

/// Whether per-position state can be rolled back (a speculative verify's
/// requirement — `State<P>: CacheRewind` by construction when `Ring`).
pub trait Rewind:
    sealed::Rewind + Copy + Default + std::fmt::Debug + Send + Sync + 'static
{
    const NAME: &'static str;
    const REWINDABLE: bool;
    /// The evidence a speculative driver takes (`RewindProof<P>`): `Some`
    /// on `Ring`, `None` on `No` — so generic code over any `P` asks the
    /// axis and the `No` shapes refuse by name, while the driver's signature
    /// cannot be called without the evidence at all.
    fn rewindable<Q: Profile<Rewind = Self>>() -> Option<RewindProof<Q>>;
}

/// One sequence, or paged KV with row pools.
///
/// The axis selects a family's state type through [`Fork`]: a family names
/// the two candidate types once, and `Pick<F>` is the one this axis takes,
/// so a `State<P>` field typed `<P::Rows as Rows>::Pick<F>` holds exactly the
/// selected arm — the other is not a field that allocates nothing, it is not
/// a field (`profiles-and-memory-design.md` §3, §7.1).
pub trait Rows: sealed::Rows + Copy + Default + std::fmt::Debug + Send + Sync + 'static {
    const NAME: &'static str;
    const PAGED: bool;
    /// The arm of `F` this axis selects.
    type Pick<F: Fork>: Picked<F>;
    /// Build the selected arm: `one` on `One`, `paged` on `Paged`. The
    /// unselected constructor is never called and compiles to nothing.
    fn pick<F: Fork, E>(
        one: impl FnOnce() -> std::result::Result<F::One, E>,
        paged: impl FnOnce() -> std::result::Result<F::Paged, E>,
    ) -> std::result::Result<Self::Pick<F>, E>;
}

/// A family's two candidate state types for the [`Rows`] axis: what a
/// sequence's own store is, and what a row of a shared pool is. The axis
/// picks one ([`Rows::Pick`]); generic code reads whichever it holds through
/// [`Picked::arm`] — a `match` the monomorphization resolves to one arm, no
/// `dyn`, no branch left at run time.
pub trait Fork: 'static {
    type One: Clone + std::fmt::Debug + Send + Sync + 'static;
    type Paged: Clone + std::fmt::Debug + Send + Sync + 'static;
}

/// The selected arm, borrowed.
pub enum Arm<'a, F: Fork> {
    One(&'a F::One),
    Paged(&'a F::Paged),
}

/// The selected arm, borrowed mutably.
pub enum ArmMut<'a, F: Fork> {
    One(&'a mut F::One),
    Paged(&'a mut F::Paged),
}

/// What [`Rows::Pick`] holds: one arm of `F`, named by the axis.
pub trait Picked<F: Fork>: Clone + std::fmt::Debug + Send + Sync + 'static {
    fn arm(&self) -> Arm<'_, F>;
    fn arm_mut(&mut self) -> ArmMut<'_, F>;
}

/// `One`'s pick: the sequence's own store.
pub struct OneOf<F: Fork>(pub F::One);
/// `Paged`'s pick: a row of the pool.
pub struct PagedOf<F: Fork>(pub F::Paged);

impl<F: Fork> Picked<F> for OneOf<F> {
    #[inline]
    fn arm(&self) -> Arm<'_, F> {
        Arm::One(&self.0)
    }
    #[inline]
    fn arm_mut(&mut self) -> ArmMut<'_, F> {
        ArmMut::One(&mut self.0)
    }
}

impl<F: Fork> Picked<F> for PagedOf<F> {
    #[inline]
    fn arm(&self) -> Arm<'_, F> {
        Arm::Paged(&self.0)
    }
    #[inline]
    fn arm_mut(&mut self) -> ArmMut<'_, F> {
        ArmMut::Paged(&mut self.0)
    }
}

impl<F: Fork> Clone for OneOf<F>
where
    F::One: Clone,
{
    fn clone(&self) -> Self {
        OneOf(self.0.clone())
    }
}

impl<F: Fork> Clone for PagedOf<F>
where
    F::Paged: Clone,
{
    fn clone(&self) -> Self {
        PagedOf(self.0.clone())
    }
}

impl<F: Fork> std::fmt::Debug for OneOf<F>
where
    F::One: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OneOf").field(&self.0).finish()
    }
}

impl<F: Fork> std::fmt::Debug for PagedOf<F>
where
    F::Paged: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PagedOf").field(&self.0).finish()
    }
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
    const CONTRACT: lloom_tensor::tuning::Contract = lloom_tensor::tuning::Contract::Deterministic;
}
impl Regime for Free {
    const NAME: &'static str = "Free";
    const PROFILE: &'static str = "fast";
    const CONTRACT: lloom_tensor::tuning::Contract = lloom_tensor::tuning::Contract::Fast;
}

/// The process held to `G`: a family built at a regime states the contract
/// (`tuning::pin_regime`), or — when one is already in force (named by
/// `LLOOM_PROFILE`, pinned by a harness, read by a kernel) — is refused
/// unless that one is `G`'s. A model's regime is
/// what its type names, never what the process inherited (the trap #898
/// found: a suite computing `Exact`'s bits under an inherited `fast`).
///
/// ```
/// use lloom::profile::{hold, Exact, Free};
/// use lloom_tensor::tuning::{pin_contract, Contract};
/// // The process is pinned to `fast`; a family built at `Exact` is refused,
/// // one at `Free` is held.
/// let _ = pin_contract(Contract::Fast);
/// assert!(hold::<Free>("doc").is_ok());
/// let err = hold::<Exact>("doc").unwrap_err().to_string();
/// assert!(err.contains("built at the Exact regime"), "{err}");
/// ```
pub fn hold<G: Regime>(who: &str) -> lloom_tensor::Result<()> {
    let held = lloom_tensor::tuning::pin_regime(G::CONTRACT).map_err(lloom_tensor::Error::msg)?;
    if held != G::CONTRACT {
        lloom_tensor::bail!(
            "{who}: built at the {} regime (`--profile {}`) in a process whose contract is `{}`: \
             a family runs at the regime its type names (profiles-and-memory-design.md §3)",
            G::NAME,
            G::PROFILE,
            held.name()
        )
    }
    Ok(())
}
impl sealed::Rewind for NoRewind {}
impl sealed::Rewind for Ring {}
impl Rewind for NoRewind {
    const NAME: &'static str = "No";
    const REWINDABLE: bool = false;
    fn rewindable<Q: Profile<Rewind = Self>>() -> Option<RewindProof<Q>> {
        None
    }
}
impl Rewind for Ring {
    const NAME: &'static str = "Ring";
    const REWINDABLE: bool = true;
    fn rewindable<Q: Profile<Rewind = Self>>() -> Option<RewindProof<Q>> {
        Some(RewindProof(PhantomData))
    }
}

/// Evidence that `Q` rewinds — a per-position state a speculative verify
/// can roll back exists by construction (`profiles-and-memory-design.md`
/// §7.3). The speculative drivers (`Session::decode_speculative`,
/// `decode_spec_batched`, `SpecBatchDecoder::new`) take one, and it is only
/// made for `Rewind = Ring` (`RewindProof::of`, `Rewind::rewindable`): a
/// driver on a `No` shape does not compile.
///
/// ```compile_fail
/// use lloom::profile::{Crown, RewindProof};
/// let _ = RewindProof::<Crown>::of(); // `Crown` is `No × One`: no `of`
/// ```
///
/// ```
/// use lloom::profile::{RewindProof, RewindableExact};
/// let _ = RewindProof::<RewindableExact>::of();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RewindProof<Q: Profile>(PhantomData<Q>);

impl<Q: Profile<Rewind = Ring>> RewindProof<Q> {
    pub fn of() -> Self {
        RewindProof(PhantomData)
    }
}

impl<Q: Profile> RewindProof<Q> {
    /// The evidence for `Q`, when its rewind axis gives it; the caller
    /// refuses by name when it does not.
    pub fn ask() -> Option<Self> {
        <Q::Rewind as Rewind>::rewindable::<Q>()
    }
}
impl sealed::Rows for One {}
impl sealed::Rows for Paged {}
impl Rows for One {
    const NAME: &'static str = "One";
    const PAGED: bool = false;
    type Pick<F: Fork> = OneOf<F>;
    #[inline]
    fn pick<F: Fork, E>(
        one: impl FnOnce() -> std::result::Result<F::One, E>,
        _paged: impl FnOnce() -> std::result::Result<F::Paged, E>,
    ) -> std::result::Result<Self::Pick<F>, E> {
        one().map(OneOf)
    }
}
impl Rows for Paged {
    const NAME: &'static str = "Paged";
    const PAGED: bool = true;
    type Pick<F: Fork> = PagedOf<F>;
    #[inline]
    fn pick<F: Fork, E>(
        _one: impl FnOnce() -> std::result::Result<F::One, E>,
        paged: impl FnOnce() -> std::result::Result<F::Paged, E>,
    ) -> std::result::Result<Self::Pick<F>, E> {
        paged().map(PagedOf)
    }
}

/// A profile: the three axes as one type parameter.
pub trait Profile:
    sealed::Profile + Copy + Default + std::fmt::Debug + Send + Sync + 'static
{
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

/// The single-sequence, no-rewind shape at `Q`'s regime: the shape of a
/// state a block keeps for itself beside the family's (the MTP head's lane).
pub type Own<Q> = P<<Q as Profile>::Regime, NoRewind, One>;

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

/// The boundary (§7.4): one `match` from a [`ProfileKey`] to the type it
/// names, running `$body` with `P` bound to that `P<G, W, R>`. Every arm is
/// written once here; a key outside `$bound` is refused by name before any
/// arm runs, so a family instantiates only the shapes its `Capabilities`
/// ship. `$body` is an expression generic in `P` (a turbofish call, a block
/// naming `P`); its value is the macro's.
///
/// ```ignore
/// let key = ProfileKey::from_mode("deterministic", false, false).unwrap();
/// let n = lloom::profile::dispatch!(key, lloom::profile::LFM2, P => { run::<P>(args) })?;
/// ```
#[macro_export]
macro_rules! dispatch {
    ($key:expr, $bound:expr, $p:ident => $body:expr) => {{
        let key: $crate::profile::ProfileKey = $key;
        let bound: $crate::profile::Bound = $bound;
        if !bound.admits(key) {
            Err(::core::convert::From::from($crate::profile::Error::msg(format!(
                "{}: the {key} profile is outside what the family ships (rewind {:?}, rows {:?}{})",
                bound.family,
                bound.rewind,
                bound.rows,
                if bound.except.is_empty() {
                    String::new()
                } else {
                    format!(", except {:?}", bound.except)
                }
            ))))
        } else {
            match (key.regime, key.rewind, key.rows) {
                ("Exact", "No", "One") => {
                    type $p = $crate::profile::CrownExact;
                    $body
                }
                ("Exact", "Ring", "One") => {
                    type $p = $crate::profile::RewindableExact;
                    $body
                }
                ("Exact", "No", "Paged") => {
                    type $p = $crate::profile::PoolExact;
                    $body
                }
                ("Exact", "Ring", "Paged") => {
                    type $p = $crate::profile::PoolSpecExact;
                    $body
                }
                ("Free", "No", "One") => {
                    type $p = $crate::profile::Crown;
                    $body
                }
                ("Free", "Ring", "One") => {
                    type $p = $crate::profile::Rewindable;
                    $body
                }
                ("Free", "No", "Paged") => {
                    type $p = $crate::profile::Pool;
                    $body
                }
                ("Free", "Ring", "Paged") => {
                    type $p = $crate::profile::PoolSpec;
                    $body
                }
                _ => unreachable!("a ProfileKey names one of the eight profiles"),
            }
        }
    }};
}
pub use dispatch;
/// The error the boundary refuses with (`dispatch!`), re-exported so the
/// macro resolves it from any crate.
pub use lloom_tensor::Error;

/// A model built at a profile: what a driver reads the profile off
/// (`RewindProof<M::Profile>` — a speculative driver needs a rewindable
/// state by construction, §7.3).
///
/// A family names its profile at construction and nowhere defaults it,
/// so a model built without one does not compile (#898's trap, closed at
/// the compiler rather than by convention):
///
/// ```compile_fail
/// use lloom::models::qwen4exp::{testing, Config};
/// let cfg = Config::tiny();
/// let model = testing::resident_model(&cfg, &lloom_tensor::Device::Cpu); // which profile?
/// let _ = model.config();
/// ```
///
/// ```no_run
/// use lloom::models::qwen4exp::{testing, Config};
/// use lloom::profile::CrownExact;
/// let cfg = Config::tiny();
/// let model = testing::resident_model::<CrownExact>(&cfg, &lloom_tensor::Device::Cpu);
/// let _ = model.config();
/// ```
pub trait Shaped {
    type Profile: Profile;
}

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

    /// The key at the contract in force (`lloom_tensor::tuning::contract`)
    /// for the shape `(rewind, rows)`: what a family reads off a resolved
    /// config to pick its type.
    pub fn at_contract(rewind: &'static str, rows: &'static str) -> Self {
        let regime = match lloom_tensor::tuning::contract() {
            lloom_tensor::tuning::Contract::Deterministic => Exact::NAME,
            lloom_tensor::tuning::Contract::Fast => Free::NAME,
        };
        Self {
            regime,
            rewind,
            rows,
        }
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
    /// Shapes inside the product the family does not ship, by key.
    pub except: &'static [ProfileKey],
}

impl Bound {
    pub fn admits(&self, key: ProfileKey) -> bool {
        self.rewind.contains(&key.rewind)
            && self.rows.contains(&key.rows)
            && !self.except.contains(&key)
    }
    pub fn profiles(&self) -> Vec<ProfileKey> {
        ALL.iter().copied().filter(|k| self.admits(*k)).collect()
    }
}

/// LFM2 ships all four shapes (§3). Qwen4-exp ships `No × One` (a chat),
/// `Ring × One` (the MTP drafter's verify window on the recurrent rings)
/// and `No × Paged` (`--batch N`: the rows' slot-indexed state and paged
/// KV pool); `Ring × Paged` it does not — the batched loop is the K = 1
/// driver, the MTP head speculates at `--batch 1`.
pub const LFM2: Bound = Bound {
    family: "lfm2",
    rewind: &["No", "Ring"],
    rows: &["One", "Paged"],
    except: &[],
};
pub const QWEN4EXP: Bound = Bound {
    family: "qwen4exp",
    rewind: &["No", "Ring"],
    rows: &["One", "Paged"],
    except: &[PoolSpec::KEY, PoolSpecExact::KEY],
};
/// Qwen3.5 ships `No × One` (one sequence) and `Ring × One` (the MTP
/// drafter's verify window over the generic decoder's per-layer state,
/// `nn::decoder`); no rows.
pub const QWEN3_5: Bound = Bound {
    family: "qwen3_5",
    rewind: &["No", "Ring"],
    rows: &["One"],
    except: &[],
};
/// Every shape Qwen4-exp can be instantiated at (`Ring × Paged` included:
/// the harness's MTP over rows, `qwen4exp-chat --mtp --rows N`); the engine
/// serves `QWEN4EXP`.
pub const QWEN4EXP_ANY: Bound = Bound {
    family: "qwen4exp",
    rewind: &["No", "Ring"],
    rows: &["One", "Paged"],
    except: &[],
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
        assert_eq!(q.len(), 6);
        assert!(q.iter().all(|k| !(k.rows == "Paged" && k.rewind == "Ring")));
        assert!(QWEN4EXP.admits(RewindableExact::KEY));
        assert!(QWEN4EXP.admits(Pool::KEY));
        assert!(!QWEN4EXP.admits(PoolSpec::KEY));
        assert!(!QWEN4EXP.admits(PoolSpecExact::KEY));
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
