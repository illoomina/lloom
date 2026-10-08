//! Native arms: what a backend's hardware executes, and the refusal of an arm
//! that needs something it does not.
//!
//! An arm exists where the hardware executes its types, and nowhere else. A
//! backend names what an arm can depend on with a [`Capability`] enum and holds
//! a device's report in a [`Set`]. Detection belongs to the backend — the CPU
//! asks the OS for ISA features ([`crate::cpu::arms`]), Metal asks the device
//! for its GPU families (`kernels::metal::arms`). Refusal is shared:
//! [`Set::require_all`] names every missing capability and says the arm is not
//! emulated in software. (§15.2a)

use crate::{Error, Result};
use std::fmt;
use std::marker::PhantomData;

/// One thing an arm can depend on. The position in [`ALL`](Self::ALL) is the
/// bit a [`Set`] holds it in, so `ALL` has at most 32 members.
pub trait Capability: Copy + Eq + fmt::Debug + 'static {
    /// Every capability of the backend, in declaration order.
    const ALL: &'static [Self];
    /// The backend, as a refusal names it: `CPU`, `Metal`.
    const BACKEND: &'static str;
    /// What executes an arm: `CPU`, `GPU`.
    const EXECUTOR: &'static str;
    /// One capability, and several: `feature`/`features`.
    const NOUN: (&'static str, &'static str);

    /// The name a disable list and a refusal use.
    fn name(self) -> &'static str;

    /// The position in [`ALL`](Self::ALL).
    fn index(self) -> u32;

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|c| c.name() == name)
    }
}

/// A set of one backend's [`Capability`]s.
pub struct Set<C> {
    bits: u32,
    _c: PhantomData<fn() -> C>,
}

impl<C> Clone for Set<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C> Copy for Set<C> {}
impl<C> PartialEq for Set<C> {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}
impl<C> Eq for Set<C> {}
impl<C> std::hash::Hash for Set<C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state)
    }
}
impl<C> Default for Set<C> {
    fn default() -> Self {
        Self::from_bits(0)
    }
}

impl<C> Set<C> {
    /// The set a [`bits`](Self::bits) word holds, for a backend that keeps a
    /// set in an atomic.
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Set {
            bits,
            _c: PhantomData,
        }
    }

    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }

    pub const fn empty() -> Self {
        Self::from_bits(0)
    }

    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self::from_bits(self.bits | other.bits)
    }

    /// The members of `self` that `other` does not hold.
    pub const fn minus(self, other: Self) -> Self {
        Self::from_bits(self.bits & !other.bits)
    }

    pub const fn contains_all(self, other: Self) -> bool {
        other.bits & !self.bits == 0
    }
}

impl<C: Capability> Set<C> {
    /// Every capability the backend knows.
    pub fn all() -> Self {
        C::ALL.iter().copied().fold(Self::empty(), Self::with)
    }

    #[inline(always)]
    pub fn with(self, capability: C) -> Self {
        Self::from_bits(self.bits | 1 << capability.index())
    }

    #[inline(always)]
    pub fn without(self, capability: C) -> Self {
        Self::from_bits(self.bits & !(1 << capability.index()))
    }

    #[inline(always)]
    pub fn has(self, capability: C) -> bool {
        self.bits & 1 << capability.index() != 0
    }

    pub fn iter(self) -> impl Iterator<Item = C> {
        C::ALL.iter().copied().filter(move |c| self.has(*c))
    }

    /// A comma list of [`Capability::name`]s, whitespace ignored. An unknown
    /// name is an error, so a typo cannot pass for a listed capability.
    pub fn parse_list(list: &str) -> Result<Self> {
        let mut set = Self::empty();
        for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match C::from_name(name) {
                Some(capability) => set = set.with(capability),
                None => {
                    return Err(Error::Msg(format!(
                        "unknown {} {} `{name}` (known: {})",
                        C::BACKEND,
                        C::NOUN.0,
                        Self::all()
                    )))
                }
            }
        }
        Ok(set)
    }

    /// The set an environment variable names as a [`parse_list`](Self::parse_list),
    /// empty when it is unset.
    ///
    /// # Panics
    ///
    /// When the list names a capability the backend does not know: it is an
    /// operator's explicit request, and ignoring it would run the arm it asked
    /// to withdraw.
    pub fn from_env(var: &str) -> Self {
        Self::from_list(std::env::var(var).ok().as_deref(), var)
    }

    /// The set a list names, empty for none; `what` names the option in the
    /// panic, as [`Self::from_env`] does.
    pub fn from_list(list: Option<&str>, what: &str) -> Self {
        match list {
            Some(list) => Self::parse_list(list).unwrap_or_else(|e| panic!("{what}: {e}")),
            None => Self::empty(),
        }
    }

    /// `Ok` when the set holds `capability`; otherwise the refusal of `arm`.
    pub fn require(self, capability: C, arm: &str) -> Result<()> {
        self.require_all(Self::empty().with(capability), arm)
    }

    /// `Ok` when the set holds every member of `needed`; otherwise the refusal
    /// of `arm`, naming each missing capability.
    pub fn require_all(self, needed: Self, arm: &str) -> Result<()> {
        match self.refusal(needed, arm) {
            None => Ok(()),
            Some(msg) => Err(Error::Msg(msg)),
        }
    }

    /// The refusal [`require_all`](Self::require_all) returns, as text, for a
    /// backend whose error type is not [`Error`]. `None` when nothing is
    /// missing.
    pub fn refusal(self, needed: Self, arm: &str) -> Option<String> {
        let missing = needed.minus(self);
        if missing.is_empty() {
            return None;
        }
        let names = missing
            .iter()
            .map(|c| format!("`{}`", c.name()))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "{backend} arm `{arm}` refused: this {executor} does not execute {names}, \
             and the arm is not emulated in software ({backend} {plural}: {self})",
            backend = C::BACKEND,
            executor = C::EXECUTOR,
            plural = C::NOUN.1,
        ))
    }
}

impl<C: Capability> fmt::Display for Set<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for capability in self.iter() {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            f.write_str(capability.name())?;
        }
        if first {
            f.write_str("none")?;
        }
        Ok(())
    }
}

impl<C: Capability> fmt::Debug for Set<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}({self})", C::BACKEND, C::NOUN.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Toy {
        A,
        B,
        C,
    }

    impl Capability for Toy {
        const ALL: &'static [Self] = &[Toy::A, Toy::B, Toy::C];
        const BACKEND: &'static str = "Toy";
        const EXECUTOR: &'static str = "toy";
        const NOUN: (&'static str, &'static str) = ("widget", "widgets");
        fn name(self) -> &'static str {
            match self {
                Toy::A => "a",
                Toy::B => "b",
                Toy::C => "c",
            }
        }
        fn index(self) -> u32 {
            self as u32
        }
    }

    #[test]
    fn a_refusal_names_every_missing_capability_and_only_those() {
        let have = Set::<Toy>::empty().with(Toy::B);
        let need = Set::empty().with(Toy::A).with(Toy::B).with(Toy::C);
        assert_eq!(
            have.require_all(need, "abc").unwrap_err().to_string(),
            "Toy arm `abc` refused: this toy does not execute `a`, `c`, \
             and the arm is not emulated in software (Toy widgets: b)"
        );
        assert!(have.require(Toy::B, "b").is_ok());
        assert!(have.require_all(Set::empty(), "nothing").is_ok());
        assert!(need.contains_all(have) && !have.contains_all(need));
        assert_eq!(need.minus(have), Set::empty().with(Toy::A).with(Toy::C));
    }

    #[test]
    fn an_unknown_name_is_refused_and_lists_the_known_ones() {
        assert_eq!(Set::<Toy>::parse_list(" a ,c,").unwrap().to_string(), "a c");
        assert_eq!(
            Set::<Toy>::parse_list("d").unwrap_err().to_string(),
            "unknown Toy widget `d` (known: a b c)"
        );
        assert_eq!(Set::<Toy>::all().iter().count(), Toy::ALL.len());
    }
}
