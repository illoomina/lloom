//! The numerical-class lattice (`design/docs/optimizer-design.md` §6.1).
//!
//! Equivalence in floating point is not an equivalence relation: `a + (b + c)`
//! and `(a + b) + c` differ in f32, and chaining "within-tolerance" rewrites
//! drifts arbitrarily far. So every value in the IR carries a class, every
//! fragment carries the class its composition lands in, and a derivation is
//! admitted only if the class it lands in is within the policy's budget.
//! Crossing the budget is refused, never coerced.

use std::fmt;

/// Where a rewrite's result sits relative to the reference computation,
/// ordered from the strongest guarantee to the weakest. `Ord` is the lattice
/// order: `a <= b` means a value in class `a` also satisfies class `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NumericClass {
    /// Bit-identical to the reference: same products, same roundings, same
    /// order. Index math, argmax, a view.
    BitExact,
    /// Same products in the same order with FEWER roundings — an epilogue
    /// applied on the f32 accumulator where the dispatch pair it replaces
    /// applied it at `T` on the already-rounded store. Bounded per element,
    /// and the bound does not grow with the reduction length: measured
    /// 2.50 f16 ulp flat for the SiLU epilogue and 2.25 for the residual
    /// (`design/docs/fusion.md` §12.2i). Strictly tighter than
    /// [`ReorderEquiv`](Self::ReorderEquiv), whose bound grows with `K`.
    Rounding,
    /// Same products, a different summation order (a `simd_sum` tree against
    /// a serial fold). The bound grows with the reduction length.
    ReorderEquiv,
    /// An approximation (`fast::exp`, a softcap `tanh`): chaining
    /// accumulates the bound.
    Tolerance,
    /// PET-style partial equivalence needing a correction subgraph.
    Partial,
}

impl NumericClass {
    /// The class of a composition: the weakest guarantee among its parts.
    pub fn join(self, other: NumericClass) -> NumericClass {
        self.max(other)
    }

    /// Whether a value in this class is within `budget`.
    pub fn within(self, budget: NumericClass) -> bool {
        self <= budget
    }

    /// The CLI spelling.
    pub fn name(self) -> &'static str {
        match self {
            NumericClass::BitExact => "bit-exact",
            NumericClass::Rounding => "rounding",
            NumericClass::ReorderEquiv => "reorder",
            NumericClass::Tolerance => "tolerance",
            NumericClass::Partial => "partial",
        }
    }

    /// Parse the CLI spelling.
    pub fn parse(s: &str) -> Result<NumericClass, String> {
        match s {
            "bit-exact" | "bitexact" | "exact" => Ok(NumericClass::BitExact),
            "rounding" => Ok(NumericClass::Rounding),
            "reorder" => Ok(NumericClass::ReorderEquiv),
            "tolerance" => Ok(NumericClass::Tolerance),
            "partial" => Ok(NumericClass::Partial),
            other => Err(format!(
                "numeric class must be one of `bit-exact`, `rounding`, `reorder`, \
                 `tolerance` or `partial`, got `{other}`"
            )),
        }
    }
}

impl fmt::Display for NumericClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lattice_is_ordered_strongest_first() {
        assert!(NumericClass::BitExact < NumericClass::Rounding);
        assert!(NumericClass::Rounding < NumericClass::ReorderEquiv);
        assert!(NumericClass::ReorderEquiv < NumericClass::Tolerance);
        assert!(NumericClass::Tolerance < NumericClass::Partial);
    }

    #[test]
    fn join_is_the_weakest_and_within_is_the_budget() {
        assert_eq!(
            NumericClass::BitExact.join(NumericClass::Rounding),
            NumericClass::Rounding
        );
        assert!(NumericClass::Rounding.within(NumericClass::Rounding));
        assert!(NumericClass::BitExact.within(NumericClass::Rounding));
        assert!(!NumericClass::Rounding.within(NumericClass::BitExact));
        assert!(!NumericClass::ReorderEquiv.within(NumericClass::Rounding));
    }

    #[test]
    fn names_round_trip() {
        for c in [
            NumericClass::BitExact,
            NumericClass::Rounding,
            NumericClass::ReorderEquiv,
            NumericClass::Tolerance,
            NumericClass::Partial,
        ] {
            assert_eq!(NumericClass::parse(c.name()), Ok(c));
        }
        assert!(NumericClass::parse("nonsense").is_err());
    }
}
