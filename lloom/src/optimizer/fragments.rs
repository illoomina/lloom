//! The fragment catalogue: the pre-verified kernel blocks a derivation may
//! compose (`design/docs/optimizer-design.md` §7, option i — eOperators are
//! compositions of existing fragments, never novel MSL).
//!
//! Read off the types in [`crate::nn::fusion`], not written down again: a
//! fragment's semantics, precision and operand are associated consts of the
//! `Elementwise`/`Epilogue` impl that already selects its `[[host_name]]`, so
//! the optimizer and the runtime seam cannot disagree about what a block does.

use crate::nn::fusion::{Epilogue, Operand, OperandKind, Precision, Semantics};

use super::numeric::{NumericClass, TargetFpModel};

/// One admissible epilogue block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// The name a derivation and an exclusion set refer to it by
    /// (`"silu"`, `"residual"`): the `[[host_name]]` suffix without its
    /// underscore.
    pub name: &'static str,
    /// What the block computes on the accumulator value.
    pub semantics: Semantics,
    /// Where it computes it.
    pub precision: Precision,
    /// The second buffer it binds, if any.
    pub operand: OperandKind,
    /// The fp model the composition computes under: the GEMV's (the block
    /// rides its store), under this build. A verdict the exclusion set
    /// records for this fragment counts only under this model (§5a).
    pub model: TargetFpModel,
}

impl Fragment {
    /// The class a composition with this block lands in, relative to the
    /// dispatch pair it replaces: the pair rounds the GEMV's store to `T`
    /// before the block reads it; a block on the accumulator skips that
    /// rounding — same products, same order, one rounding fewer
    /// ([`NumericClass::Rounding`]). A block at store precision would be
    /// bit-exact.
    pub fn class(&self) -> NumericClass {
        match self.precision {
            Precision::Accumulator => NumericClass::Rounding,
            Precision::Store => NumericClass::BitExact,
        }
    }
}

fn describe<E: Epilogue>() -> Fragment {
    Fragment {
        name: E::SUFFIX.strip_prefix('_').unwrap_or(E::SUFFIX),
        semantics: E::SEMANTICS,
        precision: E::PRECISION,
        operand: <E::Operand as Operand>::KIND,
        model: TargetFpModel::of_family("gemv").expect("the GEMV's model is declared"),
    }
}

/// Every epilogue the runtime seam admits (the `FusedArm` set —
/// `EpilogueNone` excluded, being the unfused arm itself). A new
/// `impl Epilogue` is added here, and nowhere else, to become derivable.
pub fn catalogue() -> Vec<Fragment> {
    use crate::nn::fusion::{Gate, Residual, Silu};
    vec![
        describe::<Silu>(),
        describe::<Residual>(),
        describe::<Gate>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue is read off the types: names are the `[[host_name]]`
    /// suffixes, semantics and operands are the impls'. Absolute references
    /// (`CONTRIBUTING.md` §3.1b rule 1).
    #[test]
    fn the_catalogue_is_the_three_shipped_epilogues() {
        let cat = catalogue();
        assert_eq!(cat.len(), 3);
        assert_eq!(cat[0].name, "silu");
        assert_eq!(cat[0].semantics, Semantics::Silu);
        assert_eq!(cat[0].operand, OperandKind::None);
        assert_eq!(cat[0].precision, Precision::Accumulator);
        assert_eq!(cat[0].class(), NumericClass::Rounding);
        assert_eq!(cat[1].name, "residual");
        assert_eq!(cat[1].semantics, Semantics::AddOperand);
        assert_eq!(cat[1].operand, OperandKind::RowIndexed);
        assert_eq!(cat[1].class(), NumericClass::Rounding);
        assert_eq!(cat[2].name, "mul");
        assert_eq!(cat[2].semantics, Semantics::MulOperand);
        assert_eq!(cat[2].operand, OperandKind::RowIndexed);
        assert_eq!(cat[2].class(), NumericClass::Rounding);
        // All compose into the GEMV, so all carry its model.
        let gemv = TargetFpModel::of_family("gemv").unwrap();
        for f in &cat {
            assert_eq!(f.model, gemv, "{}", f.name);
        }
    }
}
