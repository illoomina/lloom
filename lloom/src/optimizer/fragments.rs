//! The fragment catalogue: the pre-verified kernel blocks a derivation may
//! compose (`design/docs/optimizer-design.md` §7, option i — eOperators are
//! compositions of existing fragments, never novel MSL).
//!
//! Read off the runtime seams, not written down again: for the GEMV a
//! fragment's semantics, precision and operand are associated consts of the
//! `Elementwise`/`Epilogue` impl that already selects its `[[host_name]]`
//! (`nn::fusion`); for the mixed GEMV, the `MixedEpilogue` arm that selects
//! its suffix. So the optimizer and the runtime cannot disagree about what a
//! block does or what it is called.

use crate::nn::fusion::{Epilogue, Operand, OperandKind, Precision, Semantics};
use lloom_tensor::quantized::gemv_mixed::MixedEpilogue;

use super::ir::Core;
use super::numeric::{NumericClass, TargetFpModel};

/// One admissible epilogue block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// The name a derivation and an exclusion set refer to it by: the
    /// `[[host_name]]` suffix without its underscore, prefixed by the core
    /// when the core is not the GEMV's (`"silu"`, `"mixed_silu"`). The fused
    /// node's census family is `gemv_` + this name.
    pub name: &'static str,
    /// The product kernel the block rides.
    pub core: Core,
    /// What the block computes on the accumulator value.
    pub semantics: Semantics,
    /// Where it computes it.
    pub precision: Precision,
    /// The second buffer it binds, if any.
    pub operand: OperandKind,
    /// Whether the block multiplies the accumulator by a dispatch constant
    /// before its transform (the mixed arm's `alpha`): a constant scale
    /// between the product and the block is absorbed with it.
    pub scales: bool,
    /// The fp model the composition computes under: the core's (the block
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
        core: Core::Gemv,
        semantics: E::SEMANTICS,
        precision: E::PRECISION,
        operand: <E::Operand as Operand>::KIND,
        scales: false,
        model: TargetFpModel::of_family("gemv").expect("the GEMV's model is declared"),
    }
}

/// A mixed-GEMV epilogue (`gemv.metal`'s `EpilogueSiluScale` /
/// `EpilogueSigmoidScale`): `act(alpha · acc)` on the f32 accumulator at the
/// store, which the composed path writes and reads back through `affine` and
/// the activation's own launch. Declared `Rounding`, an upper bound: the arm
/// stores at its accumulator's width, so the chain it replaces rounds
/// nothing more (`optimizer-design.md` §10h).
fn describe_mixed(e: MixedEpilogue) -> Option<Fragment> {
    let (name, semantics) = match e {
        MixedEpilogue::None => return None,
        MixedEpilogue::Silu => ("mixed_silu", Semantics::Silu),
        MixedEpilogue::Sigmoid => ("mixed_sigmoid", Semantics::Sigmoid),
    };
    debug_assert_eq!(name, format!("mixed{}", e.name_suffix()));
    Some(Fragment {
        name,
        core: Core::Mixed,
        semantics,
        precision: Precision::Accumulator,
        operand: OperandKind::None,
        scales: true,
        model: TargetFpModel::of_family("gemv_mixed").expect("the mixed GEMV's model is declared"),
    })
}

/// Every epilogue the runtime seams admit: the GEMV's `FusedArm` set
/// (`EpilogueNone` excluded, being the unfused arm itself) and the mixed
/// GEMV's `MixedEpilogue` arms (`None` excluded likewise). A new
/// `impl Epilogue` or `MixedEpilogue` arm is added here, and nowhere else, to
/// become derivable.
pub fn catalogue() -> Vec<Fragment> {
    use crate::nn::fusion::{Gate, Residual, Silu, Swiglu};
    let mut cat = vec![
        describe::<Silu>(),
        describe::<Residual>(),
        describe::<Gate>(),
        describe::<Swiglu>(),
    ];
    cat.extend(
        [
            MixedEpilogue::None,
            MixedEpilogue::Silu,
            MixedEpilogue::Sigmoid,
        ]
        .into_iter()
        .filter_map(describe_mixed),
    );
    cat
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue is read off the types: names are the `[[host_name]]`
    /// suffixes, semantics and operands are the impls'. Absolute references
    /// (`CONTRIBUTING.md` §3.1b rule 1).
    #[test]
    fn the_catalogue_is_the_shipped_epilogues_of_both_cores() {
        let cat = catalogue();
        assert_eq!(cat.len(), 6);
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
        assert_eq!(cat[3].name, "swiglu");
        assert_eq!(cat[3].semantics, Semantics::SwigluGate);
        assert_eq!(cat[3].operand, OperandKind::Weight);
        assert_eq!(cat[3].class(), NumericClass::Rounding);
        // The GEMV's four compose into the GEMV and carry its model.
        let gemv = TargetFpModel::of_family("gemv").unwrap();
        for f in &cat[..4] {
            assert_eq!(
                (f.core, f.model, f.scales),
                (Core::Gemv, gemv, false),
                "{}",
                f.name
            );
        }
        // The mixed GEMV's two: scaling, operand-free, on its own core.
        assert_eq!(cat[4].name, "mixed_silu");
        assert_eq!(cat[4].semantics, Semantics::Silu);
        assert_eq!(cat[5].name, "mixed_sigmoid");
        assert_eq!(cat[5].semantics, Semantics::Sigmoid);
        let mixed = TargetFpModel::of_family("gemv_mixed").unwrap();
        for f in &cat[4..] {
            assert_eq!(
                (f.core, f.model, f.scales, f.operand, f.class()),
                (
                    Core::Mixed,
                    mixed,
                    true,
                    OperandKind::None,
                    NumericClass::Rounding
                ),
                "{}",
                f.name
            );
        }
    }
}
