//! The fragment catalogue: the pre-verified kernel blocks a derivation may
//! compose (`design/docs/optimizer-design.md` §7, option i — eOperators are
//! compositions of existing fragments, never novel MSL).
//!
//! Read off the runtime seams, not written down again: for the GEMV a
//! fragment's semantics, precision and operand are associated consts of the
//! `Elementwise`/`Epilogue`/`Prologue` impl that already selects its
//! `[[host_name]]` (`nn::fusion`); for the mixed GEMV, the `MixedEpilogue`
//! arm that selects its suffix. So the optimizer and the runtime cannot
//! disagree about what a block does or what it is called.
//!
//! A block rides one END of a product ([`Side`]): an epilogue its store, a
//! prologue its load. A prologue carrying a reduction over `K` says which
//! tile it needs ([`TileNeed`]), and the derivation reads the product's tile
//! off the dispatcher's own selection (`lloom_tensor::gemv_tile`). A block
//! whose measurement is a loss, or whose composition the rules cannot
//! express, is still catalogued — with the reason, so the derivation refuses
//! it by name and the remarks can say so.
//!
//! [`kernels`] are the other shape: two dispatches an existing kernel that is
//! not a product computes in one launch.

use crate::nn::fusion::{
    Epilogue, Operand, OperandKind, Precision, Prologue, PrologueGroupNorm, PrologueNormInline,
    PrologueRmsNorm, Semantics, TileNeed,
};
use lloom_tensor::quantized::gemv_mixed::MixedEpilogue;

use super::ir::Core;
use super::numeric::{NumericClass, TargetFpModel};

/// Which end of a product a block rides, or a seam is at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side {
    /// The product's store: the consumer rides the producer product.
    Store,
    /// The product's load: the producer rides the consumer product.
    Load,
}

/// One block a product kernel instantiates at one of its ends.
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
    /// The end of the product it rides.
    pub side: Side,
    /// The tile it needs of the product (a prologue's reduction over `K`).
    pub needs: TileNeed,
    /// Why the derivation never places it, when it must not: a measured
    /// loss, or a composition its rules cannot express. Refused by name.
    pub refusal: Option<&'static str>,
    /// For a prologue, the epilogues the kernel instantiates beside it (the
    /// prologue alone always): `gemv.metal`'s `_norm`, `_norm_silu`,
    /// `_norm_swiglu`. Empty for an epilogue.
    pub composes_with: &'static [&'static str],
}

impl Fragment {
    /// The class a composition with this block lands in, relative to the
    /// dispatch pair it replaces: the pair rounds the GEMV's store to `T`
    /// before the block reads it; a block on the accumulator skips that
    /// rounding — same products, same order, one rounding fewer
    /// ([`NumericClass::Rounding`]). A block at store precision would be
    /// bit-exact.
    pub fn class(&self) -> NumericClass {
        match (self.semantics, self.precision) {
            // The sum of squares in the K-walk's shuffle tree where the norm
            // took its own: the same products in another order, a bound that
            // grows with `K` (and one rounding fewer besides).
            (Semantics::RmsNorm, _) => NumericClass::ReorderEquiv,
            (_, Precision::Accumulator) => NumericClass::Rounding,
            (_, Precision::Store) => NumericClass::BitExact,
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
        side: Side::Store,
        needs: TileNeed::Any,
        refusal: None,
        composes_with: &[],
    }
}

/// A GEMV prologue, read off its `impl Prologue`. The fused node's census
/// family is `gemv_` + the name, then the epilogue's when one rides too
/// (`gemv_norm_silu`, the kernels' `_norm` + `_silu` suffixes).
fn describe_prologue<P: Prologue>(
    core: Core,
    refusal: Option<&'static str>,
    composes_with: &'static [&'static str],
) -> Fragment {
    let name = P::SUFFIX.strip_prefix('_').unwrap_or(P::SUFFIX);
    let family = match core {
        Core::Mixed => "gemv_mixed",
        _ => "gemv",
    };
    Fragment {
        name: match core {
            Core::Mixed => "mixed_gnorm",
            _ => name,
        },
        core,
        semantics: P::SEMANTICS,
        precision: P::PRECISION,
        operand: OperandKind::None,
        scales: false,
        model: TargetFpModel::of_family(family).expect("the product's model is declared"),
        side: Side::Load,
        needs: P::NEEDS,
        refusal,
        composes_with,
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
        side: Side::Store,
        needs: TileNeed::Any,
        refusal: None,
        composes_with: &[],
    })
}

/// Why the imported-scale norm prologue is never placed.
pub const RMSNORM_REFUSAL: &str = "a measured loss (fusion.md §12.2a): the reduction stays a \
     dispatch, and every threadgroup re-applies the transform to the whole input, 216-432x \
     the work it moves";

/// Why the mixed GEMV's grouped-norm prologue is never placed.
pub const MIXED_GNORM_REFUSAL: &str = "not expressible: the grouped norm's readers include \
     hc_collapse, which is not a product; the hand-built arm (QWEN4EXP_HC_NORM_GEMV) has the \
     collapse read the stream scales `down` writes, a side output no fragment declares \
     (measured +6.61 % end to end, tuning.md §13.5u)";

/// Every block the runtime seams admit: the GEMV's `FusedArm` epilogues
/// (`EpilogueNone` excluded, being the unfused arm itself), the mixed GEMV's
/// `MixedEpilogue` arms (`None` excluded likewise), then the prologues — the
/// GEMV's inline norm and imported-scale norm, the mixed GEMV's grouped norm.
/// A new `impl Epilogue`, `impl Prologue` or `MixedEpilogue` arm is added
/// here, and nowhere else, to become derivable.
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
    cat.push(describe_prologue::<PrologueNormInline>(
        Core::Gemv,
        None,
        &["silu", "swiglu"],
    ));
    cat.push(describe_prologue::<PrologueRmsNorm>(
        Core::Gemv,
        Some(RMSNORM_REFUSAL),
        &[],
    ));
    cat.push(describe_prologue::<PrologueGroupNorm>(
        Core::Mixed,
        Some(MIXED_GNORM_REFUSAL),
        &["mixed_silu"],
    ));
    cat
}

/// The form of a [`KernelFragment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelForm {
    /// `mul(silu(a), b)` — both operands of one extent — as the kernel
    /// `silu_mul(a, b)` (`qwen4exp.metal`: `utils.metal`'s `silu<T>` then
    /// the product, in `T`).
    SiluMul,
    /// `into(.., cast(y), ..)` as `into(.., y, ..)`: the kernel `into` is
    /// instantiated on its operand's type and reads `y` where it is, the
    /// widening the cast stored done in its load (exact: an f16 is an f32).
    LoadCast {
        into: &'static str,
        /// Which operand of `into` the cast feeds.
        operand: usize,
    },
}

/// Two dispatches an existing kernel that is not a product computes in one
/// launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelFragment {
    /// The name a derivation and an exclusion set refer to it by: the
    /// kernel's `[[host_name]]` for a fused pair (`silu_mul_f16`).
    pub name: &'static str,
    /// The kernel family the composition dispatches (`silu_mul`), or the
    /// consumer it is read through (`moe_combine`).
    pub family: &'static str,
    pub form: KernelForm,
    /// The dtypes the kernel is instantiated at, for the value it reads.
    pub dtypes: &'static [lloom_tensor::DType],
    pub class: NumericClass,
}

/// The kernel fragments: the SwiGLU's activation and product in one launch
/// (`silu_mul`, one fragment per instantiation, as the f16 routed step's
/// switch decides the f16 one alone), and the combine reading the routed
/// output at its own type
/// (`combine_cast`) — the two compositions the f16 routed step's hand-built
/// arm runs (`expert-streaming.md` §20.15d).
///
/// `silu_mul` is declared `Rounding`, the bound covering both dtypes: at f16
/// the product rounds the SiLU `silu_f16` would have stored (reassociation
/// off), so it is the pair bit for bit and the f16 arm's digests were
/// measured unmoved; at f32 nothing pins the reassociation. `combine_cast` is
/// exact by construction: an f16 widens to f32 on load as the cast did.
pub fn kernels() -> Vec<KernelFragment> {
    use lloom_tensor::DType;
    vec![
        KernelFragment {
            name: "silu_mul_f32",
            family: "silu_mul",
            form: KernelForm::SiluMul,
            dtypes: &[DType::F32],
            class: NumericClass::Rounding,
        },
        KernelFragment {
            name: "silu_mul_f16",
            family: "silu_mul",
            form: KernelForm::SiluMul,
            dtypes: &[DType::F16],
            class: NumericClass::Rounding,
        },
        KernelFragment {
            name: "combine_cast",
            family: "moe_combine",
            form: KernelForm::LoadCast {
                into: "moe_combine",
                operand: 0,
            },
            dtypes: &[DType::F16],
            class: NumericClass::BitExact,
        },
    ]
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
        assert_eq!(cat.len(), 9);
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
        for f in &cat[4..6] {
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
        for f in &cat[..6] {
            assert_eq!(
                (f.side, f.needs, f.refusal, f.composes_with.len()),
                (Side::Store, TileNeed::Any, None, 0)
            );
        }
        // The prologues: the inline norm proposable at a simdgroup-K tile,
        // in the reorder class; the imported-scale norm and the grouped norm
        // catalogued with the reason they are refused.
        assert_eq!(
            (cat[6].name, cat[6].core, cat[6].side, cat[6].needs, cat[6].class()),
            (
                "norm",
                Core::Gemv,
                Side::Load,
                TileNeed::SimdgroupKWalk,
                NumericClass::ReorderEquiv
            )
        );
        assert_eq!(cat[6].refusal, None);
        assert_eq!(cat[6].composes_with, ["silu", "swiglu"]);
        assert_eq!((cat[7].name, cat[7].side), ("rmsnorm", Side::Load));
        assert_eq!(cat[7].refusal, Some(RMSNORM_REFUSAL));
        assert_eq!(
            (cat[8].name, cat[8].core, cat[8].semantics, cat[8].class()),
            ("mixed_gnorm", Core::Mixed, Semantics::GroupRmsNorm, NumericClass::BitExact)
        );
        assert_eq!(cat[8].refusal, Some(MIXED_GNORM_REFUSAL));
        let k = kernels();
        assert_eq!(
            k.iter().map(|k| (k.name, k.class)).collect::<Vec<_>>(),
            [
                ("silu_mul_f32", NumericClass::Rounding),
                ("silu_mul_f16", NumericClass::Rounding),
                ("combine_cast", NumericClass::BitExact)
            ]
        );
    }
}
