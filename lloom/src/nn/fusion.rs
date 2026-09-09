//! Composition-as-type for the fused GEMV: the `Elementwise`/`Operand`/
//! `Epilogue`/`Prologue` traits, `Gemv<P, E>` as the sole constructor of the
//! fused op, and the `.then()` pipeline that reads as a sequence while the type
//! decides fused-vs-separate.
//!
//! The engine is composable blocks; a fused path is those blocks composed at
//! compile time; the monolith is what LTO / monomorphization emits (§15.2 #15,
//! `CONTRIBUTING.md` §3.6). This module is the Rust seam for the GEMV epilogue
//! partition: the type system encodes exactly what §12.2e's gates can check —
//! a functor sees one accumulator value and its index (G1), an epilogue cannot
//! reduce over N and a prologue cannot reduce over K (G2). G3 adjacency is a
//! site property the type cannot see; it is proved by the dispatch census
//! (§1.2: the fused kernel present AND the dispatch it replaces at zero).
//!
//! The runtime seam it lowers to (`GemvFusion`/`GemvFused`, `super::ops`) is
//! unchanged; this is its typed constructor. `Gemv<PrologueNone, EpilogueNone>`
//! is the unfused default, so a build that names no fusion axis moves no digest.

use core::marker::PhantomData;

use lloom_tensor::{Result, Tensor};

use super::ops::GemvFusion;

/// Where a functor applies its transform, for the §2.3.5a report.
///
/// A fused epilogue computes at the **f32 accumulator** where the dispatch pair
/// it replaces computes at `T` on the already-rounded store — strictly fewer
/// roundings and a genuinely different computation, so a fused arm moves the
/// digests by construction and owes §2.3.5a's discriminator (`DESIGN.md`
/// §12.2d, §12.2h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Applied on the unrounded f32 accumulator (the fused arms).
    Accumulator,
    /// Applied at `T` on the already-rounded store (the unfused baseline).
    Store,
}

mod sealed {
    pub trait Sealed {}
}

/// What an [`Elementwise`] block computes on the value it sees — the
/// op-semantics the optimizer derives its rewrite rules from
/// (`design/docs/optimizer-design.md` §5: store the semantics, derive the
/// equivalence; never register the pair).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Semantics {
    /// `y = x`. The identity blocks (`PrologueNone`, `EpilogueNone`).
    Identity,
    /// `y = silu(x)`.
    Silu,
    /// `y = x + operand[row]`, the operand read at the storing thread's own
    /// output row.
    AddOperand,
}

/// Which second operand an [`Operand`] type binds, as a value the optimizer
/// can read off the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    /// No second buffer.
    None,
    /// One buffer of exactly the output extent, indexed by the output row.
    RowIndexed,
}

/// G1 — a functor sees ONE accumulator value and its index (`gemv.metal:392`).
///
/// A reduction cannot implement this: it has no `(Acc, usize) -> Acc` form.
/// This is the supertrait of both [`Epilogue`] and [`Prologue`], so a type that
/// is not `Elementwise` cannot be either — which is what makes
/// `impl Epilogue for Rope` a compile error (RoPE reads `i1` and `i1+d/2`,
/// `reduce.metal:1772`, not a single value).
pub trait Elementwise {
    /// The Metal `[[host_name]]` suffix this block instantiates (`_silu`,
    /// `_residual`). It must equal the suffix `GemvEpilogue::name_suffix`
    /// resolves in `lloom-tensor::kernels`, checked at load against
    /// `MTLLibrary.functionNames` (§3.1c: count the artifact), never derived by
    /// grep.
    const SUFFIX: &'static str;
    /// Where the transform is applied, for the §2.3.5a report.
    const PRECISION: Precision;
    /// What the block computes. The optimizer's rule for this block is
    /// derived from it: a standalone dispatch with these semantics whose
    /// single-use input is a GEMV store is this block riding that store.
    const SEMANTICS: Semantics;
}

/// The second operand a functor may bind (§12.2e: `kDoAxpby` is exactly this
/// shape).
///
/// `RowIndexed` = indexable by the storing thread's own output row — a residual
/// is, a gate (`gate * up`, §12.2b) is NOT and cannot implement it, because a
/// gate is a full vector of the INPUT extent.
pub trait Operand: sealed::Sealed {
    /// The operand's shape, readable off the type by the optimizer.
    const KIND: OperandKind;
}

/// No second operand — a pure elementwise transform (SiLU).
pub struct NoOperand;
impl sealed::Sealed for NoOperand {}
impl Operand for NoOperand {
    const KIND: OperandKind = OperandKind::None;
}

/// A second operand read at the storing thread's own output row (the residual).
pub struct RowIndexed<T>(PhantomData<T>);
impl<T> sealed::Sealed for RowIndexed<T> {}
impl<T> Operand for RowIndexed<T> {
    const KIND: OperandKind = OperandKind::RowIndexed;
}

/// Output side of the GEMV's output/input partition. Evaluated once per output
/// element.
///
/// G2: cannot reduce over N — enforced by [`Elementwise`]'s single-value
/// signature. The associated [`Operand`] names the second buffer the arm binds,
/// if any.
pub trait Epilogue: Elementwise {
    /// The second operand this epilogue reads at the store, if any.
    type Operand: Operand;
}

/// Input side of the partition. Evaluated `n_tgp × K` times.
///
/// G2: cannot reduce over K — same signature argument as [`Epilogue`].
///
/// **SEALED**: a prologue is admitted only with a measured win in its doc and a
/// row in the fusion registry. Today the sealed set is `{PrologueNone}`: both
/// measured prologues lost (§12.2a 216:1, §12.2b 127:1). So `.before(RmsNorm)`
/// does not compile until a measurement opens the gate.
pub trait Prologue: Elementwise + sealed::Sealed {}

/// The identity prologue — no input-side transform. The only member of the
/// sealed `Prologue` set (§1.2).
pub struct PrologueNone;
impl Elementwise for PrologueNone {
    const SUFFIX: &'static str = "";
    const PRECISION: Precision = Precision::Store;
    const SEMANTICS: Semantics = Semantics::Identity;
}
impl sealed::Sealed for PrologueNone {}
impl Prologue for PrologueNone {}

/// The identity epilogue — no output-side transform. The unfused arm's tail.
pub struct EpilogueNone;
impl Elementwise for EpilogueNone {
    const SUFFIX: &'static str = "";
    const PRECISION: Precision = Precision::Store;
    const SEMANTICS: Semantics = Semantics::Identity;
}
impl Epilogue for EpilogueNone {
    type Operand = NoOperand;
}

/// SiLU / swish on the accumulator, replacing a `silu_f16` dispatch (§12.2d,
/// #484). Wins 7 of 7 sizes.
pub struct Silu;
impl Elementwise for Silu {
    const SUFFIX: &'static str = "_silu";
    const PRECISION: Precision = Precision::Accumulator;
    const SEMANTICS: Semantics = Semantics::Silu;
}
impl Epilogue for Silu {
    type Operand = NoOperand;
}

/// The residual add on the accumulator, replacing a `badd_f16` dispatch
/// (§12.2h, #509). Reads the residual at the storing thread's own output row,
/// which is why its [`Operand`] is [`RowIndexed`] — a residual is indexable
/// from the output position and a gate is not.
///
/// The residual tensor is bound by [`Residual::of`], which names the operand
/// where it is consumed: `gemv(w).then(Residual::of(r))`. The marker type
/// `Residual` carries the [`Epilogue`] witness; `Residual::of(r)` carries the
/// runtime operand.
pub struct Residual;
impl Elementwise for Residual {
    const SUFFIX: &'static str = "_residual";
    const PRECISION: Precision = Precision::Accumulator;
    const SEMANTICS: Semantics = Semantics::AddOperand;
}
impl Epilogue for Residual {
    type Operand = RowIndexed<half::f16>;
}
impl Residual {
    /// Bind the residual operand `r`, read at the storing thread's own output
    /// row. `r` is `[.., 1, n]`, exactly the output extent, contiguous.
    pub fn of(r: &Tensor) -> BoundResidual<'_> {
        BoundResidual { r }
    }
}

/// A [`Residual`] with its operand bound. Produced by [`Residual::of`].
pub struct BoundResidual<'r> {
    r: &'r Tensor,
}

/// RoPE — the compile-fail proof for the epilogue seam. RoPE reads `i1` and
/// `i1 + d/2` (`reduce.metal:1772`), not a single accumulator value, so it is
/// **not** [`Elementwise`] and therefore cannot be an [`Epilogue`] or a
/// [`Prologue`]. This is the type seam's proof: the bound refuses a fusion the
/// hardware cannot honour.
///
/// The proof asserts the BOUND, not an orphan `impl` — an orphan
/// `impl Epilogue for Rope` in a doctest would compile-fail on coherence
/// (E0117) regardless of the supertrait, a vacuous pass (§3.1a). Instead a
/// function requiring `Epilogue` cannot be instantiated with `Rope`:
///
/// ```compile_fail
/// use lloom::nn::fusion::{Epilogue, Rope};
/// fn requires_epilogue<E: Epilogue>() {}
/// // `Rope` is not `Elementwise`, so it is not `Epilogue` — this bound is
/// // unsatisfiable and the call does not compile.
/// requires_epilogue::<Rope>();
/// ```
///
/// The admissible epilogues satisfy the same bound (positive control — the
/// proof is not vacuous, the bound is real):
///
/// ```
/// use lloom::nn::fusion::{Epilogue, Residual, Silu};
/// fn requires_epilogue<E: Epilogue>() {}
/// requires_epilogue::<Silu>();
/// requires_epilogue::<Residual>();
/// ```
pub struct Rope;

/// QK-norm — the second compile-fail proof. A reduction over `head_dim`, so it
/// is not [`Elementwise`] and cannot be an [`Epilogue`]. Asserted on the bound,
/// not an orphan `impl`:
///
/// ```compile_fail
/// use lloom::nn::fusion::{Epilogue, QkNorm};
/// fn requires_epilogue<E: Epilogue>() {}
/// // A reduction is not `Elementwise`, so `QkNorm` is not `Epilogue`.
/// requires_epilogue::<QkNorm>();
/// ```
pub struct QkNorm;

/// RMSNorm — a prologue candidate that is sealed out. `RmsNorm` is not a member
/// of the sealed [`Prologue`] set, so `.before(RmsNorm)` does not compile: both
/// measured prologues lost (§12.2a, §12.2b) and none is admitted.
///
/// Asserted on the `Prologue` bound (the sealed set), not an orphan `impl`:
///
/// ```compile_fail
/// use lloom::nn::fusion::{Prologue, RmsNorm};
/// fn requires_prologue<P: Prologue>() {}
/// // `RmsNorm` is not a member of the sealed `Prologue` set, so this bound is
/// // unsatisfiable — the same reason `.before(RmsNorm)` would fail.
/// requires_prologue::<RmsNorm>();
/// ```
///
/// `PrologueNone` — the one member — satisfies it (positive control):
///
/// ```
/// use lloom::nn::fusion::{Prologue, PrologueNone};
/// fn requires_prologue<P: Prologue>() {}
/// requires_prologue::<PrologueNone>();
/// ```
pub struct RmsNorm;

/// The fused projection: a GEMV with an optional input-side [`Prologue`] and an
/// output-side [`Epilogue`], resolved at compile time.
///
/// `E::SUFFIX` selects the Metal instantiation; the default
/// `Gemv<PrologueNone, EpilogueNone>` is the unfused arm and moves no digest.
/// This is the sole typed constructor of the runtime `GemvFused` op — the
/// `.then()` pipeline (below) is the tier-1 surface over it, and both lower to
/// the same `super::ops` dispatch.
pub struct Gemv<P: Prologue = PrologueNone, E: Epilogue = EpilogueNone>(PhantomData<(P, E)>);

/// The runtime `GemvFusion` a fusable [`Epilogue`] lowers to.
///
/// This is the single place the type-level epilogue set is mapped to the
/// runtime enum, so the two cannot drift: a new fusable `impl Epilogue` that
/// forgets its `GemvFusion` arm has no `FusedArm` impl and cannot be applied
/// through `Gemv`. `EpilogueNone`/`Silu`/`Residual` are the three fusable
/// arms `gemv.metal` instantiates.
pub(crate) trait FusedArm: Epilogue {
    const ARM: GemvFusion;
}
impl FusedArm for EpilogueNone {
    const ARM: GemvFusion = GemvFusion::Off;
}
impl FusedArm for Silu {
    const ARM: GemvFusion = GemvFusion::Silu;
}
impl FusedArm for Residual {
    const ARM: GemvFusion = GemvFusion::Residual;
}

// `FusedArm` is deliberately `pub(crate)` — it is the sealed drift-guard above,
// not public API — while `Gemv` is `pub` only as a type-level marker. The one
// method here (`apply`) is private and reached solely through `GemvBuilder`, so
// the crate-private bound leaks nothing; the lint fires on the visibility gap
// alone, which the seal is the whole point of.
#[allow(private_bounds)]
impl<E: FusedArm> Gemv<PrologueNone, E> {
    /// The sole constructor of the runtime fused GEMV op: dispatch `x @ w.t()`
    /// with epilogue `E` riding the f32 accumulator, plus the epilogue's
    /// operand `r` if it binds one.
    ///
    /// `x` is `[.., 1, k]`, `w` is `[n, k]` (the untransposed `Linear` layout),
    /// `r` is the [`Epilogue::Operand`] tensor (the residual) or `None`. The
    /// arm is `E::ARM`; the `[[host_name]]` it resolves ends in `E::SUFFIX`.
    fn apply(x: &Tensor, w: &Tensor, r: Option<&Tensor>) -> Result<Tensor> {
        super::ops::dispatch_gemv_fused(E::ARM, x, w, r)
    }
}

/// Start a fused-GEMV pipeline over weight `w`.
///
/// `w` is `[n, k]`, the untransposed [`super::Linear`] layout. Reads as a
/// pipeline: `gemv(w).then(Silu).eval(x)`. The verb at the seam names what
/// happens; `.fuse()` never appears.
pub fn gemv(w: &Tensor) -> GemvBuilder<'_> {
    GemvBuilder { w }
}

/// A GEMV awaiting a tail. Built by [`gemv`]; consumed by `.then()`/`.eval()`.
pub struct GemvBuilder<'w> {
    w: &'w Tensor,
}

impl<'w> GemvBuilder<'w> {
    /// Attach an output-side tail. If the tail is a fusable [`Epilogue`] the
    /// pipeline runs in one dispatch; if it is merely [`Elementwise`] (a legal
    /// transform that is not an admissible epilogue) it de-fuses to a separate
    /// dispatch. The caller's code is identical either way — an illegal fusion
    /// de-fuses, it does not fail to compile.
    pub fn then<T>(self, tail: T) -> Pipeline<'w, T> {
        Pipeline { w: self.w, tail }
    }

    /// Run the bare GEMV (no epilogue). Equivalent to `.then(EpilogueNone)`.
    pub fn eval(self, x: &Tensor) -> Result<Tensor> {
        self.then(EpilogueNone).eval(x)
    }

    /// Attach an input-side prologue. `Prologue` is sealed to `{PrologueNone}`
    /// today, so `.before(RmsNorm)` does not compile until a measured win opens
    /// the gate. Present so the surface reads `gemv(w).before(P).then(E)`.
    pub fn before<P: Prologue>(self, _prologue: P) -> Self {
        self
    }
}

/// A GEMV composed with a tail stage. The tail decides, by its type, whether
/// the pipeline fuses (one dispatch) or de-fuses (two dispatches).
///
/// Zero-sized but for the borrowed weight and the tail's own operands; it
/// monomorphizes to the same code the hand-written `Gemv<_, E>` produces.
pub struct Pipeline<'w, T> {
    w: &'w Tensor,
    tail: T,
}

/// How a tail runs after the GEMV. Blanket-implemented for every [`Epilogue`]
/// (the fused, one-dispatch path) and explicitly for the de-fuse tails (a
/// separate dispatch). There is no overlap: the de-fuse tails are not
/// `Epilogue`. This is the sealed seam that keeps `.then()` uniform while the
/// type picks fused-vs-separate.
pub trait EvalTail: sealed::Sealed {
    fn eval_after_gemv(&self, w: &Tensor, x: &Tensor) -> Result<Tensor>;
}

impl<'w, T: EvalTail> Pipeline<'w, T> {
    /// Run the pipeline against input `x`. `x` is `[.., 1, k]`.
    pub fn eval(self, x: &Tensor) -> Result<Tensor> {
        self.tail.eval_after_gemv(self.w, x)
    }
}

// --- Fused tails: one dispatch, the epilogue rides the GEMV's store. ---

impl sealed::Sealed for EpilogueNone {}
impl EvalTail for EpilogueNone {
    fn eval_after_gemv(&self, w: &Tensor, x: &Tensor) -> Result<Tensor> {
        // The bare GEMV — `x @ w.t()`, no epilogue.
        x.matmul(&w.t()?)
    }
}

impl sealed::Sealed for Silu {}
impl EvalTail for Silu {
    fn eval_after_gemv(&self, w: &Tensor, x: &Tensor) -> Result<Tensor> {
        Gemv::<PrologueNone, Silu>::apply(x, w, None)
    }
}

impl sealed::Sealed for BoundResidual<'_> {}
impl EvalTail for BoundResidual<'_> {
    fn eval_after_gemv(&self, w: &Tensor, x: &Tensor) -> Result<Tensor> {
        Gemv::<PrologueNone, Residual>::apply(x, w, Some(self.r))
    }
}

// --- De-fuse tails: a legal `Elementwise` that is NOT an admissible
// `Epilogue`, so it runs as GEMV then a separate dispatch. No overlap with the
// fused impls above: `Rope`/`QkNorm` are not `Epilogue`. ---

impl sealed::Sealed for Rope {}
impl EvalTail for Rope {
    fn eval_after_gemv(&self, w: &Tensor, x: &Tensor) -> Result<Tensor> {
        // De-fuses: GEMV, then RoPE separately. RoPE is not `Elementwise`
        // (`impl Epilogue for Rope` does not compile), so it can never ride the
        // store; the pipeline degrades to two dispatches rather than to a
        // compile error. The separate RoPE application is the caller's — this
        // tail runs the GEMV half and returns it for a following `rope` call,
        // matching the "same caller code, kernel-choice is whatever's legal"
        // contract (#602). A future measured RoPE epilogue would move this arm
        // into the fused set.
        x.matmul(&w.t()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The type-level epilogue set maps to exactly the three runtime
    /// `GemvFusion` arms `gemv.metal` instantiates. A new fusable epilogue that
    /// forgets its arm has no `FusedArm` impl and cannot be applied — this pins
    /// the mapping that does exist. Backend-independent.
    #[test]
    fn fused_arm_matches_runtime_enum() {
        assert_eq!(<EpilogueNone as FusedArm>::ARM, GemvFusion::Off);
        assert_eq!(<Silu as FusedArm>::ARM, GemvFusion::Silu);
        assert_eq!(<Residual as FusedArm>::ARM, GemvFusion::Residual);
    }

    /// SiLU has no second operand; the residual reads one at its output row.
    /// Backend-independent — the SUFFIX literals and associated types are type
    /// facts, checked here without a device.
    #[test]
    fn precision_operand_and_suffix_witnesses() {
        assert_eq!(Silu::PRECISION, Precision::Accumulator);
        assert_eq!(Residual::PRECISION, Precision::Accumulator);
        assert_eq!(EpilogueNone::PRECISION, Precision::Store);
        // The `[[host_name]]` suffixes, pinned to their exact strings (§3.1b
        // rule 1: absolute reference). The cross-crate reconciliation against
        // `GemvEpilogue::name_suffix` is in `suffix_matches_runtime_enum`,
        // which needs the metal-gated `kernels` module.
        assert_eq!(Silu::SUFFIX, "_silu");
        assert_eq!(Residual::SUFFIX, "_residual");
        assert_eq!(EpilogueNone::SUFFIX, "");
        // Type-level: `Silu::Operand == NoOperand`, `Residual::Operand ==
        // RowIndexed<f16>`. Named here so a change to either associated type is
        // a compile error at this assignment, not a silent kernel mismatch.
        let _: <Silu as Epilogue>::Operand = NoOperand;
        let _: <Residual as Epilogue>::Operand = RowIndexed::<half::f16>(PhantomData);
    }

    /// The typed `Epilogue::SUFFIX` equals the `[[host_name]]` suffix the
    /// runtime `GemvEpilogue` resolves, or a fused type would dispatch a kernel
    /// name that is not instantiated. This is the drift guard between the two
    /// enums; it needs the metal-gated `kernels` module.
    #[cfg(feature = "metal")]
    #[test]
    fn suffix_matches_runtime_enum() {
        use lloom_tensor::kernels::GemvEpilogue;
        assert_eq!(EpilogueNone::SUFFIX, GemvEpilogue::None.name_suffix());
        assert_eq!(Silu::SUFFIX, GemvEpilogue::Silu.name_suffix());
        assert_eq!(Residual::SUFFIX, GemvEpilogue::Residual.name_suffix());
    }
}
