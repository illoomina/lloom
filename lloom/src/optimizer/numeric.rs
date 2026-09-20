//! The numerical-class lattice (`design/docs/optimizer-design.md` §6.1) and
//! the target fp model a class is stated under (§5a).
//!
//! Equivalence in floating point is not an equivalence relation: `a + (b + c)`
//! and `(a + b) + c` differ in f32, and chaining "within-tolerance" rewrites
//! drifts arbitrarily far. So every value in the IR carries a class, every
//! fragment carries the class its composition lands in, and a derivation is
//! admitted only if the class it lands in is within the policy's budget.
//! Crossing the budget is refused, never coerced.
//!
//! A class — and every verdict the offline validation records — is relative
//! to an fp semantics: whether products contract into an `fma`, what width a
//! reduction accumulates in, what shape its tree has, what happens to a
//! denormal. Those are declarations this tree already makes, kernel by
//! kernel (§5a: *the hardware does not need characterising — it needs
//! declaring*). [`TargetFpModel`] is that declaration as a value, and a
//! verdict recorded under one model does not transfer to another: a bound
//! derived for `simd_sum`'s tree says nothing about the serial `conv_fold`.

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

/// Whether `a * b + c` contracts into one rounding. The Metal compiler
/// contracts under every `MTLMathMode` and nothing in this tree pins it
/// (`DESIGN.md` §2.3.3b: a pin is a no-op, 0 of 4096 against a control that
/// moves 2938), so every kernel here is `Contract`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FmaContraction {
    /// The compiler may fuse a product into the following add.
    Contract,
    /// Contraction pinned off (a pragma or `-ffp-contract=off`). No kernel in
    /// this tree; the spelling exists so a recorded model can say so.
    Pinned,
}

/// The width a kernel accumulates a reduction in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Accum {
    /// `float`, whatever the store type.
    F32,
    /// The tensor's own dtype (`T`): at f16 the accumulator is f16.
    Store,
    /// An integer accumulator (issue #750, the M5 native-int arm): the reduction
    /// sums integer products into an integer register (`int32`), where addition
    /// is associative and exact, so any summation order gives the same result.
    /// This is [`NumericClass::BitExact`] by construction — the class this enum's
    /// F32/Store arms cannot reach because float addition is not associative.
    Int,
    /// No reduction — an elementwise kernel.
    None,
}

/// A dtype leg of a block's `(activation, accumulate, store)` triple (issue #750,
/// `DESIGN.md` §8.1.4). The triple is the type-level declaration a matmul/reduction
/// block makes; [`TargetFpModel`] carries it so a rewrite over the block is
/// modelled under the dtypes it actually runs, and a lossy triple is refused where
/// the block declares full precision mandatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Leg {
    /// 32-bit float.
    F32,
    /// 16-bit float (IEEE half).
    F16,
    /// bfloat16.
    BF16,
    /// A signed 8-bit integer operand (an activation or store leg on the M5 arm).
    I8,
    /// A signed 32-bit integer accumulator (the M5 arm's accumulate leg).
    I32,
}

impl Leg {
    /// Whether this leg is an integer type. An all-integer accumulate is exact
    /// and associative ([`Accum::Int`]); a float accumulate is neither.
    pub fn is_int(self) -> bool {
        matches!(self, Leg::I8 | Leg::I32)
    }

    /// The CLI/model spelling.
    pub fn name(self) -> &'static str {
        match self {
            Leg::F32 => "f32",
            Leg::F16 => "f16",
            Leg::BF16 => "bf16",
            Leg::I8 => "i8",
            Leg::I32 => "i32",
        }
    }
}

/// The `(activation, accumulate, store)` dtype triple a block declares (issue
/// #750). The accumulate leg is the one that decides the numeric class: an F32
/// accumulate is the §8.1.4 default (float, non-associative, [`NumericClass::
/// Rounding`] against a store-typed epilogue), an integer accumulate is exact
/// ([`NumericClass::BitExact`]), and a store-typed float accumulate is the
/// reduction-length-growing [`NumericClass::ReorderEquiv`] the fused arms avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DtypeTriple {
    pub act: Leg,
    pub acc: Leg,
    pub store: Leg,
}

impl DtypeTriple {
    /// The reference `(f16, f32, f16)` triple LFM2's dense blocks and the #750
    /// quantized f16 arm run under.
    pub const F16_F32_F16: DtypeTriple = DtypeTriple {
        act: Leg::F16,
        acc: Leg::F32,
        store: Leg::F16,
    };
    /// The `(f32, f32, f32)` triple — the historical quantized lock.
    pub const F32_F32_F32: DtypeTriple = DtypeTriple {
        act: Leg::F32,
        acc: Leg::F32,
        store: Leg::F32,
    };
    /// The M5 native-int `(i8, i32, i8)` triple. Registered as declared-but-
    /// unimplemented (issue #750, out of scope: no M5 hardware); its numeric
    /// class is [`NumericClass::BitExact`] because the accumulate is integer.
    pub const I8_I32_I8: DtypeTriple = DtypeTriple {
        act: Leg::I8,
        acc: Leg::I32,
        store: Leg::I8,
    };

    /// The [`Accum`] leg this triple's accumulate maps to. An integer accumulate
    /// is [`Accum::Int`]; an f32 accumulate is [`Accum::F32`] (the §8.1.4 default,
    /// whatever the store); a *narrower-than-f32* float accumulate (acc at the
    /// store width, and that width below f32) is [`Accum::Store`] — the
    /// reduction-length-growing arm the fused paths avoid.
    pub fn accum(self) -> Accum {
        if self.acc.is_int() {
            Accum::Int
        } else if self.acc == Leg::F32 {
            Accum::F32
        } else {
            // A float accumulate narrower than f32 (e.g. an f16 accumulator):
            // accumulates at the store width, the `ReorderEquiv` arm.
            Accum::Store
        }
    }

    /// The numeric class of a matmul/reduction under this triple, relative to a
    /// full-precision reference. An integer accumulate is bit-exact; an f32
    /// accumulate stored at a narrower float is [`NumericClass::Rounding`] (the
    /// store rounds, the accumulator does not, and the bound does not grow with
    /// the reduction length); an accumulate at the store width is
    /// [`NumericClass::ReorderEquiv`] (the bound grows with `K`).
    pub fn numeric_class(self) -> NumericClass {
        match self.accum() {
            Accum::Int => NumericClass::BitExact,
            Accum::F32 => {
                if self.store == Leg::F32 {
                    // Full-width throughout: same products, same roundings.
                    NumericClass::BitExact
                } else {
                    // Narrower float store on an f32 accumulator.
                    NumericClass::Rounding
                }
            }
            Accum::Store => NumericClass::ReorderEquiv,
            Accum::None => NumericClass::BitExact,
        }
    }

    /// Whether a block that declares full-precision-mandatory (`acc:f32`
    /// throughout, and never a narrower store that would round — issue #750's GDN
    /// `S`, the QSA indexer score, the N-gram hash) may run under this triple.
    /// Such a path refuses any triple whose numeric class is lossier than
    /// [`NumericClass::BitExact`]: a rounding or reorder-equiv arm is exactly what
    /// it declares it must not take, and an integer arm is a different computation.
    pub fn is_full_precision(self) -> bool {
        self.acc == Leg::F32 && self.store == Leg::F32 && !self.act.is_int()
    }
}

/// The shape of a kernel's reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionTree {
    /// No reduction.
    None,
    /// One thread, taps in the window's order.
    Serial,
    /// One thread, the ring's slots in slot order against the phase-permuted
    /// weight (`shortconv_ring`): the same products as [`Serial`](Self::Serial),
    /// summed from a different starting slot.
    SerialRotated,
    /// Per-lane serial partials, a `simd_shuffle_down` tree across the
    /// simdgroup, then a serial combine across simdgroups (`block_reducer`,
    /// the GEMV's tail, `simd_sum`).
    Simd,
}

/// What a denormal operand or result becomes, at one width. Measured on
/// this machine (`tests/optimizer_fp_model.rs`), it is a property of the
/// WIDTH and not of the math mode: f32 flushes under `Fast` and `Relaxed`
/// alike, f16 preserves under both — so the model carries one per width
/// where §5a's sketch carried one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Denormals {
    /// Preserved: a subnormal operand or result keeps its value.
    Preserve,
    /// Flushed to zero, as an operand and as a result.
    Flush,
}

/// The fp semantics a kernel computes under — the Kind-A environment a
/// numerical class is stated in (§5a, §3.2). Three of the five are
/// properties of the build (`fma`, the two denormal modes) and two of the
/// kernel (`accum`, `tree`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetFpModel {
    pub fma: FmaContraction,
    pub accum: Accum,
    pub tree: ReductionTree,
    /// Denormals at f32 — the accumulator width of every reduction here.
    pub denormals_f32: Denormals,
    /// Denormals at f16 — the store width the model decodes at.
    pub denormals_f16: Denormals,
}

impl TargetFpModel {
    /// The model of a kernel FAMILY under this build, or `None` for a family
    /// this table has not read. Declared from the kernel source, cited per
    /// row; a family missing here is unvalidated by construction, never
    /// guessed.
    ///
    /// | family | source |
    /// |---|---|
    /// | `gemv`, `gemv_silu`, `gemv_residual` | `gemv.metal`: `AccT = float` (`GemvDefaultAccT`), per-lane serial over `K`, `simd_shuffle_down` over `SN`, serial over `BN` simdgroups; the epilogue reads the accumulator |
    /// | `shortconv_step`, `conv1d_depthwise` | `conv.metal`: `A = float`, taps in window order, one thread |
    /// | `shortconv_ring` | `conv.metal` `shortconv_ring_body`: `A = float`, slots in slot order against the phase-permuted weight, the loop pinned serial |
    /// | `fast_sum` | `reduce.metal` `init_reduce`: `R = T` — the store type — through `block_reducer`'s shuffle tree |
    /// | `rmsnorm`, `rmsnorm_rope` | `reduce.metal` `rms_norm_case`: `RMS<float>` through `block_reducer` |
    /// | `sdpa_vector` | `scaled_dot_product_attention.metal`: `U = float`, `simd_sum` |
    /// | `unary`, `silu`, `bmul`, `badd`, `copy2d`, `dequantize_kv` | elementwise at `T`; no reduction |
    pub fn of_family(family: &str) -> Option<TargetFpModel> {
        let (accum, tree) = match family {
            "gemv" | "gemv_silu" | "gemv_residual" => (Accum::F32, ReductionTree::Simd),
            "shortconv_step" | "conv1d_depthwise" => (Accum::F32, ReductionTree::Serial),
            "shortconv_ring" => (Accum::F32, ReductionTree::SerialRotated),
            "fast_sum" => (Accum::Store, ReductionTree::Simd),
            "rmsnorm" | "rmsnorm_rope" => (Accum::F32, ReductionTree::Simd),
            "sdpa_vector" => (Accum::F32, ReductionTree::Simd),
            "unary" | "silu" | "bmul" | "badd" | "copy2d" | "dequantize_kv" => {
                (Accum::None, ReductionTree::None)
            }
            _ => return None,
        };
        let build = build_model();
        Some(TargetFpModel {
            fma: build.fma,
            accum,
            tree,
            denormals_f32: build.denormals_f32,
            denormals_f16: build.denormals_f16,
        })
    }

    /// Parse the spelling [`Display`](fmt::Display) writes:
    /// `fma=<>,acc=<>,tree=<>,denorm32=<>,denorm16=<>`, one token, every
    /// field required.
    pub fn parse(s: &str) -> Result<TargetFpModel, String> {
        let mut fma = None;
        let mut accum = None;
        let mut tree = None;
        let mut denormals_f32 = None;
        let mut denormals_f16 = None;
        let denorm = |v: &str| -> Result<Denormals, String> {
            match v {
                "preserve" => Ok(Denormals::Preserve),
                "flush" => Ok(Denormals::Flush),
                other => Err(format!("denorm must be preserve|flush, got `{other}`")),
            }
        };
        for field in s.split(',') {
            let (k, v) = field
                .split_once('=')
                .ok_or_else(|| format!("fp model field `{field}` is not `key=value`"))?;
            match k {
                "fma" => {
                    fma = Some(match v {
                        "contract" => FmaContraction::Contract,
                        "pinned" => FmaContraction::Pinned,
                        other => return Err(format!("fma must be contract|pinned, got `{other}`")),
                    })
                }
                "acc" => {
                    accum = Some(match v {
                        "f32" => Accum::F32,
                        "store" => Accum::Store,
                        "int" => Accum::Int,
                        "none" => Accum::None,
                        other => {
                            return Err(format!("acc must be f32|store|int|none, got `{other}`"))
                        }
                    })
                }
                "tree" => {
                    tree = Some(match v {
                        "none" => ReductionTree::None,
                        "serial" => ReductionTree::Serial,
                        "serial-rot" => ReductionTree::SerialRotated,
                        "simd" => ReductionTree::Simd,
                        other => {
                            return Err(format!(
                                "tree must be none|serial|serial-rot|simd, got `{other}`"
                            ))
                        }
                    })
                }
                "denorm32" => denormals_f32 = Some(denorm(v)?),
                "denorm16" => denormals_f16 = Some(denorm(v)?),
                other => return Err(format!("unknown fp model field `{other}`")),
            }
        }
        match (fma, accum, tree, denormals_f32, denormals_f16) {
            (Some(fma), Some(accum), Some(tree), Some(denormals_f32), Some(denormals_f16)) => {
                Ok(TargetFpModel {
                    fma,
                    accum,
                    tree,
                    denormals_f32,
                    denormals_f16,
                })
            }
            _ => Err(format!(
                "fp model `{s}` must carry fma, acc, tree, denorm32 and denorm16"
            )),
        }
    }
}

fn denorm_name(d: Denormals) -> &'static str {
    match d {
        Denormals::Preserve => "preserve",
        Denormals::Flush => "flush",
    }
}

impl fmt::Display for TargetFpModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fma={},acc={},tree={},denorm32={},denorm16={}",
            match self.fma {
                FmaContraction::Contract => "contract",
                FmaContraction::Pinned => "pinned",
            },
            match self.accum {
                Accum::F32 => "f32",
                Accum::Store => "store",
                Accum::Int => "int",
                Accum::None => "none",
            },
            match self.tree {
                ReductionTree::None => "none",
                ReductionTree::Serial => "serial",
                ReductionTree::SerialRotated => "serial-rot",
                ReductionTree::Simd => "simd",
            },
            denorm_name(self.denormals_f32),
            denorm_name(self.denormals_f16)
        )
    }
}

/// Whether this process compiles its kernels under fast math (Metal's own
/// default, `DESIGN.md` §2.3.9). Without the `metal` feature there are no
/// Metal kernels; the declaration then describes the build the exclusion set
/// was recorded under, which is the fast arm every recorded digest belongs to.
fn fast_math() -> bool {
    #[cfg(feature = "metal")]
    {
        lloom_tensor::kernels::fast_math_enabled()
    }
    #[cfg(not(feature = "metal"))]
    {
        true
    }
}

/// The build-wide half of the model: what every kernel in this process is
/// compiled under. `accum` and `tree` are meaningless here (`None`); read
/// [`TargetFpModel::of_family`] for a kernel's.
///
/// `fma` is `Contract` under every math mode (§2.3.3b). The denormal modes
/// are what `tests/optimizer_fp_model.rs` measures on this machine: f32
/// flushed and f16 preserved, under `Fast` and under `Relaxed` alike — a
/// probe asserted against this declaration, so a toolchain or a machine that
/// changes it turns that test red rather than silently moving the model
/// every recorded verdict belongs to. Neither depends on the math mode, so
/// the mode itself is reported ([`build_math_mode`]) rather than keyed.
pub fn build_model() -> TargetFpModel {
    TargetFpModel {
        fma: FmaContraction::Contract,
        accum: Accum::None,
        tree: ReductionTree::None,
        denormals_f32: Denormals::Flush,
        denormals_f16: Denormals::Preserve,
    }
}

/// The name of the build's math mode, for a report.
pub fn build_math_mode() -> &'static str {
    if fast_math() {
        "fast"
    } else {
        "relaxed"
    }
}

/// A region of the decode step, by whether its values are carried across
/// tokens (§5a: the real boundary is the recurrence). A per-rewrite verdict
/// composes over a straight-line region by addition; across a recurrence it
/// composes only with the recurrence's gain known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    /// Straight-line: the value lives and dies within the token.
    Straight,
    /// The short-conv state of conv layer `l`: a window of the last `k − 1`
    /// `b·x` values, shifted or ring-written per token and read by the fold.
    /// The fold's result never re-enters the window, so the horizon is `k − 1`
    /// tokens by construction; the fold's gain on a discrepancy in the window
    /// is a norm of the layer's conv weights, known at load.
    ConvRing(usize),
    /// The KV cache of attention layer `l`: appended per token, read over
    /// `kv_len` by every later token. Data-dependent gain, growing length —
    /// the case §5a can only mark, not bound.
    Kv(usize),
}

impl Region {
    pub fn is_stateful(self) -> bool {
        !matches!(self, Region::Straight)
    }

    /// The region's kind, without its layer: the spelling the exclusion set
    /// records a transform under.
    pub fn kind(self) -> RegionKind {
        match self {
            Region::Straight => RegionKind::Straight,
            Region::ConvRing(_) => RegionKind::ConvRing,
            Region::Kv(_) => RegionKind::Kv,
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Region::Straight => f.write_str("straight"),
            Region::ConvRing(l) => write!(f, "conv-ring@{l}"),
            Region::Kv(l) => write!(f, "kv@{l}"),
        }
    }
}

/// A [`Region`] without its layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionKind {
    Straight,
    ConvRing,
    Kv,
}

impl RegionKind {
    pub fn name(self) -> &'static str {
        match self {
            RegionKind::Straight => "straight",
            RegionKind::ConvRing => "conv-ring",
            RegionKind::Kv => "kv",
        }
    }

    pub fn parse(s: &str) -> Result<RegionKind, String> {
        match s {
            "straight" => Ok(RegionKind::Straight),
            "conv-ring" => Ok(RegionKind::ConvRing),
            "kv" => Ok(RegionKind::Kv),
            other => Err(format!(
                "region must be straight|conv-ring|kv, got `{other}`"
            )),
        }
    }

    pub fn is_stateful(self) -> bool {
        !matches!(self, RegionKind::Straight)
    }
}

impl fmt::Display for RegionKind {
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

    /// The model spelling round-trips, every field is required — a recorded
    /// model missing one is the retrofit §5a prices as expensive — and the
    /// GEMV's tree is not the conv fold's.
    #[test]
    fn the_fp_model_spelling_round_trips_and_needs_every_field() {
        let m = TargetFpModel::of_family("gemv").unwrap();
        assert_eq!(TargetFpModel::parse(&m.to_string()), Ok(m));
        assert_eq!(m.accum, Accum::F32);
        assert_eq!(m.tree, ReductionTree::Simd);
        assert_eq!(m.fma, FmaContraction::Contract);
        let c = TargetFpModel::of_family("shortconv_step").unwrap();
        assert_eq!(c.tree, ReductionTree::Serial);
        assert_ne!(m, c, "the GEMV's tree is not the conv fold's");
        assert_eq!(
            TargetFpModel::of_family("shortconv_ring").unwrap().tree,
            ReductionTree::SerialRotated
        );
        assert_eq!(
            TargetFpModel::of_family("fast_sum").unwrap().accum,
            Accum::Store,
            "the unfused fold accumulates at the store type"
        );
        assert_eq!(TargetFpModel::of_family("not-a-kernel"), None);
        assert!(TargetFpModel::parse("fma=contract,acc=f32,tree=simd,denorm32=flush").is_err());
        assert!(TargetFpModel::parse(
            "fma=contract,acc=f32,tree=simd,denorm32=flush,denorm16=maybe"
        )
        .is_err());
        assert!(TargetFpModel::parse(
            "fma=contract,acc=f32,tree=simd,denorm32=flush,denorm16=preserve,x=1"
        )
        .is_err());
        assert_eq!(
            m.to_string(),
            "fma=contract,acc=f32,tree=simd,denorm32=flush,denorm16=preserve"
        );
    }

    /// Issue #750: the `(activation, accumulate, store)` triple is modelable —
    /// every declared triple maps to a numeric class — and an fp-mandatory path
    /// refuses a lossy triple.
    #[test]
    fn a_dtype_triple_is_modelable_and_an_fp_mandatory_path_refuses_a_lossy_one() {
        // The reference f16 arm: f32 accumulate, f16 store -> Rounding (the store
        // rounds, the accumulator does not; the bound does not grow with K).
        let f16 = DtypeTriple::F16_F32_F16;
        assert_eq!(f16.accum(), Accum::F32);
        assert_eq!(f16.numeric_class(), NumericClass::Rounding);

        // The historical lock: full float width throughout -> BitExact.
        let f32 = DtypeTriple::F32_F32_F32;
        assert_eq!(f32.accum(), Accum::F32);
        assert_eq!(f32.numeric_class(), NumericClass::BitExact);

        // The M5 native-int arm: integer accumulate is exact and associative ->
        // BitExact, the class no float accumulate can reach (issue #750).
        let int = DtypeTriple::I8_I32_I8;
        assert_eq!(int.accum(), Accum::Int);
        assert_eq!(int.numeric_class(), NumericClass::BitExact);
        assert_eq!(int.acc.name(), "i32");

        // A triple-declared fusion is modelable within a budget iff its class is
        // within that budget — the same `within` the rewrite admission uses.
        assert!(f16.numeric_class().within(NumericClass::Rounding));
        assert!(!f16.numeric_class().within(NumericClass::BitExact));
        assert!(int.numeric_class().within(NumericClass::BitExact));

        // An fp-mandatory path (GDN `S`, QSA indexer score, N-gram hash: acc:f32
        // throughout) refuses any lossy triple. Only the full-precision float
        // triple is accepted; the f16-store arm rounds and the int arm is a
        // different computation, so both are refused.
        assert!(f32.is_full_precision());
        assert!(!f16.is_full_precision(), "f16 store rounds -> refused");
        assert!(!int.is_full_precision(), "int is a different computation");

        // The `Accum::Int` leg round-trips through the model spelling, so a
        // recorded integer-accumulate model parses back to itself.
        let m = TargetFpModel {
            fma: FmaContraction::Contract,
            accum: Accum::Int,
            tree: ReductionTree::Simd,
            denormals_f32: Denormals::Flush,
            denormals_f16: Denormals::Preserve,
        };
        assert_eq!(TargetFpModel::parse(&m.to_string()), Ok(m));
        assert!(m.to_string().contains("acc=int"));
    }

    #[test]
    fn regions_spell_their_kind() {
        assert_eq!(Region::ConvRing(3).to_string(), "conv-ring@3");
        assert_eq!(Region::Kv(7).kind(), RegionKind::Kv);
        assert!(!Region::Straight.is_stateful());
        assert!(Region::ConvRing(0).is_stateful());
        for k in [RegionKind::Straight, RegionKind::ConvRing, RegionKind::Kv] {
            assert_eq!(RegionKind::parse(k.name()), Ok(k));
        }
        assert!(RegionKind::parse("ring").is_err());
    }
}
