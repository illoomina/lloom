//! Variant names for the kernels in `metal_src/reduce.metal`.
//!
//! The same coupling [`crate::kernels::ConvKernel`] documents for `conv.metal`, in the
//! file that carries the decode path's reductions: `reduce.metal` spells each
//! `[[host_name]]` in an instantiation, and callers spell it again to load a
//! pipeline. `candle-core`'s Metal backend has two `match (op, dtype)` tables
//! of string literals for the reductions, and `candle-nn` has five more for
//! softmax, the norms and RoPE. Nothing compared them with the Metal side, so a
//! rename or a dtype added to one side only failed at *runtime*.
//!
//! `reduce_names_resolve` in `tests.rs` loads every name declared here against
//! the compiled `reduce.metal`, so a disagreement is a test failure instead.
//!
//! Two shapes here that the conv registry did not have:
//!
//! * **`_strided` pairs.** Every reduction and arg-reduction is instantiated
//!   twice, contiguous and strided, from one `init_reduce` row. They are
//!   declared as separate families sharing a stem so that the spelling test
//!   still checks the full name, rather than treating the suffix as free-form.
//! * **A prefix family.** `rope`, `rope_i` and `rope_thd` come from one
//!   `init_rope` row but are three different stems, so each is its own family.
//!   Since #659 those three families compile into `rope/rope.metal`
//!   (`Source::Rope`), not `reduce.metal`; each family carries its own `source`
//!   so the resolution tests load it from the library that defines it.
//!
//! What is deliberately *not* here is the threadgroup-size axis. `reduce.metal`
//! instantiates each kernel for eleven block sizes, but all eleven live behind
//! one `[[host_name]]` and are selected inside the kernel by a switch on
//! `block_dim`. No name varies with it, so there is nothing for a name registry
//! to check; see the comment on `reduce_switch` in `reduce.metal` for why the
//! axis stays runtime-selected.

/// A kernel family in `reduce.metal` whose variants differ only by dtype.
///
/// Mirrors [`crate::kernels::ConvKernel`] deliberately, including the `tail` field, so
/// the two registries read the same way and a future consolidation is a
/// mechanical one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceKernel {
    /// The [`Source`] library this family's kernels compile into. Reductions,
    /// softmax and the norms are `Source::Reduce`; the three RoPE families are
    /// `Source::Rope` since #659 split them into `rope/rope.metal`. Carried as
    /// data so the resolution tests load each family from the library that
    /// actually defines it, rather than assuming one source for the registry.
    source: crate::kernels::Source,
    stem: &'static str,
    /// Trailing segment after the dtype suffix. Empty for the contiguous
    /// families, `"_strided"` for their strided counterparts.
    ///
    /// Kept explicit for the same reason `ConvKernel` keeps it: the spelling
    /// test checks `stem_dtype_tail` in full, so a strided row that resolved to
    /// the contiguous kernel — which reads a `strides` argument that was never
    /// bound — is caught rather than accommodated.
    tail: &'static str,
    /// `(dtype suffix, full `[[host_name]]`)`, one row per instantiation.
    ///
    /// Stored verbatim rather than formatted, so every string is greppable
    /// against `reduce.metal` and is covered by the resolution test.
    variants: &'static [(&'static str, &'static str)],
}

/// The dtypes every plain reduction is instantiated for, in `reduce.metal`'s
/// order. `i64` is behind `__METAL_VERSION__ >= 220` and `bf16` behind
/// `__HAVE_BFLOAT__`; both guards hold on every Metal 3 target we build for, and
/// the resolution test is what would catch it if that stopped being true.
macro_rules! reduce_variants {
    ($stem:literal, $tail:literal) => {
        &[
            ("f32", concat!($stem, "_f32", $tail)),
            ("f16", concat!($stem, "_f16", $tail)),
            ("bf16", concat!($stem, "_bf16", $tail)),
            ("u8", concat!($stem, "_u8", $tail)),
            ("u32", concat!($stem, "_u32", $tail)),
            ("i64", concat!($stem, "_i64", $tail)),
        ]
    };
}

/// Declares the contiguous and strided families for one reduction, which
/// `init_reduce` in `reduce.metal` emits as a pair.
macro_rules! reduce_family {
    ($contig:ident, $strided:ident, $stem:literal) => {
        pub const $contig: Self = Self {
            source: crate::kernels::Source::Reduce,
            stem: $stem,
            tail: "",
            variants: reduce_variants!($stem, ""),
        };

        pub const $strided: Self = Self {
            source: crate::kernels::Source::Reduce,
            stem: $stem,
            tail: "_strided",
            variants: reduce_variants!($stem, "_strided"),
        };
    };
}

impl ReduceKernel {
    reduce_family!(SUM, SUM_STRIDED, "fast_sum");
    reduce_family!(MUL, MUL_STRIDED, "fast_mul");
    reduce_family!(MIN, MIN_STRIDED, "fast_min");
    reduce_family!(MAX, MAX_STRIDED, "fast_max");
    reduce_family!(ARGMIN, ARGMIN_STRIDED, "fast_argmin");
    reduce_family!(ARGMAX, ARGMAX_STRIDED, "fast_argmax");

    /// Float-only: softmax and the norms are instantiated for the float dtypes
    /// alone, so an integer suffix must return `None` rather than a name that
    /// would fail to resolve.
    pub const SOFTMAX: Self = Self {
        source: crate::kernels::Source::Reduce,
        stem: "softmax",
        tail: "",
        variants: &[
            ("f32", "softmax_f32"),
            ("f16", "softmax_f16"),
            ("bf16", "softmax_bf16"),
        ],
    };

    pub const RMSNORM: Self = Self {
        source: crate::kernels::Source::Reduce,
        stem: "rmsnorm",
        tail: "",
        variants: &[
            ("f32", "rmsnorm_f32"),
            ("f16", "rmsnorm_f16"),
            ("bf16", "rmsnorm_bf16"),
        ],
    };

    pub const LAYERNORM: Self = Self {
        source: crate::kernels::Source::Reduce,
        stem: "layernorm",
        tail: "",
        variants: &[
            ("f32", "layernorm_f32"),
            ("f16", "layernorm_f16"),
            ("bf16", "layernorm_bf16"),
        ],
    };

    /// The three RoPE entry points. One `init_rope` row emits all three, but
    /// they are separate stems rather than one stem with a suffix, so each is
    /// its own family and `name()` stays a total lookup by dtype.
    pub const ROPE: Self = Self {
        source: crate::kernels::Source::Rope,
        stem: "rope",
        tail: "",
        variants: &[
            ("f32", "rope_f32"),
            ("f16", "rope_f16"),
            ("bf16", "rope_bf16"),
        ],
    };

    pub const ROPE_I: Self = Self {
        source: crate::kernels::Source::Rope,
        stem: "rope_i",
        tail: "",
        variants: &[
            ("f32", "rope_i_f32"),
            ("f16", "rope_i_f16"),
            ("bf16", "rope_i_bf16"),
        ],
    };

    pub const ROPE_THD: Self = Self {
        source: crate::kernels::Source::Rope,
        stem: "rope_thd",
        tail: "",
        variants: &[
            ("f32", "rope_thd_f32"),
            ("f16", "rope_thd_f16"),
            ("bf16", "rope_thd_bf16"),
        ],
    };

    /// Every family declared above. The resolution test iterates this, so a
    /// family added here is checked without touching the test.
    pub const ALL: &'static [Self] = &[
        Self::SUM,
        Self::SUM_STRIDED,
        Self::MUL,
        Self::MUL_STRIDED,
        Self::MIN,
        Self::MIN_STRIDED,
        Self::MAX,
        Self::MAX_STRIDED,
        Self::ARGMIN,
        Self::ARGMIN_STRIDED,
        Self::ARGMAX,
        Self::ARGMAX_STRIDED,
        Self::SOFTMAX,
        Self::RMSNORM,
        Self::LAYERNORM,
        Self::ROPE,
        Self::ROPE_I,
        Self::ROPE_THD,
    ];

    /// The family's name without a dtype suffix, for diagnostics.
    pub const fn stem(&self) -> &'static str {
        self.stem
    }

    /// The [`Source`](crate::kernels::Source) library this family loads from.
    /// `Source::Rope` for the three RoPE families (split into `rope/rope.metal`
    /// by #659), `Source::Reduce` for everything else in this registry.
    pub const fn source(&self) -> crate::kernels::Source {
        self.source
    }

    /// The segment following the dtype suffix — `"_strided"` for the strided
    /// families, empty otherwise.
    pub const fn tail(&self) -> &'static str {
        self.tail
    }

    /// The `[[host_name]]` string for this family at `dtype_suffix`, or `None`
    /// if `reduce.metal` does not instantiate that combination.
    ///
    /// Returning `None` rather than a formatted string is the point: an
    /// unsupported dtype is refused here, where the caller can report it
    /// against its own dtype enum, instead of reaching `load_pipeline` as a
    /// name that will fail to resolve.
    ///
    /// Linear over at most six entries and called once per op, not per dispatch
    /// element; `Kernels`'s pipeline cache is what the hot path hits.
    pub fn name(&self, dtype_suffix: &str) -> Option<&'static str> {
        self.variants
            .iter()
            .find(|(suffix, _)| *suffix == dtype_suffix)
            .map(|(_, name)| *name)
    }

    /// Every `(dtype suffix, `[[host_name]]`)` pair this family declares.
    pub fn variants(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.variants.iter().copied()
    }
}
