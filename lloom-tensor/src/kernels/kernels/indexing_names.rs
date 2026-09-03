//! Variant names for the kernels in `metal_src/indexing.metal`.
//!
//! The same coupling [`crate::kernels::ConvKernel`] and [`crate::kernels::ReduceKernel`] document
//! for their files, in the last decode-path family that lacked a registry.
//! `indexing.metal` spells each `[[host_name]]` in an instantiation row, and
//! `candle-core`'s Metal backend spelled all 72 of them again in five
//! `match (ids.dtype, self.dtype)` tables of string literals. Nothing compared
//! the two, so a rename or a dtype added to one side only failed at *runtime*.
//!
//! **It had already failed.** `candle-core` named `is_i64_u8` and `is_i64_u32`;
//! `indexing.metal` declared neither, so `index_select` on a `U8` or `U32`
//! tensor with `I64` indices was a `LoadFunctionError` from inside a forward
//! pass. That is the fourth firing of the absent-variant class `DESIGN.md`
//! §8.1b tracks, after #26's 48 absent reduce variants and `conv`'s — and the
//! first found by cross-checking *across a crate boundary*, which is why it
//! survived three previous registry passes.
//!
//! `indexing_names_resolve` in `tests.rs` loads every name declared here against
//! the compiled `indexing.metal`, so a disagreement is a test failure instead of
//! a runtime one.
//!
//! # The key is a pair, and that is what is new here
//!
//! `ConvKernel` and `ReduceKernel` vary over one dtype, so a family is a list of
//! `(suffix, name)` and lookup takes a suffix. Indexing varies over **two**
//! independent dtypes — the index type and the value type — and the name
//! carries both: `is_u32_f16` is `index` over an f16 tensor with u32 indices.
//! So a family here is a list of `((index suffix, value suffix), name)` and
//! lookup takes the pair.
//!
//! That is a genuine structural difference rather than a stylistic one, and it
//! is why this is a third registry instead of a row in an existing one. The
//! `stem`/`tail` shape is kept identical to the other two so a future
//! consolidation stays mechanical.
//!
//! # Why the registry moved down rather than a suffix threading up
//!
//! `indexing`'s names were chosen a crate up, in
//! `candle-core/src/metal_backend/mod.rs`, where every other family names its
//! kernels inside `candle-metal-kernels`. Two ways to close that, and this file
//! is the first:
//!
//! * **Registry moves down** — the names are declared here and `candle-core`
//!   looks them up, exactly as it already does for `conv` via
//!   `conv_kernel_name`. The resolution test can then load every declared name
//!   against the compiled library, which is the check that catches the absent
//!   variants.
//! * **Suffix threads up** — `candle-core` learns to append `_packed`, which is
//!   what a *binding-style* axis would need.
//!
//! The second is only required once this family carries packed siblings, and it
//! is deliberately not done here: a binding-style change and a name-table change
//! have different failure modes — a wrong template substitution changes
//! numerics, a wrong binding silently corrupts slots — and landing them together
//! means a bisect cannot tell them apart. Keying on the dtype *suffix* strings
//! both crates already spell (`DType::as_str`) is what lets the registry stay
//! self-contained here, per `ConvKernel`'s reasoning: `candle-core`'s `DType`
//! has 14 variants against this crate's 6, and a cross-crate conversion would be
//! lossy in one direction.

/// A kernel family in `indexing.metal` whose variants differ by a
/// `(index dtype, value dtype)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexingKernel {
    stem: &'static str,
    /// `((index suffix, value suffix), full `[[host_name]]`)`, one row per
    /// instantiation in `indexing.metal`.
    ///
    /// Full names are stored verbatim rather than formatted from the stem so
    /// that every string this table can hand out is greppable against
    /// `indexing.metal` and is covered by the resolution test.
    variants: &'static [((&'static str, &'static str), &'static str)],
}

/// The `(index, value)` grid shared by `index_select` and `index_add`: three
/// index types over six value types.
macro_rules! full_grid {
    ($stem:literal) => {
        &[
            (("i64", "i64"), concat!($stem, "_i64_i64")),
            (("i64", "f32"), concat!($stem, "_i64_f32")),
            (("i64", "f16"), concat!($stem, "_i64_f16")),
            (("i64", "bf16"), concat!($stem, "_i64_bf16")),
            (("i64", "u8"), concat!($stem, "_i64_u8")),
            (("i64", "u32"), concat!($stem, "_i64_u32")),
            (("u32", "i64"), concat!($stem, "_u32_i64")),
            (("u32", "f32"), concat!($stem, "_u32_f32")),
            (("u32", "f16"), concat!($stem, "_u32_f16")),
            (("u32", "bf16"), concat!($stem, "_u32_bf16")),
            (("u32", "u8"), concat!($stem, "_u32_u8")),
            (("u32", "u32"), concat!($stem, "_u32_u32")),
            (("u8", "i64"), concat!($stem, "_u8_i64")),
            (("u8", "f32"), concat!($stem, "_u8_f32")),
            (("u8", "f16"), concat!($stem, "_u8_f16")),
            (("u8", "bf16"), concat!($stem, "_u8_bf16")),
            (("u8", "u8"), concat!($stem, "_u8_u8")),
            (("u8", "u32"), concat!($stem, "_u8_u32")),
        ]
    };
}

/// The `(index, value)` grid shared by `scatter` and `scatter_add`: float value
/// types for all three index types, plus `u32` values for `u32` indices only.
///
/// Not a sub-grid of [`full_grid`] — `indexing.metal` declares exactly these
/// ten, and the asymmetry (`sa_u32_u32` exists, `sa_u8_u32` does not) is the
/// file's, not a simplification here.
macro_rules! scatter_grid {
    ($stem:literal) => {
        &[
            (("u32", "f32"), concat!($stem, "_u32_f32")),
            (("u8", "f32"), concat!($stem, "_u8_f32")),
            (("i64", "f32"), concat!($stem, "_i64_f32")),
            (("u32", "u32"), concat!($stem, "_u32_u32")),
            (("u32", "f16"), concat!($stem, "_u32_f16")),
            (("u8", "f16"), concat!($stem, "_u8_f16")),
            (("i64", "f16"), concat!($stem, "_i64_f16")),
            (("u32", "bf16"), concat!($stem, "_u32_bf16")),
            (("u8", "bf16"), concat!($stem, "_u8_bf16")),
            (("i64", "bf16"), concat!($stem, "_i64_bf16")),
        ]
    };
}

impl IndexingKernel {
    /// `index_select`. `is_u32_f16` is the LFM2 embedding lookup — the one
    /// kernel in this file on the decode path (`DESIGN.md` §11.3h), one
    /// dispatch per token.
    ///
    /// `is_i64_u8` and `is_i64_u32` are the two that `indexing.metal` did not
    /// declare until this change.
    pub const INDEX_SELECT: Self = Self {
        stem: "is",
        variants: full_grid!("is"),
    };

    /// `index_add`.
    pub const INDEX_ADD: Self = Self {
        stem: "ia",
        variants: full_grid!("ia"),
    };

    /// `gather`. Sixteen rather than eighteen: `indexing.metal` declares no
    /// `gather_u32_u8` or `gather_i64_u8`.
    pub const GATHER: Self = Self {
        stem: "gather",
        variants: &[
            (("u8", "f32"), "gather_u8_f32"),
            (("u8", "f16"), "gather_u8_f16"),
            (("u8", "bf16"), "gather_u8_bf16"),
            (("u8", "u8"), "gather_u8_u8"),
            (("u8", "i64"), "gather_u8_i64"),
            (("u8", "u32"), "gather_u8_u32"),
            (("i64", "f32"), "gather_i64_f32"),
            (("i64", "f16"), "gather_i64_f16"),
            (("i64", "bf16"), "gather_i64_bf16"),
            (("i64", "u32"), "gather_i64_u32"),
            (("i64", "i64"), "gather_i64_i64"),
            (("u32", "f32"), "gather_u32_f32"),
            (("u32", "f16"), "gather_u32_f16"),
            (("u32", "bf16"), "gather_u32_bf16"),
            (("u32", "u32"), "gather_u32_u32"),
            (("u32", "i64"), "gather_u32_i64"),
        ],
    };

    /// `scatter`.
    pub const SCATTER: Self = Self {
        stem: "s",
        variants: scatter_grid!("s"),
    };

    /// `scatter_add`.
    pub const SCATTER_ADD: Self = Self {
        stem: "sa",
        variants: scatter_grid!("sa"),
    };

    /// Every family declared above. The resolution test iterates this, so a
    /// family added here is checked without touching the test.
    pub const ALL: &'static [Self] = &[
        Self::INDEX_SELECT,
        Self::INDEX_ADD,
        Self::GATHER,
        Self::SCATTER,
        Self::SCATTER_ADD,
    ];

    /// The family's name without dtype suffixes, for diagnostics.
    pub const fn stem(&self) -> &'static str {
        self.stem
    }

    /// The `[[host_name]]` for this family at `(index_suffix, value_suffix)`, or
    /// `None` if `indexing.metal` does not instantiate that combination.
    ///
    /// Both suffixes are the spelling both crates already use — `candle-core`'s
    /// `DType::as_str`. Returning `None` rather than a formatted string is the
    /// point: an unsupported pair is refused here, where the caller can report
    /// it against its own dtype enum, and cannot reach `load_pipeline` as a name
    /// that will fail to resolve. That is precisely the failure `is_i64_u8`
    /// produced.
    ///
    /// Linear over at most 18 entries and called once per op, not per dispatch
    /// element; the pipeline cache in `Kernels` is what the hot path actually
    /// hits, so `DESIGN.md` §15.2 #10 is not engaged.
    pub fn name(&self, index_suffix: &str, value_suffix: &str) -> Option<&'static str> {
        self.variants
            .iter()
            .find(|((i, v), _)| *i == index_suffix && *v == value_suffix)
            .map(|(_, name)| *name)
    }

    /// Every `((index suffix, value suffix), `[[host_name]]`)` this family
    /// declares.
    pub fn variants(
        &self,
    ) -> impl Iterator<Item = ((&'static str, &'static str), &'static str)> + '_ {
        self.variants.iter().copied()
    }
}
