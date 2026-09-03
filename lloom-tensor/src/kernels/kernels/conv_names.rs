//! Variant names for the kernels in `metal_src/conv.metal`.
//!
//! The Metal side instantiates each kernel per dtype via macro expansion, and
//! callers have to name the resulting `[[host_name]]` string to load a
//! pipeline. Those two lists were hand-synced: `conv.metal` spells the names in
//! macro arguments, and `candle-core`'s Metal backend spelled them again in ten
//! separate `match dtype` tables. Nothing checked that the spellings agreed, so
//! a rename or a newly-supported dtype on one side failed at *runtime* — a
//! `LoadFunctionError` from deep inside a forward pass, or, where a caller had
//! a fallback path, a silent switch to the slow one.
//!
//! This table is the single place the conv family's names are written on the
//! Rust side. `conv_names_resolve` in `tests.rs` loads every name in it against
//! the compiled Metal library, so a disagreement between this table and
//! `conv.metal` is a test failure rather than a runtime one.
//!
//! Deliberately a plain table rather than generated code: `conv.metal` is
//! compiled at *runtime* by `new_library_with_source` (there is no build step
//! emitting a metallib), so the compiled library is available to a test and is
//! a stronger oracle than a generator — it checks the name against what the GPU
//! will actually be asked for, not against what a manifest said to emit.
//!
//! Lookup is keyed on the dtype *suffix* (`"f32"`, `"bf16"`, …) rather than on
//! an enum. `candle-core`'s `DType` has 14 variants against this crate's 6, and
//! the conv kernels are instantiated for 5 of them; keying on the suffix both
//! crates already spell lets the registry stay self-contained here instead of
//! requiring a lossy conversion between two enums that do not correspond.

/// A kernel family in `conv.metal` whose variants differ only by dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvKernel {
    stem: &'static str,
    /// Trailing segment after the dtype suffix, for families carrying a
    /// compile-tier axis in the name. Empty for the plain `<stem>_<dtype>`
    /// families; `"_k3"` and friends for the `k_size`-specialized depthwise
    /// variants, per `DESIGN.md` §7.4's `<op>_<dtype>_k<K>` convention.
    ///
    /// Kept as an explicit field rather than folded into the verbatim names so
    /// that `conv_names_match_their_stem_and_suffix` still checks the full
    /// spelling rule and does not have to be weakened to accommodate a longer
    /// name.
    tail: &'static str,
    /// `(dtype suffix, full `[[host_name]]`)`, one row per instantiation in
    /// `conv.metal`.
    ///
    /// Full names are stored verbatim rather than formatted from the stem so
    /// that every string this table can hand out is greppable against
    /// `conv.metal` and is covered by the resolution test.
    variants: &'static [(&'static str, &'static str)],
}

impl ConvKernel {
    pub const IM2COL1D: Self = Self {
        stem: "im2col1d",
        tail: "",
        variants: &[
            ("f32", "im2col1d_f32"),
            ("f16", "im2col1d_f16"),
            ("bf16", "im2col1d_bf16"),
            ("u8", "im2col1d_u8"),
            ("u32", "im2col1d_u32"),
        ],
    };

    pub const IM2COL: Self = Self {
        stem: "im2col",
        tail: "",
        variants: &[
            ("f32", "im2col_f32"),
            ("f16", "im2col_f16"),
            ("bf16", "im2col_bf16"),
            ("u8", "im2col_u8"),
            ("u32", "im2col_u32"),
        ],
    };

    pub const COL2IM1D: Self = Self {
        stem: "col2im1d",
        tail: "",
        variants: &[
            ("f32", "col2im1d_f32"),
            ("f16", "col2im1d_f16"),
            ("bf16", "col2im1d_bf16"),
            ("u8", "col2im1d_u8"),
            ("u32", "col2im1d_u32"),
        ],
    };

    /// Float-only on the Metal side: the fused depthwise kernel accumulates in
    /// `float`, which the integer instantiations would not do meaningfully.
    pub const CONV1D_DEPTHWISE: Self = Self {
        stem: "conv1d_depthwise",
        tail: "",
        variants: &[
            ("f32", "conv1d_depthwise_f32"),
            ("f16", "conv1d_depthwise_f16"),
            ("bf16", "conv1d_depthwise_bf16"),
        ],
    };

    /// The `k_size`-specialized depthwise variants, one family per compile-time
    /// `K`. Separate constants rather than one family because the `(dtype, K)`
    /// pair is the key: `name()` looks up by dtype suffix alone, so each `K` is
    /// its own family for that lookup to stay total.
    ///
    /// A `K` absent from this set is not an error — `conv1d_depthwise_k` refuses
    /// and the caller uses the generic kernel — so the set is a performance
    /// choice, never a correctness one.
    pub const CONV1D_DEPTHWISE_K2: Self = Self {
        stem: "conv1d_depthwise",
        tail: "_k2",
        variants: &[
            ("f32", "conv1d_depthwise_f32_k2"),
            ("f16", "conv1d_depthwise_f16_k2"),
            ("bf16", "conv1d_depthwise_bf16_k2"),
        ],
    };

    pub const CONV1D_DEPTHWISE_K3: Self = Self {
        stem: "conv1d_depthwise",
        tail: "_k3",
        variants: &[
            ("f32", "conv1d_depthwise_f32_k3"),
            ("f16", "conv1d_depthwise_f16_k3"),
            ("bf16", "conv1d_depthwise_bf16_k3"),
        ],
    };

    pub const CONV1D_DEPTHWISE_K4: Self = Self {
        stem: "conv1d_depthwise",
        tail: "_k4",
        variants: &[
            ("f32", "conv1d_depthwise_f32_k4"),
            ("f16", "conv1d_depthwise_f16_k4"),
            ("bf16", "conv1d_depthwise_bf16_k4"),
        ],
    };

    /// The specialized depthwise family for a compile-time `k_size`, or `None`
    /// if no variant is instantiated for that `k`. `None` means "use the generic
    /// kernel", not "unsupported".
    pub fn conv1d_depthwise_k(k_size: usize) -> Option<Self> {
        match k_size {
            2 => Some(Self::CONV1D_DEPTHWISE_K2),
            3 => Some(Self::CONV1D_DEPTHWISE_K3),
            4 => Some(Self::CONV1D_DEPTHWISE_K4),
            _ => None,
        }
    }

    pub const CONV_TRANSPOSE1D: Self = Self {
        stem: "conv_transpose1d",
        tail: "",
        variants: &[
            ("f32", "conv_transpose1d_f32"),
            ("f16", "conv_transpose1d_f16"),
            ("bf16", "conv_transpose1d_bf16"),
            ("u8", "conv_transpose1d_u8"),
            ("u32", "conv_transpose1d_u32"),
        ],
    };

    /// Float-only: `conv.metal` declares no integer `CONVT2D_OP`.
    pub const CONV_TRANSPOSE2D: Self = Self {
        stem: "conv_transpose2d",
        tail: "",
        variants: &[
            ("f32", "conv_transpose2d_f32"),
            ("f16", "conv_transpose2d_f16"),
            ("bf16", "conv_transpose2d_bf16"),
        ],
    };

    pub const UPSAMPLE_NEAREST2D: Self = Self {
        stem: "upsample_nearest2d",
        tail: "",
        variants: &[
            ("f32", "upsample_nearest2d_f32"),
            ("f16", "upsample_nearest2d_f16"),
            ("bf16", "upsample_nearest2d_bf16"),
            ("u8", "upsample_nearest2d_u8"),
            ("u32", "upsample_nearest2d_u32"),
        ],
    };

    pub const UPSAMPLE_BILINEAR2D: Self = Self {
        stem: "upsample_bilinear2d",
        tail: "",
        variants: &[
            ("f32", "upsample_bilinear2d_f32"),
            ("f16", "upsample_bilinear2d_f16"),
            ("bf16", "upsample_bilinear2d_bf16"),
            ("u8", "upsample_bilinear2d_u8"),
            ("u32", "upsample_bilinear2d_u32"),
        ],
    };

    pub const MAX_POOL2D: Self = Self {
        stem: "max_pool2d",
        tail: "",
        variants: &[
            ("f32", "max_pool2d_f32"),
            ("f16", "max_pool2d_f16"),
            ("bf16", "max_pool2d_bf16"),
            ("u8", "max_pool2d_u8"),
            ("u32", "max_pool2d_u32"),
        ],
    };

    pub const AVG_POOL2D: Self = Self {
        stem: "avg_pool2d",
        tail: "",
        variants: &[
            ("f32", "avg_pool2d_f32"),
            ("f16", "avg_pool2d_f16"),
            ("bf16", "avg_pool2d_bf16"),
            ("u8", "avg_pool2d_u8"),
            ("u32", "avg_pool2d_u32"),
        ],
    };

    /// Every family declared above. The resolution test iterates this, so a
    /// family added here is checked without touching the test.
    pub const ALL: &'static [Self] = &[
        Self::IM2COL1D,
        Self::IM2COL,
        Self::COL2IM1D,
        Self::CONV1D_DEPTHWISE,
        Self::CONV1D_DEPTHWISE_K2,
        Self::CONV1D_DEPTHWISE_K3,
        Self::CONV1D_DEPTHWISE_K4,
        Self::CONV_TRANSPOSE1D,
        Self::CONV_TRANSPOSE2D,
        Self::UPSAMPLE_NEAREST2D,
        Self::UPSAMPLE_BILINEAR2D,
        Self::MAX_POOL2D,
        Self::AVG_POOL2D,
    ];

    /// The family's name without a dtype suffix, for diagnostics.
    pub const fn stem(&self) -> &'static str {
        self.stem
    }

    /// The segment following the dtype suffix — empty for the plain families,
    /// `"_k3"` and friends for the `k_size`-specialized depthwise variants.
    pub const fn tail(&self) -> &'static str {
        self.tail
    }

    /// The `[[host_name]]` string for this family at `dtype_suffix`, or `None`
    /// if `conv.metal` does not instantiate that combination.
    ///
    /// `dtype_suffix` is the spelling both crates already use — `candle-core`'s
    /// `DType::as_str`, or the suffix in a kernel name. Returning `None` rather
    /// than a formatted string is the point: an unsupported dtype is refused
    /// here, where the caller can report it against its own dtype enum, and
    /// cannot reach `load_pipeline` as a name that will fail to resolve.
    ///
    /// Linear over at most five entries and called once per op, not per
    /// dispatch element; the pipeline cache in `Kernels` is what the hot path
    /// actually hits.
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
