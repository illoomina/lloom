//! The variant names `unary`, `binary`, `cast` and `affine` actually
//! instantiate, declared once and checked against the compiled library.
//!
//! # Why this is a list rather than a cross product
//!
//! `DESIGN.md` §8.1b explains why `unary`/`binary`/`cast` were left out when the
//! `conv` and `reduce` registries were built: those families "build names by
//! token pasting inside Metal macros, so the name never appears as a literal on
//! either side and the Rust list becomes a cross product." That is exactly the
//! trap this module has to avoid.
//!
//! The `ops!` macro in `kernels/macros.rs` generates six dtypes for every op --
//! `cos_f32`, `cos_f16`, `cos_bf16`, `cos_i64`, `cos_u32`, `cos_u8` -- but
//! `unary.metal` instantiates `init_unary_float` for `cos`, which is three
//! dtypes. The other three constants name kernels that do not exist. Taking
//! `ops!` as the registry would therefore declare names that cannot resolve,
//! and the resolution test would fail on the *test's* error rather than on a
//! real one.
//!
//! So the axes are declared here as they appear in each `.metal` file's
//! instantiation section, and `packed_names_resolve` loads every resulting name
//! against the real metallib. That is `DESIGN.md` §8.1b's argument -- resolution
//! against the compiled artifact, not agreement between two lists -- and #26
//! demonstrated the need for it by shipping 48 variants absent from a metallib
//! that compiled cleanly.

/// Ops instantiated by `init_unary_float` -- f32, f16, and bf16 when available.
pub const UNARY_FLOAT_OPS: &[&str] = &[
    "gelu_erf", "sqrt", "sqr", "neg", "recip", "copy", "silu", "gelu", "relu", "cos", "sin", "exp",
    "log", "abs", "ceil", "floor", "round", "erf", "sign", "sigmoid", "tanh",
];

/// Float dtypes every `init_unary_float` row covers.
///
/// `bf16` is behind `__HAVE_BFLOAT__` in the Metal source and is therefore
/// checked separately by the resolution test rather than assumed present.
pub const FLOAT_TNAMES: &[&str] = &["f32", "f16"];

/// Integer rows `unary.metal` instantiates by hand: `copy` only.
///
/// `i64` is excluded here and everywhere below because it sits behind
/// `#if __METAL_VERSION__ >= 220`, as `bf16` sits behind `__HAVE_BFLOAT__`.
/// Rather than encode a guess about which guards this machine's runtime
/// compiler takes, the resolution test checks a conditional dtype's `_packed`
/// name **only when its classical name resolves** -- so the guards are read off
/// the compiled library instead of predicted. See `packed_names_resolve`.
pub const UNARY_INT_COPY_TNAMES: &[&str] = &["u8", "u32"];

/// dtypes `init_copy2d` covers unconditionally.
pub const COPY2D_TNAMES: &[&str] = &["f32", "f16", "u8", "u32", "i32", "i16"];

/// dtypes `init_const_set` covers unconditionally.
pub const CONST_SET_TNAMES: &[&str] = &["f32", "f16", "u8", "u32"];

/// dtypes that are present only behind a preprocessor guard, per family.
///
/// Each entry is checked conditionally: if the classical name resolves, the
/// packed one must too. That turns "which guards fired?" from an assumption
/// into an observation, which is the same instinct as checking names against
/// the compiled library rather than against another list.
pub const CONDITIONAL_TNAMES: &[&str] = &["bf16", "i64"];

/// Binary ops returning their input type (`init_binary`).
pub const BINARY_OPS: &[&str] = &["badd", "bsub", "bmul", "bdiv", "bminimum", "bmaximum"];

/// Binary ops returning `bool` (`init_boolean_binary`).
///
/// The `bool` here is the *output element type* -- a `device U*` binding -- not
/// a scalar parameter, so MSL's 1-byte `bool` never reaches a packed struct.
/// Worth stating because issue #38 flags `primitive!(bool)` as a hazard for the
/// families that follow it, and this is the family where one might expect it to
/// fire. It does not: no kernel in these four files takes a `bool` scalar.
pub const BINARY_BOOL_OPS: &[&str] = &["eq", "ne", "le", "lt", "ge", "gt"];

/// dtypes `init_binary` and `init_boolean_binary` cover unconditionally.
pub const BINARY_TNAMES: &[&str] = &["f32", "f16", "u8", "u32", "i64"];

/// The nine indexer suffixes `init_binary_k` emits per dtype.
///
/// `""` is the contiguous kernel; the rest are `binary_kernel_strided` with a
/// different `(l_indexer, r_indexer)` pair.
pub const BINARY_SUFFIXES: &[&str] = &[
    "",
    "_strided",
    "_lstrided",
    "_rstrided",
    "_scalar",
    "_cs",
    "_sc",
    "_rss",
    "_lss",
];

/// dtypes `init_cast_all` covers unconditionally, as both source and target.
pub const CAST_TNAMES: &[&str] = &["f32", "f16", "i64", "u32", "u8"];

/// dtypes `init_affine` covers unconditionally.
pub const AFFINE_TNAMES: &[&str] = &["u8", "u32", "i64", "f32", "f16"];

/// dtypes `init_powf` and `init_elu` cover unconditionally.
pub const SCALE_TNAMES: &[&str] = &["f32", "f16"];

/// A declared variant, and whether its presence is guaranteed.
///
/// `Unconditional` names must resolve. `Conditional` names sit behind a
/// preprocessor guard (`__HAVE_BFLOAT__`, `__METAL_VERSION__ >= 220`), so the
/// test requires only that a packed form exists wherever the classical one
/// does -- which is the property that actually matters and does not depend on
/// guessing what the runtime compiler defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    Unconditional,
    Conditional,
}

/// Every classical `[[host_name]]` in `unary.metal`, in both contiguous and
/// strided form where the file instantiates one.
pub fn unary_names() -> Vec<(String, Presence)> {
    use Presence::{Conditional, Unconditional};
    let mut v = Vec::new();
    for op in UNARY_FLOAT_OPS {
        for t in FLOAT_TNAMES {
            v.push((format!("{op}_{t}"), Unconditional));
            v.push((format!("{op}_{t}_strided"), Unconditional));
        }
        // bf16 only when `__HAVE_BFLOAT__`.
        v.push((format!("{op}_bf16"), Conditional));
        v.push((format!("{op}_bf16_strided"), Conditional));
    }
    for t in UNARY_INT_COPY_TNAMES {
        v.push((format!("copy_{t}"), Unconditional));
        v.push((format!("copy_{t}_strided"), Unconditional));
    }
    v.push(("copy_i64".to_string(), Conditional));
    v.push(("copy_i64_strided".to_string(), Conditional));
    for t in COPY2D_TNAMES {
        v.push((format!("copy2d_{t}"), Unconditional));
    }
    for t in CONDITIONAL_TNAMES {
        v.push((format!("copy2d_{t}"), Conditional));
    }
    for t in CONST_SET_TNAMES {
        v.push((format!("const_set_{t}"), Unconditional));
        v.push((format!("const_set_{t}_strided"), Unconditional));
    }
    for t in CONDITIONAL_TNAMES {
        v.push((format!("const_set_{t}"), Conditional));
        v.push((format!("const_set_{t}_strided"), Conditional));
    }
    v
}

/// Every classical `[[host_name]]` in `binary.metal`.
pub fn binary_names() -> Vec<(String, Presence)> {
    use Presence::{Conditional, Unconditional};
    let mut v = Vec::new();
    for op in BINARY_OPS.iter().chain(BINARY_BOOL_OPS.iter()) {
        for t in BINARY_TNAMES {
            for suffix in BINARY_SUFFIXES {
                v.push((format!("{op}_{t}{suffix}"), Unconditional));
            }
        }
        for suffix in BINARY_SUFFIXES {
            v.push((format!("{op}_bf16{suffix}"), Conditional));
        }
    }
    v
}

/// Every classical `[[host_name]]` in `cast.metal`.
pub fn cast_names() -> Vec<(String, Presence)> {
    use Presence::{Conditional, Unconditional};
    let mut v = Vec::new();
    for from in CAST_TNAMES {
        for to in CAST_TNAMES {
            v.push((format!("cast_{from}_{to}"), Unconditional));
            v.push((format!("cast_{from}_{to}_strided"), Unconditional));
        }
        // bf16 as a target of every source, and as a source of every target.
        v.push((format!("cast_{from}_bf16"), Conditional));
        v.push((format!("cast_{from}_bf16_strided"), Conditional));
        v.push((format!("cast_bf16_{from}"), Conditional));
        v.push((format!("cast_bf16_{from}_strided"), Conditional));
    }
    v
}

/// Every classical `[[host_name]]` in `affine.metal`.
pub fn affine_names() -> Vec<(String, Presence)> {
    use Presence::{Conditional, Unconditional};
    let mut v = Vec::new();
    for t in AFFINE_TNAMES {
        v.push((format!("affine_{t}"), Unconditional));
        v.push((format!("affine_{t}_strided"), Unconditional));
    }
    for t in SCALE_TNAMES {
        v.push((format!("powf_{t}"), Unconditional));
        v.push((format!("powf_{t}_strided"), Unconditional));
        v.push((format!("elu_{t}"), Unconditional));
        v.push((format!("elu_{t}_strided"), Unconditional));
    }
    for stem in ["affine", "powf", "elu"] {
        v.push((format!("{stem}_bf16"), Conditional));
        v.push((format!("{stem}_bf16_strided"), Conditional));
    }
    v
}
