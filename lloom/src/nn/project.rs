//! The projection op: `y = x · Wᵀ (+ b)` over a prepared weight, at the
//! dtype triple the block declares (`docs/invariants.md` §15.2a, lloom #1155).
//!
//! A block declares its [`Triple`]; [`project`] reads the activation at the
//! triple's activation leg, accumulates in f32 and stores at its store leg.
//! The kernel is chosen by the weight's route: a dense weight takes
//! `nn::Linear`'s product, packed blocks take the quantized cores, whose
//! route [`lloom_tensor::quantized::route::resolve`] decides on Metal. A
//! triple no route serves is refused by name, never downgraded.

use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Module, Result, Tensor};

/// The `(activation, accumulate, store)` dtypes a projection runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Triple {
    pub act: DType,
    pub acc: DType,
    pub store: DType,
}

impl Triple {
    /// `(dt, f32, dt)`: activations and store at `dt`, the sum in f32 — the
    /// triple of every float route (§8.1 #4).
    pub const fn at(dt: DType) -> Self {
        Self {
            act: dt,
            acc: DType::F32,
            store: dt,
        }
    }

    pub const F32: Self = Self::at(DType::F32);
    pub const F16: Self = Self::at(DType::F16);

    /// `(f16, f32, f16)`-style spelling.
    pub fn name(self) -> String {
        format!(
            "({}, {}, {})",
            self.act.as_str(),
            self.acc.as_str(),
            self.store.as_str()
        )
    }
}

/// What the projection does after the product. Only the plain store is a
/// route of this op; the fused epilogues (SiLU/GLU, residual, the norm
/// prologue) are reached through their own kernels until they become
/// epilogues of the shared core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epilogue {
    Store,
}

/// One projection request: the declared triple, the activation rows and the
/// epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub triple: Triple,
    pub rows: usize,
    pub epilogue: Epilogue,
}

/// A weight prepared for [`project`]: a dense `nn::Linear`, or the
/// quantized matmul over packed blocks (or over the dense copy the
/// `qmm.dequantize_all{,_f16}` reference arms make of them at load).
#[derive(Debug, Clone)]
pub enum Weight {
    Dense(crate::nn::Linear),
    Blocks(QMatMul),
}

impl Weight {
    /// `[out, in]` as stored.
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Dense(l) => l.weight().dims(),
            Self::Blocks(QMatMul::QTensor(q)) => q.shape().dims(),
            Self::Blocks(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.dims(),
        }
    }

    /// The reduction length.
    pub fn k(&self) -> usize {
        let d = self.dims();
        d[d.len() - 1]
    }
}

/// The kernel family a projection takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `nn::Linear`'s product at the weight's dtype.
    Dense(DType),
    /// The Metal quantized cores' route over packed blocks.
    #[cfg(feature = "metal")]
    Packed(lloom_tensor::quantized::route::Route),
    /// The quantized matmul's own storage op (CPU, CUDA).
    PackedHost,
    /// The quantized matmul over the dense copy a reference arm made at load
    /// (`QMatMul::Tensor` at f32, `QMatMul::TensorF16` at f16).
    Dequantized(DType),
}

/// The route [`project`] takes for one request: a cast of the input to the
/// triple's activation leg when it arrives at another dtype, then the
/// product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub cast_from: Option<DType>,
    pub route: Route,
}

impl std::fmt::Display for Resolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(from) = self.cast_from {
            write!(f, "cast {} + ", from.as_str())?;
        }
        match self.route {
            Route::Dense(dt) => write!(f, "dense {}", dt.as_str()),
            // The matmul-vec at a capacity of more than one row is the
            // `_r<rows>` core; at one row, the plain core.
            #[cfg(feature = "metal")]
            Route::Packed(lloom_tensor::quantized::route::Route::Matvec { triple, rows: 1 }) => {
                write!(f, "Matvec({triple:?})")
            }
            #[cfg(feature = "metal")]
            Route::Packed(lloom_tensor::quantized::route::Route::Matvec { triple, rows }) => {
                write!(f, "Matvec_r{rows}({triple:?})")
            }
            #[cfg(feature = "metal")]
            Route::Packed(r) => write!(f, "{r:?}"),
            Route::PackedHost => write!(f, "packed (host)"),
            Route::Dequantized(dt) => write!(f, "dequantized {}", dt.as_str()),
        }
    }
}

const F32_COPY: &str = "the f32 copy `qmm.dequantize_all` makes at load reads f32 activations";

fn refuse(w: &Weight, spec: &Spec, why: &str) -> lloom_tensor::Error {
    lloom_tensor::Error::msg(format!(
        "projection over {} [{:?}] at {} × {} rows: {why}",
        kind(w),
        w.dims(),
        spec.triple.name(),
        spec.rows
    ))
}

/// The storage's name (`f16`, `q8_0`, `q4_k (dequantized f16)`, …).
pub fn kind(w: &Weight) -> String {
    match w {
        Weight::Dense(l) => l.weight().dtype().as_str().to_string(),
        Weight::Blocks(QMatMul::QTensor(q)) => format!("{:?}", q.dtype()).to_lowercase(),
        Weight::Blocks(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => {
            format!("{} (dequantized)", t.dtype().as_str())
        }
    }
}

/// The tuning a packed route reads: the quantized resolver's policy on
/// Metal, nothing elsewhere.
#[cfg(feature = "metal")]
pub type Policy = lloom_tensor::quantized::route::Policy;
#[cfg(not(feature = "metal"))]
pub type Policy = ();

/// The process's [`Policy`], read where the dispatch reads it.
#[cfg(feature = "metal")]
pub fn current_policy() -> Policy {
    lloom_tensor::quantized::route::Policy::current()
}

/// The process's [`Policy`]: off Metal no route reads one.
#[cfg(not(feature = "metal"))]
pub fn current_policy() -> Policy {}

/// The route `spec` takes over `w` for an activation arriving at `input`.
/// Pure: it allocates no buffer and dispatches nothing.
pub fn resolve(w: &Weight, spec: &Spec, input: DType) -> Result<Resolved> {
    resolve_under(w, spec, input, current_policy())
}

/// [`resolve`] under `policy` rather than the process's.
pub fn resolve_under(w: &Weight, spec: &Spec, input: DType, policy: Policy) -> Result<Resolved> {
    let t = spec.triple;
    if t.acc != DType::F32 {
        return Err(refuse(w, spec, "every projection route accumulates in f32"));
    }
    if t.store != t.act {
        return Err(refuse(
            w,
            spec,
            "no projection route stores at a dtype other than its activation's",
        ));
    }
    if !matches!(t.act, DType::F32 | DType::F16 | DType::BF16) {
        return Err(refuse(w, spec, "the activation leg must be a float dtype"));
    }
    let cast_from = (input != t.act).then_some(input);
    let route = match w {
        Weight::Dense(l) => {
            if l.weight().dtype() != t.act {
                return Err(refuse(
                    w,
                    spec,
                    "a dense weight is read at its own dtype; the triple's activation leg must match it",
                ));
            }
            Route::Dense(t.act)
        }
        Weight::Blocks(QMatMul::Tensor(_)) => {
            if t.act != DType::F32 {
                return Err(refuse(w, spec, F32_COPY));
            }
            Route::Dequantized(DType::F32)
        }
        Weight::Blocks(QMatMul::TensorF16(_)) => Route::Dequantized(DType::F16),
        Weight::Blocks(QMatMul::QTensor(q)) => packed_route(q, w, spec, policy)?,
    };
    Ok(Resolved { cast_from, route })
}

#[cfg(feature = "metal")]
fn packed_route(
    q: &lloom_tensor::quantized::QTensor,
    w: &Weight,
    spec: &Spec,
    policy: Policy,
) -> Result<Route> {
    use lloom_tensor::quantized::route;
    if !q.device().is_metal() {
        return Ok(Route::PackedHost);
    }
    route::resolve(
        route::Weight {
            kind: q.dtype().into(),
            rank: q.shape().rank(),
            k: w.k(),
        },
        route::Spec {
            act: spec.triple.act,
            rows: spec.rows,
            row_per_batch: spec.rows == 1,
        },
        policy,
    )
    .map(Route::Packed)
    .map_err(|r| refuse(w, spec, &r.to_string()))
}

#[cfg(not(feature = "metal"))]
fn packed_route(
    _: &lloom_tensor::quantized::QTensor,
    _: &Weight,
    _: &Spec,
    _: Policy,
) -> Result<Route> {
    Ok(Route::PackedHost)
}

/// `x · Wᵀ` at `spec`'s triple. An input at another dtype than the triple's
/// activation leg is cast once; the product is the weight's route, the same
/// call the weight's own forward makes, so a request at the dtype the model
/// already computes in takes exactly the kernels it took before this op.
pub fn project(x: &Tensor, w: &Weight, spec: Spec) -> Result<Tensor> {
    if spec.epilogue != Epilogue::Store {
        return Err(refuse(
            w,
            &spec,
            "no fused epilogue is a route of this op yet",
        ));
    }
    let t = spec.triple;
    if t.acc != DType::F32 || t.store != t.act {
        return Err(refuse(w, &spec, "an unserved triple"));
    }
    match w {
        Weight::Dense(l) if l.weight().dtype() != t.act => {
            return Err(refuse(
                w,
                &spec,
                "a dense weight is read at its own dtype; the triple's activation leg must match it",
            ));
        }
        Weight::Blocks(QMatMul::Tensor(_)) if t.act != DType::F32 => {
            return Err(refuse(w, &spec, F32_COPY));
        }
        _ => {}
    }
    let cast;
    let x = if x.dtype() == t.act {
        x
    } else {
        cast = x.to_dtype(t.act)?;
        &cast
    };
    match w {
        Weight::Dense(l) => l.forward(x),
        // The quantized cores read a packed row; a strided view is copied
        // first (a no-op on a contiguous input).
        Weight::Blocks(q) => q.forward(&x.contiguous()?),
    }
}

/// The activation rows of `x` for a weight reducing over `k`.
pub fn rows_of(x: &Tensor, k: usize) -> usize {
    x.elem_count() / k.max(1)
}

/// The routes `triple` takes over `w` across the row bands a model runs
/// (every row count the matmul-vec serves, the first GEMM row, the
/// dequantise-once threshold), for the load line and `/health`. Bands that
/// resolve alike are merged.
pub fn describe(w: &Weight, triple: Triple, input: DType) -> String {
    describe_under(w, triple, input, current_policy())
}

/// [`describe`] under `policy` rather than the process's.
pub fn describe_under(w: &Weight, triple: Triple, input: DType, policy: Policy) -> String {
    let mut bands: Vec<(usize, String)> = Vec::new();
    for rows in band_starts(policy) {
        let spec = Spec {
            triple,
            rows,
            epilogue: Epilogue::Store,
        };
        let r = match resolve_under(w, &spec, input, policy) {
            Ok(r) => r.to_string(),
            Err(e) => format!("refused: {e}"),
        };
        if bands.last().is_none_or(|(_, prev)| *prev != r) {
            bands.push((rows, r));
        }
    }
    bands
        .iter()
        .map(|(rows, r)| format!("rows {rows}+ {r}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Every row count the matmul-vec serves (its capacity can change at each),
/// the first GEMM row and the dequantise-once threshold.
#[cfg(feature = "metal")]
fn band_starts(p: Policy) -> Vec<usize> {
    let mut v: Vec<usize> = (1..=p.mv_rows_max + 1).collect();
    if p.dequant_from > p.mv_rows_max + 1 {
        v.push(p.dequant_from);
    }
    v
}

#[cfg(not(feature = "metal"))]
fn band_starts(_: Policy) -> Vec<usize> {
    vec![1]
}
