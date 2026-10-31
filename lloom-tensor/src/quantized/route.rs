//! The Metal route of a projection over a packed weight: which kernel family
//! a request takes, decided once, in front of the dispatch, and named when
//! refused (`docs/invariants.md` §15.2a, lloom #1155).
//!
//! A request is the weight's kind and geometry, the activation dtype and the
//! activation row count. The resolver reads the kind table
//! ([`crate::kernels::QuantKind`]) and the tuning the dispatch reads
//! ([`Policy`]); it allocates nothing and dispatches nothing, so a refused
//! request has cost nothing. A cell no kernel serves is refused by name,
//! never downgraded.

use crate::kernels::{GemvTriple, GgmlDType};
use crate::DType;

/// The packed weight as the route sees it.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub kind: GgmlDType,
    /// The weight tensor's rank (2 for one matrix, more when batched).
    pub rank: usize,
    /// The reduction length.
    pub k: usize,
}

/// One request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    /// The activation dtype; the accumulator is f32 on every route (§8.1 #4).
    pub act: DType,
    /// Activation rows over every batch.
    pub rows: usize,
    /// Each batch carries one row (`[B, 1, k]`): a decode batch, which
    /// takes the matmul-vec whatever `B` is.
    pub row_per_batch: bool,
}

/// The tuning a route depends on, read where the dispatch reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Rows up to which the matmul-vec serves a request (`qmv.mm_from` − 1).
    pub mv_rows_max: usize,
    /// Rows from which a stem dequantises to half once (`qmm.dequant_from`;
    /// 0 = never).
    pub dequant_from: usize,
    /// The steel GEMM reads the dequantised scratch (`qmm.dequant_gemm`).
    pub dequant_steel: bool,
    /// The matmul-vec's row capacity policy.
    pub rows: RowsPolicy,
}

/// What decides the matmul-vec's row capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowsPolicy {
    /// The process's: `qmv.rows`, `qmv.rows_max` and the contract, read
    /// only when the kind's ceiling admits more than one row
    /// ([`crate::kernels::mv_rows_capacity`]), so a one-row request reads
    /// no contract.
    Process,
    /// Stated: `axis` is the row axis on under `fast`, `rows_max` the
    /// override of every kind's ceiling.
    Fixed { axis: bool, rows_max: Option<usize> },
}

impl Policy {
    /// The process's policy.
    pub fn current() -> Self {
        Self {
            mv_rows_max: super::metal::QMetalStorage::qmv_rows_max(),
            dequant_from: super::metal::dequant_once_from(),
            dequant_steel: super::metal::dequant_gemm_steel(),
            rows: RowsPolicy::Process,
        }
    }
}

/// The kernel family a request takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `kernel_mul_mv_<kind>_<triple>` with `rows` activation rows a weight
    /// read: 1 the one-row core over each row, 2, 4 or 8 the core at that
    /// capacity (`_r<rows>`, the kind's measured ceiling, `docs/tuning.md`
    /// §13.5w).
    Matvec { triple: GemvTriple, rows: usize },
    /// `kernel_mul_mm_<kind>_<triple>`, unpacking the weight per tile.
    Gemm(GemvTriple),
    /// `kernel_dequant_rows_f16_<kind>` into a scratch, then the quantized
    /// GEMM's f16-weight arm (`kernel_mul_mm_f16_<triple>`).
    DequantOnceGemm(GemvTriple),
    /// The same scratch read by the steel GEMM, the activations cast to half
    /// (f32 activations only).
    DequantOnceSteel,
}

/// A request no route serves: the cell, and the capability it lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub cell: String,
    pub need: &'static str,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quantized projection on Metal: no route for {}; it needs {}",
            self.cell, self.need
        )
    }
}

impl std::error::Error for Refused {}

fn cell(w: Weight, s: Spec) -> String {
    format!(
        "{} weight × {:?} activations × {} rows",
        w.kind.kind().stem,
        s.act,
        s.rows
    )
}

/// The route `s` takes over `w` under `p`.
pub fn resolve(w: Weight, s: Spec, p: Policy) -> Result<Route, Refused> {
    if s.row_per_batch || s.rows <= p.mv_rows_max {
        let triple = match s.act {
            DType::F32 => GemvTriple::F32F32F32,
            DType::F16 if w.kind.has_gemv_triple() => GemvTriple::F16F32F16,
            _ => {
                return Err(Refused {
                    cell: cell(w, s),
                    need: "a matmul-vec arm reading this activation dtype",
                })
            }
        };
        let rows = match p.rows {
            RowsPolicy::Process => crate::kernels::mv_rows_capacity(w.kind, s.rows, false),
            RowsPolicy::Fixed { axis, rows_max } => {
                crate::kernels::mv_rows_capacity_under(w.kind, s.rows, false, axis, rows_max)
            }
        };
        return Ok(Route::Matvec { triple, rows });
    }
    if w.kind.kind().mul_mm.is_none() {
        return Err(Refused {
            cell: cell(w, s),
            need: "a GEMM arm for this weight kind",
        });
    }
    let triple = match s.act {
        DType::F32 => GemvTriple::F32F32F32,
        DType::F16 if w.kind.kind().mul_mm_f16_arm() => GemvTriple::F16F32F16,
        _ => {
            return Err(Refused {
                cell: cell(w, s),
                need: "a GEMM arm reading this activation dtype",
            })
        }
    };
    let dequant = p.dequant_from > 0
        && s.rows >= p.dequant_from
        && w.rank == 2
        && w.k.is_multiple_of(16)
        && w.kind.kind().dequant_rows_f16;
    match (dequant, p.dequant_steel, triple) {
        (false, _, t) => Ok(Route::Gemm(t)),
        (true, false, t) => Ok(Route::DequantOnceGemm(t)),
        (true, true, GemvTriple::F32F32F32) => Ok(Route::DequantOnceSteel),
        (true, true, _) => Err(Refused {
            cell: cell(w, s),
            need: "the f32 activations the steel dequantise-once arm reads",
        }),
    }
}
