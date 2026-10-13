//! The CPU f16 matmul: `dst[i][j] = Σ_k x[i][k] · w[j][k]` over f16 operands,
//! read as f16, with the accumulator a type parameter of one kernel body
//! ([`Accum`]) and the result rounded to f16 once at the end.
//!
//! The accumulator is an axis (`kernels.matmul_f16.accumulate`: the type and
//! the promotion interval), resolved once per process from the tuning
//! ([`resolved`]) and recorded on the `CONTRACT` line. Two combinations are
//! declared:
//!
//! - `f32/1`: f32 throughout — `fmlal` where the CPU has FEAT_FHM, widening
//!   (`fcvtl` + `fmla .4s`, or `vcvtph2ps` + `vfmadd`, or a scalar fused
//!   multiply-add) elsewhere. Every form is the same arithmetic — one
//!   rounding of the exact product-sum per lane step — so the bits are the
//!   same wherever the hardware fuses. The test reference and the
//!   `deterministic` contract's accumulator.
//! - `f16/16`: f16 chains of sixteen products per lane, each chain promoted
//!   into f32 lanes (`fmla .8h`, then `fcvtl` + `fadd .4s`); aarch64 with
//!   `fp16` only. The `fast` arm, declared and selectable, not yet the
//!   `fast` default (no served CPU path reaches this matmul).
//!
//! Every output is eight f32 lanes along `k`, reduced in one fixed tree, so a
//! row's bits do not depend on how many rows the pass carries or on the
//! thread count. (architecture.md §2.3.7)

use crate::{Error, Result};
use half::f16;
use std::sync::OnceLock;

/// The accumulator unit of the kernel body: how one output's eight k-lanes
/// accumulate a step of eight products, and when a chain is promoted.
pub trait Accum {
    /// Eight f16 operands, as the step consumes them.
    type Lanes: Copy;
    /// One output's working accumulator over its eight k-lanes.
    type Acc: Copy;
    /// The f32 lanes a chain is promoted into.
    type Partial: Copy;
    /// Steps (of eight products) a chain runs before [`promote`](Self::promote);
    /// `usize::MAX` when the accumulator is f32 throughout.
    const PROMOTE_EVERY: usize;
    /// `<type>/<promote_every>`, as the option tree spells it.
    const NAME: &'static str;

    /// # Safety
    /// `p` points at eight readable `f16`s.
    unsafe fn load(p: *const f16) -> Self::Lanes;
    /// # Safety
    /// Target-feature instructions; the caller holds the feature.
    unsafe fn zero() -> Self::Acc;
    /// # Safety
    /// As [`zero`](Self::zero).
    unsafe fn zero_partial() -> Self::Partial;
    /// One step: `acc[l] += x[l] · w[l]` for the eight lanes.
    /// # Safety
    /// As [`zero`](Self::zero).
    unsafe fn step(acc: Self::Acc, x: Self::Lanes, w: Self::Lanes) -> Self::Acc;
    /// Fold a finished chain into the partials.
    /// # Safety
    /// As [`zero`](Self::zero).
    unsafe fn promote(acc: Self::Acc, partial: &mut Self::Partial);
    /// The eight f32 lanes of the whole chain: the partials plus what the
    /// last chain holds.
    /// # Safety
    /// As [`zero`](Self::zero).
    unsafe fn reduce(acc: Self::Acc, partial: Self::Partial) -> [f32; 8];
}

/// The declared accumulator: its type and promotion interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// f32 throughout, rounded to f16 once.
    F32,
    /// f16 chains of sixteen products, promoted into f32.
    F16Chain16,
}

impl Kind {
    pub const fn name(self) -> &'static str {
        match self {
            Kind::F32 => "f32/1",
            Kind::F16Chain16 => "f16/16",
        }
    }
}

/// The instruction form the kind runs as on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instr {
    /// aarch64 FEAT_FHM: `fmlal` / `fmlal2`.
    Fmlal,
    /// aarch64: `fcvtl` / `fcvtl2` then `fmla .4s`.
    Widen,
    /// aarch64 `fp16`: `fmla .8h`, promoted by `fcvtl` + `fadd .4s`.
    Fmla16,
    /// x86 AVX2 + F16C + FMA: `vcvtph2ps` then `vfmadd231ps`.
    Avx2Fma,
    /// A scalar fused multiply-add per lane (`f32::mul_add`).
    Scalar,
}

impl Instr {
    pub const fn name(self) -> &'static str {
        match self {
            Instr::Fmlal => "fmlal",
            Instr::Widen => "widen",
            Instr::Fmla16 => "fmla.8h",
            Instr::Avx2Fma => "avx2-fma",
            Instr::Scalar => "scalar-fma",
        }
    }
}

/// What the process's CPU f16 matmuls run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    pub kind: Kind,
    pub instr: Instr,
}

impl Resolved {
    /// `<type>/<promote_every>:<instruction>`, the `CONTRACT` line's token.
    pub fn report(&self) -> String {
        format!("{}:{}", self.kind.name(), self.instr.name())
    }
}

/// The key the option tree spells the axis under.
pub const KEY: &str = "kernels.matmul_f16.accumulate";

/// The kind the tuning declares (`auto` resolves to `f32/1` under both
/// contracts), or the refusal naming the undeclared combination.
pub fn declared(ty: &str, promote_every: &str) -> std::result::Result<Kind, String> {
    match (ty, promote_every) {
        ("auto", "auto") | ("f32", "auto") | ("auto", "1") | ("f32", "1") => Ok(Kind::F32),
        ("f16", "auto") | ("auto", "16") | ("f16", "16") => Ok(Kind::F16Chain16),
        (t, p) => Err(format!(
            "{KEY}: `{{type={t}, promote_every={p}}}` is not a declared combination \
             (declared: f32/1, f16/16)"
        )),
    }
}

/// The instruction form `kind` takes on this CPU, or the refusal by name.
#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
pub fn instruction(kind: Kind) -> std::result::Result<Instr, String> {
    use crate::cpu::arms::{native, Feature};
    Ok(match kind {
        Kind::F32 if native(Feature::Fhm) => Instr::Fmlal,
        Kind::F32 => Instr::Widen,
        Kind::F16Chain16 => Instr::Fmla16,
    })
}

/// The instruction form `kind` takes on this CPU, or the refusal by name.
#[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
pub fn instruction(kind: Kind) -> std::result::Result<Instr, String> {
    match kind {
        Kind::F32 => Ok(arch::f32_instr()),
        Kind::F16Chain16 => Err(format!(
            "{KEY}: accumulator `{}` is not implemented on this CPU (f16 chains need \
             aarch64 with `fp16`; implemented here: f32/1)",
            kind.name()
        )),
    }
}

/// The accumulator this process's CPU f16 matmuls run, resolved once from
/// the tuning; a refused combination is the error every matmul returns.
pub fn resolved() -> Result<Resolved> {
    static RESOLVED: OnceLock<std::result::Result<Resolved, String>> = OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            let t = &crate::tuning::tuning().cpu;
            let kind = declared(&t.f16_matmul_accumulate, &t.f16_matmul_promote_every)?;
            let instr = instruction(kind)?;
            Ok(Resolved { kind, instr })
        })
        .clone()
        .map_err(Error::Msg)
}

/// `resolved()` as the `CONTRACT` line prints it; a refusal prints as such.
pub fn report() -> String {
    match resolved() {
        Ok(r) => r.report(),
        Err(_) => "refused".to_string(),
    }
}

/// The operands of one matmul: `x` is `[m, k]` with rows `x_rs` apart, `w`
/// is `[n, k]` with rows `w_rs` apart (both contiguous along `k`), `dst` is
/// `[m, n]` with strides `dst_rs`, `dst_cs`.
#[derive(Clone, Copy)]
pub struct Operands {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub x: *const f16,
    pub x_rs: usize,
    pub w: *const f16,
    pub w_rs: usize,
    pub dst: *mut f16,
    pub dst_rs: usize,
    pub dst_cs: usize,
}

// SAFETY: the pointers are shared read-only (`x`, `w`) or written at disjoint
// columns by the workers (`dst`), which `run` guarantees by partitioning.
unsafe impl Send for Operands {}
unsafe impl Sync for Operands {}

/// Columns a worker takes at once.
const CHUNK: usize = 64;

type ChunkFn = unsafe fn(&Operands, usize, usize);

fn chunk_fn(instr: Instr) -> ChunkFn {
    match instr {
        Instr::Fmlal => arch::chunk_fmlal,
        Instr::Widen => arch::chunk_widen,
        Instr::Fmla16 => arch::chunk_fmla16,
        Instr::Avx2Fma => arch::chunk_avx2_fma,
        Instr::Scalar => scalar::chunk,
    }
}

/// `dst = x · wᵀ` with the process's accumulator ([`resolved`]), over
/// `threads` rayon threads when the product is worth splitting.
///
/// # Safety
/// `o`'s pointers are valid for the extents its sizes and strides describe.
pub unsafe fn matmul(o: &Operands, threads: usize) -> Result<()> {
    let r = resolved()?;
    run(o, threads, chunk_fn(r.instr));
    Ok(())
}

/// `matmul` with the accumulator named, for a test that compares the arms
/// in one process; refused by name where the CPU lacks it.
///
/// # Safety
/// As [`matmul`].
pub unsafe fn matmul_with(o: &Operands, kind: Kind, threads: usize) -> Result<()> {
    let instr = instruction(kind).map_err(Error::Msg)?;
    run(o, threads, chunk_fn(instr));
    Ok(())
}

unsafe fn run(o: &Operands, threads: usize, f: ChunkFn) {
    if o.m == 0 || o.n == 0 {
        return;
    }
    let chunks = o.n.div_ceil(CHUNK);
    // A product too small to split pays rayon's hand-off for nothing.
    if threads <= 1 || chunks < 2 || o.m * o.n * o.k < 1 << 16 {
        f(o, 0, o.n);
        return;
    }
    use rayon::prelude::*;
    (0..chunks).into_par_iter().for_each(|c| {
        let lo = c * CHUNK;
        let hi = (lo + CHUNK).min(o.n);
        // SAFETY: each chunk writes its own columns of `dst`; the reads are
        // shared and immutable for the call.
        unsafe { f(o, lo, hi) }
    });
}

/// The fixed reduction of an output's eight lanes and its scalar tail.
#[inline(always)]
fn reduce_lanes(l: [f32; 8], tail: f32) -> f16 {
    let s = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
    f16::from_f32(s + tail)
}

/// The products past the last full step, in f32, one fused multiply-add
/// each.
#[inline(always)]
unsafe fn tail(x: *const f16, w: *const f16, k8: usize, k: usize) -> f32 {
    let mut t = 0f32;
    for kk in k8..k {
        t = (*x.add(kk)).to_f32().mul_add((*w.add(kk)).to_f32(), t);
    }
    t
}

/// One register tile of `MR` rows by `NR` columns over the whole `k`: the
/// kernel body, one definition for every accumulator.
#[inline(always)]
unsafe fn tile<A: Accum, const MR: usize, const NR: usize>(o: &Operands, i0: usize, j0: usize) {
    let k8 = o.k / 8 * 8;
    let mut acc = [[A::zero(); NR]; MR];
    let mut part = [[A::zero_partial(); NR]; MR];
    let mut steps = 0usize;
    let mut kk = 0usize;
    while kk < k8 {
        let xs: [A::Lanes; MR] = core::array::from_fn(|i| A::load(o.x.add((i0 + i) * o.x_rs + kk)));
        let ws: [A::Lanes; NR] = core::array::from_fn(|j| A::load(o.w.add((j0 + j) * o.w_rs + kk)));
        for i in 0..MR {
            for j in 0..NR {
                acc[i][j] = A::step(acc[i][j], xs[i], ws[j]);
            }
        }
        kk += 8;
        if A::PROMOTE_EVERY != usize::MAX {
            steps += 1;
            if steps == A::PROMOTE_EVERY {
                for i in 0..MR {
                    for j in 0..NR {
                        A::promote(acc[i][j], &mut part[i][j]);
                        acc[i][j] = A::zero();
                    }
                }
                steps = 0;
            }
        }
    }
    for i in 0..MR {
        for j in 0..NR {
            let lanes = A::reduce(acc[i][j], part[i][j]);
            let t = tail(
                o.x.add((i0 + i) * o.x_rs),
                o.w.add((j0 + j) * o.w_rs),
                k8,
                o.k,
            );
            *o.dst.add((i0 + i) * o.dst_rs + (j0 + j) * o.dst_cs) = reduce_lanes(lanes, t);
        }
    }
}

/// Columns `[lo, hi)` of every row, tiled 2×4 while two rows remain and
/// 1×8 on a lone row (the decode shape), the remainders 1×1; every tile
/// computes an output by the same arithmetic.
#[inline(always)]
unsafe fn chunk_body<A: Accum>(o: &Operands, lo: usize, hi: usize) {
    let mut i = 0;
    while i + 2 <= o.m {
        let mut j = lo;
        while j + 4 <= hi {
            tile::<A, 2, 4>(o, i, j);
            j += 4;
        }
        while j < hi {
            tile::<A, 2, 1>(o, i, j);
            j += 1;
        }
        i += 2;
    }
    if i < o.m {
        let mut j = lo;
        while j + 8 <= hi {
            tile::<A, 1, 8>(o, i, j);
            j += 8;
        }
        while j < hi {
            tile::<A, 1, 1>(o, i, j);
            j += 1;
        }
    }
}

/// The portable form: eight f32 lanes, one scalar fused multiply-add each.
pub mod scalar {
    use super::*;

    /// `f32/1` by `f32::mul_add` per lane.
    pub struct F32Scalar;

    impl Accum for F32Scalar {
        type Lanes = [f32; 8];
        type Acc = [f32; 8];
        type Partial = ();
        const PROMOTE_EVERY: usize = usize::MAX;
        const NAME: &'static str = "f32/1";

        #[inline(always)]
        unsafe fn load(p: *const f16) -> [f32; 8] {
            core::array::from_fn(|l| (*p.add(l)).to_f32())
        }
        #[inline(always)]
        unsafe fn zero() -> [f32; 8] {
            [0f32; 8]
        }
        #[inline(always)]
        unsafe fn zero_partial() {}
        #[inline(always)]
        unsafe fn step(mut acc: [f32; 8], x: [f32; 8], w: [f32; 8]) -> [f32; 8] {
            for l in 0..8 {
                acc[l] = x[l].mul_add(w[l], acc[l]);
            }
            acc
        }
        #[inline(always)]
        unsafe fn promote(_acc: [f32; 8], _partial: &mut ()) {}
        #[inline(always)]
        unsafe fn reduce(acc: [f32; 8], _partial: ()) -> [f32; 8] {
            acc
        }
    }

    pub(super) unsafe fn chunk(o: &Operands, lo: usize, hi: usize) {
        chunk_body::<F32Scalar>(o, lo, hi)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
mod arch {
    use super::*;
    use core::arch::aarch64::*;
    use std::arch::asm;

    /// The low and high four halves of a register, widened.
    #[inline(always)]
    unsafe fn widen(h: uint16x8_t) -> [float32x4_t; 2] {
        let lo: float32x4_t;
        let hi: float32x4_t;
        asm!(
            "fcvtl {lo:v}.4s, {h:v}.4h",
            "fcvtl2 {hi:v}.4s, {h:v}.8h",
            h = in(vreg) h,
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            options(pure, nomem, nostack)
        );
        [lo, hi]
    }

    #[inline(always)]
    unsafe fn lanes(v: [float32x4_t; 2]) -> [f32; 8] {
        let mut out = [0f32; 8];
        vst1q_f32(out.as_mut_ptr(), v[0]);
        vst1q_f32(out.as_mut_ptr().add(4), v[1]);
        out
    }

    /// `f32/1` by `fmlal` / `fmlal2` (FEAT_FHM): the products of the low and
    /// high halves accumulated into f32 lanes by one instruction each.
    pub struct F32Fmlal;

    impl Accum for F32Fmlal {
        type Lanes = uint16x8_t;
        type Acc = [float32x4_t; 2];
        type Partial = ();
        const PROMOTE_EVERY: usize = usize::MAX;
        const NAME: &'static str = "f32/1";

        #[inline(always)]
        unsafe fn load(p: *const f16) -> uint16x8_t {
            vld1q_u16(p as *const u16)
        }
        #[inline(always)]
        unsafe fn zero() -> Self::Acc {
            [vdupq_n_f32(0.0); 2]
        }
        #[inline(always)]
        unsafe fn zero_partial() {}
        #[inline]
        #[target_feature(enable = "fhm")]
        unsafe fn step(acc: Self::Acc, x: uint16x8_t, w: uint16x8_t) -> Self::Acc {
            let lo: float32x4_t;
            let hi: float32x4_t;
            asm!(
                "fmlal {lo:v}.4s, {x:v}.4h, {w:v}.4h",
                "fmlal2 {hi:v}.4s, {x:v}.4h, {w:v}.4h",
                lo = inout(vreg) acc[0] => lo,
                hi = inout(vreg) acc[1] => hi,
                x = in(vreg) x,
                w = in(vreg) w,
                options(pure, nomem, nostack)
            );
            [lo, hi]
        }
        #[inline(always)]
        unsafe fn promote(_acc: Self::Acc, _partial: &mut ()) {}
        #[inline(always)]
        unsafe fn reduce(acc: Self::Acc, _partial: ()) -> [f32; 8] {
            lanes(acc)
        }
    }

    /// `f32/1` by widening: `fcvtl` both operands, `fmla .4s`.
    pub struct F32Widen;

    impl Accum for F32Widen {
        type Lanes = [float32x4_t; 2];
        type Acc = [float32x4_t; 2];
        type Partial = ();
        const PROMOTE_EVERY: usize = usize::MAX;
        const NAME: &'static str = "f32/1";

        #[inline(always)]
        unsafe fn load(p: *const f16) -> Self::Lanes {
            widen(vld1q_u16(p as *const u16))
        }
        #[inline(always)]
        unsafe fn zero() -> Self::Acc {
            [vdupq_n_f32(0.0); 2]
        }
        #[inline(always)]
        unsafe fn zero_partial() {}
        #[inline(always)]
        unsafe fn step(acc: Self::Acc, x: Self::Lanes, w: Self::Lanes) -> Self::Acc {
            [vfmaq_f32(acc[0], x[0], w[0]), vfmaq_f32(acc[1], x[1], w[1])]
        }
        #[inline(always)]
        unsafe fn promote(_acc: Self::Acc, _partial: &mut ()) {}
        #[inline(always)]
        unsafe fn reduce(acc: Self::Acc, _partial: ()) -> [f32; 8] {
            lanes(acc)
        }
    }

    /// `f16/16`: `fmla .8h` for sixteen steps, the chain widened and added
    /// into the f32 lanes.
    pub struct F16Chain16;

    impl Accum for F16Chain16 {
        type Lanes = uint16x8_t;
        type Acc = uint16x8_t;
        type Partial = [float32x4_t; 2];
        const PROMOTE_EVERY: usize = 16;
        const NAME: &'static str = "f16/16";

        #[inline(always)]
        unsafe fn load(p: *const f16) -> uint16x8_t {
            vld1q_u16(p as *const u16)
        }
        #[inline(always)]
        unsafe fn zero() -> uint16x8_t {
            vdupq_n_u16(0)
        }
        #[inline(always)]
        unsafe fn zero_partial() -> Self::Partial {
            [vdupq_n_f32(0.0); 2]
        }
        #[inline(always)]
        unsafe fn step(acc: uint16x8_t, x: uint16x8_t, w: uint16x8_t) -> uint16x8_t {
            let r: uint16x8_t;
            asm!(
                "fmla {a:v}.8h, {x:v}.8h, {w:v}.8h",
                a = inout(vreg) acc => r,
                x = in(vreg) x,
                w = in(vreg) w,
                options(pure, nomem, nostack)
            );
            r
        }
        #[inline(always)]
        unsafe fn promote(acc: uint16x8_t, partial: &mut Self::Partial) {
            let w = widen(acc);
            partial[0] = vaddq_f32(partial[0], w[0]);
            partial[1] = vaddq_f32(partial[1], w[1]);
        }
        #[inline(always)]
        unsafe fn reduce(acc: uint16x8_t, mut partial: Self::Partial) -> [f32; 8] {
            Self::promote(acc, &mut partial);
            lanes(partial)
        }
    }

    #[target_feature(enable = "fhm")]
    pub(super) unsafe fn chunk_fmlal(o: &Operands, lo: usize, hi: usize) {
        chunk_body::<F32Fmlal>(o, lo, hi)
    }
    pub(super) unsafe fn chunk_widen(o: &Operands, lo: usize, hi: usize) {
        chunk_body::<F32Widen>(o, lo, hi)
    }
    pub(super) unsafe fn chunk_fmla16(o: &Operands, lo: usize, hi: usize) {
        chunk_body::<F16Chain16>(o, lo, hi)
    }
    pub(super) unsafe fn chunk_avx2_fma(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: avx2-fma is not an aarch64 instruction form")
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod arch {
    use super::*;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// The f32 form this CPU runs: the fused AVX2 one where the three
    /// features are native, the scalar one otherwise.
    pub(super) fn f32_instr() -> Instr {
        use crate::cpu::arms::{native, Feature};
        if native(Feature::Avx2) && native(Feature::F16c) && native(Feature::Fma) {
            Instr::Avx2Fma
        } else {
            Instr::Scalar
        }
    }

    /// `f32/1` by `vcvtph2ps` + `vfmadd231ps`: eight lanes in one register,
    /// the same lane order as the aarch64 forms.
    pub struct F32Avx2Fma;

    impl Accum for F32Avx2Fma {
        type Lanes = __m256;
        type Acc = __m256;
        type Partial = ();
        const PROMOTE_EVERY: usize = usize::MAX;
        const NAME: &'static str = "f32/1";

        #[inline]
        #[target_feature(enable = "avx2,f16c")]
        unsafe fn load(p: *const f16) -> __m256 {
            _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
        }
        #[inline]
        #[target_feature(enable = "avx")]
        unsafe fn zero() -> __m256 {
            _mm256_setzero_ps()
        }
        #[inline(always)]
        unsafe fn zero_partial() {}
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn step(acc: __m256, x: __m256, w: __m256) -> __m256 {
            _mm256_fmadd_ps(x, w, acc)
        }
        #[inline(always)]
        unsafe fn promote(_acc: __m256, _partial: &mut ()) {}
        #[inline]
        #[target_feature(enable = "avx")]
        unsafe fn reduce(acc: __m256, _partial: ()) -> [f32; 8] {
            let mut out = [0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), acc);
            out
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn chunk_avx2_fma(o: &Operands, lo: usize, hi: usize) {
        chunk_body::<F32Avx2Fma>(o, lo, hi)
    }
    pub(super) unsafe fn chunk_fmlal(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: fmlal is not an x86 instruction form")
    }
    pub(super) unsafe fn chunk_widen(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: widen is not an x86 instruction form")
    }
    pub(super) unsafe fn chunk_fmla16(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: fmla.8h is not an x86 instruction form")
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "fp16"),
    target_arch = "x86",
    target_arch = "x86_64"
)))]
mod arch {
    use super::*;

    pub(super) fn f32_instr() -> Instr {
        Instr::Scalar
    }
    pub(super) unsafe fn chunk_fmlal(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: fmlal is not an instruction form of this target")
    }
    pub(super) unsafe fn chunk_widen(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: widen is not an instruction form of this target")
    }
    pub(super) unsafe fn chunk_fmla16(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: fmla.8h is not an instruction form of this target")
    }
    pub(super) unsafe fn chunk_avx2_fma(_o: &Operands, _lo: usize, _hi: usize) {
        unreachable!("{KEY}: avx2-fma is not an instruction form of this target")
    }
}
