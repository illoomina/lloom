//! Do the tree's SiLU/sigmoid spellings actually compute different numbers?
//!
//! Measurement instrument for lloom issue #599.
//!
//! # The claim this exists to settle
//!
//! `gemv.metal:125-130` warns, in the source, that
//! *"`x * (1/(1+exp(-x)))` and `x/(1+exp(-x))` are not the same float
//! operation, and a different spelling would move the digests for a reason
//! that is not the fusion"*. #599's premise builds on that warning: it holds
//! that the tree carries divergent SiLU spellings *"known to disagree"* and
//! that **"a digest could move today purely because two kernels spell SiLU
//! differently"**, calling it a latent §2.3.5a defect.
//!
//! **The warning has never been measured.** Nothing in `measurements/`
//! quantifies the disagreement, and the claim that one is latent in the
//! shipping tree is a claim about REACHABILITY, not about spelling. This probe
//! measures the first half: given identical inputs at identical precision, do
//! the spellings produce different bits, and by how much?
//!
//! A null here is a result (§1.2). If the spellings agree bit-for-bit at every
//! input, the "divergent spellings" premise is false as stated and the
//! consolidation loses its correctness argument -- leaving only the
//! duplication argument, which is a different and weaker case.
//!
//! # Why a private library rather than the shipping kernels
//!
//! The shipping arms are not separable on the spelling axis. `usilu` is
//! reachable only at `T` through `define_unary_op`'s `static_cast<T>(op)`, and
//! `EpilogueSilu` is reachable only fused to a GEMV whose f32 accumulator
//! supplies its input -- so comparing them measures **precision** (f16 vs f32
//! accumulator), which `gemv.metal:132-141` already documents, rather than
//! **spelling**. To isolate spelling the precision must be held constant, so
//! each spelling is compiled here as its own kernel over the same buffer at the
//! same dtype with nothing else differing.
//!
//! The bodies are copied character-for-character from the sites named below. A
//! paraphrase would measure the paraphrase (§3.1c).
//!
//! # Non-vacuity
//!
//! Two controls, both asserted rather than eyeballed, because a probe that
//! silently compared a buffer against itself would print a clean row of
//! "identical" and look like a strong null (§3.1a rule 4):
//!
//! * an **identity control** -- the same spelling compiled under two different
//!   kernel names must be bit-identical. If it is not, the harness itself is
//!   nondeterministic and no other row is readable.
//! * a **divergence control** -- a deliberately different function
//!   (`x * 0.5`) must be reported as differing. If the comparison cannot see a
//!   difference that large, it cannot see a one-ulp one either.
//!
//! ```sh
//! cargo run --release --example silu_spelling_divergence -p candle-metal-kernels
//! ```

use half::f16;
use lloom_tensor::kernels::metal::{Device, Library};
use lloom_tensor::kernels::RESOURCE_OPTIONS;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// The spellings, as Metal source.
///
/// Templated on `T` so one text serves both precisions. Every body is copied
/// from its site rather than rewritten.
const SPELLINGS_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `unary.metal:304-306`, verbatim.
template <typename T> METAL_FUNC T recip_(T x) {
    return static_cast<T>(1.0 / x);
}

// A: `unary.metal:326` and `gemv.metal:157` -- the same expression at both.
template <typename T> METAL_FUNC T s_div(T x) {
    return static_cast<T>(x / (1 + exp(-x)));
}

// B: `unary.metal:307-309`'s sigmoid spelling, composed into a swish.
//    `usigmoid` is `recip(1 + exp(-x))`; SiLU from it is that times x.
template <typename T> METAL_FUNC T s_recip(T x) {
    return static_cast<T>(x * static_cast<T>(recip_(static_cast<T>(1 + exp(-x)))));
}

// C: the spelling `gemv.metal:128` explicitly names as a DIFFERENT float
//    operation. Not present in the tree; included because it is the one the
//    warning is about.
template <typename T> METAL_FUNC T s_mul_reciprocal(T x) {
    return static_cast<T>(x * (1 / (1 + exp(-x))));
}

// D: `quantized.metal:1563` -- `1.0f` literal rather than `1`.
template <typename T> METAL_FUNC T s_literal_f(T x) {
    return static_cast<T>(x / (1.0f + exp(-x)));
}

// E: `quantized.metal:1503`'s sigmoid, composed into a swish.
template <typename T> METAL_FUNC T s_sigmoid_then_mul(T x) {
    return static_cast<T>(static_cast<T>(1.0f / (1.0f + exp(-x))) * x);
}

// The divergence control: obviously not SiLU. If the comparison reports this
// as identical to A, the comparison is broken.
template <typename T> METAL_FUNC T s_control_half(T x) {
    return static_cast<T>(x * 0.5);
}

#define INSTANTIATE(NAME, FN, T, SUFFIX)                       \
kernel void NAME##_##SUFFIX(                                   \
    device const T *inp [[buffer(0)]],                         \
    device T *out       [[buffer(1)]],                         \
    uint gid [[thread_position_in_grid]]                       \
) { out[gid] = FN<T>(inp[gid]); }

INSTANTIATE(a_div, s_div, float, f32)
INSTANTIATE(b_recip, s_recip, float, f32)
INSTANTIATE(c_mul_reciprocal, s_mul_reciprocal, float, f32)
INSTANTIATE(d_literal_f, s_literal_f, float, f32)
INSTANTIATE(e_sigmoid_then_mul, s_sigmoid_then_mul, float, f32)
INSTANTIATE(z_identity, s_div, float, f32)
INSTANTIATE(z_control, s_control_half, float, f32)

INSTANTIATE(a_div, s_div, half, f16)
INSTANTIATE(b_recip, s_recip, half, f16)
INSTANTIATE(c_mul_reciprocal, s_mul_reciprocal, half, f16)
INSTANTIATE(d_literal_f, s_literal_f, half, f16)
INSTANTIATE(e_sigmoid_then_mul, s_sigmoid_then_mul, half, f16)
INSTANTIATE(z_identity, s_div, half, f16)
INSTANTIATE(z_control, s_control_half, half, f16)
"#;

/// The five spellings, in report order. `a_div` is the baseline every other
/// arm is compared against, because it is what both live sites spell.
const ARMS: &[(&str, &str)] = &[
    (
        "a_div",
        "x / (1 + exp(-x))            unary.metal:326, gemv.metal:157",
    ),
    (
        "b_recip",
        "x * recip(1 + exp(-x))        unary.metal:308 (sigmoid) * x",
    ),
    (
        "c_mul_reciprocal",
        "x * (1 / (1 + exp(-x)))       gemv.metal:128's named alternative",
    ),
    (
        "d_literal_f",
        "x / (1.0f + exp(-x))          quantized.metal:1563",
    ),
    (
        "e_sigmoid_then_mul",
        "(1.0f/(1.0f+exp(-x))) * x     quantized.metal:1503 composed",
    ),
];

/// The f64 reference. Not one of the orderings being compared (§10.2g).
fn reference(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// A sweep wide enough to include the regions where the spellings could
/// plausibly differ: large negative x (where `exp(-x)` is large and the
/// division is by a big number), the origin, and the near-identity tail.
fn inputs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = (i as f64) / ((n - 1) as f64);
            -20.0 + t * 40.0
        })
        .collect()
}

/// Below this input, `exp(-x)` exceeds f16's 65504 maximum and every spelling
/// returns `x/inf == 0`.
///
/// **This is an f16 RANGE limit, not a spelling difference**, and it must be
/// excluded from the f16 comparison or it dominates it: the relative error is
/// 1.0 by construction at every point in the region, for every arm equally, so
/// a headline computed over the whole sweep reports the overflow rather than
/// the thing being measured. `exp(11.09) = 65504`, so the boundary is x =
/// -11.09; -11.0 is the conservative side of it.
///
/// Reported rather than silently trimmed (§1.1), and the f32 arm keeps the
/// whole sweep because `exp(20)` is nowhere near f32's range.
const F16_EXP_OVERFLOW_BELOW: f64 = -11.0;

fn dispatch<T: Copy>(device: &Device, library: &Library, name: &str, input: &[T]) -> Vec<T> {
    let func = library
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("kernel {name} must exist in the probe library: {e}"));
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|e| panic!("pipeline for {name}: {e}"));

    let nbytes = std::mem::size_of_val(input);
    let inbuf = device
        .new_buffer_with_data(
            input.as_ptr() as *const core::ffi::c_void,
            nbytes,
            RESOURCE_OPTIONS,
        )
        .expect("input buffer");
    let outbuf = device
        .new_buffer(nbytes, RESOURCE_OPTIONS)
        .expect("output buffer");

    let queue = device.new_command_queue().expect("command queue");
    let cb = queue.commandBuffer().expect("command buffer");
    let enc = cb.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(pipeline.as_ref());
    // SAFETY: both buffers are this function's and outlive the wait below;
    // nothing else binds or writes them.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(inbuf.raw_unchecked()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(outbuf.raw_unchecked()), 0, 1);
    }
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: input.len(),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    unsafe { cb.waitUntilCompleted() };

    // SAFETY: Shared, `input.len()` elements, read after the wait above.
    let ptr = unsafe { outbuf.contents() } as *const T;
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let library = device
        .new_library_with_source(SPELLINGS_SRC, None)
        .expect("the spelling probe library must compile");

    let n = 4096usize;
    let xs = inputs(n);

    println!("# SiLU spelling divergence -- lloom issue #599");
    println!("# n = {n} inputs, linear sweep over [-20, 20]");
    println!("# reference: f64 x/(1+exp(-x)), computed on the CPU");
    println!();

    let xs_f32: Vec<f32> = xs.iter().map(|v| *v as f32).collect();
    let xs_f16: Vec<f16> = xs.iter().map(|v| f16::from_f64(*v)).collect();

    // ---- controls first (§3.1a rule 1: the clean arm before the results) ----
    println!("## Controls");
    println!();
    let a32 = dispatch::<f32>(&device, &library, "a_div_f32", &xs_f32);
    let id32 = dispatch::<f32>(&device, &library, "z_identity_f32", &xs_f32);
    let ct32 = dispatch::<f32>(&device, &library, "z_control_f32", &xs_f32);

    let identity_ok = a32 == id32;
    let control_differs = a32 != ct32;
    println!("| control | expected | observed |");
    println!("|---|---|---|");
    println!(
        "| identity: same body, two names | bit-identical | {} |",
        if identity_ok {
            "bit-identical ✓"
        } else {
            "**DIFFERS — harness unreliable**"
        }
    );
    println!(
        "| divergence: `x*0.5` vs SiLU | differs | {} |",
        if control_differs {
            "differs ✓"
        } else {
            "**identical — comparison is blind**"
        }
    );
    println!();
    assert!(
        identity_ok,
        "identity control failed: the same source compiled under two names gave \
         different results, so no row below is readable"
    );
    assert!(
        control_differs,
        "divergence control failed: the comparison cannot distinguish x*0.5 from \
         SiLU, so it cannot distinguish a one-ulp difference either"
    );

    report::<f32>(
        "f32",
        &xs,
        &a32,
        &ARMS
            .iter()
            .map(|(k, _)| dispatch::<f32>(&device, &library, &format!("{k}_f32"), &xs_f32))
            .collect::<Vec<_>>(),
        None,
    );

    println!();

    let a16 = dispatch::<f16>(&device, &library, "a_div_f16", &xs_f16);
    report::<f16>(
        "f16",
        &xs,
        &a16,
        &ARMS
            .iter()
            .map(|(k, _)| dispatch::<f16>(&device, &library, &format!("{k}_f16"), &xs_f16))
            .collect::<Vec<_>>(),
        Some(F16_EXP_OVERFLOW_BELOW),
    );
}

/// One dtype's table. Generic over the element so f32 and f16 share one body
/// and cannot drift in what they report.
trait Elem: Copy + PartialEq {
    fn to_f64(self) -> f64;
    fn ulps_from(self, want: f64) -> u64;
}

impl Elem for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn ulps_from(self, want: f64) -> u64 {
        let w = want as f32;
        (w.to_bits() as i64 - self.to_bits() as i64).unsigned_abs()
    }
}

impl Elem for f16 {
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
    fn ulps_from(self, want: f64) -> u64 {
        let w = f16::from_f64(want);
        (w.to_bits() as i64 - self.to_bits() as i64).unsigned_abs()
    }
}

/// One dtype's table.
///
/// `skip_below` excludes the f16 `exp` overflow region from the accuracy
/// columns; it is `None` at f32. The **bit-identity** columns are always
/// computed over the WHOLE sweep, because two spellings agreeing inside the
/// overflow region is a real agreement (they both return zero) and excluding
/// it could only ever hide a difference.
fn report<T: Elem>(label: &str, xs: &[f64], base: &[T], arms: &[Vec<T>], skip_below: Option<f64>) {
    println!("## {label}");
    println!();
    if let Some(b) = skip_below {
        let excluded = xs.iter().filter(|x| **x < b).count();
        println!(
            "Accuracy columns exclude x < {b} — {excluded} of {} points where `exp(-x)` \
             overflows f16 (max 65504) and every arm returns `x/inf == 0`, giving a \
             relative error of 1.0 for all of them equally. That is an f16 RANGE limit, \
             not a spelling difference. Bit-identity columns cover the whole sweep.",
            xs.len()
        );
        println!();
    }
    println!("| arm | spelling and site | max rel err vs f64 | max ulp vs f64 | bit-identical to A? | max ulp vs A |");
    println!("|---|---|---|---|---|---|");
    for (i, (name, site)) in ARMS.iter().enumerate() {
        let got = &arms[i];
        let mut worst_rel = 0.0f64;
        let mut worst_ulp = 0u64;
        let mut worst_vs_a = 0u64;
        for (j, x) in xs.iter().enumerate() {
            // Bit-identity: always the whole sweep.
            worst_vs_a = worst_vs_a.max(got[j].ulps_from(base[j].to_f64()));
            if skip_below.is_some_and(|b| *x < b) {
                continue;
            }
            let want = reference(*x);
            let denom = want.abs().max(1e-6);
            worst_rel = worst_rel.max((got[j].to_f64() - want).abs() / denom);
            worst_ulp = worst_ulp.max(got[j].ulps_from(want));
        }
        let identical = got == base;
        println!(
            "| `{name}` | `{site}` | {worst_rel:.3e} | {worst_ulp} | {} | {worst_vs_a} |",
            if identical { "YES" } else { "**no**" }
        );
    }

    // Where a differing arm differs, and by how many points. A max-ulp on its
    // own does not say whether the difference is one stray point or the whole
    // sweep, and those are different findings.
    for (i, (name, _)) in ARMS.iter().enumerate() {
        let got = &arms[i];
        if got == base {
            continue;
        }
        let diffs: Vec<usize> = (0..xs.len()).filter(|j| got[*j] != base[*j]).collect();
        let first = diffs.first().copied().unwrap_or(0);
        let last = diffs.last().copied().unwrap_or(0);
        println!();
        println!(
            "`{name}` differs from A at **{} of {} points**, x ∈ [{:.3}, {:.3}]. \
             First: x = {:.4}, A = {:.6e}, {name} = {:.6e}.",
            diffs.len(),
            xs.len(),
            xs[first],
            xs[last],
            xs[first],
            base[first].to_f64(),
            got[first].to_f64()
        );
    }
}
