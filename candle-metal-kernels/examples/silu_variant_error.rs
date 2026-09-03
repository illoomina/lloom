//! How much error does each `SiluImpl` arm introduce, and do its names exist?
//!
//! Measurement instrument for lloom issue #599, second half.
//!
//! # What this measures and why it is a different probe from its sibling
//!
//! `silu_spelling_divergence.rs` (#604, `DESIGN.md` §12.2k) asked whether the
//! spellings ALREADY IN THE TREE disagree, and answered no: bit-identical at
//! f32, 0-3 ulp apart at f16. That is the premise check, and it is done.
//!
//! This asks the question #599's title asks, whose answer that null makes
//! necessary rather than unnecessary: **how much error does the model absorb
//! before it degenerates?** A spread of 0-3 ulp cannot locate a threshold, so
//! `unary.metal` now carries four further arms built to span it -- from 1 ulp
//! to a piecewise-linear approximation -- and this probe measures where each
//! one sits on that ladder. The end-to-end half is
//! `measurements/issue-599-raw/absorption.sh`; this is the per-kernel half that
//! tells it what the x-axis is.
//!
//! # Why the SHIPPING library and not a private one
//!
//! Its sibling compiles a private library deliberately, to hold precision
//! constant while varying spelling. This probe does the opposite and for the
//! opposite reason: the arms it measures are the ones a decode token will
//! actually dispatch, so measuring a private copy would measure a paraphrase
//! (§3.1c) and could not detect the failure that matters most here -- **a
//! variant whose `[[host_name]]` is not in the compiled library**.
//!
//! #26 shipped 48 names absent from a metallib that compiled cleanly, and
//! `DESIGN.md` §8.1b is the standing argument for checking a name against the
//! artifact rather than against the macro that should have produced it. Every
//! arm here is resolved through `Source::Unary` by the same
//! `[[host_name]]` a dispatch would ask for, so a missing instantiation is a
//! panic in this probe rather than a `LoadFunctionError` in a decode run.
//!
//! # Non-vacuity
//!
//! Three controls, asserted rather than printed (§3.1a rule 4):
//!
//! * **identity** -- `Exact` dispatched twice must be bit-identical. If it is
//!   not, nothing else here is readable.
//! * **the `Recip` arm must read ZERO.** #604 measured it bit-identical to
//!   `Exact` at both dtypes, so it is carried as an internal control rather
//!   than as a rung: a ladder whose every arm differs cannot demonstrate that
//!   the instrument reports no difference when there is none.
//! * **divergence** -- `HardSwish` must differ. If the comparison cannot see
//!   the coarsest arm, it cannot see the finest.
//!
//! ```sh
//! cargo run --release --example silu_variant_error -p candle-metal-kernels
//! ```

use candle_metal_kernels::metal::Device;
use candle_metal_kernels::source::Source;
use candle_metal_kernels::{Kernels, SiluDType, SiluImpl, RESOURCE_OPTIONS};
use half::f16;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// The f64 reference. True SiLU, computed on the CPU, and not one of the
/// implementations being compared (§10.2g).
fn reference(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// The sweep. Linear over [-20, 20], matching the sibling probe so the two
/// tables are read on one x-axis.
fn inputs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| -20.0 + (i as f64) / ((n - 1) as f64) * 40.0)
        .collect()
}

/// Below this input `exp(-x)` exceeds f16's 65504 maximum, every exp-based arm
/// returns `x/inf == 0`, and the relative error pins to 1.0 for all of them
/// equally.
///
/// **This is an f16 RANGE limit, not a variant difference**, and #604 recorded
/// what happens without the exclusion: the headline read `1.000e0` / 2442 ulp
/// and looked like wild divergence, where separating the region gave
/// 4.027e-3 / 7 ulp and the actual answer. 922 of 4096 points.
///
/// **The region is reported separately rather than only excluded**, because an
/// exclusion a reader cannot see the inside of is an exclusion they have to
/// take on trust. Measured, every arm reads 1.7-1.8e-4 there and the ladder is
/// flat: SiLU itself is ~1e-4 by x = -11, so all six arms are near zero and
/// none of them is distinguishable. That is the honest reason the region
/// carries no information about the variants — not that some arms avoid the
/// overflow.
const F16_EXP_OVERFLOW_BELOW: f64 = -11.0;

fn dispatch<T: Copy>(
    device: &Device,
    kernels: &Kernels,
    name: &'static str,
    input: &[T],
) -> Result<Vec<T>, String> {
    let pipeline = kernels
        .load_pipeline(device, Source::Unary, name)
        .map_err(|e| format!("{name}: {e}"))?;

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
    // `unary_kernel`'s contiguous signature: (dim, input, output).
    let dim = input.len() as u32;
    unsafe {
        enc.setBytes_length_atIndex(
            core::ptr::NonNull::from(&dim).cast(),
            std::mem::size_of::<u32>(),
            0,
        );
        enc.setBuffer_offset_atIndex(Some(inbuf.as_ref()), 0, 1);
        enc.setBuffer_offset_atIndex(Some(outbuf.as_ref()), 0, 2);
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

    let ptr = outbuf.contents() as *const T;
    Ok(unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec())
}

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

/// One arm's error summary, in the units the absorption sweep needs.
struct Row {
    arm: SiluImpl,
    max_abs: f64,
    max_rel: f64,
    max_ulp_vs_ref: u64,
    max_ulp_vs_exact: u64,
    differing_points: usize,
    identical_to_exact: bool,
}

fn measure<T: Elem>(
    xs: &[f64],
    exact: &[T],
    got: &[T],
    skip_below: Option<f64>,
    arm: SiluImpl,
) -> Row {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_ulp_vs_ref = 0u64;
    let mut max_ulp_vs_exact = 0u64;
    let mut differing = 0usize;
    for (j, x) in xs.iter().enumerate() {
        // Bit-identity spans the whole sweep: two arms agreeing inside the
        // overflow region is a real agreement, and excluding it could only
        // hide a difference.
        let d = got[j].ulps_from(exact[j].to_f64());
        max_ulp_vs_exact = max_ulp_vs_exact.max(d);
        if got[j] != exact[j] {
            differing += 1;
        }
        if skip_below.is_some_and(|b| *x < b) {
            continue;
        }
        let want = reference(*x);
        max_abs = max_abs.max((got[j].to_f64() - want).abs());
        max_rel = max_rel.max((got[j].to_f64() - want).abs() / want.abs().max(1e-6));
        max_ulp_vs_ref = max_ulp_vs_ref.max(got[j].ulps_from(want));
    }
    Row {
        arm,
        max_abs,
        max_rel,
        max_ulp_vs_ref,
        max_ulp_vs_exact,
        differing_points: differing,
        identical_to_exact: got == exact,
    }
}

fn report(label: &str, xs: &[f64], rows: &[Row], skip_below: Option<f64>) {
    println!("## {label}");
    println!();
    if let Some(b) = skip_below {
        let excluded = xs.iter().filter(|x| **x < b).count();
        println!(
            "Accuracy columns exclude x < {b} — {excluded} of {} points where `exp(-x)` \
             overflows f16 (max 65504). Bit-identity and differing-point columns cover the \
             whole sweep.",
            xs.len()
        );
        println!();
    }
    println!(
        "| arm | max abs err vs f64 | max rel err | max ulp vs f64 | ulp vs `Exact` | points differing | bit-identical |"
    );
    println!("|---|---|---|---|---|---|---|");
    for r in rows {
        println!(
            "| `{}` | {:.4e} | {:.3e} | {} | {} | {} / {} | {} |",
            r.arm.name(),
            r.max_abs,
            r.max_rel,
            r.max_ulp_vs_ref,
            r.max_ulp_vs_exact,
            r.differing_points,
            xs.len(),
            if r.identical_to_exact {
                "YES"
            } else {
                "**no**"
            }
        );
    }
    println!();
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new();

    let n = 4096usize;
    let xs = inputs(n);
    let xs_f32: Vec<f32> = xs.iter().map(|v| *v as f32).collect();
    let xs_f16: Vec<f16> = xs.iter().map(|v| f16::from_f64(*v)).collect();

    println!("# SiLU variant error ladder — lloom issue #599");
    println!("# n = {n} inputs, linear sweep over [-20, 20]");
    println!("# reference: f64 x/(1+exp(-x)) on the CPU");
    println!("# arms resolved through Source::Unary by [[host_name]] — the SHIPPING library");
    println!();

    // ---- name resolution, before any measurement -------------------------
    //
    // §8.1b: check the name against the compiled artifact, not against the
    // macro that should have produced it. A variant that does not resolve is
    // reported here as a table row rather than as a panic three sections down.
    println!("## Name resolution against the compiled library");
    println!();
    println!("| arm | contiguous name | resolves | strided name | resolves |");
    println!("|---|---|---|---|---|");
    let mut all_resolve = true;
    for arm in SiluImpl::ALL {
        let c = arm.contiguous_name(SiluDType::F16);
        let s = arm.strided_name(SiluDType::F16);
        let cok = kernels.load_pipeline(&device, Source::Unary, c).is_ok();
        let sok = kernels.load_pipeline(&device, Source::Unary, s).is_ok();
        all_resolve &= cok && sok;
        println!(
            "| `{}` | `{c}` | {} | `{s}` | {} |",
            arm.name(),
            if cok { "yes" } else { "**NO**" },
            if sok { "yes" } else { "**NO**" }
        );
    }
    println!();
    // A positive control on the resolution check itself: a name that must NOT
    // exist. Without it, a `load_pipeline` that returned `Ok` for everything
    // would print a clean column of "yes" and read as a strong result (#517 --
    // put a positive control in any census, or the zeros mean nothing).
    let bogus = kernels
        .load_pipeline(&device, Source::Unary, "silu_nosuchvariant_f16")
        .is_ok();
    println!(
        "Positive control: `silu_nosuchvariant_f16` resolves = {} (must be false, or the \
         column above is blind).",
        bogus
    );
    println!();
    assert!(
        !bogus,
        "resolution check is blind: a nonexistent name resolved"
    );
    assert!(
        all_resolve,
        "a variant name is absent from the compiled library"
    );

    // ---- controls --------------------------------------------------------
    println!("## Controls");
    println!();
    let exact32 = dispatch::<f32>(&device, &kernels, "silu_f32", &xs_f32).expect("silu_f32");
    let exact32b = dispatch::<f32>(&device, &kernels, "silu_f32", &xs_f32).expect("silu_f32 again");
    let recip32 = dispatch::<f32>(&device, &kernels, "silu_recip_f32", &xs_f32).expect("recip f32");
    let hard32 =
        dispatch::<f32>(&device, &kernels, "silu_hardswish_f32", &xs_f32).expect("hardswish f32");

    let identity_ok = exact32 == exact32b;
    let recip_zero = exact32 == recip32;
    let hard_differs = exact32 != hard32;
    println!("| control | expected | observed |");
    println!("|---|---|---|");
    println!(
        "| identity: `Exact` dispatched twice | bit-identical | {} |",
        if identity_ok {
            "bit-identical ✓"
        } else {
            "**DIFFERS**"
        }
    );
    println!(
        "| `Recip` reads zero (#604's null, reproduced) | bit-identical | {} |",
        if recip_zero {
            "bit-identical ✓"
        } else {
            "**DIFFERS**"
        }
    );
    println!(
        "| divergence: `HardSwish` vs `Exact` | differs | {} |",
        if hard_differs {
            "differs ✓"
        } else {
            "**identical — blind**"
        }
    );
    println!();
    assert!(
        identity_ok,
        "identity control failed: the harness is unreliable"
    );
    assert!(
        hard_differs,
        "divergence control failed: the comparison is blind"
    );
    // `Recip` is asserted too: it is #604's measured null, so a non-zero here
    // means either this probe or that measurement is wrong, and continuing
    // would report a ladder whose bottom rung is unexplained.
    //
    // **Asserted at f16 as well as f32, and the f16 half is the load-bearing
    // one.** The first version of this probe checked only f32, and the #599
    // mutation harness caught it: giving `Exact` the `1.0f` literal moves
    // `Recip` to 1 ulp over 440 points **at f16 only** -- f32 is a clean sweep
    // of zeros because at `T = float` the two are the same expression (§12.2k).
    // So the f32-only control PASSED under a mutation that had visibly moved
    // the arm it was watching, and the probe printed "All controls asserted"
    // beside a table showing `recip` differing. That is §3.1b's shape exactly:
    // the control was checking the axis on which the effect cannot appear.
    let recip16 = dispatch::<f16>(&device, &kernels, "silu_recip_f16", &xs_f16).expect("recip f16");
    let exact16_ctl = dispatch::<f16>(&device, &kernels, "silu_f16", &xs_f16).expect("exact f16");
    assert!(
        recip_zero && recip16 == exact16_ctl,
        "the Recip control moved — it was bit-identical at BOTH dtypes in #604"
    );

    // ---- the ladder ------------------------------------------------------
    let mut rows32 = Vec::new();
    for arm in SiluImpl::ALL {
        let got = dispatch::<f32>(
            &device,
            &kernels,
            arm.contiguous_name(SiluDType::F32),
            &xs_f32,
        )
        .expect("f32 arm");
        rows32.push(measure(&xs, &exact32, &got, None, arm));
    }
    report("f32", &xs, &rows32, None);

    let exact16 = dispatch::<f16>(&device, &kernels, "silu_f16", &xs_f16).expect("silu_f16");
    let mut rows16 = Vec::new();
    for arm in SiluImpl::ALL {
        let got = dispatch::<f16>(
            &device,
            &kernels,
            arm.contiguous_name(SiluDType::F16),
            &xs_f16,
        )
        .expect("f16 arm");
        rows16.push(measure(
            &xs,
            &exact16,
            &got,
            Some(F16_EXP_OVERFLOW_BELOW),
            arm,
        ));
    }
    report(
        "f16 — the precision LFM2 decodes in",
        &xs,
        &rows16,
        Some(F16_EXP_OVERFLOW_BELOW),
    );

    // ---- the overflow region, reported rather than only excluded ---------
    //
    // The exp-free arms are BETTER here and the table above cannot say so,
    // because the region is excluded from it. Reported because an absorption
    // curve that treats "coarser" as monotone would be wrong about this
    // region, and because a reader comparing the two tables would otherwise
    // find the exclusion unexplained.
    println!("## The f16 overflow region, x < {F16_EXP_OVERFLOW_BELOW}");
    println!();
    println!(
        "Excluded from the accuracy columns above, and printed here so the exclusion can be \
         checked rather than trusted. SiLU itself is ~1e-4 by this point, so every arm is near \
         zero and the ladder is FLAT here — the region carries no information about the \
         variants, which is why excluding it does not hide one."
    );
    println!();
    println!("| arm | max abs err vs f64, x < {F16_EXP_OVERFLOW_BELOW} |");
    println!("|---|---|");
    for arm in SiluImpl::ALL {
        let got = dispatch::<f16>(
            &device,
            &kernels,
            arm.contiguous_name(SiluDType::F16),
            &xs_f16,
        )
        .expect("f16 arm");
        let mut worst = 0.0f64;
        for (j, x) in xs.iter().enumerate() {
            if *x >= F16_EXP_OVERFLOW_BELOW {
                continue;
            }
            worst = worst.max((got[j].to_f64() - reference(*x)).abs());
        }
        println!("| `{}` | {:.4e} |", arm.name(), worst);
    }
    println!();
    println!("All controls asserted. Every arm resolved against the compiled library.");
}
