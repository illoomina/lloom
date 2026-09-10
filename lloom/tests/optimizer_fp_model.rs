//! The target fp model's build half, measured (`design/docs/optimizer-design.md`
//! §5a; `lloom::optimizer::numeric::build_model`): what this build's Metal
//! kernels do with a denormal, at each width, asserted against the
//! declaration every exclusion entry carries. The corpus declares contraction
//! (`DESIGN.md` §2.3.3b), the accumulator widths and the reduction trees per
//! kernel; the denormal mode was the one of §5a's four it had never pinned,
//! and pinned it turned out to be a property of the WIDTH, not of the math
//! mode — which is why the model carries two.
//!
//! Three arithmetic sites, each with a subnormal RESULT (normal operands whose
//! product is subnormal) and a subnormal OPERAND: `affine` (float `fma`, cast
//! to `T`) and the binary `mul` at `T`, at f32 and f16, with the CPU backend
//! beside them as the reference that preserves both. Run under both math
//! modes (`CANDLE_METAL_ENABLE_FAST_MATH=0` for the other) — the declaration
//! is the same under each and the test says which it ran under.

#![cfg(feature = "metal")]

use lloom::optimizer::numeric::{build_math_mode, build_model, Denormals};
use lloom_tensor::{DType, Device, Result, Tensor};

/// `(subnormal result preserved, subnormal operand preserved)` for one op on
/// one dtype.
fn probe(dev: &Device, dtype: DType, op: &str) -> Result<(bool, bool)> {
    // f32: min normal 1.1755e-38. f16: min normal 6.1035e-5.
    let (normal_in, mul_to_sub, sub_in, mul_up) = match dtype {
        DType::F32 => (2.0e-38f64, 0.5f64, 1.0e-39f64, 4.0f64),
        DType::F16 => (1.0e-4f64, 0.25f64, 3.0e-5f64, 4.0f64),
        _ => unreachable!(),
    };
    let run = |x: f64, m: f64| -> Result<f64> {
        // Built on the CPU at the probe's dtype, then moved: Metal has no
        // f64, and the cast from f64 is the CPU's.
        let mk = |v: f64| -> Result<Tensor> {
            Tensor::new(&[v as f32], &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(dev)
        };
        let t = mk(x)?;
        let y = match op {
            "affine" => t.affine(m, 0.0)?,
            "mul" => (&t * &mk(m)?)?,
            _ => unreachable!(),
        };
        // Read back through f32 (exact for every f16 value; Metal has no f64).
        Ok(y.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as f64)
    };
    let result = run(normal_in, mul_to_sub)?;
    let operand = run(sub_in, mul_up)?;
    let expect_result = normal_in * mul_to_sub;
    let expect_operand = sub_in * mul_up;
    eprintln!(
        "{dtype:?} {op}: normal {normal_in:e} * {mul_to_sub} = {result:e} (exact {expect_result:e}); \
         subnormal {sub_in:e} * {mul_up} = {operand:e} (exact {expect_operand:e})"
    );
    // A preserved subnormal is within a few ulp of the exact product at this
    // scale; a flush is exactly zero. Nothing in between is a third answer.
    let preserved = |got: f64, exact: f64| -> bool {
        if got == 0.0 {
            false
        } else {
            assert!(
                (got - exact).abs() <= exact * 0.1,
                "{dtype:?} {op}: {got:e} is neither zero nor the product {exact:e}"
            );
            true
        }
    };
    Ok((
        preserved(result, expect_result),
        preserved(operand, expect_operand),
    ))
}

/// **The declaration is the measurement.** f32 flushes (operand and result,
/// on the float `fma` path and on the `T`-typed multiply), f16 preserves,
/// under whichever math mode this process runs — `build_model()` says so,
/// and a build where that stops being true fails here rather than silently
/// re-keying every recorded verdict.
#[test]
fn what_a_denormal_becomes_under_this_build_is_what_the_model_declares() -> Result<()> {
    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let cpu = Device::Cpu;
    let declared = build_model();
    eprintln!(
        "math mode {} (CANDLE_METAL_ENABLE_FAST_MATH={:?}); declared {declared}",
        build_math_mode(),
        std::env::var("CANDLE_METAL_ENABLE_FAST_MATH").ok()
    );
    for dtype in [DType::F32, DType::F16] {
        let expect = match dtype {
            DType::F32 => declared.denormals_f32,
            _ => declared.denormals_f16,
        };
        for op in ["affine", "mul"] {
            let (r_cpu, o_cpu) = probe(&cpu, dtype, op)?;
            assert!(
                r_cpu && o_cpu,
                "the CPU reference preserves subnormals; the probe's inputs are wrong otherwise"
            );
            let (r, o) = probe(&metal, dtype, op)?;
            let measured = match (r, o) {
                (true, true) => Denormals::Preserve,
                (false, false) => Denormals::Flush,
                other => panic!("{dtype:?} {op}: result/operand disagree {other:?}; the model has no spelling for that"),
            };
            eprintln!("DENORMAL {dtype:?} {op}: measured {measured:?}, declared {expect:?}");
            assert_eq!(
                measured, expect,
                "{dtype:?} {op}: the build's denormal mode is not what `build_model` declares"
            );
        }
    }
    Ok(())
}
