#![cfg(feature = "metal")]
//! An emulated dtype arm on Metal is refused unless the operator opts in, and
//! an admitted one warns once (`DESIGN.md` §15.2a).
//!
//! `bfloat` arithmetic runs emulated on Apple7. Where the device's families
//! register `bf16` as native, the gate has nothing to refuse, and these tests
//! assert that instead of the refusal.
//!
//! Every test builds its own device, so an opt-in scoped to one device
//! (`MetalDevice::allow_emulated`) cannot reach a sibling test's. The process-wide
//! opt-in (the tuning key and its env alias) is exercised in a child process,
//! because the tuning group is read once per process.

use lloom_tensor::kernels::metal::arms::{Feature, Features, ALLOW_EMULATED_ENV};
use lloom_tensor::{DType, Device, Result, Tensor};

const WARNING: &str = "warning: Metal `bf16` runs emulated in software";

fn device() -> Result<Device> {
    Device::new_metal(0)
}

fn bf16_is_native(dev: &Device) -> Result<bool> {
    Ok(dev
        .as_metal_device()?
        .emulated_arms()
        .native()
        .has(Feature::Bf16))
}

fn opted_in_by_the_process() -> bool {
    lloom_tensor::tuning::tuning()
        .metal
        .allow_emulated
        .is_some()
}

/// `a + b`, `a * b` and `a @ b` at bf16 on `dev`, read back as f32. The CPU
/// has no bf16 matmul, so on the CPU the product is taken at f32 over the same
/// bf16 inputs and rounded to bf16, which is what the Metal kernel's float
/// accumulator and bf16 store compute on these exactly representable values.
fn bf16_ops(dev: &Device) -> Result<Vec<Vec<f32>>> {
    let a = Tensor::new(&[[1f32, 2.], [3., 4.]], dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::new(&[[0.5f32, -1.], [2., 0.25]], dev)?.to_dtype(DType::BF16)?;
    let product = if dev.is_cpu() {
        let (a, b) = (a.to_dtype(DType::F32)?, b.to_dtype(DType::F32)?);
        a.matmul(&b)?.to_dtype(DType::BF16)?
    } else {
        a.matmul(&b)?
    };
    [(&a + &b)?, (&a * &b)?, product]
        .iter()
        .map(|t| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
        .collect()
}

/// Without the opt-in, a bf16 result is refused: the cast into bf16, and an op
/// on bf16 data that was only uploaded. The refusal names why and the opt-in.
#[test]
fn a_bf16_op_is_refused_without_the_opt_in() -> Result<()> {
    assert!(
        !opted_in_by_the_process(),
        "run without {ALLOW_EMULATED_ENV}: this test is the default arm"
    );
    let dev = device()?;
    if bf16_is_native(&dev)? {
        bf16_ops(&dev)?;
        assert_eq!(dev.as_metal_device()?.emulated_arms().warnings(), 0);
        eprintln!("bf16 is native on this GPU: nothing to refuse");
        return Ok(());
    }
    let cast = Tensor::new(&[1f32, 2., 3.], &dev)?.to_dtype(DType::BF16);
    let err = cast.unwrap_err().to_string();
    assert!(
        err.contains("Metal arm `a `bf16` op` refused: this GPU does not execute `bf16` natively"),
        "{err}"
    );
    for opt_in in [
        "`--allow-emulated bf16` on `lloom chat|serve`",
        "`tuning.metal.allow_emulated=bf16`",
        "`LLOOM_METAL_ALLOW_EMULATED=bf16`",
        "about 2.6x the `float` time",
    ] {
        assert!(err.contains(opt_in), "missing {opt_in:?}: {err}");
    }

    let host = [half::bf16::from_f32(1.5), half::bf16::from_f32(-2.0)];
    let uploaded = Tensor::new(&host, &dev)?;
    let add = (&uploaded + &uploaded).unwrap_err().to_string();
    assert!(add.contains("refused"), "{add}");
    assert_eq!(dev.as_metal_device()?.emulated_arms().warnings(), 0);
    Ok(())
}

/// An upload at bf16 and its widening to f16 or f32 are not bf16 arithmetic:
/// a bf16 checkpoint loaded at f16 (the DSpark drafter's load) runs without the
/// opt-in and warns nothing.
#[test]
fn a_bf16_upload_widened_to_f16_needs_no_opt_in() -> Result<()> {
    assert!(
        !opted_in_by_the_process(),
        "run without {ALLOW_EMULATED_ENV}"
    );
    let dev = device()?;
    let host: Vec<half::bf16> = [1.5f32, -2.0, 0.1, 1e4]
        .iter()
        .map(|&x| half::bf16::from_f32(x))
        .collect();
    let on_gpu = Tensor::new(host.as_slice(), &dev)?;
    let on_cpu = Tensor::new(host.as_slice(), &Device::Cpu)?;
    for dtype in [DType::F16, DType::F32] {
        let gpu = on_gpu
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let cpu = on_cpu
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(gpu, cpu, "{dtype:?}");
    }
    assert_eq!(dev.as_metal_device()?.emulated_arms().warnings(), 0);
    Ok(())
}

/// With the opt-in the arm runs as it did before the gate, matching the CPU's
/// bf16 to the bit, and the warning is printed once however many ops ran.
#[test]
fn a_bf16_op_runs_with_the_opt_in_and_warns_once() -> Result<()> {
    let dev = device()?;
    let metal = dev.as_metal_device()?;
    metal.allow_emulated(Features::empty().with(Feature::Bf16));
    let got = bf16_ops(&dev)?;
    let again = bf16_ops(&dev)?;
    let want = bf16_ops(&Device::Cpu)?;
    assert_eq!(got, want);
    assert_eq!(again, want);
    let expected = if bf16_is_native(&dev)? { 0 } else { 1 };
    assert_eq!(metal.emulated_arms().warnings(), expected);
    Ok(())
}

/// A load at bf16 is refused before its first weight (`admit_compute_dtype`,
/// the check `VarBuilder` makes), and admitted once allowed, with one warning.
/// f16 and f32 are admitted either way and warn nothing.
#[test]
fn a_load_at_bf16_is_admitted_only_by_the_opt_in() -> Result<()> {
    assert!(
        !opted_in_by_the_process(),
        "run without {ALLOW_EMULATED_ENV}"
    );
    let dev = device()?;
    let metal = dev.as_metal_device()?;
    dev.admit_compute_dtype(DType::F16)?;
    dev.admit_compute_dtype(DType::F32)?;
    Device::Cpu.admit_compute_dtype(DType::BF16)?;
    let native = bf16_is_native(&dev)?;
    // `supports_bf16` answers from the same gate, so a caller choosing its
    // dtype there (`bf16_default_to_f32`) is not handed one that is refused.
    assert_eq!(dev.supports_bf16(), native);
    assert_eq!(
        dev.bf16_default_to_f32(),
        if native { DType::BF16 } else { DType::F32 }
    );
    match dev.admit_compute_dtype(DType::BF16) {
        Ok(()) => assert!(
            native,
            "bf16 admitted without the opt-in on a GPU that emulates it"
        ),
        Err(e) => {
            assert!(!native, "bf16 refused on a GPU that executes it: {e}");
            let e = e.to_string();
            assert!(e.contains("Metal arm `computing at `bf16`` refused"), "{e}");
        }
    }
    metal.allow_emulated(Features::empty().with(Feature::Bf16));
    assert!(dev.supports_bf16());
    assert_eq!(
        metal.emulated_arms().warnings(),
        0,
        "asking is not admitting"
    );
    dev.admit_compute_dtype(DType::BF16)?;
    dev.admit_compute_dtype(DType::BF16)?;
    assert_eq!(metal.emulated_arms().warnings(), if native { 0 } else { 1 });
    Ok(())
}

/// The child of [`the_env_alias_opts_the_process_in_and_warns_once`]: a fresh
/// device, no programmatic opt-in, so only the tuning group can admit bf16.
#[test]
#[ignore = "run by its parent with the env alias set"]
fn child_bf16_under_the_env_alias() -> Result<()> {
    let dev = device()?;
    let metal = dev.as_metal_device()?;
    dev.admit_compute_dtype(DType::BF16)?;
    assert_eq!(bf16_ops(&dev)?, bf16_ops(&Device::Cpu)?);
    let arms = metal.emulated_arms();
    println!(
        "CHILD allowed={} warnings={}",
        arms.allowed(),
        arms.warnings()
    );
    Ok(())
}

/// The env alias opts the whole process in: the child's load and ops run, and
/// its stderr carries exactly one warning. Without the alias the same child
/// fails on the refusal.
#[test]
fn the_env_alias_opts_the_process_in_and_warns_once() -> Result<()> {
    let dev = device()?;
    let native = bf16_is_native(&dev)?;
    let run = |alias: Option<&str>| {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "child_bf16_under_the_env_alias",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("CARGO_TERM_COLOR", "never");
        match alias {
            Some(v) => cmd.env(ALLOW_EMULATED_ENV, v),
            None => cmd.env_remove(ALLOW_EMULATED_ENV),
        };
        let out = cmd.output().expect("the test binary runs itself");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    let (ok, stdout, stderr) = run(Some("bf16"));
    assert!(ok, "child failed under the alias:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "the child did not run:\n{stdout}"
    );
    let warnings = stderr.lines().filter(|l| l.starts_with(WARNING)).count();
    assert_eq!(warnings, if native { 0 } else { 1 }, "{stderr}");
    assert!(
        stdout.contains(&format!("CHILD allowed=bf16 warnings={warnings}")),
        "{stdout}"
    );

    let (ok, stdout, stderr) = run(None);
    if native {
        assert!(
            ok,
            "child failed on a GPU that executes bf16:\n{stdout}\n{stderr}"
        );
    } else {
        assert!(!ok, "child passed without the alias:\n{stdout}\n{stderr}");
        assert!(
            stdout.contains("1 failed"),
            "the child did not run:\n{stdout}"
        );
        assert!(
            stdout.contains("Metal arm `computing at `bf16`` refused")
                || stderr.contains("Metal arm `computing at `bf16`` refused"),
            "the child failed for another reason:\n{stdout}\n{stderr}"
        );
    }
    Ok(())
}
