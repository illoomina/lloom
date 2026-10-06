//! The float-code kinds (MXFP4, FP8 E4M3) through a whole model: a tiny
//! Qwen4-exp checkpoint (`docs/porting-models.md` §21.8) whose dense
//! matrices and routed experts are stored in the kind, written as a GGUF and
//! read back through the family's real loader.
//!
//! - CPU: the model in memory and the written file compute the same logits
//!   bit for bit, with the packed weights multiplied by the CPU's own
//!   `vec_dot` (not a dequantized copy).
//! - Metal: the same checkpoint against its CPU f32 reference within the
//!   tiny manifests' tolerance, every workload, and the dispatch census
//!   shows the kind's own kernels ran — the plain, routed and paired
//!   matmul-vecs and the embedding gather.

use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{self, Synthesis, TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};

const KINDS: [GgmlDType; 2] = [GgmlDType::MXFP4, GgmlDType::FP8E4M3];

fn stem(kind: GgmlDType) -> &'static str {
    match kind {
        GgmlDType::MXFP4 => "mxfp4",
        GgmlDType::FP8E4M3 => "fp8_e4m3",
        _ => unreachable!(),
    }
}

/// The template with every matrix row a whole number of 32-element blocks,
/// so the kind reaches the experts' down projection too (16 by default).
fn checkpoint(kind: GgmlDType, synthesis: Synthesis) -> TinyCheckpoint<Qwen4ExpTiny> {
    let mut cfg = Qwen4ExpTiny::template();
    cfg.vocab_size = Qwen4ExpTiny::tokenizer().vocab_size();
    cfg.moe_intermediate_size = 32;
    cfg.shared_expert_intermediate_size = 32;
    TinyCheckpoint::new(
        cfg,
        Weights {
            dense: kind,
            experts: kind,
            synthesis,
            ..Weights::default()
        },
    )
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|s| s.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-tiny-fp-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_float_code_checkpoint_runs_on_the_cpu_from_memory_and_from_its_file() -> Result<()> {
    for kind in KINDS {
        for synthesis in [Synthesis::Packed, Synthesis::Quantized] {
            let ckpt = checkpoint(kind, synthesis);
            let schema = ckpt.schema()?;
            let in_kind = schema.tensors.iter().filter(|t| t.dtype == kind).count();
            let experts_in_kind = schema
                .tensors
                .iter()
                .filter(|t| t.dtype == kind && t.name.contains("_exps"))
                .count();
            assert!(in_kind > experts_in_kind && experts_in_kind > 0, "{kind:?}");
            let dir = scratch(&format!("{kind:?}-{synthesis:?}"));
            let path = ckpt.write(&dir)?;
            let files = Qwen4ExpTiny::load::<CrownExact>(&path, &Device::Cpu)?;
            let mem = ckpt.model::<CrownExact>(&Device::Cpu)?;
            for w in Qwen4ExpTiny::workloads(&ckpt.cfg) {
                let a = tiny::run(&mem, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &w)?;
                let b = tiny::run(&files, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &w)?;
                assert!(a.iter().flatten().all(|x| x.is_finite()), "{kind:?}");
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "{kind:?} {synthesis:?} {w:?}: files != memory"
                );
                println!(
                    "cpu {kind:?} {synthesis:?} prompt={} decode={}: {in_kind} tensors in kind \
                     ({experts_in_kind} expert stacks), logits[0][..3] {:?}",
                    w.prompt,
                    w.decode,
                    &a[0][..3]
                );
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn a_float_code_checkpoint_runs_on_metal_against_its_cpu_reference() -> Result<()> {
    use lloom_tensor::kernels::metal::profile;
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let gpu = Device::new_metal(0)?;
    assert!(
        profile::enabled(),
        "the census names the kernels each kind ran"
    );
    // The tiny manifests' bound on the relative CPU-vs-Metal logit
    // difference (`tests/tiny_manifests/qwen4exp.json`'s `tol`).
    let tol = 0.05f32;
    for kind in KINDS {
        let ckpt = checkpoint(kind, Synthesis::Packed);
        let dir = scratch(&format!("metal-{kind:?}"));
        let path = ckpt.write(&dir)?;
        let reference = Qwen4ExpTiny::reference::<CrownExact>(&ckpt, &Device::Cpu)?;
        gpu.synchronize()?;
        profile::reset();
        let model = Qwen4ExpTiny::load::<CrownExact>(&path, &gpu)?;
        for w in Qwen4ExpTiny::workloads(&ckpt.cfg) {
            let want = tiny::run(&reference, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &w)?;
            let got = tiny::run(&model, &ckpt.cfg, ckpt.cfg.vocab_size, &gpu, &w)?;
            let mut rel = 0f32;
            for (c, m) in want.iter().zip(&got) {
                let scale = c.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
                let d = c
                    .iter()
                    .zip(m)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(d.is_finite(), "{kind:?}: a non-finite logit");
                rel = rel.max(d / scale);
            }
            println!(
                "metal {kind:?} prompt={} decode={}: max |metal - cpu| / max|cpu| = {rel:e}",
                w.prompt, w.decode
            );
            assert!(rel <= tol, "{kind:?} {w:?}: CPU vs Metal {rel:e} > {tol}");
        }
        gpu.synchronize()?;
        let labels: Vec<String> = profile::snapshot()
            .by_label
            .into_iter()
            .map(|(l, n)| format!("{l} x{n}"))
            .filter(|l| l.contains(stem(kind)))
            .collect();
        println!("metal {kind:?} kernels: {labels:?}");
        for want in [
            format!("kernel_mul_mv_{}_f32 ", stem(kind)),
            format!("kernel_mul_mv_id_{}_", stem(kind)),
            format!("kernel_get_rows_{} ", stem(kind)),
        ] {
            assert!(
                labels.iter().any(|l| l.starts_with(&want)),
                "{kind:?}: no {want}* in the census {labels:?}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
    Ok(())
}
