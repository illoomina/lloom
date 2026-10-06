//! The routed matmul-vec over DENSE (f32) expert stacks on Metal — the
//! `kernel_mul_mv_id_f32_f32` arm every f32-expert test model decodes
//! through — against the CPU, at the tiny geometry and around it. Its
//! dispatch was one threadgroup per 8 rows for a template that computes
//! one row per threadgroup (7 rows in 8 unwritten) with a zero row stride
//! (every written row read row 0): every Metal test model's MoE was garbage
//! and no test compared it. The Q8_0 row is the control (the quantized
//! cores address rows themselves).
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Resident};
use lloom::profile::CrownExact;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Tensor};
use std::sync::Arc;

fn probe(cfg: &Config, dtype: GgmlDType, tag: &str) {
    let cpu = Device::Cpu;
    let gpu = Device::new_metal(0).unwrap();
    let mk = |dev: &Device| {
        let [g, u, d] = testing::expert_stacks(cfg, 0, dev, dtype);
        testing::moe_with::<CrownExact, _>(
            cfg,
            dev,
            Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)),
        )
    };
    let (mc, mg) = (mk(&cpu), mk(&gpu));
    let d = cfg.hidden_size;
    let x = Tensor::from_vec(
        (0..d)
            .map(|i| ((i as f32 * 0.05).sin()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, 1, d),
        &cpu,
    )
    .unwrap();
    let a: Vec<f32> = mc
        .forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b: Vec<f32> = mg
        .forward(&x.to_device(&gpu).unwrap())
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let diff = a
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max);
    println!("PROBE {tag} dtype={dtype:?} hidden={} n_ff={} experts={} topk={} maxdiff={diff:e} scale={scale:e} rel={:e}", cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts, cfg.n_experts_per_tok, diff / scale);
    assert!(
        diff <= 1e-5 * scale,
        "{tag} {dtype:?}: the routed arm differs from the CPU by {diff} at scale {scale}"
    );
}

#[test]
fn routed_dense_arm_matches_cpu_around_the_tiny_geometry() {
    let base = Config::tiny();
    probe(&base, GgmlDType::F32, "tiny");
    let mut c = base.clone();
    c.moe_intermediate_size = 32;
    probe(&c, GgmlDType::F32, "nff32");
    probe(&c, GgmlDType::Q8_0, "nff32");
    let mut c = base.clone();
    c.moe_intermediate_size = 64;
    c.hidden_size = 64;
    probe(&c, GgmlDType::F32, "h64_nff64");
    let mut c = base.clone();
    c.n_experts_per_tok = 1;
    probe(&c, GgmlDType::F32, "topk1");
    let mut c = base.clone();
    c.shared_expert_intermediate_size = 32;
    probe(&c, GgmlDType::F32, "shexp32");
}
