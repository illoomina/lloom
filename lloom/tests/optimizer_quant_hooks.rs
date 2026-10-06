//! Step (c)'s question, asked of the derivation (`design/docs/optimizer-design.md`
//! §10k): with the quantized matvec's store blocks in the catalogue — the
//! SiLU, the product multiplied into its operand — what does the plan place
//! on Qwen4-exp's real step, and what does the barrier model say each
//! costs? Asked before the kernels were written (the blocks then declared
//! here, hypothetically, on the quantized core); kept as the record of the
//! answer the shipped catalogue gives.

use lloom::optimizer::{derive, walk};
use lloom_tensor::Device;

#[test]
fn the_quantized_store_hooks_priced_on_the_real_step() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    let g = walk(&model);
    // The step as the switches run it; the model's own policy, which
    // refuses the quantized store blocks by their measurement (a pair that
    // does not pay end to end); and that policy admitting them — the plan
    // they make.
    let shipped = derive(&g, &model.switches_policy()).expect("derive");
    let mut policy =
        lloom::models::qwen4exp::model::fusion_policy(&lloom::optimizer::Policy::default());
    let refused = derive(&g, &policy).expect("derive");
    for f in ["quant_silu", "quant_mul"] {
        assert!(
            refused
                .refused
                .iter()
                .any(|(n, w)| n == f && w == lloom::optimizer::fragments::QUANT_STORE_LOSS),
            "{:?}",
            refused.refused
        );
    }
    assert_eq!(refused.plan.dispatches_after(), 1579);
    policy.admit = vec!["quant_silu".to_string(), "quant_mul".to_string()];
    let d = derive(&g, &policy).expect("derive");
    eprintln!("shipped: {}", shipped.plan);
    eprintln!("hooked:  {}", d.plan);
    for m in &d.marginals {
        eprintln!(
            "marginal {} sites {} {:+.1} µs/site {:?}",
            m.fragment,
            m.sites,
            m.per_site_us(),
            m.between
        );
    }
    for (f, why) in &d.refused {
        eprintln!("refused {f}: {why}");
    }
    // The blocks land on the shared experts' SwiGLU only — the SiLU on the
    // gate projection, the product on the up projection — and nowhere else
    // a quantized product runs: 48 launches fewer, the barrier between the
    // two sweeps priced and admitted (both marginals negative).
    assert_eq!(shipped.plan.dispatches_after(), 1579);
    assert_eq!(d.plan.dispatches_after(), 1579 - 48);
    let silu = d.plan.sites_of("quant_silu");
    let mul = d.plan.sites_of("quant_mul");
    assert_eq!((silu.len(), mul.len()), (48, 48));
    assert!(
        silu.iter().all(|s| s.ends_with("ffn_gate_shexp.weight")),
        "{silu:?}"
    );
    assert!(
        mul.iter().all(|s| s.ends_with("ffn_up_shexp.weight")),
        "{mul:?}"
    );
    assert_eq!(
        d.plan.kernels.get("silu_mul_f32"),
        Some(&48),
        "the routed SwiGLUs keep the kernel"
    );
    for f in ["quant_silu", "quant_mul"] {
        let m = d.marginals.iter().find(|m| m.fragment == f).unwrap();
        assert!(m.delta_us < 0.0, "{f}: {m:?}");
    }
}
