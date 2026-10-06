//! Step (c)'s question, asked of the derivation before any kernel is
//! written (`design/docs/optimizer-design.md` §10k): if the quantized
//! matvec took the blocks the GEMV takes on its store — the SiLU, the
//! row-indexed product — what would the plan place on Qwen4-exp's real
//! step, and what would the barrier model say each costs? The blocks are
//! declared here, hypothetically, on the quantized core; the rules and the
//! pricing are the shipped ones, read off the blocks' semantics.

use lloom::nn::fusion::{OperandKind, Precision, Semantics, TileNeed};
use lloom::optimizer::fragments::{catalogue, Fragment, Side};
use lloom::optimizer::{derive_with, walk, Core, Policy, TargetFpModel};
use lloom_tensor::Device;

fn quant(name: &'static str, semantics: Semantics, operand: OperandKind) -> Fragment {
    Fragment {
        name,
        core: Core::Quant,
        semantics,
        precision: Precision::Accumulator,
        operand,
        scales: false,
        model: TargetFpModel::of_family("gemv").unwrap(),
        side: Side::Store,
        needs: TileNeed::Any,
        refusal: None,
        composes_with: &[],
    }
}

#[test]
fn the_quantized_store_hooks_priced_on_the_real_step() {
    let model = lloom::models::qwen4exp::testing::ghost_model(&Device::Cpu).expect("ghost build");
    let g = walk(&model);
    let policy = lloom::models::qwen4exp::model::fusion_policy(&Policy::default());
    let shipped = lloom::optimizer::derive(&g, &policy).expect("derive");
    let mut cat = catalogue();
    cat.push(quant("quant_silu", Semantics::Silu, OperandKind::None));
    cat.push(quant("quant_mul", Semantics::MulOperand, OperandKind::RowIndexed));
    let d = derive_with(&g, &policy, &cat).expect("derive");
    eprintln!("shipped: {}", shipped.plan);
    eprintln!("hooked:  {}", d.plan);
    for m in &d.marginals {
        eprintln!("marginal {} sites {} {:+.1} µs/site {:?}", m.fragment, m.sites, m.per_site_us(), m.between);
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
    assert!(silu.iter().all(|s| s.ends_with("ffn_gate_shexp.weight")), "{silu:?}");
    assert!(mul.iter().all(|s| s.ends_with("ffn_up_shexp.weight")), "{mul:?}");
    assert_eq!(d.plan.kernels.get("silu_mul_f32"), Some(&48), "the routed SwiGLUs keep the kernel");
    for f in ["quant_silu", "quant_mul"] {
        let m = d.marginals.iter().find(|m| m.fragment == f).unwrap();
        assert!(m.delta_us < 0.0, "{f}: {m:?}");
    }
}
