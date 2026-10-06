//! `Fusion=Derived` refuses a model whose config selects an excluded arm.
//!
//! Its own process, apart from `lfm2_fusion_derived.rs`: building the
//! `silu_impl=relu` model sets the process-wide SiLU steer
//! (`lloom_tensor::kernels::set_silu_impl`), and every test there compares
//! logits a concurrent `Relu` steer would move. The fixture is that file's
//! two-layer `tiny_config` with every hand flag off.

#![cfg(feature = "metal")]

use lloom::models::lfm2::{Config, FusionPolicy, KvAppend, LayerType, Model};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::kernels::SiluImpl;
use lloom_tensor::{DType, Device, Result};

fn tiny_config(fusion: FusionPolicy, silu_impl: SiluImpl) -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        tie_embedding: true,
        kv_append: KvAppend::InPlace,
        fusion,
        silu_impl,
        ..Default::default()
    }
}

fn build(cfg: &Config, dev: &Device) -> Result<Model<CrownExact>> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<CrownExact>(dev)
}

/// P2: the exclusion set is consulted for the arms the config selects, not
/// only for the fragments the derivation composes. `silu_impl=relu` is the
/// transform §12.2l measured degenerate (2307 f16 ulp, `ARAARAARA…`); under
/// `Derived` a model selecting it is refused at construction, by name and
/// with its numbers. Under `Manual` nothing changes — the arm stays selectable
/// for the ladder it exists to measure.
#[test]
fn derived_refuses_an_excluded_silu_arm_at_construction() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let cfg = tiny_config(FusionPolicy::Derived, SiluImpl::Relu);
    let err = match build(&cfg, &dev) {
        Ok(_) => panic!("Fusion=Derived must refuse the excluded `silu_impl=relu` arm"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("silu_impl=relu") && err.contains("excluded"),
        "{err}"
    );
    eprintln!("{err}");
    // Manual is untouched: the arm builds.
    build(&tiny_config(FusionPolicy::Manual, SiluImpl::Relu), &dev)?;
    Ok(())
}
