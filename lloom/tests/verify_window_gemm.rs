//! A speculative verify window under `deterministic` (`architecture.md`
//! §2.3.5b): the target's pass over `[last, draft…]` runs its half matmuls on
//! the GEMM (`tuning::VerifyWindow`), counted by the window's engagement
//! counter, while the prefill and the K=1 decode step keep the one-row
//! reduction and count nothing. With the window's arm off the verify counts
//! nothing either, which is the arm the K=1 reduction would charge a window.
//!
//! Reachability is what this file holds: the reduction against the GEMM at
//! one matmul is `lloom-tensor`'s `f16_rows_identity`.
//!
//! Dual (`lloom::testing::fixture`): under `LLOOM_FIXTURE=real|synthetic`
//! the same assertions run at LFM2.5-2.6B's geometry.

#![cfg(feature = "metal")]

use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::RewindableExact;
use lloom::testing::fixture::Fixture;
use lloom_tensor::tuning::{set_verify_window_arm, verify_window_gemms};
use lloom_tensor::{DType, Device, Result, Tensor};

const K: usize = 3;

fn tiny_geometry() -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: 3,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        ..Default::default()
    }
}

/// The rewindable arms a verify needs: a rotating conv ring of `K + 1`
/// history slots over the in-place KV.
fn arms(c: &mut Config) {
    c.use_flash_attn = false;
    c.attn_impl = Default::default();
    c.kv_append = KvAppend::InPlace;
    c.kv_quant = Default::default();
    c.conv_state = ConvState::RotatingRing { k: K + 1 };
    c.memory_budget = None;
    c.gemv_silu = false;
    c.gemv_residual = false;
    c.tap_layers = Vec::new();
    c.tap_dtype = DType::F32;
}

/// What each phase of one run counted on the window's counter, and what the
/// verify accepted.
#[derive(Debug, PartialEq)]
struct Counted {
    prefill: u64,
    step: u64,
    verify: u64,
    accepted: Vec<u32>,
}

fn run(model: &Model<RewindableExact>, cfg: &Config, dev: &Device) -> Result<Counted> {
    let mut cache = State::<RewindableExact>::new(true, DType::F16, cfg, dev)?;
    let prompt: Vec<u32> = (1..=10).collect();
    let mut at = verify_window_gemms();
    let mut since = || {
        let now = verify_window_gemms();
        let d = now - at;
        at = now;
        d
    };
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    model.forward(&inp, 0, &mut cache)?;
    dev.synchronize()?;
    let prefill = since();
    let step = Tensor::new(&[11u32], dev)?.reshape((1, 1))?;
    model.forward(&step, prompt.len(), &mut cache)?;
    dev.synchronize()?;
    let step = since();
    let accepted = model.verify_step(12, &[13, 14, 15], prompt.len() + 1, &mut cache, &[])?;
    let verify = since();
    Ok(Counted {
        prefill,
        step,
        verify,
        accepted,
    })
}

/// The verify takes the window's GEMM and the prefill and the K=1 step do
/// not; the count is a property of the window's shape (two runs agree); and
/// with the arm off the verify counts nothing.
#[test]
fn the_verify_window_takes_the_gemm_and_the_k1_path_does_not() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("verify_window_gemm", tiny_geometry())? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    assert!(
        !lloom_tensor::tuning::contract().is_fast(),
        "the regime pins `deterministic`"
    );

    set_verify_window_arm(true);
    let on = run(&model, &cfg, &dev)?;
    let again = run(&model, &cfg, &dev)?;
    set_verify_window_arm(false);
    let off = run(&model, &cfg, &dev)?;
    set_verify_window_arm(true);
    eprintln!("VERIFY_WINDOW_COUNTS on={on:?} off={off:?}");

    assert_eq!(on.prefill, 0, "a prefill is not a window: {on:?}");
    assert_eq!(
        on.step, 0,
        "the K=1 step keeps the one-row reduction: {on:?}"
    );
    assert!(
        on.verify > 0,
        "the verify window never reached the GEMM: {on:?}"
    );
    assert_eq!(
        on, again,
        "the window's launches and tokens are its shape's"
    );
    assert_eq!(
        (off.prefill, off.step, off.verify),
        (0, 0, 0),
        "with the arm off nothing takes the window's GEMM: {off:?}"
    );
    Ok(())
}
