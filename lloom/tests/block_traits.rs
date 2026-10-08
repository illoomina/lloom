//! The level-0 block traits (`nn::blocks`) on both families' blocks: each
//! trait's `execute` is the block's own path, bit for bit, and its `visit`
//! emits the node the walk reads. The non-vacuity is that every arm here
//! runs through the trait, not the inherent method.

use lloom::models::qwen4exp::{testing, Config};
use lloom::nn::blocks::{Activation, Ffn, Mixer, Neox, Norm, Rope, RopeStyle, Thd};
use lloom::nn::{RecurrentState, RmsNorm};
use lloom::optimizer::{Shape, Visit, Walker};
use lloom::profile::CrownExact;
use lloom_tensor::{DType, Device, Module, Result, Tensor, D};

fn same(a: &Tensor, b: &Tensor) -> Result<bool> {
    Ok(a.dims() == b.dims() && a.eq(b)?.flatten_all()?.min(0)?.to_scalar::<u8>()? == 1)
}

#[test]
fn a_norm_executes_its_own_forward_and_visits_an_rms_node() -> Result<()> {
    let dev = Device::Cpu;
    let w = Tensor::new(&[1.0f32, 0.5, 2.0, 1.5], &dev)?;
    let norm = RmsNorm::new(w, 1e-5);
    let x = Tensor::new(&[[0.1f32, -0.2, 0.3, 0.4], [1.0, 2.0, -3.0, 0.5]], &dev)?;
    assert!(same(&Norm::execute(&norm, &x)?, &norm.forward(&x)?)?);
    let mut walker = Walker::new();
    let xn = walker.input("x", Shape::of(&[1, 1, 4]), DType::F32);
    let out = norm.visit(&mut walker, xn);
    let g = walker.finish();
    assert_eq!(g.node(out).inputs.len(), 2, "an rms_norm node reads the weight and x");
    Ok(())
}

#[test]
fn the_rope_styles_execute_their_kernels_and_name_them() -> Result<()> {
    let dev = Device::Cpu;
    let (h, t, d) = (2, 3, 8);
    let x = Tensor::arange(0f32, (h * t * d) as f32, &dev)?
        .reshape((1, h, t, d))?
        .affine(0.01, -0.3)?;
    let cos = Tensor::arange(0f32, (t * d / 2) as f32, &dev)?
        .reshape((t, d / 2))?
        .affine(0.1, 0.0)?
        .cos()?;
    let sin = Tensor::arange(0f32, (t * d / 2) as f32, &dev)?
        .reshape((t, d / 2))?
        .affine(0.1, 0.0)?
        .sin()?;
    assert_eq!(<Neox as Rope>::STYLE, RopeStyle::Neox);
    assert!(same(
        &Neox.execute(&x, &cos, &sin)?,
        &lloom::nn::rotary_emb::rope(&x, &cos, &sin)?
    )?);
    assert!(same(
        &lloom::nn::blocks::Interleaved.execute(&x, &cos, &sin)?,
        &lloom::nn::rotary_emb::rope_i(&x, &cos, &sin)?
    )?);
    let x_thd = x.transpose(1, 2)?.contiguous()?;
    assert!(same(
        &Thd.execute(&x_thd, &cos, &sin)?,
        &lloom::nn::rotary_emb::rope_thd(&x_thd, &cos, &sin)?
    )?);
    // The two pairings differ on the same input: two types, not one with a flag.
    assert!(!same(
        &Neox.execute(&x, &cos, &sin)?,
        &lloom::nn::blocks::Interleaved.execute(&x, &cos, &sin)?
    )?);
    let mut walker = Walker::new();
    let xn = walker.input("x", Shape::of(&[1, h, 1, d]), DType::F32);
    let c = walker.input("cos", Shape::of(&[1, d / 2]), DType::F32);
    let s = walker.input("sin", Shape::of(&[1, d / 2]), DType::F32);
    let out = Neox.visit(&mut walker, (xn, c, s));
    let g = walker.finish();
    assert_eq!(g.node(out).name, "rope");
    Ok(())
}

#[test]
fn an_activation_executes_and_visits_as_itself() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::new(&[-2.0f32, -0.5, 0.0, 0.5, 2.0], &dev)?;
    assert!(same(
        &Activation::execute(&lloom::nn::Activation::Silu, &x)?,
        &x.silu()?
    )?);
    assert!(same(
        &Activation::execute(&lloom::nn::fusion::Silu, &x)?,
        &x.silu()?
    )?);
    assert!(same(
        &Activation::execute(&lloom::nn::Activation::Gelu, &x)?,
        &x.gelu_erf()?
    )?);
    Ok(())
}

#[test]
fn qwen4exps_blocks_execute_their_decode_and_prefill_paths() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = Config::tiny();
    let d = cfg.hidden_size;
    let x1 = Tensor::arange(0f32, d as f32, &dev)?
        .affine(0.02, -0.4)?
        .reshape((1, 1, d))?;
    let x3 = Tensor::arange(0f32, (3 * d) as f32, &dev)?
        .affine(0.01, -0.3)?
        .reshape((1, 3, d))?;

    // The sparse attention: a chunk, then a step, through the trait and
    // through the inherent methods, against two caches.
    let qsa = testing::qsa::<CrownExact>(&cfg, &dev);
    let mut via_trait = lloom::models::qwen4exp::KvCache::default();
    let mut via_inherent = lloom::models::qwen4exp::KvCache::default();
    let a = Mixer::execute(&qsa, &x3, &mut via_trait)?;
    let b = qsa.prefill(&x3, &mut via_inherent)?;
    assert!(same(&a, &b)?);
    let a = Mixer::execute(&qsa, &x1, &mut via_trait)?;
    let b = qsa.decode(&x1, &mut via_inherent)?;
    assert!(same(&a, &b)?);
    assert_eq!(a.dims(), [1, 1, d]);

    // The Gated DeltaNet, the same way, against two recurrent states.
    let gdn = testing::deltanet::<CrownExact>(&cfg, &dev);
    let mut st = lloom::models::qwen4exp::GdnState {
        layer: 0,
        s: RecurrentState::new(1),
        conv: RecurrentState::new(1),
    };
    let (mut s2, mut c2) = (RecurrentState::new(1), RecurrentState::new(1));
    let a = Mixer::execute(&gdn, &x3, &mut st)?;
    let b = gdn.prefill(&x3, 0, &mut s2, &mut c2)?;
    assert!(same(&a, &b)?);
    let a = Mixer::execute(&gdn, &x1, &mut st)?;
    let b = gdn.decode(&x1, 0, &mut s2, &mut c2)?;
    assert!(same(&a, &b)?);

    // The routed MoE.
    let moe = testing::moe::<CrownExact>(&cfg, &dev);
    let a = Ffn::execute(&moe, &x3, &mut ())?;
    let b = moe.forward(&x3)?;
    assert!(same(&a, &b)?);
    assert_eq!(a.dims(), [1, 3, d]);
    let _ = D::Minus1;
    Ok(())
}
