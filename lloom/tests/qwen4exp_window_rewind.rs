//! A speculative window rewinds the cache to where it was opened — every
//! state channel (the fused kernels write the slabs in place on Metal), the
//! KV lengths, the position — so a pass replayed after the rewind is the
//! pass that ran before it, bit for bit. CPU and Metal.
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Config;
use lloom::nn::model::CausalLM;
use lloom_tensor::{Device, Result, Tensor};

fn check_on(dev: &Device, tag: &str) -> Result<()> {
    let cfg = Config::tiny();
    let model = testing::resident_model(&cfg, dev);
    let mut cache = model.new_cache(&cfg, dev)?;
    let prompt = Tensor::from_vec(vec![3u32, 5, 2, 8, 8, 1, 9], (1, 7), dev)?;
    let _ = model.forward(&prompt, 0, &mut cache)?;
    let step = Tensor::from_vec(vec![4u32], (1, 1), dev)?;
    let _ = model.forward(&step, cache.pos, &mut cache)?; // one decode step first: every slab written by the decode path
    let before = cache.checksums()?;
    let pos = cache.pos;
    cache.open_window()?;
    let a: Vec<f32> = model
        .forward(&step, pos, &mut cache)?
        .flatten_all()?
        .to_vec1()?;
    let during = cache.checksums()?;
    assert_ne!(before, during, "{tag}: the pass wrote no state (vacuous)");
    cache.rewind_window()?;
    let after = cache.checksums()?;
    assert_eq!(before, after, "{tag}: the rewind did not restore the state");
    assert_eq!(cache.pos, pos, "{tag}: the position");
    cache.open_window()?;
    let b: Vec<f32> = model
        .forward(&step, pos, &mut cache)?
        .flatten_all()?
        .to_vec1()?;
    cache.resolve_window()?;
    assert_eq!(a, b, "{tag}: the replayed pass differs");
    Ok(())
}

#[test]
fn window_rewind_restores_the_cache_cpu() -> Result<()> {
    check_on(&Device::Cpu, "cpu")
}

#[cfg(feature = "metal")]
#[test]
fn window_rewind_restores_the_cache_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => check_on(&dev, "metal"),
        Err(_) => Ok(()),
    }
}
