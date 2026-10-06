//! A hotlist gathered from one run warms a fresh pool for the next: the
//! same stream decodes the same ids with fewer misses and fewer replays —
//! none, when every layer's working set fits the pool.
//!
//! The model is the loader's, from a GGUF, on every backend
//! (`lloom::testing::fixture`): the tiny checkpoint on the CPU by default;
//! under `LLOOM_FIXTURE=real|synthetic` the Flash-Next checkpoint (or its
//! synthetic of the same shape) on Metal, at a pool far smaller than the
//! run's working set, so the real geometry's misses are the ones counted.
use lloom::models::qwen4exp::hotlist::Hotlist;
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Config, LoadOptions, Qwen4Exp, SlotBudget, StreamLoader, StreamOptions, Streamed,
};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::fixture::Fixture;
use lloom_tensor::{Device, Result, Tensor};

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
    }
}

fn decode(
    model: &Qwen4Exp<CrownExact, Streamed>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, u64)> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let mut ids = Vec::new();
    for _ in 0..steps {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = row
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &v)| if v > m.1 { (i, v) } else { m },
            )
            .0 as u32;
        ids.push(next);
        logits = model.forward(
            &Tensor::from_vec(vec![next], (1, 1), dev)?,
            cache.pos,
            &mut cache,
        )?;
    }
    Ok((ids, cache.redos))
}

#[test]
fn a_hotlist_warms_the_next_load() -> Result<()> {
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_hotlist_warm", Config::tiny())? else {
        return Ok(());
    };
    let Some(dev) = fx.device(Device::Cpu) else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    // Tiny: 6 slots of 8 experts at top-2. At scale: 64 of 512 at top-10,
    // ~9.5 GB with the stems — against a 19-token run's working set.
    let slots = fx.pick(6, 64);
    let opts = LoadOptions::default();

    // The cold run: misses, replays; its usage is the hotlist. Dropped
    // before the warm load, so at scale one pool is on the device at a time.
    let (ids_cold, redos_cold, cold_misses, hot) = {
        let cold = fx.streamed(&cfg, &dev, &opts, &stream(slots))?;
        let (ids, redos) = decode(&cold, &cfg, &dev, &prompt, 10)?;
        (
            ids,
            redos,
            cold.stream_stats().misses,
            Hotlist::from_usage(&cold.stream_usage()),
        )
    };
    let file = fx.scratch("hotlist.txt");
    hot.write(&file)?;
    let hot = Hotlist::read(&file)?;
    assert!(!hot.is_empty());

    // A fresh pool, warmed from the file, decodes the same stream.
    let warm = fx.streamed(&cfg, &dev, &opts, &stream(slots))?;
    let loaded = warm.warm_from_hotlist(&hot)?;
    assert!(loaded > 0, "nothing warmed");
    let after_warm = warm.stream_stats();
    assert_eq!(
        after_warm.misses as usize, loaded,
        "the warm's loads are its misses"
    );
    let (ids_warm, redos_warm) = decode(&warm, &cfg, &dev, &prompt, 10)?;
    let warm_stats = warm.stream_stats();
    assert_eq!(ids_warm, ids_cold, "the warmed pool decodes another stream");
    let misses_after_warm = warm_stats.misses - after_warm.misses;
    println!(
        "HOTLIST backend={} slots={slots} cold: misses={cold_misses} redos={redos_cold}; warm: loaded={loaded} misses_after={misses_after_warm} redos={redos_warm}",
        fx.backend().name()
    );
    assert!(
        misses_after_warm < cold_misses,
        "the warm pool did not miss less"
    );
    assert!(redos_warm <= redos_cold);
    // Where every layer's working set fits the pool, the warm run is clean.
    let fits = hot.layers.iter().all(|l| l.len() <= slots);
    if fits {
        assert_eq!(
            (misses_after_warm, redos_warm),
            (0, 0),
            "a fitting hotlist still missed"
        );
    }
    Ok(())
}
