//! The streamed tier planned on the device (no per-layer readback; one
//! settle per token; a token that computed on a missing expert is replayed
//! from the window) decodes the ids and the logits the resident model
//! decodes — bit for bit — on a pool small enough that tokens miss and are
//! replayed. CPU and, when present, Metal (the map kernel).
//!
//! The model is the loader's, from a GGUF (`lloom::testing::fixture`). On
//! it, 3 slots of 8 miss on every pass, so the adaptive switch never arms
//! the device plan (`expert-streaming.md` §20.14's steady-miss shape); 6
//! slots over a 24-token decode arm it and replay.
//!
//! At scale (`LLOOM_FIXTURE=real|synthetic`, Metal) the resident model is
//! 47 GB and refused at load (`memory.md` §9.5q), so the reference is the
//! same tier at twice the pool, and the subject's pool must evict what the
//! reference's holds: the stream may not depend on what a pool held. Whether
//! the device plan arms at scale is the workload's routing, not the tier's:
//! on the real checkpoint no probed prompt armed it within 24 tokens
//! (`design/measurements/issue-865-real-scale.md`), so at scale its counters
//! are reported and the eviction is what is asserted.
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Cache, Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::testing::fixture::{Backend, Fixture};
use lloom_tensor::{Device, Result, Tensor};

fn stream(slots: usize, prefetch: bool) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch,
        io_hints: false,
    }
}

type Decoded = (Vec<u32>, Vec<Vec<f32>>, Cache);

fn decode<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<Decoded> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let (mut ids, mut rows) = (Vec::new(), Vec::new());
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
        rows.push(row);
        ids.push(next);
        let input = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
    }
    Ok((ids, rows, cache))
}

fn check_on(fx: &Fixture<Qwen4ExpTiny>, dev: &Device, tag: &str, prefetch: bool) -> Result<()> {
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    // At scale a 192-token prompt, in blocks of distinct ids: a prefill that
    // groups rows to the pool and evicts from it.
    let base = [3u32, 5, 2, 8, 8, 1, 9, 3, 5];
    let prompt: Vec<u32> = (0..fx.pick(9u32, 192))
        .map(|i| base[(i % 9) as usize] + 1000 * (i / 9))
        .collect();
    let steps = 24;
    // The reference: every expert resident, or at scale the tier at twice
    // the pool. Dropped before the subject loads.
    let (want_ids, want_rows) = if fx.at_scale() {
        let wide = fx.streamed(&cfg, dev, &opts, &stream(256, prefetch))?;
        let (ids, rows, _) = decode(&wide, &cfg, dev, &prompt, steps)?;
        (ids, rows)
    } else {
        let resident = fx.resident(&cfg, dev, &opts)?;
        let (ids, rows, _) = decode(&resident, &cfg, dev, &prompt, steps)?;
        (ids, rows)
    };
    // Tiny: 6 slots for 8 experts at top-2. At scale: 128 for 512 at
    // top-10. Tokens miss, settles load, passes replay.
    let streamed = fx.streamed(&cfg, dev, &opts, &stream(fx.pick(6, 128), prefetch))?;
    let (ids, rows, cache) = decode(&streamed, &cfg, dev, &prompt, steps)?;
    assert_eq!(ids, want_ids, "{tag}: the device-planned ids differ");
    for (t, (a, b)) in rows.iter().zip(&want_rows).enumerate() {
        assert_eq!(a, b, "{tag}: step {t} logits differ");
    }
    let st = streamed.stream_stats();
    println!(
        "DEVICEPLAN {tag} backend={} steps={steps} device_plans={} settled_misses={} redos={} hits={} misses={} evictions={}",
        fx.backend().name(),
        st.device_plans,
        st.settled_misses,
        cache.redos,
        st.hits,
        st.misses,
        st.evictions
    );
    if fx.at_scale() {
        assert!(
            st.evictions > 0,
            "{tag}: the pool evicted nothing, so it held what the reference held: {st:?}"
        );
        return Ok(());
    }
    assert!(st.device_plans > 0, "{tag}: no device plan ran: {st:?}");
    assert!(
        cache.redos > 0 && st.settled_misses > 0,
        "{tag}: no token was replayed — the pool never missed: {st:?} redos {}",
        cache.redos
    );
    // The arms, when this binary runs under them: each must have fired —
    // a counter at 0 says the arm was not exercised, and the identity
    // above was vacuous for it.
    if std::env::var("QWEN4EXP_REPLAY").is_ok_and(|v| v == "device") {
        assert!(
            cache.device_replays > 0,
            "{tag}: QWEN4EXP_REPLAY=device: no device-planned replay (redos {}, fallbacks {})",
            cache.redos,
            cache.replay_fallbacks
        );
    }
    if prefetch && std::env::var("QWEN4EXP_PREDICT").is_ok_and(|v| v == "hotlist") {
        // The prediction lands only in free slots and stages only as many
        // as there are (`memory.md` §9.9): with spare slots kept free
        // (`QWEN4EXP_SPARE_SLOTS`, recycled a settle later) it must have
        // landed; on the in-place arm this pool is full from the prefill
        // on and the arm must have staged nothing.
        let spare = std::env::var("QWEN4EXP_SPARE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if spare > 0 {
            assert!(
                st.predict_landed > 0,
                "{tag}: QWEN4EXP_PREDICT=hotlist with {spare} spare slots: nothing landed: {st:?}"
            );
        } else {
            assert_eq!(
                (st.predict_landed, st.predict_wasted),
                (0, 0),
                "{tag}: QWEN4EXP_PREDICT=hotlist on a full in-place pool staged something: {st:?}"
            );
        }
    }
    if std::env::var("QWEN4EXP_MASK").is_ok_and(|v| v == "1") {
        assert!(
            lloom::models::qwen4exp::moe::masked_combines() > 0,
            "{tag}: QWEN4EXP_MASK=1: no combine ran masked"
        );
    }
    println!(
        "DEVICEPLAN-ARMS {tag} device_replays={} replay_fallbacks={} predict_landed={} predict_wasted={} masked_combines={}",
        cache.device_replays,
        cache.replay_fallbacks,
        st.predict_landed,
        st.predict_wasted,
        lloom::models::qwen4exp::moe::masked_combines()
    );
    Ok(())
}

/// The tiny model on the CPU. At scale the Metal test runs the checkpoint.
#[test]
fn device_plan_decodes_the_resident_stream_cpu() -> Result<()> {
    if Backend::current().at_scale() {
        println!("at scale: device_plan_decodes_the_resident_stream_metal runs the checkpoint");
        return Ok(());
    }
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_device_plan_cpu", Config::tiny())? else {
        return Ok(());
    };
    check_on(&fx, &Device::Cpu, "cpu", false)
}

#[cfg(feature = "metal")]
#[test]
fn device_plan_decodes_the_resident_stream_metal() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_device_plan_metal", Config::tiny())?
    else {
        return Ok(());
    };
    check_on(&fx, &dev, "metal", false)
}

/// Metal, the tier with its prefetcher (the `pread` arm's reader
/// threads): the device plan lands what was staged for it before it maps
/// the pool (`QWEN4EXP_PREDICT=hotlist` stages the hotlist's experts for
/// a device-planned token; the leftovers of a host plan's hint otherwise)
/// and the stream is the resident model's still.
#[cfg(feature = "metal")]
#[test]
fn device_plan_with_the_prefetcher_decodes_the_resident_stream_metal() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) =
        Fixture::<Qwen4ExpTiny>::new("qwen4exp_device_plan_prefetch_metal", Config::tiny())?
    else {
        return Ok(());
    };
    check_on(&fx, &dev, "metal-prefetch", true)
}
