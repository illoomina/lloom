//! A prompt's prefill under `deterministic` (`architecture.md` §2.3.5b):
//! the driver's prefill forwards (`Session::prefill`, the K=1 decode's own
//! prefill, the warmup) open `tuning::VerifyWindow::prefill`, so their half
//! matmuls take the GEMM, counted by the prefill window's counter, while the
//! K=1 decode steps after them count nothing. The promise is per tile shape:
//! a prompt prefilled at the same tile widths is the same bits run to run.
//! The CPU's forward over the same tiny checkpoint is the independent
//! reference for both reductions; with the arm off the prefill is the
//! one-row reduction and counts nothing.
//!
//! Reachability is what this file holds: the reduction against the GEMM at
//! one matmul is `lloom-tensor`'s `f16_rows_identity`.

#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::RewindableExact;
use lloom::testing::fixture::Fixture;
use lloom_tensor::tuning::Window;
use lloom_tensor::{DType, Device, Result};

const EOS: u32 = 63;

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

/// The verify test's arms, and the conv's left-context carry a prompt fed
/// in more than one tile needs.
fn arms(c: &mut Config) {
    c.use_flash_attn = false;
    c.attn_impl = Default::default();
    c.kv_append = KvAppend::InPlace;
    c.kv_quant = Default::default();
    c.conv_state = ConvState::RotatingRing { k: 4 };
    c.conv_prefill_ctx = true;
    c.memory_budget = None;
    c.gemv_silu = false;
    c.gemv_residual = false;
    c.tap_layers = Vec::new();
    c.tap_dtype = DType::F32;
}

/// One fresh session: the prompt prefilled at `tile`, the held logits, a
/// one-token suffix prefilled onto it, then a K=1 greedy decode; what each
/// phase counted on the prefill window.
#[derive(Debug, PartialEq)]
struct Run {
    prefill: u64,
    suffix: u64,
    decode: u64,
    logits: Vec<u32>,
    tokens: Vec<u32>,
}

impl Run {
    fn logits_f32(&self) -> Vec<f32> {
        self.logits.iter().map(|&b| f32::from_bits(b)).collect()
    }
}

fn run(
    model: &Model<RewindableExact>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    tile: usize,
) -> Result<Run> {
    let cache = State::<RewindableExact>::new(true, DType::F16, cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(&[EOS], dev)?;
    let mut opts = DecodeOptions::new(6);
    opts.prefill_tile = tile;
    let mut at = Window::Prefill.gemms();
    let mut since = || {
        dev.synchronize()?;
        let now = Window::Prefill.gemms();
        let d = now - at;
        at = now;
        Result::Ok(d)
    };
    session.prefill(&mut rb, prompt, &opts)?;
    let prefill = since()?;
    let logits = session
        .last_logits()
        .expect("a prefill holds its logits")
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect();
    session.prefill(&mut rb, &[5], &opts)?;
    let suffix = since()?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let tokens = session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut sampler,
            &[],
            &opts,
            &mut |_: &[u32]| Ok(()),
        )?
        .ids;
    let decode = since()?;
    Ok(Run {
        prefill,
        suffix,
        decode,
        logits,
        tokens,
    })
}

/// Bound on [`rel_err`] against the CPU's forward: four times the largest
/// measured (`3.6e-3`, the one-row reduction's) rounded up to a power of two.
const CPU_BOUND: f32 = 1.0 / 64.0;

/// The largest `|a − b|` over the largest `|b|`.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale.max(f32::MIN_POSITIVE)
}

/// The prefill takes the window's GEMM at each tiling and the K=1 path does
/// not (the one-token suffix is the GEMV under either contract, the decode
/// steps outside every window); a tiling's prefill repeats bit for bit;
/// both reductions sit within the bound of the CPU's forward; and with the
/// arm off nothing counts.
#[test]
fn the_prefill_takes_the_gemm_per_tile_shape_and_the_k1_path_does_not() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("prefill_window_gemm", tiny_geometry())? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    assert!(
        !lloom_tensor::tuning::contract().is_fast(),
        "the regime pins `deterministic`"
    );
    let prompt: Vec<u32> = (0..37).map(|i| 1 + (i * 7) % 60).collect();

    let cpu = {
        let cpu_model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &Device::Cpu)?;
        run(&cpu_model, &cfg, &Device::Cpu, &prompt, 0)?.logits_f32()
    };

    for tile in [0usize, 16] {
        Window::Prefill.set_arm(true);
        let on = run(&model, &cfg, &dev, &prompt, tile)?;
        let again = run(&model, &cfg, &dev, &prompt, tile)?;
        Window::Prefill.set_arm(false);
        let off = run(&model, &cfg, &dev, &prompt, tile)?;
        Window::Prefill.set_arm(true);
        let moved = on
            .logits
            .iter()
            .zip(&off.logits)
            .filter(|(a, b)| a != b)
            .count();
        let (err_on, err_off) = (
            rel_err(&on.logits_f32(), &cpu),
            rel_err(&off.logits_f32(), &cpu),
        );
        eprintln!(
            "PREFILL_WINDOW_COUNTS tile={tile} on=({},{},{}) off=({},{},{}) logits_moved={moved}/{} cpu_rel_err on={err_on:.3e} off={err_off:.3e} tokens on={:?} off={:?}",
            on.prefill, on.suffix, on.decode, off.prefill, off.suffix, off.decode,
            on.logits.len(), on.tokens, off.tokens
        );
        assert!(
            on.prefill > 0,
            "tile={tile}: the prefill never reached the GEMM: {on:?}"
        );
        assert_eq!(
            (on.suffix, on.decode),
            (0, 0),
            "tile={tile}: a one-token suffix and the K=1 decode steps take no window launch"
        );
        assert_eq!(
            on, again,
            "tile={tile}: the prefill's launches and bits are its tiling's"
        );
        assert_eq!(
            (off.prefill, off.suffix, off.decode),
            (0, 0, 0),
            "tile={tile}: with the arm off nothing takes the prefill's GEMM"
        );
        assert!(
            moved > 0,
            "tile={tile}: the GEMM and the one-row reduction gave the same logits: no other arithmetic ran"
        );
        for (arm, err) in [("on", err_on), ("off", err_off)] {
            assert!(
                err <= CPU_BOUND,
                "tile={tile} arm {arm}: the prefill's logits are {err:.3e} of their scale from the CPU's (bound {CPU_BOUND:.3e})"
            );
        }
        assert!(
            err_on <= 2.0 * err_off,
            "tile={tile}: the GEMM's logits ({err_on:.3e}) sit further from the CPU's than the one-row reduction's ({err_off:.3e})"
        );
    }
    Ok(())
}

/// The warmup's prefills are prefill windows, so they compile the pipelines
/// the real prefill takes; with the arm off they count nothing.
#[test]
fn the_warmup_prefills_take_the_prefill_window() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    // The warmup's prefills and its decode step run past the other tests'
    // 128 positions.
    let geometry = Config {
        max_position_embeddings: 512,
        ..tiny_geometry()
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("prefill_window_warmup", geometry)? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    let mut counted = Vec::new();
    for on in [true, false] {
        Window::Prefill.set_arm(on);
        let cache = State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let before = Window::Prefill.gemms();
        session.warmup(&dev)?;
        counted.push(Window::Prefill.gemms() - before);
    }
    Window::Prefill.set_arm(true);
    eprintln!("PREFILL_WINDOW_WARMUP on={} off={}", counted[0], counted[1]);
    assert!(
        counted[0] > 0,
        "the warmup's prefills took no prefill launch"
    );
    assert_eq!(counted[1], 0, "with the arm off the warmup counts nothing");
    Ok(())
}

/// The batched scheduler admits a row by prefilling its whole prompt in a
/// prefill window, then decodes the rows as one batched K=1 step outside
/// every window: over the whole run the prefill window counts exactly what
/// the rows' solo prefills count (the decode rows add nothing), and every
/// row decodes its solo run's tokens.
#[test]
fn the_batched_scheduler_prefills_in_the_window_and_its_decode_rows_do_not() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("prefill_window_batched", tiny_geometry())? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    Window::Prefill.set_arm(true);
    let prompts: Vec<Vec<u32>> = vec![
        (0..37).map(|i| 1 + (i * 7) % 60).collect(),
        (0..20).map(|i| 2 + (i * 11) % 59).collect(),
    ];
    let opts = DecodeOptions::new(6);
    let fresh = || State::<RewindableExact>::new(true, DType::F16, &cfg, &dev);

    let mut solo_prefill = 0u64;
    let mut solo_tokens = Vec::new();
    for p in &prompts {
        let mut rb = Blocking::new(&[EOS], &dev)?;
        let mut session = Session::new(&model, fresh()?);
        let before = Window::Prefill.gemms();
        session.prefill(&mut rb, p, &opts)?;
        dev.synchronize()?;
        solo_prefill += Window::Prefill.gemms() - before;

        let mut session = Session::new(&model, fresh()?);
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut sampler,
            p,
            &opts,
            &mut |_: &[u32]| Ok(()),
        )?;
        solo_tokens.push(out.ids);
    }

    let mut sessions = vec![
        Session::new(&model, fresh()?),
        Session::new(&model, fresh()?),
    ];
    let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
    let before = Window::Prefill.gemms();
    let outs = Session::decode_batched(
        &mut sessions,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        &refs,
        &opts,
        &[EOS],
        &dev,
        &mut |_, _| Ok(()),
    )?;
    dev.synchronize()?;
    let batched = Window::Prefill.gemms() - before;
    let batched_tokens: Vec<Vec<u32>> = outs.into_iter().map(|o| o.ids).collect();
    eprintln!(
        "PREFILL_WINDOW_BATCHED solo_prefill={solo_prefill} batched_run={batched} solo={solo_tokens:?} batched={batched_tokens:?}"
    );
    assert!(solo_prefill > 0, "the solo prefills took no prefill launch");
    assert_eq!(
        batched, solo_prefill,
        "the batched run's prefill launches are its rows' solo prefills', and its decode rows take none"
    );
    assert_eq!(
        batched_tokens, solo_tokens,
        "every batched row decodes its solo run's tokens"
    );
    Ok(())
}

/// A prefill that fails leaves no window behind: a prompt past the model's
/// position ceiling is refused by the session's prefill and by a forward run
/// inside a prefill window, and afterwards this thread's two-row half matmul
/// is the one-row reduction's, row for row — the K=1 path's bits.
#[test]
fn a_failed_prefill_leaves_no_window_behind() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("prefill_window_failed", tiny_geometry())? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    Window::Prefill.set_arm(true);
    let too_long: Vec<u32> = (0..200).map(|i| 1 + i % 60).collect();

    let mut session = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?,
    );
    let mut rb = Blocking::new(&[EOS], &dev)?;
    let refused = session.prefill(&mut rb, &too_long, &DecodeOptions::new(4));
    assert!(
        refused.is_err(),
        "a prompt past the ceiling was not refused"
    );
    assert!(
        !Window::Prefill.is_open(),
        "the failed session prefill left its window open"
    );

    let mut cache = State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?;
    let input = lloom_tensor::Tensor::new(too_long.as_slice(), &dev)?.unsqueeze(0)?;
    let inside = {
        let _prefill = lloom_tensor::tuning::VerifyWindow::prefill();
        model.forward(&input, 0, &mut cache)
    };
    assert!(
        inside.is_err(),
        "the forward past the ceiling was not refused"
    );
    assert!(
        !Window::Prefill.is_open(),
        "the failed forward left its window open"
    );

    let w = lloom_tensor::Tensor::from_vec(
        (0..64 * 128)
            .map(|i| (i as f32 * 0.61).sin() * 0.05)
            .collect::<Vec<_>>(),
        (64, 128),
        &dev,
    )?
    .to_dtype(DType::F16)?;
    let x = lloom_tensor::Tensor::from_vec(
        (0..2 * 128)
            .map(|i| (i as f32 * 0.013).sin())
            .collect::<Vec<_>>(),
        (2, 128),
        &dev,
    )?
    .to_dtype(DType::F16)?;
    let bits = |t: &lloom_tensor::Tensor| -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let before = Window::Prefill.gemms();
    let rows = x.matmul(&w.t()?)?;
    for r in 0..2 {
        let one = x.narrow(0, r, 1)?.matmul(&w.t()?)?;
        assert_eq!(
            bits(&rows.narrow(0, r, 1)?)?,
            bits(&one)?,
            "row {r} after a failed prefill is not the one-row reduction's"
        );
    }
    assert_eq!(
        Window::Prefill.gemms(),
        before,
        "a matmul after the failed prefill counted on its window"
    );
    eprintln!(
        "PREFILL_WINDOW_FAILED session_err={} forward_err={}",
        refused.is_err(),
        inside.is_err()
    );
    Ok(())
}

/// Two sessions on two threads: one prefills through `Session::prefill`
/// again and again while the other runs batched K=1 decode steps
/// (`forward_rows`, two rows, no window). Every decode step's logits are
/// the ones the same step gives with nothing else running, the prefill
/// window's count is exactly the prefilling session's, and the same step
/// inside a window of its own gives other logits — so a window leaking from
/// the prefilling thread would show.
#[test]
fn a_session_decoding_beside_another_sessions_prefill_keeps_its_bits() -> Result<()> {
    use lloom::nn::Row;
    use std::sync::atomic::{AtomicBool, Ordering};
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let Some(fx) = Fixture::<Lfm2Tiny>::new("prefill_window_two_sessions", tiny_geometry())? else {
        return Ok(());
    };
    let cfg = fx.config(arms);
    let model = fx.model_in::<RewindableExact>(&cfg, Some(DType::F16), &dev)?;
    Window::Prefill.set_arm(true);
    let prompts: [Vec<u32>; 2] = [vec![3, 9, 27, 17], vec![5, 25, 13, 41, 2]];
    let decode_step = |windowed: bool| -> Result<Vec<u32>> {
        let mut caches = Vec::new();
        for p in &prompts {
            let mut cache = State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?;
            let input = lloom_tensor::Tensor::new(p.as_slice(), &dev)?.unsqueeze(0)?;
            model.forward(&input, 0, &mut cache)?;
            caches.push(cache);
        }
        let next = lloom_tensor::Tensor::new(&[7u32, 11], &dev)?.reshape((2, 1))?;
        let mut rows: Vec<Row<'_, State<RewindableExact>>> = caches
            .iter_mut()
            .zip(&prompts)
            .map(|(cache, p)| Row {
                pos: p.len(),
                cache,
            })
            .collect();
        let logits = {
            let _w = windowed.then(lloom_tensor::tuning::VerifyWindow::prefill);
            model.forward_rows(&next, &mut rows)?
        };
        Ok(logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let reference = decode_step(false)?;
    assert_ne!(
        decode_step(true)?,
        reference,
        "a window over the decode step moves nothing: this comparison could not see a leak"
    );
    let prefill_prompt: Vec<u32> = (0..37).map(|i| 1 + (i * 7) % 60).collect();
    let prefill_once = || -> Result<()> {
        let cache = State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut rb = Blocking::new(&[EOS], &dev)?;
        session.prefill(&mut rb, &prefill_prompt, &DecodeOptions::new(4))?;
        Ok(())
    };
    let before = Window::Prefill.gemms();
    prefill_once()?;
    dev.synchronize()?;
    let per_prefill = Window::Prefill.gemms() - before;
    assert!(
        per_prefill > 0,
        "the session's prefill took no window launch"
    );

    // Tells the prefilling thread to stop on every way out of the decoding
    // loop, an assertion's unwind included, so a failure fails rather than
    // leaving the scope waiting on a thread that never stops.
    struct Release<'a>(&'a AtomicBool);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let done = AtomicBool::new(false);
    let before = Window::Prefill.gemms();
    let (prefills, steps) = std::thread::scope(|s| -> Result<(u64, usize)> {
        let prefilling = s.spawn(|| -> Result<u64> {
            let mut n = 0u64;
            while !done.load(Ordering::Acquire) || n == 0 {
                prefill_once()?;
                n += 1;
            }
            Ok(n)
        });
        let decoding = (|| -> Result<usize> {
            let _release = Release(&done);
            let mut steps = 0usize;
            for _ in 0..24 {
                assert_eq!(
                    decode_step(false)?,
                    reference,
                    "a decode step beside another session's prefill moved"
                );
                steps += 1;
            }
            Ok(steps)
        })();
        let n = prefilling.join().expect("the prefilling thread panicked")?;
        Ok((n, decoding?))
    })?;
    dev.synchronize()?;
    assert_eq!(
        Window::Prefill.gemms() - before,
        prefills * per_prefill,
        "the window counted launches other than the prefilling session's"
    );
    eprintln!(
        "PREFILL_WINDOW_TWO_SESSIONS decode_steps={steps} prefills={prefills} launches_per_prefill={per_prefill}"
    );
    Ok(())
}
