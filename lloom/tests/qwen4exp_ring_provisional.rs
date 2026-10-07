//! Provisional steps on a lookahead ring (`Provisional::Behind`,
//! `Session::decode` on the ring's fixed cadence at `lookahead = 1`): each
//! device-planned step is settled after the NEXT step is encoded, while the
//! GPU runs it. A step that computed on a missing expert is redone and the
//! forward after it undone: the cache two steps back through the window held
//! behind the standing one, the ring's draw taken back (its slot written
//! again under the next stamp), the PLE host fill's token taken back, the
//! draw made again and the forward encoded again. The stream is the
//! resident model's, token for token, with redraws > 0 — with the PLE's host
//! fill (its producer rewound) and without it.
//!
//! Metal only. Each run on a thread of its own and bounded: a take-back that
//! leaves a wait unsignalled parks the queue rather than failing.
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen4exp::testing::{self, Qwen4ExpTiny};
use lloom::models::qwen4exp::{
    Config, ExpertResidency, LoadOptions, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp,
    SlotBudget, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom::testing::fixture::Fixture;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 24;
/// An id no draw produces: the runs end on their step count.
const NO_EOS: u32 = u32::MAX - 1;
const PARKED: Duration = Duration::from_secs(120);

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
        drafter_bytes: 0,
    }
}

fn table_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-ring-prov-{}-{tag}.gguf", std::process::id()))
}

/// A PLE table in a GGUF, as the loader names it.
fn write_table(cfg: &Config, rows: usize, path: &Path) -> Result<()> {
    let table = testing::wt((rows, cfg.ple_head_dim), 3.0, &Device::Cpu);
    let q = QTensor::quantize(&table, GgmlDType::F32)?;
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &[("per_layer_token_embd.weight", &q)])?;
    let _ = Content::read(&mut File::open(path)?)?;
    Ok(())
}

/// The PLE attached at layer 1 over the table at `path` (the fixture of
/// `qwen4exp_host_feed`).
fn attach_ple<R: ExpertResidency>(
    model: &mut Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    path: &Path,
    dev: &Device,
) -> Result<()> {
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::<CrownExact>::new(
        testing::lin(hc_dim, ple_dim, 21.0, dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::wt((hc_dim, 4), 23.0, dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, PleTable::Mmap(MmapPleTable::open(path)?), block)
}

fn watched<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(PARKED) {
        Ok(v) => v,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: no answer in {PARKED:?} — a wait was left parked")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{what}: the run panicked (above)")
        }
    }
}

fn ring(dev: &Device) -> Result<Ring> {
    Ok(Ring::new(&[NO_EOS], dev, 64, EosRead::Poll(PollMechanism::default()))?.with_poll_budget(20_000))
}

fn ring_options() -> Result<DecodeOptions> {
    let mut opts = DecodeOptions::new(STEPS);
    opts.lookahead = 1;
    opts.budget = CommitBudget::steps(4)?;
    Ok(opts)
}

fn session_decode<M: CausalLM, R: lloom::generation::ReadbackStrategy>(
    model: &M,
    cache: M::Cache,
    rb: &mut R,
    opts: &DecodeOptions,
    sampling: Sampling,
) -> Result<DecodeOut> {
    let mut session = Session::new(model, cache);
    let mut sampler = LogitsProcessor::from_sampling(7, sampling);
    session.decode(
        &mut IdentityProposer,
        rb,
        &mut sampler,
        &PROMPT,
        opts,
        &mut |_| Ok(()),
    )
}

/// The resident model, then the streamed one on the ring at `lookahead = 1`,
/// each with the PLE when `ple`, drawing by `sampling`. The reference runs
/// on the blocking arm under argmax, and on the same ring otherwise (the
/// ring's temperature draw is the device's gumbel-max, the blocking arm's
/// the host's): a resident model declines provisional steps, so its ring
/// run takes nothing back.
fn check(tag: &'static str, ple: bool, sampling: Sampling) -> Result<()> {
    let path = table_path(tag);
    let want = {
        let path = path.clone();
        let sampling = sampling.clone();
        watched(&format!("{tag}: resident"), move || -> Result<Option<Vec<u32>>> {
            let Ok(dev) = Device::new_metal(0) else {
                return Ok(None);
            };
            if lloom::testing::fixture::Backend::current().at_scale() {
                return Ok(None);
            }
            let Some(fx) = Fixture::<Qwen4ExpTiny>::new(&format!("{tag}_resident"), Config::tiny())? else {
                return Ok(None);
            };
            let cfg = fx.config(|_| {});
            if ple {
                write_table(&cfg, 80, &path)?;
            }
            let mut model = fx.resident::<CrownExact>(&cfg, &dev, &LoadOptions::default())?;
            if ple {
                attach_ple(&mut model, &cfg, &path, &dev)?;
            }
            let out = if sampling == Sampling::ArgMax {
                let mut rb = Blocking::new(&[NO_EOS], &dev)?;
                let opts = DecodeOptions::new(STEPS);
                session_decode(&model, model.new_cache(&cfg, &dev)?, &mut rb, &opts, sampling)?
            } else {
                let mut rb = ring(&dev)?;
                let opts = ring_options()?;
                let out =
                    session_decode(&model, model.new_cache(&cfg, &dev)?, &mut rb, &opts, sampling)?;
                assert!(!out.stats.provisional, "a resident model took provisional steps");
                out
            };
            Ok(Some(out.ids))
        })?
    };
    let Some(want) = want else {
        return Ok(());
    };
    let got = {
        let path = path.clone();
        let sampling = sampling.clone();
        watched(&format!("{tag}: streamed ring"), move || -> Result<(DecodeOut, u64)> {
            let dev = Device::new_metal(0)?;
            let fx = Fixture::<Qwen4ExpTiny>::new(&format!("{tag}_ring"), Config::tiny())?
                .expect("the tiny fixture built above");
            let cfg = fx.config(|_| {});
            let mut model = fx.streamed::<CrownExact>(&cfg, &dev, &LoadOptions::default(), &stream(6))?;
            if ple {
                attach_ple(&mut model, &cfg, &path, &dev)?;
            }
            let mut rb = ring(&dev)?;
            let opts = ring_options()?;
            let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
            let mut sampler = LogitsProcessor::from_sampling(7, sampling);
            let out = session.decode(
                &mut IdentityProposer,
                &mut rb,
                &mut sampler,
                &PROMPT,
                &opts,
                &mut |_| Ok(()),
            )?;
            assert!(!model.step_settle_trailing(), "{tag}: a provisional step left unsettled");
            assert!(!session.cache_mut().window_held(), "{tag}: a window left held");
            Ok((out, session.cache_mut().provisional_redos))
        })?
    };
    std::fs::remove_file(&path).ok();
    let (out, redos) = got;
    let rc = out.stats.readback;
    println!(
        "RINGPROV {tag} provisional={} redraws={} provisional_redos={redos} ring_publishes={} ring_retracted={} host_feed={:?}",
        out.stats.provisional,
        out.stats.redraws,
        rc.ring_publishes,
        rc.ring_retracted,
        out.stats.host_feed.as_ref().map(|f| (f.forwards, f.fills)),
    );
    assert_eq!(out.ids, want, "{tag}: the ring's provisional stream differs from the resident one");
    assert!(out.stats.provisional, "{tag}: the ring run's steps were not provisional");
    assert!(
        out.stats.redraws > 0,
        "{tag}: no step was redone behind the next forward — the identity says nothing about \
         the take-back"
    );
    assert_eq!(rc.ring_retracted, out.stats.redraws, "{tag}: a redraw without a take-back");
    assert_eq!(
        rc.ring_publishes,
        out.ids.len() as u64 + rc.ring_eos_publishes + rc.ring_discarded + rc.ring_retracted,
        "{tag}: the ring's publishes do not reconcile"
    );
    if ple {
        assert!(out.stats.host_feed.is_some(), "{tag}: the ring run took no host fill");
    }
    Ok(())
}

#[test]
fn a_ring_step_redone_behind_the_next_forward_is_the_resident_stream_metal() -> Result<()> {
    check("nople", false, Sampling::ArgMax)
}

#[test]
fn the_host_fill_takes_its_token_back_with_the_draw_metal() -> Result<()> {
    check("ple", true, Sampling::ArgMax)
}

/// A temperature draw on the ring is the device's gumbel-max over noise from
/// the sampler's rng: a draw made again must start from the rng the taken-
/// back draw started from, or every later token moves.
#[test]
fn a_draw_made_again_on_the_ring_puts_the_sampler_back_first_metal() -> Result<()> {
    check("temp", false, Sampling::All { temperature: 1.0 })
}
