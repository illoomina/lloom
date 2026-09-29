//! Tiny Qwen4-exp models built, decoded and dropped on ONE Metal device from
//! several threads at once return the ids the same decode returns alone.
//!
//! A `Device` is `Clone + Send + Sync`, and every clone of one device shares
//! its command stream, pools and fence map; the decodes here share nothing
//! else. Each thread loops: build a resident tiny model on the shared device,
//! decode greedily on its arm, drop the model. The ring arms wait for the
//! device (`synchronize`) at construction and at the final drain, which is
//! what reaches the other threads' in-flight work.
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Config;
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::{Device, Result};

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 12;
const THREADS: usize = 4;
const REPS: usize = 6;

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Blocking,
    /// The ring on the fixed cadence, lookahead 0.
    Ring,
}

/// Build, decode `PROMPT` greedily on `arm`, drop: the ids.
fn decode_once(dev: &Device, arm: Arm) -> Result<Vec<u32>> {
    let cfg = tiny();
    let model = testing::resident_model(&cfg, dev);
    let cache = model.new_cache(&cfg, dev)?;
    let mut session = Session::new(&model, cache);
    let mut spec = IdentityProposer;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut sink = |_: &[u32]| -> Result<()> { Ok(()) };
    let mut opts = DecodeOptions::new(STEPS);
    let out = match arm {
        Arm::Blocking => {
            let mut rb = Blocking::new(&[u32::MAX], dev)?;
            session.decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
        }
        Arm::Ring => {
            opts.lookahead = 0;
            opts.budget = CommitBudget::steps(4)?;
            let mut rb = Ring::new(
                &[u32::MAX],
                dev,
                64,
                EosRead::Poll(PollMechanism::default()),
            )?
            .with_poll_budget(20_000);
            session.decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
        }
    };
    Ok(out.ids)
}

/// `(runs whose ids differ from the reference, runs)` over `THREADS` threads,
/// each on `device(thread)`, alternating the arms across threads.
fn churn(device: impl Fn(usize) -> Device, reference: &[u32]) -> Result<(usize, usize)> {
    let workers: Vec<_> = (0..THREADS)
        .map(|t| {
            let dev = device(t);
            let arm = if t % 2 == 0 { Arm::Blocking } else { Arm::Ring };
            let reference = reference.to_vec();
            std::thread::spawn(move || -> Result<usize> {
                let mut bad = 0;
                for _ in 0..REPS {
                    let ids = decode_once(&dev, arm)?;
                    if ids != reference {
                        eprintln!("thread {t} {arm:?}: {ids:?} != {reference:?}");
                        bad += 1;
                    }
                }
                Ok(bad)
            })
        })
        .collect();
    let mut bad = 0;
    for w in workers {
        bad += w.join().expect("a worker panicked")?;
    }
    Ok((bad, THREADS * REPS))
}

/// The reference: one decode, alone on its device. Both arms agree on it.
fn reference() -> Result<Vec<u32>> {
    let dev = Device::new_metal(0)?;
    let blocking = decode_once(&dev, Arm::Blocking)?;
    let ring = decode_once(&dev, Arm::Ring)?;
    assert_eq!(blocking, ring, "the arms disagree alone on one device");
    assert_eq!(blocking.len(), STEPS, "the reference stopped early");
    Ok(blocking)
}

/// The control: one device per thread.
#[test]
fn threads_on_their_own_devices_decode_the_reference() -> Result<()> {
    let reference = reference()?;
    let (bad, total) = churn(
        |_| Device::new_metal(0).expect("a Metal device"),
        &reference,
    )?;
    eprintln!("own devices: {bad} of {total} runs wrong");
    assert_eq!(bad, 0, "{bad} of {total} runs wrong on per-thread devices");
    Ok(())
}

/// The case: every thread on clones of one device.
#[test]
fn threads_sharing_one_device_decode_the_reference() -> Result<()> {
    let reference = reference()?;
    let dev = Device::new_metal(0)?;
    let (bad, total) = churn(|_| dev.clone(), &reference)?;
    eprintln!("one shared device: {bad} of {total} runs wrong");
    assert_eq!(bad, 0, "{bad} of {total} runs wrong on one shared device");
    Ok(())
}
