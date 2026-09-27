#![cfg(feature = "metal")]
//! `Device::words_publish`: a host that observes the stamp must read the words
//! behind it, not the previous round's.
//!
//! The shape is the speculative verify's readback (`lloom`'s `VerifyWords`):
//! one CPU-mapped `out` and one stamp word reused round after round, the words
//! a `cat` of an argmax that sits behind a matmul chain (the verify's forward)
//! and the host-uploaded proposals, one publish, a commit, then the host spins
//! on the stamp and reads the words the moment it moves. Every word changes
//! every round, so a stamp that reached the host before its words reads as the
//! previous round's value (or the zero fill on the first).
//!
//! Each round's words are checked against their expected values; the stale
//! ones are classified by what they read. `the_published_words_are_never_older_
//! than_their_stamp` gates at a fixed count; the ignored `words_publish_stale_
//! rate` counts at a count and on arms the environment picks, and asserts
//! nothing (`LLOOM_WORDS_STRESS=<rounds>`, `LLOOM_WORDS_STRESS_ARMS="c0k9 c4k9
//! c4k32"` — `c` the matmul chain length, `k` the proposals).

use lloom_tensor::{DType, Device, Result, Tensor, D};

const DIM: usize = 1024;

#[derive(Clone, Copy, Debug)]
struct Arm {
    /// Identity matmuls of `DIM²` the argmax waits behind (0: none).
    chain: usize,
    /// Proposals per round; `2k + 1` words are published.
    k: usize,
}

/// What a round read, per stale word: the previous round's value, the zero
/// fill, or neither.
#[derive(Default, Debug)]
struct Tally {
    rounds: usize,
    stale_rounds: usize,
    previous: usize,
    zero: usize,
    other: usize,
    first: Option<(usize, Vec<u32>, Vec<u32>)>,
}

fn token(round: usize, i: usize) -> u32 {
    (1 + (round * 131 + i * 17) % (DIM - 2)) as u32
}

fn proposal(round: usize, k: usize, i: usize) -> u32 {
    (5000 + round * k + i) as u32
}

fn expected(round: usize, k: usize) -> Vec<u32> {
    (0..=k)
        .map(|i| token(round, i))
        .chain((0..k).map(|i| proposal(round, k, i)))
        .collect()
}

fn run(arm: Arm, rounds: usize) -> Result<Option<Tally>> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(None);
    };
    let Device::Metal(metal) = &device else {
        return Ok(None);
    };
    let Arm { chain, k } = arm;
    let n = 2 * k + 1;
    let eye = Tensor::eye(DIM, DType::F32, &device)?;
    let cap = n.max(64).next_power_of_two();
    let out = Tensor::from_vec(vec![0u32; cap], cap, &device)?;
    let step = Tensor::from_vec(vec![0u32], 1, &device)?;
    // SAFETY: both tensors are held for the whole run and never reallocated;
    // the words are read only after the stamp names this round.
    let out_ptr = unsafe { out.peek_scalar_ptr::<u32>(0) }.expect("out is CPU-mapped");
    let step_ptr = unsafe { step.peek_scalar_ptr::<u32>(0) }.expect("step is CPU-mapped");
    let mut tally = Tally {
        rounds,
        ..Tally::default()
    };
    let mut previous = vec![0u32; n];
    let mut got = vec![0u32; n];
    for round in 0..rounds {
        let want = expected(round, k);
        let mut onehot = vec![0f32; (k + 1) * DIM];
        for i in 0..=k {
            onehot[i * DIM + token(round, i) as usize] = 1.0;
        }
        let mut x = Tensor::from_vec(onehot, (k + 1, DIM), &device)?;
        if chain > 0 {
            let mut m = eye.clone();
            for _ in 0..chain {
                m = m.matmul(&eye)?;
            }
            x = x.matmul(&m)?;
        }
        let argmax = x.argmax(D::Minus1)?;
        let proposed = Tensor::from_vec(want[k + 1..].to_vec(), k, &device)?;
        let words = Tensor::cat(&[&argmax, &proposed], 0)?;
        let generation = round as u32 + 1;
        device.words_publish(&words, n, &out, &step, generation)?;
        metal.flush_and_prune()?;
        let mut spins = 0u64;
        // SAFETY: `step_ptr` is the mapped stamp of `step`, held above.
        while unsafe { std::ptr::read_volatile(step_ptr) } != generation {
            spins += 1;
            if spins > 4_000_000_000 {
                lloom_tensor::bail!("round {round}: stamp {generation} never published");
            }
            std::hint::spin_loop();
        }
        for (i, g) in got.iter_mut().enumerate() {
            // SAFETY: `out_ptr` maps `cap >= n` words of `out`, held above.
            *g = unsafe { std::ptr::read_volatile(out_ptr.add(i)) };
        }
        if got != want {
            tally.stale_rounds += 1;
            for i in 0..n {
                if got[i] == want[i] {
                    continue;
                }
                if got[i] == previous[i] {
                    tally.previous += 1;
                } else if got[i] == 0 {
                    tally.zero += 1;
                } else {
                    tally.other += 1;
                }
            }
            if tally.first.is_none() {
                tally.first = Some((round, want.clone(), got.clone()));
            }
        }
        previous = want;
    }
    device.synchronize()?;
    Ok(Some(tally))
}

fn report(arm: Arm, t: &Tally) -> String {
    format!(
        "STRESS words_publish chain={} k={}: {} of {} rounds stale ({} words previous, {} zero, {} other){}",
        arm.chain,
        arm.k,
        t.stale_rounds,
        t.rounds,
        t.previous,
        t.zero,
        t.other,
        t.first
            .as_ref()
            .map(|(r, w, g)| format!("; first at round {r}: want {w:?} read {g:?}"))
            .unwrap_or_default()
    )
}

/// `metal-load`'s contention, in process: a second Metal device instance —
/// its own command queue — running f16 matmuls until `stop` is raised.
fn contend(stop: &std::sync::atomic::AtomicBool) -> Result<u64> {
    let device = Device::new_metal(0)?;
    let a = Tensor::randn(0f32, 1f32, (2048, 2048), &device)?.to_dtype(DType::F16)?;
    let mut n = 0u64;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = a
            .matmul(&a)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        n += 1;
    }
    Ok(n)
}

/// The verify's shape at the draft block LFM2.5-2.6B-DSpark runs (`k = 9`,
/// 19 words): quiet with the forward's latency ahead of the publish, and
/// with a second queue contending for the GPU, where the stale read was
/// measured about a hundred times as often as quiet. A race gate: it kills a
/// stamp that runs ahead of its words at a rate, not by construction
/// (`lloom`'s `ring_publish_binds` reads the ordering off the dispatch trace).
#[test]
fn the_published_words_are_never_older_than_their_stamp() -> Result<()> {
    let quiet = Arm { chain: 2, k: 9 };
    let Some(t) = run(quiet, 1500)? else {
        return Ok(());
    };
    let line = report(quiet, &t);
    eprintln!("{line} (quiet)");
    assert_eq!(t.stale_rounds, 0, "{line} (quiet)");

    let loaded = Arm { chain: 0, k: 9 };
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (t, matmuls) = std::thread::scope(|s| {
        let load = s.spawn(|| contend(&stop));
        std::thread::sleep(std::time::Duration::from_millis(200));
        let t = run(loaded, 3000);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        (t, load.join().expect("the contending thread panicked"))
    });
    let (t, matmuls) = (t?.expect("a Metal device"), matmuls?);
    let line = report(loaded, &t);
    eprintln!("{line} (under {matmuls} contending matmuls)");
    assert!(matmuls > 0, "the contending queue ran nothing");
    assert_eq!(t.stale_rounds, 0, "{line} (under load)");
    Ok(())
}

#[test]
#[ignore = "a rate, for the A/B; LLOOM_WORDS_STRESS=<rounds>"]
fn words_publish_stale_rate() -> Result<()> {
    let rounds: usize = std::env::var("LLOOM_WORDS_STRESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let arms = std::env::var("LLOOM_WORDS_STRESS_ARMS").unwrap_or_else(|_| "c0k9 c2k9".into());
    for spec in arms.split_whitespace() {
        let (c, k) = spec
            .trim_start_matches('c')
            .split_once('k')
            .unwrap_or_else(|| panic!("arm `{spec}`: want c<chain>k<proposals>"));
        let arm = Arm {
            chain: c.parse().expect("chain"),
            k: k.parse().expect("k"),
        };
        if let Some(t) = run(arm, rounds)? {
            println!("{}", report(arm, &t));
        }
    }
    Ok(())
}
