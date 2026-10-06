//! Which of LFM2's zero sites read their zeros, measured rather than argued
//! (`docs/pass-plan-design.md` §18): each site's memory filled with a NaN,
//! then with a finite sentinel, instead of zeros, on every conv and KV arm,
//! and the logits compared bit for bit with the unpoisoned run.
//!
//! A site that moves under NaN and not under the finite sentinel is read only
//! against a zero weight (`0 × NaN` is NaN, `0 × x` is 0); one that moves
//! under both is read as a real zero; one that moves under neither has no
//! output that depends on it. A site the run never reached is reported as
//! such, never as clean. The expected table is pinned below, and the
//! unpoisoned reference is taken again after every poisoned run, so a site
//! that skips its fill and reads a recycled buffer's sentinel shows too.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{
    AttnImpl, Config, ConvState, KvAppend, Model, Shape as Lfm2Shape, State,
};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::ops::KvQuant;
use lloom::nn::zeros;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};

/// The poison is process-wide and so is the pool it recycles through.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// The site was not reached by this arm's workloads.
    Unreached,
    /// Reached, and every logit bit-identical to the reference.
    Unread,
    /// Moves under NaN only: read, against a zero weight.
    ReadTimesZero,
    /// Moves under both sentinels: read as a real zero.
    Read,
}

struct Arm {
    name: &'static str,
    set: fn(&mut Config),
}

fn arms() -> Vec<Arm> {
    vec![
        Arm {
            name: "generic/cat/shuffle",
            set: |c| {
                c.attn_impl = AttnImpl::Generic;
                c.kv_append = KvAppend::Cat;
                c.conv_state = ConvState::Shuffle;
            },
        },
        Arm {
            name: "crown",
            set: lloom::models::lfm2_presets::crown,
        },
        Arm {
            name: "crown+int8",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.kv_quant = KvQuant::Int8;
            },
        },
        Arm {
            name: "crown+int4",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.kv_quant = KvQuant::Int4;
            },
        },
        Arm {
            name: "flash/in-place",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.attn_impl = AttnImpl::FlashDecoding;
            },
        },
        Arm {
            name: "pool",
            set: lloom::models::lfm2_presets::pool,
        },
        Arm {
            name: "sliding:0:2",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.conv_state = ConvState::SlidingRing { k: 0, slack: 2 };
                c.conv_fused = false;
            },
        },
        Arm {
            name: "rotating:4 unfused",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.conv_state = ConvState::RotatingRing { k: 4 };
                c.conv_fused = false;
            },
        },
        Arm {
            name: "rotating:4 fused",
            set: |c| {
                lloom::models::lfm2_presets::crown(c);
                c.conv_state = ConvState::RotatingRing { k: 4 };
            },
        },
    ]
}

/// One sequence: prompts fed in `tiles`-wide forwards (a later prompt is a
/// second turn on the same cache), each followed by `decode` single steps.
struct Workload {
    turns: &'static [(&'static [usize], usize)],
}

/// The workloads every arm runs, and what each reaches:
/// - a two-token prompt: the prefill's carry is shorter than the window;
/// - a one-token prompt: the first pass is a decode step with no state;
/// - three tiles, a long decode and a second turn: the ring's slack, the
///   sliding ring's compaction, the KV slab's growth (8-token buckets) and
///   the warm carry.
const WORKLOADS: &[Workload] = &[
    Workload {
        turns: &[(&[2], 6)],
    },
    Workload {
        turns: &[(&[1], 6)],
    },
    Workload {
        turns: &[(&[5, 5, 3], 20), (&[4, 2], 5)],
    },
];

fn model<P: Lfm2Shape>(cfg: &Config, dev: &Device) -> Result<Model<P>> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default()).model::<P>(dev)
}

/// Every logit of every workload, as bits.
fn run<P: Lfm2Shape>(model: &Model<P>, cfg: &Config, dev: &Device) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut id = 1u32;
    for w in WORKLOADS {
        let mut cache = State::<P>::new(true, DType::F16, cfg, dev)?.with_kv_bucket(8)?;
        let mut pos = 0usize;
        for &(tiles, decode) in w.turns {
            for &n in tiles {
                let ids: Vec<u32> = (0..n as u32).map(|i| (id + 7 * i) % 60 + 2).collect();
                id += 3;
                let input = Tensor::from_vec(ids, (1, n), dev)?;
                let logits = model.forward(&input, pos, &mut cache)?;
                pos += n;
                push(&mut out, &logits)?;
            }
            for _ in 0..decode {
                let input = Tensor::from_vec(vec![id % 60 + 2], (1, 1), dev)?;
                id += 5;
                let logits = model.forward(&input, pos, &mut cache)?;
                pos += 1;
                push(&mut out, &logits)?;
            }
        }
    }
    Ok(out)
}

fn push(out: &mut Vec<u32>, logits: &Tensor) -> Result<()> {
    let v = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    out.extend(v.iter().map(|x| x.to_bits()));
    Ok(())
}

/// `(moved, fills)` for one site under one sentinel, after checking that the
/// unpoisoned run still reproduces `reference`.
fn poisoned<P: Lfm2Shape>(
    model: &Model<P>,
    cfg: &Config,
    dev: &Device,
    reference: &[u32],
    site: &str,
    value: &str,
) -> Result<(bool, u64)> {
    zeros::set_poison(site, value)?;
    let before = zeros::poisoned();
    let got = run::<P>(model, cfg, dev);
    let fills = zeros::poisoned() - before;
    zeros::set_poison("", "nan")?;
    let got = got?;
    let again = run::<P>(model, cfg, dev)?;
    assert!(
        again == reference,
        "the unpoisoned run moved after poisoning {site} with {value}: a site that skips \
         its fill read a recycled buffer's sentinel"
    );
    Ok((got != reference, fills))
}

fn classify(arm: &Arm, dev: &Device) -> Result<Vec<(&'static str, Verdict)>> {
    let mut cfg = TinyCheckpoint::<Lfm2Tiny>::template().cfg;
    (arm.set)(&mut cfg);
    // A multi-tile prefill carries the conv context, as the engine runs it.
    cfg.conv_prefill_ctx = true;
    let key = cfg.exact_key();
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => classify_at::<P>(arm, &cfg, dev))
}

/// `classify` at the profile the arm's config names.
fn classify_at<P: Lfm2Shape>(
    arm: &Arm,
    cfg: &Config,
    dev: &Device,
) -> Result<Vec<(&'static str, Verdict)>> {
    let cfg = cfg.clone();
    let model = model::<P>(&cfg, dev)?;
    zeros::set_poison("", "nan")?;
    let reference = run(&model, &cfg, dev)?;
    assert_eq!(
        reference,
        run(&model, &cfg, dev)?,
        "{}: the reference is not stable",
        arm.name
    );
    assert!(
        reference.iter().all(|b| f32::from_bits(*b).is_finite()),
        "{}: the reference has a non-finite logit",
        arm.name
    );
    let mut table = Vec::new();
    for &site in zeros::SITES {
        let (nan_moved, nan_fills) = poisoned(&model, &cfg, dev, &reference, site, "nan")?;
        let (fin_moved, fin_fills) = poisoned(&model, &cfg, dev, &reference, site, "finite")?;
        assert_eq!(
            nan_fills, fin_fills,
            "{} {site}: the two sentinels filled differently",
            arm.name
        );
        let v = match (nan_fills, nan_moved, fin_moved) {
            (0, false, false) => Verdict::Unreached,
            (0, ..) => panic!("{} {site}: moved with nothing filled", arm.name),
            (_, false, false) => Verdict::Unread,
            (_, true, false) => Verdict::ReadTimesZero,
            (_, _, true) => Verdict::Read,
        };
        eprintln!(
            "[zero sites] {:<20} {site:<24} fills={nan_fills:<3} {v:?}",
            arm.name
        );
        table.push((site, v));
    }
    Ok(table)
}

#[test]
fn every_zero_site_reads_its_zeros_as_the_table_says() -> Result<()> {
    use Verdict::*;
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    // Per arm, in `zeros::SITES` order: ring_slack, ring_history, left_pad,
    // state, compact_pad, first_state, kv.slab, kv.quant, kv.pages.
    #[rustfmt::skip]
    let expected: &[(&str, [Verdict; 9])] = &[
        ("generic/cat/shuffle", [Unreached, Unreached, Unread, Unread, Unreached, Read, Unreached, Unreached, Unreached]),
        ("crown", [Unreached, Unreached, Unread, Unread, Unreached, Read, Unread, Unreached, Unreached]),
        ("crown+int8", [Unreached, Unreached, Unread, Unread, Unreached, Read, Unreached, Unread, Unreached]),
        ("crown+int4", [Unreached, Unreached, Unread, Unread, Unreached, Read, Unreached, Unread, Unreached]),
        ("flash/in-place", [Unreached, Unreached, Unread, Unread, Unreached, Read, Unread, Unreached, Unreached]),
        ("pool", [Unreached, Unread, ReadTimesZero, Unread, Unreached, Read, Unreached, Unreached, Unread]),
        ("sliding:0:2", [Unread, Unreached, Unread, Unread, Unread, Read, Unread, Unreached, Unreached]),
        ("rotating:4 unfused", [Unreached, ReadTimesZero, ReadTimesZero, Unread, Unreached, Read, Unread, Unreached, Unreached]),
        ("rotating:4 fused", [Unreached, ReadTimesZero, ReadTimesZero, Unread, Unreached, Read, Unread, Unreached, Unreached]),
    ];
    assert_eq!(zeros::SITES.len(), 9, "a site was added: give it a column");
    let mut wrong = Vec::new();
    for arm in arms() {
        let got = classify(&arm, &dev)?;
        let want = expected
            .iter()
            .find(|(n, _)| *n == arm.name)
            .map(|(_, v)| v)
            .expect("every arm has an expected row");
        for ((site, g), w) in got.iter().zip(want.iter()) {
            if g != w {
                wrong.push(format!(
                    "{} {site}: measured {g:?}, expected {w:?}",
                    arm.name
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "zero-site classification moved:\n{}",
        wrong.join("\n")
    );
    Ok(())
}

/// Logits of: a prompt and some decode steps, then, for each warm tile, the
/// conv state filled with NaN when `stale_nan` (what a recycled buffer may
/// hold), the tile, and more decode steps. Only the warm half is returned.
fn warm_run<P: Lfm2Shape>(
    model: &Model<P>,
    cfg: &Config,
    dev: &Device,
    stale_nan: bool,
) -> Result<Vec<u32>> {
    let mut cache = State::<P>::new(true, DType::F16, cfg, dev)?.with_kv_bucket(8)?;
    let feed = |cache: &mut State<P>, ids: Vec<u32>, pos: usize| -> Result<Tensor> {
        let n = ids.len();
        model.forward(&Tensor::from_vec(ids, (1, n), dev)?, pos, cache)
    };
    let mut pos = 0usize;
    feed(&mut cache, vec![3, 9, 14, 20, 27], pos)?;
    pos += 5;
    // More decode steps than any arm's ring is wide: every history slot is
    // written with a real column before the warm tiles.
    for t in 0..9u32 {
        feed(&mut cache, vec![t * 7 % 60 + 2], pos)?;
        pos += 1;
    }
    let mut out = Vec::new();
    for tile in [vec![5u32, 41], vec![11, 12, 13, 44, 45]] {
        if stale_nan {
            for block in 0..cfg.num_hidden_layers {
                if let Some(state) = cache.conv_state_for_test(block) {
                    state.const_set(half::f16::NAN.into())?;
                }
            }
        }
        let n = tile.len();
        push(&mut out, &feed(&mut cache, tile, pos)?)?;
        pos += n;
        for t in 0..5u32 {
            push(&mut out, &feed(&mut cache, vec![t * 13 % 60 + 2], pos)?)?;
            pos += 1;
        }
    }
    Ok(out)
}

/// A multi-token pass onto a warm cache rebuilds the conv state in the
/// storage the cache owns: the carry, then the regions around it set in
/// place. Whatever that storage held before must not reach the output, on
/// any arm: filling it with NaN first leaves every logit bit-identical. This
/// is the test the in-place zeroing answers to — the rotating ring's tail
/// (folded against a zero weight) and its left pad — since a stale column is
/// otherwise finite and `0 × x` hides it. Without the conv context, so the
/// pass reads nothing of the previous window by design.
#[test]
fn a_warm_prefill_lets_no_stale_conv_column_reach_the_output() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    zeros::set_poison("", "nan")?;
    for arm in arms() {
        let mut cfg = TinyCheckpoint::<Lfm2Tiny>::template().cfg;
        (arm.set)(&mut cfg);
        cfg.conv_prefill_ctx = false;
        let key = cfg.exact_key();
        lloom::profile::dispatch!(key, lloom::profile::LFM2, P => warm_case::<P>(&arm, &cfg, &dev))?;
    }
    Ok(())
}

/// One arm of `a_warm_prefill_lets_no_stale_conv_column_reach_the_output`,
/// at the profile its config names.
fn warm_case<P: Lfm2Shape>(arm: &Arm, cfg: &Config, dev: &Device) -> Result<()> {
    let model = model::<P>(&cfg, &dev)?;
    let clean = warm_run::<P>(&model, &cfg, &dev, false)?;
    assert!(
        clean.iter().all(|b| f32::from_bits(*b).is_finite()),
        "{}: the clean run has a non-finite logit",
        arm.name
    );
    let stale = warm_run::<P>(&model, &cfg, &dev, true)?;
    let bad = stale
        .iter()
        .filter(|b| !f32::from_bits(**b).is_finite())
        .count();
    eprintln!(
        "[warm prefill] {:<20} non-finite logits after a NaN-filled state: {bad}",
        arm.name
    );
    assert!(
        stale == clean,
        "{}: a column of the previous conv state reached the output after a warm \
         prefill ({bad} non-finite logits)",
        arm.name
    );
    Ok(())
}
