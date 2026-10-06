//! A sequence opened by a decode step on `ConvState::SlidingRing`.
//!
//! The sliding ring is bit-identical to the shuffle by construction: its live
//! window is `l_cache` contiguous slots ending at the write, in the shuffle's
//! slot order (`DESIGN.md` §10.2e). That holds only if the first write lands
//! where a full window fits behind it. A one-token first pass is a decode step
//! on a fresh cache, so the window must open at the slot a prefill leaves the
//! newest column in, with the zeroed first state behind it standing where the
//! shuffle's zero history stands.
//!
//! The reference is the shuffle, a different mechanism over the same weights
//! (`CONTRIBUTING.md` §3.1b), compared bit for bit on every pass; and the slot
//! a one-token open leaves is compared with the slot a multi-token open leaves,
//! both read from the model rather than restated here.

use lloom::models::lfm2::{Config, ConvState, Model, Shape as Lfm2Shape, State};
use lloom::models::lfm2_testing::{run_dtype, Lfm2Tiny};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};

/// Sliding arms: slack 1 compacts on every other token, slack 16 not until
/// well into the decode; `k > 0` puts history slots behind the window.
const SLIDING: &[(usize, usize)] = &[(0, 1), (0, 2), (0, 16), (2, 3), (3, 1)];

/// One sequence, every pass opened on a fresh cache by a one-token prompt:
/// `(tile, decode)` pairs, a tile being a multi-token pass onto the warm cache
/// (a second turn), each followed by `decode` single steps.
const WORKLOADS: &[&[(usize, usize)]] = &[
    // The open, a decode long enough to compact every arm, a warm tile.
    &[(1, 24), (4, 6)],
    // A warm tile straight after the open: its left context is read from
    // the window the one-token pass opened.
    &[(1, 0), (3, 8)],
];

fn config(conv_state: ConvState) -> Config {
    let mut cfg = TinyCheckpoint::<Lfm2Tiny>::template().cfg;
    cfg.conv_state = conv_state;
    cfg.conv_fused = false;
    cfg.conv_prefill_ctx = true;
    cfg
}

fn model<P: Lfm2Shape>(cfg: &Config, dev: &Device) -> Result<Model<P>> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default()).model::<P>(dev)
}

/// Every logit of every pass of `work`, as bits, and the conv phase the
/// one-token open left.
fn run<P: Lfm2Shape>(
    model: &Model<P>,
    cfg: &Config,
    dev: &Device,
    work: &[(usize, usize)],
) -> Result<(Vec<u32>, usize)> {
    let mut cache = State::<P>::new(true, run_dtype(dev), cfg, dev)?;
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut id = 3u32;
    let mut opened = None;
    for &(tile, decode) in work {
        let ids: Vec<u32> = (0..tile as u32).map(|i| (id + 7 * i) % 60 + 2).collect();
        id += 11;
        push(
            &mut out,
            &model.forward(&Tensor::from_vec(ids, (1, tile), dev)?, pos, &mut cache)?,
        )?;
        pos += tile;
        opened.get_or_insert(cache.conv_phase());
        for _ in 0..decode {
            let ids = vec![id % 60 + 2];
            id += 5;
            push(
                &mut out,
                &model.forward(&Tensor::from_vec(ids, (1, 1), dev)?, pos, &mut cache)?,
            )?;
            pos += 1;
        }
    }
    Ok((out, opened.expect("a workload has a first pass")))
}

fn push(out: &mut Vec<u32>, logits: &Tensor) -> Result<()> {
    let v = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    out.extend(v.iter().map(|x| x.to_bits()));
    Ok(())
}

/// The conv phase a multi-token first pass leaves: where a prefill seeds the
/// newest column.
fn prefill_seed<P: Lfm2Shape>(model: &Model<P>, cfg: &Config, dev: &Device) -> Result<usize> {
    let mut cache = State::<P>::new(true, run_dtype(dev), cfg, dev)?;
    model.forward(
        &Tensor::from_vec(vec![5u32, 9], (1, 2), dev)?,
        0,
        &mut cache,
    )?;
    Ok(cache.conv_phase())
}

fn opens_like_the_shuffle(dev: &Device) -> Result<()> {
    let shuffle_cfg = config(ConvState::Shuffle);
    let shuffle = model::<CrownExact>(&shuffle_cfg, dev)?;
    for (w, work) in WORKLOADS.iter().enumerate() {
        let (reference, _) = run(&shuffle, &shuffle_cfg, dev, work)?;
        assert!(
            reference.iter().all(|b| f32::from_bits(*b).is_finite()),
            "{dev:?} workload {w}: the shuffle's reference has a non-finite logit"
        );
        for &(k, slack) in SLIDING {
            let cfg = config(ConvState::SlidingRing { k, slack });
            let m = model::<CrownExact>(&cfg, dev)?;
            let (got, opened) = run(&m, &cfg, dev, work).map_err(|e| {
                lloom_tensor::Error::Msg(format!(
                    "{dev:?} sliding:{k}:{slack} workload {w}: a one-token first pass failed: {e}"
                ))
            })?;
            // The values first: a window that is wrong must fail here, not
            // behind the slot check below.
            let first = got.iter().zip(&reference).position(|(a, b)| a != b);
            assert!(
                got.len() == reference.len() && first.is_none(),
                "{dev:?} sliding:{k}:{slack} workload {w}: diverged from the shuffle at logit \
                 {first:?} of {} (lengths {} / {})",
                reference.len(),
                got.len(),
                reference.len()
            );
            // A fresh state is zero in every slot, so any open at or past the
            // seed gives these logits; the slot is pinned separately, against
            // the one a prefill leaves.
            assert_eq!(
                opened,
                prefill_seed(&m, &cfg, dev)?,
                "{dev:?} sliding:{k}:{slack}: a one-token open leaves the window somewhere a \
                 prefill does not"
            );
        }
    }
    Ok(())
}

#[test]
fn a_one_token_open_on_the_sliding_ring_matches_the_shuffle_on_the_cpu() -> Result<()> {
    opens_like_the_shuffle(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn a_one_token_open_on_the_sliding_ring_matches_the_shuffle_on_metal() -> Result<()> {
    opens_like_the_shuffle(&Device::new_metal(0)?)
}
