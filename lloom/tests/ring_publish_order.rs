//! The ring's unfused publish: a host that observes the stamp must read the
//! history slot behind it, when `copy2d` runs as a compute kernel.
//!
//! Its own binary because the routing is a process-wide knob read once
//! (`tuning().metal.copy2d_no_blit`, the routing `--pass-plans` forces through
//! `MetalDevice::base_mode`), installed before any tensor exists.
//!
//! The shape that exposes the defect is the one the decode loop makes under a
//! lookahead: the slot write sits behind a long dependency chain (the forward's
//! logits), a command-buffer cut separates it from the stamp, and the stamp's
//! own dispatch depends on nothing. A slow matmul chain stands in for the
//! forward. The history is zero-filled and every expected token is non-zero and
//! different from the one before, so a stamp that ran ahead of its slot reads
//! as a zero or as the previous trial's token.
#![cfg(feature = "metal")]

use lloom::generation::{EosRead, Geometry, LogitsProcessor, ReadbackStrategy, Ring, Sampling};
use lloom_tensor::{DType, Device, Result, Tensor};

const EOS: u32 = 4095;
const DIM: usize = 1024;
const CHAIN: usize = 12;
const TRIALS: usize = 64;
/// Slots in one history segment (`readback.rs`'s `SEG_MAX_SLOTS`).
const SEGMENT: usize = 4092;

/// The process-wide knobs, installed once before any tensor exists: copies as
/// compute kernels, and a history of exactly two segments for a ring built
/// from its geometry (the consumer slack, in bytes).
fn install_knobs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.metal.copy2d_no_blit = true;
        let _ = lloom_tensor::tuning::install(t);
        let mut t = lloom::tuning::Tuning::default();
        t.ring.slack_bytes = 2 * SEGMENT * std::mem::size_of::<u32>();
        let _ = lloom::tuning::install(t);
    });
    assert!(
        lloom_tensor::tuning::tuning().metal.copy2d_no_blit,
        "the compute routing must be in force before any tensor exists"
    );
}

/// A Metal device, optionally cutting a command buffer at every dispatch.
fn metal(cut_every_dispatch: bool) -> Option<Device> {
    install_knobs();
    let device = Device::new_metal(0).ok()?;
    if cut_every_dispatch {
        if let Device::Metal(m) = &device {
            m.set_compute_per_buffer(1);
        }
    }
    Some(device)
}

/// A one-hot row at `token` pushed through `chain` identity matmuls: the
/// argmax is `token`, and the GPU pays `chain` full `DIM³` products for it.
fn logits(token: usize, chain: usize, eye: &Tensor, device: &Device) -> Result<Tensor> {
    let mut row = vec![0f32; DIM];
    row[token] = 1.0;
    let mut x = Tensor::from_vec(row, (1, DIM), device)?;
    if chain == 0 {
        return x.flatten_all();
    }
    x = x.broadcast_as((DIM, DIM))?.contiguous()?;
    for _ in 0..chain {
        x = x.matmul(eye)?;
    }
    x.narrow(0, 0, 1)?.flatten_all()
}

fn commit(device: &Device) -> Result<()> {
    if let Device::Metal(m) = device {
        m.flush_and_prune()?;
    }
    Ok(())
}

/// Spin, without waiting on the GPU, until the ring's stamp reaches `want`.
fn await_stamp(ring: &Ring, want: u32) -> Result<()> {
    let mut spins = 0u64;
    while ring
        .token_ring()
        .published_generation()
        .is_none_or(|g| g < want)
    {
        spins += 1;
        if spins > 2_000_000_000 {
            lloom_tensor::bail!("stamp {want} never published");
        }
        std::hint::spin_loop();
    }
    Ok(())
}

/// How a trial is shaped around the publish.
#[derive(Clone, Copy)]
struct Shape {
    /// Cut a command buffer at every dispatch, so a stamp encoded inside one
    /// publish (the poll arm's) lands in a later buffer than the slot.
    cut_every_dispatch: bool,
    /// Wait for the GPU before reading: the control. The same reads, with no
    /// ordering left to the stamp, must be right on every build.
    sync_before_read: bool,
}

const PLAIN: Shape = Shape {
    cut_every_dispatch: false,
    sync_before_read: false,
};

const CUT: Shape = Shape {
    cut_every_dispatch: true,
    sync_before_read: false,
};

/// Each wrong window as `(trial, expected, read)`.
type Wrong = Vec<(usize, u32, u32)>;

/// One publish per trial: the slot write behind the chain, a cut, the stamp,
/// a cut; then the host polls the stamp without waiting on the GPU and reads
/// the window it certifies. Returns `(trials, wrong)`.
fn publish_then_read(eos: EosRead, shape: Shape) -> Result<(usize, Wrong)> {
    let Some(device) = metal(shape.cut_every_dispatch) else {
        return Ok((0, Vec::new()));
    };
    let eye = Tensor::eye(DIM, DType::F32, &device)?;
    let mut ring = Ring::new(&[EOS], &device, 8, eos)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut wrong = Vec::new();
    for trial in 0..TRIALS {
        let token = 1 + (trial * 37) % (DIM - 2);
        let logits = logits(token, CHAIN, &eye, &device)?;
        ring.sample_and_feed(&mut sampler, &logits)?;
        commit(&device)?;
        ring.publish_marker(true)?;
        commit(&device)?;
        await_stamp(&ring, (trial + 1) as u32)?;
        if shape.sync_before_read {
            device.synchronize()?;
        }
        let got = ring.take_ids()?;
        if got.as_slice() != [token as u32] {
            wrong.push((
                trial,
                token as u32,
                got.first().copied().unwrap_or(u32::MAX),
            ));
        }
        if matches!(eos, EosRead::Poll(_)) {
            // The poll arm's flag is certified by the same stamp.
            assert!(
                !ring.eos_of_last()?,
                "trial {trial}: a non-EOS token read as EOS"
            );
        }
    }
    device.synchronize()?;
    Ok((TRIALS, wrong))
}

fn assert_none_wrong(arm: &str, (n, wrong): (usize, Wrong)) {
    eprintln!(
        "{arm}: {} of {n} windows read a slot the stamp had not ordered: {wrong:?}",
        wrong.len()
    );
    assert!(
        wrong.is_empty(),
        "{arm}: {} of {n} wrong: {wrong:?}",
        wrong.len()
    );
}

/// Both cut shapes: the loop's (a commit between the slot and the window's
/// stamp) and a cut at every dispatch, which exposes the race far more often.
#[test]
fn a_deferred_arms_stamp_is_ordered_after_the_slot_it_certifies() -> Result<()> {
    assert_none_wrong(
        "host-drain",
        publish_then_read(EosRead::HostOnDrain, PLAIN)?,
    );
    assert_none_wrong(
        "host-drain, cut",
        publish_then_read(EosRead::HostOnDrain, CUT)?,
    );
    Ok(())
}

/// The poll arm stamps inside the publish, a few dispatches after the slot;
/// the EOS test's own barriers order the slot within one encoder, so only a
/// cut between them exposes it. A cut at every dispatch puts one there.
#[test]
fn the_poll_arms_stamp_is_ordered_after_the_slot_and_the_flag() -> Result<()> {
    assert_none_wrong(
        "poll",
        publish_then_read(EosRead::Poll(Default::default()), CUT)?,
    );
    Ok(())
}

/// The control: the harness reads the right slot when nothing is left to the
/// stamp's order. A red here means the harness is wrong, not the stamp.
#[test]
fn the_same_reads_after_a_wait_are_right_on_every_build() -> Result<()> {
    let shape = Shape {
        sync_before_read: true,
        ..CUT
    };
    assert_none_wrong(
        "host-drain, synced",
        publish_then_read(EosRead::HostOnDrain, shape)?,
    );
    Ok(())
}

/// A window that leaves one history segment for the next is certified per
/// segment. The window's first slot is the last of a segment and sits behind
/// the slow chain; its second is the next segment's first and is written at
/// once. A stamp that certified only the segment the window ended in would
/// be ordered after the fast slot and not the slow one.
#[test]
fn a_window_across_a_segment_boundary_is_stamped_per_segment() -> Result<()> {
    const CROSSINGS: usize = 4;
    let Some(device) = metal(true) else {
        return Ok(());
    };
    let geometry = Geometry {
        batch: 1,
        period: 16,
        ..Default::default()
    };
    let mut ring = Ring::with_geometry(&[EOS], &device, geometry, EosRead::HostOnDrain)?;
    let eye = Tensor::eye(DIM, DType::F32, &device)?;
    let fast = logits(7, 0, &eye, &device)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut fed = 0usize;
    let mut wrong = Vec::new();
    for crossing in 0..CROSSINGS {
        // Fast publishes up to the segment's last slot, drained as they go.
        let boundary = (crossing + 1) * SEGMENT;
        while fed < boundary - 1 {
            ring.sample_and_feed(&mut sampler, &fast)?;
            fed += 1;
            if fed.is_multiple_of(64) || fed == boundary - 1 {
                ring.publish_marker(true)?;
                device.synchronize()?;
                ring.take_ids()?;
            }
        }
        let slow = 1 + (crossing * 37) % (DIM - 2);
        let slow = if slow == 7 { 8 } else { slow };
        ring.sample_and_feed(&mut sampler, &logits(slow, CHAIN, &eye, &device)?)?;
        ring.sample_and_feed(&mut sampler, &fast)?;
        fed += 2;
        commit(&device)?;
        ring.publish_marker(true)?;
        commit(&device)?;
        await_stamp(&ring, fed as u32)?;
        let got = ring.take_ids()?;
        if got.as_slice() != [slow as u32, 7] {
            wrong.push((
                crossing,
                slow as u32,
                got.first().copied().unwrap_or(u32::MAX),
            ));
        }
    }
    device.synchronize()?;
    assert_none_wrong("segment crossing", (CROSSINGS, wrong));
    Ok(())
}
