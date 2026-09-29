//! What the ring's stamp dispatch binds, read off the dispatch trace.
//!
//! The command stream orders a dispatch after the last writers of the buffers
//! it binds and after nothing else, so "the stamp certifies the slot" is a
//! statement about the stamp's bindings. `ring_publish_order.rs` observes the
//! consequence as a race, which can only show the slots whose writers finish
//! late; on the poll arm the slot's copy and the flag's chain hang off the
//! same argmax and the flag's is the longer, so a stamp certifying the flag
//! alone reads right there by timing. This checks the binding itself.
//!
//! Each test reads its own device's trace window (`trace::DeviceTrace`,
//! `invariants.md` §15.1n), never the process-wide trace: cargo runs these
//! tests on parallel threads, and one test's `take_dispatches` or
//! `set_recording(false)` would empty or stop another's.
#![cfg(feature = "metal")]

use std::sync::Barrier;

use lloom::generation::{EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling};
use lloom_tensor::metal_backend::trace;
use lloom_tensor::test_utils::Rendezvous;
use lloom_tensor::{Device, Result, Storage, Tensor};

const EOS: u32 = 63;

fn install_knobs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        // Copies as compute kernels, so every write of the publish is a
        // dispatch the trace records.
        t.metal.copy2d_no_blit = true;
        t.instrument.trace = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    let t = lloom_tensor::tuning::tuning();
    assert!(
        t.metal.copy2d_no_blit && t.instrument.trace,
        "knobs installed too late"
    );
}

fn addr(t: &Tensor) -> usize {
    match &*t.storage_and_layout().0 {
        Storage::Metal(s) => s.buffer().raw_addr(),
        _ => panic!("not a Metal tensor"),
    }
}

/// The dispatches `f` encodes on `device`, read through the device's own
/// trace window.
fn traced(device: &Device, f: impl FnOnce() -> Result<()>) -> Result<Vec<trace::Dispatch>> {
    traced_meeting(device, None, f)
}

/// As [`traced`], meeting `meet` once after the window opens and once before
/// it closes, so a peer's window is open across all of `f`'s work.
fn traced_meeting(
    device: &Device,
    mut meet: Option<&mut Rendezvous<'_>>,
    f: impl FnOnce() -> Result<()>,
) -> Result<Vec<trace::Dispatch>> {
    let window = device
        .as_metal_device()?
        .dispatch_trace()
        .open()
        .expect("one trace window on this device");
    if let Some(m) = meet.as_mut() {
        m.meet();
    }
    let r = f();
    if let Some(m) = meet.as_mut() {
        m.meet();
    }
    r?;
    Ok(window.close())
}

fn bound(d: &trace::Dispatch, output: bool) -> Vec<usize> {
    d.bindings
        .iter()
        .filter(|b| b.is_output == output)
        .map(|b| b.buffer_addr)
        .collect()
}

/// The history segment the publish wrote: the one copy output that is not
/// the input slot, the EOS flag or the stamp.
fn history_segment(ds: &[trace::Dispatch], ring: &Ring) -> usize {
    let tr = ring.token_ring();
    let known = [
        addr(tr.input_device()),
        addr(tr.eos_flag()),
        addr(tr.step_counter()),
    ];
    let mut written: Vec<usize> = ds
        .iter()
        .filter(|d| d.pipeline.starts_with("copy2d"))
        .flat_map(|d| bound(d, true))
        .filter(|a| !known.contains(a))
        .collect();
    written.dedup();
    assert_eq!(
        written.len(),
        1,
        "one history write per publish, got {written:?}"
    );
    written[0]
}

fn stamps(ds: &[trace::Dispatch]) -> Vec<&trace::Dispatch> {
    ds.iter().filter(|d| d.pipeline == "ring_stamp").collect()
}

#[test]
fn every_stamp_binds_the_slots_and_the_flag_it_certifies() -> Result<()> {
    install_knobs();
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let mut row = vec![0f32; 64];
    row[5] = 1.0;
    let logits = Tensor::from_vec(row, 64, &device)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // The poll arm stamps inside the publish; the stamp certifies the slot
    // AND the EOS flag the poll reads after it.
    let mut ring = Ring::new(&[EOS], &device, 8, EosRead::Poll(Default::default()))?;
    let ds = traced(&device, || {
        ring.sample_and_feed(&mut sampler, &logits).map(|_| ())
    })?;
    let history = history_segment(&ds, &ring);
    let flag = addr(ring.token_ring().eos_flag());
    let step = addr(ring.token_ring().step_counter());
    let st = stamps(&ds);
    assert_eq!(st.len(), 1, "poll: one stamp per publish");
    let inputs = bound(st[0], false);
    eprintln!("poll: stamp inputs {inputs:x?}, history {history:x}, flag {flag:x}");
    assert!(
        inputs.contains(&history),
        "poll: the stamp does not bind the history segment"
    );
    assert!(
        inputs.contains(&flag),
        "poll: the stamp does not bind the EOS flag"
    );
    assert_eq!(
        bound(st[0], true),
        vec![step],
        "poll: the stamp writes step and nothing else"
    );
    device.synchronize()?;

    // A deferred arm stamps at the window; the stamp certifies the segment
    // the window's slots went to.
    let mut ring = Ring::new(&[EOS], &device, 8, EosRead::HostOnDrain)?;
    let ds = traced(&device, || {
        ring.sample_and_feed(&mut sampler, &logits)?;
        ring.sample_and_feed(&mut sampler, &logits)?;
        ring.publish_marker(true)
    })?;
    let history = history_segment(&ds, &ring);
    let step = addr(ring.token_ring().step_counter());
    let st = stamps(&ds);
    assert_eq!(
        st.len(),
        1,
        "host-drain: one stamp per window inside one segment"
    );
    let inputs = bound(st[0], false);
    eprintln!("host-drain: stamp inputs {inputs:x?}, history {history:x}");
    assert!(
        inputs.contains(&history),
        "host-drain: the stamp does not bind the history segment"
    );
    assert_eq!(
        bound(st[0], true),
        vec![step],
        "host-drain: the stamp writes step and nothing else"
    );
    device.synchronize()?;
    Ok(())
}

/// The fused publish's stamp is a `ring_stamp` of its own, dispatched after
/// the kernel and binding the history segment the kernel wrote. The kernel's
/// in-dispatch fence orders its slot and its stamp for the GPU and not for a
/// host reading the mapped words; a stamp in a later dispatch that binds the
/// segment is ordered after the slot by the command stream (§11.5z-xvi). A
/// race test sees the other shape only at a rate; this reads the shape.
#[test]
fn the_fused_publish_is_stamped_by_a_dispatch_that_binds_its_slot() -> Result<()> {
    install_knobs();
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let mut row = vec![0f32; 64];
    row[5] = 1.0;
    let logits = Tensor::from_vec(row, 64, &device)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut ring = Ring::new(&[EOS], &device, 8, EosRead::Poll(Default::default()))?;
    ring.set_fused_publish(true);
    let ds = traced(&device, || {
        ring.sample_and_feed(&mut sampler, &logits).map(|_| ())
    })?;
    assert!(
        ring.publishes_fused(),
        "the step did not take the fused publish"
    );
    let step = addr(ring.token_ring().step_counter());
    let input = addr(ring.token_ring().input_device());
    let fused: Vec<usize> = ds
        .iter()
        .enumerate()
        .filter(|(_, d)| d.pipeline.starts_with("argmax_publish"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(fused.len(), 1, "one fused dispatch per publish");
    let mut history: Vec<usize> = bound(&ds[fused[0]], true)
        .into_iter()
        .filter(|a| *a != step && *a != input)
        .collect();
    history.dedup();
    assert_eq!(
        history.len(),
        1,
        "the fused kernel writes one history segment: {history:x?}"
    );
    let st: Vec<usize> = ds
        .iter()
        .enumerate()
        .filter(|(_, d)| d.pipeline == "ring_stamp")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(st.len(), 1, "fused: one ring_stamp per publish, got {st:?}");
    assert!(
        st[0] > fused[0],
        "fused: the stamp is dispatched before the kernel"
    );
    let inputs = bound(&ds[st[0]], false);
    eprintln!("fused: stamp inputs {inputs:x?}, history {:x}", history[0]);
    assert!(
        inputs.contains(&history[0]),
        "fused: the stamp does not bind the history segment the kernel wrote"
    );
    assert_eq!(
        bound(&ds[st[0]], true),
        vec![step],
        "fused: the stamp writes step and nothing else"
    );
    device.synchronize()?;
    Ok(())
}

/// The speculative verify's publish (`Device::words_publish`, read by `lfm2`'s
/// `VerifyWords`) has the fused publish's shape: a kernel writes the words the
/// host reads in place, and the host polls a stamp to know they are there. So
/// its stamp is a `ring_stamp` of its own, dispatched after the copy and
/// binding `out`, whose only output is `step`; the copy's in-dispatch barrier
/// orders its stores for the GPU and not for the host (§11.5z-xvii).
#[test]
fn the_verify_words_are_stamped_by_a_dispatch_that_binds_them() -> Result<()> {
    install_knobs();
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let words: Vec<u32> = (1..=17).collect();
    let src = Tensor::from_vec(words.clone(), 17, &device)?;
    let out = Tensor::from_vec(vec![0u32; 64], 64, &device)?;
    let step = Tensor::from_vec(vec![0u32], 1, &device)?;
    let ds = traced(&device, || device.words_publish(&src, 17, &out, &step, 9))?;
    let (out_a, step_a) = (addr(&out), addr(&step));
    let at = |name: &str| -> Vec<usize> {
        ds.iter()
            .enumerate()
            .filter(|(_, d)| d.pipeline == name)
            .map(|(i, _)| i)
            .collect()
    };
    let copy = at("words_publish");
    assert_eq!(copy.len(), 1, "one words_publish per publish, got {copy:?}");
    assert!(
        bound(&ds[copy[0]], true).contains(&out_a),
        "the copy does not write `out`"
    );
    let st = at("ring_stamp");
    assert_eq!(
        st.len(),
        1,
        "verify words: one ring_stamp per publish, got {st:?}"
    );
    assert!(
        st[0] > copy[0],
        "verify words: the stamp is dispatched before the copy"
    );
    let inputs = bound(&ds[st[0]], false);
    eprintln!("verify words: stamp inputs {inputs:x?}, out {out_a:x}");
    assert!(
        inputs.contains(&out_a),
        "verify words: the stamp does not bind the words it certifies"
    );
    assert_eq!(
        bound(&ds[st[0]], true),
        vec![step_a],
        "verify words: the stamp writes step and nothing else"
    );
    device.synchronize()?;
    assert_eq!(step.to_vec1::<u32>()?, vec![9], "the stamp's value");
    assert_eq!(&out.to_vec1::<u32>()?[..17], words.as_slice(), "the words");
    Ok(())
}

/// The kernels a trace names, in order.
fn names(ds: &[trace::Dispatch]) -> Vec<&str> {
    ds.iter().map(|d| d.pipeline.as_str()).collect()
}

/// A ring publish on one device and a verify-words publish on another,
/// traced at once on two threads with both windows open across both
/// threads' work: each window holds its own publish and nothing of the
/// other's. The process-wide trace is the shape this file used to read, and
/// under it each test saw the other's dispatches or none of its own.
#[test]
fn two_publishes_traced_at_once_on_two_devices_see_only_their_own() -> Result<()> {
    install_knobs();
    if Device::new_metal(0).is_err() {
        return Ok(());
    }
    let both = Barrier::new(2);
    let (ring_ds, words_ds) = std::thread::scope(|s| {
        let ring = s.spawn(|| -> Result<Vec<trace::Dispatch>> {
            let mut meet = Rendezvous::new(&both, 2);
            let device = Device::new_metal(0)?;
            let mut row = vec![0f32; 64];
            row[5] = 1.0;
            let logits = Tensor::from_vec(row, 64, &device)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let mut ring = Ring::new(&[EOS], &device, 8, EosRead::Poll(Default::default()))?;
            device.synchronize()?;
            let ds = traced_meeting(&device, Some(&mut meet), || {
                ring.sample_and_feed(&mut sampler, &logits).map(|_| ())
            })?;
            device.synchronize()?;
            Ok(ds)
        });
        let words = s.spawn(|| -> Result<Vec<trace::Dispatch>> {
            let mut meet = Rendezvous::new(&both, 2);
            let device = Device::new_metal(0)?;
            let src = Tensor::from_vec((1..=17).collect::<Vec<u32>>(), 17, &device)?;
            let out = Tensor::from_vec(vec![0u32; 64], 64, &device)?;
            let step = Tensor::from_vec(vec![0u32], 1, &device)?;
            device.synchronize()?;
            let ds = traced_meeting(&device, Some(&mut meet), || {
                device.words_publish(&src, 17, &out, &step, 9)
            })?;
            device.synchronize()?;
            Ok(ds)
        });
        (ring.join().unwrap(), words.join().unwrap())
    });
    let (ring_ds, words_ds) = (ring_ds?, words_ds?);
    eprintln!(
        "ring device: {:?}\nwords device: {:?}",
        names(&ring_ds),
        names(&words_ds)
    );
    assert_eq!(
        names(&words_ds),
        vec!["words_publish", "ring_stamp"],
        "the words device's window holds its publish and nothing else"
    );
    assert!(
        !names(&ring_ds).contains(&"words_publish"),
        "the ring device's window holds the other device's words_publish"
    );
    assert_eq!(
        names(&ring_ds)
            .iter()
            .filter(|n| **n == "ring_stamp")
            .count(),
        1,
        "the ring device's window holds one stamp, its own: {:?}",
        names(&ring_ds)
    );
    Ok(())
}
