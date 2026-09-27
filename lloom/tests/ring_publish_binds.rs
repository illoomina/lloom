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
//! Its own binary: the trace is process-wide and enabled once.
#![cfg(feature = "metal")]

use lloom::generation::{EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling};
use lloom_tensor::metal_backend::trace;
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

fn traced(f: impl FnOnce() -> Result<()>) -> Result<Vec<trace::Dispatch>> {
    let _ = trace::take_dispatches();
    trace::set_recording(true);
    let r = f();
    trace::set_recording(false);
    r?;
    Ok(trace::take_dispatches())
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
    let ds = traced(|| ring.sample_and_feed(&mut sampler, &logits).map(|_| ()))?;
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
    let ds = traced(|| {
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
    let ds = traced(|| ring.sample_and_feed(&mut sampler, &logits).map(|_| ()))?;
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
    let ds = traced(|| device.words_publish(&src, 17, &out, &step, 9))?;
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
