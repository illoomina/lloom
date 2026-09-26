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
