#![cfg(feature = "metal")]
//! An IO load that does not land its bytes, and the compute held behind it
//! (`IoQueue::await_value`, `docs/invariants.md` §15.1q).
//!
//! The work behind a load reads the bytes it landed. A load that landed
//! none must not release that work: it is refused before it is issued, or
//! the work is cancelled through the load's value, the way a host producer
//! cancels a value it did not write, and the request sees the error. Not a
//! hang (a held recording nothing releases), and not a run over the bytes
//! the slot held before. What Metal does with such a load, and with the
//! signal after it, is what the probe below records: a load past the file's
//! end completes, lands only the bytes that exist, and signals.

use lloom_tensor::kernels::metal::io_queue::{IoLoad, IoQueue};
use lloom_tensor::kernels::metal::HostInputArm;
use lloom_tensor::{Device, Result, Tensor};
use std::time::{Duration, Instant};

const LEN: usize = 16 * 1024;

fn temp_file(tag: &str) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("lloom-io-failed-{tag}-{}.bin", std::process::id()));
    std::fs::write(&path, vec![7u8; LEN]).unwrap();
    path
}

/// A load whose source range starts at the file's end never releases the
/// work behind it: the IO queue refuses it at `issue` (the range against
/// the file's length at `open`), or cancels the work behind its value.
#[test]
fn a_failed_io_load_cancels_the_work_held_behind_it() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    if m.commands().host_input_arm() != HostInputArm::Commit {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return Ok(());
    }
    let path = temp_file("past-end");
    let queue = IoQueue::new(m.device()).map_err(lloom_tensor::Error::wrap)?;
    let file = queue.open(&path).map_err(lloom_tensor::Error::wrap)?;
    let dst = m.new_persistent_shared(LEN)?;
    let before = queue.signaled();
    // The source range starts at the file's end: no byte of it exists.
    let issued = queue.issue(
        m.commands(),
        &[IoLoad {
            dst: &dst,
            dst_offset: 0,
            file: &file,
            src_offset: LEN as u64,
            len: LEN,
        }],
    );
    let value = match issued {
        Ok(value) => value,
        Err(e) => {
            eprintln!("IO_FAILED_LOAD refused at issue: {e}");
            std::fs::remove_file(&path).ok();
            assert_eq!(m.commands().produced_stats().held_now, 0);
            return Ok(());
        }
    };
    queue
        .await_value(m.commands(), value)
        .map_err(lloom_tensor::Error::wrap)?;
    // The reader of the load's bytes: held behind the load's value.
    let read = Tensor::full(1.0f32, 256, &dev)?.affine(2.0, 0.0)?;
    m.flush_and_prune()?;
    // Read it back on another thread: a held recording nothing releases
    // would park this one for good.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(read.to_vec1::<f32>().map(|v| v[0]));
    });
    let t0 = Instant::now();
    let outcome = rx.recv_timeout(Duration::from_secs(10));
    let waited = t0.elapsed();
    let reached = queue.signaled();
    let stats = m.commands().produced_stats();
    eprintln!(
        "IO_FAILED_LOAD api={} value={value} before={before} signaled={reached} waited_ms={} outcome={outcome:?} stats={stats:?}",
        m.commands().actual_command_api(),
        waited.as_millis(),
    );
    std::fs::remove_file(&path).ok();
    let outcome = outcome.map_err(|_| {
        lloom_tensor::Error::Msg(format!(
            "the work behind a failed load was neither run nor cancelled in 10 s (signaled {reached} of {value}): a hang"
        ))
    })?;
    match outcome {
        Ok(v) => {
            panic!("the work behind a failed load ran (read {v}) over bytes the load never landed")
        }
        Err(e) => {
            let why = e.to_string();
            assert!(
                why.contains("load"),
                "the request's error does not name the failed load: {why}"
            );
        }
    }
    assert_eq!(stats.held_now, 0, "{stats:?}");
    Ok(())
}

/// A probe of Metal itself, not of lloom: what a failing load does to its
/// IO command buffer — its status, its error, and whether the `signalEvent`
/// encoded after it still runs. Beside a load inside the file (the control,
/// which must complete and signal), the arms are a load whose range starts at
/// the file's end, one that straddles it, one through a compressed handle
/// opened over a file that is not compressed, and a buffer cancelled
/// (`tryCancel`) while it waits on an event. The host-pointer load
/// (`loadBytes:`) keeps lloom's buffers out of it. Only the control is
/// asserted; every arm prints a `PROBE_IO` line.
#[test]
fn probe_a_failing_load_and_the_signal_after_it() -> Result<()> {
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{
        MTLDevice, MTLIOCommandBuffer, MTLIOCommandQueue, MTLIOCommandQueueDescriptor,
        MTLIOCompressionMethod, MTLIOStatus, MTLSharedEvent,
    };
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let native = m.device().as_ref();
    let queue = native
        .newIOCommandQueueWithDescriptor_error(&MTLIOCommandQueueDescriptor::new())
        .map_err(|e| lloom_tensor::Error::Msg(format!("{e:?}")))?;
    let path = temp_file("probe");
    let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str().unwrap()));
    let raw = native
        .newIOFileHandleWithURL_error(&url)
        .map_err(|e| lloom_tensor::Error::Msg(format!("{e:?}")))?;
    let compressed =
        native.newIOFileHandleWithURL_compressionMethod_error(&url, MTLIOCompressionMethod::LZ4);
    if let Err(e) = &compressed {
        eprintln!(
            "PROBE_IO compressed-over-raw open refused: {}",
            e.localizedDescription()
        );
    }
    let event = native.newSharedEvent().expect("newSharedEvent");
    let gate = native.newSharedEvent().expect("newSharedEvent");
    let arms: [(&str, usize, bool, bool); 5] = [
        ("inside", 0, false, false),
        ("past-end", LEN, false, false),
        ("straddling", LEN / 2, false, false),
        ("compressed-over-raw", 0, true, false),
        ("cancelled-while-waiting", 0, false, true),
    ];
    let mut control = None;
    for (n, (label, offset, use_compressed, cancel)) in arms.into_iter().enumerate() {
        let handle = match (use_compressed, &compressed) {
            (false, _) => &raw,
            (true, Ok(h)) => h,
            (true, Err(_)) => continue,
        };
        let value = n as u64 + 1;
        let mut host = vec![0x5Au8; LEN];
        let cb = queue.commandBuffer();
        if cancel {
            cb.waitForEvent_value(&gate, value);
        }
        // SAFETY: `host` is LEN writable bytes that outlive the command
        // buffer: it is read and dropped only after the buffer has left
        // `Pending` (completed, failed or cancelled), waited for below.
        unsafe {
            cb.loadBytes_size_sourceHandle_sourceHandleOffset(
                std::ptr::NonNull::new(host.as_mut_ptr().cast()).unwrap(),
                LEN,
                handle,
                offset,
            );
        }
        cb.signalEvent_value(&event, value);
        cb.commit();
        let mut note = String::new();
        if cancel {
            cb.tryCancel();
            let t0 = std::time::Instant::now();
            while cb.status() == MTLIOStatus::Pending && t0.elapsed().as_secs() < 2 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            if cb.status() == MTLIOStatus::Pending {
                note = " (tryCancel left it pending 2 s; the gate is raised)".into();
            }
            gate.setSignaledValue(value);
        }
        cb.waitUntilCompleted();
        let status = cb.status();
        let error = cb.error().map(|e| e.localizedDescription().to_string());
        let reached = event.signaledValue();
        let written = host.iter().filter(|&&b| b != 0x5A).count();
        eprintln!(
            "PROBE_IO {label} offset={offset} status={status:?} error={error:?} \
             signal={value} event={reached} signalled={} bytes_changed={written} of {LEN}{note}",
            reached >= value
        );
        if n == 0 {
            control = Some((status, reached >= value, written));
        }
    }
    std::fs::remove_file(&path).ok();
    assert_eq!(
        control,
        Some((MTLIOStatus::Complete, true, LEN)),
        "the control load inside the file did not complete, signal and land every byte"
    );
    Ok(())
}
