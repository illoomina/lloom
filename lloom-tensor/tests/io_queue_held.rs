#![cfg(feature = "metal")]
//! An IO load ordered behind work held for a host-produced input
//! (`docs/invariants.md` §15.1q): the IO command buffer waits on a compute
//! signal the held recording carries, so it is committed only once that
//! recording is — and the compute behind the load is committed by the load's
//! own notification, not by a GPU wait on `ev_io`.

use lloom_tensor::kernels::metal::io_queue::{IoLoad, IoQueue};
use lloom_tensor::kernels::metal::HostInputArm;
use lloom_tensor::metal_backend::HostEvent;
use lloom_tensor::{Device, Result};
use std::time::Duration;

const LEN: usize = 64 * 1024;

fn file_bytes() -> Vec<u8> {
    (0..LEN).map(|i| (i * 31 % 251) as u8).collect()
}

fn temp_file(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "lloom-io-held-{tag}-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, file_bytes()).unwrap();
    path
}

/// Held load: the IO buffer is not committed while the recording carrying
/// its compute signal is held; once the gate's input is produced the load
/// lands and the compute behind it reads the file's bytes.
#[test]
fn a_load_behind_a_held_signal_lands_after_its_release() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let commit_arm = m.commands().host_input_arm() == HostInputArm::Commit;
    let path = temp_file("release");
    let queue = IoQueue::new(m.device()).map_err(lloom_tensor::Error::wrap)?;
    let file = queue.open(&path).map_err(lloom_tensor::Error::wrap)?;
    let dst = m.new_persistent_shared(LEN)?;
    let gate = std::sync::Arc::new(HostEvent::new(m)?);
    let before = queue.signaled();
    gate.commit_when_produced(1)?;
    // Work on the stream, held behind the gate, ahead of the load's signal.
    let _held = lloom_tensor::Tensor::full(1.0f32, 256, &dev)?.affine(2.0, 0.0)?;
    let value = queue
        .issue(
            m.commands(),
            &[IoLoad {
                dst: &dst,
                dst_offset: 0,
                file: &file,
                src_offset: 0,
                len: LEN,
            }],
        )
        .map_err(lloom_tensor::Error::wrap)?;
    if commit_arm {
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            queue.signaled(),
            before,
            "the load ran before the recording carrying its compute signal was committed"
        );
    }
    let producer = {
        let gate = std::sync::Arc::clone(&gate);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            gate.signal(1);
        })
    };
    queue
        .await_value(m.commands(), value)
        .map_err(lloom_tensor::Error::wrap)?;
    // A reader of the loaded bytes, committed by the load's notification.
    m.flush_and_prune()?;
    producer.join().expect("producer");
    queue.wait_landed(value);
    dev.synchronize()?;
    // SAFETY: a CPU-mapped Shared buffer of LEN bytes; its only writer (the
    // load) has landed and nothing else writes it.
    let got = unsafe { std::slice::from_raw_parts(dst.contents() as *const u8, LEN) };
    assert_eq!(got, file_bytes().as_slice(), "the load's bytes");
    if commit_arm {
        let stats = m.commands().produced_stats();
        assert_eq!(stats.held_now, 0, "{stats:?}");
        assert!(stats.released >= 1, "{stats:?}");
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
