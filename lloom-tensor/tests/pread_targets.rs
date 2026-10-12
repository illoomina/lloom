//! Where a `pread` lands matters: the same 512 MB from the model shard into
//! a `Vec`, a pooled Shared buffer, a persistent Shared buffer, and a
//! persistent Private buffer's mapping (issue #749 P27: the direct read
//! into the slot pools ran at half the ring's throughput).
#![cfg(feature = "metal")]
use std::os::unix::fs::FileExt;

#[test]
#[ignore]
fn pread_into_each_target() {
    let Ok(path) = std::env::var("LLOOM_PREAD_FILE") else {
        return;
    };
    let f = std::fs::File::open(&path).unwrap();
    let dev = lloom_tensor::Device::new_metal(0).unwrap();
    let lloom_tensor::Device::Metal(m) = &dev else {
        unreachable!()
    };
    let total: usize = std::env::var("LLOOM_PREAD_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        << 20;
    let chunk = 1usize << 20; // 1 MiB reads, like an expert plane (~0.6 MB)
    let base: u64 = 4 << 30;
    let time = |name: &str, dst: *mut u8| {
        // Touch once so page faults are not what is timed, then time.
        for pass in 0..2 {
            let t0 = std::time::Instant::now();
            let threads = 8;
            std::thread::scope(|s| {
                for t in 0..threads {
                    let f = &f;
                    let p = dst as usize;
                    s.spawn(move || {
                        let per = total / threads;
                        let mut off = t * per;
                        while off < (t + 1) * per {
                            let slice = unsafe {
                                std::slice::from_raw_parts_mut((p + off) as *mut u8, chunk)
                            };
                            f.read_exact_at(slice, base + off as u64).unwrap();
                            off += chunk;
                        }
                    });
                }
            });
            let dt = t0.elapsed().as_secs_f64();
            if pass == 1 {
                println!("{name}: {:.2} GB/s", total as f64 / dt / 1e9);
            }
        }
    };
    let mut v = vec![0u8; total];
    time("vec", v.as_mut_ptr());
    let pooled = m
        .new_buffer_builder()
        .with_size(total)
        .with_label("pooled")
        .build()
        .unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    time("pooled shared", unsafe { pooled.contents() });
    let shared = m.new_persistent_shared(total).unwrap();
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    time("persistent shared", unsafe { shared.contents() });
    let private = m.new_persistent_private(total).unwrap();
    // A Private buffer's `contents()` is invalid on the CPU: the ring arm
    // never writes one. Timed here only through the pooled Shared one.
    let _ = private;
}
