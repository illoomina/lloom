//! The profiler's counter arm (`CANDLE_METAL_PROFILE_COUNTERS`, with one
//! encoder per dispatch): every dispatch runs in a counter-sampled encoder
//! whose interval lies inside its command buffer, carries its kernel's label
//! and its weight's bytes, and is not faster than the memory system can be.
//!
//! One test in its own file: the three switches are read once per process.
#![cfg(feature = "metal")]
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{DType, Device, Result, Tensor};

/// Past the 48 MB system-level cache, so every repetition streams from DRAM.
const N: usize = 8192;
const K: usize = 8192;
const REPS: usize = 8;

#[test]
fn every_dispatch_is_a_sampled_encoder_inside_its_command_buffer() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    std::env::set_var("CANDLE_METAL_PROFILE_COUNTERS", "1");
    std::env::set_var("CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH", "1");
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    assert!(
        profile::counters_enabled(),
        "the switch did not reach the profiler"
    );
    assert!(profile::encoder_per_dispatch());

    // A chain, as decode is: each GEMV reads the previous one's output, so the
    // encoders are ordered by a fence and cannot overlap. Independent GEMVs
    // may run concurrently, and then a sum of their intervals exceeds the
    // time they took. 2^-13 is exact in f16, so every link reads K * 2^-13 = 1.
    let w = (Tensor::ones((N, K), DType::F16, &dev)? * (1.0 / K as f64))?;
    let mut x = Tensor::ones((1, K), DType::F16, &dev)?;
    dev.synchronize()?;
    profile::reset();
    for _ in 0..REPS {
        x = x.matmul(&w.t()?)?;
    }
    dev.synchronize()?;
    let s = profile::snapshot();
    let y: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    assert!(
        y.iter().all(|&v| v == 1.0),
        "the GEMV chain must compute: K * 2^-13 = 1 at every link"
    );

    assert_eq!(
        s.counters_unavailable, None,
        "this device samples at stage boundaries (§13.1d)"
    );
    assert_eq!(s.encoder_samples_dropped, 0);
    assert_eq!(s.encoder_samples_bad, 0);
    assert_eq!(
        s.encoder_buffers_unresolved, 0,
        "every sampled command buffer resolved at completion"
    );
    assert_eq!(
        profile::encoder_containment(&s.encoder_records, 1e-6),
        0,
        "a sample outside its command buffer is a clock or unit error: {:?}",
        s.encoder_records.first()
    );
    let attributed: u64 = s
        .encoder_records
        .iter()
        .map(|r| u64::from(r.dispatches))
        .sum();
    assert_eq!(
        attributed, s.dispatches,
        "every dispatch belongs to exactly one sampled encoder"
    );
    assert!(
        s.encoder_records.iter().all(|r| r.dispatches == 1),
        "one encoder per dispatch: {:?}",
        s.encoder_records
            .iter()
            .map(|r| r.dispatches)
            .collect::<Vec<_>>()
    );

    let gemv: Vec<_> = s
        .encoder_records
        .iter()
        .filter(|r| r.label.as_deref().is_some_and(|l| l.starts_with("gemv")))
        .collect();
    assert_eq!(gemv.len(), REPS, "labels: {:?}", s.by_label);
    for r in &gemv {
        assert_eq!(
            r.input_bytes,
            (N * K * 2) as u64,
            "a GEMV's largest input is its weight, {N} x {K} f16"
        );
        let gbs = r.input_bytes as f64 / (r.end - r.start) / 1e9;
        eprintln!(
            "gemv encoder: {:.2} µs, {gbs:.1} GB/s, {:.2} µs after its buffer's start",
            (r.end - r.start) * 1e6,
            (r.start - r.cb_start) * 1e6
        );
        // 400 GB/s is the part's spec peak (§3.4). An interval that implies more
        // than that is not the kernel's execution — the residency artifact
        // §13.5r measured at ~790 GB/s under one buffer per dispatch.
        assert!(
            (20.0..=400.0).contains(&gbs),
            "{gbs:.1} GB/s from a {:.1} µs interval is not a DRAM sweep",
            (r.end - r.start) * 1e6
        );
    }
    let sum: f64 = s.encoder_records.iter().map(|r| r.end - r.start).sum();
    assert!(
        sum <= s.gpu_busy_union_s + 1e-6,
        "encoders ({sum:.6} s) cannot cover more than their command buffers ({:.6} s)",
        s.gpu_busy_union_s
    );
    Ok(())
}
