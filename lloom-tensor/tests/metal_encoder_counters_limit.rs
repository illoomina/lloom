//! The counter arm under the device's limit on live sample buffers (32,
//! whatever their size, `DESIGN.md` §13.1n): more command buffers in flight
//! than that are all sampled, from shared buffers; and when the device does
//! refuse a buffer, the encoders that wanted one are counted as dropped while
//! the counters stay on.
//!
//! One test in its own file: the switches are read once per process.
#![cfg(feature = "metal")]
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{DType, Device, Result, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice, MTLStorageMode,
};

/// 134 MB of f16 weights: each GEMV takes ~0.37 ms on the GPU, far longer than
/// encoding and committing one, so the command buffers pile up in flight.
const N: usize = 8192;
const K: usize = 8192;
/// More than the device's 32 live sample buffers, and more than the queue's
/// in-flight limit, so the peak is set by the queue rather than by this loop.
const REPS: usize = 128;
const HOGGED_REPS: usize = 6;
/// Tiny dispatches over time, far more than two sample buffers hold.
const TINY_REPS: usize = 10_000;

type SampleBuf = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

/// Every counter sample buffer the device will give, held until dropped.
fn hog_sample_buffers() -> Vec<SampleBuf> {
    let dev = MTLCreateSystemDefaultDevice().expect("no Metal device");
    // SAFETY: a framework-provided static NSString.
    let ts = unsafe { MTLCommonCounterSetTimestamp };
    let set = dev
        .counterSets()
        .and_then(|sets| sets.iter().find(|s| s.name().isEqualToString(ts)))
        .expect("no timestamp counter set");
    let mut held = Vec::new();
    while held.len() < 1024 {
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(&set));
        d.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a plain count in the device's range.
        unsafe { d.setSampleCount(2) };
        match dev.newCounterSampleBufferWithDescriptor_error(&d) {
            Ok(b) => held.push(b),
            Err(_) => break,
        }
    }
    held
}

fn chain(w: &Tensor, dev: &Device, reps: usize) -> Result<Tensor> {
    let mut x = Tensor::ones((1, K), DType::F16, dev)?;
    for _ in 0..reps {
        x = x.matmul(&w.t()?)?;
    }
    dev.synchronize()?;
    Ok(x)
}

#[test]
fn more_command_buffers_in_flight_than_the_device_has_sample_buffers() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    std::env::set_var("CANDLE_METAL_PROFILE_COUNTERS", "1");
    std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1");
    // Taken before the profiler samples anything, so it holds no buffer: every
    // encoder is dropped, each refusal is counted, and the counters are not
    // reported off.
    let hog = hog_sample_buffers();
    assert!(
        (1..1024).contains(&hog.len()),
        "the device has a finite limit on live sample buffers: {}",
        hog.len()
    );
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    assert!(profile::counters_enabled());

    let w = (Tensor::ones((N, K), DType::F16, &dev)? * (1.0 / K as f64))?;
    dev.synchronize()?;
    profile::reset();
    chain(&w, &dev, HOGGED_REPS)?;
    let s = profile::snapshot();
    eprintln!(
        "hogged {} buffers: dispatches {}, dropped {}, refused {} ({:?}), sampled {}",
        hog.len(),
        s.dispatches,
        s.encoder_samples_dropped,
        s.counter_buffers_refused,
        s.counter_refusal,
        s.encoder_records.len()
    );
    assert_eq!(
        s.counters_unavailable, None,
        "a refused buffer is a dropped sample, not a device that cannot sample"
    );
    assert!(
        s.counter_buffers_refused >= 1,
        "the refusal was not counted"
    );
    assert!(s.encoder_records.is_empty());
    assert_eq!(
        s.encoder_samples_dropped, s.dispatches,
        "one dispatch per command buffer: every encoder is one dropped sample"
    );
    drop(hog);

    // Released: more command buffers in flight than the device has buffers,
    // every encoder sampled, from one shared buffer.
    profile::reset();
    let x = chain(&w, &dev, REPS)?;
    let s = profile::snapshot();
    let y: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    assert!(y.iter().all(|&v| v == 1.0), "the GEMV chain must compute");
    eprintln!(
        "released: dispatches {}, sampled {}, dropped {}, refused {}, peak in flight {}, buffers {}, slowest allocation {:.1} us",
        s.dispatches,
        s.encoder_records.len(),
        s.encoder_samples_dropped,
        s.counter_buffers_refused,
        s.encoder_buffers_in_flight_peak,
        s.counter_buffers_live,
        1e6 * s.counter_alloc_max_s
    );
    assert_eq!(
        (s.counter_buffers_refused, s.encoder_samples_dropped),
        (0, 0),
        "a sample buffer was refused: {:?}",
        s.counter_refusal
    );
    // Only sampled command buffers are counted in flight, so this is also what
    // a buffer per command buffer could never exceed.
    assert!(
        s.encoder_buffers_in_flight_peak > 32,
        "the test must put more command buffers in flight than the device has \
         sample buffers, or it does not exercise the limit: peak {}",
        s.encoder_buffers_in_flight_peak
    );
    assert_eq!(s.encoder_samples_bad, 0);
    assert_eq!(s.encoder_buffers_unresolved, 0);
    assert_eq!(
        s.counter_buffers_live, 1,
        "{} encoders in flight fit one 4096-sample buffer",
        s.encoder_buffers_in_flight_peak
    );
    assert_eq!(s.encoder_records.len() as u64, s.dispatches);
    assert!(s.encoder_records.iter().all(|r| r.dispatches == 1));
    assert_eq!(
        profile::encoder_containment(&s.encoder_records, 1e-6),
        0,
        "a sample outside its command buffer: a shared slot read before it was written, or a clock error"
    );

    // Many more encoders over time than two buffers hold: a buffer whose
    // encoders have all resolved is reused from slot 0, so the count stays at
    // what is in flight rather than growing with the run.
    profile::reset();
    let mut z = Tensor::zeros(1, DType::F32, &dev)?;
    for i in 0..TINY_REPS {
        z = (z + 1.0)?;
        if i % 256 == 255 {
            dev.synchronize()?;
        }
    }
    dev.synchronize()?;
    let s = profile::snapshot();
    assert_eq!(z.to_vec1::<f32>()?, [TINY_REPS as f32]);
    eprintln!(
        "over time: dispatches {}, sampled {}, dropped {}, buffers {}",
        s.dispatches,
        s.encoder_records.len(),
        s.encoder_samples_dropped,
        s.counter_buffers_live
    );
    assert!(
        s.dispatches > 2 * 2048,
        "the run must outgrow two 4096-sample buffers to exercise reuse"
    );
    assert_eq!(s.encoder_samples_dropped, 0);
    assert_eq!(s.encoder_samples_bad, 0);
    assert_eq!(s.encoder_records.len() as u64, s.dispatches);
    assert!(
        s.counter_buffers_live <= 2,
        "a drained sample buffer was not reused: {} buffers for {} encoders",
        s.counter_buffers_live,
        s.dispatches
    );
    assert_eq!(profile::encoder_containment(&s.encoder_records, 1e-6), 0);
    Ok(())
}
