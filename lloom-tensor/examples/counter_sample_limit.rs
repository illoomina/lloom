//! How many timestamp counter sample buffers will this device hold at once, and
//! what does it cost when it refuses one?
//!
//! The profiler's counter arm (`CANDLE_METAL_PROFILE_COUNTERS`) keeps one
//! sample buffer per command buffer in flight. Metal documents no limit, and
//! the arm was once refused a buffer mid-run ("Cannot allocate sample
//! buffer"). This asks the device directly: for each sample count it creates
//! and HOLDS buffers until one is refused, and reports how many it held, how
//! many samples that is in total, and how long a created and a refused
//! allocation take. Then it releases them all and checks that one can be made
//! again (a limit on live buffers, not a lifetime count), and — with the
//! largest size's buffers still held — whether a small one is still refused
//! (a count limit) or accepted (a bytes limit).
//!
//! ```bash
//! ./target/release/examples/counter_sample_limit --sizes 2,64,4096,32768
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice, MTLStorageMode,
};
use std::time::Instant;

type Dev = ProtocolObject<dyn MTLDevice>;
type SampleBuf = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;
type Set = Retained<ProtocolObject<dyn MTLCounterSet>>;

#[derive(Parser, Debug)]
#[command(about = "the device's limit on live counter sample buffers")]
struct Args {
    /// Sample counts per buffer to probe, each from an empty start.
    #[arg(long, value_delimiter = ',', default_values_t = [2usize, 8, 64, 512, 4096, 32768])]
    sizes: Vec<usize>,
    /// Stop holding at this many buffers even if none was refused.
    #[arg(long, default_value_t = 1 << 16)]
    max_buffers: usize,
    /// Repetitions of the whole sweep, to show the limit is not a one-off.
    #[arg(long, default_value_t = 2)]
    reps: usize,
}

fn alloc(dev: &Dev, set: &Set, samples: usize) -> (Option<SampleBuf>, f64, Option<String>) {
    let d = MTLCounterSampleBufferDescriptor::new();
    d.setCounterSet(Some(set));
    d.setStorageMode(MTLStorageMode::Shared);
    // SAFETY: a plain count; the device validates it and fails the creation
    // rather than overrunning anything.
    unsafe { d.setSampleCount(samples) };
    let t = Instant::now();
    let r = dev.newCounterSampleBufferWithDescriptor_error(&d);
    let us = t.elapsed().as_secs_f64() * 1e6;
    match r {
        Ok(b) => (Some(b), us, None),
        Err(e) => (None, us, Some(e.localizedDescription().to_string())),
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    // SAFETY: a framework-provided static NSString.
    let ts = unsafe { MTLCommonCounterSetTimestamp };
    let set = dev
        .counterSets()
        .and_then(|sets| sets.iter().find(|s| s.name().isEqualToString(ts)))
        .context("no timestamp counter set")?;
    println!("device: {}", dev.name());

    for rep in 1..=args.reps {
        let mut largest_held: Vec<SampleBuf> = Vec::new();
        for &size in &args.sizes {
            let mut held: Vec<SampleBuf> = Vec::new();
            let mut ok_us = Vec::new();
            let mut refusal = None;
            while held.len() < args.max_buffers {
                let (b, us, err) = alloc(&dev, &set, size);
                match b {
                    Some(b) => {
                        ok_us.push(us);
                        held.push(b);
                    }
                    None => {
                        refusal = Some((us, err.unwrap_or_default()));
                        break;
                    }
                }
            }
            let n = held.len();
            let first_ok = ok_us.first().copied().unwrap_or(f64::NAN);
            let med_ok = median(&mut ok_us);
            match &refusal {
                Some((us, why)) => println!(
                    "rep {rep} size {size:>6}: held {n:>6} buffers = {:>9} samples ({:>9} B) \
                     before a refusal; created p50 {med_ok:.1} us (first {first_ok:.1}); \
                     refused in {us:.1} us: {why}",
                    n * size,
                    n * size * 8
                ),
                None => println!(
                    "rep {rep} size {size:>6}: held {n:>6} buffers = {:>9} samples, NO refusal \
                     up to --max-buffers; created p50 {med_ok:.1} us",
                    n * size
                ),
            }
            if refusal.is_some() {
                // A second refusal in a row, to time it without the first's
                // possible one-off cost.
                let (b, us, _) = alloc(&dev, &set, size);
                println!(
                    "    again at the limit: {} in {us:.1} us",
                    if b.is_some() { "CREATED" } else { "refused" }
                );
                let (b2, us2, _) = alloc(&dev, &set, 2);
                println!(
                    "    a 2-sample buffer at the limit: {} in {us2:.1} us",
                    if b2.is_some() { "CREATED" } else { "refused" }
                );
                drop((b, b2));
            }
            if size == *args.sizes.iter().max().unwrap_or(&size) {
                largest_held = held;
            } else {
                drop(held);
                let (b, us, why) = alloc(&dev, &set, size);
                if b.is_none() {
                    bail!("after releasing every buffer a {size}-sample one is still refused ({us:.1} us): {why:?}");
                }
            }
        }
        println!(
            "rep {rep}: released and re-created at every size: ok; still holding {} of the largest",
            largest_held.len()
        );
    }
    Ok(())
}
