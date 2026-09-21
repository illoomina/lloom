#![cfg(feature = "metal")]
//! The Metal reduce and softmax kernels, hammered under concurrent GPU load,
//! must return the same bits on every repetition (issue #748).
//!
//! # Why this exists, and what it is not
//!
//! `reduce.metal`'s `block_reducer` (the threadgroup-memory tree every reduce
//! with ≥ 64 threads per threadgroup runs) and the softmax's `md_total`
//! broadcast synchronised with `threadgroup_barrier(mem_flags::mem_none)`:
//! an execution barrier with **no memory fence**, so a lane could read a
//! partner's slot before that partner's store had landed -- undefined per the
//! MSL spec, and on hardware a timing-dependent stale read. The barriers now
//! carry `mem_threadgroup`.
//!
//! **This test did not reproduce the defect.** With `mem_none` in place it ran
//! 20 000 repetitions of the strided `[48,128,128].sum(1)` and of the
//! contiguous `[4096,2048].sum(1)` beside a second queue of 2048² matmuls, and
//! every repetition matched bit-for-bit; the model-level gate (the Qwen4-exp
//! 3749-token needle prompt, `measurements/issue-748-raw/`) is what saw the
//! defect and what the fix is judged by. This test is the repeatability
//! hammer that stays: every repetition compared (the earlier
//! `metal_strided_reduce_repeat_tests.rs` sampled one in ten), four kernel
//! shapes, a load arm. `LLOOM_REDUCE_HAMMER_REPS` scales the count.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lloom_tensor::{Device, Result, Tensor};

fn reps() -> usize {
    std::env::var("LLOOM_REDUCE_HAMMER_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// A second command queue on the same GPU, running 2048² matmuls until told
/// to stop -- the "load" arm. Never synced except to bound memory.
fn spawn_load(stop: Arc<AtomicBool>) -> std::thread::JoinHandle<Result<()>> {
    std::thread::spawn(move || {
        let dev = Device::new_metal(0)?;
        let big = Tensor::from_vec(
            (0..2048 * 2048)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect::<Vec<_>>(),
            (2048, 2048),
            &dev,
        )?;
        let mut acc = big.clone();
        let mut n = 0usize;
        while !stop.load(Ordering::Relaxed) {
            acc = (big.matmul(&acc)? * 1e-4)?;
            n += 1;
            if n % 8 == 0 {
                let _ = acc.narrow(0, 0, 1)?.to_device(&Device::Cpu)?;
            }
        }
        Ok(())
    })
}

/// Runs `op` `reps` times in batches that are only read back once per batch
/// (no per-rep sync: the race needs the pipeline full), returning the
/// repetitions whose bits differ from the first.
fn hammer(name: &str, reps: usize, op: impl Fn() -> Result<Tensor>) -> Result<Vec<usize>> {
    let first = bits(&op()?)?;
    let mut bad = Vec::new();
    let batch = 25;
    let mut rep = 0;
    while rep < reps {
        let n = batch.min(reps - rep);
        let outs: Vec<Tensor> = (0..n).map(|_| op()).collect::<Result<_>>()?;
        let stacked = Tensor::stack(&outs, 0)?;
        let all = bits(&stacked)?;
        for (i, chunk) in all.chunks(first.len()).enumerate() {
            if chunk != first.as_slice() {
                bad.push(rep + i);
            }
        }
        rep += n;
    }
    eprintln!("{name}: {reps} reps, {} mismatched", bad.len());
    Ok(bad)
}

fn ramp(n: usize, f: f32, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * f).sin() * s).collect()
}

#[test]
fn reduce_and_softmax_kernels_are_repeatable_under_load() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let reps = reps();
    let stop = Arc::new(AtomicBool::new(false));
    let load = spawn_load(Arc::clone(&stop));

    // (a) strided middle-dim sum, BLOCKSIZE 64: the DeltaNet decode's read-out.
    let (h, dk, dv) = (48usize, 128usize, 128usize);
    let s = Tensor::from_vec(ramp(h * dk * dv, 0.0137, 0.3), (h, dk, dv), &dev)?;
    let k = Tensor::from_vec(ramp(h * dk, 0.071, 1.0), (h, dk, 1), &dev)?;
    let a = hammer("strided sum [48,128,128].sum(1)", reps, || {
        s.broadcast_mul(&k)?.sum(1)
    })?;

    // (b) contiguous last-dim sum, BLOCKSIZE 1024: the mean inside a norm over
    // a 2048-wide hidden, 1024 rows at once (two prefill chunks' worth).
    let x = Tensor::from_vec(ramp(1024 * 2048, 0.00731, 1.0), (1024, 2048), &dev)?;
    let b = hammer("contiguous sum [1024,2048].sum(1)", reps, || {
        x.sum_keepdim(1)
    })?;

    // (c) the softmax over the last dim, BLOCKSIZE 1024: attention scores.
    let sc = Tensor::from_vec(ramp(256 * 2048, 0.0113, 4.0), (256, 2048), &dev)?;
    let c = hammer("softmax [256,2048]", reps, || {
        lloom::nn::ops::softmax_last_dim(&sc)
    })?;

    // (d) control: a middle-dim sum with 8 threads per threadgroup never
    // touches threadgroup memory (the MoE's expert combine).
    let m = Tensor::from_vec(ramp(512 * 10 * 2048, 0.0031, 1.0), (512, 10, 2048), &dev)?;
    let d = hammer(
        "strided sum [512,10,2048].sum(1) (no shared memory)",
        reps,
        || m.sum(1),
    )?;

    stop.store(true, Ordering::Relaxed);
    load.join().expect("load thread")?;

    assert!(
        a.is_empty() && b.is_empty() && c.is_empty() && d.is_empty(),
        "bits moved between repetitions: strided {a:?}, contiguous {b:?}, softmax {c:?}, control {d:?}"
    );
    Ok(())
}
