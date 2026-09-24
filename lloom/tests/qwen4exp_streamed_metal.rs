//! On Metal, both loader arms of the streamed expert tier — `pread` + the
//! compute-encoder copy, and `MTLIOCommandQueue` — produce the resident
//! model's logits bit for bit (issue #749, §20.4a's A/B, correctness only).
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    set_prefill_chunk, set_prefill_pair, Config, Qwen4Exp, Streamed, PREFILL_CHUNK,
};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm};
use lloom::nn::model::CausalLM;
use lloom_tensor::kernels::metal::io_queue::IoQueue;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn write_checkpoint(cfg: &Config, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &dev, GgmlDType::Q8_0);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &refs)?;
    Ok(())
}

fn streamed(
    cfg: &Config,
    path: &Path,
    slots: usize,
    arm: LoaderArm,
    dev: &Device,
) -> Result<Qwen4Exp<Streamed>> {
    streamed_with(cfg, path, slots, arm, dev, false)
}

/// [`streamed`], with the `io` arm's hints on (`io_hints`).
fn streamed_with(
    cfg: &Config,
    path: &Path,
    slots: usize,
    arm: LoaderArm,
    dev: &Device,
    io_hints: bool,
) -> Result<Qwen4Exp<Streamed>> {
    let mut f = File::open(path)?;
    let content = Content::read(&mut f)?;
    testing::model_with(cfg, dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        let mut t = Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            slots,
            arm.clone(),
            dev,
        )?;
        t.set_io_hints(io_hints);
        Ok(t)
    })
}

fn run<R: lloom::nn::ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
) -> Result<Vec<Vec<f32>>> {
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let mut cache = model.new_cache(cfg, dev)?;
    let mut rows = Vec::new();
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
    for _ in 0..6 {
        let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        let input = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
        rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(rows)
}

/// The prefill knobs (`set_prefill_chunk`, `set_prefill_pair`) are process
/// statics: the tests here take turns.
static KNOBS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn both_loader_arms_match_the_resident_model_on_metal() -> Result<()> {
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    // The direct read (opt-in) is what these exercise on the pread arm:
    // pinned on before the first tier exists (read once a process).
    std::env::set_var("LLOOM_PREAD_DIRECT", "1");
    let dev = Device::new_metal(0)?;
    let mut cfg = Config::tiny();
    cfg.n_experts = 12;
    cfg.n_experts_per_tok = 3;
    // Q8_0 blocks are 32 wide: the down projection's k must be a multiple.
    cfg.moe_intermediate_size = 32;
    let path = std::env::temp_dir().join(format!("lloom-749-metal-{}.gguf", std::process::id()));
    write_checkpoint(&cfg, &path)?;

    // The resident reference on Metal, from the SAME Q8_0 stacks (read back
    // from the file so the bytes are the file's on every arm).
    let resident = {
        let mut f = File::open(&path)?;
        let content = Content::read(&mut f)?;
        testing::model_with(&cfg, &dev, |l| {
            let mut f = File::open(&path)?;
            let mut t = |n: &str| {
                content
                    .tensor(&mut f, &format!("blk.{l}.{n}"), &dev)
                    .map(Arc::new)
            };
            Ok(lloom::models::qwen4exp::Resident::new(
                t("ffn_gate_exps.weight")?,
                t("ffn_up_exps.weight")?,
                t("ffn_down_exps.weight")?,
            ))
        })?
    };
    let want = run(&resident, &cfg, &dev)?;
    assert!(want[0].iter().any(|v| v.abs() > 1e-6));

    let pread = streamed(&cfg, &path, 5, LoaderArm::Pread, &dev)?;
    let got = run(&pread, &cfg, &dev)?;
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(a == b, "pread arm: logits row {i} differs from resident");
    }

    let queue = match &dev {
        Device::Metal(d) => Arc::new(IoQueue::new(d.device()).map_err(lloom_tensor::Error::wrap)?),
        _ => unreachable!(),
    };
    let io = streamed(&cfg, &path, 5, LoaderArm::Io(queue.clone()), &dev)?;
    let got = run(&io, &cfg, &dev)?;
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(a == b, "io arm: logits row {i} differs from resident");
    }
    let (loads, bytes) = queue.counters();
    assert!(
        loads > 0 && bytes > 0,
        "the io queue must have loaded: {loads} loads, {bytes} bytes"
    );
    let mut stats = lloom::nn::expert_stream::StreamStats::default();
    for l in io.layers() {
        stats += l.moe().experts().stats();
    }
    assert_eq!(stats.bytes_loaded, bytes);
    assert!(stats.evictions > 0);

    // The io arm with its hints: the prefill's `prefetch_ahead` gives the
    // hinted experts slots at hint time and loads them on the IO queue; the
    // layer's use waits for them on the compute stream. Same logits, every
    // hinted expert either used (an in-flight hit) or overwritten (wasted),
    // no in-flight entry left dangling past a use, bytes as the queue's.
    let queue2 = match &dev {
        Device::Metal(d) => Arc::new(IoQueue::new(d.device()).map_err(lloom_tensor::Error::wrap)?),
        _ => unreachable!(),
    };
    let io_hinted = streamed_with(&cfg, &path, 5, LoaderArm::Io(queue2.clone()), &dev, true)?;
    let got = run(&io_hinted, &cfg, &dev)?;
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            a == b,
            "io arm with hints: logits row {i} differs from resident"
        );
    }
    let (_, bytes2) = queue2.counters();
    let mut hinted = lloom::nn::expert_stream::StreamStats::default();
    let mut inflight = 0usize;
    for l in io_hinted.layers() {
        hinted += l.moe().experts().stats();
        inflight += l.moe().experts().inflight_count();
    }
    assert!(hinted.io_hints > 0, "no hint was issued: {hinted:?}");
    // Whether a use found its expert still in flight (and waited) or the
    // host saw it landed first is the DMA's timing against a tiny model;
    // either way the hint reached a slot.
    assert!(
        hinted.io_inflight_hits + hinted.io_inflight_landed > 0,
        "no hinted expert landed: {hinted:?}"
    );
    assert!(
        hinted.io_inflight_hits
            + hinted.io_inflight_wasted
            + hinted.io_inflight_landed
            + inflight as u64
            == hinted.io_hints,
        "hints {} = in-flight hits {} + wasted {} + landed unwaited {} + still in flight {inflight}",
        hinted.io_hints,
        hinted.io_inflight_hits,
        hinted.io_inflight_wasted,
        hinted.io_inflight_landed
    );
    assert_eq!(hinted.bytes_loaded, bytes2, "bytes must be the queue's");
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// The paired prefill (two chunks interleaved a layer apart, one pass)
/// against chunk after chunk, on the pread arm with a pool smaller than a
/// chunk's working set (loads at every layer of both tiles): the same
/// logits bit for bit, for the prefill's last position and the decode
/// steps after it (the KV, the recurrent state and the PLE window the pair
/// left behind are what the single path leaves).
#[test]
fn paired_prefill_matches_chunk_after_chunk_on_metal() -> Result<()> {
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    // The direct read (opt-in) is what these exercise on the pread arm:
    // pinned on before the first tier exists (read once a process).
    std::env::set_var("LLOOM_PREAD_DIRECT", "1");
    let dev = Device::new_metal(0)?;
    let mut cfg = Config::tiny();
    cfg.n_experts = 12;
    cfg.n_experts_per_tok = 3;
    cfg.moe_intermediate_size = 32;
    let path = std::env::temp_dir().join(format!("lloom-749-pair-{}.gguf", std::process::id()));
    write_checkpoint(&cfg, &path)?;
    // Chunks of 3 over an 8-token prompt: (3, 3) paired, then 2 alone.
    set_prefill_chunk(3);
    let run_with = |pair: bool| -> Result<Vec<Vec<f32>>> {
        set_prefill_pair(pair);
        let model = streamed(&cfg, &path, 5, LoaderArm::Pread, &dev)?;
        let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 4];
        let mut cache = model.new_cache(&cfg, &dev)?;
        let mut rows = Vec::new();
        let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
        let mut logits = model.forward(&input, 0, &mut cache)?;
        rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(
            cache.pos,
            prompt.len(),
            "the pass must advance the position"
        );
        for _ in 0..4 {
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            let input = Tensor::from_vec(vec![next], (1, 1), &dev)?;
            logits = model.forward(&input, cache.pos, &mut cache)?;
            rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(rows)
    };
    let single = run_with(false)?;
    let paired = run_with(true)?;
    set_prefill_chunk(PREFILL_CHUNK);
    set_prefill_pair(lloom::models::qwen4exp::DEFAULT_PREFILL_PAIR);
    assert!(single[0].iter().any(|v| v.abs() > 1e-6));
    for (i, (a, b)) in single.iter().zip(&paired).enumerate() {
        assert!(
            a == b,
            "paired prefill: logits row {i} differs from chunk after chunk"
        );
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
