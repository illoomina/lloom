//! The streamed tier planned on the device (no per-layer readback; one
//! settle per token; a token that computed on a missing expert is replayed
//! from the window) decodes the ids and the logits the resident model
//! decodes — bit for bit — on a pool small enough that tokens miss and are
//! replayed. CPU and, when present, Metal (the map kernel).
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::CausalLM;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lloom-device-plan-{}-{tag}.gguf",
        std::process::id()
    ))
}

/// Every layer's expert stacks in one GGUF, named as the loader names them.
fn write_experts(cfg: &Config, path: &Path) -> Result<Vec<[QTensor; 3]>> {
    let dev = Device::Cpu;
    let stacks: Vec<[QTensor; 3]> = (0..cfg.n_layers)
        .map(|l| testing::expert_stacks(cfg, l, &dev, GgmlDType::F32))
        .collect();
    let names: Vec<String> = (0..cfg.n_layers)
        .flat_map(|l| {
            [
                format!("blk.{l}.ffn_gate_exps.weight"),
                format!("blk.{l}.ffn_up_exps.weight"),
                format!("blk.{l}.ffn_down_exps.weight"),
            ]
        })
        .collect();
    let refs: Vec<(&str, &QTensor)> = names
        .iter()
        .zip(stacks.iter().flat_map(|s| s.iter()))
        .map(|(n, t)| (n.as_str(), t))
        .collect();
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &refs)?;
    Ok(stacks)
}

fn streamed_model(
    cfg: &Config,
    dev: &Device,
    path: &Path,
    slots: usize,
) -> Result<Qwen4Exp<Streamed>> {
    let content = Content::read(&mut File::open(path)?)?;
    testing::model_with(cfg, dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, n);
        Streamed::new(
            src(&format!("blk.{l}.ffn_gate_exps.weight"))?,
            src(&format!("blk.{l}.ffn_up_exps.weight"))?,
            src(&format!("blk.{l}.ffn_down_exps.weight"))?,
            slots,
            LoaderArm::Pread,
            dev,
        )
    })
}

fn decode<R: lloom::nn::expert_stream::ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>, lloom::models::qwen4exp::Cache)> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let (mut ids, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = row
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &v)| if v > m.1 { (i, v) } else { m },
            )
            .0 as u32;
        rows.push(row);
        ids.push(next);
        let input = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
    }
    Ok((ids, rows, cache))
}

fn check_on(dev: &Device, tag: &str) -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    let path = temp_gguf(tag);
    let stacks = write_experts(&cfg, &path)?;
    let resident: Qwen4Exp<Resident> = testing::model_with(&cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(&cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })?;
    drop(stacks);
    // 3 slots for 8 experts at top-2: tokens miss, settles load, passes replay.
    let streamed = streamed_model(&cfg, dev, &path, 3)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let steps = 10;
    let (want_ids, want_rows, _) = decode(&resident, &cfg, dev, &prompt, steps)?;
    let (ids, rows, cache) = decode(&streamed, &cfg, dev, &prompt, steps)?;
    assert_eq!(ids, want_ids, "{tag}: the device-planned ids differ");
    for (t, (a, b)) in rows.iter().zip(&want_rows).enumerate() {
        assert_eq!(a, b, "{tag}: step {t} logits differ");
    }
    let st = streamed.stream_stats();
    assert!(st.device_plans > 0, "{tag}: no device plan ran: {st:?}");
    assert!(
        cache.redos > 0 && st.settled_misses > 0,
        "{tag}: no token was replayed — the pool never missed: {st:?} redos {}",
        cache.redos
    );
    println!("DEVICEPLAN {tag} steps={steps} device_plans={} settled_misses={} redos={} hits={} misses={}",
        st.device_plans, st.settled_misses, cache.redos, st.hits, st.misses);
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn device_plan_decodes_the_resident_stream_cpu() -> Result<()> {
    check_on(&Device::Cpu, "cpu")
}

#[cfg(feature = "metal")]
#[test]
fn device_plan_decodes_the_resident_stream_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => check_on(&dev, "metal"),
        Err(_) => Ok(()),
    }
}
