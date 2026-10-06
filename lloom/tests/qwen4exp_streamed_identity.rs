//! The streamed expert tier is byte-identical to the resident one, on a whole
//! synthetic Qwen4-exp model (issue #749): the same routed kernels read the
//! same bytes from a slot pool instead of the stacked tensor, so every logit
//! row matches exactly, over a chunked prefill (groups + evictions) and
//! decode steps (hits). The counters are the non-vacuity arm: bytes were
//! loaded, slots were evicted, hits happened.
//!
//! Second gate, same shape: the N-gram tier's hot-row cache gathers the same
//! rows as the plain memory map.
//!
//! CPU: the question is the residency bookkeeping, not a kernel.

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    CachedPleTable, Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident, State,
    Streamed,
};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-749-{}-{tag}.gguf", std::process::id()))
}

/// Write every layer's expert stacks (and a PLE table) to one GGUF, as the
/// loader names them.
fn write_checkpoint(cfg: &Config, ple_rows: usize, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &dev, GgmlDType::F32);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let table = testing::wt((ple_rows, cfg.ple_head_dim), 3.0, &dev);
    owned.push((
        "per_layer_token_embd.weight".to_string(),
        QTensor::quantize(&table, GgmlDType::F32)?,
    ));
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &refs)?;
    Ok(())
}

fn streamed_model(
    cfg: &Config,
    path: &Path,
    slots: usize,
) -> Result<Qwen4Exp<CrownExact, Streamed>> {
    let mut f = File::open(path)?;
    let content = Content::read(&mut f)?;
    let dev = Device::Cpu;
    testing::model_with::<CrownExact, _>(cfg, &dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            slots,
            LoaderArm::Pread,
            &dev,
        )
    })
}

/// Prompt (chunk prefill) + decode steps; every logits row.
fn run<R: lloom::nn::ExpertResidency>(
    model: &Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<Vec<f32>>, State<CrownExact>)> {
    let dev = Device::Cpu;
    let mut cache = model.new_cache(cfg, &dev)?;
    let mut rows = Vec::new();
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
    for _ in 0..steps {
        let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        let input = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
        rows.push(logits.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok((rows, cache))
}

fn ple_parts(cfg: &Config, path: &Path) -> Result<(NgramHash, MmapPleTable, PleLayer<CrownExact>)> {
    let dev = Device::Cpu;
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let table = MmapPleTable::open(path)?;
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::<CrownExact>::new(
        testing::lin(hc_dim, ple_dim, 21.0, &dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::wt((hc_dim, 4), 23.0, &dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    Ok((hash, table, block))
}

#[test]
fn streamed_model_is_byte_identical_to_resident() -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 12;
    cfg.n_experts_per_tok = 3;
    let path = temp_gguf("streamed");
    write_checkpoint(&cfg, 80, &path)?;
    let dev = Device::Cpu;

    let resident = testing::resident_model::<CrownExact>(&cfg, &dev);
    // 5 slots: a 7-row chunk's union (up to 12) needs several groups, and
    // decode churns the pool.
    let streamed = streamed_model(&cfg, &path, 5)?;

    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let (a, _) = run(&resident, &cfg, &prompt, 6)?;
    let (b, _) = run(&streamed, &cfg, &prompt, 6)?;
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            x == y,
            "logits row {i} differs between resident and streamed"
        );
    }
    // non-vacuity: the rows are not degenerate and the streaming happened
    assert!(a[0].iter().any(|v| v.abs() > 1e-6));
    let mut stats = lloom::nn::expert_stream::StreamStats::default();
    for l in streamed.layers() {
        stats += l.moe().experts().stats();
    }
    // A plan a layer a pass at least, host or device: under
    // `QWEN4EXP_REPLAY=device` a token that missed replays on the device
    // plan, so its layers' plans land in `device_plans`, not `plans`.
    assert!(
        stats.plans + stats.device_plans >= 7 * cfg.n_layers as u64,
        "plans {} device_plans {}",
        stats.plans,
        stats.device_plans
    );
    assert!(
        stats.groups > stats.plans,
        "the chunk must have been grouped: {stats:?}"
    );
    assert!(
        stats.hits > 0 && stats.misses > 0 && stats.evictions > 0,
        "{stats:?}"
    );
    assert!(stats.bytes_loaded > 0);
    // The arm, when this binary runs under it: the chunk's groups ran in
    // readiness order at least once (a count at 0 says the arm did not
    // fire, and the identity above was vacuous for it).
    if std::env::var("QWEN4EXP_READY_FIRST").is_ok_and(|v| v == "1") {
        let n = lloom::models::qwen4exp::moe::ready_reorders();
        assert!(
            n > 0,
            "QWEN4EXP_READY_FIRST=1: no tile's groups were reordered"
        );
        println!("READY-FIRST reorders={n}");
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn ple_hot_row_cache_gathers_the_same_rows() -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    let path = temp_gguf("ple");
    write_checkpoint(&cfg, 80, &path)?;
    let dev = Device::Cpu;

    // the gather itself
    let plain = PleTable::Mmap(MmapPleTable::open(&path)?);
    let cached = PleTable::Cached(CachedPleTable::new(MmapPleTable::open(&path)?, 16)?);
    // A skewed stream: every third lookup is one of 5 hot rows (hits), the
    // rest walk 40 distinct rows (misses + evictions in a 16-row cache).
    let ids: Vec<i64> = (0..200)
        .map(|i| if i % 3 == 0 { (i / 3) % 5 } else { (i * 7) % 40 } as i64)
        .collect();
    for chunk in ids.chunks(8) {
        let a = plain.gather(chunk, &dev)?.to_vec1::<f32>()?;
        let b = cached.gather(chunk, &dev)?.to_vec1::<f32>()?;
        assert!(a == b, "gather differs on {chunk:?}");
    }
    let (hits, misses) = cached.cache_stats();
    assert!(hits > 0 && misses > 0, "hits {hits} misses {misses}");
    assert!(
        misses > 16,
        "a 16-row cache over 40 ids must evict: misses {misses}"
    );

    // through the model: cached PLE == mmap PLE
    let mut with_mmap = testing::resident_model::<CrownExact>(&cfg, &dev);
    let (hash, table, block) = ple_parts(&cfg, &path)?;
    with_mmap.set_ple(1, hash, PleTable::Mmap(table), block)?;
    let mut with_cache = testing::resident_model::<CrownExact>(&cfg, &dev);
    let (hash, table, block) = ple_parts(&cfg, &path)?;
    with_cache.set_ple(
        1,
        hash,
        PleTable::Cached(CachedPleTable::new(table, 8)?),
        block,
    )?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let (a, _) = run(&with_mmap, &cfg, &prompt, 5)?;
    let (b, _) = run(&with_cache, &cfg, &prompt, 5)?;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x == y, "logits row {i} differs between mmap and cached PLE");
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// The seam's `Resident` arm is what the tests above compare against; make
/// sure the resident model here is the same object the model tests build.
#[test]
fn resident_reference_is_deterministic() -> Result<()> {
    let cfg = Config::tiny();
    let dev = Device::Cpu;
    let a = testing::resident_model::<CrownExact>(&cfg, &dev);
    let b: Qwen4Exp<CrownExact, Resident> = testing::resident_model::<CrownExact>(&cfg, &dev);
    let prompt: Vec<u32> = vec![1, 2, 3];
    let (x, _) = run(&a, &cfg, &prompt, 2)?;
    let (y, _) = run(&b, &cfg, &prompt, 2)?;
    assert!(x == y);
    Ok(())
}
