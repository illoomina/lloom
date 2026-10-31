//! The N-gram PLE of a batch's rows and of a verify window's positions read
//! in one request into one region (`models.qwen4exp.ple.rows_batch`), against
//! each position hashed, read and uploaded on its own: the same logits bit for
//! bit at B = 1, 2 and 8, and the same window argmaxes and states after every
//! accept, at f32 and IQ4_NL. The batched arm reads a step's rows in ONE
//! request where the per-position arm reads one a position, and on Metal it
//! uploads nothing at a decode step while the prompt's chunk still uploads
//! (the counter's positive control).
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    set_ple_rows_batch, Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident,
    State,
};
use lloom::nn::model::{CausalLM, Row};
use lloom::nn::ngram_memory::{read_worker_stats, ReadPolicy, ReadPool};
use lloom::profile::{CrownExact, Profile, RewindableExact};
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

const TABLE_ROWS: usize = 80;

/// Read requests the table has made so far: those the read threads took and
/// those they handed back to rayon's pool (another request in flight).
fn requests() -> u64 {
    let s = read_worker_stats().unwrap_or_default();
    s.requests + s.refused
}

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lloom-ple-rows-batch-{}-{tag}.gguf",
        std::process::id()
    ))
}

/// A `[TABLE_ROWS, head_dim]` table: f32 values distinct per element, or
/// IQ4_NL blocks built byte by byte with a finite scale and varied nibbles.
fn write_table(head_dim: usize, dtype: GgmlDType, path: &Path) -> Result<()> {
    let q = match dtype {
        GgmlDType::F32 => {
            let t = testing::wt((TABLE_ROWS, head_dim), 3.0, &Device::Cpu);
            QTensor::quantize(&t, GgmlDType::F32)?
        }
        GgmlDType::IQ4NL => {
            let blocks = TABLE_ROWS * head_dim / dtype.block_size();
            let mut bytes = Vec::with_capacity(blocks * dtype.type_size());
            for b in 0..blocks {
                let d = half::f16::from_f32(0.01 + (b % 13) as f32 * 0.003);
                bytes.extend_from_slice(&d.to_le_bytes());
                bytes.extend((0..16).map(|j| ((b * 7 + j * 11) % 256) as u8));
            }
            let st = QStorage::from_data(std::borrow::Cow::Owned(bytes), &Device::Cpu, dtype)?;
            QTensor::new(st, (TABLE_ROWS, head_dim))?
        }
        other => panic!("no fixture for {other:?}"),
    };
    gguf_file::write(
        &mut File::create(path)?,
        &[],
        &[("per_layer_token_embd.weight", &q)],
    )?;
    Ok(())
}

fn model_with_ple<P: Profile>(
    cfg: &Config,
    dev: &Device,
    path: &Path,
) -> Result<Qwen4Exp<P, Resident>> {
    let mut model = testing::resident_model::<P>(cfg, dev);
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::<P>::new(
        testing::lin(hc_dim, ple_dim, 21.0, dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::ones(hc_dim, dev),
        testing::wt((hc_dim, 4), 23.0, dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, PleTable::Mmap(MmapPleTable::open(path)?), block)?;
    // Every row through the read threads, so a request is counted per read.
    let rows = model
        .ple_table()
        .and_then(|t| t.mmap())
        .expect("a mapped table");
    rows.set_read_policy(ReadPolicy::Concurrent);
    rows.set_read_pool(ReadPool::Workers);
    rows.set_hedge_us(0);
    Ok(model)
}

fn config(dev: &Device) -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg.ple_head_dim = 32;
    if dev.is_metal() {
        cfg.head_dim = 32;
    }
    cfg
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, &x)| if x > m.1 { (i, x) } else { m },
        )
        .0 as u32
}

/// `b` prompts of different lengths and tokens.
fn prompts(b: usize) -> Vec<Vec<u32>> {
    (0..b)
        .map(|r| {
            (0..3 + r % 4)
                .map(|i| ((r * 5 + i * 3 + 1) % 17) as u32)
                .collect()
        })
        .collect()
}

/// What one arm's batched decode left: every row's logits at every step, the
/// PLE uploads and the read requests of the decode steps alone, and the
/// uploads of the prefill.
struct Run {
    logits: Vec<Vec<Vec<f32>>>,
    step_uploads: u64,
    step_requests: u64,
    prefill_uploads: u64,
}

fn batched(
    model: &Qwen4Exp<CrownExact, Resident>,
    cfg: &Config,
    dev: &Device,
    b: usize,
    steps: usize,
) -> Result<Run> {
    let mut caches: Vec<State<CrownExact>> = Vec::with_capacity(b);
    let mut next = Vec::with_capacity(b);
    let u0 = model.ple_row_uploads();
    for p in prompts(b) {
        let mut cache = model.new_cache(cfg, dev)?;
        let input = Tensor::from_vec(p.clone(), (1, p.len()), dev)?;
        let logits: Vec<f32> = model
            .forward(&input, 0, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        next.push(argmax(&logits));
        caches.push(cache);
    }
    let prefill_uploads = model.ple_row_uploads() - u0;
    let u1 = model.ple_row_uploads();
    let r1 = requests();
    let mut logits = vec![Vec::new(); b];
    for _ in 0..steps {
        let tokens = Tensor::from_vec(next.clone(), (b, 1), dev)?;
        let mut rows: Vec<Row<'_, State<CrownExact>>> = caches
            .iter_mut()
            .map(|c| Row {
                pos: c.pos,
                cache: c,
            })
            .collect();
        let out = model.forward_rows(&tokens, &mut rows)?;
        for (r, l) in logits.iter_mut().enumerate() {
            let row: Vec<f32> = out.narrow(0, r, 1)?.flatten_all()?.to_vec1()?;
            next[r] = argmax(&row);
            l.push(row);
        }
    }
    Ok(Run {
        logits,
        step_uploads: model.ple_row_uploads() - u1,
        step_requests: requests() - r1,
        prefill_uploads,
    })
}

fn rows_on(dev: &Device, tag: &str) -> Result<()> {
    let cfg = config(dev);
    let steps = 4;
    for dtype in [GgmlDType::F32, GgmlDType::IQ4NL] {
        let path = temp_gguf(&format!("{tag}-{dtype:?}"));
        write_table(cfg.ple_head_dim, dtype, &path)?;
        let model = model_with_ple::<CrownExact>(&cfg, dev, &path)?;
        for b in [1usize, 2, 8] {
            set_ple_rows_batch(false);
            let rows = batched(&model, &cfg, dev, b, steps)?;
            set_ple_rows_batch(true);
            let batch = batched(&model, &cfg, dev, b, steps)?;
            for r in 0..b {
                for t in 0..steps {
                    assert!(
                        batch.logits[r][t] == rows.logits[r][t],
                        "{tag} {dtype:?} B={b}: row {r} step {t}: the batched PLE moved the logits"
                    );
                }
            }
            if b > 1 {
                // Non-vacuity: the rows hash different windows.
                assert!(
                    batch.logits[0] != batch.logits[1],
                    "{tag} {dtype:?} B={b}: rows 0 and 1 decode alike"
                );
            }
            assert_eq!(
                batch.step_requests, steps as u64,
                "{tag} {dtype:?} B={b}: one read request a step"
            );
            assert_eq!(
                rows.step_requests,
                (b * steps) as u64,
                "{tag} {dtype:?} B={b}: the per-position arm reads once a row a step"
            );
            if dev.is_metal() {
                assert_eq!(
                    batch.step_uploads, 0,
                    "{tag} {dtype:?} B={b}: a decode step uploaded PLE rows"
                );
                assert!(
                    batch.prefill_uploads > 0,
                    "{tag} {dtype:?} B={b}: the prompt's chunk uploaded nothing: the counter is dead"
                );
            }
            println!(
                "PLEROWS {tag} {dtype:?} B={b}: requests {} vs {}, step uploads {} vs {}, prefill uploads {}",
                batch.step_requests,
                rows.step_requests,
                batch.step_uploads,
                rows.step_uploads,
                batch.prefill_uploads
            );
        }
        std::fs::remove_file(&path).ok();
    }
    Ok(())
}

/// A verify window's argmaxes and, after every accept, its state on both
/// arms.
type Window = (Vec<u32>, Vec<Vec<(String, u64, usize)>>);

fn window_on(dev: &Device, tag: &str) -> Result<()> {
    let cfg = config(dev);
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let window: Vec<u32> = vec![4, 11, 7, 2];
    let t = window.len();
    for dtype in [GgmlDType::F32, GgmlDType::IQ4NL] {
        let path = temp_gguf(&format!("{tag}-window-{dtype:?}"));
        write_table(cfg.ple_head_dim, dtype, &path)?;
        let model = model_with_ple::<RewindableExact>(&cfg, dev, &path)?;
        let run = |on: bool| -> Result<Window> {
            set_ple_rows_batch(on);
            let mut argmax_all = Vec::new();
            let mut sums = Vec::new();
            for a in 0..=t {
                let pools = model.spec_pools(1, t - 1, dev);
                let mut c = State::<RewindableExact>::in_ring(&cfg, &pools)?;
                let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev)?;
                model.forward(&input, 0, &mut c)?;
                let r1 = requests();
                let argmax = model.verify_window(&window, &mut c)?;
                let asked = requests() - r1;
                if dev.is_metal() {
                    let want = if on { 1 } else { t as u64 };
                    assert_eq!(
                        asked, want,
                        "{tag} {dtype:?}: the window's read requests (batched {on})"
                    );
                }
                c.accept_window(a)?;
                argmax_all.push(argmax);
                sums.push(c.checksums()?);
            }
            Ok((argmax_all.concat(), sums))
        };
        let rows = run(false)?;
        let batch = run(true)?;
        assert_eq!(batch.0, rows.0, "{tag} {dtype:?}: the window's argmaxes");
        for a in 0..=t {
            assert_eq!(
                batch.1[a], rows.1[a],
                "{tag} {dtype:?}: the state after accepting {a}"
            );
        }
        println!("PLEWINDOW {tag} {dtype:?} ok (T={t})");
        std::fs::remove_file(&path).ok();
    }
    set_ple_rows_batch(true);
    Ok(())
}

#[test]
fn a_batch_reads_its_rows_ple_in_one_request_cpu() -> Result<()> {
    rows_on(&Device::Cpu, "cpu")
}

#[cfg(feature = "metal")]
#[test]
fn a_batch_reads_its_rows_ple_in_one_request_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => rows_on(&dev, "metal"),
        Err(_) => Ok(()),
    }
}

#[cfg(feature = "metal")]
#[test]
fn a_verify_window_reads_its_positions_ple_in_one_request_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => window_on(&dev, "metal"),
        Err(_) => Ok(()),
    }
}
