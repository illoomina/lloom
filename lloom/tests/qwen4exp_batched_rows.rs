//! Each row of a batched decode step (`BatchedCausalLM::forward_rows`) is
//! its own forward: three sequences of different prompts and lengths,
//! prefilled through `forward`, then decoded together for six steps — every
//! row's logits equal the logits of that row decoded alone, on the CPU
//! (the composed paths) and on Metal (the fused kernels, the stems batched).
//! And the streamed tier under the device plan over rows decodes what the
//! resident model decodes, replays included.
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Cache, Config, Qwen4Exp, Resident};
use lloom::nn::expert_stream::{ExpertResidency, ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::{CausalLM, Row};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![3, 5, 2, 8, 8],
        vec![1, 9, 3, 5, 7, 7, 2, 4, 6],
        vec![11, 2, 13],
    ]
}

/// Per row: the ids, the logits of every step; and the caches after.
type Batched = (Vec<Vec<u32>>, Vec<Vec<Vec<f32>>>, Vec<Cache>);

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0
}

/// Every row alone: `steps` greedy tokens after its prompt, the logits of
/// every step.
fn solo<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let (mut ids, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = argmax(&row) as u32;
        rows.push(row);
        ids.push(next);
        logits = model.forward(
            &Tensor::from_vec(vec![next], (1, 1), dev)?,
            cache.pos,
            &mut cache,
        )?;
    }
    Ok((ids, rows))
}

/// The rows together: each prefilled alone, then `steps` batched steps.
fn batched<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompts: &[Vec<u32>],
    steps: usize,
) -> Result<Batched> {
    let b = prompts.len();
    let mut caches: Vec<Cache> = Vec::with_capacity(b);
    let mut next: Vec<u32> = Vec::with_capacity(b);
    let mut ids = vec![Vec::new(); b];
    let mut rows_logits = vec![Vec::new(); b];
    for p in prompts {
        let mut cache = model.new_cache(cfg, dev)?;
        let input = Tensor::from_vec(p.clone(), (1, p.len()), dev)?;
        let logits: Vec<f32> = model
            .forward(&input, 0, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        next.push(argmax(&logits) as u32);
        caches.push(cache);
    }
    for _ in 0..steps {
        for r in 0..b {
            ids[r].push(next[r]);
        }
        let tokens = Tensor::from_vec(next.clone(), (b, 1), dev)?;
        let mut rows: Vec<Row<'_, Cache>> = caches
            .iter_mut()
            .map(|c| Row {
                pos: c.pos,
                cache: c,
            })
            .collect();
        let logits = model.forward_rows(&tokens, &mut rows)?;
        assert_eq!(logits.dims(), &[b, cfg.vocab_size]);
        for r in 0..b {
            let row: Vec<f32> = logits.narrow(0, r, 1)?.flatten_all()?.to_vec1()?;
            next[r] = argmax(&row) as u32;
            rows_logits[r].push(row);
        }
    }
    Ok((ids, rows_logits, caches))
}

fn check_rows(
    tag: &str,
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    dev: &Device,
    tol: f32,
) -> Result<()> {
    let prompts = prompts();
    let steps = 6;
    let (ids_b, logits_b, caches) = batched(model, cfg, dev, &prompts, steps)?;
    let mut worst = 0f32;
    for (r, p) in prompts.iter().enumerate() {
        let (ids_s, logits_s) = solo(model, cfg, dev, p, steps)?;
        // The batched step fed the argmax of the PREVIOUS logits; the solo
        // ids are the argmaxes of the same logits — the streams align from
        // the prompt's own argmax onward.
        let want: Vec<u32> = std::iter::once(ids_s[0])
            .chain(ids_s[1..].iter().copied())
            .collect();
        assert_eq!(ids_b[r], want, "{tag}: row {r}: the ids diverge");
        for (t, (a, b)) in logits_s.iter().skip(1).zip(logits_b[r].iter()).enumerate() {
            assert_eq!(
                argmax(a),
                argmax(b),
                "{tag}: row {r} step {t}: the argmax moved"
            );
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale);
            assert!(
                diff <= tol * scale,
                "{tag}: row {r} step {t}: differs by {diff} at {scale}"
            );
        }
        assert_eq!(caches[r].pos, p.len() + steps, "{tag}: row {r}'s position");
    }
    // Non-vacuity: the rows are not the same stream.
    assert!(
        ids_b[0] != ids_b[1] || ids_b[1] != ids_b[2],
        "{tag}: identical rows"
    );
    println!("ROWS {tag} worst relative diff over rows/steps: {worst:e}");
    Ok(())
}

#[test]
fn each_row_of_a_batched_step_is_its_own_forward_on_the_cpu() -> Result<()> {
    let cfg = Config::tiny();
    let dev = Device::Cpu;
    let model = testing::resident_model(&cfg, &dev);
    check_rows("cpu", &model, &cfg, &dev, 1e-4)
}

#[cfg(feature = "metal")]
#[test]
fn each_row_of_a_batched_step_is_its_own_forward_on_metal() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let mut cfg = Config::tiny();
    cfg.head_dim = 32; // the fused QSA step needs a head the sdpa kernel is built for
    let model = testing::resident_model(&cfg, &dev);
    check_rows("metal", &model, &cfg, &dev, 1e-4)
}

#[test]
fn forward_rows_refuses_what_it_cannot_batch() -> Result<()> {
    let cfg = Config::tiny();
    let dev = Device::Cpu;
    let model = testing::resident_model(&cfg, &dev);
    let mut a = model.new_cache(&cfg, &dev)?;
    let mut b = model.new_cache(&cfg, &dev)?;
    let input = Tensor::from_vec(vec![3u32, 5, 2], (1, 3), &dev)?;
    model.forward(&input, 0, &mut a)?;
    model.forward(&input, 0, &mut b)?;
    let tokens = Tensor::from_vec(vec![4u32, 4], (2, 1), &dev)?;
    // A row whose position is not its cache's.
    {
        let mut rows = vec![
            Row {
                pos: 3,
                cache: &mut a,
            },
            Row {
                pos: 7,
                cache: &mut b,
            },
        ];
        assert!(model.forward_rows(&tokens, &mut rows).is_err());
    }
    // Two tokens per row.
    {
        let two = Tensor::from_vec(vec![4u32, 4, 4, 4], (2, 2), &dev)?;
        let mut rows = vec![
            Row {
                pos: 3,
                cache: &mut a,
            },
            Row {
                pos: 3,
                cache: &mut b,
            },
        ];
        assert!(model.forward_rows(&two, &mut rows).is_err());
    }
    // An open speculative window.
    b.open_window()?;
    {
        let mut rows = vec![
            Row {
                pos: 3,
                cache: &mut a,
            },
            Row {
                pos: 3,
                cache: &mut b,
            },
        ];
        assert!(model.forward_rows(&tokens, &mut rows).is_err());
    }
    b.resolve_window()?;
    let mut rows = vec![
        Row {
            pos: 3,
            cache: &mut a,
        },
        Row {
            pos: 3,
            cache: &mut b,
        },
    ];
    assert!(model.forward_rows(&tokens, &mut rows).is_ok());
    Ok(())
}

// ---- the streamed tier over rows ----------------------------------------------

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-rows-{}-{tag}.gguf", std::process::id()))
}

fn write_experts(cfg: &Config, path: &Path) -> Result<()> {
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
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    Ok(())
}

fn streamed_rows_on(dev: &Device, tag: &str) -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    if dev.is_metal() {
        cfg.head_dim = 32;
    }
    let path = temp_gguf(tag);
    write_experts(&cfg, &path)?;
    let resident: Qwen4Exp<Resident> = testing::model_with(&cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(&cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })?;
    let content = Content::read(&mut File::open(&path)?)?;
    let prompts = prompts();
    let steps = 8;
    let (ids_r, logits_r, _) = batched(&resident, &cfg, dev, &prompts, steps)?;
    // A pool that thrashes (5 slots for 3 rows × top-2 of 8: the host plan
    // most steps, replays when the device plan runs) and one that holds
    // everything once seen (8 slots: the device plan from the second step).
    let mut any_device = 0u64;
    for slots in [5usize, 8] {
        let streamed: Qwen4Exp<Streamed> = testing::model_with(&cfg, dev, |l| {
            let src = |n: &str| ExpertSource::from_gguf(&content, &path, n);
            Streamed::new(
                src(&format!("blk.{l}.ffn_gate_exps.weight"))?,
                src(&format!("blk.{l}.ffn_up_exps.weight"))?,
                src(&format!("blk.{l}.ffn_down_exps.weight"))?,
                slots,
                LoaderArm::Pread,
                dev,
            )
        })?;
        let (ids_s, logits_s, caches) = batched(&streamed, &cfg, dev, &prompts, steps)?;
        assert_eq!(
            ids_s, ids_r,
            "{tag} slots {slots}: the streamed rows decode other ids"
        );
        for r in 0..prompts.len() {
            for (t, (a, b)) in logits_r[r].iter().zip(&logits_s[r]).enumerate() {
                assert_eq!(
                    a, b,
                    "{tag} slots {slots}: row {r} step {t}: the streamed logits differ"
                );
            }
        }
        let st = streamed.stream_stats();
        let redos: u64 = caches.iter().map(|c| c.redos).sum();
        println!(
            "ROWS {tag} streamed slots={slots}: device_plans={} settled_misses={} redos={redos} host_plan_passes={}",
            st.device_plans,
            st.settled_misses,
            caches.iter().map(|c| c.host_plan_passes).sum::<u64>()
        );
        any_device += st.device_plans;
        if slots == 8 {
            assert!(
                st.device_plans > 0,
                "{tag}: a pool holding every expert never planned on the device"
            );
        }
    }
    assert!(any_device > 0);
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn the_streamed_rows_decode_the_resident_rows_cpu() -> Result<()> {
    streamed_rows_on(&Device::Cpu, "cpu")
}

#[cfg(feature = "metal")]
#[test]
fn the_streamed_rows_decode_the_resident_rows_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => streamed_rows_on(&dev, "metal"),
        Err(_) => Ok(()),
    }
}
