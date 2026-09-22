//! A hotlist gathered from one run warms a fresh pool for the next: the
//! same stream decodes the same ids with fewer misses and fewer replays —
//! none, when every layer's working set fits the pool. CPU.
use lloom::models::qwen4exp::hotlist::Hotlist;
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::CausalLM;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-hotlist-{}-{tag}", std::process::id()))
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

fn decode(
    model: &Qwen4Exp<Streamed>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, u64)> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let mut ids = Vec::new();
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
        ids.push(next);
        logits = model.forward(
            &Tensor::from_vec(vec![next], (1, 1), dev)?,
            cache.pos,
            &mut cache,
        )?;
    }
    Ok((ids, cache.redos))
}

#[test]
fn a_hotlist_warms_the_next_load() -> Result<()> {
    let dev = Device::Cpu;
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    let gguf = temp("experts.gguf");
    write_experts(&cfg, &gguf)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let slots = 6;

    // The cold run: misses, replays; its usage is the hotlist.
    let cold = streamed_model(&cfg, &dev, &gguf, slots)?;
    let (ids_cold, redos_cold) = decode(&cold, &cfg, &dev, &prompt, 10)?;
    let cold_stats = cold.stream_stats();
    let hot = Hotlist::from_usage(&cold.stream_usage());
    let file = temp("hotlist.txt");
    hot.write(&file)?;
    let hot = Hotlist::read(&file)?;
    assert!(!hot.is_empty());

    // A fresh pool, warmed from the file, decodes the same stream.
    let warm = streamed_model(&cfg, &dev, &gguf, slots)?;
    let loaded = warm.warm_from_hotlist(&hot)?;
    assert!(loaded > 0, "nothing warmed");
    let after_warm = warm.stream_stats();
    assert_eq!(
        after_warm.misses as usize, loaded,
        "the warm's loads are its misses"
    );
    let (ids_warm, redos_warm) = decode(&warm, &cfg, &dev, &prompt, 10)?;
    let warm_stats = warm.stream_stats();
    assert_eq!(ids_warm, ids_cold, "the warmed pool decodes another stream");
    let misses_after_warm = warm_stats.misses - after_warm.misses;
    println!(
        "HOTLIST cold: misses={} redos={redos_cold}; warm: loaded={loaded} misses_after={misses_after_warm} redos={redos_warm}",
        cold_stats.misses
    );
    assert!(
        misses_after_warm < cold_stats.misses,
        "the warm pool did not miss less"
    );
    assert!(redos_warm <= redos_cold);
    // Where every layer's working set fits the pool, the warm run is clean.
    let fits = hot.layers.iter().all(|l| l.len() <= slots);
    if fits {
        assert_eq!(
            (misses_after_warm, redos_warm),
            (0, 0),
            "a fitting hotlist still missed"
        );
    }
    std::fs::remove_file(&gguf).ok();
    std::fs::remove_file(&file).ok();
    Ok(())
}
