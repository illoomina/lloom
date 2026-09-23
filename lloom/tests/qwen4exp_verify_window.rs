//! The verify window as one pass (`Qwen4Exp::verify_window`) against the
//! sequential decode: the same argmax at every position, and after
//! `accept_window(a)` the cache — every recurrent channel, every KV, the
//! position — equals a cache stepped `a` times sequentially (checksums),
//! for every `a` in `0..=T`. With the PLE attached (its history in a ring
//! too). CPU (the composed per-position path) and Metal (the multi-step
//! kernels).
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    Cache, Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident,
};
use lloom::nn::model::CausalLM;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lloom-verify-window-{}-{tag}.gguf",
        std::process::id()
    ))
}

fn write_table(cfg: &Config, ple_rows: usize, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let table = testing::wt((ple_rows, cfg.ple_head_dim), 3.0, &dev);
    let q = QTensor::quantize(&table, GgmlDType::F32)?;
    let refs: Vec<(&str, &QTensor)> = vec![("per_layer_token_embd.weight", &q)];
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    let _ = Content::read(&mut File::open(path)?)?;
    Ok(())
}

fn model_with_ple(cfg: &Config, dev: &Device, path: &Path) -> Result<Qwen4Exp<Resident>> {
    let mut model = testing::resident_model(cfg, dev);
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
    let block = PleLayer::new(
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
    model.set_ple(1, hash, PleTable::Mmap(table), block)?;
    Ok(model)
}

fn step_argmax(
    model: &Qwen4Exp<Resident>,
    tok: u32,
    cache: &mut Cache,
    dev: &Device,
) -> Result<u32> {
    let input = Tensor::from_vec(vec![tok], (1, 1), dev)?;
    let logits = model.forward(&input, cache.pos, cache)?;
    let v: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    Ok(v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |m, (i, &x)| if x > m.1 { (i, x) } else { m },
        )
        .0 as u32)
}

fn check_on(dev: &Device, tag: &str, with_ple: bool) -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    if dev.is_metal() {
        cfg.head_dim = 32;
        // The rows pass is bit-identical on the f16 stems' row gemvs (the
        // checkpoint's kind); f32 `Tensor` stems go through the gemm at
        // `m > 1` and would not be.
        testing::set_f16_stems(true);
    }
    let path = temp_gguf(tag);
    let model = if with_ple {
        write_table(&cfg, 80, &path)?;
        model_with_ple(&cfg, dev, &path)?
    } else {
        testing::resident_model(&cfg, dev)
    };
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let window: Vec<u32> = vec![4, 11, 7, 2];
    let t = window.len();
    let k = t - 1;
    let prefill = |cache: &mut Cache| -> Result<()> {
        let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev)?;
        model.forward(&input, 0, cache)?;
        // one decode step so every slab is written by the decode path
        step_argmax(&model, 4, cache, dev)?;
        Ok(())
    };
    // The sequential reference: argmaxes and the state after every a.
    let mut seq_argmax = Vec::with_capacity(t);
    let mut seq_sums: Vec<Vec<(String, u64, usize)>> = Vec::with_capacity(t + 1);
    let mut seq_pos = Vec::with_capacity(t + 1);
    {
        let mut c = model.new_cache(&cfg, dev)?;
        c.want_nextn = true;
        prefill(&mut c)?;
        let _ = c.take_nextn()?;
        seq_sums.push(c.checksums()?);
        seq_pos.push(c.pos);
        for &tok in &window {
            seq_argmax.push(step_argmax(&model, tok, &mut c, dev)?);
            seq_sums.push(c.checksums()?);
            seq_pos.push(c.pos);
        }
    }
    // The window pass, then an accept of every a — each from a fresh run.
    for a in 0..=t {
        let pools = model.spec_pools(1, k, dev);
        let mut c = Cache::in_pools(&cfg, &pools)?;
        c.want_nextn = true;
        assert!(
            c.ring_holds(t, with_ple),
            "{tag}: the ring does not hold {t}"
        );
        prefill(&mut c)?;
        let _ = c.take_nextn()?;
        assert_eq!(
            c.checksums()?,
            seq_sums[0],
            "{tag}: the pooled cache after the prefill"
        );
        let argmax = model.verify_window(&window, &mut c)?;
        assert_eq!(argmax, seq_argmax, "{tag}: the window's argmaxes (a = {a})");
        c.accept_window(a)?;
        assert_eq!(c.pos, seq_pos[a], "{tag}: the position after accepting {a}");
        assert_eq!(
            c.checksums()?,
            seq_sums[a],
            "{tag}: the state after accepting {a}"
        );
        let probe = c.take_nextn()?;
        assert_eq!(
            probe.map(|p| p.dim(1).unwrap()).unwrap_or(0),
            a,
            "{tag}: the retained carries"
        );
        // And the cache decodes on from there as the sequential one does.
        let next_seq = {
            let mut s = model.new_cache(&cfg, dev)?;
            prefill(&mut s)?;
            for &tok in &window[..a] {
                step_argmax(&model, tok, &mut s, dev)?;
            }
            step_argmax(&model, 6, &mut s, dev)?
        };
        assert_eq!(
            step_argmax(&model, 6, &mut c, dev)?,
            next_seq,
            "{tag}: the step after accepting {a}"
        );
    }
    std::fs::remove_file(&path).ok();
    testing::set_f16_stems(false);
    println!("VERIFYWINDOW {tag} ok (T={t}, argmax={seq_argmax:?})");
    Ok(())
}

#[test]
fn verify_window_is_the_sequential_decode_cpu() -> Result<()> {
    check_on(&Device::Cpu, "cpu", false)?;
    check_on(&Device::Cpu, "cpu-ple", true)
}

#[cfg(feature = "metal")]
#[test]
fn verify_window_is_the_sequential_decode_metal() -> Result<()> {
    match Device::new_metal(0) {
        Ok(dev) => {
            check_on(&dev, "metal", false)?;
            check_on(&dev, "metal-ple", true)
        }
        Err(_) => Ok(()),
    }
}
