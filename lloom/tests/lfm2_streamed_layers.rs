//! The dense MLPs on the weight-group tier (`profiles-and-memory-design.md`
//! §4.8, #892 A6a): a tiny LFM2 built twice from the same checkpoint file —
//! every weight resident, and the MLPs streamed through three slots (one in
//! use, one the previous stage still reads, one loading) — decodes the
//! same bits. Streaming moves bytes,
//! not math. A third build with a slot per layer (every group resident, no
//! load after the first pass) is the full-quota arm: identical as well.
//!
//! On Metal when a device is there, else on the CPU (the tier copies with
//! `slice_set` there; on Metal the host writes the Shared slots).

use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State, StreamLayers};
use lloom::nn::VarBuilder;
use lloom::profile::RewindableExact;
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

fn tiny_config() -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: 2 },
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// The checkpoint, written once: every tensor a value that names it.
fn write_checkpoint(cfg: &Config, dtype: DType, path: &std::path::Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    let mut seed = 0u32;
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        seed += 1;
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| (((i as u32 * 7 + seed * 13) % 17) as f32 - 8.0) / 32.0)
            .collect();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(v, shape, &dev)?.to_dtype(dtype)?,
        );
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    lloom_tensor::safetensors::save(&tensors, path)
}

fn build(
    cfg: &Config,
    dtype: DType,
    dev: &Device,
    path: &std::path::Path,
) -> Result<Model<RewindableExact>> {
    // SAFETY: the file is written above and not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, dev)? };
    Model::<RewindableExact>::new(cfg, vb)
}

const PROMPT: [u32; 5] = [3, 11, 20, 5, 9];

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// A prefill and `steps` greedy decode steps: the logits of each.
fn run(
    model: &Model<RewindableExact>,
    cfg: &Config,
    dtype: DType,
    dev: &Device,
    steps: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut cache = State::<RewindableExact>::new(true, dtype, cfg, dev)?.with_kv_bucket(2)?;
    let input = Tensor::new(&PROMPT, dev)?.unsqueeze(0)?;
    let mut out = Vec::new();
    let first = model
        .forward(&input, 0, &mut cache)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut tok = argmax(&first);
    out.push(first);
    let mut pos = PROMPT.len();
    // `LLOOM_TEST_GPU_BUSY=1`: a slow kernel ahead of every step's layers,
    // so the GPU reaches a stage's slot long after the host asked for the
    // load that evicts it — the race a loader that does not wait for the
    // slot's reader loses deterministically (the harness's M-F).
    let busy = std::env::var("LLOOM_TEST_GPU_BUSY").is_ok();
    let heavy = if busy && dev.is_metal() {
        Some(Tensor::ones((2048, 2048), DType::F32, dev)?)
    } else {
        None
    };
    for _ in 0..steps {
        if let Some(h) = &heavy {
            let mut acc = h.clone();
            for _ in 0..6 {
                acc = acc.matmul(h)?.affine(1e-3, 0.0)?;
            }
            // Keep it on the device, ordered before the layers: a readback
            // here would make the host wait and hide the race.
            std::hint::black_box(acc);
        }
        let input = Tensor::new(&[tok], dev)?.unsqueeze(0)?;
        let logits = model
            .forward(&input, pos, &mut cache)?
            .i(0)?
            .to_vec1::<f32>()?;
        tok = argmax(&logits);
        pos += 1;
        out.push(logits);
    }
    Ok(out)
}

#[test]
fn the_streamed_layers_decode_the_same_bits_as_the_resident_ones() -> Result<()> {
    // `LLOOM_TEST_CPU=1` runs the same test on the CPU (the tier copies
    // with `slice_set` there): a Metal-only difference is a kernel's.
    let dev = if std::env::var("LLOOM_TEST_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    };
    let dtype = DType::F32;
    let cfg = tiny_config();
    let dir = std::env::temp_dir().join(format!("lfm2-streamed-mlp-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("model.safetensors");
    write_checkpoint(&cfg, dtype, &path)?;

    let resident = build(&cfg, dtype, &dev, &path)?;
    let want = run(&resident, &cfg, dtype, &dev, 6)?;
    drop(resident);

    // `LLOOM_TEST_SLOTS=n` runs one slot count (a diagnosis knob).
    let counts: Vec<usize> = match std::env::var("LLOOM_TEST_SLOTS") {
        Ok(n) => vec![n.parse().unwrap()],
        Err(_) => vec![3, 4, cfg.num_hidden_layers],
    };
    for slots in counts {
        let mut streamed_cfg = cfg.clone();
        streamed_cfg.stream_layers = Some(StreamLayers {
            files: vec![path.clone()],
            slots,
            depth: 1,
        });
        let t0 = std::time::Instant::now();
        let streamed = build(&streamed_cfg, dtype, &dev, &path)?;
        let t1 = std::time::Instant::now();
        let got = run(&streamed, &streamed_cfg, dtype, &dev, 6)?;
        eprintln!(
            "[test] slots {slots}: build {:.0} ms, run {:.0} ms",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(got.len(), want.len());
        for (step, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                let audit = streamed.streamed_layers_audit()?;
                panic!(
                    "slots {slots}: the streamed model's logits differ at step {step}; audit {audit:?}"
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
