//! The CPU numerics probe: the same arithmetic on every platform, each result
//! folded to a hash and printed, so two platforms' runs can be compared op by
//! op and the first divergence named (`design/measurements/issue-939.md`).
//! Every input is built from integer arithmetic (no libm in the fixture), so
//! two platforms start from the same bits and differ only where the op does.
//!
//! What is asserted here is run-to-run identity inside one process (§2.3's
//! invariant) and non-vacuity; what differs ACROSS platforms is read from the
//! printed `PROBE` lines, never asserted, because the probe exists to measure
//! it. Run with `--nocapture` and keep the lines beside the machine's
//! `/proc/cpuinfo` (or `sysctl -n machdep.cpu.brand_string`).
use lloom::models::lfm2::{AttnImpl, Config, KvAppend, LayerType, QkvLayout, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::CrownExact;
use lloom::testing::tiny::init::splitmix64;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};

fn fold(bits: impl IntoIterator<Item = u32>) -> u64 {
    bits.into_iter()
        .fold(0x761, |h, b| splitmix64(h ^ b as u64))
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn hash(t: &Tensor) -> Result<u64> {
    Ok(fold(bits(t)?))
}

/// `n` values in [-0.5, 0.5) from integer arithmetic: identical bits on every
/// platform (a small-integer quotient is one correctly rounded division).
fn ramp(n: usize, mul: usize, modulus: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul) % modulus) as f32 / modulus as f32 - 0.5)
        .collect()
}

fn mat(rows: usize, cols: usize, mul: usize, modulus: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(ramp(rows * cols, mul, modulus), (rows, cols), dev)
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// `x[m, k] · w[n, k]ᵀ` in `dtype`: the hash, and the row-count invariance
/// of row 0 against the same row computed alone (`m = 1` takes gemm's gemv
/// route, `m > 1` its tiled route).
fn gemm(out: &mut Vec<String>, dtype: DType, dev: &Device) -> Result<()> {
    let tag = format!("{dtype:?}").to_ascii_lowercase();
    for (k, n) in [
        (32usize, 64usize),
        (64, 128),
        (64, 192),
        (512, 512),
        (512, 1536),
        // The attention shapes of a 23-token pass: q·kᵀ (k = head_dim, n = kv)
        // and softmax·v (k = kv, n = head_dim).
        (32, 23),
        (23, 32),
        (32, 15),
        (15, 32),
    ] {
        let w = mat(n, k, 7919, 1009, dev)?.to_dtype(dtype)?;
        let x_all = mat(70, k, 104_729, 977, dev)?.to_dtype(dtype)?;
        let wt = w.t()?;
        let alone = x_all.narrow(0, 0, 1)?.matmul(&wt)?;
        let alone_bits = bits(&alone)?;
        let alone_f32: Vec<f32> = alone.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        // Row 0 beside one other row: the tiled route's reference, so a
        // difference among `m >= 2` is told apart from the gemv seam.
        let pair_bits: Vec<u32> = bits(&x_all.narrow(0, 0, 2)?.matmul(&wt)?)?[..n].to_vec();
        // The f64 reference over the dtype-rounded inputs: what the f16 arm
        // loses to its accumulator is measured against this, not against f32.
        let wf: Vec<f32> = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for m in [1usize, 2, 5, 8, 15, 23, 24, 70] {
            let x = x_all.narrow(0, 0, m)?;
            let y = x.matmul(&wt)?;
            let h = hash(&y)?;
            let yf: Vec<f32> = y.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            let xf: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let mut err = 0f64;
            let mut scale = 0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += xf[i * k + kk] as f64 * wf[j * k + kk] as f64;
                    }
                    err = err.max((yf[i * n + j] as f64 - acc).abs());
                    scale = scale.max(acc.abs());
                }
            }
            let row0 = &yf[..n];
            let row0_bits: Vec<u32> = row0.iter().map(|v| v.to_bits()).collect();
            let rowinv = if row0_bits == alone_bits {
                "equal".to_string()
            } else {
                format!("differs maxdiff {:e}", max_diff(row0, &alone_f32))
            };
            let pairinv = if row0_bits == pair_bits { "equal" } else { "differs" };
            // Every row of the m-row pass against the same row computed in
            // a 2-row pass: the rows a tile's remainder handles are the ones
            // a row-0 comparison cannot see.
            let mut rows_moved = 0usize;
            for i in 0..m {
                let pair = bits(&x_all.narrow(0, i, 2.min(70 - i))?.matmul(&wt)?)?;
                let y_bits: Vec<u32> = yf[i * n..(i + 1) * n].iter().map(|v| v.to_bits()).collect();
                if y_bits != pair[..n] {
                    rows_moved += 1;
                }
            }
            // Rows 0..2 of every pass, hashed: the passes whose rows agree
            // share the value. And the last eight rows against an 8-row pass
            // of the same rows (a 23-token prefill's tail against an 8-token
            // second turn, `turn_checkpoint.rs`).
            let head2 = if m >= 2 {
                format!("{:#018x}", fold(yf[..2 * n].iter().map(|v| v.to_bits())))
            } else {
                "-".to_string()
            };
            let tail8 = if m >= 8 {
                let alone8 = bits(&x_all.narrow(0, m - 8, 8)?.matmul(&wt)?)?;
                let tail: Vec<u32> = yf[(m - 8) * n..].iter().map(|v| v.to_bits()).collect();
                if tail == alone8 { "equal" } else { "differs" }
            } else {
                "-"
            };
            out.push(format!(
                "PROBE gemm_{tag} m={m} k={k} n={n} hash={h:#018x} err/scale={:e} row0_vs_m1={rowinv} row0_vs_m2={pairinv} rows_vs_pairs_moved={rows_moved} head2={head2} tail8_vs_alone={tail8}",
                err / scale
            ));
        }
    }
    Ok(())
}

/// Element-wise transcendentals (libm) and the reductions behind softmax
/// and rmsnorm (`lloom-tensor/src/cpu/mod.rs`'s SIMD-or-scalar paths).
fn elementwise(out: &mut Vec<String>, dev: &Device) -> Result<()> {
    let v = Tensor::from_vec(
        (0..4096usize)
            .map(|i| ((i * 7919) % 2003) as f32 / 100.0 - 10.0)
            .collect::<Vec<f32>>(),
        (8, 512),
        dev,
    )?;
    out.push(format!("PROBE exp hash={:#018x}", hash(&v.exp()?)?));
    out.push(format!("PROBE silu hash={:#018x}", hash(&v.silu()?)?));
    out.push(format!(
        "PROBE softmax_last_dim hash={:#018x}",
        hash(&lloom::nn::ops::softmax_last_dim(&v)?)?
    ));
    out.push(format!("PROBE sum_last_dim hash={:#018x}", hash(&v.sum_keepdim(1)?)?));
    let alpha = Tensor::from_vec(ramp(512, 31, 97), 512, dev)?;
    out.push(format!(
        "PROBE rms_norm hash={:#018x}",
        hash(&lloom::nn::ops::rms_norm(&v, &alpha, 1e-5)?)?
    ));
    // The depthwise conv the LFM2 short-conv layer runs on the CPU
    // (`vec_dot` per output element).
    let c = 64usize;
    let x = Tensor::from_vec(ramp(c * 24, 7919, 1009), (1, c, 24), dev)?;
    let kernel = Tensor::from_vec(ramp(c * 3, 31, 97), (c, 1, 3), dev)?;
    out.push(format!(
        "PROBE conv1d_depthwise hash={:#018x}",
        hash(&x.conv1d(&kernel, 2, 1, 1, c)?)?
    ));
    let x16 = x.to_dtype(DType::F16)?;
    let k16 = kernel.to_dtype(DType::F16)?;
    out.push(format!(
        "PROBE conv1d_depthwise_f16 hash={:#018x}",
        hash(&x16.conv1d(&k16, 2, 1, 1, c)?)?
    ));
    Ok(())
}

/// `lfm2_qkv_layout.rs`'s tiny config (its CPU fingerprint is pinned on the
/// M1), truncated to its first `layers` layers.
fn fingerprint_config(layers: usize) -> Config {
    let all = [
        LayerType::Conv,
        LayerType::FullAttention,
        LayerType::FullAttention,
    ];
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: layers,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: all[..layers].to_vec(),
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: AttnImpl::Generic,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state: Default::default(),
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        qkv_layout: QkvLayout::Contiguous,
        ..Default::default()
    }
}

fn last_row(logits: &Tensor) -> Result<Tensor> {
    if logits.rank() == 3 {
        let s = logits.dim(1)?;
        logits.narrow(1, s - 1, 1)
    } else {
        Ok(logits.clone())
    }
}

/// The fingerprint test's model, layer by layer: the prefill logits and the
/// conv state after it, at one, two and three layers (the weights are keyed
/// by name and index, so a prefix of the model is the model's prefix).
fn lfm2_by_layer(out: &mut Vec<String>, dev: &Device) -> Result<()> {
    for layers in 1..=3 {
        let cfg = fingerprint_config(layers);
        let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(DType::F32)
            .model::<CrownExact>(dev)?;
        let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, dev)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
        let logits = model.forward(&inp, 0, &mut cache)?;
        let conv = cache.conv_state_for_test(0).expect("layer 0 is a conv layer");
        out.push(format!(
            "PROBE lfm2_fingerprint layers={layers} prefill_logits={:#018x} conv0_state={:#018x}",
            hash(&logits)?,
            hash(&conv)?
        ));
        let mut steps = Vec::new();
        for (i, tok) in [7u32, 9u32].into_iter().enumerate() {
            let step = Tensor::new(&[tok], dev)?.reshape((1, 1))?;
            steps.push(hash(&model.forward(&step, prompt.len() + i, &mut cache)?)?);
        }
        out.push(format!(
            "PROBE lfm2_fingerprint layers={layers} decode_logits={steps:#018x?}"
        ));
    }
    Ok(())
}

/// `turn_checkpoint.rs`'s geometry: the same 23 tokens as one pass, as two
/// passes, and as a prefill plus a decode step (the left context carried,
/// `conv_prefill_ctx`), compared bit for bit on the last position's logits
/// and on every conv layer's state. The `Exact` regime promises the result
/// does not depend on how many rows a pass carries.
fn lfm2_row_count(out: &mut Vec<String>, dev: &Device) -> Result<()> {
    let cfg = Config {
        vocab_size: 48,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 512,
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
        eos_token_id: Some(7),
        use_flash_attn: false,
        attn_impl: AttnImpl::Generic,
        kv_append: KvAppend::InPlace,
        conv_prefill_ctx: true,
        gemv_silu: false,
        gemv_residual: false,
        tap_dtype: DType::F32,
        ..Default::default()
    };
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F32)
        .model::<CrownExact>(dev)?;
    let prompt: Vec<u32> = (0..23u32).map(|i| (i * 5 + 3) % 46).collect();
    let run = |splits: &[usize]| -> Result<(Vec<u32>, Vec<Option<Vec<f32>>>)> {
        let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, dev)?;
        let mut pos = 0;
        let mut last = None;
        for &w in splits {
            let inp = Tensor::new(&prompt[pos..pos + w], dev)?.reshape((1, w))?;
            last = Some(model.forward(&inp, pos, &mut cache)?);
            pos += w;
        }
        let logits = bits(&last_row(&last.expect("a pass"))?)?;
        let states = (0..cfg.num_hidden_layers)
            .map(|l| {
                cache
                    .conv_state_for_test(l)
                    .map(|t| t.flatten_all()?.to_vec1::<f32>())
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((logits, states))
    };
    let (one_logits, one_states) = run(&[23])?;
    out.push(format!(
        "PROBE lfm2_rows passes=[23] last_logits={:#018x}",
        fold(one_logits.iter().copied())
    ));
    for splits in [vec![15, 8], vec![22, 1], vec![1; 23], vec![20, 1, 1, 1]] {
        let (logits, states) = run(&splits)?;
        let lg = if logits == one_logits {
            "equal".to_string()
        } else {
            let a: Vec<f32> = logits.iter().map(|b| f32::from_bits(*b)).collect();
            let b: Vec<f32> = one_logits.iter().map(|b| f32::from_bits(*b)).collect();
            format!(
                "differs maxdiff {:e} scale {:e}",
                max_diff(&a, &b),
                max_abs(&b)
            )
        };
        let mut conv = Vec::new();
        for (l, (a, b)) in states.iter().zip(&one_states).enumerate() {
            if let (Some(a), Some(b)) = (a, b) {
                let d = max_diff(a, b);
                conv.push(format!("L{l}:{}", if d == 0.0 { "equal".into() } else { format!("{d:e}") }));
            }
        }
        out.push(format!(
            "PROBE lfm2_rows passes={splits:?} last_logits={} conv_state=[{}]",
            lg,
            conv.join(" ")
        ));
    }
    Ok(())
}

fn run_all(dev: &Device) -> Result<Vec<String>> {
    let mut out = Vec::new();
    gemm(&mut out, DType::F32, dev)?;
    gemm(&mut out, DType::F16, dev)?;
    elementwise(&mut out, dev)?;
    lfm2_by_layer(&mut out, dev)?;
    lfm2_row_count(&mut out, dev)?;
    Ok(out)
}

/// Every probe reproduces itself within the process, and none is vacuous.
#[test]
fn the_cpu_probes_reproduce_within_a_process() -> Result<()> {
    let dev = Device::Cpu;
    let first = run_all(&dev)?;
    let second = run_all(&dev)?;
    println!(
        "PROBE platform arch={} os={} threads={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        lloom_tensor::utils::get_num_threads()
    );
    for line in &first {
        println!("{line}");
    }
    assert!(first.len() > 40, "non-vacuous: {} probe lines", first.len());
    assert!(
        first.iter().any(|l| l.contains("row0_vs_m1=")),
        "the row-count probes ran"
    );
    assert_eq!(first, second, "a probe did not reproduce within the process");
    Ok(())
}

/// The long run: the fingerprint model, widened to six layers and 1024
/// positions, decoded greedily over three turns of 200 tokens on one cache,
/// every position's logits folded to a hash and, when `LLOOM_PROBE_DUMP`
/// names a file, written raw (f32, position-major) so two platforms' streams
/// can be diffed position by position (`issue-939-raw/longrun/drift.py`).
fn long_run(out: &mut Vec<String>, dev: &Device) -> Result<Vec<f32>> {
    let mut cfg = fingerprint_config(3);
    cfg.num_hidden_layers = 6;
    cfg.layer_types = vec![
        LayerType::Conv,
        LayerType::FullAttention,
        LayerType::Conv,
        LayerType::FullAttention,
        LayerType::Conv,
        LayerType::FullAttention,
    ];
    cfg.max_position_embeddings = 1024;
    cfg.conv_prefill_ctx = true;
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F32)
        .model::<CrownExact>(dev)?;
    let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, dev)?;
    let mut pos = 0usize;
    let mut dump: Vec<f32> = Vec::new();
    let turns: [Vec<u32>; 3] = [
        (0..16u32).map(|i| (i * 7 + 3) % 64).collect(),
        (0..24u32).map(|i| (i * 11 + 5) % 64).collect(),
        (0..12u32).map(|i| (i * 13 + 1) % 64).collect(),
    ];
    let steps = 200usize;
    for (turn, prompt) in turns.iter().enumerate() {
        let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
        let mut logits = last_row(&model.forward(&inp, pos, &mut cache)?)?.flatten_all()?;
        pos += prompt.len();
        let mut toks = Vec::with_capacity(steps);
        let mut hashes = Vec::with_capacity(steps);
        for _ in 0..steps {
            let row: Vec<f32> = logits.to_vec1()?;
            let next = row
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
                .0 as u32;
            hashes.push(fold(row.iter().map(|v| v.to_bits())));
            dump.extend_from_slice(&row);
            toks.push(next);
            let step = Tensor::new(&[next], dev)?.reshape((1, 1))?;
            logits = last_row(&model.forward(&step, pos, &mut cache)?)?.flatten_all()?;
            pos += 1;
        }
        out.push(format!(
            "PROBE longrun turn={turn} prompt={} end_pos={pos} tokens={toks:?}",
            prompt.len()
        ));
        out.push(format!(
            "PROBE longrun turn={turn} logits_hash_every_20={:#018x?}",
            hashes.iter().step_by(20).collect::<Vec<_>>()
        ));
    }
    Ok(dump)
}

/// The long run reproduces within the process; its stream is dumped when
/// asked for.
#[test]
fn the_long_run_reproduces_within_a_process() -> Result<()> {
    let dev = Device::Cpu;
    let mut first = Vec::new();
    let dump = long_run(&mut first, &dev)?;
    let mut second = Vec::new();
    let dump2 = long_run(&mut second, &dev)?;
    for line in &first {
        println!("{line}");
    }
    assert_eq!(first, second, "the long run did not reproduce within the process");
    assert_eq!(dump.len(), 3 * 200 * 64, "600 positions of 64 logits");
    assert!(
        dump.iter().map(|v| v.to_bits()).eq(dump2.iter().map(|v| v.to_bits())),
        "the dumped logits did not reproduce"
    );
    if let Ok(path) = std::env::var("LLOOM_PROBE_DUMP") {
        let bytes: Vec<u8> = dump.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("write the dump");
        println!("PROBE longrun dump={path} floats={}", dump.len());
    }
    Ok(())
}
