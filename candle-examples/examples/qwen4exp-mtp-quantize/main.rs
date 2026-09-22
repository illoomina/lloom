//! The Qwen3.8-Flash-Next MTP module, quantized in-tree into a sidecar GGUF
//! (issue #749): no mainline-lineage GGUF carries the `mtp.*` tensors (the
//! converter has no NEXTN mapping for qwen4exp), so they are taken from the HF
//! bf16 checkpoint — the 31 tensors range-fetched into a raw store — and
//! written in ds4's naming (`blk.{n_layers}.*` for the layer body,
//! `blk.{n_layers}.nextn.*` for the MTP-specific pieces,
//! `qwen4exp.nextn_predict_layers`) so one loader reads either file.
//!
//! `check` grounds the HF → GGUF transforms on a MAIN layer of the checkpoint
//! we already load: the same raw store for layer 3 (`experts`: expert 0 only)
//! is transformed and compared, tensor by tensor, against the GGUF's `blk.3.*`
//! dequantized — the norm folding, the `gate_up_proj` split, the
//! `index_qk_proj` split and the `attn_q` layout are read off the artifact,
//! not assumed.
//!
//! ```bash
//! cargo run --release --example qwen4exp-mtp-quantize -- check --layer 3
//! cargo run --release --example qwen4exp-mtp-quantize -- write --out ~/.cache/lloom/qwen4exp-mtp-q4k.gguf
//! ```

use anyhow::{Context, Result};
use candle::quantized::gguf_file::{self, Content, Value};
use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Tensor, D};
use clap::{Parser, Subcommand};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Qwen3.8-Flash-Next MTP: quantize the HF bf16 module into a sidecar GGUF")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Compare the transformed HF tensors of a main layer against the GGUF's.
    Check {
        #[arg(long, default_value_t = 3)]
        layer: usize,
        /// The raw store of that layer (`fetch.py`'s manifest + .bin files).
        #[arg(long)]
        store: Option<PathBuf>,
        /// The first GGUF shard of the checkpoint the layer is compared against.
        #[arg(long)]
        gguf: Option<PathBuf>,
    },
    /// Quantize the `mtp.*` raw store into the sidecar.
    Write {
        #[arg(long)]
        store: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// The main model's layer count (the MTP layer becomes `blk.{n}`).
        #[arg(long, default_value_t = 48)]
        n_layers: usize,
        /// Quant for the routed gate/up stacks (`q4_k` | `q8_0` | `q6_k`).
        #[arg(long, default_value = "q4_k")]
        experts: String,
    },
}

#[derive(serde::Deserialize, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    file: String,
}

struct Store {
    dir: PathBuf,
    entries: HashMap<String, Entry>,
}

impl Store {
    fn open(dir: PathBuf) -> Result<Self> {
        let m = std::fs::read_to_string(dir.join("manifest.json"))
            .with_context(|| format!("manifest in {}", dir.display()))?;
        let entries: HashMap<String, Entry> = serde_json::from_str(&m)?;
        Ok(Self { dir, entries })
    }
    /// A tensor by its HF name suffix (the store's keys carry the checkpoint's
    /// prefix), as f32 on the CPU.
    fn f32(&self, suffix: &str) -> Result<Tensor> {
        let (name, e) = self
            .entries
            .iter()
            .find(|(k, _)| k.ends_with(suffix) || *k == suffix)
            .with_context(|| format!("store has no tensor ending in `{suffix}`"))?;
        if e.dtype != "BF16" {
            anyhow::bail!("{name}: dtype {} (expected BF16)", e.dtype);
        }
        let bytes = std::fs::read(self.dir.join(&e.file))?;
        let t = Tensor::from_raw_buffer(&bytes, DType::BF16, &e.shape, &Device::Cpu)?;
        Ok(t.to_dtype(DType::F32)?)
    }
}

fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(base)
        .ok()?
        .flatten()
        .next()
        .map(|e| e.path())
}

fn home(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(rel)
}

/// Gemma-style norm weights are stored folded: `γ_gguf = 1 + w_hf`.
fn fold(w: &Tensor) -> Result<Tensor> {
    Ok((w + 1.0)?)
}

/// Max |a − b| / max |b| and the plain max |a − b|.
fn err(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(a.len() == b.len(), "length {} vs {}", a.len(), b.len());
    let mut md = 0f32;
    let mut mb = 0f32;
    for (x, y) in a.iter().zip(&b) {
        md = md.max((x - y).abs());
        mb = mb.max(y.abs());
    }
    Ok((md / mb.max(1e-12), md))
}

fn check(layer: usize, store: PathBuf, gguf: PathBuf) -> Result<()> {
    let st = Store::open(store)?;
    let dev = Device::Cpu;
    let mut f = std::fs::File::open(&gguf)?;
    let content = Content::read(&mut f)?;
    let p = format!("blk.{layer}");
    let g = |name: &str| -> Result<Tensor> {
        let qt = content.tensor(
            &mut std::fs::File::open(&gguf)?,
            &format!("{p}.{name}"),
            &dev,
        )?;
        Ok(qt.dequantize(&dev)?)
    };
    let gq = |name: &str| -> Result<GgmlDType> {
        Ok(content
            .tensor_infos
            .get(&format!("{p}.{name}"))
            .context(name.to_string())?
            .ggml_dtype)
    };
    let mut report = |what: &str, gguf_name: &str, candidates: Vec<(&str, Tensor)>| -> Result<()> {
        let want = g(gguf_name)?;
        let ty = gq(gguf_name)?;
        let mut line = format!("{what:36} {gguf_name:24} {ty:?}");
        for (label, cand) in candidates {
            let (rel, abs) = err(&cand, &want)?;
            line.push_str(&format!("  [{label}: rel {rel:.3e} abs {abs:.3e}]"));
        }
        println!("{line}");
        Ok(())
    };

    // norms: folded vs raw
    for (hf, gg) in [
        (
            "attn_hyper_connection.hc_norm.weight",
            "hc_attn_norm.weight",
        ),
        ("mlp_hyper_connection.hc_norm.weight", "hc_ffn_norm.weight"),
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        (
            "self_attn.indexer.q_layernorm.weight",
            "indexer.q_norm.weight",
        ),
        (
            "self_attn.indexer.k_layernorm.weight",
            "indexer.k_norm.weight",
        ),
    ] {
        let w = st.f32(hf)?;
        report(hf, gg, vec![("1+w", fold(&w)?), ("w", w.clone())])?;
    }
    // HC low-rank + inject: identity
    for (hf, gg) in [
        (
            "attn_hyper_connection.input_mix_weight_down.weight",
            "hc_attn_down.weight",
        ),
        (
            "attn_hyper_connection.input_mix_weight_up.weight",
            "hc_attn_up.weight",
        ),
        (
            "attn_hyper_connection.block_inject_weight.weight",
            "hc_attn_inject.weight",
        ),
        (
            "mlp_hyper_connection.input_mix_weight_down.weight",
            "hc_ffn_down.weight",
        ),
        (
            "mlp_hyper_connection.input_mix_weight_up.weight",
            "hc_ffn_up.weight",
        ),
        (
            "mlp_hyper_connection.block_inject_weight.weight",
            "hc_ffn_inject.weight",
        ),
        ("mlp.gate.weight", "ffn_gate_inp.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        (
            "mlp.shared_expert.gate_proj.weight",
            "ffn_gate_shexp.weight",
        ),
        ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
        (
            "mlp.shared_expert.down_proj.weight",
            "ffn_down_shexp.weight",
        ),
    ] {
        let w = st.f32(hf)?;
        report(hf, gg, vec![("identity", w)])?;
    }
    // shared-expert gate: [1, 2560] → [2560]
    let w = st.f32("mlp.shared_expert_gate.weight")?.flatten_all()?;
    report(
        "mlp.shared_expert_gate.weight",
        "ffn_gate_inp_shexp.weight",
        vec![("flatten", w)],
    )?;
    // attn_q: identity vs per-head [q|gate] de-interleave alternatives
    let q = st.f32("self_attn.q_proj.weight")?; // [12288, 2560]
    let (rows, cols) = q.dims2()?;
    let alt = q
        .reshape((2, rows / 2, cols))? // [2, 6144, 2560]: all q then all gate?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((rows, cols))?;
    let alt2 = q
        .reshape((24, 2, 256, cols))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, cols))?;
    report(
        "self_attn.q_proj.weight",
        "attn_q.weight",
        vec![
            ("identity", q.clone()),
            ("halves-interleaved", alt),
            ("head-pairs-swapped", alt2),
        ],
    )?;
    // indexer split
    let qk = st.f32("self_attn.indexer.index_qk_proj.weight")?; // [640, 2560]
    let q_first = qk.narrow(0, 0, 512)?;
    let k_last = qk.narrow(0, 512, 128)?;
    let k_first = qk.narrow(0, 0, 128)?;
    let q_last = qk.narrow(0, 128, 512)?;
    report(
        "indexer.index_qk_proj (q part)",
        "indexer.q_proj.weight",
        vec![("rows 0..512", q_first), ("rows 128..640", q_last)],
    )?;
    report(
        "indexer.index_qk_proj (k part)",
        "indexer.k_proj.weight",
        vec![("rows 512..640", k_last), ("rows 0..128", k_first)],
    )?;
    // experts (expert 0): gate_up split order, down identity
    let gu = st.f32("mlp.experts.gate_up_proj")?; // [1, 1280, 2560]
    let first = gu.narrow(1, 0, 640)?.squeeze(0)?;
    let second = gu.narrow(1, 640, 640)?.squeeze(0)?;
    let gate0 = g("ffn_gate_exps.weight")?.get(0)?;
    let up0 = g("ffn_up_exps.weight")?.get(0)?;
    let down0 = g("ffn_down_exps.weight")?.get(0)?;
    let dn = st.f32("mlp.experts.down_proj")?.squeeze(0)?;
    for (what, want, cands) in [
        (
            "gate_up_proj → ffn_gate_exps[0]",
            &gate0,
            vec![("rows 0..640", &first), ("rows 640..1280", &second)],
        ),
        (
            "gate_up_proj → ffn_up_exps[0]",
            &up0,
            vec![("rows 0..640", &first), ("rows 640..1280", &second)],
        ),
        (
            "down_proj → ffn_down_exps[0]",
            &down0,
            vec![("identity", &dn)],
        ),
    ] {
        let mut line = format!("{what:36}");
        for (label, c) in cands {
            let (rel, abs) = err(c, want)?;
            line.push_str(&format!("  [{label}: rel {rel:.3e} abs {abs:.3e}]"));
        }
        println!("{line}");
    }
    println!("(quantized targets: a rel error near the quant's own error on ONE candidate and ~1 on the other is the discriminating signal)");
    Ok(())
}

/// Quantize `t` (f32, CPU) to `dtype`, expert by expert for a stacked
/// `[e, n, k]`, so the f32 expansion never holds more than one expert.
fn quantize_stacked(t: &Tensor, dtype: GgmlDType) -> Result<QTensor> {
    let (e, n, k) = t.dims3()?;
    let mut bytes: Vec<u8> =
        Vec::with_capacity(e * n * (k / dtype.block_size()) * dtype.type_size());
    for i in 0..e {
        let q = QTensor::quantize(&t.get(i)?.contiguous()?, dtype)?;
        bytes.extend_from_slice(&q.data()?);
    }
    let storage = QStorage::from_data(Cow::Owned(bytes), &Device::Cpu, dtype)?;
    Ok(QTensor::new(storage, (e, n, k))?)
}

fn write(store: PathBuf, out: PathBuf, n_layers: usize, experts: String) -> Result<()> {
    let st = Store::open(store)?;
    let expert_q = match experts.as_str() {
        "q4_k" => GgmlDType::Q4K,
        "q6_k" => GgmlDType::Q6K,
        "q8_0" => GgmlDType::Q8_0,
        other => anyhow::bail!("--experts {other}: q4_k | q6_k | q8_0"),
    };
    let p = format!("blk.{n_layers}");
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    let mut push = |name: String, t: Tensor, dtype: GgmlDType| -> Result<()> {
        let q = QTensor::quantize(&t.contiguous()?, dtype)?;
        println!("  {name:44} {dtype:?} {:?}", t.dims());
        owned.push((name, q));
        Ok(())
    };
    let (f32_, f16_, q8) = (GgmlDType::F32, GgmlDType::F16, GgmlDType::Q8_0);
    let name = |s: &str| format!("{p}.{s}");

    // --- the MTP-specific pieces (ds4's nextn.* names)
    let fc_e = st.f32("mtp.fc_embedding.weight")?; // [2560, 2560]
    let fc_h = st.f32("mtp.fc_hidden.weight")?;
    let eh = Tensor::cat(&[&fc_e, &fc_h], D::Minus1)?; // [2560 out, 5120 in] = [fc_e | fc_h]
    push(name("nextn.eh_proj.weight"), eh, q8)?;
    push(
        name("nextn.enorm.weight"),
        fold(&st.f32("mtp.pre_fc_norm_embedding.weight")?)?,
        f32_,
    )?;
    push(
        name("nextn.hnorm.weight"),
        fold(&st.f32("mtp.pre_fc_norm_hidden.weight")?)?,
        f32_,
    )?;
    push(
        name("nextn.hc_head_norm.weight"),
        fold(&st.f32("mtp.hyper_connection_mixer.hc_norm.weight")?)?,
        f32_,
    )?;
    push(
        name("nextn.hc_head_down.weight"),
        st.f32("mtp.hyper_connection_mixer.input_mix_weight_down.weight")?,
        f16_,
    )?;
    push(
        name("nextn.hc_head_up.weight"),
        st.f32("mtp.hyper_connection_mixer.input_mix_weight_up.weight")?,
        f16_,
    )?;

    // --- the layer body, in the main layers' names
    for (hc, gg) in [
        ("attn_hyper_connection", "hc_attn"),
        ("mlp_hyper_connection", "hc_ffn"),
    ] {
        push(
            name(&format!("{gg}_norm.weight")),
            fold(&st.f32(&format!("mtp.layers.0.{hc}.hc_norm.weight"))?)?,
            f32_,
        )?;
        push(
            name(&format!("{gg}_down.weight")),
            st.f32(&format!("mtp.layers.0.{hc}.input_mix_weight_down.weight"))?,
            f16_,
        )?;
        push(
            name(&format!("{gg}_up.weight")),
            st.f32(&format!("mtp.layers.0.{hc}.input_mix_weight_up.weight"))?,
            f16_,
        )?;
        push(
            name(&format!("{gg}_inject.weight")),
            st.f32(&format!("mtp.layers.0.{hc}.block_inject_weight.weight"))?,
            f16_,
        )?;
    }
    push(
        name("attn_q.weight"),
        st.f32("mtp.layers.0.self_attn.q_proj.weight")?,
        q8,
    )?;
    push(
        name("attn_k.weight"),
        st.f32("mtp.layers.0.self_attn.k_proj.weight")?,
        q8,
    )?;
    push(
        name("attn_v.weight"),
        st.f32("mtp.layers.0.self_attn.v_proj.weight")?,
        q8,
    )?;
    push(
        name("attn_output.weight"),
        st.f32("mtp.layers.0.self_attn.o_proj.weight")?,
        q8,
    )?;
    push(
        name("attn_q_norm.weight"),
        fold(&st.f32("mtp.layers.0.self_attn.q_norm.weight")?)?,
        f32_,
    )?;
    push(
        name("attn_k_norm.weight"),
        fold(&st.f32("mtp.layers.0.self_attn.k_norm.weight")?)?,
        f32_,
    )?;
    let qk = st.f32("mtp.layers.0.self_attn.indexer.index_qk_proj.weight")?; // [640, 2560]
    push(name("indexer.q_proj.weight"), qk.narrow(0, 0, 512)?, q8)?;
    push(name("indexer.k_proj.weight"), qk.narrow(0, 512, 128)?, q8)?;
    push(
        name("indexer.q_norm.weight"),
        fold(&st.f32("mtp.layers.0.self_attn.indexer.q_layernorm.weight")?)?,
        f32_,
    )?;
    push(
        name("indexer.k_norm.weight"),
        fold(&st.f32("mtp.layers.0.self_attn.indexer.k_layernorm.weight")?)?,
        f32_,
    )?;
    push(
        name("ffn_gate_inp.weight"),
        st.f32("mtp.layers.0.mlp.gate.weight")?,
        f32_,
    )?;
    push(
        name("ffn_gate_shexp.weight"),
        st.f32("mtp.layers.0.mlp.shared_expert.gate_proj.weight")?,
        q8,
    )?;
    push(
        name("ffn_up_shexp.weight"),
        st.f32("mtp.layers.0.mlp.shared_expert.up_proj.weight")?,
        q8,
    )?;
    push(
        name("ffn_down_shexp.weight"),
        st.f32("mtp.layers.0.mlp.shared_expert.down_proj.weight")?,
        q8,
    )?;
    push(
        name("ffn_gate_inp_shexp.weight"),
        st.f32("mtp.layers.0.mlp.shared_expert_gate.weight")?
            .flatten_all()?,
        f32_,
    )?;

    // --- the routed experts: split gate_up, quantize per expert
    let gu = st.f32("mtp.layers.0.mlp.experts.gate_up_proj")?; // [512, 1280, 2560]
    let gate = gu.narrow(1, 0, 640)?.contiguous()?;
    let up = gu.narrow(1, 640, 640)?.contiguous()?;
    drop(gu);
    println!(
        "  {:44} {expert_q:?} {:?} (per expert)",
        name("ffn_gate_exps.weight"),
        gate.dims()
    );
    owned.push((
        name("ffn_gate_exps.weight"),
        quantize_stacked(&gate, expert_q)?,
    ));
    drop(gate);
    println!(
        "  {:44} {expert_q:?} {:?} (per expert)",
        name("ffn_up_exps.weight"),
        up.dims()
    );
    owned.push((name("ffn_up_exps.weight"), quantize_stacked(&up, expert_q)?));
    drop(up);
    let down = st.f32("mtp.layers.0.mlp.experts.down_proj")?; // [512, 2560, 640]: k = 640 → Q8_0
    println!(
        "  {:44} Q8_0 {:?} (per expert)",
        name("ffn_down_exps.weight"),
        down.dims()
    );
    owned.push((name("ffn_down_exps.weight"), quantize_stacked(&down, q8)?));
    drop(down);

    let arch = Value::String("qwen4exp".to_string());
    let nextn = Value::U32(1);
    let block = Value::U32(n_layers as u32 + 1);
    let note = Value::String(
        "Qwen3.8-Flash-Next MTP module quantized in-tree by lloom (issue #749) from the HF bf16 mtp.* tensors; ds4 naming".to_string(),
    );
    let meta: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("general.description", &note),
        ("qwen4exp.nextn_predict_layers", &nextn),
        ("qwen4exp.block_count", &block),
    ];
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut f = std::fs::File::create(&out)?;
    gguf_file::write(&mut f, &meta, &refs)?;
    let bytes: usize = owned.iter().map(|(_, t)| t.storage_size_in_bytes()).sum();
    println!(
        "WROTE {} tensors={} bytes={bytes}",
        out.display(),
        owned.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Check { layer, store, gguf } => {
            let store =
                store.unwrap_or_else(|| home(&format!(".cache/lloom/qwen4exp-l{layer}-check")));
            let gguf = match gguf {
                Some(p) => p,
                None => hf_snapshot("models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
                    .context("no HF snapshot for the GGUF; pass --gguf")?
                    .join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
            };
            check(layer, store, gguf)
        }
        Cmd::Write {
            store,
            out,
            n_layers,
            experts,
        } => {
            let store = store.unwrap_or_else(|| home(".cache/lloom/qwen4exp-mtp-bf16"));
            write(store, out, n_layers, experts)
        }
    }
}
