//! Real-checkpoint read gate for the IQ / Q2_0 GGUF reader (issue #747).
//!
//! Opens the downloaded Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS shard, parses its
//! header (which fails on the first IQ tensor without the `from_u32` extension),
//! and CPU-dequantizes one Q2_0 expert tensor and one IQ tensor — the two paths
//! that were unloadable before this issue. `#[ignore]` because it needs the
//! 44 GB file present; run with `--ignored` on a machine that has it.

use lloom_tensor::quantized::gguf_file;

fn checkpoint_path() -> Option<std::path::PathBuf> {
    // Resolve the HF cache snapshot of the IQ3_XXS first shard.
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    let p = snap.join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf");
    p.exists().then_some(p)
}

#[test]
#[ignore = "needs the 44 GB IQ3_XXS checkpoint present"]
fn header_parses_and_iq_q2_0_tensors_dequant() {
    let Some(path) = checkpoint_path() else {
        eprintln!("checkpoint not present; skipping");
        return;
    };
    let mut f = std::fs::File::open(&path).expect("open gguf");
    let content = gguf_file::Content::read(&mut f).expect("parse gguf header");
    assert_eq!(content.tensor_infos.len(), 1223, "tensor count");

    // Every real type code parsed (no `unknown dtype` bail) — the from_u32 gate.
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for ti in content.tensor_infos.values() {
        *hist.entry(format!("{:?}", ti.ggml_dtype)).or_default() += 1;
    }
    eprintln!("dtype histogram: {hist:?}");
    // The types this issue added must be present in the real file.
    for want in ["Q2_0", "IQ3XXS", "IQ4NL", "IQ2XS"] {
        assert!(hist.contains_key(want), "expected {want} tensors in the checkpoint");
    }

    // CPU-dequant one Q2_0 expert-down tensor and one IQ tensor: exercises the
    // ported `to_float` on real bytes end to end (device = CPU).
    let dev = lloom_tensor::Device::Cpu;
    for name in [
        "blk.1.ffn_down_exps.weight", // Q2_0
        "blk.0.attn_qkv.weight",      // IQ4_XS
    ] {
        let qt = content.tensor(&mut f, name, &dev).expect("load tensor");
        let deq = qt.dequantize(&dev).expect("dequantize");
        let v = deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(!v.is_empty());
        // Sanity: dequantized weights are finite and not all zero.
        assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite weight");
        assert!(v.iter().any(|&x| x != 0.0), "{name}: all-zero dequant");
        eprintln!("{name}: {} elems, first {:?}", v.len(), &v[..4.min(v.len())]);
    }
}

/// Shape/type dump of the tensors the loader binds (blk.0 linear, blk.3 full,
/// top level) — run with `--ignored --nocapture` to read the artifact.
#[test]
#[ignore = "needs the 44 GB IQ3_XXS checkpoint present"]
fn dump_loader_facing_shapes() {
    let Some(path) = checkpoint_path() else { return };
    let mut f = std::fs::File::open(&path).expect("open gguf");
    let content = gguf_file::Content::read(&mut f).expect("parse gguf header");
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    for name in names {
        if !name.starts_with("blk.") || name.starts_with("blk.0.") || name.starts_with("blk.3.") || name.starts_with("blk.1.ple") {
            let ti = &content.tensor_infos[name];
            eprintln!("{name}: {:?} {:?}", ti.ggml_dtype, ti.shape.dims());
        }
    }
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        if k.starts_with("qwen4exp.") || k.starts_with("general.") {
            let v = &content.metadata[k];
            let s = format!("{v:?}");
            eprintln!("META {k} = {}", &s[..s.len().min(160)]);
        }
    }
}

/// Values of the small SSM vectors the DeltaNet gate reads, to check the sign
/// convention the converter stored (`ssm_a` = -exp(A_log) or raw A_log).
#[test]
#[ignore = "needs the 44 GB IQ3_XXS checkpoint present"]
fn dump_ssm_scalars() {
    let Some(path) = checkpoint_path() else { return };
    let mut f = std::fs::File::open(&path).expect("open gguf");
    let content = gguf_file::Content::read(&mut f).expect("parse gguf header");
    let dev = lloom_tensor::Device::Cpu;
    for name in ["blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.0.hc_attn_norm.weight", "blk.0.ffn_gate_inp_shexp.weight"] {
        let qt = content.tensor(&mut f, name, &dev).expect("load");
        let v = qt.dequantize(&dev).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("{name}: n={} min={mn} max={mx} first={:?}", v.len(), &v[..6.min(v.len())]);
    }
}

/// Scan every tensor's raw bytes for a non-finite block scale (the f16 `d` at
/// offset 0 of every quantized block type this checkpoint uses) or a non-finite
/// BF16/F16/F32 value — a poisoned weight is the cheapest explanation of a NaN
/// that appears at one layer's MoE for one routing and not another.
#[test]
#[ignore = "needs the 44 GB IQ3_XXS checkpoint present"]
fn scan_for_nonfinite_weights() {
    use lloom_tensor::quantized::GgmlDType;
    use std::io::{Read, Seek, SeekFrom};
    let Some(path) = checkpoint_path() else { return };
    let mut f = std::fs::File::open(&path).expect("open gguf");
    let content = gguf_file::Content::read(&mut f).expect("parse gguf header");
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    let mut bad = 0usize;
    let mut buf = Vec::new();
    for name in names {
        let ti = &content.tensor_infos[name];
        let dt = ti.ggml_dtype;
        let n_elems = ti.shape.elem_count();
        let size = n_elems / dt.block_size() * dt.type_size();
        buf.clear();
        buf.resize(size, 0);
        f.seek(SeekFrom::Start(content.tensor_data_offset + ti.offset)).unwrap();
        f.read_exact(&mut buf).unwrap();
        let mut nonfinite = 0usize;
        let mut zero_scale = 0usize;
        match dt {
            GgmlDType::F32 => {
                for c in buf.chunks_exact(4) {
                    if !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite() { nonfinite += 1; }
                }
            }
            GgmlDType::F16 => {
                for c in buf.chunks_exact(2) {
                    if !half::f16::from_le_bytes([c[0], c[1]]).to_f32().is_finite() { nonfinite += 1; }
                }
            }
            GgmlDType::BF16 => {
                for c in buf.chunks_exact(2) {
                    if !half::bf16::from_le_bytes([c[0], c[1]]).to_f32().is_finite() { nonfinite += 1; }
                }
            }
            _ => {
                // every block type here starts with its f16 scale `d`
                // (Q4K/Q5K/Q6K/Q2K/Q3K too: `d` at offset 0 or after the scales —
                // Q4K/Q5K: d at 0; Q6K: d at the END; Q2K/Q3K: d after scales).
                let ts = dt.type_size();
                let d_off = match dt {
                    GgmlDType::Q6K => ts - 2,
                    GgmlDType::Q2K => 16 + 64,
                    GgmlDType::Q3K => 32 + 64 + 12,
                    _ => 0,
                };
                for b in buf.chunks_exact(ts) {
                    let d = half::f16::from_le_bytes([b[d_off], b[d_off + 1]]).to_f32();
                    if !d.is_finite() { nonfinite += 1; }
                    if d == 0.0 { zero_scale += 1; }
                }
            }
        }
        if nonfinite > 0 {
            bad += 1;
            eprintln!("BAD {name}: {dt:?} nonfinite={nonfinite} zero_scale={zero_scale} of {} blocks/elems", buf.len() / dt.type_size().max(1));
        }
    }
    eprintln!("SCAN tensors={} with_nonfinite={bad}", content.tensor_infos.len());
}
