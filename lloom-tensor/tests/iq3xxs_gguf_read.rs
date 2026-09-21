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
