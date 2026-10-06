//! The tiny checkpoints' one-source, three-outputs contract
//! (`docs/porting-models.md` §21.8, INV-2/3a/5), on the CPU:
//!
//! - the model built in memory and the same checkpoint written to files and
//!   read back through the family's real loader compute the same logits,
//!   bit for bit;
//! - the schema is exactly what the loader reads — every tensor of the file
//!   loaded, none missing;
//! - weights are keyed by (seed, name, index): a different seed moves the
//!   logits, the same one does not, whatever the order;
//! - a checkpoint that drifts from the schema is refused with the tensor
//!   named.

use lloom::models::lfm2::Config as LfmConfig;
use lloom::models::lfm2_testing::{self, Lfm2Tiny};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::Config as QwenConfig;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{self, TinyCheckpoint, TinyFamily, ValueLaw, Weights, Workload};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-tiny-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|s| s.iter().map(|x| x.to_bits()).collect())
        .collect()
}

const W: Workload = Workload {
    prompt: 5,
    decode: 2,
};

fn lfm2_logits(ckpt: &TinyCheckpoint<Lfm2Tiny>) -> Result<Vec<Vec<f32>>> {
    let m = ckpt.model::<CrownExact>(&Device::Cpu)?;
    tiny::run(&m, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &W)
}

/// In memory == written and loaded, at each store dtype; the recorded
/// schema is the file's table and every tensor of it is loaded.
#[test]
fn lfm2_memory_and_files_agree() -> Result<()> {
    for dense in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16] {
        let mut cfg = Lfm2Tiny::template();
        cfg.vocab_size = Lfm2Tiny::tokenizer().vocab_size();
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::new(
            cfg,
            Weights {
                dense,
                ..Weights::default()
            },
        );
        let mem = lfm2_logits(&ckpt)?;
        assert!(mem[0].iter().all(|x| x.is_finite()));
        assert_ne!(
            bits(&mem[1..2]),
            bits(&mem[2..3]),
            "two decode steps must differ"
        );

        let dir = scratch(&format!("lfm2-{dense:?}"));
        let path = ckpt.write(&dir)?;
        let (model, record) = lfm2_testing::load_recorded::<CrownExact>(&path, &Device::Cpu, None)?;
        let loaded: LfmConfig = ckpt.cfg.clone();
        let files = tiny::run(&model, &loaded, loaded.vocab_size, &Device::Cpu, &W)?;
        assert_eq!(bits(&files), bits(&mem), "{dense:?}: files != memory");

        let schema = ckpt.schema()?;
        let header: std::collections::BTreeSet<String> =
            record.header().map(|(n, _)| n.to_string()).collect();
        let listed: std::collections::BTreeSet<String> =
            schema.tensors.iter().map(|t| t.name.clone()).collect();
        assert_eq!(
            header, listed,
            "the file holds the schema, no more, no less"
        );
        assert_eq!(
            record.loaded(),
            listed,
            "the loader read every tensor of the file"
        );
        println!(
            "lfm2 {dense:?}: {} tensors, {} elements, logits[0][..3] {:?}",
            schema.tensors.len(),
            schema.elements(),
            &mem[0][..3]
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    Ok(())
}

/// The config a written LFM2 checkpoint carries reads back as the same
/// geometry through the checkpoint parser the engine uses.
#[test]
fn lfm2_config_round_trips() -> Result<()> {
    let mut cfg = Lfm2Tiny::template();
    cfg.vocab_size = Lfm2Tiny::tokenizer().vocab_size();
    let ckpt = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default());
    let dir = scratch("lfm2-cfg");
    ckpt.write(&dir)?;
    let back: lloom::models::lfm2::Lfm2Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?).unwrap();
    let back = back.into_config(false);
    for (a, b) in [
        (back.vocab_size, cfg.vocab_size),
        (back.hidden_size, cfg.hidden_size),
        (back.intermediate_size, cfg.intermediate_size),
        (back.num_hidden_layers, cfg.num_hidden_layers),
        (back.num_attention_heads, cfg.num_attention_heads),
        (back.num_key_value_heads, cfg.num_key_value_heads),
        (back.conv_l_cache, cfg.conv_l_cache),
    ] {
        assert_eq!(a, b);
    }
    assert_eq!(back.layer_types, cfg.layer_types);
    assert_eq!(back.tie_embedding, cfg.tie_embedding);
    assert_eq!(back.eos_token_id, Lfm2Tiny::tokenizer().id("<|im_end|>"));
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// Keyed: the same seed → the same logits; another seed → other logits;
/// the law is part of the key.
#[test]
fn weights_are_keyed_by_seed_and_law() -> Result<()> {
    let base = TinyCheckpoint::<Lfm2Tiny>::template();
    let a = lfm2_logits(&base)?;
    let again = lfm2_logits(&base)?;
    assert_eq!(bits(&a), bits(&again));
    let mut other = base.clone();
    other.weights.seed += 1;
    assert_ne!(bits(&lfm2_logits(&other)?), bits(&a));
    let mut ramp = base.clone();
    ramp.weights.law = ValueLaw::RampMod17;
    assert_ne!(bits(&lfm2_logits(&ramp)?), bits(&a));
    Ok(())
}

fn qwen_tiny(vocab: usize, weights: Weights) -> TinyCheckpoint<Qwen4ExpTiny> {
    let mut cfg = Qwen4ExpTiny::template();
    cfg.vocab_size = vocab;
    TinyCheckpoint::new(cfg, weights)
}

/// In memory (the GGUF's bytes through the loader) == the written file
/// through `from_gguf_shards`; the header reads back as the config.
#[test]
fn qwen4exp_memory_and_files_agree() -> Result<()> {
    let tok = Qwen4ExpTiny::tokenizer();
    for dense in [GgmlDType::F32, GgmlDType::F16, GgmlDType::Q8_0] {
        let ckpt = qwen_tiny(
            tok.vocab_size(),
            Weights {
                dense,
                ..Weights::default()
            },
        );
        let m = ckpt.model::<CrownExact>(&Device::Cpu)?;
        let mem = tiny::run(&m, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &W)?;
        assert!(mem.iter().flatten().all(|x| x.is_finite()), "{dense:?}");
        assert_ne!(bits(&mem[1..2]), bits(&mem[2..3]));

        let dir = scratch(&format!("qwen-{dense:?}"));
        let path = ckpt.write(&dir)?;
        let f = Qwen4ExpTiny::load::<CrownExact>(&path, &Device::Cpu)?;
        let files = tiny::run(&f, &ckpt.cfg, ckpt.cfg.vocab_size, &Device::Cpu, &W)?;
        assert_eq!(bits(&files), bits(&mem), "{dense:?}: files != memory");

        let mut r = std::fs::File::open(&path)?;
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        let mut back = QwenConfig::from_gguf_metadata(&content.metadata)?;
        back.vocab_size = ckpt.cfg.vocab_size;
        let mut want = ckpt.cfg.clone();
        want.norm_eps = want.norm_eps as f32 as f64;
        want.eos_token_id = tok.id("<|endoftext|>").unwrap();
        want.bos_token_id = want.eos_token_id;
        assert_eq!(format!("{back:?}"), format!("{want:?}"));
        let schema = ckpt.schema()?;
        assert_eq!(content.tensor_infos.len(), schema.tensors.len());
        println!(
            "qwen4exp {dense:?}: {} tensors, {} elements, logits[0][..3] {:?}",
            schema.tensors.len(),
            schema.elements(),
            &mem[0][..3]
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    Ok(())
}

/// A file missing a tensor, or holding one at the wrong shape, is refused
/// before anything loads, with every discrepancy named.
#[test]
fn qwen4exp_drift_from_the_schema_is_refused_by_name() -> Result<()> {
    let ckpt = qwen_tiny(128, Weights::default());
    let schema = ckpt.schema()?;
    let table = |skip: &str, reshape: &str| -> Vec<(String, Vec<usize>)> {
        schema
            .tensors
            .iter()
            .filter(|t| t.name != skip)
            .map(|t| {
                let mut s = t.shape.clone();
                if t.name == reshape {
                    s[0] += 1;
                }
                (t.name.clone(), s)
            })
            .collect()
    };
    let check = |t: &[(String, Vec<usize>)]| {
        schema.validate(t.iter().map(|(n, s)| (n.as_str(), s.as_slice())))
    };
    check(&table("", ""))?;
    let e = check(&table(
        "blk.3.indexer.k_norm.weight",
        "blk.0.ssm_out.weight",
    ))
    .unwrap_err()
    .to_string();
    assert!(e.contains("missing: blk.3.indexer.k_norm.weight"), "{e}");
    assert!(e.contains("shape: blk.0.ssm_out.weight is [33, 32]"), "{e}");
    assert!(e.contains("(2 of "), "{e}");

    // Through the loader: a GGUF without one tensor never gets to build.
    let bytes = lloom::models::qwen4exp::testing::gguf_bytes(&ckpt, None)?;
    let mut r = std::io::Cursor::new(bytes);
    let mut content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
    content.tensor_infos.remove("blk.1.ssm_a");
    let e = match lloom::models::qwen4exp::Qwen4Exp::<CrownExact, _>::from_gguf(
        content,
        &mut r,
        &Device::Cpu,
    ) {
        Ok(_) => panic!("a checkpoint without blk.1.ssm_a loaded"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("missing: blk.1.ssm_a [4]"), "{e}");
    Ok(())
}

/// Packed kinds the CPU cannot multiply by run on the CPU through the
/// dequantized checkpoint, and every kind's packed experts stay finite.
#[test]
fn qwen4exp_packed_experts_have_a_cpu_reference() -> Result<()> {
    for experts in [
        GgmlDType::IQ4XS,
        GgmlDType::IQ3XXS,
        GgmlDType::Q2_0,
        GgmlDType::Q4K,
    ] {
        let mut cfg = Qwen4ExpTiny::template();
        // Every K of the expert stacks a whole number of 256-element blocks.
        cfg.hidden_size = 256;
        cfg.moe_intermediate_size = 256;
        let ckpt = TinyCheckpoint::<Qwen4ExpTiny>::new(
            cfg,
            Weights {
                experts,
                ..Weights::default()
            },
        );
        let reference = ckpt.dequantized();
        let m = reference.model::<CrownExact>(&Device::Cpu)?;
        let out = tiny::run(
            &m,
            &reference.cfg,
            reference.cfg.vocab_size,
            &Device::Cpu,
            &W,
        )?;
        assert!(out.iter().flatten().all(|x| x.is_finite()), "{experts:?}");
        println!(
            "qwen4exp experts {experts:?}: reference logits[0][..3] {:?}",
            &out[0][..3]
        );
    }
    Ok(())
}
