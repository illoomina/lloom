//! Qwen4-exp on the engine's decode driver (`Session::decode`, the blocking
//! readback arm) produces the ids the harness's own loop produces — with the
//! N-gram PLE attached, so the input ids the driver hands the model
//! (`CausalLM::note_input_ids`) are the ids the PLE hashes. And the note is
//! consumed by exactly the forward it was made for: a note the driver did
//! make wins over the input tensor; a note that does not fit the next forward
//! is discarded and the tensor's ids are read back instead.
//!
//! CPU: the question is the driver seam and the id bookkeeping, not a kernel.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident,
};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-engine-id-{}-{tag}.gguf", std::process::id()))
}

/// A PLE table in a GGUF, as the loader names it.
fn write_table(cfg: &Config, ple_rows: usize, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let table = testing::wt((ple_rows, cfg.ple_head_dim), 3.0, &dev);
    let q = QTensor::quantize(&table, GgmlDType::F32)?;
    let refs: Vec<(&str, &QTensor)> = vec![("per_layer_token_embd.weight", &q)];
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &refs)?;
    let _ = Content::read(&mut File::open(path)?)?;
    Ok(())
}

/// A resident tiny model with the PLE attached at layer 1.
fn model_with_ple(cfg: &Config, path: &Path) -> Result<Qwen4Exp<Resident>> {
    let dev = Device::Cpu;
    let mut model = testing::resident_model(cfg, &dev);
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
        testing::lin(hc_dim, ple_dim, 21.0, &dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::wt((hc_dim, 4), 23.0, &dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, PleTable::Mmap(table), block)?;
    Ok(model)
}

/// The harness's loop: the whole prompt, then greedy steps with the sampled
/// id read back and re-uploaded.
fn harness_loop(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<u32>> {
    let dev = Device::Cpu;
    let mut cache = model.new_cache(cfg, &dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let mut ids = Vec::new();
    for _ in 0..steps {
        let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        ids.push(next);
        let input = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
    }
    Ok(ids)
}

#[test]
fn the_blocking_arm_through_session_decode_matches_the_harness_loop() -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    let path = temp_gguf("session");
    write_table(&cfg, 80, &path)?;
    let dev = Device::Cpu;
    let model = model_with_ple(&cfg, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let steps = 6;
    let want = harness_loop(&model, &cfg, &prompt, steps)?;
    assert!(
        want.iter().any(|&t| t != want[0]),
        "a degenerate stream cannot tell the arms apart: {want:?}"
    );

    let cache = model.new_cache(&cfg, &dev)?;
    let mut session = Session::new(&model, cache);
    // An EOS no stream contains, so the run is exactly `steps` long.
    let mut rb = Blocking::new(&[u32::MAX], &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut spec = IdentityProposer;
    let opts = DecodeOptions::new(steps);
    let mut streamed: Vec<u32> = Vec::new();
    let mut sink = |ids: &[u32]| -> Result<()> {
        streamed.extend_from_slice(ids);
        Ok(())
    };
    let out = session.decode(&mut spec, &mut rb, &mut sampler, &prompt, &opts, &mut sink)?;
    assert_eq!(
        out.ids, want,
        "the driver's ids differ from the harness loop's"
    );
    assert_eq!(streamed, want, "the streamed ids differ from the run's");
    // The driver noted every fed id; the model consumed every note.
    assert!(
        !session.cache_mut().pending_noted,
        "a note was left unconsumed after the run"
    );
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn a_note_is_consumed_by_the_next_forward_and_only_it() -> Result<()> {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    let path = temp_gguf("note");
    write_table(&cfg, 80, &path)?;
    let dev = Device::Cpu;
    let model = model_with_ple(&cfg, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
    let step = Tensor::from_vec(vec![4u32], (1, 1), &dev)?;

    // Reference: no notes, the ids read back from the tensors.
    let mut plain = model.new_cache(&cfg, &dev)?;
    model.forward(&input, 0, &mut plain)?;
    let reference = model
        .forward(&step, plain.pos, &mut plain)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // A note that fits the next forward is what the PLE hashes: noting a
    // DIFFERENT id than the tensor carries moves the logits — the note wins,
    // which is the only way to see it was used.
    let mut noted = model.new_cache(&cfg, &dev)?;
    model.forward(&input, 0, &mut noted)?;
    model.note_input_ids(&mut noted, &[11])?;
    assert!(noted.pending_noted);
    let moved = model
        .forward(&step, noted.pos, &mut noted)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(!noted.pending_noted, "the forward must consume the note");
    assert!(
        moved != reference,
        "a note of another id must change the PLE's row"
    );

    // The right note reproduces the reference exactly.
    let mut same = model.new_cache(&cfg, &dev)?;
    model.forward(&input, 0, &mut same)?;
    model.note_input_ids(&mut same, &[4])?;
    let via_note = model
        .forward(&step, same.pos, &mut same)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        via_note == reference,
        "the noted id must hash as the read-back id does"
    );

    // A note that does not fit (two ids for a one-token forward) is dropped
    // and the tensor's ids are read back: the reference again.
    let mut misfit = model.new_cache(&cfg, &dev)?;
    model.forward(&input, 0, &mut misfit)?;
    model.note_input_ids(&mut misfit, &[11, 12])?;
    let fallback = model
        .forward(&step, misfit.pos, &mut misfit)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(!misfit.pending_noted);
    assert!(
        fallback == reference,
        "a misfit note must not reach the hash"
    );
    std::fs::remove_file(&path).ok();
    Ok(())
}
