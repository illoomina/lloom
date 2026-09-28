//! The declared tensor schema against the real Qwen3.8-Flash-Next headers
//! (`docs/porting-models.md` §21.8, INV-2): headers only, no weight is read.
//! Every quant in the HF cache must validate with the PLE block, and the
//! MTP sidecar with the MTP layer; a checkpoint absent from this machine is
//! reported, not passed silently.

use lloom::models::qwen4exp::schema::{tensor_schema, Kinds, SchemaOptions};
use lloom::models::qwen4exp::Config;
use lloom_tensor::quantized::gguf_file::Content;
use std::path::{Path, PathBuf};

fn shards(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| {
            r.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn checkpoints() -> Vec<(String, Vec<PathBuf>)> {
    let hub = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/huggingface/hub");
    let mut out = Vec::new();
    for (repo, sub) in [
        (
            "models--bartowski--Qwen3.8-Flash-Next-GGUF",
            "Qwen3.8-Flash-Next-IQ4_XS",
        ),
        (
            "models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
            "IQ3_XXS",
        ),
    ] {
        let Ok(snaps) = std::fs::read_dir(hub.join(repo).join("snapshots")) else {
            continue;
        };
        for s in snaps.filter_map(|e| e.ok()) {
            let files = shards(&s.path().join(sub));
            if !files.is_empty() {
                out.push((format!("{repo}/{sub}"), files));
            }
        }
    }
    out
}

fn read_headers(files: &[PathBuf]) -> Vec<Content> {
    files
        .iter()
        .map(|p| Content::read(&mut std::fs::File::open(p).unwrap()).unwrap())
        .collect()
}

#[test]
fn the_real_headers_match_the_declared_schema() {
    let found = checkpoints();
    if found.is_empty() {
        println!("SKIPPED: no Qwen3.8-Flash-Next GGUF in the HF cache");
        return;
    }
    for (name, files) in found {
        let heads = read_headers(&files);
        let mut cfg = Config::from_gguf_metadata(&heads[0].metadata).unwrap();
        let table: Vec<(String, Vec<usize>)> = heads
            .iter()
            .flat_map(|c| {
                c.tensor_infos
                    .iter()
                    .map(|(n, t)| (n.clone(), t.shape.dims().to_vec()))
            })
            .collect();
        cfg.vocab_size = table
            .iter()
            .find(|(n, _)| n == "token_embd.weight")
            .unwrap()
            .1[0];
        let schema = tensor_schema(
            &cfg,
            &SchemaOptions {
                ple: true,
                mtp: false,
            },
            &Kinds::default(),
        );
        schema
            .validate(table.iter().map(|(n, s)| (n.as_str(), s.as_slice())))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let unlisted: Vec<&str> = table
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| schema.get(n).is_none())
            .collect();
        assert_eq!(unlisted, vec!["per_layer_token_embd.weight"], "{name}");
        println!(
            "{name}: {} shards, {} tensors, schema {} + the PLE table: OK",
            files.len(),
            table.len(),
            schema.tensors.len()
        );

        let sidecar = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".cache/lloom/qwen4exp-mtp-q8.gguf");
        if sidecar.is_file() {
            let mtp = read_headers(std::slice::from_ref(&sidecar));
            let mtp_table: Vec<(String, Vec<usize>)> = mtp[0]
                .tensor_infos
                .iter()
                .map(|(n, t)| (n.clone(), t.shape.dims().to_vec()))
                .collect();
            let with = tensor_schema(
                &cfg,
                &SchemaOptions {
                    ple: false,
                    mtp: true,
                },
                &Kinds::default(),
            );
            let all = table
                .iter()
                .chain(&mtp_table)
                .map(|(n, s)| (n.as_str(), s.as_slice()));
            with.validate(all)
                .unwrap_or_else(|e| panic!("{name} + MTP: {e}"));
            println!(
                "{name} + {}: {} MTP tensors: OK",
                sidecar.display(),
                mtp_table.len()
            );
        } else {
            println!("{name}: no MTP sidecar at {}", sidecar.display());
        }
    }
}
