//! Real-scale synthetic checkpoints (`docs/porting-models.md` §21.8b).
//!
//! - The writer reproduces a checkpoint's layout, fills what it fills with
//!   the keyed values the tiny checkpoint holds (so a sparse copy of the
//!   tiny checkpoint IS the tiny checkpoint where it is filled), and leaves
//!   the rest a hole that reads as zeros and costs no disk.
//! - The committed manifests are the real checkpoints' whenever those are
//!   on this machine, and hold the byte counts #749 P18 measured.
//!
//! CPU: the question is the file, not a kernel.

use lloom::models::qwen4exp::testing::{
    gguf_bytes, scale_config_of, scale_manifest_path, scale_roles, Qwen4ExpTiny,
};
use lloom::models::qwen4exp::{Config, Qwen4Exp};
use lloom::profile::CrownExact;
use lloom::testing::scale::{Bulk, Fill, ShapeManifest, ShardShape, Synthetic, TensorShape};
use lloom::testing::tiny::{run, Role, TinyCheckpoint, Workload};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lloom-scale-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Every tensor of the file at `path`, by name, as the bytes it stores.
fn tensors(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let mut f = File::open(path)?;
    let content = Content::read(&mut f)?;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut out = Vec::new();
    for n in names {
        let q = content.tensor(&mut f, &n, &Device::Cpu)?;
        out.push((n, q.data()?.into_owned()));
    }
    Ok(out)
}

fn role_of(roles: &std::collections::HashMap<String, Role>) -> impl Fn(&TensorShape) -> Role + '_ {
    |t| {
        roles.get(&t.name).copied().unwrap_or(Role::Matrix {
            fan_in: *t.dims.last().unwrap_or(&1),
        })
    }
}

/// A sparse copy of the tiny Qwen4-exp checkpoint, from its own manifest:
/// every tensor it fills is the tiny checkpoint's bytes, every expert stack
/// it holes reads as zeros, and the model the loader builds from the filled
/// copy computes the tiny model's logits bit for bit.
#[test]
fn a_sparse_copy_is_the_checkpoint_where_filled_and_zeros_where_holed() -> Result<()> {
    let dir = temp("copy");
    let ckpt = TinyCheckpoint::<Qwen4ExpTiny>::template();
    let real = dir.join("tiny.gguf");
    std::fs::write(&real, gguf_bytes(&ckpt, None)?)?;

    let m = ShapeManifest::from_gguf(std::slice::from_ref(&real), "tiny")?;
    let back = ShapeManifest::from_json(&m.to_json())?;
    assert_eq!(back.to_json(), m.to_json(), "the manifest round-trips");
    let roles = scale_roles(&m)?;
    assert_eq!(
        roles.len(),
        m.tensors().count(),
        "every tensor has its schema role"
    );

    let seed = ckpt.weights.seed;
    let filled = Synthetic {
        seed,
        experts: Fill::Pool,
        tables: Fill::Pool,
        ..Synthetic::default()
    };
    let holed = Synthetic {
        seed,
        ..Synthetic::default()
    };
    let full = m.write_sparse(&dir.join("filled"), &filled, &role_of(&roles))?;
    let hole = m.write_sparse(&dir.join("holed"), &holed, &role_of(&roles))?;

    let want = tensors(&real)?;
    let got_full = tensors(&full[0])?;
    let got_hole = tensors(&hole[0])?;
    assert_eq!(want.len(), got_full.len());
    let mut n_experts = 0;
    for ((name, w), ((_, f), (_, h))) in want.iter().zip(got_full.iter().zip(&got_hole)) {
        assert_eq!(
            f, w,
            "{name}: the filled copy is not the checkpoint's bytes"
        );
        if Bulk::of(name) == Bulk::Experts {
            n_experts += 1;
            assert!(
                h.iter().all(|b| *b == 0),
                "{name}: a hole that is not zeros"
            );
            assert!(
                w.iter().any(|b| *b != 0),
                "{name}: the reference is zeros too"
            );
        } else {
            assert_eq!(h, w, "{name}: a stem of the holed copy");
        }
    }
    assert_eq!(
        n_experts,
        3 * ckpt.cfg.n_layers,
        "every layer's three stacks"
    );

    let dev = Device::Cpu;
    let opts = lloom::models::qwen4exp::LoadOptions::default();
    let work = Workload {
        prompt: 5,
        decode: 2,
    };
    let v = ckpt.cfg.vocab_size;
    let reference = run(
        &Qwen4Exp::<CrownExact, _>::from_gguf_shards(&[real], &dev, &opts)?,
        &ckpt.cfg,
        v,
        &dev,
        &work,
    )?;
    let copy = run(
        &Qwen4Exp::<CrownExact, _>::from_gguf_shards(&full, &dev, &opts)?,
        &ckpt.cfg,
        v,
        &dev,
        &work,
    )?;
    assert_eq!(
        copy, reference,
        "the filled copy's model is the checkpoint's"
    );
    let holed_run = run(
        &Qwen4Exp::<CrownExact, _>::from_gguf_shards(&hole, &dev, &opts)?,
        &ckpt.cfg,
        v,
        &dev,
        &work,
    )?;
    assert!(
        holed_run.iter().flatten().all(|x| x.is_finite()),
        "zero experts are a valid model"
    );
    assert_ne!(holed_run, reference, "the holes are read, not skipped");
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// A hole is the file's length and none of its disk: a 64 MiB expert stack
/// written as a hole takes under a MiB of blocks, and reads back as zeros.
#[cfg(unix)]
#[test]
fn a_hole_costs_no_disk() -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let dir = temp("hole");
    let dims = vec![64, 512, 512];
    let m = ShapeManifest {
        source: "a hole".into(),
        shards: vec![ShardShape {
            file: "hole.gguf".into(),
            metadata: vec![],
            tensors: vec![TensorShape {
                name: "blk.0.ffn_up_exps.weight".into(),
                dims: dims.clone(),
                dtype: GgmlDType::F32,
                offset: 0,
            }],
        }],
    };
    let paths = m.write_sparse(&dir, &Synthetic::default(), &|_| Role::Matrix {
        fan_in: 512,
    })?;
    let md = std::fs::metadata(&paths[0])?;
    let bytes = 64 * 512 * 512 * 4u64;
    println!(
        "HOLE len={} allocated={} tensor={bytes}",
        md.len(),
        md.blocks() * 512
    );
    assert!(md.len() >= bytes, "the file is its layout's length");
    assert!(
        md.blocks() * 512 < 1 << 20,
        "a hole took {} B of disk",
        md.blocks() * 512
    );
    let t = tensors(&paths[0])?;
    assert!(t[0].1.iter().all(|b| *b == 0));
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// `Fill::Heads`: every expert of a stack non-zero in its leading rows and
/// zero past them, no two experts' heads alike, and the stack's disk a few
/// KiB an expert — 64 experts of 1 MiB on under a MiB of blocks.
#[cfg(unix)]
#[test]
fn heads_make_every_expert_nonzero_and_distinct_for_a_few_kib_each() -> Result<()> {
    use lloom::testing::scale::HEAD_BYTES;
    use std::os::unix::fs::MetadataExt;
    let dir = temp("heads");
    let (experts, rows, cols) = (64usize, 512usize, 512usize);
    let m = ShapeManifest {
        source: "heads".into(),
        shards: vec![ShardShape {
            file: "heads.gguf".into(),
            metadata: vec![],
            tensors: vec![TensorShape {
                name: "blk.0.ffn_down_exps.weight".into(),
                dims: vec![experts, rows, cols],
                dtype: GgmlDType::F32,
                offset: 0,
            }],
        }],
    };
    let syn = Synthetic {
        experts: Fill::Heads,
        ..Synthetic::default()
    };
    let paths = m.write_sparse(&dir, &syn, &|_| Role::Matrix { fan_in: cols })?;
    let md = std::fs::metadata(&paths[0])?;
    let data = &tensors(&paths[0])?[0].1;
    let plane = rows * cols * 4;
    let head = HEAD_BYTES.div_ceil(cols * 4) * cols * 4;
    println!(
        "HEADS len={} allocated={} experts={experts} plane={plane} head={head}",
        md.len(),
        md.blocks() * 512
    );
    let mut heads: Vec<&[u8]> = Vec::with_capacity(experts);
    for e in 0..experts {
        let p = &data[e * plane..(e + 1) * plane];
        assert!(
            p[..head].iter().any(|b| *b != 0),
            "expert {e}: its head is zeros"
        );
        assert!(
            p[head..].iter().all(|b| *b == 0),
            "expert {e}: written past its head"
        );
        heads.push(&p[..head]);
    }
    for (e, h) in heads.iter().enumerate() {
        assert!(
            heads[..e].iter().all(|o| o != h),
            "expert {e}'s head repeats an earlier expert's"
        );
    }
    assert!(
        md.blocks() * 512 < 1 << 20,
        "{experts} heads of {head} B took {} B of disk",
        md.blocks() * 512
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// The committed Qwen3.8-Flash-Next shape holds #749 P18's byte counts
/// (`qwen4exp-dispatches-raw/p18-*.txt`: `resident_bytes=4115863040
/// expert_bytes_per_slot=83814400`, 47.03 GB predicted resident) and the
/// verified geometry (`Config::flash_next`).
#[test]
fn the_committed_qwen_shape_holds_the_real_byte_counts() -> Result<()> {
    let m = ShapeManifest::read(&scale_manifest_path())?;
    let (stems, experts, tables) = m.bytes_by_bulk();
    let cfg = scale_config_of(&m)?;
    println!(
        "SHAPE {}: {} shards, {} tensors, stems {stems} B, experts {experts} B, tables {tables} B",
        m.source,
        m.shards.len(),
        m.tensors().count()
    );
    assert_eq!(stems, 4_115_863_040, "P18's resident_bytes");
    assert_eq!(
        experts / cfg.n_experts as u64,
        83_814_400,
        "P18's expert_bytes_per_slot"
    );
    assert_eq!(experts % cfg.n_experts as u64, 0);
    assert_eq!(
        format!("{:.2}", (stems + experts) as f64 / 1e9),
        "47.03",
        "P18's predicted resident load"
    );
    assert_eq!(m.tensors().count(), 1224, "1223 + the PLE table");
    assert_eq!(
        tables, 28_800_138_240,
        "the n-gram table (`ple.rs`: 28.8 GB)"
    );
    let want = Config::flash_next();
    for (field, a, b) in [
        ("n_layers", cfg.n_layers, want.n_layers),
        ("hidden_size", cfg.hidden_size, want.hidden_size),
        ("vocab_size", cfg.vocab_size, want.vocab_size),
        ("n_experts", cfg.n_experts, want.n_experts),
        (
            "n_experts_per_tok",
            cfg.n_experts_per_tok,
            want.n_experts_per_tok,
        ),
        ("head_dim", cfg.head_dim, want.head_dim),
        (
            "moe_intermediate_size",
            cfg.moe_intermediate_size,
            want.moe_intermediate_size,
        ),
    ] {
        assert_eq!(a, b, "{field}");
    }
    Ok(())
}

/// The committed shape is the real checkpoint's, when the checkpoint is on
/// this machine: every tensor, kind and offset, every header key. A new
/// revision of the checkpoint fails here by name.
#[test]
fn the_committed_qwen_shape_is_the_real_headers() -> Result<()> {
    use lloom::testing::fixture::Scaled;
    let Some(first) = Qwen4ExpTiny::locate() else {
        println!("SKIPPED: no Qwen3.8-Flash-Next GGUF on this machine");
        return Ok(());
    };
    let real = ShapeManifest::from_gguf(&lloom::hub::gguf_shards(&first), "")?;
    let committed = ShapeManifest::read(&scale_manifest_path())?;
    let mut a = real.to_json();
    let mut b = committed.to_json();
    a["source"] = "".into();
    b["source"] = "".into();
    assert!(a == b, "the committed shape is not {}'s", first.display());
    Ok(())
}

/// The committed LFM2 geometry is the real checkpoint's, when the checkpoint
/// is on this machine.
#[test]
fn the_committed_lfm2_geometry_is_the_real_config() -> Result<()> {
    use lloom::models::lfm2::Lfm2Config;
    use lloom::models::lfm2_testing::{geometry_json, scale_geometry_path, Lfm2Tiny};
    use lloom::testing::fixture::Scaled;
    let Some(dir) = Lfm2Tiny::locate() else {
        println!("SKIPPED: no LFM2.5-2.6B on this machine");
        return Ok(());
    };
    let read = |p: &Path| -> Result<serde_json::Value> {
        let c: Lfm2Config =
            serde_json::from_slice(&std::fs::read(p)?).map_err(lloom_tensor::Error::wrap)?;
        Ok(geometry_json(&c.into_config(false)))
    };
    assert_eq!(
        read(&dir.join("config.json"))?,
        read(&scale_geometry_path())?
    );
    Ok(())
}

/// The committed MTP sidecar shape is the sidecar's headers, when the
/// sidecar is on this machine; `LLOOM_BLESS_LISTINGS=1` writes it.
#[test]
fn the_committed_mtp_sidecar_shape_is_the_real_headers() -> Result<()> {
    use lloom::models::qwen4exp::testing::mtp_sidecar_manifest_path;
    let Some(home) = std::env::var_os("HOME") else {
        println!("SKIPPED: no HOME");
        return Ok(());
    };
    let sidecar = Path::new(&home).join(".cache/lloom/qwen4exp-mtp-q8.gguf");
    if !sidecar.exists() {
        println!("SKIPPED: no MTP sidecar at {}", sidecar.display());
        return Ok(());
    }
    let real =
        ShapeManifest::from_gguf(std::slice::from_ref(&sidecar), "qwen4exp-mtp-quantize Q8")?;
    let path = mtp_sidecar_manifest_path();
    if std::env::var_os("LLOOM_BLESS_LISTINGS").is_some() {
        real.write(&path)?;
    }
    let committed = ShapeManifest::read(&path)?;
    assert!(
        real.to_json() == committed.to_json(),
        "the committed sidecar shape is not {}'s",
        sidecar.display()
    );
    Ok(())
}
