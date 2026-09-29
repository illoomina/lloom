//! Real-scale synthetic checkpoints (`docs/porting-models.md` §21.8b): a
//! checkpoint with a real one's file layout and none of its values.
//!
//! A [`ShapeManifest`] is what a real GGUF's headers say — every tensor's
//! name, shape, kind and offset, and the header keys a loader reads — taken
//! once from the real files and committed, so a machine without the weights
//! can write the same layout. [`ShapeManifest::write_sparse`] writes it as a
//! sparse file: the tensors that decide a run's dynamics (stems, norms,
//! routers, embeddings) carry keyed values from a small pool of synthesized
//! blocks, and the bulk the real-scale failures are about (the routed
//! expert stacks, the n-gram table) is a hole. A hole costs no disk, reads
//! as zero pages — a valid tensor of every kind — and goes through the page
//! cache like any written page, so every path that opens the file (`pread`,
//! a memory map, an IO queue) runs unchanged.

use super::tiny::init::{name_hash, splitmix64, Role, ValueLaw};
use super::tiny::packed::{self, ALL_KINDS};
use lloom_tensor::quantized::gguf_file::{self, Content, HeaderTensor, Value};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{bail, Error, Result, Shape};
use serde_json::{json, Map, Value as Json};
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The manifest file's format line.
pub const FORMAT: &str = "lloom shape manifest v1";

/// A stem at most this large is synthesized whole; a larger one repeats a
/// pool of [`POOL_BYTES`] of whole rows.
pub const WHOLE_BYTES: usize = 4 << 20;
pub const POOL_BYTES: usize = 1 << 20;

/// One tensor of a shard: its name, row-major dims, kind, and offset from
/// the shard's data start.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
    pub offset: u64,
}

impl TensorShape {
    /// The packed bytes the file holds for it.
    pub fn bytes(&self) -> usize {
        self.dims.iter().product::<usize>() / self.dtype.block_size() * self.dtype.type_size()
    }
}

/// One shard: its file name, its header keys and its tensors in file order.
#[derive(Debug, Clone)]
pub struct ShardShape {
    pub file: String,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorShape>,
}

/// What a real checkpoint's GGUF headers say, without the tokenizer.
#[derive(Debug, Clone)]
pub struct ShapeManifest {
    /// Where it was read from (repo, revision, quant), for the reader.
    pub source: String,
    pub shards: Vec<ShardShape>,
}

/// What a synthetic checkpoint stores where the real one has bulk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// Zero pages: no disk, no values.
    Hole,
    /// A pool of keyed blocks, repeated: real bytes on disk, so a read costs
    /// what a read of the real file does.
    Pool,
}

/// How a synthetic checkpoint is made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Synthetic {
    pub law: ValueLaw,
    pub seed: u64,
    /// The routed expert stacks (`*_exps.weight`).
    pub experts: Fill,
    /// The row tables read by lookup, not swept (the n-gram table).
    pub tables: Fill,
}

impl Default for Synthetic {
    fn default() -> Self {
        Self {
            law: ValueLaw::DEFAULT,
            seed: 0x7131,
            experts: Fill::Hole,
            tables: Fill::Hole,
        }
    }
}

/// What a tensor is to the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bulk {
    Experts,
    Table,
    Stem,
}

impl Bulk {
    pub fn of(name: &str) -> Self {
        if name.ends_with("_exps.weight") {
            Bulk::Experts
        } else if name == "per_layer_token_embd.weight" {
            Bulk::Table
        } else {
            Bulk::Stem
        }
    }
}

impl ShapeManifest {
    /// Read every shard's header (no tensor data) at `paths`. The
    /// tokenizer's vocabulary, merges and chat template are dropped — a
    /// synthetic checkpoint carries no tokenizer — and its scalar ids kept,
    /// since a loader may read them (the end-of-text id).
    pub fn from_gguf(paths: &[PathBuf], source: &str) -> Result<Self> {
        let mut shards = Vec::with_capacity(paths.len());
        for p in paths {
            let content = Content::read(&mut std::fs::File::open(p)?)?;
            let mut metadata: Vec<(String, Value)> = content
                .metadata
                .into_iter()
                .filter(|(k, v)| {
                    !(k.starts_with("tokenizer.")
                        && (matches!(v, Value::Array(_)) || k == "tokenizer.chat_template"))
                })
                .collect();
            metadata.sort_by(|a, b| a.0.cmp(&b.0));
            let mut tensors: Vec<TensorShape> = content
                .tensor_infos
                .into_iter()
                .map(|(name, t)| TensorShape {
                    name,
                    dims: t.shape.dims().to_vec(),
                    dtype: t.ggml_dtype,
                    offset: t.offset,
                })
                .collect();
            tensors.sort_by_key(|t| t.offset);
            shards.push(ShardShape {
                file: p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("shard.gguf")
                    .to_string(),
                metadata,
                tensors,
            });
        }
        Ok(Self {
            source: source.to_string(),
            shards,
        })
    }

    /// The header keys of the first shard, as `Content::metadata` holds them.
    pub fn metadata(&self) -> HashMap<String, Value> {
        self.shards
            .first()
            .map(|s| s.metadata.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Every tensor, every shard.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorShape> {
        self.shards.iter().flat_map(|s| s.tensors.iter())
    }

    /// Bytes of tensor data of each [`Bulk`] class: `(stems, experts, tables)`.
    pub fn bytes_by_bulk(&self) -> (u64, u64, u64) {
        let mut b = (0u64, 0u64, 0u64);
        for t in self.tensors() {
            let n = t.bytes() as u64;
            match Bulk::of(&t.name) {
                Bulk::Stem => b.0 += n,
                Bulk::Experts => b.1 += n,
                Bulk::Table => b.2 += n,
            }
        }
        b
    }

    pub fn to_json(&self) -> Json {
        let shards: Vec<Json> = self
            .shards
            .iter()
            .map(|s| {
                let md: Map<String, Json> = s
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_json(v)))
                    .collect();
                let tensors: Vec<Json> = s
                    .tensors
                    .iter()
                    .map(|t| json!([t.name, t.dims, format!("{:?}", t.dtype), t.offset]))
                    .collect();
                json!({ "file": s.file, "metadata": md, "tensors": tensors })
            })
            .collect();
        json!({ "format": FORMAT, "source": self.source, "shards": shards })
    }

    pub fn from_json(j: &Json) -> Result<Self> {
        if j["format"].as_str() != Some(FORMAT) {
            bail!("not a {FORMAT:?} file: format {:?}", j["format"])
        }
        let bad = |what: &str| Error::msg(format!("shape manifest: {what}"));
        let mut shards = Vec::new();
        for s in j["shards"].as_array().ok_or_else(|| bad("no shards"))? {
            let metadata = s["metadata"]
                .as_object()
                .ok_or_else(|| bad("no metadata"))?
                .iter()
                .map(|(k, v)| Ok((k.clone(), value_from_json(v)?)))
                .collect::<Result<Vec<_>>>()?;
            let tensors = s["tensors"]
                .as_array()
                .ok_or_else(|| bad("no tensors"))?
                .iter()
                .map(|t| {
                    let name = t[0].as_str().ok_or_else(|| bad("a tensor name"))?;
                    let dims = t[1]
                        .as_array()
                        .ok_or_else(|| bad("a tensor's dims"))?
                        .iter()
                        .map(|d| d.as_u64().map(|d| d as usize))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| bad("a dim"))?;
                    let kind = t[2].as_str().ok_or_else(|| bad("a tensor kind"))?;
                    Ok(TensorShape {
                        name: name.to_string(),
                        dims,
                        dtype: kind_named(kind)?,
                        offset: t[3].as_u64().ok_or_else(|| bad("an offset"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            shards.push(ShardShape {
                file: s["file"].as_str().ok_or_else(|| bad("a file"))?.to_string(),
                metadata,
                tensors,
            });
        }
        Ok(Self {
            source: j["source"].as_str().unwrap_or("").to_string(),
            shards,
        })
    }

    pub fn read(path: &Path) -> Result<Self> {
        let j: Json = serde_json::from_slice(&std::fs::read(path)?).map_err(Error::wrap)?;
        Self::from_json(&j)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut s = serde_json::to_string_pretty(&self.to_json()).map_err(Error::wrap)?;
        s.push('\n');
        std::fs::write(path, s)?;
        Ok(())
    }

    /// A stable 64-bit digest of the manifest and `syn`: the name a cached
    /// synthetic checkpoint lives under.
    pub fn digest(&self, syn: &Synthetic) -> u64 {
        let text = format!("{}{syn:?}", self.to_json());
        splitmix64(name_hash(&text))
    }

    /// Write the synthetic checkpoint into `dir`, one file per shard, and
    /// return their paths in shard order. Each tensor's values follow
    /// `role(name)` under `syn.law`; a stem is keyed values, the bulk is what
    /// `syn` says.
    pub fn write_sparse(
        &self,
        dir: &Path,
        syn: &Synthetic,
        role: &dyn Fn(&TensorShape) -> Role,
    ) -> Result<Vec<PathBuf>> {
        std::fs::create_dir_all(dir)?;
        let mut paths = Vec::with_capacity(self.shards.len());
        for s in &self.shards {
            let path = dir.join(&s.file);
            let mut f = std::fs::File::create(&path)?;
            let md: Vec<(&str, &Value)> = s.metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let header: Vec<HeaderTensor<'_>> = s
                .tensors
                .iter()
                .map(|t| HeaderTensor {
                    name: &t.name,
                    dims: &t.dims,
                    dtype: t.dtype,
                    offset: t.offset,
                })
                .collect();
            let start = gguf_file::write_header(&mut f, &md, &header)?;
            let mut end = start;
            for t in &s.tensors {
                let at = start + t.offset;
                let n = t.bytes();
                end = end.max(at + n as u64);
                let fill = match Bulk::of(&t.name) {
                    Bulk::Experts => syn.experts,
                    Bulk::Table => syn.tables,
                    Bulk::Stem => Fill::Pool,
                };
                if fill == Fill::Hole {
                    continue;
                }
                let block = pool_block(t, syn, role(t))?;
                f.seek(SeekFrom::Start(at))?;
                let mut left = n;
                while left > 0 {
                    let k = left.min(block.len());
                    f.write_all(&block[..k])?;
                    left -= k;
                }
            }
            // The holes past the last written byte: the file is its layout's
            // size whatever was written.
            f.set_len(end.div_ceil(32) * 32)?;
            paths.push(path);
        }
        Ok(paths)
    }

    /// [`Self::write_sparse`] under `root` (default: `LLOOM_SYNTHETIC_DIR`,
    /// else `~/.cache/lloom/synthetic`), reused when a complete copy for the
    /// same manifest and `syn` is already there. Written into a temporary
    /// directory and renamed, so a reader never sees a half-written one.
    pub fn cached(
        &self,
        syn: &Synthetic,
        role: &dyn Fn(&TensorShape) -> Role,
        root: Option<&Path>,
    ) -> Result<Vec<PathBuf>> {
        let root = match root {
            Some(r) => r.to_path_buf(),
            None => synthetic_root(),
        };
        let slug: String = self
            .source
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let dir = root.join(format!("{slug}-{:016x}", self.digest(syn)));
        let paths: Vec<PathBuf> = self.shards.iter().map(|s| dir.join(&s.file)).collect();
        if dir.join(".complete").exists() && paths.iter().all(|p| p.exists()) {
            return Ok(paths);
        }
        let tmp = root.join(format!(
            ".tmp-{}-{:016x}",
            std::process::id(),
            self.digest(syn)
        ));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        self.write_sparse(&tmp, syn, role)?;
        std::fs::write(tmp.join(".complete"), self.source.as_bytes())?;
        match std::fs::rename(&tmp, &dir) {
            Ok(()) => {}
            // Another process finished the same checkpoint first: use its.
            Err(_) if dir.join(".complete").exists() => std::fs::remove_dir_all(&tmp)?,
            Err(e) => return Err(e.into()),
        }
        Ok(paths)
    }
}

/// Where cached synthetic checkpoints live.
pub fn synthetic_root() -> PathBuf {
    if let Some(d) = std::env::var_os("LLOOM_SYNTHETIC_DIR") {
        return PathBuf::from(d);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".cache/lloom/synthetic")
}

/// The bytes repeated through `t`'s region: the whole tensor when it is
/// small, else whole rows up to [`POOL_BYTES`], keyed by the tensor's name.
fn pool_block(t: &TensorShape, syn: &Synthetic, role: Role) -> Result<Vec<u8>> {
    let row = *t.dims.last().unwrap_or(&1);
    let rows_total = t.dims.iter().product::<usize>() / row.max(1);
    let row_bytes = t.bytes() / rows_total.max(1);
    let rows = if t.bytes() <= WHOLE_BYTES {
        rows_total
    } else {
        (POOL_BYTES / row_bytes.max(1)).clamp(1, rows_total)
    };
    let shape = if t.dims.len() <= 1 {
        Shape::from(vec![row])
    } else {
        Shape::from(vec![rows, row])
    };
    let q = packed::qtensor(
        syn.law,
        syn.seed,
        &t.name,
        role,
        &shape,
        t.dtype,
        packed::Synthesis::Packed,
        &lloom_tensor::Device::Cpu,
    )?;
    Ok(q.data()?.into_owned())
}

/// The kind `name` (`format!("{:?}")`) names.
pub fn kind_named(name: &str) -> Result<GgmlDType> {
    ALL_KINDS
        .iter()
        .copied()
        .find(|k| format!("{k:?}") == name)
        .ok_or_else(|| Error::msg(format!("no GGUF kind is named {name:?}")))
}

/// A header value as `{type: value}`, so the manifest writes back the type
/// the real header stored, not the one JSON would infer.
fn value_to_json(v: &Value) -> Json {
    match v {
        Value::U8(x) => json!({ "u8": x }),
        Value::I8(x) => json!({ "i8": x }),
        Value::U16(x) => json!({ "u16": x }),
        Value::I16(x) => json!({ "i16": x }),
        Value::U32(x) => json!({ "u32": x }),
        Value::I32(x) => json!({ "i32": x }),
        Value::U64(x) => json!({ "u64": x }),
        Value::I64(x) => json!({ "i64": x }),
        Value::F32(x) => json!({ "f32": x.to_bits() }),
        Value::F64(x) => json!({ "f64": x.to_bits() }),
        Value::Bool(x) => json!({ "bool": x }),
        Value::String(x) => json!({ "string": x }),
        Value::Array(a) => json!({ "array": a.iter().map(value_to_json).collect::<Vec<_>>() }),
    }
}

fn value_from_json(j: &Json) -> Result<Value> {
    let bad = || Error::msg(format!("shape manifest: a header value {j}"));
    let (t, v) = j
        .as_object()
        .and_then(|o| o.iter().next())
        .ok_or_else(bad)?;
    let u = || v.as_u64().ok_or_else(bad);
    let i = || v.as_i64().ok_or_else(bad);
    Ok(match t.as_str() {
        "u8" => Value::U8(u()? as u8),
        "i8" => Value::I8(i()? as i8),
        "u16" => Value::U16(u()? as u16),
        "i16" => Value::I16(i()? as i16),
        "u32" => Value::U32(u()? as u32),
        "i32" => Value::I32(i()? as i32),
        "u64" => Value::U64(u()?),
        "i64" => Value::I64(i()?),
        "f32" => Value::F32(f32::from_bits(u()? as u32)),
        "f64" => Value::F64(f64::from_bits(u()?)),
        "bool" => Value::Bool(v.as_bool().ok_or_else(bad)?),
        "string" => Value::String(v.as_str().ok_or_else(bad)?.to_string()),
        "array" => Value::Array(
            v.as_array()
                .ok_or_else(bad)?
                .iter()
                .map(value_from_json)
                .collect::<Result<_>>()?,
        ),
        _ => return Err(bad()),
    })
}
