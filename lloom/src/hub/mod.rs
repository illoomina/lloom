//! Checkpoint loading: `Hub` -> `Checkpoint` -> `Pretrained<M>`.
//!
//! The engine loads models from two on-disk formats — a directory of
//! safetensors shards with a JSON config and a `tokenizer.json`, or a single
//! GGUF file that carries its config and tokenizer *inside* the weights
//! (`RESTRUCTURING_DESIGN.md` §3.6, B11). Both formats route through one
//! intermediate, [`Checkpoint`], so a caller — and the model builder above it —
//! sees one shape regardless of where the bytes came from.
//!
//! The layering:
//!
//! - [`Hub`] parameterises the *fetch*: [`HfHub`] pulls a repo from the Hugging
//!   Face hub into the local cache, [`LocalDir`] names a directory already on
//!   disk. A provider never parameterises the model.
//! - [`Checkpoint`] is the format-agnostic staging point: it holds the
//!   [`Weights`] (safetensors shards or a GGUF path), the raw config JSON, and —
//!   when the format carries them — the tokenizer, chat template and generation
//!   config. `from_pretrained` (safetensors) and `from_gguf` are two
//!   constructors of the SAME `Checkpoint`.
//! - [`Pretrained`] builds a [`crate::nn::Model`] from a `Checkpoint` and pairs
//!   it with the tokenizer and templates a serving loop needs.
//!
//! Deserialization is the checkpoint layer's concern, not the model trait's: a
//! model's *runtime* `Config` is often derived from a *wire* config rather than
//! deserialized directly (LFM2 recomputes `intermediate_size` in `into_config`,
//! `RESTRUCTURING_DESIGN.md` §3.1), so `Checkpoint` exposes the config as raw
//! JSON and the model-specific builder reconstructs the runtime config from it.
//!
//! Data is interoperable, code is not: no `trust_remote_code`, no Python.

use crate::nn::{Model, VarBuilder};
use lloom_tensor::{DType, Device, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Where a checkpoint's bytes came from, recorded for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A Hugging Face repo id (optionally at a revision).
    Hub { repo: String, revision: String },
    /// A directory already on disk.
    Local(PathBuf),
}

/// The weight files a checkpoint loads from.
///
/// Safetensors is one or more shards (a single-file checkpoint is a one-element
/// `shards`); GGUF is one self-describing file that also carries the tokenizer
/// and config, which is why the two formats cannot share a `Vec<PathBuf>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights {
    Safetensors { shards: Vec<PathBuf> },
    Gguf(PathBuf),
}

impl Weights {
    /// The paths this variant reads, for a caller that mmaps them itself.
    pub fn paths(&self) -> Vec<PathBuf> {
        match self {
            Weights::Safetensors { shards } => shards.clone(),
            Weights::Gguf(p) => vec![p.clone()],
        }
    }
}

/// A model's generation defaults, as `generation_config.json` carries them.
///
/// Only the fields a decode loop reads are named; the rest stay in `Checkpoint`'s
/// `extra` untouched. Every field is optional because the file is optional and
/// its keys are model-specific.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// `eos_token_id` is a scalar in some configs and a list in others; a decode
    /// loop wants the set, so both spellings collapse to a `Vec`.
    #[serde(default, deserialize_with = "de_u32_or_vec")]
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub min_p: Option<f64>,
}

fn de_u32_or_vec<'de, D>(de: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(u32),
        Many(Vec<u32>),
        None,
    }
    Ok(match OneOrMany::deserialize(de)? {
        OneOrMany::One(v) => vec![v],
        OneOrMany::Many(v) => v,
        OneOrMany::None => Vec::new(),
    })
}

/// A staged checkpoint: weights plus everything the loader recovered alongside
/// them, in a format-agnostic shape.
///
/// `config` is raw JSON on purpose — the model-specific builder deserializes its
/// wire config from it and derives the runtime config (see the module docs). The
/// tokenizer, chat template and generation config are `Option` because a bare
/// safetensors directory need not carry them and a GGUF always does.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub source: Source,
    pub weights: Weights,
    pub config: serde_json::Value,
    pub tokenizer: Option<Tokenizer>,
    pub chat_template: Option<String>,
    pub generation_config: Option<GenerationConfig>,
    /// Anything else the directory carried, keyed by file name, kept verbatim.
    pub extra: BTreeMap<String, Vec<u8>>,
}

impl Checkpoint {
    /// Build a checkpoint from a directory of safetensors shards.
    ///
    /// This is the load path `from_pretrained` routes through once a provider has
    /// materialised the files locally: it does not fetch, it reads what is on
    /// disk. `config.json` is required (a model cannot be built without it);
    /// everything else is optional and absent-means-`None`.
    pub fn from_safetensors_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = read_json(&dir.join("config.json"))?;
        let shards = safetensors_shards(dir)?;
        if shards.is_empty() {
            lloom_tensor::bail!(
                "no safetensors weights in {}: expected `model.safetensors` or a \
                 `model.safetensors.index.json` shard set",
                dir.display()
            );
        }
        let tokenizer = read_tokenizer_json(&dir.join("tokenizer.json"))?;
        let chat_template = read_chat_template(dir)?;
        let generation_config = read_generation_config(dir)?;
        Ok(Self {
            source: Source::Local(dir.to_path_buf()),
            weights: Weights::Safetensors { shards },
            config,
            tokenizer,
            chat_template,
            generation_config,
            extra: BTreeMap::new(),
        })
    }

    /// Build a checkpoint from a single GGUF file.
    ///
    /// GGUF carries its config and tokenizer inside the weights, so this opens
    /// the file, reads the metadata, reconstructs the tokenizer from it
    /// (`Tokenizer::from_checkpoint`'s GGUF arm), and lifts the scalar metadata
    /// into `config` as JSON so the same `Checkpoint` shape holds.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)
            .map_err(|e| lloom_tensor::Error::msg(format!("opening {}: {e}", path.display())))?;
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut file)?;
        let config = gguf_metadata_to_json(&content);
        let tokenizer = tokenizer_from_gguf(&content).ok();
        let chat_template = gguf_string(&content, "tokenizer.chat_template");
        Ok(Self {
            source: Source::Local(path.to_path_buf()),
            weights: Weights::Gguf(path.to_path_buf()),
            config,
            tokenizer,
            chat_template,
            generation_config: None,
            extra: BTreeMap::new(),
        })
    }

    /// Which of the optional parts this checkpoint carries. The order is fixed so
    /// a diff of two manifests is readable, and the tuple is `(name, present)`
    /// rather than a filtered list so an absent part is visible, not silent.
    pub fn manifest(&self) -> Vec<(&'static str, bool)> {
        vec![
            (
                "safetensors",
                matches!(self.weights, Weights::Safetensors { .. }),
            ),
            ("gguf", matches!(self.weights, Weights::Gguf(_))),
            ("config", !self.config.is_null()),
            ("tokenizer", self.tokenizer.is_some()),
            ("chat_template", self.chat_template.is_some()),
            ("generation_config", self.generation_config.is_some()),
        ]
    }

    /// A `VarBuilder` over this checkpoint's safetensors shards.
    ///
    /// Errors for a GGUF checkpoint: quantized weights are not loaded through a
    /// `VarBuilder`, they are read tensor-by-tensor by the quantized model's own
    /// `from_gguf` (`quantized_lfm2::ModelWeights::from_gguf`). The two model
    /// types are different, which is why `Pretrained<M: Model>` is the
    /// safetensors arm and the quantized arm builds directly from the file.
    ///
    /// # Safety
    ///
    /// mmaps the shard files; the caller must not mutate them for the lifetime of
    /// the returned `VarBuilder`. This is `VarBuilder::from_mmaped_safetensors`'
    /// own contract, forwarded.
    pub unsafe fn var_builder<'a>(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'a>> {
        match &self.weights {
            Weights::Safetensors { shards } => {
                VarBuilder::from_mmaped_safetensors(shards, dtype, device)
            }
            Weights::Gguf(_) => lloom_tensor::bail!(
                "a GGUF checkpoint is not loaded through a VarBuilder; build the \
                 quantized model from `weights.paths()` directly"
            ),
        }
    }
}

/// Fetch a checkpoint's files into a local directory, so a `Checkpoint` can read
/// them. The provider parameterises where the bytes come from and nothing else.
pub trait Hub {
    /// Resolve the checkpoint to a directory (or single file, for GGUF) on disk,
    /// fetching if necessary, and return a `Checkpoint` over it.
    fn checkpoint(&self, id: &str) -> Result<Checkpoint>;
}

/// A directory already on disk. No network; the offline provider.
///
/// `checkpoint(id)` joins `id` onto the base directory, so `""` names the base
/// itself. A GGUF is named by a path ending in `.gguf`; anything else is read as
/// a safetensors directory.
#[derive(Debug, Clone)]
pub struct LocalDir {
    pub base: PathBuf,
}

impl LocalDir {
    pub fn new(base: impl AsRef<Path>) -> Self {
        Self {
            base: base.as_ref().to_path_buf(),
        }
    }
}

impl Hub for LocalDir {
    fn checkpoint(&self, id: &str) -> Result<Checkpoint> {
        let path = if id.is_empty() {
            self.base.clone()
        } else {
            self.base.join(id)
        };
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            Checkpoint::from_gguf(path)
        } else {
            Checkpoint::from_safetensors_dir(path)
        }
    }
}

/// The Hugging Face hub. Pulls a repo into the local cache, then reads it as a
/// safetensors directory checkpoint.
///
/// A GGUF file is pulled by naming it as the `id`'s file component (a caller that
/// wants a specific `.gguf` uses [`HfHub::checkpoint_gguf`]).
#[derive(Debug, Clone)]
pub struct HfHub {
    pub revision: String,
}

impl Default for HfHub {
    fn default() -> Self {
        Self {
            revision: "main".to_string(),
        }
    }
}

impl HfHub {
    /// Pull a specific GGUF file from a repo and stage it as a checkpoint.
    pub fn checkpoint_gguf(&self, repo_str: &str, file: &str) -> Result<Checkpoint> {
        let api = hf_hub::api::sync::Api::new().map_err(lloom_tensor::Error::wrap)?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            repo_str.to_string(),
            hf_hub::RepoType::Model,
            self.revision.clone(),
        ));
        let path = repo.get(file).map_err(lloom_tensor::Error::wrap)?;
        let repo_id = repo_str.to_string();
        let mut cp = Checkpoint::from_gguf(path)?;
        cp.source = Source::Hub {
            repo: repo_id,
            revision: self.revision.clone(),
        };
        Ok(cp)
    }
}

impl Hub for HfHub {
    fn checkpoint(&self, id: &str) -> Result<Checkpoint> {
        let api = hf_hub::api::sync::Api::new().map_err(lloom_tensor::Error::wrap)?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            id.to_string(),
            hf_hub::RepoType::Model,
            self.revision.clone(),
        ));
        // `config.json` anchors the snapshot directory; the shards sit beside it.
        let config_path = repo.get("config.json").map_err(lloom_tensor::Error::wrap)?;
        let dir = config_path
            .parent()
            .ok_or_else(|| lloom_tensor::Error::msg("config.json has no parent directory"))?;
        // Fetch the shards named by the index (or the single-file weights) so
        // they are present in the cache before `from_safetensors_dir` reads them.
        fetch_safetensors_shards(&repo, dir)?;
        let mut cp = Checkpoint::from_safetensors_dir(dir)?;
        cp.source = Source::Hub {
            repo: id.to_string(),
            revision: self.revision.clone(),
        };
        Ok(cp)
    }
}

/// A loaded model, paired with the tokenizer and templates a serving loop needs.
///
/// Generic over `M: Model` — the safetensors arm. The quantized (GGUF) model is
/// a different type that does not implement `Model`, so it is built from the
/// checkpoint's `weights.paths()` directly rather than through `Pretrained`.
pub struct Pretrained<M: Model> {
    pub model: M,
    pub tokenizer: Option<Tokenizer>,
    pub chat_template: Option<String>,
    pub generation_config: Option<GenerationConfig>,
    pub source: Source,
}

impl<M: Model> Pretrained<M> {
    /// Build a model from a safetensors checkpoint and the runtime config the
    /// model-specific builder derived from `checkpoint.config`.
    ///
    /// The `config` is the model's *runtime* `M::Config`, already reconstructed
    /// from `checkpoint.config` (the wire JSON) by the caller — this layer does
    /// not know how a given model derives its runtime config, only that
    /// `M::from_weights` takes it.
    ///
    /// # Safety
    ///
    /// mmaps the shards; see [`Checkpoint::var_builder`].
    pub unsafe fn from_checkpoint(
        checkpoint: Checkpoint,
        config: &M::Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = checkpoint.var_builder(dtype, device)?;
        let model = M::from_weights(vb, config)?;
        Ok(Self {
            model,
            tokenizer: checkpoint.tokenizer,
            chat_template: checkpoint.chat_template,
            generation_config: checkpoint.generation_config,
            source: checkpoint.source,
        })
    }
}

/// A tokenizer recovered from a checkpoint, whatever its format.
///
/// For safetensors the tokenizer is a `tokenizer.json` file already parsed into
/// `Checkpoint::tokenizer`; for GGUF it was reconstructed from the weights'
/// metadata. `from_checkpoint` returns the one the checkpoint carries, erroring
/// only when neither format supplied one.
pub trait TokenizerFromCheckpoint: Sized {
    fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self>;
}

impl TokenizerFromCheckpoint for Tokenizer {
    fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        checkpoint
            .tokenizer
            .clone()
            .ok_or_else(|| lloom_tensor::Error::msg("checkpoint carries no tokenizer"))
    }
}

/// `file` of the Hugging Face repo `repo` at its `main` ref, if the local
/// cache already holds it (`HF_HOME` honoured). Never fetches: a test or a
/// harness asks whether a checkpoint is on this machine, and a missing one
/// is an answer, not a download.
pub fn cached_file(repo: &str, file: &str) -> Option<PathBuf> {
    hf_hub::Cache::from_env()
        .model(repo.to_string())
        .get(file)
        .filter(|p| p.exists())
}

/// Every shard of a split GGUF, from its first (`…-00001-of-0000N.gguf`):
/// the siblings the name promises, in order, as far as they exist. A file
/// not named as a split is its own only shard.
pub fn gguf_shards(first: &Path) -> Vec<PathBuf> {
    let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let split = name.rsplit_once("-of-").and_then(|(head, tail)| {
        let count: usize = tail.strip_suffix(".gguf")?.parse().ok()?;
        let head = head.get(..head.len().checked_sub(5)?)?;
        Some((head.to_string(), count))
    });
    match split {
        Some((head, count)) => (1..=count)
            .map(|k| first.with_file_name(format!("{head}{k:05}-of-{count:05}.gguf")))
            .take_while(|p| p.exists())
            .collect(),
        None => vec![first.to_path_buf()],
    }
}

// ---------------------------------------------------------------------------
// Readers.
// ---------------------------------------------------------------------------

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| lloom_tensor::Error::msg(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&raw)
        .map_err(|e| lloom_tensor::Error::msg(format!("parsing {}: {e}", path.display())))
}

fn read_tokenizer_json(path: &Path) -> Result<Option<Tokenizer>> {
    if !path.exists() {
        return Ok(None);
    }
    Tokenizer::from_file(path)
        .map(Some)
        .map_err(|e| lloom_tensor::Error::msg(format!("parsing {}: {e}", path.display())))
}

/// The chat template lives either in its own `chat_template.jinja` /
/// `chat_template.json`, or as a `chat_template` field of `tokenizer_config.json`.
fn read_chat_template(dir: &Path) -> Result<Option<String>> {
    let jinja = dir.join("chat_template.jinja");
    if jinja.exists() {
        return std::fs::read_to_string(&jinja)
            .map(Some)
            .map_err(|e| lloom_tensor::Error::msg(format!("reading {}: {e}", jinja.display())));
    }
    let tok_cfg = dir.join("tokenizer_config.json");
    if tok_cfg.exists() {
        let v = read_json(&tok_cfg)?;
        if let Some(t) = v.get("chat_template").and_then(|t| t.as_str()) {
            return Ok(Some(t.to_string()));
        }
    }
    Ok(None)
}

/// `generation_config.json` in `dir`, when present (the sampling defaults a
/// serving binary may take under `--sampling-defaults checkpoint`).
pub fn read_generation_config(dir: &Path) -> Result<Option<GenerationConfig>> {
    let path = dir.join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let v = read_json(&path)?;
    let gc: GenerationConfig = serde_json::from_value(v)
        .map_err(|e| lloom_tensor::Error::msg(format!("parsing {}: {e}", path.display())))?;
    Ok(Some(gc))
}

/// The safetensors shards in a directory: the shards named by
/// `model.safetensors.index.json` if present, else the single `model.safetensors`.
fn safetensors_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v = read_json(&index)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                lloom_tensor::Error::msg(format!("{}: no `weight_map` object", index.display()))
            })?;
        // A shard is named once per tensor it holds; dedup and sort for a stable
        // load order.
        let mut files: Vec<String> = map
            .values()
            .filter_map(|f| f.as_str().map(str::to_string))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Ok(Vec::new())
}

/// Fetch every safetensors shard referenced by the index (or the single file)
/// into the cache. `get` is a no-op fetch when the file is already present.
fn fetch_safetensors_shards(repo: &hf_hub::api::sync::ApiRepo, dir: &Path) -> Result<()> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v = read_json(&index)?;
        if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
            let mut names: Vec<&str> = map.values().filter_map(|f| f.as_str()).collect();
            names.sort();
            names.dedup();
            for name in names {
                repo.get(name).map_err(lloom_tensor::Error::wrap)?;
            }
        }
    } else {
        repo.get("model.safetensors")
            .map_err(lloom_tensor::Error::wrap)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GGUF metadata.
// ---------------------------------------------------------------------------

fn tokenizer_from_gguf(content: &lloom_tensor::quantized::gguf_file::Content) -> Result<Tokenizer> {
    use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
    Tokenizer::from_gguf(content)
}

fn gguf_string(content: &lloom_tensor::quantized::gguf_file::Content, key: &str) -> Option<String> {
    content
        .metadata
        .get(key)
        .and_then(|v| v.to_string().ok())
        .cloned()
}

/// Lift the GGUF's scalar and short-array metadata into a JSON object, so a
/// `Checkpoint` from either format exposes its config the same way. Tensor data
/// is not touched — only the metadata table.
fn gguf_metadata_to_json(
    content: &lloom_tensor::quantized::gguf_file::Content,
) -> serde_json::Value {
    use lloom_tensor::quantized::gguf_file::Value as GV;
    fn conv(v: &GV) -> serde_json::Value {
        use serde_json::Value as J;
        match v {
            GV::U8(x) => J::from(*x),
            GV::I8(x) => J::from(*x),
            GV::U16(x) => J::from(*x),
            GV::I16(x) => J::from(*x),
            GV::U32(x) => J::from(*x),
            GV::I32(x) => J::from(*x),
            GV::U64(x) => J::from(*x),
            GV::I64(x) => J::from(*x),
            GV::F32(x) => serde_json::Number::from_f64(*x as f64)
                .map(J::Number)
                .unwrap_or(J::Null),
            GV::F64(x) => serde_json::Number::from_f64(*x)
                .map(J::Number)
                .unwrap_or(J::Null),
            GV::Bool(x) => J::from(*x),
            GV::String(s) => J::from(s.clone()),
            GV::Array(a) => J::Array(a.iter().map(conv).collect()),
        }
    }
    let map: serde_json::Map<String, serde_json::Value> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), conv(v)))
        .collect();
    serde_json::Value::Object(map)
}
