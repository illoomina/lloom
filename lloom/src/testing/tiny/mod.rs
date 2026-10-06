//! Tiny models (`docs/porting-models.md` §21.8): a family's exact topology at
//! the smallest geometry that reaches the kernel arms under test, with keyed
//! weights, in the family's real file formats.
//!
//! - [`TinyField`] / [`TinyConfig`]: the `#[tiny(...)]` annotations a
//!   family's config carries on `#[derive(Config)]`, and the surface the
//!   derive emits for them — what a shape search may mutate.
//! - [`init`]: keyed weights, `f(seed, name, index)`, under a [`ValueLaw`].
//! - [`packed`]: quantized weights synthesized as packed blocks, every kind.
//! - [`backend`]: the recording backend — a constructor's requests are the
//!   schema of a family that loads through a `VarBuilder`.
//! - [`tokenizer`]: 256 bytes, the family's added tokens and chat template.
//! - [`TinyFamily`] / [`TinyCheckpoint`]: one source, three outputs — a
//!   model in memory and files the real loader reads (a trainable map is
//!   the third, not built).

pub mod backend;
pub mod init;
pub mod packed;
pub mod tokenizer;

pub use init::{Role, ValueLaw};
pub use packed::Synthesis;
pub use tokenizer::TinyTokenizer;

use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{bail, DType, Device, Result, Shape};
use std::path::{Path, PathBuf};

/// What the shape search may do with one config field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TinyRole {
    /// A searched dimension.
    Axis,
    /// Kept at the template's value.
    Fixed,
    /// Recomputed from the other fields; the text is the expression.
    Derived(&'static str),
}

/// One `#[tiny]`-tagged field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyField {
    pub name: &'static str,
    pub role: TinyRole,
    /// Every layer kind the template has must survive a mutation.
    pub cover: bool,
    /// The largest value a search may set (an axis only).
    pub max: Option<usize>,
}

/// The surface `#[derive(Config)]` emits for a config with `#[tiny]`
/// fields, as a trait so a family's campaign is generic over it.
pub trait TinyConfig: Clone + std::fmt::Debug {
    /// The annotations, in field order.
    fn fields() -> &'static [TinyField];
    /// The value of the axis `name`.
    fn axis(&self, name: &str) -> Option<usize>;
    /// Set the axis `name`; `false` if there is none. Derived fields are
    /// stale until [`TinyConfig::derive`].
    fn set_axis(&mut self, name: &str, value: usize) -> bool;
    /// Recompute every derived field.
    fn derive(&mut self);

    /// The searched axes, `(name, largest value)`, in field order;
    /// `default_max` for an axis without its own.
    fn axes(default_max: usize) -> Vec<(&'static str, usize)> {
        Self::fields()
            .iter()
            .filter(|f| f.role == TinyRole::Axis)
            .map(|f| (f.name, f.max.unwrap_or(default_max)))
            .collect()
    }
}

/// One tensor of a checkpoint: its name, its shape (row-major, as the
/// constructor asks for it), what it is to the value law, and what the file
/// stores it as.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub role: Role,
    pub dtype: GgmlDType,
}

/// Every tensor a family's constructor reads at one config.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Schema {
    pub tensors: Vec<TensorSpec>,
}

impl Schema {
    /// Weight elements, summed.
    pub fn elements(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum()
    }

    pub fn get(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Check a checkpoint's tensor table against the schema: every tensor it
    /// names is present with its shape. Every discrepancy is reported at
    /// once, by name; tensors the schema does not name are not its business
    /// (an optional head, a table the load did not ask for).
    pub fn validate<'a>(
        &self,
        have: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    ) -> Result<()> {
        let have: std::collections::HashMap<&str, &[usize]> = have.into_iter().collect();
        let mut bad = Vec::new();
        for t in &self.tensors {
            match have.get(t.name.as_str()) {
                None => bad.push(format!("  missing: {} {:?}", t.name, t.shape)),
                Some(s) if *s != t.shape.as_slice() => bad.push(format!(
                    "  shape: {} is {s:?}, the schema says {:?}",
                    t.name, t.shape
                )),
                Some(_) => {}
            }
        }
        if !bad.is_empty() {
            bail!(
                "the checkpoint does not match the tensor schema ({} of {} tensors):\n{}",
                bad.len(),
                self.tensors.len(),
                bad.join("\n")
            )
        }
        Ok(())
    }
}

/// How a tiny checkpoint's weights are made and stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub law: ValueLaw,
    pub seed: u64,
    /// What the dense matrices are stored as.
    pub dense: GgmlDType,
    /// What the routed expert stacks are stored as (MoE families).
    pub experts: GgmlDType,
    /// How a packed kind is produced.
    pub synthesis: Synthesis,
    /// Store every packed tensor as its dequantized F32: the CPU reference's
    /// checkpoint, for kinds the CPU cannot multiply by.
    pub dequantized: bool,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            law: ValueLaw::DEFAULT,
            seed: 0x7131,
            dense: GgmlDType::F32,
            experts: GgmlDType::F32,
            synthesis: Synthesis::Packed,
            dequantized: false,
        }
    }
}

/// What a run feeds a model beside its geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workload {
    /// Prompt tokens, run as one prefill.
    pub prompt: usize,
    /// Decode steps after it, one token each.
    pub decode: usize,
}

impl Workload {
    /// The tokens a workload feeds a model of `vocab` ids: the prompt, then
    /// one token per decode step. Fixed, so a manifest entry replays.
    pub fn tokens(&self, vocab: usize) -> (Vec<u32>, Vec<u32>) {
        let v = vocab.max(1) as u32;
        let prompt = (0..self.prompt as u32).map(|i| (i * 7 + 3) % v).collect();
        let decode = (0..self.decode as u32).map(|j| (j * 5 + 1) % v).collect();
        (prompt, decode)
    }
}

/// A served model family at tiny geometry.
pub trait TinyFamily: Sized {
    type Config: TinyConfig;
    /// The family at a profile: a checkpoint has no profile, a built model
    /// names one (`profiles-and-memory-design.md` §3; #892 B3).
    type Model<P: crate::profile::Profile>: crate::nn::model::CausalLM<Config = Self::Config>;
    /// The family's name (`lloom`'s family names).
    const NAME: &'static str;
    /// The kinds a dense matrix may be stored as.
    const DENSE_KINDS: &'static [GgmlDType];
    /// The kinds a routed expert stack may be stored as (empty: no experts).
    const EXPERT_KINDS: &'static [GgmlDType];

    /// The smallest config that has every block type of the family.
    fn template() -> Self::Config;
    /// Whether `cfg` keeps every layer kind `template` has
    /// (`#[tiny(cover)]`).
    fn covers(template: &Self::Config, cfg: &Self::Config) -> bool;
    /// The tokenizer its files carry.
    fn tokenizer() -> TinyTokenizer;
    /// Every tensor the family's loader reads at `ckpt`'s config, stored as
    /// `ckpt.weights` says.
    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<Schema>;
    /// The workloads that reach the family's arms at `cfg`.
    fn workloads(cfg: &Self::Config) -> Vec<Workload>;
    /// The model in memory.
    fn build<P: crate::profile::Profile>(
        ckpt: &TinyCheckpoint<Self>,
        dev: &Device,
    ) -> Result<Self::Model<P>>;
    /// The checkpoint's files in `dir`; the path `lloom --model` takes.
    fn write(ckpt: &TinyCheckpoint<Self>, dir: &Path) -> Result<PathBuf>;
    /// A written checkpoint, through the family's real loader.
    fn load<P: crate::profile::Profile>(path: &Path, dev: &Device) -> Result<Self::Model<P>>;

    /// The CPU oracle for `ckpt`: the same values, every packed tensor
    /// dequantized, computed at f32.
    fn reference<P: crate::profile::Profile>(
        ckpt: &TinyCheckpoint<Self>,
        dev: &Device,
    ) -> Result<Self::Model<P>> {
        Self::build::<P>(&ckpt.dequantized().with_dtype(DType::F32), dev)
    }

    /// Runtime arms a campaign varies beside the geometry (an attention
    /// kernel, a KV layout), by name; the first is the family's default.
    const ARMS: &'static [&'static str] = &["default"];
    /// Select arm `arm` (an index into [`TinyFamily::ARMS`]) on `cfg`.
    fn set_arm(_cfg: &mut Self::Config, _arm: usize) {}

    /// `template()` with `axes` (in [`TinyConfig::axes`] order) applied and
    /// the derived fields recomputed; `None` if it drops a layer kind.
    fn apply(axes: &[usize]) -> Option<Self::Config> {
        let template = Self::template();
        let mut cfg = template.clone();
        for ((name, _), v) in Self::Config::axes(usize::MAX).iter().zip(axes) {
            cfg.set_axis(name, *v);
        }
        cfg.derive();
        Self::covers(&template, &cfg).then_some(cfg)
    }
}

/// One tiny checkpoint: a config and how its weights are made. Every output
/// is derived from these two alone.
pub struct TinyCheckpoint<F: TinyFamily> {
    pub cfg: F::Config,
    pub weights: Weights,
    /// The dtype the in-memory model computes in; `None` is the family's
    /// rule for the device (what its engine loads with).
    pub dtype: Option<DType>,
}

impl<F: TinyFamily> Clone for TinyCheckpoint<F> {
    fn clone(&self) -> Self {
        Self {
            cfg: self.cfg.clone(),
            weights: self.weights,
            dtype: self.dtype,
        }
    }
}

impl<F: TinyFamily> std::fmt::Debug for TinyCheckpoint<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TinyCheckpoint")
            .field("family", &F::NAME)
            .field("cfg", &self.cfg)
            .field("weights", &self.weights)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl<F: TinyFamily> TinyCheckpoint<F> {
    pub fn new(cfg: F::Config, weights: Weights) -> Self {
        Self {
            cfg,
            weights,
            dtype: None,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// The template with default weights.
    pub fn template() -> Self {
        Self::new(F::template(), Weights::default())
    }

    pub fn schema(&self) -> Result<Schema> {
        F::schema(self)
    }

    /// The model in memory (unit tests), at the profile the test names.
    pub fn model<P: crate::profile::Profile>(&self, dev: &Device) -> Result<F::Model<P>> {
        F::build::<P>(self, dev)
    }

    /// The files (loader and e2e tests); the path `lloom --model` takes.
    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        F::write(self, dir)
    }

    /// The same checkpoint with every packed tensor stored as its
    /// dequantized F32 — what a CPU reference runs.
    pub fn dequantized(&self) -> Self {
        let mut c = self.clone();
        c.weights.dequantized = true;
        c
    }

    /// One tensor of the schema, as the file stores it, on `dev`.
    pub fn qtensor(&self, spec: &TensorSpec, dev: &Device) -> Result<QTensor> {
        let w = &self.weights;
        let shape = Shape::from(spec.shape.clone());
        if w.dequantized && packed::is_packed(spec.dtype) {
            let q = packed::qtensor(
                w.law,
                w.seed,
                &spec.name,
                spec.role,
                &shape,
                spec.dtype,
                w.synthesis,
                &Device::Cpu,
            )?;
            let v = q
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            return QTensor::new(
                QStorage::from_data(std::borrow::Cow::Owned(bytes), dev, GgmlDType::F32)?,
                shape,
            );
        }
        packed::qtensor(
            w.law,
            w.seed,
            &spec.name,
            spec.role,
            &shape,
            spec.dtype,
            w.synthesis,
            dev,
        )
    }
}

/// The logits of `w` on `model`: the prefill's last position, then each
/// decode step's, as f32 — what the tiny-model tests and the campaign
/// compare.
pub fn run<M: crate::nn::model::CausalLM>(
    model: &M,
    cfg: &M::Config,
    vocab: usize,
    dev: &Device,
    w: &Workload,
) -> Result<Vec<Vec<f32>>> {
    use lloom_tensor::Tensor;
    let mut cache = model.new_cache(cfg, dev)?;
    let (prompt, decode) = w.tokens(vocab);
    let mut out = Vec::with_capacity(1 + decode.len());
    let flat =
        |t: Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>() };
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev)?;
    out.push(flat(model.forward(&input, 0, &mut cache)?)?);
    for (pos, t) in (prompt.len()..).zip(decode) {
        let input = Tensor::from_vec(vec![t], (1, 1), dev)?;
        out.push(flat(model.forward(&input, pos, &mut cache)?)?);
    }
    Ok(out)
}

/// One candidate of a shape search, in a family's terms: the axes (in
/// [`TinyConfig::axes`] order), a workload, the stored kinds and the
/// runtime arm. The byte form is positional and total — a missing field keeps
/// the template's value, every value is clamped into range — so any byte
/// string is a geometry and a campaign's input, a manifest entry and a
/// drift test's case are the same bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub axes: Vec<usize>,
    pub work: Workload,
    /// An index into [`TinyFamily::DENSE_KINDS`].
    pub dense: usize,
    /// An index into [`TinyFamily::EXPERT_KINDS`] (0 without experts).
    pub experts: usize,
    /// An index into [`TinyFamily::ARMS`].
    pub arm: usize,
}

/// The longest prompt and the most decode steps a geometry's workload runs.
pub const MAX_PROMPT: usize = 64;
pub const MAX_DECODE: usize = 4;
/// The largest value an axis without its own `max` may take.
pub const AXIS_CAP: usize = 512;

impl Geometry {
    /// The family's template with `work`, the default kinds and arm.
    pub fn template<F: TinyFamily>(work: Workload) -> Self {
        let t = F::template();
        Self {
            axes: F::Config::axes(AXIS_CAP)
                .iter()
                .map(|(n, _)| t.axis(n).expect("an axis"))
                .collect(),
            work,
            dense: 0,
            experts: 0,
            arm: 0,
        }
    }

    /// Little-endian `u16` per axis, then the prompt, the decode steps, the
    /// dense kind, the expert kind and the arm, a byte each.
    pub fn decode<F: TinyFamily>(bytes: &[u8]) -> Self {
        let mut g = Self::template::<F>(Workload {
            prompt: 7,
            decode: 1,
        });
        let mut at = 0usize;
        let mut next = |n: usize| -> Option<&[u8]> {
            let s = bytes.get(at..at + n)?;
            at += n;
            Some(s)
        };
        for (v, (_, max)) in g.axes.iter_mut().zip(F::Config::axes(AXIS_CAP)) {
            if let Some(b) = next(2) {
                *v = (u16::from_le_bytes([b[0], b[1]]) as usize).clamp(1, max);
            }
        }
        let modulo = |b: u8, n: usize| (b as usize) % n.max(1);
        if let Some(b) = next(1) {
            g.work.prompt = (b[0] as usize).clamp(1, MAX_PROMPT);
        }
        if let Some(b) = next(1) {
            g.work.decode = (b[0] as usize).min(MAX_DECODE);
        }
        if let Some(b) = next(1) {
            g.dense = modulo(b[0], F::DENSE_KINDS.len());
        }
        if let Some(b) = next(1) {
            g.experts = modulo(b[0], F::EXPERT_KINDS.len());
        }
        if let Some(b) = next(1) {
            g.arm = modulo(b[0], F::ARMS.len());
        }
        g
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.axes.len() * 2 + 5);
        for v in &self.axes {
            out.extend_from_slice(&(*v as u16).to_le_bytes());
        }
        out.extend_from_slice(&[
            self.work.prompt as u8,
            self.work.decode as u8,
            self.dense as u8,
            self.experts as u8,
            self.arm as u8,
        ]);
        out
    }

    /// The checkpoint this geometry names under `weights`' law and seed, or
    /// `None` if it drops a layer kind the template has.
    pub fn checkpoint<F: TinyFamily>(&self, weights: Weights) -> Option<TinyCheckpoint<F>> {
        let mut cfg = F::apply(&self.axes)?;
        F::set_arm(&mut cfg, self.arm);
        let kind = |kinds: &[GgmlDType], i: usize| kinds.get(i).copied().unwrap_or(GgmlDType::F32);
        Some(TinyCheckpoint::new(
            cfg,
            Weights {
                dense: kind(F::DENSE_KINDS, self.dense),
                experts: kind(F::EXPERT_KINDS, self.experts),
                ..weights
            },
        ))
    }

    /// `name=value` for every axis, then the workload, the kinds, the arm.
    pub fn describe<F: TinyFamily>(&self) -> String {
        let mut s: Vec<String> = F::Config::axes(AXIS_CAP)
            .iter()
            .zip(&self.axes)
            .map(|((n, _), v)| format!("{n}={v}"))
            .collect();
        s.push(format!(
            "prompt={} decode={}",
            self.work.prompt, self.work.decode
        ));
        s.push(format!(
            "dense={:?}",
            F::DENSE_KINDS
                .get(self.dense)
                .copied()
                .unwrap_or(GgmlDType::F32)
        ));
        if !F::EXPERT_KINDS.is_empty() {
            s.push(format!(
                "experts={:?}",
                F::EXPERT_KINDS[self.experts.min(F::EXPERT_KINDS.len() - 1)]
            ));
        }
        s.push(format!(
            "arm={}",
            F::ARMS.get(self.arm).copied().unwrap_or("?")
        ));
        s.join(" ")
    }
}
