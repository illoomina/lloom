//! Dual-backend fixtures (`docs/porting-models.md` §21.8a): a test that uses
//! a model says what it needs — a family and the geometry it runs at — and
//! runs on the tiny model by default. One switch, `LLOOM_FIXTURE`, runs the
//! same test at the family's real scale:
//!
//! - `tiny` (the default): the test's own geometry, keyed weights;
//! - `real`: the family's checkpoint when this machine has it (the family's
//!   variable first, then the Hugging Face cache) — skipped, with a line
//!   saying so, when it does not;
//! - `synthetic`: the real checkpoint's geometry and layout with keyed
//!   values (`super::scale`); needs no weights.
//!
//! At scale a test keeps the assertions that hold at any geometry — arms
//! against each other, a run against itself, a mechanism's own counters —
//! and gates the references that are the tiny model's own on
//! [`Fixture::at_scale`].
//!
//! Every fixture prints one `FIXTURE` line when it is dropped: the backend,
//! the checkpoint, the wall time and the device's memory over the test. The
//! periodic run (`lloom/tests/scale_fixtures.sh`) records those lines.

use super::tiny::{TinyCheckpoint, TinyFamily, Weights};
use crate::nn::memory::MemGauge;
use lloom_tensor::{DType, Device, Result};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

/// The switch's name.
pub const SWITCH: &str = "LLOOM_FIXTURE";

/// What a fixture runs its model from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Tiny,
    Synthetic,
    Real,
}

impl Backend {
    pub const ALL: [Backend; 3] = [Backend::Tiny, Backend::Synthetic, Backend::Real];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Tiny => "tiny",
            Backend::Synthetic => "synthetic",
            Backend::Real => "real",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == s)
    }

    /// The backend [`SWITCH`] names, `tiny` when it is unset; read once. An
    /// unknown value is a mistyped switch, and running the tiny model under
    /// it would report a real-model run that never happened — so it panics.
    pub fn current() -> Self {
        static CURRENT: OnceLock<Backend> = OnceLock::new();
        *CURRENT.get_or_init(|| match std::env::var(SWITCH) {
            Err(_) => Backend::Tiny,
            Ok(v) if v.is_empty() => Backend::Tiny,
            Ok(v) => Self::parse(&v)
                .unwrap_or_else(|| panic!("{SWITCH}={v:?}: expected one of tiny, synthetic, real")),
        })
    }

    /// Whether the model is at the real checkpoint's geometry.
    pub fn at_scale(self) -> bool {
        self != Backend::Tiny
    }
}

/// Where a fixture at scale reads its checkpoint from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// The family's real checkpoint: a directory, or a GGUF's first shard.
    Real(PathBuf),
    /// A synthetic checkpoint of the real one's geometry.
    Synthetic,
}

/// A served family whose tests also run at its real checkpoint's scale.
pub trait Scaled: TinyFamily {
    /// The variable that names the real checkpoint, ahead of the cache.
    const CHECKPOINT_ENV: &'static str;
    /// The Hugging Face repo and the file in it that locates the checkpoint.
    const REPO: (&'static str, &'static str);

    /// The checkpoint a cached [`Scaled::REPO`] file stands for (its
    /// directory, for a family whose checkpoint is one).
    fn checkpoint_of(file: PathBuf) -> PathBuf {
        file
    }

    /// The real checkpoint, if this machine has it.
    fn locate() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os(Self::CHECKPOINT_ENV) {
            let p = PathBuf::from(p);
            return p.exists().then_some(p);
        }
        crate::hub::cached_file(Self::REPO.0, Self::REPO.1).map(Self::checkpoint_of)
    }

    /// The geometry `source` holds, with the family's default arms.
    fn scale_config(source: &Source) -> Result<Self::Config>;

    /// The model at `cfg` from `source`, computing in `dtype` (`None`: the
    /// family's rule for the device). A synthetic one draws its values from
    /// `weights`' law and seed.
    fn scale_model(
        source: &Source,
        cfg: &Self::Config,
        dtype: Option<DType>,
        weights: &Weights,
        dev: &Device,
    ) -> Result<Self::Model>;
}

/// One test's model: its need, the backend the switch chose, and the
/// record of what the test cost.
pub struct Fixture<F: Scaled> {
    test: String,
    backend: Backend,
    source: Option<Source>,
    geometry: F::Config,
    weights: Weights,
    started: Instant,
    gauge: Mutex<Option<(Device, MemGauge)>>,
    scratch: Mutex<Vec<PathBuf>>,
}

impl<F: Scaled> Fixture<F> {
    /// The fixture for `test`, which runs at `tiny` by default. `None` when
    /// the switch asks for the real checkpoint and this machine has none:
    /// the test returns, and the line printed says why.
    pub fn new(test: &str, tiny: F::Config) -> Result<Option<Self>> {
        let backend = Backend::current();
        let source = match backend {
            Backend::Tiny => None,
            Backend::Synthetic => Some(Source::Synthetic),
            Backend::Real => match F::locate() {
                Some(p) => Some(Source::Real(p)),
                None => {
                    eprintln!(
                        "FIXTURE test={test} family={} backend=real SKIPPED: no checkpoint ({} unset; {} {} not in the Hugging Face cache)",
                        F::NAME,
                        F::CHECKPOINT_ENV,
                        F::REPO.0,
                        F::REPO.1
                    );
                    return Ok(None);
                }
            },
        };
        let geometry = match &source {
            None => tiny,
            Some(s) => F::scale_config(s)?,
        };
        Ok(Some(Self {
            test: test.to_string(),
            backend,
            source,
            geometry,
            weights: Weights::default(),
            started: Instant::now(),
            gauge: Mutex::new(None),
            scratch: Mutex::new(Vec::new()),
        }))
    }

    /// The weights' law, seed and kinds: the tiny model's, and the values of
    /// a synthetic one. A real checkpoint's are its own.
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn at_scale(&self) -> bool {
        self.backend.at_scale()
    }

    pub fn source(&self) -> Option<&Source> {
        self.source.as_ref()
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn test(&self) -> &str {
        &self.test
    }

    /// The geometry the model runs at: the test's tiny one, or the
    /// checkpoint's.
    pub fn geometry(&self) -> &F::Config {
        &self.geometry
    }

    /// The geometry with the test's arms applied.
    pub fn config(&self, arms: impl FnOnce(&mut F::Config)) -> F::Config {
        let mut cfg = self.geometry.clone();
        arms(&mut cfg);
        cfg
    }

    /// `tiny` on the tiny model, `scale` at the real one's: a count, a
    /// length or a pool size the test sizes to the geometry.
    pub fn pick<T>(&self, tiny: T, scale: T) -> T {
        if self.at_scale() {
            scale
        } else {
            tiny
        }
    }

    /// The device to run on: the test's, except that a model at scale runs
    /// on Metal — the CPU at a real geometry is hours. `None` (with a line)
    /// when there is no Metal device to run it on.
    pub fn device(&self, tiny: Device) -> Option<Device> {
        if !self.at_scale() || !tiny.is_cpu() {
            return Some(tiny);
        }
        match Device::new_metal(0) {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!(
                    "FIXTURE test={} family={} backend={} SKIPPED: a model at scale needs a Metal device",
                    self.test,
                    F::NAME,
                    self.backend.name()
                );
                None
            }
        }
    }

    /// The model at `cfg` on `dev`, computing in the family's dtype for the
    /// device.
    pub fn model(&self, cfg: &F::Config, dev: &Device) -> Result<F::Model> {
        self.model_in(cfg, None, dev)
    }

    /// The model at `cfg` on `dev`, computing in `dtype`.
    pub fn model_in(
        &self,
        cfg: &F::Config,
        dtype: Option<DType>,
        dev: &Device,
    ) -> Result<F::Model> {
        self.watch(dev);
        match &self.source {
            None => {
                let mut ckpt = TinyCheckpoint::<F>::new(cfg.clone(), self.weights);
                if let Some(d) = dtype {
                    ckpt = ckpt.with_dtype(d);
                }
                ckpt.model(dev)
            }
            Some(s) => F::scale_model(s, cfg, dtype, &self.weights, dev),
        }
    }

    /// Start the memory gauge on `dev`, once: the device's allocation before
    /// the first model, and at scale its peak every 10 ms until the drop.
    pub fn watch(&self, dev: &Device) {
        let mut g = self.gauge.lock();
        if g.is_none() {
            *g = Some((dev.clone(), MemGauge::start(dev, self.at_scale())));
        }
    }

    /// A fresh path under the system's temporary directory, removed when
    /// the fixture is dropped.
    pub fn scratch(&self, name: &str) -> PathBuf {
        let mut s = self.scratch.lock();
        let p = std::env::temp_dir().join(format!(
            "lloom-fixture-{}-{}-{}-{name}",
            std::process::id(),
            self.test,
            s.len()
        ));
        s.push(p.clone());
        p
    }

    fn checkpoint(&self) -> String {
        match &self.source {
            None => "tiny".to_string(),
            Some(Source::Real(p)) => p.display().to_string(),
            Some(Source::Synthetic) => "synthetic".to_string(),
        }
    }
}

impl<F: Scaled> Drop for Fixture<F> {
    fn drop(&mut self) {
        let wall = self.started.elapsed().as_millis();
        let mem = match self.gauge.lock().take() {
            Some((dev, g)) if !dev.is_cpu() => g.finish(&dev).line(),
            _ => "mem: cpu".to_string(),
        };
        let outcome = if std::thread::panicking() {
            "panicked"
        } else {
            "ok"
        };
        eprintln!(
            "FIXTURE test={} family={} backend={} checkpoint={} outcome={outcome} wall_ms={wall} {mem}",
            self.test,
            F::NAME,
            self.backend.name(),
            self.checkpoint()
        );
        for p in self.scratch.lock().drain(..) {
            let _ = std::fs::remove_file(&p);
        }
    }
}
