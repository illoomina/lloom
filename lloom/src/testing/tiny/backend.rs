//! The loader as the schema (`docs/porting-models.md` §21.8, INV-2): a
//! `SimpleBackend` that answers each `get(shape, name)` a constructor makes
//! with a keyed tensor ([`super::init`]) and records the request. Building a
//! model through it both fabricates the model and lists every tensor the
//! constructor needs, by name, shape and role — so a family whose
//! constructor reads a `VarBuilder` declares no tensor table of its own.

use super::init::{Role, ValueLaw};
use super::{Schema, TensorSpec};
use crate::nn::var_builder::SimpleBackend;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{bail, DType, Device, Result, Shape, Tensor};
use parking_lot::Mutex;
use std::sync::Arc;

/// What a [`TinyBackend`] was asked for, in request order.
#[derive(Clone, Default)]
pub struct Recording(Arc<Mutex<Vec<TensorSpec>>>);

impl Recording {
    /// The requests so far, as a schema whose tensors are stored as `dtype`.
    pub fn schema(&self, dtype: GgmlDType) -> Schema {
        Schema {
            tensors: self
                .0
                .lock()
                .iter()
                .map(|t| TensorSpec { dtype, ..t.clone() })
                .collect(),
        }
    }
}

/// Fabricates every tensor a constructor asks for, and records the asks.
///
/// Every value passes through `store` (the dtype a written checkpoint holds)
/// before the dtype the constructor asks for, so a model built in memory and
/// the same checkpoint loaded from its files see the same bits.
pub struct TinyBackend {
    law: ValueLaw,
    seed: u64,
    store: DType,
    recording: Recording,
}

impl TinyBackend {
    pub fn new(law: ValueLaw, seed: u64) -> (Self, Recording) {
        Self::with_store(law, seed, DType::F32)
    }

    pub fn with_store(law: ValueLaw, seed: u64, store: DType) -> (Self, Recording) {
        let recording = Recording::default();
        (
            Self {
                law,
                seed,
                store,
                recording: recording.clone(),
            },
            recording,
        )
    }

    fn fabricate(&self, spec: &TensorSpec, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.law
            .tensor(
                self.seed,
                &spec.name,
                spec.role,
                &Shape::from(spec.shape.clone()),
                &Device::Cpu,
            )?
            .to_dtype(self.store)?
            .to_dtype(dtype)?
            .to_device(dev)
    }
}

impl SimpleBackend for TinyBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let spec = {
            let mut rec = self.recording.0.lock();
            match rec.iter().find(|t| t.name == name) {
                Some(t) if t.shape != s.dims() => bail!(
                    "tiny backend: `{name}` asked for as {s:?} after {:?}",
                    t.shape
                ),
                Some(t) => t.clone(),
                None => {
                    let t = TensorSpec {
                        name: name.to_string(),
                        shape: s.dims().to_vec(),
                        role: Role::of_request(&s, &h),
                        dtype: GgmlDType::F32,
                    };
                    rec.push(t.clone());
                    t
                }
            }
        };
        self.fabricate(&spec, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let spec = self
            .recording
            .0
            .lock()
            .iter()
            .find(|t| t.name == name)
            .cloned();
        match spec {
            Some(spec) => self.fabricate(&spec, dtype, dev),
            None => bail!("tiny backend: `{name}` has no shape until a constructor asks for it"),
        }
    }

    /// Every name exists: the backend fabricates whatever is asked for. A
    /// constructor that probes for an optional tensor gets it.
    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}
