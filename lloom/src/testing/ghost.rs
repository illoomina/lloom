//! Ghost weights: a model built through its own constructor at a real
//! checkpoint's geometry, every tensor a broadcast of one zero element, so
//! its structure — what [`crate::optimizer::walk`] reads — costs no memory
//! and no checkpoint. A ghost is walked, never run: nothing here computes.

use crate::nn::var_builder::{SimpleBackend, VarBuilder};
use lloom_tensor::{bail, DType, Device, Result, Shape, Tensor};

/// One zero element broadcast to `shape`: the right extents and dtype, two
/// bytes of storage.
pub fn tensor(shape: impl Into<Shape>, dtype: DType, dev: &Device) -> Result<Tensor> {
    Tensor::zeros((), dtype, dev)?.broadcast_as(shape)
}

/// A `VarBuilder` backend answering every request with a [`tensor`] of the
/// asked shape: a model built through it has the checkpoint's layout and
/// none of its values.
pub struct GhostBackend;

impl SimpleBackend for GhostBackend {
    fn get(
        &self,
        s: Shape,
        _name: &str,
        _h: crate::nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        tensor(s, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        bail!("ghost backend: `{name}` asked for without a shape")
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

/// A ghost [`VarBuilder`] at `dtype` on `dev`.
pub fn var_builder(dtype: DType, dev: &Device) -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(GhostBackend), dtype, dev.clone())
}

/// `LFM2.5-2.6B`'s `config.json`
/// (`models--LiquidAI--LFM2.5-2.6B/snapshots/654f9463…/config.json`): 30
/// layers, 22 conv and 8 attention in the listed order, hidden 2048,
/// intermediate 10752, 32 heads over 8 KV heads, `conv_L_cache` 3, tied
/// embeddings over a 128000 vocabulary. Every other field the default arm.
pub fn lfm2_2_6b() -> crate::models::lfm2::Config {
    use crate::models::lfm2::{Config, LayerType::Conv as C, LayerType::FullAttention as A};
    Config {
        vocab_size: 128000,
        hidden_size: 2048,
        intermediate_size: 10752,
        num_hidden_layers: 30,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            C, C, A, C, C, A, C, C, C, A, C, C, C, A, C, C, C, A, C, C, C, A, C, C, A, C, C, A, C,
            C,
        ],
        tie_embedding: true,
        ..Default::default()
    }
}

/// An LFM2 model of `cfg` built by `Model::new` over ghost weights.
pub fn lfm2<P: crate::profile::Profile>(
    cfg: &crate::models::lfm2::Config,
    dtype: DType,
    dev: &Device,
) -> Result<crate::models::lfm2::Model<P>> {
    crate::models::lfm2::Model::<P>::new(cfg, var_builder(dtype, dev))
}

/// The walk of `cfg`'s LFM2 model over ghost weights, at the profile the
/// config's axes name (`Config::profile_key`, one `dispatch!` arm): what a
/// test that walks many configs calls, naming no profile of its own.
#[cfg(not(target_arch = "wasm32"))]
pub fn lfm2_graph(
    cfg: &crate::models::lfm2::Config,
    dtype: DType,
    dev: &Device,
) -> Result<crate::optimizer::Graph> {
    let key = cfg.exact_key();
    crate::profile::dispatch!(key, crate::profile::LFM2, P => {
        lfm2::<P>(cfg, dtype, dev).map(|m| crate::optimizer::walk(&m))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ghost_has_the_shape_and_no_storage_to_speak_of() {
        let t = tensor((128000, 2048), DType::F16, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[128000, 2048]);
        assert_eq!(t.dtype(), DType::F16);
        assert!(!t.is_contiguous());
    }
}
