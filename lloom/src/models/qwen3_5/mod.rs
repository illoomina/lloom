//! The Qwen3.5 architecture — `qwen3_5` (Qwen3.6-27B, Qwen3.8-27B, the
//! small Qwen3.5 sizes) and `qwen3_5_moe` (Qwen3.6-35B-A3B) — on the generic
//! decoder topology (`nn::decoder`, `porting-models.md` §21.9 rung 1): a
//! per-layer pre-norm hybrid of the Gated DeltaNet and the gated GQA
//! attention Qwen4-exp built, over a dense SwiGLU or the routed MoE. No
//! forward of its own: the family is a type alias, a config, a schema and a
//! loader. Port doc: `design/docs/models/qwen3_5.md`. Reference dataflow:
//! llama.cpp `src/models/qwen35.cpp` / `qwen35moe.cpp`.

pub mod config;
pub mod loader;
pub mod schema;
pub mod testing;

pub use config::{Config, ConfigArgs, ARCH_CLEF, ARCH_DENSE, ARCH_MOE};
pub use loader::{arch_of, LoadOptions, DEFAULT_MEMORY_FRACTION};

use crate::models::qwen4exp::{GatedDeltaNet, MoeLayer, QsaAttention, Resident};
use crate::nn::blocks::SwiGlu;
use crate::nn::decoder::{Decoder, DecoderState, Hybrid, NormHead, PreNorm, TokenEmbed};
use crate::nn::mtp::NextnHead;
use crate::nn::RmsNorm;

/// A layer's token mixer: the Gated DeltaNet (`linear_attention`) or the
/// gated attention (`full_attention`), by the layer-type array.
pub type Qwen3_5Mixer<P> = Hybrid<GatedDeltaNet<P>, QsaAttention<P>>;
/// A dense layer: pre-norm, the mixer, post-attention norm, SwiGLU.
pub type Qwen3_5Layer<P> = PreNorm<RmsNorm, Qwen3_5Mixer<P>, SwiGlu>;
/// A MoE layer: the same with the routed + shared experts.
pub type Qwen3_5MoeLayer<P, R> = PreNorm<RmsNorm, Qwen3_5Mixer<P>, MoeLayer<P, R>>;
/// `qwen3_5`, dense.
pub type Qwen3_5<P> = Decoder<P, Config, TokenEmbed, Qwen3_5Layer<P>, NormHead<RmsNorm>>;
/// `qwen3_5_moe`, every expert resident.
pub type Qwen3_5Moe<P, R = Resident> =
    Decoder<P, Config, TokenEmbed, Qwen3_5MoeLayer<P, R>, NormHead<RmsNorm>>;
/// The states.
pub type Qwen3_5State<P> = DecoderState<Qwen3_5Layer<P>>;
pub type Qwen3_5MoeState<P, R = Resident> = DecoderState<Qwen3_5MoeLayer<P, R>>;
/// The MTP block (`blk.{n_layers}`): a full-attention layer of the family's
/// shape, with its own KV; the `nextn.*` projections around it
/// (`nn::mtp`).
pub type Qwen3_5MtpLayer<P> = PreNorm<RmsNorm, QsaAttention<P>, SwiGlu>;
pub type Qwen3_5MoeMtpLayer<P, R = Resident> = PreNorm<RmsNorm, QsaAttention<P>, MoeLayer<P, R>>;
pub type Qwen3_5MtpHead<P> = NextnHead<Qwen3_5MtpLayer<P>>;
pub type Qwen3_5MoeMtpHead<P, R = Resident> = NextnHead<Qwen3_5MoeMtpLayer<P, R>>;
