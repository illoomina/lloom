//! candle-nn
//!
//! ## Other Crates
//!
//! Candle consists of a number of crates. This crate holds structs and functions
//! that allow you to build and train neural nets. You may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
//! - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
//! - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
//! - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
//! - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
//! - [candle-transformers](https://docs.rs/candle-transformers/). Candle implementation of many published transformer models.
//!

pub mod activation;
pub mod attention;
pub mod batch_norm;
pub mod batched;
mod batched_spec;
pub mod conv;
pub mod cpu_flash_attention;
pub mod decode;
pub mod embedding;
pub mod encoding;
pub mod expert_stream;
pub mod func;
pub mod fusion;
pub mod group_norm;
pub mod init;
pub mod kv_cache;
pub mod layer_norm;
pub mod linear;
pub mod loss;
pub mod memory;
pub mod model;
pub mod moe;
pub mod observer;
pub mod ops;
pub mod optim;
pub mod recurrent;
pub mod rnn;
pub mod rotary_emb;
pub mod sampling;
pub mod sequential;
pub mod speculator;
pub mod var_builder;
pub mod var_map;
/// Re-export of [`attention::varlen`] for backward compatibility.
pub mod varlen_attention {
    pub use crate::nn::attention::varlen::*;
}

pub use activation::{prelu, Activation, PReLU};
pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use batched::{BatchDecoder, RowDriver, RowId, StepReport};
pub use batched_spec::SpecBatchDecoder;
pub use conv::{
    conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv_transpose1d, conv_transpose1d_no_bias,
    conv_transpose2d, conv_transpose2d_no_bias, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
pub use decode::{
    ladder_width, ragged_layout, DecodeOptions, DecodeOut, DecodeStats, IdentityProposer,
    PassWidths, PrefillOut, RoundStats, StepPhases, FIRST_STEPS_TRACED,
};
pub use embedding::{embedding, Embedding};
pub use expert_stream::{
    ExpertResidency, ExpertSource, LoaderArm, Resident, RoutedGroup, StagingStats, Streamed,
};
pub use func::{func, func_t, Func, FuncT};
pub use group_norm::{group_norm, GroupNorm};
pub use init::Init;
pub use layer_norm::{
    layer_norm, layer_norm_no_bias, rms_norm, LayerNorm, LayerNormConfig, RmsNorm,
};
pub use linear::{linear, linear_b, linear_no_bias, Linear};
pub use model::{
    BatchedCausalLM, CacheReset, CacheRewind, CacheShare, CausalLM, Model, RaggedRow, Row,
    Speculative, VisionEncoder,
};
pub use observer::{
    ForwardObserver, HcStatePoint, HookKind, HookPlan, HookPoint, LayerInputPoint,
    LayerOutputPoint, NextnHiddenPoint,
};
pub use ops::Dropout;
pub use optim::{AdamW, Optimizer, ParamsAdamW, SGD};
pub use recurrent::{RecurrentState, StatePool};
pub use rnn::{gru, lstm, GRUConfig, LSTMConfig, GRU, LSTM, RNN};
pub use sequential::{seq, Sequential};
pub use speculator::{
    Anchor, Draft, DraftTokens, KvSource, LayerTaps, NextnHidden, NoProbe, ObserveRow, PhaseCensus,
    Prepared, ProbeHooks, Probes, ReuseAction, RoundProbe, RoundTiming, SeqId, Sequence, Session,
    Speculator, TapSide, TapSpec, TargetProbe, VerifyRowRef,
};
pub use var_builder::VarBuilder;
pub use var_map::VarMap;

pub use lloom_tensor::{Module, ModuleT};
