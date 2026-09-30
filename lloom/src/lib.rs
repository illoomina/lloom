// The neural-net building blocks, merged in from candle-nn (RESTRUCTURING §7.1):
// VarBuilder, ops, the layers and the Model/CausalLM/Speculative traits are now
// the `nn` module of this crate rather than a separate crate below it. Reached
// as `crate::nn::` internally and `lloom::nn::` by consumers.
pub mod nn;

// The within-crate axis registry (RESTRUCTURING §1.6): the authoritative,
// link-time-collected `AXES` slice every `#[derive(Config)]` contributes to.
pub mod axes;
pub mod tuning;

pub mod fused_moe;
pub mod generation;
pub mod load_warning;
// The hub module downloads checkpoints (hf-hub) and owns the tokenizer
// (tokenizers), neither of which compiles to wasm32; a browser loads bytes and
// tokenizes in JS (#656). Host-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
pub mod models;
pub mod object_detection;
// The optimizer (`design/docs/optimizer-design.md`): the IR a model emits, the
// egglog derivation over it, the fusion plan a construction consults. Host-only
// for the same reason `hub` is — egglog is not built for wasm32 here, and a
// browser build takes the manual arms.
#[cfg(not(target_arch = "wasm32"))]
pub mod optimizer;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
#[doc(hidden)]
pub mod testing;
pub mod utils;
