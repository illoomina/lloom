// The neural-net building blocks, merged in from candle-nn (RESTRUCTURING §7.1):
// VarBuilder, ops, the layers and the Model/CausalLM/Speculative traits are now
// the `nn` module of this crate rather than a separate crate below it. Reached
// as `crate::nn::` internally and `lloom::nn::` by consumers.
pub mod nn;

pub mod fused_moe;
pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod utils;
