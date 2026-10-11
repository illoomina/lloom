//! Compatibility path for the shared GGUF demand plans.
#[cfg(test)]
pub(super) use crate::nn::gguf_weights::assert_fixture_demands;
pub(super) use crate::nn::gguf_weights::{
    LinearWeights, NormWeights, Weight, WeightDType, WeightPlan,
};
