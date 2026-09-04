pub mod err;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

// `#[macro_export]` places these at the crate root (`lloom_tensor::`) — a
// property of the attribute, unchanged by the merge (RESTRUCTURING §7.1).
// Re-export them into this module so the moved kernel sources' existing
// `use crate::kernels::{debug_group, set_params, metal_label, …}` imports
// resolve exactly as they did when this was the `candle_metal_kernels` crate
// root.
pub use crate::{debug_group, metal_label, set_params};

pub use err::MetalKernelError;
pub use kernel::Kernels;
// The two COMPILE-TIER arms, exported so a harness can state them on its
// `RESULT` line (`DESIGN.md` §7.1a, §14.5a). Both are read back from the
// `OnceLock` the compiler itself reads rather than echoed from a flag: an axis
// rendered from the flag rather than from the mechanism is §7.1a-i's
// *present-and-wrong* case, which is the unsafe direction because a false arm
// compares equal to real runs of that arm.
pub use kernel::{merged_translation_unit, metal_compile_flags};
// `ParamStyle` used to arrive via `reduce::*`, where it was declared when
// `reduce.metal` was the only family carrying both binding styles. It moved to
// `kernels::params` when `conv.metal` followed (issue #42), so it is named
// explicitly here; the spelling callers use is unchanged.
pub use kernels::{
    affine::*, arena_alloc::*, call_binary_contiguous, call_binary_contiguous_with,
    call_binary_strided, call_binary_strided_with, call_mlx_gemm, call_mlx_gemv_epilogue,
    call_mlx_gemv_full, call_mlx_gemv_with, cast::*, convolution::*, fill::*, flash_decoding::*,
    indexing::*, quantized::*, random::*, reduce::*, scratch::*, sdpa::*, sort::*, ternary::*,
    unary, unary::*, ConvKernel, GemmDType, GemvEpilogue, GemvRmsNorm, GgmlDType, IndexingKernel,
    ParamStyle, ReduceKernel, SdpaQuantParams, SdpaQuantWriteParams,
};
// The binding-style default, so a harness can reach the packed entry points
// without every `call_*` growing an argument (issue #115). Named rather than
// arriving through a glob for the same reason as the lines below: this is the
// switch that decides whether a decode dispatch is ICB-encodable at all, and
// `set_default_param_style` says why the choice lives here and not at the call
// sites.
pub use kernels::params::{
    constants_pool, default_param_style, set_constants_pool_enabled, set_default_param_style,
    ConstantsPool, FlashCombineParams, FlashPartialParams, ShortconvRingParams,
    ShortconvStepParams,
};
// The SiLU variant axis (issue #599, `DESIGN.md` §12.2l). Named rather than
// arriving through a glob for `set_default_param_style`'s reason: this is the
// switch that decides which arithmetic 30 of the step's dispatches perform, and
// `SiluImpl`'s own docs say why the choice lives here and not at the call sites.
pub use kernels::unary::{set_silu_impl, silu_impl, SiluDType, SiluImpl};
// The arena's GPU-side allocator vocabulary (`DESIGN.md` §9.2d, issue #70).
// Named rather than arriving through a glob, because `ARENA_DECLINED` is a
// cross-language constant -- `arena_alloc.metal` writes the same sentinel, and
// `arena_alloc_reports_alignment` is what checks the two agree.
pub use metal::{ArenaCursor, ArenaOffsets, ARENA_DECLINED};
// The `supportIndirectCommandBuffers` switch (issue #115). Named here because
// getting it wrong is §3.7d's segfault at encode time rather than a wrong
// answer, so it should be as reachable as the executor that requires it.
pub use metal::device::{pipelines_support_icb, set_pipelines_support_icb};
// The ICB executor itself (`DESIGN.md` §17 Phase 2 item 10).
pub use metal::icb::{BarrierScope, Coverage, IcbExecutor, ReplayBarriers};
// The scratch class's vocabulary (`DESIGN.md` §9.1, issue #71). Named rather
// than arriving through a glob for the same reason as the line above: `Sizing`
// is a compile-tier policy whose spellings must agree with the `[[host_name]]`
// instantiations in `scratch.metal`, and `ScratchKernel` is what checks they do.
pub use metal::scratch::{
    plan_scratch, CombineOrder, PartialsGeometry, ScratchLayout, ScratchPlan, ScratchRegion,
    Sizing, BUCKET_LADDER, PARTIAL_ELEM_BYTES, PARTIAL_STATS,
};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function,
    Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use source::Source;
use utils::{get_block_dims, get_tile_size, linear_split, EncoderParam, EncoderProvider};
pub use utils::{BufferOffset, Output};

pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[cfg(test)]
mod tests;
