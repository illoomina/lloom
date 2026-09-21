pub mod affine;
pub mod arena_alloc;
pub mod binary;
pub mod cast;
pub mod conv_names;
pub mod convolution;
pub mod elementwise_names;
pub mod eos_gate;
pub mod eos_index;
pub mod fill;
pub mod flash_decoding;
pub mod indexing;
pub mod indexing_names;
mod macros;
pub mod mlx_gemm;
pub mod params;
pub mod quantized;
pub mod random;
pub mod reduce;
pub mod reduce_names;
pub mod scratch;
pub mod sdpa;
pub mod sort;
pub mod ternary;
pub mod unary;

pub use affine::*;
pub use arena_alloc::{
    call_arena_alloc_report, call_arena_bump, call_arena_bump_concurrent, call_arena_reset,
};
pub use binary::{
    call_binary_contiguous, call_binary_contiguous_with, call_binary_strided,
    call_binary_strided_with,
};
pub use cast::{
    call_cast_contiguous, call_cast_contiguous_with, call_cast_strided, call_cast_strided_with,
};
pub use conv_names::ConvKernel;
pub use convolution::*;
pub use eos_gate::{call_argmax_publish, call_eos_gate, call_words_publish};
pub use eos_index::call_eos_index;
pub use fill::*;
pub use flash_decoding::{
    call_flash_decoding, flash_chunk_count, ChunkTable, FlashDType, FlashKernel, FLASH_HEAD_DIMS,
};
pub use indexing::*;
pub use indexing_names::IndexingKernel;
pub use mlx_gemm::{
    call_mlx_gemm, call_mlx_gemv, call_mlx_gemv_epilogue, call_mlx_gemv_full, call_mlx_gemv_with,
    GemmDType, GemvEpilogue, GemvRmsNorm, SKINNY_TILE_MAX_M,
};
pub use scratch::{
    call_scratch_combine, call_scratch_partials, call_scratch_report, ScratchKernel, ScratchParams,
};
// `ParamStyle` is re-exported here rather than from `reduce`, where it was
// declared when `reduce.metal` was the only family carrying both binding
// styles. It moved to `params` when `conv.metal` followed (issue #42) so every
// family shares one declaration; `pub use reduce::*` used to carry it, and
// this line is what keeps that spelling working for callers.
pub use params::{
    AffineParams, AffineStridedParams, BinaryParams, BinaryStridedParams, CastParams,
    CastStridedParams, Col2im1dParams, Conv1dDepthwiseKParams, Conv1dDepthwiseParams,
    ConvTranspose1dParams, ConvTranspose2dParams, Copy2dParams, GatherParams, GemvParams,
    Im2col1dParams, Im2colParams, IndexAddParams, IndexParams, LayoutDescriptor, LayoutFamily,
    NormParams, ParamStyle, Pool2dParams, ReduceParams, RopeIParams, RopeParams, RopeThdParams,
    ScaleParams, ScaleStridedParams, ScatterParams, SdpaQuantParams, SdpaQuantWriteParams,
    SoftmaxParams, UnaryParams, UnaryStridedParams, UpsampleBilinear2dParams,
    UpsampleNearest2dParams,
};
pub use quantized::{
    call_quantized_get_rows, call_quantized_matmul_mm_t, call_quantized_matmul_mv_id_t,
    call_quantized_matmul_mv_t, GemvFloat, GemvTriple, GgmlDType,
};
pub use random::*;
pub use reduce::*;
pub use reduce_names::ReduceKernel;
pub use sdpa::{
    call_dequantize_kv, call_quantize_kv, call_sdpa_full, call_sdpa_full_with, call_sdpa_vector,
    call_sdpa_vector_2pass, call_sdpa_vector_quant, quantize_group_reference, sdpa_full_tile,
    KvQuantBits, SdpaDType, SdpaPageTable,
};
pub use sort::{call_arg_sort, call_mlx_arg_sort};
pub use ternary::call_where_cond;
pub use unary::*;
