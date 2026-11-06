use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, CastParams, CastStridedParams, ParamStyle,
};
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use crate::kernels::get_tile_size;

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `cast_params_layout_matches_metal` is what proves those
/// mirrors agree with `cast.metal`.
const CAST_PARAMS_ALIGN: usize = core::mem::align_of::<CastParams>();
const CAST_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<CastStridedParams>();

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_cast_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_cast_contiguous_with(
            device,
            ep,
            kernels,
            kernel_name,
            dtype_size,
            length,
            input,
            output,
            ParamStyle::default(),
        )
    }
}

/// As [`call_cast_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_cast_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let spec = CastKernelSpec::named(kernel_name)?;
    if spec.strided || spec.input_bytes != dtype_size {
        return Err(cast_refusal("contiguous cast kernel/type contract mismatch"));
    }
    let geometry = CastGeometry::contiguous(length,spec.input_bytes,spec.output_bytes)?;
    geometry.validate_views(&input,output)?;
    if length == 0 { return Ok(()); }
    let pipeline = kernels.load_pipeline(device, Source::Cast, spec.name(geometry.wide,style))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "cast {kernel_name} elems={length}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (length, &input, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, CAST_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = geometry.grid(&pipeline)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_cast_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_cast_strided_with(
            device,
            ep,
            kernels,
            kernel_name,
            shape,
            input,
            input_strides,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_cast_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let spec = CastKernelSpec::named(kernel_name)?;
    if !spec.strided { return Err(cast_refusal("strided cast kernel contract mismatch")); }
    let geometry = CastGeometry::strided(shape,input_strides,spec.input_bytes,spec.output_bytes)?;
    geometry.validate_views(&input,output)?;
    let length = geometry.length;
    if length == 0 { return Ok(()); }
    let pipeline = kernels.load_pipeline(device, Source::Cast, spec.name(geometry.wide,style))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    debug_group!(encoder, "cast_strided {kernel_name} elems={length}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length,
                shape.len(),
                shape,
                input_strides,
                &input,
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, CAST_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = geometry.grid(&pipeline)?;

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}


/// Product for an ALREADY existing source layout, not a fresh Tensor/axes
/// constructor. There are no accesses for an empty layout, regardless of the
/// unused nonzero dimensions. Fresh owned axes/strides require their producer's
/// separate checked construction before a Tensor can be published.
fn cast_existing_layout_elements(shape: &[usize]) -> Result<usize, MetalKernelError> {
    if shape.contains(&0) { return Ok(0); }
    shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim)
        .ok_or(cast_refusal("cast logical element count overflow")))
}

fn cast_refusal(reason:&'static str)->MetalKernelError { MetalKernelError::QueueUnavailable(reason) }

/// Checked metadata only: no allocations, pipeline lookup or encoder effect.
/// Extents are relative to each Buffer view, never heap/native base offsets.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct CastGeometry {
    length:usize, source_bytes:usize, output_bytes:usize,
    input_width:usize,output_width:usize,strided:bool,wide:bool,
}
impl CastGeometry {
    fn contiguous(length:usize,input_width:usize,output_width:usize)->Result<Self,MetalKernelError>{
        if input_width==0 || output_width==0 {return Err(cast_refusal("cast zero type width"));}
        let source_bytes=length.checked_mul(input_width).ok_or(cast_refusal("cast source byte extent overflow"))?;
        let output_bytes=length.checked_mul(output_width).ok_or(cast_refusal("cast destination byte extent overflow"))?;
        Ok(Self{length,source_bytes,output_bytes,input_width,output_width,strided:false,
            // The final unused uint increment stays representable on this arm.
            // Larger valid tensors use the size_t route; this is NOT a cap.
            wide:length>(u32::MAX as usize)/2})
    }
    fn strided(shape:&[usize],strides:&[usize],input_width:usize,output_width:usize)->Result<Self,MetalKernelError>{
        if shape.len()!=strides.len(){return Err(cast_refusal("cast shape/stride rank mismatch"));}
        let length=cast_existing_layout_elements(shape)?;
        let mut geometry=Self::contiguous(length,input_width,output_width)?;geometry.strided=true;
        if length==0{return Ok(geometry);}
        let mut last=0usize;
        for (&dim,&stride) in shape.iter().zip(strides) {
            let term=(dim-1).checked_mul(stride).ok_or(cast_refusal("cast strided source extent overflow"))?;
            last=last.checked_add(term).ok_or(cast_refusal("cast strided source extent overflow"))?;
        }
        geometry.source_bytes=last.checked_add(1).and_then(|n|n.checked_mul(input_width))
            .ok_or(cast_refusal("cast strided source byte extent overflow"))?;
        geometry.wide|=last>u32::MAX as usize ||shape.len()>u32::MAX as usize;
        Ok(geometry)
    }
    fn validate_views(&self,input:&BufferOffset<'_>,output:&Buffer)->Result<(),MetalKernelError>{
        let end=input.offset_in_bytes.checked_add(self.source_bytes).ok_or(cast_refusal("cast input offset overflow"))?;
        if end>input.buffer.length() || self.output_bytes>output.length(){
            return Err(cast_refusal("cast access exceeds originating buffer view"));
        }
        let base=input.buffer.base_offset().checked_add(input.offset_in_bytes).ok_or(cast_refusal("cast native input base overflow"))?;
        if base%self.input_width!=0 || output.base_offset()%self.output_width!=0 {
            return Err(cast_refusal("cast typed buffer alignment refused"));
        }
        Ok(())
    }
    fn work_items(&self)->usize {
        if self.strided{self.length}else{self.length.div_ceil(get_tile_size(self.input_width))}
    }
    fn grid(&self,pipeline:&crate::kernels::metal::ComputePipeline)
        ->Result<(crate::kernels::MTLSize,crate::kernels::MTLSize),MetalKernelError>{
        self.grid_with_width(pipeline.max_total_threads_per_threadgroup())
    }
    fn grid_with_width(&self,max_width:usize)
        ->Result<(crate::kernels::MTLSize,crate::kernels::MTLSize),MetalKernelError>{
        use crate::kernels::MTLSize;
        let items=self.work_items();
        if items==0 || max_width==0{return Err(cast_refusal("empty cast has no dispatch grid"));}
        let width=max_width.min(items);let count=items.div_ceil(width);
        if !self.wide {
            return Ok((MTLSize{width:count,height:1,depth:1},MTLSize{width,height:1,depth:1}));
        }
        // Each native group coordinate is uint. Flattening is size_t and
        // checked against actual work before multiplication in cast.metal.
        let axis=u32::MAX as usize;
        let x=count.min(axis);let rest=count.div_ceil(x);
        let y=rest.min(axis);let z=rest.div_ceil(y);
        Ok((MTLSize{width:x,height:y,depth:z},MTLSize{width,height:1,depth:1}))
    }
}

#[derive(Clone,Copy)]
struct CastKernelSpec {
    input_bytes:usize,output_bytes:usize,strided:bool,
    narrow:&'static str,narrow_packed:&'static str,wide:&'static str,wide_packed:&'static str,
}
impl CastKernelSpec {
    fn name(self,wide:bool,style:ParamStyle)->&'static str {
        match (wide,style){(false,ParamStyle::Split)=>self.narrow,(false,ParamStyle::Packed)=>self.narrow_packed,
            (true,ParamStyle::Split)=>self.wide,(true,ParamStyle::Packed)=>self.wide_packed}
    }
    fn named(name:&str)->Result<Self,MetalKernelError>{
        macro_rules! rows {
            ($(($a:literal,$aw:expr,$b:literal,$bw:expr)),* $(,)?)=>{
                match name {
                    $(concat!("cast_",$a,"_",$b)=>Ok(Self{input_bytes:$aw,output_bytes:$bw,strided:false,
                        narrow:concat!("cast_",$a,"_",$b),narrow_packed:concat!("cast_",$a,"_",$b,"_packed"),
                        wide:concat!("cast_",$a,"_",$b,"_wide"),wide_packed:concat!("cast_",$a,"_",$b,"_wide_packed")}),
                      concat!("cast_",$a,"_",$b,"_strided")=>Ok(Self{input_bytes:$aw,output_bytes:$bw,strided:true,
                        narrow:concat!("cast_",$a,"_",$b,"_strided"),narrow_packed:concat!("cast_",$a,"_",$b,"_strided_packed"),
                        wide:concat!("cast_",$a,"_",$b,"_strided_wide"),wide_packed:concat!("cast_",$a,"_",$b,"_strided_wide_packed")}),)*
                    _=>Err(cast_refusal("cast entry has no checked type/layout contract")),
                }
            }
        }
        rows!(
            ("u8",1,"u8",1),
            ("u8",1,"u32",4),
            ("u8",1,"i64",8),
            ("u8",1,"f16",2),
            ("u8",1,"bf16",2),
            ("u8",1,"f32",4),
            ("u32",4,"u8",1),
            ("u32",4,"u32",4),
            ("u32",4,"i64",8),
            ("u32",4,"f16",2),
            ("u32",4,"bf16",2),
            ("u32",4,"f32",4),
            ("i64",8,"u8",1),
            ("i64",8,"u32",4),
            ("i64",8,"i64",8),
            ("i64",8,"f16",2),
            ("i64",8,"bf16",2),
            ("i64",8,"f32",4),
            ("f16",2,"u8",1),
            ("f16",2,"u32",4),
            ("f16",2,"i64",8),
            ("f16",2,"f16",2),
            ("f16",2,"bf16",2),
            ("f16",2,"f32",4),
            ("bf16",2,"u8",1),
            ("bf16",2,"u32",4),
            ("bf16",2,"i64",8),
            ("bf16",2,"f16",2),
            ("bf16",2,"bf16",2),
            ("bf16",2,"f32",4),
            ("f32",4,"u8",1),
            ("f32",4,"u32",4),
            ("f32",4,"i64",8),
            ("f32",4,"f16",2),
            ("f32",4,"bf16",2),
            ("f32",4,"f32",4),
        )
    }
}

/// The actual ordinary Metal to_dtype supported set, factored ONCE. Same-type
/// Tensor conversion remains its incumbent clone route, not a new kernel.
pub(crate) fn cast_kernel_name(input:crate::DType,output:crate::DType,strided:bool)->Option<&'static str>{
    use crate::DType;
    match (input,output){
        (DType::U32,DType::BF16)=>Some(if strided{"cast_u32_bf16_strided"}else{"cast_u32_bf16"}),
        (DType::U32,DType::F16)=>Some(if strided{"cast_u32_f16_strided"}else{"cast_u32_f16"}),
        (DType::U32,DType::F32)=>Some(if strided{"cast_u32_f32_strided"}else{"cast_u32_f32"}),
        (DType::U32,DType::I64)=>Some(if strided{"cast_u32_i64_strided"}else{"cast_u32_i64"}),
        (DType::U32,DType::U8)=>Some(if strided{"cast_u32_u8_strided"}else{"cast_u32_u8"}),
        (DType::U8,DType::BF16)=>Some(if strided{"cast_u8_bf16_strided"}else{"cast_u8_bf16"}),
        (DType::U8,DType::F16)=>Some(if strided{"cast_u8_f16_strided"}else{"cast_u8_f16"}),
        (DType::U8,DType::F32)=>Some(if strided{"cast_u8_f32_strided"}else{"cast_u8_f32"}),
        (DType::U8,DType::I64)=>Some(if strided{"cast_u8_i64_strided"}else{"cast_u8_i64"}),
        (DType::U8,DType::U32)=>Some(if strided{"cast_u8_u32_strided"}else{"cast_u8_u32"}),
        (DType::F32,DType::BF16)=>Some(if strided{"cast_f32_bf16_strided"}else{"cast_f32_bf16"}),
        (DType::F32,DType::F16)=>Some(if strided{"cast_f32_f16_strided"}else{"cast_f32_f16"}),
        (DType::F32,DType::I64)=>Some(if strided{"cast_f32_i64_strided"}else{"cast_f32_i64"}),
        (DType::F32,DType::U32)=>Some(if strided{"cast_f32_u32_strided"}else{"cast_f32_u32"}),
        (DType::F32,DType::U8)=>Some(if strided{"cast_f32_u8_strided"}else{"cast_f32_u8"}),
        (DType::I64,DType::BF16)=>Some(if strided{"cast_i64_bf16_strided"}else{"cast_i64_bf16"}),
        (DType::I64,DType::F16)=>Some(if strided{"cast_i64_f16_strided"}else{"cast_i64_f16"}),
        (DType::I64,DType::F32)=>Some(if strided{"cast_i64_f32_strided"}else{"cast_i64_f32"}),
        (DType::I64,DType::U32)=>Some(if strided{"cast_i64_u32_strided"}else{"cast_i64_u32"}),
        (DType::I64,DType::U8)=>Some(if strided{"cast_i64_u8_strided"}else{"cast_i64_u8"}),
        (DType::F16,DType::BF16)=>Some(if strided{"cast_f16_bf16_strided"}else{"cast_f16_bf16"}),
        (DType::F16,DType::F32)=>Some(if strided{"cast_f16_f32_strided"}else{"cast_f16_f32"}),
        (DType::F16,DType::I64)=>Some(if strided{"cast_f16_i64_strided"}else{"cast_f16_i64"}),
        (DType::F16,DType::U32)=>Some(if strided{"cast_f16_u32_strided"}else{"cast_f16_u32"}),
        (DType::F16,DType::U8)=>Some(if strided{"cast_f16_u8_strided"}else{"cast_f16_u8"}),
        (DType::BF16,DType::F16)=>Some(if strided{"cast_bf16_f16_strided"}else{"cast_bf16_f16"}),
        (DType::BF16,DType::F32)=>Some(if strided{"cast_bf16_f32_strided"}else{"cast_bf16_f32"}),
        (DType::BF16,DType::I64)=>Some(if strided{"cast_bf16_i64_strided"}else{"cast_bf16_i64"}),
        (DType::BF16,DType::U32)=>Some(if strided{"cast_bf16_u32_strided"}else{"cast_bf16_u32"}),
        (DType::BF16,DType::U8)=>Some(if strided{"cast_bf16_u8_strided"}else{"cast_bf16_u8"}),
        _=>None,
    }
}

#[cfg(test)]
mod cast_geometry_controls {
    use super::*;
    #[test]
    fn large_contiguous_and_strided_metadata_select_wide_without_rank_or_uint_cap(){
        let len=(u32::MAX as usize).checked_add(1).unwrap();
        let a=CastGeometry::contiguous(len,1,2).unwrap();assert!(a.wide);
        assert_eq!((a.source_bytes,a.output_bytes),(len,len*2));
        let b=CastGeometry::strided(&[2,2],&[len,1],1,2).unwrap();assert!(b.wide);
        assert_eq!(b.source_bytes,len+2);
        let small=CastGeometry::strided(&[2,3],&[5,1],4,2).unwrap();assert!(!small.wide);
        assert_eq!((small.length,small.source_bytes,small.output_bytes),(6,32,12));
        let long=vec![1;1024];assert_eq!(CastGeometry::strided(&long,&long,1,2).unwrap().length,1);
    }
    #[test]
    fn checked_shape_stride_bytes_empty_and_grid_keep_their_distinct_contracts(){
        assert!(CastGeometry::strided(&[2,3],&[1],1,2).is_err());
        assert!(CastGeometry::strided(&[2,2],&[usize::MAX,1],1,2).is_err());
        assert!(CastGeometry::contiguous(usize::MAX,1,2).is_err());
        let empty=CastGeometry::strided(&[0,3],&[usize::MAX,1],4,2).unwrap();
        assert_eq!((empty.length,empty.source_bytes,empty.output_bytes),(0,0,0));
        assert!(empty.grid_with_width(1024).is_err(),"EMPTY_CAST_MUST_RETURN_BEFORE_DISPATCH_GRID");
        let scalar=CastGeometry::strided(&[],&[],4,2).unwrap();assert_eq!(scalar.source_bytes,4);
    }
    #[test]
    fn existing_layout_product_checks_overflow_without_constructing_fresh_axes() {
        assert_eq!(cast_existing_layout_elements(&[]).unwrap(), 1);
        assert_eq!(cast_existing_layout_elements(&[2,3,7]).unwrap(), 42);
        assert!(cast_existing_layout_elements(&[usize::MAX,2]).is_err());
        assert_eq!(cast_existing_layout_elements(&[usize::MAX,2,0]).unwrap(), 0);
        assert_eq!(cast_existing_layout_elements(&[0,usize::MAX,2]).unwrap(), 0);
    }
    #[test]
    fn every_actual_pair_has_both_layouts_and_both_binding_styles(){
        let types=[crate::DType::U8,crate::DType::U32,crate::DType::I64,crate::DType::F16,crate::DType::BF16,crate::DType::F32];
        let mut pairs=0;
        for a in types{for b in types {for strided in [false,true] {
            let name=cast_kernel_name(a,b,strided);
            if a==b {assert!(name.is_none());continue;}
            let spec=CastKernelSpec::named(name.unwrap()).unwrap();
            assert_eq!((spec.input_bytes,spec.output_bytes,spec.strided),(a.size_in_bytes(),b.size_in_bytes(),strided));
            assert_ne!(spec.name(false,ParamStyle::Split),spec.name(true,ParamStyle::Split));
            assert!(spec.name(true,ParamStyle::Packed).ends_with("_wide_packed"));pairs+=1;
        }}}
        assert_eq!(pairs,60);
    }
}

/// Native kernel controls, not an Original/Holding/Ready or whole-load proof.
/// Run one explicitly selected API per isolated nextest process. The raw
/// interoperability fixture retains every buffer and parameter until actual D;
/// it exposes no cached ICB/future emitter and does not claim supplied-recording
/// funding. Production original-leaf custody remains a separate required gate.
#[cfg(test)]
mod cast_native_controls {
    use super::*;
    use crate::kernels::metal::{Commands, ResidencySet};
    use objc2_metal::{MTLStorageMode,MTLResource};
    use std::sync::Arc;

    // Exact prior production fragment, SHA256 dcbb18fb7e5a329414c5f5663719fea9c1d6dd9f240ecea7a54887ae66fd5052.
    // Frozen test oracle ONLY; runtime code has one shared conversion body.
    const LEGACY_FRAGMENT: &str = r###"// NOTE: this file does NOT open with `#include <metal_stdlib>`.
// `source.rs` supplies the include prologue and splices `utils.metal`'s
// shared helpers in after it (issue #458), so this file is a FRAGMENT and
// does not compile standalone. `newLibraryWithSource` sees prologue +
// utils + this text as one translation unit.

// `get_strided_index`, `div_ceil` and `work_per_thread` are shared: they live
// in `utils.metal`, concatenated ahead of this file by `source.rs` (#458).

// Packed parameter blocks.
//
// An ICB command cannot carry an inline constant -- `MTLIndirectComputeCommand`
// has no `setBytes` in any form (`DESIGN.md` §3.7c) -- so every scalar a kernel
// takes as `constant size_t &n` has to arrive in a buffer instead. These are the
// structs the `_packed` entry points read, mirrored by `#[repr(C)]` types in
// `kernels/params.rs` and checked against them by `elementwise_params_layout`.
//
// `size_t` is 8 bytes in MSL, so these are 8-aligned -- unlike `reduce.metal`'s
// `uint` structs, which are 4-aligned. That difference is the reason the layout
// check ships the numbers across the boundary rather than trusting either side.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length.
struct CastParams {
    size_t dim;
};

struct CastStridedParams {
    size_t dim;
    size_t num_dims;
};

// Kernels
//
// One body per kernel, two entry points around it. The classical wrapper binds
// its scalars with `setBytes` exactly as before; the `_packed` one reads them
// from a single `device const Params*`. Neither the arithmetic nor the loop
// structure is duplicated, so the two styles cannot compute different things --
// which is what makes the bit-identical parity test meaningful rather than
// merely reassuring.
//
// Unlike `reduce.metal`, nothing here declares threadgroup memory, so the
// whole-body-plus-two-thin-wrappers factoring that #38 could not use does work
// in this file. (MSL permits a `threadgroup` variable only inside a
// `[[kernel]]`-qualified function; there is no such variable in any of these
// four families.)
template <typename T, typename U, typename IR, int W>
METAL_FUNC void cast_body(
    size_t dim,
    device const T* input,
    device U* output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(static_cast<IR>(input[i]));
    }
}

template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    cast_body<T, U, IR, W>(dim, input, output, tid);
}

template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel_packed(
    device const CastParams *pp,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    cast_body<T, U, IR, W>(pp->dim, input, output, tid);
}

// `InPtrT` is templated alongside `PtrT` so the classical wrapper keeps its
// `constant const T *input` exactly as it was. That address space is not
// incidental -- changing it would be a behaviour change to the classical path
// smuggled in beside a binding change, and the acceptance bar for this work is
// that the classical path is untouched.
template <typename T, typename U, typename IR, typename PtrT, typename InPtrT>
METAL_FUNC void cast_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    InPtrT input,
    device U *output,
    uint tid
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        static_cast<IR>(input[get_strided_index(tid, num_dims, dims, strides)])
    );
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    cast_strided_body<T, U, IR, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, input, output, tid);
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided_packed(
    device const CastStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    cast_strided_body<T, U, IR, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, input, output, tid);
}

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `cast_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
//
// `size_t` is 8 bytes here, hence 8-aligned. That is the substantive difference
// from `reduce.metal`, whose packed structs are all 4-aligned `uint`.
static_assert(sizeof(CastParams) == 8, "CastParams layout");
static_assert(alignof(CastParams) == 8, "CastParams alignment");

static_assert(sizeof(CastStridedParams) == 16, "CastStridedParams layout");
static_assert(alignof(CastStridedParams) == 8, "CastStridedParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void cast_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    CastParams        probe_CastParams;
    CastStridedParams probe_CastStridedParams;
    out[0] = sizeof(CastParams);
    out[1] = offsetof_rt(CastParams, dim);

    out[2] = sizeof(CastStridedParams);
    out[3] = offsetof_rt(CastStridedParams, dim);
    out[4] = offsetof_rt(CastStridedParams, num_dims);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both binding styles from one instantiation row, so a variant cannot exist in
// one style and not the other. `_packed` is a name segment appended after the
// dtype and any `_strided`, and `packed_names_resolve` checks every result
// against the compiled library rather than against this macro -- which is
// `DESIGN.md` §8.1b's argument, and #26 shipped 48 names absent from a metallib
// that compiled cleanly.
#define init_cast(tname, t, uname, u)                                                          \
    init_kernel("cast_" #tname "_" #uname, cast_kernel, t, u)                                  \
    init_kernel("cast_" #tname "_" #uname "_packed", cast_kernel_packed, t, u)                 \
    init_kernel("cast_" #tname "_" #uname "_strided", cast_kernel_strided, t, u)               \
    init_kernel("cast_" #tname "_" #uname "_strided_packed", cast_kernel_strided_packed, t, u)

#if defined(__HAVE_BFLOAT__)
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, bf16, bfloat)   \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#else
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#endif


init_cast_all(f32, float);
init_cast_all(f16, half);
#if defined(__HAVE_BFLOAT__)
init_cast_all(bf16, bfloat);
#endif
init_cast_all(i64, int64_t);
init_cast_all(u32, uint32_t);
init_cast_all(u8, uint8_t);
"###;

    fn setup() -> (Device,Commands) {
        let api=std::env::var("LLOOM_METAL_API").expect("explicit metal3|metal4 API required");
        assert!(matches!(api.as_str(),"metal3"|"metal4"));
        let mut tuning=crate::tuning::Tuning::default();
        tuning.from_env().unwrap();
        // This baseline raw-parameter fixture retains its native references;
        // original supplied/unretained ownership is NOT inferred from it.
        tuning.metal.cb_unretained=false;
        crate::tuning::install(tuning).unwrap();
        let device=Device::system_default().expect("native Metal device required");
        let commands=Commands::new(device.new_command_queue().unwrap(),Arc::new(ResidencySet::new(&device))).unwrap();
        assert_eq!(commands.actual_command_api(),api,"native API fallback is not a control");
        (device,commands)
    }
    fn library(device:&Device,fragment:&str)->crate::kernels::metal::Library {
        // The same shared utilities and actual MathMode/language/compiler
        // options as the production Source::Cast, not a CPU/precise oracle.
        let text=format!("#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
            include_str!("../metal_src/utils.metal"),fragment);
        device.new_library_with_source(&text,Some(&crate::kernels::kernel::get_compile_options())).unwrap()
    }
    fn pipeline(device:&Device,lib:&crate::kernels::metal::Library,name:&str)->crate::kernels::metal::ComputePipeline {
        device.new_compute_pipeline_state_with_function(&lib.get_function(name,None).unwrap()).unwrap()
    }
    fn shared(device:&Device,bytes:&[u8])->Buffer {
        assert!(!bytes.is_empty());
        device.new_buffer_with_data(bytes.as_ptr().cast(),bytes.len(),objc2_metal::MTLResourceOptions::StorageModeShared).unwrap()
    }
    fn private(device:&Device,bytes:usize)->Buffer {
        let buffer=device.new_buffer(bytes.max(4).next_multiple_of(4),objc2_metal::MTLResourceOptions::StorageModePrivate).unwrap();
        assert_eq!(unsafe{buffer.raw_unchecked().storageMode()},MTLStorageMode::Private);
        buffer
    }
    fn readback(device:&Device,commands:&Commands,source:&Buffer,bytes:usize)->Vec<u8> {
        let len=bytes.max(4).next_multiple_of(4);
        let target=device.new_buffer(len,objc2_metal::MTLResourceOptions::StorageModeShared).unwrap();
        {
            let mut encoder=commands.blit_command_encoder().unwrap();
            // SAFETY: fixture keeps source+target alive and no other writer;
            // aligned disjoint actual buffers, read only after real D below.
            unsafe{encoder.copy_from_buffer(source,0,&target,0,len)}.unwrap();
        }
        commands.flush_and_wait().unwrap();
        assert_eq!(unsafe{target.raw_unchecked().storageMode()},MTLStorageMode::Shared);
        // SAFETY: completed blit into legal Shared storage, real owner retained.
        unsafe{std::slice::from_raw_parts(target.contents(),bytes)}.to_vec()
    }
    fn values(dtype:crate::DType,n:usize)->Vec<u8> {
        let mut out=Vec::with_capacity(n*dtype.size_in_bytes());
        for i in 0..n {
            // Nonnegative finite values fit every integer conversion. Include
            // fractional/half rounding cases; no undefined float→int input.
            let x=((i%127) as f32)+[0.0,0.125,0.499,0.501][i%4];
            match dtype {
                crate::DType::U8=>out.push(x as u8),
                crate::DType::U32=>out.extend_from_slice(&(x as u32).to_ne_bytes()),
                crate::DType::I64=>out.extend_from_slice(&(x as i64).to_ne_bytes()),
                crate::DType::F16=>out.extend_from_slice(&half::f16::from_f32(x).to_bits().to_ne_bytes()),
                crate::DType::BF16=>out.extend_from_slice(&half::bf16::from_f32(x).to_bits().to_ne_bytes()),
                crate::DType::F32=>out.extend_from_slice(&x.to_ne_bytes()),
                _=>unreachable!(),
            }
        }
        out
    }
    // Compile and dispatch the actual old/new entry bodies, not a restatement
    // of cast_value in Rust. Parameters use the SAME production binding codec.
    fn execute(device:&Device,commands:&Commands,pipeline:&crate::kernels::metal::ComputePipeline,
        geometry:CastGeometry,shape:&[usize],strides:&[usize],input:&Buffer,offset:usize,
        style:ParamStyle)->Vec<u8> {
        let output=private(device,geometry.output_bytes);
        {
            let guard=commands.command_encoder().unwrap();let encoder=guard.as_ref();
            // SAFETY: this fixture owns the live recording and exact retained pipeline.
            unsafe{encoder.set_compute_pipeline_state(pipeline)};
            // SAFETY: this fixture alone owns capture and holds staged bytes through D.
            let _capture=unsafe{begin_packed_params(encoder,style)};
            let input=BufferOffset{buffer:input,offset_in_bytes:offset};
            // SAFETY: actual validated extents/ABI below, all fixture owners
            // including staged packed bytes remain alive through flush+wait.
            unsafe{
                if geometry.strided {set_params!(encoder,(geometry.length,shape.len(),shape,strides,&input,Output::new(&output)));}
                else {set_params!(encoder,(geometry.length,&input,Output::new(&output)));}
            }
            // SAFETY: same genuine raw-fixture owners/parameter buffers,
            // retained through the actual wait before staged is dropped.
            let staged=unsafe{finish_packed_params(device,encoder,style,
                if geometry.strided{CAST_STRIDED_PARAMS_ALIGN}else{CAST_PARAMS_ALIGN})}.unwrap();
            let (mut groups,mut threads)=geometry.grid(pipeline).unwrap();
            if geometry.wide {
                // TEST-ONLY legal short mixed grid, not a production geometry
                // choice. Force nonzero x/y/z; every layout has padding lanes,
                // but not every layout has excess groups (1147 items/32 gives
                // exactly 36 groups). The old 1x1xG case never exercised group.y.
                let width=threads.width.min(32);
                assert!(width>0,"SETUP_INCOMPLETE_CAST_NATIVE_MIXED_GRID_REQUIRES_POSITIVE_WIDTH");
                let needed=geometry.work_items().div_ceil(width);
                assert!(needed>6,"SETUP_INCOMPLETE_CAST_NATIVE_MIXED_GRID_REQUIRES_ALL_AXES");
                threads=crate::kernels::MTLSize{width,height:1,depth:1};
                groups=crate::kernels::MTLSize{width:2,height:3,depth:needed.div_ceil(6)};
                eprintln!("CAST_NATIVE_WIDE_MIXED_GRID groups={}x{}x{} threads={} work={}",
                    groups.width,groups.height,groups.depth,threads.width,geometry.work_items());
            }
            // SAFETY: exact shader schema/extents/nonzero checked grid above;
            // every native input/output/parameter owner remains through actual D.
            unsafe{encoder.dispatch_thread_groups(groups,threads)};
            drop(guard);
            commands.flush_and_wait().unwrap();
            drop(staged);
        }
        readback(device,commands,&output,geometry.output_bytes)
    }

    #[test]
    #[ignore="native explicit LLOOM_METAL_API=metal3|metal4; one isolated GPU process"]
    fn actual_cast_padding_threads_write_every_output_exactly_once() {
        let _autorelease=crate::AutoreleaseScope::new();
        let (device,commands)=setup();
        let production=library(&device,include_str!("../metal_src/cast.metal"));
        assert!(!production.function_names().iter().any(|n|n=="cast_count_writers"),
            "CAST_TEST_OBSERVER_MUST_NOT_SHIP");
        let fragment=format!("#define LLOOM_CAST_WRITE_COUNTS\n{}",include_str!("../metal_src/cast.metal"));
        let counted=library(&device,&fragment);
        let pipeline=pipeline(&device,&counted,"cast_count_writers");
        let width=pipeline.max_total_threads_per_threadgroup();
        assert!(width>1,"SETUP_INCOMPLETE_CAST_WRITE_CENSUS_REQUIRES_ACTUAL_PADDING width={width}");
        let length=width.checked_mul(2).unwrap()+1;
        let geometry=CastGeometry::contiguous(length,4,4).unwrap();
        let input=shared(&device,&values(crate::DType::F32,length));
        let output=private(&device,length*4);
        let zero=vec![0u8;length*4];let counts=shared(&device,&zero);
        {
            let guard=commands.command_encoder().unwrap();let encoder=guard.as_ref();
            // SAFETY: live fixture recording and exact retained counted-test pipeline.
            unsafe{encoder.set_compute_pipeline_state(&pipeline)};
            // SAFETY: frozen test-only schema (len, input, output, uint counts),
            // exact extents, all actual native owners retained through D.
            unsafe{set_params!(encoder,(length,&BufferOffset::zero_offset(&input),Output::new(&output),Output::new(&counts)));}
            let (groups,threads)=geometry.grid(&pipeline).unwrap();
            // SAFETY: exact shader schema/extents/nonzero checked grid above;
            // every native input/output/parameter owner remains through actual D.
            unsafe{encoder.dispatch_thread_groups(groups,threads)};
        }
        commands.flush_and_wait().unwrap();
        assert_eq!(unsafe{counts.raw_unchecked().storageMode()},MTLStorageMode::Shared);
        // SAFETY: Shared counts, GPU complete, no writer/future emitter survives.
        let actual=unsafe{std::slice::from_raw_parts(counts.contents().cast::<u32>(),length)};
        for (i,&count) in actual.iter().enumerate(){
            assert_eq!(count,1,"CAST_PADDING_THREADS_MUST_NOT_DUPLICATE_WRITERS index={i} width={width} length={length}");
        }
        assert_eq!(readback(&device,&commands,&output,length*4),values(crate::DType::F32,length));
        eprintln!("CAST_NATIVE_WRITE_CENSUS actual_api={} width={width} length={length} outputs={length} every_count=1 production_observer=false",commands.actual_command_api());
    }

    #[test]
    #[ignore="native explicit LLOOM_METAL_API=metal3|metal4; one isolated GPU process"]
    fn actual_cast_legacy_narrow_wide_layout_and_parameter_styles_keep_bits() {
        let _autorelease=crate::AutoreleaseScope::new();
        let (device,commands)=setup();
        let old=library(&device,LEGACY_FRAGMENT);
        let new=library(&device,include_str!("../metal_src/cast.metal"));
        let types=[crate::DType::U8,crate::DType::U32,crate::DType::I64,crate::DType::F16,crate::DType::BF16,crate::DType::F32];
        let mut cells=0;
        for input_type in types {for output_type in types {
            if input_type==output_type{continue;}
            for layout in 0..3 {
                let shape=[31,37];let strides=[79,2];
                let strided=layout==2;let offset=if layout==0{0}else{3*input_type.size_in_bytes()};
                let shape=if strided{&shape[..]}else{&[][..]};
                let strides=if strided{&strides[..]}else{&[][..]};
                let geometry=if strided{CastGeometry::strided(shape,strides,input_type.size_in_bytes(),output_type.size_in_bytes()).unwrap()}
                    else{CastGeometry::contiguous(2051,input_type.size_in_bytes(),output_type.size_in_bytes()).unwrap()};
                let data=values(input_type,3+geometry.source_bytes/input_type.size_in_bytes());
                let input=shared(&device,&data);
                let name=cast_kernel_name(input_type,output_type,strided).unwrap();let spec=CastKernelSpec::named(name).unwrap();
                for style in [ParamStyle::Split,ParamStyle::Packed] {
                    let baseline=pipeline(&device,&old,spec.name(false,style));
                    let expected=execute(&device,&commands,&baseline,geometry,shape,strides,&input,offset,style);
                    for wide in [false,true] {
                        let candidate=pipeline(&device,&new,spec.name(wide,style));
                        let mut actual_geometry=geometry;actual_geometry.wide=wide;
                        let actual=execute(&device,&commands,&candidate,actual_geometry,shape,strides,&input,offset,style);
                        assert_eq!(actual,expected,"CAST_SAME_MATH_BITS input={input_type:?} output={output_type:?} layout={layout} style={style:?} wide={wide}");
                        cells+=1;
                    }
                }
            }
        }}
        assert_eq!(cells,360,"all 30 supported pairs x3 layouts x2 styles x2 widths");
        eprintln!("CAST_NATIVE_PARITY actual_api={} cells={cells} legacy_fragment_sha256=dcbb18fb7e5a329414c5f5663719fea9c1d6dd9f240ecea7a54887ae66fd5052 test_scope=kernel_only",commands.actual_command_api());
    }
}
