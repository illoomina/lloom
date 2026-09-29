use crate::kernels::metal::{Buffer, CommandsGuard, ComputeCommandEncoder, ComputePipeline};
use crate::kernels::MTLSize;
use std::ffi::OsStr;
use std::ops::Deref;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub struct WrappedEncoder<'a> {
    inner: &'a ComputeCommandEncoder,
    end_encoding_on_drop: bool,
}

impl Drop for WrappedEncoder<'_> {
    fn drop(&mut self) {
        if self.end_encoding_on_drop {
            self.inner.end_encoding()
        }
    }
}

impl AsRef<ComputeCommandEncoder> for WrappedEncoder<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.inner
    }
}

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
pub(crate) fn linear_split(pipeline: &ComputePipeline, length: usize) -> (MTLSize, MTLSize) {
    let size = length;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = size.div_ceil(width);
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

// https://github.com/ml-explore/mlx/blob/bddf23f175726a57f0e443cd45518c0757daa166/mlx/backend/metal/utils.h#L96
pub fn get_block_dims(dim0: usize, dim1: usize, dim2: usize) -> MTLSize {
    let mut pows0 = 0;
    let mut pows1 = 0;
    let mut pows2 = 0;
    let mut sum = 0;
    loop {
        let presum = sum;
        // Check all the pows
        if dim0 >= (1 << (pows0 + 1)) {
            pows0 += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim1 >= (1 << (pows1 + 1)) {
            pows1 += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim2 >= (1 << (pows2 + 1)) {
            pows2 += 1;
            sum += 1;
        }
        if sum == presum || sum == 10 {
            break;
        }
    }
    MTLSize {
        width: 1 << pows0,
        height: 1 << pows1,
        depth: 1 << pows2,
    }
}

/// Calculate preferred tile size given the size of a data type in bytes.
/// f32 -> 2, f16 -> 4, u8 -> 8.
#[inline(always)]
pub fn get_tile_size(dtype_size: usize) -> usize {
    1.max(8 / dtype_size)
}

pub fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoder, position: usize, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self);
}
/// Lowers a primitive to the encoder, as either an inline constant or a field
/// of a packed params block.
///
/// This one function is the entire Rust half of issue #38. An ICB command has
/// no `setBytes` in any form (`DESIGN.md` §3.7b), so a kernel whose scalars
/// arrive that way cannot be encoded into one; routing them into a buffer is
/// the prerequisite. Because every inline constant in the crate already funnels
/// through here -- 56 `set_params!` sites across 51 kernel entry points -- the
/// diversion is invisible to all of them.
///
/// `capture_scalar` returns `false` unless a caller has opened a capture, so
/// the classical path is the same `set_bytes` call it always was, reached after
/// one relaxed atomic load.
///
/// The `to_ne_bytes`-shaped lowering is deliberate: the packed block has to
/// match the C++ struct the kernel reads, so each field is appended at its own
/// alignment. `usize` is 8 bytes here and maps to MSL's `size_t`, which is
/// also 8 -- see `kernels/params.rs` for why the RoPE mirrors are `u64` rather
/// than narrowed to `u32`.
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
                let bytes = data.to_ne_bytes();
                if encoder.capture_scalar(&bytes, core::mem::align_of::<Self>()) {
                    return;
                }
                encoder.set_bytes(position, &data);
            }
        }
    };
}
/// `bool` has no `to_ne_bytes`, and it is also the one primitive whose packed
/// width is worth stating out loud: **MSL's `bool` is 1 byte**, matching Rust's,
/// so a `bool` field packs as a single byte and pads to whatever follows it.
/// That is the second of the two hazards issue #38 names, and it is handled
/// here rather than being left for a future family to trip over.
///
/// No kernel in `reduce.metal` binds a `bool`; this exists so the trait stays
/// total, and so the width is recorded at the point that decides it.
impl EncoderParam for bool {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        let bytes = [data as u8];
        if encoder.capture_scalar(&bytes, core::mem::align_of::<Self>()) {
            return;
        }
        encoder.set_bytes(position, &data);
    }
}

primitive!(usize);
primitive!(i32);
primitive!(i64);
primitive!(u8);
primitive!(u32);
primitive!(u64);
primitive!(f32);
primitive!(f64);
primitive!(half::bf16);
primitive!(half::f16);

pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl<'a> BufferOffset<'a> {
    pub fn zero_offset(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset_in_bytes: 0,
        }
    }
}

/// Arrays -- `dims` and `strides` -- are the one thing that does *not* fold
/// into a packed params struct, and the reason is structural rather than
/// incidental: their length is a property of the tensor's layout, so a
/// fixed-layout struct cannot hold them. They stay a separate binding.
///
/// That is not a gap in the ICB story. `setKernelBuffer` binds a buffer of any
/// length; what an ICB command cannot do is `setBytes` (`DESIGN.md` §3.7b). So
/// under capture these are promoted to a device buffer -- which an ICB *can*
/// express -- rather than being packed or left inline.
impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        if encoder.capture_array(core::mem::size_of_val(data), data.as_ptr().cast()) {
            return;
        }
        encoder.set_bytes_directly(position, core::mem::size_of_val(data), data.as_ptr().cast());
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data.buffer), data.offset_in_bytes);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for () {
    fn set_param(_: &ComputeCommandEncoder, _: usize, _: Self) {}
}

/// Marks a buffer as a read input in `set_params!` calls, enabling hazard tracking.
///
/// # Examples
/// ```ignore
/// set_params!(encoder, (length, Input::new(input), output));
/// ```
#[derive(Copy, Clone)]
pub struct Input<'a> {
    buffer: &'a Buffer,
    offset: usize,
}

impl<'a> Input<'a> {
    #[inline]
    pub fn new(buffer: &'a Buffer) -> Self {
        Self { buffer, offset: 0 }
    }

    #[inline]
    pub fn with_offset(buffer: &'a Buffer, offset: usize) -> Self {
        Self { buffer, offset }
    }

    #[inline]
    pub fn from_buffer_offset(bo: &'a BufferOffset<'a>) -> Self {
        Self {
            buffer: bo.buffer,
            offset: bo.offset_in_bytes,
        }
    }
}

impl<'a> EncoderParam for Input<'a> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_input_buffer(position, Some(data.buffer), data.offset);
    }
}

/// Marks a buffer as a write output in `set_params!` calls, enabling hazard tracking.
///
/// Use this wrapper wherever a kernel writes to a buffer so the encoder can detect
/// read-after-write (RAW) hazards and insert barriers before concurrent dispatches.
///
/// # Examples
/// ```ignore
/// set_params!(encoder, (length, &input, Output::new(output)));
/// set_params!(encoder, (k, m, n, Output::from_buffer_offset(dst_bo)));
/// set_params!(encoder, (n, Output::with_offset(dst, dst_offset)));
/// ```
#[derive(Copy, Clone)]
pub struct Output<'a> {
    buffer: &'a Buffer,
    offset: usize,
}

impl<'a> Output<'a> {
    #[inline]
    pub fn new(buffer: &'a Buffer) -> Self {
        Self { buffer, offset: 0 }
    }

    #[inline]
    pub fn with_offset(buffer: &'a Buffer, offset: usize) -> Self {
        Self { buffer, offset }
    }

    #[inline]
    pub fn from_buffer_offset(bo: &'a BufferOffset<'a>) -> Self {
        Self {
            buffer: bo.buffer,
            offset: bo.offset_in_bytes,
        }
    }
}

impl<'a> EncoderParam for Output<'a> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_output_buffer(position, Some(data.buffer), data.offset);
    }
}

#[macro_export]
macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            $crate::kernels::utils::set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

/// Scope a Metal debug group around the current dispatch, e.g.
/// `debug_group!(encoder, "kernel {a}")`. The guard binds into the caller's
/// block (like `set_params!`), so the group pops after the dispatch. Gated at
/// the macro definition: off-feature it expands to nothing.
#[cfg(feature = "debug-labels")]
#[macro_export]
macro_rules! debug_group {
    ($enc:expr, $($arg:tt)+) => {
        let _debug_group_guard = $enc.debug_group(&::std::format!($($arg)+));
    };
}

#[cfg(not(feature = "debug-labels"))]
#[macro_export]
macro_rules! debug_group {
    ($enc:expr, $($arg:tt)+) => {};
}

/// Apply a persistent `MTLBuffer`/queue debug label, e.g.
/// `metal_label!(buffer, "name")`. Gated like [`debug_group!`].
#[cfg(feature = "debug-labels")]
#[macro_export]
macro_rules! metal_label {
    ($obj:expr, $($arg:tt)+) => {
        $obj.set_label(&::std::format!($($arg)+));
    };
}

#[cfg(not(feature = "debug-labels"))]
#[macro_export]
macro_rules! metal_label {
    ($obj:expr, $($arg:tt)+) => {};
}

pub trait EncoderProvider {
    type Encoder<'a>: AsRef<ComputeCommandEncoder>
    where
        Self: 'a;

    fn encoder(&self) -> Self::Encoder<'_>;
}

impl EncoderProvider for &ComputeCommandEncoder {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self,
            end_encoding_on_drop: false,
        }
    }
}

impl EncoderProvider for &CommandsGuard<'_> {
    type Encoder<'a>
        = &'a CommandsGuard<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self
    }
}

pub enum RwLockGuard<'a, T> {
    Read(RwLockReadGuard<'a, T>),
    Write(RwLockWriteGuard<'a, T>),
}

impl<'a, T> Deref for RwLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            RwLockGuard::Read(g) => g.deref(),
            RwLockGuard::Write(g) => g.deref(),
        }
    }
}

impl<'a, T> From<RwLockReadGuard<'a, T>> for RwLockGuard<'a, T> {
    fn from(g: RwLockReadGuard<'a, T>) -> Self {
        RwLockGuard::Read(g)
    }
}

impl<'a, T> From<RwLockWriteGuard<'a, T>> for RwLockGuard<'a, T> {
    fn from(g: RwLockWriteGuard<'a, T>) -> Self {
        RwLockGuard::Write(g)
    }
}

fn is_truthy(s: String) -> bool {
    match s.as_str() {
        "true" | "t" | "yes" | "y" | "1" => true,
        _ => false,
    }
}

pub(crate) fn get_env_bool<K: AsRef<OsStr>>(key: K, default: bool) -> bool {
    std::env::var(key).map(is_truthy).unwrap_or(default)
}

/// One threadgroup per `groups` items of `width` threads.
pub(crate) fn dispatch_groups(encoder: &ComputeCommandEncoder, groups: usize, width: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups.max(1),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

/// A `gx × gy` grid of threadgroups of `width` threads.
pub(crate) fn dispatch_groups_2d(
    encoder: &ComputeCommandEncoder,
    gx: usize,
    gy: usize,
    width: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: gx.max(1),
            height: gy.max(1),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

/// A threadgroup of one head's `d` lanes: at least a simdgroup, a power of two.
pub(crate) fn head_width(d: usize, pipeline: &crate::kernels::ComputePipeline) -> usize {
    d.max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup())
}
