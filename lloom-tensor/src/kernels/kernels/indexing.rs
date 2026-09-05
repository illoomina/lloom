use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, GatherParams, IndexAddParams, IndexParams,
    ParamStyle, ScatterParams,
};
use crate::kernels::linear_split;
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `indexing_params_layout_matches_metal` is what proves those
/// mirrors agree with `indexing.metal`.
///
/// `IndexParams` is the one where this does work rather than being a formality:
/// its fields sum to 49 and it is 56, because a `bool` sits between two
/// `size_t`. Without the trailing pad the capture would hand the kernel 49
/// bytes for a 56-byte read.
const INDEX_PARAMS_ALIGN: usize = core::mem::align_of::<IndexParams>();
const GATHER_PARAMS_ALIGN: usize = core::mem::align_of::<GatherParams>();
const SCATTER_PARAMS_ALIGN: usize = core::mem::align_of::<ScatterParams>();
const INDEX_ADD_PARAMS_ALIGN: usize = core::mem::align_of::<IndexAddParams>();

// A note on widths, since #38 found a latent mismatch of exactly this kind in
// `reduce.metal`'s callers and predicted it in the remaining files.
//
// `indexing.metal` declares 24 `constant size_t &` -- 8 bytes -- and one
// `constant bool &`. Every `call_*` below binds a `usize` (8) or a `bool`
// respectively, so the mismatch #38 describes (a `usize` bound to a
// `constant uint &`, benign under `setBytes` because it writes 8 bytes and a
// little-endian `uint` read takes the low 4, and **wrong once packed**) does
// **not** occur in this family. Checked by enumerating every `constant &`
// parameter in the file rather than assumed; there is no `uint` in it.
//
// The other latent #41 found does not occur either: none of the four entry
// points passes `()` for an optional buffer, so nothing here binds nothing
// while consuming no slot -- the failure that shifts every later binding one
// slot low and hangs the GPU rather than corrupting silently.
//
// # `left_size` is a host-side local and never was a kernel parameter's worth
//
// It was 29 `constant size_t &` before #219, and the five it loses are the
// `left_size` one per kernel. **The name survives in this file** -- four times,
// as a local -- and that is the finding rather than an oversight: `left_size`
// is a *factor of the grid extent*, `dst_el`, which the host computes and
// `linear_split` consumes. The kernel never needed the factor because it is
// handed the product; every body recovers what it wants from `tid /
// right_size`, which is why the mutation in §11.3k finding 3 survived.
//
// So removing it takes nothing away from the GPU side. The value that was bound
// is still computed, still used, and still on the host, one line above the call
// that no longer passes it.

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    contiguous: bool,
    src_dims: &[usize],
    src_strides: &[usize],
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_index_select_with(
        device,
        ep,
        kernels,
        name,
        shape,
        ids_size,
        dim,
        contiguous,
        src_dims,
        src_strides,
        input,
        ids,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_index_select`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
///
/// # Where the style parameter lives, and why it is here rather than a crate up
///
/// This family chooses its `[[host_name]]` in `candle-core`, where every other
/// family chooses it here (`DESIGN.md` §8.1e). #64 moved the *registry* down
/// and deliberately left the style un-threaded, noting that "`_packed` has to
/// be appended where the name is chosen" -- which reads as though `candle-core`
/// must learn about binding styles.
///
/// It does not, and the reason is that "where the name is chosen" and "where
/// the *suffix* is appended" are different places. `candle-core` picks a
/// classical `&'static str` from [`crate::kernels::IndexingKernel`] and hands it over;
/// `style.kernel_name(name)` below is what turns that into a pipeline name, and
/// it runs *here*. So the style parameter belongs on this entry point, exactly
/// as it does for `conv`, and `candle-core` is untouched -- it keeps calling
/// [`call_index_select`], which is [`ParamStyle::Split`] and byte-for-byte what
/// shipped.
///
/// That is the same shape every other converted family has, so the crate
/// boundary turns out not to be a special case for *this* axis at all. It was
/// a special case for the registry, because a test can only resolve names
/// against the metallib from inside this crate.
#[allow(clippy::too_many_arguments)]
pub fn call_index_select_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    contiguous: bool,
    src_dims: &[usize],
    src_strides: &[usize],
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;
    // The source tensor's rank, which is what the kernel's `get_strided_index`
    // walks. Derived from the slice rather than taken as a parameter so the
    // public signature is unchanged and the two cannot disagree: `src_dims` and
    // `src_strides` are the same layout's arrays, so their common length *is*
    // the rank. The kernel previously received `src_dim_size` here -- see the
    // comment at the call site in `indexing.metal`.
    let src_num_dims = src_dims.len();
    debug_assert_eq!(
        src_dims.len(),
        src_strides.len(),
        "index_select: dims and strides describe one layout and must agree in length"
    );

    let pipeline = kernels.load_pipeline(device, Source::Indexing, style.kernel_name(name))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "index_select {name} dim={dim} dst_el={dst_el}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            src_dim_size,
            right_size,
            ids_size,
            contiguous,
            src_num_dims,
            src_dims,
            src_strides,
            &input,
            &ids,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, INDEX_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_gather_with(
        device,
        ep,
        kernels,
        name,
        shape,
        ids_size,
        dim,
        input,
        ids,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_gather`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_gather_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, style.kernel_name(name))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gather {name} dim={dim} dst_el={dst_el}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            src_dim_size,
            right_size,
            ids_size,
            &input,
            &ids,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, GATHER_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_scatter(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_scatter_with(
        device,
        ep,
        kernels,
        name,
        src_shape,
        dst_shape,
        dim,
        input,
        ids,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_scatter`], choosing how the scalars are bound.
///
/// Serves `scatter` and `scatter_add` alike -- `candle-core` picks between them
/// by name, and the two bind the same five scalars, which is why they share
/// [`ScatterParams`].
#[allow(clippy::too_many_arguments)]
pub fn call_scatter_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, style.kernel_name(name))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "scatter {name} dim={dim} dst_el={dst_el}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            src_dim_size,
            right_size,
            dst_dim_size,
            &input,
            &ids,
            Output::from_buffer_offset(&output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SCATTER_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_add(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    ids_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_index_add_with(
        device,
        ep,
        kernels,
        name,
        src_shape,
        dst_shape,
        ids_shape,
        dim,
        input,
        ids,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_index_add`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_index_add_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    ids_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];
    let ids_dim_size = ids_shape[0];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, style.kernel_name(name))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "index_add {name} dim={dim} dst_el={dst_el}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            src_dim_size,
            right_size,
            dst_dim_size,
            ids_dim_size,
            &input,
            &ids,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, INDEX_ADD_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
