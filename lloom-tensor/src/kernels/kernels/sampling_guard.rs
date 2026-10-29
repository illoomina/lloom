use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    set_params, Buffer, ComputeCommandEncoder, Kernels, MetalKernelError, Output, Source,
};

pub(crate) enum Consumer<'a> {
    Eos {
        ids: &'a Buffer,
        error: &'a Buffer,
        eos_ids: &'a Buffer,
        flag: &'a Buffer,
        count: u32,
    },
    Validate {
        ids: &'a Buffer,
        status: &'a Buffer,
        error: &'a Buffer,
        rows: u32,
        vocab: u32,
    },
    Embedding {
        ids: &'a Buffer,
        error: &'a Buffer,
        table: (&'a Buffer, usize),
        out: (&'a Buffer, usize),
        row_bytes: u64,
        bytes: u64,
    },
    QuantizedEmbedding {
        ids: &'a Buffer,
        error: &'a Buffer,
        table: &'a Buffer,
        out: (&'a Buffer, usize),
        dtype: super::quantized::GgmlDType,
        hidden: usize,
        row_stride: usize,
        rows: usize,
    },
    Copy {
        ids: &'a Buffer,
        error: &'a Buffer,
        out: (&'a Buffer, usize),
        rows: u32,
    },
    Counts {
        ids: &'a Buffer,
        error: &'a Buffer,
        counts: &'a Buffer,
        rows: u32,
    },
    Feed {
        ids: &'a Buffer,
        error: &'a Buffer,
        forced: &'a Buffer,
        close: &'a Buffer,
        state: &'a Buffer,
        out: &'a Buffer,
        free: u32,
        forced_len: u32,
        close_len: u32,
        vocab: u32,
    },
    Marker {
        error: &'a Buffer,
        marker: &'a Buffer,
        generation: u32,
        history: &'a Buffer,
        input: &'a Buffer,
        eos: &'a Buffer,
    },
}

/// Encode a consumer whose first action is a device-side validity guard.
///
/// # Safety
/// Every span is validated by the carrier, all outputs are disjoint from inputs
/// (read/write state is intentional), and each buffer keeps its originating
/// allocation authority through the last encoded reader. A Validate precedes
/// every consumer on this queue. Error/state buffers are request-exclusive;
/// captured replay requires a separate generation-ownership contract.
pub(crate) unsafe fn call(
    device: &crate::kernels::Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    consumer: Consumer<'_>,
) -> Result<(), MetalKernelError> {
    let (name, count) = match &consumer {
        Consumer::QuantizedEmbedding {
            ids,
            error,
            table,
            out,
            dtype,
            hidden,
            row_stride,
            rows,
        } => {
            if *rows == 0 {
                return Ok(());
            }
            // SAFETY: the opaque carrier checks packed table/output extents,
            // ids, exact device/origin and sticky request ownership first.
            return unsafe {
                super::quantized::call_quantized_get_rows_guarded(
                    device,
                    ep,
                    kernels,
                    *dtype,
                    *hidden,
                    *row_stride,
                    *rows,
                    table,
                    ids,
                    0,
                    out.0,
                    out.1,
                    error,
                )
            };
        }
        Consumer::Eos { .. } => ("sampling_guarded_eos", 1),
        Consumer::Validate { .. } => ("sampling_validate_tokens", 1),
        Consumer::Embedding { bytes, .. } => ("sampling_guarded_embedding", *bytes as usize),
        Consumer::Copy { rows, .. } => ("sampling_guarded_copy", *rows as usize),
        Consumer::Counts { .. } => ("sampling_guarded_counts", 1),
        Consumer::Feed { .. } => ("sampling_guarded_feed", 1),
        Consumer::Marker { .. } => ("sampling_guarded_marker", 1),
    };
    if count == 0 {
        return Ok(());
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let limit = if count > u32::MAX as usize - 255 {
        1
    } else {
        256
    };
    let width = pipeline
        .max_total_threads_per_threadgroup()
        .min(limit)
        .min(count);
    if width == 0 {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "{name}: zero native width"
        )));
    }
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    // SAFETY: carrier preflight covers every buffer and scalar span above.
    unsafe {
        match consumer {
            Consumer::QuantizedEmbedding { .. } => unreachable!("lowered through shared get_rows"),
            Consumer::Eos {
                ids,
                error,
                eos_ids,
                flag,
                count,
            } => {
                set_params!(encoder, (ids, error, eos_ids, Output::new(flag), count));
            }
            Consumer::Validate {
                ids,
                status,
                error,
                rows,
                vocab,
            } => {
                set_params!(encoder, (ids, status, Output::new(error), rows, vocab));
            }
            Consumer::Embedding {
                ids,
                error,
                table,
                out,
                row_bytes,
                bytes,
            } => {
                set_params!(
                    encoder,
                    (
                        ids,
                        error,
                        table,
                        Output::with_offset(out.0, out.1),
                        row_bytes,
                        bytes
                    )
                );
            }
            Consumer::Copy {
                ids,
                error,
                out,
                rows,
            } => {
                set_params!(
                    encoder,
                    (ids, error, Output::with_offset(out.0, out.1), rows)
                );
            }
            Consumer::Counts {
                ids,
                error,
                counts,
                rows,
            } => {
                set_params!(encoder, (ids, error, Output::new(counts), rows));
            }
            Consumer::Feed {
                ids,
                error,
                forced,
                close,
                state,
                out,
                free,
                forced_len,
                close_len,
                vocab,
            } => {
                set_params!(
                    encoder,
                    (
                        ids,
                        Output::new(error),
                        forced,
                        close,
                        Output::new(state),
                        Output::new(out),
                        free,
                        forced_len,
                        close_len,
                        vocab
                    )
                );
            }
            Consumer::Marker {
                error,
                marker,
                generation,
                history,
                input,
                eos,
            } => {
                set_params!(
                    encoder,
                    (error, Output::new(marker), generation, history, input, eos)
                );
            }
        }
    }
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: count.div_ceil(width),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
