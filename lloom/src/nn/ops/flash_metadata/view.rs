//! One concrete immutable borrowed publication shape for every safe owner.
use crate::nn::attention::flash_validation::PageSummary;
use lloom_tensor::Tensor;

/// A borrowed, authenticated table/count publication for one FlashDecoding pass.
///
/// Obtained only from an owned upload or resident owner. Its private device
/// payloads cannot be replaced or mutated by safe callers. Every layer still
/// validates its own Q/K/V geometry and pool capacity against the cached span.
/// This is a fixed stack view, not a trait object, refcount or second upload.
///
/// A frame-backed constructor is deliberately absent until the producer owns
/// a scoped lease covering both future encoding and already-queued readers.
///
/// Native payloads are not available to a safe caller:
/// ```compile_fail,E0616
/// use lloom::nn::ops::FlashDecodingMetadata;
/// use lloom_tensor::Device;
/// let metadata = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
/// let view = metadata.view();
/// let _escaped_native_table = view.table_t;
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
pub struct FlashDecodingMetadataView<'a> {
    pub(super) table: &'a [u32],
    pub(super) keys: &'a [u32],
    pub(super) table_stride: usize,
    pub(super) summary: PageSummary,
    pub(super) table_t: &'a Tensor,
    pub(super) keys_t: &'a Tensor,
}

impl std::fmt::Debug for FlashDecodingMetadataView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlashDecodingMetadataView")
            .field("table_entries", &self.table.len())
            .field("count_entries", &self.keys.len())
            .field("table_stride", &self.table_stride)
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}
impl FlashDecodingMetadataView<'_> {
    pub fn table(&self) -> &[u32] {
        self.table
    }
    pub fn keys(&self) -> &[u32] {
        self.keys
    }
    pub fn table_stride(&self) -> usize {
        self.table_stride
    }
    pub fn page_geometry(&self) -> (usize, usize) {
        self.summary.route()
    }
    pub(in crate::nn::ops) fn summary(&self) -> &PageSummary {
        &self.summary
    }
    pub(in crate::nn::ops) fn bindings(&self) -> (&Tensor, &Tensor) {
        (self.table_t, self.keys_t)
    }
}
