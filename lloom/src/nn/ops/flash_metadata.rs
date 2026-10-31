//! Authenticated immutable upload metadata for FlashDecoding. The frame
//! producer seam and resident owner are separate lifetime contracts.
//! A private child module of nn::ops, whose parent alone can bind
//! its tensors. Public callers receive host slices, never Tensor aliases.
use crate::nn::attention::flash_validation::{prepare_pages, PageSummary, Pages};
use lloom_tensor::{Device, Result, Tensor};

// No production caller until the #1147 owner migration is reviewed. The
// module exercises the actual publication mechanism, not a second allocator.
#[allow(dead_code)]
#[path = "flash_metadata/resident.rs"]
mod resident;

/// A published table/count pair with one authenticated source of values.
///
/// Owns the allocations the caller already made for per-pass metadata. No
/// cloning, additional table copy, per-layer upload or per-layer scan is needed.
/// The type deliberately does not implement Clone and exports no device Tensor.
///
/// ```
/// use lloom::nn::ops::FlashDecodingMetadata;
/// use lloom_tensor::Device;
/// let metadata = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
/// assert_eq!(metadata.table(), &[0]);
/// assert_eq!(metadata.keys(), &[1]);
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
/// Device tensors are not available to a safe caller (including mutation via
/// Tensor's `&self` setters):
/// ```compile_fail,E0616
/// use lloom::nn::ops::FlashDecodingMetadata;
/// use lloom_tensor::Device;
/// let metadata = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
/// let _escaped_table_alias = &metadata.table_t;
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
pub struct FlashDecodingMetadata {
    table: Vec<u32>,
    keys: Vec<u32>,
    table_stride: usize,
    summary: PageSummary,
    table_t: Tensor,
    keys_t: Tensor,
}

impl std::fmt::Debug for FlashDecodingMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Tensor::Debug reads scalar/small U32 tensors back to the host; an
        // ownership/geometry diagnostic must not accidentally synchronize.
        f.debug_struct("FlashDecodingMetadata")
            .field("table_entries", &self.table.len())
            .field("count_entries", &self.keys.len())
            .field("table_stride", &self.table_stride)
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}

impl FlashDecodingMetadata {
    /// Publish one immutable metadata pair, validated before either upload.
    /// Per-pass callers move their two host vectors into this constructor;
    /// publication performs one table upload and one live-count upload.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: Vec<u32>,
        table_stride: usize,
        keys: Vec<u32>,
        rows: usize,
        page_size: usize,
        pages_per_chunk: usize,
        device: &Device,
    ) -> Result<Self> {
        let summary = prepare_pages(
            Pages {
                entries: &table,
                stride: table_stride,
                keys: &keys,
            },
            rows,
            page_size,
            pages_per_chunk,
        )?;
        let table_t = Tensor::new(table.as_slice(), device)?;
        let keys_t = Tensor::new(keys.as_slice(), device)?;
        Ok(Self {
            table,
            keys,
            table_stride,
            summary,
            table_t,
            keys_t,
        })
    }

    /// Host table; immutable, including while native consumers are queued.
    pub fn table(&self) -> &[u32] {
        &self.table
    }
    /// Host counts; immutable, including while native consumers are queued.
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }
    pub fn table_stride(&self) -> usize {
        self.table_stride
    }

    // This module must remain a private child of nn::ops. In particular do not
    // widen these accessors to pub(crate): safe caller mutation through a Tensor
    // alias would invalidate the authenticated publication proof.
    pub(super) fn route(&self) -> (usize, usize) {
        self.summary.route()
    }
    pub(super) fn summary(&self) -> &PageSummary {
        &self.summary
    }
    pub(super) fn bindings(&self) -> (&Tensor, &Tensor) {
        (&self.table_t, &self.keys_t)
    }
}

#[cfg(test)]
mod flash_validation_tests {
    use super::*;
    use crate::nn::attention::flash_validation::preparation_count;
    use crate::nn::ops::{flash_decoding_paged_metadata, FlashScratchSizing};
    use lloom_tensor::DType;

    #[test]
    fn upload_owner_moves_host_vectors_and_initializes_matching_values() -> Result<()> {
        let table = vec![2, 0, 1, 3];
        let keys = vec![1, 257];
        let (table_ptr, keys_ptr) = (table.as_ptr(), keys.as_ptr());
        let owner = FlashDecodingMetadata::new(table, 2, keys, 2, 256, 1, &Device::Cpu)?;
        assert_eq!(owner.table().as_ptr(), table_ptr);
        assert_eq!(owner.keys().as_ptr(), keys_ptr);
        assert_eq!(owner.bindings().0.to_vec1::<u32>()?, owner.table());
        assert_eq!(owner.bindings().1.to_vec1::<u32>()?, owner.keys());
        Ok(())
    }

    #[test]
    fn cached_owner_does_not_rescan_metadata_for_each_layer() -> Result<()> {
        let device = Device::Cpu;
        let owner = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &device)?;
        let prepared = preparation_count();
        let q = Tensor::zeros((1, 1, 1, 64), DType::F32, &device)?;
        let kv = Tensor::zeros((1, 1, 256, 64), DType::F32, &device)?;
        for _ in 0..2 {
            let error = flash_decoding_paged_metadata(
                &q,
                &kv,
                &kv,
                1.,
                1.,
                FlashScratchSizing::Grow,
                0,
                &owner,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("no cpu support"), "{error}");
        }
        assert_eq!(preparation_count(), prepared);
        Ok(())
    }

    #[test]
    fn each_layer_pool_is_checked_against_the_same_published_span() -> Result<()> {
        let device = Device::Cpu;
        let owner = FlashDecodingMetadata::new(vec![1], 1, vec![1], 1, 256, 1, &device)?;
        let q = Tensor::zeros((1, 1, 1, 64), DType::F32, &device)?;
        let k = Tensor::zeros((1, 1, 512, 64), DType::F32, &device)?;
        let short_v = Tensor::zeros((1, 1, 256, 64), DType::F32, &device)?;
        let error = flash_decoding_paged_metadata(
            &q,
            &k,
            &short_v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &owner,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("page span exceeds K/V pool"), "{error}");
        Ok(())
    }

    #[test]
    fn owner_diagnostics_do_not_delegate_to_device_tensor_debug() -> Result<()> {
        let owner = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
        let printed = format!("{owner:?}");
        assert!(!printed.contains("Tensor["), "{printed}");
        assert!(printed.contains("table_entries"));
        Ok(())
    }
}
