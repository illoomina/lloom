//! Authenticated immutable upload metadata for FlashDecoding. The frame
//! producer seam and resident owner are separate lifetime contracts.
//! A private child module of nn::ops, whose parent alone can bind
//! its tensors. Public callers receive host slices, never Tensor aliases.
use crate::nn::attention::flash_validation::{prepare_pages, PageSummary, Pages};
use lloom_tensor::{Device, Result, Tensor};

/// Only table publications use the pre-existing by-value cache. Counts keep
/// their existing once/pass ordinary upload. This tag is publication-local,
/// never consulted by a layer or native dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetadataUpload {
    Table,
    Counts,
}

fn upload_metadata(data: &[u32], device: &Device, kind: MetadataUpload) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if kind == MetadataUpload::Table {
        if let Device::Metal(metal) = device {
            let constant = metal.constant_buffer(super::slice_bytes(data))?;
            // SAFETY: exact initialized U32 bytes from the sealed cache. The
            // counted creator handle transfers into this PRIVATE Tensor; no
            // native payload or mutable alias is exposed by metadata/view APIs.
            // Every consumer reads only, with ordinary Tensor retirement.
            let handle = unsafe { constant.into_handle_unchecked() };
            // Match ordinary storage_from_slice: choose/check the actual
            // active allocation namespace; never relabel through unchecked new.
            let storage = lloom_tensor::MetalStorage::new_allocated(
                handle,
                metal.clone(),
                data.len(),
                lloom_tensor::DType::U32,
            )?;
            return Ok(Tensor::from_storage(
                lloom_tensor::Storage::Metal(storage),
                data.len(),
                lloom_tensor::op::BackpropOp::none(),
                false,
            ));
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = kind;
    Tensor::new(data, device)
}

// The owner has one implementation; the protected model/frame migration is
// still pending. These safe types do not expose their native Tensor payloads.
#[path = "flash_metadata/resident.rs"]
mod resident;
pub use resident::{FlashDecodingCount, FlashDecodingResidentMetadata};

#[path = "flash_metadata/view.rs"]
mod view;
pub use view::FlashDecodingMetadataView;

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
/// let view = metadata.view();
/// assert_eq!(view.table(), &[0]);
/// assert_eq!(view.keys(), &[1]);
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
    /// Per-pass callers move their two host vectors into this constructor.
    /// Metal tables use the existing by-value cache once/publication (one
    /// lookup, reusing an equal entry or uploading on a miss); counts retain
    /// one ordinary upload. A cache miss keeps its existing key allocations.
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
        let table_t = upload_metadata(table.as_slice(), device, MetadataUpload::Table)?;
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

    /// Publish a new pass while retaining unchanged private table storage.
    ///
    /// Performs shared metadata validation and exact table comparison once per
    /// pass, not per attention layer. Only a changed table/device is uploaded;
    /// live counts are uploaded once per pass. Returned old vectors can be
    /// reused as the producer's staging allocations. Failure keeps this owner's
    /// prior host values, native publication and checked summary unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn republish(
        &mut self,
        table: Vec<u32>,
        table_stride: usize,
        keys: Vec<u32>,
        rows: usize,
        device: &Device,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        self.republish_with(table, table_stride, keys, rows, device, upload_metadata)
    }

    fn republish_with(
        &mut self,
        table: Vec<u32>,
        table_stride: usize,
        keys: Vec<u32>,
        rows: usize,
        device: &Device,
        mut upload: impl FnMut(&[u32], &Device, MetadataUpload) -> Result<Tensor>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let (page_size, pages_per_chunk) = self.page_geometry();
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
        let replacement = if table != self.table || !self.table_t.device().same_device(device) {
            Some(upload(&table, device, MetadataUpload::Table)?)
        } else {
            None
        };
        let keys_t = upload(&keys, device, MetadataUpload::Counts)?;
        let staging = (
            std::mem::replace(&mut self.table, table),
            std::mem::replace(&mut self.keys, keys),
        );
        self.table_stride = table_stride;
        self.summary = summary;
        if let Some(table_t) = replacement {
            self.table_t = table_t;
        }
        self.keys_t = keys_t;
        Ok(staging)
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

    /// Recover the unique host staging vectors for the next publication.
    ///
    /// Consuming this owner closes future views; an outstanding future use of a
    /// borrowed view prevents the move. Tensor::new initialized independent
    /// device storage, so later host edits cannot rewrite its old payload.
    /// Already-encoded native readers separately retain that storage through the
    /// normal Tensor/command retirement contract; host-vector ownership is not
    /// a substitute for their completion.
    pub fn into_host_vectors(self) -> (Vec<u32>, Vec<u32>) {
        let Self { table, keys, .. } = self;
        (table, keys)
    }

    /// Retain an ordinary immutable publication for a producer's own Clone
    /// implementation. This is NOT a public Clone capability and never applies
    /// to mapped resident tables or mutable frame/count writers.
    ///
    /// Host Vecs are copied and existing Tensor ownership is incremented; this
    /// is not allocation-free. No native upload, recording, readback, new
    /// validation or fallible native operation is performed. Subsequent
    /// publication replaces the private payloads instead of mutating them.
    /// The post-PR1166 PagedKv migration consumes this helper; it is otherwise
    /// intentionally not wired to a model in this owned-source candidate.
    pub(crate) fn clone_immutable_publication(&self) -> Self {
        Self {
            table: self.table.clone(),
            keys: self.keys.clone(),
            table_stride: self.table_stride,
            summary: self.summary,
            table_t: self.table_t.clone(),
            keys_t: self.keys_t.clone(),
        }
    }

    /// Borrow the existing publication; no table/count copy, upload or scan.
    pub fn view(&self) -> FlashDecodingMetadataView<'_> {
        let (table_t, keys_t) = self.bindings();
        FlashDecodingMetadataView {
            table: &self.table,
            keys: &self.keys,
            table_stride: self.table_stride,
            summary: *self.summary(),
            table_t,
            keys_t,
        }
    }
    pub fn page_geometry(&self) -> (usize, usize) {
        self.summary.route()
    }

    // This module remains private to nn::ops; do not expose native Tensor
    // aliases to any safe external caller.

    pub(super) fn summary(&self) -> &PageSummary {
        &self.summary
    }
    pub(super) fn bindings(&self) -> (&Tensor, &Tensor) {
        (&self.table_t, &self.keys_t)
    }
}

// Same identity checks as the existing completion-hook GPU fixture. This is
// test-only; it neither changes process configuration nor exists on a hot path.
#[cfg(all(test, feature = "metal"))]
fn assert_native_fixture_arm(device: &lloom_tensor::MetalDevice, control: &str) {
    let actual = device.commands().actual_command_api();
    eprintln!(
        "FLASH_METADATA_NATIVE control={control} api={actual} cb_kind={:?}",
        device.command_buffer_kind()
    );
    let selected = &lloom_tensor::tuning::tuning().metal.command_api;
    if selected != "auto" {
        assert_eq!(
            actual,
            selected.as_str(),
            "native fixture resolved a different API"
        );
    }
    if let Ok(requested) = std::env::var("LLOOM_METAL_API") {
        let requested = requested.trim().to_ascii_lowercase();
        if requested != "auto" {
            assert_eq!(
                actual, requested,
                "native fixture did not honor its explicit API arm"
            );
        }
    }
}

#[cfg(test)]
mod flash_validation_tests {
    use super::*;
    use crate::nn::attention::flash_validation::preparation_count;
    use crate::nn::ops::{flash_decoding_paged_metadata, FlashScratchSizing};
    use lloom_tensor::DType;

    // Test-only backing-object identity; scoped guards never survive a later
    // write/readback. This is not native reuse or concurrency authority.
    fn shares_storage(left: &Tensor, right: &Tensor) -> bool {
        let (left, _) = left.storage_and_layout();
        let (right, _) = right.storage_and_layout();
        std::ptr::eq(&*left, &*right)
    }

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
    fn consuming_owner_recovers_unique_host_staging_without_rewriting_payloads() -> Result<()> {
        let table = vec![0];
        let keys = vec![1];
        let (table_ptr, keys_ptr) = (table.as_ptr(), keys.as_ptr());
        let owner = FlashDecodingMetadata::new(table, 1, keys, 1, 256, 1, &Device::Cpu)?;
        // Privileged test aliases observe initialization independently; the
        // public owner/view API never exports these Tensor references.
        let old_table = owner.table_t.clone();
        let old_keys = owner.keys_t.clone();
        let (mut table, mut keys) = owner.into_host_vectors();
        assert_eq!(table.as_ptr(), table_ptr);
        assert_eq!(keys.as_ptr(), keys_ptr);
        table[0] = 3;
        keys[0] = 2;
        assert_eq!(old_table.to_vec1::<u32>()?, &[0]);
        assert_eq!(old_keys.to_vec1::<u32>()?, &[1]);
        let next = FlashDecodingMetadata::new(table, 1, keys, 1, 256, 1, &Device::Cpu)?;
        assert_eq!(next.table(), &[3]);
        assert_eq!(next.keys(), &[2]);
        Ok(())
    }

    #[test]
    fn resident_view_uses_the_shared_safe_frontdoor_without_rescanning() -> Result<()> {
        use crate::nn::ops::flash_decoding_paged_view;
        let device = Device::Cpu;
        let mut table = FlashDecodingResidentMetadata::new(256, 1, 4)?;
        table.publish_suffix(&[3], &device)?;
        let count = FlashDecodingCount::new(1, &device)?;
        let view = table.view(&count)?;
        let prepared = preparation_count();
        let q = Tensor::zeros((1, 1, 1, 64), DType::F32, &device)?;
        let kv = Tensor::zeros((1, 1, 769, 64), DType::F32, &device)?;
        let short_v = Tensor::zeros((1, 1, 768, 64), DType::F32, &device)?;
        for _ in 0..2 {
            let error =
                flash_decoding_paged_view(&q, &kv, &kv, 1., 1., FlashScratchSizing::Grow, 0, &view)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("no cpu support"), "{error}");
        }
        let error = flash_decoding_paged_view(
            &q,
            &kv,
            &short_v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &view,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("page span exceeds K/V pool"), "{error}");
        assert_eq!(preparation_count(), prepared);
        Ok(())
    }

    #[test]
    fn owner_diagnostics_do_not_delegate_to_device_tensor_debug() -> Result<()> {
        let owner = FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
        let count = FlashDecodingCount::new(1, &Device::Cpu)?;
        for printed in [
            format!("{owner:?}"),
            format!("{:?}", owner.view()),
            format!("{count:?}"),
        ] {
            assert!(!printed.contains("Tensor["), "{printed}");
        }
        assert!(format!("{owner:?}").contains("table_entries"));
        Ok(())
    }

    #[test]
    fn immutable_clone_keeps_payload_custody_but_not_host_vector_aliases() -> Result<()> {
        let owner = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &Device::Cpu)?;
        let preparations = preparation_count();
        let copy = owner.clone_immutable_publication();
        assert_eq!(
            preparation_count(),
            preparations,
            "clone reuses existing checked proof"
        );
        assert_ne!(owner.table.as_ptr(), copy.table.as_ptr());
        assert_ne!(owner.keys.as_ptr(), copy.keys.as_ptr());
        assert!(shares_storage(&owner.table_t, &copy.table_t));
        assert!(shares_storage(&owner.keys_t, &copy.keys_t));
        assert_eq!(
            format!("{:?}", owner.summary),
            format!("{:?}", copy.summary)
        );
        let saved_table = copy.table_t.clone();
        let saved_keys = copy.keys_t.clone();
        let (mut table, mut keys) = copy.into_host_vectors();
        table[0] = 7;
        keys[0] = 9;
        assert_eq!(owner.table(), &[0, 1]);
        assert_eq!(owner.keys(), &[1]);
        assert_eq!(saved_table.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(saved_keys.to_vec1::<u32>()?, vec![1]);
        Ok(())
    }

    #[test]
    fn immutable_clone_preserves_padded_stride_and_nondefault_page_geometry() -> Result<()> {
        // Two rows, two pages/chunk, and three table entries/row although only
        // two chunks dispatch. Row1's second chunk is padded/empty.
        let owner = FlashDecodingMetadata::new(
            vec![2, 4, 9, 6, 8, 10],
            3,
            vec![129, 65],
            2,
            64,
            2,
            &Device::Cpu,
        )?;
        let copy = owner.clone_immutable_publication();
        assert_eq!(copy.table_stride(), 3);
        assert_eq!(copy.page_geometry(), (64, 2));
        assert_eq!(copy.table(), &[2, 4, 9, 6, 8, 10]);
        assert_eq!(copy.keys(), &[129, 65]);
        assert_eq!(
            format!("{:?}", copy.summary),
            format!("{:?}", owner.summary)
        );
        let view = copy.view();
        assert_eq!(view.table_stride(), 3);
        assert_eq!(view.page_geometry(), (64, 2));
        assert_eq!(view.table(), owner.table());
        assert_eq!(view.keys(), owner.keys());
        assert!(shares_storage(&copy.table_t, &owner.table_t));
        assert!(shares_storage(&copy.keys_t, &owner.keys_t));
        Ok(())
    }

    #[test]
    fn immutable_clone_republication_and_failure_leave_sibling_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let owner = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        let mut changed = owner.clone_immutable_publication();
        changed.republish(vec![1, 0], 1, vec![2], 2, &device)?;
        assert_eq!(changed.table_t.to_vec1::<u32>()?, vec![1, 0]);
        assert_eq!(changed.keys_t.to_vec1::<u32>()?, vec![2]);
        assert_eq!(owner.table_t.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(owner.keys_t.to_vec1::<u32>()?, vec![1]);
        assert_eq!(owner.table(), &[0, 1]);
        assert_eq!(owner.keys(), &[1]);
        assert!(!shares_storage(&owner.table_t, &changed.table_t));
        assert!(!shares_storage(&owner.keys_t, &changed.keys_t));

        let mut failed = owner.clone_immutable_publication();
        let before_summary = format!("{:?}", failed.summary);
        let error = failed
            .republish_with(vec![2, 3], 1, vec![2], 2, &device, |data, d, kind| {
                if kind == MetadataUpload::Counts {
                    lloom_tensor::bail!("injected clone count refusal")
                }
                upload_metadata(data, d, kind)
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected clone count refusal"), "{error}");
        assert_eq!(failed.table(), &[0, 1]);
        assert_eq!(failed.keys(), &[1]);
        assert_eq!(format!("{:?}", failed.summary), before_summary);
        assert!(shares_storage(&owner.table_t, &failed.table_t));
        assert!(shares_storage(&owner.keys_t, &failed.keys_t));
        assert_eq!(owner.table_t.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(owner.keys_t.to_vec1::<u32>()?, vec![1]);

        let mut stable = owner.clone_immutable_publication();
        let mut stages = Vec::new();
        stable.republish_with(vec![0, 1], 1, vec![3], 2, &device, |data, d, kind| {
            stages.push(kind);
            upload_metadata(data, d, kind)
        })?;
        assert_eq!(stages, [MetadataUpload::Counts]);
        assert!(shares_storage(&owner.table_t, &stable.table_t));
        assert!(!shares_storage(&owner.keys_t, &stable.keys_t));
        assert_eq!(stable.keys_t.to_vec1::<u32>()?, vec![3]);
        assert_eq!(owner.keys_t.to_vec1::<u32>()?, vec![1]);
        Ok(())
    }

    #[test]
    fn stable_table_republication_uploads_only_counts_and_recovers_staging() -> Result<()> {
        let device = Device::Cpu;
        let table = vec![0, 1];
        let keys = vec![1];
        let old_ptrs = (table.as_ptr(), keys.as_ptr());
        let mut owner = FlashDecodingMetadata::new(table, 1, keys, 2, 256, 1, &device)?;
        let old_table = owner.table_t.clone();
        let old_keys = owner.keys_t.clone();
        let next_table = vec![0, 1];
        let next_keys = vec![2];
        let next_ptrs = (next_table.as_ptr(), next_keys.as_ptr());
        let mut uploads = 0;
        let staging =
            owner.republish_with(next_table, 1, next_keys, 2, &device, |data, d, _kind| {
                uploads += 1;
                Tensor::new(data, d)
            })?;
        assert_eq!(
            uploads, 1,
            "an unchanged private table must not be uploaded"
        );
        assert!(shares_storage(&owner.table_t, &old_table));
        assert!(!shares_storage(&owner.keys_t, &old_keys));
        assert_eq!((staging.0.as_ptr(), staging.1.as_ptr()), old_ptrs);
        assert_eq!((owner.table.as_ptr(), owner.keys.as_ptr()), next_ptrs);
        assert_eq!(old_keys.to_vec1::<u32>()?, vec![1]);
        assert_eq!(owner.keys_t.to_vec1::<u32>()?, vec![2]);
        Ok(())
    }

    #[test]
    fn changed_table_republication_retains_old_native_values() -> Result<()> {
        let device = Device::Cpu;
        let mut owner =
            FlashDecodingMetadata::new(vec![0, 1, 2, 3], 2, vec![257], 2, 256, 1, &device)?;
        let old_table = owner.table_t.clone();
        let old_keys = owner.keys_t.clone();
        let mut uploads = 0;
        owner.republish_with(
            vec![3, 1, 0, 2],
            2,
            vec![258],
            2,
            &device,
            |data, d, _kind| {
                uploads += 1;
                Tensor::new(data, d)
            },
        )?;
        assert_eq!(uploads, 2);
        assert!(!shares_storage(&owner.table_t, &old_table));
        assert_eq!(old_table.to_vec1::<u32>()?, vec![0, 1, 2, 3]);
        assert_eq!(old_keys.to_vec1::<u32>()?, vec![257]);
        assert_eq!(owner.table_t.to_vec1::<u32>()?, vec![3, 1, 0, 2]);
        assert_eq!(owner.keys_t.to_vec1::<u32>()?, vec![258]);
        Ok(())
    }

    #[test]
    fn unchanged_table_bytes_with_new_stride_reuse_payload_but_revalidate_span() -> Result<()> {
        let device = Device::Cpu;
        let mut owner =
            FlashDecodingMetadata::new(vec![0, 1, 2, 3], 1, vec![1], 2, 256, 1, &device)?;
        let old_table = owner.table_t.clone();
        let mut uploads = 0;
        owner.republish_with(
            vec![0, 1, 2, 3],
            2,
            vec![2],
            2,
            &device,
            |data, d, _kind| {
                uploads += 1;
                Tensor::new(data, d)
            },
        )?;
        assert_eq!(uploads, 1, "a stride change is not a native byte change");
        assert!(shares_storage(&owner.table_t, &old_table));
        assert_eq!(owner.table_stride(), 2);
        let q = Tensor::zeros((2, 1, 1, 64), DType::F32, &device)?;
        let short = Tensor::zeros((1, 1, 513, 64), DType::F32, &device)?;
        let error = flash_decoding_paged_metadata(
            &q,
            &short,
            &short,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &owner,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("page span exceeds K/V pool"), "{error}");
        let valid = Tensor::zeros((1, 1, 514, 64), DType::F32, &device)?;
        let error = flash_decoding_paged_metadata(
            &q,
            &valid,
            &valid,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &owner,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no cpu support"), "{error}");
        Ok(())
    }

    #[test]
    fn failed_count_upload_keeps_previous_publication_transactional() -> Result<()> {
        let device = Device::Cpu;
        let mut owner = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        let old_table = owner.table_t.clone();
        let old_keys = owner.keys_t.clone();
        let old_ptrs = (owner.table.as_ptr(), owner.keys.as_ptr());
        let mut uploads = 0;
        let error = owner
            .republish_with(vec![2, 3], 1, vec![2], 2, &device, |data, d, _kind| {
                uploads += 1;
                if uploads == 2 {
                    lloom_tensor::bail!("injected count upload failure")
                }
                Tensor::new(data, d)
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected count upload failure"), "{error}");
        assert_eq!(uploads, 2);
        assert_eq!(owner.table(), &[0, 1]);
        assert_eq!(owner.keys(), &[1]);
        assert_eq!((owner.table.as_ptr(), owner.keys.as_ptr()), old_ptrs);
        assert!(shares_storage(&owner.table_t, &old_table));
        assert!(shares_storage(&owner.keys_t, &old_keys));
        let q = Tensor::zeros((2, 1, 1, 64), DType::F32, &device)?;
        let pool = Tensor::zeros((1, 1, 257, 64), DType::F32, &device)?;
        let error = flash_decoding_paged_metadata(
            &q,
            &pool,
            &pool,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &owner,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("no cpu support"),
            "stale new summary: {error}"
        );
        Ok(())
    }

    #[test]
    fn malformed_republication_refuses_before_any_upload_or_host_change() -> Result<()> {
        let device = Device::Cpu;
        let mut owner = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        let mut uploads = 0;
        let error = owner
            .republish_with(vec![0], 1, vec![1], 2, &device, |data, d, _kind| {
                uploads += 1;
                Tensor::new(data, d)
            })
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("chunk table is shorter than live rows"),
            "{error}"
        );
        assert_eq!(uploads, 0);
        assert_eq!(owner.table(), &[0, 1]);
        assert_eq!(owner.keys(), &[1]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn same_native_table_handle(a: &Tensor, b: &Tensor) -> bool {
        use lloom_tensor::{kernels::metal::BufferHandle, Storage};
        let (a, _) = a.storage_and_layout();
        let (b, _) = b.storage_and_layout();
        let (Storage::Metal(a), Storage::Metal(b)) = (&*a, &*b) else {
            panic!("native table identity control requires Metal");
        };
        BufferHandle::ptr_eq(a.buffer_handle(), b.buffer_handle())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn hybrid_table_shares_equal_bytes_without_relooking_up_unchanged_publication() -> Result<()> {
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        assert_native_fixture_arm(metal, "hybrid-table-sharing");
        let before = metal.constants_counters();
        let mut first = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        let second = FlashDecodingMetadata::new(vec![0, 1], 1, vec![2], 2, 256, 1, &device)?;
        assert!(same_native_table_handle(&first.table_t, &second.table_t));
        let after = metal.constants_counters();
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 1));
        let old = first.table_t.clone();
        first.republish(vec![0, 1], 1, vec![3], 2, &device)?;
        assert_eq!(
            metal.constants_counters(),
            after,
            "unchanged table does not hash/lock cache"
        );
        assert!(same_native_table_handle(&old, &first.table_t));
        first.republish(vec![1, 0], 1, vec![4], 2, &device)?;
        assert!(!same_native_table_handle(&first.table_t, &second.table_t));
        assert_eq!(old.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(second.table_t.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(first.table_t.to_vec1::<u32>()?, vec![1, 0]);
        assert_eq!(first.keys_t.to_vec1::<u32>()?, vec![4]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn hybrid_old_publication_keeps_creator_after_cache_eviction() -> Result<()> {
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        assert_native_fixture_arm(metal, "hybrid-table-eviction");
        let first = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        // Existing cache is bounded to512 entries. Each value here is unique
        // and four bytes, distinct from the table's eight-byte cache key.
        for value in 0..513u32 {
            drop(metal.constant_of(&value)?);
        }
        let before = metal.constants_counters();
        let again = FlashDecodingMetadata::new(vec![0, 1], 1, vec![2], 2, 256, 1, &device)?;
        let after = metal.constants_counters();
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (0, 1),
            "old cache entry was actually evicted"
        );
        assert!(!same_native_table_handle(&first.table_t, &again.table_t));
        assert_eq!(first.table_t.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(again.table_t.to_vec1::<u32>()?, vec![0, 1]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn hybrid_count_failure_does_not_commit_cached_replacement() -> Result<()> {
        let device = Device::new_metal(0)?;
        assert_native_fixture_arm(device.as_metal_device()?, "hybrid-count-rollback");
        let mut owner = FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &device)?;
        let old = owner.table_t.clone();
        let old_keys = owner.keys_t.clone();
        let old_summary = format!("{:?}", owner.summary);
        let mut stages = Vec::new();
        let error = owner
            .republish_with(vec![2, 3], 1, vec![2], 2, &device, |data, d, kind| {
                stages.push(kind);
                if kind == MetadataUpload::Counts {
                    lloom_tensor::bail!("injected native hybrid count refusal")
                }
                upload_metadata(data, d, kind)
            })
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("injected native hybrid count refusal"),
            "{error}"
        );
        assert_eq!(stages, [MetadataUpload::Table, MetadataUpload::Counts]);
        assert_eq!(owner.table(), &[0, 1]);
        assert_eq!(owner.keys(), &[1]);
        assert_eq!(format!("{:?}", owner.summary), old_summary);
        assert!(same_native_table_handle(&owner.table_t, &old));
        assert!(same_native_table_handle(&owner.keys_t, &old_keys));
        assert_eq!(old.to_vec1::<u32>()?, vec![0, 1]);
        assert_eq!(old_keys.to_vec1::<u32>()?, vec![1]);
        Ok(())
    }

    /// Numeric/accounting fixture only, not machine admission/RSS proof. Two
    /// public namespace APIs share one actual device while retaining distinct
    /// caches and allocation-origin authority. No new fixture allocator.
    #[cfg(feature = "metal")]
    #[test]
    fn hybrid_table_wrapper_preserves_active_allocation_namespace() -> Result<()> {
        use lloom_tensor::memory_admission::process_account;
        use lloom_tensor::metal_backend::MetalAllocationDomain;
        let base = Device::new_metal(0)?;
        let metal = base.as_metal_device()?;
        assert_native_fixture_arm(metal, "hybrid-table-namespace");
        let a = MetalAllocationDomain::new(
            metal,
            process_account().unadmitted_domain(Some(128 << 20))?,
        )?;
        let b = MetalAllocationDomain::new(
            metal,
            process_account().unadmitted_domain(Some(128 << 20))?,
        )?;
        assert_ne!(a.domain().key(), b.domain().key());
        let pass_a = a.domain().reserve(32 << 20)?;
        let pass_b = b.domain().reserve(32 << 20)?;
        let scope_a = a.scope(pass_a.context())?;
        let scope_b = b.scope(pass_b.context())?;
        let before = metal.constants_counters();
        // Deliberately supply the ORIGINAL device; new_allocated must resolve
        // the checked active view instead of assigning this original tag.
        let mut first = scope_a
            .run(|_| FlashDecodingMetadata::new(vec![0, 1], 1, vec![1], 2, 256, 1, &base))?;
        let second = scope_b
            .run(|_| FlashDecodingMetadata::new(vec![0, 1], 1, vec![2], 2, 256, 1, &base))?;
        assert!(base.same_device(first.table_t.device()));
        assert!(base.same_device(second.table_t.device()));
        assert!(!same_native_table_handle(&first.table_t, &second.table_t));
        assert_eq!(
            metal.constants_counters(),
            before,
            "base cache must not receive scoped uploads"
        );
        // An accounted storage's saved device refuses allocation queries with
        // no active scope. Unchecked new(handle,original,...) would return Ok.
        for table in [&first.table_t, &second.table_t] {
            let error = table
                .device()
                .as_metal_device()?
                .allocation_origin()
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("requires an active execution scope"),
                "{error}"
            );
        }
        let old_table = first.table_t.clone();
        scope_b.run(|m| {
            let cache_before = m.constants_counters();
            first.republish(vec![0, 1], 1, vec![3], 2, &base)?;
            assert_eq!(
                m.constants_counters(),
                cache_before,
                "existing input origin is not republished/reclassified"
            );
            assert!(same_native_table_handle(&first.table_t, &old_table));
            Ok(())
        })?;
        // Read only inside a checked scope; other-domain immutable input is
        // legal and keeps its own original handle throughout.
        scope_b.run(|_| {
            assert_eq!(first.table_t.to_vec1::<u32>()?, vec![0, 1]);
            assert_eq!(first.keys_t.to_vec1::<u32>()?, vec![3]);
            Ok(())
        })?;
        drop((first, second, old_table));
        drop((scope_a, scope_b, pass_a, pass_b));
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn native_views_add_buffer_base_and_layout_offsets_independently() -> Result<()> {
        use lloom_tensor::kernels::metal::{Buffer, BufferHandle, PooledBuffer};
        use lloom_tensor::{op::BackpropOp, MetalStorage, Storage};
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        assert_native_fixture_arm(metal, "view-base-plus-layout-offset");
        let mut q_data = vec![0f32; 112];
        q_data[48] = 1.;
        let mut k_data = vec![0f32; 200];
        k_data[72] = 1.;
        k_data[136] = -1.;
        let mut v_data = vec![-7f32; 192];
        v_data[64..128].fill(1.);
        v_data[128..192].fill(3.);
        // Keep the genuine creator tensors through final readback. A native
        // Buffer view alone is not the legacy pool's creator reuse census.
        let q_base = Tensor::new(q_data.as_slice(), &device)?;
        let k_base = Tensor::new(k_data.as_slice(), &device)?;
        let v_base = Tensor::new(v_data.as_slice(), &device)?;
        let view_tensor = |base: &Tensor,
                           at: usize,
                           bytes: usize,
                           skip: usize,
                           shape: [usize; 4]|
         -> Result<Tensor> {
            let (storage, _) = base.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                unreachable!("Metal fixture input")
            };
            let native: Buffer = storage.buffer().view(at, bytes);
            let handle = BufferHandle::new(PooledBuffer::unpooled(native, bytes));
            let tensor = Tensor::from_storage(
                Storage::Metal(MetalStorage::new(
                    handle,
                    metal.clone(),
                    bytes / 4,
                    DType::F32,
                )),
                bytes / 4,
                BackpropOp::none(),
                false,
            );
            tensor
                .narrow(0, skip, shape.iter().product())?
                .reshape(shape)
        };
        let q = view_tensor(&q_base, 128, 320, 16, [1, 1, 1, 64])?;
        let k = view_tensor(&k_base, 192, 608, 24, [1, 1, 2, 64])?;
        let v = view_tensor(&v_base, 128, 640, 32, [1, 1, 2, 64])?;
        assert_eq!(q.layout().start_offset(), 16);
        assert_eq!(k.layout().start_offset(), 24);
        assert_eq!(v.layout().start_offset(), 32);
        let metadata = FlashDecodingMetadata::new(vec![0], 1, vec![2], 1, 256, 1, &device)?;
        let output = flash_decoding_paged_metadata(
            &q,
            &k,
            &v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &metadata,
        )?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = (1f32.exp() + 3. * (-1f32).exp()) / (1f32.exp() + (-1f32).exp());
        assert_eq!(actual.len(), 64);
        for (at, &value) in actual.iter().enumerate() {
            assert!(
                (value - expected).abs() <= 2e-6,
                "offset control channel {at}: {value} vs {expected}"
            );
        }
        drop((q, k, v));
        drop((q_base, k_base, v_base));
        Ok(())
    }

    /// This native fixture separately proves that both instruments can see a
    /// valid dispatch. Refusal alone is not evidence of no allocation/encoding
    /// if a disabled instrument always reports zero.
    #[cfg(feature = "metal")]
    #[test]
    fn owner_refusals_precede_native_pool_lookup_and_dispatch() -> Result<()> {
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        assert_native_fixture_arm(metal, "pre-builder-refusals");
        let q = Tensor::zeros((1, 1, 1, 64), DType::F32, &device)?;
        let k = Tensor::zeros((1, 1, 769, 64), DType::F32, &device)?;
        let short_v = Tensor::zeros((1, 1, 768, 64), DType::F32, &device)?;
        let valid_v = Tensor::ones((1, 1, 769, 64), DType::F32, &device)?;
        let metadata = FlashDecodingMetadata::new(vec![3], 1, vec![1], 1, 256, 1, &device)?;
        let cpu_metadata =
            FlashDecodingMetadata::new(vec![0], 1, vec![1], 1, 256, 1, &Device::Cpu)?;
        let lookups = || {
            let (shared, private) = metal.pool_counters();
            (shared.lookups, private.lookups)
        };
        let before = lookups();
        let census = metal.dispatch_census().open().map_err(|e| {
            lloom_tensor::Error::Msg(format!("native refusal control census unavailable: {e:?}"))
        })?;
        let bad_publication =
            FlashDecodingMetadata::new(vec![u32::MAX], 1, vec![1], 1, 256, 1, &device)
                .unwrap_err()
                .to_string();
        assert!(
            bad_publication.contains("signed native token range"),
            "{bad_publication}"
        );
        let bad_pool = flash_decoding_paged_metadata(
            &q,
            &k,
            &short_v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &metadata,
        )
        .unwrap_err()
        .to_string();
        assert!(
            bad_pool.contains("page span exceeds K/V pool"),
            "{bad_pool}"
        );
        let bad_device = flash_decoding_paged_metadata(
            &q,
            &k,
            &valid_v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &cpu_metadata,
        )
        .unwrap_err()
        .to_string();
        assert!(
            bad_device.contains("metadata must share the query device"),
            "{bad_device}"
        );
        assert_eq!(
            lookups(),
            before,
            "a refused native call visited a buffer pool"
        );
        assert!(
            census.close().is_empty(),
            "a refused native call encoded a dispatch"
        );

        let before_positive = lookups();
        let positive = metal.dispatch_census().open().map_err(|e| {
            lloom_tensor::Error::Msg(format!("native positive census unavailable: {e:?}"))
        })?;
        let output = flash_decoding_paged_metadata(
            &q,
            &k,
            &valid_v,
            1.,
            1.,
            FlashScratchSizing::Grow,
            0,
            &metadata,
        )?;
        let dispatches = positive.close();
        let after_positive = lookups();
        assert!(
            after_positive.0 > before_positive.0 || after_positive.1 > before_positive.1,
            "the pool-lookup instrument did not observe an actual valid output/scratch allocation",
        );
        assert!(
            dispatches.iter().any(|(_, count)| *count > 0),
            "the dispatch instrument missed the valid attention kernel"
        );
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![1.; 64]);
        Ok(())
    }
}
