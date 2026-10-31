//! Resident metadata custody, not yet wired into the protected PagedKv producer.
//!
//! The authoritative prefix and the native table have one owner. Publication
//! appends only beyond every prior published count. A borrowed read excludes
//! future mutation while it may still be encoded; it does NOT by itself retain
//! resources for queued GPU work. The ordinary Tensor upload/pool/in-flight
//! lifetime remains responsible for those already-encoded readers.
use crate::nn::attention::flash_validation::{PageSummary, ResidentSpans};
use lloom_tensor::{Device, Result, Tensor};

pub(in crate::nn::ops) struct ResidentMetadata {
    spans: ResidentSpans,
    reserve_entries: usize,
    // No safe Tensor accessor, clone, or mutable alias may escape this module.
    table_t: Option<Tensor>,
}

impl Clone for ResidentMetadata {
    fn clone(&self) -> Self {
        // Host ownership clones normally; native publication must not alias an
        // independently writable clone. Re-publish before the clone's first use.
        Self {
            spans: self.spans.clone(),
            reserve_entries: self.reserve_entries,
            table_t: None,
        }
    }
}

impl std::fmt::Debug for ResidentMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentMetadata")
            .field("published_entries", &self.spans.entries().len())
            .field("reserve_entries", &self.reserve_entries)
            .field("has_native_publication", &self.table_t.is_some())
            .finish_non_exhaustive()
    }
}

impl ResidentMetadata {
    pub(in crate::nn::ops) fn new(
        page_size: usize,
        pages_per_chunk: usize,
        reserve_entries: usize,
    ) -> Result<Self> {
        Ok(Self {
            spans: ResidentSpans::new(page_size, pages_per_chunk)?,
            reserve_entries,
            table_t: None,
        })
    }

    pub(in crate::nn::ops) fn entries(&self) -> &[u32] {
        self.spans.entries()
    }

    /// Detach rather than rewrite a prefix still reachable by queued work.
    /// Existing Tensor/command retirement retains the old physical allocation;
    /// this owner does not introduce a second pool, ring or retirement protocol.
    pub(in crate::nn::ops) fn reset(&mut self) {
        self.table_t = None;
        self.spans.clear();
    }

    /// Append a previously unpublished suffix. An empty suffix re-publishes a
    /// clone's detached host prefix; it does nothing for an already-ready owner.
    /// The caller moves the existing PagedKv shadow state here, not a second copy.
    pub(in crate::nn::ops) fn publish_suffix(
        &mut self,
        suffix: &[u32],
        device: &Device,
    ) -> Result<()> {
        self.publish_with(suffix, device, |data, device| Tensor::new(data, device))
    }

    fn publish_with(
        &mut self,
        suffix: &[u32],
        device: &Device,
        upload: impl FnOnce(&[u32], &Device) -> Result<Tensor>,
    ) -> Result<()> {
        if let Some(table) = &self.table_t {
            if !table.device().same_device(device) {
                lloom_tensor::bail!("flash-decoding: resident publication device changed")
            }
            if suffix.is_empty() {
                return Ok(());
            }
        }
        // Reserve both existing host publication vectors BEFORE any native
        // side-effect. A failed reserve/upload leaves their semantic lengths and
        // the old native prefix unchanged (capacity itself may grow).
        let append = self.spans.prepare_append(suffix)?;
        let wrote_suffix = self.table_t.as_ref().is_some_and(|table| {
            // SAFETY: this module never exports native aliases. append.start()
            // is the first index no previously returned read could address:
            // summary checked its chunk count against the then-published prefix.
            // Published entries/counts remain immutable; the mutable borrow also
            // excludes any still-usable future safe read. Existing queued users
            // may read only the disjoint old prefix. write_host_slice returns
            // false before memcpy, not after a partially performed write.
            unsafe { table.write_host_slice::<u32>(append.start(), suffix) }
        });
        if !wrote_suffix {
            // The same replacement-upload fallback used by PagedKv today.
            // Not a new upload alongside the successful mapped append.
            let staged = append.stage(self.reserve_entries)?;
            let replacement = upload(&staged, device)?;
            self.table_t = Some(replacement);
        }
        append.commit();
        Ok(())
    }

    /// A lifetime-bound checked view; no Tensor or writer alias is exposed.
    /// Preparation is O(1) prefix/tail arithmetic, never a table scan/upload.
    pub(in crate::nn::ops) fn read<'a>(
        &'a self,
        count: &'a ResidentCount,
    ) -> Result<ResidentRead<'a>> {
        let table = self.table_t.as_ref().ok_or_else(|| {
            lloom_tensor::Error::Msg("flash-decoding: resident table is not published".into())
        })?;
        if !table.device().same_device(count.tensor.device()) {
            lloom_tensor::bail!("flash-decoding: resident table/count devices differ")
        }
        let summary = self.spans.summary(count.keys)?;
        Ok(ResidentRead {
            owner: self,
            count,
            summary,
        })
    }
}

/// One immutable step count: inline host value and the existing one-value upload.
/// A frame-backed variant is deliberately absent pending #1147's scoped lease.
pub(in crate::nn::ops) struct ResidentCount {
    keys: u32,
    tensor: Tensor,
}

impl ResidentCount {
    pub(in crate::nn::ops) fn new(keys: u32, device: &Device) -> Result<Self> {
        if keys == 0 || keys > i32::MAX as u32 {
            lloom_tensor::bail!("flash-decoding: resident live count must be positive signed U32")
        }
        Ok(Self {
            keys,
            tensor: Tensor::new(&[keys], device)?,
        })
    }
}

/// The existing op may bind these private tensors only while this view lives.
/// No Arc, atomic, new allocator or ring is added to layer consumption.
pub(in crate::nn::ops) struct ResidentRead<'a> {
    owner: &'a ResidentMetadata,
    count: &'a ResidentCount,
    summary: PageSummary,
}
impl ResidentRead<'_> {
    pub(in crate::nn::ops) fn summary(&self) -> &PageSummary {
        &self.summary
    }
    pub(in crate::nn::ops) fn entries(&self) -> &[u32] {
        self.owner.entries()
    }
    pub(in crate::nn::ops) fn keys(&self) -> &[u32] {
        std::slice::from_ref(&self.count.keys)
    }
    // Kept private to the encapsulation module until the actual op binding seam
    // is integrated. Widening this outside nn::ops would invalidate custody.
    fn bindings(&self) -> (&Tensor, &Tensor) {
        (
            self.owner.table_t.as_ref().expect("checked by read"),
            &self.count.tensor,
        )
    }
}

#[cfg(test)]
mod flash_validation_tests {
    use super::*;
    use crate::nn::attention::flash_validation::validate;
    use lloom_tensor::{DType, Layout};

    fn refused<T>(result: Result<T>, message: &str) {
        let Err(error) = result else {
            panic!("missing refusal: {message}")
        };
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn resident_publication_and_counts_authenticate_one_matching_pair() -> Result<()> {
        let mut owner = ResidentMetadata::new(256, 1, 8)?;
        owner.publish_suffix(&[2, 0], &Device::Cpu)?;
        let count = ResidentCount::new(257, &Device::Cpu)?;
        let read = owner.read(&count)?;
        assert_eq!(read.entries(), &[2, 0]);
        assert_eq!(read.keys(), &[257]);
        assert_eq!(
            read.bindings().0.to_vec1::<u32>()?,
            &[2, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(read.bindings().1.to_vec1::<u32>()?, read.keys());
        Ok(())
    }

    #[test]
    fn resident_failed_append_retains_old_prefix_and_refuses_new_live_count() -> Result<()> {
        let mut owner = ResidentMetadata::new(256, 1, 2)?;
        owner.publish_suffix(&[1], &Device::Cpu)?;
        let error = owner.publish_with(&[3], &Device::Cpu, |_, _| {
            lloom_tensor::bail!("injected resident upload refusal")
        });
        refused(error, "injected resident upload refusal");
        assert_eq!(owner.entries(), &[1]);
        let old_count = ResidentCount::new(1, &Device::Cpu)?;
        assert_eq!(
            owner.read(&old_count)?.bindings().0.to_vec1::<u32>()?,
            &[1, 0]
        );
        let new_count = ResidentCount::new(257, &Device::Cpu)?;
        refused(owner.read(&new_count), "shorter than the live prefix");
        owner.publish_suffix(&[3], &Device::Cpu)?;
        assert_eq!(owner.read(&new_count)?.entries(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn resident_clone_detaches_publication_and_reset_requires_republication() -> Result<()> {
        let mut source = ResidentMetadata::new(256, 1, 2)?;
        source.publish_suffix(&[1], &Device::Cpu)?;
        let count = ResidentCount::new(1, &Device::Cpu)?;
        let mut clone = source.clone();
        refused(clone.read(&count), "not published");
        clone.publish_suffix(&[], &Device::Cpu)?;
        clone.reset();
        refused(clone.read(&count), "not published");
        clone.publish_suffix(&[3], &Device::Cpu)?;
        assert_eq!(source.read(&count)?.entries(), &[1]);
        assert_eq!(clone.read(&count)?.entries(), &[3]);
        assert_eq!(source.read(&count)?.bindings().0.to_vec1::<u32>()?, &[1, 0]);
        Ok(())
    }

    #[test]
    fn resident_step_summary_checks_each_layers_independent_pool_capacity() -> Result<()> {
        let mut owner = ResidentMetadata::new(256, 1, 2)?;
        owner.publish_suffix(&[0, 3], &Device::Cpu)?;
        let count = ResidentCount::new(257, &Device::Cpu)?;
        let read = owner.read(&count)?;
        let q = Layout::contiguous(&[1, 1, 1, 64]);
        let good = Layout::contiguous(&[1, 1, 769, 64]);
        let short = Layout::contiguous(&[1, 1, 768, 64]);
        for _ in 0..2 {
            validate(
                &q,
                &good,
                &good,
                DType::F32,
                DType::F32,
                DType::F32,
                256,
                1,
                Some(read.summary()),
            )?;
        }
        refused(
            validate(
                &q,
                &good,
                &short,
                DType::F32,
                DType::F32,
                DType::F32,
                256,
                1,
                Some(read.summary()),
            ),
            "page span exceeds K/V pool",
        );
        Ok(())
    }
}
