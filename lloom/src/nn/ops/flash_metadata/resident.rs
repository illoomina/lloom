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

    #[cfg(feature = "metal")]
    #[test]
    fn resident_mapped_suffix_preserves_an_encoded_old_prefix_reader() -> Result<()> {
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        eprintln!(
            "FLASH_METADATA_NATIVE control=mapped-suffix-encoded-uncommitted api={} cb_kind={:?}",
            metal.commands().actual_command_api(),
            metal.command_buffer_kind()
        );
        // Keep this fixture's two casts encoded/pending until after the host append.
        // No waited event, giant tensor or intentionally invalid GPU access.
        metal.set_compute_per_buffer(usize::MAX);
        let mut owner = ResidentMetadata::new(256, 1, 4)?;
        owner.publish_suffix(&[2], &device)?;
        let old_count = ResidentCount::new(1, &device)?;
        let clock = metal.commands().clock();
        let epoch = clock.current_epoch();
        let completed = clock.completed_count();
        let (old_prefix, old_live) = {
            let read = owner.read(&old_count)?;
            let (table, count) = read.bindings();
            // to_dtype(U32->F32) reaches actual cast_u32_f32 dispatches; not a
            // same-dtype Tensor clone or CPU readback masquerading as a GPU reader.
            (
                table.narrow(0, 0, 1)?.to_dtype(DType::F32)?,
                count.to_dtype(DType::F32)?,
            )
        };
        assert_eq!(
            clock.current_epoch(),
            epoch,
            "fixture unexpectedly cut its command buffer"
        );
        assert_eq!(
            clock.completed_count(),
            completed,
            "fixture unexpectedly completed before append"
        );
        // Failure-inject ONLY the replacement-upload branch: success now proves
        // that the real unsafe mapped suffix-write path was exercised.
        owner.publish_with(&[3], &device, |_, _| {
            lloom_tensor::bail!("native control unexpectedly used replacement upload")
        })?;
        let next_count = ResidentCount::new(257, &device)?;
        let next = owner.read(&next_count)?;
        let (table, count) = next.bindings();
        let new_prefix = table.narrow(0, 0, 2)?.to_dtype(DType::F32)?;
        let new_live = count.to_dtype(DType::F32)?;
        // The readbacks flush after the append; old encoded work must still observe
        // its old immutable prefix/count, while new work sees the appended table.
        assert_eq!(old_prefix.to_vec1::<f32>()?, &[2.]);
        assert_eq!(old_live.to_vec1::<f32>()?, &[1.]);
        assert_eq!(new_prefix.to_vec1::<f32>()?, &[2., 3.]);
        assert_eq!(new_live.to_vec1::<f32>()?, &[257.]);
        Ok(())
    }

    // Same release-before-drain fixture discipline as the actual tensor
    // completion_hook_gpu_tests. This is test cleanup, not a new executor.
    #[cfg(feature = "metal")]
    struct ReleaseBeforeDrain<R: FnMut(), D: FnMut()> {
        release: R,
        drain: D,
    }
    #[cfg(feature = "metal")]
    impl<R: FnMut(), D: FnMut()> Drop for ReleaseBeforeDrain<R, D> {
        fn drop(&mut self) {
            (self.release)();
            (self.drain)();
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn resident_native_fixture_releases_before_drain_on_unwind() {
        let released = std::cell::Cell::new(false);
        let drained_after_release = std::cell::Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _cleanup = ReleaseBeforeDrain {
                release: || released.set(true),
                drain: || drained_after_release.set(released.get()),
            };
            panic!("controlled resident fixture assertion unwind");
        }));
        assert!(result.is_err());
        assert!(
            drained_after_release.get(),
            "fixture drained before releasing its event"
        );
    }

    /// Distinct from the encoded-but-uncommitted fixture: an unsignalled
    /// existing shared-event wait keeps an actually submitted old reader held.
    #[cfg(feature = "metal")]
    #[test]
    fn resident_mapped_suffix_preserves_a_committed_inflight_prefix_reader() -> Result<()> {
        use objc2_metal::{MTLDevice as _, MTLSharedEvent as _};
        let device = Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        eprintln!(
            "FLASH_METADATA_NATIVE control=mapped-suffix-committed-inflight api={} cb_kind={:?}",
            metal.commands().actual_command_api(),
            metal.command_buffer_kind()
        );
        metal.set_compute_per_buffer(usize::MAX);
        // Complete the exact cast before the GPU is held: a cold compiler or
        // pipeline creation has no bounded relation to a native event timeout.
        assert_eq!(
            Tensor::new(&[0u32], &device)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
            &[0.]
        );
        let mut owner = ResidentMetadata::new(256, 1, 4)?;
        owner.publish_suffix(&[2], &device)?;
        let old_count = ResidentCount::new(1, &device)?;
        let next_count = ResidentCount::new(257, &device)?;
        let event = metal
            .metal_device()
            .as_ref()
            .newSharedEvent()
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("resident fixture newSharedEvent returned nil".into())
            })?;
        // Independent owners survive an unknown terminality outcome. Tensor
        // aliases are confined to this privileged test; no caller can get them
        // from the safe resident owner. All actual data writes still use its
        // production publish_suffix/publish_with implementation.
        let pins = std::cell::RefCell::new(Some((
            device.clone(),
            event.clone(),
            vec![
                owner.table_t.as_ref().expect("published").clone(),
                old_count.tensor.clone(),
                next_count.tensor.clone(),
            ],
        )));
        let cleanup_terminal = std::cell::Cell::new(false);
        // Arm BEFORE encoding even the first possibly blocking wait. On every
        // early return or assertion unwind release the event before draining.
        let cleanup = ReleaseBeforeDrain {
            release: || event.setSignaledValue(1),
            drain: || {
                let drained = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    metal.flush_and_wait_current()
                }));
                if matches!(drained, Ok(Ok(()))) {
                    cleanup_terminal.set(true);
                    drop(pins.borrow_mut().take());
                } else {
                    std::mem::forget(pins.borrow_mut().take());
                    eprintln!("resident fixture cleanup unknown; independent owners retained");
                }
            },
        };
        metal
            .commands()
            .encode_wait_for_event(&event, 1)
            .map_err(lloom_tensor::Error::wrap)?;
        let (old_prefix, old_live) = {
            let read = owner.read(&old_count)?;
            let (table, count) = read.bindings();
            (
                table.narrow(0, 0, 1)?.to_dtype(DType::F32)?,
                count.to_dtype(DType::F32)?,
            )
        };
        pins.borrow_mut()
            .as_mut()
            .expect("cleanup not run")
            .2
            .extend([old_prefix.clone(), old_live.clone()]);
        let held = metal.flush_marker()?;
        assert!(
            !metal.commands().marker_failed(&held),
            "fixture reader failed before append"
        );
        metal
            .commands()
            .check_failed()
            .map_err(lloom_tensor::Error::wrap)?;
        assert!(
            !metal.commands().marker_complete(&held),
            "fixture did not hold the submitted old reader"
        );
        // The only permitted mutation is the unpublished suffix. Refusing the
        // upload fallback makes successful mapped-write coverage non-vacuous.
        owner.publish_with(&[3], &device, |_, _| {
            lloom_tensor::bail!("inflight control unexpectedly used replacement upload")
        })?;
        let (new_prefix, new_live) = {
            let read = owner.read(&next_count)?;
            let (table, count) = read.bindings();
            (
                table.narrow(0, 0, 2)?.to_dtype(DType::F32)?,
                count.to_dtype(DType::F32)?,
            )
        };
        pins.borrow_mut()
            .as_mut()
            .expect("cleanup not run")
            .2
            .extend([new_prefix.clone(), new_live.clone()]);
        assert!(
            !metal.commands().marker_complete(&held),
            "submitted old reader completed before release"
        );

        assert!(
            !metal.commands().marker_failed(&held),
            "fixture reader failed during mapped append"
        );
        metal
            .commands()
            .check_failed()
            .map_err(lloom_tensor::Error::wrap)?;
        event.setSignaledValue(1);
        assert_eq!(old_prefix.to_vec1::<f32>()?, &[2.]);
        assert_eq!(old_live.to_vec1::<f32>()?, &[1.]);
        assert_eq!(new_prefix.to_vec1::<f32>()?, &[2., 3.]);
        assert_eq!(new_live.to_vec1::<f32>()?, &[257.]);
        assert!(metal.commands().marker_complete(&held));
        drop(cleanup);
        assert!(
            cleanup_terminal.get(),
            "resident fixture terminality is unknown"
        );
        Ok(())
    }
}
