//! Resident publication summaries. No per-layer table scan or device read.
//! Integration with the existing PagedKv producer remains on #1147.
use super::{native_int, product, validate_chunk_count, PageSummary};
use lloom_tensor::Result;

// The table and its index are the authoritative immutable published prefix.
// One u32 prefix maximum per published chunk. U32::MAX denotes an end outside
// the signed native domain; a partial final chunk does not consume its full end.
#[derive(Clone, Debug)]
pub(crate) struct ResidentSpans {
    page_size: usize,
    pages_per_chunk: usize,
    chunk: usize,
    entries: Vec<u32>,
    full_prefix_ends: Vec<u32>,
}

impl ResidentSpans {
    pub(crate) fn new(page_size: usize, pages_per_chunk: usize) -> Result<Self> {
        if page_size == 0 || pages_per_chunk == 0 {
            lloom_tensor::bail!("flash-decoding: resident chunk geometry must be positive")
        }
        let chunk = product(&[page_size, pages_per_chunk])?;
        for n in [page_size, pages_per_chunk, chunk] {
            native_int(n)?;
        }
        Ok(Self {
            page_size,
            pages_per_chunk,
            chunk,
            entries: Vec::new(),
            full_prefix_ends: Vec::new(),
        })
    }

    pub(crate) fn entries(&self) -> &[u32] {
        &self.entries
    }
    pub(crate) fn chunk(&self) -> usize {
        self.chunk
    }
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.full_prefix_ends.clear();
    }

    pub(crate) fn prepare_append<'a, 'b>(
        &'a mut self,
        suffix: &'b [u32],
    ) -> Result<PreparedResidentAppend<'a, 'b>> {
        // Every published first-page index participates in signed native
        // arithmetic before an empty-range branch. A full chunk end is checked
        // only when live, so valid partial-tail capacity is not rejected.
        for &page in suffix {
            if u64::from(page) * self.page_size as u64 > i32::MAX as u64 - 31 {
                lloom_tensor::bail!(
                    "flash-decoding: resident page start exceeds signed native token range"
                )
            }
        }
        self.entries
            .len()
            .checked_add(suffix.len())
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("flash-decoding: resident entry count overflow".into())
            })?;
        self.entries.try_reserve(suffix.len()).map_err(|e| {
            lloom_tensor::Error::Msg(format!("flash-decoding: resident host table reserve: {e}"))
        })?;
        self.full_prefix_ends
            .try_reserve(suffix.len())
            .map_err(|e| {
                lloom_tensor::Error::Msg(format!(
                    "flash-decoding: resident prefix index reserve: {e}"
                ))
            })?;
        Ok(PreparedResidentAppend {
            spans: self,
            suffix,
        })
    }

    pub(crate) fn summary(&self, keys: u32) -> Result<PageSummary> {
        let keys = keys as usize;
        if keys == 0 {
            lloom_tensor::bail!("flash-decoding: resident live count must be positive")
        }
        native_int(keys)?;
        let chunks = keys.div_ceil(self.chunk);
        validate_chunk_count(chunks)?;
        if chunks > self.entries.len() {
            lloom_tensor::bail!("flash-decoding: resident table is shorter than the live prefix")
        }
        let full = keys / self.chunk;
        let complete_end = if full == 0 {
            0
        } else {
            self.full_prefix_ends[full - 1] as usize
        };
        if complete_end > i32::MAX as usize - 31 {
            lloom_tensor::bail!(
                "flash-decoding: resident full prefix exceeds signed native token range"
            )
        }
        let tail = keys % self.chunk;
        let tail_end = if tail == 0 {
            0
        } else {
            let start = product(&[self.entries[full] as usize, self.page_size])?;
            if start > i32::MAX as usize - 31 {
                lloom_tensor::bail!(
                    "flash-decoding: resident tail start exceeds signed native token range"
                )
            }
            let end = start.checked_add(tail).ok_or_else(|| {
                lloom_tensor::Error::Msg("flash-decoding: resident tail end overflow".into())
            })?;
            if end > i32::MAX as usize - 31 {
                lloom_tensor::bail!(
                    "flash-decoding: resident tail end exceeds signed native token range"
                )
            }
            end
        };
        Ok(PageSummary {
            rows: 1,
            keys,
            page_size: self.page_size,
            pages_per_chunk: self.pages_per_chunk,
            max_read_end: complete_end.max(tail_end),
            min_ragged_keys: None,
        })
    }
}

/// Reserves host metadata before a native write; Drop commits no semantic state.
pub(crate) struct PreparedResidentAppend<'a, 'b> {
    spans: &'a mut ResidentSpans,
    suffix: &'b [u32],
}
impl PreparedResidentAppend<'_, '_> {
    pub(crate) fn start(&self) -> usize {
        self.spans.entries.len()
    }
    pub(crate) fn total(&self) -> usize {
        self.start() + self.suffix.len()
    }
    pub(crate) fn stage(&self, capacity: usize) -> Result<Vec<u32>> {
        let capacity = capacity.max(self.total());
        product(&[capacity, std::mem::size_of::<u32>()])?;
        let mut stage = Vec::new();
        stage.try_reserve_exact(capacity).map_err(|e| {
            lloom_tensor::Error::Msg(format!(
                "flash-decoding: resident upload staging reserve: {e}"
            ))
        })?;
        stage.extend_from_slice(&self.spans.entries);
        stage.extend_from_slice(self.suffix);
        stage.resize(capacity, 0);
        Ok(stage)
    }
    pub(crate) fn commit(self) {
        let mut end = self.spans.full_prefix_ends.last().copied().unwrap_or(0);
        for &page in self.suffix {
            // page_size/chunk fit I32, so this U64 expression is <2^63.
            let candidate = (u64::from(page) * self.spans.page_size as u64
                + self.spans.chunk as u64)
                .min(u64::from(u32::MAX)) as u32;
            end = end.max(candidate);
            self.spans.full_prefix_ends.push(end);
        }
        self.spans.entries.extend_from_slice(self.suffix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::attention::flash_validation::{prepare_pages, Pages};

    fn append(spans: &mut ResidentSpans, suffix: &[u32]) -> Result<()> {
        spans.prepare_append(suffix)?.commit();
        Ok(())
    }

    #[test]
    fn resident_prefixes_match_publication_oracle_through_permutation_and_rewind() -> Result<()> {
        let mut spans = ResidentSpans::new(256, 2)?;
        append(&mut spans, &[6, 0, 4])?;
        for keys in [1u32, 512, 513, 1024, 1025, 1536, 513, 1, 1536] {
            let indexed = spans.summary(keys)?;
            let oracle = prepare_pages(
                Pages {
                    entries: spans.entries(),
                    stride: 0,
                    keys: &[keys],
                },
                1,
                256,
                2,
            )?;
            assert_eq!(indexed.max_read_end, oracle.max_read_end, "keys={keys}");
            assert_eq!(indexed.keys, oracle.keys);
        }
        Ok(())
    }

    #[test]
    fn resident_partial_tail_does_not_require_the_unused_full_chunk() -> Result<()> {
        let mut spans = ResidentSpans::new(256, 1)?;
        append(&mut spans, &[3])?;
        assert_eq!(spans.summary(1)?.max_read_end, 769);
        assert_eq!(spans.summary(256)?.max_read_end, 1024);
        // A physically representable partial tail may have a full end outside
        // the signed loop domain. Its unused end must not cause early refusal.
        let page = (i32::MAX as u32 - 31) / 256;
        let mut edge = ResidentSpans::new(256, 1)?;
        append(&mut edge, &[page])?;
        assert_eq!(edge.summary(1)?.max_read_end, page as usize * 256 + 1);
        assert!(edge.summary(256).is_err());
        Ok(())
    }

    #[test]
    fn resident_dropped_append_preserves_prefix_and_failed_entry_never_publishes() -> Result<()> {
        let mut spans = ResidentSpans::new(256, 1)?;
        append(&mut spans, &[2])?;
        let before = spans.summary(1)?.max_read_end;
        drop(spans.prepare_append(&[3])?);
        assert_eq!(spans.entries(), &[2]);
        assert_eq!(spans.summary(1)?.max_read_end, before);
        assert!(spans.summary(257).is_err());
        assert!(spans.prepare_append(&[u32::MAX]).is_err());
        assert_eq!(spans.entries(), &[2]);
        Ok(())
    }

    #[test]
    fn resident_staging_is_the_single_padded_replacement_upload() -> Result<()> {
        let mut spans = ResidentSpans::new(256, 1)?;
        append(&mut spans, &[2])?;
        let prepared = spans.prepare_append(&[3])?;
        assert_eq!(prepared.start(), 1);
        assert_eq!(prepared.total(), 2);
        assert_eq!(prepared.stage(4)?, &[2, 3, 0, 0]);
        assert_eq!(prepared.stage(1)?, &[2, 3]);
        prepared.commit();
        assert_eq!(spans.entries(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn resident_clear_discards_old_index_before_recycled_prefix() -> Result<()> {
        let mut spans = ResidentSpans::new(256, 1)?;
        append(&mut spans, &[8, 9])?;
        spans.clear();
        assert!(spans.summary(1).is_err());
        append(&mut spans, &[0])?;
        assert_eq!(spans.summary(1)?.max_read_end, 1);
        Ok(())
    }
}
