//! `PagePool`: a page allocator with its liveness
//! (`profiles-and-memory-design.md` §4.4, L3 `PagePool<B>`).
//!
//! [`KvPagePool`] is a pure function of an allocation history — the tables,
//! the free list, the refcounts — with no device, buffer or dispatch, which
//! is what makes it testable without a GPU (§10.3i). What every owner of one
//! then wrote beside it, four times, is the half it leaves out: pages given
//! up on the host are not free until the command buffer that last read them
//! completes (§10.3h), so they wait in a [`Retire`] against the epoch open at
//! the release, come back at the next pass's drain, and a pass is sized by
//! the pages it can take — free, or drainable — never by positions alone
//! (`memory.md` §9.5r). This type is that half, once. The K / V storage the
//! pages index stays the owner's: its geometry is the model's.

use lloom_tensor::kv_pages::{KvPagePool, PageError, PageIdx};
use lloom_tensor::retire::{Completed, Epoch, Retire};
use lloom_tensor::Device;
#[cfg(feature = "metal")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PagePool {
    pages: KvPagePool,
    quarantine: Retire<PageIdx>,
    page_size: usize,
    #[cfg(feature = "metal")]
    clock: Option<Arc<lloom_tensor::kernels::metal::GpuClock>>,
}

impl PagePool {
    /// `pool_pages` pages of `page_size` tokens for `max_seqs` sequences whose
    /// tables hold `table_capacity_pages` entries ([`KvPagePool::new`]), with
    /// the device's clock for the liveness (none off Metal: every release is
    /// reusable at once, there is nothing in flight).
    pub fn new(
        page_size: usize,
        pool_pages: usize,
        max_seqs: usize,
        table_capacity_pages: usize,
        device: &Device,
    ) -> Self {
        #[cfg(feature = "metal")]
        let clock = match device {
            Device::Metal(d) => Some(d.gpu_clock()),
            _ => None,
        };
        #[cfg(not(feature = "metal"))]
        let _ = device;
        Self {
            pages: KvPagePool::new(page_size, pool_pages, max_seqs, table_capacity_pages),
            quarantine: Retire::new(),
            page_size: page_size.max(1),
            #[cfg(feature = "metal")]
            clock,
        }
    }

    /// The same pool over a clock the owner holds (a tier's layers share one).
    #[cfg(feature = "metal")]
    pub fn with_clock(
        page_size: usize,
        pool_pages: usize,
        max_seqs: usize,
        table_capacity_pages: usize,
        clock: Arc<lloom_tensor::kernels::metal::GpuClock>,
    ) -> Self {
        Self {
            pages: KvPagePool::new(page_size, pool_pages, max_seqs, table_capacity_pages),
            quarantine: Retire::new(),
            page_size: page_size.max(1),
            clock: Some(clock),
        }
    }

    /// The allocator: tables, free list, refcounts.
    pub fn pages(&self) -> &KvPagePool {
        &self.pages
    }

    /// The allocator, to append to, reserve from, rewind or share.
    pub fn pages_mut(&mut self) -> &mut KvPagePool {
        &mut self.pages
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// The epoch a page released now must outlive: the one open on the queue.
    pub fn epoch(&self) -> Epoch {
        #[cfg(feature = "metal")]
        {
            self.clock
                .as_ref()
                .map(|c| c.current_epoch())
                .unwrap_or(Epoch::ZERO)
        }
        #[cfg(not(feature = "metal"))]
        {
            Epoch::ZERO
        }
    }

    /// Epochs the GPU has finished (every one, without a clock).
    pub fn completed(&self) -> Completed {
        #[cfg(feature = "metal")]
        {
            self.clock
                .as_ref()
                .map(|c| c.completed_count())
                .unwrap_or(Completed::ALL)
        }
        #[cfg(not(feature = "metal"))]
        {
            Completed::ALL
        }
    }

    /// Give `seq`'s pages up: their refcounts drop, and those with no other
    /// referent wait for the epoch open now. Returns them, for an owner that
    /// keeps something per page.
    pub fn release(&mut self, seq: usize) -> Vec<PageIdx> {
        let freed = self.pages.release(seq);
        self.quarantine.retire_all(self.epoch(), freed.clone());
        freed
    }

    /// The raw recycle is the quarantine's to call: a page goes back through
    /// `recycle_ready`, never from a caller's own list.
    #[allow(dead_code)]
    fn recycle(&mut self, _pages: &[PageIdx]) -> Result<(), PageError> {
        Err(PageError::NotAllocated {
            page: PageIdx(u32::MAX),
        })
    }

    /// Pages whose last reader has completed, back on the free list; how
    /// many. Once per pass, before its appends.
    pub fn recycle_ready(&mut self) -> Result<usize, PageError> {
        let ready = self.quarantine.drain(self.completed());
        let n = ready.len();
        self.pages.recycle(&ready)?;
        Ok(n)
    }

    /// Pages held in the quarantine, not yet reusable.
    pub fn held(&self) -> usize {
        self.quarantine.held()
    }

    /// Whether anything is held.
    pub fn holds_any(&self) -> bool {
        !self.quarantine.is_empty()
    }

    /// Pages a set of appends — `(seq, tokens)` each — would fail to take from
    /// the pool as it stands: what their appends pop, less the free list and
    /// what the next drain releases. Zero means they fit without a grow.
    pub fn pages_short(&self, appends: impl IntoIterator<Item = (usize, usize)>) -> usize {
        let needed: usize = appends
            .into_iter()
            .map(|(seq, n)| {
                let (len, held) = self
                    .pages
                    .seq(seq)
                    .map(|q| (q.len(), q.pages().len()))
                    .unwrap_or((0, 0));
                (len + n).div_ceil(self.page_size).saturating_sub(held)
            })
            .sum();
        let available = self.pages.pages_free() + self.quarantine.ready(self.completed());
        needed.saturating_sub(available)
    }

    /// Add `extra` pages beneath the free list ([`KvPagePool::grow`]).
    pub fn grow(&mut self, extra: usize) {
        self.pages.grow(extra)
    }
}

/// The allocator's own methods — `append`, `reserve`, `rewind`, `seq`,
/// `table_for`, the counts — reach through; the ones with a liveness half
/// (`release`, `recycle`, `grow`) are this type's and shadow them.
impl std::ops::Deref for PagePool {
    type Target = KvPagePool;
    fn deref(&self) -> &KvPagePool {
        &self.pages
    }
}

impl std::ops::DerefMut for PagePool {
    fn deref_mut(&mut self) -> &mut KvPagePool {
        &mut self.pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_release_waits_for_its_epoch_and_the_shortfall_counts_it() {
        let dev = Device::Cpu;
        let mut p = PagePool::new(4, 4, 2, 5, &dev);
        p.pages_mut().append(0, 8).unwrap();
        assert_eq!(
            p.pages_short([(0, 8)]),
            0,
            "two free pages take the next eight"
        );
        assert_eq!(p.pages_short([(0, 12)]), 1);
        let freed = p.release(0);
        assert_eq!(freed.len(), 2);
        // Off Metal every epoch is complete: the pages are back at the drain.
        assert_eq!(p.held(), 2);
        assert_eq!(p.pages_short([(1, 16)]), 0, "the drainable pages count");
        assert_eq!(p.recycle_ready().unwrap(), 2);
        assert_eq!(p.pages().pages_free(), 4);
        p.grow(2);
        assert_eq!(p.pages().pool_pages(), 6);
    }
}
