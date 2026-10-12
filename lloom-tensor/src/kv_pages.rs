//! The paged KV pool: allocation, free list, refcounts and the page table
//! (`DESIGN.md` §10.3c–§10.3g, issue #148).
//!
//! # Why this half, and why first
//!
//! §10.3i costs paging and routes it here rather than to #116: *"paging is
//! #148's prerequisite, not #116's"*. It also says what to build first, and it
//! is **not** the kernel:
//!
//! > *"What should be built first if this is picked up, and it is not the page
//! > table. The pool + free list + refcount is CPU-side, testable without a GPU,
//! > and independently useful; the page-walking kernel is the risky half.
//! > **Sequencing them the other way — kernel first, allocator second — is what
//! > makes the determinism hazard in §10.3h invisible until it is expensive**,
//! > because a kernel tested against a contiguously-allocated pool is a kernel
//! > tested at `page_table[c] == c`."*
//!
//! That is the sequencing this module follows. Everything here is a pure
//! function of a configuration and an allocation history: no device, no buffer,
//! no dispatch. What it produces is the **table a kernel would be handed**, so
//! the permuted-table arm that separates an index from a walk
//! (`flash_decoding_tests::permuted_chunk_table_is_read_by_index_not_by_walk`)
//! can be driven from a real allocation history rather than from a hand-written
//! `vec![1, 0]`.
//!
//! # What paging buys, and the three thresholds it buys it at
//!
//! §10.3i's verdict is that at **B=1 short context paging is a net loss** — it
//! saves at most 60 MiB, 1.1 % of a pool #23 already bounded — and that the
//! inversion has three separate mechanisms which must not be collapsed into one
//! number:
//!
//! | threshold | mechanism |
//! |---|---|
//! | ~8k tokens, B=1 | `Reserve` becomes a real fraction of the pool: 2.4 % at 128 MiB, **9.6 % at 32k** |
//! | **B ≥ 4, any context** | the reserve is multiplied by B and paging is not — **256 MiB at B=4, 1 GiB at B=16** |
//! | **any context, with sharing** | a shared prefix is stored once instead of B times — **unbounded in B** |
//!
//! The second and third are #148's, which is why this lands here. [`KvPagePool`]
//! is what makes the second measurable: [`KvPagePool::pages_in_use`] against
//! [`reserve_pages_for`] is the saving, computed from an allocation history
//! rather than from the arithmetic.
//!
//! # The determinism hazard this module exists to keep testable
//!
//! §10.3h names a **second** wrong merge order that is not completion order: the
//! combine must **index** the page table by chunk number and never **walk** it.
//! Both visit every chunk; only the first is bit-stable, because a table's
//! *contents* depend on allocation history, which under B>1 depends on what
//! other sequences did — so a sequence's logits would depend on its batch
//! neighbours, violating §2.3.3 #7.
//!
//! **A B=1 gate structurally cannot detect it**, because at B=1 with a fresh
//! pool the table is the identity and the two orders coincide. This module is
//! what produces a table that is *not* the identity from an ordinary sequence of
//! allocations and frees — see [`KvPagePool::table_for`] and the
//! `a_recycled_pool_produces_a_non_identity_table` test — so the hazard has a
//! fixture rather than only a paragraph.
//!
//! # What this module deliberately does not do
//!
//! **No kernel, no buffer, no dispatch.** §10.3e is explicit that a paged read
//! cannot be built out of candle `Tensor` operations — `slice_set` requires both
//! sides contiguous and `narrow` produces a strided view of one allocation, so
//! neither can express *"tokens 0..255 are at pool offset 7·P and tokens
//! 256..511 are at 2·P"*. The page-walking append and attention kernels are the
//! risky half and are not here.
//!
//! **No eviction policy.** §10.3f decides the *mechanism* — a page returns to
//! the free list when its refcount hits zero — and defers the *policy* (LRU,
//! recompute, swap-to-host, preempt-a-sequence) to a scheduler that does not
//! exist. Exhaustion of a pool at its ceiling is a loud failure, following
//! #150. **The pool does grow below the ceiling** — [`KvPagePool::grow`], driven
//! by the cache in buckets and admitted against the device first — and the
//! cost §10.3f named for that is paid as named: the pool buffer is reallocated
//! and copied and its base pointer moves, once per bucket, with every page
//! index and every table unchanged.
//!
//! **No eviction policy, and no page-walking kernel** — see above. What this
//! module *does* now carry is the liveness half, added 2026-09-02 (#148):
//!
//! §10.3h's one non-fence obligation is that a page returns to the free list
//! *when the command buffer holding its last read completes*, not when its CPU
//! refcount hits zero — §6.7 L4, and the bug #23 fixed in the buffer pool
//! reappearing one layer up. [`KvPagePool::release`] is the refcount half and
//! [`PageQuarantine`] is the epoch half, in the shape §10.3h names (§6.3b's
//! counter, which is [`GpuClock`](super::buffer_pool::GpuClock)).
//!
//! **The two are separate types on purpose.** The pool stays a pure function of
//! a configuration and an allocation history — no device, no buffer, no
//! dispatch, which is what makes it testable without a GPU and is what §10.3i
//! asked to be built first. The quarantine holds the epoch, so the thing that
//! needs to know about command buffers is the thing that knows about command
//! buffers, and `release` → `quarantine` → `drain_completed` → `recycle` is a
//! path whose unsafe shortcut (`release` → `recycle`) is still expressible and
//! is now named as the defect it is rather than being the only option.

/// A page index into the pool. `u32` and not a pointer, per §10.3c.
///
/// **The width is a decision and the reason is the ICB.** §10.3c: page
/// *addresses* would make the table a vector of pointers into possibly-different
/// allocations, which reopens the residency question §9.2d closed and which
/// §3.5's `useResource` requirement makes expensive. Indices keep everything
/// `base + offset` — the kernel computes `kv_pool + page_idx * page_stride`, one
/// base pointer and arithmetic on a number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PageIdx(pub u32);

impl PageIdx {
    /// The page's position in the pool's arrays.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The page at position `i`.
    pub fn from_index(i: usize) -> Self {
        Self(i as u32)
    }

    /// The indices as the kernel reads them (`base + page × stride`): the
    /// same bytes, viewed as `u32`.
    pub fn as_u32s(pages: &[PageIdx]) -> &[u32] {
        // SAFETY: `PageIdx` is `repr(transparent)` over `u32`, so a slice of
        // one is a slice of the other, same length and alignment.
        unsafe { std::slice::from_raw_parts(pages.as_ptr() as *const u32, pages.len()) }
    }
}

impl std::fmt::Display for PageIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub use crate::retire::{Completed, Epoch, Retire};

/// Why a page allocation failed.
///
/// Exhaustion is an error rather than a grow *here*: the pool is a pure
/// function of its history and allocates no buffer, so it cannot widen itself.
/// The cache widens it ([`KvPagePool::grow`]) before a pass that needs the
/// room, up to its ceiling; a pool that is exhausted was pinned
/// (`LLOOM_148_POOL_PAGES`) or is at that ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageError {
    /// The free list is empty.
    ///
    /// Carries the pool size so the message names the knob a caller must raise,
    /// which is the half of #150's loud failure that made it actionable.
    Exhausted { pool_pages: usize },
    /// A sequence asked for a page beyond the table it was given.
    TableFull { seq: usize, capacity_pages: usize },
    /// A free or share named a page that is not allocated.
    ///
    /// Distinguished from `Exhausted` because it is a caller defect rather than
    /// a capacity one, and silence here would be §3.5's silent corruption: a
    /// double free returns a live page to the list and a later allocation
    /// aliases two sequences onto it.
    NotAllocated { page: PageIdx },
}

impl std::fmt::Display for PageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageError::Exhausted { pool_pages } => write!(
                f,
                "KV page pool exhausted: all {pool_pages} pages are allocated. Raise the pool \
                 size: it grows with the KV slab up to the cache's ceiling (`kv_capacity`) \
                 unless LLOOM_148_POOL_PAGES pins it, and one of those two is what it hit \
                 (DESIGN.md 10.3f)"
            ),
            PageError::TableFull {
                seq,
                capacity_pages,
            } => write!(
                f,
                "sequence {seq} has filled its page table ({capacity_pages} pages): its context \
                 is longer than the table was sized for (DESIGN.md 10.3c)"
            ),
            PageError::NotAllocated { page } => write!(
                f,
                "page {page} is not allocated: a double free would hand a live page back to the \
                 free list and alias two sequences onto it, which DESIGN.md 3.5 makes silent"
            ),
        }
    }
}

impl std::error::Error for PageError {}

/// Pages a `Reserve`-style cache would hold for `b` sequences at `max_context`.
///
/// **This is the quantity paging removes**, and it is the arithmetic behind
/// §10.3i's second inversion threshold: the reserve is multiplied by B and
/// paging is not. `4096 × B` tokens is 256 MiB at B=4 and 1 GiB at B=16.
///
/// A free function rather than a method because it describes the *incumbent*
/// (#150's per-sequence reservation), not the pool — a saving is only meaningful
/// against what it replaces, and computing both from one place is what keeps the
/// comparison honest (§3.1b rule 1: prefer an absolute reference).
pub fn reserve_pages_for(b: usize, max_context: usize, page_size: usize) -> usize {
    b * max_context.div_ceil(page_size.max(1))
}

/// One sequence's page table: the pages holding its KV, in token order.
///
/// Entry `i` names the page holding tokens `[i·page_size, (i+1)·page_size)`.
/// §10.3c stores every sequence's entries in **one** buffer with a per-sequence
/// base offset; this type is the logical view of one sequence's slice of it, and
/// [`KvPagePool::table_for`] is what materialises the entries a kernel is
/// handed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeqPages {
    pages: Vec<PageIdx>,
    len: usize,
}

impl SeqPages {
    /// Live tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pages backing this sequence, in token order.
    pub fn pages(&self) -> &[PageIdx] {
        &self.pages
    }

    /// The pool cell of each token in `tokens` — `page · page_size + offset`,
    /// what a scatter writes to and a gather reads from. The one formula
    /// behind every owner's slot list.
    pub fn cells(
        &self,
        page_size: usize,
        tokens: std::ops::Range<usize>,
    ) -> impl Iterator<Item = u32> + '_ {
        tokens.map(move |t| self.pages[t / page_size].0 * page_size as u32 + (t % page_size) as u32)
    }
}

/// A fixed pool of KV pages with a free list and per-page refcounts.
///
/// # The three structures, and why the free list cannot be a stack alone
///
/// §10.3g rule 1: a refcounted page returns only when the **last** referent
/// drops it, so the free list is a `Vec<u32>` *plus* a `refcount: Vec<u16>`.
/// Building the refcount array now costs 2 bytes per page — 1 KiB for a
/// 512-page pool — and building it later means auditing every return site. It is
/// here from the first commit for that reason, and because prefix sharing
/// (§10.3g) is one of the two mechanisms that make paging pay at B>1 at all.
///
/// # Allocation is a pop, and it is off the per-token path
///
/// §10.3f: a page is allocated at `len % page_size == 0` — the token that would
/// overflow the current tail page — so at page size 256 and 18.76 ms/token that
/// is one allocation per ~4.8 seconds of decode at B=1. Every other token writes
/// into a page that already exists, making the steady-state per-token allocator
/// cost **zero**, which is §6.7 L4b's invariant and stronger than #150's
/// (allocate once, then bail).
///
/// It is CPU-side and that is decided rather than defaulted: §9.3 records that
/// MSL `device` atomics accept only `memory_order_relaxed`, which §9.2d case 3
/// says makes a general GPU-side free list *"a research project rather than an
/// engineering task"*.
#[derive(Clone, Debug)]
pub struct KvPagePool {
    page_size: usize,
    /// Free pages, popped from the back — O(1), §10.3f.
    free: Vec<PageIdx>,
    /// Referents per page. `u16` per §10.3g rule 1.
    refcount: Vec<u16>,
    /// Per-sequence tables, and the cap each may reach (§10.3c).
    seqs: Vec<SeqPages>,
    table_capacity_pages: usize,
}

impl KvPagePool {
    /// A pool of `pool_pages` pages of `page_size` tokens each, for `max_seqs`
    /// sequences whose tables hold `table_capacity_pages` entries.
    ///
    /// **The table is reserved outright and the pool is not**, which is §10.3c's
    /// sizing argument and the asymmetry that makes paging work at all: at page
    /// size 256 one 128k sequence is 512 entries = **2 KiB** of table against a
    /// 2 GB page pool, so *"the table is 0.0001 % of what it describes"*. The
    /// thing that must be exact is large; the thing that indexes it is tiny.
    pub fn new(
        page_size: usize,
        pool_pages: usize,
        max_seqs: usize,
        table_capacity_pages: usize,
    ) -> Self {
        // Descending, so the first `pop` yields page 0 and a fresh pool hands
        // out 0, 1, 2, ... — the identity table. That is deliberate: it makes a
        // fresh-pool run reproduce the contiguous cache's layout exactly, so
        // `ChunkTable::Contiguous` and a fresh paged pool are the same table and
        // #568's page-invariance digests stay comparable. A recycled pool then
        // diverges from it, which is the fixture §10.3h needs.
        let free: Vec<PageIdx> = (0..pool_pages).rev().map(PageIdx::from_index).collect();
        Self {
            page_size: page_size.max(1),
            free,
            refcount: vec![0; pool_pages],
            seqs: vec![SeqPages::default(); max_seqs],
            table_capacity_pages,
        }
    }

    /// Tokens per page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Total pages in the pool.
    pub fn pool_pages(&self) -> usize {
        self.refcount.len()
    }

    /// Pages currently allocated to at least one sequence.
    ///
    /// **This is the numerator of the saving** §10.3i's second threshold
    /// describes: against [`reserve_pages_for`] it is what paging holds where
    /// `Reserve` holds `B × max_context`.
    pub fn pages_in_use(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 0).count()
    }

    /// Pages on the free list.
    pub fn pages_free(&self) -> usize {
        self.free.len()
    }

    /// A sequence's table.
    pub fn seq(&self, seq: usize) -> Option<&SeqPages> {
        self.seqs.get(seq)
    }

    /// Add `extra` pages to the pool, beneath the free pages it already holds.
    ///
    /// Growth is the cache's: it reallocates the pool buffer wider and copies
    /// the old one into the prefix, so every existing page index still names
    /// the run of tokens it did and no table needs a rewrite. The new pages
    /// are spliced *under* the free list rather than pushed onto it, so a
    /// recycled page keeps popping before a fresh one — the allocation order a
    /// pool created this wide would have followed, which keeps a grown pool's
    /// tables a function of the allocation history alone (§10.3h) rather than
    /// of when it grew. The table capacity widens by the same count: a table
    /// that could not name the new pages would report `TableFull` for a pool
    /// with room.
    pub fn grow(&mut self, extra: usize) {
        let first = self.refcount.len();
        let fresh: Vec<PageIdx> = (first..first + extra)
            .rev()
            .map(PageIdx::from_index)
            .collect();
        self.free.splice(0..0, fresh);
        self.refcount.extend(std::iter::repeat_n(0, extra));
        self.table_capacity_pages += extra;
    }

    /// Append `n` tokens to `seq`, allocating pages as they are needed.
    ///
    /// Returns the pages allocated by this call, which is empty in the common
    /// case: only the token that crosses a page boundary allocates.
    pub fn append(&mut self, seq: usize, n: usize) -> Result<Vec<PageIdx>, PageError> {
        let mut fresh = Vec::new();
        for _ in 0..n {
            let s = &self.seqs[seq];
            // A page is needed exactly when the live length fills the pages
            // already held -- `len % page_size == 0` at a page boundary (§10.3f).
            if s.len == s.pages.len() * self.page_size {
                if s.pages.len() == self.table_capacity_pages {
                    return Err(PageError::TableFull {
                        seq,
                        capacity_pages: self.table_capacity_pages,
                    });
                }
                let page = self.free.pop().ok_or(PageError::Exhausted {
                    pool_pages: self.refcount.len(),
                })?;
                self.refcount[page.index()] += 1;
                self.seqs[seq].pages.push(page);
                fresh.push(page);
            }
            self.seqs[seq].len += 1;
        }
        Ok(fresh)
    }

    /// Hold pages for `seq` so that `extra` more tokens fit past its live
    /// length without allocating, returning the pages allocated by this call.
    ///
    /// A caller that writes past `len` without moving it — a speculative
    /// drafter's own block, keyed after its context and attended in the same
    /// pass, then overwritten by the next append (#708) — needs the slots to
    /// exist while nothing but that caller reads them. The reserved pages are
    /// the sequence's: [`Self::table_for`] names them, [`Self::append`] fills
    /// them before allocating more, [`Self::rewind`] keeps them and
    /// [`Self::release`] returns them with the rest.
    pub fn reserve(&mut self, seq: usize, extra: usize) -> Result<Vec<PageIdx>, PageError> {
        let mut fresh = Vec::new();
        loop {
            let s = &self.seqs[seq];
            if s.pages.len() * self.page_size >= s.len + extra {
                return Ok(fresh);
            }
            if s.pages.len() == self.table_capacity_pages {
                return Err(PageError::TableFull {
                    seq,
                    capacity_pages: self.table_capacity_pages,
                });
            }
            let page = self.free.pop().ok_or(PageError::Exhausted {
                pool_pages: self.refcount.len(),
            })?;
            self.refcount[page.index()] += 1;
            self.seqs[seq].pages.push(page);
            fresh.push(page);
        }
    }

    /// Discard the last `n` tokens of `seq`, keeping its pages.
    ///
    /// A speculative rollback (§10.2a) is a length decrement: the discarded
    /// positions are above `len` and no read reaches them, and the page that
    /// held them is the one the next append writes into again. Freeing it
    /// would be wrong twice over — the sequence still owns the tokens below
    /// it, and a page released while a command buffer reads it is the hazard
    /// `PageQuarantine` exists for.
    pub fn rewind(&mut self, seq: usize, n: usize) -> Result<(), PageError> {
        let s = &mut self.seqs[seq];
        if n > s.len {
            return Err(PageError::TableFull {
                seq,
                capacity_pages: self.table_capacity_pages,
            });
        }
        s.len -= n;
        Ok(())
    }

    /// The chunk table a kernel is handed for `seq`, at `pages_per_chunk`.
    ///
    /// Entry `c` is the **first page** of chunk `c`, which is exactly
    /// `ChunkTable::Explicit`'s contract and what
    /// `flash_decoding.metal` reads: *"`chunk_table[c]` names a PAGE, and a
    /// chunk spans `pages_per_chunk` of them."*
    ///
    /// # This is the function that makes §10.3h testable
    ///
    /// On a fresh pool the result is `[0, k, 2k, ...]` — byte-identical to
    /// `ChunkTable::Contiguous`, so a paged run and a contiguous one are the same
    /// table and any digest difference is attributable elsewhere. After frees and
    /// reallocations it is **not** the identity, and that is the arm no B=1
    /// fixture with a fresh pool can produce: §10.3h's *"a fixture that does not
    /// permute its table is testing the identity function"*, reached from an
    /// ordinary allocation history rather than from a literal.
    ///
    /// **A chunk must not straddle a discontiguity.** The kernel resolves one
    /// page per chunk and then walks `pages_per_chunk * page_size` tokens
    /// linearly from it, so at `k > 1` the pages of a chunk must be consecutive.
    /// [`Self::chunk_pages_are_contiguous`] is the check, and it is a check
    /// rather than an assumption because §3.5 makes the failure silent.
    pub fn table_for(&self, seq: usize, pages_per_chunk: usize) -> Vec<u32> {
        let mut table = Vec::new();
        self.extend_table_for(seq, pages_per_chunk, &mut table);
        table
    }

    /// Append a sequence's chunk table into caller-owned staging, retaining its
    /// allocation across passes. The owned form delegates here so addressing
    /// cannot drift between the reusable and convenience APIs.
    pub fn extend_table_for(&self, seq: usize, pages_per_chunk: usize, out: &mut Vec<u32>) {
        if let Some(s) = self.seqs.get(seq) {
            out.extend(s.pages.iter().step_by(pages_per_chunk.max(1)).map(|p| p.0));
        }
    }

    /// Whether every chunk's pages are consecutive in the pool, at `k > 1`.
    ///
    /// Trivially true at `k == 1` (a chunk is one page) and at `k > 1` it is a
    /// real constraint a recycled pool can violate — which is why the allocator
    /// exists before the kernel. A caller that cannot satisfy it must either
    /// allocate chunk-aligned runs or keep `k == 1`; discovering it as a wrong
    /// read would be silent (§3.5).
    pub fn chunk_pages_are_contiguous(&self, seq: usize, pages_per_chunk: usize) -> bool {
        let k = pages_per_chunk.max(1);
        if k == 1 {
            return true;
        }
        let Some(s) = self.seqs.get(seq) else {
            return true;
        };
        s.pages
            .chunks(k)
            .all(|run| run.windows(2).all(|w| w[1].0 == w[0].0 + 1))
    }

    /// Point `dst`'s first `n_pages` at `src`'s, sharing them (§10.3g).
    ///
    /// **This is the third inversion threshold, and it is unbounded in B**: a
    /// shared prefix is stored once instead of B times. §10.3c's
    /// shared-table-with-base-offset form makes it an assignment — two sequences
    /// share by their entries naming the same page index, with no aliasing of
    /// buffers and no residency change.
    ///
    /// **Only complete pages may be shared**, per §10.3g rule 3: a partial tail
    /// page is still being written, so sharing it would need copy-on-write on
    /// every token. Sharing whole pages makes CoW a boundary case rather than a
    /// steady-state one, and is why this takes a page count rather than a token
    /// count.
    pub fn share_prefix(
        &mut self,
        src: usize,
        dst: usize,
        n_pages: usize,
    ) -> Result<(), PageError> {
        let pages: Vec<PageIdx> = self.seqs[src].pages.iter().take(n_pages).copied().collect();
        if self.seqs[dst].pages.len() + pages.len() > self.table_capacity_pages {
            return Err(PageError::TableFull {
                seq: dst,
                capacity_pages: self.table_capacity_pages,
            });
        }
        for &p in &pages {
            if self.refcount[p.index()] == 0 {
                return Err(PageError::NotAllocated { page: p });
            }
        }
        for &p in &pages {
            self.refcount[p.index()] += 1;
            self.seqs[dst].pages.push(p);
            self.seqs[dst].len += self.page_size;
        }
        Ok(())
    }

    /// Drop `seq`'s pages, returning those whose last referent went away.
    ///
    /// # The refcount is the mechanism and the epoch is the missing half
    ///
    /// §10.3f decides the mechanism: a page returns to the free list when its
    /// refcount hits zero. §10.3h names the obligation that is **not** a fence
    /// and that no fence can substitute for — a page returned and immediately
    /// reallocated to another sequence is a **write-after-read across
    /// sequences**, and the new owner writes bytes the old reader may still be
    /// reading on the GPU. That is §6.7 L4 exactly, and the bug #23 fixed in the
    /// buffer pool reappearing one layer up.
    ///
    /// **The fix is known and is not implemented here**: a page returns to the
    /// free list when the command buffer holding its last read *completes*, not
    /// when its CPU refcount reaches zero — §6.3b's epoch counter is the shape.
    /// This function is the refcount half, and it returns the freed pages rather
    /// than pushing them straight back so a GPU-liveness gate can sit between
    /// the two without this type having to know about command buffers. **A
    /// caller that pushes the result back immediately has the #23 bug**, which
    /// is why the return value is not `()`.
    pub fn release(&mut self, seq: usize) -> Vec<PageIdx> {
        let pages = std::mem::take(&mut self.seqs[seq].pages);
        self.seqs[seq].len = 0;
        let mut freed = Vec::new();
        for p in pages {
            let rc = &mut self.refcount[p.index()];
            *rc -= 1;
            if *rc == 0 {
                freed.push(p);
            }
        }
        freed
    }

    /// Return pages to the free list, after their last GPU read has completed.
    ///
    /// Separate from [`Self::release`] so the GPU-liveness obligation §10.3h
    /// names has somewhere to live: the caller holds the freed pages until the
    /// command buffer that read them completes, then calls this. See
    /// [`Self::release`] for why the two are not one function.
    ///
    /// **Prefer [`PageQuarantine`] to calling this directly.** This is the
    /// unguarded primitive; the quarantine is the thing that decides *when* the
    /// call is safe, and it is the half §10.3h says no fence can substitute for.
    pub fn recycle(&mut self, pages: &[PageIdx]) -> Result<(), PageError> {
        for &p in pages {
            if self.refcount[p.index()] != 0 {
                return Err(PageError::NotAllocated { page: p });
            }
            self.free.push(p);
        }
        Ok(())
    }
}

/// Items freed on the CPU and not yet safe to reuse, held until the command
/// buffer that last read them completes: [`Retire`], the one liveness rule
/// (`retire.rs`). The name survives for the pages' half of §10.3h — *"a page
/// returns to the free list when the command buffer holding its last read
/// completes, not when its refcount hits zero on the CPU"* — which is the
/// obligation [`KvPagePool::release`] hands over and [`KvPagePool::recycle`]
/// discharges; the two stay separate so the pool remains a pure function of
/// its allocation history, with no device, buffer or dispatch.
pub type EpochQuarantine<I> = Retire<I>;

/// The KV pages' quarantine (§10.3h).
pub type PageQuarantine = Retire<PageIdx>;
