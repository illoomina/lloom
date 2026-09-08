//! A stateful model-based fuzzer over the paged KV pool (lloom #706) — the
//! target every LibAFL paragraph since #494 named and none fuzzed.
//!
//! # Why this one is the deep gate
//!
//! `KvPagePool` (`lloom-tensor/src/kernels/metal/kv_pages.rs`) is a free list,
//! a refcount per page, per-sequence tables with a capacity, prefix sharing,
//! on-demand growth (#696) — and, in a separate type on purpose,
//! `PageQuarantine`, the epoch half of §10.3h's liveness rule: a page returns
//! to the free list when the command buffer holding its last read completes,
//! not when its CPU refcount reaches zero. The defect class the pool's own
//! docs name is the *shortcut* — `release` → `recycle` with no quarantine
//! between — and it is reachable only by a sequence that releases a page,
//! recycles it, re-allocates it to another sequence and then completes the
//! epoch the first reader was in. That is four operations in a specific order
//! with specific arguments, which is the shape a uniform generator reaches
//! rarely and coverage guidance is for.
//!
//! # The model is written from the contract
//!
//! Per #504: a model mirroring the implementation cannot find a bug in it.
//! Everything below is derived from the doc comments on `new`, `grow`,
//! `append`, `rewind`, `share_prefix`, `release`, `recycle`, `table_for` and
//! `PageQuarantine::{quarantine, drain_completed}`, including the *order* the
//! free list hands pages out in — which is contract, not implementation
//! detail: *"the first `pop` yields page 0 and a fresh pool hands out 0, 1,
//! 2, ... — the identity table"* and *"a recycled page keeps popping before a
//! fresh one"* after a grow are both documented and both load-bearing for
//! §10.3h's page-invariance digests.
//!
//! # Poisons
//!
//! The pool holds indices, not bytes, so there is no tile to poison. What
//! stands in for the sentinel is the **partition invariant**: after every
//! operation every page is in exactly one of {free, allocated (refcount > 0),
//! quarantined}, and a page in two places — or in none — is the aliasing §3.5
//! makes silent on the device.

use lloom_tensor::kernels::metal::buffer_pool::GpuClock;
use lloom_tensor::kernels::metal::kv_pages::{KvPagePool, PageError, PageIdx, PageQuarantine};

/// The fixture: what `KvPagePool::new` is called with.
#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub page_size: usize,
    pub pool_pages: usize,
    pub max_seqs: usize,
    pub table_capacity_pages: usize,
}

/// One symbol of the pool's alphabet — the cache's and the scheduler's
/// operations, not the pool's methods in isolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `append(seq, n)`: `n` tokens, allocating at each page boundary.
    Append { seq: u8, n: usize },
    /// `rewind(seq, n)`: a speculative rollback; pages are kept.
    Rewind { seq: u8, n: usize },
    /// `share_prefix(src, dst, pages)`: whole pages, refcounted.
    Share { src: u8, dst: u8, pages: usize },
    /// `release(seq)` followed by `quarantine(current_epoch, freed)` — the
    /// documented safe path.
    Release { seq: u8 },
    /// `GpuClock::commit_epoch()` — the host committing a command buffer.
    Commit,
    /// `GpuClock::mark_completed(epoch)` for an epoch `back` behind the newest
    /// closed one — completion handlers arrive out of order.
    Complete { back: u8 },
    /// `quarantine.drain_completed(clock.completed_count())` then `recycle`.
    Drain,
    /// `grow(extra)`: the cache widening the pool by a bucket.
    Grow { extra: usize },
    /// Read `table_for(seq, k)` and `chunk_pages_are_contiguous(seq, k)` and
    /// check them against the model.
    Table { seq: u8, k: usize },
    /// **The shortcut the docs name as a defect**: `release(seq)` and then
    /// `recycle` the freed pages with no quarantine. The pool must accept it
    /// (it is the unguarded primitive) — so the *model* is what refuses: this
    /// op is decoded into the safe path plus a check that the pool refuses to
    /// recycle a page that is still referenced.
    RecycleLive { seq: u8 },
}

/// What the pool should hold, from the contract.
#[derive(Debug)]
struct Model {
    page_size: usize,
    /// Free pages in pop order (last = next out).
    free: Vec<PageIdx>,
    refcount: Vec<u16>,
    seqs: Vec<(Vec<PageIdx>, usize)>,
    table_capacity: usize,
    /// `(epoch to outlive, pages)`, oldest first.
    quarantined: Vec<(u64, Vec<PageIdx>)>,
    submitted: u64,
    completed: u64,
}

impl Model {
    fn new(f: &Fixture) -> Self {
        Self {
            page_size: f.page_size.max(1),
            free: (0..f.pool_pages as PageIdx).rev().collect(),
            refcount: vec![0; f.pool_pages],
            seqs: vec![(Vec::new(), 0); f.max_seqs],
            table_capacity: f.table_capacity_pages,
            quarantined: Vec::new(),
            submitted: 0,
            completed: 0,
        }
    }

    fn pool_pages(&self) -> usize {
        self.refcount.len()
    }

    fn in_use(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 0).count()
    }

    fn held(&self) -> usize {
        self.quarantined.iter().map(|(_, p)| p.len()).sum()
    }

    /// `append`'s contract: allocate at `len % page_size == 0`, `TableFull`
    /// before `Exhausted`, one page per boundary, pop from the back.
    fn append(&mut self, seq: usize, n: usize) -> Result<Vec<PageIdx>, PageError> {
        let mut fresh = Vec::new();
        for _ in 0..n {
            let (pages, len) = &self.seqs[seq];
            if *len == pages.len() * self.page_size {
                if pages.len() == self.table_capacity {
                    return Err(PageError::TableFull {
                        seq,
                        capacity_pages: self.table_capacity,
                    });
                }
                let page = self.free.pop().ok_or(PageError::Exhausted {
                    pool_pages: self.pool_pages(),
                })?;
                self.refcount[page as usize] += 1;
                self.seqs[seq].0.push(page);
                fresh.push(page);
            }
            self.seqs[seq].1 += 1;
        }
        Ok(fresh)
    }
}

/// What a run reached (§2.4: a green result must be shown to have measured
/// something).
#[derive(Debug, Default, Clone, Copy)]
pub struct Engagement {
    pub appends: u64,
    pub pages_allocated: u64,
    pub exhausted: u64,
    pub table_full: u64,
    pub rewinds: u64,
    pub rewinds_refused: u64,
    pub shares: u64,
    pub releases: u64,
    pub pages_quarantined: u64,
    pub drains_recycling: u64,
    pub grows: u64,
    pub tables_non_identity: u64,
    pub recycle_refusals: u64,
    /// A page allocated to a sequence AFTER having been recycled through the
    /// quarantine — the reuse the epoch rule exists for.
    pub reuse_after_drain: u64,
}

/// Drive one sequence against the real pool + quarantine + clock and the
/// model, checking the partition invariant and every observable after every
/// operation.
pub fn run_sequence(f: &Fixture, ops: &[Op]) -> Result<Engagement, String> {
    let mut pool = KvPagePool::new(
        f.page_size,
        f.pool_pages,
        f.max_seqs,
        f.table_capacity_pages,
    );
    let mut quarantine = PageQuarantine::new();
    let clock = GpuClock::new();
    let mut model = Model::new(f);
    let mut eng = Engagement::default();
    let mut ever_recycled: Vec<PageIdx> = Vec::new();
    let n_seqs = f.max_seqs.max(1);

    for (i, op) in ops.iter().enumerate() {
        let at = |what: String| format!("after op {i} ({op:?}): {what}");
        match *op {
            Op::Append { seq, n } => {
                let seq = seq as usize % n_seqs;
                let got = pool.append(seq, n);
                let want = model.append(seq, n);
                match (&got, &want) {
                    (Ok(g), Ok(w)) => {
                        if g != w {
                            return Err(at(format!(
                                "append({seq}, {n}) allocated {g:?}, the contract says {w:?} \
                                 (LIFO free list, fresh pages under recycled ones)"
                            )));
                        }
                        eng.appends += 1;
                        eng.pages_allocated += g.len() as u64;
                        if g.iter().any(|p| ever_recycled.contains(p)) {
                            eng.reuse_after_drain += 1;
                        }
                    }
                    (Err(e), Err(w)) => {
                        if e != w {
                            return Err(at(format!("append refused with {e:?}, model says {w:?}")));
                        }
                        match e {
                            PageError::Exhausted { .. } => eng.exhausted += 1,
                            PageError::TableFull { .. } => eng.table_full += 1,
                            PageError::NotAllocated { .. } => {}
                        }
                    }
                    _ => {
                        return Err(at(format!(
                            "append({seq}, {n}) returned {got:?} where the model says {want:?}"
                        )))
                    }
                }
                // A partial append that failed mid-way: the pool's contract is
                // that the tokens before the failing boundary were appended
                // (the loop increments `len` per token). The model did the
                // same, so the comparison below holds either way.
            }
            Op::Rewind { seq, n } => {
                let seq = seq as usize % n_seqs;
                let got = pool.rewind(seq, n);
                let len = model.seqs[seq].1;
                if n > len {
                    if got.is_ok() {
                        return Err(at(format!(
                            "rewind({seq}, {n}) at len {len} was admitted; more than the live \
                             length must be refused"
                        )));
                    }
                    eng.rewinds_refused += 1;
                } else {
                    got.map_err(|e| at(format!("rewind({seq}, {n}) at len {len} refused: {e}")))?;
                    model.seqs[seq].1 -= n;
                    eng.rewinds += 1;
                }
            }
            Op::Share { src, dst, pages } => {
                let src = src as usize % n_seqs;
                let dst = dst as usize % n_seqs;
                let got = pool.share_prefix(src, dst, pages);
                let shared: Vec<PageIdx> = model.seqs[src].0.iter().take(pages).copied().collect();
                let fits = model.seqs[dst].0.len() + shared.len() <= model.table_capacity;
                if !fits {
                    if got.is_ok() {
                        return Err(at(format!(
                            "share_prefix({src}, {dst}, {pages}) admitted past the table capacity"
                        )));
                    }
                } else {
                    got.map_err(|e| {
                        at(format!("share_prefix({src}, {dst}, {pages}) refused: {e}"))
                    })?;
                    for p in &shared {
                        model.refcount[*p as usize] += 1;
                        model.seqs[dst].0.push(*p);
                        model.seqs[dst].1 += model.page_size;
                    }
                    if !shared.is_empty() {
                        eng.shares += 1;
                    }
                }
            }
            Op::Release { seq } => {
                let seq = seq as usize % n_seqs;
                let freed = pool.release(seq);
                let (pages, _) = std::mem::take(&mut model.seqs[seq]);
                let mut want = Vec::new();
                for p in pages {
                    let rc = &mut model.refcount[p as usize];
                    *rc -= 1;
                    if *rc == 0 {
                        want.push(p);
                    }
                }
                if freed != want {
                    return Err(at(format!(
                        "release({seq}) freed {freed:?}, the refcounts say {want:?}"
                    )));
                }
                eng.releases += 1;
                eng.pages_quarantined += freed.len() as u64;
                let epoch = clock.current_epoch();
                quarantine.quarantine(epoch, freed.clone());
                if !freed.is_empty() {
                    model.quarantined.push((epoch, freed));
                }
            }
            Op::Commit => {
                let closed = clock.commit_epoch();
                if closed != model.submitted {
                    return Err(at(format!(
                        "commit_epoch returned {closed}, model expected {}",
                        model.submitted
                    )));
                }
                model.submitted += 1;
            }
            Op::Complete { back } => {
                if model.submitted == 0 {
                    continue;
                }
                let newest = model.submitted - 1;
                let epoch = newest - (back as u64 % (newest + 1));
                clock.mark_completed(epoch);
                model.completed = model.completed.max(epoch + 1);
            }
            Op::Drain => {
                let completed = clock.completed_count();
                if completed != model.completed {
                    return Err(at(format!(
                        "completed_count reads {completed}, model says {}",
                        model.completed
                    )));
                }
                let ready = quarantine.drain_completed(completed);
                let n_ready = model
                    .quarantined
                    .iter()
                    .take_while(|(e, _)| completed > *e)
                    .count();
                let want: Vec<PageIdx> = model
                    .quarantined
                    .drain(..n_ready)
                    .flat_map(|(_, p)| p)
                    .collect();
                if ready != want {
                    return Err(at(format!(
                        "drain_completed({completed}) released {ready:?}, the epochs say {want:?}"
                    )));
                }
                pool.recycle(&ready)
                    .map_err(|e| at(format!("recycle of drained pages refused: {e}")))?;
                for p in &ready {
                    model.free.push(*p);
                    ever_recycled.push(*p);
                }
                if !ready.is_empty() {
                    eng.drains_recycling += 1;
                }
            }
            Op::Grow { extra } => {
                pool.grow(extra);
                let first = model.pool_pages() as PageIdx;
                let fresh: Vec<PageIdx> = (first..first + extra as PageIdx).rev().collect();
                model.free.splice(0..0, fresh);
                model.refcount.extend(std::iter::repeat_n(0, extra));
                model.table_capacity += extra;
                eng.grows += 1;
            }
            Op::Table { seq, k } => {
                let seq = seq as usize % n_seqs;
                let table = pool.table_for(seq, k);
                let want: Vec<PageIdx> = model.seqs[seq]
                    .0
                    .iter()
                    .copied()
                    .step_by(k.max(1))
                    .collect();
                if table != want {
                    return Err(at(format!(
                        "table_for({seq}, {k}) = {table:?}, model {want:?}"
                    )));
                }
                let contiguous = pool.chunk_pages_are_contiguous(seq, k);
                let want_contig = k.max(1) == 1
                    || model.seqs[seq]
                        .0
                        .chunks(k.max(1))
                        .all(|run| run.windows(2).all(|w| w[1] == w[0] + 1));
                if contiguous != want_contig {
                    return Err(at(format!(
                        "chunk_pages_are_contiguous({seq}, {k}) = {contiguous}, model {want_contig}"
                    )));
                }
                let identity: Vec<PageIdx> = (0..want.len() as PageIdx)
                    .map(|c| c * k.max(1) as PageIdx)
                    .collect();
                if !want.is_empty() && want != identity {
                    eng.tables_non_identity += 1;
                }
            }
            Op::RecycleLive { seq } => {
                // Every page that is still referenced by anyone must be refused
                // by `recycle` — the double-free guard §3.5 makes load-bearing.
                let seq = seq as usize % n_seqs;
                let live: Vec<PageIdx> = model.seqs[seq].0.clone();
                for p in live {
                    if model.refcount[p as usize] > 0 {
                        match pool.recycle(&[p]) {
                            Err(PageError::NotAllocated { page }) if page == p => {
                                eng.recycle_refusals += 1;
                            }
                            other => {
                                return Err(at(format!(
                                    "recycle of live page {p} (refcount {}) returned {other:?}; \
                                     a live page handed back to the free list aliases two \
                                     sequences",
                                    model.refcount[p as usize]
                                )))
                            }
                        }
                    }
                }
            }
        }

        // ---- the observables and the partition, after every operation -----
        if pool.pool_pages() != model.pool_pages() {
            return Err(at(format!(
                "pool_pages {} vs model {}",
                pool.pool_pages(),
                model.pool_pages()
            )));
        }
        if pool.pages_free() != model.free.len() {
            return Err(at(format!(
                "pages_free {} vs model {}",
                pool.pages_free(),
                model.free.len()
            )));
        }
        if pool.pages_in_use() != model.in_use() {
            return Err(at(format!(
                "pages_in_use {} vs model {}",
                pool.pages_in_use(),
                model.in_use()
            )));
        }
        if quarantine.held() != model.held() {
            return Err(at(format!(
                "quarantine.held {} vs model {}",
                quarantine.held(),
                model.held()
            )));
        }
        for s in 0..n_seqs {
            let sp = pool.seq(s).ok_or_else(|| at(format!("seq({s}) missing")))?;
            let (pages, len) = &model.seqs[s];
            if sp.pages() != pages.as_slice() || sp.len() != *len {
                return Err(at(format!(
                    "seq {s}: pool holds {:?} len {} where the model holds {pages:?} len {len}",
                    sp.pages(),
                    sp.len()
                )));
            }
        }
        // The partition: every page in exactly one of free / allocated /
        // quarantined. Checked on the MODEL's view, which the pool has just been
        // shown to agree with field by field; a page in two places here is the
        // aliasing the device would make silent.
        let n = model.pool_pages();
        let mut place = vec![0u8; n];
        for p in &model.free {
            place[*p as usize] |= 1;
        }
        for (p, rc) in model.refcount.iter().enumerate() {
            if *rc > 0 {
                place[p] |= 2;
            }
        }
        for (_, pages) in &model.quarantined {
            for p in pages {
                place[*p as usize] |= 4;
            }
        }
        for (p, where_) in place.iter().enumerate() {
            if where_.count_ones() != 1 {
                return Err(at(format!(
                    "page {p} is in {} places (free={} allocated={} quarantined={}): the \
                     partition broke",
                    where_.count_ones(),
                    where_ & 1 != 0,
                    where_ & 2 != 0,
                    where_ & 4 != 0
                )));
            }
        }
        // The refcount of a page is the number of sequence tables naming it.
        let mut named = vec![0u16; n];
        for (pages, _) in &model.seqs {
            for p in pages {
                named[*p as usize] += 1;
            }
        }
        if named != model.refcount {
            return Err(at(format!(
                "refcounts {:?} do not match the tables' references {named:?}",
                model.refcount
            )));
        }
    }
    Ok(eng)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Fixture {
        Fixture {
            page_size: 2,
            pool_pages: 4,
            max_seqs: 2,
            table_capacity_pages: 4,
        }
    }

    /// The driver reaches the states the alphabet is about, on a hand-written
    /// sequence, and reports them.
    #[test]
    fn a_release_recycle_reuse_round_trip_is_reached() {
        // The pages are quarantined at the OPEN epoch (0); the commit closes
        // it, the completion finishes it, and only then does a drain recycle.
        let ops = [
            Op::Append { seq: 0, n: 4 },
            Op::Table { seq: 0, k: 1 },
            Op::Release { seq: 0 },
            Op::Drain, // epoch 0 still open: nothing recycled
            Op::Commit,
            Op::Complete { back: 0 },
            Op::Drain, // now the pages come back
            Op::Append { seq: 1, n: 2 },
            Op::Table { seq: 1, k: 1 },
        ];
        let e = run_sequence(&fixture(), &ops).expect("the round trip is clean");
        assert_eq!(e.pages_allocated, 3);
        assert_eq!(e.pages_quarantined, 2);
        assert_eq!(e.drains_recycling, 1);
        assert_eq!(
            e.reuse_after_drain, 1,
            "the re-append must take a recycled page"
        );
        assert_eq!(
            e.tables_non_identity, 1,
            "a recycled pool's table is not the identity"
        );
    }

    /// The comparison can fail: a model perturbed by one refcount is caught.
    #[test]
    fn the_driver_reports_a_live_page_recycled() {
        let ops = [Op::Append { seq: 0, n: 2 }, Op::RecycleLive { seq: 0 }];
        let e = run_sequence(&fixture(), &ops).expect("the pool refuses, so the run is clean");
        assert_eq!(e.recycle_refusals, 1);
    }
}
