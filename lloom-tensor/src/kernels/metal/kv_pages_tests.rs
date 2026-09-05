//! The paged KV pool, tested without a device (`DESIGN.md` §10.3c–§10.3g, #148).
//!
//! Every test here is a pure function of an allocation history. §10.3i's
//! sequencing argument is the reason they exist before a page-walking kernel
//! does: *"a kernel tested against a contiguously-allocated pool is a kernel
//! tested at `page_table[c] == c`"*, so the allocator has to be able to produce
//! a table that is **not** the identity before the kernel that reads one is
//! worth writing.

use super::kv_pages::{reserve_pages_for, KvPagePool, PageError, PageQuarantine};

/// A pool sized so the tests below can exhaust it deliberately.
fn pool(pool_pages: usize) -> KvPagePool {
    KvPagePool::new(4, pool_pages, 4, 64)
}

/// A page is allocated only by the token that crosses a page boundary.
///
/// §10.3f: *"Every other token writes into a page that already exists, so the
/// steady-state per-token cost is **zero allocator work**"* — §6.7 L4b's
/// invariant. Asserted as the absolute page count after each append rather than
/// as a shape, per `CONTRIBUTING.md` §3.1b rule 1.
#[test]
fn only_the_boundary_token_allocates() {
    let mut p = pool(16);
    // Page size 4: tokens 1..4 live on one page, token 5 opens the second.
    let counts: Vec<usize> = (0..9)
        .map(|_| {
            let fresh = p.append(0, 1).unwrap();
            fresh.len()
        })
        .collect();
    assert_eq!(
        counts,
        vec![1, 0, 0, 0, 1, 0, 0, 0, 1],
        "a page must be allocated exactly at `len % page_size == 0` (DESIGN.md 10.3f)"
    );
    assert_eq!(p.seq(0).unwrap().len(), 9);
    assert_eq!(p.seq(0).unwrap().pages().len(), 3);
}

/// A fresh pool hands out the identity table, so a paged run and a contiguous
/// one are the same table.
///
/// **This is the control that makes the next test readable.** Without it, a
/// non-identity table would be consistent with the pool simply being disordered,
/// and "the table is permuted" would carry no information about *why*
/// (§3.1c: a zero is not evidence unless something nonzero came from the same
/// command).
#[test]
fn a_fresh_pool_produces_the_identity_table() {
    let mut p = pool(16);
    p.append(0, 12).unwrap();
    assert_eq!(
        p.table_for(0, 1),
        vec![0, 1, 2],
        "a fresh pool must reproduce ChunkTable::Contiguous exactly, so that a digest \
         difference between a paged and a contiguous run is attributable to something else"
    );
}

/// A recycled pool produces a table that is **not** the identity.
///
/// # This is the fixture §10.3h says a B=1 gate cannot produce
///
/// That section's hazard is that the combine must *index* the page table and
/// never *walk* it, and that the two are indistinguishable at
/// `page_table[c] == c`. `permuted_chunk_table_is_read_by_index_not_by_walk`
/// separates them with a hand-written `vec![1, 0]`; this shows the same shape
/// arises from **ordinary allocation and release**, which is what makes the
/// hazard reachable in production rather than only in a fixture.
///
/// The history is the one §10.3h describes: a sequence allocates, a *second*
/// sequence's frees return pages to the list, and the next allocation takes them
/// back in free-list order — so the table's contents depend on what another
/// sequence did, which is exactly the dependence §2.3.3 #7 forbids from reaching
/// a sequence's logits.
#[test]
fn a_recycled_pool_produces_a_non_identity_table() {
    let mut p = pool(8);
    p.append(0, 8).unwrap(); // pages 0,1
    p.append(1, 8).unwrap(); // pages 2,3
    assert_eq!(p.table_for(0, 1), vec![0, 1]);
    assert_eq!(p.table_for(1, 1), vec![2, 3]);

    // Sequence 0 goes away and its pages are recycled.
    let freed = p.release(0);
    assert_eq!(freed, vec![0, 1]);
    p.recycle(&freed).unwrap();

    // A third sequence now takes them back in free-list order (LIFO), so its
    // table is descending where a fresh pool's would ascend.
    p.append(2, 8).unwrap();
    let table = p.table_for(2, 1);
    assert_eq!(
        table,
        vec![1, 0],
        "a recycled pool must be able to produce a NON-identity table: this is the arm that \
         separates an indexed combine from a walked one (DESIGN.md 10.3h), and at \
         page_table[c] == c the two are indistinguishable"
    );
    assert_ne!(
        table,
        vec![0, 1],
        "the fixture is degenerate: if recycling reproduced the identity, the walk/index \
         distinction would be untestable here"
    );
}

/// Sharing a prefix stores it once, and the saving is unbounded in B.
///
/// §10.3i's third inversion threshold. Asserted as an absolute page count
/// against the un-shared cost, which is what makes it a measurement of the
/// mechanism rather than a restatement of it.
#[test]
fn a_shared_prefix_is_stored_once() {
    let mut p = KvPagePool::new(4, 32, 8, 64);
    p.append(0, 8).unwrap();
    let unshared_after_one = p.pages_in_use();
    assert_eq!(unshared_after_one, 2);

    // Three more sequences share the same two-page prefix.
    for dst in 1..4 {
        p.share_prefix(0, dst, 2).unwrap();
    }
    assert_eq!(
        p.pages_in_use(),
        2,
        "a prefix shared by 4 sequences must occupy the pages of ONE (DESIGN.md 10.3g)"
    );

    // The comparison that gives it meaning: four private copies would be four
    // times the pages.
    let mut private = KvPagePool::new(4, 32, 8, 64);
    for s in 0..4 {
        private.append(s, 8).unwrap();
    }
    assert_eq!(private.pages_in_use(), 8);
}

/// A shared page survives until its last referent drops it.
///
/// §10.3g rule 1: *"a refcounted page returns only when the last referent drops
/// it"* — the reason the free list cannot be a stack alone.
#[test]
fn a_shared_page_returns_only_on_the_last_release() {
    let mut p = KvPagePool::new(4, 16, 4, 64);
    p.append(0, 8).unwrap();
    p.share_prefix(0, 1, 2).unwrap();

    assert!(
        p.release(0).is_empty(),
        "releasing one of two referents must free NOTHING: a stack-only free list would hand \
         a live page back and alias two sequences onto it"
    );
    assert_eq!(p.pages_in_use(), 2);
    assert_eq!(
        p.release(1),
        vec![0, 1],
        "the last referent's release is what frees the pages"
    );
    assert_eq!(p.pages_in_use(), 0);
}

/// Exhaustion is a loud failure and never a silent grow.
///
/// §10.3f, following #150: growing the pool means reallocating the one buffer
/// every kernel addresses, moving the base pointer §9.2d's `base + offset`
/// reasoning rests on, at the moment the cache is largest.
#[test]
fn exhaustion_is_a_loud_failure() {
    let mut p = pool(2);
    p.append(0, 8).unwrap();
    let err = p.append(0, 1).unwrap_err();
    assert_eq!(err, PageError::Exhausted { pool_pages: 2 });
    // The message must name the knob, which is the half of #150's loud failure
    // that made it actionable rather than merely loud.
    assert!(err.to_string().contains("Raise the pool size"), "{err}");
}

/// Recycling a page that is still referenced is refused.
///
/// A double free would return a live page to the list and let a later allocation
/// alias two sequences onto it — silent under §3.5, so it is an error here.
#[test]
fn recycling_a_live_page_is_refused() {
    let mut p = pool(8);
    p.append(0, 4).unwrap();
    let err = p.recycle(&[0]).unwrap_err();
    assert_eq!(err, PageError::NotAllocated { page: 0 });
}

/// The saving paging buys at B>1, computed from an allocation history rather
/// than from the arithmetic.
///
/// §10.3i's **second** inversion threshold — *"the reserve is multiplied by B and
/// paging is not"* — is the one that belongs to #148. At B=8 holding 8 tokens
/// each, `Reserve@4096` holds `8 × 4096` tokens of pages and paging holds what
/// is live.
#[test]
fn paging_holds_what_is_live_where_reserve_holds_b_times_max_context() {
    let (page_size, max_context, b) = (256usize, 4096usize, 8usize);
    let mut p = KvPagePool::new(page_size, 4096, b, 64);
    for s in 0..b {
        p.append(s, 8).unwrap();
    }
    let paged = p.pages_in_use();
    let reserved = reserve_pages_for(b, max_context, page_size);
    assert_eq!(paged, 8, "one partial page per sequence");
    assert_eq!(reserved, 128, "16 pages of reserve per sequence, times 8");
    assert!(
        reserved / paged == 16,
        "at this shape Reserve holds 16x what paging does: {reserved} against {paged}"
    );
}

/// `k > 1` requires each chunk's pages to be consecutive, and a recycled pool
/// can violate it.
///
/// The kernel resolves one page per chunk and walks `k * page_size` tokens
/// linearly from it, so a chunk straddling a discontiguity reads the wrong keys.
/// Checked rather than assumed because §3.5 makes it silent.
#[test]
fn a_straddling_chunk_is_detectable_at_k_greater_than_one() {
    let mut p = pool(8);
    p.append(0, 8).unwrap(); // pages 0,1 -- consecutive
    assert!(p.chunk_pages_are_contiguous(0, 2));

    p.append(1, 8).unwrap(); // pages 2,3
    let freed = p.release(0);
    p.recycle(&freed).unwrap(); // free list: [.., 0, 1] -> pops 1 then 0
    p.append(2, 8).unwrap(); // pages 1,0 -- DESCENDING
    assert!(
        !p.chunk_pages_are_contiguous(2, 2),
        "a chunk spanning pages [1, 0] is not contiguous and the kernel would walk from page \
         1 into page 2's tokens: this must be detectable, since DESIGN.md 3.5 makes the wrong \
         read silent"
    );
    // At k = 1 a chunk is one page, so the constraint is vacuous by construction
    // -- asserted so the check above is known to be about `k` and not about the
    // pool being disordered.
    assert!(p.chunk_pages_are_contiguous(2, 1));
}

/// A rewind is a length decrement that keeps the pages, so the next append
/// writes into the page the discarded tokens occupied rather than allocating.
///
/// §10.2a's KV half of `resolve`: the discarded positions are above `len` and
/// unreachable. Asserted as absolute page counts (`CONTRIBUTING.md` §3.1b rule
/// 1): after rewinding 3 of 9 tokens the sequence still holds 3 pages, the
/// re-append of 3 allocates **nothing**, and a rewind past the start is refused.
#[test]
fn a_rewind_keeps_the_pages_and_the_re_append_allocates_nothing() {
    let mut p = pool(16);
    p.append(0, 9).unwrap(); // pages 0,1,2 at page size 4
    assert_eq!(p.seq(0).unwrap().pages().len(), 3);

    p.rewind(0, 3).unwrap();
    assert_eq!(p.seq(0).unwrap().len(), 6);
    assert_eq!(
        p.seq(0).unwrap().pages().len(),
        3,
        "a rewind must not free the page the discarded tokens sat in: the sequence \
         still owns the tokens below them, and a page released while a command \
         buffer reads it is the hazard the quarantine exists for"
    );
    assert_eq!(p.pages_free(), 13, "no page returned to the free list");

    let fresh = p.append(0, 3).unwrap();
    assert!(
        fresh.is_empty(),
        "re-appending into the kept page must allocate nothing, got {fresh:?}"
    );
    assert_eq!(p.seq(0).unwrap().len(), 9);
    assert_eq!(p.table_for(0, 1), vec![0, 1, 2]);

    assert!(
        p.rewind(0, 10).is_err(),
        "a rewind past the start is refused"
    );
}

// ---------------------------------------------------------------------------
// The GPU-liveness epoch (§10.3h's one non-fence obligation), #148.
// ---------------------------------------------------------------------------

/// A page held for an epoch that has not finished is NOT handed back.
///
/// This is the defect §10.3h names, in the smallest form that exhibits it:
/// sequence 0 is released while the command buffer that read its pages is still
/// in flight, and sequence 1 then asks for a page. Without the quarantine the
/// pool hands back the page the GPU is reading — **§6.7 L4 exactly, and the bug
/// #23 fixed in the buffer pool one layer up.**
///
/// The reference is absolute rather than a shape (`CONTRIBUTING.md` §3.1b rule
/// 1): the test names the page indices, so a defect that shifts *every* index
/// cannot pass the way an *all-different* invariant would.
#[test]
fn a_page_is_not_reallocated_while_its_last_reader_is_in_flight() {
    let mut p = pool(4);
    let mut q = PageQuarantine::new();

    // Sequence 0 takes pages 0 and 1; a command buffer in epoch 7 reads them.
    let taken = p.append(0, 8).unwrap();
    assert_eq!(taken, vec![0, 1], "a fresh pool hands out the identity");
    let reading_epoch = 7u64;

    // The CPU gives the sequence up. The refcount half says the pages are free.
    let freed = p.release(0);
    assert_eq!(freed, vec![0, 1], "both pages lost their last CPU referent");

    // The epoch half says they are not yet SAFE, which is the distinction the
    // whole mechanism exists to make.
    q.quarantine(reading_epoch, freed);
    assert_eq!(q.held(), 2);

    // Epoch 7 has not completed: `completed_count` counts FINISHED epochs, so
    // 7 means epochs 0..=6 are done and 7 is still running (GpuClock's own
    // off-by-one, inherited rather than re-derived).
    assert_eq!(
        q.drain_completed(7),
        Vec::<u32>::new(),
        "the reader is still in flight; nothing may be recycled"
    );
    assert_eq!(q.held(), 2, "and the pages are still held");

    // So the next allocation must NOT be page 0 or 1 -- it comes from the pages
    // the pool never handed out.
    let next = p.append(1, 8).unwrap();
    assert_eq!(
        next,
        vec![2, 3],
        "sequence 1 must be served from untouched pages while 0 and 1 are in \
         flight; serving [0, 1] here is the write-after-read across sequences \
         DESIGN.md 10.3h forbids"
    );

    // Once epoch 7 finishes, the pages become eligible and the pool takes them.
    let ready = q.drain_completed(8);
    assert_eq!(ready, vec![0, 1]);
    assert_eq!(q.held(), 0);
    p.recycle(&ready).unwrap();
    assert_eq!(p.pages_free(), 2, "both are back on the free list");
}

/// The unguarded path is still expressible, and it produces the defect.
///
/// The counterpart to the test above, and the reason it is not redundant with
/// it: a test showing the safe path is safe cannot show that the unsafe one is
/// unsafe, and §3.1b rule 2 asks for the second arm that pins the level. Here
/// the two arms differ in **one call** and disagree about which page sequence 1
/// gets, which is what makes the quarantine's contribution visible rather than
/// assumed.
#[test]
fn recycling_without_the_epoch_hands_back_a_page_the_gpu_is_reading() {
    let mut p = pool(4);
    p.append(0, 8).unwrap();
    let freed = p.release(0);

    // The shortcut DESIGN.md 10.3h names: refcount hit zero, so back it goes.
    p.recycle(&freed).unwrap();

    let next = p.append(1, 8).unwrap();
    assert_eq!(
        next,
        vec![1, 0],
        "this is the DEFECT, pinned so the quarantine's contribution is a \
         measured difference rather than a claim: sequence 1 is handed the very \
         pages an in-flight command buffer is reading"
    );
}

/// Epochs drain in order, and a later-quarantined batch does not overtake an
/// earlier one.
///
/// The prefix walk in `drain_completed` is what keeps this off §15.2 #10's
/// O(N)-per-token list; this pins that it is also *correct*, since a walk that
/// stopped at the first unfinished entry would be wrong if the epochs were not
/// non-decreasing.
#[test]
fn the_quarantine_drains_in_epoch_order_and_stops_at_the_first_unfinished() {
    let mut q = PageQuarantine::new();
    q.quarantine(1, vec![10, 11]);
    q.quarantine(2, vec![20]);
    q.quarantine(5, vec![50, 51]);
    assert_eq!(q.held(), 5);

    // Epoch 1 finished (count 2 == epochs 0 and 1 done), 2 has not.
    assert_eq!(q.drain_completed(2), vec![10, 11]);
    assert_eq!(q.held(), 3, "epoch 2's and epoch 5's pages stay");

    // Epoch 2 finished; epoch 5 has not, so the walk must stop rather than
    // sweeping the whole list.
    assert_eq!(q.drain_completed(3), vec![20]);
    assert_eq!(q.held(), 2);

    assert_eq!(q.drain_completed(6), vec![50, 51]);
    assert!(q.is_empty());
}

/// An empty release costs nothing, so a step that frees no page records no
/// entry.
///
/// Pinned because the alternative -- pushing an empty `Vec` per step -- grows
/// `pending` once per token forever, which is exactly the
/// climbs-without-plateauing shape §3.4a-iv records this project having missed
/// three times.
#[test]
fn quarantining_nothing_records_nothing() {
    let mut q = PageQuarantine::new();
    for epoch in 0..1000 {
        q.quarantine(epoch, Vec::new());
    }
    assert!(
        q.is_empty(),
        "1000 empty steps must leave the quarantine empty"
    );
    assert_eq!(q.held(), 0);
}
