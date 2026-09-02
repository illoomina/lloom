//! The paged KV pool, tested without a device (`DESIGN.md` §10.3c–§10.3g, #148).
//!
//! Every test here is a pure function of an allocation history. §10.3i's
//! sequencing argument is the reason they exist before a page-walking kernel
//! does: *"a kernel tested against a contiguously-allocated pool is a kernel
//! tested at `page_table[c] == c`"*, so the allocator has to be able to produce
//! a table that is **not** the identity before the kernel that reads one is
//! worth writing.

use super::kv_pages::{reserve_pages_for, KvPagePool, PageError};

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
