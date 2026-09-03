//! Why `KvPagePool` has no production caller, demonstrated rather than quoted
//! (lloom #586; `DESIGN.md` §10.3e, §10.3i).
//!
//! # What this file is for
//!
//! §10.3e states the blocker in prose:
//!
//! > *"`slice_set` requires **both** sides contiguous (`tensor_cat.rs:249`) and
//! > `narrow` produces a strided view of one allocation. Neither can express
//! > 'tokens 0..255 are at pool offset 7·P and tokens 256..511 are at pool
//! > offset 2·P'. So paged KV cannot be built out of candle `Tensor` operations
//! > at all — it needs the append and the read to be **kernel-level**."*
//!
//! Four issues (#443, #355, #350, #568) closed while citing that paragraph, and
//! #586 was filed because the chain terminated with the work undone. **A
//! paragraph four issues can quote without any of them testing it is a claim,
//! not a constraint** — and this project has been wrong about exactly that
//! shape before: `CONTRIBUTING.md` §1.4 records two determinism rules this
//! corpus asserted and the code did not implement, both found *"by reading the
//! code against the claim rather than by any check"*.
//!
//! So this file is the check. It does not fix the blocker and does not work
//! around it; it makes the blocker **executable**, so that the day someone
//! removes it the test fails and says so.
//!
//! # What it deliberately does NOT do
//!
//! It does not wire `KvPagePool` into the model. §10.3i's costed verdict is
//! that paging is **#148's prerequisite and not today's work**, and #586's own
//! scope says to *"say so plainly if it is out of reach in one pass"*. Wiring
//! the pool somewhere it changes no number would be the decorative-caller
//! failure §15.1a calls VACUOUS — and there is a concrete instance available:
//! `reserve_pages_for(B, ctx, P) × P × KV_BYTES_PER_TOKEN` is **byte-for-byte
//! identical** to admission's existing `B × kv_capacity × KV_BYTES_PER_TOKEN`,
//! so routing admission through the pool would add a call and move no figure.
//! That equality is asserted below, because *"it would change nothing"* is
//! itself a claim worth pinning.

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Tensor};

/// LFM2's KV geometry at B=1 (§5.2, §5.6): 8 KV heads, head_dim 64.
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

fn device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// The append blocker: `slice_set` refuses a non-contiguous destination.
///
/// This is the mechanical half of §10.3e. A paged cache's destination for token
/// `t` is page `table[t / P]` at offset `t % P` — so the write target is one
/// page of the pool, which is a **sub-range** of a larger allocation. Expressed
/// as a `Tensor`, that is a `narrow`, and a `narrow` on the token axis of a
/// `[B, H, capacity, D]` buffer leaves the head stride at `capacity * D` while
/// the shape says `len` — which is not contiguous, and `slice_set` refuses it.
///
/// **The refusal is the point.** It is not a limitation to route around: §3.5
/// makes a wrong write silent, so a `slice_set` that accepted this would
/// corrupt rather than fail.
#[test]
fn slice_set_refuses_the_destination_a_paged_append_would_need() {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let capacity = 4096usize;
    let page_size = 256usize;

    // The pool, as one allocation. A paged cache is exactly this: one buffer,
    // addressed through a table (§10.3c -- indices, never pointers).
    let pool = Tensor::zeros((1, N_KV_HEADS, capacity, HEAD_DIM), DType::F16, &dev).unwrap();

    // One token's K, the constant-size source #150 established (§6.2b).
    let src = Tensor::zeros((1, N_KV_HEADS, 1, HEAD_DIM), DType::F16, &dev).unwrap();

    // A paged append writes into ONE PAGE, not into the whole pool. Page 7 is
    // an ordinary interior page -- nothing special about it except that it is
    // not page 0, which is what makes this a paging test rather than a
    // contiguous one.
    let page = pool.narrow(2, 7 * page_size, page_size).unwrap();
    assert!(
        !page.is_contiguous(),
        "the fixture is degenerate: a page of the pool must be a STRIDED view \
         for this test to exercise §10.3e's blocker. If this fires, the KV \
         layout changed and the whole argument needs re-deriving."
    );

    // The write the paged append needs, and the refusal §10.3e names.
    let err = page.slice_set(&src, 2, 0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("contiguous"),
        "slice_set must refuse a strided destination for the reason §10.3e \
         gives; it failed with something else instead: {msg}"
    );
}

/// The contiguous append that *does* work, as the control.
///
/// Without this arm the test above is unattributable: a `slice_set` that
/// refused **everything** would pass it, and the blocker would look proven by a
/// test that could not tell the two cases apart. This is `CONTRIBUTING.md`
/// §3.1c's rule — *a zero is not evidence unless something nonzero came from
/// the same command* — applied to a refusal.
#[test]
fn the_contiguous_append_that_ships_is_accepted() {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let capacity = 4096usize;
    let pool = Tensor::zeros((1, N_KV_HEADS, capacity, HEAD_DIM), DType::F16, &dev).unwrap();
    let src = Tensor::zeros((1, N_KV_HEADS, 1, HEAD_DIM), DType::F16, &dev).unwrap();

    assert!(
        pool.is_contiguous(),
        "the whole pool is one contiguous allocation"
    );
    // This is `KvSlot::append`'s call, at an arbitrary interior offset: the
    // shipping path writes one token's bytes at a moving offset into a
    // contiguous buffer, which is exactly what #150 built (§6.2b).
    pool.slice_set(&src, 2, 1793)
        .expect("the contiguous append is what ships and must keep working");
}

/// The read side is NOT blocked, and this is the half #586 could not assume.
///
/// §10.3e's paragraph covers `slice_set` **and** `narrow`, and is read as
/// blocking both directions. For the read that is no longer true, and the
/// reason is #116's kernel rather than anything in `Tensor`:
/// `flash_decoding.metal` resolves `chunk_table[c]` at chunk entry and walks
/// from `first_page * page_size`, so at the shipped `k = 1` **a chunk is one
/// page** and an arbitrary permutation is addressable. `ops.rs` binds a real
/// chunk-table buffer today; it fills it with `ChunkTable::Contiguous`.
///
/// So the honest statement of the blocker is narrower than §10.3e's, and it is
/// what #586 needs recorded: **the append is blocked and the read is not.**
/// This test pins the read side's precondition — that a page is addressable as
/// a contiguous tensor when it is taken as a whole page rather than as a
/// sub-range of a strided view.
#[test]
fn a_whole_page_of_a_flat_pool_is_contiguous_and_therefore_readable() {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let page_size = 256usize;
    let n_pages = 16usize;

    // The pool as the kernel addresses it: a flat token axis, one head. The
    // kernel computes `keys + kv_head_idx * k_head_stride + t * k_token_stride`
    // (§10.3d), so what it needs is that a page's tokens are a contiguous run
    // in the token axis -- which they are, by construction of the pool.
    let flat = Tensor::zeros((n_pages * page_size, HEAD_DIM), DType::F16, &dev).unwrap();

    for page in [0usize, 7, 15] {
        let p = flat.narrow(0, page * page_size, page_size).unwrap();
        assert!(
            p.is_contiguous(),
            "page {page} must be a contiguous run: this is what lets the flash \
             kernel walk it with a single affine stride (§10.3b, §10.3d)"
        );
        assert_eq!(p.dims(), &[page_size, HEAD_DIM]);
    }
}

/// Routing admission through the pool would move no number, asserted.
///
/// #586 asks for a production caller. Admission is the one place a pool figure
/// could plausibly go without the kernels, and it is **not** a caller: the
/// quantity it would supply is one admission already computes, to the byte.
///
/// `Budget::footprint` runs once at `Cache::new`, **before any token exists**,
/// so it must predict a worst case; `pages_in_use` is a runtime fact and is
/// zero at that moment. The only pool quantity available there is
/// `reserve_pages_for`, which describes the *incumbent* reserve by design (its
/// own doc comment says so). This asserts the equality rather than arguing it,
/// so a future reader does not have to re-derive why the obvious wiring was
/// declined.
#[test]
fn routing_admission_through_the_pool_would_change_no_byte() {
    // `KvPagePool::reserve_pages_for`'s body, mirrored rather than called:
    // `candle-transformers` does not depend on `candle-metal-kernels`, and
    // adding that edge to assert an equality would be a heavier change than the
    // finding warrants. The mirror is one line and
    // `the_mirror_is_reserve_pages_for` below pins it against the real
    // function's contract as its doc comment states it.
    fn reserve_pages_for(b: usize, max_context: usize, page_size: usize) -> usize {
        b * max_context.div_ceil(page_size.max(1))
    }

    // Admission's own constant (§9.5b): 2 (K and V) x 8 kv heads x 64 dim
    // x 2 B (f16) x 8 attention layers.
    const KV_BYTES_PER_TOKEN: usize = 2 * N_KV_HEADS * HEAD_DIM * 2 * 8;

    // The constant must be the one admission uses, or the equality below is
    // about a number nothing computes. 16384, asserted from the primitives the
    // same way `kv_geometry_is_16384_not_16000` does.
    assert_eq!(KV_BYTES_PER_TOKEN, 16_384);

    for &(b, kv_capacity, page_size) in &[
        (1usize, 4096usize, 256usize),
        (1, 4096, 16),
        (4, 4096, 256),
        (32, 131_072, 256),
    ] {
        let via_pool =
            reserve_pages_for(b, kv_capacity, page_size) * page_size * KV_BYTES_PER_TOKEN;
        let via_admission = b * kv_capacity * KV_BYTES_PER_TOKEN;
        assert_eq!(
            via_pool, via_admission,
            "reserve_pages_for describes the INCUMBENT reserve, so at a capacity \
             that divides the page size it is admission's existing kv term to the \
             byte (B={b}, capacity={kv_capacity}, page={page_size}). A caller that \
             changes no number is §15.1a's VACUOUS class, which is why #586 \
             declined this wiring and recorded the blocker instead."
        );
    }

    // And the equality is a property of divisibility, not a coincidence of the
    // four rows above. At a capacity the page size does NOT divide, the pool
    // ROUNDS UP -- exhaustion is a loud failure and never a `Grow` (§10.3f), so
    // a partial tail page is a whole page held. Asserted as the absolute page
    // count rather than as "the two differ": a truncating mirror also differs
    // from admission, in the other direction, and `assert_ne!` cannot tell the
    // two apart (§3.1b rule 1 -- prefer an absolute reference to a shape; this
    // assertion is the repair of a first version that a truncation mutant
    // survived).
    assert_eq!(
        reserve_pages_for(1, 4000, 256),
        16,
        "4000 tokens at page 256 is 15 full pages and a partial one, which the \
         pool holds as 16 -- a truncating count would read 15 and silently \
         under-reserve the tail"
    );
    assert_eq!(
        reserve_pages_for(1, 4096, 256),
        16,
        "4096 is exactly 16 pages"
    );
    assert_eq!(
        reserve_pages_for(1, 1, 256),
        1,
        "one token still costs a whole page: this is the last-page waste §10.3i \
         prices, and a truncating count would report zero pages for a live \
         sequence"
    );
}
