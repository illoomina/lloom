//! FlashDecoding on the device: parity, the permuted chunk table, and the
//! merge order (`DESIGN.md` §10.4, §10.3h, issue #116).
//!
//! # The three properties, and which one needs a fixture nothing else has
//!
//! 1. **Parity against a scalar reference.** `CONTRIBUTING.md` §3.1: every
//!    kernel owes a reference implementation and a parity test against it,
//!    mutation-tested. The reference here is a single-pass softmax computed in
//!    `f64` on the CPU, which is *not* the thing being changed — §2.3.5a's
//!    load-bearing discriminator, and the only check that asks whether the
//!    answer is *right* rather than merely stable.
//!
//! 2. **The combine INDEXES its chunk table and never WALKS it** (§10.3h,
//!    §3.7f). Both visit every chunk; only the first is bit-stable, because a
//!    table's contents depend on allocation history, which under B>1 depends on
//!    what other sequences did. **At B=1 the table is the identity and the two
//!    orders coincide**, so a B=1 gate structurally cannot tell them apart —
//!    which is why [`permuted_chunk_table_is_read_by_index_not_by_walk`] builds
//!    a table that is *not* the identity. #197's probe established that a
//!    permuted arm is what separates them, and that a fixture which does not
//!    permute is testing the identity function.
//!
//! 3. **Every declared `[[host_name]]` resolves against the compiled library**
//!    (§8.1b). Not against a second copy of the list: a generator proves two
//!    lists came from one source and says nothing about whether Metal compiles
//!    the name. This is also what proves `flash_decoding.metal` compiles at
//!    all, since the offline toolchain is not installed on this machine.

use crate::kernels::kernels::params::{FlashCombineParams, FlashPartialParams, FlashScatterParams};
use crate::kernels::kernels::{
    flash_chunk_count, ChunkTable, FlashDType, FlashKernel, FLASH_HEAD_DIMS,
};
use crate::kernels::metal::kv_pages::KvPagePool;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{call_flash_decoding, call_flash_kv_scatter, Device, Kernels, Source};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

fn read_f32(buf: &crate::kernels::Buffer, n: usize) -> Vec<f32> {
    let p = buf.contents() as *const f32;
    assert!(!p.is_null(), "buffer has no CPU mapping");
    // SAFETY: shared storage holding at least `n` f32, and the caller has
    // waited for the command buffer.
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn new_f32(device: &Device, values: &[f32]) -> crate::kernels::Buffer {
    let buf = device
        .new_buffer(std::mem::size_of_val(values).max(4), SHARED)
        .unwrap();
    let dst = buf.contents() as *mut f32;
    // SAFETY: freshly allocated shared storage of exactly this length.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len()) };
    buf
}

fn new_u32(device: &Device, values: &[u32]) -> crate::kernels::Buffer {
    let buf = device
        .new_buffer(std::mem::size_of_val(values).max(4), SHARED)
        .unwrap();
    let dst = buf.contents() as *mut u32;
    // SAFETY: as `new_f32`.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len()) };
    buf
}

fn write_struct<T>(device: &Device, value: T) -> crate::kernels::Buffer {
    let buf = device.new_buffer(std::mem::size_of::<T>(), SHARED).unwrap();
    let dst = buf.contents() as *mut T;
    // SAFETY: freshly allocated shared storage sized for exactly one `T`.
    unsafe { dst.write(value) };
    buf
}

/// The scalar reference: one-pass softmax attention in `f64`.
///
/// **Deliberately not the chunked formulation.** A reference that split the same
/// way the kernel does would agree with it for the same wrong reason; this
/// computes the whole thing in one pass at higher precision, so agreement is
/// evidence that the *split and merge* are right rather than that two
/// implementations of the split agree.
///
/// `keys`/`values` are `[n_kv_heads][capacity][head_dim]`, and only the first
/// `n_keys` tokens of each head are live — the reserved-capacity shape §6.2b's
/// pre-allocated cache produces.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    scale: f32,
) -> Vec<f32> {
    let gqa = n_q_heads / n_kv_heads;
    let mut out = vec![0f32; n_q_heads * head_dim];
    for h in 0..n_q_heads {
        let kv = h / gqa;
        let q = &queries[h * head_dim..(h + 1) * head_dim];

        let mut scores = Vec::with_capacity(n_keys);
        for t in 0..n_keys {
            let base = kv * capacity * head_dim + t * head_dim;
            let mut s = 0f64;
            for d in 0..head_dim {
                s += (q[d] as f64) * (keys[base + d] as f64);
            }
            scores.push(s * scale as f64);
        }

        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        for s in &scores {
            denom += (s - m).exp();
        }

        for d in 0..head_dim {
            let mut acc = 0f64;
            for (t, s) in scores.iter().enumerate() {
                let base = kv * capacity * head_dim + t * head_dim;
                acc += (s - m).exp() * values[base + d] as f64;
            }
            out[h * head_dim + d] = (acc / denom) as f32;
        }
    }
    out
}

/// A deterministic pseudo-random fixture. No `rand` dependency and no seed
/// plumbing: the values only have to be varied and reproducible.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// One end-to-end run at `Sizing::Grow` — the region sized for exactly the
/// chunks the step computes.
///
/// **What every arm did before #234**, and now said rather than implied: the
/// live count and the capacity are the same number here because that is what
/// `Grow` means, not because a caller had no second one to pass.
#[allow(clippy::too_many_arguments)]
fn run_flash(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    pages_per_chunk: usize,
    scale: f32,
    table: &ChunkTable,
) -> Vec<f32> {
    let n_chunks = flash_chunk_count(n_keys, page_size * pages_per_chunk);
    run_flash_sized(
        device,
        kernels,
        queries,
        keys,
        values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        pages_per_chunk,
        scale,
        table,
        n_chunks,
    )
}

/// One end-to-end run, returning the kernel's output.
///
/// `chunk_capacity` is what the scratch region is **sized** for and `n_chunks`
/// is what this step computes — the pair `ScratchSizing` decides (#234). They
/// are equal under `Grow` and differ under `Reserve` and `Bucket`, and the
/// difference is what `flash_reserving_capacity_agrees_with_grow` exercises:
/// without a capacity above the live count, every fixture runs at a stride the
/// two readings of that field agree on, and a stride bug is invisible.
#[allow(clippy::too_many_arguments)]
fn run_flash_sized(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    pages_per_chunk: usize,
    scale: f32,
    table: &ChunkTable,
    chunk_capacity: usize,
) -> Vec<f32> {
    let cmds = commands(device);
    let chunk_size = page_size * pages_per_chunk;
    let n_chunks = flash_chunk_count(n_keys, chunk_size);
    assert!(
        chunk_capacity >= n_chunks,
        "a region sized for {chunk_capacity} chunks cannot hold the {n_chunks} this step \
         computes; the partial pass would write past it"
    );

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    // Sized for the CAPACITY, exactly as `FlashDecoding::metal_fwd` sizes it,
    // so a reserving arm here allocates the region the kernel then strides by.
    let partials = device
        .new_buffer(n_q_heads * chunk_capacity * head_dim * 4, SHARED)
        .unwrap();
    let sums = device
        .new_buffer(n_q_heads * chunk_capacity * 4, SHARED)
        .unwrap();
    let maxs = device
        .new_buffer(n_q_heads * chunk_capacity * 4, SHARED)
        .unwrap();
    let table_buf = new_u32(device, &table.entries(n_chunks, pages_per_chunk));
    // The LIVE count and not the capacity: the combine walks `n_chunks` and
    // writes one entry per walked chunk, so a larger buffer would leave a tail
    // the merge-order assertion cannot interpret.
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let keys_buf = new_u32(device, &[n_keys as u32]);

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    // `[b, n_kv, capacity, head_dim]`: index 1 is the head stride (the RESERVED
    // capacity) and index 2 is the token stride. Those being different numbers
    // is what lets a pre-allocated cache be read without a copy (§10.3b).
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        FlashPartialParams::for_step(
            &q_shape,
            &k_shape,
            &k_stride,
            &k_stride,
            page_size,
            pages_per_chunk,
            n_chunks,
            chunk_capacity,
            0,
            scale,
            1.0,
        )
        .unwrap(),
    );
    let cp = write_struct(
        device,
        FlashCombineParams::for_step(n_chunks, chunk_capacity),
    );

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            &keys_buf,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    read_f32(&out, n_q_heads * head_dim)
}

/// **Every declared name resolves against the compiled library.**
///
/// §8.1b. This is also what proves `flash_decoding.metal` compiles at all: the
/// offline Metal toolchain is absent on this machine, so runtime compilation is
/// the only compiler available — which #8.1b notes *inverts the usual ranking*,
/// since the library a test interrogates is the one the GPU will be asked for.
#[test]
fn flash_names_resolve() {
    let device = device();
    let kernels = Kernels::new();
    for name in FlashKernel::all() {
        kernels
            .load_pipeline(&device, Source::FlashDecoding, name.clone())
            .unwrap_or_else(|e| {
                panic!("{name} does not resolve against flash_decoding.metal: {e}")
            });
    }
}

/// **A head dimension the file does not instantiate is refused, not resolved.**
///
/// The inverse of #103's finding, guarded from the other side: there, eight
/// head dimensions were instantiated and six reachable, so two were dead
/// instantiations paid for at every cold compile. Here the registry and the
/// instantiation list are one list, and asking for a dimension outside it is an
/// error rather than a `LoadFunctionError` from inside a forward pass.
#[test]
fn flash_refuses_an_uninstantiated_head_dim() {
    assert!(FlashKernel::partial(FlashDType::F16, 63).is_err());
    assert!(FlashKernel::combine(FlashDType::F16, 512).is_err());
    for hd in FLASH_HEAD_DIMS {
        assert!(FlashKernel::partial(FlashDType::F16, hd).is_ok());
    }
}

/// **Parity against the scalar reference, over several chunk counts.**
///
/// The chunk counts are chosen so that `n_keys` is *not* a multiple of the
/// chunk size in three of five cases: the last chunk is partial, and
/// `n_this_chunk` must walk the live count rather than the reserved one.
/// Merging over a full last chunk folds in uninitialised memory, which §9.1a
/// records as a silent wrong answer that no size check catches — so a fixture
/// whose `n_keys` always divides evenly would not exercise it.
#[test]
fn flash_matches_the_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 8;
    let n_kv_heads = 2; // GQA 4:1, as LFM2 is (§5.2)
    let capacity = 512;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // `n_keys` and the chunk size, chosen so the last chunk is partial in most.
    for &(n_keys, page_size) in &[(1usize, 32usize), (32, 32), (33, 32), (100, 32), (255, 64)] {
        let queries = fill(n_q_heads * head_dim, 1);
        let keys = fill(n_kv_heads * capacity * head_dim, 2);
        let values = fill(n_kv_heads * capacity * head_dim, 3);

        let got = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        let want = reference_attention(
            &queries, &keys, &values, n_q_heads, n_kv_heads, head_dim, capacity, n_keys, scale,
        );

        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w).abs());
        }
        assert!(
            worst < 2e-5,
            "n_keys={n_keys} page_size={page_size}: worst |delta| = {worst:e} against the f64 \
             reference; chunks={}",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **The answer does not depend on how the KV is split.**
///
/// One chunk, two chunks and eleven chunks over the *same* keys must agree to
/// float tolerance — which is the associativity property online softmax exists
/// to provide (§10.4, glossary), and the thing a wrong merge breaks.
///
/// §10.6 is the other half of it: *contiguous is paged with one page, and
/// single-pass attention is FlashDecoding with `n_chunks == 1`*. The
/// `page_size >= n_keys` arm below **is** that degenerate case, so it is tested
/// rather than argued.
#[test]
fn flash_is_invariant_to_the_chunk_count() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 512;
    let n_keys = 200;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 7);
    let keys = fill(n_kv_heads * capacity * head_dim, 8);
    let values = fill(n_kv_heads * capacity * head_dim, 9);

    let single = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        256,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    assert_eq!(
        flash_chunk_count(n_keys, 256),
        1,
        "the degenerate arm must be one chunk"
    );

    for &page_size in &[128usize, 64, 32] {
        let split = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        let mut worst = 0f32;
        for (a, b) in single.iter().zip(split.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst < 2e-5,
            "splitting into {} chunks moved the answer by {worst:e}",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **A region sized for more chunks than the step computes gives the same
/// bits.**
///
/// This is the arm `ScratchSizing`'s reserving policies produce (#234), and
/// **the only fixture in this file where `chunk_capacity != n_chunks`.** Every
/// other one runs at `Grow`, where the two readings of that field coincide —
/// so a partial pass striding by the live count instead of the capacity is
/// invisible to all of them. That is #116's own `k_token_stride` finding in a
/// second field: *a parameter nothing exercises is a parameter that can be
/// silently ignored*, and then the policies the axis exists to compare are not
/// buildable after all.
///
/// **Bit-identical and not merely close**, which is the required result rather
/// than a pleasant one: sizing decides *where bytes live* and not *what they
/// are* (§2.3.5a), so a moved answer here is a defect. The chunks between the
/// live count and the capacity are never dispatched, so they are never
/// written; the combine's loop bound is the live count, so it never reads
/// them. Both halves have to hold — writing at the wrong stride scatters the
/// partials, and merging to the capacity folds in uninitialised memory, which
/// §10.4 and §10.3d both record as a silent wrong answer no size check
/// catches.
#[test]
fn flash_reserving_capacity_is_bit_identical_to_grow() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    // More than one head: the capacity is the stride BETWEEN heads, so at
    // `n_q_heads == 1` there is no second head to land in the wrong place and
    // the fixture would be blind to exactly the defect it is written for.
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 512;
    let n_keys = 200;
    let page_size = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 41);
    let keys = fill(n_kv_heads * capacity * head_dim, 42);
    let values = fill(n_kv_heads * capacity * head_dim, 43);

    let live = flash_chunk_count(n_keys, page_size);
    assert_eq!(live, 7, "the fixture's live chunk count");

    let grow = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        1,
        scale,
        &ChunkTable::Contiguous,
    );

    // 16 is `Reserve`'s shape at this fixture's own capacity —
    // `ceil(512 / 32)` — and 9 is a rung that is neither the live count nor the
    // maximum, so the test does not accidentally depend on the reservation
    // being the largest number in sight.
    for &chunk_capacity in &[9usize, 16] {
        assert!(chunk_capacity > live, "the reserving arm must reserve more");
        let reserved = run_flash_sized(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
            chunk_capacity,
        );
        assert_eq!(
            grow, reserved,
            "a region sized for {chunk_capacity} chunks against the {live} this step computes \
             moved the answer. Sizing decides where bytes live, not what they are, so this is \
             a defect and not a variant (DESIGN.md 2.3.5a, 9.1a)"
        );
    }
}

/// **`k > 1` computes the same answer as `k = 1` at the same chunk size.**
///
/// `chunk_size = k * page_size` (§9.1d). `k = 2` at `page_size = 32` and `k = 1`
/// at `page_size = 64` are the same *computation* unit over different
/// *allocation* units, so they must agree — which is what makes `k` a real
/// parameter rather than a field nothing reads.
///
/// **This is the arm that would fail if `pages_per_chunk` were ignored.** §9.1d
/// warns that a sweep fixing `k = 1` cannot separate a page-size effect from a
/// tile-size one; carrying `k` is what makes that separable, and this test is
/// what says the field is live.
#[test]
fn flash_k_greater_than_one_agrees_with_k_one() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 256;
    let n_keys = 150;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 11);
    let keys = fill(n_kv_heads * capacity * head_dim, 12);
    let values = fill(n_kv_heads * capacity * head_dim, 13);

    // k = 1 at page 64: chunk_size 64.
    let k1 = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        64,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    // k = 2 at page 32: chunk_size 64, from two pages.
    let k2 = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        32,
        2,
        scale,
        &ChunkTable::Contiguous,
    );

    let mut worst = 0f32;
    for (a, b) in k1.iter().zip(k2.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst == 0.0,
        "k=2 x page 32 and k=1 x page 64 are the same chunking and must be bit-identical; \
         worst |delta| = {worst:e}"
    );
}

/// **The chunk table is read by INDEX and not by WALK, shown with a table that
/// is not the identity.**
///
/// This is the test §10.3h asks for and the one a B=1 gate cannot substitute
/// for. With `chunk_table[c] == c` an index and a walk are indistinguishable —
/// #197's probe had to add `page_table[3] = 6` for exactly this reason, and
/// records that *a fixture that does not permute its table is testing the
/// identity function*.
///
/// The construction: two chunks whose **pages are swapped** in the table. Chunk
/// 0 names page 1 and chunk 1 names page 0, so the kernel reads the second half
/// of the KV for chunk 0 and the first half for chunk 1. Attention over a set of
/// keys does not depend on the order the *set* is visited in — softmax is over
/// all of them — so the ANSWER must be unchanged, while a kernel that ignored
/// the table and walked `c * chunk_size` would also produce the unchanged
/// answer. **What separates them is where the partials land**, which is why the
/// assertion is on the partials rather than only on the output.
#[test]
fn permuted_chunk_table_is_read_by_index_not_by_walk() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 2;
    let n_kv_heads = 1;
    let capacity = 128;
    let page_size = 32;
    let n_keys = 64; // exactly two chunks
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 21);
    let keys = fill(n_kv_heads * capacity * head_dim, 22);
    let values = fill(n_kv_heads * capacity * head_dim, 23);

    let n_chunks = flash_chunk_count(n_keys, page_size);
    assert_eq!(n_chunks, 2);

    // The two arms differ ONLY in the table.
    let identity = run_partials(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        scale,
        &ChunkTable::Contiguous,
    );
    let permuted = run_partials(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        scale,
        &ChunkTable::Explicit(vec![1, 0]),
    );

    // Under a permuted table, chunk 0's partial must be what chunk 1's was, and
    // vice versa: the kernel resolved its range THROUGH the table.
    //
    // A kernel that ignored the table and walked `c * chunk_size` would produce
    // `identity` under both, so this comparison is capable of the other answer
    // -- which is what makes it a test rather than a display (§15.1 #1).
    let (id_maxs, perm_maxs) = (&identity.1, &permuted.1);
    assert_ne!(
        id_maxs[0], id_maxs[1],
        "the fixture is degenerate: the two chunks have equal maxima, so a swap is \
         unobservable and this test would pass under a walk"
    );
    for h in 0..n_q_heads {
        assert_eq!(
            perm_maxs[h * 2],
            id_maxs[h * 2 + 1],
            "head {h}: chunk 0 under the permuted table must read what chunk 1 read under the \
             identity -- it did not, so the table was WALKED rather than INDEXED"
        );
        assert_eq!(
            perm_maxs[h * 2 + 1],
            id_maxs[h * 2],
            "head {h}: chunk 1 under the permuted table must read what chunk 0 read under the \
             identity"
        );
    }
}

/// A permuted table with a PARTIAL last page reads the live count PER CHUNK,
/// so a recycled pool's table reproduces the contiguous answer bit for bit.
///
/// # The limit #586 recorded, and its repair
///
/// The kernel's clamp used to be `chunk_end = min(first_page * page_size +
/// chunk_size, n_keys)` — keyed on the **physical** position, which assumes
/// pages appear in ascending token order. A recycled pool does not honour
/// that: `KvPagePool` hands pages back in free-list order, so a chunk placed
/// above the tail page read a **truncated** range and a chunk placed below it
/// read past its live tokens. #586 pinned that as a boundary of the kernel's
/// correctness; this is the same fixture with the boundary removed, and the
/// assertion inverted: the live count is now `min(chunk_size, n_keys - c *
/// chunk_size)` — a property of the logical token axis — and the permuted arm
/// must agree with the contiguous one exactly.
///
/// The table is produced by an ordinary allocate / release / recycle history
/// rather than written as a literal (§10.3i's reason for building the
/// allocator first), and the KV is laid out **physically** through that table:
/// logical chunk `c` lives at page `table[c]`. So the two arms read the same
/// logical keys from different physical places, and a kernel that got any
/// chunk's extent wrong reads different keys — which the bit comparison sees.
#[test]
fn a_recycled_table_with_a_partial_last_page_reproduces_the_contiguous_bits() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 2;
    let page_size = 32;
    let pool_pages = 8;
    // 80 keys at page 32: three chunks, the third HALF full. The shape every
    // real paged cache has — a tail page mid-fill — and the one the identity
    // table makes indistinguishable from a physical clamp.
    let n_keys = 80;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // An ordinary history: sequence 0 takes three pages, sequence 1 one page,
    // sequence 0 leaves and its pages are recycled, sequence 2 takes them back
    // in free-list order.
    let mut pool = KvPagePool::new(page_size, pool_pages, 4, 8);
    pool.append(0, 3 * page_size).unwrap();
    pool.append(1, page_size).unwrap();
    let freed = pool.release(0);
    pool.recycle(&freed).unwrap();
    pool.append(2, n_keys).unwrap();
    let table = pool.table_for(2, 1);
    assert_eq!(
        table,
        vec![2, 1, 0],
        "the recycled history must yield a NON-identity table"
    );
    assert!(pool.chunk_pages_are_contiguous(2, 1));

    let queries = fill(n_q_heads * head_dim, 31);
    // The logical KV: token `t` of head `h` at `h * n_keys * D + t * D`.
    let logical_k = fill(n_kv_heads * n_keys * head_dim, 32);
    let logical_v = fill(n_kv_heads * n_keys * head_dim, 33);

    // Contiguous arm: the logical layout, reserved capacity 256.
    let capacity = 256;
    let (contig_k, contig_v) = lay_out_contiguous(
        &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
    );
    let n_chunks = flash_chunk_count(n_keys, page_size);
    let contig = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &contig_k,
        &contig_v,
        FlashGeometry {
            n_seqs: 1,
            heads_per_seq: n_q_heads,
            n_kv_heads,
            head_dim,
            n_keys,
            page_size,
            pages_per_chunk: 1,
            k_stride: [
                n_kv_heads * capacity * head_dim,
                capacity * head_dim,
                head_dim,
                1,
            ],
            table: ChunkTable::Contiguous.entries(n_chunks, 1),
            table_stride: 0,
        },
        scale,
    );

    // Paged arm: one pool of `pool_pages` pages per head, logical chunk `c`
    // stored at page `table[c]`.
    let (paged_k, paged_v) = lay_out_paged(
        &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, page_size, pool_pages, &table,
    );
    let pool_geometry = |table: Vec<u32>| FlashGeometry {
        n_seqs: 1,
        heads_per_seq: n_q_heads,
        n_kv_heads,
        head_dim,
        n_keys,
        page_size,
        pages_per_chunk: 1,
        // A pool has no sequence stride; the head stride spans every page.
        k_stride: [0, pool_pages * page_size * head_dim, head_dim, 1],
        table,
        table_stride: 0,
    };
    let paged = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &paged_k,
        &paged_v,
        pool_geometry(table.clone()),
        scale,
    );

    assert_eq!(
        contig.out, paged.out,
        "a recycled table [2, 1, 0] with a half-full last page must reproduce the contiguous \
         answer BIT FOR BIT: the live count of chunk c is min(chunk_size, n_keys - c * \
         chunk_size), a property of the token axis, not of where the page sits (DESIGN.md \
         10.3d, 10.3h)"
    );
    // Per chunk, not only in the merged output: chunk `c` must have read the
    // same 32 (or 16) keys under both arms.
    assert_eq!(
        contig.maxs, paged.maxs,
        "every chunk's maximum must agree, chunk for chunk"
    );
    assert_eq!(
        contig.sums, paged.sums,
        "every chunk's denominator must agree, chunk for chunk"
    );
    for h in 0..n_q_heads {
        assert_eq!(
            &paged.walk[h * n_chunks..(h + 1) * n_chunks],
            &(0..n_chunks as u32).collect::<Vec<_>>()[..],
            "head {h}: the combine must walk chunk INDICES ascending whatever the table holds"
        );
    }

    // The fixture is capable of the other answer: the same pool read through
    // the identity table — a kernel that ignored its table — is a different
    // answer, because the pages are genuinely permuted (§15.1 #1).
    let walked = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &paged_k,
        &paged_v,
        pool_geometry(ChunkTable::Contiguous.entries(n_chunks, 1)),
        scale,
    );
    assert_ne!(
        contig.out, walked.out,
        "degenerate fixture: reading the permuted pool through the identity table gave the \
         contiguous answer, so the table is not being exercised"
    );
}

/// The single-chunk path reads the page its table names, not page 0.
///
/// `flash_decoding_single` is dispatched at `n_chunks == 1` — the common decode
/// band — and used to read `[0, n_keys)` on the ground that *"at one chunk the
/// table is the identity"*. True of a contiguous cache and of a fresh pool;
/// false of a recycled one, where the single page a short sequence occupies is
/// whatever the free list handed back. A B=1 fresh-pool gate cannot see it
/// (§10.3h): the arm below puts the data at page 3 and asks for it.
#[test]
fn the_single_chunk_path_reads_the_page_its_table_names() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 2;
    let n_kv_heads = 1;
    let page_size = 32;
    let pool_pages = 4;
    let n_keys = 20;
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(
        flash_chunk_count(n_keys, page_size),
        1,
        "the fixture must take the single path"
    );

    let queries = fill(n_q_heads * head_dim, 61);
    let logical_k = fill(n_kv_heads * n_keys * head_dim, 62);
    let logical_v = fill(n_kv_heads * n_keys * head_dim, 63);

    let capacity = 64;
    let (contig_k, contig_v) = lay_out_contiguous(
        &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
    );
    let contig = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &contig_k,
        &contig_v,
        FlashGeometry {
            n_seqs: 1,
            heads_per_seq: n_q_heads,
            n_kv_heads,
            head_dim,
            n_keys,
            page_size,
            pages_per_chunk: 1,
            k_stride: [
                n_kv_heads * capacity * head_dim,
                capacity * head_dim,
                head_dim,
                1,
            ],
            table: vec![0],
            table_stride: 0,
        },
        scale,
    );

    let table = vec![3u32];
    let (paged_k, paged_v) = lay_out_paged(
        &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, page_size, pool_pages, &table,
    );
    let geometry = |table: Vec<u32>| FlashGeometry {
        n_seqs: 1,
        heads_per_seq: n_q_heads,
        n_kv_heads,
        head_dim,
        n_keys,
        page_size,
        pages_per_chunk: 1,
        k_stride: [0, pool_pages * page_size * head_dim, head_dim, 1],
        table,
        table_stride: 0,
    };
    let paged = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &paged_k,
        &paged_v,
        geometry(table),
        scale,
    );
    assert_eq!(
        contig.out, paged.out,
        "the single-chunk path must read from page 3, where the table put the sequence, and \
         reproduce the contiguous bits"
    );
    let page_zero = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &paged_k,
        &paged_v,
        geometry(vec![0]),
        scale,
    );
    assert_ne!(
        contig.out, page_zero.out,
        "degenerate fixture: page 0 of the pool gave the right answer, so the table is not \
         being exercised"
    );
}

/// B sequences read ONE pool through B tables, and each row reproduces its own
/// B=1 answer bit for bit.
///
/// The batch is reached through `table_stride` and a sequence stride of 0,
/// never by folding the batch into the head index — which is address-identical
/// for a `[B, H_kv, T, D]` cache and wrong for a pool, where row `b`'s KV is at
/// whatever pages its table names. Two sequences with DIFFERENT keys, queries
/// and page sets, so a row reading its neighbour's pages is a different answer
/// (§2.3.3 #7: a sequence's logits must not depend on its batch neighbours).
#[test]
fn b_sequences_read_one_pool_through_their_own_tables() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let heads_per_seq = 4;
    let n_kv_heads = 2;
    let page_size = 32;
    let pool_pages = 4;
    let n_keys = 40; // two chunks, the second a quarter full
    let n_seqs = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_chunks = flash_chunk_count(n_keys, page_size);
    assert_eq!(n_chunks, 2);

    // Interleaved page sets: sequence 0 at pages [3, 1], sequence 1 at [0, 2].
    let tables: [Vec<u32>; 2] = [vec![3, 1], vec![0, 2]];
    let table_stride = 2;

    let capacity = 64;
    let mut references = Vec::new();
    let mut pool_k = vec![0f32; n_kv_heads * pool_pages * page_size * head_dim];
    let mut pool_v = pool_k.clone();
    let mut all_queries = Vec::new();
    for (b, table) in tables.iter().enumerate() {
        let queries = fill(heads_per_seq * head_dim, 71 + b as u32);
        let logical_k = fill(n_kv_heads * n_keys * head_dim, 81 + b as u32);
        let logical_v = fill(n_kv_heads * n_keys * head_dim, 91 + b as u32);
        let (contig_k, contig_v) = lay_out_contiguous(
            &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
        );
        references.push(run_flash_raw(
            &device,
            &kernels,
            &queries,
            &contig_k,
            &contig_v,
            FlashGeometry {
                n_seqs: 1,
                heads_per_seq,
                n_kv_heads,
                head_dim,
                n_keys,
                page_size,
                pages_per_chunk: 1,
                k_stride: [
                    n_kv_heads * capacity * head_dim,
                    capacity * head_dim,
                    head_dim,
                    1,
                ],
                table: ChunkTable::Contiguous.entries(n_chunks, 1),
                table_stride: 0,
            },
            scale,
        ));
        // Both sequences into the one pool.
        let (pk, pv) = lay_out_paged(
            &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, page_size, pool_pages, table,
        );
        for (c, table_page) in table.iter().enumerate() {
            for h in 0..n_kv_heads {
                let live = (n_keys - c * page_size).min(page_size);
                let base = h * pool_pages * page_size * head_dim
                    + (*table_page as usize) * page_size * head_dim;
                let n = live * head_dim;
                pool_k[base..base + n].copy_from_slice(&pk[base..base + n]);
                pool_v[base..base + n].copy_from_slice(&pv[base..base + n]);
            }
        }
        all_queries.extend_from_slice(&queries);
    }
    assert_ne!(
        references[0].out, references[1].out,
        "degenerate fixture: the two sequences agree"
    );

    let batched = run_flash_raw(
        &device,
        &kernels,
        &all_queries,
        &pool_k,
        &pool_v,
        FlashGeometry {
            n_seqs,
            heads_per_seq,
            n_kv_heads,
            head_dim,
            n_keys,
            page_size,
            pages_per_chunk: 1,
            k_stride: [0, pool_pages * page_size * head_dim, head_dim, 1],
            table: tables.concat(),
            table_stride,
        },
        scale,
    );
    let row = heads_per_seq * head_dim;
    for (b, reference) in references.iter().enumerate() {
        assert_eq!(
            &batched.out[b * row..(b + 1) * row],
            &reference.out[..],
            "row {b} must reproduce its own B=1 answer bit for bit through its own table \
             (DESIGN.md 2.3.3 #7, 10.3c)"
        );
    }
}

/// A RAGGED batch — three sequences of 40, 17 and 100 live keys over one pool
/// through one dispatch pair — reads each row's OWN live count
/// (`n_keys_per_seq`, `batched-decode-design.md` §12.2) and reproduces each
/// row's B=1 answer bit for bit. The dispatch depth is the longest row's
/// (4 chunks at page 32); the shorter rows' spare chunks are empty ranges
/// the partial writes as the merge's identity, and the combine — unchanged,
/// walking every chunk by index — must skip them, which the walk order
/// (every chunk visited, for every row) and the bits together assert.
#[test]
fn ragged_rows_read_their_own_live_counts_bit_for_bit() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let heads_per_seq = 4;
    let n_kv_heads = 2;
    let page_size = 32;
    let pool_pages = 8;
    let keys = [40usize, 17, 100];
    let n_seqs = keys.len();
    let longest = 100;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_chunks = flash_chunk_count(longest, page_size);
    assert_eq!(n_chunks, 4);

    // Interleaved page sets; every row's table is padded to the longest
    // row's chunk count with a page the row never reads.
    let tables: [Vec<u32>; 3] = [vec![6, 1, 0, 0], vec![3, 0, 0, 0], vec![7, 2, 5, 4]];
    let table_stride = 4;

    let capacity = 128;
    let mut references = Vec::new();
    let mut pool_k = vec![-7.0f32; n_kv_heads * pool_pages * page_size * head_dim];
    let mut pool_v = pool_k.clone();
    let mut all_queries = Vec::new();
    for (b, table) in tables.iter().enumerate() {
        let n_keys = keys[b];
        let own_chunks = flash_chunk_count(n_keys, page_size);
        let queries = fill(heads_per_seq * head_dim, 171 + b as u32);
        let logical_k = fill(n_kv_heads * n_keys * head_dim, 181 + b as u32);
        let logical_v = fill(n_kv_heads * n_keys * head_dim, 191 + b as u32);
        let (contig_k, contig_v) = lay_out_contiguous(
            &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
        );
        references.push(run_flash_raw(
            &device,
            &kernels,
            &queries,
            &contig_k,
            &contig_v,
            FlashGeometry {
                n_seqs: 1,
                heads_per_seq,
                n_kv_heads,
                head_dim,
                n_keys,
                page_size,
                pages_per_chunk: 1,
                k_stride: [
                    n_kv_heads * capacity * head_dim,
                    capacity * head_dim,
                    head_dim,
                    1,
                ],
                table: ChunkTable::Contiguous.entries(own_chunks, 1),
                table_stride: 0,
            },
            scale,
        ));
        let (pk, pv) = lay_out_paged(
            &logical_k,
            &logical_v,
            n_kv_heads,
            n_keys,
            head_dim,
            page_size,
            pool_pages,
            &table[..own_chunks],
        );
        for (c, table_page) in table[..own_chunks].iter().enumerate() {
            for h in 0..n_kv_heads {
                let live = (n_keys - c * page_size).min(page_size);
                let base = h * pool_pages * page_size * head_dim
                    + (*table_page as usize) * page_size * head_dim;
                let n = live * head_dim;
                pool_k[base..base + n].copy_from_slice(&pk[base..base + n]);
                pool_v[base..base + n].copy_from_slice(&pv[base..base + n]);
            }
        }
        all_queries.extend_from_slice(&queries);
    }

    let batched = run_flash_with(
        &device,
        &kernels,
        &all_queries,
        &pool_k,
        &pool_v,
        FlashGeometry {
            n_seqs,
            heads_per_seq,
            n_kv_heads,
            head_dim,
            n_keys: longest,
            page_size,
            pages_per_chunk: 1,
            k_stride: [0, pool_pages * page_size * head_dim, head_dim, 1],
            table: tables.concat(),
            table_stride,
        },
        Some(&keys),
        scale,
    );
    let row = heads_per_seq * head_dim;
    for (b, reference) in references.iter().enumerate() {
        assert_eq!(
            &batched.out[b * row..(b + 1) * row],
            &reference.out[..],
            "row {b} ({} keys) must reproduce its own B=1 answer bit for bit through its own \
             table and live count (DESIGN.md 2.3.3 #7; batched-decode-design.md 12.2)",
            keys[b]
        );
        // The spare chunks past the row's own are the merge's identity.
        let own_chunks = flash_chunk_count(keys[b], page_size);
        for h in 0..heads_per_seq {
            let head = b * heads_per_seq + h;
            for c in own_chunks..n_chunks {
                assert_eq!(batched.sums[head * n_chunks + c], 0.0, "row {b} head {h} chunk {c}");
                assert_eq!(
                    batched.maxs[head * n_chunks + c],
                    f32::NEG_INFINITY,
                    "row {b} head {h} chunk {c}"
                );
            }
            // The combine walked every chunk by index, empties included.
            let walked = &batched.walk[head * n_chunks..(head + 1) * n_chunks];
            assert_eq!(walked, &[0, 1, 2, 3], "row {b} head {h} walk order");
        }
    }
}

/// The same ragged batch on the SINGLE-CHUNK path: 20, 7 and 32 live keys at
/// page 32 is one chunk for every row, so `flash_decoding_single` runs and
/// reads each row's own count — the path the multi-chunk fixture above never
/// reaches (a mutation there survived it; this is the test that kills it).
#[test]
fn ragged_rows_on_the_single_chunk_path_read_their_own_live_counts() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let heads_per_seq = 4;
    let n_kv_heads = 2;
    let page_size = 32;
    let pool_pages = 4;
    let keys = [20usize, 7, 32];
    let n_seqs = keys.len();
    let longest = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(flash_chunk_count(longest, page_size), 1);

    let tables: [Vec<u32>; 3] = [vec![2], vec![0], vec![3]];
    let table_stride = 1;

    let capacity = 64;
    let mut references = Vec::new();
    let mut pool_k = vec![-7.0f32; n_kv_heads * pool_pages * page_size * head_dim];
    let mut pool_v = pool_k.clone();
    let mut all_queries = Vec::new();
    for (b, table) in tables.iter().enumerate() {
        let n_keys = keys[b];
        let queries = fill(heads_per_seq * head_dim, 271 + b as u32);
        let logical_k = fill(n_kv_heads * n_keys * head_dim, 281 + b as u32);
        let logical_v = fill(n_kv_heads * n_keys * head_dim, 291 + b as u32);
        let (contig_k, contig_v) = lay_out_contiguous(
            &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
        );
        references.push(run_flash_raw(
            &device,
            &kernels,
            &queries,
            &contig_k,
            &contig_v,
            FlashGeometry {
                n_seqs: 1,
                heads_per_seq,
                n_kv_heads,
                head_dim,
                n_keys,
                page_size,
                pages_per_chunk: 1,
                k_stride: [
                    n_kv_heads * capacity * head_dim,
                    capacity * head_dim,
                    head_dim,
                    1,
                ],
                table: ChunkTable::Contiguous.entries(1, 1),
                table_stride: 0,
            },
            scale,
        ));
        let (pk, pv) = lay_out_paged(
            &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, page_size, pool_pages, table,
        );
        for h in 0..n_kv_heads {
            let base = h * pool_pages * page_size * head_dim
                + (table[0] as usize) * page_size * head_dim;
            let n = n_keys * head_dim;
            pool_k[base..base + n].copy_from_slice(&pk[base..base + n]);
            pool_v[base..base + n].copy_from_slice(&pv[base..base + n]);
        }
        all_queries.extend_from_slice(&queries);
    }

    let batched = run_flash_with(
        &device,
        &kernels,
        &all_queries,
        &pool_k,
        &pool_v,
        FlashGeometry {
            n_seqs,
            heads_per_seq,
            n_kv_heads,
            head_dim,
            n_keys: longest,
            page_size,
            pages_per_chunk: 1,
            k_stride: [0, pool_pages * page_size * head_dim, head_dim, 1],
            table: tables.concat(),
            table_stride,
        },
        Some(&keys),
        scale,
    );
    let row = heads_per_seq * head_dim;
    for (b, reference) in references.iter().enumerate() {
        assert_eq!(
            &batched.out[b * row..(b + 1) * row],
            &reference.out[..],
            "row {b} ({} keys) on the single-chunk path must reproduce its own B=1 answer",
            keys[b]
        );
    }
}

/// The KV write lands each `(sequence, token)` row at the slot its table names
/// and nowhere else — asserted against a CPU placement of the same bytes,
/// poison included.
///
/// An absolute reference (`CONTRIBUTING.md` §3.1b rule 1): the expected pool is
/// built on the CPU from the same slot table, so a kernel writing one row to
/// the wrong slot, the wrong head, or one element off fails on a named index —
/// and the untouched poison is part of the comparison, so an extra write is as
/// visible as a missing one.
#[test]
fn the_kv_scatter_writes_each_row_at_its_slot_and_nothing_else() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_kv_heads = 2;
    let n_seqs = 2;
    let seq_len = 3;
    let page_size = 8;
    let pool_pages = 4;
    let tokens = pool_pages * page_size;

    // Slots: sequence 0 fills page 1 from offset 1; sequence 1 is scattered.
    let slots: Vec<u32> = vec![9, 10, 11, 24, 3, 5];
    let src = fill(n_seqs * n_kv_heads * seq_len * head_dim, 101);
    let src_stride = [
        n_kv_heads * seq_len * head_dim,
        seq_len * head_dim,
        head_dim,
        1,
    ];
    let dst_head_stride = tokens * head_dim;

    // The CPU placement.
    let mut expected = vec![-7.0f32; n_kv_heads * tokens * head_dim];
    for b in 0..n_seqs {
        for h in 0..n_kv_heads {
            for t in 0..seq_len {
                let slot = slots[b * seq_len + t] as usize;
                for d in 0..head_dim {
                    expected[h * dst_head_stride + slot * head_dim + d] =
                        src[b * src_stride[0] + h * src_stride[1] + t * src_stride[2] + d];
                }
            }
        }
    }

    let got = run_scatter(
        &device,
        &kernels,
        &src,
        &vec![-7.0f32; n_kv_heads * tokens * head_dim],
        &slots,
        FlashScatterParams::for_write(
            n_seqs,
            n_kv_heads,
            seq_len,
            head_dim,
            &src_stride,
            0,
            dst_head_stride,
            head_dim,
        ),
        n_kv_heads,
        head_dim,
    );
    assert_eq!(got.len(), expected.len());
    if let Some((i, (g, e))) = got
        .iter()
        .zip(expected.iter())
        .enumerate()
        .find(|(_, (g, e))| g != e)
    {
        let h = i / dst_head_stride;
        let slot = (i % dst_head_stride) / head_dim;
        let d = i % head_dim;
        panic!("pool differs from the CPU placement at head {h} slot {slot} dim {d}: got {g}, want {e}");
    }
}

/// Write through the scatter with a pool's slots, read back through the paged
/// attention with the pool's table, and reproduce the contiguous bits.
///
/// The composition the model path takes: the slot of logical token `t` is
/// `table[t / P] * P + t % P` (vLLM's `slot_mapping`), and the chunk table the
/// reader is handed is the same table. Both halves have to agree about what a
/// table means for the answer to come out; a disagreement is a wrong key read
/// silently (§3.5).
#[test]
fn scatter_then_paged_read_reproduces_the_contiguous_answer() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 2;
    let page_size = 32;
    let pool_pages = 8;
    let n_keys = 70; // three chunks, the third 6/32 full
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut pool = KvPagePool::new(page_size, pool_pages, 4, 8);
    pool.append(0, 2 * page_size).unwrap();
    pool.append(1, 2 * page_size).unwrap();
    let freed = pool.release(0);
    pool.recycle(&freed).unwrap();
    pool.append(2, n_keys).unwrap();
    let table = pool.table_for(2, 1);
    assert_eq!(
        table,
        vec![1, 0, 4],
        "recycled pages first, then a fresh one"
    );

    let queries = fill(n_q_heads * head_dim, 111);
    // The logical KV as the model produces it: `[1, H_kv, n_keys, D]`.
    let logical_k = fill(n_kv_heads * n_keys * head_dim, 112);
    let logical_v = fill(n_kv_heads * n_keys * head_dim, 113);

    let capacity = 256;
    let (contig_k, contig_v) = lay_out_contiguous(
        &logical_k, &logical_v, n_kv_heads, n_keys, head_dim, capacity,
    );
    let n_chunks = flash_chunk_count(n_keys, page_size);
    let contig = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &contig_k,
        &contig_v,
        FlashGeometry {
            n_seqs: 1,
            heads_per_seq: n_q_heads,
            n_kv_heads,
            head_dim,
            n_keys,
            page_size,
            pages_per_chunk: 1,
            k_stride: [
                n_kv_heads * capacity * head_dim,
                capacity * head_dim,
                head_dim,
                1,
            ],
            table: ChunkTable::Contiguous.entries(n_chunks, 1),
            table_stride: 0,
        },
        scale,
    );

    // One "prefill" write of all n_keys tokens through the scatter.
    let slots: Vec<u32> = (0..n_keys)
        .map(|t| table[t / page_size] * page_size as u32 + (t % page_size) as u32)
        .collect();
    let tokens = pool_pages * page_size;
    let poison = vec![-7.0f32; n_kv_heads * tokens * head_dim];
    let src_stride = [
        n_kv_heads * n_keys * head_dim,
        n_keys * head_dim,
        head_dim,
        1,
    ];
    let params = FlashScatterParams::for_write(
        1,
        n_kv_heads,
        n_keys,
        head_dim,
        &src_stride,
        0,
        tokens * head_dim,
        head_dim,
    );
    let pool_k = run_scatter(
        &device, &kernels, &logical_k, &poison, &slots, params, n_kv_heads, head_dim,
    );
    let pool_v = run_scatter(
        &device, &kernels, &logical_v, &poison, &slots, params, n_kv_heads, head_dim,
    );

    let paged = run_flash_raw(
        &device,
        &kernels,
        &queries,
        &pool_k,
        &pool_v,
        FlashGeometry {
            n_seqs: 1,
            heads_per_seq: n_q_heads,
            n_kv_heads,
            head_dim,
            n_keys,
            page_size,
            pages_per_chunk: 1,
            k_stride: [0, tokens * head_dim, head_dim, 1],
            table: table.clone(),
            table_stride: 0,
        },
        scale,
    );
    assert_eq!(
        contig.out, paged.out,
        "scatter through the table then read through the table must reproduce the contiguous \
         bits; the poison the writer left untouched is what a wrong slot would read"
    );
}

/// The paged fixtures' geometry, in one place so the tests above differ only
/// in the numbers.
struct FlashGeometry {
    n_seqs: usize,
    heads_per_seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_keys: usize,
    page_size: usize,
    pages_per_chunk: usize,
    /// `[seq, head, token, dim]` strides of the K/V buffer. `[0]` is 0 for a
    /// shared pool.
    k_stride: [usize; 4],
    table: Vec<u32>,
    table_stride: usize,
}

struct FlashRun {
    out: Vec<f32>,
    sums: Vec<f32>,
    maxs: Vec<f32>,
    walk: Vec<u32>,
}

/// `[H_kv][capacity][D]` from a logical `[H_kv][n_keys][D]`.
fn lay_out_contiguous(
    logical_k: &[f32],
    logical_v: &[f32],
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    capacity: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0f32; n_kv_heads * capacity * head_dim];
    let mut v = vec![0f32; n_kv_heads * capacity * head_dim];
    for h in 0..n_kv_heads {
        let n = n_keys * head_dim;
        k[h * capacity * head_dim..h * capacity * head_dim + n]
            .copy_from_slice(&logical_k[h * n..(h + 1) * n]);
        v[h * capacity * head_dim..h * capacity * head_dim + n]
            .copy_from_slice(&logical_v[h * n..(h + 1) * n]);
    }
    (k, v)
}

/// `[H_kv][pool_pages * page_size][D]` from a logical `[H_kv][n_keys][D]`,
/// logical chunk `c` stored at page `table[c]`. Unwritten pool bytes are a
/// poison, so a read past a chunk's live count is visible.
#[allow(clippy::too_many_arguments)]
fn lay_out_paged(
    logical_k: &[f32],
    logical_v: &[f32],
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    page_size: usize,
    pool_pages: usize,
    table: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let tokens = pool_pages * page_size;
    let mut k = vec![-7.0f32; n_kv_heads * tokens * head_dim];
    let mut v = vec![-7.0f32; n_kv_heads * tokens * head_dim];
    for h in 0..n_kv_heads {
        for t in 0..n_keys {
            let slot = table[t / page_size] as usize * page_size + t % page_size;
            let src = h * n_keys * head_dim + t * head_dim;
            let dst = h * tokens * head_dim + slot * head_dim;
            k[dst..dst + head_dim].copy_from_slice(&logical_k[src..src + head_dim]);
            v[dst..dst + head_dim].copy_from_slice(&logical_v[src..src + head_dim]);
        }
    }
    (k, v)
}

/// One run at `Sizing::Grow` over an explicit geometry, returning the output,
/// the per-chunk statistics and the walk order.
fn run_flash_raw(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    g: FlashGeometry,
    scale: f32,
) -> FlashRun {
    run_flash_with(device, kernels, queries, keys, values, g, None, scale)
}

/// [`run_flash_raw`] with a live key count PER SEQUENCE when `keys_per_seq`
/// is given (`g.n_keys` is then the longest row's, the dispatch depth), the
/// ragged batch `batched-decode-design.md` §12.2 reads in one dispatch.
#[allow(clippy::too_many_arguments)]
fn run_flash_with(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    g: FlashGeometry,
    keys_per_seq: Option<&[usize]>,
    scale: f32,
) -> FlashRun {
    let cmds = commands(device);
    let n_q_heads = g.n_seqs * g.heads_per_seq;
    let chunk_size = g.page_size * g.pages_per_chunk;
    let n_chunks = flash_chunk_count(g.n_keys, chunk_size);
    if let Some(k) = keys_per_seq {
        assert_eq!(k.len(), g.n_seqs, "one live count per sequence");
        assert_eq!(
            k.iter().copied().max(),
            Some(g.n_keys),
            "`n_keys` is the longest row's"
        );
    }
    let key_counts: Vec<u32> = match keys_per_seq {
        Some(k) => k.iter().map(|&n| n as u32).collect(),
        None => vec![g.n_keys as u32],
    };
    let keys_buf = new_u32(device, &key_counts);
    assert!(
        g.table.len() >= n_chunks + (g.n_seqs - 1) * g.table_stride,
        "the table must cover every sequence's chunks"
    );

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device
        .new_buffer(n_q_heads * g.head_dim * 4, SHARED)
        .unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * g.head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &g.table);
    let walk = new_u32(device, &vec![u32::MAX; n_q_heads * n_chunks]);

    let q_shape = [g.n_seqs, g.heads_per_seq, 1, g.head_dim];
    let k_shape = [g.n_seqs, g.n_kv_heads, g.n_keys, g.head_dim];
    let mut partial = FlashPartialParams::for_step(
        &q_shape,
        &k_shape,
        &g.k_stride,
        &g.k_stride,
        g.page_size,
        g.pages_per_chunk,
        n_chunks,
        n_chunks,
        g.table_stride,
        scale,
        1.0,
    )
    .unwrap();
    if keys_per_seq.is_some() {
        partial = partial.with_keys_per_sequence();
    }
    let pp = write_struct(device, partial);
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            &keys_buf,
            n_q_heads,
            g.head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    let walk_p = walk.contents() as *const u32;
    // SAFETY: shared storage sized for `n_q_heads * n_chunks` u32, waited.
    let walk = unsafe { std::slice::from_raw_parts(walk_p, n_q_heads * n_chunks) }.to_vec();
    FlashRun {
        out: read_f32(&out, n_q_heads * g.head_dim),
        sums: read_f32(&sums, n_q_heads * n_chunks),
        maxs: read_f32(&maxs, n_q_heads * n_chunks),
        walk,
    }
}

/// One scatter of `src` into a copy of `dst_initial`, returning the pool.
#[allow(clippy::too_many_arguments)]
fn run_scatter(
    device: &Device,
    kernels: &Kernels,
    src: &[f32],
    dst_initial: &[f32],
    slots: &[u32],
    params: FlashScatterParams,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let cmds = commands(device);
    let src_buf = new_f32(device, src);
    let dst_buf = new_f32(device, dst_initial);
    let slots_buf = new_u32(device, slots);
    let pp = write_struct(device, params);
    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_kv_scatter(
            device,
            &guard,
            kernels,
            0,
            &src_buf,
            0,
            &dst_buf,
            &slots_buf,
            &pp,
            params.n_rows as usize,
            n_kv_heads,
            head_dim,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    read_f32(&dst_buf, dst_initial.len())
}

/// As [`run_flash`], returning `(partials, maxs)` so a test can assert on where
/// a chunk's work landed rather than only on the merged output.
#[allow(clippy::too_many_arguments)]
fn run_partials(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    scale: f32,
    table: &ChunkTable,
) -> (Vec<f32>, Vec<f32>) {
    let cmds = commands(device);
    let n_chunks = flash_chunk_count(n_keys, page_size);

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &table.entries(n_chunks, 1));
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let keys_buf = new_u32(device, &[n_keys as u32]);

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        // `n_chunks` twice: the live count and the capacity, which is
        // `Sizing::Grow` and is what this fixture takes (#234).
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, n_chunks, 0, scale,
            1.0,
        )
        .unwrap(),
    );
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            &keys_buf,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    (
        read_f32(&partials, n_q_heads * n_chunks * head_dim),
        read_f32(&maxs, n_q_heads * n_chunks),
    )
}

/// **The same input gives the same bits, over repeated runs.**
///
/// §2.3.1's run-to-run kind, at the kernel level. It is a *necessary* check and
/// not a sufficient one — §2.3.5a: a kernel with a wrong constant is perfectly
/// deterministic — which is why it sits beside the parity test rather than
/// instead of it.
///
/// What it does catch is the failure mode §10.4 names as the most likely in this
/// design: a merge whose order depends on which chunk finished first. With 11
/// chunks over 8 heads there are 88 threadgroups whose completion order the
/// scheduler chooses, so a completion-ordered merge has ample opportunity to
/// differ between runs.
#[test]
fn flash_is_bit_stable_across_runs() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 8;
    let n_kv_heads = 2;
    let capacity = 4096;
    let n_keys = 2720; // §9.1's "largest ever measured"
    let page_size = 256; // §10.4's proposed size
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 31);
    let keys = fill(n_kv_heads * capacity * head_dim, 32);
    let values = fill(n_kv_heads * capacity * head_dim, 33);

    assert_eq!(flash_chunk_count(n_keys, page_size), 11);

    let first = run_flash(
        &device,
        &kernels,
        &queries,
        &keys,
        &values,
        n_q_heads,
        n_kv_heads,
        head_dim,
        capacity,
        n_keys,
        page_size,
        1,
        scale,
        &ChunkTable::Contiguous,
    );
    for run in 1..5 {
        let again = run_flash(
            &device,
            &kernels,
            &queries,
            &keys,
            &values,
            n_q_heads,
            n_kv_heads,
            head_dim,
            capacity,
            n_keys,
            page_size,
            1,
            scale,
            &ChunkTable::Contiguous,
        );
        assert_eq!(
            first,
            again,
            "run {run} differs from run 0 at kv_len={n_keys}, {} chunks -- a merge whose order \
             depends on completion is the first thing to suspect (DESIGN.md 10.4)",
            flash_chunk_count(n_keys, page_size)
        );
    }
}

/// **The token stride is read from the parameter, not assumed to be
/// `head_dim`.**
///
/// This is §9.1d's field — the one `sdpa_vector` could not vary, because its
/// token step is `constexpr int stride = BN * D`. Carrying it as a parameter is
/// what makes a second KV layout a different *number* rather than a different
/// kernel, and therefore what makes the arm #200 lacked buildable.
///
/// **A fixture whose token stride happens to equal `head_dim` cannot show
/// that**, and every other test in this file is such a fixture. Measured
/// rather than assumed: a mutation replacing `k_token_stride` with `D` left all
/// eight of them passing. That is #70's `align_up` lesson in a second quantity
/// (§9.2c) — *a parity test built only from the model's own shapes is blind to
/// a whole defect class*.
///
/// So this fixture pads each token's row: the KV is laid out with a stride of
/// `head_dim + PAD` and only the first `head_dim` elements of each row are the
/// real vector. The padding is filled with a value large enough that reading it
/// as data would move the answer visibly rather than subtly.
#[test]
fn token_stride_is_read_from_the_parameter() {
    let device = device();
    let kernels = Kernels::new();

    const PAD: usize = 8;
    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 128;
    let n_keys = 100;
    let page_size = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let row = head_dim + PAD;

    let queries = fill(n_q_heads * head_dim, 51);
    let tight_k = fill(n_kv_heads * capacity * head_dim, 52);
    let tight_v = fill(n_kv_heads * capacity * head_dim, 53);

    // The same values, re-laid at a padded row stride. Padding is 7.0 rather
    // than 0.0: a zero would be a plausible dot-product contribution and could
    // cancel, where 7.0 across 8 lanes cannot be mistaken for rounding.
    let mut padded_k = vec![7.0f32; n_kv_heads * capacity * row];
    let mut padded_v = vec![7.0f32; n_kv_heads * capacity * row];
    for h in 0..n_kv_heads {
        for t in 0..capacity {
            for d in 0..head_dim {
                padded_k[h * capacity * row + t * row + d] =
                    tight_k[h * capacity * head_dim + t * head_dim + d];
                padded_v[h * capacity * row + t * row + d] =
                    tight_v[h * capacity * head_dim + t * head_dim + d];
            }
        }
    }

    let n_chunks = flash_chunk_count(n_keys, page_size);
    let cmds = commands(&device);
    let q_buf = new_f32(&device, &queries);
    let k_buf = new_f32(&device, &padded_k);
    let v_buf = new_f32(&device, &padded_v);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(&device, &ChunkTable::Contiguous.entries(n_chunks, 1));
    let walk = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let keys_buf = new_u32(&device, &[n_keys as u32]);

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    // Index 2 is `row`, not `head_dim`. **That is the whole fixture.**
    let k_stride = [n_kv_heads * capacity * row, capacity * row, row, 1];

    let pp = write_struct(
        &device,
        // `n_chunks` twice: the live count and the capacity, which is
        // `Sizing::Grow` and is what this fixture takes (#234).
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, n_chunks, 0, scale,
            1.0,
        )
        .unwrap(),
    );
    let cp = write_struct(&device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            &device,
            &guard,
            &kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            &keys_buf,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    let got = read_f32(&out, n_q_heads * head_dim);

    // The reference reads the TIGHT layout: the padded cache holds the same
    // vectors, so a kernel honouring the stride must agree with it, and one
    // assuming `head_dim` reads the padding and cannot.
    let want = reference_attention(
        &queries, &tight_k, &tight_v, n_q_heads, n_kv_heads, head_dim, capacity, n_keys, scale,
    );
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        worst = worst.max((g - w).abs());
    }
    assert!(
        worst < 2e-5,
        "at a padded token stride ({row} against head_dim {head_dim}) the kernel diverged by \
         {worst:e} -- it is assuming head_dim rather than reading k_token_stride, which is the \
         field 9.1d asks this kernel to carry"
    );
}

/// **The combine walks chunk indices in ASCENDING order, asserted against the
/// order the kernel recorded rather than against its output.**
///
/// This is §10.4's own instruction and the test the rest of this file cannot
/// substitute for. That section records the reversal as *"caught by that
/// assertion and by nothing else"* — the bit-equality tests stayed green under
/// it, because floating-point non-associativity happened not to bite on that
/// fixture.
///
/// **Measured here rather than inherited: reversing both combine loops left all
/// seven of this family's other tests passing**, parity against the f64
/// reference included. So the claim is not that an output comparison is
/// *theoretically* insufficient; it is that on this kernel, at these shapes, it
/// *was* insufficient, and this assertion is what fired.
///
/// §2.3.3 #1 is the rule: every reduction merges in fixed index order, never
/// completion order. `CombineOrder::Completion` exists as a spelling in
/// `scratch.rs` so the rule has something to reject; this is the same
/// discipline applied to the kernel that does the merging.
#[test]
fn combine_walks_chunks_in_ascending_index_order() {
    let device = device();
    let kernels = Kernels::new();

    let head_dim = 64;
    let n_q_heads = 4;
    let n_kv_heads = 1;
    let capacity = 512;
    let n_keys = 300;
    let page_size = 32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let queries = fill(n_q_heads * head_dim, 41);
    let keys = fill(n_kv_heads * capacity * head_dim, 42);
    let values = fill(n_kv_heads * capacity * head_dim, 43);
    let n_chunks = flash_chunk_count(n_keys, page_size);
    assert!(
        n_chunks > 2,
        "a one- or two-chunk fixture cannot show an order"
    );

    let walked = walk_order_of(
        &device, &kernels, &queries, &keys, &values, n_q_heads, n_kv_heads, head_dim, capacity,
        n_keys, page_size, scale,
    );

    let expected: Vec<u32> = (0..n_chunks as u32).collect();
    for h in 0..n_q_heads {
        let got = &walked[h * n_chunks..(h + 1) * n_chunks];
        assert_eq!(
            got,
            &expected[..],
            "head {h} walked {got:?} where 10.4 requires ascending index order {expected:?}"
        );
    }
}

/// As [`run_flash`], returning the chunk indices the combine recorded walking.
#[allow(clippy::too_many_arguments)]
fn walk_order_of(
    device: &Device,
    kernels: &Kernels,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    n_keys: usize,
    page_size: usize,
    scale: f32,
) -> Vec<u32> {
    let cmds = commands(device);
    let n_chunks = flash_chunk_count(n_keys, page_size);

    let q_buf = new_f32(device, queries);
    let k_buf = new_f32(device, keys);
    let v_buf = new_f32(device, values);
    let out = device.new_buffer(n_q_heads * head_dim * 4, SHARED).unwrap();
    let partials = device
        .new_buffer(n_q_heads * n_chunks * head_dim * 4, SHARED)
        .unwrap();
    let sums = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let maxs = device.new_buffer(n_q_heads * n_chunks * 4, SHARED).unwrap();
    let table_buf = new_u32(device, &ChunkTable::Contiguous.entries(n_chunks, 1));
    // Poisoned, so an unwritten slot is visibly unwritten rather than a
    // plausible 0 -- the same reason 9.1a reads its partials back against a
    // poisoned buffer instead of trusting a zero.
    let walk = new_u32(device, &vec![u32::MAX; n_q_heads * n_chunks]);
    let keys_buf = new_u32(device, &[n_keys as u32]);

    let q_shape = [1usize, n_q_heads, 1, head_dim];
    let k_shape = [1usize, n_kv_heads, n_keys, head_dim];
    let k_stride = [
        n_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    ];

    let pp = write_struct(
        device,
        // `n_chunks` twice: the live count and the capacity, which is
        // `Sizing::Grow` and is what this fixture takes (#234).
        FlashPartialParams::for_step(
            &q_shape, &k_shape, &k_stride, &k_stride, page_size, 1, n_chunks, n_chunks, 0, scale,
            1.0,
        )
        .unwrap(),
    );
    let cp = write_struct(device, FlashCombineParams::for_step(n_chunks, n_chunks));

    {
        let guard = cmds.command_encoder().unwrap();
        call_flash_decoding(
            device,
            &guard,
            kernels,
            0,
            &q_buf,
            0,
            &k_buf,
            0,
            &v_buf,
            &out,
            &partials,
            &sums,
            &maxs,
            &table_buf,
            &pp,
            &cp,
            &walk,
            &keys_buf,
            n_q_heads,
            head_dim,
            n_chunks,
            FlashDType::F32,
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();

    let p = walk.contents() as *const u32;
    // SAFETY: shared storage sized for `n_q_heads * n_chunks` u32, waited.
    let got = unsafe { std::slice::from_raw_parts(p, n_q_heads * n_chunks) }.to_vec();
    assert!(
        !got.contains(&u32::MAX),
        "the combine left a slot unwritten, so the recorded order is incomplete and \
         asserting on it would assert nothing"
    );
    got
}

// The layout check is **not** here, and that is #58's mechanism working.
//
// `every_family_params_layout_matches_metal` iterates `LayoutFamily::ALL`, so
// registering `LayoutFamily::Flash` is what enrols this family — and
// `layout_registry_covers_every_family`'s exhaustive `match` makes a family
// that fails to register an `error[E0004]` rather than a silently unchecked
// one. §11.3i: *a family absent from the call-site list was never checked,
// which looks exactly like a family that passes.* Writing a per-family body
// here would put back the call site that mechanism removed.
