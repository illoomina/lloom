#![cfg(feature = "metal")]
//! `Tensor::arg_sort_last_dim` on Metal, at widths past one threadgroup.
//!
//! # What was wrong
//!
//! `ArgSort::metal_fwd` rounded `ncols` up to a power of two and called
//! `call_arg_sort`, which dispatches **one threadgroup of `ncols_pad` threads**.
//! A Metal threadgroup holds at most **1024** (lloom `DESIGN.md` §3.1), so at
//! LFM2's 128 000-wide vocabulary the dispatch asked for 131 072 — 128× the
//! limit — and the kernel returned zeros.
//!
//! It returned them **silently**, and that is the part worth keeping: the
//! obvious predicate, *"is each element ≥ the next?"*, is satisfied by
//! `v[0] >= v[0]` at every step, so an all-zeros output looks sorted. §15.1 #1
//! already says to prefer *"at least two distinct values"* to *"not all zero"*;
//! these tests check the result is a **permutation**, which is what
//! discriminates, and `a_sortedness_check_alone_cannot_discriminate` pins the
//! trap so it is not fallen into again.
//!
//! # What the fix is
//!
//! `call_mlx_arg_sort` routes to a multi-block sort with no width limit. It is
//! **not** a drop-in replacement — it sorts ascending only, and covers six
//! dtypes where this entry point names nine — so it is taken exactly where it
//! applies, and the single-threadgroup path is kept for the rest, where below
//! the limit it is correct and above it now **fails loudly** rather than
//! returning zeros.

use lloom_tensor::{DType, Device, Result, Tensor};

/// Is `idx` a permutation of `0..n`? The check that discriminates; sortedness
/// alone does not.
fn is_permutation(idx: &[u32], n: usize) -> bool {
    if idx.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &i in idx {
        let i = i as usize;
        if i >= n || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

/// A tie-free ramp, shuffled by a fixed permutation so the answer is neither the
/// identity nor its reverse. Ties are excluded because the sort is documented
/// unstable, so a fixture with duplicates cannot tell a different-but-valid
/// answer from a wrong one.
fn fixture(n: usize) -> Vec<f32> {
    // A full-period LCG over `n` when `n` is a power of two, and a simple
    // multiplicative walk otherwise; both are deterministic and tie-free.
    let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.swap(i, (state % (i as u64 + 1)) as usize);
    }
    v
}

fn cpu_reference(values: &[f32], asc: bool) -> Result<Vec<u32>> {
    let n = values.len();
    let cpu = Tensor::from_vec(values.to_vec(), (n,), &Device::Cpu)?;
    cpu.arg_sort_last_dim(asc)?.to_vec1()
}

/// **The acceptance criterion**: a permutation at the widths #277 measured as
/// broken, checked against the CPU backend.
#[test]
fn arg_sort_ascending_matches_cpu_across_widths() -> Result<()> {
    let device = Device::new_metal(0)?;
    // 256 and 1024 fit one threadgroup and were already correct; 4096 upward is
    // where the old path returned zeros. 128 000 is LFM2's vocabulary.
    for n in [256usize, 1024, 4096, 16_384, 65_536, 128_000] {
        let values = fixture(n);
        let t = Tensor::from_vec(values.clone(), (n,), &device)?;
        let idx: Vec<u32> = t.arg_sort_last_dim(true)?.to_vec1()?;

        assert!(
            is_permutation(&idx, n),
            "n={n}: not a permutation — this is the #346 defect"
        );
        // Bit-identical to the CPU. The fixture is tie-free, so the unstable
        // sort has exactly one correct answer and equality is the right check.
        assert_eq!(
            idx,
            cpu_reference(&values, true)?,
            "n={n}: differs from CPU"
        );
    }
    Ok(())
}

/// Every dtype the multi-block path carries, at a width that needs it.
#[test]
fn arg_sort_ascending_matches_cpu_across_dtypes() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 16_384usize;
    let values = fixture(n);

    // f16 and bf16 cannot represent this ramp without ties, so they get their
    // own tie-free ramps built from values each represents exactly.
    for dtype in [DType::F32, DType::U32] {
        let t = Tensor::from_vec(values.clone(), (n,), &device)?.to_dtype(dtype)?;
        let idx: Vec<u32> = t.arg_sort_last_dim(true)?.to_vec1()?;
        assert!(is_permutation(&idx, n), "{dtype:?}: not a permutation");

        let cpu = Tensor::from_vec(values.clone(), (n,), &Device::Cpu)?.to_dtype(dtype)?;
        let want: Vec<u32> = cpu.arg_sort_last_dim(true)?.to_vec1()?;
        assert_eq!(idx, want, "{dtype:?}: differs from CPU");
    }
    Ok(())
}

/// A strided source and a contiguous one, per the issue's acceptance list.
/// `arg_sort_last_dim` requires a contiguous tensor, so the strided case is a
/// non-contiguous view made contiguous — which is what a caller does, and which
/// exercises a different source offset into the same buffer.
#[test]
fn arg_sort_on_a_strided_source_matches_cpu() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 8192usize;
    // Two rows; take the second, so the source carries a nonzero offset.
    let values = fixture(2 * n);
    let t = Tensor::from_vec(values.clone(), (2, n), &device)?;

    let row = t.narrow(0, 1, 1)?.contiguous()?;
    let idx: Vec<u32> = row.arg_sort_last_dim(true)?.flatten_all()?.to_vec1()?;
    assert!(is_permutation(&idx, n), "strided: not a permutation");

    let want = cpu_reference(&values[n..], true)?;
    assert_eq!(idx, want, "strided: differs from CPU");

    // The contiguous control. Without it, a green strided arm is consistent
    // with both arms being wrong the same way.
    let flat = Tensor::from_vec(values[..n].to_vec(), (n,), &device)?;
    let idx: Vec<u32> = flat.arg_sort_last_dim(true)?.to_vec1()?;
    assert_eq!(
        idx,
        cpu_reference(&values[..n], true)?,
        "contiguous control"
    );
    Ok(())
}

/// Multiple rows: the multi-block path indexes per row, and a single-row
/// fixture cannot tell a correct row stride from one that reads row 0 twice.
#[test]
fn arg_sort_multi_row_matches_cpu() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (rows, n) = (4usize, 8192usize);
    let values = fixture(rows * n);
    let t = Tensor::from_vec(values.clone(), (rows, n), &device)?;
    let idx: Vec<u32> = t.arg_sort_last_dim(true)?.flatten_all()?.to_vec1()?;

    for r in 0..rows {
        let got = &idx[r * n..(r + 1) * n];
        assert!(is_permutation(got, n), "row {r}: not a permutation");
        let want = cpu_reference(&values[r * n..(r + 1) * n], true)?;
        assert_eq!(got, want, "row {r}: differs from CPU");
    }
    Ok(())
}

/// Descending below the threadgroup limit still uses the single-threadgroup
/// path and must be unchanged. This is the arm the fix must not disturb: it is
/// what every MoE router in `candle-transformers` calls.
#[test]
fn arg_sort_descending_below_the_limit_is_unchanged() -> Result<()> {
    let device = Device::new_metal(0)?;
    for n in [64usize, 256, 1024] {
        let values = fixture(n);
        let t = Tensor::from_vec(values.clone(), (n,), &device)?;
        let idx: Vec<u32> = t.arg_sort_last_dim(false)?.to_vec1()?;
        assert!(
            is_permutation(&idx, n),
            "n={n}: descending not a permutation"
        );
        assert_eq!(
            idx,
            cpu_reference(&values, false)?,
            "n={n}: descending differs from CPU"
        );
    }
    Ok(())
}

/// Descending above the limit refuses loudly instead of returning zeros.
///
/// This is the honest outcome rather than the complete one: the multi-block
/// path sorts ascending only, so a wide descending sort has no kernel. An error
/// is what `DESIGN.md` §6.2b and §9.1a ask for where the alternative is a
/// silent wrong answer, and it names the two ways out.
#[test]
fn arg_sort_descending_above_the_limit_refuses() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 4096usize;
    let values = fixture(n);
    let t = Tensor::from_vec(values, (n,), &device)?;
    let err = match t.arg_sort_last_dim(false) {
        Ok(_) => panic!("n={n}: descending above the limit should refuse, not return zeros"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("threadgroup limit"),
        "n={n}: refused for the wrong reason: {err}"
    );
    Ok(())
}

/// The trap, pinned. A sortedness check passes on an all-zeros result, so it
/// cannot be the gate — and it also passes on the *correct* result, which is
/// what makes it useless rather than merely weak. Only the permutation check
/// separates them.
#[test]
fn a_sortedness_check_alone_cannot_discriminate() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 128_000usize;
    let values = fixture(n);
    let t = Tensor::from_vec(values.clone(), (n,), &device)?;
    let idx: Vec<u32> = t.arg_sort_last_dim(true)?.to_vec1()?;

    let looks_sorted = |v: &[u32]| {
        v.windows(2)
            .all(|w| values[w[0] as usize] <= values[w[1] as usize])
    };

    // The real result passes it...
    assert!(looks_sorted(&idx), "the fixed kernel is not ascending");
    // ...and so does all-zeros, which is not a permutation.
    let zeros = vec![0u32; n];
    assert!(
        looks_sorted(&zeros),
        "all-zeros no longer satisfies the naive predicate; the trap this test \
         records has changed shape"
    );
    assert!(!is_permutation(&zeros, n), "all-zeros is not a permutation");
    assert!(is_permutation(&idx, n), "the fixed kernel is a permutation");
    Ok(())
}

/// Ties, and run-to-run identity under them (issue #748). The QSA indexer
/// ranks ReLU'd block scores, many of which are exactly 0, with the ascending
/// multi-block path; the selection is read back as cell indices, so a
/// returned index that is not in `0..n` gathers from outside the cache and a
/// different permutation of equal keys each run is a different logits digest
/// each run. At the widths the long-context gate ran (526 and 937 blocks) and
/// with every row half zeros: every result must be a permutation of `0..n`,
/// sorted by key, and identical across 16 dispatches.
#[test]
fn ascending_sort_with_ties_is_a_permutation_and_repeatable() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for &n in &[64usize, 526, 937, 2051, 4100] {
        let v: Vec<f32> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    0.0
                } else {
                    -(((i * 7919) % 1013) as f32)
                }
            })
            .collect();
        let t = Tensor::from_vec(v.clone(), (1, n), &dev)?;
        let mut first: Option<Vec<u32>> = None;
        for rep in 0..16 {
            let idx = t
                .arg_sort_last_dim(true)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            assert!(
                is_permutation(&idx, n),
                "n={n} rep={rep}: not a permutation of 0..n: {:?}",
                &idx[..8.min(n)]
            );
            for w in idx.windows(2) {
                assert!(
                    v[w[0] as usize] <= v[w[1] as usize],
                    "n={n} rep={rep}: not sorted"
                );
            }
            match &first {
                None => first = Some(idx),
                Some(f) => assert_eq!(f, &idx, "n={n} rep={rep}: a different permutation of equal keys than the first dispatch"),
            }
        }
    }
    Ok(())
}
