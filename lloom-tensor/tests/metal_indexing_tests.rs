//! `indexing.metal`'s op families must agree with the CPU backend, and every
//! `(index dtype, value dtype)` pair `candle-core` can ask for must exist.
//!
//! Three distinct things are checked here, and they fail for different reasons.
//!
//! **Parity** — the kernels compute the same values as the CPU backend. This is
//! the check a templating conversion needs: `indexing.metal`'s macro form
//! restated each kernel's signature beside the template it forwarded to, and
//! `DESIGN.md` §8.1c records the hazard that shape carries — reordering two
//! same-typed parameters in one of the two lists compiles, resolves, and reads
//! its bindings wrongly. `index` takes five consecutive `constant size_t &`
//! parameters, so that shape was reachable here.
//!
//! **Existence** — the pair resolves to a kernel at all. `index_select` with
//! `I64` indices over a `U8` or `U32` tensor was a runtime failure on
//! `lloom/integration` before this file existed: `candle-core` named
//! `is_i64_u8` and `is_i64_u32`, and `indexing.metal` declared neither. That is
//! the third and fourth name in the absent-variant class `DESIGN.md` §8.1b
//! tracks, after #26's 48 reduce variants and `conv`'s.
//!
//! **Agreement on what is refused** — the backends decline the same calls, not
//! only compute the same answers. A backend that accepts an input its reference
//! rejects returns a *plausible wrong tensor* where an error belongs, which is
//! the worse failure direction because nothing announces it.
//! `gather_declines_a_strided_source_like_cpu_and_cuda` is that check (#87) and
//! `cpu_declines_strided_index_select_source` records the one place the three
//! backends still legitimately differ.
//!
//! The reference is the **CPU backend**, per `CONTRIBUTING.md` §3.1 and the
//! standing guidance in `DESIGN.md` §2.3.8a: it is bit-stable, and a Metal-side
//! reference would make this test depend on the path it is meant to validate.

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

/// Values chosen so a transposed or misread dimension changes the answer.
///
/// A tensor of equal rows would pass a parity test with `left_size` and
/// `right_size` swapped, so the generator varies along both axes.
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32) - 50.0).collect()
}

/// Every dtype pair runs, bf16 included, which this GPU may only emulate
/// (§15.2a): the device is opted in to it.
fn cpu_and_metal() -> Option<(Device, Device)> {
    let metal = Device::new_metal(0).ok()?;
    lloom_tensor::test_utils::allow_emulated(&metal, DType::BF16);
    Some((Device::Cpu, metal))
}

/// Compare a tensor against the same computation on the CPU, exactly.
///
/// Exact rather than tolerant: every op in this file *moves* values rather than
/// computing on them, so any difference is an addressing bug, not float noise.
/// `scatter_add` and `index_add` are the exceptions and are compared in `f32`
/// where the accumulation order is identical between backends.
fn assert_same(metal: &Tensor, cpu: &Tensor, what: &str) {
    let m = metal.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let c = cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        m.len(),
        c.len(),
        "{what}: metal produced {} values, cpu {}",
        m.len(),
        c.len()
    );
    assert_eq!(m, c, "{what}: metal and cpu disagree");
}

/// Every `(ids dtype, value dtype)` pair `MetalStorage::index_select` names.
///
/// Written out rather than derived so this list is the *caller's* view: it is
/// what `candle-core`'s match arms ask for, and the point of the test is to
/// compare that against what `indexing.metal` declares. Deriving it from the
/// registry would compare the registry against itself.
const INDEX_SELECT_PAIRS: &[(DType, DType)] = &[
    (DType::U8, DType::U8),
    (DType::U8, DType::U32),
    (DType::U8, DType::I64),
    (DType::U8, DType::BF16),
    (DType::U8, DType::F32),
    (DType::U8, DType::F16),
    (DType::U32, DType::U8),
    (DType::U32, DType::U32),
    (DType::U32, DType::I64),
    (DType::U32, DType::F32),
    (DType::U32, DType::F16),
    (DType::U32, DType::BF16),
    // The two that did not exist. `index_select` on a U8 or U32 tensor with
    // I64 indices was a `LoadFunctionError` from inside a forward pass.
    (DType::I64, DType::U8),
    (DType::I64, DType::U32),
    (DType::I64, DType::I64),
    (DType::I64, DType::F32),
    (DType::I64, DType::F16),
    (DType::I64, DType::BF16),
];

/// `index_select` must resolve and agree with the CPU for all 18 pairs.
///
/// The two `I64`-indexed integer pairs are the regression guard: before the
/// kernels were declared this test failed on them with a `LoadFunctionError`,
/// which is the failure a user saw at runtime.
#[test]
fn index_select_every_dtype_pair_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (6usize, 4usize);
    let raw = values(rows * cols);
    // Deliberately unsorted and repeating: a kernel that ignored `ids` and
    // copied rows in order would pass against `[0, 1, 2]`.
    let ids_raw: Vec<u32> = vec![4, 0, 3, 3, 1];

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in INDEX_SELECT_PAIRS {
        // Values must survive a round trip through the value dtype on both
        // backends, so u8 gets a range it can hold. Otherwise the comparison
        // would be against differing truncation, not against the kernel.
        let src: Vec<f32> = match val_dtype {
            DType::U8 => raw.iter().map(|v| (v.abs() % 128.0).trunc()).collect(),
            DType::U32 | DType::I64 => raw.iter().map(|v| v.abs().trunc()).collect(),
            _ => raw.clone(),
        };

        for dim in [0usize, 1usize] {
            let n = if dim == 0 { rows } else { cols };
            let ids_v: Vec<u32> = ids_raw.iter().map(|i| i % n as u32).collect();

            let build = |dev: &Device| -> Result<Tensor> {
                let t = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
                let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?.to_dtype(id_dtype)?;
                t.index_select(&ids, dim)?.to_dtype(DType::F32)
            };

            let got = build(&metal).unwrap_or_else(|e| {
                panic!("index_select ids={id_dtype:?} values={val_dtype:?} dim={dim}: {e}")
            });
            let want = build(&cpu)?;
            assert_same(
                &got,
                &want,
                &format!("index_select ids={id_dtype:?} values={val_dtype:?} dim={dim}"),
            );
            checked += 1;
        }
    }

    // Non-vacuity: 18 pairs x 2 dims. A loop that silently ran zero iterations
    // would otherwise be a green test (`DESIGN.md` §8.1b, and the guard
    // `conv_names_resolve` carries for the same reason).
    assert_eq!(checked, 36, "expected 36 index_select cases, ran {checked}");
    Ok(())
}

/// `index_select` on a **non-contiguous source** must agree with the CPU.
///
/// This replaces `index_select_strided_source_is_known_wrong`, which asserted
/// the *inequality* that the defect below produced. The defect is fixed, so the
/// assertion is now the equality it always ought to have been. The old test is
/// referenced by name here deliberately: it is the record that this was known,
/// pinned, and then corrected rather than silently changed.
///
/// **The defect it pinned**, in `indexing.metal`'s `index`:
///
/// ```text
/// get_strided_index(src_i, src_dim_size, src_dims, src_strides)
/// //                       ^^^^^^^^^^^^ num_dims (the tensor's rank) belongs here
/// ```
///
/// `src_dim_size` is the extent of the *indexed dimension*; `get_strided_index`
/// walks one iteration per *axis* and so wants the **rank**. Unrelated
/// quantities sharing a type, which is why it compiled and read wrong elements
/// in silence. The CUDA kernel for the same op passes the rank
/// (`candle-kernels/src/indexing.cu:65`), so this was a **Metal-only divergence
/// from the reference**, established by reading that source rather than
/// inferred from the symptom.
///
/// **Which arm this exercises: the strided one.** `contiguous` is false here,
/// so the kernel takes `get_strided_index` rather than the `contiguous ? src_i`
/// fast arm. That distinction is the whole point of this test — LFM2's decode
/// path uses `is_u32_f16` on a *contiguous* embedding table, so it takes the
/// other arm and unchanged LFM2 digests say nothing about this fix. The tests
/// above (`index_select_every_dtype_pair_matches_cpu`,
/// `embedding_lookup_shape_matches_cpu`) exercise the contiguous arm.
///
/// **Why the reference is `contiguous()` on the CPU rather than the CPU's own
/// strided path.** The CPU backend *rejects* a non-contiguous source outright
/// (`cpu_backend/mod.rs`'s `IndexSelect::f` bails with `RequiresContiguous`), so
/// there is no CPU strided arm to compare against. The reference is therefore
/// the same *logical* operation — materialize the transpose, then index-select —
/// computed entirely on the CPU. That is a stronger anchor than comparing
/// Metal's two arms against each other, which would only prove they agree.
///
/// The case set spans ranks 2 and 3 and both `dim` values, including shapes
/// where `rank == dims[dim]`. Those coincidence cases pass even with the defect
/// present; they are kept so the test does not merely encode one lucky shape,
/// and the mutation test below identifies which cases discriminate.
#[test]
fn index_select_strided_source_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    // (shape, dim, n_ids). Chosen so `rank` and `dims[dim]` differ in most
    // cases -- that inequality is the condition under which the old argument
    // and the correct one diverge.
    let cases: &[(&[usize], usize, usize)] = &[
        (&[3, 5], 0, 4),
        (&[3, 5], 1, 3),
        (&[4, 7], 0, 5),
        (&[7, 4], 1, 6),
        (&[2, 3, 4], 0, 3),
        (&[2, 3, 4], 1, 4),
        (&[2, 3, 4], 2, 2),
        // rank == dims[dim]: the coincidence cases, which the defect did not
        // affect. Kept so a future change that breaks only these is still seen.
        (&[3, 2], 0, 3),
        (&[5, 2], 1, 4),
    ];

    let mut checked = 0usize;
    let mut discriminating = 0usize;

    for &(shape, dim, n_ids) in cases {
        let n: usize = shape.iter().product();
        let raw = values(n);

        // Transposing the last two axes makes the source non-contiguous while
        // leaving every axis extent reachable, so `dim` still selects a real
        // dimension of the transposed view.
        let build = |dev: &Device| -> Result<(Tensor, Tensor)> {
            let t = Tensor::from_vec(raw.clone(), shape, dev)?.to_dtype(DType::F32)?;
            let rank = shape.len();
            let strided = t.transpose(rank - 2, rank - 1)?;
            let extent = strided.dims()[dim];
            let ids_v: Vec<u32> = (0..n_ids).map(|i| ((i * 3 + 1) % extent) as u32).collect();
            let ids = Tensor::from_vec(ids_v, n_ids, dev)?;
            Ok((strided, ids))
        };

        let (m_strided, m_ids) = build(&metal)?;
        assert!(
            !m_strided.is_contiguous(),
            "shape={shape:?} dim={dim}: the source must be non-contiguous, or this tests nothing"
        );

        let got = m_strided
            .index_select(&m_ids, dim)
            .unwrap_or_else(|e| panic!("strided index_select shape={shape:?} dim={dim}: {e}"));

        // The CPU reference: same logical op, but materialized first, because
        // the CPU backend declines a strided source.
        let (c_strided, c_ids) = build(&cpu)?;
        let want = c_strided.contiguous()?.index_select(&c_ids, dim)?;

        assert_same(
            &got,
            &want,
            &format!("strided index_select shape={shape:?} dim={dim}"),
        );

        if m_strided.dims().len() != m_strided.dims()[dim] {
            discriminating += 1;
        }
        checked += 1;
    }

    // Non-vacuity, per the guard the other tests in this file carry: a loop that
    // ran zero iterations would otherwise be green.
    assert_eq!(checked, 9, "expected 9 strided cases, ran {checked}");
    // And a sharper one specific to this test: at least one case must actually
    // have `rank != dims[dim]`, or every case is a coincidence shape and the
    // suite would pass with the defect reinstated.
    assert!(
        discriminating >= 5,
        "only {discriminating} cases have rank != dims[dim]; this test would not \
         discriminate the defect it exists to catch"
    );
    Ok(())
}

/// The CPU backend declines a strided `index_select` source; Metal accepts one.
///
/// Recorded as a test rather than a comment because it is the premise the test
/// above rests on — its reference is `contiguous()` *because* of this — and
/// because it is a second, larger CPU/Metal divergence than the argument bug,
/// found while building that reference. It is **not** fixed here: making the two
/// backends agree means either teaching the CPU the strided path or having Metal
/// decline one, and both are behaviour changes well outside a one-argument
/// numerics fix. CUDA, like Metal, supports the strided path
/// (`candle-kernels/src/indexing.cu:65`), so the CPU is the odd one out and
/// removing Metal's capability would be a divergence in the other direction.
#[test]
fn cpu_declines_strided_index_select_source() -> Result<()> {
    let cpu = Device::Cpu;
    let t = Tensor::from_vec(values(3 * 5), (3, 5), &cpu)?.to_dtype(DType::F32)?;
    let strided = t.t()?;
    assert!(!strided.is_contiguous());
    let ids = Tensor::from_vec(vec![2u32, 0, 1, 1], 4, &cpu)?;

    let err = strided
        .index_select(&ids, 0)
        .expect_err("the CPU backend is expected to decline a strided source");
    assert!(
        err.to_string().contains("contiguous"),
        "expected a contiguity complaint, got: {err}"
    );

    // The same op succeeds once materialized, which is what the parity test uses.
    strided.contiguous()?.index_select(&ids, 0)?;
    Ok(())
}

/// `gather` must agree with the CPU across its declared dtype pairs.
#[test]
fn gather_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (4usize, 5usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U32, DType::I64),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::U8, DType::BF16),
        (DType::U8, DType::U8),
        (DType::U8, DType::U32),
        (DType::U8, DType::I64),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::BF16),
        (DType::I64, DType::U32),
        (DType::I64, DType::I64),
    ];

    // One index per output element, varying along both axes so a swapped
    // `right_size`/`left_size` changes the result.
    let idx: Vec<u32> = (0..rows * cols).map(|i| ((i * 3) % cols) as u32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let src: Vec<f32> = match val_dtype {
            DType::U8 => values(rows * cols)
                .iter()
                .map(|v| (v.abs() % 128.0).trunc())
                .collect(),
            DType::U32 | DType::I64 => values(rows * cols)
                .iter()
                .map(|v| v.abs().trunc())
                .collect(),
            _ => values(rows * cols),
        };

        let build = |dev: &Device| -> Result<Tensor> {
            let t = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(idx.clone(), (rows, cols), dev)?.to_dtype(id_dtype)?;
            t.gather(&ids, 1)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("gather ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("gather ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 16, "expected 16 gather cases, ran {checked}");
    Ok(())
}

/// `scatter_add` must agree with the CPU.
///
/// Compared in `f32` and with each destination slot written at most once, so
/// the comparison is of *addressing* rather than of accumulation order —
/// float addition is not associative (`DESIGN.md` §2.3.2) and a test that
/// depended on summation order would be checking something this kernel does not
/// promise.
#[test]
fn scatter_add_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (3usize, 4usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::U8, DType::BF16),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::BF16),
    ];

    // A permutation per row: every destination slot receives exactly one value.
    let idx: Vec<u32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| ((c + r) % cols) as u32))
        .collect();
    let src: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let build = |dev: &Device| -> Result<Tensor> {
            let dst = Tensor::zeros((rows, cols), val_dtype, dev)?;
            let s = Tensor::from_vec(src.clone(), (rows, cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(idx.clone(), (rows, cols), dev)?.to_dtype(id_dtype)?;
            dst.scatter_add(&ids, &s, 1)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("scatter_add ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("scatter_add ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 10, "expected 10 scatter_add cases, ran {checked}");
    Ok(())
}

/// `index_add` must agree with the CPU.
///
/// Distinct indices for the same reason `scatter_add` uses a permutation: this
/// checks addressing, not accumulation order.
#[test]
fn index_add_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    let (rows, cols) = (5usize, 3usize);
    let pairs = [
        (DType::U32, DType::F32),
        (DType::U32, DType::F16),
        (DType::U32, DType::BF16),
        (DType::U32, DType::U32),
        (DType::U8, DType::F32),
        (DType::U8, DType::F16),
        (DType::I64, DType::F32),
        (DType::I64, DType::F16),
        (DType::I64, DType::U32),
    ];

    // Distinct destination rows, unsorted so order is exercised.
    let ids_v: Vec<u32> = vec![3, 0, 4];
    let src: Vec<f32> = (0..ids_v.len() * cols).map(|i| (i % 9) as f32).collect();

    let mut checked = 0usize;
    for &(id_dtype, val_dtype) in &pairs {
        let build = |dev: &Device| -> Result<Tensor> {
            let dst = Tensor::zeros((rows, cols), val_dtype, dev)?;
            let s = Tensor::from_vec(src.clone(), (ids_v.len(), cols), dev)?.to_dtype(val_dtype)?;
            let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?.to_dtype(id_dtype)?;
            dst.index_add(&ids, &s, 0)?.to_dtype(DType::F32)
        };

        let got = build(&metal)
            .unwrap_or_else(|e| panic!("index_add ids={id_dtype:?} values={val_dtype:?}: {e}"));
        assert_same(
            &got,
            &build(&cpu)?,
            &format!("index_add ids={id_dtype:?} values={val_dtype:?}"),
        );
        checked += 1;
    }

    assert_eq!(checked, 9, "expected 9 index_add cases, ran {checked}");
    Ok(())
}

/// The embedding-lookup shape specifically: `is_u32_f16`, one dispatch per
/// decode token.
///
/// `DESIGN.md` §11.3h calls this out as the only `indexing.metal` kernel on the
/// decode path, so it gets a case at the shape LFM2 actually uses — a row
/// gathered from a `[vocab, hidden]` table — rather than only at the small
/// shapes above. A whole-file conversion that broke exactly this one would
/// change generated text while every other case stayed green.
#[test]
fn embedding_lookup_shape_matches_cpu() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    // LFM2's hidden size; a small vocab, since the width is what matters.
    let (vocab, hidden) = (64usize, 2048usize);
    let table = values(vocab * hidden);
    let ids_v: Vec<u32> = vec![37];

    let build = |dev: &Device| -> Result<Tensor> {
        let t = Tensor::from_vec(table.clone(), (vocab, hidden), dev)?.to_dtype(DType::F16)?;
        let ids = Tensor::from_vec(ids_v.clone(), ids_v.len(), dev)?;
        t.index_select(&ids, 0)?.to_dtype(DType::F32)
    };

    let got = build(&metal)?;
    assert_same(&got, &build(&cpu)?, "embedding lookup is_u32_f16");

    // And that it is the row asked for, not row 0 -- an addressing bug that
    // returned a *valid* row would otherwise pass only against the CPU's same
    // bug, which cannot happen here but is cheap to pin.
    let row = got.i(0)?.to_vec1::<f32>()?;
    let want: Vec<f32> = table[37 * hidden..38 * hidden].to_vec();
    assert_eq!(row, want, "embedding lookup returned the wrong row");
    Ok(())
}

/// `gather` must **decline** a non-contiguous source, as the CPU and CUDA do.
///
/// This replaces `gather_strided_source_is_known_wrong`, which asserted the
/// *inequality* the defect produced — written to pass while the bug was present
/// so the behaviour stayed on the record, and to turn red the moment it was
/// corrected. It has been corrected, so the assertion is now that the call is
/// refused. The old test is named here deliberately: it is the record that this
/// was known, pinned, and then fixed rather than silently changed. Same
/// mechanism `index_select_strided_source_matches_cpu` carries above.
///
/// **The defect it pinned.** `MetalStorage::gather` checked
/// `ids_l.is_contiguous()` and never `src_l`, where `scatter_set`,
/// `scatter_add_set` and `index_add` all check every operand they take. The
/// kernel it dispatches computes
///
/// ```text
/// src_i = (left_rank_i * src_dim_size + input_i) * right_size + right_rank_i
/// ```
///
/// a flat contiguous offset, and receives neither `src_dims` nor `src_strides`
/// to do anything else with — so a strided source was read as though it were
/// contiguous and produced a plausible wrong tensor.
///
/// **Why a guard and not a strided arm, which is the opposite of #82's fix.**
/// Both reference backends decline this call. The CPU bails in
/// `cpu_backend/mod.rs`'s `Gather::f`, and **so does CUDA** —
/// `cuda_backend/mod.rs`'s `Gather::f` carries a second `RequiresContiguous` on
/// the source, distinct from the one on the ids, and `indexing.cu`'s `gather`
/// kernel takes no `num_dims`/`info` at all. So the guard makes Metal agree with
/// both references, where a strided arm would have made it the only backend that
/// computes. #82 faced the mirror image — CUDA's `index_select` *does* take
/// `num_dims` and branch on `is_contiguous`, so there only the CPU declined and
/// making Metal's strided arm correct was right. Read from those two sources
/// rather than inferred from the symptom.
///
/// **What is asserted, and why each part is here.** That Metal declines; that it
/// declines *for the contiguity reason* rather than incidentally; that the
/// contiguous arm still computes and still agrees with the CPU, so the guard
/// rejected only what it should; and that all three backends now behave alike.
#[test]
fn gather_declines_a_strided_source_like_cpu_and_cuda() -> Result<()> {
    let Some((cpu, metal)) = cpu_and_metal() else {
        return Ok(());
    };

    // Several shapes and both `dim` values, so this does not encode one lucky
    // layout. `(shape, dim)`; each is transposed on its last two axes to make
    // the source non-contiguous.
    let cases: &[(&[usize], usize)] = &[(&[3, 4], 0), (&[3, 4], 1), (&[4, 7], 1), (&[2, 3, 4], 2)];

    let mut checked = 0usize;
    for &(shape, dim) in cases {
        let n: usize = shape.iter().product();
        let raw = values(n);
        let rank = shape.len();

        let build = |dev: &Device| -> Result<(Tensor, Tensor)> {
            let t = Tensor::from_vec(raw.clone(), shape, dev)?.to_dtype(DType::F32)?;
            let strided = t.transpose(rank - 2, rank - 1)?;
            // `gather`'s ids must have the source's shape outside `dim`, so they
            // are built against the transposed view.
            let ids_shape = strided.dims().to_vec();
            let extent = ids_shape[dim];
            let total: usize = ids_shape.iter().product();
            let ids_v: Vec<u32> = (0..total).map(|i| ((i * 3) % extent) as u32).collect();
            let ids = Tensor::from_vec(ids_v, ids_shape, dev)?;
            Ok((strided, ids))
        };

        let (m_strided, m_ids) = build(&metal)?;
        assert!(
            !m_strided.is_contiguous(),
            "shape={shape:?} dim={dim}: the source must be non-contiguous, or this tests nothing"
        );

        // 1. Metal declines it now.
        let err = m_strided.gather(&m_ids, dim).expect_err(&format!(
            "shape={shape:?} dim={dim}: Metal is expected to decline a strided gather source"
        ));
        // 2. And for the contiguity reason -- not, say, a shape complaint that
        //    would pass this test while the defect was still reachable by
        //    another route.
        assert!(
            err.to_string().contains("contiguous"),
            "shape={shape:?} dim={dim}: expected a contiguity complaint, got: {err}"
        );

        // 3. The contiguous arm still computes, and still agrees with the CPU.
        //    Without this the guard could reject everything and stay green.
        let (c_strided, c_ids) = build(&cpu)?;
        assert_same(
            &m_strided.contiguous()?.gather(&m_ids, dim)?,
            &c_strided.contiguous()?.gather(&c_ids, dim)?,
            &format!("gather contiguous arm shape={shape:?} dim={dim}"),
        );

        // 4. The CPU declines the same call, which is the agreement this fix
        //    establishes. (CUDA also declines -- `cuda_backend/mod.rs`'s
        //    `Gather::f` -- but is not compiled here, so it is cited rather
        //    than exercised.)
        assert!(
            c_strided.gather(&c_ids, dim).is_err(),
            "shape={shape:?} dim={dim}: the CPU backend is expected to decline it too"
        );

        checked += 1;
    }

    // Non-vacuity, per the guard the other tests in this file carry: a loop that
    // ran zero iterations would otherwise be green.
    assert_eq!(checked, 4, "expected 4 strided gather cases, ran {checked}");
    Ok(())
}
