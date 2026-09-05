//! The fused depthwise conv1d must agree with the generic grouped path.
//!
//! The fused kernel writes straight to `(b, c, l_out)` and skips im2col, so a
//! layout or padding mistake would show up as plausible-but-wrong numbers
//! rather than a failure. These compare against the path it replaces.

#![cfg(feature = "metal")]

// Fused depthwise conv1d must match the generic grouped path exactly.
use lloom_tensor::{DType, Device, Tensor};

fn reference(x: &Tensor, w: &Tensor, pad: usize, groups: usize) -> lloom_tensor::Result<Tensor> {
    // Force the generic path by running each group separately.
    let xs = x.chunk(groups, 1)?;
    let ws = w.chunk(groups, 0)?;
    let parts = xs
        .iter()
        .zip(&ws)
        .map(|(xi, wi)| xi.conv1d(wi, pad, 1, 1, 1))
        .collect::<lloom_tensor::Result<Vec<_>>>()?;
    Tensor::cat(&parts, 1)
}

fn check(dev: &Device, dtype: DType, b: usize, c: usize, l: usize, k: usize, pad: usize) {
    let n = b * c * l;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    let x = Tensor::from_vec(xv, (b, c, l), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let w = Tensor::from_vec(wv, (c, 1, k), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let fused = x.conv1d(&w, pad, 1, 1, c).unwrap();
    let refr = reference(&x, &w, pad, c).unwrap();

    assert_eq!(
        fused.dims(),
        refr.dims(),
        "shape mismatch {dtype:?} b{b} c{c} l{l} k{k} pad{pad}"
    );

    let a = fused
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let e = refr
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
    let mut worst = 0f32;
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
        }
        assert!(
            d <= tol,
            "{dtype:?} b{b} c{c} l{l} k{k} pad{pad}: idx {i} {x} vs {y}"
        );
    }
    println!("  ok {dtype:?} b={b} c={c} l={l} k={k} pad={pad} worst={worst:.2e}");
}

#[test]
fn depthwise_conv1d_matches_generic_path() {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    for dtype in [DType::F32, DType::F16] {
        // LFM2's real shape (k=3, pad=2) plus edge cases.
        check(&dev, dtype, 1, 2048, 16, 3, 2);
        check(&dev, dtype, 1, 8, 5, 3, 2);
        check(&dev, dtype, 2, 4, 7, 3, 2);
        check(&dev, dtype, 1, 4, 1, 3, 2); // single position
        check(&dev, dtype, 1, 3, 9, 1, 0); // k=1, no padding
        check(&dev, dtype, 1, 5, 6, 5, 0); // wide kernel, no padding
        check(&dev, dtype, 3, 16, 12, 3, 1);
    }
    println!("all depthwise conv parity checks passed");
}

/// The `k_size`-specialized kernel must agree with the CPU backend.
///
/// Reference is the CPU backend rather than the Metal generic path on purpose.
/// The generic path is the *chunked* grouped convolution, which is itself
/// nondeterministic at high channel counts — measured at 2-5 distinct results
/// over 8 repetitions at c=2048 (see `measurements/issue-10-*` in the lloom
/// repo). Comparing a new kernel against an unstable reference cannot
/// distinguish "my kernel is wrong" from "the reference moved", so this test
/// uses the one reference that is bit-stable.
///
/// The shapes below straddle the dispatch decision in
/// `MetalStorage::conv1d_depthwise`: `k` in {2, 3, 4} takes the specialized
/// kernel, `k` in {1, 5} falls back to the generic one. Both sides are checked
/// here so a mistake in the *predicate* — not just in the kernel — fails.
fn check_vs_cpu(dtype: DType, b: usize, c: usize, l: usize, k: usize, pad: usize) {
    let Ok(dev) = Device::new_metal(0) else {
        return;
    };
    let cpu = Device::Cpu;

    let n = b * c * l;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    let build = |d: &Device| -> (Tensor, Tensor) {
        (
            Tensor::from_vec(xv.clone(), (b, c, l), d)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
            Tensor::from_vec(wv.clone(), (c, 1, k), d)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        )
    };

    let (xm, wm) = build(&dev);
    let (xc, wc) = build(&cpu);

    let got = xm.conv1d(&wm, pad, 1, 1, c).unwrap();
    let want = xc.conv1d(&wc, pad, 1, 1, c).unwrap();

    assert_eq!(
        got.dims(),
        want.dims(),
        "shape {dtype:?} b{b} c{c} l{l} k{k} pad{pad}"
    );

    let a = got
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let e = want
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
    let mut worst = 0f32;
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
        }
        assert!(
            d <= tol,
            "{dtype:?} b{b} c{c} l{l} k{k} pad{pad}: idx {i} {x} vs {y}"
        );
    }
    println!("  ok {dtype:?} b={b} c={c} l={l} k={k} pad={pad} worst={worst:.2e}");
}

#[test]
fn depthwise_conv1d_specialized_matches_cpu() {
    if Device::new_metal(0).is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    for dtype in [DType::F32, DType::F16] {
        // Specialized path: k in {2, 3, 4}, stride 1, dilation 1, contiguous.
        check_vs_cpu(dtype, 1, 2048, 16, 3, 2); // LFM2's real shape
        check_vs_cpu(dtype, 1, 2048, 736, 3, 2); // LFM2 prefill length
        check_vs_cpu(dtype, 1, 8, 5, 3, 2);
        check_vs_cpu(dtype, 2, 4, 7, 3, 2);
        check_vs_cpu(dtype, 1, 4, 1, 3, 2); // l_out shorter than one simdgroup
        check_vs_cpu(dtype, 1, 16, 10, 2, 1); // k=2
        check_vs_cpu(dtype, 1, 16, 10, 4, 3); // k=4
        check_vs_cpu(dtype, 1, 16, 10, 3, 0); // no padding at all
        check_vs_cpu(dtype, 3, 16, 12, 3, 1);
        // Fallback path: k outside {2,3,4} must still be correct.
        check_vs_cpu(dtype, 1, 3, 9, 1, 0); // k=1
        check_vs_cpu(dtype, 1, 5, 6, 5, 0); // k=5
    }
    println!("all specialized depthwise conv parity checks passed");
}

/// A non-contiguous input must not reach the specialized kernel.
///
/// The specialized kernel indexes its source as a contiguous `(b, c, l_in)` and
/// ignores `src_strides[]` entirely, so dispatching a strided layout to it reads
/// the wrong elements — quietly, with plausible numbers. The guard that prevents
/// that is one `layout.is_contiguous()` in the dispatch predicate, and nothing
/// else in this file exercises it: every other case builds its input directly
/// and is contiguous by construction.
///
/// `narrow` along the length axis gives a view whose `l_in` no longer matches
/// its stride, which is exactly the shape the specialized addressing gets wrong.
#[test]
fn depthwise_conv1d_noncontiguous_input_matches_cpu() {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let (b, c, l_full, k, pad) = (2usize, 64usize, 24usize, 3usize, 2usize);
    let l = 10usize; // the narrowed window
    let n = b * c * l_full;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    for dtype in [DType::F32, DType::F16] {
        let run = |d: &Device| -> Tensor {
            let x = Tensor::from_vec(xv.clone(), (b, c, l_full), d)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .narrow(2, 3, l)
                .unwrap();
            assert!(!x.is_contiguous(), "narrow should give a strided view");
            let w = Tensor::from_vec(wv.clone(), (c, 1, k), d)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            x.conv1d(&w, pad, 1, 1, c).unwrap()
        };
        let got = run(&dev);
        let want = run(&Device::Cpu);

        let a = got
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let e = want
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
        let mut worst = 0f32;
        for (i, (x, y)) in a.iter().zip(&e).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
            }
            assert!(d <= tol, "{dtype:?} non-contiguous: idx {i} {x} vs {y}");
        }
        println!(
            "  ok {dtype:?} non-contiguous b={b} c={c} l={l} k={k} pad={pad} worst={worst:.2e}"
        );
    }
}

/// The specialized kernel must be bit-stable across repeated dispatches.
///
/// `DESIGN.md` §2.3 makes determinism an invariant rather than a nicety, and
/// §15.1 #7 asks for N runs with identical output. This is the kernel-level
/// form of that gate: the same inputs through the same pipeline, repeated, must
/// give byte-identical results.
#[test]
fn depthwise_conv1d_specialized_is_deterministic() {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let (b, c, l, k, pad) = (1usize, 2048usize, 736usize, 3usize, 2usize);
    let n = b * c * l;
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let wv: Vec<f32> = (0..c * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
        .collect();

    for dtype in [DType::F32, DType::F16] {
        let x = Tensor::from_vec(xv.clone(), (b, c, l), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let w = Tensor::from_vec(wv.clone(), (c, 1, k), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };

        let first = bits(&x.conv1d(&w, pad, 1, 1, c).unwrap());
        for run in 1..16 {
            let again = bits(&x.conv1d(&w, pad, 1, 1, c).unwrap());
            assert_eq!(first, again, "{dtype:?} run {run} differs from run 0");
        }
        println!(
            "  ok {dtype:?}: 16 runs bit-identical over {} elements",
            first.len()
        );
    }
}
