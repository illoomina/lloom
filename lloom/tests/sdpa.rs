#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    use lloom_tensor::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;
    use std::ops::{Div, Mul};

    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let vs: Vec<f32> = (0..elem_count).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(vs, &shape, dev)
    }

    #[test]
    fn sdpa_full() -> Result<()> {
        // Test the full SDPA kernel path (q_seq > 8)
        const BS: usize = 4;
        const R: usize = 16;
        const L: usize = 16;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = lloom::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = lloom::nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        // Larger sequences have higher accumulated error
        assert!(error <= 0.02, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = lloom::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = lloom::nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.000, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_softcapping() -> Result<()> {
        // Test softcapping with sdpa_vector kernel (q_seq = 1)
        // NOTE: Vector kernel only supports q_seq = 1 correctly
        // Full kernel does NOT support softcapping
        const BS: usize = 4;
        const R: usize = 1; // Vector kernel requires q_seq = 1
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = lloom::nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output =
            lloom::nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        // Slightly higher error for cross-attention case (R=1, L=4)
        assert!(error <= 0.002, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector_softcapping() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = lloom::nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output =
            lloom::nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.0001, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector_cross() -> Result<()> {
        // Allow vectorized, seqlen = 1. Simulat cross attention case where R != L, R = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = lloom::nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = lloom::nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // GQA against the CPU backend, at LFM2's decode shape (lloom issue #97).
    //
    // Every test above sets q heads == kv heads, so `gqa_factor` is 1 and the
    // kernel's `kv_head_idx = head_idx / gqa_factor` is exercised only at its
    // identity value. LFM2 decode runs it at 4 — 32 q heads over 8 kv heads,
    // `DESIGN.md` §5.2 — which is the path #97 routes attention onto, so it was
    // the one case the suite did not cover.
    //
    // The reference is the **CPU backend** per `CONTRIBUTING.md` §3.1: it is
    // bit-stable and upstreamable, where a Metal ground truth shares a backend
    // with the thing under test. It cannot be "the same call on CPU" —
    // `Sdpa::cpu_fwd` bails with "SDPA has no cpu impl" — so the reference is
    // the same *mathematics*: repeat_kv, matmul, softmax, matmul, in f32.
    // ---------------------------------------------------------------------

    /// LFM2 decode geometry (§5.2): 32 query heads, 8 kv heads, head_dim 64.
    const LFM2_Q_HEADS: usize = 32;
    const LFM2_KV_HEADS: usize = 8;
    const LFM2_HEAD_DIM: usize = 64;

    /// Broadcast each kv head to `repeat` query heads.
    ///
    /// Spelled out rather than taken from `lloom::utils`, which
    /// would make `candle-nn`'s tests depend on a crate above it.
    fn repeat_kv(x: &Tensor, repeat: usize) -> Result<Tensor> {
        if repeat == 1 {
            return Ok(x.clone());
        }
        let (b, kv_heads, seq, dim) = x.dims4()?;
        Tensor::cat(&vec![x; repeat], 2)?.reshape((b, kv_heads * repeat, seq, dim))
    }

    /// repeat_kv + matmul + softmax + matmul, in f32. Mirrors the arm
    /// `AttnImpl::Generic` takes in `lfm2.rs`.
    fn reference_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (_, q_heads, _, _) = q.dims4()?;
        let (_, kv_heads, _, _) = k.dims4()?;
        let repeat = q_heads / kv_heads;

        let k = repeat_kv(k, repeat)?;
        let v = repeat_kv(v, repeat)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?.contiguous()?)? * scale)?;
        let att = lloom::nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v.contiguous()?)
    }

    /// Asserts agreement *and* that the comparison was not vacuous.
    ///
    /// Two tensors of zeros agree perfectly, and `DESIGN.md` §3.7a records
    /// all-zero output as the signature of a silently-nonfunctional Metal
    /// pipeline. So the guard lives in the comparison both arms route through
    /// rather than copied into each (§15.1 #1, lloom issue #53), and it counts
    /// distinct values rather than testing "not all zero", for #53's reason.
    fn assert_matches_reference(got: &Tensor, want: &Tensor, tol: f32, what: &str) -> Result<f32> {
        assert_eq!(got.shape(), want.shape(), "{what}: shape");

        let got_v = got.flatten_all()?.to_vec1::<f32>()?;
        let distinct = got_v
            .iter()
            .map(|f| f.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct >= 2,
            "{what}: vacuous comparison — wrote {distinct} distinct value(s) across {} elements",
            got_v.len()
        );

        let want_v = want.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = got_v
            .iter()
            .zip(want_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_abs <= tol, "{what}: max abs error {max_abs} > {tol}");
        Ok(max_abs)
    }

    /// Metal `sdpa` at GQA 4:1 against the CPU backend, f32.
    ///
    /// f32 so the comparison is about the *kernel* rather than about f16
    /// rounding: a disagreement here is the GQA indexing or the online-softmax
    /// accumulation, not storage precision.
    #[test]
    fn sdpa_vector_gqa_matches_cpu_backend_f32() -> Result<()> {
        const L: usize = 137; // deliberately not a multiple of the kernel's BN=32

        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(970001);

        let q = randn(&mut rng, (1, LFM2_Q_HEADS, 1, LFM2_HEAD_DIM), &cpu)?;
        let k = randn(&mut rng, (1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), &cpu)?;
        let v = randn(&mut rng, (1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), &cpu)?;

        let want = reference_attention(&q, &k, &v, scale)?;

        let got = lloom::nn::ops::sdpa(
            &q.to_device(&metal)?,
            &k.to_device(&metal)?,
            &v.to_device(&metal)?,
            None,
            false,
            scale as f32,
            1.,
        )?
        .to_device(&cpu)?;

        let err = assert_matches_reference(&got, &want, 2e-5, "gqa f32")?;
        println!("sdpa_vector_gqa_matches_cpu_backend_f32: max abs error {err:e}");
        Ok(())
    }

    /// The same at f16, which is what LFM2 actually decodes in.
    ///
    /// The tolerance is looser because the reference upcasts f16 to f32 before
    /// the multiply where the kernel widens per element, so the two round the
    /// inputs differently by construction. That difference is *why* #97 predicts
    /// a changed digest, and this bounds it rather than waving at it.
    #[test]
    fn sdpa_vector_gqa_matches_cpu_backend_f16() -> Result<()> {
        const L: usize = 137;

        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(970002);

        let q = randn(&mut rng, (1, LFM2_Q_HEADS, 1, LFM2_HEAD_DIM), &cpu)?;
        let k = randn(&mut rng, (1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), &cpu)?;
        let v = randn(&mut rng, (1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), &cpu)?;

        // Round-trip through f16 on both arms so the inputs are identical bits.
        let q16 = q.to_dtype(DType::F16)?;
        let k16 = k.to_dtype(DType::F16)?;
        let v16 = v.to_dtype(DType::F16)?;

        let want = reference_attention(&q16, &k16, &v16, scale)?;

        let got = lloom::nn::ops::sdpa(
            &q16.to_device(&metal)?,
            &k16.to_device(&metal)?,
            &v16.to_device(&metal)?,
            None,
            false,
            scale as f32,
            1.,
        )?
        .to_device(&cpu)?
        .to_dtype(DType::F32)?;

        let err = assert_matches_reference(&got, &want, 5e-3, "gqa f16")?;
        println!("sdpa_vector_gqa_matches_cpu_backend_f16: max abs error {err:e}");
        Ok(())
    }

    /// The kernel must read the kv head `gqa_factor` selects, not another one.
    ///
    /// The two tests above compare against a reference that would disagree only
    /// numerically if the mapping were wrong, and on random inputs a wrong-head
    /// read is a plausible-looking tensor. Here every kv head holds a distinct
    /// constant and `k` is zeros, so softmax is uniform and each query head's
    /// output is *exactly* the constant of the kv head it read. A wrong mapping
    /// is then an unmistakable integer-sized error rather than a small one.
    #[test]
    fn sdpa_vector_gqa_reads_the_right_kv_head() -> Result<()> {
        const L: usize = 8;
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        let q = Tensor::ones((1, LFM2_Q_HEADS, 1, LFM2_HEAD_DIM), DType::F32, &cpu)?;
        let k = Tensor::zeros((1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), DType::F32, &cpu)?;

        // kv head h is filled with the value (h + 1).
        let mut v_data = Vec::with_capacity(LFM2_KV_HEADS * L * LFM2_HEAD_DIM);
        for h in 0..LFM2_KV_HEADS {
            for _ in 0..(L * LFM2_HEAD_DIM) {
                v_data.push((h + 1) as f32);
            }
        }
        let v = Tensor::from_vec(v_data, (1, LFM2_KV_HEADS, L, LFM2_HEAD_DIM), &cpu)?;

        let got = lloom::nn::ops::sdpa(
            &q.to_device(&metal)?,
            &k.to_device(&metal)?,
            &v.to_device(&metal)?,
            None,
            false,
            1.0,
            1.,
        )?
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        let gqa_factor = LFM2_Q_HEADS / LFM2_KV_HEADS;
        for qh in 0..LFM2_Q_HEADS {
            let want_head = qh / gqa_factor;
            let expected = (want_head + 1) as f32;
            for d in 0..LFM2_HEAD_DIM {
                let seen = got[qh * LFM2_HEAD_DIM + d];
                assert!(
                    (seen - expected).abs() <= 1e-5,
                    "q head {qh} lane {d}: read value {seen}, expected {expected} \
                     (kv head {want_head})"
                );
            }
        }
        Ok(())
    }
    /// `sdpa_vector` over a **narrowed** K/V, which is the shape an in-place KV
    /// cache hands it (lloom issue #142).
    ///
    /// The cache is pre-allocated to `capacity` and only `live` positions are
    /// filled, so the kernel receives `n = live` with a head stride still
    /// spanning `capacity`. Nothing in this suite covered that: every other arm
    /// passes a tensor whose dim-2 extent *is* its stride, so a kernel that
    /// ignored `k_stride` and assumed contiguity would pass all of them and
    /// silently read across head boundaries here.
    ///
    /// **Both vector kernels at every length.** `Sdpa::metal_fwd` picks the
    /// single-pass or the 2-pass kernel by `ops::sdpa_vector_two_pass`, so
    /// the two are different code over the same input. Each is checked here
    /// at every `live` through `ops::sdpa_vector_pinned`, whatever the route
    /// would choose, and the routed result must be bit-identical to the
    /// kernel the route names. The padding past `live` is filled with a large
    /// sentinel rather than zeros: zeros would contribute `exp(0)` mass that a
    /// wrong length might hide, where a sentinel makes an over-read dominate
    /// the softmax and show up as a gross error.
    #[test]
    fn sdpa_vector_over_a_narrowed_kv_matches_cpu_backend() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        // 1 and 31 leave simdgroups with no key; 1024 is a multiple of every
        // block stride and 1100 is not; 1600 is past both the default
        // threshold and the measured arm's, so the route takes each kernel at
        // least once under either.
        for &(live, capacity) in &[
            (1usize, 64usize),
            (31, 64),
            (137, 512),
            (600, 1024),
            (1024, 2048),
            (1100, 2048),
            (1600, 2048),
        ] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(142_000 + live as u64);

            let q = randn(&mut rng, (1, LFM2_Q_HEADS, 1, LFM2_HEAD_DIM), &cpu)?;
            let k_live = randn(&mut rng, (1, LFM2_KV_HEADS, live, LFM2_HEAD_DIM), &cpu)?;
            let v_live = randn(&mut rng, (1, LFM2_KV_HEADS, live, LFM2_HEAD_DIM), &cpu)?;

            // The reference sees only the live prefix -- that is the whole claim.
            let want = reference_attention(&q, &k_live, &v_live, scale)?;

            // Build the padded buffer the way the cache does, then narrow it.
            let pad_len = capacity - live;
            let sentinel = 40f32; // exp(40 * scale) dwarfs any live score
            let k_pad = Tensor::full(sentinel, (1, LFM2_KV_HEADS, pad_len, LFM2_HEAD_DIM), &cpu)?;
            let v_pad = Tensor::full(sentinel, (1, LFM2_KV_HEADS, pad_len, LFM2_HEAD_DIM), &cpu)?;
            let k_all = Tensor::cat(&[&k_live, &k_pad], 2)?.contiguous()?;
            let v_all = Tensor::cat(&[&v_live, &v_pad], 2)?.contiguous()?;

            let k = k_all.to_device(&metal)?.narrow(2, 0, live)?;
            let v = v_all.to_device(&metal)?.narrow(2, 0, live)?;
            // A view, not a copy: this is the property that makes it the cache's
            // shape rather than a re-materialised contiguous tensor.
            assert!(
                !k.is_contiguous(),
                "live={live}: the narrow must stay strided"
            );
            let qm = q.to_device(&metal)?;

            let mut pinned = Vec::new();
            for two_pass in [false, true] {
                let got =
                    lloom::nn::ops::sdpa_vector_pinned(&qm, &k, &v, scale as f32, 1., two_pass)?
                        .to_device(&cpu)?;
                let err = assert_matches_reference(
                    &got,
                    &want,
                    2e-5,
                    &format!("narrowed kv live={live} cap={capacity} two_pass={two_pass}"),
                )?;
                println!(
                    "sdpa_vector_over_a_narrowed_kv: live={live} cap={capacity} \
                     two_pass={two_pass} max abs error {err:e}"
                );
                pinned.push(got.flatten_all()?.to_vec1::<f32>()?);
            }
            let routed = lloom::nn::ops::sdpa(&qm, &k, &v, None, false, scale as f32, 1.)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let route = lloom::nn::ops::sdpa_vector_two_pass(live, LFM2_HEAD_DIM, qm.dtype());
            let bits = |x: &[f32]| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&routed),
                bits(&pinned[usize::from(route)]),
                "live={live}: sdpa routed two_pass={route} but its output is not that kernel's"
            );
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // The tiled multi-query kernel (`steel_attention`, `call_sdpa_full`) at
    // the shapes the model runs it in: a `seq_len > 1` window attending over a
    // cache it sits at the END of (`DESIGN.md` §10.2v). A DSpark verify is
    // `q_len = K + 1` over `kv_len = index_pos + K + 1`; a prefill is
    // `q_len = kv_len`; both are GQA 4:1. The mask is the kernel's own
    // (`do_causal`, offset `qL_off = kv_len - q_len`), so the reference below
    // builds the same mask explicitly on the CPU backend.
    //
    // The sweep crosses every tile boundary the kernel has (`BQ = BK = 32`):
    // `q_len` below, at and above one block; `kv_len` unaligned, aligned and
    // a block past aligned. `kv_len % 32 == 0` with `q_len < 32` is the case
    // that separates a bounded `kb_lim` from an unbounded one — an unbounded
    // loop runs one K block past `kv_len` and skips the causal mask on the
    // block that holds the window's own future keys.
    // ---------------------------------------------------------------------

    /// repeat_kv + matmul + causal mask + softmax + matmul, in f32, with the
    /// query rows offset to the end of the key range. Mirrors what
    /// `AttnImpl::Generic` computes for a `seq_len > 1` pass over a cache.
    fn reference_causal_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        let (_, q_heads, q_len, _) = q.dims4()?;
        let (_, kv_heads, kv_len, _) = k.dims4()?;
        assert!(q_len <= kv_len, "every query must have a key");
        let repeat = q_heads / kv_heads;

        let k = repeat_kv(k, repeat)?;
        let v = repeat_kv(v, repeat)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?.contiguous()?)? * scale)?;
        let off = kv_len - q_len;
        let masked: Vec<u8> = (0..q_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > i + off)))
            .collect();
        let mask =
            Tensor::from_vec(masked, (q_len, kv_len), q.device())?.broadcast_as(att.shape())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, q.device())?.broadcast_as(att.shape())?;
        let att = mask.where_cond(&neg_inf, &att)?;
        let att = lloom::nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v.contiguous()?)
    }

    /// `(q_len, kv_len)` pairs: the verify window at unaligned, aligned and
    /// past-aligned cache lengths, one-block and multi-block prefills, and a
    /// partial trailing query block over a cache.
    const FULL_CAUSAL_SHAPES: [(usize, usize); 14] = [
        (10, 10),
        (10, 11),
        (10, 32),
        (10, 41),
        (10, 64),
        (10, 70),
        (10, 96),
        (10, 137),
        (2, 33),
        (32, 32),
        (32, 64),
        (33, 33),
        (34, 469),
        (100, 128),
    ];

    /// Metal `sdpa` (`do_causal`, no mask tensor) at GQA 4:1 over a cache,
    /// against the CPU backend, f32, across the tile-boundary sweep.
    #[test]
    fn sdpa_full_causal_gqa_over_a_cache_matches_cpu_backend_f32() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        for &(q_len, kv_len) in &FULL_CAUSAL_SHAPES {
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(675_000 + (q_len * 1000 + kv_len) as u64);

            let q = randn(&mut rng, (1, LFM2_Q_HEADS, q_len, LFM2_HEAD_DIM), &cpu)?;
            let k = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;
            let v = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;

            let want = reference_causal_attention(&q, &k, &v, scale)?;

            let got = lloom::nn::ops::sdpa(
                &q.to_device(&metal)?,
                &k.to_device(&metal)?,
                &v.to_device(&metal)?,
                None,
                true,
                scale as f32,
                1.,
            )?
            .to_device(&cpu)?;

            let err = assert_matches_reference(
                &got,
                &want,
                1e-5,
                &format!("full causal f32 q_len={q_len} kv_len={kv_len}"),
            )?;
            println!(
                "sdpa_full_causal_gqa_over_a_cache_matches_cpu_backend_f32: \
                 q_len={q_len} kv_len={kv_len} max abs error {err:e}"
            );
        }
        Ok(())
    }

    /// The same at f16, what the model runs. The kernel scales `q` in f16
    /// before the dot where the reference scales in f32 after it, so the two
    /// round differently by construction; this bounds that difference.
    #[test]
    fn sdpa_full_causal_gqa_over_a_cache_matches_cpu_backend_f16() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        for &(q_len, kv_len) in &FULL_CAUSAL_SHAPES {
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(675_500 + (q_len * 1000 + kv_len) as u64);

            let q = randn(&mut rng, (1, LFM2_Q_HEADS, q_len, LFM2_HEAD_DIM), &cpu)?;
            let k = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;
            let v = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;

            // Round-trip through f16 on both arms so the inputs are identical bits.
            let q16 = q.to_dtype(DType::F16)?;
            let k16 = k.to_dtype(DType::F16)?;
            let v16 = v.to_dtype(DType::F16)?;

            let want = reference_causal_attention(&q16, &k16, &v16, scale)?;

            let got = lloom::nn::ops::sdpa(
                &q16.to_device(&metal)?,
                &k16.to_device(&metal)?,
                &v16.to_device(&metal)?,
                None,
                true,
                scale as f32,
                1.,
            )?
            .to_device(&cpu)?
            .to_dtype(DType::F32)?;

            let err = assert_matches_reference(
                &got,
                &want,
                5e-3,
                &format!("full causal f16 q_len={q_len} kv_len={kv_len}"),
            )?;
            println!(
                "sdpa_full_causal_gqa_over_a_cache_matches_cpu_backend_f16: \
                 q_len={q_len} kv_len={kv_len} max abs error {err:e}"
            );
        }
        Ok(())
    }

    /// The cache's own shape: a `[1, kv_heads, capacity, head_dim]` slot
    /// narrowed to the live prefix, read through strides rather than copied.
    /// The padding past `kv_len` is a sentinel that dwarfs any live score, so a
    /// read past the live length shows up as a gross error rather than hiding
    /// in the tolerance. Live lengths on both sides of a 32-block boundary.
    #[test]
    fn sdpa_full_causal_over_a_narrowed_kv_matches_cpu_backend() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        const Q_LEN: usize = 10;

        for &(live, capacity) in &[(41usize, 512usize), (64, 512), (137, 512), (469, 2048)] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(675_900 + live as u64);

            let q = randn(&mut rng, (1, LFM2_Q_HEADS, Q_LEN, LFM2_HEAD_DIM), &cpu)?;
            let k_live = randn(&mut rng, (1, LFM2_KV_HEADS, live, LFM2_HEAD_DIM), &cpu)?;
            let v_live = randn(&mut rng, (1, LFM2_KV_HEADS, live, LFM2_HEAD_DIM), &cpu)?;

            let want = reference_causal_attention(&q, &k_live, &v_live, scale)?;

            let pad_len = capacity - live;
            let sentinel = 40f32;
            let k_pad = Tensor::full(sentinel, (1, LFM2_KV_HEADS, pad_len, LFM2_HEAD_DIM), &cpu)?;
            let v_pad = Tensor::full(sentinel, (1, LFM2_KV_HEADS, pad_len, LFM2_HEAD_DIM), &cpu)?;
            let k_all = Tensor::cat(&[&k_live, &k_pad], 2)?.contiguous()?;
            let v_all = Tensor::cat(&[&v_live, &v_pad], 2)?.contiguous()?;

            let k = k_all.to_device(&metal)?.narrow(2, 0, live)?;
            let v = v_all.to_device(&metal)?.narrow(2, 0, live)?;
            assert!(
                !k.is_contiguous(),
                "live={live}: the narrow must stay strided"
            );

            let got =
                lloom::nn::ops::sdpa(&q.to_device(&metal)?, &k, &v, None, true, scale as f32, 1.)?
                    .to_device(&cpu)?;

            let err = assert_matches_reference(
                &got,
                &want,
                1e-5,
                &format!("full causal narrowed kv live={live} cap={capacity}"),
            )?;
            println!(
                "sdpa_full_causal_over_a_narrowed_kv: live={live} cap={capacity} max abs error {err:e}"
            );
        }
        Ok(())
    }

    /// Run to run, the kernel is bit-stable (`DESIGN.md` §2.3.3): every
    /// reduction is fixed-order, so five dispatches of one input give one bit
    /// pattern. Asserted on the f16 verify shape, at a cache length that
    /// exercises the partial K block and the causal window together.
    #[test]
    fn sdpa_full_causal_is_bit_stable_run_to_run() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(675_777);

        let q = randn(&mut rng, (1, LFM2_Q_HEADS, 10, LFM2_HEAD_DIM), &cpu)?
            .to_dtype(DType::F16)?
            .to_device(&metal)?;
        let k = randn(&mut rng, (1, LFM2_KV_HEADS, 137, LFM2_HEAD_DIM), &cpu)?
            .to_dtype(DType::F16)?
            .to_device(&metal)?;
        let v = randn(&mut rng, (1, LFM2_KV_HEADS, 137, LFM2_HEAD_DIM), &cpu)?
            .to_dtype(DType::F16)?
            .to_device(&metal)?;

        let bits = |t: &Tensor| -> Result<Vec<u16>> {
            Ok(t.to_device(&cpu)?
                .flatten_all()?
                .to_vec1::<half::f16>()?
                .into_iter()
                .map(|x| x.to_bits())
                .collect())
        };
        let first = bits(&lloom::nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            true,
            scale as f32,
            1.,
        )?)?;
        let distinct = first.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct >= 2, "vacuous: {distinct} distinct value(s)");
        for run in 1..5 {
            let again = bits(&lloom::nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                true,
                scale as f32,
                1.,
            )?)?;
            assert!(again == first, "run {run} differs from run 0");
        }
        println!("sdpa_full_causal_is_bit_stable_run_to_run: 5 runs, {} elements, {distinct} distinct, identical", first.len());
        Ok(())
    }

    // ---------------------------------------------------------------------
    // The paged arm of the tiled kernel (`sdpa_paged`, `is_paged`; issue #708):
    // the same `steel_attention`, its K/V block loaders re-seated per 32-key
    // block through a page table. K and V are one pool `[1, kv_heads,
    // pool_tokens, head_dim]` and sequence `b`'s key `t` sits at
    // `table[b * stride + t / P] * P + t % P`. The reference is the contiguous
    // kernel over the same keys laid out in order — the SAME arithmetic in the
    // SAME block order, so the two must agree bit for bit (§2.3.5a: the
    // bit-identical case, preferred where it costs nothing) — and the eager
    // f32 reference on the CPU backend beside it, so "both wrong the same
    // way" is excluded rather than assumed.
    //
    // Every table here is a NON-IDENTITY permutation (§10.3h: a fixture that
    // does not permute its table is testing the identity function), the pool
    // holds unreferenced pages, and every slot the live keys do not occupy is a
    // sentinel that dwarfs any live score, so a block seated on the wrong page
    // or walked past its page is a gross error rather than a rounding one.
    // ---------------------------------------------------------------------

    /// A pool + permuted table holding `kv_len` live keys at `page_size`.
    struct PagedFixture {
        k_live: Tensor,
        v_live: Tensor,
        k_pool: Tensor,
        v_pool: Tensor,
        table: Vec<u32>,
        page_size: usize,
    }

    /// Lay `kv_len` keys of `[1, kv_heads, kv_len, D]` (f32, CPU) into a pool
    /// through a permuted table. `salt` moves the permutation so two sequences
    /// in one test get different tables.
    fn paged_fixture(
        rng: &mut rand::rngs::StdRng,
        kv_len: usize,
        page_size: usize,
        kv_heads: usize,
        head_dim: usize,
        salt: usize,
        cpu: &Device,
    ) -> Result<PagedFixture> {
        let n_pages = kv_len.div_ceil(page_size);
        // Two pages the table never names, so a wrong page index has
        // somewhere to land that is not another live page.
        let pool_pages = n_pages + 2;
        // A rotation, with entry 0 forced off page 0: never the identity, for
        // any n_pages >= 1 and any salt.
        let mut perm: Vec<u32> = (0..pool_pages as u32).collect();
        perm.rotate_left((1 + salt) % pool_pages);
        if perm[0] == 0 {
            perm.swap(0, 1);
        }
        let table: Vec<u32> = perm[..n_pages].to_vec();
        assert!(
            table.iter().enumerate().any(|(i, &p)| p as usize != i),
            "the fixture's table must not be the identity"
        );

        let k_live = randn(rng, (1, kv_heads, kv_len, head_dim), cpu)?;
        let v_live = randn(rng, (1, kv_heads, kv_len, head_dim), cpu)?;
        let pool_tokens = pool_pages * page_size;
        let sentinel = 40f32;
        let lay = |live: &Tensor| -> Result<Tensor> {
            let src = live.flatten_all()?.to_vec1::<f32>()?;
            let mut pool = vec![sentinel; kv_heads * pool_tokens * head_dim];
            for h in 0..kv_heads {
                for t in 0..kv_len {
                    let slot = table[t / page_size] as usize * page_size + t % page_size;
                    let s = (h * kv_len + t) * head_dim;
                    let d = (h * pool_tokens + slot) * head_dim;
                    pool[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
                }
            }
            Tensor::from_vec(pool, (1, kv_heads, pool_tokens, head_dim), cpu)
        };
        Ok(PagedFixture {
            k_pool: lay(&k_live)?,
            v_pool: lay(&v_live)?,
            k_live,
            v_live,
            table,
            page_size,
        })
    }

    fn bits_of(t: &Tensor, cpu: &Device) -> Result<Vec<u32>> {
        let t = t.to_device(cpu)?.flatten_all()?;
        Ok(match t.dtype() {
            DType::F32 => t.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect(),
            DType::F16 => t
                .to_vec1::<half::f16>()?
                .into_iter()
                .map(|x| x.to_bits() as u32)
                .collect(),
            DType::BF16 => t
                .to_vec1::<half::bf16>()?
                .into_iter()
                .map(|x| x.to_bits() as u32)
                .collect(),
            other => panic!("bits_of: {other:?}"),
        })
    }

    /// `(q_len, kv_len, page_size)`: the drafter's block (γ = 9) over a context
    /// on both sides of a page boundary and of the 32-key block, a verify
    /// window, a prefill-sized window, and page sizes 32 (one block per page,
    /// every block seated on a different page), 64 and 256 (the shipped
    /// default, where a partial last block sits mid-page).
    const PAGED_SHAPES: [(usize, usize, usize); 12] = [
        (9, 9, 32),
        (9, 41, 32),
        (9, 64, 32),
        (9, 73, 64),
        (9, 137, 64),
        (9, 300, 256),
        (9, 512, 256),
        (10, 137, 32),
        (32, 96, 32),
        (33, 265, 64),
        (34, 469, 256),
        (100, 128, 32),
    ];

    /// The paged kernel against the contiguous kernel (bit for bit) and the
    /// CPU eager reference, f32, bidirectional (`do_causal = false`, the
    /// drafter's block) and causal (a verify window over a cache).
    #[test]
    fn sdpa_paged_matches_contiguous_bit_for_bit_f32() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        for &(q_len, kv_len, page_size) in &PAGED_SHAPES {
            for causal in [false, true] {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    708_000 + (q_len * 100_000 + kv_len * 100 + page_size) as u64,
                );
                let q = randn(&mut rng, (1, LFM2_Q_HEADS, q_len, LFM2_HEAD_DIM), &cpu)?;
                let f = paged_fixture(
                    &mut rng,
                    kv_len,
                    page_size,
                    LFM2_KV_HEADS,
                    LFM2_HEAD_DIM,
                    0,
                    &cpu,
                )?;
                let want = if causal {
                    reference_causal_attention(&q, &f.k_live, &f.v_live, scale)?
                } else {
                    reference_attention(&q, &f.k_live, &f.v_live, scale)?
                };
                let qm = q.to_device(&metal)?;
                let contiguous = lloom::nn::ops::sdpa(
                    &qm,
                    &f.k_live.to_device(&metal)?,
                    &f.v_live.to_device(&metal)?,
                    None,
                    causal,
                    scale as f32,
                    1.,
                )?;
                let paged = lloom::nn::ops::sdpa_paged(
                    &qm,
                    &f.k_pool.to_device(&metal)?,
                    &f.v_pool.to_device(&metal)?,
                    causal,
                    scale as f32,
                    f.page_size,
                    &f.table,
                    f.table.len(),
                    kv_len,
                )?;
                let what = format!(
                    "paged f32 q_len={q_len} kv_len={kv_len} P={page_size} causal={causal}"
                );
                let err = assert_matches_reference(&paged.to_device(&cpu)?, &want, 1e-5, &what)?;
                assert!(
                    bits_of(&paged, &cpu)? == bits_of(&contiguous, &cpu)?,
                    "{what}: the paged kernel differs from the contiguous kernel in bits"
                );
                println!("sdpa_paged_matches_contiguous_bit_for_bit_f32: {what} max abs error vs cpu {err:e}, bit-identical to contiguous");
            }
        }
        Ok(())
    }

    /// The same at f16 and bf16 — the dtypes the drafter runs at (F16 in
    /// `lfm2-chat`, the checkpoint's BF16 elsewhere). Bit-identical to the
    /// contiguous kernel; the CPU reference bounds the f16 rounding as the
    /// contiguous tests do.
    #[test]
    fn sdpa_paged_matches_contiguous_bit_for_bit_f16_bf16() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        // The bf16 arm on purpose, where the GPU only emulates it (§15.2a).
        lloom_tensor::test_utils::allow_emulated(&metal, DType::BF16);
        let cpu = Device::Cpu;
        for &(q_len, kv_len, page_size) in &PAGED_SHAPES {
            for (dtype, causal) in [
                (DType::F16, false),
                (DType::F16, true),
                (DType::BF16, false),
            ] {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    708_500 + (q_len * 100_000 + kv_len * 100 + page_size) as u64,
                );
                let q = randn(&mut rng, (1, LFM2_Q_HEADS, q_len, LFM2_HEAD_DIM), &cpu)?;
                let f = paged_fixture(
                    &mut rng,
                    kv_len,
                    page_size,
                    LFM2_KV_HEADS,
                    LFM2_HEAD_DIM,
                    1,
                    &cpu,
                )?;
                // Round-trip through the dtype on every arm so the inputs are
                // identical bits; the pool is rounded per element the same way.
                let q16 = q.to_dtype(dtype)?;
                let k16 = f.k_live.to_dtype(dtype)?;
                let v16 = f.v_live.to_dtype(dtype)?;
                let want = if causal {
                    reference_causal_attention(&q16, &k16, &v16, scale)?
                } else {
                    reference_attention(&q16, &k16, &v16, scale)?
                };
                let qm = q16.to_device(&metal)?;
                let contiguous = lloom::nn::ops::sdpa(
                    &qm,
                    &k16.to_device(&metal)?,
                    &v16.to_device(&metal)?,
                    None,
                    causal,
                    scale as f32,
                    1.,
                )?;
                let paged = lloom::nn::ops::sdpa_paged(
                    &qm,
                    &f.k_pool.to_dtype(dtype)?.to_device(&metal)?,
                    &f.v_pool.to_dtype(dtype)?.to_device(&metal)?,
                    causal,
                    scale as f32,
                    f.page_size,
                    &f.table,
                    f.table.len(),
                    kv_len,
                )?;
                let what = format!(
                    "paged {dtype:?} q_len={q_len} kv_len={kv_len} P={page_size} causal={causal}"
                );
                let tol = if dtype == DType::BF16 { 4e-2 } else { 5e-3 };
                let err = assert_matches_reference(
                    &paged.to_device(&cpu)?.to_dtype(DType::F32)?,
                    &want,
                    tol,
                    &what,
                )?;
                assert!(
                    bits_of(&paged, &cpu)? == bits_of(&contiguous, &cpu)?,
                    "{what}: the paged kernel differs from the contiguous kernel in bits"
                );
                println!("sdpa_paged_matches_contiguous_bit_for_bit_f16_bf16: {what} max abs error vs cpu {err:e}, bit-identical to contiguous");
            }
        }
        Ok(())
    }

    /// Two sequences over ONE pool through two different tables (#148's B>1
    /// verify shape): each row is bit-identical to its own B=1 contiguous
    /// answer, and the rows differ from each other — a kernel that read every
    /// row through table 0, or the batch through a stride, fails one or the
    /// other.
    #[test]
    fn sdpa_paged_b2_reads_each_sequence_through_its_own_table() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        const Q_LEN: usize = 10;
        const KV_LEN: usize = 137;
        const P: usize = 32;
        let mut rng = rand::rngs::StdRng::seed_from_u64(708_900);
        let n_pages = KV_LEN.div_ceil(P);
        // One pool holding both sequences' keys: sequence 0 on pages
        // [0..n_pages) permuted, sequence 1 on the pages after them, permuted
        // differently. Built by hand so the two tables are disjoint and
        // neither is the identity.
        let pool_pages = 2 * n_pages + 1;
        let mut t0: Vec<u32> = (0..n_pages as u32).rev().collect();
        t0.rotate_left(1);
        let mut t1: Vec<u32> = (n_pages as u32..2 * n_pages as u32).collect();
        t1.rotate_left(2);
        t1.swap(0, n_pages - 1);
        for (i, t) in [&t0, &t1].into_iter().enumerate() {
            assert!(
                t.iter()
                    .enumerate()
                    .any(|(j, &p)| p as usize != j + i * n_pages),
                "table {i} must not be the identity over its own range"
            );
        }
        let pool_tokens = pool_pages * P;
        let sentinel = 40f32;
        let mut k_pool = vec![sentinel; LFM2_KV_HEADS * pool_tokens * LFM2_HEAD_DIM];
        let mut v_pool = k_pool.clone();
        let mut lives = Vec::new();
        for table in [&t0, &t1] {
            let k = randn(&mut rng, (1, LFM2_KV_HEADS, KV_LEN, LFM2_HEAD_DIM), &cpu)?;
            let v = randn(&mut rng, (1, LFM2_KV_HEADS, KV_LEN, LFM2_HEAD_DIM), &cpu)?;
            for (live, pool) in [(&k, &mut k_pool), (&v, &mut v_pool)] {
                let src = live.flatten_all()?.to_vec1::<f32>()?;
                for h in 0..LFM2_KV_HEADS {
                    for t in 0..KV_LEN {
                        let slot = table[t / P] as usize * P + t % P;
                        let s = (h * KV_LEN + t) * LFM2_HEAD_DIM;
                        let d = (h * pool_tokens + slot) * LFM2_HEAD_DIM;
                        pool[d..d + LFM2_HEAD_DIM].copy_from_slice(&src[s..s + LFM2_HEAD_DIM]);
                    }
                }
            }
            lives.push((k, v));
        }
        let k_pool =
            Tensor::from_vec(k_pool, (1, LFM2_KV_HEADS, pool_tokens, LFM2_HEAD_DIM), &cpu)?
                .to_device(&metal)?;
        let v_pool =
            Tensor::from_vec(v_pool, (1, LFM2_KV_HEADS, pool_tokens, LFM2_HEAD_DIM), &cpu)?
                .to_device(&metal)?;
        let q = randn(&mut rng, (2, LFM2_Q_HEADS, Q_LEN, LFM2_HEAD_DIM), &cpu)?;

        let mut table = t0.clone();
        table.extend_from_slice(&t1);
        for causal in [false, true] {
            let got = lloom::nn::ops::sdpa_paged(
                &q.to_device(&metal)?,
                &k_pool,
                &v_pool,
                causal,
                scale as f32,
                P,
                &table,
                n_pages,
                KV_LEN,
            )?;
            let mut row_bits = Vec::new();
            for (b, (k, v)) in lives.iter().enumerate() {
                let solo = lloom::nn::ops::sdpa(
                    &q.narrow(0, b, 1)?.to_device(&metal)?,
                    &k.to_device(&metal)?,
                    &v.to_device(&metal)?,
                    None,
                    causal,
                    scale as f32,
                    1.,
                )?;
                let want = if causal {
                    reference_causal_attention(&q.narrow(0, b, 1)?, k, v, scale)?
                } else {
                    reference_attention(&q.narrow(0, b, 1)?, k, v, scale)?
                };
                let row = got.narrow(0, b, 1)?.contiguous()?;
                let what = format!("paged B=2 row {b} causal={causal}");
                assert_matches_reference(&row.to_device(&cpu)?, &want, 1e-5, &what)?;
                let bits = bits_of(&row, &cpu)?;
                assert!(
                    bits == bits_of(&solo, &cpu)?,
                    "{what}: differs from its own B=1 answer"
                );
                row_bits.push(bits);
            }
            assert!(row_bits[0] != row_bits[1], "causal={causal}: the two rows are identical, so the test cannot tell the tables apart");
            println!("sdpa_paged_b2_reads_each_sequence_through_its_own_table: causal={causal} both rows bit-identical to their B=1 contiguous answers");
        }
        Ok(())
    }

    /// Three sequences of DIFFERENT live lengths over one pool in ONE
    /// dispatch (`sdpa_paged_ragged`, `batched-decode-design.md` §12.2 change
    /// 3): every row bit-identical to its own B=1 `sdpa_paged` call over its
    /// own table, and to the contiguous kernel on the same keys — causal (a
    /// verify window at the end of each row's own key range) and not (the
    /// drafter's block). The lengths straddle the 32-key block differently
    /// (one aligned, two not), so the per-row partial block is exercised in
    /// the same dispatch as an aligned row.
    #[test]
    fn sdpa_paged_ragged_rows_read_their_own_live_counts_bit_for_bit() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        const Q_LEN: usize = 10;
        const P: usize = 32;
        let kv_lens = [45usize, 128, 301];
        let mut rng = rand::rngs::StdRng::seed_from_u64(148_301);
        let pages: Vec<usize> = kv_lens.iter().map(|n| n.div_ceil(P)).collect();
        let stride = *pages.iter().max().unwrap();
        let pool_pages: usize = pages.iter().sum::<usize>() + 2;
        // Disjoint, permuted tables: row r's pages are a rotated slice of the
        // pool's pages after the rows before it, never the identity.
        let mut tables: Vec<Vec<u32>> = Vec::new();
        let mut base = 1u32;
        for (r, &n) in pages.iter().enumerate() {
            let mut t: Vec<u32> = (base..base + n as u32).collect();
            // One rotation per row plus one: never the identity at n >= 2,
            // and a different permutation per row.
            t.rotate_left((r + 1) % n.max(1));
            if t.iter()
                .enumerate()
                .all(|(j, &p)| p as usize == j + base as usize)
            {
                t.rotate_left(1);
            }
            assert!(
                t.iter()
                    .enumerate()
                    .any(|(j, &p)| p as usize != j + base as usize),
                "row {r}'s table must not be the identity over its range"
            );
            tables.push(t);
            base += n as u32;
        }
        let pool_tokens = pool_pages * P;
        let sentinel = 40f32;
        let mut k_pool = vec![sentinel; LFM2_KV_HEADS * pool_tokens * LFM2_HEAD_DIM];
        let mut v_pool = k_pool.clone();
        let mut lives = Vec::new();
        for (r, &kv_len) in kv_lens.iter().enumerate() {
            let k = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;
            let v = randn(&mut rng, (1, LFM2_KV_HEADS, kv_len, LFM2_HEAD_DIM), &cpu)?;
            for (live, pool) in [(&k, &mut k_pool), (&v, &mut v_pool)] {
                let src = live.flatten_all()?.to_vec1::<f32>()?;
                for h in 0..LFM2_KV_HEADS {
                    for t in 0..kv_len {
                        let slot = tables[r][t / P] as usize * P + t % P;
                        let s = (h * kv_len + t) * LFM2_HEAD_DIM;
                        let d = (h * pool_tokens + slot) * LFM2_HEAD_DIM;
                        pool[d..d + LFM2_HEAD_DIM].copy_from_slice(&src[s..s + LFM2_HEAD_DIM]);
                    }
                }
            }
            lives.push((k, v));
        }
        let k_pool =
            Tensor::from_vec(k_pool, (1, LFM2_KV_HEADS, pool_tokens, LFM2_HEAD_DIM), &cpu)?
                .to_device(&metal)?;
        let v_pool =
            Tensor::from_vec(v_pool, (1, LFM2_KV_HEADS, pool_tokens, LFM2_HEAD_DIM), &cpu)?
                .to_device(&metal)?;
        let q = randn(
            &mut rng,
            (kv_lens.len(), LFM2_Q_HEADS, Q_LEN, LFM2_HEAD_DIM),
            &cpu,
        )?;

        let mut ragged_table = Vec::with_capacity(kv_lens.len() * stride);
        for t in &tables {
            let mut t = t.clone();
            t.resize(stride, 0);
            ragged_table.extend(t);
        }
        let n_keys: Vec<u32> = kv_lens.iter().map(|&n| n as u32).collect();
        for causal in [false, true] {
            let got = lloom::nn::ops::sdpa_paged_ragged(
                &q.to_device(&metal)?,
                &k_pool,
                &v_pool,
                causal,
                scale as f32,
                P,
                &ragged_table,
                stride,
                &n_keys,
            )?;
            let mut row_bits = Vec::new();
            for (b, (k, v)) in lives.iter().enumerate() {
                let q_b = q.narrow(0, b, 1)?;
                let own = lloom::nn::ops::sdpa_paged(
                    &q_b.to_device(&metal)?,
                    &k_pool,
                    &v_pool,
                    causal,
                    scale as f32,
                    P,
                    &tables[b],
                    tables[b].len(),
                    kv_lens[b],
                )?;
                let solo = lloom::nn::ops::sdpa(
                    &q_b.to_device(&metal)?,
                    &k.to_device(&metal)?,
                    &v.to_device(&metal)?,
                    None,
                    causal,
                    scale as f32,
                    1.,
                )?;
                let want = if causal {
                    reference_causal_attention(&q_b, k, v, scale)?
                } else {
                    reference_attention(&q_b, k, v, scale)?
                };
                let row = got.narrow(0, b, 1)?.contiguous()?;
                let what = format!("paged ragged row {b} (kv {}) causal={causal}", kv_lens[b]);
                assert_matches_reference(&row.to_device(&cpu)?, &want, 1e-5, &what)?;
                let bits = bits_of(&row, &cpu)?;
                assert!(
                    bits == bits_of(&own, &cpu)?,
                    "{what}: differs from its own B=1 paged answer"
                );
                assert!(
                    bits == bits_of(&solo, &cpu)?,
                    "{what}: differs from the contiguous kernel"
                );
                row_bits.push(bits);
            }
            assert!(
                row_bits[0] != row_bits[1] && row_bits[1] != row_bits[2],
                "causal={causal}: two rows are identical, so the test cannot tell the counts apart"
            );
            println!(
                "sdpa_paged_ragged_rows_read_their_own_live_counts_bit_for_bit: causal={causal} \
                 rows {kv_lens:?} each bit-identical to its own B=1 paged and contiguous answers"
            );
        }
        Ok(())
    }

    /// Run to run, the paged kernel is bit-stable (`DESIGN.md` §2.3.3): the
    /// page walk changes addresses, never the reduction order.
    #[test]
    fn sdpa_paged_is_bit_stable_run_to_run() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(708_777);
        let q = randn(&mut rng, (1, LFM2_Q_HEADS, 9, LFM2_HEAD_DIM), &cpu)?
            .to_dtype(DType::F16)?
            .to_device(&metal)?;
        let f = paged_fixture(&mut rng, 137, 32, LFM2_KV_HEADS, LFM2_HEAD_DIM, 2, &cpu)?;
        let k = f.k_pool.to_dtype(DType::F16)?.to_device(&metal)?;
        let v = f.v_pool.to_dtype(DType::F16)?.to_device(&metal)?;
        let run = || {
            lloom::nn::ops::sdpa_paged(
                &q,
                &k,
                &v,
                false,
                scale as f32,
                32,
                &f.table,
                f.table.len(),
                137,
            )
        };
        let first = bits_of(&run()?, &cpu)?;
        let distinct = first.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct >= 2, "vacuous: {distinct} distinct value(s)");
        for r in 1..5 {
            assert!(
                bits_of(&run()?, &cpu)? == first,
                "run {r} differs from run 0"
            );
        }
        println!("sdpa_paged_is_bit_stable_run_to_run: 5 runs, {} elements, {distinct} distinct, identical", first.len());
        Ok(())
    }

    /// The refusals: a decode shape, a page that does not hold a whole key
    /// block, and a table naming a page past the pool are errors, never a
    /// plausible answer (§3.5).
    #[test]
    fn sdpa_paged_refuses_what_the_kernel_cannot_read() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(708_666);
        let f = paged_fixture(&mut rng, 41, 32, LFM2_KV_HEADS, LFM2_HEAD_DIM, 0, &cpu)?;
        let k = f.k_pool.to_device(&metal)?;
        let v = f.v_pool.to_device(&metal)?;
        let q9 = randn(&mut rng, (1, LFM2_Q_HEADS, 9, LFM2_HEAD_DIM), &cpu)?.to_device(&metal)?;
        let q1 = randn(&mut rng, (1, LFM2_Q_HEADS, 1, LFM2_HEAD_DIM), &cpu)?.to_device(&metal)?;
        let ok = lloom::nn::ops::sdpa_paged(
            &q9,
            &k,
            &v,
            false,
            scale as f32,
            32,
            &f.table,
            2,
            41,
        );
        assert!(ok.is_ok(), "the control must dispatch: {:?}", ok.err());
        let decode = lloom::nn::ops::sdpa_paged(
            &q1,
            &k,
            &v,
            false,
            scale as f32,
            32,
            &f.table,
            2,
            41,
        );
        assert!(decode.is_err(), "q_seq == 1 must be refused");
        let bad_page = lloom::nn::ops::sdpa_paged(
            &q9,
            &k,
            &v,
            false,
            scale as f32,
            48,
            &f.table,
            2,
            41,
        );
        assert!(
            bad_page.is_err(),
            "page_size 48 is not a multiple of the key block"
        );
        let past =
            lloom::nn::ops::sdpa_paged(&q9, &k, &v, false, scale as f32, 32, &[0, 99], 2, 41);
        assert!(past.is_err(), "a page past the pool must be refused");
        let short =
            lloom::nn::ops::sdpa_paged(&q9, &k, &v, false, scale as f32, 32, &[0], 1, 41);
        assert!(
            short.is_err(),
            "a table shorter than the live pages must be refused"
        );
        Ok(())
    }
}
