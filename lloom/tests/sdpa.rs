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
    /// **Both sides of the two-pass threshold.** `Sdpa::metal_fwd` routes to
    /// `call_sdpa_vector_2pass` at `k_seq >= 1024` (`candle-nn/src/ops.rs`), so
    /// the single-pass and chunked kernels are different code over the same
    /// input. The `live` values below straddle it deliberately, and the padding
    /// past `live` is filled with a large sentinel rather than zeros: zeros
    /// would contribute `exp(0)` mass that a wrong length might hide, where a
    /// sentinel makes an over-read dominate the softmax and show up as a gross
    /// error.
    #[test]
    fn sdpa_vector_over_a_narrowed_kv_matches_cpu_backend() -> Result<()> {
        let scale: f64 = f64::from(LFM2_HEAD_DIM as u32).sqrt().recip();
        let metal = Device::new_metal(0)?;
        let cpu = Device::Cpu;

        // 137 and 1100 sit either side of the 1024 two-pass threshold; 1024
        // itself is the boundary case, which is the one an off-by-one moves.
        for &(live, capacity) in &[(137usize, 512usize), (1024, 2048), (1100, 2048)] {
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

            let got =
                lloom::nn::ops::sdpa(&q.to_device(&metal)?, &k, &v, None, false, scale as f32, 1.)?
                    .to_device(&cpu)?;

            let err = assert_matches_reference(
                &got,
                &want,
                2e-5,
                &format!("narrowed kv live={live} cap={capacity}"),
            )?;
            println!(
                "sdpa_vector_over_a_narrowed_kv: live={live} cap={capacity} \
                 two_pass={} max abs error {err:e}",
                live >= 1024
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
}
