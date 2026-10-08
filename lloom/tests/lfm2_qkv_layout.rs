//! The `QkvLayout` axis (#727, `DESIGN.md` §6.2c, §12.2 row 6): the three
//! arms compute the same logits, bit for bit, on a multi-token pass (where
//! the `transpose` views are strided and the copies are real) and on the
//! decode steps after it; and the dispatch census shows what each arm
//! replaced -- the norm + rope pair fused on `Fused`, the transposed copies
//! gone on `Strided` and `Fused`.
//!
//! The weights are the tiny LFM2's (`lloom::testing::tiny`, keyed by
//! seed, name and index), so a difference between two arms is the arms'. The
//! arms agreeing is a shape; the absolute reference beside it is, on Metal,
//! the CPU backend within a tolerance (CONTRIBUTING §3.1b), so a defect every
//! Metal arm shares moves it.
//!
//! There is no absolute fingerprint of the CPU logits: they are not one value
//! across machines. The CPU matmul is the `gemm` crate's, which fuses `a·b+c`
//! on NEON and x86-FMA and not on SSE, and picks its kernel by the product's
//! shape and the CPU; `expf` is the platform's libm; the reductions are
//! NEON, AVX2 or scalar by target feature. Four CPUs gave four fingerprints
//! of this model at one commit, every arm agreeing on each
//! (`design/measurements/issue-939.md` §2.1). The CPU half of the test is the
//! arms' agreement, bit for bit, on whatever CPU runs it.

use lloom::models::lfm2::{AttnImpl, Config, KvAppend, LayerType, Model, QkvLayout, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

fn tiny_config(qkv_layout: QkvLayout, attn_impl: AttnImpl) -> Config {
    Config {
        vocab_size: 64,
        hidden_size: 512,
        intermediate_size: 512,
        num_hidden_layers: 3,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl,
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state: Default::default(),
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        qkv_layout,
        ..Default::default()
    }
}

/// The tiny LFM2 at `cfg`'s geometry and arms, computing in `dtype`.
fn build(cfg: &Config, dev: &Device, dtype: DType) -> Result<Model<CrownExact>> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(dtype)
        .model::<CrownExact>(dev)
}

/// The CPU backend's logits for `cfg` at f32, over the weights a `dtype`
/// model holds (rounded through `dtype`): the reference an arm is held to.
fn cpu_reference(cfg: &Config, dtype: DType) -> Result<Vec<Vec<f32>>> {
    let dense = if dtype == DType::F16 {
        GgmlDType::F16
    } else {
        GgmlDType::F32
    };
    let mut cpu = cfg.clone();
    cpu.attn_impl = AttnImpl::Generic;
    let model = TinyCheckpoint::<Lfm2Tiny>::new(
        cpu.clone(),
        Weights {
            dense,
            ..Weights::default()
        },
    )
    .with_dtype(DType::F32)
    .model::<CrownExact>(&Device::Cpu)?;
    Ok(steps(&model, &cpu, &Device::Cpu, DType::F32)?
        .into_iter()
        .map(|s| s.into_iter().map(f32::from_bits).collect())
        .collect())
}

/// A 5-token prefill (strided views, real copies on `Contiguous`), then two
/// decode steps (contiguous views); the logits of every step as f32 bits.
fn run(cfg: &Config, dev: &Device, dtype: DType) -> Result<Vec<Vec<u32>>> {
    let model = build(cfg, dev, dtype)?;
    steps(&model, cfg, dev, dtype)
}

fn steps(
    model: &Model<CrownExact>,
    cfg: &Config,
    dev: &Device,
    dtype: DType,
) -> Result<Vec<Vec<u32>>> {
    let mut cache = State::<CrownExact>::new(true, dtype, cfg, dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let mut out = Vec::new();
    let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
    let logits = model.forward(&inp, 0, &mut cache)?;
    out.push(bits(&logits)?);
    for (i, tok) in [7u32, 9u32].into_iter().enumerate() {
        let step = Tensor::new(&[tok], dev)?.reshape((1, 1))?;
        let logits = model.forward(&step, prompt.len() + i, &mut cache)?;
        out.push(bits(&logits)?);
    }
    Ok(out)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn arms_agree_on(dev: &Device, attn_impl: AttnImpl, dtype: DType) -> Result<()> {
    let reference = run(&tiny_config(QkvLayout::Contiguous, attn_impl), dev, dtype)?;
    // Non-vacuity: the steps compute something, and different things.
    assert!(reference[0].iter().any(|b| *b != 0));
    assert_ne!(
        reference[1], reference[2],
        "two decode steps must not agree"
    );
    // The absolute reference: the CPU backend within the dtype's
    // tolerance, over the same (rounded) weights.
    let want = cpu_reference(&tiny_config(QkvLayout::Contiguous, attn_impl), dtype)?;
    let tol = if dtype == DType::F16 { 4e-3 } else { 1e-4 };
    for (step, (g, r)) in reference.iter().zip(&want).enumerate() {
        let scale = r.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = g
            .iter()
            .zip(r)
            .map(|(a, b)| (f32::from_bits(*a) - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{dev:?} {attn_impl:?} {dtype:?} step {step}: |Contiguous - CPU| {d:e} at scale {scale:e}");
        assert!(
            d <= tol * scale,
            "Contiguous on {dev:?} {attn_impl:?} {dtype:?}: step {step} is {d:e} from the CPU backend \
             at scale {scale:e} (tolerance {tol} x scale)"
        );
    }
    for arm in [QkvLayout::Strided, QkvLayout::Fused] {
        let got = run(&tiny_config(arm, attn_impl), dev, dtype)?;
        for (step, (g, r)) in got.iter().zip(&reference).enumerate() {
            let diff = g.iter().zip(r).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff,
                0,
                "{arm:?} vs Contiguous on {dev:?} {attn_impl:?} {dtype:?}: step {step} \
                 differs in {diff} of {} logits",
                r.len()
            );
        }
    }
    Ok(())
}

/// Every arm leaves the logits where `Contiguous` has them, on the CPU
/// (where `Strided` materialises and `Fused` composes) and on Metal (where
/// the strided and fused kernels run), on both attention arms at f16.
#[test]
fn every_arm_computes_the_same_logits() -> Result<()> {
    arms_agree_on(&Device::Cpu, AttnImpl::Generic, DType::F32)?;
    #[cfg(feature = "metal")]
    {
        let dev = Device::new_metal(0)?;
        arms_agree_on(&dev, AttnImpl::Generic, DType::F16)?;
        arms_agree_on(&dev, AttnImpl::Sdpa, DType::F16)?;
        arms_agree_on(&dev, AttnImpl::Sdpa, DType::F32)?;
    }
    Ok(())
}

/// A batched-rows pass takes the per-row rotation on every arm (`Fused`
/// degrades to `Strided` there), and agrees with the batch-of-one path.
#[cfg(feature = "metal")]
#[test]
fn the_rows_path_agrees_with_the_single_path_on_every_arm() -> Result<()> {
    use lloom::nn::model::Row;
    let dev = Device::new_metal(0)?;
    for arm in [QkvLayout::Contiguous, QkvLayout::Strided, QkvLayout::Fused] {
        let cfg = tiny_config(arm, AttnImpl::Sdpa);
        let model = build(&cfg, &dev, DType::F16)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), &dev)?.reshape((1, prompt.len()))?;
        // Two caches, same prefill, then one decode step through each path.
        let mut c1 = State::<CrownExact>::new(true, DType::F16, &cfg, &dev)?;
        let mut c2 = State::<CrownExact>::new(true, DType::F16, &cfg, &dev)?;
        model.forward(&inp, 0, &mut c1)?;
        model.forward(&inp, 0, &mut c2)?;
        let step = Tensor::new(&[7u32], &dev)?.reshape((1, 1))?;
        let single = bits(&model.forward(&step, prompt.len(), &mut c1)?)?;
        let mut rows = [Row {
            pos: prompt.len(),
            cache: &mut c2,
        }];
        let batched = bits(&model.forward_rows(&step, &mut rows)?)?;
        assert_eq!(
            single, batched,
            "{arm:?}: rows path differs from the single path"
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
mod census {
    use super::*;

    /// The per-label dispatch census of one forward pass of `width` tokens
    /// at `pos`, after a 5-token prefill (a decode step at `width == 1`, a
    /// second multi-token pass at `width > 1`), from `dev`'s own census, so a
    /// test running beside it on another device lands nothing in it. Labels
    /// are the `[[host_name]]`s the census records. `both`, when given, is
    /// met with the window open, before and after the pass, and kept on
    /// failure too (`Rendezvous`), so a peer thread is never left waiting.
    fn census_of(
        cfg: &Config,
        dev: &Device,
        width: usize,
        both: Option<&std::sync::Barrier>,
    ) -> Result<Vec<(String, u64)>> {
        let mut meet = both.map(|b| lloom_tensor::test_utils::Rendezvous::new(b, 2));
        let model = build(cfg, dev, DType::F16)?;
        let mut cache = State::<CrownExact>::new(true, DType::F16, cfg, dev)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
        model.forward(&inp, 0, &mut cache)?;
        dev.synchronize()?;
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("one census window on a fresh device");
        if let Some(m) = meet.as_mut() {
            m.meet();
        }
        let toks: Vec<u32> = (0..width as u32).map(|i| 7 + i).collect();
        let step = Tensor::new(toks.as_slice(), dev)?.reshape((1, width))?;
        let logits = model.forward(&step, prompt.len(), &mut cache)?;
        logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;
        if let Some(m) = meet.as_mut() {
            m.meet();
        }
        let mut by_label = window.close();
        by_label.sort();
        Ok(by_label)
    }

    fn count(census: &[(String, u64)], label: &str) -> u64 {
        census
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, n)| *n)
            .sum()
    }

    fn total(census: &[(String, u64)]) -> u64 {
        census.iter().map(|(_, n)| *n).sum()
    }

    fn dump(tag: &str, census: &[(String, u64)]) {
        eprintln!("--- census: {tag}");
        for (label, n) in census {
            eprintln!("    {n:>4}  {label}");
        }
    }

    /// `Fused`: `rmsnorm_rope_f16` appears, `rope_f16` disappears, and
    /// `rmsnorm_f16` falls by the two QK norms per attention layer -- the
    /// fused kernel REPLACES the pair. `Strided` at decode is the contiguous
    /// path (the views are contiguous, §6.2c), so its census is
    /// `Contiguous`'s, label for label.
    #[test]
    fn the_decode_census_shows_the_fused_norm_replacing_the_pair() -> Result<()> {
        if !lloom_tensor::metal_backend::profile::enabled() {
            eprintln!("SKIPPED: needs CANDLE_METAL_PROFILE=1 (the census is off otherwise)");
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let l = 2u64;
        let contiguous = census_of(
            &tiny_config(QkvLayout::Contiguous, AttnImpl::Sdpa),
            &dev,
            1,
            None,
        )?;
        let strided = census_of(
            &tiny_config(QkvLayout::Strided, AttnImpl::Sdpa),
            &dev,
            1,
            None,
        )?;
        let fused = census_of(
            &tiny_config(QkvLayout::Fused, AttnImpl::Sdpa),
            &dev,
            1,
            None,
        )?;
        dump("Contiguous decode", &contiguous);
        dump("Fused decode", &fused);
        assert!(total(&contiguous) > 0, "blind census");
        assert_eq!(
            count(&contiguous, "rope_f16"),
            2 * l,
            "rope per attention layer"
        );
        assert_eq!(count(&contiguous, "rmsnorm_rope_f16"), 0);
        assert_eq!(count(&contiguous, "copy_nd_f16"), 0);
        assert_eq!(
            strided, contiguous,
            "Strided at seq_len 1 must be the contiguous path"
        );
        assert_eq!(
            count(&fused, "rope_f16"),
            0,
            "Fused: no separate rope dispatch"
        );
        assert_eq!(count(&fused, "rmsnorm_rope_f16"), 2 * l);
        assert_eq!(
            count(&fused, "rmsnorm_f16") + 2 * l,
            count(&contiguous, "rmsnorm_f16"),
            "Fused: the two QK norms per attention layer leave the rmsnorm count"
        );
        assert_eq!(
            total(&contiguous) - total(&fused),
            2 * l,
            "Fused removes exactly two dispatches per attention layer at decode"
        );
        Ok(())
    }

    /// Two decode censuses taken AT ONCE, on two devices from two threads,
    /// both windows open across both passes: each reads exactly what it reads
    /// alone, so a census test here holds for any test beside it. The
    /// absolute references are the rope counts `Fused` moves (2 per attention
    /// layer, and none); the process-wide census over the same span is the
    /// control that saw both passes.
    #[test]
    fn two_censuses_taken_at_once_on_two_devices_see_only_their_own() -> Result<()> {
        use lloom_tensor::metal_backend::profile;
        if !profile::enabled() {
            eprintln!("SKIPPED: needs CANDLE_METAL_PROFILE=1 (the census is off otherwise)");
            return Ok(());
        }
        let l = 2u64;
        let (c_cfg, f_cfg) = (
            tiny_config(QkvLayout::Contiguous, AttnImpl::Sdpa),
            tiny_config(QkvLayout::Fused, AttnImpl::Sdpa),
        );
        let alone_c = census_of(&c_cfg, &Device::new_metal(0)?, 1, None)?;
        let alone_f = census_of(&f_cfg, &Device::new_metal(0)?, 1, None)?;
        let rope_before = count(&profile::snapshot().by_label, "rmsnorm_rope_f16");
        let both = std::sync::Barrier::new(2);
        let (c, f) = std::thread::scope(|s| {
            let c = s.spawn(|| census_of(&c_cfg, &Device::new_metal(0)?, 1, Some(&both)));
            let f = s.spawn(|| census_of(&f_cfg, &Device::new_metal(0)?, 1, Some(&both)));
            (c.join().unwrap(), f.join().unwrap())
        });
        let (c, f) = (c?, f?);
        let rope_after = count(&profile::snapshot().by_label, "rmsnorm_rope_f16");
        assert!(
            rope_after - rope_before >= 2 * l,
            "control: the process-wide census saw the Fused pass ({} fused norms)",
            rope_after - rope_before
        );
        assert_eq!(count(&c, "rope_f16"), 2 * l, "Contiguous, beside Fused");
        assert_eq!(count(&c, "rmsnorm_rope_f16"), 0, "Contiguous, beside Fused");
        assert_eq!(count(&f, "rope_f16"), 0, "Fused, beside Contiguous");
        assert_eq!(
            count(&f, "rmsnorm_rope_f16"),
            2 * l,
            "Fused, beside Contiguous"
        );
        assert_eq!(c, alone_c, "Contiguous: at once vs alone");
        assert_eq!(f, alone_f, "Fused: at once vs alone");
        Ok(())
    }

    /// A multi-token pass: `Contiguous` copies the three transposed views
    /// per attention layer (`copy_f16_strided`) and `slice_set` writes `V`
    /// with a `copy2d_f16`; `Strided` norms the views where they lie
    /// (`rmsnorm_f16_strided`) and writes `V` with one `copy_nd_f16`.
    #[test]
    fn the_multi_token_census_shows_the_copies_gone() -> Result<()> {
        if !lloom_tensor::metal_backend::profile::enabled() {
            eprintln!("SKIPPED: needs CANDLE_METAL_PROFILE=1 (the census is off otherwise)");
            return Ok(());
        }
        let dev = Device::new_metal(0)?;
        let l = 2u64;
        let contiguous = census_of(
            &tiny_config(QkvLayout::Contiguous, AttnImpl::Sdpa),
            &dev,
            4,
            None,
        )?;
        let strided = census_of(
            &tiny_config(QkvLayout::Strided, AttnImpl::Sdpa),
            &dev,
            4,
            None,
        )?;
        let fused = census_of(
            &tiny_config(QkvLayout::Fused, AttnImpl::Sdpa),
            &dev,
            4,
            None,
        )?;
        dump("Contiguous seq 4", &contiguous);
        dump("Strided seq 4", &strided);
        dump("Fused seq 4", &fused);
        assert!(total(&contiguous) > 0, "blind census");
        let c = |census: &[(String, u64)], label: &str| count(census, label);
        // The three transposed copies per attention layer leave ...
        assert_eq!(
            c(&contiguous, "copy_f16_strided") - c(&strided, "copy_f16_strided"),
            3 * l,
            "Strided: three strided copies per attention layer leave"
        );
        // ... the norms take the strided kernel ...
        assert_eq!(c(&contiguous, "rmsnorm_f16_strided"), 0);
        assert_eq!(c(&strided, "rmsnorm_f16_strided"), 2 * l);
        assert_eq!(
            c(&contiguous, "rmsnorm_f16") - c(&strided, "rmsnorm_f16"),
            2 * l
        );
        // ... and the V append is one `copy_nd` where it was a `copy2d`.
        assert_eq!(c(&contiguous, "copy_nd_f16"), 0);
        assert_eq!(c(&strided, "copy_nd_f16"), l);
        assert_eq!(c(&contiguous, "copy2d_f16") - c(&strided, "copy2d_f16"), l);
        assert_eq!(
            total(&contiguous) - total(&strided),
            3 * l,
            "Strided removes three dispatches per attention layer on a multi-token pass"
        );
        assert_eq!(c(&contiguous, "rope_f16"), 2 * l);
        assert_eq!(c(&strided, "rope_f16"), 2 * l);
        assert_eq!(c(&fused, "rope_f16"), 0);
        assert_eq!(c(&fused, "rmsnorm_rope_f16_strided"), 2 * l);
        assert_eq!(c(&fused, "rmsnorm_f16_strided"), 0);
        assert_eq!(
            total(&strided) - total(&fused),
            2 * l,
            "Fused removes the two rope dispatches per attention layer on top of Strided"
        );
        Ok(())
    }
}
