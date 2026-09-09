//! The `QkvLayout` axis (#727, `DESIGN.md` §6.2c, §12.2 row 6): the three
//! arms compute the same logits, bit for bit, on a multi-token pass (where
//! the `transpose` views are strided and the copies are real) and on the
//! decode steps after it; and the dispatch census shows what each arm
//! replaced -- the norm + rope pair fused on `Fused`, the transposed copies
//! gone on `Strided` and `Fused`.
//!
//! Same tiny-model fixture as `lfm2_gemv_epilogue_census.rs`: a fixed ramp
//! for every weight, so a difference between two arms is the arms'.

use lloom::models::lfm2::{AttnImpl, Cache, Config, KvAppend, LayerType, Model, QkvLayout};
use lloom::nn::VarBuilder;
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

fn build(cfg: &Config, dev: &Device, dtype: DType) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, dtype, dev);
    Model::new(cfg, vb)
}

/// A 5-token prefill (strided views, real copies on `Contiguous`), then two
/// decode steps (contiguous views); the logits of every step as f32 bits.
fn run(cfg: &Config, dev: &Device, dtype: DType) -> Result<Vec<Vec<u32>>> {
    let model = build(cfg, dev, dtype)?;
    let mut cache = Cache::new(true, dtype, cfg, dev)?;
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
        let mut c1 = Cache::new(true, DType::F16, &cfg, &dev)?;
        let mut c2 = Cache::new(true, DType::F16, &cfg, &dev)?;
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
    use std::sync::Mutex;

    /// The dispatch recorder is process-global, so the two census tests
    /// must not run concurrently: one's dispatches would land in the
    /// other's window. Held for the whole of each test.
    static RECORDER: Mutex<()> = Mutex::new(());

    /// The per-label dispatch census of one forward pass of `width` tokens
    /// at `pos`, after a 5-token prefill has been taken and the counters
    /// reset (a decode step at `width == 1`, a second multi-token pass at
    /// `width > 1`). Labels are the `[[host_name]]`s the census records.
    fn census_of(cfg: &Config, dev: &Device, width: usize) -> Result<Vec<(String, u64)>> {
        let model = build(cfg, dev, DType::F16)?;
        let mut cache = Cache::new(true, DType::F16, cfg, dev)?;
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let inp = Tensor::new(prompt.as_slice(), dev)?.reshape((1, prompt.len()))?;
        model.forward(&inp, 0, &mut cache)?;
        dev.synchronize()?;
        lloom_tensor::metal_backend::profile::reset();
        let toks: Vec<u32> = (0..width as u32).map(|i| 7 + i).collect();
        let step = Tensor::new(toks.as_slice(), dev)?.reshape((1, width))?;
        let logits = model.forward(&step, prompt.len(), &mut cache)?;
        logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;
        let mut by_label = lloom_tensor::metal_backend::profile::snapshot().by_label;
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
        let _serial = RECORDER.lock().unwrap();
        let dev = Device::new_metal(0)?;
        let l = 2u64;
        let contiguous = census_of(&tiny_config(QkvLayout::Contiguous, AttnImpl::Sdpa), &dev, 1)?;
        let strided = census_of(&tiny_config(QkvLayout::Strided, AttnImpl::Sdpa), &dev, 1)?;
        let fused = census_of(&tiny_config(QkvLayout::Fused, AttnImpl::Sdpa), &dev, 1)?;
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
        let _serial = RECORDER.lock().unwrap();
        let dev = Device::new_metal(0)?;
        let l = 2u64;
        let contiguous = census_of(&tiny_config(QkvLayout::Contiguous, AttnImpl::Sdpa), &dev, 4)?;
        let strided = census_of(&tiny_config(QkvLayout::Strided, AttnImpl::Sdpa), &dev, 4)?;
        let fused = census_of(&tiny_config(QkvLayout::Fused, AttnImpl::Sdpa), &dev, 4)?;
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
