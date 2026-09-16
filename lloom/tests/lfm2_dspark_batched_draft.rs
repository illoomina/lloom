//! The batched draft (issue #148, `batched-decode-design.md` §12.9): one
//! drafter pass over every active sequence's block computes, per row, what
//! the one-row draft computes for that row alone.
//!
//! The batched pass runs the projections, norms and MLPs at `m = B·γ` and,
//! on a shared paged pool, the block attention of all rows in ONE
//! `sdpa_paged_ragged` dispatch; the reference for each row is the one-row
//! path (`block_hidden`, `draft` over a single sequence) — a different code
//! path over the same caches, not the subject restated (`CONTRIBUTING.md`
//! §3.1b). Rows have different context lengths and different anchors, so a
//! batched pass that read one row's offset, one row's table or one row's
//! live count for every row fails.
//!
//! Metal only: the shared paged pool is Metal, and the drafted ids are
//! compared on the arm the serving loop runs.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::models::lfm2_dspark::{CtxKv, DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::speculator::{DraftTokens, SeqId, Speculator};
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand_distr::Distribution;
use std::collections::HashMap;

const PAGE: usize = 32;
const HIDDEN: usize = 64;
const VOCAB: usize = 48;

/// The drafter: two layers over the target's `[1, 3]` taps, γ = 9, a Markov
/// head and a confidence head (both batched arms exercised).
fn drafter_cfg() -> DSparkConfig {
    DSparkConfig {
        hidden_size: HIDDEN,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 96,
        vocab_size: VOCAB,
        rms_norm_eps: 1e-5,
        rope_theta: 1e4,
        rope_is_neox_style: false,
        block_size: 9,
        mask_token_id: 47,
        target_layer_ids: vec![0, 2],
        num_target_layers: 6,
        markov_rank: 8,
        enable_confidence_head: true,
        logits_start: 0,
    }
}

/// The target: a tiny LFM2 on the pool arms with the drafter's taps.
fn target_cfg() -> Config {
    let dcfg = drafter_cfg();
    Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: 128,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 256,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(7),
        use_flash_attn: false,
        attn_impl: AttnImpl::FlashDecoding,
        flash_page_size: PAGE,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::Paged,
        kv_quant: Default::default(),
        conv_state: ConvState::RotatingRing { k: 10 },
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: dcfg.tap_layers(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

fn randn(rng: &mut rand::rngs::StdRng, shape: Vec<usize>, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let normal = rand_distr::Normal::new(0.0f32, 0.2).unwrap();
    let vs: Vec<f32> = (0..n).map(|_| normal.sample(rng)).collect();
    Tensor::from_vec(vs, shape.as_slice(), dev)
}

fn tiny_drafter(dtype: DType, dev: &Device, seed: u64) -> Result<DSparkDrafter> {
    let cfg = drafter_cfg();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let q = cfg.num_attention_heads * hd;
    let kv = cfg.num_key_value_heads * hd;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut fill = |name: String, shape: Vec<usize>| -> Result<()> {
        t.insert(name, randn(&mut rng, shape, dev)?.to_dtype(dtype)?);
        Ok(())
    };
    fill("fc.weight".into(), vec![h, cfg.target_layer_ids.len() * h])?;
    fill("hidden_norm.weight".into(), vec![h])?;
    fill("norm.weight".into(), vec![h])?;
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        fill(format!("{p}.self_attn.q_proj.weight"), vec![q, h])?;
        fill(format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.o_proj.weight"), vec![h, q])?;
        fill(format!("{p}.self_attn.q_norm.weight"), vec![hd])?;
        fill(format!("{p}.self_attn.k_norm.weight"), vec![hd])?;
        fill(
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.up_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.down_proj.weight"),
            vec![h, cfg.intermediate_size],
        )?;
        fill(format!("{p}.input_layernorm.weight"), vec![h])?;
        fill(format!("{p}.post_attention_layernorm.weight"), vec![h])?;
    }
    fill(
        "markov_head.markov_w1.weight".into(),
        vec![cfg.vocab_size, cfg.markov_rank],
    )?;
    fill(
        "markov_head.markov_w2.weight".into(),
        vec![cfg.vocab_size, cfg.markov_rank],
    )?;
    fill(
        "confidence_head.proj.weight".into(),
        vec![1, cfg.hidden_size + cfg.markov_rank],
    )?;
    fill("confidence_head.proj.bias".into(), vec![1])?;
    let vb = VarBuilder::from_tensors(t, dtype, dev);
    DSparkDrafter::new(cfg, 4096, vb)
}

/// Fixed-ramp target weights (the batched-rows fixture's).
fn tiny_target(cfg: &Config, dtype: DType, dev: &Device) -> Result<Model> {
    let mut tensors = HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(v, shape, dev)?.to_dtype(dtype)?,
        );
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

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t.to_device(&Device::Cpu)?.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => t.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect(),
        DType::F16 => t
            .to_vec1::<half::f16>()?
            .into_iter()
            .map(|x| x.to_bits() as u32)
            .collect(),
        other => panic!("bits: {other:?}"),
    })
}

/// Three rows' contexts of different lengths (5, 40 and 70 positions — one
/// inside a page, one over two, one over three) keyed into a shared paged
/// pool AND into contiguous caches; `block_hidden_rows` over the three
/// blocks must equal each row's own `block_hidden`, bit for bit, on both
/// arms and at f32 and f16.
#[test]
fn block_hidden_rows_is_each_rows_own_block_hidden_bit_for_bit() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for dtype in [DType::F32, DType::F16] {
        let drafter = tiny_drafter(dtype, &dev, 148)?;
        let cfg = drafter.config().clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(148_001);
        let lens = [5usize, 40, 70];
        for shared in [true, false] {
            let ctx_kv = if shared {
                CtxKv::Paged {
                    page_size: PAGE,
                    shared: true,
                }
            } else {
                CtxKv::Contiguous
            };
            let mut caches = drafter.make_ctx_caches_for(ctx_kv, lens.len(), &dev)?;
            for (r, &n) in lens.iter().enumerate() {
                let taps: Vec<Tensor> = (0..cfg.target_layer_ids.len())
                    .map(|_| randn(&mut rng, vec![1, n, cfg.hidden_size], &dev))
                    .collect::<Result<_>>()?;
                drafter.update_context(&taps, 0, &mut caches[r])?;
                assert_eq!(caches[r][0].len(), n);
            }
            let blocks = randn(
                &mut rng,
                vec![lens.len(), cfg.block_size, cfg.hidden_size],
                &dev,
            )?
            .to_dtype(dtype)?;
            let refs: Vec<&[lloom::models::lfm2_dspark::CtxCache]> =
                caches.iter().map(|c| c.as_slice()).collect();
            let rows = drafter.block_hidden_rows(&blocks, &lens, &refs, None)?;
            assert_eq!(rows.dims(), [lens.len(), cfg.block_size, cfg.hidden_size]);
            let mut all = Vec::new();
            for (r, &n) in lens.iter().enumerate() {
                let one = drafter.block_hidden(&blocks.narrow(0, r, 1)?, n, &caches[r])?;
                let (a, b) = (bits(&rows.narrow(0, r, 1)?)?, bits(&one)?);
                let distinct = b.iter().collect::<std::collections::HashSet<_>>().len();
                assert!(distinct > 8, "row {r} vacuous ({distinct} distinct)");
                assert!(
                    a == b,
                    "{dtype:?} shared={shared}: row {r} (ctx {n}) differs between the batched and \
                     the one-row draft"
                );
                all.push(b);
            }
            assert!(
                all[0] != all[1] && all[1] != all[2],
                "rows identical: the test cannot tell them apart"
            );
            println!("{dtype:?} shared={shared}: block_hidden_rows bit-identical to block_hidden on {} rows", lens.len());
        }
    }
    Ok(())
}

/// The seam the loop drives: a `DSparkSpeculator` over a tiny target,
/// three sequences seeded from their own prompts' taps, `draft` over all
/// three (the batched pass) against `draft` over each alone (the one-row
/// path) — the same drafted ids per row; then over a subset in another
/// order, and `warmup_rows`. On the shared paged pool and on the contiguous
/// arm.
#[test]
fn a_batched_draft_proposes_what_each_rows_own_draft_proposes() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let dtype = DType::F16;
    let tcfg = target_cfg();
    let model = tiny_target(&tcfg, dtype, &dev)?;
    let prompts: Vec<Vec<u32>> = vec![
        vec![3, 11, 20, 5, 9],
        (0..36u32).map(|i| (i * 7 + 3) % 46).collect(),
        vec![40, 2, 17, 17, 6, 21, 8, 30, 31, 32, 33, 1],
    ];
    for shared in [true, false] {
        let ctx_kv = if shared {
            CtxKv::Paged {
                page_size: PAGE,
                shared: true,
            }
        } else {
            CtxKv::Contiguous
        };
        let mut spec = DSparkSpeculator::new_with_ctx(
            tiny_drafter(dtype, &dev, 149)?,
            model.draft_embedding().clone(),
            model.lm_head_linear().clone(),
            prompts.len(),
            ctx_kv,
        )?;
        spec.warmup_rows(prompts.len(), &dev)?;
        assert_eq!(spec.batched_drafts(), 1, "the warmup took the batched arm");
        // Seed every sequence from its own prompt's taps, the prefill's
        // argmax the anchor.
        for (i, p) in prompts.iter().enumerate() {
            let mut cache = Cache::new(true, dtype, &tcfg, &dev)?;
            let input = Tensor::new(p.as_slice(), &dev)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0, &mut cache)?;
            let taps = cache.take_taps();
            let anchor = logits
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?;
            spec.begin(SeqId(i), p)?;
            spec.observe(SeqId(i), &taps, &[anchor])?;
        }
        let ids = |d: &lloom::nn::speculator::Draft| -> Result<Vec<u32>> {
            match &d.tokens {
                DraftTokens::Device(t) => t.to_vec1::<u32>(),
                DraftTokens::Host(v) => Ok(v.clone()),
            }
        };
        // Reference: each row alone.
        let mut want = Vec::new();
        for i in 0..prompts.len() {
            let d = spec.draft(&[SeqId(i)])?;
            want.push(ids(&d[0])?);
        }
        assert_eq!(
            spec.batched_drafts(),
            1,
            "one-row drafts are the one-row path"
        );
        // Subject: all rows in one pass, then a subset in another order.
        let all: Vec<SeqId> = (0..prompts.len()).map(SeqId).collect();
        let got = spec.draft(&all)?;
        assert_eq!(spec.batched_drafts(), 2);
        for (i, d) in got.iter().enumerate() {
            let g = ids(d)?;
            assert_eq!(g.len(), want[i].len());
            assert_eq!(
                g, want[i],
                "shared={shared}: row {i}'s batched draft differs from its own"
            );
        }
        assert!(
            want[0] != want[1] || want[1] != want[2],
            "every row drafted the same block"
        );
        let sub = spec.draft(&[SeqId(2), SeqId(0)])?;
        assert_eq!(ids(&sub[0])?, want[2], "shared={shared}: subset row 2");
        assert_eq!(ids(&sub[1])?, want[0], "shared={shared}: subset row 0");
        // With the confidence hint asked for, the batched head agrees too.
        spec.want_confidence(true);
        let mut want_c = Vec::new();
        for i in 0..prompts.len() {
            want_c.push(spec.draft(&[SeqId(i)])?.remove(0).confidence);
        }
        let got_c = spec.draft(&all)?;
        for (i, d) in got_c.iter().enumerate() {
            assert_eq!(d.confidence.len(), want_c[i].len());
            assert!(!d.confidence.is_empty(), "no confidence hint");
            for (a, b) in d.confidence.iter().zip(&want_c[i]) {
                assert!(
                    (a - b).abs() <= 2e-3,
                    "shared={shared}: row {i} confidence {a} vs {b}"
                );
            }
        }
        println!(
            "shared={shared}: batched drafts equal the one-row drafts on {} rows: {want:?}",
            prompts.len()
        );
    }
    Ok(())
}
