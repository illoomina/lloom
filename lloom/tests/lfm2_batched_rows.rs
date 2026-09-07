//! `Model::forward_rows` computes, per row, what `Model::forward` computes for
//! that row alone (issue #694, `design/docs/batched-decode-design.md` §10).
//!
//! # What this pins, and the reference it is checked against
//!
//! A batched step applies every weight matrix to all rows in one `m = B` GEMM
//! and runs the per-sequence half — RoPE at each row's own position, the KV
//! append and read, the conv window — per row against that row's own cache.
//! The property that makes it a correct batched decode is that a row's logits
//! do not depend on which rows sit beside it (§2.3.3 #7), and the reference
//! for that is `forward` on the same cache at the same position: a genuinely
//! different code path (the single-sequence layer loop), not the subject
//! restated (`CONTRIBUTING.md` §3.1b).
//!
//! Three rows with different prompts, different lengths and different cache
//! contents run one batched step; each row's `[vocab]` logits are compared
//! against `forward` of the same token on a cache holding the same prefix.
//!
//! # Exact on the GEMM kernel set, close across the GEMV/GEMM seam
//!
//! At `m = 1` the weight matmuls take the GEMV route and at `m = B > 1` the
//! tiled GEMM; the two accumulate in different orders (§13.5b, §2.3.3c tier
//! 2), so `forward` and `forward_rows` are not bit-identical across that seam
//! and no test can make them so. The Metal test pins bit-identity where it is
//! owed: with `LLOOM_386_FORCE_GEMM=m` the `m = 1` reference takes the same
//! GEMM the batch takes, and then every row's logits must match to the bit.
//! Both tests also pin that the argmax agrees and the rows differ from one
//! another (non-vacuity: a `forward_rows` that used row 0's cache for every
//! row would fail the second).

use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::{Row, VarBuilder};
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

/// A tiny LFM2 on the crown cache arms (`InPlace × Shuffle`), no taps: six
/// layers, conv/attention interleaved, `hidden_size = 8`, `head_dim = 4`.
fn tiny_config() -> Config {
    tiny_config_on(ConvState::Shuffle)
}

/// The same model on a given conv arm. The rotating ring is the arm whose
/// conv state carries a per-row write phase; the shuffle's does not, so only
/// the ring can tell a step that forgot to advance a row's phase from one
/// that did (the mutation harness records the shuffle-only survival).
fn tiny_config_on(conv_state: ConvState) -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
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
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state,
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// Fixed-ramp weights, so the two arms agree on the model and any divergence
/// is the paths'.
fn build(cfg: &Config, dtype: DType, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
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

/// Three prompts of different lengths and content, and the token each row
/// decodes next.
fn rows() -> Vec<(Vec<u32>, u32)> {
    vec![
        (vec![3, 11, 20, 5, 9], 12),
        (vec![40, 2], 33),
        (vec![1, 1, 8, 19, 27, 30, 6, 14], 21),
    ]
}

/// The batched step's `[vocab]` logits per row, and `forward`'s for the same
/// token on a cache holding the same prefix, both from freshly prefilled
/// caches.
fn both_arms(dtype: DType, dev: &Device) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    both_arms_on(tiny_config(), dtype, dev)
}

fn both_arms_on(cfg: Config, dtype: DType, dev: &Device) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let model = build(&cfg, dtype, dev)?;
    let rows = rows();

    // Reference: each row alone through `forward`.
    let mut solo = Vec::with_capacity(rows.len());
    for (prompt, next) in &rows {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0, &mut cache)?;
        let input = Tensor::new(&[*next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt.len(), &mut cache)?;
        solo.push(logits.i(0)?.to_vec1::<f32>()?);
    }

    // Subject: the same prefills, then one batched step over all rows.
    let mut caches = Vec::with_capacity(rows.len());
    for (prompt, _) in &rows {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0, &mut cache)?;
        caches.push(cache);
    }
    let nexts: Vec<u32> = rows.iter().map(|(_, n)| *n).collect();
    let input = Tensor::new(nexts.as_slice(), dev)?.reshape((rows.len(), 1))?;
    let mut row_refs: Vec<Row<'_, Cache>> = caches
        .iter_mut()
        .zip(rows.iter())
        .map(|(cache, (prompt, _))| Row {
            pos: prompt.len(),
            cache,
        })
        .collect();
    let logits = model.forward_rows(&input, &mut row_refs)?;
    assert_eq!(logits.dims(), &[rows.len(), cfg.vocab_size]);
    let batched: Vec<Vec<f32>> = (0..rows.len())
        .map(|r| logits.i(r)?.to_vec1::<f32>())
        .collect::<Result<_>>()?;
    Ok((solo, batched))
}

/// The CPU tolerance, set from the printed differences (§3.1b: read the
/// values beside `ok`). Clean, the `m = 1` against `m = 3` CPU matmul reads
/// 1.5e-8 to 3.0e-8 on every row; the smallest mutation the harness applies
/// (a batched step that does not advance the ring's write phase) reads
/// 1.2e-5 to 4.2e-5. A first tolerance of 1e-4 let that mutation survive —
/// recorded in `measurements/issue-694-raw/mutate_output.txt` — so the bound
/// sits two decades below it and two above the clean reading.
const TOL: f32 = 1e-6;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The rows must differ from one another, or the per-row comparison could
/// pass with every row reading row 0's cache.
fn assert_rows_differ(batched: &[Vec<f32>]) {
    for i in 0..batched.len() {
        for j in i + 1..batched.len() {
            assert!(
                max_abs_diff(&batched[i], &batched[j]) > 1e-3,
                "rows {i} and {j} produced the same logits — the rows are not independent \
                 sequences"
            );
        }
    }
}

/// CPU, f32: every row's batched logits are its solo logits to within the
/// reordering of an `m = 1` against an `m = B` matmul, and the argmax agrees.
#[test]
fn each_row_of_a_batched_step_is_its_own_forward_on_the_cpu() -> Result<()> {
    let (solo, batched) = both_arms(DType::F32, &Device::Cpu)?;
    assert_rows_differ(&batched);
    for (r, (s, b)) in solo.iter().zip(&batched).enumerate() {
        let d = max_abs_diff(s, b);
        eprintln!("shuffle row {r}: max |solo - batched| = {d:e}");
        assert!(d < TOL, "row {r}: max |solo - batched| = {d}");
        assert_eq!(argmax(s), argmax(b), "row {r}: the argmax moved");
    }
    Ok(())
}

/// The rotating ring, CPU: the per-row conv phase is the row's own — the
/// reference here writes each row's ring at its own position after its own
/// prefill, so a batched step that advanced no phase, or one phase for all
/// rows, reads a stale slot and moves the logits.
#[test]
fn each_row_of_a_batched_step_is_its_own_forward_on_the_rotating_ring() -> Result<()> {
    let (solo, batched) = both_arms_on(
        tiny_config_on(ConvState::RotatingRing { k: 2 }),
        DType::F32,
        &Device::Cpu,
    )?;
    assert_rows_differ(&batched);
    for (r, (s, b)) in solo.iter().zip(&batched).enumerate() {
        let d = max_abs_diff(s, b);
        eprintln!("ring row {r}: max |solo - batched| = {d:e}");
        assert!(d < TOL, "row {r}: max |solo - batched| = {d}");
        assert_eq!(argmax(s), argmax(b), "row {r}: the argmax moved");
    }
    Ok(())
}

/// Metal, f16, the crown arms: with the `m = 1` reference forced onto the GEMM
/// the batch takes (`LLOOM_386_FORCE_GEMM=m`, read once per process — set
/// before the first matmul of this process, which is this test's), every
/// row's logits are bit-identical to its solo forward.
#[cfg(feature = "metal")]
#[test]
fn each_row_of_a_batched_step_is_bit_identical_to_its_forward_under_the_gemm() -> Result<()> {
    // Read once by `mlx_gemm.rs` on first use; this test file's tests are the
    // only matmuls in this process and all of them want the same arm.
    std::env::set_var("LLOOM_386_FORCE_GEMM", "m");
    let dev = Device::new_metal(0)?;
    let (solo, batched) = both_arms(DType::F16, &dev)?;
    assert_rows_differ(&batched);
    for (r, (s, b)) in solo.iter().zip(&batched).enumerate() {
        let d = max_abs_diff(s, b);
        assert_eq!(
            d, 0.0,
            "row {r}: the batched row differs from its solo forward by {d}"
        );
        assert_eq!(argmax(s), argmax(b));
    }
    Ok(())
}

/// The refusals: a multi-token step, a row-count mismatch, an empty batch.
#[test]
fn forward_rows_refuses_what_it_cannot_batch() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let model = build(&cfg, DType::F32, &dev)?;
    let mut a = Cache::new(true, DType::F32, &cfg, &dev)?;
    let mut b = Cache::new(true, DType::F32, &cfg, &dev)?;

    let two_tokens = Tensor::new(&[[1u32, 2]], &dev)?;
    let mut rows = vec![Row {
        pos: 0,
        cache: &mut a,
    }];
    let err = model
        .forward_rows(&two_tokens, &mut rows)
        .err()
        .expect("seq_len 2 is refused");
    assert!(err.to_string().contains("one token per row"), "{err}");

    let one = Tensor::new(&[[1u32]], &dev)?;
    let mut rows = vec![
        Row {
            pos: 0,
            cache: &mut a,
        },
        Row {
            pos: 0,
            cache: &mut b,
        },
    ];
    let err = model
        .forward_rows(&one, &mut rows)
        .err()
        .expect("1 token for 2 rows is refused");
    assert!(err.to_string().contains("caches"), "{err}");

    let mut rows: Vec<Row<'_, Cache>> = Vec::new();
    let none = Tensor::new(&[0u32; 0], &dev)?.reshape((0, 1))?;
    assert!(model.forward_rows(&none, &mut rows).is_err());
    Ok(())
}
