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
use lloom::nn::{RaggedRow, Row, VarBuilder};
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

/// The RAGGED pass (`Model::forward_ragged`, #728's fused chunk): two decode
/// rows and one prefill piece in one forward, each row's logits its own
/// `forward`'s. The piece row's reference is the strongest one available —
/// ONE `forward` over its whole prompt, so the chunk boundary (a piece onto
/// a warm cache, the conv writer carrying its left context) is checked
/// against one pass over the concatenation, not against another chunked
/// pass. `conv_prefill_ctx` is on for the reason §13.2f gives.
fn ragged_arms(cfg: Config, dtype: DType, dev: &Device) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let model = build(&cfg, dtype, dev)?;
    let decode_rows = rows();
    let piece_prompt: Vec<u32> = vec![7, 22, 13, 4, 35, 18, 9, 26, 2, 41];
    let split = 6;

    // References: each decode row alone; the piece row as one whole prefill.
    let mut solo = Vec::new();
    for (prompt, next) in &decode_rows[..2] {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0, &mut cache)?;
        let input = Tensor::new(&[*next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt.len(), &mut cache)?;
        solo.push(logits.i(0)?.to_vec1::<f32>()?);
    }
    {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(piece_prompt.as_slice(), dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0, &mut cache)?;
        solo.push(logits.i(0)?.to_vec1::<f32>()?);
    }

    // Subject: the decode rows' prompts prefilled, the piece row's first
    // `split` tokens prefilled, then ONE ragged pass over
    // `[next_0, next_1, piece_prompt[split..]]`.
    let mut caches = Vec::new();
    for (prompt, _) in &decode_rows[..2] {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0, &mut cache)?;
        caches.push(cache);
    }
    {
        let mut cache = Cache::new(true, dtype, &cfg, dev)?;
        let input = Tensor::new(&piece_prompt[..split], dev)?.unsqueeze(0)?;
        let _ = model.forward(&input, 0, &mut cache)?;
        caches.push(cache);
    }
    let mut tokens: Vec<u32> = decode_rows[..2].iter().map(|(_, n)| *n).collect();
    tokens.extend_from_slice(&piece_prompt[split..]);
    let input = Tensor::new(tokens.as_slice(), dev)?.unsqueeze(0)?;
    let lens = [1usize, 1, piece_prompt.len() - split];
    let poss = [decode_rows[0].0.len(), decode_rows[1].0.len(), split];
    let mut row_refs: Vec<RaggedRow<'_, Cache>> = caches
        .iter_mut()
        .zip(lens.iter().zip(poss.iter()))
        .map(|(cache, (&len, &pos))| RaggedRow { pos, len, cache })
        .collect();
    let logits = model.forward_ragged(&input, &mut row_refs)?;
    assert_eq!(logits.dims(), &[3, cfg.vocab_size]);
    let ragged: Vec<Vec<f32>> = (0..3)
        .map(|r| logits.i(r)?.to_vec1::<f32>())
        .collect::<Result<_>>()?;
    Ok((solo, ragged))
}

#[test]
fn each_row_of_a_ragged_pass_is_its_own_forward_on_the_cpu() -> Result<()> {
    for conv_state in [ConvState::Shuffle, ConvState::RotatingRing { k: 2 }] {
        let mut cfg = tiny_config_on(conv_state);
        cfg.conv_prefill_ctx = true;
        let (solo, ragged) = ragged_arms(cfg, DType::F32, &Device::Cpu)?;
        assert_rows_differ(&ragged);
        for (r, (s, b)) in solo.iter().zip(&ragged).enumerate() {
            let d = max_abs_diff(s, b);
            eprintln!("{conv_state:?} ragged row {r}: max |solo - ragged| = {d:e}");
            assert!(d < TOL, "{conv_state:?} row {r}: max |solo - ragged| = {d}");
            assert_eq!(argmax(s), argmax(b), "row {r}: the argmax moved");
        }
        // The control that makes the piece row's tolerance evidence: without
        // the left-context carry the chunked prefill computes something one
        // pass does not (§13.2f defect 1), and the piece row's logits move.
        let mut off = tiny_config_on(conv_state);
        off.conv_prefill_ctx = false;
        let (solo_off, ragged_off) = ragged_arms(off, DType::F32, &Device::Cpu)?;
        let d = max_abs_diff(&solo_off[2], &ragged_off[2]);
        eprintln!("{conv_state:?} ragged piece row, carry off: max |solo - ragged| = {d:e}");
        // Clean reads ~7e-9 on the piece row and the control ~8e-5 (four
        // decades apart on this tiny model); the bound sits a decade above
        // the tolerance so a control that stopped biting is a failure.
        assert!(
            d > 10.0 * TOL,
            "{conv_state:?}: the carry-off control did not move the piece row ({d})"
        );
    }
    Ok(())
}

/// Metal, f16, under the forced GEMM: every row of the ragged pass is
/// bit-identical to its own forward, the piece row's to one pass over its
/// whole prompt.
#[cfg(feature = "metal")]
#[test]
fn each_row_of_a_ragged_pass_is_bit_identical_to_its_forward_under_the_gemm() -> Result<()> {
    std::env::set_var("LLOOM_386_FORCE_GEMM", "m");
    let dev = Device::new_metal(0)?;
    let mut cfg = tiny_config();
    cfg.conv_prefill_ctx = true;
    let (solo, ragged) = ragged_arms(cfg, DType::F16, &dev)?;
    assert_rows_differ(&ragged);
    for (r, (s, b)) in solo.iter().zip(&ragged).enumerate() {
        let d = max_abs_diff(s, b);
        eprintln!("metal ragged row {r}: max |solo - ragged| = {d:e}");
        assert_eq!(
            d, 0.0,
            "row {r}: the ragged row differs from its solo forward by {d}"
        );
        assert_eq!(argmax(s), argmax(b));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The shared pool's arms (`batched-decode-design.md` §12.2): every row a slot
// of one `BatchPool`, the attention and the conv of the whole batch in one
// dispatch per kernel — against the per-row loop over the SAME pool, which
// is the same arithmetic (each row's own table, own live count, own ring at
// its own phase) and must therefore agree to the bit; and the pool's per-row
// path against the private paged cache it is built from.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
mod pool_arms {
    use super::*;
    use lloom::models::lfm2::{AttnImpl, BatchPool};

    /// The pool's arms: `Paged × FlashDecoding × RotatingRing`, the fused
    /// ring, a 4-token page so the prompts span pages.
    fn pool_config() -> Config {
        let mut cfg = tiny_config_on(ConvState::RotatingRing { k: 2 });
        // The flash kernels instantiate head dims 32/64/96/128: two heads of 32.
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.kv_append = KvAppend::Paged;
        cfg.attn_impl = AttnImpl::FlashDecoding;
        // The steel paged read needs a page of whole 32-key blocks; the
        // prompts below span two pages, so both flash paths (one chunk, two
        // chunks) and a page boundary are exercised.
        cfg.flash_page_size = 32;
        cfg.flash_pages_per_chunk = 1;
        cfg.conv_fused = true;
        cfg
    }

    /// Three rows: two short, one past a page boundary (36 tokens at page 32).
    fn pool_rows() -> Vec<(Vec<u32>, u32)> {
        let long: Vec<u32> = (0..36u32).map(|i| (i * 7 + 3) % 48).collect();
        vec![(vec![3, 11, 20, 5, 9], 12), (vec![40, 2], 33), (long, 21)]
    }

    /// A 40-token prompt for the piece row: 6 prefilled, 34 fed as the piece.
    fn piece_prompt() -> Vec<u32> {
        (0..40u32).map(|i| (i * 11 + 5) % 48).collect()
    }

    fn logits_rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
        (0..t.dim(0)?)
            .map(|r| t.i(r)?.to_vec1::<f32>())
            .collect()
    }

    fn assert_bit_equal(what: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
        assert_eq!(a.len(), b.len(), "{what}: row count");
        for (r, (x, y)) in a.iter().zip(b).enumerate() {
            let bits_a: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
            let bits_b: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                bits_a, bits_b,
                "{what}: row {r} differs (max |d| = {})",
                max_abs_diff(x, y)
            );
        }
    }

    /// Two decode steps over three rows on a pool, with the batched arms on
    /// or off: the logits of both steps.
    fn pool_steps(cfg: &Config, dtype: DType, dev: &Device, batched: bool) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, u64)> {
        let model = build(cfg, dtype, dev)?;
        let rows = pool_rows();
        let pool = BatchPool::new(cfg, dev, 4, 64)?;
        pool.set_batched_arms(batched);
        let mut caches = Vec::with_capacity(rows.len());
        for (prompt, _) in &rows {
            let mut cache = Cache::for_batch(true, dtype, cfg, dev, &pool)?;
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
        let step1 = model.forward_rows(&input, &mut row_refs)?;
        drop(row_refs);
        // A second step at the next position, feeding fixed tokens.
        let nexts2: Vec<u32> = nexts.iter().map(|n| (n + 5) % cfg.vocab_size as u32).collect();
        let input2 = Tensor::new(nexts2.as_slice(), dev)?.reshape((rows.len(), 1))?;
        let mut row_refs: Vec<Row<'_, Cache>> = caches
            .iter_mut()
            .zip(rows.iter())
            .map(|(cache, (prompt, _))| Row {
                pos: prompt.len() + 1,
                cache,
            })
            .collect();
        let step2 = model.forward_rows(&input2, &mut row_refs)?;
        Ok((logits_rows(&step1)?, logits_rows(&step2)?, pool.shared_calls()))
    }

    #[test]
    fn the_batched_arms_reproduce_the_per_row_loop_over_the_same_pool_bit_for_bit() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = pool_config();
        for dtype in [DType::F32, DType::F16] {
            let (b1, b2, calls_on) = pool_steps(&cfg, dtype, &dev, true)?;
            let (p1, p2, calls_off) = pool_steps(&cfg, dtype, &dev, false)?;
            assert!(calls_on > 0, "the batched arms ran ({calls_on} layer calls)");
            assert_eq!(calls_off, 0, "the per-row arm took no batched call");
            assert_bit_equal(&format!("step 1 {dtype:?}"), &b1, &p1);
            assert_bit_equal(&format!("step 2 {dtype:?}"), &b2, &p2);
            assert_rows_differ(&b1);
            assert_rows_differ(&b2);
        }
        Ok(())
    }

    /// The pool's per-row path against a private paged cache on the same
    /// config: the same kernels over a table of one sequence.
    #[test]
    fn a_pool_row_alone_is_the_private_paged_cache() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = pool_config();
        let dtype = DType::F32;
        let model = build(&cfg, dtype, &dev)?;
        let pool = BatchPool::new(&cfg, &dev, 2, 64)?;
        for (prompt, next) in pool_rows() {
            let mut private = Cache::new(true, dtype, &cfg, &dev)?;
            let mut pooled = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
            let input = Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?;
            let _ = model.forward(&input, 0, &mut private)?;
            let _ = model.forward(&input, 0, &mut pooled)?;
            let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
            let a = model.forward(&input, prompt.len(), &mut private)?;
            let b = model.forward(&input, prompt.len(), &mut pooled)?;
            assert_bit_equal("pool row vs private cache", &logits_rows(&a)?, &logits_rows(&b)?);
            assert_eq!(pooled.kv_len(), prompt.len() + 1);
        }
        Ok(())
    }

    /// The ragged pass on the pool — two decode rows beside a prompt piece —
    /// with the batched arms on against off.
    fn pool_ragged(cfg: &Config, dtype: DType, dev: &Device, batched: bool) -> Result<(Vec<Vec<f32>>, u64)> {
        let model = build(cfg, dtype, dev)?;
        let decode_rows = pool_rows();
        let piece_prompt = piece_prompt();
        let split = 6;
        let pool = BatchPool::new(cfg, dev, 4, 64)?;
        pool.set_batched_arms(batched);
        let mut caches = Vec::new();
        for (prompt, _) in &decode_rows[..2] {
            let mut cache = Cache::for_batch(true, dtype, cfg, dev, &pool)?;
            let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
            let _ = model.forward(&input, 0, &mut cache)?;
            caches.push(cache);
        }
        {
            let mut cache = Cache::for_batch(true, dtype, cfg, dev, &pool)?;
            let input = Tensor::new(&piece_prompt[..split], dev)?.unsqueeze(0)?;
            let _ = model.forward(&input, 0, &mut cache)?;
            caches.push(cache);
        }
        let mut tokens: Vec<u32> = decode_rows[..2].iter().map(|(_, n)| *n).collect();
        tokens.extend_from_slice(&piece_prompt[split..]);
        let input = Tensor::new(tokens.as_slice(), dev)?.unsqueeze(0)?;
        let lens = [1usize, 1, piece_prompt.len() - split];
        let poss = [decode_rows[0].0.len(), decode_rows[1].0.len(), split];
        let mut row_refs: Vec<RaggedRow<'_, Cache>> = caches
            .iter_mut()
            .zip(lens.iter().zip(poss.iter()))
            .map(|(cache, (&len, &pos))| RaggedRow { pos, len, cache })
            .collect();
        let logits = model.forward_ragged(&input, &mut row_refs)?;
        drop(row_refs);
        // And a plain step after it, every row now a decode row.
        let nexts: Vec<u32> = vec![12, 33, 21];
        let input = Tensor::new(nexts.as_slice(), dev)?.reshape((3, 1))?;
        let poss2 = [poss[0] + 1, poss[1] + 1, piece_prompt.len()];
        let mut row_refs: Vec<Row<'_, Cache>> = caches
            .iter_mut()
            .zip(poss2.iter())
            .map(|(cache, &pos)| Row { pos, cache })
            .collect();
        let step = model.forward_rows(&input, &mut row_refs)?;
        let mut out = logits_rows(&logits)?;
        out.extend(logits_rows(&step)?);
        Ok((out, pool.shared_calls()))
    }

    #[test]
    fn the_ragged_batched_arms_reproduce_the_per_row_loop_bit_for_bit() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = pool_config();
        for dtype in [DType::F32, DType::F16] {
            let (on, calls_on) = pool_ragged(&cfg, dtype, &dev, true)?;
            let (off, calls_off) = pool_ragged(&cfg, dtype, &dev, false)?;
            assert!(calls_on > 0);
            assert_eq!(calls_off, 0);
            assert_bit_equal(&format!("ragged + step {dtype:?}"), &on, &off);
            assert_rows_differ(&on[..3]);
        }
        Ok(())
    }

    /// A slot returns to the pool when its cache drops, and a cleared row
    /// starts over on a zeroed ring.
    #[test]
    fn slots_are_leased_and_returned() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = pool_config();
        let pool = BatchPool::new(&cfg, &dev, 2, 64)?;
        let a = Cache::for_batch(true, DType::F32, &cfg, &dev, &pool)?;
        let b = Cache::for_batch(true, DType::F32, &cfg, &dev, &pool)?;
        assert!(Cache::for_batch(true, DType::F32, &cfg, &dev, &pool).is_err());
        assert_eq!(pool.report().slots_free, 0);
        drop(a);
        assert_eq!(pool.report().slots_free, 1);
        let c = Cache::for_batch(true, DType::F32, &cfg, &dev, &pool)?;
        assert_ne!(c.batch_slot(), b.batch_slot());
        Ok(())
    }
}

#[cfg(feature = "metal")]
mod prefix_share {
    use super::*;
    use lloom::models::lfm2::{AttnImpl, BatchPool};

    fn share_config() -> Config {
        let mut cfg = tiny_config_on(ConvState::RotatingRing { k: 2 });
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.kv_append = KvAppend::Paged;
        cfg.attn_impl = AttnImpl::FlashDecoding;
        cfg.flash_page_size = 32;
        cfg.flash_pages_per_chunk = 1;
        cfg.conv_fused = true;
        // A prefill in pieces reads the ring the previous piece left — the
        // writer a shared prefix's remainder needs (§10.10, §12.3).
        cfg.conv_prefill_ctx = true;
        cfg
    }

    /// Prefill `prompt` in pieces of `piece` tokens through `forward`, so
    /// every piece boundary is a pass boundary (and a page boundary when
    /// `piece` is the page): the state a sharer starts from is snapshotted
    /// there. Returns the last piece's logits.
    fn prefill_pieces(model: &Model, cache: &mut Cache, prompt: &[u32], piece: usize, dev: &Device) -> Result<Tensor> {
        let mut pos = 0;
        let mut last = None;
        while pos < prompt.len() {
            let n = piece.min(prompt.len() - pos);
            let input = Tensor::new(&prompt[pos..pos + n], dev)?.unsqueeze(0)?;
            last = Some(model.forward(&input, pos, cache)?);
            pos += n;
        }
        Ok(last.expect("a non-empty prompt"))
    }

    /// A row that SHARES two pages of another's prompt and prefills the six
    /// tokens after them computes what a fresh row prefilling all 70 does —
    /// the remainder's logits and a decode step after it, bit for bit — and
    /// the pool holds the shared pages once.
    #[test]
    fn a_shared_prefix_reproduces_the_fresh_prefill_bit_for_bit() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = share_config();
        for dtype in [DType::F32, DType::F16] {
            let model = build(&cfg, dtype, &dev)?;
            let pool = BatchPool::new(&cfg, &dev, 4, 128)?;
            let prompt: Vec<u32> = (0..70u32).map(|i| (i * 13 + 7) % 48).collect();
            let question: Vec<u32> = prompt[64..].to_vec();
            let next = 21u32;

            // The source: the whole prompt, in page-sized pieces (its passes
            // begin at 32 and 64: two snapshots).
            let mut src = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
            let _ = prefill_pieces(&model, &mut src, &prompt, 32, &dev)?;
            assert_eq!(src.kv_len(), 70);
            assert!(pool.snapshots() >= 2, "snapshots at the two page boundaries");

            // The reference: a fresh row, the same pieces.
            let mut fresh = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
            let want_tail = prefill_pieces(&model, &mut fresh, &prompt, 32, &dev)?;
            let pages_before = pool.report().pages_in_use;

            // The sharer: two pages from the source, then the remainder.
            let mut shared = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
            let n = shared.share_prefix_from(&src, 64)?;
            assert_eq!(n, 64, "two whole pages shared");
            assert_eq!(shared.kv_len(), 64);
            let input = Tensor::new(question.as_slice(), &dev)?.unsqueeze(0)?;
            let got_tail = model.forward(&input, 64, &mut shared)?;
            assert_eq!(shared.kv_len(), 70);
            // The shared pages are held once: the sharer added only its tail page.
            assert_eq!(pool.report().pages_in_use, pages_before + 1);

            let bits = |t: &Tensor| -> Result<Vec<u32>> {
                Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect())
            };
            assert_eq!(bits(&got_tail)?, bits(&want_tail)?, "the remainder's logits ({dtype:?})");

            // A decode step on both.
            let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
            let a = model.forward(&input, 70, &mut fresh)?;
            let b = model.forward(&input, 70, &mut shared)?;
            assert_eq!(bits(&a)?, bits(&b)?, "the step after ({dtype:?})");
            // Non-vacuity: the logits depend on the prefix — a row that
            // prefilled only the remainder differs.
            let mut bare = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
            let q = Tensor::new(question.as_slice(), &dev)?.unsqueeze(0)?;
            let c = model.forward(&q, 0, &mut bare)?;
            assert_ne!(bits(&c)?, bits(&want_tail)?, "the prefix matters");
        }
        Ok(())
    }

    /// The share refuses what it cannot do: a non-empty sharer, a source in
    /// another pool, a prefix shorter than a page (0 shared).
    #[test]
    fn share_refusals_and_the_page_floor() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = share_config();
        let dtype = DType::F32;
        let model = build(&cfg, dtype, &dev)?;
        let pool = BatchPool::new(&cfg, &dev, 3, 128)?;
        let other = BatchPool::new(&cfg, &dev, 1, 128)?;
        let prompt: Vec<u32> = (0..40u32).map(|i| (i * 5 + 1) % 48).collect();
        let mut src = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
        let _ = prefill_pieces(&model, &mut src, &prompt, 32, &dev)?;
        let mut a = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
        assert_eq!(a.share_prefix_from(&src, 20)?, 0, "below a page: nothing");
        assert_eq!(a.share_prefix_from(&src, 40)?, 32, "one page of the 40");
        assert!(a.share_prefix_from(&src, 32).is_err(), "a non-empty sharer is refused");
        let mut foreign = Cache::for_batch(true, dtype, &cfg, &dev, &other)?;
        assert!(foreign.share_prefix_from(&src, 32).is_err(), "another pool is refused");
        Ok(())
    }
}

#[cfg(feature = "metal")]
mod ragged_verify {
    use super::*;
    use lloom::models::lfm2::{AttnImpl, BatchPool, VerifyRow};

    /// The pool's arms with a speculative ring (K = 2 drafts + the anchor)
    /// and two tapped layers, as a DSpark target has.
    fn verify_config() -> Config {
        let mut cfg = tiny_config_on(ConvState::RotatingRing { k: 3 });
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.kv_append = KvAppend::Paged;
        cfg.attn_impl = AttnImpl::FlashDecoding;
        cfg.flash_page_size = 32;
        cfg.flash_pages_per_chunk = 1;
        cfg.conv_fused = true;
        cfg.conv_prefill_ctx = true;
        cfg.tap_layers = vec![1, 3];
        cfg.tap_dtype = DType::F32;
        cfg
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    /// Three rows' windows verified as one ragged pass (the batched arms on)
    /// against each row's own `verify_tokens_enqueue`/`complete` on a pool
    /// with the arms off: the accepted tokens, the probes (per row, trimmed)
    /// and a decode step after the resolve — bit for bit.
    #[test]
    fn a_ragged_verify_is_each_rows_own_verify_bit_for_bit() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = verify_config();
        let eos = [7u32];
        for dtype in [DType::F32, DType::F16] {
            let model = build(&cfg, dtype, &dev)?;
            let rows = pool_rows_for_verify();
            // Two pools, one per arm, the same three rows prefilled the same way.
            let mut arms = Vec::new();
            for batched in [false, true] {
                let pool = BatchPool::new(&cfg, &dev, 4, 128)?;
                pool.set_batched_arms(batched);
                let mut caches = Vec::new();
                for (prompt, _) in &rows {
                    let mut c = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
                    let input = Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?;
                    let _ = model.forward(&input, 0, &mut c)?;
                    let _ = c.take_taps();
                    caches.push(c);
                }
                arms.push((pool, caches));
            }
            // The drafts: two tokens per row; row 0's first draft is the
            // target's own prediction (found by a throwaway one-row verify),
            // so it accepts past position 0; the others are arbitrary.
            let lasts: Vec<u32> = rows.iter().map(|(_, n)| *n).collect();
            let predicted = {
                let pool = BatchPool::new(&cfg, &dev, 1, 128)?;
                let mut c = Cache::for_batch(true, dtype, &cfg, &dev, &pool)?;
                let input = Tensor::new(rows[0].0.as_slice(), &dev)?.unsqueeze(0)?;
                let _ = model.forward(&input, 0, &mut c)?;
                let _ = c.take_taps();
                let proposed = Tensor::new(&[0u32, 0], &dev)?;
                let pending = model.verify_tokens_enqueue(lasts[0], &proposed, rows[0].0.len(), &mut c)?;
                let (accepted, _) = model.verify_tokens_complete(pending, &mut c, &eos)?;
                accepted[0]
            };
            let drafts: Vec<Vec<u32>> = vec![vec![predicted, 33], vec![3, 3], vec![21, 40]];

            // Reference: per row, the one-row verify (arms off).
            let (_, ref_caches) = &mut arms[0];
            let mut want = Vec::new();
            for (r, c) in ref_caches.iter_mut().enumerate() {
                let proposed = Tensor::new(drafts[r].as_slice(), &dev)?;
                let pending = model.verify_tokens_enqueue(lasts[r], &proposed, rows[r].0.len(), c)?;
                let (accepted, probe) = model.verify_tokens_complete(pending, c, &eos)?;
                want.push((accepted, probe));
            }
            // Subject: the ragged verify (arms on).
            let (pool_on, sub_caches) = &mut arms[1];
            let proposed: Vec<Tensor> = drafts
                .iter()
                .map(|d| Tensor::new(d.as_slice(), &dev))
                .collect::<Result<_>>()?;
            let mut vrows: Vec<VerifyRow<'_>> = sub_caches
                .iter_mut()
                .enumerate()
                .map(|(r, c)| VerifyRow {
                    cache: c,
                    last: lasts[r],
                    pos: rows[r].0.len(),
                    proposed: &proposed[r],
                })
                .collect();
            let pending = model.verify_rows_enqueue(&mut vrows)?;
            drop(vrows);
            let mut refs: Vec<&mut Cache> = sub_caches.iter_mut().collect();
            let got = model.verify_rows_complete(pending, &mut refs, &eos)?;
            drop(refs);
            assert!(pool_on.shared_calls() > 0, "the batched arms ran the verify");

            for r in 0..rows.len() {
                assert_eq!(got[r].0, want[r].0, "row {r} accepted ({dtype:?})");
                assert!(!got[r].0.is_empty());
                assert_eq!(got[r].1.len(), want[r].1.len(), "row {r} probe count");
                for (a, b) in got[r].1.iter().zip(want[r].1.iter()) {
                    assert_eq!(a.dims(), b.dims(), "row {r} probe shape");
                    assert_eq!(bits(a)?, bits(b)?, "row {r} probe bits ({dtype:?})");
                }
            }
            // At least one row accepted past position 0 and one did not.
            let ns: Vec<usize> = got.iter().map(|(a, _)| a.len()).collect();
            assert!(ns.iter().any(|&n| n > 1) && ns.iter().any(|&n| n == 1), "accepted {ns:?}");

            // A decode step after the resolve, per row, both arms per-row.
            for r in 0..rows.len() {
                let next = *got[r].0.last().unwrap();
                let pos = rows[r].0.len() + got[r].0.len();
                let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
                let a = model.forward(&input, pos, &mut arms[0].1[r])?;
                let b = model.forward(&input, pos, &mut arms[1].1[r])?;
                assert_eq!(bits(&a)?, bits(&b)?, "row {r} step after ({dtype:?})");
            }
        }
        Ok(())
    }

    fn pool_rows_for_verify() -> Vec<(Vec<u32>, u32)> {
        let long: Vec<u32> = (0..36u32).map(|i| (i * 7 + 3) % 48).collect();
        vec![(vec![3, 11, 20, 5, 9], 12), (vec![40, 2], 33), (long, 21)]
    }
}
