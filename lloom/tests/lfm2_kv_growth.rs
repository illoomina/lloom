//! The KV slab grows in buckets and the stream does not notice (lloom #696).
//!
//! `Cache::prepare_pass` widens the `InPlace` slab by `kv_bucket` positions
//! whenever a pass would overrun it, copying the live slab into the wider one.
//! Growth has to be invisible to the computation: a run whose slab grew thirty
//! times and a run whose slab never grew read the same bytes at every step and
//! emit the same logits. The reference is the SAME model on the SAME arm with
//! a bucket wide enough that no grow fires — the fixed-slab behaviour every
//! recorded digest was taken under — so the only thing that differs between
//! the arms is the number of reallocations; and `Cat`, the arm with no slab at
//! all, is a second reference from a different mechanism (`CONTRIBUTING.md`
//! §3.1b).
//!
//! Non-vacuity: the grown arm asserts `kv_report().grows` — a bucket that
//! matched the reference because it never grew is #504's shape.
//!
//! The ceiling refusal is pinned with the property that matters: nothing
//! moved. The conv phase and `kv_len` read the same before and after a refused
//! pass, on both arms — which is what makes the refusal retryable rather than a
//! poisoned cache (a bail from inside a layer leaves the conv ring one step
//! ahead of the KV, because the ring advances once per pass before the layers
//! run).
//!
//! CPU: the question is which bytes the slab holds, not a kernel, and the CPU
//! backend is bit-stable (`CONTRIBUTING.md` §3.1). The memory refusal needs a
//! device with a `recommendedMaxWorkingSetSize` and is `lfm2_kv_growth_metal`'s.

use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::{Row, VarBuilder};
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

/// A tiny LFM2: six layers, conv/attention interleaved, `hidden_size = 8`,
/// `head_dim = 4`, a 128-position ceiling.
fn tiny_config(kv_append: KvAppend, conv_state: ConvState) -> Config {
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
        eos_token_id: None,
        kv_append,
        conv_state,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// Fixed-ramp weights, so every arm agrees on the model and any divergence is
/// the cache's.
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

const PROMPT: [u32; 5] = [3, 11, 20, 5, 9];

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Greedy decode: prefill `prompt`, then `steps` tokens each fed the argmax of
/// the last. Returns every pass's logits, the prefill's first.
fn decode(
    model: &Model,
    cache: &mut Cache,
    prompt: &[u32],
    steps: usize,
    dev: &Device,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(steps + 1);
    let input = Tensor::new(prompt, dev)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0, cache)?.i(0)?.to_vec1::<f32>()?;
    let mut pos = prompt.len();
    let mut next = argmax(&logits);
    out.push(logits);
    for _ in 0..steps {
        let input = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos, cache)?.i(0)?.to_vec1::<f32>()?;
        pos += 1;
        next = argmax(&logits);
        out.push(logits);
    }
    Ok(out)
}

fn assert_same_stream(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: step counts differ");
    for (step, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x == y,
            "{what}: logits differ at step {step} (first differing index {:?})",
            x.iter().zip(y).position(|(p, q)| p != q)
        );
    }
    assert!(
        !a.is_empty() && a[0].iter().any(|v| *v != 0.0),
        "{what}: an empty stream"
    );
}

/// Thirty-odd grows against none against no slab at all: one stream.
#[test]
fn a_slab_grown_thirty_times_emits_the_stream_of_one_that_never_grew() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(KvAppend::InPlace, ConvState::Shuffle);
    let model = build(&cfg, DType::F32, &dev)?;
    let steps = 60;

    let mut wide = Cache::new(true, DType::F32, &cfg, &dev)?;
    let mut narrow = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(2)?;
    let wide_stream = decode(&model, &mut wide, &PROMPT, steps, &dev)?;
    let narrow_stream = decode(&model, &mut narrow, &PROMPT, steps, &dev)?;
    assert_same_stream(
        &wide_stream,
        &narrow_stream,
        "bucket 2 against a slab that never grew",
    );

    let (w, n) = (wide.kv_report(), narrow.kv_report());
    assert_eq!(
        w.grows, 0,
        "the wide slab must never grow, or the reference is not one"
    );
    assert!(n.grows >= 30, "the narrow slab grew only {} times", n.grows);
    assert_eq!(w.kv_len, PROMPT.len() + steps);
    assert_eq!(n.kv_len, w.kv_len);
    assert_eq!(n.slab, 66, "ceil(65 / 2) * 2");
    assert_eq!(w.slab, 128, "one bucket, clamped to the ceiling");
    assert!(
        n.bytes < w.bytes,
        "a grown slab holds what it uses: {} < {}",
        n.bytes,
        w.bytes
    );

    // The arm with no slab at all is the second reference.
    let cat_cfg = tiny_config(KvAppend::Cat, ConvState::Shuffle);
    let cat_model = build(&cat_cfg, DType::F32, &dev)?;
    let mut cat = Cache::new(true, DType::F32, &cat_cfg, &dev)?;
    let cat_stream = decode(&cat_model, &mut cat, &PROMPT, steps, &dev)?;
    assert_same_stream(&cat_stream, &narrow_stream, "bucket 2 against `Cat`");
    assert_eq!(cat.kv_report().grows, 0);
    assert_eq!(cat.kv_report().kv_len, PROMPT.len() + steps);
    Ok(())
}

/// The ceiling refuses at the top of the pass, on every KV arm and both conv
/// arms, and moves nothing: not the conv phase (the rotating ring's advances
/// once per pass, before the layers), not a conv layer's state (the shuffle's
/// is rewritten inside the layer, before an attention layer could refuse),
/// not the KV length, not the slab.
#[test]
fn the_ceiling_refuses_before_any_state_moves() -> Result<()> {
    let dev = Device::Cpu;
    let conv_layers = [0usize, 2, 4];
    for arm in [KvAppend::InPlace, KvAppend::Cat] {
        for conv in [ConvState::Shuffle, ConvState::RotatingRing { k: 2 }] {
            let cfg = tiny_config(arm, conv);
            let model = build(&cfg, DType::F32, &dev)?;
            let mut cache = Cache::new_with(true, DType::F32, &cfg, &dev, 8)?.with_kv_bucket(3)?;
            // 5 prefilled + 3 decoded = 8 = the ceiling, admitted.
            let stream = decode(&model, &mut cache, &PROMPT, 3, &dev)?;
            let report = cache.kv_report();
            assert_eq!(report.kv_len, 8);
            assert_eq!(report.capacity, 8);
            let phase = cache.conv_phase();
            let states: Vec<Vec<f32>> = conv_layers
                .iter()
                .map(|&l| {
                    cache
                        .conv_state_for_test(l)
                        .expect("a conv layer holds state after a pass")
                        .flatten_all()?
                        .to_vec1::<f32>()
                })
                .collect::<Result<_>>()?;
            let next = argmax(stream.last().unwrap());
            let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
            for attempt in 0..2 {
                let err = model
                    .forward(&input, 8, &mut cache)
                    .err()
                    .unwrap_or_else(|| {
                        panic!("{arm:?} x {conv:?}: the ninth position was admitted past a ceiling of 8")
                    })
                    .to_string();
                assert!(
                    err.contains("KV ceiling"),
                    "{arm:?} x {conv:?} attempt {attempt}: {err}"
                );
                assert!(
                    err.contains("8 + 1 = 9"),
                    "the message names the arithmetic: {err}"
                );
                assert_eq!(
                    cache.conv_phase(),
                    phase,
                    "{arm:?} x {conv:?}: the conv phase moved"
                );
                assert_eq!(cache.kv_len(), 8, "{arm:?} x {conv:?}: the KV length moved");
                assert_eq!(
                    cache.kv_report(),
                    report,
                    "{arm:?} x {conv:?}: the slab moved"
                );
                for (&l, before) in conv_layers.iter().zip(&states) {
                    let after = cache
                        .conv_state_for_test(l)
                        .unwrap()
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(
                        &after == before,
                        "{arm:?} x {conv:?}: conv layer {l}'s state moved on a refusal"
                    );
                }
            }
        }
    }
    // A ceiling past the RoPE tables is refused at construction, not at the
    // first narrow past them.
    let cfg = tiny_config(KvAppend::InPlace, ConvState::Shuffle);
    let err = Cache::new_with(true, DType::F32, &cfg, &dev, 129)
        .err()
        .expect("a ceiling past max_position_embeddings")
        .to_string();
    assert!(err.contains("kv_capacity 129 is outside 1..=128"), "{err}");
    let err = Cache::new_with(true, DType::F32, &cfg, &dev, 0)
        .err()
        .expect("a zero ceiling")
        .to_string();
    assert!(err.contains("outside 1..=128"), "{err}");
    Ok(())
}

/// `forward_rows` grows every row's cache before any row's state moves, and a
/// batched step over grown caches is the step over caches that never grew.
#[test]
fn forward_rows_grows_each_row_on_its_own() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(KvAppend::InPlace, ConvState::RotatingRing { k: 2 });
    let model = build(&cfg, DType::F32, &dev)?;
    let prompts: Vec<Vec<u32>> = vec![
        vec![3, 11, 20, 5, 9],
        vec![40, 2],
        vec![1, 1, 8, 19, 27, 30, 6, 14],
    ];
    let steps = 12;

    let run = |bucket: Option<usize>| -> Result<(Vec<Vec<Vec<f32>>>, Vec<Cache>)> {
        let mut caches = Vec::new();
        let mut nexts = Vec::new();
        for prompt in &prompts {
            let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
            if let Some(b) = bucket {
                cache = cache.with_kv_bucket(b)?;
            }
            let input = Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?;
            let logits = model
                .forward(&input, 0, &mut cache)?
                .i(0)?
                .to_vec1::<f32>()?;
            nexts.push(argmax(&logits));
            caches.push(cache);
        }
        let mut streams: Vec<Vec<Vec<f32>>> = vec![Vec::new(); prompts.len()];
        let mut pos: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        for _ in 0..steps {
            let input = Tensor::new(nexts.as_slice(), &dev)?.reshape((prompts.len(), 1))?;
            let mut rows: Vec<Row<'_, Cache>> = caches
                .iter_mut()
                .zip(pos.iter())
                .map(|(cache, &pos)| Row { pos, cache })
                .collect();
            let logits = model.forward_rows(&input, &mut rows)?;
            for r in 0..prompts.len() {
                let l = logits.i(r)?.to_vec1::<f32>()?;
                nexts[r] = argmax(&l);
                streams[r].push(l);
                pos[r] += 1;
            }
        }
        Ok((streams, caches))
    };
    let (wide, wide_caches) = run(None)?;
    let (narrow, narrow_caches) = run(Some(2))?;
    for r in 0..prompts.len() {
        assert_same_stream(&wide[r], &narrow[r], &format!("row {r}"));
        assert_eq!(wide_caches[r].kv_report().grows, 0);
        assert!(
            narrow_caches[r].kv_report().grows >= 5,
            "row {r} grew {} times",
            narrow_caches[r].kv_report().grows
        );
        assert_eq!(narrow_caches[r].kv_len(), prompts[r].len() + steps);
    }
    Ok(())
}

/// A speculative window that crosses a bucket boundary — `forward_all` over
/// `k` positions after `advance(k)` — reads what an unrefused slab reads, and
/// the rollback after it leaves the grown slab where the wide one is.
#[test]
fn a_speculative_window_that_crosses_a_bucket_matches_one_that_does_not() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(KvAppend::InPlace, ConvState::RotatingRing { k: 4 });
    let model = build(&cfg, DType::F32, &dev)?;
    let proposed: [u32; 3] = [12, 33, 21];

    let run = |bucket: Option<usize>| -> Result<(Vec<Vec<f32>>, Cache)> {
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        if let Some(b) = bucket {
            cache = cache.with_kv_bucket(b)?;
        }
        let mut out = Vec::new();
        let input = Tensor::new(&PROMPT, &dev)?.unsqueeze(0)?;
        out.push(
            model
                .forward(&input, 0, &mut cache)?
                .i(0)?
                .to_vec1::<f32>()?,
        );
        // The window: 5 held + 3 proposed = 8, past a slab of 6 on bucket 2.
        let tok = cache.advance(proposed.len())?;
        let input = Tensor::new(&proposed, &dev)?.unsqueeze(0)?;
        let all = model.forward_all(&input, PROMPT.len(), &mut cache)?;
        assert_eq!(all.dims(), &[1, proposed.len(), cfg.vocab_size]);
        out.push(all.flatten_all()?.to_vec1::<f32>()?);
        cache.resolve(tok, 2)?;
        assert_eq!(cache.kv_len(), PROMPT.len() + 2, "two accepted");
        // One more ordinary step after the rollback.
        let input = Tensor::new(&[proposed[2]], &dev)?.unsqueeze(0)?;
        out.push(
            model
                .forward(&input, PROMPT.len() + 2, &mut cache)?
                .i(0)?
                .to_vec1::<f32>()?,
        );
        Ok((out, cache))
    };
    let (wide, wide_cache) = run(None)?;
    let (narrow, narrow_cache) = run(Some(2))?;
    assert_same_stream(&wide, &narrow, "a speculative window across a bucket");
    assert_eq!(wide_cache.kv_report().grows, 0);
    assert!(
        narrow_cache.kv_report().grows >= 2,
        "grew {} times",
        narrow_cache.kv_report().grows
    );
    assert_eq!(narrow_cache.kv_len(), wide_cache.kv_len());
    Ok(())
}

/// A batched step refuses before ANY row's state moves. The row at its
/// ceiling is the last one, so a loop that advanced rows as it prepared them
/// would have moved row 0's conv phase before row 1 refused — which is the
/// mutation this pins (`forward_rows` prepares every row, then advances).
#[test]
fn forward_rows_refuses_before_any_rows_state_moves() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(KvAppend::InPlace, ConvState::RotatingRing { k: 2 });
    let model = build(&cfg, DType::F32, &dev)?;
    // Row 0 has room; row 1's ceiling is exactly what its prompt plus three
    // steps reach.
    let mut roomy = Cache::new(true, DType::F32, &cfg, &dev)?.with_kv_bucket(2)?;
    let mut tight = Cache::new_with(true, DType::F32, &cfg, &dev, 8)?.with_kv_bucket(2)?;
    let prompts: [&[u32]; 2] = [&[3, 11, 20, 5, 9, 4, 4], &PROMPT];
    let mut nexts = Vec::new();
    for (cache, prompt) in [(&mut roomy, prompts[0]), (&mut tight, prompts[1])] {
        let input = Tensor::new(prompt, &dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0, cache)?.i(0)?.to_vec1::<f32>()?;
        nexts.push(argmax(&logits));
    }
    let mut pos = [prompts[0].len(), prompts[1].len()];
    let mut step =
        |nexts: &[u32], pos: &[usize; 2], roomy: &mut Cache, tight: &mut Cache| -> Result<Tensor> {
            let input = Tensor::new(nexts, &dev)?.reshape((2, 1))?;
            let mut rows = [
                Row {
                    pos: pos[0],
                    cache: roomy,
                },
                Row {
                    pos: pos[1],
                    cache: tight,
                },
            ];
            model.forward_rows(&input, &mut rows)
        };
    for _ in 0..3 {
        let logits = step(&nexts, &pos, &mut roomy, &mut tight)?;
        for r in 0..2 {
            nexts[r] = argmax(&logits.i(r)?.to_vec1::<f32>()?);
            pos[r] += 1;
        }
    }
    assert_eq!(tight.kv_len(), 8, "row 1 sits on its ceiling");
    let (phase0, phase1) = (roomy.conv_phase(), tight.conv_phase());
    let (report0, report1) = (roomy.kv_report(), tight.kv_report());
    let err = step(&nexts, &pos, &mut roomy, &mut tight)
        .err()
        .expect("row 1's ninth position was admitted past a ceiling of 8")
        .to_string();
    assert!(err.contains("KV ceiling"), "{err}");
    assert_eq!(
        roomy.conv_phase(),
        phase0,
        "row 0's conv phase moved on row 1's refusal"
    );
    assert_eq!(
        tight.conv_phase(),
        phase1,
        "row 1's conv phase moved on its own refusal"
    );
    assert_eq!(
        roomy.kv_report(),
        report0,
        "row 0's slab moved on row 1's refusal"
    );
    assert_eq!(tight.kv_report(), report1);
    Ok(())
}
