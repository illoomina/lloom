//! `forward_embeds` is `forward` with the embedding lifted out — pinned.
//!
//! #597 added two entry points to `lfm2::Model` so a caller holding projected
//! image features can build the residual stream itself: `embed` (token ids to
//! embeddings) and `forward_embeds` (embeddings to logits). Everything after
//! the embedding is `forward`'s code, **duplicated rather than shared**, for
//! `forward_all`'s own recorded reason: `forward` is on the per-token path and
//! its shape is what every §2.3.8c digest belongs to, so a branch inside it
//! would put a caller decode does not have onto the decode path.
//!
//! **Duplication is the cost of that choice, and drift is its failure mode.**
//! This file is what makes the drift visible: it asserts that
//! `forward_embeds(embed(ids))` is **bit-identical** to `forward(ids)`, so a
//! change to one loop and not the other turns this red rather than silently
//! giving the image path a different model from the text path.
//!
//! # Why bit-identical rather than close
//!
//! The two run the same dispatches over the same weights in the same order, so
//! there is no reordering to permit. A tolerance here would be a licence for
//! exactly the divergence the file exists to catch (`CONTRIBUTING.md` §3.1b
//! rule 1: prefer an absolute reference to a shape).
//!
//! # Why the check is over BOTH regimes
//!
//! `seq_len == 1` and `seq_len > 1` take different branches inside the layers
//! (`DESIGN.md` §10.2h enumerates four), and an image prefill is a `seq_len > 1`
//! pass while every decode step after it is `seq_len == 1`. Checking one regime
//! would leave the other free to drift.
//!
//! CPU only, f32: the question is whether two code paths compute the same
//! thing, not what a kernel computes, and the CPU backend is bit-stable
//! (`CONTRIBUTING.md` §3.1).

use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};
use lloom::nn::VarBuilder;
use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};

fn tiny_config() -> Config {
    Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 64,
        conv_l_cache: 3,
        conv_bias: false,
        // One of each, so the check covers both layer kinds.
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: None,
        use_flash_attn: false,
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state: ConvState::Shuffle,
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        tap_layers: Vec::new(),
        tap_dtype: DType::F32,
    }
}

fn build(cfg: &Config, dev: &Device) -> Result<Model> {
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
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Model::new(cfg, vb)
}

fn logits_of(model: &Model, cfg: &Config, ids: &[u32], via_embeds: bool) -> Result<Vec<f32>> {
    let dev = Device::Cpu;
    // A fresh cache per arm: the conv phase and the KV length are state, and
    // reusing one would make the second arm a continuation rather than a repeat.
    let mut cache = Cache::new(true, DType::F32, cfg, &dev)?;
    let input = Tensor::new(ids, &dev)?.unsqueeze(0)?;
    let logits = if via_embeds {
        let embeds = model.embed(&input)?;
        model.forward_embeds(&embeds, 0, &mut cache)?
    } else {
        model.forward(&input, 0, &mut cache)?
    };
    logits.i(0)?.to_vec1::<f32>()
}

/// The whole point of the file: the two entry points are one model.
#[test]
fn forward_embeds_over_plain_embeddings_is_bit_identical_to_forward() -> Result<()> {
    let cfg = tiny_config();
    let model = build(&cfg, &Device::Cpu)?;

    // Both regimes. `seq_len > 1` is what an image prefill takes; `seq_len == 1`
    // is every decode step after it.
    for ids in [vec![3u32], vec![3u32, 9, 1, 27, 5]] {
        let direct = logits_of(&model, &cfg, &ids, false)?;
        let via = logits_of(&model, &cfg, &ids, true)?;
        assert_eq!(
            direct.len(),
            via.len(),
            "seq_len {}: logit widths differ",
            ids.len()
        );
        assert!(
            direct
                .iter()
                .zip(via.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "seq_len {}: forward and forward_embeds disagree — the two layer \
             loops have drifted apart. Largest difference {:e}.",
            ids.len(),
            direct
                .iter()
                .zip(via.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        );
    }
    Ok(())
}

/// The non-vacuity arm. Without it the test above passes on a `forward_embeds`
/// that ignores its argument and calls `forward` on a constant, and on any
/// model whose logits happen not to depend on the input.
///
/// `CONTRIBUTING.md` §3.1a rule 4 and §15.1a's VACUOUS class: an equality that
/// cannot fail is not a comparison. This shows the quantity being compared
/// actually moves with what is fed in.
#[test]
fn the_comparison_is_not_vacuous_because_different_embeddings_give_different_logits() -> Result<()>
{
    let cfg = tiny_config();
    let model = build(&cfg, &Device::Cpu)?;
    let a = logits_of(&model, &cfg, &[3u32, 9, 1], true)?;
    let b = logits_of(&model, &cfg, &[7u32, 2, 30], true)?;
    let moved = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        moved > 1e-6,
        "two different prompts produced the same logits (max difference {moved:e}) — \
         the equality test above would pass on anything"
    );
    Ok(())
}

/// `forward_embeds` must reject a rank-2 input rather than reinterpreting it.
///
/// The two entry points differ in the rank they take (`[b, seq]` ids against
/// `[b, seq, hidden]` embeddings), and a caller that passes the wrong one
/// should be told at the boundary rather than somewhere inside layer 0.
#[test]
fn forward_embeds_refuses_token_ids() -> Result<()> {
    let cfg = tiny_config();
    let model = build(&cfg, &Device::Cpu)?;
    let dev = Device::Cpu;
    let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
    let ids = Tensor::new(&[3u32, 9], &dev)?.unsqueeze(0)?;
    assert!(
        model.forward_embeds(&ids, 0, &mut cache).is_err(),
        "a rank-2 tensor was accepted where `[b, seq, hidden]` is required"
    );
    Ok(())
}
