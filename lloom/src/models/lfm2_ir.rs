//! The IR of one LFM2 decode step, emitted from a [`Config`]
//! (`design/docs/optimizer-design.md` §6.3, §8.1: the graph is emitted as a
//! byproduct of what the model knows about itself, never recovered from an
//! execution).
//!
//! What is emitted is the `seq_len == 1`, `B == 1` step at the arms the config
//! selects — the unfused elementwise chain the optimizer derives over, and
//! every other dispatch as an opaque node so the census reconciles. The
//! dispatch sequence is reconciled against a committed trace of the real
//! model (`design/measurements/issue-727-raw/pass-plan/runs/
//! trace-arena-shuffle.txt`, 538 dispatches per token at `Shuffle × Sdpa ×
//! InPlace × Contiguous`, no fusion) in `tests/optimizer_derivation.rs`; the
//! arms that trace did not take are described from the source and say so
//! below.
//!
//! What this is NOT: a second implementation of the forward pass. It names
//! kernel families and shapes; it computes nothing.

use lloom_tensor::DType;

use super::lfm2::{AttnImpl, Config, ConvState, KvAppend, LayerType, QkvLayout};
use crate::optimizer::{Dim, Graph, NodeId, Shape};

/// The IR of one decode step of `cfg` at `dtype`.
pub fn decode_step(cfg: &Config, dtype: DType) -> Graph {
    let mut g = Graph::new();
    let h = cfg.hidden_size;

    // The token: `index_select` over the embedding table (`is_u32_<dtype>`).
    let ids = g.input("input_ids", Shape::of(&[1, 1]), DType::U32);
    let embed = g.weight(
        "model.embed_tokens.weight",
        Shape::of(&[cfg.vocab_size, h]),
        dtype,
    );
    let mut hidden = g.opaque("is_u32", &[embed, ids], Shape::of(&[1, 1, h]), dtype);

    for (l, layer_type) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let residual = hidden;
        let norm_w = g.weight(&format!("{p}.operator_norm.weight"), Shape::of(&[h]), dtype);
        let xn = g.rms_norm(norm_w, residual);
        let block = match layer_type {
            LayerType::FullAttention => attention(&mut g, cfg, dtype, &p, xn, l),
            LayerType::Conv => conv(&mut g, cfg, dtype, &p, xn, l),
        };
        // `forward(..)? + residual`: the block's projection on the left.
        hidden = g.add(block, residual);

        let residual = hidden;
        let ffn_w = g.weight(&format!("{p}.ffn_norm.weight"), Shape::of(&[h]), dtype);
        let xn = g.rms_norm(ffn_w, residual);
        let down = mlp(&mut g, cfg, dtype, &p, xn);
        hidden = g.add(down, residual);
    }

    let norm_w = g.weight("model.embedding_norm.weight", Shape::of(&[h]), dtype);
    let hn = g.rms_norm(norm_w, hidden);
    // `i((.., seq_len - 1, ..)).contiguous()`: at `seq_len == 1` a view.
    let last = g.view("index", hn, Shape::of(&[1, h]));
    let lm_head = if cfg.tie_embedding {
        embed
    } else {
        g.weight("lm_head.weight", Shape::of(&[cfg.vocab_size, h]), dtype)
    };
    let logits = g.gemv(lm_head, last);
    let f32_logits = g.opaque(
        "cast",
        &[logits],
        Shape::of(&[1, cfg.vocab_size]),
        DType::F32,
    );
    g.mark_root(f32_logits);
    g
}

/// The SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`. Returns the down projection,
/// unadded.
fn mlp(g: &mut Graph, cfg: &Config, dtype: DType, p: &str, xn: NodeId) -> NodeId {
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let w1 = g.weight(
        &format!("{p}.feed_forward.w1.weight"),
        Shape::of(&[i, h]),
        dtype,
    );
    let w3 = g.weight(
        &format!("{p}.feed_forward.w3.weight"),
        Shape::of(&[i, h]),
        dtype,
    );
    let w2 = g.weight(
        &format!("{p}.feed_forward.w2.weight"),
        Shape::of(&[h, i]),
        dtype,
    );
    let gate = g.gemv(w1, xn);
    let gate = g.silu(gate);
    let up = g.gemv(w3, xn);
    let gated = g.mul(gate, up);
    g.gemv(w2, gated)
}

/// The attention block. Returns the output projection, unadded.
///
/// Reconciled against the trace at `Sdpa × InPlace × Contiguous`: three
/// projections, two QK norms, two RoPE dispatches, two `copy2d` appends, one
/// `sdpa_vector`, the output projection. `Strided` has the same dispatches
/// (the copies it removes are clones at `seq_len == 1`, `DESIGN.md` §6.2c);
/// `Fused` folds each norm+RoPE pair into one `rmsnorm_rope`. `Cat` and
/// `Paged` appends and the `Generic`/`FlashDecoding` kernels are named from
/// the source, not from a trace: their counts are not reconciled.
fn attention(g: &mut Graph, cfg: &Config, dtype: DType, p: &str, xn: NodeId, l: usize) -> NodeId {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let heads = cfg.num_attention_heads;
    let kvh = cfg.num_key_value_heads;
    let kv = kvh * hd;

    let wq = g.weight(
        &format!("{p}.self_attn.q_proj.weight"),
        Shape::of(&[h, h]),
        dtype,
    );
    let wk = g.weight(
        &format!("{p}.self_attn.k_proj.weight"),
        Shape::of(&[kv, h]),
        dtype,
    );
    let wv = g.weight(
        &format!("{p}.self_attn.v_proj.weight"),
        Shape::of(&[kv, h]),
        dtype,
    );
    let q = g.gemv(wq, xn);
    let k = g.gemv(wk, xn);
    let v = g.gemv(wv, xn);
    let q = g.view("reshape", q, Shape::of(&[1, heads, 1, hd]));
    let k = g.view("reshape", k, Shape::of(&[1, kvh, 1, hd]));
    let v = g.view("reshape", v, Shape::of(&[1, kvh, 1, hd]));

    let qn_w = g.weight(
        &format!("{p}.self_attn.q_layernorm.weight"),
        Shape::of(&[hd]),
        dtype,
    );
    let kn_w = g.weight(
        &format!("{p}.self_attn.k_layernorm.weight"),
        Shape::of(&[hd]),
        dtype,
    );
    let cos = g.input(&format!("rope.cos@{l}"), Shape::of(&[1, hd / 2]), dtype);
    let sin = g.input(&format!("rope.sin@{l}"), Shape::of(&[1, hd / 2]), dtype);

    let (qr, kr) = match cfg.qkv_layout {
        QkvLayout::Contiguous | QkvLayout::Strided => {
            let qn = g.rms_norm(qn_w, q);
            let kn = g.rms_norm(kn_w, k);
            let qr = g.opaque(
                "rope",
                &[qn, cos, sin],
                Shape::of(&[1, heads, 1, hd]),
                dtype,
            );
            let kr = g.opaque("rope", &[kn, cos, sin], Shape::of(&[1, kvh, 1, hd]), dtype);
            (qr, kr)
        }
        QkvLayout::Fused => {
            let qr = g.opaque(
                "rmsnorm_rope",
                &[qn_w, q, cos, sin],
                Shape::of(&[1, heads, 1, hd]),
                dtype,
            );
            let kr = g.opaque(
                "rmsnorm_rope",
                &[kn_w, k, cos, sin],
                Shape::of(&[1, kvh, 1, hd]),
                dtype,
            );
            (qr, kr)
        }
    };

    let kv_shape = Shape(vec![
        Dim::Const(1),
        Dim::Const(kvh),
        Dim::Sym("kv_len"),
        Dim::Const(hd),
    ]);
    let k_cache = g.input(&format!("kv.k@{l}"), kv_shape.clone(), dtype);
    let v_cache = g.input(&format!("kv.v@{l}"), kv_shape.clone(), dtype);
    let append = match cfg.kv_append {
        KvAppend::InPlace => "copy2d",
        KvAppend::Cat => "kv_cat",
        KvAppend::Paged => "flash_kv_scatter",
    };
    let k_all = g.opaque(append, &[kr, k_cache], kv_shape.clone(), dtype);
    let v_all = g.opaque(append, &[v, v_cache], kv_shape, dtype);

    let attn_family = match cfg.attn_impl {
        AttnImpl::Sdpa => "sdpa_vector",
        AttnImpl::FlashDecoding => "flash_decoding",
        AttnImpl::Generic => "attention_generic",
    };
    let y = g.opaque(
        attn_family,
        &[qr, k_all, v_all],
        Shape::of(&[1, heads, 1, hd]),
        dtype,
    );
    let y = g.view("reshape", y, Shape::of(&[1, 1, h]));
    let wo = g.weight(
        &format!("{p}.self_attn.out_proj.weight"),
        Shape::of(&[h, h]),
        dtype,
    );
    g.gemv(wo, y)
}

/// The short-conv block. Returns the output projection, unadded.
///
/// Reconciled against the trace on the unfused `Shuffle` arm (`bmul`, two
/// `copy2d`, `bmul`, `fast_sum`, `bmul`) and, through the rotating trace
/// (516 per token), the unfused `RotatingRing` arm (one `copy2d`). The fused
/// arms are one kernel each (`shortconv_step`, `shortconv_ring`), named from
/// the source; `SlidingRing` is described as the rotating arm without the
/// permuted weight and without its occasional compaction.
fn conv(g: &mut Graph, cfg: &Config, dtype: DType, p: &str, xn: NodeId, l: usize) -> NodeId {
    let h = cfg.hidden_size;
    let lc = cfg.conv_l_cache;
    let width = cfg.conv_state.width(lc);

    let w_in = g.weight(
        &format!("{p}.conv.in_proj.weight"),
        Shape::of(&[3 * h, h]),
        dtype,
    );
    let bcx = g.gemv(w_in, xn);
    let bcx = g.view("transpose", bcx, Shape::of(&[1, 3 * h, 1]));
    let b = g.view("narrow", bcx, Shape::of(&[1, h, 1]));
    let c = g.view("narrow", bcx, Shape::of(&[1, h, 1]));
    let x = g.view("narrow", bcx, Shape::of(&[1, h, 1]));
    let w_conv = g.weight(
        &format!("{p}.conv.conv.weight"),
        Shape::of(&[h, 1, lc]),
        dtype,
    );
    let w_conv = g.view("squeeze", w_conv, Shape::of(&[h, lc]));
    let state = g.input(&format!("conv.state@{l}"), Shape::of(&[1, h, width]), dtype);

    let out = match (cfg.conv_state, cfg.conv_fused) {
        // The whole block as one kernel: `bx`, the shift, the fold, `*c`, the
        // next state.
        (ConvState::Shuffle, true) => g.opaque(
            "shortconv_step",
            &[b, x, c, state, w_conv],
            Shape::of(&[1, h, 1]),
            dtype,
        ),
        // The ring writer as one kernel; the output arrives transposed with
        // `*c` applied, so the block ends here.
        (ConvState::RotatingRing { .. }, true) => {
            let out = g.opaque(
                "shortconv_ring",
                &[b, x, c, state, w_conv],
                Shape::of(&[1, 1, h]),
                dtype,
            );
            let w_out = g.weight(
                &format!("{p}.conv.out_proj.weight"),
                Shape::of(&[h, h]),
                dtype,
            );
            return g.gemv(w_out, out);
        }
        (arm, fused) => {
            let bx = g.mul(b, x);
            let window = match arm {
                ConvState::Shuffle => {
                    // `cat([state[1..], bx])`: two copies into the new state.
                    let tail = g.view("narrow", state, Shape::of(&[1, h, lc - 1]));
                    let shifted = g.opaque("copy2d", &[tail], Shape::of(&[1, h, lc]), dtype);
                    g.opaque("copy2d", &[bx, shifted], Shape::of(&[1, h, lc]), dtype)
                }
                ConvState::RotatingRing { .. } | ConvState::SlidingRing { .. } => {
                    // `slice_set` at the write slot: one copy into the ring.
                    let ring = g.opaque("copy2d", &[bx, state], Shape::of(&[1, h, width]), dtype);
                    if matches!(arm, ConvState::SlidingRing { .. }) {
                        g.view("narrow", ring, Shape::of(&[1, h, lc]))
                    } else {
                        ring
                    }
                }
            };
            let taps = g.node(window).shape.last().unwrap_or(Dim::Const(lc));
            let taps = match taps {
                Dim::Const(t) => t,
                Dim::Sym(_) => lc,
            };
            let conv = if fused {
                // `conv1d_depthwise_k`; the scaled `_c` variant folds `*c`
                // only on the contiguous `Shuffle` window, handled above.
                let w = g.view("unsqueeze", w_conv, Shape::of(&[h, 1, taps]));
                g.opaque(
                    "conv1d_depthwise",
                    &[window, w],
                    Shape::of(&[1, h, 1]),
                    dtype,
                )
            } else {
                let w = g.view("unsqueeze", w_conv, Shape::of(&[1, h, taps]));
                let prod = g.mul(window, w);
                g.opaque("fast_sum", &[prod], Shape::of(&[1, h, 1]), dtype)
            };
            g.mul(c, conv)
        }
    };
    let out = g.view("transpose", out, Shape::of(&[1, 1, h]));
    let w_out = g.weight(
        &format!("{p}.conv.out_proj.weight"),
        Shape::of(&[h, h]),
        dtype,
    );
    g.gemv(w_out, out)
}
