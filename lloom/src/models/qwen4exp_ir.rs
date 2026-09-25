//! The IR of one Qwen4-exp decode step, emitted from a [`Config`]
//! (`design/docs/optimizer-design.md` §6.3; lloom #749 P31 (6)): the step's
//! dispatches as the optimizer's nodes, so the census, the seam report and a
//! derivation exist for this model as they do for LFM2 (`lfm2_ir`).
//!
//! What is emitted is the `B == 1`, one-token step of the streamed tier at
//! the arms the tree runs (the device plan armed, the routed gate|up as one
//! launch, P31 (3)), reconciled family by family against the traced step
//! `design/measurements/qwen4exp-dispatches-raw/p31-trace-driver-pair.txt`
//! (run 1, step 10: 1 628 dispatches) in `tests/qwen4exp_ir.rs`. The
//! quantized cores are named by their family (`mul_mv`, `mul_mv_id`,
//! `mul_mv_id2`) and not by the weight kind, which the header decides per
//! layer and the config does not know.
//!
//! Every kernel of this step is opaque to the derivation except the few
//! plain elementwise dispatches: the model's chains are hand-fused already
//! (`silu_mul`, `hc_*`, `moe_combine`, the mixed gemv epilogues), and the
//! quantized matvec cores carry no epilogue seam the catalogue could bind to
//! — the derivation's report on this graph is the list of those seams.
//!
//! **Named from the trace, not from the source:** the PLE block's composed
//! chain (the one `ple_layers` bracket: three norms, the taps as `bmul` /
//! `badd`, the hash's scalar ops) and the driver's tail. They are emitted at
//! the counts the trace shows so the census reconciles; deriving them from
//! `ple.rs` is the emitter's next pass.
//!
//! What this is NOT: a second implementation of the forward pass. It names
//! kernel families and shapes; it computes nothing.

use lloom_tensor::DType;

use super::qwen4exp::Config;
use crate::optimizer::{Graph, NodeId, Shape};

const F32: DType = DType::F32;

/// The IR of one decode step of `cfg`.
pub fn decode_step(cfg: &Config) -> Graph {
    let mut g = Graph::new();
    let h = cfg.hidden_size;
    let n = cfg.hc_count;
    let act = |d: usize| Shape::of(&[1, 1, d]);

    // The token: `get_rows` over the quantized embedding.
    let ids = g.input("input_ids", Shape::of(&[1, 1]), DType::U32);
    let embed = g.weight("token_embd.weight", Shape::of(&[cfg.vocab_size, h]), F32);
    let x = g.opaque("get_rows", &[embed, ids], act(h), F32);
    // The hyper-connection carry `[1, 1, n·h]`, expanded from the token.
    let mut carry = g.opaque("copy2d_f32", &[x], act(n * h), F32);
    carry = g.opaque("copy2d_f32", &[carry], act(n * h), F32);

    for l in 0..cfg.n_layers {
        let p = format!("blk.{l}");
        // The attention / DeltaNet block in its bracket.
        let (xn, mixw) = hc_mix(&mut g, cfg, &format!("{p}.attn"), carry, true);
        let block = if cfg.is_full(l) {
            qsa(&mut g, cfg, &p, xn)
        } else {
            deltanet(&mut g, cfg, &p, xn)
        };
        carry = g.opaque("hc_combine", &[carry, block, mixw], act(n * h), F32);

        // The MoE block in its bracket.
        let (xn, mixw) = hc_mix(&mut g, cfg, &format!("{p}.ffn"), carry, true);
        let block = moe(&mut g, cfg, &p, xn);
        carry = g.opaque("hc_combine", &[carry, block, mixw], act(n * h), F32);

        if cfg.ple_layers.contains(&l) {
            carry = ple(&mut g, cfg, &p, carry);
        }
    }

    // The head: the carry's streams summed (`fast_sum` over `n`), the
    // lm_head, the argmax on the device.
    carry = g.opaque("copy2d_f32", &[carry], act(n * h), F32);
    let hn = g.opaque("fast_sum", &[carry], act(h), F32);
    let lm_head = g.weight("output.weight", Shape::of(&[cfg.vocab_size, h]), F32);
    let logits = g.opaque("mul_mv", &[lm_head, hn], Shape::of(&[1, cfg.vocab_size]), F32);
    let next = g.opaque("fast_argmax", &[logits], Shape::of(&[1]), DType::U32);
    g.mark_root(next);
    g
}

/// One hyper-connection prologue (`HyperConnection::mix`, `hc.rs`): the
/// grouped norm, the inject projection (when the bracket has one), the down projection with its SiLU
/// and the up projection with its sigmoid (both mixed gemv epilogues), the
/// gated collapse of the `n` streams into the block's input. Returns the
/// collapsed input and the mixing weights the epilogue combines with.
fn hc_mix(g: &mut Graph, cfg: &Config, p: &str, carry: NodeId, inject: bool) -> (NodeId, NodeId) {
    let (h, n, lr) = (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank);
    let act = |d: usize| Shape::of(&[1, 1, d]);
    let xn = g.opaque("hc_norm", &[carry], act(n * h), F32);
    let w_down = g.weight(&format!("{p}.hc_down.weight"), Shape::of(&[lr, n * h]), F32);
    let lo = g.opaque("gemv_mixed_silu", &[w_down, xn], act(lr), F32);
    let w_up = g.weight(&format!("{p}.hc_up.weight"), Shape::of(&[n * n + n, lr]), F32);
    let mut up_in = vec![w_up, lo];
    if inject {
        // The PLE's bracket carries no inject weight (`hc.rs` `inject: None`).
        let w_inject = g.weight(&format!("{p}.hc_inject.weight"), Shape::of(&[n, n * h]), F32);
        up_in.push(g.opaque("gemv_mixed", &[w_inject, xn], act(n), F32));
    }
    let mixw = g.opaque("gemv_mixed_sigmoid", &up_in, act(n * n + n), F32);
    let x = g.opaque("hc_collapse", &[xn, mixw], act(h), F32);
    (x, mixw)
}

/// One quantized stem `[out, k] · x` (`kernel_mul_mv_<kind>`), the kind the
/// header's.
fn stem(g: &mut Graph, name: &str, out: usize, k: usize, x: NodeId) -> NodeId {
    let w = g.weight(name, Shape::of(&[out, k]), F32);
    g.opaque("mul_mv", &[w, x], Shape::of(&[1, 1, out]), F32)
}

/// The Gated DeltaNet block (`deltanet_block.rs`): the stacked `attn_qkv`
/// stem, the `attn_gate` stem, the f16 `dt`/`A` projection (a mixed gemv),
/// the prologue (conv + gate + write strength, the state pool's window), the
/// recurrence step, the gated norm, `ssm_out`.
fn deltanet(g: &mut Graph, cfg: &Config, p: &str, xn: NodeId) -> NodeId {
    let (h, inner, dt) = (cfg.hidden_size, cfg.ssm_inner_size, cfg.ssm_dt_rank);
    let (h_v, d) = (dt, cfg.ssm_state_size);
    let act = |d: usize| Shape::of(&[1, 1, d]);
    let qkv = stem(g, &format!("{p}.attn_qkv.weight"), 2 * 2048 + inner, h, xn);
    let gate = stem(g, &format!("{p}.attn_gate.weight"), inner, h, xn);
    let w_dt = g.weight(&format!("{p}.ssm_dt.weight"), Shape::of(&[2 * dt, h]), F32);
    let ab = g.opaque("gemv_mixed", &[w_dt, xn], act(2 * dt), F32);
    let heads = g.opaque("gdn_prologue", &[qkv, ab], Shape::of(&[3 * h_v, d]), F32);
    let o = g.opaque("gdn_step", &[heads, ab], Shape::of(&[h_v, d]), F32);
    let on = g.opaque("gdn_gated_norm", &[o, gate], act(inner), F32);
    stem(g, &format!("{p}.ssm_out.weight"), h, inner, on)
}

/// The Qwen Sparse Attention block (`qsa.rs`): the stacked `attn_qkv`, the
/// indexer's projection, the gate stem, the f16 rope/scale projection, the
/// prologue (norms + partial rope), the KV appends, one `sdpa_vector`, the
/// sigmoid gate, `attn_output`.
fn qsa(g: &mut Graph, cfg: &Config, p: &str, xn: NodeId) -> NodeId {
    let (h, nh, nkv, hd) = (cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let act = |d: usize| Shape::of(&[1, 1, d]);
    let qkv = stem(g, &format!("{p}.attn_qkv.weight"), (2 * nh + 2 * nkv) * hd, h, xn);
    let idx = stem(
        g,
        &format!("{p}.attn_indexer.weight"),
        cfg.indexer_n_heads * cfg.indexer_head_dim,
        h,
        xn,
    );
    let gate = stem(g, &format!("{p}.attn_gate.weight"), nh * hd, h, xn);
    let w_f16 = g.weight(&format!("{p}.attn_f16.weight"), Shape::of(&[nh, h]), F32);
    let scale = g.opaque("gemv_mixed", &[w_f16, xn], act(nh), F32);
    let prep = g.opaque("qsa_prep", &[qkv, scale], Shape::of(&[2 * nh + nkv, hd]), F32);
    let k = g.opaque("copy2d_f32", &[prep], Shape::of(&[nkv, hd]), F32);
    let v = g.opaque("copy2d_f32", &[prep], Shape::of(&[nkv, hd]), F32);
    let ik = g.opaque("copy2d_f32", &[idx], Shape::of(&[cfg.indexer_n_heads, cfg.indexer_head_dim]), F32);
    let o = g.opaque("sdpa_vector", &[prep, k, v, ik], Shape::of(&[nh, hd]), F32);
    let og = g.opaque("sigmoid_mul", &[o, gate], act(nh * hd), F32);
    stem(g, &format!("{p}.attn_output.weight"), h, nh * hd, og)
}

/// The MoE block (`moe.rs`): the router (a mixed gemv), the fused top-k, the
/// streamed tier's slot map and the ids' copy, the routed gate|up in one
/// launch (`mul_mv_id2`), the SwiGLU product, the routed down (`mul_mv_id`),
/// the shared expert (its scalar gate, three stems, the product), the
/// combine.
fn moe(g: &mut Graph, cfg: &Config, p: &str, xn: NodeId) -> NodeId {
    let (h, e, k, mi, si) = (
        cfg.hidden_size,
        cfg.n_experts,
        cfg.n_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.shared_expert_intermediate_size,
    );
    let act = |d: usize| Shape::of(&[1, 1, d]);
    let w_router = g.weight(&format!("{p}.ffn_gate_inp.weight"), Shape::of(&[e, h]), F32);
    let logits = g.opaque("gemv_mixed", &[w_router, xn], act(e), F32);
    let ids = g.opaque("moe_topk", &[logits], Shape::of(&[1, k]), DType::U32);
    let slots = g.opaque("moe_slot_map", &[ids], Shape::of(&[1, k]), DType::U32);
    let slots = g.opaque("copy2d_u32", &[slots], Shape::of(&[1, k]), DType::U32);
    let w_gate = g.weight(&format!("{p}.ffn_gate_exps.weight"), Shape::of(&[e, mi, h]), F32);
    let w_up = g.weight(&format!("{p}.ffn_up_exps.weight"), Shape::of(&[e, mi, h]), F32);
    let gu = g.opaque("mul_mv_id2", &[w_gate, w_up, xn, slots], Shape::of(&[2, 1, k, mi]), F32);
    let gate = g.view("narrow", gu, Shape::of(&[1, k, mi]));
    let up = g.view("narrow", gu, Shape::of(&[1, k, mi]));
    let hmid = g.opaque("silu_mul", &[gate, up], Shape::of(&[1, k, mi]), F32);
    let w_down = g.weight(&format!("{p}.ffn_down_exps.weight"), Shape::of(&[e, h, mi]), F32);
    let y = g.opaque("mul_mv_id", &[w_down, hmid, slots], Shape::of(&[1, k, h]), F32);

    let w_sh = g.weight(&format!("{p}.ffn_gate_inp_shexp.weight"), Shape::of(&[1, h]), F32);
    let sh_gate = g.opaque("gemv_f32", &[w_sh, xn], act(1), F32);
    let sg = stem(g, &format!("{p}.ffn_gate_shexp.weight"), si, h, xn);
    let su = stem(g, &format!("{p}.ffn_up_shexp.weight"), si, h, xn);
    let sm = g.opaque("silu_mul", &[sg, su], act(si), F32);
    let sd = stem(g, &format!("{p}.ffn_down_shexp.weight"), h, si, sm);
    g.opaque("moe_combine", &[y, logits, sd, sh_gate], act(h), F32)
}

/// The per-layer embedding block after `ple_layers` (`ple.rs`): its bracket
/// (no combine — the block writes the carry through its own epilogue), the
/// key and value projections (mixed gemvs), and the composed chain the trace
/// shows — three norms, the conv taps as `bmul`/`badd`, the hash's scalar
/// ops, the SiLU and sigmoid. **Counts from the trace** (module docs).
fn ple(g: &mut Graph, cfg: &Config, p: &str, carry: NodeId) -> NodeId {
    let (h, n, pd) = (cfg.hidden_size, cfg.hc_count, cfg.ple_head_dim);
    let act = |d: usize| Shape::of(&[1, 1, d]);
    let (xn, _) = hc_mix(g, cfg, &format!("{p}.ple"), carry, false);
    let rows = g.input("ple_rows", act(cfg.ple_heads_per_ngram * pd), F32);
    let w_key = g.weight(&format!("{p}.ple_key.weight"), Shape::of(&[n * h, pd], ), F32);
    let key = g.opaque("gemv_mixed", &[w_key, rows], act(n * h), F32);
    let w_val = g.weight(&format!("{p}.ple_value.weight"), Shape::of(&[h, pd]), F32);
    let val = g.opaque("gemv_mixed", &[w_val, rows], act(h), F32);
    let mut t = g.opaque("rmsnorm", &[key], act(n * h), F32);
    t = g.opaque("rmsnorm", &[t, xn], act(n * h), F32);
    t = g.opaque("rmsnorm", &[t], act(n * h), F32);
    for _ in 0..5 {
        t = g.opaque("bmul", &[t, val], act(n * h), F32);
        t = g.opaque("bmul", &[t], act(n * h), F32);
        t = g.opaque("badd", &[t, carry], act(n * h), F32);
    }
    t = g.opaque("silu", &[t], act(n * h), F32);
    t = g.opaque("sigmoid", &[t], act(n * h), F32);
    for f in [
        "sqrt", "abs", "affine", "bmaximum", "bminimum", "bsub", "gt", "lt", "cast_u8",
        "cast_u8", "copy_f32_strided",
    ] {
        t = g.opaque(f, &[t], act(n * h), F32);
    }
    t
}
