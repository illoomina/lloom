//! Qwen3.5's tensor schema (`docs/porting-models.md` §21.8, INV-2): every
//! tensor the GGUF loader reads at a config, by name and candle row-major
//! shape, in llama.cpp's `qwen35` / `qwen35moe` naming (`load_arch_tensors`).
//! The loader validates a checkpoint against it before a byte loads, and
//! the tiny checkpoints are written from it.

use super::config::Config;
use crate::models::qwen4exp::LayerKind;
use crate::testing::tiny::{Role, Schema, TensorSpec};
use lloom_tensor::quantized::GgmlDType;

/// What a tiny checkpoint stores each class of tensor as.
pub type Kinds = crate::models::qwen4exp::schema::Kinds;

struct Builder<'a> {
    kinds: &'a Kinds,
    out: Vec<TensorSpec>,
}

/// `kind`, or F16 where a row is not a whole number of `kind`'s blocks —
/// the converter's fallback.
fn fit(kind: GgmlDType, row: usize) -> GgmlDType {
    if row.is_multiple_of(kind.block_size()) {
        kind
    } else {
        GgmlDType::F16
    }
}

impl Builder<'_> {
    fn matrix(&mut self, name: String, shape: &[usize]) {
        let fan_in = *shape.last().expect("a matrix has a last dim");
        self.push(
            name,
            shape,
            Role::Matrix { fan_in },
            fit(self.kinds.dense, fan_in),
        );
    }
    fn experts(&mut self, name: String, shape: [usize; 3]) {
        let kind = fit(self.kinds.experts, shape[2]);
        self.push(name, &shape, Role::Matrix { fan_in: shape[2] }, kind);
    }
    fn vector(&mut self, name: String, n: usize, role: Role) {
        self.push(name, &[n], role, GgmlDType::F32);
    }
    fn push(&mut self, name: String, shape: &[usize], role: Role, dtype: GgmlDType) {
        self.out.push(TensorSpec {
            name,
            shape: shape.to_vec(),
            role,
            dtype,
        });
    }

    /// Layer `blk.{l}` of `kind`: the two norms, the mixer, the FFN.
    fn layer(&mut self, cfg: &Config, l: usize, kind: LayerKind) {
        let p = format!("blk.{l}");
        let d = cfg.hidden_size;
        self.vector(format!("{p}.attn_norm.weight"), d, Role::GAIN);
        self.vector(format!("{p}.post_attention_norm.weight"), d, Role::GAIN);
        match kind {
            LayerKind::Linear => {
                let g = cfg.gdn();
                let (cc, vd, hv) = (g.conv_channels(), g.value_dim(), g.dims.h_v);
                self.matrix(format!("{p}.attn_qkv.weight"), &[cc, d]);
                self.matrix(format!("{p}.attn_gate.weight"), &[vd, d]);
                self.matrix(format!("{p}.ssm_alpha.weight"), &[hv, d]);
                self.matrix(format!("{p}.ssm_beta.weight"), &[hv, d]);
                let k = cfg.ssm_conv_kernel;
                self.push(
                    format!("{p}.ssm_conv1d.weight"),
                    &[cc, k],
                    Role::Matrix { fan_in: k },
                    GgmlDType::F32,
                );
                // The decay, stored as `-exp(A_log)`: negative.
                self.vector(
                    format!("{p}.ssm_a"),
                    hv,
                    Role::Range {
                        lo: -1.0,
                        hi: -0.25,
                    },
                );
                self.vector(format!("{p}.ssm_dt.bias"), hv, Role::BIAS);
                self.vector(format!("{p}.ssm_norm.weight"), g.dims.d_v, Role::GAIN);
                self.matrix(format!("{p}.ssm_out.weight"), &[d, vd]);
            }
            LayerKind::Full => {
                let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
                // The query and its output gate, interleaved per head.
                self.matrix(format!("{p}.attn_q.weight"), &[nh * hd * 2, d]);
                self.matrix(format!("{p}.attn_k.weight"), &[nkv * hd, d]);
                self.matrix(format!("{p}.attn_v.weight"), &[nkv * hd, d]);
                self.matrix(format!("{p}.attn_output.weight"), &[d, nh * hd]);
                self.vector(format!("{p}.attn_q_norm.weight"), hd, Role::GAIN);
                self.vector(format!("{p}.attn_k_norm.weight"), hd, Role::GAIN);
            }
        }
        if cfg.is_moe() {
            let (ne, ff, sh) = (
                cfg.n_experts,
                cfg.moe_intermediate_size,
                cfg.shared_expert_intermediate_size,
            );
            self.matrix(format!("{p}.ffn_gate_inp.weight"), &[ne, d]);
            self.experts(format!("{p}.ffn_gate_exps.weight"), [ne, ff, d]);
            self.experts(format!("{p}.ffn_up_exps.weight"), [ne, ff, d]);
            self.experts(format!("{p}.ffn_down_exps.weight"), [ne, d, ff]);
            self.matrix(format!("{p}.ffn_gate_shexp.weight"), &[sh, d]);
            self.matrix(format!("{p}.ffn_up_shexp.weight"), &[sh, d]);
            self.matrix(format!("{p}.ffn_down_shexp.weight"), &[d, sh]);
            // The shared expert's scalar gate: a one-row projection, stored
            // as a vector.
            self.push(
                format!("{p}.ffn_gate_inp_shexp.weight"),
                &[d],
                Role::Matrix { fan_in: d },
                GgmlDType::F32,
            );
        } else {
            let ff = cfg.intermediate_size;
            self.matrix(format!("{p}.ffn_gate.weight"), &[ff, d]);
            self.matrix(format!("{p}.ffn_up.weight"), &[ff, d]);
            self.matrix(format!("{p}.ffn_down.weight"), &[d, ff]);
        }
    }
}

/// Every tensor the loader reads at `cfg` (whose `vocab_size` is the
/// embedding's row count), in load order. `output.weight` is listed: every
/// Qwen3.5 checkpoint is untied (`tie_word_embeddings: false`).
pub fn tensor_schema(cfg: &Config, kinds: &Kinds) -> Schema {
    let mut b = Builder {
        kinds,
        out: Vec::new(),
    };
    let d = cfg.hidden_size;
    b.matrix("token_embd.weight".into(), &[cfg.vocab_size, d]);
    b.vector("output_norm.weight".into(), d, Role::GAIN);
    b.matrix("output.weight".into(), &[cfg.vocab_size, d]);
    for (l, kind) in cfg.layer_types.iter().enumerate() {
        b.layer(cfg, l, *kind);
    }
    // The MTP blocks a file keeps after the stack (`nextn_predict_layers`):
    // each a full-attention layer plus its `nextn.*` projections
    // (llama.cpp `load_block_mtp`).
    for m in 0..cfg.n_nextn_layers {
        let l = cfg.n_layers + m;
        b.layer(cfg, l, LayerKind::Full);
        let p = format!("blk.{l}.nextn");
        b.matrix(format!("{p}.eh_proj.weight"), &[d, 2 * d]);
        b.vector(format!("{p}.enorm.weight"), d, Role::GAIN);
        b.vector(format!("{p}.hnorm.weight"), d, Role::GAIN);
        b.vector(format!("{p}.shared_head_norm.weight"), d, Role::GAIN);
    }
    Schema { tensors: b.out }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real checkpoints' geometry reproduces their headers' tensor
    /// counts: unsloth's Qwen3.8-27B-UD-Q4_K_M keeps the MTP block, 866
    /// tensors = 3 + 48 × 14 + 16 × 11 + a 65th full-attention layer's 11
    /// + its 4 `nextn.*` (read 2026-10-08); unsloth's Qwen3.6-27B-Q4_K_M
    /// strips it, 851 (`nextn_predict_layers` absent from its header, so
    /// `n_nextn_layers` 0); the 35B-A3B's MoE layers carry 8 FFN tensors
    /// each, 733 stripped.
    #[test]
    fn the_schema_counts_match_the_real_checkpoints() {
        let with_mtp = tensor_schema(&Config::qwen3_6_27b(), &Kinds::default());
        assert_eq!(with_mtp.tensors.len(), 851 + 11 + 4);
        let mut stripped = Config::qwen3_6_27b();
        stripped.n_nextn_layers = 0;
        let dense = tensor_schema(&stripped, &Kinds::default());
        assert_eq!(dense.tensors.len(), 851);
        let mut moe_cfg = Config::qwen3_6_35b_a3b();
        moe_cfg.n_nextn_layers = 0;
        let moe = tensor_schema(&moe_cfg, &Kinds::default());
        assert_eq!(moe.tensors.len(), 3 + 30 * (2 + 9 + 8) + 10 * (2 + 6 + 8));
        for s in [&with_mtp, &dense, &moe] {
            let names: std::collections::BTreeSet<&str> =
                s.tensors.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(names.len(), s.tensors.len(), "a name twice");
        }
        assert_eq!(
            dense.get("blk.0.ssm_conv1d.weight").unwrap().shape,
            vec![10240, 4]
        );
        assert_eq!(
            dense.get("blk.3.attn_q.weight").unwrap().shape,
            vec![12288, 5120]
        );
        assert!(dense.get("blk.64.attn_norm.weight").is_none());
        assert_eq!(
            with_mtp.get("blk.64.nextn.eh_proj.weight").unwrap().shape,
            vec![5120, 10240]
        );
        assert_eq!(
            with_mtp.get("blk.64.attn_q.weight").unwrap().shape,
            vec![12288, 5120]
        );
    }
}
