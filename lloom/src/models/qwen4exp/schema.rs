//! Qwen4-exp's tensor schema (`docs/porting-models.md` §21.8, INV-2): every
//! tensor the GGUF loader reads at a config, by name and candle row-major
//! shape. The loader validates a checkpoint against it before it loads a
//! byte (so a converter that drifts is refused with every discrepancy named),
//! and the tiny checkpoints are written from it (so what a tiny model holds
//! is what the loader reads, by construction).

use super::config::{Config, LayerKind};
use crate::testing::tiny::{Role, Schema, TensorSpec};
use lloom_tensor::quantized::GgmlDType;

/// What besides the text model the load reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchemaOptions {
    /// The MTP layer `blk.{n_layers}` and its `nextn.*` head.
    pub mtp: bool,
    /// The N-gram PLE block on `cfg.ple_layers` (the table itself is a
    /// memory map, read by row count, and not listed).
    pub ple: bool,
}

/// What a tiny checkpoint stores each class of tensor as. Vectors, the
/// convolutions and the decays are always F32, as in the real checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kinds {
    /// The dense matrices: projections, stems, routers, the embedding.
    pub dense: GgmlDType,
    /// The routed expert stacks.
    pub experts: GgmlDType,
}

impl Default for Kinds {
    fn default() -> Self {
        Self {
            dense: GgmlDType::F32,
            experts: GgmlDType::F32,
        }
    }
}

struct Builder<'a> {
    kinds: &'a Kinds,
    out: Vec<TensorSpec>,
}

/// `kind`, or F16 where a row is not a whole number of `kind`'s blocks —
/// the converter's fallback (llama.cpp quantizes such a tensor to another
/// kind rather than refuse), so a tiny geometry may mix kinds as a real
/// checkpoint does.
fn fit(kind: GgmlDType, row: usize) -> GgmlDType {
    if row % kind.block_size() == 0 {
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

    /// One HyperConnection bracket `{p}_{norm,down,up[,inject]}`.
    fn hc(&mut self, cfg: &Config, p: &str, inject: bool) {
        let nd = cfg.hc_dim();
        self.vector(format!("{p}_norm.weight"), nd, Role::GAIN);
        self.matrix(format!("{p}_down.weight"), &[cfg.hc_lowrank, nd]);
        self.matrix(format!("{p}_up.weight"), &[nd, cfg.hc_lowrank]);
        if inject {
            self.matrix(format!("{p}_inject.weight"), &[cfg.hc_count, nd]);
        }
    }

    /// Layer `blk.{l}` of `kind`.
    fn layer(&mut self, cfg: &Config, l: usize, kind: LayerKind) {
        let p = format!("blk.{l}");
        let d = cfg.hidden_size;
        self.hc(cfg, &format!("{p}.hc_attn"), true);
        self.hc(cfg, &format!("{p}.hc_ffn"), true);
        match kind {
            LayerKind::Linear => {
                let cc = cfg.conv_channels();
                let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
                let hv = cfg.ssm_dt_rank;
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
                // The decay: negative, as `A = -exp(A_log)` is.
                self.vector(
                    format!("{p}.ssm_a"),
                    hv,
                    Role::Range {
                        lo: -1.0,
                        hi: -0.25,
                    },
                );
                self.vector(format!("{p}.ssm_dt.bias"), hv, Role::BIAS);
                self.vector(
                    format!("{p}.ssm_norm.weight"),
                    cfg.ssm_state_size,
                    Role::GAIN,
                );
                self.matrix(format!("{p}.ssm_out.weight"), &[d, vd]);
            }
            LayerKind::Full => {
                let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
                let ihd = cfg.indexer_head_dim;
                self.matrix(
                    format!("{p}.indexer.q_proj.weight"),
                    &[cfg.indexer_n_heads * ihd, d],
                );
                self.matrix(format!("{p}.indexer.k_proj.weight"), &[ihd, d]);
                self.vector(format!("{p}.indexer.q_norm.weight"), ihd, Role::GAIN);
                self.vector(format!("{p}.indexer.k_norm.weight"), ihd, Role::GAIN);
                // The query and its output gate, interleaved per head.
                self.matrix(format!("{p}.attn_q.weight"), &[nh * hd * 2, d]);
                self.matrix(format!("{p}.attn_k.weight"), &[nkv * hd, d]);
                self.matrix(format!("{p}.attn_v.weight"), &[nkv * hd, d]);
                self.matrix(format!("{p}.attn_output.weight"), &[d, nh * hd]);
                self.vector(format!("{p}.attn_q_norm.weight"), hd, Role::GAIN);
                self.vector(format!("{p}.attn_k_norm.weight"), hd, Role::GAIN);
            }
        }
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
        // The shared expert's scalar gate: a one-row projection, stored as
        // a vector.
        self.push(
            format!("{p}.ffn_gate_inp_shexp.weight"),
            &[d],
            Role::Matrix { fan_in: d },
            GgmlDType::F32,
        );
    }
}

/// Every tensor [`super::Qwen4Exp`]'s GGUF loader reads at `cfg` (whose
/// `vocab_size` is the embedding's row count), in load order.
pub fn tensor_schema(cfg: &Config, opts: &SchemaOptions, kinds: &Kinds) -> Schema {
    let mut b = Builder {
        kinds,
        out: Vec::new(),
    };
    let d = cfg.hidden_size;
    b.matrix("token_embd.weight".into(), &[cfg.vocab_size, d]);
    b.matrix("output.weight".into(), &[cfg.vocab_size, d]);
    b.hc(cfg, "output_hc", false);
    for (l, kind) in cfg.layer_types.iter().enumerate() {
        b.layer(cfg, l, *kind);
    }
    if opts.mtp {
        let l = cfg.n_layers;
        let p = format!("blk.{l}");
        b.layer(cfg, l, LayerKind::Full);
        b.vector(format!("{p}.nextn.enorm.weight"), d, Role::GAIN);
        b.vector(format!("{p}.nextn.hnorm.weight"), cfg.hc_dim(), Role::GAIN);
        b.matrix(format!("{p}.nextn.eh_proj.weight"), &[d, 2 * d]);
        b.hc(cfg, &format!("{p}.nextn.hc_head"), false);
    }
    if opts.ple {
        if let Some(&l) = cfg.ple_layers.first() {
            let p = format!("blk.{l}");
            let nd = cfg.hc_dim();
            let ple_dim = ple_input_width(cfg);
            b.matrix(format!("{p}.ple_key.weight"), &[nd, ple_dim]);
            b.matrix(format!("{p}.ple_value.weight"), &[d, ple_dim]);
            b.vector(format!("{p}.ple_norm_key.weight"), nd, Role::GAIN);
            b.vector(format!("{p}.ple_norm_query.weight"), nd, Role::GAIN);
            b.vector(format!("{p}.ple_norm_conv.weight"), nd, Role::GAIN);
            let k = cfg.ple_conv_kernel;
            b.push(
                format!("{p}.ple_conv1d.weight"),
                &[nd, k],
                Role::Matrix { fan_in: k },
                GgmlDType::F32,
            );
        }
    }
    Schema { tensors: b.out }
}

/// The width of one position's gathered PLE row: one `ple_head_dim` slice
/// per hash head, `heads_per_ngram` heads for each n-gram order from 2 to
/// `ngram_size`.
pub fn ple_input_width(cfg: &Config) -> usize {
    cfg.ple_heads_per_ngram * cfg.ple_ngram_size.saturating_sub(1) * cfg.ple_head_dim
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real checkpoint's geometry reproduces the counts of its header:
    /// 1 224 tensors across the three shards of every quant of
    /// Qwen3.8-Flash-Next, the PLE table among them; 32 in the MTP sidecar.
    #[test]
    fn the_schema_counts_match_the_real_checkpoint() {
        let mut cfg = Config::flash_next();
        cfg.vocab_size = 248320;
        let base = tensor_schema(&cfg, &SchemaOptions::default(), &Kinds::default());
        let ple = tensor_schema(
            &cfg,
            &SchemaOptions {
                ple: true,
                mtp: false,
            },
            &Kinds::default(),
        );
        // + the PLE table, which is not listed.
        assert_eq!(ple.tensors.len() + 1, 1224);
        assert_eq!(ple.tensors.len() - base.tensors.len(), 6);
        let mtp = tensor_schema(
            &cfg,
            &SchemaOptions {
                mtp: true,
                ple: false,
            },
            &Kinds::default(),
        );
        assert_eq!(mtp.tensors.len() - base.tensors.len(), 32);
        assert_eq!(ple_input_width(&cfg), 2560);
        let names: std::collections::BTreeSet<&str> =
            ple.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), ple.tensors.len(), "a name twice");
    }
}
