//! Synthetic Qwen4-exp models at tiny geometry, for tests: deterministic
//! weights from a seed, every block real, any residency strategy. Shared by
//! the unit tests here and the crate's integration tests.
#![doc(hidden)]

use super::config::{Config, LayerKind};
use super::deltanet_block::GatedDeltaNet;
use super::hc::HyperConnection;
use super::model::{Layer, Mixer, Qwen4Exp};
use super::moe::{ExpertResidency, MoeLayer, Resident};
use super::mtp::MtpHead;
use super::qsa::QsaAttention;
use crate::nn::RmsNorm;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::Arc;

/// A dense `[out, inp]` matrix from a seed.
pub fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
    let v: Vec<f32> = (0..out * inp)
        .map(|i| ((i as f32 * 0.007 + seed).sin()) * 0.08)
        .collect();
    QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
}

pub fn ones(n: usize, dev: &Device) -> Tensor {
    Tensor::ones(n, DType::F32, dev).unwrap()
}

pub fn wt(shape: (usize, usize), seed: f32, dev: &Device) -> Tensor {
    let (r, c) = shape;
    Tensor::from_vec(
        (0..r * c)
            .map(|i| ((i as f32 * 0.009 + seed).cos()) * 0.05)
            .collect::<Vec<f32>>(),
        (r, c),
        dev,
    )
    .unwrap()
}

/// A stacked `[e, out, inp]` expert weight from a seed, quantized to `dtype`
/// (`F32` keeps every arm exact).
pub fn stacked(
    e: usize,
    out: usize,
    inp: usize,
    seed: f32,
    dev: &Device,
    dtype: GgmlDType,
) -> QTensor {
    let v: Vec<f32> = (0..e * out * inp)
        .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
        .collect();
    let t = Tensor::from_vec(v, (e, out, inp), dev).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

/// The three expert stacks of layer `l` (gate, up, down), the seeds the
/// resident and streamed builds share.
pub fn expert_stacks(cfg: &Config, l: usize, dev: &Device, dtype: GgmlDType) -> [QTensor; 3] {
    let d = cfg.hidden_size;
    let ff = cfg.moe_intermediate_size;
    let ne = cfg.n_experts;
    let s = l as f32 * 0.37;
    [
        stacked(ne, ff, d, 1.0 + s, dev, dtype),
        stacked(ne, ff, d, 100.0 + s, dev, dtype),
        stacked(ne, d, ff, 200.0 + s, dev, dtype),
    ]
}

pub fn hc(cfg: &Config, seed: f32, dev: &Device) -> HyperConnection {
    hc_with(cfg, seed, dev, true)
}

pub fn hc_with(cfg: &Config, seed: f32, dev: &Device, inject: bool) -> HyperConnection {
    let hc_dim = cfg.hc_dim();
    HyperConnection::new(
        ones(hc_dim, dev),
        lin(cfg.hc_lowrank, hc_dim, seed, dev),
        lin(hc_dim, cfg.hc_lowrank, seed + 1.0, dev),
        inject.then(|| lin(cfg.hc_count, hc_dim, seed + 2.0, dev)),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.norm_eps as f32,
    )
    .unwrap()
}

pub fn deltanet(cfg: &Config, dev: &Device) -> GatedDeltaNet {
    let d = cfg.hidden_size;
    let cc = cfg.conv_channels();
    let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let hv = cfg.ssm_dt_rank;
    GatedDeltaNet::new(
        lin(cc, d, 1.0, dev),
        lin(vd, d, 2.0, dev),
        wt((cc, cfg.ssm_conv_kernel), 3.0, dev),
        Tensor::from_vec((0..hv).map(|_| -0.5f32).collect::<Vec<_>>(), hv, dev).unwrap(),
        Tensor::zeros(hv, DType::F32, dev).unwrap(),
        lin(hv, d, 4.0, dev),
        lin(hv, d, 5.0, dev),
        ones(cfg.ssm_state_size, dev),
        lin(d, vd, 6.0, dev),
        cfg,
    )
}

pub fn qsa(cfg: &Config, dev: &Device) -> QsaAttention {
    qsa_seeded(cfg, dev, 0.0)
}

pub fn qsa_seeded(cfg: &Config, dev: &Device, seed: f32) -> QsaAttention {
    let d = cfg.hidden_size;
    let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    QsaAttention::new(
        lin(nh * hd * 2, d, 1.0 + seed, dev),
        lin(nkv * hd, d, 2.0 + seed, dev),
        lin(nkv * hd, d, 3.0 + seed, dev),
        lin(d, nh * hd, 4.0 + seed, dev),
        RmsNorm::new(ones(hd, dev), cfg.norm_eps),
        RmsNorm::new(ones(hd, dev), cfg.norm_eps),
        cfg,
    )
}

/// One layer's MoE around `experts`.
pub fn moe_with<R: ExpertResidency>(cfg: &Config, dev: &Device, experts: R) -> MoeLayer<R> {
    let d = cfg.hidden_size;
    let ne = cfg.n_experts;
    let sh = cfg.shared_expert_intermediate_size;
    MoeLayer::new(
        lin(ne, d, 1.0, dev),
        experts,
        lin(sh, d, 2.0, dev),
        lin(sh, d, 3.0, dev),
        lin(d, sh, 4.0, dev),
        lin(1, d, 5.0, dev),
        cfg,
    )
}

/// Layer `l`'s resident experts.
pub fn resident_experts(cfg: &Config, l: usize, dev: &Device) -> Resident {
    let [g, u, dn] = expert_stacks(cfg, l, dev, GgmlDType::F32);
    Resident::new(Arc::new(g), Arc::new(u), Arc::new(dn))
}

pub fn moe(cfg: &Config, dev: &Device) -> MoeLayer<Resident> {
    moe_with(cfg, dev, resident_experts(cfg, 0, dev))
}

/// A whole model with `experts(l)` supplying layer `l`'s residency strategy.
pub fn model_with<R: ExpertResidency>(
    cfg: &Config,
    dev: &Device,
    mut experts: impl FnMut(usize) -> Result<R>,
) -> Result<Qwen4Exp<R>> {
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let mixer = match cfg.layer_types[l] {
            LayerKind::Linear => Mixer::Linear(deltanet(cfg, dev)),
            LayerKind::Full => Mixer::Full(qsa(cfg, dev)),
        };
        layers.push(Layer::new(
            hc(cfg, l as f32, dev),
            mixer,
            hc(cfg, l as f32 + 50.0, dev),
            moe_with(cfg, dev, experts(l)?),
        ));
    }
    Ok(Qwen4Exp::new(
        QMatMul::Tensor(wt((cfg.vocab_size, cfg.hidden_size), 0.5, dev)),
        layers,
        hc(cfg, 999.0, dev),
        lin(cfg.vocab_size, cfg.hidden_size, 7.0, dev),
        cfg.clone(),
    ))
}

/// A whole model, every expert resident.
pub fn resident_model(cfg: &Config, dev: &Device) -> Qwen4Exp<Resident> {
    model_with(cfg, dev, |l| Ok(resident_experts(cfg, l, dev))).expect("resident build")
}

/// A one-layer MTP head around `experts` (a full-attention layer).
pub fn mtp_head<R: ExpertResidency>(cfg: &Config, dev: &Device, experts: R) -> MtpHead<R> {
    let d = cfg.hidden_size;
    let nd = cfg.hc_dim();
    let layer = Layer::new(
        hc(cfg, 300.0, dev),
        Mixer::Full(qsa_seeded(cfg, dev, 11.0)),
        hc(cfg, 350.0, dev),
        moe_with(cfg, dev, experts),
    );
    MtpHead::new(
        RmsNorm::new(ones(d, dev), cfg.norm_eps),
        RmsNorm::new(ones(nd, dev), cfg.norm_eps),
        lin(d, 2 * d, 21.0, dev),
        layer,
        hc_with(cfg, 400.0, dev, false),
        cfg,
    )
}
