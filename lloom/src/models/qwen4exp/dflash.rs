//! DFlash on Qwen4-exp: the target side of the hidden-state drafter family
//! (`design/docs/speculation.md` §10.2z-vi).
//!
//! The drafter is the DSpark family's (`models::lfm2_dspark`: projection,
//! bidirectional block backbone, context KV, `begin_at`); what the target
//! supplies is
//!
//! - the probe, [`LayerTaps`]: per tapped layer `l`, the d-wide input the
//!   attention bracket's mix hands layer `l`'s mixer — the HC-contracted
//!   residual the layers before `l` left. The drafter's `target_layer_ids`
//!   name the layers whose output it was trained on, so the cache taps
//!   `id + 1` (`DSparkConfig::tap_layers`), the boundary DeepSpec's exporter
//!   tags (`layers[id + 1].attn_hyper_connection`, its `mix()[1]`);
//! - the verify window, the MTP's own (`Qwen4Exp::window_enqueue`): one
//!   pass over the phase rings, the accept by phase, the streamed tier
//!   settled after the accept;
//! - the shared pair, [`QwenHead`]: the target's quantized `token_embd` and
//!   `lm_head`, which the drafter checkpoint does not ship.

use super::model::{Cache, Qwen4Exp};
use super::moe::ExpertResidency;
use super::mtp::WindowPending;
use crate::models::lfm2_dspark::SharedHead;
use crate::nn::speculator::{Anchor, Draft, LayerTaps, Probes, VerifyRowRef};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor};

/// Qwen4-exp's shared pair: clones of the target's `token_embd` and
/// `lm_head` (`QMatMul`s, their storage shared), so the drafter pins no
/// target alive.
#[derive(Debug, Clone)]
pub struct QwenHead {
    embed: QMatMul,
    lm_head: QMatMul,
    device: Device,
}

impl QwenHead {
    pub fn of<R: ExpertResidency>(model: &Qwen4Exp<R>) -> Self {
        Self {
            embed: model.embed_tokens().clone(),
            lm_head: model.lm_head().clone(),
            device: model.device(),
        }
    }
}

impl SharedHead for QwenHead {
    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed.embedding(ids)
    }

    /// The head reads F32, as the target's own head does after its mixer.
    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head
            .forward(&hidden.to_dtype(DType::F32)?)?
            .to_dtype(DType::F32)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl<R: ExpertResidency, const N: usize> Probes<LayerTaps<N>> for Qwen4Exp<R> {
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Vec<Tensor>)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "qwen4exp verify_and_probe: one sequence; got {} drafts and {} anchors",
                drafts.len(),
                anchors.len()
            )
        }
        let pending = self.window_enqueue::<LayerTaps<N>>(&drafts[0], &anchors[0], cache)?;
        let (accepted, probe) = self.window_complete::<LayerTaps<N>>(pending, cache, eos_ids)?;
        Ok((vec![accepted], probe))
    }

    type Pending = WindowPending;
    type PendingRows = Vec<WindowPending>;

    fn verify_rows_enqueue(
        &self,
        rows: &mut [VerifyRowRef<'_, Cache>],
    ) -> Result<Vec<WindowPending>> {
        rows.iter_mut()
            .map(|r| self.window_enqueue::<LayerTaps<N>>(r.draft, &r.anchor, r.cache))
            .collect()
    }

    fn verify_rows_complete(
        &self,
        pending: Vec<WindowPending>,
        caches: &mut [&mut Cache],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, Vec<Tensor>)>> {
        if pending.len() != caches.len() || pending.len() != eos_sets.len() {
            lloom_tensor::bail!(
                "qwen4exp verify_rows_complete: {} pending, {} caches, {} EOS sets",
                pending.len(),
                caches.len(),
                eos_sets.len()
            )
        }
        pending
            .into_iter()
            .zip(caches.iter_mut())
            .zip(eos_sets)
            .map(|((p, c), eos)| self.window_complete::<LayerTaps<N>>(p, c, eos))
            .collect()
    }

    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Cache,
    ) -> Result<WindowPending> {
        self.window_enqueue::<LayerTaps<N>>(draft, anchor, cache)
    }

    fn verify_complete(
        &self,
        pending: WindowPending,
        cache: &mut Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        self.window_complete::<LayerTaps<N>>(pending, cache, eos_ids)
    }

    /// The taps the last forwards left, every position they ran; none is an
    /// empty set (a drafter fed it refuses it by name).
    fn take_probe(&self, cache: &mut Cache) -> Result<Vec<Tensor>> {
        Ok(cache.take_taps()?.unwrap_or_default())
    }

    fn probe_drops(&self, cache: &Cache) -> u64 {
        cache.tap_drops()
    }
}
