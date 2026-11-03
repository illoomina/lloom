//! The tiny LFM2 with an injected failure, shared by the scheduler's
//! failed-batch tests: `static_batch_recovery` (the static error branch,
//! `batched-decode-design.md` §11.10) and `continuous_batch_recovery` (the
//! continuous one, §11.11). Each file uses a part of it.
#![allow(dead_code)]

use lloom::generation::LogitsProcessor;
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::nn::model::{
    BatchedCausalLM, CausalLM, Model as ModelTrait, Provisional, RaggedRow, Row,
};
use lloom::nn::speculator::{
    Anchor, Draft, LayerTaps, Probes, SeqId, Speculator, TargetProbe, VerifyRowRef,
};
use lloom::nn::ReuseAction;
use lloom::nn::{CheckpointPolicy, VarBuilder};
use lloom::profile::Profile;
use lloom_engine::{BatchDone, BatchEvent, BatchHandle, BatchRequest, DecodeParams};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

pub(super) const K: usize = 2;
pub(super) const EOS: u32 = 47;

/// The speculative driver's target shape on the CPU: a five-layer tap,
/// rewindable arms.
pub(super) fn config() -> Config {
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
        eos_token_id: Some(EOS),
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: K + 1 },
        tap_layers: vec![1, 2, 3, 4, 5],
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// [`config`] with 24 positions: the RoPE tables and every cache's
/// `kv_capacity` end there, so a prompt or a verify window can reach the
/// ceiling in a few tokens.
pub(super) fn small_config() -> Config {
    Config {
        max_position_embeddings: 24,
        ..config()
    }
}

/// The native shape: a head the Metal attention kernels take, three layers,
/// five taps; `paged` reads the KV through page tables and FlashDecoding,
/// the shape both a private pool-spec cache and a row of a shared
/// `BatchPool` take.
#[cfg(feature = "metal")]
pub(super) fn native_config(paged: bool) -> Config {
    let mut c = Config {
        vocab_size: 48,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(EOS),
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: K + 1 },
        tap_layers: vec![0, 1, 2, 0, 1],
        tap_dtype: DType::F32,
        ..Default::default()
    };
    if paged {
        c.kv_append = KvAppend::Paged;
        c.attn_impl = lloom::models::lfm2::AttnImpl::FlashDecoding;
        c.flash_page_size = 32;
        c.flash_pages_per_chunk = 1;
    }
    c
}

/// [`native_config`] with 64 positions, two of its 32-position pages: the
/// native arms' ceiling, as [`small_config`] is the CPU's.
#[cfg(feature = "metal")]
pub(super) fn small_native_config(paged: bool) -> Config {
    Config {
        max_position_embeddings: 64,
        ..native_config(paged)
    }
}

/// Where an armed [`Faulty`] fails, after the inner model's call returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Site {
    /// `forward` over more than one position: a prompt's prefill.
    Prefill,
    /// `Probes::verify_complete`: one row's verify read back and resolved.
    VerifyComplete,
    /// `forward_rows`: a batched decode step over every live row, encoded.
    Rows,
    /// `Probes::verify_rows_enqueue`: a speculative round's one ragged
    /// verify over every live row, encoded and committed, its windows open.
    VerifyRowsEnqueue,
    /// As [`Site::VerifyRowsEnqueue`], the failure reported as having
    /// touched no state although the windows are open: what a model's
    /// refusal from inside the verify pass reports about that pass alone.
    VerifyRowsEnqueueSaysUntouched,
    /// `forward` over more than one position, run by the real model at the
    /// end of its RoPE table: the first layer runs and writes its state,
    /// the second fails. The model's own error, reported touched.
    PrefillPastRope,
}

/// What was still on the device when the injected failure returned: the
/// dispatches encoded into the open command buffer, and whether the last
/// committed buffer had completed. Zero and `None` off Metal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Queued {
    pub open_dispatches: usize,
    pub last_committed_complete: Option<bool>,
}

/// The tiny LFM2 with an injected failure: armed with `n` at a [`Site`], the
/// `n`-th call of that site runs the model's own work and then fails, so the
/// state that call moved stays moved. Every other method is the model's.
pub(super) struct Faulty<P: Profile> {
    pub inner: Model<P>,
    device: Device,
    site: parking_lot::Mutex<Site>,
    left: AtomicUsize,
    failed: AtomicUsize,
    queued: parking_lot::Mutex<Vec<Queued>>,
    /// The first position past the RoPE table ([`Site::PrefillPastRope`]).
    rope_end: usize,
}

impl<P: Profile> Faulty<P> {
    pub(super) fn around(inner: Model<P>, device: &Device) -> Self {
        Self {
            inner,
            device: device.clone(),
            site: parking_lot::Mutex::new(Site::Prefill),
            left: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            queued: parking_lot::Mutex::new(Vec::new()),
            rope_end: usize::MAX,
        }
    }

    /// The model's RoPE table ends at `rope_end` (its
    /// `max_position_embeddings`).
    pub(super) fn with_rope_end(mut self, rope_end: usize) -> Self {
        self.rope_end = rope_end;
        self
    }

    pub(super) fn arm(&self, site: Site, n: usize) {
        *self.site.lock() = site;
        self.left.store(n, Ordering::Release);
    }

    pub(super) fn failed(&self) -> usize {
        self.failed.load(Ordering::Acquire)
    }

    /// What was queued at each injected failure, in order.
    pub(super) fn queued(&self) -> Vec<Queued> {
        self.queued.lock().clone()
    }

    fn queued_now(&self) -> Queued {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &self.device {
            return Queued {
                open_dispatches: m.open_dispatches(),
                last_committed_complete: m
                    .last_committed_marker()
                    .ok()
                    .flatten()
                    .map(|mk| m.marker_complete(&mk)),
            };
        }
        Queued::default()
    }

    /// Whether this call of `site` is the armed one.
    fn fires(&self, site: Site) -> bool {
        if *self.site.lock() != site {
            return false;
        }
        match self.left.load(Ordering::Acquire) {
            0 => false,
            1 => {
                self.left.store(0, Ordering::Release);
                true
            }
            n => {
                self.left.store(n - 1, Ordering::Release);
                false
            }
        }
    }

    fn note_failure(&self) {
        self.queued.lock().push(self.queued_now());
        self.failed.fetch_add(1, Ordering::AcqRel);
    }

    fn after(&self, site: Site) -> Result<()> {
        if !self.fires(site) {
            return Ok(());
        }
        self.note_failure();
        let error =
            lloom_tensor::Error::msg(format!("injected: the {site:?} failed after its work ran"));
        Err(match site {
            Site::VerifyRowsEnqueueSaysUntouched => {
                error.with_state_touched(lloom_tensor::StateTouched::No)
            }
            _ => error,
        })
    }
}

impl<P: Profile> ModelTrait for Faulty<P> {
    type Config = Config;
    fn from_weights(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("built around a model")
    }
}

impl<P: Profile> CausalLM for Faulty<P> {
    type Cache = State<P>;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        let out = self.inner.forward_embeds(embeds, pos, cache)?;
        if embeds.dim(1)? > 1 {
            self.after(Site::Prefill)?;
        }
        Ok(out)
    }
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        if tokens.dim(1)? > 1 && self.fires(Site::PrefillPastRope) {
            self.note_failure();
            return match self.inner.forward(tokens, self.rope_end, cache) {
                Ok(_) => lloom_tensor::bail!("the forward past the RoPE table succeeded"),
                Err(e) => Err(e.context("injected: the prefill ran past the RoPE table")),
            };
        }
        let out = self.inner.forward(tokens, pos, cache)?;
        if tokens.dim(1)? > 1 {
            self.after(Site::Prefill)?;
        }
        Ok(out)
    }
    #[cfg(feature = "metal")]
    fn guarded_embedding_layout(&self) -> Result<lloom_tensor::quantized::GuardedEmbeddingLayout> {
        self.inner.guarded_embedding_layout()
    }
    #[cfg(feature = "metal")]
    fn check_guarded_embedding(&self, out: &Tensor) -> Result<()> {
        self.inner.check_guarded_embedding(out)
    }
    #[cfg(feature = "metal")]
    fn embed_guarded_into(
        &self,
        tokens: &lloom_tensor::metal_backend::sampling::GuardedSample<'_>,
        out: &Tensor,
    ) -> Result<()> {
        self.inner.embed_guarded_into(tokens, out)
    }
    #[cfg(feature = "metal")]
    fn forward_prepared_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        self.inner.forward_prepared_embeds(embeds, pos, cache)
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.inner.prepare_pass(cache, seq_len)
    }
    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        self.inner.note_input_ids(cache, ids)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        self.inner.checkpoint_at(cache, pos)
    }
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        self.inner.open_checkpoint_request(cache, policy, end)
    }
    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.close_checkpoint_request(cache)
    }
    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        self.inner.pin_checkpoint(cache, pos)
    }
    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        self.inner.pins_checkpoints(cache)
    }
    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        self.inner.pinned_checkpoint(cache)
    }
    fn host_feed(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Option<lloom::nn::ngram_memory::HostFeed>> {
        self.inner.host_feed(cache)
    }
    fn host_feed_end(&self, cache: &mut Self::Cache, published: &[u32]) -> Result<()> {
        self.inner.host_feed_end(cache, published)
    }
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.settle_pending(cache)
    }
    fn provisional(&self, cache: &mut Self::Cache, mode: Provisional) -> Result<Provisional> {
        self.inner.provisional(cache, mode)
    }
    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        self.inner.settle_step(cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

impl<P: Profile> lloom::profile::Shaped for Faulty<P> {
    type Profile = P;
}

impl<P: Profile> BatchedCausalLM for Faulty<P> {
    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Self::Cache>]) -> Result<Tensor> {
        let out = self.inner.forward_rows(tokens, rows)?;
        self.after(Site::Rows)?;
        Ok(out)
    }
    fn forward_ragged(
        &self,
        tokens: &Tensor,
        rows: &mut [RaggedRow<'_, Self::Cache>],
    ) -> Result<Tensor> {
        self.inner.forward_ragged(tokens, rows)
    }
}

impl<P: Profile> Probes<LayerTaps<5>> for Faulty<P> {
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, <LayerTaps<5> as TargetProbe>::Output)> {
        Probes::<LayerTaps<5>>::verify_and_probe(&self.inner, drafts, anchors, cache, eos_ids)
    }
    fn verify_and_probe_sampled(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
        sampler: &mut LogitsProcessor,
    ) -> Result<(Vec<Vec<u32>>, <LayerTaps<5> as TargetProbe>::Output)> {
        Probes::<LayerTaps<5>>::verify_and_probe_sampled(
            &self.inner,
            drafts,
            anchors,
            cache,
            eos_ids,
            sampler,
        )
    }
    fn prepare_verify(&self, cache: &mut Self::Cache, t: usize) -> Result<()> {
        Probes::<LayerTaps<5>>::prepare_verify(&self.inner, cache, t)
    }
    type Pending = <Model<P> as Probes<LayerTaps<5>>>::Pending;
    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Self::Cache,
    ) -> Result<Self::Pending> {
        Probes::<LayerTaps<5>>::verify_enqueue(&self.inner, draft, anchor, cache)
    }
    fn verify_complete(
        &self,
        pending: Self::Pending,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, <LayerTaps<5> as TargetProbe>::Output)> {
        let out = Probes::<LayerTaps<5>>::verify_complete(&self.inner, pending, cache, eos_ids)?;
        self.after(Site::VerifyComplete)?;
        Ok(out)
    }
    type PendingRows = <Model<P> as Probes<LayerTaps<5>>>::PendingRows;
    fn verify_rows_enqueue(
        &self,
        rows: &mut [VerifyRowRef<'_, Self::Cache>],
    ) -> Result<Self::PendingRows> {
        let pending = Probes::<LayerTaps<5>>::verify_rows_enqueue(&self.inner, rows)?;
        self.after(Site::VerifyRowsEnqueue)?;
        self.after(Site::VerifyRowsEnqueueSaysUntouched)?;
        Ok(pending)
    }
    fn verify_rows_complete(
        &self,
        pending: Self::PendingRows,
        caches: &mut [&mut Self::Cache],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, <LayerTaps<5> as TargetProbe>::Output)>> {
        Probes::<LayerTaps<5>>::verify_rows_complete(&self.inner, pending, caches, eos_sets)
    }
    fn take_probe(&self, cache: &mut Self::Cache) -> Result<<LayerTaps<5> as TargetProbe>::Output> {
        Probes::<LayerTaps<5>>::take_probe(&self.inner, cache)
    }
    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        Probes::<LayerTaps<5>>::probe_drops(&self.inner, cache)
    }
    fn verifies_anchor_alone(&self) -> bool {
        Probes::<LayerTaps<5>>::verifies_anchor_alone(&self.inner)
    }
}

/// A drafter with `n` slots that proposes garbage: greedy verification keeps
/// only the target's own argmax, so every round runs a verify. Counts the
/// aborts that reach it.
pub(super) struct Garbage {
    pub n: usize,
    pub aborts: Arc<AtomicUsize>,
}

impl Garbage {
    pub(super) fn new(n: usize) -> Self {
        Self {
            n,
            aborts: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Speculator for Garbage {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;
    fn n_seq(&self) -> usize {
        self.n
    }
    fn max_draft(&self) -> usize {
        K
    }
    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active.iter().map(|_| Draft::host(vec![0; K])).collect())
    }
    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<LayerTaps<5> as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
    fn abort(&mut self) {
        self.aborts.fetch_add(1, Ordering::AcqRel);
    }
}

/// Hold the scheduler in a closed job until the returned sender fires: every
/// request submitted meanwhile is waiting when it opens, so the scheduler
/// takes them together (one static batch, or one admission into the
/// continuous driver).
pub(super) fn hold<M: BatchedCausalLM, E>(handle: &BatchHandle<M, E>) -> Sender<()> {
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let (held_tx, held_rx) = mpsc::channel::<()>();
    assert!(handle.run_closed(Box::new(move |_| {
        let _ = held_tx.send(());
        let _ = release_rx.recv();
    })));
    held_rx
        .recv_timeout(Duration::from_secs(60))
        .expect("the gate job runs");
    release_tx
}

/// `prompts` submitted together, greedy, `max` tokens each, `cold` resetting
/// the routed sequence first: each row's tokens, or its error.
pub(super) fn round<M: BatchedCausalLM, E>(
    handle: &BatchHandle<M, E>,
    prompts: &[Vec<u32>],
    cold: bool,
    max: usize,
) -> Vec<std::result::Result<Vec<u32>, String>> {
    submit_together(handle, prompts, cold, max)
        .iter()
        .map(outcome)
        .collect()
}

/// A served row's tokens, the prepare's action and the positions it kept.
pub(super) type Reused = (Vec<u32>, ReuseAction, usize);

/// [`round`], each served row with how its sequence was reused: the
/// prepare's action and the positions it kept.
pub(super) fn round_reuse<M: BatchedCausalLM, E>(
    handle: &BatchHandle<M, E>,
    prompts: &[Vec<u32>],
    cold: bool,
    max: usize,
) -> Vec<std::result::Result<Reused, String>> {
    submit_together(handle, prompts, cold, max)
        .iter()
        .map(|rx| outcome_done(rx).map(|d| (d.tokens, d.reuse, d.reused)))
        .collect()
}

fn submit_together<M: BatchedCausalLM, E>(
    handle: &BatchHandle<M, E>,
    prompts: &[Vec<u32>],
    cold: bool,
    max: usize,
) -> Vec<Receiver<BatchEvent>> {
    let gate = hold(handle);
    let rxs: Vec<Receiver<BatchEvent>> = prompts
        .iter()
        .map(|p| {
            handle.submit(BatchRequest {
                prompt_ids: p.clone(),
                params: DecodeParams {
                    max_tokens: max,
                    eos_ids: vec![EOS],
                    cold,
                    ..DecodeParams::default()
                },
                cancel: Arc::new(AtomicBool::new(false)),
            })
        })
        .collect();
    drop(gate);
    rxs
}

/// A row's outcome: its tokens (checked against what it streamed), or its
/// error.
pub(super) fn outcome(rx: &Receiver<BatchEvent>) -> std::result::Result<Vec<u32>, String> {
    outcome_done(rx).map(|d| d.tokens)
}

fn outcome_done(rx: &Receiver<BatchEvent>) -> std::result::Result<BatchDone, String> {
    let mut streamed = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(60)).expect("an event") {
            BatchEvent::Tokens(ids) => streamed.extend(ids),
            BatchEvent::Done(d) => {
                assert_eq!(d.tokens, streamed, "the stream is the reply");
                return Ok(*d);
            }
            BatchEvent::Error(e) => return Err(e),
        }
    }
}

/// A row of `prompt` past `len` tokens: `prompt` extended by ids that are
/// not the EOS.
pub(super) fn padded(prompt: &[u32], len: usize) -> Vec<u32> {
    let mut p = prompt.to_vec();
    let mut t = 11u32;
    while p.len() < len {
        p.push(t);
        t = (t + 7) % EOS;
    }
    p
}
