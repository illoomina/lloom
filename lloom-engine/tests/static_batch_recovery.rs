//! A static speculative batch that fails part-way hands every sequence back
//! reset, so the next request — prefix reuse on, no cold flag — decodes what
//! a fresh sequence decodes (`batched-decode-design.md` §11.10).
//!
//! The batch runs the pipelined speculative driver, as the
//! `--dspark --batch N --rows private` body does, over a tiny LFM2 on the
//! rewindable arms with an injected failure that lands after the model's own
//! work: a prefill forward that has advanced the conv phase and written KV,
//! or a verify completion that resolved one row's window while the other
//! row's verify is still enqueued, its cache inside an open speculative
//! window. Every row of the failed batch gets the error.
//!
//! Each test then sends both rows' prompts extended by a token, so a
//! sequence returned as it was would be extended in place; and the same
//! prompts again with the cold flag, which resets the sequence, as the
//! reference. The scheduler's `extra` counts the recovery it runs: one fence
//! and one abort for the failed batch, none for any batch that succeeded.
//!
//! Runs on the CPU, and with the `metal` feature natively too, on the
//! rewindable arms and on the private paged pool-spec cache
//! (`--rows private --cache pool-spec`), so the fence waits on real queued
//! verifies. All through the public route a server uses (`spawn_batch`,
//! `BatchHandle::submit`).

use lloom::generation::LogitsProcessor;
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{
    BatchedCausalLM, CausalLM, Model as ModelTrait, Provisional, RaggedRow, Row,
};
use lloom::nn::speculator::{
    Anchor, Draft, LayerTaps, Probes, SeqId, Speculator, TargetProbe, VerifyRowRef,
};
use lloom::nn::{CheckpointPolicy, DecodeOptions, Session, VarBuilder};
use lloom::profile::{Profile, RewindProof, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{
    spawn_batch, BatchConfig, BatchEvent, BatchHandle, BatchMode, BatchRecovery, BatchRequest,
    DecodeParams, Driver, MakeDriver, RingPool, SlotPool, StaticBody,
};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

const K: usize = 2;
const EOS: u32 = 47;
const ROWS: usize = 2;
const MAX: usize = 6;

/// The speculative driver's target shape on the CPU: a five-layer tap,
/// rewindable arms.
fn config() -> Config {
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

/// Where an armed [`Faulty`] fails, after the inner model's call returned.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Site {
    /// `forward` over more than one position: a prompt's prefill.
    Prefill,
    /// `Probes::verify_complete`: one row's verify read back and resolved.
    VerifyComplete,
}

/// The tiny LFM2 with an injected failure: armed with `n` at a [`Site`], the
/// `n`-th call of that site runs the model's own work and then fails, so the
/// state that call moved stays moved. Every other method is the model's.
struct Faulty<P: Profile> {
    inner: Model<P>,
    site: parking_lot::Mutex<Site>,
    left: AtomicUsize,
    failed: AtomicUsize,
}

impl<P: Profile> Faulty<P> {
    fn arm(&self, site: Site, n: usize) {
        *self.site.lock() = site;
        self.left.store(n, Ordering::Release);
    }

    fn failed(&self) -> usize {
        self.failed.load(Ordering::Acquire)
    }

    fn after(&self, site: Site) -> Result<()> {
        if *self.site.lock() != site {
            return Ok(());
        }
        match self.left.load(Ordering::Acquire) {
            0 => Ok(()),
            1 => {
                self.left.store(0, Ordering::Release);
                self.failed.fetch_add(1, Ordering::AcqRel);
                lloom_tensor::bail!("injected: the {site:?} failed after its work ran")
            }
            n => {
                self.left.store(n - 1, Ordering::Release);
                Ok(())
            }
        }
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
        self.inner.forward_rows(tokens, rows)
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
        Probes::<LayerTaps<5>>::verify_rows_enqueue(&self.inner, rows)
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
/// only the target's own argmax, so every round runs a verify.
struct Garbage {
    n: usize,
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
}

/// The scheduler's `extra` on the static arm: the batch thread's one
/// drafter, and a count of the recovery the scheduler runs.
struct Drafter {
    spec: Garbage,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
}

impl<P: Profile> MakeDriver<Faulty<P>> for Drafter {
    fn make_driver<'m>(
        &mut self,
        _model: &'m Faulty<P>,
        _opts: &DecodeOptions,
        _device: &Device,
    ) -> Result<Option<Driver<'m, Faulty<P>>>> {
        Ok(None)
    }
}

impl BatchRecovery for Drafter {
    fn fence(&mut self, device: &Device) -> Result<()> {
        self.fences.fetch_add(1, Ordering::AcqRel);
        lloom_engine::recovery::fence_failed_execution(device)
    }
    fn abort(&mut self) {
        self.aborts.fetch_add(1, Ordering::AcqRel);
    }
}

/// The static body `--dspark --batch N --rows private` serves: every row
/// through the pipelined driver over the batch thread's one drafter.
fn pipelined<P: Profile>() -> StaticBody<Faulty<P>, Drafter> {
    Box::new(|idle, sessions, prompts, specs, on_tokens| {
        let caps: Vec<usize> = specs.iter().map(|s| s.max_tokens).collect();
        let rewind = RewindProof::<P>::ask().expect("a rewindable profile");
        Session::decode_speculative_pipelined_with_caps(
            sessions,
            rewind,
            &mut idle.extra.spec,
            prompts,
            idle.opts,
            &caps,
            &specs[0].eos_ids,
            idle.device,
            on_tokens,
        )
    })
}

struct Served<P: Profile> {
    model: Arc<Faulty<P>>,
    handle: BatchHandle<Faulty<P>, Drafter>,
    thread: std::thread::JoinHandle<()>,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
}

impl<P: Profile> Served<P> {
    /// The static scheduler over two private sequences of `cfg`'s cache on
    /// `device`, prefix reuse on, the pipelined body.
    fn start(device: Device, cfg: Config, dtype: DType) -> Self {
        let inner = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(dtype)
            .model::<P>(&device)
            .unwrap();
        let model = Arc::new(Faulty {
            inner,
            site: parking_lot::Mutex::new(Site::Prefill),
            left: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        });
        let pool = SlotPool::new(ROWS, || {
            State::<P>::new(true, dtype, &cfg, &device).unwrap()
        });
        let fences = Arc::new(AtomicUsize::new(0));
        let aborts = Arc::new(AtomicUsize::new(0));
        let bcfg = BatchConfig {
            device: device.clone(),
            pool,
            rings: RingPool::new(),
            opts: DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(0)
            },
            max_rows: ROWS,
            reuse_prefix: true,
            share_prefix: false,
            prefill_batch: 1,
            prefill_decode: lloom::nn::PrefillDecode::Ride,
            intake_per_tick: lloom_engine::options::DEFAULT_INTAKE_PER_TICK,
            mode: BatchMode::Static(pipelined::<P>()),
            extra: Drafter {
                spec: Garbage { n: ROWS },
                fences: Arc::clone(&fences),
                aborts: Arc::clone(&aborts),
            },
            on_counters: None,
        };
        let (handle, thread) = spawn_batch(Arc::clone(&model), |m: &Faulty<P>| m, bcfg).unwrap();
        Self {
            model,
            handle,
            thread,
            fences,
            aborts,
        }
    }

    fn recovery(&self) -> (usize, usize) {
        (
            self.fences.load(Ordering::Acquire),
            self.aborts.load(Ordering::Acquire),
        )
    }

    /// Hold the scheduler in a closed job until the returned sender fires:
    /// every request submitted meanwhile is waiting when it opens, so the
    /// next static batch takes them together.
    fn hold(&self) -> Sender<()> {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (held_tx, held_rx) = mpsc::channel::<()>();
        assert!(self.handle.run_closed(Box::new(move |_| {
            let _ = held_tx.send(());
            let _ = release_rx.recv();
        })));
        held_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("the gate job runs");
        release_tx
    }

    /// One static batch over `prompts`: each row's tokens, or its error.
    fn batch(
        &self,
        prompts: &[Vec<u32>],
        cold: bool,
    ) -> Vec<std::result::Result<Vec<u32>, String>> {
        let gate = self.hold();
        let rxs: Vec<Receiver<BatchEvent>> = prompts
            .iter()
            .map(|p| {
                self.handle.submit(BatchRequest {
                    prompt_ids: p.clone(),
                    params: DecodeParams {
                        max_tokens: MAX,
                        eos_ids: vec![EOS],
                        cold,
                        ..DecodeParams::default()
                    },
                    cancel: Arc::new(AtomicBool::new(false)),
                })
            })
            .collect();
        drop(gate);
        rxs.iter().map(outcome).collect()
    }

    fn stop(self) {
        self.handle.shutdown();
        self.thread.join().unwrap();
    }
}

fn outcome(rx: &Receiver<BatchEvent>) -> std::result::Result<Vec<u32>, String> {
    let mut streamed = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(60)).expect("an event") {
            BatchEvent::Tokens(ids) => streamed.extend(ids),
            BatchEvent::Done(d) => {
                assert_eq!(d.tokens, streamed, "the stream is the reply");
                return Ok(d.tokens);
            }
            BatchEvent::Error(e) => return Err(e),
        }
    }
}

/// Batch 1 caches both rows' prompts and replies; batch 2 extends them and
/// fails at `site`'s `n`-th call; batch 3 extends batch 2's prompts by one
/// token, reusing whatever each sequence came back with; batch 4 sends
/// batch 3's prompts cold, from reset sequences — the reference. Every row
/// of batch 3 is compared before the recovery's counts are, so a sequence
/// handed back as it was fails on what it decodes.
///
/// A second scheduler runs batch 1 then batch 3 with no failure between:
/// what a clean sequence decodes when batch 3 EXTENDS it. A warm extension
/// is not byte-identical to a cold prefill on every arm (the zero-padded
/// conv context, `sequence-reuse-design.md` §2.5), so it is printed beside
/// each row to tell a stale state from a clean reuse; the assertion is the
/// cold reference, which is what a reset sequence decodes.
fn a_failed_batch_then_the_next<P: Profile>(
    arm: &str,
    device: Device,
    cfg: Config,
    dtype: DType,
    site: Site,
    n: usize,
) {
    let clean = Served::<P>::start(device.clone(), cfg.clone(), dtype);
    let served = Served::<P>::start(device, cfg, dtype);
    let first = [vec![1u32, 2, 3, 4], vec![5u32, 6, 7]];
    let replies = served.batch(&first, false);
    let mut extended: Vec<Vec<u32>> = Vec::new();
    for (i, (p, r)) in first.iter().zip(&replies).enumerate() {
        let r = r.as_ref().expect("batch 1 is served");
        let mut e = p.clone();
        e.extend(r);
        e.push(9 + i as u32);
        extended.push(e);
    }
    assert_eq!(
        served.recovery(),
        (0, 0),
        "a batch that succeeded ran the recovery"
    );
    assert_eq!(
        clean.batch(&first, false),
        replies,
        "the clean run's batch 1"
    );

    served.model.arm(site, n);
    let failed = served.batch(&extended, false);
    assert_eq!(served.model.failed(), 1, "the {site:?} injection fired");
    for (i, f) in failed.iter().enumerate() {
        let why = f
            .as_ref()
            .expect_err("every row of the failed batch gets its error");
        assert!(why.contains("injected"), "row {i}: {why}");
    }

    let next: Vec<Vec<u32>> = extended
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut n = e.clone();
            n.push(20 + i as u32);
            n
        })
        .collect();
    let reused = served.batch(&next, false);
    let reference = served.batch(&next, true);
    let warm = clean.batch(&next, false);
    clean.stop();
    let mut wrong = Vec::new();
    for (i, (got, want)) in reused.iter().zip(&reference).enumerate() {
        let want = want.as_ref().expect("the cold reference is served");
        let warm = warm[i].as_ref().expect("the clean extension is served");
        eprintln!(
            "ARM {arm} SITE {site:?} row {i}: after {got:?} cold {want:?} warm {warm:?} \
             (after==cold {}, after==warm {})",
            got.as_ref().is_ok_and(|g| g == want),
            got.as_ref().is_ok_and(|g| g == warm)
        );
        match got {
            Ok(got) if got == want => {}
            Ok(got) => wrong.push(format!(
                "row {i}: decoded {got:?}, a fresh sequence {want:?}"
            )),
            Err(e) => wrong.push(format!("row {i}: refused: {e}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{arm}: the requests after the failed batch: {}",
        wrong.join("; ")
    );
    assert_eq!(
        served.recovery(),
        (1, 1),
        "(fences, aborts): only the failed batch recovered"
    );
    served.stop();
}

/// The failure lands in row 0's first prefill forward of batch 2, after it
/// advanced the conv phase and wrote KV; row 1 never ran.
#[test]
fn a_static_batch_failing_in_a_prefill_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        config(),
        DType::F32,
        Site::Prefill,
        1,
    );
}

/// The failure lands in row 0's first verify completion of batch 2, after it
/// resolved row 0's window; row 1's verify is still enqueued, its cache
/// inside the window the verify opened.
#[test]
fn a_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        config(),
        DType::F32,
        Site::VerifyComplete,
        1,
    );
}

/// The native shape: a head the Metal attention kernels take, three layers,
/// five taps; `paged` gives each sequence its own page pool read through
/// FlashDecoding, the private pool-spec cache.
#[cfg(feature = "metal")]
fn native_config(paged: bool) -> Config {
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

#[cfg(feature = "metal")]
#[test]
fn natively_a_paged_static_batch_failing_in_a_prefill_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-private-paged",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Site::Prefill,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_paged_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-private-paged",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Site::VerifyComplete,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_rewindable_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset(
) {
    a_failed_batch_then_the_next::<RewindableExact>(
        "metal-rewindable",
        Device::new_metal(0).unwrap(),
        native_config(false),
        DType::F16,
        Site::VerifyComplete,
        1,
    );
}
