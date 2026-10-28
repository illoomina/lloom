//! A request the batch scheduler checks a sequence out for and the row
//! driver then refuses must not cost the pool that sequence
//! (`batched-decode-design.md` §11.2: admission is the pool's).
//!
//! The scheduler validates a request against the K=1 batched loop's body
//! (`Body::BatchedStep`) before it checks a sequence out of its `SlotPool`;
//! the speculative driver the pooled DSpark arm installs
//! (`SpecBatchDecoder`, `app/lfm2.rs` `Speculators::make_driver`) validates
//! the stricter `Body::BatchedWindow` at `admit`, after the checkout. A
//! sampled request passes the first and is refused by the second. Each
//! such refusal must hand the sequence back: the pool's idle count is read
//! after every refusal through a closed job (which runs with nothing
//! live), and the valid greedy request after them must be served.
//!
//! Runs on the CPU over a tiny LFM2 with a garbage drafter, through the
//! public route a server uses (`spawn_batch`, `BatchHandle::submit`).

use lloom::generation::Sampling;
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::speculator::{Draft, LayerTaps, SeqId, Speculator, TargetProbe};
use lloom::nn::{DecodeOptions, Session, SpecBatchDecoder};
use lloom::profile::RewindableExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{
    spawn_batch, BatchConfig, BatchEvent, BatchHandle, BatchMode, BatchRequest, DecodeParams,
    Driver, MakeDriver, RingPool, SlotPool,
};
use lloom_tensor::{DType, Device, Result};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

const K: usize = 2;
const EOS: u32 = 47;
const ROWS: usize = 2;

type Target = Model<RewindableExact>;

/// The speculative driver's target shape: a five-layer tap, rewindable arms.
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

/// A drafter with `n` slots that proposes garbage: greedy verification
/// keeps only the target's own argmax.
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
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

/// The scheduler's `extra`: the speculative driver, as the pooled DSpark
/// arm builds it.
struct Speculative {
    spec: Option<Garbage>,
}

impl MakeDriver<Target> for Speculative {
    fn make_driver<'m>(
        &mut self,
        _model: &'m Target,
        opts: &DecodeOptions,
        device: &Device,
    ) -> Result<Option<Driver<'m, Target>>> {
        let spec = self.spec.take().expect("one driver per scheduler");
        let dec = SpecBatchDecoder::<'m, Target, Garbage, Session<'m, Target>>::new(
            lloom::profile::RewindProof::of(),
            spec,
            opts.clone(),
            device.clone(),
        )?;
        Ok(Some(Box::new(dec)))
    }
}

fn start() -> (
    BatchHandle<Target, Speculative>,
    std::thread::JoinHandle<()>,
) {
    let device = Device::Cpu;
    let cfg = config();
    let model = Arc::new(
        TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(DType::F32)
            .model::<RewindableExact>(&device)
            .unwrap(),
    );
    let pool = SlotPool::new(ROWS, || {
        State::<RewindableExact>::new(true, DType::F32, &cfg, &device).unwrap()
    });
    let bcfg = BatchConfig {
        device: device.clone(),
        pool,
        rings: RingPool::new(),
        opts: DecodeOptions {
            lookahead: 1,
            ..DecodeOptions::new(0)
        },
        max_rows: ROWS,
        reuse_prefix: false,
        share_prefix: false,
        prefill_batch: 1,
        prefill_decode: lloom::nn::PrefillDecode::Ride,
        intake_per_tick: lloom_engine::options::DEFAULT_INTAKE_PER_TICK,
        mode: BatchMode::Continuous,
        extra: Speculative {
            spec: Some(Garbage { n: ROWS }),
        },
        on_counters: None,
    };
    spawn_batch(model, |m: &Target| m, bcfg).unwrap()
}

fn params(temperature: Option<f64>) -> DecodeParams {
    DecodeParams {
        max_tokens: 4,
        eos_ids: vec![EOS],
        temperature,
        seed: 3,
        ..DecodeParams::default()
    }
}

fn submit(handle: &BatchHandle<Target, Speculative>, p: DecodeParams) -> Receiver<BatchEvent> {
    handle.submit(BatchRequest {
        prompt_ids: vec![1, 2, 3, 4],
        params: p,
        cancel: Arc::new(AtomicBool::new(false)),
    })
}

/// The row's outcome: `Ok(tokens)` on its `Done`, `Err(message)` on its
/// `Error`.
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

/// The sequences checked in, read on the scheduler thread with nothing live.
fn idle(handle: &BatchHandle<Target, Speculative>) -> usize {
    handle
        .run_closed_blocking(|idle| idle.pool.idle())
        .expect("the scheduler runs the closed job")
}

/// More sampled requests than the pool has sequences, each refused by the
/// speculative driver's own validation, then a greedy request: every
/// refusal leaves the pool whole, and the greedy request is served.
#[test]
fn a_request_the_speculative_driver_refuses_keeps_its_sequence() {
    let (handle, thread) = start();
    assert_eq!(idle(&handle), ROWS);
    // Non-vacuity: the refusal is the driver's, after the checkout — the
    // request passed the scheduler's own (`BatchedStep`) validation.
    assert!(!matches!(
        params(Some(0.8)).sampling().unwrap(),
        Sampling::ArgMax
    ));
    for i in 0..=ROWS {
        let refused = outcome(&submit(&handle, params(Some(0.8))));
        let why = refused.expect_err("a sampled row is refused by the speculative loop");
        assert!(
            why.contains("decode_spec_batched"),
            "request {i}: refused by the driver, not the pool: {why}"
        );
        assert_eq!(
            idle(&handle),
            ROWS,
            "request {i}: a refused admission kept its sequence out of the pool"
        );
    }
    let served = outcome(&submit(&handle, params(None)));
    let tokens = served.expect("the greedy request after the refusals is served");
    assert!(!tokens.is_empty());
    assert_eq!(idle(&handle), ROWS);
    handle.shutdown();
    thread.join().unwrap();
}
