//! The MTP per-position arms (`expert-streaming.md` §20.16d), each against
//! the incumbent it must equal and with its engagement shown:
//!
//! - **the batched QSA window** (`QWEN4EXP_QSA_WINDOW=batched`): the verify
//!   window's positions in one launch per kernel, every carry the per-row
//!   window's bit for bit, inside the indexer's dense range and past it;
//! - **the verify key by sequence** (`QWEN4EXP_MTP_VERIFY_KEY=variant`): a
//!   window's pass keyed by the rows that complete an indexer block, the plan
//!   kind and the dense range, so its plans hold where one key per length is
//!   evicted — the streams and carries the pool path's;
//! - **the folded observe** (`QWEN4EXP_MTP_FOLD`): the head run on every
//!   window row inside the verify, the accepted row's draft taken after the
//!   accept — the greedy stream, the head's lane keyed with exactly the
//!   accepted rows, the drafts the observe's;
//! - **the staged readback counted**: a device-planned round's one readback
//!   reaches `RoundTiming::verify_readbacks`;
//! - **a prefill in units** (`DecodeOptions::prefill_ubatch`) seeds the MTP
//!   lane unit by unit instead of refusing the first unit.
//!
//! The tiny geometry with `head_dim = 32`: the fused QSA paths (and so both
//! window arms) need a head width the vector kernel carries, which the tiny
//! default (16) is not. Metal; one binary, the arms are process-wide.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{qsa, Config, MtpDrafter, Qwen4Exp, Resident, State, Streamed};
use lloom::models::qwen4exp::{ExpertResidency, MtpHead};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm};
use lloom::nn::speculator::{Draft, KvSource, NextnHidden, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::PathBuf;

/// The arms, the recorder and the dispatch clock are process globals.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg.eos_token_id = 127;
    cfg.head_dim = 32;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

/// Every arm off: the incumbent.
fn incumbent() {
    qsa::set_qsa_window_batched(false);
    lloom::models::qwen4exp::mtp::set_verify_key_variant(false);
    lloom::models::qwen4exp::mtp::set_mtp_fold(false);
}

/// The resident tiny model with the QSA indexer, and an MTP head it shares
/// with its drafter.
fn model(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<RewindableExact, Resident>> {
    let mut m = testing::model_with_indexer::<RewindableExact, _>(cfg, dev, |l| {
        Ok(testing::resident_experts(cfg, l, dev))
    })?;
    m.set_mtp(testing::mtp_head::<CrownExact, _>(
        cfg,
        dev,
        testing::resident_experts(cfg, 7, dev),
    ));
    Ok(m)
}

fn greedy<R: ExpertResidency>(
    m: &Qwen4Exp<RewindableExact, R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    max: usize,
) -> Result<Vec<u32>> {
    let mut session = Session::new(m, RewindableExact::single(&m, &dev, 8)?);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &mut |_| Ok(()),
    )?;
    Ok(out.ids)
}

/// A prompt of `len` tokens whose greedy stream is at least `want` long.
fn prompt_with_stream<R: ExpertResidency>(
    m: &Qwen4Exp<RewindableExact, R>,
    cfg: &Config,
    dev: &Device,
    len: u32,
    want: usize,
    max: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    for seed in 0..16u32 {
        let prompt: Vec<u32> = (0..len).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let ids = greedy(m, cfg, dev, &prompt, max)?;
        if ids.len() >= want {
            return Ok((prompt, ids));
        }
    }
    lloom_tensor::bail!("no prompt of {len} gave a greedy stream of {want}")
}

/// Proposes the greedy stream's next `k`, wrong on every `wrong`-th round,
/// so the accepted lengths vary; records every carry it is handed.
struct Scripted {
    stream: Vec<u32>,
    produced: usize,
    k: usize,
    wrong: usize,
    round: usize,
    carries: Vec<Vec<u32>>,
}

impl Scripted {
    fn new(stream: &[u32], k: usize, wrong: usize) -> Self {
        Self {
            stream: stream.to_vec(),
            produced: 0,
            k,
            wrong,
            round: 0,
            carries: Vec::new(),
        }
    }
}

impl Speculator for Scripted {
    type Probe = NextnHidden;
    type Family = lloom::nn::speculator::Mtp;
    fn n_seq(&self) -> usize {
        1
    }
    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }
    fn max_draft(&self) -> usize {
        self.k
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        self.produced = 0;
        self.round = 0;
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        self.round += 1;
        let bad = self.wrong > 0 && self.round.is_multiple_of(self.wrong);
        let toks = (0..self.k)
            .map(|i| {
                let t = self.stream.get(self.produced + i).copied().unwrap_or(0);
                if bad {
                    (t + 1) % 120
                } else {
                    t
                }
            })
            .collect();
        Ok(vec![Draft::host(toks)])
    }
    fn observe(&mut self, _: SeqId, probe: &Tensor, accepted: &[u32]) -> Result<()> {
        if !accepted.is_empty() {
            self.produced += accepted.len();
        }
        let v: Vec<f32> = probe.flatten_all()?.to_vec1()?;
        self.carries.push(v.into_iter().map(f32::to_bits).collect());
        Ok(())
    }
}

/// A drafter's proposals, recorded round by round as it hands them over.
struct Recording<S> {
    inner: S,
    drafts: Vec<Vec<u32>>,
}

impl<S: Speculator<Probe = NextnHidden>> Speculator for Recording<S> {
    type Probe = NextnHidden;
    type Family = S::Family;
    fn n_seq(&self) -> usize {
        self.inner.n_seq()
    }
    fn kv_source(&self) -> KvSource {
        self.inner.kv_source()
    }
    fn max_draft(&self) -> usize {
        self.inner.max_draft()
    }
    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        self.inner.begin(seq, prompt)
    }
    fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
        self.inner.begin_at(seq, prompt, kept)
    }
    fn context_len(&self, seq: SeqId) -> Option<usize> {
        self.inner.context_len(seq)
    }
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let drafts = self.inner.draft(active)?;
        for d in &drafts {
            match &d.tokens {
                lloom::nn::speculator::DraftTokens::Host(v) => self.drafts.push(v.clone()),
                lloom::nn::speculator::DraftTokens::Device(t) => {
                    self.drafts.push(t.flatten_all()?.to_vec1()?)
                }
            }
        }
        Ok(drafts)
    }
    fn want_confidence(&mut self, wanted: bool) {
        self.inner.want_confidence(wanted)
    }
    fn observe(&mut self, seq: SeqId, probe: &Tensor, accepted: &[u32]) -> Result<()> {
        self.inner.observe(seq, probe, accepted)
    }
}

struct Run {
    ids: Vec<u32>,
    accepted: Vec<usize>,
    verify_readbacks: Vec<u64>,
    window_device: u64,
}

fn speculate<R: ExpertResidency, S: Speculator<Probe = NextnHidden>>(
    m: &Qwen4Exp<RewindableExact, R>,
    cfg: &Config,
    dev: &Device,
    spec: &mut S,
    prompt: &[u32],
    opts: &DecodeOptions,
) -> Result<Run> {
    let k = spec.max_draft();
    let mut cache = State::<RewindableExact>::in_ring(cfg, &m.spec_pools(1, k, dev))?;
    cache.want_nextn = true;
    let mut session = Session::new(m, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        opts,
        &eos,
        &mut |_| Ok(()),
    )?;
    Ok(Run {
        ids: out.ids,
        accepted: out.stats.rounds.iter().map(|r| r.accepted).collect(),
        verify_readbacks: out
            .stats
            .rounds
            .iter()
            .map(|r| r.timing.verify_readbacks)
            .collect(),
        window_device: session.cache_mut().window_device,
    })
}

/// The batched window's carries are the per-row window's, bit for bit, at
/// K = 1 and 2, on windows inside the indexer's dense range (19 positions
/// at the tiny budget) and past it, and the stream is greedy's.
#[test]
fn the_batched_qsa_window_is_the_per_row_window() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let dev = Device::new_metal(0)?;
    let m = model(&cfg, &dev)?;
    let max = 20;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 14, 12, max)?;
    for k in [1usize, 2] {
        let mut per_arm = Vec::new();
        for batched in [false, true] {
            qsa::set_qsa_window_batched(batched);
            let before = qsa::qsa_window_counts();
            let mut spec = Scripted::new(&want, k, 3);
            let run = speculate(&m, &cfg, &dev, &mut spec, &prompt, &DecodeOptions::new(max))?;
            let after = qsa::qsa_window_counts();
            assert_eq!(
                run.ids, want,
                "k = {k} batched = {batched}: not the greedy stream"
            );
            let (windows, sparse) = (after.0 - before.0, after.1 - before.1);
            println!("QSAWINDOW k={k} batched={batched} rounds={} windows={windows} sparse_rows={sparse}", run.accepted.len());
            if batched {
                assert!(windows > 0, "k = {k}: the batched arm never ran");
                assert!(sparse > 0, "k = {k}: no window row past the dense range");
            } else {
                assert_eq!(
                    windows, 0,
                    "k = {k}: the per-row arm ran the batched window"
                );
            }
            per_arm.push((run.accepted, spec.carries));
        }
        qsa::set_qsa_window_batched(false);
        let ((acc_r, car_r), (acc_b, car_b)) = (&per_arm[0], &per_arm[1]);
        assert_eq!(acc_r, acc_b, "k = {k}: the rounds differ between the arms");
        assert_eq!(car_r.len(), car_b.len());
        for (i, (a, b)) in car_r.iter().zip(car_b).enumerate() {
            if let Some(j) = (0..a.len()).find(|&j| a[j] != b[j]) {
                panic!(
                    "k = {k}: observe {i}'s carry element {j} is {} batched, {} per row",
                    f32::from_bits(b[j]),
                    f32::from_bits(a[j])
                );
            }
        }
    }
    Ok(())
}

/// Three requests through the plan cache with the window's accepted length
/// alternating, so its position moves through every indexer block phase.
/// Keyed by its sequence, no verify plan is evicted and the passes are
/// served; the streams and carries are the pool path's either way.
#[test]
fn a_verify_keyed_by_its_sequence_keeps_its_plans() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let max = 16;
    let dev = Device::new_metal(0)?;
    let m = model(&cfg, &dev)?;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 7, 10, max)?;
    let mut pool = Scripted::new(&want, 1, 2);
    let pool_run = speculate(&m, &cfg, &dev, &mut pool, &prompt, &DecodeOptions::new(max))?;
    assert_eq!(pool_run.ids, want);
    let mut evicted = Vec::new();
    for variant in [false, true] {
        lloom::models::qwen4exp::mtp::set_verify_key_variant(variant);
        let dev = Device::new_metal(0)?;
        let md = metal(&dev);
        md.enable_pass_plans(ArenaLayout::Packed);
        let m = model(&cfg, &dev)?;
        for request in 0..3 {
            let mut spec = Scripted::new(&want, 1, 2);
            let run = speculate(&m, &cfg, &dev, &mut spec, &prompt, &DecodeOptions::new(max))?;
            assert_eq!(
                run.ids, want,
                "variant = {variant} request {request}: the stream moved"
            );
            assert_eq!(run.accepted, pool_run.accepted);
            assert_eq!(
                spec.carries, pool.carries,
                "variant = {variant} request {request}: a carry moved"
            );
        }
        let c = md.pass_plan_counters().expect("the plan cache is on");
        println!("VERIFYKEY variant={variant} {c:?}");
        assert!(c.served > 0, "variant = {variant}: nothing was served");
        evicted.push(c.evicted);
    }
    lloom::models::qwen4exp::mtp::set_verify_key_variant(false);
    assert_eq!(
        evicted[1], 0,
        "keyed by its sequence, a verify plan was still evicted"
    );
    assert!(
        evicted[0] > 0,
        "the incumbent key evicted nothing: the fixture does not move the window's sequence"
    );
    Ok(())
}

/// The observe folded into the verify drafts what the observe drafts: the
/// same rounds, the greedy stream, the head's lane keyed with exactly the
/// accepted rows; at K = 1 and 2, and under the plan cache.
#[test]
fn the_folded_observe_drafts_what_the_observe_drafts() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let max = 20;
    // The reference is the pool path's greedy stream, for both cache arms.
    let (prompt, want) = {
        let dev = Device::new_metal(0)?;
        let m = model(&cfg, &dev)?;
        prompt_with_stream(&m, &cfg, &dev, 9, 10, max)?
    };
    for plans in [false, true] {
        let dev = Device::new_metal(0)?;
        if plans {
            metal(&dev).enable_pass_plans(ArenaLayout::Packed);
        }
        let m = model(&cfg, &dev)?;
        let head = m.mtp_head().expect("the model carries the head");
        for k in [1usize, 2] {
            let mut rounds = Vec::new();
            for fold in [false, true] {
                lloom::models::qwen4exp::mtp::set_mtp_fold(fold);
                let mut mtp =
                    MtpDrafter::new(head, m.embed_tokens(), m.lm_head(), &cfg, &dev).with_k(k);
                let mut rec = Recording {
                    inner: &mut mtp,
                    drafts: Vec::new(),
                };
                let run = speculate(&m, &cfg, &dev, &mut rec, &prompt, &DecodeOptions::new(max))?;
                let drafted = rec.drafts;
                assert_eq!(
                    run.ids, want,
                    "plans {plans} k = {k} fold = {fold}: not the greedy stream"
                );
                let (folded, fallbacks) = mtp.fold_stats();
                println!(
                    "MTPFOLD plans={plans} k={k} fold={fold} rounds={} accepted={:?} folded={folded} fallbacks={fallbacks} rows_seen={} pos={}",
                    run.accepted.len(),
                    run.accepted,
                    mtp.rows_seen(),
                    mtp.cache().pos
                );
                assert_eq!(
                    mtp.cache().pos,
                    mtp.rows_seen(),
                    "the head's lane holds rows the accept did not keep"
                );
                assert!(mtp.rows_seen() >= prompt.len() + want.len() - 1);
                if fold {
                    assert_eq!(
                        folded as usize,
                        run.accepted.len(),
                        "k = {k}: a round did not take its folded draft"
                    );
                    assert_eq!(fallbacks, 0);
                } else {
                    assert_eq!(folded, 0);
                }
                rounds.push((run.accepted, drafted));
            }
            assert_eq!(
                rounds[0].1, rounds[1].1,
                "plans {plans} k = {k}: a folded round drafted other tokens than the observe's"
            );
            assert_eq!(
                rounds[0], rounds[1],
                "plans {plans} k = {k}: the folded drafts are not the observe's"
            );
        }
    }
    lloom::models::qwen4exp::mtp::set_mtp_fold(false);
    Ok(())
}

/// Every expert stack of `cfg` to a GGUF, as the loader names them.
fn write_checkpoint(cfg: &Config, path: &std::path::Path) -> Result<()> {
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &Device::Cpu, GgmlDType::F32);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    Ok(())
}

fn streamed(
    cfg: &Config,
    path: &std::path::Path,
    slots: usize,
    dev: &Device,
) -> Result<(Qwen4Exp<RewindableExact, Streamed>, Content)> {
    let content = Content::read(&mut File::open(path)?)?;
    let src = |l: usize, n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
    let mut m = testing::model_with_indexer::<RewindableExact, _>(cfg, dev, |l| {
        Streamed::new(
            src(l, "ffn_gate_exps.weight")?,
            src(l, "ffn_up_exps.weight")?,
            src(l, "ffn_down_exps.weight")?,
            slots,
            LoaderArm::Pread,
            dev,
        )
    })?;
    let head: MtpHead<CrownExact, Streamed> = testing::mtp_head::<CrownExact, _>(
        cfg,
        dev,
        Streamed::new(
            src(0, "ffn_gate_exps.weight")?,
            src(0, "ffn_up_exps.weight")?,
            src(0, "ffn_down_exps.weight")?,
            slots,
            LoaderArm::Pread,
            dev,
        )?,
    );
    m.set_mtp(head);
    Ok((m, content))
}

/// A device-planned round's readback — the argmax and the layers' records
/// in one staged read — is counted in its verify readbacks: every round
/// reads back at least once, and the pool holding every expert the windows
/// run on the device plan.
#[test]
fn a_device_planned_round_counts_its_readback() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let dev = Device::new_metal(0)?;
    let path: PathBuf =
        std::env::temp_dir().join(format!("lloom-mtpvar-rb-{}.gguf", std::process::id()));
    write_checkpoint(&cfg, &path)?;
    let (m, _content) = streamed(&cfg, &path, cfg.n_experts, &dev)?;
    let max = 16;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 7, 10, max)?;
    let mut spec = Scripted::new(&want, 1, 3);
    let run = speculate(&m, &cfg, &dev, &mut spec, &prompt, &DecodeOptions::new(max))?;
    std::fs::remove_file(&path).ok();
    println!(
        "READBACKS window_device={} verify_readbacks={:?}",
        run.window_device, run.verify_readbacks
    );
    assert_eq!(run.ids, want);
    assert!(run.window_device > 0, "no window ran on the device plan");
    assert!(
        run.verify_readbacks.iter().all(|&r| r >= 1),
        "a round's readback was not counted: {:?}",
        run.verify_readbacks
    );
    Ok(())
}

/// The folded observe over a STREAMED target and head on a pool of a third
/// of the experts (the head's plan and settle, the target's misses and
/// redos): every round folds or falls back to the observe, the stream is
/// greedy's and the head's lane holds exactly the accepted rows.
#[test]
fn a_folded_observe_over_a_streamed_head_is_greedy() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let dev = Device::new_metal(0)?;
    let path: PathBuf =
        std::env::temp_dir().join(format!("lloom-mtpvar-fold-{}.gguf", std::process::id()));
    write_checkpoint(&cfg, &path)?;
    let (m, _content) = streamed(&cfg, &path, 2, &dev)?;
    let max = 16;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 7, 10, max)?;
    let head = m.mtp_head().expect("the head");
    lloom::models::qwen4exp::mtp::set_mtp_fold(true);
    let mut mtp = MtpDrafter::new(head, m.embed_tokens(), m.lm_head(), &cfg, &dev);
    let run = speculate(&m, &cfg, &dev, &mut mtp, &prompt, &DecodeOptions::new(max));
    lloom::models::qwen4exp::mtp::set_mtp_fold(false);
    std::fs::remove_file(&path).ok();
    let run = run?;
    let (folded, fallbacks) = mtp.fold_stats();
    println!(
        "MTPFOLDSTREAM rounds={} folded={folded} fallbacks={fallbacks} rows_seen={} pos={}",
        run.accepted.len(),
        mtp.rows_seen(),
        mtp.cache().pos
    );
    assert_eq!(run.ids, want, "not the greedy stream");
    assert_eq!(mtp.cache().pos, mtp.rows_seen());
    assert_eq!(
        (folded + fallbacks) as usize,
        run.accepted.len(),
        "a round neither folded nor fell back"
    );
    Ok(())
}

/// A verify window's routing on the host plan under the plan cache with f16
/// routed activations (the `fast` contract's `moe-act-f16`): the window's
/// ids are read on the host by the tier's plan after the pass has cast the
/// activations to f16, and a served plan that ended the ids' interval at
/// the top-k gave their memory to that cast — `lloom chat --mtp` under
/// `fast` stopped with "expert N out of 512" on `dev` `e8c7cf776`, the ids
/// read back as f16 bytes. Three requests served from the plans, every
/// stream and carry the pool path's.
#[test]
fn a_planned_verify_under_f16_routing_reads_its_routing() -> Result<()> {
    use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
    use lloom::models::qwen4exp::{LoadOptions, SlotBudget, StreamLoader, StreamOptions};
    use lloom::testing::fixture::Fixture;
    use lloom::testing::tiny::Weights;
    let _serial = SERIAL.lock();
    incumbent();
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_mtpvar_f16", Config::tiny())? else {
        return Ok(());
    };
    // Q8_0 experts: an f32 expert has no f16 activation arm.
    let fx = fx.with_weights(Weights {
        experts: GgmlDType::Q8_0,
        ..Weights::default()
    });
    if fx.at_scale() {
        return Ok(());
    }
    let cfg = fx.config(|c| {
        c.moe_intermediate_size = 32;
        c.eos_token_id = 127;
    });
    let stream = StreamOptions {
        slots: SlotBudget::Fixed(cfg.n_experts),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
        drafter_bytes: 0,
    };
    // Every window host-planned: its ids read back at each layer.
    let build = |dev: &Device| -> Result<Qwen4Exp<RewindableExact, Streamed>> {
        let mut m = fx.streamed(&cfg, dev, &LoadOptions::default(), &stream)?;
        m.set_routed_act(lloom_tensor::DType::F16);
        m.set_device_plan(false);
        Ok(m)
    };
    let max = 16;
    let dev = Device::new_metal(0)?;
    let m = build(&dev)?;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 7, 8, max)?;
    let mut pool = Scripted::new(&want, 1, 2);
    let pool_run = speculate(&m, &cfg, &dev, &mut pool, &prompt, &DecodeOptions::new(max))?;
    assert_eq!(pool_run.ids, want, "the pool path's speculative stream");
    drop(m);
    let dev = Device::new_metal(0)?;
    metal(&dev).enable_pass_plans(ArenaLayout::Packed);
    let m = build(&dev)?;
    for request in 0..3 {
        let mut spec = Scripted::new(&want, 1, 2);
        let run = speculate(&m, &cfg, &dev, &mut spec, &prompt, &DecodeOptions::new(max))?;
        assert_eq!(run.ids, want, "request {request}: the stream moved");
        assert_eq!(
            spec.carries, pool.carries,
            "request {request}: a carry moved"
        );
    }
    let c = metal(&dev).pass_plan_counters().expect("on");
    println!("F16VERIFY {c:?}");
    assert!(c.served > 0, "no pass was served: {c:?}");
    Ok(())
}

/// A prefill in units seeds the MTP lane unit by unit: the stream is
/// greedy's and the lane holds every prompt row, as the one-unit prefill's.
#[test]
fn a_prefill_in_units_seeds_the_mtp_lane() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let dev = Device::new_metal(0)?;
    let m = model(&cfg, &dev)?;
    let head = m.mtp_head().expect("the head");
    let max = 12;
    let (prompt, want) = prompt_with_stream(&m, &cfg, &dev, 9, 6, max)?;
    let mut keyed = Vec::new();
    for ubatch in [0usize, 4] {
        let mut mtp = MtpDrafter::new(head, m.embed_tokens(), m.lm_head(), &cfg, &dev);
        let opts = DecodeOptions {
            prefill_ubatch: ubatch,
            ..DecodeOptions::new(max)
        };
        let run = speculate(&m, &cfg, &dev, &mut mtp, &prompt, &opts)?;
        let seq_keyed = mtp.seq_keyed(SeqId(0)).to_vec();
        println!(
            "UBATCH ubatch={ubatch} rows_seen={} pos={} rounds={} keyed={}",
            mtp.rows_seen(),
            mtp.cache().pos,
            run.accepted.len(),
            seq_keyed.len()
        );
        assert_eq!(run.ids, want, "ubatch = {ubatch}: not the greedy stream");
        assert_eq!(mtp.cache().pos, mtp.rows_seen());
        assert!(mtp.rows_seen() >= prompt.len() + run.ids.len() - 1);
        // Row `j` reads the embedding of `keyed[j + 1]`, so the prompt's
        // own rows key `prompt ++ stream` whatever the unit count: a unit
        // keyed at the wrong offset pairs its rows with other tokens, which
        // the greedy stream (the verify corrects every draft) and the row
        // count both survive.
        let want_keyed: Vec<u32> = prompt.iter().chain(&run.ids).copied().collect();
        assert_eq!(
            seq_keyed,
            want_keyed[..seq_keyed.len().min(want_keyed.len())],
            "ubatch = {ubatch}: the lane's rows were keyed from other tokens"
        );
        keyed.push(seq_keyed);
    }
    assert_eq!(
        keyed[0], keyed[1],
        "the prefill's unit count changed what the lane was keyed from"
    );
    Ok(())
}

/// The greedy decode across the indexer's dense range under the plan cache
/// is the pool path's, request after request. A step past the range runs
/// the selection (its score, sort and cells) that a step inside it does
/// not, so the two are different sequences under one decode kind; served a
/// plan recorded inside the range, the first step past it read slots packed
/// for other values (the stream left the pool path's at its first step past
/// the range, position 20 here, on `dev` `e8c7cf776`).
#[test]
fn the_greedy_decode_past_the_dense_range_under_plans_is_the_pool_paths() -> Result<()> {
    let _serial = SERIAL.lock();
    incumbent();
    let cfg = config();
    let max = 20;
    let mut crossed = 0;
    for seed in 0..4u32 {
        let prompt: Vec<u32> = (0..9).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let want = {
            let dev = Device::new_metal(0)?;
            greedy(&model(&cfg, &dev)?, &cfg, &dev, &prompt, max)?
        };
        // The dense range is 19 positions at the tiny budget.
        crossed += usize::from(prompt.len() + want.len() > 20);
        let dev = Device::new_metal(0)?;
        metal(&dev).enable_pass_plans(ArenaLayout::Packed);
        let m = model(&cfg, &dev)?;
        for request in 0..3 {
            let got = greedy(&m, &cfg, &dev, &prompt, max)?;
            assert_eq!(
                got, want,
                "seed {seed} request {request}: a served decode step moved the stream"
            );
        }
        let c = metal(&dev).pass_plan_counters().expect("on");
        println!("PASTRANGE seed={seed} len={} {c:?}", want.len());
    }
    assert!(
        crossed >= 2,
        "only {crossed} streams crossed the dense range"
    );
    Ok(())
}
