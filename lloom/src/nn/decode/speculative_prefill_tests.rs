//! CPU protocol controls over the actual serial, static and pooled join entries.
//! The cap of one isolates seed execution: no verify or checkpoint load.
//! Pacer hooks record calls/depth, not native markers or GPU retirement.
use super::*;
use crate as subject;
use crate::nn::speculator::LayerTaps;
use crate::nn::{BatchedCausalLM, Model, Row, RowDriver};
use crate::profile::{RewindProof, RewindableExact, Shaped};
use std::cell::{Cell, RefCell};

#[path = "../../../tests/support/chain.rs"]
mod chain_fixture;
use chain_fixture::RecordingChain;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Event {
    Depth(usize),
    Before(usize),
    After(usize),
    Begin(usize, Vec<u32>, Vec<u32>),
    Forward(usize, usize),
    Probe(usize),
    Observe(usize, Vec<(usize, u32)>, Vec<u32>),
}
thread_local! {
    static TRACE: RefCell<Option<Vec<Event>>> = const { RefCell::new(None) };
}
pub(super) fn note(event: Event) {
    TRACE.with(|t| {
        if let Some(events) = t.borrow_mut().as_mut() {
            events.push(event);
        }
    });
}
struct Trace;
impl Trace {
    fn start() -> Self {
        TRACE.with(|t| assert!(t.replace(Some(Vec::new())).is_none()));
        Self
    }
    fn read(&self) -> Vec<Event> {
        TRACE.with(|t| t.borrow().as_ref().unwrap().clone())
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        TRACE.with(|t| {
            t.replace(None);
        });
    }
}

// Observability wrapper around the existing fake, with no second logits math.
struct Target {
    inner: RecordingChain,
    pending: RefCell<Option<Vec<(usize, u32)>>>,
    drops: Cell<u64>,
}
impl Target {
    fn cpu() -> Result<Self> {
        Ok(Self {
            inner: RecordingChain::cpu()?,
            pending: RefCell::new(None),
            drops: Cell::new(0),
        })
    }
}
impl Model for Target {
    type Config = ();
    fn from_weights(_: crate::nn::VarBuilder, _: &()) -> Result<Self> {
        Self::cpu()
    }
}
impl Shaped for Target {
    type Profile = RewindableExact;
}
impl CausalLM for Target {
    type Cache = usize;
    fn new_cache(&self, c: &(), d: &Device) -> Result<usize> {
        self.inner.new_cache(c, d)
    }
    fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.embed(x)
    }
    fn forward_embeds(&self, x: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        note(Event::Forward(pos, x.dim(1)?));
        let ids = x.flatten_all()?.to_vec1::<u32>()?;
        let probe = ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| (pos + i, id))
            .collect();
        if self.pending.replace(Some(probe)).is_some() {
            self.drops.set(self.drops.get() + 1);
        }
        self.inner.forward_embeds(x, pos, cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}
impl BatchedCausalLM for Target {
    fn forward_rows(&self, _: &Tensor, _: &mut [Row<'_, usize>]) -> Result<Tensor> {
        lloom_tensor::bail!("seed-only control reached a batched decode step")
    }
}
impl Probes<LayerTaps<5>> for Target {
    type Pending = ();
    type PendingRows = ();
    fn verify_and_probe(
        &self,
        _: &[Draft],
        _: &[Anchor],
        _: &mut usize,
        _: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Vec<Tensor>)> {
        lloom_tensor::bail!("seed-only control reached verify")
    }
    fn verify_enqueue(&self, _: &Draft, _: &Anchor, _: &mut usize) -> Result<()> {
        lloom_tensor::bail!("seed-only control enqueued verify")
    }
    fn verify_complete(&self, _: (), _: &mut usize, _: &[u32]) -> Result<(Vec<u32>, Vec<Tensor>)> {
        lloom_tensor::bail!("seed-only control completed verify")
    }
    fn take_probe(&self, cache: &mut usize) -> Result<Vec<Tensor>> {
        note(Event::Probe(*cache));
        match self.pending.take() {
            None => Ok(Vec::new()),
            Some(rows) => {
                let words: Vec<u32> = rows.iter().flat_map(|&(p, id)| [p as u32, id]).collect();
                let t = Tensor::from_vec(words, (1, rows.len(), 2), &Device::Cpu)?;
                Ok(vec![t; 5])
            }
        }
    }
    fn probe_drops(&self, _: &usize) -> u64 {
        self.drops.get()
    }
}

// The existing identity proposer supplies begin; the recorder carries typed
// position/token probes like the existing RecordingDrafter control. A positive
// ceiling lets all three callers run; any draft is a failed seed-only control.
struct SeedRecorder {
    identity: IdentityProposer,
    fail_observe: bool,
}
impl SeedRecorder {
    fn new() -> Self {
        Self {
            identity: IdentityProposer,
            fail_observe: false,
        }
    }
}
impl Speculator for SeedRecorder {
    type Probe = LayerTaps<5>;
    type Family = crate::nn::speculator::DSpark;
    fn n_seq(&self) -> usize {
        2
    }
    fn max_draft(&self) -> usize {
        1
    }
    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        self.identity.begin(seq, prompt)
    }
    fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
        note(Event::Begin(seq.0, prompt.to_vec(), kept.to_vec()));
        self.identity.begin_at(seq, prompt, kept)
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        lloom_tensor::bail!("seed-only control reached draft")
    }
    fn observe(&mut self, seq: SeqId, probe: &Vec<Tensor>, accepted: &[u32]) -> Result<()> {
        assert_eq!(probe.len(), 5);
        let words = probe[0].flatten_all()?.to_vec1::<u32>()?;
        for t in probe {
            assert_eq!(t.flatten_all()?.to_vec1::<u32>()?, words);
        }
        let rows = words
            .chunks_exact(2)
            .map(|w| (w[0] as usize, w[1]))
            .collect();
        note(Event::Observe(seq.0, rows, accepted.to_vec()));
        if self.fail_observe {
            lloom_tensor::bail!("injected observe failure");
        }
        Ok(())
    }
}

struct Case {
    options: DecodeOptions,
    kept: Vec<u32>,
    depth: usize,
    // Independent literal oracle; no call to prefill_widths/tile_widths here.
    spans: Vec<(usize, usize)>,
}
fn cases() -> Vec<Case> {
    let opts = |spark, tile, schedule| DecodeOptions {
        prefill_ubatch: spark,
        prefill_tile: tile,
        prefill_schedule: schedule,
        prefill_bulk: 16,
        in_flight_depth: 3,
        prefill_pacer_depth: 5,
        ..DecodeOptions::new(1)
    };
    vec![
        Case {
            options: opts(0, 16, None),
            depth: 5,
            kept: vec![],
            spans: vec![(0, 16), (16, 16), (32, 5)],
        },
        Case {
            options: opts(3, 16, None),
            depth: 5,
            kept: vec![],
            spans: vec![(0, 3), (3, 16), (19, 16), (35, 2)],
        },
        Case {
            options: opts(3, 0, None),
            depth: 5,
            kept: vec![],
            spans: vec![(0, 3), (3, 34)],
        },
        Case {
            options: opts(0, 0, None),
            depth: 5,
            kept: vec![],
            spans: vec![(0, 37)],
        },
        Case {
            options: opts(0, 16, Some(vec![0, 3])),
            depth: 5,
            kept: vec![],
            spans: vec![(0, 3), (3, 16), (19, 16), (35, 1), (36, 1)],
        },
        Case {
            options: opts(0, 16, None),
            depth: 5,
            kept: vec![8; 7],
            spans: vec![(7, 16), (23, 16), (39, 5)],
        },
        Case {
            options: DecodeOptions {
                prefill_pacer_depth: 0,
                ..opts(0, 16, None)
            },
            depth: 3,
            kept: vec![],
            spans: vec![(0, 16), (16, 16), (32, 5)],
        },
        Case {
            options: DecodeOptions {
                prefill_pacer_depth: 1,
                ..opts(0, 16, None)
            },
            depth: 2,
            kept: vec![],
            spans: vec![(0, 16), (16, 16), (32, 5)],
        },
    ]
}
fn session<'m>(model: &'m Target, kept: &[u32]) -> Result<Session<'m, Target>> {
    let mut s = Session::new(model, 0);
    if !kept.is_empty() {
        let mut rb = crate::generation::Blocking::new(&[63], &Device::Cpu)?;
        s.prefill(&mut rb, kept, &DecodeOptions::new(1))?;
    }
    model.inner.forwards.borrow_mut().clear();
    Ok(s)
}
fn seed_events(case: &Case, seq: usize, prompt: &[u32]) -> Vec<Event> {
    let mut events = vec![
        Event::Depth(case.depth),
        Event::Begin(seq, prompt.to_vec(), case.kept.clone()),
        Event::Probe(case.kept.len()),
    ];
    for (i, &(pos, w)) in case.spans.iter().enumerate() {
        events.extend([
            Event::Before(case.depth),
            Event::Forward(pos, w),
            Event::Probe(pos + w),
        ]);
        if i + 1 < case.spans.len() {
            events.push(Event::Observe(
                seq,
                (pos..pos + w)
                    .zip(
                        prompt[pos - case.kept.len()..pos - case.kept.len() + w]
                            .iter()
                            .copied(),
                    )
                    .collect(),
                vec![],
            ));
        }
        events.push(Event::After(case.depth));
    }
    events
}
fn final_observe(case: &Case, seq: usize, prompt: &[u32], anchor: u32) -> Event {
    let &(pos, w) = case.spans.last().unwrap();
    Event::Observe(
        seq,
        (pos..pos + w)
            .zip(
                prompt[pos - case.kept.len()..pos - case.kept.len() + w]
                    .iter()
                    .copied(),
            )
            .collect(),
        vec![anchor],
    )
}
fn check(out: &DecodeOut, s: &Session<'_, Target>, case: &Case, prompt: &[u32], anchor: u32) {
    assert_eq!(out.ids, [anchor]);
    assert_eq!(out.stats.prefill_units, case.spans.len());
    assert_eq!(out.stats.overrun, 0);
    assert_eq!(s.pos(), case.kept.len() + 37);
    assert_eq!(*s.sequence().cache(), s.pos());
    assert_eq!(s.tokens(), [case.kept.as_slice(), prompt].concat());
    assert!(
        s.last_logits().is_none(),
        "a seed must not leave stale resumable logits"
    );
    assert_eq!(
        out.stats.step_markers, 0,
        "CPU hooks do not pretend to be native markers"
    );
}

#[test]
fn actual_speculative_seed_serial_cpu() -> Result<()> {
    for case in cases() {
        let model = Target::cpu()?;
        let mut s = session(&model, &case.kept)?;
        let prompt = vec![0; 37];
        let mut spec = SeedRecorder::new();
        let mut rb = crate::generation::Blocking::new(&[63], &Device::Cpu)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let trace = Trace::start();
        let out = s.decode_speculative(
            RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &case.options,
            &[63],
            &mut |_| Ok(()),
        )?;
        check(&out, &s, &case, &prompt, 3);
        assert_eq!(*model.inner.forwards.borrow(), case.spans);
        let mut want = vec![Event::Depth(3)];
        want.extend(seed_events(&case, 0, &prompt));
        want.push(final_observe(&case, 0, &prompt, 3));
        assert_eq!(trace.read(), want);
        eprintln!(
            "ACTUAL serial spans={:?} protocol={:?}",
            case.spans,
            trace.read()
        );
    }
    Ok(())
}

#[test]
fn actual_speculative_seed_static_cpu() -> Result<()> {
    for case in cases() {
        let model = Target::cpu()?;
        let mut sessions = vec![session(&model, &case.kept)?, session(&model, &case.kept)?];
        let prompt = vec![0; 37];
        let mut other = prompt.clone();
        other[36] = 1;
        let mut spec = SeedRecorder::new();
        let trace = Trace::start();
        let outs = Session::decode_speculative_pipelined(
            &mut sessions,
            RewindProof::of(),
            &mut spec,
            &[&prompt, &other],
            &case.options,
            &[63],
            &Device::Cpu,
            &mut |_, _| Ok(()),
        )?;
        check(&outs[0], &sessions[0], &case, &prompt, 3);
        assert_eq!(outs[1].ids, [10], "each lane retains its own final logits");
        assert_eq!(outs[1].stats.prefill_units, case.spans.len());
        let mut want = vec![Event::Depth(3)];
        want.extend(seed_events(&case, 0, &prompt));
        want.extend(seed_events(&case, 1, &other));
        // Final probes are observed only after all lane forwards, each with
        // its own anchor, not the following lane's or an earlier unit's.
        want.extend([
            final_observe(&case, 0, &prompt, 3),
            final_observe(&case, 1, &other, 10),
        ]);
        assert_eq!(trace.read(), want);
        assert_eq!(
            *model.inner.forwards.borrow(),
            [case.spans.as_slice(), case.spans.as_slice()].concat()
        );
        eprintln!(
            "ACTUAL static spans={:?} protocol={:?}",
            model.inner.forwards.borrow(),
            trace.read()
        );
    }
    Ok(())
}

#[test]
fn actual_speculative_seed_pooled_join_cpu() -> Result<()> {
    for case in cases() {
        let model = Target::cpu()?;
        let prompt = vec![0; 37];
        let trace = Trace::start();
        let mut driver = crate::nn::batched_spec::SpecBatchDecoder::new(
            RewindProof::of(),
            SeedRecorder::new(),
            case.options.clone(),
            Device::Cpu,
        )?;
        // Two successive occupants of the same row exercise slot reuse and
        // the same pacer queue; this is the real admit/step/take_finished path.
        let mut want = vec![Event::Depth(3)];
        for occupant in 0..2 {
            let mut prompt = prompt.clone();
            prompt[36] = occupant;
            let anchor = if occupant == 0 { 3 } else { 10 };
            // Prefix preparation is outside the seed trace.
            let s = {
                let saved = TRACE.with(|t| t.replace(None));
                let s = session(&model, &case.kept)?;
                TRACE.with(|t| {
                    t.replace(saved);
                });
                s
            };
            let id = driver.admit_with_eos(s, prompt.clone(), Sampling::ArgMax, 0, &[63], 1)?;
            let report = driver.step(&mut |_, _| Ok(()))?;
            assert_eq!(report.retired, 1);
            let mut finished = driver.take_finished();
            assert_eq!(finished.len(), 1);
            let (done_id, s, out) = finished.pop().unwrap();
            assert_eq!(done_id, id);
            check(&out, &s, &case, &prompt, anchor);
            want.extend(seed_events(&case, 0, &prompt));
            want.extend([Event::Depth(3), final_observe(&case, 0, &prompt, anchor)]);
            assert_eq!(trace.read(), want);
            assert_eq!(*model.inner.forwards.borrow(), case.spans);
            assert!(driver.is_idle());
        }
        eprintln!("ACTUAL pooled protocol={:?}", trace.read());
    }
    Ok(())
}

#[test]
fn actual_speculative_seed_rejects_mismatched_work_before_begin_cpu() -> Result<()> {
    let model = Target::cpu()?;
    let mut s = session(&model, &[])?;
    let opts = DecodeOptions::new(1);
    let work = PrefillWork::resolve(&opts, 1, 3, true)?;
    let mut pacer = StepPacer::for_decode(&opts);
    let mut spec = SeedRecorder::new();
    let trace = Trace::start();
    assert!(s
        .prefill_streamed(
            &mut spec,
            SeqId(0),
            &[0; 3],
            &work,
            &mut pacer,
            &Device::Cpu
        )
        .is_err());
    assert!(trace.read().is_empty());
    assert_eq!(s.pos(), 0);
    assert!(PrefillWork::resolve(&opts, usize::MAX, 1, true).is_err());
    Ok(())
}

#[test]
fn actual_speculative_seed_observe_error_stops_before_next_forward_cpu() -> Result<()> {
    let model = Target::cpu()?;
    let mut s = session(&model, &[])?;
    let case = cases().remove(0);
    let mut spec = SeedRecorder::new();
    spec.fail_observe = true;
    let work = PrefillWork::resolve(&case.options, 0, 37, true)?;
    let mut pacer = StepPacer::for_decode(&case.options);
    let trace = Trace::start();
    let err = s
        .prefill_streamed(
            &mut spec,
            SeqId(0),
            &[0; 37],
            &work,
            &mut pacer,
            &Device::Cpu,
        )
        .unwrap_err();
    assert!(err.to_string().contains("injected observe failure"));
    assert_eq!(
        trace.read(),
        vec![
            Event::Depth(5),
            Event::Begin(0, vec![0; 37], vec![]),
            Event::Probe(0),
            Event::Before(5),
            Event::Forward(0, 16),
            Event::Probe(16),
            Event::Observe(0, (0..16).map(|p| (p, 0)).collect(), vec![])
        ]
    );
    assert_eq!(*model.inner.forwards.borrow(), [(0, 16)]);
    Ok(())
}
