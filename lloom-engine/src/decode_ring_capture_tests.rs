//! Correctness-only snapshots at the existing ring sampling seam.
//! All storage is prepared outside passes; no recorder bytes enter the reader.
use super::*;
use lloom::generation::{ReadbackArms, ReadbackCounters, RingReader};
use lloom_tensor::{DType, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Resolve fixture input through the same profile parser as production tuning.
/// This prepares values only; no OnceLock or device is initialized here.
pub(crate) fn fixture_contract(profile: Option<&str>) -> Result<lloom_tensor::tuning::Contract> {
    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning
        .set("contract.profile", profile.unwrap_or("deterministic"))
        .map_err(lloom_tensor::Error::msg)?;
    lloom_tensor::tuning::Contract::parse(&tuning.contract.profile)
        .map_err(lloom_tensor::Error::msg)
}

#[derive(Clone)]
pub(crate) struct Capture(Rc<RefCell<Rows>>);
struct Rows {
    bank: Tensor,
    scores: Option<Tensor>,
    shift_first_key: bool,
    positions: Vec<usize>,
    capacity: usize,
    vocab: usize,
    closed: bool,
}
pub(crate) struct Completed {
    pub(crate) rows: Vec<Vec<f32>>,
    pub(crate) positions: Vec<usize>,
}
impl Completed {
    pub(crate) fn check_matrix(&self, first: usize, count: usize, vocab: usize) -> Result<()> {
        let end = first
            .checked_add(count)
            .ok_or_else(|| lloom_tensor::Error::msg("ring capture: position overflow"))?;
        if self.rows.len() != count
            || self.positions != (first..end).collect::<Vec<_>>()
            || self.rows.iter().any(|row| row.len() != vocab)
        {
            lloom_tensor::bail!(
                "ring capture: missing, duplicate, shifted or malformed row receipt"
            );
        }
        Ok(())
    }
}

impl Capture {
    pub(crate) fn new(
        device: &Device,
        first: usize,
        capacity: usize,
        vocab: usize,
    ) -> Result<Self> {
        let bytes = capacity.checked_mul(vocab).and_then(|n| n.checked_mul(4));
        if capacity == 0
            || vocab == 0
            || bytes.is_none_or(|n| n > isize::MAX as usize)
            || first.checked_add(capacity).is_none()
        {
            lloom_tensor::bail!("ring capture: invalid bounded row extent");
        }
        Ok(Self(Rc::new(RefCell::new(Rows {
            bank: Tensor::zeros((capacity, vocab), DType::F32, device)?,
            scores: None,
            shift_first_key: false,
            positions: Vec::with_capacity(capacity),
            capacity,
            vocab,
            closed: false,
        }))))
    }
    /// Optional valid score fixture, uploaded cold. Shipping sampling is untouched.
    #[cfg(feature = "metal")]
    pub(crate) fn with_scores(self, scores: Tensor) -> Result<Self> {
        {
            let mut rows = self.0.borrow_mut();
            if scores.dims() != [rows.capacity, rows.vocab]
                || scores.dtype() != DType::F32
                || !scores.device().same_device(rows.bank.device())
                || !scores.is_contiguous()
            {
                lloom_tensor::bail!("ring capture: planted row geometry/device");
            }
            rows.scores = Some(scores);
        }
        Ok(self)
    }
    #[cfg(feature = "metal")]
    pub(crate) fn shifted_first_key_control(self) -> Self {
        self.0.borrow_mut().shift_first_key = true;
        self
    }
    fn shift_first_key(&self) -> bool {
        let rows = self.0.borrow();
        rows.shift_first_key && rows.positions.is_empty()
    }
    pub(crate) fn observed_positions(&self) -> Vec<usize> {
        self.0.borrow().positions.clone()
    }
    pub(crate) fn record(&self, logits: &Tensor, position: Option<u64>) -> Result<Tensor> {
        let position = position.ok_or_else(|| {
            lloom_tensor::Error::msg("ring capture: sampler has no actual position")
        })?;
        let position = usize::try_from(position).map_err(|_| {
            lloom_tensor::Error::msg("ring capture: sampler position exceeds usize")
        })?;
        let mut rows = self.0.borrow_mut();
        let ordinal = rows.positions.len();
        if rows.closed
            || ordinal >= rows.capacity
            || logits.elem_count() != rows.vocab
            || logits.dtype() != DType::F32
            || !logits.is_contiguous()
            || !logits.device().same_device(rows.bank.device())
        {
            lloom_tensor::bail!("ring capture: unexpected sample or closed/full recorder");
        }
        let selected = match &rows.scores {
            Some(scores) => scores.narrow(0, ordinal, 1)?.reshape(logits.shape())?,
            None => logits.clone(),
        };
        // Session::hold_logits's owned-destination primitive, with a distinct
        // never-rewritten row for every draw, not a shallow Tensor::copy.
        rows.bank
            .slice_set(&selected.reshape((1, rows.vocab))?, 0, ordinal)?;
        rows.positions.push(position);
        Ok(selected)
    }
    /// Caller has returned/joined the producer. Complete copies before any
    /// assertion, CPU replay, or release of the independently held recorder.
    pub(crate) fn finish(&self) -> Result<Completed> {
        let mut rows = self.0.borrow_mut();
        rows.closed = true;
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = rows.bank.device() {
            let marker = device.flush_marker()?;
            device.wait_marker_prefix_data(&marker)?;
        } else {
            rows.bank.device().synchronize()?;
        }
        #[cfg(not(feature = "metal"))]
        rows.bank.device().synchronize()?;
        println!("RING_CAPTURE_TERMINAL rows={}", rows.positions.len());
        Ok(Completed {
            rows: rows
                .bank
                .narrow(0, 0, rows.positions.len())?
                .to_vec2::<f32>()?,
            positions: rows.positions.clone(),
        })
    }
}

pub(super) struct Recording<'a, R> {
    pub(super) inner: &'a mut R,
    pub(super) capture: &'a Capture,
}
impl<R: ReadbackStrategy> ReadbackStrategy for Recording<'_, R> {
    const PASS_INPUT: lloom_tensor::backend::PassInput = R::PASS_INPUT;

    fn device(&self) -> &Device {
        self.inner.device()
    }
    fn arms(&self) -> ReadbackArms {
        self.inner.arms()
    }
    fn admits_lookahead(&self) -> bool {
        self.inner.admits_lookahead()
    }
    fn loop_commits(&self) -> bool {
        self.inner.loop_commits()
    }
    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        self.inner.check_sampler(sampling)
    }
    fn check_request(&self, sampler: &LogitsProcessor) -> Result<()> {
        self.inner.check_request(sampler)
    }
    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        // Perturb the real key only in the explicit causal negative fixture.
        // Metadata observes the accessor AFTER the change, never manufactures
        // first+ordinal, and the CPU oracle keeps its independent expected key.
        if self.capture.shift_first_key() {
            let position = sampler
                .position()
                .and_then(|p| p.checked_add(1))
                .ok_or_else(|| {
                    lloom_tensor::Error::msg(
                        "ring capture: shifted key requires a bounded actual key",
                    )
                })?;
            sampler.at_position(Some(position));
        }
        let row = self.capture.record(logits, sampler.position())?;
        self.inner.sample_and_feed(sampler, &row)
    }
    fn forward_input<M: CausalLM>(
        &mut self,
        model: &M,
        input: &Tensor,
        pos: usize,
        cache: &mut M::Cache,
    ) -> Result<Tensor> {
        self.inner.forward_input(model, input, pos, cache)
    }
    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        self.inner.set_thinking_budget(budget)
    }
    fn stage_prompt_chunk(&mut self, tokens: &[u32]) -> Result<Tensor> {
        self.inner.stage_prompt_chunk(tokens)
    }
    fn publish_marker(&mut self, force_live: bool) -> Result<()> {
        self.inner.publish_marker(force_live)
    }
    fn close_window(&mut self, ramp: bool) -> Result<()> {
        self.inner.close_window(ramp)
    }
    fn fed_host_ids(&self) -> Option<&[u32]> {
        self.inner.fed_host_ids()
    }
    fn retracts_draw(&self) -> bool {
        self.inner.retracts_draw()
    }
    fn retract_draw(&mut self) -> Result<()> {
        self.inner.retract_draw()
    }
    fn eos_of_last(&mut self) -> Result<bool> {
        self.inner.eos_of_last()
    }
    fn take_ids(&mut self) -> Result<Vec<u32>> {
        self.inner.take_ids()
    }
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.inner.take_ids_into(out)
    }
    fn drain_all(&mut self) -> Result<Vec<u32>> {
        self.inner.drain_all()
    }
    fn eos_in_drained(&self, batch: &[u32]) -> Option<usize> {
        self.inner.eos_in_drained(batch)
    }
    fn note_eos_truncation(&mut self, dropped: usize) {
        self.inner.note_eos_truncation(dropped)
    }
    fn begin_run(&mut self) {
        self.inner.begin_run()
    }
    fn eos_read_arm(&self) -> &'static str {
        self.inner.eos_read_arm()
    }
    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        self.inner.eos_ids_tensor()
    }
    fn reader(&self) -> Option<RingReader> {
        self.inner.reader()
    }
    fn published_count(&self) -> Option<usize> {
        self.inner.published_count()
    }
    fn terminal_observers(&self) -> lloom::nn::terminal_trace::Observers {
        self.inner.terminal_observers()
    }
    fn stamps_each_token(&self, adaptive: bool) -> bool {
        self.inner.stamps_each_token(adaptive)
    }
    fn arm_publish_event(
        &mut self,
        for_reader: bool,
        eos_park: bool,
        adaptive: bool,
    ) -> Result<()> {
        self.inner.arm_publish_event(for_reader, eos_park, adaptive)
    }
    fn set_fused_publish(&mut self, on: bool) {
        self.inner.set_fused_publish(on)
    }
    fn publishes_fused(&self) -> bool {
        self.inner.publishes_fused()
    }
    fn counters(&self) -> ReadbackCounters {
        self.inner.counters()
    }
}

#[test]
fn snapshot_rows_survive_source_overwrite_and_refuse_overflow() -> Result<()> {
    let capture = Capture::new(&Device::Cpu, 7, 2, 3)?;
    let source = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    capture.record(&source, Some(7))?;
    source.slice_set(&Tensor::new(&[4f32, 5., 6.], &Device::Cpu)?, 0, 0)?;
    capture.record(&source, Some(8))?;
    let missing = capture.record(&source, None);
    let overflow = capture.record(&source, Some(9));
    let done = capture.finish()?;
    assert!(missing.is_err());
    assert!(overflow.is_err());
    assert_eq!(done.positions, [7, 8]);
    assert_eq!(done.rows, [vec![1., 2., 3.], vec![4., 5., 6.]]);
    assert!(capture.record(&source, Some(9)).is_err());
    assert_eq!(capture.observed_positions(), [7, 8]);
    Ok(())
}

#[test]
fn row_matrix_refuses_missing_duplicate_shifted_or_wrong_width_receipts() -> Result<()> {
    let valid = Completed {
        rows: vec![vec![0.; 3]; 2],
        positions: vec![7, 8],
    };
    valid.check_matrix(7, 2, 3)?;
    for (rows, positions) in [
        (vec![vec![0.; 3]], vec![7]),
        (vec![vec![0.; 3]; 2], vec![7, 7]),
        (vec![vec![0.; 3]; 2], vec![8, 9]),
        (vec![vec![0.; 2]; 2], vec![7, 8]),
    ] {
        assert!(Completed { rows, positions }.check_matrix(7, 2, 3).is_err());
    }
    Ok(())
}

#[test]
fn sampler_seam_records_observed_key_instead_of_expected_ordinal() -> Result<()> {
    let row = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    for (observed, should_match) in [(7u64, true), (8u64, false)] {
        let capture = Capture::new(&Device::Cpu, 7, 1, 3)?;
        let mut blocking = Blocking::new(&[99], &Device::Cpu)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        sampler.at_position(Some(observed));
        let sampled = Recording {
            inner: &mut blocking,
            capture: &capture,
        }
        .sample_and_feed(&mut sampler, &row);
        let completed = capture.finish()?;
        sampled?;
        assert_eq!(completed.positions, [observed as usize]);
        assert_eq!(completed.check_matrix(7, 1, 3).is_ok(), should_match);
    }
    Ok(())
}

#[test]
fn legal_request_fixture_validates_and_replays_forced_two_then_one() -> Result<()> {
    for gumbel in [false, true] {
        for forced in [false, true] {
            let request = DecodeParams {
                seed: 781,
                temperature: Some(0.001),
                gumbel,
                processors: Processors {
                    frequency_penalty: 0.5,
                    ..Default::default()
                },
                thinking: forced
                    .then(|| ThinkingBudget::new(0, vec![2], vec![2]))
                    .transpose()?,
                max_tokens: 2,
                eos_ids: vec![99],
                ..Default::default()
            };
            // Validate through the same public request entry before evaluating
            // the reference, not only arithmetic over a forbidden parameter.
            let mut sampler = request.sampler()?;
            assert!(matches!(
                sampler.sampling(),
                Sampling::All { .. } | Sampling::GumbelSoftmax { .. }
            ));
            sampler.observe_prompt(&[0; 7]);
            let mut blocking = Blocking::new(&request.eos_ids, &Device::Cpu)?;
            blocking.set_thinking_budget(request.thinking.as_ref())?;
            let mut ids = Vec::new();
            for (ordinal, row) in [[-10_000f32, 0., -10_000.], [-10_000., 0., 0.25]]
                .into_iter()
                .enumerate()
            {
                sampler.at_position(Some(7 + ordinal as u64));
                let id = blocking
                    .sample_and_feed(&mut sampler, &Tensor::new(&row, &Device::Cpu)?)?
                    .to_vec2::<u32>()?[0][0];
                ids.push(id);
            }
            assert_eq!(
                ids,
                if forced { vec![2, 1] } else { vec![1, 2] },
                "gumbel={gumbel} forced={forced}"
            );
            assert_eq!(blocking.drain_all()?, ids);
        }
    }
    Ok(())
}

#[test]
fn fixture_contract_preserves_default_and_validates_explicit_fast_before_install() -> Result<()> {
    use lloom_tensor::tuning::Contract;
    assert_eq!(fixture_contract(None)?, Contract::Deterministic);
    assert_eq!(
        fixture_contract(Some("deterministic"))?,
        Contract::Deterministic
    );
    assert_eq!(fixture_contract(Some("fast"))?, Contract::Fast);
    assert_eq!(fixture_contract(Some("Fast"))?, Contract::Fast);
    for invalid in ["", "relaxed", "metla4", "fast,deterministic"] {
        assert!(fixture_contract(Some(invalid)).is_err(), "{invalid:?}");
    }
    Ok(())
}

#[test]
fn recording_preserves_token_input_contract_without_a_device() {
    use lloom_tensor::backend::PassInput;
    assert_eq!(<Recording<'static, Blocking> as ReadbackStrategy>::PASS_INPUT, PassInput::TokenIds);
    assert_eq!(<Recording<'static, lloom::generation::Ring> as ReadbackStrategy>::PASS_INPUT, PassInput::TokenIds);
}

#[cfg(feature = "metal")]
#[test]
fn recording_preserves_prepared_input_but_native_blocking_stays_tokens() {
    use lloom_tensor::backend::PassInput;
    type Model = lloom::models::lfm2::Model<lloom::profile::CrownExact>;
    type Prepared = lloom::generation::readback::NativeRing<'static, Model>;
    assert_eq!(<Prepared as ReadbackStrategy>::PASS_INPUT, PassInput::PreparedEmbeddings);
    assert_eq!(<Recording<'static, Prepared> as ReadbackStrategy>::PASS_INPUT, PassInput::PreparedEmbeddings);
    assert_eq!(<lloom::generation::readback::NativeBlocking as ReadbackStrategy>::PASS_INPUT, PassInput::TokenIds);
    assert_eq!(<Recording<'static, lloom::generation::readback::NativeBlocking> as ReadbackStrategy>::PASS_INPUT, PassInput::TokenIds);
}
