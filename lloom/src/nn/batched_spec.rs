//! The speculative batched loop as a scheduler: every live row's draft
//! window verified in ONE target pass per round (`Model::verify_rows_enqueue`
//! through [`Probes`]), rows joining between rounds through their own
//! streamed prefill (which seeds their drafter slot, `speculation.md`
//! §10.2w), leaving at EOS or at their cap — [`BatchDecoder`]'s surface
//! ([`RowDriver`]) over speculative rounds
//! (`design/docs/batched-decode-design.md` §12.4).
//!
//! A round: `spec.draft` for every live row (the drafter's own per-sequence
//! contexts), the rows' `[last, draft…]` windows laid end to end as one
//! ragged pass over the rows' caches — the weight sweep once for the batch,
//! each row against its own KV and ring — one readback of every position's
//! argmax, then per row the accepted prefix and the cache resolved to it,
//! every row's probe handed to the drafter in ONE `observe_rows` (the
//! drafter's context re-keyed for the batch in one pass), the tokens
//! streamed. Argmax only: the verify accepts the target's own prediction.

use std::borrow::{Borrow, BorrowMut};

use crate::generation::{HotThread, Sampling};
use crate::nn::decode::{
    end_executor_step, ragged_layout, DecodeOptions, DecodeOut, DecodeStats, OpenPass, PassShape,
    PrefillWork, RoundStats, StepPacer,
};
use crate::nn::model::BatchedCausalLM;
use crate::nn::speculator::{
    readbacks, Anchor, ObserveRow, Probes, SeqId, Session, Speculator, VerifyRowRef,
};
use crate::nn::RoundTiming;
use crate::nn::{RowDriver, RowId, StepReport};
use lloom_tensor::{Device, Result};

struct Lane {
    anchor_tok: u32,
    produced: usize,
    eos: bool,
    done: bool,
    cancelled: bool,
    retired: bool,
    /// The prompt has been fed and the drafter seeded.
    joined: bool,
    all_ids: Vec<u32>,
    pos_before: usize,
    stats: DecodeStats,
    max_tokens: usize,
    round_start_ns: u64,
    /// The row's own EOS set: the verify's accept rule reads it per window.
    eos_ids: Vec<u32>,
}

impl Lane {
    fn reached(&self, produced: usize) -> bool {
        self.max_tokens != 0 && produced >= self.max_tokens
    }
}

struct Slot<'m, M: BatchedCausalLM, R: BorrowMut<Session<'m, M>>> {
    session: R,
    prompt: Vec<u32>,
    lane: Lane,
    generation: u64,
    /// The row's drafter slot, owned here until it goes back to the free
    /// list: `None` once it has. Retirement returns it, so a retired row
    /// still holding its [`Slot`] (retired, not yet taken) owns no drafter
    /// slot, and the drain that removes it returns nothing. A slot returned
    /// twice would be handed to two live rows, whose drafter contexts would
    /// then alias.
    seq: Option<SeqId>,
    _model: std::marker::PhantomData<&'m M>,
}

impl<'m, M: BatchedCausalLM, R: BorrowMut<Session<'m, M>>> Slot<'m, M, R> {
    /// The row's drafter slot while it owns one.
    fn seq(&self) -> SeqId {
        self.seq
            .expect("a row that has not retired owns its drafter slot")
    }
}

/// The speculative batched loop, one round at a time.
pub struct SpecBatchDecoder<'m, M, S, R>
where
    M: BatchedCausalLM + Probes<S::Probe> + crate::profile::Shaped,
    S: Speculator,
    R: BorrowMut<Session<'m, M>>,
{
    spec: S,
    opts: DecodeOptions,
    device: Device,
    slots: Vec<Option<Slot<'m, M, R>>>,
    free: Vec<usize>,
    /// The drafter's free sequence slots.
    free_seq: Vec<usize>,
    generation: u64,
    rounds: u64,
    finished: Vec<(usize, DecodeOut)>,
    pacer: StepPacer,
    hot_thread: HotThread,
    run_start: std::time::Instant,
}

impl<'m, M, S, R> SpecBatchDecoder<'m, M, S, R>
where
    M: BatchedCausalLM + Probes<S::Probe> + crate::profile::Shaped,
    S: Speculator,
    R: BorrowMut<Session<'m, M>>,
{
    /// An empty batch over `spec`, whose sequence slots bound the width. There
    /// is no batch-level EOS set: every row carries its own (`admit_with_eos`,
    /// which refuses an empty one) and the ragged verify reads it per window —
    /// the cells' never-EOS rows sit beside a request's.
    pub fn new(
        _rewind: crate::profile::RewindProof<M::Profile>,
        spec: S,
        opts: DecodeOptions,
        device: Device,
    ) -> Result<Self> {
        if spec.max_draft() == 0 {
            lloom_tensor::bail!(
                "decode_spec_batched: the proposer has max_draft 0 (the K=1 identity); it belongs \
                 on the K=1 batched loop"
            );
        }
        if spec.n_seq() == 0 {
            lloom_tensor::bail!("decode_spec_batched: the proposer has no sequence slots");
        }
        let pacer = StepPacer::for_decode(&opts);
        Ok(Self {
            free_seq: (0..spec.n_seq()).rev().collect(),
            spec,
            opts,
            device,
            slots: Vec::new(),
            free: Vec::new(),
            generation: 0,
            rounds: 0,
            finished: Vec::new(),
            pacer,
            hot_thread: HotThread::claim(),
            run_start: std::time::Instant::now(),
        })
    }

    /// The proposer, for a caller that wants it back.
    pub fn into_speculator(self) -> S {
        self.spec
    }

    /// Rounds run.
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    fn slot_of(&self, id: RowId) -> Option<&Slot<'m, M, R>> {
        self.slots
            .get(id.slot)
            .and_then(|s| s.as_ref())
            .filter(|s| s.generation == id.generation)
    }

    /// Join every row whose prompt is not yet fed: its own streamed prefill
    /// (tiled by `opts.prefill_tile`, paced), seeding its drafter slot; its
    /// first token is the prefill's argmax.
    fn join_pending(
        &mut self,
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<usize> {
        let mut joined = 0usize;
        for i in 0..self.slots.len() {
            let Some(slot) = self.slots[i].as_mut() else {
                continue;
            };
            if slot.lane.joined || slot.lane.cancelled || slot.lane.done {
                continue;
            }
            let id = RowId {
                slot: i,
                generation: slot.generation,
            };
            let t0 = std::time::Instant::now();
            let drafter_slot = slot.seq();
            let s: &mut Session<'m, M> = slot.session.borrow_mut();
            s.seq.last_logits = None;
            let work = PrefillWork::resolve(&self.opts, s.seq.pos, slot.prompt.len(), true)?;
            let before = self.pacer.counters();
            let seeded = s.prefill_streamed(
                &mut self.spec,
                drafter_slot,
                &slot.prompt,
                &work,
                &mut self.pacer,
                &self.device,
            );
            // Keep the same marker queue across joins and rounds. Restoring
            // the round's depth never discards an outstanding seed marker.
            self.pacer.set_depth(self.opts.pacer_depth());
            let (logits, probe, units) = seeded?;
            let after = self.pacer.counters();
            let anchor = logits
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?;
            self.spec
                .observe(drafter_slot, &probe, std::slice::from_ref(&anchor))?;
            let lane = &mut slot.lane;
            lane.stats.prefill_ns = t0.elapsed().as_nanos() as u64;
            lane.stats.prefill_units = units;
            lane.stats.step_markers = after.0 - before.0;
            lane.stats.step_waits = after.1 - before.1;
            lane.anchor_tok = anchor;
            lane.joined = true;
            joined += 1;
            if lane.eos_ids.contains(&anchor) {
                lane.eos = true;
                lane.done = true;
            } else if !lane.reached(lane.produced) {
                on_tokens(id, std::slice::from_ref(&anchor))?;
                lane.all_ids.push(anchor);
                lane.produced += 1;
                lane.done = lane.reached(lane.produced);
            } else {
                lane.done = true;
            }
        }
        Ok(joined)
    }

    /// One speculative round over the live rows.
    fn round(
        &mut self,
        live: &[usize],
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<()> {
        let round_start_ns = self.run_start.elapsed().as_nanos() as u64;
        let seqs: Vec<SeqId> = live
            .iter()
            .map(|&i| self.slots[i].as_ref().expect("live").seq())
            .collect();
        let t = std::time::Instant::now();
        let rb = readbacks();
        let drafts = self.spec.draft(&seqs)?;
        let draft_ns = t.elapsed().as_nanos() as u64;
        let draft_readbacks = readbacks() - rb;
        if drafts.len() != live.len() {
            lloom_tensor::bail!(
                "decode_spec_batched: {} drafts for {} live rows",
                drafts.len(),
                live.len()
            );
        }

        // Disjoint `&mut` borrows of the live rows, in `live`'s order.
        let mut taken: Vec<Option<&mut Slot<'m, M, R>>> =
            self.slots.iter_mut().map(|s| s.as_mut()).collect();
        let mut picked: Vec<&mut Slot<'m, M, R>> = Vec::with_capacity(live.len());
        for &i in live {
            match taken[i].take() {
                Some(s) => picked.push(s),
                None => {
                    lloom_tensor::bail!("decode_spec_batched: row {i} appears twice in a round")
                }
            }
        }
        let model = picked
            .first()
            .map(|s| {
                let s: &Session<'m, M> = Borrow::borrow(&s.session);
                s.model
            })
            .ok_or_else(|| lloom_tensor::Error::msg("decode_spec_batched: a round over no rows"))?;
        let k = self.spec.max_draft();
        for s in picked.iter_mut() {
            let sess: &mut Session<'m, M> = s.session.borrow_mut();
            model.prepare_pass(&mut sess.seq.cache, k + 1)?;
        }
        let mut rows: Vec<VerifyRowRef<'_, M::Cache>> = picked
            .iter_mut()
            .zip(drafts.iter())
            .map(|(s, draft)| {
                let last = s.lane.anchor_tok;
                let sess: &mut Session<'m, M> = s.session.borrow_mut();
                VerifyRowRef {
                    anchor: Anchor {
                        last,
                        pos: sess.seq.pos,
                    },
                    cache: &mut sess.seq.cache,
                    draft,
                }
            })
            .collect();
        let t = std::time::Instant::now();
        let rb = readbacks();
        let shape = PassShape::Ragged {
            live: 0,
            pieces: rows.len() as u16,
            layout: ragged_layout(rows.iter().map(|_| (k + 1, false))),
        };
        self.pacer.before_forward(&self.device)?;
        let pass = OpenPass::begin(&self.device, shape)?;
        let pending = model.verify_rows_enqueue(&mut rows);
        pass.close(&pending)?;
        self.pacer.after_forward(&self.device)?;
        let pending = pending?;
        drop(rows);
        let verify_enqueue_ns = t.elapsed().as_nanos() as u64;
        let enqueue_readbacks = readbacks() - rb;

        let t = std::time::Instant::now();
        let rb = readbacks();
        let eos_owned: Vec<Vec<u32>> = picked.iter().map(|s| s.lane.eos_ids.clone()).collect();
        let eos_sets: Vec<&[u32]> = eos_owned.iter().map(|v| v.as_slice()).collect();
        let mut caches: Vec<&mut M::Cache> = picked
            .iter_mut()
            .map(|s| {
                let sess: &mut Session<'m, M> = s.session.borrow_mut();
                &mut sess.seq.cache
            })
            .collect();
        let results = model.verify_rows_complete(pending, &mut caches, &eos_sets)?;
        drop(caches);
        let verify_wait_ns = t.elapsed().as_nanos() as u64;
        let verify_readbacks = enqueue_readbacks + readbacks() - rb;
        end_executor_step(&self.device)?;

        // Every row's accepted taps to the drafter in one call: the context
        // re-key of B rows is one projection and one K/V pass per layer
        // (`Speculator::observe_rows`), where a call per row was B serial
        // passes and B scatters per layer (`batched-decode-design.md` §12.10).
        let t = std::time::Instant::now();
        let rb = readbacks();
        {
            let rows: Vec<ObserveRow<'_, S::Probe>> = picked
                .iter()
                .zip(results.iter())
                .map(|(s, (accepted, probe))| (s.seq(), probe, accepted.as_slice()))
                .collect();
            if rows.iter().any(|(_, _, accepted)| accepted.is_empty()) {
                lloom_tensor::bail!(
                    "decode_spec_batched: a verify round accepted zero tokens; the target's own \
                     prediction at position 0 is always correct, so this is a broken verify"
                );
            }
            self.spec.observe_rows(&rows)?;
        }
        let observe_ns = t.elapsed().as_nanos() as u64;
        let observe_readbacks = readbacks() - rb;
        for ((s, (accepted, _probe)), &i) in picked.iter_mut().zip(results.into_iter()).zip(live) {
            {
                let sess: &mut Session<'m, M> = s.session.borrow_mut();
                sess.seq.pos += accepted.len();
            }
            let lane = &mut s.lane;
            lane.anchor_tok = *accepted.last().expect("non-empty");
            lane.stats.steps += 1;
            lane.stats.rounds.push(RoundStats {
                accepted: accepted.len(),
                timing: RoundTiming {
                    draft_ns,
                    draft_readbacks,
                    verify_enqueue_ns,
                    verify_wait_ns,
                    verify_ns: verify_enqueue_ns + verify_wait_ns,
                    verify_readbacks,
                    observe_ns,
                    observe_readbacks,
                    ..RoundTiming::default()
                },
                start_ns: round_start_ns,
            });
            let id = RowId {
                slot: i,
                generation: s.generation,
            };
            if lane.cancelled {
                lane.done = true;
                continue;
            }
            let mut batch: Vec<u32> = Vec::with_capacity(accepted.len());
            for &tok in &accepted {
                if lane.reached(lane.produced + batch.len()) {
                    break;
                }
                if lane.eos_ids.contains(&tok) {
                    lane.eos = true;
                    break;
                }
                batch.push(tok);
            }
            if !batch.is_empty() {
                on_tokens(id, &batch)?;
                lane.all_ids.extend_from_slice(&batch);
                lane.produced += batch.len();
            }
            lane.stats
                .step_ns
                .push((self.run_start.elapsed().as_nanos() as u64).saturating_sub(round_start_ns));
            lane.done = lane.eos || lane.reached(lane.produced);
        }
        Ok(())
    }

    /// Return `slot`'s drafter slot to the free list, once: a row whose
    /// ownership was already consumed returns nothing.
    fn release_drafter_slot(free_seq: &mut Vec<usize>, slot: &mut Slot<'m, M, R>) {
        if let Some(seq) = slot.seq.take() {
            debug_assert!(
                !free_seq.contains(&seq.0),
                "drafter slot {} was already on the free list",
                seq.0
            );
            free_seq.push(seq.0);
        }
    }

    /// Per-row bookkeeping at retirement: the accepted ids are the stream's
    /// prefix of the cache (the verify resolved every window to its accepted
    /// length, so nothing overran).
    fn retire(&mut self, i: usize) {
        let free_seq = &mut self.free_seq;
        let slot = self.slots[i].as_mut().expect("a retiring row is held");
        let lane = &mut slot.lane;
        let s: &mut Session<'m, M> = slot.session.borrow_mut();
        let fed = s
            .seq
            .pos
            .saturating_sub(lane.pos_before + slot.prompt.len());
        let in_cache = fed.min(lane.all_ids.len());
        s.seq.tokens.extend_from_slice(&lane.all_ids[..in_cache]);
        s.seq.overrun = fed - in_cache;
        s.seq.last_logits = None;
        let mut stats = std::mem::take(&mut lane.stats);
        stats.overrun = s.seq.overrun;
        stats.produced = lane.produced;
        stats.hot_thread = self.hot_thread.report();
        lane.retired = true;
        let out = DecodeOut {
            ids: std::mem::take(&mut lane.all_ids),
            eos: lane.eos,
            stats,
        };
        Self::release_drafter_slot(free_seq, slot);
        self.finished.push((i, out));
    }
}

impl<'m, M, S, R> RowDriver<'m, M, R> for SpecBatchDecoder<'m, M, S, R>
where
    M: BatchedCausalLM + Probes<S::Probe> + crate::profile::Shaped + 'm,
    S: Speculator,
    R: BorrowMut<Session<'m, M>>,
{
    fn admit_with_eos(
        &mut self,
        mut session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        _seed: u64,
        eos_ids: &[u32],
        max_tokens: usize,
    ) -> crate::nn::batched::Admission<R> {
        // Every refusal — this loop's stricter body, an empty EOS set, an
        // empty prompt, an unreconciled overrun — is decided before a
        // drafter slot or the session is taken, and hands the session back
        // (`Rejected`): a refused admission costs the caller's pool
        // nothing.
        let refused = crate::nn::decode::refuse(
            crate::nn::decode::Body::BatchedWindow,
            &crate::nn::decode::Asks {
                sampling: &sampling,
                processors: false,
                budget: false,
                digest: false,
                logprobs: None,
            },
        )
        .err()
        .map(|e| lloom_tensor::Error::Msg(format!("decode_spec_batched: {e}")))
        .or_else(|| {
            if eos_ids.is_empty() {
                return Some(lloom_tensor::Error::msg(
                    "decode_spec_batched: an empty EOS set; the row would never end",
                ));
            }
            if prompt.is_empty() {
                return Some(lloom_tensor::Error::msg(
                    "decode_spec_batched: an empty prompt (no resume on this loop)",
                ));
            }
            let s: &mut Session<'m, M> = session.borrow_mut();
            (s.seq.overrun > 0).then(|| {
                lloom_tensor::Error::msg(format!(
                    "decode_spec_batched: the sequence's cache is {} positions past its last \
                     token; `Session::reconcile` or `Session::reset` first",
                    s.seq.overrun
                ))
            })
        });
        if let Some(error) = refused {
            return Err(crate::nn::batched::Rejected { session, error });
        }
        let Some(seq) = self.free_seq.pop() else {
            let error = lloom_tensor::Error::msg(format!(
                "decode_spec_batched: every drafter slot is taken ({} rows)",
                self.spec.n_seq()
            ));
            return Err(crate::nn::batched::Rejected { session, error });
        };
        self.generation += 1;
        let generation = self.generation;
        let pos_before = session.borrow_mut().seq.pos;
        let lane = Lane {
            anchor_tok: 0,
            produced: 0,
            eos: false,
            done: false,
            cancelled: false,
            retired: false,
            joined: false,
            all_ids: Vec::with_capacity(max_tokens),
            pos_before,
            stats: DecodeStats {
                step_ns: Vec::with_capacity(max_tokens),
                ..DecodeStats::default()
            },
            max_tokens,
            round_start_ns: 0,
            eos_ids: eos_ids.to_vec(),
        };
        let _ = &lane.round_start_ns;
        let slot = Slot {
            session,
            prompt,
            lane,
            generation,
            seq: Some(SeqId(seq)),
            _model: std::marker::PhantomData,
        };
        let idx = match self.free.pop() {
            Some(i) => {
                self.slots[i] = Some(slot);
                i
            }
            None => {
                self.slots.push(Some(slot));
                self.slots.len() - 1
            }
        };
        Ok(RowId {
            slot: idx,
            generation,
        })
    }

    fn step(
        &mut self,
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<StepReport> {
        let _pool = lloom_tensor::AutoreleaseScope::new();
        let joined = self.join_pending(on_tokens)?;
        let live: Vec<usize> = (0..self.slots.len())
            .filter(|&i| {
                self.slots[i]
                    .as_ref()
                    .map(|s| s.lane.joined && !s.lane.done && !s.lane.cancelled && !s.lane.retired)
                    .unwrap_or(false)
            })
            .collect();
        let b = live.len();
        if b > 0 {
            self.round(&live, on_tokens)?;
            self.rounds += 1;
        }
        let mut retired = 0usize;
        for i in 0..self.slots.len() {
            let done = match self.slots[i].as_ref() {
                Some(s) => (s.lane.done || s.lane.cancelled) && !s.lane.retired,
                None => false,
            };
            if done {
                self.retire(i);
                retired += 1;
            }
        }
        Ok(StepReport {
            live: b,
            pieces: joined,
            piece_tokens: 0,
            retired,
            pending: self.pending_rows(),
            held: 0,
        })
    }

    fn cancel(&mut self, id: RowId) -> bool {
        match self.slots.get_mut(id.slot).and_then(|s| s.as_mut()) {
            Some(s) if s.generation == id.generation && !s.lane.retired => {
                s.lane.cancelled = true;
                true
            }
            _ => false,
        }
    }

    fn take_finished(&mut self) -> Vec<(RowId, R, DecodeOut)> {
        let mut out = Vec::with_capacity(self.finished.len());
        for (i, done) in self.finished.drain(..) {
            let slot = self.slots[i].take().expect("a finished row is held");
            self.free.push(i);
            out.push((
                RowId {
                    slot: i,
                    generation: slot.generation,
                },
                slot.session,
                done,
            ));
        }
        out
    }

    fn drain_rows(&mut self) -> Vec<R> {
        let mut out = Vec::new();
        self.finished.clear();
        for i in 0..self.slots.len() {
            if let Some(mut slot) = self.slots[i].take() {
                self.free.push(i);
                // A retired row returned its drafter slot at its
                // retirement: this takes nothing, rather than putting the
                // same slot on the free list twice.
                Self::release_drafter_slot(&mut self.free_seq, &mut slot);
                out.push(slot.session);
            }
        }
        out
    }

    fn is_idle(&self) -> bool {
        self.slots.iter().all(|s| s.is_none()) && self.finished.is_empty()
    }

    fn rows(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    fn live_rows(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| s.lane.joined && !s.lane.done && !s.lane.retired)
            .count()
    }

    fn pending_rows(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| !s.lane.joined && !s.lane.retired)
            .count()
    }

    fn row_ids(&self) -> Vec<RowId> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.as_ref().map(|s| RowId {
                    slot: i,
                    generation: s.generation,
                })
            })
            .collect()
    }

    fn row_source(&self, id: RowId) -> Option<(&[u32], &M::Cache, bool)> {
        let s = self.slot_of(id)?;
        let session: &Session<'m, M> = Borrow::borrow(&s.session);
        Some((session.seq.tokens(), &session.seq.cache, s.lane.joined))
    }

    fn row_prospective_log(&self, id: RowId) -> Option<Vec<u32>> {
        let s = self.slot_of(id)?;
        let session: &Session<'m, M> = Borrow::borrow(&s.session);
        let mut log = session.seq.tokens().to_vec();
        if !s.lane.joined {
            log.extend_from_slice(&s.prompt);
        }
        Some(log)
    }

    fn holds(&self, id: RowId) -> bool {
        self.slot_of(id).is_some()
    }
}

impl<'m, M> Session<'m, M>
where
    M: BatchedCausalLM,
{
    /// Decode `sessions.len()` sequences together through the speculative
    /// batched loop as one closed batch: every row admitted at once, rounds
    /// until every row has ended (`batched-decode-design.md` §12.4) — the
    /// closed twin of the scheduler's continuous use of
    /// [`SpecBatchDecoder`], for a cell that measures it. Greedy; one EOS set.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_spec_batched<S>(
        sessions: &mut [Self],
        rewind: crate::profile::RewindProof<M::Profile>,
        spec: &mut S,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        S: Speculator,
        M: Probes<S::Probe> + crate::profile::Shaped + 'm,
    {
        let n = sessions.len();
        if n == 0 {
            lloom_tensor::bail!("decode_spec_batched: no sessions");
        }
        if prompts.len() != n {
            lloom_tensor::bail!(
                "decode_spec_batched: {} sessions and {} prompts",
                n,
                prompts.len()
            );
        }
        let mut dec: SpecBatchDecoder<'m, M, &mut S, &mut Session<'m, M>> =
            SpecBatchDecoder::new(rewind, spec, opts.clone(), device.clone())?;
        let mut ids = Vec::with_capacity(n);
        for (s, p) in sessions.iter_mut().zip(prompts) {
            ids.push(dec.admit_with_eos(
                s,
                p.to_vec(),
                Sampling::ArgMax,
                0,
                eos_ids,
                opts.max_tokens,
            )?);
        }
        let mut outs: Vec<Option<DecodeOut>> = (0..n).map(|_| None).collect();
        while !dec.is_idle() {
            dec.step(&mut |row, toks| {
                let i = ids
                    .iter()
                    .position(|&id| id == row)
                    .expect("an admitted row");
                on_tokens(i, toks)
            })?;
            for (row, _session, out) in dec.take_finished() {
                let i = ids
                    .iter()
                    .position(|&id| id == row)
                    .expect("an admitted row");
                outs[i] = Some(out);
            }
        }
        Ok(outs
            .into_iter()
            .map(|o| o.expect("every row finished"))
            .collect())
    }
}
