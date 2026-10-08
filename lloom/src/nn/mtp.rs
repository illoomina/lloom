//! MTP on the generic decoder topology: the NextN head and the drafter
//! that fills the seam's MTP slot (`nn::speculator`) for a `Decoder`
//! family whose checkpoint carries a `nextn` block (Qwen3.5's
//! `blk.{n_layers}.*` + `blk.{n_layers}.nextn.*`).
//!
//! Dataflow, from the reference graph (llama.cpp `src/models/qwen35.cpp`
//! `graph_mtp`): `h_nextn` is the target's residual after its FINAL norm at
//! a position; the NEXT token's embedding is RMS-normed (`enorm`), `h_nextn`
//! is RMS-normed (`hnorm`), the two are concatenated `[e | h]` and projected
//! by `eh_proj [d, 2d]`; one decoder layer of the family's full-attention
//! shape (its own KV) runs on the result; `shared_head_norm` and the
//! SHARED `lm_head` give the draft's logits. The head is one layer, so a
//! draft is one token per observed row; `k > 1` re-applies the layer on its
//! own output with the previous draft's embedding (the Qwen4-exp drafter's
//! scheme), under a window on the layer's state that is rewound after.
//!
//! The verify is the target's (`Decoder: Probes<NextnHidden>`): `k + 1`
//! sequential decode steps inside a window on every layer's state, so the
//! accepted stream is greedy's by construction. A drafter changes the
//! acceptance rate and nothing else.

use crate::nn::decoder::{Layer, StateCapacity};
use crate::nn::speculator::{Draft, KvSource, NextnHidden, SeqId, Speculator};
use crate::nn::RmsNorm;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Module, Result, Tensor, D};

/// The one-layer NextN head over a [`Layer`] of the family's own shape.
#[derive(Debug)]
pub struct NextnHead<L: Layer> {
    /// `nextn.enorm` — RMS over the next token's embedding.
    enorm: RmsNorm,
    /// `nextn.hnorm` — RMS over the target's `h_nextn`.
    hnorm: RmsNorm,
    /// `nextn.eh_proj [d, 2d]` over `[e_normed | h_normed]`.
    eh_proj: QMatMul,
    /// `blk.{n_layers}`: a full-attention layer with its own state.
    layer: L,
    /// `nextn.shared_head_norm` — the norm the shared `lm_head` reads after.
    head_norm: RmsNorm,
    hidden: usize,
}

impl<L: Layer> NextnHead<L> {
    pub fn new(
        enorm: RmsNorm,
        hnorm: RmsNorm,
        eh_proj: QMatMul,
        layer: L,
        head_norm: RmsNorm,
        hidden: usize,
    ) -> Self {
        Self {
            enorm,
            hnorm,
            eh_proj,
            layer,
            head_norm,
            hidden,
        }
    }

    pub fn layer(&self) -> &L {
        &self.layer
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `hidden [1, a, d]` (the target's `h_nextn` at `a` positions) with
    /// `embeds [1, a, d]` (the embeddings of the tokens that FOLLOW those
    /// positions) → `(sample [1, a, d], carry [1, a, d])`: the head-normed
    /// rows the `lm_head` reads, and the layer's output a further draft
    /// step continues from. Appends `a` positions to `state`.
    pub fn forward(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        state: &mut L::State,
    ) -> Result<(Tensor, Tensor)> {
        let (b, a, d) = hidden.dims3()?;
        if b != 1 || d != self.hidden || embeds.dims3()? != (1, a, d) {
            lloom_tensor::bail!(
                "nextn head: hidden {:?} / embeds {:?} for d {}",
                hidden.shape(),
                embeds.shape(),
                self.hidden
            )
        }
        let e = self.enorm.forward(embeds)?;
        let hn = self.hnorm.forward(hidden)?;
        let eh = Tensor::cat(&[&e, &hn], D::Minus1)?.reshape((a, 2 * d))?;
        let h0 = self.eh_proj.forward(&eh)?.reshape((1, a, d))?;
        let carry = self.layer.execute(&h0, state)?;
        let sample = self.head_norm.forward(&carry)?;
        Ok((sample, carry))
    }
}

/// The drafter's per-sequence state, held across requests by a caller that
/// keeps its target cache across them ([`NextnDrafter::with_lanes`],
/// [`NextnDrafter::into_lanes`]).
pub struct NextnLanes<S>(Vec<NextnLane<S>>);

/// What the drafter keeps for one sequence: the head layer's state and
/// where the sequence's drafting stands.
struct NextnLane<S> {
    state: S,
    prompt: Vec<u32>,
    /// The tokens the lane's rows were keyed from: row `j` pairs the
    /// target's `h_nextn` at position `j` with the embedding of
    /// `keyed[j + 1]`.
    keyed: Vec<u32>,
    resets: u64,
    seeded: bool,
    /// The prompt rows the seeding observes have keyed so far (a prefill in
    /// units seeds the lane unit by unit).
    seed_rows: usize,
    draft: Option<u32>,
    /// The layer's output after the last observed row — what a second
    /// draft step continues from.
    carry: Option<Tensor>,
    rows_seen: usize,
}

/// The MTP drafter over a [`NextnHead`]: fills the seam's MTP slot for a
/// `Decoder` target (`Probes<NextnHidden>`).
pub struct NextnDrafter<'a, L: Layer> {
    head: &'a NextnHead<L>,
    /// The target's token embedding (shared).
    embed: &'a QMatMul,
    /// The target's `lm_head` (shared).
    lm_head: &'a QMatMul,
    device: Device,
    cap: StateCapacity,
    /// Tokens proposed per round: the first from the head at the last
    /// observed row, each further one from the layer re-applied on its own
    /// carry with the previous draft's embedding.
    k: usize,
    lanes: Vec<NextnLane<L::State>>,
}

impl<'a, L: Layer> NextnDrafter<'a, L> {
    /// A drafter for one sequence (`SeqId(0)`); the lane's state is sized
    /// by `cap` (the target's KV capacity).
    pub fn new(
        head: &'a NextnHead<L>,
        embed: &'a QMatMul,
        lm_head: &'a QMatMul,
        device: &Device,
        cap: StateCapacity,
    ) -> Result<Self> {
        let lane = NextnLane {
            state: head.layer().new_state(device, &cap)?,
            prompt: Vec::new(),
            keyed: Vec::new(),
            resets: 0,
            seeded: false,
            seed_rows: 0,
            draft: None,
            carry: None,
            rows_seen: 0,
        };
        Ok(Self {
            head,
            embed,
            lm_head,
            device: device.clone(),
            cap,
            k: 1,
            lanes: vec![lane],
        })
    }

    /// Propose `k` tokens a round (`k ≥ 1`; the target verifies `k + 1`
    /// positions, byte-identical to greedy whatever `k`).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }

    /// Draft over `lanes`, a previous drafter's state, in place of fresh
    /// ones: each lane is kept for the prefix its next `begin_at` reuses,
    /// or reset there.
    pub fn with_lanes(mut self, lanes: Option<NextnLanes<L::State>>) -> Self {
        if let Some(NextnLanes(l)) = lanes {
            if l.len() == self.lanes.len() {
                self.lanes = l;
            }
        }
        self
    }

    /// The lanes, for the next request's drafter.
    pub fn into_lanes(mut self) -> NextnLanes<L::State> {
        NextnLanes(std::mem::take(&mut self.lanes))
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Positions the head layer's state holds, over every sequence.
    pub fn rows_seen(&self) -> usize {
        self.lanes.iter().map(|l| l.rows_seen).sum()
    }

    /// `begin_at`s that reset a lane for a reused prefix it was not keyed
    /// from.
    pub fn lane_resets(&self) -> u64 {
        self.lanes.iter().map(|l| l.resets).sum()
    }

    pub fn capacity(&self) -> StateCapacity {
        self.cap
    }

    fn lane(&mut self, seq: SeqId) -> Result<&mut NextnLane<L::State>> {
        let n = self.lanes.len();
        self.lanes.get_mut(seq.0).ok_or_else(|| {
            lloom_tensor::Error::msg(format!("NextnDrafter serves {n} sequence(s), got {seq:?}"))
        })
    }

    /// One sequence's draft: the head's token at the last observed row,
    /// then `k − 1` re-applications under a window rewound after.
    fn draft_one(&mut self, seq: SeqId) -> Result<Vec<u32>> {
        let (head, embed, lm_head, k, d) = (
            self.head,
            self.embed,
            self.lm_head,
            self.k,
            self.head.hidden(),
        );
        let device = self.device.clone();
        let lane = self.lane(seq)?;
        // Before the first `observe` there is no row to draft from; propose
        // id 0 — any proposal is corrected by the verify.
        let first = lane.draft.unwrap_or(0);
        let mut tokens = vec![first];
        if k > 1 {
            if let Some(carry0) = lane.carry.clone() {
                let mark = L::open_window(&mut lane.state)?;
                let mut carry = carry0;
                let mut result = Ok(());
                for _ in 1..k {
                    let prev = *tokens.last().expect("one token");
                    let step = (|| -> Result<u32> {
                        let toks = Tensor::from_vec(vec![prev], (1, 1), &device)?;
                        let emb = embed.embedding(&toks)?;
                        let (sample, next) = head.forward(&carry, &emb, &mut lane.state)?;
                        let logits = lm_head.forward(&sample.reshape((1, d))?)?;
                        let id = logits
                            .flatten_all()?
                            .argmax(D::Minus1)?
                            .to_scalar::<u32>()?;
                        carry = next;
                        Ok(id)
                    })();
                    match step {
                        Ok(id) => tokens.push(id),
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
                L::rewind_window(&mut lane.state, mark)?;
                result?;
            }
        }
        Ok(tokens)
    }
}

impl<L: Layer> Speculator for NextnDrafter<'_, L> {
    type Probe = NextnHidden;
    type Family = crate::nn::speculator::Mtp;

    fn n_seq(&self) -> usize {
        self.lanes.len()
    }

    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        self.begin_at(seq, prompt, &[])
    }

    /// The lane is kept for `kept` iff its rows were keyed from exactly
    /// those tokens and the one after them, `prompt[0]`: it is cut back to
    /// `kept.len()` rows and `prompt` is keyed after them. Otherwise it is
    /// reset and keys `prompt` from its own position 0 (the head's
    /// attention reads positions relative to each other, so the restart
    /// costs the reused prefix's context, which the acceptance rate pays
    /// for, and nothing else).
    fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
        let lane = self.lane(seq)?;
        let at = kept.len();
        let follow = at > 0
            && lane.rows_seen >= at
            && lane.keyed.len() > at
            && lane.keyed[..at] == kept[..]
            && prompt.first() == Some(&lane.keyed[at]);
        if follow && L::truncate_state(&mut lane.state, at).is_ok() {
            lane.keyed.truncate(at + 1);
            lane.rows_seen = at;
        } else {
            L::reset_state(&mut lane.state);
            lane.keyed.clear();
            lane.rows_seen = 0;
            if at > 0 {
                lane.resets += 1;
            }
        }
        lane.prompt = prompt.to_vec();
        lane.seeded = false;
        lane.seed_rows = 0;
        lane.draft = None;
        lane.carry = None;
        Ok(())
    }

    fn context_len(&self, seq: SeqId) -> Option<usize> {
        self.lanes.get(seq.0).map(|l| l.rows_seen)
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let mut drafts = Vec::with_capacity(active.len());
        for &seq in active {
            drafts.push(Draft::host(self.draft_one(seq)?));
        }
        Ok(drafts)
    }

    /// `probe [1, a, d]` = the target's `h_nextn` at the `a` positions the
    /// round ran; `accepted` = the tokens the target kept at those
    /// positions. Row `i` pairs with the token that follows it: during the
    /// prefill the prompt's next token, and the accepted anchor for the
    /// prompt's last row. A prefill in units seeds the lane unit by unit.
    fn observe(&mut self, seq: SeqId, probe: &Tensor, accepted: &[u32]) -> Result<()> {
        let a = probe.dim(1)?;
        if a == 0 {
            return Ok(());
        }
        let (head, embed, lm_head, d) = (self.head, self.embed, self.lm_head, self.head.hidden());
        let device = self.device.clone();
        let lane = self.lane(seq)?;
        let nexts: Vec<u32> = if !lane.seeded {
            let off = lane.seed_rows;
            let mut v: Vec<u32> = lane.prompt.iter().skip(off + 1).take(a).copied().collect();
            if off + a >= lane.prompt.len() {
                v.extend_from_slice(accepted);
                lane.seeded = true;
            }
            lane.seed_rows = off + a;
            v
        } else {
            accepted.to_vec()
        };
        if nexts.len() != a {
            lloom_tensor::bail!(
                "NextnDrafter::observe: {a} probe rows but {} next tokens (prompt {}, accepted {})",
                nexts.len(),
                lane.prompt.len(),
                accepted.len()
            )
        }
        let toks = Tensor::from_vec(nexts.clone(), (1, a), &device)?;
        let emb = embed.embedding(&toks)?; // [1, a, d]
        let (sample, carry) = head.forward(probe, &emb, &mut lane.state)?;
        let last = sample.narrow(1, a - 1, 1)?.reshape((1, d))?;
        let next = lm_head
            .forward(&last)?
            .flatten_all()?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        if lane.keyed.is_empty() {
            lane.keyed.push(lane.prompt.first().copied().unwrap_or(0));
        }
        lane.keyed.extend_from_slice(&nexts);
        lane.draft = Some(next);
        lane.carry = Some(carry.narrow(1, a - 1, 1)?);
        lane.rows_seen += a;
        Ok(())
    }
}
