//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use lloom_tensor::{DType, Result, Tensor};
use rand::SeedableRng;

mod distribution;
pub mod budget;
pub mod hot_thread;
pub mod poll;
pub mod processors;
pub mod readback;
pub mod token_ring;
/// The token ring's stateful fuzz model and driver (#508), outside `#[cfg(test)]`
/// under the `fuzzing` feature so `lloom-fuzz` (#706) can drive it.
#[cfg(any(test, feature = "fuzzing"))]
pub mod token_ring_fuzz;

pub use budget::{ThinkingBudget, ThinkingReport};
pub use hot_thread::{HotThread, HotThreadArm, HotThreadReport, QosClass};
pub use processors::Processors;
pub use readback::{
    log_softmax_row, Blocking, CommitBudget, EosRead, Geometry, LogprobCapture, Logprobs,
    ReadbackArms, ReadbackCounters, ReadbackStrategy, Ring, RingLap, RingReader,
    WAKE_LIVE_EVERY_DEFAULT,
};

/// The draw a sampler makes. Every temperature arm is keyed by the seed and
/// the absolute position of the token it draws when the caller arms one
/// ([`LogitsProcessor::at_position`]), so a draw does not depend on how many
/// draws preceded it — a speculative window and a step-by-step decode land
/// on the same tokens at one seed.
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    /// Multinomial at the temperature, by inverse CDF: ONE uniform per
    /// position, the smallest token whose cumulative probability under
    /// the processed-score distribution passes it. Host draws use the shared
    /// pinned f32 contract with filters disabled. Metal retains the incumbent
    /// inverse-CDF route until its native bridge is accepted; it is not yet
    /// this contract. `generation.sampling.method = multinomial`, the default.
    All {
        temperature: f64,
    },
    TopK {
        k: usize,
        temperature: f64,
    },
    TopP {
        p: f64,
        temperature: f64,
    },
    TopKThenTopP {
        k: usize,
        p: f64,
        temperature: f64,
    },
    /// Centered Gumbel-max: separately rounded f32 `(z-max)*f32(1/T)+G`,
    /// one variate per original vocabulary id from seeded host noise.
    /// CPU/CUDA share the filtered host implementation; public Metal keeps
    /// its incumbent expression until guarded native activation is admitted
    /// (`generation.sampling.method = gumbel`, `--gumbel`).
    GumbelSoftmax {
        temperature: f64,
    },
    /// Filtered Gumbel is currently an explicit host draw, not native-fast.
    FilteredGumbel {
        temperature: f64,
        k: usize,
        p: f64,
    },
}

/// Temperatures below this resolve to greedy for pre-release. This is a
/// request policy, not equality of every tiny-positive distribution to argmax.
pub const GREEDY_BELOW: f64 = 1e-7;

impl Sampling {
    /// The arm a request's sampling fields name, every field validated
    /// whether or not the arm reads it (`docs/dispatch.md` §11.2j's edges).
    ///
    /// - `temperature`: unset, or below [`GREEDY_BELOW`], is greedy; a value
    ///   that is not a finite number `>= 0` is refused.
    /// - `top_k`: 0 is off; a `k` at least the vocabulary is off where the
    ///   row is known (the filter's).
    /// - `top_p`: 1 is off, 0 keeps the top candidate; a value outside
    ///   [0, 1] is refused.
    /// - `gumbel` preserves its method and applies requested filters first.
    ///   Filtered Gumbel currently resolves to an explicit host arm.
    pub fn resolve(
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        gumbel: bool,
    ) -> Result<Sampling> {
        if let Some(t) = temperature.filter(|t| !(t.is_finite() && *t >= 0.0)) {
            lloom_tensor::bail!(
                "temperature = {t}: not a finite number >= 0 (unset, 0 or below {GREEDY_BELOW} \
                 is greedy)"
            );
        }
        if let Some(p) = top_p.filter(|p| !(0.0..=1.0).contains(p)) {
            lloom_tensor::bail!("top_p = {p}: outside [0, 1] (1 is off, 0 keeps the top token)");
        }
        let k = top_k.filter(|&k| k > 0);
        let p = top_p.filter(|&p| p < 1.0);
        let temperature = match temperature {
            Some(t) if t >= GREEDY_BELOW => t,
            _ => return Ok(Sampling::ArgMax),
        };
        if gumbel {
            if k.is_some() || p.is_some() {
                return Ok(Sampling::FilteredGumbel {
                    temperature,
                    k: k.unwrap_or(0),
                    p: p.unwrap_or(1.0),
                });
            }
            return Ok(Sampling::GumbelSoftmax { temperature });
        }
        Ok(match (k, p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (None, None) => Sampling::All { temperature },
        })
    }

    /// Arms that require host filtering until the native bridge is accepted.
    pub fn requires_host(&self) -> bool {
        matches!(self, Self::TopK { .. } | Self::TopP { .. }
            | Self::TopKThenTopP { .. } | Self::FilteredGumbel { .. })
    }

    fn distribution(&self) -> Option<(f64, usize, f64)> {
        Some(match *self {
            Self::All { temperature } => (temperature, 0, 1.0),
            Self::TopK { temperature, k } => (temperature, k, 1.0),
            Self::TopP { temperature, p } => (temperature, 0, p),
            Self::TopKThenTopP { temperature, k, p }
            | Self::FilteredGumbel { temperature, k, p } => (temperature, k, p),
            Self::GumbelSoftmax { temperature } => (temperature, 0, 1.0),
            _ => return None,
        })
    }

    /// The arm as a report line names it: the method and its parameters.
    pub fn name(&self) -> String {
        match self {
            Sampling::ArgMax => "argmax".into(),
            Sampling::All { temperature } => format!("multinomial:t={temperature}"),
            Sampling::GumbelSoftmax { temperature } => format!("gumbel:t={temperature}"),
            Sampling::FilteredGumbel { temperature, k, p } => {
                format!("host-filtered-gumbel:k={k}:p={p}:t={temperature}")
            }
            Sampling::TopK { k, temperature } => format!("top-k:k={k}:t={temperature}"),
            Sampling::TopP { p, temperature } => format!("top-p:p={p}:t={temperature}"),
            Sampling::TopKThenTopP { k, p, temperature } => {
                format!("top-k-top-p:k={k}:p={p}:t={temperature}")
            }
        }
    }
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    /// The seed the run was given, kept so a per-position noise stream can be
    /// derived from `(seed, position)` ([`Self::at_position`]).
    seed: u64,
    /// The absolute position the next draw is for, when the caller keys the
    /// noise by position rather than by draw count. `None` is the running
    /// `rng` — the incumbent behaviour, and what every non-speculative caller
    /// takes unless it arms one.
    position: Option<u64>,
    sampling: Sampling,
    host: distribution::Scratch,
    /// The processors between the logits and the draw; `None` when every one
    /// is neutral, so the sampler's path is exactly the one it has without them.
    stage: Option<processors::Stage>,
    /// Persistent Shared readback for host processed scores or callback
    /// probabilities. Grown on first use / vocabulary growth; native draws
    /// do not allocate it.
    #[cfg(feature = "metal")]
    prs_slot: Option<lloom_tensor::metal_backend::HostSlot>,
    /// The drawn id's host-read slot ([`Self::draw_and_read`]).
    #[cfg(feature = "metal")]
    draw_slot: Option<lloom_tensor::metal_backend::HostSlot>,
    /// A running SHA-256 over every step's raw logits
    /// ([`Self::with_logits_digest`]): a diagnostic, off by default.
    digest: Option<sha2::Sha256>,
    /// The raw logits of the last verify window drawn while digesting, so
    /// the rows the round keeps are hashed ([`Self::commit_window`]).
    window_logits: Option<Tensor>,
    /// The thinking budget a verify window's draws are fed through
    /// ([`Self::with_window_budget`]); the K=1 loop's is its readback arm's.
    window_forcing: Option<budget::HostForcing>,
    /// Rows of logits the digest has taken.
    digest_rows: usize,
}

/// A [`LogitsProcessor`]'s state before a draw ([`LogitsProcessor::mark`]).
#[derive(Clone)]
pub struct DrawMark {
    rng: rand::rngs::StdRng,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng,
            seed,
            position: None,
            sampling,
            host: distribution::Scratch::default(),
            stage: None,
            #[cfg(feature = "metal")]
            prs_slot: None,
            #[cfg(feature = "metal")]
            draw_slot: None,
            digest: None,
            window_logits: None,
            window_forcing: None,
            digest_rows: 0,
        }
    }

    /// Hash every step's raw logits — before the processors, as the
    /// harnesses' `logits_sha256` does: each value's f32 bits little-endian,
    /// one running SHA-256 over the run — beside the draw. A diagnostic: it
    /// reads the whole vocabulary back every step, so a caller runs it on a
    /// host-drawing arm and says so.
    pub fn with_logits_digest(mut self, on: bool) -> Self {
        self.digest = on.then(sha2::Sha256::default);
        self
    }

    /// Feed a verify window's draws through a thinking budget: what a row
    /// FEEDS is the budget's rule over the draws before it
    /// ([`Self::window_fed`]), as the K=1 loop's readback arm feeds a
    /// forced token in place of a draw. `None` removes it.
    pub fn with_window_budget(mut self, budget: Option<&ThinkingBudget>) -> Self {
        self.window_forcing = budget.cloned().map(budget::HostForcing::new);
        self
    }

    /// The tokens a verify window's rows feed, given their draws: the draws
    /// themselves, or under a thinking budget its rule applied row by row
    /// from the run's position in it — the forced tail where the budget is
    /// spent and the block still open. Moves nothing: the run's position
    /// in the budget advances by what the round keeps
    /// ([`Self::commit_window`]).
    pub fn window_fed(&self, drawn: &[u32]) -> Vec<u32> {
        match &self.window_forcing {
            None => drawn.to_vec(),
            Some(f) => {
                let mut f = f.clone();
                drawn.iter().map(|&d| f.feed(d)).collect()
            }
        }
    }

    /// Whether a verify window must draw its rows with this sampler rather
    /// than read their raw argmax: a temperature, processors, a thinking
    /// budget or a logits digest each need the rows' logits.
    pub fn draws_windows(&self) -> bool {
        self.sampling != Sampling::ArgMax
            || self.stage.is_some()
            || self.window_forcing.is_some()
            || self.digest.is_some()
    }

    pub fn digests(&self) -> bool {
        self.digest.is_some()
    }

    /// Rows of logits the digest has taken: one a step on the K=1 loop, one
    /// a kept row under speculation, the EOS step's included either way.
    pub fn digest_rows(&self) -> usize {
        self.digest_rows
    }

    /// The digest so far, hex; `None` unless [`Self::with_logits_digest`].
    pub fn logits_digest(&self) -> Option<String> {
        use sha2::Digest as _;
        self.digest
            .as_ref()
            .map(|h| format!("{:x}", h.clone().finalize()))
    }

    fn digest_step(&mut self, logits: &Tensor) -> Result<()> {
        use sha2::Digest as _;
        let Some(h) = self.digest.as_mut() else {
            return Ok(());
        };
        let v = logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        self.digest_rows += logits.dims()[..logits.rank().saturating_sub(1)]
            .iter()
            .product::<usize>()
            .max(1);
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        h.update(&bytes);
        Ok(())
    }

    /// Run `processors` before every draw (bias, penalties, min-p;
    /// [`processors`]). Neutral settings install nothing.
    pub fn with_processors(mut self, processors: Processors) -> Self {
        self.stage = processors
            .is_active()
            .then(|| processors::Stage::new(processors));
        self
    }

    /// The sampler's state before a draw, so a draw taken back can be made
    /// again as if the first had not been (a provisional step's redraw,
    /// `Session::decode`): the rng, which is all a draw without processors
    /// advances. `None` with processors — their history advances from the
    /// drawn token, on the device for a device draw — so such a sampler's
    /// draws are not taken back.
    pub fn mark(&self) -> Option<DrawMark> {
        self.stage.is_none().then(|| DrawMark {
            rng: self.rng.clone(),
        })
    }

    /// Put the sampler back to `mark` ([`Self::mark`]).
    pub fn rewind(&mut self, mark: DrawMark) {
        self.rng = mark.rng;
    }

    /// Key this sampler's noise by the ABSOLUTE position a draw is for,
    /// rather than by how many draws preceded it.
    ///
    /// A speculative round draws a token per window position and keeps only
    /// the accepted prefix, so the rows past the first mismatch are drawn and
    /// thrown away. On the running stream those discarded draws advance it,
    /// and every later token's noise moves with the drafter's luck — so the
    /// same seed gives a different reply depending on what was accepted,
    /// which is not a sampler a `deterministic` run can carry. Armed with a
    /// position, each draw derives its own stream from `(seed, position)`:
    /// the row at position `p` is the same row whether or not any row before
    /// it was rejected, and a sampled speculative decode reproduces a
    /// sampled non-speculative one at one seed.
    ///
    /// `None` returns to the running stream. Every temperature arm reads
    /// the key: gumbel-max its noise, multinomial and the host-sort arms
    /// (top-k, top-p: the filter, then the inverse CDF) their one uniform.
    pub fn at_position(&mut self, position: Option<u64>) {
        self.position = position;
    }

    /// The position this sampler's noise is keyed to, when any
    /// ([`Self::at_position`]).
    pub fn position(&self) -> Option<u64> {
        self.position
    }

    /// Whether this sampler's draws are position-keyed — so a caller that
    /// must not advance a shared stream (a verify's per-row draw) can check
    /// rather than assume.
    pub fn position_keyed(&self) -> bool {
        self.position.is_some()
    }

    /// The processors this sampler runs, when any is active. A caller whose
    /// path draws without this sampler (the ring's fused argmax publish, a
    /// verify that accepts the target's argmax) must not take it when this
    /// is `Some`.
    pub fn processors(&self) -> Option<&Processors> {
        self.stage.as_ref().map(|s| s.config())
    }

    /// The prompt the repetition penalty covers — the whole sequence the run
    /// continues. Restarts the output history.
    pub fn observe_prompt(&mut self, prompt: &[u32]) {
        if let Some(stage) = &mut self.stage {
            stage.observe_prompt(prompt);
        }
    }

    /// The draw's temperature, `None` for argmax (min-p's input).
    fn temperature(&self) -> Option<f64> {
        match self.sampling {
            Sampling::ArgMax => None,
            Sampling::All { temperature }
            | Sampling::TopK { temperature, .. }
            | Sampling::TopP { temperature, .. }
            | Sampling::TopKThenTopP { temperature, .. }
            | Sampling::GumbelSoftmax { temperature }
            | Sampling::FilteredGumbel { temperature, .. } => Some(temperature),
        }
    }

    /// The sampling arm this processor runs, so a caller that must refuse a
    /// host-only arm (the token ring, `DESIGN.md` §11.2c) can read it back
    /// from the resolved value rather than from the flag that selected it.
    pub fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        // `manual_filter` suggests `.filter(|&v| v >= 1e-7)`, but `f64` is only `PartialOrd`:
        // that drops `NaN` where the comparison below keeps it. Left as-is on purpose.
        #[allow(clippy::manual_filter)]
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        self.draw_on_device(&logits)?.to_scalar::<u32>()
    }

    /// The device draw AND its id: the tensor for the next forward's input,
    /// the id for the host's streaming and EOS test (#814 part 5, item A).
    ///
    /// The id is read from a host-read slot the draw's argmax wrote, not from
    /// the draw's own buffer: whether THAT is CPU-mapped is the pool's
    /// rounding (§15.2 #12a), so a per-token read of it takes the blit by the
    /// allocator's luck. The slot is Shared for the session's life, so the
    /// mapped read is structural.
    ///
    /// `None` where there is no slot to write (off Metal): the caller reads
    /// the draw back as it always did.
    pub fn draw_and_read(&mut self, logits: &Tensor) -> Result<Option<(Tensor, u32)>> {
        #[cfg(feature = "metal")]
        {
            // The device is decided BEFORE the draw: a draw runs the
            // processors' stage, which advances their history, and a caller
            // that took `None` would draw again and advance it twice.
            let Ok(m) = logits.device().as_metal_device() else {
                return Ok(None);
            };
            let m = m.clone();
            // Through the processors' stage, so the penalties, bias and min-p
            // apply and their history advances from the drawn token: this is
            // `sample_argmax_on_device`'s path, not `draw_on_device` bare.
            let draw = self.sample_argmax_on_device(logits)?;
            if !matches!(&self.draw_slot, Some(s) if s.cap() >= 1) {
                self.draw_slot = Some(lloom_tensor::metal_backend::HostSlot::new(
                    &m,
                    1,
                    DType::U32,
                )?);
            }
            let slot = self.draw_slot.as_ref().expect("made above");
            // One `copy2d` of the drawn id into the slot, in the step's own
            // command buffer — what the ring's publish encodes. The read that
            // follows is of the session's mapping, not of whatever bucket the
            // pool rounded the draw's own buffer into.
            slot.write_from(&draw)?;
            let id = slot.read_committed::<u32>(1)?[0];
            Ok(Some((draw, id)))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = logits;
            Ok(None)
        }
    }

    /// A device draw's id on the host: through the session's draw slot on
    /// Metal (one copy into a Shared slot the step already commits, read in
    /// place — [`Self::draw_and_read`]'s read), else read back.
    fn read_drawn(&mut self, draw: &Tensor) -> Result<u32> {
        #[cfg(feature = "metal")]
        if let Ok(m) = draw.device().as_metal_device() {
            if !matches!(&self.draw_slot, Some(s) if s.cap() >= 1) {
                self.draw_slot = Some(lloom_tensor::metal_backend::HostSlot::new(
                    m,
                    1,
                    DType::U32,
                )?);
            }
            let slot = self.draw_slot.as_ref().expect("made above");
            slot.write_from(&draw.flatten_all()?)?;
            return Ok(slot.read_committed::<u32>(1)?[0]);
        }
        Ok(draw.flatten_all()?.to_vec1::<u32>()?[0])
    }

    /// The draw as a `U32` tensor on the input device. Argmax, Metal All,
    /// and unfiltered Gumbel keep their device route. Filtered methods use
    /// the explicit host contract and upload the checked id; they are not
    /// ring-eligible until the native bridge is accepted.
    ///
    /// Processors, when installed, run first and advance their history from
    /// the drawn tensor.
    pub fn sample_argmax_on_device(&mut self, logits: &Tensor) -> Result<Tensor> {
        self.sample_argmax_on_device_fed(logits, Ok)
    }

    /// [`Self::sample_argmax_on_device`], with the step's fed token
    /// `feed(draw)` in place of the draw: the token the processors' history
    /// counts is the one the step feeds (a thinking budget's forced tail,
    /// [`budget`]).
    pub fn sample_argmax_on_device_fed(
        &mut self,
        logits: &Tensor,
        feed: impl FnOnce(Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let temperature = self.temperature();
        let Some(stage) = self.stage.as_mut() else {
            return feed(self.draw_on_device(logits)?);
        };
        let processed = stage.apply(logits, temperature)?;
        let token = feed(self.draw_on_device(&processed)?)?;
        if let Some(stage) = self.stage.as_mut() {
            stage.record(&token)?;
        }
        Ok(token)
    }

    /// One drawn token per row of a verify window's logits — the target's own
    /// token at each of the `t` positions `[anchor, draft…]` covers — read
    /// back to the host ([`Self::window_draw_device`] with no window inputs).
    pub fn window_draw(&mut self, logits: &Tensor, first_pos: u64) -> Result<Vec<u32>> {
        self.window_draw_device(logits, first_pos, None)?
            .to_vec1::<u32>()
    }

    /// One drawn token per row of a verify window's logits, left on the
    /// device as a `[t]` `U32` tensor.
    ///
    /// This is what a sampled verify compares the draft against, in place of
    /// the single `logits.argmax(D::Minus1)` the greedy verify takes. The
    /// acceptance rule is unchanged ([`crate::nn::speculator::accept_prefix`]
    /// compares the target's token to the draft's and keeps the target's at
    /// the first disagreement); what changes is only how that token is
    /// produced, so the window's output is the target's own distribution
    /// rather than its mode.
    ///
    /// `logits` is `[1, t, vocab]` or `[t, vocab]`; `first_pos` is the
    /// ABSOLUTE position of the token row 0 draws. Row `i` draws keyed to
    /// `first_pos + i` (its noise, or its uniform), so the token at a
    /// position does not depend on how many rows before it were drawn and
    /// then rejected — which is what makes the sampled window reproduce a
    /// sampled step-by-step decode at one seed. The rows of every arm are
    /// drawn together on the existing unfiltered native routes. Host methods
    /// read processed scores once and reuse one rank workspace row by row;
    /// filtered Gumbel consumes original-id noise without a second noise vector.
    ///
    /// Processors read a per-row history: row `i` counts the run's output
    /// so far and the window's inputs `1..=i` — the drafts the rows before
    /// it were fed, which are the run's output whenever row `i` is read at
    /// all (a row past a rejection is discarded). `inputs` is the window's
    /// `[t]` `U32` input ids `[anchor, d1…]`; without it a sampler with
    /// processors is refused by name. The history itself does not move:
    /// the caller counts what the round kept ([`Self::commit_window`]).
    pub fn window_draw_device(
        &mut self,
        logits: &Tensor,
        first_pos: u64,
        inputs: Option<&Tensor>,
    ) -> Result<Tensor> {
        let rows = match logits.rank() {
            3 => {
                let (b, _t, _v) = logits.dims3()?;
                if b != 1 {
                    lloom_tensor::bail!(
                        "window_draw: the window is single-sequence (B=1), got batch {b}"
                    );
                }
                logits.squeeze(0)?.contiguous()?
            }
            2 => logits.contiguous()?,
            r => lloom_tensor::bail!("window_draw: expected [1, t, vocab] or [t, vocab], rank {r}"),
        };
        let (t, vocab) = rows.dims2()?;
        if t == 0 || vocab == 0 {
            lloom_tensor::bail!("window_draw: an empty window or vocabulary");
        }
        if first_pos.checked_add(t as u64 - 1).is_none() {
            lloom_tensor::bail!("window_draw: absolute position overflow");
        }
        let raw_rows = self.digest.is_some().then(|| rows.clone());
        let temperature = self.temperature();
        let rows = match (self.stage.as_mut(), inputs) {
            (None, _) => rows,
            (Some(_), None) => lloom_tensor::bail!(
                "window_draw: logits processors (logit_bias, repetition/presence/frequency \
                 penalties, min_p) have per-token history, and this window carries no inputs \
                 to build each row's from"
            ),
            (Some(stage), Some(inputs)) => stage.apply_rows(&rows, inputs, temperature)?,
        };
        let dev = rows.device().clone();
        let seed = self.seed;
        let drawn = match self.sampling.clone() {
            Sampling::ArgMax => rows.argmax(lloom_tensor::D::Minus1)?,
            Sampling::GumbelSoftmax { temperature } if dev.is_metal() => {
                use rayon::prelude::*;
                let rows = rows.to_dtype(DType::F32)?;
                let mut g = vec![0f32; t * vocab];
                g.par_chunks_mut(vocab)
                    .enumerate()
                    .for_each(|(i, row)| Self::fill_gumbel(seed, first_pos + i as u64, row));
                let noise = Tensor::from_vec(g, (t, vocab), &dev)?;
                (rows + noise.affine(temperature, 0.0)?)?.argmax(lloom_tensor::D::Minus1)?
            }
            Sampling::All { temperature } if dev.is_metal() => {
                let u: Vec<f32> = (0..t as u64)
                    .map(|i| Self::uniform_at(seed, first_pos + i))
                    .collect();
                rows.sample_inverse_cdf(temperature, &Tensor::from_vec(u, t, &dev)?)?
            }
            ref arm => {
                let (temperature, k, p) = arm.distribution().expect("host distribution arm");
                let rows = rows.to_dtype(DType::F32)?;
                // Preserve window's single read; do not copy a probability row
                // or allocate rank/noise scratch for each position.
                self.host.load(&rows)?;
                let mut ids = Vec::with_capacity(t);
                // Move capacity locally without discarding it on return/error.
                let mut values = std::mem::take(&mut self.host.values);
                let result = (|| -> Result<()> {
                    for (i, row) in values.chunks_mut(vocab).enumerate() {
                        self.host.prepare_row(row, temperature, k, p, matches!(arm, Sampling::FilteredGumbel { .. } | Sampling::GumbelSoftmax { .. }))?;
                        let pos = first_pos + i as u64;
                        let id = if matches!(arm, Sampling::FilteredGumbel { .. } | Sampling::GumbelSoftmax { .. }) {
                            let mut rng = rand::rngs::StdRng::seed_from_u64(Self::stream_key(seed, pos));
                            self.host.gumbel_row(row, temperature, &mut rng)
                        } else {
                            self.host.draw_row(row, Self::uniform_at(seed, pos))
                        };
                        ids.push(id);
                    }
                    Ok(())
                })();
                self.host.values = values;
                result?;
                Tensor::from_vec(ids, t, &dev)?
            }
        };
        self.window_logits = raw_rows;
        Ok(drawn)
    }

    /// Count the tokens a verify round kept into the processors' history,
    /// in order — each is the run's output, as a step-by-step decode counts
    /// its draws. A no-op without processors.
    ///
    /// Under a thinking budget the run's position in it advances by the
    /// kept tokens (each a fed token, which the rule returns unchanged), and
    /// under a logits digest the kept rows of the last window drawn are
    /// hashed, as the K=1 loop hashes each step's logits.
    pub fn commit_window(&mut self, accepted: &[u32]) -> Result<()> {
        if let Some(stage) = self.stage.as_mut() {
            for &id in accepted {
                stage.record_id(id)?;
            }
        }
        if let Some(f) = self.window_forcing.as_mut() {
            for &id in accepted {
                f.feed(id);
            }
        }
        if let Some(rows) = self.window_logits.take() {
            let kept = accepted.len().min(rows.dim(0)?);
            if kept > 0 {
                self.digest_step(&rows.narrow(0, 0, kept)?)?;
            }
        }
        Ok(())
    }

    /// Fill `out` with the Gumbel variates of `(seed, pos)`'s stream: the
    /// position-keyed noise one draw at `pos` adds (`gumbel_variates`).
    fn fill_gumbel(seed: u64, pos: u64, out: &mut [f32]) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(Self::stream_key(seed, pos));
        for g in out.iter_mut() {
            *g = Self::gumbel_variate(&mut rng);
        }
    }

    fn gumbel_variate(rng: &mut rand::rngs::StdRng) -> f32 {
        use rand::Rng;
        let u: f64 = rng.random_range(1e-9..0.9999999);
        (-((-u.ln()).ln())) as f32
    }

    fn draw_on_device(&mut self, logits: &Tensor) -> Result<Tensor> {
        match self.sampling {
            // t = 0 -- a bare device argmax, the original mechanism.
            Sampling::ArgMax => logits.argmax(lloom_tensor::D::Minus1),
            // t > 0 without a nucleus/top-k -- gumbel-max with the Gumbel noise
            // generated on the HOST and fed as an input, not `rand_like` on the
            // GPU. The noise is data-independent (it never reads the logits), so
            // producing it on the device, inline, per token, is work on the
            // critical path that need not be there -- llama.cpp's backend dist
            // sampler feeds `inp_uniforms` the same way (host `std::uniform_real`
            // -> a GPU input tensor). The device then does only `logits/T + G ->
            // argmax`, one add over the reduction it was already going to run.
            Sampling::GumbelSoftmax { temperature } if logits.device().is_metal() => {
                self.gumbel_max_host_noise(logits, temperature)
            }
            Sampling::GumbelSoftmax { temperature } => {
                self.host_gumbel_rows(logits, temperature)
            }
            // Multinomial by inverse CDF: one host uniform uploaded, the
            // draw on the device, so the token never comes down here either.
            Sampling::All { temperature } if logits.device().is_metal() => {
                self.multinomial_on_device(logits, temperature)
            }
            _ => {
                let id = self.host_draw(logits)?;
                let dims = logits.dims();
                Tensor::new(id, logits.device())?.reshape(&dims[..dims.len().saturating_sub(1)])
            }
        }
    }

    /// Gumbel-max with the Gumbel variates generated host-side.
    ///
    /// `G = -log(-log u)`, `u ~ U(1e-9, 0.9999999)` (the tight clamp §11.2c/#345
    /// fixed -- `(1e-7, 0.999)` failed χ² at 2281, this passes at 280), drawn
    /// from the processor's own seeded host `StdRng` -- so `--seed` reaches the
    /// sampler directly, without going through `Device::set_seed` and the Metal
    /// RNG at all. The vector is uploaded with `Tensor::from_vec` and added to
    /// the logits; the argmax stays on the device, so the token never comes down
    /// (the ring's `to_cpu_blit` stays 0 at t > 0). Sampling from `softmax(z/T)`
    /// is `argmax(z/T + G)`, i.e. `argmax(z + T·G)` up to the shared `1/T`.
    fn gumbel_max_host_noise(&mut self, logits: &Tensor, temperature: f64) -> Result<Tensor> {
        let logits = logits.to_dtype(DType::F32)?;
        let n = logits.elem_count();
        let dev = logits.device();
        let noise = Tensor::from_vec(self.gumbel_variates(n), logits.shape(), dev)?;
        // argmax(z + T·G): scaling G by T is the temperature, and the shared
        // 1/T on z is monotone so it drops out of the argmax.
        (logits + noise.affine(temperature, 0.0)?)?.argmax(lloom_tensor::D::Minus1)
    }

    /// `n` Gumbel variates for one draw: `G = -log(-log u)`,
    /// `u ~ U(1e-9, 0.9999999)`.
    ///
    /// The stream is the position's when one is armed ([`Self::at_position`])
    /// and the running `rng`'s otherwise. The two differ in what a draw
    /// COSTS the next one: the running stream advances `n` variates per
    /// draw, so a draw's noise is a function of how many draws preceded it,
    /// and a speculative round that draws rows for drafts its verify then
    /// rejects shifts every later token's noise. A position's stream is
    /// derived from `(seed, position)` and shares no state, so a rejected
    /// row consumes nothing the next position needs — which is what makes a
    /// sampled verify reproduce a sampled non-speculative decode at one seed
    /// (`design/docs/speculation.md` §10.2z-viii).
    fn gumbel_variates(&mut self, n: usize) -> Vec<f32> {
        match self.position {
            Some(pos) => {
                let mut g = vec![0f32; n];
                Self::fill_gumbel(self.seed, pos, &mut g);
                g
            }
            None => (0..n)
                .map(|_| Self::gumbel_variate(&mut self.rng))
                .collect(),
        }
    }

    /// The multinomial draw on the logits' device: the uniform of this draw
    /// ([`Self::next_uniform`]) uploaded, one inverse-CDF dispatch, the id
    /// left on the device in the shape an argmax over the last dimension
    /// gives.
    fn multinomial_on_device(&mut self, logits: &Tensor, temperature: f64) -> Result<Tensor> {
        let u = self.next_uniform();
        let ut = Tensor::new(&[u], logits.device())?;
        let dims = logits.dims();
        let out = &dims[..dims.len().saturating_sub(1)];
        let rows: usize = out.iter().product();
        if rows != 1 {
            lloom_tensor::bail!(
                "multinomial draw: one row per draw, got logits {:?} (a window draws through \
                 `window_draw_device`)",
                dims
            );
        }
        logits.sample_inverse_cdf(temperature, &ut)?.reshape(out)
    }

    /// The uniform in `[0, 1)` one draw takes: the position's
    /// ([`Self::uniform_at`]) when one is armed, else the running stream's
    /// next.
    fn next_uniform(&mut self) -> f32 {
        use rand::Rng;
        match self.position {
            Some(pos) => Self::uniform_at(self.seed, pos),
            None => self.rng.random::<f32>(),
        }
    }

    /// The uniform of `(seed, pos)`'s stream: the first `f32` of the
    /// position's own generator, as [`Self::fill_gumbel`] derives its noise.
    pub fn uniform_at(seed: u64, pos: u64) -> f32 {
        use rand::{Rng, SeedableRng};
        rand::rngs::StdRng::seed_from_u64(Self::stream_key(seed, pos)).random::<f32>()
    }

    /// The per-position stream key: splitmix64 over `seed + pos·φ`.
    ///
    /// What is MEASURED is the key's two-halves property — a stream is a
    /// function of the seed and of the position, and neither may be dropped
    /// (`generation_tests.rs`, and a mutation that drops either is killed).
    /// The splitmix itself is **not** load-bearing and is not claimed to be:
    /// removing it leaves every assertion green, because `StdRng` is ChaCha12
    /// and already avalanches its seed — measured over 64 adjacent position
    /// pairs at 4096 variates, worst |corr| 0.0332 unmixed against 0.0399
    /// mixed, i.e. both at the noise floor. It is kept as the conventional
    /// derivation of a sub-stream from a counter, not as a fix for a defect
    /// that was demonstrated.
    fn stream_key(seed: u64, pos: u64) -> u64 {
        let mut z = seed.wrapping_add(pos.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The slot the host-sort arms read their probabilities from, made on
    /// first use and grown with the vocabulary. `None` off Metal, and on a
    /// device that refuses the Shared allocation — then the draw reads back
    /// as it always did.
    #[cfg(feature = "metal")]
    fn prs_slot_for<'a>(
        slot: &'a mut Option<lloom_tensor::metal_backend::HostSlot>,
        logits: &Tensor,
    ) -> Result<Option<&'a lloom_tensor::metal_backend::HostSlot>> {
        let Ok(m) = logits.device().as_metal_device() else {
            return Ok(None);
        };
        let n = logits.elem_count();
        if !matches!(slot, Some(s) if s.cap() >= n) {
            *slot = Some(lloom_tensor::metal_backend::HostSlot::new(
                m,
                n,
                DType::F32,
            )?);
        }
        Ok(slot.as_ref())
    }

    /// The inverse CDF over host probabilities (not necessarily normalised):
    /// the smallest index whose running sum, in index order and in f64,
    /// passes `u` times the total; a walk that rounding keeps from crossing
    /// takes the last index with mass. A zero entry is never drawn.
    fn pick(prs: &[f32], u: f32) -> u32 {
        let total: f64 = prs.iter().map(|&p| p as f64).sum();
        let target = u as f64 * total;
        let mut acc = 0f64;
        let mut last = 0u32;
        for (i, &p) in prs.iter().enumerate() {
            if p > 0.0 {
                last = i as u32;
            }
            acc += p as f64;
            if acc > target {
                return i as u32;
            }
        }
        last
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f_fed(logits, None::<fn(&mut [f32])>, |id| id)
    }

    /// [`Self::sample`] with `f` over the probabilities the draw reads, on
    /// the arms that draw from probabilities: plain temperature and the
    /// host-sort arms (argmax and gumbel-max read none, and never called it).
    /// Plain temperature then draws on the host — the same keyed uniform's
    /// inverse CDF over the probabilities `f` left — since the device draw
    /// reads the logits, never the probabilities.
    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        self.sample_f_fed(logits, Some(f), |id| id)
    }

    /// [`Self::sample`], with the step's fed token `feed(draw)` in place of
    /// the draw, as [`Self::sample_argmax_on_device_fed`].
    pub fn sample_fed(&mut self, logits: &Tensor, feed: impl FnOnce(u32) -> u32) -> Result<u32> {
        self.sample_f_fed(logits, None::<fn(&mut [f32])>, feed)
    }

    fn sample_f_fed(
        &mut self,
        logits: &Tensor,
        f: Option<impl FnOnce(&mut [f32])>,
        feed: impl FnOnce(u32) -> u32,
    ) -> Result<u32> {
        let temperature = self.temperature();
        let Some(stage) = self.stage.as_mut() else {
            let id = self.draw(logits, f)?;
            self.digest_step(logits)?;
            return Ok(feed(id));
        };
        let processed = stage.apply(logits, temperature)?;
        let drawn = self.draw(&processed, f)?;
        self.digest_step(logits)?;
        let id = feed(drawn);
        if let Some(stage) = self.stage.as_mut() {
            stage.record_id(id)?;
        }
        Ok(id)
    }

    fn draw(&mut self, logits: &Tensor, f: Option<impl FnOnce(&mut [f32])>) -> Result<u32> {
        // Greedy keeps its uncast device argmax. Host distributions widen
        // processed scores; Metal All and unfiltered Gumbel keep their route.
        if self.sampling == Sampling::ArgMax {
            return self.sample_argmax(logits.clone());
        }
        if let Sampling::GumbelSoftmax { temperature } = self.sampling {
            if logits.device().is_metal() {
                let draw = self.gumbel_max_host_noise(logits, temperature)?;
                return self.read_drawn(&draw);
            }
            if logits.elem_count() != logits.dim(lloom_tensor::D::Minus1)? {
                let draw = self.host_gumbel_rows(logits, temperature)?;
                return self.read_drawn(&draw);
            }
        }
        let (temperature, k, p) = self.sampling.distribution().expect("distribution arm");
        distribution::validate(temperature, p)?;
        if let Some(f) = f.filter(|_| !matches!(self.sampling, Sampling::FilteredGumbel { .. } | Sampling::GumbelSoftmax { .. })) {
            // Preserve the public probability callback's whole transformation,
            // after processors and before filters. This is explicitly host-side.
            let logits = logits.to_dtype(DType::F32)?;
            let scaled = (&logits / temperature)?;
            #[cfg(feature = "metal")]
            {
                if let Some(slot) = Self::prs_slot_for(&mut self.prs_slot, &scaled)? {
                    let out = slot.tensor(scaled.shape())?;
                    let _ = crate::nn::ops::softmax_last_dim_into(&scaled, out)?;
                    self.host.values = slot.read_committed::<f32>(scaled.elem_count())?;
                } else {
                    self.host.load(&crate::nn::ops::softmax_last_dim(&scaled)?)?;
                }
            }
            #[cfg(not(feature = "metal"))]
            self.host.load(&crate::nn::ops::softmax_last_dim(&scaled)?)?;
            f(&mut self.host.values);
            self.host.filter_probabilities(k, p);
            let u = self.next_uniform();
            return Ok(Self::pick(&self.host.values, u));
        }
        if matches!(self.sampling, Sampling::All { .. }) && logits.device().is_metal() {
            let draw = self.multinomial_on_device(logits, temperature)?;
            return self.read_drawn(&draw);
        }
        self.host_draw(logits)
    }

    fn host_draw(&mut self, logits: &Tensor) -> Result<u32> {
        let (temperature, k, p) = self.sampling.distribution().expect("host distribution arm");
        let vocab = logits.dim(lloom_tensor::D::Minus1)?;
        if vocab == 0 || logits.elem_count() != vocab {
            lloom_tensor::bail!("sampling: expected one nonempty row, got {:?}", logits.dims());
        }
        let logits = logits.to_dtype(DType::F32)?;
        #[cfg(feature = "metal")]
        {
            if let Some(slot) = Self::prs_slot_for(&mut self.prs_slot, &logits)? {
                slot.write_from(&logits)?;
                self.host.values = slot.read_committed::<f32>(vocab)?;
            } else {
                self.host.load(&logits)?;
            }
        }
        #[cfg(not(feature = "metal"))]
        self.host.load(&logits)?;
        self.host.prepare(temperature, k, p, matches!(self.sampling, Sampling::FilteredGumbel { .. } | Sampling::GumbelSoftmax { .. }))?;
        Ok(if matches!(self.sampling, Sampling::FilteredGumbel { .. } | Sampling::GumbelSoftmax { .. }) {
            match self.position {
                Some(pos) => {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(Self::stream_key(self.seed, pos));
                    self.host.gumbel(temperature, &mut rng)
                }
                None => self.host.gumbel(temperature, &mut self.rng),
            }
        } else {
            let u = self.next_uniform();
            self.host.draw(u)
        })
    }

    /// Device-return Gumbel accepts multiple rows. One call consumes the
    /// original flattened noise stream, unlike a window's per-position keys.
    fn host_gumbel_rows(&mut self, logits: &Tensor, temperature: f64) -> Result<Tensor> {
        distribution::validate(temperature, 1.0)?;
        let vocab = logits.dim(lloom_tensor::D::Minus1)?;
        if vocab == 0 { lloom_tensor::bail!("gumbel draw: empty vocabulary"); }
        let scores = logits.to_dtype(DType::F32)?;
        self.host.load(&scores)?;
        let mut values = std::mem::take(&mut self.host.values);
        let mut keyed = self.position.map(|p| rand::rngs::StdRng::seed_from_u64(Self::stream_key(self.seed, p)));
        let rng = keyed.as_mut().unwrap_or(&mut self.rng);
        let mut ids = Vec::with_capacity(values.len() / vocab);
        let result = (|| -> Result<()> {
            for row in values.chunks_mut(vocab) {
                self.host.prepare_row(row, temperature, 0, 1.0, true)?;
                ids.push(self.host.gumbel_row(row, temperature, rng));
            }
            Ok(())
        })();
        self.host.values = values;
        result?;
        let dims = logits.dims();
        Tensor::from_vec(ids, &dims[..dims.len() - 1], logits.device())
    }
}
