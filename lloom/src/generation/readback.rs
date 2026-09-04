//! The readback strategy: how a decode step feeds the next forward pass and how
//! the host learns EOS (`design/docs/decode-driver-design.md`, seam A).
//!
//! Three arms on one axis, and they differ only in where the round trip is:
//!
//! | arm | token comes to the host | EOS is learned by | per-token wait |
//! |---|---|---|---|
//! | [`Blocking`] | every step (`to_scalar`) | a host compare | `waitUntilCompleted`, inside the readback |
//! | [`Ring`] with [`EosRead::Blocking`] | never — device-resident slot | one 1-byte readback per step | `waitUntilCompleted`, inside the flag read |
//! | [`Ring`] with [`EosRead::Poll`] | never | polling the kernel-written counter | a poll, no Mach trap (§11.5n) |
//!
//! The loop ([`crate::nn::Session::decode`]) names the trait and never a
//! concrete arm, so the harness's `RESULT` line renders the strategy that ran
//! from [`ReadbackStrategy::arms`] rather than echoing a flag (§7.1a-i).
//!
//! # What the ring arms hold that the blocking arm does not
//!
//! The device-side history. A loop that never learns its tokens cannot hash
//! them, and reading each token back is the round trip the ring removes, so the
//! ids accumulate into one `[capacity]` `U32` slot by the same `slice_set` the
//! ring uses for its input and are read back **once**, by [`ReadbackStrategy::take_ids`],
//! after the loop (§11.5j). An EOS is written to the slot too — the write is
//! encoded before the host knows — and is retracted by [`ReadbackStrategy::eos_of_last`]
//! rolling the write cursor back one, so the next token overwrites it.

use crate::generation::poll::PollMechanism;
use crate::generation::token_ring::TokenRing;
use crate::generation::{LogitsProcessor, Sampling};
use lloom_tensor::{DType, Device, Result, Tensor};

/// Which readback arms a strategy is, for the `RESULT` line.
///
/// Rendered from the strategy that ran rather than from the flags that selected
/// it, which is the whole of §7.1a-i: a wrong arm compares equal to every run
/// of that arm, where an absent one compares equal to nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbackArms {
    /// The token stays on the device (`TokenRing=on`).
    pub token_ring: bool,
    /// The EOS flag is polled rather than read back (`NonblockingEos=on`).
    pub nonblocking_eos: bool,
}

/// The engagement counters a strategy reports, for the `RESULT` line.
///
/// A mechanism whose engagement nobody counts is one whose null reads as a
/// result (§2.4, #500): `ring_publishes` must reconcile against the ids the
/// loop returns plus the EOS tokens it retracted, or one of the two is wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackCounters {
    /// Tokens published into the ring's input slot.
    pub ring_publishes: u64,
    /// Publishes that turned out to be EOS and were retracted from the history.
    pub ring_eos_publishes: u64,
    /// Iterations of the EOS wait, across every step.
    pub poll_iters: u64,
    /// Steps on which the non-blocking poll gave up and took the blocking read.
    pub poll_fallbacks: u64,
}

/// How a decode step feeds the next forward pass and learns EOS.
///
/// The trait is the seam; the loop is generic over it and monomorphizes
/// (`CONTRIBUTING.md` §3.6: no `dyn` on the decode path).
///
/// # Protocol
///
/// One [`Self::sample_and_feed`] per step, then exactly one
/// [`Self::eos_of_last`] for that step before the next `sample_and_feed`. The
/// loop may encode the next forward pass between the two — that is the
/// lookahead the design's hot-path invariant asks for — iff
/// [`Self::admits_lookahead`] says the answer is not already known at feed
/// time.
pub trait ReadbackStrategy {
    /// The device the strategy's tensors live on; the loop's uploads and its
    /// periodic commit go through it.
    fn device(&self) -> &Device;

    /// Which arms this strategy renders as.
    fn arms(&self) -> ReadbackArms;

    /// Whether the loop may encode the next forward before [`Self::eos_of_last`]
    /// has answered. `false` when the answer is known at feed time (the
    /// blocking arm), where a lookahead would only add a wasted pass.
    fn admits_lookahead(&self) -> bool;

    /// Whether the loop owns the commit: `true` for a strategy whose wait
    /// needs the publish committed first and whose in-flight queue nothing
    /// else prunes (the ring arms), `false` for one whose own readback commits
    /// and waits every step (the blocking arm), where a loop-side commit would
    /// be a second command buffer per token the incumbent never cut.
    fn loop_commits(&self) -> bool;

    /// Refuse a sampling arm this strategy cannot serve without a host round
    /// trip it would not report (§7.1a-i's wrong-arm shape).
    fn check_sampler(&self, sampling: &Sampling) -> Result<()>;

    /// Sample this step's token from `logits` and return the next forward's
    /// `[1, 1]` input. The blocking arm reads the id to the host here; the ring
    /// arms leave it on the device and return the ring's own slot.
    fn sample_and_feed(&mut self, sampler: &mut LogitsProcessor, logits: &Tensor)
        -> Result<Tensor>;

    /// Was the token fed by the last [`Self::sample_and_feed`] an EOS? Waits
    /// by this arm's own mechanism until it knows; a `true` retracts that
    /// token from the ids [`Self::take_ids`] will return.
    fn eos_of_last(&mut self) -> Result<bool>;

    /// The ids published SO FAR and not yet drained, EOS excluded — the
    /// incremental, non-blocking drain for streaming. On the ring's poll arm it
    /// returns only what the GPU has published (it never waits and never drains
    /// past the visibility signal), so a caller mid-loop may get fewer than were
    /// fed; the rest arrive on a later call or from [`Self::drain_all`]. On the
    /// blocking arm every fed id is already visible, so it returns them all.
    fn take_ids(&mut self) -> Result<Vec<u32>>;

    /// EVERY id not yet drained, waiting for the GPU if it must — the final drain
    /// after the loop, where the contract is completeness rather than
    /// non-blocking. On the ring arms this synchronises up to the last fed slot
    /// (the poll arm cannot otherwise know the tail landed) and returns the whole
    /// remainder; on the blocking arm it is the same as [`Self::take_ids`], which
    /// already had everything. A caller streams this too, so nothing is lost
    /// between the last incremental drain and the end.
    fn drain_all(&mut self) -> Result<Vec<u32>>;

    /// The engagement counters so far.
    fn counters(&self) -> ReadbackCounters;
}

/// Today's arm: the sampled id comes to the host every step, the EOS test is a
/// host compare, and the 441 µs seam (§11.5a) lives here, honestly.
pub struct Blocking {
    device: Device,
    eos_ids: Vec<u32>,
    ids: Vec<u32>,
    last: Option<u32>,
}

impl Blocking {
    pub fn new(eos_ids: &[u32], device: &Device) -> Result<Self> {
        if eos_ids.is_empty() {
            lloom_tensor::bail!("Blocking: eos_ids is empty, so the loop would never terminate");
        }
        Ok(Self {
            device: device.clone(),
            eos_ids: eos_ids.to_vec(),
            ids: Vec::new(),
            last: None,
        })
    }
}

impl ReadbackStrategy for Blocking {
    fn device(&self) -> &Device {
        &self.device
    }

    fn arms(&self) -> ReadbackArms {
        ReadbackArms {
            token_ring: false,
            nonblocking_eos: false,
        }
    }

    fn admits_lookahead(&self) -> bool {
        false
    }

    fn loop_commits(&self) -> bool {
        false
    }

    fn check_sampler(&self, _sampling: &Sampling) -> Result<()> {
        // Every arm round-trips here, so every arm is honest.
        Ok(())
    }

    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        let id = sampler.sample(logits)?;
        self.ids.push(id);
        self.last = Some(id);
        Tensor::new(&[id], &self.device)?.unsqueeze(0)
    }

    fn eos_of_last(&mut self) -> Result<bool> {
        let Some(id) = self.last.take() else {
            lloom_tensor::bail!("Blocking::eos_of_last called before sample_and_feed");
        };
        if self.eos_ids.contains(&id) {
            self.ids.pop();
            return Ok(true);
        }
        Ok(false)
    }

    fn take_ids(&mut self) -> Result<Vec<u32>> {
        Ok(std::mem::take(&mut self.ids))
    }

    fn drain_all(&mut self) -> Result<Vec<u32>> {
        // The blocking arm's ids are already host-side and always complete, so
        // the final drain is the incremental one.
        Ok(std::mem::take(&mut self.ids))
    }

    fn counters(&self) -> ReadbackCounters {
        ReadbackCounters::default()
    }
}

/// How a [`Ring`] learns EOS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EosRead {
    /// Read the 1-byte flag through `to_vec1` — one blocking readback per step,
    /// narrowed from the token to a flag (§11.5j's arm).
    Blocking,
    /// Poll the kernel-written generation counter with the given mechanism,
    /// falling back to the blocking read after a bounded number of iterations
    /// (§11.5n's arm).
    Poll(PollMechanism),
}

/// The device-resident arms: the token never comes to the host, the next
/// forward reads the ring's own slot, and the ids accumulate device-side.
pub struct Ring {
    ring: TokenRing,
    eos: EosRead,
    /// Poll iterations before the non-blocking read gives up on a step.
    poll_budget: u64,
    /// The device-side ids, `[capacity]` `U32`.
    history: Tensor,
    capacity: usize,
    /// Tokens written into `history` so far, EOS retracted.
    fed: usize,
    /// Where the next `take_ids` starts reading.
    cursor: usize,
    /// The ordinal stamped on the last publish; 0 before any.
    generation: u32,
    counters: ReadbackCounters,
}

/// Iterations before the non-blocking poll takes the blocking read instead.
///
/// `poll_eos` returns `None` both for *not published yet* and for *this
/// buffer is not CPU-mapped*, and the second is an allocator property (§15.2
/// #12a) that #454 measures as unstable run to run — so an unbounded loop
/// would hang silently. Generous against a ~19 ms step: it converts a hang
/// into a counted fallback and times nothing.
pub const POLL_BUDGET: u64 = 50_000_000;

/// The id history slot count, chosen to stay CPU-mapped under **Private**
/// storage so the streaming drain reads it through a mapped pointer with no copy
/// and no blit (§15.2 #12: no blits on the decode path). Metal maps a
/// `StorageModePrivate` buffer iff it is **< 16384 bytes** (§15.2 #12a); at 4
/// bytes per `U32` that is **< 4096 slots**, so 4092 slots (16368 B) stays just
/// under the line while giving the host a wide window — it drains every commit,
/// far fewer than this behind. This is deliberately NOT Shared: Shared maps at
/// any size but its GPU-write coherency cost on the per-token `slice_set` is
/// unmeasured, and staying inside the Private mapping window keeps the write on
/// the same footing as every other decode buffer.
///
/// A generation longer than this many tokens is a real limit today —
/// `sample_and_feed` bails rather than wrapping — and the fix when it matters is
/// a genuine ring (wrap the write index, track a consumer lag) or a re-alloc,
/// not a bigger flat buffer. Left as a bounded slot count for now, loudly.
pub const RING_HISTORY_SLOTS: usize = 4092;

impl Ring {
    /// `capacity` bounds the ids one ring will accumulate across every decode
    /// call it serves; exceeding it is an error rather than a silent drop. The
    /// allocation is floored at [`RING_HISTORY_FLOOR`], so a caller that asks for
    /// less still gets a generous buffer.
    pub fn new(eos_ids: &[u32], device: &Device, capacity: usize, eos: EosRead) -> Result<Self> {
        let ring = TokenRing::new(0, eos_ids, device)?;
        // Sized to hold the whole generation, but at least a wide mapped window
        // and — for the zero-copy drain — ideally within [`RING_HISTORY_SLOTS`]
        // so a Private allocation stays CPU-mapped. `Tensor::zeros` uses the
        // default (Private on macOS) pool; below the mapping limit its
        // `contents()` is non-null and `take_ids` reads it through a pointer with
        // no copy. Above the limit (a > `RING_HISTORY_SLOTS`-token run) the pool
        // hands back an unmapped buffer and `take_ids` falls back to a `to_vec1`
        // snapshot, which is correct if slower — the honest degradation, not a
        // silent one. Kept Private rather than forced Shared: the per-token
        // `slice_set` write stays on the same footing as every other decode
        // buffer, whose Shared-vs-Private write cost is unmeasured.
        let capacity = capacity.max(RING_HISTORY_SLOTS);
        let history = Tensor::zeros(capacity, DType::U32, device)?;
        Ok(Self {
            ring,
            eos,
            history,
            capacity,
            fed: 0,
            cursor: 0,
            generation: 0,
            poll_budget: POLL_BUDGET,
            counters: ReadbackCounters::default(),
        })
    }

    /// Bound the non-blocking poll at `iterations` before it falls back to the
    /// blocking read. The default is [`POLL_BUDGET`]; a test with a model too
    /// small to reach the command-buffer cut wants a short one.
    pub fn with_poll_budget(mut self, iterations: u64) -> Self {
        self.poll_budget = iterations.max(1);
        self
    }

    /// The ring itself, for a caller that wants to observe its counters.
    pub fn token_ring(&self) -> &TokenRing {
        &self.ring
    }

    /// Whether the non-blocking poll reads through resolved addresses (#578).
    pub fn poll_is_mapped(&self) -> bool {
        self.ring.poll_is_mapped()
    }

    /// The generation ordinal of the last publish.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl ReadbackStrategy for Ring {
    fn device(&self) -> &Device {
        self.ring.input().device()
    }

    fn arms(&self) -> ReadbackArms {
        ReadbackArms {
            token_ring: true,
            nonblocking_eos: matches!(self.eos, EosRead::Poll(_)),
        }
    }

    fn admits_lookahead(&self) -> bool {
        true
    }

    fn loop_commits(&self) -> bool {
        true
    }

    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        // The ring is built on `sample_argmax_on_device` leaving the token on
        // the device; that holds for argmax and plain-temperature gumbel-max
        // and NOT for the top-{k,p} arms, which need a host sort (§11.2c). A
        // ring run on those would take the host path and report the ring arm.
        match sampling {
            Sampling::ArgMax | Sampling::All { .. } | Sampling::GumbelSoftmax { .. } => Ok(()),
            other => lloom_tensor::bail!(
                "the ring needs a device-resident sampler (argmax, or plain temperature via \
                 gumbel-max); {other:?} takes a host sort and cannot ride it (DESIGN.md §11.2c)"
            ),
        }
    }

    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        if self.fed >= self.capacity {
            lloom_tensor::bail!(
                "Ring: history capacity {} exhausted; size the ring for the whole generation",
                self.capacity
            );
        }
        let sampled = sampler.sample_argmax_on_device(logits)?;
        self.counters.ring_publishes += 1;

        // Order matters, and it is the streaming protocol. The history slot is
        // written FIRST, then the stamp — so a host that observes the generation
        // counter at `>= N` has, in queue order, `history[N-1]` written behind
        // it. `publish_step` already documents its own three writes as
        // "input, eos, stamp, stamp last"; the history write joins that ordering
        // *ahead* of the stamp, which is what lets the zero-copy `take_ids` read
        // `history[cursor..fed]` through the mapped pointer once the counter says
        // those steps landed — without a `to_vec1` sync. Writing the history
        // AFTER the stamp (the previous order) would let the counter run ahead of
        // the slot and the drain read an unwritten slot (§9.3's ordering, applied
        // to a second buffer).
        self.generation += 1;
        // Into the history at its own index, before the host knows whether it is
        // EOS; `eos_of_last` retracts by moving the cursor back.
        self.history.slice_set(&sampled.reshape(1)?, 0, self.fed)?;
        match self.eos {
            EosRead::Poll(_) => self.ring.publish_step(&sampled, Some(self.generation))?,
            EosRead::Blocking => self.ring.publish(&sampled)?,
        }
        self.fed += 1;
        Ok(self.ring.input().clone())
    }

    fn eos_of_last(&mut self) -> Result<bool> {
        let is_eos = match self.eos {
            EosRead::Blocking => {
                self.counters.poll_iters += 1;
                self.ring.eos_flag().to_vec1::<u8>()?[0] != 0
            }
            // A ring whose words are not CPU-mapped cannot be polled at all —
            // `poll_eos` declines every time — so the wait is the blocking read
            // from the start, and counted as the fallback it is.
            EosRead::Poll(_) if !self.ring.poll_is_mapped() => {
                self.counters.poll_iters += 1;
                self.counters.poll_fallbacks += 1;
                self.ring.eos_flag().to_vec1::<u8>()?[0] != 0
            }
            EosRead::Poll(poll) => {
                let generation = self.generation;
                let poll_budget = self.poll_budget;
                let mut spins: u64 = 0;
                // The watched word, resolved once; `Wfe` arms the exclusive
                // monitor on it and every other arm ignores it.
                //
                // SAFETY: the pointer is valid while `self.ring` is alive, and
                // the ring outlives this loop.
                let watched = unsafe { self.ring.step_addr() };
                loop {
                    self.counters.poll_iters += 1;
                    match self.ring.poll_eos(generation) {
                        Some(eos) => break eos,
                        None => {
                            spins += 1;
                            if spins > poll_budget {
                                // Counted, not silent: a fallback nobody
                                // counts is a null that reads as a result.
                                self.counters.poll_fallbacks += 1;
                                break self.ring.eos_flag().to_vec1::<u8>()?[0] != 0;
                            }
                            match watched {
                                // Wait for the counter to CHANGE from what the
                                // poll just saw; the compare skips the wait
                                // when the word has already moved.
                                Some(p) => {
                                    // SAFETY: as above; the word may be
                                    // concurrently written, which is the point.
                                    let seen = unsafe { p.read_volatile() };
                                    unsafe { poll.wait_on(p, seen) }
                                }
                                None => poll.wait(),
                            }
                        }
                    }
                }
            }
        };
        if is_eos {
            self.counters.ring_eos_publishes += 1;
            self.fed -= 1;
        }
        Ok(is_eos)
    }

    fn take_ids(&mut self) -> Result<Vec<u32>> {
        if self.cursor == self.fed {
            return Ok(Vec::new());
        }
        let start = self.cursor;

        // How far it is SAFE to drain. On the stamped (poll) arm the loop may be
        // ahead of the GPU (that is the lookahead), so `fed` counts slots the
        // host has *enqueued* — not all of them are written yet. The published
        // generation is the visibility signal (`published_generation` reads the
        // GPU-written counter, which `sample_and_feed` writes AFTER the history
        // slot): a value `g` means `history[0..g]` are written, so draining past
        // it would read an unwritten slot as the buffer's init value — silent
        // corruption (§3.5). Clamp to it. On the blocking arm there is no stamp,
        // but its own per-step EOS `to_vec1` already synchronised up to `fed`, so
        // every enqueued slot is visible and the full range is safe.
        let end = match self.eos {
            EosRead::Poll(_) => match self.ring.published_generation() {
                // `g` counts published steps; each retracted EOS decremented
                // `fed` but not the counter, so clamp both ways.
                Some(g) => self.fed.min(g as usize),
                // Unmapped counter (no visibility signal): fall back to the
                // synchronising snapshot below, which is correct by force.
                None => 0,
            },
            EosRead::Blocking => self.fed,
        };

        // Fast path: read `[start..end]` straight from the mapped buffer — no copy
        // of the whole history and, crucially, **no blit and no GPU drain**
        // (`peek_scalar_ptr` "reaches nothing that waits on or orders against the
        // GPU"). This is what keeps streaming non-blocking: called once per
        // commit, it must not become a `waitUntilCompleted` per commit, which a
        // `to_vec1` is (§15.2 #12 forbids that blit on the decode path). `end`
        // bounds it to published slots; `read_volatile` because the GPU may still
        // be writing *later* slots into the same buffer.
        //
        // SAFETY: `peek_scalar_ptr(start)` returns a pointer valid for
        // `[start, count)` (it checks `start < count`); `history` is written in
        // place and never reallocated for this ring's lifetime, so the mapping
        // stays valid; `end <= fed <= capacity = count`, so every read is in
        // bounds.
        if end > start {
            if let Some(base) = unsafe { self.history.peek_scalar_ptr::<u32>(start) } {
                let n = end - start;
                let mut ids = Vec::with_capacity(n);
                for i in 0..n {
                    ids.push(unsafe { base.add(i).read_volatile() });
                }
                self.cursor = end;
                return Ok(ids);
            }
        }

        // Snapshot / fallback path. Taken when: the counter is unmapped (`end`
        // forced to 0 above), or the history buffer itself is not CPU-mapped (a
        // > `RING_HISTORY_SLOTS`-token run whose Private allocation exceeds the
        // 16 KiB window). A `to_vec1` of the live prefix synchronises up to `fed`
        // and is correct if a copy — the honest degradation. It is also the path
        // the one-shot final drain and the digest gate take, where a contiguous
        // `Vec` is the actual need. Drains the FULL `[start..fed]` because the
        // `to_vec1` made all of it visible.
        let n = self.fed - start;
        let all = self.history.narrow(0, start, n)?.to_vec1::<u32>()?;
        self.cursor = self.fed;
        Ok(all)
    }

    fn drain_all(&mut self) -> Result<Vec<u32>> {
        if self.cursor == self.fed {
            return Ok(Vec::new());
        }
        // The final drain: completeness over non-blocking. `to_vec1` over the
        // live `[cursor..fed]` **synchronises** up to the last enqueued slot,
        // which is exactly what the poll arm needs here — it cannot otherwise
        // know the tail landed, and its incremental `take_ids` deliberately
        // refuses to read past the published counter. One `narrow` + `to_vec1`
        // is the honest cost of learning the whole tail once, after the clock.
        let start = self.cursor;
        let n = self.fed - start;
        let ids = self.history.narrow(0, start, n)?.to_vec1::<u32>()?;
        self.cursor = self.fed;
        Ok(ids)
    }

    fn counters(&self) -> ReadbackCounters {
        self.counters
    }
}

/// How often the loop commits and prunes the in-flight queue
/// (`design/docs/decode-driver-design.md`, "where the CommitBudget's cadence
/// comes from").
///
/// `steps(1)` is the incumbent ring arm's per-token commit, rendered
/// `per-token`; `steps(N)` commits and prunes every N steps. Under a loop that
/// waits before encoding the next step the cadence is moot — the publish must
/// be committed before the host can wait on it, and the loop commits there
/// regardless — so the budget only becomes real with lookahead on. The
/// `commits` counter says which happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitBudget {
    steps: usize,
}

impl CommitBudget {
    /// Commit and prune every step — the incumbent ring arm.
    pub fn per_token() -> Self {
        Self { steps: 1 }
    }

    /// Commit and prune every `n` steps. `n = 0` is refused: a budget that
    /// never prunes lets `Commands::in_flight` grow by one command buffer per
    /// commit for the length of the generation (§11.5h).
    pub fn steps(n: usize) -> Result<Self> {
        if n == 0 {
            lloom_tensor::bail!("CommitBudget::steps(0): the queue would never be pruned");
        }
        Ok(Self { steps: n })
    }

    /// Whether `since` steps since the last commit are enough to commit again.
    pub fn due(&self, since: usize) -> bool {
        since >= self.steps
    }

    /// Steps per commit.
    pub fn period(&self) -> usize {
        self.steps
    }

    /// The arm's name for a `RESULT` line: `per-token` or `steps:<N>`, the
    /// period carried verbatim (§13.5m: an arm whose period is dropped pools
    /// with every other period of itself).
    pub fn arm(&self) -> String {
        if self.steps == 1 {
            "per-token".to_string()
        } else {
            format!("steps:{}", self.steps)
        }
    }
}

impl std::str::FromStr for CommitBudget {
    type Err = String;

    /// Parses `per-token` and `steps:<N>`. `steps:1` is refused in favour of
    /// its own name, so one arm has one spelling.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "per-token" => Ok(Self::per_token()),
            other => match other.strip_prefix("steps:") {
                Some(n) => {
                    let n: usize = n
                        .parse()
                        .map_err(|e| format!("commit budget {other:?}: bad step count: {e}"))?;
                    if n == 1 {
                        return Err(format!(
                            "commit budget {other:?}: a one-step budget is spelled `per-token`"
                        ));
                    }
                    Self::steps(n).map_err(|e| e.to_string())
                }
                None => Err(format!(
                    "unknown commit budget {other:?} (expected per-token or steps:<N>)"
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn budget_parses_and_renders_its_period() {
        assert_eq!(
            CommitBudget::from_str("per-token").unwrap().arm(),
            "per-token"
        );
        assert_eq!(CommitBudget::from_str("steps:8").unwrap().arm(), "steps:8");
        assert_eq!(CommitBudget::from_str("steps:8").unwrap().period(), 8);
        assert!(CommitBudget::from_str("steps:0").is_err());
        assert!(CommitBudget::from_str("steps:1").is_err());
        assert!(CommitBudget::from_str("steps:").is_err());
        assert!(CommitBudget::from_str("every-token").is_err());
        assert!(CommitBudget::from_str("").is_err());
    }

    #[test]
    fn budget_is_due_at_its_period() {
        let b = CommitBudget::from_str("steps:4").unwrap();
        assert!(!b.due(3));
        assert!(b.due(4));
        assert!(b.due(5));
        assert!(CommitBudget::per_token().due(1));
        assert!(!CommitBudget::per_token().due(0));
    }
}
