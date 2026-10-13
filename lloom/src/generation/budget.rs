//! The thinking budget (vLLM's `thinking_token_budget`): a cap on the tokens a
//! run generates inside the thinking block its prompt left open. When the cap
//! is reached with the block still open, the next steps feed the budget's
//! forced tail — an optional transition, then the end marker — in place of
//! the draw, and the model continues after it.
//!
//! The block closes on the end marker or on a tool-call start: a model that
//! calls a tool has left its reasoning, and a forced end inside the call's
//! arguments corrupts the call. Which ids those are is the family's, read
//! from its chat template; nothing here names a marker. The budget counts the
//! block's tokens from its start marker, the prompt's own included, as vLLM
//! does.
//!
//! What a step feeds is a function of the draws alone, so the blocking arm
//! computes it on the host ([`HostForcing`]) and the ring arms on the device
//! ([`DeviceForcing`]): a closed flag raised by each fed token and, inside
//! the forced window, a select between the forced id and the draw. Neither
//! reads anything back (`docs/decode-driver-design.md`, Stage 7's budgets).
//! [`ThinkingBudget::report`] applies the same rule to the ids a run returned.

use lloom_tensor::{DType, Device, Result, Tensor};

/// A thinking budget, resolved against a family's markers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingBudget {
    budget: usize,
    /// Tokens of the block the prompt already holds, counted against the
    /// budget (vLLM's count: everything after the block's start marker).
    prior: usize,
    close_ids: Vec<u32>,
    forced: Vec<u32>,
}

/// What a run did inside the thinking block, from the ids it returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingReport {
    /// Tokens inside the block, as the budget counts them: the prompt's,
    /// then every generated id before the one that closed it (every id when
    /// nothing did). A forced transition counts; the end marker does not.
    pub thought: usize,
    /// The block was still open at the budget and the forced tail began.
    pub fired: bool,
    /// The id that closed the block, if one did.
    pub closed_by: Option<u32>,
}

impl ThinkingBudget {
    /// `budget` tokens inside the block; `close_ids` end it (the end marker
    /// and the tool-call starts); `forced` is the tail fed when the budget is
    /// spent, and must end with a closing id and hold none before it, so the
    /// whole tail is fed once it begins.
    pub fn new(budget: usize, close_ids: Vec<u32>, forced: Vec<u32>) -> Result<Self> {
        let Some((&last, transition)) = forced.split_last() else {
            lloom_tensor::bail!("thinking budget: the forced tail is empty");
        };
        if !close_ids.contains(&last) {
            lloom_tensor::bail!(
                "thinking budget: the forced tail ends with {last}, which does not close the \
                 block ({close_ids:?})"
            );
        }
        if let Some(id) = transition.iter().find(|id| close_ids.contains(id)) {
            lloom_tensor::bail!(
                "thinking budget: the transition holds {id}, which closes the block before the \
                 end marker"
            );
        }
        Ok(Self {
            budget,
            prior: 0,
            close_ids,
            forced,
        })
    }

    /// Count `prior` tokens the prompt already holds inside the block
    /// against the budget.
    pub fn with_prior(mut self, prior: usize) -> Self {
        self.prior = prior;
        self
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn prior(&self) -> usize {
        self.prior
    }

    /// The generated tokens the block may take before the tail is fed.
    fn free(&self) -> usize {
        self.budget.saturating_sub(self.prior)
    }

    pub fn close_ids(&self) -> &[u32] {
        &self.close_ids
    }

    pub fn forced(&self) -> &[u32] {
        &self.forced
    }

    /// The first step whose fed token no longer depends on the budget.
    fn window_end(&self) -> usize {
        self.free() + self.forced.len()
    }

    /// The report for a run that returned `ids`.
    pub fn report(&self, ids: &[u32]) -> ThinkingReport {
        let first_close = ids.iter().position(|id| self.close_ids.contains(id));
        let n = self.free();
        ThinkingReport {
            thought: self.prior + first_close.unwrap_or(ids.len()),
            fired: ids.len() > n && first_close.is_none_or(|p| p >= n),
            closed_by: first_close.map(|p| ids[p]),
        }
    }
}

/// The blocking arm's budget: the rule on host ids.
#[derive(Clone)]
pub(crate) struct HostForcing {
    cfg: ThinkingBudget,
    step: usize,
    closed: bool,
}

/// A [`HostForcing`]'s position before a draw, to take the draw back.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ForcingMark {
    step: usize,
    closed: bool,
}

impl HostForcing {
    pub(crate) fn mark(&self) -> ForcingMark {
        ForcingMark {
            step: self.step,
            closed: self.closed,
        }
    }

    pub(crate) fn rewind(&mut self, m: ForcingMark) {
        self.step = m.step;
        self.closed = m.closed;
    }

    pub(crate) fn new(cfg: ThinkingBudget) -> Self {
        Self {
            cfg,
            step: 0,
            closed: false,
        }
    }

    /// The token this step feeds, given its draw.
    pub(crate) fn feed(&mut self, drawn: u32) -> u32 {
        let s = self.step;
        self.step += 1;
        if s >= self.cfg.window_end() {
            return drawn;
        }
        let n = self.cfg.free();
        let fed = if !self.closed && s >= n {
            self.cfg.forced[s - n]
        } else {
            drawn
        };
        self.closed |= self.cfg.close_ids.contains(&fed);
        fed
    }
}

/// The ring arms' budget: the same rule as [`HostForcing`], on device
/// tensors, so the token never comes to the host. Three dispatches a step
/// before the budget (the closed flag's update), two inside the forced window
/// (the select, and the update where a later step reads it), none after.
pub(crate) struct DeviceForcing {
    cfg: ThinkingBudget,
    step: usize,
    /// `[1]` `U8`: 1 once a fed token closed the block.
    closed: Tensor,
    /// `[1, m]` `U32`: the ids that close it.
    close: Tensor,
    /// `[k]` `U32`: the forced tail.
    forced: Tensor,
}

impl DeviceForcing {
    pub(crate) fn new(cfg: ThinkingBudget, device: &Device) -> Result<Self> {
        let m = cfg.close_ids.len();
        Ok(Self {
            closed: Tensor::from_vec(vec![0u8], 1, device)?,
            close: Tensor::from_vec(cfg.close_ids.clone(), (1, m), device)?,
            forced: Tensor::new(cfg.forced.as_slice(), device)?,
            step: 0,
            cfg,
        })
    }

    /// Whether this step's fed token can still differ from its draw.
    pub(crate) fn live(&self) -> bool {
        self.step < self.cfg.window_end()
    }

    /// The `[1]` `U32` token this step feeds, given its `[1]` `U32` draw.
    pub(crate) fn feed(&mut self, drawn: &Tensor) -> Result<Tensor> {
        let s = self.step;
        self.step += 1;
        let drawn = drawn.reshape(1)?;
        if s >= self.cfg.window_end() {
            return Ok(drawn);
        }
        let n = self.cfg.free();
        let fed = if s >= n {
            let forced = self.forced.narrow(0, s - n, 1)?;
            self.closed.where_cond(&drawn, &forced)?
        } else {
            drawn
        };
        if s + 1 < self.cfg.window_end() {
            let hit = fed
                .reshape((1, 1))?
                .broadcast_eq(&self.close)?
                .max_keepdim(1)?
                .reshape(1)?;
            self.closed = self.closed.maximum(&hit)?;
        }
        debug_assert_eq!(fed.dtype(), DType::U32);
        Ok(fed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const END: u32 = 9;
    const TOOL: u32 = 7;

    fn budget(n: usize, forced: &[u32]) -> ThinkingBudget {
        ThinkingBudget::new(n, vec![END, TOOL], forced.to_vec()).unwrap()
    }

    /// The fed ids of a run whose draws are `draws`, on the host rule.
    fn host(b: &ThinkingBudget, draws: &[u32]) -> Vec<u32> {
        let mut f = HostForcing::new(b.clone());
        draws.iter().map(|&d| f.feed(d)).collect()
    }

    /// The same on the device rule, one step at a time as the ring feeds.
    fn device(b: &ThinkingBudget, draws: &[u32], dev: &Device) -> Vec<u32> {
        let mut f = DeviceForcing::new(b.clone(), dev).unwrap();
        draws
            .iter()
            .map(|&d| {
                let t = Tensor::new(&[d], dev).unwrap();
                f.feed(&t).unwrap().to_vec1::<u32>().unwrap()[0]
            })
            .collect()
    }

    fn devices() -> Vec<Device> {
        let mut v = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        v.push(Device::new_metal(0).expect("a Metal device"));
        v
    }

    /// Open at the budget: the tail replaces the draws, then the draws
    /// resume. Closed before it by the end marker or by a tool-call start:
    /// the draws are fed untouched, the call's arguments included. A budget
    /// of zero ends the block at the first step. Host and device agree on
    /// every case, and the report reads each run back.
    #[test]
    fn the_tail_replaces_the_draws_only_while_the_block_is_open() {
        let tail = [3, 4, END];
        let cases: [(&str, usize, Vec<u32>, Vec<u32>, ThinkingReport); 5] = [
            (
                "fires",
                2,
                vec![1, 2, 1, 2, 1, 2, 1],
                vec![1, 2, 3, 4, END, 2, 1],
                ThinkingReport {
                    thought: 4,
                    fired: true,
                    closed_by: Some(END),
                },
            ),
            (
                "closed by the end marker",
                3,
                vec![1, END, 1, 2, 1, 2, 1],
                vec![1, END, 1, 2, 1, 2, 1],
                ThinkingReport {
                    thought: 1,
                    fired: false,
                    closed_by: Some(END),
                },
            ),
            (
                "closed by a tool call",
                3,
                vec![1, TOOL, 5, 5, 5, 5, 5],
                vec![1, TOOL, 5, 5, 5, 5, 5],
                ThinkingReport {
                    thought: 1,
                    fired: false,
                    closed_by: Some(TOOL),
                },
            ),
            (
                "zero",
                0,
                vec![1, 2, 1, 2],
                vec![3, 4, END, 2],
                ThinkingReport {
                    thought: 2,
                    fired: true,
                    closed_by: Some(END),
                },
            ),
            (
                "never reached",
                9,
                vec![1, 2, 1],
                vec![1, 2, 1],
                ThinkingReport {
                    thought: 3,
                    fired: false,
                    closed_by: None,
                },
            ),
        ];
        for (name, n, draws, want, report) in cases {
            let b = budget(n, &tail);
            assert_eq!(host(&b, &draws), want, "{name}: host");
            for dev in devices() {
                assert_eq!(device(&b, &draws, &dev), want, "{name}: {dev:?}");
            }
            assert_eq!(b.report(&want), report, "{name}: report");
        }
    }

    /// The edge: a closing id fed on the last step inside the budget ends the
    /// block and no tail follows; the step at the budget is the tail's.
    #[test]
    fn the_budget_edge_is_exact() {
        let b = budget(2, &[END]);
        assert_eq!(host(&b, &[1, END, 1, 1]), vec![1, END, 1, 1]);
        assert_eq!(host(&b, &[1, 1, END, 1]), vec![1, 1, END, 1]);
        assert_eq!(host(&b, &[1, 1, 1, 1]), vec![1, 1, END, 1]);
        for dev in devices() {
            assert_eq!(device(&b, &[1, END, 1, 1], &dev), vec![1, END, 1, 1]);
            assert_eq!(device(&b, &[1, 1, 1, 1], &dev), vec![1, 1, END, 1]);
        }
    }

    /// Tokens the prompt already holds inside the block count against the
    /// budget: a budget of 3 with 1 in the prompt forces at the second
    /// generated step, and the report counts the prompt's token; a prompt
    /// already over the budget forces at once.
    #[test]
    fn the_prompts_thinking_counts_against_the_budget() {
        let b = budget(3, &[END]).with_prior(1);
        assert_eq!(host(&b, &[1, 1, 1, 1]), vec![1, 1, END, 1]);
        for dev in devices() {
            assert_eq!(device(&b, &[1, 1, 1, 1], &dev), vec![1, 1, END, 1]);
        }
        assert_eq!(
            b.report(&[1, 1, END, 1]),
            ThinkingReport {
                thought: 3,
                fired: true,
                closed_by: Some(END)
            }
        );
        let over = budget(3, &[END]).with_prior(5);
        assert_eq!(host(&over, &[1, 1]), vec![END, 1]);
    }

    /// A tail that does not end the block, or ends it early, is refused.
    #[test]
    fn a_tail_must_end_with_the_marker_and_only_there() {
        let close = vec![END, TOOL];
        assert!(ThinkingBudget::new(1, close.clone(), vec![]).is_err());
        assert!(ThinkingBudget::new(1, close.clone(), vec![3, 4]).is_err());
        assert!(ThinkingBudget::new(1, close.clone(), vec![TOOL, END]).is_err());
        assert!(ThinkingBudget::new(1, close, vec![3, END]).is_ok());
    }
}
