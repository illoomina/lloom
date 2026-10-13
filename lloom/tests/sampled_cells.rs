//! The sampled identity's cells, shared by the drafters' test files: a
//! request's sampler (an arm, processors, a thinking budget), the K=1 run of
//! it and the speculative run of it, so each target's file asserts
//! `speculative = K=1` over the same set (#943).

use lloom::generation::{
    Blocking, LogitsProcessor, Processors, ReadbackStrategy, Sampling, ThinkingBudget,
};
use lloom::nn::model::CausalLM;
use lloom::nn::speculator::{Probes, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{Profile, RewindProof, Ring, Shaped};
use lloom_tensor::{Device, Result};

/// A speculative run, labelled: its name, then the ids, whether it hit EOS,
/// and the rounds that kept a draft (`spec`'s output).
pub type LabelledRun = (String, (Vec<u32>, bool, usize));

/// One request's sampler.
#[derive(Clone, Debug)]
pub struct Cell {
    pub sampling: Sampling,
    pub processors: Processors,
    pub budget: Option<ThinkingBudget>,
}

impl Cell {
    fn arm(sampling: Sampling) -> Self {
        Cell {
            sampling,
            processors: Processors::default(),
            budget: None,
        }
    }

    pub fn sampler(&self, seed: u64) -> LogitsProcessor {
        LogitsProcessor::from_sampling(seed, self.sampling.clone())
            .with_processors(self.processors.clone())
    }
}

/// The cells every drafter is held to: the multinomial default, gumbel-max,
/// top-k, top-p, processors and a thinking budget (closed by `close`, a
/// vocabulary id, its tail `[transition, close]`).
pub fn cells(transition: u32, close: u32) -> Vec<Cell> {
    let procs = Processors {
        repetition_penalty: 1.3,
        presence_penalty: 0.3,
        frequency_penalty: 0.4,
        min_p: 0.05,
        logit_bias: vec![(transition, 1.0)],
    };
    let budget =
        ThinkingBudget::new(5, vec![close], vec![transition, close]).expect("a valid budget");
    vec![
        Cell::arm(Sampling::All { temperature: 1.0 }),
        Cell::arm(Sampling::GumbelSoftmax { temperature: 0.9 }),
        Cell::arm(Sampling::TopK {
            k: 16,
            temperature: 1.0,
        }),
        Cell::arm(Sampling::TopP {
            p: 0.9,
            temperature: 1.1,
        }),
        Cell {
            processors: procs.clone(),
            ..Cell::arm(Sampling::All { temperature: 1.0 })
        },
        Cell {
            processors: procs,
            ..Cell::arm(Sampling::ArgMax)
        },
        Cell {
            budget: Some(budget),
            ..Cell::arm(Sampling::All { temperature: 1.0 })
        },
    ]
}

/// The K=1 run on the blocking arm, the budget the arm's.
#[allow(clippy::too_many_arguments)]
pub fn k1<M: CausalLM>(
    model: &M,
    cache: M::Cache,
    dev: &Device,
    eos: &[u32],
    cell: &Cell,
    seed: u64,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool)> {
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(eos, dev)?;
    rb.set_thinking_budget(cell.budget.as_ref())?;
    let mut sampler = cell.sampler(seed);
    sampler.observe_prompt(prompt);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &mut |_: &[u32]| Ok(()),
    )?;
    Ok((out.ids, out.eos))
}

/// The speculative run with `spec` drafting: the ids, EOS, and the rounds
/// that kept a draft.
#[allow(clippy::too_many_arguments)]
pub fn spec<M, S>(
    model: &M,
    cache: M::Cache,
    dev: &Device,
    eos: &[u32],
    cell: &Cell,
    seed: u64,
    prompt: &[u32],
    max: usize,
    spec: &mut S,
) -> Result<(Vec<u32>, bool, usize)>
where
    S: Speculator,
    M: Probes<S::Probe> + Shaped,
    M::Profile: Profile<Rewind = Ring>,
{
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(eos, dev)?;
    let mut sampler = cell.sampler(seed).with_window_budget(cell.budget.as_ref());
    sampler.observe_prompt(prompt);
    let mut streamed = Vec::new();
    let out = session.decode_speculative(
        RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        eos,
        &mut |b: &[u32]| {
            streamed.extend_from_slice(b);
            Ok(())
        },
    )?;
    assert_eq!(out.ids, streamed, "the stream must equal the returned ids");
    let kept = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
    Ok((out.ids, out.eos, kept))
}
