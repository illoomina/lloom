//! Request-local native arithmetic at the existing blocking ID boundary.
//! Unlike NativeRing, no sampled id is consumed before completed validation.
use super::*;
use crate::generation::native::PreparedNativeSampler;
use lloom_tensor::metal_backend::sampling::SampleReadback;

pub struct NativeBlocking {
    blocking: Blocking,
    native: PreparedNativeSampler,
    snapshot: SampleReadback,
    poisoned: bool,
}

impl NativeBlocking {
    /// All sampler/readback ownership is prepared before session mutation.
    /// One slot suffices: each scalar sample completes before the next draw.
    /// This quotes logical payload, not complete physical allocator admission.
    pub fn new(blocking: Blocking, vocab: usize) -> Result<Self> {
        if let Some(forcing) = &blocking.forcing {
            forcing.check_vocab(vocab)?;
        }
        let native = PreparedNativeSampler::new(&blocking.device, 1, vocab, 1)?;
        let snapshot = SampleReadback::new(blocking.device.as_metal_device()?, 1)?;
        Ok(Self {
            blocking,
            native,
            snapshot,
            poisoned: false,
        })
    }

    pub fn take_logprobs(&mut self) -> Option<Logprobs> {
        self.blocking.take_logprobs()
    }

    fn check_live(&self) -> Result<()> {
        if self.poisoned {
            lloom_tensor::bail!("native blocking: prior sample did not complete");
        }
        Ok(())
    }
}

impl ReadbackStrategy for NativeBlocking {
    fn device(&self) -> &Device {
        self.blocking.device()
    }
    fn arms(&self) -> ReadbackArms {
        self.blocking.arms()
    }
    fn admits_lookahead(&self) -> bool {
        false
    }
    fn loop_commits(&self) -> bool {
        false
    }
    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        self.native.check_sampling_kind(sampling)
    }
    fn check_request(&self, sampler: &LogitsProcessor) -> Result<()> {
        self.check_live()?;
        // Same blocking capability table: raw digest/logprobs stay supported.
        self.blocking.check_request(sampler)?;
        self.check_sampler(sampler.sampling())
    }
    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        self.check_live()?;
        self.poisoned = true;
        // No history or forcing mutation before the entire completed sample
        // passes status/id/error checks. The guard/storage retain queue custody
        // if encoding, waiting, validation or subsequent host work fails.
        let token = self.native.sample_for_host_readback(sampler, logits)?;
        token.snapshot(&mut self.snapshot)?;
        drop(token);
        // The snapshot owns its exact copy recording; callback-prefix progress
        // is not required to validate these completed bytes.
        let checked = self.snapshot.read_committed()?.ok_or_else(|| {
            lloom_tensor::Error::msg("native blocking: snapshot incomplete after blocking wait")
        })?;
        let [drawn] = checked.ids() else {
            lloom_tensor::bail!("native blocking: expected one checked id");
        };
        let drawn = *drawn;
        self.blocking.forcing_before = self.blocking.forcing.as_ref().map(HostForcing::mark);
        let forcing = &mut self.blocking.forcing;
        let id = sampler.accept_drawn(logits, drawn, |drawn| match forcing {
            Some(forcing) => forcing.feed(drawn),
            None => drawn,
        })?;
        let input = self.blocking.finish_fed(logits, id)?;
        self.poisoned = false;
        Ok(input)
    }
    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        self.check_live()?;
        if let Some(budget) = budget {
            HostForcing::new(budget.clone()).check_vocab(self.native.vocab())?;
        }
        self.blocking.set_thinking_budget(budget)
    }
    fn fed_host_ids(&self) -> Option<&[u32]> {
        self.blocking.fed_host_ids()
    }
    fn retracts_draw(&self) -> bool {
        self.blocking.retracts_draw()
    }
    fn retract_draw(&mut self) -> Result<()> {
        self.check_live()?;
        self.blocking.retract_draw()
    }
    fn eos_of_last(&mut self) -> Result<bool> {
        self.check_live()?;
        self.blocking.eos_of_last()
    }
    fn take_ids(&mut self) -> Result<Vec<u32>> {
        self.check_live()?;
        self.blocking.take_ids()
    }
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.check_live()?;
        self.blocking.take_ids_into(out)
    }
    fn drain_all(&mut self) -> Result<Vec<u32>> {
        self.check_live()?;
        self.blocking.drain_all()
    }
    fn eos_read_arm(&self) -> &'static str {
        "host"
    }
    fn counters(&self) -> ReadbackCounters {
        self.blocking.counters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::Processors;

    fn device() -> Result<Device> {
        lloom_tensor::isolation::claim("native blocking shared arithmetic controls");
        let mut tuning = lloom_tensor::tuning::Tuning::default();
        tuning.from_env().expect("valid test environment");
        tuning.contract.profile = "deterministic".into();
        lloom_tensor::tuning::install(tuning).expect("install before first device");
        let device = Device::new_metal(0)?;
        println!(
            "NATIVE_BLOCKING_SETUP actual_api={}",
            device.as_metal_device()?.commands().actual_command_api()
        );
        Ok(device)
    }

    #[test]
    fn unfiltered_native_blocking_matches_cpu_contract_and_raw_diagnostics() -> Result<()> {
        let dev = device()?;
        println!(
            "NATIVE_BLOCKING_SETUP actual_api={}",
            dev.as_metal_device()?.commands().actual_command_api()
        );
        for sampling in [
            Sampling::All { temperature: 0.7 },
            Sampling::GumbelSoftmax { temperature: 0.7 },
        ] {
            for processors in [false, true] {
                let mut cpu = Blocking::new(&[99], &Device::Cpu)?.with_logprobs(3, true);
                let mut native =
                    NativeBlocking::new(Blocking::new(&[99], &dev)?.with_logprobs(3, true), 3)?;
                let make = || {
                    LogitsProcessor::from_sampling(43, sampling.clone())
                        .with_processors(if processors {
                            Processors {
                                presence_penalty: 0.5,
                                frequency_penalty: 0.25,
                                ..Default::default()
                            }
                        } else {
                            Processors::default()
                        })
                        .with_logits_digest(true)
                };
                let (mut host, mut metal) = (make(), make());
                host.observe_prompt(&[1]);
                metal.observe_prompt(&[1]);
                for (position, row) in [[0f32, 0.3, -0.2], [1e30, 1e30, 1e30], [0., -1., -3.]]
                    .into_iter()
                    .cycle()
                    .take(48)
                    .enumerate()
                {
                    host.at_position(Some(position as u64));
                    metal.at_position(Some(position as u64));
                    let expected =
                        cpu.sample_and_feed(&mut host, &Tensor::new(&row, &Device::Cpu)?)?;
                    let actual = native.sample_and_feed(&mut metal, &Tensor::new(&row, &dev)?)?;
                    assert_eq!(
                        actual.to_vec2::<u32>()?,
                        expected.to_vec2::<u32>()?,
                        "{sampling:?} processors={processors} pos={position}"
                    );
                }
                assert_eq!(native.drain_all()?, cpu.drain_all()?);
                assert_eq!(metal.digest_rows(), host.digest_rows());
                assert_eq!(metal.logits_digest(), host.logits_digest());
                assert_eq!(native.take_logprobs(), cpu.take_logprobs());
            }
        }
        Ok(())
    }

    #[test]
    fn native_blocking_forced_fed_id_owns_history_and_raw_logprob() -> Result<()> {
        let dev = device()?;
        let budget = ThinkingBudget::new(0, vec![2], vec![2])?;
        let mut blocking = Blocking::new(&[99], &dev)?.with_logprobs(3, true);
        blocking.set_thinking_budget(Some(&budget))?;
        let mut native = NativeBlocking::new(blocking, 3)?;
        let mut p = LogitsProcessor::from_sampling(
            7,
            Sampling::TopK {
                temperature: 1.,
                k: 1,
            },
        )
        .with_processors(Processors {
            frequency_penalty: 0.5,
            ..Default::default()
        });
        let row = Tensor::new(&[-100f32, 0., -100.], &dev)?;
        assert_eq!(
            native.sample_and_feed(&mut p, &row)?.to_vec2::<u32>()?,
            [[2]]
        );
        assert_eq!(native.fed_host_ids(), Some(&[2][..]));
        let next = p
            .stage
            .as_mut()
            .expect("active")
            .apply(&Tensor::new(&[0f32; 3], &dev)?, Some(1.))?
            .to_vec1::<f32>()?;
        assert_eq!(
            next,
            [0., 0., -0.5],
            "record forced id once, not drawn id1 or forced id twice"
        );
        let lp = native.take_logprobs().expect("capture");
        assert_eq!(lp.sampled, [log_softmax_row(&[-100., 0., -100.])[2]]);
        let next_row = Tensor::new(&[-100f32, 0., 0.25], &dev)?;
        assert_eq!(
            native
                .sample_and_feed(&mut p, &next_row)?
                .to_vec2::<u32>()?,
            [[1]],
            "next native draw must read the exact fed-id device history, not raw winner2"
        );
        assert_eq!(native.drain_all()?, [2, 1]);
        Ok(())
    }

    #[test]
    fn native_blocking_retract_replays_host_budget_and_excludes_eos_capture() -> Result<()> {
        let dev = device()?;
        let budget = ThinkingBudget::new(0, vec![2], vec![2])?;
        let mut native =
            NativeBlocking::new(Blocking::new(&[2], &dev)?.with_logprobs(1, false), 3)?;
        native.set_thinking_budget(Some(&budget))?;
        let mut p = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1. });
        let row = Tensor::new(&[-100f32, 0., -100.], &dev)?;
        let mark = p.mark().expect("no history");
        assert_eq!(
            native.sample_and_feed(&mut p, &row)?.to_vec2::<u32>()?,
            [[2]]
        );
        native.retract_draw()?;
        p.rewind(mark);
        assert_eq!(
            native.sample_and_feed(&mut p, &row)?.to_vec2::<u32>()?,
            [[2]],
            "budget rewind preserved"
        );
        assert!(native.eos_of_last()?);
        assert!(native.drain_all()?.is_empty());
        assert!(native.take_logprobs().expect("capture").sampled.is_empty());
        Ok(())
    }

    #[test]
    fn invalid_native_blocking_row_publishes_no_history_digest_capture_or_id() -> Result<()> {
        let dev = device()?;
        let mut native =
            NativeBlocking::new(Blocking::new(&[99], &dev)?.with_logprobs(1, true), 3)?;
        let mut p = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1. })
            .with_processors(Processors {
                frequency_penalty: 0.5,
                ..Default::default()
            })
            .with_logits_digest(true);
        assert!(native
            .sample_and_feed(&mut p, &Tensor::new(&[f32::NAN, 0., 0.], &dev)?)
            .is_err());
        assert_eq!(p.digest_rows(), 0);
        assert_eq!(
            p.stage
                .as_mut()
                .expect("active")
                .apply(&Tensor::new(&[0f32; 3], &dev)?, Some(1.))?
                .to_vec1::<f32>()?,
            [0.; 3],
            "invalid draw recorded no device count"
        );
        assert!(native.blocking.ids.is_empty());
        assert!(native.blocking.last.is_none());
        assert!(native.take_logprobs().expect("capture").sampled.is_empty());
        assert!(
            native
                .sample_and_feed(&mut p, &Tensor::new(&[0f32; 3], &dev)?)
                .is_err(),
            "failed adapter is not reset/reused"
        );
        let mut next = NativeBlocking::new(Blocking::new(&[99], &dev)?, 3)?;
        let mut next_p = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1. });
        next.sample_and_feed(&mut next_p, &Tensor::new(&[0f32; 3], &dev)?)?;
        assert_eq!(next.drain_all()?.len(), 1);
        Ok(())
    }
    #[test]
    fn explicit_probability_callback_keeps_nonzero_reweighting_and_processed_order() -> Result<()> {
        let dev = device()?;
        let mut p = LogitsProcessor::from_sampling(17, Sampling::All { temperature: 1. })
            .with_processors(Processors {
                logit_bias: vec![(0, 0.5)],
                ..Default::default()
            });
        let row = Tensor::new(&[0f32; 2], &dev)?;
        let mut calls = 0;
        let mut seen = [false; 2];
        for position in 0..64 {
            p.at_position(Some(position));
            let actual = p.sample_f(&row, |probabilities| {
                calls += 1;
                assert!(
                    probabilities[0] > probabilities[1],
                    "processors precede hook"
                );
                probabilities.copy_from_slice(&[1., 9.]);
            })?;
            let expected = u32::from(LogitsProcessor::uniform_at(17, position) >= 0.1);
            assert_eq!(
                actual, expected,
                "full positive weights, not a boolean mask"
            );
            seen[actual as usize] = true;
        }
        assert_eq!(calls, 64, "one callback per draw");
        assert_eq!(seen, [true, true], "positive mass preserved on both ids");
        Ok(())
    }
}
