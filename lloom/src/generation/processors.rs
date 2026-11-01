//! Logits processors: the stage between the model's logits and the draw.
//!
//! Logit bias, the repetition / frequency / presence penalties and min-p, in
//! vLLM's order and to its definitions (`vllm/v1/sample/sampler.py`,
//! `vllm/model_executor/layers/utils.py::apply_penalties`): bias, then the
//! repetition penalty over prompt and output, then frequency and presence over
//! the output, then — for a sampled arm only — min-p against the tempered
//! distribution. Every step is a tensor op on the logits' device, and the
//! output history the penalties read is a per-vocabulary count vector on that
//! device, advanced from the drawn token tensor itself: the ring arm's token
//! never comes to the host for it, and a step costs the same whatever the
//! generation's length (`docs/dispatch.md` §11.2h).

use lloom_tensor::{DType, Device, Result, Tensor};

/// A request's processor settings. The default is neutral: no stage runs, and
/// the sampler's path is the one it has without processors.
#[derive(Clone, Debug, PartialEq)]
pub struct Processors {
    /// Divides a positive logit, multiplies a negative one, for every token in
    /// the prompt or the output so far; `1.0` is off.
    pub repetition_penalty: f64,
    /// Subtracted once from every token the output already holds; `0.0` is off.
    pub presence_penalty: f64,
    /// Subtracted per occurrence in the output so far; `0.0` is off.
    pub frequency_penalty: f64,
    /// Drops a token whose tempered probability is below `min_p` times the
    /// most likely token's; `0.0` is off. Argmax-invariant, so greedy ignores it.
    pub min_p: f64,
    /// Added to the named token's logit.
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for Processors {
    fn default() -> Self {
        Processors {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
        }
    }
}

impl Processors {
    /// Whether any processor changes the logits.
    pub fn is_active(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.min_p != 0.0
            || !self.logit_bias.is_empty()
    }

    /// Whether a processor reads the output history.
    pub fn reads_history(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }

    /// Refuse a value outside the field's range, by the field's name (vLLM's
    /// `SamplingParams._verify_args` ranges; a bias outside OpenAI's ±100 is
    /// refused rather than clamped).
    pub fn validate(&self) -> std::result::Result<(), String> {
        let r = self.repetition_penalty;
        if !r.is_finite() || r <= 0.0 {
            return Err(format!(
                "repetition_penalty must be a finite number > 0, got {r}"
            ));
        }
        for (name, v) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if !(-2.0..=2.0).contains(&v) {
                return Err(format!("{name} must be in [-2, 2], got {v}"));
            }
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        for &(id, b) in &self.logit_bias {
            if !(-100.0..=100.0).contains(&b) {
                return Err(format!(
                    "logit_bias for token {id} must be in [-100, 100], got {b}"
                ));
            }
        }
        Ok(())
    }
}

/// The device-resident half of a [`Stage`], built on the first logits it sees.
struct DeviceState {
    vocab: usize,
    /// Occurrences of each token in the output so far.
    counts: Tensor,
    /// `1.0` where the prompt holds the token (the repetition penalty's
    /// prompt half), when that penalty is on.
    prompt_seen: Option<Tensor>,
    bias: Option<Tensor>,
    one: Tensor,
    neg_inf: Option<Tensor>,
}

/// One request's processors and the history they read.
pub(crate) struct Stage {
    cfg: Processors,
    prompt: Vec<u32>,
    dev: Option<DeviceState>,
}

impl Stage {
    pub(crate) fn new(cfg: Processors) -> Self {
        Stage {
            cfg,
            prompt: Vec::new(),
            dev: None,
        }
    }

    pub(crate) fn config(&self) -> &Processors {
        &self.cfg
    }

    /// The prompt the repetition penalty covers; the output history restarts.
    pub(crate) fn observe_prompt(&mut self, prompt: &[u32]) {
        self.prompt.clear();
        self.prompt.extend_from_slice(prompt);
        self.dev = None;
    }

    fn state(&mut self, vocab: usize, device: &Device) -> Result<&mut DeviceState> {
        if self.dev.as_ref().is_some_and(|d| d.vocab != vocab) {
            lloom_tensor::bail!(
                "logits processors: the vocabulary changed mid-run ({} -> {vocab})",
                self.dev.as_ref().map_or(0, |d| d.vocab)
            );
        }
        if self.dev.is_none() {
            let dense =
                |pairs: &mut dyn Iterator<Item = (u32, f32)>, what: &str| -> Result<Tensor> {
                    let mut v = vec![0f32; vocab];
                    for (id, x) in pairs {
                        let slot = v.get_mut(id as usize).ok_or_else(|| {
                            lloom_tensor::Error::msg(format!(
                                "{what} names token {id}, past the vocabulary ({vocab})"
                            ))
                        })?;
                        *slot += x;
                    }
                    Tensor::from_vec(v, vocab, device)
                };
            let prompt_seen = if self.cfg.repetition_penalty != 1.0 && !self.prompt.is_empty() {
                let mut v = vec![0f32; vocab];
                for &id in &self.prompt {
                    let slot = v.get_mut(id as usize).ok_or_else(|| {
                        lloom_tensor::Error::msg(format!(
                            "the prompt holds token {id}, past the vocabulary ({vocab})"
                        ))
                    })?;
                    *slot = 1.0;
                }
                Some(Tensor::from_vec(v, vocab, device)?)
            } else {
                None
            };
            let bias = if self.cfg.logit_bias.is_empty() {
                None
            } else {
                Some(dense(
                    &mut self.cfg.logit_bias.iter().copied(),
                    "logit_bias",
                )?)
            };
            let neg_inf = if self.cfg.min_p > 0.0 {
                Some(Tensor::full(f32::NEG_INFINITY, vocab, device)?)
            } else {
                None
            };
            self.dev = Some(DeviceState {
                vocab,
                counts: Tensor::zeros(vocab, DType::F32, device)?,
                prompt_seen,
                bias,
                one: Tensor::ones(1, DType::F32, device)?,
                neg_inf,
            });
        }
        Ok(self.dev.as_mut().expect("built above"))
    }

    /// The processed logits, f32, in `logits`' shape. `temperature` is the
    /// draw's (`None` for argmax, where min-p cannot change the answer).
    pub(crate) fn apply(&mut self, logits: &Tensor, temperature: Option<f64>) -> Result<Tensor> {
        let dims = logits.dims().to_vec();
        let x = logits.flatten_all()?.to_dtype(DType::F32)?;
        let vocab = x.elem_count();
        let x = x.reshape((1, vocab))?;
        let counts = self
            .state(vocab, logits.device())?
            .counts
            .reshape((1, vocab))?;
        self.process(x, &counts, temperature)?.reshape(dims)
    }

    /// [`Self::apply`] over the `t` rows of a verify window `[t, vocab]`,
    /// row `i` against the history it would have in a step-by-step decode:
    /// the output so far plus the window's inputs `1..=i` (`inputs`, the
    /// `[t]` ids `[anchor, d1…]`; the anchor is the output's last token and
    /// counted already). The history itself does not move.
    pub(crate) fn apply_rows(
        &mut self,
        rows: &Tensor,
        inputs: &Tensor,
        temperature: Option<f64>,
    ) -> Result<Tensor> {
        let (t, vocab) = rows.dims2()?;
        if inputs.elem_count() != t {
            lloom_tensor::bail!(
                "logits processors: a window of {t} rows with {} inputs",
                inputs.elem_count()
            );
        }
        let x = rows.to_dtype(DType::F32)?;
        let reads = self.cfg.reads_history();
        let dev = rows.device().clone();
        let d = self.state(vocab, &dev)?;
        let counts = if reads && t > 1 {
            // One count at (i, inputs[i]) for every row past the anchor,
            // then each row the sum of the rows up to it: small integers,
            // exact in f32 whatever the summation order.
            let ids = inputs
                .flatten_all()?
                .to_dtype(DType::U32)?
                .reshape((t, 1))?;
            let past_anchor: Vec<f32> = (0..t).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
            let ones = Tensor::from_vec(past_anchor, (t, 1), &dev)?;
            let fed = Tensor::zeros((t, vocab), DType::F32, &dev)?.scatter_add(&ids, &ones, 1)?;
            let mut tril = vec![0f32; t * t];
            for i in 0..t {
                for j in 0..=i {
                    tril[i * t + j] = 1.0;
                }
            }
            let tril = Tensor::from_vec(tril, (t, t), &dev)?;
            tril.matmul(&fed)?.broadcast_add(&d.counts)?
        } else {
            d.counts.reshape((1, vocab))?
        };
        self.process(x, &counts, temperature)
    }

    /// The processors over `x` `[r, vocab]` against the per-row output
    /// counts `counts` (`[r, vocab]`, or `[1, vocab]` for every row).
    fn process(
        &mut self,
        mut x: Tensor,
        counts: &Tensor,
        temperature: Option<f64>,
    ) -> Result<Tensor> {
        let cfg = self.cfg.clone();
        let vocab = x.dim(1)?;
        let d = self.state(vocab, &x.device().clone())?;
        if let Some(bias) = &d.bias {
            x = x.broadcast_add(bias)?;
        }
        if cfg.repetition_penalty != 1.0 {
            let seen = match &d.prompt_seen {
                Some(p) => counts.broadcast_add(p)?.gt(0.0)?,
                None => counts.gt(0.0)?,
            };
            let r = cfg.repetition_penalty;
            let scaled = x
                .gt(0.0)?
                .where_cond(&x.affine(1.0 / r, 0.0)?, &x.affine(r, 0.0)?)?;
            x = seen.broadcast_as(x.shape())?.where_cond(&scaled, &x)?;
        }
        if cfg.frequency_penalty != 0.0 {
            x = x.broadcast_sub(&counts.affine(cfg.frequency_penalty, 0.0)?)?;
        }
        if cfg.presence_penalty != 0.0 {
            let present = counts.gt(0.0)?.to_dtype(DType::F32)?;
            x = x.broadcast_sub(&present.affine(cfg.presence_penalty, 0.0)?)?;
        }
        if let (Some(t), Some(neg_inf)) = (temperature, &d.neg_inf) {
            // p_i >= min_p * p_max  <=>  z_i >= z_max + T * ln(min_p), over
            // p = softmax(z / T): the threshold needs no softmax.
            let threshold = x.max_keepdim(1)?.affine(1.0, t * cfg.min_p.ln())?;
            x = x
                .broadcast_ge(&threshold)?
                .where_cond(&x, &neg_inf.broadcast_as(x.shape())?)?;
        }
        Ok(x)
    }

    /// Count the drawn token, a device `U32` of one element, into the history.
    pub(crate) fn record(&mut self, token: &Tensor) -> Result<()> {
        if !self.cfg.reads_history() {
            return Ok(());
        }
        let Some(d) = self.dev.as_mut() else {
            lloom_tensor::bail!("logits processors: a token recorded before any logits");
        };
        let idx = token.reshape(1)?;
        d.counts.scatter_add_set(&idx, &d.one, 0)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn record_guarded(
        &mut self,
        token: &lloom_tensor::metal_backend::sampling::GuardedSample<'_>,
    ) -> Result<()> {
        if !self.cfg.reads_history() {
            return Ok(());
        }
        let Some(d) = self.dev.as_ref() else {
            lloom_tensor::bail!("logits processors: a token recorded before any logits");
        };
        let (storage, layout) = d.counts.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            lloom_tensor::bail!("guarded counts: contiguous prepared counts required");
        }
        let lloom_tensor::Storage::Metal(storage) = &*storage else {
            lloom_tensor::bail!("guarded counts: Metal history required");
        };
        token.record_counts(storage)
    }

    /// [`Self::record`] for a token the host already holds.
    pub(crate) fn record_id(&mut self, id: u32) -> Result<()> {
        if !self.cfg.reads_history() {
            return Ok(());
        }
        let Some(d) = self.dev.as_ref() else {
            lloom_tensor::bail!("logits processors: a token recorded before any logits");
        };
        let token = Tensor::new(&[id], d.counts.device())?;
        self.record(&token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// vLLM's definitions over host vectors, one token at a time.
    pub(crate) fn reference(
        logits: &[f32],
        prompt: &[u32],
        output: &[u32],
        p: &Processors,
        temperature: Option<f64>,
    ) -> Vec<f32> {
        let v = logits.len();
        let mut counts = vec![0f32; v];
        for &o in output {
            counts[o as usize] += 1.0;
        }
        let mut in_prompt = vec![false; v];
        for &t in prompt {
            in_prompt[t as usize] = true;
        }
        let mut x = logits.to_vec();
        for &(id, b) in &p.logit_bias {
            x[id as usize] += b;
        }
        let r = p.repetition_penalty;
        for i in 0..v {
            if r != 1.0 && (in_prompt[i] || counts[i] > 0.0) {
                x[i] *= if x[i] > 0.0 {
                    (1.0 / r) as f32
                } else {
                    r as f32
                };
            }
            x[i] -= counts[i] * p.frequency_penalty as f32;
            if counts[i] > 0.0 {
                x[i] -= p.presence_penalty as f32;
            }
        }
        if let (Some(t), true) = (temperature, p.min_p > 0.0) {
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let threshold = max + (t * p.min_p.ln()) as f32;
            for xi in &mut x {
                if *xi < threshold {
                    *xi = f32::NEG_INFINITY;
                }
            }
        }
        x
    }

    fn devices() -> Vec<Device> {
        let mut v = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        v.push(Device::new_metal(0).expect("the `metal` feature is on and no Metal device opened"));
        v
    }

    const VOCAB: usize = 97;

    /// Logits on a grid of 0.25 in [-6, 6), both signs, from a fixed LCG.
    fn logits() -> Vec<f32> {
        let mut s = 0x2545_f491_u32;
        (0..VOCAB)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) % 48) as f32 * 0.25 - 6.0
            })
            .collect()
    }

    const PROMPT: &[u32] = &[3, 5, 8, 13, 21, 34, 55, 89];
    const OUTPUT: &[u32] = &[5, 7, 7, 11, 7, 60, 60, 90];

    /// The stage's logits after `OUTPUT` was recorded, against the reference.
    fn check(p: Processors, temperature: Option<f64>) {
        let host = logits();
        let want = reference(&host, PROMPT, OUTPUT, &p, temperature);
        for device in devices() {
            let mut stage = Stage::new(p.clone());
            stage.observe_prompt(PROMPT);
            let t = Tensor::from_vec(host.clone(), (1, VOCAB), &device).unwrap();
            // The first apply builds the device state; the history starts empty.
            let first = stage.apply(&t, temperature).unwrap();
            assert_eq!(first.dims(), &[1, VOCAB], "{device:?}: the shape is kept");
            for (k, &id) in OUTPUT.iter().enumerate() {
                if k % 2 == 0 {
                    stage.record_id(id).unwrap();
                } else {
                    stage.record(&Tensor::new(id, &device).unwrap()).unwrap();
                }
            }
            let got = stage
                .apply(&t, temperature)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut moved = 0;
            for i in 0..VOCAB {
                if want[i] != host[i] {
                    moved += 1;
                }
                if want[i].is_infinite() || got[i].is_infinite() {
                    assert_eq!(
                        got[i], want[i],
                        "{device:?} {p:?}: token {i} masked differently"
                    );
                } else {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                        "{device:?} {p:?}: token {i}: {} against the reference {}",
                        got[i],
                        want[i]
                    );
                }
            }
            assert!(
                moved > 0,
                "{p:?}: the reference moved no logit — a vacuous fixture"
            );
        }
    }

    #[test]
    fn logit_bias_matches_the_reference() {
        check(
            Processors {
                logit_bias: vec![(0, -100.0), (7, 3.5), (96, 100.0)],
                ..Default::default()
            },
            None,
        );
    }

    #[test]
    fn repetition_penalty_covers_prompt_and_output() {
        check(
            Processors {
                repetition_penalty: 1.3,
                ..Default::default()
            },
            None,
        );
        check(
            Processors {
                repetition_penalty: 0.7,
                ..Default::default()
            },
            None,
        );
    }

    /// The prompt half alone: no output recorded, the prompt's tokens move.
    #[test]
    fn repetition_penalty_reads_the_prompt_before_any_output() {
        let p = Processors {
            repetition_penalty: 1.5,
            ..Default::default()
        };
        let host = logits();
        let want = reference(&host, PROMPT, &[], &p, None);
        for device in devices() {
            let mut stage = Stage::new(p.clone());
            stage.observe_prompt(PROMPT);
            let t = Tensor::from_vec(host.clone(), VOCAB, &device).unwrap();
            let got = stage.apply(&t, None).unwrap().to_vec1::<f32>().unwrap();
            for &i in PROMPT {
                let i = i as usize;
                assert_ne!(
                    want[i], host[i],
                    "token {i} is in the prompt and did not move"
                );
                assert!(
                    (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                    "{device:?}: token {i}"
                );
            }
        }
    }

    /// The repetition arm against the tree's host implementation
    /// (`utils::apply_repeat_penalty`, the examples' form) over prompt and
    /// output: a reference written independently of this module's.
    #[test]
    fn repetition_penalty_matches_the_host_helper() {
        let r = 1.4;
        let host = logits();
        let context: Vec<u32> = PROMPT.iter().chain(OUTPUT).copied().collect();
        let cpu = Tensor::from_vec(host.clone(), VOCAB, &Device::Cpu).unwrap();
        let want = crate::utils::apply_repeat_penalty(&cpu, r as f32, &context)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for device in devices() {
            let mut stage = Stage::new(Processors {
                repetition_penalty: r,
                ..Default::default()
            });
            stage.observe_prompt(PROMPT);
            let t = Tensor::from_vec(host.clone(), VOCAB, &device).unwrap();
            stage.apply(&t, None).unwrap();
            for &id in OUTPUT {
                stage.record_id(id).unwrap();
            }
            let got = stage.apply(&t, None).unwrap().to_vec1::<f32>().unwrap();
            for i in 0..VOCAB {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                    "{device:?}: token {i}: {} against the helper's {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn frequency_penalty_counts_occurrences() {
        check(
            Processors {
                frequency_penalty: 0.75,
                ..Default::default()
            },
            None,
        );
        check(
            Processors {
                frequency_penalty: -1.5,
                ..Default::default()
            },
            None,
        );
    }

    #[test]
    fn presence_penalty_counts_once() {
        check(
            Processors {
                presence_penalty: 1.25,
                ..Default::default()
            },
            None,
        );
        check(
            Processors {
                presence_penalty: -2.0,
                ..Default::default()
            },
            None,
        );
    }

    #[test]
    fn min_p_masks_against_the_tempered_top_token() {
        check(
            Processors {
                min_p: 0.1,
                ..Default::default()
            },
            Some(0.7),
        );
        check(
            Processors {
                min_p: 0.35,
                ..Default::default()
            },
            Some(1.3),
        );
    }

    /// Argmax cannot change under min-p, so the greedy draw skips it.
    #[test]
    fn min_p_is_argmax_invariant_and_skipped_by_greedy() {
        let p = Processors {
            min_p: 0.5,
            ..Default::default()
        };
        let host = logits();
        for device in devices() {
            let mut stage = Stage::new(p.clone());
            let t = Tensor::from_vec(host.clone(), VOCAB, &device).unwrap();
            let got = stage.apply(&t, None).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(got, host, "{device:?}");
        }
    }

    #[test]
    fn every_processor_composes_in_vllm_order() {
        check(
            Processors {
                repetition_penalty: 1.2,
                presence_penalty: 0.5,
                frequency_penalty: 0.3,
                min_p: 0.05,
                logit_bias: vec![(7, -2.0), (40, 4.0)],
            },
            Some(0.8),
        );
    }

    #[test]
    fn a_token_past_the_vocabulary_is_refused_by_name() {
        let mut stage = Stage::new(Processors {
            logit_bias: vec![(VOCAB as u32, 1.0)],
            ..Default::default()
        });
        let t = Tensor::zeros(VOCAB, DType::F32, &Device::Cpu).unwrap();
        let e = stage.apply(&t, None).unwrap_err().to_string();
        assert!(
            e.contains("logit_bias names token 97, past the vocabulary (97)"),
            "{e}"
        );
    }

    #[test]
    fn out_of_range_values_are_refused_by_name() {
        let bad = [
            (
                Processors {
                    repetition_penalty: 0.0,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
            (
                Processors {
                    repetition_penalty: f64::NAN,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
            (
                Processors {
                    presence_penalty: 2.5,
                    ..Default::default()
                },
                "presence_penalty",
            ),
            (
                Processors {
                    frequency_penalty: -2.1,
                    ..Default::default()
                },
                "frequency_penalty",
            ),
            (
                Processors {
                    min_p: 1.5,
                    ..Default::default()
                },
                "min_p",
            ),
            (
                Processors {
                    logit_bias: vec![(3, 101.0)],
                    ..Default::default()
                },
                "logit_bias",
            ),
        ];
        for (p, name) in bad {
            let e = p.validate().unwrap_err();
            assert!(e.starts_with(name), "{p:?}: {e}");
        }
        assert!(Processors::default().validate().is_ok());
        assert!(!Processors::default().is_active());
    }
}
