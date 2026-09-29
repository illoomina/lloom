//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use lloom_tensor::{DType, Error, Result, Tensor};
use rand::{distr::Distribution, SeedableRng};

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
    Blocking, CommitBudget, EosRead, Geometry, ReadbackArms, ReadbackCounters, ReadbackStrategy,
    Ring, RingLap, RingReader, WAKE_LIVE_EVERY_DEFAULT,
};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
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
    /// Gumbel-max at the temperature, its noise from the processor's seeded rng.
    GumbelSoftmax {
        temperature: f64,
    },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
    /// The processors between the logits and the draw; `None` when every one
    /// is neutral, so the sampler's path is exactly the one it has without them.
    stage: Option<processors::Stage>,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng,
            sampling,
            stage: None,
        }
    }

    /// Run `processors` before every draw (bias, penalties, min-p;
    /// [`processors`]). Neutral settings install nothing.
    pub fn with_processors(mut self, processors: Processors) -> Self {
        self.stage = processors
            .is_active()
            .then(|| processors::Stage::new(processors));
        self
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
            | Sampling::GumbelSoftmax { temperature } => Some(temperature),
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

    /// The draw as a `U32` **device** tensor, never read back: the ring
    /// writes it into the next step's input and its history slot, so nothing
    /// comes down and nothing goes up again (§11.5g).
    ///
    /// Argmax is the device `argmax`; plain temperature and gumbel are
    /// gumbel-max, `argmax(z + T·G)` with the Gumbel noise from the seeded host
    /// rng uploaded as an input (§11.2e). The top-{k,p} arms need a host sort
    /// and fall back to the argmax here; the ring refuses them (§11.2c).
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
            Sampling::All { temperature } | Sampling::GumbelSoftmax { temperature } => {
                self.gumbel_max_host_noise(logits, temperature)
            }
            // top-k / top-p need a host sort (§11.2c) -- fall back to the plain
            // argmax on the device, and let the caller's blocking `sample` path
            // handle the nucleus. (These arms disable the ring in the harness.)
            Sampling::TopK { .. } | Sampling::TopP { .. } | Sampling::TopKThenTopP { .. } => {
                logits.argmax(lloom_tensor::D::Minus1)
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
        use rand::Rng;
        let logits = logits.to_dtype(DType::F32)?;
        let n = logits.elem_count();
        let dev = logits.device();
        // Host RNG -> Gumbel variates. Uniform in the tight open interval, then
        // G = -log(-log u).
        let mut g = Vec::<f32>::with_capacity(n);
        for _ in 0..n {
            let u: f64 = self.rng.random_range(1e-9..0.9999999);
            g.push((-((-u.ln()).ln())) as f32);
        }
        let noise = Tensor::from_vec(g, logits.shape(), dev)?;
        // argmax(z + T·G): scaling G by T is the temperature, and the shared
        // 1/T on z is monotone so it drops out of the argmax.
        (logits + noise.affine(temperature, 0.0)?)?.argmax(lloom_tensor::D::Minus1)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        self.sample_f_fed(logits, f, |id| id)
    }

    /// [`Self::sample`], with the step's fed token `feed(draw)` in place of
    /// the draw, as [`Self::sample_argmax_on_device_fed`].
    pub fn sample_fed(&mut self, logits: &Tensor, feed: impl FnOnce(u32) -> u32) -> Result<u32> {
        self.sample_f_fed(logits, |_| {}, feed)
    }

    fn sample_f_fed(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
        feed: impl FnOnce(u32) -> u32,
    ) -> Result<u32> {
        let temperature = self.temperature();
        let Some(stage) = self.stage.as_mut() else {
            return Ok(feed(self.draw(logits, f)?));
        };
        let processed = stage.apply(logits, temperature)?;
        let id = feed(self.draw(&processed, f)?);
        if let Some(stage) = self.stage.as_mut() {
            stage.record_id(id)?;
        }
        Ok(id)
    }

    fn draw(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        // The `ArgMax` arm takes its own path over the UNCAST logits.
        //
        // `to_dtype(F32)` below is one `cast_f16_f32` dispatch over the whole
        // vocabulary -- 128 000 elements for LFM2 (`DESIGN.md` §11.2a's dispatch
        // census) -- and `argmax` reads none of the f32 values it produces: it
        // returns an *index*, and `reduce_op` already has `fast_argmax_f16`,
        // `_bf16` and every other dtype wired
        // (`candle-core/src/metal_backend/mod.rs`, the contiguous reduce table).
        // So on this arm the cast is work whose result is discarded.
        //
        // Widening f16 -> f32 is monotone and injective, so it cannot reorder
        // two logits and cannot move the argmax index. Ties are the one place
        // the two paths could differ -- both kernels break them by returning the
        // lowest index, which is asserted rather than assumed
        // (`the_f16_argmax_index_matches_the_f32_path`).
        //
        // Only `ArgMax` is claimed. Every temperature > 0 arm reads the f32
        // values through `prs`, needs the f32 softmax, and is untouched
        // (`DESIGN.md` §11.2c measures that path at 73.7 % host sort).
        if self.sampling == Sampling::ArgMax {
            return self.sample_argmax(logits.clone());
        }
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = crate::nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            // Handled above, over the uncast logits, so the `ArgMax` arm does
            // not pay the whole-vocabulary cast it never reads.
            Sampling::ArgMax => self.sample_argmax(logits)?,
            // The device draw's gumbel-max, from the processor's seeded host
            // noise, so `--seed` reproduces on this path as on the ring.
            Sampling::GumbelSoftmax { temperature } => self
                .gumbel_max_host_noise(&logits, *temperature)?
                .to_scalar::<u32>()?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}
