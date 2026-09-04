//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use lloom_tensor::{DType, Error, Result, Tensor};
use rand::{distr::Distribution, SeedableRng};

pub mod poll;
pub mod token_ring;

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
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
        self.sample_argmax_on_device(&logits)?.to_scalar::<u32>()
    }

    /// Argmax without forcing the result host-side.
    ///
    /// Returns the sampled token as a rank-0 `U32` **device** tensor. This is
    /// the whole of what [`Self::sample_argmax`] does minus its `to_scalar`,
    /// and separating them is what `DESIGN.md` §11.5g's non-blocking token path
    /// is built on: `argmax` already dispatches on the GPU (`fast_argmax_*`,
    /// one per token) and `reduce_op` gives its output `DType::U32`
    /// (`candle-core/src/metal_backend/mod.rs`), so the `u32` the next step
    /// wants is **already in device memory** in the dtype the next step wants
    /// it in. `to_scalar` is the only thing that forces it back.
    ///
    /// So a caller that can consume a device tensor -- writing it into a
    /// persistent input buffer with `slice_set`, comparing it against a
    /// device-resident EOS constant -- never pays the round trip, and the
    /// write leg §11.5f found (the token *goes out again* as the next step's
    /// input) is **deleted rather than relocated**: there is nothing to upload
    /// because nothing came down.
    ///
    /// Temperature > 0 stays on the device too, via gumbel-max. Sampling from
    /// `softmax(logits/T)` is `argmax(logits/T + G)` with `G` an i.i.d. Gumbel
    /// per logit (`-log(-log u)`) -- an argmax over a perturbed logit vector, so
    /// the token is produced by the same device `argmax` the t=0 arm uses and
    /// **nothing comes down**. §11.2c's original blocker (a tier-3 distributional
    /// failure at χ² 36 440) is DISCHARGED: the two root causes -- `rand_uniform`'s
    /// seed advance collapsing onto a period-257 orbit, and `gumbel_softmax`'s
    /// `(1e-7, 0.999)` clamp truncating one Gumbel variate in a thousand -- were
    /// fixed, and the check now passes at χ² 216/280 against a critical 331
    /// (`nn::sampling::gumbel_softmax`, which this delegates to and which returns
    /// a device tensor). So temperature is no longer `--temperature 0` only: the
    /// ring's `to_cpu_blit` stays at 0 at t > 0, where the host path took a
    /// 512 KB whole-vocab readback per token (§11.5f) plus a 74 %-of-arm host
    /// `sort_by` (§11.2c). ArgMax and the two top-{k,p} arms that need a host
    /// sort still round-trip; plain temperature and gumbel-softmax do not.
    pub fn sample_argmax_on_device(&mut self, logits: &Tensor) -> Result<Tensor> {
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

    fn sample_gumbel_softmax(&mut self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let sampled =
            crate::nn::sampling::gumbel_softmax(logits, temperature, lloom_tensor::D::Minus1)?;
        sampled.to_scalar::<u32>()
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
            Sampling::GumbelSoftmax { temperature } => {
                self.sample_gumbel_softmax(&logits, *temperature)?
            }
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
