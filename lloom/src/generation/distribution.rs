//! Host distribution/filter arithmetic. Scratch belongs to the existing sampler;
//! scalar draws and verify rows use the same original-id masses.
use lloom_tensor::{Result, Storage, Tensor};

#[derive(Clone, Copy)]
struct Rank {
    score: f32,
    id: u32,
    rank: u32,
}

#[derive(Default)]
pub(super) struct Scratch {
    pub values: Vec<f32>,
    order: Vec<Rank>,
    // Cached while preparing the same row, before scores become weights.
    // Centering preserves small Gumbel noise beside very large equal logits.
    gumbel_max: f32,
}

impl Scratch {
    /// CPU reads reuse capacity instead of allocating another row. Other backends
    /// retain their existing completion-aware readback; no native bridge is used.
    pub fn load(&mut self, logits: &Tensor) -> Result<()> {
        if !logits.device().is_cpu() {
            self.values = logits.flatten_all()?.to_vec1::<f32>()?;
            return Ok(());
        }
        let (storage, layout) = logits.storage_and_layout();
        let Storage::Cpu(lloom_tensor::CpuStorage::F32(data)) = &*storage else {
            lloom_tensor::bail!("host distribution needs f32 CPU scores");
        };
        self.values.clear();
        if let Some((lo, hi)) = layout.contiguous_offsets() {
            self.values.extend_from_slice(&data[lo..hi]);
        } else {
            self.values.extend(logits.strided_index().map(|i| data[i]));
        }
        Ok(())
    }

    fn rank(&mut self, values: &[f32], probabilities: bool, k: usize, p: f64) {
        self.order.clear();
        self.order
            .extend(values.iter().enumerate().map(|(id, &score)| Rank {
                score,
                id: id as u32,
                rank: 0,
            }));
        let compare = |a: &Rank, b: &Rank| {
            let cmp = if probabilities {
                b.score.total_cmp(&a.score)
            } else if a.score == b.score {
                std::cmp::Ordering::Equal
            } else {
                b.score.total_cmp(&a.score)
            };
            cmp.then_with(|| a.id.cmp(&b.id))
        };
        let n = self.order.len();
        let kept = if k == 0 { n } else { k.min(n) };
        if kept < n {
            self.order.select_nth_unstable_by(kept, compare);
        }
        if p != 1.0 {
            self.order[..kept].sort_unstable_by(compare);
        }
        for (rank, entry) in self.order.iter_mut().enumerate() {
            entry.rank = rank as u32;
        }
        // Invert the permutation in place: retain original scores and ranks
        // addressable by token id without a second vocabulary-sized vector.
        for i in 0..self.order.len() {
            while self.order[i].id as usize != i {
                let target = self.order[i].id as usize;
                self.order.swap(i, target);
            }
        }
    }

    pub fn prepare(&mut self, temperature: f64, k: usize, p: f64, keep_scores: bool) -> Result<()> {
        let mut values = std::mem::take(&mut self.values);
        let result = self.prepare_row(&mut values, temperature, k, p, keep_scores);
        self.values = values;
        result
    }

    pub fn prepare_row(
        &mut self,
        values: &mut [f32],
        temperature: f64,
        k: usize,
        p: f64,
        keep_scores: bool,
    ) -> Result<()> {
        validate(temperature, p)?;
        let mut max = f32::NEG_INFINITY;
        for (i, &z) in values.iter().enumerate() {
            if z.is_nan() || z == f32::INFINITY {
                lloom_tensor::bail!("sampling: invalid score at token {i}: {z}");
            }
            max = max.max(z);
        }
        if !max.is_finite() || values.len() > u32::MAX as usize {
            lloom_tensor::bail!(
                "sampling: a nonempty vocabulary with a finite candidate is required"
            );
        }
        if k != 0 || p != 1.0 || keep_scores {
            self.rank(values, false, k, p);
        } else {
            self.order.clear();
        }
        if keep_scores {
            self.gumbel_max = max;
        }
        let inv = (1.0 / temperature) as f32;
        for z in values.iter_mut() {
            *z = exp32((*z - max) * inv);
        }
        self.filter(values, k, p);
        Ok(())
    }

    /// Arbitrary callback probabilities retain their full values, not just a
    /// nonzero mask. Like the public callback API, this adapter does not add an
    /// invalid-output refusal policy. Its draw remains the host f64 walk.
    pub fn filter_probabilities(&mut self, k: usize, p: f64) {
        if p == 1.0 && (k == 0 || k >= self.values.len()) {
            self.order.clear();
            return;
        }
        let mut values = std::mem::take(&mut self.values);
        self.rank(&values, true, k, p);
        self.filter(&mut values, k, p);
        self.values = values;
    }

    fn mass(&self, values: &[f32], ranks: usize, last_id: usize) -> f32 {
        mass(values, |i| {
            i <= last_id && (self.order.is_empty() || (self.order[i].rank as usize) < ranks)
        })
    }

    fn filter(&mut self, values: &mut [f32], k: usize, p: f64) {
        let n = values.len();
        let kept = if k == 0 { n } else { k.min(n) };
        if !self.order.is_empty() {
            for (w, rank) in values.iter_mut().zip(&self.order) {
                if rank.rank as usize >= kept {
                    *w = 0.0;
                }
            }
        }
        if p == 1.0 || kept == 0 {
            return;
        }
        let target = p as f32 * self.mass(values, kept, usize::MAX);
        let (mut lo, mut hi) = (1, kept);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.mass(values, mid, usize::MAX) >= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        for (w, rank) in values.iter_mut().zip(&self.order) {
            if rank.rank as usize >= lo {
                *w = 0.0;
            }
        }
    }

    pub fn draw(&self, u: f32) -> u32 {
        self.draw_row(&self.values, u)
    }

    /// Locate the crossing within the same group/lane folds as mass. This
    /// avoids a vocabulary pass for every bit of the token id.
    pub fn draw_row(&self, values: &[f32], u: f32) -> u32 {
        let total = mass(values, |_| true);
        let target = u * total;
        let last = values
            .iter()
            .rposition(|&w| w > 0.0)
            .expect("maximum has weight one");
        if total.partial_cmp(&target) != Some(std::cmp::Ordering::Greater) {
            return last as u32;
        }
        let chunk = values.len().div_ceil(1024);
        let mut before_groups = 0.0f32;
        for group in 0..32 {
            let mut before_lanes = 0.0f32;
            for lane in group * 32..(group + 1) * 32 {
                let lo = (lane * chunk).min(values.len());
                let hi = (lo + chunk).min(values.len());
                let mut whole = 0.0f32;
                for &w in &values[lo..hi] {
                    whole += w;
                }
                if before_groups + (before_lanes + whole) > target {
                    let mut partial = 0.0f32;
                    for (j, &w) in values[lo..hi].iter().enumerate() {
                        partial += w;
                        if w > 0.0 && before_groups + (before_lanes + partial) > target {
                            return (lo + j) as u32;
                        }
                    }
                }
                before_lanes += whole;
            }
            before_groups += before_lanes;
        }
        last as u32
    }

    /// Noise is consumed for every original id, even when that candidate is
    /// excluded. No masked candidate participates in the maximum.
    pub fn gumbel(&self, temperature: f64, rng: &mut rand::rngs::StdRng) -> u32 {
        self.gumbel_row(&self.values, temperature, rng)
    }

    pub fn gumbel_row(
        &self,
        values: &[f32],
        temperature: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> u32 {
        self.gumbel_noise(
            values,
            temperature,
            (0..values.len()).map(|_| super::LogitsProcessor::gumbel_variate(rng)),
        )
    }

    fn gumbel_noise(
        &self,
        values: &[f32],
        temperature: f64,
        noise: impl Iterator<Item = f32>,
    ) -> u32 {
        let mut best = None;
        let max = self.gumbel_max;
        let inv = (1.0 / temperature) as f32;
        for (i, (&w, g)) in values.iter().zip(noise).enumerate() {
            if w > 0.0 {
                let score = centered_gumbel(self.order[i].score, max, inv, g);
                if best.is_none_or(|(_, value)| score > value) {
                    best = Some((i as u32, score));
                }
            }
        }
        best.expect("maximum has weight one").0
    }
}

/// Separate f32 rounding at subtraction, reciprocal multiplication and noise
/// addition. Centering keeps noise observable beside large common offsets.
fn centered_gumbel(z: f32, maximum: f32, inv_t: f32, noise: f32) -> f32 {
    let delta = z - maximum;
    let scaled = delta * inv_t;
    scaled + noise
}

pub(super) fn validate(temperature: f64, p: f64) -> Result<()> {
    if !(temperature.is_finite() && temperature > 0.0 && ((1.0 / temperature) as f32).is_finite()) {
        lloom_tensor::bail!(
            "sampling: temperature needs a finite f32 reciprocal, got {temperature}"
        );
    }
    if !(0.0..=1.0).contains(&p) {
        lloom_tensor::bail!("sampling: top_p must be in [0, 1], got {p}");
    }
    Ok(())
}

/// The logical 1024 lanes / 32 groups are independent of CPU vector width.
fn mass(weights: &[f32], include: impl Fn(usize) -> bool) -> f32 {
    let chunk = weights.len().div_ceil(1024);
    let mut total = 0.0f32;
    for group in 0..32 {
        let mut sum = 0.0f32;
        for lane in group * 32..(group + 1) * 32 {
            let mut partial = 0.0f32;
            let lo = (lane * chunk).min(weights.len());
            let hi = (lo + chunk).min(weights.len());
            for (j, &weight) in weights[lo..hi].iter().enumerate() {
                if include(lo + j) {
                    partial += weight;
                }
            }
            sum += partial;
        }
        total += sum;
    }
    total
}

/// The pinned separately rounded polynomial, not the platform exponential.
fn exp32(x: f32) -> f32 {
    if !matches!(
        x.partial_cmp(&-87.0),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    ) {
        return 0.0;
    }
    let n = (x * f32::from_bits(0x3fb8_aa3b)).round_ties_even();
    let r = n.mul_add(-f32::from_bits(0x3f31_8000), x);
    let r = n.mul_add(-f32::from_bits(0xb95e_8083), r);
    let mut a = f32::from_bits(0x3950_6967).mul_add(r, f32::from_bits(0x3ab7_43ce));
    for c in [0x3c08_8908, 0x3d2a_a9c1, 0x3e2a_aaaa, 0x3f00_0000] {
        a = a.mul_add(r, f32::from_bits(c));
    }
    let y = a.mul_add(r * r, r) + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[cfg(test)]
#[path = "../../../lloom-tensor/tests/sampling_contract/fixtures.rs"]
mod fixtures;
#[cfg(test)]
#[path = "../../../lloom-tensor/tests/sampling_contract/oracle.rs"]
mod oracle;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_f32_gumbel_keeps_the_approved_rounding_discriminator() {
        let g = f32::from_bits(0x3eaa_aaab);
        let inv = (1.0f64 / 3.0) as f32;
        assert!(-1.0f64 / 3.0 + f64::from(g) > 0.0);
        assert_eq!(centered_gumbel(-1.0, 0.0, inv, g).to_bits(), 0);
        assert_eq!(centered_gumbel(1e30, 1e30, inv, g), g);
        assert_eq!(
            centered_gumbel(0.0, 0.0, 1.0, f32::from_bits(1)).to_bits(),
            1
        );
        let mut scratch = Scratch {
            values: vec![0.0, -1.0],
            ..Default::default()
        };
        scratch.prepare(3.0, 0, 1.0, true).unwrap();
        assert_eq!(
            scratch.gumbel_noise(&scratch.values, 3.0, [0.0, g].into_iter()),
            0
        );
    }

    #[test]
    fn hand_derived_draws_and_refusals_hold_on_the_production_arithmetic() {
        let (mut kept, mut draws, mut refusals) = (0, 0, 0);
        for case in fixtures::all() {
            let mut scratch = Scratch {
                values: case.z.clone(),
                ..Default::default()
            };
            let p = case.params;
            let result = scratch.prepare(p.temperature, p.k, p.p, false);
            match case.expect {
                fixtures::Expect::Refused(_) => {
                    assert!(result.is_err(), "{}", case.name);
                    refusals += 1;
                }
                fixtures::Expect::Kept(want) => {
                    result.unwrap();
                    let support: Vec<u32> = scratch
                        .values
                        .iter()
                        .enumerate()
                        .filter_map(|(id, &w)| (w > 0.0).then_some(id as u32))
                        .collect();
                    assert_eq!(support, want, "{} support", case.name);
                    kept += 1;
                    for (u, want) in case.draws {
                        assert_eq!(scratch.draw(u), want, "{} u={u}", case.name);
                        draws += 1;
                    }
                }
            }
        }
        assert_eq!((kept, draws, refusals), (33, 48, 11));
    }

    #[test]
    fn accepted_subnormal_and_overflow_inputs_keep_the_literal_host_draw() {
        let mut scratch = Scratch {
            values: vec![0.0; 37],
            ..Default::default()
        };
        scratch.values[0] = -87.0;
        scratch.prepare(1.0, 0, 1.0, false).unwrap();
        assert_eq!(scratch.draw(f32::from_bits(0x007f_ffff)), 1);
        scratch.values = vec![-f32::MAX, 0.0];
        scratch.prepare(1e39, 0, 1.0, false).unwrap();
        assert_eq!(scratch.draw(0.45), 1);
        scratch.values = vec![-f32::MAX, f32::MAX];
        scratch.prepare(f64::MAX, 0, 1.0, false).unwrap();
        assert_eq!(scratch.draw(0.0), 1);
    }

    #[test]
    fn literal_exponential_and_original_mass_order_are_not_platform_foils() {
        for (x, bits) in [
            (-0.500_202_9, 0x3f1b_3d88),
            (-0.503_848_7, 0x3f1a_ace8),
            (-87.0, 0x00b3_3687),
        ] {
            assert_eq!(exp32(x).to_bits(), bits);
        }
        // Independent frozen fold discriminator from sampling_contract::PINS.
        let (mut state, mut row) = (10752749459413695495u64, Vec::with_capacity(3000));
        for _ in 0..3000 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = state;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            row.push((x >> 40) as f32 / (1u64 << 24) as f32 * 12.0 - 6.0);
        }
        let mut scratch = Scratch {
            values: row,
            ..Default::default()
        };
        scratch.prepare(0.5, 0, 1.0, false).unwrap();
        assert_eq!(scratch.draw(0.16891079), 490);
    }
}
