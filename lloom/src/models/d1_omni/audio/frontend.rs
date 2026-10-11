//! Explicit host-F32 frontend for the pinned omni waveform/MelFrontend contract.
//! No device readback, decoder, resampler, thread pool or backend fallback.
use crate::nn::signal::real_fft;
use lloom_tensor::Result;

const RATE: usize = 16000;
const MIN_SAMPLES: usize = 8000;
const MAX_SAMPLES: usize = 30 * RATE;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const WINDOW: usize = 400;
const HOP: usize = 160;

fn bytes(n: usize, e: usize) -> Result<usize> {
    n.checked_mul(e)
        .filter(|&x| x <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("audio frontend numeric size overflow".into()))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|&x| x <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("audio frontend numeric size overflow".into()))
}

/// Host numeric demand only: retained tables, borrowed caller input and the
/// subsequent device transfer/cast are separate. Allocator metadata is not
/// represented as zero-cost workspace. This is not an allocation entitlement.
#[derive(Clone, Copy, Debug)]
pub struct FrontendScratch {
    pub samples: usize,
    pub computed_frames: usize,
    pub valid_frames: usize,
    pub output_bytes: usize,
    pub allocation_volume_bytes: usize,
    pub peak_numeric_bytes_bound: usize,
}

/// Owned row-major [features,computed_frames] F32; the last invalid STFT row
/// is zero, not removed. A loaded-model caller transfers/casts explicitly.
#[derive(Debug)]
pub struct MelFeatures {
    pub values: Vec<f32>,
    pub features: usize,
    pub computed_frames: usize,
    pub valid_frames: usize,
}

/// Retained Slaney bank and nonperiodic Hann window. The reference runs the
/// frontend on its model device; this API instead explicitly selects host F32.
/// FFT/reduction order differs and requires tolerance-based reference parity.
#[derive(Debug)]
pub struct MelFrontend {
    features: usize,
    filterbank: Vec<f32>,
    window: Vec<f32>,
}
impl MelFrontend {
    /// Numeric table footprint, usable before allocation by the outer loader.
    /// Construction also uses (features+2) F64 mel-edge values.
    pub fn table_bytes(features: usize) -> Result<(usize, usize)> {
        if features == 0 {
            lloom_tensor::bail!("audio frontend needs positive feature count")
        }
        let edges = features
            .checked_add(2)
            .ok_or_else(|| lloom_tensor::Error::Msg("audio mel-edge count overflow".into()))?;
        // Linspace's integer positions must be exactly representable in F64.
        if edges as u128 > (1u128 << 53) {
            lloom_tensor::bail!("audio mel-edge count exceeds exact F64 index range")
        }
        let table = bytes(features, BINS)?;
        Ok((bytes(add(table, FFT)?, 4)?, bytes(edges, 8)?))
    }
    pub fn new(features: usize) -> Result<Self> {
        Self::table_bytes(features)?;
        let f_sp = 200. / 3.;
        let min_log_mel = 1000. / f_sp;
        let logstep = (6.4f64).ln() / 27.;
        let max_mel = min_log_mel + (8f64).ln() / logstep;
        let step = max_mel / (features + 1) as f64;
        let edges: Vec<f64> = (0..features + 2)
            .map(|i| {
                let mel = if i == features + 1 {
                    max_mel
                } else {
                    i as f64 * step
                };
                if mel >= min_log_mel {
                    1000. * (logstep * (mel - min_log_mel)).exp()
                } else {
                    f_sp * mel
                }
            })
            .collect();
        let mut filterbank = vec![0f32; features * BINS];
        for m in 0..features {
            let left = edges[m + 1] - edges[m];
            let right = edges[m + 2] - edges[m + 1];
            if !(left > 0. && right > 0.) {
                lloom_tensor::bail!("audio Slaney edges are not increasing")
            }
            let norm = 2. / (edges[m + 2] - edges[m]);
            for bin in 0..BINS {
                let hz = bin as f64 * RATE as f64 / FFT as f64;
                let triangle = ((hz - edges[m]) / left)
                    .min((edges[m + 2] - hz) / right)
                    .max(0.);
                // numpy stores each triangle row in F32 BEFORE its in-place
                // multiplication by F64 normalization, then casts back to F32.
                filterbank[m * BINS + bin] = ((triangle as f32) as f64 * norm) as f32;
            }
        }
        let mut window = vec![0f32; FFT];
        for i in 0..WINDOW {
            window[(FFT - WINDOW) / 2 + i] =
                0.5 * (1. - (2. * std::f32::consts::PI * i as f32 / (WINDOW - 1) as f32).cos());
        }
        Ok(Self {
            features,
            filterbank,
            window,
        })
    }
    pub fn scratch(&self, input_samples: usize) -> Result<FrontendScratch> {
        let samples = input_samples.min(MAX_SAMPLES).max(MIN_SAMPLES);
        let valid_frames = samples / HOP;
        let computed_frames = valid_frames + 1;
        let output_bytes = bytes(bytes(self.features, computed_frames)?, 4)?;
        let fixed = add(bytes(samples, 4)?, bytes(FFT + BINS, 4)?)?;
        // Shared radix-2 FFT recurrence V(N)=3N+2V(N/2), V(1)=2:
        // N=512 gives 3*512*9 + 2*512 allocated F32 elements per call.
        let fft_volume = bytes(3 * FFT * 9 + 2 * FFT, 4)?;
        let allocation_volume_bytes = add(
            add(fixed, output_bytes)?,
            bytes(computed_frames, fft_volume)?,
        )?;
        // All recursive FFT vectors alive together are bounded by 8*N;
        // full output, preemphasis, frame and powers remain alive throughout.
        let peak_numeric_bytes_bound = add(add(fixed, output_bytes)?, bytes(8 * FFT, 4)?)?;
        Ok(FrontendScratch {
            samples,
            computed_frames,
            valid_frames,
            output_bytes,
            allocation_volume_bytes,
            peak_numeric_bytes_bound,
        })
    }
    /// Mono16kHz samples; no hidden resampling or range clamping. As in the
    /// reference, nonfinite/out-of-range F32 values are not silently changed.
    pub fn prepare_f32(&self, samples: &[f32]) -> Result<MelFeatures> {
        self.prepare(samples.len(), |i| samples[i])
    }
    /// Mono16kHz signed PCM; convert by the pinned F32 divisor32768.
    pub fn prepare_i16(&self, samples: &[i16]) -> Result<MelFeatures> {
        self.prepare(samples.len(), |i| samples[i] as f32 / 32768.)
    }
    fn prepare<F: Fn(usize) -> f32>(&self, len: usize, sample: F) -> Result<MelFeatures> {
        let demand = self.scratch(len)?;
        let cut = len.min(MAX_SAMPLES);
        let at = |i| if i < cut { sample(i) } else { 0f32 };
        // Padding precedes preemphasis: first padded sample contains -0.97
        // times the final input value, not an already-preemphasized zero.
        let mut emphasis = Vec::with_capacity(demand.samples);
        emphasis.push(at(0));
        for i in 1..demand.samples {
            emphasis.push(at(i) - 0.97 * at(i - 1));
        }
        let t = demand.computed_frames;
        let mut mel = vec![0f32; self.features * t];
        let mut frame = vec![0f32; FFT];
        let mut power = vec![0f32; BINS];
        for row in 0..t {
            for (j, x) in frame.iter_mut().enumerate() {
                let padded = row * HOP + j;
                *x = if padded >= FFT / 2 && padded - FFT / 2 < emphasis.len() {
                    emphasis[padded - FFT / 2] * self.window[j]
                } else {
                    0.
                };
            }
            let spectrum = real_fft(&frame)?;
            for bin in 0..BINS {
                let re = spectrum[2 * bin];
                let im = spectrum[2 * bin + 1];
                let magnitude = (re * re + im * im).sqrt();
                power[bin] = magnitude * magnitude;
            }
            for m in 0..self.features {
                let mut sum = 0f32;
                for (bin, &power) in power.iter().enumerate() {
                    sum += self.filterbank[m * BINS + bin] * power;
                }
                mel[m * t + row] = (sum + 2f32.powi(-24)).ln();
            }
        }
        for m in 0..self.features {
            let row = &mut mel[m * t..(m + 1) * t];
            let n = demand.valid_frames;
            let mean = row[..n].iter().copied().sum::<f32>() / n as f32;
            let variance = row[..n]
                .iter()
                .map(|&x| {
                    let y = x - mean;
                    y * y
                })
                .sum::<f32>()
                / (n - 1) as f32;
            let std = variance.sqrt();
            let divisor = (if std.is_nan() { 0. } else { std }) + 1e-5;
            for x in &mut row[..n] {
                *x = (*x - mean) / divisor;
            }
            row[n..].fill(0.);
        }
        Ok(MelFeatures {
            values: mel,
            features: self.features,
            computed_frames: t,
            valid_frames: demand.valid_frames,
        })
    }
}

#[cfg(test)]
mod tests;
