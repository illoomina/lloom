//! Shared host real-input Fourier transform, extracted from Whisper.
//!
//! The recursive FFT and odd-length DFT retain their incumbent arithmetic.
//! Output is interleaved real/imaginary for all N frequencies, without
//! normalization. This is an allocating host implementation, not a device
//! fallback or an allocation-free plan. Callers own admission for the input,
//! output and recursive temporary vectors; no worker threads are spawned.

use lloom_tensor::Result;

pub trait Float:
    num_traits::Float + num_traits::FloatConst + num_traits::NumAssign + Send + Sync
{
}

impl Float for f32 {}
impl Float for f64 {}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2357
pub(crate) fn fft<T: Float>(inp: &[T]) -> Vec<T> {
    let n = inp.len();
    let zero = T::zero();
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![zero; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = T::PI() + T::PI();
    let n_t = T::from(n).unwrap();
    for k in 0..n / 2 {
        let k_t = T::from(k).unwrap();
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2337
pub(crate) fn dft<T: Float>(inp: &[T]) -> Vec<T> {
    let zero = T::zero();
    let n = inp.len();
    let two_pi = T::PI() + T::PI();

    let mut out = Vec::with_capacity(2 * n);
    let n_t = T::from(n).unwrap();
    for k in 0..n {
        let k_t = T::from(k).unwrap();
        let mut re = zero;
        let mut im = zero;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = T::from(j).unwrap();
            let angle = two_pi * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

/// Checked entry for a nonempty real-input transform.
///
/// The largest individual numeric allocation contains 2*N elements. Recursive
/// temporary allocation volume is larger, and must be included by the caller's
/// enclosing pass reservation. Representability is not memory admission.
pub fn real_fft<T: Float>(input: &[T]) -> Result<Vec<T>> {
    if input.is_empty() {
        lloom_tensor::bail!("real FFT input must be nonempty")
    }
    input
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("real FFT output size overflow".into()))?;
    Ok(fft(input))
}

#[cfg(test)]
mod tests;
