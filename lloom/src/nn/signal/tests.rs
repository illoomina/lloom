use super::*;

// Independent O(N²) scalar reference: each complex bin is a direct sum, not the
// production even/odd recursion or odd-length DFT implementation.
fn reference(input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .flat_map(|k| {
            let mut re = 0.;
            let mut im = 0.;
            for (j, x) in input.iter().enumerate() {
                let phase = -2. * std::f64::consts::PI * k as f64 * j as f64 / input.len() as f64;
                re += x * phase.cos();
                im += x * phase.sin();
            }
            [re, im]
        })
        .collect()
}

#[test]
fn real_fft_matches_scalar_even_odd_and_repeats() -> Result<()> {
    for n in [1usize, 2, 3, 5, 6, 8, 16] {
        let input = (0..n)
            .map(|i| ((i * 7 + 3) % 13) as f64 / 8. - 0.5)
            .collect::<Vec<_>>();
        let expected = reference(&input);
        let got64 = real_fft(&input)?;
        assert_eq!(got64.len(), 2 * n);
        for (a, b) in got64.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "F64 N{n}: {a} != {b}");
        }
        let input32 = input.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let got32 = real_fft(&input32)?;
        assert_eq!(got32.len(), 2 * n);
        for (a, b) in got32.iter().zip(&expected) {
            assert!((*a as f64 - b).abs() < 2e-5, "F32 N{n}: {a} != {b}");
        }
        for _ in 0..3 {
            assert_eq!(
                real_fft(&input)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                got64.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                real_fft(&input32)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                got32.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[test]
fn real_fft_absolute_impulse_sign_and_empty_contract() -> Result<()> {
    assert!(real_fft::<f32>(&[]).is_err());
    assert_eq!(real_fft(&[2f32])?, [2., 0.]);
    assert_eq!(
        real_fft(&[1f32, 0., 0., 0.])?,
        [1., 0., 1., 0., 1., 0., 1., 0.]
    );
    let output = real_fft(&[0f64, 1., 0., 0.])?;
    assert_eq!(output[3], -1.);
    assert_eq!(output[7], 1.);
    Ok(())
}
