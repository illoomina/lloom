//! Host noise for prepared native draws and the incumbent tensor routes.

use super::LogitsProcessor;
use lloom_tensor::Result;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub(super) enum Key {
    Running,
    Position(u64),
    Window(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Method {
    Multinomial,
    Gumbel,
}

pub(super) fn fill_gumbel(seed: u64, position: u64, out: &mut [f32]) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(LogitsProcessor::stream_key(seed, position));
    for value in out {
        *value = LogitsProcessor::gumbel_variate(&mut rng);
    }
}

/// Fill an already prepared upload. Geometry and position overflow are checked
/// before consuming the running stream or modifying any destination element.
pub(super) fn fill(
    out: &mut [f32],
    rows: usize,
    vocab: usize,
    method: Method,
    key: Key,
    seed: u64,
    running: &mut rand::rngs::StdRng,
) -> Result<()> {
    let width = if method == Method::Gumbel { vocab } else { 1 };
    if rows == 0 || vocab == 0 || rows.checked_mul(width) != Some(out.len()) {
        lloom_tensor::bail!("sampling noise: prepared row/vocabulary extent mismatch");
    }
    if let Key::Window(first) = key {
        if u64::try_from(rows - 1)
            .ok()
            .and_then(|n| first.checked_add(n))
            .is_none()
        {
            lloom_tensor::bail!("sampling noise: absolute position overflow");
        }
        for (row, values) in out.chunks_exact_mut(width).enumerate() {
            let position = first + row as u64;
            match method {
                Method::Multinomial => values[0] = LogitsProcessor::uniform_at(seed, position),
                Method::Gumbel => fill_gumbel(seed, position, values),
            }
        }
        return Ok(());
    }
    let mut keyed = match key {
        Key::Position(position) => Some(rand::rngs::StdRng::seed_from_u64(
            LogitsProcessor::stream_key(seed, position),
        )),
        _ => None,
    };
    let rng = keyed.as_mut().unwrap_or(running);
    for value in out {
        *value = match method {
            Method::Multinomial => rng.random(),
            Method::Gumbel => LogitsProcessor::gumbel_variate(rng),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn each_window_row_is_its_absolute_position_not_its_batch_neighbor() -> Result<()> {
        for method in [Method::Multinomial, Method::Gumbel] {
            let width = if method == Method::Gumbel { 7 } else { 1 };
            let mut running = rand::rngs::StdRng::seed_from_u64(901);
            let before: u64 = running.clone().random();
            let mut window = vec![0.; 3 * width];
            fill(&mut window, 3, 7, method, Key::Window(19), 42, &mut running)?;
            for row in 0..3 {
                let mut scalar = vec![0.; width];
                fill(
                    &mut scalar,
                    1,
                    7,
                    method,
                    Key::Position(19 + row),
                    42,
                    &mut running,
                )?;
                assert_eq!(
                    bits(&window[row as usize * width..(row as usize + 1) * width]),
                    bits(&scalar)
                );
            }
            assert_eq!(
                running.random::<u64>(),
                before,
                "keyed draws leave the running stream unchanged"
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_matrix_retains_one_flattened_key_and_original_id_variates() -> Result<()> {
        let mut running = rand::rngs::StdRng::seed_from_u64(901);
        let mut matrix = [0.; 10];
        fill(
            &mut matrix,
            2,
            5,
            Method::Gumbel,
            Key::Position(13),
            42,
            &mut running,
        )?;
        let mut expected = rand::rngs::StdRng::seed_from_u64(LogitsProcessor::stream_key(42, 13));
        let literal: Vec<f32> = (0..10)
            .map(|_| LogitsProcessor::gumbel_variate(&mut expected))
            .collect();
        assert_eq!(bits(&matrix), bits(&literal));
        let mut window = [0.; 10];
        fill(
            &mut window,
            2,
            5,
            Method::Gumbel,
            Key::Window(13),
            42,
            &mut running,
        )?;
        assert_ne!(
            bits(&matrix[5..]),
            bits(&window[5..]),
            "matrix noise is not silently rekeyed per row"
        );
        Ok(())
    }

    #[test]
    fn running_gumbel_consumes_every_original_id_even_when_support_will_filter() -> Result<()> {
        let mut running = rand::rngs::StdRng::seed_from_u64(42);
        let mut expected = running.clone();
        let mut noise = [0.; 13];
        fill(
            &mut noise,
            1,
            13,
            Method::Gumbel,
            Key::Running,
            0,
            &mut running,
        )?;
        for value in noise {
            assert_eq!(
                value.to_bits(),
                LogitsProcessor::gumbel_variate(&mut expected).to_bits()
            );
        }
        assert_eq!(running.random::<u64>(), expected.random::<u64>());
        Ok(())
    }

    #[test]
    fn invalid_prepared_extent_or_position_moves_neither_noise_nor_rng() {
        for (rows, vocab, method, key) in [
            (0, 3, Method::Gumbel, Key::Running),
            (1, 0, Method::Multinomial, Key::Running),
            (usize::MAX, 3, Method::Gumbel, Key::Running),
            (2, 3, Method::Multinomial, Key::Window(u64::MAX)),
            (2, 3, Method::Gumbel, Key::Running),
        ] {
            let mut running = rand::rngs::StdRng::seed_from_u64(42);
            let before: u64 = running.clone().random();
            let mut noise = [99.; 2];
            assert!(fill(&mut noise, rows, vocab, method, key, 42, &mut running).is_err());
            assert_eq!(noise, [99.; 2]);
            assert_eq!(running.random::<u64>(), before);
        }
    }
}
