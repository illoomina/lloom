//! Decode the private sampler ABI before any token can leave its carrier.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowError {
    Nan(u32),
    PositiveInfinity(u32),
    Empty,
    Variate(u32),
    Mass,
    InvalidResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SampleError {
    pub row: usize,
    pub reason: RowError,
}

/// Request-wide sticky error uses the same reason codes as row status. Code 6
/// is an incoherent result or invalid forced id, never a publishable token.
pub(super) fn checked_error(words: &[u32]) -> Result<(), SampleError> {
    if words == [0; 4] {
        return Ok(());
    }
    let row = words.get(1).copied().unwrap_or(0) as usize;
    let detail = words.get(2).copied().unwrap_or(u32::MAX);
    let reason = match words.first().copied() {
        Some(1) => RowError::Nan(detail),
        Some(2) => RowError::PositiveInfinity(detail),
        Some(3) => RowError::Empty,
        Some(4) => RowError::Variate(detail),
        Some(5) => RowError::Mass,
        _ => RowError::InvalidResult,
    };
    Err(SampleError { row, reason })
}

/// Status words precede ids in the completed snapshot. Validate the entire
/// window before exposing even its first id: a later row may have failed.
pub(super) fn checked_ids(words: &[u32], rows: usize, vocab: usize) -> Result<&[u32], SampleError> {
    let invalid = |row| SampleError {
        row,
        reason: RowError::InvalidResult,
    };
    if vocab == 0 || vocab > u32::MAX as usize || rows.checked_mul(5) != Some(words.len()) {
        return Err(invalid(0));
    }
    let (statuses, ids) = words.split_at(rows * 4);
    for (row, (&id, status)) in ids.iter().zip(statuses.chunks_exact(4)).enumerate() {
        let reason = match status[0] {
            0 if id < vocab as u32
                && status[1] == u32::MAX
                && status[2] > 0
                && status[2] as usize <= vocab
                && status[3] > 0
                && status[3] <= status[2] =>
            {
                continue
            }
            1 => RowError::Nan(status[1]),
            2 => RowError::PositiveInfinity(status[1]),
            3 => RowError::Empty,
            4 => RowError::Variate(status[1]),
            5 => RowError::Mass,
            _ => RowError::InvalidResult,
        };
        return Err(SampleError { row, reason });
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticky_error_cannot_be_hidden_by_a_later_successful_row() {
        assert!(checked_error(&[0; 4]).is_ok());
        assert_eq!(
            checked_error(&[1, 7, 2, 0]),
            Err(SampleError {
                row: 7,
                reason: RowError::Nan(2)
            })
        );
        assert_eq!(
            checked_error(&[6, 1, u32::MAX, 0]).unwrap_err().reason,
            RowError::InvalidResult
        );
        assert!(checked_error(&[0, 0, 1, 0]).is_err());
        assert!(checked_error(&[]).is_err());
    }

    #[test]
    fn complete_window_is_checked_before_any_id_is_exposed() {
        let mut words = vec![0, u32::MAX, 4, 2, 0, u32::MAX, 4, 2, 1, 3];
        assert_eq!(checked_ids(&words, 2, 4), Ok(&[1, 3][..]));
        words[4] = 1;
        words[5] = 2;
        assert_eq!(
            checked_ids(&words, 2, 4),
            Err(SampleError {
                row: 1,
                reason: RowError::Nan(2)
            })
        );
    }

    #[test]
    fn every_native_refusal_stays_an_error_even_with_an_in_range_id() {
        for (code, reason) in [
            (1, RowError::Nan(1)),
            (2, RowError::PositiveInfinity(1)),
            (3, RowError::Empty),
            (4, RowError::Variate(1)),
            (5, RowError::Mass),
            (99, RowError::InvalidResult),
        ] {
            assert_eq!(
                checked_ids(&[code, 1, 2, 0, 0], 1, 4),
                Err(SampleError { row: 0, reason })
            );
        }
    }

    #[test]
    fn success_cannot_smuggle_a_sentinel_or_incoherent_counts() {
        for words in [
            [0, u32::MAX, 4, 2, u32::MAX],
            [0, u32::MAX, 4, 2, 4],
            [0, u32::MAX, 0, 0, 0],
            [0, u32::MAX, 5, 1, 0],
            [0, u32::MAX, 4, 5, 0],
            [0, 0, 4, 1, 0],
            [0, u32::MAX, 4, 0, 0],
        ] {
            assert_eq!(
                checked_ids(&words, 1, 4).unwrap_err().reason,
                RowError::InvalidResult
            );
        }
    }

    #[test]
    fn shape_and_overflow_are_errors_not_partial_windows() {
        for (words, rows, vocab) in [
            (&[][..], usize::MAX, 4),
            (&[0][..], 0, 4),
            (&[][..], 0, 0),
            (&[0, u32::MAX, 4, 1][..], 1, 4),
        ] {
            assert!(checked_ids(words, rows, vocab).is_err());
        }
        assert_eq!(checked_ids(&[], 0, 4), Ok(&[][..]));
    }
}
