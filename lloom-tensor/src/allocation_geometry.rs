//! Checked request geometry, before allocator state changes. This is integer
//! representability, not a memory budget, native maximum, or complete Shape audit.

use crate::{DType, Error, Result};

pub(crate) fn typed_bytes(elements: usize, dtype: DType) -> Result<usize> {
    elements.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
        Error::Msg(format!(
            "allocation byte count overflow: {elements} elements of {dtype:?}"
        ))
    })
}

/// Preserve zero-request minimum and selected alignment without wrap/panic.
/// Alignment need only be nonzero; backend configuration applies its own policy.
pub(crate) fn aligned_nonzero_bytes(size: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        crate::bail!("allocation alignment must be nonzero");
    }
    let size = size.max(1);
    let remainder = size % alignment;
    if remainder == 0 {
        return Ok(size);
    }
    size.checked_add(alignment - remainder).ok_or_else(|| {
        Error::Msg(format!(
            "allocation alignment overflow: {size} bytes aligned to {alignment}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_typed_bytes_exact_and_overflow() {
        for dtype in [
            DType::U8,
            DType::U32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
        ] {
            let width = dtype.size_in_bytes();
            for elements in [0, 1, 17, usize::MAX / width] {
                assert_eq!(
                    typed_bytes(elements, dtype).unwrap() as u128,
                    elements as u128 * width as u128
                );
            }
            if width > 1 {
                assert!(typed_bytes(usize::MAX / width + 1, dtype).is_err());
            }
        }
    }

    #[test]
    fn numeric_alignment_matches_wide_oracle_at_boundaries() {
        assert!(aligned_nonzero_bytes(0, 0).is_err());
        for alignment in [1usize, 3, 16, 64, 128, 256, 4096, usize::MAX] {
            for size in [
                0usize,
                1,
                2,
                127,
                128,
                129,
                4104,
                21504,
                299008,
                usize::MAX - alignment,
                usize::MAX - 1,
                usize::MAX,
            ] {
                let wide = (size.max(1) as u128).div_ceil(alignment as u128) * alignment as u128;
                match usize::try_from(wide) {
                    Ok(exact) => assert_eq!(aligned_nonzero_bytes(size, alignment).unwrap(), exact),
                    Err(_) => assert!(aligned_nonzero_bytes(size, alignment).is_err()),
                }
            }
        }
        assert_eq!(aligned_nonzero_bytes(0, 128).unwrap(), 128);
        assert_eq!(aligned_nonzero_bytes(4104, 128).unwrap(), 4224);
        assert_eq!(aligned_nonzero_bytes(usize::MAX, 1).unwrap(), usize::MAX);
    }
}
