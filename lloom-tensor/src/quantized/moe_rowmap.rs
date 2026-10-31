//! Checked coordinates for the routed GEMM's ushort2 row map.

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub pairs: u32,
    pub topk: u32,
    pub experts: u32,
    pub block_n: u32,
    pub bound: u32,
}

pub(crate) fn geometry(
    rows: usize,
    topk: usize,
    experts: usize,
    block_n: usize,
) -> Result<Geometry> {
    const COORDINATES: usize = u16::MAX as usize + 1;
    if rows > COORDINATES || topk > COORDINATES {
        crate::bail!("moe_rowmap: rows {rows} and topk {topk} must be in 0..={COORDINATES}")
    }
    if experts == 0 || experts > 1024 {
        crate::bail!("moe_rowmap: n_experts {experts} must be in 1..=1024")
    }
    if !matches!(block_n, 16 | 32) {
        crate::bail!("moe_rowmap: block_n {block_n} must be 16 or 32")
    }
    let pairs = rows
        .checked_mul(topk)
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: pair count overflow".into()))?;
    let pairs = u32::try_from(pairs)
        .map_err(|_| crate::Error::Msg("moe_rowmap: pair count exceeds u32".into()))?;
    let bound = (pairs as usize)
        .div_ceil(block_n)
        .checked_add(experts)
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: block count overflow".into()))?;
    let bound = u32::try_from(bound)
        .map_err(|_| crate::Error::Msg("moe_rowmap: block count exceeds u32".into()))?;
    (bound as usize)
        .checked_mul(3)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: block bytes exceed usize".into()))?;
    (pairs as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: row bytes exceed usize".into()))?;
    Ok(Geometry {
        pairs,
        topk: topk as u32,
        experts: experts as u32,
        block_n: block_n as u32,
        bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_include_the_last_ushort_row_without_truncation() {
        for extent in [0usize, 1, 65535, 65536] {
            for bn in [16, 32] {
                let g = geometry(extent, 1, 2, bn).unwrap();
                assert_eq!(g.pairs as usize, extent);
                assert_eq!(g.experts, 2);
                assert_eq!(g.block_n as usize, bn);
                assert_eq!(g.bound as usize, extent.div_ceil(bn) + 2);
                assert_eq!(geometry(1, extent, 2, bn).unwrap().topk as usize, extent);
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn the_largest_nonempty_representable_pair_count_is_preserved() {
        // Each coordinate fits ushort2. Full 65536² is not a u32 count;
        // the next product leaves more than the mapper's 1024-thread stride
        // below u32::MAX, so its histogram induction cannot wrap either.
        for bn in [16, 32] {
            let g = geometry(65536, 65535, 1024, bn).unwrap();
            assert_eq!(g.pairs, 65536 * 65535);
            assert!(u32::MAX - g.pairs > 1024);
        }
    }

    #[test]
    fn unrepresentable_coordinates_and_products_are_refused() {
        for (rows, topk, experts, bn) in [
            (65537usize, 1, 2, 16),
            (1, 65537, 2, 16),
            (65536, 65536, 2, 16),
            (usize::MAX, 2, 2, 16),
            (1, 1, 1025, 16),
            (1, 1, 0, 16),
            (1, 1, 2, 0),
            (1, 1, 2, 17),
        ] {
            assert!(
                geometry(rows, topk, experts, bn).is_err(),
                "{rows}/{topk}/{experts}/{bn}"
            );
        }
    }
    #[cfg(feature = "metal")]
    #[test]
    fn safe_mapped_matmul_checks_coordinates_and_preserves_empty_tiles() {
        use crate::quantized::{GgmlDType, MoeArm, QTensor};
        use crate::{DType, Device, Tensor};
        let dev = Device::new_metal(0).expect("native row-map control requires Metal");
        let actual = dev
            .as_metal_device()
            .unwrap()
            .commands()
            .actual_command_api();
        eprintln!("ROWMAP_CONTROL actual_api={actual}");
        if let Ok(requested) = std::env::var("LLOOM_METAL_API") {
            if matches!(requested.as_str(), "metal3" | "metal4") {
                assert_eq!(
                    actual, requested,
                    "requested native control must not fall back"
                );
            }
        }
        let weights = Tensor::zeros((2, 1, 32), DType::F32, &dev).unwrap();
        let weights = QTensor::quantize(&weights, GgmlDType::F32).unwrap();
        let input = Tensor::zeros((65537, 1, 32), DType::F32, &dev).unwrap();
        let ids = Tensor::zeros((65537, 1), DType::U32, &dev).unwrap();
        let pools_before = dev.as_metal_device().unwrap().pool_counters();
        let error = weights
            .indexed_moe_forward_with(&input, &ids, MoeArm::Gemm)
            .unwrap_err();
        let pools_after = dev.as_metal_device().unwrap().pool_counters();
        assert_eq!(
            (pools_after.0.lookups, pools_after.1.lookups),
            (pools_before.0.lookups, pools_before.1.lookups),
            "invalid coordinates must be refused before output/map pool requests"
        );
        assert!(
            error.to_string().contains("moe_rowmap: rows 65537"),
            "{error}"
        );
        for (rows, topk) in [(0usize, 1usize), (1, 0)] {
            let input = Tensor::zeros((rows, 1, 32), DType::F32, &dev).unwrap();
            let ids = Tensor::zeros((rows, topk), DType::U32, &dev).unwrap();
            let output = weights
                .indexed_moe_forward_with(&input, &ids, MoeArm::Gemm)
                .unwrap();
            assert_eq!(output.dims(), &[rows, topk, 1]);
        }
        dev.synchronize().unwrap();
    }
}
