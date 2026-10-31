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
    if rows != 0 && topk != 0 && (rows > COORDINATES || topk > COORDINATES) {
        crate::bail!(
            "moe_rowmap: nonempty rows {rows} and topk {topk} must be in 1..={COORDINATES}"
        )
    }
    // Empty maps form no ushort coordinates, but the actual native arguments
    // still have types: mapper topk is uint, mapped GEMM batch is int64_t.
    let topk_native = u32::try_from(topk)
        .map_err(|_| crate::Error::Msg("moe_rowmap: topk exceeds u32".into()))?;
    i64::try_from(rows)
        .map_err(|_| crate::Error::Msg("moe_rowmap: rows exceeds native i64".into()))?;
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
        topk: topk_native,
        experts: experts as u32,
        block_n: block_n as u32,
        bound,
    })
}

/// Empty maps still take the production mapper/mapped-GEMM route. Check the
/// dimensions and stride products the Rust wrapper actually forms, even though
/// meta=0 makes the MSL return before reading coordinates. No arbitrary empty
/// cap and no alternate kernel route. The nonempty route is unchanged.
pub(crate) fn empty_native_strides(
    (rows, topk, in_dim1): (usize, usize, usize),
    (experts, n, k): (usize, usize, usize),
    weight_block: usize,
    weight_type: usize,
) -> Result<()> {
    if rows != 0 && topk != 0 {
        return Ok(());
    }
    u32::try_from(topk).map_err(|_| crate::Error::Msg("moe_rowmap: topk exceeds u32".into()))?;
    for (name, value) in [
        ("rows", rows),
        ("topk", topk),
        ("input dim1", in_dim1),
        ("experts", experts),
        ("n", n),
        ("k", k),
    ] {
        i64::try_from(value)
            .map_err(|_| crate::Error::Msg(format!("moe_rowmap: {name} exceeds native i64")))?;
    }
    let row_bytes = k
        .checked_div(weight_block)
        .and_then(|blocks| blocks.checked_mul(weight_type))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: weight row bytes overflow".into()))?;
    u64::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(row_bytes))
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: weight expert stride overflow".into()))?;
    u64::try_from(k)
        .ok()
        .and_then(|k| k.checked_mul(4))
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: activation row stride overflow".into()))?;
    in_dim1
        .checked_mul(k)
        .and_then(|elems| u64::try_from(elems).ok())
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| crate::Error::Msg("moe_rowmap: activation batch stride overflow".into()))?;
    Ok(())
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
    #[test]
    fn empty_maps_do_not_form_ushort_coordinates() {
        for (rows, topk) in [(65537usize, 0usize), (0, 65537)] {
            for bn in [16, 32] {
                let g = geometry(rows, topk, 2, bn).unwrap();
                assert_eq!(g.pairs, 0);
                assert_eq!(g.topk as usize, topk);
                assert_eq!(g.bound, 2);
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn empty_native_representations_are_checked_without_coordinate_caps() {
        assert!(geometry(0, u32::MAX as usize, 2, 16).is_ok());
        assert!(geometry(i64::MAX as usize, 0, 2, 16).is_ok());
        assert!(geometry(0, u32::MAX as usize + 1, 2, 16).is_err());
        assert!(geometry(i64::MAX as usize + 1, 0, 2, 16).is_err());
        for args in [(65537, 0, 1), (0, 65537, 1), (0, 65537, 65537)] {
            empty_native_strides(args, (2, 1, 32), 1, 4).unwrap();
        }
        // Representable dimensions do not imply representable native strides.
        let topk = u32::MAX as usize;
        let k = (u64::MAX / 4 / topk as u64 + 1) as usize;
        let error = empty_native_strides((0, topk, topk), (2, 1, k), 1, 4).unwrap_err();
        assert!(
            error.to_string().contains("activation batch stride"),
            "{error}"
        );
        assert!(empty_native_strides((0, 1, 1), (2, i64::MAX as usize, 1), 1, 4).is_err());
        assert!(empty_native_strides((0, 1, 1), (2, 1, i64::MAX as usize), 1, 4).is_err());
        assert!(empty_native_strides((0, 1, 1), (2, 1, 32), 0, 4).is_err());
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
        assert!(error.to_string().contains("rows 65537"), "{error}");
        for (rows, topk) in [(0usize, 1usize), (1, 0), (0, 65537), (65537, 0)] {
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
