//! CPU-testable preflight for FlashDecoding's existing native address contract.
//! This validates geometry, not the contents of arbitrary device metadata.
use lloom_tensor::{DType, Layout, Result};

// The resident producer migration is tracked on #1147. Keep the owner
// implementation private and unwired until that lifecycle is delegated.
#[allow(dead_code)]
#[path = "flash_validation/resident.rs"]
mod resident;
#[allow(unused_imports)]
pub(crate) use resident::ResidentSpans;

#[derive(Clone, Copy)]
pub(crate) struct Pages<'a> {
    pub entries: &'a [u32],
    pub stride: usize,
    pub keys: &'a [u32],
}

// Fields are consumed by Metal; CPU builds retain the common refusal preflight.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub heads: usize,
    pub dim: usize,
    pub keys: usize,
    pub chunk: usize,
    pub chunks: usize,
    pub output_elements: usize,
    pub min_ragged_keys: Option<u32>,
    required_bytes: [usize; 3],
}

fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            lloom_tensor::Error::Msg("flash-decoding: extent/byte arithmetic overflow".into())
        })
}

fn read_span(
    layout: &Layout,
    rows: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    width: usize,
) -> Result<usize> {
    let used = [rows, heads, tokens, dim];
    let last =
        used.iter()
            .zip(layout.stride())
            .try_fold(layout.start_offset(), |at, (&n, &stride)| {
                at.checked_add(product(&[n - 1, stride])?).ok_or_else(|| {
                    lloom_tensor::Error::Msg("flash-decoding: physical read span overflow".into())
                })
            })?;
    let end = last.checked_add(1).ok_or_else(|| {
        lloom_tensor::Error::Msg("flash-decoding: physical read span overflow".into())
    })?;
    product(&[end, width])
}

fn native_int(value: usize) -> Result<()> {
    i32::try_from(value).map(|_| ()).map_err(|_| {
        lloom_tensor::Error::Msg(
            "flash-decoding: geometry exceeds signed native index range".into(),
        )
    })
}

fn validate_chunk_count(chunks: usize) -> Result<()> {
    // Every dtype/head variant shares combine<..., D> with BN=8. Its signed
    // strided loop executes c += 8 even after the last consumed chunk.
    if chunks == 0 || chunks > i32::MAX as usize - 7 {
        lloom_tensor::bail!("flash-decoding: chunk count exceeds signed combine loop range")
    }
    Ok(())
}

fn layout(layout: &Layout, dtype: DType) -> Result<[usize; 4]> {
    let dims: [usize; 4] = layout.dims().try_into().map_err(|_| {
        lloom_tensor::Error::Msg("flash-decoding: Q/K/V must have rank four".into())
    })?;
    if dims.contains(&0) || layout.stride().len() != 4 {
        lloom_tensor::bail!("flash-decoding: Q/K/V dimensions must be positive")
    }
    product(&[product(&dims)?, dtype.size_in_bytes()])?;
    let last =
        dims.iter()
            .zip(layout.stride())
            .try_fold(layout.start_offset(), |last, (&n, &s)| {
                last.checked_add(product(&[n - 1, s])?).ok_or_else(|| {
                    lloom_tensor::Error::Msg("flash-decoding: view extent overflow".into())
                })
            })?;
    let extent = last
        .checked_add(1)
        .ok_or_else(|| lloom_tensor::Error::Msg("flash-decoding: view extent overflow".into()))?;
    product(&[extent, dtype.size_in_bytes()])?;
    Ok(dims)
}

/// Publication-time proof over authenticated host metadata. No Tensor aliases.
/// This summary is independent of a particular layer's K/V head count/capacity.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageSummary {
    rows: usize,
    keys: usize,
    page_size: usize,
    pages_per_chunk: usize,
    max_read_end: usize,
    min_ragged_keys: Option<u32>,
}

impl PageSummary {
    pub(crate) fn route(&self) -> (usize, usize) {
        (self.page_size, self.pages_per_chunk)
    }
}

#[cfg(test)]
std::thread_local! {
    static PREPARATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn preparation_count() -> usize {
    PREPARATIONS.with(|n| n.get())
}

pub(crate) fn prepare_pages(
    pages: Pages<'_>,
    rows: usize,
    page_size: usize,
    pages_per_chunk: usize,
) -> Result<PageSummary> {
    #[cfg(test)]
    PREPARATIONS.with(|n| n.set(n.get().wrapping_add(1)));
    if rows == 0 || page_size == 0 || pages_per_chunk == 0 {
        lloom_tensor::bail!("flash-decoding: rows and chunk geometry must be positive")
    }
    let chunk = product(&[page_size, pages_per_chunk])?;
    for n in [rows, page_size, pages_per_chunk, chunk] {
        native_int(n)?;
    }
    let mut max_read_end = 0;
    if !(pages.keys.len() == 1 || pages.keys.len() == rows) || pages.keys.contains(&0) {
        lloom_tensor::bail!("flash-decoding: live counts must be positive, one or one per row")
    }
    if pages.keys.len() > 1 && pages.stride == 0 {
        lloom_tensor::bail!("flash-decoding: per-row live counts require per-row tables")
    }
    native_int(pages.stride)?;
    let max_keys = pages.keys.iter().copied().max().unwrap_or(0) as usize;
    native_int(max_keys)?;
    let chunks = max_keys.div_ceil(chunk);
    validate_chunk_count(chunks)?;
    let required = (rows - 1)
        .checked_mul(pages.stride)
        .and_then(|n| n.checked_add(chunks))
        .ok_or_else(|| {
            lloom_tensor::Error::Msg("flash-decoding: table index arithmetic overflow".into())
        })?;
    if pages.entries.len() < required {
        lloom_tensor::bail!("flash-decoding: chunk table is shorter than live rows")
    }
    // Even an empty padded chunk reads its table entry before entering
    // the attention core. Check every dispatched entry, but require an
    // in-pool range only for chunks that contain logical keys.
    for row in 0..rows {
        let count = pages.keys[if pages.keys.len() == 1 { 0 } else { row }] as usize;
        for c in 0..chunks {
            let start = product(&[pages.entries[row * pages.stride + c] as usize, page_size])?;
            let logical_start = product(&[c, chunk])?;
            native_int(logical_start)?;
            // Mirror the shader exactly: n_live is upper-clamped only.
            // Empty padded chunks can therefore have a negative end.
            let live = (count as i64 - logical_start as i64).min(chunk as i64);
            if start > i32::MAX as usize - 31 {
                lloom_tensor::bail!("flash-decoding: page span exceeds signed native token range")
            }
            let end = start as i64 + live;
            if end < i32::MIN as i64 || end > i32::MAX as i64 - 31 {
                lloom_tensor::bail!("flash-decoding: page span exceeds signed native token range")
            }
            if live > 0 {
                max_read_end = max_read_end.max(end as usize);
            }
        }
    }
    let min_ragged_keys = if pages.keys.len() > 1 {
        pages.keys.iter().copied().min()
    } else {
        None
    };
    Ok(PageSummary {
        rows,
        keys: max_keys,
        page_size,
        pages_per_chunk,
        max_read_end,
        min_ragged_keys,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate(
    q: &Layout,
    k: &Layout,
    v: &Layout,
    dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
    page_size: usize,
    pages_per_chunk: usize,
    pages: Option<&PageSummary>,
) -> Result<Geometry> {
    let qd = layout(q, dtype)?;
    let kd = layout(k, k_dtype)?;
    let vd = layout(v, v_dtype)?;
    if dtype != k_dtype || dtype != v_dtype || !matches!(dtype, DType::F16 | DType::F32) {
        lloom_tensor::bail!("flash-decoding: Q/K/V require one F16 or F32 dtype")
    }
    if qd[2] != 1 {
        lloom_tensor::bail!("flash-decoding: query sequence length must be one")
    }
    if qd[3] != kd[3] || qd[3] != vd[3] {
        lloom_tensor::bail!("flash-decoding: Q/K/V channel dimensions must match")
    }
    if kd[1] != vd[1] || qd[1] % kd[1] != 0 {
        lloom_tensor::bail!("flash-decoding: K/V heads must match and divide query heads")
    }
    if !q.is_contiguous() || k.stride()[3] != 1 || v.stride()[3] != 1 {
        lloom_tensor::bail!("flash-decoding: packed Q and unit K/V channel strides required")
    }
    if page_size == 0 || pages_per_chunk == 0 {
        lloom_tensor::bail!("flash-decoding: page size and pages per chunk must be positive")
    }
    let chunk = product(&[page_size, pages_per_chunk])?;
    let heads = product(&[qd[0], qd[1]])?;
    for n in [
        qd[0],
        qd[1],
        kd[1],
        qd[3],
        heads,
        page_size,
        pages_per_chunk,
        chunk,
    ] {
        native_int(n)?;
    }
    let keys = match pages {
        None => {
            if kd[0] < qd[0] || vd[0] < qd[0] {
                lloom_tensor::bail!("flash-decoding: K/V batch extents must cover query rows")
            }
            if vd[2] < kd[2] {
                lloom_tensor::bail!("flash-decoding: V key extent must cover K live keys")
            }

            kd[2]
        }
        Some(p) => {
            if kd[0] != 1 || vd[0] != 1 {
                lloom_tensor::bail!("flash-decoding: paged K/V pools must have one batch row")
            }
            if p.rows != qd[0] || p.page_size != page_size || p.pages_per_chunk != pages_per_chunk {
                lloom_tensor::bail!("flash-decoding: prepared metadata route does not match query")
            }
            if p.max_read_end > kd[2] || p.max_read_end > vd[2] {
                lloom_tensor::bail!("flash-decoding: page span exceeds K/V pool")
            }
            p.keys
        }
    };
    if pages.is_none() && keys > i32::MAX as usize - 31 {
        lloom_tensor::bail!("flash-decoding: key count exceeds signed native token range")
    }
    let chunks = keys.div_ceil(chunk);
    validate_chunk_count(chunks)?;
    if pages.is_none() {
        // The generated identity table's final first-page index is cast to
        // U32, then multiplied in signed native arithmetic.
        let last_start = product(&[chunks - 1, chunk])?;
        if last_start > i32::MAX as usize - 31 {
            lloom_tensor::bail!("flash-decoding: identity table exceeds signed native token range")
        }
    }
    let output_elements = product(&[heads, qd[3]])?;
    // The core receives `uint head_idx`: its query pointer arithmetic uses
    // U32 multiplication/addition, unlike the outer kernels' signed indices.
    let query_last = product(&[heads - 1, qd[3]])?
        .checked_add(product(&[qd[3].div_ceil(32), 32])? - 1)
        .ok_or_else(|| {
            lloom_tensor::Error::Msg("flash-decoding: query index arithmetic overflow".into())
        })?;
    u32::try_from(query_last).map_err(|_| {
        lloom_tensor::Error::Msg("flash-decoding: query index exceeds U32 native range".into())
    })?;
    product(&[output_elements, dtype.size_in_bytes()])?;
    let (pool_rows, read_keys) = match pages {
        Some(p) => (1, p.max_read_end),
        None => (qd[0], keys),
    };
    let width = dtype.size_in_bytes();
    let required_bytes = [
        read_span(q, qd[0], qd[1], 1, qd[3], width)?,
        read_span(k, pool_rows, kd[1], read_keys, kd[3], width)?,
        read_span(v, pool_rows, vd[1], read_keys, vd[3], width)?,
    ];
    Ok(Geometry {
        heads,
        dim: qd[3],
        keys,
        chunk,
        chunks,
        output_elements,
        min_ragged_keys: pages.and_then(|p| p.min_ragged_keys),
        required_bytes,
    })
}

#[cfg(any(feature = "metal", test))]
impl Geometry {
    /// The kernel reads only the selected rows and published physical ranges.
    /// Logical spare capacity cannot substitute for authentic backing bytes.
    pub fn validate_backings(self, bytes: [usize; 3]) -> Result<()> {
        for ((required, actual), name) in self
            .required_bytes
            .into_iter()
            .zip(bytes)
            .zip(["Q", "K", "V"])
        {
            if required > actual {
                lloom_tensor::bail!("flash-decoding: {name} read span exceeds native buffer view")
            }
        }
        Ok(())
    }

    /// Validate every scratch shape and native cast before allocating output.
    pub fn validate_capacity(self, capacity: usize) -> Result<()> {
        validate_chunk_count(self.chunks)?;
        native_int(capacity)?;
        if capacity < self.chunks {
            lloom_tensor::bail!("flash-decoding: scratch capacity is below live chunks")
        }
        product(&[self.heads, capacity, self.dim, 4])?;
        product(&[self.heads, capacity, 4])?;
        product(&[self.heads, self.chunks, 4])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(q: &Layout, k: &Layout, v: &Layout, page: usize, ppc: usize) -> Result<Geometry> {
        validate(q, k, v, DType::F32, DType::F32, DType::F32, page, ppc, None)
    }

    #[test]
    fn rank_zero_and_native_integer_extents_refuse_without_native_work() {
        let good = Layout::contiguous(&[1, 1, 1, 64]);
        assert!(plain(&Layout::contiguous(&[1, 1, 64]), &good, &good, 256, 1).is_err());
        let zero = Layout::contiguous(&[1, 0, 1, 64]);
        assert!(plain(&good, &zero, &zero, 256, 1).is_err());
        assert!(plain(&good, &good, &good, usize::MAX, 2).is_err());
        // No tensor allocation is attempted for synthetic huge broadcast geometry.
        let huge = Layout::new((&[1, 1, i32::MAX as usize, 64]).into(), vec![0, 0, 0, 1], 0);
        assert!(plain(&good, &huge, &huge, 256, 1).is_err());
        let offset_overflow = Layout::contiguous_with_offset(&[1, 1, 1, 64], usize::MAX);
        assert!(plain(&offset_overflow, &good, &good, 256, 1).is_err());
    }

    #[test]
    fn live_and_empty_chunks_have_distinct_physical_extent_obligations() -> Result<()> {
        let s = prepare_pages(
            Pages {
                entries: &[2, 100, 0, 3],
                stride: 2,
                keys: &[1, 257],
            },
            2,
            256,
            1,
        )?;
        assert_eq!(s.max_read_end, 769);
        let q = Layout::contiguous(&[2, 2, 1, 64]);
        let k = Layout::contiguous(&[1, 1, 1024, 64]);
        let v = Layout::contiguous(&[1, 1, 768, 64]);
        let error = validate(
            &q,
            &k,
            &v,
            DType::F32,
            DType::F32,
            DType::F32,
            256,
            1,
            Some(&s),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("page span exceeds K/V pool"), "{error}");
        assert!(prepare_pages(
            Pages {
                entries: &[2, u32::MAX, 0, 3],
                stride: 2,
                keys: &[1, 257]
            },
            2,
            256,
            1
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn every_dispatched_padded_table_entry_must_exist() {
        assert!(prepare_pages(
            Pages {
                entries: &[0, 1, 2],
                stride: 2,
                keys: &[257, 1]
            },
            2,
            256,
            1
        )
        .is_err());
        assert!(prepare_pages(
            Pages {
                entries: &[0, 1],
                stride: 0,
                keys: &[1, 257]
            },
            2,
            256,
            1
        )
        .is_err());
        assert!(prepare_pages(
            Pages {
                entries: &[],
                stride: 0,
                keys: &[]
            },
            1,
            256,
            1
        )
        .is_err());
    }

    #[test]
    fn reserve_scratch_checks_shape_products_before_allocation() -> Result<()> {
        let q = Layout::contiguous(&[1, 1, 1, 64]);
        let k = Layout::contiguous(&[1, 1, 257, 64]);
        let g = plain(&q, &k, &k, 256, 1)?;
        assert_eq!(g.chunks, 2);
        assert!(g.validate_capacity(1).is_err());
        assert!(g.validate_capacity(usize::MAX).is_err());
        g.validate_capacity(8)
    }

    #[test]
    fn explicit_sequence_strides_preserve_padded_batch_views() -> Result<()> {
        let q = Layout::contiguous(&[2, 4, 1, 64]);
        let padded = Layout::new(
            (&[2, 2, 17, 64]).into(),
            vec![5 * 512 * 64, 512 * 64, 64, 1],
            64,
        );
        assert_eq!(plain(&q, &padded, &padded, 256, 1)?.heads, 8);
        Ok(())
    }

    #[test]
    fn core_query_offsets_use_u32_not_signed_i32_arithmetic() -> Result<()> {
        let q = Layout::contiguous(&[8_388_609, 4, 1, 64]);
        let k = Layout::contiguous(&[8_388_609, 1, 1, 64]);
        // The last query index is above I32::MAX, but the core uses uint.
        plain(&q, &k, &k, 256, 1)?;
        let overflow = Layout::contiguous(&[16_777_217, 4, 1, 64]);
        let k = Layout::contiguous(&[16_777_217, 1, 1, 64]);
        assert!(plain(&overflow, &k, &k, 256, 1)
            .unwrap_err()
            .to_string()
            .contains("U32"));
        Ok(())
    }

    #[test]
    fn independent_longer_value_capacity_remains_supported() -> Result<()> {
        let q = Layout::contiguous(&[2, 4, 1, 64]);
        let k = Layout::contiguous(&[2, 2, 17, 64]);
        let v = Layout::contiguous(&[2, 2, 512, 64]);
        assert_eq!(plain(&q, &k, &v, 256, 1)?.keys, 17);
        Ok(())
    }

    #[test]
    fn spare_kv_batch_rows_remain_supported() -> Result<()> {
        let q = Layout::contiguous(&[2, 4, 1, 64]);
        let k = Layout::contiguous(&[3, 2, 17, 64]);
        let v = Layout::contiguous(&[4, 2, 512, 64]);
        // The native sequence index is bounded by Q.B; spare rows are unread.
        assert_eq!(plain(&q, &k, &v, 256, 1)?.heads, 8);
        Ok(())
    }

    #[test]
    fn physical_spans_consume_only_query_rows_and_live_value_keys() -> Result<()> {
        let q = Layout::contiguous(&[2, 4, 1, 64]);
        let k = Layout::contiguous(&[3, 2, 17, 64]);
        let v = Layout::contiguous(&[4, 2, 512, 64]);
        let geometry = plain(&q, &k, &v, 256, 1)?;
        assert_eq!(geometry.required_bytes, [2048, 17408, 397568]);
        geometry.validate_backings([2048, 17408, 397568])?;
        for (at, label) in ["Q", "K", "V"].into_iter().enumerate() {
            let mut bytes = geometry.required_bytes;
            bytes[at] -= 1;
            let error = geometry.validate_backings(bytes).unwrap_err().to_string();
            assert!(
                error.contains(&format!("{label} read span exceeds native buffer view")),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn physical_spans_include_each_offset_and_element_width() -> Result<()> {
        let q = Layout::contiguous_with_offset(&[1, 1, 1, 32], 3);
        let k = Layout::contiguous_with_offset(&[1, 1, 2, 32], 7);
        let v = Layout::contiguous_with_offset(&[1, 1, 2, 32], 11);
        let geometry = validate(&q, &k, &v, DType::F16, DType::F16, DType::F16, 256, 1, None)?;
        assert_eq!(geometry.required_bytes, [70, 142, 150]);
        geometry.validate_backings([70, 142, 150])?;
        assert!(geometry.validate_backings([69, 142, 150]).is_err());
        assert!(geometry.validate_backings([70, 141, 150]).is_err());
        assert!(geometry.validate_backings([70, 142, 149]).is_err());
        Ok(())
    }

    #[test]
    fn paged_physical_spans_use_live_partial_tail_not_padded_entries() -> Result<()> {
        let summary = prepare_pages(
            Pages {
                entries: &[2, 100, 0, 3],
                stride: 2,
                keys: &[1, 257],
            },
            2,
            256,
            1,
        )?;
        let q = Layout::contiguous(&[2, 4, 1, 64]);
        let k = Layout::contiguous(&[1, 2, 1024, 64]);
        let v = Layout::contiguous(&[1, 2, 800, 64]);
        let geometry = validate(
            &q,
            &k,
            &v,
            DType::F32,
            DType::F32,
            DType::F32,
            256,
            1,
            Some(&summary),
        )?;
        assert_eq!(geometry.required_bytes, [2048, 459008, 401664]);
        geometry.validate_backings([2048, 459008, 401664])?;
        assert!(geometry.validate_backings([2048, 459007, 401664]).is_err());
        assert!(geometry.validate_backings([2048, 459008, 401663]).is_err());
        Ok(())
    }

    #[test]
    fn physical_span_arithmetic_refuses_before_any_native_access() {
        let q = Layout::contiguous_with_offset(&[1, 1, 1, 64], usize::MAX - 63);
        assert!(read_span(&q, 1, 1, 1, 64, 4).is_err());
        let stride = Layout::new((&[2, 1, 1, 64]).into(), vec![usize::MAX, 64, 64, 1], 0);
        assert!(read_span(&stride, 2, 1, 1, 64, 4).is_err());
    }

    #[test]
    fn combine_stride_increment_is_checked_without_giant_metadata() {
        let mut g = Geometry {
            heads: 1,
            dim: 64,
            keys: i32::MAX as usize,
            chunk: 1,
            chunks: i32::MAX as usize,
            output_elements: 64,
            min_ragged_keys: None,
            required_bytes: [256; 3],
        };
        let error = g.validate_capacity(g.chunks).unwrap_err().to_string();
        assert!(error.contains("signed combine loop range"), "{error}");
        g.chunks = i32::MAX as usize - 7;
        // This checks arithmetic only; it never allocates the synthetic scratch.
        assert!(g.validate_capacity(g.chunks).is_ok());
    }

    #[test]
    fn channel_and_inner_stride_guards_are_independently_required() -> Result<()> {
        let q = Layout::contiguous(&[1, 1, 1, 64]);
        let short = Layout::contiguous(&[1, 1, 1, 32]);
        assert!(plain(&q, &q, &short, 256, 1)
            .unwrap_err()
            .to_string()
            .contains("channel dimensions"));
        let strided = Layout::new((&[1, 1, 1, 64]).into(), vec![128, 128, 128, 2], 0);
        assert!(plain(&q, &strided, &q, 256, 1)
            .unwrap_err()
            .to_string()
            .contains("channel strides"));
        let k = Layout::contiguous(&[1, 1, 2, 64]);
        assert!(plain(&q, &k, &q, 256, 1)
            .unwrap_err()
            .to_string()
            .contains("V key extent"));
        Ok(())
    }

    #[test]
    fn oversized_scratch_capacity_refuses_without_relying_on_byte_overflow() -> Result<()> {
        let q = Layout::contiguous(&[1, 1, 1, 64]);
        let g = plain(&q, &q, &q, 256, 1)?;
        let error = g
            .validate_capacity(i32::MAX as usize + 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("signed native index range"), "{error}");
        Ok(())
    }

    #[test]
    fn empty_chunk_start_is_checked_even_when_signed_end_is_representable() {
        // Row zero's padding start is I32_MAX+1; its actual signed n_live is
        // -255, so end would fit. Checking only end would miss t=start+lane.
        let error = prepare_pages(
            Pages {
                entries: &[0, 8_388_608, 0, 1],
                stride: 2,
                keys: &[1, 257],
            },
            2,
            256,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("signed native token range"), "{error}");
    }

    #[test]
    fn positive_chunk_end_is_checked_independently_of_start() {
        let error = prepare_pages(
            Pages {
                entries: &[(i32::MAX as u32) - 32],
                stride: 1,
                keys: &[64],
            },
            1,
            1,
            64,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("signed native token range"), "{error}");
    }

    #[test]
    fn published_ragged_minimum_preserves_the_existing_selector_membership() -> Result<()> {
        for counts in [&[1, 8][..], &[257, 300], &[1, 257], &[257]] {
            let summary = prepare_pages(
                Pages {
                    entries: &[0; 4],
                    stride: 2,
                    keys: counts,
                },
                counts.len(),
                256,
                1,
            )?;
            let chunks = summary.keys.div_ceil(256);
            let original = chunks > 1 && counts.len() > 1 && counts.iter().any(|&n| n <= 256);
            let cached = chunks > 1 && summary.min_ragged_keys.as_slice().iter().any(|&n| n <= 256);
            assert_eq!(original, cached, "counts={counts:?}");
            assert_eq!(
                summary.min_ragged_keys,
                if counts.len() > 1 {
                    counts.iter().copied().min()
                } else {
                    None
                }
            );
        }
        // Match the shader's negative end for an empty padded chunk at page0.
        let padded = prepare_pages(
            Pages {
                entries: &[2, 0, 0, 3],
                stride: 2,
                keys: &[1, 257],
            },
            2,
            256,
            1,
        )?;
        assert_eq!(padded.max_read_end, 769);
        Ok(())
    }
}
