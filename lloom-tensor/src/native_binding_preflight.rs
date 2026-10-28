//! Pure production preflight shared by native carrier and bounded scalar controls.
//! No native object, allocator root, mutex, callback dispatch or thread lives here.

/// Obtain all fallible custody before consuming a nonrecycled attempt credit.
/// Error drops P on this caller: production P MUST be counted metadata/pin only;
/// this generic ordering helper itself establishes no native destructor contract.
#[cfg(test)]
pub(crate) fn prepare_before_reserve<P, S, E>(
    prepare: impl FnOnce() -> Result<P, E>,
    reserve: impl FnOnce() -> Result<S, E>,
) -> Result<(P, S), E> {
    let prepared = prepare()?;
    let slot = reserve()?;
    Ok((prepared, slot))
}

/// Stack-only whole-batch credit plan. Callbacks only read an immutable forward
/// chain and current indexes; no reservation/effect occurs on partial failure.
/// Sole original encoder EDIT keeps indexes stable until caller commits the
/// returned plan. Concurrent cold append may only extend the chain.
pub(crate) struct PageCreditPlan<P, const N: usize> {
    pub(crate) slots: [(P, usize); N],
    pub(crate) last_page: P,
}
pub(crate) fn plan_page_credits<P: Copy, const N: usize>(
    start: P,
    width: usize,
    mut used: impl FnMut(P) -> Option<usize>,
    mut next: impl FnMut(P) -> Option<P>,
) -> Option<PageCreditPlan<P, N>> {
    if N != 0 && width == 0 {
        return None;
    }
    let mut page = start;
    let mut index = if N == 0 { 0 } else { used(page)? };
    let mut slots = [(start, 0usize); N];
    for slot in &mut slots {
        loop {
            if index > width {
                return None;
            }
            if index < width {
                break;
            }
            page = next(page)?;
            index = used(page)?;
        }
        *slot = (page, index);
        index += 1;
    }
    Some(PageCreditPlan {
        slots,
        last_page: page,
    })
}

/// Exact byte geometry, including the relative view boundary. The kernel's
/// accessed extent remains its own validated argument contract, not inferred.
pub(crate) fn checked_bind_offset(base: usize, length: usize, relative: usize) -> Option<usize> {
    if relative > length {
        return None;
    }
    base.checked_add(relative)
}

/// Byte-transfer geometry only, not allocation identity/custody. The caller
/// supplies SAME-native identity while both authentic owned handles are live;
/// distinct native objects additionally require real disjoint placement proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteCopyRoute {
    Empty,
    AlignedBlit {
        source: usize,
        destination: usize,
    },
    Compute {
        source: usize,
        destination: usize,
        overlap: bool,
    },
}
pub(crate) fn byte_copy_route(
    source_base: usize,
    source_length: usize,
    source_offset: usize,
    destination_base: usize,
    destination_length: usize,
    destination_offset: usize,
    size: usize,
    same_native: bool,
) -> Option<ByteCopyRoute> {
    let source = checked_byte_range(source_base, source_length, source_offset, size)?;
    let destination = checked_byte_range(
        destination_base,
        destination_length,
        destination_offset,
        size,
    )?;
    if size == 0 {
        return Some(ByteCopyRoute::Empty);
    }
    let overlap = same_native
        && source < destination.checked_add(size)?
        && destination < source.checked_add(size)?;
    // macOS blit contract: all three are multiples of4. Do not round an
    // unaligned view into adjacent bytes. Compute/staged recovery remains a
    // genuine plan-owned route, never a permanent U8 refusal.
    if !overlap
        && source.is_multiple_of(4)
        && destination.is_multiple_of(4)
        && size.is_multiple_of(4)
    {
        Some(ByteCopyRoute::AlignedBlit {
            source,
            destination,
        })
    } else {
        Some(ByteCopyRoute::Compute {
            source,
            destination,
            overlap,
        })
    }
}
pub(crate) fn checked_byte_range(
    base: usize,
    length: usize,
    offset: usize,
    size: usize,
) -> Option<usize> {
    if offset.checked_add(size)? > length {
        return None;
    }
    let absolute = base.checked_add(offset)?;
    absolute.checked_add(size)?;
    Some(absolute)
}

pub(crate) struct BufferIndexPlan {
    packed_start: Option<usize>,
}
impl BufferIndexPlan {
    /// Called only at enumerated positions in the SAME preflighted slice. If
    /// packed, start+position was checked as part of the complete cursor end.
    pub(crate) fn resolve(&self, position: usize, original: usize) -> usize {
        self.packed_start.map_or(original, |start| start + position)
    }
}

/// Validate the complete effective-index batch under the caller's EXISTING
/// capture mutex. Refusal leaves both cursor and effects latch unchanged.
/// Success commits its packed range once; caller releases that mutex before
/// lowering and uses resolve, never a second capture renumbering.
///
/// This helper does not serialize old raw/capture APIs or make their custody
/// safe. Production exact encoder admission and later caller migration remain
/// separate requirements. Validation callback contains no native/user effect.
pub(crate) fn preflight_buffer_indices<T, E>(
    capture_cursor: Option<&mut usize>,
    entries: &[T],
    mut original_index: impl FnMut(&T) -> usize,
    mut validate: impl FnMut(usize, &T) -> Result<(), E>,
    error: impl Fn(&'static str) -> E,
    entered_effects: &mut bool,
) -> Result<BufferIndexPlan, E> {
    let start = capture_cursor.as_deref().copied();
    let end = if let Some(start) = start {
        let end = start
            .checked_add(entries.len())
            .ok_or_else(|| error("packed binding cursor overflow"))?;
        if !entries.is_empty() && end > 31 {
            return Err(error("packed binding native index refused"));
        }
        Some(end)
    } else {
        None
    };
    let plan = BufferIndexPlan {
        packed_start: start,
    };
    for (position, entry) in entries.iter().enumerate() {
        let index = plan.resolve(position, original_index(entry));
        if index >= 31 {
            return Err(error("native binding index refused"));
        }
        validate(index, entry)?;
    }
    // From effect-arm/cursor store through return: nonpanicking moves only.
    // Empty batch has no new effect and cannot clear a prior effects latch.
    // A previous legacy cursor beyond range is not mutated by an empty batch.
    *entered_effects |= !entries.is_empty();
    if !entries.is_empty() {
        if let (Some(cursor), Some(end)) = (capture_cursor, end) {
            *cursor = end;
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn native_binding_whole_batch_refusal_keeps_all_page_credits() {
        let used = [Cell::new(2usize), Cell::new(3usize)];
        for _ in 0..32 {
            let result = plan_page_credits::<_, 2>(
                0usize,
                3,
                |page| used.get(page).map(Cell::get),
                |page| (page == 0).then_some(1),
            );
            assert!(
                result.is_none(),
                "partial batch incorrectly reserved capacity"
            );
            assert_eq!(
                [used[0].get(), used[1].get()],
                [2, 3],
                "refused batch consumed a page credit"
            );
        }
        // Same unchanged first credit remains usable after every refusal.
        let plan = plan_page_credits::<_, 1>(
            0usize,
            3,
            |page| used.get(page).map(Cell::get),
            |page| (page == 0).then_some(1),
        )
        .unwrap();
        assert_eq!(plan.slots, [(0, 2)]);
        assert_eq!(plan.last_page, 0);
    }
    #[test]
    fn native_binding_whole_batch_crosses_forward_pages_once() {
        let used = [Cell::new(2usize), Cell::new(1usize)];
        let plan = plan_page_credits::<_, 3>(
            0usize,
            3,
            |page| used.get(page).map(Cell::get),
            |page| (page == 0).then_some(1),
        )
        .unwrap();
        assert_eq!(
            plan.slots,
            [(0, 2), (1, 1), (1, 2)],
            "whole batch lost exact forward page/index ordering"
        );
        assert_eq!([used[0].get(), used[1].get()], [2, 1]);
        assert_eq!(plan.last_page, 1);
        for (page, index) in plan.slots {
            used[page].set(index + 1);
        }
        assert_eq!([used[0].get(), used[1].get()], [3, 3]);
    }

    #[test]
    fn native_binding_repeated_refusal_preserves_attempt_credit() {
        let next = Cell::new(0usize);
        let reserve = || {
            let index = next.get();
            if index == 1 {
                return Err("full");
            }
            next.set(index + 1);
            Ok(index)
        };
        for _ in 0..32 {
            let refused = prepare_before_reserve(|| Err::<usize, _>("mint refused"), reserve);
            assert_eq!(refused, Err("mint refused"));
            assert_eq!(next.get(), 0, "refused custody consumed attempt credit");
        }
        assert_eq!(prepare_before_reserve(|| Ok(7usize), reserve), Ok((7, 0)));
        assert_eq!(next.get(), 1, "successful custody did not reserve credit");
        assert_eq!(prepare_before_reserve(|| Ok(8usize), reserve), Err("full"));
        assert_eq!(next.get(), 1, "full reservation changed capacity");
    }

    #[test]
    fn native_binding_invalid_batch_keeps_packed_cursor() {
        let entries = [(usize::MAX, 0usize), (99, 17)];
        let mut cursor = 1usize;
        let mut entered = false;
        let mut validated = 0usize;
        let result = preflight_buffer_indices(
            Some(&mut cursor),
            &entries,
            |entry| entry.0,
            |_, entry| {
                validated += 1;
                checked_bind_offset(64, 16, entry.1)
                    .map(|_| ())
                    .ok_or("view offset")
            },
            |message| message,
            &mut entered,
        );
        assert!(matches!(result, Err("view offset")));
        assert_eq!(validated, 2, "complete geometry preflight was not reached");
        assert_eq!(cursor, 1, "invalid batch mutated capture cursor");
        assert!(!entered, "invalid batch entered encoder effects");
    }

    #[test]
    fn native_binding_high_original_maps_once_to_valid_packed_indices() {
        let entries = [usize::MAX, 31, 1000];
        let mut cursor = 1usize;
        let mut entered = false;
        let mut observed = [0usize; 3];
        let mut count = 0usize;
        let plan = preflight_buffer_indices(
            Some(&mut cursor),
            &entries,
            |entry| *entry,
            |index, _| {
                observed[count] = index;
                count += 1;
                Ok::<_, &'static str>(())
            },
            |message| message,
            &mut entered,
        )
        .unwrap();
        assert_eq!(
            observed,
            [1, 2, 3],
            "original index bypassed packed preflight"
        );
        assert_eq!(cursor, 4, "packed batch was not committed exactly once");
        assert!(entered);
        for (position, entry) in entries.iter().enumerate() {
            assert_eq!(
                plan.resolve(position, *entry),
                observed[position],
                "lowering remapped packed index twice"
            );
        }
        assert_eq!(cursor, 4, "resolved lowering advanced packed cursor");
        let empty: [usize; 0] = [];
        preflight_buffer_indices(
            Some(&mut cursor),
            &empty,
            |entry| *entry,
            |_, _| Ok::<_, &'static str>(()),
            |message| message,
            &mut entered,
        )
        .unwrap();
        assert!(entered, "empty batch cleared prior encoder effects");
        assert_eq!(cursor, 4, "empty batch advanced packed cursor");
    }

    #[test]
    fn native_binding_view_offset_checks_boundary_and_overflow() {
        assert_eq!(checked_bind_offset(64, 16, 0), Some(64));
        assert_eq!(checked_bind_offset(64, 16, 16), Some(80));
        assert_eq!(
            checked_bind_offset(64, 16, 17),
            None,
            "relative view boundary ignored"
        );
        assert_eq!(
            checked_bind_offset(usize::MAX, 1, 1),
            None,
            "absolute view offset wrapped"
        );
        let mut cursor = usize::MAX;
        let mut entered = false;
        let result = preflight_buffer_indices(
            Some(&mut cursor),
            &[0usize],
            |entry| *entry,
            |_, _| Ok::<_, &'static str>(()),
            |message| message,
            &mut entered,
        );
        assert!(matches!(result, Err("packed binding cursor overflow")));
        assert_eq!(cursor, usize::MAX);
        assert!(!entered);
    }
}

/// Any 64-bit usize byte length needs at most three uint2/signed64 rectangles.
/// Rows cap derives from signed shader addressing, NOT workload capacity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteRectangle {
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}
pub(crate) fn byte_rectangles(bytes: usize) -> Option<[Option<ByteRectangle>; 3]> {
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let width = u32::MAX as usize;
    let max_rows = (i64::MAX as usize) / width;
    let mut rectangles = [None; 3];
    let mut remaining = bytes;
    let mut offset = 0usize;
    for rectangle in &mut rectangles {
        if remaining == 0 {
            break;
        }
        let whole_rows = (remaining / width).min(max_rows);
        let (rows, columns) = if whole_rows == 0 {
            (1, remaining)
        } else {
            (whole_rows, width)
        };
        let covered = rows.checked_mul(columns)?;
        *rectangle = Some(ByteRectangle {
            offset,
            rows,
            columns,
        });
        remaining -= covered;
        offset = offset.checked_add(covered)?;
    }
    (remaining == 0).then_some(rectangles)
}

#[cfg(test)]
mod byte_geometry_controls {
    use super::*;
    #[test]
    fn byte_rectangles_cover_boundaries_and_exact_usize_max_without_wrap() {
        assert_eq!(
            std::mem::size_of::<usize>(),
            8,
            "fixture requires actual64 target"
        );
        let b = u32::MAX as usize;
        let r = (i64::MAX as usize) / b;
        let m = r.checked_mul(b).unwrap();
        let samples = [
            0,
            1,
            b - 1,
            b,
            b + 1,
            m - 1,
            m,
            m + 1,
            2 * m - 1,
            2 * m,
            usize::MAX,
        ];
        for bytes in samples {
            let rectangles = byte_rectangles(bytes).expect("all64 byte lengths represented");
            let mut covered = 0usize;
            for rectangle in rectangles.iter().flatten() {
                assert_eq!(
                    rectangle.offset, covered,
                    "rectangles must cover exact contiguous prefix"
                );
                assert!(rectangle.rows > 0 && rectangle.rows <= u32::MAX as usize);
                assert!(rectangle.columns > 0 && rectangle.columns <= u32::MAX as usize);
                let extent = rectangle.rows.checked_mul(rectangle.columns).unwrap();
                assert!(
                    extent <= i64::MAX as usize,
                    "signed shader rectangle extent"
                );
                let last = (rectangle.rows - 1)
                    .checked_mul(rectangle.columns)
                    .unwrap()
                    .checked_add(rectangle.columns - 1)
                    .unwrap();
                assert_eq!(last, extent - 1, "exact final uint2 shader address");
                covered = covered.checked_add(extent).unwrap();
            }
            assert_eq!(covered, bytes, "byte rectangles lost or duplicated suffix");
        }
        let full = byte_rectangles(usize::MAX).unwrap();
        assert_eq!(
            full.map(|part| part.unwrap().rows),
            [r, r, 1],
            "full64 length needs exact third rectangle"
        );
        assert_eq!(full.map(|part| part.unwrap().columns), [b, b, b]);
    }
    #[test]
    fn byte_copy_route_preserves_u8_overlap_and_touching_boundaries() {
        assert_eq!(
            byte_copy_route(0, 16, 0, 32, 16, 0, 8, false),
            Some(ByteCopyRoute::AlignedBlit {
                source: 0,
                destination: 32
            })
        );
        assert_eq!(
            byte_copy_route(0, 16, 1, 32, 16, 0, 7, false),
            Some(ByteCopyRoute::Compute {
                source: 1,
                destination: 32,
                overlap: false
            }),
            "unaligned legal U8 must retain compute capability"
        );
        assert_eq!(
            byte_copy_route(0, 16, 0, 0, 16, 4, 8, true),
            Some(ByteCopyRoute::Compute {
                source: 0,
                destination: 4,
                overlap: true
            }),
            "overlap must not enter illegal same-buffer blit"
        );
        assert_eq!(
            byte_copy_route(0, 16, 0, 0, 16, 8, 8, true),
            Some(ByteCopyRoute::AlignedBlit {
                source: 0,
                destination: 8
            }),
            "touching half-open ranges are disjoint"
        );
        assert_eq!(
            byte_copy_route(0, 16, 0, 0, 16, 4, 8, false),
            Some(ByteCopyRoute::AlignedBlit {
                source: 0,
                destination: 4
            }),
            "native identity is explicit geometry input, not inferred"
        );
        assert_eq!(
            byte_copy_route(0, 0, 0, 0, 0, 0, 0, true),
            Some(ByteCopyRoute::Empty)
        );
    }
    #[test]
    fn byte_copy_route_checks_views_and_absolute_overflow_before_effects() {
        assert_eq!(
            byte_copy_route(64, 16, 16, 128, 16, 16, 0, false),
            Some(ByteCopyRoute::Empty)
        );
        assert_eq!(
            byte_copy_route(64, 16, 9, 128, 16, 0, 8, false),
            None,
            "source view end escaped"
        );
        assert_eq!(
            byte_copy_route(64, 16, 0, 128, 16, 9, 8, false),
            None,
            "destination view end escaped"
        );
        assert_eq!(
            byte_copy_route(usize::MAX, 1, 0, 0, 1, 0, 1, false),
            None,
            "absolute source end overflow"
        );
        assert_eq!(
            byte_copy_route(0, 1, 0, usize::MAX, 1, 0, 1, false),
            None,
            "absolute destination end overflow"
        );
        assert_eq!(
            checked_byte_range(usize::MAX - 8, 8, 0, 8),
            Some(usize::MAX - 8)
        );
        assert_eq!(
            checked_byte_range(0, usize::MAX, usize::MAX, 1),
            None,
            "relative addition overflow"
        );
    }
}
