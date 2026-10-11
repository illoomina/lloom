//! Checked placement free ranges shared by production Metal heap chunks and
//! CPU controls. Range return is allocator reuse, NOT physical charge release.

use crate::memory_admission::DomainKey;

pub(crate) fn reusable_for(
    ready: bool,
    owner: Option<&DomainKey>,
    requested: Option<&DomainKey>,
) -> bool {
    ready && owner == requested
}

/// Caller-owned reusable-capacity scratch. Entries are advisory free-space
/// snapshots, NOT proof an aligned/fragmented request fits. Clear retains host
/// capacity between plans; growth allocates only when the source census grows.
#[derive(Debug, Clone, Default)]
pub struct ReuseCredit {
    sources: std::collections::HashMap<usize, usize>,
}
impl ReuseCredit {
    pub fn clear(&mut self) {
        self.sources.clear();
    }
    /// Identity must stay retained across collection. Repeated snapshots of a
    /// source use minimum free capacity, never sum aliased reports.
    pub fn observe(&mut self, source: usize, free: usize) {
        self.sources
            .entry(source)
            .and_modify(|old| *old = (*old).min(free))
            .or_insert(free);
    }
    pub fn checked_total(&self) -> Option<usize> {
        self.sources
            .values()
            .try_fold(0usize, |sum, &free| sum.checked_add(free))
    }
}

#[derive(Debug)]
pub(crate) struct Ranges {
    capacity: usize,
    free: Vec<(usize, usize)>,
    used: usize,
}
impl Ranges {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
            used: 0,
        }
    }
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn used(&self) -> usize {
        self.used
    }
    pub(crate) fn available(&self) -> usize {
        self.capacity - self.used
    }
    pub(crate) fn fit(&self, size: usize, alignment: usize) -> Option<(usize, usize)> {
        if size == 0 || alignment == 0 {
            return None;
        }
        let mut best = None;
        for (index, &(offset, length)) in self.free.iter().enumerate() {
            let Some(start) = aligned(offset, alignment) else {
                continue;
            };
            let (Some(end), Some(limit)) = (start.checked_add(size), offset.checked_add(length))
            else {
                continue;
            };
            if end <= limit {
                let waste = length - size;
                if best.is_none_or(|(_, old)| waste < old) {
                    best = Some((index, waste));
                }
            }
        }
        best
    }
    pub(crate) fn take_at(&mut self, index: usize, size: usize, alignment: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        let &(offset, length) = self.free.get(index)?;
        let start = aligned(offset, alignment)?;
        let limit = offset.checked_add(length)?;
        let end = start.checked_add(size)?;
        if end > limit {
            return None;
        }
        let used = self.used.checked_add(size)?;
        // Every check precedes mutation. Preserve split order/coalescing shape.
        let mut rest = [(0usize, 0usize); 2];
        let mut count = 0;
        if start > offset {
            rest[count] = (offset, start - offset);
            count += 1;
        }
        if limit > end {
            rest[count] = (end, limit - end);
            count += 1;
        }
        self.free
            .splice(index..index + 1, rest[..count].iter().copied());
        self.used = used;
        Some(start)
    }
    pub(crate) fn take(&mut self, size: usize, alignment: usize) -> Option<usize> {
        let (index, _) = self.fit(size, alignment)?;
        self.take_at(index, size, alignment)
    }
    pub(crate) fn give_back(&mut self, offset: usize, size: usize) -> bool {
        let Some(end) = offset.checked_add(size) else {
            return false;
        };
        let Some(used) = self.used.checked_sub(size) else {
            return false;
        };
        if size == 0 || end > self.capacity {
            return false;
        }
        let index = self.free.partition_point(|&(off, _)| off < offset);
        let previous_end = if index > 0 {
            self.free[index - 1].0.checked_add(self.free[index - 1].1)
        } else {
            Some(0)
        };
        let Some(previous_end) = previous_end else {
            return false;
        };
        if previous_end > offset || self.free.get(index).is_some_and(|&(off, _)| off < end) {
            return false;
        }
        self.free.insert(index, (offset, size));
        if index + 1 < self.free.len() && end == self.free[index + 1].0 {
            let next = self.free.remove(index + 1);
            // Both ranges were bounded by capacity before insertion.
            self.free[index].1 += next.1;
        }
        if index > 0 && previous_end == offset {
            let current = self.free.remove(index);
            self.free[index - 1].1 += current.1;
        }
        self.used = used;
        true
    }
}
pub(crate) fn aligned(value: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_reuse_aliases_deduplicate_and_scratch_capacity_is_retained() {
        let mut credit = ReuseCredit::default();
        credit.observe(7, 8);
        credit.observe(7, 8);
        credit.observe(9, 16);
        assert_eq!(credit.checked_total(), Some(24));
        credit.observe(7, 4);
        assert_eq!(credit.checked_total(), Some(20));
        let capacity = credit.sources.capacity();
        credit.clear();
        assert_eq!(credit.sources.capacity(), capacity);
        assert_eq!(credit.checked_total(), Some(0));
        credit.observe(7, usize::MAX);
        credit.observe(9, 1);
        assert_eq!(credit.checked_total(), None);
    }
    #[test]
    fn numeric_reuse_requires_ready_and_same_domain() {
        use crate::memory_admission::process_account;
        let a = process_account().unadmitted_domain(Some(8)).unwrap().key();
        let b = process_account().unadmitted_domain(Some(8)).unwrap().key();
        assert!(reusable_for(true, None, None));
        assert!(reusable_for(true, Some(&a), Some(&a)));
        assert!(!reusable_for(false, None, None));
        assert!(!reusable_for(false, Some(&a), Some(&a)));
        assert!(!reusable_for(true, Some(&a), Some(&b)));
        assert!(!reusable_for(true, None, Some(&a)));
        assert!(!reusable_for(true, Some(&a), None));
    }
    #[test]
    fn numeric_ranges_split_align_coalesce_and_reuse() {
        let mut ranges = Ranges::new(32);
        let a = ranges.take(5, 4).unwrap();
        let b = ranges.take(5, 4).unwrap();
        let c = ranges.take(5, 4).unwrap();
        assert_eq!((a, b, c), (0, 8, 16));
        assert_eq!(ranges.used(), 15);
        assert!(ranges.give_back(b, 5));
        assert!(ranges.give_back(a, 5));
        assert!(ranges.give_back(c, 5));
        assert_eq!(ranges.free, vec![(0, 32)]);
        assert_eq!(ranges.available(), 32);
        assert_eq!(ranges.take(8, 16), Some(0));
        assert_eq!(ranges.take(8, 16), Some(16));
    }
    #[test]
    fn numeric_ranges_invalid_operations_leave_state_unchanged() {
        let mut ranges = Ranges::new(32);
        let offset = ranges.take(8, 4).unwrap();
        let before = (ranges.free.clone(), ranges.used());
        for (off, size) in [(usize::MAX, 2), (0, 0), (31, 2), (0, 9), (4, 8)] {
            assert!(!ranges.give_back(off, size));
            assert_eq!((ranges.free.clone(), ranges.used()), before);
        }
        for (size, align) in [(usize::MAX, 2), (8, 0), (0, 4)] {
            assert_eq!(ranges.take(size, align), None);
            assert_eq!((ranges.free.clone(), ranges.used()), before);
        }
        assert!(ranges.give_back(offset, 8));
        assert!(!ranges.give_back(offset, 8));
        assert_eq!(aligned(usize::MAX, 1), Some(usize::MAX));
        assert_eq!(aligned(usize::MAX, 2), None);
    }
    // Original fixture retained, now exercising production Ranges rather than a copy.
    #[test]
    fn ranges_split_align_and_coalesce() {
        let mut c = Ranges::new(1 << 20);
        let a = c.take(1000, 256).unwrap();
        let b = c.take(1000, 256).unwrap();
        let d = c.take(1000, 256).unwrap();
        assert_eq!((a, b, d), (0, 1024, 2048));
        assert_eq!(c.used, 3000);
        c.give_back(b, 1000);
        assert_eq!(c.free.len(), 2, "{:?}", c.free);
        c.give_back(a, 1000);
        c.give_back(d, 1000);
        assert_eq!(c.free, vec![(0, 1 << 20)]);
        assert_eq!(c.used, 0);
        let x = c.take(1000, 256).unwrap();
        assert_eq!(x, 0);
        let y = c.take(3000, 4096).unwrap();
        assert_eq!(y, 4096);
    }
}
