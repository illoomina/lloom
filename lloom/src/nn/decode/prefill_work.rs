//! Immutable outer forwards for ordinary prefill and speculative seeds, not a
//! memory quote or an admission lease. Ragged batch pieces and direct family
//! schedules remain separate.

use super::{prefill_widths, tile_widths, DecodeOptions, IN_FLIGHT_FLOOR};
use lloom_tensor::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ForwardSpan {
    pub past: usize,
    pub width: usize,
}

#[derive(Debug)]
pub(crate) struct PrefillWork {
    start: usize,
    end: usize,
    widths: Vec<usize>,
    active_depth: Option<usize>,
}

impl PrefillWork {
    /// Resolve once at the driver's prologue using the consumer's
    /// commit behavior. Multiple forwards need pacing even on a blocking arm,
    /// whose per-token readback does not bound prompt forwards.
    pub(crate) fn resolve(
        options: &DecodeOptions,
        start: usize,
        prompt_len: usize,
        loop_commits: bool,
    ) -> Result<Self> {
        // Both width functions preserve the prompt total and emit only positive
        // widths, so a representable end also bounds every iterator prefix.
        let end = start
            .checked_add(prompt_len)
            .ok_or_else(|| lloom_tensor::Error::Msg("prefill work position overflows".into()))?;
        let widths = tile_widths(
            prefill_widths(
                prompt_len,
                options.prefill_ubatch(),
                &options.prefill_schedule,
                options.prefill_bulk,
            ),
            options.prefill_tile,
        );
        let paced = !widths.is_empty() && (loop_commits || widths.len() > 1);
        let depth = if options.prefill_pacer_depth > 0 {
            options.prefill_pacer_depth
        } else {
            options.pacer_depth()
        };
        Ok(Self {
            start,
            end,
            widths,
            active_depth: paced.then_some(depth.max(IN_FLIGHT_FLOOR)),
        })
    }

    pub(crate) fn start(&self) -> usize {
        self.start
    }

    pub(crate) fn end(&self) -> usize {
        self.end
    }

    pub(crate) fn active_depth(&self) -> Option<usize> {
        self.active_depth
    }

    pub(crate) fn forwards(&self) -> impl Iterator<Item = ForwardSpan> + '_ {
        let mut past = self.start;
        self.widths.iter().map(move |&width| {
            let span = ForwardSpan { past, width };
            past += width;
            span
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_resolves_actual_tail_and_depth_without_capacity_clamp() {
        eprintln!("EXECUTABLE {}", std::env::current_exe().unwrap().display());
        let mut options = DecodeOptions::new(0);
        options.prefill_tile = 4;
        options.in_flight_depth = 65;
        let work = PrefillWork::resolve(&options, 17, 9, false).unwrap();
        assert_eq!(work.widths, [4, 4, 1]);
        assert_eq!(work.active_depth(), Some(65));
        assert_eq!(
            work.forwards().collect::<Vec<_>>(),
            vec![
                ForwardSpan { past: 17, width: 4 },
                ForwardSpan { past: 21, width: 4 },
                ForwardSpan { past: 25, width: 1 },
            ]
        );
        assert_eq!(work.end(), 26);
    }

    #[test]
    fn tile_zero_preserves_spark_then_whole_remainder() {
        let mut options = DecodeOptions::new(0);
        options.prefill_ubatch = 3;
        let work = PrefillWork::resolve(&options, 5, 11, false).unwrap();
        assert_eq!(work.widths, [3, 8]);
        assert_eq!(work.active_depth(), Some(2));
        assert_eq!(work.end(), 16);
        options.prefill_tile = 4;
        assert_eq!(
            PrefillWork::resolve(&options, 5, 11, false).unwrap().widths,
            [3, 4, 4]
        );
    }

    #[test]
    fn schedule_one_token_tail_and_floor_are_not_family_chunk() {
        let mut options = DecodeOptions::new(0);
        options.prefill_schedule = Some(vec![0, 3]);
        options.prefill_bulk = 4;
        options.prefill_pacer_depth = 1;
        let work = PrefillWork::resolve(&options, 17, 9, false).unwrap();
        assert_eq!(work.widths, [3, 4, 1, 1]);
        assert_eq!(work.active_depth(), Some(2));
        assert_eq!(
            work.forwards().collect::<Vec<_>>(),
            vec![
                ForwardSpan { past: 17, width: 3 },
                ForwardSpan { past: 20, width: 4 },
                ForwardSpan { past: 24, width: 1 },
                ForwardSpan { past: 25, width: 1 },
            ]
        );
        assert_eq!(work.end(), 26);
    }

    #[test]
    fn one_shot_pacing_empty_and_position_overflow() {
        let mut options = DecodeOptions::new(0);
        options.in_flight_depth = 7;
        let work = PrefillWork::resolve(&options, 5, 9, false).unwrap();
        assert_eq!(work.widths, [9]);
        assert_eq!(work.active_depth(), None);
        assert_eq!(
            PrefillWork::resolve(&options, 5, 9, true)
                .unwrap()
                .active_depth(),
            Some(7)
        );
        options.prefill_pacer_depth = 8;
        assert_eq!(
            PrefillWork::resolve(&options, 5, 9, true)
                .unwrap()
                .active_depth(),
            Some(8)
        );
        let empty = PrefillWork::resolve(&options, usize::MAX, 0, true).unwrap();
        assert!(empty.widths.is_empty());
        assert_eq!(empty.active_depth(), None);
        assert_eq!(empty.start(), usize::MAX);
        assert_eq!(empty.end(), usize::MAX);
        assert!(PrefillWork::resolve(&options, usize::MAX, 1, false).is_err());
        let boundary = PrefillWork::resolve(&options, usize::MAX - 1, 1, true).unwrap();
        assert_eq!(
            boundary.forwards().collect::<Vec<_>>(),
            vec![ForwardSpan {
                past: usize::MAX - 1,
                width: 1
            },]
        );
        assert_eq!(boundary.end(), usize::MAX);
    }
}
