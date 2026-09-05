//! Sequence slots: the serving side of content-addressed reuse (issue #677,
//! `design/docs/sequence-reuse-design.md`).
//!
//! A slot holds one [`lloom::nn::Sequence`] between requests. A request arrives
//! as a whole token sequence and is routed to the slot whose cached tokens share
//! the longest prefix with it — llama.cpp's `get_available_slot`
//! (`tools/server/server-context.cpp:1547`): the best similarity above a
//! threshold wins, otherwise the least recently used idle slot is taken fresh.
//! The engine never sees a client: what crosses into `lloom` is `(tokens, a
//! sequence)`, and [`lloom::nn::Session::prepare`] decides how much of it is
//! reused. A continuation, a request that diverges from an earlier one, and a
//! fresh request are the same routing at different prefix lengths.
//!
//! The pool owns no model borrow — a sequence is separable from the
//! [`lloom::nn::Session`] that drives it — so a server holds a pool for the
//! process lifetime and builds a session per request.

use lloom::nn::Sequence;

/// llama.cpp's default `slot_prompt_similarity` (`common/common.h:695`): a
/// slot is reused only if its common prefix covers at least this fraction of
/// the request.
pub const DEFAULT_SIMILARITY: f32 = 0.1;

/// How a request was routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Routed {
    /// The slot taken.
    pub slot: usize,
    /// The common prefix between the request and the slot's cached tokens.
    pub common_prefix: usize,
    /// Whether the slot was chosen for its prefix (`true`) or as the idle
    /// least-recently-used slot with nothing worth reusing (`false`).
    pub by_prefix: bool,
}

struct Entry<C> {
    /// `None` while the sequence is checked out to a request.
    seq: Option<Sequence<C>>,
    last_used: u64,
}

/// A fixed set of sequence slots with prefix-based routing.
pub struct SlotPool<C> {
    slots: Vec<Entry<C>>,
    clock: u64,
    similarity: f32,
}

impl<C> SlotPool<C> {
    /// `n` slots over the caches `make` builds, at [`DEFAULT_SIMILARITY`].
    pub fn new(n: usize, mut make: impl FnMut() -> C) -> Self {
        let slots = (0..n)
            .map(|_| Entry {
                seq: Some(Sequence::new(make())),
                last_used: 0,
            })
            .collect();
        Self {
            slots,
            clock: 0,
            similarity: DEFAULT_SIMILARITY,
        }
    }

    /// A pool over sequences the caller already holds — restoring a server's
    /// slots, or a test seeding token logs.
    pub fn from_sequences(seqs: Vec<Sequence<C>>) -> Self {
        let slots = seqs
            .into_iter()
            .enumerate()
            .map(|(i, seq)| Entry {
                seq: Some(seq),
                last_used: i as u64,
            })
            .collect();
        Self {
            slots,
            clock: 0,
            similarity: DEFAULT_SIMILARITY,
        }
    }

    /// The reuse threshold: a slot whose common prefix covers less than this
    /// fraction of the request is not chosen for its prefix. `0.0` disables
    /// prefix routing and every request takes the LRU idle slot.
    pub fn with_similarity(mut self, similarity: f32) -> Self {
        self.similarity = similarity;
        self
    }

    /// Slot count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the pool has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Slots not checked out.
    pub fn idle(&self) -> usize {
        self.slots.iter().filter(|e| e.seq.is_some()).count()
    }

    /// Route `full` to a slot and check its sequence out, or `None` when every
    /// slot is busy. The caller drives the sequence and hands it back with
    /// [`Self::release`]; the prefix reuse itself is [`lloom::nn::Session::prepare`]'s.
    pub fn acquire(&mut self, full: &[u32]) -> Option<(Routed, Sequence<C>)> {
        let mut best: Option<(usize, usize, f32)> = None;
        if self.similarity > 0.0 && !full.is_empty() {
            for (i, e) in self.slots.iter().enumerate() {
                let Some(seq) = &e.seq else { continue };
                if seq.tokens().is_empty() {
                    continue;
                }
                let lcp = seq.common_prefix(full);
                let sim = lcp as f32 / full.len() as f32;
                let better = match best {
                    None => sim > self.similarity,
                    Some((_, _, b)) => sim > b && sim > self.similarity,
                };
                if better {
                    best = Some((i, lcp, sim));
                }
            }
        }
        let (slot, lcp, by_prefix) = match best {
            Some((i, lcp, _)) => (i, lcp, true),
            None => {
                let i = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.seq.is_some())
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(i, _)| i)?;
                (i, 0, false)
            }
        };
        self.clock += 1;
        let entry = &mut self.slots[slot];
        entry.last_used = self.clock;
        let seq = entry.seq.take()?;
        Some((
            Routed {
                slot,
                common_prefix: lcp,
                by_prefix,
            },
            seq,
        ))
    }

    /// Hand a sequence back to its slot after the request that drove it.
    pub fn release(&mut self, slot: usize, seq: Sequence<C>) {
        self.clock += 1;
        let entry = &mut self.slots[slot];
        entry.seq = Some(seq);
        entry.last_used = self.clock;
    }

    /// The cached token log of an idle slot, for a caller reporting reuse.
    pub fn tokens(&self, slot: usize) -> Option<&[u32]> {
        self.slots.get(slot)?.seq.as_ref().map(|s| s.tokens())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool of `()`-cache sequences with the given token logs, slot `i`
    /// released `i`-th so the LRU order is the listing order. Routing is a
    /// function of the logs alone; a decode over a real cache is `lloom`'s
    /// own tests' business.
    fn pool_with(logs: &[&[u32]]) -> SlotPool<()> {
        SlotPool::from_sequences(
            logs.iter()
                .map(|log| Sequence::with_tokens((), log.to_vec()))
                .collect(),
        )
    }

    #[test]
    fn a_request_routes_to_the_slot_with_the_longest_common_prefix() {
        let mut pool = pool_with(&[&[1, 2, 3, 4], &[1, 2, 9, 9, 9], &[7, 7]]);
        let (r, seq) = pool.acquire(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(r.slot, 0);
        assert_eq!(r.common_prefix, 4);
        assert!(r.by_prefix);
        pool.release(r.slot, seq);

        let (r, seq) = pool.acquire(&[1, 2, 9, 9]).unwrap();
        assert_eq!((r.slot, r.common_prefix, r.by_prefix), (1, 4, true));
        pool.release(r.slot, seq);
    }

    #[test]
    fn below_the_threshold_the_lru_idle_slot_is_taken_fresh() {
        let mut pool = pool_with(&[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 9]]);
        // One token of twelve in common: 0.083 < 0.1, so no slot is reused.
        let req: Vec<u32> = std::iter::once(1)
            .chain(std::iter::repeat_n(0, 11))
            .collect();
        let (r, seq) = pool.acquire(&req).unwrap();
        assert!(!r.by_prefix);
        assert_eq!(r.common_prefix, 0);
        // Slot 0 was released first in `pool_with`, so it is the LRU.
        assert_eq!(r.slot, 0);
        pool.release(r.slot, seq);
    }

    #[test]
    fn a_busy_slot_is_never_routed_to_and_an_exhausted_pool_says_so() {
        let mut pool = pool_with(&[&[1, 2, 3]]);
        let (r, seq) = pool.acquire(&[1, 2, 3, 4]).unwrap();
        assert!(r.by_prefix);
        assert_eq!(pool.idle(), 0);
        assert!(pool.acquire(&[1, 2, 3, 4]).is_none());
        pool.release(r.slot, seq);
        assert_eq!(pool.idle(), 1);
    }

    #[test]
    fn similarity_zero_disables_prefix_routing() {
        let mut pool = pool_with(&[&[1, 2, 3, 4], &[9]]).with_similarity(0.0);
        let (r, _) = pool.acquire(&[1, 2, 3, 4, 5]).unwrap();
        assert!(!r.by_prefix);
        assert_eq!(r.slot, 0);
    }
}
