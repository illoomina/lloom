//! `diagnostics.trace.draws`: one stderr line per host draw, in the builds
//! that carry the `profile` instruments.
//!
//! A seeded draw is a pure function of the scores it reads and its uniform,
//! so a line records both: a hash of the scores' f32 bits, their top entries,
//! the uniform's bits and the id drawn, keyed by the sampler's seed and draw
//! count. Two runs of one seeded workload then part at a line whose hash
//! differs (the scores differed) or whose hash agrees and uniform differs
//! (the draw did). The batched driver names the step and the row.

use std::cell::Cell;

thread_local! {
    static ROW: Cell<Option<(u64, u32, u32)>> = const { Cell::new(None) };
}

/// The batched step, its live-row count and the row about to be drawn, for
/// the draws on this thread until [`clear_row`].
pub(crate) fn set_row(step: u64, live: usize, k: usize) {
    ROW.with(|c| c.set(Some((step, live as u32, k as u32))));
}

pub(crate) fn clear_row() {
    ROW.with(|c| c.set(None));
}

const TOP: usize = 8;

/// What a draw read, taken before the scores become weights.
pub(super) struct Read {
    hash: u64,
    len: usize,
    top: [(u32, f32); TOP],
}

impl Read {
    pub(super) fn of(values: &[f32]) -> Self {
        // FNV-1a over the bits: an equality witness, not a digest.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut top = [(u32::MAX, f32::NEG_INFINITY); TOP];
        for (i, &v) in values.iter().enumerate() {
            hash = (hash ^ v.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
            if v > top[TOP - 1].1 {
                let mut j = TOP - 1;
                while j > 0 && v > top[j - 1].1 {
                    top[j] = top[j - 1];
                    j -= 1;
                }
                top[j] = (i as u32, v);
            }
        }
        Self {
            hash,
            len: values.len(),
            top,
        }
    }

    pub(super) fn emit(&self, seed: u64, n: u64, u: Option<f32>, id: u32) {
        let row = match ROW.with(|c| c.get()) {
            Some((step, live, k)) => format!("step={step} live={live} row={k}"),
            None => "step=- live=- row=-".into(),
        };
        let u = u.map_or("-".into(), |u| format!("{:08x}", u.to_bits()));
        let top: Vec<String> = self
            .top
            .iter()
            .map(|(i, v)| format!("{i}:{:08x}", v.to_bits()))
            .collect();
        eprintln!(
            "[draw: seed={seed} n={n} {row} vocab={} hash={:016x} u={u} id={id} top={}]",
            self.len,
            self.hash,
            top.join(",")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Read;

    #[test]
    fn a_read_keeps_the_top_scores_in_order_and_sees_one_bit() {
        let mut v: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919) % 1000) as f32 * 0.001)
            .collect();
        v[123] = 5.0;
        v[456] = 4.0;
        let r = Read::of(&v);
        assert_eq!(r.len, 1000);
        assert_eq!(r.top[0], (123, 5.0));
        assert_eq!(r.top[1], (456, 4.0));
        assert!(r.top.windows(2).all(|w| w[0].1 >= w[1].1), "{:?}", r.top);
        let mut w = v.clone();
        w[999] = f32::from_bits(w[999].to_bits() ^ 1);
        assert_ne!(
            Read::of(&w).hash,
            r.hash,
            "a one-ulp change must move the hash"
        );
        assert_eq!(Read::of(&v).hash, r.hash);
    }
}
