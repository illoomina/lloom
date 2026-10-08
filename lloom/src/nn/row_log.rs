//! Per-position values a forward retains for a drafter until it takes
//! them (a verify window's probe: Qwen4-exp's carries and taps, a
//! `Decoder`'s `h_nextn`): one `[1, rows, w]` part per pass, in position
//! order, concatenated on take.

use lloom_tensor::{Result, Tensor};

/// Per-position values a forward retains for a drafter until it takes them:
/// one `[1, rows, w]` part per pass, in position order.
#[derive(Debug, Clone, Default)]
pub struct RowLog {
    parts: Vec<Tensor>,
}

impl RowLog {
    pub fn push(&mut self, part: Tensor) {
        self.parts.push(part);
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn clear(&mut self) {
        self.parts.clear();
    }

    pub fn rows(&self) -> usize {
        self.parts.iter().map(|t| t.dim(1).unwrap_or(0)).sum()
    }

    /// The retained rows concatenated along the sequence, `None` if none.
    pub fn take(&mut self) -> Result<Option<Tensor>> {
        let parts = std::mem::take(&mut self.parts);
        match parts.len() {
            0 => Ok(None),
            1 => Ok(parts.into_iter().next()),
            _ => Ok(Some(Tensor::cat(&parts, 1)?)),
        }
    }

    /// Keep the first `rows` rows: whole parts, the last one narrowed.
    pub fn keep_first(&mut self, rows: usize) -> Result<()> {
        let mut left = rows;
        let mut kept = Vec::with_capacity(self.parts.len());
        for t in std::mem::take(&mut self.parts) {
            if left == 0 {
                break;
            }
            let n = t.dim(1)?;
            kept.push(if n <= left { t } else { t.narrow(1, 0, left)? });
            left -= n.min(left);
        }
        self.parts = kept;
        Ok(())
    }
}
