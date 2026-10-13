//! The default attention behind d1-omni's callbacks: compact additive key
//! masks over query tiles (`nn::attention::bounded`) for the trunk, the head
//! and the vision tower, and the conformer's relative attention for audio. A
//! caller with another policy supplies its own callbacks.
use super::{RelativeAttentionPlan, RelativeInputs, TrunkBatch};
use crate::nn::attention::bounded::MaskedAttentionPlan;
use lloom_tensor::{Result, Tensor};
use std::num::NonZeroUsize;

/// Query rows per tile: bounds one dispatch, not the work outstanding.
const QUERY_TILE: NonZeroUsize = match NonZeroUsize::new(256) {
    Some(tile) => tile,
    None => unreachable!(),
};

/// `[B,H,Q,D]` attention over `[B,Hkv,K,D]` keys under an additive `[B,1,1,K]`
/// mask; no mask is every key visible.
pub fn masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let open;
    let mask = match mask {
        Some(mask) => mask,
        None => {
            let (batch, _, keys, _) = k.dims4()?;
            open = Tensor::zeros((batch, 1, 1, keys), k.dtype(), k.device())?;
            &open
        }
    };
    let q = q.contiguous()?;
    MaskedAttentionPlan::new(&q, k, v, mask, scale, QUERY_TILE)?.execute()
}

/// The trunk's attention: each row's query segments (media queries see the
/// media, text queries see every real key) against that row's keys.
pub fn trunk(q: &Tensor, k: &Tensor, v: &Tensor, batch: &TrunkBatch, scale: f64) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(batch.dimensions().0);
    for row in 0..batch.dimensions().0 {
        let (q_row, k_row, v_row) = (
            q.narrow(0, row, 1)?,
            k.narrow(0, row, 1)?,
            v.narrow(0, row, 1)?,
        );
        let mut pieces = Vec::new();
        for (i, segment) in batch.segments().iter().enumerate() {
            if segment.batch_row != row {
                continue;
            }
            pieces.push(masked(
                &q_row.narrow(2, segment.start, segment.length)?,
                &k_row,
                &v_row,
                Some(&batch.key_mask(i)?),
                scale,
            )?);
        }
        rows.push(Tensor::cat(&pieces, 2)?);
    }
    Tensor::cat(&rows, 0)
}

/// The audio encoder's relative-position attention.
pub fn relative(inputs: RelativeInputs<'_>) -> Result<Tensor> {
    RelativeAttentionPlan::new(inputs)?.execute()
}
