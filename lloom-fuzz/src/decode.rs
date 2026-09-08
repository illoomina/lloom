//! Byte input → operation sequence, one decoder per target.
//!
//! The fuzzer mutates bytes; the models consume `Op`s. Each decoder is total
//! (every byte string decodes to a valid, possibly empty, sequence) so the
//! campaign never rejects an input for shape, and it is *positional* — byte `k`
//! always decides the same thing given the bytes before it — so a havoc
//! mutation of one byte changes one operation, which is what makes the
//! coverage feedback attributable and the minimiser's deletions meaningful.
//!
//! The weights the proptest generators use (`prop_oneof!` with weights) are
//! reproduced as tag ranges: a tag byte in `0..W` picks the variant whose weight
//! band contains it, so the two generators draw the same marginal distribution
//! and the comparison in the report is between *search strategies*, not
//! between alphabets.

/// The most operations one input decodes to. Above the proptest sequence
/// lengths (80–160) so a long input can reach deeper states, below the point
/// where a Metal target's per-op cost makes an execution slow.
pub const MAX_OPS: usize = 160;

/// A cursor over the input. Every read past the end yields zero, which keeps
/// the decoders total without a length check per field.
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    pub fn u8(&mut self) -> u8 {
        let b = self.bytes.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    pub fn u16(&mut self) -> u16 {
        let lo = self.u8() as u16;
        let hi = self.u8() as u16;
        lo | (hi << 8)
    }

    /// A value in `0..n` (`0` when `n == 0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        self.u8() as usize % n
    }

    /// A value in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi.saturating_sub(lo) + 1)
    }

    /// A 16-bit value in `lo..=hi`, for ranges wider than a byte.
    pub fn range16(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.u16() as usize) % (hi.saturating_sub(lo) + 1)
    }

    /// A weighted variant pick: `weights[i]` is the band width of variant `i`.
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        let mut t = (self.u8() as u32) % total.max(1);
        for (i, w) in weights.iter().enumerate() {
            if t < *w {
                return i;
            }
            t -= w;
        }
        weights.len() - 1
    }
}

/// The token ring's alphabet (`lloom::generation::token_ring_fuzz::Op`), with
/// the proptest weights 6/4/3/3/2/2.
pub fn token_ring(bytes: &[u8]) -> Vec<lloom::generation::token_ring_fuzz::Op> {
    use lloom::generation::token_ring_fuzz::Op;
    let mut r = Reader::new(bytes);
    let mut ops = Vec::new();
    while r.remaining() > 0 && ops.len() < MAX_OPS {
        ops.push(match r.weighted(&[6, 4, 3, 3, 2, 2]) {
            0 => Op::PublishToken { id: r.u16() },
            1 => Op::StageChunk {
                seed: r.u16(),
                len: r.u8(),
            },
            2 => Op::ReadFlag,
            3 => Op::ReadInput { width: r.u8() },
            4 => Op::PublishEos { which: r.u8() },
            _ => Op::NextTurn,
        });
    }
    ops
}

/// The `KvSlot` alphabet (`lloom::models::lfm2::kv_slot_fuzz::SlotOp`) plus
/// the capacity the sequence runs at (the proptest draws `4..=64`). Weights
/// 8/6/1/2/1/1, and every window fits the capacity by construction (`n, k <= 4
/// <= capacity`), which is what the proptest's filter enforced.
pub fn kv_slot(bytes: &[u8]) -> (usize, Vec<lloom::models::lfm2::kv_slot_fuzz::SlotOp>) {
    use lloom::models::lfm2::kv_slot_fuzz::SlotOp;
    let mut r = Reader::new(bytes);
    let capacity = r.range(4, 64);
    let mut ops = Vec::new();
    while r.remaining() > 0 && ops.len() < MAX_OPS {
        ops.push(match r.weighted(&[8, 6, 1, 2, 1, 1]) {
            0 => SlotOp::Append { n: r.range(1, 4) },
            1 => {
                let k = r.range(1, 4);
                SlotOp::Speculate {
                    k,
                    accept: r.range(0, k),
                }
            }
            2 => SlotOp::RewindPastStart,
            3 => SlotOp::FillToCapacityThenOne,
            4 => SlotOp::Grow { by: r.range(1, 8) },
            _ => SlotOp::Reset {
                heads: [1usize, 2, 4][r.below(3)],
            },
        });
    }
    (capacity, ops)
}

/// The buffer pool's alphabet (`buffer_pool_fuzz::Op`), sizes drawn the way the
/// proptest draws them (repeated buckets 8 : nearby 3 : large 1) and the eleven
/// variants at 6/5/3/3/3/1/1/1/1/1/1.
#[cfg(feature = "metal")]
pub fn buffer_pool(bytes: &[u8]) -> Vec<lloom_tensor::kernels::metal::buffer_pool_fuzz::Op> {
    use lloom_tensor::kernels::metal::buffer_pool_fuzz::Op;
    fn size(r: &mut Reader) -> usize {
        match r.weighted(&[8, 3, 1]) {
            0 => [512usize, 1024, 2048, 4096, 8192][r.below(5)],
            1 => r.range16(1, 16383),
            _ => 64 * 1024 + r.range16(0, 1024 * 1024 - 64 * 1024 - 1),
        }
    }
    let mut r = Reader::new(bytes);
    let mut ops = Vec::new();
    while r.remaining() > 0 && ops.len() < MAX_OPS {
        ops.push(match r.weighted(&[6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1]) {
            0 => Op::Alloc { size: size(&mut r) },
            1 => Op::DropLive {
                index: r.u16() as usize,
            },
            2 => Op::CommitEpoch,
            3 => Op::MarkCompleted {
                back: r.u8() as usize,
            },
            4 => Op::Drain,
            5 => Op::Trim,
            6 => Op::SetFreeBudget {
                bytes: [0usize, 4096, 64 * 1024, 1024 * 1024][r.below(4)],
            },
            7 => Op::SetResidualCap {
                limit: r.range16(0, 4 * 1024 * 1024 - 1),
                planned: r.range16(0, 1024 * 1024 - 1),
            },
            8 => Op::ClearResidualCap,
            9 => Op::CheckResidual { size: size(&mut r) },
            _ => Op::ResetCounters,
        });
    }
    ops
}

/// The command queue's alphabet (`commands_fuzz::Op`), 6/3/2/2/1/1.
#[cfg(feature = "metal")]
pub fn commands(bytes: &[u8]) -> Vec<lloom_tensor::kernels::metal::commands_fuzz::Op> {
    use lloom_tensor::kernels::metal::commands_fuzz::Op;
    let mut r = Reader::new(bytes);
    let mut ops = Vec::new();
    // The proptest runs 1..40 ops; every op here submits real command buffers
    // and `Settle` sleeps, so the cap is the proptest's rather than MAX_OPS.
    while r.remaining() > 0 && ops.len() < 40 {
        ops.push(match r.weighted(&[6, 3, 2, 2, 1, 1]) {
            0 => Op::Encode {
                dispatches: r.range(1, 23),
            },
            1 => Op::FlushAndPrune,
            2 => Op::Flush,
            3 => Op::Settle,
            4 => Op::FlushAndWaitCurrent,
            _ => Op::FlushAndWait,
        });
    }
    ops
}

/// The scratch planner's fixture and alphabet (`scratch_fuzz::Op`): geometry,
/// layer count, policy and layout from the first bytes, then ops at
/// 10/4/1/2/2.
#[cfg(feature = "metal")]
pub struct ScratchCase {
    pub geometry: lloom_tensor::kernels::metal::scratch::PartialsGeometry,
    pub layers: usize,
    pub sizing: lloom_tensor::kernels::metal::scratch::Sizing,
    pub layout: lloom_tensor::kernels::metal::scratch::ScratchLayout,
    pub ops: Vec<lloom_tensor::kernels::metal::scratch_fuzz::Op>,
}

#[cfg(feature = "metal")]
pub fn scratch(bytes: &[u8]) -> ScratchCase {
    use lloom_tensor::kernels::metal::scratch::{
        PartialsGeometry, ScratchLayout, Sizing, BUCKET_LADDER,
    };
    use lloom_tensor::kernels::metal::scratch_fuzz::Op;
    const MAX_CONTEXT: usize = 131_072;
    let mut r = Reader::new(bytes);
    let n_heads = [1usize, 8, 32][r.below(3)];
    let head_dim = [32usize, 64, 96, 128][r.below(4)];
    let page_size = match r.weighted(&[6, 2, 1]) {
        0 => 256,
        1 => [64usize, 128, 512, 1024][r.below(4)],
        _ => 100,
    };
    let geometry = PartialsGeometry {
        n_heads,
        head_dim,
        page_size,
        batch: 1,
    };
    let layers = r.range(1, 32);
    let sizing = Sizing::ALL[r.below(Sizing::ALL.len())];
    let layout = if r.below(2) == 0 {
        ScratchLayout::Planes
    } else {
        ScratchLayout::Interleaved
    };
    let mut ops = Vec::new();
    while r.remaining() > 0 && ops.len() < MAX_OPS {
        ops.push(match r.weighted(&[10, 4, 1, 2, 2]) {
            0 => Op::Append {
                n: match r.weighted(&[6, 3, 1]) {
                    0 => 1,
                    1 => r.range(1, 63),
                    _ => r.range16(1, 4095),
                },
            },
            1 => {
                let boundary = match r.below(4) {
                    0 => BUCKET_LADDER[r.below(BUCKET_LADDER.len())],
                    1 => MAX_CONTEXT,
                    2 => 16_384,
                    _ => r.range16(1, 511) * 256,
                };
                Op::SeekNear {
                    boundary,
                    delta: (r.below(5) as i8) - 2,
                }
            }
            2 => Op::Reset,
            3 => Op::Repolicy {
                sizing: Sizing::ALL[r.below(Sizing::ALL.len())],
            },
            _ => Op::Relayout {
                layout: if r.below(2) == 0 {
                    ScratchLayout::Planes
                } else {
                    ScratchLayout::Interleaved
                },
            },
        });
    }
    ScratchCase {
        geometry,
        layers,
        sizing,
        layout,
        ops,
    }
}

/// The paged pool's alphabet (`crate::kv_pages_model::Op`), with the fixture.
#[cfg(feature = "metal")]
pub fn kv_pages(
    bytes: &[u8],
) -> (
    crate::kv_pages_model::Fixture,
    Vec<crate::kv_pages_model::Op>,
) {
    use crate::kv_pages_model::{Fixture, Op};
    let mut r = Reader::new(bytes);
    let fixture = Fixture {
        page_size: [1usize, 2, 4, 8, 256][r.below(5)],
        pool_pages: r.range(1, 24),
        max_seqs: r.range(1, 4),
        table_capacity_pages: r.range(1, 16),
    };
    let mut ops = Vec::new();
    while r.remaining() > 0 && ops.len() < MAX_OPS {
        ops.push(match r.weighted(&[8, 3, 2, 3, 3, 3, 3, 2, 2, 1]) {
            0 => Op::Append {
                seq: r.u8(),
                n: r.range(1, 9),
            },
            1 => Op::Rewind {
                seq: r.u8(),
                n: r.range(1, 9),
            },
            2 => Op::Share {
                src: r.u8(),
                dst: r.u8(),
                pages: r.range(1, 4),
            },
            3 => Op::Release { seq: r.u8() },
            4 => Op::Commit,
            5 => Op::Complete { back: r.u8() },
            6 => Op::Drain,
            7 => Op::Grow {
                extra: r.range(1, 8),
            },
            8 => Op::Table {
                seq: r.u8(),
                k: r.range(1, 3),
            },
            _ => Op::RecycleLive { seq: r.u8() },
        });
    }
    (fixture, ops)
}

/// The command-buffer liveness alphabet (`crate::cb_liveness::Op`), weights
/// 5/6/4/2/2/2/3/1/1: copies and drops dominate, so a drop of a buffer whose
/// copy is in flight is the common case rather than the rare one.
#[cfg(feature = "metal")]
pub fn cb_liveness(bytes: &[u8]) -> Vec<crate::cb_liveness::Op> {
    use crate::cb_liveness::Op;
    let mut r = Reader::new(bytes);
    let mut ops = Vec::new();
    // Every op reaches a real queue and `Settle` sleeps: capped like `commands`.
    while r.remaining() > 0 && ops.len() < 48 {
        ops.push(match r.weighted(&[5, 6, 4, 2, 2, 2, 3, 1, 1]) {
            0 => Op::Alloc {
                size: [256usize, 1024, 4096, 16384, 65536][r.below(5)],
            },
            1 => Op::Copy {
                index: r.u8() as usize,
            },
            2 => Op::Drop {
                index: r.u8() as usize,
            },
            3 => Op::Flush,
            4 => Op::FlushAndPrune,
            5 => Op::FlushAndWait,
            6 => Op::Drain,
            7 => Op::Trim,
            _ => Op::Settle,
        });
    }
    ops
}
