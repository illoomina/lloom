//! What the encoder will do with a command list: where its sessions open and
//! which commands it orders behind a barrier
//! (`design/docs/planned-step-design.md` §6.3).
//!
//! Both are predictions of the encoder's own rules, computed by its own
//! functions: the cut is `CutPolicy`'s arithmetic, and each binding is
//! tested with the `read_hazards` / `write_hazards` the encoder's bind path
//! calls, over the `BoundSet`s it keeps. A test holds each prediction against
//! what the encoder did.

use super::list::{offset_at, Arg, BindTable, CommandList, StepValues};
use crate::kernels::metal::encoder::{read_hazards, write_hazards, BoundRange, BoundSet};
use crate::kernels::metal::{CutPolicy, HazardKey};
use crate::kernels::MetalKernelError;

/// The queue's state as a list begins: the dispatch count since its last
/// cut (`Commands::open_dispatches`), and whether an encoder session is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentStart {
    pub count: usize,
    pub open: bool,
}

/// Where the list's commands fall in the queue's sessions and command buffers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segments {
    /// Commands that open a new encoder session.
    pub sessions: Vec<usize>,
    /// Of those, the commands that also cut a new command buffer.
    pub cuts: Vec<usize>,
}

/// Why the segments of a list cannot be predicted from the list alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unpredictable {
    /// The node policies cut where the model marks a graph node, which a
    /// command list does not carry.
    NodePolicy,
    /// The per-kernel timing arm closes the encoder at every dispatch.
    EncoderPerDispatch,
}

/// The sessions and cuts `len` commands meet, each taking one
/// `Commands::command_encoder`, as E1 encodes them.
pub fn segments(
    len: usize,
    policy: CutPolicy,
    period: usize,
    start: SegmentStart,
) -> Result<Segments, Unpredictable> {
    if !matches!(policy, CutPolicy::DispatchCounter) {
        return Err(Unpredictable::NodePolicy);
    }
    if crate::kernels::metal::profile::encoder_per_dispatch() {
        return Err(Unpredictable::EncoderPerDispatch);
    }
    let period = period.max(1);
    let (mut count, mut open) = (start.count, start.open);
    let mut out = Segments::default();
    for i in 0..len {
        // `command_encoder`: the count before the increment decides the cut,
        // and a cut leaves one dispatch counted.
        if policy.cuts_at_dispatch(count, period) {
            out.cuts.push(i);
            open = false;
            count = 1;
        } else {
            count += 1;
        }
        if !open {
            out.sessions.push(i);
            open = true;
        }
    }
    Ok(out)
}

/// One session's hazard state, as `EncoderState` keeps it.
#[derive(Default)]
struct Hazards {
    prev_outputs: BoundSet,
    next_outputs: BoundSet,
    prev_inputs: BoundSet,
    next_inputs: BoundSet,
    needs_barrier: bool,
}

/// The commands the encoder will put a barrier before, when `list` is
/// encoded with `table` at `values`, a new session opening at each of
/// `sessions`, under hazard key `key`. A session open before the list began
/// is taken as empty: the prediction is exact from a fresh session.
pub fn barriers(
    list: &CommandList,
    table: &BindTable,
    values: &StepValues,
    key: HazardKey,
    sessions: &[usize],
) -> Result<Vec<usize>, MetalKernelError> {
    let mut out = Vec::new();
    let mut s = Hazards::default();
    for (i, command) in list.commands().iter().enumerate() {
        if sessions.contains(&i) {
            s = Hazards::default();
        }
        let patches = list.patches(command);
        for (j, arg) in list.args(command).iter().enumerate() {
            let Arg::Buffer {
                buffer,
                offset,
                write,
                ..
            } = *arg
            else {
                continue;
            };
            let id = command.args.start + j as u32;
            let mut offset = offset;
            for p in patches.iter().filter(|p| p.arg == id) {
                offset = offset_at(p, values)?;
            }
            let bound = table.get(buffer).ok_or_else(|| {
                MetalKernelError::InvalidInput(format!(
                    "command list binds buffer {} the table does not hold",
                    buffer.0
                ))
            })?;
            // The range the encoder's bind path keys: the handle's allocation,
            // the byte the kernel starts at, the handle's extent.
            let range = BoundRange {
                ptr: bound.raw_ptr() as usize,
                offset: offset + bound.base_offset(),
                len: bound.length(),
            };
            let kinds = if write {
                write_hazards(&s.prev_outputs, &s.prev_inputs, key, &range)
            } else {
                read_hazards(&s.prev_outputs, key, &range)
            };
            if !kinds.is_empty() {
                s.needs_barrier = true;
            }
            if write {
                s.next_outputs.insert(key, range);
            } else {
                s.next_inputs.insert(key, range);
            }
        }
        // `auto_barrier`'s two rotations.
        if s.needs_barrier {
            out.push(i);
            s.needs_barrier = false;
            BoundSet::rotate_after_barrier(&mut s.prev_outputs, &mut s.next_outputs);
            BoundSet::rotate_after_barrier(&mut s.prev_inputs, &mut s.next_inputs);
        } else {
            BoundSet::fold_without_barrier(&mut s.prev_outputs, &mut s.next_outputs);
            BoundSet::fold_without_barrier(&mut s.prev_inputs, &mut s.next_inputs);
        }
    }
    Ok(out)
}
