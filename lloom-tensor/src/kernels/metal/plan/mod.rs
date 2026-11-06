//! A planned pass's command list, and the loop that encodes it
//! (`design/docs/planned-step-design.md` §6).
//!
//! A [`CommandList`] holds Metal-level commands — a pipeline, its arguments
//! and a grid — not ops. The planning layer emits it once per plan; a planned
//! step encodes it with [`PlanEncoder`], patching the fields that follow the
//! step's parameters ([`StepValues`]) as it goes. Buffers are named through a
//! [`BindTable`], so a replaced session buffer or base is a table update, not
//! a new list.
//!
//! The encoder drives [`ComputeCommandEncoder`](super::ComputeCommandEncoder)'s
//! public methods, one `Commands::command_encoder` guard per command as every
//! `call_*` takes one. Barriers, fences, command-buffer cuts, the EOS gate and
//! both Metal APIs are therefore the encoder's own, unchanged (§6.2 there).
//!
//! A list is built by lowering each declared node through the op library on
//! a [`Lowering`]: a view of the device whose queue captures each dispatch
//! instead of running it (§6.1). [`lower_node`] lowers a node at sampled
//! parameter values and fits every field that moved into an affine patch
//! (§5); [`segments`] and [`barriers`] predict what the encoder will do with
//! the result (§6.3).

mod capture;
mod encode;
mod fit;
mod list;
mod lower;
mod predict;

#[cfg(test)]
mod capture_gpu_tests;

pub use capture::{Buffers, Capture, CaptureFault, CapturedArg, CapturedCommand};
pub use encode::PlanEncoder;
#[cfg(any(test, feature = "planned-validate"))]
pub use fit::predicts;
pub use fit::{lower_node, Node, Probe, Sweep};
pub use list::{
    Affine, Arg, ArgId, BindTable, BufferRef, Command, CommandBuilder, CommandList, Patch,
    PatchTarget, StepParam, StepValues, Width,
};
pub use lower::{same_pipeline, LoweredArg, LoweredCommand, Lowering, NodeAlloc, Refusal, Refused};
pub use predict::{barriers, segments, SegmentStart, Segments, Unpredictable};
