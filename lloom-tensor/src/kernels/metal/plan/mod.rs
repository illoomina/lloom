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

mod encode;
mod list;

pub use encode::PlanEncoder;
pub use list::{
    Affine, Arg, ArgId, BindTable, BufferRef, Command, CommandBuilder, CommandList, Patch,
    PatchTarget, StepParam, StepValues, Width,
};
