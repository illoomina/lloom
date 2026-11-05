//! Encoding a command list through the encoder the eager path drives
//! (`design/docs/planned-step-design.md` §6.2, E1).

use objc2_metal::MTLSize;

use super::list::{apply_bytes, offset_at, Arg, BindTable, CommandList, StepValues};
use crate::kernels::metal::{Commands, ComputeCommandEncoder, Grid, Size3};
use crate::kernels::MetalKernelError;

fn mtl(size: Size3) -> MTLSize {
    MTLSize {
        width: size.width,
        height: size.height,
        depth: size.depth,
    }
}

/// Encodes a [`CommandList`] one command at a time, each under its own
/// `Commands::command_encoder` guard as every `call_*` takes one. Barriers,
/// cross-encoder fences, command-buffer cuts, the EOS gate's binding and
/// both Metal APIs are the encoder's, exactly as for the eager dispatches the
/// list stands for.
///
/// Holds the scratch a patched bytes argument is assembled in, sized once
/// from the list, so a step allocates nothing.
pub struct PlanEncoder {
    scratch: Vec<u8>,
}

impl PlanEncoder {
    pub fn new(list: &CommandList) -> Self {
        Self {
            scratch: vec![0; list.max_bytes()],
        }
    }

    /// Encode `list` onto `commands`, its patched fields evaluated at
    /// `values`, its buffers taken from `table`.
    ///
    /// # Safety
    /// Every buffer `table` holds stays alive, and the host does not write
    /// it, until every command buffer this call encodes into has completed:
    /// the obligation each `call_*` places on its caller
    /// ([`crate::kernels::utils::EncoderParam`]).
    pub unsafe fn encode(
        &mut self,
        list: &CommandList,
        table: &BindTable,
        values: &StepValues,
        commands: &Commands,
    ) -> Result<(), MetalKernelError> {
        if values.len() < list.params() {
            return Err(MetalKernelError::InvalidInput(format!(
                "command list reads {} step parameters, {} given",
                list.params(),
                values.len()
            )));
        }
        if self.scratch.len() < list.max_bytes() {
            self.scratch.resize(list.max_bytes(), 0);
        }
        for command in list.commands() {
            let guard = commands.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_compute_pipeline_state(command.pipeline());
            let patches = list.patches(command);
            let mut next = 0;
            for (i, arg) in list.args(command).iter().enumerate() {
                let id = command.args.start + i as u32;
                match *arg {
                    Arg::Bytes { index, start, len } => {
                        let bytes = list.bytes(start, len);
                        if patches.get(next).is_some_and(|p| p.arg == id) {
                            let scratch = &mut self.scratch[..len as usize];
                            scratch.copy_from_slice(bytes);
                            while let Some(p) = patches.get(next).filter(|p| p.arg == id) {
                                apply_bytes(p, values, scratch)?;
                                next += 1;
                            }
                            encoder.set_bytes_directly(
                                index as usize,
                                len as usize,
                                scratch.as_ptr().cast(),
                            );
                        } else {
                            encoder.set_bytes_directly(
                                index as usize,
                                len as usize,
                                bytes.as_ptr().cast(),
                            );
                        }
                    }
                    Arg::Buffer {
                        index,
                        buffer,
                        mut offset,
                        write,
                    } => {
                        while let Some(p) = patches.get(next).filter(|p| p.arg == id) {
                            offset = offset_at(p, values)?;
                            next += 1;
                        }
                        let bound = table.get(buffer).ok_or_else(|| {
                            MetalKernelError::InvalidInput(format!(
                                "command list binds buffer {} the table does not hold",
                                buffer.0
                            ))
                        })?;
                        if offset > bound.length() {
                            return Err(MetalKernelError::InvalidInput(format!(
                                "offset {offset} past a {}-byte binding",
                                bound.length()
                            )));
                        }
                        // SAFETY: the caller's contract above covers every
                        // buffer the table holds.
                        unsafe {
                            if write {
                                encoder.set_output_buffer(index as usize, Some(bound), offset);
                            } else {
                                encoder.set_input_buffer(index as usize, Some(bound), offset);
                            }
                        }
                    }
                    Arg::ThreadgroupMemory { index, len } => {
                        encoder.set_threadgroup_memory_length(index as usize, len as usize);
                    }
                }
            }
            match command.grid() {
                Grid::Threadgroups(groups) => {
                    encoder.dispatch_thread_groups(mtl(groups), mtl(command.threads()))
                }
                Grid::Threads(grid) => encoder.dispatch_threads(mtl(grid), mtl(command.threads())),
            }
        }
        Ok(())
    }
}
