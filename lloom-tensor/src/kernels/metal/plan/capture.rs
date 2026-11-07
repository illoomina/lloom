//! Recording the dispatches an op library call makes, without running them
//! (`design/docs/planned-step-design.md` §6.1).
//!
//! [`Capture`] is an executor: installed on a queue, it takes each dispatch's
//! pipeline, arguments and grid from the encoder's hooks and suppresses the
//! dispatch, so a lowering puts nothing on the GPU.

use std::collections::HashMap;
use std::sync::Arc;

use super::list::{BindTable, BufferRef};
use crate::kernels::metal::{
    Buffer, ComputePipeline, DispatchAction, DispatchRecord, Executor, Grid, Size3, Slot,
};

/// One argument of a captured dispatch, in the order the encoder received it.
#[derive(Clone, Debug)]
pub enum CapturedArg {
    Bytes {
        index: u32,
        bytes: Vec<u8>,
    },
    /// The encoder could not expose the argument's content (see
    /// [`Executor::will_set_bytes`]).
    Opaque {
        index: u32,
    },
    /// `offset` is from the start of `buffer`'s handle, not of its
    /// allocation: the encoder's own offset less the handle's `base_offset`.
    Buffer {
        index: u32,
        buffer: Buffer,
        offset: usize,
        write: bool,
    },
    ThreadgroupMemory {
        index: u32,
        len: u32,
    },
}

/// One dispatch an op made while a [`Capture`] was installed.
#[derive(Clone, Debug)]
pub struct CapturedCommand {
    pub pipeline: ComputePipeline,
    pub args: Vec<CapturedArg>,
    pub grid: Grid,
    pub threads: Size3,
}

/// What a capture saw that a command list cannot hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureFault {
    /// A dispatch arrived with no pipeline set since the capture began.
    NoPipeline,
    /// Arguments were set after the last dispatch, so no command holds them.
    Leftover(usize),
    /// A bound handle could not be retained.
    Retain(String),
    /// A bound offset lay before its handle's start.
    Offset { index: u32 },
}

#[derive(Default)]
struct State {
    pipeline: Option<ComputePipeline>,
    args: Vec<CapturedArg>,
    commands: Vec<CapturedCommand>,
    fault: Option<CaptureFault>,
}

impl State {
    fn fault(&mut self, fault: CaptureFault) {
        self.fault.get_or_insert(fault);
    }
}

/// An executor that records every dispatch and encodes none of them.
#[derive(Default)]
pub struct Capture {
    state: parking_lot::Mutex<State>,
}

impl Capture {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The dispatches captured since the last call, in order, and a fresh
    /// start: the next dispatch must set its own pipeline.
    pub fn take(&self) -> Result<Vec<CapturedCommand>, CaptureFault> {
        let mut s = self.state.lock();
        let leftover = s.args.len();
        let fault = s.fault.take();
        let commands = std::mem::take(&mut s.commands);
        s.args.clear();
        s.pipeline = None;
        match fault {
            Some(fault) => Err(fault),
            None if leftover > 0 => Err(CaptureFault::Leftover(leftover)),
            None => Ok(commands),
        }
    }
}

impl Executor for Capture {
    fn dispatch_action(
        &self,
        record: &DispatchRecord,
    ) -> Result<DispatchAction, crate::kernels::MetalKernelError> {
        let mut s = self.state.lock();
        match s.pipeline.clone() {
            Some(pipeline) => {
                let args = std::mem::take(&mut s.args);
                s.commands.push(CapturedCommand {
                    pipeline,
                    args,
                    grid: record.grid,
                    threads: record.threads_per_threadgroup,
                });
            }
            // Recorded as a capture fault, as before; the plan build refuses
            // on it. Not an executor refusal of the eager step.
            None => s.fault(CaptureFault::NoPipeline),
        }
        Ok(DispatchAction::Suppress)
    }

    fn will_bind_buffer(&self, index: usize, buffer: &Buffer, offset: usize, is_output: bool) {
        let mut s = self.state.lock();
        let Some(offset) = offset.checked_sub(buffer.base_offset()) else {
            s.fault(CaptureFault::Offset {
                index: index as u32,
            });
            return;
        };
        match buffer.try_clone() {
            Ok(buffer) => s.args.push(CapturedArg::Buffer {
                index: index as u32,
                buffer,
                offset,
                write: is_output,
            }),
            Err(e) => s.fault(CaptureFault::Retain(e.to_string())),
        }
    }

    fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        self.state.lock().pipeline = Some(pipeline.clone());
    }

    fn will_set_bytes(&self, index: usize, bytes: &[u8]) {
        let arg = if bytes.is_empty() {
            CapturedArg::Opaque {
                index: index as u32,
            }
        } else {
            CapturedArg::Bytes {
                index: index as u32,
                bytes: bytes.to_vec(),
            }
        };
        self.state.lock().args.push(arg);
    }

    fn will_set_threadgroup_memory(&self, index: usize, len: usize) {
        self.state.lock().args.push(CapturedArg::ThreadgroupMemory {
            index: index as u32,
            len: len as u32,
        });
    }
}

/// A buffer handle's identity as the encoder binds it: the allocation, and
/// the range of it the handle addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    addr: usize,
    base: usize,
    len: usize,
}

impl Key {
    fn of(buffer: &Buffer) -> Self {
        Self {
            addr: buffer.raw_addr(),
            base: buffer.base_offset(),
            len: buffer.length(),
        }
    }
}

/// What a [`BufferRef`] stands for when a table is built.
enum Binding {
    /// A slot of the pass base: bound as a view of whichever base the step
    /// runs on, so the encoder sees the same extent the eager pass did.
    Slot(Slot),
    /// A buffer bound as itself: a weight, an input, session state, an escape.
    Buffer(Buffer),
}

/// The buffers a lowering may bind, each named by the [`BufferRef`] its
/// commands carry. A bind of anything else is a buffer the declaration did
/// not name, and refuses the node.
#[derive(Default)]
pub struct Buffers {
    refs: HashMap<Key, BufferRef>,
    bindings: Vec<Binding>,
}

impl Buffers {
    /// The slots of `base`, as refs `0..slots.len()` in slot order.
    pub fn with_slots(base: &Buffer, slots: &[Slot]) -> Self {
        let mut buffers = Self::default();
        for slot in slots {
            let key = Key {
                addr: base.raw_addr(),
                base: base.base_offset() + slot.offset,
                len: slot.size,
            };
            buffers.push(key, Binding::Slot(slot.clone()));
        }
        buffers
    }

    fn push(&mut self, key: Key, binding: Binding) -> BufferRef {
        let at = BufferRef(self.bindings.len() as u32);
        self.refs.insert(key, at);
        self.bindings.push(binding);
        at
    }

    /// Name `buffer`, bound as itself. Declaring the same handle twice
    /// returns its first ref.
    pub fn declare(&mut self, buffer: &Buffer) -> BufferRef {
        let key = Key::of(buffer);
        if let Some(&at) = self.refs.get(&key) {
            return at;
        }
        self.push(key, Binding::Buffer(buffer.clone()))
    }

    /// The ref a bind of `buffer` resolves to.
    pub fn get(&self, buffer: &Buffer) -> Option<BufferRef> {
        self.refs.get(&Key::of(buffer)).copied()
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// A table binding every ref: slots as views of `base`, the rest as
    /// declared.
    pub fn table(&self, base: &Buffer) -> BindTable {
        let mut table = BindTable::with_len(self.bindings.len());
        for (i, binding) in self.bindings.iter().enumerate() {
            let buffer = match binding {
                Binding::Slot(slot) => base.view(slot.offset, slot.size),
                Binding::Buffer(buffer) => buffer.clone(),
            };
            table.set(BufferRef(i as u32), buffer);
        }
        table
    }
}
