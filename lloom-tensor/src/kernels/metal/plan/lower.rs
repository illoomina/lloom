//! Lowering one declared node by calling the op library on a view of the
//! device that captures instead of dispatching
//! (`design/docs/planned-step-design.md` §6.1).
//!
//! The view runs on a queue of its own with a [`Capture`] installed, so its
//! encoders never share a command buffer, a hazard set or a fence map with a
//! live pass: a suppressed dispatch still binds, emits barriers and waits on
//! fences on its raw encoder, and none of that may land in a live stream. The
//! view's arena is built per lowering from the node's declared outputs over
//! the plan's slots, so each output an op allocates lands in its slot, and an
//! allocation the declaration did not name falls out of the arena and refuses
//! the node.

use std::sync::Arc;

use super::capture::{Buffers, Capture, CaptureFault, CapturedArg, CapturedCommand};
use super::list::BufferRef;
use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{
    observing_binds, Arena, ArenaLayout, Buffer, BufferHandle, CommandBufferKind, Commands,
    ComputePipeline, ExecutorSlot, Grid, PooledBuffer, ResidencySet, Size3, Slot, StepPlan,
};
use crate::kernels::{default_param_style, ParamStyle};
use crate::memory_admission::AllocationOrigin;
use crate::metal_backend::{MetalDevice, MetalStorage};
use crate::DType;

/// Why a node could not be lowered into commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The device has a pass or an arena step open: lowering would move the
    /// process-wide step dispatch position under it.
    LiveStep,
    /// A recorder observes binds process-wide and would record the lowering.
    Observed,
    /// The default parameter style is `Packed`, whose staged arrays bind
    /// without reaching the executor.
    PackedStyle,
    /// The op returned an error.
    Op(String),
    /// Output `alloc` names a slot the plan does not have, or one too small.
    Slot {
        alloc: usize,
    },
    /// The arena did not serve exactly the declared outputs: `offers` were
    /// made and `served` granted, for `declared` named.
    Outputs {
        declared: usize,
        offers: u64,
        served: u64,
    },
    Capture(CaptureFault),
    /// Argument `index` of command `command` set a struct the encoder cannot
    /// expose.
    Opaque {
        command: usize,
        index: u32,
    },
    /// Argument `index` of command `command` bound a buffer the declaration
    /// did not name.
    Unresolved {
        command: usize,
        index: u32,
    },
    /// The lowerings at two parameter values differ in `what`.
    Structure {
        command: usize,
        what: &'static str,
    },
    /// A bytes field moved, but is not a 4- or 8-byte scalar argument.
    Width {
        command: usize,
        index: u32,
        len: usize,
    },
    /// A field moved, but not as an exact affine function of `param`.
    NonAffine {
        command: usize,
        index: u32,
        param: u16,
    },
    /// The node's lowering read a step parameter it did not name.
    Undeclared {
        param: u16,
    },
    /// A parameter's three sample values are not distinct.
    Samples {
        param: u16,
    },
    /// The fitted field did not predict the lowering at the held-out value.
    HeldOut {
        command: usize,
        index: u32,
    },
    /// The fitted command could not be written into the list.
    List(String),
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Refused {}

/// A node's refusal, by node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub node: u32,
    pub reason: Refused,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node {}: {}", self.node, self.reason)
    }
}

impl std::error::Error for Refusal {}

/// One output a node allocates: the plan slot that holds it, its size in
/// bytes as the op requests it, and how many of the node's own dispatches
/// precede the allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeAlloc {
    pub slot: usize,
    pub size: usize,
    pub at: usize,
}

/// A captured argument with its buffer named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoweredArg {
    Bytes {
        index: u32,
        bytes: Vec<u8>,
    },
    Buffer {
        index: u32,
        buffer: BufferRef,
        offset: usize,
        write: bool,
    },
    ThreadgroupMemory {
        index: u32,
        len: u32,
    },
}

/// One captured dispatch, its buffers resolved to the lowering's refs.
#[derive(Clone, Debug)]
pub struct LoweredCommand {
    pub pipeline: ComputePipeline,
    pub args: Vec<LoweredArg>,
    pub grid: Grid,
    pub threads: Size3,
}

impl LoweredCommand {
    /// Whether two lowerings name the same compiled pipeline.
    pub fn same_pipeline(&self, other: &Self) -> bool {
        same_pipeline(&self.pipeline, &other.pipeline)
    }
}

/// Whether `a` and `b` are one compiled pipeline.
pub fn same_pipeline(a: &ComputePipeline, b: &ComputePipeline) -> bool {
    let a: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState> = a.as_ref();
    std::ptr::eq(a, b.as_ref())
}

/// The view a plan build lowers its nodes on, and the buffers they may bind.
pub struct Lowering {
    view: MetalDevice,
    commands: Arc<Commands>,
    capture: Arc<Capture>,
    base: Buffer,
    slots: Vec<Slot>,
    buffers: Buffers,
}

impl Lowering {
    /// A lowering over `slots` of `base`, which stands for the pass base the
    /// program will run on: commands name its slots, never its address.
    /// `base` is the device's own allocation, as the pass base is: an op's
    /// output in a slot of it is stamped with the device's allocation origin.
    pub fn new(device: &MetalDevice, base: Buffer, slots: Vec<Slot>) -> crate::Result<Self> {
        let origin = base
            .allocation_owner()
            .map_or(AllocationOrigin::Legacy, |owner| {
                AllocationOrigin::Domain(owner.domain_key())
            });
        device
            .declared_allocation_origin()
            .validate_reuse(&origin)?;
        let queue = device
            .metal_device()
            .new_command_queue()
            .map_err(crate::Error::wrap)?;
        let residency = Arc::new(ResidencySet::new(device.metal_device()));
        let commands = Arc::new(
            Commands::new_with(
                queue,
                residency,
                CompletionWait::Synchronize,
                CommandBufferKind::Retaining,
            )
            .map_err(crate::Error::wrap)?,
        );
        let capture = Capture::new();
        commands.set_executor(Arc::new(ExecutorSlot::Custom(capture.clone())));
        let mut view = device.clone();
        view.commands = Arc::clone(&commands);
        view.arena = Default::default();
        view.arena_recorder = Default::default();
        view.session_region = Default::default();
        view.pass_plans = Default::default();
        view.pass_state = Default::default();
        view.base_mode = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let buffers = Buffers::with_slots(&base, &slots);
        Ok(Self {
            view,
            commands,
            capture,
            base,
            slots,
            buffers,
        })
    }

    /// The device the op library is called on during a lowering.
    pub fn device(&self) -> &MetalDevice {
        &self.view
    }

    /// The ref for slot `slot` of the base.
    pub fn slot(&self, slot: usize) -> BufferRef {
        assert!(
            slot < self.slots.len(),
            "slot {slot} of {}",
            self.slots.len()
        );
        BufferRef(slot as u32)
    }

    /// Name `buffer`, bound as itself in every program built from this
    /// lowering.
    pub fn declare(&mut self, buffer: &Buffer) -> BufferRef {
        self.buffers.declare(buffer)
    }

    pub fn buffers(&self) -> &Buffers {
        &self.buffers
    }

    /// Storage over slot `slot`, on the lowering's view, for an op's input.
    pub fn slot_storage(&self, slot: usize, count: usize, dtype: DType) -> MetalStorage {
        let s = &self.slots[slot];
        let view = self.base.view(s.offset, s.size);
        MetalStorage::new(
            BufferHandle::new(PooledBuffer::unpooled(view, s.size)),
            self.view.clone(),
            count,
            dtype,
        )
    }

    /// Storage over a declared `buffer`, on the lowering's view.
    pub fn storage(&self, buffer: &Buffer, count: usize, dtype: DType) -> MetalStorage {
        MetalStorage::new(
            BufferHandle::new(PooledBuffer::unpooled(buffer.clone(), buffer.length())),
            self.view.clone(),
            count,
            dtype,
        )
    }

    /// Why `device` cannot be lowered beside right now, if it cannot.
    fn check_quiet(device: &MetalDevice) -> Result<(), Refused> {
        let pass_open = device.pass_state.load(std::sync::atomic::Ordering::Relaxed) & 0b10 != 0;
        let step_open = device.arena.read().as_ref().is_some_and(Arena::is_active);
        if pass_open || step_open {
            return Err(Refused::LiveStep);
        }
        if observing_binds() {
            return Err(Refused::Observed);
        }
        if default_param_style() == ParamStyle::Packed {
            return Err(Refused::PackedStyle);
        }
        Ok(())
    }

    /// Run `op` on the view with `allocs` as the node's outputs, and return
    /// the dispatches it made, resolved. `live` is the device the plan is
    /// built for: the lowering refuses while it has a pass open.
    pub fn lower(
        &self,
        live: &MetalDevice,
        node: u32,
        allocs: &[NodeAlloc],
        op: impl FnOnce(&MetalDevice) -> crate::Result<()>,
    ) -> Result<Vec<LoweredCommand>, Refusal> {
        let refuse = |reason| Refusal { node, reason };
        Self::check_quiet(live).map_err(refuse)?;
        let arena = self.arena(allocs).map_err(refuse)?;
        *self.view.arena.write() = Some(arena.clone());
        // Drop anything an earlier, refused lowering left behind.
        let _ = self.capture.take();
        let ran = {
            let _step = Step::begin(&arena);
            op(&self.view)
        };
        *self.view.arena.write() = None;
        let captured = self.capture.take();
        ran.map_err(|e| refuse(Refused::Op(e.to_string())))?;
        let captured = captured.map_err(|f| refuse(Refused::Capture(f)))?;
        let c = arena.counters();
        if c.offers != allocs.len() as u64 || c.hits != allocs.len() as u64 {
            return Err(refuse(Refused::Outputs {
                declared: allocs.len(),
                offers: c.offers,
                served: c.hits,
            }));
        }
        self.resolve(captured).map_err(refuse)
    }

    /// An arena serving exactly `allocs`, by ordinal, at the positions they
    /// declare.
    fn arena(&self, allocs: &[NodeAlloc]) -> Result<Arena, Refused> {
        let mut by_ordinal = Vec::with_capacity(allocs.len());
        for (alloc, a) in allocs.iter().enumerate() {
            if self.slots.get(a.slot).is_none_or(|s| a.size > s.size) {
                return Err(Refused::Slot { alloc });
            }
            by_ordinal.push(Some((a.slot, a.size)));
        }
        let plan = StepPlan::new(self.slots.clone(), by_ordinal)
            .with_alloc_positions(allocs.iter().map(|a| a.at).collect());
        Arena::new(self.base.clone(), plan, ArenaLayout::Packed).map_err(Refused::Op)
    }

    fn resolve(&self, captured: Vec<CapturedCommand>) -> Result<Vec<LoweredCommand>, Refused> {
        captured
            .into_iter()
            .enumerate()
            .map(|(command, c)| {
                let args = c
                    .args
                    .into_iter()
                    .map(|arg| match arg {
                        CapturedArg::Bytes { index, bytes } => {
                            Ok(LoweredArg::Bytes { index, bytes })
                        }
                        CapturedArg::Opaque { index } => Err(Refused::Opaque { command, index }),
                        CapturedArg::Buffer {
                            index,
                            buffer,
                            offset,
                            write,
                        } => match self.buffers.get(&buffer) {
                            Some(buffer) => Ok(LoweredArg::Buffer {
                                index,
                                buffer,
                                offset,
                                write,
                            }),
                            None => Err(Refused::Unresolved { command, index }),
                        },
                        CapturedArg::ThreadgroupMemory { index, len } => {
                            Ok(LoweredArg::ThreadgroupMemory { index, len })
                        }
                    })
                    .collect::<Result<_, _>>()?;
                Ok(LoweredCommand {
                    pipeline: c.pipeline,
                    args,
                    grid: c.grid,
                    threads: c.threads,
                })
            })
            .collect()
    }

    /// Commit and wait for what the lowering's queue holds: binds and
    /// barriers, no dispatch.
    pub fn finish(&self) -> crate::Result<()> {
        self.commands.flush_and_wait().map_err(crate::Error::wrap)
    }
}

impl Drop for Lowering {
    fn drop(&mut self) {
        let _ = self.commands.flush_and_wait();
    }
}

/// An arena step held open for one lowering, closed on every exit.
struct Step<'a>(&'a Arena);

impl<'a> Step<'a> {
    fn begin(arena: &'a Arena) -> Self {
        arena.begin_step();
        Self(arena)
    }
}

impl Drop for Step<'_> {
    fn drop(&mut self) {
        self.0.end_step();
    }
}
