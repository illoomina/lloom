//! ML framework for Rust
//!
//! ```rust
//! use lloom_tensor::{Tensor, DType, Device};
//! # use lloom_tensor::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and feels like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Candle?
//!
//! Candle stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)
//!
//! ## Other Crates
//!
//! Candle consists of a number of crates. This crate holds core the common data structures but you may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
//! - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
//! - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
//! - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
//! - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
//! - [candle-transformers](https://docs.rs/candle-transformers/). Candle implementation of many published transformer models.
//!

#[cfg(test)]
mod free_observer;
#[cfg(test)]
#[global_allocator]
static OBSERVED_SYSTEM: free_observer::ObservedSystem = free_observer::ObservedSystem;

#[cfg(feature = "accelerate")]
mod accelerate;
#[cfg(any(feature = "metal", test))]
mod allocation_admission;
mod allocation_alias;
#[cfg(any(feature = "metal", test))]
mod allocation_geometry;
#[cfg(any(feature = "metal", test))]
mod allocation_header;
#[cfg(any(feature = "metal", test))]
mod allocation_header_root;
#[cfg(any(feature = "metal", test))]
mod allocation_header_storage;
#[cfg(any(feature = "metal", test))]
mod allocation_owner;
#[cfg(any(feature = "metal", test))]
mod allocation_physical_ready;
pub mod arms;
pub mod autorelease;
pub mod backend;
pub mod backprop;
#[doc(hidden)]
pub mod checkout;
mod cold_box;
mod cold_bytes;
mod command_lifecycle;
mod command_policy;
mod command_segments;
mod completion_frontier;
#[cfg(any(feature = "metal", test))]
mod completion_setup;
#[cfg(any(feature = "metal", test))]
mod completion_scope;
#[cfg(any(feature = "metal", test))]
mod completion_wake;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
mod current_command;
mod custom_op;
#[cfg(any(feature = "metal", test))]
mod deferred_failure_relay;
mod device;
pub mod display;
mod dtype;
pub mod dummy_cuda_backend;
pub mod dummy_dtype;
mod dummy_metal_backend;
pub mod error;
#[cfg(any(feature = "metal", test))]
mod heap_ranges;
mod indexer;
pub mod isolation;
pub mod layout;
pub mod machine;
mod managed_layout;
#[cfg(any(feature = "metal", test))]
mod membership_groups;
mod memory_accounting;
mod memory_accounting_machine;
pub mod memory_admission;
pub mod memory_arbitration;
// Compile the actual production ledger and its lifecycle tests without Metal.
// The ledger itself depends only on numeric accounting and machine ownership.
#[cfg(all(test, not(feature = "metal")))]
#[path = "kernels/metal/ledger.rs"]
mod memory_ledger_cpu_tests;
#[cfg(any(feature = "metal", test))]
mod native_binding_preflight;
#[cfg(any(feature = "metal", test))]
mod native_reuse;
pub use autorelease::AutoreleaseScope;
/// Declares the allocations made while it is alive as outliving their
/// forward pass, so no pass plan ever serves them (`DESIGN.md` §9.2aa).
#[cfg(feature = "metal")]
pub use metal_backend::EscapeGuard;
/// The same declaration on a backend with no plan cache: nothing there hands
/// a later pass the memory an earlier one wrote, so it is a no-op.
#[cfg(not(feature = "metal"))]
#[derive(Default)]
pub struct EscapeGuard(());
#[cfg(not(feature = "metal"))]
impl EscapeGuard {
    pub fn new() -> Self {
        Self(())
    }
}
/// Waits, before the process exits, for every dropped Metal device to finish
/// retiring on the host retirement service (`tuning.md` §6.3m).
#[cfg(feature = "metal")]
pub use kernels::metal::{join_device_retirement, DeviceRetirementReport};
/// No backend but Metal retires a device asynchronously.
#[cfg(not(feature = "metal"))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRetirementReport {
    pub completed: bool,
    pub sets_retiring: usize,
    pub records_pending: usize,
    pub devices_live: usize,
    pub worker_stopped: bool,
    pub listed_at_release: u64,
}
#[cfg(not(feature = "metal"))]
pub fn join_device_retirement(_deadline: std::time::Instant) -> DeviceRetirementReport {
    DeviceRetirementReport {
        completed: true,
        worker_stopped: true,
        ..Default::default()
    }
}
#[cfg(feature = "metal")]
pub mod metal_backend;
// Merged from candle-metal-kernels (RESTRUCTURING §7.1): the Metal kernel layer
// is now a module of this crate rather than a separate leaf crate. Gated on the
// `metal` feature exactly as its former crate dependency was.
/// The GEMV's tile selection as plain arithmetic, for the dispatcher and for
/// a reader with no device (the optimizer's legality facts).
pub mod gemv_tile;
#[cfg(any(feature = "metal", test))]
mod host_anchor;
#[cfg(feature = "metal")]
pub mod kernels;
/// The paged-KV page allocator (§10.3c–h): pure host bookkeeping — a free
/// list, per-page refcounts, per-sequence page tables, a quarantine keyed
/// on GPU epochs — shared by every model's page pool on every backend
/// (LFM2's `BatchPool`, Qwen4-exp's `KvPool`; #749 P29 (8)). Re-exported at
/// `kernels::metal::kv_pages`, its former home, for the Metal callers.
pub mod kv_pages;
#[cfg(feature = "mkl")]
mod mkl;
pub mod nditer;
pub mod npy;
pub mod op;
pub mod options;
#[cfg(any(feature = "metal", test))]
mod ordered_metadata;
pub mod pickle;
#[cfg(any(feature = "metal", test))]
mod planner_transaction;
#[cfg(any(feature = "metal", test))]
mod prefix_data;
pub mod quantized;
#[cfg(any(feature = "metal", test))]
mod residency_ownership;
pub mod retire;
#[cfg(any(feature = "metal", test))]
mod retirement_mailbox;
mod retirement_owners;
#[cfg(any(feature = "metal", test))]
mod retirement_registry;
#[cfg(any(feature = "metal", test))]
mod retirement_service;
pub mod safetensors;
pub mod scalar;
pub mod shape;
pub mod schedule_hook;
pub mod shape_probe;
mod sort;
mod storage;
pub mod streaming;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod test_utils;
pub mod tuning;
pub mod typed;
pub mod utils;
mod variable;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
#[cfg(feature = "ug")]
pub use custom_op::UgIOp1;
pub use custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
pub use device::{Device, DeviceLocation, EosGateReport, EosGateSuspended, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use nditer::NdIter;
pub use shape::{Shape, D};
pub use storage::{KvScatterGeometry, Storage};
pub use streaming::{StreamTensor, StreamingBinOp, StreamingModule};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{inverse_cdf_row, StagedRead, Tensor, TensorId};
pub use typed::{Backend, Cpu, Cuda, Metal, Typed};
pub use variable::Var;

#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend as cuda;

pub use cuda::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(not(feature = "metal"))]
pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

/// A single forward method using a single tensor argument and a flag to
/// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}
