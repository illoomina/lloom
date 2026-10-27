//! What an allocation is for, as the Metal backend's residency sets see it
//! (`kernels.residency.*`, `tuning.md` §6.3n): the class a loader or a cache
//! names for the allocations it makes. Backend-agnostic, so a model names
//! its weights the same way whatever it runs on; only the Metal residency
//! router reads it.
use std::cell::Cell;

/// What an allocation in the device's residency sets is for. The router
/// keeps each class in sets of its own, so the OS's per-set residency
/// decision costs one class's set rather than the model, and the keep-warm
/// can ask for the classes that matter (`kernels.residency.*`, §6.3n).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyClass {
    /// Parameters: the loaders' allocations.
    Weights,
    /// Sequence state: KV pages and arenas, conv state, the session region.
    Kv,
    /// A transient heap chunk: the pools' misses when transient heaps are on.
    HeapChunk,
    /// A pool buffer, one per miss: activations, staging.
    Scratch,
    /// The streamed expert tier's slot pools.
    Experts,
    /// Anything no allocation path names.
    Other,
}

impl ResidencyClass {
    pub const ALL: [ResidencyClass; 6] = [
        ResidencyClass::Weights,
        ResidencyClass::Kv,
        ResidencyClass::HeapChunk,
        ResidencyClass::Scratch,
        ResidencyClass::Experts,
        ResidencyClass::Other,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ResidencyClass::Weights => "weights",
            ResidencyClass::Kv => "kv",
            ResidencyClass::HeapChunk => "heap",
            ResidencyClass::Scratch => "scratch",
            ResidencyClass::Experts => "experts",
            ResidencyClass::Other => "other",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name.trim())
    }
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub(crate) fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

thread_local! {
    static SCOPED_CLASS: Cell<Option<ResidencyClass>> = const { Cell::new(None) };
}

/// Runs `f` with the residency class of every allocation it makes on this
/// thread named `class`: a loader names its weights, a cache its pages. An
/// allocation path that names no class takes its backing's own. Nested
/// scopes restore the outer class on exit, unwinding included.
pub fn residency_class_scope<R>(class: ResidencyClass, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<ResidencyClass>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPED_CLASS.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(SCOPED_CLASS.with(|c| c.replace(Some(class))));
    f()
}

/// The class [`residency_class_scope`] names on this thread, if any.
pub fn scoped_residency_class() -> Option<ResidencyClass> {
    SCOPED_CLASS.with(Cell::get)
}

/// The most device sets the residency router makes: Metal's 32 residency
/// sets per command queue, less the 8 per-slot sets of a Metal 4 queue.
pub const MAX_DEVICE_SETS: usize = 24;

/// What a Metal 4 queue's per-slot residency sets did since the queue was
/// made (`kernels.residency.slot_trace`): the resources the slots added and
/// the host time those `addAllocation`s took, the pipelines added, the
/// bindings left to the device's sets (`kernels.residency.metal4 = *-diff`),
/// the device sets marked used per command buffer (`cb-*`), the set
/// commits (one per
/// command buffer) and their host time, total and longest, and the
/// committing set's `allocatedSize` at the last commit and at its largest.
/// `census` counts every resource a slot met for the first time in a
/// generation, by why it was or was not left to the device's sets
/// ([`SlotListing`]) and by the residency class the device's sets list it
/// under (the last column: none), whatever the declaration arm.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SlotResidency {
    pub adds: u64,
    pub add_ns: u64,
    pub pipeline_adds: u64,
    pub covered: u64,
    pub set_uses: u64,
    pub commits: u64,
    pub commit_ns: u64,
    pub commit_max_ns: u64,
    pub last_bytes: u64,
    pub max_bytes: u64,
    pub census: [[CensusCell; CENSUS_CLASSES]; SlotListing::ALL.len()],
}

/// Resources and their `allocatedSize`, summed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CensusCell {
    pub adds: u64,
    pub bytes: u64,
}

/// The census's class columns: each [`ResidencyClass`], then none.
pub const CENSUS_CLASSES: usize = ResidencyClass::ALL.len() + 1;

/// The census column of `class`.
pub fn census_column(class: Option<ResidencyClass>) -> usize {
    class.map_or(ResidencyClass::ALL.len(), |c| c as usize)
}

/// The name of census column `i`.
pub fn census_column_name(i: usize) -> &'static str {
    ResidencyClass::ALL.get(i).map_or("none", |c| c.name())
}

/// Why a Metal 4 slot set listed a resource itself, or left it to the
/// device's residency sets (`invariants.md` §15.1r).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotListing {
    /// A pool buffer or a heap sub-buffer: covered since PR1107.
    Covered,
    /// An unpooled buffer whose removal waits in the in-flight hold
    /// (`kernels.residency.metal4_cover = held`).
    Held,
    /// In the device's sets, but its removal can precede its GPU readers.
    Listed,
    /// A buffer the device's sets do not list.
    Unlisted,
    /// A constant page of the slot's own (`Encoder::bytes`).
    Page,
    /// A resource bound through a raw path: a blit operand, an ICB run's
    /// `useResources`, an ICB.
    Raw,
}

impl SlotListing {
    pub const ALL: [SlotListing; 6] = [
        SlotListing::Covered,
        SlotListing::Held,
        SlotListing::Listed,
        SlotListing::Unlisted,
        SlotListing::Page,
        SlotListing::Raw,
    ];
    pub fn name(self) -> &'static str {
        match self {
            SlotListing::Covered => "covered",
            SlotListing::Held => "held",
            SlotListing::Listed => "listed",
            SlotListing::Unlisted => "unlisted",
            SlotListing::Page => "page",
            SlotListing::Raw => "raw",
        }
    }
}

/// The resources one allocation site's allocations account for in the slot
/// census: the reason and class, the `lloom*` frames of the stack that
/// listed them in the device's sets (innermost first; empty where no site
/// was recorded), and the count. Recorded in builds with symbols only
/// (`debug_assertions` or the `profile` feature).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotSite {
    pub listing: SlotListing,
    pub class: Option<ResidencyClass>,
    pub frames: Vec<String>,
    pub cell: CensusCell,
}
