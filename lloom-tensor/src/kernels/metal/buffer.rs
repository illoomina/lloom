use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource};
use std::{collections::HashMap, sync::Arc};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,

    /// Byte offset this handle addresses within `raw`.
    ///
    /// Zero for every buffer candle allocates today, which is why this is not a
    /// behaviour change: an ordinary allocation owns its whole `MTLBuffer` and
    /// starts at 0. It is nonzero only for an activation-arena slot, which is a
    /// *region* of one shared allocation rather than an allocation of its own
    /// (`DESIGN.md` §9.2).
    ///
    /// # Why the offset lives here rather than at the call sites
    ///
    /// The arena binds every activation as `base + offset`. Threading that
    /// offset outward would mean touching the ~51 kernel entry points that take
    /// `output: &Buffer` and the 62 `buffer_o` call sites in the Metal backend
    /// -- and under `HazardTrackingModeUntracked` (§3.5) every one of those is
    /// an opportunity for a silent wrong-offset bind, which is precisely the
    /// failure class §9.3 says rests entirely on our offset arithmetic.
    ///
    /// Carrying it in the handle instead means the addition happens in exactly
    /// two places -- `set_input_buffer` and `set_output_buffer`, the choke point
    /// every binding already funnels through. That is the same argument §11.1b
    /// makes for putting the executor seam at dispatch submission rather than at
    /// the kernel entry points: one place to review, not fifty.
    base_offset: usize,

    /// Bytes this handle addresses, when it is a view.
    ///
    /// `None` means "all of `raw`", which is every ordinary allocation. A view
    /// carries its slot size so `length()` cannot report the whole arena.
    length: Option<usize>,

    /// Whether `contents()` may be trusted at all: false for a Private buffer
    /// placed in a heap, decided once at construction (see `new`).
    cpu_visible: bool,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        // A Private buffer placed in an `MTLHeap` answers `contents()` with
        // null plus its heap offset: non-null, not a CPU mapping, and reading
        // it faults (§6.3k). A Private buffer of its own answers truthfully: a
        // mapping below 16384 B, null at or above (§15.2 #12a). So heap
        // placement is what makes the answer untrustworthy, and only it is
        // overridden.
        let cpu_visible =
            raw.storageMode() != objc2_metal::MTLStorageMode::Private || raw.heap().is_none();
        super::arena::note_new_buffer(Retained::as_ptr(&raw) as usize);
        Buffer {
            raw,
            base_offset: 0,
            length: None,
            cpu_visible,
        }
    }

    /// A handle addressing `base_offset..base_offset + length` of this buffer.
    ///
    /// Shares the underlying `MTLBuffer` -- this is a view, not a copy, and
    /// nothing is allocated. Used by the activation arena to hand out slots.
    ///
    /// The shared `MTLBuffer` identity is deliberate on two counts. It is what
    /// makes buffer identity *stable* at a dispatch position (§11.1a.1's 674),
    /// since every slot presents the same allocation. And it makes the
    /// encoder's hazard tracking, which keys on the raw pointer, treat the whole
    /// arena as one resource -- so a read-after-write on any slot orders against
    /// a write to any other. That is conservative rather than unsound: Metal
    /// barriers are resource-granular anyway (§3.5), so a finer key could not
    /// have expressed a narrower edge.
    pub fn view(&self, base_offset: usize, length: usize) -> Buffer {
        debug_assert!(
            base_offset + length <= self.length(),
            "arena view {base_offset}..{} runs past its {} byte parent",
            base_offset + length,
            self.length()
        );
        Buffer {
            raw: self.raw.clone(),
            base_offset: self.base_offset + base_offset,
            length: Some(length),
            cpu_visible: self.cpu_visible,
        }
    }

    /// Byte offset this handle addresses within the underlying `MTLBuffer`.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Whether this handle addresses a **sub-range** of an `MTLBuffer` it does
    /// not own, rather than a whole allocation.
    ///
    /// # Why the pool asks (lloom issue #507)
    ///
    /// `BufferPool::acquire` keys on **size alone** and knows nothing of
    /// provenance -- it hands back whatever is in the bucket. That is correct
    /// for allocations, which own their whole `MTLBuffer` and are
    /// interchangeable at a given size, and it is silent corruption for views,
    /// which are N handles onto **one** allocation: measured on the shipping
    /// arena, five "separate" activation buffers at one base address and five
    /// offsets, under `HazardTrackingModeUntracked` where §9.3 says there is no
    /// safety net (§9.2m, issue #489).
    ///
    /// So the pool's invariant is that a view never enters it, and `adopt` is
    /// where that is checked -- see [`super::BufferPool::adopt`] for why there
    /// rather than at `acquire`.
    ///
    /// **`base_offset != 0` is NOT the test**, which is why this exists as its
    /// own predicate: an arena's first slot is a view at offset 0 (the measured
    /// trace has one), so an offset check would pass exactly the aliasing handle
    /// that shares its address with the base. What makes a handle a view is that
    /// it carries its own extent.
    pub fn is_view(&self) -> bool {
        self.length.is_some()
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    /// CPU pointer to the first byte *this handle addresses*.
    ///
    /// Offset by `base_offset`, so a view reads and writes its own region
    /// rather than the parent's start. Keeping the whole `Buffer` API
    /// consistent about what "this buffer" means is what stops a view from
    /// being a silent-corruption hazard: `read_to_vec` and every other CPU-side
    /// consumer takes this pointer without knowing whether it holds a view.
    ///
    /// Null for a Private-storage allocation at or above 16384 B, which has no
    /// CPU mapping -- unchanged by the offset, since `null + n` is still not
    /// readable and callers already assert on it. Below that threshold a
    /// Private allocation *is* mapped, which is `DESIGN.md` §15.2 #12a's
    /// correction to the older "null for Private" reading. Null, too, for a
    /// Private buffer placed in a heap, whatever `contents()` returns (`new`).
    ///
    /// # This function's own null check folded away (#366, repaired #454)
    ///
    /// §11.5d records the early return below being eliminated in the release
    /// binary: `base_offset` is 0 for every non-arena allocation, so the
    /// optimiser proves `null + 0 == null`, concludes the branch is redundant
    /// and drops it -- and with it the only thing standing between a null
    /// mapping and the `memmove` in a caller. The two production callers in
    /// `metal_backend` were repaired with `black_box` by #366 and #445; this
    /// one is the function *they call*, and it was not.
    ///
    /// Measured for #454 (`measurements/issue-454-raw/guard-probe/`, with the
    /// disassembly in `measurements/issue-454-raw/shape-disassembly.txt`): in
    /// an unguarded function the pointer `objc_msgSend` returns reaches `ldrb`
    /// with **no `cbz` on it at all**, at `opt-level=3` with LTO off as well as
    /// on; with `black_box` the same function stores it, reloads it and
    /// branches on `cbz`. Guarding here means every consumer of `data()`
    /// inherits a pointer whose nullity was actually tested, rather than each
    /// having to defend itself.
    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        // `black_box` is load-bearing, not a micro-optimisation guard: it makes
        // the pointer opaque so the `is_null` below cannot be folded against
        // what the optimiser can prove about `contents()`.
        if !self.cpu_visible {
            return std::ptr::null_mut();
        }
        let base = std::hint::black_box(self.as_ref().contents().as_ptr()) as *mut u8;
        if base.is_null() {
            return base;
        }
        // SAFETY: `base_offset` is within the parent allocation by construction
        // -- `view` is only called by the arena, with an offset and length that
        // the plan has already fitted inside the arena buffer.
        unsafe { base.add(self.base_offset) }
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(&self.raw)
    }

    /// Address of the underlying `MTLBuffer`, as the encoder sees it.
    ///
    /// The same value the hazard tracking and the dispatch trace key on, so an
    /// arena recording can attribute a bind back to the allocation that produced
    /// the buffer. It identifies the *buffer*, which is deliberately not the
    /// same thing as the value occupying it (§9.2c).
    pub fn raw_addr(&self) -> usize {
        self.raw_ptr() as usize
    }

    /// Bytes this handle addresses.
    ///
    /// A view reports its slot's extent, not the arena's. `fill_buffer` and
    /// `copy_from_buffer` both take `buffer.length()` as a range, so reporting
    /// the parent's length would let a blit over one slot run across every slot
    /// after it -- silent corruption of the exact kind §9.3 says rests on our
    /// offset arithmetic.
    pub fn length(&self) -> usize {
        match self.length {
            Some(len) => len,
            None => self.as_ref().length(),
        }
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;

#[cfg(test)]
mod issue_454_guard_shape {
    //! The null guards on `contents()` are load-bearing and were eliminated
    //! twice (#366, #445). This pins their SHAPE in source.
    //!
    //! It cannot be a runtime test: the defect is that a branch is absent from
    //! the emitted code, and a test that runs is executing the build it is
    //! trying to indict. #366 and #445 both verified by reading the
    //! disassembly, and #454's own probe does the same
    //! (`measurements/issue-454-raw/shape-disassembly.txt`). What a test CAN
    //! do cheaply is fail when a site loses its `black_box` in a refactor,
    //! which is how both regressions arrived -- a correct-looking source diff.
    //!
    //! Scoped to the sites where a null is REACHABLE, i.e. buffers that can be
    //! Private storage. Shared-storage buffers (`RESOURCE_OPTIONS`,
    //! `CURSOR_OPTIONS`) are CPU-mapped at every size, so their guards are
    //! unreachable and are deliberately not required to carry one.

    /// Every load-bearing `contents()` site keeps its optimisation barrier.
    ///
    /// Mutation-tested: removing the `black_box` from `Buffer::data` makes this
    /// fail. A first version sliced the function body loosely and passed under
    /// that mutation, because `black_box` also appears in the doc comment above
    /// it -- §15.1 #1's vacuous-comparison class, in the guard written to
    /// prevent a regression. It matches the STATEMENT now, not the region.
    #[test]
    fn load_bearing_contents_sites_keep_their_black_box() {
        let src = include_str!("buffer.rs");

        // Only the production half. This module mentions `.contents()` in its
        // own filter literal and in its messages, and a check that reads
        // itself reports on the wrong file -- which is how the first version
        // failed its own clean arm.
        let production = src
            .split("mod issue_454_guard_shape")
            .next()
            .expect("split always yields one element");

        // The line that actually reads the pointer, matched exactly.
        let reads: Vec<&str> = production
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && l.contains(".contents()"))
            .collect();

        assert!(
            !reads.is_empty(),
            "no `contents()` call found in buffer.rs -- was `data` renamed? \
             A guard that finds nothing to check passes for the wrong reason."
        );

        for line in reads {
            assert!(
                line.contains("black_box"),
                "Buffer::data's null test lost its `black_box`:\n    {line}\n\
                 Without it the optimiser proves the pointer non-null and drops \
                 the branch -- DESIGN.md §11.5d, §14.5c, and #454's own \
                 disassembly (`measurements/issue-454-raw/shape-disassembly.txt`). \
                 `data()` is what every CPU-side consumer reaches, so this is the \
                 one site whose guard covers the others. If removing it is \
                 deliberate, re-read the disassembly rather than the source."
            );
        }
    }
}
