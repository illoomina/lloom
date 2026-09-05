use super::{Buffer, Device};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLAllocation, MTLDevice as _, MTLResidencySet, MTLResidencySetDescriptor};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// What the set has done, so the cost model can be checked rather than quoted.
///
/// `DESIGN.md` §6.3d prices eager unregistration at +0.062 ms/token and
/// attributes the cost to `commit()` -- *"any eager scheme puts a `commit()` on
/// the per-token path"*. That is a claim about a **rate**, and until these
/// counters existed nothing measured the rate on either side of it.
///
/// What they make checkable is the asymmetry the claim assumes. An allocation
/// already commits -- [`ResidencySet::insert`] is one buffer and one commit, and
/// it is called from every pool miss -- so the question is not whether a commit
/// reaches the per-token path but **how many more** a batched eager release
/// adds to the ones already there. See [`Self::commits`].
///
/// # These count calls, not effects, and that bound is measured
///
/// Each field is incremented **beside** the Metal call rather than derived from
/// it, so they say the call site ran and not that Metal acted. Mutation-tested:
/// deleting `set.commit()` from `remove_batch` while leaving the increment
/// **survives every test in this tree**. That is recorded rather than fixed
/// because no cheap oracle exists -- `MTLResidencySet` exposes `allocationCount`
/// but not a commit count, so the only instrument that can see a dropped commit
/// is `MTLDevice.currentAllocatedSize` over a long run, which is #206's
/// measurement and not a unit test. Treat a `commits` figure as *"the code
/// intended this many"*.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyCounters {
    /// `MTLResidencySet::commit()` calls, from every path that makes one.
    ///
    /// **This is the quantity §6.3d's cost argument is about**, and the one to
    /// compare between arms. A batched eager release adds *one* commit per
    /// eviction round, against the one already paid per allocation.
    pub commits: u64,
    /// Allocations added, i.e. `addAllocation` calls actually made.
    pub added: u64,
    /// Allocations removed via `removeAllocation`.
    pub removed: u64,
    /// Membership keys forgotten **without** a Metal call -- the retention
    /// (§6.3e). Nonzero here while `removed` stays at zero is the behaviour
    /// that shipped, and it is what reached 48 GB at long context.
    pub retired: u64,
    /// Calls into `insert_batch`, whether or not anything was added.
    pub insert_batches: u64,
    /// Calls into `remove_batch`, whether or not anything was removed.
    pub remove_batches: u64,
}

/// Keeps Metal buffers resident in GPU memory, removing the need for `useResource` calls.
///
/// Add the set to the command queue once via `MTLCommandQueue::addResidencySet`. Then
/// register every buffer at allocation time and unregister at free time.
///
/// # Membership is tracked, and that is a correctness requirement
///
/// `removeAllocation` on an object the set does not hold reaches
/// `IOGPUGroupMemory::remove_memory_object()`, which does not return an error to
/// the user client -- it **panics the machine** (`DESIGN.md` §6.3c, issue #163).
/// So the set cannot be a write-through wrapper over the Metal object: it has to
/// know what it holds, and refuse a remove for anything absent.
///
/// The membership set is keyed on the **`MTLBuffer` object address**, which is
/// what `addAllocation` and `removeAllocation` take. Deliberately not the GPU
/// address: an arena slot is a `Buffer::view` sharing one parent `MTLBuffer`
/// (§9.2), so N views are one allocation to Metal and must be one entry here.
/// Keying on anything finer would let a view's removal drop the parent, or make
/// the parent's insertion look like N.
///
/// # Cost
///
/// One `HashSet` operation per **allocation** and per eviction, and one
/// `removeAllAllocations` at teardown. Nothing here is reached per dispatch or
/// per bind: §11.2's non-GPU budget is 6.1 % of a decode token, so a per-bind
/// membership test would be a real regression where a per-allocation one is
/// free.
///
/// **The `HashSet` is not the cost; the Metal call is** -- measured (#167's
/// ablation, §6.3d), and it is why [`Self::retire_batch`] exists at all.
///
/// **What that measurement did not price is the retention, and it is 48 GB**
/// (§6.3e, #206). Retiring a key without calling `removeAllocation` leaves
/// Metal holding the allocation until the device drops; at §6.3b's 11.6
/// evictions per token that is harmless, and at `KvAppend=Cat`'s ~220 per token
/// at long context it exhausts a 64 GB device. **So the eager path is what
/// ships as of #210** -- see [`Self::remove_batch`] and
/// `ResidencyEvictionObserver`, and [`ResidencyCounters`] for the rate that
/// decides it.
pub struct ResidencySet {
    raw: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,

    /// `MTLBuffer` object addresses currently in the set.
    ///
    /// `Mutex` rather than a lock-free structure because this is an
    /// allocation-path structure, not a per-dispatch one, and because the
    /// invariant that matters -- that a `removeAllocation` is issued exactly
    /// when the address was present -- is a test-and-act pair that has to be
    /// atomic against a concurrent remove of the same buffer. That pair is
    /// precisely the double-unregister #165's audit found reachable in
    /// principle.
    members: Mutex<HashSet<usize>>,

    /// See [`ResidencyCounters`]. Atomics rather than fields inside `members`
    /// so a reader never contends with an allocation for the lock, and
    /// `Relaxed` because these are diagnostics: no other state is published
    /// through them, and the only ordering that matters -- that a commit is
    /// counted on the path that makes it -- is program order on one thread.
    commits: AtomicU64,
    added: AtomicU64,
    removed: AtomicU64,
    retired: AtomicU64,
    insert_batches: AtomicU64,
    remove_batches: AtomicU64,
}

unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &Device) -> Self {
        let descriptor = MTLResidencySetDescriptor::new();
        let raw = device
            .as_ref()
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()
            .inspect(|set| set.requestResidency());
        ResidencySet {
            raw,
            members: Mutex::new(HashSet::new()),
            commits: AtomicU64::new(0),
            added: AtomicU64::new(0),
            removed: AtomicU64::new(0),
            retired: AtomicU64::new(0),
            insert_batches: AtomicU64::new(0),
            remove_batches: AtomicU64::new(0),
        }
    }

    /// What this set has done. See [`ResidencyCounters`].
    pub fn counters(&self) -> ResidencyCounters {
        ResidencyCounters {
            commits: self.commits.load(Ordering::Relaxed),
            added: self.added.load(Ordering::Relaxed),
            removed: self.removed.load(Ordering::Relaxed),
            retired: self.retired.load(Ordering::Relaxed),
            insert_batches: self.insert_batches.load(Ordering::Relaxed),
            remove_batches: self.remove_batches.load(Ordering::Relaxed),
        }
    }

    /// Zeroes the counters, so a caller can measure a window rather than a
    /// process. Used to exclude model load from a decode figure.
    pub fn reset_counters(&self) {
        self.commits.store(0, Ordering::Relaxed);
        self.added.store(0, Ordering::Relaxed);
        self.removed.store(0, Ordering::Relaxed);
        self.retired.store(0, Ordering::Relaxed);
        self.insert_batches.store(0, Ordering::Relaxed);
        self.remove_batches.store(0, Ordering::Relaxed);
    }

    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.raw.as_deref()
    }

    pub fn insert(&self, buf: &Buffer) -> usize {
        self.insert_batch(std::iter::once(buf))
    }

    /// Adds multiple buffers in a single commit.
    ///
    /// A buffer already in the set is skipped rather than added twice: the
    /// membership set is what decides, so a second `addAllocation` would make
    /// the set's own bookkeeping disagree with Metal's about how many removes
    /// the allocation needs.
    ///
    /// Returns how many allocations were actually added.
    ///
    /// **The count is returned because nothing else can observe the skip.**
    /// `addAllocation` is idempotent on Metal's side, so `allocationCount()`
    /// reads the same whether or not a duplicate was forwarded, and a
    /// `HashSet` holding one key has the same length either way. A test
    /// asserting on either quantity **passes** under a mutation that re-adds
    /// unconditionally -- measured while mutation-testing this change, which is
    /// why this returns a count rather than `()`.
    pub fn insert_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) -> usize {
        let Some(set) = &self.raw else {
            return 0;
        };
        self.insert_batches.fetch_add(1, Ordering::Relaxed);
        let Ok(mut members) = self.members.lock() else {
            return 0;
        };
        let mut added = 0usize;
        for buf in bufs {
            if members.insert(allocation_key(buf)) {
                set.addAllocation(as_allocation(buf));
                added += 1;
            }
        }
        if added > 0 {
            set.commit();
            self.commits.fetch_add(1, Ordering::Relaxed);
            self.added.fetch_add(added as u64, Ordering::Relaxed);
        }
        added
    }

    /// Removes multiple buffers in a single commit, skipping any the set does
    /// not hold.
    ///
    /// Returns how many were actually removed, so a caller can assert the
    /// mechanism engaged rather than trusting that it did (`DESIGN.md` §2.4).
    ///
    /// **The membership test is the point.** `removeAllocation` on an absent
    /// object aborts the machine rather than failing (§6.3c), so this is the
    /// half of the guard that makes the kernel's assertion unreachable from our
    /// side -- independently of whether anything ever calls it in the wrong
    /// order.
    pub fn remove_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) -> usize {
        let Some(set) = &self.raw else {
            return 0;
        };
        self.remove_batches.fetch_add(1, Ordering::Relaxed);
        let Ok(mut members) = self.members.lock() else {
            return 0;
        };
        let mut removed = 0usize;
        for buf in bufs {
            if members.remove(&allocation_key(buf)) {
                set.removeAllocation(as_allocation(buf));
                removed += 1;
            }
        }
        if removed > 0 {
            set.commit();
            self.commits.fetch_add(1, Ordering::Relaxed);
            self.removed.fetch_add(removed as u64, Ordering::Relaxed);
        }
        removed
    }

    pub fn remove(&self, buf: &Buffer) -> usize {
        self.remove_batch(std::iter::once(buf))
    }

    /// Forgets these buffers **without** calling Metal, for allocations whose
    /// handles are about to be destroyed.
    ///
    /// Returns how many keys were retired.
    ///
    /// # This is no longer the eviction path -- kept as the measured arm
    ///
    /// It **was** the eviction path (§6.3d, #166), on the argument that
    /// `removeAllocation` marks an allocation *"to be removed on the next
    /// commit"* and so puts a `commit()` on the per-token path -- priced at
    /// **+0.062 ms/token even batched** against §11.2's whole 6.1 % budget.
    /// That pricing is real and it was taken at §6.3b's **11.6 evictions per
    /// token**.
    ///
    /// **What it did not price is what the retention weighs.** Under
    /// `KvAppend=Cat` at long context the eviction rate is **~220 per token**
    /// and the retention reaches **48 GB**, exhausting the device at 97 % of
    /// `recommendedMaxWorkingSetSize` (§6.3e, #206). A retention that is safe is
    /// not thereby bounded.
    ///
    /// So eviction calls [`Self::remove_batch`] as of #210, and this stays
    /// **selectable** rather than deleted: it is the arm the +0.062 ms was
    /// measured on, so a re-measurement has something to measure against, and
    /// §12.3's rule -- variants coexist and are swappable -- applies to an
    /// allocator policy as much as to a kernel.
    ///
    /// What the membership record is *for* is deciding whether a later
    /// `removeAllocation` may be issued. Retiring the key discharges that job
    /// exactly: after this, no per-buffer remove will name the allocation, so
    /// the kernel's assertion stays unreachable. The Metal-side entry is
    /// cleared at teardown by [`Self::remove_all`], which uses
    /// `removeAllAllocations` and therefore names no object at all.
    ///
    /// The residual is a **retention**, not a dangling reference -- which is why
    /// the deferral was safe, and is exactly the property that made its size
    /// easy not to ask about.
    pub fn retire_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) -> usize {
        if self.raw.is_none() {
            return 0;
        }
        let Ok(mut members) = self.members.lock() else {
            return 0;
        };
        let n = bufs
            .into_iter()
            .filter(|buf| members.remove(&allocation_key(buf)))
            .count();
        self.retired.fetch_add(n as u64, Ordering::Relaxed);
        n
    }

    /// How many allocations the set currently holds.
    ///
    /// Exists so a test can observe the mechanism directly. The defect this
    /// guard exists to prevent is a machine panic, which is not a testable
    /// assertion -- so the tests assert on membership, which is.
    pub fn len(&self) -> usize {
        self.members.lock().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this buffer's allocation is currently in the set.
    pub fn contains(&self, buf: &Buffer) -> bool {
        self.members
            .lock()
            .map(|m| m.contains(&allocation_key(buf)))
            .unwrap_or(false)
    }

    /// Empties the set, in one commit.
    ///
    /// This is the teardown path (`DESIGN.md` §6.3c): the residency set must be
    /// emptied *before* the pools free the `MTLBuffer`s it lists, or tearing the
    /// set down asks the kernel to remove objects that are already gone.
    ///
    /// # Why `removeAllAllocations` and not a loop over the members
    ///
    /// Because it **takes no object argument**, so it cannot name a freed
    /// allocation however stale the record has become. A loop would have to
    /// name each one, and at teardown there is no longer a live `Buffer` handle
    /// for every allocation Metal still holds — which is precisely the state
    /// that panics.
    ///
    /// That is what makes [`Self::retire_batch`] safe: an evicted buffer's key
    /// is forgotten without a Metal call, and the entry it leaves behind is
    /// swept up here without being named.
    ///
    /// **Runs whether or not the membership record is empty**, for the same
    /// reason: retired allocations are absent from the record and present in
    /// Metal's set, so gating on `members.len()` would skip exactly the entries
    /// this exists to clear.
    ///
    /// Returns how many keys the membership record still held, which is a lower
    /// bound on what Metal removed rather than an exact count of it.
    pub fn remove_all(&self) -> usize {
        let Some(set) = &self.raw else {
            return 0;
        };
        let Ok(mut members) = self.members.lock() else {
            return 0;
        };
        let n = members.len();
        set.removeAllAllocations();
        set.commit();
        self.commits.fetch_add(1, Ordering::Relaxed);
        members.clear();
        n
    }
}

/// Identity of the allocation Metal sees for this handle.
///
/// The underlying `MTLBuffer` object address, **not** the handle's address and
/// not the GPU address. An arena slot is a view over a shared parent
/// (`Buffer::view`, §9.2), so every slot maps to the parent's key and the arena
/// is one member rather than N.
fn allocation_key(buf: &Buffer) -> usize {
    buf.raw_ptr() as usize
}

/// Cast a `&Buffer` to `&ProtocolObject<dyn MTLAllocation>`.
///
/// Safe because `MTLBuffer: MTLResource: MTLAllocation`. All `ProtocolObject<P>` are
/// `repr(C)` thin ObjC pointers — the cast only changes the static protocol marker,
/// not the pointer value or runtime dispatch.
fn as_allocation(buf: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    unsafe {
        &*(buf.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}
