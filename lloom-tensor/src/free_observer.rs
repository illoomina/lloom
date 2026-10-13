//! Unit-test allocator observation only; no production allocator/hook change.
//! System frees the exact observed block BEFORE its charge snapshot is recorded.
use crate::memory_admission::{AccountSnapshot, ModelDomain};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

pub(crate) struct ObservedSystem;
// SAFETY: every operation forwards exactly once to System with its original
// pointer/Layout contract. In the isolated manual-service fixture observation
// performs no allocation/parking/unwind or freed-memory access. General
// try_lock guard Drop can wake contenders; no universal syscall-free claim.
// TLS is const fixed Copy state; Busy prevents recursive tracing.
unsafe impl GlobalAlloc for ObservedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        born(pointer, layout);
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        born(pointer, layout);
        pointer
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // The bounded observed constructor owns fixed blocks, not reallocating
        // storage. An observed realloc invalidates the test, never hides it.
        let result = unsafe { System.realloc(pointer, layout, size) };
        let _ = TRACE.try_with(|state| {
            let mut trace = state.get();
            if trace.mode == BIRTH
                || (trace.mode == WATCH
                    && trace.blocks.iter().any(|b| b.pointer == pointer as usize))
            {
                trace.reallocated = true;
                state.set(trace);
            }
        });
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded caller GlobalAlloc contract, no observation of the
        // allocation contents occurs either before or after this real free.
        unsafe { System.dealloc(pointer, layout) };
        freed(pointer as usize, layout);
    }
}
const OFF: u8 = 0;
const BIRTH: u8 = 1;
const WATCH: u8 = 2;
const BUSY: u8 = 3;
const LIMIT: usize = 8;
#[derive(Clone, Copy, Default)]
pub(crate) struct Block {
    pub pointer: usize,
    pub size: usize,
    pub align: usize,
}
#[derive(Clone, Copy)]
struct Trace {
    mode: u8,
    domain: *const ModelDomain,
    blocks: [Block; LIMIT],
    born: usize,
    freed: usize,
    order: [usize; LIMIT],
    snapshots: [Option<AccountSnapshot>; LIMIT],
    overflow: bool,
    reallocated: bool,
    wrong_layout: bool,
    duplicate_free: bool,
}
impl Trace {
    const EMPTY: Self = Self {
        mode: OFF,
        domain: std::ptr::null(),
        blocks: [Block {
            pointer: 0,
            size: 0,
            align: 0,
        }; LIMIT],
        born: 0,
        freed: 0,
        order: [usize::MAX; LIMIT],
        snapshots: [None; LIMIT],
        overflow: false,
        reallocated: false,
        wrong_layout: false,
        duplicate_free: false,
    };
}
thread_local! { static TRACE: Cell<Trace> = const { Cell::new(Trace::EMPTY) }; }
fn born(pointer: *mut u8, layout: Layout) {
    if pointer.is_null() {
        return;
    }
    let _ = TRACE.try_with(|state| {
        let mut trace = state.get();
        if trace.mode != BIRTH {
            return;
        }
        if trace.born == LIMIT {
            trace.overflow = true;
        } else {
            trace.blocks[trace.born] = Block {
                pointer: pointer as usize,
                size: layout.size(),
                align: layout.align(),
            };
            trace.born += 1;
        }
        state.set(trace);
    });
}
fn freed(pointer: usize, layout: Layout) {
    let _ = TRACE.try_with(|state| {
        let mut trace = state.get();
        if trace.mode != WATCH {
            return;
        }
        let Some(index) = trace.blocks[..trace.born]
            .iter()
            .position(|b| b.pointer == pointer)
        else {
            return;
        };
        let block = trace.blocks[index];
        trace.wrong_layout |= block.size != layout.size() || block.align != layout.align();
        trace.duplicate_free |= trace.order[..trace.freed].contains(&index);
        if trace.freed == LIMIT {
            trace.overflow = true;
            state.set(trace);
            return;
        }
        trace.mode = BUSY;
        state.set(trace);
        // SAFETY: observe's frame borrows this SAME live domain and retains its
        // internal Reset through the closure; normal return/unwind clears TLS. This test-only method
        // uses nonblocking try_lock and copies only numeric state; no allocation,
        // callback, map mutation or missing-domain panic. This isolated manual
        // fixture has no contenders; arbitrary try_lock guard Drop could wake
        // waiters, so this is NOT a universal syscall-free claim. None records
        // a failed observation.
        let snapshot = unsafe { (&*trace.domain).test_account_snapshot_nonblocking() };
        trace.order[trace.freed] = index;
        trace.snapshots[trace.freed] = snapshot;
        trace.freed += 1;
        trace.mode = WATCH;
        state.set(trace);
    });
}
/// Non-owning phase capability lent only to observe's closure. No safe
/// constructor/Clone/Default/owned guard escape or observation pointer export.
pub(crate) struct Watch {
    _private: (),
}
pub(crate) struct Report {
    pub blocks: [Block; LIMIT],
    pub born: usize,
    pub freed: usize,
    pub order: [usize; LIMIT],
    pub snapshots: [Option<AccountSnapshot>; LIMIT],
    pub overflow: bool,
    pub reallocated: bool,
    pub wrong_layout: bool,
    pub duplicate_free: bool,
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        let _ = TRACE.try_with(|state| state.set(Trace::EMPTY));
    }
}
/// Closure-scoped observer. Internal Reset is never lent/returned; it clears
/// TLS before normal return AND unwind. Forgetting the borrowed Watch cannot
/// prevent reset or leave a dangling domain pointer after the call returns.
pub(crate) fn observe(domain: &ModelDomain, run: impl FnOnce(&mut Watch)) -> Report {
    TRACE.with(|state| {
        assert_eq!(state.get().mode, OFF, "no nested allocator control");
        state.set(Trace {
            mode: BIRTH,
            domain,
            ..Trace::EMPTY
        });
    });
    let reset = Reset;
    let mut watch = Watch { _private: () };
    run(&mut watch);
    let trace = TRACE.with(Cell::get);
    drop(reset); // clear BEFORE returning any report or ending domain borrow
    Report {
        blocks: trace.blocks,
        born: trace.born,
        freed: trace.freed,
        order: trace.order,
        snapshots: trace.snapshots,
        overflow: trace.overflow,
        reallocated: trace.reallocated,
        wrong_layout: trace.wrong_layout,
        duplicate_free: trace.duplicate_free,
    }
}
impl Watch {
    pub fn watch(&mut self) {
        TRACE.with(|state| {
            let mut trace = state.get();
            assert_eq!(trace.mode, BIRTH);
            trace.mode = WATCH;
            state.set(trace);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observer_reset_normal_return_with_nonowning_watch() {
        let domain = ModelDomain::isolated_numeric_fixture(0);
        let report = observe(&domain, |watch| {
            watch.watch();
            // Suppressing Drop of this borrowed capability suppresses no Reset:
            // observe retains the private frame-local owner, not the closure.
            let _forgotten_borrow = std::mem::ManuallyDrop::new(watch);
        });
        assert_eq!(TRACE.with(|state| state.get().mode), OFF);
        assert!(TRACE.with(|state| state.get().domain.is_null()));
        assert_eq!(report.born, 0);
        assert_eq!(report.freed, 0);
        drop(domain);
    }
    #[test]
    fn observer_reset_catch_unwind_then_reenter() {
        let domain = ModelDomain::isolated_numeric_fixture(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observe(&domain, |watch| {
                watch.watch();
                panic!("bounded observer reset witness");
            });
        }));
        let panic = result.expect_err("planted observer witness must unwind");
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"bounded observer reset witness")
        );
        assert_eq!(TRACE.with(|state| state.get().mode), OFF);
        assert!(TRACE.with(|state| state.get().domain.is_null()));
        let report = observe(&domain, |watch| watch.watch());
        assert_eq!(report.born, 0);
        assert_eq!(report.freed, 0);
        assert_eq!(TRACE.with(|state| state.get().mode), OFF);
        drop(domain);
    }
}
