//! A scoped autorelease pool for a loop iteration that drives Metal.
//!
//! # Why the decode loop needs one
//!
//! Every `commandBuffer`, every compute and blit encoder and a handful of
//! runtime objects come back **autoreleased**: the Objective-C runtime parks
//! them in the innermost autorelease pool of the calling thread and releases
//! them when that pool drains. A Rust thread has no pool unless something
//! pushes one, and then the runtime's own detector says what happens
//! (`OBJC_DEBUG_MISSING_POOLS=YES`): *"autoreleased with no pool in place -
//! just leaking"*. Measured on the release decode loop (`lfm2-chat -n 60`):
//! **12 047 such objects** — ~200 per token, 2 205 of them blit encoders and
//! 578 compute encoders (lloom #706 follow-up, `measurements/issue-706-raw/
//! probes.log`). In a debug build the command buffers themselves join them
//! (objc2's retain of an autoreleased return takes the claim fast path only
//! when optimised), and ~4 600 of those exhaust the driver: `Internal Error
//! (00000206)` on the next commit.
//!
//! # Shape
//!
//! A guard, not a closure: the loops it goes into `break`, `continue` and `?`
//! from the middle of a 250-line body, and a closure would have to thread all
//! of that through. `AutoreleaseScope::new()` pushes a pool;
//! its `Drop` pops it, which drains everything autoreleased since. Declared
//! FIRST in the iteration so it drops LAST. Everything the loop keeps across
//! iterations is held by a `Retained` (+1 of its own), so the drain touches
//! nothing that is still owned.
//!
//! Two calls into the runtime per iteration (`objc_autoreleasePoolPush` /
//! `Pop`), against a ~17 ms decode step. Without the `metal` feature the type
//! is a zero-sized no-op.

/// A pushed autorelease pool, popped on drop. See the module docs.
#[must_use = "the pool is popped when this is dropped; bind it to `_pool`, not `_`"]
pub struct AutoreleaseScope {
    #[cfg(feature = "metal")]
    context: *mut core::ffi::c_void,
}

impl AutoreleaseScope {
    /// Push a pool for the current scope.
    pub fn new() -> Self {
        #[cfg(feature = "metal")]
        {
            // SAFETY: push/pop are balanced by construction (the pop is in
            // `Drop`, on the same thread, in LIFO order with any pool pushed
            // inside this scope, since inner guards drop first).
            let context = unsafe { objc2::ffi::objc_autoreleasePoolPush() };
            Self { context }
        }
        #[cfg(not(feature = "metal"))]
        {
            Self {}
        }
    }
}

impl Default for AutoreleaseScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoreleaseScope {
    fn drop(&mut self) {
        #[cfg(feature = "metal")]
        // SAFETY: `context` came from the matching push on this thread.
        unsafe {
            objc2::ffi::objc_autoreleasePoolPop(self.context)
        }
    }
}
