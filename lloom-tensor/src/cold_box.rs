//! Cold exact-Layout Box birth with ownership-preserving null refusal.
//! This does not change Arc's existing global allocation-error behavior.
use std::alloc::{alloc, Layout};

pub(crate) fn try_new<T>(value: T) -> Result<Box<T>, T> {
    // SAFETY: Global alloc returns null or unique, correctly aligned storage
    // for precisely this Layout, compatible with Box's Global deallocation.
    unsafe { try_new_with(value, alloc) }
}

/// # Safety
/// Allocator returns null without allocation, or a unique std-Global-dealloc-
/// compatible pointer valid for exactly the supplied Layout. It must not unwind
/// after allocating. No references to the returned storage may escape it.
unsafe fn try_new_with<T>(value: T, allocator: unsafe fn(Layout) -> *mut u8) -> Result<Box<T>, T> {
    let layout = Layout::new::<T>();
    if layout.size() == 0 {
        // Box allocates no storage for ZSTs; preserve its alignment/provenance.
        return Ok(Box::new(value));
    }
    // SAFETY: caller's allocator contract, checked nonzero exact Layout.
    let pointer = unsafe { allocator(layout) }.cast::<T>();
    if pointer.is_null() {
        return Err(value);
    }
    // SAFETY: unique initialized ownership; write and from_raw cannot fail or
    // invoke arbitrary code. Box later frees using this exact Global Layout.
    unsafe {
        pointer.write(value);
        Ok(Box::from_raw(pointer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    #[repr(align(128))]
    struct Owner {
        marker: u64,
        dropped: Arc<AtomicUsize>,
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe fn null_without_allocation(_: Layout) -> *mut u8 {
        std::ptr::null_mut()
    }
    #[test]
    fn cold_box_null_returns_original_owner_without_drop() {
        let dropped = Arc::new(AtomicUsize::new(0));
        // SAFETY: injected null function allocates nothing, never unwinds.
        let owner = match unsafe {
            try_new_with(
                Owner {
                    marker: 91,
                    dropped: dropped.clone(),
                },
                null_without_allocation,
            )
        } {
            Err(owner) => owner,
            Ok(_) => panic!("null injection unexpectedly allocated"),
        };
        assert_eq!(owner.marker, 91);
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            0,
            "null refusal destroyed returned input"
        );
        drop(owner);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn cold_box_actual_overaligned_owner_drops_once() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let block = match try_new(Owner {
            marker: 72,
            dropped: dropped.clone(),
        }) {
            Ok(block) => block,
            Err(_) => panic!("bounded cold birth refused"),
        };
        assert_eq!((&*block as *const Owner as usize) % 128, 0);
        assert_eq!(block.marker, 72);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(block);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
