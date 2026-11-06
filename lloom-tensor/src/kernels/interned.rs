//! Kernel `[[host_name]]`s leaked once per process (#1235 step 1c).
//!
//! The dispatch paths that name a kernel from parts (`FlashKernel`, the gemv
//! names, `MetalStorage::binary`) cache each name per thread, so a warm
//! dispatch is a thread-local probe: no allocation, no lock, no atomic. A
//! thread's FIRST request for a variant comes here instead, under one lock,
//! and gets the process's string for it. So a name is leaked the first time
//! the process sees it and never again for a later thread. The names are
//! finite, which bounds the leaked storage by them whatever the thread churn
//! (it was once per thread per variant: #1235 step 1b leaked again on every
//! new thread).

use std::collections::HashSet;
use std::sync::OnceLock;

static NAMES: OnceLock<parking_lot::Mutex<HashSet<&'static str>>> = OnceLock::new();

/// The process's `&'static str` for `name`. Leaks `name` only when no thread
/// has interned the same text before; otherwise `name` is dropped and the
/// existing string returned. Cold: a thread calls it once per variant.
pub(crate) fn intern(name: String) -> &'static str {
    let mut names = NAMES.get_or_init(Default::default).lock();
    if let Some(&existing) = names.get(name.as_str()) {
        return existing;
    }
    let leaked: &'static str = Box::leak(name.into_boxed_str());
    names.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::intern;

    /// Two threads interning the same text get one string: the second
    /// thread's copy is dropped, not leaked.
    #[test]
    fn a_name_interned_by_two_threads_is_one_string() {
        let text = "interned_test_name_a";
        let first = std::thread::spawn(move || intern(text.to_string()) as *const str as *const u8 as usize)
            .join()
            .unwrap();
        let second = std::thread::spawn(move || intern(text.to_string()) as *const str as *const u8 as usize)
            .join()
            .unwrap();
        assert_eq!(first, second, "a later thread must get the process's string, not leak another");
        let other = intern("interned_test_name_b".to_string());
        assert_ne!(other.as_ptr() as usize, first);
        assert_eq!(intern(text.to_string()), text);
    }
}
