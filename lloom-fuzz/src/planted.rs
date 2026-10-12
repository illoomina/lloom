//! Planted defects for validating the persistent record against deliveries
//! the kernel makes, not `raise`. No engine harness reaches this module:
//! `record_selftest --planted-fault` is its one caller, and the flag is part
//! of that campaign's identity.

use std::sync::OnceLock;

/// The first byte of an input that takes the planted fault.
pub const FAULT_TRIGGER: u8 = b'F';

/// A page mapped with no access, created on first use outside any signal
/// context and never unmapped.
fn guard_page() -> usize {
    static PAGE: OnceLock<usize> = OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: an anonymous private mapping owned by this static; nothing
        // but the planted load below ever addresses it.
        let page = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page_size(),
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(page, libc::MAP_FAILED, "planted guard page");
        page as usize
    })
}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).expect("positive page size")
}

/// Where the planted load reads for `bytes`: inside the guard page.
pub fn fault_address(bytes: &[u8]) -> usize {
    guard_page() + bytes.len() % page_size()
}

/// Load from the guard page when `bytes` starts with [`FAULT_TRIGGER`]: a
/// protection fault taken in instrumented code. Returns otherwise.
#[inline(never)]
pub fn fault_if_triggered(bytes: &[u8]) {
    if bytes.first() == Some(&FAULT_TRIGGER) {
        let address = fault_address(bytes);
        // SAFETY: none; this load is the planted defect. The page is mapped
        // PROT_NONE, so the kernel delivers the fault at `address`.
        unsafe { std::ptr::read_volatile(address as *const u8) };
    }
}
