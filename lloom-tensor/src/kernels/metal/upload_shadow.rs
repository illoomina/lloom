//! `LLOOM_UPLOAD_SHADOW` (read once): an A/B instrument for lloom #748.
//!
//! Every pooled host upload (`MetalDevice::new_buffer_with_data`'s small
//! path, a host `memcpy` into a recycled pool buffer) keeps a copy of its
//! bytes here, keyed by the buffer's host address. When the buffer's last
//! handle drops, the bytes still in the buffer are compared with the copy: an
//! upload buffer has no legitimate writer after the upload, so a difference
//! means something -- a late GPU write from the buffer's previous tenant, or
//! any other writer -- landed on live data. Mismatches are reported on stderr
//! with the first differing offset; a running count is printed every 50 000
//! checks so an unarmed-looking run can be told from a silent one.
//!
//! Off, both entry points are one relaxed atomic load. Never the shipping
//! path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::Buffer;

fn armed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().instrument.upload_shadow)
}

fn map() -> &'static Mutex<HashMap<usize, Vec<u8>>> {
    static MAP: OnceLock<Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

static CHECKS: AtomicU64 = AtomicU64::new(0);
static MISMATCHES: AtomicU64 = AtomicU64::new(0);

/// Remember the bytes just uploaded to the buffer whose host address is `ptr`.
pub fn record(ptr: usize, bytes: &[u8]) {
    if !armed() {
        return;
    }
    let mut m = map().lock().unwrap_or_else(|e| e.into_inner());
    m.insert(ptr, bytes.to_vec());
}

/// Compare the buffer's current bytes with the recorded upload, if any.
pub fn check_at_release(buffer: &Buffer) {
    if !armed() {
        return;
    }
    let ptr = buffer.contents() as usize;
    let recorded = {
        let mut m = map().lock().unwrap_or_else(|e| e.into_inner());
        m.remove(&ptr)
    };
    let Some(recorded) = recorded else { return };
    let n = CHECKS.fetch_add(1, Ordering::Relaxed) + 1;
    // SAFETY: the buffer is Shared storage, mapped for the host, and at least
    // `recorded.len()` bytes long (that is what was uploaded into it).
    let now = unsafe { std::slice::from_raw_parts(buffer.contents(), recorded.len()) };
    if now != recorded.as_slice() {
        let first = now
            .iter()
            .zip(&recorded)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let total = MISMATCHES.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "[upload shadow] MISMATCH #{total} ptr=0x{ptr:x} len={} first_diff@{first} was={:02x?} now={:02x?} (check {n})",
            recorded.len(),
            &recorded[first..(first + 8).min(recorded.len())],
            &now[first..(first + 8).min(now.len())],
        );
    }
    if n.is_multiple_of(50_000) {
        eprintln!(
            "[upload shadow] {n} checks, {} mismatches",
            MISMATCHES.load(Ordering::Relaxed)
        );
    }
}
