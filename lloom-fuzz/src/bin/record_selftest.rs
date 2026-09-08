//! The persistent record's validation target (lloom #706): a harness whose
//! failure depends on the PREVIOUS input, so that the crashing input alone
//! replays clean and the record replays red.
//!
//! **Planted, and labelled as such.** Every engine harness in this crate
//! constructs its pool / queue / ring per input, and the one process-wide
//! state (the Metal device's own pool under `token_ring --device metal`) has
//! no cheap seeded defect whose failure depends on the previous input. The
//! record's *function* is what this validates — the ring, the dump on an
//! oracle failure, the ordered replay — on a dependency written here rather
//! than found in the engine:
//!
//! the harness keeps one process-global byte, the first byte of the last
//! input it saw; an input whose first byte is `0x43` fails **iff** that
//! global reads `0x42`. Neither input fails alone.

use clap::Parser;
use std::sync::atomic::{AtomicU8, Ordering};

static LAST_FIRST_BYTE: AtomicU8 = AtomicU8::new(0);

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    lloom_fuzz::campaign::run(&cli, "record_selftest", move |bytes| {
        let first = bytes.first().copied().unwrap_or(0);
        let prev = LAST_FIRST_BYTE.swap(first, Ordering::AcqRel);
        if first == 0x43 && prev == 0x42 {
            return Err(format!(
                "planted state-carryover failure: first byte {first:#04x} after an input whose \
                 first byte was {prev:#04x}"
            ));
        }
        Ok(())
    })
}
