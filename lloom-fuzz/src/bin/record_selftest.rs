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
//!
//! `--planted-fault` adds a second, independent defect: an input whose first
//! byte is `F` loads from a page mapped without access
//! ([`lloom_fuzz::planted`]), so the signal route is exercised by a fault the
//! kernel delivers inside instrumented code. The harness prints the address
//! before the load, for comparison with the record's `si_addr`.

use clap::Parser;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};

static LAST_FIRST_BYTE: AtomicU8 = AtomicU8::new(0);

#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    cli: lloom_fuzz::campaign::Cli,
    /// Plant a real protection fault on inputs starting with `F`.
    #[arg(long)]
    planted_fault: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let planted = args.planted_fault;
    let config = if planted {
        serde_json::json!({ "planted_fault": true })
    } else {
        serde_json::Value::Null
    };
    lloom_fuzz::campaign::run_with_config(&args.cli, "record_selftest", &config, move |bytes| {
        let first = bytes.first().copied().unwrap_or(0);
        let prev = LAST_FIRST_BYTE.swap(first, Ordering::AcqRel);
        if planted && first == lloom_fuzz::planted::FAULT_TRIGGER {
            let address = lloom_fuzz::planted::fault_address(bytes);
            println!(
                "PLANTED_FAULT pid={} addr={address:#018x}",
                std::process::id()
            );
            std::io::stdout().flush().ok();
            lloom_fuzz::planted::fault_if_triggered(bytes);
        }
        if first == 0x43 && prev == 0x42 {
            return Err(format!(
                "planted state-carryover failure: first byte {first:#04x} after an input whose \
                 first byte was {prev:#04x}"
            ));
        }
        Ok(())
    })
}
