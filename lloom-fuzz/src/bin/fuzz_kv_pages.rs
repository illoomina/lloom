//! The paged KV pool fuzzer (`lloom_fuzz::kv_pages_model`), coverage-guided.
//! No device is touched: the pool, the quarantine and the clock are host-side.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    lloom_fuzz::campaign::run(&cli, "kv_pages", move |bytes| {
        let (fixture, ops) = lloom_fuzz::decode::kv_pages(bytes);
        lloom_fuzz::kv_pages_model::run_sequence(&fixture, &ops).map(|_| ())
    })
}
