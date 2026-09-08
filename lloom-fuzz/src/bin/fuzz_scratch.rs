//! The scratch planner's long-context fuzzer (`scratch_fuzz`), coverage-guided.
//! Pure arithmetic: no device.

use clap::Parser;
use lloom_tensor::kernels::metal::scratch_fuzz::run_sequence;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    lloom_fuzz::campaign::run(&cli, "scratch", move |bytes| {
        let c = lloom_fuzz::decode::scratch(bytes);
        run_sequence(c.geometry, c.layers, c.sizing, c.layout, &c.ops).map(|_| ())
    })
}
