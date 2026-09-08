//! The command queue's sequence fuzzer (`commands_fuzz`), coverage-guided.

use clap::Parser;
use lloom_tensor::kernels::metal::commands_fuzz::run_sequence;
use lloom_tensor::kernels::metal::device::Device;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    let dev = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    lloom_fuzz::campaign::run(&cli, "commands", move |bytes| {
        let ops = lloom_fuzz::decode::commands(bytes);
        run_sequence(&dev, &ops).map(|_| ())
    })
}
