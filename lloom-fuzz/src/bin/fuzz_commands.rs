//! The command queue's sequence fuzzer (`commands_fuzz`), coverage-guided.

use clap::Parser;
use lloom_tensor::kernels::metal::commands_fuzz::run_sequence;
use lloom_tensor::kernels::metal::device::Device;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    // Created on the first execution, in the process that executes: the
    // launcher's parent never opens a device (a queue created in a child of a
    // process holding one failed in `initWithDevice:` on every execution --
    // `issue-706-raw/probes.log`).
    let dev: std::cell::OnceCell<Device> = std::cell::OnceCell::new();
    lloom_fuzz::campaign::run(&cli, "commands", move |bytes| {
        let ops = lloom_fuzz::decode::commands(bytes);
        let dev = dev.get_or_init(|| Device::system_default().expect("no Metal device"));
        run_sequence(dev, &ops).map(|_| ())
    })
}
