//! The buffer pool's sequence fuzzer (`buffer_pool_fuzz`), coverage-guided.

use clap::Parser;
use lloom_tensor::kernels::metal::buffer_pool_fuzz::run_sequence;
use lloom_tensor::kernels::metal::device::Device;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    // Created on the first execution, in the process that executes: the
    // launcher's parent never opens a device (a queue created in a child of a
    // process holding one failed in `initWithDevice:` on every execution --
    // `issue-706-raw/probes.log`).
    let dev: std::cell::OnceCell<Device> = std::cell::OnceCell::new();
    lloom_fuzz::campaign::run(&cli, "buffer_pool", move |bytes| {
        let ops = lloom_fuzz::decode::buffer_pool(bytes);
        let dev = dev.get_or_init(|| Device::system_default().expect("no Metal device"));
        run_sequence(dev, &ops)
    })
}
