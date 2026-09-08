//! The buffer pool's sequence fuzzer (`buffer_pool_fuzz`), coverage-guided.

use clap::Parser;
use lloom_tensor::kernels::metal::buffer_pool_fuzz::run_sequence;
use lloom_tensor::kernels::metal::device::Device;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    let dev = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    lloom_fuzz::campaign::run(&cli, "buffer_pool", move |bytes| {
        let ops = lloom_fuzz::decode::buffer_pool(bytes);
        run_sequence(&dev, &ops)
    })
}
