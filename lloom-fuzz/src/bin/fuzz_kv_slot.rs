//! The `KvSlot` sequence fuzzer (`lloom::models::lfm2::kv_slot_fuzz`),
//! coverage-guided, on the CPU (the proptest's device).

use clap::Parser;
use lloom::models::lfm2::kv_slot_fuzz::run_slot_sequence;
use lloom_tensor::Device;

fn main() -> anyhow::Result<()> {
    let cli = lloom_fuzz::campaign::Cli::parse();
    lloom_fuzz::campaign::run(&cli, "kv_slot", move |bytes| {
        let (capacity, ops) = lloom_fuzz::decode::kv_slot(bytes);
        run_slot_sequence(capacity, &ops, &Device::Cpu).map(|_| ())
    })
}
