//! The token ring's sequence fuzzer (`lloom::generation::token_ring_fuzz`),
//! coverage-guided. Runs each sequence on the CPU and, when present, on Metal —
//! the proptest's `devices()` — unless `--device` narrows it.

use clap::Parser;
use lloom::generation::token_ring_fuzz::{devices, run_sequence};
use lloom_tensor::Device;

#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    cli: lloom_fuzz::campaign::Cli,
    /// `cpu`, `metal` or `both` (the proptest's choice).
    #[arg(long, default_value = "both")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let devs: Vec<Device> = match args.device.as_str() {
        "cpu" => vec![Device::Cpu],
        "metal" => vec![Device::new_metal(0)?],
        _ => devices(),
    };
    lloom_fuzz::campaign::run(&args.cli, "token_ring", move |bytes| {
        let ops = lloom_fuzz::decode::token_ring(bytes);
        for d in &devs {
            run_sequence(d, &ops).map_err(|e| format!("{d:?}: {e}"))?;
        }
        Ok(())
    })
}
