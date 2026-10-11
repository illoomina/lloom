//! Command-buffer liveness under retained and unretained command buffers
//! (`lloom_fuzz::cb_liveness`), with the pool's free poison as the detector.
//!
//! `--cb unretained` sets `LLOOM_CB_UNRETAINED=1` for this process and the
//! clients the launcher spawns, which is the switch `Commands` reads once.

use clap::Parser;
use lloom_tensor::kernels::metal::device::Device;

#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    cli: lloom_fuzz::campaign::Cli,
    /// `retained` (Metal's default) or `unretained`
    /// (`commandBufferWithUnretainedReferences`).
    #[arg(long, default_value = "retained")]
    cb: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.cb.as_str() {
        "unretained" => std::env::set_var("LLOOM_CB_UNRETAINED", "1"),
        "retained" => std::env::set_var("LLOOM_CB_UNRETAINED", "0"),
        other => anyhow::bail!("--cb {other}: expected retained | unretained"),
    }
    // Created on the first execution, in the process that executes: the
    // launcher's parent never opens a device (a queue created in a child of a
    // process holding one failed in `initWithDevice:` on every execution --
    // `issue-706-raw/probes.log`).
    let dev: std::cell::OnceCell<Device> = std::cell::OnceCell::new();
    let name = format!("cb_liveness_{}", args.cb);
    lloom_fuzz::campaign::run_with_config(
        &args.cli,
        &name,
        &serde_json::json!({"cb": args.cb}),
        move |bytes| {
            let ops = lloom_fuzz::decode::cb_liveness(bytes);
            let dev = dev.get_or_init(|| Device::system_default().expect("no Metal device"));
            lloom_fuzz::cb_liveness::run_sequence(dev, &ops).map(|_| ())
        },
    )
}
