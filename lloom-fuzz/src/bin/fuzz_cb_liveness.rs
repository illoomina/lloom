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
    let dev = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    let name = format!("cb_liveness_{}", args.cb);
    lloom_fuzz::campaign::run(&args.cli, &name, move |bytes| {
        let ops = lloom_fuzz::decode::cb_liveness(bytes);
        lloom_fuzz::cb_liveness::run_sequence(&dev, &ops).map(|_| ())
    })
}
