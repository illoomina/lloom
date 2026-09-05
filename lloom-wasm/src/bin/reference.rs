//! Native CPU reference for the #656 oracle check. Runs the exact same
//! `lloom_wasm::compute()` on the host CPU backend and prints the fingerprint,
//! so the browser fingerprint can be compared byte-for-byte (§2.3.5a).

fn main() -> lloom_tensor::Result<()> {
    let (values, fp, json) = lloom_wasm::compute()?;
    println!("native CPU reference (#656)");
    println!("values      = {values:?}");
    println!("fingerprint = {fp:016x}");
    println!("json        = {json}");
    Ok(())
}
