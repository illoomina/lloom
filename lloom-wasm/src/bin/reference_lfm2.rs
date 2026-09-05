//! Native LFM2 oracle for the #623 browser cell.
//!
//! Runs the exact same `lloom_wasm::lfm2::decode_one` the wasm export runs — a
//! real quantized-LFM2 forward pass, one greedily-sampled token — on the host
//! CPU backend, and prints the token and the logit fingerprint. The browser
//! then reproduces both byte-for-byte (§2.3.5a): if the fingerprints match, the
//! browser is running the real decode path, not a look-alike.
//!
//! Usage:
//!   reference_lfm2 <path-to.gguf> [prompt]

use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: reference_lfm2 <path-to.gguf> [prompt]")?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());

    let mut bytes = Vec::new();
    std::fs::File::open(&path)?.read_to_end(&mut bytes)?;

    let t0 = std::time::Instant::now();
    let d = lloom_wasm::lfm2::decode_one(bytes, &prompt, 8)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("native LFM2 reference (#623)");
    println!("source      = {path}");
    println!("prompt      = {:?}", d.prompt);
    println!("prompt_ids  = {:?}", d.prompt_ids);
    println!("token_id    = {}", d.token_id);
    println!("token_text  = {:?}", d.token_text);
    println!("logits_fp   = {:016x}", d.logits_fp);
    println!("forward_ms  = {ms:.1}");
    println!("top-8:");
    for (rank, (id, text, logit)) in d.topk.iter().enumerate() {
        println!("  {rank}. id={id:<8} logit={logit:+.4}  {text:?}");
    }

    // The one-line JSON the browser page compares against.
    let topk_json: Vec<String> = d
        .topk
        .iter()
        .map(|(id, text, logit)| {
            format!(
                "{{\"id\":{id},\"text\":{},\"logit\":{logit}}}",
                serde_json::to_string(text).unwrap()
            )
        })
        .collect();
    println!(
        "json        = {{\"token_id\":{},\"token_text\":{},\"logits_fp\":\"{:016x}\",\"topk\":[{}]}}",
        d.token_id,
        serde_json::to_string(&d.token_text)?,
        d.logits_fp,
        topk_json.join(",")
    );

    Ok(())
}
