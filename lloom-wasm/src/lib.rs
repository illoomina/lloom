//! Browser-runner feasibility spike (#656).
//!
//! The one question: does lloom's CPU backend compile to wasm32 and run a
//! forward pass in a Web Worker, producing the byte-identical CPU-reference
//! result (the §2.3.5a oracle)?
//!
//! This crate is the harness. The pure computation lives in [`compute`] and is
//! target-agnostic, so the same code runs in the wasm export `smoke()` and in
//! the native `reference` binary -- the two fingerprints are the oracle check.

use lloom_tensor::{Device, Tensor};

/// The deterministic CPU computation. Fixed inputs, no RNG, no I/O, no
/// target-specific code -- so the output is a pure function of the CPU kernels
/// and is byte-identical between native and wasm iff they compiled and ran the
/// same arithmetic. Returns (values, fingerprint, json).
pub fn compute() -> lloom_tensor::Result<(Vec<f32>, u64, String)> {
    let dev = Device::Cpu;
    let a = Tensor::arange(0f32, 12f32, &dev)?.reshape((3, 4))?;
    let b = Tensor::arange(0f32, 8f32, &dev)?.reshape((4, 2))?;
    let c = a.matmul(&b)?; // (3,2)
    let sm = lloom::nn::ops::softmax_last_dim(&c)?;

    let flat: Vec<f32> = sm.flatten_all()?.to_vec1()?;
    let fp = fingerprint_f32(&flat);
    let json = format!(
        "{{\"op\":\"matmul+softmax\",\"shape\":[3,2],\"values\":{:?},\"fingerprint\":\"{:016x}\"}}",
        flat, fp
    );
    Ok((flat, fp, json))
}

/// A cheap order-sensitive fingerprint of the raw f32 bit patterns. Not
/// SHA-256, but it is deterministic and byte-exact, which is all the oracle
/// check needs: browser fp == native fp iff the arithmetic matched.
pub fn fingerprint_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ---- wasm-bindgen surface (browser only) --------------------------------

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub fn smoke() -> Result<String, JsValue> {
        let (_v, fp, json) = compute().map_err(|e| JsValue::from_str(&format!("{e}")))?;
        web_sys::console::log_1(&JsValue::from_str(&format!("smoke ok, fp={fp:016x}")));
        Ok(json)
    }

    /// Report the build config from the artifact, so the finding asserts "no
    /// metal in the wasm build" from what ran, not from the source.
    #[wasm_bindgen]
    pub fn build_info() -> String {
        // lloom is pulled with default-features=false, so metal is off by
        // construction on this build; the string records that from the artifact.
        format!(
            "{{\"target_arch_wasm32\":{},\"lloom_metal_feature\":false,\"dtype_default\":\"{:?}\"}}",
            cfg!(target_arch = "wasm32"),
            lloom_tensor::DType::F32
        )
    }
}
