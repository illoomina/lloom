//! Campaign identity prepared once, before target execution. Not a schedule log.
use sha2::{Digest, Sha256};
use std::io::{self, Read};

/// Hash the executable file with fixed scratch storage, not a whole-file Vec.
/// This identifies bytes at current_exe(), not loaded dylibs, driver/hardware,
/// mutable environment or an immutable snapshot of the running process image.
fn executable_digest(mut reader: impl Read) -> io::Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Canonical JSON carries explicit coverage limits. Callers with target options
/// use campaign::run_with_config; unknown environment/native state is NOT silently
/// promoted to verified configuration. Schedule hooks add their own schema later.
pub(crate) fn campaign(
    name: &str,
    arm: &str,
    timeout_ms: u64,
    max_len: usize,
    target: &serde_json::Value,
) -> io::Result<Vec<u8>> {
    let binary = executable_digest(std::fs::File::open(std::env::current_exe()?)?)?;
    serde_json::to_vec(&serde_json::json!({
        "schema": "lloom-campaign-identity-v1",
        "executable_sha256": binary,
        "target": name,
        "instrumentation": arm,
        "input_schema": "target-raw-bytes-v1",
        "scheduling": "uncontrolled-os-no-schedule-record",
        "campaign": {"timeout_ms": timeout_ms, "max_len": max_len},
        "target_configuration": target,
        "configuration_coverage": "explicit target/campaign only; environment, dylibs, hardware and driver not captured"
    })).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_executable_digest_matches_known_bytes() {
        assert_eq!(
            executable_digest(&b"abc"[..]).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
