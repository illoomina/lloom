//! Source-audited requested Arc Layout policy; legacy builds never fail closed.
//! FNV/length detect source drift, NOT authenticity. Trust boundary: official
//! rustup matching compiler/std/source distribution, no replacement sysroot.
use std::{env, fs, path::Path, process::Command};
fn output(rustc: &std::ffi::OsStr, args: &[&str]) -> Option<String> {
    let result = Command::new(rustc).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8(result.stdout).ok())
        .flatten()
}
fn matches_source(root: &Path, path: &str, length: usize, expected: u64) -> bool {
    let source = root.join(path);
    let Ok(bytes) = fs::read(source) else {
        return false;
    };
    let fingerprint = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    bytes.len() == length && fingerprint == expected
}
fn main() {
    println!("cargo:rustc-check-cfg=cfg(lloom_audited_arc_layout)");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let supported = (|| {
        // A substituted compilation sysroot is not the audited distribution.
        if env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .contains("--sysroot")
        {
            return None;
        }
        let rustc = env::var_os("RUSTC")?;
        let version = output(&rustc, &["-vV"])?;
        let sysroot = output(&rustc, &["--print", "sysroot"])?;
        let source = Path::new(sysroot.trim()).join("lib/rustlib/src/rust/library");
        // Declare BOTH paths even when absent or the first validation fails.
        // Later source installation/drift must invalidate Cargo's cached policy.
        for (path, _, _) in SOURCE_CHECKS {
            println!("cargo:rerun-if-changed={}", source.join(path).display());
        }
        let target = env::var("TARGET").ok()?;
        let audited = AUDITED_COMPILERS.iter().any(|(commit, audited_target)| {
            target == *audited_target
                && version
                    .lines()
                    .any(|line| line.strip_prefix("commit-hash: ") == Some(*commit))
        });
        if !audited {
            return None;
        }

        if SOURCE_CHECKS.iter().all(|(path, length, hash)| {
            // Official CI image omits rust-src. Exact compiler/target's
            // published source audit is the fallback; a present divergent
            // source remains a refusal until its provenance is reconciled.
            !source.join(path).exists() || matches_source(&source, path, *length, *hash)
        }) {
            Some(())
        } else {
            None
        }
    })()
    .is_some();
    if supported {
        println!("cargo:rustc-cfg=lloom_audited_arc_layout");
    } else {
        println!("cargo:warning=owned requested Arc quotes unavailable: audit matching std source/toolchain; legacy builds and allocation APIs remain available");
    }
}
const SOURCE_CHECKS: &[(&str, usize, u64)] = &[
    ("alloc/src/sync.rs", 176642, 0xe76e897febad33f7),
    ("core/src/sync/atomic.rs", 199166, 0x618ee2fca340369f),
];

// Each exact official distribution entry has independently retrieved matching
// source audit; unknown commits/targets never inherit this table by version.
const AUDITED_COMPILERS: &[(&str, &str)] = &[
    (
        "ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96",
        "aarch64-apple-darwin",
    ),
    (
        "31fca3adb283cc9dfd56b49cdee9a96eb9c96ffd",
        "x86_64-unknown-linux-gnu",
    ),
];
