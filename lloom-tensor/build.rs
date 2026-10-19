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
/// The path itself when it exists, else its nearest existing ancestor: what
/// `rerun-if-changed` may watch without declaring the build permanently stale.
fn watch_point(path: std::path::PathBuf) -> std::path::PathBuf {
    let mut p = path;
    while !p.exists() {
        match p.parent() {
            Some(parent) => p = parent.to_path_buf(),
            None => break,
        }
    }
    p
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
        // Watch both sources, present or absent. A present file: its drift
        // reruns this script. An absent one: Cargo treats a missing
        // `rerun-if-changed` path as always stale, which made every cargo
        // invocation on a host without `rust-src` (the CI images, the NAS,
        // the Pi) rerun this script and recompile lloom-tensor and everything
        // above it — measured 2026-10-13, `nas-runner-v2-2026-10.md` Run 12.
        // So an absent file is watched through its nearest existing ancestor
        // (`lib/rustlib/` in a rust-src-less toolchain): installing the
        // component creates `src/` under it and reruns the script once.
        for (path, _, _) in SOURCE_CHECKS {
            println!(
                "cargo:rerun-if-changed={}",
                watch_point(source.join(path)).display()
            );
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
// source audit (memory.md §9.10a); unknown commits/targets never inherit this
// table by version.
const AUDITED_COMPILERS: &[(&str, &str)] = &[
    (
        "ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96",
        "aarch64-apple-darwin",
    ),
    (
        "31fca3adb283cc9dfd56b49cdee9a96eb9c96ffd",
        "x86_64-unknown-linux-gnu",
    ),
    (
        "31fca3adb283cc9dfd56b49cdee9a96eb9c96ffd",
        "aarch64-unknown-linux-gnu",
    ),
];
