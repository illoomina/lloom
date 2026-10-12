//! The crash recorder's signal route calls libSystem from a signal handler. A
//! lazily bound import would enter dyld's stub binder on its first call there,
//! taking dyld's lock in a context that may have interrupted dyld itself, so
//! every link target of this package binds its imports at load instead.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Set by sancov-wrapper.sh on the ASan arm only.
    println!("cargo::rustc-check-cfg=cfg(lloom_fuzz_asan)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-bind_at_load");
    }
}
