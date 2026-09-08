//! Empty `__sanitizer_cov_trace_*cmp*` hooks for the non-cmplog build.
//!
//! Every build of this crate is compiled with
//! `-sanitizer-coverage-trace-compares` (`.cargo/config.toml`), so the two arms
//! run the SAME codegen and differ only in who receives the comparison
//! callbacks: with the `cmplog` feature `libafl_targets` provides them and fills
//! the cmplog map; without it these no-ops satisfy the linker. One set of
//! instrumentation flags is what keeps "the pc-guard arm" and "the cmplog arm"
//! a property of the manifest rather than of two invocations (§14.5a).
//!
//! The cost on the regular arm is a call to an empty `extern "C"` function per
//! instrumented comparison. With LTO it is a candidate for inlining to nothing;
//! measured either way in the report rather than assumed.

#![cfg(not(feature = "cmplog"))]

macro_rules! cmp_stub {
    ($name:ident, $t:ty) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(_a: $t, _b: $t) {}
    };
}

cmp_stub!(__sanitizer_cov_trace_cmp1, u8);
cmp_stub!(__sanitizer_cov_trace_cmp2, u16);
cmp_stub!(__sanitizer_cov_trace_cmp4, u32);
cmp_stub!(__sanitizer_cov_trace_cmp8, u64);
cmp_stub!(__sanitizer_cov_trace_const_cmp1, u8);
cmp_stub!(__sanitizer_cov_trace_const_cmp2, u16);
cmp_stub!(__sanitizer_cov_trace_const_cmp4, u32);
cmp_stub!(__sanitizer_cov_trace_const_cmp8, u64);

#[unsafe(no_mangle)]
pub extern "C" fn __sanitizer_cov_trace_switch(_val: u64, _cases: *const u64) {}
