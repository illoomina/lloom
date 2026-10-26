//! One integration-test binary for the crate (lloom #939): every
//! `tests/*.rs` file is a module here, so the suite links the engine once
//! instead of once per file, and a new test file is one `mod` line. The
//! files keep their names and their `#![cfg(feature = ...)]` gates. Each
//! test still runs in its own process under `cargo nextest run` (the gate's
//! runner); under plain `cargo test` the tests share a process, and a test
//! that installs process-global state fails by name when another test did
//! (`lloom_tensor::isolation`).
//!
//! Their own binaries: `metal_pool_alloc_free.rs`, because a test binary
//! carries at most one `#[global_allocator]`; `metal_cb_failure_tests.rs`,
//! because it makes the GPU fail command buffers on purpose and a fault must
//! not share a process with tests it could disturb.
//!
//! `every_test_file_is_a_module` below reads this directory and this file,
//! so a file that is not a module here fails the suite instead of silently
//! not running.
#[cfg(not(feature = "metal"))]
use lloom_tensor::retire;

mod bilinear_tests;
mod conv_tests;
mod cpu_f16_matmul_accum;
mod custom_op_tests;
mod depthwise_conv;
mod display_tests;
mod eos_gate_drain;
mod error_display;
mod error_size;
mod f16_rows_axis;
mod f16_rows_decode;
mod f16_rows_identity;
mod f16_rows_round;
mod f16_rows_timing;
mod fp_kind_codes;
mod gate_contract_pinned;
mod gate_contract_unpinned;
mod gemm_row_batch_invariance;
mod gemv_mixed;
mod gemv_mixed_tiny_shapes;
mod gguf_tests;
mod grad_tests;
mod heap_transients;
mod host_fed_ring;
mod indexing_tests;
mod io_queue_held;
mod iq3xxs_gguf_read;
mod iq_metal;
mod kv_pages_cpu;
mod layout_tests;
mod matmul_tests;
mod metal4_slot_trace;
mod metal_arena_lifetime_fuzz;
mod metal_arena_teardown_tests;
mod metal_arg_sort_tests;
mod metal_blit_ordering_tests;
mod metal_buffer_pool_tests;
mod metal_concurrent_tests;
mod metal_dispatch_census_scope;
mod metal_dispatch_trace_scope;
mod metal_emulated_arms;
mod metal_encoder_counters;
mod metal_encoder_counters_limit;
mod metal_indexing_tests;
mod metal_rand_uniform_iid_tests;
mod metal_retire_order_tests;
mod metal_rng_seed_tests;
mod metal_shared_device_threads;
mod metal_strided_reduce_repeat_tests;
mod mmap_advice_installed;
mod moe_pair_order;
mod moe_routed_metal;
mod moe_routed_rows;
mod moe_routed_union;
mod options_derive;
mod pool_tests;
mod pread_targets;
mod pth_tests;
mod q2_0_metal;
mod qmatmul_dequant_once;
mod qmatmul_f16w_mm;
mod qmatmul_id_mm;
mod qmatmul_id_mm_arm;
mod qmatmul_id_mm_k_block;
mod qmatmul_rows_arm;
mod qmatmul_rows_ceiling;
mod qmatmul_rows_default;
mod qmatmul_rows_timing;
mod qmatmul_rows_under_fast;
mod qmm_f16w_tiny_shapes;
mod qmm_narrow_weights;
mod quant_mv_store;
#[cfg(feature = "metal")]
mod quant_parity;
mod quant_parity_matrix;
mod quant_parity_matrix_rows;
mod quantized_tests;
#[cfg(feature = "metal")]
mod residency_router_device;
#[cfg(feature = "metal")]
mod rows_identity;
mod sampling_contract;
mod serialization_tests;
mod slice_set_strided_tests;
mod staged_readback_plans;
mod tensor_tests;
mod typed_tests;
mod words_publish_order;

/// Every `tests/*.rs` beside this file is declared above (or is one of the
/// files that owns its binary), read from the directory at test time.
#[test]
fn every_test_file_is_a_module() {
    let own: &[&str] = &[
        "metal_pool_alloc_free",
        "metal_cb_failure_tests",
        "metal_commit_alloc_free",
        "allocation_refusal_alloc_free",
    ];
    let dir = format!("{}/tests", lloom_manifest_dir());
    let me = include_str!("main.rs");
    let mut missing = Vec::new();
    let mut files = 0;
    for entry in std::fs::read_dir(dir).expect("tests/ is readable") {
        let name = entry
            .expect("an entry")
            .file_name()
            .into_string()
            .expect("utf-8");
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        if stem == "main" || own.contains(&stem) {
            continue;
        }
        files += 1;
        if !me.lines().any(|l| l == format!("mod {stem};")) {
            missing.push(stem.to_string());
        }
    }
    assert!(files > 0, "no test files beside main.rs");
    assert!(
        missing.is_empty(),
        "test files that are not modules of tests/main.rs (add `mod <name>;`): {missing:?}"
    );
}

/// The crate's manifest directory **at run time** — what cargo and nextest
/// export to the test process — falling back to the build-time path. A test
/// binary archived on one host and run on another (`cargo nextest archive` /
/// `--workspace-remap`) reads its fixtures where the archive was unpacked;
/// `env!("CARGO_MANIFEST_DIR")` would bake the building host's path in
/// (measured 2026-10-13: `every_test_file_is_a_module` panicked with NotFound
/// on silverberry from a Mac-built archive).
fn lloom_manifest_dir() -> String {
    std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string())
}
