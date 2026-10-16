//! One integration-test binary for the crate (lloom #939): every
//! `tests/*.rs` file is a module here, so the suite links the engine once
//! instead of once per file, and a new test file is one `mod` line. The
//! files keep their names and their `#![cfg(feature = ...)]` gates. Each
//! test still runs in its own process under `cargo nextest run` (the gate's
//! runner); under plain `cargo test` the tests share a process, and a test
//! that installs process-global state fails by name when another test did
//! (`lloom_tensor::isolation`).
//!
//! Their own binaries, because a test binary carries at most one
//! `#[global_allocator]`: `token_ring_nonblocking.rs`, `decode_step_census.rs`.
//!
//! `every_test_file_is_a_module` below reads this directory and this file,
//! so a file that is not a module here fails the suite instead of silently
//! not running.
mod accept_frontier;
mod axes_registry;
mod batch_norm;
mod batched_decode_tests;
mod block_traits;
mod cpu_flash_attn;
mod cpu_numerics_probe;
mod decode_driver_tests;
mod decoder_memory;
mod decoder_standing_window;
mod decoder_turn_checkpoint;
mod dspark_decode_identity;
mod emulated_bf16_load;
mod fixture_switch;
mod flash_decoding_batch;
mod gdn_step_forms;
mod gemv_norm;
mod gemv_swiglu_shape;
mod generation_tests;
mod group_norm;
mod hub_routing;
mod kv_cache;
mod layer_norm;
mod lfm2_admission_device_tests;
mod lfm2_admission_tests;
mod lfm2_attn_head_dim_fallback;
mod lfm2_batched_rows;
mod lfm2_conv_ring_tests;
mod lfm2_dspark_backbone;
mod lfm2_dspark_batched_draft;
mod lfm2_dspark_ctx_paged;
mod lfm2_dspark_projection;
mod lfm2_fusion_derived;
mod lfm2_fusion_derived_refusal;
mod lfm2_gemv_epilogue_census;
mod lfm2_gemv_epilogue_wiring;
mod lfm2_hidden_state_tap;
mod lfm2_kv_growth;
mod lfm2_kv_growth_metal;
mod lfm2_kv_growth_refusal_metal;
mod lfm2_paged_kv_blocker;
mod lfm2_pass_plans;
mod lfm2_qkv_layout;
mod lfm2_sliding_ring_open;
mod lfm2_spec_conv_rollback;
mod lfm2_spec_conv_shape;
mod lfm2_streamed_layers;
mod lfm2_vl_embeds_entry;
mod lfm2_walk_census;
mod lfm2_zero_sites;
mod logits_processors;
mod logits_processors_readback;
mod logits_slot_readback;
mod loss;
mod metal_reduce_barrier_tests;
mod metal_weight_view_offset;
mod nms_tests;
mod one_hot;
mod ops;
mod optim;
mod optimizer_derivation;
mod optimizer_fp_model;
mod optimizer_quant_hooks;
mod optimizer_remarks;
mod optimizer_walk;
mod page_cache_probe;
mod pos_embed_resample_tests;
mod prefill_window_gemm;
mod prepare_path;
mod quant_kind_throughput;
mod qwen3_5_mtp;
mod qwen3_5_tiny;
mod qwen4exp_batched_rows;
mod qwen4exp_chunk_past_dense;
mod qwen4exp_decode_plan_variants;
mod qwen4exp_device_plan;
mod qwen4exp_dflash;
mod qwen4exp_dispatch_census;
mod qwen4exp_engine_identity;
mod qwen4exp_eviction_no_drain;
mod qwen4exp_fetch;
mod qwen4exp_fusion_derived;
mod qwen4exp_host_feed;
mod qwen4exp_host_plan_requests;
mod qwen4exp_hotlist_warm;
mod qwen4exp_moe_act_tile;
mod qwen4exp_mtp_carries;
mod qwen4exp_mtp_identity;
mod qwen4exp_mtp_observe_plan;
mod qwen4exp_mtp_probe;
mod qwen4exp_mtp_streamed;
mod qwen4exp_mtpvar;
mod qwen4exp_pass_plan_openers;
mod qwen4exp_pass_plan_requests;
mod qwen4exp_pass_plans_warmup;
mod qwen4exp_prefill_plan_variants;
mod qwen4exp_provisional;
mod qwen4exp_ring_provisional;
mod qwen4exp_routed_dense_arm;
mod qwen4exp_rows_timing;
mod qwen4exp_scale_admission;
mod qwen4exp_schema_real;
mod qwen4exp_shape_identity;
mod qwen4exp_shared_device;
mod qwen4exp_split_tile;
mod qwen4exp_streamed_identity;
mod qwen4exp_streamed_metal;
mod qwen4exp_tiny_determinism;
mod qwen4exp_tiny_metal_blocks;
mod qwen4exp_turn_checkpoint;
mod qwen4exp_verify_window;
mod qwen4exp_walk;
mod qwen4exp_walk_census;
mod qwen4exp_window_rewind;
mod qwen_eos_gate;
mod recovery_frontier;
mod recurrent_pool_write;
mod recurrent_state_metal;
mod ring_publish_binds;
mod ring_publish_order;
mod rnn;
mod sampled_cells;
mod scale_synthetic;
mod sdpa;
mod sdpa_full_variants;
mod sdpa_masked_query;
mod sdpa_vector_head_dims;
mod sequence_reuse_tests;
mod shape_probe_sites;
mod short_conv_block;
mod short_conv_golden;
mod shortconv_ring;
mod shortconv_step;
mod state_capacity;
mod terminal_trace;
mod tiny_checkpoints;
mod tiny_fp_kinds;
mod tiny_manifest_drift;
mod tiny_packed_metal;
mod token_ring_fuzz;
mod token_ring_tests;
mod turn_checkpoint;
mod turn_checkpoint_policy;
mod varlen_flash_attention;
mod verify_window_gemm;

/// Every `tests/*.rs` beside this file is declared above (or is one of the
/// files that owns its binary), read from the directory at test time.
#[test]
fn every_test_file_is_a_module() {
    let own: &[&str] = &["token_ring_nonblocking", "decode_step_census"];
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
