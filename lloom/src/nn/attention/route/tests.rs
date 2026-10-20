//! The routing and workspace rules, against a stated kernel set.
//!
//! The kernel set below is a fixture, not the device's: it states which head
//! dims each family serves so each rule is exercised alone. Whether the Metal
//! set matches the compiled kernels is the GPU tests' question.

use super::*;

/// The one-query kernels serve these head dims; the tiled kernel all of
/// `FULL` (not 512 at f32), with a key block of 32 below 128 and 16 above.
struct Fixture {
    two_pass_from: usize,
}

const VECTOR: [usize; 6] = [32, 64, 96, 128, 256, 512];
const FULL: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];
const BLOCKS: usize = 32;

impl KernelSet for Fixture {
    fn vector(&self, head_dim: usize, dtype: DType) -> bool {
        VECTOR.contains(&head_dim) && !(head_dim == 512 && dtype == DType::F32)
    }
    fn full_key_block(&self, head_dim: usize, dtype: DType) -> Option<usize> {
        if !FULL.contains(&head_dim) || (head_dim == 512 && dtype == DType::F32) {
            return None;
        }
        Some(if head_dim < 128 { 32 } else { 16 })
    }
    fn two_pass_from(&self, _head_dim: usize, _dtype: DType) -> usize {
        self.two_pass_from
    }
    fn two_pass_blocks(&self) -> usize {
        BLOCKS
    }
}

const K: Fixture = Fixture {
    two_pass_from: 1024,
};

fn spec(head_dim: usize, mask: MaskKind) -> AttentionSpec {
    AttentionSpec {
        q_heads: 32,
        kv_heads: 8,
        head_dim,
        v_head_dim: head_dim,
        dtype: DType::F16,
        kv_dtype: DType::F16,
        mask,
        softcap: 1.0,
        layout: KvLayout::Strided,
        bounds: AttentionBounds {
            batch: 2,
            min_queries: 1,
            queries: 512,
            keys: 4096,
        },
    }
}

fn prepare(s: AttentionSpec) -> PreparedAttention {
    PreparedAttention::prepare(s, &K).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn one_unmasked_query_takes_the_vector_kernel_and_two_passes_from_the_threshold() {
    let r = prepare(spec(64, MaskKind::Causal));
    assert_eq!(r.kernel_for(1, 1, 1023).unwrap(), AttentionKernel::Vector);
    assert_eq!(
        r.kernel_for(1, 1, 1024).unwrap(),
        AttentionKernel::VectorTwoPass
    );
    assert_eq!(r.kernel_for(1, 2, 1024).unwrap(), AttentionKernel::Full);
}

/// A masked single query cannot take the vector kernel, which has no mask
/// input; it takes the tiled kernel, whose partial-query tile serves it.
#[test]
fn a_masked_single_query_takes_the_tiled_kernel() {
    let r = prepare(spec(64, MaskKind::Additive));
    assert_eq!(r.kernel_for(1, 1, 1).unwrap(), AttentionKernel::Full);
    assert_eq!(r.kernel_for(1, 1, 4096).unwrap(), AttentionKernel::Full);
}

/// 72 and 80 have tiled kernels and no one-query entry point: a one-query
/// request takes the tiled kernel instead of failing after allocating.
#[test]
fn head_dims_without_a_vector_entry_take_the_tiled_kernel_for_one_query() {
    for hd in [72, 80] {
        let r = prepare(spec(hd, MaskKind::None));
        assert_eq!(
            r.kernel_for(1, 1, 100).unwrap(),
            AttentionKernel::Full,
            "hd {hd}"
        );
    }
    let mut capped = spec(72, MaskKind::None);
    capped.softcap = 30.0;
    let err = PreparedAttention::prepare(capped, &K).unwrap_err();
    assert_eq!(err.phase, "single-query");
    assert!(err.need.contains("soft cap"), "{err}");
}

/// A shape no kernel serves is refused at preparation with the capability
/// named, never routed to the composed path.
#[test]
fn an_uncovered_shape_is_refused_at_preparation_with_its_capability_named() {
    let err = PreparedAttention::prepare(spec(48, MaskKind::Causal), &K).unwrap_err();
    assert!(err.need.contains("head dim 48"), "{err}");

    let mut f32_512 = spec(512, MaskKind::Causal);
    f32_512.dtype = DType::F32;
    f32_512.kv_dtype = DType::F32;
    assert!(PreparedAttention::prepare(f32_512, &K).is_err());

    let mut gqa = spec(64, MaskKind::Causal);
    gqa.kv_heads = 5;
    assert!(PreparedAttention::prepare(gqa, &K).is_err());

    // A soft cap the vector kernel serves is fine for decode bounds alone.
    let mut decode_only = spec(64, MaskKind::None);
    decode_only.softcap = 30.0;
    decode_only.bounds.queries = 1;
    assert!(PreparedAttention::prepare(decode_only, &K).is_ok());
    decode_only.bounds.queries = 2;
    let err = PreparedAttention::prepare(decode_only, &K).unwrap_err();
    assert_eq!(err.phase, "multi-query");
}

/// One query over a pool is flash decoding's, so a pool route is a window
/// route; its pages hold whole key blocks of the tiled kernel.
#[test]
fn a_page_pool_takes_the_tiled_kernel_at_a_whole_key_block() {
    let mut paged = spec(64, MaskKind::Causal);
    paged.layout = KvLayout::Paged { page_size: 64 };
    let err = PreparedAttention::prepare(paged, &K).unwrap_err();
    assert_eq!(err.phase, "single-query");

    paged.bounds.min_queries = 2;
    let r = prepare(paged);
    assert_eq!(r.kernel_for(2, 2, 100).unwrap(), AttentionKernel::Full);
    assert!(
        r.kernel_for(1, 1, 100).is_err(),
        "one query on a window route"
    );
    assert!(r.describe().contains("one query -> none (windows only)"));

    paged.layout = KvLayout::Paged { page_size: 48 };
    let err = PreparedAttention::prepare(paged, &K).unwrap_err();
    assert!(err.need.contains("key block 32"), "{err}");

    let mut masked = paged;
    masked.layout = KvLayout::Paged { page_size: 64 };
    masked.mask = MaskKind::Additive;
    assert!(PreparedAttention::prepare(masked, &K).is_err());
}

#[test]
fn a_request_beyond_the_bounds_is_refused_before_it_is_sized() {
    let r = prepare(spec(64, MaskKind::Causal));
    assert!(r.kernel_for(3, 1, 10).is_err(), "batch");
    assert!(r.kernel_for(1, 513, 1000).is_err(), "queries");
    assert!(r.kernel_for(1, 1, 4097).is_err(), "keys");
    assert!(
        r.kernel_for(1, 20, 10).is_err(),
        "masked queries past the keys"
    );
    assert!(r.workspace(1, 20, 10).is_err());
    assert!(r.kernel_for(0, 1, 10).is_err(), "empty batch");
}

/// The two-pass scratch is the kernel's partial outputs, maxima and sums:
/// `4 * batch * q_heads * blocks * (v_head_dim + 2)` bytes, from the route.
#[test]
fn the_workspace_counts_output_scratch_conversion_mask_and_metadata() {
    let r = prepare(spec(64, MaskKind::None));
    let one = r.workspace(2, 1, 100).unwrap();
    assert_eq!(one.output, 2 * 32 * 64 * 2);
    assert_eq!(one.scratch, 0, "single pass below the threshold");
    let two = r.workspace(2, 1, 2048).unwrap();
    assert_eq!(two.scratch, 4 * 2 * 32 * BLOCKS * (64 + 2));
    assert_eq!(two.output, one.output);

    let mut f32_q = spec(64, MaskKind::Causal);
    f32_q.dtype = DType::F32;
    let r = prepare(f32_q);
    let w = r.workspace(1, 4, 100).unwrap();
    assert_eq!(w.conversion, 2 * 8 * 100 * 64 * 4, "f16 K and V up to f32");

    let r = prepare(spec(64, MaskKind::Additive));
    let w = r.workspace(2, 4, 100).unwrap();
    assert_eq!(w.mask, 2 * 32 * 4 * 100 * 2);

    let mut paged = spec(64, MaskKind::Causal);
    paged.layout = KvLayout::Paged { page_size: 64 };
    paged.bounds.min_queries = 2;
    let r = prepare(paged);
    let w = r.workspace(2, 4, 130).unwrap();
    assert_eq!(
        w.metadata,
        2 * 3 * 4 + 2 * 4,
        "three pages a row, one count a row"
    );
    assert_eq!(w.conversion, 0);
}

#[test]
fn the_worst_case_workspace_bounds_every_request_within_the_bounds() {
    for mask in [MaskKind::None, MaskKind::Causal, MaskKind::Additive] {
        let r = prepare(spec(64, mask));
        let worst = r.worst_case_workspace().unwrap();
        let b = r.spec().bounds;
        for batch in [1, b.batch] {
            for queries in [1, 2, 17, b.queries] {
                for keys in [1, 17, 1023, 1024, 2000, b.keys] {
                    let Ok(w) = r.workspace(batch, queries, keys) else {
                        continue;
                    };
                    assert!(
                        w.total().unwrap() <= worst.total().unwrap()
                            && w.output <= worst.output
                            && w.scratch <= worst.scratch,
                        "{mask:?} {batch}x{queries}x{keys}: {w:?} above {worst:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_route_line_names_both_phases() {
    let line = prepare(spec(64, MaskKind::Causal)).describe();
    assert!(line.contains("sdpa_vector_2pass from 1024 keys"), "{line}");
    assert!(line.contains("more -> steel_attention"), "{line}");
    let line = prepare(spec(80, MaskKind::Causal)).describe();
    assert!(line.contains("one query -> steel_attention"), "{line}");
}

/// The execution purpose admits by route, not by what selected it: on Metal
/// production refuses any composed attention, naming the reason it was
/// given and the purpose; the reference admits it; a prepared route (no
/// reason) is admitted under both; off Metal the composed attention is the
/// backend's own and both purposes admit it.
#[test]
fn production_refuses_a_composed_attention_on_metal_and_only_there() {
    use lloom_tensor::Device;
    let why = "a knob turned the fused route off";
    let cpu = Device::Cpu;
    for purpose in [ExecutionPurpose::Production, ExecutionPurpose::Reference] {
        purpose.admit("fixture", &cpu, Some(why)).unwrap();
        purpose.admit("fixture", &cpu, None).unwrap();
    }
    assert_eq!(ExecutionPurpose::default(), ExecutionPurpose::Reference);
    for s in ["production", "reference"] {
        assert_eq!(ExecutionPurpose::parse(s).unwrap().name(), s);
    }
    assert!(ExecutionPurpose::parse("serve").is_err());
    #[cfg(feature = "metal")]
    {
        let gpu = Device::new_metal(0).unwrap();
        ExecutionPurpose::Reference
            .admit("fixture", &gpu, Some(why))
            .unwrap();
        ExecutionPurpose::Production
            .admit("fixture", &gpu, None)
            .unwrap();
        let err = ExecutionPurpose::Production
            .admit("fixture", &gpu, Some(why))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with(&format!("fixture: {why}: "))
                && err.contains("materializes the scores")
                && err.contains("execution purpose `production`"),
            "{err}"
        );
    }
}
