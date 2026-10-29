//! The own-cache multi-query route reads pages directly without changing logits.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Config, KvAppend, LayerType, QkvLayout, Shape, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::{CrownExact, PoolExact};
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::TinyFamily;
use lloom_tensor::{DType, Device, Result, Tensor};

struct Run {
    logits: Vec<Vec<f32>>,
    index_selects: Vec<u64>,
    attention: Vec<u64>,
}

// DispatchCensus records pipeline names, not debug-group labels or dimensions.
fn is_index_select_pipeline(label: &str, dtype: DType) -> bool {
    let name = lloom_tensor::kernels::IndexingKernel::INDEX_SELECT
        .name("u32", dtype.as_str())
        .expect("tested index-select dtype");
    label == name || label.strip_suffix("_packed") == Some(name)
}

fn run<P: Shape>(
    fixture: &Fixture<Lfm2Tiny>,
    cfg: &Config,
    dtype: DType,
    dev: &Device,
) -> Result<Run> {
    let model = fixture.model_in::<P>(cfg, Some(dtype), dev)?;
    let mut cache = State::<P>::new(true, dtype, cfg, dev)?;
    let census = dev.as_metal_device()?.dispatch_census().clone();
    let mut out = Run {
        logits: Vec::new(),
        index_selects: Vec::new(),
        attention: Vec::new(),
    };
    // The second round reuses pages after clear; nonidentity pages must not
    // change the logical sequence or the causal prefix offset.
    // Grouped decode chunks require physically consecutive pages; the existing
    // allocator does not promise that after recycling. Exercise recycling
    // only at one page per decode chunk, where arbitrary page order is valid.
    let rounds = if cfg.flash_pages_per_chunk == 1 { 2 } else { 1 };
    for _ in 0..rounds {
        let mut pos = 0;
        for len in [33, 257, 19, 2, 1] {
            let ids: Vec<u32> = (pos..pos + len)
                .map(|i| (3 + i * 7 % (cfg.vocab_size - 3)) as u32)
                .collect();
            let input = Tensor::new(ids.as_slice(), dev)?.unsqueeze(0)?;
            dev.synchronize()?;
            let window = census.open().expect("one census window");
            let y = model.forward(&input, pos, &mut cache)?;
            dev.synchronize()?;
            let labels = window.close();
            out.logits
                .push(y.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?);
            if pos > 0 && len > 1 {
                out.index_selects.push(
                    labels
                        .iter()
                        .filter(|(label, _)| is_index_select_pipeline(label, dtype))
                        .map(|(_, count)| *count)
                        .sum(),
                );
                out.attention.push(
                    labels
                        .iter()
                        .filter(|(label, _)| label.contains("steel_attention"))
                        .map(|(_, count)| *count)
                        .sum(),
                );
            }
            pos += len;
        }
        cache.clear();
        dev.synchronize()?;
    }
    Ok(out)
}

fn check(enabled: bool, page_size: usize, pages_per_chunk: usize, churn: usize) -> Result<()> {
    check_shape(enabled, page_size, pages_per_chunk, churn, 64)
}

fn check_shape(
    enabled: bool,
    page_size: usize,
    pages_per_chunk: usize,
    churn: usize,
    head_dim: usize,
) -> Result<()> {
    check_shape_layout(
        enabled,
        page_size,
        pages_per_chunk,
        churn,
        head_dim,
        QkvLayout::Contiguous,
    )
}

fn check_shape_layout(
    enabled: bool,
    page_size: usize,
    pages_per_chunk: usize,
    churn: usize,
    head_dim: usize,
    qkv_layout: QkvLayout,
) -> Result<()> {
    let mut tuning = lloom::tuning::Tuning::default();
    tuning.from_env().expect("tuning environment");
    tuning.lfm2.paged_prefill = enabled;
    tuning.lfm2.pool_churn = churn;
    assert!(lloom::tuning::install(tuning).is_ok());
    // Each nextest case is a process; enable the production census before
    // constructing any Metal device, as its switch is read once.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let mut tiny = Lfm2Tiny::template();
    tiny.hidden_size = tiny.num_attention_heads * head_dim;
    let Some(fixture) = Fixture::<Lfm2Tiny>::new("lfm2_paged_prefill", tiny)? else {
        return Ok(());
    };
    let cfg = fixture.config(|cfg| {
        cfg.attn_impl = AttnImpl::FlashDecoding;
        cfg.kv_append = KvAppend::Paged;
        cfg.flash_page_size = page_size;
        cfg.flash_pages_per_chunk = pages_per_chunk;
        cfg.qkv_layout = qkv_layout;
    });
    let layers = cfg
        .layer_types
        .iter()
        .filter(|k| **k == LayerType::FullAttention)
        .count() as u64;
    assert!(layers > 0, "positive control: fixture has attention");
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        if dtype == DType::BF16 {
            use lloom_tensor::kernels::metal::arms::{Feature, Features};
            // Correctness coverage explicitly includes emulation; shipping's
            // refusal policy stays covered by emulated_bf16_load.rs unchanged.
            dev.as_metal_device()?
                .allow_emulated(Features::empty().with(Feature::Bf16));
            eprintln!(
                "PAGED_PREFILL BF16 correctness population: explicit device-local emulation opt-in"
            );
        }
        // A deliberately executed gather proves that the name filter observes
        // a real dispatch. Zero counts alone cannot establish route reachability.
        let source = Tensor::new(&[0f32, 1., 2., 3., 4., 5., 6., 7.], &dev)?
            .to_dtype(dtype)?
            .reshape((1, 1, 4, 2))?;
        let ids = Tensor::new(&[2u32, 0], &dev)?;
        dev.synchronize()?;
        let census = dev.as_metal_device()?.dispatch_census().clone();
        let window = census.open().expect("positive gather control");
        let selected = source.index_select(&ids, 2)?;
        dev.synchronize()?;
        let labels = window.close();
        let observed: u64 = labels
            .iter()
            .filter(|(label, _)| is_index_select_pipeline(label, dtype))
            .map(|(_, count)| *count)
            .sum();
        assert_eq!(observed, 1, "index-select positive control: {labels:?}");
        assert_eq!(
            selected
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
            [4., 5., 0., 1.]
        );
        let mut reference = cfg.clone();
        reference.kv_append = KvAppend::InPlace;
        let want = run::<CrownExact>(&fixture, &reference, dtype, &dev)?;
        let got = run::<PoolExact>(&fixture, &cfg, dtype, &dev)?;
        let repeated = run::<PoolExact>(&fixture, &cfg, dtype, &dev)?;
        let rounds = if pages_per_chunk == 1 { 2 } else { 1 };
        assert_eq!(got.logits.len(), 5 * rounds);
        for (pass, ((got, repeated), want)) in got
            .logits
            .iter()
            .zip(&repeated.logits)
            .zip(&want.logits)
            .enumerate()
        {
            assert!(got.iter().all(|x| x.is_finite()));
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(got),
                bits(want),
                "{dtype:?} page={page_size} k={pages_per_chunk} pass={pass}: in-place reference"
            );
            assert_eq!(
                bits(got),
                bits(repeated),
                "{dtype:?} pass={pass}: repeat determinism"
            );
        }
        use lloom_tensor::kernels::{sdpa_full_tile, SdpaDType};
        let sdpa_dtype = match dtype {
            DType::F16 => SdpaDType::F16,
            DType::BF16 => SdpaDType::BF16,
            DType::F32 => SdpaDType::F32,
            _ => unreachable!(),
        };
        let (_, bk, _, _) =
            sdpa_full_tile(cfg.head_dim(), sdpa_dtype).map_err(lloom_tensor::Error::wrap)?;
        let direct = enabled && page_size.is_multiple_of(bk);
        assert_eq!(
            got.index_selects,
            want.index_selects
                .iter()
                .map(|count| count + if direct { 0 } else { 2 * layers })
                .collect::<Vec<_>>()
        );
        assert_eq!(got.index_selects.len(), 3 * rounds);
        assert_eq!(
            got.index_selects, repeated.index_selects,
            "repeat dispatch census"
        );
        assert!(
            got.attention.iter().all(|&n| n == layers),
            "positive control: attention dispatched: {:?}",
            got.attention
        );
        eprintln!("PAGED_PREFILL enabled={enabled} dtype={dtype:?} page={page_size} k={pages_per_chunk} churn={churn} index_selects={:?} attention={:?} logits=bit-identical repeat=bit-identical", got.index_selects, got.attention);
    }
    Ok(())
}

#[test]
fn direct_pages_match_contiguous_with_recycled_pages_and_tail_tiles() -> Result<()> {
    check(true, 32, 1, 3)
}

#[test]
fn full_page_table_is_independent_of_decode_chunk_grouping() -> Result<()> {
    check(true, 32, 2, 0)
}

#[test]
fn shipped_page_geometry_reaches_direct_prefill() -> Result<()> {
    check(true, 256, 1, 3)
}

#[test]
fn incompatible_page_geometry_keeps_the_gathered_path() -> Result<()> {
    check(true, 16, 1, 3)
}

#[test]
fn disabled_arm_keeps_the_gathered_path() -> Result<()> {
    check(false, 32, 1, 3)
}

#[test]
fn the_128_head_dimension_uses_its_own_key_block_alignment() -> Result<()> {
    check_shape(true, 16, 1, 3, 128)
}

#[test]
fn deterministic_contract_preserves_paged_continuation_logits() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    lloom_tensor::tuning::pin_contract(lloom_tensor::tuning::Contract::Deterministic)
        .expect("isolated deterministic test process");
    check(true, 32, 1, 3)
}

#[test]
fn fused_qkv_layout_reaches_direct_prefill_without_extra_gathers() -> Result<()> {
    check_shape_layout(true, 32, 1, 3, 64, QkvLayout::Fused)
}

#[derive(Debug, PartialEq)]
struct VerifyRun {
    accepted: Vec<u32>,
    continuation_bits: Vec<u32>,
    index_selects: u64,
    attention: u64,
}

fn verify_run<P: Shape>(
    fixture: &Fixture<Lfm2Tiny>,
    cfg: &Config,
    dev: &Device,
) -> Result<VerifyRun> {
    let dtype = DType::F16;
    let model = fixture.model_in::<P>(cfg, Some(dtype), dev)?;
    let mut state = State::<P>::new(true, dtype, cfg, dev)?;
    // Four verify positions cross the 32-position page boundary.
    let prompt: Vec<u32> = (0..31)
        .map(|i| (3 + i * 7 % (cfg.vocab_size - 3)) as u32)
        .collect();
    let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
    model.forward(&input, 0, &mut state)?;
    dev.synchronize()?;
    let census = dev.as_metal_device()?.dispatch_census().clone();
    let window = census.open().expect("explicit verify census");
    let accepted = model.verify_step(3, &[5, 6, 7], prompt.len(), &mut state, &[])?;
    dev.synchronize()?;
    let labels = window.close();
    assert!(!accepted.is_empty() && accepted.len() <= 4);
    let next = Tensor::new(&[*accepted.last().unwrap()], dev)?.unsqueeze(0)?;
    let logits = model.forward(&next, prompt.len() + accepted.len(), &mut state)?;
    let values = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert!(values.iter().all(|x| x.is_finite()));
    Ok(VerifyRun {
        accepted,
        continuation_bits: values.into_iter().map(f32::to_bits).collect(),
        index_selects: labels
            .iter()
            .filter(|(label, _)| is_index_select_pipeline(label, dtype))
            .map(|(_, n)| *n)
            .sum(),
        attention: labels
            .iter()
            .filter(|(label, _)| label.contains("steel_attention"))
            .map(|(_, n)| *n)
            .sum(),
    })
}

#[test]
fn explicit_paged_verify_matches_in_place_and_preserves_next_step() -> Result<()> {
    use lloom::models::lfm2::ConvState;
    use lloom::profile::{PoolSpecExact, RewindableExact};
    let mut tuning = lloom::tuning::Tuning::default();
    tuning.from_env().expect("tuning environment");
    tuning.lfm2.paged_prefill = true;
    assert!(lloom::tuning::install(tuning).is_ok());
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let mut tiny = Lfm2Tiny::template();
    tiny.hidden_size = tiny.num_attention_heads * 64;
    let Some(fixture) = Fixture::<Lfm2Tiny>::new("lfm2_paged_prefill_verify", tiny)? else {
        return Ok(());
    };
    for layout in [QkvLayout::Contiguous, QkvLayout::Fused] {
        let cfg = fixture.config(|cfg| {
            cfg.attn_impl = AttnImpl::FlashDecoding;
            cfg.kv_append = KvAppend::Paged;
            cfg.flash_page_size = 32;
            cfg.flash_pages_per_chunk = 1;
            cfg.conv_state = ConvState::RotatingRing { k: 8 };
            cfg.qkv_layout = layout;
        });
        let layers = cfg
            .layer_types
            .iter()
            .filter(|k| **k == LayerType::FullAttention)
            .count() as u64;
        assert!(layers > 0);
        let mut reference = cfg.clone();
        reference.kv_append = KvAppend::InPlace;
        let want = verify_run::<RewindableExact>(&fixture, &reference, &dev)?;
        let got = verify_run::<PoolSpecExact>(&fixture, &cfg, &dev)?;
        let repeated = verify_run::<PoolSpecExact>(&fixture, &cfg, &dev)?;
        assert_eq!(got.attention, layers, "verify must execute tiled attention");
        assert_eq!(
            got, want,
            "{layout:?}: explicit verify plus subsequent state"
        );
        assert_eq!(got, repeated, "{layout:?}: repeat verify/state identity");
        eprintln!("PAGED_VERIFY layout={layout:?} accepted={:?} index_selects={} attention={} continuation=bit-identical", got.accepted, got.index_selects, got.attention);
    }
    Ok(())
}
