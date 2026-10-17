//! Concrete static-body controls; these do not certify B=1 handler forwarding
//! or record literal seed spans. The root owns compilation and execution.

use super::*;
use lloom::models::lfm2::{ConvState, LayerType};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::RewindableExact;
use lloom::testing::fixture::{Backend, Fixture};
use lloom::testing::tiny::backend::TinyBackend;
use lloom::testing::tiny::Weights;

fn pair() -> lloom_tensor::Result<(
    Fixture<Lfm2Tiny>,
    Model<RewindableExact>,
    Config,
    DSparkDrafter,
)> {
    assert_eq!(
        Backend::current(),
        Backend::Tiny,
        "this bounded CPU control needs a separately prepared pair for scale"
    );
    let dcfg = DSparkConfig {
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 48,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_is_neox_style: false,
        block_size: 3,
        mask_token_id: 47,
        target_layer_ids: vec![0, 1, 2, 3, 4],
        num_target_layers: 6,
        markov_rank: 0,
        enable_confidence_head: false,
        logits_start: 0,
    };
    let cfg = Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: 4 },
        conv_prefill_ctx: true,
        tap_layers: dcfg.tap_layers(),
        tap_dtype: DType::F32,
        ..Default::default()
    };
    let weights = Weights::default();
    let fixture = Fixture::<Lfm2Tiny>::new("engine-static-seed", cfg.clone())?
        .expect("tiny fixture is available")
        .with_weights(weights);
    let model = fixture.model_in::<RewindableExact>(&cfg, Some(DType::F32), &Device::Cpu)?;
    let (backend, _) = TinyBackend::new(weights.law, weights.seed);
    let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
    let drafter = DSparkDrafter::new(dcfg, 128, vb)?;
    Ok((fixture, model, cfg, drafter))
}

fn run_static(
    opts: &DecodeOptions,
    specs: &[RowSpec],
) -> lloom_tensor::Result<(Vec<DecodeOut>, Vec<(usize, Vec<u32>)>)> {
    let (_fixture, model, cfg, drafter) = pair()?;
    let device = Device::Cpu;
    let mut extra = Speculators {
        spec: Some(DSparkSpeculator::new_with_ctx(
            drafter,
            model.draft_embedding().clone(),
            model.lm_head_linear().clone(),
            specs.len().max(1),
            CtxKv::Contiguous,
        )?),
        continuous: false,
        eos_ids: vec![u32::MAX],
    };
    let cache = || State::<RewindableExact>::new_with(true, DType::F32, &cfg, &device, 128);
    let mut sessions = Vec::new();
    for _ in specs {
        sessions.push(Session::new(&model, cache()?));
    }
    let mut pool = SlotPool::new(1, || cache().unwrap());
    let mut rings = RingPool::new();
    let mut driver: crate::Driver<'_, Model<RewindableExact>> = Box::new(
        lloom::nn::BatchDecoder::new(opts.clone(), &[u32::MAX], device.clone())?,
    );
    let mut idle = Idle {
        model: &model,
        device: &device,
        pool: &mut pool,
        rings: &mut rings,
        opts,
        extra: &mut extra,
        driver: &mut driver,
    };
    let prompt: Vec<u32> = (0..37).map(|i| (i % 47) as u32).collect();
    let prompts: Vec<&[u32]> = specs.iter().map(|_| prompt.as_slice()).collect();
    let mut tokens = Vec::new();
    let outs = static_dspark_body(&mut idle, &mut sessions, &prompts, specs, &mut |i, ids| {
        tokens.push((i, ids.to_vec()));
        Ok(())
    })?;
    Ok((outs, tokens))
}

fn row(max_tokens: usize) -> RowSpec {
    RowSpec {
        sampling: Sampling::ArgMax,
        seed: 42,
        eos_ids: vec![u32::MAX],
        max_tokens,
    }
}

#[test]
fn static_scheduler_resolves_group_and_legacy_without_losing_fields() {
    let mut decode = crate::DecodeConfig::default();
    decode.prefill.tile = 11;
    decode.prefill.chunk = 5;
    decode.prefill.ubatch = 3;
    decode.prefill.spark = Some("progressive:1,16/tile:16".into());
    decode.prefill.in_flight = 7;
    decode.prefill.pass_widths = lloom::nn::PassWidths::Ladder;
    decode.in_flight = 6;
    decode.phase_sync = true;
    for (tile, chunk, want_tile, want_chunk) in [(0, 0, 11, 5), (16, 9, 16, 9)] {
        let actual = static_dspark_scheduler_options(&decode, tile, chunk);
        let mut expected = decode.options(0, ReadbackArm::Blocking, tile);
        expected.lookahead = 1;
        expected.prefill_chunk = want_chunk;
        assert_eq!(actual.prefill_tile, want_tile);
        // Debug covers every DecodeOptions field, including future additions.
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(actual.prefill_ubatch, 3);
        assert_eq!(actual.prefill_schedule, Some(vec![1, 16]));
        assert_eq!(actual.prefill_bulk, 16);
        assert_eq!(actual.prefill_pacer_depth, 7);
        assert_eq!(actual.in_flight_depth, 6);
        assert!(actual.sample_split);
        assert_eq!(actual.max_tokens, 0);
    }
}

#[test]
fn actual_static_dspark_body_consumes_scheduler_seed_options_and_row_cap() {
    // These pin forward COUNTS in the concrete body, not literal spans or B1.
    for (group, legacy, spark, units) in [
        (16, 0, 0, 3),
        (11, 16, 0, 3),
        (11, 0, 0, 4),
        (16, 0, 3, 4),
        (0, 0, 3, 2),
        (0, 0, 0, 1),
    ] {
        let mut decode = crate::DecodeConfig::default();
        decode.prefill.tile = group;
        decode.prefill.ubatch = spark;
        decode.prefill.in_flight = 5;
        let opts = static_dspark_scheduler_options(&decode, legacy, 0);
        let before = format!("{opts:?}");
        let (outs, streamed) = run_static(&opts, &[row(1)]).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].stats.prefill_units, units);
        assert_eq!(outs[0].ids.len(), 1);
        assert_eq!(outs[0].stats.produced, 1);
        assert!(!outs[0].eos);
        assert_eq!(streamed, vec![(0, outs[0].ids.clone())]);
        assert_eq!(
            format!("{opts:?}"),
            before,
            "scheduler options are borrowed"
        );
    }
    let mut decode = crate::DecodeConfig::default();
    decode.prefill.tile = 16;
    decode.prefill.spark = Some("progressive:1,16/tile:16".into());
    let opts = static_dspark_scheduler_options(&decode, 0, 0);
    let (outs, _) = run_static(&opts, &[row(1)]).unwrap();
    assert_eq!(outs[0].stats.prefill_units, 7);
    assert_eq!(outs[0].ids.len(), 1);
}

#[test]
fn actual_static_dspark_body_keeps_sampling_and_eos_refusals() {
    let opts = DecodeOptions::new(99);
    let mut sampled = row(1);
    sampled.sampling = Sampling::TopK {
        k: 2,
        temperature: 1.0,
    };
    let err = run_static(&opts, &[sampled]).unwrap_err().to_string();
    assert!(err.contains("argmax"), "{err}");
    let first = row(1);
    let mut second = row(1);
    second.eos_ids = vec![7];
    let err = run_static(&opts, &[first, second]).unwrap_err().to_string();
    assert!(
        err.contains("row 1") && err.contains("one EOS set"),
        "{err}"
    );
}
