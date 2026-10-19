//! The engine's sampling resolution: one fallible entry point, and the
//! config and request layers refuse a bad field by name.
use lloom::generation::{Sampling, GREEDY_BELOW};
use lloom_engine::{DecodeParams, ReadbackArm, RingArm, SamplingConfig};

#[test]
fn public_engine_resolver_keeps_method_filters_and_defaults() {
    let s = DecodeParams::sampling_of(Some(0.7), Some(0.9), Some(20), true).unwrap();
    assert_eq!(
        s,
        Sampling::FilteredGumbel {
            temperature: 0.7,
            k: 20,
            p: 0.9
        }
    );
    assert_eq!(
        ReadbackArm::Auto.resolve(&s, RingArm::CROWN),
        ReadbackArm::Blocking
    );
    assert_eq!(DecodeParams::default().sampling().unwrap(), Sampling::ArgMax);
    assert_eq!(SamplingConfig::default().sampling().unwrap(), Sampling::ArgMax);
    assert_eq!(
        DecodeParams::sampling_of(Some(0.7), Some(1.0), Some(0), false).unwrap(),
        Sampling::All { temperature: 0.7 }
    );
    assert_eq!(
        DecodeParams::sampling_of(Some(GREEDY_BELOW / 2.0), Some(0.9), Some(20), true).unwrap(),
        Sampling::ArgMax
    );
    assert_eq!(
        DecodeParams::sampling_of(Some(GREEDY_BELOW), None, None, false).unwrap(),
        Sampling::All {
            temperature: GREEDY_BELOW
        }
    );
}

#[test]
fn public_config_and_request_validation_refuse_invalid_fields_before_a_sampler() {
    for temperature in [Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
        let params = DecodeParams {
            temperature,
            ..DecodeParams::default()
        };
        assert!(params.sampler().is_err());
        let config = SamplingConfig {
            temperature,
            ..SamplingConfig::default()
        };
        let refused = config.check().unwrap_err();
        assert!(refused.starts_with("sampling: generation.temperature = "), "{refused}");
        assert!(!refused.contains('\n'), "a refusal carries no backtrace: {refused}");
    }
    for p in [-0.1, 1.01, f64::NAN] {
        let params = DecodeParams {
            top_p: Some(p),
            ..DecodeParams::default()
        };
        assert!(params.sampler().is_err()); // greedy validates unused p too
        let config = SamplingConfig {
            top_p: Some(p),
            ..SamplingConfig::default()
        };
        let refused = config.check().unwrap_err();
        assert!(refused.starts_with("sampling: generation.top_p = "), "{refused}");
        assert!(refused.contains("outside [0, 1]"), "{refused}");
        let mut options = lloom_engine::EngineOptions::default();
        options.sampling.top_p = Some(p);
        assert_eq!(options.check().unwrap_err(), refused);
    }
    let filtered = DecodeParams {
        temperature: Some(0.7),
        top_k: Some(20),
        gumbel: true,
        ..DecodeParams::default()
    };
    assert!(filtered
        .refuse_on(lloom::nn::Body::Step { ring: true }, "test")
        .is_err());
    assert!(filtered
        .refuse_on(lloom::nn::Body::Step { ring: false }, "test")
        .is_ok());
    assert!(filtered.refuse_on(lloom::nn::Body::Window, "test").is_ok());
}
