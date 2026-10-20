//! Canonical body admission and its real public request/readback consumers.
use lloom::generation::{
    Blocking, EosRead, LogitsProcessor, Processors, ReadbackStrategy, Ring, Sampling,
    ThinkingBudget, GREEDY_BELOW,
};
use lloom::nn::{refuse, Asks, Body};
use lloom_engine::{DecodeParams, EngineOptions, LogprobRequest, ReadbackArm, RingArm};
use lloom_tensor::{Device, Result};

#[test]
fn method_temperature_filter_matrix_reaches_one_body_table() -> Result<()> {
    let device = Device::Cpu;
    let ring = Ring::new(&[7], &device, 8, EosRead::Blocking)?;
    let blocking = Blocking::new(&[7], &device)?;
    let mut rows = 0;
    for gumbel in [false, true] {
        for t in [
            None,
            Some(0.0),
            Some(GREEDY_BELOW / 2.0),
            Some(GREEDY_BELOW),
            Some(0.7),
        ] {
            for k in [None, Some(0), Some(1), Some(8)] {
                for p in [None, Some(0.0), Some(1.0), Some(0.6)] {
                    // V is not known at admission: even k=V keeps its resolved
                    // filter enum until the existing row filter sees V.
                    let hot = t.is_some_and(|t| t >= GREEDY_BELOW);
                    let active_k = k.is_some_and(|k| k > 0);
                    let active_p = p.is_some_and(|p| p < 1.0);
                    let temperature = t.unwrap_or(0.0);
                    let expected = match (hot, gumbel, active_k, active_p) {
                        (false, _, _, _) => Sampling::ArgMax,
                        (true, true, false, false) => Sampling::GumbelSoftmax { temperature },
                        (true, true, _, _) => Sampling::FilteredGumbel {
                            temperature,
                            k: k.unwrap_or(0),
                            p: p.unwrap_or(1.0),
                        },
                        (true, false, true, true) => Sampling::TopKThenTopP {
                            temperature,
                            k: k.unwrap(),
                            p: p.unwrap(),
                        },
                        (true, false, true, false) => Sampling::TopK {
                            temperature,
                            k: k.unwrap(),
                        },
                        (true, false, false, true) => Sampling::TopP {
                            temperature,
                            p: p.unwrap(),
                        },
                        (true, false, false, false) => Sampling::All { temperature },
                    };
                    let params = DecodeParams {
                        temperature: t,
                        top_k: k,
                        top_p: p,
                        gumbel,
                        ..Default::default()
                    };
                    assert_eq!(Sampling::resolve(t, k, p, gumbel)?, expected);
                    assert_eq!(params.sampling()?, expected);
                    assert_eq!(params.sampler()?.sampling(), &expected);
                    assert_eq!(
                        LogitsProcessor::from_sampling(7, expected.clone()).sampling(),
                        &expected
                    );
                    let mut options = EngineOptions::default();
                    options.sampling.temperature = t;
                    options.sampling.top_k = k;
                    options.sampling.top_p = p;
                    options.sampling.gumbel = gumbel;
                    options.check().unwrap();
                    assert_eq!(options.sampling.sampling().unwrap(), expected);
                    let host = hot && (active_k || active_p);
                    let asks = Asks::sampling(&expected);
                    assert_eq!(refuse(Body::Step { ring: true }, &asks).is_err(), host);
                    assert_eq!(ring.check_sampler(&expected).is_err(), host);
                    assert!(blocking.check_sampler(&expected).is_ok());
                    assert_eq!(
                        ReadbackArm::Auto.resolve(&expected, RingArm::CROWN),
                        if host {
                            ReadbackArm::Blocking
                        } else {
                            ReadbackArm::CROWN
                        }
                    );
                    // A ring named in the options is accepted for every
                    // sampler and demoted, with its note, for a host-side
                    // one (C2).
                    options.decode.readback.kind = lloom_engine::ReadbackKind::Ring;
                    options.check().unwrap();
                    assert_eq!(
                        ReadbackArm::CROWN.resolve(&expected, RingArm::CROWN),
                        if host {
                            ReadbackArm::Blocking
                        } else {
                            ReadbackArm::CROWN
                        }
                    );
                    assert_eq!(ReadbackArm::CROWN.demotion(&expected).is_some(), host);
                    for body in [
                        Body::Step { ring: false },
                        Body::Window,
                        Body::BatchedStep,
                        Body::BatchedWindow,
                    ] {
                        let rejected = body == Body::BatchedWindow && hot;
                        assert_eq!(
                            refuse(body, &asks).is_err(),
                            rejected,
                            "{body:?} {expected:?}"
                        );
                        assert_eq!(params.refuse_on(body, "matrix").is_err(), rejected);
                    }
                    rows += 1;
                }
            }
        }
    }
    assert_eq!(rows, 160);
    Ok(())
}

#[test]
fn every_feature_combination_matches_canonical_diagnostics() -> Result<()> {
    let arms = [
        Sampling::ArgMax,
        Sampling::All { temperature: 0.7 },
        Sampling::GumbelSoftmax { temperature: 0.7 },
        Sampling::TopK {
            temperature: 0.7,
            k: 1,
        },
        Sampling::TopP {
            temperature: 0.7,
            p: 0.6,
        },
        Sampling::TopKThenTopP {
            temperature: 0.7,
            k: 2,
            p: 0.6,
        },
        Sampling::FilteredGumbel {
            temperature: 0.7,
            k: 2,
            p: 0.6,
        },
    ];
    let mut checked = 0;
    for arm in arms {
        for bits in 0..16 {
            let processors = bits & 1 != 0;
            let budget = bits & 2 != 0;
            let digest = bits & 4 != 0;
            let logprobs = (bits & 8 != 0).then_some(0);
            let (temperature, top_k, top_p, gumbel) = match arm {
                Sampling::ArgMax => (None, None, None, false),
                Sampling::All { temperature } => (Some(temperature), None, None, false),
                Sampling::GumbelSoftmax { temperature } => (Some(temperature), None, None, true),
                Sampling::TopK { temperature, k } => (Some(temperature), Some(k), None, false),
                Sampling::TopP { temperature, p } => (Some(temperature), None, Some(p), false),
                Sampling::TopKThenTopP { temperature, k, p } => {
                    (Some(temperature), Some(k), Some(p), false)
                }
                Sampling::FilteredGumbel { temperature, k, p } => {
                    (Some(temperature), Some(k), Some(p), true)
                }
            };
            let params = DecodeParams {
                temperature,
                top_k,
                top_p,
                gumbel,
                processors: Processors {
                    presence_penalty: if processors { 0.5 } else { 0.0 },
                    ..Default::default()
                },
                thinking: if budget {
                    Some(ThinkingBudget::new(2, vec![7], vec![7])?)
                } else {
                    None
                },
                logits_digest: digest,
                logprobs: logprobs.map(|top| LogprobRequest { top, rows: false }),
                ..Default::default()
            };
            for body in [
                Body::Step { ring: true },
                Body::Step { ring: false },
                Body::Window,
                Body::BatchedStep,
                Body::BatchedWindow,
            ] {
                let asks = Asks {
                    sampling: &arm,
                    processors,
                    budget,
                    digest,
                    logprobs,
                };
                let rejected = match body {
                    Body::Step { ring: true } => {
                        arm.requires_host() || digest || logprobs.is_some()
                    }
                    Body::Step { ring: false } => false,
                    Body::Window => logprobs.is_some(),
                    Body::BatchedStep => processors || budget || logprobs.is_some(),
                    Body::BatchedWindow => {
                        processors || budget || logprobs.is_some() || arm != Sampling::ArgMax
                    }
                };
                let canonical = refuse(body, &asks);
                assert_eq!(canonical.is_err(), rejected, "{body:?} {arm:?} bits={bits}");
                let wrapper = params.refuse_on(body, "consumer");
                match canonical {
                    Ok(()) => wrapper?,
                    Err(e) => {
                        assert_eq!(wrapper.unwrap_err().to_string(), format!("consumer: {e}"))
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 560);
    Ok(())
}

#[test]
fn actual_config_merge_preserves_numeric_string_and_readback_shorthand_grammar() {
    use lloom_engine::option_tree::index;
    use serde_json::json;
    // (config, accepted, the readback kind it names, gumbel). A ring named
    // with a host-side sampler is accepted: the request runs on the blocking
    // arm with a note (coupling C2 resolves).
    let ring = Some(lloom_engine::ReadbackKind::Ring);
    let cases = [
        (
            json!({"generation": {"temperature": 0.7, "top_k": 2, "top_p": 0.6, "sampling": {"method": "gumbel"}}}),
            true,
            None,
            true,
        ),
        (
            json!({"generation": {"temperature": " 0.7 ", "top_k": "+2", "top_p": "0.6", "sampling": {"method": "gumbel"}}}),
            true,
            None,
            true,
        ),
        (
            json!({"decode": {"readback": "blocking"}, "generation": {"temperature": 0.7, "top_k": 2, "sampling": {"method": "gumbel"}}}),
            true,
            Some(lloom_engine::ReadbackKind::Blocking),
            true,
        ),
        (
            json!({"decode": {"readback": "ring"}, "generation": {"temperature": 0.7, "top_k": 2, "sampling": {"method": "gumbel"}}}),
            true,
            ring,
            true,
        ),
        (
            json!({"decode": {"readback": {"kind": "ring"}}, "generation": {"temperature": "0.7", "top_k": "2"}}),
            true,
            ring,
            false,
        ),
        (json!({"generation": {"temperature": -1}}), false, None, false),
        (json!({"generation": {"top_p": "1.1"}}), false, None, false),
        (json!({"generation": {"top_k": -1}}), false, None, false),
    ];
    for (value, allowed, kind, gumbel) in cases {
        let mut options = EngineOptions::default();
        let result = index()
            .merge(&mut options, &value)
            .and_then(|()| options.check());
        assert_eq!(result.is_ok(), allowed, "{value}: {result:?}");
        if allowed {
            let sampling = options.sampling.sampling().unwrap();
            assert!(sampling.requires_host(), "{value}: {sampling:?}");
            assert_eq!(
                matches!(sampling, Sampling::FilteredGumbel { .. }),
                gumbel,
                "{value}: {sampling:?}"
            );
            if let Some(kind) = kind {
                assert_eq!(options.decode.readback.kind, kind, "{value}");
            }
        }
    }
}

#[test]
fn actual_config_pipeline_checks_the_root_validator_corpus() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/sampling_ingress.json")).unwrap();
    let mut seen = 0;
    for case in cases.as_array().unwrap() {
        let mut options = EngineOptions::default();
        let result = lloom_engine::option_tree::index()
            .merge(&mut options, &case["config"])
            .and_then(|()| options.check());
        assert_eq!(
            result.is_ok(),
            case["runtime_accepts"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
        seen += 1;
    }
    assert_eq!(seen, 511);
    // This calls the production generator, not a hand-built schema or a
    // partial validator. Root separately feeds its output to a real validator.
    let schema = lloom_engine::option_schema::schema();
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
}

#[test]
fn old_host_sort_diagnostic_and_global_logprob_bound_stay_exact() {
    for sampling in [
        Sampling::TopK {
            temperature: 0.7,
            k: 2,
        },
        Sampling::TopP {
            temperature: 0.7,
            p: 0.6,
        },
        Sampling::TopKThenTopP {
            temperature: 0.7,
            k: 2,
            p: 0.6,
        },
    ] {
        let error = refuse(Body::Step { ring: true }, &Asks::sampling(&sampling)).unwrap_err();
        assert_eq!(error.to_string(), format!("the ring needs a device-resident sampler (argmax, multinomial or gumbel-max); {sampling:?} takes a host sort and cannot ride it (DESIGN.md §11.2c)"));
    }
    let arm = Sampling::FilteredGumbel {
        temperature: 0.7,
        k: 2,
        p: 0.6,
    };
    for body in [
        Body::Step { ring: true },
        Body::Step { ring: false },
        Body::Window,
        Body::BatchedStep,
        Body::BatchedWindow,
    ] {
        let asks = Asks {
            logprobs: Some(65),
            ..Asks::sampling(&arm)
        };
        assert_eq!(
            refuse(body, &asks).unwrap_err().to_string(),
            "per-token logprobs name at most 64 candidates beside the drawn id, not 65"
        );
    }
}
