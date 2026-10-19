//! Model-free compiled-recorder controls. Creates no Metal device or native resource.
//! Synthetic intervals validate recorder plumbing, not GPU timing or kernel execution.
use lloom_tensor::kernels::metal::{encoder::HazardKinds, profile, trace};
use lloom_tensor::tuning::{self, Tuning, PROFILE_INSTRUMENTS};
use std::sync::Arc;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "exercise".into());
    if mode == "status" {
        let requested = &tuning::tuning().instrument;
        let trace_active = trace::trace_requested();
        assert_eq!(trace_active, requested.trace && PROFILE_INSTRUMENTS);
        assert_eq!(trace::trace_requested(), trace_active);
        assert_eq!(profile::enabled(), requested.profile);
        assert_eq!(
            profile::recorder_enabled(),
            requested.profile && PROFILE_INSTRUMENTS
        );
        let counters_active = profile::counters_enabled();
        assert_eq!(
            counters_active,
            requested.profile && requested.profile_counters && PROFILE_INSTRUMENTS
        );
        assert_eq!(profile::counters_enabled(), counters_active);
        println!("status: trace_requested={} trace_active={} cheap_profile={} recorder_active={} counters_requested={} counters_active={} profile_compiled={}", requested.trace,trace_active,profile::enabled(),profile::recorder_enabled(),requested.profile_counters,counters_active,PROFILE_INSTRUMENTS);
        return Ok(());
    }
    if mode == "setup" || mode == "legacy-setup" {
        if mode == "legacy-setup" {
            let _ = tuning::tuning();
        }
        match tuning::try_tuning() {
            Ok(_) => println!("setup=accepted profile_compiled={PROFILE_INSTRUMENTS}"),
            Err(error) => {
                if mode == "setup" {
                    assert!(!tuning::installed(), "refusal must not freeze env tuning");
                }
                println!("setup=refused reason={error}");
                return Err(error.into());
            }
        }
        return Ok(());
    }
    let expected = std::env::args().nth(2).expect("exercise available|absent");
    assert!(
        matches!(expected.as_str(), "available" | "absent"),
        "unknown arm"
    );
    assert_eq!(
        PROFILE_INSTRUMENTS,
        expected == "available",
        "wrong build arm"
    );
    if !PROFILE_INSTRUMENTS {
        let mut rejected = Tuning::default();
        rejected.instrument.trace = true;
        let error = tuning::install(rejected.clone()).unwrap_err();
        assert!(matches!(
            &error.reason,
            tuning::TuningSetupError::NotCompiled {
                key: "diagnostics.trace.dispatch",
                capability: "profile"
            }
        ));
        assert_eq!(error.into_tuning(), rejected);
        assert!(!tuning::installed());
    }
    assert_eq!(
        lloom_tensor::metal_backend::set_served_request_trace(true).is_ok(),
        PROFILE_INSTRUMENTS
    );
    lloom_tensor::metal_backend::set_served_request_trace(false)?;
    assert_eq!(
        lloom_tensor::metal_backend::set_readback_trace(true).is_ok(),
        PROFILE_INSTRUMENTS
    );
    lloom_tensor::metal_backend::set_readback_trace(false)?;
    let mut t = Tuning::default();
    t.instrument.profile = true; // usable scalar counters in both arms
    t.instrument.trace = PROFILE_INSTRUMENTS;
    tuning::install(t)?;
    assert!(profile::enabled());
    assert_eq!(profile::recorder_enabled(), PROFILE_INSTRUMENTS);
    assert!(!profile::counters_enabled());
    assert!(!profile::encoder_per_dispatch());
    profile::reset();
    let a: Arc<str> = Arc::from("synthetic-a");
    let b: Arc<str> = Arc::from("synthetic-b");
    profile::record_dispatch(Some(&a), 64);
    profile::record_dispatch(Some(&b), 32);
    profile::record_dispatch(Some(&a), 64);
    profile::record_encoder();
    profile::record_command_buffer(1.0, 3.0, 0.5, 1);
    profile::record_command_buffer(2.0, 4.0, 1.5, 2);
    profile::record_commit(Duration::from_nanos(13));
    profile::record_create_cb(Duration::from_nanos(7));
    profile::record_fence_stats(9, 4, 2);
    let p = profile::snapshot();
    assert_eq!(
        (
            p.dispatches,
            p.encoders,
            p.command_buffers,
            p.commits,
            p.commit_ns,
            p.create_cb_ns,
            p.fence_probes,
            p.fence_hits,
            p.fence_waits
        ),
        (3, 1, 2, 1, 13, 7, 9, 4, 2)
    );
    if PROFILE_INSTRUMENTS {
        assert_eq!(
            p.by_label,
            vec![("synthetic-a".into(), 2), ("synthetic-b".into(), 1)]
        );
        assert_eq!(
            (
                p.timed_command_buffers,
                p.gpu_busy_sum_s,
                p.gpu_busy_union_s
            ),
            (2, 4.0, 3.0)
        );
    } else {
        assert!(p.by_label.is_empty());
        assert!(p.command_buffer_records.is_empty());
        assert_eq!(p.timed_command_buffers, 0);
    }
    println!("profile: dispatches={} encoders={} command_buffers={} by_label={:?} timed={} sum={} union={}",p.dispatches,p.encoders,p.command_buffers,p.by_label,p.timed_command_buffers,p.gpu_busy_sum_s,p.gpu_busy_union_s);

    let before = trace::shape_counts();
    let suppressed = trace::barriers_suppressed();
    let ranges = trace::execute_range_count();
    let resources = trace::use_resource_count();
    trace::record_barrier(None, HazardKinds::NONE);
    trace::note_encoder_session();
    trace::note_command_buffer();
    trace::record_barrier_suppressed();
    trace::record_execute_range(5);
    let after = trace::shape_counts();
    assert_eq!(
        (
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            trace::barriers_suppressed() - suppressed,
            trace::execute_range_count() - ranges,
            trace::use_resource_count() - resources
        ),
        (1, 1, 1, 1, 1, 5)
    );
    println!("engagement: barriers=1 encoders=1 command_buffers=1 suppressed=1 execute_ranges=1 use_resource=5");
    trace::set_recording(false)?;
    assert_eq!(trace::set_recording(true).is_ok(), PROFILE_INSTRUMENTS);
    let device_trace = Arc::new(trace::DeviceTrace::default());
    let window = device_trace.open();
    if PROFILE_INSTRUMENTS {
        let window = window.map_err(|e| anyhow::anyhow!("trace window: {e:?}"))?;
        trace::record_pipeline(Some(&device_trace), "synthetic-trace");
        trace::record_binding(Some(&device_trace), 2, 0x1000, 16, 64, true);
        trace::record_dispatch(Some(&device_trace), (7, 1, 1), (4, 1, 1), false);
        let local = window.close();
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].pipeline, "synthetic-trace");
        assert_eq!(local[0].bindings[0].offset, 16);
        assert_eq!(local[0].bindings[0].len, 64);
        let global = trace::take_dispatches();
        assert_eq!(global.len(), 1);
        assert_eq!(global[0].signature(), local[0].signature());
        assert!(trace::take_dispatches().is_empty());
        trace::set_recording(false)?;
        println!("trace: local=1 global=1 binding_offset=16 binding_len=64 drain=empty");
    } else {
        assert_eq!(window.err(), Some(trace::TraceRefused::NotCompiled));
        assert_eq!(
            Arc::new(profile::DeviceCensus::default()).open().err(),
            Some(profile::CensusRefused::NotCompiled)
        );
        println!("trace/census: explicit_window=NotCompiled enable=Err disable=Ok");
    }
    // Supplied payload analysis remains available, even without recorder storage.
    let records = [profile::EncoderRecord {
        start: 1.0,
        end: 2.0,
        step: 0,
        label: Some(a),
        dispatches: 1,
        input_bytes: 64,
        cb_start: 0.0,
        cb_end: 3.0,
    }];
    assert_eq!(profile::encoder_containment(&records, 0.0), 0);
    assert_eq!(profile::encoder_union_s(&records), 1.0);
    assert_eq!(profile::encoder_time_by_label(&records)[0].input_bytes, 64);
    profile::reset();
    assert_eq!(profile::snapshot().dispatches, 0);
    assert!(profile::snapshot().by_label.is_empty());
    println!("payload_analysis=retained reset=empty control=host_only");
    Ok(())
}
