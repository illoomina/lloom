use super::*;
use crate::{Device, Layout, Tensor};

fn device() -> Result<Device> {
    crate::isolation::claim("guarded sampled-token consumers");
    let mut tuning = crate::tuning::Tuning::default();
    tuning.from_env().expect("valid environment");
    tuning.contract.profile = "deterministic".into();
    crate::tuning::install(tuning).expect("before device");
    let d = Device::new_metal(0)?;
    {
        let encoder = d.as_metal_device()?.command_encoder()?;
        println!(
            "GUARDED_SAMPLE_SETUP metal4={} merged={}",
            encoder.as_ref().command_buffer.is_metal4(),
            crate::kernels::kernel::merged_translation_unit()
        );
    }
    Ok(d)
}

fn storage(t: &Tensor) -> MetalStorage {
    let (s, l) = t.storage_and_layout();
    assert_eq!(l.start_offset(), 0);
    let Storage::Metal(s) = &*s else {
        unreachable!()
    };
    s.clone()
}

fn sample(d: &Device, ids: &[u32], status: &[u32]) -> Result<PrivateMultinomial> {
    Ok(PrivateMultinomial {
        ids: storage(&Tensor::new(ids, d)?),
        status: storage(&Tensor::new(status, d)?),
        support: storage(&Tensor::zeros((ids.len(), 3), DType::U32, d)?),
        rows: ids.len(),
        vocab: 3,
    })
}

fn finish(d: &Device) -> Result<()> {
    let m = d.as_metal_device()?;
    m.commands().flush().map_err(MetalError::from)?;
    m.wait_until_completed()
}

#[test]
fn generated_noise_slot_refuses_busy_before_invoking_its_filler() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let scores = Tensor::new(&[0f32, 0., 0.], &d)?;
    let scores_storage = storage(&scores);
    let layout = Layout::contiguous(3);
    let mut prepared = super::super::prepared::PreparedSampler::with_noise(m, 1, 3)?;
    let guard = SampleGuard::new(m)?;
    let mut readback = super::super::readback::SampleReadback::new(m, 1)?;
    let token = prepared.draw_generated(
        &guard,
        &scores_storage,
        &layout,
        1.,
        0,
        1.,
        false,
        |out, rows, vocab| {
            assert_eq!((rows, vocab), (1, 3));
            out[0] = 0.5;
            Ok(())
        },
    )?;
    token.snapshot(&mut readback)?;
    drop(token);
    let called = Cell::new(false);
    assert!(prepared
        .draw_generated(
            &guard,
            &scores_storage,
            &layout,
            1.,
            0,
            1.,
            false,
            |_, _, _| {
                called.set(true);
                Ok(())
            }
        )
        .is_err());
    assert!(
        !called.get(),
        "a busy upload must not advance the caller's RNG"
    );
    finish(&d)?;
    assert_eq!(readback.poll()?.expect("completed").ids(), [1]);
    let token = prepared.draw_generated(
        &guard,
        &scores_storage,
        &layout,
        1.,
        0,
        1.,
        false,
        |out, _, _| {
            out[0] = 0.;
            Ok(())
        },
    )?;
    token.snapshot(&mut readback)?;
    drop(token);
    finish(&d)?;
    assert_eq!(readback.poll()?.expect("completed").ids(), [0]);
    Ok(())
}

#[test]
fn generated_noise_parameter_refusal_precedes_noise_or_rng_mutation() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let scores = storage(&Tensor::new(&[0f32, 0., 0.], &d)?);
    let layout = Layout::contiguous(3);
    let mut prepared = super::super::prepared::PreparedSampler::with_noise(m, 1, 3)?;
    let guard = SampleGuard::new(m)?;
    for (temperature, p) in [
        (f64::NAN, 1.),
        (0., 1.),
        (1., f64::NAN),
        (1., -0.1),
        (1., 1.1),
    ] {
        let called = Cell::new(false);
        assert!(prepared
            .draw_generated(
                &guard,
                &scores,
                &layout,
                temperature,
                0,
                p,
                false,
                |_, _, _| {
                    called.set(true);
                    Ok(())
                }
            )
            .is_err());
        assert!(
            !called.get(),
            "invalid request must not call the noise generator"
        );
    }
    Ok(())
}

fn consumer_graph(force_cut: bool) -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    m.set_compute_per_buffer(if force_cut { 1 } else { usize::MAX });
    for (ids, status, error) in [
        ([1, 1], [0, u32::MAX, 3, 2, 0, u32::MAX, 3, 2], [0; 4]),
        ([1, u32::MAX], [0, u32::MAX, 3, 2, 1, 2, 2, 0], [1, 1, 2, 0]),
        (
            [1, u32::MAX],
            [0, u32::MAX, 3, 2, 0, u32::MAX, 3, 2],
            [6, 1, u32::MAX, 0],
        ),
    ] {
        let mut guard = SampleGuard::new(m)?;
        let history = Tensor::new(&[77u32, 77, 77, 77], &d)?;
        let input = Tensor::new(&[88u32, 88], &d)?;
        let marker = Tensor::new(&[9u32], &d)?;
        let eos = Tensor::new(&[9u8], &d)?;
        let counts = Tensor::zeros(3, DType::F32, &d)?;
        let table = Tensor::new(&[1f32, 2., 3., 4., 5., 6.], &d)?;
        let emb = Tensor::new(&[-99f32; 6], &d)?;
        let before = crate::metal_backend::readback_counts();
        let token = guard.bind(sample(&d, &ids, &status)?)?;
        token.embedding_into(&storage(&table), 0, 2, &storage(&emb), 1)?;
        token.record_counts(&storage(&counts))?;
        token.copy_into(&storage(&history), 1)?;
        token.copy_into(&storage(&input), 0)?;
        token.publish_marker(
            &storage(&marker),
            10,
            &storage(&history),
            &storage(&input),
            &storage(&eos),
        )?;
        drop(token);
        assert_eq!(
            crate::metal_backend::readback_counts(),
            before,
            "no intermediate host read"
        );
        assert_eq!(guard.poll_error()?, None, "encoding has not committed");
        finish(&d)?;
        assert_eq!(guard.poll_error()?, Some(error));
        if error[0] == 0 {
            assert_eq!(history.to_vec1::<u32>()?, [77, 1, 1, 77]);
            assert_eq!(input.to_vec1::<u32>()?, [1, 1]);
            assert_eq!(marker.to_vec1::<u32>()?, [10]);
            assert_eq!(counts.to_vec1::<f32>()?, [0., 2., 0.]);
            assert_eq!(emb.to_vec1::<f32>()?, [-99., 3., 4., 3., 4., -99.]);
        } else {
            assert_eq!(history.to_vec1::<u32>()?, [77; 4]);
            assert_eq!(input.to_vec1::<u32>()?, [88; 2]);
            assert_eq!(marker.to_vec1::<u32>()?, [9]);
            assert_eq!(counts.to_vec1::<f32>()?, [0.; 3]);
            assert_eq!(emb.to_vec1::<f32>()?, [-99.; 6]);
        }
    }
    Ok(())
}

#[test]
fn invalid_later_row_guards_embedding_counts_history_and_publication_before_completion(
) -> Result<()> {
    consumer_graph(false)
}

#[test]
fn invalid_later_row_guards_the_same_consumers_across_forced_command_buffer_cuts() -> Result<()> {
    consumer_graph(true)
}

#[test]
fn sticky_error_suppresses_already_queued_valid_successor_and_forced_tail() -> Result<()> {
    let d = device()?;
    let mut guard = SampleGuard::new(d.as_metal_device()?)?;
    let state = Tensor::new(&[0u32, 0], &d)?;
    let forced = Tensor::new(&[2u32], &d)?;
    let close = Tensor::new(&[2u32], &d)?;
    let output = Tensor::new(&[77u32], &d)?;
    let history = Tensor::new(&[88u32], &d)?;
    let invalid = guard.bind(sample(&d, &[u32::MAX], &[3, u32::MAX, 0, 0])?)?;
    drop(invalid);
    let mut next = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
    // SAFETY: fixture owns every state/output alias, does not mutate them
    // through the final consumer, and waits before inspecting/reusing them.
    unsafe {
        next.feed_into(
            &storage(&forced),
            &storage(&close),
            &storage(&state),
            &storage(&output),
            0,
        )?;
    }
    next.copy_into(&storage(&history), 0)?;
    drop(next);
    finish(&d)?;
    assert_eq!(guard.poll_error()?, Some([3, 0, u32::MAX, 0]));
    assert_eq!(state.to_vec1::<u32>()?, [0, 0]);
    assert_eq!(output.to_vec1::<u32>()?, [77]);
    assert_eq!(history.to_vec1::<u32>()?, [88]);
    Ok(())
}

#[test]
fn fed_id_not_drawn_id_reaches_history_and_counts_and_invalid_forcing_is_terminal() -> Result<()> {
    let d = device()?;
    for forced_id in [2u32, 3] {
        let mut guard = SampleGuard::new(d.as_metal_device()?)?;
        let state = Tensor::new(&[0u32, 0], &d)?;
        let forced = Tensor::new(&[forced_id], &d)?;
        let close = Tensor::new(&[2u32], &d)?;
        let output = Tensor::new(&[77u32], &d)?;
        let counts = Tensor::zeros(3, DType::F32, &d)?;
        let history = Tensor::new(&[88u32], &d)?;
        let eos = Tensor::new(&[9u8], &d)?;
        let mut token = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
        // SAFETY: fixture owns every state/output alias, does not mutate them
        // through the final consumer, and waits before inspecting/reusing them.
        unsafe {
            token.feed_into(
                &storage(&forced),
                &storage(&close),
                &storage(&state),
                &storage(&output),
                0,
            )?;
        }
        token.record_counts(&storage(&counts))?;
        token.eos_flag_into(&storage(&close), &storage(&eos))?;
        token.copy_into(&storage(&history), 0)?;
        drop(token);
        finish(&d)?;
        if forced_id == 2 {
            assert_eq!(guard.poll_error()?, Some([0; 4]));
            assert_eq!(state.to_vec1::<u32>()?, [1, 0]);
            assert_eq!(history.to_vec1::<u32>()?, [2]);
            assert_eq!(counts.to_vec1::<f32>()?, [0., 0., 1.]);
            assert_eq!(eos.to_vec1::<u8>()?, [1]);
        } else {
            assert_eq!(guard.poll_error()?, Some([6, 0, 3, 0]));
            assert_eq!(state.to_vec1::<u32>()?, [0, 0]);
            assert_eq!(history.to_vec1::<u32>()?, [88]);
            assert_eq!(counts.to_vec1::<f32>()?, [0.; 3]);
            assert_eq!(eos.to_vec1::<u8>()?, [9]);
        }
    }
    Ok(())
}

#[test]
fn aliases_and_overflow_refuse_before_consumer_encoding() -> Result<()> {
    let d = device()?;
    let guard = SampleGuard::new(d.as_metal_device()?)?;
    let token = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
    assert!(token.copy_into(&token.sample.ids, 0).is_err());
    assert!(token.copy_into(&token.sample.status, usize::MAX).is_err());
    let table = storage(&Tensor::zeros(6, DType::F32, &d)?);
    assert!(token.embedding_into(&table, 0, 2, &table, 0).is_err());
    assert!(token
        .embedding_into(&table, 0, usize::MAX, &table, 0)
        .is_err());
    drop(token);
    finish(&d)
}

#[test]
fn embedding_preserves_f16_bf16_bytes_and_nonzero_element_offsets() -> Result<()> {
    let d = device()?;
    for dtype in [DType::F16, DType::BF16] {
        // Apple7 executes `bf16` only emulated, which §15.2a refuses without
        // the operator's opt-in; the f16 arm covers the byte path there.
        if dtype == DType::BF16
            && !d
                .as_metal_device()?
                .emulated_arms()
                .native()
                .has(crate::kernels::metal::arms::Feature::Bf16)
        {
            continue;
        }
        let guard = SampleGuard::new(d.as_metal_device()?)?;
        let table = Tensor::new(&[99f32, 1., 2., 3., 4., 5., 6., 99.], &d)?.to_dtype(dtype)?;
        let out = Tensor::new(&[-99f32; 4], &d)?.to_dtype(dtype)?;
        let token = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
        token.embedding_into(&storage(&table), 1, 2, &storage(&out), 1)?;
        drop(token);
        finish(&d)?;
        assert_eq!(
            out.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            [-99., 3., 4., -99.]
        );
    }
    Ok(())
}

#[test]
fn prepared_both_draws_share_support_and_checked_snapshot_without_early_reuse() -> Result<()> {
    use super::super::{
        prepared::{Draw, PreparedSampler},
        readback::SampleReadback,
    };
    let d = device()?;
    let m = d.as_metal_device()?;
    let mut scratch = PreparedSampler::new(m, 1, 2)?;
    let guard = SampleGuard::new(m)?;
    let mut reader = SampleReadback::new(m, 1)?;
    let z = storage(&Tensor::new(&[0f32, -1.0], &d)?);
    let g = storage(&Tensor::new(&[0f32, f32::from_bits(0x3eaa_aaab)], &d)?);
    let u = storage(&Tensor::new(&[0.99f32], &d)?);
    let zl = Layout::contiguous(2);
    let ul = Layout::contiguous(1);
    let history = Tensor::new(&[99u32], &d)?;
    for gumbel in [false, true] {
        let draw = if gumbel {
            Draw::Gumbel {
                variates: &g,
                layout: &zl,
            }
        } else {
            Draw::Multinomial {
                uniforms: &u,
                layout: &ul,
            }
        };
        let token = scratch.draw(&guard, &z, &zl, 3.0, 0, 1.0, draw)?;
        token.copy_into(&storage(&history), 0)?;
        token.snapshot(&mut reader)?;
        drop(token);
        assert!(reader.poll()?.is_none());
        assert!(scratch
            .draw(
                &guard,
                &z,
                &zl,
                3.0,
                0,
                1.0,
                Draw::Multinomial {
                    uniforms: &u,
                    layout: &ul
                }
            )
            .is_err());
        finish(&d)?;
        let expected = if gumbel { 0 } else { 1 };
        assert_eq!(reader.poll()?.expect("completed").ids(), &[expected]);
        assert_eq!(history.to_vec1::<u32>()?, [expected]);
        assert!(reader.poll()?.is_none());
    }
    assert_eq!(PreparedSampler::payload_bytes(2, 3)?, 144);
    assert!(PreparedSampler::payload_bytes(usize::MAX, 3).is_err());
    Ok(())
}

#[test]
fn prepared_invalid_window_and_queued_successor_stay_errors_in_the_host_bridge() -> Result<()> {
    use super::super::{
        prepared::{Draw, PreparedSampler},
        readback::SampleReadback,
    };
    let d = device()?;
    let m = d.as_metal_device()?;
    let guard = SampleGuard::new(m)?;
    let mut first = PreparedSampler::new(m, 2, 3)?;
    let mut next = PreparedSampler::new(m, 2, 3)?;
    let mut reader = SampleReadback::new(m, 2)?;
    let invalid = storage(&Tensor::new(&[[0f32; 3], [f32::NEG_INFINITY; 3]], &d)?);
    let valid = storage(&Tensor::new(&[[0f32; 3]; 2], &d)?);
    let u = storage(&Tensor::new(&[0.9f32; 2], &d)?);
    let zl = Layout::contiguous((2, 3));
    let ul = Layout::contiguous(2);
    let token = first.draw(
        &guard,
        &invalid,
        &zl,
        1.0,
        0,
        1.0,
        Draw::Multinomial {
            uniforms: &u,
            layout: &ul,
        },
    )?;
    drop(token);
    // No host poll/commit between the failed producer and its queued successor.
    let token = next.draw(
        &guard,
        &valid,
        &zl,
        1.0,
        0,
        1.0,
        Draw::Multinomial {
            uniforms: &u,
            layout: &ul,
        },
    )?;
    token.snapshot(&mut reader)?;
    drop(token);
    finish(&d)?;
    let error = match reader.poll() {
        Err(e) => e,
        _ => panic!("sticky error exposed ids"),
    };
    assert!(error.to_string().contains("row 1: Empty"), "{error}");
    Ok(())
}

#[test]
fn retained_publisher_observes_later_writer_and_survives_guard_drop() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let guard = SampleGuard::new(m)?;
    let mut publisher = guard.publisher();
    assert!(publisher.check_error()?, "initialized request is clean");
    let valid = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
    drop(valid);
    finish(&d)?;
    assert!(publisher.check_error()?);

    let invalid = guard.bind(sample(&d, &[u32::MAX], &[3, u32::MAX, 0, 0])?)?;
    drop(invalid);
    assert!(
        !publisher.check_error()?,
        "later open-epoch writer supersedes the completed epoch"
    );
    drop(guard);
    finish(&d)?;
    assert!(
        publisher.check_error().is_err(),
        "sticky diagnostic survives producer drop without any token stamp"
    );
    Ok(())
}

#[test]
fn retained_publisher_checks_poison_and_uses_guarded_final_stamp() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let guard = SampleGuard::new(m)?;
    let mut publisher = guard.publisher();
    let marker = Tensor::new(&[0u32], &d)?;
    let history = Tensor::new(&[1u32], &d)?;
    let input = Tensor::new(&[1u32], &d)?;
    let eos = Tensor::new(&[0u8], &d)?;
    let token = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
    drop(token);
    publisher.publish_marker(
        &storage(&marker),
        7,
        &storage(&history),
        &storage(&input),
        &storage(&eos),
    )?;
    finish(&d)?;
    assert_eq!(marker.to_vec1::<u32>()?, [7]);
    assert!(publisher.check_error()?);
    guard.progress.poisoned.set(true);
    assert!(
        publisher.check_error().is_err(),
        "a poisoned native entry must not expose old completed words"
    );
    assert!(publisher
        .publish_marker(
            &storage(&marker),
            8,
            &storage(&history),
            &storage(&input),
            &storage(&eos)
        )
        .is_err());
    Ok(())
}

#[test]
fn failed_native_entry_is_visible_to_older_publisher_without_a_stamp() -> Result<()> {
    let d = device()?;
    let guard = SampleGuard::new(d.as_metal_device()?)?;
    let mut publisher = guard.publisher();
    assert!(publisher.check_error()?);
    drop(guard.native_entry());
    assert!(publisher.check_error().is_err());
    assert!(guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?).is_err());
    let fresh = SampleGuard::new(d.as_metal_device()?)?;
    let mut fresh_publisher = fresh.publisher();
    assert!(fresh_publisher.check_error()?);
    assert!(
        publisher.check_error().is_err(),
        "fresh request cannot clear its predecessor's failed native entry"
    );
    Ok(())
}

#[test]
fn invalid_later_row_and_valid_queued_successor_never_index_a_packed_table() -> Result<()> {
    use crate::quantized::{GgmlDType, QMatMul, QTensor};
    let d = device()?;
    let weights = Tensor::from_vec(
        (0..96).map(|x| (x as f32 - 31.) / 17.).collect(),
        (3, 32),
        &d,
    )?;
    let table = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
        &weights,
        GgmlDType::Q4_0,
    )?));
    let mut guard = SampleGuard::new(d.as_metal_device()?)?;
    let output = Tensor::full(-99f32, (2, 32), &d)?;
    let bad = guard.bind(sample(
        &d,
        &[1, u32::MAX],
        &[0, u32::MAX, 3, 2, 3, u32::MAX, 0, 0],
    )?)?;
    table.embedding_guarded_into(&bad, &output)?;
    drop(bad);
    let good = guard.bind(sample(
        &d,
        &[1, 2],
        &[0, u32::MAX, 3, 2, 0, u32::MAX, 3, 2],
    )?)?;
    table.embedding_guarded_into(&good, &output)?;
    drop(good);
    finish(&d)?;
    assert!(guard.check_error().is_err());
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        vec![-99.; 64],
        "all rows fail before get_rows can index; queued valid successors stay suppressed"
    );
    Ok(())
}

// The current encoder may differ from the installed default without a race:
// replacement affects only the next encoder. A forward-only custom executor
// makes the old predicate observable without constructing unsafe native work.
fn retained_custom_encoder(
    m: &MetalDevice,
) -> Result<std::sync::Arc<std::sync::atomic::AtomicUsize>> {
    use crate::kernels::metal::executor::{DispatchRecord, Executor, ExecutorSlot};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Count(Arc<AtomicUsize>);
    impl Executor for Count {
        fn dispatch(&self, _: &DispatchRecord) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            true
        }
    }
    m.commands().flush().map_err(MetalError::from)?;
    m.wait_until_completed()?;
    assert_eq!(
        m.cut_policy(),
        crate::kernels::metal::CutPolicy::DispatchCounter,
        "actual-encoder fixture requires counter cuts"
    );
    m.set_compute_per_buffer(usize::MAX);
    let count = Arc::new(AtomicUsize::new(0));
    m.set_executor(Arc::new(ExecutorSlot::Custom(Arc::new(Count(
        count.clone(),
    )))));
    {
        let encoder = m.command_encoder()?;
        assert!(!encoder.as_ref().executor.is_classical());
    }
    m.set_executor(Arc::new(ExecutorSlot::Classical));
    assert!(m.commands().executor().is_classical());
    {
        let encoder = m.command_encoder()?;
        assert!(
            !encoder.as_ref().executor.is_classical(),
            "fixture must retain old encoder"
        );
    }
    Ok(count)
}

#[test]
fn prepared_draw_checks_retained_encoder_not_installed_default() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let scores = storage(&Tensor::new(&[0f32, 0., 0.], &d)?);
    let layout = Layout::contiguous(3);
    let mut prepared = super::super::prepared::PreparedSampler::with_noise(m, 1, 3)?;
    let guard = SampleGuard::new(m)?;
    let dispatched = retained_custom_encoder(m)?;
    let result =
        prepared.draw_generated(&guard, &scores, &layout, 1., 0, 1., false, |out, _, _| {
            out[0] = 0.5;
            Ok(())
        });
    let error = match result {
        Ok(token) => {
            drop(token);
            None
        }
        Err(e) => Some(e),
    };
    finish(&d)?;
    println!("ACTUAL_ENCODER_TERMINAL prepared=true");
    let err = error.expect("prepared sampler admitted a retained nonclassical encoder");
    assert!(err.to_string().contains("actual encoder"), "{err}");
    assert_eq!(dispatched.load(std::sync::atomic::Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn guarded_consumer_checks_retained_encoder_not_installed_default() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let guard = SampleGuard::new(m)?;
    let input = sample(&d, &[1], &[0, u32::MAX, 3, 2])?;
    let dispatched = retained_custom_encoder(m)?;
    let error = match guard.bind(input) {
        Ok(token) => {
            drop(token);
            None
        }
        Err(e) => Some(e),
    };
    finish(&d)?;
    println!("ACTUAL_ENCODER_TERMINAL guard=true");
    let err = error.expect("guard admitted a retained nonclassical encoder");
    assert!(err.to_string().contains("actual encoder"), "{err}");
    assert_eq!(dispatched.load(std::sync::atomic::Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn sample_readback_checks_retained_encoder_not_installed_default() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let mut readback = super::super::readback::SampleReadback::new(m, 1)?;
    let input = sample(&d, &[1], &[0, u32::MAX, 3, 2])?;
    let dispatched = retained_custom_encoder(m)?;
    let error = readback.enqueue(input).err();
    finish(&d)?;
    println!("ACTUAL_ENCODER_TERMINAL readback=true");
    let err = error.expect("readback admitted nonclassical encoder");
    assert!(err.to_string().contains("actual encoder"), "{err}");
    assert_eq!(dispatched.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(
        readback.poll().is_err(),
        "failed snapshot must remain poisoned"
    );
    Ok(())
}

#[test]
fn prepared_draw_accepts_classical_encoder_after_a_cut() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let scores = storage(&Tensor::new(&[0f32, 0., 0.], &d)?);
    let layout = Layout::contiguous(3);
    let mut prepared = super::super::prepared::PreparedSampler::with_noise(m, 1, 3)?;
    let guard = SampleGuard::new(m)?;
    let mut readback = super::super::readback::SampleReadback::new(m, 1)?;
    let dispatched = retained_custom_encoder(m)?;
    // A cut replaces the old custom encoder with the installed Classical
    // executor. Rejecting the queue merely because it once held Custom would
    // disable a supported route instead of validating the actual dispatch.
    m.set_compute_per_buffer(1);
    let token =
        prepared.draw_generated(&guard, &scores, &layout, 1., 0, 1., false, |out, _, _| {
            out[0] = 0.5;
            Ok(())
        })?;
    token.snapshot(&mut readback)?;
    drop(token);
    finish(&d)?;
    assert_eq!(readback.poll()?.expect("completed snapshot").ids(), [1]);
    assert_eq!(dispatched.load(std::sync::atomic::Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn packed_embedding_accepts_foreign_readonly_owner_but_not_foreign_output() -> Result<()> {
    use crate::metal_backend::MetalAllocationDomain;
    use crate::quantized::{GgmlDType, QMatMul, QTensor};
    let d = device()?;
    let weights = Tensor::from_vec((0..96).map(|x| x as f32 / 17.).collect(), (3, 32), &d)?;
    let table = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
        &weights,
        GgmlDType::Q4_0,
    )?));
    let foreign_output = Tensor::full(-99f32, (1, 32), &d)?;
    let domain = crate::memory_admission::ModelDomain::isolated_numeric_fixture(64 << 20);
    let namespace = MetalAllocationDomain::new(d.as_metal_device()?, domain)?;
    let promise = namespace.domain().reserve(16 << 20)?;
    let scope = namespace.scope(promise.context())?;
    scope.run(|metal| {
        let local = Device::Metal(metal.clone());
        let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let guard = SampleGuard::new(metal)?;
            let output = Tensor::full(-99f32, (1, 32), &local)?;
            let token = guard.bind(sample(&local, &[1], &[0, u32::MAX, 3, 2])?)?;
            let error = table
                .embedding_guarded_into(&token, &foreign_output)
                .unwrap_err();
            assert!(error.to_string().contains("domain"), "{error}");
            table.embedding_guarded_into(&token, &output)?;
            drop(token);
            finish(&local)?;
            // Same dequantization/get_rows representation, not original float weights.
            let expected = table.embedding(&Tensor::new(&[1u32], &local)?)?;
            assert_eq!(output.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
            assert_eq!(
                foreign_output.flatten_all()?.to_vec1::<f32>()?,
                vec![-99.; 32]
            );
            Ok(())
        }));
        let cleanup = metal.wait_until_completed();
        match body {
            Ok(result) => {
                result?;
                cleanup
            }
            Err(p) => {
                let _ = cleanup;
                std::panic::resume_unwind(p)
            }
        }
    })
}

#[test]
fn filler_reentrant_invalid_guard_remains_sticky_for_its_own_draw() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let scores = storage(&Tensor::new(&[0f32; 3], &d)?);
    let bad = sample(&d, &[u32::MAX], &[3, u32::MAX, 0, 0])?;
    let guard = SampleGuard::new(m)?;
    let mut prepared = super::super::prepared::PreparedSampler::with_noise(m, 1, 3)?;
    let mut readback = super::super::readback::SampleReadback::new(m, 1)?;
    let token = prepared.draw_generated(
        &guard,
        &scores,
        &Layout::contiguous(3),
        1.,
        0,
        1.,
        false,
        |out, _, _| {
            drop(guard.bind(bad)?);
            out[0] = 0.5;
            Ok(())
        },
    )?;
    token.snapshot(&mut readback)?;
    drop(token);
    let result = readback.read_committed();
    assert!(
        result.is_err(),
        "reentrant invalid row was reset by outer draw"
    );
    finish(&d)
}

#[test]
fn packed_embedding_rejects_other_queue_and_aliased_weight_output() -> Result<()> {
    use crate::kernels::metal::{BufferHandle, PooledBuffer};
    use crate::quantized::{metal::QMetalStorage, GgmlDType};
    let d = device()?;
    let m = d.as_metal_device()?;
    let other = Device::new_metal(0)?;
    let foreign = QMetalStorage::zeros(other.as_metal_device()?, 96, GgmlDType::F32)?;
    let table = QMetalStorage::zeros(m, 96, GgmlDType::F32)?;
    let guard = SampleGuard::new(m)?;
    let output = Tensor::full(-99f32, (1, 32), &d)?;
    let token = guard.bind(sample(&d, &[1], &[0, u32::MAX, 3, 2])?)?;
    assert!(
        !foreign.device().same_device(m),
        "fixture needs distinct Commands identity"
    );
    assert!(token
        .quantized_embedding_into(&foreign, 3, 32, &output)
        .is_err());
    // Retain the creator until all uses finish. This second owned view is only
    // passed to the rejecting preflight, not submitted as independent storage.
    let alias = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(
            BufferHandle::new(PooledBuffer::unpooled(
                table.buffer().clone(),
                table.buffer().length(),
            )),
            m.clone(),
            96,
            DType::F32,
        )),
        (1, 32),
        crate::op::BackpropOp::none(),
        false,
    );
    let error = token
        .quantized_embedding_into(&table, 3, 32, &alias)
        .unwrap_err();
    assert!(
        error.to_string().contains("aliased weight/output"),
        "{error}"
    );
    drop(token);
    finish(&d)?;
    finish(&other)?;
    assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![-99.; 32]);
    Ok(())
}
