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
    next.feed_into(
        &storage(&forced),
        &storage(&close),
        &storage(&state),
        &storage(&output),
        0,
    )?;
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
        token.feed_into(
            &storage(&forced),
            &storage(&close),
            &storage(&state),
            &storage(&output),
            0,
        )?;
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
