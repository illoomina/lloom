use crate::backend::BackendStorage;
use crate::{DType, Device, Layout, MetalStorage, Result, Storage, Tensor};

// These process-global readback counters require nextest's one process per test.
// A claim diagnoses competing claimants, not unrelated parallel libtest readers.
fn test_device(force_blit: bool) -> Result<Device> {
    crate::isolation::claim("sampling top-k readback counters");
    let mut tuning = crate::tuning::Tuning::default();
    tuning.from_env().expect("valid test environment");
    tuning.contract.profile = "deterministic".into();
    tuning.instrument.profile = true;
    tuning.metal.to_cpu_blit = force_blit;
    crate::tuning::install(tuning).expect("install before the first tuning read");
    assert!(crate::kernels::metal::profile::enabled());
    println!(
        "TOPK_TEST_SETUP profile=deterministic merged_unit={}",
        crate::kernels::kernel::merged_translation_unit()
    );
    Device::new_metal(0)
}

use crate::{fixtures, oracle};

fn select(z: &Tensor, k: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    let (storage, layout) = z.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        panic!("the candidate tests require Metal storage");
    };
    let (mask, status) = checked_selection(storage, layout, k)?;
    Ok((mask.to_cpu::<u32>()?, status.to_cpu::<u32>()?))
}

// Neither uploads nor output readbacks belong in the selector census window.
fn checked_selection(
    storage: &MetalStorage,
    layout: &Layout,
    k: usize,
) -> Result<(MetalStorage, MetalStorage)> {
    let before = super::super::readback_counts();
    let window = storage
        .device()
        .dispatch_census()
        .open()
        .expect("a census window");
    let outputs = storage.sampling_topk_mask(layout, k)?;
    let census = window.close();
    assert_eq!(
        super::super::readback_counts(),
        before,
        "selection read back"
    );
    let selector_dispatches: u64 = census
        .iter()
        .filter(|(label, _)| label == "sampling_topk_mask_f32")
        .map(|(_, n)| *n)
        .sum();
    let expected = u64::from(layout.shape().elem_count() != 0);
    assert_eq!(selector_dispatches, expected, "selector census: {census:?}");
    Ok(outputs)
}

fn check(row: &[f32], k: usize, mask: &[u32], status: &[u32]) {
    assert_eq!(mask.len(), row.len());
    assert!(mask.iter().all(|&x| x <= 1), "mask must be binary");
    assert_eq!(status.len(), 4);
    let ids: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (x != 0).then_some(i))
        .collect();
    let finite = row.iter().filter(|x| x.is_finite()).count() as u32;
    match oracle::top_k(row, k) {
        Ok(mut expected) => {
            expected.sort_unstable();
            assert_eq!(ids, expected, "k={k}");
            assert_eq!(status, &[0, u32::MAX, finite, ids.len() as u32]);
        }
        Err(why) => {
            let (code, id) = match why {
                oracle::Refusal::NotANumber(i) => (1, i as u32),
                oracle::Refusal::PositiveInfinity(i) => (2, i as u32),
                oracle::Refusal::NoCandidate => (3, u32::MAX),
                other => panic!("a selection-only oracle returned {other:?}"),
            };
            assert!(ids.is_empty());
            assert_eq!(status, &[code, id, finite, 0]);
        }
    }
}

#[test]
fn fixtures_match_the_topk_stage_and_hand_derived_support() -> Result<()> {
    let device = test_device(false)?;
    let (mut rows, mut hand_sets, mut widened) = (0, 0, 0);
    for case in fixtures::all() {
        let z = Tensor::from_vec(case.z.clone(), case.z.len(), &device)?;
        let (mask, status) = select(&z, case.params.k)?;
        check(&case.z, case.params.k, &mask, &status);
        if case.params.p == 1.0 && oracle::inv_t(case.params.temperature).is_ok() {
            if let fixtures::Expect::Kept(expected) = &case.expect {
                let ids: Vec<u32> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| (m != 0).then_some(i as u32))
                    .collect();
                assert_eq!(&ids, expected, "{}", case.name);
                hand_sets += 1;
            }
        }
        if case.f16_exact {
            let back = z.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            let (got_mask, got_status) = select(&back, case.params.k)?;
            assert_eq!(got_mask, mask, "{} widened f16 mask", case.name);
            assert_eq!(got_status, status, "{} widened f16 status", case.name);
            widened += 1;
        }
        let after = z.to_vec1::<f32>()?;
        assert_eq!(
            after.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            case.z.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{}: selection mutated processed scores",
            case.name
        );
        rows += 1;
    }
    assert_eq!((rows, widened), (44, 41));
    assert_eq!(hand_sets, 18, "hand-derived top-k expectations did not run");
    println!("TOPK_FIXTURES rows={rows} widened={widened} hand_sets={hand_sets}");
    Ok(())
}

#[test]
fn bit_rank_masks_and_large_k_match_the_independent_reference() -> Result<()> {
    let device = test_device(false)?;
    let mut seed = 0x1036_1045u32;
    for v in [1usize, 31, 257, 1025, 4099] {
        let mut row: Vec<f32> = (0..v)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                match i % 5 {
                    0 => f32::NEG_INFINITY,
                    1 => (i % 7) as f32 - 3.0,
                    _ => f32::from_bits(seed & 0xfeff_ffff),
                }
            })
            .collect();
        row[0] = 0.0;
        let z = Tensor::from_vec(row.clone(), v, &device)?;
        for k in [0, 1, 2, 20, 50, 64, 65, 1024, v - 1, v, v + 1, usize::MAX] {
            let (mask, status) = select(&z, k)?;
            check(&row, k, &mask, &status);
        }
    }
    let row = [
        f32::MIN,
        -1.0,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.0,
        f32::MAX,
    ];
    let z = Tensor::new(&row, &device)?;
    for k in 0..=row.len() + 1 {
        let (mask, status) = select(&z, k)?;
        check(&row, k, &mask, &status);
    }
    Ok(())
}

#[test]
fn wide_flat_rows_keep_exactly_k_lowest_ids() -> Result<()> {
    let device = test_device(false)?;
    for v in [128_000usize, 248_320] {
        let z = Tensor::from_vec(vec![0.0f32; v], v, &device)?;
        for k in [1, 20, 50, 65, v / 2, v - 1, v] {
            let (mask, status) = select(&z, k)?;
            assert!(mask[..k].iter().all(|&m| m == 1));
            assert!(mask[k..].iter().all(|&m| m == 0));
            assert_eq!(status, [0, u32::MAX, v as u32, k as u32]);
        }
    }
    Ok(())
}

#[test]
fn windows_offsets_refusals_permutation_and_replay_are_row_local() -> Result<()> {
    let device = test_device(false)?;
    for v in [128_000usize, 248_320] {
        let mut rows = Vec::new();
        for r in 0..10 {
            let mut row = fixtures::planted(v, -4.0, &[(v - 1 - r, 3.0), (r, 3.0)]);
            match r {
                1 => row.fill(f32::NEG_INFINITY),
                2 => {
                    row[65] = f32::NAN;
                    row[0] = f32::INFINITY;
                }
                3 => {
                    row[65] = f32::INFINITY;
                    row[0] = f32::from_bits(0xffc0_0001);
                }
                4 => {
                    row.fill(f32::NEG_INFINITY);
                    row[v - 1] = -0.0;
                }
                _ => (),
            }
            rows.push(row);
        }
        let mut all = vec![f32::NAN; v];
        all.extend(rows.iter().flatten().copied());
        all.extend(vec![f32::INFINITY; v]);
        let backing = Tensor::from_vec(all, (12, v), &device)?;
        let window = backing.narrow(0, 1, 10)?.reshape((1, 10, v))?;
        for k in [0, 1, 20, 50, v] {
            let baseline = select(&window, k)?;
            for (i, row) in rows.iter().enumerate() {
                check(
                    row,
                    k,
                    &baseline.0[i * v..(i + 1) * v],
                    &baseline.1[i * 4..(i + 1) * 4],
                );
                let single = window.reshape((10, v))?.narrow(0, i, 1)?;
                assert_eq!(
                    select(&single, k)?,
                    (
                        baseline.0[i * v..(i + 1) * v].to_vec(),
                        baseline.1[i * 4..(i + 1) * 4].to_vec()
                    )
                );
            }
            for _ in 0..3 {
                assert_eq!(select(&window, k)?, baseline);
            }
            let reversed: Vec<f32> = rows.iter().rev().flatten().copied().collect();
            let reversed = Tensor::from_vec(reversed, (10, v), &device)?;
            let got = select(&reversed, k)?;
            for i in 0..10 {
                assert_eq!(
                    &got.0[i * v..(i + 1) * v],
                    &baseline.0[(9 - i) * v..(10 - i) * v]
                );
                assert_eq!(
                    &got.1[i * 4..(i + 1) * 4],
                    &baseline.1[(9 - i) * 4..(10 - i) * 4]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn malformed_layouts_are_refused_before_encoding() -> Result<()> {
    let device = test_device(false)?;
    let z = Tensor::new(&[0f32; 8], &device)?;
    let (storage, _) = z.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        unreachable!()
    };
    for layout in [
        Layout::contiguous(()),
        Layout::contiguous((1, 0)),
        Layout::contiguous_with_offset((2, 4), 1),
        Layout::new((2, 4).into(), vec![1, 2], 0),
        Layout::new((usize::MAX, 2).into(), vec![2, 1], 0),
        Layout::contiguous_with_offset((1, 4), usize::MAX),
    ] {
        assert!(storage.sampling_topk_mask(&layout, 1).is_err());
    }
    let empty = checked_selection(storage, &Layout::contiguous((0, 4)), 1)?;
    assert!(empty.0.to_cpu::<u32>()?.is_empty());
    assert!(empty.1.to_cpu::<u32>()?.is_empty());
    let half = z.to_dtype(DType::F16)?;
    let (storage, layout) = half.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        unreachable!()
    };
    assert!(storage.sampling_topk_mask(layout, 1).is_err());
    Ok(())
}

#[test]
fn raw_dispatch_initializes_outputs_and_preserves_tail_canaries() -> Result<()> {
    let device = test_device(false)?;
    let (rows, vocab, guard) = (3usize, 37usize, 19usize);
    let mut input = vec![f32::NAN; 5];
    for r in 0..rows {
        input.extend((0..vocab).map(|i| ((i + r) % 7) as f32 - 3.0));
    }
    input.extend(vec![f32::INFINITY; guard]);
    let input = Tensor::from_vec(input.clone(), input.len(), &device)?;
    let mask = Tensor::from_vec(
        vec![0xdecafbad_u32; rows * vocab + guard],
        rows * vocab + guard,
        &device,
    )?;
    let status = Tensor::from_vec(
        vec![0xdecafbad_u32; 4 * rows + guard],
        4 * rows + guard,
        &device,
    )?;
    {
        let (input, _) = input.storage_and_layout();
        let (mask, _) = mask.storage_and_layout();
        let (status, _) = status.storage_and_layout();
        let (Storage::Metal(input), Storage::Metal(mask), Storage::Metal(status)) =
            (&*input, &*mask, &*status)
        else {
            unreachable!()
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        // SAFETY: the three storages own disjoint live buffers, sized above;
        // readback below waits for the dispatch before observing output.
        unsafe {
            crate::kernels::kernels::sampling_topk::call_sampling_topk_mask(
                device.metal_device(),
                &encoder,
                device.kernels(),
                (input.buffer(), 5 * 4),
                rows as u32,
                vocab as u32,
                20,
                mask.buffer(),
                status.buffer(),
            )
        }
        .map_err(crate::Error::wrap)?;
    }
    let mask = mask.to_vec1::<u32>()?;
    let status = status.to_vec1::<u32>()?;
    let scores = input.to_vec1::<f32>()?;
    for r in 0..rows {
        check(
            &scores[5 + r * vocab..5 + (r + 1) * vocab],
            20,
            &mask[r * vocab..(r + 1) * vocab],
            &status[r * 4..(r + 1) * 4],
        );
    }
    assert!(mask[rows * vocab..].iter().all(|&x| x == 0xdecafbad));
    assert!(status[rows * 4..].iter().all(|&x| x == 0xdecafbad));
    Ok(())
}

#[test]
fn readback_counters_observe_ordinary_staged_and_slot_reads() -> Result<()> {
    let device = test_device(false)?;
    let expected = vec![7u32, 11, 13, 17];
    let z = Tensor::from_vec(expected.clone(), expected.len(), &device)?;
    let slot = super::super::HostSlot::new(device.as_metal_device()?, 4, DType::U32)?;
    slot.write_from(&z)?;
    let before = super::super::readback_counts();
    assert_eq!(slot.read_committed::<u32>(4)?, expected);
    assert_eq!(super::super::readback_counts().since(&before).slot, 1);

    // Shared slot storage gives an ordinary read a guaranteed mapped source.
    let shared = slot.tensor(4)?;
    let before = super::super::readback_counts();
    assert_eq!(shared.to_vec1::<u32>()?, expected);
    assert_eq!(super::super::readback_counts().since(&before).fast, 1);

    let (storage, _) = z.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        unreachable!()
    };
    let before = super::super::readback_counts();
    assert_eq!(storage.stage_read(0, 4)?.read::<u32>()?, expected);
    let staged = super::super::readback_counts().since(&before);
    assert_eq!(staged.staged, 1);
    assert_eq!(staged.staging_allocs, 1);

    let before = super::super::readback_counts();
    assert_eq!(super::super::stage_readback(&z)?.finish::<u32>()?, expected);
    let readback = super::super::readback_counts().since(&before);
    assert_eq!(readback.staged_readback, 1);
    assert_eq!(readback.staged_wait, 1);
    assert_eq!(readback.staging_allocs, 1);
    println!(
        "TOPK_READBACK_CONTROLS fast=1 slot=1 stage_read={staged:?} stage_readback={readback:?}"
    );
    Ok(())
}

#[test]
fn forced_blit_readback_counter_is_observable() -> Result<()> {
    let device = test_device(true)?;
    let z = Tensor::new(&[7u32, 11, 13, 17], &device)?;
    let before = super::super::readback_counts();
    assert_eq!(z.to_vec1::<u32>()?, [7, 11, 13, 17]);
    let readback = super::super::readback_counts().since(&before);
    assert_eq!(readback.blit, 1);
    assert_eq!(readback.staging_allocs, 1);
    println!("TOPK_READBACK_BLIT_CONTROL {readback:?}");
    Ok(())
}

// Keep every originating storage and both persistent destinations alive through
// successful queue completion. No mapping or value read separates the dispatches.
fn queued_dependency_fixture(force_cut: bool) -> Result<()> {
    use crate::kernels::metal::commands::CommandBufferKind;
    use objc2_metal::{MTLComputePipelineState as _, MTLDevice as _};
    use std::sync::Arc;

    let device = test_device(false)?;
    let metal = device.as_metal_device()?;
    let tuning = crate::tuning::tuning();
    assert_eq!(tuning.metal.command_api, "metal3");
    // This source base predates the selectable reuse-sync arm.
    assert!(!tuning.metal.transient_heaps);
    assert!(!tuning.metal.private_cpu_mapping);
    assert!(!tuning.instrument.profile_encoder_per_dispatch);
    assert_eq!(metal.commands().kind(), CommandBufferKind::Retaining);
    assert_eq!(super::super::MetalDevice::buffer_alignment(), 128);
    assert!(!metal.base_mode());
    assert!(metal.commands().executor().is_classical());
    assert_eq!(
        crate::kernels::kernels::params::default_param_style(),
        crate::kernels::kernels::params::ParamStyle::Split
    );
    assert_eq!(metal.cut_policy(), super::super::CutPolicy::DispatchCounter);
    metal.set_compute_per_buffer(usize::MAX);
    let pipeline = metal
        .kernels()
        .load_pipeline(
            metal.metal_device(),
            crate::kernels::source::Source::Random,
            "sampling_topk_mask_f32",
        )
        .map_err(crate::Error::wrap)?;
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    let static_bytes = pipeline.as_ref().staticThreadgroupMemoryLength();
    let limit = metal.metal_device().as_ref().maxThreadgroupMemoryLength();
    assert!(width > 0);
    assert!(static_bytes <= limit);
    println!("TOPK_NATIVE_LIMITS width={width} static_bytes={static_bytes} limit={limit}");

    let values = [3.0f32, 1.0, -2.0, -2.0];
    let input = Tensor::from_vec(values.to_vec(), (1, 4), &device)?;
    let mask_slot = super::super::HostSlot::new(metal, 4, DType::U32)?;
    let status_slot = super::super::HostSlot::new(metal, 4, DType::U32)?;
    mask_slot.fill_u32(0xdecafbad, 4)?;
    status_slot.fill_u32(0xdecafbad, 4)?;
    let (input_storage, layout) = input.storage_and_layout();
    let Storage::Metal(input_storage) = &*input_storage else {
        unreachable!()
    };
    let snapshot = || -> Result<_> {
        let encoder = metal.command_encoder()?;
        let encoder = encoder.as_ref();
        assert!(!encoder.command_buffer.is_metal4());
        Ok((encoder.command_buffer.clone(), Arc::clone(&encoder.state)))
    };

    let before = super::super::readback_counts();
    let produced = input_storage.affine(layout, -1.0, 1.0)?;
    let producer = snapshot()?;
    if force_cut {
        metal.commands().flush().map_err(crate::Error::wrap)?;
    }
    let (mask, status) = checked_selection(&produced, layout, 1)?;
    let selector = snapshot()?;
    assert_eq!(Arc::ptr_eq(&producer.1, &selector.1), !force_cut);
    assert_eq!(producer.0.identity() == selector.0.identity(), !force_cut);
    if force_cut {
        metal.commands().flush().map_err(crate::Error::wrap)?;
    }
    let mask = Tensor::from_storage(
        Storage::Metal(mask),
        (1, 4),
        crate::op::BackpropOp::none(),
        false,
    );
    let status = Tensor::from_storage(
        Storage::Metal(status),
        (1, 4),
        crate::op::BackpropOp::none(),
        false,
    );
    mask_slot.write_from(&mask)?;
    let mask_consumer = snapshot()?;
    status_slot.write_from(&status)?;
    let status_consumer = snapshot()?;
    assert_eq!(Arc::ptr_eq(&selector.1, &mask_consumer.1), !force_cut);
    assert_eq!(
        selector.0.identity() == mask_consumer.0.identity(),
        !force_cut
    );
    assert!(Arc::ptr_eq(&mask_consumer.1, &status_consumer.1));
    assert_eq!(mask_consumer.0.identity(), status_consumer.0.identity());
    assert_eq!(super::super::readback_counts(), before);

    metal.wait_until_completed()?;
    let expected: Vec<f32> = values.iter().map(|&x| 1.0 - x).collect();
    let copied_mask = mask_slot.read_unchecked::<u32>(4)?;
    let copied_status = status_slot.read_unchecked::<u32>(4)?;
    check(&expected, 1, &copied_mask, &copied_status);
    assert_eq!(copied_mask, [0, 0, 1, 0]);
    assert_eq!(copied_status, [0, u32::MAX, 4, 1]);
    assert_eq!(super::super::readback_counts().since(&before).slot, 2);
    println!(
        "TOPK_DEPENDENCY rows=1 vocab=4 k=1 force_cut={force_cut} owner_completed=true mask={copied_mask:?} status={copied_status:?}"
    );
    Ok(())
}

#[test]
fn queued_producer_selector_consumers_share_one_encoder() -> Result<()> {
    queued_dependency_fixture(false)
}

#[test]
fn queued_producer_selector_consumers_cross_forced_cuts() -> Result<()> {
    queued_dependency_fixture(true)
}
