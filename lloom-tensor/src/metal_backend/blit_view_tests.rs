//! Byte copies address a view at its base, and stay inside Metal's
//! buffer-copy contract (`docs/invariants.md` §15.2 #12c).
//!
//! A `Buffer::view` shares its parent's `MTLBuffer`, so a native blit must be
//! given `base_offset + offset`; a compute binding adds the base itself. And
//! on macOS a buffer-to-buffer blit needs its source offset, destination offset
//! and size to be multiples of 4: anything else takes the compute byte copy.
//!
//! Each test asserts the bytes AND what was emitted (`encoder::emitted`): a
//! numerically green readback does not show the contract was met, because
//! Apple silicon may copy an unaligned range correctly anyway.
//!
//! The fixtures need a device outside every pass plan, where the legacy blit
//! arms run; `Device::new_metal(0)` is process-wide, so each asserts that.

use super::{stage_readback, stage_readback_deferred, MetalDevice, MetalStorage};
use crate::backend::BackendStorage;
use crate::kernels::metal::encoder::emitted::{self, Emitted};
use crate::kernels::metal::{ArenaLayout, BufferHandle, PooledBuffer};
use crate::op::BackpropOp;
use crate::{DType, Device, Result, Storage, Tensor};

fn metal() -> Option<(Device, MetalDevice)> {
    match Device::new_metal(0) {
        Ok(Device::Metal(m)) => Some((Device::Metal(m.clone()), m)),
        _ => None,
    }
}

fn non_base(m: &MetalDevice) {
    assert!(
        !m.base_mode(),
        "fixture precondition: no pass plan or arena, so the legacy blit arms run"
    );
}

/// Lane C's sentinel (#985): `[17, 23, u32::MAX, 47]` uploaded, viewed at
/// byte 4 for 12 bytes, narrowed to element 1. The view base is B = 4, the
/// layout start L = 4 bytes, so the right byte is B + L = 8 (`u32::MAX`); a
/// lost base reads 4 (`23`) and a doubled one 12 (`47`). Every address is
/// inside the 16 initialized bytes, so no wrong-address variant copies out of
/// bounds.
fn sentinel(m: &MetalDevice) -> Result<(BufferHandle, Tensor)> {
    let creator = m.new_buffer_with_data(&[17u32, 23, u32::MAX, 47])?;
    let view = BufferHandle::new(PooledBuffer::unpooled(creator.view(4, 12), 12));
    assert_eq!(view.base_offset(), 4);
    let storage = MetalStorage::new(view, m.clone(), 3, DType::U32);
    let t = Tensor::from_storage(Storage::Metal(storage), 3, BackpropOp::none(), false);
    Ok((creator, t.narrow(0, 1, 1)?))
}

/// The same shape over bytes: 16 initialized bytes, viewed at 4 for 12,
/// narrowed to `start..start + len`. The right first byte is `4 + start`.
fn byte_view(m: &MetalDevice, start: usize, len: usize) -> Result<(BufferHandle, Tensor)> {
    let bytes: Vec<u8> = (0..16).map(|i| 100 + i as u8).collect();
    let creator = m.new_buffer_with_data(&bytes)?;
    let view = BufferHandle::new(PooledBuffer::unpooled(creator.view(4, 12), 12));
    let storage = MetalStorage::new(view, m.clone(), 12, DType::U8);
    let t = Tensor::from_storage(Storage::Metal(storage), 12, BackpropOp::none(), false);
    Ok((creator, t.narrow(0, start, len)?))
}

fn expected_bytes(start: usize, len: usize) -> Vec<u8> {
    (0..len).map(|i| 100 + (4 + start + i) as u8).collect()
}

/// A fresh destination with base 0, and the log cleared of its creation.
fn zeros(dev: &Device, n: usize, dtype: DType) -> Result<Tensor> {
    let t = Tensor::zeros(n, dtype, dev)?;
    emitted::take();
    Ok(t)
}

// --- view base: the sentinel through every raw-blit entry -----------------

#[test]
fn a_view_is_staged_for_readback_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    emitted::take();
    let v: Vec<u32> = stage_readback(&t)?.finish()?;
    let log = emitted::take();
    assert_eq!(
        v,
        [u32::MAX],
        "lost base reads 23, doubled 47; emitted {log:?}"
    );
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

#[test]
fn a_deferred_stage_of_a_view_reads_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    emitted::take();
    let v: Vec<u32> = stage_readback_deferred(&t)?.finish()?;
    let log = emitted::take();
    assert_eq!(v, [u32::MAX], "emitted {log:?}");
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

#[test]
fn a_staged_read_of_a_view_reads_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    emitted::take();
    let crate::StagedRead::Metal(staged) = t.stage_read()? else {
        panic!("a Metal tensor stages a Metal read")
    };
    let v: Vec<u32> = staged.read()?;
    let log = emitted::take();
    assert_eq!(v, [u32::MAX], "emitted {log:?}");
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

/// `slice_set` reaches `copy2d`'s contiguous fast path (`d2 == src_s ==
/// dst_s`).
#[test]
fn copy2d_copies_a_view_from_its_base() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    let dst = zeros(&dev, 1, DType::U32)?;
    dst.slice_set(&t, 0, 0)?;
    let log = emitted::take();
    assert_eq!(dst.to_vec1::<u32>()?, [u32::MAX], "emitted {log:?}");
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

/// The fast path called directly, with a view on BOTH sides: the destination
/// base is added as well.
#[test]
fn copy2d_writes_a_view_destination_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    let dst_creator = m.new_buffer_with_data(&[0u32; 4])?;
    let dst_view = BufferHandle::new(PooledBuffer::unpooled(dst_creator.view(4, 12), 12));
    let mut dst = MetalStorage::new(dst_view, m.clone(), 3, DType::U32);
    let (src, _) = t.storage_and_layout();
    let Storage::Metal(src) = &*src else {
        unreachable!()
    };
    emitted::take();
    // Element 1 of the source view to element 1 of the destination view.
    src.copy2d(&mut dst, 1, 1, 1, 1, 1, 1)?;
    let log = emitted::take();
    m.wait_until_completed()?;
    let whole: Vec<u32> = super::read_to_vec(&dst_creator, 4)?;
    assert_eq!(whole, [0, 0, u32::MAX, 0], "emitted {log:?}");
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 8,
            size: 4
        }]
    );
    Ok(())
}

/// `force_contiguous` reaches `copy_strided_src`'s contiguous blit.
#[test]
fn a_contiguous_strided_copy_reads_a_view_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let (_creator, t) = sentinel(&m)?;
    emitted::take();
    let c = t.force_contiguous()?;
    let log = emitted::take();
    assert_eq!(c.to_vec1::<u32>()?, [u32::MAX], "emitted {log:?}");
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

// --- the realistic producer: an arena slot retained past `clear_arena` ----

/// An arena's slots are views of its base (`Arena::new`), and a tensor kept
/// past `clear_arena` keeps its view while the device leaves base mode. Its
/// copies and readbacks then take the legacy blit arms.
#[test]
fn an_arena_tensor_kept_past_clear_arena_is_read_at_its_slot() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    m.clear_arena();
    non_base(&m);
    const N: usize = 1024;
    let src: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let a = Tensor::new(src.as_slice(), &dev)?;
    let step = || -> Result<(Tensor, Tensor)> {
        let b = a.affine(1.0, 0.0)?;
        let c = b.affine(1.0, 100_000.0)?;
        Ok((b, c))
    };
    // Two recorded steps: the planner tells an activation from session state
    // by comparing them (`MetalDevice::next_arena_recording_step`).
    m.begin_arena_recording();
    step()?;
    m.next_arena_recording_step();
    step()?;
    let plan = m
        .finish_arena_recording(ArenaLayout::NonAliasing)
        .expect("the recorded steps allocated");
    m.install_arena(plan, ArenaLayout::NonAliasing)?;
    m.begin_decode_step();
    let (b, c) = step()?;
    m.end_decode_step();
    let served = m.arena_counters();
    m.clear_arena();
    non_base(&m);
    {
        let (storage, _) = c.storage_and_layout();
        let Storage::Metal(ms) = &*storage else {
            unreachable!()
        };
        assert!(
            ms.buffer().is_view() && ms.buffer().base_offset() > 0,
            "fixture precondition: the kept tensor is an arena slot past the base \
             (view={}, base={}, arena {served:?})",
            ms.buffer().is_view(),
            ms.buffer().base_offset()
        );
        assert!(
            !ms.buffer().may_be_cpu_mapped(),
            "fixture precondition: a Private slot, so `to_cpu` takes its copy"
        );
    }
    let want: Vec<f32> = src.iter().map(|x| x + 100_000.0).collect();
    drop(b);

    let readbacks = super::readback_counts();
    emitted::take();
    let via_to_cpu = c.to_vec1::<f32>()?;
    let to_cpu_log = emitted::take();
    let blits = super::readback_counts().since(&readbacks).blit;
    assert_eq!(blits, 1, "precondition: `to_cpu` took its copy arm");
    assert!(
        via_to_cpu == want,
        "to_cpu: first {:?}; emitted {to_cpu_log:?}",
        &via_to_cpu[..4]
    );

    let staged: Vec<f32> = stage_readback(&c)?.finish()?;
    assert!(staged == want, "stage_readback: first {:?}", &staged[..4]);

    let dst = zeros(&dev, N, DType::F32)?;
    dst.slice_set(&c, 0, 0)?;
    let copied = dst.to_vec1::<f32>()?;
    assert!(copied == want, "copy2d: first {:?}", &copied[..4]);

    let contiguous = c.force_contiguous()?.to_vec1::<f32>()?;
    assert!(
        contiguous == want,
        "copy_strided_src: first {:?}",
        &contiguous[..4]
    );
    Ok(())
}

// --- alignment: an unaligned range takes the compute byte copy ------------

/// The issue's case: a U8 view at start 1, count 1. Source offset 1 and size 1
/// break the 4-byte rule, so no native blit may be emitted for it.
#[test]
fn an_unaligned_byte_readback_takes_the_compute_copy() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let t = Tensor::new(&[10u8, 11, 12, 13, 14, 15, 16, 17], &dev)?.narrow(0, 1, 1)?;
    emitted::take();
    let v: Vec<u8> = stage_readback(&t)?.finish()?;
    let log = emitted::take();
    assert_eq!(v, [11]);
    assert_eq!(
        log,
        [Emitted::Compute {
            source: 1,
            destination: 0,
            size: 1
        }],
        "the route, not the bytes: an unaligned native blit may still copy right"
    );
    Ok(())
}

/// Unaligned AND based: relative offsets reach the compute binding, which adds
/// the base once. A compute arm handed absolute offsets reads `4 + 4 + start`.
#[test]
fn an_unaligned_view_takes_the_compute_copy_at_its_base() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    for (start, len) in [(1, 1), (1, 3), (2, 5), (0, 3)] {
        let (_creator, t) = byte_view(&m, start, len)?;
        let want = expected_bytes(start, len);

        emitted::take();
        let staged: Vec<u8> = stage_readback(&t)?.finish()?;
        let log = emitted::take();
        assert_eq!(
            staged, want,
            "stage_readback {start}+{len}: emitted {log:?}"
        );
        assert_eq!(
            log,
            [Emitted::Compute {
                source: start,
                destination: 0,
                size: len
            }],
            "stage_readback {start}+{len}"
        );

        let crate::StagedRead::Metal(read) = t.stage_read()? else {
            unreachable!()
        };
        let log = emitted::take();
        assert_eq!(
            read.read::<u8>()?,
            want,
            "stage_read {start}+{len}: emitted {log:?}"
        );
        assert_eq!(
            log,
            [Emitted::Compute {
                source: start,
                destination: 0,
                size: len
            }],
            "stage_read {start}+{len}"
        );

        let dst = zeros(&dev, len, DType::U8)?;
        dst.slice_set(&t, 0, 0)?;
        let log = emitted::take();
        assert_eq!(
            dst.to_vec1::<u8>()?,
            want,
            "copy2d {start}+{len}: emitted {log:?}"
        );
        assert_eq!(
            log,
            [Emitted::Compute {
                source: start,
                destination: 0,
                size: len
            }],
            "copy2d {start}+{len}"
        );

        emitted::take();
        let c = t.force_contiguous()?;
        let log = emitted::take();
        assert_eq!(
            c.to_vec1::<u8>()?,
            want,
            "copy_strided_src {start}+{len}: emitted {log:?}"
        );
        assert_eq!(
            log,
            [Emitted::Compute {
                source: start,
                destination: 0,
                size: len
            }],
            "copy_strided_src {start}+{len}"
        );
    }
    Ok(())
}

/// A two-byte element at an odd element offset: byte offset 2.
#[test]
fn an_odd_half_offset_takes_the_compute_copy() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let h: Vec<half::f16> = (0..6).map(|i| half::f16::from_f32(i as f32)).collect();
    let t = Tensor::new(h.as_slice(), &dev)?.narrow(0, 1, 2)?;
    emitted::take();
    let v: Vec<half::f16> = stage_readback(&t)?.finish()?;
    let log = emitted::take();
    assert_eq!(v, &h[1..3]);
    assert_eq!(
        log,
        [Emitted::Compute {
            source: 2,
            destination: 0,
            size: 4
        }]
    );
    Ok(())
}

/// `to_cpu`'s whole-storage copy at offset 0 is unaligned when the size is:
/// three bytes of a Private storage.
#[test]
fn an_unaligned_whole_storage_to_cpu_takes_the_compute_copy() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let p = Tensor::new(&[1u8, 2, 3], &dev)?.force_contiguous()?;
    {
        let (storage, _) = p.storage_and_layout();
        let Storage::Metal(ms) = &*storage else {
            unreachable!()
        };
        assert!(
            !ms.buffer().may_be_cpu_mapped(),
            "fixture precondition: a Private storage, so `to_cpu` takes its copy"
        );
    }
    emitted::take();
    let v = p.to_vec1::<u8>()?;
    let log = emitted::take();
    assert_eq!(v, [1, 2, 3]);
    assert_eq!(
        log,
        [Emitted::Compute {
            source: 0,
            destination: 0,
            size: 3
        }]
    );
    Ok(())
}

/// The aligned control: an aligned byte range stays an ordinary blit, so the
/// route is not blanket compute.
#[test]
fn an_aligned_byte_range_stays_a_blit() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let (_creator, t) = byte_view(&m, 4, 8)?;
    emitted::take();
    let v: Vec<u8> = stage_readback(&t)?.finish()?;
    let log = emitted::take();
    assert_eq!(v, expected_bytes(4, 8));
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 8
        }]
    );
    let dst = zeros(&dev, 8, DType::U8)?;
    dst.slice_set(&t, 0, 0)?;
    let log = emitted::take();
    assert_eq!(dst.to_vec1::<u8>()?, expected_bytes(4, 8));
    assert_eq!(
        log,
        [Emitted::Blit {
            source: 8,
            destination: 0,
            size: 8
        }]
    );
    Ok(())
}

/// Empty emits nothing, and the stage still carries a marker `finish` waits
/// on.
#[test]
fn an_empty_readback_emits_nothing() -> Result<()> {
    let Some((dev, m)) = metal() else {
        return Ok(());
    };
    non_base(&m);
    let t = Tensor::new(&[1u32, 2, 3], &dev)?.narrow(0, 1, 0)?;
    emitted::take();
    let v: Vec<u32> = stage_readback(&t)?.finish()?;
    let d: Vec<u32> = stage_readback_deferred(&t)?.finish()?;
    let log = emitted::take();
    assert!(v.is_empty() && d.is_empty());
    assert_eq!(log, []);
    Ok(())
}

// --- fills: the sibling wrapper, same contract ----------------------------

/// A Q4_0 block is 18 bytes: the persistent storage's zero fill and its
/// whole-buffer read are both unaligned.
#[test]
fn an_unaligned_quantized_fill_and_read_take_the_compute_arms() -> Result<()> {
    use crate::quantized::{metal::QMetalStorage, GgmlDType};
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    emitted::take();
    let q = QMetalStorage::persistent(&m, 32, GgmlDType::Q4_0)?;
    let fill_log = emitted::take();
    assert_eq!(q.storage_size_in_bytes(), 18);
    let data = q.data()?;
    let read_log = emitted::take();
    assert_eq!(data, [0u8; 18]);
    assert_eq!(fill_log, [Emitted::ComputeFill { size: 18 }]);
    assert_eq!(
        read_log,
        [Emitted::Compute {
            source: 0,
            destination: 0,
            size: 18
        }]
    );
    Ok(())
}

/// The fill wrapper addresses a view at its base as well: an aligned fill of a
/// view at byte 4 and an unaligned one at byte 1 each touch only the view.
#[test]
fn fills_address_a_view_at_its_base() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    for (base, size, route) in [
        (4, 8, Emitted::Fill { offset: 4, size: 8 }),
        (1, 6, Emitted::ComputeFill { size: 6 }),
    ] {
        let creator = m.new_buffer_with_data(&[0xffu8; 16])?;
        let view = creator.view(base, 8);
        emitted::take();
        // SAFETY: the test owns `creator` and reads it only after the wait.
        unsafe { m.fill_bytes("fill_view_test", &view, size, 0) }?;
        let log = emitted::take();
        m.wait_until_completed()?;
        let mut want = [0xffu8; 16];
        want[base..base + size].fill(0);
        assert_eq!(
            super::read_to_vec::<u8>(&creator, 16)?,
            want,
            "emitted {log:?}"
        );
        assert_eq!(log, [route]);
    }
    Ok(())
}

/// Overlapping ranges of one buffer are refused before anything is encoded:
/// neither the blit nor the element-parallel copy kernel orders them. Disjoint
/// ranges of the same buffer stay an ordinary blit.
#[test]
fn an_overlapping_copy_within_one_buffer_is_refused() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    let creator = m.new_buffer_with_data(&[1u32, 2, 3, 4])?;
    let (src, dst) = (creator.view(0, 12), creator.view(4, 12));
    emitted::take();
    // SAFETY: the test owns `creator` and reads it only after the wait.
    let overlapping = unsafe { m.copy_bytes("overlap_test", &src, 0, &dst, 0, 8) };
    let log = emitted::take();
    assert!(
        overlapping.is_err(),
        "an overlapping copy was admitted: emitted {log:?}"
    );
    assert_eq!(log, []);
    // SAFETY: as above.
    unsafe { m.copy_bytes("overlap_test", &src, 0, &dst, 4, 4) }?;
    assert_eq!(
        emitted::take(),
        [Emitted::Blit {
            source: 0,
            destination: 8,
            size: 4
        }]
    );
    m.wait_until_completed()?;
    assert_eq!(super::read_to_vec::<u32>(&creator, 4)?, [1, 2, 1, 4]);
    Ok(())
}

// --- receipts: the routed writes name the recording that holds them --------

/// The receipt variants return the receipt of the recording that encodes the
/// write on each arm (aligned: the blit, unaligned: the compute byte arm), and
/// nothing for an empty range, which encodes nothing. The period is unbounded,
/// so no cut can fall between the probe and the write: each receipt names the
/// probe's recording at the probe's epoch, and proves its DATA once the work
/// completes.
#[test]
fn routed_byte_writes_return_the_receipt_of_their_own_recording() -> Result<()> {
    let Some((_, m)) = metal() else { return Ok(()) };
    non_base(&m);
    m.set_compute_per_buffer(usize::MAX);
    eprintln!("BYTE_RECEIPT actual_api={}", m.commands().actual_command_api());
    let src = m.new_buffer_with_data(&(0..16u8).map(|i| 100 + i).collect::<Vec<_>>())?;
    for (arm, size, fill_route, copy_route) in [
        (
            "aligned",
            8,
            Some(Emitted::Fill { offset: 0, size: 8 }),
            Some(Emitted::Blit {
                source: 0,
                destination: 0,
                size: 8,
            }),
        ),
        (
            "unaligned",
            6,
            Some(Emitted::ComputeFill { size: 6 }),
            Some(Emitted::Compute {
                source: 0,
                destination: 0,
                size: 6,
            }),
        ),
        ("empty", 0, None, None),
    ] {
        let dst = m.new_buffer_with_data(&[0xffu8; 16])?;
        for (op, route) in [("fill", fill_route), ("copy", copy_route)] {
            let (probe_epoch, probe) = m.blit_command_encoder()?.recording_receipt();
            emitted::take();
            // SAFETY: the test owns both buffers and reads them only after the wait.
            let receipt = unsafe {
                match op {
                    "fill" => m.fill_bytes_with_receipt("receipt_test", &dst, size, 0),
                    _ => m.copy_bytes_with_receipt("receipt_test", &src, 0, &dst, 0, size),
                }
            }?;
            let log = emitted::take();
            assert_eq!(
                log,
                route.into_iter().collect::<Vec<_>>(),
                "{arm} {op} route"
            );
            let Some(route) = route else {
                assert!(receipt.is_none(), "an empty {op} returned a receipt");
                continue;
            };
            let (epoch, marker) =
                receipt.unwrap_or_else(|| panic!("{arm} {op} ({route:?}) returned no receipt"));
            assert!(
                marker.names_same_recording(&probe),
                "{arm} {op} ({route:?}) receipt names another recording"
            );
            assert_eq!(epoch, probe_epoch, "{arm} {op} receipt epoch");
            m.wait_until_completed()?;
            assert!(
                marker.poll_data_ready()?,
                "{arm} {op} receipt never proves DATA"
            );
        }
        let mut want = [0xffu8; 16];
        want[..size].copy_from_slice(&(0..size as u8).map(|i| 100 + i).collect::<Vec<_>>());
        assert_eq!(super::read_to_vec::<u8>(&dst, 16)?, want, "{arm} bytes");
    }
    Ok(())
}
