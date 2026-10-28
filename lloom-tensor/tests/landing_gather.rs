//! The streamed tier's landing as one dispatch per pool per group
//! (`QTensor::write_staged_gather`, `write_with_ring_gather`): every staged
//! plane lands in its slot with the bytes the host staged, the same bytes the
//! per-plane copies (`write_staged`, one `copy2d` a plane) land, in one
//! `copy_gather_u32` launch where the per-plane path takes one launch a
//! plane. Covered on each Metal command API: many planes, a partial group
//! (slots never landed stay zero), an empty group (no launch), slot reuse (a
//! reader encoded before a second landing sees the first one's bytes), a
//! plane the gather cannot take (not a whole number of words: the per-plane
//! copies, the same bytes), and repetition (the same bytes every run).
#![cfg(feature = "metal")]

use lloom_tensor::kernels::metal::staging::StagingRing;
use lloom_tensor::kernels::metal::BufferHandle;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, MetalDevice, Tensor};

const SLOTS: usize = 256;

/// The device on `api` with the dispatch census on, or `None` when this
/// machine cannot run that API.
fn device(api: &str) -> Option<(Device, MetalDevice)> {
    // Read once, before the device exists (one process per test).
    std::env::set_var("LLOOM_METAL_API", api);
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0).ok()?;
    let m = dev.as_metal_device().unwrap().clone();
    if m.commands().actual_command_api() != api {
        eprintln!("LANDING_GATHER skipped: no {api} on this device");
        return None;
    }
    assert!(
        lloom_tensor::kernels::metal::profile::enabled(),
        "the census decides which kernels ran"
    );
    eprintln!("LANDING_GATHER api={}", m.commands().actual_command_api());
    Some((dev, m))
}

/// A zeroed pool of `SLOTS` planes of `(n, k)` elements in `dtype`, and the
/// plane's bytes.
fn pool(dev: &Device, dtype: GgmlDType, n: usize, k: usize) -> (QTensor, usize) {
    let qt = QTensor::zeros_persistent((SLOTS, n, k), dtype, dev).unwrap();
    let plane = n * k / dtype.block_size() * dtype.type_size();
    (qt, plane)
}

/// The bytes a plane of `slot` carries in landing `round`: distinct per slot
/// and per round, so a plane landed in the wrong slot or from the wrong
/// round reads differently.
fn plane_bytes(slot: usize, round: usize, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_mul(31) ^ slot.wrapping_mul(131) ^ round.wrapping_mul(17)) as u8)
        .collect()
}

/// Ring entries holding each slot's plane for `round`, as the prefetcher
/// hands them to the landing: `(offset, buffer, entry)`.
fn staged(
    m: &MetalDevice,
    ring: &mut StagingRing,
    slots: &[usize],
    plane: usize,
    round: usize,
) -> Vec<(usize, BufferHandle, Option<usize>)> {
    slots
        .iter()
        .map(|&s| {
            let (entry, buf) = ring.acquire(m, plane).unwrap();
            let bytes = plane_bytes(s, round, plane);
            // SAFETY: a Shared ring buffer of at least `plane` bytes, acquired
            // and not yet stamped: no GPU work reads it.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents(), plane) };
            (s * plane, buf, entry)
        })
        .collect()
}

/// The pool's bytes as the host expects them after `landings` (slot, round)
/// in order: zero where nothing landed.
fn expected(plane: usize, landings: &[(usize, usize)]) -> Vec<u8> {
    let mut v = vec![0u8; SLOTS * plane];
    for &(s, r) in landings {
        v[s * plane..(s + 1) * plane].copy_from_slice(&plane_bytes(s, r, plane));
    }
    v
}

fn launches(m: &MetalDevice, f: impl FnOnce()) -> Vec<(String, u64)> {
    m.wait_until_completed().unwrap();
    let window = m
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    f();
    m.wait_until_completed().unwrap();
    window.close()
}

fn count(census: &[(String, u64)], label: &str) -> u64 {
    census
        .iter()
        .filter(|(l, _)| l.contains(label))
        .map(|(_, n)| *n)
        .sum()
}

/// Land `slots` for `round` into `qt` on one path: the gather, or one
/// `write_staged` a plane.
fn land(
    m: &MetalDevice,
    qt: &mut QTensor,
    ring: &mut StagingRing,
    slots: &[usize],
    plane: usize,
    round: usize,
    gather: bool,
) {
    let planes = staged(m, ring, slots, plane, round);
    if gather {
        qt.write_staged_gather(plane, &planes, ring).unwrap();
    } else {
        for (off, buf, entry) in planes {
            qt.write_staged(off, plane, buf, ring, entry).unwrap();
        }
    }
}

/// Many planes, a partial group and an empty group, each on both paths: the
/// pools hold the host's bytes (zero where nothing landed) and each other's,
/// the gather in one launch a call where the per-plane path takes one a
/// plane.
fn many_partial_empty(api: &str) {
    let Some((dev, m)) = device(api) else { return };
    let mut ring = StagingRing::new(&m);
    // An f32 plane of 2 × 64: 512 bytes, a whole number of words.
    let (mut gathered, plane) = pool(&dev, GgmlDType::F32, 2, 64);
    let (mut per_plane, _) = pool(&dev, GgmlDType::F32, 2, 64);
    // Many: 200 planes, scattered over the pool, in no order.
    let many: Vec<usize> = (0..200).map(|i| (i * 97 + 13) % SLOTS).collect();
    // Partial: seven slots of the pool, the rest never landed.
    let partial = [3usize, 250, 17, 101, 64, 2, 199];
    for (name, slots) in [
        ("many", &many[..]),
        ("partial", &partial[..]),
        ("empty", &[][..]),
    ] {
        let census = launches(&m, || {
            land(&m, &mut gathered, &mut ring, slots, plane, 1, true)
        });
        assert_eq!(
            count(&census, "copy_gather_u32"),
            u64::from(!slots.is_empty()),
            "{api} {name}: the gather is one launch per call, none for no planes ({census:?})"
        );
        assert_eq!(
            count(&census, "copy2d_"),
            0,
            "{api} {name}: the gather took per-plane copies ({census:?})"
        );
        let census = launches(&m, || {
            land(&m, &mut per_plane, &mut ring, slots, plane, 1, false)
        });
        assert_eq!(
            count(&census, "copy2d_u32"),
            slots.len() as u64,
            "{api} {name}: the per-plane path is one copy a plane ({census:?})"
        );
    }
    let landed: Vec<(usize, usize)> = many.iter().chain(partial.iter()).map(|&s| (s, 1)).collect();
    let want = expected(plane, &landed);
    let got = gathered.data().unwrap();
    assert_eq!(got.len(), want.len());
    let wrong = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(
        wrong, 0,
        "{api}: {wrong} bytes of the gathered pool differ from the host's"
    );
    assert_eq!(
        &*got,
        &*per_plane.data().unwrap(),
        "{api}: the gathered pool is not the per-plane pool"
    );
}

#[test]
fn many_partial_empty_metal3() {
    many_partial_empty("metal3");
}

#[test]
fn many_partial_empty_metal4() {
    many_partial_empty("metal4");
}

/// Finite f32 planes for the reuse test's reader: slot- and round-dependent,
/// rounds far apart in value.
fn plane_f32(slot: usize, round: usize, len: usize) -> Vec<u8> {
    (0..len / 4)
        .flat_map(|i| {
            let v = ((i * 7 + slot * 3) % 11) as f32 * 0.25 + (round * 10) as f32;
            v.to_le_bytes()
        })
        .collect()
}

/// Land `slots` with [`plane_f32`]'s bytes for `round`.
fn land_f32(
    m: &MetalDevice,
    qt: &mut QTensor,
    ring: &mut StagingRing,
    slots: &[usize],
    plane: usize,
    round: usize,
    gather: bool,
) {
    let planes: Vec<(usize, BufferHandle, Option<usize>)> = slots
        .iter()
        .map(|&s| {
            let (entry, buf) = ring.acquire(m, plane).unwrap();
            let bytes = plane_f32(s, round, plane);
            // SAFETY: as in `staged`.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents(), plane) };
            (s * plane, buf, entry)
        })
        .collect();
    if gather {
        qt.write_staged_gather(plane, &planes, ring).unwrap();
    } else {
        for (off, buf, entry) in planes {
            qt.write_staged(off, plane, buf, ring, entry).unwrap();
        }
    }
}

/// The routed read of `slots` (one row, every slot one expert): a GPU
/// reader of the pool, encoded and not waited for.
fn routed(dev: &Device, qt: &QTensor, slots: &[usize]) -> Tensor {
    let x = Tensor::from_vec(
        (0..64)
            .map(|i| (i as f32 * 0.13).sin())
            .collect::<Vec<f32>>(),
        (1, 1, 64),
        dev,
    )
    .unwrap();
    let ids = Tensor::from_vec(
        slots.iter().map(|&s| s as u32).collect::<Vec<u32>>(),
        (1, slots.len()),
        dev,
    )
    .unwrap();
    qt.indexed_moe_forward(&x, &ids).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Slot reuse: a routed read encoded after the first landing and before the
/// second, and not waited for, computes on the first landing's bytes — the
/// same bits as the read of a pool that only ever held them — and the pool
/// ends with the second's: the gather's write is ordered after every earlier
/// reader of the pool, as the per-plane copy's is.
fn slot_reuse(api: &str) {
    let Some((dev, m)) = device(api) else { return };
    let mut ring = StagingRing::new(&m);
    let slots = [5usize, 6, 7, 40, 41];
    let (mut qt, plane) = pool(&dev, GgmlDType::F32, 2, 64);
    land_f32(&m, &mut qt, &mut ring, &slots, plane, 1, true);
    let read = routed(&dev, &qt, &slots);
    land_f32(&m, &mut qt, &mut ring, &slots, plane, 2, true);
    // The references: pools that only ever held round 1, and round 2.
    let (mut r1, _) = pool(&dev, GgmlDType::F32, 2, 64);
    land_f32(&m, &mut r1, &mut ring, &slots, plane, 1, false);
    let want1 = bits(&routed(&dev, &r1, &slots));
    let (mut r2, _) = pool(&dev, GgmlDType::F32, 2, 64);
    land_f32(&m, &mut r2, &mut ring, &slots, plane, 2, false);
    let want2 = bits(&routed(&dev, &r2, &slots));
    assert_ne!(want1, want2, "the rounds must read differently");
    assert_eq!(
        bits(&read),
        want1,
        "{api}: the reader encoded before the second landing did not compute on the first"
    );
    assert_eq!(
        bits(&routed(&dev, &qt, &slots)),
        want2,
        "{api}: the second landing did not overwrite the reused slots"
    );
    assert!(
        *qt.data().unwrap() == *r2.data().unwrap(),
        "{api}: the reused pool is not the round-2 pool"
    );
}

#[test]
fn slot_reuse_metal3() {
    slot_reuse("metal3");
}

#[test]
fn slot_reuse_metal4() {
    slot_reuse("metal4");
}

/// A plane that is not a whole number of words (Q8_0, one 34-byte block)
/// cannot take the gather's word copy: it lands through the per-plane
/// copies, with the same bytes.
fn unaligned_falls_back(api: &str) {
    let Some((dev, m)) = device(api) else { return };
    let mut ring = StagingRing::new(&m);
    let (mut qt, plane) = pool(&dev, GgmlDType::Q8_0, 1, 32);
    assert_eq!(
        plane % 4,
        2,
        "the fixture's plane must not be a whole number of words"
    );
    let slots = [1usize, 9, 100];
    let census = launches(&m, || land(&m, &mut qt, &mut ring, &slots, plane, 3, true));
    assert_eq!(count(&census, "copy_gather_u32"), 0, "{api}: {census:?}");
    assert_eq!(
        count(&census, "copy2d_u8"),
        slots.len() as u64,
        "{api}: {census:?}"
    );
    let want = expected(plane, &slots.iter().map(|&s| (s, 3)).collect::<Vec<_>>());
    assert!(
        *qt.data().unwrap() == *want,
        "{api}: the fallback landed other bytes"
    );
}

#[test]
fn unaligned_falls_back_metal3() {
    unaligned_falls_back("metal3");
}

#[test]
fn unaligned_falls_back_metal4() {
    unaligned_falls_back("metal4");
}

/// The synchronous arm (`write_with_ring_gather`): each plane filled by the
/// host into its own ring entry, the group landed in one launch — the bytes
/// `fill` wrote, in the slots asked, and the same on every one of five runs.
fn ring_fill_repeats(api: &str) {
    let Some((dev, m)) = device(api) else { return };
    let ring = StagingRing::shared(&m);
    let slots: Vec<usize> = (0..60).map(|i| (i * 37 + 5) % SLOTS).collect();
    let mut first: Option<Vec<u8>> = None;
    for run in 0..5 {
        let (mut qt, plane) = pool(&dev, GgmlDType::F32, 2, 64);
        let offsets: Vec<usize> = slots.iter().map(|&s| s * plane).collect();
        let mut reported = 0;
        let census = launches(&m, || {
            reported = qt
                .write_with_ring_gather(plane, &offsets, &ring, &mut |i, dst| {
                    dst.copy_from_slice(&plane_bytes(slots[i], 4, plane));
                    Ok(())
                })
                .unwrap();
        });
        assert_eq!(count(&census, "copy_gather_u32"), 1, "{api}: {census:?}");
        assert_eq!(
            reported, 1,
            "{api}: the launches reported are not the census's"
        );
        let got = qt.data().unwrap().into_owned();
        let want = expected(plane, &slots.iter().map(|&s| (s, 4)).collect::<Vec<_>>());
        assert!(
            got == want,
            "{api} run {run}: the pool is not what fill wrote"
        );
        match &first {
            None => first = Some(got),
            Some(f) => assert!(*f == got, "{api} run {run}: differs from run 0"),
        }
    }
}

#[test]
fn ring_fill_repeats_metal3() {
    ring_fill_repeats("metal3");
}

#[test]
fn ring_fill_repeats_metal4() {
    ring_fill_repeats("metal4");
}
