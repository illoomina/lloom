//! `kernels.residency.slot_trace`: the Metal 4 queue's per-slot residency
//! sets, counted and timed. A command buffer binding three distinct buffers
//! adds at least those three allocations to its slot's set, adds its
//! pipeline, and commits the set once; the adds and the commit take host
//! time, and the set's `allocatedSize` at that commit is at least the three
//! buffers' bytes. With the instrument off nothing is counted
//! (`metal4_slot_residency` is `None`).
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Tensor};

#[test]
fn the_slot_trace_counts_a_command_buffers_adds_commit_and_bytes() {
    // Read once, before the device exists (one process per test).
    std::env::set_var("LLOOM_METAL_API", "metal4");
    std::env::set_var("LLOOM_METAL4_SLOT_TRACE", "1");
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    if m.commands().actual_command_api() != "metal4" {
        eprintln!("SLOT_TRACE skipped: no Metal 4 on this device");
        return;
    }
    let n = 1 << 20;
    let a = Tensor::full(1f32, n, &dev).unwrap();
    let b = Tensor::full(2f32, n, &dev).unwrap();
    m.wait_until_completed().unwrap();
    let before = m.metal4_slot_residency().expect("the instrument is on");
    let c = (&a + &b).unwrap();
    m.wait_until_completed().unwrap();
    let after = m.metal4_slot_residency().unwrap();
    eprintln!("SLOT_TRACE api=metal4 before={before:?} after={after:?}");
    assert!(
        after.adds - before.adds >= 3,
        "a, b and the sum bound: {before:?} -> {after:?}"
    );
    assert!(after.commits > before.commits, "{before:?} -> {after:?}");
    assert!(
        after.pipeline_adds > before.pipeline_adds,
        "the sum's pipeline is listed: {before:?} -> {after:?}"
    );
    assert!(
        after.add_ns > before.add_ns,
        "the adds are timed: {before:?} -> {after:?}"
    );
    assert!(
        after.commit_ns > before.commit_ns,
        "the commit is timed: {before:?} -> {after:?}"
    );
    assert!(
        after.commit_max_ns > 0 && after.commit_max_ns <= after.commit_ns,
        "the longest commit is one of the commits: {after:?}"
    );
    assert!(
        after.last_bytes >= 3 * (n as u64) * 4,
        "the committing set holds at least the three buffers: {after:?}"
    );
    assert!(after.max_bytes >= after.last_bytes);
    assert_eq!(
        c.sum_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        3.0 * n as f32
    );
}
