//! The residency router through a device (`tuning.md` §6.3n): with the sets
//! split and the transient pair on heap chunks, a pool miss its thread names
//! `Weights` lands in a weights set on a buffer of its own, an unnamed miss
//! in a heap chunk's set, and the loaded line's layout names both. Its
//! eviction leaves the set that lists it, which Metal's own count shows.
#![cfg(feature = "metal")]
use lloom_tensor::residency_class::{residency_class_scope, ResidencyClass};
use lloom_tensor::{DType, Device, Tensor};

#[test]
fn a_named_pool_miss_lands_in_its_class_set_and_leaves_it_on_eviction() {
    // Read once, before the device exists (one process per test).
    std::env::set_var("LLOOM_TRANSIENT_HEAPS", "1");
    std::env::set_var("LLOOM_RESIDENCY_SET_CAP_PCT", "1");
    std::env::set_var("LLOOM_RESIDENCY_HOT", "weights,kv");
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let set = m.residency_set_handle();
    assert!(set.policy().is_split(), "{:?}", set.policy());
    let n = 16 * 1024 * 1024 / 4;
    let w = residency_class_scope(ResidencyClass::Weights, || {
        Tensor::full(1f32, n, &dev).unwrap()
    });
    let a = Tensor::full(2f32, n, &dev).unwrap();
    m.wait_until_completed().unwrap();
    let of = |class: ResidencyClass| -> usize {
        set.layout()
            .iter()
            .filter(|s| s.class == Some(class))
            .map(|s| s.allocations)
            .sum()
    };
    assert_eq!(of(ResidencyClass::Weights), 1, "{:?}", set.layout());
    assert!(of(ResidencyClass::HeapChunk) >= 1, "{:?}", set.layout());
    let ours: usize = set.layout().iter().map(|s| s.allocations).sum();
    assert_eq!(
        set.native_allocation_counts().iter().sum::<usize>(),
        ours,
        "Metal's counts and the router's"
    );
    let line = m.residency_line();
    assert!(
        line.contains("weights 1×") && line.contains("heap "),
        "{line}"
    );
    assert_eq!(
        w.to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        n as f32
    );
    assert_eq!(
        a.sum_all().unwrap().to_scalar::<f32>().unwrap(),
        2.0 * n as f32
    );

    // Evicted: the weights buffer leaves its set, and Metal's count says so.
    drop(w);
    m.wait_until_completed().unwrap();
    m.set_free_budget(0);
    m.trim_unused_buffers();
    let _ = Tensor::zeros(1, DType::F32, &dev).unwrap();
    m.wait_until_completed().unwrap();
    assert_eq!(of(ResidencyClass::Weights), 0, "{:?}", set.layout());
    // An eviction's removal is staged and published at a cut; Metal counts
    // an allocation marked for removal until that commit. Publish whatever
    // is staged, then compare per set: the removal went to the set that
    // listed the buffer.
    set.commit_pending();
    let ours: Vec<usize> = set.layout().iter().map(|s| s.allocations).collect();
    assert_eq!(set.native_allocation_counts(), ours, "{:?}", set.layout());
    drop(a);
}
