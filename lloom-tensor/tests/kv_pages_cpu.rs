//! Run the existing device-free page allocator corpus without the Metal feature.
//! The legacy module lives under kernels/metal; a CPU --lib filter selected zero
//! tests. Reuse that corpus rather than copying its allocation-history cases.
#![cfg(not(feature = "metal"))]

use lloom_tensor::kv_pages;

#[path = "../src/kernels/metal/kv_pages_tests.rs"]
mod corpus;

#[test]
fn reusable_table_staging_preserves_prefix_and_capacity() {
    let mut pool = kv_pages::KvPagePool::new(4, 16, 4, 64);
    pool.append(0, 12).unwrap();
    let mut staging = Vec::with_capacity(8);
    staging.push(99);
    let capacity = staging.capacity();
    pool.extend_table_for(0, 1, &mut staging);
    assert_eq!(staging, [99, 0, 1, 2]);
    staging.clear();
    pool.extend_table_for(0, 2, &mut staging);
    assert_eq!(staging, [0, 2]);
    pool.extend_table_for(usize::MAX, 1, &mut staging);
    assert_eq!(staging, [0, 2], "absent sequence changed staging");
    assert_eq!(staging.capacity(), capacity);
    assert_eq!(pool.table_for(0, 2), staging);
}
