//! The residency router (`tuning.md` §6.3n): each class in sets of its own
//! up to the cap, every allocation listed by exactly one native set, and
//! every `removeAllocation` sent to the set that lists it. Each assertion on
//! the router's own layout is checked against Metal's `allocationCount` of
//! each native set, an independent count (`CONTRIBUTING.md` §3.1b).
use super::{Buffer, Device, ResidencySet};
use crate::kernels::metal::ResidencyPolicy;
use crate::residency_class::ResidencyClass::{Kv, Scratch, Weights};
use crate::residency_class::{residency_class_scope, scoped_residency_class, ResidencyClass};
use objc2_metal::MTLResourceOptions;

const KIB: u64 = 1024;

fn fixture(policy: ResidencyPolicy) -> Option<(Device, ResidencySet)> {
    let device = Device::system_default()?;
    let set = ResidencySet::with_policy(&device, policy);
    // SAFETY: only asks whether a native set exists.
    unsafe { set.raw() }?;
    Some((device, set))
}

fn buffer(device: &Device, bytes: usize) -> Buffer {
    device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
        .expect("a test buffer")
}

fn allocations(set: &ResidencySet) -> Vec<usize> {
    set.layout().iter().map(|s| s.allocations).collect()
}

/// The default layout is today's: one set, whatever the class.
#[test]
fn one_set_holds_every_class_when_the_cap_is_off() {
    let Some((device, set)) = fixture(ResidencyPolicy::single()) else {
        return;
    };
    let bufs: Vec<Buffer> = (0..3).map(|_| buffer(&device, 64 * 1024)).collect();
    assert_eq!(set.add_in(&bufs[0], Weights), 1);
    assert_eq!(set.add_in(&bufs[1], Kv), 1);
    assert_eq!(set.add_in(&bufs[2], Scratch), 1);
    set.commit_pending();
    assert_eq!(allocations(&set), vec![3]);
    assert_eq!(set.native_allocation_counts(), vec![3]);
    assert_eq!(
        set.layout()[0].class,
        None,
        "an unsplit set claims no class"
    );
    assert!(
        set.layout_line().starts_with("1 set "),
        "{}",
        set.layout_line()
    );
    assert_eq!(set.remove_batch(bufs.iter()), 3);
    assert_eq!(set.native_allocation_counts(), vec![0]);
}

/// Split at a 64 KiB cap: a class fills its set to the cap and opens the
/// next; classes never share a set; a remove reaches the set that lists the
/// allocation, and Metal's counts agree with the router's at every step.
#[test]
fn a_split_layout_keeps_classes_apart_and_caps_each_set() {
    let Some((device, set)) = fixture(ResidencyPolicy::split(64 * KIB, &[Weights, Kv])) else {
        return;
    };
    // Shared buffers allocate whole 16 KiB pages: 16 KiB each, four a set.
    let w: Vec<Buffer> = (0..6).map(|_| buffer(&device, 16 * 1024)).collect();
    let k: Vec<Buffer> = (0..2).map(|_| buffer(&device, 16 * 1024)).collect();
    for b in &w {
        assert_eq!(set.add_in(b, Weights), 1);
    }
    for b in &k {
        assert_eq!(set.add_in(b, Kv), 1);
    }
    set.commit_pending();
    let layout = set.layout();
    let classes: Vec<_> = layout.iter().map(|s| s.class).collect();
    assert_eq!(
        classes,
        vec![Some(Weights), Some(Weights), Some(Kv)],
        "{layout:?}"
    );
    assert_eq!(allocations(&set), vec![4, 2, 2]);
    assert_eq!(set.native_allocation_counts(), vec![4, 2, 2]);
    assert!(layout.iter().all(|s| s.bytes <= 64 * KIB), "{layout:?}");
    assert_eq!(set.len(), 8);

    // One remove from each weight set and one KV: each leaves its own set.
    assert_eq!(set.remove_batch([&w[0], &w[5], &k[1]]), 3);
    assert_eq!(set.native_allocation_counts(), vec![3, 1, 1]);
    assert_eq!(allocations(&set), vec![3, 1, 1]);
    // A second remove is refused before it reaches Metal (§6.3c).
    assert_eq!(set.remove(&w[0]), 0);
    assert_eq!(set.native_allocation_counts(), vec![3, 1, 1]);

    // Re-added as another class, an allocation lands in that class's set:
    // its record moved with it.
    assert_eq!(set.add_in(&w[0], Kv), 1);
    set.commit_pending();
    assert_eq!(set.native_allocation_counts(), vec![3, 1, 2]);
    assert_eq!(set.remove(&w[0]), 1);
    assert_eq!(set.native_allocation_counts(), vec![3, 1, 1]);
    assert!(
        set.layout_line().contains("weights 2×") && set.layout_line().contains("kv 1×"),
        "{}",
        set.layout_line()
    );
    assert_eq!(set.remove_all(), 5);
    assert_eq!(set.native_allocation_counts(), vec![0, 0, 0]);
    assert_eq!(set.len(), 0);
}

/// At the most sets a placement goes to its class's emptiest set and is
/// counted, never refused.
#[test]
fn past_the_most_sets_an_allocation_is_placed_and_counted() {
    let mut policy = ResidencyPolicy::split(16 * KIB, &[Weights]);
    policy.max_sets = 2;
    let Some((device, set)) = fixture(policy) else {
        return;
    };
    let bufs: Vec<Buffer> = (0..4).map(|_| buffer(&device, 16 * 1024)).collect();
    for b in &bufs {
        assert_eq!(set.add_in(b, Weights), 1);
    }
    set.commit_pending();
    assert_eq!(set.layout().len(), 2);
    assert_eq!(set.overflow(), 2);
    assert_eq!(set.native_allocation_counts().iter().sum::<usize>(), 4);
    assert_eq!(set.remove_batch(bufs.iter()), 4);
    assert_eq!(set.native_allocation_counts(), vec![0, 0]);
}

/// The keep-warm asks for the hot classes' sets only; pressure ends the
/// cold ones; the last device ends every one. A cut commits each set with
/// staged changes once, and none when nothing is staged.
#[test]
fn hot_sets_are_requested_cold_sets_ended_and_a_cut_commits_each_dirty_set() {
    let Some((device, set)) = fixture(ResidencyPolicy::split(16 * KIB, &[Weights])) else {
        return;
    };
    let w = buffer(&device, 16 * 1024);
    let s1 = buffer(&device, 16 * 1024);
    let s2 = buffer(&device, 16 * 1024);
    set.add_in(&w, Weights);
    set.add_in(&s1, Scratch);
    set.add_in(&s2, Scratch);
    let before = set.counters();
    assert!(set.commit_pending());
    assert_eq!(
        set.counters().commits - before.commits,
        3,
        "three dirty sets, one commit each"
    );
    assert!(!set.commit_pending(), "nothing staged, nothing committed");
    assert_eq!(set.counters().commits - before.commits, 3);

    let hot: Vec<bool> = set.layout().iter().map(|s| s.hot).collect();
    assert_eq!(hot, vec![true, false, false]);
    let r = set.counters().requests;
    set.request_residency();
    assert_eq!(set.counters().requests - r, 1, "only the weights set");
    let e = set.counters().ends;
    set.end_residency();
    assert_eq!(set.counters().ends - e, 2, "the two scratch sets");
    set.end_all_residency();
    assert_eq!(set.counters().ends - e, 5);
    set.remove_all();
}

/// A queue the set is attached to takes every set made after it too.
#[test]
fn a_set_made_after_the_attachment_joins_the_queue() {
    let Some((device, set)) = fixture(ResidencyPolicy::split(16 * KIB, &[Weights])) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    set.attach_queue(&queue);
    assert_eq!(set.queue_attachments(), 1);
    let a = buffer(&device, 16 * 1024);
    let b = buffer(&device, 16 * 1024);
    set.add_in(&a, Weights);
    set.add_in(&b, Weights);
    assert_eq!(set.layout().len(), 2);
    assert_eq!(
        set.queue_attachments(),
        2,
        "the second set joined the queue"
    );
    set.remove_all();
}

/// The scope names the class for its thread and restores the outer one on
/// exit, unwinding included; `add` takes it.
#[test]
fn the_class_scope_nests_restores_and_reaches_add() {
    assert_eq!(scoped_residency_class(), None);
    residency_class_scope(Weights, || {
        assert_eq!(scoped_residency_class(), Some(Weights));
        residency_class_scope(Kv, || assert_eq!(scoped_residency_class(), Some(Kv)));
        assert_eq!(scoped_residency_class(), Some(Weights));
        let caught = std::panic::catch_unwind(|| {
            residency_class_scope(Scratch, || panic!("unwinds through the scope"))
        });
        assert!(caught.is_err());
        assert_eq!(scoped_residency_class(), Some(Weights));
    });
    assert_eq!(scoped_residency_class(), None);

    let Some((device, set)) = fixture(ResidencyPolicy::split(64 * KIB, &[Weights])) else {
        return;
    };
    let a = buffer(&device, 16 * 1024);
    let b = buffer(&device, 16 * 1024);
    residency_class_scope(Kv, || assert_eq!(set.add(&a), 1));
    assert_eq!(set.add(&b), 1);
    let classes: Vec<_> = set.layout().iter().map(|s| s.class).collect();
    assert_eq!(classes, vec![Some(Kv), Some(ResidencyClass::Other)]);
    set.remove_all();
}
