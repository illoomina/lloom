//! A served value still referenced when a later pass opens is an undeclared
//! escape (`pass-plan-design.md` §26): its slot's bytes are declined to every
//! handout while it lives, the decline is counted and its producer named, and
//! a use of it after the later pass began is reported whatever the layouts.
//!
//! Each test sets its mode: the process-wide default in a debug build is the
//! hard error, which the tests that count turn into a report.
#![cfg(feature = "metal")]
use lloom_tensor::metal_backend::{
    set_escape_check, ArenaLayout, EscapeCheck, EscapeKind, MetalDevice, PassKey, PassKind,
};
use lloom_tensor::{Device, Tensor};

const N: usize = 4096;

fn device() -> (Device, MetalDevice) {
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let m = m.clone();
    m.enable_pass_plans(ArenaLayout::Packed);
    (dev, m)
}

/// Key A: two allocations, `[N]` f32 each. `y` is the second (ordinal 1).
fn key_a() -> PassKey {
    PassKey::new(PassKind::Prefill {
        w: 8,
        fresh: false,
    })
}

/// Key B: three small allocations, so its plan's slots end below A's `y`.
fn key_b() -> PassKey {
    PassKey::new(PassKind::Decode { b: 1 })
}

fn pass_a(m: &MetalDevice, x: &Tensor, k: f64) -> Tensor {
    m.begin_pass(key_a()).unwrap();
    let a = x.affine(2.0, k).unwrap();
    let y = a.affine(1.0, 1.0).unwrap();
    drop(a);
    m.finish_pass().unwrap();
    y
}

fn pass_b(m: &MetalDevice, s: &Tensor) {
    m.begin_pass(key_b()).unwrap();
    let a = s.affine(3.0, 0.0).unwrap();
    let b = a.affine(1.0, 1.0).unwrap();
    let c = b.affine(1.0, 1.0).unwrap();
    let _ = c.to_vec1::<f32>().unwrap();
    m.finish_pass().unwrap();
}

/// A's values as the host computes them.
fn want_a(k: f64) -> Vec<f32> {
    (0..N).map(|i| 2.0 * i as f32 + k as f32 + 1.0).collect()
}

/// Run A until it is served, dropping its output each time: no escape.
fn warm_a(m: &MetalDevice, x: &Tensor) {
    for k in 0..4 {
        drop(pass_a(m, x, k as f64));
    }
    assert!(m.pass_plan_cached(&key_a()), "A must be planned before the test");
}

/// The control: a served output dropped before the next pass opens is no
/// escape, however many passes run. Its zero is what the tests below are
/// read against.
#[test]
fn an_output_dropped_before_the_next_pass_is_no_escape() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let served = m.pass_plan_counters().unwrap().served;
    for k in 10..14 {
        let y = pass_a(&m, &x, k as f64);
        assert_eq!(y.to_vec1::<f32>().unwrap(), want_a(k as f64));
    }
    let c = m.pass_plan_counters().unwrap();
    println!("{c:?}");
    assert_eq!(c.served - served, 4, "the four passes must be served");
    assert_eq!(c.undeclared_escapes, 0);
    assert_eq!(c.escapes_used, 0);
    assert_eq!(c.escapes_held_unused, 0);
    assert!(m.escape_log().unwrap().take_hits().is_empty());
}

/// A served output held into the next pass of its own key: that pass hands
/// the same slot to the same ordinal, so the slot is declined (the held value
/// stays right), counted once, and the held value named as A's ordinal 1.
/// Read after the pass, it is also a use.
#[test]
fn a_value_held_into_the_next_pass_is_declined_its_slot_and_named() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 20.0);
    let next = pass_a(&m, &x, 21.0);
    let c = m.pass_plan_counters().unwrap();
    println!("{c:?}");
    // Read only now: what the declined slot protected.
    assert_eq!(held.to_vec1::<f32>().unwrap(), want_a(20.0));
    assert_eq!(next.to_vec1::<f32>().unwrap(), want_a(21.0));
    assert_eq!(
        c.undeclared_escapes, 1,
        "exactly A's ordinal 1 overlaps the held value: one declined handout"
    );
    let log = m.escape_log().unwrap();
    assert_eq!(log.named(), vec![(key_a(), 1)]);
    let hits = log.take_hits();
    println!("{hits:#?}");
    assert_eq!(hits[0].kind, EscapeKind::Declined);
    assert_eq!(hits[0].held.key, key_a());
    assert_eq!(hits[0].held.ordinal, 1);
    assert_eq!(hits[0].by, key_a());
    assert!(
        hits.iter().any(|h| h.kind == EscapeKind::Used),
        "the read after the pass is the use: {hits:?}"
    );
}

/// The layout-independent half: A's output held across B, whose plan does
/// not reach its bytes. No handout is declined, and the read after B began is
/// still the escape. A read of the same value before B began is not.
#[test]
fn a_value_used_after_a_later_pass_is_found_without_a_layout_collision() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    let s = Tensor::arange(0f32, 64f32, &dev).unwrap();
    for k in 0..5 {
        drop(pass_a(&m, &x, k as f64));
        pass_b(&m, &s);
    }
    assert!(m.pass_plan_cached(&key_a()) && m.pass_plan_cached(&key_b()));
    let held = pass_a(&m, &x, 30.0);
    // Before any later pass: not an escape.
    assert_eq!(held.to_vec1::<f32>().unwrap(), want_a(30.0));
    let before = m.pass_plan_counters().unwrap();
    assert_eq!(before.escapes_used, 0, "a read before the next pass is not a use");
    pass_b(&m, &s);
    let after_b = m.pass_plan_counters().unwrap();
    assert_eq!(held.to_vec1::<f32>().unwrap(), want_a(30.0));
    let c = m.pass_plan_counters().unwrap();
    println!("{after_b:?}\n{c:?}");
    assert_eq!(
        after_b.undeclared_escapes, 0,
        "B's plan does not reach A's output: nothing to decline"
    );
    assert_eq!(c.escapes_used, 1, "the read after B began is the escape");
    let hits = m.escape_log().unwrap().take_hits();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].kind, EscapeKind::Used);
    assert_eq!((hits[0].held.key, hits[0].held.ordinal), (key_a(), 1));
    assert_eq!(hits[0].by, key_b());
}

/// A value held across a pass that cost it a decline, then dropped unused:
/// a hold to move before the pass, reported when the next pass opens.
#[test]
fn a_value_held_unused_is_reported_when_it_is_gone() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 40.0);
    drop(pass_a(&m, &x, 41.0));
    drop(held);
    drop(pass_a(&m, &x, 42.0));
    let c = m.pass_plan_counters().unwrap();
    println!("{c:?}");
    assert_eq!(c.undeclared_escapes, 1);
    assert_eq!(c.escapes_used, 0);
    assert_eq!(c.escapes_held_unused, 1);
    let kinds: Vec<EscapeKind> = m
        .escape_log()
        .unwrap()
        .take_hits()
        .iter()
        .map(|h| h.kind)
        .collect();
    assert_eq!(kinds, vec![EscapeKind::Declined, EscapeKind::HeldUnused]);
}

/// Under the hard error, the use is a panic naming the producer and the pass.
#[test]
#[should_panic(expected = "undeclared escape (used): ordinal 1 of Prefill")]
fn under_the_hard_error_a_use_panics_naming_the_producer() {
    set_escape_check(EscapeCheck::Error);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    let s = Tensor::arange(0f32, 64f32, &dev).unwrap();
    for k in 0..5 {
        drop(pass_a(&m, &x, k as f64));
        pass_b(&m, &s);
    }
    let held = pass_a(&m, &x, 50.0);
    pass_b(&m, &s);
    let _ = held.to_vec1::<f32>();
}

/// Under the hard error, a hold that cost a decline and was dropped unused
/// panics at the next pass's opening.
#[test]
#[should_panic(expected = "undeclared escape (held): ordinal 1 of Prefill")]
fn under_the_hard_error_a_held_value_panics_when_it_is_gone() {
    set_escape_check(EscapeCheck::Error);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 60.0);
    drop(pass_a(&m, &x, 61.0));
    drop(held);
    drop(pass_a(&m, &x, 62.0));
}

/// The count mode decides nothing a report would not: the same decline,
/// the same named producer, and no hits listed.
#[test]
fn the_count_mode_counts_and_names_without_listing() {
    set_escape_check(EscapeCheck::Count);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 70.0);
    drop(pass_a(&m, &x, 71.0));
    assert_eq!(held.to_vec1::<f32>().unwrap(), want_a(70.0));
    let c = m.pass_plan_counters().unwrap();
    assert_eq!(c.undeclared_escapes, 1);
    let log = m.escape_log().unwrap();
    assert_eq!(log.named(), vec![(key_a(), 1)]);
    assert!(log.take_hits().is_empty());
}

/// A carried value that dies inside the next pass frees its bytes there: the
/// value served on them afterwards is that pass's own, and binding it is no
/// use of the dead one.
#[test]
fn a_carried_value_dropped_inside_the_next_pass_frees_its_bytes() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 80.0);
    m.begin_pass(key_a()).unwrap();
    drop(held);
    let a = x.affine(2.0, 81.0).unwrap();
    let y = a.affine(1.0, 1.0).unwrap();
    drop(a);
    let got = y.to_vec1::<f32>().unwrap();
    m.finish_pass().unwrap();
    assert_eq!(got, want_a(81.0));
    let c = m.pass_plan_counters().unwrap();
    println!("{c:?}");
    assert_eq!(c.undeclared_escapes, 0, "the slot was served once the holder was gone");
    assert_eq!(c.escapes_used, 0, "the bind was the new value's, not the dead one's");
}

/// The site that makes a named producer is taken the next time its key
/// serves it (debug and `profile` builds).
#[test]
fn a_named_producer_is_given_its_making_site() {
    set_escape_check(EscapeCheck::Report);
    let (dev, m) = device();
    let x = Tensor::arange(0f32, N as f32, &dev).unwrap();
    warm_a(&m, &x);
    let held = pass_a(&m, &x, 90.0);
    drop(pass_a(&m, &x, 91.0));
    drop(held);
    drop(pass_a(&m, &x, 92.0));
    let producers = m.escape_log().unwrap().producers();
    println!("{producers:?}");
    assert_eq!(producers.len(), 1, "{producers:?}");
    assert_eq!(producers[0].0, (key_a(), 1));
    assert!(!producers[0].1.is_empty());
}
