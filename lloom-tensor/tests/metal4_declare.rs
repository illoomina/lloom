//! `kernels.residency.metal4`: what a Metal 4 command buffer declares
//! resident. Under a `*-diff` arm the slot's set leaves out a pool buffer (the
//! device's sets list it until its last GPU reader completes) and still lists
//! an unpooled upload (it can leave the device's sets under a running command
//! buffer); under `slots` and the `*-full` controls it lists both. The
//! `queue-*` arms put the device's set on the Metal 4 queue; the `cb-*` arms
//! mark it used by every command buffer. Every arm computes the same sum.
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Tensor};

/// What one arm did: the slot adds of an op binding an unpooled upload and a
/// pool buffer, less those of the same op binding two pool buffers (the
/// upload's own listing), the bindings left to the device's sets over both,
/// the sets used per command buffer, the device set's queue attachments.
struct Arm {
    upload_listed: u64,
    covered: u64,
    set_uses: u64,
    attachments: u64,
}

fn run(arm: &str) -> Option<Arm> {
    // Read once, before the device exists (one process per test).
    std::env::set_var("LLOOM_METAL_API", "metal4");
    std::env::set_var("LLOOM_METAL4_SLOT_TRACE", "1");
    std::env::set_var("LLOOM_RESIDENCY_METAL4", arm);
    let dev = Device::new_metal(0).ok()?;
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    if m.commands().actual_command_api() != "metal4" {
        eprintln!("METAL4_DECLARE skipped: no Metal 4 on this device");
        return None;
    }
    // 4 MiB: past the small-upload bound, so `from_vec` is an unpooled upload.
    let n = 1 << 20;
    let upload = Tensor::from_vec(vec![1f32; n], n, &dev).unwrap();
    let pooled = (&upload * 2.0).unwrap();
    let pooled2 = (&upload * 3.0).unwrap();
    m.wait_until_completed().unwrap();

    let s0 = m.metal4_slot_residency().expect("the instrument is on");
    let with_upload = (&upload + &pooled).unwrap();
    m.wait_until_completed().unwrap();
    let s1 = m.metal4_slot_residency().unwrap();
    let pooled_only = (&pooled2 + &pooled).unwrap();
    m.wait_until_completed().unwrap();
    let s2 = m.metal4_slot_residency().unwrap();
    eprintln!("METAL4_DECLARE arm={arm} s0={s0:?} s1={s1:?} s2={s2:?}");

    let sum = |t: &Tensor| {
        t.sum_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    assert_eq!(sum(&with_upload), 3.0 * n as f32, "arm {arm}");
    assert_eq!(sum(&pooled_only), 5.0 * n as f32, "arm {arm}");
    assert!(s1.commits > s0.commits && s2.commits > s1.commits, "arm {arm}");
    Some(Arm {
        upload_listed: (s1.adds - s0.adds) - (s2.adds - s1.adds),
        covered: s2.covered - s0.covered,
        set_uses: s2.set_uses - s0.set_uses,
        attachments: m.residency_set_handle().queue_attachments(),
    })
}

/// Today's declaration: the slot's set lists every binding, nothing else.
#[test]
fn slots_lists_every_binding() {
    let Some(a) = run("slots") else { return };
    assert_eq!(a.covered, 0, "slots leaves nothing to the device's sets");
    assert_eq!(
        a.upload_listed, 0,
        "the upload and a pool buffer each cost one listing"
    );
    assert_eq!(a.set_uses, 0);
    assert_eq!(a.attachments, 1, "the device set is on the Metal 3 queue only");
}

/// (a): the device set on the Metal 4 queue, the slot's set the diff.
#[test]
fn queue_diff_lists_the_upload_and_leaves_pool_buffers_to_the_queue() {
    let Some(a) = run("queue-diff") else { return };
    assert_eq!(
        a.upload_listed, 1,
        "the unpooled upload is the one binding the slot's set still lists"
    );
    assert!(
        a.covered >= 5,
        "each op's pool operands and output are left to the device set: {}",
        a.covered
    );
    assert_eq!(a.set_uses, 0);
    assert_eq!(
        a.attachments, 2,
        "the device set is on the Metal 3 queue and the Metal 4 queue"
    );
}

/// (b): each command buffer uses the device set, the slot's set the diff.
#[test]
fn cb_diff_lists_the_upload_and_uses_the_device_set_per_command_buffer() {
    let Some(a) = run("cb-diff") else { return };
    assert_eq!(a.upload_listed, 1);
    assert!(a.covered >= 5, "{}", a.covered);
    assert!(
        a.set_uses >= 2,
        "both ops' command buffers use the device set: {}",
        a.set_uses
    );
    assert_eq!(a.attachments, 1, "not on the Metal 4 queue");
}

/// The controls attach or use the device set and keep the full lists.
#[test]
fn the_full_controls_attach_and_keep_every_listing() {
    let Some(a) = run("queue-full") else { return };
    assert_eq!(a.covered, 0);
    assert_eq!(a.upload_listed, 0);
    assert_eq!(a.attachments, 2);
}

#[test]
fn the_cb_full_control_uses_the_set_and_keeps_every_listing() {
    let Some(a) = run("cb-full") else { return };
    assert_eq!(a.covered, 0);
    assert_eq!(a.upload_listed, 0);
    assert!(a.set_uses >= 2, "{}", a.set_uses);
}
