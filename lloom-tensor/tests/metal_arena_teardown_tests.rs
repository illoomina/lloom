//! What `clear_arena` unwinds, against what `install_arena` established
//! (lloom issue #489).
//!
//! `install_arena`'s doc comment has promised since #69 that removing the arena
//! "returns the device to the classical path exactly". Nothing had ever called
//! `clear_arena` -- #487's poisoning probe became its first caller in the tree's
//! history and found otherwise, and #481 forbade fixing it there.
//!
//! This is `DESIGN.md` §15.2 #11b on a route that item does not name. Not a
//! capability nothing calls, but a **lifetime path nothing exercises**: the
//! function has existed and been documented as exact since #69, and until #487
//! had never been run. §6.7 L4's second corollary is the rule it broke --
//! *"which mechanisms hold a reference to this buffer" is a question a liveness
//! change must answer exhaustively, not for the two participants that happen to
//! be in view* -- and there are three participants here, not one.
//!
//! # Why these assert on the mechanism rather than on the failure
//!
//! The residency defect's natural expression is `IOGPUGroupMemory`'s assertion,
//! which panics the machine (§6.3c) rather than failing a test, and the pool
//! defect's is silent corruption under `HazardTrackingModeUntracked` (§9.3).
//! Neither is a testable outcome, so what is tested is the state each leaves --
//! which is the same discipline §6.3d's teardown tests already follow.

#![cfg(feature = "metal")]

use lloom_tensor::kernels::metal::{plan_from_sizes, ArenaLayout};
use lloom_tensor::{DType, Device, Result, Tensor};

/// Serializes these against each other and against the pool tests.
///
/// `Device::new_metal(0)` hands back one process-wide device, so every test here
/// drives the *same* pool, residency set and arena slot. Run concurrently they
/// install arenas over each other, and the failure reads as an arena bug rather
/// than a harness one.
static ARENA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// §9.2c's measured arena, which is #68's shape and the size
/// `arena_bytes_is_the_packed_peak` pins at 69 632 B.
///
/// The fixture is the shipping arena rather than a round number for #487's
/// reason: an arm that stopped being the shipping arena would otherwise
/// silently measure something else. The total is asserted below.
const SHIPPING_SLOTS: [usize; 5] = [21504, 21504, 21504, 4096, 1024];

/// The shipping plan, installed through the shipping path.
fn shipping_plan() -> lloom_tensor::kernels::metal::StepPlan {
    let last_use: Vec<usize> = (0..SHIPPING_SLOTS.len()).collect();
    plan_from_sizes(&SHIPPING_SLOTS, &last_use, ArenaLayout::NonAliasing)
}

/// The fixture is the arena that ships, so a fixture that drifts fails here
/// rather than quietly measuring a different object (#487, §2.4).
#[test]
fn fixture_is_the_shipping_arena() {
    let plan = shipping_plan();
    assert_eq!(
        plan.arena_bytes(),
        69632,
        "the fixture is no longer §9.2c's 69 632 B arena, so every assertion \
         below is about a different object"
    );
}

/// `clear_arena` takes the arena's base back out of the residency set.
///
/// This is #489's registered claim and the defect #487 found. `install_arena`
/// inserts the base (`device.rs`, §9.2's "residency is a CPU-side fact
/// established once"); nothing removed it, so the device retained an allocation
/// the arena had dropped -- **§6.7 L4 at the arena**, the mechanism §6.3c
/// records panicking the machine.
///
/// # Why membership rather than `len()`
///
/// The set is process-wide and other tests allocate into it concurrently with
/// nothing here able to stop them, so an exact length is not a stable
/// assertion. `contains` names *this* allocation, which is the property the
/// defect is about, and it is what `remove_batch`'s own membership guard keys
/// on.
#[test]
fn clear_arena_removes_the_base_from_the_residency_set() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    metal.install_arena(shipping_plan(), ArenaLayout::NonAliasing)?;

    let base = metal
        .arena()
        .expect("install_arena just installed one")
        .base()
        .clone();
    let residency = metal.residency_set_handle();

    assert!(
        residency.contains(&base),
        "install_arena did not register the base -- the fixture is not \
         exercising the path this test is about"
    );

    metal.clear_arena();

    assert!(
        !residency.contains(&base),
        "clear_arena left the arena's base in the residency set: the device \
         still holds an allocation the arena has dropped (§6.7 L4, #489)"
    );

    Ok(())
}

/// `clear_arena` does not leave the arena's slots in the pool's free list.
///
/// **The defect #489 asked not to be assumed away, and it is the one with teeth.**
/// `Arena::new` used to build each slot handle through `BufferPool::adopt`,
/// which installs a back-reference to the pool -- so when the arena dropped,
/// every slot's `Drop` called `PoolInner::release` and the slot was parked, then
/// drained into the **free list** keyed by its size.
///
/// Those slots are `Buffer::view`s over the arena's base. `BufferPool::acquire`
/// pops by size alone and knows nothing of provenance, so after a clear the pool
/// handed arena slots back as ordinary activation buffers -- several "separate"
/// allocations that are all one dropped base at different offsets, under
/// `HazardTrackingModeUntracked` where §9.3 says there is no safety net. A
/// *silent* correctness failure, where the residency leak above is a retention.
///
/// This asserts the **install** side: the arena's slots never enter the pool at
/// all, which is §9.2a's layer 3 stated as a checkable property rather than as
/// a comment. `a_cleared_arenas_slots_are_not_handed_out_again` asserts the
/// consequence on the clear side.
///
/// Note what is *not* asserted: the pool's free-list totals. The epoch-driving
/// work a clear-side test needs is itself allocating, so a total measures the
/// test's own scaffolding -- 16512 B of it, measured, which is nothing to do
/// with the arena's 69632.
#[test]
fn an_arenas_slots_never_enter_the_pool() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Settle anything an earlier test left in flight, so the occupancy read
    // below is not attributing another test's buffers to this one.
    device.synchronize()?;

    let live = |m: &lloom_tensor::MetalDevice| {
        let (_shared, p) = m.pool_occupancy();
        (p.live_buffers, p.live_bytes)
    };

    let before = live(metal);

    metal.install_arena(shipping_plan(), ArenaLayout::NonAliasing)?;

    assert!(
        metal.arena().is_some(),
        "install_arena did not install one -- the fixture is not exercising \
         the path this test is about"
    );
    assert_eq!(
        metal.arena().expect("just installed").slots().len(),
        SHIPPING_SLOTS.len(),
        "the arena did not build its slot handles, so the assertion below would \
         hold for a reason that is not the fix"
    );

    assert_eq!(
        live(metal),
        before,
        "installing an arena moved the private pool's live accounting: its \
         slots are views over the arena's base, and a slot the pool knows \
         about is a slot the pool will hand out once the arena drops (#489)"
    );

    metal.clear_arena();
    Ok(())
}

/// After a clear, the pool never hands back a buffer aliasing the old arena.
///
/// The test above asserts the accounting; this asserts the consequence, because
/// the accounting is a proxy and the aliasing is the defect. A tensor allocated
/// at a slot's exact size after the clear must not land inside the arena's old
/// base -- which is what `acquire` popping a released slot view would produce.
///
/// The sizes are the arena's own, because a size the plan never used could not
/// collide whatever the pool did, and a test that cannot fail is not a test
/// (§15.1 #1).
#[test]
fn a_cleared_arenas_slots_are_not_handed_out_again() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    metal.install_arena(shipping_plan(), ArenaLayout::NonAliasing)?;
    // The parent `MTLBuffer`'s address. Every slot is a `Buffer::view` over it
    // and `view` clones `raw`, so a slot reports this same value -- which is
    // exactly what makes it the discriminator here, and it is the identity
    // `allocation_key` and the hazard tracking both key on.
    let base_addr = metal
        .arena()
        .expect("install_arena just installed one")
        .base()
        .raw_addr();

    // Kept alive to the end: a freed base's object address can be reused by a
    // fresh allocation (when a hold frees it inside the loop below), which
    // would read as aliasing; while the base lives, an equal address is a view
    // of it.
    let base_alive = metal
        .arena()
        .expect("install_arena just installed one")
        .base()
        .try_clone()
        .expect("a clone of the base");
    metal.clear_arena();

    // A released buffer is *parked* until the epoch it was released in
    // completes (§6.3b), so the slots do not reach a free list until some GPU
    // work retires. Without this the free list is empty and the assertions
    // below pass while proving nothing -- §15.1a's VACUOUS class, and it is how
    // the first version of this test passed on the unfixed tree.
    for _ in 0..3 {
        let t = Tensor::zeros(4096, DType::F32, &device)?;
        let u = (&t + 1.0)?;
        let _ = u.sum_all()?.to_scalar::<f32>()?;
    }
    device.synchronize()?;

    // `new_buffer`, not `Tensor::zeros`: the arena adopts into
    // **`private_buffers`** and `Tensor::zeros` allocates from the *shared*
    // pool (`allocate_zeros` -> `allocate_buffer` -> `self.buffers`). A tensor
    // therefore cannot collide with an arena slot whatever the pool does, and
    // asserting on one is a test that cannot fail (§15.1 #1).
    let mut held = Vec::new();
    for size in SHIPPING_SLOTS {
        held.push(metal.new_buffer(size / DType::F32.size_in_bytes(), DType::F32, "489")?);
    }

    for (i, b) in held.iter().enumerate() {
        assert_ne!(
            b.raw_addr(),
            base_addr,
            "allocation {i} (size {}) landed on the cleared arena's own \
             allocation at offset {}: the pool handed back a view of an arena \
             it no longer serves. `acquire` keys on size alone, so two \
             unrelated activations now alias one buffer with \
             `HazardTrackingModeUntracked` giving no safety net (§9.3, #489)",
            SHIPPING_SLOTS[i],
            b.base_offset()
        );
    }
    drop(base_alive);

    Ok(())
}

/// Installing a second arena unwinds the first, by the same route.
///
/// `install_arena`'s own comment says it replaces "any previous one", and
/// replacement is the path #483's ring and any per-phase arena (§9.2's sizing
/// discussion) would take -- at which point the leak is per-replacement rather
/// than one-off. This pins that the replacement path unwinds what it replaces
/// rather than only the explicit clear doing so.
#[test]
fn reinstalling_an_arena_unwinds_the_previous_one() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    device.synchronize()?;

    metal.install_arena(shipping_plan(), ArenaLayout::NonAliasing)?;
    let first = metal
        .arena()
        .expect("install_arena just installed one")
        .base()
        .clone();
    let residency = metal.residency_set_handle();
    assert!(residency.contains(&first));

    metal.install_arena(shipping_plan(), ArenaLayout::NonAliasing)?;
    let second = metal
        .arena()
        .expect("install_arena just installed one")
        .base()
        .clone();

    assert!(
        residency.contains(&second),
        "the replacement arena's base is not registered"
    );
    assert!(
        !residency.contains(&first),
        "installing a second arena left the first one's base in the residency \
         set -- the leak is per-replacement, which is the regime #483's ring \
         and any per-phase arena reach (#489)"
    );

    metal.clear_arena();
    Ok(())
}
