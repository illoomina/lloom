//! A stateful fuzzer over the arena's **lifetime** path (lloom issue #494).
//!
//! # Why this target, and why the residency set is the oracle
//!
//! #489 is the defect this exists to reach: `install_arena` inserts the new
//! base into the residency set and **nothing removes the old one**. There is no
//! `impl Drop for Arena`, and the only `remove` sites are the pools'. So
//! `clear_arena` and a re-install both drop the base while the set still lists
//! it. The doc comment on `install_arena` has promised since #69 that
//! `clear_arena` *"returns the device to the classical path exactly"*.
//!
//! **It was invisible to the whole suite because nothing had ever called it**:
//! #487's probe became the first caller of `clear_arena` in the tree's history.
//! That is §15.2 #11b on a route that section does not name -- not a capability
//! nothing calls, but a **lifetime path nothing exercises**. A generator over
//! `install / clear / re-install` sequences reaches it in three operations.
//!
//! # The property, stated so it cannot pass vacuously
//!
//! **The oracle is a MARGINAL figure, and getting that wrong is what these
//! tests were re-derived for in #507.** The obvious statement -- *"installing N
//! arenas and clearing them returns the set to the length it started at"* -- is
//! what they asserted, and it is **not a property of `clear_arena`**. The
//! residency set's length is process-wide, and `install_one` moves it for two
//! independent reasons: the arena base, which the arena owns, and the recorded
//! step's own tensors, which are ordinary pool buffers retained until eviction
//! by design (§6.3f). A clear was only ever going to give back the first.
//!
//! So these compare **deltas against a control** rather than a total against a
//! baseline: what a clear recovers (exactly one, the base), and what a
//! *replacement* costs against what a first install costs (exactly one less,
//! because it hands the outgoing base back as it takes the incoming one).
//!
//! Both failed against the total on a tree where the defect was **fixed**,
//! which is §15.1a's INEXPRESSIVE class in the tests written to catch it: no
//! reachable world made them report "clean", because the pool always retains
//! something.
//!
//! # Latent today, and not staying latent
//!
//! Nothing in the shipping engine replaces an arena -- it is installed once and
//! lives for the process, so a leaked entry costs one buffer that would have
//! been retained anyway. #493's ring makes replacement **per-generation**, and
//! any per-phase arena reaches it too. Under either the leak is
//! per-replacement, and a retained buffer the pool believes it released is the
//! class §6.3c records as **panicking the machine**.

#![cfg(feature = "metal")]

use candle_core::{DType, Device, Result, Tensor};

/// Shares `metal_buffer_pool_tests.rs`'s lock discipline: `Device::new_metal(0)`
/// hands back one process-wide device, so these tests drive the same residency
/// set as everything else in the binary. Measuring a *set length* while another
/// test allocates is how a harness bug gets reported as a device bug.
static ARENA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// Records an arena plan by running one small step under the recorder, which is
/// how `install_arena` is reached through the shipping path rather than by
/// hand-building a `StepPlan`.
fn install_one(dev: &Device) -> Result<bool> {
    let metal = match dev {
        Device::Metal(m) => m,
        _ => return Ok(false),
    };
    metal.record_and_install_arena(Default::default(), || {
        let a = Tensor::zeros((8, 8), DType::F32, dev)?;
        let b = a.matmul(&a)?;
        let _ = b.sum_all()?;
        Ok(())
    })
}

/// **The #489 property.** `clear_arena` gives back what `install_arena` took.
///
/// # The oracle is the MARGINAL figure, not the set's length (#507)
///
/// This test was `#[ignore]`d as documenting an unfixed defect, and #505 fixed
/// it. It did not start passing, because the property it asserted was never
/// quite the property it named: it compared the set's length after a clear
/// against a baseline taken **before anything allocated**, and the set's length
/// is a **process-wide** quantity that `install_one` moves for two independent
/// reasons.
///
/// `install_one` runs a *recorded step*, and that step's tensors allocate
/// through `new_buffer` -> `residency_set.insert`. Those are ordinary pool
/// buffers: retained until eviction, by design (§6.3f), and no business of
/// `clear_arena`'s. So each install adds **the arena base plus the step's own
/// buffers**, and a clear was only ever going to give back the base. Measured
/// on the fixed tree, `after_install=8 -> after_clear=7`, and the old assertion
/// read that one recovered entry as three leaked ones.
///
/// That is §15.1a's **INEXPRESSIVE** class in the test written to catch the
/// defect -- no reachable world makes it report "clean", because the pool always
/// retains something -- and the repair is the instrument, not the production
/// code. **What is attributable to the arena is the DELTA across the install and
/// clear**, which is what this now measures.
#[test]
fn clear_arena_gives_back_what_install_arena_took() -> Result<()> {
    let _g = lock();
    let Some(dev) = metal_device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let metal = match &dev {
        Device::Metal(m) => m,
        _ => return Ok(()),
    };

    // Settle first: an arena from an earlier test in this binary would make the
    // figures meaningless.
    metal.clear_arena();

    // A no-arena control of the SAME SHAPE, run first, so the recorded step's
    // own allocations are warm in the pool rather than counted as the arena's.
    // This is the half the old fixture had; what it lacked was measuring the
    // arena against a bracket rather than against a fixed baseline.
    {
        let a = Tensor::zeros((8, 8), DType::F32, &dev)?;
        let b = a.matmul(&a)?;
        let _ = b.sum_all()?;
    }

    let before_install = metal.residency_set_len();
    let installed = install_one(&dev)?;
    if !installed {
        eprintln!("arena declined for this plan; nothing to measure");
        return Ok(());
    }
    let after_install = metal.residency_set_len();
    metal.clear_arena();
    let after_clear = metal.residency_set_len();

    let added = after_install as i64 - before_install as i64;
    let recovered = after_install as i64 - after_clear as i64;
    println!(
        "ARENA507 before_install={before_install} after_install={after_install} \
         after_clear={after_clear} added={added} recovered={recovered}"
    );

    // Non-vacuity: if installing did not grow the set at all, this fixture
    // cannot observe a leak and must say so rather than pass (§15.1a).
    assert!(
        added > 0,
        "install_arena did not grow the residency set ({before_install} -> \
         {after_install}): this fixture cannot observe a leak and its clean arm \
         proves nothing"
    );

    // The property, stated over the quantity the arena owns. `clear_arena`
    // removes the base and nothing else -- the step's pool buffers are the
    // pool's to retain (§6.3f), so requiring the whole `added` figure back would
    // be asserting a property of a mechanism this test does not exercise.
    assert_eq!(
        recovered, 1,
        "#489: clear_arena recovered {recovered} residency-set entries, expected \
         exactly 1 -- the arena base. `install_arena`'s doc comment has promised \
         since #69 that clear_arena \"returns the device to the classical path \
         exactly\". A 0 here is the #489 leak returning (DESIGN.md §6.7 L4, and \
         §6.3c records this class panicking the machine); anything above 1 means \
         clear_arena is unregistering buffers that are not its own, which is the \
         same defect with the sign flipped and is worse -- §6.3c's panic is what \
         a removeAllocation on a live buffer produces."
    );
    Ok(())
}

/// The same property under **replacement**, which is the shape #493's ring makes
/// per-generation.
///
/// `install_arena` replaces any previous arena, and #505 made the replacement
/// path unregister the outgoing base while the arena still owns it. So each
/// install after the first should hand back exactly one entry as it takes one,
/// and the arena's contribution to the set should be **flat in the number of
/// replacements** rather than linear.
///
/// # What this measures, and what it deliberately does not (#507)
///
/// Like its sibling above, this was `#[ignore]`d against a total and re-reads as
/// a leak the pool buffers that `install_one`'s recorded step allocates on every
/// call. Those are the pool's (§6.3f) and they grow with the number of installs
/// for a reason that has nothing to do with the arena.
///
/// **The arena-attributable quantity is the SECOND DIFFERENCE**: whatever one
/// install costs, a *replacement* must not cost more than a first install does,
/// because it gives one base back as it takes one. #504 measured the deltas at a
/// constant 4 before #505 and they are a constant 3 after it -- the arena base
/// is exactly the one that stopped accumulating.
#[test]
fn repeated_arena_replacement_does_not_accumulate_arena_entries() -> Result<()> {
    let _g = lock();
    let Some(dev) = metal_device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let metal = match &dev {
        Device::Metal(m) => m,
        _ => return Ok(()),
    };

    metal.clear_arena();

    // Warm the recorded step's own allocations first, so the first install is
    // not additionally paying for buffers every later one finds in the pool --
    // which would make the first delta incomparable with the rest.
    {
        let a = Tensor::zeros((8, 8), DType::F32, &dev)?;
        let b = a.matmul(&a)?;
        let _ = b.sum_all()?;
    }

    // **The control arm, and it is what makes the rest a comparison.** One
    // install onto an EMPTY slot: nothing is replaced, so its cost is the
    // recorded step's own allocations **plus one arena base** that stays.
    //
    // Every later install additionally *replaces*, and a correct replacement
    // path unregisters the outgoing base as it registers the incoming one -- so
    // a replacement costs exactly **one less** than a first install. That
    // difference of one IS the arena base, and it is the only quantity here
    // that belongs to `install_arena` rather than to the pool.
    //
    // Without this arm the assertion is "the deltas are equal to each other",
    // which #507 verified is passed by a tree with the replacement-path
    // unregistration reverted: the deltas go 3,3,3 -> 4,4,4 and stay **flat**.
    // A test that pins a shape without pinning its VALUE cannot see a defect
    // that preserves the shape, and this one preserves it exactly.
    let before_first = metal.residency_set_len();
    if !install_one(&dev)? {
        eprintln!("arena declined; nothing to measure");
        return Ok(());
    }
    let first_install_cost = metal.residency_set_len() as i64 - before_first as i64;

    const REPLACEMENTS: usize = 4;
    let mut lens = vec![metal.residency_set_len()];
    for _ in 0..REPLACEMENTS {
        if !install_one(&dev)? {
            eprintln!("arena declined; nothing to measure");
            return Ok(());
        }
        lens.push(metal.residency_set_len());
    }
    let before_final_clear = metal.residency_set_len();
    metal.clear_arena();
    let after_clear = metal.residency_set_len();

    let deltas: Vec<i64> = lens.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect();
    println!(
        "ARENA507 first_install_cost={first_install_cost} per_install={lens:?} \
         per_replacement_deltas={deltas:?} after_clear={after_clear} \
         replacements={REPLACEMENTS}"
    );

    // Non-vacuity: a fixture where an install costs nothing cannot tell a fixed
    // replacement path from a broken one (§15.1a).
    assert!(
        first_install_cost > 0,
        "installing onto an empty slot did not grow the residency set: this \
         fixture cannot observe a leak and its clean arm proves nothing"
    );
    assert!(
        !deltas.is_empty() && deltas.iter().all(|d| *d >= 0),
        "replacement deltas went negative ({deltas:?}) -- clear/install is \
         unregistering buffers it does not own, which is §6.3c's panic class"
    );

    // **The property, against the control**: a replacement costs exactly one
    // less than a first install, because the outgoing base is unregistered as
    // the incoming one is registered. A replacement costing the SAME as a first
    // install means the outgoing base was retained -- #489's leak, which is
    // flat and which only this comparison can see.
    let expected = first_install_cost - 1;
    for (i, d) in deltas.iter().enumerate() {
        assert_eq!(
            *d, expected,
            "#489 under replacement: replacement {i} cost {d} residency-set \
             entries where a first install costs {first_install_cost}, so a \
             correct replacement costs {expected} -- one less, because it hands \
             the outgoing arena base back as it takes the incoming one. Full \
             deltas {deltas:?}. Equality with the first install means the \
             outgoing base was RETAINED by a set whose handle has been dropped. \
             Latent while the engine installs once per process; #493's ring \
             makes replacement per-generation, and then this is per-replacement \
             growth of a set whose stale entries are what make a later \
             removeAllocation a machine panic (DESIGN.md §6.3c)."
        );
    }

    // And the final clear recovers exactly the one live base, as the
    // single-shot case asserts.
    assert_eq!(
        before_final_clear as i64 - after_clear as i64,
        1,
        "the final clear did not recover exactly the live arena base"
    );
    Ok(())
}
