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
//! The residency set's length is the oracle: **installing N arenas and clearing
//! them must return the set to the length it started at.** A hand-written test
//! that installs once and clears once is the case the code was written for; the
//! sequence that finds the leak is any sequence with a *replacement* in it.
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

/// **The #489 property.** Install, clear, and the residency set is back where it
/// started.
///
/// Run over a *sequence* of install/clear pairs rather than one, because one
/// install and one clear is the shape the code was written for and the shape
/// the existing suite would have had if it had one. The leak is per
/// replacement.
#[test]
#[ignore = "#494/#489: documents a real, unfixed defect; un-ignore with the fix"]
fn arena_install_and_clear_returns_the_residency_set_to_its_prior_length() -> Result<()> {
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
    // baseline meaningless.
    metal.clear_arena();
    let baseline = metal.residency_set_len();

    // Isolate the arena's OWN contribution: run the same recorded step with no
    // arena installed, so whatever the tensors themselves add to the set is
    // measured rather than attributed to `install_arena`. Without this the
    // headline number is "4 entries leaked" and 3 of them may be ordinary pool
    // buffers -- a figure that cannot be defended is not a finding (§1.1).
    {
        let a = Tensor::zeros((8, 8), DType::F32, &dev)?;
        let b = a.matmul(&a)?;
        let _ = b.sum_all()?;
    }
    let after_warmup = metal.residency_set_len();
    println!("ARENA494 after_warmup_no_arena={after_warmup}");

    let installed = install_one(&dev)?;
    if !installed {
        eprintln!("arena declined for this plan; nothing to measure");
        return Ok(());
    }
    let after_install = metal.residency_set_len();
    metal.clear_arena();
    let after_clear = metal.residency_set_len();

    println!(
        "ARENA494 baseline={baseline} after_warmup={after_warmup} \
         after_install={after_install} after_clear={after_clear} \
         arena_attributable={}",
        after_install as i64 - after_warmup as i64
    );

    // Non-vacuity: if installing did not grow the set, this test cannot see a
    // leak and must say so rather than pass (§15.1a).
    assert!(
        after_install > after_warmup,
        "install_arena did not grow the residency set ({after_warmup} -> {after_install}): \
         this fixture cannot observe a leak and its clean arm proves nothing"
    );

    assert_eq!(
        after_clear, after_warmup,
        "#489: clear_arena leaked {} residency-set entr{} -- the set still lists the \
         arena base whose handle is gone. `install_arena`'s doc comment has promised \
         since #69 that clear_arena \"returns the device to the classical path exactly\". \
         DESIGN.md §6.7 L4, and §6.3c records this class panicking the machine.",
        after_clear - after_warmup,
        if after_clear - after_warmup == 1 { "y" } else { "ies" }
    );
    Ok(())
}

/// The same property under **replacement**, which is the shape #493's ring
/// makes per-generation.
///
/// `install_arena` replaces any previous arena, so N installs and one clear
/// should leave the set where it started -- N-1 replacements plus one teardown.
#[test]
#[ignore = "#494/#489: documents a real, unfixed defect; un-ignore with the fix"]
fn repeated_arena_replacement_does_not_accumulate_residency_entries() -> Result<()> {
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
    let baseline = metal.residency_set_len();

    const REPLACEMENTS: usize = 4;
    let mut lens = Vec::new();
    for _ in 0..REPLACEMENTS {
        if !install_one(&dev)? {
            eprintln!("arena declined; nothing to measure");
            return Ok(());
        }
        lens.push(metal.residency_set_len());
    }
    metal.clear_arena();
    let after_clear = metal.residency_set_len();

    println!(
        "ARENA494 baseline={baseline} per_install={lens:?} after_clear={after_clear} \
         replacements={REPLACEMENTS}"
    );

    // The MARGINAL figure is the defensible one. `install_one` runs a recorded
    // step, so each install allocates the step's own tensors as well as the
    // arena base -- attributing all of it to `install_arena` would overstate
    // the leak. What is unambiguous is that a REPLACEMENT recovers nothing:
    // the deltas between consecutive installs are constant, and every one of
    // those buffers is retained by a set whose arena handle has been dropped.
    let deltas: Vec<i64> = lens
        .windows(2)
        .map(|w| w[1] as i64 - w[0] as i64)
        .collect();
    println!("ARENA494 per_replacement_deltas={deltas:?}");
    assert!(
        lens[0] > baseline,
        "install did not grow the set; this fixture cannot observe a leak"
    );
    assert!(
        deltas.iter().all(|d| *d > 0),
        "a replacement recovered entries ({deltas:?}) -- #489 may be fixed; re-read \
         `clear_arena` and `install_arena` before trusting this test's premise"
    );
    assert_eq!(
        after_clear, baseline,
        "#489 under replacement: {} entries leaked over {REPLACEMENTS} installs and one \
         clear. Latent while the engine installs once per process; #493's ring makes \
         replacement per-generation, and then this is per-replacement growth of a set \
         whose stale entries are what makes a later removeAllocation a machine panic \
         (DESIGN.md §6.3c).",
        after_clear - baseline
    );
    Ok(())
}
