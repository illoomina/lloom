//! The resident load at the default memory fraction, through the loader
//! (`memory.md` §9.5q). On the tiny checkpoint it fits and loads. At scale —
//! the Flash-Next checkpoint or its synthetic of the same shape — it is
//! refused before a weight is read, with the arithmetic #749 P18 recorded
//! when the machine panicked on it (`qwen4exp-dispatches-raw/
//! p18-resident-refused.txt`): 47.03 GB = 4.12 GB of stems + 42.91 GB of
//! experts, 83.8 MB a slot.
//!
//! The admission unit tests (`loader.rs`) hold the plan's arithmetic on
//! constants; this holds the plan the headers produce, so a loader that
//! misreads a kind's size, or a checkpoint whose layout moved, fails here.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{Config, LoadOptions, DEFAULT_MEMORY_FRACTION};
use lloom::nn::memory::MemSnapshot;
use lloom::testing::fixture::Fixture;
use lloom_tensor::{Device, Result};

/// P18's predicted resident load: every tensor but the memory-mapped n-gram
/// table, in bytes (`resident_bytes` + 512 × `expert_bytes_per_slot`).
const P18_PREDICTED: u64 = 4_115_863_040 + 512 * 83_814_400;

#[test]
fn the_resident_load_is_admitted_by_its_headers() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_scale_admission", Config::tiny())? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let before = MemSnapshot::read(&dev);
    if !fx.at_scale() {
        fx.resident(&cfg, &dev, &LoadOptions::default())?;
        return Ok(());
    }
    let budget = before.working_set as f64 * DEFAULT_MEMORY_FRACTION;
    if P18_PREDICTED as f64 <= budget {
        println!(
            "SKIPPED: this machine's budget {:.2} GB holds the 47.03 GB resident load; a test does not load it",
            budget / 1e9
        );
        return Ok(());
    }
    let refused = match fx.resident(&cfg, &dev, &LoadOptions::default()) {
        Ok(_) => panic!(
            "47.03 GB resident was admitted under a {:.2} GB budget",
            budget / 1e9
        ),
        Err(e) => e.to_string(),
    };
    let after = MemSnapshot::read(&dev);
    println!("{refused}");
    for want in [
        "memory budget exceeded at load: predicted 47.03 GB",
        "= resident 4.12 GB + experts every expert, 42.91 GB (83.8 MB a slot)",
        "Nothing was loaded",
    ] {
        assert!(refused.contains(want), "the refusal lacks {want:?}");
    }
    let grew = after
        .device_allocated
        .saturating_sub(before.device_allocated);
    println!("device allocation over the refused load: +{grew} B");
    assert!(
        grew < 64 << 20,
        "the refused load allocated {grew} B on the device: it read weights before refusing"
    );
    Ok(())
}
