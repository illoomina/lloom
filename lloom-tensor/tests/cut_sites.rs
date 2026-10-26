//! The commit attribution (`diagnostics.trace.cut_sites`, `profile::cut_sites`)
//! against commits made on purpose: two call paths, each committing a known
//! number of command buffers, must each appear as their own site with exactly
//! that count — and the instrument off must attribute nothing.
#![cfg(feature = "metal")]

use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{Device, Tensor};

fn device(cut_sites: bool) -> Device {
    let mut t = lloom_tensor::tuning::Tuning::default();
    t.instrument.profile = true;
    t.instrument.cut_sites = cut_sites;
    let _ = lloom_tensor::tuning::install(t);
    Device::new_metal(0).expect("a Metal device")
}

/// One dispatch into the open command buffer, so the next cut has work to commit.
fn encode_one(dev: &Device, v: f32) -> Tensor {
    (Tensor::ones(64, lloom_tensor::DType::F32, dev).unwrap() * v as f64).unwrap()
}

/// Commits through `Commands::flush_marker`, one per call.
#[inline(never)]
fn commit_by_marker(dev: &Device, n: usize) -> Vec<Tensor> {
    let m = dev.as_metal_device().unwrap();
    (0..n)
        .map(|i| {
            let t = encode_one(dev, i as f32);
            let marker = m.flush_marker().unwrap();
            m.wait_marker(&marker).unwrap();
            t
        })
        .collect()
}

/// Commits through `Commands::flush_and_wait`, one per call.
#[inline(never)]
fn commit_by_wait(dev: &Device, n: usize) -> Vec<Tensor> {
    let m = dev.as_metal_device().unwrap();
    (0..n)
        .map(|i| {
            let t = encode_one(dev, i as f32);
            m.wait_until_completed().unwrap();
            t
        })
        .collect()
}

fn commits_where(sites: &[profile::CutSite], frame: &str) -> u64 {
    sites
        .iter()
        .filter(|s| s.frames.iter().any(|f| f.ends_with(frame)))
        .map(|s| s.commits)
        .sum()
}

#[test]
fn each_commit_path_is_its_own_site_with_its_own_count() {
    let dev = device(true);
    assert!(
        profile::cut_sites_enabled(),
        "the instrument must be on for this test"
    );
    dev.synchronize().unwrap();
    profile::reset_cut_sites();
    let _a = commit_by_marker(&dev, 3);
    let _b = commit_by_wait(&dev, 5);
    let sites = profile::cut_sites();
    let marker = commits_where(&sites, "Commands::flush_marker");
    let wait = commits_where(&sites, "Commands::flush_and_wait");
    assert_eq!(marker, 3, "flush_marker commits: {sites:#?}");
    assert_eq!(wait, 5, "flush_and_wait commits: {sites:#?}");
    let total: u64 = sites.iter().map(|s| s.commits).sum();
    assert_eq!(
        total, 8,
        "every commit attributed, nothing else committed: {sites:#?}"
    );
    let me = std::thread::current();
    let name = me.name().unwrap_or_default();
    assert!(
        sites.iter().all(|s| s.thread == name),
        "every commit came from this thread ({name}): {sites:#?}"
    );
    // The profiler's window reset (the batched scheduler opens one per burst
    // of requests) leaves the attribution alone, so a reader differencing two
    // reads never sees it go backwards.
    profile::reset();
    let kept: u64 = profile::cut_sites().iter().map(|s| s.commits).sum();
    assert_eq!(kept, 8, "profile::reset must not clear the attribution");
    profile::reset_cut_sites();
    assert!(profile::cut_sites().is_empty(), "reset_cut_sites clears it");
}

#[test]
fn off_the_instrument_attributes_nothing() {
    let dev = device(false);
    assert!(!profile::cut_sites_enabled());
    profile::reset_cut_sites();
    let _a = commit_by_marker(&dev, 2);
    assert!(profile::cut_sites().is_empty());
}
