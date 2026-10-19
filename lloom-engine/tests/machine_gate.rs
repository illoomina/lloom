//! The machine gate end to end (`memory.md` §9.10): every family's tiny
//! checkpoint through the shipped `lloom chat`, with the machine's available
//! memory and the registry injected (`memory.machine.available_gb`,
//! `memory.machine.registry`), so no test depends on — or exhausts — the
//! memory of the machine it runs on.
//!
//! Per family and device:
//! - with room, the load prints its predicted peak and the machine's
//!   figures, reserves, decodes, and leaves the registry empty on exit;
//! - with the injected available memory under the reserve, it is refused
//!   with the arithmetic and decodes nothing;
//! - with another process's reservation standing in the registry (held by
//!   this test, not yet allocated), the same load is refused naming it —
//!   the case two processes both seeing the memory free would race on —
//!   and admitted once that reservation is released.
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use lloom_tensor::machine::Registry;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-machine-gate-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write<F: TinyFamily>(dir: &Path) -> PathBuf {
    let mut cfg = F::template();
    cfg.set_axis("vocab_size", F::tokenizer().vocab_size());
    cfg.derive();
    TinyCheckpoint::<F>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// One `lloom chat --raw` turn against `registry` with `available_gb`
/// injected and a 4 GB reserve: (exit ok, stderr).
fn chat(model: &Path, extra: &[&str], registry: &Path, available_gb: f64) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["--raw", "-p", "Hi", "-n", "4"])
        .args(extra)
        .arg("--set")
        .arg(format!("memory.machine.registry={}", registry.display()))
        .arg("--set")
        .arg(format!("memory.machine.available_gb={available_gb}"))
        .args(["--set", "memory.machine.reserve_gb=4"])
        .output()
        .expect("lloom runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The three arms on one family and device.
fn gate_arms(family: &str, model: &Path, extra: &[&str]) {
    let registry = scratch(&format!("{family}-registry"));
    let what = format!("{family} {extra:?}");

    // Room: admitted, reported, released at exit.
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(ok, "{what}: admitted with room, but:\n{err}");
    let peak = err
        .lines()
        .find(|l| l.starts_with("[machine: peak "))
        .unwrap_or_else(|| panic!("{what}: no peak line:\n{err}"));
    assert!(peak.contains("weights"), "{what}: {peak}");
    let verdict = err
        .lines()
        .find(|l| l.starts_with("[machine: need="))
        .unwrap_or_else(|| panic!("{what}: no verdict line:\n{err}"));
    for want in ["available=100.00 GB (injected)", "reserve=4.00 GB", "fits"] {
        assert!(verdict.contains(want), "{what}: {want:?} not in {verdict}");
    }
    assert!(err.contains("[turn 1:"), "{what}: no turn:\n{err}");
    assert!(
        Registry::open(&registry)
            .unwrap()
            .entries()
            .unwrap()
            .is_empty(),
        "{what}: the reservation outlived the process"
    );

    // Under the reserve: refused with the arithmetic, nothing decoded.
    let (ok, err) = chat(model, extra, &registry, 3.0);
    assert!(
        !ok,
        "{what}: admitted with 3 GB available under a 4 GB reserve:\n{err}"
    );
    for want in [
        "the machine cannot hold this now",
        "available  3000000000 B (3.00 GB) (injected",
        "reserve    4000000000 B (4.00 GB) (memory.machine.reserve_gb)",
        "Nothing was allocated",
        "--force",
    ] {
        assert!(
            err.contains(want),
            "{what}: {want:?} not in the refusal:\n{err}"
        );
    }
    assert!(
        !err.contains("[turn 1:"),
        "{what}: decoded after a refusal:\n{err}"
    );

    // Another process's reservation, not yet allocated: 100 GB available,
    // 4 kept, 99.9 reserved by the stand-in (this test process, which holds
    // far less than the 3.9 GB that would leave room) — no room, whatever
    // the tiny peak. Refused naming the stand-in; admitted once it drops.
    let reg = Registry::open(&registry).unwrap();
    let stand_in = reg
        .reserve_unchecked("stand-in for a first load", 99_900_000_000)
        .unwrap();
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(!ok, "{what}: admitted beside a 99.9 GB reservation:\n{err}");
    let pid = format!("pid {}", std::process::id());
    for want in [
        "stand-in for a first load",
        pid.as_str(),
        "reserved 99.90 GB",
    ] {
        assert!(
            err.contains(want),
            "{what}: {want:?} not in the refusal:\n{err}"
        );
    }
    assert!(
        !err.contains("[turn 1:"),
        "{what}: decoded after a refusal:\n{err}"
    );
    drop(stand_in);
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(
        ok,
        "{what}: still refused after the stand-in released:\n{err}"
    );
    std::fs::remove_dir_all(&registry).ok();
}

#[test]
fn lfm2_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    let dir = scratch("lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    gate_arms("lfm2", &model, &["--cpu"]);
    #[cfg(feature = "metal")]
    gate_arms("lfm2", &model, &[]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qwen3_5_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    for (tag, model) in [
        ("qwen3_5", write::<Qwen3_5Tiny> as fn(&Path) -> PathBuf),
        ("qwen3_5_moe", write::<Qwen3_5MoeTiny>),
    ] {
        let dir = scratch(tag);
        let model = model(&dir);
        let tok = dir.join("tokenizer.json");
        let tok = tok.to_str().unwrap();
        gate_arms(tag, &model, &["--tokenizer", tok, "--cpu"]);
        #[cfg(feature = "metal")]
        gate_arms(tag, &model, &["--tokenizer", tok]);
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn qwen4exp_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    let dir = scratch("qwen4exp");
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    gate_arms(
        "qwen4exp",
        &model,
        &["--tokenizer", tok, "--experts", "resident", "--cpu"],
    );
    // `deterministic`: the tiny's f32 expert weights have no arm under
    // `fast`'s f16 routed activations on Metal, which is not this gate's.
    #[cfg(feature = "metal")]
    for experts in ["resident", "stream"] {
        gate_arms(
            "qwen4exp",
            &model,
            &[
                "--tokenizer",
                tok,
                "--experts",
                experts,
                "--profile",
                "deterministic",
            ],
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// With no `memory.machine.reserve_gb`, the load line names the default it
/// took: what the OS keeps from the Metal device's working set where the
/// load runs on one; on the CPU, macOS keeps max(4 GiB, 10 %) and other
/// hosts keep max(256 MiB, 5 %). A binding floor names the fraction it beat.
#[test]
fn the_default_reserve_names_its_basis_on_each_device() {
    let dir = scratch("reserve-basis");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("reserve-basis-registry");
    let total = lloom_tensor::machine::HostMemory::read()
        .expect("the host's memory")
        .total;
    // State the documented policy independently of default_reserve and its
    // formatter: deriving this oracle from either would hide a policy drift.
    let (cpu_reserve, cpu_basis) = if cfg!(target_os = "macos") {
        let fraction = total / 10;
        let floor = 4 * (1_u64 << 30);
        let basis = if fraction >= floor {
            "10 %".to_string()
        } else {
            format!("4 GiB floor; 10 % = {:.2} GB", fraction as f64 / 1e9)
        };
        (fraction.max(floor), basis)
    } else {
        let fraction = total / 20;
        let floor = 256 * (1_u64 << 20);
        let basis = if fraction >= floor {
            "host: 5 %".to_string()
        } else {
            format!("256 MiB floor; host: 5 % = {:.2} GB", fraction as f64 / 1e9)
        };
        (fraction.max(floor), basis)
    };
    let cpu_want = format!("reserve={:.2} GB ({cpu_basis})", cpu_reserve as f64 / 1e9);
    #[allow(unused_mut)]
    let mut arms = vec![(&["--cpu"][..], cpu_want.as_str())];
    #[cfg(feature = "metal")]
    arms.push((&[][..], "memsize-working_set"));
    for (extra, basis) in arms {
        let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
            .args(["chat", "--model"])
            .arg(&model)
            .args(["--raw", "-p", "Hi", "-n", "4"])
            .args(extra)
            .arg("--set")
            .arg(format!("memory.machine.registry={}", registry.display()))
            .args(["--set", "memory.machine.available_gb=100"])
            .output()
            .expect("lloom runs");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{extra:?}: not admitted:\n{err}");
        let verdict = err
            .lines()
            .find(|l| l.starts_with("[machine: need="))
            .unwrap_or_else(|| panic!("{extra:?}: no verdict line:\n{err}"));
        eprintln!("{extra:?}: {verdict}");
        assert!(
            verdict.contains(basis),
            "{extra:?}: {basis} not in {verdict}"
        );
    }
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// The refusal a small CI host met (`memory.md` §9.10b), as arithmetic: a
/// 9.7 GB host with 8.85 GB available and the 4 GiB floor kept, and two
/// sibling tiny Qwen4-exp servers' reservations standing (1.54 GB each,
/// nearly none of it held, as the CI log printed them). The third tiny
/// server was refused there: `needs 1.54 GB and the room is 1.53 GB`, the
/// family's 3 MB a position over a 512 tile. Its claim is its own geometry
/// now, and it loads; the load line names where the transient came from.
#[test]
fn qwen4exp_tiny_fits_beside_two_siblings_on_a_small_host() {
    let dir = scratch("qwen4exp-small-host");
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let registry = scratch("qwen4exp-small-host-registry");
    let reg = Registry::open(&registry).unwrap();
    let siblings: Vec<_> = (0..2)
        .map(|i| {
            reg.reserve_unchecked(
                &format!("sibling lloom qwen4exp tiny-qwen4exp {i}"),
                1_540_000_000,
            )
            .unwrap()
        })
        .collect();
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&model)
        .arg("--tokenizer")
        .arg(&tok)
        .args([
            "--raw",
            "-p",
            "Hi",
            "-n",
            "4",
            "--experts",
            "resident",
            "--cpu",
        ])
        .arg("--set")
        .arg(format!("memory.machine.registry={}", registry.display()))
        .args([
            "--set",
            "memory.machine.available_gb=8.85",
            "--set",
            "memory.machine.reserve_gb=4.294967296",
        ])
        .output()
        .expect("lloom runs");
    let err = String::from_utf8_lossy(&out.stderr);
    let lines = || err.lines().filter(|l| l.starts_with("[machine:"));
    for l in lines() {
        eprintln!("{l}");
    }
    assert!(
        out.status.success(),
        "the third tiny server was refused beside two siblings:\n{err}"
    );
    let peak = lines()
        .find(|l| l.starts_with("[machine: peak "))
        .unwrap_or_else(|| panic!("no peak line:\n{err}"));
    assert!(
        peak.contains("qwen4exp-transient (1 x 512-position forward")
            && peak.contains("eager route, from the geometry"),
        "{peak}"
    );
    drop(siblings);
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// This process's cgroup v2 memory limit as a container sees it (its group
/// at the mount's root): `(limit, usage − inactive file)`, or `None` with no
/// numeric limit. Read here rather than through the gate's own reader, so
/// the test states the limit independently of the code it checks.
fn cgroup_limit() -> Option<(u64, u64)> {
    let read = |f: &str| std::fs::read_to_string(format!("/sys/fs/cgroup/{f}")).ok();
    let limit: u64 = read("memory.max")?.trim().parse().ok()?;
    let usage: u64 = read("memory.current")?.trim().parse().ok()?;
    let inactive: u64 = read("memory.stat")?
        .lines()
        .find_map(|l| l.strip_prefix("inactive_file "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    Some((limit, usage.saturating_sub(inactive)))
}

/// Inside a control group whose memory limit is under the host's free
/// memory (a capped container), the gate reads the group's limit beside the
/// host and refuses what the group cannot hold though the host could: a
/// standing reservation takes the group's room less its tenth kept back,
/// the host keeping plenty, and the tiny load is refused. Where no such
/// limit binds there is nothing to show, and the test says so.
#[test]
fn the_machine_gate_refuses_what_the_container_cannot_hold() {
    let Some((limit, charged)) = cgroup_limit() else {
        println!("SKIPPED: no cgroup v2 memory limit at /sys/fs/cgroup here");
        return;
    };
    let dir = scratch("container");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("container-registry");
    let run = |reg: &Path| {
        Command::new(env!("CARGO_BIN_EXE_lloom"))
            .args(["chat", "--model"])
            .arg(&model)
            .args(["--raw", "-p", "Hi", "-n", "4", "--cpu"])
            .arg("--set")
            .arg(format!("memory.machine.registry={}", reg.display()))
            .output()
            .expect("lloom runs")
    };
    let out = run(&registry);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "the tiny load was refused alone:\n{err}"
    );
    for l in err.lines().filter(|l| l.starts_with("[machine:")) {
        eprintln!("alone: {l}");
    }
    // The group's room after its tenth, taken by a reservation this test
    // holds (its own footprint is subtracted from what it reserved) — plus a
    // slack well past any drift: `charged` is the live container's usage,
    // which the suite's other `lloom` processes move between this read and
    // the load below, and the gate's own subtraction rounds the footprint
    // to pages. Without the slack the second load was admitted by 2.5 MiB on
    // silverberry (3/3 runs, 2026-10-19; `ci-c5091ad2`, `ci-96c4f30a`,
    // `ci-7fa67f79`) and once on the NAS. The premise is unchanged: a
    // container that cannot hold the load refuses it.
    const SLACK: i64 = 256 << 20;
    let room = (limit - charged) as i64 - (limit / 10) as i64;
    let held = lloom_tensor::machine::footprint_of(std::process::id()).unwrap_or(0) as i64;
    let stand_in = Registry::open(&registry)
        .unwrap()
        .reserve_unchecked(
            "stand-in filling the container",
            (room + held + SLACK).max(0) as u64,
        )
        .unwrap();
    eprintln!("limit={limit} charged={charged} room={room} held={held}");
    let out = run(&registry);
    let err = String::from_utf8_lossy(&out.stderr);
    for l in err
        .lines()
        .filter(|l| l.contains("machine") || l.contains("room") || l.contains("container"))
    {
        eprintln!("filled: {l}");
    }
    assert!(
        !out.status.success(),
        "admitted past the container's limit:\n{err}"
    );
    assert!(
        err.contains("the machine cannot hold this now"),
        "no refusal:\n{err}"
    );
    assert!(
        err.contains("the container binds"),
        "the refusal does not name the container:\n{err}"
    );
    drop(stand_in);
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// `--force` loads what the machine refuses, and still records the
/// reservation for the other processes to see.
#[test]
fn force_loads_what_the_machine_refuses_and_says_so() {
    let dir = scratch("force");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("force-registry");
    let (ok, err) = chat(&model, &["--cpu", "--force"], &registry, 3.0);
    assert!(ok, "--force did not load:\n{err}");
    assert!(
        err.contains("FORCED"),
        "the line does not say it was forced:\n{err}"
    );
    assert!(err.contains("[turn 1:"), "{err}");
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// `serve` is admitted at startup through the same load: refused with the
/// arithmetic, it exits before binding the socket.
#[test]
fn serve_refuses_at_startup() {
    let dir = scratch("serve");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("serve-registry");
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["serve", "--model"])
        .arg(&model)
        .args(["--cpu", "--port", "0"])
        .arg("--set")
        .arg(format!("memory.machine.registry={}", registry.display()))
        .args([
            "--set",
            "memory.machine.available_gb=3",
            "--set",
            "memory.machine.reserve_gb=4",
        ])
        .output()
        .expect("lloom runs");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "serve started without room:\n{err}");
    assert!(
        err.contains("the machine cannot hold this now"),
        "no refusal:\n{err}"
    );
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}
