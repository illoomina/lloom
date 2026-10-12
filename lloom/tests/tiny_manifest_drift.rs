//! The per-family manifests' drift test (`docs/porting-models.md` §21.8,
//! INV-1): every entry — the smallest agreeing geometry the shape campaign
//! found for a probe outcome or a census label — is re-run here, on Metal
//! against the family's CPU reference, and must still agree and still reach
//! every point it is the smallest for. A kernel route that moves, a probe
//! renamed or a check that stops being reached fails here by name, before
//! a campaign is re-run.
//!
//! The manifests are `tests/tiny_manifests/<family>.json`, written by
//! `lloom-fuzz`'s `fuzz_geometry --manifest <corpus>`; the law, the weight
//! seed, the tolerance and the tuning they were made under travel with them
//! (`hc_mix`, and the Gated DeltaNet step's form, `gdn_step_reg` — absent in
//! a manifest made before the field, whose campaign ran the loop form).
#![cfg(feature = "metal")]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::profile::{Crown, CrownExact, Profile};
use lloom::testing::tiny::{self, Geometry, TinyConfig, TinyFamily, ValueLaw, Weights};
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::shape_probe;
use lloom_tensor::Device;
use std::collections::BTreeSet;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// What one entry reached on Metal, as the campaign names points, and the
/// largest relative CPU-vs-Metal logit difference.
fn replay<F: TinyFamily, P: Profile>(
    g: &Geometry,
    weights: Weights,
    gpu: &Device,
) -> (BTreeSet<String>, f32) {
    let ckpt = g
        .checkpoint::<F>(weights)
        .unwrap_or_else(|| panic!("{}: the geometry drops a layer kind", g.describe::<F>()));
    let vocab = ckpt.cfg.axis("vocab_size").unwrap_or(1);
    let reference = F::reference::<P>(&ckpt, &Device::Cpu).expect("the CPU reference builds");
    let want = tiny::run(&reference, &ckpt.cfg, vocab, &Device::Cpu, &g.work)
        .expect("the CPU reference runs");
    let armed = shape_probe::arm();
    gpu.synchronize().unwrap();
    profile::reset();
    let model = ckpt.model::<P>(gpu).expect("the Metal model builds");
    let got = tiny::run(&model, &ckpt.cfg, vocab, gpu, &g.work).expect("the Metal model runs");
    gpu.synchronize().unwrap();
    let events = armed.take();
    let mut points: BTreeSet<String> = events
        .iter()
        .map(|e| format!("probe|{}|{}|{}", e.site, e.cond, e.ok))
        .collect();
    points.extend(
        profile::snapshot()
            .by_label
            .into_iter()
            .map(|(l, _)| format!("census|{l}")),
    );
    let mut rel = 0f32;
    assert_eq!(want.len(), got.len(), "CPU/Metal step count");
    for (c, m) in want.iter().zip(&got) {
        assert_eq!(c.len(), m.len(), "CPU/Metal logit width");
        assert!(
            c.iter().chain(m).all(|v| v.is_finite()),
            "{}: a non-finite CPU or Metal logit",
            g.describe::<F>()
        );
        let scale = c.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        let d = c
            .iter()
            .zip(m)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d.is_finite(), "{}: a non-finite logit", g.describe::<F>());
        rel = rel.max(d / scale);
    }
    (points, rel)
}

fn check<F: TinyFamily, P: Profile>(gpu: &Device, contract: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/tiny_manifests")
        .join(if contract == "fast" {
            format!("{}.json", F::NAME)
        } else {
            format!("{}-deterministic.json", F::NAME)
        });
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(m["contract"].as_str(), Some(contract), "manifest contract");
    let weights = Weights {
        law: ValueLaw::parse(m["law"].as_str().unwrap()).unwrap(),
        seed: m["weight_seed"].as_u64().unwrap(),
        ..Weights::default()
    };
    let tol = m["tol"].as_f64().unwrap() as f32;
    let entries = m["entries"].as_array().unwrap();
    assert!(!entries.is_empty(), "{}: an empty manifest", path.display());
    let mut points = 0;
    for e in entries {
        let g = Geometry::decode::<F>(&hex(e["geometry"].as_str().unwrap()));
        assert_eq!(
            g.describe::<F>(),
            e["describe"].as_str().unwrap(),
            "the geometry encoding moved"
        );
        let (reached, rel) = replay::<F, P>(&g, weights, gpu);
        assert!(
            rel <= tol,
            "{}: CPU vs Metal {rel:e} > {tol}",
            g.describe::<F>()
        );
        let missing: Vec<&str> = e["smallest_for"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .filter(|p| !reached.contains(*p))
            .collect();
        assert!(
            missing.is_empty(),
            "{}: no longer reaches {missing:?}",
            g.describe::<F>()
        );
        points += e["smallest_for"].as_array().unwrap().len();
    }
    eprintln!(
        "{}: {} entries, {points} points replayed, all reached",
        F::NAME,
        entries.len()
    );
}

/// One process, one device, one test: the census and the probes are
/// process-wide and the tuning is read once.
fn replay_manifests<P: Profile>(contract: &str) {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let mut t = lloom::tuning::Tuning::default();
    // The Qwen4-exp campaign ran with the fused HC prologue on, which is what
    // reaches `hc_mix_fused`'s checks.
    t.qwen4exp.hc_mix = true;
    let _ = lloom::tuning::install(t);
    // And with the step form its manifest records: the census names the
    // step's kernel, so the replay runs the form the campaign ran.
    let qwen = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/tiny_manifests")
        .join(if contract == "fast" {
            "qwen4exp.json"
        } else {
            "qwen4exp-deterministic.json"
        });
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(qwen).unwrap()).unwrap();
    let mut tt = lloom_tensor::tuning::Tuning::default();
    tt.from_env().unwrap();
    tt.contract.profile = contract.into();
    tt.gdn.step_reg = m["gdn_step_reg"].as_bool().unwrap_or(false);
    assert!(
        lloom_tensor::tuning::install(tt).is_ok(),
        "the kernels' tuning was read before the manifest's was installed"
    );
    let Ok(gpu) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return;
    };
    assert!(profile::enabled(), "the census must be on");
    check::<Lfm2Tiny, P>(&gpu, contract);
    check::<Qwen4ExpTiny, P>(&gpu, contract);
}

// Separate nextest processes: a regime is process-wide and immutable.
#[test]
fn every_manifest_entry_still_reaches_its_points() {
    replay_manifests::<Crown>("fast");
}

#[test]
fn deterministic_manifest_entries_still_reach_their_points() {
    replay_manifests::<CrownExact>("deterministic");
}
