//! The shape campaign over a family's tiny models (`docs/porting-models.md`
//! §21.8): an input is a geometry
//! ([`lloom::testing::tiny::Geometry`] — the family's `#[tiny(axis)]` fields,
//! a workload, the stored kinds and a runtime arm) and an execution builds
//! the checkpoint it names, runs it on Metal and against the family's CPU
//! reference, and compares the logits.
//!
//! What guides the search is what the run reached, read two ways:
//!
//! - the shape probes (`lloom_tensor::shape_probe`): which annotated check
//!   was evaluated, on which condition, with which outcome;
//! - the dispatch census: which Metal kernels ran.
//!
//! Each distinct `(site, condition, outcome)` and each census label is a
//! point in the campaign's shape map ([`crate::campaign::shape_hit`]); the
//! probes' operand values and the conditions' constants feed the
//! structured mutator's dictionary ([`mutate`]).
//!
//! The objective is a CPU-vs-Metal defect: logits apart by more than the
//! tolerance, a non-finite logit on Metal only, or a panic on Metal after the
//! CPU ran clean. A geometry the CPU refuses or panics on is outside the
//! family's domain, not a defect; the outcome class is still a map point.
//!
//! The model is the family's checkpoint through its real construction — the
//! recording backend for LFM2, the GGUF loader for Qwen4-exp — so a
//! geometry runs what a checkpoint of that shape would run. The CPU side is
//! the family's reference ([`TinyFamily::reference`]): every packed tensor
//! dequantized, the activations at f32.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use lloom::testing::tiny::{
    self, Geometry, TinyConfig, TinyFamily, Weights, Workload, AXIS_CAP, MAX_DECODE, MAX_PROMPT,
};
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::shape_probe::{self, Event};
use lloom_tensor::Device;

use crate::campaign::shape_hit;

/// Select the Free profile for a Fast-contract campaign. Default false
/// preserves the historical Exact-profile campaign entry point.
pub static FAST_CONTRACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Build Metal through the f32 reference too, to distinguish served precision
/// from a defect shared by both constructions.
pub static METAL_REFERENCE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The largest model (weight elements) an execution builds; above it the
/// geometry is skipped, not run, so one execution stays well under a second.
pub const MAX_ELEMENTS: usize = 4 << 20;

/// How an execution ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Not run: a dropped layer kind or over the element budget.
    Skipped,
    /// The CPU reference refused (`Err`) or panicked: outside the domain.
    CpuRefused(String),
    /// Metal refused with an error where the CPU ran: a refusal, not a defect.
    MetalRefused(String),
    /// Both ran; the largest relative logit difference and the CPU logits' scale.
    Agree { maxdiff: f32, scale: f32 },
    /// A defect: the message says which.
    Defect(String),
}

impl Outcome {
    /// The outcome's kind, one word.
    pub fn class(&self) -> &'static str {
        match self {
            Outcome::Skipped => "skipped",
            Outcome::CpuRefused(_) => "cpu-refused",
            Outcome::MetalRefused(_) => "metal-refused",
            Outcome::Agree { .. } => "agree",
            Outcome::Defect(_) => "defect",
        }
    }
}

/// The logits of the model `make` builds on `dev`, over `w`.
fn run_model<F: TinyFamily, P: lloom::profile::Profile>(
    make: impl FnOnce(&Device) -> lloom_tensor::Result<F::Model<P>>,
    cfg: &F::Config,
    vocab: usize,
    dev: &Device,
    w: &Workload,
) -> Result<Vec<Vec<f32>>, String> {
    let err = |e: lloom_tensor::Error| e.to_string();
    let model = make(dev).map_err(err)?;
    tiny::run(&model, cfg, vocab, dev, w).map_err(err)
}

/// `f` with panics caught and the process's panic hook (LibAFL's crash
/// report, the record's dump) out of the way, so a panic is a value here.
fn quietly<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    r.map_err(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "a panic".into())
    })
}

/// Digits stripped, so a message is one map point whatever the shape.
fn shape_of(msg: &str) -> String {
    msg.chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .take(80)
        .collect()
}

thread_local! {
    /// The dictionary the structured mutator draws "set to a constant" from:
    /// every probe operand and every constant in a probed condition. One
    /// client is one thread; the mutator and the executor share it.
    static DICT: RefCell<BTreeSet<usize>> = const { RefCell::new(BTreeSet::new()) };
}
const DICT_CAP: usize = 4096;

fn learn(events: &[Event]) {
    DICT.with(|d| {
        let mut d = d.borrow_mut();
        for e in events {
            for v in e.operands.iter().map(|(_, v)| *v).chain(e.constants()) {
                if v > 0 && d.len() < DICT_CAP {
                    d.insert(v as usize);
                }
            }
        }
    });
}

/// The dictionary as it stands (for the report and the tests).
pub fn dictionary() -> Vec<usize> {
    DICT.with(|d| d.borrow().iter().copied().collect())
}

/// What one execution reached: its probes' `(site, condition, outcome)`
/// and the census labels.
pub type Reach = (Vec<(String, String, bool)>, Vec<String>);

thread_local! {
    static REACH: RefCell<Reach> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// The probes and census labels the last [`execute`] reached (for the
/// corpus report and the manifest).
pub fn last_reach() -> Reach {
    REACH.with(|r| r.borrow().clone())
}

/// The executor: run `g` on both devices, feed the shape map and the
/// dictionary, and judge. `tol` is relative to the CPU logits' scale.
pub fn execute<F: TinyFamily>(
    g: &Geometry,
    weights: Weights,
    cpu: &Device,
    gpu: &Device,
    tol: f32,
) -> Outcome {
    REACH.with(|r| *r.borrow_mut() = (Vec::new(), Vec::new()));
    STEPS.with(|s| s.borrow_mut().clear());
    let outcome = if FAST_CONTRACT.load(std::sync::atomic::Ordering::Relaxed) {
        judge::<F, lloom::profile::Crown>(g, weights, cpu, gpu, tol)
    } else {
        judge::<F, lloom::profile::CrownExact>(g, weights, cpu, gpu, tol)
    };
    shape_hit(&format!("outcome|{}", outcome.class()));
    match &outcome {
        Outcome::CpuRefused(m) => shape_hit(&format!("cpu|{}", shape_of(m))),
        Outcome::MetalRefused(m) => shape_hit(&format!("metal|{}", shape_of(m))),
        _ => {}
    }
    outcome
}

fn judge<F: TinyFamily, P: lloom::profile::Profile>(
    g: &Geometry,
    weights: Weights,
    cpu: &Device,
    gpu: &Device,
    tol: f32,
) -> Outcome {
    let Some(ckpt) = g.checkpoint::<F>(weights) else {
        return Outcome::Skipped;
    };
    match quietly(|| ckpt.schema().map(|s| s.elements())) {
        Ok(Ok(n)) if n <= MAX_ELEMENTS => {}
        Ok(Ok(_)) => return Outcome::Skipped,
        Ok(Err(e)) => return Outcome::CpuRefused(e.to_string()),
        Err(p) => return Outcome::CpuRefused(format!("panic: {p}")),
    }
    let vocab = ckpt.cfg.axis("vocab_size").unwrap_or(1);
    let w = g.work;
    let reference = match quietly(|| {
        run_model::<F, P>(|d| F::reference(&ckpt, d), &ckpt.cfg, vocab, cpu, &w)
    }) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Outcome::CpuRefused(e),
        Err(p) => return Outcome::CpuRefused(format!("panic: {p}")),
    };
    let armed = shape_probe::arm();
    profile::reset();
    let as_reference = METAL_REFERENCE.load(std::sync::atomic::Ordering::Relaxed);
    let metal = quietly(|| {
        run_model::<F, P>(
            |d| {
                if as_reference {
                    F::reference(&ckpt, d)
                } else {
                    ckpt.model::<P>(d)
                }
            },
            &ckpt.cfg,
            vocab,
            gpu,
            &w,
        )
    });
    let _ = gpu.synchronize();
    let events = armed.take();
    drop(armed);
    let census = profile::snapshot();
    let mut seen = BTreeSet::new();
    for e in &events {
        if seen.insert((e.site, e.cond, e.ok)) {
            shape_hit(&format!("probe|{}|{}|{}", e.site, e.cond, e.ok));
        }
    }
    for (label, _) in &census.by_label {
        shape_hit(&format!("census|{label}"));
    }
    REACH.with(|r| {
        *r.borrow_mut() = (
            seen.iter()
                .map(|(s, c, ok)| (s.to_string(), c.to_string(), *ok))
                .collect(),
            census.by_label.iter().map(|(l, _)| l.clone()).collect(),
        )
    });
    learn(&events);
    let got = match metal {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Outcome::MetalRefused(e),
        Err(p) => return Outcome::Defect(format!("Metal panicked where the CPU ran: {p}")),
    };
    compare(&reference, &got, tol)
}

/// One step's comparison: the largest difference, the CPU logits' scale,
/// and each device's argmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    pub maxdiff: f32,
    pub scale: f32,
    pub argmax_cpu: usize,
    pub argmax_metal: usize,
}

thread_local! {
    static STEPS: RefCell<Vec<Step>> = const { RefCell::new(Vec::new()) };
}

/// The per-step comparison of the last [`execute`] that ran both devices.
pub fn last_steps() -> Vec<Step> {
    STEPS.with(|s| s.borrow().clone())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0
}

fn compare(cpu: &[Vec<f32>], gpu: &[Vec<f32>], tol: f32) -> Outcome {
    STEPS.with(|s| {
        *s.borrow_mut() = cpu
            .iter()
            .zip(gpu)
            .map(|(c, m)| Step {
                maxdiff: c
                    .iter()
                    .zip(m)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
                scale: c.iter().fold(0f32, |a, v| a.max(v.abs())),
                argmax_cpu: argmax(c),
                argmax_metal: argmax(m),
            })
            .collect()
    });
    let mut maxdiff = 0f32;
    let mut scale = 0f32;
    for (step, (c, m)) in cpu.iter().zip(gpu).enumerate() {
        if c.len() != m.len() {
            return Outcome::Defect(format!(
                "step {step}: {} logits on the CPU, {} on Metal",
                c.len(),
                m.len()
            ));
        }
        let c_bad = c.iter().filter(|v| !v.is_finite()).count();
        let m_bad = m.iter().filter(|v| !v.is_finite()).count();
        if m_bad > c_bad {
            return Outcome::Defect(format!(
                "step {step}: {m_bad} non-finite logits on Metal, {c_bad} on the CPU"
            ));
        }
        let s = c
            .iter()
            .filter(|v| v.is_finite())
            .fold(0f32, |a, v| a.max(v.abs()))
            .max(1e-3);
        let d = c
            .iter()
            .zip(m)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if d > tol * s {
            return Outcome::Defect(format!(
                "step {step}: Metal differs from the CPU by {d:e} at scale {s:e} (tolerance {tol} × scale)"
            ));
        }
        maxdiff = maxdiff.max(d / s);
        scale = scale.max(s);
    }
    Outcome::Agree { maxdiff, scale }
}

/// The structured mutation (`campaign::set_structured_mutator`): one field
/// of the decoded geometry by ×2, ÷2, ±1, a dictionary value or the
/// template's value; a kind or the arm by one step. `false` if nothing moved.
pub fn mutate<F: TinyFamily>(bytes: &mut Vec<u8>, seed: u64) -> bool {
    let mut rng = seed;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut g = Geometry::decode::<F>(bytes);
    let before = g.clone();
    let names = F::Config::axes(AXIS_CAP);
    let n = names.len() + 5;
    let field = (next() % n as u64) as usize;
    let op = next() % 6;
    let dict = dictionary();
    let pick = |v: usize, min: usize, max: usize, next: &mut dyn FnMut() -> u64| -> usize {
        let t = match op {
            0 => v.saturating_mul(2),
            1 => v / 2,
            2 => v + 1,
            3 => v.saturating_sub(1),
            4 if !dict.is_empty() => dict[(next() % dict.len() as u64) as usize],
            _ => v,
        };
        t.clamp(min, max)
    };
    let template = Geometry::template::<F>(before.work);
    let step = |i: usize, len: usize| (i + 1) % len.max(1);
    match field {
        f if f < names.len() => {
            let v = if op == 5 { template.axes[f] } else { g.axes[f] };
            g.axes[f] = pick(v, 1, names[f].1, &mut next);
        }
        f if f == names.len() => g.work.prompt = pick(g.work.prompt, 1, MAX_PROMPT, &mut next),
        f if f == names.len() + 1 => g.work.decode = pick(g.work.decode, 0, MAX_DECODE, &mut next),
        f if f == names.len() + 2 => g.dense = step(g.dense, F::DENSE_KINDS.len()),
        f if f == names.len() + 3 => g.experts = step(g.experts, F::EXPERT_KINDS.len()),
        _ => g.arm = step(g.arm, F::ARMS.len()),
    }
    let moved = g != before || bytes.len() != before.encode().len();
    *bytes = g.encode();
    moved
}
