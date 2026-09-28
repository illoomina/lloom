//! The shape campaign over the tiny Qwen4-exp model (`docs/porting-models.md`
//! §21.8, lloom #761): an input is a geometry — the `#[tiny(axis)]` fields
//! of `qwen4exp::Config` plus a workload — and an execution builds the model
//! at that geometry on the CPU and on Metal, runs the same tokens through
//! both, and compares the logits.
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
//! # The input
//!
//! Positional and total, like every decoder in [`crate::decode`]: one
//! little-endian `u16` per axis in `Config::TINY` order, then the prompt
//! length, the decode steps and a flag byte (bit 0: the dense stems as
//! `TensorF16`, the real checkpoint's kind). A missing field keeps
//! `Config::tiny()`'s value; every value is clamped to `1..=max`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use lloom::models::qwen4exp::{testing, Config, LayerKind};
use lloom::nn::model::CausalLM;
use lloom::testing::tiny::TinyRole;
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::shape_probe::{self, Event};
use lloom_tensor::{Device, Tensor};

use crate::campaign::shape_hit;
use crate::decode::Reader;

/// The largest value an axis without its own `max` may take.
pub const AXIS_CAP: usize = 512;
/// The longest prompt and the most decode steps a workload runs.
pub const MAX_PROMPT: usize = 64;
pub const MAX_DECODE: usize = 4;
/// The largest model (weight elements) an execution builds; above it the
/// geometry is skipped, not run, so one execution stays well under a second.
pub const MAX_ELEMENTS: usize = 4 << 20;

/// What a run feeds the model beside its geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workload {
    pub prompt: usize,
    pub decode: usize,
    pub f16_stems: bool,
}

/// One candidate: the axis values in `Config::TINY` order, and a workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub axes: Vec<usize>,
    pub work: Workload,
}

/// The searched axes: `(name, max)`, in `Config::TINY` order.
pub fn axes() -> Vec<(&'static str, usize)> {
    Config::TINY
        .iter()
        .filter(|f| f.role == TinyRole::Axis)
        .map(|f| (f.name, f.max.unwrap_or(AXIS_CAP)))
        .collect()
}

impl Geometry {
    /// `Config::tiny()` with a workload.
    pub fn template(work: Workload) -> Self {
        let t = Config::tiny();
        Self {
            axes: axes()
                .iter()
                .map(|(n, _)| t.tiny_axis(n).expect("an axis"))
                .collect(),
            work,
        }
    }

    pub fn decode(bytes: &[u8]) -> Self {
        let mut g = Self::template(Workload {
            prompt: 7,
            decode: 1,
            f16_stems: false,
        });
        let mut r = Reader::new(bytes);
        for (v, (_, max)) in g.axes.iter_mut().zip(axes()) {
            if r.remaining() >= 2 {
                *v = (r.u16() as usize).clamp(1, max);
            }
        }
        if r.remaining() >= 1 {
            g.work.prompt = (r.u8() as usize).clamp(1, MAX_PROMPT);
        }
        if r.remaining() >= 1 {
            g.work.decode = (r.u8() as usize).min(MAX_DECODE);
        }
        if r.remaining() >= 1 {
            g.work.f16_stems = r.u8() & 1 == 1;
        }
        g
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.axes.len() * 2 + 3);
        for v in &self.axes {
            out.extend_from_slice(&(*v as u16).to_le_bytes());
        }
        out.push(self.work.prompt as u8);
        out.push(self.work.decode as u8);
        out.push(self.work.f16_stems as u8);
        out
    }

    /// The config this geometry names, or `None` if it drops a layer kind
    /// the template has (`#[tiny(cover)]`) or is larger than
    /// [`MAX_ELEMENTS`].
    pub fn config(&self) -> Option<Config> {
        let mut cfg = Config::tiny();
        for ((name, _), v) in axes().iter().zip(&self.axes) {
            cfg.set_tiny_axis(name, *v);
        }
        cfg.tiny_derive();
        let covers = |c: &Config, k: LayerKind| c.layer_types.contains(&k);
        for k in [LayerKind::Linear, LayerKind::Full] {
            if covers(&Config::tiny(), k) && !covers(&cfg, k) {
                return None;
            }
        }
        (elements(&cfg) <= MAX_ELEMENTS).then_some(cfg)
    }

    pub fn describe(&self) -> String {
        let mut s: Vec<String> = axes()
            .iter()
            .zip(&self.axes)
            .map(|((n, _), v)| format!("{n}={v}"))
            .collect();
        s.push(format!(
            "prompt={} decode={} f16_stems={}",
            self.work.prompt, self.work.decode, self.work.f16_stems
        ));
        s.join(" ")
    }
}

/// The weight elements `testing::resident_model` builds for `cfg`.
fn elements(c: &Config) -> usize {
    let d = c.hidden_size;
    let nd = c.hc_count * d;
    let hc = 3 * (nd + c.hc_lowrank * nd * 2 + c.hc_count * nd);
    let moe = c.n_experts * 3 * c.moe_intermediate_size * d
        + 3 * c.shared_expert_intermediate_size * d
        + c.n_experts * d;
    let cc = c.conv_channels();
    let vd = c.ssm_state_size * c.ssm_dt_rank;
    let linear = cc * d + 2 * vd * d + 2 * c.ssm_dt_rank * d + cc * c.ssm_conv_kernel;
    let full = d * c.n_heads * c.head_dim * 3 + 2 * d * c.n_kv_heads * c.head_dim;
    let layers: usize = c
        .layer_types
        .iter()
        .map(|k| {
            hc + moe
                + match k {
                    LayerKind::Linear => linear,
                    LayerKind::Full => full,
                }
        })
        .sum();
    layers.saturating_add(2 * c.vocab_size * d)
}

/// How an execution ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Not run: a dropped layer kind or over the element budget.
    Skipped,
    /// The CPU reference refused (`Err`) or panicked: outside the domain.
    CpuRefused(String),
    /// Metal refused with an error where the CPU ran: a refusal, not a defect.
    MetalRefused(String),
    /// Both ran; the largest logit difference and the CPU logits' scale.
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

/// The tokens a workload feeds: the prompt, then one token per decode step.
fn tokens(vocab: usize, w: &Workload) -> (Vec<u32>, Vec<u32>) {
    let v = vocab.max(1) as u32;
    let prompt = (0..w.prompt as u32).map(|i| (i * 7 + 3) % v).collect();
    let decode = (0..w.decode as u32).map(|j| (j * 5 + 1) % v).collect();
    (prompt, decode)
}

/// The logits of the prefill's last position, then of each decode step.
fn run_model(
    cfg: &Config,
    dev: &Device,
    f16_stems: bool,
    w: &Workload,
) -> Result<Vec<Vec<f32>>, String> {
    testing::set_f16_stems(f16_stems);
    let model = testing::resident_model(cfg, dev);
    let (prompt, decode) = tokens(cfg.vocab_size, w);
    let err = |e: lloom_tensor::Error| e.to_string();
    let mut cache = model.new_cache(cfg, dev).map_err(err)?;
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev).map_err(err)?;
    let mut out = Vec::with_capacity(1 + decode.len());
    let logits = model.forward(&input, 0, &mut cache).map_err(err)?;
    out.push(
        logits
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(err)?,
    );
    let mut pos = prompt.len();
    for t in decode {
        let input = Tensor::from_vec(vec![t], (1, 1), dev).map_err(err)?;
        let logits = model.forward(&input, pos, &mut cache).map_err(err)?;
        out.push(
            logits
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(err)?,
        );
        pos += 1;
    }
    Ok(out)
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
    let s: String = msg
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .take(80)
        .collect();
    s
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

thread_local! {
    /// What the last execution reached: its probes' `(site, condition,
    /// outcome)` and the census labels, for [`last_reach`].
    static REACH: RefCell<(Vec<(String, String, bool)>, Vec<String>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
}

/// The probes and census labels the last [`execute`] reached (for the
/// corpus report).
pub fn last_reach() -> (Vec<(String, String, bool)>, Vec<String>) {
    REACH.with(|r| r.borrow().clone())
}

/// The executor: run `g` on both devices, feed the shape map and the
/// dictionary, and judge. `tol` is relative to the CPU logits' scale.
pub fn execute(g: &Geometry, cpu: &Device, gpu: &Device, tol: f32) -> Outcome {
    REACH.with(|r| *r.borrow_mut() = (Vec::new(), Vec::new()));
    STEPS.with(|s| s.borrow_mut().clear());
    let outcome = judge(g, cpu, gpu, tol);
    shape_hit(&format!("outcome|{}", outcome.class()));
    match &outcome {
        Outcome::CpuRefused(m) => shape_hit(&format!("cpu|{}", shape_of(m))),
        Outcome::MetalRefused(m) => shape_hit(&format!("metal|{}", shape_of(m))),
        _ => {}
    }
    outcome
}

fn judge(g: &Geometry, cpu: &Device, gpu: &Device, tol: f32) -> Outcome {
    let Some(cfg) = g.config() else {
        return Outcome::Skipped;
    };
    let w = g.work;
    let reference = match quietly(|| run_model(&cfg, cpu, w.f16_stems, &w)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Outcome::CpuRefused(e),
        Err(p) => return Outcome::CpuRefused(format!("panic: {p}")),
    };
    let armed = shape_probe::arm();
    profile::reset();
    let metal = quietly(|| run_model(&cfg, gpu, w.f16_stems, &w));
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
/// template's value; the flag byte by a bit flip. `false` if nothing moved.
pub fn mutate(bytes: &mut Vec<u8>, seed: u64) -> bool {
    let mut rng = seed;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut g = Geometry::decode(bytes);
    let before = g.clone();
    let names = axes();
    let n = names.len() + 3;
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
    let template = Geometry::template(before.work);
    match field {
        f if f < names.len() => {
            let v = if op == 5 { template.axes[f] } else { g.axes[f] };
            g.axes[f] = pick(v, 1, names[f].1, &mut next);
        }
        f if f == names.len() => g.work.prompt = pick(g.work.prompt, 1, MAX_PROMPT, &mut next),
        f if f == names.len() + 1 => g.work.decode = pick(g.work.decode, 0, MAX_DECODE, &mut next),
        _ => g.work.f16_stems = !g.work.f16_stems,
    }
    let moved = g != before || bytes.len() != before.encode().len();
    *bytes = g.encode();
    moved
}
