//! The forward-pass observation seam: `ForwardObserver`, `HookPoint`.
//!
//! A model's forward loop exposes named points — a layer's input residual, its
//! output, the next-token hidden — that a caller may want to read or divert:
//! DSpark taps residuals, MTP reads `h_nextn`, expert-streaming prefetch wants a
//! later layer's gate run early, a debug build logs an activation. These are one
//! abstraction, monomorphized per path (`design/docs/porting-models.md` §21.6).
//!
//! The pattern deliberately avoided is HF `output_hidden_states` /
//! `register_forward_hook`: capture-everything-select-later (an alloc per layer
//! thrown away, against §3.3a), stringly/index-typed (checkable only by
//! crashing), dynamic dispatch (an indirect call per layer, against §3.6). This
//! seam is the opposite on all three:
//!
//! - the forward is generic over an observer and `()` compiles to nothing, so the
//!   crown path pays zero (strictly better than an `is_empty()` per layer);
//! - which points are live is resolved at plan time into `[Option<slot>;
//!   n_layers]` and indexed O(1), never a `contains` scan (§3.3);
//! - a `HookPoint` is a ZST type-tag, so wiring a probe to a point the model does
//!   not expose is a trait-bound failure at compile time, never a runtime string
//!   miss.
//!
//! This is the capture-side complement to [`crate::nn::speculator::Probes`]:
//! `Probes<P>` names what a target must expose for a drafter; `ForwardObserver`
//! is how the forward loop delivers those points to whatever is observing.
//!
//! ## The one real trade (§21.6)
//!
//! `at<P>` is generic, so `ForwardObserver` is **not object-safe** — there is no
//! `dyn ForwardObserver`. That is correct here: we never want `dyn` on decode,
//! and the set of hook points is fixed at the call's monomorphization. A debugger
//! wanting runtime-arbitrary points is a separate `dyn`-based path behind a debug
//! cfg (§3.3a "instrument behind a gate"), not this seam.

use lloom_tensor::{Result, Tensor};

/// A named point in the forward pass, as a zero-sized type-tag.
///
/// The contract is a type: a consumer that reads a point declares it as `P:
/// HookPoint`, and a model that does not run that point cannot be asked for it
/// without a compile error. `Kind` disambiguates the tags at runtime where a
/// plan needs to (a bitset over the tag kinds), without ever comparing strings.
pub trait HookPoint {
    /// The tag's runtime discriminant, for the plan-time slot map. Distinct per
    /// tag; the only place a `HookPoint` is a value rather than a type.
    const KIND: HookKind;
}

/// The runtime discriminant of a [`HookPoint`]. One variant per tag, so a
/// plan-time map is an array indexed by `as usize` rather than a hash of names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookKind {
    /// The residual entering a layer.
    LayerInput,
    /// The residual leaving a layer.
    LayerOutput,
    /// The target's own next-token hidden state (`h_nextn`), for MTP.
    NextnHidden,
    /// A model's internal multi-stream residual carry — e.g. Qwen4-exp's n-wide
    /// HyperConnection state `[b, seq, n*d]`, distinct in width from the
    /// collapsed `[b, seq, d]` a `LayerOutput` tap exposes. A consumer selecting
    /// this tag is declaring it is aware of (and wants) the wide carry; the type
    /// makes a width mismatch a compile error rather than a runtime shape bug.
    HcState,
}

impl HookKind {
    /// The number of hook kinds — the width of a per-layer slot bitset.
    pub const COUNT: usize = 4;

    /// A dense index for a slot array.
    pub const fn index(self) -> usize {
        match self {
            HookKind::LayerInput => 0,
            HookKind::LayerOutput => 1,
            HookKind::NextnHidden => 2,
            HookKind::HcState => 3,
        }
    }
}

/// The residual entering a layer — one layer early relative to
/// [`LayerOutputPoint`], the tap side a DSpark drafter reads at `--tap-shift -1`.
#[derive(Debug, Clone, Copy)]
pub struct LayerInputPoint;

/// The residual leaving a layer, delivered as the input of the next — the
/// reference tap side (llama.cpp's `target_layer_ids + 1`).
#[derive(Debug, Clone, Copy)]
pub struct LayerOutputPoint;

/// The target's own next-token hidden state, the point an MTP head reads.
#[derive(Debug, Clone, Copy)]
pub struct NextnHiddenPoint;

/// A model's internal multi-stream residual carry (Qwen4-exp's n-wide
/// HyperConnection state, `[b, seq, n*d]`). Distinct type from the collapsed
/// `[b, seq, d]` points, so an n-unaware consumer cannot receive it by accident.
#[derive(Debug, Clone, Copy)]
pub struct HcStatePoint;

impl HookPoint for LayerInputPoint {
    const KIND: HookKind = HookKind::LayerInput;
}
impl HookPoint for LayerOutputPoint {
    const KIND: HookKind = HookKind::LayerOutput;
}
impl HookPoint for NextnHiddenPoint {
    const KIND: HookKind = HookKind::NextnHidden;
}
impl HookPoint for HcStatePoint {
    const KIND: HookKind = HookKind::HcState;
}

/// Something that observes named points in a forward pass.
///
/// The forward loop calls `at::<P>(layer, value)` at each point it exposes. The
/// crown path passes `&mut ()`, whose `at` inlines to nothing — zero cost, no
/// branch, no allocation. An observing caller passes `&mut C` for its collector.
///
/// A new observation family (a tap collector, an MTP reader, a prefetch-gate
/// hint, an activation logger) is a new `impl ForwardObserver`, with no
/// forward-pass edit: "swap the impl → swap the strategy" (§3.6) applied to
/// observation.
pub trait ForwardObserver {
    /// Called at hook point `P` for `layer`, with the point's `value`. The
    /// generic `P` is what makes wiring a consumer to a point a model does not
    /// expose a compile error rather than a runtime miss.
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> Result<()>;

    /// Whether this observer wants point `P` at `layer` at all, so the forward
    /// loop can skip forming a value nothing reads. Resolved at plan time by the
    /// observer (a `[Option<slot>; n_layers]` index, §21.6), never a per-layer
    /// scan. The default is `false` — the `()` observer wants nothing — so the
    /// crown path's `at` is never even reached.
    #[inline]
    fn wants<P: HookPoint>(&self, _layer: usize) -> bool {
        false
    }
}

/// The "no hooks" observer: every method inlines to nothing, so a `forward(x,
/// &mut ())` compiles to the same code a forward with no observation would.
impl ForwardObserver for () {
    #[inline]
    fn at<P: HookPoint>(&mut self, _layer: usize, _value: &Tensor) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn wants<P: HookPoint>(&self, _layer: usize) -> bool {
        false
    }
}

/// A borrowed observer is an observer, so a caller holding its collector can
/// hand `&mut C` to a forward it does not own.
impl<O: ForwardObserver + ?Sized> ForwardObserver for &mut O {
    #[inline]
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> Result<()> {
        (**self).at::<P>(layer, value)
    }

    #[inline]
    fn wants<P: HookPoint>(&self, layer: usize) -> bool {
        (**self).wants::<P>(layer)
    }
}

/// A plan-time-resolved set of live observation points: for each layer, which
/// hook kinds are wanted, as one word per layer rather than a per-layer scan.
///
/// Built once from the points a consumer declares (`Session`/`Probes` wiring),
/// indexed O(1) in the layer loop (`if plan.wants(layer, P::KIND)`), so the loop
/// never runs a `contains` over a `Vec<usize>` — the residue §21.6 removes from
/// LFM2's `for &l in &config.tap_layers`.
#[derive(Debug, Clone, Default)]
pub struct HookPlan {
    /// One bitset per layer; bit `HookKind::index()` is set when that kind is
    /// wanted at that layer. `SmallVec` is not used here because the plan is
    /// built once, off the hot path, and lives for the run.
    per_layer: Vec<u8>,
}

impl HookPlan {
    /// An empty plan for `n_layers` — nothing wanted anywhere (the crown path).
    pub fn none(n_layers: usize) -> Self {
        Self {
            per_layer: vec![0u8; n_layers],
        }
    }

    /// Mark hook `kind` wanted at `layer`. Off the hot path; called at plan time.
    pub fn want(&mut self, layer: usize, kind: HookKind) {
        if layer < self.per_layer.len() {
            self.per_layer[layer] |= 1u8 << kind.index();
        }
    }

    /// Whether `kind` is wanted at `layer` — one array index and a bit test, no
    /// scan. `false` for a layer past the plan's length.
    #[inline]
    pub fn wants(&self, layer: usize, kind: HookKind) -> bool {
        self.per_layer
            .get(layer)
            .is_some_and(|w| w & (1u8 << kind.index()) != 0)
    }

    /// Whether any point is wanted at all, so a forward with a real observer can
    /// still take the crown path when the plan is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.per_layer.iter().all(|&w| w == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `()` observer's `wants` is `const false` for every point, which is
    /// what makes the crown path compile to nothing. A type-level check: if `()`
    /// ever grew a non-trivial `wants`, this would still pass, so it is paired
    /// with the value check below.
    #[test]
    fn unit_observer_wants_nothing() {
        let obs = ();
        assert!(!obs.wants::<LayerInputPoint>(0));
        assert!(!obs.wants::<LayerOutputPoint>(3));
        assert!(!obs.wants::<NextnHiddenPoint>(47));
    }

    /// The `()` observer's `at` is `Ok(())` at every point and never touches the
    /// value — the zero-cost property stated as a runnable assertion.
    #[test]
    fn unit_observer_at_is_noop() {
        let dev = lloom_tensor::Device::Cpu;
        let t = Tensor::zeros((1, 4), lloom_tensor::DType::F32, &dev).unwrap();
        let mut obs = ();
        obs.at::<LayerInputPoint>(0, &t).unwrap();
        obs.at::<LayerOutputPoint>(1, &t).unwrap();
        obs.at::<NextnHiddenPoint>(2, &t).unwrap();
    }

    /// A plan-time slot map is a per-layer bit test, not a `contains` scan: the
    /// residue §21.6 removes. Marking layers 5 and 11 and reading back is O(1).
    #[test]
    fn hook_plan_is_indexed_not_scanned() {
        let mut plan = HookPlan::none(48);
        plan.want(5, HookKind::LayerOutput);
        plan.want(11, HookKind::LayerOutput);
        assert!(plan.wants(5, HookKind::LayerOutput));
        assert!(plan.wants(11, HookKind::LayerOutput));
        assert!(!plan.wants(5, HookKind::LayerInput));
        assert!(!plan.wants(6, HookKind::LayerOutput));
        assert!(!plan.wants(99, HookKind::LayerOutput));
        assert!(!plan.is_empty());
        assert!(HookPlan::none(48).is_empty());
    }

    /// A collector that records `LayerOutput` at chosen layers — the DSpark/MTP
    /// shape — implemented as one `impl ForwardObserver`, no forward edit.
    struct TapCollector {
        plan: HookPlan,
        hits: Vec<(usize, HookKind)>,
    }

    impl ForwardObserver for TapCollector {
        fn at<P: HookPoint>(&mut self, layer: usize, _value: &Tensor) -> Result<()> {
            if self.plan.wants(layer, P::KIND) {
                self.hits.push((layer, P::KIND));
            }
            Ok(())
        }
        fn wants<P: HookPoint>(&self, layer: usize) -> bool {
            self.plan.wants(layer, P::KIND)
        }
    }

    #[test]
    fn collector_records_only_planned_points() {
        let dev = lloom_tensor::Device::Cpu;
        let t = Tensor::zeros((1, 4), lloom_tensor::DType::F32, &dev).unwrap();
        let mut plan = HookPlan::none(4);
        plan.want(1, HookKind::LayerOutput);
        plan.want(3, HookKind::LayerOutput);
        let mut c = TapCollector {
            plan,
            hits: Vec::new(),
        };
        // A loop that offers LayerOutput at every layer records only 1 and 3.
        for l in 0..4 {
            if c.wants::<LayerOutputPoint>(l) {
                c.at::<LayerOutputPoint>(l, &t).unwrap();
            }
        }
        assert_eq!(
            c.hits,
            vec![
                (1, HookKind::LayerOutput),
                (3, HookKind::LayerOutput)
            ]
        );
    }
}
