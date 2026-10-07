//! MoE FFN block behind the `ExpertResidency` seam (§20.6).
//!
//! `MoeLayer` is generic over an `ExpertResidency` strategy
//! (`nn::expert_stream`, the model-independent seam) so a `Resident` model
//! (all experts in RAM) monomorphizes to a no-loader path (§3.6 HARD RULE,
//! static dispatch — no `dyn` on decode) and `Streamed` plugs in with the same
//! forward: the strategy groups the routed rows and resolves their expert ids
//! to what the kernel reads; the dispatch and the combine are shared.
//!
//! Routing (reference `build_moe_ffn`, softmax gating): `probs = softmax(router·x)`
//! over all experts → top-k → renormalize the top-k weights → weighted sum of the
//! selected experts' SwiGLU. Plus a shared expert gated by `sigmoid(shared_gate·x)`.
//! Router top-k reuses candle `arg_sort` (§3.6a — a selection of existing blocks);
//! the selected experts' GEMVs are ONE routed dispatch over the stacked,
//! packed-quantized expert weight (`QTensor::indexed_moe_forward`, the
//! `mul_mv_id` kernels on Metal — dequant in-kernel, no per-expert slice, no
//! host readback of the ids).

use super::config::Config;
use crate::nn::ops;
use crate::profile::Profile;
use lloom_tensor::quantized::{MoeArm, QMatMul};
use lloom_tensor::{DType, Module, Result, Tensor, D};
use std::marker::PhantomData;

pub use crate::nn::expert_stream::{
    ExpertResidency, Resident, RoutedGroup, Streamed, WindowRecord,
};

/// What a device-planned layer does at its two join points: right after
/// the routed experts are planned (`after_plan`, handed the plan's record)
/// and after the routed experts and the shared expert are encoded, before
/// the combine (`at_join`) — where a pass that fetches its misses instead
/// of replaying would wait for them. `()` does nothing and compiles away.
pub trait PlanHook {
    fn after_plan(&mut self, _record: &Tensor) -> Result<()> {
        Ok(())
    }
    fn at_join(&mut self) -> Result<()> {
        Ok(())
    }
    /// The fetch inside the token (`QWEN4EXP_FETCH=inside`): this pass
    /// fetches its misses — each layer's slot map and routed experts wait
    /// on the device for a worker that loads what the router chose.
    fn fetching(&self) -> bool {
        false
    }
    /// At a fetching layer, after the router and the shared expert: hand
    /// the worker this layer's routing `ids` and encode the device's wait
    /// for it (`ExpertResidency::fetch_register`). `false` when nothing was
    /// encoded (the tier does not fetch).
    fn fetch_join<R: ExpertResidency>(&mut self, _experts: &R, _ids: &Tensor) -> Result<bool> {
        Ok(false)
    }
}

impl PlanHook for () {}

/// `QWEN4EXP_MASK=1` (read once): on a device-planned pass a plane whose
/// expert the plan found missing contributes nothing to the combine (the
/// record's flag — #842's C, the mask for an unpredicted miss): the pass
/// without that expert, defined and finite, instead of slot 0's expert
/// under its weight. The token is settled and replayed as without it.
/// Off by default: the combine kernel is the plain one.
pub fn mask_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN4EXP_MASK").is_ok_and(|v| v == "1"))
}

/// `QWEN4EXP_READY_FIRST=1` (read once): a host-planned tile's groups run
/// in readiness order — the group with the fewest experts to load first —
/// so the later groups' loads overlap the earlier groups' compute the
/// longest (`profiles-and-memory-design.md` §4.8 "the schedule covers the
/// load", §10 A6c, on the routed stage: a dense chain has one order). Each
/// row's arithmetic is the tile's whatever the order (the arm is the
/// tile's, §20.15), and the outputs are placed by row. Off, the plan's
/// order: the rows'.
pub fn ready_first_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN4EXP_READY_FIRST").is_ok_and(|v| v == "1"))
}

/// Tiles whose groups ran in an order other than the rows' (the arm's
/// engagement, for its tests).
static READY_REORDERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn ready_reorders() -> u64 {
    READY_REORDERS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Combines run with the mask bound (the arm's engagement, for its tests).
static MASKED_COMBINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn masked_combines() -> u64 {
    MASKED_COMBINES.load(std::sync::atomic::Ordering::Relaxed)
}

/// How a speculative window's routed experts are planned
/// ([`MoeLayer::decode_window`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowPlanKind {
    /// On the device: no readback at the layer; the record is read after
    /// the pass.
    Device,
    /// On the host: the ids read back at the layer; the first `real_rows`
    /// rows loaded and counted as used, the rows up to `loaded_rows` loaded
    /// and counted only if accepted, the rest — the drafts — resolved
    /// against what is resident, a miss flagged.
    Host {
        real_rows: usize,
        loaded_rows: usize,
    },
}

/// A window layer's plan record ([`MoeLayer::decode_window`]): a device
/// plan's `[3·n]` tensor, read back after the pass, or a host plan's
/// record already on the host.
#[derive(Debug)]
pub enum WindowPlan {
    Device(Tensor),
    Host(WindowRecord),
}

impl WindowPlan {
    /// A device plan's record tensor; `None` for a host plan's.
    pub fn into_device(self) -> Option<Tensor> {
        match self {
            WindowPlan::Device(t) => Some(t),
            WindowPlan::Host(_) => None,
        }
    }
}

/// One layer's MoE, generic over the residency strategy (static dispatch).
#[derive(Debug, Clone)]
pub struct MoeLayer<P: Profile, R: ExpertResidency> {
    profile: PhantomData<P>,
    /// Router `hidden -> n_experts` (`ffn_gate_inp`).
    router: QMatMul,
    /// The routed experts, behind the residency seam.
    experts: R,
    /// Shared expert `up`/`gate`/`down` (`ffn_*_shexp`).
    sh_gate: QMatMul,
    sh_up: QMatMul,
    sh_down: QMatMul,
    /// Shared-expert gate `hidden -> 1` (`ffn_gate_inp_shexp`).
    sh_router: QMatMul,
    n_experts_per_tok: usize,
    /// The routed experts' count and intermediate width, which the residency
    /// does not expose (the walk reads them here).
    n_experts: usize,
    moe_intermediate_size: usize,
    /// The last routing's ids, under the hint-recall probe (see `probe_ids`).
    probe_ids: std::sync::Arc<parking_lot::Mutex<Option<Tensor>>>,
    /// The activation dtype of the routed dispatch — its `(activation, f32,
    /// store)` triple (§15.2a): `F32` is the byte-identical `_f32` routed arm,
    /// `F16` the `(f16, f32, f16)` arm (issue #748). The combine over slots
    /// stays f32 either way.
    routed_act: DType,
    /// Whether an f16 routed step runs its SwiGLU as one launch and its
    /// combine on the f16 routed output, as the f32 step does
    /// (`qwen4exp.f16_fuse`); off, `silu` + multiply and a cast to f32.
    f16_fuse: bool,
    /// Whether an f16 routed step reads its f32 activation in gate|up,
    /// rounded to half on load (`qwen4exp.f16_fold`); off, a cast launch
    /// first. The same bits either way.
    f16_fold: bool,
    /// Whether the shared expert's SwiGLU rides its quantized projections'
    /// stores at a one-row step (`qwen4exp.shared_store`, or a derived
    /// plan's verdict): the SiLU on the gate's store, the up projection
    /// multiplied into the gate's buffer — `silu_mul`'s launch removed.
    shared_store: bool,
}

/// What a routed MoE layer needs of a family's config: the expert count,
/// the top-k and the experts' width. A family builds one from its own
/// config ([`From<&Config>`] is Qwen4-exp's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGeometry {
    pub n_experts: usize,
    pub n_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
}

impl From<&Config> for MoeGeometry {
    fn from(cfg: &Config) -> Self {
        Self {
            n_experts: cfg.n_experts,
            n_experts_per_tok: cfg.n_experts_per_tok,
            moe_intermediate_size: cfg.moe_intermediate_size,
        }
    }
}

impl<P: Profile, R: ExpertResidency> MoeLayer<P, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: QMatMul,
        experts: R,
        sh_gate: QMatMul,
        sh_up: QMatMul,
        sh_down: QMatMul,
        sh_router: QMatMul,
        cfg: &Config,
    ) -> Self {
        Self::with_geometry(
            router,
            experts,
            sh_gate,
            sh_up,
            sh_down,
            sh_router,
            &MoeGeometry::from(cfg),
        )
    }

    /// [`Self::new`] from the geometry alone, for a family whose config is
    /// not Qwen4-exp's.
    #[allow(clippy::too_many_arguments)]
    pub fn with_geometry(
        router: QMatMul,
        experts: R,
        sh_gate: QMatMul,
        sh_up: QMatMul,
        sh_down: QMatMul,
        sh_router: QMatMul,
        geo: &MoeGeometry,
    ) -> Self {
        Self {
            profile: PhantomData,
            router,
            experts,
            sh_gate,
            sh_up,
            sh_down,
            sh_router,
            n_experts_per_tok: geo.n_experts_per_tok,
            n_experts: geo.n_experts,
            moe_intermediate_size: geo.moe_intermediate_size,
            routed_act: DType::F32,
            f16_fuse: crate::tuning::tuning().qwen4exp.f16_fuse,
            f16_fold: crate::tuning::tuning().qwen4exp.f16_fold,
            shared_store: crate::tuning::tuning().qwen4exp.shared_store,
            probe_ids: std::sync::Arc::new(parking_lot::Mutex::new(None)),
        }
    }

    /// The ids of the last routing this layer ran, kept only under the
    /// hint-recall probe (`QWEN4EXP_HINT_RECALL=1`): what a hint predicted
    /// is scored against them (`Qwen4Exp::hint_recall`).
    pub fn probe_ids(&self) -> Option<Tensor> {
        self.probe_ids.lock().clone()
    }

    /// Select the routed dispatch's activation dtype (`F32` or `F16`).
    pub fn set_routed_act(&mut self, dtype: DType) {
        self.routed_act = dtype;
    }

    pub fn routed_act(&self) -> DType {
        self.routed_act
    }

    /// Select whether the f16 routed step fuses its SwiGLU and its combine's
    /// read (see the field); the tuning's `qwen4exp.f16_fuse` by default.
    pub fn set_f16_fuse(&mut self, on: bool) {
        self.f16_fuse = on;
    }

    /// Select whether the shared expert's SwiGLU rides its projections'
    /// stores (see the field).
    pub fn set_shared_store(&mut self, on: bool) {
        self.shared_store = on;
    }

    pub fn shared_store(&self) -> bool {
        self.shared_store
    }

    pub fn f16_fuse(&self) -> bool {
        self.f16_fuse
    }

    /// Select whether the f16 routed step folds its activation cast into
    /// gate|up (see the field); the tuning's `qwen4exp.f16_fold` by default.
    pub fn set_f16_fold(&mut self, on: bool) {
        self.f16_fold = on;
    }

    /// The activation dtype a tile of `rows` rows routes in. The routed
    /// GEMM has the f32 activation arm only, so a tile on it keeps f32 and
    /// computes what the f32 default computes; the `F16` choice reaches the
    /// matmul-vec rows (a decode step, a verify window, a small tile). An
    /// f16 tile would leave the GEMM for the matmul-vec planes, a third of a
    /// prefill (`architecture.md` §2.3.5b, arm 8).
    fn tile_act(&self, rows: usize) -> DType {
        if MoeArm::for_rows(rows) == MoeArm::Gemm {
            DType::F32
        } else {
            self.routed_act
        }
    }

    /// Whether the routed experts read an `x` activation as `act` themselves
    /// (the f16 step's cast folded into gate|up) rather than a cast copy.
    fn folds(&self, x: DType, act: DType) -> bool {
        self.f16_fold && x == DType::F32 && act == DType::F16
    }

    /// The residency strategy behind the routed experts.
    pub fn experts(&self) -> &R {
        &self.experts
    }

    /// Experts routed per token.
    pub fn topk(&self) -> usize {
        self.n_experts_per_tok
    }

    /// One decode token through the MoE. `x` is `[1, 1, hidden]`, returns
    /// `[1, 1, hidden]` = routed top-k combine + gated shared expert.
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    /// `x [1, seq, hidden]` → `[1, seq, hidden]`, every position routed on
    /// its own: routed top-k combine + gated shared expert.
    ///
    /// Everything stays on the device: the top-k ids feed the routed dispatch
    /// directly (`seq` planes of `topk` slots) and the weighted sum over slots
    /// is a broadcast-multiply and a reduction. Correctness-first: the routing
    /// is a softmax over all experts, top-k, renormalize (reference
    /// `build_moe_ffn`).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_inner(x, false)?.0)
    }

    /// [`Self::forward`] with the routed experts planned ON THE DEVICE when
    /// the residency can (`ExpertResidency::plan_device`): no readback at
    /// the layer; the plan's record comes back with the output and the
    /// caller settles it at the end of the pass (`settle`), replaying the
    /// pass if it computed on a missing expert. `None` record = the host
    /// plan ran (nothing to settle).
    pub fn decode_planned(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        self.decode_planned_with(x, &mut ())
    }

    /// [`Self::decode_planned`] with `hook` run at the layer's join points
    /// (see [`PlanHook`]); a host plan (no record) runs neither.
    pub fn decode_planned_with<H: PlanHook>(
        &self,
        x: &Tensor,
        hook: &mut H,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let r = self.prefill_route(x)?;
        if hook.fetching() {
            return self.decode_fetched(&r, hook);
        }
        let Some((slot_ids, rec)) = self.experts.plan_device(&r.ids)? else {
            let groups = self.experts.plan(&r.ids)?;
            let y = self.routed_groups(&r.x3, &groups)?;
            return Ok((self.finish(&r, &y)?, None));
        };
        hook.after_plan(&rec)?;
        let y = self.routed_one_group(&r.x3, &slot_ids)?;
        let (shared, sh_logit) = self.shared_expert(&r)?;
        hook.at_join()?;
        let flags = self.mask_flags(&r, &rec)?;
        Ok((
            self.combine(&r, &y, &shared, &sh_logit, flags.as_ref())?,
            Some(rec),
        ))
    }

    /// A fetching step's MoE (`QWEN4EXP_FETCH=inside`,
    /// `expert-streaming.md` §20.14c): the router, the shared expert, then
    /// the device's wait for the worker that loads what the router chose
    /// (`PlanHook::fetch_join`), and only after it the slot map and the
    /// routed experts — so they read the pool the worker filled. The record
    /// is settled after the pass as the device plan's is; a flag in it is
    /// an expert the worker found no slot for, and the step is redone. A
    /// tier that does not fetch takes the plain device plan.
    fn decode_fetched<H: PlanHook>(
        &self,
        r: &PrefillRouted,
        hook: &mut H,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (shared, sh_logit) = self.shared_expert(r)?;
        hook.fetch_join(&self.experts, &r.ids)?;
        let Some((slot_ids, rec)) = self.experts.plan_device(&r.ids)? else {
            let groups = self.experts.plan(&r.ids)?;
            let y = self.routed_groups(&r.x3, &groups)?;
            return Ok((self.combine(r, &y, &shared, &sh_logit, None)?, None));
        };
        let y = self.routed_one_group(&r.x3, &slot_ids)?;
        let flags = self.mask_flags(r, &rec)?;
        Ok((
            self.combine(r, &y, &shared, &sh_logit, flags.as_ref())?,
            Some(rec),
        ))
    }

    /// A speculative window's `T` positions through the MoE (`x [T, 1,
    /// hidden]`, row 0 the real next token, the rows after it drafts),
    /// planned as `kind` says: on the device (`plan_device`, settled after
    /// the pass) or on the host (`plan_window`: the real rows loaded, a
    /// draft row's miss flagged, never loaded). Either way the plan's record comes back
    /// for the caller to settle AFTER the accept (`settle_window`), so a
    /// rejected row touches, counts and loads nothing (`expert-streaming.md`
    /// §20.16). `None` record: a strategy with no pool (`Resident`) took the
    /// plain groups.
    pub fn decode_window(
        &self,
        x: &Tensor,
        kind: WindowPlanKind,
    ) -> Result<(Tensor, Option<WindowPlan>)> {
        let r = self.prefill_route(x)?;
        match kind {
            WindowPlanKind::Device => {
                if let Some((slot_ids, rec)) = self.experts.plan_device(&r.ids)? {
                    let y = self.routed_one_group(&r.x3, &slot_ids)?;
                    let flags = self.mask_flags(&r, &rec)?;
                    return Ok((
                        self.finish_with(&r, &y, flags.as_ref())?,
                        Some(WindowPlan::Device(rec)),
                    ));
                }
            }
            WindowPlanKind::Host {
                real_rows,
                loaded_rows,
            } => {
                if let Some((slot_ids, rec)) =
                    self.experts.plan_window(&r.ids, real_rows, loaded_rows)?
                {
                    let y = self.routed_one_group(&r.x3, &slot_ids)?;
                    return Ok((self.finish(&r, &y)?, Some(WindowPlan::Host(rec))));
                }
            }
        }
        let groups = self.experts.plan(&r.ids)?;
        let y = self.routed_groups(&r.x3, &groups)?;
        Ok((self.finish(&r, &y)?, None))
    }

    /// The router alone: the top-k expert ids `[seq, topk]` (u32) of `x
    /// [1, seq, hidden]` — the routing `forward` computes, without the
    /// experts. For the streamed tier's residual speculation (§20.3 P2):
    /// this layer's router on a carry from BEFORE this layer's mixer, whose
    /// ids name the experts to stage ahead; a hint only, the real routing
    /// still decides. Untraced (a traced run keeps its one `moe.logits`).
    pub fn route_ids(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq, hidden) = x.dims3()?;
        let x1 = x.reshape((seq, hidden))?;
        let topk = self.n_experts_per_tok;
        let logits = self.router.forward(&x1)?.to_dtype(DType::F32)?;
        match super::kernels::moe_topk(&logits, topk) {
            Some(r) => Ok(r?.0),
            None => {
                let probs = ops::softmax_last_dim(&logits)?;
                probs
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, topk)?
                    .contiguous()
            }
        }
    }

    fn forward_inner(&self, x: &Tensor, device_plan: bool) -> Result<(Tensor, Option<Tensor>)> {
        let r = self.prefill_route(x)?;
        let device_plan = if device_plan {
            self.experts.plan_device(&r.ids)?
        } else {
            None
        };
        let (y, record) = match device_plan {
            // One group over the slot ids the device resolved.
            Some((slot_ids, rec)) => (self.routed_one_group(&r.x3, &slot_ids)?, Some(rec)),
            None => {
                let groups = self.experts.plan(&r.ids)?;
                (self.routed_groups(&r.x3, &groups)?, None)
            }
        };
        Ok((self.finish(&r, &y)?, record))
    }

    /// The routing of `x [1, seq, hidden]`, encoded and not read back:
    /// the router, the top-k ids and weights, the routed arm's activations.
    /// The first of the three stages a pipelined prefill drives separately
    /// (`DESIGN.md` §11.5z-ix item 2): route (encode) → plan (the readback)
    /// → execute (encode). `forward` is the three in a row.
    pub fn prefill_route(&self, x: &Tensor) -> Result<PrefillRouted> {
        let (b0, seq0, hidden) = x.dims3()?;
        // The MoE is position-wise: `B` rows of one token are `B` positions
        // of one sequence to it (the batched decode's `[B, 1, hidden]`).
        let (b, seq) = if b0 > 1 && seq0 == 1 {
            (1, b0)
        } else {
            (b0, seq0)
        };
        if b != 1 {
            lloom_tensor::bail!("moe takes one sequence, got {:?}", x.shape())
        }
        let x = &x.reshape((1, seq, hidden))?;
        let topk = self.n_experts_per_tok;
        let x1 = x.reshape((seq, hidden))?; // [seq, hidden]

        // Router: softmax over all experts, then top-k select + renormalize —
        // one launch on Metal (`kernels::MoeTopk`), the composed six elsewhere.
        let logits = self.router.forward(&x1)?.to_dtype(DType::F32)?; // [seq, n_experts]
        super::trace("moe.logits", usize::MAX, &logits);
        // One row: the ids go straight into the residency's host-read slot,
        // which its host plan reads in place (#814 part 5).
        let ids_slot = if seq == 1 {
            self.experts.decode_ids_slot(topk)?
        } else {
            None
        };
        // More rows' ids are read on the host by a streamed tier's plan,
        // after the pass has allocated past them (the activations' cast
        // below, under f16): a read no recorder sees, so they are a declared
        // escape — a plan would end their interval at the top-k and serve
        // their memory on (§9.2aa).
        let _declared = (ids_slot.is_none() && self.experts.plans_on_device())
            .then(lloom_tensor::EscapeGuard::new);
        let (ids, w) = match super::kernels::moe_topk_into(&logits, topk, ids_slot) {
            Some(r) => r?,
            None => {
                let probs = ops::softmax_last_dim(&logits)?; // [seq, n_experts]
                let ids = probs
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, topk)?
                    .contiguous()?; // [seq, topk] u32
                let w = probs.gather(&ids, D::Minus1)?; // [seq, topk]
                let w = w.broadcast_div(&w.sum_keepdim(D::Minus1)?)?;
                (ids, w)
            }
        };
        drop(_declared);
        if super::trace_enabled() {
            eprintln!(
                "TRACE moe.ids={:?} w={:?}",
                ids.flatten_all()?.to_vec1::<u32>()?,
                w.flatten_all()?.to_vec1::<f32>()?
            );
        }
        if hint_recall_probe() {
            *self.probe_ids.lock() = Some(ids.clone());
        }

        // The selected experts' SwiGLU, one routed dispatch per projection
        // per group, in the routed arm's activation dtype. `Resident` plans
        // one group over the ids as they are; `Streamed` groups the rows to
        // its slot pool and resolves each group's ids to slots first. A
        // folded f16 step hands gate|up the f32 activation, which they round
        // as they load it.
        let act = self.tile_act(seq);
        let x3 = if self.folds(x1.dtype(), act) {
            x1.reshape((seq, 1, hidden))?
        } else {
            x1.to_dtype(act)?.reshape((seq, 1, hidden))?
        }; // [seq, 1, hidden]
        Ok(PrefillRouted {
            x1,
            x3,
            ids,
            w,
            b0,
            seq0,
            seq,
            hidden,
            topk,
        })
    }

    /// The host plan of a routing: the ids read back and grouped to the
    /// pool (`ExpertResidency::plan`) — the one host wait of a layer.
    pub fn prefill_plan(&self, r: &PrefillRouted) -> Result<Vec<RoutedGroup>> {
        self.experts.plan(&r.ids)
    }

    /// Stage the ids' readback now (a blit and a marker, Metal), so
    /// [`Self::prefill_plan_staged`] waits for that alone later — the
    /// pipelined prefill's split of the one host wait. `None` where the
    /// residency plans on the device tensor (nothing to read back) or off
    /// Metal.
    pub fn prefill_stage_ids(&self, r: &PrefillRouted) -> Result<Option<StagedIds>> {
        #[cfg(feature = "metal")]
        if self.experts.plans_on_device() && r.ids.device().is_metal() {
            return Ok(Some(lloom_tensor::metal_backend::stage_readback(&r.ids)?));
        }
        let _ = r;
        Ok(None)
    }

    /// [`Self::prefill_plan`] from a staged readback: the ids from the
    /// copy, the groups from them.
    pub fn prefill_plan_staged(
        &self,
        r: &PrefillRouted,
        staged: Option<StagedIds>,
    ) -> Result<Vec<RoutedGroup>> {
        #[cfg(feature = "metal")]
        if let Some(st) = staged {
            let ids: Vec<u32> = st.finish()?;
            if let Some(groups) = self.experts.plan_host(&ids, r.seq, r.topk)? {
                return Ok(groups);
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = staged;
        self.prefill_plan(r)
    }

    /// The experts over the planned groups, the shared expert, the combine:
    /// the layer's MoE output `[b, seq, hidden]`.
    pub fn prefill_execute(&self, r: &PrefillRouted, groups: &[RoutedGroup]) -> Result<Tensor> {
        let y = self.routed_groups(&r.x3, groups)?;
        self.finish(r, &y)
    }

    /// The routed experts over one device-resolved group of slot ids. Its
    /// outputs are shaped by the pass alone (`[rows, topk, ·]`, one group),
    /// so a pass plan serves them (`DESIGN.md` §9.2aa).
    fn routed_one_group(&self, x3: &Tensor, slot_ids: &Tensor) -> Result<Tensor> {
        let arm = MoeArm::for_rows(x3.dim(0)?);
        let act = self.tile_act(x3.dim(0)?);
        let (g, u) = self.experts.gate_up(x3, slot_ids, arm, act)?;
        let h = self.routed_swiglu(&g, &u)?;
        self.experts.down(&h, slot_ids, arm)
    }

    /// The routed experts over the host plan's groups, each ensured (its
    /// misses loaded) as the pass reaches it, the later groups' misses
    /// prefetched while it runs. Every group runs the tile's routed arm, so
    /// how the pool splits the tile never changes a row's arithmetic
    /// (`expert-streaming.md` §20.15).
    fn routed_groups(&self, x3: &Tensor, groups: &[RoutedGroup]) -> Result<Tensor> {
        let arm = MoeArm::for_rows(x3.dim(0)?);
        let act = self.tile_act(x3.dim(0)?);
        // The routed GEMMs' outputs are sized by the plan's groups, which
        // the pool's state decides: not a pass-shape allocation, so
        // declared outside the plan (§9.2aa) and served by the pool. One
        // group of one row (a B = 1 decode step) is the pass's shape
        // whatever the pool holds, and a plan serves it.
        let one_row = groups.len() == 1 && groups[0].rows.len() == 1;
        let _escape = if one_row { None } else { escape_guard() };
        // The order the groups run in: the rows', or under the arm the
        // readiness order (`ready_first_on`): fewest experts to load first,
        // ties in row order.
        let reordered: Option<Vec<usize>> = if groups.len() > 1 && ready_first_on() {
            let misses: Vec<usize> = groups
                .iter()
                .map(|g| self.experts.group_misses(g))
                .collect();
            let mut order: Vec<usize> = (0..groups.len()).collect();
            order.sort_by_key(|&i| misses[i]);
            if order.iter().enumerate().any(|(k, &i)| k != i) {
                READY_REORDERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(order)
            } else {
                None
            }
        } else {
            None
        };
        let mut ys: Vec<Option<Tensor>> = vec![None; groups.len()];
        for k in 0..groups.len() {
            let gi = reordered.as_ref().map_or(k, |o| o[k]);
            let group = &groups[gi];
            let slot_ids = self.experts.ensure(group)?;
            // The later groups' misses read while this group runs.
            if k + 1 < groups.len() {
                match &reordered {
                    None => self.experts.prefetch_groups(&groups[k + 1..])?,
                    Some(o) => {
                        let later: Vec<RoutedGroup> =
                            o[k + 1..].iter().map(|&j| groups[j].clone()).collect();
                        self.experts.prefetch_groups(&later)?
                    }
                }
            }
            let xg = if groups.len() == 1 {
                x3.clone()
            } else {
                x3.narrow(0, group.rows.start, group.rows.len())?
            };
            let (g, u) = self.experts.gate_up(&xg, &slot_ids, arm, act)?; // [rows, topk, n_ff] each
            super::trace("moe.g", usize::MAX, &g);
            super::trace("moe.u", usize::MAX, &u);
            let h = self.routed_swiglu(&g, &u)?;
            ys[gi] = Some(self.experts.down(&h, &slot_ids, arm)?); // [rows, topk, hidden]
        }
        let mut ys: Vec<Tensor> = ys
            .into_iter()
            .map(|y| y.expect("every group ran"))
            .collect();
        if ys.len() == 1 {
            Ok(ys.pop().expect("one group"))
        } else {
            Tensor::cat(&ys, 0)
        }
    }

    /// `silu(g) · u` over the routed experts' gate|up: one launch at f32,
    /// and at f16 while `f16_fuse` holds; the composed pair otherwise.
    fn routed_swiglu(&self, g: &Tensor, u: &Tensor) -> Result<Tensor> {
        if g.dtype() != DType::F16 || self.f16_fuse {
            if let Some(h) = super::kernels::silu_mul(g, u) {
                return h;
            }
        }
        ops::silu(g)? * u
    }

    /// The shared expert and the combine over the routed output `y [seq,
    /// topk, hidden]` → `[b, seq, hidden]`.
    fn finish(&self, r: &PrefillRouted, y: &Tensor) -> Result<Tensor> {
        self.finish_with(r, y, None)
    }

    /// [`Self::finish`] with the mask arm's flags (a device plan's).
    fn finish_with(&self, r: &PrefillRouted, y: &Tensor, flags: Option<&Tensor>) -> Result<Tensor> {
        super::trace("moe.y", usize::MAX, y);
        let (shared, sh_logit) = self.shared_expert(r)?;
        self.combine(r, y, &shared, &sh_logit, flags)
    }

    /// The mask arm's flags from a device plan's record `[3·n]` (slot ids
    /// | ids | a miss flag per plane): `[seq, topk]` u32; `None` with the
    /// arm off.
    fn mask_flags(&self, r: &PrefillRouted, rec: &Tensor) -> Result<Option<Tensor>> {
        if !mask_on() {
            return Ok(None);
        }
        let n = r.seq * r.topk;
        let flags = rec.narrow(0, 2 * n, n)?.reshape((r.seq, r.topk))?;
        MASKED_COMBINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(flags))
    }

    /// The shared expert — SwiGLU, gated by `sigmoid(sh_router·x)`, always
    /// on — and its gate's logit `[seq, 1]`.
    fn shared_expert(&self, r: &PrefillRouted) -> Result<(Tensor, Tensor)> {
        let x1 = &r.x1;
        let shared = swiglu(
            x1,
            &self.sh_gate,
            &self.sh_up,
            &self.sh_down,
            self.shared_store,
        )?;
        let sh_logit = self.sh_router.forward(x1)?.to_dtype(shared.dtype())?; // [seq, 1]
        Ok((shared, sh_logit))
    }

    /// The routed combine over `y [seq, topk, hidden]` plus the gated
    /// shared expert → `[b, seq, hidden]`.
    fn combine(
        &self,
        r: &PrefillRouted,
        y: &Tensor,
        shared: &Tensor,
        sh_logit: &Tensor,
        flags: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (x1, w, seq, hidden, topk) = (&r.x1, &r.w, r.seq, r.hidden, r.topk);

        // The routed combine over the top-k plus the gated shared expert: one
        // launch on Metal (`kernels::MoeCombine`), the composed five elsewhere.
        // The launch reads an f16 routed output as it is, unless `f16_fuse`
        // is off.
        let y_in = if y.dtype() == DType::F16 && !self.f16_fuse {
            y.to_dtype(DType::F32)?
        } else {
            y.clone()
        };
        let out = match super::kernels::moe_combine(&y_in, w, shared, sh_logit, topk, hidden, flags)
        {
            Some(o) => o?.to_dtype(x1.dtype())?,
            None => {
                let y32 = y_in.to_dtype(DType::F32)?;
                // The mask: a flagged plane's weight is zero (`1 − flag`).
                let w = match flags {
                    Some(f) => (w * (f.to_dtype(DType::F32)?.neg()? + 1.0)?)?,
                    None => w.clone(),
                };
                let routed = y32
                    .broadcast_mul(&w.reshape((seq, topk, 1))?)?
                    .sum(1)?
                    .to_dtype(x1.dtype())?; // [seq, hidden]
                let sh_gate_v = ops::sigmoid(sh_logit)?;
                let shared = shared.broadcast_mul(&sh_gate_v)?;
                (routed + shared)?
            }
        };
        out.reshape((r.b0, r.seq0, hidden))
    }
}

#[cfg(feature = "metal")]
fn escape_guard() -> Option<lloom_tensor::metal_backend::EscapeGuard> {
    Some(lloom_tensor::metal_backend::EscapeGuard::new())
}
#[cfg(not(feature = "metal"))]
fn escape_guard() -> Option<()> {
    None
}

/// `QWEN4EXP_HINT_RECALL=1` (read once): score the residual-speculation
/// hint at decode — layer l+k's router on the carry entering layer l
/// against layer l+k's real routing (`Qwen4Exp::hint_recall`). An
/// instrument: extra router runs and readbacks per pass.
pub fn hint_recall_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.hint_recall)
}

/// A staged readback of a routing's ids (Metal); nothing elsewhere.
#[cfg(feature = "metal")]
pub type StagedIds = lloom_tensor::metal_backend::StagedReadback;
#[cfg(not(feature = "metal"))]
pub type StagedIds = ();

/// A layer's routing, encoded (`MoeLayer::prefill_route`): what the plan
/// reads back (`ids`) and what the execute needs after it.
pub struct PrefillRouted {
    x1: Tensor,
    x3: Tensor,
    ids: Tensor,
    w: Tensor,
    b0: usize,
    seq0: usize,
    seq: usize,
    hidden: usize,
    topk: usize,
}

/// SwiGLU: `down( silu(gate(x)) * up(x) )`.
/// One position's MoE at a decode step ([`MoeLayer::decode_planned_with`]):
/// the router, the fused top-k, the device plan's slot map when the
/// residency plans on the device (and the pass's plan is armed: a cold
/// pass's host plan reads the ids back instead), gate|up in one routed
/// launch, the routed
/// SwiGLU, the routed down, the shared expert (its two stems, its SwiGLU,
/// its down, its gate's logit), the combine. `In` is the bracket's output.
#[cfg(not(target_arch = "wasm32"))]
/// The routed MoE as a level-0 block (`nn::blocks::Ffn`): every position
/// routed on its own, the host plan; the device-planned pass and the
/// streamed tier's settle are the family's own path.
impl<P: Profile, R: ExpertResidency> crate::nn::blocks::Ffn for MoeLayer<P, R> {
    type State = ();
    fn execute(&self, x: &Tensor, _state: &mut ()) -> Result<Tensor> {
        self.forward(x)
    }
}

impl<P: Profile, R: ExpertResidency> crate::optimizer::Visit for MoeLayer<P, R> {
    type In = crate::optimizer::NodeId;
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, x: Self::In) -> Self::Out {
        self.visit_planned(w, x, self.experts.plans_on_device())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile, R: ExpertResidency> MoeLayer<P, R> {
    /// The visit with the routed experts planned on the device
    /// (`device_plan`: the slot map, one launch) or on the host (the ids
    /// read back, no launch) — a pass that plans on the host whatever the
    /// residency can (`Self::forward`, the MTP head's prefill) says so.
    pub(crate) fn visit_planned(
        &self,
        w: &mut crate::optimizer::Walker,
        x: crate::optimizer::NodeId,
        device_plan: bool,
    ) -> crate::optimizer::NodeId {
        use crate::optimizer::{Dim, Shape};
        let f32 = DType::F32;
        let h = match w.node(x).shape.last() {
            Some(Dim::Const(h)) => h,
            _ => unreachable!("a decode step's input has a constant width"),
        };
        let topk = self.n_experts_per_tok;
        let (e, mi) = (self.n_experts, self.moe_intermediate_size);
        let x1 = w.view("reshape", x, Shape::of(&[1, h]));
        let logits = w.project_qmat("ffn_gate_inp.weight", &self.router, x1);
        // The ids and their weights, one launch.
        let route = w.opaque("moe_topk", &[logits], Shape::of(&[1, topk]), f32);
        let act = self.tile_act(1);
        let x3 = if act != f32 && !self.folds(f32, act) {
            w.opaque("cast", &[x1], Shape::of(&[1, h]), act)
        } else {
            x1
        };
        let x3 = w.view("reshape", x3, Shape::of(&[1, 1, h]));
        // The device plan's slot map writes its record into the tier's
        // host-read slot in place (`Streamed::plan_device`): one launch.
        let slots = if device_plan {
            let plan = w.opaque("moe_slot_map", &[route], Shape::of(&[3 * topk]), DType::U32);
            w.view("narrow", plan, Shape::of(&[1, topk]))
        } else {
            route
        };
        let wg = w.weight("ffn_gate_exps.weight", Shape::of(&[e, mi, h]), DType::U8);
        let wu = w.weight("ffn_up_exps.weight", Shape::of(&[e, mi, h]), DType::U8);
        let wd = w.weight("ffn_down_exps.weight", Shape::of(&[e, h, mi]), DType::U8);
        let gu = w.opaque(
            "mul_mv_id2",
            &[wg, wu, x3, slots],
            Shape::of(&[2, 1, topk, mi]),
            act,
        );
        let g = w.view("narrow", gu, Shape::of(&[1, topk, mi]));
        let u = w.view("narrow", gu, Shape::of(&[1, topk, mi]));
        // The SwiGLU and, at f16, the combine's widening, as what they
        // compute: the derivation composes them (`silu_mul`, the combine
        // reading f16) where the step's switch does.
        let s = w.silu(g);
        let hmid = w.mul(s, u);
        let y = w.opaque(
            "mul_mv_id",
            &[wd, hmid, slots],
            Shape::of(&[1, topk, h]),
            act,
        );
        let y = if act == DType::F16 {
            w.opaque("cast", &[y], Shape::of(&[1, topk, h]), f32)
        } else {
            y
        };
        let sg = w.project_qmat("ffn_gate_shexp.weight", &self.sh_gate, x1);
        let su = w.project_qmat("ffn_up_shexp.weight", &self.sh_up, x1);
        let ss = w.silu(sg);
        let sm = w.mul(ss, su);
        let shared = w.project_qmat("ffn_down_shexp.weight", &self.sh_down, sm);
        let sh_logit = w.project_qmat("ffn_gate_inp_shexp.weight", &self.sh_router, x1);
        let out = w.opaque(
            "moe_combine",
            &[y, route, shared, sh_logit],
            Shape::of(&[1, h]),
            f32,
        );
        w.view("reshape", out, Shape::of(&[1, 1, h]))
    }
}

pub(crate) fn swiglu(
    x: &Tensor,
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    store: bool,
) -> Result<Tensor> {
    // On the projections' stores, where both serve the row: the SiLU on the
    // gate's, the up projection multiplied into the gate's buffer (which
    // nothing else reads).
    if store && gate.mv_store_serves(x) && up.mv_store_serves(x) {
        if let Some(g) = gate.forward_silu(x) {
            let g = g?;
            if let Some(done) = up.forward_mul_into(x, &g) {
                done?;
                return down.forward(&g);
            }
        }
    }
    let g = gate.forward(x)?;
    let u = up.forward(x)?;
    let h = match super::kernels::silu_mul(&g, &u) {
        Some(h) => h?,
        None => (ops::silu(&g)? * u)?,
    };
    down.forward(&h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::CrownExact;
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    use lloom_tensor::Device;
    use std::sync::Arc;

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let n = out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.1)
            .collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }

    /// A stacked `[e, out, inp]` expert weight as an F32-typed `QTensor` (the
    /// routed path's storage; F32 blocks so the CPU arm is exact).
    pub(crate) fn stacked(
        e: usize,
        out: usize,
        inp: usize,
        seed: f32,
        dev: &Device,
    ) -> Arc<QTensor> {
        let n = e * out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
            .collect();
        let t = Tensor::from_vec(v, (e, out, inp), dev).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap())
    }

    fn resident(cfg: &Config, dev: &Device) -> Resident {
        let (h, ff, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts);
        Resident::new(
            stacked(ne, ff, h, 1.0, dev),
            stacked(ne, ff, h, 2.0, dev),
            stacked(ne, h, ff, 3.0, dev),
        )
    }

    /// The MoE decodes end to end (Resident): shapes thread, output finite, and
    /// the seam monomorphizes to the no-loader path.
    #[test]
    fn moe_resident_decodes() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let h = cfg.hidden_size;
        let moe = MoeLayer::<CrownExact, _>::new(
            lin(cfg.n_experts, h, 1.0, &dev), // router
            resident(&cfg, &dev),
            lin(cfg.shared_expert_intermediate_size, h, 2.0, &dev), // sh_gate
            lin(cfg.shared_expert_intermediate_size, h, 3.0, &dev), // sh_up
            lin(h, cfg.shared_expert_intermediate_size, 4.0, &dev), // sh_down
            lin(1, h, 5.0, &dev),                                   // sh_router (1 value/token)
            &cfg,
        );
        let x = Tensor::from_vec(
            (0..h)
                .map(|i| ((i as f32) * 0.05).cos() * 0.3)
                .collect::<Vec<f32>>(),
            (1, 1, h),
            &dev,
        )
        .unwrap();
        let o = moe.decode(&x).unwrap();
        assert_eq!(o.dims(), &[1, 1, h]);
        let ov = o.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(ov.iter().all(|v| v.is_finite()));
        // Not degenerate zero (the shared expert is always on).
        assert!(ov.iter().any(|&v| v.abs() > 1e-9));
    }

    /// The routed combine equals an explicit per-expert reference: softmax the
    /// router, take the top-k, renormalize, and sum `w_e · down_e(silu(gate_e x)
    /// · up_e x)` with the dense per-expert matrices. Pins the routed dispatch
    /// (ids → stacked slices) and the on-device weighted sum against a value
    /// computed by a different route (§3.1b: a reference, not a shape).
    #[test]
    fn routed_combine_matches_per_expert_reference() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 6;
        cfg.n_experts_per_tok = 3;
        let (h, ff, ne, k) = (
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.n_experts,
            cfg.n_experts_per_tok,
        );
        let (gate, up, down) = (
            stacked(ne, ff, h, 1.0, &dev),
            stacked(ne, ff, h, 2.0, &dev),
            stacked(ne, h, ff, 3.0, &dev),
        );
        let router = lin(ne, h, 1.0, &dev);
        // zero shared expert so only the routed half is compared
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let moe = MoeLayer::<CrownExact, _>::new(
            router.clone(),
            Resident::new(gate.clone(), up.clone(), down.clone()),
            zero(ff, h),
            zero(ff, h),
            zero(h, ff),
            zero(1, h),
            &cfg,
        );
        let x = Tensor::from_vec(
            (0..h)
                .map(|i| ((i as f32) * 0.07).sin() * 0.4)
                .collect::<Vec<f32>>(),
            (1, 1, h),
            &dev,
        )
        .unwrap();
        let got = moe
            .decode(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // reference
        let x1 = x.reshape((1, h)).unwrap();
        let probs = ops::softmax_last_dim(&router.forward(&x1).unwrap()).unwrap();
        let pv = probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| pv[b].partial_cmp(&pv[a]).unwrap());
        let sel = &order[..k];
        let wsum: f32 = sel.iter().map(|&e| pv[e]).sum();
        let (gd, ud, dd) = (
            gate.dequantize(&dev).unwrap(),
            up.dequantize(&dev).unwrap(),
            down.dequantize(&dev).unwrap(),
        );
        let mut want = vec![0f32; h];
        for &e in sel {
            let g = x1.matmul(&gd.get(e).unwrap().t().unwrap()).unwrap();
            let u = x1.matmul(&ud.get(e).unwrap().t().unwrap()).unwrap();
            let hh = (ops::silu(&g).unwrap() * u).unwrap();
            let y = hh.matmul(&dd.get(e).unwrap().t().unwrap()).unwrap();
            let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (w, yv) in want.iter_mut().zip(yv) {
                *w += pv[e] / wsum * yv;
            }
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "mismatch at {i}: got {g} want {w}");
        }
        assert!(want.iter().any(|w| w.abs() > 1e-6), "degenerate reference");
    }

    /// A chunk through `forward` equals its positions through `decode` one by
    /// one (per-position routing, the `seq`-plane routed dispatch, the
    /// per-row combine).
    #[test]
    fn forward_over_seq_matches_per_position_decode() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 6;
        cfg.n_experts_per_tok = 3;
        let (h, ff, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts);
        let moe = MoeLayer::<CrownExact, _>::new(
            lin(ne, h, 1.0, &dev),
            Resident::new(
                stacked(ne, ff, h, 1.0, &dev),
                stacked(ne, ff, h, 2.0, &dev),
                stacked(ne, h, ff, 3.0, &dev),
            ),
            lin(ff, h, 2.0, &dev),
            lin(ff, h, 3.0, &dev),
            lin(h, ff, 4.0, &dev),
            lin(1, h, 5.0, &dev),
            &cfg,
        );
        let seq = 5;
        let x = Tensor::from_vec(
            (0..seq * h)
                .map(|i| ((i as f32) * 0.07).sin() * 0.4)
                .collect::<Vec<f32>>(),
            (1, seq, h),
            &dev,
        )
        .unwrap();
        let all = moe.forward(&x).unwrap();
        assert_eq!(all.dims(), &[1, seq, h]);
        for t in 0..seq {
            let one = moe.decode(&x.narrow(1, t, 1).unwrap()).unwrap();
            let (a, b) = (
                all.narrow(1, t, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-5, "position {t}: {a:?} vs {b:?}");
            }
        }
    }

    /// Router selects exactly `n_experts_per_tok` experts and the path runs
    /// with k=2 over 4 experts to the right shape.
    #[test]
    fn router_selects_topk_renormalized() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 4;
        cfg.n_experts_per_tok = 2;
        let (h, ff) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let moe = MoeLayer::<CrownExact, _>::new(
            lin(4, h, 1.0, &dev),
            resident(&cfg, &dev),
            zero(ff, h),
            zero(ff, h),
            zero(h, ff),
            zero(1, h),
            &cfg,
        );
        let x = Tensor::ones((1, 1, h), DType::F32, &dev).unwrap();
        let o = moe.decode(&x).unwrap();
        assert_eq!(o.dims(), &[1, 1, h]);
    }

    /// `route_ids` (the speculative hint's router) is the forward's routing:
    /// the same top-k as the composed softmax + sort over the router's
    /// logits, per position, over a chunk.
    #[test]
    fn route_ids_is_the_forwards_routing() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 8;
        cfg.n_experts_per_tok = 3;
        let (h, ff) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let router = lin(8, h, 0.7, &dev);
        let moe = MoeLayer::<CrownExact, _>::new(
            lin(8, h, 0.7, &dev),
            resident(&cfg, &dev),
            zero(ff, h),
            zero(ff, h),
            zero(h, ff),
            zero(1, h),
            &cfg,
        );
        let seq = 5;
        let x = Tensor::randn(0f32, 1.0, (1, seq, h), &dev).unwrap();
        let ids: Vec<u32> = moe
            .route_ids(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let logits = router.forward(&x.reshape((seq, h)).unwrap()).unwrap();
        let expect: Vec<u32> = ops::softmax_last_dim(&logits)
            .unwrap()
            .arg_sort_last_dim(false)
            .unwrap()
            .narrow(D::Minus1, 0, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(ids, expect);
        assert_eq!(ids.len(), seq * 3);
    }
}
