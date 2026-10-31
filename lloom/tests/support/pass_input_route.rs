//! Qualification-only physical input adapter. NOT NativeRing sampler/engine recovery.
//! Existing Ring provides token production; existing LFM embed runs before OpenPass.
//! Existing LFM forward_embeds runs inside OpenPass. No new model math/loop.
//! Diagnostic profile only; norm_gemv disabled explicitly; no performance claim.
use super::*;
use lloom::generation::{EosRead, ReadbackStrategy, Ring};
use lloom_tensor::backend::PassInput;
use std::io::Write;

// Fixed-count fixture: EOS is compared as a value, never used as a token index.
// Match the parent failure fixtures' unreachable sentinel, preserving actual
// argmax outputs rather than forcing/biasing them to avoid a sampled stop ID.
const NEVER_EOS: [u32; 1] = [u32::MAX];

type PhysicalBody = FailingBody<Model<CrownExact>>;

struct InputModel<'a> {
    inner: &'a PhysicalBody,
    observer: BodyObserver,
}
impl ModelTrait for InputModel<'_> {
    type Config = Config;
    fn from_weights(_: VarBuilder, _: &Config) -> Result<Self> {
        lloom_tensor::bail!("qualification wrapper borrows existing real model")
    }
}
impl CausalLM for InputModel<'_> {
    type Cache = State<CrownExact>;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        let out = self.inner.forward_embeds(embeds, pos, cache)?;
        Ok(out)
    }
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        let prepared = tokens.rank() == 3;
        self.observer.observe(
            if prepared { "prepared" } else { "token" },
            tokens,
            pos,
            || {
                if prepared {
                    self.inner.forward_embeds(tokens, pos, cache)
                } else {
                    self.inner.forward(tokens, pos, cache)
                }
            },
        )
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.inner.prepare_pass(cache, seq_len)
    }
    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        self.inner.note_input_ids(cache, ids)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        self.inner.checkpoint_at(cache, pos)
    }
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        self.inner.open_checkpoint_request(cache, policy, end)
    }
    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.close_checkpoint_request(cache)
    }
    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        self.inner.pin_checkpoint(cache, pos)
    }
    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        self.inner.pins_checkpoints(cache)
    }
    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        self.inner.pinned_checkpoint(cache)
    }
    fn host_feed(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Option<lloom::nn::ngram_memory::HostFeed>> {
        self.inner.host_feed(cache)
    }
    fn host_feed_end(&self, cache: &mut Self::Cache, published: &[u32]) -> Result<()> {
        self.inner.host_feed_end(cache, published)
    }
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.settle_pending(cache)
    }
    fn provisional(&self, cache: &mut Self::Cache, mode: Provisional) -> Result<Provisional> {
        self.inner.provisional(cache, mode)
    }
    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        self.inner.settle_step(cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

struct InputRing<'a, const PREPARED: bool> {
    ring: Ring,
    model: &'a PhysicalBody,
}
impl<const PREPARED: bool> ReadbackStrategy for InputRing<'_, PREPARED> {
    const PASS_INPUT: PassInput = if PREPARED {
        PassInput::PreparedEmbeddings
    } else {
        PassInput::TokenIds
    };
    fn device(&self) -> &Device {
        self.ring.device()
    }
    fn arms(&self) -> lloom::generation::ReadbackArms {
        self.ring.arms()
    }
    fn admits_lookahead(&self) -> bool {
        self.ring.admits_lookahead()
    }
    fn loop_commits(&self) -> bool {
        self.ring.loop_commits()
    }
    fn check_sampler(&self, s: &Sampling) -> Result<()> {
        self.ring.check_sampler(s)
    }
    fn sample_and_feed(&mut self, s: &mut LogitsProcessor, z: &Tensor) -> Result<Tensor> {
        let ids = self.ring.sample_and_feed(s, z)?;
        if PREPARED {
            self.model.embed(&ids)
        } else {
            Ok(ids)
        }
    }
    fn eos_of_last(&mut self) -> Result<bool> {
        self.ring.eos_of_last()
    }
    fn take_ids(&mut self) -> Result<Vec<u32>> {
        self.ring.take_ids()
    }
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.ring.take_ids_into(out)
    }
    fn drain_all(&mut self) -> Result<Vec<u32>> {
        self.ring.drain_all()
    }
    fn begin_run(&mut self) {
        self.ring.begin_run()
    }
    fn eos_read_arm(&self) -> &'static str {
        self.ring.eos_read_arm()
    }
    fn stage_prompt_chunk(&mut self, ids: &[u32]) -> Result<Tensor> {
        self.ring.stage_prompt_chunk(ids)
    }
    fn publish_marker(&mut self, force: bool) -> Result<()> {
        self.ring.publish_marker(force)
    }
    fn close_window(&mut self, ttft: bool) -> Result<()> {
        self.ring.close_window(ttft)
    }
    fn eos_in_drained(&self, ids: &[u32]) -> Option<usize> {
        self.ring.eos_in_drained(ids)
    }
    fn note_eos_truncation(&mut self, dropped: usize) {
        self.ring.note_eos_truncation(dropped)
    }
    fn fed_host_ids(&self) -> Option<&[u32]> {
        self.ring.fed_host_ids()
    }
    fn retracts_draw(&self) -> bool {
        self.ring.retracts_draw()
    }
    fn retract_draw(&mut self) -> Result<()> {
        self.ring.retract_draw()
    }
    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        self.ring.eos_ids_tensor()
    }
    fn reader(&self) -> Option<lloom::generation::readback::RingReader> {
        ReadbackStrategy::reader(&self.ring)
    }
    fn published_count(&self) -> Option<usize> {
        self.ring.published_count()
    }
    fn terminal_observers(&self) -> lloom::nn::terminal_trace::Observers {
        self.ring.terminal_observers()
    }
    fn stamps_each_token(&self, adaptive: bool) -> bool {
        self.ring.stamps_each_token(adaptive)
    }
    fn arm_publish_event(&mut self, reader: bool, park: bool, adaptive: bool) -> Result<()> {
        self.ring.arm_publish_event(reader, park, adaptive)
    }
    fn set_fused_publish(&mut self, on: bool) {
        self.ring.set_fused_publish(on)
    }
    fn publishes_fused(&self) -> bool {
        self.ring.publishes_fused()
    }
    fn counters(&self) -> lloom::generation::ReadbackCounters {
        self.ring.counters()
    }
    fn set_thinking_budget(&mut self, b: Option<&lloom::generation::ThinkingBudget>) -> Result<()> {
        self.ring.set_thinking_budget(b)
    }
}
struct BodyObserver {
    frames: std::cell::RefCell<Vec<BodyFrame>>,
    capacity: usize,
    lost: std::cell::Cell<bool>,
}

struct BodyFrame {
    route: &'static str,
    pos: usize,
    shape: Vec<usize>,
    plan: Option<serde_json::Value>,
    outcome: String,
    dispatches: Vec<lloom_tensor::kernels::metal::trace::Dispatch>,
}

struct BodyScope<'a> {
    owner: &'a BodyObserver,
    frame: Option<BodyFrame>,
    window: Option<lloom_tensor::kernels::metal::trace::TraceWindow>,
}

impl Drop for BodyScope<'_> {
    fn drop(&mut self) {
        // Preserve the attempted trace even if the delegated body unwinds.
        // The existing window closes before any later body may open a window.
        if let (Some(mut frame), Some(window)) = (self.frame.take(), self.window.take()) {
            frame.dispatches = window.close();
            if let Ok(mut frames) = self.owner.frames.try_borrow_mut() {
                if frames.len() < self.owner.capacity {
                    frames.push(frame);
                    return;
                }
            }
            self.owner.lost.set(true);
        }
    }
}

impl BodyObserver {
    fn new(capacity: usize) -> Self {
        Self {
            frames: std::cell::RefCell::new(Vec::with_capacity(capacity)),
            capacity,
            lost: std::cell::Cell::new(false),
        }
    }

    fn observe(
        &self,
        route: &'static str,
        input: &Tensor,
        pos: usize,
        body: impl FnOnce() -> lloom_tensor::Result<Tensor>,
    ) -> lloom_tensor::Result<Tensor> {
        if self.frames.borrow().len() >= self.capacity {
            self.lost.set(true);
            lloom_tensor::bail!("body observer capacity exhausted; no complete witness");
        }
        let metal = input.device().as_metal_device()?;
        // Read installed metadata inside the pass, before end_pass clears it.
        // Keep only plain host metadata: do not retain the arena's allocation.
        let plan = metal.arena().map(|arena| {
            serde_json::json!({
                "base_addr": arena.base().raw_addr(),
                "base_offset": arena.base().base_offset(),
                "base_bytes": arena.base().length(),
                "served_counter_inside_body": metal.pass_plan_counters().map(|c| c.served),
                "plan_debug": format!("{:?}", arena.plan_snapshot()),
                "layout": format!("{:?}", arena.layout()),
                "offsets": format!("{:?}", arena.offsets()),
            })
        });
        let window = metal
            .dispatch_trace()
            .open()
            .map_err(|e| lloom_tensor::Error::Msg(format!("body observer trace refused: {e:?}")))?;
        let mut scope = BodyScope {
            owner: self,
            frame: Some(BodyFrame {
                route,
                pos,
                shape: input.dims().to_vec(),
                plan,
                outcome: "unwound".into(),
                dispatches: Vec::new(),
            }),
            window: Some(window),
        };
        let result = body(); // exactly one call, unchanged output Tensor
        scope.frame.as_mut().unwrap().outcome = match &result {
            Ok(_) => "returned".into(),
            Err(error) => format!("error: {error}"),
        };
        drop(scope);
        result
    }
}
struct FirstConvEdge<'a> {
    residual_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    up_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    down_dispatch: &'a lloom_tensor::kernels::metal::trace::Dispatch,
    residual: &'a lloom_tensor::kernels::metal::trace::Binding,
    up: &'a lloom_tensor::kernels::metal::trace::Binding,
    down_residual: &'a lloom_tensor::kernels::metal::trace::Binding,
    overlap: bool,
}

fn explicit_edge_binding(
    dispatch: &lloom_tensor::kernels::metal::trace::Dispatch,
    index: usize,
    output: bool,
    logical_bytes: usize,
) -> anyhow::Result<&lloom_tensor::kernels::metal::trace::Binding> {
    let mut candidates = dispatch.bindings.iter().filter(|b| b.index == index);
    let b = candidates.next().ok_or_else(|| {
        anyhow::anyhow!("model edge witness incomplete: missing explicit argument {index}")
    })?;
    anyhow::ensure!(
        candidates.next().is_none()
            && b.is_output == output
            && b.buffer_addr != 0
            && b.len >= logical_bytes
            && b.offset.checked_add(b.len).is_some(),
        "model edge witness incomplete: duplicate, unknown or short argument {index}"
    );
    Ok(b)
}

fn first_conv_edge(
    dispatches: &[lloom_tensor::kernels::metal::trace::Dispatch],
) -> anyhow::Result<FirstConvEdge<'_>> {
    let first = dispatches
        .iter()
        .position(|d| d.pipeline == "shortconv_step_f16_k3")
        .ok_or_else(|| anyhow::anyhow!("model edge witness incomplete: first shortconv route"))?;
    let d = dispatches
        .get(first..first + 7)
        .ok_or_else(|| anyhow::anyhow!("model edge witness incomplete: first MLP sequence"))?;
    let gemv = "gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
    anyhow::ensure!(
        d[1].pipeline == format!("{gemv}_residual")
            && d[2].pipeline == "rmsnorm_f16"
            && d[3].pipeline == format!("{gemv}_silu")
            && d[4].pipeline == gemv
            && d[5].pipeline == "bmul_f16"
            && d[6].pipeline == format!("{gemv}_residual")
            && d.windows(2).all(|w| w[0].seq < w[1].seq),
        "model edge witness incomplete: changed conv/norm/gate/up/product/down route"
    );
    let residual = explicit_edge_binding(&d[1], 3, true, 256)?;
    let up = explicit_edge_binding(&d[4], 3, true, 512)?;
    let down_residual = explicit_edge_binding(&d[6], 15, false, 256)?;
    anyhow::ensure!(
        residual.buffer_addr == down_residual.buffer_addr
            && residual.offset == down_residual.offset,
        "model edge witness incomplete: down does not read original residual"
    );
    // Object identity + within-object byte offsets; NOT GPU virtual addresses.
    let overlap = residual.buffer_addr == up.buffer_addr
        && residual.offset < up.offset + 512
        && up.offset < residual.offset + 256;
    Ok(FirstConvEdge {
        residual_dispatch: &d[1],
        up_dispatch: &d[4],
        down_dispatch: &d[6],
        residual,
        up,
        down_residual,
        overlap,
    })
}

fn edge_binding_json(
    b: &lloom_tensor::kernels::metal::trace::Binding,
    logical_bytes: usize,
) -> serde_json::Value {
    serde_json::json!({"argument": b.index, "output": b.is_output,
        "buffer_object": b.buffer_addr, "interned_id": b.buffer_id,
        "offset": b.offset, "logical_bytes": logical_bytes,
        "logical_end": b.offset + logical_bytes, "handle_bytes": b.len})
}

impl BodyObserver {
    fn check_first_conv_edges(&self, arm: &str, prefill_units: usize) -> anyhow::Result<()> {
        let frames = self.frames.borrow();
        let mut installed_bodies = 0usize;
        for (index, frame) in frames.iter().enumerate().skip(prefill_units) {
            let edge = first_conv_edge(&frame.dispatches)?;
            let witness = serde_json::json!({
                "arm": arm, "frame": index, "input_pos": frame.pos, "route": frame.route,
                "r": edge_binding_json(edge.residual, 256),
                "up": edge_binding_json(edge.up, 512),
                "down_residual": edge_binding_json(edge.down_residual, 256),
                "pipelines": [edge.residual_dispatch.pipeline, edge.up_dispatch.pipeline,
                    edge.down_dispatch.pipeline],
                "trace_seq": [edge.residual_dispatch.seq, edge.up_dispatch.seq, edge.down_dispatch.seq],
                "coordinate": "trace seq, NOT STEP_DISPATCH; MTLBuffer object, NOT GPU VA",
                "plan": frame.plan, "overlap": edge.overlap,
            });
            writeln!(std::io::stdout().lock(), "BODY_MODEL_EDGE {witness}")?;
            if edge.overlap {
                writeln!(std::io::stdout().lock(), "BODY_ALIAS_CONFLICT {witness}")?;
                anyhow::bail!("model residual/up alias before down read");
            }
            if let Some(plan) = &frame.plan {
                let base = plan["base_addr"].as_u64().ok_or_else(|| {
                    anyhow::anyhow!("model edge witness incomplete: installed base identity")
                })?;
                anyhow::ensure!(
                    plan["served_counter_inside_body"]
                        .as_u64()
                        .is_some_and(|n| n > 0),
                    "model edge witness incomplete: no serving counter inside installed body"
                );
                // A body actually bound an operand to its installed arena,
                // stronger than finding a key in the cache AFTER the request.
                if [edge.residual, edge.up, edge.down_residual]
                    .into_iter()
                    .any(|b| b.buffer_addr as u64 == base)
                {
                    installed_bodies += 1;
                }
            }
        }
        anyhow::ensure!(
            installed_bodies > 0,
            "model edge witness incomplete: route executed no arena-bound decode body"
        );
        writeln!(std::io::stdout().lock(), "BODY_PLAN_SERVED arm={arm} arena_bound_bodies={installed_bodies} active_key_event=UNOBSERVED served_offer_ordinals=UNOBSERVED")?;
        Ok(())
    }
}

fn one<const PREPARED: bool>(
    model: &InputModel<'_>,
    cfg: &Config,
    dev: &Device,
    check_edges: bool,
) -> anyhow::Result<Vec<u32>> {
    model.observer.frames.borrow_mut().clear();
    model.observer.lost.set(false);
    let cache = model.new_cache(cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut readback = InputRing::<PREPARED> {
        ring: Ring::new(&NEVER_EOS, dev, 16, EosRead::Blocking)?,
        model: model.inner,
    };
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let result = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &SHORT,
        &DecodeOptions::new(8),
        &mut |_| Ok(()),
    );
    // Same cleanup rule as archived observer: fence before inspecting/dropping
    // captured metadata. Preserve primary error if the cleanup also refuses.
    let fence = dev.synchronize();
    let out = match result {
        Ok(out) => {
            fence?;
            out
        }
        Err(error) => {
            eprintln!("PASS_INPUT_BODY_ERROR cleanup={fence:?}");
            return Err(error.into());
        }
    };
    anyhow::ensure!(!model.observer.lost.get(), "body observer lost frame");
    anyhow::ensure!(
        out.ids.len() == 8 && !out.eos,
        "route body did not run eight steps"
    );
    let frames = model.observer.frames.borrow();
    let decode = frames
        .iter()
        .filter(|f| f.pos >= SHORT.len())
        .collect::<Vec<_>>();
    anyhow::ensure!(decode.len() == 8, "actual decode forward count");
    for (i, frame) in decode.iter().enumerate() {
        anyhow::ensure!(
            frame.pos == SHORT.len() + i && frame.outcome == "returned",
            "body sequence"
        );
        anyhow::ensure!(
            frame.route == if PREPARED { "prepared" } else { "token" },
            "input route"
        );
        anyhow::ensure!(
            frame.shape
                == if PREPARED {
                    vec![1, 1, 128]
                } else {
                    vec![1, 1]
                },
            "input shape"
        );
    }
    let prefill_units = frames.len() - decode.len();
    drop(frames);
    if check_edges {
        model
            .observer
            .check_first_conv_edges(if PREPARED { "prepared" } else { "token" }, prefill_units)?;
    }
    Ok(out.ids)
}

// The public embed entry may differ in the gather prefix only. Verify the
// actual remaining core/head pipeline sequence; configured flags alone do not
// establish same math. Preserve the strict semantic first-conv validator.
fn cold_kernel_routes(model: &InputModel<'_>) -> anyhow::Result<Vec<Vec<String>>> {
    let frames = model.observer.frames.borrow();
    frames
        .iter()
        .filter(|f| f.pos >= SHORT.len())
        .map(|frame| {
            first_conv_edge(&frame.dispatches)?;
            let start = frame
                .dispatches
                .iter()
                .position(|d| d.pipeline == "shortconv_step_f16_k3")
                .ok_or_else(|| anyhow::anyhow!("cold body has no qualified conv route"))?;
            Ok(frame.dispatches[start..]
                .iter()
                .map(|d| d.pipeline.to_string())
                .collect())
        })
        .collect()
}

fn run_order(prepared_first: bool) -> anyhow::Result<()> {
    // Tiny-only qualification; never silently run a checkpoint or skip.
    anyhow::ensure!(
        std::env::var("LLOOM_FIXTURE").unwrap_or_else(|_| "tiny".into()) == "tiny",
        "tiny fixture required"
    );
    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning.from_env().map_err(anyhow::Error::msg)?;
    tuning.instrument.profile = true;
    tuning.instrument.trace = true;
    tuning.contract.profile = "deterministic".into();
    lloom_tensor::tuning::install(tuning).map_err(anyhow::Error::msg)?;
    let (_fixture, dev, mut body, cfg) = lfm2_as::<CrownExact>("pass-input-route", |cfg| {
        // Same qualified route as the historical Engine cache=crown fixture.
        // CrownExact + Shuffle + fused shortconv, NOT the parent's rewindable
        // RotatingRing arm (which selects shortconv_ring and a different oracle).
        lloom::models::lfm2_presets::crown(cfg);
        cfg.conv_prefill_ctx = false;
    })?
    .ok_or_else(|| anyhow::anyhow!("tiny fixture unexpectedly absent"))?;
    anyhow::ensure!(
        cfg.vocab_size <= u32::MAX as usize,
        "sentinel inside vocabulary"
    );
    body.inner.set_norm_gemv(false);
    let model = InputModel {
        inner: &body,
        observer: BodyObserver::new(64),
    };
    let m = metal(&dev);
    eprintln!("PASS_INPUT_OPENPASS actual_api={} prepared_first={prepared_first} norm_gemv=false adapter=shipping-ring-public-embed sampler_native=false", m.commands().actual_command_api());
    // Independent no-plan cold comparator. Both execute actual model math.
    anyhow::ensure!(
        !m.pass_plans_enabled() && m.arena().is_none(),
        "cold comparator already has a planner/arena"
    );
    let cold_tokens = one::<false>(&model, &cfg, &dev, false)?;
    let token_route = cold_kernel_routes(&model)?;
    anyhow::ensure!(
        !m.pass_plans_enabled() && m.arena().is_none(),
        "token comparator installed planner/arena"
    );
    let cold_prepared = one::<true>(&model, &cfg, &dev, false)?;
    let prepared_route = cold_kernel_routes(&model)?;
    anyhow::ensure!(
        !m.pass_plans_enabled() && m.arena().is_none(),
        "prepared comparator installed planner/arena"
    );
    anyhow::ensure!(
        token_route == prepared_route,
        "cold adapters selected different core/head kernel routes"
    );
    eprintln!("PASS_INPUT_COLD_KERNELS {:?}", token_route);
    anyhow::ensure!(cold_tokens == cold_prepared, "cold route token divergence");
    m.enable_pass_plans(ArenaLayout::Packed);
    let run = |prepared: bool, edges: bool| -> anyhow::Result<Vec<u32>> {
        if prepared {
            one::<true>(&model, &cfg, &dev, edges)
        } else {
            one::<false>(&model, &cfg, &dev, edges)
        }
    };
    // Each executable selector starts a fresh device and independently records
    // the opposite order. Cold sequence reset does not clear the plan cache.
    for prepared in [prepared_first, !prepared_first] {
        for _ in 0..2 {
            anyhow::ensure!(
                run(prepared, true)? == cold_tokens,
                "warm/cold output mismatch"
            );
        }
    }
    let before = counters(m);
    for prepared in [
        prepared_first,
        !prepared_first,
        prepared_first,
        !prepared_first,
    ] {
        anyhow::ensure!(
            run(prepared, true)? == cold_tokens,
            "served/cold output mismatch"
        );
    }
    let after = counters(m);
    anyhow::ensure!(
        after.served > before.served
            && after.evicted == before.evicted
            && after.recorded == before.recorded,
        "warm plans not stably served: before={before:?} after={after:?}"
    );
    // Physical edge assertion above precedes these key-label checks.
    let (plans, _) = m
        .pass_plan_report()
        .ok_or_else(|| anyhow::anyhow!("plan report missing"))?;
    for input in [PassInput::TokenIds, PassInput::PreparedEmbeddings] {
        anyhow::ensure!(
            plans.iter().any(|(key, covered, ..)| key.kind
                == lloom_tensor::metal_backend::PassKind::Decode { b: 1 }
                && key.input == input
                && *covered > 0),
            "expected input route plan absent: {input:?}"
        );
    }
    // Scoped recovery: actual model-body failure inside production OpenPass.
    // NOT the archived engine invalid-sampled-row/next-public-request gate.
    for prepared in [false, true] {
        body.arm(1, Fail::Err);
        let failed_before = body.failed();
        let error = run(prepared, false).expect_err("body injection must fire");
        body.arm(0, Fail::Err);
        anyhow::ensure!(
            body.failed() == failed_before + 1 && error.to_string().contains("injected"),
            "wrong injected failure: {error}"
        );
        anyhow::ensure!(!m.pass_open(), "OpenPass left failed body open");
        anyhow::ensure!(
            run(prepared, true)? == cold_tokens,
            "post-body-failure request differs from cold"
        );
    }
    eprintln!("PASS_INPUT_OPENPASS_COMPLETE prepared_first={prepared_first} before={before:?} after={after:?} ids={cold_tokens:?} recovery=model_body_only");
    Ok(())
}

#[test]
#[ignore = "separately admitted tiny Metal physical-route qualification; not timing"]
fn token_then_prepared_through_openpass() -> anyhow::Result<()> {
    run_order(false)
}

#[test]
#[ignore = "separately admitted tiny Metal reverse physical-route qualification; not timing"]
fn prepared_then_token_through_openpass() -> anyhow::Result<()> {
    run_order(true)
}

// Same prompt and forced feed sequence as the archived owner witness. The
// sampler/owner remain different: this exercises the public Session substrate.
const WARM_PROMPT: [u32; 11] = [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100];
const WARM_FED: [u32; 8] = [97, 97, 97, 97, 97, 97, 97, 98];

fn reused_request<const PREPARED: bool>(
    session: &mut Session<'_, InputModel<'_>>,
    model: &InputModel<'_>,
    dev: &Device,
    expect_reused: usize,
    check_edges: bool,
) -> anyhow::Result<()> {
    use lloom::nn::ReuseAction;
    model.observer.frames.borrow_mut().clear();
    model.observer.lost.set(false);
    let before = session.pos();
    let prepared = session.prepare(&WARM_PROMPT)?;
    anyhow::ensure!(
        prepared.reused == expect_reused
            && prepared.suffix == WARM_PROMPT[expect_reused..]
            && session.pos() == expect_reused,
        "warm preparation mismatch: before={before} prepared={prepared:?} pos={}",
        session.pos()
    );
    if expect_reused == 0 {
        anyhow::ensure!(prepared.action == ReuseAction::Fresh, "{prepared:?}");
    } else {
        anyhow::ensure!(
            before == 19
                && prepared.action == ReuseAction::Rewound(9)
                && prepared.reused == 10
                && prepared.suffix == [100],
            "original warm transition not exercised: before={before} prepared={prepared:?}"
        );
    }
    let mut readback = InputRing::<PREPARED> {
        ring: Ring::new(&NEVER_EOS, dev, 16, EosRead::Blocking)?,
        model: model.inner,
    };
    let budget = lloom::generation::ThinkingBudget::new(0, vec![98], WARM_FED.to_vec())?;
    readback.set_thinking_budget(Some(&budget))?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    sampler.observe_prompt(&WARM_PROMPT);
    model.open_checkpoint_request(
        session.cache_mut(),
        CheckpointPolicy::End,
        WARM_PROMPT.len(),
    )?;
    let result = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &prepared.suffix,
        &DecodeOptions::new(8),
        &mut |_| Ok(()),
    );
    // Attempt both cleanup operations before returning a primary body error.
    // No reset is disguised as successful prefix recovery.
    let closed = model.close_checkpoint_request(session.cache_mut());
    let fence = dev.synchronize();
    let out = match result {
        Ok(out) => {
            closed?;
            fence?;
            out
        }
        Err(error) => {
            eprintln!("PASS_INPUT_WARM_ERROR close={closed:?} fence={fence:?}");
            return Err(error.into());
        }
    };
    anyhow::ensure!(out.ids == WARM_FED && !out.eos, "forced feed contract");
    anyhow::ensure!(
        session.pos() == 19 && session.overrun() == 0,
        "warm terminal frontier"
    );
    anyhow::ensure!(
        session.tokens() == [&WARM_PROMPT[..], &WARM_FED[..]].concat(),
        "warm token log"
    );
    anyhow::ensure!(!model.observer.lost.get(), "warm observer lost frame");
    let frames = model.observer.frames.borrow();
    let units = out.stats.prefill_units;
    anyhow::ensure!(units == 1 && frames.len() == units + 8, "warm body count");
    let first = &frames[0];
    anyhow::ensure!(
        first.route == "token"
            && first.pos == expect_reused
            && first.shape == [1, WARM_PROMPT.len() - expect_reused],
        "preparation-derived prefill span/route"
    );
    for (i, frame) in frames[units..].iter().enumerate() {
        anyhow::ensure!(
            frame.pos == 11 + i
                && frame.route == if PREPARED { "prepared" } else { "token" }
                && frame.shape
                    == if PREPARED {
                        vec![1, 1, 128]
                    } else {
                        vec![1, 1]
                    },
            "warm decode position/input route"
        );
    }
    anyhow::ensure!(
        frames
            .iter()
            .all(|f| f.outcome == "returned" && !f.dispatches.is_empty()),
        "warm body failed or empty"
    );
    drop(frames);
    eprintln!("PASS_INPUT_REUSED_PREFIX prepared_input={PREPARED} before={before} preparation={prepared:?} prefill_units={units} positions={}..19 forced_feed_only=true", expect_reused);
    if check_edges {
        model.observer.check_first_conv_edges(
            if PREPARED {
                "warm-prepared"
            } else {
                "warm-token"
            },
            units,
        )?;
    } else {
        // Cold storage need not be arena-backed, but the same logical
        // residual/up/down lifetime still has to be disjoint.
        for (index, frame) in model
            .observer
            .frames
            .borrow()
            .iter()
            .enumerate()
            .skip(units)
        {
            let edge = first_conv_edge(&frame.dispatches)?;
            let witness = serde_json::json!({
                "frame": index, "input_pos": frame.pos, "route": frame.route,
                "r": edge_binding_json(edge.residual, 256),
                "up": edge_binding_json(edge.up, 512),
                "down_residual": edge_binding_json(edge.down_residual, 256),
                "pipelines": [edge.residual_dispatch.pipeline, edge.up_dispatch.pipeline,
                    edge.down_dispatch.pipeline],
                "trace_seq": [edge.residual_dispatch.seq, edge.up_dispatch.seq, edge.down_dispatch.seq],
                "coordinate": "MTLBuffer object and byte offset, NOT GPU VA",
                "overlap": edge.overlap, "arena_required": false,
            });
            writeln!(std::io::stdout().lock(), "BODY_COLD_MODEL_EDGE {witness}")?;
            anyhow::ensure!(
                !edge.overlap,
                "cold model residual/up alias before down read"
            );
        }
    }
    Ok(())
}

fn run_reused_order(prepared_first: bool) -> anyhow::Result<()> {
    use lloom::testing::tiny::TinyFamily;
    anyhow::ensure!(
        std::env::var("LLOOM_FIXTURE").unwrap_or_else(|_| "tiny".into()) == "tiny",
        "tiny fixture required"
    );
    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning.from_env().map_err(anyhow::Error::msg)?;
    tuning.instrument.profile = true;
    tuning.instrument.trace = true;
    tuning.contract.profile = "deterministic".into();
    lloom_tensor::tuning::install(tuning).map_err(anyhow::Error::msg)?;
    let mut template = Lfm2Tiny::template();
    template.vocab_size = Lfm2Tiny::tokenizer().vocab_size();
    let fixture = Fixture::<Lfm2Tiny>::new("pass-input-reused-prefix", template)?
        .ok_or_else(|| anyhow::anyhow!("tiny fixture unexpectedly absent"))?;
    let dev = Device::new_metal(0)?;
    let cfg = fixture.config(|cfg| {
        lloom::models::lfm2_presets::crown(cfg);
        cfg.conv_prefill_ctx = false;
    });
    let mut inner = fixture.model_in::<CrownExact>(&cfg, Some(DType::F16), &dev)?;
    anyhow::ensure!(
        cfg.vocab_size <= u32::MAX as usize,
        "sentinel inside vocabulary"
    );
    anyhow::ensure!(
        WARM_PROMPT
            .iter()
            .chain(WARM_FED.iter())
            .all(|id| (*id as usize) < cfg.vocab_size),
        "fixed warm IDs outside fixture vocabulary"
    );
    inner.set_norm_gemv(false);
    let body = FailingBody::new(inner);
    let model = InputModel {
        inner: &body,
        observer: BodyObserver::new(64),
    };
    let m = metal(&dev);
    eprintln!("PASS_INPUT_REUSED_API actual_api={} prepared_first={prepared_first} norm_gemv=false sampler_native=false owner=public-session", m.commands().actual_command_api());
    // Explicit cold comparators remain full-prefill controls, not a numeric
    // oracle across the warm restore's different prefill split. Forced IDs
    // validate feeds only; physical edge checks establish storage correctness.
    anyhow::ensure!(
        !m.pass_plans_enabled() && m.arena().is_none(),
        "cold planner active"
    );
    for prepared in [false, true] {
        let mut cold = Session::new(&model, model.new_cache(&cfg, &dev)?);
        if prepared {
            reused_request::<true>(&mut cold, &model, &dev, 0, false)?;
        } else {
            reused_request::<false>(&mut cold, &model, &dev, 0, false)?;
        }
        anyhow::ensure!(
            !m.pass_plans_enabled() && m.arena().is_none(),
            "cold installed planner"
        );
    }
    m.enable_pass_plans(ArenaLayout::Packed);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    for (index, prepared) in [prepared_first, !prepared_first].into_iter().enumerate() {
        let reused = if index == 0 { 0 } else { 10 };
        if prepared {
            reused_request::<true>(&mut session, &model, &dev, reused, true)?;
        } else {
            reused_request::<false>(&mut session, &model, &dev, reused, true)?;
        }
    }
    anyhow::ensure!(!m.pass_open(), "warm pass left open");
    let (plans, _) = m
        .pass_plan_report()
        .ok_or_else(|| anyhow::anyhow!("warm plan report absent"))?;
    // Physical route/edge witnesses run first in both orders.
    for input in [PassInput::TokenIds, PassInput::PreparedEmbeddings] {
        anyhow::ensure!(
            plans.iter().any(|(key, covered, ..)| key.kind
                == lloom_tensor::metal_backend::PassKind::Decode { b: 1 }
                && key.input == input
                && *covered > 0),
            "warm route key absent: {input:?}"
        );
    }
    eprintln!("PASS_INPUT_REUSED_COMPLETE prepared_first={prepared_first} recovery=checkpoint-rewind actual_owner_invalid_row=NOT_TESTED");
    Ok(())
}

#[test]
#[ignore = "separately admitted reused-prefix public-session witness; not native owner"]
fn reused_prefix_prepared_then_token_through_openpass() -> anyhow::Result<()> {
    run_reused_order(true)
}

#[test]
#[ignore = "separately admitted reverse reused-prefix public-session witness; not native owner"]
fn reused_prefix_token_then_prepared_through_openpass() -> anyhow::Result<()> {
    run_reused_order(false)
}
