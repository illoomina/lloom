//! One loaded causal model, one retained state prefix, many answer-slot suffixes.
use super::{CausalReadout, OrderedJson, Question, SystemOneResponse, Temperature};
use lloom::generation::ReadbackStrategy;
use lloom::nn::branch::{
    prefill_trunk, ragged_suffixes, AdoptRetained, BranchSession, RetainState, RetainedTrunk,
};
use lloom::nn::{BatchedCausalLM, CausalLM, DecodeOptions};
use lloom_tensor::{DType, Error, Result, Tensor};

/// A fully tokenized prompt and its selected-token readout, in option order.
/// Tokenization must include the template's special tokens but no implicit
/// tokens beyond that template. Host preparation allocations need caller bounds.
pub struct CausalTask {
    id: String,
    question: Question<OrderedJson>,
    prompt: Vec<u32>,
    readout: CausalReadout,
    temperature: Temperature,
    selected: Vec<u32>,
}
impl CausalTask {
    pub fn new(
        id: String,
        question: Question<OrderedJson>,
        prompt: Vec<u32>,
        readout: CausalReadout,
        temperature: Temperature,
    ) -> Result<Self> {
        question.validate()?;
        if prompt.is_empty() || readout.groups().len() != question.option_count() {
            lloom_tensor::bail!("causal task requires a prompt and one readout group per option")
        }
        let count = readout
            .groups()
            .iter()
            .try_fold(0usize, |n, g| n.checked_add(g.len()))
            .ok_or_else(|| Error::Msg("causal label count overflow".into()))?;
        std::alloc::Layout::array::<u32>(count).map_err(Error::wrap)?;
        let mut selected = Vec::new();
        selected.try_reserve_exact(count).map_err(Error::wrap)?;
        for &id in readout.groups().iter().flatten() {
            if !selected.contains(&id) {
                selected.push(id);
            }
        }
        Ok(Self {
            id,
            question,
            prompt,
            readout,
            temperature,
            selected,
        })
    }
}

/// Prepared ordered tasks over their maximal common token prefix. At least one
/// suffix token is retained even for identical prompts, so every task evaluates
/// its answer slot. The split is token-exact; numeric comparisons must use the
/// same pass partition, not assume cross-partition bit identity.
pub struct CausalRequest {
    tasks: Vec<CausalTask>,
    trunk: usize,
    /// The certified state prefix's length: what the questions share by
    /// construction, where `trunk` is what they happen to share.
    state: usize,
    input_tokens: u64,
}
impl CausalRequest {
    /// The caller supplies the complete tokenized state-bearing template prefix,
    /// certified by its renderer, not a prefix guessed from prompt LCP. This
    /// constructor checks token nesting; it cannot inspect raw state semantics.
    /// An empty certificate is appropriate only when no state bytes are present.
    pub fn new(
        tasks: Vec<CausalTask>,
        state_prefix: &[u32],
        max_prompt_tokens: usize,
    ) -> Result<Self> {
        let first = tasks
            .first()
            .ok_or_else(|| Error::Msg("causal request has no tasks".into()))?;
        let mut trunk = first.prompt.len() - 1;
        let mut input_tokens = 0u64;
        for (i, task) in tasks.iter().enumerate() {
            if !task.prompt.starts_with(state_prefix) || state_prefix.len() >= task.prompt.len() {
                lloom_tensor::bail!("causal state prefix must be shared completely with a nonempty answer-slot suffix")
            }
            if task.prompt.len() > max_prompt_tokens {
                lloom_tensor::bail!("causal prompt exceeds configured context capacity")
            }
            if tasks[..i].iter().any(|other| other.id == task.id) {
                lloom_tensor::bail!("causal request has duplicate question IDs")
            }
            let count = u64::try_from(task.prompt.len()).map_err(Error::wrap)?;
            input_tokens = input_tokens
                .checked_add(count)
                .ok_or_else(|| Error::Msg("causal usage count overflow".into()))?;
            trunk = trunk.min(task.prompt.len() - 1);
            trunk = first.prompt[..trunk]
                .iter()
                .zip(&task.prompt)
                .take_while(|(a, b)| a == b)
                .count();
        }
        Ok(Self {
            tasks,
            trunk,
            state: state_prefix.len(),
            input_tokens,
        })
    }

    pub fn trunk_tokens(&self) -> &[u32] {
        &self.tasks[0].prompt[..self.trunk]
    }

    /// The whole first prompt: every image position lies in it (in the
    /// state, which all prompts share), whichever pass reads it.
    pub(crate) fn splice_prompt(&self) -> &[u32] {
        &self.tasks[0].prompt
    }

    /// The certified state prefix: the trunk a ragged read prefills and
    /// keeps, the same for any questions asked over the same state.
    pub fn state_tokens(&self) -> &[u32] {
        &self.tasks[0].prompt[..self.state]
    }
    /// Media rows must be consumed entirely by the certified state prefix,
    /// which every trunk (the tree's and the ragged read's) holds. Every full
    /// task is checked too: no branch may introduce another row.
    pub(crate) fn validate_shared_token_positions(&self, token: u32, count: usize) -> Result<()> {
        if self
            .state_tokens()
            .iter()
            .filter(|&&id| id == token)
            .count()
            != count
            || self
                .tasks
                .iter()
                .any(|task| task.prompt.iter().filter(|&&id| id == token).count() != count)
        {
            lloom_tensor::bail!("causal image positions must all be in the shared trunk");
        }
        Ok(())
    }

    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    /// Longest complete task prompt, for the loaded engine's capacity check.
    pub fn max_prompt_tokens(&self) -> usize {
        self.tasks
            .iter()
            .map(|task| task.prompt.len())
            .max()
            .unwrap_or(0)
    }

    /// Validate prompts and selected readout IDs before any family model work.
    pub(crate) fn validate_tokens(&self, vocab: usize) -> Result<()> {
        for task in &self.tasks {
            if task
                .prompt
                .iter()
                .chain(&task.selected)
                .any(|id| *id as usize >= vocab)
            {
                lloom_tensor::bail!("causal request token is outside loaded vocabulary")
            }
        }
        Ok(())
    }

    /// Production programmatic executor; no model load, generation or chat-cache
    /// clone. The same borrowed model can also serve chat. The caller serializes
    /// model executor access and reserves the fresh cache, retained recurrent
    /// state, token/logit holds, readout gathers and entire asynchronous pass in
    /// that model's allocation domain before entering. Device identity is not
    /// admission. This lower-level API does not silently install a quota.
    pub fn execute<M, R>(
        &self,
        model: &M,
        cfg: &M::Config,
        readback: &mut R,
        opts: &DecodeOptions,
    ) -> Result<SystemOneResponse>
    where
        M: CausalLM,
        M::Cache: RetainState,
        R: ReadbackStrategy,
    {
        self.execute_with_cache_factory(model, readback, opts, |model, device| {
            model.new_cache(cfg, device)
        })
    }

    /// The same common executor with a family's fresh-cache constructor.
    /// Validate task tokens before invoking `make_cache`, exactly once. The
    /// factory runs in the caller's already reserved pass and must return an
    /// exclusively owned fresh cache, never the held chat cache or its clone.
    /// Capacity is validated when preparing this request; an engine additionally
    /// checks `max_prompt_tokens()` against its actual configured ceiling before
    /// this entry. The factory preserves its capacity/bucket/grow policy.
    pub fn execute_with_cache_factory<M, R>(
        &self,
        model: &M,
        readback: &mut R,
        opts: &DecodeOptions,
        make_cache: impl FnOnce(&M, &lloom_tensor::Device) -> Result<M::Cache>,
    ) -> Result<SystemOneResponse>
    where
        M: CausalLM,
        M::Cache: RetainState,
        R: ReadbackStrategy,
    {
        let vocab = model.vocab_size();
        self.validate_tokens(vocab)?;
        std::alloc::Layout::array::<(String, super::Answer<OrderedJson>)>(self.tasks.len())
            .map_err(Error::wrap)?;
        let mut answers = Vec::new();
        answers
            .try_reserve_exact(self.tasks.len())
            .map_err(Error::wrap)?;
        let device = readback.device().clone();
        let cache = make_cache(model, &device)?;
        let mut session = BranchSession::from_cache(model, cache);
        if let [task] = self.tasks.as_slice() {
            let logits = session.plain(readback, &task.prompt, opts)?;
            answers.push(task.answer(logits, vocab, &device)?);
            return SystemOneResponse::new(answers, self.input_tokens).map_err(Error::wrap);
        }
        session.begin(readback, self.trunk_tokens(), opts)?;
        for task in &self.tasks {
            let logits = session.suffix(readback, &task.prompt[self.trunk..], opts)?;
            answers.push(task.answer(logits, vocab, &device)?);
        }
        SystemOneResponse::new(answers, self.input_tokens).map_err(Error::wrap)
    }

    /// The `decision-ragged` arm (`fast` only; `docs/models/d1.md`): the
    /// certified state prefilled once as the trunk (or taken from `trunks`
    /// when its tokens are the kept ones) and every question's remainder one
    /// row of a ragged batch over it ([`ragged_suffixes`]), where
    /// [`Self::execute_with_cache_factory`] prefills the suffixes one after
    /// another, each from a restore of the trunk. A single question whose
    /// state is not kept is one plain pass, as there.
    ///
    /// `trunk_model` prefills the trunk (a view that splices image rows into
    /// it); `model` runs the rows, which carry no media. `trunks` counts the
    /// read and, with `keep`, holds the state for the next request; a trunk
    /// with image rows is not kept (its tokens do not name its pixels). The
    /// caller holds the same request exclusion and admission as for the tree.
    pub fn execute_ragged<T, M, R>(
        &self,
        trunk_model: &T,
        model: &M,
        readback: &mut R,
        opts: &DecodeOptions,
        mut make_cache: impl FnMut(&M, &lloom_tensor::Device) -> Result<M::Cache>,
        trunks: Option<&TrunkCache<<M::Cache as RetainState>::Retained>>,
        keep: bool,
    ) -> Result<SystemOneResponse>
    where
        T: CausalLM<Cache = M::Cache>,
        M: BatchedCausalLM,
        M::Cache: AdoptRetained,
        R: ReadbackStrategy,
    {
        let vocab = model.vocab_size();
        self.validate_tokens(vocab)?;
        std::alloc::Layout::array::<(String, super::Answer<OrderedJson>)>(self.tasks.len())
            .map_err(Error::wrap)?;
        let mut answers = Vec::new();
        answers
            .try_reserve_exact(self.tasks.len())
            .map_err(Error::wrap)?;
        let device = readback.device().clone();
        let kept = trunks
            .filter(|_| keep)
            .and_then(|t| t.get(self.state_tokens()));
        if let ([task], None) = (self.tasks.as_slice(), &kept) {
            let cache = make_cache(model, &device)?;
            let mut session = BranchSession::from_cache(trunk_model, cache);
            let logits = session.plain(readback, &task.prompt, opts)?;
            answers.push(task.answer(logits, vocab, &device)?);
            return SystemOneResponse::new(answers, self.input_tokens).map_err(Error::wrap);
        }
        let trunk = match kept {
            Some(trunk) => trunk,
            None => {
                let cache = make_cache(model, &device)?;
                let trunk = std::sync::Arc::new(prefill_trunk(
                    trunk_model,
                    cache,
                    readback,
                    self.state_tokens(),
                    opts,
                )?);
                if let Some(trunks) = trunks.filter(|_| keep) {
                    trunks.put(trunk.clone());
                }
                trunk
            }
        };
        let mut suffixes = Vec::new();
        suffixes
            .try_reserve_exact(self.tasks.len())
            .map_err(Error::wrap)?;
        suffixes.extend(self.tasks.iter().map(|task| &task.prompt[self.state..]));
        let logits = ragged_suffixes(
            model,
            &trunk,
            || make_cache(model, &device),
            readback,
            &suffixes,
            opts,
        )?;
        if let Some(trunks) = trunks {
            trunks.note_read(suffixes.len());
        }
        for (task, logits) in self.tasks.iter().zip(&logits) {
            answers.push(task.answer(logits, vocab, &device)?);
        }
        SystemOneResponse::new(answers, self.input_tokens).map_err(Error::wrap)
    }
}

impl CausalTask {
    /// This task's answer from its answer slot's `[vocab]` logits.
    fn answer(
        &self,
        logits: &Tensor,
        vocab: usize,
        device: &lloom_tensor::Device,
    ) -> Result<(String, super::Answer<OrderedJson>)> {
        if logits.dims() != [vocab] {
            lloom_tensor::bail!("causal answer slot has unexpected logit shape")
        }
        let indices = Tensor::new(self.selected.as_slice(), device)?;
        let values = logits
            .index_select(&indices, 0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let probabilities = self.readout.probabilities(
            |id| {
                let i = self
                    .selected
                    .iter()
                    .position(|other| *other == id)
                    .ok_or_else(|| Error::Msg("causal readout omitted a label".into()))?;
                Ok(f64::from(values[i]))
            },
            f64::from(self.temperature),
        )?;
        Ok((
            self.id.clone(),
            self.question.systemone_answer(&probabilities)?,
        ))
    }
}

/// The last trunk a ragged read prefilled, kept for the next request over
/// the same tokens: N single-question requests over one state, and a
/// multi-question one after them, prefill that state once. One entry, so it
/// holds at most one trunk's KV and conv state; a request over another
/// state replaces it.
pub struct TrunkCache<S> {
    entry: parking_lot::Mutex<Option<std::sync::Arc<RetainedTrunk<S>>>>,
    hits: std::sync::atomic::AtomicU64,
    reads: std::sync::atomic::AtomicU64,
    rows: std::sync::atomic::AtomicU64,
}

impl<S> Default for TrunkCache<S> {
    fn default() -> Self {
        Self {
            entry: parking_lot::Mutex::new(None),
            hits: std::sync::atomic::AtomicU64::new(0),
            reads: std::sync::atomic::AtomicU64::new(0),
            rows: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl<S> TrunkCache<S> {
    /// The kept trunk when its tokens are `tokens`. An empty trunk is never
    /// kept: there is nothing to save.
    fn get(&self, tokens: &[u32]) -> Option<std::sync::Arc<RetainedTrunk<S>>> {
        let entry = self.entry.lock();
        let hit = entry
            .as_ref()
            .filter(|trunk| !tokens.is_empty() && trunk.tokens() == tokens)
            .cloned();
        if hit.is_some() {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        hit
    }

    fn put(&self, trunk: std::sync::Arc<RetainedTrunk<S>>) {
        if !trunk.tokens().is_empty() {
            *self.entry.lock() = Some(trunk);
        }
    }

    /// Requests that started from the kept trunk.
    pub fn hits(&self) -> u64 {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn note_read(&self, rows: usize) {
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.rows
            .fetch_add(rows as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// The arm's engagement, for a family's `/health`: requests read as
    /// ragged rows, the rows, the kept-state hits and the kept state's length.
    pub fn report(&self) -> serde_json::Value {
        let load = |n: &std::sync::atomic::AtomicU64| n.load(std::sync::atomic::Ordering::Relaxed);
        serde_json::json!({
            "ragged_reads": load(&self.reads),
            "ragged_rows": load(&self.rows),
            "kept_state_hits": load(&self.hits),
            "kept_state_tokens": self.kept_tokens(),
        })
    }

    /// The kept trunk's length in tokens, 0 when none is kept.
    pub fn kept_tokens(&self) -> usize {
        self.entry.lock().as_ref().map_or(0, |trunk| trunk.pos())
    }
}

/// The reference's token budget for one packed tree (`runner.py`,
/// `SystemOne.token_budget`).
pub const PACK_TOKEN_BUDGET: usize = 65536;

/// Consecutive groups of at most `budget` prompt tokens, every group holding
/// at least one item: the reference's batch planner (`runner.py` `_plan`).
pub fn plan_packs(lengths: &[usize], budget: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut used) = (0, 0usize);
    for (i, &n) in lengths.iter().enumerate() {
        if i > start && used.saturating_add(n) > budget {
            out.push(start..i);
            (start, used) = (i, 0);
        }
        used = used.saturating_add(n);
    }
    if start < lengths.len() {
        out.push(start..lengths.len());
    }
    out
}

/// Several single-question requests read as one tree: their common token
/// prefix is the trunk and each request's remainder a branch, as the
/// reference packs single-question requests (`runner.py` `run`). Each
/// request keeps its own question id and its own `usage`.
pub struct Packed {
    request: CausalRequest,
    parts: Vec<(String, u64)>,
}

impl CausalRequest {
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// The token prefix every part's trunk shares.
    fn common_trunk(parts: &[CausalRequest]) -> &[u32] {
        let first = parts.first().map_or(&[][..], CausalRequest::trunk_tokens);
        let common = parts.iter().fold(first.len(), |n, part| {
            first[..n]
                .iter()
                .zip(part.trunk_tokens())
                .take_while(|(a, b)| a == b)
                .count()
        });
        &first[..common]
    }

    /// Whether `parts` can be read as one tree: two or more, one question
    /// each, sharing at least one token (an empty trunk is not a tree).
    pub fn packable(parts: &[CausalRequest]) -> bool {
        parts.len() >= 2
            && parts.iter().all(|part| part.tasks.len() == 1)
            && !Self::common_trunk(parts).is_empty()
    }

    /// Pack [`Self::packable`] `parts` into one request.
    pub fn pack(parts: Vec<CausalRequest>) -> Result<Packed> {
        if !Self::packable(&parts) {
            lloom_tensor::bail!("decision requests that share no trunk cannot be packed")
        }
        let common = Self::common_trunk(&parts);
        // The packed state is what every part's certified state starts with.
        let state = parts
            .iter()
            .map(|part| part.state)
            .min()
            .unwrap_or(0)
            .min(common.len());
        let common = common[..state].to_vec();
        let max_prompt_tokens = parts
            .iter()
            .map(CausalRequest::max_prompt_tokens)
            .max()
            .unwrap_or(0);
        let mut tasks = Vec::new();
        tasks.try_reserve_exact(parts.len()).map_err(Error::wrap)?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(parts.len()).map_err(Error::wrap)?;
        for (i, part) in parts.into_iter().enumerate() {
            let input_tokens = part.input_tokens;
            let mut task = part
                .tasks
                .into_iter()
                .next()
                .expect("packable: one task per part");
            // Question ids are unique within a request, not across requests.
            ids.push((std::mem::replace(&mut task.id, i.to_string()), input_tokens));
            tasks.push(task);
        }
        Ok(Packed {
            request: Self::new(tasks, &common, max_prompt_tokens)?,
            parts: ids,
        })
    }
}

impl Packed {
    pub fn request(&self) -> &CausalRequest {
        &self.request
    }

    /// The packed tree's answers, one response per packed request, in order.
    pub fn unpack(self, response: SystemOneResponse) -> Result<Vec<SystemOneResponse>> {
        let answers = response.answers();
        if answers.len() != self.parts.len() {
            lloom_tensor::bail!(
                "packed decision returned {} answers for {} requests",
                answers.len(),
                self.parts.len()
            )
        }
        self.parts
            .into_iter()
            .zip(answers)
            .enumerate()
            .map(|(i, ((id, input_tokens), (packed_id, answer)))| {
                if *packed_id != i.to_string() {
                    lloom_tensor::bail!("packed decision answers are out of order")
                }
                SystemOneResponse::new(vec![(id, answer.clone())], input_tokens)
                    .map_err(Error::wrap)
            })
            .collect()
    }
}

#[cfg(feature = "serve")]
impl CausalRequest {
    /// Named-template front end feeding the same loaded-model executor.
    /// Rendering/encoding allocations are covered by caller request admission;
    /// this validates token capacity but is not a byte or allocation limiter.
    pub fn render(
        template: &super::SystemOneTemplate,
        state: &OrderedJson,
        questions: Vec<(String, Question<OrderedJson>, Temperature)>,
        max_prompt_tokens: usize,
        encode_label: impl FnMut(&str) -> Result<Vec<u32>>,
        encode_prompt: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        Self::render_with_media(
            template,
            state,
            questions,
            max_prompt_tokens,
            super::SystemOneMedia::Text,
            encode_label,
            encode_prompt,
        )
    }

    /// Same certified renderer and branch request, with actual supplied media.
    /// The prompt encoder may expand each placeholder using the loaded family's
    /// image layout; it receives WHOLE prefix/full strings, never fragments.
    /// Decoding/encoding tensors and their admission remain family obligations.
    pub fn render_with_media(
        template: &super::SystemOneTemplate,
        state: &OrderedJson,
        questions: Vec<(String, Question<OrderedJson>, Temperature)>,
        max_prompt_tokens: usize,
        media: super::SystemOneMedia<'_>,
        mut encode_label: impl FnMut(&str) -> Result<Vec<u32>>,
        mut encode_prompt: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        if template.kind() != super::SystemOneTemplateKind::D1Causal {
            lloom_tensor::bail!("causal request requires a causal systemone template")
        }
        // Source certificate first, then full byte/token nesting. Tokenizers
        // may merge across the split; refusing that case avoids partial state
        // replay. We do not concatenate separately encoded prompt fragments.
        let state_text = match media {
            super::SystemOneMedia::Text => template.causal_state_prefix(state)?,
            _ => template.causal_state_prefix_with_media(state, media)?,
        };
        let state_prefix = encode_prompt(&state_text)?;
        let mut tasks = Vec::new();
        tasks
            .try_reserve_exact(questions.len())
            .map_err(Error::wrap)?;
        for (id, question, temperature) in questions {
            let labels = super::CausalLabels::systemone(&question, &mut encode_label)?;
            let rendered = template.render(&id, state, &question, labels.texts(), media)?;
            if !rendered.starts_with(&state_text) {
                lloom_tensor::bail!(
                    "causal rendered state prefix differs from full template output"
                )
            }
            let prompt = encode_prompt(&rendered)?;
            tasks.push(CausalTask::new(
                id,
                question,
                prompt,
                labels.into_readout(),
                temperature,
            )?);
        }
        Self::new(tasks, &state_prefix, max_prompt_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Answer, CausalLabels};
    use super::*;
    use lloom::generation::Blocking;
    use lloom::nn::{Model, VarBuilder};
    use lloom_tensor::Device;
    use std::cell::Cell;
    use std::rc::Rc;

    fn question() -> Question<OrderedJson> {
        Question::Choice {
            instructions: OrderedJson::String("choose".into()),
            criteria: vec![("A".into(), None), ("B".into(), None)],
        }
    }
    fn labels(text: &str) -> Result<Vec<u32>> {
        Ok(text.trim().bytes().map(u32::from).collect())
    }
    fn task(id: &str, prompt: Vec<u32>) -> Result<CausalTask> {
        let q = question();
        let labels = CausalLabels::systemone(&q, labels)?;
        CausalTask::new(id.into(), q, prompt, labels.into_readout(), 1.0.try_into()?)
    }

    #[test]
    fn certified_state_prefix_must_fit_every_prompt_before_any_execution() -> Result<()> {
        let r = CausalRequest::new(
            vec![task("z", vec![1, 2, 3])?, task("a", vec![1, 2, 4])?],
            &[1, 2],
            3,
        )?;
        assert_eq!(r.trunk_tokens(), &[1, 2]);
        assert_eq!(r.input_tokens(), 6);
        for (prompts, prefix, max) in [
            (vec![vec![1, 2, 3], vec![1, 4, 3]], vec![1, 2], 3),
            (vec![vec![1, 2]], vec![1, 2], 3),
            (vec![vec![1, 2, 3]], vec![1], 2),
        ] {
            let ts = prompts
                .into_iter()
                .enumerate()
                .map(|(i, p)| task(&i.to_string(), p))
                .collect::<Result<Vec<_>>>()?;
            assert!(CausalRequest::new(ts, &prefix, max).is_err());
        }
        assert!(CausalRequest::new(
            vec![task("same", vec![1, 2])?, task("same", vec![1, 3])?],
            &[1],
            3
        )
        .is_err());
        Ok(())
    }

    #[derive(Default)]
    struct Tiny {
        cache_count: Cell<usize>,
        tokens_fed: Cell<usize>,
        captures: Rc<Cell<usize>>,
    }
    struct Cache {
        tokens: Vec<u32>,
        captures: Rc<Cell<usize>>,
        continuation: bool,
    }
    impl RetainState for Cache {
        type Retained = Vec<u32>;
        fn prepare_branching(&mut self) -> Result<()> {
            self.continuation = true;
            Ok(())
        }
        fn retain_state(&mut self) -> Result<Vec<u32>> {
            self.captures.set(self.captures.get() + 1);
            Ok(self.tokens.clone())
        }
        fn restore_retained(&mut self, saved: &Vec<u32>) -> Result<()> {
            assert!(self.tokens.starts_with(saved));
            self.tokens.truncate(saved.len());
            Ok(())
        }
    }
    impl Model for Tiny {
        type Config = ();
        fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
            Ok(Self::default())
        }
    }
    impl CausalLM for Tiny {
        type Cache = Cache;
        fn new_cache(&self, _: &(), _: &Device) -> Result<Cache> {
            self.cache_count.set(self.cache_count.get() + 1);
            Ok(Cache {
                tokens: vec![],
                captures: self.captures.clone(),
                continuation: false,
            })
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn embed(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn forward_embeds(&self, x: &Tensor, pos: usize, c: &mut Cache) -> Result<Tensor> {
            assert!(
                c.continuation,
                "continuation hook must precede first trunk tile"
            );
            assert_eq!(pos, c.tokens.len());
            let tokens = x.to_vec2::<u32>()?;
            self.tokens_fed.set(self.tokens_fed.get() + tokens[0].len());
            c.tokens.extend_from_slice(&tokens[0]);
            let mut logits = vec![0f32; 256];
            logits[65] = (c.tokens.iter().sum::<u32>() % 7) as f32 * 0.125;
            logits[66] = -0.25;
            Tensor::from_vec(logits, (1, 256), &Device::Cpu)
        }
    }

    #[test]
    fn cache_factory_runs_once_and_only_after_token_validation() -> Result<()> {
        let model = Tiny::default();
        let calls = Cell::new(0);
        let mut rb = Blocking::new(&[255], &Device::Cpu)?;
        let opts = DecodeOptions::new(0);
        let invalid = CausalRequest::new(vec![task("bad", vec![1, 256])?], &[1], 8)?;
        let error = invalid
            .execute_with_cache_factory(&model, &mut rb, &opts, |model, device| {
                calls.set(calls.get() + 1);
                model.new_cache(&(), device)
            })
            .unwrap_err();
        assert!(error.to_string().contains("outside loaded vocabulary"));
        assert_eq!(
            calls.get(),
            0,
            "cache allocated before invalid token refusal"
        );
        assert_eq!(model.cache_count.get(), 0);
        let valid = CausalRequest::new(vec![task("ok", vec![1, 2])?], &[1], 8)?;
        assert_eq!(valid.max_prompt_tokens(), 2);
        let response =
            valid.execute_with_cache_factory(&model, &mut rb, &opts, |model, device| {
                calls.set(calls.get() + 1);
                model.new_cache(&(), device)
            })?;
        assert_eq!(calls.get(), 1);
        assert_eq!(model.cache_count.get(), 1);
        assert_eq!(
            model.captures.get(),
            0,
            "a single question is one plain pass: nothing is retained"
        );
        assert_eq!(model.tokens_fed.get(), 2, "the whole prompt, once");
        assert_eq!(response.answers().len(), 1);
        Ok(())
    }

    #[test]
    fn actual_common_executor_uses_one_loaded_model_and_one_trunk() -> Result<()> {
        let model = Tiny::default();
        let mut rb = Blocking::new(&[255], &Device::Cpu)?;
        let tasks = (0..12u32)
            .map(|q| {
                let mut prompt = vec![2, 3, 4];
                prompt.extend(vec![q + 1; 13]);
                task(&format!("q{q}"), prompt)
            })
            .collect::<Result<Vec<_>>>()?;
        let request = CausalRequest::new(tasks, &[2, 3, 4], 32)?;
        let opts = DecodeOptions {
            prefill_tile: 2,
            ..DecodeOptions::new(1)
        };
        let response = request.execute(&model, &(), &mut rb, &opts)?;
        assert_eq!(model.cache_count.get(), 1);
        assert_eq!(model.captures.get(), 1);
        assert_eq!(model.tokens_fed.get(), 3 + 12 * 13, "shared state replayed");
        assert_eq!(response.input_tokens(), 12 * 16);
        assert_eq!(response.answers().len(), 12);
        for (q, (id, answer)) in response.answers().iter().enumerate() {
            assert_eq!(id, &format!("q{q}"));
            let a = ((9 + 13 * (q as u32 + 1)) % 7) as f64 * 0.125;
            let expected = 1.0 / (1.0 + (-0.25 - a).exp());
            match answer {
                Answer::Choice {
                    choice,
                    probabilities,
                    ..
                } => {
                    assert_eq!(choice, "A");
                    assert_eq!(probabilities.len(), 2);
                    assert!((probabilities[0].1 - expected).abs() < 1e-12);
                    assert!((probabilities[1].1 - (1.0 - expected)).abs() < 1e-12);
                }
                _ => panic!("wrong response kind"),
            }
        }
        Ok(())
    }

    #[cfg(feature = "serve")]
    #[test]
    fn named_renderer_certifies_full_state_not_a_matching_substring() -> Result<()> {
        let t = super::super::SystemOneTemplate::compile(
            include_str!("fixtures/d1-systemone.jinja"),
            super::super::SystemOneTemplateKind::D1Causal,
        )?;
        let state = OrderedJson::from_slice(br#"{"z":"QUESTION:\n first","a":[1,false]}"#)
            .map_err(Error::wrap)?;
        let prefix = t.causal_state_prefix(&state)?;
        assert!(prefix.ends_with("\n\n\nQUESTION:\n"));
        assert!(prefix.contains("\"a\": ["));
        let encode = |s: &str| Ok(s.bytes().map(u32::from).collect::<Vec<_>>());
        let request = CausalRequest::render(
            &t,
            &state,
            vec![
                ("z".into(), question(), 1.0.try_into()?),
                ("a".into(), question(), 1.0.try_into()?),
            ],
            2048,
            labels,
            encode,
        )?;
        assert!(request.trunk_tokens().starts_with(&encode(&prefix)?));
        // Token boundary merge: a separately encoded prefix is not necessarily
        // the full prompt's token prefix. Refuse rather than re-feed state.
        let incompatible = |s: &str| {
            let mut ids = s.bytes().map(u32::from).collect::<Vec<_>>();
            if s.ends_with("\n\n\nQUESTION:\n") {
                *ids.last_mut().unwrap() = 999;
            }
            Ok(ids)
        };
        assert!(CausalRequest::render(
            &t,
            &state,
            vec![("q".into(), question(), 1.0.try_into()?)],
            2048,
            labels,
            incompatible
        )
        .is_err());
        Ok(())
    }

    #[cfg(feature = "serve")]
    #[test]
    fn uncertified_templates_remain_renderable_but_cannot_claim_state_once() -> Result<()> {
        let t = super::super::SystemOneTemplate::compile(
            "{{ instructions }}:{{ state }}:{{ state }}",
            super::super::SystemOneTemplateKind::D1Causal,
        )?;
        let state = OrderedJson::String("retained".into());
        assert_eq!(
            t.render(
                "q",
                &state,
                &question(),
                &["A".into(), "B".into()],
                super::super::SystemOneMedia::Text
            )?,
            "choose:retained:retained"
        );
        let mut calls = 0;
        assert!(CausalRequest::render(
            &t,
            &state,
            vec![("q".into(), question(), 1.0.try_into()?)],
            100,
            labels,
            |_| {
                calls += 1;
                Ok(vec![1])
            }
        )
        .is_err());
        assert_eq!(calls, 0, "uncertified template reached token preparation");
        Ok(())
    }
}

#[cfg(all(test, feature = "serve"))]
mod media_render_tests;
