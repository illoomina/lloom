//! One loaded causal model, one retained state prefix, many answer-slot suffixes.
use super::{CausalReadout, OrderedJson, Question, SystemOneResponse, Temperature};
use lloom::generation::ReadbackStrategy;
use lloom::nn::branch::{BranchSession, RetainState};
use lloom::nn::{CausalLM, DecodeOptions};
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
            input_tokens,
        })
    }

    pub fn trunk_tokens(&self) -> &[u32] {
        &self.tasks[0].prompt[..self.trunk]
    }
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
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
        let vocab = model.vocab_size();
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
        std::alloc::Layout::array::<(String, super::Answer<OrderedJson>)>(self.tasks.len())
            .map_err(Error::wrap)?;
        let mut answers = Vec::new();
        answers
            .try_reserve_exact(self.tasks.len())
            .map_err(Error::wrap)?;
        let device = readback.device().clone();
        let mut session = BranchSession::new(model, cfg, &device)?;
        session.begin(readback, self.trunk_tokens(), opts)?;
        for task in &self.tasks {
            let logits = session.suffix(readback, &task.prompt[self.trunk..], opts)?;
            if logits.dims() != [vocab] {
                lloom_tensor::bail!("causal answer slot has unexpected logit shape")
            }
            let indices = Tensor::new(task.selected.as_slice(), &device)?;
            let values = logits
                .index_select(&indices, 0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let probabilities = task.readout.probabilities(
                |id| {
                    let i = task
                        .selected
                        .iter()
                        .position(|other| *other == id)
                        .ok_or_else(|| Error::Msg("causal readout omitted a label".into()))?;
                    Ok(f64::from(values[i]))
                },
                f64::from(task.temperature),
            )?;
            answers.push((
                task.id.clone(),
                task.question.systemone_answer(&probabilities)?,
            ));
        }
        SystemOneResponse::new(answers, self.input_tokens).map_err(Error::wrap)
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
        mut encode_label: impl FnMut(&str) -> Result<Vec<u32>>,
        mut encode_prompt: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        if template.kind() != super::SystemOneTemplateKind::D1Causal {
            lloom_tensor::bail!("causal request requires a causal systemone template")
        }
        // Source certificate first, then full byte/token nesting. Tokenizers
        // may merge across the split; refusing that case avoids partial state
        // replay. We do not concatenate separately encoded prompt fragments.
        let state_text = template.causal_state_prefix(state)?;
        let state_prefix = encode_prompt(&state_text)?;
        let mut tasks = Vec::new();
        tasks
            .try_reserve_exact(questions.len())
            .map_err(Error::wrap)?;
        for (id, question, temperature) in questions {
            let labels = super::CausalLabels::systemone(&question, &mut encode_label)?;
            let rendered = template.render(
                &id,
                state,
                &question,
                labels.texts(),
                super::SystemOneMedia::Text,
            )?;
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
