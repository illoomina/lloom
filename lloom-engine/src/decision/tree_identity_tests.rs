//! N questions over one state, read as one tree (the state prefilled once,
//! each question a branch restored to it), against N single-question
//! requests, on the tiny quantized LFM2 (attention and conv layers both
//! carry state a branch could leak through). CPU, F32, no checkpoint.
use super::{
    Answer, CausalLabels, CausalRequest, CausalTask, OrderedJson, Question, SystemOneResponse,
    TrunkCache,
};
use crate::{DecodeConfig, ReadbackArm};
use lloom::models::quantized_lfm2::{Model, Retained};
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::{Device, Result};
use std::io::Cursor;

#[path = "../../../lloom/src/models/quantized_lfm2/fixture.rs"]
mod fixture;

const STATE: [u32; 5] = [2, 5, 3, 7, 4];
/// Each question's suffix: different lengths, and the second shares its
/// first token with the first, so a leaked branch is not position-neutral.
const SUFFIXES: [&[u32]; 3] = [&[6, 1], &[6], &[3, 2, 6, 5]];

fn choice() -> Question<OrderedJson> {
    Question::Choice {
        instructions: OrderedJson::String("?".into()),
        criteria: vec![("A".into(), None), ("B".into(), None)],
    }
}

fn task(i: usize) -> Result<CausalTask> {
    task_as(&format!("q{i}"), prompt(i))
}

fn prompt(i: usize) -> Vec<u32> {
    let mut prompt = STATE.to_vec();
    prompt.extend_from_slice(SUFFIXES[i]);
    prompt
}

fn task_as(id: &str, prompt: Vec<u32>) -> Result<CausalTask> {
    let labels = CausalLabels::systemone(&choice(), |code| match code.trim() {
        "A" => Ok(vec![0]),
        "B" => Ok(vec![1]),
        other => lloom_tensor::bail!("unexpected label {other}"),
    })?;
    CausalTask::new(
        id.into(),
        choice(),
        prompt,
        labels.into_readout(),
        1.0.try_into()?,
    )
}

fn probability_of_a(model: &Model, tasks: Vec<CausalTask>) -> Result<Vec<f64>> {
    let request = CausalRequest::new(tasks, &STATE, 32)?;
    let device = Device::Cpu;
    let mut readback = lloom::generation::Blocking::new(&[u32::MAX], &device)?;
    let options = DecodeConfig::default().options(0, ReadbackArm::Blocking, 0);
    let response =
        request.execute_with_cache_factory(model, &mut readback, &options, |m, _| m.new_cache())?;
    Ok(response
        .answers()
        .iter()
        .map(|(_, answer)| match answer {
            Answer::Choice { probabilities, .. } => probabilities[0].1,
            other => panic!("a choice question answered {other:?}"),
        })
        .collect())
}

/// The tree equals the separate requests within F32 tolerance (the two
/// partition the prefill differently, so bit identity is not promised), and
/// the questions' answers differ from each other, so a branch reading
/// another's state would move them.
#[test]
fn n_questions_as_one_tree_equal_n_single_requests() -> Result<()> {
    let bytes = fixture::fixture(None)?;
    let mut reader = Cursor::new(&bytes);
    let model = Model::from_gguf(
        gguf_file::Content::read(&mut reader)?,
        &mut reader,
        &Device::Cpu,
    )?;
    let tree = probability_of_a(&model, (0..3).map(task).collect::<Result<_>>()?)?;
    let single = (0..3)
        .map(|i| Ok(probability_of_a(&model, vec![task(i)?])?[0]))
        .collect::<Result<Vec<_>>>()?;
    eprintln!("tree   P(A) = {tree:?}");
    eprintln!("single P(A) = {single:?}");
    for (i, (t, s)) in tree.iter().zip(&single).enumerate() {
        assert!(
            (t - s).abs() < 1e-5,
            "question {i}: tree P(A) {t} != single-request P(A) {s}"
        );
    }
    let spread = single.iter().cloned().fold(f64::MIN, f64::max)
        - single.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 1e-3,
        "the questions' answers are too close ({single:?}) for a leaked branch to be seen"
    );
    Ok(())
}

fn model() -> Result<Model> {
    let bytes = fixture::fixture(None)?;
    let mut reader = Cursor::new(&bytes);
    Model::from_gguf(
        gguf_file::Content::read(&mut reader)?,
        &mut reader,
        &Device::Cpu,
    )
}

/// Request `i` alone: one question over the state, its id `q` in every
/// request (ids are unique within a request, not across requests).
fn single_request(i: usize) -> Result<CausalRequest> {
    CausalRequest::new(vec![task_as("q", prompt(i))?], &STATE, 32)
}

fn run(model: &Model, request: &CausalRequest) -> Result<super::SystemOneResponse> {
    let device = Device::Cpu;
    let mut readback = lloom::generation::Blocking::new(&[u32::MAX], &device)?;
    let options = DecodeConfig::default().options(0, ReadbackArm::Blocking, 0);
    request.execute_with_cache_factory(model, &mut readback, &options, |m, _| m.new_cache())
}

fn p_a(response: &super::SystemOneResponse) -> f64 {
    match &response.answers()[0].1 {
        Answer::Choice { probabilities, .. } => probabilities[0].1,
        other => panic!("a choice question answered {other:?}"),
    }
}

/// Three single-question requests packed into one tree, as `serve` packs
/// requests that waited together: each gets back its own answer (within F32
/// tolerance of its read alone), its own question id and its own usage.
#[test]
fn packed_single_question_requests_each_get_their_own_answer() -> Result<()> {
    let model = model()?;
    let alone = (0..3)
        .map(|i| run(&model, &single_request(i)?))
        .collect::<Result<Vec<_>>>()?;
    let parts = (0..3).map(single_request).collect::<Result<Vec<_>>>()?;
    assert!(CausalRequest::packable(&parts));
    let packed = CausalRequest::pack(parts)?;
    assert_eq!(
        packed.request().trunk_tokens(),
        STATE,
        "the requests' shared trunk"
    );
    let response = run(&model, packed.request())?;
    let answers = packed.unpack(response)?;
    assert_eq!(answers.len(), 3);
    for (i, (packed, alone)) in answers.iter().zip(&alone).enumerate() {
        assert_eq!(packed.answers().len(), 1);
        assert_eq!(packed.answers()[0].0, "q");
        assert_eq!(packed.input_tokens(), alone.input_tokens());
        let (p, a) = (p_a(packed), p_a(alone));
        assert!(
            (p - a).abs() < 1e-5,
            "request {i}: packed P(A) {p} != alone {a}"
        );
    }
    let alone: Vec<f64> = alone.iter().map(p_a).collect();
    let spread = alone.iter().cloned().fold(f64::MIN, f64::max)
        - alone.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 1e-3,
        "the requests' answers are too close ({alone:?}) to tell apart"
    );
    Ok(())
}

/// Requests sharing no token, or one with two questions, are not packed.
#[test]
fn only_single_question_requests_with_a_shared_trunk_pack() -> Result<()> {
    let one = || single_request(0);
    assert!(!CausalRequest::packable(&[one()?]));
    let two_questions = CausalRequest::new(vec![task(0)?, task(1)?], &STATE, 32)?;
    assert!(!CausalRequest::packable(&[one()?, two_questions]));
    let elsewhere = CausalRequest::new(vec![task_as("q", vec![9, 1, 2])?], &[], 32)?;
    assert!(!CausalRequest::packable(&[one()?, elsewhere]));
    assert!(CausalRequest::pack(vec![one()?]).is_err());
    Ok(())
}

/// `decide_together` (the families' batch) packs the text-only
/// single-question requests only when asked (`fast`), passes a request's
/// refusal through in place, and without packing answers each request
/// bit-identically to its read alone.
#[cfg(feature = "serve")]
#[test]
fn a_batch_packs_only_when_asked_and_keeps_each_answer_in_place() -> Result<()> {
    use crate::app::lfm2::decision_media::{decide_together, Prepared};
    use crate::decision::SystemOneError;
    let model = model()?;
    let alone = (0..3)
        .map(|i| run(&model, &single_request(i)?))
        .collect::<Result<Vec<_>>>()?;
    let bodies: Vec<OrderedJson> = (0..4)
        .map(|n| OrderedJson::Number(serde_json::Number::from(n)))
        .collect();
    let refs: Vec<&OrderedJson> = bodies.iter().collect();
    for pack in [true, false] {
        let mut runs = Vec::new();
        let answers = decide_together(
            &refs,
            pack,
            |body| match body {
                OrderedJson::Number(n) if n.as_u64() == Some(3) => Err(
                    SystemOneError::InvalidRequest("request 3 is refused".into()),
                ),
                OrderedJson::Number(n) => single_request(n.as_u64().unwrap() as usize)
                    .map(Prepared::Text)
                    .map_err(|e| SystemOneError::InvalidRequest(e.to_string())),
                _ => unreachable!(),
            },
            |request| {
                runs.push(request.task_count());
                run(&model, request)
            },
            |_| unreachable!("no image request"),
        );
        assert_eq!(
            runs,
            if pack { vec![3] } else { vec![1, 1, 1] },
            "pack={pack}"
        );
        assert!(
            matches!(&answers[3], Err(SystemOneError::InvalidRequest(m)) if m.contains("request 3"))
        );
        for (i, answer) in answers[..3].iter().enumerate() {
            let answer = answer.as_ref().expect("an answer");
            let (p, a) = (p_a(answer), p_a(&alone[i]));
            if pack {
                assert!(
                    (p - a).abs() < 1e-5,
                    "request {i}: packed P(A) {p} != alone {a}"
                );
            } else {
                assert_eq!(
                    p.to_bits(),
                    a.to_bits(),
                    "request {i} moved without packing"
                );
            }
        }
    }
    Ok(())
}

fn run_ragged(
    model: &Model,
    request: &CausalRequest,
    trunks: Option<&TrunkCache<Retained>>,
) -> Result<SystemOneResponse> {
    let device = Device::Cpu;
    let mut readback = lloom::generation::Blocking::new(&[u32::MAX], &device)?;
    let options = DecodeConfig::default().options(0, ReadbackArm::Blocking, 0);
    request.execute_ragged(
        model,
        model,
        &mut readback,
        &options,
        |m, _| m.new_cache(),
        trunks,
    )
}

fn all_p_a(response: &SystemOneResponse) -> Vec<f64> {
    response
        .answers()
        .iter()
        .map(|(_, answer)| match answer {
            Answer::Choice { probabilities, .. } => probabilities[0].1,
            other => panic!("a choice question answered {other:?}"),
        })
        .collect()
}

/// The ragged read (the state prefilled once, the three questions' rows in
/// one pass over it) equals the tree and three single requests within F32
/// tolerance: the three partition the work differently, so bits are not
/// promised. The answers differ from each other, so a row reading another
/// row's tokens or state would move them.
#[test]
fn n_questions_as_ragged_rows_equal_the_tree_and_n_single_requests() -> Result<()> {
    let model = model()?;
    let request = || CausalRequest::new((0..3).map(task).collect::<Result<_>>()?, &STATE, 32);
    let ragged = all_p_a(&run_ragged(&model, &request()?, None)?);
    let tree = all_p_a(&run(&model, &request()?)?);
    let single = (0..3)
        .map(|i| Ok(p_a(&run(&model, &single_request(i)?)?)))
        .collect::<Result<Vec<_>>>()?;
    eprintln!("ragged P(A) = {ragged:?}");
    eprintln!("tree   P(A) = {tree:?}");
    eprintln!("single P(A) = {single:?}");
    for i in 0..3 {
        assert!(
            (ragged[i] - tree[i]).abs() < 1e-5 && (ragged[i] - single[i]).abs() < 1e-5,
            "question {i}: ragged P(A) {} against tree {} and single {}",
            ragged[i],
            tree[i],
            single[i]
        );
    }
    let spread = single.iter().cloned().fold(f64::MIN, f64::max)
        - single.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread > 1e-3, "answers too close to tell apart: {single:?}");
    Ok(())
}

/// A ragged read keeps its state; the next requests over the same state start
/// from it (one question each, no state prefill) and answer as they would
/// alone; a request over another state does not take it, and a ragged read
/// over that state replaces it.
#[test]
fn a_kept_state_serves_the_requests_over_it_and_no_other() -> Result<()> {
    let model = model()?;
    let trunks = TrunkCache::<Retained>::default();
    let multi = CausalRequest::new((0..3).map(task).collect::<Result<_>>()?, &STATE, 32)?;
    let multi = all_p_a(&run_ragged(&model, &multi, Some(&trunks))?);
    assert_eq!(trunks.kept_tokens(), STATE.len());
    assert_eq!(trunks.hits(), 0);
    for i in 0..3 {
        let kept = p_a(&run_ragged(&model, &single_request(i)?, Some(&trunks))?);
        let alone = p_a(&run(&model, &single_request(i)?)?);
        assert_eq!(
            trunks.hits(),
            i as u64 + 1,
            "request {i} did not start from the kept state"
        );
        assert!(
            (kept - alone).abs() < 1e-5 && (kept - multi[i]).abs() < 1e-5,
            "request {i}: from the kept state P(A) {kept}, alone {alone}, in the batch {}",
            multi[i]
        );
    }
    // Another state: the same suffix lengths, one state token changed.
    let other: [u32; 5] = [2, 5, 3, 7, 6];
    let elsewhere = |i: usize| {
        let mut prompt = other.to_vec();
        prompt.extend_from_slice(SUFFIXES[i]);
        task_as(&format!("q{i}"), prompt)
    };
    let one = CausalRequest::new(vec![elsewhere(0)?], &other, 32)?;
    let got = p_a(&run_ragged(&model, &one, Some(&trunks))?);
    let alone = p_a(&run(&model, &one)?);
    assert_eq!(
        trunks.hits(),
        3,
        "a request over another state took the kept one"
    );
    assert!(
        (got - alone).abs() < 1e-5,
        "another state: {got} != alone {alone}"
    );
    let two = CausalRequest::new(vec![elsewhere(0)?, elsewhere(1)?], &other, 32)?;
    run_ragged(&model, &two, Some(&trunks))?;
    assert_eq!(trunks.hits(), 3);
    run_ragged(&model, &single_request(0)?, Some(&trunks))?;
    assert_eq!(trunks.hits(), 3, "the replaced state was still served");
    Ok(())
}
