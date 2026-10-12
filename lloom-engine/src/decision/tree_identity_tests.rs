//! N questions over one state, read as one tree (the state prefilled once,
//! each question a branch restored to it), against N single-question
//! requests, on the tiny quantized LFM2 (attention and conv layers both
//! carry state a branch could leak through). CPU, F32, no checkpoint.
use super::{Answer, CausalLabels, CausalRequest, CausalTask, OrderedJson, Question};
use crate::{DecodeConfig, ReadbackArm};
use lloom::models::quantized_lfm2::Model;
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
    let labels = CausalLabels::systemone(&choice(), |code| match code.trim() {
        "A" => Ok(vec![0]),
        "B" => Ok(vec![1]),
        other => lloom_tensor::bail!("unexpected label {other}"),
    })?;
    let mut prompt = STATE.to_vec();
    prompt.extend_from_slice(SUFFIXES[i]);
    CausalTask::new(
        format!("q{i}"),
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
