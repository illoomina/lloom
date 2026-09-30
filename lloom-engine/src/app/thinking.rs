//! Thinking on or off (`lloom chat --think` / `--no-think`, a request's
//! `enable_thinking`), on any family.
//!
//! A template that reads `enable_thinking` switches itself (Qwen4-exp). One
//! that does not has one behaviour, read from its generation prompt: LFM2.5's
//! opens the thinking block on every turn, so it thinks unless told
//! otherwise. There the engine turns thinking off by closing that block in
//! the prompt with the template's own end marker ([`ThinkTags::end`], read by
//! rendering the template), so the model continues after an empty block, as
//! it does after a thinking budget of 0 (`budgets::thinking`). Thinking on
//! where the prompt leaves no block open and nothing can open one is refused.
//! Asked neither way, the prompt is the template's own.
//! (`docs/decode-driver-design.md`, Stage 7's thinking switch.)

use super::served::Served;
use super::template::{ThinkTags, ENABLE_THINKING};
use serde_json::{Map, Value};

/// Why thinking on cannot apply to a prompt.
pub const CANNOT_OPEN: &str = "the chat template leaves no thinking block open for this prompt \
     and has no enable_thinking switch that opens one";

/// Who decided the prompt's thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum By {
    /// The template, by its switch or its own behaviour.
    Template,
    /// The engine, which closed the block the template left open.
    Engine,
}

impl By {
    pub fn name(self) -> &'static str {
        match self {
            By::Template => "template",
            By::Engine => "engine",
        }
    }
}

/// A prompt's thinking, as it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thinking {
    /// The prompt ends inside an open thinking block.
    pub on: bool,
    pub by: By,
}

impl Thinking {
    pub fn name(self) -> &'static str {
        if self.on {
            "on"
        } else {
            "off"
        }
    }
}

/// The template's `enable_thinking` for `want`: set where the template reads
/// it, removed where it does not (the engine applies it there).
pub fn set_kwarg(engine: &dyn Served, want: Option<bool>, kwargs: &mut Map<String, Value>) {
    kwargs.remove(ENABLE_THINKING);
    if let Some(w) = want {
        if engine.template_reads(ENABLE_THINKING) == Some(true) {
            kwargs.insert(ENABLE_THINKING.into(), Value::Bool(w));
        }
    }
}

/// Apply `want` to a rendered and encoded prompt: off closes a block the
/// prompt leaves open (the end marker appended), on requires one open. `Err`
/// says why thinking on cannot apply.
pub fn apply(
    want: Option<bool>,
    tags: Option<&ThinkTags>,
    ids: &mut Vec<u32>,
) -> Result<Thinking, &'static str> {
    let open = tags.filter(|t| t.open_at_end(ids).is_some());
    match (want, open) {
        (Some(false), Some(t)) => {
            ids.push(t.end);
            Ok(Thinking {
                on: false,
                by: By::Engine,
            })
        }
        (Some(true), None) => Err(CANNOT_OPEN),
        (_, open) => Ok(Thinking {
            on: open.is_some(),
            by: By::Template,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAGS: ThinkTags = ThinkTags {
        start: 1,
        end: 2,
        tool_call: Some(3),
    };

    /// Off closes an open block with the end marker, on leaves it, neither
    /// is the prompt as rendered; a closed prompt stays as it is off or
    /// unasked, and on is refused; a template with no tags is never open.
    #[test]
    fn off_closes_the_open_block_and_on_needs_one() {
        let run = |want, tags: Option<&ThinkTags>, ids: &[u32]| {
            let mut ids = ids.to_vec();
            apply(want, tags, &mut ids).map(|t| (t, ids))
        };
        let on = Thinking {
            on: true,
            by: By::Template,
        };
        let off = Thinking {
            on: false,
            by: By::Template,
        };
        let open = [9, 1];
        assert_eq!(
            run(Some(false), Some(&TAGS), &open),
            Ok((
                Thinking {
                    on: false,
                    by: By::Engine
                },
                vec![9, 1, 2]
            ))
        );
        assert_eq!(run(Some(true), Some(&TAGS), &open), Ok((on, open.to_vec())));
        assert_eq!(run(None, Some(&TAGS), &open), Ok((on, open.to_vec())));
        let closed = [9, 1, 4, 2, 4];
        assert_eq!(
            run(Some(false), Some(&TAGS), &closed),
            Ok((off, closed.to_vec()))
        );
        assert_eq!(run(None, Some(&TAGS), &closed), Ok((off, closed.to_vec())));
        assert_eq!(run(Some(true), Some(&TAGS), &closed), Err(CANNOT_OPEN));
        assert_eq!(run(Some(false), None, &open), Ok((off, open.to_vec())));
        assert_eq!(run(Some(true), None, &open), Err(CANNOT_OPEN));
    }
}
