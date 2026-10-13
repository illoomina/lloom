//! The two token budgets a turn or a request carries, resolved against its
//! prompt: the thinking budget against the family's reasoning tags (the
//! decode loop enforces it, `lloom::generation::budget`), and the prompt
//! budget against the prompt's length, applied before the prompt reaches the
//! engine so the chunked prefill and the prefix reuse see the prompt that
//! runs (`docs/decode-driver-design.md`, Stage 7's budgets).

use super::template::ThinkTags;
use lloom::generation::ThinkingBudget;
use serde_json::Value;

/// Why a thinking budget does not apply to a prompt.
pub const NO_THINKING_BLOCK: &str = "the chat template writes no thinking block";
pub const NOT_THINKING: &str = "the prompt leaves no thinking block open";
pub const BATCHED: &str =
    "this process decodes through the batch scheduler (--batch N), which feeds no forced token";

/// The budget of `budget` tokens for `prompt`, or why it does not apply:
/// the decode path cannot feed it (`refused`), the template writes no block,
/// or the prompt leaves none open. The forced tail is `message` (its ids,
/// which must not close the block) and the template's end marker.
pub fn thinking(
    refused: Option<&'static str>,
    tags: Option<&ThinkTags>,
    prompt: &[u32],
    budget: usize,
    message: &[u32],
) -> anyhow::Result<Result<ThinkingBudget, &'static str>> {
    if let Some(why) = refused {
        return Ok(Err(why));
    }
    let Some(tags) = tags else {
        return Ok(Err(NO_THINKING_BLOCK));
    };
    let Some(prior) = tags.open_at_end(prompt) else {
        return Ok(Err(NOT_THINKING));
    };
    let mut forced = message.to_vec();
    forced.push(tags.end);
    ThinkingBudget::new(budget, tags.close_ids(), forced)
        .map(|b| Ok(b.with_prior(prior)))
        .map_err(|e| anyhow::anyhow!("the thinking budget's message: {e}"))
}

/// Over the prompt budget: refuse, or truncate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    Refuse,
    Truncate,
}

impl PromptMode {
    pub fn parse(s: &str) -> Self {
        if s == "truncate" {
            PromptMode::Truncate
        } else {
            PromptMode::Refuse
        }
    }
}

/// What a truncation dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dropped {
    pub tokens: usize,
    pub messages: usize,
}

/// The refusal of a prompt over its budget.
pub fn over_budget(len: usize, budget: usize, how: &str) -> String {
    format!("the prompt is {len} tokens, over the prompt budget of {budget} ({how})")
}

/// A rendered-as-given prompt (no template) cut to its last `budget`
/// tokens, the leading BOS kept: the only structure a raw prompt has.
pub fn truncate_raw(ids: &[u32], budget: usize, bos: Option<u32>) -> (Vec<u32>, Dropped) {
    if ids.len() <= budget {
        return (ids.to_vec(), Dropped::default());
    }
    let keep_bos = bos.is_some() && ids.first() == bos.as_ref() && budget > 1;
    let head = usize::from(keep_bos);
    let mut out = ids[..head].to_vec();
    out.extend_from_slice(&ids[ids.len() - (budget - head)..]);
    (
        out,
        Dropped {
            tokens: ids.len() - budget,
            messages: 0,
        },
    )
}

/// A conversation fitted to a prompt budget: the messages kept, their ids,
/// and what was dropped.
#[derive(Debug)]
pub struct Fitted {
    pub messages: Vec<Value>,
    pub ids: Vec<u32>,
    pub dropped: Dropped,
}

/// The conversation fitted to `budget` tokens by dropping its oldest
/// messages whole: the leading system message is kept, and the kept part
/// starts at a user message, so what runs is the template's own rendering of
/// a shorter conversation. `encode` renders and encodes a candidate.
/// `Ok(None)` when even the last user turn does not fit.
pub fn fit_messages(
    messages: &[Value],
    budget: usize,
    full_len: usize,
    mut encode: impl FnMut(&[Value]) -> anyhow::Result<Vec<u32>>,
) -> anyhow::Result<Option<Fitted>> {
    fn role(m: &Value) -> &str {
        m.get("role").and_then(Value::as_str).unwrap_or("")
    }
    let head = usize::from(messages.first().is_some_and(|m| role(m) == "system"));
    for drop in 1..messages.len().saturating_sub(head) {
        let first = &messages[head + drop];
        if role(first) != "user" {
            continue;
        }
        let mut kept = messages[..head].to_vec();
        kept.extend_from_slice(&messages[head + drop..]);
        let ids = encode(&kept)?;
        if ids.len() <= budget {
            let dropped = Dropped {
                tokens: full_len - ids.len(),
                messages: drop,
            };
            return Ok(Some(Fitted {
                messages: kept,
                ids,
                dropped,
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A raw prompt keeps its BOS and its last tokens; one within the
    /// budget is untouched; a budget of one keeps the last token alone.
    #[test]
    fn a_raw_prompt_keeps_its_bos_and_its_tail() {
        let ids = [1, 10, 11, 12, 13, 14];
        assert_eq!(
            truncate_raw(&ids, 4, Some(1)),
            (
                vec![1, 12, 13, 14],
                Dropped {
                    tokens: 2,
                    messages: 0
                }
            )
        );
        assert_eq!(truncate_raw(&ids, 4, None).0, vec![11, 12, 13, 14]);
        assert_eq!(truncate_raw(&ids, 4, Some(99)).0, vec![11, 12, 13, 14]);
        assert_eq!(truncate_raw(&ids, 6, Some(1)).0, ids.to_vec());
        assert_eq!(truncate_raw(&ids, 1, Some(1)).0, vec![14]);
    }

    fn msg(role: &str, n: usize) -> Value {
        json!({ "role": role, "content": "x".repeat(n) })
    }

    /// One token per content character plus one per message: the oldest
    /// turns go first, whole, the system message stays, and the kept part
    /// starts at a user turn; when the last user turn alone is over, nothing
    /// fits.
    #[test]
    fn a_conversation_drops_its_oldest_turns_whole() {
        let conv = vec![
            msg("system", 2),
            msg("user", 5),
            msg("assistant", 5),
            msg("user", 3),
            msg("assistant", 3),
            msg("user", 1),
        ];
        let encode = |ms: &[Value]| -> anyhow::Result<Vec<u32>> {
            let n: usize = ms
                .iter()
                .map(|m| 1 + m["content"].as_str().unwrap().len())
                .sum();
            Ok(vec![0; n])
        };
        let full = encode(&conv).unwrap().len();
        assert_eq!(full, 25);
        let Fitted {
            messages: kept,
            ids,
            dropped,
        } = fit_messages(&conv, 13, full, encode).unwrap().unwrap();
        assert_eq!(
            kept,
            vec![
                conv[0].clone(),
                conv[3].clone(),
                conv[4].clone(),
                conv[5].clone()
            ]
        );
        assert_eq!(ids.len(), 13);
        assert_eq!(
            dropped,
            Dropped {
                tokens: 12,
                messages: 2
            }
        );
        let kept = fit_messages(&conv, 12, full, encode)
            .unwrap()
            .unwrap()
            .messages;
        assert_eq!(kept, vec![conv[0].clone(), conv[5].clone()]);
        assert!(fit_messages(&conv, 4, full, encode).unwrap().is_none());
        let no_system = &conv[1..];
        let kept = fit_messages(no_system, 10, 22, encode)
            .unwrap()
            .unwrap()
            .messages;
        assert_eq!(
            kept,
            vec![conv[3].clone(), conv[4].clone(), conv[5].clone()]
        );
    }

    /// The forced tail is the message then the end marker; a prompt with no
    /// open block, or a template with no tags, says why it does not apply.
    #[test]
    fn a_thinking_budget_applies_only_inside_an_open_block() {
        let tags = ThinkTags {
            start: 1,
            end: 2,
            tool_call: Some(3),
        };
        let b = thinking(None, Some(&tags), &[9, 1], 4, &[7, 8])
            .unwrap()
            .unwrap();
        assert_eq!(b.forced(), &[7, 8, 2]);
        assert_eq!(b.prior(), 0);
        let b = thinking(None, Some(&tags), &[1, 9, 9], 4, &[])
            .unwrap()
            .unwrap();
        assert_eq!(b.prior(), 2);
        assert_eq!(b.close_ids(), &[2, 3]);
        assert_eq!(b.budget(), 4);
        assert_eq!(
            thinking(None, Some(&tags), &[1, 9, 2], 4, &[]).unwrap(),
            Err(NOT_THINKING)
        );
        assert_eq!(
            thinking(None, None, &[1], 4, &[]).unwrap(),
            Err(NO_THINKING_BLOCK)
        );
        assert!(thinking(None, Some(&tags), &[1], 4, &[7, 2]).is_err());
        assert_eq!(
            thinking(Some(BATCHED), Some(&tags), &[1], 4, &[]).unwrap(),
            Err(BATCHED)
        );
    }
}
