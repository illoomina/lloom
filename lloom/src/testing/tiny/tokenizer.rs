//! The tiny models' tokenizer (`docs/porting-models.md` §21.8, INV-6): the
//! 256 bytes, then the family's added tokens in the family's order, with a
//! chat template of lloom's own for the family beside it (`templates/`: the
//! checkpoints' turn tokens, none of their text). A byte-level BPE with no
//! merges, so any text round-trips and no checkpoint's vocabulary is needed.

/// One added token, as the family's `tokenizer.json` lists it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddedToken {
    pub content: &'static str,
    /// Special: skipped on decode with `skip_special_tokens`.
    pub special: bool,
}

const fn tok(content: &'static str, special: bool) -> AddedToken {
    AddedToken { content, special }
}

/// Qwen3.8-Flash-Next's added tokens, in id order (its `tokenizer.json`,
/// snapshot `de4b8e4d`).
pub const QWEN4EXP_ADDED: &[AddedToken] = &[
    tok("<|endoftext|>", true),
    tok("<|im_start|>", true),
    tok("<|im_end|>", true),
    tok("<|object_ref_start|>", true),
    tok("<|object_ref_end|>", true),
    tok("<|box_start|>", true),
    tok("<|box_end|>", true),
    tok("<|quad_start|>", true),
    tok("<|quad_end|>", true),
    tok("<|vision_start|>", true),
    tok("<|vision_end|>", true),
    tok("<|vision_pad|>", true),
    tok("<|image_pad|>", true),
    tok("<|video_pad|>", true),
    tok("<tool_call>", false),
    tok("</tool_call>", false),
    tok("<|fim_prefix|>", false),
    tok("<|fim_middle|>", false),
    tok("<|fim_suffix|>", false),
    tok("<|fim_pad|>", false),
    tok("<|repo_name|>", false),
    tok("<|file_sep|>", false),
    tok("<tool_response>", false),
    tok("</tool_response>", false),
    tok("<think>", false),
    tok("</think>", false),
    tok("<|audio_start|>", true),
    tok("<|audio_end|>", true),
    tok("<tts_pad>", true),
    tok("<tts_text_bos>", true),
    tok("<tts_text_eod>", true),
    tok("<tts_text_bos_single>", true),
    tok("<|audio_pad|>", true),
];

/// LFM2.5's added tokens before the image grid, in id order (its
/// `tokenizer.json`, snapshot `654f9463`); [`lfm2_added`] inserts the
/// grid's 100 `<|img_row_R_col_C|>` tokens after `<image>`, where the
/// checkpoint has them.
pub const LFM2_ADDED_HEAD: &[AddedToken] = &[
    tok("<|pad|>", true),
    tok("<|startoftext|>", true),
    tok("<|endoftext|>", true),
    tok("<|fim_pre|>", true),
    tok("<|fim_mid|>", true),
    tok("<|fim_suf|>", true),
    tok("<|im_start|>", true),
    tok("<|im_end|>", true),
    tok("<think>", false),
    tok("</think>", false),
    tok("<|tool_list_start|>", true),
    tok("<|tool_list_end|>", true),
    tok("<|tool_call_start|>", false),
    tok("<|tool_call_end|>", false),
    tok("<image>", true),
];

/// LFM2.5's added tokens after the image grid.
pub const LFM2_ADDED_TAIL: &[AddedToken] = &[
    tok("<|img_thumbnail|>", true),
    tok("<|image_start|>", true),
    tok("<|image_end|>", true),
    tok("<|image_split|>", true),
    tok("<|audio_start|>", true),
    tok("<|text_start|>", true),
    tok("<|text_end|>", true),
    tok("<|mixed_start|>", true),
    tok("<|mixed_end|>", true),
];

/// LFM2.5's added tokens, all 124, in id order.
pub fn lfm2_added() -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = LFM2_ADDED_HEAD
        .iter()
        .map(|t| (t.content.to_string(), t.special))
        .collect();
    for r in 1..=10 {
        for c in 1..=10 {
            out.push((format!("<|img_row_{r}_col_{c}|>"), true));
        }
    }
    out.extend(
        LFM2_ADDED_TAIL
            .iter()
            .map(|t| (t.content.to_string(), t.special)),
    );
    out
}

/// The byte ↔ character map byte-level BPE uses (GPT-2's `bytes_to_unicode`):
/// printable bytes map to themselves, the rest to `U+0100` onwards.
pub fn byte_char(b: u8) -> char {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable(b) {
        return b as char;
    }
    let n = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).expect("a valid code point")
}

/// The `tokenizer.json` text: ids `0..256` are the bytes, `256..` the added
/// tokens. The pre-tokenizer, decoder and post-processor are the families'
/// `ByteLevel` without a regex split (every byte is its own token).
pub fn tokenizer_json(added: &[(String, bool)]) -> String {
    use serde_json::{json, Map, Value};
    let mut vocab = Map::new();
    for b in 0..=255u8 {
        vocab.insert(byte_char(b).to_string(), json!(b as u32));
    }
    let added: Vec<Value> = added
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            json!({
                "id": 256 + i,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": special,
            })
        })
        .collect();
    let byte_level = json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false,
    });
    let t = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": [],
        },
    });
    serde_json::to_string_pretty(&t).expect("a JSON value serializes")
}

/// A family's tokenizer: the added tokens and the tiny chat template, which
/// covers every template path the engine drives on the family
/// (`lloom-engine/tests/tiny_templates.rs` pins each one, and checks the
/// checkpoints' own templates against the same paths when they are cached).
#[derive(Debug, Clone)]
pub struct TinyTokenizer {
    pub added: Vec<(String, bool)>,
    pub chat_template: &'static str,
}

impl TinyTokenizer {
    pub fn qwen4exp() -> Self {
        Self {
            added: QWEN4EXP_ADDED
                .iter()
                .map(|t| (t.content.to_string(), t.special))
                .collect(),
            chat_template: include_str!("templates/qwen4exp.jinja"),
        }
    }

    pub fn lfm2() -> Self {
        Self {
            added: lfm2_added(),
            chat_template: include_str!("templates/lfm2.jinja"),
        }
    }

    /// Ids a model must be able to embed: the bytes and the added tokens.
    pub fn vocab_size(&self) -> usize {
        256 + self.added.len()
    }

    /// The id of an added token.
    pub fn id(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .position(|(c, _)| c == content)
            .map(|i| 256 + i as u32)
    }

    /// Write `tokenizer.json` and `chat_template.jinja` into `dir`.
    pub fn write(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json(&self.added))?;
        std::fs::write(dir.join("chat_template.jinja"), self.chat_template)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// The map is a bijection onto exactly the characters `tokenizers`'
    /// byte-level pre-tokenizer produces (its own alphabet, the external
    /// reference), the printable ASCII range maps to itself, and space is
    /// `Ġ` (U+0120), as in every byte-level BPE vocabulary.
    #[test]
    fn the_byte_map_is_gpt2s() {
        let chars: std::collections::BTreeSet<char> = (0..=255u8).map(byte_char).collect();
        assert_eq!(chars.len(), 256);
        let reference: std::collections::BTreeSet<char> =
            tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
                .into_iter()
                .collect();
        assert_eq!(chars, reference);
        for b in b'!'..=b'~' {
            assert_eq!(byte_char(b), b as char);
        }
        assert_eq!(byte_char(b'A'), 'A');
        assert_eq!(byte_char(b' '), '\u{120}');
        assert_eq!(byte_char(b'\n'), '\u{10A}');
        assert_eq!(byte_char(0), '\u{100}');
    }

    /// The real `tokenizers` reads it; any UTF-8 text encodes to its bytes
    /// and decodes back; an added token is one id at its place.
    #[test]
    fn the_tokenizer_round_trips_text_and_finds_the_turn_tokens() {
        for tt in [TinyTokenizer::qwen4exp(), TinyTokenizer::lfm2()] {
            let tk = tokenizers::Tokenizer::from_bytes(tokenizer_json(&tt.added)).unwrap();
            assert_eq!(tk.get_vocab_size(true), tt.vocab_size());
            let ascii: String = (b' '..=b'~').map(char::from).collect();
            let text = format!("The capital of France is Paris. ¿Qué? 東京\n{ascii}\t");
            let text = text.as_str();
            let enc = tk.encode(text, false).unwrap();
            assert_eq!(enc.get_ids().len(), text.len(), "one id per byte");
            assert_eq!(
                enc.get_ids(),
                text.bytes()
                    .map(|b| b as u32)
                    .collect::<Vec<_>>()
                    .as_slice()
            );
            assert_eq!(tk.decode(enc.get_ids(), false).unwrap(), text);
            let im_end = tt.id("<|im_end|>").unwrap();
            assert_eq!(tk.token_to_id("<|im_end|>"), Some(im_end));
            let enc = tk.encode("hi<|im_end|>", false).unwrap();
            assert_eq!(enc.get_ids(), &[b'h' as u32, b'i' as u32, im_end]);
        }
        assert_eq!(TinyTokenizer::qwen4exp().vocab_size(), 256 + 33);
        assert_eq!(TinyTokenizer::lfm2().vocab_size(), 256 + 124);
        assert_eq!(
            TinyTokenizer::lfm2().id("<|img_thumbnail|>"),
            Some(256 + 115)
        );
    }
}
