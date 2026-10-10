use candle::Result;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        // Emit whenever the decoded tail is a complete character. The rule
        // this replaces (emit only when the last char is alphanumeric, TGI's)
        // held every run of punctuation, whitespace or newlines until the next
        // letter — on a streamed reply that is a visible gap of up to a dozen
        // tokens (measured: gaps of 90–220 ms at 63 tok/s, `lfm2-chat`'s
        // cadence line) while the loop ran at full rate. What the hold was for
        // is the one case kept: a token that is a prefix of a multi-byte
        // character decodes to U+FFFD until the rest arrives.
        if text.len() > prev_text.len() && text.chars().last().unwrap() != '\u{FFFD}' {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::TokenOutputStream;

    fn lfm2_tokenizer() -> Option<tokenizers::Tokenizer> {
        let home = std::env::var("HOME").ok()?;
        let snaps = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
        let snap = std::fs::read_dir(snaps).ok()?.flatten().next()?.path();
        tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).ok()
    }

    /// The stream's pieces concatenate to the whole decode, and a run of
    /// punctuation is emitted as it arrives rather than held for the next
    /// letter. Skipped where the LFM2 tokenizer is not in the HF cache.
    #[test]
    fn pieces_concatenate_to_the_whole_decode_and_punctuation_is_not_held() {
        let Some(tok) = lfm2_tokenizer() else {
            eprintln!("no LFM2 tokenizer in the HF cache; skipped");
            return;
        };
        let text = "Paris, of course: the capital (since 508) — yes!!! Really... \"quoted\" — 1, 2, 3; done.\n\n- item\n- item two";
        let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
        let mut tos = TokenOutputStream::new(tok.clone());
        let mut out = String::new();
        let mut writes = 0usize;
        for &id in &ids {
            if let Some(piece) = tos.next_token(id).unwrap() {
                out.push_str(&piece);
                writes += 1;
            }
        }
        if let Some(rest) = tos.decode_rest().unwrap() {
            out.push_str(&rest);
        }
        assert_eq!(
            out,
            tok.decode(&ids, true).unwrap(),
            "the pieces are the whole decode"
        );
        // Under the alphanumeric rule the punctuation runs above collapse into
        // their neighbours' writes; every token that decodes to a complete
        // character is now its own write.
        assert!(
            writes >= ids.len() - 4,
            "{writes} writes for {} tokens: punctuation is still being held",
            ids.len()
        );
    }
}
