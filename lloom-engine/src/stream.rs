//! The text side of a token stream: incremental detokenization and stop
//! strings, shared by the single-session loop and the batch scheduler.
//!
//! A fragment is emitted only when the tokens since the last emission decode
//! to complete text — a multi-byte character can span several tokens, so a
//! run ending on a `U+FFFD` replacement char is held until the next token
//! completes it. Decoding only the not-yet-emitted tail keeps the cost linear
//! in the output (the whole sequence each step would be quadratic).

use lloom_tensor::Result;

/// One request's generated tokens and their text, fragment by fragment.
#[derive(Debug, Default)]
pub struct TextStream {
    generated: Vec<u32>,
    text: String,
    /// Start of the run of generated tokens not yet emitted as text.
    pending: usize,
    /// How much of `text` [`Self::hit_stop`] has already searched: a stop
    /// string found there ended the run, so a later call searches only what
    /// was appended since (plus the overlap a straddling stop needs). Without
    /// it every token re-scanned the whole reply — O(reply) a token, a cost
    /// that grows with the work done (§15.2 #10's corollary).
    checked: std::cell::Cell<usize>,
}

impl TextStream {
    pub fn new(capacity: usize) -> Self {
        Self {
            generated: Vec::with_capacity(capacity),
            text: String::new(),
            pending: 0,
            checked: std::cell::Cell::new(0),
        }
    }

    /// Append `ids` and return the text they complete, if any; the text is
    /// also accumulated in [`Self::text`].
    pub fn push(
        &mut self,
        ids: &[u32],
        detokenize: &mut impl FnMut(&[u32]) -> Result<String>,
    ) -> Result<Option<String>> {
        self.generated.extend_from_slice(ids);
        let chunk = detokenize(&self.generated[self.pending..])?;
        if chunk.is_empty() || chunk.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        self.text.push_str(&chunk);
        self.pending = self.generated.len();
        Ok(Some(chunk))
    }

    /// Whether any of `stop_strings` has appeared in the text so far. Only
    /// the text past the last call is searched, widened back by the longest
    /// stop's length less one so a stop straddling two fragments is found.
    pub fn hit_stop(&self, stop_strings: &[String]) -> bool {
        let Some(longest) = stop_strings.iter().map(String::len).max() else {
            return false;
        };
        let mut from = self.checked.get().saturating_sub(longest.saturating_sub(1));
        while !self.text.is_char_boundary(from) {
            from -= 1;
        }
        self.checked.set(self.text.len());
        let window = &self.text[from..];
        stop_strings.iter().any(|s| window.contains(s.as_str()))
    }

    /// The text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The ids so far.
    pub fn tokens(&self) -> &[u32] {
        &self.generated
    }

    /// The text, cut before the earliest stop string, and the ids.
    pub fn finish(mut self, stop_strings: &[String]) -> (String, Vec<u32>) {
        if let Some(cut) = stop_strings
            .iter()
            .filter_map(|s| self.text.find(s.as_str()))
            .min()
        {
            self.text.truncate(cut);
        }
        (self.text, self.generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A detokenizer over a two-token "character": id 1 alone is an
    /// incomplete byte (rendered as U+FFFD), 1 followed by 2 is "é".
    fn detok(ids: &[u32]) -> Result<String> {
        let mut s = String::new();
        let mut i = 0;
        while i < ids.len() {
            match ids[i] {
                1 if ids.get(i + 1) == Some(&2) => {
                    s.push('é');
                    i += 2;
                }
                1 => {
                    s.push('\u{FFFD}');
                    i += 1;
                }
                n => {
                    s.push((b'a' + (n as u8 % 26)) as char);
                    i += 1;
                }
            }
        }
        Ok(s)
    }

    #[test]
    fn a_fragment_is_held_until_its_character_completes() -> Result<()> {
        let mut st = TextStream::new(8);
        let mut d = detok;
        assert_eq!(st.push(&[3], &mut d)?, Some("d".to_string()));
        assert_eq!(
            st.push(&[1], &mut d)?,
            None,
            "an incomplete character is held"
        );
        assert_eq!(st.push(&[2], &mut d)?, Some("é".to_string()));
        assert_eq!(st.text(), "dé");
        assert_eq!(st.tokens(), &[3, 1, 2]);
        Ok(())
    }

    #[test]
    fn a_stop_straddling_two_fragments_is_found_by_the_windowed_scan() -> Result<()> {
        // `detok` maps id n (other than 1) to the letter 'a' + n % 26.
        let mut st = TextStream::new(16);
        let mut d = detok;
        let stops = vec!["jkl".to_string()];
        st.push(&[7, 8], &mut d)?; // "hi"
        assert!(!st.hit_stop(&stops));
        st.push(&[9, 10], &mut d)?; // "jk"
        assert!(!st.hit_stop(&stops));
        st.push(&[11, 12], &mut d)?; // "lm"
        assert!(st.hit_stop(&stops), "the stop spans the last two fragments");
        Ok(())
    }

    #[test]
    fn the_scan_does_not_revisit_checked_text() -> Result<()> {
        let mut st = TextStream::new(16);
        let mut d = detok;
        st.push(&[3, 4, 5, 6, 7, 8, 9, 10], &mut d)?; // "defghijk"
        assert!(!st.hit_stop(&["zz".to_string()]));
        assert_eq!(st.checked.get(), 8);
        // Text already checked is not searched again: a stop there would have
        // ended the run at that call. The window is the new text plus the
        // longest stop's length less one.
        st.push(&[11], &mut d)?; // "l"
                                 // One call per push, as the decode loop makes it: "def" lies wholly
                                 // in text an earlier call searched.
        assert!(!st.hit_stop(&["def".to_string()]));
        assert_eq!(st.checked.get(), 9);
        Ok(())
    }

    #[test]
    fn a_stop_string_is_detected_and_cut() -> Result<()> {
        let mut st = TextStream::new(8);
        let mut d = detok;
        st.push(&[0, 1, 2, 3], &mut d)?;
        let stops = vec!["é".to_string()];
        assert!(st.hit_stop(&stops));
        assert!(!st.hit_stop(&["zz".to_string()]));
        let (text, ids) = st.finish(&stops);
        assert_eq!(text, "a");
        assert_eq!(ids, vec![0, 1, 2, 3]);
        Ok(())
    }
}
