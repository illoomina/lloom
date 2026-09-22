//! The expert hotlist (`docs/expert-streaming.md` §20.3 R2): per layer, the
//! experts by how often a pass needed them, gathered from real use
//! (`Streamed::usage`) and kept in a text file, so the next load can warm
//! the streamed tier's pools before the first token instead of paying the
//! misses one token at a time (the device plan's cold run: a replay per
//! token for the first 48). A hint only: nothing pins, nothing evicts, the
//! router is untouched — a wrong hotlist costs what a cold pool costs.
//!
//! The file: `# lloom qwen4exp hotlist v1`, then one `layer expert hits`
//! line per (layer, expert) with hits > 0; merged by adding hits.
use lloom_tensor::Result;
use std::path::Path;

const HEADER: &str = "# lloom qwen4exp hotlist v1";

/// Per layer: `(expert, hits)`, hits descending, ties by expert id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hotlist {
    pub layers: Vec<Vec<(u32, u32)>>,
}

impl Hotlist {
    /// From the tier's per-layer usage counts (`Qwen4Exp::stream_usage`).
    pub fn from_usage(usage: &[Vec<u32>]) -> Self {
        let layers = usage
            .iter()
            .map(|u| {
                let mut v: Vec<(u32, u32)> = u
                    .iter()
                    .enumerate()
                    .filter(|(_, &h)| h > 0)
                    .map(|(e, &h)| (e as u32, h))
                    .collect();
                Self::sort(&mut v);
                v
            })
            .collect();
        Self { layers }
    }

    fn sort(v: &mut [(u32, u32)]) {
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }

    /// Add another profile's hits (a run's usage on top of the file's).
    pub fn merge(&mut self, other: &Hotlist) {
        if self.layers.len() < other.layers.len() {
            self.layers.resize(other.layers.len(), Vec::new());
        }
        for (mine, theirs) in self.layers.iter_mut().zip(&other.layers) {
            for &(e, h) in theirs {
                match mine.iter_mut().find(|(x, _)| *x == e) {
                    Some(entry) => entry.1 = entry.1.saturating_add(h),
                    None => mine.push((e, h)),
                }
            }
            Self::sort(mine);
        }
    }

    /// Layer `l`'s experts in order of hits, at most `n`.
    pub fn top(&self, l: usize, n: usize) -> Vec<u32> {
        self.layers
            .get(l)
            .map(|v| v.iter().take(n).map(|&(e, _)| e).collect())
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|l| l.is_empty())
    }

    pub fn to_text(&self) -> String {
        let mut s = String::from(HEADER);
        s.push('\n');
        for (l, layer) in self.layers.iter().enumerate() {
            for &(e, h) in layer {
                s.push_str(&format!("{l} {e} {h}\n"));
            }
        }
        s
    }

    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        match lines.next() {
            Some(h) if h.trim() == HEADER => {}
            other => lloom_tensor::bail!("hotlist: expected `{HEADER}`, got {other:?}"),
        }
        let mut layers: Vec<Vec<(u32, u32)>> = Vec::new();
        for (i, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let (l, e, h) = match (it.next(), it.next(), it.next()) {
                (Some(l), Some(e), Some(h)) => {
                    (l.parse::<usize>(), e.parse::<u32>(), h.parse::<u32>())
                }
                _ => lloom_tensor::bail!("hotlist line {}: `{line}`", i + 2),
            };
            let (l, e, h) = match (l, e, h) {
                (Ok(l), Ok(e), Ok(h)) => (l, e, h),
                _ => lloom_tensor::bail!("hotlist line {}: `{line}`", i + 2),
            };
            if layers.len() <= l {
                layers.resize(l + 1, Vec::new());
            }
            layers[l].push((e, h));
        }
        for v in &mut layers {
            Self::sort(v);
        }
        Ok(Self { layers })
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_text(&text)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_text())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_roundtrip_sorted_and_merged() {
        let usage = vec![vec![0, 3, 0, 7, 3], vec![1, 0, 0, 0, 0]];
        let h = Hotlist::from_usage(&usage);
        assert_eq!(h.layers[0], vec![(3, 7), (1, 3), (4, 3)]);
        assert_eq!(h.top(0, 2), vec![3, 1]);
        assert_eq!(h.top(1, 5), vec![0]);
        assert_eq!(h.top(2, 5), Vec::<u32>::new());
        let back = Hotlist::from_text(&h.to_text()).unwrap();
        assert_eq!(back, h);
        let mut m = h.clone();
        m.merge(&Hotlist::from_usage(&[
            vec![0, 5, 0, 0, 0],
            vec![],
            vec![0, 0, 2],
        ]));
        assert_eq!(m.layers[0], vec![(1, 8), (3, 7), (4, 3)]);
        assert_eq!(m.layers[2], vec![(2, 2)]);
        assert!(Hotlist::from_text("nope").is_err());
        assert!(Hotlist::from_text(&format!("{HEADER}\n1 x 2\n")).is_err());
    }
}
