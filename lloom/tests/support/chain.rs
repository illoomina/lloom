//! Shared chain fixture: the integration controls and private seed-driver tests
//! execute the same fake CausalLM, not independent schedule implementations.
use super::subject;
use lloom_tensor::{DType, Device, Result, Tensor};
use subject::nn::{CausalLM, Model};
const VOCAB: usize = 64;
fn next_of(tok: u32) -> u32 {
    (tok * 7 + 3) % VOCAB as u32
}

/// A decoder whose logits peak at `next_of(last token)`, device-side.
pub(super) struct Chain {
    /// `[VOCAB]` f32: `table[t] = next_of(t)`.
    table: Tensor,
    /// `[VOCAB, VOCAB]` f32 identity: row `t` is the one-hot logits for `t`.
    eye: Tensor,
    /// A square f32 matrix every forward multiplies by itself and drops: GPU
    /// time no output depends on, so a loop that nothing paces runs ahead of
    /// the device. `None` on the plain fixture.
    ballast: Option<Tensor>,
}

impl Chain {
    pub(super) fn new(device: &Device) -> Result<Self> {
        Self::with_ballast(device, 0)
    }

    /// The fixture with a `[n, n]` matmul of ballast per forward (`n = 0`: none).
    pub(super) fn with_ballast(device: &Device, n: usize) -> Result<Self> {
        let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
        let ballast = (n > 0)
            .then(|| Tensor::ones((n, n), DType::F32, device))
            .transpose()?;
        Ok(Self {
            table: Tensor::from_vec(table, VOCAB, device)?,
            eye: Tensor::eye(VOCAB, DType::F32, device)?,
            ballast,
        })
    }
}

impl Model for Chain {
    type Config = ();
    fn from_weights(_vb: subject::nn::VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("Chain has no weights")
    }
}

impl CausalLM for Chain {
    /// Positions seen, so a test can check `pos` against the cache.
    type Cache = usize;

    fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<usize> {
        Ok(0)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }

    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        let (_b, seq) = embeds.dims2()?;
        assert_eq!(
            *cache, pos,
            "forward at pos {pos} against a cache of {cache}"
        );
        *cache += seq;
        if let Some(b) = &self.ballast {
            b.matmul(b)?;
        }
        // The last position's token, `[1]` U32, then its successor, then the
        // one-hot row: all on the device.
        let last = embeds.narrow(1, seq - 1, 1)?.flatten_all()?;
        let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
        self.eye.index_select(&next, 0)
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

/// Reuse the chain's computation while observing actual production forwards.
pub(super) struct RecordingChain {
    chain: Chain,
    pub(super) forwards: std::cell::RefCell<Vec<(usize, usize)>>,
}

impl RecordingChain {
    pub(super) fn cpu() -> Result<Self> {
        Ok(Self {
            chain: Chain::new(&Device::Cpu)?,
            forwards: std::cell::RefCell::new(Vec::new()),
        })
    }
}

impl Model for RecordingChain {
    type Config = ();

    fn from_weights(_vb: subject::nn::VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("RecordingChain has no weights")
    }
}

impl CausalLM for RecordingChain {
    type Cache = usize;

    fn new_cache(&self, cfg: &(), device: &Device) -> Result<usize> {
        self.chain.new_cache(cfg, device)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.chain.embed(tokens)
    }

    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        self.forwards.borrow_mut().push((pos, embeds.dim(1)?));
        self.chain.forward_embeds(embeds, pos, cache)
    }

    fn vocab_size(&self) -> usize {
        self.chain.vocab_size()
    }
}
