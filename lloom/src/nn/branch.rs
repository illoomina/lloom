//! Request-owned causal branching over the incumbent paced prefill loop.

use super::decode::{DecodeOptions, PrefillOut};
use super::model::{CacheReset, CausalLM};
use super::speculator::Session;
use crate::generation::ReadbackStrategy;
use lloom_tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// State retained outside the finite speculative/turn-checkpoint rings.
///
/// The caller exclusively owns the cache and its writable backing from capture
/// through the final restore. Between restores it only appends after the retained
/// prefix; it never resets, truncates below that prefix, or gives out aliases.
/// Recurrent snapshots own independent storage; KV snapshots may retain lengths
/// and immutable prefix handles. Capture/restore allocations belong to the
/// caller's model domain and reserved pass, not a separate branching budget.
/// A failed restore is not transactional: the caller discards the request.
pub trait RetainState {
    type Retained;
    /// Enable exact multi-tile continuation on this private request cache before
    /// the first trunk tile. Incumbent chat caches and global model flags stay unchanged.
    fn prepare_branching(&mut self) -> Result<()> {
        Ok(())
    }
    fn retain_state(&mut self) -> Result<Self::Retained>;
    fn restore_retained(&mut self, retained: &Self::Retained) -> Result<()>;
}

struct Trunk<S> {
    owner: Arc<()>,
    pos: usize,
    state: S,
}

/// A fresh private sequence of an already loaded model, with one retained trunk.
///
/// No mutable cache or sequence escapes. Each suffix restores the trunk and uses
/// `Session::prefill`; it never invokes the chat reset-and-refill fallback. The
/// owner must admit cache construction, snapshot copies, token/logit storage and
/// each complete asynchronous pass before calling these methods. This type does
/// not create an entitlement or infer one from the device.
pub struct BranchSession<'m, M: CausalLM>
where
    M::Cache: RetainState,
{
    session: Session<'m, M>,
    owner: Arc<()>,
    trunk: Option<Trunk<<M::Cache as RetainState>::Retained>>,
    failed: bool,
}

impl<'m, M: CausalLM> BranchSession<'m, M>
where
    M::Cache: RetainState,
{
    pub fn new(model: &'m M, cfg: &M::Config, device: &Device) -> Result<Self> {
        Ok(Self {
            session: Session::new(model, model.new_cache(cfg, device)?),
            owner: Arc::new(()),
            trunk: None,
            failed: false,
        })
    }

    /// Prefill the shared state once, including an empty or one-token trunk.
    /// A second call requires explicit reset. Any execution/capture failure
    /// invalidates the request; no previous logits remain observable.
    pub fn begin<R: ReadbackStrategy>(
        &mut self,
        readback: &mut R,
        tokens: &[u32],
        opts: &DecodeOptions,
    ) -> Result<PrefillOut> {
        if self.failed || self.trunk.is_some() {
            lloom_tensor::bail!("branch session: reset before starting another trunk")
        }
        self.failed = true;
        self.session.seq.cache.prepare_branching()?;
        let out = self.session.prefill(readback, tokens, opts)?;
        self.session
            .model
            .settle_pending(&mut self.session.seq.cache)?;
        self.trunk = Some(Trunk {
            owner: self.owner.clone(),
            pos: self.session.pos(),
            state: self.session.seq.cache.retain_state()?,
        });
        self.session.seq.last_logits = None;
        self.failed = false;
        Ok(out)
    }

    /// Append one nonempty answer-slot suffix and borrow its final logits.
    /// Read selected values before the next call; a cloned Tensor may alias
    /// the session's reusable logits hold and is not a retained prediction.
    pub fn suffix<R: ReadbackStrategy>(
        &mut self,
        readback: &mut R,
        tokens: &[u32],
        opts: &DecodeOptions,
    ) -> Result<&Tensor> {
        if self.failed {
            lloom_tensor::bail!("branch session: previous execution failed; reset required")
        }
        if tokens.is_empty() {
            lloom_tensor::bail!("branch session: an answer-slot suffix must be nonempty")
        }
        let trunk = self
            .trunk
            .as_ref()
            .ok_or_else(|| lloom_tensor::Error::Msg("branch session: no retained trunk".into()))?;
        if !Arc::ptr_eq(&trunk.owner, &self.owner) || trunk.pos > self.session.pos() {
            lloom_tensor::bail!("branch session: foreign or invalidated trunk")
        }
        self.failed = true;
        self.session.seq.last_logits = None;
        self.session
            .model
            .settle_pending(&mut self.session.seq.cache)?;
        self.session.seq.cache.restore_retained(&trunk.state)?;
        self.session.seq.pos = trunk.pos;
        self.session.seq.tokens.truncate(trunk.pos);
        self.session.prefill(readback, tokens, opts)?;
        self.session
            .model
            .settle_pending(&mut self.session.seq.cache)?;
        if self.session.last_logits().is_none() {
            lloom_tensor::bail!("branch session: suffix produced no answer-slot logits")
        }
        self.failed = false;
        Ok(self.session.last_logits().expect("checked above"))
    }

    /// Settle the model's pending work, then invalidate the trunk and reset.
    /// A failed settle leaves the request poisoned. This is not a general
    /// transaction rollback for arbitrary native/executor failures.
    pub fn reset(&mut self) -> Result<()>
    where
        M::Cache: CacheReset,
    {
        self.failed = true;
        self.session.seq.last_logits = None;
        self.session
            .model
            .settle_pending(&mut self.session.seq.cache)?;
        self.trunk = None;
        self.owner = Arc::new(());
        self.session.reset();
        self.failed = false;
        Ok(())
    }
}

/// Fallible host capacity, checked before any element initializer runs.
pub(crate) fn retained_vec<T>(len: usize) -> Result<Vec<T>> {
    std::alloc::Layout::array::<T>(len)
        .map_err(|e| lloom_tensor::Error::Msg(format!("retained state capacity: {e}")))?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|e| lloom_tensor::Error::Msg(format!("retained state allocation: {e}")))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::Blocking;
    use crate::nn::{Model, VarBuilder};
    use std::cell::Cell;

    #[derive(Default)]
    struct Cache {
        tokens: Vec<u32>,
        captures: usize,
    }
    impl RetainState for Cache {
        type Retained = Vec<u32>;
        fn retain_state(&mut self) -> Result<Vec<u32>> {
            self.captures += 1;
            Ok(self.tokens.clone())
        }
        fn restore_retained(&mut self, saved: &Vec<u32>) -> Result<()> {
            if !self.tokens.starts_with(saved) {
                lloom_tensor::bail!("retained prefix was overwritten")
            }
            self.tokens.truncate(saved.len());
            Ok(())
        }
    }
    impl CacheReset for Cache {
        fn reset(&mut self) {
            self.tokens.clear();
        }
    }
    #[derive(Default)]
    struct Tiny {
        calls: Cell<usize>,
        fail: Cell<bool>,
        fail_settle: Cell<bool>,
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
            Ok(Cache::default())
        }
        fn settle_pending(&self, _: &mut Cache) -> Result<()> {
            if self.fail_settle.get() {
                lloom_tensor::bail!("injected pending settle failure")
            }
            Ok(())
        }
        fn vocab_size(&self) -> usize {
            2
        }
        fn embed(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn forward_embeds(&self, x: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
            assert_eq!(pos, cache.tokens.len());
            self.calls.set(self.calls.get() + 1);
            cache.tokens.extend_from_slice(&x.to_vec2::<u32>()?[0]);
            if self.fail.get() {
                lloom_tensor::bail!("injected after cache write")
            }
            Tensor::new(
                &[[
                    cache.tokens.iter().sum::<u32>() as f32,
                    cache.tokens.len() as f32,
                ]],
                &Device::Cpu,
            )
        }
    }
    fn opts() -> DecodeOptions {
        DecodeOptions {
            prefill_tile: 2,
            ..DecodeOptions::new(1)
        }
    }

    #[test]
    fn trunk_once_long_suffixes_zero_and_one_token() -> Result<()> {
        for trunk in [vec![], vec![7], vec![2, 3, 4]] {
            let model = Tiny::default();
            let mut rb = Blocking::new(&[1], &Device::Cpu)?;
            let mut s = BranchSession::new(&model, &(), &Device::Cpu)?;
            s.begin(&mut rb, &trunk, &opts())?;
            for q in 0..12u32 {
                let suffix = vec![q + 1; 13];
                let got = s.suffix(&mut rb, &suffix, &opts())?.to_vec1::<f32>()?;
                assert_eq!(
                    got,
                    vec![
                        (trunk.iter().sum::<u32>() + 13 * (q + 1)) as f32,
                        (trunk.len() + 13) as f32
                    ]
                );
                assert_eq!(
                    s.session.seq.cache.captures, 1,
                    "capture repeated per question"
                );
            }
            assert_eq!(
                model.calls.get(),
                trunk.len().div_ceil(2) + 12 * 7,
                "trunk was replayed or paced tiling bypassed"
            );
        }
        Ok(())
    }

    #[test]
    fn foreign_reset_and_failed_request_refuse_stale_predictions() -> Result<()> {
        let model = Tiny::default();
        let mut rb = Blocking::new(&[1], &Device::Cpu)?;
        let mut a = BranchSession::new(&model, &(), &Device::Cpu)?;
        let mut b = BranchSession::new(&model, &(), &Device::Cpu)?;
        a.begin(&mut rb, &[2], &opts())?;
        b.begin(&mut rb, &[3], &opts())?;
        std::mem::swap(&mut a.trunk, &mut b.trunk);
        let before = model.calls.get();
        assert!(a
            .suffix(&mut rb, &[4], &opts())
            .unwrap_err()
            .to_string()
            .contains("foreign"));
        assert_eq!(model.calls.get(), before);
        a.reset()?;
        assert!(a.suffix(&mut rb, &[4], &opts()).is_err());
        a.begin(&mut rb, &[5], &opts())?;
        model.fail.set(true);
        assert!(a.suffix(&mut rb, &[6], &opts()).is_err());
        model.fail.set(false);
        assert!(a
            .suffix(&mut rb, &[6], &opts())
            .unwrap_err()
            .to_string()
            .contains("reset required"));
        model.fail_settle.set(true);
        assert!(a.reset().is_err());
        assert!(a.failed);
        assert!(a.session.last_logits().is_none());
        assert!(a.begin(&mut rb, &[], &opts()).is_err());
        model.fail_settle.set(false);
        a.reset()?;
        a.begin(&mut rb, &[], &opts())?;
        assert_eq!(
            a.suffix(&mut rb, &[8], &opts())?.to_vec1::<f32>()?,
            vec![8., 1.]
        );
        Ok(())
    }
}
