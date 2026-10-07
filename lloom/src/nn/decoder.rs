//! The generic causal decoder topology (`porting-models.md` §21.9, level 1):
//! `Decoder<P, C, E, L, H>` owns the embedding, the layer stack and its
//! per-layer `State`, and the head, and implements `CausalLM` once for every
//! family whose layer is a `Layer` — a dataflow of the form
//!
//! ```text
//! h = E(ids);  for l in layers { h = L_l(h, state_l) };  logits = H(h[last])
//! ```
//!
//! A layer is a composition of level-0 blocks (`nn::blocks`): [`PreNorm`] is
//! `x + mixer(norm(x))`, then `x + ffn(norm(x))`; [`Hybrid`] is two mixers
//! chosen per layer (enum dispatch, never `dyn` — CONTRIBUTING §3.6). A
//! family that fits is a type alias and a loader, no forward of its own
//! (Qwen3.5 is the first, `models::qwen3_5`); one whose state is not per
//! layer keeps its hand-written forward (§21.10's ladder).
//!
//! Static dispatch only: every slot is a type parameter, so the forward
//! monomorphizes to the blocks' own calls.

use crate::nn::blocks::{Ffn, Mixer, Norm};
use crate::nn::model::{CacheReset, CacheRewind, CausalLM, Model};
use crate::nn::ops::KvPrecision;
use crate::nn::var_builder::VarBuilder;
use crate::optimizer::{NodeId, Shape, Visit, Walker};
use crate::profile::Profile;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor};
use std::marker::PhantomData;

/// What a sequence's state is sized by: the KV capacity in tokens and the
/// KV store's precision (the recurrent channels need neither).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateCapacity {
    pub kv_tokens: usize,
    pub kv_precision: KvPrecision,
}

impl StateCapacity {
    pub fn new(kv_tokens: usize) -> Self {
        Self {
            kv_tokens,
            kv_precision: KvPrecision::dense(DType::F32),
        }
    }
}

/// A token mixer the topology can own the state of: how to make a fresh
/// state for one layer, clear it, and walk the mixer's step.
pub trait MixerBlock: Mixer + Send + Sync {
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    /// The mixer's decode step as IR, layer `l` (`Visit`, with the layer index
    /// and the capacity a cache-bearing mixer's walk reads).
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId;
}

/// A feed-forward the topology can own the state of (`()` for a dense MLP
/// and a resident MoE).
pub trait FfnBlock: Ffn + Send + Sync {
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize) -> NodeId;
}

/// One layer of a [`Decoder`]: a unit over `[1, seq, hidden]` with its own
/// per-sequence state.
pub trait Layer: Send + Sync {
    type State: Send + Sync;
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
    fn visit_layer(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId;
}

/// The token embedding of a [`Decoder`].
pub trait Embed: Send + Sync {
    /// `[b, seq]` ids → `[b, seq, hidden]`.
    fn embed(&self, ids: &Tensor) -> Result<Tensor>;
    fn visit_embed(&self, w: &mut Walker, ids: NodeId) -> NodeId;
}

/// The head of a [`Decoder`]: the last position's residual → logits.
pub trait Head: Send + Sync {
    /// `[1, 1, hidden]` → `[1, vocab]`.
    fn head(&self, h: &Tensor) -> Result<Tensor>;
    fn visit_head(&self, w: &mut Walker, x: NodeId) -> NodeId;
    fn vocab_size(&self) -> usize;
}

/// A token-embedding table (packed or dense).
#[derive(Debug, Clone)]
pub struct TokenEmbed {
    table: QMatMul,
    hidden: usize,
}

impl TokenEmbed {
    pub fn new(table: QMatMul, hidden: usize) -> Self {
        Self { table, hidden }
    }
    pub fn table(&self) -> &QMatMul {
        &self.table
    }
}

impl Embed for TokenEmbed {
    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.table.embedding(ids)
    }
    fn visit_embed(&self, w: &mut Walker, ids: NodeId) -> NodeId {
        let (_, shape, dt) = crate::optimizer::walk::qmat_operand(&self.table);
        let table = w.weight("token_embd.weight", shape, dt);
        // A packed table's rows are gathered by the quantized kernel; a dense
        // one's by `index_select`.
        let gather = match &self.table {
            QMatMul::QTensor(_) => "get_rows",
            _ => "is_u32",
        };
        w.opaque(
            gather,
            &[table, ids],
            Shape::of(&[1, 1, self.hidden]),
            DType::F32,
        )
    }
}

/// A final norm and a projection to the vocabulary.
#[derive(Debug, Clone)]
pub struct NormHead<N> {
    norm: N,
    proj: QMatMul,
    vocab: usize,
}

impl<N: Norm> NormHead<N> {
    pub fn new(norm: N, proj: QMatMul, vocab: usize) -> Self {
        Self { norm, proj, vocab }
    }
    pub fn proj(&self) -> &QMatMul {
        &self.proj
    }
}

impl<N: Norm + Send + Sync> Head for NormHead<N> {
    fn head(&self, h: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = h.dims3()?;
        if b != 1 || seq != 1 {
            lloom_tensor::bail!("the head takes one position, got {:?}", h.shape())
        }
        let n = self.norm.execute(h)?;
        self.proj
            .forward(&n.reshape((1, hidden))?)?
            .reshape((1, self.vocab))
    }
    fn visit_head(&self, w: &mut Walker, x: NodeId) -> NodeId {
        let n = w.scoped("output_norm", |w| self.norm.visit(w, x));
        let logits = w.project_qmat("output.weight", &self.proj, n);
        w.view("reshape", logits, Shape::of(&[1, self.vocab]))
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// `x + mixer(norm_in(x))`, then `x + ffn(norm_mid(x))` — the pre-norm
/// residual layer (Llama's, Qwen's, LFM2's shape).
#[derive(Debug, Clone)]
pub struct PreNorm<N, M, F> {
    pub norm_in: N,
    pub mixer: M,
    pub norm_mid: N,
    pub ffn: F,
}

/// A [`PreNorm`] layer's state: its mixer's and its feed-forward's.
#[derive(Debug)]
pub struct PreNormState<MS, FS> {
    pub mixer: MS,
    pub ffn: FS,
}

impl<N, M, F> Layer for PreNorm<N, M, F>
where
    N: Norm + Send + Sync,
    M: MixerBlock,
    F: FfnBlock,
    M::State: Send + Sync,
    F::State: Send + Sync,
{
    type State = PreNormState<M::State, F::State>;
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State> {
        Ok(PreNormState {
            mixer: self.mixer.new_state(dev, cap)?,
            ffn: self.ffn.new_state(dev, cap)?,
        })
    }
    fn reset_state(state: &mut Self::State) {
        M::reset_state(&mut state.mixer);
        F::reset_state(&mut state.ffn);
    }
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor> {
        let h = self.norm_in.execute(x)?;
        let m = self.mixer.execute(&h, &mut state.mixer)?;
        let x = (x + m)?;
        let h = self.norm_mid.execute(&x)?;
        let f = self.ffn.execute(&h, &mut state.ffn)?;
        x + f
    }
    fn visit_layer(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId {
        let h = w.scoped("attn_norm", |w| self.norm_in.visit(w, x));
        let m = self.mixer.visit_step(w, h, l, cap);
        let x = w.add(x, m);
        let h = w.scoped("post_attention_norm", |w| self.norm_mid.visit(w, x));
        let f = self.ffn.visit_step(w, h, l);
        w.add(x, f)
    }
}

/// Two mixers, one chosen per layer by the checkpoint's layer-type array:
/// what a hybrid family's layer loop matches on (§21.10's rung 2 inside a
/// rung-1 topology).
#[derive(Debug, Clone)]
pub enum Hybrid<A, B> {
    A(A),
    B(B),
}

#[derive(Debug)]
pub enum HybridState<SA, SB> {
    A(SA),
    B(SB),
}

impl<A: Mixer, B: Mixer> Mixer for Hybrid<A, B> {
    type State = HybridState<A::State, B::State>;
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor> {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.execute(x, s),
            (Hybrid::B(m), HybridState::B(s)) => m.execute(x, s),
            _ => lloom_tensor::bail!("a hybrid layer's state is the other mixer's"),
        }
    }
}

impl<A: Mixer, B: Mixer> Visit for Hybrid<A, B> {
    type In = ();
    type Out = ();
    /// The walk goes through [`MixerBlock::visit_step`] (it needs the layer).
    fn visit(&self, _w: &mut Walker, (): ()) {}
}

impl<A: MixerBlock, B: MixerBlock> MixerBlock for Hybrid<A, B>
where
    A::State: Send + Sync,
    B::State: Send + Sync,
{
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State> {
        Ok(match self {
            Hybrid::A(m) => HybridState::A(m.new_state(dev, cap)?),
            Hybrid::B(m) => HybridState::B(m.new_state(dev, cap)?),
        })
    }
    fn reset_state(state: &mut Self::State) {
        match state {
            HybridState::A(s) => A::reset_state(s),
            HybridState::B(s) => B::reset_state(s),
        }
    }
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId {
        match self {
            Hybrid::A(m) => m.visit_step(w, x, l, cap),
            Hybrid::B(m) => m.visit_step(w, x, l, cap),
        }
    }
}

/// A decoder's per-sequence state: one per layer, and the position.
pub struct DecoderState<L: Layer> {
    pub layers: Vec<L::State>,
    pub pos: usize,
}

impl<L: Layer> CacheReset for DecoderState<L> {
    fn reset(&mut self) {
        for s in &mut self.layers {
            L::reset_state(s);
        }
        self.pos = 0;
    }
}

/// No rewind yet: the first rung keeps no window, so the horizon is zero and
/// a driver decodes without lookahead (`CacheRewind`'s contract for a cache
/// with no rewindable arm).
impl<L: Layer> CacheRewind for DecoderState<L> {
    fn rewind_horizon(&self) -> usize {
        0
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        lloom_tensor::bail!(
            "a Decoder's state keeps no window to rewind {n} positions through (horizon 0)"
        )
    }
}

/// The topology: `E` embeds, `L` is the layer, `H` the head; `P` the
/// profile the model runs at, `C` the family's config (what `CausalLM`
/// names as `Config`).
pub struct Decoder<P: Profile, C, E, L, H> {
    profile: PhantomData<P>,
    cfg: C,
    embed: E,
    layers: Vec<L>,
    head: H,
    cap: StateCapacity,
    /// The longest chunk a multi-position forward runs through the layers at
    /// once (a prefill's attention is `chunk × context`).
    chunk: usize,
    family: &'static str,
}

/// The chunk a prefill runs through the layers at once.
pub const PREFILL_CHUNK: usize = 512;

impl<P, C, E, L, H> Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    /// Assemble a decoder. The process is held to the regime `P` names
    /// (`profile::hold`): a family at `Exact` never computes under an
    /// inherited `fast`.
    pub fn new(
        family: &'static str,
        cfg: C,
        embed: E,
        layers: Vec<L>,
        head: H,
        cap: StateCapacity,
    ) -> Result<Self> {
        crate::profile::hold::<P::Regime>(family)?;
        Ok(Self {
            profile: PhantomData,
            cfg,
            embed,
            layers,
            head,
            cap,
            chunk: PREFILL_CHUNK,
            family,
        })
    }

    pub fn config(&self) -> &C {
        &self.cfg
    }
    pub fn layers(&self) -> &[L] {
        &self.layers
    }
    pub fn layers_mut(&mut self) -> &mut [L] {
        &mut self.layers
    }
    pub fn embed_block(&self) -> &E {
        &self.embed
    }
    pub fn head_block(&self) -> &H {
        &self.head
    }
    pub fn family(&self) -> &'static str {
        self.family
    }
    pub fn capacity(&self) -> StateCapacity {
        self.cap
    }
    /// The KV capacity (tokens) every state built after this call takes.
    pub fn set_capacity(&mut self, cap: StateCapacity) {
        self.cap = cap;
    }
    pub fn set_prefill_chunk(&mut self, chunk: usize) {
        self.chunk = chunk.max(1);
    }

    /// Run `embeds [1, seq, hidden]` from the state's position through the
    /// stack, in chunks of at most [`Self::chunk`] positions, and return the
    /// last position's logits `[1, vocab]`.
    pub fn forward_from(&self, embeds: &Tensor, state: &mut DecoderState<L>) -> Result<Tensor> {
        let (b, seq, _) = embeds.dims3()?;
        if b != 1 || seq == 0 {
            lloom_tensor::bail!(
                "{}: a forward takes one sequence of >= 1 tokens, got {:?}",
                self.family,
                embeds.shape()
            )
        }
        let mut last = None;
        let mut start = 0;
        while start < seq {
            let len = (seq - start).min(self.chunk);
            let mut h = embeds.narrow(1, start, len)?;
            for (l, layer) in self.layers.iter().enumerate() {
                h = layer.execute(&h, &mut state.layers[l])?;
            }
            state.pos += len;
            last = Some(h.narrow(1, len - 1, 1)?);
            start += len;
        }
        self.head.head(&last.expect("seq >= 1"))
    }
}

impl<P, C, E, L, H> Model for Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    type Config = C;
    /// A decoder is assembled by its family's loader from the blocks it
    /// read; the safetensors `VarBuilder` path has no weights to offer.
    fn from_weights(_vb: VarBuilder, _cfg: &Self::Config) -> Result<Self> {
        lloom_tensor::bail!("a Decoder is built by its family's loader, not from a VarBuilder")
    }
}

impl<P, C, E, L, H> CausalLM for Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    type Cache = DecoderState<L>;

    fn new_cache(&self, _cfg: &Self::Config, dev: &Device) -> Result<Self::Cache> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.new_state(dev, &self.cap))
            .collect::<Result<Vec<_>>>()?;
        Ok(DecoderState { layers, pos: 0 })
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed.embed(tokens)
    }

    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        if pos != cache.pos {
            lloom_tensor::bail!(
                "{}: pos {pos} != the state's position {}",
                self.family,
                cache.pos
            )
        }
        self.forward_from(embeds, cache)
    }

    fn vocab_size(&self) -> usize {
        self.head.vocab_size()
    }
}

impl<P, C, E, L, H> Visit for Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    type In = ();
    type Out = NodeId;
    /// The decode step: the token's row, every layer, the head.
    fn visit(&self, w: &mut Walker, (): ()) -> NodeId {
        let ids = w.input("input_ids", Shape::of(&[1, 1]), DType::U32);
        let mut h = self.embed.visit_embed(w, ids);
        for (l, layer) in self.layers.iter().enumerate() {
            h = w.scoped(format!("blk.{l}"), |w| {
                layer.visit_layer(w, h, l, &self.cap)
            });
        }
        self.head.visit_head(w, h)
    }
}

// ---- The engine families' blocks as topology blocks ---------------------------

impl<P: Profile> MixerBlock for crate::models::qwen4exp::GatedDeltaNet<P> {
    fn new_state(&self, _dev: &Device, _cap: &StateCapacity) -> Result<Self::State> {
        Ok(crate::models::qwen4exp::GdnState {
            layer: 0,
            s: crate::nn::RecurrentState::new(1),
            conv: crate::nn::RecurrentState::new(1),
        })
    }
    fn reset_state(state: &mut Self::State) {
        state.s.reset();
        state.conv.reset();
    }
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize, _cap: &StateCapacity) -> NodeId {
        self.visit(w, (x, l))
    }
}

impl<P: Profile> MixerBlock for crate::models::qwen4exp::QsaAttention<P> {
    fn new_state(&self, _dev: &Device, cap: &StateCapacity) -> Result<Self::State> {
        crate::models::qwen4exp::KvCache::with_precision(cap.kv_tokens, cap.kv_precision)
    }
    fn reset_state(state: &mut Self::State) {
        state.reset();
    }
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId {
        self.visit(w, (x, l, cap.kv_precision))
    }
}

impl FfnBlock for crate::nn::blocks::SwiGlu {
    fn new_state(&self, _dev: &Device, _cap: &StateCapacity) -> Result<()> {
        Ok(())
    }
    fn reset_state(_state: &mut ()) {}
    fn visit_step(&self, w: &mut Walker, x: NodeId, _l: usize) -> NodeId {
        self.visit(w, x)
    }
}

impl<P: Profile, R: crate::models::qwen4exp::ExpertResidency + Send + Sync> FfnBlock
    for crate::models::qwen4exp::MoeLayer<P, R>
{
    fn new_state(&self, _dev: &Device, _cap: &StateCapacity) -> Result<()> {
        Ok(())
    }
    fn reset_state(_state: &mut ()) {}
    fn visit_step(&self, w: &mut Walker, x: NodeId, _l: usize) -> NodeId {
        self.visit(w, x)
    }
}
