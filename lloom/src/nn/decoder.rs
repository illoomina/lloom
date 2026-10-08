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
use crate::nn::ring::Ring;
use crate::nn::row_log::RowLog;
use crate::nn::speculator::{accept_prefix, Anchor, Draft, DraftTokens, NextnHidden, Probes};
use crate::nn::var_builder::VarBuilder;
use crate::nn::zeros as zs;
use crate::optimizer::{NodeId, Shape, Visit, Walker};
use crate::profile::Profile;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor, D};
use std::marker::PhantomData;

/// What a sequence's state is sized by: the KV capacity in tokens and the
/// KV store's precision (the recurrent channels need neither).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateCapacity {
    pub kv_tokens: usize,
    pub kv_precision: KvPrecision,
    /// The longest verify window (positions) a state built at the `Ring`
    /// profile runs as ONE pass: its recurrent channels get `window + 1`
    /// phases (`GatedDeltaNet::pooled_state`). 0: no rings; a window runs
    /// as sequential steps inside a snapshot.
    pub window: usize,
    /// Turn checkpoints a state keeps (`DecoderState::restore`,
    /// `sequence-reuse-design.md` §2.3b): the last `checkpoints`
    /// multi-token passes' states, in a ring. 0: none, every request past
    /// the horizon is a fresh prefill.
    pub checkpoints: usize,
}

impl StateCapacity {
    pub fn new(kv_tokens: usize) -> Self {
        Self {
            kv_tokens,
            kv_precision: KvPrecision::dense(DType::F32),
            window: 0,
            checkpoints: 0,
        }
    }
}

/// What a mixer's state records when a speculative window opens, for its
/// rewind: a KV cache its lengths (tokens, compressed blocks); a recurrent
/// state snapshots in place (`RecurrentState::open_window`) and records
/// nothing here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowMark {
    pub tokens: usize,
    pub blocks: usize,
}

/// A token mixer the topology can own the state of: how to make a fresh
/// state for one layer, clear it, walk the mixer's step, and roll the state
/// back over a speculative window (`sequence-state.md` §10.2a: a KV cache
/// by a length, a recurrent state by a snapshot and a replay).
pub trait MixerBlock: Mixer + Send + Sync {
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    /// The mixer's decode step as IR, layer `l` (`Visit`, with the layer index
    /// and the capacity a cache-bearing mixer's walk reads).
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId;
    /// Open a speculative window: record what [`Self::rewind_window`]
    /// restores. One window at a time.
    fn open_window(state: &mut Self::State) -> Result<WindowMark>;
    /// Put the state back where the window opened (the caller replays the
    /// accepted prefix afterwards).
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    /// Close the window keeping everything written inside it.
    fn resolve_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    /// Cut the state back to its first `tokens` positions outside a window
    /// (an attention-only lane cut to a reused prefix, `Speculator::begin_at`).
    /// A recurrent state has no length to cut to and refuses.
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()>;
    /// Whether the state runs a window of `steps` positions as one pass
    /// ([`Self::execute_window`]): phase rings with more than `steps` phases
    /// on a recurrent state; always on a KV cache.
    fn ring_holds(state: &Self::State, steps: usize) -> bool;
    /// Whether the state lives on phase rings (a recurrent state at the
    /// `Ring` profile): such a state rewinds by phase, so a window longer
    /// than its rings cannot be stepped through and rewound.
    fn pooled(state: &Self::State) -> bool;
    /// `T` positions of one sequence at once, inside an open window: `x [T,
    /// 1, hidden]` → `[T, 1, hidden]`, position for position the decode
    /// step's arithmetic (the Qwen4-exp `decode_seq` forms), every
    /// intermediate state kept so any prefix can be accepted. Pushes the
    /// mark after each position onto `after` (a KV cache's lengths; a
    /// recurrent state accepts by phase and pushes a blank).
    fn execute_window(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor>;
    /// Accept the first `a` positions of the open window and close it: a
    /// recurrent state to phase `base + a`, a KV cache to `at` (the mark
    /// after position `a`; the opening mark when `a` is 0).
    fn accept_window(state: &mut Self::State, a: usize, at: WindowMark) -> Result<()>;
    /// f32 elements a turn checkpoint of this mixer's state holds in a ring
    /// frame: a recurrent state's `S` and conv window; 0 for a KV cache,
    /// whose checkpoint is a length (the mark).
    fn checkpoint_elems(&self, cap: &StateCapacity) -> usize;
    /// [`Mixer::execute`] over `x`, leaving a turn checkpoint of the state
    /// after the chunk's first `rows` positions in `into` (`[checkpoint_elems]`
    /// f32; `None` for a mixer that keeps none) and returning the mark a
    /// restore needs. `rows == seq` is the state the pass leaves, `rows ==
    /// 0` the one before it; in between, a recurrent state runs its
    /// recurrence a second time over the prefix (the pass is unchanged).
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        rows: usize,
        into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)>;
    /// Put the state back where a checkpoint was taken, outside a window.
    fn restore_checkpoint(
        state: &mut Self::State,
        from: Option<&Tensor>,
        mark: WindowMark,
    ) -> Result<()>;
}

/// A feed-forward the topology can own the state of (`()` for a dense MLP
/// and a resident MoE).
pub trait FfnBlock: Ffn + Send + Sync {
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    fn visit_step(&self, w: &mut Walker, x: NodeId, l: usize) -> NodeId;
    /// `T` rows of a verify window, `x [T, 1, hidden]`, each row the decode
    /// step's arithmetic (a routed MoE through its window form).
    fn execute_window(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
}

/// One layer of a [`Decoder`]: a unit over `[1, seq, hidden]` with its own
/// per-sequence state, rolled back over a speculative window as its mixer's
/// ([`MixerBlock`]).
pub trait Layer: Send + Sync {
    type State: Send + Sync;
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
    fn visit_layer(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId;
    fn open_window(state: &mut Self::State) -> Result<WindowMark>;
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    fn resolve_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()>;
    fn ring_holds(state: &Self::State, steps: usize) -> bool;
    fn pooled(state: &Self::State) -> bool;
    fn execute_window(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor>;
    fn accept_window(state: &mut Self::State, a: usize, at: WindowMark) -> Result<()>;
    fn checkpoint_elems(&self, cap: &StateCapacity) -> usize;
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        rows: usize,
        into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)>;
    fn restore_checkpoint(
        state: &mut Self::State,
        from: Option<&Tensor>,
        mark: WindowMark,
    ) -> Result<()>;
}

/// The token embedding of a [`Decoder`].
pub trait Embed: Send + Sync {
    /// `[b, seq]` ids → `[b, seq, hidden]`.
    fn embed(&self, ids: &Tensor) -> Result<Tensor>;
    fn visit_embed(&self, w: &mut Walker, ids: NodeId) -> NodeId;
}

/// The head of a [`Decoder`]: the last position's residual → logits, in
/// two halves — the final norm over every position ([`Self::normed`], what
/// an MTP head reads as `h_nextn`: llama.cpp `qwen35.cpp` takes it after
/// `output_norm`) and the projection of one position ([`Self::project`]).
pub trait Head: Send + Sync {
    /// `[1, seq, hidden]` → `[1, seq, hidden]`: the residual after the
    /// final norm.
    fn normed(&self, h: &Tensor) -> Result<Tensor>;
    /// `[1, 1, hidden]` (normed) → `[1, vocab]`.
    fn project(&self, normed: &Tensor) -> Result<Tensor>;
    /// `[1, 1, hidden]` → `[1, vocab]`.
    fn head(&self, h: &Tensor) -> Result<Tensor> {
        self.project(&self.normed(h)?)
    }
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
    fn normed(&self, h: &Tensor) -> Result<Tensor> {
        self.norm.execute(h)
    }
    fn project(&self, normed: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = normed.dims3()?;
        if b != 1 || seq != 1 {
            lloom_tensor::bail!("the head takes one position, got {:?}", normed.shape())
        }
        self.proj
            .forward(&normed.reshape((1, hidden))?)?
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
    /// The feed-forward's state is `()` on every block here (a dense MLP, a
    /// resident MoE): the window is the mixer's.
    fn open_window(state: &mut Self::State) -> Result<WindowMark> {
        M::open_window(&mut state.mixer)
    }
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()> {
        M::rewind_window(&mut state.mixer, mark)
    }
    fn resolve_window(state: &mut Self::State, mark: WindowMark) -> Result<()> {
        M::resolve_window(&mut state.mixer, mark)
    }
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()> {
        M::truncate_state(&mut state.mixer, tokens)
    }
    fn ring_holds(state: &Self::State, steps: usize) -> bool {
        M::ring_holds(&state.mixer, steps)
    }
    fn pooled(state: &Self::State) -> bool {
        M::pooled(&state.mixer)
    }
    fn execute_window(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor> {
        let h = self.norm_in.execute(x)?;
        let m = self.mixer.execute_window(&h, &mut state.mixer, after)?;
        let x = (x + m)?;
        let h = self.norm_mid.execute(&x)?;
        let f = self.ffn.execute_window(&h, &mut state.ffn)?;
        x + f
    }
    fn accept_window(state: &mut Self::State, a: usize, at: WindowMark) -> Result<()> {
        M::accept_window(&mut state.mixer, a, at)
    }
    fn checkpoint_elems(&self, cap: &StateCapacity) -> usize {
        self.mixer.checkpoint_elems(cap)
    }
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        rows: usize,
        into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)> {
        let h = self.norm_in.execute(x)?;
        let (m, mark) = self
            .mixer
            .execute_checkpointing(&h, &mut state.mixer, rows, into)?;
        let x = (x + m)?;
        let h = self.norm_mid.execute(&x)?;
        let f = self.ffn.execute(&h, &mut state.ffn)?;
        Ok(((x + f)?, mark))
    }
    fn restore_checkpoint(
        state: &mut Self::State,
        from: Option<&Tensor>,
        mark: WindowMark,
    ) -> Result<()> {
        M::restore_checkpoint(&mut state.mixer, from, mark)
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
    fn open_window(state: &mut Self::State) -> Result<WindowMark> {
        match state {
            HybridState::A(s) => A::open_window(s),
            HybridState::B(s) => B::open_window(s),
        }
    }
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()> {
        match state {
            HybridState::A(s) => A::rewind_window(s, mark),
            HybridState::B(s) => B::rewind_window(s, mark),
        }
    }
    fn resolve_window(state: &mut Self::State, mark: WindowMark) -> Result<()> {
        match state {
            HybridState::A(s) => A::resolve_window(s, mark),
            HybridState::B(s) => B::resolve_window(s, mark),
        }
    }
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()> {
        match state {
            HybridState::A(s) => A::truncate_state(s, tokens),
            HybridState::B(s) => B::truncate_state(s, tokens),
        }
    }
    fn ring_holds(state: &Self::State, steps: usize) -> bool {
        match state {
            HybridState::A(s) => A::ring_holds(s, steps),
            HybridState::B(s) => B::ring_holds(s, steps),
        }
    }
    fn pooled(state: &Self::State) -> bool {
        match state {
            HybridState::A(s) => A::pooled(s),
            HybridState::B(s) => B::pooled(s),
        }
    }
    fn execute_window(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor> {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.execute_window(x, s, after),
            (Hybrid::B(m), HybridState::B(s)) => m.execute_window(x, s, after),
            _ => lloom_tensor::bail!("a hybrid layer's state is the other mixer's"),
        }
    }
    fn accept_window(state: &mut Self::State, a: usize, at: WindowMark) -> Result<()> {
        match state {
            HybridState::A(s) => A::accept_window(s, a, at),
            HybridState::B(s) => B::accept_window(s, a, at),
        }
    }
    fn checkpoint_elems(&self, cap: &StateCapacity) -> usize {
        match self {
            Hybrid::A(m) => m.checkpoint_elems(cap),
            Hybrid::B(m) => m.checkpoint_elems(cap),
        }
    }
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        rows: usize,
        into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)> {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.execute_checkpointing(x, s, rows, into),
            (Hybrid::B(m), HybridState::B(s)) => m.execute_checkpointing(x, s, rows, into),
            _ => lloom_tensor::bail!("a hybrid layer's state is the other mixer's"),
        }
    }
    fn restore_checkpoint(
        state: &mut Self::State,
        from: Option<&Tensor>,
        mark: WindowMark,
    ) -> Result<()> {
        match state {
            HybridState::A(s) => A::restore_checkpoint(s, from, mark),
            HybridState::B(s) => B::restore_checkpoint(s, from, mark),
        }
    }
}

/// A decoder's per-sequence state: one per layer, the position, and the
/// speculative window a verify rolls the layers back over
/// (`sequence-state.md` §10.2a, `speculation.md` §10.2h): every layer's
/// mark, the position, and the rows of the retained probe when it opened.
pub struct DecoderState<L: Layer> {
    pub layers: Vec<L::State>,
    pub pos: usize,
    window: Option<DecoderWindow>,
    /// The final norm's output per position (`h_nextn`), retained for an
    /// MTP drafter while [`Self::set_want_nextn`] is on.
    nextn: RowLog,
    want_nextn: bool,
    nextn_drops: u64,
    /// The turn checkpoints (`sequence-reuse-design.md` §2.3b): `ckpt_slots`
    /// frames of one ring, each every layer's [`Layer::checkpoint_elems`]
    /// f32s laid end to end (`ckpt_layout`: offset and count per layer), the
    /// KV layers' lengths in the marks. The ring is allocated at a
    /// sequence's first checkpoint and kept; a slot is reused oldest first.
    ckpt_slots: usize,
    ckpt_layout: Vec<(usize, usize)>,
    ckpt_ring: Option<Ring>,
    checkpoints: Vec<TurnCheckpoint>,
    ckpt_next: usize,
    /// Where the passes that follow should leave their checkpoint
    /// ([`Self::checkpoint_at`]).
    ckpt_want: Option<usize>,
    /// Until a caller hints, every multi-token pass checkpoints its end; a
    /// hinted state is driven by its hints alone (a tiled prefill's other
    /// tiles take none), until a reset.
    ckpt_auto: bool,
}

#[derive(Debug)]
struct DecoderWindow {
    pos: usize,
    marks: Vec<WindowMark>,
    retained: usize,
}

/// A turn checkpoint: the position it stands at and, per layer, the mark a
/// restore needs beside the ring frame (a KV cache's lengths).
#[derive(Debug, Clone)]
struct TurnCheckpoint {
    pos: usize,
    valid: bool,
    marks: Vec<WindowMark>,
}

impl<L: Layer> DecoderState<L> {
    pub fn new(layers: Vec<L::State>) -> Self {
        Self {
            layers,
            pos: 0,
            window: None,
            nextn: RowLog::default(),
            want_nextn: false,
            nextn_drops: 0,
            ckpt_slots: 0,
            ckpt_layout: Vec::new(),
            ckpt_ring: None,
            checkpoints: Vec::new(),
            ckpt_next: 0,
            ckpt_want: None,
            ckpt_auto: true,
        }
    }

    /// Keep `slots` turn checkpoints, each `elems[l]` f32s for layer `l`.
    pub fn with_checkpoints(mut self, elems: &[usize], slots: usize) -> Self {
        let mut off = 0;
        self.ckpt_layout = elems
            .iter()
            .map(|&n| {
                let o = off;
                off += n;
                (o, n)
            })
            .collect();
        self.ckpt_slots = slots;
        self
    }

    fn ckpt_elems(&self) -> usize {
        self.ckpt_layout.iter().map(|(_, n)| n).sum()
    }

    /// The checkpoint ring's bytes (allocated at the first checkpoint).
    pub fn checkpoint_bytes(&self) -> usize {
        self.ckpt_slots * self.ckpt_elems() * DType::F32.size_in_bytes()
    }

    pub fn checkpoint_slots(&self) -> usize {
        self.ckpt_slots
    }

    /// The positions the valid checkpoints stand at.
    pub fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints
            .iter()
            .filter(|c| c.valid)
            .map(|c| c.pos)
            .collect()
    }

    /// Have the passes that follow leave their checkpoint at `pos` (one a
    /// pass covers takes it there instead of at its end). Ignored without
    /// checkpoint slots.
    pub fn checkpoint_at(&mut self, pos: usize) {
        if self.ckpt_slots > 0 {
            self.ckpt_want = Some(pos);
            self.ckpt_auto = false;
        }
    }

    pub fn checkpoint_hint(&self) -> Option<usize> {
        self.ckpt_want
    }

    /// Frame `slot`'s view for layer `l`, `[elems]` f32; `None` for a layer
    /// whose checkpoint is its mark alone.
    fn frame_view(&self, slot: usize, l: usize) -> Result<Option<Tensor>> {
        let (off, n) = self.ckpt_layout[l];
        if n == 0 {
            return Ok(None);
        }
        let ring = self
            .ckpt_ring
            .as_ref()
            .ok_or_else(|| lloom_tensor::Error::msg("decoder state: no checkpoint ring"))?;
        Ok(Some(ring.frame(slot)?.narrow(0, off, n)?))
    }

    /// The slot the next checkpoint, at `pos`, goes in: the oldest, its
    /// frame rewritten; the ring allocated at the first (one request, outside
    /// every plan, kept for the sequence).
    fn open_checkpoint(&mut self, pos: usize, dev: &Device) -> Result<usize> {
        let elems = self.ckpt_elems();
        if elems > 0 && self.ckpt_ring.is_none() {
            self.ckpt_ring = Some(Ring::allocate(
                self.ckpt_slots,
                elems,
                DType::F32,
                dev,
                zs::NN_DECODER_CHECKPOINTS,
                "decoder-turn-checkpoints",
            )?);
        }
        let slot = self.ckpt_next % self.ckpt_slots;
        let c = TurnCheckpoint {
            pos,
            valid: false,
            marks: Vec::new(),
        };
        if slot < self.checkpoints.len() {
            self.checkpoints[slot] = c;
        } else {
            self.checkpoints.push(c);
        }
        self.ckpt_next = (slot + 1) % self.ckpt_slots;
        Ok(slot)
    }

    fn close_checkpoint(&mut self, slot: usize, marks: Vec<WindowMark>) {
        let c = &mut self.checkpoints[slot];
        c.marks = marks;
        c.valid = true;
        if self.ckpt_want == Some(c.pos) {
            self.ckpt_want = None;
        }
    }

    /// Checkpoints past `pos` describe positions the state no longer holds.
    fn forget_checkpoints_past(&mut self, pos: usize) {
        for c in &mut self.checkpoints {
            if c.pos > pos {
                c.valid = false;
            }
        }
    }

    fn checkpoint_index(&self, pos: usize) -> Option<usize> {
        self.checkpoints
            .iter()
            .position(|c| c.valid && c.pos == pos)
    }

    /// Whether [`Self::restore`] can put the state back at exactly `pos`: a
    /// checkpoint stands there and no window is open.
    pub fn can_restore(&self, pos: usize) -> bool {
        self.window.is_none() && pos <= self.pos && self.checkpoint_index(pos).is_some()
    }

    /// Put every layer back where the checkpoint at `pos` was taken — a KV
    /// cache to its length, a recurrent state to the frame's copy — and the
    /// position with them; what the discarded positions retained goes.
    pub fn restore(&mut self, pos: usize) -> Result<()> {
        let Some(idx) = self.checkpoint_index(pos).filter(|_| self.can_restore(pos)) else {
            lloom_tensor::bail!(
                "decoder state: no turn checkpoint stands at {pos} (position {}, checkpoints at {:?}{})",
                self.pos,
                self.checkpoint_positions(),
                if self.window.is_some() {
                    ", a window is open"
                } else {
                    ""
                }
            )
        };
        let marks = self.checkpoints[idx].marks.clone();
        if marks.len() != self.layers.len() {
            lloom_tensor::bail!(
                "decoder state: a checkpoint of {} layers for {} layers",
                marks.len(),
                self.layers.len()
            )
        }
        let frames = (0..self.layers.len())
            .map(|l| self.frame_view(idx, l))
            .collect::<Result<Vec<_>>>()?;
        for ((s, frame), mark) in self.layers.iter_mut().zip(&frames).zip(&marks) {
            L::restore_checkpoint(s, frame.as_ref(), *mark)?;
        }
        self.pos = pos;
        self.nextn.clear();
        self.forget_checkpoints_past(pos);
        Ok(())
    }

    /// Open a speculative window over every layer's state and the
    /// position. One at a time.
    pub fn open_window(&mut self) -> Result<()> {
        if self.window.is_some() {
            lloom_tensor::bail!("decoder state: a speculative window is already open")
        }
        let mut marks = Vec::with_capacity(self.layers.len());
        for s in &mut self.layers {
            marks.push(L::open_window(s)?);
        }
        self.window = Some(DecoderWindow {
            pos: self.pos,
            marks,
            retained: self.nextn.rows(),
        });
        Ok(())
    }

    /// Put every layer and the position back where the window opened; what
    /// the window's passes retained goes with them. The caller replays the
    /// accepted prefix.
    pub fn rewind_window(&mut self) -> Result<()> {
        let Some(w) = self.window.take() else {
            lloom_tensor::bail!("decoder state: no speculative window is open")
        };
        for (s, m) in self.layers.iter_mut().zip(&w.marks) {
            L::rewind_window(s, *m)?;
        }
        self.pos = w.pos;
        self.forget_checkpoints_past(self.pos);
        self.nextn.keep_first(w.retained)
    }

    /// Close the window keeping everything the passes wrote.
    pub fn resolve_window(&mut self) -> Result<()> {
        let Some(w) = self.window.take() else {
            lloom_tensor::bail!("decoder state: no speculative window is open")
        };
        for (s, m) in self.layers.iter_mut().zip(&w.marks) {
            L::resolve_window(s, *m)?;
        }
        Ok(())
    }

    pub fn window_open(&self) -> bool {
        self.window.is_some()
    }

    /// Whether every layer runs a window of `steps` positions as one pass
    /// (`Layer::ring_holds`): the `Ring` profile with a `StateCapacity::window`
    /// of at least `steps`.
    pub fn ring_holds(&self, steps: usize) -> bool {
        self.layers.iter().all(|s| L::ring_holds(s, steps))
    }

    /// Whether any layer's state lives on phase rings (the `Ring` profile
    /// with a `StateCapacity::window`).
    pub fn pooled(&self) -> bool {
        self.layers.iter().any(L::pooled)
    }

    /// Accept the first `a` positions of the open window and close it:
    /// every layer to the mark after position `a` (`after[l][a − 1]`; the
    /// opening mark when `a` is 0, a rewind), the position, the retained
    /// probe cut to `a` rows of the window. The window's positions past `a`
    /// cost nothing — no replay.
    pub fn accept_window(&mut self, a: usize, after: &[Vec<WindowMark>]) -> Result<()> {
        let Some(w) = self.window.take() else {
            lloom_tensor::bail!("decoder state: no speculative window is open")
        };
        if after.len() != self.layers.len() {
            lloom_tensor::bail!(
                "decoder state: {} layers of marks for {} layers",
                after.len(),
                self.layers.len()
            )
        }
        for (l, s) in self.layers.iter_mut().enumerate() {
            let at = if a == 0 {
                w.marks[l]
            } else {
                match after[l].get(a - 1) {
                    Some(m) => *m,
                    None => lloom_tensor::bail!(
                        "decoder state: layer {l} accepts {a} positions of a {}-position window",
                        after[l].len()
                    ),
                }
            };
            L::accept_window(s, a, at)?;
        }
        self.pos = w.pos + a;
        self.forget_checkpoints_past(self.pos);
        self.nextn.keep_first(w.retained + a)
    }

    /// Cut every layer back to its first `tokens` positions outside a
    /// window (an attention-only lane, `MixerBlock::truncate_state`).
    pub fn truncate(&mut self, tokens: usize) -> Result<()> {
        if self.window.is_some() {
            lloom_tensor::bail!("decoder state: truncating under an open window")
        }
        if tokens > self.pos {
            lloom_tensor::bail!(
                "decoder state: truncating {} positions to {tokens}",
                self.pos
            )
        }
        for s in &mut self.layers {
            L::truncate_state(s, tokens)?;
        }
        self.pos = tokens;
        self.forget_checkpoints_past(tokens);
        self.nextn.clear();
        Ok(())
    }

    /// Retain `h_nextn` on every forward from now (`on`); returns the
    /// previous setting.
    pub fn set_want_nextn(&mut self, on: bool) -> bool {
        std::mem::replace(&mut self.want_nextn, on)
    }

    pub fn want_nextn(&self) -> bool {
        self.want_nextn
    }

    /// The rows retained since the last take, `[1, positions, hidden]`, or
    /// `None` if nothing was.
    pub fn take_nextn(&mut self) -> Result<Option<Tensor>> {
        self.nextn.take()
    }

    /// Discard whatever is retained, counting it as a dropped probe.
    pub fn drop_nextn(&mut self) {
        if !self.nextn.is_empty() {
            self.nextn_drops += 1;
            self.nextn.clear();
        }
    }

    pub fn nextn_drops(&self) -> u64 {
        self.nextn_drops
    }

    fn keep_nextn(&mut self, rows: Tensor) {
        self.nextn.push(rows);
    }
}

impl<L: Layer> CacheReset for DecoderState<L> {
    fn reset(&mut self) {
        for s in &mut self.layers {
            L::reset_state(s);
        }
        self.pos = 0;
        self.window = None;
        self.nextn.clear();
        // The ring stays (the next sequence writes over it); its contents
        // describe nothing now.
        for c in &mut self.checkpoints {
            c.valid = false;
        }
        self.ckpt_want = None;
        self.ckpt_auto = true;
    }
}

/// No standing window: the horizon is zero and a driver decodes without
/// lookahead (`CacheRewind`'s contract for a cache with no rewindable arm).
/// Past it, a turn checkpoint ([`DecoderState::restore`]) puts the state
/// back at the position a multi-token pass left one at.
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
    fn can_restore(&self, pos: usize) -> bool {
        DecoderState::can_restore(self, pos)
    }
    fn restore(&mut self, pos: usize) -> Result<()> {
        DecoderState::restore(self, pos)
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
    /// The residual width, for a probe of no rows.
    hidden: usize,
    /// The device the weights live on (a zero-row probe is made there).
    device: Device,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: &'static str,
        cfg: C,
        embed: E,
        layers: Vec<L>,
        head: H,
        cap: StateCapacity,
        hidden: usize,
        device: &Device,
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
            hidden,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn device_of(&self, _state: &DecoderState<L>) -> Result<Device> {
        Ok(self.device.clone())
    }

    fn hidden_of(&self) -> usize {
        self.hidden
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

    /// The bytes a state's turn-checkpoint ring takes at this capacity
    /// (`StateCapacity::checkpoints` frames of every layer's
    /// [`Layer::checkpoint_elems`] f32s).
    pub fn checkpoint_bytes(&self) -> usize {
        self.cap.checkpoints * self.checkpoint_frame_bytes()
    }

    /// The bytes one turn checkpoint takes.
    pub fn checkpoint_frame_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.checkpoint_elems(&self.cap))
            .sum::<usize>()
            * DType::F32.size_in_bytes()
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
        // A hint at a position already passed is stale; one a checkpoint
        // already stands at is met.
        if let Some(w) = state.ckpt_want {
            if w < state.pos || state.checkpoint_index(w).is_some() {
                state.ckpt_want = None;
            }
        }
        let ckpt_on = state.ckpt_slots > 0 && state.window.is_none();
        let mut last = None;
        let mut start = 0;
        while start < seq {
            let len = (seq - start).min(self.chunk);
            let at = state.pos;
            // The checkpoint this chunk leaves: at the hinted position when
            // the chunk covers it, else at the end of a multi-token pass
            // with no hint pending.
            let rows = if !ckpt_on {
                None
            } else if let Some(w) = state.ckpt_want.filter(|&w| w >= at && w <= at + len) {
                Some(w - at)
            } else if state.ckpt_auto && state.ckpt_want.is_none() && start + len == seq && seq >= 2
            {
                Some(len)
            } else {
                None
            };
            let mut h = embeds.narrow(1, start, len)?;
            match rows {
                Some(rows) => {
                    let slot = state.open_checkpoint(at + rows, embeds.device())?;
                    let mut marks = Vec::with_capacity(self.layers.len());
                    for (l, layer) in self.layers.iter().enumerate() {
                        let frame = state.frame_view(slot, l)?;
                        let (out, mark) = layer.execute_checkpointing(
                            &h,
                            &mut state.layers[l],
                            rows,
                            frame.as_ref(),
                        )?;
                        h = out;
                        marks.push(mark);
                    }
                    state.close_checkpoint(slot, marks);
                }
                None => {
                    for (l, layer) in self.layers.iter().enumerate() {
                        h = layer.execute(&h, &mut state.layers[l])?;
                    }
                }
            }
            state.pos += len;
            last = Some(if state.want_nextn {
                // The final norm over the whole chunk, retained for the
                // drafter; the head projects its last row.
                let n = self.head.normed(&h)?;
                state.keep_nextn(n.clone());
                self.head.project(&n.narrow(1, len - 1, 1)?)?
            } else {
                self.head.head(&h.narrow(1, len - 1, 1)?)?
            });
            start += len;
        }
        last.ok_or_else(|| lloom_tensor::Error::msg("seq >= 1"))
    }

    /// One decode step of `tok` at the state's position, returning the
    /// device-side argmax (no readback).
    fn step_argmax(&self, tok: u32, state: &mut DecoderState<L>, dev: &Device) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![tok], (1, 1), dev)?;
        let logits = CausalLM::forward(self, &input, state.pos, state)?;
        logits.flatten_all()?.argmax(D::Minus1)?.reshape(1)
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
        let elems: Vec<usize> = self
            .layers
            .iter()
            .map(|l| l.checkpoint_elems(&self.cap))
            .collect();
        Ok(DecoderState::new(layers).with_checkpoints(&elems, self.cap.checkpoints))
    }

    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        cache.checkpoint_at(pos);
        Ok(())
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

impl<P, C, E, L, H> crate::profile::Shaped for Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    type Profile = P;
}

/// A verify window encoded and read back, awaiting the accept decision.
#[derive(Debug)]
pub struct DecoderPending {
    /// The target's argmax at each of the `k + 1` window positions.
    argmax: Vec<u32>,
    /// The window's input tokens `[anchor, proposed…]`.
    tokens: Vec<u32>,
    /// Whether `h_nextn` was being retained before the window.
    want_was: bool,
    /// The window ran as one pass: every layer's mark after each position,
    /// for the accept by phase / by length. `None`: the sequential steps,
    /// rewound and replayed on a partial accept.
    after: Option<Vec<Vec<WindowMark>>>,
}

/// The verify on the generic topology: the window `[anchor, proposed…]`
/// inside an open window on every layer's state, the per-position argmax
/// read back once. On Metal with the rings holding it (the `Ring` profile,
/// `StateCapacity::window`) the window is ONE pass — the stems and
/// projections once over the `k + 1` rows, the recurrences and attentions
/// position for position through the decode kernels' window forms
/// (`GatedDeltaNet::decode_seq`, `QsaAttention::decode_seq`), every
/// intermediate state in the phase rings — accepted by phase, no replay.
/// Otherwise `k + 1` sequential decode steps, rewound and the accepted
/// prefix replayed on a partial accept (§10.2a: the recurrence is
/// replayable; the CPU's gemm sums in a row-count-dependent order, so the
/// sequential form is the reference there). Either way greedy's own math
/// position by position, so the accepted stream is byte-identical to
/// greedy by construction (`speculation.md` §10.2n). The probe is `h_nextn`
/// at each accepted position.
impl<P, C, E, L, H> Probes<NextnHidden> for Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    type Pending = DecoderPending;
    type PendingRows = Vec<DecoderPending>;

    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Tensor)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "{} verify_and_probe: one sequence; got {} drafts and {} anchors",
                self.family,
                drafts.len(),
                anchors.len()
            )
        }
        let pending = self.verify_enqueue(&drafts[0], &anchors[0], cache)?;
        let (accepted, probe) = self.verify_complete(pending, cache, eos_ids)?;
        Ok((vec![accepted], probe))
    }

    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Self::Cache,
    ) -> Result<DecoderPending> {
        if anchor.pos != cache.pos {
            lloom_tensor::bail!(
                "{} verify: anchor pos {} != the state's position {}",
                self.family,
                anchor.pos,
                cache.pos
            )
        }
        let proposed: Vec<u32> = match &draft.tokens {
            DraftTokens::Host(v) => v.clone(),
            DraftTokens::Device(t) => t.flatten_all()?.to_vec1::<u32>()?,
        };
        let mut tokens = Vec::with_capacity(proposed.len() + 1);
        tokens.push(anchor.last);
        tokens.extend_from_slice(&proposed);
        // Anything retained untaken is a dropped probe; the window retains
        // its own rows.
        cache.drop_nextn();
        let want_was = cache.set_want_nextn(true);
        let dev = self.device_of(cache)?;
        let t = tokens.len();
        // A state on rings rewinds by phase: a window longer than its rings
        // cannot be stepped through and put back (the rings are sized to
        // the drafter's `k + 1`, `StateCapacity::window`).
        if cache.pooled() && !cache.ring_holds(t) {
            cache.set_want_nextn(want_was);
            lloom_tensor::bail!(
                "{} verify: a window of {t} positions on rings built for a shorter one \
                 (`StateCapacity::window` must be >= {t})",
                self.family
            )
        }
        cache.open_window()?;
        if dev.is_metal() && cache.ring_holds(t) {
            let input = Tensor::from_vec(tokens.clone(), (t, 1), &dev)?;
            let mut h = self.embed.embed(&input)?; // [T, 1, hidden]
            let mut after: Vec<Vec<WindowMark>> = Vec::with_capacity(self.layers.len());
            for (l, layer) in self.layers.iter().enumerate() {
                let mut marks = Vec::with_capacity(t);
                h = layer.execute_window(&h, &mut cache.layers[l], &mut marks)?;
                after.push(marks);
            }
            cache.pos += t;
            let n = self.head.normed(&h)?; // [T, 1, hidden]
            cache.keep_nextn(n.reshape((1, t, self.hidden))?);
            let mut argmaxes = Vec::with_capacity(t);
            for p in 0..t {
                let logits = self.head.project(&n.narrow(0, p, 1)?)?;
                argmaxes.push(logits.flatten_all()?.argmax(D::Minus1)?.reshape(1)?);
            }
            let argmax = Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?;
            return Ok(DecoderPending {
                argmax,
                tokens,
                want_was,
                after: Some(after),
            });
        }
        let mut argmaxes = Vec::with_capacity(t);
        for &tok in &tokens {
            argmaxes.push(self.step_argmax(tok, cache, &dev)?);
        }
        let argmax = Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?;
        Ok(DecoderPending {
            argmax,
            tokens,
            want_was,
            after: None,
        })
    }

    fn verify_complete(
        &self,
        pending: DecoderPending,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Tensor)> {
        let DecoderPending {
            argmax,
            tokens,
            want_was,
            after,
        } = pending;
        let a = accept_prefix(&argmax, &tokens[1..], eos_ids).len();
        let accepted: Vec<u32> = argmax[..a].to_vec();
        if let Some(after) = after {
            // The accepted prefix by phase and by length: no replay.
            cache.accept_window(a, &after)?;
        } else if a == tokens.len() {
            cache.resolve_window()?;
        } else {
            cache.rewind_window()?;
            cache.drop_nextn();
            let dev = self.device_of(cache)?;
            for &t in &tokens[..a] {
                let _ = self.step_argmax(t, cache, &dev)?;
            }
        }
        let probe = cache.take_nextn()?;
        cache.set_want_nextn(want_was);
        let Some(probe) = probe else {
            lloom_tensor::bail!("{} verify: the window retained no probe", self.family)
        };
        let rows = probe.dim(1)?;
        if rows != a {
            lloom_tensor::bail!(
                "{} verify: the probe covers {rows} positions for {a} accepted",
                self.family
            )
        }
        Ok((accepted, probe))
    }

    fn take_probe(&self, cache: &mut Self::Cache) -> Result<Tensor> {
        match cache.take_nextn()? {
            Some(t) => Ok(t),
            None => {
                let dev = self.device_of(cache)?;
                Tensor::zeros((1, 0, self.hidden_of()), DType::F32, &dev)
            }
        }
    }

    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        cache.nextn_drops()
    }

    /// The window `[anchor]` is a decode step.
    fn verifies_anchor_alone(&self) -> bool {
        true
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
    /// At the `Ring` profile with a window, the two channels on their own
    /// phase rings (`cap.window + 1` phases); else an unpooled pair that a
    /// window snapshots.
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State> {
        if <P::Rewind as crate::profile::Rewind>::REWINDABLE && cap.window > 0 {
            return self.pooled_state(dev, cap.window + 1);
        }
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
    /// The recurrence's two channels snapshot in place
    /// (`RecurrentState::open_window`); nothing to mark.
    fn open_window(state: &mut Self::State) -> Result<WindowMark> {
        state.s.open_window()?;
        state.conv.open_window()?;
        Ok(WindowMark::default())
    }
    fn rewind_window(state: &mut Self::State, _mark: WindowMark) -> Result<()> {
        state.s.rewind_to_window()?;
        state.conv.rewind_to_window()
    }
    fn resolve_window(state: &mut Self::State, _mark: WindowMark) -> Result<()> {
        state.s.resolve_window()?;
        state.conv.resolve_window()
    }
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()> {
        if tokens == 0 {
            Self::reset_state(state);
            return Ok(());
        }
        lloom_tensor::bail!("a recurrent state has no length to cut to {tokens} positions")
    }
    fn ring_holds(state: &Self::State, steps: usize) -> bool {
        state.s.ring_holds(steps) && state.conv.ring_holds(steps)
    }
    fn pooled(state: &Self::State) -> bool {
        state.s.is_pooled() || state.conv.is_pooled()
    }
    fn execute_window(
        &self,
        x: &Tensor,
        st: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let out = self.decode_seq(x, st.layer, &mut st.s, &mut st.conv)?;
        after.extend(std::iter::repeat_n(WindowMark::default(), t));
        Ok(out)
    }
    fn accept_window(state: &mut Self::State, a: usize, _at: WindowMark) -> Result<()> {
        state.s.accept_window(a)?;
        state.conv.accept_window(a)
    }
    fn checkpoint_elems(&self, _cap: &StateCapacity) -> usize {
        let (s, conv) = self.checkpoint_parts();
        s + conv
    }
    /// `S` and the conv window into the frame: a decode step's (`rows` 0 or
    /// 1: before or after it), a prefill's at `rows`
    /// (`GatedDeltaNet::prefill_checkpointing`).
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        st: &mut Self::State,
        rows: usize,
        into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)> {
        let Some(frame) = into else {
            lloom_tensor::bail!("a recurrent state's checkpoint needs a ring frame")
        };
        let (ns, nc) = self.checkpoint_parts();
        let s_view = frame.narrow(0, 0, ns)?;
        let c_view = frame.narrow(0, ns, nc)?;
        let ck = crate::models::qwen4exp::GdnCheckpoint {
            s: &s_view,
            conv: &c_view,
        };
        let seq = x.dim(1)?;
        if rows > seq {
            lloom_tensor::bail!("deltanet checkpoint: {rows} rows of a {seq}-position chunk")
        }
        let out = if seq == 1 {
            if rows == 0 {
                self.store_checkpoint(st.layer, &st.s, &st.conv, x.device(), &ck)?;
            }
            let out = self.decode(x, st.layer, &mut st.s, &mut st.conv)?;
            if rows == 1 {
                self.store_checkpoint(st.layer, &st.s, &st.conv, x.device(), &ck)?;
            }
            out
        } else {
            self.prefill_checkpointing(x, st.layer, &mut st.s, &mut st.conv, rows, &ck)?
        };
        Ok((out, WindowMark::default()))
    }
    /// The frame's `S` and window written into the channels (in place, or
    /// the pool's next phase), at the slabs' shape and dtype.
    fn restore_checkpoint(
        st: &mut Self::State,
        from: Option<&Tensor>,
        _mark: WindowMark,
    ) -> Result<()> {
        let Some(frame) = from else {
            lloom_tensor::bail!("a recurrent state restores from a ring frame")
        };
        let l = st.layer;
        let (Some(s_cur), Some(c_cur)) = (st.s.get(l), st.conv.get(l)) else {
            lloom_tensor::bail!("a recurrent state restores over a written state")
        };
        let (s_dims, s_dt) = (s_cur.dims().to_vec(), s_cur.dtype());
        let (c_dims, c_dt) = (c_cur.dims().to_vec(), c_cur.dtype());
        let ns: usize = s_dims.iter().product();
        let nc: usize = c_dims.iter().product();
        if frame.elem_count() != ns + nc {
            lloom_tensor::bail!(
                "a recurrent state's frame holds {} elements for a state of {}",
                frame.elem_count(),
                ns + nc
            )
        }
        let s = frame
            .narrow(0, 0, ns)?
            .reshape(s_dims)?
            .to_dtype(s_dt)?
            .contiguous()?;
        let w = frame
            .narrow(0, ns, nc)?
            .reshape(c_dims)?
            .to_dtype(c_dt)?
            .contiguous()?;
        st.s.write(l, s)?;
        st.conv.write(l, w)
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
    /// A KV cache rewinds by its lengths.
    fn open_window(state: &mut Self::State) -> Result<WindowMark> {
        let (tokens, blocks) = state.lens();
        Ok(WindowMark { tokens, blocks })
    }
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()> {
        state.truncate(mark.tokens, mark.blocks);
        Ok(())
    }
    fn resolve_window(_state: &mut Self::State, _mark: WindowMark) -> Result<()> {
        Ok(())
    }
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()> {
        if tokens > state.len() {
            lloom_tensor::bail!(
                "a KV cache of {} positions cannot be cut to {tokens}",
                state.len()
            )
        }
        state.truncate(tokens, state.n_blocks().min(tokens));
        Ok(())
    }
    fn ring_holds(_state: &Self::State, _steps: usize) -> bool {
        true
    }
    fn pooled(_state: &Self::State) -> bool {
        false
    }
    fn execute_window(
        &self,
        x: &Tensor,
        kv: &mut Self::State,
        after: &mut Vec<WindowMark>,
    ) -> Result<Tensor> {
        let (out, lens) = self.decode_seq(x, kv)?;
        after.extend(
            lens.into_iter()
                .map(|(tokens, blocks)| WindowMark { tokens, blocks }),
        );
        Ok(out)
    }
    fn accept_window(state: &mut Self::State, _a: usize, at: WindowMark) -> Result<()> {
        state.truncate(at.tokens, at.blocks);
        Ok(())
    }
    /// A KV cache's checkpoint is its lengths at the position.
    fn checkpoint_elems(&self, _cap: &StateCapacity) -> usize {
        0
    }
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        kv: &mut Self::State,
        rows: usize,
        _into: Option<&Tensor>,
    ) -> Result<(Tensor, WindowMark)> {
        let seq = x.dim(1)?;
        if rows > seq {
            lloom_tensor::bail!("qsa checkpoint: {rows} rows of a {seq}-position chunk")
        }
        let (tokens, blocks) = kv.lens();
        let out = Mixer::execute(self, x, kv)?;
        let mark = if rows == seq {
            let (tokens, blocks) = kv.lens();
            WindowMark { tokens, blocks }
        } else if self.indexer().is_some() {
            lloom_tensor::bail!(
                "qsa checkpoint: a position inside a pass is not kept with an indexer (its \
                 compressed blocks are the pass's)"
            )
        } else {
            WindowMark {
                tokens: tokens + rows,
                blocks,
            }
        };
        Ok((out, mark))
    }
    fn restore_checkpoint(
        kv: &mut Self::State,
        _from: Option<&Tensor>,
        mark: WindowMark,
    ) -> Result<()> {
        if mark.tokens > kv.len() {
            lloom_tensor::bail!(
                "a KV cache of {} positions cannot be put back at {}",
                kv.len(),
                mark.tokens
            )
        }
        kv.truncate(mark.tokens, mark.blocks);
        Ok(())
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
    /// The rows through the same projections a decode step takes (the
    /// shared expert's window form on Qwen4-exp).
    fn execute_window(&self, x: &Tensor, _state: &mut ()) -> Result<Tensor> {
        self.forward(x)
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
    /// The routed experts' window form: every row real, the resident tier's
    /// plain groups (`MoeLayer::decode_window`).
    fn execute_window(&self, x: &Tensor, _state: &mut ()) -> Result<Tensor> {
        let t = x.dim(0)?;
        let (y, _) = self.decode_window(
            x,
            crate::models::qwen4exp::moe::WindowPlanKind::Host {
                real_rows: t,
                loaded_rows: t,
            },
        )?;
        Ok(y)
    }
}
