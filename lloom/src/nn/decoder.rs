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
use crate::nn::row_log::RowLog;
use crate::nn::speculator::{accept_prefix, Anchor, Draft, DraftTokens, NextnHidden, Probes};
use crate::nn::turn_checkpoint::CheckpointRing;
use crate::nn::var_builder::VarBuilder;
use crate::optimizer::{NodeId, Shape, Visit, Walker};
use crate::profile::Profile;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor, D};
use std::marker::PhantomData;

/// What a sequence's state is sized by: the KV capacity in tokens and the
/// KV store's precision, and the three things it keeps besides the current
/// state, each its own field because each buys its own storage and work.
/// How far a producer may run ahead is the driver's
/// (`DecodeOptions::adaptive_lead`), not the state's: nothing here is
/// sized from it, and it bounds no recovery (`sequence-reuse-design.md`
/// §2.3e).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateCapacity {
    pub kv_tokens: usize,
    pub kv_precision: KvPrecision,
    /// The longest verify window (positions) a state built at a rewindable
    /// profile runs as ONE pass over phase rings. 0: a window runs as
    /// sequential steps inside a snapshot.
    pub verify: usize,
    /// How a decode step outside a verify window stays undoable.
    pub recovery: DecodeRecovery,
    /// Turn checkpoints a state keeps (`DecoderState::restore`): the last
    /// `checkpoints` multi-token passes' states, in a ring. 0: none, every
    /// request past the horizon is a fresh prefill.
    pub checkpoints: usize,
    /// One frame more in the ring, for a checkpoint a session pins
    /// (`CheckpointRing::pin`): its head's, which the turn checkpoints never
    /// take (`sessions.md` §8).
    pub pin: bool,
}

/// What a decode step leaves undoable after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeRecovery {
    /// Nothing: the step is final once it runs.
    #[default]
    None,
    /// The standing window: the step stays undoable until the next forward
    /// (`CacheRewind`, horizon 1).
    Standing,
}

impl DecodeRecovery {
    /// The decode steps it keeps undoable.
    pub fn depth(self) -> usize {
        match self {
            Self::None => 0,
            Self::Standing => 1,
        }
    }
}

impl StateCapacity {
    /// No verify window, no recovery, no checkpoints: a state that holds
    /// its current value and nothing else.
    pub fn new(kv_tokens: usize) -> Self {
        Self {
            kv_tokens,
            kv_precision: KvPrecision::dense(DType::F32),
            verify: 0,
            recovery: DecodeRecovery::None,
            checkpoints: 0,
            pin: false,
        }
    }

    /// The frames a state's checkpoint ring holds: the turn checkpoints'
    /// and the pin's.
    pub fn checkpoint_frames(&self) -> usize {
        self.checkpoints + usize::from(self.pin)
    }

    /// Phases a recurrent channel's ring holds at a rewindable profile:
    /// the base, and as many past it as the longer of the verify window
    /// and the decode recovery. The two never hold versions at once (a
    /// verify settles the standing window before it opens), so they share
    /// the ring. `None`: the channel is unpooled, a slab and its spare.
    pub fn recurrent_phases(&self, rewindable: bool) -> Option<usize> {
        let positions = self.verify.max(self.recovery.depth());
        (rewindable && positions > 0).then_some(positions + 1)
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

/// The turn checkpoints one chunk of a pass leaves
/// (`CheckpointRing::takes_for`), in row order: per take, the rows into the
/// chunk and the ring frame it is copied into (`None` for a mixer whose
/// checkpoint is its mark alone). At most [`MAX_TAKES`].
pub type ChunkTakes<'a> = [(usize, Option<&'a Tensor>)];

/// The most turn checkpoints a chunk leaves: a request's own and the pin's.
pub const MAX_TAKES: usize = 2;

/// The marks a chunk's takes leave, one per take, in the takes' order.
pub type TakeMarks = [WindowMark; MAX_TAKES];

/// Bytes a sequence's state holds, or a pass would add to it, by kind: the
/// attention layers' KV and the recurrent layers' state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateBytes {
    pub kv: usize,
    pub recurrent: usize,
}

impl StateBytes {
    pub fn total(&self) -> usize {
        self.kv + self.recurrent
    }
}

impl std::ops::Add for StateBytes {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            kv: self.kv + o.kv,
            recurrent: self.recurrent + o.recurrent,
        }
    }
}

/// A token mixer the topology can own the state of: how to make a fresh
/// state for one layer, clear it, walk the mixer's step, and roll the state
/// back over a speculative window (`sequence-state.md` §10.2a: a KV cache
/// by a length, a recurrent state by a snapshot and a replay).
pub trait MixerBlock: Mixer + Send + Sync {
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State>;
    fn reset_state(state: &mut Self::State);
    /// Bytes `state` holds on the device now.
    fn held_bytes(&self, state: &Self::State) -> StateBytes;
    /// Bytes a pass of `n` positions would allocate for `state` — a KV slab
    /// widened, a recurrent state's first write; zero when nothing grows.
    /// Sized, not performed.
    fn grow_bytes(&self, state: &Self::State, n: usize) -> StateBytes;
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
    /// Move the counters back over the last `n` forwards, which ran under
    /// a fired EOS gate (`CacheRewind::rewind_unwritten`): what they would
    /// have written is not the state's. A KV cache moves its length back; a
    /// recurrent state's gated steps carried it through and it has nothing
    /// to move. Refused by a mixer whose gated forwards are not known to
    /// leave its state unwritten.
    fn rewind_unwritten(_state: &mut Self::State, n: usize) -> Result<()> {
        lloom_tensor::bail!("this mixer cannot take back {n} gated forwards")
    }
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
    /// [`Mixer::execute`] over `x`, leaving for each take a turn checkpoint
    /// of the state after the chunk's first `rows` positions in its frame
    /// (`[checkpoint_elems]` f32; `None` for a mixer that keeps none) and
    /// returning, per take, the mark a restore needs. `rows == seq` is the
    /// state the pass leaves, `rows == 0` the one before it; in between, a
    /// recurrent state runs its recurrence a second time over the prefix,
    /// once per such take (the pass is unchanged).
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        state: &mut Self::State,
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)>;
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
    /// [`MixerBlock::held_bytes`] over the layer's state.
    fn held_bytes(&self, state: &Self::State) -> StateBytes;
    /// [`MixerBlock::grow_bytes`] over the layer's state.
    fn grow_bytes(&self, state: &Self::State, n: usize) -> StateBytes;
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
    fn visit_layer(&self, w: &mut Walker, x: NodeId, l: usize, cap: &StateCapacity) -> NodeId;
    fn open_window(state: &mut Self::State) -> Result<WindowMark>;
    fn rewind_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    fn resolve_window(state: &mut Self::State, mark: WindowMark) -> Result<()>;
    fn truncate_state(state: &mut Self::State, tokens: usize) -> Result<()>;
    /// [`MixerBlock::rewind_unwritten`] over the layer's state.
    fn rewind_unwritten(_state: &mut Self::State, n: usize) -> Result<()> {
        lloom_tensor::bail!("this layer cannot take back {n} gated forwards")
    }
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
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)>;
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
    /// The feed-forward holds no per-sequence state on any block here.
    fn held_bytes(&self, state: &Self::State) -> StateBytes {
        self.mixer.held_bytes(&state.mixer)
    }
    fn grow_bytes(&self, state: &Self::State, n: usize) -> StateBytes {
        self.mixer.grow_bytes(&state.mixer, n)
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
    fn rewind_unwritten(state: &mut Self::State, n: usize) -> Result<()> {
        M::rewind_unwritten(&mut state.mixer, n)
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
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)> {
        let h = self.norm_in.execute(x)?;
        let (m, marks) = self
            .mixer
            .execute_checkpointing(&h, &mut state.mixer, takes)?;
        let x = (x + m)?;
        let h = self.norm_mid.execute(&x)?;
        let f = self.ffn.execute(&h, &mut state.ffn)?;
        Ok(((x + f)?, marks))
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
    fn held_bytes(&self, state: &Self::State) -> StateBytes {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.held_bytes(s),
            (Hybrid::B(m), HybridState::B(s)) => m.held_bytes(s),
            _ => StateBytes::default(),
        }
    }
    fn grow_bytes(&self, state: &Self::State, n: usize) -> StateBytes {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.grow_bytes(s, n),
            (Hybrid::B(m), HybridState::B(s)) => m.grow_bytes(s, n),
            _ => StateBytes::default(),
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
    fn rewind_unwritten(state: &mut Self::State, n: usize) -> Result<()> {
        match state {
            HybridState::A(s) => A::rewind_unwritten(s, n),
            HybridState::B(s) => B::rewind_unwritten(s, n),
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
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)> {
        match (self, state) {
            (Hybrid::A(m), HybridState::A(s)) => m.execute_checkpointing(x, s, takes),
            (Hybrid::B(m), HybridState::B(s)) => m.execute_checkpointing(x, s, takes),
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
    /// The open window is the STANDING one: the last decode step's, left
    /// open after it so the step can be undone (`CacheRewind`, horizon 1 —
    /// an overrun past an unlearned EOS, a one-token divergence). Settled
    /// by the next forward or by any `open_window`; invisible to
    /// [`Self::window_open`], which answers for a verify's window only.
    standing: bool,
    /// Whether a decode step leaves a standing window: the state was built
    /// with `DecodeRecovery::Standing`.
    keeps_standing: bool,
    /// The final norm's output per position (`h_nextn`), retained for an
    /// MTP drafter while [`Self::set_want_nextn`] is on.
    nextn: RowLog,
    want_nextn: bool,
    nextn_drops: u64,
    /// The state's rows on the device's ledger — the KV and the recurrent
    /// state — kept current at the passes that allocate.
    #[cfg(feature = "metal")]
    ledger: Option<StateTenants>,
    /// The turn checkpoints (`sequence-reuse-design.md` §2.3b): the ring
    /// of every layer's [`Layer::checkpoint_elems`] f32s, the KV layers'
    /// lengths in the marks (`nn::turn_checkpoint`).
    ckpt: CheckpointRing,
    /// The marks' storage between windows: a window takes it and gives it
    /// back, so a decode step's window allocates nothing.
    marks_spare: Vec<WindowMark>,
    /// Why the state cannot be used until a reset: a window that failed to
    /// put every layer back left them inconsistent.
    unusable: Option<String>,
}

/// Owned layer snapshots and the logical end of a privately owned sequence.
pub struct DecoderRetained<S> {
    layers: Vec<S>,
    pos: usize,
}

impl<L: Layer> crate::nn::branch::RetainState for DecoderState<L>
where
    L::State: crate::nn::branch::RetainState,
{
    type Retained = DecoderRetained<<L::State as crate::nn::branch::RetainState>::Retained>;

    fn retain_state(&mut self) -> Result<Self::Retained> {
        self.settle_standing()?;
        if self.window.is_some() {
            lloom_tensor::bail!("decoder retention requires a settled window")
        }
        let mut layers = crate::nn::branch::retained_vec(self.layers.len())?;
        for layer in &mut self.layers {
            layers.push(layer.retain_state()?);
        }
        Ok(DecoderRetained {
            layers,
            pos: self.pos,
        })
    }

    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        self.settle_standing()?;
        if self.window.is_some() || saved.pos > self.pos || saved.layers.len() != self.layers.len()
        {
            lloom_tensor::bail!("decoder retention: invalid window, position or layer count")
        }
        for (layer, retained) in self.layers.iter_mut().zip(&saved.layers) {
            layer.restore_retained(retained)?;
        }
        self.pos = saved.pos;
        self.nextn.clear();
        self.ckpt.forget_past(saved.pos);
        Ok(())
    }
}

impl<MS: crate::nn::branch::RetainState> crate::nn::branch::RetainState for PreNormState<MS, ()> {
    type Retained = MS::Retained;
    fn retain_state(&mut self) -> Result<Self::Retained> {
        self.mixer.retain_state()
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        self.mixer.restore_retained(saved)
    }
}

impl<SA: crate::nn::branch::RetainState, SB: crate::nn::branch::RetainState>
    crate::nn::branch::RetainState for HybridState<SA, SB>
{
    type Retained = HybridState<SA::Retained, SB::Retained>;
    fn retain_state(&mut self) -> Result<Self::Retained> {
        Ok(match self {
            Self::A(a) => HybridState::A(a.retain_state()?),
            Self::B(b) => HybridState::B(b.retain_state()?),
        })
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        match (self, saved) {
            (Self::A(a), HybridState::A(saved)) => a.restore_retained(saved),
            (Self::B(b), HybridState::B(saved)) => b.restore_retained(saved),
            _ => lloom_tensor::bail!("retained hybrid state belongs to the other mixer"),
        }
    }
}

impl crate::nn::branch::RetainState for crate::models::qwen4exp::GdnState {
    type Retained = (
        crate::nn::recurrent::RecurrentRetained,
        crate::nn::recurrent::RecurrentRetained,
    );
    fn retain_state(&mut self) -> Result<Self::Retained> {
        Ok((self.s.retain_state()?, self.conv.retain_state()?))
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        self.s.restore_retained(&saved.0)?;
        self.conv.restore_retained(&saved.1)
    }
}

impl crate::nn::branch::RetainState for crate::models::qwen4exp::KvCache {
    type Retained = WindowMark;
    fn retain_state(&mut self) -> Result<Self::Retained> {
        let (tokens, blocks) = self.lens();
        Ok(WindowMark { tokens, blocks })
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        let (tokens, blocks) = self.lens();
        if saved.tokens > tokens || saved.blocks > blocks {
            lloom_tensor::bail!("retained KV prefix was truncated")
        }
        self.truncate(saved.tokens, saved.blocks);
        Ok(())
    }
}

/// A decoder state's two ledger rows.
#[cfg(feature = "metal")]
#[derive(Debug)]
struct StateTenants {
    kv: std::sync::Arc<lloom_tensor::kernels::metal::ledger::Gauge>,
    recurrent: std::sync::Arc<lloom_tensor::kernels::metal::ledger::Gauge>,
}

#[cfg(feature = "metal")]
impl StateTenants {
    /// Register `family`'s two rows on `m`'s ledger: capabilities (the
    /// caller chose the context), whose floor is what the state holds.
    fn register(m: &lloom_tensor::MetalDevice, family: &str) -> Self {
        use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
        let row = |what: &str| {
            let g = Gauge::new(
                format!("{family}-{what}"),
                Claim {
                    floor: 0,
                    want: 0,
                    unit: 1,
                    class: Class::Capability,
                },
                0,
            );
            let t: std::sync::Arc<dyn Tenant> = g.clone();
            m.ledger().register(&t);
            g
        };
        Self {
            kv: row("kv"),
            recurrent: row("state"),
        }
    }

    fn note(&self, held: StateBytes) {
        use lloom_tensor::kernels::metal::ledger::Tenant;
        for (g, b) in [(&self.kv, held.kv), (&self.recurrent, held.recurrent)] {
            let mut c = g.claim();
            c.floor = b;
            c.want = b;
            g.set_claim(c);
            g.set_held(b);
        }
    }
}

#[derive(Debug)]
struct DecoderWindow {
    pos: usize,
    marks: Vec<WindowMark>,
    retained: usize,
}

impl<L: Layer> DecoderState<L> {
    pub fn new(layers: Vec<L::State>) -> Self {
        Self {
            layers,
            pos: 0,
            window: None,
            standing: false,
            keeps_standing: false,
            nextn: RowLog::default(),
            want_nextn: false,
            nextn_drops: 0,
            #[cfg(feature = "metal")]
            ledger: None,
            ckpt: CheckpointRing::none(),
            marks_spare: Vec::new(),
            unusable: None,
        }
    }

    /// Refused while the state is unusable (`Self::unusable`).
    fn usable(&self) -> Result<()> {
        match &self.unusable {
            None => Ok(()),
            Some(why) => lloom_tensor::bail!("decoder state: unusable until a reset ({why})"),
        }
    }

    /// Why the state refuses every forward and window until a reset, if it
    /// does.
    pub fn unusable(&self) -> Option<&str> {
        self.unusable.as_deref()
    }

    /// Keep `slots` turn checkpoints, each `elems[l]` f32s for layer `l`.
    pub fn with_checkpoints(mut self, elems: &[usize], slots: usize) -> Self {
        self.ckpt = CheckpointRing::new(elems, slots);
        self
    }

    /// One frame more in the checkpoint ring, for a pinned checkpoint
    /// (`StateCapacity::pin`).
    pub fn with_pin_frame(mut self, on: bool) -> Self {
        self.ckpt = std::mem::replace(&mut self.ckpt, CheckpointRing::none()).with_pin_frame(on);
        self
    }

    /// Leave every decode step's window open as the standing one (`on`).
    pub fn with_standing_window(mut self, on: bool) -> Self {
        self.keeps_standing = on;
        self
    }

    /// Whether a decode step leaves a standing window.
    pub fn keeps_standing_window(&self) -> bool {
        self.keeps_standing
    }

    /// The checkpoint ring's bytes (allocated at the first checkpoint).
    pub fn checkpoint_bytes(&self) -> usize {
        self.ckpt.bytes()
    }

    pub fn checkpoint_slots(&self) -> usize {
        self.ckpt.slots()
    }

    /// The positions the valid checkpoints stand at.
    pub fn checkpoint_positions(&self) -> Vec<usize> {
        self.ckpt.positions()
    }

    /// Have the passes that follow leave their checkpoint at `pos` (one a
    /// pass covers takes it there instead of at its end). Ignored without
    /// checkpoint slots.
    pub fn checkpoint_at(&mut self, pos: usize) {
        self.ckpt.hint(pos);
    }

    pub fn checkpoint_hint(&self) -> Option<usize> {
        self.ckpt.hinted()
    }

    /// Open a request whose prefill ends at `end`
    /// (`CheckpointRing::open_request`).
    pub fn open_checkpoint_request(&mut self, policy: crate::nn::CheckpointPolicy, end: usize) {
        self.ckpt.open_request(policy, end);
    }

    /// Close the open request (`CheckpointRing::close_request`).
    pub fn close_checkpoint_request(&mut self) {
        self.ckpt.close_request();
    }

    /// The policy of the open request, if one is open.
    pub fn checkpoint_request(&self) -> Option<crate::nn::CheckpointPolicy> {
        self.ckpt.request_policy()
    }

    /// Pin the checkpoint at `pos` (`CheckpointRing::pin_at`: the one
    /// standing there, else the open request's second target), or with
    /// `None` release the pin. Returns where the pinned checkpoint stands
    /// after the call.
    pub fn pin_checkpoint(&mut self, pos: Option<usize>) -> Option<usize> {
        match pos {
            Some(p) => {
                self.ckpt.pin_at(p);
            }
            None => self.ckpt.unpin(),
        }
        self.ckpt.pinned()
    }

    /// Where the pinned checkpoint stands.
    pub fn pinned_checkpoint(&self) -> Option<usize> {
        self.ckpt.pinned()
    }

    /// Whether the checkpoint ring holds a frame for a pin.
    pub fn has_pin_frame(&self) -> bool {
        self.ckpt.has_pin_frame()
    }

    /// Whether [`Self::restore`] can put the state back at exactly `pos`: a
    /// checkpoint stands there and no verify window is open (a standing
    /// one is settled by the restore).
    pub fn can_restore(&self, pos: usize) -> bool {
        !self.window_open() && pos <= self.pos && self.ckpt.index(pos).is_some()
    }

    /// The latest checkpoint at or before `pos` (`CacheRewind::restore_point`).
    pub fn restore_point(&self, pos: usize) -> Option<usize> {
        if self.window_open() {
            return None;
        }
        self.ckpt
            .latest_at_or_before(pos.min(self.pos))
            .map(|(_, p)| p)
    }

    /// Put every layer back where the checkpoint at `pos` was taken — a KV
    /// cache to its length, a recurrent state to the frame's copy — and the
    /// position with them; the standing window settled first, what the
    /// discarded positions retained gone.
    pub fn restore(&mut self, pos: usize) -> Result<()> {
        self.usable()?;
        if self.can_restore(pos) {
            self.settle_standing()?;
        }
        let Some(idx) = self.ckpt.index(pos).filter(|_| self.can_restore(pos)) else {
            lloom_tensor::bail!(
                "decoder state: no turn checkpoint stands at {pos} (position {}, checkpoints at {:?}{})",
                self.pos,
                self.ckpt.positions(),
                if self.window_open() {
                    ", a window is open"
                } else {
                    ""
                }
            )
        };
        let marks = self.ckpt.marks(idx).to_vec();
        if marks.len() != self.layers.len() {
            lloom_tensor::bail!(
                "decoder state: a checkpoint of {} layers for {} layers",
                marks.len(),
                self.layers.len()
            )
        }
        let frames = (0..self.layers.len())
            .map(|l| self.ckpt.frame(idx, l))
            .collect::<Result<Vec<_>>>()?;
        for ((s, frame), mark) in self.layers.iter_mut().zip(&frames).zip(&marks) {
            L::restore_checkpoint(s, frame.as_ref(), *mark)?;
        }
        self.pos = pos;
        self.nextn.clear();
        self.ckpt.forget_past(pos);
        Ok(())
    }

    /// Open a speculative window over every layer's state and the
    /// position, the standing window settled first. One at a time. A layer
    /// that fails to open closes the ones before it (nothing was written
    /// inside them), so a failed opening leaves no window anywhere.
    pub fn open_window(&mut self) -> Result<()> {
        self.usable()?;
        self.settle_standing()?;
        if self.window.is_some() {
            lloom_tensor::bail!("decoder state: a speculative window is already open")
        }
        let mut marks = std::mem::take(&mut self.marks_spare);
        marks.clear();
        for l in 0..self.layers.len() {
            match L::open_window(&mut self.layers[l]) {
                Ok(m) => marks.push(m),
                Err(e) => {
                    for (s, m) in self.layers.iter_mut().zip(&marks) {
                        if let Err(r) = L::resolve_window(s, *m) {
                            self.unusable = Some(format!(
                                "layer {l} failed to open a window ({e}) and an earlier one \
                                 failed to close its own ({r})"
                            ));
                            return Err(e);
                        }
                    }
                    marks.clear();
                    self.marks_spare = marks;
                    return Err(e);
                }
            }
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
    /// accepted prefix. A layer that fails to rewind leaves the layers
    /// inconsistent: the state is unusable until a reset.
    pub fn rewind_window(&mut self) -> Result<()> {
        self.usable()?;
        let Some(w) = self.window.take() else {
            lloom_tensor::bail!("decoder state: no speculative window is open")
        };
        self.standing = false;
        for (l, (s, m)) in self.layers.iter_mut().zip(&w.marks).enumerate() {
            if let Err(e) = L::rewind_window(s, *m) {
                self.unusable = Some(format!("layer {l} failed to rewind its window ({e})"));
                return Err(e);
            }
        }
        self.pos = w.pos;
        self.marks_spare = w.marks;
        self.ckpt.forget_past(self.pos);
        self.nextn.keep_first(w.retained)
    }

    /// Close the window keeping everything the passes wrote. A layer that
    /// fails to close leaves the state unusable until a reset.
    pub fn resolve_window(&mut self) -> Result<()> {
        self.usable()?;
        let Some(w) = self.window.take() else {
            lloom_tensor::bail!("decoder state: no speculative window is open")
        };
        self.standing = false;
        for (l, (s, m)) in self.layers.iter_mut().zip(&w.marks).enumerate() {
            if let Err(e) = L::resolve_window(s, *m) {
                self.unusable = Some(format!("layer {l} failed to close its window ({e})"));
                return Err(e);
            }
        }
        self.marks_spare = w.marks;
        Ok(())
    }

    /// Whether a verify's window is open (a standing one does not count).
    pub fn window_open(&self) -> bool {
        self.window.is_some() && !self.standing
    }

    /// Resolve the standing window, if any: the last decode step is final.
    pub fn settle_standing(&mut self) -> Result<()> {
        if self.standing {
            self.resolve_window()?;
        }
        Ok(())
    }

    /// The positions the standing window would undo: 0 without one. A
    /// standing window opened past the position would be one a take-back
    /// left unsettled (`rewind_unwritten`): checked in debug builds.
    pub fn standing_horizon(&self) -> usize {
        match (&self.window, self.standing) {
            (Some(w), true) => {
                debug_assert!(
                    self.pos >= w.pos,
                    "decoder state: a standing window opened at {} past the position {}",
                    w.pos,
                    self.pos
                );
                self.pos.saturating_sub(w.pos)
            }
            _ => 0,
        }
    }

    /// Whether every layer runs a window of `steps` positions as one pass
    /// (`Layer::ring_holds`): the `Ring` profile with a
    /// `StateCapacity::verify` of at least `steps`.
    pub fn ring_holds(&self, steps: usize) -> bool {
        self.layers.iter().all(|s| L::ring_holds(s, steps))
    }

    /// Whether any layer's state lives on phase rings (the `Ring` profile
    /// with a verify window or a decode recovery,
    /// `StateCapacity::recurrent_phases`).
    pub fn pooled(&self) -> bool {
        self.layers.iter().any(L::pooled)
    }

    /// Accept the first `a` positions of the open window and close it:
    /// every layer to the mark after position `a` (`after[l][a − 1]`; the
    /// opening mark when `a` is 0, a rewind), the position, the retained
    /// probe cut to `a` rows of the window. The window's positions past `a`
    /// cost nothing — no replay.
    pub fn accept_window(&mut self, a: usize, after: &[Vec<WindowMark>]) -> Result<()> {
        self.usable()?;
        if self.standing {
            lloom_tensor::bail!("decoder state: accepting positions of a standing window")
        }
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
            if let Err(e) = L::accept_window(s, a, at) {
                self.unusable = Some(format!("layer {l} failed to accept its window ({e})"));
                return Err(e);
            }
        }
        self.pos = w.pos + a;
        self.marks_spare = w.marks;
        self.ckpt.forget_past(self.pos);
        self.nextn.keep_first(w.retained + a)
    }

    /// Take back the last `n` forwards, which ran under a fired EOS gate
    /// (`CacheRewind::rewind_unwritten`): the counters only. The position,
    /// every layer's (a KV cache its length; a recurrent state, carried,
    /// nothing), the standing window (settled: its base and its gated step
    /// hold the same values) and the checkpoints past the new end. A layer
    /// that fails to move back leaves the state unusable.
    pub fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.usable()?;
        if self.window_open() {
            lloom_tensor::bail!(
                "decoder state: taking back {n} gated forwards under a verify's window"
            )
        }
        if n > self.pos {
            lloom_tensor::bail!(
                "decoder state: taking back {n} gated forwards at position {}",
                self.pos
            )
        }
        self.settle_standing()?;
        for (l, s) in self.layers.iter_mut().enumerate() {
            if let Err(e) = L::rewind_unwritten(s, n) {
                self.unusable = Some(format!(
                    "layer {l} failed to take back {n} gated forwards ({e})"
                ));
                return Err(e);
            }
        }
        self.pos -= n;
        self.ckpt.forget_past(self.pos);
        self.nextn.clear();
        Ok(())
    }

    /// Cut every layer back to its first `tokens` positions outside a
    /// window (an attention-only lane, `MixerBlock::truncate_state`).
    pub fn truncate(&mut self, tokens: usize) -> Result<()> {
        self.usable()?;
        if self.window_open() {
            lloom_tensor::bail!("decoder state: truncating under an open window")
        }
        if tokens > self.pos {
            lloom_tensor::bail!(
                "decoder state: truncating {} positions to {tokens}",
                self.pos
            )
        }
        self.settle_standing()?;
        for s in &mut self.layers {
            L::truncate_state(s, tokens)?;
        }
        self.pos = tokens;
        self.ckpt.forget_past(tokens);
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
        if let Some(w) = self.window.take() {
            self.marks_spare = w.marks;
        }
        self.standing = false;
        self.unusable = None;
        self.nextn.clear();
        // The ring stays (the next sequence writes over it); its contents
        // describe nothing now.
        self.ckpt.forget_all();
    }
}

/// The rewindable arm is the standing window (a state built with
/// `DecodeRecovery::Standing`): every decode step leaves its window open
/// until the next forward, so the driver can undo exactly that step — the
/// ring arm's lookahead overrun, a one-token divergence of a reused prefix
/// (`Session::prepare`). A prefill leaves no horizon, nor does a step run
/// inside a verify's window. Past it, a turn checkpoint
/// ([`DecoderState::restore`]) puts the state back at the position a
/// multi-token pass left one at.
impl<L: Layer> CacheRewind for DecoderState<L> {
    fn check_acceptance_state(&self) -> Result<()> {
        self.usable()
    }
    fn rewind_horizon(&self) -> usize {
        self.standing_horizon()
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let horizon = self.standing_horizon();
        if n != horizon {
            lloom_tensor::bail!(
                "decoder state: a rewind of {n} positions against a standing horizon of \
                 {horizon}: only the whole last step can be undone"
            )
        }
        self.rewind_window()
    }
    fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        DecoderState::rewind_unwritten(self, n)
    }
    fn can_restore(&self, pos: usize) -> bool {
        DecoderState::can_restore(self, pos)
    }
    fn restore_point(&self, pos: usize) -> Option<usize> {
        DecoderState::restore_point(self, pos)
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
    /// What the load reserved on the machine (`lloom_tensor::machine`),
    /// released when the model drops.
    reservation: Option<std::sync::Arc<lloom_tensor::machine::Holding>>,
    /// What else the load registered for the model's life (a ledger row of
    /// a block the model owns), dropped with it.
    keep: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// The bytes a multi-position pass allocates per position, from the
    /// family's geometry (0: the family says nothing, and a pass's working
    /// set is not admitted).
    prefill_bytes_per_position: usize,
    /// Whether the family's dependency inventory admits the EOS gate's
    /// cores arm (`CausalLM::eos_gate_cores`); set by its loader.
    gate_cores: bool,
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
            reservation: None,
            keep: Vec::new(),
            prefill_bytes_per_position: 0,
            gate_cores: false,
        })
    }

    /// The family's loader says its forwards admit the EOS gate's cores arm
    /// (`CausalLM::eos_gate_cores`, `sequence-reuse-design.md` §2.3e).
    pub fn set_eos_gate_cores(&mut self, admitted: bool) {
        self.gate_cores = admitted;
    }

    /// The bytes a multi-position pass allocates per position: what
    /// `prepare_pass` admits on the machine before a prefill pass runs.
    pub fn set_prefill_bytes_per_position(&mut self, bytes: usize) {
        self.prefill_bytes_per_position = bytes;
    }

    /// Keep the load's machine reservation for the model's life.
    pub fn hold_reservation(
        &mut self,
        reservation: Option<std::sync::Arc<lloom_tensor::machine::Holding>>,
    ) {
        self.reservation = reservation;
    }

    pub fn reservation(&self) -> Option<&std::sync::Arc<lloom_tensor::machine::Holding>> {
        self.reservation.as_ref()
    }

    /// Keep `held` for the model's life.
    pub fn keep_alive(&mut self, held: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.keep.push(held);
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

    /// Bytes `state` holds on the device, over every layer.
    pub fn held_bytes(&self, state: &DecoderState<L>) -> StateBytes {
        self.layers
            .iter()
            .zip(&state.layers)
            .fold(StateBytes::default(), |a, (l, s)| a + l.held_bytes(s))
    }

    /// Bytes a pass of `n` positions would add to `state`, over every layer.
    pub fn grow_bytes(&self, state: &DecoderState<L>, n: usize) -> StateBytes {
        self.layers
            .iter()
            .zip(&state.layers)
            .fold(StateBytes::default(), |a, (l, s)| a + l.grow_bytes(s, n))
    }

    /// The bytes a state's turn-checkpoint ring takes at this capacity
    /// (`StateCapacity::checkpoint_frames` frames of every layer's
    /// [`Layer::checkpoint_elems`] f32s).
    pub fn checkpoint_bytes(&self) -> usize {
        self.cap.checkpoint_frames() * self.checkpoint_frame_bytes()
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
    /// last position's logits `[1, vocab]`. The last decode step is final
    /// once this forward begins; a decode step outside a verify's window
    /// leaves a standing window of its own on a state that keeps one.
    pub fn forward_from(&self, embeds: &Tensor, state: &mut DecoderState<L>) -> Result<Tensor> {
        let (b, seq, _) = embeds.dims3()?;
        if b != 1 || seq == 0 {
            lloom_tensor::bail!(
                "{}: a forward takes one sequence of >= 1 tokens, got {:?}",
                self.family,
                embeds.shape()
            )
        }
        state.usable()?;
        state.settle_standing()?;
        let windowed = state.window_open();
        if seq > 1 || windowed || !state.keeps_standing {
            return self.run_from(embeds, state, windowed);
        }
        state.open_window()?;
        match self.run_from(embeds, state, windowed) {
            Ok(logits) => {
                state.standing = true;
                Ok(logits)
            }
            // A failed step is undone: its writes never reached the base.
            Err(e) => match state.rewind_window() {
                Ok(()) => Err(e),
                Err(r) => lloom_tensor::bail!(
                    "{}: a decode step failed ({e}) and its window could not be put back ({r})",
                    self.family
                ),
            },
        }
    }

    /// [`Self::forward_from`]'s pass, `windowed` when it runs inside a
    /// verify's window (no turn checkpoint is taken there).
    fn run_from(
        &self,
        embeds: &Tensor,
        state: &mut DecoderState<L>,
        windowed: bool,
    ) -> Result<Tensor> {
        let seq = embeds.dim(1)?;
        let pass = state.pos;
        state.ckpt.drop_stale_hint(pass);
        state.ckpt.settle_pass(pass, seq, windowed);
        let mut last = None;
        let mut start = 0;
        while start < seq {
            let len = (seq - start).min(self.chunk);
            let at = state.pos;
            let takes = state
                .ckpt
                .takes_for(at, len, start + len == seq, pass, seq, windowed);
            let mut h = embeds.narrow(1, start, len)?;
            if takes.is_empty() {
                for (l, layer) in self.layers.iter().enumerate() {
                    h = layer.execute(&h, &mut state.layers[l])?;
                }
            } else {
                // Each take's slot: the pin's frame for the second target,
                // else a turn slot.
                let mut slots = [(0, 0); MAX_TAKES];
                let n = takes.len();
                for (k, t) in takes.iter().enumerate() {
                    let slot = if t.pin {
                        state.ckpt.open_pin(at + t.rows, embeds.device())?
                    } else {
                        state.ckpt.open(at + t.rows, embeds.device())?
                    };
                    slots[k] = (t.rows, slot);
                }
                let mut marks: [Vec<WindowMark>; MAX_TAKES] = std::array::from_fn(|k| {
                    Vec::with_capacity(if k < n { self.layers.len() } else { 0 })
                });
                for (l, layer) in self.layers.iter().enumerate() {
                    let frames = [
                        state.ckpt.frame(slots[0].1, l)?,
                        if n > 1 {
                            state.ckpt.frame(slots[1].1, l)?
                        } else {
                            None
                        },
                    ];
                    let chunk_takes = [
                        (slots[0].0, frames[0].as_ref()),
                        (slots[1].0, frames[1].as_ref()),
                    ];
                    let (out, took) =
                        layer.execute_checkpointing(&h, &mut state.layers[l], &chunk_takes[..n])?;
                    h = out;
                    for (m, mark) in marks.iter_mut().zip(took).take(n) {
                        m.push(mark);
                    }
                }
                for ((_, slot), m) in slots.into_iter().zip(marks).take(n) {
                    state.ckpt.close(slot, m);
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
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut state = DecoderState::new(layers)
            .with_checkpoints(&elems, self.cap.checkpoints)
            .with_pin_frame(self.cap.pin)
            .with_standing_window(self.cap.recovery == DecodeRecovery::Standing);
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = dev {
            let t = StateTenants::register(m, self.family);
            t.note(self.held_bytes(&state));
            state.ledger = Some(t);
        }
        Ok(state)
    }

    /// Admit what the pass allocates before it runs, refused with the
    /// arithmetic before anything moves: the state's growth (the KV slabs
    /// widened, the recurrent states' first writes) against the ledger's
    /// quota and the machine; and a multi-position pass's working set
    /// against the machine (`MetalDevice::admit_bytes`, `memory.md` §9.10)
    /// — a pass inside the load's reservation asks nothing. A decode step
    /// that grows nothing reads nothing.
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        // A pass that leaves a turn checkpoint runs another allocation
        // sequence than one that does not: its own plan.
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &self.device {
            let m = m.execution_device()?;
            if m.pass_plans_enabled() {
                m.set_next_pass_variant(cache.ckpt.pass_variant(
                    cache.pos,
                    seq_len,
                    cache.window.is_some(),
                ));
            }
        }
        let grow = self.grow_bytes(cache, seq_len);
        let pass = if seq_len > 1 {
            seq_len * self.prefill_bytes_per_position
        } else {
            0
        };
        if grow.total() == 0 && pass == 0 {
            return Ok(());
        }
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &self.device {
            use lloom_tensor::memory_admission::{AllocationError, RefusalDomain};
            let gb = |b: usize| b as f64 / 1e9;
            if grow.total() > 0 {
                let quota = m.ledger().quota();
                let budget = if quota == 0 { usize::MAX } else { quota };
                let a = m.admit_bytes(grow.total(), budget);
                if !a.fits {
                    let refusal = a.machine.unwrap_or(AllocationError::Refused {
                        requested: grow.total(),
                        available: budget.saturating_sub(a.committed),
                        domain: RefusalDomain::Model,
                    });
                    m.report_refusal(
                        &format_args!(
                            "{}: the state's growth at position {} needs {:.3} GB (KV {:.3}, \
                             recurrent {:.3}); the ledger holds {:.2} GB of the {:.2} GB quota \
                             after giving back {:.3} GB of free lists; the pass stops here, before \
                             anything is written (DESIGN.md 9.5): a shorter context, \
                             --memory-fraction, or bound the run",
                            self.family,
                            cache.pos,
                            gb(grow.total()),
                            gb(grow.kv),
                            gb(grow.recurrent),
                            gb(a.committed),
                            gb(budget),
                            gb(a.reclaimed),
                        ),
                        &refusal,
                    );
                    return Err(refusal.into());
                }
                if let Some(t) = &cache.ledger {
                    t.note(self.held_bytes(cache) + grow);
                }
            }
            if pass > 0 {
                if let Some(refusal) = m.admit_bytes(pass, usize::MAX).machine {
                    m.report_refusal(
                        &format_args!(
                            "{}: the {seq_len}-position pass at position {} ({:.2} GB of working \
                             set) is refused before it runs",
                            self.family,
                            cache.pos,
                            gb(pass)
                        ),
                        &refusal,
                    );
                    return Err(refusal.into());
                }
            }
        }
        Ok(())
    }

    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        cache.checkpoint_at(pos);
        Ok(())
    }

    fn eos_gate_cores(&self) -> bool {
        self.gate_cores
    }

    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: crate::nn::CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        cache.open_checkpoint_request(policy, end);
        Ok(())
    }

    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        cache.close_checkpoint_request();
        Ok(())
    }

    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        Ok(cache.pin_checkpoint(pos))
    }

    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        cache.has_pin_frame()
    }

    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        cache.pinned_checkpoint()
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
/// `StateCapacity::verify`) the window is ONE pass — the stems and
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
        self.window_enqueue(draft, anchor, cache, None)
    }

    /// The window's rows drawn by `sampler`, each keyed to the position of
    /// the token it draws, in place of their argmax (the sampled verify);
    /// the pass and the accept are the greedy verify's.
    fn verify_and_probe_sampled(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
        sampler: &mut crate::generation::LogitsProcessor,
    ) -> Result<(Vec<Vec<u32>>, Tensor)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "{} verify_and_probe_sampled: one sequence; got {} drafts and {} anchors",
                self.family,
                drafts.len(),
                anchors.len()
            )
        }
        let pending = self.window_enqueue(&drafts[0], &anchors[0], cache, Some(sampler))?;
        let (accepted, probe) = self.verify_complete(pending, cache, eos_ids)?;
        Ok((vec![accepted], probe))
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

impl<P, C, E, L, H> Decoder<P, C, E, L, H>
where
    P: Profile,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    E: Embed,
    L: Layer,
    H: Head,
{
    /// The window `[anchor, proposed…]` run inside an open window on every
    /// layer's state, each row's token read back once: the argmax, or
    /// `draw`'s keyed draw over the rows' logits (the sampled verify).
    fn window_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut DecoderState<L>,
        draw: Option<&mut crate::generation::LogitsProcessor>,
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
        // Row 0 draws the token after the anchor.
        let first_pos = anchor.pos as u64 + 1;
        // A state on rings rewinds by phase: a window longer than its rings
        // cannot be stepped through and put back (the rings are sized to
        // the drafter's `k + 1`, `StateCapacity::verify`).
        if cache.pooled() && !cache.ring_holds(t) {
            cache.set_want_nextn(want_was);
            lloom_tensor::bail!(
                "{} verify: a window of {t} positions on rings built for a shorter one \
                 (`StateCapacity::verify` must be >= {t})",
                self.family
            )
        }
        cache.open_window()?;
        if dev.is_metal() && cache.ring_holds(t) {
            let _window = lloom_tensor::tuning::VerifyWindow::open();
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
            let argmax = match draw {
                None => {
                    let mut argmaxes = Vec::with_capacity(t);
                    for p in 0..t {
                        let logits = self.head.project(&n.narrow(0, p, 1)?)?;
                        argmaxes.push(logits.flatten_all()?.argmax(D::Minus1)?.reshape(1)?);
                    }
                    Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?
                }
                Some(s) => {
                    let mut rows = Vec::with_capacity(t);
                    for p in 0..t {
                        rows.push(self.head.project(&n.narrow(0, p, 1)?)?.flatten_all()?);
                    }
                    let inputs = Tensor::from_vec(tokens.clone(), t, &dev)?;
                    let drawn = s
                        .window_draw_device(&Tensor::stack(&rows, 0)?, first_pos, Some(&inputs))?
                        .to_vec1::<u32>()?;
                    s.window_fed(&drawn)
                }
            };
            return Ok(DecoderPending {
                argmax,
                tokens,
                want_was,
                after: Some(after),
            });
        }
        let argmax = match draw {
            None => {
                let mut argmaxes = Vec::with_capacity(t);
                for &tok in &tokens {
                    argmaxes.push(self.step_argmax(tok, cache, &dev)?);
                }
                Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?
            }
            Some(s) => {
                let mut rows = Vec::with_capacity(t);
                for &tok in &tokens {
                    let input = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
                    rows.push(CausalLM::forward(self, &input, cache.pos, cache)?.flatten_all()?);
                }
                let inputs = Tensor::from_vec(tokens.clone(), t, &dev)?;
                let drawn = s
                    .window_draw_device(&Tensor::stack(&rows, 0)?, first_pos, Some(&inputs))?
                    .to_vec1::<u32>()?;
                s.window_fed(&drawn)
            }
        };
        Ok(DecoderPending {
            argmax,
            tokens,
            want_was,
            after: None,
        })
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
    /// At the `Ring` profile with a verify window or a decode recovery, the
    /// two channels on their own phase rings
    /// (`StateCapacity::recurrent_phases`); else an unpooled pair that a
    /// window snapshots.
    fn new_state(&self, dev: &Device, cap: &StateCapacity) -> Result<Self::State> {
        let rewindable = <P::Rewind as crate::profile::Rewind>::REWINDABLE;
        if let Some(phases) = cap.recurrent_phases(rewindable) {
            return self.pooled_state(dev, phases);
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
    fn held_bytes(&self, state: &Self::State) -> StateBytes {
        StateBytes {
            kv: 0,
            recurrent: state.s.bytes_held() + state.conv.bytes_held(),
        }
    }
    /// The first write allocates the whole state (the ring's phases pooled,
    /// the slab and its spare else); after it nothing grows.
    fn grow_bytes(&self, state: &Self::State, _n: usize) -> StateBytes {
        let recurrent = match state.s.pool_slot(0) {
            Some((pool, _, _, false)) => self.state_bytes(pool.phases()),
            Some(_) => 0,
            None if state.s.get(0).is_none() => self.state_bytes(2),
            None => 0,
        };
        StateBytes { kv: 0, recurrent }
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
    /// The gated decode step carries `S` and the window into the buffers it
    /// swaps in (`gdn.metal`), so the state is the one before the gated
    /// forwards: nothing to move.
    fn rewind_unwritten(_state: &mut Self::State, _n: usize) -> Result<()> {
        Ok(())
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
    /// `S` and the conv window into each take's frame: a decode step's
    /// (`rows` 0 or 1: before or after it), a prefill's at `rows`
    /// (`GatedDeltaNet::prefill_checkpointing_at`).
    fn execute_checkpointing(
        &self,
        x: &Tensor,
        st: &mut Self::State,
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)> {
        if takes.len() > MAX_TAKES {
            lloom_tensor::bail!("deltanet checkpoint: {} takes in one chunk", takes.len())
        }
        let (ns, nc) = self.checkpoint_parts();
        let seq = x.dim(1)?;
        let views = takes
            .iter()
            .map(|&(rows, into)| {
                let Some(frame) = into else {
                    lloom_tensor::bail!("a recurrent state's checkpoint needs a ring frame")
                };
                if rows > seq {
                    lloom_tensor::bail!(
                        "deltanet checkpoint: {rows} rows of a {seq}-position chunk"
                    )
                }
                Ok((rows, frame.narrow(0, 0, ns)?, frame.narrow(0, ns, nc)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let at: Vec<(usize, crate::models::qwen4exp::GdnCheckpoint<'_>)> = views
            .iter()
            .map(|(rows, s, conv)| (*rows, crate::models::qwen4exp::GdnCheckpoint { s, conv }))
            .collect();
        let out = if seq == 1 {
            for (_, ck) in at.iter().filter(|(rows, _)| *rows == 0) {
                self.store_checkpoint(st.layer, &st.s, &st.conv, x.device(), ck)?;
            }
            let out = self.decode(x, st.layer, &mut st.s, &mut st.conv)?;
            for (_, ck) in at.iter().filter(|(rows, _)| *rows == 1) {
                self.store_checkpoint(st.layer, &st.s, &st.conv, x.device(), ck)?;
            }
            out
        } else {
            self.prefill_checkpointing_at(x, st.layer, &mut st.s, &mut st.conv, &at)?
        };
        Ok((out, [WindowMark::default(); MAX_TAKES]))
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
    fn held_bytes(&self, state: &Self::State) -> StateBytes {
        let (nkv, hd) = self.kv_geometry();
        StateBytes {
            kv: state.bytes_held(nkv, hd),
            recurrent: 0,
        }
    }
    fn grow_bytes(&self, state: &Self::State, n: usize) -> StateBytes {
        let (nkv, hd) = self.kv_geometry();
        StateBytes {
            kv: state.grow_bytes_for(n, nkv, hd),
            recurrent: 0,
        }
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
    /// The KV's length back over the gated forwards' appends; the rows past
    /// it are unread. Refused with the indexer's compressed blocks, which a
    /// length alone does not put back.
    fn rewind_unwritten(state: &mut Self::State, n: usize) -> Result<()> {
        if state.n_blocks() > 0 {
            lloom_tensor::bail!(
                "a KV cache with the indexer's blocks cannot take back {n} gated forwards"
            )
        }
        let len = state.len();
        if n > len {
            lloom_tensor::bail!("a KV cache of {len} positions cannot take back {n}")
        }
        state.truncate(len - n, 0);
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
        takes: &ChunkTakes<'_>,
    ) -> Result<(Tensor, TakeMarks)> {
        let seq = x.dim(1)?;
        if takes.len() > MAX_TAKES {
            lloom_tensor::bail!("qsa checkpoint: {} takes in one chunk", takes.len())
        }
        for &(rows, _) in takes {
            if rows > seq {
                lloom_tensor::bail!("qsa checkpoint: {rows} rows of a {seq}-position chunk")
            }
        }
        let (tokens, blocks) = kv.lens();
        let out = Mixer::execute(self, x, kv)?;
        let mut marks = [WindowMark::default(); MAX_TAKES];
        for (mark, &(rows, _)) in marks.iter_mut().zip(takes) {
            *mark = if rows == seq {
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
        }
        Ok((out, marks))
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

#[cfg(test)]
mod window_tests {
    //! `DecoderState`'s window over layers that fail on cue: a failed opening
    //! leaves no layer open, a failed rewind or resolve leaves the state
    //! refusing until a reset, and the marks' storage is reused.
    use super::*;

    struct Faulty;

    #[derive(Default)]
    struct FaultyState {
        open: bool,
        fail_open: bool,
        fail_rewind: bool,
        fail_resolve: bool,
    }

    impl Layer for Faulty {
        type State = FaultyState;
        fn new_state(&self, _: &Device, _: &StateCapacity) -> Result<FaultyState> {
            Ok(FaultyState::default())
        }
        fn reset_state(s: &mut FaultyState) {
            s.open = false;
        }
        fn held_bytes(&self, _: &FaultyState) -> StateBytes {
            StateBytes::default()
        }
        fn grow_bytes(&self, _: &FaultyState, _: usize) -> StateBytes {
            StateBytes::default()
        }
        fn execute(&self, x: &Tensor, _: &mut FaultyState) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn visit_layer(&self, _: &mut Walker, x: NodeId, _: usize, _: &StateCapacity) -> NodeId {
            x
        }
        fn open_window(s: &mut FaultyState) -> Result<WindowMark> {
            if s.fail_open {
                lloom_tensor::bail!("refused to open")
            }
            if s.open {
                lloom_tensor::bail!("already open")
            }
            s.open = true;
            Ok(WindowMark::default())
        }
        fn rewind_window(s: &mut FaultyState, _: WindowMark) -> Result<()> {
            if s.fail_rewind {
                lloom_tensor::bail!("refused to rewind")
            }
            if !s.open {
                lloom_tensor::bail!("not open")
            }
            s.open = false;
            Ok(())
        }
        fn resolve_window(s: &mut FaultyState, _: WindowMark) -> Result<()> {
            if s.fail_resolve {
                lloom_tensor::bail!("refused to resolve")
            }
            if !s.open {
                lloom_tensor::bail!("not open")
            }
            s.open = false;
            Ok(())
        }
        fn truncate_state(_: &mut FaultyState, _: usize) -> Result<()> {
            Ok(())
        }
        fn ring_holds(_: &FaultyState, _: usize) -> bool {
            false
        }
        fn pooled(_: &FaultyState) -> bool {
            false
        }
        fn execute_window(
            &self,
            x: &Tensor,
            _: &mut FaultyState,
            _: &mut Vec<WindowMark>,
        ) -> Result<Tensor> {
            Ok(x.clone())
        }
        fn accept_window(s: &mut FaultyState, _: usize, _: WindowMark) -> Result<()> {
            s.open = false;
            Ok(())
        }
        fn checkpoint_elems(&self, _: &StateCapacity) -> usize {
            0
        }
        fn execute_checkpointing(
            &self,
            x: &Tensor,
            _: &mut FaultyState,
            _: &ChunkTakes<'_>,
        ) -> Result<(Tensor, TakeMarks)> {
            Ok((x.clone(), [WindowMark::default(); MAX_TAKES]))
        }
        fn restore_checkpoint(
            _: &mut FaultyState,
            _: Option<&Tensor>,
            _: WindowMark,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn state(n: usize) -> DecoderState<Faulty> {
        DecoderState::new((0..n).map(|_| FaultyState::default()).collect())
    }

    // No forward executes in these acceptance controls. The actual decoder
    // cache supplies the window/horizon/poisoning behavior under test.
    impl Model for Faulty {
        type Config = ();
        fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
            Ok(Self)
        }
    }

    impl CausalLM for Faulty {
        type Cache = DecoderState<Self>;
        fn vocab_size(&self) -> usize {
            32
        }
        fn new_cache(&self, _: &(), _: &Device) -> Result<Self::Cache> {
            Ok(state(3))
        }
        fn embed(&self, _: &Tensor) -> Result<Tensor> {
            lloom_tensor::bail!("the acceptance fixture must not embed")
        }
        fn forward_embeds(&self, _: &Tensor, _: usize, _: &mut Self::Cache) -> Result<Tensor> {
            lloom_tensor::bail!("the acceptance fixture must not forward")
        }
    }

    fn acceptance_sequence(overrun: usize) -> crate::nn::Sequence<DecoderState<Faulty>> {
        let mut st = state(3);
        st.pos = 3 + overrun - 1;
        st.open_window().unwrap();
        st.pos += 1;
        st.standing = true;
        let mut seq = crate::nn::Sequence::with_tokens(st, vec![10, 20, 30]);
        seq.pos += overrun;
        seq.overrun = overrun;
        seq
    }

    #[test]
    fn accept_propagates_a_partial_rewind_failure_without_claiming_an_obligation() {
        let mut seq = acceptance_sequence(1);
        seq.cache.layers[1].fail_rewind = true;
        let mut session = crate::nn::Session::from_sequence(&Faulty, seq);
        let result = session.accept(2);
        let seq = session.sequence();
        let cache = seq.cache();
        assert!(cache
            .unusable()
            .expect("partial rewind poisoned the cache")
            .contains("layer 1"));
        assert!(!cache.layers[0].open, "the first layer really rewound");
        assert!(
            cache.layers[1].open && cache.layers[2].open,
            "later layers did not rewind"
        );
        eprintln!(
            "ACCEPT partial rewind: result={result:?} unusable={:?} pos={} overrun={} tokens={:?}",
            cache.unusable(),
            seq.pos(),
            seq.overrun(),
            seq.tokens()
        );
        assert!(
            result.is_err(),
            "an operational partial rewind must not become Ok(Obligation): {result:?}"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("refused to rewind"));
        assert_eq!(
            seq.tokens(),
            &[10, 20, 30],
            "an error does not accept a new boundary"
        );
        assert_eq!((seq.pos(), seq.overrun()), (4, 1));
        assert!(
            cache.usable().is_err(),
            "no synthetic recovery of the poisoned cache"
        );
    }

    #[test]
    fn accept_retry_after_a_partial_rewind_does_not_claim_recovery() {
        let mut seq = acceptance_sequence(1);
        seq.cache.layers[1].fail_rewind = true;
        let mut session = crate::nn::Session::from_sequence(&Faulty, seq);
        let first = session.accept(2);
        assert!(session.sequence().cache().unusable().is_some());
        assert_eq!(session.sequence().cache().rewind_horizon(), 0);
        let before = (session.tokens().to_vec(), session.pos(), session.overrun());
        let retry = session.accept(1);
        eprintln!("ACCEPT poisoned retry: first={first:?} retry={retry:?} before={before:?}");
        assert!(
            retry.is_err(),
            "an already-poisoned cache must not claim an accepted recovery obligation: {retry:?}"
        );
        assert_eq!(
            (session.tokens(), session.pos(), session.overrun()),
            (before.0.as_slice(), before.1, before.2)
        );
        assert!(
            session.sequence().cache().unusable().is_some(),
            "no synthetic recovery"
        );
    }

    #[test]
    fn accept_checks_known_poison_after_construction_move_and_explicit_reset() {
        let mut seq = acceptance_sequence(1);
        seq.cache.layers[1].fail_rewind = true;
        assert!(seq.cache.rewind(1).is_err());
        assert!(seq.cache.unusable().is_some());
        // Reconstructing a Sequence cannot clear the cache's actual sentinel.
        // This has NO overrun: even Whole must check known invalid state.
        let seq = crate::nn::Sequence::with_tokens(seq.cache, vec![10, 20, 30]);
        let mut session = crate::nn::Session::from_sequence(&Faulty, seq);
        for keep in [3, 2] {
            let error = session.accept(keep).unwrap_err();
            assert!(error.to_string().contains("unusable until a reset"));
            assert_eq!(session.tokens(), &[10, 20, 30]);
        }
        // The invalidity belongs to the cache, not a particular Session.
        let mut session = crate::nn::Session::from_sequence(&Faulty, session.into_sequence());
        assert!(session.accept(1).is_err());
        assert_eq!(session.tokens(), &[10, 20, 30]);
        session.reset();
        assert!(session.sequence().cache().unusable().is_none());
        assert_eq!((session.pos(), session.overrun()), (0, 0));
        assert_eq!(session.accept(0).unwrap(), crate::nn::Acceptance::Whole);
    }

    #[test]
    fn accept_keeps_healthy_exact_and_reconciled_obligation_paths() {
        let mut seq = acceptance_sequence(1);
        seq.tokens.push(40);
        seq.overrun = 0;
        let mut session = crate::nn::Session::from_sequence(&Faulty, seq);
        assert_eq!(
            session.accept(3).unwrap(),
            crate::nn::Acceptance::Exact { dropped: 1 }
        );
        assert_eq!((session.pos(), session.overrun()), (3, 0));
        assert_eq!(session.tokens(), &[10, 20, 30]);
        assert!(session
            .sequence()
            .cache()
            .layers
            .iter()
            .all(|layer| !layer.open));

        let mut session = crate::nn::Session::from_sequence(&Faulty, acceptance_sequence(1));
        assert_eq!(
            session.accept(2).unwrap(),
            crate::nn::Acceptance::Obligation {
                ahead: 1,
                restore_point: None
            }
        );
        assert_eq!((session.pos(), session.overrun()), (3, 1));
        assert_eq!(session.tokens(), &[10, 20]);
        assert!(session.sequence().cache().unusable().is_none());
        assert!(session
            .sequence()
            .cache()
            .layers
            .iter()
            .all(|layer| !layer.open));
    }

    #[test]
    fn accept_keeps_the_obligation_when_the_overrun_exceeds_the_horizon() {
        let seq = acceptance_sequence(2);
        assert_eq!(seq.cache.rewind_horizon(), 1);
        let mut session = crate::nn::Session::from_sequence(&Faulty, seq);
        let result = session.accept(2).unwrap();
        assert_eq!(
            result,
            crate::nn::Acceptance::Obligation {
                ahead: 3,
                restore_point: None
            }
        );
        let seq = session.sequence();
        assert_eq!(seq.tokens(), &[10, 20]);
        assert_eq!((seq.pos(), seq.overrun()), (5, 3));
        assert!(
            seq.cache().unusable().is_none(),
            "a horizon refusal did not poison the cache"
        );
        assert!(
            seq.cache().layers.iter().all(|layer| layer.open),
            "no layer rewound past its horizon"
        );
    }

    #[test]
    fn a_failed_opening_leaves_no_layer_open() {
        let mut st = state(4);
        st.layers[2].fail_open = true;
        assert!(st.open_window().is_err());
        assert!(
            st.layers.iter().all(|l| !l.open),
            "a layer before the one that failed stayed open"
        );
        assert!(!st.window_open() && st.unusable().is_none());
        st.layers[2].fail_open = false;
        st.open_window().expect("the next opening");
        assert!(st.layers.iter().all(|l| l.open));
        st.resolve_window().expect("and its close");
    }

    #[test]
    fn a_failed_rewind_leaves_the_state_unusable_until_a_reset() {
        let mut st = state(3);
        st.open_window().unwrap();
        st.layers[1].fail_rewind = true;
        assert!(st.rewind_window().is_err());
        let why = st.unusable().expect("marked unusable").to_string();
        assert!(why.contains("layer 1"), "{why}");
        assert!(st.open_window().is_err(), "no window on an unusable state");
        assert!(st.truncate(0).is_err(), "no truncate on an unusable state");
        st.layers[1].fail_rewind = false;
        st.reset();
        assert!(st.unusable().is_none());
        st.open_window().expect("usable after the reset");
    }

    #[test]
    fn a_failed_resolve_leaves_the_state_unusable_until_a_reset() {
        let mut st = state(3);
        st.open_window().unwrap();
        st.layers[2].fail_resolve = true;
        assert!(st.resolve_window().is_err());
        assert!(st.unusable().expect("marked unusable").contains("layer 2"));
        assert!(st.open_window().is_err());
        st.layers[2].fail_resolve = false;
        st.reset();
        st.open_window().expect("usable after the reset");
    }

    #[test]
    fn a_window_keeps_its_marks_storage() {
        let mut st = state(4);
        st.open_window().unwrap();
        st.resolve_window().unwrap();
        let (ptr, cap) = (st.marks_spare.as_ptr(), st.marks_spare.capacity());
        assert!(cap >= 4);
        // Another allocation of the same size alive the whole time, so a new
        // `Vec` per opening could not land on the freed block by chance.
        let _decoy: Vec<WindowMark> = Vec::with_capacity(cap);
        let after = vec![Vec::new(); 4];
        for _ in 0..8 {
            st.open_window().unwrap();
            st.rewind_window().unwrap();
            st.open_window().unwrap();
            st.resolve_window().unwrap();
            st.open_window().unwrap();
            st.accept_window(0, &after).unwrap();
            assert_eq!(
                (st.marks_spare.as_ptr(), st.marks_spare.capacity()),
                (ptr, cap),
                "an opening took new storage for its marks"
            );
        }
    }
}
