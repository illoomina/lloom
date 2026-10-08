//! The level-0 block contract (`porting-models.md` §21.9, RESTRUCTURING
//! §1.5): a block is a type generic over its variant, with `execute` (what it
//! dispatches) and `visit` (its IR node, [`Visit`]) and, where it carries
//! per-sequence state, a `type State`. Each trait exists because two
//! implementors do (RESTRUCTURING §3.2): both engine families' blocks
//! implement them, and a port composes against the trait rather than against
//! a family's struct by name. Static dispatch only: generics, never `dyn` on
//! a decode path (CONTRIBUTING §3.6).

use crate::optimizer::{NodeId, Shape, Visit, Walker};
use lloom_tensor::{Result, Tensor};

/// A normalization over the last axis, the weight the block's.
pub trait Norm: Visit<In = NodeId, Out = NodeId> {
    fn execute(&self, x: &Tensor) -> Result<Tensor>;
}

/// Which pairs a rotary embedding rotates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// `(x[i], x[i + d/2])`: the GPT-NeoX pairing.
    Neox,
    /// `(x[2i], x[2i + 1])`: the GPT-J / interleaved pairing.
    Interleaved,
    /// NeoX pairs over a `[b, t, h, d]` layout.
    Thd,
}

/// A rotary embedding: `x` rotated at the positions `cos` / `sin` were
/// built for. The pairing is the type's, so two kernels that produce the
/// same shapes and pass the short-context text gate (`porting-models.md`
/// §21.4a) are two types.
pub trait Rope: Visit<In = (NodeId, NodeId, NodeId), Out = NodeId> {
    const STYLE: RopeStyle;
    fn execute(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor>;
}

/// An elementwise activation.
pub trait Activation: Visit<In = NodeId, Out = NodeId> {
    fn execute(&self, x: &Tensor) -> Result<Tensor>;
}

/// A feed-forward block over the residual width: `[b, seq, hidden]` in and
/// out. `State` is what it keeps per sequence (`()` for a dense MLP and a
/// routed MoE; a streamed tier's is its owner's).
pub trait Ffn: Visit {
    type State;
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
}

/// A token mixer: `[b, seq, hidden]` in and out, appending `seq` positions
/// to its per-sequence `State` (a KV cache, a recurrent state).
pub trait Mixer: Visit {
    type State;
    fn execute(&self, x: &Tensor, state: &mut Self::State) -> Result<Tensor>;
}

/// The NeoX-paired rotation (`nn::rotary_emb::rope`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Neox;
/// The interleaved rotation (`nn::rotary_emb::rope_i`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Interleaved;
/// The NeoX rotation over `[b, t, h, d]` (`nn::rotary_emb::rope_thd`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Thd;

macro_rules! rope_style {
    ($ty:ident, $style:expr, $op:path, $kernel:literal) => {
        impl Visit for $ty {
            type In = (NodeId, NodeId, NodeId);
            type Out = NodeId;
            fn visit(&self, w: &mut Walker, (x, cos, sin): Self::In) -> Self::Out {
                let n = w.node(x);
                let (shape, dt) = (n.shape.clone(), n.dtype);
                w.opaque($kernel, &[x, cos, sin], shape, dt)
            }
        }
        impl Rope for $ty {
            const STYLE: RopeStyle = $style;
            fn execute(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
                $op(x, cos, sin)
            }
        }
    };
}
rope_style!(Neox, RopeStyle::Neox, crate::nn::rotary_emb::rope, "rope");
rope_style!(
    Interleaved,
    RopeStyle::Interleaved,
    crate::nn::rotary_emb::rope_i,
    "rope_i"
);
rope_style!(
    Thd,
    RopeStyle::Thd,
    crate::nn::rotary_emb::rope_thd,
    "rope_thd"
);

impl Visit for crate::nn::layer_norm::RmsNorm {
    type In = NodeId;
    type Out = NodeId;
    fn visit(&self, w: &mut Walker, x: Self::In) -> Self::Out {
        let weight = self.weight();
        let wn = w.weight("weight", Shape::of(weight.dims()), weight.dtype());
        w.rms_norm(wn, x)
    }
}

impl Norm for crate::nn::layer_norm::RmsNorm {
    fn execute(&self, x: &Tensor) -> Result<Tensor> {
        lloom_tensor::Module::forward(self, x)
    }
}

impl Visit for crate::models::with_tracing::RmsNorm {
    type In = NodeId;
    type Out = NodeId;
    fn visit(&self, w: &mut Walker, x: Self::In) -> Self::Out {
        let weight = self.weight();
        let wn = w.weight("weight", Shape::of(weight.dims()), weight.dtype());
        w.rms_norm(wn, x)
    }
}

impl Norm for crate::models::with_tracing::RmsNorm {
    fn execute(&self, x: &Tensor) -> Result<Tensor> {
        lloom_tensor::Module::forward(self, x)
    }
}

impl Visit for crate::nn::activation::Activation {
    type In = NodeId;
    type Out = NodeId;
    fn visit(&self, w: &mut Walker, x: Self::In) -> Self::Out {
        use crate::nn::activation::Activation as A;
        match self {
            A::Silu => w.silu(x),
            other => {
                let n = w.node(x);
                let (shape, dt) = (n.shape.clone(), n.dtype);
                let name = format!("{other:?}").to_ascii_lowercase();
                w.opaque(&name, &[x], shape, dt)
            }
        }
    }
}

impl Activation for crate::nn::activation::Activation {
    fn execute(&self, x: &Tensor) -> Result<Tensor> {
        lloom_tensor::Module::forward(self, x)
    }
}

/// The SiLU a GEMV store fuses (`nn::fusion::Silu`) is the same activation
/// as a dispatch of its own: one definition, two placements (§12.2k).
impl Visit for crate::nn::fusion::Silu {
    type In = NodeId;
    type Out = NodeId;
    fn visit(&self, w: &mut Walker, x: Self::In) -> Self::Out {
        w.silu(x)
    }
}

impl Activation for crate::nn::fusion::Silu {
    fn execute(&self, x: &Tensor) -> Result<Tensor> {
        x.silu()
    }
}

/// A dense SwiGLU feed-forward over quantized or f16 stems: `down(silu(gate(x))
/// · up(x))` — the shared expert's arithmetic (`qwen4exp::moe::swiglu`) as a
/// block of its own, so a dense family (Qwen3.5) and the routed MoE's shared
/// expert compute one definition.
#[derive(Debug, Clone)]
pub struct SwiGlu {
    gate: lloom_tensor::quantized::QMatMul,
    up: lloom_tensor::quantized::QMatMul,
    down: lloom_tensor::quantized::QMatMul,
    /// `(hidden, intermediate)`.
    dims: (usize, usize),
    /// Whether a one-row step runs the SiLU and the product on the
    /// projections' stores (`qwen4exp.shared_store`); off, a `silu_mul`
    /// launch after the two projections.
    store: bool,
}

impl SwiGlu {
    pub fn new(
        gate: lloom_tensor::quantized::QMatMul,
        up: lloom_tensor::quantized::QMatMul,
        down: lloom_tensor::quantized::QMatMul,
        hidden: usize,
        intermediate: usize,
    ) -> Self {
        Self {
            gate,
            up,
            down,
            dims: (hidden, intermediate),
            store: crate::tuning::tuning().qwen4exp.shared_store,
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    /// `[.., hidden]` → `[.., hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::models::qwen4exp::moe::swiglu(x, &self.gate, &self.up, &self.down, self.store)
    }
}

impl Visit for SwiGlu {
    type In = NodeId;
    type Out = NodeId;
    fn visit(&self, w: &mut Walker, x: Self::In) -> Self::Out {
        let gate = w.project_qmat("ffn_gate.weight", &self.gate, x);
        let gate = w.silu(gate);
        let up = w.project_qmat("ffn_up.weight", &self.up, x);
        let h = w.mul(gate, up);
        w.project_qmat("ffn_down.weight", &self.down, h)
    }
}

impl Ffn for SwiGlu {
    type State = ();
    fn execute(&self, x: &Tensor, _state: &mut ()) -> Result<Tensor> {
        self.forward(x)
    }
}
