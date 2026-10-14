//! Explicit rowwise Transformer-XL attention policy for the pinned omni audio
//! encoder. This is not ordinary SDPA: the relative-position score and the
//! post-softmax zero mask are both part of the checkpoint's contract.
use crate::nn::attention::bounded::OpaqueWorkspace;
use lloom_tensor::{DType, Device, Result, Tensor};

/// Projection owner supplies Q+u, Q+v, K, V as contiguous [1,H,T,D], and
/// projected positions [1,H,2*T-1,D]. A single clip has `valid` real frames.
/// No tensor-value scan or implicit CPU readback occurs.
pub struct RelativeInputs<'a> {
    pub content_query: &'a Tensor,
    pub position_query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub positions: &'a Tensor,
    pub valid: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RelativeScratch {
    /// Conservative total Tensor allocation volume for one execute, including
    /// ALL query rows and final concatenation. Not a per-row/physical peak.
    pub tensor_allocation_volume: usize,
    /// Explicit host key-validity bytes uploaded by one execute.
    pub host_mask_volume: usize,
    /// Backend workspace/driver/runtime obligations remain additional.
    pub opaque_workspace: OpaqueWorkspace,
}

/// Checked borrowed plan, not admission. Every execute requires a fresh outer
/// model-domain pass reservation covering this volume and projection owners.
pub struct RelativeAttentionPlan<'a> {
    input: RelativeInputs<'a>,
    heads: usize,
    time: usize,
    width: usize,
    scratch: RelativeScratch,
}
impl<'a> RelativeAttentionPlan<'a> {
    pub fn new(input: RelativeInputs<'a>) -> Result<Self> {
        let (b, h, t, d) = input.content_query.dims4()?;
        if b != 1 || [h, t, d].contains(&0) || input.valid > t {
            lloom_tensor::bail!("relative audio attention requires one positive clip and valid<=T")
        }
        let positions = t
            .checked_mul(2)
            .and_then(|x| x.checked_sub(1))
            .ok_or_else(|| lloom_tensor::Error::Msg("relative position length overflow".into()))?;
        let dtype = input.content_query.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
            lloom_tensor::bail!("relative audio attention supports F16/BF16/F32 explicitly")
        }
        let product = |dims: &[usize]| -> Result<usize> {
            dims.iter()
                .try_fold(1usize, |n, &x| n.checked_mul(x))
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| {
                    lloom_tensor::Error::Msg("relative attention product overflow".into())
                })
        };
        for dim in [h, t, d, positions] {
            i32::try_from(dim).map_err(|_| {
                lloom_tensor::Error::Msg("relative attention exceeds backend I32 dimensions".into())
            })?;
        }
        // CUDA softmax rows are H for each query; dimensions above bound H.
        for (x, shape) in [
            (input.content_query, [1, h, t, d]),
            (input.position_query, [1, h, t, d]),
            (input.key, [1, h, t, d]),
            (input.value, [1, h, t, d]),
            (input.positions, [1, h, positions, d]),
        ] {
            if x.dims() != shape
                || x.dtype() != dtype
                || !x.device().same_device(input.content_query.device())
                || !x.is_contiguous()
            {
                lloom_tensor::bail!(
                    "relative audio attention input shape/dtype/device/contiguity mismatch"
                )
            }
            product(&[product(x.dims())?, dtype.size_in_bytes()])?;
            for &s in x.stride() {
                i64::try_from(s).map_err(|_| {
                    lloom_tensor::Error::Msg("relative attention stride overflow".into())
                })?;
            }
        }
        if matches!(input.content_query.device(), Device::Cuda(_)) {
            // CUDA affine/Map2/where use U32 element launch counts. Strided
            // contiguous copies use the same narrowing. Cat's copy2d has
            // d1=H, d2=D, source stride D, destination stride T*D; all must fit.
            // GEMM batch H and softmax row H/column T are I32-bounded above.
            for dims in [
                &[h, t, d][..],     // K transpose and final output backing
                &[h, positions, d], // position transpose
                &[h, t],            // AC, score arithmetic, masks, softmax
                &[h, positions],    // full BD output
                &[h, d],            // query row materialization and cat copy2d H*D
                &[t, d],            // cat destination stride
                &[t],               // compact mask backing
            ] {
                u32::try_from(product(dims)?).map_err(|_| {
                    lloom_tensor::Error::Msg(
                        "relative attention CUDA launch/copy count exceeds U32".into(),
                    )
                })?;
            }
        }
        let e = dtype.size_in_bytes();
        // Once: K/P transposes materialized, two U8 masks, zero/finite-negative
        // row tensors. Per row: two query copies, AC, full BD, sum, scale,
        // pre-mask, softmax, post-mask and attended output. Cat retains rows.
        let terms = [
            product(&[h, t, d, e])?,
            product(&[h, positions, d, e])?,
            product(&[2, t])?,
            product(&[2, h, t, e])?,
            product(&[t, 3, h, d, e])?,
            product(&[t, 6, h, t, e])?,
            product(&[t, h, positions, e])?,
            product(&[h, t, d, e])?,
        ];
        let volume = terms
            .iter()
            .try_fold(0usize, |n, &x| n.checked_add(x))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("relative attention allocation volume overflow".into())
            })?;
        product(&[t, std::mem::size_of::<Tensor>()])?;
        let opaque_workspace = match input.content_query.device() {
            Device::Cpu => OpaqueWorkspace::CpuRuntime,
            Device::Metal(_) => OpaqueWorkspace::MetalDriver,
            Device::Cuda(_) => OpaqueWorkspace::CudaBlas,
        };
        Ok(Self {
            input,
            heads: h,
            time: t,
            width: d,
            scratch: RelativeScratch {
                tensor_allocation_volume: volume,
                host_mask_volume: t,
                opaque_workspace,
            },
        })
    }
    pub fn scratch(&self) -> RelativeScratch {
        self.scratch
    }
    pub fn execute(&self) -> Result<Tensor> {
        let (h, t, d) = (self.heads, self.time, self.width);
        let device = self.input.content_query.device();
        let dtype = self.input.content_query.dtype();
        let keys = self.input.key.transpose(2, 3)?.contiguous()?;
        let positions = self.input.positions.transpose(2, 3)?.contiguous()?;
        let valid = Tensor::from_vec(
            (0..t)
                .map(|k| u8::from(k < self.input.valid))
                .collect::<Vec<_>>(),
            (1, 1, 1, t),
            device,
        )?
        .broadcast_as((1, h, 1, t))?;
        let invalid = Tensor::zeros((1, 1, 1, t), DType::U8, device)?.broadcast_as((1, h, 1, t))?;
        let zero = Tensor::zeros((1, h, 1, t), dtype, device)?;
        let negative = zero.affine(0., -10000.)?;
        let mut rows = Vec::with_capacity(t);
        for q in 0..t {
            let qc = self.input.content_query.narrow(2, q, 1)?.contiguous()?;
            let qp = self.input.position_query.narrow(2, q, 1)?.contiguous()?;
            let ac = qc.matmul(&keys)?;
            let bd = qp.matmul(&positions)?;
            // Pinned pad/view relative_shift: destination(q,k) maps to padded
            // flat q*(2T-1)+k+T. Its source row is q and source column T-1-q+k.
            // Keep full BD arithmetic before selecting this contiguous window.
            let bd = bd.narrow(3, t - 1 - q, t)?;
            let scores = ((ac + bd)? / (d as f64).sqrt())?;
            let mask = if q < self.input.valid {
                &valid
            } else {
                &invalid
            };
            let scores = mask.where_cond(&scores, &negative)?;
            let weights = crate::nn::ops::softmax_last_dim(&scores)?;
            let weights = mask.where_cond(&weights, &zero)?;
            rows.push(weights.matmul(self.input.value)?);
        }
        Tensor::cat(&rows, 2)
    }
}
