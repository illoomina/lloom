// Concrete catalog payload companion to reviewed affine-owner V1.
// Inactive: the catalog lease install/caller and authentic input creator must
// compose before this becomes the safe BackendStorage operation route.
use crate::kernels::kernels::affine::{AffineDType, AffineOperation, PreparedAffinePipeline};
use crate::kernels::kernels::params::{ParamStyle, PreparedParamSlot};
use crate::memory_admission::AllocationContext;
use crate::metal_backend::MetalDevice;

/// This enum is stored INLINE in the existing catalog Node. Its exact Node
/// Layout is quoted before creation. No erased callback or per-operation Arc.
/// Adding families extends this same node projection, not another registry.
pub(crate) enum OperationPayload {
    Affine(PreparedAffineOperation),
}

/// Immutable schema/pipeline plus separately mutable authentic staging state.
/// No Commands/namespace/catalog/HeaderPool/position/context backedge exists.
pub(crate) struct PreparedAffineOperation {
    shader: PreparedAffinePipeline,
    params: PreparedParamSlot,
    arrays: Option<AffineArrays>,
}
struct AffineArrays {
    dims: PreparedParamSlot,
    strides: PreparedParamSlot,
    // Caller-specified cold layout capacity; Full requires admitted alternate
    // position/growth before encoder entry, not a permanent rank restriction.
    rank_capacity: usize,
}
#[derive(Debug)]
pub(crate) enum OperationPreparationError {
    Creator(crate::Error),
    Shader(crate::kernels::MetalKernelError),
    Params(crate::kernels::kernels::params::PreparedParamError),
    RankOverflow,
    NeedsColdGrowth,
    LayoutMismatch,
}

/// Projection is available ONLY while the catalog's unique writer authority
/// is held. It cannot replace shader/enum/slot ownership. Pipeline is read-only.
/// The enclosing writer lease moves to actual generation inventory BEFORE
/// native effects and stays there until terminal AND future-emission closure.
pub(crate) struct AffinePreparation<'a> {
    shader: &'a PreparedAffinePipeline,
    params: &'a mut PreparedParamSlot,
    arrays: Option<AffineArrayPreparation<'a>>,
}
struct AffineArrayPreparation<'a> {
    dims: &'a mut PreparedParamSlot,
    strides: &'a mut PreparedParamSlot,
    rank: usize,
}
impl AffinePreparation<'_> {
    pub(crate) fn shader(&self) -> &PreparedAffinePipeline {
        self.shader
    }
    /// Narrow slot operation; no &mut slot/ownership can escape to mem::replace.
    /// Returned Handle is genuine and independently rooted, not a mapped alias.
    pub(crate) fn prepare_params(
        &mut self,
        bytes: &[u8],
        context: &AllocationContext,
        commands: &crate::kernels::metal::Commands,
    ) -> Result<crate::kernels::metal::BufferHandle, OperationPreparationError> {
        self.prepare_params_under(bytes, &context.operation_context(), commands)
    }
    pub(crate) fn prepare_params_under(
        &mut self,
        bytes: &[u8],
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &crate::kernels::metal::Commands,
    ) -> Result<crate::kernels::metal::BufferHandle, OperationPreparationError> {
        self.params
            .prepare_under(bytes, context, commands)
            .map_err(OperationPreparationError::Params)
    }
    pub(crate) fn prepare_arrays(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        context: &AllocationContext,
        commands: &crate::kernels::metal::Commands,
    ) -> Result<
        (
            crate::kernels::metal::BufferHandle,
            crate::kernels::metal::BufferHandle,
        ),
        OperationPreparationError,
    > {
        self.prepare_arrays_under(shape, strides, &context.operation_context(), commands)
    }
    pub(crate) fn prepare_arrays_under(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        context: &crate::memory_admission::OperationContext<'_>,
        commands: &crate::kernels::metal::Commands,
    ) -> Result<
        (
            crate::kernels::metal::BufferHandle,
            crate::kernels::metal::BufferHandle,
        ),
        OperationPreparationError,
    > {
        let arrays = self
            .arrays
            .as_mut()
            .ok_or(OperationPreparationError::LayoutMismatch)?;
        if shape.len() != arrays.rank || strides.len() != arrays.rank {
            return Err(OperationPreparationError::LayoutMismatch);
        }
        // Initialized padding-free integer host slices; cold shader constructor
        // verifies 64-bit usize. No CPU borrow of native Shared storage exists.
        let dims = unsafe {
            std::slice::from_raw_parts(shape.as_ptr().cast::<u8>(), core::mem::size_of_val(shape))
        };
        let strides = unsafe {
            std::slice::from_raw_parts(
                strides.as_ptr().cast::<u8>(),
                core::mem::size_of_val(strides),
            )
        };
        let dims = arrays
            .dims
            .prepare_under(dims, context, commands)
            .map_err(OperationPreparationError::Params)?;
        let strides = arrays
            .strides
            .prepare_under(strides, context, commands)
            .map_err(OperationPreparationError::Params)?;
        Ok((dims, strides))
    }
}
impl PreparedAffineOperation {
    /// Cold factory using actual same-origin backend staging constructors.
    /// # Safety
    /// context is the namespace's SAME admitted promise. The caller has quoted
    /// pipeline/cache/control/native/capsule/shadow terms before birth; shadow
    /// quote covers ACTUAL returned E, or a proven backing upper bound, not only
    /// requested R. Rejected fields are service-rooted; arbitrary destructor
    /// custody remains with the enclosing cold factory, never an engine guard.
    /// Native/IO/raw/foreign/emission uses retain genuine counted authorities.
    pub(crate) unsafe fn cold_in(
        device: &MetalDevice,
        context: &AllocationContext,
        dtype: AffineDType,
        operation: AffineOperation,
        style: ParamStyle,
        strided_rank: Option<usize>,
    ) -> Result<Self, OperationPreparationError> {
        context
            .check_open()
            .map_err(|e| OperationPreparationError::Creator(crate::Error::wrap(e)))?;
        // Rank arithmetic and schema refusal precede any native staging birth.
        let array_bytes = strided_rank
            .map(|rank| {
                rank.checked_mul(core::mem::size_of::<usize>())
                    .ok_or(OperationPreparationError::RankOverflow)
            })
            .transpose()?;
        let shader = unsafe {
            PreparedAffinePipeline::cold(device, dtype, style, operation, strided_rank.is_some())
        }
        .map_err(OperationPreparationError::Shader)?;
        let block_bytes = if strided_rank.is_some() { 24 } else { 16 };
        let params = unsafe { Self::cold_slot(device, context, block_bytes) }?;
        let arrays = match (strided_rank, array_bytes) {
            (Some(rank_capacity), Some(bytes)) => Some(AffineArrays {
                dims: unsafe { Self::cold_slot(device, context, bytes.max(8)) }?,
                strides: unsafe { Self::cold_slot(device, context, bytes.max(8)) }?,
                rank_capacity,
            }),
            _ => None,
        };
        Ok(Self {
            shader,
            params,
            arrays,
        })
    }
    unsafe fn cold_slot(
        device: &MetalDevice,
        context: &AllocationContext,
        bytes: usize,
    ) -> Result<PreparedParamSlot, OperationPreparationError> {
        // Existing real CustodiedAllocation -> PhysicalReady -> rich header.
        // No legacy BufferHandle/native pointer is converted into provenance.
        let slot = unsafe { device.new_prepared_staging_in(bytes, Some(context)) }
            .map_err(OperationPreparationError::Creator)?;
        unsafe { PreparedParamSlot::new_in(slot, context, device.commands()) }
            .map_err(OperationPreparationError::Params)
    }
    /// Caller holds unique authentic catalog writer; no independent &mut
    /// payload acquisition exists. No mutation of immutable shader is exposed.
    pub(crate) fn preparation(
        &mut self,
        rank: Option<usize>,
    ) -> Result<AffinePreparation<'_>, OperationPreparationError> {
        match (rank, self.arrays.as_mut()) {
            (None, None) => Ok(AffinePreparation {
                shader: &self.shader,
                params: &mut self.params,
                arrays: None,
            }),
            (Some(rank), Some(arrays)) if rank <= arrays.rank_capacity => Ok(AffinePreparation {
                shader: &self.shader,
                params: &mut self.params,
                arrays: Some(AffineArrayPreparation {
                    dims: &mut arrays.dims,
                    strides: &mut arrays.strides,
                    rank,
                }),
            }),
            _ => Err(OperationPreparationError::NeedsColdGrowth),
        }
    }
    /// Read-only access after writer transfer; borrow must remain tied to the
    /// actual installed generation lease, not a raw node/pipeline pointer.
    pub(crate) fn shader(&self) -> &PreparedAffinePipeline {
        &self.shader
    }
}
