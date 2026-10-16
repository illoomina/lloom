//! API-specific encoding primitives; hazard tracking stays in `encoder.rs`.
//!
//! Every native barrier, fence operation, blit and `endEncoding` passes
//! through here, so this is where the hazard audit observes them.
#![allow(non_snake_case)]
use super::hazard_audit::{self, StageScope, Stages};
use super::metal4;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::{ffi::c_void, ptr::NonNull};

#[derive(Clone)]
pub(crate) enum Compute {
    Metal3(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Metal4(metal4::Encoder),
}

macro_rules! common {
    ($name:ident($($arg:ident: $ty:ty),*)) => {
        pub fn $name(&self, $($arg: $ty),*) {
            match self { Self::Metal3(e) => e.$name($($arg),*), Self::Metal4(e) => e.raw.$name($($arg),*) }
        }
    };
}

impl Compute {
    common!(dispatchThreads_threadsPerThreadgroup(grid: MTLSize, group: MTLSize));
    common!(dispatchThreadgroups_threadsPerThreadgroup(grid: MTLSize, group: MTLSize));
    common!(setLabel(label: Option<&NSString>));
    #[cfg(feature = "debug-labels")]
    common!(pushDebugGroup(string: &NSString));
    #[cfg(feature = "debug-labels")]
    common!(popDebugGroup());

    /// The native encoder's address: the pass identity the hazard audit keys
    /// on while the encoder is open.
    #[inline]
    pub fn audit_id(&self) -> usize {
        match self {
            Self::Metal3(e) => Retained::as_ptr(e) as usize,
            Self::Metal4(e) => Retained::as_ptr(&e.raw) as usize,
        }
    }
    pub fn endEncoding(&self) {
        match self {
            Self::Metal3(e) => e.endEncoding(),
            Self::Metal4(e) => e.raw.endEncoding(),
        }
        hazard_audit::record_pass_end(self.audit_id());
    }

    pub unsafe fn setThreadgroupMemoryLength_atIndex(&self, length: usize, index: usize) {
        match self {
            Self::Metal3(e) => unsafe { e.setThreadgroupMemoryLength_atIndex(length, index) },
            Self::Metal4(e) => unsafe { e.raw.setThreadgroupMemoryLength_atIndex(length, index) },
        }
    }
    pub fn setComputePipelineState(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        match self {
            Self::Metal3(e) => e.setComputePipelineState(pipeline),
            Self::Metal4(e) => e.pipeline(pipeline),
        }
    }
    pub unsafe fn setBuffer_offset_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        index: usize,
    ) {
        match self {
            Self::Metal3(e) => unsafe { e.setBuffer_offset_atIndex(buffer, offset, index) },
            Self::Metal4(e) => e.buffer(index, buffer, offset),
        }
    }
    pub unsafe fn setBytes_length_atIndex(
        &self,
        bytes: NonNull<c_void>,
        length: usize,
        index: usize,
    ) {
        match self {
            Self::Metal3(e) => unsafe { e.setBytes_length_atIndex(bytes, length, index) },
            Self::Metal4(e) => unsafe { e.bytes(index, bytes, length) },
        }
    }
    pub fn memoryBarrierWithScope(&self, scope: MTLBarrierScope) {
        let stages = match self {
            Self::Metal3(e) => {
                e.memoryBarrierWithScope(scope);
                Stages::DISPATCH
            }
            Self::Metal4(e) => {
                e.raw
                    .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                        MTLStages::Dispatch | MTLStages::Blit,
                        MTLStages::Dispatch | MTLStages::Blit,
                        MTL4VisibilityOptions::Device,
                    );
                Stages::ALL
            }
        };
        hazard_audit::record_pass_barrier(
            self.audit_id(),
            StageScope {
                after: stages,
                before: stages,
            },
        );
    }
    pub fn waitForFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        let stages = match self {
            Self::Metal3(e) => {
                e.waitForFence(fence);
                Stages::DISPATCH
            }
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .waitForFence_beforeEncoderStages(fence, MTLStages::Dispatch | MTLStages::Blit);
                Stages::ALL
            }
        };
        hazard_audit::record_fence_wait(self.audit_id(), fence, stages);
    }
    pub fn updateFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        let stages = match self {
            Self::Metal3(e) => {
                e.updateFence(fence);
                Stages::DISPATCH
            }
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .updateFence_afterEncoderStages(fence, MTLStages::Dispatch | MTLStages::Blit);
                Stages::ALL
            }
        };
        hazard_audit::record_fence_update(self.audit_id(), fence, stages);
    }
    pub unsafe fn useResources_count_usage(
        &self,
        resources: NonNull<NonNull<ProtocolObject<dyn MTLResource>>>,
        count: usize,
        usage: MTLResourceUsage,
    ) {
        match self {
            Self::Metal3(e) => unsafe { e.useResources_count_usage(resources, count, usage) },
            Self::Metal4(e) => {
                for p in unsafe { std::slice::from_raw_parts(resources.as_ptr(), count) } {
                    e.resource(unsafe { p.as_ref() });
                }
            }
        }
    }
    pub unsafe fn executeCommandsInBuffer_withRange(
        &self,
        buffer: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        range: NSRange,
    ) {
        match self {
            Self::Metal3(e) => unsafe { e.executeCommandsInBuffer_withRange(buffer, range) },
            Self::Metal4(e) => {
                e.resource(ProtocolObject::from_ref(buffer));
                unsafe { e.raw.executeCommandsInBuffer_withRange(buffer, range) }
            }
        }
    }
}

pub(crate) enum Blit {
    Metal3(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    Metal4(metal4::Encoder),
}
impl Blit {
    common!(setLabel(label: Option<&NSString>));

    /// The native encoder's address; see [`Compute::audit_id`].
    #[inline]
    pub fn audit_id(&self) -> usize {
        match self {
            Self::Metal3(e) => Retained::as_ptr(e) as usize,
            Self::Metal4(e) => Retained::as_ptr(&e.raw) as usize,
        }
    }
    pub fn endEncoding(&self) {
        match self {
            Self::Metal3(e) => e.endEncoding(),
            Self::Metal4(e) => e.raw.endEncoding(),
        }
        hazard_audit::record_pass_end(self.audit_id());
    }
    pub fn waitForFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.waitForFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .waitForFence_beforeEncoderStages(fence, MTLStages::Blit);
            }
        }
        hazard_audit::record_fence_wait(self.audit_id(), fence, Stages::BLIT);
    }
    pub fn updateFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.updateFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw.updateFence_afterEncoderStages(fence, MTLStages::Blit);
            }
        }
        hazard_audit::record_fence_update(self.audit_id(), fence, Stages::BLIT);
    }
    pub unsafe fn copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
        &self,
        src: &ProtocolObject<dyn MTLBuffer>,
        src_offset: usize,
        dst: &ProtocolObject<dyn MTLBuffer>,
        dst_offset: usize,
        size: usize,
    ) {
        match self {
            Self::Metal3(e) => unsafe {
                e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src, src_offset, dst, dst_offset, size,
                )
            },
            Self::Metal4(e) => {
                e.resource(ProtocolObject::from_ref(src));
                e.resource(ProtocolObject::from_ref(dst));
                unsafe {
                    e.raw
                        .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            src, src_offset, dst, dst_offset, size,
                        )
                }
            }
        }
        hazard_audit::record_blit(
            self.audit_id(),
            "blit_copy",
            &[
                (buffer_id(src), src_offset, size, false),
                (buffer_id(dst), dst_offset, size, true),
            ],
        );
    }
    pub fn fillBuffer_range_value(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        range: NSRange,
        value: u8,
    ) {
        match self {
            Self::Metal3(e) => e.fillBuffer_range_value(buffer, range, value),
            Self::Metal4(e) => {
                e.resource(ProtocolObject::from_ref(buffer));
                unsafe { e.raw.fillBuffer_range_value(buffer, range, value) }
            }
        }
        hazard_audit::record_blit(
            self.audit_id(),
            "blit_fill",
            &[(buffer_id(buffer), range.location, range.length, true)],
        );
    }
}

/// A native buffer's address, the key compute bindings record
/// (`Buffer::raw_ptr`).
fn buffer_id(buffer: &ProtocolObject<dyn MTLBuffer>) -> usize {
    buffer as *const ProtocolObject<dyn MTLBuffer> as usize
}
