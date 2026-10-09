//! API-specific encoding primitives; hazard tracking stays in `encoder.rs`.
#![allow(non_snake_case)]
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
    common!(endEncoding());
    common!(setLabel(label: Option<&NSString>));

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
        match self {
            Self::Metal3(e) => e.memoryBarrierWithScope(scope),
            Self::Metal4(e) => e
                .raw
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch | MTLStages::Blit,
                    MTLStages::Dispatch | MTLStages::Blit,
                    MTL4VisibilityOptions::Device,
                ),
        }
    }
    pub fn waitForFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.waitForFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .waitForFence_beforeEncoderStages(fence, MTLStages::Dispatch | MTLStages::Blit);
            }
        }
    }
    pub fn updateFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.updateFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .updateFence_afterEncoderStages(fence, MTLStages::Dispatch | MTLStages::Blit);
            }
        }
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
    common!(endEncoding());
    common!(setLabel(label: Option<&NSString>));
    pub fn waitForFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.waitForFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw
                    .waitForFence_beforeEncoderStages(fence, MTLStages::Blit);
            }
        }
    }
    pub fn updateFence(&self, fence: &ProtocolObject<dyn MTLFence>) {
        match self {
            Self::Metal3(e) => e.updateFence(fence),
            Self::Metal4(e) => {
                e.fence(fence);
                e.raw.updateFence_afterEncoderStages(fence, MTLStages::Blit);
            }
        }
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
    }
}
