//! What an ICB executor would have to get right, tested before one exists.
//!
//! `DESIGN.md` §11.1 defers the ICB path; issue #32 established by probe that
//! the mechanism works on this machine and recorded three ways it fails
//! *silently*. These tests pin the one that produces plausible-looking wrong
//! output (trap 1) and the one that is a per-step obligation (trap 2), so that
//! whoever builds the executor inherits a failing test rather than a paragraph.
//!
//! They are deliberately written against raw Metal rather than through
//! `ComputeCommandEncoder`: candle has no ICB path yet, and the point is to
//! characterise Metal's behaviour, not candle's.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLLibrary, MTLPipelineOption, MTLResourceOptions, MTLSize,
};

const N: usize = 256;

/// Writes `in[i] + 1` so an all-zero output is unambiguously wrong: the input
/// is 1.0, so a correct run yields 2.0 everywhere and a silently-skipped
/// dispatch yields 0.0. A kernel that merely copied would make the trap-1
/// failure indistinguishable from a zeroed buffer.
const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device const float *in  [[buffer(0)]],
                    device float       *out [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    out[i] = in[i] + 1.0f;
}
"#;

struct Fixture {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

fn fixture() -> Fixture {
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("no command queue");
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SRC), None)
        .expect("probe kernel failed to compile");
    Fixture {
        device,
        queue,
        library,
    }
}

/// Build a pipeline for `add_one`, with ICB support on or off.
///
/// The flag is the whole subject of trap 1, so it is a parameter rather than a
/// constant: the test asserts on the difference between the two.
fn pipeline(
    f: &Fixture,
    support_icb: bool,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let func = f
        .library
        .newFunctionWithName(&NSString::from_str("add_one"))
        .expect("add_one missing from library");
    let desc = MTLComputePipelineDescriptor::new();
    desc.setComputeFunction(Some(&func));
    desc.setSupportIndirectCommandBuffers(support_icb);
    f.device
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &desc,
            MTLPipelineOption::None,
            None,
        )
        .expect("pipeline creation failed")
}

fn icb(f: &Fixture, max_commands: usize) -> Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>> {
    let desc = MTLIndirectCommandBufferDescriptor::new();
    // Dispatch cannot be mixed with any other command type in one descriptor
    // (MTLIndirectCommandBuffer.h), so this is the only type set.
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatchThreads);
    desc.setInheritBuffers(false);
    desc.setInheritPipelineState(false);
    desc.setMaxKernelBufferBindCount(2);
    unsafe {
        f.device
            .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &desc,
                max_commands,
                MTLResourceOptions::empty(),
            )
    }
    .expect("ICB creation failed")
}

/// Encode one `add_one` command at `index` over the whole buffer.
fn encode_add_one(
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    index: usize,
    ps: &ProtocolObject<dyn MTLComputePipelineState>,
    io: &Io,
) {
    assert!(index < icb.size(), "command index out of range");
    // SAFETY: `index` is in range, asserted above -- the only precondition.
    let cmd = unsafe { icb.indirectComputeCommandAtIndex(index) };
    cmd.setComputePipelineState(ps);
    unsafe {
        cmd.setKernelBuffer_offset_atIndex(&io.input, 0, 0);
        cmd.setKernelBuffer_offset_atIndex(&io.output, 0, 1);
    }
    cmd.concurrentDispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: N,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
}

/// An input/output pair, allocated once so the buffers an ICB command was
/// encoded against are the same ones the execution reads.
///
/// Getting this wrong is easy and silent: allocating fresh buffers at execute
/// time leaves the encoded commands pointing at the originals, and the result
/// is that the output under test is never written. That failure looks exactly
/// like the trap-1 failure this file exists to detect, which is why the buffers
/// are owned by the caller and passed to both halves.
struct Io {
    input: Retained<ProtocolObject<dyn MTLBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl Io {
    fn new(f: &Fixture) -> Io {
        let input = f
            .device
            .newBufferWithLength_options(N * 4, MTLResourceOptions::empty())
            .unwrap();
        let output = f
            .device
            .newBufferWithLength_options(N * 4, MTLResourceOptions::empty())
            .unwrap();
        Io { input, output }
    }

    /// Set the input to 1.0 and poison the output with NaN.
    ///
    /// NaN rather than 0.0 so that "the dispatch never ran" is distinguishable
    /// from "the dispatch ran and wrote zeros" -- the second is trap 1, the
    /// first is a broken test, and they must not be confused.
    fn prime(&self) {
        // SAFETY: both buffers are shared-storage, `N * 4` bytes, and no GPU
        // work is in flight against them at this point.
        unsafe {
            let p = self.input.contents().as_ptr() as *mut f32;
            let q = self.output.contents().as_ptr() as *mut f32;
            for i in 0..N {
                p.add(i).write(1.0);
                q.add(i).write(f32::NAN);
            }
        }
    }

    fn read_output(&self) -> Vec<f32> {
        // SAFETY: the command buffer has completed before this is called, and
        // the buffer holds `N` f32 in shared storage.
        unsafe { std::slice::from_raw_parts(self.output.contents().as_ptr() as *const f32, N) }
            .to_vec()
    }
}

/// Execute `range` of `icb` against `io`, and return what landed in the output.
fn run(
    f: &Fixture,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    range: std::ops::Range<usize>,
    io: &Io,
) -> Vec<f32> {
    io.prime();
    let cb = f.queue.commandBuffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();
    // SAFETY: the resources outlive the command buffer (owned by `io`), and the
    // range is within the ICB's command count at every call site.
    //
    // Residency is mandatory for ICB-referenced resources and omitting it is
    // another silent-corruption source (`DESIGN.md` §3.7). Unified memory can
    // mask the omission, so it is done correctly here rather than relied upon.
    unsafe {
        enc.useResource_usage(
            ProtocolObject::from_ref(&*io.input),
            objc2_metal::MTLResourceUsage::Read,
        );
        enc.useResource_usage(
            ProtocolObject::from_ref(&*io.output),
            objc2_metal::MTLResourceUsage::Write,
        );
        enc.executeCommandsInBuffer_withRange(
            icb,
            objc2_foundation::NSRange {
                location: range.start,
                length: range.len(),
            },
        );
    }
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    assert!(
        cb.error().is_none(),
        "command buffer reported an error: {:?}",
        cb.error()
    );
    io.read_output()
}

/// Trap 1 (`DESIGN.md` §3.7a, issue #32): a pipeline encoded into an ICB must
/// be built with `supportIndirectCommandBuffers`.
///
/// This asserts the **positive** arm: with the flag set, a CPU-encoded ICB
/// dispatch computes `in + 1` everywhere. That is the property an executor
/// depends on, and it is the one that can be checked in-process.
///
/// # What the negative arm actually does, and why it is not asserted here
///
/// Issue #32 records the failure as `status=completed`, `error=nil`, **output
/// all zeros** -- a silent wrong answer. Measured on this machine (M1 Max,
/// macOS 26.5) it is not silent, and it differs by which route encodes the
/// command:
///
/// | encode route | flag off |
/// |---|---|
/// | CPU-side, `MTLIndirectComputeCommand` (this file) | **SIGSEGV inside `setComputePipelineState:`**, at encode time |
/// | GPU-side via `MTLArgumentEncoder` (#32's probe) | **GPU hang**, `status=Error`, `kIOGPUCommandBufferCallbackErrorHang`, output untouched |
///
/// Neither is the recorded all-zeros behaviour, and neither can be a unit test:
/// one kills the process and the other faults the device, so a `#[should_panic]`
/// would take the test binary or the GPU down with it. Both were reproduced in
/// standalone Objective-C, which is the honest place for a probe that is
/// expected to crash -- see the PR for issue #35 and `DESIGN.md` §3.7a.
///
/// The practical guidance is unchanged and if anything stronger: set the flag.
/// What changes is the failure you should expect if you forget, which matters
/// because "output all zeros" would send you looking at your kernel and a hang
/// or a segfault sends you to the pipeline descriptor.
#[test]
fn icb_requires_support_indirect_command_buffers() {
    let f = fixture();
    let io = Io::new(&f);

    let good = icb(&f, 1);
    let ps_good = pipeline(&f, true);
    encode_add_one(&good, 0, &ps_good, &io);
    let with_support = run(&f, &good, 0..1, &io);

    assert!(
        with_support.iter().all(|&v| v == 2.0),
        "a supported pipeline should compute in+1 everywhere, got {:?}",
        &with_support[..8]
    );
    // Guards against the failure this file's first draft actually hit: buffers
    // allocated at execute time rather than encode time leave the encoded
    // command pointing elsewhere, so the output keeps its NaN poison and every
    // "all zeros" assertion below would be vacuous.
    assert!(
        !with_support.iter().any(|v| v.is_nan()),
        "output still holds its poison value, so the dispatch never wrote it"
    );
}

/// Trap 2 (`DESIGN.md` §3.7a, issue #32): commands the CPU never initialised
/// are undefined behaviour, so a `MAX_CHUNKS`-slot ICB must `reset()` the tail
/// it does not use -- every step, not once at setup.
///
/// What this test can honestly assert is the *executable* half: after encoding
/// fewer commands than the ICB holds, executing only the encoded sub-range is
/// correct, and `reset` on the remainder is accepted and leaves that sub-range
/// correct. It deliberately does **not** assert anything about the contents
/// produced by executing an uninitialised command, because that is UB -- a test
/// that pinned today's UB behaviour would be asserting something Metal is free
/// to change, and would fail for a reason unrelated to a defect.
#[test]
fn icb_unused_tail_must_be_reset() {
    let f = fixture();
    let max_commands = 8;
    let used = 3;

    let buf = icb(&f, max_commands);
    let ps = pipeline(&f, true);
    let io = Io::new(&f);

    // Every encoded command covers the whole buffer and writes the same value,
    // so they are idempotent rather than racing: `DESIGN.md` §3.5 gives no
    // ordering between concurrent dispatches, and a test that depended on one
    // would be testing scheduling rather than the trap.
    for i in 0..used {
        encode_add_one(&buf, i, &ps, &io);
    }
    // The per-step obligation. Encoding fewer commands than the ICB holds is
    // the normal case for a MAX_CHUNKS layout, and this is the line that makes
    // it defined.
    for i in used..max_commands {
        // SAFETY: `i < max_commands`, the count the ICB was created with.
        unsafe { buf.indirectComputeCommandAtIndex(i) }.reset();
    }

    let out = run(&f, &buf, 0..used, &io);
    assert!(
        out.iter().all(|&v| v == 2.0),
        "executing the encoded sub-range should be correct, got {:?}",
        &out[..8]
    );

    // Resetting is idempotent and may be repeated per step, which is what makes
    // "reset the tail every step" a viable obligation rather than a one-off.
    for i in used..max_commands {
        // SAFETY: as above -- `i < max_commands`.
        unsafe { buf.indirectComputeCommandAtIndex(i) }.reset();
    }
    let again = run(&f, &buf, 0..used, &io);
    assert_eq!(
        out, again,
        "re-resetting the unused tail must not disturb the encoded commands"
    );
}

/// The constraint that decides whether candle's decode stream can be replayed
/// through an ICB at all, asserted against the Objective-C runtime rather than
/// a header.
///
/// `MTLIndirectComputeCommand` has no `setBytes` in any form. Candle binds
/// inline constants at essentially every dispatch -- `call_rms_norm` alone is
/// 77 of the 675 dispatches in a decode token and passes `length`,
/// `elements_to_sum` and `eps` that way -- so an ICB path requires promoting
/// every such constant into a device buffer first. See
/// [`crate::kernels::metal::executor::IcbFeasibility`].
///
/// This is a test rather than a comment because it is the kind of claim that
/// should be re-checked by machine when the SDK moves: if Apple ever adds the
/// method, this fails and the ICB path becomes cheaper than currently recorded.
#[test]
fn icb_command_cannot_bind_inline_constants() {
    use objc2::runtime::NSObjectProtocol;

    let f = fixture();
    let buf = icb(&f, 1);
    // SAFETY: the ICB was created with one command, so index 0 is in range.
    let cmd = unsafe { buf.indirectComputeCommandAtIndex(0) };

    for selector in [
        c"setKernelBytes:length:atIndex:",
        c"setBytes:length:atIndex:",
    ] {
        let sel = objc2::runtime::Sel::register(selector);
        assert!(
            !cmd.respondsToSelector(sel),
            "MTLIndirectComputeCommand gained `{selector:?}`; the setBytes blocker on the \
             ICB path (DESIGN.md §11.3a) may no longer hold and should be re-measured"
        );
    }

    // The positive control: the binding primitive that does exist. Without it a
    // typo in the selector strings above would make this test vacuous.
    let sel = objc2::runtime::Sel::register(c"setKernelBuffer:offset:atIndex:");
    assert!(
        cmd.respondsToSelector(sel),
        "setKernelBuffer:offset:atIndex: missing -- the probe is not testing what it thinks"
    );
}
