//! The EOS gate on raw kernels (lloom #735 step 4, `eos_gate.rs`): a gated
//! kernel (`copy2d`) writes while the sampled token is not a stop token,
//! writes nothing once the gate kernel has seen one, stays shut after
//! (sticky), and an ungated kernel (`affine`) keeps running through it.

use crate::kernels::kernels::unary::copy2d;
use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{call_affine, call_copy2d, BufferOffset, Device, Kernels};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn read_f32(b: &crate::kernels::metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}

fn write_u32(b: &crate::kernels::metal::Buffer, v: &[u32]) {
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut u32, v.len()) }
}

fn zero(b: &crate::kernels::metal::Buffer, bytes: usize) {
    unsafe { std::ptr::write_bytes(b.contents(), 0, bytes) }
}

const N: usize = 64;
const EOS: [u32; 2] = [42, 99];

struct Rig {
    device: Device,
    cmds: Commands,
    kernels: Kernels,
    src: crate::kernels::metal::Buffer,
    dst: crate::kernels::metal::Buffer,
    token: crate::kernels::metal::Buffer,
    eos: crate::kernels::metal::Buffer,
}

impl Rig {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue().unwrap();
        let residency_set = Arc::new(ResidencySet::new(&device));
        let cmds =
            Commands::new_with_completion_wait(queue, residency_set, CompletionWait::Synchronize)
                .unwrap();
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let bytes = N * 4;
        let src = device
            .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, SHARED)
            .unwrap();
        let dst = device.new_buffer(bytes, SHARED).unwrap();
        zero(&dst, bytes);
        let token = device.new_buffer(4, SHARED).unwrap();
        let eos = device.new_buffer(8, SHARED).unwrap();
        write_u32(&eos, &EOS);
        Self {
            device,
            cmds,
            kernels: Kernels::new(),
            src,
            dst,
            token,
            eos,
        }
    }

    /// Arm as `MetalDevice::eos_gate_begin` does: the flag, the open
    /// encoder's binding, then the kernel variants.
    fn begin(&self) {
        self.cmds.eos_gate().begin(&self.device).unwrap();
        self.cmds.bind_eos_gate_current();
        self.kernels.set_eos_gated(true);
    }

    fn end(&self) -> crate::EosGateReport {
        self.kernels.set_eos_gated(false);
        self.cmds.eos_gate().end()
    }

    fn gate_step(&self, token: u32, enc: &crate::kernels::metal::CommandsGuard<'_>) {
        write_u32(&self.token, &[token]);
        self.cmds
            .eos_gate()
            .step(
                &self.device,
                enc,
                &self.kernels,
                &self.token,
                0,
                &self.eos,
                0,
                EOS.len(),
            )
            .unwrap();
    }

    /// One step: the gate on `token`, then `copy2d` (gated) `src → dst`.
    fn step_copy2d(&self, token: Option<u32>) -> Vec<f32> {
        {
            let enc = self.cmds.command_encoder().unwrap();
            if let Some(t) = token {
                self.gate_step(t, &enc);
            }
            call_copy2d(
                &self.device,
                &enc,
                &self.kernels,
                copy2d::FLOAT,
                &self.src,
                &self.dst,
                1,
                N,
                N,
                N,
                0,
                0,
            )
            .unwrap();
        }
        self.cmds.flush_and_wait_current().unwrap();
        read_f32(&self.dst, N)
    }

    /// One step: the gate on `token`, then `affine` (ungated) into `dst`.
    fn step_affine(&self, token: u32, mul: f32, add: f32) -> Vec<f32> {
        {
            let enc = self.cmds.command_encoder().unwrap();
            self.gate_step(token, &enc);
            call_affine(
                &self.device,
                &enc,
                &self.kernels,
                "affine_f32",
                4,
                N,
                BufferOffset::zero_offset(&self.src),
                &self.dst,
                mul,
                add,
            )
            .unwrap();
        }
        self.cmds.flush_and_wait_current().unwrap();
        read_f32(&self.dst, N)
    }
}

fn identity() -> Vec<f32> {
    (0..N).map(|i| i as f32).collect()
}

fn expected(mul: f32, add: f32) -> Vec<f32> {
    (0..N).map(|i| i as f32 * mul + add).collect()
}

#[test]
fn a_gated_kernel_writes_until_the_token_is_eos_then_never_again() {
    let rig = Rig::new();
    assert!(!rig.cmds.eos_gate().is_armed());
    rig.begin();
    assert!(rig.cmds.eos_gate().is_armed());

    // Not a stop token: copy2d runs.
    assert_eq!(rig.step_copy2d(Some(7)), identity());

    // A stop token: the gate kernel raises the flag; copy2d returns at once.
    zero(&rig.dst, N * 4);
    assert_eq!(rig.step_copy2d(Some(EOS[1])), vec![0.0; N]);

    // Sticky: a later non-stop token does not reopen it.
    assert_eq!(rig.step_copy2d(Some(7)), vec![0.0; N]);

    let report = rig.end();
    assert!(!rig.cmds.eos_gate().is_armed());
    assert!(report.fired, "{report:?}");
    assert_eq!(report.steps, 3);

    // Disarmed, the plain variant runs regardless of the flag left behind.
    assert_eq!(rig.step_copy2d(None), identity());

    // `begin` clears the flag: a new run starts open.
    zero(&rig.dst, N * 4);
    rig.begin();
    assert_eq!(rig.step_copy2d(Some(7)), identity());
    let report = rig.end();
    assert!(!report.fired);
    assert_eq!(report.steps, 1);
}

#[test]
fn an_ungated_kernel_keeps_running_through_a_raised_gate() {
    let rig = Rig::new();
    rig.begin();
    assert_eq!(rig.step_affine(EOS[0], 2.0, 1.0), expected(2.0, 1.0));
    let report = rig.end();
    assert!(report.fired);
}

#[test]
fn the_gated_variant_is_a_separate_pipeline_and_the_plain_one_is_untouched() {
    let rig = Rig::new();
    let plain = rig
        .kernels
        .load_pipeline(&rig.device, crate::kernels::Source::Unary, "copy2d_f32")
        .unwrap();
    rig.begin();
    let gated = rig
        .kernels
        .load_pipeline(&rig.device, crate::kernels::Source::Unary, "copy2d_f32")
        .unwrap();
    rig.end();
    let plain_again = rig
        .kernels
        .load_pipeline(&rig.device, crate::kernels::Source::Unary, "copy2d_f32")
        .unwrap();
    assert!(!std::ptr::eq(plain.as_ref(), gated.as_ref()));
    assert!(std::ptr::eq(plain.as_ref(), plain_again.as_ref()));
}
