//! The EOS gate on raw kernels (lloom #735 step 4, `eos_gate.rs`): a gated
//! kernel (`copy2d`) writes while the sampled token is not a stop token,
//! writes nothing once the gate kernel has seen one, stays shut after
//! (sticky), and an ungated kernel (`affine`) keeps running through it.

use crate::kernels::kernels::unary::copy2d;
use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{
    call_affine, call_copy2d, call_shortconv_step, BufferOffset, Device, Kernels,
};
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

/// The fused publish (`call_argmax_publish`, lloom #735 step 5) against the
/// host: the greatest value, the lowest index on ties — `reduce.metal`'s
/// `indexed<T>` rule, which `fast_argmax` implements — into both slots, the
/// flag iff the token is a stop token, the stamp last; and with generation 0
/// no stamp at all (the ring stamps with `ring_stamp` instead, whose order a
/// host reading mapped memory can rely on), the stamp word left as it was.
#[test]
fn the_fused_publish_matches_the_host_argmax_and_stamps_after_the_slots() {
    use crate::kernels::call_argmax_publish;
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let cmds =
        Commands::new_with_completion_wait(queue, residency_set, CompletionWait::Synchronize)
            .unwrap();
    let kernels = Kernels::new();
    // A fixed LCG so the case is reproducible; values on a coarse grid so ties
    // are common (about one in eight values shares the maximum bucket).
    let mut seed = 0x2545_F491u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for (n, planted) in [
        (65_536usize, None),
        (1_000, Some(7usize)),
        (64, Some(63)),
        (3, None),
    ] {
        for _ in 0..3 {
            let mut vals: Vec<f32> = (0..n).map(|_| (next() % 64) as f32 - 32.0).collect();
            if let Some(p) = planted {
                // Two copies of the maximum: the lower index must win.
                let m = vals.iter().cloned().fold(f32::MIN, f32::max);
                vals[p] = m;
                vals[(p + 5) % n] = m;
            }
            let host = vals
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| {
                    if v > bv || (v == bv && i < bi) {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as u32;
            let halves: Vec<u16> = vals
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let logits16 = device
                .new_buffer_with_data(halves.as_ptr() as *const std::ffi::c_void, n * 2, SHARED)
                .unwrap();
            let logits32 = device
                .new_buffer_with_data(vals.as_ptr() as *const std::ffi::c_void, n * 4, SHARED)
                .unwrap();
            for (kernel, logits) in [
                ("argmax_publish_f16", &logits16),
                ("argmax_publish_f32", &logits32),
            ] {
                for (eos_is_the_token, generation) in [(false, 77u32), (true, 77), (false, 0)] {
                    let slots = device.new_buffer(64, SHARED).unwrap();
                    zero(&slots, 64);
                    const UNTOUCHED: u32 = 55;
                    // SAFETY: a fresh Shared buffer of 64 bytes, nothing in flight.
                    unsafe { (slots.contents() as *mut u32).add(5).write(UNTOUCHED) };
                    let eos = device.new_buffer(8, SHARED).unwrap();
                    write_u32(
                        &eos,
                        &[if eos_is_the_token { host } else { host + 1 }, 999_999],
                    );
                    let flag = device.new_buffer(4, SHARED).unwrap();
                    zero(&flag, 4);
                    {
                        let enc = cmds.command_encoder().unwrap();
                        call_argmax_publish(
                            &device,
                            &enc,
                            &kernels,
                            kernel,
                            BufferOffset::zero_offset(logits),
                            n,
                            BufferOffset {
                                buffer: &slots,
                                offset_in_bytes: 4,
                            },
                            BufferOffset {
                                buffer: &slots,
                                offset_in_bytes: 12,
                            },
                            BufferOffset::zero_offset(&eos),
                            2,
                            &flag,
                            BufferOffset {
                                buffer: &slots,
                                offset_in_bytes: 20,
                            },
                            generation,
                        )
                        .unwrap();
                    }
                    cmds.flush_and_wait_current().unwrap();
                    let out =
                        unsafe { std::slice::from_raw_parts(slots.contents() as *const u32, 8) };
                    assert_eq!(out[1], host, "{kernel} n={n}: the token slot");
                    assert_eq!(out[3], host, "{kernel} n={n}: the history slot");
                    assert_eq!(
                        out[5],
                        if generation == 0 {
                            UNTOUCHED
                        } else {
                            generation
                        },
                        "{kernel} n={n} generation {generation}: the stamp"
                    );
                    assert_eq!(
                        [out[0], out[2], out[4], out[6]],
                        [0; 4],
                        "{kernel}: neighbours untouched"
                    );
                    let f = unsafe { (flag.contents() as *const u32).read() };
                    assert_eq!(f, eos_is_the_token as u32, "{kernel} n={n}: the flag");
                }
            }
        }
    }
}

/// The fused Shuffle conv step writes the next state into a buffer its caller
/// swaps into the cache every step, so past an EOS its gated variant carries
/// the state through instead of returning bare: `state_out` is `state_in` bit
/// for bit, and the output is not written. With the flag clear the gated
/// variant is the plain kernel. A bare return here leaves the cache a buffer
/// no kernel wrote (`docs/dispatch.md` §11.5z-x).
#[test]
fn a_gated_shortconv_step_carries_its_state_through_an_eos() {
    const HIDDEN: usize = 16;
    const K: usize = 3;
    let rig = Rig::new();
    let f = |v: &[f32]| {
        rig.device
            .new_buffer_with_data(v.as_ptr() as *const std::ffi::c_void, v.len() * 4, SHARED)
            .unwrap()
    };
    let proj: Vec<f32> = (0..HIDDEN).map(|i| 0.25 + i as f32 / 8.0).collect();
    let (b_in, x_in, c_in) = (f(&proj), f(&proj), f(&proj));
    let state: Vec<f32> = (0..HIDDEN * K).map(|i| 1.0 + i as f32).collect();
    let weight: Vec<f32> = (0..HIDDEN * K).map(|i| 0.5 - i as f32 / 64.0).collect();
    let (state_in, weight) = (f(&state), f(&weight));
    let run = |gate_token: Option<u32>| -> (Vec<f32>, Vec<f32>) {
        let state_out = rig.device.new_buffer(HIDDEN * K * 4, SHARED).unwrap();
        let dst = rig.device.new_buffer(HIDDEN * 4, SHARED).unwrap();
        // A poison the kernel must overwrite (or, gated, leave).
        unsafe {
            std::ptr::write_bytes(state_out.contents(), 0xff, HIDDEN * K * 4);
            std::ptr::write_bytes(dst.contents(), 0xff, HIDDEN * 4);
        }
        {
            let enc = rig.cmds.command_encoder().unwrap();
            if let Some(t) = gate_token {
                rig.gate_step(t, &enc);
            }
            call_shortconv_step(
                &rig.device,
                &enc,
                &rig.kernels,
                "shortconv_step_f32_k3",
                (HIDDEN, HIDDEN, 0, 1),
                BufferOffset::zero_offset(&b_in),
                BufferOffset::zero_offset(&x_in),
                BufferOffset::zero_offset(&c_in),
                BufferOffset::zero_offset(&state_in),
                BufferOffset::zero_offset(&weight),
                &state_out,
                &dst,
            )
            .unwrap();
        }
        rig.cmds.flush_and_wait_current().unwrap();
        (read_f32(&state_out, HIDDEN * K), read_f32(&dst, HIDDEN))
    };
    let (plain_state, plain_out) = run(None);
    let shifted: Vec<f32> = (0..HIDDEN)
        .flat_map(|c| [state[c * K + 1], state[c * K + 2], proj[c] * proj[c]])
        .collect();
    assert_eq!(plain_state, shifted, "the plain step shifts the window");

    rig.begin();
    let (clear_state, clear_out) = run(Some(7));
    let (fired_state, fired_out) = run(Some(EOS[0]));
    let report = rig.end();
    assert!(report.fired);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&clear_state),
        bits(&plain_state),
        "gate clear: the plain step"
    );
    assert_eq!(
        bits(&clear_out),
        bits(&plain_out),
        "gate clear: the plain output"
    );
    assert_eq!(
        bits(&fired_state),
        bits(&state),
        "gate raised: the state carries through"
    );
    assert!(
        fired_out.iter().all(|x| x.to_bits() == u32::MAX),
        "gate raised: the output is not written"
    );
}

/// The ring's stamp (`call_ring_stamp`) writes the generation at the step's
/// offset and nothing else: the words around it and the buffers it certifies
/// (bound, never read) are unchanged, and it lands the same on every repeat.
/// What it orders is `lloom`'s `ring_publish_order` / `ring_publish_binds`.
#[test]
fn the_ring_stamp_writes_the_generation_and_nothing_else() {
    use crate::kernels::call_ring_stamp;
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let cmds =
        Commands::new_with_completion_wait(queue, residency_set, CompletionWait::Synchronize)
            .unwrap();
    let kernels = Kernels::new();
    let step = device.new_buffer(32, SHARED).unwrap();
    zero(&step, 32);
    let certified: Vec<u32> = (0..8).map(|i| 1000 + i).collect();
    let cert = device
        .new_buffer_with_data(certified.as_ptr() as *const std::ffi::c_void, 32, SHARED)
        .unwrap();
    for generation in [1u32, 77, u32::MAX, 77] {
        {
            let enc = cmds.command_encoder().unwrap();
            call_ring_stamp(
                &device,
                &enc,
                &kernels,
                BufferOffset {
                    buffer: &step,
                    offset_in_bytes: 12,
                },
                generation,
                [
                    BufferOffset::zero_offset(&cert),
                    BufferOffset {
                        buffer: &cert,
                        offset_in_bytes: 16,
                    },
                ],
            )
            .unwrap();
        }
        cmds.flush_and_wait_current().unwrap();
        let out = unsafe { std::slice::from_raw_parts(step.contents() as *const u32, 8) };
        let mut want = [0u32; 8];
        want[3] = generation;
        assert_eq!(out, want, "the stamp at word 3, every other word untouched");
        let cert_now = unsafe { std::slice::from_raw_parts(cert.contents() as *const u32, 8) };
        assert_eq!(
            cert_now,
            certified.as_slice(),
            "the certified buffer is not written"
        );
    }
}
