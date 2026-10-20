//! The EOS gate on raw kernels (lloom #735 step 4, `eos_gate.rs`): a gated
//! kernel (`copy2d`) writes while the sampled token is not a stop token,
//! writes nothing once the gate kernel has seen one, stays shut after
//! (sticky), and an ungated kernel (`affine`) keeps running through it.

use crate::kernels::kernels::unary::copy2d;
use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{
    call_affine, call_copy2d, call_gdn_prologue, call_gdn_step, call_shortconv_step, BufferOffset,
    Device, Kernels,
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
        self.cmds.bind_eos_gate_current().unwrap();
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
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
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
            }
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
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
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
            }
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

/// An ungated scope (`kernels::kernel::UngatedScope`) runs a gated source's
/// kernel past a raised gate, while it lives and only then.
#[test]
fn an_ungated_scope_runs_a_gated_kernel_past_a_raised_gate() {
    let rig = Rig::new();
    rig.begin();
    zero(&rig.dst, N * 4);
    assert_eq!(
        rig.step_copy2d(Some(EOS[0])),
        vec![0.0; N],
        "the gate is up"
    );
    {
        let _ungated = crate::kernels::kernel::UngatedScope::open();
        assert_eq!(
            rig.step_copy2d(Some(7)),
            identity(),
            "inside the scope the copy runs"
        );
    }
    zero(&rig.dst, N * 4);
    assert_eq!(
        rig.step_copy2d(Some(7)),
        vec![0.0; N],
        "after it the gate holds again"
    );
    assert!(rig.end().fired);
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
                        // SAFETY: the test owns every buffer bound or mapped here and reads its
                        // results only after waiting for the command buffer that writes them.
                        unsafe {
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
                        }
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
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
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
            }
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

/// The Gated DeltaNet's decode step under the gate (#993's arm B): the
/// prologue and the recurrence write the next conv window and `S` out of
/// place, into the buffers the host swaps in as the state, so their gated
/// variants carry the state through instead of returning bare. Gate raised:
/// `win_out` is `win` and `s_out` is `s`, bit for bit, and `heads` / `o`
/// keep their poison. Gate clear: the plain kernels, bit for bit (the gated
/// pipeline runs at the plain one's split). Both forms of the step: the loop
/// form and the register form (`gdn_step_r16_f32` at `d` = 16).
#[test]
fn a_gated_gdn_step_carries_its_state_through_an_eos() {
    const D: usize = 16;
    const H_K: usize = 2;
    const H_V: usize = 4;
    const KK: usize = 4;
    const CH: usize = (2 * H_K + H_V) * D;
    let rig = Rig::new();
    let f = |v: &[f32]| {
        rig.device
            .new_buffer_with_data(v.as_ptr() as *const std::ffi::c_void, v.len() * 4, SHARED)
            .unwrap()
    };
    let wave = |n: usize, a: f32, b: f32| -> Vec<f32> {
        (0..n)
            .map(|i| a * ((i as f32) * b).sin() + 0.01 * i as f32)
            .collect()
    };
    let qkv = f(&wave(CH, 0.5, 0.37));
    let conv_w = f(&wave(CH * KK, 0.25, 0.11));
    let win: Vec<f32> = wave(CH * KK, 1.0, 0.23);
    let win_in = f(&win);
    let heads_in = f(&wave(3 * H_V * D, 0.3, 0.19));
    let (alpha, beta) = (f(&wave(H_V, 0.4, 1.3)), f(&wave(H_V, 0.6, 0.7)));
    let (dt_bias, ssm_a) = (f(&wave(H_V, 0.2, 0.9)), f(&vec![-0.5; H_V]));
    let s: Vec<f32> = wave(D * D * H_V, 0.1, 0.013);
    let s_in = f(&s);
    let poisoned = |n: usize| {
        let b = rig.device.new_buffer(n * 4, SHARED).unwrap();
        // A poison the kernel must overwrite (or, gated, leave).
        unsafe { std::ptr::write_bytes(b.contents(), 0xff, n * 4) };
        b
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // (the state written, the output) for one prologue and one step.
    let prologue = |gate_token: Option<u32>| -> (Vec<u32>, Vec<u32>) {
        let (win_out, heads) = (poisoned(CH * KK), poisoned(3 * H_V * D));
        {
            let enc = rig.cmds.command_encoder().unwrap();
            if let Some(t) = gate_token {
                rig.gate_step(t, &enc);
            }
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
                call_gdn_prologue(
                    &rig.device,
                    &enc,
                    &rig.kernels,
                    D,
                    H_K,
                    H_V,
                    KK,
                    1e-6,
                    0.25,
                    BufferOffset::zero_offset(&qkv),
                    BufferOffset::zero_offset(&conv_w),
                    BufferOffset::zero_offset(&win_in),
                    &heads,
                    BufferOffset::zero_offset(&win_out),
                )
            }
            .unwrap();
        }
        rig.cmds.flush_and_wait_current().unwrap();
        (
            bits(&read_f32(&win_out, CH * KK)),
            bits(&read_f32(&heads, 3 * H_V * D)),
        )
    };
    let step = |gate_token: Option<u32>, register: bool| -> (Vec<u32>, Vec<u32>) {
        let (s_out, o) = (poisoned(D * D * H_V), poisoned(H_V * D));
        {
            let enc = rig.cmds.command_encoder().unwrap();
            if let Some(t) = gate_token {
                rig.gate_step(t, &enc);
            }
            // SAFETY: as the prologue's above.
            unsafe {
                call_gdn_step(
                    &rig.device,
                    &enc,
                    &rig.kernels,
                    D,
                    H_V,
                    BufferOffset::zero_offset(&heads_in),
                    BufferOffset::zero_offset(&alpha),
                    BufferOffset::zero_offset(&beta),
                    BufferOffset::zero_offset(&dt_bias),
                    BufferOffset::zero_offset(&ssm_a),
                    BufferOffset::zero_offset(&s_in),
                    &o,
                    BufferOffset::zero_offset(&s_out),
                    register,
                )
            }
            .unwrap();
        }
        rig.cmds.flush_and_wait_current().unwrap();
        (
            bits(&read_f32(&s_out, D * D * H_V)),
            bits(&read_f32(&o, H_V * D)),
        )
    };
    let all_poison = |v: &[u32]| v.iter().all(|&x| x == u32::MAX);

    let plain_pro = prologue(None);
    let plain_loop = step(None, false);
    let plain_reg = step(None, true);
    for (what, (state, out)) in [
        ("prologue", &plain_pro),
        ("loop step", &plain_loop),
        ("register step", &plain_reg),
    ] {
        assert!(
            !all_poison(state) && !all_poison(out),
            "{what}: the plain kernel writes its state and output"
        );
    }
    assert_ne!(
        plain_pro.0,
        bits(&win),
        "the plain prologue moves the window"
    );
    assert_ne!(plain_loop.0, bits(&s), "the plain step moves S");

    rig.begin();
    let clear = (prologue(Some(7)), step(Some(7), false), step(Some(7), true));
    let fired = (
        prologue(Some(EOS[0])),
        step(Some(EOS[0]), false),
        step(Some(EOS[0]), true),
    );
    let report = rig.end();
    assert!(report.fired);
    assert_eq!(clear.0, plain_pro, "gate clear: the plain prologue");
    assert_eq!(clear.1, plain_loop, "gate clear: the plain loop step");
    assert_eq!(clear.2, plain_reg, "gate clear: the plain register step");
    assert_eq!(
        fired.0 .0,
        bits(&win),
        "gate raised: the window carries through"
    );
    assert_eq!(
        fired.1 .0,
        bits(&s),
        "gate raised: the loop step carries S through"
    );
    assert_eq!(
        fired.2 .0,
        bits(&s),
        "gate raised: the register step carries S through"
    );
    assert!(
        all_poison(&fired.0 .1),
        "gate raised: the heads are not written"
    );
    assert!(
        all_poison(&fired.1 .1),
        "gate raised: the loop step's output is not written"
    );
    assert!(
        all_poison(&fired.2 .1),
        "gate raised: the register step's output is not written"
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
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe {
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
            }
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

/// `call_words_publish` copies exactly `n` words at `out`'s offset and writes
/// the stamp only for a non-zero generation: a caller whose stamp a host
/// polls passes 0 and stamps with `ring_stamp` (`Device::words_publish`).
/// `n = 300` runs past the threadgroup's 256 threads, so each thread copies
/// more than one word.
#[test]
fn the_words_publish_copies_n_words_and_stamps_only_off_generation_0() {
    use crate::kernels::call_words_publish;
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let cmds =
        Commands::new_with_completion_wait(queue, residency_set, CompletionWait::Synchronize)
            .unwrap();
    let kernels = Kernels::new();
    const POISON: u32 = 0xdead_beef;
    const UNTOUCHED: u32 = 55;
    for n in [19usize, 300] {
        let words: Vec<u32> = (0..n as u32).map(|i| 7 * i + 3).collect();
        let src = device
            .new_buffer_with_data(words.as_ptr() as *const std::ffi::c_void, n * 4, SHARED)
            .unwrap();
        for generation in [77u32, 0] {
            // `out` at word 2 of a poisoned buffer, the stamp at word 3 of another.
            let out = device.new_buffer((n + 4) * 4, SHARED).unwrap();
            write_u32(&out, &vec![POISON; n + 4]);
            let step = device.new_buffer(32, SHARED).unwrap();
            write_u32(&step, &[UNTOUCHED; 8]);
            {
                let enc = cmds.command_encoder().unwrap();
                // SAFETY: the test owns every buffer bound or mapped here and reads its
                // results only after waiting for the command buffer that writes them.
                unsafe {
                    call_words_publish(
                        &device,
                        &enc,
                        &kernels,
                        BufferOffset::zero_offset(&src),
                        n,
                        BufferOffset {
                            buffer: &out,
                            offset_in_bytes: 8,
                        },
                        BufferOffset {
                            buffer: &step,
                            offset_in_bytes: 12,
                        },
                        generation,
                    )
                }
                .unwrap();
            }
            cmds.flush_and_wait_current().unwrap();
            let got = unsafe { std::slice::from_raw_parts(out.contents() as *const u32, n + 4) };
            assert_eq!(&got[2..n + 2], words.as_slice(), "n={n}: the words");
            assert_eq!(
                [got[0], got[1], got[n + 2], got[n + 3]],
                [POISON; 4],
                "n={n}: the words around them untouched"
            );
            let st = unsafe { std::slice::from_raw_parts(step.contents() as *const u32, 8) };
            let mut want = [UNTOUCHED; 8];
            if generation != 0 {
                want[3] = generation;
            }
            assert_eq!(st, want, "n={n} generation {generation}: the stamp");
        }
    }
}

/// The gate's cores arm on the quantized mat-vec (`kernel_mul_mv_*`,
/// `quantized.metal`): the cores the Qwen3.6 checkpoints decode through
/// (Q4_K, Q6_K) and the simplest block (Q8_0), each against its own plain
/// output on one weight and one activation.
/// - gate armed, the token not a stop: the gated variant is the plain one,
///   bit for bit (the gated pipeline at the plain one's split);
/// - gate fired, cores arm off: the core runs (only the gated sources stop);
/// - gate fired, cores arm on: nothing written, the destination's poison
///   kept bit for bit.
#[test]
fn under_the_cores_arm_a_quantized_matvec_returns_at_once_past_an_eos() {
    use crate::kernels::kernels::quantized::{call_quantized_matmul_mv_t, GemvTriple, GgmlDType};
    use crate::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType};
    const K: usize = 256;
    const NOUT: usize = 32;
    fn blocks<T: GgmlType>(w: &[f32]) -> Vec<u8> {
        let mut ys = vec![T::zeros(); w.len() / T::BLCK_SIZE];
        T::from_float(w, &mut ys);
        let bytes = std::mem::size_of_val(ys.as_slice());
        // SAFETY: the blocks are plain bytes (`repr(C)` structs of ints and
        // halves); this views them as such for the upload.
        unsafe { std::slice::from_raw_parts(ys.as_ptr() as *const u8, bytes) }.to_vec()
    }
    let rig = Rig::new();
    let w: Vec<f32> = (0..NOUT * K)
        .map(|i| ((i as f32) * 0.37).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.11).cos()).collect();
    let lhs = rig
        .device
        .new_buffer_with_data(x.as_ptr() as *const std::ffi::c_void, K * 4, SHARED)
        .unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (dtype, bytes) in [
        (GgmlDType::Q8_0, blocks::<BlockQ8_0>(&w)),
        (GgmlDType::Q4K, blocks::<BlockQ4K>(&w)),
        (GgmlDType::Q6K, blocks::<BlockQ6K>(&w)),
    ] {
        let rhs = rig
            .device
            .new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len(),
                SHARED,
            )
            .unwrap();
        // One mat-vec into a poisoned destination, after the gate's kernel
        // on `token` when the gate is armed.
        let matvec = |token: Option<u32>| -> Vec<f32> {
            let dst = rig.device.new_buffer(NOUT * 4, SHARED).unwrap();
            // A poison the core must overwrite (or, gated, leave).
            unsafe { std::ptr::write_bytes(dst.contents(), 0xff, NOUT * 4) };
            {
                let enc = rig.cmds.command_encoder().unwrap();
                if let Some(t) = token {
                    rig.gate_step(t, &enc);
                }
                // SAFETY: the test owns every buffer bound here and reads the
                // destination only after the command buffer completes.
                unsafe {
                    call_quantized_matmul_mv_t(
                        &rig.device,
                        &enc,
                        &rig.kernels,
                        dtype,
                        GemvTriple::F32F32F32,
                        (1, 1, NOUT, K),
                        &lhs,
                        0,
                        &rhs,
                        0,
                        &dst,
                    )
                }
                .unwrap();
            }
            rig.cmds.flush_and_wait_current().unwrap();
            read_f32(&dst, NOUT)
        };
        let plain = matvec(None);
        assert!(
            plain.iter().all(|v| v.is_finite()),
            "{dtype:?}: the plain core wrote its output"
        );
        rig.begin();
        let clear = matvec(Some(7));
        assert_eq!(
            rig.end(),
            crate::EosGateReport {
                fired: false,
                steps: 1,
                cores: false
            }
        );
        assert_eq!(
            bits(&clear),
            bits(&plain),
            "{dtype:?}: armed, no stop: the plain output"
        );
        rig.begin();
        let cores_off = matvec(Some(EOS[0]));
        assert!(rig.end().fired);
        assert_eq!(
            bits(&cores_off),
            bits(&plain),
            "{dtype:?}: fired, cores arm off: the core still runs"
        );
        rig.begin();
        rig.kernels.set_eos_gate_cores(true);
        let gated_clear = matvec(Some(7));
        assert!(!rig.end().fired);
        rig.begin();
        let gated_fired = matvec(Some(EOS[1]));
        assert!(rig.end().fired);
        rig.kernels.set_eos_gate_cores(false);
        assert_eq!(
            bits(&gated_clear),
            bits(&plain),
            "{dtype:?}: cores arm on, no stop: the gated core is the plain one bit for bit"
        );
        assert!(
            gated_fired.iter().all(|v| v.to_bits() == u32::MAX),
            "{dtype:?}: cores arm on, fired: the core wrote {:?}",
            &gated_fired[..4]
        );
    }
}
