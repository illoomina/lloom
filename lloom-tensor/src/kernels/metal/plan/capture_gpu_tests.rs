//! Lowering through the op library: what the capture records, how a node's
//! moving fields are fitted, and what the encoder will do with the result
//! (`design/docs/planned-step-design.md` §10.1 rows 3, 4 and 5).
//!
//! Each reference is outside the mechanism under test: the encoder's own
//! dispatch trace for what was bound and where it put barriers and sessions,
//! the eager `call_*` run on its own queue for what a fitted list computes,
//! and a fresh lowering at a value the fit never saw for what a patch
//! predicts. The trace switch is read once per process, so these run one
//! test per process (nextest).

use std::sync::Arc;

use super::{
    barriers, lower_node, predicts, segments, Affine, Arg, Command, CommandList, LoweredArg,
    LoweredCommand, Lowering, Node, NodeAlloc, PatchTarget, PlanEncoder, Probe, Refused,
    SegmentStart, StepParam, StepValues, Sweep, Width,
};
use crate::kernels::metal::{
    set_hazard_key, trace, Buffer, Commands, CutPolicy, Grid, HazardKey, ResidencySet, Slot,
};
use crate::kernels::{
    call_arg_sort, call_copy2d, call_unary_contiguous, call_unary_strided, unary, BufferOffset,
    RESOURCE_OPTIONS,
};
use crate::metal_backend::MetalDevice;
use crate::{op::BackpropOp, DType, Storage, Tensor};

const P0: StepParam = StepParam(0);
const P1: StepParam = StepParam(1);

/// Turn the dispatch trace on for this process, before anything reads the
/// tuning. A process where something already did cannot see the trace, and
/// every comparison below would then be against nothing.
fn trace_on() {
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    assert!(
        trace::trace_requested(),
        "the dispatch trace is off in this process: run these tests one per process \
         (cargo nextest), so the trace switch is read after it is set"
    );
}

fn live() -> MetalDevice {
    crate::Device::new_metal(0)
        .unwrap()
        .as_metal_device()
        .unwrap()
        .clone()
}

fn commands(device: &MetalDevice) -> Commands {
    let d = device.metal_device();
    Commands::new(
        d.new_command_queue().unwrap(),
        Arc::new(ResidencySet::new(d)),
    )
    .unwrap()
}

fn upload(device: &MetalDevice, data: &[f32]) -> Buffer {
    device
        .metal_device()
        .new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            RESOURCE_OPTIONS,
        )
        .unwrap()
}

/// `n` f32 from `buffer`, `at` bytes in, once its writers have completed.
fn read(buffer: &Buffer, at: usize, n: usize) -> Vec<f32> {
    // SAFETY: every caller waits for the command buffers writing `buffer`
    // first, and the buffer is shared storage holding at least `at + 4n` bytes.
    unsafe {
        let ptr = (buffer.contents() as *const u8).add(at) as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A comparison is evidence only over an output that varies
/// (`invariants.md` §15.1 #1).
fn assert_varied(what: &str, v: &[f32]) {
    let mut distinct = bits(v);
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 2 && v.iter().all(|x| x.is_finite()),
        "{what}: the output is constant or not finite, so equality proves nothing"
    );
}

const N: usize = 1003;
const PAD: usize = 7 * 4;
const SLOT: usize = 4096;
const SENTINEL: f32 = -7.5;

fn input() -> Vec<f32> {
    let mut v = vec![0.0f32; PAD / 4];
    v.extend((0..N).map(|i| (i as f32 * 0.37).sin() * 3.0 + 0.1));
    v
}

fn slots(n: usize) -> Vec<Slot> {
    (0..n)
        .map(|i| Slot {
            offset: i * SLOT,
            size: SLOT,
        })
        .collect()
}

/// A base of `n` slots, every byte of it the sentinel, allocated by the
/// device as the pass base is. The handle keeps it out of the pool.
fn base(device: &MetalDevice, n: usize) -> (crate::kernels::metal::BufferHandle, Buffer) {
    let handle = device
        .new_buffer_with_data(&vec![SENTINEL; n * SLOT / 4])
        .unwrap();
    let buffer = AsRef::<Buffer>::as_ref(&handle).clone();
    (handle, buffer)
}

fn op_err(e: crate::kernels::MetalKernelError) -> crate::Error {
    crate::Error::wrap(e)
}

/// `name` over `N` elements of `src` from byte `at` into `dst`: the
/// contiguous unary every eager elementwise op lowers to.
fn unary_op(
    d: &MetalDevice,
    name: unary::contiguous::Kernel,
    src: &Buffer,
    at: usize,
    dst: &Buffer,
) -> crate::Result<()> {
    let guard = d.command_encoder()?;
    // SAFETY: the caller owns every buffer; under a lowering nothing is
    // dispatched, and eagerly the test reads only after waiting.
    unsafe {
        call_unary_contiguous(
            d.metal_device(),
            &guard,
            &d.kernels,
            name,
            4,
            N,
            BufferOffset {
                buffer: src,
                offset_in_bytes: at,
            },
            dst,
        )
    }
    .map_err(op_err)
}

fn scalars(l: &LoweredCommand) -> Vec<(u32, Vec<u8>)> {
    l.args
        .iter()
        .filter_map(|a| match a {
            LoweredArg::Bytes { index, bytes } => Some((*index, bytes.clone())),
            _ => None,
        })
        .collect()
}

fn words(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Row 3: a lowered op's commands are what the encoder bound (pipeline,
/// every `(index, offset, direction, extent)`, the grid), with the bytes and
/// threadgroup memory the op passed, and the lowering dispatches nothing.
#[test]
fn a_capture_is_what_the_encoder_bound_and_dispatches_nothing() {
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 3);
    let mut lowering = Lowering::new(&device, lowering_base.clone(), slots(3)).unwrap();
    let src = upload(&device, &input());
    let src_ref = lowering.declare(&src);
    let slot = |i: usize| lowering_base.view(i * SLOT, SLOT);
    let (s0, s1, s2) = (slot(0), slot(1), slot(2));

    let window = lowering.device().dispatch_trace().open().unwrap();
    let mut lowered = Vec::new();
    lowered.extend(
        lowering
            .lower(&device, 0, &[], |d| {
                unary_op(d, unary::contiguous::cos::FLOAT, &src, PAD, &s0)
            })
            .unwrap(),
    );
    // The shape and strides reach the encoder as `setBytes` arrays.
    lowered.extend(
        lowering
            .lower(&device, 1, &[], |d| {
                let guard = d.command_encoder()?;
                // SAFETY: as `unary_op`.
                unsafe {
                    call_unary_strided(
                        d.metal_device(),
                        &guard,
                        &d.kernels,
                        unary::strided::exp::FLOAT,
                        &[17, 59],
                        BufferOffset::zero_offset(&s0),
                        &[1, 17],
                        BufferOffset::zero_offset(&s1),
                    )
                }
                .map_err(op_err)
            })
            .unwrap(),
    );
    // Threadgroup memory reaches the encoder by a call of its own.
    lowered.extend(
        lowering
            .lower(&device, 2, &[], |d| {
                let guard = d.command_encoder()?;
                // SAFETY: as `unary_op`.
                unsafe {
                    call_arg_sort(
                        d.metal_device(),
                        &guard,
                        &d.kernels,
                        "asort_asc_f32",
                        3,
                        37,
                        64,
                        BufferOffset::zero_offset(&s1),
                        &s2,
                    )
                }
                .map_err(op_err)
            })
            .unwrap(),
    );
    let seen = window.close();
    lowering.finish().unwrap();

    assert_eq!(lowered.len(), 3, "three ops, one dispatch each");
    assert_eq!(seen.len(), 3, "the trace saw the same three dispatches");
    let table = lowering.buffers().table(&lowering_base);
    for (i, (l, t)) in lowered.iter().zip(&seen).enumerate() {
        assert_eq!(l.pipeline.name(), Some(t.pipeline.as_str()), "command {i}");
        let (grid, by_groups) = match l.grid {
            Grid::Threadgroups(g) => (g, true),
            Grid::Threads(g) => (g, false),
        };
        assert_eq!((grid.width, grid.height, grid.depth), t.grid, "command {i}");
        assert_eq!(by_groups, t.by_threadgroups, "command {i}");
        let threads = (l.threads.width, l.threads.height, l.threads.depth);
        assert_eq!(threads, t.threadgroup, "command {i}");
        let binds: Vec<_> = l
            .args
            .iter()
            .filter_map(|a| match a {
                LoweredArg::Buffer {
                    index,
                    buffer,
                    offset,
                    write,
                } => {
                    let b = table.get(*buffer).unwrap();
                    Some((
                        *index as usize,
                        b.raw_addr(),
                        b.base_offset() + offset,
                        b.length(),
                        *write,
                    ))
                }
                _ => None,
            })
            .collect();
        let traced: Vec<_> = t
            .bindings
            .iter()
            .map(|b| (b.index, b.buffer_addr, b.offset, b.len, b.is_output))
            .collect();
        assert_eq!(binds, traced, "command {i}'s bindings");
    }

    assert_eq!(scalars(&lowered[0]), vec![(0, words(&[N as u64]))]);
    assert_eq!(
        scalars(&lowered[1]),
        vec![
            (0, words(&[17 * 59])),
            (1, words(&[2])),
            (2, words(&[17, 59])),
            (3, words(&[1, 17])),
        ]
    );
    assert_eq!(
        scalars(&lowered[2]),
        vec![(2, words(&[37])), (3, words(&[64]))]
    );
    assert!(lowered[2]
        .args
        .contains(&LoweredArg::ThreadgroupMemory { index: 0, len: 256 }));
    assert_eq!(
        lowered[0].args[1],
        LoweredArg::Buffer {
            index: 1,
            buffer: src_ref,
            offset: PAD,
            write: false
        }
    );
    assert_eq!(
        lowered[0].args[2],
        LoweredArg::Buffer {
            index: 2,
            buffer: lowering.slot(0),
            offset: 0,
            write: true
        }
    );

    let after = read(&lowering_base, 0, 3 * SLOT / 4);
    assert!(
        after.iter().all(|&x| x == SENTINEL),
        "a lowering wrote the base: a dispatch reached the GPU"
    );
}

/// `exp(cos(x))` eagerly, on its own queue and buffers.
fn eager_chain(device: &MetalDevice) -> Vec<f32> {
    let src = upload(device, &input());
    let mid = upload(device, &vec![0.0; N]);
    let out = upload(device, &vec![0.0; N]);
    let cmds = commands(device);
    for (name, from, at, to) in [
        (unary::contiguous::cos::FLOAT, &src, PAD, &mid),
        (unary::contiguous::exp::FLOAT, &mid, 0, &out),
    ] {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns the buffers and reads after the wait.
        unsafe {
            call_unary_contiguous(
                device.metal_device(),
                &guard,
                &device.kernels,
                name,
                4,
                N,
                BufferOffset {
                    buffer: from,
                    offset_in_bytes: at,
                },
                to,
            )
            .unwrap();
        }
    }
    cmds.flush_and_wait().unwrap();
    read(&out, 0, N)
}

/// The capture's commands are a program: lowered on one base, bound to
/// another, encoded by E1, they compute the eager chain bit for bit.
#[test]
fn a_lowered_chain_runs_on_another_base_and_computes_the_eager_bits() {
    trace_on();
    let device = live();
    let eager = eager_chain(&device);
    assert_varied("eager exp(cos(x))", &eager);

    let (_lowering_handle, lowering_base) = base(&device, 2);
    let mut lowering = Lowering::new(&device, lowering_base.clone(), slots(2)).unwrap();
    let src = upload(&device, &input());
    let out = upload(&device, &vec![0.0; N]);
    lowering.declare(&src);
    lowering.declare(&out);
    let s0 = lowering_base.view(0, SLOT);
    let mut list = CommandList::new(0);
    let values = StepValues::new(0);
    for (tag, name, from, at, to) in [
        (0, unary::contiguous::cos::FLOAT, &src, PAD, &s0),
        (1, unary::contiguous::exp::FLOAT, &s0, 0, &out),
    ] {
        let node = Node {
            tag,
            allocs: &[],
            sweeps: &[],
        };
        let n = lower_node(&lowering, &device, &mut list, &node, &values, |d, _| {
            unary_op(d, unary::contiguous::Kernel(name.0), from, at, to)
        })
        .unwrap();
        assert_eq!(n, 1);
    }
    assert_eq!(list.len(), 2);

    let (_live_handle, live_base) = base(&device, 2);
    let table = lowering.buffers().table(&live_base);
    let s0_live = table.get(lowering.slot(0)).unwrap();
    assert_eq!(s0_live.raw_addr(), live_base.raw_addr());
    assert_ne!(live_base.raw_addr(), lowering_base.raw_addr());
    let cmds = commands(&device);
    // SAFETY: the table's buffers live until after the wait below.
    unsafe {
        PlanEncoder::new(&list)
            .encode(&list, &table, &values, &cmds)
            .unwrap()
    };
    cmds.flush_and_wait().unwrap();
    let planned = read(&out, 0, N);
    eprintln!(
        "API={} eager[0..3]={:?} planned[0..3]={:?}",
        cmds.actual_command_api(),
        &eager[..3],
        &planned[..3]
    );
    assert_eq!(bits(&planned), bits(&eager));
}

/// An op that allocates its output takes it from the lowering's arena, so
/// the command binds the declared slot. An output the declaration did not
/// name, one of another size, or one at another dispatch position is not
/// served, and refuses the node.
#[test]
fn an_op_s_output_lands_in_its_declared_slot_and_nowhere_else() {
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 3);
    let mut lowering = Lowering::new(&device, lowering_base.clone(), slots(3)).unwrap();
    let src = upload(&device, &input());
    lowering.declare(&src);
    let elems = N + PAD / 4;
    let exp = |_: &MetalDevice| -> crate::Result<()> {
        let x = Tensor::from_storage(
            Storage::Metal(lowering.storage(&src, elems, DType::F32)),
            (elems,),
            BackpropOp::none(),
            false,
        );
        x.exp().map(drop)
    };
    let bytes = elems * 4;

    let lowered = lowering
        .lower(
            &device,
            7,
            &[NodeAlloc {
                slot: 2,
                size: bytes,
                at: 0,
            }],
            exp,
        )
        .unwrap();
    assert_eq!(lowered.len(), 1);
    let output = lowered[0]
        .args
        .iter()
        .find_map(|a| match a {
            LoweredArg::Buffer {
                buffer,
                write: true,
                offset,
                ..
            } => Some((*buffer, *offset)),
            _ => None,
        })
        .unwrap();
    assert_eq!(output, (lowering.slot(2), 0));

    for (allocs, what) in [
        (vec![], "an undeclared output"),
        (
            vec![NodeAlloc {
                slot: 2,
                size: bytes - 4,
                at: 0,
            }],
            "an output of another size",
        ),
        (
            vec![NodeAlloc {
                slot: 2,
                size: bytes,
                at: 1,
            }],
            "an output at another position",
        ),
    ] {
        let refused = lowering.lower(&device, 7, &allocs, exp).unwrap_err();
        assert_eq!(refused.node, 7);
        assert!(
            matches!(
                refused.reason,
                Refused::Outputs {
                    served: 0,
                    offers: 1,
                    ..
                }
            ),
            "{what}: {refused}"
        );
    }
}

const D1: usize = 4;
const D2: usize = 8;
const BLOCK: usize = D1 * D2;
const STEPS: usize = 6;

fn gather_source() -> Vec<f32> {
    (0..96).map(|i| (i as f32 * 1.7).cos() * 9.0).collect()
}

/// A strided gather of `D1 x D2` elements, the source stride and the output
/// offset as given: the two fields a step can move here.
fn gather(d: &MetalDevice, src: &Buffer, dst: &Buffer, stride: i64, at: i64) -> crate::Result<()> {
    let guard = d.command_encoder()?;
    // SAFETY: as `unary_op`.
    unsafe {
        call_copy2d(
            d.metal_device(),
            &guard,
            &d.kernels,
            unary::copy2d::FLOAT,
            src,
            dst,
            D1,
            D2,
            stride as usize,
            D2,
            0,
            at as usize,
        )
    }
    .map_err(op_err)
}

/// The list index of patch `p`'s argument within its command.
fn patched_arg(list: &CommandList, c: &Command, p: &super::Patch) -> Arg {
    list.args(c)[(p.arg - c.args.start) as usize]
}

/// Row 4: the fit names exactly the fields that moved, with the functions
/// they follow, and the fitted list computes the eager steps bit for bit,
/// including at values the fit never lowered at.
#[test]
fn a_fitted_node_patches_what_moved_and_computes_every_step() {
    trace_on();
    let device = live();
    let source = gather_source();
    let eager_out = upload(&device, &vec![SENTINEL; STEPS * BLOCK]);
    {
        let src = upload(&device, &source);
        let cmds = commands(&device);
        for p in 0..STEPS {
            let guard = cmds.command_encoder().unwrap();
            // SAFETY: the test owns the buffers and reads after the wait.
            unsafe {
                call_copy2d(
                    device.metal_device(),
                    &guard,
                    &device.kernels,
                    unary::copy2d::FLOAT,
                    &src,
                    &eager_out,
                    D1,
                    D2,
                    D2 + p,
                    D2,
                    0,
                    p * BLOCK * 4,
                )
                .unwrap();
            }
        }
        cmds.flush_and_wait().unwrap();
    }

    let (_lowering_handle, lowering_base) = base(&device, 1);
    let mut lowering = Lowering::new(&device, lowering_base, slots(1)).unwrap();
    let src = upload(&device, &source);
    let out = upload(&device, &vec![SENTINEL; STEPS * BLOCK]);
    let (src_ref, out_ref) = (lowering.declare(&src), lowering.declare(&out));
    // An origin other than zero: a fit that dropped its constant's
    // correction is then off by the coefficient.
    let sweeps = [Sweep {
        param: P0,
        at: [1, 2, 4],
        held_out: 3,
    }];
    let node = Node {
        tag: 9,
        allocs: &[],
        sweeps: &sweeps,
    };
    let op = |d: &MetalDevice, p: &Probe<'_>| {
        let s = p.get(P0);
        gather(d, &src, &out, D2 as i64 + s, s * (BLOCK * 4) as i64)
    };
    let mut list = CommandList::new(1);
    lower_node(
        &lowering,
        &device,
        &mut list,
        &node,
        &StepValues::new(1),
        op,
    )
    .unwrap();
    assert_eq!(list.len(), 1);
    let c = &list.commands()[0];
    assert_eq!(c.tag(), 9);
    let patched: Vec<_> = list
        .patches(c)
        .iter()
        .map(|p| (patched_arg(&list, c, p), p.target(), p.value().clone()))
        .collect();
    let Arg::Bytes { start, .. } = list.args(c)[2] else {
        panic!("argument 2 is the source stride");
    };
    assert_eq!(
        patched,
        vec![
            (
                Arg::Bytes {
                    index: 2,
                    start,
                    len: 8
                },
                PatchTarget::Bytes {
                    at: 0,
                    width: Width::I64
                },
                Affine::new(D2 as i64, &[(P0, 1)]),
            ),
            (
                Arg::Buffer {
                    index: 5,
                    buffer: out_ref,
                    offset: BLOCK * 4,
                    write: true
                },
                PatchTarget::Offset,
                Affine::new(0, &[(P0, (BLOCK * 4) as i64)]),
            ),
        ],
        "the fit patches the stride and the output offset, and nothing else"
    );
    assert!(list.args(c).contains(&Arg::Buffer {
        index: 4,
        buffer: src_ref,
        offset: 0,
        write: false
    }));

    // The fitted list predicts lowerings at values it never saw.
    for v in [3, 5] {
        let mut values = StepValues::new(1);
        values.set(P0, v);
        let fresh = lowering
            .lower(&device, 9, &[], |d| op(d, &Probe::new(&values, &sweeps)))
            .unwrap();
        predicts(&list, 0, &values, &fresh).unwrap();
    }

    let planned_out = upload(&device, &vec![SENTINEL; STEPS * BLOCK]);
    let (_live_handle, live_base) = base(&device, 1);
    let mut table = lowering.buffers().table(&live_base);
    table.set(src_ref, upload(&device, &source));
    table.set(out_ref, planned_out.clone());
    let cmds = commands(&device);
    let mut encoder = PlanEncoder::new(&list);
    let mut values = StepValues::new(1);
    for p in 0..STEPS as i64 {
        values.set(P0, p);
        // SAFETY: the table's buffers live until after the wait below.
        unsafe { encoder.encode(&list, &table, &values, &cmds).unwrap() };
    }
    cmds.flush_and_wait().unwrap();
    let (eager, planned) = (
        read(&eager_out, 0, STEPS * BLOCK),
        read(&planned_out, 0, STEPS * BLOCK),
    );
    assert!(eager.iter().all(|&x| x != SENTINEL));
    // The stride must matter: block 0 and block 4 read different elements.
    assert_ne!(bits(&eager[..BLOCK]), bits(&eager[4 * BLOCK..]));
    assert_eq!(bits(&planned), bits(&eager));
}

/// Row 4's refusals: each shape a patch cannot express refuses the node,
/// names why, and leaves the list as it was.
#[test]
fn a_field_a_patch_cannot_express_refuses_the_node() {
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 1);
    let mut lowering = Lowering::new(&device, lowering_base, slots(1)).unwrap();
    let src = upload(&device, &gather_source());
    let other = upload(&device, &gather_source());
    let out = upload(&device, &vec![0.0; 64 * BLOCK]);
    for b in [&src, &other, &out] {
        lowering.declare(b);
    }
    let sweeps = [Sweep {
        param: P0,
        at: [0, 1, 3],
        held_out: 2,
    }];
    let node = Node {
        tag: 4,
        allocs: &[],
        sweeps: &sweeps,
    };
    let values = StepValues::new(2);
    let refuse = |op: &dyn Fn(&MetalDevice, &Probe<'_>) -> crate::Result<()>| {
        let mut list = CommandList::new(2);
        let r = lower_node(&lowering, &device, &mut list, &node, &values, op).unwrap_err();
        assert_eq!(r.node, 4);
        assert!(list.is_empty(), "a refused node leaves the list as it was");
        r.reason
    };
    let block = (BLOCK * 4) as i64;

    let r = refuse(&|d, p| {
        let s = p.get(P0);
        gather(d, &src, &out, D2 as i64, s * s * block)
    });
    assert!(
        matches!(
            r,
            Refused::NonAffine {
                index: 5,
                param: 0,
                ..
            }
        ),
        "an offset quadratic in the step: {r:?}"
    );

    let r = refuse(&|d, p| gather(d, &src, &out, D2 as i64 + p.get(P1), 0));
    assert_eq!(
        r,
        Refused::Undeclared { param: 1 },
        "a read of an unnamed parameter"
    );

    let r = refuse(&|d, p| {
        let guard = d.command_encoder()?;
        let rows = 1 + p.get(P0) as usize;
        // SAFETY: as `unary_op`.
        unsafe {
            call_copy2d(
                d.metal_device(),
                &guard,
                &d.kernels,
                unary::copy2d::FLOAT,
                &src,
                &out,
                rows,
                D2,
                D2,
                D2,
                0,
                0,
            )
        }
        .map_err(op_err)
    });
    assert_eq!(
        r,
        Refused::Structure {
            command: 0,
            what: "grid"
        },
        "a grid that follows the step"
    );

    let r = refuse(&|d, p| {
        let from = if p.get(P0) == 0 { &src } else { &other };
        gather(d, from, &out, D2 as i64, 0)
    });
    assert_eq!(
        r,
        Refused::Structure {
            command: 0,
            what: "binding"
        },
        "a binding that follows the step"
    );

    let mut list = CommandList::new(2);
    let repeated = [Sweep {
        param: P0,
        at: [0, 1, 1],
        held_out: 2,
    }];
    let r = lower_node(
        &lowering,
        &device,
        &mut list,
        &Node {
            tag: 4,
            allocs: &[],
            sweeps: &repeated,
        },
        &values,
        |d, _| gather(d, &src, &out, D2 as i64, 0),
    )
    .unwrap_err();
    assert_eq!(r.reason, Refused::Samples { param: 0 });
}

/// The held-out value catches what three samples per axis cannot: a field
/// following the product of two parameters is affine along each axis, so
/// the fit accepts it, and the lowering at the held-out point refuses it.
#[test]
fn a_cross_term_is_caught_only_at_the_held_out_value() {
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 1);
    let mut lowering = Lowering::new(&device, lowering_base, slots(1)).unwrap();
    let src = upload(&device, &gather_source());
    let out = upload(&device, &vec![0.0; 64 * BLOCK]);
    lowering.declare(&src);
    lowering.declare(&out);
    let sweeps = [
        Sweep {
            param: P0,
            at: [1, 2, 3],
            held_out: 5,
        },
        Sweep {
            param: P1,
            at: [1, 2, 3],
            held_out: 4,
        },
    ];
    let node = Node {
        tag: 2,
        allocs: &[],
        sweeps: &sweeps,
    };
    let op = |d: &MetalDevice, p: &Probe<'_>| {
        gather(
            d,
            &src,
            &out,
            D2 as i64,
            p.get(P0) * p.get(P1) * (BLOCK * 4) as i64,
        )
    };
    let mut list = CommandList::new(2);
    let fitted = lower_node(
        &lowering,
        &device,
        &mut list,
        &node,
        &StepValues::new(2),
        op,
    );
    if cfg!(feature = "planned-validate") {
        let r = fitted.unwrap_err();
        assert!(matches!(r.reason, Refused::HeldOut { index: 5, .. }), "{r}");
        return;
    }
    assert_eq!(
        fitted.unwrap(),
        1,
        "without the check, the cross term is fitted"
    );
    let mut held = StepValues::new(2);
    held.set(P0, 5);
    held.set(P1, 4);
    let fresh = lowering
        .lower(&device, 2, &[], |d| op(d, &Probe::new(&held, &sweeps)))
        .unwrap();
    let r = predicts(&list, 0, &held, &fresh).unwrap_err();
    assert!(matches!(r, Refused::HeldOut { index: 5, .. }), "{r:?}");
}

/// Row 5: over a list whose commands chain through slots of one base, the
/// predicted sessions and barriers are the ones the encoder emitted, under
/// both hazard keys, at three cut periods, and from a queue whose session
/// was already open.
#[test]
fn the_predicted_sessions_and_barriers_are_the_encoders() {
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 4);
    let mut lowering = Lowering::new(&device, lowering_base.clone(), slots(4)).unwrap();
    let src = upload(&device, &input());
    lowering.declare(&src);
    let s = |i: usize| lowering_base.view(i * SLOT, SLOT);
    let (c, e, n) = (
        unary::contiguous::cos::FLOAT.0,
        unary::contiguous::exp::FLOAT.0,
        unary::contiguous::neg::FLOAT.0,
    );
    // RAW through slot 0, an independent write to slot 2, RAW through slot
    // 1, a write over slot 0 after its read, and so on: under the range key
    // only the true edges order, under the pointer key every slot is one
    // resource.
    let steps: [(&'static str, Option<usize>, usize); 7] = [
        (c, None, 0),
        (e, Some(0), 1),
        (c, None, 2),
        (n, Some(1), 0),
        (e, Some(2), 3),
        (c, None, 1),
        (n, Some(3), 2),
    ];
    let mut list = CommandList::new(0);
    let values = StepValues::new(0);
    for (tag, &(name, from, to)) in steps.iter().enumerate() {
        let node = Node {
            tag: tag as u32,
            allocs: &[],
            sweeps: &[],
        };
        lower_node(
            &lowering,
            &device,
            &mut list,
            &node,
            &values,
            |d, _| match from {
                Some(f) => unary_op(d, unary::contiguous::Kernel(name), &s(f), 0, &s(to)),
                None => unary_op(d, unary::contiguous::Kernel(name), &src, PAD, &s(to)),
            },
        )
        .unwrap();
    }
    let (_live_handle, live_base) = base(&device, 4);
    let table = lowering.buffers().table(&live_base);
    let scratch = upload(&device, &vec![1.0; N]);
    let scratch_out = upload(&device, &vec![0.0; N]);

    let mut counts = Vec::new();
    for key in [HazardKey::Range, HazardKey::Pointer] {
        set_hazard_key(key);
        for (period, before) in [(50, 0usize), (3, 0), (3, 2), (4, 3)] {
            let cmds = commands(&device);
            cmds.set_compute_per_buffer(period);
            // Unrelated work first, so the list starts inside an open
            // session, `before` dispatches past its last cut.
            for _ in 0..before {
                let guard = cmds.command_encoder().unwrap();
                // SAFETY: the test owns both buffers.
                unsafe {
                    call_unary_contiguous(
                        device.metal_device(),
                        &guard,
                        &device.kernels,
                        unary::contiguous::Kernel(e),
                        4,
                        N,
                        BufferOffset::zero_offset(&scratch),
                        &scratch_out,
                    )
                    .unwrap();
                }
            }
            let start = SegmentStart {
                count: cmds.open_dispatches(),
                open: before > 0,
            };
            let window = cmds.dispatch_trace().open().unwrap();
            // SAFETY: the table's buffers live until after the wait below.
            unsafe {
                PlanEncoder::new(&list)
                    .encode(&list, &table, &values, &cmds)
                    .unwrap()
            };
            let seen = window.close();
            cmds.flush_and_wait().unwrap();
            assert_eq!(seen.len(), list.len());

            let predicted =
                segments(list.len(), CutPolicy::DispatchCounter, period, start).unwrap();
            let mut sessions = Vec::new();
            let mut last = start.open.then_some(0);
            for (i, d) in seen.iter().enumerate() {
                if last != Some(d.encoder) {
                    sessions.push(i);
                    last = Some(d.encoder);
                }
            }
            let what = format!("{key:?}, period {period}, {before} before");
            assert_eq!(predicted.sessions, sessions, "sessions: {what}");
            assert!(
                predicted
                    .cuts
                    .iter()
                    .all(|c| predicted.sessions.contains(c)),
                "a cut opens a session: {what}"
            );
            let observed: Vec<usize> = (0..seen.len()).filter(|&i| seen[i].barrier).collect();
            let at = barriers(&list, &table, &values, key, &predicted.sessions).unwrap();
            assert_eq!(at, observed, "barriers: {what}");
            eprintln!(
                "{what}: sessions {sessions:?} cuts {:?} barriers {observed:?}",
                predicted.cuts
            );
            counts.push((key, period, before, observed.len()));
        }
    }
    type Count = (HazardKey, usize, usize, usize);
    let (range, pointer): (Vec<&Count>, Vec<&Count>) =
        counts.iter().partition(|(k, ..)| *k == HazardKey::Range);
    assert!(
        range.iter().all(|r| r.3 > 0),
        "a chain that orders nothing proves nothing: {counts:?}"
    );
    assert!(
        range.iter().zip(&pointer).all(|(r, p)| r.3 < p.3),
        "the keys must disagree at every arm, or one was not exercised: {counts:?}"
    );
}

/// A lowering moves the process-wide step dispatch position and binds
/// through the process-wide recorder hooks, and a `Packed` default binds
/// staged arrays out of the executor's sight: it refuses beside each, and
/// lowers once each has ended.
#[test]
fn a_lowering_refuses_beside_a_live_step_a_recorder_or_packed_params() {
    use crate::kernels::metal::{ArenaLayout, StepPlan};
    use crate::kernels::{set_default_param_style, ParamStyle};
    trace_on();
    let device = live();
    let (_lowering_handle, lowering_base) = base(&device, 1);
    let mut lowering = Lowering::new(&device, lowering_base.clone(), slots(1)).unwrap();
    let src = upload(&device, &input());
    lowering.declare(&src);
    let s0 = lowering_base.view(0, SLOT);
    let lower = || {
        lowering
            .lower(&device, 3, &[], |d| {
                unary_op(d, unary::contiguous::cos::FLOAT, &src, PAD, &s0)
            })
            .map(|l| l.len())
            .map_err(|r| r.reason)
    };
    assert_eq!(lower(), Ok(1), "a quiet device lowers");

    let plan = StepPlan::new(
        vec![Slot {
            offset: 0,
            size: 256,
        }],
        vec![Some((0, 256))],
    );
    device.install_arena(plan, ArenaLayout::Packed).unwrap();
    device.begin_decode_step();
    assert_eq!(lower(), Err(Refused::LiveStep), "beside an open arena step");
    device.end_decode_step();
    device.clear_arena();
    assert_eq!(lower(), Ok(1), "once the step has ended");

    device.begin_arena_recording();
    assert_eq!(
        lower(),
        Err(Refused::Observed),
        "while a recorder observes binds"
    );
    device.finish_arena_recording(ArenaLayout::Packed);
    assert_eq!(lower(), Ok(1), "once the recorder is gone");

    set_default_param_style(ParamStyle::Packed);
    assert_eq!(
        lower(),
        Err(Refused::PackedStyle),
        "under the packed default"
    );
    set_default_param_style(ParamStyle::Split);
    assert_eq!(lower(), Ok(1), "back under the split default");
}
