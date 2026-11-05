//! A command list encoded by `PlanEncoder` computes what the eager `call_*`
//! entry points compute, bit for bit, on whichever command API the process
//! selected (`LLOOM_METAL_API`), with its patched fields following the step
//! (`design/docs/planned-step-design.md` §10.1).
//!
//! The reference in every test is the production `call_*` run eagerly on its
//! own queue and buffers; the subject is a list declared with the arguments
//! that `call_*` binds. A wrong argument, index, offset or patch shows as a
//! different output, not as a different list.

use super::plan::{
    Affine, Arg, BindTable, BufferRef, CommandList, PlanEncoder, StepParam, StepValues, Width,
};
use crate::kernels::metal::{Buffer, Commands, ResidencySet, Size3};
use crate::kernels::{
    call_copy2d, call_unary_contiguous, get_block_dims, get_tile_size, linear_split, unary,
    BufferOffset, Device, Kernels, Source,
};
use std::sync::Arc;

const P0: StepParam = StepParam(0);
const P1: StepParam = StepParam(1);

fn device() -> Device {
    Device::system_default().unwrap()
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    Commands::new(queue, Arc::new(ResidencySet::new(device))).unwrap()
}

fn upload(device: &Device, data: &[f32]) -> Buffer {
    device
        .new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            crate::kernels::RESOURCE_OPTIONS,
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

fn size3(s: objc2_metal::MTLSize) -> Size3 {
    s.into()
}

const N: usize = 1003;
/// Bytes of padding before the input: the bindings are unaligned, so an
/// offset the list drops or zeroes cannot pass by coincidence.
const PAD: usize = 7 * 4;
/// Where the intermediate sits in the planned base, past a slot of its own.
const MID: usize = 256;

fn input() -> Vec<f32> {
    let mut v = vec![0.0f32; PAD / 4];
    v.extend((0..N).map(|i| (i as f32 * 0.37).sin() * 3.0 + 0.1));
    v
}

/// `exp(cos(x))`, the eager way: two `call_unary_contiguous`.
fn eager_chain(device: &Device, kernels: &Kernels) -> Vec<f32> {
    let cmds = commands(device);
    let src = upload(device, &input());
    let opts = crate::kernels::RESOURCE_OPTIONS;
    let mid = device.new_buffer(N * 4, opts).unwrap();
    let out = device.new_buffer(N * 4, opts).unwrap();
    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound here and reads them only
        // after waiting for the command buffers that write them.
        unsafe {
            call_unary_contiguous(
                device,
                &guard,
                kernels,
                unary::contiguous::cos::FLOAT,
                4,
                N,
                BufferOffset {
                    buffer: &src,
                    offset_in_bytes: PAD,
                },
                &mid,
            )
            .unwrap();
        }
    }
    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: as above.
        unsafe {
            call_unary_contiguous(
                device,
                &guard,
                kernels,
                unary::contiguous::exp::FLOAT,
                4,
                N,
                BufferOffset::zero_offset(&mid),
                &out,
            )
            .unwrap();
        }
    }
    cmds.flush_and_wait().unwrap();
    read(&out, 0, N)
}

const SRC: BufferRef = BufferRef(0);
const BASE: BufferRef = BufferRef(1);
const OUT: BufferRef = BufferRef(2);

/// The same chain declared as a list: the arguments `call_unary_contiguous`
/// sets under `ParamStyle::Split` (`length`, input, output), its grid, and
/// the intermediate in a slot of a shared base rather than a buffer of its
/// own.
fn planned_chain(device: &Device, kernels: &Kernels) -> CommandList {
    let mut list = CommandList::new(0);
    for (i, (name, input, offset, output, at)) in [
        (unary::contiguous::cos::FLOAT.0, SRC, PAD, BASE, MID),
        (unary::contiguous::exp::FLOAT.0, BASE, MID, OUT, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let pipeline = kernels.load_pipeline(device, Source::Unary, name).unwrap();
        let (groups, threads) = linear_split(&pipeline, N.div_ceil(get_tile_size(4)));
        let mut c = list.command(pipeline, i as u32);
        c.usize(0, N);
        c.buffer(1, input, offset, false);
        c.buffer(2, output, at, true);
        c.dispatch_threadgroups(size3(groups), size3(threads));
    }
    list
}

fn chain_table(device: &Device) -> (BindTable, Buffer) {
    let opts = crate::kernels::RESOURCE_OPTIONS;
    let mut table = BindTable::with_len(3);
    table.set(SRC, upload(device, &input()));
    table.set(BASE, device.new_buffer(MID + N * 4, opts).unwrap());
    let out = device.new_buffer(N * 4, opts).unwrap();
    table.set(OUT, out.clone());
    (table, out)
}

#[test]
fn a_declared_chain_computes_the_eager_chain_bit_for_bit() {
    let device = device();
    let kernels = Kernels::new();
    let eager = eager_chain(&device, &kernels);
    assert_varied("eager exp(cos(x))", &eager);

    let list = planned_chain(&device, &kernels);
    let (table, out) = chain_table(&device);
    let cmds = commands(&device);
    // SAFETY: the table's buffers live until after the wait below.
    unsafe {
        PlanEncoder::new(&list)
            .encode(&list, &table, &StepValues::new(0), &cmds)
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

/// Eight encodings of one list on one queue: one output, run to run
/// (`architecture.md` §2.3.6).
#[test]
fn a_list_encoded_again_computes_the_same_bits() {
    let device = device();
    let kernels = Kernels::new();
    let list = planned_chain(&device, &kernels);
    let (table, out) = chain_table(&device);
    let cmds = commands(&device);
    let mut encoder = PlanEncoder::new(&list);
    let mut runs = Vec::new();
    for _ in 0..8 {
        // SAFETY: the table's buffers live until after the wait below, and
        // the host reads `out` only after it.
        unsafe {
            encoder
                .encode(&list, &table, &StepValues::new(0), &cmds)
                .unwrap()
        };
        cmds.flush_and_wait().unwrap();
        runs.push(bits(&read(&out, 0, N)));
    }
    assert_varied("planned exp(cos(x))", &read(&out, 0, N));
    assert!(runs.windows(2).all(|w| w[0] == w[1]), "the runs differ");
}

const ROWS: usize = 8;
const COLS: usize = 64;
const ROW: usize = COLS * 4;
const SENTINEL: f32 = -7.5;

fn rows() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 8.0)
        .collect()
}

/// An in-place KV append: step `s` copies source row `7 - s` into cache row
/// `s`. Both offsets follow the step, one with a negative coefficient.
#[test]
fn patched_offsets_follow_the_step_like_an_in_place_append() {
    let device = device();
    let kernels = Kernels::new();

    let eager_cache = upload(&device, &vec![SENTINEL; ROWS * COLS]);
    {
        let cmds = commands(&device);
        let src = upload(&device, &rows());
        for s in 0..ROWS {
            let guard = cmds.command_encoder().unwrap();
            // SAFETY: the test owns the buffers and reads after the wait.
            unsafe {
                call_copy2d(
                    &device,
                    &guard,
                    &kernels,
                    unary::copy2d::FLOAT,
                    &src,
                    &eager_cache,
                    1,
                    COLS,
                    COLS,
                    COLS,
                    (ROWS - 1 - s) * ROW,
                    s * ROW,
                )
                .unwrap();
            }
        }
        cmds.flush_and_wait().unwrap();
    }

    let (src_ref, cache_ref) = (BufferRef(0), BufferRef(1));
    let mut list = CommandList::new(2);
    {
        let pipeline = kernels
            .load_pipeline(&device, Source::Unary, unary::copy2d::FLOAT.0)
            .unwrap();
        let mut c = list.command(pipeline, 0);
        c.i64(0, 1);
        c.i64(1, COLS as i64);
        c.i64(2, COLS as i64);
        c.i64(3, COLS as i64);
        let from = c.buffer(4, src_ref, 0, false);
        let to = c.buffer(5, cache_ref, 0, true);
        c.patch_offset(
            from,
            Affine::new(((ROWS - 1) * ROW) as i64, &[(P1, -(ROW as i64))]),
        )
        .unwrap();
        c.patch_offset(to, Affine::new(0, &[(P0, ROW as i64)]))
            .unwrap();
        c.dispatch_threads(
            Size3 {
                width: 1,
                height: COLS,
                depth: 1,
            },
            size3(get_block_dims(1, COLS, 1)),
        );
    }
    let planned_cache = upload(&device, &vec![SENTINEL; ROWS * COLS]);
    let mut table = BindTable::with_len(2);
    table.set(src_ref, upload(&device, &rows()));
    table.set(cache_ref, planned_cache.clone());
    let cmds = commands(&device);
    let mut encoder = PlanEncoder::new(&list);
    let mut values = StepValues::new(2);
    for s in 0..ROWS as i64 {
        values.set(P0, s);
        values.set(P1, s);
        // SAFETY: the table's buffers live until after the wait below.
        unsafe { encoder.encode(&list, &table, &values, &cmds).unwrap() };
    }
    cmds.flush_and_wait().unwrap();

    let (eager, planned) = (
        read(&eager_cache, 0, ROWS * COLS),
        read(&planned_cache, 0, ROWS * COLS),
    );
    assert!(
        eager.iter().all(|&x| x != SENTINEL),
        "the eager reference left a row unwritten, so it is not the append it stands for"
    );
    assert_varied("the appended cache", &eager);
    assert_eq!(bits(&planned), bits(&eager));
}

/// An inline scalar that follows the step: a strided gather whose source
/// stride is `8 + p`, each step's rows written `p` blocks along. One command,
/// a bytes patch and an offset patch.
#[test]
fn a_patched_scalar_follows_the_step() {
    const STEPS: usize = 5;
    const D1: usize = 4;
    const D2: usize = 8;
    const BLOCK: usize = D1 * D2;
    let device = device();
    let kernels = Kernels::new();
    let source: Vec<f32> = (0..64).map(|i| (i as f32 * 1.7).cos() * 9.0).collect();

    let eager_out = upload(&device, &vec![SENTINEL; STEPS * BLOCK]);
    {
        let cmds = commands(&device);
        let src = upload(&device, &source);
        for p in 0..STEPS {
            let guard = cmds.command_encoder().unwrap();
            // SAFETY: the test owns the buffers and reads after the wait.
            unsafe {
                call_copy2d(
                    &device,
                    &guard,
                    &kernels,
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

    let (src_ref, out_ref) = (BufferRef(0), BufferRef(1));
    let mut list = CommandList::new(1);
    {
        let pipeline = kernels
            .load_pipeline(&device, Source::Unary, unary::copy2d::FLOAT.0)
            .unwrap();
        let mut c = list.command(pipeline, 0);
        c.i64(0, D1 as i64);
        c.i64(1, D2 as i64);
        let stride = c.i64(2, D2 as i64);
        c.i64(3, D2 as i64);
        c.buffer(4, src_ref, 0, false);
        let to = c.buffer(5, out_ref, 0, true);
        // Attached out of argument order: the list keeps a command's patches
        // in argument order, which the encoder's single pass relies on.
        c.patch_offset(to, Affine::new(0, &[(P0, (BLOCK * 4) as i64)]))
            .unwrap();
        c.patch_bytes(stride, 0, Width::I64, Affine::new(D2 as i64, &[(P0, 1)]))
            .unwrap();
        c.dispatch_threads(
            Size3 {
                width: D1,
                height: D2,
                depth: 1,
            },
            size3(get_block_dims(D1, D2, 1)),
        );
    }
    let planned_out = upload(&device, &vec![SENTINEL; STEPS * BLOCK]);
    let mut table = BindTable::with_len(2);
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

/// What a builder refuses, and that a refused or abandoned command leaves
/// the list as it was.
#[test]
fn a_builder_refuses_patches_it_cannot_apply() {
    let device = device();
    let kernels = Kernels::new();
    let pipeline = kernels
        .load_pipeline(&device, Source::Unary, unary::copy2d::FLOAT.0)
        .unwrap();
    let mut list = CommandList::new(1);
    let first_buffer = {
        let mut c = list.command(pipeline.clone(), 0);
        c.i64(0, 1);
        let b = c.buffer(1, BufferRef(3), 0, false);
        c.dispatch_threads(
            Size3 {
                width: 1,
                height: 1,
                depth: 1,
            },
            Size3 {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        b
    };
    assert_eq!(list.buffers(), 4, "the table must reach the largest ref");

    let mut c = list.command(pipeline.clone(), 1);
    let scalar = c.u32(0, 3);
    let buffer = c.buffer(1, BufferRef(9), 0, false);
    c.bytes(2, &[0u8; 32]);
    assert!(c.patch_offset(first_buffer, Affine::constant(0)).is_err());
    assert!(c.patch_offset(scalar, Affine::constant(0)).is_err());
    assert!(c
        .patch_bytes(buffer, 0, Width::U32, Affine::constant(0))
        .is_err());
    assert!(c
        .patch_bytes(scalar, 1, Width::U32, Affine::constant(0))
        .is_err());
    assert!(c
        .patch_bytes(scalar, 0, Width::U64, Affine::constant(0))
        .is_err());
    assert!(c
        .patch_bytes(scalar, 0, Width::U32, Affine::new(0, &[(StepParam(1), 1)]))
        .is_err());
    assert!(c
        .patch_bytes(scalar, 0, Width::U32, Affine::new(0, &[(P0, 1)]))
        .is_ok());
    drop(c);
    assert_eq!(list.len(), 1, "an abandoned command is not in the list");
    let only = &list.commands()[0];
    assert_eq!(list.args(only).len(), 2);
    assert!(list.patches(only).is_empty());
    assert!(matches!(
        list.args(only)[0],
        Arg::Bytes {
            index: 0,
            start: 0,
            len: 8
        }
    ));
    assert_eq!(list.bound(), vec![BufferRef(3)]);
    assert_eq!(
        list.buffers(),
        4,
        "an abandoned command's buffers are not counted"
    );
    assert_eq!(
        list.max_bytes(),
        8,
        "an abandoned command's bytes do not size the scratch"
    );
}

/// A step that cannot be encoded is an error, never a binding of nothing.
#[test]
fn a_missing_buffer_or_parameter_is_an_error() {
    let device = device();
    let kernels = Kernels::new();
    let list = planned_chain(&device, &kernels);
    let cmds = commands(&device);
    let mut encoder = PlanEncoder::new(&list);
    let empty = BindTable::with_len(3);
    // SAFETY: nothing is bound: the first argument lookup fails.
    let missing = unsafe { encoder.encode(&list, &empty, &StepValues::new(0), &cmds) };
    assert!(missing.is_err());

    let mut patched = CommandList::new(2);
    {
        let pipeline = kernels
            .load_pipeline(&device, Source::Unary, unary::copy2d::FLOAT.0)
            .unwrap();
        let mut c = patched.command(pipeline, 0);
        let b = c.buffer(0, BufferRef(0), 0, false);
        c.patch_offset(b, Affine::new(0, &[(P1, -4)])).unwrap();
        c.dispatch_threads(
            Size3 {
                width: 1,
                height: 1,
                depth: 1,
            },
            Size3 {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    let mut table = BindTable::with_len(1);
    table.set(BufferRef(0), upload(&device, &[0.0; 4]));
    let mut encoder = PlanEncoder::new(&patched);
    // SAFETY: each call fails before it dispatches.
    unsafe {
        assert!(encoder
            .encode(&patched, &table, &StepValues::new(1), &cmds)
            .is_err());
        let mut values = StepValues::new(2);
        values.set(P1, 1);
        assert!(
            encoder.encode(&patched, &table, &values, &cmds).is_err(),
            "a negative offset is refused"
        );
        values.set(P1, -100);
        assert!(
            encoder.encode(&patched, &table, &values, &cmds).is_err(),
            "an offset past the binding is refused"
        );
    }
}
