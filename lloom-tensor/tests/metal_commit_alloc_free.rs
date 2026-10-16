#![cfg(feature = "metal")]
//! What a command-buffer commit costs the encoding thread in host
//! allocations (CONTRIBUTING §3.3a, "zero allocations per token";
//! `docs/tuning.md` §13.5x).
//!
//! Counted at libmalloc, below Rust's allocator: `malloc_logger` sees every
//! allocation the thread makes, Rust's and Objective-C's and Metal's. Each
//! counted allocation records its return addresses, and the report names the
//! engine frame that asked and the frame that allocated.
//!
//! Its own binary because it arms the process-wide malloc logger. The count is
//! per thread: only the thread inside a counted window is counted, so the
//! harness's threads and Metal's completion threads cannot move it.

use block2::{RcBlock, StackBlock};
use lloom_tensor::kernels::metal::commands::CommandBufferKind;
use lloom_tensor::kernels::metal::completion::CompletionWait;
use lloom_tensor::kernels::metal::{Buffer, Commands, Device, ResidencySet};
use lloom_tensor::kernels::{call_affine, BufferOffset, Kernels};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLResourceOptions};
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// `MALLOC_LOG_TYPE_ALLOCATE` (`<malloc/malloc.h>`): set on malloc, calloc,
/// realloc and every zone variant.
const LOG_ALLOCATE: u32 = 2;
const FRAMES: usize = 32;
const RECORDED: usize = 1 << 13;

static HOOK: AtomicUsize = AtomicUsize::new(0);
/// `pthread_self` of the thread being counted; 0 counts nothing.
static OWNER: AtomicUsize = AtomicUsize::new(0);
static EVENTS: AtomicU64 = AtomicU64::new(0);
static RECORD: AtomicBool = AtomicBool::new(false);

struct Table(UnsafeCell<[[usize; FRAMES]; RECORDED]>);
// SAFETY: written only by the counted thread inside a window, read by the
// same thread after the window has closed.
unsafe impl Sync for Table {}
static TABLE: Table = Table(UnsafeCell::new([[0; FRAMES]; RECORDED]));

/// One counted window at a time across the binary's tests.
static WINDOW: parking_lot::Mutex<()> = parking_lot::const_mutex(());

extern "C" fn logger(ty: u32, _: usize, _: usize, _: usize, _: usize, _: u32) {
    if ty & LOG_ALLOCATE == 0 {
        return;
    }
    // SAFETY: reads the calling thread's own identity; allocates nothing.
    if OWNER.load(Ordering::Relaxed) != unsafe { libc::pthread_self() } as usize {
        return;
    }
    let n = EVENTS.fetch_add(1, Ordering::Relaxed) as usize;
    if RECORD.load(Ordering::Relaxed) && n < RECORDED {
        // SAFETY: slot `n` is this thread's alone (single counted thread);
        // `backtrace` walks frame pointers and allocates nothing.
        unsafe {
            let slot = &mut (*TABLE.0.get())[n];
            *slot = [0; FRAMES];
            libc::backtrace(slot.as_mut_ptr().cast(), FRAMES as libc::c_int);
        }
    }
}

/// Arm libmalloc's logger with [`logger`], once. Resolved before anything is
/// counted: `dlsym` takes dyld's lock and may allocate.
fn arm() {
    if HOOK.load(Ordering::Relaxed) != 0 {
        return;
    }
    // SAFETY: dlsym of an exported data symbol; null if absent.
    let hook = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"malloc_logger".as_ptr()) };
    assert!(!hook.is_null(), "libmalloc's malloc_logger is not exported");
    // SAFETY: `malloc_logger` is libmalloc's function-pointer variable; a
    // volatile store of a matching C function is how its own tools arm it.
    unsafe { std::ptr::write_volatile(hook as *mut usize, logger as *const () as usize) };
    HOOK.store(hook as usize, Ordering::Relaxed);
}

/// Allocations made on this thread while `f` runs.
fn counted<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = EVENTS.load(Ordering::Relaxed);
    // SAFETY: the calling thread's own identity.
    OWNER.store(unsafe { libc::pthread_self() } as usize, Ordering::Relaxed);
    let r = f();
    OWNER.store(0, Ordering::Relaxed);
    (r, EVENTS.load(Ordering::Relaxed) - before)
}

/// Start recording return addresses from event 0.
fn record_from_zero() {
    EVENTS.store(0, Ordering::Relaxed);
    RECORD.store(true, Ordering::Relaxed);
}

/// The positive control: one allocation in a window must count, or every
/// zero this binary prints is the instrument's.
fn control() -> u64 {
    let ((), n) = counted(|| drop(std::hint::black_box(vec![0u8; 64])));
    assert!(
        n >= 1,
        "instrument blind: {n} events for one Vec allocation"
    );
    n
}

fn symbol(pc: usize) -> Option<String> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr writes `info` for any address; a null name is checked.
    if unsafe { libc::dladdr(pc as *const libc::c_void, &mut info) } == 0
        || info.dli_sname.is_null()
    {
        return None;
    }
    // SAFETY: dladdr returned a NUL-terminated name in a loaded image.
    let raw = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) }.to_string_lossy();
    Some(demangle(&raw))
}

/// Rust's legacy mangling (`_ZN` + length-prefixed path + hash) to a path;
/// anything else verbatim.
fn demangle(s: &str) -> String {
    let Some(mut rest) = s.strip_prefix("_ZN").or_else(|| s.strip_prefix("__ZN")) else {
        return s.to_string();
    };
    let mut parts = Vec::new();
    while let Some(c) = rest.chars().next() {
        if !c.is_ascii_digit() {
            break;
        }
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        let len: usize = rest[..digits].parse().unwrap_or(0);
        rest = &rest[digits..];
        if len > rest.len() {
            break;
        }
        let part = &rest[..len];
        rest = &rest[len..];
        if !(part.len() == 17 && part.starts_with('h')) {
            parts.push(
                part.replace("$LT$", "<")
                    .replace("$GT$", ">")
                    .replace("$u20$", " ")
                    .replace("$C$", ",")
                    .replace("$RF$", "&")
                    .replace("..", "::"),
            );
        }
    }
    parts.join("::")
}

/// The recorded events of the last window, grouped by (engine frame that
/// asked, frame that allocated). Frames inside libmalloc, `backtrace` and
/// this file's logger are skipped.
fn sites(events: usize) -> BTreeMap<(String, String), (u64, Vec<String>)> {
    let mut out: BTreeMap<(String, String), (u64, Vec<String>)> = BTreeMap::new();
    let n = events.min(RECORDED);
    for i in 0..n {
        // SAFETY: recording is off and this thread wrote every slot read.
        let frames = unsafe { (*TABLE.0.get())[i] };
        let syms: Vec<String> = frames
            .iter()
            .take_while(|&&pc| pc != 0)
            .filter_map(|&pc| symbol(pc))
            .collect();
        let skip = |s: &str| {
            s.contains("logger")
                || s.starts_with("backtrace")
                || s.contains("malloc")
                || s.contains("calloc")
                || s.contains("realloc")
                || s.starts_with("_malloc")
                || s.starts_with("nanov2")
                || s.starts_with("szone")
                || s.starts_with("xzm")
                || s.contains("__rdl_")
                || s.contains("__rust_alloc")
                || s.contains("alloc::alloc::")
                || s.contains("System as core::alloc")
        };
        let leaf = syms
            .iter()
            .find(|s| !skip(s))
            .cloned()
            .unwrap_or_else(|| "?".into());
        let site = syms
            .iter()
            .find(|s| s.contains("lloom_tensor") && !skip(s))
            .cloned()
            .unwrap_or_else(|| "?".into());
        let chain: Vec<String> = syms.iter().filter(|s| !skip(s)).take(8).cloned().collect();
        out.entry((site, leaf)).or_insert((0, chain)).0 += 1;
    }
    out
}

/// The frames above libmalloc and the logger, as `dladdr` names them.
fn callers_of_malloc(frames: &[usize; FRAMES]) -> Vec<String> {
    frames
        .iter()
        .take_while(|&&pc| pc != 0)
        .filter_map(|&pc| raw_symbol(pc))
        .skip_while(|s| {
            s.contains("logger")
                || s.starts_with("backtrace")
                || s.contains("malloc")
                || s.contains("calloc")
                || s.contains("realloc")
                || s.starts_with("nanov2")
                || s.starts_with("szone")
                || s.starts_with("xzm")
        })
        .collect()
}

/// Recorded events the engine made: Rust code allocated (its own and its
/// crates'), or Rust asked the blocks runtime to copy a block to the heap
/// (`RcBlock::new`) — not Metal's or the runtimes' own allocations. Keyed on
/// the frames above libmalloc because Rust's allocator shims (`__rust_alloc`,
/// `__rdl_alloc`) tail-call `malloc` and leave no frame.
fn our_allocations(events: usize) -> u64 {
    let mut n = 0;
    for i in 0..events.min(RECORDED) {
        // SAFETY: recording is off and this thread wrote every slot read.
        let frames = unsafe { (*TABLE.0.get())[i] };
        let callers = callers_of_malloc(&frames);
        let ours = match callers.first() {
            Some(first) if is_rust(first) => true,
            Some(first) if first == "_Block_copy" => callers.get(1).is_some_and(|s| is_rust(s)),
            _ => false,
        };
        n += ours as u64;
    }
    n
}

/// A Rust symbol as `dladdr` names it: v0 (`_R…`), or legacy (`_ZN…`)
/// ending in its hash segment (`17h` + 16 hex + `E`), which C++'s Itanium
/// `_ZN…` names do not carry.
fn is_rust(raw: &str) -> bool {
    if raw.starts_with("_R") {
        return true;
    }
    let b = raw.as_bytes();
    raw.starts_with("_ZN")
        && b.len() > 20
        && b[b.len() - 1] == b'E'
        && &raw[b.len() - 20..b.len() - 17] == "17h"
        && b[b.len() - 17..b.len() - 1]
            .iter()
            .all(u8::is_ascii_hexdigit)
}

fn raw_symbol(pc: usize) -> Option<String> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr writes `info` for any address; a null name is checked.
    if unsafe { libc::dladdr(pc as *const libc::c_void, &mut info) } == 0
        || info.dli_sname.is_null()
    {
        return None;
    }
    // SAFETY: dladdr returned a NUL-terminated name in a loaded image.
    Some(
        unsafe { std::ffi::CStr::from_ptr(info.dli_sname) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// The classifier's controls: a `Vec` allocation and an `RcBlock` are ours;
/// `strdup`'s, made by libc on Rust's behalf, is not. All must read as such,
/// or `ours=` means nothing.
fn classifier_control() {
    record_from_zero();
    let ((), n) = counted(|| drop(std::hint::black_box(vec![0u8; 64])));
    RECORD.store(false, Ordering::Relaxed);
    assert_eq!(
        our_allocations(n as usize),
        n,
        "a Vec allocation did not read as ours"
    );
    record_from_zero();
    let ((), n) = counted(|| {
        drop(std::hint::black_box(RcBlock::new(
            |_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {},
        )))
    });
    RECORD.store(false, Ordering::Relaxed);
    assert!(n >= 1, "RcBlock::new allocated nothing");
    assert_eq!(
        our_allocations(n as usize),
        n,
        "an RcBlock did not read as ours"
    );
    record_from_zero();
    // SAFETY: a NUL-terminated literal; the copy is freed below.
    let ((), n) = counted(|| unsafe {
        libc::free(std::hint::black_box(libc::strdup(c"control".as_ptr())).cast())
    });
    RECORD.store(false, Ordering::Relaxed);
    assert_eq!(n, 1, "strdup made {n} allocations");
    assert_eq!(
        our_allocations(n as usize),
        0,
        "libc's allocation read as ours"
    );
}

fn print_sites(label: &str, events: u64, per: u64) -> u64 {
    let ours = our_allocations(events as usize);
    let sites = sites(events as usize);
    println!(
        "SITES {label} events={events} recorded={} ours={ours} ours_per_commit={:.2}",
        events.min(RECORDED as u64),
        ours as f64 / per as f64
    );
    for ((site, leaf), (n, chain)) in sites {
        println!(
            "  {:>8.2}/commit  {n:>6}  site={site}  leaf={leaf}",
            n as f64 / per as f64
        );
        println!("           chain: {}", chain.join(" <- "));
    }
    ours
}

fn api() -> String {
    lloom_tensor::tuning::tuning().metal.command_api.clone()
}

/// **Metal's copy of a heap block is a retain, not an allocation.**
///
/// `addCompletedHandler:` copies the block it is given (`Block_copy`). A heap
/// block (`RcBlock`, flagged `BLOCK_NEEDS_FREE`) is copied by incrementing its
/// reference count; a stack block is copied to the heap, which allocates. The
/// stack-block arm is the control: it shows the window can see the copy, so a
/// lower count on the heap arm is the copy not allocating, not a blind window.
/// What remains on the heap arm is Metal's own storage for the handler.
#[test]
fn metal_copies_a_heap_block_without_allocating_it() {
    let _w = WINDOW.lock();
    arm();
    let control = control();
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().unwrap();
    // Below Metal's default 64 outstanding command buffers per queue (an
    // uncommitted buffer counts): `commandBuffer` blocks at the limit.
    let n = 16;
    let fresh = |q: &objc2::rc::Retained<ProtocolObject<dyn MTLCommandQueue>>| {
        (0..n)
            .map(|_| q.commandBuffer().expect("command buffer"))
            .collect::<Vec<_>>()
    };
    let heap = RcBlock::new(|_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {});
    // Warm Metal's lazily created per-queue state on one handler first.
    let warm = queue.commandBuffer().unwrap();
    // SAFETY: a valid block for the handler's signature, copied by Metal.
    unsafe { warm.addCompletedHandler(RcBlock::as_ptr(&heap)) };

    let cbs = fresh(&queue);
    let ((), heap_events) = counted(|| {
        for cb in &cbs {
            // SAFETY: as above.
            unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&heap)) };
        }
    });
    let cbs_stack = fresh(&queue);
    let ((), stack_events) = counted(|| {
        for cb in &cbs_stack {
            let stack = StackBlock::new(|_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {});
            // SAFETY: Metal copies the stack block before this returns.
            unsafe { cb.addCompletedHandler(&*stack as *const _ as *mut _) };
        }
    });
    // A second handler on a buffer that already has one: Metal's storage.
    let ((), second_events) = counted(|| {
        for cb in &cbs {
            // SAFETY: as above.
            unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&heap)) };
        }
    });
    println!(
        "BLOCK_COPY control={control} handlers={n} heap_block_events={heap_events} \
         stack_block_events={stack_events} second_handler_events={second_events}"
    );
    assert!(
        stack_events >= heap_events + n as u64,
        "the stack-block control allocated no more than the heap block: \
         the window cannot see a block copy (heap {heap_events}, stack {stack_events})"
    );
    drop((cbs, cbs_stack, warm));
}

/// One dispatch per command buffer, cut and committed. `flush` is the commit
/// the decode path makes at every cut (`commit_swap_locked`): close the
/// encoder, publish residency, register completion, commit, recover a
/// recording buffer.
fn commit_loop(kind: CommandBufferKind, subscriber: bool, commits: u64) -> (u64, u64) {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let cmds = Commands::new_with(queue, residency_set, CompletionWait::Synchronize, kind).unwrap();
    run_commits(&device, &cmds, subscriber, commits)
}

/// As [`commit_loop`], on the queue a `MetalDevice` builds: the one
/// production decodes on, whose every command buffer is bound to a binding
/// generation that stages its epoch at commit and refills its credit after.
fn bound_commit_loop(commits: u64) -> (u64, u64, CommandBufferKind) {
    let gpu = lloom_tensor::Device::new_metal(0).expect("no Metal device");
    let metal = gpu.as_metal_device().expect("a Metal device");
    let cmds = metal.commands();
    let (events, encode) = run_commits(metal.metal_device(), cmds, false, commits);
    (events, encode, cmds.kind())
}

fn run_commits(device: &Device, cmds: &Commands, subscriber: bool, commits: u64) -> (u64, u64) {
    // Production registers no subscriber (the pools drain on the host); the
    // other arm registers one, to price the subscriber path itself.
    let fired = Arc::new(AtomicU64::new(0));
    if subscriber {
        let f = Arc::clone(&fired);
        cmds.on_command_buffer_complete(move || {
            f.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    }
    let kernels = Kernels::new();
    let n = 256usize;
    let len = n * std::mem::size_of::<f32>();
    let src: Buffer = device.new_buffer(len, SHARED).unwrap();
    let dst: Buffer = device.new_buffer(len, SHARED).unwrap();
    let step = || {
        let encoder = cmds.command_encoder().unwrap();
        // SAFETY: the loop owns `src` and `dst`, binds them to every command
        // buffer it commits, and drains every buffer before it returns.
        unsafe {
            call_affine(
                device,
                &encoder,
                &kernels,
                "affine_f32",
                std::mem::size_of::<f32>(),
                n,
                BufferOffset::zero_offset(&src),
                &dst,
                2.0,
                1.0,
            )
        }
        .unwrap();
    };
    for i in 0..128 {
        step();
        cmds.flush().unwrap();
        if i % 8 == 7 {
            cmds.flush_and_wait_current().unwrap();
        }
    }
    // Drained the way a decode drains, by a wait that prunes in place
    // (`flush_and_wait_current`), outside the counted windows.
    let mut events = 0;
    record_from_zero();
    for i in 0..commits {
        step();
        let ((), e) = counted(|| cmds.flush().unwrap());
        events += e;
        if i % 8 == 7 {
            cmds.flush_and_wait_current().unwrap();
        }
    }
    RECORD.store(false, Ordering::Relaxed);
    let recorded_events = events;
    cmds.flush_and_wait().unwrap();
    // Encoding one dispatch, for scale: the same counter around a step.
    let mut encode = 0;
    for _ in 0..64 {
        let ((), e) = counted(step);
        encode += e;
        cmds.flush_and_wait().unwrap();
    }
    if subscriber {
        assert!(
            fired.load(Ordering::Relaxed) >= commits,
            "the subscriber ran {} times over {commits} commits",
            fired.load(Ordering::Relaxed)
        );
    }
    (recorded_events, encode / 64)
}

/// Host allocations per commit on the encoding thread, per command-buffer
/// kind, with every allocation attributed. Reports; asserts the instrument.
#[test]
fn allocations_per_commit_on_the_encoding_thread() {
    let _w = WINDOW.lock();
    arm();
    let control = control();
    classifier_control();
    let commits = 256;
    for kind in [CommandBufferKind::Retaining, CommandBufferKind::Unretained] {
        for subscriber in [false, true] {
            let (events, encode) = commit_loop(kind, subscriber, commits);
            let api = api();
            let subscribers = subscriber as u8;
            println!(
                "COMMIT api={api} kind={kind:?} subscribers={subscribers} control={control} \
                 commits={commits} allocations={events} per_commit={:.2} \
                 encode_per_dispatch={encode}",
                events as f64 / commits as f64
            );
            let ours = print_sites(
                &format!("api={api} kind={kind:?} subscribers={subscribers}"),
                events,
                commits,
            );
            // The default command API's commit makes no allocation of ours,
            // with or without a subscriber: no callback box, no capture
            // vector, no subscriber clone, no per-buffer block
            // (`docs/tuning.md` §13.5x). Metal 4 is reported: its commit
            // still takes a new submission, commit options and a block.
            if api == "metal3" {
                assert_eq!(
                    ours, 0,
                    "{ours} allocations of ours over {commits} commits (kind {kind:?}, \
                     subscribers {subscribers}): see the SITES table above"
                );
            }
        }
    }
    // The bound queue production runs, in the kind the process selects
    // (`LLOOM_CB_UNRETAINED`): the same zero, with the generation's staging and
    // its credit refill inside the window.
    let (events, encode, kind) = bound_commit_loop(commits);
    let api = api();
    println!(
        "COMMIT api={api} kind={kind:?} bound=1 subscribers=0 control={control} \
         commits={commits} allocations={events} per_commit={:.2} \
         encode_per_dispatch={encode}",
        events as f64 / commits as f64
    );
    let ours = print_sites(
        &format!("api={api} kind={kind:?} bound=1 subscribers=0"),
        events,
        commits,
    );
    if api == "metal3" {
        assert_eq!(
            ours, 0,
            "{ours} allocations of ours over {commits} bound commits (kind {kind:?}): \
             see the SITES table above"
        );
    }
}
