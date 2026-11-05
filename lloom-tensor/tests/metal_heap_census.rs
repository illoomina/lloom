#![cfg(feature = "metal")]
//! #1235 step 0: every heap event in the process while the production queue
//! commits, not only the encoding thread's (`metal_commit_alloc_free` counts
//! that one thread). For each allocation: its size, the thread that made it
//! and the thread that freed it; and every VM-level allocation the kernel
//! logs (`__syscall_logger`: `mmap`, `munmap`, `vm_allocate`,
//! `vm_deallocate`). This grounds a malloc zone's size classes, its
//! cross-thread free path and the "no syscall in steady state" baseline
//! libmalloc is held against. A measurement, not a gate: it asserts only its
//! own instrument.
//!
//! Its own binary because it arms the process-wide malloc and syscall
//! loggers. The loggers allocate nothing: each event takes one slot of a
//! static table by an atomic index and writes plain words into it; the table
//! is read after the window closes.

use lloom_tensor::kernels::metal::{Buffer, Commands, Device};
use lloom_tensor::kernels::{call_affine, BufferOffset, Kernels};
use objc2_metal::MTLResourceOptions;
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// `<malloc/malloc.h>` / libmalloc's stack-logging types: the bits a logger
/// is called with.
const ALLOCATE: u32 = 2;
const DEALLOCATE: u32 = 4;
const VM_ALLOCATE: u32 = 0x10;
const VM_DEALLOCATE: u32 = 0x20;
const CAPACITY: usize = 1 << 18;

#[derive(Clone, Copy, Default)]
struct Event {
    ty: u32,
    a1: usize,
    a2: usize,
    a3: usize,
    result: usize,
    thread: usize,
    /// Which logger: 0 malloc, 1 syscall.
    source: u8,
}

struct Table(UnsafeCell<[Event; CAPACITY]>);
// SAFETY: slot `n` is written by the one logger call that took index `n`,
// and read only after recording stopped and the writers' stores are ordered
// before the reader by the `RECORDING` release/acquire pair and a fence.
unsafe impl Sync for Table {}
static TABLE: Table = Table(UnsafeCell::new(
    [Event {
        ty: 0,
        a1: 0,
        a2: 0,
        a3: 0,
        result: 0,
        thread: 0,
        source: 0,
    }; CAPACITY],
));
static NEXT: AtomicUsize = AtomicUsize::new(0);
static DROPPED: AtomicUsize = AtomicUsize::new(0);
static RECORDING: AtomicBool = AtomicBool::new(false);
static ARMED: AtomicBool = AtomicBool::new(false);

fn record(source: u8, ty: u32, a1: usize, a2: usize, a3: usize, result: usize) {
    if !RECORDING.load(Ordering::Acquire) {
        return;
    }
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    if n >= CAPACITY {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    // SAFETY: index `n` is this call's alone; plain stores, no allocation.
    unsafe {
        (*TABLE.0.get())[n] = Event {
            ty,
            a1,
            a2,
            a3,
            result,
            thread: libc::pthread_self() as usize,
            source,
        };
    }
}

extern "C" fn malloc_logger(ty: u32, a1: usize, a2: usize, a3: usize, result: usize, _: u32) {
    record(0, ty, a1, a2, a3, result);
}
extern "C" fn syscall_logger(ty: u32, a1: usize, a2: usize, a3: usize, result: usize, _: u32) {
    record(1, ty, a1, a2, a3, result);
}

/// Arm both loggers once, before anything is recorded: `dlsym` may allocate.
fn arm() {
    if ARMED.swap(true, Ordering::AcqRel) {
        return;
    }
    for (name, f) in [
        (c"malloc_logger", malloc_logger as *const () as usize),
        (c"__syscall_logger", syscall_logger as *const () as usize),
    ] {
        // SAFETY: dlsym of an exported data symbol; null if absent.
        let hook = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        assert!(!hook.is_null(), "{name:?} is not exported");
        // SAFETY: both are libmalloc's / libsystem's function-pointer
        // variables; a volatile store of a matching C function is how their
        // own tools arm them.
        unsafe { std::ptr::write_volatile(hook as *mut usize, f) };
    }
}

/// The events recorded while `f` ran, every thread's.
fn window<R>(f: impl FnOnce() -> R) -> (R, Vec<Event>, usize) {
    NEXT.store(0, Ordering::Relaxed);
    DROPPED.store(0, Ordering::Relaxed);
    RECORDING.store(true, Ordering::Release);
    let r = f();
    RECORDING.store(false, Ordering::Release);
    // Let a logger call already past its `RECORDING` check finish its store.
    std::thread::sleep(std::time::Duration::from_millis(50));
    std::sync::atomic::fence(Ordering::SeqCst);
    let n = NEXT.load(Ordering::Relaxed).min(CAPACITY);
    // SAFETY: recording stopped; the slots below `n` are written.
    let events = unsafe { (&(*TABLE.0.get()))[..n].to_vec() };
    (r, events, DROPPED.load(Ordering::Relaxed))
}

/// The size class a zone might serve `size` from: 16 B steps to 128, then
/// four classes per power of two, then "large".
fn class(size: usize) -> String {
    if size <= 128 {
        return format!("{:>5}", size.div_ceil(16).max(1) * 16);
    }
    if size > 32 << 10 {
        return "large".into();
    }
    let p = usize::BITS - 1 - (size - 1).leading_zeros(); // floor(log2(size-1))
    let base = 1usize << p;
    let step = base / 4;
    format!("{:>5}", base + (size - base).div_ceil(step) * step)
}

struct Report {
    allocations: usize,
    frees: usize,
    reallocs: usize,
    by_thread: BTreeMap<String, (usize, usize)>,
    classes: BTreeMap<String, usize>,
    cross_thread_frees: usize,
    frees_of_earlier: usize,
    vm: BTreeMap<String, (usize, usize)>,
    /// Allocations per malloc zone (`a1`, the zone pointer).
    zones: BTreeMap<usize, usize>,
}

fn analyse(events: &[Event], encoder: usize) -> Report {
    let role = |t: usize| {
        if t == encoder {
            "encoding".to_string()
        } else {
            format!("other-{t:x}")
        }
    };
    let mut live: HashMap<usize, usize> = HashMap::new(); // ptr -> allocating thread
    let mut r = Report {
        allocations: 0,
        frees: 0,
        reallocs: 0,
        by_thread: BTreeMap::new(),
        classes: BTreeMap::new(),
        cross_thread_frees: 0,
        frees_of_earlier: 0,
        vm: BTreeMap::new(),
        zones: BTreeMap::new(),
    };
    for e in events {
        if e.source == 1 {
            let v = r.vm.entry(role(e.thread)).or_default();
            if e.ty & VM_ALLOCATE != 0 {
                v.0 += 1;
            }
            if e.ty & VM_DEALLOCATE != 0 {
                v.1 += 1;
            }
            continue;
        }
        let alloc = e.ty & ALLOCATE != 0;
        let free = e.ty & DEALLOCATE != 0;
        if alloc {
            *r.zones.entry(e.a1).or_default() += 1;
        }
        fn freed(live: &mut HashMap<usize, usize>, ptr: usize, thread: usize, r: &mut Report) {
            match live.remove(&ptr) {
                Some(owner) if owner != thread => r.cross_thread_frees += 1,
                Some(_) => {}
                None => r.frees_of_earlier += 1,
            }
        }
        match (alloc, free) {
            (true, true) => {
                // realloc: a2 = old pointer, a3 = new size, result = new pointer
                r.reallocs += 1;
                if e.a2 != 0 {
                    freed(&mut live, e.a2, e.thread, &mut r);
                }
                live.insert(e.result, e.thread);
                *r.classes.entry(class(e.a3)).or_default() += 1;
                r.by_thread.entry(role(e.thread)).or_default().0 += 1;
            }
            (true, false) => {
                // malloc / calloc / valloc / memalign: a2 = size, result = pointer
                r.allocations += 1;
                live.insert(e.result, e.thread);
                *r.classes.entry(class(e.a2)).or_default() += 1;
                r.by_thread.entry(role(e.thread)).or_default().0 += 1;
            }
            (false, true) => {
                // free: a2 = pointer
                r.frees += 1;
                freed(&mut live, e.a2, e.thread, &mut r);
                r.by_thread.entry(role(e.thread)).or_default().1 += 1;
            }
            (false, false) => {}
        }
    }
    r
}

fn print(label: &str, r: &Report, commits: usize, dropped: usize) {
    let per = |n: usize| n as f64 / commits as f64;
    println!(
        "HEAP {label} commits={commits} allocations={} ({:.2}/commit) reallocs={} frees={} ({:.2}/commit) \
         cross_thread_frees={} ({:.2}/commit) frees_of_earlier={} dropped={dropped}",
        r.allocations,
        per(r.allocations),
        r.reallocs,
        r.frees,
        per(r.frees),
        r.cross_thread_frees,
        per(r.cross_thread_frees),
        r.frees_of_earlier
    );
    for (t, (a, f)) in &r.by_thread {
        println!("  THREAD {t:<24} allocs={a:>7} ({:.2}/commit) frees={f:>7} ({:.2}/commit)", per(*a), per(*f));
    }
    for (c, n) in &r.classes {
        println!("  CLASS {c} B  {n:>7}  ({:.2}/commit)", per(*n));
    }
    for (z, n) in &r.zones {
        println!("  ZONE {z:#x}  allocations={n}");
    }
    if r.vm.is_empty() {
        println!("  VM none");
    }
    for (t, (a, d)) in &r.vm {
        println!("  VM {t:<24} vm_allocate={a} ({:.3}/commit) vm_deallocate={d}", per(*a));
    }
}

/// The instrument's controls, asserted: a Vec allocation is seen with its
/// size and freed on its own thread; a Vec freed on another thread is a
/// cross-thread free; an `mmap` is a VM event.
fn controls() {
    let (_, events, dropped) = window(|| drop(std::hint::black_box(vec![0u8; 100])));
    assert_eq!(dropped, 0);
    let me = unsafe { libc::pthread_self() } as usize;
    let r = analyse(&events, me);
    assert!(r.allocations >= 1 && r.classes.contains_key(&class(100)), "blind to a Vec");
    assert_eq!(r.cross_thread_frees, 0);
    let (_, events, _) = window(|| {
        let v = std::hint::black_box(vec![0u8; 100]);
        std::thread::scope(|s| {
            s.spawn(move || drop(v));
        });
    });
    let r = analyse(&events, me);
    assert!(r.cross_thread_frees >= 1, "blind to a cross-thread free");
    let (_, events, _) = window(|| unsafe {
        let p = libc::mmap(
            std::ptr::null_mut(),
            1 << 20,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        );
        assert_ne!(p, libc::MAP_FAILED);
        libc::munmap(p, 1 << 20);
    });
    let r = analyse(&events, me);
    let vm: usize = r.vm.values().map(|(a, d)| a + d).sum();
    println!("HEAP_CONTROLS vec_class={} cross_thread=ok mmap_vm_events={vm}", class(100));
    assert!(vm >= 1, "blind to an mmap: __syscall_logger saw nothing");
}

#[test]
fn heap_events_per_commit_on_every_thread() {
    arm();
    controls();
    let gpu = lloom_tensor::Device::new_metal(0).expect("no Metal device");
    let metal = gpu.as_metal_device().expect("a Metal device");
    let cmds: &Commands = metal.commands();
    let device: &Device = metal.metal_device();
    let api = lloom_tensor::tuning::tuning().metal.command_api.clone();
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
    // Warm, then the steady state: the shape `metal_commit_alloc_free`'s
    // bound loop counts on the encoding thread alone.
    for i in 0..128 {
        step();
        cmds.flush().unwrap();
        if i % 8 == 7 {
            cmds.flush_and_wait_current().unwrap();
        }
    }
    cmds.flush_and_wait().unwrap();
    let commits = 256usize;
    let encoder = unsafe { libc::pthread_self() } as usize;
    let ((), events, dropped) = window(|| {
        for i in 0..commits {
            step();
            cmds.flush().unwrap();
            if i % 8 == 7 {
                cmds.flush_and_wait_current().unwrap();
            }
        }
        cmds.flush_and_wait().unwrap();
    });
    let r = analyse(&events, encoder);
    print(&format!("api={api} kind={:?} bound=1", cmds.kind()), &r, commits, dropped);
    assert_eq!(dropped, 0, "the table overflowed: raise CAPACITY");
}
