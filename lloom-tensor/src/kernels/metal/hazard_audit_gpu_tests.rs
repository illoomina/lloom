//! The recorder observes what the engine actually encodes.
//!
//! `hazard_audit_tests` exercises the cover rules on synthetic streams. These
//! drive real `Commands` passes and assert that the recorder saw the fence
//! updates and waits the engine encoded, at the passes and positions it
//! encoded them, and that the verdict follows from them. The cross-pass
//! reader case is the design's counterexample on a stream the engine produces
//! today: the overwrite waits on the last writer's fence and on no reader's
//! (`docs/dependency-ordered-reuse-design.md` §2).

#![cfg(feature = "hazard-audit")]

use crate::kernels::metal::encoder::HazardKind;
use crate::kernels::metal::hazard_audit::{self, SeamPolicy, Stages, SyncKind};
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{call_unary_contiguous, unary, Buffer, BufferOffset, Device, Kernels};
use std::sync::Arc;

const N: usize = 256;

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

struct Fixture {
    device: Device,
    kernels: Kernels,
    cmds: Commands,
}

impl Fixture {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let cmds = commands(&device);
        Some(Fixture {
            device,
            kernels: Kernels::new(),
            cmds,
        })
    }

    fn buffer(&self) -> Buffer {
        self.device
            .new_buffer(N * 4, crate::kernels::RESOURCE_OPTIONS)
            .unwrap()
    }

    /// One dispatch reading `src` and writing `dst` in the open compute pass.
    fn cos(&self, src: &Buffer, dst: &Buffer) {
        let guard = self.cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound here and reads nothing back
        // before `flush_and_wait`.
        unsafe {
            call_unary_contiguous(
                &self.device,
                &guard,
                &self.kernels,
                unary::contiguous::cos::FLOAT,
                4,
                N,
                BufferOffset::zero_offset(src),
                dst,
            )
        }
        .unwrap();
    }

    /// End the open compute pass without committing, so its outputs stay
    /// registered against its fence while later passes bind them.
    fn end_pass(&self) {
        let _blit = self.cmds.blit_command_encoder().unwrap();
    }
}

fn uncovered(rep: &hazard_audit::AuditReport) -> Vec<(u64, u64, HazardKind)> {
    rep.uncovered
        .iter()
        .map(|e| (e.writer_seq, e.reader_seq, e.kind))
        .collect()
}

/// Every recorded command's pass was seen opening on a known command buffer.
fn assert_passes_identified(stream: &hazard_audit::AuditStream) {
    for d in &stream.dispatches {
        assert!(
            stream.syncs.iter().any(|s| matches!(
                s.kind,
                SyncKind::Open { encoder, command_buffer: Some(_) } if encoder == d.encoder
            )),
            "pass {} of command #{} was never seen opening",
            d.encoder,
            d.seq
        );
    }
}

/// Pass 0 writes X; pass 1 reads X; pass 2 overwrites X. Pass 2 binds X as an
/// output, finds pass 0 as X's last writer and waits on its fence only.
#[test]
fn the_overwrite_after_a_cross_pass_reader_is_reported() {
    let Some(f) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (a, x, y, b) = (f.buffer(), f.buffer(), f.buffer(), f.buffer());

    hazard_audit::begin("cross-pass-reader");
    f.cos(&a, &x);
    f.end_pass();
    f.cos(&x, &y);
    f.end_pass();
    f.cos(&b, &x);
    f.cmds.flush_and_wait().unwrap();
    let region = hazard_audit::finish_recording(&[]);
    let rep = region.stream.cover(&region.region, SeamPolicy::Observed);
    eprintln!("{}", rep.render());

    let stages: Vec<Stages> = region.stream.dispatches.iter().map(|d| d.stage).collect();
    assert_eq!(stages, vec![Stages::DISPATCH; 3], "three compute commands");
    assert_eq!(region.commands_compute_passes, 3);
    assert_eq!(rep.passes, 5, "three compute passes and two blit passes");
    assert_passes_identified(&region.stream);
    // One update at every pass close. Waits: pass 1 on X's writer, pass 2 on
    // X's writer, blit pass 0 on pass 0, blit pass 1 on passes 0 and 1.
    assert_eq!(rep.fence_updates, 5, "{:?}", region.stream.syncs);
    assert_eq!(rep.fence_waits, 5, "{:?}", region.stream.syncs);
    assert_eq!(rep.unmatched_fence_waits, 0);
    assert_eq!(rep.same_pass_fence_waits, 0);

    assert_eq!((rep.edges_raw, rep.edges_waw, rep.edges_war), (1, 1, 1));
    assert_eq!(rep.by_fence, 2, "RAW and WAW against pass 0's fence");
    assert_eq!(
        uncovered(&rep),
        vec![(1, 2, HazardKind::War)],
        "the overwrite was taken as ordered after a reader it never waited on"
    );
    assert_ne!(
        rep.uncovered[0].writer_encoder,
        rep.uncovered[0].reader_encoder
    );
}

/// The same stream with pass 2 waiting on the reader's fence before the
/// overwrite is clean; the same wait after the overwrite is not.
#[test]
fn a_wait_on_the_readers_fence_before_the_overwrite_covers_it() {
    for wait_first in [true, false] {
        let Some(f) = Fixture::new() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let (a, x, y, b) = (f.buffer(), f.buffer(), f.buffer(), f.buffer());

        hazard_audit::begin("reader-fence");
        f.cos(&a, &x);
        f.end_pass();
        let reader_fence = {
            f.cos(&x, &y);
            let guard = f.cmds.command_encoder().unwrap();
            let fence = Arc::clone(&guard.as_ref().fence);
            fence
        };
        f.end_pass();
        {
            let guard = f.cmds.command_encoder().unwrap();
            if wait_first {
                guard.as_ref().wait_for_fence(&reader_fence);
            }
        }
        f.cos(&b, &x);
        if !wait_first {
            let guard = f.cmds.command_encoder().unwrap();
            guard.as_ref().wait_for_fence(&reader_fence);
        }
        f.cmds.flush_and_wait().unwrap();
        let region = hazard_audit::finish_recording(&[]);
        let rep = region.stream.cover(&region.region, SeamPolicy::Observed);
        eprintln!("wait_first={wait_first}\n{}", rep.render());

        assert_eq!(rep.fence_waits, 6, "{:?}", region.stream.syncs);
        if wait_first {
            assert!(rep.is_clean(), "{}", rep.render());
            assert_eq!(rep.by_fence, 3);
        } else {
            assert_eq!(
                uncovered(&rep),
                vec![(1, 2, HazardKind::War)],
                "a wait encoded after the overwrite was credited"
            );
        }
    }
}

/// A pass that reads X and then overwrites it: the pass barrier orders its own
/// read before its write, and nothing orders the earlier pass's read.
#[test]
fn a_read_to_write_upgrade_is_reported_against_the_earlier_reader() {
    let Some(f) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (a, x, y, z, b) = (f.buffer(), f.buffer(), f.buffer(), f.buffer(), f.buffer());

    hazard_audit::begin("upgrade");
    f.cos(&a, &x);
    f.end_pass();
    f.cos(&x, &y);
    f.end_pass();
    f.cos(&x, &z);
    f.cos(&b, &x);
    f.cmds.flush_and_wait().unwrap();
    let region = hazard_audit::finish_recording(&[]);
    let rep = region.stream.cover(&region.region, SeamPolicy::Observed);
    eprintln!("{}", rep.render());

    let d = &region.stream.dispatches;
    assert_eq!(d.len(), 4);
    assert_eq!(
        d[2].encoder, d[3].encoder,
        "the read and the write share a pass"
    );
    assert!(
        d[3].barrier,
        "the engine emitted no barrier before the in-pass write"
    );
    assert!(d[3].kinds.contains(HazardKind::War));
    assert_eq!(rep.by_barrier, 1, "the in-pass WAR is the barrier's");
    assert_eq!(
        uncovered(&rep),
        vec![(1, 3, HazardKind::War)],
        "the earlier pass's reader was taken as ordered before the write"
    );
}

/// A blit is observed as a blit-stage command, and the blit pass's fence wait
/// and update order it against the compute passes on each side.
#[test]
fn a_blit_between_compute_passes_is_observed_and_ordered() {
    let Some(f) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (a, x, y, z) = (f.buffer(), f.buffer(), f.buffer(), f.buffer());

    hazard_audit::begin("blit");
    f.cos(&a, &x);
    {
        let mut blit = f.cmds.blit_command_encoder().unwrap();
        // SAFETY: the test owns both buffers and reads nothing back before
        // `flush_and_wait`.
        unsafe { blit.copy_from_buffer(&x, 0, &y, 0, N * 4) };
    }
    f.cos(&y, &z);
    f.cmds.flush_and_wait().unwrap();
    let region = hazard_audit::finish_recording(&[]);
    let rep = region.stream.cover(&region.region, SeamPolicy::Observed);
    eprintln!("{}", rep.render());

    let stages: Vec<Stages> = region.stream.dispatches.iter().map(|d| d.stage).collect();
    assert_eq!(
        stages,
        vec![Stages::DISPATCH, Stages::BLIT, Stages::DISPATCH]
    );
    assert_eq!(
        rep.edges_raw, 2,
        "compute -> blit on X, blit -> compute on Y"
    );
    assert!(rep.is_clean(), "{}", rep.render());
    assert_eq!(rep.by_fence, 2);
}
