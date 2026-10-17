//! The terminal boundaries of one decode run, for the diagnostic arm
//! (`design/measurements/issue-993.md` §14): when the device published the
//! stop, when the host saw it (the streaming reader and the producer apart),
//! when the run's last work was submitted and when it retired, and when the
//! state was ready for the next request.
//!
//! Compiled only in a debug build or with `profile`, and recorded only under
//! `CANDLE_METAL_PROFILE=1` on Metal: the shipping binary carries none of it,
//! and an unprofiled run pays one cached-bool check per call site. Count
//! snapshots are lazy: neither the cfg-disabled nor runtime-disabled path
//! invokes the closure that polls the publication counter. Times are
//! `profile::now_s` seconds (`CLOCK_UPTIME_RAW`), the clock the command
//! buffers' GPU times are in. A run is resolved against the profile
//! recorder's command-buffer records once the buffers it needs have
//! completed — never by waiting for them, so the diagnostic adds no wait to
//! the run it observes. A resolved run is printed as one `TERMINAL-TRACE`
//! JSON line on stderr.

#[cfg(all(feature = "metal", any(debug_assertions, feature = "profile")))]
mod observer;

#[cfg(all(feature = "metal", any(debug_assertions, feature = "profile")))]
mod on {
    use lloom_tensor::kernels::metal::profile;
    use lloom_tensor::kernels::metal::CommandMarker;
    use lloom_tensor::Device;
    use parking_lot::Mutex;

    /// How the producer learned the run was over.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum StopBy {
        /// The reader's stop flag (`DecodeOptions::stop`).
        Flag,
        /// The loop's own EOS watch.
        Watch,
        /// The producer's own EOS read (fixed cadence, blocking).
        EosOfLast,
        /// `max_tokens`.
        Length,
        /// The streaming hook's `false` on the producer's thread.
        Hook,
    }

    impl StopBy {
        fn name(self) -> &'static str {
            match self {
                StopBy::Flag => "flag",
                StopBy::Watch => "watch",
                StopBy::EosOfLast => "eos_of_last",
                StopBy::Length => "length",
                StopBy::Hook => "hook",
            }
        }
    }

    #[derive(Debug, Default)]
    struct Run {
        id: u64,
        prepare: Option<(f64, f64, String)>,
        start: f64,
        decode: Option<f64>,
        observers: Observers,
        issues: u16,
        aborted: Option<f64>,
        failure_observer: Option<lloom_tensor::kernels::metal::CommandFailureObserver>,
        stop: Option<(f64, &'static str, usize, Option<usize>)>,
        ret: Option<f64>,
        produced: usize,
        overrun: usize,
        eos: bool,
        open_dispatches: usize,
        last: Option<CommandMarker>,
        published0: Option<usize>,
    }

    #[derive(Default)]
    struct Trace {
        next_id: u64,
        pending: Vec<Run>,
        /// The last resolved lines, for a test to read ([`take_lines`]).
        lines: Vec<String>,
    }

    /// The most recently resolved runs' lines, oldest first, taken.
    pub fn take_lines() -> Vec<String> {
        std::mem::take(&mut trace().lock().lines)
    }

    fn trace() -> &'static Mutex<Trace> {
        static T: std::sync::OnceLock<Mutex<Trace>> = std::sync::OnceLock::new();
        T.get_or_init(|| Mutex::new(Trace::default()))
    }

    #[inline]
    fn on() -> bool {
        profile::enabled()
    }

    pub type Observers = Option<std::sync::Arc<ObserverGroup>>;
    pub use super::observer::{ObserverGroup, ObserverSlot, ReaderObserver};
    #[derive(Default)]
    pub struct PreparedTrace(Option<(f64, f64, String)>);
    const MAX_RETAINED_RUNS: usize = 64;
    const CONCURRENT: u16 = 1;
    const CAPACITY: u16 = 2;
    const EARLY_EXIT: u16 = 4;
    const PRODUCER_UNWIND: u16 = 8;
    const UNSUPPORTED_DEVICE: u16 = 16;
    const MARKER_QUERY_ERROR: u16 = 32;
    const NATIVE_MARKER_ERROR: u16 = 64;
    const QUEUE_FAILURE: u16 = 128;

    /// Exact producer ownership; never clones or falls back to current/last.
    pub struct RunToken {
        id: Option<u64>,
        observers: Observers,
        finished: bool,
    }
    impl RunToken {
        fn disabled() -> Self {
            Self {
                id: None,
                observers: None,
                finished: true,
            }
        }
    }
    impl Drop for RunToken {
        fn drop(&mut self) {
            if self.finished {
                return;
            }
            // Group is released BEFORE recorder lock, never Group -> Trace.
            if let Some(group) = &self.observers {
                group.seal();
            }
            let Some(id) = self.id else { return };
            let now = profile::now_s();
            let mut t = trace().lock();
            t.abort_run(id, now, std::thread::panicking());
            // Drop does not resolve/print or call user code while unwinding.
        }
    }

    impl Trace {
        fn abort_run(&mut self, id: u64, now: f64, unwinding: bool) {
            if let Some(run) = self.run_mut(id) {
                run.aborted = Some(now);
                run.issues |= if unwinding {
                    PRODUCER_UNWIND
                } else {
                    EARLY_EXIT
                };
            }
        }
        fn run_mut(&mut self, id: u64) -> Option<&mut Run> {
            self.pending.iter_mut().find(|r| r.id == id)
        }
        fn insert_run(&mut self, mut run: Run) -> Option<Run> {
            let overlap =
                |r: &Run| r.aborted.is_none() && (r.ret.is_none() || r.open_dispatches > 0);
            if self.pending.iter().any(overlap) {
                run.issues |= CONCURRENT;
                for active in self.pending.iter_mut().filter(|r| overlap(r)) {
                    active.issues |= CONCURRENT;
                }
            }
            let displaced = if self.pending.len() == MAX_RETAINED_RUNS {
                let old = self.pending.remove(0);
                emit(self, incomplete_line(&old, CAPACITY));
                Some(old)
            } else {
                None
            };
            self.pending.push(run);
            displaced
        }
    }

    /// Public only for the existing diagnostic cold Ring setup surface.
    #[inline]
    pub fn enabled() -> bool {
        on()
    }

    pub fn prepare_begin() -> Option<f64> {
        on().then(profile::now_s)
    }
    pub fn prepare_end<A: std::fmt::Debug>(t0: Option<f64>, action: &A) -> PreparedTrace {
        let Some(t0) = t0 else {
            return PreparedTrace::default();
        };
        PreparedTrace(Some((t0, profile::now_s(), format!("{action:?}"))))
    }

    pub fn run_begin(
        published: impl FnOnce() -> Option<usize>,
        prepare: PreparedTrace,
        observers: Observers,
    ) -> RunToken {
        if !on() {
            return RunToken::disabled();
        }
        // Arm observer seal before evaluating a caller-provided snapshot.
        let mut token = RunToken {
            id: None,
            observers,
            finished: false,
        };
        let published = published();
        let now = profile::now_s();
        profile::set_step(0);
        let mut t = trace().lock();
        // No wrapping identity: an exhausted diagnostic refuses capture, not
        // production work, and never reuses an old token id.
        let Some(id) = t.next_id.checked_add(1) else {
            eprintln!(
                "TERMINAL-TRACE {{\"status\":\"incomplete\",\"reason\":\"identity_exhausted\"}}"
            );
            return RunToken::disabled();
        };
        t.next_id = id;
        let run = Run {
            id,
            prepare: prepare.0,
            start: now,
            published0: published,
            observers: token.observers.clone(),
            ..Run::default()
        };
        let displaced = t.insert_run(run);
        token.id = Some(id);
        drop(t);
        drop(displaced); // pre-existing marker/native owner drops outside recorder lock
        flush();
        token
    }

    pub fn decode_begin(run: &RunToken) {
        let Some(id) = run.id else { return };
        let now = profile::now_s();
        let mut t = trace().lock();
        if let Some(r) = t.run_mut(id) {
            r.decode.get_or_insert(now);
        }
    }

    pub fn producer_stop(
        run: &RunToken,
        by: StopBy,
        counts: impl FnOnce() -> (usize, Option<usize>),
    ) {
        let Some(id) = run.id else { return }; // counts remain lazy when disabled
        let (submitted, published) = counts();
        let now = profile::now_s();
        let mut t = trace().lock();
        if let Some(r) = t.run_mut(id) {
            let completed = match (published, r.published0) {
                (Some(p), Some(p0)) => Some(p.saturating_sub(p0)),
                _ => None,
            };
            r.stop.get_or_insert((now, by.name(), submitted, completed));
        }
    }

    pub fn run_end(mut run: RunToken, produced: usize, overrun: usize, eos: bool, device: &Device) {
        let Some(id) = run.id else { return };
        let now = profile::now_s();
        // Seal registration before finalization; reader events remain open
        // until EOS/error/close. No reader/GPU wait or callback is added.
        if let Some(group) = &run.observers {
            group.seal();
        }
        let (open_dispatches, last, failure_observer, issue) = match device {
            Device::Metal(m) => {
                let open_dispatches = m.open_dispatches();
                let (last, issue) = match m.last_committed_marker() {
                    Ok(last) => (last, 0),
                    Err(_) => (None, MARKER_QUERY_ERROR),
                };
                (
                    open_dispatches,
                    last,
                    Some(m.commands().failure_observer()),
                    issue,
                )
            }
            _ => (0, None, None, UNSUPPORTED_DEVICE),
        };
        let mut t = trace().lock();
        if let Some(r) = t.run_mut(id) {
            r.ret = Some(now);
            r.produced = produced;
            r.overrun = overrun;
            r.eos = eos;
            r.open_dispatches = open_dispatches;
            r.last = last;
            r.failure_observer = failure_observer;
            r.issues |= issue;
        }
        drop(t);
        flush();
        run.finished = true;
    }

    pub fn flush() {
        if on() {
            let retired = {
                let mut t = trace().lock();
                resolve(&mut t)
            };
            drop(retired); // no arbitrary callback/native owner drop under Trace
        }
    }
    fn emit(t: &mut Trace, line: String) {
        eprintln!("TERMINAL-TRACE {line}");
        if t.lines.len() == MAX_RETAINED_RUNS {
            t.lines.remove(0);
        }
        t.lines.push(line);
    }
    fn incomplete_line(r: &Run, additional: u16) -> String {
        let observations = r
            .observers
            .as_ref()
            .map(|g| g.snapshot())
            .unwrap_or_default();
        let issues = r.issues | additional;
        let mut reasons = Vec::new();
        if issues & CONCURRENT != 0 {
            reasons.push("concurrent_global_profile");
        }
        if issues & CAPACITY != 0 {
            reasons.push("recorder_capacity");
        }
        if issues & EARLY_EXIT != 0 {
            reasons.push("producer_early_exit");
        }
        if issues & PRODUCER_UNWIND != 0 {
            reasons.push("producer_unwind");
        }
        if issues & UNSUPPORTED_DEVICE != 0 {
            reasons.push("unsupported_device");
        }
        if issues & MARKER_QUERY_ERROR != 0 {
            reasons.push("marker_query_error");
        }
        if issues & NATIVE_MARKER_ERROR != 0 {
            reasons.push("native_marker_error");
        }
        if issues & QUEUE_FAILURE != 0 {
            reasons.push("queue_failure");
        }
        if observations.overflow {
            reasons.push("observer_capacity");
        }
        if observations.errors != 0 {
            reasons.push("reader_error");
        }
        if observations.unwinds != 0 {
            reasons.push("reader_unwind");
        }
        serde_json::json!({
            "run":r.id, "status":"incomplete", "reasons":reasons,
            "native_success":"unverified", "record_coverage":"not_certified",
            "known_queue_failure":issues & QUEUE_FAILURE != 0,
            "known_marker_failure":issues & NATIVE_MARKER_ERROR != 0,
            "start":r.start, "decode":r.decode, "return":r.ret, "aborted":r.aborted,
            "reader_eos":observations.reader_eos, "watch_eos":observations.watch_eos,
            "stop":r.stop.map(|v|v.0), "stop_by":r.stop.map(|v|v.1),
            "submitted_at_stop":r.stop.map(|v|v.2), "completed_at_stop":r.stop.and_then(|v|v.3),
            "produced":r.produced, "eos":r.eos, "overrun":r.overrun,
            "observers":observations.registered, "observers_open":observations.open,
            "observers_closed":observations.closed, "reader_errors":observations.errors,
            "reader_unwinds":observations.unwinds,
            "last_submit":null, "retired":null, "eos_buf_start":null, "eos_buf_end":null
        })
        .to_string()
    }

    /// Resolve and print every pending run whose buffers have completed.
    fn resolve(t: &mut Trace) -> Vec<Run> {
        let mut retired = Vec::new();
        if t.pending.is_empty() {
            return retired;
        }
        let records = profile::command_buffer_records();
        let mut i = 0;
        while i < t.pending.len() {
            match line(&t.pending[i], &records) {
                Some(l) => {
                    emit(t, l);
                    retired.push(t.pending.remove(i));
                }
                None => i += 1,
            }
        }
        retired
    }

    /// The run's line, once every buffer it needs has completed.
    fn line(r: &Run, records: &[profile::CommandBufferRecord]) -> Option<String> {
        if r.ret.is_none() && r.aborted.is_none() {
            return None;
        }
        let observed = r
            .observers
            .as_ref()
            .map(|g| g.snapshot())
            .unwrap_or_else(|| super::observer::ObservationSnapshot {
                sealed: true,
                ..Default::default()
            });
        if !observed.terminal() {
            return None;
        }
        let known_failure = if r.failure_observer.as_ref().is_some_and(|f| f.has_failed()) {
            QUEUE_FAILURE
        } else {
            0
        } | if r.last.as_ref().is_some_and(|m| m.is_failed()) {
            NATIVE_MARKER_ERROR
        } else {
            0
        };
        if r.issues != 0 || known_failure != 0 || observed.incomplete() {
            return Some(incomplete_line(r, known_failure));
        }
        let ret = r.ret?;
        let mut by_commit: Vec<&profile::CommandBufferRecord> =
            records.iter().filter(|c| c.committed >= r.start).collect();
        by_commit.sort_by(|a, b| a.committed.total_cmp(&b.committed));
        // The run's last submission and its retirement: the first buffer
        // committed after the return when the run left work encoded, else the
        // last buffer it committed (complete once its marker is), else (all
        // retired by the return) the last record committed before it.
        let last = if r.open_dispatches > 0 {
            *by_commit.iter().find(|c| c.committed > ret)?
        } else if let Some(m) = &r.last {
            if !m.is_complete() {
                return None;
            }
            let end = m.gpu_end_time();
            by_commit
                .iter()
                .filter(|c| c.committed <= ret)
                .find(|c| c.interval.end == end)
                .copied()?
        } else {
            by_commit
                .iter()
                .rev()
                .find(|c| c.committed <= ret)
                .copied()?
        };
        let retired = by_commit
            .iter()
            .filter(|c| c.committed <= last.committed)
            .map(|c| c.interval.end)
            .fold(f64::MIN, f64::max);
        let decode = r.decode.unwrap_or(ret);
        // The prefill's last buffer is the first that carries a decode step
        // (the first token's sample rides it): its end is when the prefill
        // had retired and the first token was on the device.
        let first_token = by_commit
            .iter()
            .find(|c| c.step >= 1)
            .map(|c| c.interval.end);
        // The buffer that carried the stop's publication: the first committed
        // in the decode with a step at or past the stop's.
        let eos_step = r.eos.then_some(r.produced as u64 + 1);
        let eos_buf = eos_step.and_then(|e| {
            by_commit
                .iter()
                .find(|c| c.committed >= decode && c.step >= e)
                .copied()
        });
        let f = |x: Option<f64>| x.map_or("null".to_string(), |v| format!("{v:.9}"));
        let (stop_t, stop_by, submitted, completed) = match r.stop {
            Some((t, by, s, c)) => (Some(t), by, s, c),
            None => (None, "none", 0, None),
        };
        let (p0, p1, action) = match &r.prepare {
            Some((a, b, s)) => (Some(*a), Some(*b), s.clone()),
            None => (None, None, String::new()),
        };
        Some(format!(
            "{{\"run\":{},\"status\":\"observed\",\"native_success\":\"unverified\",\"record_coverage\":\"not_certified\",\"observers\":{},\"observers_closed\":{},\"prepare0\":{},\"prepare1\":{},\"action\":\"{}\",\"start\":{:.9},\
             \"decode\":{},\"first_token_end\":{},\"eos\":{},\"eos_step\":{},\
             \"eos_buf_committed\":{},\"eos_buf_start\":{},\"eos_buf_end\":{},\"eos_buf_step\":{},\
             \"reader_eos\":{},\"watch_eos\":{},\"stop\":{},\"stop_by\":\"{}\",\
             \"submitted_at_stop\":{},\"completed_at_stop\":{},\"return\":{:.9},\
             \"open_dispatches\":{},\"produced\":{},\"overrun\":{},\
             \"last_submit\":{:.9},\"retired\":{:.9}}}",
            r.id,
            observed.registered,
            observed.closed,
            f(p0),
            f(p1),
            action.replace('"', "'"),
            r.start,
            f(r.decode),
            f(first_token),
            r.eos,
            eos_step.map_or("null".into(), |e| e.to_string()),
            f(eos_buf.map(|c| c.committed)),
            f(eos_buf.map(|c| c.interval.start)),
            f(eos_buf.map(|c| c.interval.end)),
            eos_buf.map_or("null".into(), |c| c.step.to_string()),
            f(observed.reader_eos),
            f(observed.watch_eos),
            f(stop_t),
            stop_by,
            submitted,
            completed.map_or("null".into(), |c| c.to_string()),
            ret,
            r.open_dispatches,
            r.produced,
            r.overrun,
            last.committed,
            retired,
        ))
    }
    #[cfg(test)]
    mod identity_tests {
        use super::*;
        use std::sync::Arc;
        fn records() -> [profile::CommandBufferRecord; 1] {
            [profile::CommandBufferRecord {
                interval: profile::GpuInterval {
                    start: 1.2,
                    end: 1.8,
                },
                committed: 1.1,
                step: 1,
                label: None,
            }]
        }
        #[test]
        fn late_reader_event_delays_line_not_producer_return_and_keeps_its_time() {
            let slot = ObserverSlot::default();
            let reader = slot.reader(true).unwrap();
            slot.begin(true);
            let group = slot.group(true).unwrap();
            group.seal();
            let run = Run {
                id: 1,
                start: 1.0,
                decode: Some(1.05),
                ret: Some(2.0),
                eos: true,
                observers: Some(group),
                ..Run::default()
            };
            assert!(line(&run, &records()).is_none(), "observer is still open");
            reader.saw_eos(3.0); // ACTUAL event may follow the producer return
            let value: serde_json::Value =
                serde_json::from_str(&line(&run, &records()).unwrap()).unwrap();
            assert_eq!(value["reader_eos"].as_f64(), Some(3.0));
            assert_eq!(value["return"].as_f64(), Some(2.0));
            assert_eq!(value["retired"].as_f64(), Some(1.8));
        }
        #[test]
        fn two_run_tokens_and_preparations_never_share_writer_slots() {
            let mut trace = Trace::default();
            let _displaced = trace.insert_run(Run {
                id: 1,
                start: 1.0,
                prepare: Some((0.1, 0.2, "first".into())),
                ..Run::default()
            });
            let _displaced = trace.insert_run(Run {
                id: 2,
                start: 2.0,
                prepare: Some((0.3, 0.4, "second".into())),
                ..Run::default()
            });
            trace.run_mut(1).unwrap().stop = Some((3.0, "length", 4, Some(4)));
            trace.run_mut(2).unwrap().stop = Some((4.0, "flag", 7, Some(6)));
            assert_eq!(trace.run_mut(1).unwrap().stop.unwrap().2, 4);
            assert_eq!(trace.run_mut(2).unwrap().stop.unwrap().2, 7);
            assert!(trace.run_mut(3).is_none());
            for run in &trace.pending {
                assert_ne!(run.issues & CONCURRENT, 0);
                let value: serde_json::Value =
                    serde_json::from_str(&incomplete_line(run, 0)).unwrap();
                assert!(value["retired"].is_null());
            }
            assert_eq!(
                trace.run_mut(1).unwrap().prepare.as_ref().unwrap().2,
                "first"
            );
            assert_eq!(
                trace.run_mut(2).unwrap().prepare.as_ref().unwrap().2,
                "second"
            );
        }
        #[test]
        fn retained_run_overflow_emits_incomplete_and_never_rebinds_missing_id() {
            let mut trace = Trace::default();
            for id in 1..=MAX_RETAINED_RUNS as u64 + 1 {
                let _displaced = trace.insert_run(Run {
                    id,
                    start: id as f64,
                    ret: Some(id as f64 + 0.1),
                    ..Run::default()
                });
            }
            assert_eq!(trace.pending.len(), MAX_RETAINED_RUNS);
            assert!(trace.run_mut(1).is_none());
            let value: serde_json::Value = serde_json::from_str(&trace.lines[0]).unwrap();
            assert_eq!(value["run"].as_u64(), Some(1));
            assert_eq!(value["status"], "incomplete");
            assert_eq!(value["reasons"][0], "recorder_capacity");
            assert!(value["reader_eos"].is_null() && value["retired"].is_null());
        }
        #[test]
        fn actual_producer_token_unwind_seals_only_its_group_without_fake_return() {
            let slot = ObserverSlot::default();
            let reader = slot.reader(true).unwrap();
            slot.begin(true);
            let group = slot.group(true).unwrap();
            let id = {
                let mut t = trace().lock();
                t.next_id = t.next_id.checked_add(1).unwrap();
                let id = t.next_id;
                let displaced = t.insert_run(Run {
                    id,
                    start: 1.0,
                    observers: Some(Arc::clone(&group)),
                    ..Run::default()
                });
                drop(t);
                drop(displaced);
                id
            };
            let run = RunToken {
                id: Some(id),
                observers: Some(Arc::clone(&group)),
                finished: false,
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _reader = reader;
                let _run = run;
                panic!("controlled producer/reader unwind");
            }));
            assert!(result.is_err());
            let observations = group.snapshot();
            assert!(observations.terminal() && observations.incomplete());
            assert_eq!(observations.unwinds, 1);
            let mut t = trace().lock();
            let r = t.run_mut(id).expect("exact producer record");
            assert!(r.ret.is_none() && r.aborted.is_some());
            assert_ne!(r.issues & PRODUCER_UNWIND, 0);
            let v: serde_json::Value = serde_json::from_str(&incomplete_line(r, 0)).unwrap();
            assert!(v["return"].is_null() && v["reader_eos"].is_null());
            assert_eq!(v["reasons"][0], "producer_unwind");
            t.pending.retain(|r| r.id != id);
        }
        #[test]
        fn closed_reader_without_eos_is_explicit_not_synthesized() {
            let slot = ObserverSlot::default();
            let reader = slot.reader(true).unwrap();
            slot.begin(true);
            let group = slot.group(true).unwrap();
            group.seal();
            drop(reader);
            let run = Run {
                id: 1,
                start: 1.0,
                ret: Some(2.0),
                observers: Some(Arc::clone(&group)),
                ..Run::default()
            };
            let value: serde_json::Value =
                serde_json::from_str(&line(&run, &records()).unwrap()).unwrap();
            assert!(value["reader_eos"].is_null());
            assert_eq!(group.snapshot().closed, 1);
        }
    }
}

#[cfg(all(feature = "metal", any(debug_assertions, feature = "profile")))]
pub use on::*;

#[cfg(not(all(feature = "metal", any(debug_assertions, feature = "profile"))))]
mod off {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum StopBy {
        Flag,
        Watch,
        EosOfLast,
        Length,
        Hook,
    }
    #[derive(Default)]
    pub struct PreparedTrace;
    #[derive(Default)]
    pub struct Observers;
    pub struct RunToken;
    #[inline(always)]
    pub fn prepare_begin() -> Option<f64> {
        None
    }
    #[inline(always)]
    pub fn prepare_end<A: std::fmt::Debug>(_t0: Option<f64>, _action: &A) -> PreparedTrace {
        PreparedTrace
    }
    #[inline(always)]
    pub fn run_begin(
        _published: impl FnOnce() -> Option<usize>,
        _prepare: PreparedTrace,
        _observers: Observers,
    ) -> RunToken {
        RunToken
    }
    #[inline(always)]
    pub fn decode_begin(_run: &RunToken) {}
    #[inline(always)]
    pub fn producer_stop(
        _run: &RunToken,
        _by: StopBy,
        _counts: impl FnOnce() -> (usize, Option<usize>),
    ) {
    }
    #[inline(always)]
    pub fn run_end(
        _run: RunToken,
        _produced: usize,
        _overrun: usize,
        _eos: bool,
        _device: &lloom_tensor::Device,
    ) {
    }
    pub fn flush() {}
    pub fn take_lines() -> Vec<String> {
        Vec::new()
    }
}

#[cfg(not(all(feature = "metal", any(debug_assertions, feature = "profile"))))]
pub use off::*;

#[cfg(all(
    test,
    not(all(feature = "metal", any(debug_assertions, feature = "profile")))
))]
mod disabled_tests {
    #[test]
    fn compiled_out_trace_does_not_evaluate_count_snapshots() {
        let run = super::run_begin(
            || panic!("compiled-out run begin polled"),
            Default::default(),
            Default::default(),
        );
        super::producer_stop(&run, super::StopBy::Length, || {
            panic!("compiled-out stop polled")
        });
    }
}
