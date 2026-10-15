//! The terminal boundaries of one decode run, for the diagnostic arm
//! (`design/measurements/issue-993.md` §14): when the device published the
//! stop, when the host saw it (the streaming reader and the producer apart),
//! when the run's last work was submitted and when it retired, and when the
//! state was ready for the next request.
//!
//! Compiled only in a debug build or with `profile`, and recorded only under
//! `CANDLE_METAL_PROFILE=1` on Metal: the shipping binary carries none of it,
//! and an unprofiled run pays one cached-bool check per call site. Times are
//! `profile::now_s` seconds (`CLOCK_UPTIME_RAW`), the clock the command
//! buffers' GPU times are in. A run is resolved against the profile
//! recorder's command-buffer records once the buffers it needs have
//! completed — never by waiting for them, so the diagnostic adds no wait to
//! the run it observes. A resolved run is printed as one `TERMINAL-TRACE`
//! JSON line on stderr.

#[cfg(all(feature = "metal", any(debug_assertions, feature = "profile")))]
mod on {
    use lloom_tensor::kernels::metal::profile;
    use lloom_tensor::kernels::metal::CommandMarker;
    use lloom_tensor::Device;
    use std::sync::Mutex;

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
        reader_eos: Option<f64>,
        watch_eos: Option<f64>,
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
        prepare: Option<(f64, f64, String)>,
        current: Option<Run>,
        pending: Vec<Run>,
        /// The last resolved lines, for a test to read ([`take_lines`]).
        lines: Vec<String>,
    }

    /// The most recently resolved runs' lines, oldest first, taken.
    pub fn take_lines() -> Vec<String> {
        std::mem::take(&mut trace().lock().unwrap().lines)
    }

    fn trace() -> &'static Mutex<Trace> {
        static T: std::sync::OnceLock<Mutex<Trace>> = std::sync::OnceLock::new();
        T.get_or_init(|| Mutex::new(Trace::default()))
    }

    #[inline]
    fn on() -> bool {
        profile::enabled()
    }

    pub fn prepare_begin() -> Option<f64> {
        on().then(profile::now_s)
    }

    pub fn prepare_end<A: std::fmt::Debug>(t0: Option<f64>, action: &A) {
        let Some(t0) = t0 else { return };
        let t1 = profile::now_s();
        let mut t = trace().lock().unwrap();
        t.prepare = Some((t0, t1, format!("{action:?}")));
    }

    /// A run begins (after its request's `prepare`): earlier runs whose
    /// buffers have completed are resolved and printed.
    pub fn run_begin(published: Option<usize>) {
        if !on() {
            return;
        }
        let now = profile::now_s();
        profile::set_step(0);
        let mut t = trace().lock().unwrap();
        t.next_id += 1;
        let prepare = t.prepare.take();
        t.current = Some(Run {
            id: t.next_id,
            prepare,
            start: now,
            published0: published,
            ..Run::default()
        });
        resolve(&mut t);
    }

    pub fn decode_begin() {
        if !on() {
            return;
        }
        let now = profile::now_s();
        if let Some(r) = trace().lock().unwrap().current.as_mut() {
            r.decode.get_or_insert(now);
        }
    }

    /// A ring reader found a stop id: the loop's watch (`watch`) or a
    /// streaming reader.
    pub fn reader_eos(watch: bool) {
        if !on() {
            return;
        }
        let now = profile::now_s();
        if let Some(r) = trace().lock().unwrap().current.as_mut() {
            let slot = if watch {
                &mut r.watch_eos
            } else {
                &mut r.reader_eos
            };
            slot.get_or_insert(now);
        }
    }

    /// The producer broke out of its loop: how, with the forwards it had
    /// submitted past the prompt and the tokens the GPU had published.
    pub fn producer_stop(by: StopBy, submitted: usize, published: Option<usize>) {
        if !on() {
            return;
        }
        let now = profile::now_s();
        if let Some(r) = trace().lock().unwrap().current.as_mut() {
            let completed = match (published, r.published0) {
                (Some(p), Some(p0)) => Some(p.saturating_sub(p0)),
                _ => None,
            };
            r.stop.get_or_insert((now, by.name(), submitted, completed));
        }
    }

    /// The run returns: what it produced and whether its stop was an EOS;
    /// read off the device, what it left encoded and not committed, and the
    /// last buffer it committed.
    pub fn run_end(produced: usize, overrun: usize, eos: bool, device: &Device) {
        if !on() {
            return;
        }
        let now = profile::now_s();
        let Device::Metal(m) = device else { return };
        let open_dispatches = m.open_dispatches();
        let last = m.last_committed_marker().ok().flatten();
        let mut t = trace().lock().unwrap();
        if let Some(mut r) = t.current.take() {
            r.ret = Some(now);
            r.produced = produced;
            r.overrun = overrun;
            r.eos = eos;
            r.open_dispatches = open_dispatches;
            r.last = last;
            t.pending.push(r);
        }
        resolve(&mut t);
    }

    /// Resolve and print every pending run whose buffers have completed.
    pub fn flush() {
        if on() {
            resolve(&mut trace().lock().unwrap());
        }
    }

    fn resolve(t: &mut Trace) {
        if t.pending.is_empty() {
            return;
        }
        let records = profile::command_buffer_records();
        let mut i = 0;
        while i < t.pending.len() {
            match line(&t.pending[i], &records) {
                Some(l) => {
                    eprintln!("TERMINAL-TRACE {l}");
                    if t.lines.len() >= 64 {
                        t.lines.remove(0);
                    }
                    t.lines.push(l);
                    t.pending.remove(i);
                }
                None => i += 1,
            }
        }
    }

    /// The run's line, once every buffer it needs has completed.
    fn line(r: &Run, records: &[profile::CommandBufferRecord]) -> Option<String> {
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
                .copied()
                .or_else(|| by_commit.iter().rev().find(|c| c.committed <= ret).copied())?
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
            "{{\"run\":{},\"prepare0\":{},\"prepare1\":{},\"action\":\"{}\",\"start\":{:.9},\
             \"decode\":{},\"first_token_end\":{},\"eos\":{},\"eos_step\":{},\
             \"eos_buf_committed\":{},\"eos_buf_start\":{},\"eos_buf_end\":{},\"eos_buf_step\":{},\
             \"reader_eos\":{},\"watch_eos\":{},\"stop\":{},\"stop_by\":\"{}\",\
             \"submitted_at_stop\":{},\"completed_at_stop\":{},\"return\":{:.9},\
             \"open_dispatches\":{},\"produced\":{},\"overrun\":{},\
             \"last_submit\":{:.9},\"retired\":{:.9}}}",
            r.id,
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
            f(r.reader_eos),
            f(r.watch_eos),
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
    #[inline(always)]
    pub fn prepare_begin() -> Option<f64> {
        None
    }
    #[inline(always)]
    pub fn prepare_end<A: std::fmt::Debug>(_t0: Option<f64>, _action: &A) {}
    #[inline(always)]
    pub fn run_begin(_published: Option<usize>) {}
    #[inline(always)]
    pub fn decode_begin() {}
    #[inline(always)]
    pub fn reader_eos(_watch: bool) {}
    #[inline(always)]
    pub fn producer_stop(_by: StopBy, _submitted: usize, _published: Option<usize>) {}
    #[inline(always)]
    pub fn run_end(_produced: usize, _overrun: usize, _eos: bool, _device: &lloom_tensor::Device) {}
    #[inline(always)]
    pub fn flush() {}
    pub fn take_lines() -> Vec<String> {
        Vec::new()
    }
}

#[cfg(not(all(feature = "metal", any(debug_assertions, feature = "profile"))))]
pub use off::*;
