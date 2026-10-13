//! Bounded schedule choices and exact checkpoint replay shared by test runtimes.
//!
//! Exploration maps input bytes onto the CURRENT runnable set. Replay does not:
//! it requires the recorded set, selected task, site, ordinal and terminal flag.
//! This module adds no synchronization and does not model weak memory or GPU
//! ordering. Callers supply only API-permitted runnable sets and hook events.
//! The trace is a bounded payload for the existing campaign/record machinery;
//! build/configuration identity still belongs to that outer archive.

pub const MAX_TASKS: u8 = 32;
pub const MAX_STEPS: u16 = 256;
const MAGIC: &[u8; 8] = b"LSCH\x01\0\0\0";
const HEADER: usize = 30;
const ENTRY: usize = 12;
pub const MAX_TRACE_BYTES: usize = HEADER + ENTRY * MAX_STEPS as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub site_schema: u64,
    /// Bits 1..=63 name the declared hook sites. Site zero is reserved.
    pub sites: u64,
    pub tasks: u8,
    pub budget: u16,
}
impl Config {
    fn validate(self) -> Result<(), Error> {
        if self.site_schema == 0
            || self.sites == 0
            || self.sites & 1 != 0
            || self.tasks == 0
            || self.tasks > MAX_TASKS
            || self.budget == 0
            || self.budget > MAX_STEPS
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }
    fn task_mask(self) -> u32 {
        u32::MAX >> (MAX_TASKS - self.tasks)
    }
    fn allows_site(self, site: u16) -> bool {
        site < 64 && self.sites & (1u64 << site) != 0
    }
}

impl Config {
    /// The engine's schedule points (`lloom_tensor::schedule_hook`): its site
    /// ids declared, its schema, so a trace recorded against another point
    /// list is refused as configuration drift.
    pub fn engine(tasks: u8, budget: u16) -> Config {
        use lloom_tensor::schedule_hook::{SCHEMA, SITES};
        Config {
            site_schema: SCHEMA,
            sites: SITES.iter().fold(0, |sites, site| sites | 1 << *site as u8),
            tasks,
            budget,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub task: u8,
    pub site: u16,
    /// Zero-based visit across ALL sites for this task, including terminal.
    pub ordinal: u32,
    pub terminal: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    pub runnable: u32,
    pub point: Point,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    config: Config,
    steps: Vec<Step>,
}
impl Trace {
    pub fn config(&self) -> Config {
        self.config
    }
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER + ENTRY * self.steps.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.config.site_schema.to_le_bytes());
        bytes.extend_from_slice(&self.config.sites.to_le_bytes());
        bytes.push(self.config.tasks);
        bytes.push(0); // Reserved; nonzero is incompatible, not ignored.
        bytes.extend_from_slice(&self.config.budget.to_le_bytes());
        bytes.extend_from_slice(&(self.steps.len() as u16).to_le_bytes());
        for step in &self.steps {
            bytes.extend_from_slice(&step.runnable.to_le_bytes());
            bytes.push(step.point.task);
            bytes.push(u8::from(step.point.terminal));
            bytes.extend_from_slice(&step.point.site.to_le_bytes());
            bytes.extend_from_slice(&step.point.ordinal.to_le_bytes());
        }
        bytes
    }
    /// Validate the complete bounded payload before a runtime can resume tasks.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER
            || bytes.len() > MAX_TRACE_BYTES
            || &bytes[..8] != MAGIC
            || bytes[25] != 0
        {
            return Err(Error::InvalidTrace);
        }
        let config = Config {
            site_schema: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            sites: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            tasks: bytes[24],
            budget: u16::from_le_bytes(bytes[26..28].try_into().unwrap()),
        };
        config.validate()?;
        let count = u16::from_le_bytes(bytes[28..30].try_into().unwrap()) as usize;
        if count > config.budget as usize || bytes.len() != HEADER + ENTRY * count {
            return Err(Error::InvalidTrace);
        }
        let mut trace = Trace {
            config,
            steps: Vec::with_capacity(count),
        };
        let mut ordinals = [0u32; MAX_TASKS as usize];
        let mut terminal = 0u32;
        for chunk in bytes[HEADER..].chunks_exact(ENTRY) {
            if chunk[5] > 1 {
                return Err(Error::InvalidTrace);
            }
            let step = Step {
                runnable: u32::from_le_bytes(chunk[..4].try_into().unwrap()),
                point: Point {
                    task: chunk[4],
                    terminal: chunk[5] == 1,
                    site: u16::from_le_bytes(chunk[6..8].try_into().unwrap()),
                    ordinal: u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                },
            };
            validate_runnable(config, terminal, step.runnable)?;
            validate_point(config, &ordinals, step.runnable, step.point)?;
            ordinals[step.point.task as usize] += 1;
            if step.point.terminal {
                terminal |= 1 << step.point.task;
            }
            trace.steps.push(step);
        }
        Ok(trace)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidConfig,
    InvalidTrace,
    ConfigurationDrift,
    NoRunnable,
    InvalidRunnable,
    InvalidPoint,
    ChoiceExhausted,
    StepBudgetExhausted,
    ReplayExhausted,
    RunnableDrift { expected: u32, actual: u32 },
    PointDrift { expected: Point, actual: Point },
    PendingObservation,
    NoSelection,
    IncompleteReplay { remaining: usize },
}

fn validate_runnable(config: Config, terminal: u32, runnable: u32) -> Result<(), Error> {
    if runnable == 0 {
        return Err(Error::NoRunnable);
    }
    if runnable & !config.task_mask() != 0 || runnable & terminal != 0 {
        return Err(Error::InvalidRunnable);
    }
    Ok(())
}
fn validate_point(
    config: Config,
    ordinals: &[u32; MAX_TASKS as usize],
    runnable: u32,
    point: Point,
) -> Result<(), Error> {
    if point.task >= config.tasks
        || runnable & (1 << point.task) == 0
        || !config.allows_site(point.site)
        || point.ordinal != ordinals[point.task as usize]
    {
        return Err(Error::InvalidPoint);
    }
    Ok(())
}

enum Mode<'a> {
    Explore(&'a [u8]),
    Replay(&'a Trace),
}
/// Controller-owned scheduling state; neither a thread executor nor a lock adapter.
/// One successful `select` requires exactly one successful `observe` before
/// another resume. Ordinary refusal changes no cursor, ordinal or task lifetime.
pub struct Schedule<'a> {
    config: Config,
    mode: Mode<'a>,
    trace: Trace,
    ordinals: [u32; MAX_TASKS as usize],
    terminal: u32,
    pending: Option<(u32, u8)>,
}
impl<'a> Schedule<'a> {
    pub fn explore(config: Config, choices: &'a [u8]) -> Result<Self, Error> {
        Self::new(config, Mode::Explore(choices))
    }
    pub fn replay(config: Config, trace: &'a Trace) -> Result<Self, Error> {
        if config != trace.config {
            return Err(Error::ConfigurationDrift);
        }
        Self::new(config, Mode::Replay(trace))
    }
    fn new(config: Config, mode: Mode<'a>) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config,
            mode,
            trace: Trace {
                config,
                steps: Vec::with_capacity(config.budget as usize),
            },
            ordinals: [0; MAX_TASKS as usize],
            terminal: 0,
            pending: None,
        })
    }
    pub fn select(&mut self, runnable: u32) -> Result<u8, Error> {
        if self.pending.is_some() {
            return Err(Error::PendingObservation);
        }
        validate_runnable(self.config, self.terminal, runnable)?;
        let index = self.trace.steps.len();
        if index >= self.config.budget as usize {
            return Err(Error::StepBudgetExhausted);
        }
        let task = match &self.mode {
            Mode::Explore(choices) => {
                let choice = *choices.get(index).ok_or(Error::ChoiceExhausted)?;
                let mut rank = u32::from(choice) % runnable.count_ones();
                let mut candidates = runnable;
                while rank != 0 {
                    candidates &= candidates - 1;
                    rank -= 1;
                }
                candidates.trailing_zeros() as u8
            }
            Mode::Replay(trace) => {
                let expected = trace.steps.get(index).ok_or(Error::ReplayExhausted)?;
                if runnable != expected.runnable {
                    return Err(Error::RunnableDrift {
                        expected: expected.runnable,
                        actual: runnable,
                    });
                }
                expected.point.task
            }
        };
        self.pending = Some((runnable, task));
        Ok(task)
    }
    pub fn observe(&mut self, point: Point) -> Result<(), Error> {
        let (runnable, task) = self.pending.ok_or(Error::NoSelection)?;
        if task != point.task {
            return Err(Error::InvalidPoint);
        }
        validate_point(self.config, &self.ordinals, runnable, point)?;
        if let Mode::Replay(trace) = self.mode {
            let expected = trace.steps[self.trace.steps.len()].point;
            if point != expected {
                return Err(Error::PointDrift {
                    expected,
                    actual: point,
                });
            }
        }
        self.trace.steps.push(Step { runnable, point });
        self.ordinals[task as usize] += 1;
        if point.terminal {
            self.terminal |= 1 << task;
        }
        self.pending = None;
        Ok(())
    }
    /// Ending a prefix is valid exploration, not proof all workers completed.
    /// Replay must consume every recorded checkpoint; it cannot silently truncate.
    pub fn finish(self) -> Result<Trace, Error> {
        if self.pending.is_some() {
            return Err(Error::PendingObservation);
        }
        if let Mode::Replay(trace) = self.mode {
            if self.trace.steps.len() != trace.steps.len() {
                return Err(Error::IncompleteReplay {
                    remaining: trace.steps.len() - self.trace.steps.len(),
                });
            }
        }
        Ok(self.trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            site_schema: 1,
            sites: 6,
            tasks: 2,
            budget: 4,
        }
    }
    fn point(task: u8, ordinal: u32, terminal: bool) -> Point {
        Point {
            task,
            ordinal,
            site: if terminal { 2 } else { 1 },
            terminal,
        }
    }
    fn sample() -> Trace {
        let mut s = Schedule::explore(config(), &[1, 0, 0, 0]).unwrap();
        assert_eq!(s.select(3), Ok(1));
        s.observe(point(1, 0, false)).unwrap();
        assert_eq!(s.select(3), Ok(0));
        s.observe(point(0, 0, true)).unwrap();
        assert_eq!(s.select(2), Ok(1));
        s.observe(point(1, 1, true)).unwrap();
        s.finish().unwrap()
    }
    #[test]
    fn engine_points_declare_a_valid_configuration() {
        use lloom_tensor::schedule_hook::{SCHEMA, SITES};
        let engine = Config::engine(2, 16);
        assert_eq!(engine.validate(), Ok(()));
        for site in SITES {
            assert!(engine.allows_site(site as u16), "{site:?}");
        }
        assert!(!engine.allows_site(SITES.len() as u16 + 1));
        assert_eq!(engine.sites.count_ones() as usize, SITES.len());
        let mut s = Schedule::explore(engine, &[0]).unwrap();
        assert_eq!(s.select(1), Ok(0));
        let first = SITES[0] as u16;
        s.observe(Point {
            task: 0,
            site: first,
            ordinal: 0,
            terminal: false,
        })
        .unwrap();
        let trace = s.finish().unwrap();
        let other = Config {
            site_schema: SCHEMA ^ 1,
            ..engine
        };
        assert_eq!(
            Schedule::replay(other, &trace).err(),
            Some(Error::ConfigurationDrift)
        );
        assert!(Schedule::replay(engine, &trace).is_ok());
    }
    #[test]
    fn exact_round_trip_preserves_sets_sites_ordinals_and_terminal() {
        let trace = sample();
        let bytes = trace.encode();
        let decoded = Trace::decode(&bytes).unwrap();
        assert_eq!(trace, decoded);
        let mut s = Schedule::replay(config(), &decoded).unwrap();
        for step in trace.steps() {
            assert_eq!(s.select(step.runnable), Ok(step.point.task));
            s.observe(step.point).unwrap();
        }
        assert_eq!(s.finish().unwrap().encode(), bytes);
    }
    #[test]
    fn runnable_drift_refuses_without_modulo_remapping_or_consumption() {
        let trace = sample();
        let mut s = Schedule::replay(config(), &trace).unwrap();
        assert_eq!(
            s.select(1),
            Err(Error::RunnableDrift {
                expected: 3,
                actual: 1
            })
        );
        assert_eq!(s.select(3), Ok(1));
        s.observe(point(1, 0, false)).unwrap();
    }
    #[test]
    fn point_drift_retains_pending_selection_and_exact_expected_event() {
        let trace = sample();
        let mut s = Schedule::replay(config(), &trace).unwrap();
        s.select(3).unwrap();
        let mut wrong = point(1, 0, false);
        wrong.site = 2;
        assert_eq!(
            s.observe(wrong),
            Err(Error::PointDrift {
                expected: trace.steps()[0].point,
                actual: wrong
            })
        );
        assert_eq!(s.select(3), Err(Error::PendingObservation));
        s.observe(point(1, 0, false)).unwrap();
    }
    #[test]
    fn budget_choice_and_replay_exhaustion_are_distinct() {
        let mut s = Schedule::explore(config(), &[]).unwrap();
        assert_eq!(s.select(3), Err(Error::ChoiceExhausted));
        let mut c = config();
        c.budget = 1;
        let mut s = Schedule::explore(c, &[0, 0]).unwrap();
        s.select(3).unwrap();
        s.observe(point(0, 0, false)).unwrap();
        assert_eq!(s.select(3), Err(Error::StepBudgetExhausted));
        let trace = sample();
        let mut s = Schedule::replay(config(), &trace).unwrap();
        for step in trace.steps() {
            s.select(step.runnable).unwrap();
            s.observe(step.point).unwrap();
        }
        // Valid runnable here is intentionally hypothetical; both actual tasks
        // terminated, so use a nonterminal prefix to test replay exhaustion.
        assert_eq!(s.select(3), Err(Error::InvalidRunnable));
        let mut s = Schedule::explore(config(), &[0]).unwrap();
        s.select(3).unwrap();
        s.observe(point(0, 0, false)).unwrap();
        let prefix = s.finish().unwrap();
        let mut r = Schedule::replay(config(), &prefix).unwrap();
        r.select(3).unwrap();
        r.observe(point(0, 0, false)).unwrap();
        assert_eq!(r.select(3), Err(Error::ReplayExhausted));
    }
    #[test]
    fn unknown_sites_bad_ordinals_and_terminal_reentry_are_refused() {
        let mut s = Schedule::explore(config(), &[0, 0]).unwrap();
        s.select(3).unwrap();
        let mut p = point(0, 0, true);
        p.site = 3;
        assert_eq!(s.observe(p), Err(Error::InvalidPoint));
        p.site = 2;
        p.ordinal = 1;
        assert_eq!(s.observe(p), Err(Error::InvalidPoint));
        p.ordinal = 0;
        s.observe(p).unwrap();
        assert_eq!(s.select(3), Err(Error::InvalidRunnable));
        assert_eq!(s.select(2), Ok(1));
    }
    #[test]
    fn finishing_with_pending_or_missing_replay_is_not_success() {
        let mut s = Schedule::explore(config(), &[0]).unwrap();
        s.select(3).unwrap();
        assert_eq!(s.finish(), Err(Error::PendingObservation));
        let trace = sample();
        let s = Schedule::replay(config(), &trace).unwrap();
        assert_eq!(s.finish(), Err(Error::IncompleteReplay { remaining: 3 }));
    }
    #[test]
    fn configuration_drift_and_invalid_bounds_are_refused_before_resume() {
        let trace = sample();
        let mut c = config();
        c.site_schema = 2;
        assert!(matches!(
            Schedule::replay(c, &trace),
            Err(Error::ConfigurationDrift)
        ));
        for c in [
            Config {
                tasks: 0,
                ..config()
            },
            Config {
                tasks: 33,
                ..config()
            },
            Config {
                sites: 1,
                ..config()
            },
            Config {
                budget: 257,
                ..config()
            },
        ] {
            assert!(matches!(
                Schedule::explore(c, &[]),
                Err(Error::InvalidConfig)
            ));
        }
    }
    #[test]
    fn malformed_trace_is_bounded_and_never_silently_migrated() {
        let bytes = sample().encode();
        for len in 0..bytes.len() {
            assert!(Trace::decode(&bytes[..len]).is_err());
        }
        for (index, value) in [
            (4, 2),
            (25, 1),
            (30 + 5, 2),
            (30 + 4, 32),
            (30 + 6, 0),
            (30 + 8, 1),
        ] {
            let mut bad = bytes.clone();
            bad[index] = value;
            assert!(Trace::decode(&bad).is_err(), "accepted bad byte {index}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Trace::decode(&extra).is_err());
        assert!(Trace::decode(&vec![0; MAX_TRACE_BYTES + 1]).is_err());
    }
    #[test]
    fn maximum_task_bit_and_maximum_trace_have_exact_bounded_encoding() {
        let c = Config {
            tasks: 32,
            budget: MAX_STEPS,
            ..config()
        };
        let choices = [255; MAX_STEPS as usize];
        let mut s = Schedule::explore(c, &choices).unwrap();
        for ordinal in 0..MAX_STEPS {
            assert_eq!(s.select(1 << 31), Ok(31));
            s.observe(point(31, ordinal as u32, false)).unwrap();
        }
        let trace = s.finish().unwrap();
        assert_eq!(trace.encode().len(), MAX_TRACE_BYTES);
        assert_eq!(Trace::decode(&trace.encode()).unwrap(), trace);
    }
}
