//! What a request reused of its session (`sequence-reuse-design.md` §2.3f):
//! the accepted boundary the session held when the request arrived, how the
//! state there was made ready, and what was kept, replayed and newly
//! prefilled. Every decode path that prepares a session reports through
//! [`prepare`], so the report is the preparation's own, not a reader's
//! reconstruction.

use lloom::nn::{CacheReset, CacheRewind, CausalLM, Prepared, ReuseAction, Session};
use lloom_tensor::Result;

/// How the state at the accepted boundary was made ready for the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StateReady {
    /// The state ended at the boundary: nothing ran past it.
    #[default]
    Exact,
    /// The forwards past it were taken back by the rewind horizon (a
    /// standing window, a conv ring's history).
    Window,
    /// The forwards past it wrote nothing (the EOS gate): counters moved
    /// back.
    Gate,
    /// The forwards past it could not be taken back: the state was restored
    /// to the turn checkpoint at this position, before the boundary.
    Checkpoint(usize),
    /// Nothing could take them back: the sequence was reset.
    Lost,
}

impl StateReady {
    pub fn name(self) -> String {
        match self {
            Self::Exact => "exact".into(),
            Self::Window => "window".into(),
            Self::Gate => "gate".into(),
            Self::Checkpoint(p) => format!("checkpoint@{p}"),
            Self::Lost => "lost".into(),
        }
    }
}

/// What the request did with the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Outcome {
    /// The whole log is a prefix of the request.
    Extend,
    /// The request is the log, and its last prediction was held.
    Resume,
    /// The cache was taken back to where the request leaves the log by the
    /// rewind horizon.
    Rewind,
    /// The cache was restored to a turn checkpoint and the request
    /// prefilled from there.
    Restore,
    /// Shared from another row's cache by page.
    Shared,
    /// Nothing kept.
    #[default]
    Fresh,
}

impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Self::Extend => "extend",
            Self::Resume => "resume",
            Self::Rewind => "rewind",
            Self::Restore => "restore",
            Self::Shared => "shared",
            Self::Fresh => "fresh",
        }
    }
}

/// Why the request did not simply extend or resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reason {
    /// It extended or resumed.
    #[default]
    None,
    /// The owner reuses nothing between requests (`context.prefix.reuse`).
    PolicyOff,
    /// The session held nothing.
    Cold,
    /// The session's cache held state of unknown tokens.
    Opaque,
    /// The request shares no prefix with the log.
    NoOverlap,
    /// The request leaves the log before its end.
    Diverged,
    /// The request is the log whole and no prediction was held: its last
    /// id is fed again.
    Refeed,
    /// The state had run past the boundary further than it could take back.
    Overrun,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PolicyOff => "policy-off",
            Self::Cold => "cold",
            Self::Opaque => "opaque",
            Self::NoOverlap => "no-overlap",
            Self::Diverged => "diverged",
            Self::Refeed => "refeed",
            Self::Overrun => "overrun",
        }
    }
}

/// One request's reuse of its session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TurnReuse {
    /// The log's length when the request arrived: the previous run's
    /// accepted boundary.
    pub boundary: usize,
    /// Positions the state had advanced past the boundary.
    pub overrun: usize,
    pub state: StateReady,
    pub outcome: Outcome,
    pub reason: Reason,
    /// Where the request leaves the log, when it does before its end.
    pub diverged_at: Option<usize>,
    /// Request ids whose state was kept.
    pub reused: usize,
    /// Request ids the log held but the state could not keep: prefilled
    /// again.
    pub replayed: usize,
    /// Request ids prefilled for the first time.
    pub prefilled: usize,
}

impl TurnReuse {
    /// The report for a request the owner served with reuse off: the
    /// session was reset before it, so nothing was kept by policy.
    pub fn with_policy(mut self, reuse: Option<bool>) -> Self {
        if reuse == Some(false) && self.outcome == Outcome::Fresh && self.boundary == 0 {
            self.reason = Reason::PolicyOff;
        }
        self
    }

    /// The report for a request served by sharing another row's prefix
    /// ([`Session::prepare_shared`]): `boundary` is the routed sequence's log
    /// before it.
    pub fn of_shared(boundary: usize, prepared: Prepared) -> (Prepared, Self) {
        let outcome = match prepared.action {
            ReuseAction::Shared(_) => Outcome::Shared,
            ReuseAction::Extend => Outcome::Extend,
            ReuseAction::Resume => Outcome::Resume,
            ReuseAction::Rewound(_) => Outcome::Rewind,
            ReuseAction::Fresh => Outcome::Fresh,
        };
        let report = Self {
            boundary,
            outcome,
            reason: match outcome {
                Outcome::Extend | Outcome::Resume => Reason::None,
                Outcome::Fresh if boundary == 0 => Reason::Cold,
                _ => Reason::NoOverlap,
            },
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
            ..Self::default()
        };
        (prepared, report)
    }

    /// `key=value` fields for a log line that already says `reused=`.
    pub fn fields(&self) -> String {
        format!(
            "boundary={} overrun={} state={} outcome={} reason={} diverged_at={} replayed={} prefilled_new={}",
            self.boundary,
            self.overrun,
            self.state.name(),
            self.outcome.name(),
            self.reason.name(),
            self.diverged_at
                .map_or_else(|| "-".to_string(), |p| p.to_string()),
            self.replayed,
            self.prefilled,
        )
    }
}

/// [`Session::prepare_with`], reporting what it did. The overrun is
/// reconciled here, as `prepare` would reconcile it first; the outcome is
/// then read off the preparation and the horizon it read.
pub fn prepare<M: CausalLM>(
    session: &mut Session<'_, M>,
    full: &[u32],
    resume: bool,
) -> Result<(Prepared, TurnReuse)>
where
    M::Cache: CacheRewind + CacheReset,
{
    let seq = session.sequence();
    let boundary = seq.tokens().len();
    let overrun = seq.overrun();
    let known = seq.tokens_known();
    let lcp = seq.common_prefix(full);
    let unwritten = session.overrun_unwritten();
    let state = if overrun == 0 {
        StateReady::Exact
    } else if session.reconcile().is_ok() {
        if unwritten {
            StateReady::Gate
        } else {
            StateReady::Window
        }
    } else {
        // `prepare` puts the state back at the latest checkpoint inside the
        // log, else resets.
        match session
            .sequence()
            .cache()
            .restore_point(boundary)
            .filter(|_| known)
        {
            Some(at) => StateReady::Checkpoint(at),
            None => StateReady::Lost,
        }
    };
    let horizon = session.sequence().cache().rewind_horizon();
    let prepared = session.prepare_with(full, resume)?;
    // What the request could have kept of the log: the common prefix, less
    // the last id when the request is the log whole (it is fed again for a
    // prediction).
    let target = if lcp == full.len() {
        lcp.saturating_sub(1)
    } else {
        lcp
    }
    .min(boundary);
    let restored = matches!(state, StateReady::Checkpoint(_));
    let outcome = match prepared.action {
        ReuseAction::Extend => Outcome::Extend,
        ReuseAction::Resume => Outcome::Resume,
        ReuseAction::Shared(_) => Outcome::Shared,
        ReuseAction::Fresh => Outcome::Fresh,
        ReuseAction::Rewound(_) if !restored && boundary - target <= horizon => Outcome::Rewind,
        ReuseAction::Rewound(_) => Outcome::Restore,
    };
    let diverged_at = (known && lcp < boundary && boundary > 0).then_some(lcp);
    let reason = match outcome {
        Outcome::Extend | Outcome::Resume if !restored => Reason::None,
        Outcome::Fresh if boundary == 0 => Reason::Cold,
        Outcome::Fresh if !known => Reason::Opaque,
        _ if matches!(state, StateReady::Checkpoint(_) | StateReady::Lost) => Reason::Overrun,
        Outcome::Fresh if lcp == 0 => Reason::NoOverlap,
        _ if diverged_at.is_some() => Reason::Diverged,
        Outcome::Shared => Reason::NoOverlap,
        _ => Reason::Refeed,
    };
    let reused = prepared.reused;
    let replayed = if known {
        target.saturating_sub(reused)
    } else {
        0
    };
    let report = TurnReuse {
        boundary,
        overrun,
        state,
        outcome,
        reason,
        diverged_at,
        reused,
        replayed,
        prefilled: prepared.suffix.len().saturating_sub(replayed),
    };
    Ok((prepared, report))
}

impl TurnReuse {
    /// The report as a response's `timings.turn` carries it.
    pub fn json(&self, policy: Option<bool>) -> serde_json::Value {
        serde_json::json!({
            "reuse_policy": policy_name(policy),
            "boundary": self.boundary,
            "overrun": self.overrun,
            "state": self.state.name(),
            "outcome": self.outcome.name(),
            "reason": self.reason.name(),
            "diverged_at": self.diverged_at,
            "reused": self.reused,
            "replayed": self.replayed,
            "prefilled": self.prefilled,
        })
    }
}

/// `on`, `off`, or `owner` where the owner does not say.
pub fn policy_name(policy: Option<bool>) -> &'static str {
    match policy {
        Some(true) => "on",
        Some(false) => "off",
        None => "owner",
    }
}

/// The reports of every request a process served, counted: what `/health`
/// shows.
#[derive(Debug, Default)]
pub struct Tally {
    turns: u64,
    by_outcome: std::collections::BTreeMap<&'static str, u64>,
    by_reason: std::collections::BTreeMap<&'static str, u64>,
    by_state: std::collections::BTreeMap<&'static str, u64>,
    reused: u64,
    replayed: u64,
    prefilled: u64,
    last: Option<TurnReuse>,
}

impl Tally {
    pub fn note(&mut self, t: &TurnReuse) {
        self.turns += 1;
        *self.by_outcome.entry(t.outcome.name()).or_default() += 1;
        *self.by_reason.entry(t.reason.name()).or_default() += 1;
        let state = match t.state {
            StateReady::Checkpoint(_) => "checkpoint",
            StateReady::Exact => "exact",
            StateReady::Window => "window",
            StateReady::Gate => "gate",
            StateReady::Lost => "lost",
        };
        *self.by_state.entry(state).or_default() += 1;
        self.reused += t.reused as u64;
        self.replayed += t.replayed as u64;
        self.prefilled += t.prefilled as u64;
        self.last = Some(*t);
    }

    pub fn json(&self, policy: Option<bool>) -> serde_json::Value {
        serde_json::json!({
            "reuse_policy": policy_name(policy),
            "turns": self.turns,
            "outcome": self.by_outcome,
            "reason": self.by_reason,
            "state": self.by_state,
            "reused_tokens": self.reused,
            "replayed_tokens": self.replayed,
            "prefilled_tokens": self.prefilled,
            "last": self.last.map(|t| t.json(policy)),
        })
    }
}
