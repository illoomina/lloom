//! FIFO turns on a family's request slot. The slot refuses an overlapping
//! request rather than wait (`request_slot`: a callback that re-enters must
//! not deadlock), so HTTP requests queue here instead: every decision, and
//! every other model request on a single-sequence engine.
//!
//! Decisions are grouped: one that arrives while another decision is still
//! waiting for its turn joins that one's group, and the group runs as one
//! batch when the turn comes (`Decisions::decide_batch`, packed by the family
//! under `fast`). A lone request is a group of one and runs as it would alone.
use crate::decision::{OrderedJson, SystemOneError, SystemOneResponse};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;

type Answer = Result<SystemOneResponse, SystemOneError>;

#[derive(Default)]
pub(super) struct Turns {
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    /// The next ticket handed out, and the one whose turn it is.
    next: u64,
    serving: u64,
    /// The decision group still waiting for its turn, open to joiners.
    open: Option<u64>,
    groups: HashMap<u64, Group>,
}

struct Group {
    bodies: Vec<OrderedJson>,
    /// Set when the group has run: one answer per body, each taken once.
    answers: Option<Vec<Option<Answer>>>,
    unclaimed: usize,
}

/// A held turn; the next ticket's turn begins when it drops.
pub(super) struct Turn<'a> {
    turns: &'a Turns,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.turns.state.lock().serving += 1;
        self.turns.changed.notify_all();
    }
}

impl Turns {
    /// A turn now, if nobody holds one or waits for one; never waits, and
    /// never goes ahead of a request that is queued.
    pub(super) fn try_take(&self) -> Option<Turn<'_>> {
        let mut state = self.state.lock();
        if state.next != state.serving {
            return None;
        }
        state.next += 1;
        Some(Turn { turns: self })
    }

    /// Wait for a turn of one's own.
    pub(super) fn take(&self) -> Turn<'_> {
        self.take_with_queued(|_, _| {})
    }

    // One algorithm for production and the test-only queue witness. The
    // observer runs after ticket assignment, not before attempting admission.
    fn take_with_queued(&self, queued: impl FnOnce(u64, u64)) -> Turn<'_> {
        let mut state = self.state.lock();
        let ticket = state.next;
        state.next += 1;
        queued(ticket, state.serving);
        while state.serving != ticket {
            self.changed.wait(&mut state);
        }
        Turn { turns: self }
    }

    #[cfg(test)]
    pub(super) fn take_queued(&self, witness: &std::sync::mpsc::Sender<(u64, u64)>) -> Turn<'_> {
        self.take_with_queued(|ticket, serving| {
            // A dropped test observer must not unwind after reserving a ticket.
            let _ = witness.send((ticket, serving));
        })
    }

    /// Answer `body` in its group's turn: join the group that is waiting when
    /// `join` is set (the `fast` contract), else open a group of one. The
    /// group's first request runs `run` over every body that joined it.
    pub(super) fn decide(
        &self,
        body: &OrderedJson,
        join: bool,
        run: impl FnOnce(&[&OrderedJson]) -> Vec<Answer>,
    ) -> Answer {
        let mut state = self.state.lock();
        if let Some(ticket) = state.open.filter(|_| join) {
            let group = state.groups.get_mut(&ticket).expect("an open group exists");
            group.bodies.push(body.clone());
            group.unclaimed += 1;
            let index = group.bodies.len() - 1;
            loop {
                let group = state.groups.get_mut(&ticket).expect("unclaimed group");
                if let Some(answers) = &mut group.answers {
                    let answer = answers[index].take().expect("each answer is claimed once");
                    group.unclaimed -= 1;
                    if group.unclaimed == 0 {
                        state.groups.remove(&ticket);
                    }
                    return answer;
                }
                self.changed.wait(&mut state);
            }
        }
        let ticket = state.next;
        state.next += 1;
        state.groups.insert(
            ticket,
            Group {
                bodies: vec![body.clone()],
                answers: None,
                unclaimed: 0,
            },
        );
        if join {
            state.open = Some(ticket);
        }
        while state.serving != ticket {
            self.changed.wait(&mut state);
        }
        if state.open == Some(ticket) {
            state.open = None;
        }
        let bodies = std::mem::take(
            &mut state
                .groups
                .get_mut(&ticket)
                .expect("the leader's group")
                .bodies,
        );
        drop(state);
        let mut leader = Leader {
            turns: self,
            ticket,
            answers: None,
            count: bodies.len(),
        };
        let refs: Vec<&OrderedJson> = bodies.iter().collect();
        let mut answers: Vec<Option<Answer>> = run(&refs).into_iter().map(Some).collect();
        if answers.len() != bodies.len() {
            return Err(SystemOneError::Execution(format!(
                "a decision group of {} requests returned {} answers",
                bodies.len(),
                answers.len()
            )));
        }
        let mine = answers[0].take().expect("the leader's answer");
        leader.answers = Some(answers);
        mine
    }
}

/// The group leader's turn: publishes the group's answers (an execution
/// error for each joiner if the run failed or unwound) and ends the turn.
struct Leader<'a> {
    turns: &'a Turns,
    ticket: u64,
    answers: Option<Vec<Option<Answer>>>,
    count: usize,
}

impl Drop for Leader<'_> {
    fn drop(&mut self) {
        let answers = self.answers.take().unwrap_or_else(|| {
            (0..self.count)
                .map(|i| {
                    (i > 0).then(|| {
                        Err(SystemOneError::Execution(
                            "the decision group this request joined did not complete".into(),
                        ))
                    })
                })
                .collect()
        });
        let mut state = self.turns.state.lock();
        let group = state
            .groups
            .get_mut(&self.ticket)
            .expect("the leader's group");
        if group.unclaimed == 0 {
            state.groups.remove(&self.ticket);
        } else {
            group.answers = Some(answers);
        }
        state.serving += 1;
        drop(state);
        self.turns.changed.notify_all();
    }
}

#[cfg(test)]
#[path = "turns_tests.rs"]
mod tests;
