//! Shape checks as probes (`docs/porting-models.md` §21.8).
//!
//! A kernel's shape checks come in two kinds: a refusal (`shape_ensure!`, an
//! error) and a silent route (`shape_req!`, the caller takes another kernel).
//! Each macro is the check itself, written where the check already was, so
//! the condition keeps one definition.
//!
//! Without the `shape-probe` feature both expand to exactly the expression
//! they wrap — `shape_req!` to `(cond)`, `shape_ensure!` to
//! `if !(cond) { bail!(..) }` — and the operand list is not evaluated. With
//! it, each evaluation is also logged to this thread, when armed, as
//! `(site, condition text, operand values, outcome)`: the semantic cmplog and
//! branch coverage the shape campaign in `lloom-fuzz` reads. The feature is
//! enabled only from `lloom-fuzz` (its own workspace) and from test
//! dev-dependencies, so a release build cannot acquire it through feature
//! unification.
//!
//! ```ignore
//! if !shape_req!(hc_mix_fused, d % 4 == 0; d) {
//!     return None;
//! }
//! shape_ensure!(f16_gemm, k % 32 == 0; k => "K {k} is not a multiple of 32");
//! ```
//!
//! Operands are `name` (an integer or `bool` in scope) or `name = expr`.

/// A silent-routing shape check: evaluates to `cond`, a `bool`.
#[cfg(not(feature = "shape-probe"))]
#[macro_export]
macro_rules! shape_req {
    ($site:ident, $cond:expr $(; $($name:ident $(= $val:expr)?),* $(,)?)?) => {
        ($cond)
    };
}

/// A refusing shape check: `bail!`s with the message unless `cond` holds.
#[cfg(not(feature = "shape-probe"))]
#[macro_export]
macro_rules! shape_ensure {
    ($site:ident, $cond:expr $(; $($name:ident $(= $val:expr)?),* $(,)?)? => $($msg:tt)+) => {
        if !($cond) {
            $crate::bail!($($msg)+)
        }
    };
}

/// A silent-routing shape check: evaluates to `cond`, a `bool`.
#[cfg(feature = "shape-probe")]
#[macro_export]
macro_rules! shape_req {
    ($site:ident, $cond:expr $(; $($name:ident $(= $val:expr)?),* $(,)?)?) => {{
        let ok: bool = $cond;
        $crate::shape_probe::record(
            stringify!($site),
            stringify!($cond),
            $crate::shape_probe::Kind::Route,
            ok,
            &[$($((stringify!($name), $crate::__shape_operand!($name $(, $val)?))),*)?],
        );
        ok
    }};
}

/// A refusing shape check: `bail!`s with the message unless `cond` holds.
#[cfg(feature = "shape-probe")]
#[macro_export]
macro_rules! shape_ensure {
    ($site:ident, $cond:expr $(; $($name:ident $(= $val:expr)?),* $(,)?)? => $($msg:tt)+) => {{
        let ok: bool = $cond;
        $crate::shape_probe::record(
            stringify!($site),
            stringify!($cond),
            $crate::shape_probe::Kind::Refuse,
            ok,
            &[$($((stringify!($name), $crate::__shape_operand!($name $(, $val)?))),*)?],
        );
        if !ok {
            $crate::bail!($($msg)+)
        }
    }};
}

#[doc(hidden)]
#[cfg(feature = "shape-probe")]
#[macro_export]
macro_rules! __shape_operand {
    ($name:ident) => {
        $crate::shape_probe::Operand::value($name)
    };
    ($name:ident, $val:expr) => {
        $crate::shape_probe::Operand::value($val)
    };
}

#[cfg(feature = "shape-probe")]
pub use probe::*;

#[cfg(feature = "shape-probe")]
mod probe {
    use std::cell::{Cell, RefCell};

    /// Which kind of check a probe is.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Kind {
        /// A false outcome routes to another kernel.
        Route,
        /// A false outcome is an error.
        Refuse,
    }

    /// One evaluation of a shape check.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Event {
        pub site: &'static str,
        pub cond: &'static str,
        pub kind: Kind,
        pub ok: bool,
        pub operands: Vec<(&'static str, i64)>,
    }

    impl Event {
        /// The integer literals of the condition's text: the constants the
        /// operands are compared against (`d % 4 == 0` → `[4, 0]`).
        pub fn constants(&self) -> Vec<i64> {
            let mut out = Vec::new();
            let mut cur: Option<i64> = None;
            let mut prev_ident = false;
            for ch in self.cond.chars() {
                if ch == '_' && cur.is_some() {
                    continue;
                }
                match ch.to_digit(10) {
                    Some(d) if !prev_ident => {
                        cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d as i64));
                    }
                    _ => {
                        if let Some(v) = cur.take() {
                            out.push(v);
                        }
                        prev_ident = ch.is_alphanumeric() || ch == '_';
                    }
                }
            }
            if let Some(v) = cur {
                out.push(v);
            }
            out
        }
    }

    /// A value a probe can record.
    pub trait Operand {
        fn value(self) -> i64;
    }

    macro_rules! operand_int {
        ($($t:ty),*) => {$(
            impl Operand for $t {
                fn value(self) -> i64 {
                    self as i64
                }
            }
            impl Operand for &$t {
                fn value(self) -> i64 {
                    *self as i64
                }
            }
        )*};
    }
    operand_int!(usize, isize, u8, u16, u32, u64, i8, i16, i32, i64);

    impl Operand for bool {
        fn value(self) -> i64 {
            self as i64
        }
    }

    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
        static LOG: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    }

    /// Record one evaluation (the macros' target). Nothing is kept unless
    /// this thread is [`arm`]ed.
    pub fn record(
        site: &'static str,
        cond: &'static str,
        kind: Kind,
        ok: bool,
        operands: &[(&'static str, i64)],
    ) {
        if !ARMED.with(Cell::get) {
            return;
        }
        LOG.with(|l| {
            l.borrow_mut().push(Event {
                site,
                cond,
                kind,
                ok,
                operands: operands.to_vec(),
            })
        });
    }

    /// Record this thread's probes until the guard drops.
    pub fn arm() -> Armed {
        ARMED.with(|a| a.set(true));
        LOG.with(|l| l.borrow_mut().clear());
        Armed(())
    }

    /// While alive, this thread's probes are recorded.
    pub struct Armed(());

    impl Armed {
        /// The events recorded so far, in order; the log restarts empty.
        pub fn take(&self) -> Vec<Event> {
            LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
        }
    }

    impl Drop for Armed {
        fn drop(&mut self) {
            ARMED.with(|a| a.set(false));
            LOG.with(|l| l.borrow_mut().clear());
        }
    }
}

#[cfg(all(test, feature = "shape-probe"))]
mod tests {
    use super::*;

    fn route(d: usize, lr: usize) -> Option<usize> {
        if !shape_req!(t_route, d % 4 == 0; d) {
            return None;
        }
        if !shape_req!(t_route, lr <= 1024; lr, twice = lr * 2) {
            return None;
        }
        Some(d)
    }

    fn refuse(k: usize) -> crate::Result<usize> {
        shape_ensure!(t_refuse, k % 32 == 0; k => "K {k} off the block");
        Ok(k)
    }

    #[test]
    fn unarmed_probes_record_nothing_and_keep_the_outcome() {
        assert_eq!(route(8, 16), Some(8));
        assert_eq!(route(30, 16), None);
        let a = arm();
        assert!(a.take().is_empty(), "arming starts an empty log");
    }

    #[test]
    fn armed_probes_log_site_condition_operands_and_outcome() {
        let a = arm();
        assert_eq!(route(30, 16), None);
        assert_eq!(route(8, 2048), None);
        assert!(refuse(40).is_err());
        assert_eq!(refuse(64).unwrap(), 64);
        let got = a.take();
        let want = vec![
            Event {
                site: "t_route",
                cond: "d % 4 == 0",
                kind: Kind::Route,
                ok: false,
                operands: vec![("d", 30)],
            },
            Event {
                site: "t_route",
                cond: "d % 4 == 0",
                kind: Kind::Route,
                ok: true,
                operands: vec![("d", 8)],
            },
            Event {
                site: "t_route",
                cond: "lr <= 1024",
                kind: Kind::Route,
                ok: false,
                operands: vec![("lr", 2048), ("twice", 4096)],
            },
            Event {
                site: "t_refuse",
                cond: "k % 32 == 0",
                kind: Kind::Refuse,
                ok: false,
                operands: vec![("k", 40)],
            },
            Event {
                site: "t_refuse",
                cond: "k % 32 == 0",
                kind: Kind::Refuse,
                ok: true,
                operands: vec![("k", 64)],
            },
        ];
        assert_eq!(got, want);
        assert_eq!(got[2].constants(), vec![1024]);
        assert_eq!(got[0].constants(), vec![4, 0]);
        drop(a);
        assert_eq!(route(30, 16), None);
        let a = arm();
        assert!(a.take().is_empty(), "a dropped guard disarms and clears");
    }

    #[test]
    fn identifiers_with_digits_are_not_constants() {
        let e = Event {
            site: "s",
            cond: "matches!(hd, 32 | 64) && x2 < 1_024 && f16 > 7",
            kind: Kind::Route,
            ok: true,
            operands: vec![],
        };
        assert_eq!(e.constants(), vec![32, 64, 1024, 7]);
    }
}
