//! Setup publication and irreversible sealing share one atomic ownership word.
use std::sync::atomic::{AtomicU8, Ordering};

const OPEN: u8 = 0;
const REGISTERING: u8 = 1;
const SEALED: u8 = 2;

pub(crate) struct Setup(AtomicU8);

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Busy,
    Sealed,
}

pub(crate) struct Registration<'a>(&'a Setup);

impl Setup {
    pub(crate) fn new() -> Self {
        Self(AtomicU8::new(OPEN))
    }

    pub(crate) fn register(&self) -> Result<Registration<'_>, Refusal> {
        self.0
            .compare_exchange(OPEN, REGISTERING, Ordering::Acquire, Ordering::Acquire)
            .map(|_| Registration(self))
            .map_err(|state| {
                if state == SEALED {
                    Refusal::Sealed
                } else {
                    Refusal::Busy
                }
            })
    }

    pub(crate) fn seal(&self) -> Result<(), Refusal> {
        match self
            .0
            .compare_exchange(OPEN, SEALED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) | Err(SEALED) => Ok(()),
            Err(_) => Err(Refusal::Busy),
        }
    }
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.0 .0.store(OPEN, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn seal_excludes_the_whole_publication_interval() {
        let setup = Setup::new();
        let published = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let permit = setup.register().unwrap();
            s.spawn(|| assert_eq!(setup.seal(), Err(Refusal::Busy)))
                .join()
                .unwrap();
            assert_eq!(published.load(Ordering::Relaxed), 0);
            published.store(17, Ordering::Relaxed);
            drop(permit);
            s.spawn(|| {
                setup.seal().unwrap();
                assert_eq!(published.load(Ordering::Relaxed), 17);
            })
            .join()
            .unwrap();
        });
        assert!(matches!(setup.register(), Err(Refusal::Sealed)));
        setup.seal().unwrap();
    }

    #[test]
    fn cancellation_and_unwind_release_setup_without_reopening_a_seal() {
        let setup = Setup::new();
        let result = std::panic::catch_unwind(|| {
            let _permit = setup.register().unwrap();
            panic!("publication cancelled");
        });
        assert!(result.is_err());
        drop(setup.register().unwrap());
        setup.seal().unwrap();
        assert!(matches!(setup.register(), Err(Refusal::Sealed)));
        setup.seal().unwrap();
    }
}
