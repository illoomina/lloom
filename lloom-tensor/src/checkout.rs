//! Transactional ownership of a fixed number of checked-out resources.
//! Drop releases only unfinished acquisitions; success transfers every item.

pub struct Checkout<T, F: FnMut(T), const N: usize> {
    items: [Option<T>; N],
    len: usize,
    release: F,
}
impl<T, F: FnMut(T), const N: usize> Checkout<T, F, N> {
    pub fn new(release: F) -> Self {
        Self {
            items: std::array::from_fn(|_| None),
            len: 0,
            release,
        }
    }
    /// The acquisition closure returns after its own locks are released. On
    /// error, the caller's `?` drops this transaction, returning prior items.
    pub fn acquire<E>(&mut self, acquire: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
        assert!(self.len < N, "checkout exceeds fixed transaction size");
        let item = acquire()?;
        let index = self.len;
        self.items[index] = Some(item);
        self.len += 1;
        Ok(self.items[index].as_ref().expect("just installed"))
    }
    pub fn into_array(mut self) -> [T; N] {
        assert_eq!(self.len, N, "incomplete checkout cannot be published");
        std::array::from_fn(|i| self.items[i].take().expect("complete checkout"))
    }
}
impl<T, F: FnMut(T), const N: usize> Drop for Checkout<T, F, N> {
    fn drop(&mut self) {
        for item in &mut self.items {
            if let Some(item) = item.take() {
                (self.release)(item);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    // This is the actual production guard and acquisition method. The fake
    // ring has three numeric entries; no native buffers, threads, or files.
    #[test]
    fn acquisition_refusal_returns_first_and_second_prior_entries() {
        for fail_at in [1, 2] {
            let taken = RefCell::new([false; 3]);
            let released = RefCell::new(Vec::new());
            let result = (|| -> Result<[usize; 3], ()> {
                let mut transaction = Checkout::new(|i: usize| {
                    assert!(taken.borrow()[i]);
                    taken.borrow_mut()[i] = false;
                    released.borrow_mut().push(i);
                });
                for i in 0..3 {
                    transaction.acquire(|| {
                        if i == fail_at {
                            return Err(());
                        }
                        assert!(!taken.borrow()[i]);
                        taken.borrow_mut()[i] = true;
                        Ok(i)
                    })?;
                }
                Ok(transaction.into_array())
            })();
            assert_eq!(result, Err(()));
            assert_eq!(*taken.borrow(), [false; 3]);
            assert_eq!(*released.borrow(), (0..fail_at).collect::<Vec<_>>());
        }
    }
    #[test]
    fn successful_checkout_transfers_without_releasing() {
        let released = RefCell::new(Vec::new());
        let mut transaction = Checkout::new(|i: usize| released.borrow_mut().push(i));
        for i in 0..3 {
            transaction.acquire(|| Ok::<_, ()>(i)).unwrap();
        }
        let result: [usize; 3] = transaction.into_array();
        assert_eq!(result, [0, 1, 2]);
        assert!(released.borrow().is_empty());
    }
    #[test]
    fn single_checkout_unwind_returns_unused_entry() {
        let released = std::cell::Cell::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut transaction: Checkout<usize, _, 1> = Checkout::new(|i| released.set(i));
            transaction.acquire(|| Ok::<_, ()>(7)).unwrap();
            panic!("fill refused before stamp");
        }));
        assert!(result.is_err());
        assert_eq!(released.get(), 7);
    }
}
