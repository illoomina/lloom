use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Manages a collection of threads.
///
/// A new thread is created every time all the existing threads are full.
/// Any idle thread will automatically die after a few seconds. Every thread's
/// handle is retained until it is joined, so a closed pool can be joined.
pub struct TaskPool {
    sharing: Arc<Sharing>,
}

type Task = Box<dyn FnMut() + Send>;

struct Sharing {
    // list of the tasks to be done by worker threads
    todo: Mutex<VecDeque<Task>>,

    // condvar that will be notified whenever a task is added to `todo`
    condvar: Condvar,

    // number of total worker threads running
    active_tasks: AtomicUsize,

    // number of idle worker threads
    waiting_tasks: AtomicUsize,

    // set once by close(): no task starts after it and idle threads exit
    closing: AtomicBool,

    // handles of every thread not yet joined
    handles: parking_lot::Mutex<Vec<JoinHandle<()>>>,
}

/// Minimum number of active threads.
static MIN_THREADS: usize = 4;

struct Registration<'a> {
    nb: &'a AtomicUsize,
}

impl<'a> Registration<'a> {
    fn new(nb: &'a AtomicUsize) -> Registration<'a> {
        nb.fetch_add(1, Ordering::Release);
        Registration { nb }
    }
}

impl<'a> Drop for Registration<'a> {
    fn drop(&mut self) {
        self.nb.fetch_sub(1, Ordering::Release);
    }
}

impl TaskPool {
    pub fn new() -> TaskPool {
        let pool = TaskPool {
            sharing: Arc::new(Sharing {
                todo: Mutex::new(VecDeque::new()),
                condvar: Condvar::new(),
                active_tasks: AtomicUsize::new(0),
                waiting_tasks: AtomicUsize::new(0),
                closing: AtomicBool::new(false),
                handles: parking_lot::Mutex::new(Vec::new()),
            }),
        };

        for _ in 0..MIN_THREADS {
            pool.add_thread(None)
        }

        pool
    }

    /// Executes a function in a thread.
    /// If no thread is available, spawns a new one.
    /// Returns false, dropping `code` unrun, once the pool is closed.
    pub fn spawn(&self, code: Task) -> bool {
        let mut queue = self.sharing.todo.lock().unwrap();
        if self.sharing.closing.load(Ordering::Acquire) {
            drop(queue);
            return false;
        }

        if self.sharing.waiting_tasks.load(Ordering::Acquire) == 0 {
            self.add_thread(Some(code));
        } else {
            queue.push_back(code);
            self.sharing.condvar.notify_one();
        }
        true
    }

    fn add_thread(&self, initial_fn: Option<Task>) {
        let sharing = self.sharing.clone();

        let handle = thread::spawn(move || {
            let sharing = sharing;
            let _active_guard = Registration::new(&sharing.active_tasks);

            if let Some(mut f) = initial_fn {
                f();
            }

            loop {
                let mut task: Task = {
                    let mut todo = sharing.todo.lock().unwrap();

                    let task;
                    loop {
                        if sharing.closing.load(Ordering::Acquire) {
                            return;
                        }
                        if let Some(poped_task) = todo.pop_front() {
                            task = poped_task;
                            break;
                        }
                        let _waiting_guard = Registration::new(&sharing.waiting_tasks);

                        let received =
                            if sharing.active_tasks.load(Ordering::Acquire) <= MIN_THREADS {
                                todo = sharing.condvar.wait(todo).unwrap();
                                true
                            } else {
                                let (new_lock, waitres) = sharing
                                    .condvar
                                    .wait_timeout(todo, Duration::from_millis(5000))
                                    .unwrap();
                                todo = new_lock;
                                !waitres.timed_out()
                            };

                        if !received && todo.is_empty() {
                            return;
                        }
                    }

                    task
                };

                task();
            }
        });
        let mut handles = self.sharing.handles.lock();
        // Reap threads that already exited, so the retained list stays the
        // size of the live pool rather than of its history.
        let mut index = 0;
        while index < handles.len() {
            if handles[index].is_finished() {
                let _ = handles.swap_remove(index).join();
            } else {
                index += 1;
            }
        }
        handles.push(handle);
    }

    /// Stops the pool: queued tasks are dropped unrun, idle threads exit now,
    /// busy threads exit when their task returns. Does not wait.
    pub fn close(&self) {
        let dropped = {
            let mut todo = self.sharing.todo.lock().unwrap();
            self.sharing.closing.store(true, Ordering::Release);
            self.sharing.condvar.notify_all();
            std::mem::take(&mut *todo)
        };
        drop(dropped);
    }

    /// True when every thread the pool ever started has returned.
    pub fn all_finished(&self) -> bool {
        self.sharing
            .handles
            .lock()
            .iter()
            .all(JoinHandle::is_finished)
    }

    /// Hands over the handles of every thread not joined yet.
    pub fn take_handles(&self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut *self.sharing.handles.lock())
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        self.close();
    }
}
