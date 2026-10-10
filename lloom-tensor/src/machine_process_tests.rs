//! CPU-only, numeric reservations: no tensor, model, or Metal device.
//! Helpers are this lib-test executable with one exact ignored filter.
//! IPC deadlines + an independent helper watchdog bound even a stuck flock.
use super::*;
use std::io::{BufRead, BufReader};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};

const IPC: Duration = Duration::from_secs(5);
const HELPER: &str = "machine::process_tests::registry_process_helper";

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        // Short enough for sockaddr_un even on macOS; logs live in the
        // invocation's persistent artifact, not in this disposable registry.
        let p = PathBuf::from("/tmp").join(format!(
            "l952-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Worker {
    child: Child,
    socket: Option<BufReader<UnixStream>>,
    reaped: bool,
}
impl Worker {
    fn start(dir: &Path, name: &str) -> Self {
        let socket = dir.join(name);
        let listener = UnixListener::bind(&socket).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                HELPER,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("LLOOM_MACHINE_TEST_SOCKET", &socket)
            .env("LLOOM_MACHINE_TEST_REGISTRY", dir);
        eprintln!("helper invocation: {command:?}");
        let mut w = Self {
            child: command.spawn().unwrap(),
            socket: None,
            reaped: false,
        };
        let deadline = Instant::now() + IPC;
        let stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "helper connect deadline");
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("helper accept: {e}"),
            }
        };
        // Darwin may inherit O_NONBLOCK from the listening socket. The
        // connected protocol uses blocking reads bounded by SO_RCVTIMEO.
        stream.set_nonblocking(false).unwrap();
        stream.set_read_timeout(Some(IPC)).unwrap();
        stream.set_write_timeout(Some(IPC)).unwrap();
        w.socket = Some(BufReader::new(stream));
        assert_eq!(w.recv(), "READY");
        w
    }
    fn send(&mut self, s: &str) {
        writeln!(self.socket.as_mut().unwrap().get_mut(), "{s}").unwrap();
    }
    fn recv(&mut self) -> String {
        let mut line = String::new();
        assert!(self.socket.as_mut().unwrap().read_line(&mut line).unwrap() > 0);
        let line = line.trim().to_owned();
        eprintln!("helper {}: {line}", self.child.id());
        line
    }
    fn wait(&mut self) -> ExitStatus {
        let deadline = Instant::now() + IPC;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                self.reaped = true;
                eprintln!("helper {} reaped: {status}", self.child.id());
                return status;
            }
            assert!(Instant::now() < deadline, "helper exit deadline");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn finish(mut self) {
        self.send("exit");
        assert!(self.wait().success());
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.socket.take(); // EOF ends an idle helper, including during unwind.
        if !self.reaped {
            // Only our owned Child, never a PID obtained from the registry.
            let _ = self.child.kill();
            let status = self.child.wait();
            eprintln!("helper {} RAII cleanup: {status:?}", self.child.id());
        }
    }
}

#[test]
#[ignore = "subprocess protocol helper; invoked only by bounded parent tests"]
fn registry_process_helper() {
    // Also bounds blocking flock/drop paths, independently of the parent.
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(15));
        eprintln!("helper lifetime exceeded 15s");
        std::process::exit(124);
    });
    let path = std::env::var_os("LLOOM_MACHINE_TEST_SOCKET").expect("parent socket");
    let stream = UnixStream::connect(path).unwrap();
    stream.set_read_timeout(Some(IPC)).unwrap();
    stream.set_write_timeout(Some(IPC)).unwrap();
    let mut io = BufReader::new(stream);
    let gate = Gate::new(GateConfig {
        registry: Some(
            std::env::var_os("LLOOM_MACHINE_TEST_REGISTRY")
                .unwrap()
                .into(),
        ),
        available: Some(30 * GIB),
        total: Some(64 * GIB),
        reserve: Some(4 * GIB),
        ..Default::default()
    });
    let mut holding: Option<Arc<Holding>> = None;
    writeln!(io.get_mut(), "READY").unwrap();
    loop {
        let mut line = String::new();
        if io.read_line(&mut line).unwrap() == 0 {
            break; // A dead parent must not leave a waiting reservation.
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["exit"] => break,
            ["load", n] => {
                assert!(holding.is_none());
                let bytes: u64 = n.parse().unwrap();
                match gate.admit(
                    "CPU numeric helper",
                    &Peak::new().with("numeric only", bytes),
                ) {
                    Ok(a) => {
                        holding = a.holding;
                        writeln!(io.get_mut(), "ADMITTED").unwrap();
                    }
                    Err(e) => {
                        assert!(
                            e.to_string().contains("the machine cannot hold this now"),
                            "{e}"
                        );
                        writeln!(io.get_mut(), "REFUSED").unwrap();
                    }
                }
            }
            ["grow", n] => {
                let h = holding.as_ref().unwrap();
                let response = match h.grow_to(n.parse().unwrap()) {
                    Ok(_) => "GROWN",
                    Err(e) => {
                        assert!(
                            e.to_string().contains("the machine cannot hold this now"),
                            "{e}"
                        );
                        "REFUSED"
                    }
                };
                writeln!(io.get_mut(), "{response} {}", h.reserved()).unwrap();
            }
            _ => panic!("unknown helper command: {line:?}"),
        }
    }
    drop(holding);
}

#[test]
fn two_process_loads_cannot_both_take_the_same_room() {
    let dir = Directory::new();
    let mut a = Worker::start(&dir.0, "a.sock");
    let mut b = Worker::start(&dir.0, "b.sock");
    // Both commands in flight before observing either verdict.
    a.send(&format!("load {}", 16 * GIB));
    b.send(&format!("load {}", 16 * GIB));
    let replies = [a.recv(), b.recv()];
    assert_eq!(replies.iter().filter(|s| *s == "ADMITTED").count(), 1);
    assert_eq!(replies.iter().filter(|s| *s == "REFUSED").count(), 1);
    let registry = Registry::open(&dir.0).unwrap();
    assert_eq!(registry.entries().unwrap().len(), 1);
    a.finish();
    b.finish();
    assert!(registry.entries().unwrap().is_empty());
}

#[test]
fn two_process_growths_are_atomic_and_refusal_preserves_the_old_amount() {
    let dir = Directory::new();
    let mut a = Worker::start(&dir.0, "a.sock");
    let mut b = Worker::start(&dir.0, "b.sock");
    for w in [&mut a, &mut b] {
        w.send(&format!("load {}", 8 * GIB));
        assert_eq!(w.recv(), "ADMITTED");
    }
    a.send(&format!("grow {}", 16 * GIB));
    b.send(&format!("grow {}", 16 * GIB));
    let mut replies = [a.recv(), b.recv()];
    replies.sort();
    assert_eq!(
        replies,
        [
            format!("GROWN {}", 16 * GIB),
            format!("REFUSED {}", 8 * GIB)
        ]
    );
    let registry = Registry::open(&dir.0).unwrap();
    let entries = registry.entries().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries.iter().map(|e| e.reserved).sum::<u64>(), 24 * GIB);
    a.finish();
    b.finish();
    assert!(registry.entries().unwrap().is_empty());
}

#[test]
fn killed_owner_is_released_and_live_pid_does_not_keep_an_unlocked_entry_alive() {
    use std::os::unix::process::ExitStatusExt;
    let dir = Directory::new();
    let mut owner = Worker::start(&dir.0, "owner.sock");
    owner.send(&format!("load {}", 16 * GIB));
    assert_eq!(owner.recv(), "ADMITTED");
    let registry = Registry::open(&dir.0).unwrap();
    let live = registry.entries().unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].pid, owner.child.id());
    let stale = dir.0.join(format!("{}-stale.res", owner.child.id()));
    fs::write(&stale, live[0].encode()).unwrap();
    assert_eq!(
        registry.entries().unwrap(),
        live,
        "unlocked stale PID entry is reaped; live locked entry stays"
    );
    assert!(!stale.exists());
    owner.child.kill().unwrap();
    assert_eq!(owner.wait().signal(), Some(libc::SIGKILL));
    assert!(
        registry.entries().unwrap().is_empty(),
        "kernel released the dead owner's flock"
    );
    let mut next = Worker::start(&dir.0, "next.sock");
    next.send(&format!("load {}", 16 * GIB));
    assert_eq!(next.recv(), "ADMITTED");
    next.finish();
    assert!(registry.entries().unwrap().is_empty());
}

#[test]
fn parent_eof_exits_the_helper_and_releases_its_reservation() {
    let dir = Directory::new();
    let mut worker = Worker::start(&dir.0, "eof.sock");
    worker.send(&format!("load {}", 16 * GIB));
    assert_eq!(worker.recv(), "ADMITTED");
    worker
        .socket
        .as_mut()
        .unwrap()
        .get_mut()
        .shutdown(Shutdown::Write)
        .unwrap();
    assert!(worker.wait().success());
    assert!(Registry::open(&dir.0)
        .unwrap()
        .entries()
        .unwrap()
        .is_empty());
}
