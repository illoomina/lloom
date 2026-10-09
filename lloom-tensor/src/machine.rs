//! The machine's memory, admitted before a load allocates (`memory.md`
//! §9.10): what the host can give now, less a reserve for the OS and the UI,
//! less what other lloom processes have reserved and not yet taken.
//!
//! The per-process budget (a fraction of `recommendedMaxWorkingSetSize`,
//! `admission.rs`, §9.5c) answers *may this process hold this much*. This
//! answers *can the machine hold it now, beside everyone else*: five
//! processes each inside their own budget still summed past a 64 GB machine.
//! The registry makes the sum visible, and reserving under its lock is what
//! stops two processes from both seeing the same memory free and both
//! taking it.
//!
//! One gate at load and one at growth, never per step: a load reserves its
//! predicted peak ([`Gate::admit`]); a process that needs more than it
//! reserved widens the reservation through the same arithmetic
//! ([`Holding::grow_to`]).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

const GIB: u64 = 1 << 30;

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

/// The host's memory at one instant, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMemory {
    pub total: u64,
    pub free: u64,
    pub inactive: u64,
    pub purgeable: u64,
    pub speculative: u64,
}

impl HostMemory {
    /// Pages the kernel can hand out without paging anything out: free,
    /// inactive, purgeable and speculative.
    pub fn available(&self) -> u64 {
        self.free + self.inactive + self.purgeable + self.speculative
    }

    /// The host's reading; `None` where it cannot be read.
    pub fn read() -> Option<Self> {
        read_host()
    }

    fn describe(&self) -> String {
        format!(
            "free {:.2} + inactive {:.2} + purgeable {:.2} + speculative {:.2}",
            gb(self.free),
            gb(self.inactive),
            gb(self.purgeable),
            gb(self.speculative)
        )
    }
}

#[cfg(target_os = "macos")]
fn read_host() -> Option<HostMemory> {
    let mut total: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a u64; `total` and `len` describe exactly that
    // buffer and outlive the call.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut total as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: a zeroed vm_statistics64 is a valid out-parameter; `count` is
    // its size in integer_t units, which the kernel clamps to its own.
    let stats = unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::vm_statistics64>()
            / std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        #[allow(deprecated)]
        let host = libc::mach_host_self();
        let kr = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as libc::host_info64_t,
            &mut count,
        );
        if kr != 0 {
            return None;
        }
        stats
    };
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Some(HostMemory {
        total,
        free: stats.free_count as u64 * page,
        inactive: stats.inactive_count as u64 * page,
        purgeable: stats.purgeable_count as u64 * page,
        speculative: stats.speculative_count as u64 * page,
    })
}

/// Linux's own estimate of what can be allocated without swapping is
/// `MemAvailable`; it stands in for the four macOS terms.
#[cfg(target_os = "linux")]
fn read_host() -> Option<HostMemory> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    Some(HostMemory {
        total: field("MemTotal:")?,
        free: field("MemAvailable:")?,
        ..Default::default()
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn read_host() -> Option<HostMemory> {
    None
}

/// The physical footprint of `pid` — what it holds that the machine cannot
/// give to anyone else (`phys_footprint`, which counts its GPU allocations).
/// `None` when it cannot be read (another user's process, or gone).
pub fn footprint_of(pid: u32) -> Option<u64> {
    footprint_impl(pid)
}

#[cfg(target_os = "macos")]
fn footprint_impl(pid: u32) -> Option<u64> {
    // SAFETY: a zeroed rusage_info_v4 is a valid out-parameter, and
    // RUSAGE_INFO_V4 tells the kernel to write exactly that struct.
    unsafe {
        let mut ri: libc::rusage_info_v4 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V4,
            &mut ri as *mut libc::rusage_info_v4 as *mut libc::rusage_info_t,
        );
        (rc == 0).then_some(ri.ri_phys_footprint)
    }
}

#[cfg(target_os = "linux")]
fn footprint_impl(pid: u32) -> Option<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn footprint_impl(_pid: u32) -> Option<u64> {
    None
}

/// The reserve kept for the OS, the window server and every non-lloom
/// process: 10 % of the machine, and never under 4 GiB.
pub fn default_reserve(total: u64) -> u64 {
    (total / 10).max(4 * GIB)
}

/// A load's predicted peak, row by row: what it will hold at its largest
/// before anything grows past its floors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Peak {
    rows: Vec<(String, u64)>,
}

impl Peak {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: impl Into<String>, bytes: u64) -> Self {
        self.push(name, bytes);
        self
    }

    /// Add a row; a zero row is kept out of the report.
    pub fn push(&mut self, name: impl Into<String>, bytes: u64) {
        if bytes > 0 {
            self.rows.push((name.into(), bytes));
        }
    }

    pub fn rows(&self) -> &[(String, u64)] {
        &self.rows
    }

    pub fn total(&self) -> u64 {
        self.rows.iter().map(|(_, b)| b).sum()
    }

    /// `24.31 GB = weights 22.13 + KV floor 0.69 + …`.
    pub fn describe(&self) -> String {
        let terms: Vec<String> = self
            .rows
            .iter()
            .map(|(n, b)| format!("{n} {:.2}", gb(*b)))
            .collect();
        format!("{:.2} GB = {}", gb(self.total()), terms.join(" + "))
    }
}

/// How the gate reads the machine and what it may do about a refusal.
/// Every field defaults to the host's reading and the default registry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateConfig {
    /// Bytes kept for the OS and the UI; `None` = [`default_reserve`].
    pub reserve: Option<u64>,
    /// The registry's directory; `None` = [`Registry::default_dir`].
    pub registry: Option<PathBuf>,
    /// Bytes taken as available in place of the host's reading — for a
    /// test or a dry run that must not depend on the machine it runs on.
    pub available: Option<u64>,
    /// The machine's size in place of the host's, for the reserve default.
    pub total: Option<u64>,
    /// Admit what does not fit, reserving it anyway so others see it.
    pub force: bool,
    /// Turn the gate off: no reading, no registry.
    pub off: bool,
}

/// What the machine looked like when a reservation was decided, and the
/// arithmetic that decided it.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// What the decision was for: a load's whole peak, or a growth.
    pub need: u64,
    pub peak: Option<Peak>,
    /// The host's reading, `None` when it was injected or unreadable.
    pub host: Option<HostMemory>,
    pub available: u64,
    pub injected: bool,
    pub total: u64,
    pub reserve: u64,
    pub reserve_is_default: bool,
    /// Every other process's reservations still to be taken, by pid:
    /// `(pid, labels, reserved, footprint)`.
    pub others: Vec<Outstanding>,
    /// This process's own reservations not yet taken (a second load in one
    /// process, or a growth beside the load's reservation).
    pub mine: Outstanding,
    pub fits: bool,
    pub forced: bool,
    /// The registry the reservations were read from.
    pub registry: Option<PathBuf>,
}

/// One process's reservations against what it already holds.
#[derive(Clone, Debug, Default)]
pub struct Outstanding {
    pub pid: u32,
    pub labels: Vec<String>,
    pub reserved: u64,
    /// `None` when its footprint cannot be read; all of it is then counted.
    pub footprint: Option<u64>,
}

impl Outstanding {
    /// Reserved and not yet held.
    pub fn bytes(&self) -> u64 {
        self.reserved.saturating_sub(self.footprint.unwrap_or(0))
    }
}

impl Verdict {
    fn outstanding(&self) -> u64 {
        self.others.iter().map(Outstanding::bytes).sum::<u64>() + self.mine.bytes()
    }

    /// `available − reserve − outstanding`, signed.
    pub fn room(&self) -> i128 {
        self.available as i128 - self.reserve as i128 - self.outstanding() as i128
    }

    /// One line for a load's report.
    pub fn line(&self) -> String {
        format!(
            "[machine: need={:.2} GB available={:.2} GB{} reserve={:.2} GB other_lloom={:.2} GB in {} process(es) room={:.2} GB {}]",
            gb(self.need),
            gb(self.available),
            if self.injected { " (injected)" } else { "" },
            gb(self.reserve),
            gb(self.others.iter().map(Outstanding::bytes).sum()),
            self.others.len(),
            self.room() as f64 / 1e9,
            match (self.fits, self.forced) {
                (true, _) => "fits",
                (false, true) => "FORCED",
                (false, false) => "REFUSED",
            }
        )
    }

    /// The arithmetic, for a refusal (or a forced admission).
    pub fn describe(&self) -> String {
        let mut s = format!(
            "the machine cannot hold this now: it needs {:.2} GB and the room is {:.2} GB\n",
            gb(self.need),
            self.room() as f64 / 1e9
        );
        if let Some(p) = &self.peak {
            s.push_str(&format!("  needed     {}\n", p.describe()));
        } else {
            s.push_str(&format!("  needed     {:.2} GB of growth\n", gb(self.need)));
        }
        match (&self.host, self.injected) {
            (_, true) => s.push_str(&format!(
                "  available  {:.2} GB (injected, memory.machine.available_gb)\n",
                gb(self.available)
            )),
            (Some(h), false) => s.push_str(&format!(
                "  available  {:.2} GB = {} (the host, now)\n",
                gb(self.available),
                h.describe()
            )),
            (None, false) => s.push_str(&format!(
                "  available  {:.2} GB (the host's memory could not be read)\n",
                gb(self.available)
            )),
        }
        if self.reserve_is_default {
            s.push_str(&format!(
                "  reserve    {:.2} GB = max(4 GiB, 10 % of {:.2} GB), for macOS and the UI\n",
                gb(self.reserve),
                gb(self.total)
            ));
        } else {
            s.push_str(&format!(
                "  reserve    {:.2} GB (memory.machine.reserve_gb), for macOS and the UI\n",
                gb(self.reserve)
            ));
        }
        let others: u64 = self.others.iter().map(Outstanding::bytes).sum();
        s.push_str(&format!(
            "  other lloom processes {:.2} GB reserved and not yet held{}\n",
            gb(others),
            match &self.registry {
                Some(d) => format!(" (registry {})", d.display()),
                None => String::new(),
            }
        ));
        for o in &self.others {
            s.push_str(&format!(
                "    pid {} {}: reserved {:.2} GB, holds {}, {:.2} GB to come\n",
                o.pid,
                o.labels.join(", "),
                gb(o.reserved),
                match o.footprint {
                    Some(f) => format!("{:.2} GB", gb(f)),
                    None => "? (unreadable; all of it counted)".into(),
                },
                gb(o.bytes())
            ));
        }
        if self.mine.reserved > 0 {
            s.push_str(&format!(
                "  this process {:.2} GB reserved, holds {:.2} GB, {:.2} GB to come\n",
                gb(self.mine.reserved),
                gb(self.mine.footprint.unwrap_or(0)),
                gb(self.mine.bytes())
            ));
        } else {
            s.push_str(&format!(
                "  this process holds {:.2} GB now\n",
                gb(self.mine.footprint.unwrap_or(0))
            ));
        }
        s.push_str(&format!(
            "  room       {:.2} GB = available - reserve - reserved and not yet held\n",
            self.room() as f64 / 1e9
        ));
        if self.forced {
            s.push_str(
                "  FORCED: admitted anyway (memory.machine.force); the reservation is recorded",
            );
        } else {
            s.push_str(
                "  Nothing was allocated. Free memory, stop another lloom process, ask for less \
                 (a smaller context or prefill tile, --experts stream), or load anyway with \
                 --force (memory.machine.force = true).",
            );
        }
        s
    }
}

/// An admitted load: its reservation (held for as long as the `Arc` lives)
/// and the arithmetic it was admitted on.
#[derive(Debug)]
pub struct Admitted {
    pub holding: Option<Arc<Holding>>,
    pub verdict: Option<Verdict>,
}

impl Admitted {
    /// The gate was off: nothing read, nothing reserved.
    fn off() -> Self {
        Self {
            holding: None,
            verdict: None,
        }
    }

    /// The load line: the verdict's, or that the gate is off.
    pub fn line(&self) -> String {
        match &self.verdict {
            Some(v) => v.line(),
            None => "[machine: gate off (memory.machine.gate = false)]".into(),
        }
    }
}

/// The machine gate: one per load, built from its configuration.
#[derive(Clone, Debug, Default)]
pub struct Gate {
    cfg: GateConfig,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &GateConfig {
        &self.cfg
    }

    /// Admit a load whose predicted peak is `peak`: under the registry's
    /// lock, read the machine and every other reservation, and reserve the
    /// peak if it fits (or if forced). Refused with the arithmetic, before
    /// anything is allocated.
    pub fn admit(&self, label: &str, peak: &Peak) -> crate::Result<Admitted> {
        if self.cfg.off {
            return Ok(Admitted::off());
        }
        let registry = match self.registry() {
            Ok(r) => Some(r),
            Err(e) if self.cfg.force => {
                eprintln!("[machine: registry unavailable ({e}); forced, loading unregistered]");
                None
            }
            Err(e) => crate::bail!(
                "the machine gate's registry could not be opened: {e} \
                 (memory.machine.registry names another directory; --force loads unregistered)"
            ),
        };
        let Some(registry) = registry else {
            let v = self.decide(peak.total(), Some(peak.clone()), &[], None);
            return Ok(Admitted {
                holding: None,
                verdict: Some(v),
            });
        };
        let guard = registry.guard().map_err(crate::Error::wrap)?;
        let entries = registry.scan(&guard).map_err(crate::Error::wrap)?;
        let v = self.decide(
            peak.total(),
            Some(peak.clone()),
            &entries,
            Some(registry.dir().to_path_buf()),
        );
        if !v.fits && !v.forced {
            crate::bail!("{}", v.describe());
        }
        let holding = registry
            .create(&guard, label, peak.total(), self.cfg.clone())
            .map_err(crate::Error::wrap)?;
        drop(guard);
        Ok(Admitted {
            holding: Some(Arc::new(holding)),
            verdict: Some(v),
        })
    }

    /// The machine's room now (`available − reserve − outstanding`), read
    /// as [`Self::admit`] reads it and reserving nothing: what a caller
    /// sizing a lever after a refusal solves against. `None` with the gate
    /// off, the registry unreadable, or the host's memory unreadable.
    pub fn room(&self) -> Option<i128> {
        if self.cfg.off {
            return None;
        }
        let registry = self.registry().ok()?;
        let guard = registry.guard().ok()?;
        let entries = registry.scan(&guard).ok()?;
        let v = self.decide(0, None, &entries, None);
        (v.injected || v.host.is_some()).then(|| v.room())
    }

    fn registry(&self) -> io::Result<Registry> {
        let dir = match &self.cfg.registry {
            Some(d) => d.clone(),
            None => Registry::default_dir().ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "no HOME for the default registry")
            })?,
        };
        Registry::open(dir)
    }

    /// The arithmetic over a scan of the registry: `need` fits when
    /// `available − reserve − Σ outstanding ≥ need`, where a process's
    /// outstanding is what it reserved less what it already holds — this
    /// process's own reservations included, so a growth is counted beside
    /// the reservation it widens.
    fn decide(
        &self,
        need: u64,
        peak: Option<Peak>,
        entries: &[Entry],
        registry: Option<PathBuf>,
    ) -> Verdict {
        let host = if self.cfg.available.is_some() {
            None
        } else {
            HostMemory::read()
        };
        let available = self
            .cfg
            .available
            .or(host.map(|h| h.available()))
            .unwrap_or(0);
        let total = self
            .cfg
            .total
            .or(host.map(|h| h.total))
            .or_else(|| HostMemory::read().map(|h| h.total))
            .unwrap_or(0);
        let reserve = self.cfg.reserve.unwrap_or_else(|| default_reserve(total));
        let me = std::process::id();
        let mut others: Vec<Outstanding> = Vec::new();
        let mut mine = Outstanding {
            pid: me,
            footprint: footprint_of(me),
            ..Default::default()
        };
        for e in entries {
            if e.pid == me {
                mine.reserved += e.reserved;
                mine.labels.push(e.label.clone());
                continue;
            }
            match others.iter_mut().find(|o| o.pid == e.pid) {
                Some(o) => {
                    o.reserved += e.reserved;
                    o.labels.push(e.label.clone());
                }
                None => others.push(Outstanding {
                    pid: e.pid,
                    labels: vec![e.label.clone()],
                    reserved: e.reserved,
                    footprint: footprint_of(e.pid),
                }),
            }
        }
        let mut v = Verdict {
            need,
            peak,
            host,
            available,
            injected: self.cfg.available.is_some(),
            total,
            reserve,
            reserve_is_default: self.cfg.reserve.is_none(),
            others,
            mine,
            fits: false,
            forced: false,
            registry,
        };
        // The machine's memory unreadable and nothing injected: there is no
        // figure to refuse on, so the gate admits and the line says 0.
        let unreadable = v.host.is_none() && !v.injected;
        v.fits = unreadable || v.room() >= need as i128;
        v.forced = !v.fits && self.cfg.force;
        v
    }
}

/// One reservation in the registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub pid: u32,
    pub label: String,
    pub reserved: u64,
    /// Unix seconds when it was made.
    pub since: u64,
}

impl Entry {
    fn encode(&self) -> String {
        format!(
            "pid={}\nreserved={}\nsince={}\nlabel={}\n",
            self.pid, self.reserved, self.since, self.label
        )
    }

    fn decode(text: &str) -> Option<Self> {
        let field = |k: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
        };
        Some(Self {
            pid: field("pid")?.parse().ok()?,
            reserved: field("reserved")?.parse().ok()?,
            since: field("since")?.parse().ok()?,
            label: field("label").unwrap_or("").to_string(),
        })
    }
}

/// The machine-wide registry of lloom processes' reservations: a directory
/// under the user's cache with one file per reservation, each locked
/// (`flock`) by its owner for as long as it lives. A file whose lock can be
/// taken belongs to a process that is gone — the kernel drops a dead
/// process's locks, so this is exact where a pid test is not (pids are
/// reused) — and is reaped. Every read and write of the directory happens
/// under `registry.lock`.
#[derive(Clone, Debug)]
pub struct Registry {
    dir: PathBuf,
}

/// Held while the registry is read or changed.
pub struct RegistryGuard {
    _file: File,
}

const LOCK_FILE: &str = "registry.lock";
const ENTRY_SUFFIX: &str = ".res";

impl Registry {
    /// `$HOME/.cache/lloom/memory`.
    pub fn default_dir() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/lloom/memory"))
    }

    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Take the registry's lock, waiting for it.
    pub fn guard(&self) -> io::Result<RegistryGuard> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.dir.join(LOCK_FILE))?;
        lock(&file, true)?;
        Ok(RegistryGuard { _file: file })
    }

    /// Every live reservation, reaping the dead.
    pub fn entries(&self) -> io::Result<Vec<Entry>> {
        let g = self.guard()?;
        self.scan(&g)
    }

    fn scan(&self, _guard: &RegistryGuard) -> io::Result<Vec<Entry>> {
        let mut out = Vec::new();
        for de in fs::read_dir(&self.dir)? {
            let path = de?.path();
            if path.extension().and_then(|e| e.to_str()) != Some(&ENTRY_SUFFIX[1..]) {
                continue;
            }
            let mut f = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            if lock(&f, false).is_ok() {
                // Its owner is gone: the kernel released the lock with it.
                let _ = fs::remove_file(&path);
                continue;
            }
            let mut text = String::new();
            if f.read_to_string(&mut text).is_ok() {
                if let Some(e) = Entry::decode(&text) {
                    out.push(e);
                }
            }
        }
        out.sort_by_key(|e| (e.since, e.pid));
        Ok(out)
    }

    /// Reserve `bytes` for `label` — the caller decided it fits under
    /// `guard`. The returned holding keeps the entry's lock until it drops.
    fn create(
        &self,
        _guard: &RegistryGuard,
        label: &str,
        bytes: u64,
        cfg: GateConfig,
    ) -> io::Result<Holding> {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let pid = std::process::id();
        let since = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let name = format!(
            "{pid}-{}-{nanos}{ENTRY_SUFFIX}",
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let path = self.dir.join(name);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        lock(&file, false)?;
        let entry = Entry {
            pid,
            label: label.replace(['\n', '\r'], " "),
            reserved: bytes,
            since,
        };
        file.write_all(entry.encode().as_bytes())?;
        file.flush()?;
        Ok(Holding {
            registry: self.clone(),
            file: parking_lot::Mutex::new(file),
            path,
            entry: parking_lot::Mutex::new(entry),
            reserved: AtomicU64::new(bytes),
            gate: Gate::new(cfg),
        })
    }

    /// Reserve `bytes` with no arithmetic: the stand-in for another process
    /// that has reserved and not yet allocated (a test, or a dry run).
    pub fn reserve_unchecked(&self, label: &str, bytes: u64) -> io::Result<Holding> {
        let g = self.guard()?;
        self.create(&g, label, bytes, GateConfig::default())
    }
}

/// `flock` on `file`, exclusive; `wait` blocks, else fails at once when
/// another holds it.
fn lock(file: &File, wait: bool) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let op = libc::LOCK_EX | if wait { 0 } else { libc::LOCK_NB };
        // SAFETY: the descriptor is open for the call's duration.
        let rc = unsafe { libc::flock(file.as_raw_fd(), op) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, wait);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the registry needs flock",
        ))
    }
}

/// A process's reservation in the registry, released when it drops (and by
/// the kernel, with its lock, when the process dies).
#[derive(Debug)]
pub struct Holding {
    registry: Registry,
    file: parking_lot::Mutex<File>,
    path: PathBuf,
    entry: parking_lot::Mutex<Entry>,
    reserved: AtomicU64,
    gate: Gate,
}

impl Holding {
    /// Bytes reserved now.
    pub fn reserved(&self) -> u64 {
        self.reserved.load(Ordering::Acquire)
    }

    pub fn label(&self) -> String {
        self.entry.lock().label.clone()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Widen the reservation to `target` bytes if the machine can hold the
    /// difference now (a want past its floor: a KV grow, a cache refilled).
    /// Under the registry's lock, as a load is admitted; refused with the
    /// arithmetic, the reservation unchanged. A target at or under the
    /// reservation is admitted with nothing read.
    pub fn grow_to(&self, target: u64) -> crate::Result<Option<Verdict>> {
        let current = self.reserved();
        if target <= current {
            return Ok(None);
        }
        let need = target - current;
        let guard = self.registry.guard().map_err(crate::Error::wrap)?;
        let entries = self.registry.scan(&guard).map_err(crate::Error::wrap)?;
        let v = self.gate.decide(
            need,
            None,
            &entries,
            Some(self.registry.dir().to_path_buf()),
        );
        if !v.fits && !v.forced {
            crate::bail!("{}", v.describe());
        }
        let mut entry = self.entry.lock();
        entry.reserved = target;
        let mut f = self.file.lock();
        f.set_len(0).map_err(crate::Error::wrap)?;
        f.seek(SeekFrom::Start(0)).map_err(crate::Error::wrap)?;
        f.write_all(entry.encode().as_bytes())
            .map_err(crate::Error::wrap)?;
        f.flush().map_err(crate::Error::wrap)?;
        self.reserved.store(target, Ordering::Release);
        drop(guard);
        Ok(Some(v))
    }
}

impl Drop for Holding {
    fn drop(&mut self) {
        // Removed under the registry's lock so no scan reads it half-gone;
        // the lock on the entry itself goes with the file.
        let _g = self.registry.guard();
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_registry(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "lloom-machine-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn gate(dir: &Path, available: u64, reserve: u64) -> Gate {
        Gate::new(GateConfig {
            reserve: Some(reserve),
            registry: Some(dir.to_path_buf()),
            available: Some(available),
            total: Some(64 * GIB),
            ..Default::default()
        })
    }

    #[test]
    fn the_default_reserve_is_a_tenth_of_the_machine_and_never_under_4_gib() {
        assert_eq!(default_reserve(64 * GIB), 64 * GIB / 10);
        assert_eq!(default_reserve(16 * GIB), 4 * GIB);
        assert_eq!(default_reserve(0), 4 * GIB);
    }

    #[test]
    fn a_peak_sums_its_rows_and_names_them() {
        let p = Peak::new()
            .with("weights", 3_000_000_000)
            .with("KV floor", 500_000_000)
            .with("empty", 0);
        assert_eq!(p.total(), 3_500_000_000);
        assert_eq!(p.rows().len(), 2, "a zero row is kept out of the report");
        assert_eq!(p.describe(), "3.50 GB = weights 3.00 + KV floor 0.50");
    }

    /// The machine's room, and nothing else, decides: a load that fits is
    /// admitted and reserved; one that does not is refused with every term
    /// of the arithmetic, and reserves nothing.
    #[test]
    fn a_load_the_room_cannot_hold_is_refused_with_the_arithmetic_and_reserves_nothing() {
        let dir = tmp_registry("room");
        let g = gate(&dir, 20 * GIB, 4 * GIB);
        let peak = Peak::new().with("weights", 10 * GIB);
        let a = g.admit("small", &peak).unwrap();
        let held = a.holding.expect("a fit reserves");
        assert_eq!(held.reserved(), 10 * GIB);
        assert!(a.verdict.unwrap().fits);
        drop(held);
        assert!(Registry::open(&dir).unwrap().entries().unwrap().is_empty());

        let big = Peak::new().with("weights", 17 * GIB);
        let e = g.admit("big", &big).unwrap_err().to_string();
        for want in [
            "it needs 18.25 GB and the room is 17.18 GB",
            "needed     18.25 GB = weights 18.25",
            "available  21.47 GB (injected",
            "reserve    4.29 GB",
            "Nothing was allocated",
            "--force",
        ] {
            assert!(e.contains(want), "{want:?} missing from:\n{e}");
        }
        assert!(
            Registry::open(&dir).unwrap().entries().unwrap().is_empty(),
            "a refusal reserves nothing"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Another process's reservation that it has not yet allocated is
    /// subtracted from the room — the case two processes both seeing the
    /// same memory free would otherwise race on.
    #[test]
    fn another_reservation_not_yet_held_takes_the_room() {
        let dir = tmp_registry("other");
        let g = gate(&dir, 40 * GIB, 4 * GIB);
        let peak = Peak::new().with("weights", 25 * GIB);
        let first = g.admit("first", &peak).unwrap();
        assert!(first.holding.is_some());
        // The second identical load: the first's 25 GiB are reserved and (in
        // this test process) almost none of it is held.
        let e = g.admit("second", &peak).unwrap_err().to_string();
        assert!(e.contains("this process 26.84 GB reserved"), "{e}");
        drop(first);
        assert!(g.admit("second", &peak).unwrap().holding.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Two threads racing for room for one: the registry's lock admits
    /// exactly one.
    #[test]
    fn two_racing_loads_with_room_for_one_admit_exactly_one() {
        let dir = tmp_registry("race");
        let g = gate(&dir, 30 * GIB, 4 * GIB);
        let peak = Peak::new().with("weights", 20 * GIB);
        let results: Vec<bool> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..2)
                .map(|i| {
                    let (g, peak) = (g.clone(), peak.clone());
                    s.spawn(move || {
                        let r = g.admit(&format!("racer {i}"), &peak);
                        // Hold it until both have decided.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        r.is_ok()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(results.iter().filter(|ok| **ok).count(), 1, "{results:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// An entry whose owner is gone is reaped by the next reader: its lock
    /// is free. A live entry's is not, even read from the same process.
    #[test]
    fn a_dead_owners_entry_is_reaped_and_a_live_one_is_not() {
        let dir = tmp_registry("reap");
        let reg = Registry::open(&dir).unwrap();
        let live = reg.reserve_unchecked("live", 5 * GIB).unwrap();
        // A dead process's file: written, never locked.
        fs::write(
            dir.join(format!("999999-0-0{ENTRY_SUFFIX}")),
            Entry {
                pid: 999_999,
                label: "dead".into(),
                reserved: 30 * GIB,
                since: 0,
            }
            .encode(),
        )
        .unwrap();
        let entries = reg.entries().unwrap();
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert_eq!(entries[0].label, "live");
        assert!(!dir.join(format!("999999-0-0{ENTRY_SUFFIX}")).exists());
        drop(live);
        assert!(reg.entries().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Growth past the reservation passes the same arithmetic: admitted
    /// while there is room and recorded for the others to see, refused once
    /// there is none, the reservation left as it was.
    #[test]
    fn growth_past_the_reservation_is_admitted_on_the_same_arithmetic() {
        let dir = tmp_registry("grow");
        let g = gate(&dir, 20 * GIB, 4 * GIB);
        let a = g
            .admit("grower", &Peak::new().with("weights", 8 * GIB))
            .unwrap();
        let h = a.holding.unwrap();
        assert!(
            h.grow_to(6 * GIB).unwrap().is_none(),
            "inside: nothing read"
        );
        // Room 16 GiB less the 8 reserved: 4 more fit, then 6 more do not.
        assert!(h.grow_to(12 * GIB).unwrap().unwrap().fits);
        assert_eq!(h.reserved(), 12 * GIB);
        let on_disk = Registry::open(&dir).unwrap().entries().unwrap();
        assert_eq!(on_disk[0].reserved, 12 * GIB, "the others see the growth");
        let e = h.grow_to(18 * GIB).unwrap_err().to_string();
        assert!(e.contains("6.44 GB of growth"), "{e}");
        assert_eq!(h.reserved(), 12 * GIB);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn force_admits_what_does_not_fit_and_still_reserves_it() {
        let dir = tmp_registry("force");
        let mut cfg = gate(&dir, 10 * GIB, 4 * GIB).config().clone();
        cfg.force = true;
        let a = Gate::new(cfg)
            .admit("forced", &Peak::new().with("weights", 20 * GIB))
            .unwrap();
        let v = a.verdict.unwrap();
        assert!(!v.fits && v.forced);
        assert!(v.line().contains("FORCED"));
        assert_eq!(a.holding.unwrap().reserved(), 20 * GIB);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_host_is_readable_here_and_its_terms_sum() {
        if let Some(h) = HostMemory::read() {
            assert!(h.total > 0);
            assert!(h.available() <= h.total, "{h:?}");
            assert!(footprint_of(std::process::id()).is_some_and(|f| f > 0));
        }
    }
}
