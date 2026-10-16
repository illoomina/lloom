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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

const GIB: u64 = 1 << 30;

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

/// Bytes as `1536000000 B (1.54 GB)`: exact for the arithmetic, rounded for
/// the eye. A boundary refusal is decided on bytes, and two figures that
/// round alike can still differ by them.
fn bytes_gb(b: u64) -> String {
    format!("{b} B ({:.2} GB)", gb(b))
}

fn signed_bytes_gb(b: i128) -> String {
    format!("{b} B ({:.2} GB)", b as f64 / 1e9)
}

/// The host's memory at one instant, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMemory {
    pub total: u64,
    pub free: u64,
    pub inactive: u64,
    pub purgeable: u64,
    pub speculative: u64,
    /// The memory controller's limit on this process's control group, when
    /// one binds below the host (Linux): a container's `--memory`. The
    /// host's terms above say nothing about it.
    pub container: Option<ContainerMemory>,
}

/// A control group's memory limit and what the group holds against it.
/// Past the limit the kernel reclaims the group's pages and then kills one
/// of its processes, whatever the host has free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContainerMemory {
    /// `memory.max` (v2) or `memory.limit_in_bytes` (v1).
    pub limit: u64,
    /// `memory.current` (v2) or `memory.usage_in_bytes` (v1): every page the
    /// group is charged for, its page cache included.
    pub usage: u64,
    /// The group's inactive file pages (`memory.stat`): cache the kernel
    /// reclaims before it fails a charge.
    pub reclaimable: u64,
}

impl ContainerMemory {
    /// What the group can still be charged without reclaiming more than its
    /// inactive cache: `limit − (usage − reclaimable)`.
    pub fn available(&self) -> u64 {
        self.limit
            .saturating_sub(self.usage.saturating_sub(self.reclaimable))
    }

    /// The reserve kept inside the group for its other processes' growth:
    /// a tenth of its limit. The host's 4 GiB floor protects the OS and the
    /// UI, which live outside the group and are the host bound's concern.
    pub fn reserve(&self) -> u64 {
        self.limit / 10
    }

    /// The signed room left after this group's own reserve. Ancestors must
    /// be compared on this term: a larger limit also keeps a larger reserve.
    fn room(&self) -> i128 {
        self.available() as i128 - self.reserve() as i128
    }

    fn describe(&self) -> String {
        format!(
            "limit {:.2} - (usage {:.2} - inactive file {:.2})",
            gb(self.limit),
            gb(self.usage),
            gb(self.reclaimable)
        )
    }
}

/// Where this process's memory controller lives, from `/proc/self/cgroup`.
#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
enum CgroupPath {
    /// The unified hierarchy: `0::<path>`.
    V2(String),
    /// The v1 `memory` controller: `<id>:<...,memory,...>:<path>`.
    V1(String),
}

#[cfg(any(target_os = "linux", test))]
fn parse_proc_cgroup(text: &str) -> Option<CgroupPath> {
    let mut v2 = None;
    for line in text.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if controllers.split(',').any(|c| c == "memory") {
            return Some(CgroupPath::V1(path.to_string()));
        }
        if id == "0" && controllers.is_empty() {
            v2 = Some(CgroupPath::V2(path.to_string()));
        }
    }
    v2
}

/// The tightest post-reserve room over this process's control group and its
/// ancestors, as `read` sees the files: `None` when no level sets one, or
/// when every limit is at or above the host's `total` (it binds nothing the
/// host bound does not). The walk stops at the mount: inside a container's
/// cgroup namespace its own group is the mount's root.
#[cfg(any(target_os = "linux", test))]
fn container_memory_from(
    proc_cgroup: &str,
    total: u64,
    read: impl Fn(&Path) -> Option<String>,
) -> Option<ContainerMemory> {
    let num = |p: &Path| read(p).and_then(|s| s.trim().parse::<u64>().ok());
    let stat = |p: &Path, keys: &[&str]| {
        let text = read(p)?;
        keys.iter().find_map(|k| {
            text.lines().find_map(|l| {
                let (name, v) = l.split_once(' ')?;
                (name == *k).then(|| v.trim().parse::<u64>().ok()).flatten()
            })
        })
    };
    let (mount, rel, v2) = match parse_proc_cgroup(proc_cgroup)? {
        CgroupPath::V2(p) => (PathBuf::from("/sys/fs/cgroup"), p, true),
        CgroupPath::V1(p) => (PathBuf::from("/sys/fs/cgroup/memory"), p, false),
    };
    let (limit_f, usage_f, stat_keys): (&str, &str, &[&str]) = if v2 {
        ("memory.max", "memory.current", &["inactive_file"])
    } else {
        (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            &["total_inactive_file", "inactive_file"],
        )
    };
    let mut dirs = Vec::new();
    let mut dir = mount.join(rel.trim_start_matches('/'));
    while dir.starts_with(&mount) {
        dirs.push(dir.clone());
        if dir == mount || !dir.pop() {
            break;
        }
    }
    // A namespaced container sees its own group at the mount, under a path
    // that names the host's view of it.
    if !dirs.contains(&mount) {
        dirs.push(mount.clone());
    }
    dirs.iter()
        .filter_map(|d| {
            // v2 writes `max` for no limit, which does not parse.
            let limit = num(&d.join(limit_f))?;
            if total > 0 && limit >= total {
                return None;
            }
            Some(ContainerMemory {
                limit,
                usage: num(&d.join(usage_f))?,
                reclaimable: stat(&d.join("memory.stat"), stat_keys).unwrap_or(0),
            })
        })
        .min_by_key(ContainerMemory::room)
}

impl HostMemory {
    /// Pages the kernel can hand out without paging anything out: free,
    /// inactive, purgeable and speculative.
    /// Saturates for reporting; admission rejects an overflowing sum.
    pub fn available(&self) -> u64 {
        self.checked_available().unwrap_or(u64::MAX)
    }

    // Admission uses this checked form, never the display ceiling above.
    fn checked_available(&self) -> Option<u64> {
        self.free
            .checked_add(self.inactive)?
            .checked_add(self.purgeable)?
            .checked_add(self.speculative)
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
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    if page == 0 {
        return None;
    }
    Some(HostMemory {
        total,
        free: (stats.free_count as u64).checked_mul(page)?,
        inactive: (stats.inactive_count as u64).checked_mul(page)?,
        purgeable: (stats.purgeable_count as u64).checked_mul(page)?,
        speculative: (stats.speculative_count as u64).checked_mul(page)?,
        container: None,
    })
}

/// Linux's own estimate of what can be allocated without swapping is
/// `MemAvailable`; it stands in for the four macOS terms. It is the host's:
/// inside a control group with a memory limit (a container's `--memory`)
/// the group's own room is read beside it.
#[cfg(target_os = "linux")]
fn read_host() -> Option<HostMemory> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .and_then(|kb| kb.checked_mul(1024))
    };
    let total = field("MemTotal:")?;
    let container = fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|c| container_memory_from(&c, total, |p| fs::read_to_string(p).ok()));
    Some(HostMemory {
        total,
        free: field("MemAvailable:")?,
        container,
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
        .and_then(|kb| kb.checked_mul(1024))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn footprint_impl(_pid: u32) -> Option<u64> {
    None
}

/// The default reserve's floor: what the OS and the UI keep on any machine.
pub const RESERVE_FLOOR: u64 = 4 * GIB;

/// Which term the reserve kept for the OS, the window server and every
/// non-lloom process came from (`memory.md` §9.10): the one that was larger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReserveBasis {
    /// `memory.machine.reserve_gb`.
    Configured,
    /// `hw.memsize − recommendedMaxWorkingSetSize`: the share of the machine
    /// the OS keeps back from the GPU's working set, at or over the floor.
    WorkingSet { working_set: u64 },
    /// 10 % of the machine, where there is no Metal device to ask, at or
    /// over the floor.
    TenthOfMachine,
    /// [`RESERVE_FLOOR`], because the default's own term — `term` bytes, the
    /// working set's difference when `working_set` is known, else 10 % —
    /// was smaller.
    Floor { term: u64, working_set: Option<u64> },
}

impl ReserveBasis {
    fn tag(self) -> String {
        match self {
            Self::Configured => "memory.machine.reserve_gb".into(),
            Self::WorkingSet { .. } => "memsize-working_set".into(),
            Self::TenthOfMachine => "10 %".into(),
            Self::Floor { term, working_set } => format!(
                "4 GiB floor; {} = {:.2} GB",
                if working_set.is_some() {
                    "memsize-working_set"
                } else {
                    "10 %"
                },
                gb(term)
            ),
        }
    }
}

/// The default reserve on a machine of `total` bytes: what the OS keeps back
/// from the device's working set when there is one, else 10 %; never under
/// [`RESERVE_FLOOR`]. The basis names the term that won.
pub fn default_reserve(total: u64, working_set: Option<u64>) -> (u64, ReserveBasis) {
    let (term, basis) = match working_set {
        Some(ws) => (
            total.saturating_sub(ws),
            ReserveBasis::WorkingSet { working_set: ws },
        ),
        None => (total / 10, ReserveBasis::TenthOfMachine),
    };
    if term >= RESERVE_FLOOR {
        (term, basis)
    } else {
        (RESERVE_FLOOR, ReserveBasis::Floor { term, working_set })
    }
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

    /// The total, with a reporting ceiling of `u64::MAX`. Admission checks
    /// the exact sum and refuses overflow, including under `--force`.
    pub fn total(&self) -> u64 {
        self.checked_total().unwrap_or(u64::MAX)
    }

    /// Exact sum, or None when the rows cannot be represented as u64 bytes.
    /// Unlike `total`, this is suitable for admission before any override.
    pub fn checked_total(&self) -> Option<u64> {
        self.rows
            .iter()
            .try_fold(0u64, |sum, (_, b)| sum.checked_add(*b))
    }

    /// `24.31 GB = weights 22.13 + KV floor 0.69 + …`.
    pub fn describe(&self) -> String {
        let terms: Vec<String> = self
            .rows
            .iter()
            .map(|(n, b)| format!("{n} {:.2}", gb(*b)))
            .collect();
        match self.checked_total() {
            Some(total) => format!("{:.2} GB = {}", gb(total), terms.join(" + ")),
            None => format!("overflow (exceeds u64 bytes) = {}", terms.join(" + ")),
        }
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
    /// The device's `recommendedMaxWorkingSetSize`, for the reserve default;
    /// `None` where there is no Metal device.
    pub working_set: Option<u64>,
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
    pub reserve_basis: ReserveBasis,
    /// The control group's limit and holding when one binds below the host
    /// (read with the host; `None` when injected): its own room, less its
    /// own reserve ([`ContainerMemory::reserve`]), is a second bound.
    pub container: Option<ContainerMemory>,
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
    fn other_bytes(&self) -> i128 {
        self.others
            .iter()
            .fold(0i128, |sum, o| sum.saturating_add(o.bytes() as i128))
    }

    fn outstanding(&self) -> i128 {
        self.other_bytes().saturating_add(self.mine.bytes() as i128)
    }

    /// The host's `available − reserve`, signed.
    fn host_free(&self) -> i128 {
        self.available as i128 - self.reserve as i128
    }

    /// The control group's `available − reserve`, signed, when one binds.
    fn container_free(&self) -> Option<i128> {
        self.container.map(|c| c.room())
    }

    /// Whether the control group, not the host, sets the room.
    pub fn container_binds(&self) -> bool {
        self.container_free().is_some_and(|c| c < self.host_free())
    }

    /// `min(host available − reserve, group available − group reserve) −
    /// outstanding`, signed.
    pub fn room(&self) -> i128 {
        let free = match self.container_free() {
            Some(c) => c.min(self.host_free()),
            None => self.host_free(),
        };
        free.saturating_sub(self.outstanding())
    }

    /// `room − need`, signed: what a fit leaves, or a refusal lacks.
    pub fn margin(&self) -> i128 {
        self.room().saturating_sub(self.need as i128)
    }

    /// One line for a load's report.
    pub fn line(&self) -> String {
        if self.host.is_none() && !self.injected {
            return format!(
                "[machine: need={:.2} GB available=UNKNOWN room=UNKNOWN {}; host observation failed twice]",
                gb(self.need),
                if self.forced { "FORCED (safety unchecked)" } else { "REFUSED" },
            );
        }
        let container = match self.container {
            Some(c) => format!(
                " container={:.2} GB of a {:.2} GB limit reserve={:.2} GB (10 % of the limit){}",
                gb(c.available()),
                gb(c.limit),
                gb(c.reserve()),
                if self.container_binds() { " binds" } else { "" }
            ),
            None => String::new(),
        };
        format!(
            "[machine: need={:.2} GB available={:.2} GB{} reserve={:.2} GB ({}){container} other_lloom={:.2} GB in {} process(es) room={:.2} GB margin={} B {}]",
            gb(self.need),
            gb(self.available),
            if self.injected { " (injected)" } else { "" },
            gb(self.reserve),
            self.reserve_basis.tag(),
            self.other_bytes() as f64 / 1e9,
            self.others.len(),
            self.room() as f64 / 1e9,
            self.margin(),
            match (self.fits, self.forced) {
                (true, _) => "fits",
                (false, true) => "FORCED",
                (false, false) => "REFUSED",
            }
        )
    }

    /// The arithmetic, for a refusal (or a forced admission): every term in
    /// bytes beside its rounded figure, so a refusal at the boundary reads
    /// as the bytes it was decided on.
    pub fn describe(&self) -> String {
        let unknown = self.host.is_none() && !self.injected;
        let mut s = if unknown {
            "the host's available memory could not be read after two attempts; \
             admission cannot establish room\n"
                .to_string()
        } else {
            format!(
                "the machine cannot hold this now: it needs {} and the room is {}, {} B {}\n",
                bytes_gb(self.need),
                signed_bytes_gb(self.room()),
                self.margin().unsigned_abs(),
                if self.margin() < 0 {
                    "short"
                } else {
                    "to spare"
                }
            )
        };
        if let Some(p) = &self.peak {
            s.push_str(&format!("  needed     {}\n", p.describe()));
        } else {
            s.push_str(&format!("  needed     {} of growth\n", bytes_gb(self.need)));
        }
        match (&self.host, self.injected) {
            (_, true) => s.push_str(&format!(
                "  available  {} (injected, memory.machine.available_gb)\n",
                bytes_gb(self.available)
            )),
            (Some(h), false) => s.push_str(&format!(
                "  available  {} = {} (the host, now)\n",
                bytes_gb(self.available),
                h.describe()
            )),
            (None, false) => s.push_str("  available  UNKNOWN (not a zero-byte measurement)\n"),
        }
        let reserve = bytes_gb(self.reserve);
        match self.reserve_basis {
            ReserveBasis::WorkingSet { working_set } => s.push_str(&format!(
                "  reserve    {reserve} = {:.2} GB memory - {:.2} GB GPU working set (over the 4 GiB floor), \
                 for macOS and the UI\n",
                gb(self.total),
                gb(working_set)
            )),
            ReserveBasis::TenthOfMachine => s.push_str(&format!(
                "  reserve    {reserve} = 10 % of {:.2} GB (no Metal working set; over the 4 GiB floor), \
                 for the OS and the UI\n",
                gb(self.total)
            )),
            ReserveBasis::Floor { term, working_set } => s.push_str(&format!(
                "  reserve    {reserve} = the 4 GiB floor, over {} = {:.2} GB, for the OS and the UI\n",
                match working_set {
                    Some(ws) => format!(
                        "{:.2} GB memory - {:.2} GB GPU working set",
                        gb(self.total),
                        gb(ws)
                    ),
                    None => format!("10 % of {:.2} GB (no Metal working set)", gb(self.total)),
                },
                gb(term)
            )),
            ReserveBasis::Configured => s.push_str(&format!(
                "  reserve    {reserve} (memory.machine.reserve_gb), for the OS and the UI\n"
            )),
        }
        if let Some(c) = self.container {
            s.push_str(&format!(
                "  container  {} = {} (this process's control group, now); reserve {} = 10 % of the limit\n",
                bytes_gb(c.available()),
                c.describe(),
                bytes_gb(c.reserve())
            ));
        }
        s.push_str(&format!(
            "  other lloom processes {} reserved and not yet held{}\n",
            signed_bytes_gb(self.other_bytes()),
            match &self.registry {
                Some(d) => format!(" (registry {})", d.display()),
                None => String::new(),
            }
        ));
        for o in &self.others {
            s.push_str(&format!(
                "    pid {} {}: reserved {:.2} GB, holds {}, {} to come\n",
                o.pid,
                o.labels.join(", "),
                gb(o.reserved),
                match o.footprint {
                    Some(f) => format!("{:.2} GB", gb(f)),
                    None => "? (unreadable; all of it counted)".into(),
                },
                bytes_gb(o.bytes())
            ));
        }
        if self.mine.reserved > 0 {
            s.push_str(&format!(
                "  this process {:.2} GB reserved, holds {:.2} GB, {} to come\n",
                gb(self.mine.reserved),
                gb(self.mine.footprint.unwrap_or(0)),
                bytes_gb(self.mine.bytes())
            ));
        } else {
            s.push_str(&format!(
                "  this process holds {:.2} GB now\n",
                gb(self.mine.footprint.unwrap_or(0))
            ));
        }
        if unknown {
            s.push_str("  room       UNKNOWN; retry on a host with a supported memory reader, \
                        or explicitly use --force (safety unchecked) or memory.machine.gate=false\n");
        } else if self.container.is_some() {
            s.push_str(&format!(
                "  room       {} = min(available - reserve, container - its reserve) - reserved \
                 and not yet held; the {} binds\n",
                signed_bytes_gb(self.room()),
                if self.container_binds() {
                    "container"
                } else {
                    "host"
                }
            ));
        } else {
            s.push_str(&format!(
                "  room       {} = available - reserve - reserved and not yet held\n",
                signed_bytes_gb(self.room())
            ));
        }
        if self.forced {
            s.push_str(
                "  FORCED: admitted anyway (memory.machine.force); the reservation is recorded",
            );
            if unknown {
                s.push_str("; safety could not be checked");
            }
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

    /// This gate with the device's working set as the reserve's basis. An
    /// injected `working_set` is kept: a test's value outranks the device's.
    pub fn with_device_working_set(&self, working_set: Option<u64>) -> Self {
        let mut cfg = self.cfg.clone();
        cfg.working_set = cfg.working_set.or(working_set);
        Self { cfg }
    }

    /// Admit a load whose predicted peak is `peak`: under the registry's
    /// lock, read the machine and every other reservation, and reserve the
    /// peak if it fits (or if forced). Refused with the arithmetic, before
    /// anything is allocated.
    pub fn admit(&self, label: &str, peak: &Peak) -> crate::Result<Admitted> {
        self.admit_with(label, peak, HostMemory::read)
    }

    fn admit_with(
        &self,
        label: &str,
        peak: &Peak,
        read_host: impl FnMut() -> Option<HostMemory>,
    ) -> crate::Result<Admitted> {
        if self.cfg.off {
            return Ok(Admitted::off());
        }
        let total = peak.checked_total().ok_or_else(|| {
            crate::Error::Msg(format!(
                "machine peak arithmetic overflow: {}; reduce the requested sizes; \
                 --force cannot represent an overflowing reservation",
                peak.describe()
            ))
        })?;
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
            let v = self.decide_with(total, Some(peak.clone()), &[], None, read_host)?;
            return Ok(Admitted {
                holding: None,
                verdict: Some(v),
            });
        };
        let guard = registry.guard().map_err(crate::Error::wrap)?;
        let entries = registry.scan(&guard).map_err(crate::Error::wrap)?;
        let v = self.decide_with(
            total,
            Some(peak.clone()),
            &entries,
            Some(registry.dir().to_path_buf()),
            read_host,
        )?;
        if !v.fits && !v.forced {
            crate::bail!("{}", v.describe());
        }
        let holding = registry
            .create(&guard, label, total, self.cfg.clone())
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
        let v = self.decide(0, None, &entries, None).ok()?;
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
    ) -> crate::Result<Verdict> {
        self.decide_with(need, peak, entries, registry, HostMemory::read)
    }

    fn decide_with(
        &self,
        need: u64,
        peak: Option<Peak>,
        entries: &[Entry],
        registry: Option<PathBuf>,
        mut read_host: impl FnMut() -> Option<HostMemory>,
    ) -> crate::Result<Verdict> {
        // At most two observations, no sleep under registry.lock. Fully
        // injected budgets do not consult the host at all.
        let observation = if self.cfg.available.is_none() || self.cfg.total.is_none() {
            read_host().or_else(&mut read_host)
        } else {
            None
        };
        let host = if self.cfg.available.is_some() {
            None
        } else {
            observation
        };
        let available = match (self.cfg.available, host) {
            (Some(bytes), _) => bytes,
            (None, Some(h)) => h.checked_available().ok_or_else(|| {
                crate::Error::Msg("machine host available-byte arithmetic overflow".into())
            })?,
            (None, None) => 0,
        };
        let total = self.cfg.total.or(observation.map(|h| h.total)).unwrap_or(0);
        let (reserve, reserve_basis) = match self.cfg.reserve {
            Some(bytes) => (bytes, ReserveBasis::Configured),
            None => default_reserve(total, self.cfg.working_set),
        };
        let me = std::process::id();
        let mut others: Vec<Outstanding> = Vec::new();
        let mut mine = Outstanding {
            pid: me,
            footprint: footprint_of(me),
            ..Default::default()
        };
        for e in entries {
            if e.pid == me {
                mine.reserved = checked_reserved(mine.reserved, e)?;
                mine.labels.push(e.label.clone());
                continue;
            }
            match others.iter_mut().find(|o| o.pid == e.pid) {
                Some(o) => {
                    o.reserved = checked_reserved(o.reserved, e)?;
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
            reserve_basis,
            // An injected availability stands in for the whole reading.
            container: host.and_then(|h| h.container),
            others,
            mine,
            fits: false,
            forced: false,
            registry,
        };
        // Unknown is not a fit. An explicit force still records a
        // representable reservation, but reports that safety is unchecked.
        let unreadable = v.host.is_none() && !v.injected;
        v.fits = !unreadable && v.room() >= need as i128;
        v.forced = !v.fits && self.cfg.force;
        Ok(v)
    }
}

fn checked_reserved(previous: u64, entry: &Entry) -> crate::Result<u64> {
    previous.checked_add(entry.reserved).ok_or_else(|| {
        crate::Error::Msg(format!(
            "machine registry reservation arithmetic overflow for pid {}; \
             its owner must reduce or release reservations before admission",
            entry.pid
        ))
    })
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

/// The shared registry of participating lloom processes' reservations: a directory
/// under the user's cache with one file per reservation, each locked
/// (`flock`) by its owner for as long as it lives. A file whose lock can be
/// taken belongs to a process that is gone — the kernel drops a dead
/// process's locks, so this is exact where a pid test is not (pids are
/// reused) — and is reaped. Every read and write of the directory happens
/// under `registry.lock`.
/// Only processes using this directory participate; the default is per user,
/// not a machine-wide allocation ceiling.
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
            let open = || OpenOptions::new().read(true).write(true).open(&path);
            let mut f = open()
                .or_else(|_| open())
                .map_err(|e| registry_entry_error(&path, e.kind(), &format!("cannot open: {e}")))?;
            match lock(&f, false) {
                Ok(()) => {
                    // Its owner is gone: the kernel released the lock with it.
                    let _ = fs::remove_file(&path);
                    continue;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    return Err(registry_entry_error(
                        &path,
                        e.kind(),
                        &format!("cannot establish owner liveness: {e}"),
                    ));
                }
            }
            // All cooperating writers hold registry.lock. Reread once to
            // recover a transient read failure; persistent uncertainty is
            // not evidence of zero reserved bytes, nor permission to reap.
            out.push(
                read_live_entry(&mut f)
                    .or_else(|_| read_live_entry(&mut f))
                    .map_err(|e| {
                        registry_entry_error(
                            &path,
                            e.kind(),
                            &format!("cannot read live reservation: {e}"),
                        )
                    })?,
            );
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
            domain_bound: AtomicBool::new(false),
            gate: Gate::new(cfg),
            last_refusal: parking_lot::Mutex::new(None),
        })
    }

    /// Reserve `bytes` with no arithmetic: the stand-in for another process
    /// that has reserved and not yet allocated (a test, or a dry run).
    pub fn reserve_unchecked(&self, label: &str, bytes: u64) -> io::Result<Holding> {
        let g = self.guard()?;
        self.create(&g, label, bytes, GateConfig::default())
    }
}

fn read_live_entry(file: &mut File) -> io::Result<Entry> {
    file.seek(SeekFrom::Start(0))?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Entry::decode(&text).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid pid/reserved/since fields",
        )
    })
}

fn write_entry(file: &mut File, entry: &Entry) -> io::Result<()> {
    let text = entry.encode();
    // A partial prefix cannot contain `since` until the complete reserved
    // field has been written. Never overwrite digits in a valid old record:
    // a torn decimal overwrite can parse as a smaller reservation.
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(text.as_bytes())?;
    file.flush()
}

// Keep the inode (and its lifetime flock) unchanged. All readers hold the
// registry lock; a failed update restores the previous record before that
// lock is released. If restoration also fails, a complete required prefix
// records either the old or the larger reservation; an incomplete one is
// refused by scans until the owner repairs it or exits.
fn replace_entry(
    file: &mut File,
    previous: &Entry,
    next: &Entry,
    mut write: impl FnMut(&mut File, &Entry) -> io::Result<()>,
) -> io::Result<()> {
    match write(file, next) {
        Ok(()) => Ok(()),
        Err(update) => match write(file, previous) {
            Ok(()) => Err(io::Error::new(
                update.kind(),
                format!("reservation update failed ({update}); previous reservation restored"),
            )),
            Err(restore) => Err(io::Error::new(
                restore.kind(),
                format!(
                    "reservation update failed ({update}) and restoration failed ({restore}); \
                     registry contents are uncertain; release this holding before retrying"
                ),
            )),
        },
    }
}

fn registry_entry_error(path: &Path, kind: io::ErrorKind, detail: &str) -> io::Error {
    io::Error::new(
        kind,
        format!(
            "machine registry entry {}: {detail}; retry after its owner repairs or releases \
             the reservation (do not delete a live entry); admission cannot establish room",
            path.display()
        ),
    )
}

/// `flock` on `file`, exclusive; `wait` blocks, else fails at once when
/// another holds it.
fn lock(file: &File, wait: bool) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let op = libc::LOCK_EX | if wait { 0 } else { libc::LOCK_NB };
        loop {
            // SAFETY: the descriptor is open for the call's duration.
            if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::Interrupted {
                return Err(e);
            }
        }
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
    // One independently counted domain may use this owner-directed target.
    // Arc< Holding > aliases in public Admitted values do not grant a second
    // domain the right to count the same high-water as its own coverage.
    domain_bound: AtomicBool,
    gate: Gate,
    last_refusal: parking_lot::Mutex<Option<Verdict>>,
}

/// An exclusive internal domain binding, retained through promises/charges.
/// Not Clone. Last drop may release Holding (registry work): host paths only.
pub(crate) struct HoldingDomainBinding {
    holding: Arc<Holding>,
}

impl Drop for HoldingDomainBinding {
    fn drop(&mut self) {
        self.holding.domain_bound.store(false, Ordering::Release);
    }
}

impl Holding {
    pub(crate) fn try_bind_domain(self: &Arc<Self>) -> crate::Result<HoldingDomainBinding> {
        if self
            .domain_bound
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            crate::bail!("machine holding already belongs to an admission domain; reuse that Domain or obtain a distinct holding with Gate::admit");
        }
        Ok(HoldingDomainBinding {
            holding: self.clone(),
        })
    }

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
        self.grow_with(target, || {})
    }

    /// [`Self::grow_to`] with the refusal typed: the verdict the room could
    /// not hold, apart from a registry or host that could not be read.
    pub fn try_grow_to(&self, target: u64) -> Result<Option<Verdict>, GrowRefusal> {
        self.try_grow_with(target, || {})
    }

    /// The verdict of the last growth this holding refused, for a caller
    /// reporting the arithmetic after a typed refusal.
    pub fn last_refusal(&self) -> Option<Verdict> {
        self.last_refusal.lock().clone()
    }

    // The callback lets the concurrency test pause after the fast-path
    // snapshot, without relying on scheduling or sleeps. Production calls
    // monomorphize the empty callback away.
    fn grow_with(&self, target: u64, before_lock: impl FnOnce()) -> crate::Result<Option<Verdict>> {
        self.try_grow_with(target, before_lock)
            .map_err(|e| match e {
                GrowRefusal::Room(v) => crate::Error::Msg(v.describe()).bt(),
                GrowRefusal::Unavailable(e) => e,
            })
    }

    fn try_grow_with(
        &self,
        target: u64,
        before_lock: impl FnOnce(),
    ) -> Result<Option<Verdict>, GrowRefusal> {
        let current = self.reserved();
        if target <= current {
            return Ok(None);
        }
        before_lock();
        let unavailable = |e: io::Error| GrowRefusal::Unavailable(crate::Error::wrap(e));
        let guard = self.registry.guard().map_err(unavailable)?;
        // Another grower may have widened this holding while we waited.
        // Both the delta and the no-growth decision belong under the lock
        // that serializes the registry update, not to the fast-path read.
        let current = self.reserved();
        if target <= current {
            return Ok(None);
        }
        let need = target - current;
        let entries = self.registry.scan(&guard).map_err(unavailable)?;
        let v = self
            .gate
            .decide(
                need,
                None,
                &entries,
                Some(self.registry.dir().to_path_buf()),
            )
            .map_err(GrowRefusal::Unavailable)?;
        if !v.fits && !v.forced {
            *self.last_refusal.lock() = Some(v.clone());
            return Err(GrowRefusal::Room(v));
        }
        let mut entry = self.entry.lock();
        let mut next = entry.clone();
        next.reserved = target;
        let mut f = self.file.lock();
        replace_entry(&mut f, &entry, &next, write_entry).map_err(unavailable)?;
        *entry = next;
        self.reserved.store(target, Ordering::Release);
        drop(guard);
        Ok(Some(v))
    }
}

/// Why [`Holding::try_grow_to`] could not widen the reservation.
#[derive(Debug)]
pub enum GrowRefusal {
    /// The room could not hold the growth; the verdict carries the arithmetic.
    Room(Verdict),
    /// The registry or the host could not be read, or the arithmetic overflowed.
    Unavailable(crate::Error),
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

    // The development machine's figures: hw.memsize and its Metal device's
    // recommendedMaxWorkingSetSize (memory.md §9.10).
    const M1MAX_MEMSIZE: u64 = 68_719_476_736;
    const M1MAX_WORKING_SET: u64 = 55_662_788_608;

    #[test]
    fn the_default_reserve_is_what_the_os_keeps_from_the_working_set_and_never_under_4_gib() {
        assert_eq!(
            default_reserve(M1MAX_MEMSIZE, Some(M1MAX_WORKING_SET)),
            (
                13_056_688_128,
                ReserveBasis::WorkingSet {
                    working_set: M1MAX_WORKING_SET
                }
            )
        );
        // A working set at or past the machine's size leaves the floor.
        assert_eq!(default_reserve(16 * GIB, Some(16 * GIB)).0, 4 * GIB);
        assert_eq!(default_reserve(16 * GIB, Some(20 * GIB)).0, 4 * GIB);
        // No Metal device: a tenth of the machine, never under 4 GiB.
        assert_eq!(
            default_reserve(64 * GIB, None),
            (64 * GIB / 10, ReserveBasis::TenthOfMachine)
        );
        assert_eq!(default_reserve(16 * GIB, None).0, 4 * GIB);
        assert_eq!(default_reserve(0, None).0, 4 * GIB);
    }

    /// A small CI host's figures: 9.70 GB and no Metal device.
    /// Its tenth is 0.97 GB, so the 4 GiB floor is what the reserve is, and
    /// the line and the arithmetic say so rather than `(10 %)`.
    #[test]
    fn the_reserve_names_the_floor_when_the_floor_is_what_fired() {
        const SILVERBERRY: u64 = 9_700_000_000;
        assert_eq!(
            default_reserve(SILVERBERRY, None),
            (
                RESERVE_FLOOR,
                ReserveBasis::Floor {
                    term: 970_000_000,
                    working_set: None
                }
            )
        );
        // A working set leaving less than the floor names its own term.
        assert_eq!(
            default_reserve(16 * GIB, Some(14 * GIB)),
            (
                RESERVE_FLOOR,
                ReserveBasis::Floor {
                    term: 2 * GIB,
                    working_set: Some(14 * GIB)
                }
            )
        );
        // At the floor exactly, the term is what fired.
        assert_eq!(
            default_reserve(40 * GIB, None).1,
            ReserveBasis::TenthOfMachine
        );

        let dir = tmp_registry("floor");
        let error = Gate::new(GateConfig {
            registry: Some(dir.clone()),
            available: Some(4_000_000_000),
            total: Some(SILVERBERRY),
            ..Default::default()
        })
        .admit("floor", &Peak::new().with("load", 1))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(
                "reserve    4294967296 B (4.29 GB) = the 4 GiB floor, over 10 % of 9.70 GB \
                 (no Metal working set) = 0.97 GB"
            ),
            "{error}"
        );
        let v = Gate::new(GateConfig {
            registry: Some(dir.clone()),
            available: Some(8_850_000_000),
            total: Some(SILVERBERRY),
            ..Default::default()
        })
        .decide(0, None, &[], None)
        .unwrap();
        assert!(
            v.line()
                .contains("reserve=4.29 GB (4 GiB floor; 10 % = 0.97 GB)"),
            "{}",
            v.line()
        );
        assert!(!v.line().contains("(10 %)"), "{}", v.line());
        fs::remove_dir_all(dir).unwrap();
    }

    /// A refusal at the boundary: need and room both read 4.56 GB at two
    /// decimals and differ by one byte. The refusal prints both in bytes
    /// and the shortfall, so the rounded figures cannot read as equal; one
    /// byte less is admitted, with nothing to spare.
    #[test]
    fn a_boundary_refusal_prints_the_bytes_it_was_decided_on() {
        let dir = tmp_registry("boundary");
        let g = gate(&dir, 8_850_000_000, RESERVE_FLOOR);
        let room = 8_850_000_000 - RESERVE_FLOOR;
        let error = g
            .admit("boundary", &Peak::new().with("load", room + 1))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "it needs 4555032705 B (4.56 GB) and the room is 4555032704 B (4.56 GB), 1 B short"
            ),
            "{error}"
        );
        assert!(
            error.contains("room       4555032704 B (4.56 GB) = available - reserve"),
            "{error}"
        );
        let v = g.decide(room + 1, None, &[], None).unwrap();
        assert!(!v.fits);
        assert!(v.line().contains("margin=-1 B REFUSED"), "{}", v.line());
        let fit = g
            .admit("boundary", &Peak::new().with("load", room))
            .unwrap();
        let v = fit.verdict.as_ref().unwrap();
        assert!(v.fits && v.margin() == 0, "{}", v.line());
        assert!(v.line().contains("margin=0 B fits"), "{}", v.line());
        drop(fit);
        fs::remove_dir_all(dir).unwrap();
    }

    fn files(entries: &[(&str, &str)]) -> impl Fn(&Path) -> Option<String> {
        let map: std::collections::HashMap<PathBuf, String> = entries
            .iter()
            .map(|(p, t)| (PathBuf::from(p), t.to_string()))
            .collect();
        move |p: &Path| map.get(p).cloned()
    }

    const HOST: u64 = 34_000_000_000;

    /// cgroup v2, as Docker's `--memory 6g` leaves it: the group's limit,
    /// its charge and its inactive file cache, read at the group's own path
    /// under a parent with no limit.
    #[test]
    fn a_v2_group_limit_is_read_with_its_charge_and_reclaimable_cache() {
        let g = "/sys/fs/cgroup/system.slice/docker-abc.scope";
        let fs = files(&[
            ("/sys/fs/cgroup/system.slice/memory.max", "max\n"),
            ("/sys/fs/cgroup/system.slice/memory.current", "9000000000\n"),
            (&format!("{g}/memory.max"), "6442450944\n"),
            (&format!("{g}/memory.current"), "600000000\n"),
            (
                &format!("{g}/memory.stat"),
                "anon 400000000\nfile 200000000\nactive_file 100000000\ninactive_file 100000000\n",
            ),
        ]);
        let c = container_memory_from("0::/system.slice/docker-abc.scope\n", HOST, fs).unwrap();
        assert_eq!(
            c,
            ContainerMemory {
                limit: 6_442_450_944,
                usage: 600_000_000,
                reclaimable: 100_000_000
            }
        );
        assert_eq!(c.available(), 6_442_450_944 - 500_000_000);
        assert_eq!(c.reserve(), 644_245_094);
    }

    /// Inside the container's cgroup namespace the group is the mount's
    /// root, whether `/proc/self/cgroup` says `/` or still names the host's
    /// path; a tighter ancestor binds over a looser group; `max`, or a
    /// limit at or past the host's memory, is no limit.
    #[test]
    fn the_tightest_visible_limit_binds_and_no_limit_is_none() {
        let root = files(&[
            ("/sys/fs/cgroup/memory.max", "6442450944"),
            ("/sys/fs/cgroup/memory.current", "1000000000"),
        ]);
        for cg in ["0::/\n", "0::/docker/abc\n"] {
            assert_eq!(
                container_memory_from(cg, HOST, &root).map(|c| c.limit),
                Some(6_442_450_944),
                "{cg:?}"
            );
        }
        let nested = files(&[
            ("/sys/fs/cgroup/a/memory.max", "4000000000"),
            ("/sys/fs/cgroup/a/memory.current", "1000000000"),
            ("/sys/fs/cgroup/a/b/memory.max", "8000000000"),
            ("/sys/fs/cgroup/a/b/memory.current", "500000000"),
        ]);
        assert_eq!(
            container_memory_from("0::/a/b\n", HOST, nested).map(|c| c.limit),
            Some(4_000_000_000)
        );
        let none = files(&[
            ("/sys/fs/cgroup/memory.max", "max"),
            ("/sys/fs/cgroup/memory.current", "1000000000"),
        ]);
        assert_eq!(container_memory_from("0::/\n", HOST, none), None);
        let over = files(&[
            ("/sys/fs/cgroup/memory.max", "64000000000"),
            ("/sys/fs/cgroup/memory.current", "1000000000"),
        ]);
        assert_eq!(container_memory_from("0::/\n", HOST, over), None);
        assert_eq!(container_memory_from("", HOST, files(&[])), None);
    }

    /// Raw availability orders these groups backwards from the room the
    /// gate may admit. The ancestor's larger reserve must still bind.
    #[test]
    fn ancestor_selection_subtracts_each_groups_reserve_before_comparing() {
        for (proc, mount, limit_file, usage_file) in [
            (
                "0::/a/b\n",
                "/sys/fs/cgroup",
                "memory.max",
                "memory.current",
            ),
            (
                "4:memory:/a/b\n",
                "/sys/fs/cgroup/memory",
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            ),
        ] {
            let fs = files(&[
                (&format!("{mount}/a/{limit_file}"), "6000000000"),
                (&format!("{mount}/a/{usage_file}"), "2900000000"),
                (&format!("{mount}/a/b/{limit_file}"), "4000000000"),
                (&format!("{mount}/a/b/{usage_file}"), "1000000000"),
            ]);
            let selected = container_memory_from(proc, HOST, fs).unwrap();
            assert_eq!(selected.limit, 6_000_000_000, "{proc}");
            assert_eq!(selected.available(), 3_100_000_000);
            assert_eq!(selected.reserve(), 600_000_000);
            assert_eq!(selected.room(), 2_500_000_000);

            let gate = Gate::default();
            for need in [2_500_000_000, 2_500_000_001, 2_550_000_000] {
                let v = gate
                    .decide_with(need, None, &[], None, || {
                        Some(HostMemory {
                            total: 9_700_000_000,
                            free: 8_850_000_000,
                            container: Some(selected),
                            ..Default::default()
                        })
                    })
                    .unwrap();
                assert_eq!(v.reserve, RESERVE_FLOOR);
                assert!(v.container_binds());
                assert_eq!(v.room(), 2_500_000_000);
                assert_eq!(v.fits, need <= 2_500_000_000, "{}", v.line());
                assert_eq!(v.margin(), 2_500_000_000i128 - need as i128);

                let mut reserved = v.clone();
                reserved.others = vec![Outstanding {
                    pid: u32::MAX,
                    reserved: 600_000_000,
                    footprint: Some(100_000_000),
                    ..Default::default()
                }];
                reserved.mine.reserved = 300_000_000;
                reserved.mine.footprint = Some(50_000_000);
                assert_eq!(reserved.room(), 1_750_000_000);
                assert_eq!(reserved.margin(), 1_750_000_000i128 - need as i128);
            }
        }
    }

    /// Exhausted groups have negative room, not a shared zero. Selection
    /// must preserve the most restrictive ancestor even after saturation.
    #[test]
    fn ancestor_room_selection_is_signed_and_bounds_every_visible_group() {
        let mut cases = 0;
        for parent_limit in [10u64, 20, 30] {
            for child_limit in [10u64, 20, 30] {
                for child_usage in [0u64, 10, 20, 30, 40] {
                    for parent_usage in [0u64, 10, 20, 30, 40] {
                        if parent_usage < child_usage {
                            continue;
                        }
                        for reclaimable in [0u64, 5] {
                            if reclaimable > child_usage {
                                continue;
                            }
                            let parent = ContainerMemory {
                                limit: parent_limit,
                                usage: parent_usage,
                                reclaimable,
                            };
                            let child = ContainerMemory {
                                limit: child_limit,
                                usage: child_usage,
                                reclaimable,
                            };
                            let mut entries = Vec::new();
                            for (path, c) in [("a", parent), ("a/b", child)] {
                                let dir = format!("/sys/fs/cgroup/{path}");
                                entries.push((format!("{dir}/memory.max"), c.limit.to_string()));
                                entries
                                    .push((format!("{dir}/memory.current"), c.usage.to_string()));
                                entries.push((
                                    format!("{dir}/memory.stat"),
                                    format!("inactive_file {}\n", c.reclaimable),
                                ));
                            }
                            let refs: Vec<_> = entries
                                .iter()
                                .map(|(p, t)| (p.as_str(), t.as_str()))
                                .collect();
                            let selected = container_memory_from("0::/a/b\n", HOST, files(&refs))
                                .expect("both finite limits are visible");
                            assert_eq!(selected.room(), parent.room().min(child.room()));
                            if parent_usage >= parent_limit + reclaimable
                                && child_usage >= child_limit + reclaimable
                            {
                                assert_eq!(
                                    selected.room(),
                                    -(parent_limit.max(child_limit) as i128 / 10),
                                    "the larger reserve binds when both groups have no availability"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 225, "the bounded matrix must not silently shrink");
    }

    /// cgroup v1's memory controller: its own mount, its own file names,
    /// the hierarchical inactive cache.
    #[test]
    fn a_v1_memory_controller_limit_is_read() {
        let fs = files(&[
            (
                "/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "6442450944",
            ),
            (
                "/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes",
                "700000000",
            ),
            (
                "/sys/fs/cgroup/memory/docker/abc/memory.stat",
                "cache 300000000\ninactive_file 1\ntotal_inactive_file 200000000\n",
            ),
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "9223372036854771712",
            ),
        ]);
        let c = container_memory_from(
            "12:pids:/docker/abc\n11:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n",
            HOST,
            fs,
        )
        .unwrap();
        assert_eq!(
            c,
            ContainerMemory {
                limit: 6_442_450_944,
                usage: 700_000_000,
                reclaimable: 200_000_000
            }
        );
    }

    /// A 6 GiB container on a host with 25 GB free:
    /// the host alone admits a 5 GB load the container cannot hold; with
    /// the group's limit read, its room binds and the load is refused with
    /// the group's arithmetic. The host-only arm is the control.
    #[test]
    fn a_container_limit_below_the_host_refuses_what_the_host_would_admit() {
        let dir = tmp_registry("container");
        let host = |container| HostMemory {
            total: HOST,
            free: 25_000_000_000,
            container,
            ..Default::default()
        };
        let gate = Gate::new(GateConfig {
            registry: Some(dir.clone()),
            ..Default::default()
        });
        let peak = Peak::new().with("load", 5_000_000_000);
        let admitted = gate
            .admit_with("host only", &peak, || Some(host(None)))
            .unwrap();
        assert!(admitted.verdict.as_ref().unwrap().fits);
        drop(admitted);

        let six_gib = ContainerMemory {
            limit: 6 * GIB,
            usage: 1_000_000_000,
            reclaimable: 0,
        };
        let error = gate
            .admit_with("in a container", &peak, || Some(host(Some(six_gib))))
            .unwrap_err()
            .to_string();
        // (6442450944 - 1000000000) - 644245094 = 4798205850 B of room.
        for want in [
            "the room is 4798205850 B (4.80 GB), 201794150 B short",
            "container  5442450944 B (5.44 GB) = limit 6.44 - (usage 1.00 - inactive file 0.00)",
            "reserve 644245094 B (0.64 GB) = 10 % of the limit",
            "the container binds",
        ] {
            assert!(error.contains(want), "{want:?} missing from:\n{error}");
        }
        assert!(Registry::open(&dir).unwrap().entries().unwrap().is_empty());
        let v = gate
            .decide_with(5_000_000_000, None, &[], None, || Some(host(Some(six_gib))))
            .unwrap();
        assert!(v.container_binds() && !v.fits);
        assert!(
            v.line().contains(
                "container=5.44 GB of a 6.44 GB limit reserve=0.64 GB (10 % of the limit) binds"
            ),
            "{}",
            v.line()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// The reader on the host it runs on: where this process's group has a
    /// numeric v2 `memory.max` below `MemTotal` (a capped container), the
    /// gate sees that limit. A plain host reads no limit and asserts nothing
    /// of it. The reading is printed for the run's record.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_container_limit_is_read_where_one_is_set() {
        let h = HostMemory::read().expect("meminfo");
        let own = fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&l| l < h.total);
        eprintln!("host {h:?} own memory.max {own:?}");
        if let Some(limit) = own {
            let c = h.container.expect("a capped group reads a limit");
            assert!(c.limit <= limit, "{c:?} against memory.max {limit}");
            assert!(c.usage > 0 && c.available() <= c.limit, "{c:?}");
        }
    }

    /// The state that killed the session on 2026-10-08 (`issue-952-raw/
    /// crash-report.txt`): 6.92 GB available. A tenth of the machine leaves
    /// +0.05 GB of room and admits a 0.04 GB load into it; what the OS keeps
    /// from the working set leaves -6.14 GB and refuses it.
    #[test]
    fn the_crash_stackshot_is_admitted_on_a_tenth_and_refused_on_the_working_set() {
        let dir = tmp_registry("stackshot");
        let need = Peak::new().with("load", 40_000_000);
        let config = |working_set| GateConfig {
            registry: Some(dir.clone()),
            available: Some(6_920_000_000),
            total: Some(M1MAX_MEMSIZE),
            working_set,
            ..Default::default()
        };

        let tenth = Gate::new(config(None)).admit("tenth", &need).unwrap();
        let v = tenth.verdict.as_ref().unwrap();
        assert_eq!(v.reserve, 6_871_947_673);
        assert_eq!(v.reserve_basis, ReserveBasis::TenthOfMachine);
        assert!(v.fits, "{}", v.line());
        assert!(v.line().contains("reserve=6.87 GB (10 %)"), "{}", v.line());
        drop(tenth);

        let error = Gate::new(config(Some(M1MAX_WORKING_SET)))
            .admit("working set", &need)
            .unwrap_err()
            .to_string();
        for want in [
            "it needs 40000000 B (0.04 GB) and the room is -6136688128 B (-6.14 GB), 6176688128 B short",
            "reserve    13056688128 B (13.06 GB) = 68.72 GB memory - 55.66 GB GPU working set (over the 4 GiB floor)",
            "Nothing was allocated",
        ] {
            assert!(error.contains(want), "{want:?} missing from:\n{error}");
        }
        assert!(Registry::open(&dir).unwrap().entries().unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_configured_reserve_and_an_injected_working_set_outrank_the_device() {
        let dir = tmp_registry("reserve-basis");
        let mut cfg = gate(&dir, 10 * GIB, 5 * GIB).config().clone();
        let v = Gate::new(cfg.clone())
            .with_device_working_set(Some(M1MAX_WORKING_SET))
            .decide(0, None, &[], None)
            .unwrap();
        assert_eq!(
            (v.reserve, v.reserve_basis),
            (5 * GIB, ReserveBasis::Configured)
        );
        assert!(
            v.line().contains("(memory.machine.reserve_gb)"),
            "{}",
            v.line()
        );

        cfg.reserve = None;
        cfg.working_set = Some(60 * GIB);
        let g = Gate::new(cfg).with_device_working_set(Some(M1MAX_WORKING_SET));
        assert_eq!(g.config().working_set, Some(60 * GIB));
        let v = g.decide(0, None, &[], None).unwrap();
        assert_eq!(v.reserve, 4 * GIB, "64 GiB - 60 GiB is the floor");
        assert!(v.line().contains("(memsize-working_set)"), "{}", v.line());

        let device = Gate::default().with_device_working_set(Some(M1MAX_WORKING_SET));
        assert_eq!(device.config().working_set, Some(M1MAX_WORKING_SET));
        fs::remove_dir_all(dir).unwrap();
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

    #[test]
    fn an_overflowing_peak_is_refused_before_registration_even_when_forced() {
        let dir = tmp_registry("peak-overflow");
        let peak = Peak::new().with("weights", u64::MAX).with("scratch", 1);
        assert_eq!(peak.total(), u64::MAX);
        assert!(peak.describe().contains("overflow"));
        for force in [false, true] {
            let mut cfg = gate(&dir, u64::MAX, 0).config().clone();
            cfg.force = force;
            let error = Gate::new(cfg)
                .admit("overflow", &peak)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("machine peak arithmetic overflow"),
                "{error}"
            );
            assert!(Registry::open(&dir).unwrap().entries().unwrap().is_empty());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn overflowing_reservations_for_one_pid_are_not_a_smaller_reservation() {
        let dir = tmp_registry("pid-overflow");
        for pid in [std::process::id(), u32::MAX] {
            let entry = Entry {
                pid,
                label: "numeric only".into(),
                reserved: u64::MAX,
                since: 0,
            };
            let mut second = entry.clone();
            second.reserved = 1;
            let error = gate(&dir, u64::MAX, 0)
                .decide(1, None, &[entry, second], None)
                .unwrap_err()
                .to_string();
            assert!(error.contains("reservation arithmetic overflow"), "{error}");
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn outstanding_across_pids_uses_wide_arithmetic_in_decisions_and_reports() {
        let dir = tmp_registry("wide-outstanding");
        let entries: Vec<_> = [u32::MAX, u32::MAX - 1]
            .into_iter()
            .map(|pid| Entry {
                pid,
                label: "numeric only".into(),
                reserved: u64::MAX,
                since: 0,
            })
            .collect();
        let verdict = gate(&dir, u64::MAX, 0)
            .decide(1, None, &entries, None)
            .unwrap();
        assert!(!verdict.fits);
        assert_eq!(verdict.room(), -(u64::MAX as i128));
        assert!(verdict.line().contains("REFUSED"));
        assert!(verdict.describe().contains("cannot hold"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn overflowing_host_terms_are_not_an_available_byte_count() {
        let host = HostMemory {
            free: u64::MAX,
            inactive: 1,
            ..Default::default()
        };
        assert_eq!(host.checked_available(), None);
        assert_eq!(host.available(), u64::MAX, "reporting ceiling only");
    }

    #[test]
    fn a_failed_host_read_is_retried_once_before_refusing() {
        let dir = tmp_registry("host-retry");
        let mut cfg = gate(&dir, 10_000, 0).config().clone();
        cfg.available = None;
        let mut reads = 0;
        let admitted = Gate::new(cfg)
            .admit_with("retry", &Peak::new().with("bytes", 100), || {
                reads += 1;
                (reads == 2).then_some(HostMemory {
                    total: 20_000,
                    free: 10_000,
                    ..Default::default()
                })
            })
            .unwrap();
        assert_eq!(reads, 2);
        assert!(admitted.verdict.as_ref().unwrap().fits);
        assert_eq!(admitted.holding.as_ref().unwrap().reserved(), 100);
        drop(admitted);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unreadable_host_refuses_unless_explicitly_forced_and_never_reports_a_fit() {
        let dir = tmp_registry("host-unknown");
        for force in [false, true] {
            let mut cfg = gate(&dir, 10_000, 0).config().clone();
            cfg.available = None;
            cfg.force = force;
            let mut reads = 0;
            let result =
                Gate::new(cfg).admit_with("unknown", &Peak::new().with("bytes", 100), || {
                    reads += 1;
                    None
                });
            assert_eq!(reads, 2);
            if force {
                let admitted = result.unwrap();
                let verdict = admitted.verdict.as_ref().unwrap();
                assert!(!verdict.fits && verdict.forced);
                assert!(verdict.line().contains("available=UNKNOWN room=UNKNOWN"));
                assert!(verdict.line().contains("safety unchecked"));
                assert!(verdict.describe().contains("safety could not be checked"));
                assert_eq!(
                    Registry::open(&dir).unwrap().entries().unwrap()[0].reserved,
                    100
                );
            } else {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("could not be read after two attempts"),
                    "{error}"
                );
                assert!(error.contains("available  UNKNOWN"), "{error}");
                assert!(error.contains("--force"), "{error}");
                assert!(Registry::open(&dir).unwrap().entries().unwrap().is_empty());
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_off_and_fully_injected_budgets_do_not_probe_the_host() {
        let dir = tmp_registry("no-host-probe");
        for off in [false, true] {
            let mut cfg = gate(&dir, 10_000, 0).config().clone();
            cfg.off = off;
            let admitted = Gate::new(cfg)
                .admit_with("injected", &Peak::new().with("bytes", 100), || {
                    panic!("an injected or disabled gate must not observe the host")
                })
                .unwrap();
            assert_eq!(admitted.holding.is_none(), off);
        }
        fs::remove_dir_all(dir).unwrap();
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
            "it needs 18253611008 B (18.25 GB) and the room is 17179869184 B (17.18 GB), 1073741824 B short",
            "needed     18.25 GB = weights 18.25",
            "available  21474836480 B (21.47 GB) (injected",
            "reserve    4294967296 B (4.29 GB) (memory.machine.reserve_gb)",
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
        let decided = std::sync::Barrier::new(2);
        let results: Vec<bool> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..2)
                .map(|i| {
                    let (g, peak) = (g.clone(), peak.clone());
                    let decided = &decided;
                    s.spawn(move || {
                        let r = g.admit(&format!("racer {i}"), &peak);
                        // Hold it until both have decided.
                        decided.wait();
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

    #[test]
    fn a_live_corrupt_reservation_refuses_then_recovers_without_reaping() {
        let dir = tmp_registry("corrupt-live");
        let reg = Registry::open(&dir).unwrap();
        let h = reg.reserve_unchecked("live", 100).unwrap();
        let original = fs::read(h.path()).unwrap();
        {
            let _guard = reg.guard().unwrap();
            fs::write(h.path(), "pid=broken\nreserved=100\nsince=0\n").unwrap();
        }
        for force in [false, true] {
            let mut cfg = gate(&dir, 10_000, 0).config().clone();
            cfg.force = force;
            let error = Gate::new(cfg)
                .admit("next", &Peak::new().with("numeric bytes only", 1))
                .unwrap_err()
                .to_string();
            assert!(error.contains("cannot read live reservation"), "{error}");
            assert!(h.path().exists(), "a live owner must never be reaped");
        }
        {
            let _guard = reg.guard().unwrap();
            fs::write(h.path(), original).unwrap();
        }
        assert_eq!(reg.entries().unwrap()[0].reserved, 100);
        assert!(gate(&dir, 10_000, 0)
            .admit("recovered", &Peak::new().with("numeric bytes only", 1))
            .is_ok());
        drop(h);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_unopenable_registry_entry_is_not_silently_ignored() {
        let dir = tmp_registry("unopenable");
        let path = dir.join("unreadable.res");
        fs::create_dir(&path).unwrap();
        let error = gate(&dir, 10_000, 0)
            .admit("next", &Peak::new().with("numeric bytes only", 1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot open"), "{error}");
        assert!(path.is_dir(), "uncertainty must not remove an entry");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_registry_write_restores_the_previous_record_under_the_lock() {
        let dir = tmp_registry("write-recovery");
        let reg = Registry::open(&dir).unwrap();
        let h = reg.reserve_unchecked("live", 100).unwrap();
        {
            let _guard = reg.guard().unwrap();
            let previous = h.entry.lock().clone();
            let mut next = previous.clone();
            next.reserved = 200;
            let mut calls = 0;
            let error = replace_entry(&mut h.file.lock(), &previous, &next, |f, e| {
                calls += 1;
                if calls == 1 {
                    f.set_len(0)?;
                    return Err(io::Error::other("injected partial write"));
                }
                write_entry(f, e)
            })
            .unwrap_err();
            assert_eq!(calls, 2, "the failed write must attempt recovery");
            assert!(error.to_string().contains("previous reservation restored"));
            assert_eq!(read_live_entry(&mut h.file.lock()).unwrap(), previous);
        }
        assert_eq!(reg.entries().unwrap()[0].reserved, 100);
        assert_eq!(h.reserved(), 100);
        drop(h);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_update_and_restore_leave_a_live_uncertain_entry_refused() {
        let dir = tmp_registry("double-write-failure");
        let reg = Registry::open(&dir).unwrap();
        let h = reg.reserve_unchecked("live", 100).unwrap();
        {
            let _guard = reg.guard().unwrap();
            let previous = h.entry.lock().clone();
            let mut next = previous.clone();
            next.reserved = 200;
            let mut calls = 0;
            let error = replace_entry(&mut h.file.lock(), &previous, &next, |f, _| {
                calls += 1;
                f.set_len(0)?;
                Err(io::Error::other("injected write and restore failure"))
            })
            .unwrap_err();
            assert_eq!(calls, 2);
            assert!(error.to_string().contains("restoration failed"), "{error}");
            assert!(
                error.to_string().contains("release this holding"),
                "{error}"
            );
        }
        assert_eq!(h.reserved(), 100, "uncommitted growth stays unpublished");
        let error = reg.entries().unwrap_err().to_string();
        assert!(error.contains("admission cannot establish room"), "{error}");
        assert!(h.path().exists(), "never delete the live uncertain entry");
        drop(h);
        assert!(reg.entries().unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_torn_record_never_decodes_a_partial_reserved_number() {
        // Decimal length changes are important: overwriting 900 with 1000
        // without truncating first could leave a valid but unsafe 100.
        for reserved in [900, 1000] {
            let record = Entry {
                pid: 123,
                label: "live".into(),
                reserved,
                since: 12345,
            };
            let text = record.encode();
            for cut in 0..=text.len() {
                if let Some(decoded) = Entry::decode(&text[..cut]) {
                    assert_eq!(decoded.reserved, reserved, "cut={cut}");
                }
            }
        }
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
        assert!(e.contains("6442450944 B (6.44 GB) of growth"), "{e}");
        assert_eq!(h.reserved(), 12 * GIB);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_waiting_grower_rechecks_both_the_target_and_the_delta() {
        use std::sync::mpsc::channel;
        use std::time::Duration;

        for target in [200, 400] {
            let dir = tmp_registry("stale-growth");
            let admitted = gate(&dir, 10_000, 0)
                .admit("grower", &Peak::new().with("numeric bytes only", 100))
                .unwrap();
            let h = admitted.holding.unwrap();
            let (paused_tx, paused_rx) = channel();
            let (resume_tx, resume_rx) = channel();
            let result = std::thread::scope(|s| {
                let holding = &h;
                let worker = s.spawn(move || {
                    holding.grow_with(target, || {
                        paused_tx.send(()).unwrap();
                        resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    })
                });
                paused_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(h.grow_to(300).unwrap().unwrap().need, 200);
                resume_tx.send(()).unwrap();
                worker.join().unwrap().unwrap()
            });
            assert_eq!(result.map(|v| v.need), (target > 300).then_some(100));
            let expected = target.max(300);
            assert_eq!(h.reserved(), expected, "growth never shrinks the holding");
            assert_eq!(
                Registry::open(&dir).unwrap().entries().unwrap()[0].reserved,
                expected,
                "the durable reservation agrees with the atomic value"
            );
            drop(h);
            fs::remove_dir_all(dir).unwrap();
        }
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

#[cfg(all(test, unix))]
#[path = "machine_process_tests.rs"]
mod process_tests;
