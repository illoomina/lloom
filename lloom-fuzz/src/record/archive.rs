//! Versioned completion evidence over the existing numbered input files.
//! This is ordinary allocating filesystem I/O, never a signal-safe writer.
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const FORMAT: &str = "lloom-input-record-v1\n";
const FORMAT_FILE: &str = "record.format";
const COMPLETE_FILE: &str = "record.complete";

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Create a new archive; existing paths are never overwritten. An interrupted
/// archive remains inspectable but cannot masquerade as a completed v1 replay.
/// Successful writes are synchronized; this is not an atomic filesystem snapshot
/// or a guarantee against later modification, media corruption or machine loss.
pub(super) fn write<'a>(
    dir: &Path,
    inputs: impl Iterator<Item = &'a [u8]>,
    reason: &[u8],
) -> io::Result<()> {
    fs::create_dir(dir)?;
    write_new(&dir.join(FORMAT_FILE), FORMAT.as_bytes())?;
    let mut lengths = Vec::new();
    for (index, bytes) in inputs.enumerate() {
        write_new(&dir.join(format!("{index:020}.bin")), bytes)?;
        lengths.push(bytes.len());
    }
    if lengths.is_empty() {
        return Err(invalid("record contains no inputs"));
    }
    write_new(&dir.join("reason.txt"), reason)?;
    let mut manifest = format!("{FORMAT}{}\n", lengths.len());
    for length in lengths {
        use std::fmt::Write as _;
        writeln!(manifest, "{length}").expect("String formatting is infallible");
    }
    manifest.push_str("END\n");
    write_new(&dir.join(COMPLETE_FILE), manifest.as_bytes())?;
    fs::File::open(dir)?.sync_all()
}

fn index(path: &Path) -> io::Result<usize> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("non-UTF8 input filename"))?;
    if stem.is_empty() || !stem.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid(format!(
            "invalid numbered input: {}",
            path.display()
        )));
    }
    stem.parse()
        .map_err(|_| invalid("input index overflows usize"))
}

fn sort_numbered(entries: &mut [(usize, PathBuf)]) {
    entries.sort_by_key(|(index, _)| *index);
}

fn ordered(entries: impl Iterator<Item = io::Result<PathBuf>>) -> io::Result<Vec<PathBuf>> {
    let mut numbered = Vec::new();
    for entry in entries {
        let path = entry?; // A failed directory entry is not an absent input.
        if path.extension().is_some_and(|x| x == "bin") {
            numbered.push((index(&path)?, path));
        }
    }
    sort_numbered(&mut numbered);
    if numbered.is_empty() {
        return Err(invalid("record directory contains no inputs"));
    }
    for (expected, (actual, _)) in numbered.iter().enumerate() {
        if expected != *actual {
            return Err(invalid(format!(
                "missing or duplicate input index {expected}"
            )));
        }
    }
    Ok(numbered.into_iter().map(|(_, path)| path).collect())
}

fn optional(path: &Path) -> io::Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Debug)]
pub(super) struct Replay {
    pub(super) paths: Vec<PathBuf>,
    pub(super) lengths: Vec<u64>,
    pub(super) legacy: bool,
}

/// Validate every directory entry and declared size before executing the first
/// target. Legacy directories remain readable with explicitly weaker guarantees:
/// their missing final input cannot be detected without a completion manifest.
pub(super) fn prepare(path: &Path, maximum: usize) -> io::Result<Replay> {
    let (paths, manifest, legacy) = if path.is_dir() {
        let format = optional(&path.join(FORMAT_FILE))?;
        let manifest = optional(&path.join(COMPLETE_FILE))?;
        match (&format, &manifest) {
            (Some(format), Some(_)) if format == FORMAT => {}
            (None, None) => {}
            (Some(format), _) if format != FORMAT => {
                return Err(invalid("unsupported record format"))
            }
            _ => return Err(invalid("incomplete record: format/completion pair missing")),
        }
        let paths = ordered(fs::read_dir(path)?.map(|entry| entry.map(|entry| entry.path())))?;
        (paths, manifest, format.is_none())
    } else {
        (vec![path.to_path_buf()], None, false)
    };
    let mut lengths = Vec::with_capacity(paths.len());
    for path in &paths {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() || metadata.len() > maximum as u64 {
            return Err(invalid(format!(
                "input is not a regular file within the {maximum}-byte bound: {}",
                path.display()
            )));
        }
        lengths.push(metadata.len());
    }
    if let Some(manifest) = manifest {
        let mut lines = manifest.lines();
        if lines.next() != Some(FORMAT.trim_end()) {
            return Err(invalid("unsupported completion format"));
        }
        let count = lines
            .next()
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| invalid("invalid input count"))?;
        if count != paths.len() {
            return Err(invalid(
                "record input count differs from completion manifest",
            ));
        }
        for length in &lengths {
            let declared = lines
                .next()
                .and_then(|n| n.parse::<u64>().ok())
                .ok_or_else(|| invalid("invalid input length"))?;
            if declared != *length {
                return Err(invalid(
                    "record input length differs from completion manifest",
                ));
            }
        }
        if lines.next() != Some("END") || lines.next().is_some() {
            return Err(invalid("incomplete or extended completion manifest"));
        }
    }
    Ok(Replay {
        paths,
        lengths,
        legacy,
    })
}

/// Bound allocation even if a file grows after preflight. Recheck size rather
/// than interpreting changed-length bytes as the recorded execution.
pub(super) fn read(path: &Path, expected: u64, maximum: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != expected || bytes.len() > maximum {
        return Err(invalid("input length changed after replay preflight"));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lloom-record-archive-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn archive(&self) -> PathBuf {
            self.0.join("record")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn fixture(dir: &Path) {
        write(dir, [b"B".as_slice(), b"C"].into_iter(), b"planted").unwrap();
    }
    #[test]
    fn completed_archive_preserves_order_and_empty_input() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        write(&dir, [b"B".as_slice(), b"", b"C"].into_iter(), b"planted").unwrap();
        let replay = prepare(&dir, 4).unwrap();
        assert!(!replay.legacy);
        let bytes: Vec<_> = replay
            .paths
            .iter()
            .zip(replay.lengths)
            .map(|(p, n)| read(p, n, 4).unwrap())
            .collect();
        assert_eq!(bytes, vec![b"B".to_vec(), vec![], b"C".to_vec()]);
        assert!(write(&dir, [b"X".as_slice()].into_iter(), b"overwrite").is_err());
    }
    #[test]
    fn incomplete_and_unsupported_records_are_not_legacy() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        fixture(&dir);
        fs::remove_file(dir.join(COMPLETE_FILE)).unwrap();
        assert!(prepare(&dir, 4)
            .unwrap_err()
            .to_string()
            .contains("incomplete record"));
        fs::write(dir.join(FORMAT_FILE), "lloom-input-record-v999\n").unwrap();
        assert!(prepare(&dir, 4)
            .unwrap_err()
            .to_string()
            .contains("unsupported record"));
    }
    #[test]
    fn missing_final_and_truncated_inputs_are_detected() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        fixture(&dir);
        let final_path = dir.join(format!("{:020}.bin", 1));
        fs::remove_file(&final_path).unwrap();
        assert!(prepare(&dir, 4)
            .unwrap_err()
            .to_string()
            .contains("input count differs"));
        fs::write(&final_path, b"").unwrap();
        assert!(prepare(&dir, 4)
            .err()
            .expect("truncated input accepted")
            .to_string()
            .contains("input length differs"));
    }
    #[test]
    fn legacy_order_is_numeric_and_entry_errors_propagate() {
        assert!(index(Path::new("10000.bin")).unwrap() > index(Path::new("9999.bin")).unwrap());
        let paths = ordered(
            ["0002.bin", "0000.bin", "0001.bin"]
                .into_iter()
                .map(|p| Ok(PathBuf::from(p))),
        )
        .unwrap();
        assert_eq!(
            paths,
            ["0000.bin", "0001.bin", "0002.bin"].map(PathBuf::from)
        );
        assert!(ordered(
            [
                Ok(PathBuf::from("0000.bin")),
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "planted entry error"
                ))
            ]
            .into_iter()
        )
        .is_err());
        assert!(ordered(
            ["0000.bin", "0002.bin"]
                .into_iter()
                .map(|p| Ok(PathBuf::from(p)))
        )
        .is_err());
        let tmp = Temp::new();
        fs::write(tmp.0.join("0000.bin"), b"B").unwrap();
        assert!(prepare(&tmp.0, 4).unwrap().legacy);
    }
    #[test]
    fn numeric_sort_crosses_legacy_four_digit_boundary() {
        let mut paths = [
            (10000, PathBuf::from("10000.bin")),
            (9999, PathBuf::from("9999.bin")),
        ];
        sort_numbered(&mut paths);
        assert_eq!(paths[0].0, 9999, "legacy input order became lexical");
        assert_eq!(paths[1].0, 10000);
    }

    #[test]
    fn changed_length_after_preflight_is_refused() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        fixture(&dir);
        let replay = prepare(&dir, 4).unwrap();
        fs::write(&replay.paths[0], b"12345").unwrap();
        assert!(read(&replay.paths[0], replay.lengths[0], 4).is_err());
    }
}
