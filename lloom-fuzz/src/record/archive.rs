//! Versioned completion evidence over the existing numbered input files.
//! This is ordinary allocating filesystem I/O, never a signal-safe writer.
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const FORMAT: &str = "lloom-input-record-v1\n";
const LEGACY_IDENTIFIED_FORMAT: &str = "lloom-input-record-v2\n";
const IDENTIFIED_FORMAT: &str = "lloom-input-record-v3\n";
const IDENTITY_FILE: &str = "record.identity";
pub(super) const MAX_IDENTITY: usize = 64 * 1024;
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
#[cfg(test)]
pub(super) fn write<'a>(
    dir: &Path,
    inputs: impl Iterator<Item = &'a [u8]>,
    reason: &[u8],
) -> io::Result<()> {
    write_identified(dir, inputs, reason, None)
}

pub(super) fn write_identified<'a>(
    dir: &Path,
    inputs: impl Iterator<Item = &'a [u8]>,
    reason: &[u8],
    identity: Option<&[u8]>,
) -> io::Result<()> {
    if identity.is_some_and(|bytes| bytes.is_empty() || bytes.len() > MAX_IDENTITY) {
        return Err(invalid("record identity is empty or exceeds its bound"));
    }
    let format = if identity.is_some() {
        IDENTIFIED_FORMAT
    } else {
        FORMAT
    };
    fs::create_dir(dir)?;
    write_new(&dir.join(FORMAT_FILE), format.as_bytes())?;
    if let Some(identity) = identity {
        write_new(&dir.join(IDENTITY_FILE), identity)?;
    }
    let mut lengths = Vec::new();
    for (index, bytes) in inputs.enumerate() {
        write_new(&dir.join(format!("{index:020}.bin")), bytes)?;
        lengths.push((bytes.len(), Sha256::digest(bytes)));
    }
    if lengths.is_empty() {
        return Err(invalid("record contains no inputs"));
    }
    write_new(&dir.join("reason.txt"), reason)?;
    let mut manifest = format!("{format}{}\n", lengths.len());
    for (length, digest) in lengths {
        use std::fmt::Write as _;
        if identity.is_some() {
            writeln!(manifest, "{length} {digest:x}").expect("String formatting is infallible");
        } else {
            writeln!(manifest, "{length}").expect("String formatting is infallible");
        }
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

#[cfg(test)]
fn ordered(entries: impl Iterator<Item = io::Result<PathBuf>>) -> io::Result<Vec<PathBuf>> {
    ordered_with_limits(
        entries,
        &mut MetadataBudget::new(super::ReplayLimits::default()),
    )
}

struct MetadataBudget {
    limits: super::ReplayLimits,
    bytes: usize,
}
impl MetadataBudget {
    fn new(limits: super::ReplayLimits) -> Self {
        Self { limits, bytes: 0 }
    }
    fn add(&mut self, bytes: usize) -> io::Result<()> {
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("replay metadata budget overflow"))?;
        if total > self.limits.max_metadata_bytes {
            return Err(invalid(
                "replay metadata budget exceeded; raise the explicit replay metadata limit",
            ));
        }
        self.bytes = total;
        Ok(())
    }
}

fn ordered_with_limits(
    entries: impl Iterator<Item = io::Result<PathBuf>>,
    budget: &mut MetadataBudget,
) -> io::Result<Vec<PathBuf>> {
    let mut numbered = Vec::new();
    for (seen, entry) in entries.enumerate() {
        if seen >= budget.limits.max_directory_entries {
            return Err(invalid(
                "replay directory entry budget exceeded; raise the explicit replay entry limit",
            ));
        }
        let path = entry?; // A failed directory entry is not an absent input.
        if path.extension().is_some_and(|x| x == "bin") {
            if numbered.len() >= budget.limits.max_inputs {
                return Err(invalid(
                    "replay input count budget exceeded; raise the explicit replay input limit",
                ));
            }
            budget.add(path.as_os_str().as_encoded_bytes().len())?;
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

fn optional(path: &Path, budget: &mut MetadataBudget) -> io::Result<Option<String>> {
    let remaining = budget
        .limits
        .max_metadata_bytes
        .saturating_sub(budget.bytes);
    match read_single(path, remaining) {
        Ok(value) => {
            budget.add(value.len())?;
            String::from_utf8(value)
                .map(Some)
                .map_err(|_| invalid("non-UTF8 replay metadata"))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Debug)]
pub(super) struct Replay {
    pub(super) paths: Vec<PathBuf>,
    pub(super) lengths: Vec<u64>,
    pub(super) legacy: bool,
    pub(super) identity: Option<Vec<u8>>,
    digests: Option<Vec<[u8; 32]>>,
}

/// Validate every directory entry and declared size before executing the first
/// target. Legacy directories remain readable with explicitly weaker guarantees:
/// their missing final input cannot be detected without a completion manifest.
#[cfg(test)]
pub(super) fn prepare(path: &Path, maximum: usize) -> io::Result<Replay> {
    prepare_with_limits(path, maximum, super::ReplayLimits::default())
}

pub(super) fn prepare_with_limits(
    path: &Path,
    maximum: usize,
    limits: super::ReplayLimits,
) -> io::Result<Replay> {
    let mut budget = MetadataBudget::new(limits);
    let (paths, manifest, legacy, format, identity) = if path.is_dir() {
        let format = optional(&path.join(FORMAT_FILE), &mut budget)?;
        let manifest = optional(&path.join(COMPLETE_FILE), &mut budget)?;
        match (&format, &manifest) {
            (Some(format), Some(_))
                if format == FORMAT
                    || format == LEGACY_IDENTIFIED_FORMAT
                    || format == IDENTIFIED_FORMAT => {}
            (None, None) => {}
            (Some(format), _)
                if format != FORMAT
                    && format != LEGACY_IDENTIFIED_FORMAT
                    && format != IDENTIFIED_FORMAT =>
            {
                return Err(invalid("unsupported record format"))
            }
            _ => return Err(invalid("incomplete record: format/completion pair missing")),
        }
        let identity = if matches!(
            format.as_deref(),
            Some(IDENTIFIED_FORMAT | LEGACY_IDENTIFIED_FORMAT)
        ) {
            let remaining = budget
                .limits
                .max_metadata_bytes
                .saturating_sub(budget.bytes)
                .min(MAX_IDENTITY);
            let bytes = read_single(&path.join(IDENTITY_FILE), remaining).map_err(|error| {
                io::Error::new(error.kind(), format!("{IDENTITY_FILE}: {error}"))
            })?;
            if bytes.is_empty() {
                return Err(invalid("empty record identity"));
            }
            budget.add(bytes.len())?;
            Some(bytes)
        } else {
            if path.join(IDENTITY_FILE).try_exists()? {
                return Err(invalid(
                    "identity without identified format; refusing silent downgrade",
                ));
            }
            None
        };
        let paths = ordered_with_limits(
            fs::read_dir(path)?.map(|entry| entry.map(|entry| entry.path())),
            &mut budget,
        )?;
        let legacy = format.is_none();
        (paths, manifest, legacy, format, identity)
    } else {
        if limits.max_inputs == 0 {
            return Err(invalid(
                "replay input count budget exceeded; raise the explicit replay input limit",
            ));
        }
        budget.add(path.as_os_str().as_encoded_bytes().len())?;
        (vec![path.to_path_buf()], None, false, None, None)
    };
    let mut lengths = Vec::with_capacity(paths.len());
    let mut total_input_bytes = 0u64;
    for path in &paths {
        let length = input_length(path, maximum)?;
        total_input_bytes = total_input_bytes
            .checked_add(length)
            .ok_or_else(|| invalid("replay total input budget overflow"))?;
        if total_input_bytes > limits.max_total_input_bytes {
            return Err(invalid(
                "replay total input byte budget exceeded; raise the explicit replay byte limit",
            ));
        }
        lengths.push(length);
    }
    let mut digests = (format.as_deref() == Some(IDENTIFIED_FORMAT)).then(Vec::new);
    if let Some(manifest) = manifest {
        let mut lines = manifest.lines();
        if lines.next() != format.as_deref().map(str::trim_end) {
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
            let row = lines
                .next()
                .ok_or_else(|| invalid("invalid input length"))?;
            let (declared, digest) = if digests.is_some() {
                let (length, digest) = row
                    .split_once(' ')
                    .ok_or_else(|| invalid("missing input content digest"))?;
                (length, Some(parse_digest(digest)?))
            } else {
                (row, None)
            };
            let declared = declared
                .parse::<u64>()
                .map_err(|_| invalid("invalid input length"))?;
            if declared != *length {
                return Err(invalid(
                    "record input length differs from completion manifest",
                ));
            }
            if let (Some(digests), Some(digest)) = (&mut digests, digest) {
                digests.push(digest);
            }
        }
        if lines.next() != Some("END") || lines.next().is_some() {
            return Err(invalid("incomplete or extended completion manifest"));
        }
    }
    let replay = Replay {
        paths,
        lengths,
        legacy,
        identity,
        digests,
    };
    // Validate the entire declared history before any target invocation. Read
    // again at invocation and verify THOSE owned bytes: preflight is not a
    // filesystem snapshot, and same-length replacements remain possible.
    if replay.digests.is_some() {
        for index in 0..replay.paths.len() {
            replay.read_input(index, maximum)?;
        }
    }
    Ok(replay)
}

fn parse_digest(text: &str) -> io::Result<[u8; 32]> {
    if text.len() != 64 {
        return Err(invalid("invalid input content digest"));
    }
    fn nibble(byte: u8) -> io::Result<u8> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => Err(invalid("invalid input content digest")),
        }
    }
    let mut digest = [0u8; 32];
    for (out, pair) in digest.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        *out = nibble(pair[0])? * 16 + nibble(pair[1])?;
    }
    Ok(digest)
}
impl Replay {
    pub(super) fn read_input(&self, index: usize, maximum: usize) -> io::Result<Vec<u8>> {
        let bytes = read(&self.paths[index], self.lengths[index], maximum)?;
        if let Some(digests) = &self.digests {
            let actual: [u8; 32] = Sha256::digest(&bytes).into();
            if actual != digests[index] {
                return Err(invalid(
                    "input content digest differs from completion manifest",
                ));
            }
        }
        Ok(bytes)
    }
    pub(super) fn content_verified(&self) -> bool {
        self.digests.is_some()
    }
}

pub(super) fn check_identity(
    actual: Option<&[u8]>,
    expected: Option<&[u8]>,
    allow_mismatch: bool,
) -> io::Result<bool> {
    match actual {
        None => Ok(false), // Existing legacy/v1/single-input support, explicitly unidentified.
        Some(actual) if Some(actual) == expected => Ok(true),
        Some(_) if allow_mismatch => Ok(false),
        Some(_) => Err(invalid("incompatible record identity: target/build/configuration differs or expected identity unavailable; use the matching binary/options, or explicitly request unchecked diagnostic replay")),
    }
}

fn input_length(path: &Path, maximum: usize) -> io::Result<u64> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > maximum as u64 {
        return Err(invalid(format!(
            "input is not a regular file within the {maximum}-byte bound: {}",
            path.display()
        )));
    }
    Ok(metadata.len())
}

/// Single-input execution shares replay's bound before allocating input bytes.
/// The streaming limit still applies if a file grows after the metadata check.
pub(crate) fn read_single(path: &Path, maximum: usize) -> io::Result<Vec<u8>> {
    read(path, input_length(path, maximum)?, maximum)
}

/// Bound allocation even if a file grows after preflight. Recheck size rather
/// than interpreting changed-length bytes as the recorded execution.
pub(super) fn read(path: &Path, expected: u64, maximum: usize) -> io::Result<Vec<u8>> {
    read_bounded(fs::File::open(path)?, expected, maximum)
}

fn read_bounded(reader: impl Read, expected: u64, maximum: usize) -> io::Result<Vec<u8>> {
    let limit = (maximum as u64)
        .checked_add(1)
        .ok_or_else(|| invalid("input bound overflows u64"))?;
    let mut bytes = Vec::new();
    reader.take(limit).read_to_end(&mut bytes)?;
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

    #[test]
    fn single_input_accepts_empty_and_exact_limit_but_refuses_oversize() {
        let tmp = Temp::new();
        let input = tmp.0.join("input.bin");
        for bytes in [b"".as_slice(), b"1234"] {
            fs::write(&input, bytes).unwrap();
            assert_eq!(read_single(&input, 4).unwrap(), bytes);
        }
        fs::write(&input, b"12345").unwrap();
        assert!(read_single(&input, 4)
            .unwrap_err()
            .to_string()
            .contains("4-byte bound"));
        assert!(read_single(&tmp.0, 4).is_err());
    }

    #[test]
    fn streaming_limit_precedes_size_validation() {
        // A finite growing-file stand-in: count bytes consumed by the SAME
        // reader used for File, not just the eventual size-error verdict.
        let mut input = io::Cursor::new([0u8; 64]);
        let error = read_bounded(&mut input, 4, 4).unwrap_err();
        assert!(error.to_string().contains("input length changed"));
        assert_eq!(
            input.position(),
            5,
            "oversized input read before validation"
        );
    }

    #[test]
    fn replay_input_budget_counts_empty_inputs_and_can_be_raised() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        write(&dir, [b"".as_slice(); 4].into_iter(), b"reason").unwrap();
        let mut limits = super::super::ReplayLimits {
            max_inputs: 3,
            ..Default::default()
        };
        let mut calls = 0;
        let error =
            super::super::replay_with_identity_and_limits(&dir, None, false, limits, |_| {
                calls += 1;
                Ok(())
            })
            .unwrap_err();
        assert!(error.1.contains("input count budget exceeded"), "{error:?}");
        assert_eq!(calls, 0, "input-count refusal executed a target");
        limits.max_inputs = 4;
        assert_eq!(
            super::super::replay_with_identity_and_limits(&dir, None, false, limits, |bytes| {
                assert!(bytes.is_empty());
                calls += 1;
                Ok(())
            })
            .unwrap(),
            4
        );
        assert_eq!(calls, 4);
    }

    #[test]
    fn replay_total_byte_budget_precedes_every_target_and_retains_integrity() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        write_identified(
            &dir,
            [b"B".as_slice(), b"C"].into_iter(),
            b"reason",
            Some(b"identity"),
        )
        .unwrap();
        let mut limits = super::super::ReplayLimits {
            max_total_input_bytes: 1,
            ..Default::default()
        };
        let mut calls = 0;
        let error = super::super::replay_with_identity_and_limits(
            &dir,
            Some(b"identity"),
            false,
            limits,
            |_| {
                calls += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            error.1.contains("total input byte budget exceeded"),
            "{error:?}"
        );
        assert_eq!(calls, 0);
        limits.max_total_input_bytes = 2;
        assert_eq!(
            super::super::replay_with_identity_and_limits(
                &dir,
                Some(b"identity"),
                false,
                limits,
                |_| {
                    calls += 1;
                    Ok(())
                }
            )
            .unwrap(),
            2
        );
        fs::write(dir.join(format!("{:020}.bin", 1)), b"X").unwrap();
        let error = super::super::replay_with_identity_and_limits(&dir, None, true, limits, |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
        assert!(error.1.contains("content digest differs"), "{error:?}");
        assert_eq!(
            calls, 2,
            "raised limits/identity override bypassed content integrity"
        );
    }

    #[test]
    fn replay_metadata_file_is_bounded_before_target() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        fixture(&dir);
        fs::write(dir.join(COMPLETE_FILE), [b'x'; 64]).unwrap();
        let limits = super::super::ReplayLimits {
            max_metadata_bytes: FORMAT.len() + 63,
            ..Default::default()
        };
        let mut calls = 0;
        let error =
            super::super::replay_with_identity_and_limits(&dir, None, false, limits, |_| {
                calls += 1;
                Ok(())
            })
            .unwrap_err();
        assert!(error.1.contains("63-byte bound"), "{error:?}");
        assert_eq!(calls, 0);
    }

    #[test]
    fn replay_metadata_budget_includes_all_paths_and_exact_limit_recovers() {
        let tmp = Temp::new();
        let dir = tmp.archive();
        fixture(&dir);
        let paths = [
            dir.join(format!("{:020}.bin", 0)),
            dir.join(format!("{:020}.bin", 1)),
        ];
        let metadata = FORMAT.len() + fs::metadata(dir.join(COMPLETE_FILE)).unwrap().len() as usize;
        let total = metadata
            + paths
                .iter()
                .map(|p| p.as_os_str().as_encoded_bytes().len())
                .sum::<usize>();
        let limits = super::super::ReplayLimits {
            max_metadata_bytes: total - 1,
            ..Default::default()
        };
        let error = prepare_with_limits(&dir, 4, limits).unwrap_err();
        assert!(
            error.to_string().contains("metadata budget exceeded"),
            "{error:?}"
        );
        let replay = prepare_with_limits(
            &dir,
            4,
            super::super::ReplayLimits {
                max_metadata_bytes: total,
                ..limits
            },
        )
        .unwrap();
        assert_eq!(replay.paths, paths);
        assert_eq!(replay.lengths, [1, 1]);
    }

    #[test]
    fn replay_directory_budget_counts_ignored_entries_with_bounded_lookahead() {
        let mut visited = 0;
        let entries = [
            "0000.bin",
            "ignored-a",
            "ignored-b",
            "ignored-c",
            "ignored-d",
        ]
        .into_iter()
        .map(|path| {
            visited += 1;
            Ok(PathBuf::from(path))
        });
        let limits = super::super::ReplayLimits {
            max_directory_entries: 2,
            ..Default::default()
        };
        let error = ordered_with_limits(entries, &mut MetadataBudget::new(limits)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("directory entry budget exceeded"),
            "{error:?}"
        );
        assert_eq!(
            visited, 3,
            "directory scan continued past one refusal lookahead"
        );
        let replay = ordered_with_limits(
            ["ignored", "0000.bin"]
                .into_iter()
                .map(|p| Ok(PathBuf::from(p))),
            &mut MetadataBudget::new(limits),
        )
        .unwrap();
        assert_eq!(replay, [PathBuf::from("0000.bin")]);
    }

    #[test]
    fn replay_zero_input_budget_refuses_single_file_before_target() {
        let tmp = Temp::new();
        let input = tmp.0.join("input.bin");
        fs::write(&input, b"").unwrap();
        let limits = super::super::ReplayLimits {
            max_inputs: 0,
            ..Default::default()
        };
        let mut calls = 0;
        let error =
            super::super::replay_with_identity_and_limits(&input, None, false, limits, |_| {
                calls += 1;
                Ok(())
            })
            .unwrap_err();
        assert!(error.1.contains("input count budget exceeded"), "{error:?}");
        assert_eq!(calls, 0);
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "lloom-record-identity-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            if self.0.exists() {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
    }
    #[test]
    fn identified_archive_roundtrip_and_mismatch_are_explicit() {
        let tmp = Temp::new();
        let identity = b"target=a;build=b;config=c;schema=1";
        write_identified(
            &tmp.0,
            [b"B".as_slice(), b"C"].into_iter(),
            b"reason",
            Some(identity),
        )
        .unwrap();
        let replay = prepare(&tmp.0, 4).unwrap();
        assert_eq!(replay.identity.as_deref(), Some(identity.as_slice()));
        assert!(check_identity(replay.identity.as_deref(), Some(identity), false).unwrap());
        for expected in [
            None,
            Some(b"target=z;build=b;config=c;schema=1".as_slice()),
            Some(b"target=a;build=z;config=c;schema=1"),
            Some(b"target=a;build=b;config=z;schema=1"),
            Some(b"target=a;build=b;config=c;schema=2"),
        ] {
            assert!(check_identity(replay.identity.as_deref(), expected, false)
                .unwrap_err()
                .to_string()
                .contains("incompatible record identity"));
            assert!(!check_identity(replay.identity.as_deref(), expected, true).unwrap());
        }
        assert!(!check_identity(None, Some(identity), false).unwrap());
    }
    #[test]
    fn identity_missing_oversized_or_downgraded_is_not_legacy() {
        let tmp = Temp::new();
        write_identified(&tmp.0, [b"B".as_slice()].into_iter(), b"reason", Some(b"x")).unwrap();
        fs::remove_file(tmp.0.join(IDENTITY_FILE)).unwrap();
        assert!(prepare(&tmp.0, 4).is_err());
        fs::write(tmp.0.join(IDENTITY_FILE), vec![0; MAX_IDENTITY + 1]).unwrap();
        assert!(prepare(&tmp.0, 4)
            .unwrap_err()
            .to_string()
            .contains("65536-byte bound"));
        fs::write(tmp.0.join(IDENTITY_FILE), b"x").unwrap();
        fs::write(tmp.0.join(FORMAT_FILE), FORMAT).unwrap();
        assert!(prepare(&tmp.0, 4)
            .unwrap_err()
            .to_string()
            .contains("silent downgrade"));
        // Original format recovers; missing completion still cannot be bypassed.
        fs::write(tmp.0.join(FORMAT_FILE), IDENTIFIED_FORMAT).unwrap();
        fs::remove_file(tmp.0.join(COMPLETE_FILE)).unwrap();
        assert!(prepare(&tmp.0, 4)
            .unwrap_err()
            .to_string()
            .contains("incomplete record"));
    }
    #[test]
    fn content_manifest_known_digest_and_empty_roundtrip() {
        let tmp = Temp::new();
        write_identified(
            &tmp.0,
            [b"B".as_slice(), b""].into_iter(),
            b"reason",
            Some(b"identity"),
        )
        .unwrap();
        let manifest = fs::read_to_string(tmp.0.join(COMPLETE_FILE)).unwrap();
        assert_eq!(
            manifest,
            concat!(
                "lloom-input-record-v3\n2\n",
                "1 df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c\n",
                "0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nEND\n"
            )
        );
        let replay = prepare(&tmp.0, 4).unwrap();
        assert!(replay.content_verified());
        assert_eq!(replay.read_input(0, 4).unwrap(), b"B");
        assert!(replay.read_input(1, 4).unwrap().is_empty());
    }
    #[test]
    fn same_length_corruption_refused_before_first_target_even_with_override() {
        let tmp = Temp::new();
        write_identified(
            &tmp.0,
            [b"B".as_slice(), b"C"].into_iter(),
            b"reason",
            Some(b"identity"),
        )
        .unwrap();
        fs::write(tmp.0.join(format!("{:020}.bin", 1)), b"D").unwrap();
        for allow_mismatch in [false, true] {
            let mut calls = 0;
            let error = super::super::replay_with_identity(
                &tmp.0,
                Some(b"identity"),
                allow_mismatch,
                |_| {
                    calls += 1;
                    Ok(())
                },
            )
            .unwrap_err();
            assert!(error.1.contains("content digest differs"), "{error:?}");
            assert_eq!(calls, 0, "corrupt later input executed earlier targets");
        }
    }
    #[test]
    fn same_length_replacement_after_preflight_refused_on_owned_read() {
        let tmp = Temp::new();
        write_identified(
            &tmp.0,
            [b"B".as_slice()].into_iter(),
            b"reason",
            Some(b"identity"),
        )
        .unwrap();
        let replay = prepare(&tmp.0, 4).unwrap();
        fs::write(&replay.paths[0], b"X").unwrap();
        assert_eq!(
            read(&replay.paths[0], replay.lengths[0], 4).unwrap(),
            b"X",
            "size-only control did not read replacement"
        );
        assert!(replay
            .read_input(0, 4)
            .unwrap_err()
            .to_string()
            .contains("content digest differs"));
    }
    #[test]
    fn legacy_v2_identity_keeps_reader_without_claiming_content_verification() {
        let tmp = Temp::new();
        fs::create_dir(&tmp.0).unwrap();
        write_new(
            &tmp.0.join(FORMAT_FILE),
            LEGACY_IDENTIFIED_FORMAT.as_bytes(),
        )
        .unwrap();
        write_new(&tmp.0.join(IDENTITY_FILE), b"identity").unwrap();
        write_new(&tmp.0.join("0000.bin"), b"B").unwrap();
        write_new(
            &tmp.0.join(COMPLETE_FILE),
            b"lloom-input-record-v2\n1\n1\nEND\n",
        )
        .unwrap();
        let replay = prepare(&tmp.0, 4).unwrap();
        assert!(!replay.content_verified());
        assert_eq!(replay.identity.as_deref(), Some(b"identity".as_slice()));
        assert_eq!(replay.read_input(0, 4).unwrap(), b"B");
        let mut calls = 0;
        assert_eq!(
            super::super::replay_with_identity(&tmp.0, Some(b"identity"), false, |bytes| {
                calls += 1;
                assert_eq!(bytes, b"B");
                Ok(())
            })
            .unwrap(),
            1
        );
        assert_eq!(calls, 1);
    }
    #[test]
    fn malformed_content_digest_refused_without_silent_legacy_fallback() {
        let tmp = Temp::new();
        write_identified(
            &tmp.0,
            [b"B".as_slice()].into_iter(),
            b"reason",
            Some(b"identity"),
        )
        .unwrap();
        for digest in [
            String::new(),
            "0".repeat(63),
            "0".repeat(65),
            "G".repeat(64),
        ] {
            fs::write(
                tmp.0.join(COMPLETE_FILE),
                format!("{IDENTIFIED_FORMAT}1\n1 {digest}\nEND\n"),
            )
            .unwrap();
            assert!(prepare(&tmp.0, 4)
                .unwrap_err()
                .to_string()
                .contains("content digest"));
        }
    }
    #[test]
    fn identity_refusal_precedes_archive_creation() {
        let tmp = Temp::new();
        for identity in [b"".as_slice(), &vec![0; MAX_IDENTITY + 1]] {
            assert!(write_identified(
                &tmp.0,
                [b"B".as_slice()].into_iter(),
                b"reason",
                Some(identity)
            )
            .is_err());
            assert!(!tmp.0.exists(), "identity refusal created partial archive");
        }
    }
}
