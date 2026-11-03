/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Content-addressed cache support for the libNVVM + nvJitLink pipeline.
//!
//! A cache hit is accepted only after the entry manifest, cubin, and optional
//! LTOIR have all been checked. Callers always receive owned bytes and must
//! load those bytes, rather than reopening a cache path after validation.

use cuda_artifact_finalizer::is_valid_cubin;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const CACHE_DIRECTORY: &str = "ltoir-cubin-cache";
const CACHE_VERSION: &str = "v1";
const MANIFEST_FILE: &str = "manifest.bin";
const CUBIN_FILE: &str = "image.cubin";
const LTOIR_FILE: &str = "image.ltoir";
const MANIFEST_MAGIC: &[u8] = b"cuda-oxide-ltoir-cache-entry\0";
const MANIFEST_VERSION: u32 = 1;
const DIGEST_LENGTH: usize = 32;
const MAX_CACHE_ARTIFACT_LENGTH: u64 = 1 << 30;
#[cfg(test)]
const ELF64_HEADER_LENGTH: usize = 64;
#[cfg(test)]
const ELF64_PROGRAM_HEADER_LENGTH: u16 = 56;
const MANIFEST_LENGTH: usize =
    MANIFEST_MAGIC.len() + 4 + DIGEST_LENGTH + 8 + DIGEST_LENGTH + 1 + 8 + DIGEST_LENGTH;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Hash bytes exactly as they appear in a cache input.
pub(crate) fn digest_bytes(bytes: &[u8]) -> [u8; DIGEST_LENGTH] {
    Sha256::digest(bytes).into()
}

/// Fresh artifacts returned by the expensive compiler/linker pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BuiltArtifacts {
    pub(crate) cubin: Vec<u8>,
    pub(crate) ltoir: Option<Vec<u8>>,
}

impl BuiltArtifacts {
    pub(crate) fn new(cubin: Vec<u8>, ltoir: Option<Vec<u8>>) -> Self {
        Self { cubin, ltoir }
    }
}

/// Owned, fully verified artifacts returned to the loader.
///
/// The immutable paths are present only when a complete cache entry was read
/// or published successfully. They are informational and useful for copying
/// compatibility sidecars; the loader should consume the owned bytes.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CacheResult {
    pub(crate) cubin: Vec<u8>,
    pub(crate) ltoir: Option<Vec<u8>>,
    pub(crate) immutable_cubin_path: Option<PathBuf>,
    pub(crate) immutable_ltoir_path: Option<PathBuf>,
    pub(crate) cache_hit: bool,
}

impl CacheResult {
    fn uncached(artifacts: BuiltArtifacts) -> Self {
        Self {
            cubin: artifacts.cubin,
            ltoir: artifacts.ltoir,
            immutable_cubin_path: None,
            immutable_ltoir_path: None,
            cache_hit: false,
        }
    }

    fn from_stored(stored: StoredArtifacts, cache_hit: bool) -> Self {
        Self {
            cubin: stored.artifacts.cubin,
            ltoir: stored.artifacts.ltoir,
            immutable_cubin_path: Some(stored.cubin_path),
            immutable_ltoir_path: stored.ltoir_path,
            cache_hit,
        }
    }
}

/// Return a verified cache entry or build it once under a per-key file lock.
///
/// `source_dir` is the directory containing the source artifact. Cache data is
/// kept below `source_dir/.oxide-artifacts/ltoir-cubin-cache/v1/`. Every cache
/// I/O error is treated as an optimization failure: the builder still runs and
/// its result is returned with no immutable path. Errors from `build` itself
/// are always propagated.
pub(crate) fn cache_or_build<E, F>(
    source_dir: &Path,
    key: &[u8; DIGEST_LENGTH],
    build: F,
) -> Result<CacheResult, E>
where
    F: FnOnce() -> Result<BuiltArtifacts, E>,
{
    let paths = CachePaths::new(source_dir, key);

    if let Ok(Some(stored)) = read_valid_entry(&paths.entry, key) {
        return Ok(CacheResult::from_stored(stored, true));
    }

    let _lock = match acquire_key_lock(&paths) {
        Ok(lock) => lock,
        Err(_) => return build().map(CacheResult::uncached),
    };

    // Another process may have published this key while this process waited.
    if let Ok(Some(stored)) = read_valid_entry(&paths.entry, key) {
        return Ok(CacheResult::from_stored(stored, true));
    }

    let artifacts = build()?;
    if !is_valid_cubin(&artifacts.cubin) {
        return Ok(CacheResult::uncached(artifacts));
    }

    match publish_entry(&paths, key, &artifacts) {
        Ok(stored) => Ok(CacheResult::from_stored(stored, false)),
        Err(_) => Ok(CacheResult::uncached(artifacts)),
    }
}

struct CachePaths {
    entries: PathBuf,
    locks: PathBuf,
    temporary: PathBuf,
    entry: PathBuf,
    lock: PathBuf,
}

impl CachePaths {
    fn new(source_dir: &Path, key: &[u8; DIGEST_LENGTH]) -> Self {
        let root = source_dir
            .join(".oxide-artifacts")
            .join(CACHE_DIRECTORY)
            .join(CACHE_VERSION);
        let entries = root.join("entries");
        let locks = root.join("locks");
        let temporary = root.join("tmp");
        let key_hex = hex_digest(key);

        Self {
            entry: entries.join(&key_hex),
            lock: locks.join(format!("{key_hex}.lock")),
            entries,
            locks,
            temporary,
        }
    }
}

fn acquire_key_lock(paths: &CachePaths) -> io::Result<File> {
    fs::create_dir_all(&paths.entries)?;
    fs::create_dir_all(&paths.locks)?;
    fs::create_dir_all(&paths.temporary)?;

    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&paths.lock)?;
    lock.lock()?;
    Ok(lock)
}

#[derive(Debug)]
struct StoredArtifacts {
    artifacts: BuiltArtifacts,
    cubin_path: PathBuf,
    ltoir_path: Option<PathBuf>,
}

fn read_valid_entry(
    entry: &Path,
    expected_key: &[u8; DIGEST_LENGTH],
) -> io::Result<Option<StoredArtifacts>> {
    let metadata = match fs::symlink_metadata(entry) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_dir() {
        return Ok(None);
    }

    let manifest_path = entry.join(MANIFEST_FILE);
    let Some(manifest_bytes) = read_entry_file(&manifest_path, MANIFEST_LENGTH as u64)? else {
        return Ok(None);
    };
    let Some(manifest) = Manifest::decode(&manifest_bytes) else {
        return Ok(None);
    };
    if &manifest.key != expected_key {
        return Ok(None);
    }

    let expected_names: &[&str] = if manifest.has_ltoir {
        &[MANIFEST_FILE, CUBIN_FILE, LTOIR_FILE]
    } else {
        &[MANIFEST_FILE, CUBIN_FILE]
    };
    if !directory_contains_exact_regular_files(entry, expected_names)? {
        return Ok(None);
    }

    let cubin_path = entry.join(CUBIN_FILE);
    let Some(cubin) = read_entry_file(&cubin_path, manifest.cubin_length)? else {
        return Ok(None);
    };
    if digest_bytes(&cubin) != manifest.cubin_digest || !is_valid_cubin(&cubin) {
        return Ok(None);
    }

    let (ltoir, ltoir_path) = if manifest.has_ltoir {
        let path = entry.join(LTOIR_FILE);
        let Some(bytes) = read_entry_file(&path, manifest.ltoir_length)? else {
            return Ok(None);
        };
        if digest_bytes(&bytes) != manifest.ltoir_digest {
            return Ok(None);
        }
        (Some(bytes), Some(path))
    } else {
        if manifest.ltoir_length != 0 || manifest.ltoir_digest != [0; DIGEST_LENGTH] {
            return Ok(None);
        }
        (None, None)
    };

    Ok(Some(StoredArtifacts {
        artifacts: BuiltArtifacts { cubin, ltoir },
        cubin_path,
        ltoir_path,
    }))
}

fn read_entry_file(path: &Path, expected_length: u64) -> io::Result<Option<Vec<u8>>> {
    match read_regular_file_exact(path, expected_length) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::InvalidData
                    | io::ErrorKind::UnexpectedEof
                    | io::ErrorKind::IsADirectory
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn read_regular_file_exact(path: &Path, expected_length: u64) -> io::Result<Vec<u8>> {
    if expected_length > MAX_CACHE_ARTIFACT_LENGTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact exceeds the maximum supported size",
        ));
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact is not a regular file",
        ));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW prevents a final symlink from escaping the cache, while
        // O_NONBLOCK prevents a raced FIFO/device replacement from blocking.
        // Named constants, not literal octal: these flag values are
        // architecture-specific. For example, O_NOFOLLOW is 0o400000 on
        // x86-64 but 0o100000 on aarch64, so the x86-64 literal does not
        // prevent an aarch64 cache read from following a symlink.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|error| {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if error.raw_os_error() == Some(libc::ELOOP) {
            return io::Error::new(io::ErrorKind::InvalidData, "cache artifact is a symlink");
        }
        error
    })?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != expected_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact is not a regular file of the expected size",
        ));
    }

    let expected_length = usize::try_from(expected_length).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact does not fit in memory on this platform",
        )
    })?;
    let mut bytes = Vec::with_capacity(expected_length);
    file.take(expected_length as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() != expected_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact changed size while it was read",
        ));
    }
    Ok(bytes)
}

fn directory_contains_exact_regular_files(dir: &Path, expected: &[&str]) -> io::Result<bool> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Ok(false);
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Ok(false);
        };
        names.push(name);
    }

    names.sort_unstable();
    let mut expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    expected.sort_unstable();
    Ok(names == expected)
}

#[derive(Debug)]
struct Manifest {
    key: [u8; DIGEST_LENGTH],
    cubin_length: u64,
    cubin_digest: [u8; DIGEST_LENGTH],
    has_ltoir: bool,
    ltoir_length: u64,
    ltoir_digest: [u8; DIGEST_LENGTH],
}

impl Manifest {
    fn for_artifacts(key: &[u8; DIGEST_LENGTH], artifacts: &BuiltArtifacts) -> io::Result<Self> {
        let cubin_length = u64::try_from(artifacts.cubin.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cubin exceeds cache format limit",
            )
        })?;
        if cubin_length > MAX_CACHE_ARTIFACT_LENGTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cubin exceeds cache size limit",
            ));
        }
        let (has_ltoir, ltoir_length, ltoir_digest) = match &artifacts.ltoir {
            Some(ltoir) => {
                let length = u64::try_from(ltoir.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "LTOIR exceeds cache format limit",
                    )
                })?;
                if length > MAX_CACHE_ARTIFACT_LENGTH {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "LTOIR exceeds cache size limit",
                    ));
                }
                (true, length, digest_bytes(ltoir))
            }
            None => (false, 0, [0; DIGEST_LENGTH]),
        };

        Ok(Self {
            key: *key,
            cubin_length,
            cubin_digest: digest_bytes(&artifacts.cubin),
            has_ltoir,
            ltoir_length,
            ltoir_digest,
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MANIFEST_LENGTH);
        bytes.extend_from_slice(MANIFEST_MAGIC);
        bytes.extend_from_slice(&MANIFEST_VERSION.to_be_bytes());
        bytes.extend_from_slice(&self.key);
        bytes.extend_from_slice(&self.cubin_length.to_be_bytes());
        bytes.extend_from_slice(&self.cubin_digest);
        bytes.push(u8::from(self.has_ltoir));
        bytes.extend_from_slice(&self.ltoir_length.to_be_bytes());
        bytes.extend_from_slice(&self.ltoir_digest);
        debug_assert_eq!(bytes.len(), MANIFEST_LENGTH);
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != MANIFEST_LENGTH || !bytes.starts_with(MANIFEST_MAGIC) {
            return None;
        }

        let mut cursor = MANIFEST_MAGIC.len();
        let version = take_array::<4>(bytes, &mut cursor).map(u32::from_be_bytes)?;
        if version != MANIFEST_VERSION {
            return None;
        }
        let key = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        let cubin_length = take_array::<8>(bytes, &mut cursor).map(u64::from_be_bytes)?;
        let cubin_digest = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        let has_ltoir = match *bytes.get(cursor)? {
            0 => false,
            1 => true,
            _ => return None,
        };
        cursor += 1;
        let ltoir_length = take_array::<8>(bytes, &mut cursor).map(u64::from_be_bytes)?;
        let ltoir_digest = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        if cursor != bytes.len() {
            return None;
        }

        Some(Self {
            key,
            cubin_length,
            cubin_digest,
            has_ltoir,
            ltoir_length,
            ltoir_digest,
        })
    }
}

fn take_array<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let value = bytes.get(*cursor..end)?.try_into().ok()?;
    *cursor = end;
    Some(value)
}

fn publish_entry(
    paths: &CachePaths,
    key: &[u8; DIGEST_LENGTH],
    artifacts: &BuiltArtifacts,
) -> io::Result<StoredArtifacts> {
    let manifest = Manifest::for_artifacts(key, artifacts)?;
    let mut temporary = PendingDirectory::create(&paths.temporary, key)?;

    write_synced_file(&temporary.path.join(CUBIN_FILE), &artifacts.cubin)?;
    if let Some(ltoir) = &artifacts.ltoir {
        write_synced_file(&temporary.path.join(LTOIR_FILE), ltoir)?;
    }
    write_synced_file(&temporary.path.join(MANIFEST_FILE), &manifest.encode())?;
    sync_directory(&temporary.path)?;

    match read_valid_entry(&paths.entry, key) {
        Ok(Some(existing)) if existing.artifacts == *artifacts => return Ok(existing),
        Ok(Some(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "cache key already contains different verified output",
            ));
        }
        Ok(None) => remove_cache_path_if_present(&paths.entry)?,
        Err(error) => return Err(error),
    }

    fs::rename(&temporary.path, &paths.entry)?;
    temporary.published = true;
    sync_directory(&paths.entries)?;

    // Read through the same verification path used for future hits. This also
    // ensures the returned immutable paths refer to the bytes being returned.
    read_valid_entry(&paths.entry, key)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "published cache entry failed verification",
        )
    })
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn remove_cache_path_if_present(path: &Path) -> io::Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };

    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

struct PendingDirectory {
    path: PathBuf,
    published: bool,
}

impl PendingDirectory {
    fn create(root: &Path, key: &[u8; DIGEST_LENGTH]) -> io::Result<Self> {
        let key = hex_digest(key);
        for _ in 0..128 {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(".{key}.{}.{}.tmp", std::process::id(), id));
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique cache staging directory",
        ))
    }
}

impl Drop for PendingDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn hex_digest(digest: &[u8; DIGEST_LENGTH]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(DIGEST_LENGTH * 2);
    for byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            for _ in 0..128 {
                let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "cuda-oxide-ltoir-cache-{name}-{}-{id}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create test directory: {error}"),
                }
            }
            panic!("failed to allocate unique test directory");
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fake_cubin(marker: u8) -> Vec<u8> {
        let mut cubin = vec![0_u8; 120];
        cubin[..4].copy_from_slice(b"\x7fELF");
        cubin[4] = 2;
        cubin[5] = 1;
        cubin[6] = 1;
        cubin[15] = marker;
        cubin[16..18].copy_from_slice(&2_u16.to_le_bytes());
        cubin[18..20].copy_from_slice(&190_u16.to_le_bytes());
        cubin[20..24].copy_from_slice(&1_u32.to_le_bytes());
        cubin[32..40].copy_from_slice(&64_u64.to_le_bytes());
        cubin[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        cubin[54..56].copy_from_slice(&ELF64_PROGRAM_HEADER_LENGTH.to_le_bytes());
        cubin[56..58].copy_from_slice(&1_u16.to_le_bytes());

        // One PT_LOAD program header spanning the small synthetic file.
        let cubin_len = cubin.len() as u64;
        cubin[64..68].copy_from_slice(&1_u32.to_le_bytes());
        cubin[72..80].copy_from_slice(&0_u64.to_le_bytes());
        cubin[96..104].copy_from_slice(&cubin_len.to_le_bytes());
        cubin[104..112].copy_from_slice(&cubin_len.to_le_bytes());
        cubin[112..120].copy_from_slice(&8_u64.to_le_bytes());
        cubin
    }

    fn artifacts(marker: u8) -> BuiltArtifacts {
        BuiltArtifacts::new(fake_cubin(marker), Some(vec![b'L', b'T', b'O', marker]))
    }

    fn truncated_cubin_prefix() -> Vec<u8> {
        let mut cubin = vec![0_u8; 20];
        cubin[..4].copy_from_slice(b"\x7fELF");
        cubin[4] = 2;
        cubin[5] = 1;
        cubin[6] = 1;
        cubin[16..18].copy_from_slice(&2_u16.to_le_bytes());
        cubin[18..20].copy_from_slice(&190_u16.to_le_bytes());
        cubin
    }

    fn test_key() -> [u8; DIGEST_LENGTH] {
        digest_bytes(b"cuda-host cache test key")
    }

    #[test]
    fn sha256_matches_published_vectors_and_incremental_updates() {
        assert_eq!(
            hex_digest(&digest_bytes(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_digest(&digest_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(&digest_bytes(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex_digest(&digest_bytes(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );

        let mut incremental = Sha256::new();
        incremental.update(b"a");
        incremental.update(b"b");
        incremental.update(b"c");
        let incremental: [u8; DIGEST_LENGTH] = incremental.finalize().into();
        assert_eq!(incremental, digest_bytes(b"abc"));

        for length in [55, 56, 63, 64, 65, 127, 128, 129] {
            let bytes = (0..length).map(|value| value as u8).collect::<Vec<_>>();
            let mut chunked = Sha256::new();
            for chunk in bytes.chunks(7) {
                chunked.update(chunk);
            }
            let chunked: [u8; DIGEST_LENGTH] = chunked.finalize().into();
            assert_eq!(chunked, digest_bytes(&bytes), "length {length}");
        }
    }

    #[test]
    fn cubin_validation_rejects_truncated_headers_and_out_of_bounds_tables() {
        assert!(!is_valid_cubin(&truncated_cubin_prefix()));
        assert!(is_valid_cubin(&fake_cubin(1)));

        let mut out_of_bounds = fake_cubin(2);
        out_of_bounds[32..40].copy_from_slice(&119_u64.to_le_bytes());
        assert!(!is_valid_cubin(&out_of_bounds));
    }

    #[test]
    fn cache_reads_reject_oversized_lengths_before_allocating() {
        let dir = TestDirectory::new("oversized");
        let path = dir.path().join("image.cubin");
        fs::write(&path, b"small").unwrap();
        let error = read_regular_file_exact(&path, MAX_CACHE_ARTIFACT_LENGTH + 1).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn cache_reads_do_not_follow_artifact_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = TestDirectory::new("symlink");
        let outside = dir.path().join("outside");
        let link = dir.path().join("image.cubin");
        fs::write(&outside, b"outside bytes").unwrap();
        symlink(&outside, &link).unwrap();

        assert!(read_regular_file_exact(&link, 13).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn symlinked_cache_artifact_is_a_miss_and_is_repaired() {
        use std::os::unix::fs::symlink;

        let dir = TestDirectory::new("symlink-repair");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(20))).unwrap();
        let cubin_path = first.immutable_cubin_path.unwrap();
        let outside = dir.path().join("outside");
        fs::write(&outside, b"outside bytes").unwrap();
        fs::remove_file(&cubin_path).unwrap();
        symlink(&outside, &cubin_path).unwrap();

        let repaired = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(21))).unwrap();
        assert!(!repaired.cache_hit);
        assert_eq!(repaired.cubin, fake_cubin(21));
        assert!(repaired.immutable_cubin_path.is_some());
        assert_eq!(fs::read(outside).unwrap(), b"outside bytes");
    }

    #[test]
    fn second_request_is_a_verified_hit() {
        let dir = TestDirectory::new("hit");
        let key = test_key();
        let builds = AtomicU64::new(0);

        let first = cache_or_build(dir.path(), &key, || {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(artifacts(1))
        })
        .unwrap();
        assert!(!first.cache_hit);
        assert_eq!(first.cubin, fake_cubin(1));
        assert_eq!(first.ltoir, Some(vec![b'L', b'T', b'O', 1]));
        assert!(
            first
                .immutable_cubin_path
                .as_ref()
                .is_some_and(|path| path.is_file())
        );
        assert!(
            first
                .immutable_ltoir_path
                .as_ref()
                .is_some_and(|path| path.is_file())
        );

        let second = cache_or_build(dir.path(), &key, || {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(artifacts(2))
        })
        .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.cubin, first.cubin);
        assert_eq!(second.ltoir, first.ltoir);
        assert_eq!(second.immutable_cubin_path, first.immutable_cubin_path);
        assert_eq!(second.immutable_ltoir_path, first.immutable_ltoir_path);
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn standalone_ltoir_route_caches_without_intermediate() {
        let dir = TestDirectory::new("without-intermediate");
        let key = test_key();
        let output = BuiltArtifacts::new(fake_cubin(3), None);

        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(output.clone())).unwrap();
        assert_eq!(first.ltoir, None);
        assert_eq!(first.immutable_ltoir_path, None);
        assert!(first.immutable_cubin_path.is_some());

        let second = cache_or_build(dir.path(), &key, || -> Result<_, ()> {
            panic!("valid entry should avoid rebuilding")
        })
        .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.cubin, output.cubin);
        assert_eq!(second.ltoir, None);
    }

    #[test]
    fn corrupt_cubin_digest_is_a_miss_and_is_repaired() {
        let dir = TestDirectory::new("corrupt-cubin");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(4))).unwrap();
        let cubin_path = first.immutable_cubin_path.unwrap();

        let mut corrupted = fake_cubin(4);
        corrupted[32] ^= 0x80;
        fs::write(&cubin_path, corrupted).unwrap();

        let repaired = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(5))).unwrap();
        assert!(!repaired.cache_hit);
        assert_eq!(repaired.cubin, fake_cubin(5));
        assert!(repaired.immutable_cubin_path.is_some());

        let hit = cache_or_build(dir.path(), &key, || -> Result<_, ()> {
            panic!("repaired entry should be reusable")
        })
        .unwrap();
        assert!(hit.cache_hit);
        assert_eq!(hit.cubin, fake_cubin(5));
    }

    #[test]
    fn truncated_ltoir_and_manifest_are_misses() {
        let dir = TestDirectory::new("truncated");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(6))).unwrap();
        fs::write(first.immutable_ltoir_path.unwrap(), b"L").unwrap();

        let second = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(7))).unwrap();
        assert!(!second.cache_hit);
        assert_eq!(second.ltoir, Some(vec![b'L', b'T', b'O', 7]));

        let manifest = second
            .immutable_cubin_path
            .as_ref()
            .unwrap()
            .parent()
            .unwrap()
            .join(MANIFEST_FILE);
        fs::write(manifest, &MANIFEST_MAGIC[..8]).unwrap();

        let third = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(8))).unwrap();
        assert!(!third.cache_hit);
        assert_eq!(third.cubin, fake_cubin(8));
    }

    #[test]
    fn complete_entries_swapped_between_keys_are_rejected() {
        let dir = TestDirectory::new("swapped-entries");
        let key_a = digest_bytes(b"source A");
        let key_b = digest_bytes(b"source B");
        cache_or_build(dir.path(), &key_a, || Ok::<_, ()>(artifacts(20))).unwrap();
        cache_or_build(dir.path(), &key_b, || Ok::<_, ()>(artifacts(21))).unwrap();

        let paths_a = CachePaths::new(dir.path(), &key_a);
        let paths_b = CachePaths::new(dir.path(), &key_b);
        let swap = paths_a.entries.join("swap");
        fs::rename(&paths_a.entry, &swap).unwrap();
        fs::rename(&paths_b.entry, &paths_a.entry).unwrap();
        fs::rename(&swap, &paths_b.entry).unwrap();

        let repaired_a = cache_or_build(dir.path(), &key_a, || Ok::<_, ()>(artifacts(22))).unwrap();
        let repaired_b = cache_or_build(dir.path(), &key_b, || Ok::<_, ()>(artifacts(23))).unwrap();
        assert!(!repaired_a.cache_hit);
        assert!(!repaired_b.cache_hit);
        assert_eq!(repaired_a.cubin, fake_cubin(22));
        assert_eq!(repaired_b.cubin, fake_cubin(23));
    }

    #[test]
    fn partial_entries_and_staging_artifacts_are_ignored() {
        let dir = TestDirectory::new("partial");
        let key = test_key();
        let paths = CachePaths::new(dir.path(), &key);
        fs::create_dir_all(&paths.entry).unwrap();
        fs::write(paths.entry.join(CUBIN_FILE), fake_cubin(9)).unwrap();
        fs::create_dir_all(&paths.temporary).unwrap();
        let abandoned = paths.temporary.join("abandoned.tmp");
        fs::create_dir(&abandoned).unwrap();
        fs::write(abandoned.join(CUBIN_FILE), fake_cubin(10)).unwrap();

        let result = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(11))).unwrap();
        assert!(!result.cache_hit);
        assert_eq!(result.cubin, fake_cubin(11));
        assert!(abandoned.exists());
    }

    #[test]
    fn same_key_concurrency_builds_once() {
        let dir = Arc::new(TestDirectory::new("concurrent"));
        let key = test_key();
        let builds = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();

        for _ in 0..8 {
            let dir = Arc::clone(&dir);
            let builds = Arc::clone(&builds);
            let barrier = Arc::clone(&barrier);
            threads.push(thread::spawn(move || {
                barrier.wait();
                cache_or_build(dir.path(), &key, || {
                    builds.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(30));
                    Ok::<_, ()>(artifacts(12))
                })
                .unwrap()
            }));
        }

        let results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(results.iter().filter(|result| result.cache_hit).count(), 7);
        assert!(results.iter().all(|result| result.cubin == fake_cubin(12)));
        assert!(
            results
                .iter()
                .all(|result| result.ltoir == Some(vec![b'L', b'T', b'O', 12]))
        );
    }

    #[test]
    fn cache_setup_failure_still_returns_fresh_bytes() {
        let dir = TestDirectory::new("setup-failure");
        fs::write(dir.path().join(".oxide-artifacts"), b"not a directory").unwrap();
        let key = test_key();
        let builds = AtomicU64::new(0);

        for marker in [13, 14] {
            let result = cache_or_build(dir.path(), &key, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(artifacts(marker))
            })
            .unwrap();
            assert_eq!(result.cubin, fake_cubin(marker));
            assert_eq!(result.immutable_cubin_path, None);
            assert_eq!(result.immutable_ltoir_path, None);
            assert!(!result.cache_hit);
        }
        assert_eq!(builds.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn publication_failure_after_build_still_returns_fresh_bytes() {
        let dir = TestDirectory::new("publication-failure");
        let key = test_key();
        let paths = CachePaths::new(dir.path(), &key);

        let result = cache_or_build(dir.path(), &key, || {
            // Lock acquisition has already created this directory. Replace it
            // with a file so the atomic rename itself cannot be published.
            fs::remove_dir_all(&paths.entries).unwrap();
            fs::write(&paths.entries, b"block publication").unwrap();
            Ok::<_, ()>(artifacts(15))
        })
        .unwrap();

        assert_eq!(result.cubin, fake_cubin(15));
        assert_eq!(result.ltoir, Some(vec![b'L', b'T', b'O', 15]));
        assert_eq!(result.immutable_cubin_path, None);
        assert_eq!(result.immutable_ltoir_path, None);
        assert!(!result.cache_hit);
    }

    #[test]
    fn non_cubin_output_is_never_cached() {
        let dir = TestDirectory::new("invalid-output");
        let key = test_key();
        let builds = AtomicU64::new(0);

        for _ in 0..2 {
            let result = cache_or_build(dir.path(), &key, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(BuiltArtifacts::new(
                    truncated_cubin_prefix(),
                    Some(vec![1, 2, 3]),
                ))
            })
            .unwrap();
            assert_eq!(result.immutable_cubin_path, None);
        }
        assert_eq!(builds.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn builder_errors_are_not_hidden_by_cache_fallback() {
        let dir = TestDirectory::new("builder-error");
        let key = test_key();
        let result = cache_or_build(dir.path(), &key, || {
            Err::<BuiltArtifacts, _>("compile failed")
        });
        assert_eq!(result.unwrap_err(), "compile failed");
    }
}
