/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent on-disk cubin cache: store trait, key derivation, entry codec,
//! and the process-global switch.
//!
//! The cache sits between bytecode serialization and the `tileiras` subprocess
//! (see `compile_bytecode_cached` in `cuda_tile_runtime_utils`). It is
//! **content-addressed**: the key is a SHA-256 over the serialized Tile IR
//! bytecode plus the remaining cache-key inputs (target, opt level, and the
//! `tileiras` binary's fingerprint). The bytecode already inlines every
//! dependency module, so changes represented in the serialized compiler input
//! change the key. Different `tileiras` binaries with the same fingerprint are
//! not distinguished.
//!
//! Disk persistence is **off by default and has no environment-variable
//! switch**. Callers opt in explicitly:
//!
//! ```rust,ignore
//! cutile::jit_cache::enable_default()?;                 // ~/.cache/cutile/kernels
//! cutile::jit_cache::enable(Arc::new(my_store));        // custom JitStore
//! ```
//!
//! Every store I/O failure is a soft error: it is counted in
//! [`stats().io_errors`](stats), logged, and the compile proceeds as if the
//! cache did not exist. The cache must never turn a working launch into a
//! failing one.

use sha2::{Digest, Sha256};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

/// Object-store-like interface for persisting compiled cubins.
///
/// `key` is the 64-char lowercase hex SHA-256 produced by the cache layer, so
/// implementations can use it directly as a file or object name. The stored
/// value is an encoded entry (header + cubin), opaque at this layer; all
/// validation happens above, in the cache layer.
///
/// Implementations must be safe under concurrent `put` of the same key from
/// multiple processes: a reader must see either nothing or one complete value,
/// never a partial write. `FileSystemJitStore` does this with a same-directory
/// temp file plus `rename`.
///
/// # Security
///
/// A cache hit hands stored bytes straight to `cuModuleLoadData` — it loads and
/// runs device code. The entry header is an *integrity* check (it catches torn
/// writes and hash collisions), **not** an authenticity check: anyone who can
/// write the backing storage can plant a valid entry wrapping a malicious cubin.
/// Back a `JitStore` only with storage writable solely by this user, or
/// otherwise trusted. On Unix, `FileSystemJitStore::default_location` creates
/// and validates a per-user `0700` directory. On Windows, it places the cache
/// under `%LOCALAPPDATA%` and relies on that location's default ACLs.
pub trait JitStore: Send + Sync + 'static {
    /// Returns the stored value, or `None` when the key is absent.
    fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>>;

    /// Stores a value, replacing any existing one for the key.
    fn put(&self, key: &str, value: &[u8]) -> io::Result<()>;

    /// Returns whether the key is present.
    ///
    /// The compile pipeline never calls this; it exists for warmup-style
    /// tooling. The default reads the value and drops it, so implementations
    /// only need to override when they have a cheaper existence check (as
    /// `FileSystemJitStore` does with a `stat`).
    fn contains(&self, key: &str) -> io::Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    /// Removes the key. Absent keys are not an error.
    fn delete(&self, key: &str) -> io::Result<()>;

    /// Removes every stored value.
    fn clear(&self) -> io::Result<()>;
}

// ── FileSystemJitStore ──────────────────────────────────────────────────────

/// Filesystem-backed [`JitStore`].
///
/// Entries live at `<root>/<first 2 hex chars>/<key>.cubin`; the two-char shard
/// keeps any single directory from accumulating tens of thousands of files.
///
/// Writes are atomic: the value goes to `<key>.tmp.<pid>.<counter>` in the same
/// directory and is `rename`d over the final name. No `fsync` — this is a
/// cache, not a database; a torn write after power loss fails the entry
/// header's payload checksum on read and is treated as a miss.
///
/// # Capacity and eviction
///
/// The store keeps no index. LRU state lives in the filesystem itself: a `get`
/// hit sets the entry's mtime to now (`atime` is unusable — `relatime` /
/// `noatime` mounts don't maintain it), and eviction deletes entries in mtime
/// order, oldest first, until total size falls to `capacity × low_watermark`.
///
/// Eviction is triggered from `put` — never from `get`, so hits never scan the
/// directory — by a coin flip weighted by the entry's size: one scan per
/// `capacity/16` bytes written, in expectation, summed over every process
/// sharing the store. The randomness is what makes that sum global; a
/// per-process byte counter sums only its own process's writes, which for any
/// realistic workload never reach the threshold, leaving the cap unenforced.
/// Cross-process
/// mutual exclusion uses a non-blocking exclusive lock on `<root>/.eviction.lock`
/// (`File::try_lock`, flock semantics): losing the race means another process
/// is already collecting, and this one just returns. Concurrent collection
/// would be correct anyway — a lost `delete` race reads as `NotFound` and is
/// skipped — the lock only avoids duplicate scans.
#[derive(Debug)]
pub struct FileSystemJitStore {
    root: PathBuf,
    /// Soft total-size target in bytes; `0` disables eviction entirely.
    capacity_bytes: u64,
    /// Eviction triggers when total size exceeds `capacity × high_watermark` …
    high_watermark: f64,
    /// … and deletes oldest-first until it is below `capacity × low_watermark`.
    low_watermark: f64,
}

/// Builder for [`FileSystemJitStore`].
pub struct FileSystemJitStoreBuilder {
    root: PathBuf,
    capacity_bytes: u64,
    high_watermark: f64,
    low_watermark: f64,
}

/// Default capacity target: 2 GiB. Matches cutile-python's 2 GiB setting.
pub const DEFAULT_CAPACITY_BYTES: u64 = 2 << 30;

/// Lock file for cross-process eviction mutual exclusion, directly under the
/// store root. Never evicted itself (eviction only scans the shard
/// subdirectories).
pub const EVICTION_LOCK_FILE_NAME: &str = ".eviction.lock";

/// Temp files this much older than "now" are leftovers of a crashed process
/// and are removed during eviction scans. Generous on purpose: a live `put`
/// holds its temp file for milliseconds, not minutes.
const STALE_TEMP_AGE: std::time::Duration = std::time::Duration::from_secs(3600);

impl FileSystemJitStoreBuilder {
    /// Soft total-size target in bytes; `0` means unbounded (no eviction).
    pub fn capacity_bytes(mut self, capacity_bytes: u64) -> Self {
        self.capacity_bytes = capacity_bytes;
        self
    }

    /// Eviction thresholds as fractions of `capacity_bytes`: collect when the
    /// total exceeds `capacity × high`, delete oldest-first down to
    /// `capacity × low`. Defaults `1.0` / `0.8`. The gap prevents a store
    /// hovering at capacity from scanning on every write.
    pub fn eviction_watermarks(mut self, high: f64, low: f64) -> Self {
        self.high_watermark = high;
        self.low_watermark = low;
        self
    }

    /// Creates the root directory (and parents) and opens the store.
    pub fn open(self) -> io::Result<FileSystemJitStore> {
        if !(self.low_watermark > 0.0
            && self.low_watermark <= self.high_watermark
            && self.high_watermark.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "eviction watermarks need 0 < low <= high, got high={} low={}",
                    self.high_watermark, self.low_watermark
                ),
            ));
        }
        std::fs::create_dir_all(&self.root)?;
        Ok(FileSystemJitStore {
            root: self.root,
            capacity_bytes: self.capacity_bytes,
            high_watermark: self.high_watermark,
            low_watermark: self.low_watermark,
        })
    }
}

/// Monotonic per-process counter for temp file names, so two threads in one
/// process writing the same key use distinct temp paths.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Uniform `u64` from a v4 uuid, for the eviction trigger's weighted coin flip.
///
/// Not a verbatim read of the first 8 bytes: a v4 uuid fixes 6 of its 128 bits
/// — the version nibble in byte 6 (always `0100`) and the two variant bits in
/// byte 8 — so `u64::from_le_bytes(&bytes[..8])` carries the version nibble at
/// bits 52–55 and every draw has bit 54 set, i.e. `draw >= 2^54`. The trigger
/// fires when `draw / 2^64 < len / threshold`; a floor of `2^-10` on the left
/// side means an entry under `threshold / 1024` — 128 KiB at the default
/// capacity, which is every real cubin — could *never* fire it, and a workload
/// of small kernels would grow the cache without bound.
///
/// XOR-folding the two halves fixes that: each fixed bit lands on an
/// independent uniformly random bit from the other half (byte 6 on byte 14,
/// byte 8 on byte 0), and the XOR of independent uniform bits is uniform, so
/// all 64 result bits are uniform. `eviction_draw_has_no_fixed_bits` pins this down.
fn eviction_draw(nonce: &uuid::Uuid) -> u64 {
    let b = nonce.as_bytes();
    let lo = u64::from_le_bytes(b[..8].try_into().expect("uuid is 16 bytes"));
    let hi = u64::from_le_bytes(b[8..].try_into().expect("uuid is 16 bytes"));
    lo ^ hi
}

impl FileSystemJitStore {
    /// Opens a store rooted at `dir` with the default capacity.
    pub fn new(dir: impl Into<PathBuf>) -> io::Result<Self> {
        Self::builder(dir).open()
    }

    pub fn builder(dir: impl Into<PathBuf>) -> FileSystemJitStoreBuilder {
        FileSystemJitStoreBuilder {
            root: dir.into(),
            capacity_bytes: DEFAULT_CAPACITY_BYTES,
            high_watermark: 1.0,
            low_watermark: 0.8,
        }
    }

    /// Opens a store at the conventional per-user cache location:
    /// `$XDG_CACHE_HOME/cutile/kernels` or `~/.cache/cutile/kernels` on Unix,
    /// and `%LOCALAPPDATA%\cutile\kernels` on Windows. Errors when none of those
    /// variables resolve rather than using a shared temp directory — the cache
    /// holds executable device code and must stay per-user.
    ///
    /// On Unix a new root is created as `0700`. A pre-existing root with any
    /// group or other permissions is rejected because tightening its permissions
    /// cannot make its existing contents trustworthy; otherwise it is tightened
    /// to `0700`. On Windows, the cache inherits `%LOCALAPPDATA%`'s default ACLs.
    /// Constructing the store does not enable anything; pass it to [`enable`].
    pub fn default_location() -> io::Result<Self> {
        let root = default_cache_dir()?.join("cutile").join("kernels");
        #[cfg(unix)]
        prepare_default_cache_root(&root)?;
        Self::new(root)
    }

    /// Root directory of this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry_path(&self, key: &str) -> io::Result<PathBuf> {
        // The key is produced by `l2_key`, but the trait is public: reject
        // anything that could escape the root or collide with temp names.
        let valid = key.len() == 64
            && key
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("jit store keys are 64 lowercase hex chars, got {key:?}"),
            ));
        }
        Ok(self.root.join(&key[..2]).join(format!("{key}.cubin")))
    }

    /// Evicts oldest entries until total size is below the low watermark.
    /// Every failure in here is soft: counted, logged,
    /// swallowed — eviction must never fail a `put`, let alone a launch.
    fn evict(&self) {
        // Cross-process mutual exclusion. flock-style: released when
        // `lock_file` drops, including on panic or process death.
        let lock_file = match std::fs::File::create(self.root.join(EVICTION_LOCK_FILE_NAME)) {
            Ok(f) => f,
            Err(e) => {
                STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                cache_log(format_args!("evict: cannot open lock file: {e}"));
                return;
            }
        };
        match lock_file.try_lock() {
            Ok(()) => {}
            // Another process is already collecting; its scan covers our writes.
            Err(std::fs::TryLockError::WouldBlock) => return,
            Err(std::fs::TryLockError::Error(e)) => {
                STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                cache_log(format_args!("evict: lock failed: {e}"));
                return;
            }
        }

        let now = std::time::SystemTime::now();
        let mut entries: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total: u64 = 0;

        let Ok(shards) = std::fs::read_dir(&self.root) else {
            STATS.io_errors.fetch_add(1, Ordering::Relaxed);
            return;
        };
        for shard in shards.flatten() {
            let is_shard = shard.file_name().len() == 2
                && shard
                    .file_name()
                    .to_str()
                    .is_some_and(|s| s.bytes().all(|b| b.is_ascii_hexdigit()));
            if !is_shard || !shard.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            let Ok(files) = std::fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let Ok(meta) = file.metadata() else { continue };
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                let name = file.file_name();
                let name = name.to_string_lossy();
                if name.contains(".tmp.") {
                    // Leftover of a crashed process; a live put holds its temp
                    // for milliseconds. Old enough → remove.
                    if now
                        .duration_since(mtime)
                        .is_ok_and(|age| age > STALE_TEMP_AGE)
                    {
                        let _ = std::fs::remove_file(file.path());
                    }
                    continue;
                }
                if !name.ends_with(".cubin") {
                    continue;
                }
                total += meta.len();
                entries.push((file.path(), meta.len(), mtime));
            }
        }

        let capacity = self.capacity_bytes as f64;
        if total as f64 <= capacity * self.high_watermark {
            return;
        }
        let target = capacity * self.low_watermark;

        // Oldest mtime first. A hit refreshed its entry's mtime in `get`, so
        // this deletes the least recently used. Readers racing a delete are
        // safe: an open fd survives the unlink (POSIX) / shares delete access
        // (Windows), and a subsequent `get` sees NotFound, which is a miss.
        entries.sort_by_key(|(_, _, mtime)| *mtime);
        let mut deleted = 0u64;
        for (path, len, _) in entries {
            if total as f64 <= target {
                break;
            }
            // A failed removal (in use on Windows, or already gone) is skipped.
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
                deleted += 1;
            }
        }
        cache_log(format_args!(
            "evict: deleted {deleted} entries, {total} bytes remain (capacity {})",
            self.capacity_bytes
        ));
    }
}

/// Logs a soft cache failure (and eviction summaries) when `CUTILE_JIT_LOG` is on.
/// The cache never escalates its own I/O problems beyond this line.
pub(crate) fn cache_log(msg: std::fmt::Arguments<'_>) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| crate::cuda_tile_runtime_utils::env_flag_enabled("CUTILE_JIT_LOG")) {
        eprintln!("[cutile::jit] {msg}");
    }
}

impl JitStore for FileSystemJitStore {
    fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        let path = self.entry_path(key)?;
        match std::fs::read(&path) {
            Ok(bytes) => {
                // LRU bookkeeping: a hit refreshes the entry's mtime so
                // eviction sees it as recently used. Best-effort — a failed
                // touch only makes the entry age faster.
                if self.capacity_bytes > 0 {
                    let times =
                        std::fs::FileTimes::new().set_modified(std::time::SystemTime::now());
                    let _ = std::fs::File::options()
                        .write(true)
                        .open(&path)
                        .and_then(|f| f.set_times(times));
                }
                Ok(Some(bytes))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn put(&self, key: &str, value: &[u8]) -> io::Result<()> {
        let final_path = self.entry_path(key)?;

        // An entry larger than the low watermark can never be retained:
        // eviction deletes down to `capacity × low_watermark`, so this value
        // alone would exceed that target and eviction would delete it (last, after
        // wiping every other entry). Decline it rather than storing it and
        // thrashing the whole cache on every write. `capacity_bytes == 0` means
        // unbounded — store anything.
        if self.capacity_bytes > 0
            && value.len() as f64 > self.capacity_bytes as f64 * self.low_watermark
        {
            cache_log(format_args!(
                "not caching {key}: entry is {} bytes, above the low watermark \
                 ({:.0} bytes); storing it would evict the entire cache",
                value.len(),
                self.capacity_bytes as f64 * self.low_watermark,
            ));
            return Ok(());
        }

        let shard = final_path
            .parent()
            .expect("entry path always has a shard parent");
        std::fs::create_dir_all(shard)?;

        // pid + counter disambiguate within a process; the random uuid also
        // disambiguates across processes that share this directory yet can
        // collide on pid — two containers in separate PID namespaces both see
        // pid 1 and counter 0 — so a concurrent put of the same key never writes
        // the same temp path and publishes a torn entry.
        let nonce = uuid::Uuid::new_v4();
        let temp_path = shard.join(format!(
            "{key}.tmp.{}.{}.{nonce}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        // Remove the temp file on either failure: a failed write (e.g. ENOSPC)
        // otherwise leaves a partial temp behind, and the only reaper is a
        // *successful* put's eviction trigger — which never fires while writes keep
        // failing, so the leak would grow unbounded.
        std::fs::write(&temp_path, value).inspect_err(|_| {
            let _ = std::fs::remove_file(&temp_path);
        })?;
        std::fs::rename(&temp_path, &final_path).inspect_err(|_| {
            let _ = std::fs::remove_file(&temp_path);
        })?;

        // Eviction trigger: scan with probability `len / threshold`, so one scan
        // happens per `threshold` bytes written *in expectation, across every
        // process sharing this store* — the amortization a running byte counter
        // cannot provide, because a counter only ever sums the writes of the
        // process holding it. That fails in both directions: a short-lived
        // process (a test run, a CI job, one rank of a multi-GPU launch) writes
        // a handful of cubins and exits, and a long-lived one only compiles each
        // specialization once, so neither accumulates the threshold — 128 MiB at
        // the default 2 GiB capacity, some 3800 cubins — and the cap goes
        // unenforced.
        //
        // `len >= threshold` makes the inequality hold for every draw, so a
        // large-enough entry always collects. Reusing the temp file's uuid costs
        // nothing: `put` is the miss path, which just paid for a `tileiras` run.
        if self.capacity_bytes > 0 {
            let threshold = u128::from((self.capacity_bytes / 16).max(1));
            let len = value.len() as u128;
            let draw = u128::from(eviction_draw(&nonce));
            // draw / 2^64 < len / threshold, in exact integer arithmetic.
            if draw * threshold < len << 64 {
                self.evict();
            }
        }
        Ok(())
    }

    fn contains(&self, key: &str) -> io::Result<bool> {
        // `metadata`, not `Path::exists()`: the latter maps every I/O error
        // (EACCES, ENOTDIR, …) to `false`, hiding a broken store behind the
        // `io::Result<bool>` signature. Match `delete`/`get`: absence is
        // `Ok(false)`, a real error propagates so the cache layer can count it.
        match std::fs::metadata(self.entry_path(key)?) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn delete(&self, key: &str) -> io::Result<()> {
        match std::fs::remove_file(self.entry_path(key)?) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn clear(&self) -> io::Result<()> {
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let is_shard = name.len() == 2
                && name
                    .to_str()
                    .is_some_and(|s| s.bytes().all(|b| b.is_ascii_hexdigit()));
            if is_shard && entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        Ok(())
    }
}

/// Creates a new default cache root privately, or validates an existing one
/// before trusting any entries it contains.
#[cfg(unix)]
fn prepare_default_cache_root(root: &Path) -> io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let parent = root.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("default JIT cache path has no parent: {}", root.display()),
        )
    })?;

    // Create missing parents privately, then create the final root separately
    // so `AlreadyExists` tells us atomically whether its contents predate us.
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(parent)?;

    match std::fs::DirBuilder::new().mode(0o700).create(root) {
        Ok(()) => {
            // A restrictive umask may remove owner bits; make the final mode
            // exactly 0700. No other user could enter the newly-created root
            // before this point.
            std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(root)?;
            if !metadata.file_type().is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "refusing to use default JIT cache path {} because it is not a directory; delete it and retry",
                        root.display()
                    ),
                ));
            }

            let mode = metadata.permissions().mode();
            if mode & 0o077 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "refusing to use pre-existing JIT cache directory {} with mode {:04o} because it grants permissions to group or others; delete it and retry",
                        root.display(),
                        mode & 0o777,
                    ),
                ));
            }

            // Retain contents only when the root was already private.
            // Merely removing group/other access now cannot make interior objects
            // trustworthy or revoke file descriptors opened beforehand.
            std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))
        }
        Err(e) => Err(e),
    }
}

fn default_cache_dir() -> io::Result<PathBuf> {
    #[cfg(unix)]
    {
        if let Some(dir) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(dir));
        }
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(home).join(".cache"));
        }
    }
    #[cfg(windows)]
    {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(dir));
        }
    }
    // Deliberately no fall back to a world-writable temp directory: a cache hit
    // loads and runs a stored cubin, and the entry checksum is integrity, not
    // authenticity, so a shared cache dir lets any local user plant executable
    // code. Make the caller pass an explicit path (or set the env var) instead.
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no per-user cache directory: set XDG_CACHE_HOME or HOME (Unix) / \
         LOCALAPPDATA (Windows), or open a FileSystemJitStore at an explicit path",
    ))
}

// ── Process-global switch ───────────────────────────────────────────────────

/// `RwLock`, not `OnceLock`: `enable`/`disable` must be repeatable so tests can
/// install their own stores. The read path runs only on an L1 miss, so the
/// lock cost is irrelevant.
static STORE: RwLock<Option<Arc<dyn JitStore>>> = RwLock::new(None);

/// Installs `store` as the process-wide cubin cache. Replaces any previous one.
///
/// A cache hit executes stored bytes as device code, so back `store` only with
/// storage writable solely by this user or otherwise trusted — see the security
/// note on [`JitStore`].
pub fn enable(store: Arc<dyn JitStore>) {
    *STORE.write().unwrap_or_else(PoisonError::into_inner) = Some(store);
}

/// Enables a [`FileSystemJitStore`] at the default location with the default
/// 2 GiB soft capacity target. See [`FileSystemJitStore::default_location`].
///
/// Errors instead of automatically falling back to a shared temp directory when
/// no per-user cache directory resolves (`HOME`/`XDG_CACHE_HOME` on Unix,
/// `LOCALAPPDATA` on Windows), so executable cubins are never automatically
/// placed in a shared temporary location.
pub fn enable_default() -> io::Result<()> {
    enable(Arc::new(FileSystemJitStore::default_location()?));
    Ok(())
}

/// Uninstalls the store. Later compiles neither read nor write the disk;
/// kernels already in the in-memory cache are unaffected.
pub fn disable() {
    *STORE.write().unwrap_or_else(PoisonError::into_inner) = None;
}

pub fn is_enabled() -> bool {
    STORE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some()
}

pub(crate) fn installed_store() -> Option<Arc<dyn JitStore>> {
    STORE.read().unwrap_or_else(PoisonError::into_inner).clone()
}

// ── Counters and stats ──────────────────────────────────────────────────────

/// Cumulative process-wide cache statistics. Snapshot via [`stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitCacheStats {
    /// Disk entries served (validated and returned).
    pub hits: u64,
    /// Lookups that found nothing usable and fell through to `tileiras`.
    pub misses: u64,
    /// Entries written.
    pub puts: u64,
    /// Total bytes handed to `JitStore::put`.
    pub bytes_written: u64,
    /// Soft I/O failures (reads, writes, deletes). Each one also logged via
    /// `CUTILE_JIT_LOG`. A nonzero value never fails a launch.
    pub io_errors: u64,
}

pub(crate) struct StatCounters {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub puts: AtomicU64,
    pub bytes_written: AtomicU64,
    pub io_errors: AtomicU64,
}

pub(crate) static STATS: StatCounters = StatCounters {
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    puts: AtomicU64::new(0),
    bytes_written: AtomicU64::new(0),
    io_errors: AtomicU64::new(0),
};

/// Snapshot of the cumulative cache statistics.
pub fn stats() -> JitCacheStats {
    JitCacheStats {
        hits: STATS.hits.load(Ordering::Relaxed),
        misses: STATS.misses.load(Ordering::Relaxed),
        puts: STATS.puts.load(Ordering::Relaxed),
        bytes_written: STATS.bytes_written.load(Ordering::Relaxed),
        io_errors: STATS.io_errors.load(Ordering::Relaxed),
    }
}

static BACKEND_COMPILES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn record_backend_compile() {
    BACKEND_COMPILES.fetch_add(1, Ordering::Relaxed);
}

/// Number of successful `tileiras` compiles in this process — the compiles the
/// disk cache did *not* absorb.
///
/// All counters here are **per process**; do not aggregate them across
/// processes that share one cache directory. With `cutile`'s in-memory cache in
/// front, the invariant
/// `jit_compile_count() == jit_backend_compile_count() + jit_disk_hit_count()`
/// holds only within a single process and only when no cached cubin is rejected
/// by the driver: on that fallback one in-memory miss both hits the disk and
/// recompiles, so the right side runs one ahead. Across processes, two racing
/// on the same key each compile and store (last write wins, one entry remains),
/// which over-counts backend compiles by one per race but never under-counts.
/// Direct calls to `run_tileiras` / `compile_tile_ir_module` (tests, tools)
/// bump only this counter.
pub fn jit_backend_compile_count() -> u64 {
    BACKEND_COMPILES.load(Ordering::Relaxed)
}

/// Number of compiles served from the disk cache in this process.
pub fn jit_disk_hit_count() -> u64 {
    STATS.hits.load(Ordering::Relaxed)
}

// ── L2 key ──────────────────────────────────────────────────────────────────

/// Domain separator, carrying the format version. Bump it when any encoding
/// rule changes: every old entry then misses naturally, no manual cache wipe.
const DOMAIN: &[u8] = b"cutile-jit-cubin-v1\0";

/// Version of the [`l2_key`] preimage. A cache entry that misses under a new
/// encoding just recompiles, but a key *persisted* outside the cache (a
/// `cutile::tune` record) has no such recovery: its holder must be able to
/// tell "the workspace drifted" from "this key was never comparable". Bump
/// this whenever the preimage changes, encoding rules and field set alike.
///
/// 1: `opt_level` alone. 2: `opt_level` plus the stage-2 flags byte.
pub const L2_KEY_SCHEMA: u32 = 2;

/// Content-addressed cache key: SHA-256 over the complete `tileiras` input.
///
/// The key deliberately contains no `module_name`, no generics, and no
/// `source_hash`: the serialized bytecode already inlines every dependency
/// module and represents those compiler-side inputs. Changes represented in
/// `bc` therefore change the key. The remaining fields record the target
/// (`gpu_name`), `opt_level`, and compiler identity (the `tileiras`
/// fingerprint).
pub fn l2_key(
    bc: &[u8],
    bc_version: cutile_ir::bytecode::BytecodeVersion,
    gpu_name: &str,
    opts: &crate::cuda_tile_runtime_utils::TileirasOptions,
    tileiras_fp: &str,
) -> String {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    // Every variable-length field is u64-length-prefixed; without it,
    // ("sm_9", "0…") and ("sm_90", "…") hash identically.
    put_field(&mut h, &[bc_version.major, bc_version.minor]);
    put_field(&mut h, &bc_version.tag.to_le_bytes());
    put_field(&mut h, bc);
    put_field(&mut h, gpu_name.as_bytes());
    // The stage-2 flags are key material: a debug or sanitizer cubin must
    // never be served for a release request (or vice versa).
    put_field(&mut h, &[opts.opt_level, opts.flags_byte()]);
    put_field(&mut h, tileiras_fp.as_bytes());
    hex(&h.finalize())
}

fn put_field(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

// ── Entry codec ─────────────────────────────────────────────────────────────
//
// A stored value is not a bare cubin; it carries a header so that a hit is
// re-validated field by field. Torn, corrupted, or request-mismatched entries
// fail validation and degrade to a recompile. This is an integrity check, not
// an authenticity check; see the security contract on `JitStore` above.
//
//   offset  size  field
//   0       12    magic            b"CUTILECUBIN\0"
//   12      2     format_version   u16 LE
//   14      2     reserved
//   16      32    payload_sha256   sha256(cubin)
//   48      32    bc_sha256        sha256(bytecode)
//   80      2     gpu_name_len     u16 LE
//   82      2     tileiras_fp_len  u16 LE
//   84      1     opt_level
//   85      3     padding
//   88      8     payload_len      u64 LE
//   96      ..    gpu_name ‖ tileiras_fp ‖ cubin

const ENTRY_MAGIC: &[u8; 12] = b"CUTILECUBIN\0";
const ENTRY_FORMAT_VERSION: u16 = 1;
const ENTRY_HEADER_LEN: usize = 96;

/// The request-side fields an entry is validated against on read.
pub struct EntryParams<'a> {
    pub bc_sha256: [u8; 32],
    pub gpu_name: &'a str,
    pub opt_level: u8,
    /// Packed stage-2 flags (see `TileirasOptions::flags_byte`). Stored in a
    /// formerly-padding header byte, which is why `ENTRY_FORMAT_VERSION` was
    /// deliberately not bumped: old readers never look at the byte, old
    /// entries carry 0 (the release encoding) in it. This is defense in
    /// depth — the flags also joined the key material, which orphaned all
    /// pre-flags entries (a one-time recompile), so the field can only
    /// disagree with a request under a key collision.
    pub flags: u8,
    pub tileiras_fp: &'a str,
}

/// Encodes a cache entry. `None` when `gpu_name` or `tileiras_fp` exceeds
/// `u16::MAX` bytes — then the compile simply is not cached.
pub fn encode_entry(params: &EntryParams<'_>, cubin: &[u8]) -> Option<Vec<u8>> {
    let gpu_len: u16 = params.gpu_name.len().try_into().ok()?;
    let fp_len: u16 = params.tileiras_fp.len().try_into().ok()?;

    let mut out = Vec::with_capacity(
        ENTRY_HEADER_LEN + params.gpu_name.len() + params.tileiras_fp.len() + cubin.len(),
    );
    out.extend_from_slice(ENTRY_MAGIC);
    out.extend_from_slice(&ENTRY_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]); // reserved
    out.extend_from_slice(&Sha256::digest(cubin));
    out.extend_from_slice(&params.bc_sha256);
    out.extend_from_slice(&gpu_len.to_le_bytes());
    out.extend_from_slice(&fp_len.to_le_bytes());
    out.push(params.opt_level);
    // Packed stage-2 flags in a formerly-padding byte (no format bump: old
    // readers ignore it, old entries carry the release encoding 0). The key
    // already separates flag sets; this validates them independently.
    out.push(params.flags);
    out.extend_from_slice(&[0u8; 2]); // padding
    out.extend_from_slice(&(cubin.len() as u64).to_le_bytes());
    debug_assert_eq!(out.len(), ENTRY_HEADER_LEN);
    out.extend_from_slice(params.gpu_name.as_bytes());
    out.extend_from_slice(params.tileiras_fp.as_bytes());
    out.extend_from_slice(cubin);
    Some(out)
}

/// Decodes and validates an entry against the current request. Returns the
/// cubin, or `None` when *anything* disagrees — magic, format version, sizes,
/// the payload checksum, or any request field. The caller treats `None` as a
/// miss and deletes the entry.
pub fn decode_entry(bytes: &[u8], params: &EntryParams<'_>) -> Option<Vec<u8>> {
    if bytes.len() < ENTRY_HEADER_LEN
        || &bytes[0..12] != ENTRY_MAGIC
        || u16::from_le_bytes(bytes[12..14].try_into().unwrap()) != ENTRY_FORMAT_VERSION
    {
        return None;
    }
    let payload_sha256: [u8; 32] = bytes[16..48].try_into().unwrap();
    let bc_sha256: [u8; 32] = bytes[48..80].try_into().unwrap();
    let gpu_len = u16::from_le_bytes(bytes[80..82].try_into().unwrap()) as usize;
    let fp_len = u16::from_le_bytes(bytes[82..84].try_into().unwrap()) as usize;
    let opt_level = bytes[84];
    let flags = bytes[85];
    let payload_len = u64::from_le_bytes(bytes[88..96].try_into().unwrap());

    let payload_len: usize = payload_len.try_into().ok()?;
    let expected_total = ENTRY_HEADER_LEN
        .checked_add(gpu_len)?
        .checked_add(fp_len)?
        .checked_add(payload_len)?;
    if bytes.len() != expected_total {
        return None;
    }

    let gpu_name = &bytes[ENTRY_HEADER_LEN..ENTRY_HEADER_LEN + gpu_len];
    let fp = &bytes[ENTRY_HEADER_LEN + gpu_len..ENTRY_HEADER_LEN + gpu_len + fp_len];
    let payload = &bytes[ENTRY_HEADER_LEN + gpu_len + fp_len..];

    if bc_sha256 != params.bc_sha256
        || gpu_name != params.gpu_name.as_bytes()
        || opt_level != params.opt_level
        || flags != params.flags
        || fp != params.tileiras_fp.as_bytes()
        || <[u8; 32]>::from(Sha256::digest(payload)) != payload_sha256
    {
        return None;
    }
    Some(payload.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile_ir::bytecode::BytecodeVersion;

    #[cfg(unix)]
    struct DefaultRootTestDir(PathBuf);

    #[cfg(unix)]
    impl DefaultRootTestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cutile_default_cache_{label}_{}_{}",
                std::process::id(),
                uuid::Uuid::new_v4(),
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn cache_root(&self) -> PathBuf {
            self.0.join("cutile").join("kernels")
        }
    }

    #[cfg(unix)]
    impl Drop for DefaultRootTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn unix_mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    #[test]
    fn default_cache_root_is_created_private() {
        let dir = DefaultRootTestDir::new("new");
        let root = dir.cache_root();

        prepare_default_cache_root(&root).unwrap();

        assert_eq!(unix_mode(&root), 0o700);
    }

    #[cfg(unix)]
    #[test]
    fn default_cache_root_retains_private_existing_directory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = DefaultRootTestDir::new("retain_private");
        let root = dir.cache_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("trusted-entry"), b"keep").unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();

        prepare_default_cache_root(&root).unwrap();

        assert_eq!(unix_mode(&root), 0o700);
        assert_eq!(std::fs::read(root.join("trusted-entry")).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn default_cache_root_rejects_existing_non_private_directory() {
        use std::os::unix::fs::PermissionsExt;

        for mode in [0o755, 0o750, 0o770, 0o702] {
            let dir = DefaultRootTestDir::new(&format!("reject_{mode:o}"));
            let root = dir.cache_root();
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("untrusted-entry"), b"do not trust").unwrap();
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(mode)).unwrap();

            let err = prepare_default_cache_root(&root).unwrap_err();

            assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
            assert!(err.to_string().contains("delete it and retry"));
            assert_eq!(unix_mode(&root), mode);
            assert!(root.join("untrusted-entry").exists());
        }
    }

    /// Every bit of the eviction draw must take both values across draws. A v4
    /// uuid fixes its version nibble (byte 6) and variant bits (byte 8), so a
    /// verbatim read of the first 8 bytes pins bit 54 to 1 — a `2^54` floor on
    /// the draw that silences the trigger for every entry under
    /// `threshold/1024` (128 KiB at the default capacity: all real cubins).
    /// 1024 draws make a stuck bit escape detection with probability
    /// `128 · 2^-1024`: never.
    #[test]
    fn eviction_draw_has_no_fixed_bits() {
        let mut seen_one = 0u64;
        let mut seen_zero = 0u64;
        for _ in 0..1024 {
            let draw = eviction_draw(&uuid::Uuid::new_v4());
            seen_one |= draw;
            seen_zero |= !draw;
        }
        assert_eq!(seen_one, u64::MAX, "bits never 1: {:#066b}", !seen_one);
        assert_eq!(
            seen_zero,
            u64::MAX,
            "bits never 0 (bit 54 stuck was the uuid version-nibble bug): {:#066b}",
            !seen_zero
        );
    }

    const V: BytecodeVersion = BytecodeVersion {
        major: 13,
        minor: 2,
        tag: 0,
    };

    fn params<'a>(bc: &[u8]) -> EntryParams<'a> {
        EntryParams {
            bc_sha256: Sha256::digest(bc).into(),
            gpu_name: "sm_90",
            opt_level: 3,
            flags: 0,
            tileiras_fp: "release 13.3, V13.3.36",
        }
    }

    fn opts(opt_level: u8) -> crate::cuda_tile_runtime_utils::TileirasOptions {
        crate::cuda_tile_runtime_utils::TileirasOptions {
            opt_level,
            ..Default::default()
        }
    }

    #[test]
    fn key_is_deterministic_and_field_sensitive() {
        let base = l2_key(b"bc", V, "sm_90", &opts(3), "fp");
        assert_eq!(base, l2_key(b"bc", V, "sm_90", &opts(3), "fp"));
        assert_eq!(base.len(), 64);
        assert!(base.bytes().all(|b| b.is_ascii_hexdigit()));

        assert_ne!(base, l2_key(b"bc2", V, "sm_90", &opts(3), "fp"));
        assert_ne!(base, l2_key(b"bc", V, "sm_80", &opts(3), "fp"));
        assert_ne!(base, l2_key(b"bc", V, "sm_90", &opts(0), "fp"));
        assert_ne!(base, l2_key(b"bc", V, "sm_90", &opts(3), "fp2"));
        let v2 = BytecodeVersion {
            major: 13,
            minor: 3,
            tag: 0,
        };
        assert_ne!(base, l2_key(b"bc", v2, "sm_90", &opts(3), "fp"));

        // Every stage-2 flag is key material: a debug, lineinfo, or
        // sanitizer cubin must never share a slot with a release one.
        let mut flagged = opts(3);
        flagged.device_debug = true;
        assert_ne!(base, l2_key(b"bc", V, "sm_90", &flagged, "fp"));
        let mut flagged = opts(3);
        flagged.lineinfo = true;
        assert_ne!(base, l2_key(b"bc", V, "sm_90", &flagged, "fp"));
        let mut flagged = opts(3);
        flagged.sanitize_memcheck = true;
        assert_ne!(base, l2_key(b"bc", V, "sm_90", &flagged, "fp"));
    }

    #[test]
    fn key_length_prefix_blocks_field_boundary_shifts() {
        // Same concatenated bytes, different field split.
        let a = l2_key(b"bc", V, "sm_9", &opts(3), "0fp");
        let b = l2_key(b"bc", V, "sm_90", &opts(3), "fp");
        assert_ne!(a, b);

        let c = l2_key(b"bcX", V, "sm_90", &opts(3), "fp");
        let d = l2_key(b"bc", V, "Xsm_90", &opts(3), "fp");
        assert_ne!(c, d);
    }

    #[test]
    fn entry_roundtrip() {
        let bc = b"some bytecode";
        let cubin = b"the compiled cubin".to_vec();
        let p = params(bc);
        let encoded = encode_entry(&p, &cubin).unwrap();
        assert_eq!(decode_entry(&encoded, &p), Some(cubin));
    }

    #[test]
    fn entry_rejects_wrong_bc_sha256() {
        // The collision defense: a key match with a different .bc digest is
        // *not* served.
        let p_write = params(b"bytecode A");
        let encoded = encode_entry(&p_write, b"cubin").unwrap();
        let p_read = params(b"bytecode B");
        assert_eq!(decode_entry(&encoded, &p_read), None);
    }

    #[test]
    fn entry_rejects_flag_mismatches_and_accepts_legacy_zero() {
        let bc = b"bc";
        // A release entry written before the flags field existed carries 0
        // in the (then-padding) byte; a release request must still accept it.
        let release = params(bc);
        let encoded = encode_entry(&release, b"cubin").unwrap();
        assert!(decode_entry(&encoded, &release).is_some());

        // A debug request must not be served the release cubin, and vice
        // versa, even under an (impossible) key collision.
        let mut debug = params(bc);
        debug.flags = 1;
        assert_eq!(decode_entry(&encoded, &debug), None);
        let debug_encoded = encode_entry(&debug, b"debug cubin").unwrap();
        assert_eq!(decode_entry(&debug_encoded, &release), None);
        assert!(decode_entry(&debug_encoded, &debug).is_some());
    }

    #[test]
    fn entry_rejects_field_mismatches() {
        let bc = b"bc";
        let encoded = encode_entry(&params(bc), b"cubin").unwrap();

        let mut p = params(bc);
        p.gpu_name = "sm_80";
        assert_eq!(decode_entry(&encoded, &p), None);

        let mut p = params(bc);
        p.opt_level = 0;
        assert_eq!(decode_entry(&encoded, &p), None);

        let mut p = params(bc);
        p.tileiras_fp = "other fp";
        assert_eq!(decode_entry(&encoded, &p), None);
    }

    #[test]
    fn entry_rejects_corruption_truncation_and_trailing_junk() {
        let bc = b"bc";
        let p = params(bc);
        let encoded = encode_entry(&p, b"a cubin payload").unwrap();

        // Flip one payload byte: checksum fails.
        let mut corrupt = encoded.clone();
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert_eq!(decode_entry(&corrupt, &p), None);

        // Truncations at every prefix length, including inside the header.
        for len in [0, 11, ENTRY_HEADER_LEN - 1, encoded.len() - 1] {
            assert_eq!(decode_entry(&encoded[..len], &p), None, "len {len}");
        }

        // Trailing junk: total length no longer matches the header.
        let mut long = encoded.clone();
        long.push(0);
        assert_eq!(decode_entry(&long, &p), None);

        // Wrong magic / version.
        let mut bad_magic = encoded.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(decode_entry(&bad_magic, &p), None);
        let mut bad_version = encoded;
        bad_version[12] ^= 0xff;
        assert_eq!(decode_entry(&bad_version, &p), None);
    }

    #[test]
    fn entry_rejects_oversized_names() {
        let p = EntryParams {
            bc_sha256: [0; 32],
            gpu_name: "sm_90",
            opt_level: 3,
            flags: 0,
            tileiras_fp: &"x".repeat(usize::from(u16::MAX) + 1),
        };
        assert_eq!(encode_entry(&p, b"cubin"), None);
    }
    #[test]
    fn l2_key_preimage_is_pinned_to_the_schema() {
        // This digest is the point: it pins the key PREIMAGE, not just its
        // properties. If this assertion fails, you changed the l2 key
        // encoding (field set, field order, or field encoding). That is
        // allowed — but you MUST bump L2_KEY_SCHEMA, update this digest,
        // and extend the schema history in its doc comment. Persisted keys
        // (cutile::tune records) rely on the schema to tell an encoding
        // migration apart from a stale winner; changing the preimage
        // without the bump silently re-creates the #236/#241 bug.
        assert_eq!(
            l2_key(b"bc", V, "sm_90", &opts(3), "fp"),
            "45274c4d279d97c5e383d8a8b035225d5ec81c91bd9f27c3027f5f5593473533",
            "l2 key preimage changed: bump L2_KEY_SCHEMA (see comment above)"
        );
    }
}
