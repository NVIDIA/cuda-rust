/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile libraries (`.tilelib`) that implement foreign tile functions,
//! configured on the host.
//!
//! A `#[cutile::foreign]` declaration names only the library function. The
//! host chooses the libraries that `tileiras` links:
//!
//! - process-wide defaults ([`set_default_tile_libraries`],
//!   [`add_default_tile_library`]), for libraries every kernel may use;
//! - per-launch additions ([`crate::hints::CompileOptions::tile_library`]).
//!
//! Both lists are combined, canonicalized, sorted and de-duplicated before
//! linking ([`resolve_tile_libraries`]). Each resolved library is identified
//! by its canonical path and the SHA-256 of its contents, plus, for an
//! implementation-provider library, the contents of the files it loads and
//! the settings that change its generated code. All of it is cache-key
//! material, in memory and on disk. A file is hashed again only when its
//! stamp (identity, length and times) changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use crate::error::JITError;

/// A tile library the host links into kernels that call foreign functions.
///
/// A *static* library carries precompiled code and needs only its path. An
/// *implementation-provider* library loads a shared library that generates
/// code for each call; declare every file it loads with
/// [`Self::dependency`] and every setting that changes the code it
/// generates with [`Self::setting`], so the JIT caches miss when either
/// changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileLibrarySpec {
    path: PathBuf,
    dependencies: Vec<PathBuf>,
    settings: BTreeMap<String, String>,
}

impl TileLibrarySpec {
    /// The library at `path`. A relative path is relative to the current
    /// directory when the kernel is compiled.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            dependencies: Vec::new(),
            settings: BTreeMap::new(),
        }
    }

    /// A file the library loads when `tileiras` links it (a provider's
    /// shared library, or data it reads). Its contents are cache-key
    /// material.
    pub fn dependency(mut self, path: impl Into<PathBuf>) -> Self {
        self.dependencies.push(path.into());
        self
    }

    /// A setting that changes the code the library generates (a version, an
    /// install root, an option the provider reads from its environment).
    /// Cache-key material; it is not passed to `tileiras`.
    pub fn setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(key.into(), value.into());
        self
    }
}

static DEFAULTS: RwLock<Vec<TileLibrarySpec>> = RwLock::new(Vec::new());
/// Bumped on every change of [`DEFAULTS`], so a launch site can tell that
/// the defaults changed without comparing them.
static DEFAULTS_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Replaces the process-wide default tile libraries, which every kernel
/// compiled afterwards links in addition to its launch's
/// [`crate::hints::CompileOptions::tile_library`] list. Kernels already
/// compiled keep the libraries they were linked with; the next launch of a
/// kernel compiles it again when the combined list differs.
pub fn set_default_tile_libraries(libraries: impl IntoIterator<Item = TileLibrarySpec>) {
    let mut defaults = DEFAULTS.write().unwrap_or_else(|e| e.into_inner());
    *defaults = libraries.into_iter().collect();
    DEFAULTS_GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// Adds one process-wide default tile library (see
/// [`set_default_tile_libraries`]). Adding a library that is already a
/// default changes nothing, so kernels keep their cache keys.
pub fn add_default_tile_library(library: TileLibrarySpec) {
    let mut defaults = DEFAULTS.write().unwrap_or_else(|e| e.into_inner());
    if defaults.contains(&library) {
        return;
    }
    defaults.push(library);
    DEFAULTS_GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// The process-wide default tile libraries.
pub fn default_tile_libraries() -> Vec<TileLibrarySpec> {
    DEFAULTS.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// A counter that changes whenever the default tile libraries change.
pub fn default_tile_libraries_generation() -> u64 {
    DEFAULTS_GENERATION.load(Ordering::Acquire)
}

/// A tile library resolved for linking: `tileiras -l <path>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileLibrary {
    /// Canonical path passed to `tileiras`.
    pub path: PathBuf,
    /// SHA-256 over the file's contents as read by
    /// [`resolve_tile_libraries`], the declared dependencies (canonical paths
    /// and contents) and the settings. Cache key material: a library rebuilt
    /// before it is resolved never serves a cubin linked against its
    /// previous contents.
    pub digest: [u8; 32],
    /// The library and dependency files as they were before they were read.
    /// A file rebuilt after it was hashed, even while `tileiras` links it,
    /// no longer matches its stamp, and the JIT caches nothing linked then.
    pub stamps: TileLibraryStamps,
}

/// A file's device and inode, length, and modification and change times. A
/// file replaced by another (a build renamed into place) gets a new inode.
/// The change time (`ctime`) cannot be set back, so a file rewritten with
/// the same length and its modification time restored still gets a new
/// stamp, unless the file system's timestamps are too coarse to tell the two
/// writes apart. The stamps decide when a file is hashed again, when a
/// launch site rebuilds its key and whether a compile is cached.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    changed: Option<(i64, i64)>,
    file: Option<(u64, u64)>,
}

/// How long after its last change a file's hash is kept for reuse. File
/// systems record times in ticks (milliseconds, or seconds on some), and a
/// file written again within the tick of its stamp keeps that stamp.
const SETTLED: Duration = Duration::from_secs(2);

impl FileStamp {
    fn of(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        let (changed, file) = {
            use std::os::unix::fs::MetadataExt;
            (
                Some((metadata.ctime(), metadata.ctime_nsec())),
                Some((metadata.dev(), metadata.ino())),
            )
        };
        #[cfg(not(unix))]
        let (changed, file) = (None, None);
        Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            changed,
            file,
        }
    }

    /// Whether the file last changed at least [`SETTLED`] ago, so that a
    /// later change gets another stamp.
    fn settled(&self) -> bool {
        let changed = match self.changed {
            Some((secs, nanos)) => u64::try_from(secs).ok().and_then(|secs| {
                SystemTime::UNIX_EPOCH.checked_add(Duration::new(secs, nanos as u32))
            }),
            None => self.modified,
        };
        changed
            .and_then(|changed| SystemTime::now().duration_since(changed).ok())
            .is_some_and(|age| age >= SETTLED)
    }
}

fn file_stamp(path: &Path) -> std::io::Result<FileStamp> {
    std::fs::metadata(path).map(|metadata| FileStamp::of(&metadata))
}

/// Canonical path, SHA-256 and stamp of the file at `path`. The stamp is
/// taken before the file is read, so a file rewritten after it no longer
/// matches it.
///
/// The canonical path and hash are kept with the stamp, by the path as
/// given: while the file keeps its stamp, this costs one `stat`. A file that
/// changed less than [`SETTLED`] ago is read every time.
fn hash_file(path: &Path, what: &str) -> Result<(PathBuf, [u8; 32], FileStamp), JITError> {
    type Hashed = (FileStamp, PathBuf, [u8; 32]);
    static HASHED: Mutex<BTreeMap<PathBuf, Hashed>> = Mutex::new(BTreeMap::new());
    let display = path.display();
    let cannot_open = |e: std::io::Error| {
        JITError::Generic(format!(
            "{what} `{display}` cannot be opened: {e}. Check the tile libraries configured \
             with `CompileOptions::tile_library` and `set_default_tile_libraries`."
        ))
    };
    let metadata = std::fs::metadata(path).map_err(cannot_open)?;
    if !metadata.is_file() {
        return Err(JITError::Generic(format!(
            "{what} `{display}` is not a file"
        )));
    }
    let stamp = FileStamp::of(&metadata);
    let hashed = HASHED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(path)
        .filter(|(hashed, ..)| *hashed == stamp)
        .map(|(_, canonical, sha256)| (canonical.clone(), *sha256));
    if let Some((canonical, sha256)) = hashed {
        return Ok((canonical, sha256, stamp));
    }
    let canonical = std::fs::canonicalize(path).map_err(cannot_open)?;
    let bytes = std::fs::read(&canonical)
        .map_err(|e| JITError::Generic(format!("cannot read {what} `{display}`: {e}")))?;
    let sha256: [u8; 32] = Sha256::digest(&bytes).into();
    if stamp.settled() {
        HASHED.lock().unwrap_or_else(|e| e.into_inner()).insert(
            path.to_path_buf(),
            (stamp.clone(), canonical.clone(), sha256),
        );
    }
    Ok((canonical, sha256, stamp))
}

fn put(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

/// Resolves the tile libraries `specs` name: canonicalizes and hashes each
/// library and its dependencies, then sorts by canonical path and merges
/// entries for the same library. Two entries for one library merge their
/// dependencies and settings; a setting given two different values is an
/// error.
pub fn resolve_tile_libraries<'a>(
    specs: impl IntoIterator<Item = &'a TileLibrarySpec>,
) -> Result<Vec<TileLibrary>, JITError> {
    // Canonical library path -> (sha256, dependencies by canonical path,
    // settings, stamps of the files as given).
    type Merged = (
        [u8; 32],
        BTreeMap<PathBuf, [u8; 32]>,
        BTreeMap<String, String>,
        Vec<(PathBuf, FileStamp)>,
    );
    let mut merged: BTreeMap<PathBuf, Merged> = BTreeMap::new();
    for spec in specs {
        let (path, sha256, stamp) = hash_file(&spec.path, "tile library")?;
        let entry = merged
            .entry(path.clone())
            .or_insert_with(|| (sha256, BTreeMap::new(), BTreeMap::new(), Vec::new()));
        entry.3.push((spec.path.clone(), stamp));
        for dependency in &spec.dependencies {
            let (dep_path, dep_sha, dep_stamp) = hash_file(dependency, "tile library dependency")?;
            entry.1.insert(dep_path, dep_sha);
            entry.3.push((dependency.clone(), dep_stamp));
        }
        for (key, value) in &spec.settings {
            match entry.2.get(key) {
                Some(existing) if existing != value => {
                    return Err(JITError::Generic(format!(
                        "tile library `{}` is configured twice with different values of \
                         setting `{key}`: `{existing}` and `{value}`",
                        path.display()
                    )));
                }
                _ => {
                    entry.2.insert(key.clone(), value.clone());
                }
            }
        }
    }
    Ok(merged
        .into_iter()
        .map(|(path, (sha256, dependencies, settings, stamps))| {
            let mut h = Sha256::new();
            put(&mut h, b"contents");
            put(&mut h, &sha256);
            put(&mut h, b"dependencies");
            for (dep_path, dep_sha) in &dependencies {
                put(&mut h, dep_path.to_string_lossy().as_bytes());
                put(&mut h, dep_sha);
            }
            put(&mut h, b"settings");
            for (key, value) in &settings {
                put(&mut h, key.as_bytes());
                put(&mut h, value.as_bytes());
            }
            TileLibrary {
                path,
                digest: h.finalize().into(),
                stamps: TileLibraryStamps(stamps),
            }
        })
        .collect())
}

/// The tile libraries a compiled `module` links: none without foreign calls,
/// else `specs` resolved (see [`resolve_tile_libraries`]). A module with
/// foreign calls and no configured library is an error that names the
/// callees.
pub fn link_tile_libraries(
    module: &cutile_ir::Module,
    specs: &[TileLibrarySpec],
) -> Result<Vec<TileLibrary>, JITError> {
    link(&module.foreign_callees(), specs.is_empty(), || {
        resolve_tile_libraries(specs)
    })
}

/// The libraries a module calling the foreign functions `callees` links:
/// none without any, else `resolve()`, or, when no library is configured,
/// an error that names the callees.
fn link(
    callees: &[String],
    unconfigured: bool,
    resolve: impl FnOnce() -> Result<Vec<TileLibrary>, JITError>,
) -> Result<Vec<TileLibrary>, JITError> {
    if callees.is_empty() {
        return Ok(Vec::new());
    }
    if unconfigured {
        return Err(JITError::Generic(format!(
            "the kernel calls foreign tile function(s) {} but no tile library is configured; \
             add the library that implements them with \
             `CompileOptions::new().tile_library(TileLibrarySpec::new(path))` or \
             `set_default_tile_libraries`",
            callees
                .iter()
                .map(|c| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    resolve()
}

/// SHA-256 over resolved libraries (paths and digests), or `None` without
/// any.
pub fn tile_libraries_digest(libraries: &[TileLibrary]) -> Option<String> {
    if libraries.is_empty() {
        return None;
    }
    let mut h = Sha256::new();
    for lib in libraries {
        put(&mut h, lib.path.to_string_lossy().as_bytes());
        h.update(lib.digest);
    }
    Some(
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
    )
}

/// The files a [`TileLibrary`] was hashed from (the library and its
/// dependencies, as given), each with the length and times it had just
/// before it was read. Checking them costs one `stat` per file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TileLibraryStamps(Vec<(PathBuf, FileStamp)>);

impl TileLibraryStamps {
    /// Whether every stamped file still has its stamp.
    pub fn unchanged(&self) -> bool {
        self.0
            .iter()
            .all(|(path, stamp)| file_stamp(path).is_ok_and(|now| now == *stamp))
    }
}

/// The tile libraries one launch links, resolved once, before its kernel
/// compiles. The in-memory key, the `-l` list and the disk-cache key, and
/// the launch site all come from this one resolution, so they describe the
/// same files.
#[derive(Debug, Clone)]
pub struct TileLibrarySnapshot {
    /// [`default_tile_libraries_generation`], read before the default
    /// libraries were.
    generation: u64,
    /// The resolved libraries, or why they cannot be resolved: a kernel
    /// without foreign calls links none, so the error is reported only when
    /// it does.
    libraries: Result<Vec<TileLibrary>, String>,
}

impl TileLibrarySnapshot {
    /// Resolves `specs`, the libraries a launch links. `generation` is
    /// [`default_tile_libraries_generation`] read before the default
    /// libraries among `specs` were, so that a change of the defaults in
    /// between makes [`Self::unchanged`] false.
    pub fn new(generation: u64, specs: &[TileLibrarySpec]) -> Self {
        Self {
            generation,
            libraries: resolve_tile_libraries(specs).map_err(|e| e.to_string()),
        }
    }

    /// The digest of the resolved libraries (see [`tile_libraries_digest`]):
    /// `None` without any, or when they cannot be resolved, in which case a
    /// compile that links them reports why.
    pub fn digest(&self) -> Option<String> {
        self.libraries
            .as_ref()
            .ok()
            .and_then(|libraries| tile_libraries_digest(libraries))
    }

    /// The libraries a module calling the foreign functions `callees`
    /// ([`cutile_ir::Module::foreign_callees`]) links (see
    /// [`link_tile_libraries`]).
    pub fn link(&self, callees: &[String]) -> Result<Vec<TileLibrary>, JITError> {
        let unconfigured = matches!(&self.libraries, Ok(libraries) if libraries.is_empty());
        link(callees, unconfigured, || {
            self.libraries.clone().map_err(JITError::Generic)
        })
    }

    /// Whether every resolved file still has the stamp it had when it was
    /// hashed. After a compile, false means `tileiras` may have linked a
    /// newer build than the digest describes.
    pub fn files_unchanged(&self) -> bool {
        self.libraries
            .iter()
            .flatten()
            .all(|library| library.stamps.unchanged())
    }

    /// Whether the default libraries and every resolved file are as they
    /// were: a launch site reuses its kernel only then.
    pub fn unchanged(&self) -> bool {
        self.generation == default_tile_libraries_generation() && self.files_unchanged()
    }

    /// This snapshot without its files, for a kernel that calls no foreign
    /// function and so links no library: [`Self::unchanged`] then checks the
    /// generation only, without a `stat` per file.
    pub fn without_files(self) -> Self {
        Self {
            generation: self.generation,
            libraries: Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cutile_tile_library_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn libraries_are_canonicalized_sorted_and_merged() {
        let dir = scratch("merge");
        std::fs::write(dir.join("b.tilelib"), b"b").unwrap();
        std::fs::write(dir.join("a.tilelib"), b"a").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        let a = TileLibrarySpec::new(dir.join("a.tilelib"));
        let a_again = TileLibrarySpec::new(dir.join("sub/../a.tilelib"));
        let b = TileLibrarySpec::new(dir.join("b.tilelib"));
        let libs = resolve_tile_libraries([&b, &a, &a_again]).unwrap();
        let names: Vec<_> = libs
            .iter()
            .map(|l| l.path.file_name().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["a.tilelib", "b.tilelib"]);
        assert_eq!(
            tile_libraries_digest(&libs),
            tile_libraries_digest(&resolve_tile_libraries([&a, &b]).unwrap()),
            "order and duplicates do not change the digest"
        );
        assert_eq!(tile_libraries_digest(&[]), None);
    }

    #[test]
    fn dependencies_and_settings_are_key_material() {
        let dir = scratch("provider");
        std::fs::write(dir.join("p.tilelib"), b"p").unwrap();
        std::fs::write(dir.join("gen.so"), b"v1").unwrap();
        let plain = TileLibrarySpec::new(dir.join("p.tilelib"));
        let provider = plain
            .clone()
            .dependency(dir.join("gen.so"))
            .setting("version", "1");
        let digest = |spec: &TileLibrarySpec| {
            tile_libraries_digest(&resolve_tile_libraries([spec]).unwrap()).unwrap()
        };
        let before = digest(&provider);
        assert_ne!(before, digest(&plain));
        assert_ne!(before, digest(&provider.clone().setting("version", "2")));
        // Rewrite the dependency with a different length so the stamp changes
        // even on a file system with coarse modification times.
        std::fs::write(dir.join("gen.so"), b"v2-longer").unwrap();
        assert_ne!(before, digest(&provider));
    }

    #[test]
    fn conflicting_settings_and_missing_files_are_errors() {
        let dir = scratch("errors");
        std::fs::write(dir.join("x.tilelib"), b"x").unwrap();
        let one = TileLibrarySpec::new(dir.join("x.tilelib")).setting("k", "1");
        let two = TileLibrarySpec::new(dir.join("x.tilelib")).setting("k", "2");
        let message = resolve_tile_libraries([&one, &two])
            .unwrap_err()
            .to_string();
        assert!(message.contains("setting `k`"), "{message}");
        let missing = TileLibrarySpec::new(dir.join("missing.tilelib"));
        let message = resolve_tile_libraries([&missing]).unwrap_err().to_string();
        assert!(message.contains("cannot be opened"), "{message}");
        let message = resolve_tile_libraries([&TileLibrarySpec::new(&dir)])
            .unwrap_err()
            .to_string();
        assert!(message.contains("is not a file"), "{message}");
    }

    /// Registering a default library again (an init path run before each
    /// use) neither lists it twice nor changes the generation, which would
    /// give every kernel a new cache key.
    #[test]
    fn adding_a_default_library_twice_changes_nothing() {
        let dir = scratch("add_twice");
        std::fs::write(dir.join("d.tilelib"), b"d").unwrap();
        let spec = TileLibrarySpec::new(dir.join("d.tilelib"));
        let before = default_tile_libraries();
        add_default_tile_library(spec.clone());
        let generation = default_tile_libraries_generation();
        add_default_tile_library(spec.clone());
        assert_eq!(default_tile_libraries_generation(), generation);
        let defaults = default_tile_libraries();
        assert_eq!(defaults.iter().filter(|d| **d == spec).count(), 1);
        set_default_tile_libraries(before);
    }

    #[test]
    fn stamps_notice_a_rebuilt_library() {
        let dir = scratch("stamps");
        std::fs::write(dir.join("s.tilelib"), b"s").unwrap();
        std::fs::write(dir.join("gen.so"), b"g").unwrap();
        let spec = TileLibrarySpec::new(dir.join("s.tilelib")).dependency(dir.join("gen.so"));
        let snapshot = TileLibrarySnapshot::new(0, std::slice::from_ref(&spec));
        assert!(snapshot.files_unchanged());
        std::fs::write(dir.join("gen.so"), b"rebuilt").unwrap();
        assert!(!snapshot.files_unchanged(), "a rebuilt dependency");
        let snapshot = TileLibrarySnapshot::new(0, &[spec]);
        std::fs::write(dir.join("s.tilelib"), b"rebuilt").unwrap();
        assert!(!snapshot.files_unchanged(), "a rebuilt library");
    }
}
