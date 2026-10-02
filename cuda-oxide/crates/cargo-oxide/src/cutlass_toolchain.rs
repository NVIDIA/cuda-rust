/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Managed installation of NVIDIA's public CUTLASS compiler bundle.
//!
//! The archive is a release artifact, not part of the CUDA Toolkit.  Keep the
//! release URL and digest together here so an install can never silently drift
//! to a newer compiler.  Publication is a same-filesystem directory rename:
//! builds either see the previous complete install or the new complete install,
//! never an extraction in progress.

use sha2::Digest as _;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CUTLASS_COMPILER_ENV: &str = "CUDA_OXIDE_CUTLASS_COMPILER";

const CUTLASS_VERSION: &str = "4.7.0";
const CUTLASS_PLATFORM: &str = "x86_64-cu13";
const ARCHIVE_NAME: &str = "cutlass-install-x86_64-cu13-4.7.0.tar.gz";
const ARCHIVE_URL: &str = "https://github.com/NVIDIA/cutlass/releases/download/v4.7.0/cutlass-install-x86_64-cu13-4.7.0.tar.gz";
const ARCHIVE_SHA256: &str = "32f0786c26ede0a5647fa25fc1b4a1429938f181ff28fd57b5632c0b00f0e24f";
const COMPILER_LIBRARY_SHA256: &str =
    "57df017e3585a10443c74c8b4cd99bda854242fb2f4c9534cf56d58c2c741628";
const INSTALL_MANIFEST: &str = ".cuda-oxide-cutlass-toolchain";
const INSTALL_MANIFEST_FORMAT: &str = "2";
const LEGACY_INSTALL_MANIFEST_FORMAT: &str = "1";

static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
struct InstallSpec {
    version: &'static str,
    platform: &'static str,
    archive_name: &'static str,
    archive_url: &'static str,
    archive_sha256: &'static str,
    compiler_library_sha256: &'static str,
}

const OFFICIAL: InstallSpec = InstallSpec {
    version: CUTLASS_VERSION,
    platform: CUTLASS_PLATFORM,
    archive_name: ARCHIVE_NAME,
    archive_url: ARCHIVE_URL,
    archive_sha256: ARCHIVE_SHA256,
    compiler_library_sha256: COMPILER_LIBRARY_SHA256,
};

/// Paths exported to compiler-backed builds after a verified installation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedCutlass {
    pub install_dir: PathBuf,
    pub compiler_library: PathBuf,
    pub compiler_library_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstallOutcome {
    pub paths: ResolvedCutlass,
    /// False when an already-complete pinned install was reused.
    pub installed: bool,
}

/// Install the one CUTLASS compiler release supported by this cargo-oxide.
pub fn install_official() -> Result<InstallOutcome, String> {
    ensure_supported_host()?;
    let managed_root = managed_root()?;
    let install_dir = install_dir(&managed_root, OFFICIAL);

    match resolve_at(&install_dir, OFFICIAL) {
        Ok(Some(paths)) => {
            return Ok(InstallOutcome {
                paths,
                installed: false,
            });
        }
        Ok(None) => {}
        Err(error) => {
            return Err(format!(
                "the managed CUTLASS directory {} exists but is not a valid pinned installation: {error}\nmove or remove that directory, then rerun `cargo oxide toolchain install cutlass`",
                install_dir.display()
            ));
        }
    }

    let version_dir = install_dir.parent().ok_or_else(|| {
        format!(
            "could not determine the parent of managed install {}",
            install_dir.display()
        )
    })?;
    fs::create_dir_all(version_dir).map_err(|error| {
        format!(
            "could not create managed CUTLASS directory {}: {error}",
            version_dir.display()
        )
    })?;

    let _lock =
        InstallLock::acquire(version_dir.join(format!(".{}.install.lock", OFFICIAL.platform)))?;

    // A cooperating installer may have completed while this process waited
    // for directory creation and lock acquisition.
    match resolve_at(&install_dir, OFFICIAL) {
        Ok(Some(paths)) => {
            return Ok(InstallOutcome {
                paths,
                installed: false,
            });
        }
        Ok(None) => {}
        Err(error) => {
            return Err(format!(
                "the managed CUTLASS directory {} became invalid: {error}",
                install_dir.display()
            ));
        }
    }

    let download_dir = TemporaryDirectory::create(version_dir, "download")?;
    let archive = download_dir.path.join(OFFICIAL.archive_name);
    eprintln!(
        "NVIDIA's CUTLASS compiler bundle is proprietary software governed by the NVIDIA EULA:"
    );
    eprintln!("  https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html");
    eprintln!(
        "Downloading NVIDIA CUTLASS compiler {} ({})...",
        OFFICIAL.version, OFFICIAL.platform
    );
    download_archive(OFFICIAL.archive_url, &archive)?;

    eprintln!("Verifying SHA-256 {}...", OFFICIAL.archive_sha256);
    let paths = install_verified_archive(OFFICIAL, &archive, &install_dir)?;
    Ok(InstallOutcome {
        paths,
        installed: true,
    })
}

/// Resolve the pinned managed install without downloading or mutating state.
///
/// Unsupported hosts and machines without a resolvable Cargo home simply have
/// no managed installation.  An existing but malformed install is an error so
/// a requested compiler backend can diagnose it instead of using partial files.
pub fn resolve_official() -> Result<Option<ResolvedCutlass>, String> {
    if !supported_host() {
        return Ok(None);
    }
    let Some(managed_root) = managed_root_optional() else {
        return Ok(None);
    };
    resolve_at(&install_dir(&managed_root, OFFICIAL), OFFICIAL)
}

pub fn print_install_outcome(outcome: &InstallOutcome) {
    if outcome.installed {
        println!(
            "Installed NVIDIA CUTLASS compiler {} ({})",
            CUTLASS_VERSION, CUTLASS_PLATFORM
        );
    } else {
        println!(
            "NVIDIA CUTLASS compiler {} ({}) is already installed",
            CUTLASS_VERSION, CUTLASS_PLATFORM
        );
    }
    println!("  root:     {}", outcome.paths.install_dir.display());
    println!("  compiler: {}", outcome.paths.compiler_library.display());
    println!();
    println!(
        "Build and run commands will provide {CUTLASS_COMPILER_ENV} automatically when it is not explicitly configured."
    );
}

fn supported_host() -> bool {
    std::env::consts::OS == "linux" && std::env::consts::ARCH == "x86_64"
}

fn ensure_supported_host() -> Result<(), String> {
    if supported_host() {
        Ok(())
    } else {
        Err(format!(
            "CUTLASS compiler {} is pinned to Linux x86_64 ({CUTLASS_PLATFORM}); this host is {} {}",
            CUTLASS_VERSION,
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    }
}

fn managed_root() -> Result<PathBuf, String> {
    managed_root_optional().ok_or_else(|| {
        "could not determine the per-user cargo-oxide directory; set CARGO_HOME or HOME".to_string()
    })
}

fn managed_root_optional() -> Option<PathBuf> {
    cargo_home_from(
        std::env::var_os("CARGO_HOME"),
        std::env::var_os("HOME"),
        std::env::current_dir().ok().as_deref(),
    )
    .map(|home| home.join("cuda-oxide/toolchains/cutlass"))
}

fn cargo_home_from(
    cargo_home: Option<OsString>,
    home: Option<OsString>,
    current_dir: Option<&Path>,
) -> Option<PathBuf> {
    let configured = cargo_home
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            home.filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .map(|path| path.join(".cargo"))
        })?;
    if configured.is_absolute() {
        Some(configured)
    } else {
        current_dir.map(|cwd| cwd.join(configured))
    }
}

fn install_dir(managed_root: &Path, spec: InstallSpec) -> PathBuf {
    managed_root.join(spec.version).join(spec.platform)
}

fn download_archive(url: &str, destination: &Path) -> Result<(), String> {
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "2",
            "--retry-connrefused",
            "--silent",
            "--show-error",
            "--user-agent",
            concat!("cargo-oxide/", env!("CARGO_PKG_VERSION")),
            "--output",
        ])
        .arg(destination)
        .arg(url)
        .output()
        .map_err(|error| {
            format!(
                "could not start `curl` to download {url}: {error}\ninstall curl and rerun `cargo oxide toolchain install cutlass`"
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "curl failed while downloading {url} (status {}):\n{}",
            display_status(output.status.code()),
            bounded_diagnostics(&output.stderr)
        ));
    }
    let metadata = fs::metadata(destination).map_err(|error| {
        format!(
            "curl reported success but {} could not be inspected: {error}",
            destination.display()
        )
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "curl reported success but downloaded an empty/non-file archive at {}",
            destination.display()
        ));
    }
    Ok(())
}

fn install_verified_archive(
    spec: InstallSpec,
    archive: &Path,
    final_dir: &Path,
) -> Result<ResolvedCutlass, String> {
    let actual = sha256_file(archive)?;
    if actual != spec.archive_sha256 {
        return Err(format!(
            "CUTLASS archive digest mismatch for {}\n  expected: {}\n  actual:   {}\nrefusing to extract or publish the archive",
            archive.display(),
            spec.archive_sha256,
            actual
        ));
    }

    if final_dir.exists() {
        return Err(format!(
            "refusing to overwrite existing managed CUTLASS directory {}",
            final_dir.display()
        ));
    }
    let parent = final_dir.parent().ok_or_else(|| {
        format!(
            "could not determine parent directory for {}",
            final_dir.display()
        )
    })?;
    let staging = TemporaryDirectory::create(parent, "extract")?;
    let extract_dir = staging.path.join("payload");
    fs::create_dir(&extract_dir).map_err(|error| {
        format!(
            "could not create extraction directory {}: {error}",
            extract_dir.display()
        )
    })?;

    validate_archive_paths(archive)?;
    extract_archive(archive, &extract_dir)?;
    let payload_root = find_payload_root(&extract_dir)?;
    let discovered = discover_payload(&payload_root, spec)?;
    write_install_manifest(&payload_root, spec, &discovered)?;

    fs::rename(&payload_root, final_dir).map_err(|error| {
        format!(
            "could not atomically publish CUTLASS compiler at {}: {error}",
            final_dir.display()
        )
    })?;

    match resolve_at(final_dir, spec) {
        Ok(Some(paths)) => Ok(paths),
        Ok(None) => Err(format!(
            "published CUTLASS directory {} disappeared",
            final_dir.display()
        )),
        Err(error) => {
            let quarantine = staging.path.join("failed-publication");
            let _ = fs::rename(final_dir, &quarantine);
            Err(format!(
                "published CUTLASS directory failed validation and was withdrawn: {error}"
            ))
        }
    }
}

fn validate_archive_paths(archive: &Path) -> Result<(), String> {
    let output = Command::new("tar")
        .args(["--list", "--gzip", "--file"])
        .arg(archive)
        .output()
        .map_err(|error| {
            format!(
                "could not start `tar` to inspect {}: {error}\ninstall GNU tar and rerun the installer",
                archive.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "tar could not list verified archive {} (status {}):\n{}",
            archive.display(),
            display_status(output.status.code()),
            bounded_diagnostics(&output.stderr)
        ));
    }
    let listing = String::from_utf8(output.stdout).map_err(|_| {
        format!(
            "archive {} contains a non-UTF-8 path; refusing extraction",
            archive.display()
        )
    })?;
    for entry in listing.lines().filter(|line| !line.is_empty()) {
        let path = Path::new(entry);
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(format!(
                "archive {} contains unsafe path {entry:?}; refusing extraction",
                archive.display()
            ));
        }
    }
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<(), String> {
    let output = Command::new("tar")
        .args([
            "--extract",
            "--gzip",
            "--file",
        ])
        .arg(archive)
        .args([
            "--directory",
        ])
        .arg(destination)
        .args([
            "--no-same-owner",
            "--no-same-permissions",
            "--delay-directory-restore",
        ])
        .output()
        .map_err(|error| {
            format!(
                "could not start `tar` to extract {}: {error}\ninstall GNU tar and rerun the installer",
                archive.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "tar failed to extract verified archive {} (status {}):\n{}",
            archive.display(),
            display_status(output.status.code()),
            bounded_diagnostics(&output.stderr)
        ));
    }
    Ok(())
}

fn find_payload_root(extract_dir: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    collect_payload_roots(extract_dir, 0, &mut candidates)?;
    match candidates.as_slice() {
        [root] => Ok(root.clone()),
        [] => Err(format!(
            "verified CUTLASS archive did not contain an exact libCutlassCompiler.so under {}",
            extract_dir.display()
        )),
        _ => Err(format!(
            "verified CUTLASS archive contained multiple compiler roots: {}",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn collect_payload_roots(
    directory: &Path,
    depth: usize,
    candidates: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if exact_compiler_library(directory).is_some() {
        candidates.push(directory.to_path_buf());
        return Ok(());
    }
    if depth >= 4 {
        return Ok(());
    }
    let mut children = fs::read_dir(directory)
        .map_err(|error| format!("could not inspect {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("could not inspect {}: {error}", directory.display()))?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let file_type = child
            .file_type()
            .map_err(|error| format!("could not inspect {}: {error}", child.path().display()))?;
        if file_type.is_dir() && !file_type.is_symlink() {
            collect_payload_roots(&child.path(), depth + 1, candidates)?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct PayloadPaths {
    compiler_library: PathBuf,
}

fn discover_payload(root: &Path, spec: InstallSpec) -> Result<PayloadPaths, String> {
    let compiler_library = exact_compiler_library(root).ok_or_else(|| {
        format!(
            "verified CUTLASS archive has no exact libCutlassCompiler.so under {}/lib or {}/lib64",
            root.display(),
            root.display()
        )
    })?;
    validate_regular_nonempty(root, &compiler_library, "libCutlassCompiler shared library")?;
    let actual = sha256_file(&compiler_library)?;
    if actual != spec.compiler_library_sha256 {
        return Err(format!(
            "CUTLASS compiler library digest mismatch for {}\n  expected: {}\n  actual:   {}\nrefusing to publish the installation",
            compiler_library.display(),
            spec.compiler_library_sha256,
            actual
        ));
    }

    Ok(PayloadPaths { compiler_library })
}

fn exact_compiler_library(root: &Path) -> Option<PathBuf> {
    for lib_dir in [root.join("lib"), root.join("lib64")] {
        let exact = lib_dir.join("libCutlassCompiler.so");
        if exact.is_file() {
            return Some(exact);
        }
    }
    None
}

fn validate_regular_nonempty(
    root: &Path,
    path: &Path,
    label: &str,
) -> Result<fs::Metadata, String> {
    let unresolved_metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect {label} {}: {error}", path.display()))?;
    if !unresolved_metadata.file_type().is_file() {
        return Err(format!(
            "{label} {} is not a nonempty regular file",
            path.display()
        ));
    }
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("could not resolve {}: {error}", root.display()))?;
    let canonical_path = fs::canonicalize(path)
        .map_err(|error| format!("could not resolve {label} {}: {error}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(format!(
            "{label} {} resolves outside the managed installation",
            path.display()
        ));
    }
    let metadata = fs::metadata(&canonical_path)
        .map_err(|error| format!("could not inspect {label} {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "{label} {} is not a nonempty regular file",
            path.display()
        ));
    }
    Ok(metadata)
}

fn write_install_manifest(
    root: &Path,
    spec: InstallSpec,
    paths: &PayloadPaths,
) -> Result<(), String> {
    let relative = |path: &Path| {
        path.strip_prefix(root)
            .map(path_to_manifest_value)
            .map_err(|_| format!("{} is not under {}", path.display(), root.display()))
    };
    let manifest = format!(
        "format={INSTALL_MANIFEST_FORMAT}\nversion={}\nplatform={}\narchive-url={}\narchive-sha256={}\ncompiler-library={}\ncompiler-library-sha256={}\n",
        spec.version,
        spec.platform,
        spec.archive_url,
        spec.archive_sha256,
        relative(&paths.compiler_library)?,
        spec.compiler_library_sha256,
    );
    let manifest_path = root.join(INSTALL_MANIFEST);
    let mut file = File::create(&manifest_path).map_err(|error| {
        format!(
            "could not create install manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    file.write_all(manifest.as_bytes()).map_err(|error| {
        format!(
            "could not write install manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    file.sync_all().map_err(|error| {
        format!(
            "could not sync install manifest {}: {error}",
            manifest_path.display()
        )
    })
}

fn path_to_manifest_value(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn resolve_at(root: &Path, spec: InstallSpec) -> Result<Option<ResolvedCutlass>, String> {
    if !root.exists() {
        return Ok(None);
    }
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let manifest_path = root.join(INSTALL_MANIFEST);
    let contents = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "could not read completion manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let fields = parse_manifest(&contents)?;
    let format = manifest_field(&fields, "format")?;
    if format != INSTALL_MANIFEST_FORMAT && format != LEGACY_INSTALL_MANIFEST_FORMAT {
        return Err(format!(
            "managed CUTLASS manifest has format={format:?}, expected {INSTALL_MANIFEST_FORMAT:?}"
        ));
    }
    require_manifest_value(&fields, "version", spec.version)?;
    require_manifest_value(&fields, "platform", spec.platform)?;
    require_manifest_value(&fields, "archive-url", spec.archive_url)?;
    require_manifest_value(&fields, "archive-sha256", spec.archive_sha256)?;
    if format == INSTALL_MANIFEST_FORMAT {
        require_manifest_value(
            &fields,
            "compiler-library-sha256",
            spec.compiler_library_sha256,
        )?;
    }

    let compiler_library = manifest_path_join(root, manifest_field(&fields, "compiler-library")?)?;

    validate_regular_nonempty(root, &compiler_library, "libCutlassCompiler shared library")?;
    let compiler_library_sha256 = sha256_file(&compiler_library)?;
    if compiler_library_sha256 != spec.compiler_library_sha256 {
        return Err(format!(
            "managed CUTLASS compiler library digest mismatch for {}\n  expected: {}\n  actual:   {}",
            compiler_library.display(),
            spec.compiler_library_sha256,
            compiler_library_sha256
        ));
    }

    Ok(Some(ResolvedCutlass {
        install_dir: root.to_path_buf(),
        compiler_library,
        compiler_library_sha256,
    }))
}

fn parse_manifest(contents: &str) -> Result<BTreeMap<String, String>, String> {
    let mut fields = BTreeMap::new();
    for (line_number, line) in contents.lines().enumerate() {
        let (key, value) = line.split_once('=').ok_or_else(|| {
            format!(
                "invalid managed CUTLASS manifest line {}: expected key=value",
                line_number + 1
            )
        })?;
        if key.is_empty() || value.is_empty() {
            return Err(format!(
                "invalid managed CUTLASS manifest line {}: empty key or value",
                line_number + 1
            ));
        }
        if fields.insert(key.to_string(), value.to_string()).is_some() {
            return Err(format!(
                "invalid managed CUTLASS manifest: duplicate field {key:?}"
            ));
        }
    }
    Ok(fields)
}

fn manifest_field<'a>(fields: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("managed CUTLASS manifest is missing {key:?}"))
}

fn require_manifest_value(
    fields: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = manifest_field(fields, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "managed CUTLASS manifest has {key}={actual:?}, expected {expected:?}"
        ))
    }
}

fn manifest_path_join(root: &Path, value: &str) -> Result<PathBuf, String> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "managed CUTLASS manifest contains unsafe relative path {value:?}"
        ));
    }
    Ok(root.join(relative))
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("could not open {} for hashing: {error}", path.display()))?;
    let mut hash = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("could not hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let digest: [u8; 32] = hash.finalize().into();
    let mut encoded = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(encoded)
}

fn display_status(code: Option<i32>) -> String {
    code.map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string())
}

fn bounded_diagnostics(bytes: &[u8]) -> String {
    const LIMIT: usize = 8 * 1024;
    let bytes = if bytes.len() > LIMIT {
        &bytes[bytes.len() - LIMIT..]
    } else {
        bytes
    };
    let diagnostics = String::from_utf8_lossy(bytes);
    let diagnostics = diagnostics.trim();
    if diagnostics.is_empty() {
        "<no diagnostics>".to_string()
    } else {
        diagnostics.to_string()
    }
}

struct InstallLock {
    path: PathBuf,
}

impl InstallLock {
    fn acquire(path: PathBuf) -> Result<Self, String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "could not acquire CUTLASS installer lock {}: {error}\nif no installer is running, remove this stale lock and retry",
                    path.display()
                )
            })?;
        writeln!(file, "pid={}", std::process::id()).map_err(|error| {
            format!("could not write installer lock {}: {error}", path.display())
        })?;
        Ok(Self { path })
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn create(parent: &Path, purpose: &str) -> Result<Self, String> {
        for _ in 0..32 {
            let unique = unique_suffix();
            let path = parent.join(format!(
                ".{CUTLASS_PLATFORM}.{purpose}.{}.{unique}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "could not create temporary installer directory {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        Err(format!(
            "could not allocate a unique installer directory under {}",
            parent.display()
        ))
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn unique_suffix() -> u128 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    time ^ u128::from(UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cargo-oxide-cutlass-{name}-{}-{}",
                std::process::id(),
                unique_suffix()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture_archive(root: &Path) -> PathBuf {
        let source = root.join("source/cutlass-fixture");
        fs::create_dir_all(source.join("lib")).unwrap();
        fs::write(
            source.join("lib/libCutlassCompiler.so"),
            b"fixture library\n",
        )
        .unwrap();
        let archive = root.join("fixture.tar.gz");
        let status = Command::new("tar")
            .args(["--create", "--gzip", "--file"])
            .arg(&archive)
            .args(["--directory"])
            .arg(root.join("source"))
            .arg("cutlass-fixture")
            .status()
            .unwrap();
        assert!(status.success());
        archive
    }

    fn fixture_spec(archive_digest: &'static str, library_digest: &'static str) -> InstallSpec {
        InstallSpec {
            version: "test-version",
            platform: CUTLASS_PLATFORM,
            archive_name: "fixture.tar.gz",
            archive_url: "https://example.invalid/fixture.tar.gz",
            archive_sha256: archive_digest,
            compiler_library_sha256: library_digest,
        }
    }

    #[test]
    fn cargo_home_prefers_explicit_value_and_absolutizes_relative_paths() {
        let cwd = Path::new("/workspace");
        assert_eq!(
            cargo_home_from(
                Some(OsString::from("relative-cargo")),
                Some(OsString::from("/home/user")),
                Some(cwd),
            ),
            Some(PathBuf::from("/workspace/relative-cargo"))
        );
        assert_eq!(
            cargo_home_from(None, Some(OsString::from("/home/user")), Some(cwd)),
            Some(PathBuf::from("/home/user/.cargo"))
        );
    }

    #[test]
    fn verified_fixture_is_published_and_resolved_without_network() {
        let temp = TestDirectory::new("install");
        let archive = fixture_archive(&temp.0);
        let digest = sha256_file(&archive).unwrap();
        let digest: &'static str = Box::leak(digest.into_boxed_str());
        let library_digest = sha256_file(
            &temp
                .0
                .join("source/cutlass-fixture/lib/libCutlassCompiler.so"),
        )
        .unwrap();
        let library_digest: &'static str = Box::leak(library_digest.into_boxed_str());
        let spec = fixture_spec(digest, library_digest);
        let final_dir = install_dir(&temp.0.join("managed"), spec);
        fs::create_dir_all(final_dir.parent().unwrap()).unwrap();

        let installed = install_verified_archive(spec, &archive, &final_dir).unwrap();
        let resolved = resolve_at(&final_dir, spec).unwrap().unwrap();
        assert_eq!(installed, resolved);
        assert_eq!(
            fs::read(&resolved.compiler_library).unwrap(),
            b"fixture library\n"
        );
        assert!(
            resolved
                .compiler_library
                .ends_with("lib/libCutlassCompiler.so")
        );
        assert!(final_dir.join(INSTALL_MANIFEST).is_file());
    }

    #[test]
    fn resolver_rejects_library_replacement_at_the_same_managed_path() {
        let temp = TestDirectory::new("replaced-library");
        let archive = fixture_archive(&temp.0);
        let archive_digest = sha256_file(&archive).unwrap();
        let archive_digest: &'static str = Box::leak(archive_digest.into_boxed_str());
        let library_digest = sha256_file(
            &temp
                .0
                .join("source/cutlass-fixture/lib/libCutlassCompiler.so"),
        )
        .unwrap();
        let library_digest: &'static str = Box::leak(library_digest.into_boxed_str());
        let spec = fixture_spec(archive_digest, library_digest);
        let final_dir = install_dir(&temp.0.join("managed"), spec);
        fs::create_dir_all(final_dir.parent().unwrap()).unwrap();
        let installed = install_verified_archive(spec, &archive, &final_dir).unwrap();

        fs::write(&installed.compiler_library, b"replaced library\n").unwrap();
        let error = resolve_at(&final_dir, spec).unwrap_err();
        assert!(error.contains("library digest mismatch"), "{error}");
        assert!(error.contains(library_digest), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn resolver_rejects_a_symlinked_compiler_library() {
        use std::os::unix::fs::symlink;

        let temp = TestDirectory::new("symlinked-library");
        let archive = fixture_archive(&temp.0);
        let archive_digest = sha256_file(&archive).unwrap();
        let archive_digest: &'static str = Box::leak(archive_digest.into_boxed_str());
        let library_digest = sha256_file(
            &temp
                .0
                .join("source/cutlass-fixture/lib/libCutlassCompiler.so"),
        )
        .unwrap();
        let library_digest: &'static str = Box::leak(library_digest.into_boxed_str());
        let spec = fixture_spec(archive_digest, library_digest);
        let final_dir = install_dir(&temp.0.join("managed"), spec);
        fs::create_dir_all(final_dir.parent().unwrap()).unwrap();
        let installed = install_verified_archive(spec, &archive, &final_dir).unwrap();
        let real_library = installed.compiler_library.with_extension("so.real");

        fs::rename(&installed.compiler_library, &real_library).unwrap();
        symlink(&real_library, &installed.compiler_library).unwrap();

        let error = resolve_at(&final_dir, spec).unwrap_err();
        assert!(error.contains("is not a nonempty regular file"), "{error}");
    }

    #[test]
    fn digest_mismatch_never_extracts_or_publishes() {
        let temp = TestDirectory::new("bad-digest");
        let archive = fixture_archive(&temp.0);
        let spec = fixture_spec(
            "0000000000000000000000000000000000000000000000000000000000000000",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        let final_dir = install_dir(&temp.0.join("managed"), spec);
        fs::create_dir_all(final_dir.parent().unwrap()).unwrap();

        let error = install_verified_archive(spec, &archive, &final_dir).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");
        assert!(error.contains("refusing to extract or publish"), "{error}");
        assert!(!final_dir.exists());
    }

    #[test]
    fn resolver_rejects_manifest_path_escape() {
        let temp = TestDirectory::new("escape");
        let root = temp.0.join("install");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join(INSTALL_MANIFEST),
            format!(
                "format=2\nversion={}\nplatform={}\narchive-url={}\narchive-sha256={}\ncompiler-library=../outside\ncompiler-library-sha256={}\n",
                OFFICIAL.version,
                OFFICIAL.platform,
                OFFICIAL.archive_url,
                OFFICIAL.archive_sha256,
                OFFICIAL.compiler_library_sha256,
            ),
        )
        .unwrap();

        let error = resolve_at(&root, OFFICIAL).unwrap_err();
        assert!(error.contains("unsafe relative path"), "{error}");
    }

    #[test]
    fn archive_path_check_rejects_parent_components() {
        let error = manifest_path_join(Path::new("/managed"), "../escape").unwrap_err();
        assert!(error.contains("unsafe relative path"), "{error}");
    }
}
