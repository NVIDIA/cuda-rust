/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use crate::error::JITError;
use cuda_core::{get_device_sm_name, Device};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion};
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

/// Environment variable used to override the `tileiras` executable.
///
/// Set this to an absolute path such as `/opt/cuda-tile/bin/tileiras` to use
/// that binary instead of the `tileiras` found on `PATH`.
pub const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";
pub const SETUP_DIAGNOSTICS_ENV: &str = "CUTILE_SETUP_DIAGNOSTICS";

const CUDA_TOOLKIT_PATH_ENV: &str = "CUDA_TOOLKIT_PATH";
/// Honored after `CUDA_TOOLKIT_PATH`, like the build scripts do: the
/// conventional name used by nvcc wrappers and CI images.
const CUDA_HOME_ENV: &str = "CUDA_HOME";
/// The toolkit environment variables naming a CUDA install root, in
/// precedence order. Mirrors `TOOLKIT_ENV_VARS` in the workspace build
/// scripts.
const TOOLKIT_ENV_VARS: [&str; 2] = [CUDA_TOOLKIT_PATH_ENV, CUDA_HOME_ENV];
const MIN_CUDA_VERSION: u32 = 13020;

/// Environment variable to force the emitted Tile IR bytecode version
/// (e.g. `13.2`). Must be supported by both the writer and selected assembler.
pub const BYTECODE_VERSION_ENV: &str = "CUTILE_BYTECODE_VERSION";

/// One immutable JIT snapshot. Discovery is process-wide and cached, but the
/// target architecture belongs to the device selected for this launch.
#[derive(Debug, Clone)]
pub struct ToolkitCapabilities {
    pub tileiras: PathBuf,
    pub tileiras_version: &'static str,
    pub target: cutile_ir::capabilities::TargetCapabilities,
    /// Driver API version, for diagnostics and host API availability. This is
    /// not a blanket veto on newer toolkits under CUDA minor compatibility.
    pub driver_version: i32,
}

impl ToolkitCapabilities {
    pub fn for_device(device_id: usize) -> Result<Arc<Self>, JITError> {
        use cuda_core::IntoResult;
        type Key = (String, Option<OsString>, String);
        static CACHE: OnceLock<Mutex<HashMap<Key, Arc<ToolkitCapabilities>>>> = OnceLock::new();
        let tileiras = tileiras_binary();
        let device = Device::raw_device(device_id)
            .map_err(|e| JITError::Generic(format!("cannot query CUDA device {device_id}: {e}")))?;
        // SAFETY: the driver returned this device handle.
        let architecture = unsafe { get_device_sm_name(device) }
            .map_err(|e| JITError::Generic(format!("cannot query target architecture: {e}")))?;
        let key = (
            stat_fingerprint(&tileiras),
            env::var_os(BYTECODE_VERSION_ENV),
            architecture.clone(),
        );
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(caps) = cache.lock().unwrap().get(&key) {
            return Ok(Arc::clone(caps));
        }
        let bytecode_version = negotiate_bytecode_version(&tileiras, key.1.as_deref())?;
        let mut driver_version = 0;
        // SAFETY: this driver query writes one initialized local integer.
        unsafe { cuda_core::sys::cuDriverGetVersion(&mut driver_version) }
            .result()
            .map_err(|e| JITError::Generic(format!("cannot query CUDA driver version: {e}")))?;
        let caps = Arc::new(Self {
            tileiras_version: fingerprint_of(&tileiras),
            tileiras,
            target: cutile_ir::capabilities::TargetCapabilities::new(
                bytecode_version,
                architecture,
            ),
            driver_version,
        });
        cache.lock().unwrap().insert(key, Arc::clone(&caps));
        Ok(caps)
    }
}

/// Returns the cutile compiler version (from the workspace Cargo.toml).
pub fn get_compiler_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// The `CUTILE_DISABLE_CHECK_HOISTING` / `CUTILE_FORCE_DEVICE_CHECKS`
// ablation switches are resolved once per compile into a
// [`crate::check_optimizations::CheckOptimizations`] (see `from_env` there);
// the compiler consults that policy, never the environment.

/// `CUTILE_JIT_LOG` (`1`/`true`/`yes`/`on`, like every other on/off switch
/// of the crate and of `cutile`) also reports every bounds check that stays
/// inside a loop body with the reason it could not hoist.
pub fn jit_hoist_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("CUTILE_JIT_LOG"))
}

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
///
/// Cached per device: the driver is queried once per device and cache hits are
/// lock-free (`OnceLock::get` is an atomic load). CUDA device ordinals are small
/// and contiguous, so a fixed array of `OnceLock` suffices; an ordinal beyond it
/// (never in practice) skips the cache and queries the driver each time.
pub fn get_gpu_name(device_id: usize) -> String {
    const MAX_CACHED_DEVICES: usize = 64;
    static NAMES: [OnceLock<String>; MAX_CACHED_DEVICES] =
        [const { OnceLock::new() }; MAX_CACHED_DEVICES];

    let query = || -> String {
        let dev = Device::raw_device(device_id).unwrap_or_else(|e| {
            panic!(
                "failed to get CUDA device {device_id}: {e}\n\
                 Ensure an NVIDIA GPU is visible to the process and the CUDA driver is installed."
            )
        });
        unsafe { get_device_sm_name(dev) }.unwrap_or_else(|e| {
            panic!(
                "failed to query CUDA SM name for device {device_id}: {e}\n\
                 Ensure the installed CUDA driver supports this GPU."
            )
        })
    };

    match NAMES.get(device_id) {
        Some(slot) => slot.get_or_init(query).clone(),
        None => query(),
    }
}

fn tileiras_executable_name() -> &'static str {
    if cfg!(windows) {
        "tileiras.exe"
    } else {
        "tileiras"
    }
}

/// A toolkit root named by the environment: `(variable, value)`, so the
/// diagnostics can say which variable was honored.
type ToolkitEnv = (&'static str, OsString);

fn cuda_toolkit_tileiras(cuda_toolkit_path: Option<ToolkitEnv>) -> Option<PathBuf> {
    let (var, tileiras) = cuda_toolkit_path
        .filter(|(_, value)| !value.as_os_str().is_empty())
        .map(|(var, value)| {
            (
                var,
                PathBuf::from(value)
                    .join("bin")
                    .join(tileiras_executable_name()),
            )
        })?;
    if tileiras.is_file() {
        emit_setup_diagnostic(format_args!(
            "using {var} tileiras at {}",
            tileiras.display()
        ));
        Some(tileiras)
    } else {
        emit_setup_diagnostic(format_args!(
            "{var} did not contain tileiras at {}",
            tileiras.display()
        ));
        None
    }
}

fn resolve_tileiras_binary(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<ToolkitEnv>,
) -> (PathBuf, Option<PathBuf>) {
    resolve_tileiras_with_toolkit_candidates(
        tileiras_override,
        cuda_toolkit_path,
        default_cuda_toolkit_candidates(),
    )
}

/// Resolves the `tileiras` binary and, when it was found via a CUDA toolkit
/// (not a `CUTILE_TILEIRAS_PATH` override or bare `PATH`), the toolkit root used
/// to locate `cuda.h` for bytecode-version selection.
fn resolve_tileiras_with_toolkit_candidates(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<ToolkitEnv>,
    default_cuda_toolkit_candidates: &[PathBuf],
) -> (PathBuf, Option<PathBuf>) {
    if let Some(path) = tileiras_override.filter(|value| !value.as_os_str().is_empty()) {
        let path = PathBuf::from(path);
        emit_setup_diagnostic(format_args!("using {TILEIRAS_PATH_ENV}={}", path.display()));
        // An overridden binary may be newer than the installed CTK, so its
        // version is decided by probing rather than the toolkit's cuda.h.
        return (path, None);
    }

    if let Some(path) = cuda_toolkit_tileiras(cuda_toolkit_path) {
        if path.is_file() {
            let toolkit = toolkit_root_of(&path);
            return (path, toolkit);
        }
    }

    if let Some(path) = default_cuda_toolkit_tileiras(default_cuda_toolkit_candidates) {
        let toolkit = toolkit_root_of(&path);
        return (path, toolkit);
    }

    emit_setup_diagnostic(format_args!(
        "falling back to {} through PATH lookup",
        tileiras_executable_name()
    ));
    (PathBuf::from(tileiras_executable_name()), None)
}

/// CUDA toolkit root for a `<root>/bin/tileiras` path (strips `bin/tileiras`).
fn toolkit_root_of(tileiras: &Path) -> Option<PathBuf> {
    tileiras.parent()?.parent().map(PathBuf::from)
}

/// Test-only helper that returns just the resolved `tileiras` path.
#[cfg(test)]
fn resolve_tileiras_binary_with_candidates(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<ToolkitEnv>,
    default_cuda_toolkit_candidates: &[PathBuf],
) -> PathBuf {
    resolve_tileiras_with_toolkit_candidates(
        tileiras_override,
        cuda_toolkit_path,
        default_cuda_toolkit_candidates,
    )
    .0
}

/// Returns the `tileiras` executable path used by the JIT.
///
/// Resolution order:
///
/// 1. [`TILEIRAS_PATH_ENV`] when set.
/// 2. `$CUDA_TOOLKIT_PATH/bin/tileiras`, then `$CUDA_HOME/bin/tileiras`,
///    when the variable is set and the binary exists there (the same two
///    variables, in the same order, that the workspace build scripts honor).
/// 3. Default CUDA install locations with CUDA 13.2+ and `bin/tileiras`.
/// 4. `tileiras` through normal `PATH` lookup.
pub fn tileiras_binary() -> PathBuf {
    tileiras_and_toolkit().0
}

/// Identifies which `tileiras` binary compiled a cubin.
///
/// This belongs in every cache key that names a cubin: without it, upgrading the
/// toolkit leaves the key unchanged and a cubin built by the previous `tileiras`
/// is served as a hit.
///
/// The fingerprint is the `--version` stdout. It carries the build number
/// (`Build local.local.37905922_`), and unlike `(size, mtime)` it survives a
/// reinstall of the same toolkit, so the cache stays warm. Measured cost on
/// CUDA 13.3: under 5 ms, `maxrss` 21.4 MB. The path resolution and `--version`
/// are both cached per process (the former by env value, the latter by path),
/// and the key path runs on cache hits too, so this must stay cheap.
///
/// Note it does not distinguish two binaries that report the same version, such
/// as a locally patched one.
///
/// Falls back to `(canonical path, size, mtime)` when `--version` fails, which
/// covers a future `tileiras` that drops the flag. An empty fingerprint is never
/// returned: that would drop the compiler out of the key.
pub fn tileiras_fingerprint() -> &'static str {
    fingerprint_of(&tileiras_binary())
}

/// Fingerprint of a specific resolved `tileiras`, cached **per path** rather than
/// once per process. A process that switches `CUTILE_TILEIRAS_PATH` mid-run then
/// keys entries by the binary actually in effect, not the one seen at the first
/// call — otherwise cubins built by the new binary are stored under the old
/// binary's fingerprint and served to a process that genuinely uses the old one.
/// Mirrors [`cached_bytecode_version`]. The `--version` spawn happens once per
/// distinct binary; the interned string lives for the process (bounded: one per
/// tileiras path, normally one).
fn fingerprint_of(tileiras: &Path) -> &'static str {
    // Fast path: the fingerprint this path resolved to within the last
    // revalidation window. A launch-site miss calls this on every launch, and
    // the `stat` fingerprint below costs `canonicalize` + `metadata` syscalls
    // (about 1.5 us); a binary swapped in place is still noticed within
    // [`REVALIDATE_EVERY`], which is the granularity the mid-process
    // switch semantics need.
    static FAST: OnceLock<Mutex<HashMap<PathBuf, (std::time::Instant, &'static str)>>> =
        OnceLock::new();
    let fast = FAST.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&(checked, fp)) = fast.lock().unwrap().get(tileiras) {
        if checked.elapsed() < REVALIDATE_EVERY {
            return fp;
        }
    }
    static CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let key = stat_fingerprint(tileiras);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Copy out of the guard: the lock must not be held while the
    // `--version` spawn runs or while the insert below re-locks.
    let cached = cache.lock().unwrap().get(&key).copied();
    let fp = match cached {
        Some(fp) => fp,
        None => {
            let fp: &'static str =
                Box::leak(compute_tileiras_fingerprint(tileiras).into_boxed_str());
            cache.lock().unwrap().insert(key, fp);
            fp
        }
    };
    fast.lock()
        .unwrap()
        .insert(tileiras.to_path_buf(), (std::time::Instant::now(), fp));
    fp
}

/// How long a cached toolchain fact (resolved binary, its fingerprint, the
/// bytecode override) is trusted before the environment and filesystem are
/// consulted again. Mid-process switches take effect within this window;
/// steady-state launches pay none of the syscalls or `PATH` reads.
const REVALIDATE_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

/// `CUTILE_BYTECODE_VERSION` as seen within the last revalidation window.
/// Part of every kernel key, so it must be cheap on the launch-site miss path.
pub fn bytecode_override() -> Option<OsString> {
    static CACHED: Mutex<Option<(std::time::Instant, Option<OsString>)>> = Mutex::new(None);
    let mut cached = CACHED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((checked, value)) = cached.as_ref() {
        if checked.elapsed() < REVALIDATE_EVERY {
            return value.clone();
        }
    }
    let value = env::var_os(BYTECODE_VERSION_ENV);
    *cached = Some((std::time::Instant::now(), value.clone()));
    value
}

fn compute_tileiras_fingerprint(tileiras: &Path) -> String {
    if let Ok(output) = Command::new(tileiras).arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !version.is_empty() {
                return version;
            }
        }
    }
    emit_setup_diagnostic(format_args!(
        "{} --version failed; fingerprinting it by path, size and mtime instead",
        tileiras.display()
    ));
    stat_fingerprint(tileiras)
}

/// `(canonical path, size, mtime)`, the fallback when `--version` is unavailable.
///
/// Weaker than the version string in one direction: reinstalling the same
/// toolkit changes `mtime`, so every key changes and the disk cache misses
/// across the board. That costs one recompile per kernel, not correctness.
fn stat_fingerprint(tileiras: &Path) -> String {
    let path = std::fs::canonicalize(tileiras).unwrap_or_else(|_| tileiras.to_path_buf());
    let (len, mtime_ns) = std::fs::metadata(&path)
        .map(|meta| {
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            (meta.len(), mtime_ns)
        })
        .unwrap_or((0, 0));
    format!("stat\0{}\0{len}\0{mtime_ns}", path.display())
}

/// The first set, non-empty toolkit variable (`CUDA_TOOLKIT_PATH`, then
/// `CUDA_HOME`) with its value.
/// The environment values that drive toolchain resolution, as one comparable
/// snapshot. Launch-site caches compare this per launch instead of re-running
/// the full [`tileiras_fingerprint`] chain, preserving the documented
/// mid-process `CUTILE_TILEIRAS_PATH` / toolkit switch semantics at the cost
/// of the env reads alone.
pub type ToolchainEnvSnapshot = (Option<OsString>, Option<(&'static str, OsString)>);

/// See [`ToolchainEnvSnapshot`].
pub fn toolchain_env_snapshot() -> ToolchainEnvSnapshot {
    let tileiras_env = env::var_os(TILEIRAS_PATH_ENV).filter(|v| !v.as_os_str().is_empty());
    (tileiras_env, toolkit_env())
}

fn toolkit_env() -> Option<ToolkitEnv> {
    TOOLKIT_ENV_VARS.iter().find_map(|&var| {
        env::var_os(var)
            .filter(|v| !v.as_os_str().is_empty())
            .map(|v| (var, v))
    })
}

/// Resolves `tileiras` together with the CUDA toolkit root (when applicable),
/// using the active `CUTILE_TILEIRAS_PATH` / `CUDA_TOOLKIT_PATH` / `CUDA_HOME`
/// environment.
///
/// Cached by the environment values that drive resolution: steady-state launches
/// only re-read the env vars, and the expensive toolkit/`cuda.h` lookup is
/// recomputed only when one of those values changes. This mirrors
/// [`cached_bytecode_version`] and [`fingerprint_of`].
fn tileiras_and_toolkit() -> (PathBuf, Option<PathBuf>) {
    let tileiras_env = env::var_os(TILEIRAS_PATH_ENV).filter(|v| !v.as_os_str().is_empty());
    let toolkit_env = toolkit_env();
    // Fast path: the resolution for these env values within the last
    // revalidation window. The full cache below is additionally keyed by
    // `PATH`, whose read and hash are too expensive for the launch-site miss
    // path; a `PATH` change is picked up at the next revalidation.
    type FastKey = (Option<OsString>, Option<ToolkitEnv>);
    static FAST: OnceLock<Mutex<HashMap<FastKey, (std::time::Instant, TileirasResolution)>>> =
        OnceLock::new();
    let fast = FAST.get_or_init(|| Mutex::new(HashMap::new()));
    let fast_key: FastKey = (tileiras_env, toolkit_env);
    if let Some((checked, result)) = fast.lock().unwrap().get(&fast_key) {
        if checked.elapsed() < REVALIDATE_EVERY {
            return result.clone();
        }
    }
    let result = cached_tileiras_and_toolkit(fast_key.0.clone(), fast_key.1.clone());
    fast.lock()
        .unwrap()
        .insert(fast_key, (std::time::Instant::now(), result.clone()));
    result
}

/// The resolved `tileiras` binary and, when found through a toolkit, that
/// toolkit's root.
type TileirasResolution = (PathBuf, Option<PathBuf>);

fn cached_tileiras_and_toolkit(
    tileiras_env: Option<OsString>,
    toolkit_env: Option<ToolkitEnv>,
) -> TileirasResolution {
    type Key = (Option<OsString>, Option<ToolkitEnv>, Option<OsString>);
    static CACHE: OnceLock<Mutex<HashMap<Key, TileirasResolution>>> = OnceLock::new();
    let key: Key = (tileiras_env, toolkit_env, env::var_os("PATH"));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(result) = cache.lock().unwrap().get(&key) {
        return result.clone();
    }
    let (mut binary, toolkit) = resolve_tileiras_binary(key.0.clone(), key.1.clone());
    if binary.components().count() == 1 {
        if let Some(path) = &key.2 {
            if let Some(found) = env::split_paths(path)
                .map(|dir| dir.join(&binary))
                .find(|p| p.is_file())
            {
                binary = found;
            }
        }
    }
    let binary = std::fs::canonicalize(&binary).unwrap_or(binary);
    let result = (binary, toolkit);
    cache.lock().unwrap().insert(key, result.clone());
    result
}

// =========================================================================
// Bytecode version selection
//
// The writer and decoder are already version-aware; this decides which
// version to emit so a newer toolchain default (13.3) is not handed to an
// older `tileiras`.
// =========================================================================

/// Select the bytecode accepted by the executable we will actually run.
/// Headers may belong to a different toolkit, especially with a path override.
pub fn selected_bytecode_version() -> Result<BytecodeVersion, JITError> {
    let (tileiras, toolkit) = tileiras_and_toolkit();
    cached_bytecode_version(&tileiras, toolkit.as_deref())
}

fn cached_bytecode_version(
    tileiras: &Path,
    toolkit_dir: Option<&Path>,
) -> Result<BytecodeVersion, JITError> {
    type Key = (String, Option<OsString>);
    static CACHE: OnceLock<Mutex<HashMap<Key, BytecodeVersion>>> = OnceLock::new();
    let key = (
        stat_fingerprint(tileiras),
        env::var_os(BYTECODE_VERSION_ENV),
    );
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&version) = cache.lock().unwrap().get(&key) {
        return Ok(version);
    }
    let version = compute_bytecode_version(tileiras, toolkit_dir)?;
    cache.lock().unwrap().insert(key, version);
    Ok(version)
}

fn compute_bytecode_version(
    tileiras: &Path,
    _toolkit_dir: Option<&Path>,
) -> Result<BytecodeVersion, JITError> {
    let requested = env::var_os(BYTECODE_VERSION_ENV).filter(|v| !v.is_empty());
    negotiate_bytecode_version(tileiras, requested.as_deref())
}

fn negotiate_bytecode_version(
    tileiras: &Path,
    requested: Option<&std::ffi::OsStr>,
) -> Result<BytecodeVersion, JITError> {
    cutile_ir::toolchain::negotiate_bytecode_version(tileiras, requested)
        .map_err(|error| JITError::Generic(error.to_string()))
}

/// `--opt-level` passed to `tileiras`. Not configurable yet.
/// Numeric, not a string: the disk-cache key and entry header store it as one
/// byte.
pub const DEFAULT_OPT_LEVEL: u8 = 3;

/// A path removed when dropped, so the error paths clean up too.
struct ScopedTempFile(Option<PathBuf>);

impl ScopedTempFile {
    fn new(path: PathBuf) -> Self {
        Self(Some(path))
    }

    fn path(&self) -> &Path {
        self.0.as_deref().expect("path is taken only by `keep`")
    }

    /// Leaves the file on disk. Used for the `.bc` a failing `tileiras` run was
    /// given, which the error message points at.
    fn keep(mut self) {
        self.0 = None;
    }
}

impl Drop for ScopedTempFile {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Serializes a `cutile_ir::Module` to Tile IR bytecode (the `.bc` image).
///
/// Runs the module verifiers first, so bytecode returned here is what `tileiras`
/// is expected to accept. Together with the target and opt level, these bytes are
/// the complete input to [`run_tileiras`].
///
/// Also returns the [`BytecodeVersion`] actually written into the image, so the
/// disk-cache key names that exact version instead of re-resolving it (which
/// could drift from the bytes if the toolchain env changed in between).
pub fn serialize_tile_ir_bytecode(
    module: &cutile_ir::Module,
) -> Result<(Vec<u8>, BytecodeVersion), JITError> {
    serialize_tile_ir_bytecode_for_version(module, selected_bytecode_version()?)
}

/// Serialize with the version already negotiated for this JIT snapshot.
pub fn serialize_tile_ir_bytecode_for_version(
    module: &cutile_ir::Module,
    bytecode_version: BytecodeVersion,
) -> Result<(Vec<u8>, BytecodeVersion), JITError> {
    module
        .verify_dominance()
        .map_err(|e| JITError::Generic(format!("tile-ir dominance verification failed: {e}")))?;

    module.verify_bytecode_indices().map_err(|e| {
        JITError::Generic(format!(
            "tile-ir bytecode value-index verification failed: {e}"
        ))
    })?;

    // Dump IR via unified CUTILE_DUMP mechanism (also honors legacy TILE_IR_DUMP).
    // `to_mlir_text` renders the whole module, so it stays behind `should_dump`.
    if crate::dump::should_dump(crate::dump::DumpStage::Ir) {
        crate::dump::dump_module(
            crate::dump::DumpStage::Ir,
            &module.name,
            &module.to_mlir_text(),
        );
    }

    let bytes = write_bytecode_version(module, bytecode_version).map_err(|e| {
        JITError::Generic(format!(
            "Failed to serialize bytecode for module {}: {e}",
            module.name
        ))
    })?;

    if crate::dump::should_dump(crate::dump::DumpStage::Bytecode) {
        let decoded = cutile_ir::decode_bytecode(&bytes)
            .unwrap_or_else(|e| format!("<bytecode decode failed: {e}>"));
        crate::dump::dump_module(crate::dump::DumpStage::Bytecode, &module.name, &decoded);
    }

    Ok((bytes, bytecode_version))
}

/// Derives the L2 cache key for bytecode using the currently resolved
/// `tileiras` toolchain.
///
/// The returned fingerprint is the exact value used in the key, so callers that
/// also validate or encode a cache entry cannot accidentally re-resolve a
/// different toolchain between key derivation and entry construction.
pub(crate) fn current_l2_key_for_bytecode(
    bytecode: &[u8],
    bytecode_version: BytecodeVersion,
    gpu_name: &str,
    opts: &TileirasOptions,
) -> (String, &'static str) {
    let tileiras_fp = tileiras_fingerprint();
    let key = crate::jit_cache::l2_key(bytecode, bytecode_version, gpu_name, opts, tileiras_fp);
    (key, tileiras_fp)
}

/// Runs the canonical JIT bytecode serializer and returns the L2 cache key that
/// the current toolchain would use for `module` and `gpu_name`.
///
/// This runs the compiler-side verifiers and serialization, but it does not
/// consult a [`crate::jit_cache::JitStore`] or compile a cubin with `tileiras`.
pub(crate) fn current_l2_key_for_module(
    module: &cutile_ir::Module,
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<String, JITError> {
    let (bytecode, bytecode_version) = serialize_tile_ir_bytecode(module)?;
    Ok(current_l2_key_for_bytecode(&bytecode, bytecode_version, gpu_name, opts).0)
}

/// Flags forwarded to the `tileiras` invocation.
///
/// These are the complete stage-2 inputs besides the bytecode, the target
/// GPU, and the binary itself — so they participate in the L2 cache key and
/// are validated in disk-cache entries. Two compiles that differ in any
/// field can never share a cubin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileirasOptions {
    /// `--opt-level <N>`.
    pub opt_level: u8,
    /// `--device-debug`: generate debug information.
    pub device_debug: bool,
    /// `--lineinfo`: generate line-number information only.
    pub lineinfo: bool,
    /// `--sanitize=memcheck`: instrument memory accesses for the sanitizer.
    pub sanitize_memcheck: bool,
}

impl Default for TileirasOptions {
    fn default() -> Self {
        Self {
            opt_level: DEFAULT_OPT_LEVEL,
            device_debug: false,
            lineinfo: false,
            sanitize_memcheck: false,
        }
    }
}

impl TileirasOptions {
    /// Resolves the launch-facing [`crate::hints::CompileOptions`] into the
    /// stage-2 flags. `device_debug` implies `--opt-level 0` unless an
    /// explicit level was requested.
    pub fn from_compile_options(options: &crate::hints::CompileOptions) -> Self {
        let opt_level = options.opt_level.unwrap_or(if options.device_debug {
            0
        } else {
            DEFAULT_OPT_LEVEL
        });
        Self {
            opt_level,
            device_debug: options.device_debug,
            lineinfo: options.lineinfo,
            sanitize_memcheck: options.sanitize_memcheck,
        }
    }

    /// The boolean flags packed into one byte, for the cache-entry header
    /// and the L2 key material.
    pub fn flags_byte(&self) -> u8 {
        (self.device_debug as u8)
            | ((self.lineinfo as u8) << 1)
            | ((self.sanitize_memcheck as u8) << 2)
    }
}

/// Compiles Tile IR bytecode to a cubin image by spawning `tileiras`.
///
/// `bytecode`, `gpu_name` and `opts`, plus the `tileiras` binary itself, are
/// the complete input to this stage.
///
/// The temporary `.bc` and `.cubin` are removed before returning. The one
/// exception is a `tileiras` run that fails or cannot be launched, which
/// leaves the `.bc` on disk because the error message names it.
pub fn run_tileiras(
    bytecode: &[u8],
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<Vec<u8>, JITError> {
    run_tileiras_at(&tileiras_binary(), bytecode, gpu_name, opts)
}

fn run_tileiras_at(
    tileiras: &Path,
    bytecode: &[u8],
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<Vec<u8>, JITError> {
    let base_filename = env::temp_dir().join(Uuid::new_v4().to_string());
    let bc_file = ScopedTempFile::new(base_filename.with_extension("bc"));
    let cubin_file = ScopedTempFile::new(base_filename.with_extension("cubin"));
    let bc_filename = bc_file.path().to_string_lossy().into_owned();
    let cubin_filename = cubin_file.path().to_string_lossy().into_owned();

    std::fs::write(bc_file.path(), bytecode).map_err(|e| {
        JITError::Generic(format!("Failed to write bytecode for {bc_filename}: {e}"))
    })?;

    let opt_level_arg = opts.opt_level.to_string();
    let mut args = vec!["--gpu-name", gpu_name, "--opt-level", &opt_level_arg];
    if opts.device_debug {
        args.push("--device-debug");
    }
    if opts.lineinfo {
        args.push("--lineinfo");
    }
    if opts.sanitize_memcheck {
        args.push("--sanitize=memcheck");
    }
    args.extend(["-o", &cubin_filename, &bc_filename]);
    let output = match Command::new(tileiras).args(&args).output() {
        Ok(output) => output,
        Err(e) => {
            // The message names the bytecode, so it has to outlive this call.
            let message = tileiras_launch_error(tileiras, &args, &bc_filename, e);
            bc_file.keep();
            return Err(JITError::Generic(message));
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        // The message points at the bytecode, so it has to outlive this call.
        bc_file.keep();
        return Err(JITError::Generic(format!(
            "{} failed while compiling Tile IR bytecode.\n\
             status: {}\n\
             command: {}\n\
             target gpu: {gpu_name}\n\
             bytecode: {bc_filename} (kept for inspection)\n\
             output cubin: {cubin_filename}\n\
             stdout:\n{stdout}\n\
             stderr:\n{stderr}\n\
             hint: run with CUTILE_DUMP=ir,bytecode to include the generated Tile IR and decoded bytecode in stderr.",
            tileiras.display(),
            output.status,
            display_command(tileiras, &args),
        )));
    }

    let cubin = std::fs::read(cubin_file.path()).map_err(|e| {
        JITError::Generic(format!(
            "{} reported success but its output cubin at {cubin_filename} could not be read: {e}",
            tileiras.display(),
        ))
    })?;
    if cubin.is_empty() {
        return Err(JITError::Generic(format!(
            "{} reported success but wrote an empty cubin at {cubin_filename}",
            tileiras.display(),
        )));
    }
    crate::jit_cache::record_backend_compile();
    Ok(cubin)
}

/// Where a stage-2 cubin came from.
///
/// `DiskCache` carries the store it was served from and the key it validated
/// against, so a `cuModuleLoadData` rejection can evict *that exact entry* from
/// *that exact store* and recompile — see [`recompile_after_disk_rejection`] —
/// without re-deriving the key (which would drift if `opt_level` ever became
/// configurable) or re-reading a possibly-swapped global store slot.
#[derive(Clone)]
pub enum Stage2Source {
    Tileiras,
    DiskCache {
        store: Arc<dyn crate::jit_cache::JitStore>,
        key: String,
    },
}

/// Compiles Tile IR bytecode to a cubin, consulting the disk cache when one is
/// installed (see [`crate::jit_cache::enable`]).
///
/// The lookup sits exactly between bytecode serialization and the `tileiras`
/// spawn: `bytecode` plus `gpu_name`, `opts` and the resolved `tileiras`
/// are the subprocess's complete input, so the content-addressed key derived
/// from them (see [`crate::jit_cache::l2_key`]) is correct by construction.
///
/// Store I/O failures are soft: counted in `stats().io_errors`, logged, and
/// the compile proceeds as if no cache were installed.
pub fn compile_bytecode_cached(
    bytecode: &[u8],
    bc_version: BytecodeVersion,
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<(Vec<u8>, Stage2Source), JITError> {
    compile_bytecode_cached_at(&tileiras_binary(), bytecode, bc_version, gpu_name, opts)
}

/// Use the exact executable/version/target snapshot that lowered this kernel.
pub fn compile_bytecode_cached_with_toolkit(
    bytecode: &[u8],
    opts: &TileirasOptions,
    capabilities: &ToolkitCapabilities,
) -> Result<(Vec<u8>, Stage2Source), JITError> {
    compile_bytecode_cached_at(
        &capabilities.tileiras,
        bytecode,
        capabilities.target.bytecode_version,
        &capabilities.target.architecture,
        opts,
    )
}

fn compile_bytecode_cached_at(
    tileiras: &Path,
    bytecode: &[u8],
    bc_version: BytecodeVersion,
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<(Vec<u8>, Stage2Source), JITError> {
    use crate::jit_cache::{self, EntryParams};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::Ordering;

    let Some(store) = jit_cache::installed_store() else {
        return run_tileiras_at(tileiras, bytecode, gpu_name, opts)
            .map(|c| (c, Stage2Source::Tileiras));
    };

    // `bc_version` is the version the caller actually serialized into `bytecode`,
    // not a fresh re-resolution — so the key's version field can never disagree
    // with the bytes it sits next to (see #7).
    let tileiras_fp = fingerprint_of(tileiras);
    let key = jit_cache::l2_key(bytecode, bc_version, gpu_name, opts, tileiras_fp);
    let params = EntryParams {
        bc_sha256: Sha256::digest(bytecode).into(),
        gpu_name,
        opt_level: opts.opt_level,
        flags: opts.flags_byte(),
        tileiras_fp,
    };

    match store.get(&key) {
        Ok(Some(entry)) => {
            if let Some(cubin) = jit_cache::decode_entry(&entry, &params) {
                jit_cache::STATS.hits.fetch_add(1, Ordering::Relaxed);
                // Hand the store and key back so a driver rejection recovers
                // against this exact entry (see `recompile_after_disk_rejection`).
                return Ok((cubin, Stage2Source::DiskCache { store, key }));
            }
            // Key matched but the entry does not validate against this request:
            // an incomplete write, accidental corruption, a request mismatch,
            // or a key collision. Drop it and recompile rather than serving it.
            crate::jit_cache::cache_log(format_args!(
                "disk cache entry {key} failed validation; deleting and recompiling"
            ));
            if let Err(e) = store.delete(&key) {
                jit_cache::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                crate::jit_cache::cache_log(format_args!(
                    "failed to delete invalid entry {key}: {e}"
                ));
            }
        }
        Ok(None) => {}
        Err(e) => {
            jit_cache::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
            crate::jit_cache::cache_log(format_args!("disk cache read for {key} failed: {e}"));
        }
    }

    jit_cache::STATS.misses.fetch_add(1, Ordering::Relaxed);
    let cubin = run_tileiras_at(tileiras, bytecode, gpu_name, opts)?;

    match jit_cache::encode_entry(&params, &cubin) {
        Some(entry) => match store.put(&key, &entry) {
            Ok(()) => {
                jit_cache::STATS.puts.fetch_add(1, Ordering::Relaxed);
                jit_cache::STATS
                    .bytes_written
                    .fetch_add(entry.len() as u64, Ordering::Relaxed);
            }
            Err(e) => {
                jit_cache::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                crate::jit_cache::cache_log(format_args!("disk cache write for {key} failed: {e}"));
            }
        },
        None => crate::jit_cache::cache_log(format_args!(
            "not caching {key}: gpu name or tileiras fingerprint exceeds the entry format's u16 length field"
        )),
    }

    Ok((cubin, Stage2Source::Tileiras))
}

/// Recovery for a structurally valid disk-served cubin that the driver
/// nevertheless rejected (invalid image, driver/toolkit skew, …).
///
/// Deletes the offending entry from the store it came from (best-effort) and
/// compiles with `tileiras` **directly, without consulting the cache**. The
/// bypass is the point: if the delete fails — a read-only or shared cache
/// directory, an entry owned by another user — reading the store again would
/// just re-serve the very cubin the driver already rejected, and the launch
/// would fail permanently. `store` and `key` come from the [`Stage2Source::DiskCache`]
/// that produced the bad cubin, so this evicts exactly that entry.
pub fn recompile_after_disk_rejection(
    store: &dyn crate::jit_cache::JitStore,
    key: &str,
    bytecode: &[u8],
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<Vec<u8>, JITError> {
    recompile_after_disk_rejection_at(&tileiras_binary(), store, key, bytecode, gpu_name, opts)
}

/// Retry against the same toolchain that produced the original cache request.
pub fn recompile_after_disk_rejection_with_toolkit(
    store: &dyn crate::jit_cache::JitStore,
    key: &str,
    bytecode: &[u8],
    opts: &TileirasOptions,
    capabilities: &ToolkitCapabilities,
) -> Result<Vec<u8>, JITError> {
    recompile_after_disk_rejection_at(
        &capabilities.tileiras,
        store,
        key,
        bytecode,
        &capabilities.target.architecture,
        opts,
    )
}

fn recompile_after_disk_rejection_at(
    tileiras: &Path,
    store: &dyn crate::jit_cache::JitStore,
    key: &str,
    bytecode: &[u8],
    gpu_name: &str,
    opts: &TileirasOptions,
) -> Result<Vec<u8>, JITError> {
    use std::sync::atomic::Ordering;

    if let Err(e) = store.delete(key) {
        crate::jit_cache::STATS
            .io_errors
            .fetch_add(1, Ordering::Relaxed);
        crate::jit_cache::cache_log(format_args!("failed to evict entry {key}: {e}"));
    }
    run_tileiras_at(tileiras, bytecode, gpu_name, opts)
}

/// Compiles a `cutile_ir::Module` to a cubin image via bytecode serialization and
/// `tileiras`, consulting the disk cache when one is installed.
///
/// Returns `Err` (not panic) on any failure so callers can propagate it and run
/// their cache-cleanup paths; a panic would unwind past that and across FFI frames.
pub fn compile_tile_ir_module(
    module: &cutile_ir::Module,
    gpu_name: &str,
) -> Result<Vec<u8>, JITError> {
    let (bytecode, bc_version) = serialize_tile_ir_bytecode(module)?;
    compile_bytecode_cached(&bytecode, bc_version, gpu_name, &TileirasOptions::default())
        .map(|(cubin, _)| cubin)
}

fn tileiras_launch_error(
    tileiras: &Path,
    args: &[&str],
    bc_filename: &str,
    error: std::io::Error,
) -> String {
    let mut message = format!(
        "failed to launch tileiras.\n\
         error: {error}\n\
         command: {}\n\
         bytecode: {bc_filename} (kept for inspection)\n\
         {TILEIRAS_PATH_ENV}: {}\n\
         {CUDA_TOOLKIT_PATH_ENV}: {}\n\
         {CUDA_HOME_ENV}: {}\n",
        display_command(tileiras, args),
        env::var(TILEIRAS_PATH_ENV).unwrap_or_else(|_| "<unset>".to_string()),
        env::var(CUDA_TOOLKIT_PATH_ENV).unwrap_or_else(|_| "<unset>".to_string()),
        env::var(CUDA_HOME_ENV).unwrap_or_else(|_| "<unset>".to_string()),
    );

    if env::var_os(TILEIRAS_PATH_ENV).is_none() {
        message.push_str(
            "hint: install CUDA 13.2+ with tileiras, set CUDA_TOOLKIT_PATH or CUDA_HOME to that \
             toolkit, set CUTILE_TILEIRAS_PATH to the absolute tileiras path, or rerun with \
             CUTILE_SETUP_DIAGNOSTICS=1 to trace toolkit discovery.",
        );
    } else {
        message
            .push_str("hint: verify CUTILE_TILEIRAS_PATH points to an executable tileiras binary.");
    }

    message
}

fn default_cuda_toolkit_candidates() -> &'static [PathBuf] {
    static CANDIDATES: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
    CANDIDATES.get_or_init(|| {
        #[cfg(windows)]
        let candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        ];
        #[cfg(not(windows))]
        let candidates = [
            "/usr/local/cuda-13.4",
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            "/usr/local/cuda",
            "/opt/cuda",
        ];

        candidates.into_iter().map(PathBuf::from).collect()
    })
}

fn default_cuda_toolkit_tileiras(candidates: &[PathBuf]) -> Option<PathBuf> {
    for candidate in candidates {
        match supported_cuda_toolkit_tileiras(candidate) {
            Ok(tileiras) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV}/{CUDA_HOME_ENV} are unset; using discovered tileiras at {}",
                    tileiras.display()
                ));
                return Some(tileiras);
            }
            Err(error) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV}/{CUDA_HOME_ENV} are unset; skipping {}: {error}",
                    candidate.display()
                ));
            }
        }
    }

    None
}

fn supported_cuda_toolkit_tileiras(cuda_toolkit: &Path) -> Result<PathBuf, String> {
    if !cuda_toolkit.is_dir() {
        return Err("not a directory".to_string());
    }

    let (_, version) = cuda_version_from_toolkit(cuda_toolkit)?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old",
            format_cuda_version(version)
        ));
    }

    let tileiras = cuda_toolkit.join("bin").join(tileiras_executable_name());
    if !tileiras.is_file() {
        return Err(format!("missing {}", tileiras.display()));
    }

    Ok(tileiras)
}

/// The `include/` directories of a toolkit root that may hold `cuda.h`, in
/// priority order: the standard top-level `include/`, then the
/// `targets/<dir>/include/` trees CUDA ships for this machine's platform
/// (`x86_64-linux`; `sbsa-linux` then `aarch64-linux` on aarch64, which a
/// Rust triple cannot tell apart). Redistributable and Tegra/sbsa layouts
/// have no top-level `include/`. Mirrors `toolkit_target_dirs` in
/// `cuda-bindings/toolkit_target.rs` for the build target; here the
/// running binary's platform is the target.
fn toolkit_include_dirs(cuda_toolkit: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![cuda_toolkit.join("include")];
    let target_dirs: &[&str] = match (env::consts::OS, env::consts::ARCH) {
        ("linux", "x86_64") => &["x86_64-linux"],
        ("linux", "aarch64") => &["sbsa-linux", "aarch64-linux"],
        _ => &[],
    };
    for dir in target_dirs {
        dirs.push(cuda_toolkit.join("targets").join(dir).join("include"));
    }
    dirs
}

/// `CUDA_VERSION` from the first `cuda.h` found under a toolkit root (see
/// [`toolkit_include_dirs`]), with the header's path. The error lists every
/// path probed.
fn cuda_version_from_toolkit(cuda_toolkit: &Path) -> Result<(PathBuf, u32), String> {
    let mut probed = Vec::new();
    for include_dir in toolkit_include_dirs(cuda_toolkit) {
        let cuda_h = include_dir.join("cuda.h");
        if cuda_h.is_file() {
            return cuda_version_from_header(&cuda_h).map(|version| (cuda_h, version));
        }
        probed.push(cuda_h.display().to_string());
    }
    Err(format!(
        "no cuda.h under {} (probed {})",
        cuda_toolkit.display(),
        probed.join(", ")
    ))
}

fn cuda_version_from_header(cuda_h: &Path) -> Result<u32, String> {
    let source = std::fs::read_to_string(cuda_h)
        .map_err(|error| format!("could not read {}: {error}", cuda_h.display()))?;
    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| format!("could not find CUDA_VERSION in {}", cuda_h.display()))
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

/// Returns whether the environment variable `var` is set to a truthy value
/// (`1` / `true` / `yes` / `on`, case-insensitive, surrounding whitespace ignored).
///
/// Shared by the crate's on/off diagnostic env vars so they all parse the same way.
pub fn env_flag_enabled(var: &str) -> bool {
    env::var(var).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn setup_diagnostics_enabled() -> bool {
    env_flag_enabled(SETUP_DIAGNOSTICS_ENV)
}

fn emit_setup_diagnostic(args: std::fmt::Arguments<'_>) {
    if setup_diagnostics_enabled() {
        eprintln!("cutile setup: {args}");
    }
}

fn display_command(program: &Path, args: &[&str]) -> String {
    std::iter::once(shell_display(program.as_os_str()))
        .chain(args.iter().map(|arg| shell_display(arg.as_ref())))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_display(value: &std::ffi::OsStr) -> String {
    let value = value.to_string_lossy();
    if value.is_empty() {
        "''".to_string()
    } else if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '='))
    {
        value.into_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use cutile_ir::bytecode::Opcode;
    use cutile_ir::ir::{Attribute, FuncType, Location, Module, Type};
    use std::fs;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// A toolkit older than the Tile floor is an `Err` from the `Result`
    /// path, never a panic (the JIT callers own cache cleanup on failure).
    #[test]
    #[cfg(unix)]
    fn pre_tile_toolkit_is_an_error_not_a_panic() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _override_guard = EnvVarGuard::unset(BYTECODE_VERSION_ENV);
        let temp_dir = env::temp_dir().join(format!("cutile_old_toolkit_{}", Uuid::new_v4()));
        let tileiras = create_fake_cuda_toolkit(&temp_dir, 13010, true);
        write_fake_tileiras_script(&tileiras, "printf '13.1\\n'\n");
        let err = compute_bytecode_version(&tileiras, Some(temp_dir.as_path()))
            .expect_err("a CUDA 13.1 toolkit must be rejected");
        let text = err.to_string();
        assert!(
            text.contains("CUDA 13.2 or newer") && text.contains("13.1"),
            "unexpected error: {text}"
        );
        let _ = fs::remove_dir_all(temp_dir);
    }

    /// `CUDA_HOME` names a toolkit root just like `CUDA_TOOLKIT_PATH`, and
    /// loses to it when both are set.
    #[test]
    #[cfg(unix)]
    fn cuda_home_is_honored_after_cuda_toolkit_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home_dir = env::temp_dir().join(format!("cutile_cuda_home_{}", Uuid::new_v4()));
        let path_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let home_tileiras = create_fake_cuda_toolkit(&home_dir, 13030, true);
        let path_tileiras = create_fake_cuda_toolkit(&path_dir, 13030, true);

        // Resolution through the CUDA_HOME value alone.
        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                Some((CUDA_HOME_ENV, home_dir.clone().into_os_string())),
                &[]
            ),
            home_tileiras
        );

        // Environment precedence: CUDA_HOME only, then both.
        let _unset_path = EnvVarGuard::unset(CUDA_TOOLKIT_PATH_ENV);
        let _home = EnvVarGuard::set(CUDA_HOME_ENV, &home_dir);
        assert_eq!(
            toolkit_env(),
            Some((CUDA_HOME_ENV, home_dir.clone().into_os_string()))
        );
        {
            let _path = EnvVarGuard::set(CUDA_TOOLKIT_PATH_ENV, &path_dir);
            assert_eq!(
                toolkit_env(),
                Some((CUDA_TOOLKIT_PATH_ENV, path_dir.clone().into_os_string()))
            );
            assert_eq!(
                cached_tileiras_and_toolkit(None, toolkit_env()).0,
                path_tileiras
            );
        }

        let _ = fs::remove_dir_all(home_dir);
        let _ = fs::remove_dir_all(path_dir);
    }

    /// `cuda.h` is found in a `targets/<dir>/include/` tree when the toolkit
    /// has no top-level `include/` (redistributable and Tegra/sbsa layouts).
    #[test]
    fn cuda_h_is_found_in_the_targets_layout() {
        let root = env::temp_dir().join(format!("cutile_targets_toolkit_{}", Uuid::new_v4()));
        let include_dirs = toolkit_include_dirs(&root);
        let Some(target_include) = include_dirs.get(1) else {
            eprintln!("skipping: CUDA ships no targets/ tree for this platform");
            return;
        };
        fs::create_dir_all(target_include).unwrap();
        fs::write(
            target_include.join("cuda.h"),
            "#define CUDA_VERSION 13020\n",
        )
        .unwrap();
        let (cuda_h, version) = cuda_version_from_toolkit(&root).expect("targets layout");
        assert_eq!(version, 13020);
        assert_eq!(cuda_h, target_include.join("cuda.h"));
        // ...and the top-level header still wins when both exist.
        fs::create_dir_all(root.join("include")).unwrap();
        fs::write(
            root.join("include").join("cuda.h"),
            "#define CUDA_VERSION 13030\n",
        )
        .unwrap();
        assert_eq!(cuda_version_from_toolkit(&root).unwrap().1, 13030);
        let err = cuda_version_from_toolkit(&root.join("nowhere")).unwrap_err();
        assert!(err.contains("no cuda.h under"), "unexpected error: {err}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tileiras_binary_defaults_to_path_lookup() {
        assert_eq!(
            resolve_tileiras_binary_with_candidates(None, None, &[]),
            PathBuf::from("tileiras")
        );
    }

    #[test]
    fn tileiras_binary_uses_override_path() {
        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                Some(OsString::from("/opt/cuda/bin/tileiras")),
                None,
                &[]
            ),
            PathBuf::from("/opt/cuda/bin/tileiras")
        );
    }

    #[test]
    fn tileiras_binary_treats_empty_override_as_default() {
        assert_eq!(
            resolve_tileiras_binary_with_candidates(Some(OsString::new()), None, &[]),
            PathBuf::from("tileiras")
        );
    }

    #[test]
    #[cfg(unix)]
    fn tileiras_binary_uses_cuda_toolkit_path_when_present() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let bin_dir = temp_dir.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let tileiras = bin_dir.join(tileiras_executable_name());
        fs::write(&tileiras, "").unwrap();

        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                Some((CUDA_TOOLKIT_PATH_ENV, temp_dir.clone().into_os_string())),
                &[]
            ),
            tileiras
        );

        let _ = fs::remove_file(bin_dir.join(tileiras_executable_name()));
        let _ = fs::remove_dir(bin_dir);
        let _ = fs::remove_dir(temp_dir);
    }

    #[test]
    fn tileiras_binary_ignores_cuda_toolkit_path_without_tileiras() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                Some((CUDA_TOOLKIT_PATH_ENV, temp_dir.into_os_string())),
                &[]
            ),
            PathBuf::from(tileiras_executable_name())
        );
    }

    #[test]
    fn tileiras_binary_uses_default_cuda_toolkit_when_supported() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let tileiras = create_fake_cuda_toolkit(&temp_dir, 13020, true);

        assert_eq!(
            resolve_tileiras_binary_with_candidates(None, None, std::slice::from_ref(&temp_dir)),
            tileiras
        );

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn tileiras_binary_skips_old_default_cuda_toolkit() {
        let old_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let new_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let _old_tileiras = create_fake_cuda_toolkit(&old_dir, 13010, true);
        let new_tileiras = create_fake_cuda_toolkit(&new_dir, 13020, true);

        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                None,
                &[old_dir.clone(), new_dir.clone()]
            ),
            new_tileiras
        );

        let _ = fs::remove_dir_all(old_dir);
        let _ = fs::remove_dir_all(new_dir);
    }

    #[test]
    #[cfg(unix)]
    fn binary_version_wins_over_toolkit_headers() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _override_guard = EnvVarGuard::unset(BYTECODE_VERSION_ENV);
        for (headers, assembler, expected) in [
            (13040, "13.1\\n13.2\\n13.3\\n", BytecodeVersion::V13_3),
            (
                13030,
                "13.1\\n13.2\\n13.3\\n13.4\\n",
                BytecodeVersion::V13_4,
            ),
        ] {
            let dir = env::temp_dir().join(format!("cutile_mixed_toolkit_{}", Uuid::new_v4()));
            let binary = create_fake_cuda_toolkit(&dir, headers, true);
            write_fake_tileiras_script(&binary, &format!("printf '{assembler}'\n"));
            assert_eq!(
                compute_bytecode_version(&binary, Some(&dir)).unwrap(),
                expected
            );
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    #[cfg(unix)]
    fn bytecode_override_is_validated_and_part_of_the_cache_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("cutile_bc_override_{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let binary = dir.join("tileiras");
        write_fake_tileiras_script(&binary, "printf '13.2\\n13.3\\n'\n");
        for requested in ["13.4", "13.9", "13.1", "nonsense", "13.3.1"] {
            assert!(
                negotiate_bytecode_version(&binary, Some(std::ffi::OsStr::new(requested))).is_err(),
                "{requested}"
            );
        }
        {
            let _override_guard = EnvVarGuard::set(BYTECODE_VERSION_ENV, Path::new("13.2"));
            assert_eq!(
                cached_bytecode_version(&binary, None).unwrap(),
                BytecodeVersion::V13_2
            );
        }
        {
            let _override_guard = EnvVarGuard::set(BYTECODE_VERSION_ENV, Path::new("13.3"));
            assert_eq!(
                cached_bytecode_version(&binary, None).unwrap(),
                BytecodeVersion::V13_3
            );
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn selects_bytecode_version_from_toolkit_binary() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _override_guard = EnvVarGuard::unset(BYTECODE_VERSION_ENV);
        let temp_dir = env::temp_dir().join(format!("cutile_bc_ver_{}", Uuid::new_v4()));
        let tileiras = create_fake_cuda_toolkit(&temp_dir, 13020, true);
        write_fake_tileiras_script(&tileiras, "printf '13.1\\n13.2\\n'\n");
        let toolkit = toolkit_root_of(&tileiras);
        assert_eq!(toolkit.as_deref(), Some(temp_dir.as_path()));
        // The binary reports 13.2, independently of the nearby header.
        assert_eq!(
            compute_bytecode_version(&tileiras, toolkit.as_deref()).expect("13.2 toolkit"),
            BytecodeVersion::V13_2
        );
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    #[cfg(unix)]
    fn compile_tile_ir_module_uses_tileiras_path_override() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_tileiras_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);

        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let module = empty_kernel_module();
        let cubin = compile_tile_ir_module(&module, "sm_120")
            .expect("compiling an empty kernel with the fake tileiras should succeed");

        let args_path = fake_tileiras.with_extension("args");
        let args = fs::read_to_string(&args_path).unwrap();
        assert!(
            args.lines()
                .next()
                .is_some_and(|line| line == fake_tileiras.to_string_lossy()),
            "expected fake tileiras to record its own path, got:\n{args}"
        );
        assert!(args.contains("--gpu-name\nsm_120"), "args:\n{args}");
        assert!(args.contains("--opt-level\n3"), "args:\n{args}");
        assert!(args.contains("-o\n"), "args:\n{args}");

        // `write_fake_tileiras` writes exactly this to the `-o` path.
        assert_eq!(cubin, b"fake cubin\n".to_vec());

        // Both temp files are removed before `run_tileiras` returns. The fake
        // tileiras recorded their paths, so check them directly.
        let cubin_path = {
            let mut lines = args.lines();
            lines.find(|line| *line == "-o");
            lines
                .next()
                .expect("fake tileiras should have recorded an -o path")
        };
        let bc_path = args.lines().last().unwrap_or_default();
        assert!(
            !PathBuf::from(cubin_path).exists(),
            "run_tileiras leaked its output cubin at {cubin_path}"
        );
        assert!(
            !PathBuf::from(bc_path).exists(),
            "run_tileiras leaked its input bytecode at {bc_path}"
        );

        let _ = fs::remove_file(args_path);
        let _ = fs::remove_file(fake_tileiras);
        let _ = fs::remove_dir(temp_dir);
    }

    /// End-to-end cache path with a fake `tileiras`: the first compile spawns
    /// the subprocess and writes the store entry, the second is served from
    /// disk without spawning. `enable`/`disable` happen under `ENV_LOCK`, the
    /// same lock the other tileiras-spawning test takes, so the global store
    /// never leaks into it.
    #[test]
    #[cfg(unix)]
    fn disk_cache_serves_second_compile_without_spawning() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_jit_cache_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);
        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let store_dir = temp_dir.join("store");
        crate::jit_cache::enable(std::sync::Arc::new(
            crate::jit_cache::FileSystemJitStore::new(&store_dir).unwrap(),
        ));

        let module = empty_kernel_module();
        let backend_before = crate::jit_cache::jit_backend_compile_count();
        let hits_before = crate::jit_cache::jit_disk_hit_count();

        let first =
            compile_tile_ir_module(&module, "sm_120").expect("first compile (miss) should succeed");
        let second =
            compile_tile_ir_module(&module, "sm_120").expect("second compile (hit) should succeed");

        // A different target is a different key: this one must miss.
        let other_arch = compile_tile_ir_module(&module, "sm_100")
            .expect("different-arch compile should succeed");

        crate::jit_cache::disable();

        assert_eq!(first, second, "hit must return the exact bytes stored");
        assert_eq!(first, other_arch, "fake tileiras writes constant bytes");
        assert_eq!(
            crate::jit_cache::jit_backend_compile_count() - backend_before,
            2,
            "exactly the two misses spawn tileiras"
        );
        assert_eq!(
            crate::jit_cache::jit_disk_hit_count() - hits_before,
            1,
            "exactly the repeat compile hits the disk"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// A malformed disk entry is detected, deleted, and replaced by a fresh
    /// compile. This is the end-to-end coverage for the delete-on-mismatch path
    /// described in PR #193: an incomplete or validation-mismatched entry must
    /// not be served.
    #[test]
    #[cfg(unix)]
    fn disk_cache_deletes_invalid_entry_and_recompiles() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_jit_cache_corrupt_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);
        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let store_dir = temp_dir.join("store");
        crate::jit_cache::enable(std::sync::Arc::new(
            crate::jit_cache::FileSystemJitStore::new(&store_dir).unwrap(),
        ));

        let module = empty_kernel_module();
        let (bytecode, bc_version) =
            serialize_tile_ir_bytecode(&module).expect("serialize should succeed");
        let gpu_name = "sm_120";
        let tileiras_fp = tileiras_fingerprint();
        let key = crate::jit_cache::l2_key(
            &bytecode,
            bc_version,
            gpu_name,
            &TileirasOptions::default(),
            tileiras_fp,
        );

        // Plant a garbage entry at the exact path the store would use.
        let shard_dir = store_dir.join(&key[..2]);
        fs::create_dir_all(&shard_dir).unwrap();
        let entry_path = shard_dir.join(format!("{key}.cubin"));
        fs::write(&entry_path, b"not a valid cache entry").unwrap();

        let backend_before = crate::jit_cache::jit_backend_compile_count();
        let hits_before = crate::jit_cache::jit_disk_hit_count();

        let result = compile_tile_ir_module(&module, gpu_name)
            .expect("recompile after corruption should succeed");

        // The corrupted entry should now be a valid hit.
        let cached = compile_tile_ir_module(&module, gpu_name)
            .expect("second call after repair should succeed");

        crate::jit_cache::disable();

        assert_eq!(
            result, cached,
            "repair must store the same bytes tileiras produced"
        );
        assert_eq!(
            crate::jit_cache::jit_backend_compile_count() - backend_before,
            1,
            "exactly one recompile after deleting the corrupted entry"
        );
        assert_eq!(
            crate::jit_cache::jit_disk_hit_count() - hits_before,
            1,
            "the repaired entry is served on the next call"
        );
        assert_ne!(
            fs::read(&entry_path).unwrap_or_default(),
            b"not a valid cache entry"[..],
            "the corrupted entry file must have been replaced"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// `recompile_after_disk_rejection` deletes the bad entry and recompiles
    /// with `tileiras` directly, bypassing the cache so a still-present bad entry
    /// cannot be re-served. This pins the bypass behavior that the GPU driver
    /// rejection path relies on.
    #[test]
    #[cfg(unix)]
    fn recompile_after_disk_rejection_deletes_and_bypasses() {
        use crate::jit_cache::JitStore;

        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_jit_reject_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);
        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let store_dir = temp_dir.join("store");
        let store: std::sync::Arc<dyn JitStore> =
            std::sync::Arc::new(crate::jit_cache::FileSystemJitStore::new(&store_dir).unwrap());
        crate::jit_cache::enable(store.clone());

        let module = empty_kernel_module();
        let (bytecode, bc_version) =
            serialize_tile_ir_bytecode(&module).expect("serialize should succeed");
        let gpu_name = "sm_120";
        let tileiras_fp = tileiras_fingerprint();
        let key = crate::jit_cache::l2_key(
            &bytecode,
            bc_version,
            gpu_name,
            &TileirasOptions::default(),
            tileiras_fp,
        );

        let first = compile_tile_ir_module(&module, gpu_name)
            .expect("first compile should populate the store");
        assert!(
            store.contains(&key).expect("contains should not error"),
            "store should contain the freshly compiled entry"
        );

        let backend_before = crate::jit_cache::jit_backend_compile_count();

        let repaired = recompile_after_disk_rejection(
            store.as_ref(),
            &key,
            &bytecode,
            gpu_name,
            &TileirasOptions::default(),
        )
        .expect("recompile_after_disk_rejection should succeed");

        crate::jit_cache::disable();

        assert_eq!(repaired, first, "recompile should produce the same cubin");
        assert_eq!(
            crate::jit_cache::jit_backend_compile_count() - backend_before,
            1,
            "recompile_after_disk_rejection must spawn tileiras exactly once"
        );
        assert!(
            store.get(&key).expect("get should not error").is_none(),
            "the rejected entry must be deleted from the store"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let previous = env::var_os(key);
            env::set_var(key, value);
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = env::var_os(key);
            env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => env::set_var(self.key, previous),
                None => env::remove_var(self.key),
            }
        }
    }

    fn empty_kernel_module() -> Module {
        let mut module = Module::new("tileiras_override_test");
        let func_type = Type::Func(FuncType {
            inputs: vec![],
            results: vec![],
        });

        let (region_id, block_id, _) = build_single_block_region(&mut module, &[]);
        let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
        append_op(&mut module, block_id, ret_id);

        let (entry_id, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", Attribute::String("empty_kernel".into()))
            .attr("function_type", Attribute::Type(func_type))
            .region(region_id)
            .build(&mut module);
        module.functions.push(entry_id);
        module
    }

    fn create_fake_cuda_toolkit(path: &Path, cuda_version: u32, include_tileiras: bool) -> PathBuf {
        let include_dir = path.join("include");
        let bin_dir = path.join("bin");
        fs::create_dir_all(&include_dir).unwrap();
        fs::create_dir_all(&bin_dir).unwrap();
        fs::write(
            include_dir.join("cuda.h"),
            format!("#define CUDA_VERSION {cuda_version}\n"),
        )
        .unwrap();

        let tileiras = bin_dir.join(tileiras_executable_name());
        if include_tileiras {
            fs::write(&tileiras, "").unwrap();
        }
        tileiras
    }

    #[cfg(unix)]
    fn write_fake_tileiras(path: &std::path::Path) {
        write_fake_tileiras_script(
            path,
            r#"args_file="$0.args"
printf '%s\n' "$0" "$@" > "$args_file"
out=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then
        shift
        out="$1"
    fi
    shift || break
done
if [ -z "$out" ]; then
    echo "missing -o output" >&2
    exit 2
fi
printf 'fake cubin\n' > "$out"
"#,
        );
    }

    /// Writes an executable `sh` script standing in for `tileiras`.
    #[cfg(unix)]
    fn write_fake_tileiras_script(path: &std::path::Path, body: &str) {
        use std::io::Write;
        use std::process::Stdio;

        // Keep writable executable fds out of this multithreaded process.
        // A concurrent child can inherit one until exec, causing ETXTBSY in
        // another launch even after fs::write has returned. Wait for a separate
        // writer process to exit before executing the fixture.
        let mut writer = Command::new("sh")
            .args([
                "-c",
                "cat > \"$1\" && chmod 755 \"$1\"",
                "write-fake-tileiras",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        writer
            .stdin
            .take()
            .unwrap()
            .write_all(format!("#!/bin/sh\nset -eu\n{body}").as_bytes())
            .unwrap();
        assert!(writer.wait().unwrap().success());
    }
}
