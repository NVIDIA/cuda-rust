/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build a cubin from cuda-oxide's NVVM IR output via libNVVM + libdevice + nvJitLink.
//!
//! When a kernel uses Rust float math intrinsics (`sin`, `cos`, `exp`, `pow`,
//! ...), cuda-oxide lowers them to CUDA `__nv_*` libdevice calls, auto-detects
//! their presence, emits NVVM IR (`<name>.ll`) instead of `.ptx`, and skips
//! `llc`. The application then has to:
//!
//! 1. Compile the NVVM IR to LTOIR via libNVVM, with libdevice added so the
//!    `__nv_*` symbols are inlined.
//! 2. Link the resulting LTOIR via nvJitLink to produce either a cubin for
//!    the same architecture, or PTX when a pre-Blackwell module is loaded on
//!    a Blackwell GPU.
//! 3. Load that image with the CUDA driver.
//!
//! This module wraps that pipeline behind file and in-memory helpers:
//!
//! - [`build_cubin_from_ll`] -- explicit form, takes a `.ll` path and arch.
//! - [`load_kernel_module`] -- the convenience form. Looks at the example's
//!   directory and selects PTX, cubin, NVVM IR, or LTOIR using the recorded
//!   artifact mode. Use this for normal module loading.
//!
//! All compilation work is delegated to [`cuda_artifact_finalizer`] (`dlopen`
//! of `libnvvm.so` and `libnvJitLink.so` from the CUDA Toolkit). No external
//! C tools are required, no symlinked `tools/` directory, no boilerplate.
//!
//! # Discovery
//!
//! - **libNVVM**: `LIBNVVM_PATH`, then `<root>/nvvm/lib64/libnvvm.so` for
//!   `<root>` in `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`,
//!   `/usr/local/cuda`, `/opt/cuda`, then the system loader.
//! - **nvJitLink**: the same order, using `LIBNVJITLINK_PATH` and
//!   `<root>/lib64/libnvJitLink.so`.
//! - **libdevice**: `CUDA_OXIDE_LIBDEVICE` env var, then
//!   `<root>/nvvm/libdevice/libdevice.10.bc` for the same roots.
//! - **Artifact target and options**: the emitted `<name>.target` records the
//!   source architecture. Versioned targets require `<name>.options`, which
//!   preserves FMA and debug policy through libNVVM and nvJitLink. Version-1
//!   sidecars imply no debug output; version 2 records line-table or full-debug
//!   preservation. The explicit `CUDA_OXIDE_TARGET` override remains the
//!   fallback for legacy artifacts. The CUDA context is queried separately for
//!   the GPU that will execute the module.
//!
//! # Native cubin cache
//!
//! File-backed NVVM IR and LTOIR use a persistent cache only when they can run
//! as a native cubin on the current GPU. Ordinary PTX and the pre-Blackwell to
//! Blackwell PTX bridge keep their normal paths.
//!
//! An entry is keyed by the exact input bytes, normalized target, module names,
//! ordered compiler/linker options (including FMA and debug policy), libdevice
//! bytes, and SHA-256 digests of the exact loaded `libnvvm.so` and
//! `libnvJitLink.so` files. Unknown tool identity
//! disables reuse. Entries are published atomically below
//! `.oxide-artifacts/ltoir-cubin-cache/v1` and are rechecked before their owned
//! bytes are passed to the CUDA driver. The first fingerprinted compiler and
//! linker handles are retained for the process lifetime, so the code that runs
//! cannot drift away from the digest. Restart the process to select another
//! toolkit or pick up an in-place toolkit upgrade.
//!
//! # Example
//!
//! ```no_run
//! use cuda_core::CudaContext;
//! use cuda_host::ltoir;
//!
//! let ctx = CudaContext::new(0)?;
//! // Loads my_kernel.cubin (or builds + loads from my_kernel.ll).
//! let module = ltoir::load_kernel_module(&ctx, "my_kernel")?;
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

use crate::ltoir_cache::{BuiltArtifacts, CacheResult, cache_or_build};
use cuda_artifact_finalizer::{
    CudaArch, CudaArchParseError, FinalizationOptions, Finalizer, FinalizerError, FinalizerOutput,
    LtoLinker, NamedInput, NvJitLinkError, NvvmError,
};
#[cfg(test)]
use cuda_artifact_finalizer::{
    ToolProvenance, ltoir_artifact_digest_with_provenance, nvvm_ir_artifact_digest_with_provenance,
};
use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactDebugPolicy, COMPILE_OPTIONS_TARGET_MARKER,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
#[cfg(test)]
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

/// Failures while building or loading a module via the LTOIR pipeline.
#[derive(Debug, Error)]
pub enum LtoirError {
    /// A target string was not a concrete CUDA architecture.
    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    /// NVVM IR/LTOIR was loaded without its original target architecture.
    #[error(
        "NVVM IR/LTOIR does not record its CUDA target. Keep the generated .target file, \
         rebuild the artifact, or explicitly set CUDA_OXIDE_TARGET to its original target."
    )]
    TargetNotFound,

    /// The caller tried to compile target-specific NVVM IR for another target.
    #[error(
        "NVVM IR target mismatch: {ir_path} was emitted for {emitted}, but the loader requested {requested}"
    )]
    TargetMismatch {
        ir_path: PathBuf,
        emitted: String,
        requested: String,
    },

    /// An NVVM IR compile-options sidecar contained an unsupported value.
    #[error("invalid cuda-oxide compile metadata in {path}: {value}")]
    InvalidCompileOptions {
        /// Path to the malformed sidecar.
        path: PathBuf,
        /// Trimmed sidecar contents.
        value: String,
    },

    /// An artifact is not compatible with the context's GPU.
    #[error("cannot execute CUDA artifact emitted for {emitted} on {execution}: {reason}")]
    IncompatibleExecutionTarget {
        /// Architecture for which the NVVM IR or LTOIR was emitted.
        emitted: String,
        /// Numeric architecture reported by the CUDA context.
        execution: String,
        /// Why the artifact is incompatible.
        reason: &'static str,
    },

    /// The installed toolkit does not accept the NVVM IR version cuda-oxide emits.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Runtime toolkit dialect discovery disagreed with cuda-oxide's target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// libNVVM failed (load, symbol resolution, or compile call). Forwards
    /// the underlying [`NvvmError`].
    #[error("libnvvm: {0}")]
    Nvvm(#[from] NvvmError),

    /// nvJitLink failed (load, symbol resolution, or link call). Forwards
    /// the underlying [`NvJitLinkError`].
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] NvJitLinkError),

    /// `libdevice.10.bc` could not be located. `tried` lists every path
    /// that was probed, in order, joined by newlines.
    #[error(
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibdeviceNotFound {
        /// Newline-joined list of paths that were probed.
        tried: String,
    },

    /// Reading or writing one of the pipeline artifacts (`.ll`,
    /// `libdevice.10.bc`, `.ltoir`, `.cubin`) failed.
    #[error("Failed reading {path}: {source}")]
    Io {
        /// Path of the file that could not be read or written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// [`load_kernel_module`] could not find a supported artifact in the
    /// binary's manifest directory.
    #[error(
        "Could not find any kernel artifact for {name} in {dir}. \
         Looked for {name}.cubin, {name}.ptx, {name}.ll, {name}.ltoir. \
         Did `cargo oxide run` complete successfully?"
    )]
    NoArtifact {
        /// Kernel artifact stem that was looked up.
        name: String,
        /// Directory that was searched.
        dir: PathBuf,
    },

    /// `cuModuleLoad` (or another driver call) returned an error after the
    /// pipeline produced a cubin.
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),

    /// The shared driver-independent finalizer rejected an invalid plan or
    /// malformed tool output.
    #[error(transparent)]
    Finalizer(FinalizerError),
}

impl From<FinalizerError> for LtoirError {
    fn from(error: FinalizerError) -> Self {
        match error {
            FinalizerError::Nvvm(error) => Self::Nvvm(error),
            FinalizerError::NvJitLink(error) => Self::NvJitLink(error),
            FinalizerError::LibdeviceNotFound { tried } => Self::LibdeviceNotFound { tried },
            FinalizerError::Io { path, source } => Self::Io { path, source },
            FinalizerError::UnsupportedNvvmIrVersion { major, minor } => {
                Self::UnsupportedNvvmIrVersion { major, minor }
            }
            FinalizerError::DialectMismatch {
                target,
                llvm_major,
                expected,
            } => Self::DialectMismatch {
                target,
                llvm_major,
                expected,
            },
            other => Self::Finalizer(other),
        }
    }
}

// ============================================================================
// Build (NVVM IR + libdevice -> LTOIR -> cubin)
// ============================================================================

/// Compile NVVM IR at `ll_path` to a cubin and return its path.
///
/// Steps:
/// 1. Read `ll_path` (NVVM IR text) and the libdevice bitcode (located via
///    [`find_libdevice`]).
/// 2. Reuse a verified content-addressed entry when every semantic input and
///    the exact loaded CUDA tool binaries match. Otherwise compile via
///    libNVVM and link via nvJitLink.
/// 3. Write compatibility copies next to `ll_path` as `<stem>.ltoir` and
///    `<stem>.cubin`, and return the sibling cubin path.
///
/// `arch` must be a concrete `sm_XX` or `compute_XX` CUDA target. It is
/// normalized to `compute_XX` for libNVVM and `sm_XX` for nvJitLink.
///
/// The cache lives below `.oxide-artifacts/ltoir-cubin-cache/v1` and is skipped
/// when the exact loaded `libnvvm.so` or `libnvJitLink.so` cannot be identified.
/// Cache failures never prevent a fresh build.
pub fn build_cubin_from_ll(ll_path: &Path, arch: &str) -> Result<PathBuf, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    Ok(build_cubin_from_ll_file(ll_path, &arch)?.path)
}

struct FileCubin {
    path: PathBuf,
    image: Vec<u8>,
}

fn build_cubin_from_ll_file(ll_path: &Path, arch: &CudaArch) -> Result<FileCubin, LtoirError> {
    // Read the source before writing target metadata or derived artifacts.
    let ll_bytes = std::fs::read(ll_path).map_err(|source| LtoirError::Io {
        path: ll_path.to_path_buf(),
        source,
    })?;
    let compile_options = read_compile_options(ll_path)?;
    validate_ir_target_sidecar(ll_path, arch)?;
    // Record the supplied target for older or manually created `.ll` files.
    // The sibling `.ltoir` can then be loaded after the `.ll` is removed.
    write_artifact_target_sidecar(ll_path, arch)?;

    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");
    let dir = ll_path.parent().unwrap_or_else(|| Path::new("."));
    let ltoir_path = dir.join(format!("{stem}.ltoir"));
    let cubin_path = dir.join(format!("{stem}.cubin"));

    let cached = cached_nvvm_ir_to_cubin_with_compile_options(
        dir,
        &ll_bytes,
        &ll_path.display().to_string(),
        &ltoir_path.display().to_string(),
        arch,
        compile_options,
    )?;
    let ltoir = cached
        .ltoir
        .as_deref()
        .expect("NVVM IR cache results always include the generated LTOIR");

    std::fs::write(&ltoir_path, ltoir).map_err(|source| LtoirError::Io {
        path: ltoir_path.clone(),
        source,
    })?;

    // Keep the historical sibling output for tools and callers that inspect
    // it, but never load it after checking the cache: another process may
    // replace a mutable sibling between those two operations.
    std::fs::write(&cubin_path, &cached.cubin).map_err(|source| LtoirError::Io {
        path: cubin_path.clone(),
        source,
    })?;

    Ok(FileCubin {
        path: cubin_path,
        image: cached.cubin,
    })
}

/// Compile NVVM IR bytes to a loadable cubin image in memory.
///
/// This is the embedded-artifact counterpart of [`build_cubin_from_ll`]. It
/// adds `libdevice.10.bc`, asks libNVVM for LTOIR, links that LTOIR with
/// nvJitLink, and returns the final cubin bytes without creating sidecar files.
pub fn build_cubin_from_nvvm_ir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    build_cubin_from_nvvm_ir_with_options(nvvm_ir, module_name, arch, true)
}

/// Compile NVVM IR bytes to a loadable cubin with an explicit FMA policy.
///
/// This compatibility helper uses no debug information. Use
/// [`build_cubin_from_nvvm_ir_with_compile_options`] when all deferred artifact
/// policy must be preserved.
pub fn build_cubin_from_nvvm_ir_with_options(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
    allow_fma_contraction: bool,
) -> Result<Vec<u8>, LtoirError> {
    build_cubin_from_nvvm_ir_with_compile_options(
        nvvm_ir,
        module_name,
        arch,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

/// Compile NVVM IR bytes to a cubin while preserving every deferred compile
/// policy recorded with the artifact.
pub fn build_cubin_from_nvvm_ir_with_compile_options(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    let options = finalization_options(&arch, compile_options);
    Ok(Finalizer::discover()?.materialize_nvvm_ir(module_name, nvvm_ir, &options)?)
}

/// Compile NVVM IR bytes to forward-compatible PTX.
///
/// `arch` is the architecture for which the IR was **emitted**, not the GPU
/// on which it will execute. nvJitLink preserves that virtual architecture;
/// the CUDA driver later JIT-compiles the resulting PTX for the current GPU.
pub fn build_ptx_from_nvvm_ir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    build_ptx_from_nvvm_ir_with_options(nvvm_ir, module_name, arch, true)
}

/// Compile NVVM IR bytes to forward-compatible PTX with an explicit FMA
/// policy.
///
/// This compatibility helper uses no debug information. Use
/// [`build_ptx_from_nvvm_ir_with_compile_options`] when all deferred artifact
/// policy must be preserved.
pub fn build_ptx_from_nvvm_ir_with_options(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
    allow_fma_contraction: bool,
) -> Result<Vec<u8>, LtoirError> {
    build_ptx_from_nvvm_ir_with_compile_options(
        nvvm_ir,
        module_name,
        arch,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

/// Compile NVVM IR bytes to PTX while preserving every deferred compile
/// policy recorded with the artifact.
pub fn build_ptx_from_nvvm_ir_with_compile_options(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    let finalizer = Finalizer::discover()?;
    let options = finalization_options(&arch, compile_options);
    let ltoir = finalizer
        .compiler()
        .compile_nvvm_ir_to_ltoir(module_name, nvvm_ir, &options)?;
    let ltoir_name = format!("{module_name}.ltoir");
    Ok(finalizer.link_ltoir(
        &[NamedInput::new(&ltoir_name, &ltoir)],
        &options,
        FinalizerOutput::Ptx,
    )?)
}

/// Link a single LTOIR payload to a loadable cubin image in memory.
pub fn link_ltoir_to_cubin(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_to_cubin_with_options(ltoir, module_name, arch, true)
}

/// Link a single LTOIR payload to a cubin with an explicit FMA policy.
///
/// This compatibility helper uses no debug information. Use
/// [`link_ltoir_to_cubin_with_compile_options`] when all deferred artifact
/// policy must be preserved.
pub fn link_ltoir_to_cubin_with_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
    allow_fma_contraction: bool,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_to_cubin_with_compile_options(
        ltoir,
        module_name,
        arch,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

/// Link LTOIR to a cubin while preserving every deferred compile policy
/// recorded with the artifact.
pub fn link_ltoir_to_cubin_with_compile_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    link_ltoir_to_cubin_parsed_with_compile_options(ltoir, module_name, &arch, compile_options)
}

/// Link one LTOIR payload to forward-compatible PTX.
///
/// `arch` must be the target recorded with the LTOIR payload. The returned PTX
/// keeps that target and is JIT-compiled by the CUDA driver for the current GPU.
pub fn link_ltoir_to_ptx(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_to_ptx_with_options(ltoir, module_name, arch, true)
}

/// Link one LTOIR payload to forward-compatible PTX with an explicit FMA
/// policy.
///
/// This compatibility helper uses no debug information. Use
/// [`link_ltoir_to_ptx_with_compile_options`] when all deferred artifact policy
/// must be preserved.
pub fn link_ltoir_to_ptx_with_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
    allow_fma_contraction: bool,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_to_ptx_with_compile_options(
        ltoir,
        module_name,
        arch,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

/// Link LTOIR to PTX while preserving every deferred compile policy recorded
/// with the artifact.
pub fn link_ltoir_to_ptx_with_compile_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    link_ltoir_to_ptx_parsed_with_compile_options(ltoir, module_name, &arch, compile_options)
}

fn link_ltoir_to_cubin_parsed_with_compile_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_with_shared_finalizer(
        ltoir,
        module_name,
        arch,
        compile_options,
        FinalizerOutput::Cubin,
    )
}

fn link_ltoir_to_ptx_parsed_with_compile_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
) -> Result<Vec<u8>, LtoirError> {
    link_ltoir_with_shared_finalizer(
        ltoir,
        module_name,
        arch,
        compile_options,
        FinalizerOutput::Ptx,
    )
}

fn link_ltoir_with_shared_finalizer(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
    output: FinalizerOutput,
) -> Result<Vec<u8>, LtoirError> {
    let options = finalization_options(arch, compile_options);
    Ok(LtoLinker::discover()?.link_ltoir(
        &[NamedInput::new(module_name, ltoir)],
        &options,
        output,
    )?)
}

fn finalization_options(
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
) -> FinalizationOptions {
    let debug = match compile_options.debug_policy() {
        ArtifactDebugPolicy::None => cuda_artifact_finalizer::DebugPolicy::None,
        ArtifactDebugPolicy::LineTables => cuda_artifact_finalizer::DebugPolicy::LineTables,
        ArtifactDebugPolicy::Full => cuda_artifact_finalizer::DebugPolicy::Full,
    };
    FinalizationOptions::new(arch.clone())
        .with_fma_contraction(compile_options.fma_contraction_enabled())
        .with_debug_policy(debug)
}

#[cfg(test)]
fn cached_nvvm_ir_to_cubin(
    source_dir: &Path,
    nvvm_ir: &[u8],
    nvvm_module_name: &str,
    ltoir_module_name: &str,
    arch: &CudaArch,
) -> Result<CacheResult, LtoirError> {
    cached_nvvm_ir_to_cubin_with_compile_options(
        source_dir,
        nvvm_ir,
        nvvm_module_name,
        ltoir_module_name,
        arch,
        ArtifactCompileOptions::new(),
    )
}

fn cached_nvvm_ir_to_cubin_with_compile_options(
    source_dir: &Path,
    nvvm_ir: &[u8],
    nvvm_module_name: &str,
    ltoir_module_name: &str,
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
) -> Result<CacheResult, LtoirError> {
    let finalizer = Finalizer::discover()?;
    let options = finalization_options(arch, compile_options);
    let key = finalizer.nvvm_ir_artifact_digest(
        nvvm_module_name,
        ltoir_module_name,
        nvvm_ir,
        &options,
        FinalizerOutput::Cubin,
    );

    let build = || -> Result<BuiltArtifacts, LtoirError> {
        let ltoir =
            finalizer
                .compiler()
                .compile_nvvm_ir_to_ltoir(nvvm_module_name, nvvm_ir, &options)?;
        let cubin = finalizer.link_ltoir(
            &[NamedInput::new(ltoir_module_name, &ltoir)],
            &options,
            FinalizerOutput::Cubin,
        )?;
        Ok(BuiltArtifacts::new(cubin, Some(ltoir)))
    };

    let result = match key {
        Some(key) => cache_or_build(source_dir, &key, build)?,
        None => uncached_result(build()?),
    };
    report_cache_result(&result);
    Ok(result)
}

#[cfg(test)]
fn cached_ltoir_to_cubin(
    source_dir: &Path,
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<CacheResult, LtoirError> {
    cached_ltoir_to_cubin_with_compile_options(
        source_dir,
        ltoir,
        module_name,
        arch,
        ArtifactCompileOptions::new(),
    )
}

fn cached_ltoir_to_cubin_with_compile_options(
    source_dir: &Path,
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    compile_options: ArtifactCompileOptions,
) -> Result<CacheResult, LtoirError> {
    let linker = LtoLinker::discover()?;
    let options = finalization_options(arch, compile_options);
    let inputs = [NamedInput::new(module_name, ltoir)];
    let key = linker.artifact_digest(&inputs, &options, FinalizerOutput::Cubin);
    let build = || -> Result<BuiltArtifacts, LtoirError> {
        Ok(BuiltArtifacts::new(
            linker.link_ltoir(&inputs, &options, FinalizerOutput::Cubin)?,
            None,
        ))
    };

    let result = match key {
        Some(key) => cache_or_build(source_dir, &key, build)?,
        None => uncached_result(build()?),
    };
    report_cache_result(&result);
    Ok(result)
}

#[cfg(test)]
fn nvvm_ir_cubin_cache_key(
    nvvm_ir: &[u8],
    nvvm_module_name: &str,
    ltoir_module_name: &str,
    arch: &CudaArch,
    libdevice: &[u8],
    libnvvm_digest: &[u8; 32],
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    nvvm_ir_cubin_cache_key_with_compile_options(
        nvvm_ir,
        nvvm_module_name,
        ltoir_module_name,
        arch,
        libdevice,
        (libnvvm_digest, nvjitlink_digest),
        ArtifactCompileOptions::new(),
    )
}

#[cfg(test)]
fn nvvm_ir_cubin_cache_key_with_options(
    nvvm_ir: &[u8],
    nvvm_module_name: &str,
    ltoir_module_name: &str,
    arch: &CudaArch,
    libdevice: &[u8],
    tool_digests: (&[u8; 32], &[u8; 32]),
    allow_fma_contraction: bool,
) -> [u8; 32] {
    nvvm_ir_cubin_cache_key_with_compile_options(
        nvvm_ir,
        nvvm_module_name,
        ltoir_module_name,
        arch,
        libdevice,
        tool_digests,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

#[cfg(test)]
fn nvvm_ir_cubin_cache_key_with_compile_options(
    nvvm_ir: &[u8],
    nvvm_module_name: &str,
    ltoir_module_name: &str,
    arch: &CudaArch,
    libdevice: &[u8],
    tool_digests: (&[u8; 32], &[u8; 32]),
    compile_options: ArtifactCompileOptions,
) -> [u8; 32] {
    let (libnvvm_digest, nvjitlink_digest) = tool_digests;
    nvvm_ir_artifact_digest_with_provenance(
        nvvm_module_name,
        ltoir_module_name,
        nvvm_ir,
        &finalization_options(arch, compile_options),
        FinalizerOutput::Cubin,
        ToolProvenance {
            libnvvm_sha256: Some(*libnvvm_digest),
            nvjitlink_sha256: Some(*nvjitlink_digest),
            libdevice_sha256: Sha256::digest(libdevice).into(),
        },
    )
    .expect("test provenance supplies exact CUDA tool identities")
}

#[cfg(test)]
fn ltoir_cubin_cache_key(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    ltoir_cubin_cache_key_with_compile_options(
        ltoir,
        module_name,
        arch,
        nvjitlink_digest,
        ArtifactCompileOptions::new(),
    )
}

#[cfg(test)]
fn ltoir_cubin_cache_key_with_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    nvjitlink_digest: &[u8; 32],
    allow_fma_contraction: bool,
) -> [u8; 32] {
    ltoir_cubin_cache_key_with_compile_options(
        ltoir,
        module_name,
        arch,
        nvjitlink_digest,
        ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction),
    )
}

#[cfg(test)]
fn ltoir_cubin_cache_key_with_compile_options(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
    nvjitlink_digest: &[u8; 32],
    compile_options: ArtifactCompileOptions,
) -> [u8; 32] {
    ltoir_artifact_digest_with_provenance(
        &[NamedInput::new(module_name, ltoir)],
        &finalization_options(arch, compile_options),
        FinalizerOutput::Cubin,
        nvjitlink_digest,
    )
}

fn uncached_result(artifacts: BuiltArtifacts) -> CacheResult {
    CacheResult {
        cubin: artifacts.cubin,
        ltoir: artifacts.ltoir,
        immutable_cubin_path: None,
        immutable_ltoir_path: None,
        cache_hit: false,
    }
}

fn report_cache_result(result: &CacheResult) {
    if std::env::var_os("CUDA_OXIDE_VERBOSE").is_none() {
        return;
    }
    match (&result.immutable_cubin_path, result.cache_hit) {
        (Some(path), true) => eprintln!("cuda-oxide: cubin cache hit: {}", path.display()),
        (Some(path), false) => eprintln!("cuda-oxide: cached cubin: {}", path.display()),
        (None, _) => eprintln!("cuda-oxide: using an uncached cubin"),
    }
}

// ============================================================================
// Convenience: pick the right artifact and load it
// ============================================================================

/// Convenience wrapper: load a kernel module by `name` from the binary's
/// own directory, building the cubin on demand if cuda-oxide emitted NVVM IR.
///
/// Lookup order, inside `CARGO_MANIFEST_DIR` (the directory containing the
/// executable's `Cargo.toml`, where cuda-oxide writes its build artifacts):
///
/// 1. A target-recorded `<name>.ll` or `<name>.ltoir` -- `<name>.target`
///    identifies NVVM output and gives it precedence over older PTX.
/// 2. `<name>.ptx` -- the PTX path when no NVVM target file exists.
/// 3. An unrecorded `<name>.ll` / `<name>.ltoir` -- accepted only when
///    `CUDA_OXIDE_TARGET` explicitly supplies its original source target.
/// 4. `<name>.cubin` -- a standalone cubin when no source artifact exists.
///
/// If none of the four exist, returns [`LtoirError::NoArtifact`].
///
/// Use [`build_cubin_from_ll`] directly if you need explicit control over
/// the path or arch.
pub fn load_kernel_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, LtoirError> {
    let dir = manifest_dir();
    let cubin = dir.join(format!("{name}.cubin"));
    let ptx = dir.join(format!("{name}.ptx"));
    let ll = dir.join(format!("{name}.ll"));
    let ltoir = dir.join(format!("{name}.ltoir"));

    let has_recorded_nvvm_target =
        emitted_target_path(&ll)
            .try_exists()
            .map_err(|source| LtoirError::Io {
                path: emitted_target_path(&ll),
                source,
            })?;
    let selected = select_file_artifact(
        ptx.exists(),
        ll.exists(),
        ltoir.exists(),
        cubin.exists(),
        has_recorded_nvvm_target,
    )
    .ok_or_else(|| LtoirError::NoArtifact {
        name: name.to_string(),
        dir,
    })?;

    match selected {
        FileArtifact::Ptx => Ok(ctx.load_module_from_file(
            ptx.to_str()
                .expect("kernel artifact path is not valid UTF-8"),
        )?),
        FileArtifact::NvvmIr => {
            let emitted = target_arch_for_artifact(&ll)?;
            let execution = execution_arch_for_context(ctx)?;
            match execution_route(&emitted, &execution)? {
                ExecutionRoute::Cubin => {
                    let cubin = build_cubin_from_ll_file(&ll, &emitted)?;
                    Ok(ctx.load_module_from_image(&cubin.image)?)
                }
                ExecutionRoute::PtxBridge => {
                    let nvvm_ir = read_artifact(&ll)?;
                    let compile_options = read_compile_options(&ll)?;
                    let ptx = build_ptx_from_nvvm_ir_with_compile_options(
                        &nvvm_ir,
                        &ll.display().to_string(),
                        &emitted.sm(),
                        compile_options,
                    )?;
                    Ok(ctx.load_module_from_image(&ptx)?)
                }
            }
        }
        FileArtifact::Ltoir => {
            let emitted = target_arch_for_artifact(&ltoir)?;
            let execution = execution_arch_for_context(ctx)?;
            let bytes = read_artifact(&ltoir)?;
            let compile_options = read_compile_options(&ltoir)?;
            let image = match execution_route(&emitted, &execution)? {
                ExecutionRoute::Cubin => {
                    cached_ltoir_to_cubin_with_compile_options(
                        ltoir.parent().unwrap_or_else(|| Path::new(".")),
                        &bytes,
                        &ltoir.display().to_string(),
                        &emitted,
                        compile_options,
                    )?
                    .cubin
                }
                ExecutionRoute::PtxBridge => link_ltoir_to_ptx_parsed_with_compile_options(
                    &bytes,
                    &ltoir.display().to_string(),
                    &emitted,
                    compile_options,
                )?,
            };
            Ok(ctx.load_module_from_image(&image)?)
        }
        FileArtifact::Cubin => Ok(ctx.load_module_from_file(
            cubin
                .to_str()
                .expect("kernel artifact path is not valid UTF-8"),
        )?),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileArtifact {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

/// Select one file artifact. A `.target` file identifies NVVM IR/LTOIR as the
/// current output; without one, PTX takes precedence.
fn select_file_artifact(
    has_ptx: bool,
    has_nvvm_ir: bool,
    has_ltoir: bool,
    has_cubin: bool,
    has_recorded_nvvm_target: bool,
) -> Option<FileArtifact> {
    if has_recorded_nvvm_target {
        if has_nvvm_ir {
            return Some(FileArtifact::NvvmIr);
        }
        if has_ltoir {
            return Some(FileArtifact::Ltoir);
        }
        // The target file identifies NVVM output, but that output is missing.
        // Do not fall back to an older PTX or cubin.
        return None;
    }

    if has_ptx {
        Some(FileArtifact::Ptx)
    } else if has_nvvm_ir {
        // Older or manual NVVM IR requires an explicit CUDA_OXIDE_TARGET.
        Some(FileArtifact::NvvmIr)
    } else if has_ltoir {
        Some(FileArtifact::Ltoir)
    } else if has_cubin {
        Some(FileArtifact::Cubin)
    } else {
        None
    }
}

// ============================================================================
// Discovery helpers (libdevice, arch, manifest dir)
// ============================================================================

/// Locate `libdevice.10.bc` from the CUDA Toolkit.
///
/// Search order:
/// 1. `CUDA_OXIDE_LIBDEVICE` env var (used as-is if it points to an
///    existing file).
/// 2. `<root>/nvvm/libdevice/libdevice.10.bc` for `<root>` in
///    `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`,
///    `/opt/cuda`.
///
/// Returns [`LtoirError::LibdeviceNotFound`] with the full list of probed
/// paths if nothing matches.
///
/// Thin wrapper over [`cuda_artifact_finalizer::find_libdevice`], which owns the probe
/// (libdevice ships in the toolkit's `nvvm/` component next to `libnvvm.so`).
pub fn find_libdevice() -> Result<PathBuf, LtoirError> {
    cuda_artifact_finalizer::find_libdevice().map_err(
        |cuda_artifact_finalizer::LibdeviceNotFound { tried }| LtoirError::LibdeviceNotFound {
            tried,
        },
    )
}

/// Read and validate an explicit source target from `CUDA_OXIDE_TARGET`.
///
/// Artifact loaders prefer the target recorded with the artifact. Older
/// artifacts without a `.target` file require this explicit value because the
/// current GPU does not reveal which target was used to produce existing IR.
pub fn target_arch() -> Result<String, LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    resolve_source_target(None, explicit.as_deref()).map(|target| target.sm())
}

/// Query the physical execution GPU. Target-related environment variables
/// describe the input artifact, not the GPU that will execute it.
pub(crate) fn execution_arch_for_context(ctx: &CudaContext) -> Result<CudaArch, LtoirError> {
    let (major, minor) = ctx.compute_capability()?;
    format!("sm_{major}{minor}")
        .parse::<CudaArch>()
        .map_err(LtoirError::from)
}

/// Directory to search for kernel artifacts (`.cubin` / `.ptx` / `.ll` /
/// `.ltoir`).
///
/// Reads `CARGO_MANIFEST_DIR`, which `cargo run` sets to the directory of
/// the executable's `Cargo.toml` -- the same directory cuda-oxide writes
/// its build artifacts to. Falls back to the current working directory if
/// the env var is unset (e.g. when the binary is launched outside cargo).
///
/// Note: `env!("CARGO_MANIFEST_DIR")` cannot be used here because it
/// resolves to *this* crate's manifest dir at compile time, not the
/// downstream binary's.
fn manifest_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(d);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// ============================================================================
// Internal utilities
// ============================================================================

fn target_arch_for_artifact(artifact_path: &Path) -> Result<CudaArch, LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_artifact_with_explicit(artifact_path, explicit.as_deref())
}

fn target_arch_for_artifact_with_explicit(
    artifact_path: &Path,
    explicit_target: Option<&str>,
) -> Result<CudaArch, LtoirError> {
    resolve_source_target(read_emitted_target(artifact_path)?, explicit_target)
}

/// Resolve the original target from artifact metadata or an explicit value.
/// The execution GPU is not used to infer the artifact's target.
pub(crate) fn resolve_source_target(
    recorded_target: Option<CudaArch>,
    explicit_target: Option<&str>,
) -> Result<CudaArch, LtoirError> {
    if let Some(target) = recorded_target {
        return Ok(target);
    }
    explicit_target
        .ok_or(LtoirError::TargetNotFound)?
        .parse()
        .map_err(LtoirError::from)
}

fn emitted_target_path(ll_path: &Path) -> PathBuf {
    ll_path.with_extension("target")
}

fn emitted_compile_options_path(ll_path: &Path) -> PathBuf {
    ll_path.with_extension("options")
}

fn read_compile_options(ll_path: &Path) -> Result<ArtifactCompileOptions, LtoirError> {
    let path = emitted_compile_options_path(ll_path);
    match std::fs::read_to_string(&path) {
        Ok(value) => ArtifactCompileOptions::from_sidecar_text(&value).map_err(|error| {
            LtoirError::InvalidCompileOptions {
                path,
                value: error.to_string(),
            }
        }),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(ArtifactCompileOptions::new())
        }
        Err(source) => Err(LtoirError::Io { path, source }),
    }
}

fn read_emitted_target(ll_path: &Path) -> Result<Option<CudaArch>, LtoirError> {
    let path = emitted_target_path(ll_path);
    match std::fs::read_to_string(&path) {
        Ok(target) => {
            let mut lines = target.lines();
            let arch = lines.next().unwrap_or_default().parse()?;
            match (lines.next(), lines.next()) {
                (None, None) => Ok(Some(arch)),
                (Some(marker), None) if marker == COMPILE_OPTIONS_TARGET_MARKER => {
                    let options_path = emitted_compile_options_path(ll_path);
                    if !options_path.try_exists().map_err(|source| LtoirError::Io {
                        path: options_path.clone(),
                        source,
                    })? {
                        return Err(LtoirError::InvalidCompileOptions {
                            path: options_path,
                            value: "required sidecar is missing".to_string(),
                        });
                    }
                    read_compile_options(ll_path)?;
                    Ok(Some(arch))
                }
                _ => Err(LtoirError::InvalidCompileOptions {
                    path,
                    value: target.trim().to_string(),
                }),
            }
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(LtoirError::Io { path, source }),
    }
}

fn validate_ir_target_sidecar(ll_path: &Path, requested: &CudaArch) -> Result<(), LtoirError> {
    let Some(emitted) = read_emitted_target(ll_path)? else {
        return Ok(());
    };
    if emitted != *requested {
        return Err(LtoirError::TargetMismatch {
            ir_path: ll_path.to_path_buf(),
            emitted: emitted.sm(),
            requested: requested.sm(),
        });
    }
    Ok(())
}

fn write_artifact_target_sidecar(
    artifact_path: &Path,
    target: &CudaArch,
) -> Result<(), LtoirError> {
    let path = emitted_target_path(artifact_path);
    let options_path = emitted_compile_options_path(artifact_path);
    let contents = if options_path.try_exists().map_err(|source| LtoirError::Io {
        path: options_path.clone(),
        source,
    })? {
        read_compile_options(artifact_path)?;
        format!("{}\n{COMPILE_OPTIONS_TARGET_MARKER}\n", target.sm())
    } else {
        format!("{}\n", target.sm())
    };
    std::fs::write(&path, contents).map_err(|source| LtoirError::Io { path, source })
}

fn read_artifact(path: &Path) -> Result<Vec<u8>, LtoirError> {
    std::fs::read(path).map_err(|source| LtoirError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionRoute {
    /// Produce native code for the architecture that emitted the IR.
    Cubin,
    /// Keep the original virtual architecture in PTX and let the driver JIT it
    /// for the newer GPU.
    PtxBridge,
}

/// Select an execution route without changing either architecture.
pub(crate) fn execution_route(
    emitted: &CudaArch,
    execution: &CudaArch,
) -> Result<ExecutionRoute, LtoirError> {
    if emitted.capability() == execution.capability() {
        return Ok(ExecutionRoute::Cubin);
    }

    let incompatible = |reason| LtoirError::IncompatibleExecutionTarget {
        emitted: emitted.sm(),
        execution: execution.sm(),
        reason,
    };

    if emitted.suffix().is_some() {
        return Err(incompatible(
            "targets with an architecture suffix, such as sm_90a, cannot be forwarded to a different GPU",
        ));
    }
    if emitted.capability() > execution.capability() {
        return Err(incompatible(
            "an artifact built for a newer GPU cannot run on an older GPU",
        ));
    }
    // NVIDIA guarantees cubin binary compatibility within one compute
    // capability major version when the execution minor is at least the
    // emitted minor. The backward case was rejected immediately above.
    if emitted.capability() / 10 == execution.capability() / 10 {
        return Ok(ExecutionRoute::Cubin);
    }
    if emitted.uses_legacy_llvm() && execution.capability() >= 100 {
        return Ok(ExecutionRoute::PtxBridge);
    }

    Err(incompatible(
        "only standard pre-Blackwell targets, such as sm_86, can be converted to PTX for Blackwell",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cuda_oxide_ltoir_{name}_{}_{}",
            std::process::id(),
            unique
        ))
    }

    #[test]
    fn recorded_target_takes_precedence_over_environment() {
        let dir = temp_dir("target_sidecar");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; test\n").unwrap();
        std::fs::write(dir.join("kernel.target"), "compute_86\n").unwrap();

        let emitted = read_emitted_target(&ll).unwrap().unwrap();
        assert_eq!(emitted.sm(), "sm_86");
        assert_eq!(
            target_arch_for_artifact_with_explicit(&ll, Some("sm_120"))
                .unwrap()
                .sm(),
            "sm_86",
            "the recorded target must take precedence"
        );
        validate_ir_target_sidecar(&ll, &"sm_86".parse().unwrap()).unwrap();

        let mismatch = validate_ir_target_sidecar(&ll, &"sm_120".parse().unwrap())
            .expect_err("cross-target reuse must fail");
        assert!(matches!(mismatch, LtoirError::TargetMismatch { .. }));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn versioned_target_accepts_v1_and_v2_compile_options() {
        let dir = temp_dir("versioned_compile_options");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; test\n").unwrap();
        std::fs::write(
            dir.join("kernel.target"),
            format!("sm_86\n{COMPILE_OPTIONS_TARGET_MARKER}\n"),
        )
        .unwrap();

        assert!(matches!(
            read_emitted_target(&ll),
            Err(LtoirError::InvalidCompileOptions { .. })
        ));

        let v1_options = ArtifactCompileOptions::new().with_fma_contraction(false);
        assert!(
            v1_options
                .sidecar_text()
                .starts_with("cuda-oxide-compile-options-v1\n")
        );
        std::fs::write(dir.join("kernel.options"), v1_options.sidecar_text()).unwrap();
        assert_eq!(read_emitted_target(&ll).unwrap().unwrap().sm(), "sm_86");
        assert_eq!(read_compile_options(&ll).unwrap(), v1_options);

        let line_tables =
            ArtifactCompileOptions::new().with_debug_policy(ArtifactDebugPolicy::LineTables);
        assert!(
            line_tables
                .sidecar_text()
                .starts_with("cuda-oxide-compile-options-v2\n")
        );
        std::fs::write(dir.join("kernel.options"), line_tables.sidecar_text()).unwrap();
        assert_eq!(read_emitted_target(&ll).unwrap().unwrap().sm(), "sm_86");
        assert_eq!(read_compile_options(&ll).unwrap(), line_tables);

        let full_debug = ArtifactCompileOptions::new()
            .with_fma_contraction(false)
            .with_debug_policy(ArtifactDebugPolicy::Full);
        std::fs::write(dir.join("kernel.options"), full_debug.sidecar_text()).unwrap();
        assert_eq!(read_emitted_target(&ll).unwrap().unwrap().sm(), "sm_86");
        assert_eq!(read_compile_options(&ll).unwrap(), full_debug);

        std::fs::write(dir.join("kernel.options"), "fma-contraction=off\n").unwrap();
        assert!(matches!(
            read_emitted_target(&ll),
            Err(LtoirError::InvalidCompileOptions { .. })
        ));

        std::fs::remove_file(dir.join("kernel.options")).unwrap();
        std::fs::write(dir.join("kernel.target"), "sm_86\n").unwrap();
        assert_eq!(read_emitted_target(&ll).unwrap().unwrap().sm(), "sm_86");
        assert_eq!(
            read_compile_options(&ll).unwrap(),
            ArtifactCompileOptions::new(),
            "legacy artifacts default to FMA enabled with no debug information"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn finalization_preserves_fma_and_debug_policy() {
        let arch: CudaArch = "sm_90a".parse().unwrap();
        for allow_fma in [false, true] {
            for (artifact_debug, finalizer_debug) in [
                (
                    ArtifactDebugPolicy::None,
                    cuda_artifact_finalizer::DebugPolicy::None,
                ),
                (
                    ArtifactDebugPolicy::LineTables,
                    cuda_artifact_finalizer::DebugPolicy::LineTables,
                ),
                (
                    ArtifactDebugPolicy::Full,
                    cuda_artifact_finalizer::DebugPolicy::Full,
                ),
            ] {
                let artifact_options = ArtifactCompileOptions::new()
                    .with_fma_contraction(allow_fma)
                    .with_debug_policy(artifact_debug);
                let finalizer_options = finalization_options(&arch, artifact_options);

                assert_eq!(finalizer_options.target(), &arch);
                assert_eq!(finalizer_options.allow_fma_contraction(), allow_fma);
                assert_eq!(finalizer_options.debug_policy(), finalizer_debug);
            }
        }
    }

    #[test]
    fn artifact_without_target_requires_explicit_target() {
        let dir = temp_dir("missing_target");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; target-specific NVVM IR\n").unwrap();

        let error = target_arch_for_artifact_with_explicit(&ll, None)
            .expect_err("the original build target is required");
        assert!(matches!(error, LtoirError::TargetNotFound));

        let asserted = target_arch_for_artifact_with_explicit(&ll, Some("compute_86")).unwrap();
        assert_eq!(asserted.sm(), "sm_86");

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_build_target_is_recorded_for_sibling_ltoir() {
        let dir = temp_dir("record_explicit_target");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; manually supplied NVVM IR\n").unwrap();
        let arch: CudaArch = "compute_86".parse().unwrap();

        validate_ir_target_sidecar(&ll, &arch).unwrap();
        write_artifact_target_sidecar(&ll, &arch).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("kernel.target")).unwrap(),
            "sm_86\n"
        );
        let ltoir = dir.join("kernel.ltoir");
        std::fs::write(&ltoir, b"placeholder").unwrap();
        std::fs::remove_file(&ll).unwrap();
        assert_eq!(
            target_arch_for_artifact_with_explicit(&ltoir, None)
                .unwrap()
                .sm(),
            "sm_86"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn recorded_nvvm_output_takes_precedence_over_stale_ptx() {
        assert_eq!(
            select_file_artifact(true, true, false, true, true),
            Some(FileArtifact::NvvmIr)
        );
        assert_eq!(
            select_file_artifact(true, false, true, true, true),
            Some(FileArtifact::Ltoir)
        );
        assert_eq!(
            select_file_artifact(true, false, false, true, true),
            None,
            "a missing NVVM artifact must not fall back to older output"
        );

        // In an ordinary PTX build there is also an LLVM `.ll`, but no NVVM
        // target sidecar. PTX is therefore the current loadable artifact.
        assert_eq!(
            select_file_artifact(true, true, false, true, false),
            Some(FileArtifact::Ptx)
        );
    }

    #[test]
    fn file_artifact_selection_accepts_older_unrecorded_nvvm_output() {
        assert_eq!(
            select_file_artifact(false, true, false, true, false),
            Some(FileArtifact::NvvmIr)
        );
        assert_eq!(
            select_file_artifact(false, false, true, true, false),
            Some(FileArtifact::Ltoir)
        );
        assert_eq!(
            select_file_artifact(false, false, false, true, false),
            Some(FileArtifact::Cubin)
        );
        assert_eq!(
            select_file_artifact(false, false, false, false, false),
            None
        );
    }

    #[test]
    fn missing_ll_cannot_reuse_old_cubin_or_publish_target() {
        let dir = temp_dir("missing_ll_old_cubin");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(dir.join("kernel.cubin"), b"old cubin").unwrap();

        let error = build_cubin_from_ll(&ll, "sm_86")
            .expect_err("a derived cubin is invalid without its source NVVM IR");
        assert!(matches!(error, LtoirError::Io { path, .. } if path == ll));
        assert!(
            !dir.join("kernel.target").exists(),
            "a missing source must not gain target metadata"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_link_target_fails_before_loading_nvidia_libraries() {
        let error = link_ltoir_to_cubin(&[], "empty", "nvvm-ir").unwrap_err();
        assert!(matches!(error, LtoirError::InvalidTarget(_)));

        let error = link_ltoir_to_ptx(&[], "empty", "nvvm-ir").unwrap_err();
        assert!(matches!(error, LtoirError::InvalidTarget(_)));
    }

    #[test]
    fn same_target_keeps_native_cubin_route() {
        for (emitted, execution) in [
            ("sm_86", "sm_86"),
            ("compute_120", "sm_120"),
            // The suffix is preserved in the nvJitLink target. The CUDA
            // driver remains responsible for accepting it on this same
            // numeric device architecture.
            ("sm_90a", "sm_90"),
        ] {
            assert_eq!(
                execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap()).unwrap(),
                ExecutionRoute::Cubin,
                "{emitted} on {execution}"
            );
        }
    }

    #[test]
    fn same_major_forward_minor_uses_binary_compatible_cubin() {
        for (emitted, execution) in [
            ("sm_80", "sm_86"),
            ("compute_86", "sm_89"),
            ("sm_120", "sm_121"),
        ] {
            assert_eq!(
                execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap()).unwrap(),
                ExecutionRoute::Cubin,
                "{emitted} on {execution}"
            );
        }
    }

    #[test]
    fn standard_legacy_target_bridges_forward_to_blackwell_as_ptx() {
        for (emitted, execution) in [
            ("sm_75", "sm_100"),
            ("compute_86", "sm_120"),
            ("sm_90", "sm_120"),
        ] {
            assert_eq!(
                execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap()).unwrap(),
                ExecutionRoute::PtxBridge,
                "{emitted} on {execution}"
            );
        }
    }

    #[test]
    fn ptx_bridge_rejects_suffixed_and_unsupported_forward_targets() {
        for (emitted, execution) in [
            ("sm_90a", "sm_120"),
            ("sm_90f", "sm_120"),
            ("sm_86", "sm_90"),
            ("sm_100", "sm_120"),
        ] {
            let error = execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap())
                .expect_err("cross-target route must be explicitly supported");
            assert!(
                matches!(error, LtoirError::IncompatibleExecutionTarget { .. }),
                "{emitted} on {execution}: {error}"
            );
        }
    }

    #[test]
    fn execution_route_rejects_backward_targets() {
        for (emitted_target, execution_target) in [("sm_89", "sm_86"), ("sm_120", "sm_86")] {
            let error = execution_route(
                &emitted_target.parse().unwrap(),
                &execution_target.parse().unwrap(),
            )
            .expect_err("a newer artifact cannot run on an older GPU");

            let LtoirError::IncompatibleExecutionTarget {
                emitted,
                execution,
                reason,
            } = error
            else {
                panic!("unexpected error: {error}");
            };
            assert_eq!(emitted, emitted_target);
            assert_eq!(execution, execution_target);
            assert!(reason.contains("newer GPU"));
        }
    }

    #[test]
    fn nvvm_cubin_cache_key_tracks_every_external_input() {
        let arch: CudaArch = "sm_86".parse().unwrap();
        let nvvm = [1_u8; 32];
        let nvjitlink = [2_u8; 32];
        let original = nvvm_ir_cubin_cache_key(
            b"nvvm ir",
            "kernel.ll",
            "kernel.ltoir",
            &arch,
            b"libdevice",
            &nvvm,
            &nvjitlink,
        );

        let changed = [
            nvvm_ir_cubin_cache_key(
                b"different nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &arch,
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "renamed.ll",
                "kernel.ltoir",
                &arch,
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "renamed.ltoir",
                &arch,
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &"sm_90".parse().unwrap(),
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &arch,
                b"different libdevice",
                &nvvm,
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &arch,
                b"libdevice",
                &[3_u8; 32],
                &nvjitlink,
            ),
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &arch,
                b"libdevice",
                &nvvm,
                &[4_u8; 32],
            ),
        ];
        assert!(changed.into_iter().all(|key| key != original));

        assert_eq!(
            original,
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &"compute_86".parse().unwrap(),
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            "equivalent target spellings must normalize to one key"
        );
        assert_ne!(
            original,
            nvvm_ir_cubin_cache_key(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &"sm_86a".parse().unwrap(),
                b"libdevice",
                &nvvm,
                &nvjitlink,
            ),
            "architecture suffixes must remain part of the key"
        );
        assert_ne!(
            original,
            nvvm_ir_cubin_cache_key_with_options(
                b"nvvm ir",
                "kernel.ll",
                "kernel.ltoir",
                &arch,
                b"libdevice",
                (&nvvm, &nvjitlink),
                false,
            ),
            "FMA policy changes both libNVVM and nvJitLink output"
        );

        let line_tables = nvvm_ir_cubin_cache_key_with_compile_options(
            b"nvvm ir",
            "kernel.ll",
            "kernel.ltoir",
            &arch,
            b"libdevice",
            (&nvvm, &nvjitlink),
            ArtifactCompileOptions::new().with_debug_policy(ArtifactDebugPolicy::LineTables),
        );
        let full_debug = nvvm_ir_cubin_cache_key_with_compile_options(
            b"nvvm ir",
            "kernel.ll",
            "kernel.ltoir",
            &arch,
            b"libdevice",
            (&nvvm, &nvjitlink),
            ArtifactCompileOptions::new().with_debug_policy(ArtifactDebugPolicy::Full),
        );
        assert_ne!(original, line_tables, "line information changes the key");
        assert_ne!(original, full_debug, "full debug changes the key");
        assert_ne!(
            line_tables, full_debug,
            "line tables and full debug are distinct compiler policies"
        );
    }

    #[test]
    fn standalone_ltoir_cache_key_is_separate_and_complete() {
        let arch: CudaArch = "sm_86".parse().unwrap();
        let nvjitlink = [5_u8; 32];
        let original = ltoir_cubin_cache_key(b"ltoir", "kernel.ltoir", &arch, &nvjitlink);

        assert_ne!(
            original,
            ltoir_cubin_cache_key(b"different ltoir", "kernel.ltoir", &arch, &nvjitlink)
        );
        assert_ne!(
            original,
            ltoir_cubin_cache_key(b"ltoir", "renamed.ltoir", &arch, &nvjitlink)
        );
        assert_ne!(
            original,
            ltoir_cubin_cache_key(
                b"ltoir",
                "kernel.ltoir",
                &"sm_90".parse().unwrap(),
                &nvjitlink,
            )
        );
        assert_ne!(
            original,
            ltoir_cubin_cache_key(b"ltoir", "kernel.ltoir", &arch, &[6_u8; 32])
        );
        assert_ne!(
            original,
            ltoir_cubin_cache_key_with_options(b"ltoir", "kernel.ltoir", &arch, &nvjitlink, false,),
            "FMA policy changes nvJitLink LTO output"
        );
        let line_tables = ltoir_cubin_cache_key_with_compile_options(
            b"ltoir",
            "kernel.ltoir",
            &arch,
            &nvjitlink,
            ArtifactCompileOptions::new().with_debug_policy(ArtifactDebugPolicy::LineTables),
        );
        let full_debug = ltoir_cubin_cache_key_with_compile_options(
            b"ltoir",
            "kernel.ltoir",
            &arch,
            &nvjitlink,
            ArtifactCompileOptions::new().with_debug_policy(ArtifactDebugPolicy::Full),
        );
        assert_ne!(original, line_tables, "line information changes the key");
        assert_ne!(original, full_debug, "full debug changes the key");
        assert_ne!(
            line_tables, full_debug,
            "line tables and full debug are distinct linker policies"
        );
        assert_ne!(
            original,
            nvvm_ir_cubin_cache_key(
                b"ltoir",
                "kernel.ltoir",
                "kernel.ltoir",
                &arch,
                b"",
                &[0_u8; 32],
                &nvjitlink,
            ),
            "NVVM IR and standalone LTOIR routes must never share a key"
        );
    }

    #[test]
    #[ignore = "requires CUDA Toolkit libraries discoverable by concrete paths"]
    fn live_file_cache_hits_and_source_changes_miss() {
        const LEGACY_NVVM_IR: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void ()* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

        let dir = temp_dir("live_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, LEGACY_NVVM_IR).unwrap();

        let first = build_cubin_from_ll(&ll, "sm_86").unwrap();
        let first_bytes = std::fs::read(&first).unwrap();
        assert!(first_bytes.starts_with(b"\x7fELF"));
        assert_eq!(first, dir.join("kernel.cubin"));

        let finalizer = Finalizer::discover().unwrap();
        assert!(finalizer.provenance_digest().is_some());

        let second = build_cubin_from_ll(&ll, "compute_86").unwrap();
        assert_eq!(second, first, "normalized target spelling should hit");
        assert_eq!(std::fs::read(&second).unwrap(), first_bytes);

        let arch: CudaArch = "sm_86".parse().unwrap();
        let ll_bytes = std::fs::read(&ll).unwrap();
        let ltoir_path = dir.join("kernel.ltoir");
        let native_hit = cached_nvvm_ir_to_cubin(
            &dir,
            &ll_bytes,
            &ll.display().to_string(),
            &ltoir_path.display().to_string(),
            &arch,
        )
        .unwrap();
        assert!(native_hit.cache_hit);
        let native_cache_path = native_hit.immutable_cubin_path.unwrap();
        let cache_modified = std::fs::metadata(&native_cache_path)
            .unwrap()
            .modified()
            .unwrap();
        let native_hit_again = cached_nvvm_ir_to_cubin(
            &dir,
            &ll_bytes,
            &ll.display().to_string(),
            &ltoir_path.display().to_string(),
            &arch,
        )
        .unwrap();
        assert!(native_hit_again.cache_hit);
        assert_eq!(
            native_hit_again.immutable_cubin_path.as_ref(),
            Some(&native_cache_path)
        );
        assert_eq!(
            std::fs::metadata(&native_cache_path)
                .unwrap()
                .modified()
                .unwrap(),
            cache_modified,
            "a cache hit must not rewrite the immutable entry"
        );

        let ltoir = std::fs::read(&ltoir_path).unwrap();
        let standalone_dir = dir.join("standalone");
        std::fs::create_dir(&standalone_dir).unwrap();
        let standalone_first =
            cached_ltoir_to_cubin(&standalone_dir, &ltoir, "kernel.ltoir", &arch).unwrap();
        assert!(!standalone_first.cache_hit);
        assert_eq!(standalone_first.ltoir, None);
        let standalone_second =
            cached_ltoir_to_cubin(&standalone_dir, &ltoir, "kernel.ltoir", &arch).unwrap();
        assert!(standalone_second.cache_hit);
        assert_eq!(standalone_second.cubin, standalone_first.cubin);
        assert_eq!(
            standalone_second.immutable_cubin_path,
            standalone_first.immutable_cubin_path
        );

        let mut changed_ir = LEGACY_NVVM_IR.to_vec();
        changed_ir.extend_from_slice(b"\n; source changed\n");
        std::fs::write(&ll, &changed_ir).unwrap();
        let third = build_cubin_from_ll(&ll, "sm_86").unwrap();
        assert_eq!(third, first, "the public sibling path is stable");
        assert!(std::fs::read(&third).unwrap().starts_with(b"\x7fELF"));
        let changed_hit = cached_nvvm_ir_to_cubin(
            &dir,
            &changed_ir,
            &ll.display().to_string(),
            &ltoir_path.display().to_string(),
            &arch,
        )
        .unwrap();
        assert!(changed_hit.cache_hit);
        assert_ne!(changed_hit.immutable_cubin_path, Some(native_cache_path));

        std::fs::remove_dir_all(dir).unwrap();
    }
}
