/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime-loaded bindings for the CUTLASS 4.7 compiler C API.
//!
//! Compile textual `PreCompiledMlir` through a pinned CuTe pipeline to the
//! C API's `ObjectArtifact`. The result is a host ELF object containing the
//! CUDA image; the backend extracts that image before the driver can load it.
//! These wrappers release compiler and artifact handles with the matching
//! C API destructors.

use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_int, c_void};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::str::FromStr;
use thiserror::Error;

const CUTLASS_ARTIFACT_PRE_COMPILED_MLIR: c_int = 1;
const CUTLASS_ARTIFACT_OBJECT: c_int = 4;
// CUTLASS 4.7's two concrete wrapper ABIs require FunctionMetadata, which the
// `gpu.module` external-module path deliberately does not carry. The
// fingerprint-pinned 4.7 library accepts TBD as a no-wrapper passthrough for
// this path. We assert below that the resulting ObjectArtifact still carries
// zero FunctionMetadata entries so a future change cannot silently introduce
// a host wrapper or reinterpret the direct CUDA kernel ABI.
const CUTLASS_ABI_TBD_EXTERNAL_MODULE_PASSTHROUGH: c_int = 0;

type CompilerRef = *mut c_void;
type ArtifactsRef = *mut c_void;

type CompilerCreateFn = unsafe extern "C" fn() -> CompilerRef;
type CompilerDestroyFn = unsafe extern "C" fn(CompilerRef);
type CompilerSetDeviceTargetFn =
    unsafe extern "C" fn(CompilerRef, *const c_char, usize, *mut *mut c_char, *mut usize) -> c_int;
type CompilerSetAbiFn =
    unsafe extern "C" fn(CompilerRef, c_int, *mut *mut c_char, *mut usize) -> c_int;
type CompilerSetPipelineFn =
    unsafe extern "C" fn(CompilerRef, c_int, *const c_char, usize) -> c_int;
type CompilerCompileToFn = unsafe extern "C" fn(
    CompilerRef,
    ArtifactsRef,
    c_int,
    *mut *mut c_char,
    *mut usize,
) -> ArtifactsRef;
type ArtifactsFromTextFn = unsafe extern "C" fn(c_int, *const c_char, usize) -> ArtifactsRef;
type ArtifactsGetTypeFn = unsafe extern "C" fn(ArtifactsRef) -> c_int;
type ArtifactsGetDataFn = unsafe extern "C" fn(ArtifactsRef, *mut *mut u8, *mut usize) -> c_int;
type ArtifactsFunctionCountFn = unsafe extern "C" fn(ArtifactsRef, *mut usize) -> c_int;
type ArtifactsDestroyFn = unsafe extern "C" fn(ArtifactsRef);

unsafe extern "C" {
    fn free(ptr: *mut c_void);
}

/// A validated SHA-256 digest used to pin the proprietary compiler library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sha256Fingerprint([u8; 32]);

impl Sha256Fingerprint {
    /// Return the raw 32-byte digest.
    #[cfg(test)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Sha256Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for Sha256Fingerprint {
    type Err = FingerprintParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64 {
            return Err(FingerprintParseError::Length {
                actual: value.len(),
            });
        }

        let mut digest = [0_u8; 32];
        let encoded = value.as_bytes();
        for (index, byte) in digest.iter_mut().enumerate() {
            let offset = index * 2;
            let high =
                hex_nibble(encoded[offset]).ok_or(FingerprintParseError::NonHex { offset })?;
            let low = hex_nibble(encoded[offset + 1])
                .ok_or(FingerprintParseError::NonHex { offset: offset + 1 })?;
            *byte = (high << 4) | low;
        }
        Ok(Self(digest))
    }
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// A malformed SHA-256 fingerprint supplied for compiler provenance.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FingerprintParseError {
    /// A SHA-256 hexadecimal digest must contain exactly 64 characters.
    #[error("SHA-256 fingerprint must contain 64 hexadecimal characters, got {actual}")]
    Length {
        /// Number of characters supplied by the caller.
        actual: usize,
    },
    /// The digest contains a character outside the hexadecimal alphabet.
    #[error("SHA-256 fingerprint contains a non-hexadecimal byte at offset {offset}")]
    NonHex {
        /// Byte offset of the invalid two-character group.
        offset: usize,
    },
}

/// Errors produced while loading or calling the CUTLASS compiler C API.
#[derive(Debug, Error)]
pub enum CutlassCompilerError {
    /// The expected compiler-library fingerprint is malformed.
    #[error(transparent)]
    InvalidFingerprint(#[from] FingerprintParseError),

    /// Runtime loading only accepts an explicit absolute path.
    #[error("CUTLASS compiler library path must be absolute: `{path}`")]
    RelativeLibraryPath {
        /// Rejected path.
        path: PathBuf,
    },

    /// The path does not name a regular file (symlinks are rejected).
    #[error("CUTLASS compiler library path is not a regular file: `{path}`")]
    LibraryNotRegular {
        /// Rejected path.
        path: PathBuf,
    },

    /// Filesystem inspection of the library failed.
    #[error("could not inspect CUTLASS compiler library `{path}`: {source}")]
    LibraryMetadata {
        /// Inspected path.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// Opening the validated library file failed.
    #[error("could not open CUTLASS compiler library `{path}`: {source}")]
    LibraryOpen {
        /// Opened path.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// Reading the library to calculate its digest failed.
    #[error("could not fingerprint CUTLASS compiler library `{path}`: {source}")]
    LibraryRead {
        /// Read path.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// The library's actual digest did not match the pinned provenance.
    #[error(
        "CUTLASS compiler library fingerprint mismatch for `{path}`: expected {expected}, got {actual}"
    )]
    FingerprintMismatch {
        /// Validated path.
        path: PathBuf,
        /// Pinned digest.
        expected: Sha256Fingerprint,
        /// Digest calculated from the opened file.
        actual: Sha256Fingerprint,
    },

    /// `dlopen` rejected the validated library.
    #[error("could not load CUTLASS compiler library `{path}`: {source}")]
    LibraryLoad {
        /// Loaded path.
        path: PathBuf,
        /// Dynamic-loader error.
        #[source]
        source: libloading::Error,
    },

    /// A required 4.7 C API symbol is absent.
    #[error("CUTLASS compiler library is missing required symbol `{symbol}`: {source}")]
    SymbolNotFound {
        /// Required symbol name.
        symbol: &'static str,
        /// Dynamic-loader error.
        #[source]
        source: libloading::Error,
    },

    /// A C API constructor unexpectedly returned a null handle.
    #[error("CUTLASS C API `{operation}` returned a null handle")]
    NullHandle {
        /// C function that returned null.
        operation: &'static str,
    },

    /// Local input validation rejected the compile request.
    #[error("invalid CUTLASS compile input: {reason}")]
    InvalidInput {
        /// Human-readable validation failure.
        reason: &'static str,
    },

    /// Compiler configuration failed.
    #[error("CUTLASS C API `{operation}` failed with status {status}: {message}")]
    Configure {
        /// C function that failed.
        operation: &'static str,
        /// Raw C status.
        status: c_int,
        /// C-allocated diagnostic, when available.
        message: String,
    },

    /// The C API rejected a source-stage pass pipeline.
    #[error(
        "CUTLASS C API `cutlass_compiler_set_pipeline` rejected artifact type {artifact_type} with status {status}"
    )]
    PipelineConfiguration {
        /// Source artifact type associated with the pipeline.
        artifact_type: c_int,
        /// Raw C status.
        status: c_int,
    },

    /// The C API could not parse the supplied textual MLIR.
    #[error("CUTLASS could not create a {artifact_kind} artifact from textual MLIR")]
    MlirParse {
        /// Human-readable C API artifact kind.
        artifact_kind: &'static str,
    },

    /// The multi-stage compile operation failed.
    #[error("CUTLASS C API `cutlass_compiler_compile_to` failed: {message}")]
    Compile {
        /// C-allocated diagnostic, when available.
        message: String,
    },

    /// An artifact query failed or violated its documented pointer contract.
    #[error("CUTLASS C API `{operation}` failed: {reason}")]
    ArtifactQuery {
        /// C function that failed.
        operation: &'static str,
        /// Failure detail.
        reason: String,
    },

    /// The returned `ObjectArtifact` was not an ELF object as documented.
    #[error("CUTLASS ObjectArtifact is not an ELF object")]
    NotElfObject,

    /// The pinned external-module passthrough unexpectedly gained wrapper
    /// metadata, so its direct CUDA-kernel ABI can no longer be assumed.
    #[error(
        "CUTLASS 4.7 external-module passthrough returned {count} FunctionMetadata entries; expected zero"
    )]
    UnexpectedFunctionMetadata {
        /// Number of entries reported by the post-object C API.
        count: usize,
    },
}

/// Output of the official CompiledMlir-to-Object compilation path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutlassObject {
    /// Raw ELF bytes returned by `cutlass_artifacts_get_data`.
    pub elf: Vec<u8>,
}

/// A loaded and SHA-pinned CUTLASS 4.7 compiler library.
pub struct CutlassCompilerLibrary {
    api: Api,
}

impl CutlassCompilerLibrary {
    /// Validate and load an official compiler library from an absolute path.
    ///
    /// The path's final component must be a regular file rather than a
    /// symlink. Its SHA-256 digest must exactly match `expected_sha256`. On
    /// Linux, `dlopen` is performed through the already-open file descriptor,
    /// so replacing the pathname after validation cannot redirect the load to
    /// a different inode.
    pub fn load(
        path: impl AsRef<Path>,
        expected_sha256: &str,
    ) -> Result<Self, CutlassCompilerError> {
        let expected = expected_sha256.parse()?;
        let validated = validate_library_file(path.as_ref(), expected)?;
        let api = Api::load(validated)?;
        Ok(Self { api })
    }

    /// Return the absolute path used for provenance validation.
    pub fn path(&self) -> &Path {
        &self.api.path
    }

    /// Return the SHA-256 digest calculated before the library was loaded.
    pub fn fingerprint(&self) -> Sha256Fingerprint {
        self.api.fingerprint
    }

    /// Compile textual `PreCompiledMlir` to a host ELF object.
    ///
    /// `precompiled_pipeline` replaces the library's first-stage CuTe pass
    /// pipeline. The caller must pin it together with the compiler-library
    /// digest: it is part of the executable compiler contract, not a user
    /// supplied optimization knob. The returned object contains the CUDA
    /// image selected by that pipeline; extracting and validating that image
    /// is deliberately handled by the higher-level backend boundary.
    pub fn compile_precompiled_mlir_to_object(
        &self,
        mlir: &str,
        device_target: &str,
        precompiled_pipeline: &str,
    ) -> Result<CutlassObject, CutlassCompilerError> {
        validate_compile_input(mlir, device_target)?;
        if precompiled_pipeline.is_empty() {
            return Err(CutlassCompilerError::InvalidInput {
                reason: "PreCompiledMlir pipeline must not be empty",
            });
        }
        if precompiled_pipeline.as_bytes().contains(&0) {
            return Err(CutlassCompilerError::InvalidInput {
                reason: "PreCompiledMlir pipeline must not contain NUL bytes",
            });
        }

        let compiler = CompilerHandle::create_cute(&self.api)?;
        compiler.set_device_target(device_target)?;
        compiler.set_external_module_passthrough_abi()?;
        compiler.set_pipeline(CUTLASS_ARTIFACT_PRE_COMPILED_MLIR, precompiled_pipeline)?;

        let mut input = ArtifactsHandle::from_textual_precompiled_mlir(&self.api, mlir)?;
        compile_object(compiler, &mut input)
    }
}

fn validate_compile_input(mlir: &str, device_target: &str) -> Result<(), CutlassCompilerError> {
    if mlir.is_empty() {
        return Err(CutlassCompilerError::InvalidInput {
            reason: "textual MLIR must not be empty",
        });
    }
    if mlir.as_bytes().contains(&0) {
        return Err(CutlassCompilerError::InvalidInput {
            reason: "textual MLIR must not contain NUL bytes",
        });
    }
    if device_target.is_empty() {
        return Err(CutlassCompilerError::InvalidInput {
            reason: "device target must not be empty",
        });
    }
    if device_target.as_bytes().contains(&0) {
        return Err(CutlassCompilerError::InvalidInput {
            reason: "device target must not contain NUL bytes",
        });
    }
    Ok(())
}

fn compile_object<'api>(
    compiler: CompilerHandle<'api>,
    input: &mut ArtifactsHandle<'api>,
) -> Result<CutlassObject, CutlassCompilerError> {
    let object = compiler.compile_to_object(input)?;
    if object.artifact_type() != CUTLASS_ARTIFACT_OBJECT {
        return Err(CutlassCompilerError::ArtifactQuery {
            operation: "cutlass_artifacts_get_type",
            reason: "compile returned an artifact whose type is not Object".to_string(),
        });
    }
    let function_metadata_count = object.function_count()?;
    if function_metadata_count != 0 {
        return Err(CutlassCompilerError::UnexpectedFunctionMetadata {
            count: function_metadata_count,
        });
    }

    let elf = object.data()?;
    if !elf.starts_with(b"\x7fELF") {
        return Err(CutlassCompilerError::NotElfObject);
    }
    Ok(CutlassObject { elf })
}

struct ValidatedLibraryFile {
    path: PathBuf,
    file: File,
    fingerprint: Sha256Fingerprint,
}

fn validate_library_file(
    path: &Path,
    expected: Sha256Fingerprint,
) -> Result<ValidatedLibraryFile, CutlassCompilerError> {
    if !path.is_absolute() {
        return Err(CutlassCompilerError::RelativeLibraryPath {
            path: path.to_path_buf(),
        });
    }

    let metadata =
        fs::symlink_metadata(path).map_err(|source| CutlassCompilerError::LibraryMetadata {
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.file_type().is_file() {
        return Err(CutlassCompilerError::LibraryNotRegular {
            path: path.to_path_buf(),
        });
    }

    let mut file = File::open(path).map_err(|source| CutlassCompilerError::LibraryOpen {
        path: path.to_path_buf(),
        source,
    })?;
    let opened_metadata =
        file.metadata()
            .map_err(|source| CutlassCompilerError::LibraryMetadata {
                path: path.to_path_buf(),
                source,
            })?;
    if !opened_metadata.is_file() {
        return Err(CutlassCompilerError::LibraryNotRegular {
            path: path.to_path_buf(),
        });
    }

    let actual =
        fingerprint_reader(&mut file).map_err(|source| CutlassCompilerError::LibraryRead {
            path: path.to_path_buf(),
            source,
        })?;
    if actual != expected {
        return Err(CutlassCompilerError::FingerprintMismatch {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }

    Ok(ValidatedLibraryFile {
        path: path.to_path_buf(),
        file,
        fingerprint: actual,
    })
}

fn fingerprint_reader(reader: &mut impl Read) -> Result<Sha256Fingerprint, std::io::Error> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(Sha256Fingerprint(hasher.finalize().into()))
}

struct Api {
    // Keep the dynamic-library handle alive for every resolved function
    // pointer. On Linux, keep the opened descriptor alive too so provenance
    // continues to name the loaded inode even if its pathname is replaced.
    _library: Library,
    _loaded_file: File,
    path: PathBuf,
    fingerprint: Sha256Fingerprint,
    cute_compiler_create: CompilerCreateFn,
    compiler_destroy: CompilerDestroyFn,
    compiler_set_device_target: CompilerSetDeviceTargetFn,
    compiler_set_abi: CompilerSetAbiFn,
    compiler_set_pipeline: CompilerSetPipelineFn,
    compiler_compile_to: CompilerCompileToFn,
    artifacts_from_textual_form: ArtifactsFromTextFn,
    artifacts_get_type: ArtifactsGetTypeFn,
    artifacts_get_data: ArtifactsGetDataFn,
    artifacts_function_count: ArtifactsFunctionCountFn,
    artifacts_destroy: ArtifactsDestroyFn,
}

impl Api {
    fn load(validated: ValidatedLibraryFile) -> Result<Self, CutlassCompilerError> {
        #[cfg(target_os = "linux")]
        let load_path = PathBuf::from(format!("/proc/self/fd/{}", validated.file.as_raw_fd()));
        #[cfg(not(target_os = "linux"))]
        let load_path = validated.path.clone();

        let library = unsafe { Library::new(&load_path) }.map_err(|source| {
            CutlassCompilerError::LibraryLoad {
                path: validated.path.clone(),
                source,
            }
        })?;

        unsafe {
            Ok(Self {
                cute_compiler_create: resolve(&library, "cutlass_cute_compiler_create")?,
                compiler_destroy: resolve(&library, "cutlass_compiler_destroy")?,
                compiler_set_device_target: resolve(
                    &library,
                    "cutlass_compiler_set_device_target",
                )?,
                compiler_set_abi: resolve(&library, "cutlass_compiler_set_abi")?,
                compiler_set_pipeline: resolve(&library, "cutlass_compiler_set_pipeline")?,
                compiler_compile_to: resolve(&library, "cutlass_compiler_compile_to")?,
                artifacts_from_textual_form: resolve(
                    &library,
                    "cutlass_artifacts_from_textual_form",
                )?,
                artifacts_get_type: resolve(&library, "cutlass_artifacts_get_type")?,
                artifacts_get_data: resolve(&library, "cutlass_artifacts_get_data")?,
                artifacts_function_count: resolve(&library, "cutlass_artifacts_function_count")?,
                artifacts_destroy: resolve(&library, "cutlass_artifacts_destroy")?,
                path: validated.path,
                fingerprint: validated.fingerprint,
                _loaded_file: validated.file,
                _library: library,
            })
        }
    }
}

unsafe fn resolve<T: Copy>(
    library: &Library,
    name: &'static str,
) -> Result<T, CutlassCompilerError> {
    let symbol: Symbol<T> = unsafe { library.get(name.as_bytes()) }.map_err(|source| {
        CutlassCompilerError::SymbolNotFound {
            symbol: name,
            source,
        }
    })?;
    Ok(unsafe { *symbol.into_raw() })
}

struct CompilerHandle<'api> {
    api: &'api Api,
    raw: CompilerRef,
}

impl<'api> CompilerHandle<'api> {
    fn create_cute(api: &'api Api) -> Result<Self, CutlassCompilerError> {
        let raw = unsafe { (api.cute_compiler_create)() };
        if raw.is_null() {
            return Err(CutlassCompilerError::NullHandle {
                operation: "cutlass_cute_compiler_create",
            });
        }
        Ok(Self { api, raw })
    }

    fn set_device_target(&self, target: &str) -> Result<(), CutlassCompilerError> {
        let mut error_ptr = ptr::null_mut();
        let mut error_len = 0;
        let status = unsafe {
            (self.api.compiler_set_device_target)(
                self.raw,
                target.as_ptr().cast(),
                target.len(),
                &mut error_ptr,
                &mut error_len,
            )
        };
        let message = unsafe { take_error_message(error_ptr, error_len) };
        if status == 0 {
            Ok(())
        } else {
            Err(CutlassCompilerError::Configure {
                operation: "cutlass_compiler_set_device_target",
                status,
                message,
            })
        }
    }

    fn set_external_module_passthrough_abi(&self) -> Result<(), CutlassCompilerError> {
        let mut error_ptr = ptr::null_mut();
        let mut error_len = 0;
        let status = unsafe {
            (self.api.compiler_set_abi)(
                self.raw,
                CUTLASS_ABI_TBD_EXTERNAL_MODULE_PASSTHROUGH,
                &mut error_ptr,
                &mut error_len,
            )
        };
        let message = unsafe { take_error_message(error_ptr, error_len) };
        if status == 0 {
            Ok(())
        } else {
            Err(CutlassCompilerError::Configure {
                operation: "cutlass_compiler_set_abi",
                status,
                message,
            })
        }
    }

    fn set_pipeline(
        &self,
        artifact_type: c_int,
        pipeline: &str,
    ) -> Result<(), CutlassCompilerError> {
        let status = unsafe {
            (self.api.compiler_set_pipeline)(
                self.raw,
                artifact_type,
                pipeline.as_ptr().cast(),
                pipeline.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(CutlassCompilerError::PipelineConfiguration {
                artifact_type,
                status,
            })
        }
    }

    fn compile_to_object(
        &self,
        input: &mut ArtifactsHandle<'api>,
    ) -> Result<ArtifactsHandle<'api>, CutlassCompilerError> {
        let mut error_ptr = ptr::null_mut();
        let mut error_len = 0;
        let raw = unsafe {
            (self.api.compiler_compile_to)(
                self.raw,
                input.raw,
                CUTLASS_ARTIFACT_OBJECT,
                &mut error_ptr,
                &mut error_len,
            )
        };
        let message = unsafe { take_error_message(error_ptr, error_len) };
        if raw.is_null() {
            return Err(CutlassCompilerError::Compile { message });
        }
        Ok(ArtifactsHandle { api: self.api, raw })
    }
}

impl Drop for CompilerHandle<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.compiler_destroy)(self.raw) };
    }
}

struct ArtifactsHandle<'api> {
    api: &'api Api,
    raw: ArtifactsRef,
}

impl<'api> ArtifactsHandle<'api> {
    fn from_textual_precompiled_mlir(
        api: &'api Api,
        mlir: &str,
    ) -> Result<Self, CutlassCompilerError> {
        Self::from_textual_mlir(
            api,
            mlir,
            CUTLASS_ARTIFACT_PRE_COMPILED_MLIR,
            "PreCompiledMlir",
        )
    }

    fn from_textual_mlir(
        api: &'api Api,
        mlir: &str,
        artifact_type: c_int,
        artifact_kind: &'static str,
    ) -> Result<Self, CutlassCompilerError> {
        let raw = unsafe {
            (api.artifacts_from_textual_form)(artifact_type, mlir.as_ptr().cast(), mlir.len())
        };
        if raw.is_null() {
            return Err(CutlassCompilerError::MlirParse { artifact_kind });
        }
        Ok(Self { api, raw })
    }

    fn artifact_type(&self) -> c_int {
        unsafe { (self.api.artifacts_get_type)(self.raw) }
    }

    fn data(&self) -> Result<Vec<u8>, CutlassCompilerError> {
        let mut data_ptr = ptr::null_mut();
        let mut data_len = 0;
        let status =
            unsafe { (self.api.artifacts_get_data)(self.raw, &mut data_ptr, &mut data_len) };
        if status != 0 {
            unsafe { free_if_nonnull(data_ptr) };
            return Err(CutlassCompilerError::ArtifactQuery {
                operation: "cutlass_artifacts_get_data",
                reason: format!("returned status {status}"),
            });
        }
        unsafe { copy_malloc_bytes(data_ptr, data_len, "cutlass_artifacts_get_data") }
    }

    fn function_count(&self) -> Result<usize, CutlassCompilerError> {
        let mut count = 0;
        let status = unsafe { (self.api.artifacts_function_count)(self.raw, &mut count) };
        if status == 0 {
            Ok(count)
        } else {
            Err(CutlassCompilerError::ArtifactQuery {
                operation: "cutlass_artifacts_function_count",
                reason: format!("returned status {status}"),
            })
        }
    }
}

impl Drop for ArtifactsHandle<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.artifacts_destroy)(self.raw) };
    }
}

unsafe fn take_error_message(ptr: *mut c_char, len: usize) -> String {
    if ptr.is_null() {
        return "no diagnostic returned".to_string();
    }
    if len > isize::MAX as usize {
        unsafe { free(ptr.cast()) };
        return format!("diagnostic length {len} exceeds Rust's addressable slice limit");
    }
    let bytes = unsafe { slice::from_raw_parts(ptr.cast::<u8>(), len) };
    let message = String::from_utf8_lossy(bytes).into_owned();
    unsafe { free(ptr.cast()) };
    message
}

unsafe fn copy_malloc_bytes(
    ptr: *mut u8,
    len: usize,
    operation: &'static str,
) -> Result<Vec<u8>, CutlassCompilerError> {
    if ptr.is_null() {
        if len == 0 {
            return Ok(Vec::new());
        }
        return Err(CutlassCompilerError::ArtifactQuery {
            operation,
            reason: format!("returned a null pointer with non-zero length {len}"),
        });
    }
    if len > isize::MAX as usize {
        unsafe { free(ptr.cast()) };
        return Err(CutlassCompilerError::ArtifactQuery {
            operation,
            reason: format!("returned length {len} exceeds Rust's addressable slice limit"),
        });
    }
    let bytes = unsafe { slice::from_raw_parts(ptr, len) }.to_vec();
    unsafe { free(ptr.cast()) };
    Ok(bytes)
}

unsafe fn free_if_nonnull(ptr: *mut u8) {
    if !ptr.is_null() {
        unsafe { free(ptr.cast()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fingerprint(data: &[u8]) -> Sha256Fingerprint {
        fingerprint_reader(&mut Cursor::new(data)).expect("in-memory digest cannot fail")
    }

    #[test]
    fn fingerprint_parser_is_exact_and_round_trips() {
        let expected = fingerprint(b"CUTLASS 4.7 compiler");
        let lower = expected.to_string();
        let upper = lower.to_ascii_uppercase();

        assert_eq!(lower.parse::<Sha256Fingerprint>().unwrap(), expected);
        assert_eq!(upper.parse::<Sha256Fingerprint>().unwrap(), expected);
        assert_eq!(expected.as_bytes().len(), 32);
        assert!(matches!(
            "00".parse::<Sha256Fingerprint>(),
            Err(FingerprintParseError::Length { actual: 2 })
        ));
        assert!(matches!(
            format!("{}zz", &lower[..62]).parse::<Sha256Fingerprint>(),
            Err(FingerprintParseError::NonHex { offset: 62 })
        ));
        let unicode = format!("{}é", "0".repeat(62));
        assert_eq!(unicode.len(), 64);
        assert!(matches!(
            unicode.parse::<Sha256Fingerprint>(),
            Err(FingerprintParseError::NonHex { offset: 62 })
        ));
    }

    #[test]
    fn library_validation_requires_an_absolute_regular_file() {
        let digest = fingerprint(b"unused");
        let relative = Path::new("libCutlassCompiler.so");
        assert!(matches!(
            validate_library_file(relative, digest),
            Err(CutlassCompilerError::RelativeLibraryPath { .. })
        ));

        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            validate_library_file(directory.path(), digest),
            Err(CutlassCompilerError::LibraryNotRegular { .. })
        ));
    }

    #[test]
    fn library_validation_checks_the_opened_file_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("libCutlassCompiler.so");
        let contents = b"not a real library, sufficient for validation";
        fs::write(&path, contents).unwrap();

        let expected = fingerprint(contents);
        let validated = validate_library_file(&path, expected).unwrap();
        assert_eq!(validated.path, path);
        assert_eq!(validated.fingerprint, expected);

        let wrong = fingerprint(b"different library");
        assert!(matches!(
            validate_library_file(&path, wrong),
            Err(CutlassCompilerError::FingerprintMismatch {
                expected: rejected,
                actual,
                ..
            }) if rejected == wrong && actual == expected
        ));
    }

    #[cfg(unix)]
    #[test]
    fn library_validation_rejects_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("real.so");
        let link = directory.path().join("libCutlassCompiler.so");
        let contents = b"compiler";
        fs::write(&target, contents).unwrap();
        symlink(&target, &link).unwrap();

        assert!(matches!(
            validate_library_file(&link, fingerprint(contents)),
            Err(CutlassCompilerError::LibraryNotRegular { .. })
        ));
    }
}
