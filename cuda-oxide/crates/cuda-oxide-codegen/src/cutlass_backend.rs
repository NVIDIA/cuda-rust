/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile MLIR with the pinned CUTLASS library and validate its CUDA image.

use crate::cutlass_compiler::CutlassCompilerLibrary;
use crate::error::PipelineError;
use object::{Object, ObjectSection, ObjectSymbol};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Exact public CUTLASS profile implemented by this checkout.
pub const CUTLASS_FULL_CUTE_MLIR22_PROFILE: &str = "cutlass-full-cute-mlir22";

/// Official CUTLASS release providing the compiler-library ABI.
pub const CUTLASS_COMPILER_VERSION: &str = "4.7.0";

/// SHA-256 of the official x86_64/CUDA-13 CUTLASS 4.7 compiler library.
pub const CUTLASS_COMPILER_LIBRARY_SHA256: &str =
    "57df017e3585a10443c74c8b4cd99bda854242fb2f4c9534cf56d58c2c741628";

/// Version-pinned ABI mode for external `gpu.module` device images. Concrete
/// CUTLASS wrapper ABIs require FunctionMetadata and cannot represent this
/// direct CUDA-launch path.
const CUTLASS_EXTERNAL_MODULE_ABI: &str =
    "tbd-external-module-passthrough; zero FunctionMetadata; validated CUDA KPARAM layout";

/// Exact first-stage pipeline validated with the fingerprint-pinned library.
///
/// The C API pass manager is already anchored on `builtin.module`; adding a
/// second `builtin.module(...)` wrapper changes the pass nesting and does not
/// lower this input. The leading canonicalization is also part of the pinned
/// contract: CUTLASS 4.7's default first stage does not accept the generic
/// operation form emitted by the profile.
const CUTLASS_PRECOMPILED_PIPELINE: &str = "canonicalize,cute-to-nvvm{check-inline-asm=false cubin-format=bin opt-level=3 use-software-pipeline-pass=true use-fold-static-pass=true use-insert-range-information-pass=true use-loop-invariant-pass=true use-strength-reduction-pass=true use-eliminate-unnecessary-sync-pass=true use-infer-loop-attrs-pass=true enable-cuda-dialect=true cuda-dialect-external-module=true}";

#[cfg(test)]
#[path = "tests/cutlass/sm100.rs"]
mod sm100_tests;

#[cfg(test)]
#[path = "tests/cutlass/tcgen05.rs"]
mod tcgen05_tests;

#[cfg(test)]
#[path = "tests/cutlass/grid_constant.rs"]
mod grid_constant_tests;

/// Compiler library and MLIR profile used by the CUTLASS backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutlassBackendConfig {
    /// Absolute path to the exact `libCutlassCompiler.so` installed by
    /// `cargo oxide toolchain install cutlass`.
    pub compiler_library: PathBuf,
    /// Pinned MLIR mapping profile. Only the in-tree profile is accepted.
    pub profile: String,
}

impl CutlassBackendConfig {
    /// Construct a backend configuration for the pinned in-tree profile.
    pub fn new(compiler_library: PathBuf) -> Self {
        Self {
            compiler_library,
            profile: CUTLASS_FULL_CUTE_MLIR22_PROFILE.to_owned(),
        }
    }
}

/// Device compiler selected after the shared MIR/CuTe preparation stages.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DeviceBackend {
    /// CUDA Oxide MIR/NVVM/LLVM backend.
    #[default]
    Native,
    /// Official CUTLASS compiler-library path, producing a CUDA image.
    CutlassMlir(CutlassBackendConfig),
}

#[derive(Debug)]
pub(crate) struct CutlassRunOutput {
    pub(crate) diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelParameterKind {
    GlobalPointer64,
    Scalar64,
    GridConstant { size: u16, alignment: u16 },
}

impl KernelParameterKind {
    const fn size(self) -> u16 {
        match self {
            Self::GlobalPointer64 | Self::Scalar64 => 8,
            Self::GridConstant { size, .. } => size,
        }
    }

    const fn alignment(self) -> u16 {
        match self {
            Self::GlobalPointer64 | Self::Scalar64 => 8,
            Self::GridConstant { alignment, .. } => alignment,
        }
    }

    const fn cuda_parameter_space(self) -> u32 {
        match self {
            Self::GlobalPointer64 => 5,
            Self::Scalar64 | Self::GridConstant { .. } => 0,
        }
    }
}

type ExpectedKernelAbis = BTreeMap<String, Vec<KernelParameterKind>>;

/// Compile one rendered profile module and atomically publish a verified CUDA
/// image suitable for `cuModuleLoadData`.
///
/// CUTLASS 4.7 returns a host relocatable object. Its exact, fingerprint-pinned
/// contract contains one local data symbol named `kernels_binary`; that symbol
/// is either a raw CUDA ELF image or a CUDA fatbinary. A fatbinary must contain
/// exactly one matching, uncompressed cubin, which is extracted so the public
/// `Cubin` artifact kind always carries an actual CUDA ELF image.
pub(crate) fn compile_to_cubin(
    config: &CutlassBackendConfig,
    rendered_mlir: &str,
    target: &str,
    expected_kernels: &[String],
    cubin_output: &Path,
    mlir_output: Option<&Path>,
) -> Result<CutlassRunOutput, PipelineError> {
    if config.profile != CUTLASS_FULL_CUTE_MLIR22_PROFILE {
        return Err(backend_error(format!(
            "unsupported MLIR profile {:?}; this compiler implements only {:?}",
            config.profile, CUTLASS_FULL_CUTE_MLIR22_PROFILE
        )));
    }
    if expected_kernels.is_empty() {
        return Err(backend_error(
            "the CUTLASS backend requires at least one named kernel entry",
        ));
    }
    parse_sm_target(target)?;

    let compiler =
        CutlassCompilerLibrary::load(&config.compiler_library, CUTLASS_COMPILER_LIBRARY_SHA256)
            .map_err(|error| {
                backend_error(format!("could not load official CUTLASS compiler: {error}"))
            })?;
    let execution_mlir = add_execution_manifest(
        rendered_mlir,
        compiler.path(),
        &compiler.fingerprint().to_string(),
        target,
        &config.profile,
    )?;
    let expected_abis = expected_kernel_abis(&execution_mlir, expected_kernels)
        .map_err(|reason| backend_error(format!("invalid Backend-B kernel ABI: {reason}")))?;
    if let Some(path) = mlir_output {
        publish_atomically(path, execution_mlir.as_bytes())?;
    }

    let object = compiler
        .compile_precompiled_mlir_to_object(&execution_mlir, target, CUTLASS_PRECOMPILED_PIPELINE)
        .map_err(|error| backend_error(format!("official CUTLASS compilation failed: {error}")))?;
    let image = extract_kernels_binary(&object.elf)
        .map_err(|reason| backend_error(format!("invalid CUTLASS ObjectArtifact: {reason}")))?;
    let (cubin, source_kind) =
        extract_validated_cubin(&image, target, &expected_abis).map_err(|reason| {
            backend_error(format!(
                "CUTLASS ObjectArtifact contains an invalid CUDA image: {reason}"
            ))
        })?;
    publish_atomically(cubin_output, &cubin)?;

    Ok(CutlassRunOutput {
        diagnostics: vec![format!(
            "official CUTLASS {CUTLASS_COMPILER_VERSION} emitted a validated direct-ABI cubin extracted from {source_kind} ({} bytes)",
            cubin.len()
        )],
    })
}

fn add_execution_manifest(
    rendered_mlir: &str,
    compiler_library: &Path,
    compiler_sha256: &str,
    target: &str,
    profile: &str,
) -> Result<String, PipelineError> {
    let pipeline_sha256 = format!("{:x}", Sha256::digest(CUTLASS_PRECOMPILED_PIPELINE));
    let manifest = json!({
        "profile": profile,
        "target": target,
        "cutlass_compiler_version": CUTLASS_COMPILER_VERSION,
        "cutlass_compiler_library": compiler_library,
        "cutlass_compiler_library_sha256": compiler_sha256,
        "cutlass_external_module_abi": CUTLASS_EXTERNAL_MODULE_ABI,
        "precompiled_pipeline_sha256": pipeline_sha256,
    });
    let manifest = serde_json::to_string_pretty(&manifest).map_err(|error| {
        backend_error(format!(
            "could not render CUTLASS execution manifest: {error}"
        ))
    })?;
    let mut output = String::from("// cuda-oxide CUTLASS Backend-B execution manifest\n");
    for line in manifest.lines() {
        output.push_str("// ");
        output.push_str(line);
        output.push('\n');
    }
    output.push_str(rendered_mlir);
    Ok(output)
}

/// Read the expected launch argument sizes and alignments from exported MLIR.
///
/// A Rust slice uses two arguments: pointer, then `i64` length. This profile also
/// accepts `i64` scalars. A grid-constant reference carries the value's bytes at
/// its Rust alignment; MLIR represents its address as an opaque pointer.
fn expected_kernel_abis(
    rendered_mlir: &str,
    expected_kernels: &[String],
) -> Result<ExpectedKernelAbis, String> {
    const PREFIX: &str = "\"func.func\"() <{";
    let mut abis = BTreeMap::new();
    for kernel in expected_kernels {
        if kernel.contains(['"', '\\']) {
            return Err(format!("kernel name {kernel:?} cannot be matched safely"));
        }
        if abis.contains_key(kernel) {
            return Err(format!("duplicate expected kernel {kernel:?}"));
        }
        let name_marker = format!("sym_name = \"{kernel}\"");
        let signatures = rendered_mlir
            .lines()
            .filter_map(|line| {
                let properties = line.trim().strip_prefix(PREFIX)?;
                let (properties, _) = properties.split_once("}> ({")?;
                properties.contains(&name_marker).then_some(properties)
            })
            .collect::<Vec<_>>();
        if signatures.len() != 1 {
            return Err(format!(
                "expected exactly one flattened func.func signature for kernel {kernel:?}, found {}",
                signatures.len()
            ));
        }
        let properties = parse_abi_dictionary(signatures[0])?;
        let quoted_name = format!("\"{kernel}\"");
        if properties.get("sym_name") != Some(&quoted_name.as_str()) {
            return Err(format!("kernel {kernel:?} has an ambiguous symbol name"));
        }
        for property in properties.keys() {
            if !matches!(*property, "sym_name" | "function_type" | "arg_attrs") {
                return Err(format!(
                    "kernel {kernel:?} has unsupported function property {property:?}"
                ));
            }
        }
        let signature = properties
            .get("function_type")
            .and_then(|value| value.strip_prefix('(')?.strip_suffix(") -> ()"))
            .ok_or_else(|| format!("kernel {kernel:?} has an unsupported function type"))?;
        let attributes = properties
            .get("arg_attrs")
            .map(|value| {
                let entries = value
                    .strip_prefix('[')
                    .and_then(|value| value.strip_suffix(']'))
                    .ok_or_else(|| format!("kernel {kernel:?} arg_attrs is not an array"))?;
                split_abi_items(entries)
            })
            .transpose()?;
        let types = split_abi_items(signature)?;
        if let Some(attributes) = &attributes
            && attributes.len() != types.len()
        {
            return Err(format!(
                "kernel {kernel:?} has {} argument attributes for {} parameters",
                attributes.len(),
                types.len()
            ));
        }
        let parameters = types
            .iter()
            .enumerate()
            .map(|(ordinal, parameter)| {
                let attributes = attributes.as_ref().map(|items| items[ordinal]);
                parse_parameter_abi(parameter, attributes)
                    .map_err(|reason| format!("kernel {kernel:?} parameter {ordinal}: {reason}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        abis.insert(kernel.clone(), parameters);
    }
    Ok(abis)
}

fn parse_parameter_abi(
    parameter: &str,
    attributes: Option<&str>,
) -> Result<KernelParameterKind, String> {
    let attributes = attributes
        .map(|value| {
            let entries = value
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
                .ok_or_else(|| "argument attributes must be a dictionary".to_owned())?;
            parse_abi_dictionary(entries)
        })
        .transpose()?
        .unwrap_or_default();
    if !attributes.is_empty() {
        if parameter != "!llvm.ptr"
            || attributes.len() != 3
            || attributes.get("nvvm.grid_constant") != Some(&"unit")
        {
            return Err("unsupported argument attributes; expected an aligned grid-constant by-value pointer".to_owned());
        }
        let size = attributes
            .get("llvm.byval")
            .and_then(|value| value.strip_prefix("!llvm.array<")?.strip_suffix(" x i8>"))
            .and_then(|value| value.parse::<u16>().ok())
            .filter(|size| (1..=0x3fff).contains(size))
            .ok_or_else(|| {
                "llvm.byval must be a nonempty byte array fitting the CUDA parameter size field"
                    .to_owned()
            })?;
        let alignment = attributes
            .get("llvm.align")
            .and_then(|value| value.strip_suffix(" : i64"))
            .and_then(|value| value.parse::<u16>().ok())
            .filter(|alignment| alignment.is_power_of_two())
            .ok_or_else(|| {
                "llvm.align must be a nonzero power-of-two i64 fitting the CUDA parameter block"
                    .to_owned()
            })?;
        return Ok(KernelParameterKind::GridConstant { size, alignment });
    }
    match parameter {
        "!llvm.ptr" => Ok(KernelParameterKind::GlobalPointer64),
        "i64" => Ok(KernelParameterKind::Scalar64),
        other => Err(format!(
            "unsupported flattened parameter type {other:?}; only !llvm.ptr and i64 have a pinned direct-launch ABI"
        )),
    }
}

fn parse_abi_dictionary(input: &str) -> Result<BTreeMap<&str, &str>, String> {
    let mut result = BTreeMap::new();
    for item in split_abi_items(input)? {
        let (name, value) = if item == "nvvm.grid_constant" {
            (item, "unit")
        } else {
            item.split_once(" = ")
                .ok_or_else(|| format!("malformed ABI dictionary entry {item:?}"))?
        };
        if result.insert(name, value).is_some() {
            return Err(format!("duplicate ABI dictionary entry {name:?}"));
        }
    }
    Ok(result)
}

/// Split argument lists at commas, keeping nested types and attributes intact.
fn split_abi_items(input: &str) -> Result<Vec<&str>, String> {
    if input.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut items = Vec::new();
    let mut brackets = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0;
    for (offset, byte) in input.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'(' => brackets.push(b')'),
            b'[' => brackets.push(b']'),
            b'{' => brackets.push(b'}'),
            b'<' => brackets.push(b'>'),
            b'>' if offset > 0 && input.as_bytes()[offset - 1] == b'-' => {}
            b')' | b']' | b'}' | b'>' => {
                if brackets.pop() != Some(byte) {
                    return Err("unbalanced delimiters in kernel ABI".to_owned());
                }
            }
            b',' if brackets.is_empty() => {
                items.push(input[start..offset].trim());
                start = offset + 1;
            }
            _ => {}
        }
    }
    if quoted || !brackets.is_empty() {
        return Err("unterminated kernel ABI property".to_owned());
    }
    items.push(input[start..].trim());
    if items.contains(&"") {
        return Err("empty entry in kernel ABI".to_owned());
    }
    Ok(items)
}

fn extract_kernels_binary(object_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let file = object::File::parse(object_bytes).map_err(|error| error.to_string())?;
    if file.format() != object::BinaryFormat::Elf
        || file.kind() != object::ObjectKind::Relocatable
        || file.architecture() != object::Architecture::X86_64
    {
        return Err(format!(
            "expected an x86-64 ELF relocatable, got {:?}/{:?}/{:?}",
            file.format(),
            file.kind(),
            file.architecture()
        ));
    }

    let mut symbols = file
        .symbols()
        .filter(|symbol| symbol.name() == Ok("kernels_binary"));
    let symbol = symbols
        .next()
        .ok_or_else(|| "missing exact `kernels_binary` data symbol".to_owned())?;
    if symbols.next().is_some() {
        return Err("multiple `kernels_binary` symbols are ambiguous".to_owned());
    }
    if !symbol.is_definition() || symbol.kind() != object::SymbolKind::Data || symbol.size() == 0 {
        return Err("`kernels_binary` is not a nonempty defined data symbol".to_owned());
    }
    let section_index = symbol
        .section_index()
        .ok_or_else(|| "`kernels_binary` does not name an object section".to_owned())?;
    let section = file
        .section_by_index(section_index)
        .map_err(|error| error.to_string())?;
    let section_data = section.data().map_err(|error| error.to_string())?;
    let relative = symbol
        .address()
        .checked_sub(section.address())
        .ok_or_else(|| "`kernels_binary` address precedes its section".to_owned())?;
    let start = usize::try_from(relative)
        .map_err(|_| "`kernels_binary` offset does not fit in memory".to_owned())?;
    let size = usize::try_from(symbol.size())
        .map_err(|_| "`kernels_binary` size does not fit in memory".to_owned())?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| "`kernels_binary` range overflows".to_owned())?;
    section_data
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| "`kernels_binary` range is outside its section".to_owned())
}

fn extract_validated_cubin(
    bytes: &[u8],
    target: &str,
    expected_abis: &ExpectedKernelAbis,
) -> Result<(Vec<u8>, &'static str), String> {
    const FATBIN_MAGIC: u32 = 0xba55_ed50;
    if bytes.starts_with(b"\x7fELF") {
        validate_cuda_cubin(bytes, target, expected_abis)?;
        return Ok((bytes.to_vec(), "a raw cubin"));
    }
    if read_u32(bytes, 0)? != FATBIN_MAGIC {
        return Err("image is neither a CUDA ELF cubin nor a CUDA fatbinary".to_owned());
    }
    let cubin = extract_cubin_from_fatbin(bytes, target, expected_abis)?;
    Ok((cubin, "a CUDA fatbinary"))
}

fn extract_cubin_from_fatbin(
    bytes: &[u8],
    target: &str,
    expected_abis: &ExpectedKernelAbis,
) -> Result<Vec<u8>, String> {
    const HEADER_SIZE: usize = 16;
    const COMMON_SIZE: usize = 64;
    const FATBIN_VERSION: u16 = 1;
    const FATBIN_KIND_ELF: u16 = 2;
    const COMPRESSED_FLAGS: u64 = 0xf000;

    if bytes.len() < HEADER_SIZE {
        return Err("truncated CUDA fatbinary header".to_owned());
    }
    if read_u16(bytes, 4)? != FATBIN_VERSION {
        return Err("unsupported CUDA fatbinary version".to_owned());
    }
    let header_size = usize::from(read_u16(bytes, 6)?);
    if header_size < HEADER_SIZE || header_size % 8 != 0 {
        return Err("invalid CUDA fatbinary header size".to_owned());
    }
    let payload_size = usize::try_from(read_u64(bytes, 8)?)
        .map_err(|_| "CUDA fatbinary size does not fit in memory".to_owned())?;
    let total_size = header_size
        .checked_add(payload_size)
        .ok_or_else(|| "CUDA fatbinary size overflows".to_owned())?;
    if total_size != bytes.len() || payload_size % 8 != 0 {
        return Err("CUDA fatbinary length does not match its header".to_owned());
    }

    let expected_sm = parse_sm_target_value(target)?;
    let mut cursor = header_size;
    let mut cubin = None;
    while cursor < bytes.len() {
        if bytes.len() - cursor < COMMON_SIZE {
            return Err("truncated CUDA fatbinary code header".to_owned());
        }
        let kind = read_u16(bytes, cursor)?;
        let version = read_u16(bytes, cursor + 2)?;
        let code_offset = usize::try_from(read_u32(bytes, cursor + 4)?)
            .map_err(|_| "fatbinary code offset does not fit in memory".to_owned())?;
        let code_size = usize::try_from(read_u64(bytes, cursor + 8)?)
            .map_err(|_| "fatbinary code size does not fit in memory".to_owned())?;
        let compressed_size = read_u32(bytes, cursor + 16)?;
        let arch = read_u32(bytes, cursor + 28)?;
        let flags = read_u64(bytes, cursor + 40)?;
        if kind != FATBIN_KIND_ELF || version != 0x0101 {
            return Err(format!(
                "unsupported fatbinary entry kind/version {kind:#x}/{version:#x}"
            ));
        }
        if compressed_size != 0 || flags & COMPRESSED_FLAGS != 0 {
            return Err(
                "compressed CUDA fatbinary entries are not supported by this pinned contract"
                    .to_owned(),
            );
        }
        if arch != expected_sm {
            return Err(format!(
                "fatbinary entry targets sm_{arch}, expected {target}"
            ));
        }
        if code_offset < COMMON_SIZE || code_offset % 8 != 0 || code_size == 0 {
            return Err("invalid CUDA fatbinary code range".to_owned());
        }
        let code_start = cursor
            .checked_add(code_offset)
            .ok_or_else(|| "fatbinary code offset overflows".to_owned())?;
        let next = code_start
            .checked_add(code_size)
            .ok_or_else(|| "fatbinary code size overflows".to_owned())?;
        let code = bytes
            .get(code_start..next)
            .ok_or_else(|| "fatbinary code range is out of bounds".to_owned())?;
        validate_cuda_cubin(code, target, expected_abis)?;
        if cubin.replace(code.to_vec()).is_some() {
            return Err(
                "CUDA fatbinary contains multiple cubin entries for one exact-target compilation"
                    .to_owned(),
            );
        }
        cursor = next;
    }
    cubin.ok_or_else(|| "CUDA fatbinary contains no cubin entries".to_owned())
}

fn validate_cuda_cubin(
    bytes: &[u8],
    target: &str,
    expected_abis: &ExpectedKernelAbis,
) -> Result<(), String> {
    const ELF_HEADER_SIZE: usize = 64;
    const SECTION_HEADER_SIZE: usize = 64;
    const EM_CUDA: u16 = 190;

    if bytes.len() < ELF_HEADER_SIZE {
        return Err(format!("cubin is only {} bytes", bytes.len()));
    }
    if &bytes[..4] != b"\x7fELF" || bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1 {
        return Err("expected a little-endian ELF64 cubin".to_owned());
    }
    if read_u16(bytes, 16)? != 2 || read_u16(bytes, 18)? != EM_CUDA {
        return Err("ELF image is not an executable NVIDIA CUDA cubin".to_owned());
    }
    let expected_sm = parse_sm_target_value(target)?;
    let actual_sm = (read_u32(bytes, 48)? >> 8) & 0xff;
    if actual_sm != expected_sm {
        return Err(format!(
            "cubin ELF flags target sm_{actual_sm}, expected {target}"
        ));
    }
    let section_offset = usize::try_from(read_u64(bytes, 40)?)
        .map_err(|_| "section table offset does not fit in memory".to_owned())?;
    let section_entry_size = usize::from(read_u16(bytes, 58)?);
    let section_count = usize::from(read_u16(bytes, 60)?);
    let names_index = usize::from(read_u16(bytes, 62)?);
    if section_entry_size < SECTION_HEADER_SIZE
        || section_count == 0
        || names_index >= section_count
    {
        return Err("invalid or unsupported cubin section table".to_owned());
    }
    let section_table_end = section_entry_size
        .checked_mul(section_count)
        .and_then(|size| section_offset.checked_add(size))
        .ok_or_else(|| "cubin section table overflows".to_owned())?;
    if section_table_end > bytes.len() {
        return Err("cubin section table is out of bounds".to_owned());
    }
    let names_header = section_offset + names_index * section_entry_size;
    let names_offset = usize::try_from(read_u64(bytes, names_header + 24)?)
        .map_err(|_| "section-name table offset does not fit in memory".to_owned())?;
    let names_size = usize::try_from(read_u64(bytes, names_header + 32)?)
        .map_err(|_| "section-name table size does not fit in memory".to_owned())?;
    let names_end = names_offset
        .checked_add(names_size)
        .ok_or_else(|| "section-name table overflows".to_owned())?;
    let names = bytes
        .get(names_offset..names_end)
        .ok_or_else(|| "cubin section-name table is out of bounds".to_owned())?;

    let mut text_sections = BTreeMap::<String, usize>::new();
    let mut parameter_sections = BTreeMap::<String, &[u8]>::new();
    let mut target_note = None;
    for index in 0..section_count {
        let header = section_offset + index * section_entry_size;
        let name_offset = usize::try_from(read_u32(bytes, header)?)
            .map_err(|_| "section name offset does not fit in memory".to_owned())?;
        let name = nul_terminated(names, name_offset)?;
        if let Some(kernel) = name.strip_prefix(".text.") {
            let body_offset = usize::try_from(read_u64(bytes, header + 24)?)
                .map_err(|_| format!("section {name:?} offset does not fit in memory"))?;
            let size = usize::try_from(read_u64(bytes, header + 32)?)
                .map_err(|_| format!("section {name:?} size does not fit in memory"))?;
            let body_end = body_offset
                .checked_add(size)
                .ok_or_else(|| format!("section {name:?} overflows"))?;
            if body_end > bytes.len() {
                return Err(format!("section {name:?} is out of bounds"));
            }
            text_sections.insert(kernel.to_owned(), size);
        } else if let Some(kernel) = name.strip_prefix(".nv.info.") {
            if !expected_abis.contains_key(kernel) {
                continue;
            }
            if read_u32(bytes, header + 4)? != 0x7000_0000 {
                return Err(format!("section {name:?} is not CUDA_INFO"));
            }
            let body_offset = usize::try_from(read_u64(bytes, header + 24)?)
                .map_err(|_| format!("section {name:?} offset does not fit in memory"))?;
            let size = usize::try_from(read_u64(bytes, header + 32)?)
                .map_err(|_| format!("section {name:?} size does not fit in memory"))?;
            let body_end = body_offset
                .checked_add(size)
                .ok_or_else(|| format!("section {name:?} overflows"))?;
            let body = bytes
                .get(body_offset..body_end)
                .ok_or_else(|| format!("section {name:?} is out of bounds"))?;
            if parameter_sections.insert(kernel.to_owned(), body).is_some() {
                return Err(format!("cubin contains multiple {name:?} sections"));
            }
        } else if name == ".note.nv.tkinfo" {
            if target_note.is_some() {
                return Err("cubin contains multiple .note.nv.tkinfo sections".to_owned());
            }
            if read_u32(bytes, header + 4)? != 7 {
                return Err(".note.nv.tkinfo is not an ELF note section".to_owned());
            }
            if read_u64(bytes, header + 48)? != 4 {
                return Err(".note.nv.tkinfo does not have 4-byte alignment".to_owned());
            }
            let body_offset = usize::try_from(read_u64(bytes, header + 24)?)
                .map_err(|_| ".note.nv.tkinfo offset does not fit in memory".to_owned())?;
            let size = usize::try_from(read_u64(bytes, header + 32)?)
                .map_err(|_| ".note.nv.tkinfo size does not fit in memory".to_owned())?;
            let body_end = body_offset
                .checked_add(size)
                .ok_or_else(|| ".note.nv.tkinfo range overflows".to_owned())?;
            target_note = Some(
                bytes
                    .get(body_offset..body_end)
                    .ok_or_else(|| ".note.nv.tkinfo is out of bounds".to_owned())?,
            );
        }
    }
    validate_target_note(
        target_note.ok_or_else(|| "cubin is missing .note.nv.tkinfo".to_owned())?,
        target,
    )?;
    for (kernel, expected_parameters) in expected_abis {
        match text_sections.get(kernel) {
            Some(size) if *size != 0 => {}
            Some(_) => return Err(format!("kernel section .text.{kernel} is empty")),
            None => return Err(format!("missing kernel code section .text.{kernel}")),
        }
        let parameter_section = parameter_sections
            .get(kernel)
            .ok_or_else(|| format!("missing kernel parameter section .nv.info.{kernel}"))?;
        validate_kernel_parameter_abi(kernel, parameter_section, expected_parameters)?;
    }
    Ok(())
}

/// Check each cubin's launch argument layout against the exported MLIR.
///
/// CUDA records parameter sizes and offsets in each kernel's `.nv.info` section.
/// We support the record format emitted by the pinned CUTLASS 4.7 / CUDA 13.x
/// toolchain and reject unknown or malformed records.
fn validate_kernel_parameter_abi(
    kernel: &str,
    section: &[u8],
    expected: &[KernelParameterKind],
) -> Result<(), String> {
    const EIFMT_NVAL: u8 = 1;
    const EIFMT_BVAL: u8 = 2;
    const EIFMT_HVAL: u8 = 3;
    const EIFMT_SVAL: u8 = 4;
    const EIATTR_KPARAM_INFO: u8 = 0x17;
    const EIATTR_CBANK_PARAM_SIZE: u8 = 0x19;
    const KPARAM_PAYLOAD_SIZE: u16 = 12;

    #[derive(Clone, Copy, Debug)]
    struct ActualParameter {
        offset: u16,
        size: u16,
        space: u32,
    }

    let mut parameters = BTreeMap::<u16, ActualParameter>::new();
    let mut parameter_block_size = None;
    let mut cursor = 0;
    while cursor < section.len() {
        let header = section
            .get(cursor..cursor + 4)
            .ok_or_else(|| format!(".nv.info.{kernel} has a truncated attribute header"))?;
        let format = header[0];
        let attribute = header[1];
        let immediate = u16::from_le_bytes([header[2], header[3]]);
        let payload_size = match format {
            EIFMT_NVAL | EIFMT_BVAL | EIFMT_HVAL => 0,
            EIFMT_SVAL => usize::from(immediate),
            _ => {
                return Err(format!(
                    ".nv.info.{kernel} uses unsupported attribute format {format:#x}"
                ));
            }
        };
        let payload_start = cursor + 4;
        let next = payload_start
            .checked_add(payload_size)
            .ok_or_else(|| format!(".nv.info.{kernel} attribute range overflows"))?;
        let payload = section
            .get(payload_start..next)
            .ok_or_else(|| format!(".nv.info.{kernel} has a truncated attribute payload"))?;

        match attribute {
            EIATTR_KPARAM_INFO => {
                if format != EIFMT_SVAL || immediate != KPARAM_PAYLOAD_SIZE {
                    return Err(format!(
                        ".nv.info.{kernel} has a malformed EIATTR_KPARAM_INFO record"
                    ));
                }
                let index = read_u32(payload, 0)?;
                let ordinal = read_u16(payload, 4)?;
                let offset = read_u16(payload, 6)?;
                let flags = read_u32(payload, 8)?;
                if index != 0 {
                    return Err(format!(
                        "kernel {kernel:?} parameter ordinal {ordinal} has unsupported index {index}"
                    ));
                }
                let size = u16::try_from((flags >> 18) & 0x3fff)
                    .expect("14-bit CUDA parameter size fits u16");
                let space = (flags >> 8) & 0xf;
                if parameters
                    .insert(
                        ordinal,
                        ActualParameter {
                            offset,
                            size,
                            space,
                        },
                    )
                    .is_some()
                {
                    return Err(format!(
                        "kernel {kernel:?} has duplicate parameter ordinal {ordinal}"
                    ));
                }
            }
            EIATTR_CBANK_PARAM_SIZE => {
                if format != EIFMT_HVAL {
                    return Err(format!(
                        ".nv.info.{kernel} has a malformed EIATTR_CBANK_PARAM_SIZE record"
                    ));
                }
                if parameter_block_size.replace(immediate).is_some() {
                    return Err(format!(
                        "kernel {kernel:?} has multiple parameter-block size records"
                    ));
                }
            }
            _ => {}
        }
        cursor = next;
    }

    if parameters.len() != expected.len() {
        return Err(format!(
            "kernel {kernel:?} has {} CUDA parameters, expected {} from its flattened MLIR signature",
            parameters.len(),
            expected.len()
        ));
    }
    let mut expected_offset = 0_u16;
    for (index, expected_kind) in expected.iter().copied().enumerate() {
        let ordinal = u16::try_from(index)
            .map_err(|_| format!("kernel {kernel:?} has too many parameters"))?;
        let alignment_mask = expected_kind.alignment() - 1;
        expected_offset = expected_offset
            .checked_add(alignment_mask)
            .map(|offset| offset & !alignment_mask)
            .ok_or_else(|| format!("kernel {kernel:?} parameter alignment overflows"))?;
        let actual = parameters
            .get(&ordinal)
            .ok_or_else(|| format!("kernel {kernel:?} is missing parameter ordinal {ordinal}"))?;
        if actual.offset != expected_offset
            || actual.size != expected_kind.size()
            || actual.space != expected_kind.cuda_parameter_space()
        {
            return Err(format!(
                "kernel {kernel:?} parameter {ordinal} is offset/size/space {}/{}/{}, expected {}/{}/{}",
                actual.offset,
                actual.size,
                actual.space,
                expected_offset,
                expected_kind.size(),
                expected_kind.cuda_parameter_space()
            ));
        }
        expected_offset = expected_offset
            .checked_add(expected_kind.size())
            .ok_or_else(|| format!("kernel {kernel:?} parameter block overflows"))?;
    }
    match parameter_block_size {
        Some(size) if size == expected_offset => Ok(()),
        Some(size) => Err(format!(
            "kernel {kernel:?} parameter block is {size} bytes, expected {expected_offset}"
        )),
        None => Err(format!(
            "kernel {kernel:?} is missing EIATTR_CBANK_PARAM_SIZE"
        )),
    }
}

fn validate_target_note(section: &[u8], target: &str) -> Result<(), String> {
    const NVIDIA_NOTE_TYPE: u32 = 0x7d0;
    const NVIDIA_OWNER: &[u8] = b"NVIDIA Corp\0";

    if section.len() < 12 {
        return Err(".note.nv.tkinfo is truncated".to_owned());
    }
    let name_size = usize::try_from(read_u32(section, 0)?)
        .map_err(|_| "NVIDIA note owner size does not fit in memory".to_owned())?;
    let descriptor_size = usize::try_from(read_u32(section, 4)?)
        .map_err(|_| "NVIDIA note descriptor size does not fit in memory".to_owned())?;
    if read_u32(section, 8)? != NVIDIA_NOTE_TYPE {
        return Err(".note.nv.tkinfo has an unexpected NVIDIA note type".to_owned());
    }
    let name_start = 12_usize;
    let name_end = name_start
        .checked_add(name_size)
        .ok_or_else(|| "NVIDIA note owner range overflows".to_owned())?;
    let descriptor_start = align_up_4(name_end)?;
    let descriptor_end = descriptor_start
        .checked_add(descriptor_size)
        .ok_or_else(|| "NVIDIA note descriptor range overflows".to_owned())?;
    let note_end = align_up_4(descriptor_end)?;
    if note_end != section.len() {
        return Err(".note.nv.tkinfo must contain exactly one bounded NVIDIA note".to_owned());
    }
    let owner = section
        .get(name_start..name_end)
        .ok_or_else(|| "NVIDIA note owner is out of bounds".to_owned())?;
    if owner != NVIDIA_OWNER {
        return Err(".note.nv.tkinfo has an unexpected owner".to_owned());
    }
    let descriptor = section
        .get(descriptor_start..descriptor_end)
        .ok_or_else(|| "NVIDIA note descriptor is out of bounds".to_owned())?;
    let tokens = descriptor
        .split(|byte| *byte == 0 || byte.is_ascii_whitespace())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut arch_values = tokens
        .windows(2)
        .filter(|window| window[0] == b"-arch")
        .map(|window| window[1]);
    let arch = arch_values
        .next()
        .ok_or_else(|| ".note.nv.tkinfo has no bounded -arch argument".to_owned())?;
    if arch_values.next().is_some() {
        return Err(".note.nv.tkinfo has multiple -arch arguments".to_owned());
    }
    if arch != target.as_bytes() {
        return Err(format!(
            ".note.nv.tkinfo targets {}, expected {target}",
            String::from_utf8_lossy(arch)
        ));
    }
    Ok(())
}

fn align_up_4(value: usize) -> Result<usize, String> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| "ELF note alignment overflows".to_owned())
}

fn parse_sm_target(target: &str) -> Result<u32, PipelineError> {
    parse_sm_target_value(target).map_err(backend_error)
}

fn parse_sm_target_value(target: &str) -> Result<u32, String> {
    let digits = target
        .strip_prefix("sm_")
        .and_then(|value| value.strip_suffix('a'))
        .ok_or_else(|| format!("unsupported CUTLASS device target {target:?}"))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("unsupported CUTLASS device target {target:?}"));
    }
    digits
        .parse::<u32>()
        .map_err(|_| format!("CUTLASS device target {target:?} is out of range"))
}

fn publish_atomically(path: &Path, bytes: &[u8]) -> Result<(), PipelineError> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
        backend_error(format!(
            "could not create a temporary artifact next to {}: {error}",
            path.display()
        ))
    })?;
    temporary.write_all(bytes).map_err(|error| {
        backend_error(format!(
            "could not stage artifact {}: {error}",
            path.display()
        ))
    })?;
    temporary.as_file().sync_all().map_err(|error| {
        backend_error(format!(
            "could not sync artifact {}: {error}",
            path.display()
        ))
    })?;
    temporary.persist(path).map_err(|error| {
        backend_error(format!(
            "could not publish artifact {}: {}",
            path.display(),
            error.error
        ))
    })?;
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let value: [u8; 2] = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| "truncated binary field".to_owned())?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(value))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated binary field".to_owned())?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(value))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value: [u8; 8] = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "truncated binary field".to_owned())?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(value))
}

fn nul_terminated(bytes: &[u8], offset: usize) -> Result<&str, String> {
    let tail = bytes
        .get(offset..)
        .ok_or_else(|| "section name offset is out of bounds".to_owned())?;
    let end = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| "unterminated ELF section name".to_owned())?;
    std::str::from_utf8(&tail[..end]).map_err(|_| "ELF section name is not UTF-8".to_owned())
}

fn backend_error(message: impl Into<String>) -> PipelineError {
    PipelineError::BackendB(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::write::{Object as WriteObject, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
    };

    fn pair_abi() -> Vec<KernelParameterKind> {
        vec![
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
        ]
    }

    fn abi_map(kernel: &str, parameters: Vec<KernelParameterKind>) -> ExpectedKernelAbis {
        BTreeMap::from([(kernel.to_owned(), parameters)])
    }

    fn parameter_info(parameters: &[KernelParameterKind]) -> Vec<u8> {
        let mut info = Vec::new();
        let mut offset = 0_u16;
        let mut offsets = Vec::new();
        for parameter in parameters {
            let mask = parameter.alignment() - 1;
            offset = (offset + mask) & !mask;
            offsets.push(offset);
            offset += parameter.size();
        }
        for (ordinal, parameter) in parameters.iter().copied().enumerate().rev() {
            info.extend_from_slice(&[4, 0x17]);
            info.extend_from_slice(&12_u16.to_le_bytes());
            info.extend_from_slice(&0_u32.to_le_bytes());
            info.extend_from_slice(&(ordinal as u16).to_le_bytes());
            info.extend_from_slice(&offsets[ordinal].to_le_bytes());
            let flags = (u32::from(parameter.size()) << 18)
                | (0x1f << 12)
                | (parameter.cuda_parameter_space() << 8);
            info.extend_from_slice(&flags.to_le_bytes());
        }
        info.extend_from_slice(&[3, 0x19]);
        info.extend_from_slice(&offset.to_le_bytes());
        info
    }

    fn minimal_cubin_with_arches(
        kernel: &str,
        sm: u32,
        arches: &[&str],
        parameters: &[KernelParameterKind],
    ) -> Vec<u8> {
        let names = format!("\0.shstrtab\0.text.{kernel}\0.note.nv.tkinfo\0.nv.info.{kernel}\0")
            .into_bytes();
        let text_name = format!(".text.{kernel}");
        let info_name = format!(".nv.info.{kernel}");
        let text_name_offset = names
            .windows(text_name.len())
            .position(|window| window == text_name.as_bytes())
            .unwrap();
        let note_name_offset = names
            .windows(b".note.nv.tkinfo".len())
            .position(|window| window == b".note.nv.tkinfo")
            .unwrap();
        let info_name_offset = names
            .windows(info_name.len())
            .position(|window| window == info_name.as_bytes())
            .unwrap();
        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(b"-O\0");
        for arch in arches {
            descriptor.extend_from_slice(b"-arch\0");
            descriptor.extend_from_slice(arch.as_bytes());
            descriptor.push(0);
        }
        let mut note = Vec::new();
        note.extend_from_slice(&12_u32.to_le_bytes());
        note.extend_from_slice(&(descriptor.len() as u32).to_le_bytes());
        note.extend_from_slice(&0x7d0_u32.to_le_bytes());
        note.extend_from_slice(b"NVIDIA Corp\0");
        note.extend_from_slice(&descriptor);
        while note.len() % 4 != 0 {
            note.push(0);
        }
        let info = parameter_info(parameters);

        let names_offset = 64_usize;
        let code_offset = (names_offset + names.len() + 7) & !7;
        let note_offset = (code_offset + 8 + 3) & !3;
        let info_offset = (note_offset + note.len() + 3) & !3;
        let section_offset = (info_offset + info.len() + 7) & !7;
        let mut bytes = vec![0_u8; section_offset + 5 * 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[40..48].copy_from_slice(&(section_offset as u64).to_le_bytes());
        bytes[48..52].copy_from_slice(&(sm << 8).to_le_bytes());
        bytes[52..54].copy_from_slice(&64_u16.to_le_bytes());
        bytes[58..60].copy_from_slice(&64_u16.to_le_bytes());
        bytes[60..62].copy_from_slice(&5_u16.to_le_bytes());
        bytes[62..64].copy_from_slice(&1_u16.to_le_bytes());
        bytes[names_offset..names_offset + names.len()].copy_from_slice(&names);
        bytes[code_offset..code_offset + 8].copy_from_slice(&[1; 8]);
        bytes[note_offset..note_offset + note.len()].copy_from_slice(&note);
        bytes[info_offset..info_offset + info.len()].copy_from_slice(&info);

        let names_header = section_offset + 64;
        bytes[names_header..names_header + 4].copy_from_slice(&1_u32.to_le_bytes());
        bytes[names_header + 4..names_header + 8].copy_from_slice(&3_u32.to_le_bytes());
        bytes[names_header + 24..names_header + 32]
            .copy_from_slice(&(names_offset as u64).to_le_bytes());
        bytes[names_header + 32..names_header + 40]
            .copy_from_slice(&(names.len() as u64).to_le_bytes());

        let text_header = section_offset + 128;
        bytes[text_header..text_header + 4]
            .copy_from_slice(&(text_name_offset as u32).to_le_bytes());
        bytes[text_header + 4..text_header + 8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[text_header + 24..text_header + 32]
            .copy_from_slice(&(code_offset as u64).to_le_bytes());
        bytes[text_header + 32..text_header + 40].copy_from_slice(&8_u64.to_le_bytes());

        let note_header = section_offset + 192;
        bytes[note_header..note_header + 4]
            .copy_from_slice(&(note_name_offset as u32).to_le_bytes());
        bytes[note_header + 4..note_header + 8].copy_from_slice(&7_u32.to_le_bytes());
        bytes[note_header + 24..note_header + 32]
            .copy_from_slice(&(note_offset as u64).to_le_bytes());
        bytes[note_header + 32..note_header + 40]
            .copy_from_slice(&(note.len() as u64).to_le_bytes());
        bytes[note_header + 48..note_header + 56].copy_from_slice(&4_u64.to_le_bytes());

        let info_header = section_offset + 256;
        bytes[info_header..info_header + 4]
            .copy_from_slice(&(info_name_offset as u32).to_le_bytes());
        bytes[info_header + 4..info_header + 8].copy_from_slice(&0x7000_0000_u32.to_le_bytes());
        bytes[info_header + 24..info_header + 32]
            .copy_from_slice(&(info_offset as u64).to_le_bytes());
        bytes[info_header + 32..info_header + 40]
            .copy_from_slice(&(info.len() as u64).to_le_bytes());
        bytes[info_header + 48..info_header + 56].copy_from_slice(&4_u64.to_le_bytes());
        bytes
    }

    fn minimal_cubin(kernel: &str, sm: u32) -> Vec<u8> {
        minimal_cubin_with_arches(kernel, sm, &["sm_120a"], &pair_abi())
    }

    fn fatbin(cubin: &[u8], sm: u32) -> Vec<u8> {
        const HEADER: usize = 16;
        const CODE_OFFSET: usize = 64;
        let payload_size = CODE_OFFSET + cubin.len();
        let mut bytes = vec![0_u8; HEADER + payload_size];
        bytes[..4].copy_from_slice(&0xba55_ed50_u32.to_le_bytes());
        bytes[4..6].copy_from_slice(&1_u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&(HEADER as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&(payload_size as u64).to_le_bytes());
        bytes[HEADER..HEADER + 2].copy_from_slice(&2_u16.to_le_bytes());
        bytes[HEADER + 2..HEADER + 4].copy_from_slice(&0x0101_u16.to_le_bytes());
        bytes[HEADER + 4..HEADER + 8].copy_from_slice(&(CODE_OFFSET as u32).to_le_bytes());
        bytes[HEADER + 8..HEADER + 16].copy_from_slice(&(cubin.len() as u64).to_le_bytes());
        bytes[HEADER + 28..HEADER + 32].copy_from_slice(&sm.to_le_bytes());
        bytes[HEADER + CODE_OFFSET..].copy_from_slice(cubin);
        bytes
    }

    fn host_object(image: &[u8]) -> Vec<u8> {
        let mut object =
            WriteObject::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let section =
            object.add_section(Vec::new(), b".lrodata".to_vec(), SectionKind::ReadOnlyData);
        object.section_mut(section).set_data(image.to_vec(), 8);
        object.add_symbol(Symbol {
            name: b"kernels_binary".to_vec(),
            value: 0,
            size: image.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: SymbolSection::Section(section),
            flags: SymbolFlags::None,
        });
        object.write().unwrap()
    }

    #[test]
    fn object_symbol_and_fatbinary_are_validated_end_to_end() {
        let cubin = minimal_cubin("add_f32", 120);
        let image = fatbin(&cubin, 120);
        let object = host_object(&image);
        let extracted = extract_kernels_binary(&object).unwrap();
        assert_eq!(extracted, image);
        let (normalized, source_kind) =
            extract_validated_cubin(&extracted, "sm_120a", &abi_map("add_f32", pair_abi()))
                .unwrap();
        assert_eq!(normalized, cubin);
        assert_eq!(source_kind, "a CUDA fatbinary");
    }

    #[test]
    fn validation_rejects_wrong_target_missing_kernel_and_compression() {
        let cubin = minimal_cubin("add_f32", 120);
        let mut image = fatbin(&cubin, 120);
        assert!(
            extract_validated_cubin(&image, "sm_100a", &abi_map("add_f32", pair_abi())).is_err()
        );
        assert!(
            extract_validated_cubin(&image, "sm_120a", &abi_map("add_f16", pair_abi())).is_err()
        );
        image[16 + 40..16 + 48].copy_from_slice(&0x1000_u64.to_le_bytes());
        assert!(
            extract_validated_cubin(&image, "sm_120a", &abi_map("add_f32", pair_abi())).is_err()
        );
    }

    #[test]
    fn validation_requires_the_exact_blackwell_target_variant_note() {
        let wrong_variant = minimal_cubin_with_arches("add_f32", 120, &["sm_120f"], &pair_abi());
        let error = validate_cuda_cubin(&wrong_variant, "sm_120a", &abi_map("add_f32", pair_abi()))
            .unwrap_err();
        assert!(error.contains("sm_120f"), "{error}");

        let duplicate =
            minimal_cubin_with_arches("add_f32", 120, &["sm_120a", "sm_120a"], &pair_abi());
        let error = validate_cuda_cubin(&duplicate, "sm_120a", &abi_map("add_f32", pair_abi()))
            .unwrap_err();
        assert!(error.contains("multiple -arch"), "{error}");

        let mut missing = minimal_cubin("add_f32", 120);
        let note_name = missing
            .windows(b".note.nv.tkinfo".len())
            .position(|window| window == b".note.nv.tkinfo")
            .unwrap();
        missing[note_name] = b'_';
        let error =
            validate_cuda_cubin(&missing, "sm_120a", &abi_map("add_f32", pair_abi())).unwrap_err();
        assert!(error.contains("missing .note.nv.tkinfo"), "{error}");
    }

    #[test]
    fn flattened_elementwise_and_gemv_signatures_have_fixed_u64_slots() {
        let elementwise = r#"
            "func.func"() <{function_type = (!llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> (), sym_name = "add_f32"}> ({
            "func.func"() <{function_type = (!llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64) -> (), sym_name = "add_f16"}> ({
        "#;
        let elementwise_abis =
            expected_kernel_abis(elementwise, &["add_f32".to_owned(), "add_f16".to_owned()])
                .unwrap();
        let elementwise_parameters = vec![
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
        ];
        assert_eq!(elementwise_abis["add_f32"], elementwise_parameters);
        assert_eq!(elementwise_abis["add_f16"], elementwise_parameters);

        let gemv = r#"
            "func.func"() <{function_type = (!llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> (), sym_name = "nvfp4_gemv"}> ({
        "#;
        let gemv_abis = expected_kernel_abis(gemv, &["nvfp4_gemv".to_owned()]).unwrap();
        assert_eq!(gemv_abis["nvfp4_gemv"].len(), 12);
        assert_eq!(
            gemv_abis["nvfp4_gemv"],
            vec![
                KernelParameterKind::GlobalPointer64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::GlobalPointer64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::GlobalPointer64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::GlobalPointer64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::GlobalPointer64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::Scalar64,
                KernelParameterKind::Scalar64,
            ]
        );
    }

    #[test]
    fn cubin_parameter_metadata_matches_elementwise_and_gemv_abis() {
        let elementwise = vec![
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
        ];
        let cubin = minimal_cubin_with_arches("add_f32", 120, &["sm_120a"], &elementwise);
        validate_cuda_cubin(&cubin, "sm_120a", &abi_map("add_f32", elementwise.clone())).unwrap();

        let gemv = vec![
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::GlobalPointer64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::Scalar64,
            KernelParameterKind::Scalar64,
        ];
        let cubin = minimal_cubin_with_arches("nvfp4_gemv", 120, &["sm_120a"], &gemv);
        validate_cuda_cubin(&cubin, "sm_120a", &abi_map("nvfp4_gemv", gemv.clone())).unwrap();

        let mut malformed = parameter_info(&elementwise);
        // The first record describes ordinal 5 at offset 40. Make that slot
        // overlap ordinal 0; the post-object ABI validator must fail closed.
        malformed[10..12].copy_from_slice(&0_u16.to_le_bytes());
        let error = validate_kernel_parameter_abi("add_f32", &malformed, &elementwise).unwrap_err();
        assert!(error.contains("offset/size/space"), "{error}");
    }

    #[test]
    fn flattened_signature_parser_rejects_unsupported_or_ambiguous_abis() {
        let unsupported =
            r#""func.func"() <{function_type = (!llvm.ptr, i32) -> (), sym_name = "kernel"}> ({"#;
        let error = expected_kernel_abis(unsupported, &["kernel".to_owned()]).unwrap_err();
        assert!(error.contains("unsupported flattened parameter"), "{error}");

        let duplicate = format!("{unsupported}\n{unsupported}").replace("i32", "i64");
        let error = expected_kernel_abis(&duplicate, &["kernel".to_owned()]).unwrap_err();
        assert!(error.contains("found 2"), "{error}");
    }

    #[test]
    fn profile_and_target_are_fail_closed() {
        assert_eq!(
            CutlassBackendConfig::new("/tmp/compiler.so".into()).profile,
            CUTLASS_FULL_CUTE_MLIR22_PROFILE
        );
        assert_eq!(parse_sm_target_value("sm_120a").unwrap(), 120);
        assert!(parse_sm_target_value("sm_120").is_err());
        assert!(parse_sm_target_value("compute_120").is_err());
    }
}
