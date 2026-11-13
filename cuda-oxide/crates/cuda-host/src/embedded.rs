/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle,
    artifact_bundles_from_binary_path, artifact_bundles_from_current_exe,
    embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use oxide_artifacts::ArtifactEntryKind;
use std::sync::Arc;
use thiserror::Error;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod mapped_image;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// The artifact anchor could not be associated with its original mapped file.
    #[error("cannot read the binary containing the CUDA artifact anchor: {0}")]
    MappedImage(#[source] std::io::Error),

    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the binary that was read.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// PTX source could not be parsed or edited safely while merging bundles.
    #[error("embedded CUDA module '{name}' contains invalid PTX: {reason}")]
    InvalidPtx { name: String, reason: String },

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are linked to an in-memory cubin for their original target.
/// A payload built for a standard pre-Blackwell target, such as `sm_86`, may
/// instead be converted to PTX and JIT-compiled by the driver on Blackwell.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load a named artifact bundle from the binary containing `anchor`.
///
/// Generated non-generic module loaders borrow their artifact anchor, which
/// the linker places in the same image as the bundle. This works for both an
/// executable and a shared library, including a library opened by a relative
/// path before the working directory changes.
///
/// On Linux and Android, this requires readable `/proc/self/maps` and
/// `/proc/self/map_files`. The original mapped file must still be accessible.
/// A missing, replaced, deleted or unreadable image is an error; it never
/// causes a search in another binary. Other operating systems are unsupported.
/// Payload selection and compilation are identical to [`load_embedded_module`].
pub fn load_embedded_module_from_anchor(
    ctx: &Arc<CudaContext>,
    name: &str,
    anchor: &u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_containing(anchor)?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

fn artifact_bundles_containing(
    anchor: &u8,
) -> Result<Vec<OwnedArtifactBundle>, EmbeddedModuleError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let bytes = mapped_image::read(anchor).map_err(EmbeddedModuleError::MappedImage)?;
        oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
            .map_err(cuda_core::EmbeddedModuleError::Artifacts)
            .map_err(Into::into)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = anchor;
        Err(EmbeddedModuleError::MappedImage(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "artifact anchor discovery requires Linux or Android procfs",
        )))
    }
}

/// Merge all PTX bundles from the current executable into a single CUDA module.
///
/// When a generic kernel is monomorphized in a consuming crate, its PTX ends
/// up in that crate's bundle rather than the defining crate's bundle. This
/// function gathers every PTX bundle in the binary, strips duplicate header
/// directives (`.version`, `.target`, `.address_size`) from all but the first
/// bundle, concatenates the bodies, and loads the result as one CUDA module.
/// All kernel symbols are therefore available regardless of which crate bundle
/// they were compiled into.
///
/// Bundles with non-PTX payloads (NVVM IR, LTOIR, cubin) are skipped; use
/// `load_embedded_module` for those.
pub fn load_all_ptx_bundles_merged(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_from_current_exe()?;
    let merged = merge_ptx_bundles(&bundles)?;

    let module = ctx.load_module_from_image(merged.as_bytes())?;
    // Retain the merged module's `.entry` names (a few dozen bytes per
    // kernel). If a later `_TID_` generic-kernel lookup misses while a
    // same-base entry exists under a different hash, the launch paths can
    // then report a host/device type-identity naming divergence instead of an
    // opaque "named symbol not found". See `crate::entry_registry`.
    crate::entry_registry::register_merged_module_entries(&module, &merged);
    Ok(module)
}

/// Merge the PTX payloads of `bundles`, in iteration order, into one PTX
/// module string: the first PTX bundle keeps its `.version` / `.target` /
/// `.address_size` header directives, every later one contributes its body
/// with those directives stripped. Bundles without a PTX payload are skipped.
///
/// This is the pure half of [`load_all_ptx_bundles_merged`], exposed so tests
/// and examples can check order-dependent merge properties: module-scope
/// symbol uniqueness and extern alignment must hold for every bundle order,
/// not just the one the current executable happens to embed (#1277).
pub fn merge_ptx_bundles<'a>(
    bundles: impl IntoIterator<Item = &'a OwnedArtifactBundle>,
) -> Result<String, EmbeddedModuleError> {
    let mut merged = String::new();
    let mut found_any = false;

    // Keep each bundle's debug file indices, helper `.func` names, bundle-local
    // labels, and coalesced DWARF `.section` blocks distinct in the merged module.
    let mut file_index_offset = 0;
    let mut seen_functions = std::collections::HashSet::new();
    let mut section_order: Vec<String> = Vec::new();
    let mut section_bodies: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut abbrev_byte_offset = 0u64;
    let mut info_byte_offset = 0u64;

    for (bundle_index, bundle) in bundles.into_iter().enumerate() {
        if let Some(ptx_bytes) = bundle.payload(ArtifactPayloadKind::Ptx) {
            let ptx_str = std::str::from_utf8(ptx_bytes)
                .map_err(|_| EmbeddedModuleError::UnsupportedPayload {
                    name: bundle.name.clone(),
                })?
                .trim_end_matches('\0');

            let strip_headers = found_any;
            let bundle_file_offset = file_index_offset;
            let (body, max_file_index, sections) = prepare_bundle_body_with_functions(
                ptx_str,
                strip_headers,
                bundle_file_offset,
                bundle_index,
                &mut seen_functions,
            )
            .map_err(|reason| EmbeddedModuleError::InvalidPtx {
                name: bundle.name.clone(),
                reason,
            })?;
            file_index_offset = file_index_offset.max(max_file_index);
            merged.push_str(&body);
            if !body.ends_with('\n') {
                merged.push('\n');
            }

            let this_abbrev_bytes = sections
                .iter()
                .find(|(name, _)| name == ".debug_abbrev")
                .map(|(_, sec_body)| debug_section_byte_size(sec_body))
                .unwrap_or(0);
            let this_info_bytes = sections
                .iter()
                .find(|(name, _)| name == ".debug_info")
                .map(|(_, sec_body)| debug_section_byte_size(sec_body))
                .unwrap_or(0);

            for (sec_name, sec_body) in sections {
                let adjusted = adjust_debug_section_body(
                    &sec_name,
                    &sec_body,
                    abbrev_byte_offset,
                    info_byte_offset,
                    bundle_file_offset,
                );
                if !section_bodies.contains_key(&sec_name) {
                    section_order.push(sec_name.clone());
                }
                let entry = section_bodies.entry(sec_name).or_default();
                if !adjusted.trim().is_empty() {
                    if !entry.is_empty() && !entry.ends_with('\n') {
                        entry.push('\n');
                    }
                    entry.push_str(adjusted.trim_matches('\n'));
                    entry.push('\n');
                }
            }

            abbrev_byte_offset = abbrev_byte_offset.saturating_add(this_abbrev_bytes);
            info_byte_offset = info_byte_offset.saturating_add(this_info_bytes);
            found_any = true;
        }
    }

    if !found_any {
        return Err(EmbeddedModuleError::NoModules);
    }

    for sec_name in section_order {
        let body = section_bodies.remove(&sec_name).unwrap_or_default();
        if body.trim().is_empty() {
            merged.push_str(&format!("\t.section\t{sec_name}\t{{\t}}\n"));
        } else {
            merged.push_str(&format!("\t.section\t{sec_name}\n\t{{\n{body}\t}}\n"));
        }
    }

    Ok(merged)
}

/// Single-bundle test wrapper around [`prepare_bundle_body_with_functions`]
/// that starts with an empty seen-function set at bundle index zero.
#[cfg(test)]
fn prepare_bundle_body(
    ptx: &str,
    strip_headers: bool,
    file_index_offset: u64,
) -> Result<(String, u64), String> {
    let mut seen_functions = std::collections::HashSet::new();
    let (body, max_file_index, sections) = prepare_bundle_body_with_functions(
        ptx,
        strip_headers,
        file_index_offset,
        0,
        &mut seen_functions,
    )?;
    let mut full = body;
    for (sec_name, sec_body) in sections {
        if !full.is_empty() && !full.ends_with('\n') {
            full.push('\n');
        }
        if sec_body.trim().is_empty() {
            full.push_str(&format!("\t.section\t{sec_name}\t{{\t}}\n"));
        } else {
            let trimmed = sec_body.trim_matches('\n');
            full.push_str(&format!("\t.section\t{sec_name}\n\t{{\n{trimmed}\n\t}}\n"));
        }
    }
    Ok((full, max_file_index))
}

/// Strip repeated headers, disambiguate colliding helper `.func` symbols and
/// per-bundle labels/local depots, extract `.section` blocks for module-level
/// coalescing, and shift debug indices past the previous bundles.
#[allow(clippy::type_complexity)]
fn prepare_bundle_body_with_functions(
    ptx: &str,
    strip_headers: bool,
    file_index_offset: u64,
    bundle_index: usize,
    seen_functions: &mut std::collections::HashSet<String>,
) -> Result<(String, u64, Vec<(String, String)>), String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = ptx_parse::EditScript::new();
    let mut max_file_index = 0u64;

    let mut bundle_functions = std::collections::HashSet::new();
    let mut renamed_symbols = std::collections::HashMap::new();
    for callable in document.callables() {
        if callable.kind() == ptx_parse::CallableKind::Function && !callable.is_extern() {
            let name = callable.name();
            bundle_functions.insert(name.to_string());
            if seen_functions.contains(name) {
                renamed_symbols.insert(name.to_string(), format!("{name}__bundle_{bundle_index}"));
            }
        }
    }
    seen_functions.extend(bundle_functions);

    // Disambiguate bundle-local labels (`$L__*`, etc.) and `.local` stack depots
    // (`__local_depot*`) in subsequent bundles so module-scoped DWARF sections
    // (`.debug_pubnames`, `.debug_pubtypes`, `.debug_loc`, `.debug_info`) never
    // define or reference colliding labels across merged bundles.
    if bundle_index > 0 {
        for label in document.labels() {
            let name = label.name();
            renamed_symbols
                .entry(name.to_string())
                .or_insert_with(|| format!("{name}__bundle_{bundle_index}"));
        }
        let tokens = document.tokens();
        for directive in document.directives() {
            if directive.name() == ".local" {
                let span = directive.arguments_span();
                let first = tokens.partition_point(|token| token.span().end <= span.start);
                let mut local_ident = None;
                for token in tokens[first..]
                    .iter()
                    .take_while(|token| token.span().start < span.end)
                {
                    if token.text(ptx) == "[" {
                        break;
                    }
                    if token.kind() == ptx_parse::TokenKind::Word
                        && !token.text(ptx).starts_with('.')
                        && !token.text(ptx).bytes().all(|b| b.is_ascii_digit())
                    {
                        local_ident = Some(token.text(ptx));
                    }
                }
                if let Some(name) = local_ident {
                    renamed_symbols
                        .entry(name.to_string())
                        .or_insert_with(|| format!("{name}__bundle_{bundle_index}"));
                }
            }
        }
    }

    if !renamed_symbols.is_empty() {
        for token in document.tokens() {
            if token.kind() == ptx_parse::TokenKind::Word
                && let Some(renamed) = renamed_symbols.get(token.text(ptx))
            {
                edits
                    .replace(token.span(), renamed.clone())
                    .map_err(|error| error.to_string())?;
            }
        }
    }

    for directive in document.directives() {
        match directive.name() {
            ".version" | ".target" | ".address_size" if strip_headers => {
                edits
                    .delete(directive.line_span())
                    .map_err(|error| error.to_string())?;
            }
            ".file" | ".loc" => {
                let span = directive.arguments_span();
                let tokens = document.tokens();
                let first = tokens.partition_point(|token| token.span().end <= span.start);
                let arguments: Vec<_> = tokens[first..]
                    .iter()
                    .take_while(|token| token.span().start < span.end)
                    .filter(|token| !token.kind().is_trivia())
                    .collect();
                let index = shift_file_index(
                    &mut edits,
                    ptx,
                    arguments.first().copied(),
                    file_index_offset,
                )?;
                max_file_index = max_file_index.max(index);
                if directive.name() == ".loc" {
                    for (position, attribute) in arguments.windows(2).enumerate() {
                        if attribute[0].text(ptx) == "," && attribute[1].text(ptx) == "inlined_at" {
                            let index = shift_file_index(
                                &mut edits,
                                ptx,
                                arguments.get(position + 2).copied(),
                                file_index_offset,
                            )?;
                            max_file_index = max_file_index.max(index);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let rewritten = edits.apply(ptx).map_err(|error| error.to_string())?;
    let (body, sections) = extract_ptx_sections(&rewritten)?;
    Ok((body, max_file_index, sections))
}

/// Extract top-level `.section <name> { ... }` blocks from `ptx` so they can be
/// coalesced into one section per name at the end of a merged PTX module.
#[allow(clippy::type_complexity)]
fn extract_ptx_sections(ptx: &str) -> Result<(String, Vec<(String, String)>), String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut section_deletes = ptx_parse::EditScript::new();
    let mut sections = Vec::new();
    let bytes = ptx.as_bytes();

    for scope in document.scopes() {
        let Some(header_id) = scope.header() else {
            continue;
        };
        let Some(directive) = document.directive_for_statement(header_id) else {
            continue;
        };
        if directive.name() != ".section" {
            continue;
        }
        let sec_name = directive.arguments().trim().to_string();
        let sec_body = ptx[scope.body_span()].to_string();
        sections.push((sec_name, sec_body));

        let start = directive.line_span().start;
        let mut end = scope.span().end;
        while end < bytes.len() && matches!(bytes[end], b' ' | b'\t' | b'\r') {
            end += 1;
        }
        if end < bytes.len() && bytes[end] == b'\n' {
            end += 1;
        }
        section_deletes
            .delete(start..end)
            .map_err(|error| error.to_string())?;
    }

    if sections.is_empty() {
        return Ok((ptx.to_string(), sections));
    }
    let body = section_deletes
        .apply(ptx)
        .map_err(|error| error.to_string())?;
    Ok((body, sections))
}

/// Compute the byte size of a PTX `.section` body from its `.b8`/`.b16`/`.b32`/`.b64`
/// data directives.
fn debug_section_byte_size(section_body: &str) -> u64 {
    let Ok(document) = ptx_parse::Document::parse(section_body) else {
        return 0;
    };
    let tokens = document.tokens();
    let mut total = 0u64;
    for directive in document.directives() {
        let elem_bytes: u64 = match directive.name() {
            ".b8" => 1,
            ".b16" => 2,
            ".b32" => 4,
            ".b64" => 8,
            _ => continue,
        };
        let span = directive.arguments_span();
        let first = tokens.partition_point(|token| token.span().end <= span.start);
        let mut items = 1u64;
        for token in tokens[first..]
            .iter()
            .take_while(|token| token.span().start < span.end)
        {
            if token.text(section_body) == "," {
                items += 1;
            }
        }
        total = total.saturating_add(elem_bytes.saturating_mul(items));
    }
    total
}

/// Adjust intra-DWARF section offsets and `.file` numbers in a bundle's debug
/// section body before coalescing it with prior bundles' debug sections.
fn adjust_debug_section_body(
    sec_name: &str,
    sec_body: &str,
    abbrev_byte_offset: u64,
    info_byte_offset: u64,
    file_index_offset: u64,
) -> String {
    match sec_name {
        ".debug_info" if abbrev_byte_offset > 0 || file_index_offset > 0 => {
            let mut out = String::with_capacity(sec_body.len() + 32);
            for line in sec_body.lines() {
                let trimmed = line.trim_start();
                if abbrev_byte_offset > 0
                    && let Some(rest) = trimmed.strip_prefix(".b32 .debug_abbrev")
                    && (rest.is_empty()
                        || rest.starts_with(char::is_whitespace)
                        || rest.starts_with('/'))
                {
                    let indent = &line[..line.len() - trimmed.len()];
                    out.push_str(indent);
                    out.push_str(&format!(".b32 .debug_abbrev+{abbrev_byte_offset}{rest}\n"));
                    continue;
                }
                if file_index_offset > 0
                    && (line.contains("DW_AT_decl_file") || line.contains("DW_AT_call_file"))
                    && let Some(rest) = trimmed.strip_prefix(".b8 ")
                {
                    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(file_idx) = digits.parse::<u64>()
                        && file_idx > 0
                    {
                        let indent = &line[..line.len() - trimmed.len()];
                        let suffix = &rest[digits.len()..];
                        let shifted = file_idx.saturating_add(file_index_offset);
                        out.push_str(&format!("{indent}.b8 {shifted}{suffix}\n"));
                        continue;
                    }
                }
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        ".debug_pubnames" | ".debug_pubtypes" if info_byte_offset > 0 => {
            let mut out = String::with_capacity(sec_body.len() + 16);
            for line in sec_body.lines() {
                let trimmed = line.trim_start();
                if let Some(rest) = trimmed.strip_prefix(".b32 .debug_info")
                    && (rest.is_empty()
                        || rest.starts_with(char::is_whitespace)
                        || rest.starts_with('/'))
                {
                    let indent = &line[..line.len() - trimmed.len()];
                    out.push_str(indent);
                    out.push_str(&format!(".b32 .debug_info+{info_byte_offset}{rest}\n"));
                    continue;
                }
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        _ => sec_body.to_string(),
    }
}

/// Shift one complete decimal index token, preserving surrounding PTX.
fn shift_file_index(
    edits: &mut ptx_parse::EditScript,
    ptx: &str,
    token: Option<&ptx_parse::Token>,
    offset: u64,
) -> Result<u64, String> {
    let token = token.ok_or("malformed debug directive: missing file index")?;
    let text = token.text(ptx);
    if token.kind() != ptx_parse::TokenKind::Word || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("malformed debug file index: {text:?}"));
    }
    let index: u64 = text
        .parse()
        .map_err(|_| format!("debug file index out of range: {text:?}"))?;
    let shifted = index
        .checked_add(offset)
        .ok_or("debug file index overflow")?;
    if offset != 0 {
        edits
            .replace(token.span(), shifted.to_string())
            .map_err(|error| error.to_string())?;
    }
    Ok(shifted)
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    let expected_kernels = bundle
        .entries
        .iter()
        .filter(|entry| entry.kind == ArtifactEntryKind::Kernel)
        .map(|entry| entry.symbol.as_str())
        .collect::<Vec<_>>();

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::build_cubin_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
                &expected_kernels,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::build_ptx_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
                &expected_kernels,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::link_ltoir_to_cubin_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
                &expected_kernels,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::link_ltoir_to_ptx_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
                &expected_kernels,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(
    bundle: &OwnedArtifactBundle,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_bundle_with_explicit(bundle, explicit.as_deref())
}

fn target_arch_for_bundle_with_explicit(
    bundle: &OwnedArtifactBundle,
    explicit_target: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    ltoir::resolve_source_target(concrete_bundle_target(&bundle.target)?, explicit_target)
}

fn concrete_bundle_target(
    target: &str,
) -> Result<Option<cuda_artifact_finalizer::CudaArch>, ltoir::LtoirError> {
    match target {
        // Compatibility for artifacts emitted before concrete NVVM targets
        // were recorded. New bundles must never use these sentinels.
        "libdevice" | "nvvm-ir" => Ok(None),
        target => Ok(Some(target.parse()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        assert_eq!(
            concrete_bundle_target("sm_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    fn ptx_bundle(name: &str, ptx: &str) -> OwnedArtifactBundle {
        use oxide_artifacts::OwnedArtifactPayload;
        OwnedArtifactBundle {
            name: name.to_string(),
            target: "sm_80".to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: vec![OwnedArtifactPayload {
                kind: ArtifactPayloadKind::Ptx,
                name: name.to_string(),
                bytes: ptx.as_bytes().to_vec(),
            }],
            entries: Vec::new(),
        }
    }

    /// The merge is order-explicit: exactly one header set (the first PTX
    /// bundle's), every body present in iteration order, non-PTX bundles
    /// skipped. #1277's collision-free-namespace example checks the same
    /// function under both bundle orders at runtime.
    #[test]
    fn merges_bundles_in_iteration_order_with_one_header_set() {
        let first = ptx_bundle(
            "first",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry a() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry b() { ret; }\n",
        );
        let skipped = bundle_with_target("sm_80");

        let forward = merge_ptx_bundles([&first, &skipped, &second]).unwrap();
        assert_eq!(forward.matches(".version").count(), 1);
        assert_eq!(forward.matches(".target sm_80").count(), 1);
        assert!(forward.find(".entry a()").unwrap() < forward.find(".entry b()").unwrap());

        let reversed = merge_ptx_bundles([&second, &first]).unwrap();
        assert_eq!(reversed.matches(".version").count(), 1);
        assert!(reversed.find(".entry b()").unwrap() < reversed.find(".entry a()").unwrap());

        assert!(matches!(
            merge_ptx_bundles([&skipped]),
            Err(EmbeddedModuleError::NoModules)
        ));
    }

    #[test]
    fn strips_only_structural_ptx_module_headers() {
        let ptx = "\
// .target sm_1
.version 8.9
.target sm_120a, debug
.address_size 64
.visible .entry kernel() { ret; }
";
        let (body, max_file_index) = prepare_bundle_body(ptx, true, 0).unwrap();
        assert_eq!(body, "// .target sm_1\n.visible .entry kernel() { ret; }\n");
        assert_eq!(max_file_index, 0);
    }

    #[test]
    fn debug_indices_ignore_comments_paths_and_label_substrings() {
        let ptx = ".file /* 90 */ 1 \"source12.rs\"\n.loc /* 80 */ 1 2 3, function_name $inlined_at7, inlined_at /* 70 */ 1 4 5 // inlined_at 60\n";
        let (body, max_index) = prepare_bundle_body(ptx, false, 3).unwrap();
        assert_eq!(
            body,
            ".file /* 90 */ 4 \"source12.rs\"\n.loc /* 80 */ 4 2 3, function_name $inlined_at7, inlined_at /* 70 */ 4 4 5 // inlined_at 60\n"
        );
        assert_eq!(max_index, 4);
    }

    #[test]
    fn malformed_debug_indices_and_overflow_return_errors() {
        for ptx in [
            ".file \"source12.rs\"\n",
            ".file /* 12 */\n",
            ".file -1 \"source.rs\"\n",
            ".file 1foo \"source.rs\"\n",
            ".loc 1 2 3, inlined_at\n",
            ".loc 1 2 3, inlined_at -1 4 5\n",
            ".file 18446744073709551616 \"source.rs\"\n",
        ] {
            assert!(prepare_bundle_body(ptx, false, 1).is_err(), "{ptx}");
        }
        assert!(
            prepare_bundle_body(".file 18446744073709551615 \"source.rs\"\n", false, 1)
                .unwrap_err()
                .contains("overflow")
        );
    }

    #[test]
    fn merged_bundles_renumber_debug_file_indices_per_bundle() {
        let first = ptx_bundle(
            "first",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"a/lib.rs\"
.file 2 \"shared/thread.rs\"
.visible .entry a()
{
\t.loc\t1 10 5
\tret;
}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"b/main.rs\"
.visible .entry b()
{
\t.loc\t1 3 1, function_name $f, inlined_at 1 7 0
\tret;
}
",
        );
        let plain = ptx_bundle(
            "plain",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry c() { ret; }\n",
        );

        let merged = merge_ptx_bundles([&first, &plain, &second]).unwrap();

        // First bundle's table and references are untouched.
        assert!(merged.contains(".file 1 \"a/lib.rs\""));
        assert!(merged.contains(".file 2 \"shared/thread.rs\""));
        assert!(merged.contains(".loc\t1 10 5"));

        // The debug-free middle bundle does not advance the offset; the
        // second debug bundle continues after the first's maximum of 2 —
        // declaration, `.loc` reference, and `inlined_at` reference alike.
        assert!(merged.contains(".file 3 \"b/main.rs\""));
        assert!(merged.contains(".loc\t3 3 1, function_name $f, inlined_at 3 7 0"));

        // No index is declared twice across the merged module.
        let declared: Vec<&str> = merged
            .lines()
            .filter(|line| line.trim_start().starts_with(".file"))
            .map(|line| line.split_whitespace().nth(1).unwrap())
            .collect();
        let mut deduped = declared.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            declared.len(),
            deduped.len(),
            "duplicate .file index: {declared:?}"
        );
    }

    #[test]
    fn merged_bundles_disambiguate_colliding_helper_functions() {
        let first = ptx_bundle(
            "first",
            "\
.version 8.9
.target sm_80, debug
.address_size 64
.visible .func (.param .b64 retval) shared_helper()
{
\tret;
}
.visible .entry a()
{
\tcall.uni (r0), shared_helper, ();
\tret;
}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 8.9
.target sm_80, debug
.address_size 64
.visible .func (.param .b64 retval) shared_helper()
{
\tret;
}
.visible .entry b()
{
\tcall.uni (r0), shared_helper, ();
\tret;
}
",
        );

        let merged = merge_ptx_bundles([&first, &second]).unwrap();
        assert_eq!(
            merged
                .matches(".visible .func (.param .b64 retval) shared_helper()")
                .count(),
            1
        );
        assert_eq!(
            merged
                .matches(".visible .func (.param .b64 retval) shared_helper__bundle_1()")
                .count(),
            1
        );
        assert!(merged.contains("call.uni (r0), shared_helper__bundle_1, ();"));
    }

    #[test]
    fn merged_bundles_coalesce_debug_sections_and_disambiguate_labels_and_local_depots() {
        let first = ptx_bundle(
            "first",
            "\
.version 7.8
.target sm_90, debug
.address_size 64
.file 1 \"a/src/lib.rs\"
.visible .entry kernel_a()
{
\t.local .align 8 .b8 \t__local_depot0[16];
$L__func_begin0:
\t.loc\t1 10 5
\tmov.b64 \t%SPL, __local_depot0;
\tret;
$L__func_end0:
}
\t.section\t.debug_abbrev
\t{
.b8 1
.b8 17
.b8 0
\t}
\t.section\t.debug_info
\t{
.b32 22
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1                                   // DW_AT_decl_file
.b64 $L__func_begin0
.b64 __local_depot0
\t}
\t.section\t.debug_pubnames
\t{
.b32 $L__pubNames_end0-$L__pubNames_start0
$L__pubNames_start0:
.b8 2
.b8 0
.b32 .debug_info
.b32 26
.b32 0
$L__pubNames_end0:
\t}
\t.section\t.debug_macinfo\t{\t}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 7.8
.target sm_90, debug
.address_size 64
.file 1 \"b/src/main.rs\"
.visible .entry kernel_b()
{
\t.local .align 8 .b8 \t__local_depot0[16];
$L__func_begin0:
\t.loc\t1 20 5
\tmov.b64 \t%SPL, __local_depot0;
\tret;
$L__func_end0:
}
\t.section\t.debug_abbrev
\t{
.b8 1
.b8 17
.b8 0
\t}
\t.section\t.debug_info
\t{
.b32 22
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1                                   // DW_AT_decl_file
.b64 $L__func_begin0
.b64 __local_depot0
\t}
\t.section\t.debug_pubnames
\t{
.b32 $L__pubNames_end0-$L__pubNames_start0
$L__pubNames_start0:
.b8 2
.b8 0
.b32 .debug_info
.b32 26
.b32 0
$L__pubNames_end0:
\t}
\t.section\t.debug_macinfo\t{\t}
",
        );

        let merged = merge_ptx_bundles([&first, &second]).unwrap();

        // Each DWARF section appears once at the end of the merged module.
        for sec in [
            ".section\t.debug_abbrev",
            ".section\t.debug_info",
            ".section\t.debug_pubnames",
            ".section\t.debug_macinfo",
        ] {
            assert_eq!(merged.matches(sec).count(), 1, "duplicate {sec}:\n{merged}");
        }

        // Second bundle's labels and local stack depot are disambiguated.
        assert!(merged.contains("$L__func_begin0__bundle_1:"));
        assert!(merged.contains(".b64 $L__func_begin0__bundle_1"));
        assert!(merged.contains("__local_depot0__bundle_1[16];"));
        assert!(merged.contains(".b64 __local_depot0__bundle_1"));
        assert!(merged.contains("$L__pubNames_start0__bundle_1:"));
        assert!(merged.contains("$L__pubNames_end0__bundle_1-$L__pubNames_start0__bundle_1"));

        // Second CU's abbrev offset, pubnames CU offset, and DW_AT_decl_file index are shifted.
        assert!(merged.contains(".b32 .debug_abbrev+3"));
        assert!(merged.contains(".b32 .debug_info+28"));
        assert!(merged.contains(".b8 2                                   // DW_AT_decl_file"));
    }

    #[test]
    fn target_arch_uses_bundle_compute_target() {
        assert_eq!(
            concrete_bundle_target("compute_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        // Older bundles used these names instead of recording a concrete
        // architecture. New bundles always record a validated `sm_*` target.
        for legacy in ["libdevice", "nvvm-ir"] {
            assert_eq!(concrete_bundle_target(legacy).unwrap(), None);
        }
    }

    #[test]
    fn legacy_bundle_without_recorded_target_requires_explicit_target() {
        for sentinel in ["libdevice", "nvvm-ir"] {
            let bundle = bundle_with_target(sentinel);
            let error = target_arch_for_bundle_with_explicit(&bundle, None)
                .expect_err("the original build target is required");
            assert!(matches!(error, ltoir::LtoirError::TargetNotFound));

            let asserted =
                target_arch_for_bundle_with_explicit(&bundle, Some("compute_86")).unwrap();
            assert_eq!(asserted.sm(), "sm_86");
        }
    }

    #[test]
    fn recorded_bundle_target_overrides_explicit_environment_target() {
        let bundle = bundle_with_target("sm_90");
        let selected = target_arch_for_bundle_with_explicit(&bundle, Some("sm_120")).unwrap();
        assert_eq!(selected.sm(), "sm_90");
    }

    #[test]
    fn target_arch_rejects_malformed_bundle_target() {
        assert!(concrete_bundle_target("sm_90x").is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn bundles_of_an_executable_anchor_match_the_current_exe() {
        static ANCHOR: u8 = 0;
        let by_anchor = artifact_bundles_containing(&ANCHOR).unwrap();
        let by_current_exe = artifact_bundles_from_current_exe().unwrap();
        assert_eq!(by_anchor, by_current_exe);
    }
}
