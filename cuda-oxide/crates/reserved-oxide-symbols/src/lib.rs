// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # `reserved-oxide-symbols` — INTERNAL workspace crate
//!
//! Single source of truth for the internal names that the procedural macros,
//! codegen backend, MIR-lowering, and LLVM-export passes exchange.
//!
//! ## Not a public API
//!
//! This crate is `publish = false` and exists only to keep the macro side
//! and the consumer side of the cuda-oxide naming contract in lockstep.
//! The constants, builders, and predicates exposed here may change
//! without notice between commits. External consumers should depend on
//! `cuda-host`, `cuda-device`, or `cuda-macros` instead.
//!
//! ## What this crate owns
//!
//! The `cuda_oxide_*` namespace, reserved for cuda-oxide internal symbols and
//! compiler attributes.
//! Every symbol prefix below contains the component `246e25db_`, which is
//! `sha256("cuda_oxide_ + rust")` truncated to 8 hex chars. The hash
//! makes accidental collisions effectively impossible — nobody writes
//! `fn cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_foo()` by accident.
//! Internal operation-attribute keys also belong here; they are not link symbols.
//!
//! ## Layered API
//!
//! - **Constants** ([`KERNEL_PREFIX`] etc.) — the raw prefix strings.
//! - **Builders** ([`kernel_symbol`] etc.) — for the macro side.
//! - **Predicates and extractors** ([`is_kernel_symbol`],
//!   [`kernel_base_name`], etc.) — for the consumer side.
//!
//! The Layer-3 helpers hide the substring matching that used to be
//! duplicated across `rustc-codegen-cuda`, `llvm-export`, and `mir-lower`.
//!
//! ## Mutual exclusion guarantee
//!
//! [`DEVICE_PREFIX`] and [`DEVICE_EXTERN_PREFIX`] are mutually exclusive
//! substrings: a symbol containing one cannot contain the other. This is
//! a property of the hash suffix and is verified by a unit test below.
//! Consumers therefore do **not** need the historical
//! `contains(DEVICE_PREFIX) && !contains(DEVICE_EXTERN_PREFIX)` ordering
//! dance — [`is_device_symbol`] handles the disambiguation internally.

#![no_std]
#![doc(hidden)]

extern crate alloc;

use alloc::format;
use alloc::string::String;

// ============================================================================
// Layer 1 — raw constants
// ============================================================================

/// Cargo-visible identity for settings that can change device code or its
/// embedded artifact. Device-owning procedural macros track this variable so
/// Cargo invalidates those crates without invalidating unrelated host crates.
pub const CODEGEN_FINGERPRINT_ENV: &str = "CUDA_OXIDE_INTERNAL_CODEGEN_FINGERPRINT";

/// Internal cargo-oxide/backend opt-in for build-time cubin materialization.
pub const MATERIALIZE_CUBIN_ENV: &str = "CUDA_OXIDE_MATERIALIZE_CUBIN";

/// Exact CUDA compiler/linker provenance discovered by cargo-oxide and checked
/// again by the codegen backend before materialization.
pub const MATERIALIZER_PROVENANCE_ENV: &str = "CUDA_OXIDE_INTERNAL_MATERIALIZER_PROVENANCE";

/// Versioned descriptor-identity handoff accompanying
/// [`MATERIALIZER_PROVENANCE_ENV`]. This is an acceleration hint; generated
/// code remains keyed by the content-derived provenance digest.
pub const MATERIALIZER_HANDSHAKE_ENV: &str = "CUDA_OXIDE_INTERNAL_MATERIALIZER_HANDSHAKE";

/// Optional comma-separated filter selecting crates that may own device code.
pub const DEVICE_CODEGEN_CRATE_ENV: &str = "CUDA_OXIDE_DEVICE_CODEGEN_CRATE";

/// Reserved root that prefixes every cuda-oxide internal symbol.
///
/// User code must not define functions whose name starts with this.
/// The `#[kernel]` and `#[device]` proc macros enforce this rule at
/// the source-code level by checking `name.starts_with(RESERVED_ROOT)`
/// and emitting a compile error.
pub const RESERVED_ROOT: &str = "cuda_oxide_";

/// MIR function-attribute prefix carrying a grid-constant pointee type.
pub const MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX: &str = "cuda_oxide_grid_constant_pointee_";

/// MIR function-attribute prefix carrying a grid-constant pointee alignment.
pub const MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX: &str = "cuda_oxide_grid_constant_align_";

/// LLVM-dialect function-attribute prefix carrying a grid-constant pointee type.
pub const LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX: &str = "cuda_oxide_param_grid_constant_pointee_";

/// LLVM-dialect function-attribute prefix carrying a grid-constant pointee alignment.
pub const LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX: &str = "cuda_oxide_param_grid_constant_align_";

/// Magic component embedded in every prefix to defend against accidental name
/// collisions in user code.
///
/// `sha256("cuda_oxide_ + rust")` truncated to 8 hex chars. The exact
/// value is irrelevant — what matters is that it's fixed forever and
/// no human would ever type it as part of a regular function name.
pub const HASH_SUFFIX: &str = "246e25db";

/// Prefix added to `#[kernel]` functions for collector detection.
///
/// `#[kernel] fn vecadd(...)` becomes
/// `fn cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_vecadd(...)`.
/// The collector finds these by name; the PTX entry name itself is the
/// unprefixed base (e.g., `vecadd`).
///
/// The `codegen_v1` component is a cache-protocol handshake. It lets a newer
/// backend reject pre-tracked-env macro expansions instead of silently reusing
/// device artifacts across output or architecture changes. The complete
/// legacy prefix remains embedded after that tag so an explicitly configured
/// older backend still recognizes and compiles a new root.
pub const KERNEL_PREFIX: &str = "cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_";

/// Kernel prefix emitted before the scoped Cargo cache protocol existed.
///
/// Consumers continue to recognize it so the backend can produce an explicit
/// compatibility diagnostic. New macro expansions must never emit it.
pub const LEGACY_KERNEL_PREFIX: &str = "cuda_oxide_kernel_246e25db_";

/// Prefix added to `#[device]` functions for collector detection.
///
/// `#[device] fn helper(...)` becomes
/// `fn cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_helper(...)`.
/// The LLVM-export layer strips this prefix to produce clean device-side
/// symbol names in the final PTX/LTOIR. As with [`KERNEL_PREFIX`], the complete
/// legacy prefix remains embedded for older-backend compatibility.
pub const DEVICE_PREFIX: &str = "cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_";

/// Device-function prefix emitted before the scoped Cargo cache protocol.
pub const LEGACY_DEVICE_PREFIX: &str = "cuda_oxide_device_246e25db_";

/// Prefix added to functions inside `#[device] extern "C" { ... }` blocks.
///
/// `#[device] extern "C" { fn foo(); }` becomes
/// `fn cuda_oxide_device_extern_246e25db_foo();`. The MIR-lowering pass
/// strips this prefix when emitting the LLVM `declare` so that the LTOIR
/// linker resolves against the original (unprefixed) external symbol.
pub const DEVICE_EXTERN_PREFIX: &str = "cuda_oxide_device_extern_246e25db_";

/// Prefix added to closure-monomorphization helper functions generated
/// by `#[kernel]` for kernels that take a closure parameter.
///
/// The `cuda_launch!` macro calls these helpers to force monomorphization
/// of a generic kernel against a specific closure type at the launch site,
/// without actually invoking the kernel on the host.
pub const INSTANTIATE_PREFIX: &str = "cuda_oxide_instantiate_246e25db_";

/// Prefix added to `#[constant]` statics for codegen detection and
/// host-side `cuModuleGetGlobal` lookup.
///
/// `constant_symbol("COEFFS")` produces `cuda_oxide_const_246e25db_COEFFS`.
/// `#[cuda_module]` passes a context-rich base name for `#[constant]` statics
/// (for example, including the module and source location), and the resulting
/// symbol is emitted into PTX address space 4 (`.const`). The host-side
/// `set_coeffs` methods generated by `#[cuda_module]` resolve that exact name.
pub const CONSTANT_PREFIX: &str = "cuda_oxide_const_246e25db_";

/// Local binding name injected by `#[kernel]` and `#[device]` for the
/// hidden thread-index scope token.
pub const KERNEL_SCOPE_LOCAL: &str = "cuda_oxide_kernel_scope_246e25db";

/// Prefix of the link-anchor symbol that pins a crate's embedded device
/// artifact (the `.oxart` section) into the final binary.
///
/// The codegen backend packages each crate's PTX/cubin/NVVM-IR/LTOIR into
/// a small host object file whose only content is the `.oxart` data
/// section. For binary crates that object is handed straight to the
/// linker, so the section always survives. For *library* crates the
/// object becomes one member of the crate's `.rlib` archive, and linkers
/// only extract archive members that define a symbol someone references.
/// A data-only object defines no symbols, so the member used to be
/// silently dropped and `load()` failed at runtime with `ModuleNotFound`.
///
/// To fix that, the backend defines one global symbol with this prefix at
/// the start of the `.oxart` data, and the `#[cuda_module]` macro makes
/// the generated `load_named()` read that symbol's address. Any caller of
/// `load()` therefore creates an undefined reference that forces the
/// linker to pull the artifact member out of the archive.
pub const ARTIFACT_ANCHOR_PREFIX: &str = "cuda_oxide_artifact_anchor_246e25db_";

/// Prefix of private statics emitted by `#[cuda_module]` for enabled generic
/// kernels.
///
/// Generic kernel specializations can be emitted into a downstream crate's
/// device artifact, so loading only the defining crate's artifact is not
/// sufficient. The host API therefore merges PTX bundles for such modules.
/// Ahead-of-time cubin materialization cannot preserve that behavior today;
/// the codegen collector uses this marker to reject the unsupported case
/// before loading libNVVM or nvJitLink.
pub const PTX_MERGE_REQUIRED_PREFIX: &str = "cuda_oxide_ptx_merge_required_246e25db_";

/// Indexed LLVM operation-attribute key carrying proven kernel reference
/// validity. This is compiler metadata, not a mangled link symbol.
pub const KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX: &str =
    "cuda_oxide_kernel_reference_param_validity_";

// ============================================================================
// Layer 2 — builders (macro side)
// ============================================================================

/// Build the mangled kernel symbol for a given base name.
///
/// ```
/// use reserved_oxide_symbols::kernel_symbol;
/// assert_eq!(
///     kernel_symbol("vecadd"),
///     "cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_vecadd"
/// );
/// ```
pub fn kernel_symbol(base: &str) -> String {
    format!("{KERNEL_PREFIX}{base}")
}

/// Build the mangled device-function symbol for a given base name.
///
/// ```
/// use reserved_oxide_symbols::device_symbol;
/// assert_eq!(
///     device_symbol("helper"),
///     "cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_helper"
/// );
/// ```
pub fn device_symbol(base: &str) -> String {
    format!("{DEVICE_PREFIX}{base}")
}

/// Build the mangled `#[device] extern` symbol for a given base name.
///
/// ```
/// use reserved_oxide_symbols::device_extern_symbol;
/// assert_eq!(
///     device_extern_symbol("cub_reduce"),
///     "cuda_oxide_device_extern_246e25db_cub_reduce",
/// );
/// ```
pub fn device_extern_symbol(base: &str) -> String {
    format!("{DEVICE_EXTERN_PREFIX}{base}")
}

/// Build the closure-monomorphization helper symbol for a given base name.
///
/// ```
/// use reserved_oxide_symbols::instantiate_symbol;
/// assert_eq!(
///     instantiate_symbol("map"),
///     "cuda_oxide_instantiate_246e25db_map",
/// );
/// ```
pub fn instantiate_symbol(base: &str) -> String {
    format!("{INSTANTIATE_PREFIX}{base}")
}

/// Build the mangled constant-static symbol for a given base name.
///
/// ```
/// use reserved_oxide_symbols::constant_symbol;
/// assert_eq!(constant_symbol("COEFFS"), "cuda_oxide_const_246e25db_COEFFS");
/// ```
pub fn constant_symbol(base: &str) -> String {
    format!("{CONSTANT_PREFIX}{base}")
}

/// Build the compiler marker associated with one generic kernel.
pub fn ptx_merge_required_marker(base: &str) -> String {
    let mut symbol = String::from(PTX_MERGE_REQUIRED_PREFIX);
    push_symbol_sanitized(&mut symbol, base);
    symbol
}

/// Build the operation-attribute key for one physical kernel parameter.
pub fn kernel_reference_param_validity_key(index: usize) -> String {
    format!("{KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX}{index}")
}

/// Build the legacy artifact link-anchor symbol for a package and version.
///
/// Both the codegen backend (which defines the symbol inside the embedded
/// artifact object) and the `#[cuda_module]` macro (which references it
/// from the generated `load_named()`) derive the name from `CARGO_PKG_NAME`
/// and `CARGO_PKG_VERSION` in the same rustc invocation.
///
/// The version is part of the name so that two different versions of one
/// package in the same dependency graph each keep their own bundle. Any
/// character that is not valid in a symbol name is mapped to `_`.
///
/// ```
/// use reserved_oxide_symbols::artifact_anchor_symbol;
/// assert_eq!(
///     artifact_anchor_symbol("julia-lib", "0.1.0"),
///     "cuda_oxide_artifact_anchor_246e25db_julia_lib_0_1_0",
/// );
/// ```
pub fn artifact_anchor_symbol(package_name: &str, package_version: &str) -> String {
    let mut symbol = String::from(ARTIFACT_ANCHOR_PREFIX);
    push_symbol_sanitized(&mut symbol, package_name);
    symbol.push('_');
    push_symbol_sanitized(&mut symbol, package_version);
    symbol
}

/// Build the target-specific v2 artifact anchor.
///
/// V2 extends the legacy identity with rustc's crate target and Cargo's
/// optional binary target name. This prevents a package's library, binaries,
/// examples, and tests from satisfying one another's anchor references. The
/// legacy builder remains available for mixed-version compatibility.
pub fn artifact_anchor_symbol_v2(
    package_name: &str,
    package_version: &str,
    crate_name: &str,
    binary_name: Option<&str>,
) -> String {
    let mut symbol = String::from(ARTIFACT_ANCHOR_PREFIX);
    symbol.push_str("v2_");
    push_symbol_sanitized(&mut symbol, package_name);
    symbol.push('_');
    push_symbol_sanitized(&mut symbol, package_version);
    symbol.push('_');
    push_symbol_sanitized(&mut symbol, crate_name);
    match binary_name {
        Some(binary_name) => {
            symbol.push_str("_bin_");
            push_symbol_sanitized(&mut symbol, binary_name);
        }
        None => symbol.push_str("_nonbin"),
    }
    symbol
}

/// Append `raw` to `symbol`, replacing every character that is not
/// `[A-Za-z0-9_]` with `_` so the result is a valid linker symbol.
fn push_symbol_sanitized(symbol: &mut String, raw: &str) {
    symbol.extend(
        raw.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }),
    );
}

// ============================================================================
// Layer 3 — predicates and base-name extractors (consumer side)
// ============================================================================

/// Returns `true` if `name` is a kernel symbol (or contains one as a
/// suffix of a longer FQDN like
/// `crate::module::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_foo`). Legacy roots
/// are recognized too so they can be rejected safely by a protocol-aware
/// backend rather than disappearing as ordinary host functions.
///
/// ```
/// use reserved_oxide_symbols::{is_kernel_symbol, kernel_symbol};
/// assert!(is_kernel_symbol(&kernel_symbol("vecadd")));
/// assert!(!is_kernel_symbol("vecadd"));
/// ```
pub fn is_kernel_symbol(name: &str) -> bool {
    name.contains(KERNEL_PREFIX) || name.contains(LEGACY_KERNEL_PREFIX)
}

/// Returns `true` only for a kernel emitted by the scoped-cache protocol.
pub fn is_current_kernel_symbol(name: &str) -> bool {
    name.rsplit("::")
        .next()
        .is_some_and(|item| item.starts_with(KERNEL_PREFIX))
}

/// Returns `true` only for a kernel emitted before the scoped-cache protocol.
pub fn is_legacy_kernel_symbol(name: &str) -> bool {
    name.rsplit("::")
        .next()
        .is_some_and(|item| item.starts_with(LEGACY_KERNEL_PREFIX))
        && !is_current_kernel_symbol(name)
}

/// Returns `true` if `name` is a device-function symbol (excluding extern).
///
/// Because the hash suffix makes [`DEVICE_PREFIX`] and [`DEVICE_EXTERN_PREFIX`]
/// mutually exclusive substrings, this check needs no special-casing for
/// extern symbols.
///
/// ```
/// use reserved_oxide_symbols::{is_device_symbol, device_symbol, device_extern_symbol};
/// assert!(is_device_symbol(&device_symbol("helper")));
/// assert!(!is_device_symbol(&device_extern_symbol("foo")));
/// ```
pub fn is_device_symbol(name: &str) -> bool {
    name.contains(DEVICE_PREFIX) || name.contains(LEGACY_DEVICE_PREFIX)
}

/// Returns `true` only for a device function emitted by the scoped-cache
/// protocol.
pub fn is_current_device_symbol(name: &str) -> bool {
    name.rsplit("::")
        .next()
        .is_some_and(|item| item.starts_with(DEVICE_PREFIX))
}

/// Returns `true` only for a device function emitted before the scoped-cache
/// protocol.
pub fn is_legacy_device_symbol(name: &str) -> bool {
    name.rsplit("::")
        .next()
        .is_some_and(|item| item.starts_with(LEGACY_DEVICE_PREFIX))
        && !is_current_device_symbol(name)
}

/// Returns `true` if `name` is a `#[device] extern` symbol.
///
/// ```
/// use reserved_oxide_symbols::{is_device_extern_symbol, device_extern_symbol};
/// assert!(is_device_extern_symbol(&device_extern_symbol("foo")));
/// ```
pub fn is_device_extern_symbol(name: &str) -> bool {
    name.contains(DEVICE_EXTERN_PREFIX)
}

/// Returns `true` for a generic-kernel PTX-merge marker path.
pub fn is_ptx_merge_required_marker(name: &str) -> bool {
    name.rsplit("::")
        .next()
        .and_then(|component| component.strip_prefix(PTX_MERGE_REQUIRED_PREFIX))
        .is_some_and(|base| !base.is_empty())
}

/// Extract the parameter-index text from a reference-validity attribute key.
///
/// Preserve malformed or empty suffixes so the compiler can diagnose invalid
/// metadata instead of treating it as an unrelated attribute.
pub fn kernel_reference_param_validity_index_text(key: &str) -> Option<&str> {
    key.strip_prefix(KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX)
}

/// Returns `true` if `name` is a closure-monomorphization helper symbol.
///
/// ```
/// use reserved_oxide_symbols::{is_instantiate_symbol, instantiate_symbol};
/// assert!(is_instantiate_symbol(&instantiate_symbol("map")));
/// ```
pub fn is_instantiate_symbol(name: &str) -> bool {
    name.contains(INSTANTIATE_PREFIX)
}

/// Returns `true` if `name` is a `#[constant]` static symbol.
///
/// Matches on substring so cross-crate / FQDN symbols are recognised, in
/// keeping with [`is_kernel_symbol`] and [`is_device_symbol`].
///
/// ```
/// use reserved_oxide_symbols::{is_constant_symbol, constant_symbol};
/// assert!(is_constant_symbol(&constant_symbol("COEFFS")));
/// assert!(is_constant_symbol("my_crate::kernels::cuda_oxide_const_246e25db_COEFFS"));
/// assert!(!is_constant_symbol("COEFFS"));
/// ```
pub fn is_constant_symbol(name: &str) -> bool {
    name.contains(CONSTANT_PREFIX)
}

/// Strip the kernel prefix from a possibly-FQDN symbol name.
///
/// Returns the part of `name` after [`KERNEL_PREFIX`], or `None` if `name`
/// is not a kernel symbol. Works for both plain forms and crate-qualified
/// FQDN forms.
///
/// ```
/// use reserved_oxide_symbols::kernel_base_name;
/// assert_eq!(
///     kernel_base_name("cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_vecadd"),
///     Some("vecadd"),
/// );
/// assert_eq!(
///     kernel_base_name("kernel_lib::cuda_oxide_kernel_246e25db_scale"),
///     Some("scale"),
/// );
/// assert_eq!(kernel_base_name("vecadd"), None);
/// ```
pub fn kernel_base_name(name: &str) -> Option<&str> {
    name.find(KERNEL_PREFIX)
        .map(|pos| &name[pos + KERNEL_PREFIX.len()..])
        .or_else(|| {
            name.find(LEGACY_KERNEL_PREFIX)
                .map(|pos| &name[pos + LEGACY_KERNEL_PREFIX.len()..])
        })
}

/// Strip the device prefix from a possibly-FQDN symbol name.
///
/// Returns the part after [`DEVICE_PREFIX`], or `None` if `name` is not a
/// device symbol. Returns `None` for `#[device] extern` symbols thanks to
/// the mutual-exclusion guarantee documented at the crate root.
///
/// ```
/// use reserved_oxide_symbols::device_base_name;
/// assert_eq!(
///     device_base_name("cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_helper"),
///     Some("helper"),
/// );
/// assert_eq!(device_base_name("cuda_oxide_device_extern_246e25db_foo"), None);
/// ```
pub fn device_base_name(name: &str) -> Option<&str> {
    name.find(DEVICE_PREFIX)
        .map(|pos| &name[pos + DEVICE_PREFIX.len()..])
        .or_else(|| {
            name.find(LEGACY_DEVICE_PREFIX)
                .map(|pos| &name[pos + LEGACY_DEVICE_PREFIX.len()..])
        })
}

/// Strip the device-extern prefix from a possibly-FQDN symbol name.
///
/// ```
/// use reserved_oxide_symbols::device_extern_base_name;
/// assert_eq!(
///     device_extern_base_name("cuda_oxide_device_extern_246e25db_cub_reduce"),
///     Some("cub_reduce"),
/// );
/// assert_eq!(device_extern_base_name("vecadd"), None);
/// ```
pub fn device_extern_base_name(name: &str) -> Option<&str> {
    name.find(DEVICE_EXTERN_PREFIX)
        .map(|pos| &name[pos + DEVICE_EXTERN_PREFIX.len()..])
}

/// Format a mangled symbol as a human-readable diagnostic label.
///
/// `cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_vecadd` becomes `vecadd (kernel)`,
/// `cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_helper` becomes `helper (device)`, and
/// device-extern symbols become `<base> (device extern)`. Returns `None`
/// for anything outside the reserved namespace.
///
/// Useful in error messages where users see the symbol name and need to
/// understand which kind of cuda-oxide construct it came from.
///
/// ```
/// use reserved_oxide_symbols::display_name;
/// assert_eq!(
///     display_name("cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_vecadd").as_deref(),
///     Some("vecadd (kernel)"),
/// );
/// assert_eq!(display_name("std::vec::Vec::new"), None);
/// ```
pub fn display_name(name: &str) -> Option<String> {
    if let Some(base) = device_extern_base_name(name) {
        Some(format!("{base} (device extern)"))
    } else if let Some(base) = kernel_base_name(name) {
        Some(format!("{base} (kernel)"))
    } else if let Some(base) = device_base_name(name) {
        Some(format!("{base} (device)"))
    } else if let Some(base) = constant_base_name(name) {
        Some(format!("{base} (constant)"))
    } else {
        instantiate_base_name(name).map(|base| format!("{base} (instantiate helper)"))
    }
}

/// Strip the instantiate prefix from a possibly-FQDN symbol name.
///
/// ```
/// use reserved_oxide_symbols::instantiate_base_name;
/// assert_eq!(
///     instantiate_base_name("cuda_oxide_instantiate_246e25db_map"),
///     Some("map"),
/// );
/// ```
pub fn instantiate_base_name(name: &str) -> Option<&str> {
    name.find(INSTANTIATE_PREFIX)
        .map(|pos| &name[pos + INSTANTIATE_PREFIX.len()..])
}

/// Strip the constant-static prefix from a possibly-FQDN symbol name.
///
/// ```
/// use reserved_oxide_symbols::constant_base_name;
/// assert_eq!(
///     constant_base_name("cuda_oxide_const_246e25db_COEFFS"),
///     Some("COEFFS"),
/// );
/// assert_eq!(
///     constant_base_name("my_crate::kernels::cuda_oxide_const_246e25db_COEFFS"),
///     Some("COEFFS"),
/// );
/// assert_eq!(constant_base_name("COEFFS"), None);
/// ```
pub fn constant_base_name(name: &str) -> Option<&str> {
    name.find(CONSTANT_PREFIX)
        .map(|pos| &name[pos + CONSTANT_PREFIX.len()..])
}

// ============================================================================
// Tests — pin the constant values, verify mutual-exclusion, round-trip
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// The hash value is locked. Changing this constant is a
    /// breaking-change to every cuda-oxide built artifact and must be
    /// done deliberately.
    #[test]
    fn hash_value_is_pinned() {
        assert_eq!(HASH_SUFFIX, "246e25db");
        assert_eq!(
            KERNEL_PREFIX,
            "cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_"
        );
        assert_eq!(LEGACY_KERNEL_PREFIX, "cuda_oxide_kernel_246e25db_");
        assert_eq!(
            DEVICE_PREFIX,
            "cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_"
        );
        assert_eq!(LEGACY_DEVICE_PREFIX, "cuda_oxide_device_246e25db_");
        assert_eq!(DEVICE_EXTERN_PREFIX, "cuda_oxide_device_extern_246e25db_");
        assert_eq!(INSTANTIATE_PREFIX, "cuda_oxide_instantiate_246e25db_");
        assert_eq!(CONSTANT_PREFIX, "cuda_oxide_const_246e25db_");
        assert_eq!(
            PTX_MERGE_REQUIRED_PREFIX,
            "cuda_oxide_ptx_merge_required_246e25db_"
        );
        assert_eq!(
            ARTIFACT_ANCHOR_PREFIX,
            "cuda_oxide_artifact_anchor_246e25db_"
        );
        assert_eq!(
            KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX,
            "cuda_oxide_kernel_reference_param_validity_"
        );
        // `KERNEL_SCOPE_LOCAL` is not a link symbol, but renaming it silently
        // disarms two hygiene regressions that spell it as an *identifier*,
        // where no compiler check ties them back to this constant:
        //
        //   * crates/cuda-macros/tests/pass/kernel_launch_context_api.rs
        //     declares a binding named `{KERNEL_SCOPE_LOCAL}_storage` so the
        //     mixed-site storage binding must survive the collision.
        //   * crates/rustc-codegen-cuda/examples/const_generic/src/main.rs
        //     declares a const generic named `KERNEL_SCOPE_LOCAL` so
        //     `call_site_ident_avoiding_item` must take its rename path.
        //
        // Both keep passing once renamed, because there is simply nothing left
        // to collide with. Update them together with this constant.
        assert_eq!(KERNEL_SCOPE_LOCAL, "cuda_oxide_kernel_scope_246e25db");
    }

    /// Every reserved name shares the reserved root. The macro guard checks
    /// for `RESERVED_ROOT` and rejects user-defined names that start
    /// with it; this test ensures the reserved root remains a true
    /// prefix of every mangled category this crate hands out, including the
    /// artifact anchor and the injected scope local.
    #[test]
    fn all_prefixes_share_reserved_root() {
        for p in [
            KERNEL_PREFIX,
            LEGACY_KERNEL_PREFIX,
            DEVICE_PREFIX,
            LEGACY_DEVICE_PREFIX,
            DEVICE_EXTERN_PREFIX,
            INSTANTIATE_PREFIX,
            CONSTANT_PREFIX,
            PTX_MERGE_REQUIRED_PREFIX,
            ARTIFACT_ANCHOR_PREFIX,
            KERNEL_SCOPE_LOCAL,
            KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX,
        ] {
            assert!(
                p.starts_with(RESERVED_ROOT),
                "prefix {p:?} must start with {RESERVED_ROOT:?}"
            );
        }
    }

    /// The hash suffix makes `DEVICE_PREFIX` and `DEVICE_EXTERN_PREFIX`
    /// mutually exclusive substrings. This is the property that lets
    /// `is_device_symbol` skip the historical extern-exclusion dance.
    #[test]
    fn device_and_device_extern_are_mutually_exclusive_substrings() {
        // DEVICE_EXTERN_PREFIX never contains DEVICE_PREFIX
        assert!(!DEVICE_EXTERN_PREFIX.contains(DEVICE_PREFIX));
        // ...and vice versa
        assert!(!DEVICE_PREFIX.contains(DEVICE_EXTERN_PREFIX));
    }

    #[test]
    fn ptx_merge_markers_are_per_kernel_and_sanitized() {
        let marker = ptx_merge_required_marker("nested::map-value");
        assert_eq!(
            marker,
            "cuda_oxide_ptx_merge_required_246e25db_nested__map_value"
        );
        assert!(is_ptx_merge_required_marker(&marker));
        assert!(is_ptx_merge_required_marker(&format!(
            "crate::module::{marker}"
        )));
        assert!(!is_ptx_merge_required_marker(PTX_MERGE_REQUIRED_PREFIX));
        assert!(!is_ptx_merge_required_marker(&format!("prefix_{marker}")));
        assert!(!is_ptx_merge_required_marker(&format!("{marker}::child")));
    }

    #[test]
    fn reference_validity_keys_preserve_parameter_indices_and_malformed_suffixes() {
        for index in [0, 1, usize::MAX] {
            let key = kernel_reference_param_validity_key(index);
            assert_eq!(
                kernel_reference_param_validity_index_text(&key),
                Some(index.to_string().as_str())
            );
            assert_eq!(
                kernel_reference_param_validity_index_text(&format!("prefix_{key}")),
                None
            );
        }
        for suffix in ["", "invalid", "0_extra"] {
            let key = format!("{KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX}{suffix}");
            assert_eq!(
                kernel_reference_param_validity_index_text(&key),
                Some(suffix)
            );
        }
        assert_eq!(
            kernel_reference_param_validity_index_text("gpu_kernel"),
            None
        );
    }

    /// `kernel_base_name(kernel_symbol(x)) == Some(x)` for any reasonable
    /// base name. Same property for device, device_extern, instantiate.
    #[test]
    fn build_then_extract_round_trips() {
        for base in ["vecadd", "scale", "fast_sqrt", "cub_reduce"] {
            assert_eq!(kernel_base_name(&kernel_symbol(base)), Some(base));
            assert_eq!(device_base_name(&device_symbol(base)), Some(base));
            assert_eq!(
                device_extern_base_name(&device_extern_symbol(base)),
                Some(base),
            );
            assert_eq!(instantiate_base_name(&instantiate_symbol(base)), Some(base),);
            assert_eq!(constant_base_name(&constant_symbol(base)), Some(base));
        }
    }

    #[test]
    fn current_and_legacy_codegen_roots_are_distinguished_and_strip_identically() {
        let current_kernel = kernel_symbol("map");
        let legacy_kernel = format!("{LEGACY_KERNEL_PREFIX}map");
        let current_device = device_symbol("helper");
        let legacy_device = format!("{LEGACY_DEVICE_PREFIX}helper");

        assert!(is_kernel_symbol(&current_kernel));
        assert!(is_current_kernel_symbol(&current_kernel));
        assert!(!is_legacy_kernel_symbol(&current_kernel));
        assert!(is_kernel_symbol(&legacy_kernel));
        assert!(!is_current_kernel_symbol(&legacy_kernel));
        assert!(is_legacy_kernel_symbol(&legacy_kernel));
        assert_eq!(kernel_base_name(&current_kernel), Some("map"));
        assert_eq!(kernel_base_name(&legacy_kernel), Some("map"));
        assert!(current_kernel.contains(LEGACY_KERNEL_PREFIX));
        assert_eq!(
            current_kernel
                .find(LEGACY_KERNEL_PREFIX)
                .map(|index| &current_kernel[index + LEGACY_KERNEL_PREFIX.len()..]),
            Some("map"),
            "an older backend must detect and strip the embedded legacy prefix"
        );

        assert!(is_device_symbol(&current_device));
        assert!(is_current_device_symbol(&current_device));
        assert!(!is_legacy_device_symbol(&current_device));
        assert!(is_device_symbol(&legacy_device));
        assert!(!is_current_device_symbol(&legacy_device));
        assert!(is_legacy_device_symbol(&legacy_device));
        assert_eq!(device_base_name(&current_device), Some("helper"));
        assert_eq!(device_base_name(&legacy_device), Some("helper"));
        assert!(current_device.contains(LEGACY_DEVICE_PREFIX));
        assert_eq!(
            current_device
                .find(LEGACY_DEVICE_PREFIX)
                .map(|index| &current_device[index + LEGACY_DEVICE_PREFIX.len()..]),
            Some("helper"),
            "an older backend must detect and strip the embedded legacy prefix"
        );

        // An old macro prefixes the user-written base verbatim. Its output
        // must remain legacy even when that base begins with the protocol's
        // spelling.
        let old_collision = format!("{LEGACY_KERNEL_PREFIX}codegen_v1_map");
        assert!(is_legacy_kernel_symbol(&old_collision));
        assert!(!is_current_kernel_symbol(&old_collision));

        // Protocol classification is about the root item, never an enclosing
        // path component with a reserved-looking name.
        let misleading_path = format!("crate::{KERNEL_PREFIX}module::{LEGACY_KERNEL_PREFIX}map");
        assert!(is_legacy_kernel_symbol(&misleading_path));
        assert!(!is_current_kernel_symbol(&misleading_path));
    }

    /// Cross-crate kernels carry path qualifiers like
    /// `kernel_lib::cuda_oxide_kernel_246e25db_scale`. The base-name
    /// extractor must skip past the qualifier.
    #[test]
    fn extracts_base_from_fqdn() {
        assert_eq!(
            kernel_base_name("kernel_lib::cuda_oxide_kernel_246e25db_scale"),
            Some("scale"),
        );
        assert_eq!(
            device_base_name("helper_fn::cuda_oxide_device_246e25db_clamp"),
            Some("clamp"),
        );
        assert_eq!(
            device_extern_base_name("ffi::cuda_oxide_device_extern_246e25db_foo"),
            Some("foo"),
        );
    }

    /// `is_device_symbol` must NOT match device-extern symbols, even
    /// without an explicit exclusion check on the caller's part.
    #[test]
    fn device_predicate_excludes_extern() {
        assert!(is_device_symbol(&device_symbol("foo")));
        assert!(!is_device_symbol(&device_extern_symbol("foo")));
        assert!(is_device_extern_symbol(&device_extern_symbol("foo")));
        assert!(!is_device_extern_symbol(&device_symbol("foo")));
    }

    /// `display_name` produces the right kind label and chooses
    /// device-extern over device when both prefixes match a substring,
    /// covering an edge case if the two ever stop being mutually
    /// exclusive in a future refactor.
    #[test]
    fn display_name_categorizes_correctly() {
        assert_eq!(
            display_name(&kernel_symbol("vecadd")).as_deref(),
            Some("vecadd (kernel)"),
        );
        assert_eq!(
            display_name(&device_symbol("helper")).as_deref(),
            Some("helper (device)"),
        );
        assert_eq!(
            display_name(&device_extern_symbol("foo")).as_deref(),
            Some("foo (device extern)"),
        );
        assert_eq!(
            display_name(&instantiate_symbol("map")).as_deref(),
            Some("map (instantiate helper)"),
        );
        assert_eq!(
            display_name(&constant_symbol("COEFFS")).as_deref(),
            Some("COEFFS (constant)"),
        );
        assert_eq!(display_name("std::vec::Vec::new"), None);
    }

    /// User-style names that don't include the hash suffix must not be
    /// matched by any predicate. This is the core safety property.
    #[test]
    fn user_names_with_old_prefix_are_not_matched() {
        // These are the legacy / accidental forms we're defending against.
        for evil in [
            "cuda_oxide_kernel_evil",
            "cuda_oxide_device_evil",
            "cuda_oxide_device_extern_evil",
            "cuda_oxide_instantiate_evil",
            "cuda_oxide_const_evil",
        ] {
            assert!(!is_kernel_symbol(evil), "unexpected match: {evil}");
            assert!(!is_device_symbol(evil), "unexpected match: {evil}");
            assert!(!is_device_extern_symbol(evil), "unexpected match: {evil}");
            assert!(!is_instantiate_symbol(evil), "unexpected match: {evil}");
            assert!(!is_constant_symbol(evil), "unexpected match: {evil}");
            assert_eq!(display_name(evil), None);
        }
    }

    /// The anchor symbol must be a valid linker symbol for any package
    /// name/version cargo can produce, and must live in the reserved root.
    #[test]
    fn artifact_anchor_symbol_is_sanitized_and_reserved() {
        let anchor = artifact_anchor_symbol_v2("my-kernels", "1.2.0-rc.3", "kernel_lib", None);
        assert_eq!(
            anchor,
            "cuda_oxide_artifact_anchor_246e25db_v2_my_kernels_1_2_0_rc_3_kernel_lib_nonbin",
        );
        assert!(anchor.starts_with(RESERVED_ROOT));
        assert!(
            anchor
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        );
        // Distinct versions of one package must get distinct anchors, so
        // both archive members can be extracted into the same binary.
        assert_ne!(
            artifact_anchor_symbol_v2("my-kernels", "0.1.0", "kernel_lib", None),
            artifact_anchor_symbol_v2("my-kernels", "0.2.0", "kernel_lib", None),
        );
        // Distinct rustc targets from one package must not satisfy each
        // other's anchor references.
        assert_ne!(
            artifact_anchor_symbol_v2("my-kernels", "0.1.0", "kernel_lib", None),
            artifact_anchor_symbol_v2("my-kernels", "0.1.0", "host_bin", Some("host-bin")),
        );
        // Cargo permits a package's library and default binary to share the
        // same crate name; CARGO_BIN_NAME still keeps their anchors distinct.
        assert_ne!(
            artifact_anchor_symbol_v2("my-kernels", "0.1.0", "same_name", None),
            artifact_anchor_symbol_v2("my-kernels", "0.1.0", "same_name", Some("same_name")),
        );
        assert_eq!(
            artifact_anchor_symbol("my-kernels", "1.2.0-rc.3"),
            "cuda_oxide_artifact_anchor_246e25db_my_kernels_1_2_0_rc_3"
        );
        assert_ne!(
            artifact_anchor_symbol("my-kernels", "1.2.0+kernel-lib.nonbin"),
            artifact_anchor_symbol_v2("my-kernels", "1.2.0", "kernel-lib", None),
        );
    }

    /// Sanity-check the reserved-root membership predicate consumers
    /// will use to flag user code that's reaching into the reserved
    /// namespace by accident.
    #[test]
    fn reserved_root_smoke() {
        assert!(kernel_symbol("foo").starts_with(RESERVED_ROOT));
        assert!("cuda_oxide_kernel_evil".starts_with(RESERVED_ROOT));
        assert!(!"vecadd".starts_with(RESERVED_ROOT));
        assert!(!"my_helper".starts_with(RESERVED_ROOT));
    }

    /// Unused-import guard for `to_string`: pull it in so the symbol
    /// is exercised even if the rest of the suite drops it later.
    #[test]
    fn to_string_works_in_no_std() {
        let s: String = "x".to_string();
        assert_eq!(s.len(), 1);
    }
}
