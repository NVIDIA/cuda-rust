/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retained PTX entry names for merged modules, and the host/device
//! type-identity divergence diagnosis built on them.
//!
//! # Why this exists
//!
//! Generic kernels are looked up by `<base>_TID_<hex32>`, where both sides
//! independently compute the same 128-bit type hash: the backend calls
//! `tcx.type_id_hash(instance_ty)` (`compute_kernel_export_name` in
//! `rustc-codegen-cuda`) and the host extracts the hash bytes from
//! [`core::any::TypeId`] (see [`crate::type_id`]). The values cannot drift
//! while `TypeId`'s runtime layout is the raw hash, but that layout is an
//! internal detail of the standard library. If a future nightly changes it,
//! the transmute in `cuda_host::type_id` still compiles, both sides keep
//! producing 32-hex-character names, and every generic launch fails with an
//! opaque `DriverError(500, "named symbol not found")`.
//!
//! To make that failure self-explaining, [`crate::load_all_ptx_bundles_merged`]
//! retains the `.entry` names it already parses out of the merged PTX. When a
//! generic-kernel lookup later misses, the macro-generated launch paths call
//! [`diagnose_generic_kernel_load_error`] /
//! [`panic_generic_kernel_load_failed`]: if the module holds the same base
//! kernel under a *different* `_TID_` hash, the failure is reported as a
//! host/device type-identity divergence with both names and the remedy,
//! instead of a bare driver error. An ordinary miss (no same-base entry) keeps
//! today's error exactly.
//!
//! # Considered and deliberately not implemented: load-time canary
//!
//! A second layer (launching a known dummy generic kernel at module load to
//! force this check before any real launch) was considered and rejected.
//! Every real generic launch already funnels through this diagnosis, so a
//! canary would only cover programs that load a module and then never launch
//! a generic kernel, a window the toolchain-bump runbook gates already cover.
//! The canary would also cost a PTX entry and a launch on every load.
//!
//! # Memory notes
//!
//! The registry retains one `String` per `.entry` in each merged module:
//! typically a few dozen bytes per kernel, i.e. well under a few KB per
//! module. Entries are held via [`Weak`], so retention never extends a
//! module's lifetime, and records of dropped modules are pruned on every new
//! registration.

use cuda_core::{CudaModule, DriverError};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Infix that separates a generic kernel's base name from its 32-hex-character
/// type-identity hash. Must match the backend's `compute_kernel_export_name`
/// (`format!("{base}_TID_{hash:032x}")`) and the host-side
/// [`crate::type_id::__intern_generic_kernel_name`].
const TYPE_ID_INFIX: &str = "_TID_";

/// Number of hex characters in the `u128` type-identity hash.
const TYPE_ID_HASH_LEN: usize = 32;

/// One retained record: the (weakly held) owning module and its entry names.
type RetainedEntries<T> = (Weak<T>, Arc<Vec<String>>);

/// Process-wide map from a loaded module to the PTX `.entry` names it was
/// built from.
///
/// Keyed by allocation identity of the owning [`Arc`]: a [`Weak`] pins the
/// allocation (not the value), so pointer equality against a live `Arc` is
/// unambiguous. A dead record's address cannot be reused while its `Weak`
/// exists, and dead records are pruned on registration.
struct EntryRegistry<T> {
    retained: Mutex<Vec<RetainedEntries<T>>>,
}

impl<T> EntryRegistry<T> {
    const fn new() -> Self {
        Self {
            retained: Mutex::new(Vec::new()),
        }
    }

    fn register(&self, owner: &Arc<T>, entries: Vec<String>) {
        let mut retained = self
            .retained
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Prune records whose module is gone so the registry stays bounded by
        // the number of *live* merged modules.
        retained.retain(|(weak, _)| weak.strong_count() > 0);
        retained.push((Arc::downgrade(owner), Arc::new(entries)));
    }

    fn entries_for(&self, owner: &Arc<T>) -> Option<Arc<Vec<String>>> {
        let retained = self
            .retained
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        retained
            .iter()
            .find(|(weak, _)| {
                weak.upgrade()
                    .is_some_and(|alive| Arc::ptr_eq(&alive, owner))
            })
            .map(|(_, entries)| entries.clone())
    }
}

static MERGED_MODULE_ENTRIES: OnceLock<EntryRegistry<CudaModule>> = OnceLock::new();

fn merged_module_entries() -> &'static EntryRegistry<CudaModule> {
    MERGED_MODULE_ENTRIES.get_or_init(EntryRegistry::new)
}

/// Retains the `.entry` names of a freshly merged-and-loaded PTX module.
///
/// Best-effort by design: if `ptx-parse` cannot parse the merged text that the
/// driver accepted, nothing is retained and a later lookup miss simply keeps
/// the ordinary driver error. Registration must never fail a load that the
/// driver was happy with.
pub(crate) fn register_merged_module_entries(module: &Arc<CudaModule>, merged_ptx: &str) {
    if let Some(entries) = entry_names_from_ptx(merged_ptx) {
        merged_module_entries().register(module, entries);
    }
}

/// Parses PTX text and returns the names of its `.entry` kernels.
///
/// `.func` device helpers are excluded: only `.entry` symbols are launchable,
/// and only they participate in `_TID_` lookup.
fn entry_names_from_ptx(ptx: &str) -> Option<Vec<String>> {
    let document = ptx_parse::Document::parse(ptx).ok()?;
    Some(
        document
            .callables()
            .iter()
            .filter(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
            .map(|callable| callable.name().to_string())
            .collect(),
    )
}

/// Splits `<base>_TID_<hex32>` into `(base, hex32)`.
///
/// Strict on purpose: the hash suffix must be exactly 32 lowercase hex
/// characters (the `{:032x}` both naming sides emit), so a kernel whose name
/// merely contains `_TID_` never triggers a false diagnosis.
fn split_type_id_name(name: &str) -> Option<(&str, &str)> {
    let index = name.rfind(TYPE_ID_INFIX)?;
    let base = &name[..index];
    let hash = &name[index + TYPE_ID_INFIX.len()..];
    let hash_is_exact = hash.len() == TYPE_ID_HASH_LEN
        && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    (!base.is_empty() && hash_is_exact).then_some((base, hash))
}

/// Returns the entries in `entries` that share `requested`'s base kernel name
/// but carry a different type-identity hash.
fn divergent_entries_in(entries: &[String], requested: &str) -> Vec<String> {
    let Some((base, hash)) = split_type_id_name(requested) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter(|entry| {
            split_type_id_name(entry)
                .is_some_and(|(entry_base, entry_hash)| entry_base == base && entry_hash != hash)
        })
        .cloned()
        .collect()
}

/// Returns the retained `.entry` names that share `requested`'s base kernel
/// name but carry a different `_TID_` type-identity hash.
///
/// Empty when `requested` is not a `_TID_` name, when `module` was not loaded
/// through [`crate::load_all_ptx_bundles_merged`], or when no same-base entry
/// exists; i.e. whenever the miss is an ordinary miss.
pub fn divergent_type_id_entries(module: &Arc<CudaModule>, requested: &str) -> Vec<String> {
    match merged_module_entries().entries_for(module) {
        Some(entries) => divergent_entries_in(&entries, requested),
        None => Vec::new(),
    }
}

/// Builds the self-explaining panic message for a host/device type-identity
/// naming divergence. Both names, the cause, and the remedy are spelled out
/// so the failure is actionable without reading compiler internals.
fn type_id_divergence_message(
    requested: &str,
    divergent: &[String],
    error: &DriverError,
) -> String {
    let (base, _) = split_type_id_name(requested)
        .expect("divergence is only diagnosed for well-formed _TID_ names");
    format!(
        "generic kernel PTX entry `{requested}` was not found, but the loaded module DOES \
         contain `{base}` under a different type-identity hash: {found}. The host \
         (core::intrinsics::type_id) and the device backend (tcx.type_id_hash) disagreed on \
         the kernel's `_TID_` name, which means the TypeId-to-u128 extraction in \
         cuda_host::type_id is no longer value-correct on this toolchain. Host and device \
         code must be built by the same pinned nightly; re-verify the TypeId contract in \
         cuda-host/src/type_id.rs and the backend's compute_kernel_export_name together \
         before trusting any generic kernel launch. Original driver error: {error:?}",
        found = divergent
            .iter()
            .map(|entry| format!("`{entry}`"))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// Routes a failed generic-kernel `load_function` through the type-identity
/// divergence diagnosis.
///
/// Called by `#[cuda_module]`-generated launchers when loading a `_TID_`
/// entry fails. If the module retains the same base kernel under a different
/// hash, this panics with the self-explaining divergence message: the
/// condition is a broken build-toolchain contract, not a runtime state a
/// caller could handle. Otherwise the original [`DriverError`] is returned
/// unchanged, preserving today's error path for ordinary misses.
#[track_caller]
pub fn diagnose_generic_kernel_load_error(
    module: &Arc<CudaModule>,
    ptx_name: &str,
    error: DriverError,
) -> DriverError {
    let divergent = divergent_type_id_entries(module, ptx_name);
    if divergent.is_empty() {
        return error;
    }
    panic!(
        "{}",
        type_id_divergence_message(ptx_name, &divergent, &error)
    );
}

/// Panics for a failed generic-kernel `load_function` on the `cuda_launch!` /
/// `cuda_launch_async!` paths, upgrading the message to the type-identity
/// divergence diagnosis when it applies.
///
/// An ordinary miss keeps the macros' long-standing message shape
/// (``Failed to load kernel `k` (expected PTX entry `e`): ...``).
#[track_caller]
pub fn panic_generic_kernel_load_failed(
    module: &Arc<CudaModule>,
    kernel: &str,
    ptx_name: &str,
    error: DriverError,
) -> ! {
    let divergent = divergent_type_id_entries(module, ptx_name);
    if divergent.is_empty() {
        panic!("Failed to load kernel `{kernel}` (expected PTX entry `{ptx_name}`): {error:?}");
    }
    panic!(
        "{}",
        type_id_divergence_message(ptx_name, &divergent, &error)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "00112233445566778899aabbccddeeff";
    const HASH_B: &str = "ffeeddccbbaa99887766554433221100";

    /// The driver code a real divergence produces: 500, "named symbol not
    /// found".
    fn probe_error() -> DriverError {
        DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_FOUND)
    }

    /// PTX in the same form the merged-bundle loader retains: `.entry`
    /// kernels plus a `.func` helper that must never be diagnosed.
    fn merged_ptx() -> String {
        format!(
            "\
.version 8.3
.target sm_80
.address_size 64
.visible .entry scale_TID_{HASH_A}()
{{
    ret;
}}
.visible .entry vecadd()
{{
    ret;
}}
.func helper_TID_{HASH_A}()
{{
    ret;
}}
"
        )
    }

    #[test]
    fn entry_names_come_from_real_ptx_parse() {
        let entries = entry_names_from_ptx(&merged_ptx()).expect("valid PTX must parse");
        assert_eq!(
            entries,
            vec![format!("scale_TID_{HASH_A}"), "vecadd".to_string()]
        );
    }

    #[test]
    fn unparseable_ptx_retains_nothing() {
        // `ptx-parse` only hard-fails on lexically ambiguous sources, such as
        // an unterminated block comment; structural noise is tolerated.
        assert_eq!(entry_names_from_ptx("/* unterminated"), None);
    }

    #[test]
    fn divergence_fires_with_both_names_for_same_base_different_hash() {
        let entries = entry_names_from_ptx(&merged_ptx()).expect("valid PTX must parse");
        let requested = format!("scale_TID_{HASH_B}");
        let divergent = divergent_entries_in(&entries, &requested);
        assert_eq!(divergent, vec![format!("scale_TID_{HASH_A}")]);

        let message = type_id_divergence_message(&requested, &divergent, &probe_error());
        // Both names must be present so the failure is diagnosable from the
        // message alone.
        assert!(
            message.contains(&format!("scale_TID_{HASH_B}")),
            "{message}"
        );
        assert!(
            message.contains(&format!("scale_TID_{HASH_A}")),
            "{message}"
        );
        // The message must identify the two hash producers and the remedy.
        assert!(message.contains("core::intrinsics::type_id"), "{message}");
        assert!(message.contains("tcx.type_id_hash"), "{message}");
        assert!(message.contains("same pinned nightly"), "{message}");
        assert!(message.contains("compute_kernel_export_name"), "{message}");
    }

    #[test]
    fn unrelated_miss_keeps_the_ordinary_error() {
        let entries = entry_names_from_ptx(&merged_ptx()).expect("valid PTX must parse");
        // Different base kernel: no same-base entry, so no diagnosis.
        let requested = format!("reduce_TID_{HASH_B}");
        assert!(divergent_entries_in(&entries, &requested).is_empty());
        // Non-generic name: `_TID_` never parsed, so no diagnosis.
        assert!(divergent_entries_in(&entries, "vecadd_missing").is_empty());
    }

    #[test]
    fn func_helpers_never_participate_in_the_diagnosis() {
        let entries = entry_names_from_ptx(&merged_ptx()).expect("valid PTX must parse");
        // `helper_TID_<hashA>` exists only as a `.func`; requesting the same
        // base with a different hash must not claim a divergence.
        let requested = format!("helper_TID_{HASH_B}");
        assert!(divergent_entries_in(&entries, &requested).is_empty());
    }

    #[test]
    fn hash_suffix_must_be_exactly_32_lowercase_hex() {
        assert!(split_type_id_name(&format!("scale_TID_{HASH_A}")).is_some());
        // Too short, uppercase, non-hex, and empty base are all rejected.
        assert!(split_type_id_name("scale_TID_0011").is_none());
        assert!(split_type_id_name("scale_TID_00112233445566778899AABBCCDDEEFF").is_none());
        assert!(split_type_id_name("scale_TID_zz112233445566778899aabbccddeeff").is_none());
        assert!(split_type_id_name(&format!("_TID_{HASH_A}")).is_none());
        assert!(split_type_id_name("no_infix_here").is_none());
    }

    #[test]
    fn registry_resolves_by_allocation_identity_and_prunes_dropped_owners() {
        let registry: EntryRegistry<u8> = EntryRegistry::new();
        let first = Arc::new(1u8);
        let second = Arc::new(2u8);
        registry.register(&first, vec!["first_entry".to_string()]);
        registry.register(&second, vec!["second_entry".to_string()]);

        assert_eq!(
            registry.entries_for(&first).unwrap().as_slice(),
            ["first_entry".to_string()]
        );
        assert_eq!(
            registry.entries_for(&second).unwrap().as_slice(),
            ["second_entry".to_string()]
        );
        // An owner the registry never saw resolves to nothing.
        assert!(registry.entries_for(&Arc::new(3u8)).is_none());

        drop(first);
        // Registration prunes records of dropped owners.
        let third = Arc::new(4u8);
        registry.register(&third, vec!["third_entry".to_string()]);
        let retained = registry.retained.lock().unwrap();
        assert_eq!(retained.len(), 2);
    }

    #[test]
    fn diagnosis_panics_with_both_names_through_the_message_builder() {
        let entries = entry_names_from_ptx(&merged_ptx()).expect("valid PTX must parse");
        let requested = format!("scale_TID_{HASH_B}");
        let divergent = divergent_entries_in(&entries, &requested);
        let outcome = std::panic::catch_unwind(|| {
            panic!(
                "{}",
                type_id_divergence_message(&requested, &divergent, &probe_error())
            );
        });
        let payload = outcome.expect_err("divergence must panic");
        let message = payload
            .downcast_ref::<String>()
            .expect("structured panic message");
        assert!(message.contains(&format!("scale_TID_{HASH_A}")));
        assert!(message.contains(&format!("scale_TID_{HASH_B}")));
    }
}
