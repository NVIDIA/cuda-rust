/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded callback admission from original identities and retained MIR.

use super::admission;
use dialect_mir::{
    library::{
        LibraryCallbackCall, LibraryCallbackDefinition as Definition,
        LibraryCallbackDispatch as Dispatch, LibraryCallbackEnvironment, LibraryCallbackSources,
        LibraryNativeMetadata, LibraryRoleAttr,
    },
    ops::MirFuncOp,
};
use pliron::{
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    printable::Printable,
};
use std::collections::{BTreeMap, BTreeSet};

/// Admit original callback bodies before native provider execution.
///
/// Source records include all original local-type classifications and only
/// monomorphically reachable call classifications, including erased expansions
/// and callees without imported symbols. They are identities, not safe verdicts.
/// Preparation cannot remove these obligations. This does not prove general
/// purity, termination, capture uniformity or the unsafe provider contract.
///
/// Return selected thunk identity to physical MIR symbol. The importer applies
/// its existing exporter normalization and retains the same actual definition.
pub fn admit_library_callbacks(
    module: Ptr<Operation>,
    ctx: &Context,
    sources: &LibraryCallbackSources,
    symbols: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, String> {
    let roles = admission::capture_library_roles(module, ctx, symbols)
        .map_err(|error| error.disp(ctx).to_string())?;
    for (site, role) in &roles {
        let callback = matches!(
            role,
            LibraryRoleAttr::Native(
                LibraryNativeMetadata::Callback { .. }
                    | LibraryNativeMetadata::CheckedBlockPayloadCallback { .. }
            )
        );
        let checked = matches!(
            role,
            LibraryRoleAttr::Native(LibraryNativeMetadata::CheckedBlockPayloadCallback { .. })
        );
        if checked != sources.checked_environments.contains_key(site) {
            return Err(format!(
                "checked callback nominal environment is missing or mismatched at {} block {}",
                site.caller, site.block
            ));
        }
        if callback != sources.roots.contains_key(site) {
            return Err(format!(
                "retained callback source identity disagrees at {} block {}",
                site.caller, site.block
            ));
        }
    }
    // Original checked obligations survive even if preparation removes a call.
    for (site, environment) in &sources.checked_environments {
        if !sources.roots.contains_key(site) {
            return Err("checked callback environment has no selected source thunk".into());
        }
        if !matches!(
            environment,
            LibraryCallbackEnvironment::FunctionItem
                | LibraryCallbackEnvironment::Closure {
                    capture_count: 0,
                    single_capture: None,
                }
                | LibraryCallbackEnvironment::Closure {
                    capture_count: 1,
                    single_capture: Some(dialect_mir::library::LibraryScalar::Signed(32)),
                }
        ) {
            return Err(format!(
                "checked callback requires a function item or Fn closure with zero captures or one i32 capture at {} block {}",
                site.caller, site.block
            ));
        }
    }
    let functions = admission::functions(ctx, module);
    let mut visited = BTreeSet::new();
    let mut pending: Vec<_> = sources.roots.values().cloned().collect();
    while let Some(identity) = pending.pop() {
        if !visited.insert(identity.clone()) {
            continue;
        }
        let function = sources.functions.get(&identity).ok_or_else(|| {
            format!("scalar callback source body classification is missing for `{identity}`")
        })?;
        pending.extend(function.callees.iter().cloned());
        match function.definition {
            Definition::Cuda => {
                return Err(format!(
                    "scalar callback cannot reach CUDA resource or collective operation `{}`",
                    function.diagnostic
                ));
            }
            Definition::Foreign => {
                return Err("scalar callback cannot reach a foreign function".into());
            }
            Definition::Intrinsic => {
                return Err(
                    "scalar callback cannot reach an unqualified compiler intrinsic".into(),
                );
            }
            Definition::Rust => {}
        }
        if function.dispatch == Dispatch::Indirect {
            return Err("scalar callback cannot contain an indirect call".into());
        }
        if !function.has_body {
            // The collector roots the actual receiver from the caller's site,
            // alongside its shim edge, rather than as a child of this shim.
            if matches!(
                function.dispatch,
                Dispatch::FunctionItemShim | Dispatch::ClosureOnceShim
            ) {
                continue;
            }
            return Err(format!(
                "scalar callback body is unavailable for `{}`",
                function.diagnostic
            ));
        }
        if let Some(ty) = function.cuda_local_types.first() {
            return Err(format!(
                "scalar callback cannot contain CUDA resource type `{ty}`"
            ));
        }
        for call in &function.calls {
            match call {
                LibraryCallbackCall::InlineAssembly => {
                    return Err("scalar callback cannot contain inline assembly".into());
                }
                LibraryCallbackCall::Indirect => {
                    return Err("scalar callback cannot contain an indirect call".into());
                }
                LibraryCallbackCall::Direct {
                    definition: Definition::Cuda | Definition::Foreign,
                    diagnostic,
                } => {
                    return Err(format!(
                        "scalar callback cannot reach CUDA resources or foreign calls: `{diagnostic}`"
                    ));
                }
                LibraryCallbackCall::Direct {
                    definition: Definition::Intrinsic,
                    ..
                } => {
                    return Err(
                        "scalar callback cannot reach an unqualified compiler intrinsic".into(),
                    );
                }
                LibraryCallbackCall::Direct {
                    definition: Definition::Rust,
                    ..
                } => {}
            }
        }
    }
    let mut retained = BTreeMap::new();
    for identity in sources.roots.values() {
        let function = &sources.functions[identity];
        let export = function
            .export
            .as_ref()
            .ok_or_else(|| format!("callback thunk `{identity}` has no collected export"))?;
        let symbol = symbols
            .get(export)
            .ok_or_else(|| format!("callback thunk `{export}` has no imported symbol"))?;
        let op = functions
            .get(symbol)
            .ok_or_else(|| format!("callback thunk `{export}` has no imported definition"))?;
        let actual = MirFuncOp::wrap(ctx, *op)
            .unwrap()
            .library_function_source(ctx);
        if !actual.is_some_and(|actual| actual.export == *export && actual.mangled == *identity)
            || op
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .iter(ctx)
                .next()
                .is_none()
        {
            return Err(format!(
                "callback thunk `{export}` does not match its actual definition"
            ));
        }
        retained.insert(identity.clone(), symbol.clone());
    }
    Ok(retained)
}
