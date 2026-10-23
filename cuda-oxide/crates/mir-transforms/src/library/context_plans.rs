/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assemble the existing contextual names and finalized plans from MIR proofs.

use super::stateful::Validation;
use dialect_mir::{
    library::{LibraryCallPlan, LibraryCallSite, LibraryFunctionSourceAttr},
    ops::MirFuncOp,
};
use oxide_artifacts::KernelLaunchFacts;
use pliron::{
    context::{Context, Ptr},
    graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op},
    operation::Operation,
};
use std::collections::{BTreeMap, BTreeSet};

/// Provider plans indexed by original root and source-native call site.
pub type RootedNativePlans = BTreeMap<String, BTreeMap<LibraryCallSite, LibraryCallPlan>>;

/// Owned final inputs consumed by the existing context materializer and lowerer.
#[derive(Default)]
pub struct LibraryAssembly {
    /// Emitted raw export to original raw export; legalization remains importer-owned.
    pub contexts: BTreeMap<String, String>,
    pub calls: BTreeMap<LibraryCallSite, LibraryCallPlan>,
    pub redirects: BTreeMap<String, BTreeMap<String, String>>,
    pub roots: BTreeMap<String, KernelLaunchFacts>,
}

fn source<'a>(
    sources: &'a BTreeMap<String, LibraryFunctionSourceAttr>,
    export: &str,
) -> Result<&'a LibraryFunctionSourceAttr, String> {
    sources
        .get(export)
        .ok_or_else(|| format!("library context has no authenticated source {export}"))
}

fn insert_context(
    contexts: &mut BTreeMap<String, String>,
    emitted: String,
    original: String,
) -> Result<(), String> {
    if let Some(previous) = contexts.insert(emitted, original.clone())
        && previous != original
    {
        return Err("finalized helper context has conflicting source identities".into());
    }
    Ok(())
}

fn insert_call(
    calls: &mut BTreeMap<LibraryCallSite, LibraryCallPlan>,
    site: LibraryCallSite,
    plan: LibraryCallPlan,
) -> Result<(), String> {
    if let Some(previous) = calls.get(&site) {
        if previous != &plan {
            return Err("library call site has conflicting provider specializations".into());
        }
    } else {
        calls.insert(site, plan);
    }
    Ok(())
}

/// Preserve the existing checked-Block contextual naming and all-target redirects.
///
/// Provider requests retain original source reachability. This function performs
/// no proof or IR mutation; the prepared-MIR proof already succeeded.
#[allow(clippy::too_many_arguments)]
pub fn assemble_library_plans(
    module: Ptr<Operation>,
    ctx: &Context,
    rooted_functions: &BTreeMap<String, BTreeSet<String>>,
    source_targets: &BTreeMap<LibraryCallSite, BTreeSet<String>>,
    validation: &Validation,
    native_plans: &RootedNativePlans,
    provider_roots: BTreeMap<String, KernelLaunchFacts>,
) -> Result<LibraryAssembly, String> {
    let mut captured = (BTreeMap::<String, LibraryFunctionSourceAttr>::new(), false);
    walk_op(
        ctx,
        &mut captured,
        &WalkConfig::default(),
        module,
        |ctx, (sources, conflict), node| {
            let IRNode::Operation(op) = node else { return };
            if let Some(function) = Operation::get_op::<MirFuncOp>(op, ctx)
                && let Some(identity) = function.library_function_source(ctx)
                && let Some(previous) = sources.insert(identity.export.clone(), identity.clone())
                && previous != identity
            {
                *conflict = true;
            }
        },
    );
    let (sources, conflict) = captured;
    if conflict {
        return Err("library context has conflicting authenticated source identities".into());
    }
    let mut result = LibraryAssembly {
        roots: provider_roots,
        ..Default::default()
    };
    let roots: BTreeSet<_> = native_plans
        .keys()
        .chain(validation.shapes.keys())
        .collect();
    for root in roots {
        let mut context = BTreeMap::new();
        if validation.shapes.contains_key(root) {
            let root_source = source(&sources, root)?;
            for original in rooted_functions
                .get(root)
                .ok_or("checked root has no source reachability")?
            {
                let Some(identity) = sources
                    .get(original)
                    .filter(|source| source.checked_ancestor)
                else {
                    continue;
                };
                let name = if original == root {
                    root.clone()
                } else {
                    format!(
                        "{}{}_{}_{}",
                        reserved_oxide_symbols::LIBRARY_CONTEXT_PREFIX,
                        root_source.mangled.len(),
                        root_source.mangled,
                        identity.mangled,
                    )
                };
                insert_context(&mut result.contexts, name.clone(), original.clone())?;
                context.insert(original.clone(), name);
            }
            for (original, emitted) in &context {
                let redirects = result.redirects.entry(emitted.clone()).or_default();
                for (site, targets) in source_targets {
                    if site.caller != *original {
                        continue;
                    }
                    for target in targets {
                        if let Some(target_context) = context.get(target) {
                            redirects.insert(target.clone(), target_context.clone());
                        }
                    }
                }
            }
        }
        if let Some(natives) = native_plans.get(root) {
            for (site, plan) in natives {
                insert_call(
                    &mut result.calls,
                    LibraryCallSite {
                        caller: context.get(&site.caller).unwrap_or(&site.caller).clone(),
                        block: site.block,
                    },
                    plan.clone(),
                )?;
            }
        }
    }
    Ok(result)
}
