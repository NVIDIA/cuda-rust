/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Realize the existing finalized helper contexts without importing bodies again.

use super::*;
use dialect_mir::{
    library::LibraryOperation,
    ops::{MirCallOp, MirFuncOp},
};
use pliron::{
    builtin::op_interfaces::SymbolOpInterface,
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::IRRewriter,
    },
};

struct Rewrite {
    op: Ptr<Operation>,
    callee: Option<String>,
    site: Option<LibraryCallSite>,
}

struct ContextRewrite {
    function: Ptr<Operation>,
    emitted_symbol: String,
    clone: bool,
    operations: Vec<Rewrite>,
}

/// Resolve one original ordinary call under an existing finalized context.
///
/// Used for both actual calls and saved pre-preparation resource edges. An exact
/// call-site Redirect takes precedence over the context's callee redirects.
/// Foreign calls without a collected target remain unchanged; contextual
/// rewrites require the actual original target to be authenticated. Mangled
/// marker identities are resolved separately from collected export names.
#[allow(clippy::too_many_arguments)]
pub fn resolve_library_call_target(
    source_site: Option<&LibraryCallSite>,
    emitted_caller: &str,
    actual_callee: &str,
    symbols: &BTreeMap<String, String>,
    source_targets: &BTreeMap<LibraryCallSite, BTreeSet<String>>,
    marker_exports: &BTreeMap<String, String>,
    plans: &BTreeMap<LibraryCallSite, LibraryCallPlan>,
    redirects: &BTreeMap<String, BTreeMap<String, String>>,
) -> std::result::Result<String, String> {
    let mut target = None;
    let mut authenticated = false;
    if let Some(site) = source_site {
        if let Some(targets) = source_targets.get(site) {
            authenticated = targets.iter().any(|target| {
                symbols
                    .get(target)
                    .is_some_and(|symbol| symbol == actual_callee)
            });
            if !authenticated {
                return Err("ordinary call does not match its authenticated source targets".into());
            }
        }
        let emitted_site = LibraryCallSite {
            caller: emitted_caller.into(),
            block: site.block,
        };
        if let Some(plan) = plans.get(&emitted_site)
            && plan.operation == LibraryOperation::Redirect
        {
            if !marker_exports
                .get(&plan.marker)
                .and_then(|original| symbols.get(original))
                .is_some_and(|symbol| symbol == actual_callee)
            {
                return Err("library redirect does not match its original callee".into());
            }
            target = Some(&plan.symbol);
        }
    }
    if target.is_none()
        && let Some(redirects) = redirects.get(emitted_caller)
    {
        target = redirects
            .iter()
            .find(|(original, _)| {
                symbols
                    .get(*original)
                    .is_some_and(|symbol| symbol == actual_callee)
            })
            .map(|(_, target)| target);
    }
    let Some(target) = target else {
        return Ok(actual_callee.into());
    };
    if !authenticated {
        return Err("library redirect requires an authenticated original call site".into());
    }
    symbols
        .get(target)
        .cloned()
        .ok_or_else(|| format!("library redirect target {target} has no legalized symbol"))
}

/// Realize finalized context names and redirects in the same prepared module.
///
/// `contexts` maps emitted raw names to original raw names. `symbols` contains
/// both, legalized by the importer's original Legaliser. Plans and redirects
/// retain their existing raw-name conventions. Ordinary call-site attributes
/// remain original; library site callers are rebound to the emitted context.
/// Only affected functions are cloned. Module globals, static source keys,
/// locations and debug attributes are preserved by Pliron's same-context clone.
/// All requested rewrites are checked before mutation; cache entries are then
/// invalidated before cloning. The importer separately remaps saved alignments
/// and reattaches debug-global owner names before resource completion.
#[allow(clippy::too_many_arguments)]
pub fn materialize_library_contexts(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    contexts: &BTreeMap<String, String>,
    symbols: &BTreeMap<String, String>,
    source_targets: &BTreeMap<LibraryCallSite, BTreeSet<String>>,
    marker_exports: &BTreeMap<String, String>,
    plans: &BTreeMap<LibraryCallSite, LibraryCallPlan>,
    redirects: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<()> {
    let functions = admission::functions(ctx, module);
    let sources: BTreeSet<_> = contexts.values().cloned().collect();
    let mut targets: BTreeMap<_, _> = symbols
        .iter()
        .filter(|(source, symbol)| !sources.contains(*source) && functions.contains_key(*symbol))
        .map(|(source, _)| (source.clone(), source.clone()))
        .collect();
    targets.extend(contexts.clone());
    let final_symbols: BTreeSet<_> = targets
        .keys()
        .filter_map(|target| symbols.get(target))
        .collect();
    let removed: BTreeSet<_> = sources
        .iter()
        .filter(|source| !targets.contains_key(*source))
        .filter_map(|source| symbols.get(source).cloned())
        .collect();
    let mut emitted_symbols = BTreeSet::new();
    let mut rewrites = Vec::new();
    let mut seen_redirects = BTreeSet::new();
    for (emitted, source) in &targets {
        let Some(&function) = symbols.get(source).and_then(|symbol| functions.get(symbol)) else {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "library context source {source} has no imported function"
            );
        };
        let Some(emitted_symbol) = symbols.get(emitted) else {
            return pliron::verify_err!(
                function.deref(ctx).loc(),
                "library context {emitted} has no legalized symbol"
            );
        };
        if !emitted_symbols.insert(emitted_symbol.clone()) {
            return pliron::verify_err!(
                function.deref(ctx).loc(),
                "duplicate emitted library context symbol {emitted_symbol}"
            );
        }
        let clone = source != emitted;
        if clone && functions.contains_key(emitted_symbol) {
            return pliron::verify_err!(
                function.deref(ctx).loc(),
                "library context collides with existing function {emitted_symbol}"
            );
        }
        if clone
            && function
                .deref(ctx)
                .attributes
                .get::<StringAttr>(&"gpu_kernel".try_into().unwrap())
                .is_some()
        {
            return pliron::verify_err!(
                function.deref(ctx).loc(),
                "library contexts cannot clone a kernel entry"
            );
        }
        let mut operations = Vec::new();
        for op in admission::operations(ctx, function) {
            if let Some(library) = Operation::get_op::<MirLibraryOp>(op, ctx) {
                op.deref(ctx).verify(ctx)?;
                let mut site = (*library.get_attr_site(ctx).unwrap()).clone();
                if site.caller != *source {
                    return pliron::verify_err!(
                        op.deref(ctx).loc(),
                        "library context contains a different source caller"
                    );
                }
                site.caller = emitted.clone();
                operations.push(Rewrite {
                    op,
                    callee: None,
                    site: Some(site),
                });
            }
            let Some(call) = Operation::get_op::<MirCallOp>(op, ctx) else {
                continue;
            };
            let Some(callee) = call.get_attr_callee(ctx) else {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "library context call has no callee"
                );
            };
            let original_callee = String::from((*callee).clone());
            let source_site = call
                .get_attr_library_source_site(ctx)
                .map(|site| (*site).clone());
            if let Some(site) = &source_site {
                if site.caller != *source {
                    return pliron::verify_err!(
                        op.deref(ctx).loc(),
                        "ordinary call site has a different source caller"
                    );
                }
                let emitted_site = LibraryCallSite {
                    caller: emitted.clone(),
                    block: site.block,
                };
                if let Some(plan) = plans.get(&emitted_site)
                    && plan.operation == LibraryOperation::Redirect
                    && !seen_redirects.insert(emitted_site)
                {
                    return pliron::verify_err!(
                        op.deref(ctx).loc(),
                        "duplicate original library helper call site"
                    );
                }
            }
            let callee = match resolve_library_call_target(
                source_site.as_ref(),
                emitted,
                &original_callee,
                symbols,
                source_targets,
                marker_exports,
                plans,
                redirects,
            ) {
                Ok(callee) => callee,
                Err(message) => return pliron::verify_err!(op.deref(ctx).loc(), "{message}"),
            };
            if callee != original_callee && !final_symbols.contains(&callee) {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "library redirect target {callee} has no emitted function"
                );
            }
            if removed.contains(&callee) {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "ordinary call still refers to replaced library helper {callee}"
                );
            }
            operations.push(Rewrite {
                op,
                callee: Some(callee),
                site: None,
            });
        }
        rewrites.push(ContextRewrite {
            function,
            emitted_symbol: emitted_symbol.clone(),
            clone,
            operations,
        });
    }
    for (site, plan) in plans {
        if plan.operation == LibraryOperation::Redirect && !seen_redirects.contains(site) {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "finalized library redirect has no original call site {} block {}",
                site.caller,
                site.block
            );
        }
    }
    if contexts.is_empty()
        && redirects.is_empty()
        && !plans
            .values()
            .any(|plan| plan.operation == LibraryOperation::Redirect)
    {
        return Ok(());
    }
    let mut invalidated = PassResult::default();
    invalidated.ir_changed = IRStatus::Changed;
    analyses.retain_preserved(&invalidated);
    let mut rewriter = IRRewriter::<DummyListener>::default();
    for rewrite in rewrites {
        let mut mapping = IrMapping::new();
        let function = if rewrite.clone {
            let clone = clone_operation(rewrite.function, ctx, &mut rewriter, &mut mapping);
            clone.insert_before(ctx, rewrite.function);
            MirFuncOp::wrap(ctx, clone)
                .unwrap()
                .set_symbol_name(ctx, rewrite.emitted_symbol.try_into().unwrap());
            clone
        } else {
            rewrite.function
        };
        for update in rewrite.operations {
            let op = if rewrite.clone {
                mapping.lookup_op(update.op).unwrap()
            } else {
                update.op
            };
            if let Some(callee) = update.callee {
                MirCallOp::new(op).set_attr_callee(ctx, StringAttr::new(callee));
            }
            if let Some(site) = update.site {
                MirLibraryOp::new(op).set_attr_site(ctx, site);
            }
        }
        // Verification of cross-context calls waits until every clone exists.
        debug_assert!(function.deref(ctx).get_parent_block().is_some());
    }
    for symbol in removed {
        Operation::erase(functions[&symbol], ctx);
    }
    module.deref(ctx).verify(ctx)
}
