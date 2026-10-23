/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provider-independent admission using original effects and prepared root IR.

use dialect_mir::{library::LibraryNativeMetadata, ops::MirFuncOp};
use dialect_mir::{
    library::{LibraryCallSite, LibraryRoleAttr},
    ops::MirLibraryOp,
};
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::{
    builtin::attributes::IntegerAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op},
    operation::Operation,
    result::Result,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn functions(ctx: &Context, module: Ptr<Operation>) -> BTreeMap<String, Ptr<Operation>> {
    let mut functions = BTreeMap::new();
    walk_op(
        ctx,
        &mut functions,
        &WalkConfig::default(),
        module,
        |ctx, functions, node| {
            if let IRNode::Operation(op) = node
                && let Some(function) = MirFuncOp::wrap(ctx, op)
            {
                functions.insert(function.get_symbol_name(ctx).to_string(), op);
            }
        },
    );
    functions
}

pub(super) fn operations(ctx: &Context, function: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut operations = Vec::new();
    walk_op(
        ctx,
        &mut operations,
        &WalkConfig::default(),
        function,
        |_, operations, node| {
            if let IRNode::Operation(op) = node {
                operations.push(op);
            }
        },
    );
    operations
}

/// Capture genuine retained effects before preparation can remove a source site.
///
/// `symbols` maps original caller identities to their imported physical names.
pub fn capture_library_roles(
    module: Ptr<Operation>,
    ctx: &Context,
    symbols: &BTreeMap<String, String>,
) -> Result<BTreeMap<LibraryCallSite, LibraryRoleAttr>> {
    let functions = functions(ctx, module);
    let mut roles = BTreeMap::new();
    for (symbol, function) in functions {
        for op in operations(ctx, function) {
            let Some(library) = Operation::get_op::<MirLibraryOp>(op, ctx) else {
                continue;
            };
            op.deref(ctx).verify(ctx)?;
            let site = (*library.get_attr_site(ctx).unwrap()).clone();
            if symbols.get(&site.caller) != Some(&symbol) {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "retained library site has a different source caller"
                );
            }
            let role = *library.get_attr_role(ctx).unwrap();
            if role == LibraryRoleAttr::NativeOperandBoundary {
                continue;
            }
            if roles.insert(site, role).is_some() {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "duplicate retained library source site"
                );
            }
        }
    }
    Ok(roles)
}

fn checked(role: LibraryRoleAttr) -> bool {
    matches!(
        role,
        LibraryRoleAttr::Native(
            LibraryNativeMetadata::CheckedBlock { .. }
                | LibraryNativeMetadata::CheckedBlockResource { .. }
                | LibraryNativeMetadata::CheckedBlockStorage { .. }
                | LibraryNativeMetadata::CheckedWarp { .. }
                | LibraryNativeMetadata::CheckedBlockPayload { .. }
                | LibraryNativeMetadata::CheckedBlockPayloadCallback { .. }
                | LibraryNativeMetadata::CheckedWarpPayload { .. }
        )
    ) || super::storage::planned_storage(role)
}

fn dimension(ctx: &Context, function: Ptr<Operation>, key: &str) -> Option<u32> {
    let key = key.try_into().unwrap();
    let function = function.deref(ctx);
    let value = function
        .attributes
        .get::<IntegerAttr>(&key)?
        .value()
        .to_u64();
    u32::try_from(value).ok().filter(|&value| value != 0)
}

/// Admit checked root geometry on the actual prepared module before provider I/O.
///
/// Root reachability and `original_roles` describe the original authenticated
/// source, not an optimized call graph. Removed calls therefore cannot remove a
/// root's obligation. Surviving effects must still match that snapshot. Raw
/// contracts retain their explicit geometry and do not require a
/// checked launch contract. This does not prove uniformity or completion.
pub fn admit_library_roots(
    module: Ptr<Operation>,
    ctx: &Context,
    roots: &BTreeMap<String, BTreeSet<String>>,
    symbols: &BTreeMap<String, String>,
    original_roles: &BTreeMap<LibraryCallSite, LibraryRoleAttr>,
) -> Result<BTreeMap<String, [u32; 3]>> {
    for (site, role) in capture_library_roles(module, ctx, symbols)? {
        if original_roles.get(&site) != Some(&role) {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "retained library effect changed after source capture at {} block {}",
                site.caller,
                site.block
            );
        }
    }
    let functions = functions(ctx, module);
    let mut shapes = BTreeMap::new();
    for (root, reached) in roots {
        let roles: Vec<_> = original_roles
            .iter()
            .filter(|(site, _)| reached.contains(&site.caller))
            .map(|(_, &role)| role)
            .collect();
        if !roles.iter().copied().any(checked) {
            continue;
        }
        let Some(&function) = symbols.get(root).and_then(|symbol| functions.get(symbol)) else {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "checked library root {root} has no imported function"
            );
        };
        let dimensions =
            ["reqntid_x", "reqntid_y", "reqntid_z"].map(|key| dimension(ctx, function, key));
        let [Some(x), Some(y), Some(z)] = dimensions else {
            return pliron::verify_err!(
                function.deref(ctx).loc(),
                "checked block collective root {root} requires #[launch_contract(block = (...))] with an exact shape"
            );
        };
        let shape = [x, y, z];
        for role in &roles {
            if let LibraryRoleAttr::Native(
                LibraryNativeMetadata::CheckedWarp { width, .. }
                | LibraryNativeMetadata::CheckedWarpPayload { width, .. },
            ) = role
                && (!(matches!(*width, 1 | 2 | 4 | 8 | 16 | 32)
                    || matches!(
                        role,
                        LibraryRoleAttr::Native(LibraryNativeMetadata::CheckedWarpPayload { .. })
                    ) && (17..=31).contains(width))
                    || x.checked_mul(y)
                        .and_then(|xy| xy.checked_mul(z))
                        .is_none_or(|threads| threads % 32 != 0))
            {
                return pliron::verify_err!(
                    function.deref(ctx).loc(),
                    "checked Warp calls require supported width and an exact block containing complete physical warps"
                );
            }
        }
        shapes.insert(root.clone(), shape);
    }
    Ok(shapes)
}
