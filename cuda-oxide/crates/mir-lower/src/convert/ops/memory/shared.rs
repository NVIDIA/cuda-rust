/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversion of `mir.shared_alloc` to static shared-memory globals.

use super::common::{anyhow_to_pliron, counter_named_global};
use crate::context::{
    SharedGlobalDeclaration, SharedGlobalKind, SharedGlobalRecord, SharedGlobalsMap,
};
use crate::convert::types::{convert_type, llvm_type_size_align};
use crate::helpers;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::ArrayType;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};

/// Convert `mir.shared_alloc` to LLVM global variable in shared address space.
///
/// GPU shared memory is represented as a global variable with address space 3.
/// Uses `shared_globals` to deduplicate: multiple allocations with the same
/// `alloc_key` share the same global.
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs the cross-function `SharedGlobalsMap`.
pub fn convert_shared_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
    next_shared_mem_index: &mut usize,
) -> Result<()> {
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};

    let (alloc_key, source_name, mir_elem_type, size, alignment, debug_info, debug_owner_function) = {
        let shared_alloc_op = dialect_mir::ops::MirSharedAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let alloc_key: Option<String> = shared_alloc_op
            .get_attr_alloc_key(ctx)
            .map(|s| String::from((*s).clone()));

        // Optional and diagnostic: the Rust path of the originating `static`,
        // carried through so the emitted global can name its source.
        let source_name: Option<String> = shared_alloc_op
            .get_attr_source_name(ctx)
            .map(|s| String::from((*s).clone()));

        let elem_type_attr = op_ref
            .attributes
            .get::<TypeAttr>(&"elem_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing elem_type TypeAttr attribute"
                ))
            })?;
        let mir_elem_type = elem_type_attr.get_type(ctx);

        let size_attr = op_ref
            .attributes
            .get::<IntegerAttr>(&"size".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing size IntegerAttr attribute"
                ))
            })?;
        let size = size_attr.value().to_u64();

        let alignment = shared_alloc_op.get_alignment_value(ctx).unwrap_or(0);
        let debug_info = llvm::debug_global_variable(ctx, op);
        let debug_owner_function = llvm::debug_global_owner_function(ctx, op);

        (
            alloc_key,
            source_name,
            mir_elem_type,
            size,
            alignment,
            debug_info,
            debug_owner_function,
        )
    };

    // A shared variable DIE must describe the physical allocation. Gate the
    // optional debug identity against the lowered element layout before it
    // can reach the declaration record or the emitted global.
    let llvm_elem_type = convert_type(ctx, mir_elem_type).map_err(anyhow_to_pliron)?;
    let debug_info = debug_info
        .filter(|info| shared_debug_type_matches_physical(ctx, info, llvm_elem_type, size));
    let debug_owner_function = debug_info.as_ref().and(debug_owner_function);

    let declaration = SharedGlobalDeclaration {
        kind: SharedGlobalKind::Static,
        mir_elem_type,
        size,
        alignment,
        debug_info,
        debug_owner_function,
    };

    // A key names one physical allocation, not merely a preferred symbol.
    // Reuse it only when the complete storage declaration agrees; otherwise
    // the later address would silently inherit the first allocation's type,
    // extent, or alignment.
    let global_name = if let Some(key) = alloc_key.as_ref()
        && let Some(existing) = shared_globals.get(key)
    {
        if !existing.declaration.same_storage(&declaration) {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "duplicate shared alloc_key {:?} has an incompatible declaration",
                key
            )));
        }
        let symbol = existing.symbol.clone();
        let debug_identity_matches = existing.declaration.debug_info == declaration.debug_info
            && existing.declaration.debug_owner_function == declaration.debug_owner_function;
        if !debug_identity_matches {
            // One physical allocation reached under two debug identities, for
            // example a function-local static whose owning function was
            // materialized (or inlined) into two kernels. Debug metadata must
            // never fail a build that a release build accepts, and DWARF
            // cannot truthfully scope one AS3 object to two subprograms, so
            // fail open: drop the attachment from the materialized global and
            // demote the cached record to metadata-free.
            strip_shared_debug_identity(ctx, op, &symbol)?;
            if let Some(record) = shared_globals.get_mut(key) {
                record.declaration.debug_info = None;
                record.declaration.debug_owner_function = None;
            }
        }
        symbol
    } else {
        create_shared_global(
            ctx,
            op,
            shared_globals,
            next_shared_mem_index,
            SharedAllocSpec {
                mir_elem_type,
                size,
                alignment,
                alloc_key,
                source_name: source_name.as_deref(),
                debug_info: declaration.debug_info.as_ref(),
                debug_owner_function: declaration.debug_owner_function.as_deref(),
            },
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

/// Everything `create_shared_global` needs about one `mir.shared_alloc`.
///
/// Mirrors [`DeviceGlobalSpec`] for the shared-memory path.
struct SharedAllocSpec<'a> {
    mir_elem_type: TypeHandle,
    size: u64,
    alignment: u64,
    alloc_key: Option<String>,
    source_name: Option<&'a str>,
    debug_info: Option<&'a llvm::DebugGlobalVariableInfo>,
    debug_owner_function: Option<&'a str>,
}

/// Create a shared memory global variable in the module.
///
/// Creates an LLVM global variable with:
/// - Array type: `[size x element_type]`
/// - Address space 3 (shared memory)
/// - Optional alignment
/// - Unique generated name (`__shared_mem_N`)
///
/// The global is inserted at the front of the module block. When
/// `spec.alloc_key` is `Some`, the key is moved into `shared_globals` so that
/// later allocations with the same key reuse this global only after the caller
/// has checked that their complete declarations agree.
///
/// `spec.source_name`, when present, is the Rust path of the `static` this
/// allocation came from. The generated symbol stays anonymous; the name is
/// recorded as an attribute on the global so the exporter can render it
/// beside the definition. Only the allocation that *creates* the global
/// contributes a name — a later allocation with the same `alloc_key` hits
/// the cache and never reaches this function — which is consistent because
/// the key and the name are both derived from the same constant.
///
/// `next_shared_mem_index` is scoped to one `MirToLlvmConversionDriver`
/// instance (one module), not a process-global counter: `N` is a function of
/// this module's own MIR walk order, not of how many other modules have
/// lowered a shared allocation earlier in the process (#706).
fn create_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    shared_globals: &mut SharedGlobalsMap,
    next_shared_mem_index: &mut usize,
    spec: SharedAllocSpec<'_>,
) -> Result<pliron::identifier::Identifier> {
    let llvm_elem_type = convert_type(ctx, spec.mir_elem_type).map_err(anyhow_to_pliron)?;
    let array_type = ArrayType::get(ctx, llvm_elem_type, spec.size);

    let counter = *next_shared_mem_index;
    *next_shared_mem_index += 1;
    let name: pliron::identifier::Identifier = counter_named_global(ctx, "__shared_mem", counter)
        .try_into()
        .unwrap();

    let global_op = if spec.alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), array_type.into(), spec.alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), array_type.into())
    };
    global_op.set_address_space(ctx, llvm_export::types::address_space::SHARED);
    if let Some(source_name) = spec.source_name {
        use llvm_export::ops::GlobalOpExt;
        global_op.set_shared_source_name(ctx, source_name);
    }
    if let Some(info) = spec.debug_info {
        llvm::set_debug_global_variable(ctx, global_op.get_operation(), info);
    }
    if let Some(owner) = spec.debug_owner_function {
        llvm::set_debug_global_owner_function(ctx, global_op.get_operation(), owner);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    if let Some(key) = spec.alloc_key {
        shared_globals.insert(
            key,
            SharedGlobalRecord {
                symbol: name.clone(),
                declaration: SharedGlobalDeclaration {
                    kind: SharedGlobalKind::Static,
                    mir_elem_type: spec.mir_elem_type,
                    size: spec.size,
                    alignment: spec.alignment,
                    debug_info: spec.debug_info.cloned(),
                    debug_owner_function: spec.debug_owner_function.map(str::to_owned),
                },
            },
        );
    }

    Ok(name)
}

/// A shared variable DIE must describe the compiler-materialized allocation,
/// never the marker ZST or an element graph whose pointer base types are not
/// representable yet. The frontend checks semantic element layout; this final
/// boundary verifies total physical bits/count against the LLVM backing.
fn shared_debug_type_matches_physical(
    ctx: &Context,
    info: &llvm::DebugGlobalVariableInfo,
    llvm_elem_type: TypeHandle,
    count: u64,
) -> bool {
    let Some((element_size, _)) = llvm_type_size_align(ctx, llvm_elem_type) else {
        return false;
    };
    let Some(element_bytes) = element_size.checked_mul(count) else {
        return false;
    };
    let Some(physical_bits) = element_bytes.checked_mul(8) else {
        return false;
    };
    if info.ty.size_bits() != physical_bits {
        return false;
    }
    match &info.ty {
        llvm::DebugLocalTypeKind::Array {
            element,
            count: debug_count,
            ..
        } => *debug_count == count && element.size_bits() == element_size.saturating_mul(8),
        // `Barrier` retains its semantic repr(C) struct while the backing is
        // one i64. Other marker types are never admitted by the frontend.
        llvm::DebugLocalTypeKind::Struct { .. } => count == 1,
        _ => false,
    }
}

/// Detach the debug identity from an already-materialized shared global.
///
/// The fail-open path for divergent debug identities on one `alloc_key`:
/// the storage stays shared and valid, only the optional DWARF attachment is
/// dropped so it cannot misattribute the allocation to the wrong static or
/// owner function.
fn strip_shared_debug_identity(
    ctx: &mut Context,
    op: Ptr<Operation>,
    name: &pliron::identifier::Identifier,
) -> Result<()> {
    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let module_block = module_op
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;
    let existing = module_block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|candidate| Operation::get_op::<llvm::GlobalOp>(candidate, ctx))
        .find(|global| global.get_symbol_name(ctx) == *name)
        .ok_or_else(|| {
            anyhow_to_pliron(anyhow::anyhow!(
                "shared allocation cache refers to missing LLVM global `@{name}`"
            ))
        })?;
    llvm::clear_debug_global_identity(ctx, existing.get_operation());
    Ok(())
}
