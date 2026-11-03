/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helpers shared by the memory-op conversions in this module.

use crate::convert::types::{
    StructLayoutInfo, build_struct_slot_map, convert_type,
    llvm_packed_struct_contains_pointer_in_address_space, mir_type_abi_align,
};
use dialect_mir::types::MirStructType;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::OperandsInfo;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Name for the `index`-th counter-named module-scope global of one family
/// (`__shared_mem`, `__device_global`). See
/// [`crate::context::module_namespaced_symbol`] for why the module
/// disambiguator is part of the name.
pub(super) fn counter_named_global(ctx: &Context, family: &str, index: usize) -> String {
    crate::context::module_namespaced_symbol(ctx, family, &index.to_string())
}

pub(super) fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Recover the `repr(align(N))` ABI alignment of `value` at conversion time.
///
/// The alignment lives on MIR aggregate types (`abi_align`); the converted
/// LLVM struct types cannot express over-alignment. The conversion driver may
/// already have converted the value's type (block arguments are converted
/// before any rewrite runs; replaced op results carry the new type), but it
/// records every such change. So check the current type first, then walk the
/// value's conversion history, newest first, for a MIR type that records an
/// alignment.
pub(super) fn value_abi_align(
    ctx: &Context,
    operands_info: &OperandsInfo,
    value: Value,
) -> Option<u64> {
    mir_type_abi_align(ctx, value.get_type(ctx)).or_else(|| {
        operands_info
            .lookup_operand_history(value)
            .iter()
            .rev()
            .find_map(|ty| mir_type_abi_align(ctx, *ty))
    })
}

/// The MIR type of `value`: its current type when that is still a MIR type,
/// else the newest MIR type in its conversion history (same walk as
/// [`value_abi_align`], for the same reason).
pub(super) fn value_mir_type(
    ctx: &Context,
    operands_info: &OperandsInfo,
    value: Value,
) -> TypeHandle {
    let current = value.get_type(ctx);
    if current.deref(ctx).is::<MirStructType>() {
        return current;
    }
    operands_info
        .lookup_operand_history(value)
        .iter()
        .rev()
        .copied()
        .find(|ty| ty.deref(ctx).is::<MirStructType>())
        .unwrap_or(current)
}

/// Refuse only packed by-value images whose physical bytes depend on the
/// selected NVPTX mode.
///
/// Shared-memory pointers are 32-bit in modern NVVM p3:32 but 64-bit in the
/// PTX/legacy layouts. Lowering runs before that target mode is selected, so a
/// packed aggregate containing AS3 cannot safely be loaded or stored as one
/// physical value. Pointer-free packed aggregates and packed aggregates whose
/// pointers use target-stable address spaces are allowed.
pub(super) fn fail_on_target_dependent_packed_aggregate(
    ctx: &mut Context,
    ty: TypeHandle,
    verb: &str,
) -> Result<()> {
    let llvm_ty = {
        let layout = {
            let ty_ref = ty.deref(ctx);
            ty_ref
                .downcast_ref::<MirStructType>()
                .map(StructLayoutInfo::of_struct)
        };
        if let Some(layout) = layout {
            let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;
            if !map.by_value_layout_faithful {
                return pliron::input_err_noloc!(
                    "{} a struct whose rustc layout cannot be represented by an LLVM \
                     struct value is not supported",
                    verb
                );
            }
            map.llvm_struct_ty
        } else {
            convert_type(ctx, ty).map_err(anyhow_to_pliron)?
        }
    };
    if llvm_packed_struct_contains_pointer_in_address_space(
        ctx,
        llvm_ty,
        llvm_export::types::address_space::SHARED,
    ) {
        return pliron::input_err_noloc!(
            "{} a packed aggregate containing a shared-memory pointer by value is \
             target-mode dependent and is not yet supported",
            verb
        );
    }
    Ok(())
}

/// Carry Rust-local provenance from the MIR alloca to the LLVM alloca.
///
/// This is compiler-only metadata stored in the in-memory dialect operation;
/// it is not emitted as LLVM metadata because an unrecognized metadata node is
/// not a preservation contract for middle-end passes. The textual exporter
/// instead consumes the attribute to choose a stable SSA name before `opt`.
pub(super) fn copy_local_memory_provenance(
    ctx: &mut Context,
    mir_op: Ptr<Operation>,
    llvm_op: Ptr<Operation>,
) {
    if let Some(provenance) = llvm_export::ops::local_memory_provenance(ctx, mir_op) {
        llvm_export::ops::set_local_memory_provenance(ctx, llvm_op, provenance);
    }
}

/// Alignment the op that computed `ptr` recorded about the address it produced.
///
/// Field addresses stamp this during their own conversion, where the aggregate's
/// `abi_align` is still in hand; by the time a load that consumes the address is
/// converted, `mir.field_addr` is already a GEP and the MIR aggregate type is no
/// longer reachable from the load's operands.
///
/// Known fidelity limit: the claim rides the address value, not the place
/// access that justified it. `&raw const (*p).x` through a merely 4-aligned
/// `p: *const Aligned8` is legal Rust so long as the place itself is never
/// accessed, yet the later read through the raw pointer consumes this same
/// stamped GEP and inherits the aggregate's alignment, an over-claim for that
/// pointer. rustc keeps the two apart by threading alignment through place
/// evaluation and dropping it at the `&raw` boundary; dialect-mir erases that
/// distinction before lowering runs, so full fidelity would require stamping
/// loads during MIR translation instead. Accepted for now: the pattern needs
/// an actually under-aligned pointer to an over-aligned aggregate, projected
/// via `&raw` inside a kernel, and any direct access through such a place is
/// UB to begin with.
pub(super) fn pointer_proved_alignment(ctx: &Context, ptr: Value) -> Option<u64> {
    let defining_op = ptr.defining_op()?;
    llvm_export::ops::address_alignment(ctx, defining_op).map(u64::from)
}
