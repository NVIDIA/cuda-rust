/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verified lowering facts for packed-AS3 carrier local storage.
//!
//! The physical carrier representation is a storage property, not Rust pointer
//! provenance. This module therefore does not extend `MirPointerKind`. Instead,
//! immediately before dialect conversion it performs a closed-world walk over
//! MIR, proves every use of an eligible compiler-owned local address, and only
//! after the complete proof succeeds stamps the exact physical LLVM type on the
//! MIR operations that consume the address.
//!
//! Recursive struct/tuple projections and bounded fixed-array projections are
//! part of that proof. No lowering converter reconstructs carrier identity from
//! `OperandsInfo` or from an LLVM defining-op chain. Calls, casts, block-argument
//! edges, returns, pointer offsets, and any other unmodelled address transport
//! still fail before any MIR operation is lowered.

use crate::convert::target_stable_storage::{StorageRewriteOptions, target_stable_storage_type};
use crate::convert::types::{
    MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES, StructLayoutInfo, build_struct_slot_map,
    convert_type, llvm_type_size_align,
};
use dialect_mir::ops::{
    MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirCallOp, MirCastOp, MirCondBranchOp,
    MirDbgValueListOp, MirDbgValueOp, MirFieldAddrOp, MirGotoOp, MirLoadOp, MirPtrOffsetOp,
    MirReturnOp, MirStoreOp,
};
use dialect_mir::types::{MirArrayType, MirFP16Type, MirPtrType, MirStructType, MirTupleType};
use llvm_export::types as llvm_types;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_hash::FxHashMap;

const CARRIER_STORAGE_TYPE_KEY: &str = "cuda_oxide_packed_shared_carrier_storage_type";
const CARRIER_GEP_SOURCE_TYPE_KEY: &str = "cuda_oxide_packed_shared_carrier_gep_source_type";

#[derive(Clone, Copy, Debug)]
struct CarrierAddress {
    physical_pointee: TypeHandle,
}

#[derive(Clone, Copy, Debug)]
struct PackedSharedLocalStorageInfo {
    storage_ty: TypeHandle,
}

#[derive(Default)]
struct CarrierFactPlan {
    storage_types: Vec<(Ptr<Operation>, TypeHandle)>,
    gep_source_types: Vec<(Ptr<Operation>, TypeHandle)>,
}

impl CarrierFactPlan {
    fn plan_storage_type(&mut self, operation: Ptr<Operation>, ty: TypeHandle) {
        self.storage_types.push((operation, ty));
    }

    fn plan_gep_source_type(&mut self, operation: Ptr<Operation>, ty: TypeHandle) {
        self.gep_source_types.push((operation, ty));
    }

    fn apply(self, ctx: &mut Context) {
        for (operation, ty) in self.storage_types {
            set_type_attr(ctx, operation, CARRIER_STORAGE_TYPE_KEY, ty);
        }
        for (operation, ty) in self.gep_source_types {
            set_type_attr(ctx, operation, CARRIER_GEP_SOURCE_TYPE_KEY, ty);
        }
    }
}

fn attr_key(name: &str) -> Identifier {
    Identifier::try_new(name.to_string()).expect("static carrier attribute key must be valid")
}

fn get_type_attr(ctx: &Context, op: Ptr<Operation>, name: &str) -> Option<TypeHandle> {
    op.deref(ctx)
        .attributes
        .get::<TypeAttr>(&attr_key(name))
        .map(|attr| attr.get_type(ctx))
}

fn set_type_attr(ctx: &mut Context, op: Ptr<Operation>, name: &str, ty: TypeHandle) {
    op.deref_mut(ctx)
        .attributes
        .set(attr_key(name), TypeAttr::new(ty));
}

/// Physical storage type proven for `mir.alloca`, `mir.load`, or `mir.store`.
///
/// The attribute is created only by [`prepare_packed_shared_local_storage`]
/// after the complete whole-tree carrier plan validates, then is consumed
/// mechanically during lowering.
pub(crate) fn carrier_storage_type(ctx: &Context, op: Ptr<Operation>) -> Option<TypeHandle> {
    get_type_attr(ctx, op, CARRIER_STORAGE_TYPE_KEY)
}

/// Physical LLVM source type that a verified carrier projection must index.
///
/// For `mir.field_addr` this is the carrier struct/tuple type. For
/// `mir.array_element_addr` it is the carrier element type used as the typed
/// GEP source. The fact is attached to the exact MIR projection only after the
/// whole address-use graph validates.
pub(crate) fn carrier_gep_source_type(ctx: &Context, op: Ptr<Operation>) -> Option<TypeHandle> {
    get_type_attr(ctx, op, CARRIER_GEP_SOURCE_TYPE_KEY)
}

fn collect_operations(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut result = Vec::new();
    let mut pending = vec![root];
    while let Some(operation) = pending.pop() {
        let nested = {
            let op = operation.deref(ctx);
            op.regions()
                .flat_map(|region| region.deref(ctx).iter(ctx))
                .flat_map(|block| block.deref(ctx).iter(ctx))
                .collect::<Vec<_>>()
        };
        pending.extend(nested);
        result.push(operation);
    }
    result
}

fn reject_preexisting_carrier_facts(ctx: &Context, operations: &[Ptr<Operation>]) -> Result<()> {
    for &operation in operations {
        if carrier_storage_type(ctx, operation).is_some()
            || carrier_gep_source_type(ctx, operation).is_some()
        {
            return pliron::input_err_noloc!(
                "packed-AS3 carrier lowering facts must be created by mir-lower preparation, not supplied by input MIR"
            );
        }
    }
    Ok(())
}

/// Whether a MIR value shape belongs to the recursive packed-AS3 local lane.
///
/// This is deliberately a local-storage predicate. It mirrors the currently
/// admitted struct/tuple/array/scalar vocabulary without delegating admission
/// to the internal device ABI classifier, so a future ABI widening cannot make
/// new local-memory shapes legal implicitly.
fn packed_shared_local_mir_shape_is_supported(ctx: &Context, mir_ty: TypeHandle) -> bool {
    fn mir_type_is_zero_sized(ctx: &Context, ty: TypeHandle) -> bool {
        let ty_ref = ty.deref(ctx);
        if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
            return array_ty.size() == 0 || mir_type_is_zero_sized(ctx, array_ty.element_type());
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            return struct_ty
                .field_types
                .iter()
                .all(|field| mir_type_is_zero_sized(ctx, *field));
        }
        if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            return tuple_ty
                .get_types()
                .iter()
                .all(|field| mir_type_is_zero_sized(ctx, *field));
        }
        false
    }

    let children = {
        let ty_ref = mir_ty.deref(ctx);
        if ty_ref.is::<IntegerType>()
            || ty_ref.is::<MirFP16Type>()
            || ty_ref.is::<llvm_types::HalfType>()
            || ty_ref.is::<FP32Type>()
            || ty_ref.is::<FP64Type>()
            || ty_ref.is::<MirPtrType>()
            || ty_ref.is::<llvm_types::PointerType>()
        {
            return true;
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            Some(struct_ty.field_types.clone())
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            Some(tuple_ty.get_types().to_vec())
        } else {
            ty_ref
                .downcast_ref::<MirArrayType>()
                .map(|array_ty| vec![array_ty.element_type()])
        }
    };

    children.is_some_and(|children| {
        children.into_iter().all(|child| {
            mir_type_is_zero_sized(ctx, child)
                || packed_shared_local_mir_shape_is_supported(ctx, child)
        })
    })
}

/// Recognize the recursive/multi-leaf local-storage shape tracked by #1094.
///
/// The root remains a byte-faithful packed struct. Nested structs/tuples,
/// multiple AS3 leaves, and fixed arrays are admitted recursively. Vectors and
/// unrelated aggregate kinds remain fail-closed.
///
/// Array expansion reuses the existing 16-leaf rewrite budget because local
/// stores/loads invoke the same element-wise target-stable coercion as the
/// internal ABI. The check is nevertheless performed here, behind the separate
/// local shape predicate above, so ABI shape changes cannot widen this lane by
/// themselves.
fn packed_shared_local_storage_info(
    ctx: &mut Context,
    mir_ty: TypeHandle,
) -> std::result::Result<Option<PackedSharedLocalStorageInfo>, anyhow::Error> {
    let layout = {
        let ty_ref = mir_ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() else {
            return Ok(None);
        };
        StructLayoutInfo::of_struct(struct_ty)
    };

    if !packed_shared_local_mir_shape_is_supported(ctx, mir_ty) {
        return Ok(None);
    }

    let map = build_struct_slot_map(ctx, &layout)?;
    if !map.by_value_layout_faithful {
        return Ok(None);
    }
    let is_packed = map
        .llvm_struct_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::StructType>()
        .is_some_and(|struct_ty| struct_ty.layout() == llvm_types::StructLayout::Packed);
    if !is_packed {
        return Ok(None);
    }

    let rewrite = target_stable_storage_type(
        ctx,
        map.llvm_struct_ty,
        StorageRewriteOptions {
            canonicalize_bool: false,
        },
        "packed shared local storage",
    )?;
    if rewrite.shared_pointer_leaves == 0
        || rewrite.array_shared_pointer_leaves > MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES
    {
        return Ok(None);
    }
    let Some((storage_size, _)) = llvm_type_size_align(ctx, rewrite.ty) else {
        return Ok(None);
    };
    if layout.total_size > 0 && storage_size != layout.total_size {
        return Ok(None);
    }

    Ok(Some(PackedSharedLocalStorageInfo {
        storage_ty: rewrite.ty,
    }))
}

fn target_stable_local_value_type(
    ctx: &mut Context,
    semantic_mir_ty: TypeHandle,
    role: &str,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let semantic_llvm_ty = convert_type(ctx, semantic_mir_ty)?;
    Ok(target_stable_storage_type(
        ctx,
        semantic_llvm_ty,
        StorageRewriteOptions {
            canonicalize_bool: false,
        },
        role,
    )?
    .ty)
}

fn seed_carrier_allocas(
    ctx: &mut Context,
    operations: &[Ptr<Operation>],
    addresses: &mut FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    for &operation in operations {
        if Operation::get_op::<MirAllocaOp>(operation, ctx).is_none() {
            continue;
        }
        let result = operation.deref(ctx).get_result(0);
        let pointee = {
            let result_ty = result.get_type(ctx);
            let result_ref = result_ty.deref(ctx);
            let Some(pointer) = result_ref.downcast_ref::<MirPtrType>() else {
                continue;
            };
            pointer.pointee
        };
        let Some(info) = packed_shared_local_storage_info(ctx, pointee)
            .map_err(|error| pliron::input_error_noloc!("{error}"))?
        else {
            continue;
        };

        plan.plan_storage_type(operation, info.storage_ty);
        addresses.insert(
            result,
            CarrierAddress {
                physical_pointee: info.storage_ty,
            },
        );
    }
    Ok(())
}

fn projection_result_pointee(
    ctx: &Context,
    operation: Ptr<Operation>,
    kind: &str,
) -> Result<TypeHandle> {
    let result_ty = operation.deref(ctx).get_result(0).get_type(ctx);
    let result_ref = result_ty.deref(ctx);
    let pointer = result_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
        pliron::input_error_noloc!(
            "{} result must be a MIR pointer before packed-AS3 carrier preparation",
            kind
        )
    })?;
    Ok(pointer.pointee)
}

fn carrier_field_physical_pointee(
    ctx: &mut Context,
    operation: Ptr<Operation>,
    carrier_source_ty: TypeHandle,
) -> Result<TypeHandle> {
    let field_addr = MirFieldAddrOp::new(operation);
    let field_index = field_addr
        .get_attr_field_index(ctx)
        .ok_or_else(|| pliron::input_error_noloc!("MirFieldAddrOp missing field_index attribute"))?
        .0 as usize;
    let semantic_aggregate = field_addr
        .get_attr_aggregate_ty(ctx)
        .ok_or_else(|| {
            pliron::input_error_noloc!("MirFieldAddrOp missing verified aggregate_ty attribute")
        })?
        .get_type(ctx);

    let layout = {
        let aggregate_ref = semantic_aggregate.deref(ctx);
        if let Some(struct_ty) = aggregate_ref.downcast_ref::<MirStructType>() {
            StructLayoutInfo::of_struct(struct_ty)
        } else if let Some(tuple_ty) = aggregate_ref.downcast_ref::<MirTupleType>() {
            StructLayoutInfo::of_tuple(tuple_ty)
        } else {
            return pliron::input_err_noloc!(
                "packed-AS3 carrier field projection requires a struct or tuple aggregate"
            );
        }
    };
    let map = build_struct_slot_map(ctx, &layout)
        .map_err(|error| pliron::input_error_noloc!("{error}"))?;
    let semantic_field = projection_result_pointee(ctx, operation, "mir.field_addr")?;
    let expected =
        target_stable_local_value_type(ctx, semantic_field, "packed shared local field projection")
            .map_err(|error| pliron::input_error_noloc!("{error}"))?;

    let Some(slot_entry) = map.decl_to_llvm.get(field_index).copied() else {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier field index {} out of bounds for aggregate with {} fields",
            field_index,
            map.decl_to_llvm.len()
        );
    };
    let Some(slot) = slot_entry else {
        // Stripped ZSTs have no carrier slot. Their pointer still participates
        // in the verified address graph, so preserve the exact target-stable
        // result type for any further zero-sized projection.
        return Ok(expected);
    };

    let carrier_ref = carrier_source_ty.deref(ctx);
    let carrier_struct = carrier_ref
        .downcast_ref::<llvm_types::StructType>()
        .ok_or_else(|| {
            pliron::input_error_noloc!(
                "packed-AS3 carrier field projection expected struct storage, got {}",
                carrier_source_ty.deref(ctx).disp(ctx)
            )
        })?;
    let physical = carrier_struct.fields().nth(slot as usize).ok_or_else(|| {
        pliron::input_error_noloc!(
            "packed-AS3 carrier field slot {} out of bounds for storage type {}",
            slot,
            carrier_source_ty.deref(ctx).disp(ctx)
        )
    })?;
    if physical != expected {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier field projection physical type {} disagrees with target-stable semantic field type {}",
            physical.deref(ctx).disp(ctx),
            expected.deref(ctx).disp(ctx)
        );
    }
    Ok(physical)
}

fn carrier_array_element_physical_pointee(
    ctx: &mut Context,
    operation: Ptr<Operation>,
    carrier_array_ty: TypeHandle,
) -> Result<TypeHandle> {
    let physical = {
        let carrier_ref = carrier_array_ty.deref(ctx);
        let carrier_array = carrier_ref
            .downcast_ref::<llvm_types::ArrayType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "packed-AS3 carrier array projection expected array storage, got {}",
                    carrier_array_ty.deref(ctx).disp(ctx)
                )
            })?;
        carrier_array.elem_type()
    };
    let semantic_element = projection_result_pointee(ctx, operation, "mir.array_element_addr")?;
    let expected = target_stable_local_value_type(
        ctx,
        semantic_element,
        "packed shared local array element projection",
    )
    .map_err(|error| pliron::input_error_noloc!("{error}"))?;
    if physical != expected {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier array element type {} disagrees with target-stable semantic element type {}",
            physical.deref(ctx).disp(ctx),
            expected.deref(ctx).disp(ctx)
        );
    }
    Ok(physical)
}

fn derive_carrier_projections(
    ctx: &mut Context,
    operations: &[Ptr<Operation>],
    addresses: &mut FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    // Projections form an SSA use graph. Iterate to a fixed point so nested
    // field/array chains are discovered regardless of block walk order. Every
    // derived child receives its physical pointee directly from the already
    // proven parent carrier type; no converted defining-op history is queried.
    let mut changed = true;
    while changed {
        changed = false;
        for &operation in operations {
            let is_field = Operation::get_op::<MirFieldAddrOp>(operation, ctx).is_some();
            let is_array = Operation::get_op::<MirArrayElementAddrOp>(operation, ctx).is_some();
            if !is_field && !is_array {
                continue;
            }

            let (base, result) = {
                let op = operation.deref(ctx);
                (op.get_operand(0), op.get_result(0))
            };
            if addresses.contains_key(&result) {
                continue;
            }
            let Some(base_state) = addresses.get(&base).copied() else {
                continue;
            };

            let physical_pointee = if is_field {
                carrier_field_physical_pointee(ctx, operation, base_state.physical_pointee)?
            } else {
                carrier_array_element_physical_pointee(ctx, operation, base_state.physical_pointee)?
            };

            // Field GEPs index the physical parent aggregate. Array-element
            // GEPs use the physical element type as their typed source.
            let gep_source_ty = if is_field {
                base_state.physical_pointee
            } else {
                physical_pointee
            };
            plan.plan_gep_source_type(operation, gep_source_ty);
            addresses.insert(result, CarrierAddress { physical_pointee });
            changed = true;
        }
    }
    Ok(())
}

fn reject_boundary_use(kind: &str) -> Result<()> {
    pliron::input_err_noloc!(
        "packed-AS3 carrier-local address cannot cross a {}; the physical storage contract must never be dropped and reconstructed later",
        kind
    )
}

fn validate_and_plan_uses(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    addresses: &FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    for &operation in operations {
        let operands = operation.deref(ctx).operands().collect::<Vec<_>>();
        for (index, operand) in operands.into_iter().enumerate() {
            let Some(state) = addresses.get(&operand).copied() else {
                continue;
            };

            if Operation::get_op::<MirLoadOp>(operation, ctx).is_some() {
                if index != 0 {
                    return pliron::input_err_noloc!(
                        "packed-AS3 carrier-local load used its address in an unexpected operand position"
                    );
                }
                plan.plan_storage_type(operation, state.physical_pointee);
                continue;
            }

            if Operation::get_op::<MirStoreOp>(operation, ctx).is_some() {
                if index != 0 {
                    return pliron::input_err_noloc!(
                        "packed-AS3 carrier-local address cannot itself be stored as a value"
                    );
                }
                plan.plan_storage_type(operation, state.physical_pointee);
                continue;
            }

            if Operation::get_op::<MirFieldAddrOp>(operation, ctx).is_some()
                || Operation::get_op::<MirArrayElementAddrOp>(operation, ctx).is_some()
            {
                if index == 0 {
                    let result = operation.deref(ctx).get_result(0);
                    if addresses.contains_key(&result) {
                        continue;
                    }
                }
                return pliron::input_err_noloc!(
                    "packed-AS3 carrier projection was not proven by the complete address-use graph"
                );
            }

            if Operation::get_op::<MirDbgValueOp>(operation, ctx).is_some()
                || Operation::get_op::<MirDbgValueListOp>(operation, ctx).is_some()
            {
                continue;
            }

            if Operation::get_op::<MirCallOp>(operation, ctx).is_some() {
                return reject_boundary_use("call boundary");
            }
            if Operation::get_op::<MirCastOp>(operation, ctx).is_some() {
                return reject_boundary_use("cast");
            }
            if Operation::get_op::<MirGotoOp>(operation, ctx).is_some()
                || Operation::get_op::<MirCondBranchOp>(operation, ctx).is_some()
                || Operation::get_op::<MirAssertOp>(operation, ctx).is_some()
            {
                return reject_boundary_use("block-argument edge");
            }
            if Operation::get_op::<MirReturnOp>(operation, ctx).is_some() {
                return reject_boundary_use("return boundary");
            }
            if Operation::get_op::<MirPtrOffsetOp>(operation, ctx).is_some() {
                return pliron::input_err_noloc!(
                    "packed-AS3 carrier locals do not support pointer arithmetic"
                );
            }

            return pliron::input_err_noloc!(
                "packed-AS3 carrier-local address has an unsupported use by {}; carrier identity would otherwise be lost",
                Operation::get_opid(operation, ctx)
            );
        }
    }
    Ok(())
}

/// Prove and stamp every use of the packed-AS3 local-storage lane.
///
/// This must run after the final MIR-producing transform and immediately before
/// dialect conversion. Its attributes are lowering-private capabilities: input
/// MIR is rejected if it tries to supply one, and every carrier address escape
/// not explicitly modelled here is rejected before LLVM pointer opacity can
/// erase the distinction between semantic and physical pointees. Planning is
/// transactional: no carrier capability is attached until every use validates.
pub(crate) fn prepare_packed_shared_local_storage(
    ctx: &mut Context,
    root: Ptr<Operation>,
) -> Result<()> {
    let operations = collect_operations(ctx, root);
    reject_preexisting_carrier_facts(ctx, &operations)?;

    let mut addresses = FxHashMap::default();
    let mut plan = CarrierFactPlan::default();
    seed_carrier_allocas(ctx, &operations, &mut addresses, &mut plan)?;
    if addresses.is_empty() {
        return Ok(());
    }

    derive_carrier_projections(ctx, &operations, &mut addresses, &mut plan)?;
    validate_and_plan_uses(ctx, &operations, &addresses, &mut plan)?;
    plan.apply(ctx);
    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer/facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::MirCastKindAttr;
    use dialect_mir::ops as mir;
    use llvm_export::op_interfaces::PointerTypeResult;
    use llvm_export::ops as llvm;
    use llvm_export::types::{PointerType, StructType, address_space as llvm_addr};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FunctionType, Signedness};
    use pliron::op::Op;

    fn packed_shared_fixture(ctx: &mut Context) -> (TypeHandle, TypeHandle, TypeHandle) {
        let tag: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get(
            ctx,
            pointee,
            true,
            dialect_mir::types::address_space::SHARED,
        )
        .into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "PackedShared".into(),
            vec!["tag".into(), "ptr".into()],
            vec![tag, shared],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        (packed, tag, shared)
    }

    fn append_alloca(ctx: &mut Context, block: Ptr<BasicBlock>, pointee: TypeHandle) -> Value {
        let pointer: TypeHandle = MirPtrType::get_generic(ctx, pointee, true).into();
        let op = Operation::new(
            ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![pointer],
            vec![],
            vec![],
            0,
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    #[test]
    fn direct_projection_uses_explicit_carrier_facts() {
        let mut ctx = make_ctx();
        let (packed, _tag, shared) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);

        let undef = mir::MirUndefOp::new(&mut ctx, packed);
        undef.get_operation().insert_at_back(block, &ctx);
        let whole_value = undef.get_operation().deref(&ctx).get_result(0);
        let whole_store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, whole_value],
            vec![],
            0,
        );
        whole_store.insert_at_back(block, &ctx);

        let field_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, shared, true).into();
        let field_addr = mir::MirFieldAddrOp::build(&mut ctx, slot, field_ptr_ty, 1)
            .expect("field address build");
        field_addr.insert_at_back(block, &ctx);
        let field_ptr = field_addr.deref(&ctx).get_result(0);

        let load = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![shared],
            vec![field_ptr],
            vec![],
            0,
        );
        load.insert_at_back(block, &ctx);
        let loaded_shared = load.deref(&ctx).get_result(0);
        let projected_store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![field_ptr, loaded_shared],
            vec![],
            0,
        );
        projected_store.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("carrier lowering failed");

        let body = kernel_blocks(&ctx, module);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected alloca");
        let storage = alloca.result_pointee_type(&ctx);
        let storage_ref = storage.deref(&ctx);
        let storage_struct = storage_ref
            .downcast_ref::<StructType>()
            .expect("carrier local must use struct storage");
        let pointer_ty = storage_struct.field_type(1);
        let pointer_ref = pointer_ty.deref(&ctx);
        let pointer = pointer_ref
            .downcast_ref::<PointerType>()
            .expect("carrier pointer field must lower to LLVM pointer");
        assert_eq!(pointer.address_space(), llvm_addr::GENERIC);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            3,
            "whole-value store, projected load, and projected store must each cross p3/p0 exactly once"
        );
    }

    #[test]
    fn whole_value_load_round_trips_through_carrier_storage() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let copy_slot = append_alloca(&mut ctx, block, packed);
        let undef = mir::MirUndefOp::new(&mut ctx, packed);
        undef.get_operation().insert_at_back(block, &ctx);
        let value = undef.get_operation().deref(&ctx).get_result(0);
        let store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, value],
            vec![],
            0,
        );
        store.insert_at_back(block, &ctx);
        let load = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![packed],
            vec![slot],
            vec![],
            0,
        );
        load.insert_at_back(block, &ctx);
        let loaded = load.deref(&ctx).get_result(0);
        let copy_store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![copy_slot, loaded],
            vec![],
            0,
        );
        copy_store.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("whole-value carrier roundtrip");
        let body = kernel_blocks(&ctx, module);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
        let load = find_first::<llvm::LoadOp>(&ctx, &body).unwrap();
        let storage_ty = alloca.result_pointee_type(&ctx);
        assert_eq!(
            load.get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx),
            storage_ty
        );
        let pointer_ty = storage_ty
            .deref(&ctx)
            .downcast_ref::<StructType>()
            .unwrap()
            .field_type(1);
        assert_eq!(
            pointer_ty
                .deref(&ctx)
                .downcast_ref::<PointerType>()
                .unwrap()
                .address_space(),
            llvm_addr::GENERIC
        );
        let address_space = |value: Value| {
            value
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<PointerType>()
                .unwrap()
                .address_space()
        };
        let conversions: Vec<_> = find_all::<llvm::AddrSpaceCastOp>(&ctx, &body)
            .iter()
            .map(|cast| {
                let op = cast.get_operation().deref(&ctx);
                (
                    address_space(op.get_operand(0)),
                    address_space(op.get_result(0)),
                )
            })
            .collect();
        assert_eq!(
            conversions,
            [
                (llvm_addr::SHARED, llvm_addr::GENERIC),
                (llvm_addr::GENERIC, llvm_addr::SHARED),
                (llvm_addr::SHARED, llvm_addr::GENERIC),
            ]
        );
    }

    #[test]
    fn local_storage_accepts_recursive_multi_leaf_and_bounded_array_shapes() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let word: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();

        let pair: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "LocalSharedPair".into(),
            vec!["left".into(), "right".into()],
            vec![shared, shared],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![shared, word],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, shared, 2).into();

        for (name, field_ty, size) in [
            ("PackedLocalNestedStruct", pair, 17),
            ("PackedLocalNestedTuple", tuple, 17),
            ("PackedLocalArray", array, 17),
        ] {
            let packed: TypeHandle = MirStructType::get_with_full_layout(
                &mut ctx,
                name.into(),
                vec!["tag".into(), "payload".into()],
                vec![tag, field_ty],
                vec![0, 1],
                vec![0, 1],
                size,
                1,
            )
            .into();
            assert!(
                packed_shared_local_storage_info(&mut ctx, packed)
                    .expect("local classification must not error")
                    .is_some(),
                "{name} must be admitted by the recursive local carrier lane"
            );
        }

        let multi: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedLocalMulti".into(),
            vec!["tag".into(), "left".into(), "right".into()],
            vec![tag, shared, shared],
            vec![0, 1, 2],
            vec![0, 1, 9],
            17,
            1,
        )
        .into();
        assert!(
            packed_shared_local_storage_info(&mut ctx, multi)
                .expect("local classification must not error")
                .is_some()
        );
    }

    #[test]
    fn local_storage_reuses_array_rewrite_bound_and_rejects_vectors() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();

        for (count, admitted) in [
            (MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES, true),
            (
                MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES + 1,
                false,
            ),
        ] {
            let array: TypeHandle = MirArrayType::get(&mut ctx, shared, count).into();
            let packed: TypeHandle = MirStructType::get_with_full_layout(
                &mut ctx,
                format!("PackedLocalArray{count}"),
                vec!["tag".into(), "ptrs".into()],
                vec![tag, array],
                vec![0, 1],
                vec![0, 1],
                1 + 8 * count,
                1,
            )
            .into();
            assert_eq!(
                packed_shared_local_storage_info(&mut ctx, packed)
                    .expect("local classification must not error")
                    .is_some(),
                admitted
            );
        }

        let shared_pointer: TypeHandle =
            llvm_types::PointerType::get(&ctx, llvm_types::address_space::SHARED).into();
        let vector: TypeHandle =
            llvm_types::VectorType::get(&ctx, shared_pointer, 2, llvm_types::VectorTypeKind::Fixed)
                .into();
        let packed_vector: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedLocalVector".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, vector],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();
        assert!(
            packed_shared_local_storage_info(&mut ctx, packed_vector)
                .expect("vector classification must not error")
                .is_none(),
            "vectors must remain outside the carrier-local lane"
        );
    }

    #[test]
    fn nested_struct_and_tuple_projections_keep_exact_carrier_types() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let word: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let pair: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "NestedProjectionPair".into(),
            vec!["left".into(), "right".into()],
            vec![shared, shared],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![shared, word],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedNestedProjection".into(),
            vec!["tag".into(), "pair".into(), "tuple".into()],
            vec![tag, pair, tuple],
            vec![0, 1, 2],
            vec![0, 1, 17],
            33,
            1,
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        for (outer_index, aggregate, inner_index) in [(1_u32, pair, 1_u32), (2, tuple, 0)] {
            let aggregate_ptr: TypeHandle =
                MirPtrType::get_generic(&mut ctx, aggregate, true).into();
            let outer = mir::MirFieldAddrOp::build(&mut ctx, slot, aggregate_ptr, outer_index)
                .expect("outer field address build");
            outer.insert_at_back(block, &ctx);
            let aggregate_address = outer.deref(&ctx).get_result(0);
            let shared_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, shared, true).into();
            let inner =
                mir::MirFieldAddrOp::build(&mut ctx, aggregate_address, shared_ptr, inner_index)
                    .expect("nested field address build");
            inner.insert_at_back(block, &ctx);
            let inner_address = inner.deref(&ctx).get_result(0);
            let load = Operation::new(
                &mut ctx,
                mir::MirLoadOp::get_concrete_op_info(),
                vec![shared],
                vec![inner_address],
                vec![],
                0,
            );
            load.insert_at_back(block, &ctx);
        }
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("recursive struct/tuple carrier projections must lower");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body), 2);
    }

    #[test]
    fn array_element_projection_uses_carrier_element_type() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, shared, 2).into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedArrayProjection".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![tag, array],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();
        let index_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let (module, block) = build_kernel(&mut ctx, vec![index_ty], vec![]);
        let index = block.deref(&ctx).get_argument(0);
        let slot = append_alloca(&mut ctx, block, packed);

        let array_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, array, true).into();
        let array_field = mir::MirFieldAddrOp::build(&mut ctx, slot, array_ptr, 1)
            .expect("array field address build");
        array_field.insert_at_back(block, &ctx);
        let array_address = array_field.deref(&ctx).get_result(0);

        let element_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, shared, true).into();
        let element_addr = Operation::new(
            &mut ctx,
            mir::MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_ptr],
            vec![array_address, index],
            vec![],
            0,
        );
        element_addr.insert_at_back(block, &ctx);
        let element_address = element_addr.deref(&ctx).get_result(0);
        let load = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![shared],
            vec![element_address],
            vec![],
            0,
        );
        load.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("carrier-backed array element projection must lower");
        let body = kernel_blocks(&ctx, module);
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), 2, "field plus element projection expected");
        let element_source = geps[1].src_elem_type(&ctx);
        let element_ref = element_source.deref(&ctx);
        let pointer = element_ref
            .downcast_ref::<PointerType>()
            .expect("carrier array element must remain pointer-typed");
        assert_eq!(pointer.address_space(), llvm_addr::GENERIC);
        assert_eq!(count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body), 1);
    }

    #[test]
    fn input_cannot_supply_carrier_storage_facts() {
        for key in [CARRIER_STORAGE_TYPE_KEY, CARRIER_GEP_SOURCE_TYPE_KEY] {
            let mut ctx = make_ctx();
            let (packed, _, _) = packed_shared_fixture(&mut ctx);
            let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
            let slot = append_alloca(&mut ctx, block, packed);
            let alloca = slot.defining_op().unwrap();
            let storage = packed_shared_local_storage_info(&mut ctx, packed)
                .unwrap()
                .unwrap()
                .storage_ty;
            set_type_attr(&mut ctx, alloca, key, storage);
            append_mir_return(&mut ctx, block, vec![]);

            let error = crate::lower_mir_to_llvm(&mut ctx, module)
                .expect_err("input carrier facts must be rejected");
            assert!(
                error.to_string().contains("not supplied by input MIR"),
                "{error}"
            );
            assert!(Operation::get_op::<MirAllocaOp>(alloca, &ctx).is_some());
        }
    }

    #[test]
    fn carrier_address_cannot_be_stored_as_a_value() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let slot_ty = slot.get_type(&ctx);
        let address_slot = append_alloca(&mut ctx, block, slot_ty);
        let store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![address_slot, slot],
            vec![],
            0,
        );
        store.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("carrier addresses cannot escape through memory");
        assert!(
            error
                .to_string()
                .contains("cannot itself be stored as a value"),
            "{error}"
        );
        assert!(carrier_storage_type(&ctx, slot.defining_op().unwrap()).is_none());
    }

    #[test]
    fn carrier_address_allows_nested_zero_sized_projection() {
        let mut ctx = make_ctx();
        let (_, tag, shared) = packed_shared_fixture(&mut ctx);
        let unit: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Unit".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            1,
        )
        .into();
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Marker".into(),
            vec!["unit".into()],
            vec![unit],
            vec![0],
            vec![0],
            0,
            1,
        )
        .into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedWithMarker".into(),
            vec!["marker".into(), "tag".into(), "ptr".into()],
            vec![marker, tag, shared],
            vec![0, 1, 2],
            vec![0, 0, 1],
            9,
            1,
        )
        .into();
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let marker_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, marker, true).into();
        let first = mir::MirFieldAddrOp::build(&mut ctx, slot, marker_ptr, 0).unwrap();
        first.insert_at_back(block, &ctx);
        let marker_address = first.deref(&ctx).get_result(0);
        let unit_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, unit, true).into();
        let nested = mir::MirFieldAddrOp::build(&mut ctx, marker_address, unit_ptr, 0).unwrap();
        nested.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("recursive carrier projections through ZSTs must lower");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            find_all::<llvm::GetElementPtrOp>(&ctx, &body).len(),
            2,
            "both zero-sized projections must keep distinct address values"
        );
    }

    #[test]
    fn carrier_address_rejects_cast_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let pointer_ty = slot.get_type(&ctx);
        let cast = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![pointer_ty],
            vec![slot],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        cast.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let alloca = slot
            .defining_op()
            .expect("alloca result must have a defining op");
        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("carrier address cast must fail closed in the full lowering pipeline");
        assert!(error.to_string().contains("cast"));
        assert!(
            carrier_storage_type(&ctx, alloca).is_none(),
            "a rejected carrier plan must not leave partial lowering capabilities behind"
        );
    }

    #[test]
    fn carrier_address_rejects_block_argument_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let slot_ty = slot.get_type(&ctx);
        let successor = append_block(&mut ctx, block, vec![slot_ty]);
        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![slot],
            vec![successor],
            0,
        );
        goto.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, successor, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "carrier address block argument must fail closed in the full lowering pipeline",
        );
        assert!(error.to_string().contains("block-argument"));
    }

    #[test]
    fn carrier_address_rejects_call_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let slot_ty = slot.get_type(&ctx);
        let call = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![],
            vec![slot],
            vec![],
            0,
        );
        let call_op = mir::MirCallOp::new(call);
        call_op.set_attr_callee(&ctx, StringAttr::new("sink".to_string()));
        let signature = FunctionType::get(&ctx, vec![slot_ty], vec![]);
        call_op.set_external_callee_signature(&mut ctx, signature.into());
        call.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("carrier address call must fail closed in the full lowering pipeline");
        assert!(error.to_string().contains("call boundary"));
    }
}
