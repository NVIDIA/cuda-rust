/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit validated native/storage plans from original typed library operands.

use super::*;
use dialect_mir::{
    attributes::{FieldIndexAttr, MirPointerKindAuthorityAttr},
    library::{LibraryCallPlan, LibraryNativeMetadata, LibraryOperation},
    ops::{
        MirCallOp, MirConstantOp, MirExternSharedOp, MirExtractFieldOp, MirInsertFieldOp, MirMulOp,
        MirPtrOffsetOp, MirUndefOp,
    },
    types::{MirPointerKind, MirPtrType, MirStructType, MirTupleType},
};
use dialect_nvvm::ops::Barrier0Op;
use pliron::{
    builtin::{type_interfaces::FunctionTypeInterface, types::FunctionType},
    r#type::Typed,
    verify_err,
};

const FIXED_SHARED_BYTES: &str = "cuda_oxide_fixed_shared_bytes";
const PLANNED_ROOT: &str = "cuda_oxide_planned_storage_root";

pub(super) fn operation(role: LibraryRoleAttr) -> Option<LibraryOperation> {
    Some(match role {
        LibraryRoleAttr::Native(_) => LibraryOperation::Native,
        LibraryRoleAttr::StorageNew(_) => LibraryOperation::StorageNew,
        LibraryRoleAttr::StorageReconfigure { .. } => LibraryOperation::StorageReconfigure,
        LibraryRoleAttr::StorageReserve { .. } => LibraryOperation::StorageReserve,
        LibraryRoleAttr::StorageRelease => LibraryOperation::StorageRelease,
        _ => return None,
    })
}

pub(super) fn validate(ctx: &Context, op: Ptr<Operation>, plan: &LibraryCallPlan) -> Result<()> {
    let library = MirLibraryOp::new(op);
    let role = *library.get_attr_role(ctx).unwrap();
    if operation(role).as_ref() != Some(&plan.operation)
        || String::from((*library.get_attr_library_callee(ctx).unwrap()).clone()) != plan.marker
    {
        return verify_err!(
            op.deref(ctx).loc(),
            "retained library role/callee does not match its finalized plan"
        );
    }
    if !plan.scratch_alignment.is_power_of_two() {
        return verify_err!(op.deref(ctx).loc(), "invalid library scratch alignment");
    }
    if let LibraryRoleAttr::Native(native) = role {
        use LibraryNativeMetadata::*;
        if matches!(native, Raw { .. } | Callback { .. }) {
            let signature = library
                .get_attr_library_signature(ctx)
                .unwrap()
                .get_type(ctx);
            let args = signature
                .deref(ctx)
                .downcast_ref::<FunctionType>()
                .unwrap()
                .arg_types();
            if args[0]
                .deref(ctx)
                .downcast_ref::<MirTupleType>()
                .unwrap()
                .get_types()
                .len()
                < 2
            {
                return verify_err!(
                    op.deref(ctx).loc(),
                    "native scratch ABI requires at least two arguments"
                );
            }
        }
        let (checked, resource, storage) = match native {
            Raw { .. } | Callback { .. } => (false, false, false),
            CheckedBlock { .. } => (true, false, false),
            CheckedBlockResource { .. } | CheckedWarp { .. } => (true, true, false),
            CheckedBlockStorage { .. } => (true, true, true),
            CheckedBlockPayload { .. } | CheckedWarpPayload { .. } => (true, false, false),
        };
        if (
            plan.checked_block,
            plan.resource_witness,
            plan.storage_witness,
        ) != (checked, resource, storage)
            || plan.payload_only
                != matches!(
                    native,
                    CheckedBlockPayload { .. } | CheckedWarpPayload { .. }
                )
            || plan.callback != matches!(native, Callback { .. })
            || plan.warp_width
                != match native {
                    CheckedWarp { width, .. } | CheckedWarpPayload { width, .. } => Some(width),
                    _ => None,
                }
            || plan.symbol.is_empty()
        {
            return verify_err!(
                op.deref(ctx).loc(),
                "native plan does not match the retained source contract"
            );
        }
        if resource && !storage && plan.warp_width.is_none() && plan.scratch_bytes != 0 {
            return verify_err!(
                op.deref(ctx).loc(),
                "checked resource library calls require scratch-free artifacts"
            );
        }
    }
    Ok(())
}

pub(super) fn lower(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    plan: &LibraryCallPlan,
) -> Option<Value> {
    let role = *MirLibraryOp::new(anchor).get_attr_role(ctx).unwrap();
    match role {
        LibraryRoleAttr::Native(native) => {
            native_call(ctx, anchor, native, plan);
            None
        }
        LibraryRoleAttr::StorageRelease => None,
        _ => Some(storage(ctx, anchor, plan)),
    }
}

fn integer(ctx: &mut Context, anchor: Ptr<Operation>, width: u32, value: u64) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Unsigned);
    let op = emit::<MirConstantOp>(ctx, anchor, vec![ty.into()], vec![]);
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(value, NonZeroUsize::new(width as usize).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

/// Lower the genuine pure query only after both proofs have consumed its scope.
pub(super) fn lower_warp_query(ctx: &mut Context, anchor: Ptr<Operation>, width: u32) -> Value {
    use dialect_mir::ops::{MirAddOp, MirDivOp, MirMulOp};
    use dialect_nvvm::ops::{
        ReadPtxSregNtidXOp, ReadPtxSregNtidYOp, ReadPtxSregTidXOp, ReadPtxSregTidYOp,
        ReadPtxSregTidZOp,
    };
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    macro_rules! register {
        ($op:ty, $marker:literal) => {{
            let op = emit::<$op>(ctx, anchor, vec![ty], vec![]);
            op.deref_mut(ctx).attributes.set(
                "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                StringAttr::new($marker.into()),
            );
            op.deref(ctx).get_result(0)
        }};
    }
    macro_rules! arithmetic {
        ($op:ty, $args:expr) => {
            emit::<$op>(ctx, anchor, vec![ty], $args)
                .deref(ctx)
                .get_result(0)
        };
    }
    let x = register!(ReadPtxSregTidXOp, "v1:i0001");
    let y = register!(ReadPtxSregTidYOp, "v1:i0005");
    let z = register!(ReadPtxSregTidZOp, "v1:i0009");
    let nx = register!(ReadPtxSregNtidXOp, "v1:i0003");
    let ny = register!(ReadPtxSregNtidYOp, "v1:i0007");
    let yz = arithmetic!(MirMulOp, vec![ny, z]);
    let yz = arithmetic!(MirAddOp, vec![y, yz]);
    let rank = arithmetic!(MirMulOp, vec![nx, yz]);
    let rank = arithmetic!(MirAddOp, vec![x, rank]);
    let divisor = integer(ctx, anchor, 32, width.into());
    arithmetic!(MirDivOp, vec![rank, divisor])
}

/// Emit the canonical aligned tile mask after participation and completion proof.
pub(super) fn lower_warp_sync(ctx: &mut Context, anchor: Ptr<Operation>, width: u32) {
    use dialect_mir::ops::{MirBitAndOp, MirShlOp};
    use dialect_nvvm::ops::{BarWarpSyncOp, ReadPtxSregLaneIdOp};
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let mask = if width == 32 {
        integer(ctx, anchor, 32, u32::MAX.into())
    } else {
        let lane = emit::<ReadPtxSregLaneIdOp>(ctx, anchor, vec![ty], vec![]);
        lane.deref_mut(ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new("v1:i0016".into()),
        );
        let lane = lane.deref(ctx).get_result(0);
        let alignment = integer(ctx, anchor, 32, (32 - width).into());
        let base = emit::<MirBitAndOp>(ctx, anchor, vec![ty], vec![lane, alignment])
            .deref(ctx)
            .get_result(0);
        let bits = integer(ctx, anchor, 32, ((1u32 << width) - 1).into());
        emit::<MirShlOp>(ctx, anchor, vec![ty], vec![bits, base])
            .deref(ctx)
            .get_result(0)
    };
    let barrier = emit::<BarWarpSyncOp>(ctx, anchor, vec![], vec![mask]);
    barrier.deref_mut(ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0049".into()),
    );
}

fn pointer_cast(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    value: Value,
    target: TypeHandle,
    authority: MirPointerKindAuthorityAttr,
) -> Value {
    if value.get_type(ctx) == target {
        return value;
    }
    let op = emit::<MirCastOp>(ctx, anchor, vec![target], vec![value]);
    let cast = MirCastOp::new(op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, authority);
    op.deref(ctx).get_result(0)
}

/// The unsafe native ABI grants only its raw scratch pointer, never a reference.
#[allow(clippy::disallowed_methods)]
fn scratch_pointer(ctx: &mut Context, pointee: TypeHandle, shared: bool) -> TypeHandle {
    if shared {
        MirPtrType::get_shared_with_kind(ctx, pointee, true, MirPointerKind::RawMut).into()
    } else {
        MirPtrType::get_generic_with_kind(ctx, pointee, true, MirPointerKind::RawMut).into()
    }
}

#[allow(clippy::disallowed_methods)]
fn environment_pointer(ctx: &mut Context, pointee: TypeHandle) -> TypeHandle {
    MirPtrType::get_generic_with_kind(ctx, pointee, false, MirPointerKind::RawConst).into()
}

fn shared_view(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    pointee: TypeHandle,
    plan: &LibraryCallPlan,
) -> Value {
    let ty = MirPtrType::get_shared(ctx, pointee, true).into();
    let op = emit::<MirExternSharedOp>(ctx, anchor, vec![ty], vec![]);
    let shared = MirExternSharedOp::new(op);
    shared.set_byte_offset_value(ctx, u64::from(plan.scratch_offset));
    shared.set_alignment_value(ctx, u64::from(plan.scratch_alignment));
    let layout_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    op.deref_mut(ctx).attributes.set(
        FIXED_SHARED_BYTES.try_into().unwrap(),
        IntegerAttr::new(
            layout_ty,
            APInt::from_u64(
                u64::from(plan.scratch_bytes),
                NonZeroUsize::new(64).unwrap(),
            ),
        ),
    );
    if let Some(root) = &plan.planned_root {
        op.deref_mut(ctx).attributes.set(
            PLANNED_ROOT.try_into().unwrap(),
            StringAttr::new(root.clone()),
        );
    }
    op.deref(ctx).get_result(0)
}

fn block_barrier(ctx: &mut Context, anchor: Ptr<Operation>) {
    let barrier = emit::<Barrier0Op>(ctx, anchor, vec![], vec![]);
    barrier.deref_mut(ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0034".into()),
    );
}

fn flatten(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> Vec<Value> {
    let fields = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirTupleType>()
        .unwrap()
        .get_types()
        .to_vec();
    fields
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            let op = emit::<MirExtractFieldOp>(ctx, anchor, vec![ty], vec![value]);
            MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
            op.deref(ctx).get_result(0)
        })
        .collect()
}

pub(super) fn lower_boundary(ctx: &mut Context, anchor: Ptr<Operation>, plan: &LibraryCallPlan) {
    if plan.barrier_before {
        if let Some(width) = plan.warp_width {
            lower_warp_sync(ctx, anchor, width);
        } else {
            block_barrier(ctx, anchor);
        }
    }
}

fn native_call(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    native: LibraryNativeMetadata,
    plan: &LibraryCallPlan,
) {
    use LibraryNativeMetadata::*;
    let library = MirLibraryOp::new(anchor);
    let signature = library
        .get_attr_library_signature(ctx)
        .unwrap()
        .get_type(ctx);
    let types = signature
        .deref(ctx)
        .downcast_ref::<FunctionType>()
        .unwrap()
        .arg_types();
    let operands = anchor.deref(ctx).operands().collect::<Vec<_>>();
    let indices: &[usize] = match native {
        Raw { .. } | Callback { .. } => &[0],
        CheckedBlock { .. } => &[0, 1],
        CheckedBlockResource { .. } | CheckedWarp { .. } => &[1, 2],
        CheckedBlockStorage { .. } => &[2, 3],
        CheckedBlockPayload { .. } => &[2, 1],
        CheckedWarpPayload { .. } => &[1, 0],
    };
    let mut arguments = Vec::new();
    let mut declared = Vec::new();
    let mut uniform_len = 0;
    for &index in indices {
        let fields = types[index]
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .unwrap()
            .get_types()
            .to_vec();
        if uniform_len == 0 {
            uniform_len = fields.len();
        }
        declared.extend(fields);
        arguments.extend(flatten(ctx, anchor, operands[index]));
    }
    if matches!(
        native,
        CheckedBlock { .. }
            | CheckedBlockResource { .. }
            | CheckedBlockStorage { .. }
            | CheckedWarp { .. }
    ) {
        arguments[1..].rotate_left(uniform_len - 1);
        declared[1..].rotate_left(uniform_len - 1);
    }
    let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let mut scratch = shared_view(ctx, anchor, byte, plan);
    if plan.scratch_stride != 0 {
        let width = plan
            .warp_width
            .expect("validated replicated library scratch");
        let group = lower_warp_query(ctx, anchor, width);
        let ty = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let cast = emit::<MirCastOp>(ctx, anchor, vec![ty], vec![group]);
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        let group = cast.deref(ctx).get_result(0);
        let stride = integer(ctx, anchor, 64, u64::from(plan.scratch_stride));
        let offset = emit::<MirMulOp>(ctx, anchor, vec![ty], vec![group, stride])
            .deref(ctx)
            .get_result(0);
        let pointer_type = scratch.get_type(ctx);
        let pointer =
            emit::<MirPtrOffsetOp>(ctx, anchor, vec![pointer_type], vec![scratch, offset]);
        MirPtrOffsetOp::new(pointer).set_inbounds(ctx, false);
        scratch = pointer.deref(ctx).get_result(0);
    }
    let raw_shared = scratch_pointer(ctx, byte, true);
    scratch = pointer_cast(
        ctx,
        anchor,
        scratch,
        raw_shared,
        MirPointerKindAuthorityAttr::RawAddress,
    );
    arguments.insert(2, scratch);
    declared.insert(2, scratch_pointer(ctx, byte, false));
    if matches!(native, Callback { .. }) {
        let environment = operands[1];
        let pointee = environment
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .unwrap()
            .pointee;
        let typed_raw = environment_pointer(ctx, pointee);
        let environment = pointer_cast(
            ctx,
            anchor,
            environment,
            typed_raw,
            MirPointerKindAuthorityAttr::RawAddress,
        );
        let opaque = environment_pointer(ctx, byte);
        // Pointee erasure belongs to this explicit unsafe callback ABI.
        let cast = emit::<MirCastOp>(ctx, anchor, vec![opaque], vec![environment]);
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        arguments.push(cast.deref(ctx).get_result(0));
        declared.push(opaque);
    }
    for (argument, &expected) in arguments.iter_mut().zip(&declared) {
        let actual = argument.get_type(ctx);
        let address_only = {
            let actual = actual.deref(ctx);
            let expected = expected.deref(ctx);
            match (
                actual.downcast_ref::<MirPtrType>(),
                expected.downcast_ref::<MirPtrType>(),
            ) {
                (Some(actual), Some(expected)) => {
                    actual.pointee == expected.pointee
                        && actual.is_mutable == expected.is_mutable
                        && actual.kind == expected.kind
                        && actual.address_space != expected.address_space
                }
                _ => false,
            }
        };
        if address_only {
            *argument = pointer_cast(
                ctx,
                anchor,
                *argument,
                expected,
                MirPointerKindAuthorityAttr::AbiBoundary,
            );
        }
    }
    let signature = FunctionType::get(ctx, declared, vec![]).into();
    let unit = MirTupleType::get(ctx, vec![]).into();
    let call = emit::<MirCallOp>(ctx, anchor, vec![unit], arguments);
    let call = MirCallOp::new(call);
    call.set_attr_callee(ctx, StringAttr::new(plan.symbol.clone()));
    call.set_external_callee_signature(ctx, signature);
    if plan.barrier_after {
        debug_assert!(plan.warp_width.is_none());
        block_barrier(ctx, anchor);
    }
}

fn storage(ctx: &mut Context, anchor: Ptr<Operation>, plan: &LibraryCallPlan) -> Value {
    let ty = anchor.deref(ctx).get_result(0).get_type(ctx);
    let field = ty
        .deref(ctx)
        .downcast_ref::<MirStructType>()
        .unwrap()
        .get_field_type(0)
        .unwrap();
    let value = if plan.operation == LibraryOperation::StorageReserve {
        let pointee = field
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .unwrap()
            .pointee;
        let value = shared_view(ctx, anchor, pointee, plan);
        pointer_cast(
            ctx,
            anchor,
            value,
            field,
            MirPointerKindAuthorityAttr::RawAddress,
        )
    } else {
        integer(ctx, anchor, 8, 0)
    };
    let undef = emit::<MirUndefOp>(ctx, anchor, vec![ty], vec![])
        .deref(ctx)
        .get_result(0);
    let result = emit::<MirInsertFieldOp>(ctx, anchor, vec![ty], vec![undef, value]);
    MirInsertFieldOp::new(result).set_attr_insert_index(ctx, FieldIndexAttr(0));
    result.deref(ctx).get_result(0)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use dialect_mir::{
        library::{
            LibraryAccess, LibraryCallSite, LibraryReserveLayout, LibraryScalar, LibraryStorageArg,
            LibraryStoragePolicy, LibraryWitness, NativeSpec,
        },
        ops::MirReturnOp,
        types::MirSliceType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{op_interfaces::SingleBlockRegionInterface, ops::ModuleOp},
        linked_list::ContainsLinkedList,
    };

    fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
        let op = MirUndefOp::new(ctx, ty).get_operation();
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    fn root(ctx: &mut Context) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
        let module = ModuleOp::new(ctx, "planned".try_into().unwrap());
        let block = module.get_body(ctx, 0);
        (module.get_operation(), block)
    }

    fn origin(ctx: &Context, mut value: Value) -> (Value, usize) {
        loop {
            let op = value.defining_op().unwrap();
            if Operation::get_op::<MirCastOp>(op, ctx).is_some() {
                value = op.deref(ctx).get_operand(0);
            } else {
                let extract = MirExtractFieldOp::new(op);
                return (
                    op.deref(ctx).get_operand(0),
                    extract.get_attr_index(ctx).unwrap().0 as usize,
                );
            }
        }
    }

    #[test]
    fn tile_sync_lowers_without_a_plan_to_the_exact_aligned_lane_mask() {
        use dialect_mir::ops::{MirBitAndOp, MirShlOp};
        use dialect_nvvm::ops::{BarWarpSyncOp, ReadPtxSregLaneIdOp};
        use pliron::location::Location;

        fn mask(ctx: &Context, value: Value, lane: u32) -> u32 {
            let op = value.defining_op().unwrap();
            if let Some(constant) = Operation::get_op::<MirConstantOp>(op, ctx) {
                constant.get_attr_value(ctx).unwrap().value().to_u64() as u32
            } else if Operation::get_op::<ReadPtxSregLaneIdOp>(op, ctx).is_some() {
                lane
            } else {
                let lhs = mask(ctx, op.deref(ctx).get_operand(0), lane);
                let rhs = mask(ctx, op.deref(ctx).get_operand(1), lane);
                if Operation::get_op::<MirBitAndOp>(op, ctx).is_some() {
                    lhs & rhs
                } else {
                    assert!(Operation::get_op::<MirShlOp>(op, ctx).is_some());
                    lhs << rhs
                }
            }
        }
        for width in [1, 2, 4, 8, 16, 32] {
            let ctx = &mut Context::new();
            let (module, block) = root(ctx);
            let site = LibraryCallSite {
                caller: "sync_root".into(),
                block: 0,
            };
            let sync = MirLibraryOp::build(
                ctx,
                LibraryRoleAttr::SyncWarp { width },
                site,
                "sync".into(),
            );
            let loc = Location::Named {
                name: "source_sync".into(),
                child_loc: Box::new(Location::Unknown),
            };
            sync.deref_mut(ctx).set_loc(loc.clone());
            sync.insert_at_back(block, ctx);
            super::super::lower_library_operations(module, ctx, &mut AnalysisManager::default())
                .unwrap();
            module.deref(ctx).verify(ctx).unwrap();
            let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
            assert!(
                !operations
                    .iter()
                    .any(|&op| Operation::get_op::<MirLibraryOp>(op, ctx).is_some())
            );
            assert!(operations.iter().all(|&op| op.deref(ctx).loc() == &loc));
            let barriers: Vec<_> = operations
                .iter()
                .copied()
                .filter(|&op| Operation::get_op::<BarWarpSyncOp>(op, ctx).is_some())
                .collect();
            assert_eq!(barriers.len(), 1);
            let operand = barriers[0].deref(ctx).get_operand(0);
            for lane in 0..32 {
                let expected = (0..32)
                    .filter(|&member| member / width == lane / width)
                    .fold(0, |bits, member| bits | (1u32 << member));
                assert_eq!(
                    mask(ctx, operand, lane),
                    expected,
                    "width {width}, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn native_roles_preserve_argument_order_and_foreign_pointer_authority() {
        use LibraryNativeMetadata::*;
        let spec = NativeSpec {
            scalar: LibraryScalar::Signed(32),
            items: 4,
        };
        let access = LibraryAccess::Read;
        let witness = LibraryWitness::ReadOnlySlice;
        let storage = LibraryStorageArg::Implicit;
        let roles = [
            Raw {
                spec,
                block_threads: 64,
            },
            Callback {
                spec,
                block_threads: 64,
            },
            CheckedBlock { spec },
            CheckedBlockResource {
                spec,
                access,
                witness,
            },
            CheckedWarp {
                spec,
                width: 8,
                access,
                witness,
            },
            CheckedBlockStorage {
                spec,
                access,
                witness,
                storage,
            },
            CheckedBlockPayload { spec, storage },
            CheckedWarpPayload { spec, width: 8 },
        ];
        for native in roles {
            let ctx = &mut Context::new();
            let (module, block) = root(ctx);
            let scalar = IntegerType::get(ctx, 32, Signedness::Signed).into();
            let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let input =
                MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::RawConst)
                    .into();
            let output =
                MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
            let flags =
                MirPtrType::get_generic_with_kind(ctx, byte, true, MirPointerKind::RawMut).into();
            let shared_output =
                MirPtrType::get_shared_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
            let uniform = MirTupleType::get(ctx, vec![input, scalar]).into();
            let payload = MirTupleType::get(ctx, vec![output, flags]).into();
            let actual_payload = MirTupleType::get(ctx, vec![shared_output, flags]).into();
            let raw = MirTupleType::get(ctx, vec![input, output, scalar]).into();
            let wide = IntegerType::get(ctx, 64, Signedness::Signed).into();
            let local_output =
                MirPtrType::get_generic_with_kind(ctx, wide, true, MirPointerKind::RawMut).into();
            let local = MirTupleType::get(ctx, vec![input, local_output, flags]).into();
            let controls = MirTupleType::get(ctx, vec![scalar]).into();
            let empty = MirTupleType::get(ctx, vec![]).into();
            let borrowed =
                MirSliceType::get_with_kind(ctx, scalar, MirPointerKind::SharedRef).into();
            let environment = MirTupleType::get(ctx, vec![scalar]).into();
            let environment = MirPtrType::get_generic_with_kind(
                ctx,
                environment,
                false,
                MirPointerKind::SharedRef,
            )
            .into();
            let types = match native {
                Raw { .. } => vec![raw],
                Callback { .. } => vec![raw, environment],
                CheckedBlock { .. } => vec![uniform, payload],
                CheckedBlockResource { .. } | CheckedWarp { .. } => {
                    vec![borrowed, uniform, payload]
                }
                CheckedBlockStorage { .. } => vec![borrowed, empty, uniform, payload],
                CheckedBlockPayload { .. } => vec![empty, empty, local],
                CheckedWarpPayload { .. } => vec![controls, local],
            };
            let site = LibraryCallSite {
                caller: "root".into(),
                block: 7,
            };
            let boundary = MirLibraryOp::build(
                ctx,
                LibraryRoleAttr::NativeOperandBoundary,
                site.clone(),
                "marker".into(),
            );
            boundary.insert_at_back(block, ctx);
            let mut operand_read = None;
            let operands: Vec<_> = types
                .iter()
                .map(|&ty| {
                    if ty == payload {
                        let address_type = MirPtrType::get_generic_with_kind(
                            ctx,
                            actual_payload,
                            false,
                            MirPointerKind::RawConst,
                        )
                        .into();
                        let address = undef(ctx, block, address_type);
                        let read = Operation::new(
                            ctx,
                            dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
                            vec![actual_payload],
                            vec![address],
                            vec![],
                            0,
                        );
                        read.insert_at_back(block, ctx);
                        operand_read = Some(read);
                        read.deref(ctx).get_result(0)
                    } else {
                        undef(ctx, block, ty)
                    }
                })
                .collect();
            let signature = FunctionType::get(ctx, types, vec![]);
            let op = MirLibraryOp::build_call(
                ctx,
                LibraryRoleAttr::Native(native),
                site.clone(),
                "marker".into(),
                signature,
                operands.clone(),
            );
            op.insert_at_back(block, ctx);
            let mut plan = LibraryCallPlan::internal(LibraryOperation::Native, "marker".into());
            plan.symbol = "native_body".into();
            plan.checked_block = !matches!(native, Raw { .. } | Callback { .. });
            plan.callback = matches!(native, Callback { .. });
            plan.resource_witness = matches!(
                native,
                CheckedBlockResource { .. } | CheckedBlockStorage { .. } | CheckedWarp { .. }
            );
            plan.storage_witness = matches!(native, CheckedBlockStorage { .. });
            plan.payload_only = matches!(
                native,
                CheckedBlockPayload { .. } | CheckedWarpPayload { .. }
            );
            plan.warp_width = match native {
                CheckedWarp { width, .. } | CheckedWarpPayload { width, .. } => Some(width),
                _ => None,
            };
            plan.scratch_bytes =
                if matches!(native, CheckedBlockResource { .. } | CheckedWarp { .. }) {
                    0
                } else {
                    32
                };
            plan.scratch_alignment = 4;
            if matches!(native, CheckedWarp { .. } | CheckedWarpPayload { .. }) {
                plan.scratch_bytes = 1184;
                plan.scratch_stride = 148;
                plan.scratch_offset = 16;
                plan.planned_root = Some("root".into());
            }
            plan.barrier_before = true;
            let mut wrong_kind = plan.clone();
            wrong_kind.callback = !plan.callback;
            assert!(validate(ctx, op, &wrong_kind).is_err());
            super::super::lower_planned_library_operations(
                module,
                ctx,
                &mut AnalysisManager::default(),
                &BTreeMap::from([(site, plan)]),
            )
            .unwrap();
            let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
            assert!(
                operations
                    .iter()
                    .all(|&op| Operation::get_op::<MirLibraryOp>(op, ctx).is_none())
            );
            if let Some(read) = operand_read {
                let barrier = operations
                    .iter()
                    .position(|&op| {
                        if matches!(native, CheckedWarp { .. } | CheckedWarpPayload { .. }) {
                            Operation::get_op::<dialect_nvvm::ops::BarWarpSyncOp>(op, ctx).is_some()
                        } else {
                            Operation::get_op::<Barrier0Op>(op, ctx).is_some()
                        }
                    })
                    .unwrap();
                assert!(barrier < operations.iter().position(|&op| op == read).unwrap());
            }
            let call = operations
                .iter()
                .find_map(|&op| Operation::get_op::<MirCallOp>(op, ctx))
                .unwrap();
            call.get_operation().deref(ctx).verify(ctx).unwrap();
            let call_op = call.get_operation();
            let expected = match native {
                Raw { .. } | Callback { .. } => {
                    vec![(operands[0], 0), (operands[0], 1), (operands[0], 2)]
                }
                CheckedBlock { .. } => vec![
                    (operands[0], 0),
                    (operands[1], 0),
                    (operands[1], 1),
                    (operands[0], 1),
                ],
                CheckedBlockResource { .. } | CheckedWarp { .. } => vec![
                    (operands[1], 0),
                    (operands[2], 0),
                    (operands[2], 1),
                    (operands[1], 1),
                ],
                CheckedBlockStorage { .. } => vec![
                    (operands[2], 0),
                    (operands[3], 0),
                    (operands[3], 1),
                    (operands[2], 1),
                ],
                CheckedBlockPayload { .. } => {
                    vec![(operands[2], 0), (operands[2], 1), (operands[2], 2)]
                }
                CheckedWarpPayload { .. } => vec![
                    (operands[1], 0),
                    (operands[1], 1),
                    (operands[1], 2),
                    (operands[0], 0),
                ],
            };
            for (index, expected) in expected.into_iter().enumerate() {
                let index = index + usize::from(index >= 2);
                assert_eq!(
                    origin(ctx, call_op.deref(ctx).get_operand(index)),
                    expected,
                    "{native:?}"
                );
            }
            if matches!(native, Callback { .. }) {
                assert_eq!(call_op.deref(ctx).get_num_operands(), 5);
                let environment = call_op.deref(ctx).get_operand(4);
                let ty = environment.get_type(ctx);
                let ty = ty.deref(ctx);
                let pointer = ty.downcast_ref::<MirPtrType>().unwrap();
                assert_eq!(
                    (pointer.pointee, pointer.kind, pointer.address_space),
                    (byte, MirPointerKind::RawConst, 0)
                );
                assert!(!pointer.is_mutable);
                let erase = environment.defining_op().unwrap();
                erase.deref(ctx).verify(ctx).unwrap();
                let raw = erase.deref(ctx).get_operand(0).defining_op().unwrap();
                raw.deref(ctx).verify(ctx).unwrap();
                assert_eq!(raw.deref(ctx).get_operand(0), operands[1]);
                assert_eq!(
                    *MirCastOp::new(raw)
                        .get_attr_pointer_kind_authority(ctx)
                        .unwrap(),
                    MirPointerKindAuthorityAttr::RawAddress
                );
            }
            if matches!(native, CheckedWarp { .. } | CheckedWarpPayload { .. }) {
                let offset = operations
                    .iter()
                    .find_map(|&op| Operation::get_op::<MirPtrOffsetOp>(op, ctx))
                    .unwrap();
                let product = offset
                    .get_operation()
                    .deref(ctx)
                    .get_operand(1)
                    .defining_op()
                    .unwrap();
                assert!(Operation::get_op::<MirMulOp>(product, ctx).is_some());
                let ty = product.deref(ctx).get_result(0).get_type(ctx);
                assert_eq!(
                    ty.deref(ctx).downcast_ref::<IntegerType>().unwrap().width(),
                    64
                );
                let group = product.deref(ctx).get_operand(0).defining_op().unwrap();
                assert!(Operation::get_op::<MirCastOp>(group, ctx).is_some());
                let view = operations
                    .iter()
                    .find_map(|&op| Operation::get_op::<MirExternSharedOp>(op, ctx))
                    .unwrap();
                assert_eq!(
                    view.get_operation()
                        .deref(ctx)
                        .attributes
                        .get::<IntegerAttr>(&FIXED_SHARED_BYTES.try_into().unwrap())
                        .unwrap()
                        .value()
                        .to_u64(),
                    1184
                );
                assert_eq!(view.get_byte_offset_value(ctx), 16);
                assert_eq!(
                    operations
                        .iter()
                        .filter(
                            |&&op| Operation::get_op::<dialect_nvvm::ops::BarWarpSyncOp>(op, ctx)
                                .is_some()
                        )
                        .count(),
                    1
                );
                assert!(
                    operations
                        .iter()
                        .all(|&op| Operation::get_op::<Barrier0Op>(op, ctx).is_none())
                );
            }
            let scratch = call_op.deref(ctx).get_operand(2);
            let cast = MirCastOp::new(scratch.defining_op().unwrap());
            assert_eq!(
                cast.get_attr_pointer_kind_authority(ctx)
                    .map(|a| (*a).clone()),
                Some(MirPointerKindAuthorityAttr::AbiBoundary)
            );
            let scratch_type = scratch.get_type(ctx);
            let pointer = scratch_type.deref(ctx);
            let pointer = pointer.downcast_ref::<MirPtrType>().unwrap();
            assert_eq!(
                (pointer.address_space, pointer.kind),
                (0, MirPointerKind::RawMut)
            );
        }
    }

    #[test]
    fn reservation_preserves_typed_result_and_release_adds_no_barrier() {
        let ctx = &mut Context::new();
        let (module, block) = root(ctx);
        let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let scalar = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let descriptor =
            MirStructType::get(ctx, "descriptor".into(), vec!["private".into()], vec![byte]).into();
        let borrow =
            MirPtrType::get_generic_with_kind(ctx, descriptor, true, MirPointerKind::UniqueRef)
                .into();
        let field =
            MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
        let result = MirStructType::get(
            ctx,
            "reservation".into(),
            vec!["private".into()],
            vec![field],
        )
        .into();
        let borrowed = undef(ctx, block, borrow);
        let policy = LibraryStoragePolicy {
            capacity: 32,
            alignment: 4,
            auto_sync: false,
            exclusive: true,
        };
        let role = LibraryRoleAttr::StorageReserve {
            policy,
            layout: LibraryReserveLayout {
                scalar: LibraryScalar::Signed(32),
                count: 4,
                bytes: 16,
                alignment: 4,
            },
        };
        let signature = FunctionType::get(ctx, vec![borrow], vec![result]);
        let site = LibraryCallSite {
            caller: "root".into(),
            block: 1,
        };
        let reserve = MirLibraryOp::build_call(
            ctx,
            role,
            site.clone(),
            "reserve".into(),
            signature,
            vec![borrowed],
        );
        reserve.insert_at_back(block, ctx);
        let token = reserve.deref(ctx).get_result(0);
        let signature = FunctionType::get(ctx, vec![result], vec![]);
        let release_site = LibraryCallSite {
            caller: "root".into(),
            block: 2,
        };
        let release = MirLibraryOp::build_call(
            ctx,
            LibraryRoleAttr::StorageRelease,
            release_site.clone(),
            "release".into(),
            signature,
            vec![token],
        );
        release.insert_at_back(block, ctx);
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![token],
            vec![],
            0,
        );
        ret.insert_at_back(block, ctx);
        let mut plan =
            LibraryCallPlan::internal(LibraryOperation::StorageReserve, "reserve".into());
        plan.scratch_offset = 12;
        plan.scratch_bytes = 16;
        plan.scratch_alignment = 4;
        plan.planned_root = Some("root".into());
        let release_plan =
            LibraryCallPlan::internal(LibraryOperation::StorageRelease, "release".into());
        super::super::lower_planned_library_operations(
            module,
            ctx,
            &mut AnalysisManager::default(),
            &BTreeMap::from([(site, plan), (release_site, release_plan)]),
        )
        .unwrap();
        let result_value = ret.deref(ctx).get_operand(0);
        assert_eq!(result_value.get_type(ctx), result);
        let insert = result_value.defining_op().unwrap();
        assert!(Operation::get_op::<MirInsertFieldOp>(insert, ctx).is_some());
        let raw = insert.deref(ctx).get_operand(1);
        assert_eq!(raw.get_type(ctx), field);
        assert_eq!(
            MirCastOp::new(raw.defining_op().unwrap())
                .get_attr_pointer_kind_authority(ctx)
                .map(|a| (*a).clone()),
            Some(MirPointerKindAuthorityAttr::RawAddress)
        );
        let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let shared = operations
            .iter()
            .find_map(|&op| Operation::get_op::<MirExternSharedOp>(op, ctx))
            .unwrap();
        assert_eq!(shared.get_byte_offset_value(ctx), 12);
        assert!(
            operations
                .iter()
                .all(|&op| Operation::get_op::<Barrier0Op>(op, ctx).is_none()
                    && Operation::get_op::<MirLibraryOp>(op, ctx).is_none())
        );
    }
}
