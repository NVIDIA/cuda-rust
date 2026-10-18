/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR aggregates, pointers, calls, and memory in standard MLIR dialects.
//!
//! MIR pointers become opaque LLVM pointers. Rust slices keep their two-word
//! ABI as an LLVM struct, and fixed arrays and structs keep their physical
//! field order. Operations then use LLVM's aggregate and memory operations.

use crate::{
    DropAttribute, MlirAttribute, MlirFloatType, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};
use dialect_mir::{
    attributes::{CompilerResultBundleAttr, FieldIndexAttr, MutabilityAttr},
    ops::{
        MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirCallOp, MirCastOp, MirConstructArrayOp,
        MirConstructSliceOp, MirConstructStructOp, MirConstructTupleOp, MirExtractArrayElementOp,
        MirExtractFieldOp, MirFieldAddrOp, MirLoadOp, MirPtrOffsetOp, MirRefOp, MirStorageDeadOp,
        MirStorageLiveOp, MirStoreOp,
    },
    types::{
        MirArrayType, MirDisjointSliceType, MirEnumType, MirFP16Type, MirPtrType, MirSliceType,
        MirStructType, MirTupleType, MirUnionType,
    },
};
use llvm_export::ops::LocalMemoryProvenanceAttr;
use pliron::{
    builtin::types::{FP16Type, FP32Type, FP64Type, IntegerType, UnitType},
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Register reviewed pointer, physical aggregate, call, and memory mappings.
pub(crate) fn register_mir_memory_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_type::<MirPtrType>(PointerTypeTranslation)?;
    registry.register_type::<MirSliceType>(SliceTypeTranslation)?;
    registry.register_type::<MirDisjointSliceType>(DisjointSliceTypeTranslation)?;
    registry.register_type::<MirArrayType>(ArrayTypeTranslation)?;
    registry.register_type::<MirStructType>(StructTypeTranslation)?;
    registry.register_type::<MirTupleType>(TupleTypeTranslation)?;
    super::storage::register(registry)?;

    registry.register_attribute::<FieldIndexAttr>(DropAttribute)?;
    registry.register_attribute::<MutabilityAttr>(DropAttribute)?;
    registry.register_attribute::<LocalMemoryProvenanceAttr>(DropAttribute)?;
    // This marks an importer-created ABI aggregate for optional forwarding.
    // If it survives preparation, preserve the aggregate normally.
    registry.register_attribute::<CompilerResultBundleAttr>(DropAttribute)?;

    registry.register_operation::<MirAllocaOp>(AllocaTranslation)?;
    registry.register_operation::<MirRefOp>(ReferenceTranslation)?;
    registry.register_operation::<MirLoadOp>(LoadTranslation)?;
    registry.register_operation::<MirStoreOp>(StoreTranslation)?;
    registry.register_operation::<MirPtrOffsetOp>(PointerOffsetTranslation)?;
    registry.register_operation::<MirExtractFieldOp>(ExtractFieldTranslation)?;
    registry.register_operation::<MirConstructStructOp>(ConstructStructTranslation)?;
    registry.register_operation::<MirConstructTupleOp>(ConstructStructTranslation)?;
    registry.register_operation::<MirConstructArrayOp>(ConstructArrayTranslation)?;
    registry.register_operation::<MirConstructSliceOp>(ConstructSliceTranslation)?;
    registry.register_operation::<MirFieldAddrOp>(FieldAddressTranslation)?;
    registry.register_operation::<MirArrayElementAddrOp>(ArrayElementAddressTranslation)?;
    registry.register_operation::<MirExtractArrayElementOp>(ExtractArrayElementTranslation)?;
    registry.register_operation::<MirCallOp>(CallTranslation)?;
    registry.register_operation::<MirAssertOp>(AssertTranslation)?;
    registry.register_operation::<MirStorageLiveOp>(DropOperation)?;
    registry.register_operation::<MirStorageDeadOp>(DropOperation)?;
    Ok(())
}

/// Erase a MIR pointee while preserving its LLVM address space.
struct PointerTypeTranslation;

impl TypeTranslation for PointerTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let pointer = source_ref
            .downcast_ref::<MirPtrType>()
            .ok_or_else(|| "expected mir.ptr".to_owned())?;
        llvm_pointer_type(pointer.address_space())
    }
}

/// Represent a Rust slice as a data pointer plus 64-bit element count.
struct SliceTypeTranslation;

impl TypeTranslation for SliceTypeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        llvm_struct_type(&[llvm_pointer_type(0)?, MlirType::Integer(64)])
    }
}

/// Preserve the disjoint slice's runtime pointer and extent carrier.
struct DisjointSliceTypeTranslation;

impl TypeTranslation for DisjointSliceTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let slice = source_ref
            .downcast_ref::<MirDisjointSliceType>()
            .ok_or_else(|| "expected mir.disjoint_slice".to_owned())?;
        let mut fields = vec![llvm_pointer_type(0)?, MlirType::Integer(64)];
        for &field in slice.space_types() {
            fields.push(registry.translate_type(ctx, field)?);
        }
        llvm_struct_type(&fields)
    }
}

/// Translate fixed-length array storage with its element type and length.
struct ArrayTypeTranslation;

impl TypeTranslation for ArrayTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let array = source_ref
            .downcast_ref::<MirArrayType>()
            .ok_or_else(|| "expected mir.array".to_owned())?;
        llvm_array_type(
            array.size(),
            &registry.translate_type(ctx, array.element_type())?,
        )
    }
}

/// Build an LLVM struct with verified Rust field order, offsets, and padding.
struct StructTypeTranslation;

impl TypeTranslation for StructTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        aggregate_plan(ctx, source, |field| registry.translate_type(ctx, field)).map(|plan| plan.ty)
    }
}

/// Build an LLVM struct retaining the tuple's physical Rust layout.
struct TupleTypeTranslation;

impl TypeTranslation for TupleTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        aggregate_plan(ctx, source, |field| registry.translate_type(ctx, field)).map(|plan| plan.ty)
    }
}

/// Allocate local storage with source element type, count, and explicit alignment.
struct AllocaTranslation;

impl OperationTranslation for AllocaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("mir.alloca", &input)?;
        let result = only_result("mir.alloca", &input)?;
        let pointee = pointer_pointee(ctx, source.deref(ctx).get_result(0).get_type(ctx))?;
        let target_pointee = session
            .translate_type(pointee)
            .map_err(|error| error.to_string())?;

        let count_id = session.fresh_value();
        let mut count = MlirOperation::new("arith.constant")?;
        count.results.push(MlirResult {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        count.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 1,
                ty: MlirType::Integer(32),
            },
        );
        count.location = input.location.clone();

        let mut alloca = MlirOperation::new("llvm.alloca")?;
        alloca.results.push(result);
        alloca.operands.push(MlirValueUse {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        alloca
            .properties
            .insert("elem_type".into(), MlirAttribute::Type(target_pointee));
        if let Some(alignment) = source_abi_align(ctx, pointee) {
            alloca.properties.insert(
                "alignment".into(),
                MlirAttribute::Integer {
                    value: alignment.into(),
                    ty: MlirType::Integer(64),
                },
            );
        }
        alloca.location = input.location;
        Ok(vec![count, alloca])
    }
}

/// Give an SSA value fresh storage at the reference operation's control-flow position.
///
/// These addresses can escape to helper calls or be compared. Unlike private
/// packing scratch, their allocation must not be hoisted or shared.
struct ReferenceTranslation;

impl OperationTranslation for ReferenceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        MirRefOp::new(source)
            .verify(ctx)
            .map_err(|error| error.to_string())?;
        ensure_empty_attributes("mir.ref", &input)?;
        if input.operands.len() != 1 || !input.regions.is_empty() || !input.successors.is_empty() {
            return Err("mir.ref requires one SSA value and one pointer result".into());
        }
        let result = only_result("mir.ref", &input)?;
        let value_type = source.deref(ctx).get_operand(0).get_type(ctx);
        super::checked::check_payload_storage(ctx, value_type)?;
        let alignment = source_abi_align(ctx, value_type)
            .ok_or("mir.ref requires a known source ABI alignment")?;
        let count_id = session.fresh_value();
        let mut count = MlirOperation::new("arith.constant")?;
        count.results.push(MlirResult {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        count.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 1,
                ty: MlirType::Integer(32),
            },
        );
        count.location = input.location.clone();

        let pointer = MlirValueUse {
            id: result.id,
            ty: result.ty.clone(),
        };
        let mut allocation = MlirOperation::new("llvm.alloca")?;
        allocation.operands.push(MlirValueUse {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        allocation.results.push(result);
        allocation.properties.insert(
            "elem_type".into(),
            MlirAttribute::Type(input.operands[0].ty.clone()),
        );
        add_alignment(&mut allocation, Some(alignment));
        allocation.location = input.location.clone();

        let mut store = MlirOperation::new("llvm.store")?;
        store.operands = vec![input.operands[0].clone(), pointer];
        add_alignment(&mut store, Some(alignment));
        memory_ordering(&mut store);
        store.location = input.location;
        Ok(vec![count, allocation, store])
    }
}

/// Load a verified pointee while preserving the source memory access contract.
struct LoadTranslation;

impl OperationTranslation for LoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        input.attributes.remove("mir_load_volatile");
        ensure_empty_attributes("mir.load", &input)?;
        let mut target = renamed("llvm.load", input)?;
        memory_ordering(&mut target);
        if MirLoadOp::new(source).is_volatile(ctx) {
            target
                .properties
                .insert("volatile_".into(), MlirAttribute::Unit);
        }
        let pointee = pointer_pointee(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        add_alignment(&mut target, source_abi_align(ctx, pointee));
        Ok(vec![target])
    }
}

/// Store a value to its verified pointee with explicit alignment and memory ordering.
struct StoreTranslation;

impl OperationTranslation for StoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        input.attributes.remove("mir_store_volatile");
        ensure_empty_attributes("mir.store", &input)?;
        if input.operands.len() != 2 {
            return Err(format!(
                "mir.store expected two operands, got {}",
                input.operands.len()
            ));
        }
        input.operands.swap(0, 1);
        let mut target = renamed("llvm.store", input)?;
        memory_ordering(&mut target);
        if MirStoreOp::new(source).is_volatile(ctx) {
            target
                .properties
                .insert("volatile_".into(), MlirAttribute::Unit);
        }
        let pointee = pointer_pointee(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        add_alignment(&mut target, source_abi_align(ctx, pointee));
        Ok(vec![target])
    }
}

/// Compute a pointer offset in source pointee elements.
struct PointerOffsetTranslation;

impl OperationTranslation for PointerOffsetTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        input.attributes.remove("mir_ptr_offset_inbounds");
        ensure_empty_attributes("mir.ptr_offset", &input)?;
        let pointee = pointer_pointee(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        let target_pointee = session
            .translate_type(pointee)
            .map_err(|error| error.to_string())?;
        let inbounds = MirPtrOffsetOp::new(source).is_inbounds(ctx);
        let mut target = renamed("llvm.getelementptr", input)?;
        gep_properties(
            &mut target,
            target_pointee,
            vec![i32::MIN],
            if inbounds { 3 } else { 0 },
        );
        Ok(vec![target])
    }
}

/// Extract a logical field using its physical LLVM aggregate slot.
struct ExtractFieldTranslation;

impl OperationTranslation for ExtractFieldTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("mir.extract_field", &input)?;
        let index = MirExtractFieldOp::new(source)
            .get_attr_index(ctx)
            .ok_or_else(|| "mir.extract_field has no index".to_owned())?
            .0 as usize;
        let operand_type = source.deref(ctx).get_operand(0).get_type(ctx);
        let operand_ref = operand_type.deref(ctx);

        if operand_ref.is::<MirUnionType>() {
            return super::storage::extract_union_field(ctx, source, input, session, index);
        }

        if operand_ref.downcast_ref::<IntegerType>().is_some() {
            if index != 0 {
                return Err("scalar newtype extraction only accepts field zero".into());
            }
            return Ok(vec![renamed("builtin.unrealized_conversion_cast", input)?]);
        }

        let slot = if operand_ref.downcast_ref::<MirSliceType>().is_some() {
            (index < 2).then_some(Some(index))
        } else if let Some(slice) = operand_ref.downcast_ref::<MirDisjointSliceType>() {
            (index < slice.field_count()).then_some(Some(index))
        } else if let Some(array) = operand_ref.downcast_ref::<MirArrayType>() {
            (index < array.size() as usize).then_some(Some(index))
        } else if operand_ref.downcast_ref::<MirStructType>().is_some()
            || operand_ref.downcast_ref::<MirTupleType>().is_some()
        {
            aggregate_plan(ctx, operand_type, |field| {
                session
                    .translate_type(field)
                    .map_err(|error| error.to_string())
            })?
            .decl_to_slot
            .get(index)
            .copied()
        } else {
            return Err("mir.extract_field needs a slice, array, struct, tuple, or scalar".into());
        }
        .ok_or_else(|| format!("mir.extract_field index {index} is out of bounds"))?;

        let Some(slot) = slot else {
            let result = only_result("mir.extract_field", &input)?;
            let mut poison = MlirOperation::new("llvm.mlir.poison")?;
            poison.results.push(result);
            poison.location = input.location;
            return Ok(vec![poison]);
        };

        let mut target = renamed("llvm.extractvalue", input)?;
        target.properties.insert(
            "position".into(),
            MlirAttribute::DenseI64Array(vec![slot as i64]),
        );
        Ok(vec![target])
    }
}

/// Build a physical aggregate in Rust declaration order while retaining layout padding.
pub(crate) struct ConstructStructTranslation;

/// Build a fixed array preserving every scalar register's source order.
struct ConstructArrayTranslation;

impl OperationTranslation for ConstructArrayTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        MirConstructArrayOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        ensure_empty_attributes("mir.construct_array", &input)?;
        let result = only_result("mir.construct_array", &input)?;
        let mut poison = MlirOperation::new("llvm.mlir.poison")?;
        let mut current = MlirValueUse {
            id: if input.operands.is_empty() {
                result.id
            } else {
                session.fresh_value()
            },
            ty: result.ty.clone(),
        };
        poison.results.push(MlirResult {
            id: current.id,
            ty: current.ty.clone(),
        });
        poison.location = input.location.clone();
        let mut operations = vec![poison];
        for (index, operand) in input.operands.iter().enumerate() {
            let id = if index + 1 == input.operands.len() {
                result.id
            } else {
                session.fresh_value()
            };
            let mut insert = MlirOperation::new("llvm.insertvalue")?;
            insert.operands = vec![current, operand.clone()];
            insert.results.push(MlirResult {
                id,
                ty: result.ty.clone(),
            });
            insert.properties.insert(
                "position".into(),
                MlirAttribute::DenseI64Array(vec![index as i64]),
            );
            insert.location = input.location.clone();
            operations.push(insert);
            current = MlirValueUse {
                id,
                ty: result.ty.clone(),
            };
        }
        Ok(operations)
    }
}

impl OperationTranslation for ConstructStructTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("mir.construct_struct", &input)?;
        let source_type = source.deref(ctx).get_result(0).get_type(ctx);
        let plan = aggregate_plan(ctx, source_type, |field| {
            session
                .translate_type(field)
                .map_err(|error| error.to_string())
        })?;
        if input.operands.len() != plan.decl_to_slot.len() {
            return Err(format!(
                "mir.construct_struct has {} operands for {} fields",
                input.operands.len(),
                plan.decl_to_slot.len()
            ));
        }
        let source_result = only_result("mir.construct_struct", &input)?;
        let live_fields = plan
            .memory_order
            .iter()
            .copied()
            .filter(|&field| plan.decl_to_slot[field].is_some())
            .collect::<Vec<_>>();

        let mut operations = Vec::with_capacity(live_fields.len() + 1);
        let poison_id = if live_fields.is_empty() {
            source_result.id
        } else {
            session.fresh_value()
        };
        let mut poison = MlirOperation::new("llvm.mlir.poison")?;
        poison.results.push(MlirResult {
            id: poison_id,
            ty: source_result.ty.clone(),
        });
        poison.location = input.location.clone();
        operations.push(poison);

        let mut current = MlirValueUse {
            id: poison_id,
            ty: source_result.ty.clone(),
        };
        for (position, field) in live_fields.iter().copied().enumerate() {
            let result_id = if position + 1 == live_fields.len() {
                source_result.id
            } else {
                session.fresh_value()
            };
            let mut insert = MlirOperation::new("llvm.insertvalue")?;
            insert.results.push(MlirResult {
                id: result_id,
                ty: source_result.ty.clone(),
            });
            insert.operands.push(current);
            insert.operands.push(input.operands[field].clone());
            insert.properties.insert(
                "position".into(),
                MlirAttribute::DenseI64Array(vec![plan.decl_to_slot[field].unwrap() as i64]),
            );
            insert.location = input.location.clone();
            operations.push(insert);
            current = MlirValueUse {
                id: result_id,
                ty: source_result.ty.clone(),
            };
        }
        Ok(operations)
    }
}

/// Construct a slice's pointer and length without changing the pointer's semantic kind.
struct ConstructSliceTranslation;

impl OperationTranslation for ConstructSliceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        MirConstructSliceOp::new(source)
            .verify(ctx)
            .map_err(|error| error.to_string())?;
        build_slice_carrier(input, session)
    }
}

/// Attach an array's static extent without changing its data pointer or ownership kind.
pub(super) fn unsize_array_pointer(
    ctx: &Context,
    source: Ptr<Operation>,
    mut input: OperationInput,
    session: &mut TranslationSession<'_>,
) -> Result<Vec<MlirOperation>, String> {
    MirCastOp::new(source)
        .verify(ctx)
        .map_err(|error| error.to_string())?;
    ensure_empty_attributes("array-to-slice coercion", &input)?;
    if input.operands.len() != 1 {
        return Err("array-to-slice coercion requires one pointer".into());
    }
    let from = source.deref(ctx).get_operand(0).get_type(ctx);
    let from_ref = from.deref(ctx);
    let pointer = from_ref
        .downcast_ref::<MirPtrType>()
        .ok_or("expected array pointer")?;
    let pointee = pointer.pointee.deref(ctx);
    let array = pointee
        .downcast_ref::<MirArrayType>()
        .ok_or("only direct array-to-slice coercion is supported")?;
    let to = source.deref(ctx).get_result(0).get_type(ctx);
    let to_ref = to.deref(ctx);
    let slice = to_ref
        .downcast_ref::<MirSliceType>()
        .ok_or("expected slice result")?;
    if pointer.address_space() != 0
        || array.element_type() != slice.element_ty
        || pointer.kind != slice.kind
        || pointer.is_mutable != slice.is_mutable
    {
        return Err(
            "array-to-slice coercion must preserve element type and pointer kind in generic memory"
                .into(),
        );
    }
    let id = session.fresh_value();
    let mut length = MlirOperation::new("arith.constant")?;
    length.results.push(MlirResult {
        id,
        ty: MlirType::Integer(64),
    });
    length.properties.insert(
        "value".into(),
        MlirAttribute::Integer {
            value: i128::from(array.size()),
            ty: MlirType::Integer(64),
        },
    );
    length.location = input.location.clone();
    input.operands.push(MlirValueUse {
        id,
        ty: MlirType::Integer(64),
    });
    let mut operations = vec![length];
    operations.extend(build_slice_carrier(input, session)?);
    Ok(operations)
}

/// Assemble a verified generic pointer and 64-bit extent into the canonical slice carrier.
fn build_slice_carrier(
    input: OperationInput,
    session: &mut TranslationSession<'_>,
) -> Result<Vec<MlirOperation>, String> {
    ensure_empty_attributes("mir.construct_slice", &input)?;
    if input.operands.len() != 2 || !input.regions.is_empty() || !input.successors.is_empty() {
        return Err("mir.construct_slice requires a pointer and length".into());
    }
    let result = only_result("mir.construct_slice", &input)?;
    if input.operands[0].ty != llvm_pointer_type(0)?
        || input.operands[1].ty != MlirType::Integer(64)
    {
        return Err("mir.construct_slice requires a generic pointer and 64-bit length".into());
    }
    let mut current = MlirValueUse {
        id: session.fresh_value(),
        ty: result.ty.clone(),
    };
    let mut poison = MlirOperation::new("llvm.mlir.poison")?;
    poison.results.push(MlirResult {
        id: current.id,
        ty: current.ty.clone(),
    });
    poison.location = input.location.clone();
    let mut operations = vec![poison];
    for (index, operand) in input.operands.into_iter().enumerate() {
        let id = if index == 1 {
            result.id
        } else {
            session.fresh_value()
        };
        let mut insert = MlirOperation::new("llvm.insertvalue")?;
        insert.operands = vec![current, operand];
        insert.results.push(MlirResult {
            id,
            ty: result.ty.clone(),
        });
        insert.properties.insert(
            "position".into(),
            MlirAttribute::DenseI64Array(vec![index as i64]),
        );
        insert.location = input.location.clone();
        operations.push(insert);
        current = MlirValueUse {
            id,
            ty: result.ty.clone(),
        };
    }
    Ok(operations)
}

/// Compute a logical field address through the verified physical aggregate layout.
struct FieldAddressTranslation;

impl OperationTranslation for FieldAddressTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        // `aggregate_ty` is the build-time-stamped aggregate fact; this
        // translation consumes it as the GEP element type source, so it must
        // not survive as a target attribute.
        let mut input = input;
        input.attributes.remove("aggregate_ty");
        ensure_empty_attributes("mir.field_addr", &input)?;
        let index = MirFieldAddrOp::new(source)
            .get_attr_field_index(ctx)
            .ok_or_else(|| "mir.field_addr has no field_index".to_owned())?
            .0 as usize;
        let pointee = match MirFieldAddrOp::new(source).get_attr_aggregate_ty(ctx) {
            Some(attr) => attr.get_type(ctx),
            None => pointer_pointee(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?,
        };
        if let Some(union) = pointee.deref(ctx).downcast_ref::<MirUnionType>() {
            MirFieldAddrOp::new(source)
                .verify(ctx)
                .map_err(|error| error.to_string())?;
            super::storage::union_storage_type(ctx, union)?;
            if index >= union.field_count() {
                return Err("mir.field_addr union field index is out of bounds".into());
            }
            // All union views begin at byte zero. Keep a distinct result for
            // the selected field while preserving the pointer address space.
            let mut target = renamed("llvm.getelementptr", input)?;
            gep_properties(&mut target, MlirType::Integer(8), vec![0], 0);
            return Ok(vec![target]);
        }
        let plan = aggregate_plan(ctx, pointee, |field| {
            session
                .translate_type(field)
                .map_err(|error| error.to_string())
        })?;
        let slot = plan
            .decl_to_slot
            .get(index)
            .ok_or_else(|| format!("mir.field_addr index {index} is out of bounds"))?;
        let (element_type, indices) = match slot {
            Some(slot) => (plan.ty, vec![0, *slot as i32]),
            None => (MlirType::Integer(8), vec![0]),
        };
        let mut target = renamed("llvm.getelementptr", input)?;
        gep_properties(&mut target, element_type, indices, 0);
        Ok(vec![target])
    }
}

/// Compute the address of a dynamic element within a fixed array.
struct ArrayElementAddressTranslation;

impl OperationTranslation for ArrayElementAddressTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("mir.array_element_addr", &input)?;
        let pointee = pointer_pointee(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        if pointee.deref(ctx).downcast_ref::<MirArrayType>().is_none() {
            return Err("mir.array_element_addr pointer does not point to an array".into());
        }
        let target_array = session
            .translate_type(pointee)
            .map_err(|error| error.to_string())?;
        let mut target = renamed("llvm.getelementptr", input)?;
        gep_properties(&mut target, target_array, vec![0, i32::MIN], 0);
        Ok(vec![target])
    }
}

/// Extract an array element while preserving constant and dynamic index behavior.
struct ExtractArrayElementTranslation;

impl OperationTranslation for ExtractArrayElementTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("mir.extract_array_element", &input)?;
        if input.operands.len() != 2 {
            return Err("mir.extract_array_element expected array and index operands".into());
        }
        let array_source_type = source.deref(ctx).get_operand(0).get_type(ctx);
        let array_ref = array_source_type.deref(ctx);
        let array = array_ref
            .downcast_ref::<MirArrayType>()
            .ok_or_else(|| "mir.extract_array_element operand is not an array".to_owned())?;
        let target_array = session
            .translate_type(array_source_type)
            .map_err(|error| error.to_string())?;
        let result = only_result("mir.extract_array_element", &input)?;
        // Constant projections are register extraction. The memory fallback
        // below remains for dynamic indexing, never for a fixed TMEM lane.
        if let Some(constant) = source
            .deref(ctx)
            .get_operand(1)
            .defining_op()
            .and_then(|op| Operation::get_op::<dialect_mir::ops::MirConstantOp>(op, ctx))
            .and_then(|op| op.get_attr_value(ctx))
        {
            let bits = constant.value();
            if bits.bw() <= 64 && bits.to_u64() < array.size() {
                let mut extract = MlirOperation::new("llvm.extractvalue")?;
                extract.operands.push(input.operands[0].clone());
                extract.results.push(result);
                extract.properties.insert(
                    "position".into(),
                    MlirAttribute::DenseI64Array(vec![bits.to_u64() as i64]),
                );
                extract.location = input.location;
                return Ok(vec![extract]);
            }
        }
        let pointer = llvm_pointer_type(0)?;
        let count_id = session.fresh_value();
        let slot_id = session.fresh_value();
        let element_ptr_id = session.fresh_value();

        let mut count = MlirOperation::new("arith.constant")?;
        count.results.push(MlirResult {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        count.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 1,
                ty: MlirType::Integer(32),
            },
        );
        count.location = input.location.clone();

        let mut alloca = MlirOperation::new("llvm.alloca")?;
        alloca.results.push(MlirResult {
            id: slot_id,
            ty: pointer.clone(),
        });
        alloca.operands.push(MlirValueUse {
            id: count_id,
            ty: MlirType::Integer(32),
        });
        alloca.properties.insert(
            "elem_type".into(),
            MlirAttribute::Type(target_array.clone()),
        );
        add_alignment(&mut alloca, source_abi_align(ctx, array_source_type));
        alloca.location = input.location.clone();

        let mut store = MlirOperation::new("llvm.store")?;
        store.operands.push(input.operands[0].clone());
        store.operands.push(MlirValueUse {
            id: slot_id,
            ty: pointer.clone(),
        });
        memory_ordering(&mut store);
        add_alignment(&mut store, source_abi_align(ctx, array_source_type));
        store.location = input.location.clone();

        let mut gep = MlirOperation::new("llvm.getelementptr")?;
        gep.results.push(MlirResult {
            id: element_ptr_id,
            ty: pointer.clone(),
        });
        gep.operands.push(MlirValueUse {
            id: slot_id,
            ty: pointer.clone(),
        });
        gep.operands.push(input.operands[1].clone());
        gep_properties(&mut gep, target_array, vec![0, i32::MIN], 0);
        gep.location = input.location.clone();

        let mut load = MlirOperation::new("llvm.load")?;
        load.results.push(result);
        load.operands.push(MlirValueUse {
            id: element_ptr_id,
            ty: pointer,
        });
        memory_ordering(&mut load);
        add_alignment(&mut load, source_abi_align(ctx, array.element_type()));
        load.location = input.location;
        Ok(vec![count, alloca, store, gep, load])
    }
}

/// Map reviewed Rust intrinsics directly and preserve ordinary function call signatures.
struct CallTranslation;

impl OperationTranslation for CallTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        input.attributes.remove("callee");
        ensure_empty_attributes("mir.call", &input)?;
        let callee = MirCallOp::new(source)
            .get_attr_callee(ctx)
            .ok_or_else(|| "mir.call has no callee".to_owned())?
            .as_str()
            .to_owned();
        if callee.is_empty() || callee.chars().any(char::is_whitespace) {
            return Err(format!("mir.call has invalid callee {callee:?}"));
        }
        // Intrinsic placeholders are compiler operations, never external symbols.
        // Preserve source type checking before dropping the internal callee name.
        use dialect_mir::rust_intrinsics::{
            CALLEE_MAXNUM_NSZ_F32, CALLEE_SATURATING_ADD, CALLEE_SATURATING_SUB,
            CALLEE_SELECT_UNPREDICTABLE, PLACEHOLDER_PREFIX,
        };
        if callee.starts_with(PLACEHOLDER_PREFIX) {
            MirCallOp::new(source)
                .verify(ctx)
                .map_err(|e| e.to_string())?;
            if callee == CALLEE_MAXNUM_NSZ_F32 {
                let scalar = MlirType::Float(MlirFloatType::F32);
                if input.operands.len() != 2
                    || input.results.len() != 1
                    || input.operands.iter().any(|value| value.ty != scalar)
                    || input.results[0].ty != scalar
                    || !input.attributes.is_empty()
                {
                    return Err("maximum_number_nsz_f32 requires exactly two FP32 operands and one FP32 result".into());
                }
                // Native PTX max.f32 ignores either quiet or signaling NaNs
                // when the other operand is numeric, preserving Rust's max
                // contract. Do not use llvm.maxnum (different sNaN semantics)
                // or enable NVVM's NaN-propagation/FTZ/absolute-value modifiers.
                return Ok(vec![renamed("nvvm.fmax", input)?]);
            }
            if callee == CALLEE_SELECT_UNPREDICTABLE {
                let supported = match &input.results[0].ty {
                    MlirType::Integer(_) | MlirType::Float(_) | MlirType::Vector { .. } => true,
                    MlirType::Dialect(t) => t == "!llvm.ptr" || t.starts_with("!llvm.ptr<"),
                    _ => false,
                };
                if !supported {
                    return Err("select_unpredictable currently requires a scalar, pointer, or vector result".into());
                }
                // The branch-weight hint does not change device semantics.
                return Ok(vec![renamed("arith.select", input)?]);
            }
            if matches!(
                callee.as_str(),
                CALLEE_SATURATING_ADD | CALLEE_SATURATING_SUB
            ) {
                let ty = source.deref(ctx).get_operand(0).get_type(ctx);
                let ty = ty.deref(ctx);
                let integer = ty
                    .downcast_ref::<IntegerType>()
                    .ok_or("saturation requires integers")?;
                let sign = if integer.is_signed() { "s" } else { "u" };
                let operation = if callee == CALLEE_SATURATING_ADD {
                    "add"
                } else {
                    "sub"
                };
                let intrinsic = format!("llvm.{sign}{operation}.sat.i{}", integer.width());
                let mut target = renamed("llvm.call_intrinsic", input)?;
                target
                    .properties
                    .insert("intrin".into(), MlirAttribute::String(intrinsic));
                target.properties.insert(
                    "op_bundle_sizes".into(),
                    MlirAttribute::DenseI32Array(vec![]),
                );
                target.properties.insert(
                    "operandSegmentSizes".into(),
                    MlirAttribute::DenseI32Array(vec![2, 0]),
                );
                return Ok(vec![target]);
            }
            return Err(format!(
                "Rust intrinsic {callee} has no CUTLASS MLIR translation"
            ));
        }
        let unit_result = if MirCallOp::new(source).has_logical_unit_result_bridge(ctx) {
            input.results.pop()
        } else {
            None
        };
        let mut target = renamed("func.call", input)?;
        target
            .properties
            .insert("callee".into(), MlirAttribute::SymbolRef(callee));
        if let Some(result) = unit_result {
            // MIR lets the logical unit flow through a destination place.
            // MLIR's call must instead match the callee's void signature.
            let mut carrier = MlirOperation::new("ub.poison")?;
            carrier.results.push(result);
            carrier.location = target.location.clone();
            Ok(vec![target, carrier])
        } else {
            Ok(vec![target])
        }
    }
}

/// Preserve Rust's mandatory trap independently of optional backend assertions.
struct AssertTranslation;

impl OperationTranslation for AssertTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        Operation::get_op::<MirAssertOp>(source, ctx)
            .ok_or("expected mir.assert")?
            .verify(ctx)
            .map_err(|error| error.to_string())?;
        ensure_empty_attributes("mir.assert", &input)?;
        if input.operands.len() != 1
            || input.operands[0].ty != MlirType::Integer(1)
            || !input.successors.is_empty()
            || !input.results.is_empty()
            || !input.regions.is_empty()
        {
            return Err("mir.assert expected one i1 condition and no successors or results".into());
        }
        // CUTLASS's cute-to-nvvm pipeline removes cf.assert by default when
        // enable-assertions=false. Rust bounds/overflow checks are mandatory
        // source effects. Expose the failure branch and its nonreturning trap
        // explicitly so later optimizations retain the same control flow.
        let true_value = MlirValueUse {
            id: session.fresh_value(),
            ty: MlirType::Integer(1),
        };
        let mut constant = MlirOperation::new("arith.constant")?;
        constant.results.push(MlirResult {
            id: true_value.id,
            ty: true_value.ty.clone(),
        });
        constant.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 1,
                ty: MlirType::Integer(1),
            },
        );
        constant.location = input.location.clone();

        let failure = MlirValueUse {
            id: session.fresh_value(),
            ty: MlirType::Integer(1),
        };
        let mut invert = MlirOperation::new("arith.xori")?;
        invert.operands = vec![input.operands[0].clone(), true_value];
        invert.results.push(MlirResult {
            id: failure.id,
            ty: failure.ty.clone(),
        });
        invert.location = input.location.clone();

        let mut trap = MlirOperation::new("llvm.intr.trap")?;
        trap.location = input.location.clone();
        let mut yield_op = MlirOperation::new("scf.yield")?;
        yield_op.location = input.location.clone();
        let mut conditional = MlirOperation::new("scf.if")?;
        conditional.operands.push(failure);
        conditional.regions = vec![
            crate::MlirRegion {
                blocks: vec![crate::MlirBlock {
                    id: session.fresh_block(),
                    arguments: vec![],
                    operations: vec![trap, yield_op],
                }],
            },
            crate::MlirRegion { blocks: vec![] },
        ];
        conditional.location = input.location;

        // Any following MIR branch is translated independently. The structured
        // failure region only adds the conditional trap before that continuation.
        Ok(vec![constant, invert, conditional])
    }
}

/// Erase a source-only operation after checking that it has no runtime results.
struct DropOperation;

impl OperationTranslation for DropOperation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ensure_empty_attributes("MIR lifetime marker", &input)?;
        if !input.operands.is_empty()
            || !input.results.is_empty()
            || !input.successors.is_empty()
            || !input.regions.is_empty()
        {
            return Err("MIR lifetime marker unexpectedly carries IR data".into());
        }
        Ok(vec![])
    }
}

/// Validated physical LLVM aggregate plus mappings from Rust declaration indices.
#[derive(Debug)]
pub(crate) struct AggregatePlan {
    /// LLVM aggregate type including any explicit padding slots.
    pub(crate) ty: MlirType,
    /// Physical slot for each declared field; zero-sized fields have no slot.
    pub(crate) decl_to_slot: Vec<Option<usize>>,
    /// Rust declaration indices ordered by their physical positions in memory.
    pub(crate) memory_order: Vec<usize>,
}

/// Rebuild Rust aggregate layout and reject overlaps, packed offsets, or incompatible total size.
pub(crate) fn aggregate_plan(
    ctx: &Context,
    source: TypeHandle,
    mut translate_type: impl FnMut(TypeHandle) -> Result<MlirType, String>,
) -> Result<AggregatePlan, String> {
    let source_ref = source.deref(ctx);
    let (fields, memory_order, field_offsets, total_size) =
        if let Some(structure) = source_ref.downcast_ref::<MirStructType>() {
            (
                structure.field_types().to_vec(),
                structure.memory_order(),
                structure.field_offsets().to_vec(),
                structure.total_size(),
            )
        } else if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
            (
                tuple.get_types().to_vec(),
                tuple.memory_order(),
                tuple.field_offsets().to_vec(),
                tuple.total_size(),
            )
        } else {
            return Err("expected a MIR struct or tuple".into());
        };

    validate_memory_order(&memory_order, fields.len())?;
    let explicit = !field_offsets.is_empty() && total_size > 0;
    if explicit && field_offsets.len() != fields.len() {
        return Err(format!(
            "aggregate records {} offsets for {} fields",
            field_offsets.len(),
            fields.len()
        ));
    }

    let translated = fields
        .iter()
        .map(|&field| translate_type(field))
        .collect::<Result<Vec<_>, _>>()?;
    let mut target_fields = Vec::new();
    let mut decl_to_slot = vec![None; fields.len()];
    let mut offset = 0u64;
    let mut max_alignment = 1u64;

    for &declaration in &memory_order {
        let source_field = fields[declaration];
        if source_is_zero_sized(ctx, source_field) {
            continue;
        }
        let field_alignment = source_natural_alignment(ctx, source_field).ok_or_else(|| {
            format!("cannot determine natural alignment of aggregate field {declaration}")
        })?;
        max_alignment = max_alignment.max(field_alignment);
        let target_offset = if explicit {
            field_offsets[declaration]
        } else {
            align_up(offset, field_alignment)?
        };
        if offset > target_offset {
            return Err(format!(
                "aggregate field {declaration} overlaps a previous field at byte {target_offset}"
            ));
        }
        if offset < target_offset {
            target_fields.push(llvm_array_type(
                target_offset - offset,
                &MlirType::Integer(8),
            )?);
            offset = target_offset;
        }
        let llvm_offset = align_up(offset, field_alignment)?;
        if llvm_offset != target_offset {
            return Err(format!(
                "aggregate field {declaration} is at byte {target_offset} in Rust but an unpacked LLVM struct would place it at byte {llvm_offset}; packed by-value aggregates are not supported"
            ));
        }
        decl_to_slot[declaration] = Some(target_fields.len());
        target_fields.push(translated[declaration].clone());
        offset = target_offset
            .checked_add(source_stored_size(ctx, source_field).ok_or_else(|| {
                format!("cannot determine stored size of aggregate field {declaration}")
            })?)
            .ok_or_else(|| "aggregate size overflow".to_owned())?;
    }

    if explicit {
        if offset > total_size {
            return Err(format!(
                "aggregate fields occupy {offset} bytes but Rust reports {total_size}"
            ));
        }
        if offset < total_size {
            target_fields.push(llvm_array_type(total_size - offset, &MlirType::Integer(8))?);
            offset = total_size;
        }
        let natural_size = align_up(offset, max_alignment)?;
        if natural_size != total_size {
            return Err(format!(
                "aggregate is {total_size} bytes in Rust but its unpacked LLVM form is {natural_size} bytes"
            ));
        }
    }

    Ok(AggregatePlan {
        ty: llvm_struct_type(&target_fields)?,
        decl_to_slot,
        memory_order,
    })
}

/// Require a complete, nonduplicated permutation of all declared field indices.
fn validate_memory_order(order: &[usize], fields: usize) -> Result<(), String> {
    if order.len() != fields {
        return Err(format!(
            "aggregate memory order has {} entries for {fields} fields",
            order.len()
        ));
    }
    let mut seen = vec![false; fields];
    for &field in order {
        if field >= fields || seen[field] {
            return Err(format!(
                "aggregate memory order {order:?} is not a permutation of 0..{fields}"
            ));
        }
        seen[field] = true;
    }
    Ok(())
}

/// Determine whether a reviewed source type occupies no stored bytes.
fn source_is_zero_sized(ctx: &Context, source: TypeHandle) -> bool {
    let source_ref = source.deref(ctx);
    if let Some(union) = source_ref.downcast_ref::<MirUnionType>() {
        return union.total_size() == 0;
    }
    if let Some(e) = source_ref.downcast_ref::<MirEnumType>() {
        return e.total_size == 0 && crate::mir::checked::enum_storage_type(e).is_ok();
    }
    if source_ref.downcast_ref::<UnitType>().is_some() {
        return true;
    }
    if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
        return array.size() == 0 || source_is_zero_sized(ctx, array.element_type());
    }
    if let Some(structure) = source_ref.downcast_ref::<MirStructType>() {
        return structure
            .field_types()
            .iter()
            .all(|&field| source_is_zero_sized(ctx, field));
    }
    if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
        return tuple
            .get_types()
            .iter()
            .all(|&field| source_is_zero_sized(ctx, field));
    }
    false
}

/// Return the reviewed source type's storage size in bytes, or None when unknown.
pub(crate) fn source_stored_size(ctx: &Context, source: TypeHandle) -> Option<u64> {
    let source_ref = source.deref(ctx);
    if let Some(union) = source_ref.downcast_ref::<MirUnionType>() {
        return Some(union.total_size());
    }
    if let Some(e) = source_ref.downcast_ref::<MirEnumType>() {
        return crate::mir::checked::enum_storage_type(e)
            .ok()
            .map(|_| e.total_size);
    }
    if let Some(integer) = source_ref.downcast_ref::<IntegerType>() {
        return Some(u64::from(integer.width()).div_ceil(8));
    }
    if source_ref.downcast_ref::<FP16Type>().is_some()
        || source_ref.downcast_ref::<MirFP16Type>().is_some()
    {
        return Some(2);
    }
    if source_ref.downcast_ref::<FP32Type>().is_some() {
        return Some(4);
    }
    if source_ref.downcast_ref::<FP64Type>().is_some() {
        return Some(8);
    }
    if source_ref.downcast_ref::<MirPtrType>().is_some() {
        return Some(8);
    }
    if source_ref.downcast_ref::<MirSliceType>().is_some() {
        return Some(16);
    }
    if let Some(slice) = source_ref.downcast_ref::<MirDisjointSliceType>() {
        let mut size = 16u64;
        let mut alignment = 8u64;
        for &field in slice.space_types() {
            let field_alignment = source_natural_alignment(ctx, field)?;
            size = align_up(size, field_alignment).ok()?;
            size = size.checked_add(source_stored_size(ctx, field)?)?;
            alignment = alignment.max(field_alignment);
        }
        return align_up(size, alignment).ok();
    }
    if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
        return source_stored_size(ctx, array.element_type())?.checked_mul(array.size());
    }
    if let Some(structure) = source_ref.downcast_ref::<MirStructType>() {
        if structure.total_size() > 0 {
            return Some(structure.total_size());
        }
        return natural_aggregate_size(ctx, structure.field_types(), &structure.memory_order());
    }
    if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
        if tuple.total_size() > 0 {
            return Some(tuple.total_size());
        }
        return natural_aggregate_size(ctx, tuple.get_types(), &tuple.memory_order());
    }
    if source_ref.downcast_ref::<UnitType>().is_some() {
        return Some(0);
    }
    None
}

/// Return the unpacked LLVM-compatible storage alignment in bytes when known.
pub(crate) fn source_natural_alignment(ctx: &Context, source: TypeHandle) -> Option<u64> {
    let source_ref = source.deref(ctx);
    if let Some(union) = source_ref.downcast_ref::<MirUnionType>() {
        return Some(union.abi_align());
    }
    if let Some(e) = source_ref.downcast_ref::<MirEnumType>() {
        return crate::mir::checked::enum_storage_type(e)
            .ok()
            .map(|_| e.abi_align);
    }
    if let Some(integer) = source_ref.downcast_ref::<IntegerType>() {
        return Some(
            u64::from(integer.width())
                .div_ceil(8)
                .next_power_of_two()
                .min(16),
        );
    }
    if source_ref.downcast_ref::<FP16Type>().is_some()
        || source_ref.downcast_ref::<MirFP16Type>().is_some()
    {
        return Some(2);
    }
    if source_ref.downcast_ref::<FP32Type>().is_some() {
        return Some(4);
    }
    if source_ref.downcast_ref::<FP64Type>().is_some()
        || source_ref.downcast_ref::<MirPtrType>().is_some()
        || source_ref.downcast_ref::<MirSliceType>().is_some()
        || source_ref.downcast_ref::<MirDisjointSliceType>().is_some()
    {
        return Some(8);
    }
    if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
        return source_natural_alignment(ctx, array.element_type());
    }
    if let Some(structure) = source_ref.downcast_ref::<MirStructType>() {
        return structure
            .field_types()
            .iter()
            .filter(|&&field| !source_is_zero_sized(ctx, field))
            .filter_map(|&field| source_natural_alignment(ctx, field))
            .max()
            .or(Some(1));
    }
    if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
        return tuple
            .get_types()
            .iter()
            .filter(|&&field| !source_is_zero_sized(ctx, field))
            .filter_map(|&field| source_natural_alignment(ctx, field))
            .max()
            .or(Some(1));
    }
    if source_ref.downcast_ref::<UnitType>().is_some() {
        return Some(1);
    }
    None
}

/// Read explicit Rust aggregate or enum alignment, falling back to natural alignment.
pub(super) fn source_abi_align(ctx: &Context, source: TypeHandle) -> Option<u64> {
    let source_ref = source.deref(ctx);
    if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
        return source_abi_align(ctx, array.element_type());
    }
    if let Some(structure) = source_ref.downcast_ref::<MirStructType>()
        && structure.abi_align > 0
    {
        return Some(structure.abi_align);
    }
    if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>()
        && tuple.abi_align() > 0
    {
        return Some(tuple.abi_align());
    }
    source_natural_alignment(ctx, source)
}

/// Compute padded byte size for fields in their physical memory order.
fn natural_aggregate_size(ctx: &Context, fields: &[TypeHandle], order: &[usize]) -> Option<u64> {
    let mut offset = 0u64;
    let mut alignment = 1u64;
    for &field in order {
        if source_is_zero_sized(ctx, fields[field]) {
            continue;
        }
        let field_alignment = source_natural_alignment(ctx, fields[field])?;
        offset = align_up(offset, field_alignment).ok()?;
        offset = offset.checked_add(source_stored_size(ctx, fields[field])?)?;
        alignment = alignment.max(field_alignment);
    }
    align_up(offset, alignment).ok()
}

/// Round a byte offset up to a nonzero power-of-two alignment, rejecting overflow.
fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("invalid alignment {alignment}"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "alignment overflow".to_owned())
}

/// Require a MIR pointer and recover its source pointee type.
fn pointer_pointee(ctx: &Context, pointer: TypeHandle) -> Result<TypeHandle, String> {
    let pointer_ref = pointer.deref(ctx);
    pointer_ref
        .downcast_ref::<MirPtrType>()
        .map(|pointer| pointer.pointee)
        .ok_or_else(|| "expected mir.ptr".to_owned())
}

/// Spell an opaque LLVM pointer in the requested numeric address space.
fn llvm_pointer_type(address_space: u32) -> Result<MlirType, String> {
    if address_space == 0 {
        MlirType::dialect("!llvm.ptr")
    } else {
        MlirType::dialect(format!("!llvm.ptr<{address_space}>"))
    }
}

/// Spell an LLVM fixed array with its element count and translated element type.
fn llvm_array_type(size: u64, element: &MlirType) -> Result<MlirType, String> {
    MlirType::dialect(format!("!llvm.array<{size} x {}>", type_spelling(element)))
}

/// Spell an unpacked LLVM struct in the supplied physical field order.
fn llvm_struct_type(fields: &[MlirType]) -> Result<MlirType, String> {
    let fields = fields
        .iter()
        .map(type_spelling)
        .collect::<Vec<_>>()
        .join(", ");
    MlirType::dialect(format!("!llvm.struct<({fields})>"))
}

/// Render a target type for embedding in an LLVM dialect type spelling.
pub(crate) fn type_spelling(ty: &MlirType) -> String {
    match ty {
        MlirType::Index => "index".into(),
        MlirType::Integer(width) => format!("i{width}"),
        MlirType::Float(MlirFloatType::F16) => "f16".into(),
        MlirType::Float(MlirFloatType::BF16) => "bf16".into(),
        MlirType::Float(MlirFloatType::F32) => "f32".into(),
        MlirType::Float(MlirFloatType::F64) => "f64".into(),
        MlirType::Float(MlirFloatType::Other(spelling)) => spelling.clone(),
        MlirType::Vector { shape, element } => format!(
            "vector<{}{}>",
            shape
                .iter()
                .map(|value| format!("{value}x"))
                .collect::<String>(),
            type_spelling(element)
        ),
        MlirType::Function { inputs, results } => format!(
            "({}) -> ({})",
            inputs
                .iter()
                .map(type_spelling)
                .collect::<Vec<_>>()
                .join(", "),
            results
                .iter()
                .map(type_spelling)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        MlirType::Tuple(elements) => format!(
            "tuple<{}>",
            elements
                .iter()
                .map(type_spelling)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        MlirType::Dialect(spelling) => spelling.clone(),
    }
}

/// Require exactly one translated result and return its SSA definition.
fn only_result(name: &str, input: &OperationInput) -> Result<MlirResult, String> {
    match input.results.as_slice() {
        [result] => Ok(result.clone()),
        results => Err(format!("{name} expected one result, got {}", results.len())),
    }
}

/// Reject source attributes not consumed by the recipe.
fn ensure_empty_attributes(name: &str, input: &OperationInput) -> Result<(), String> {
    if input.attributes.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{name} has targetless attributes: {:?}",
            input.attributes.keys().collect::<Vec<_>>()
        ))
    }
}

/// Retain translated operation structure while changing its target operation name.
fn renamed(name: &str, input: OperationInput) -> Result<MlirOperation, String> {
    let mut target = MlirOperation::new(name)?;
    target.results = input.results;
    target.operands = input.operands;
    target.successors = input.successors;
    target.regions = input.regions;
    target.attributes = input.attributes;
    target.location = input.location;
    Ok(target)
}

/// Mark an ordinary load/store as non-atomic in the LLVM dialect contract.
fn memory_ordering(operation: &mut MlirOperation) {
    operation.properties.insert(
        "ordering".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(64),
        },
    );
}

/// Attach a byte alignment property when the source records one.
fn add_alignment(operation: &mut MlirOperation, alignment: Option<u64>) {
    if let Some(alignment) = alignment {
        operation.properties.insert(
            "alignment".into(),
            MlirAttribute::Integer {
                value: alignment.into(),
                ty: MlirType::Integer(64),
            },
        );
    }
}

/// Set GEP element and index metadata without promising no-wrap behavior.
fn gep_properties(
    operation: &mut MlirOperation,
    element_type: MlirType,
    indices: Vec<i32>,
    no_wrap_flags: i128,
) {
    operation
        .properties
        .insert("elem_type".into(), MlirAttribute::Type(element_type));
    operation.properties.insert(
        "noWrapFlags".into(),
        MlirAttribute::Integer {
            value: no_wrap_flags,
            ty: MlirType::Integer(32),
        },
    );
    operation.properties.insert(
        "rawConstantIndices".into(),
        MlirAttribute::DenseI32Array(indices),
    );
}

#[cfg(test)]
mod tests {
    //! Physical aggregate, dynamic array, and effectful assertion translation checks.

    use crate::{
        CutlassFullCuteMlir22, MlirConsumerProfile,
        profile::render_mapping_module_without_cutlass_envelope,
    };
    use dialect_mir::{
        attributes::{CompilerResultBundleAttr, FieldIndexAttr},
        ops::{
            MirAddOp, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirConstructArrayOp,
            MirExtractArrayElementOp, MirExtractFieldOp, MirFieldAddrOp, MirFuncOp, MirLoadOp,
            MirReturnOp, MirStoreOp,
        },
        types::{MirArrayType, MirPtrType, MirStructType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FP32Type, FunctionType, IntegerType, Signedness, UnitType},
        },
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, TypedHandle},
        value::Value,
    };

    /// Construct a typed source operation and append it to the test block.
    fn operation<O: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        results: Vec<TypeHandle>,
        operands: Vec<Value>,
    ) -> Ptr<Operation> {
        let operation =
            Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
        operation.insert_at_back(block, ctx);
        operation
    }

    /// Obtain an unsigned source integer type with the requested bit width.
    fn integer(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Unsigned)
    }

    /// Rust assertions must trap even when the downstream compiler disables optional assertions.
    #[test]
    fn assertion_remains_an_effectful_check_before_following_operations() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("assertion").unwrap());
        let condition = integer(&mut ctx, 1);
        let word = integer(&mut ctx, 32);
        let pointer = MirPtrType::get(&mut ctx, word.into(), true, 0);
        let inputs = vec![condition.into(), word.into(), pointer.into()];
        let ty = FunctionType::get(&ctx, inputs.clone(), vec![word.into()]);
        let function_op = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(ty.into()));
        function.set_symbol_name(&mut ctx, Identifier::try_from("checked_add").unwrap());
        module.append_operation(&mut ctx, function_op, 0);
        let block = BasicBlock::new(&mut ctx, None, inputs);
        block.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
        let condition = block.deref(&ctx).get_argument(0);
        let word = block.deref(&ctx).get_argument(1);
        let pointer = block.deref(&ctx).get_argument(2);
        operation::<MirAssertOp>(&mut ctx, block, vec![], vec![condition]);
        let result_type = integer(&mut ctx, 32);
        let sum =
            operation::<MirAddOp>(&mut ctx, block, vec![result_type.into()], vec![word, word]);
        let sum = sum.deref(&ctx).get_result(0);
        operation::<MirStoreOp>(&mut ctx, block, vec![], vec![pointer, sum]);
        operation::<MirReturnOp>(&mut ctx, block, vec![], vec![sum]);
        let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
        let target = profile.translate_module(&ctx, &module).unwrap();
        let text = render_mapping_module_without_cutlass_envelope(&target, "assertion");
        assert_eq!(text.matches("\"llvm.intr.trap\"").count(), 1, "{text}");
        assert_eq!(text.matches("\"scf.if\"").count(), 1, "{text}");
        assert!(
            text.find("\"scf.if\"").unwrap() < text.find("\"llvm.store\"").unwrap(),
            "{text}"
        );
        let gpu_module = &target.root.regions[0].blocks[0].operations[0];
        let function = &gpu_module.regions[0].blocks[0].operations[0];
        let entry = &function.regions[0].blocks[0];
        let inversion = &entry.operations[1];
        assert_eq!(inversion.name, "arith.xori");
        assert_eq!(inversion.operands[0].id, entry.arguments[0].id);
        let true_constant = &entry.operations[0];
        assert_eq!(true_constant.name, "arith.constant");
        assert_eq!(
            true_constant.properties.get("value"),
            Some(&crate::MlirAttribute::Integer {
                value: 1,
                ty: crate::MlirType::Integer(1),
            }),
        );
        assert_eq!(inversion.operands[1].id, true_constant.results[0].id);
        let conditional = &entry.operations[2];
        assert_eq!(conditional.operands[0].id, inversion.results[0].id);
        let failure = &conditional.regions[0].blocks[0].operations;
        assert_eq!(failure[0].name, "llvm.intr.trap");
        assert_eq!(failure[1].name, "scf.yield");
        assert!(conditional.regions[1].blocks.is_empty());
        assert!(!text.contains("\"cf.assert\""), "{text}");
        assert!(!text.contains("\"cf.br\""), "{text}");
    }

    /// Conditions wider than a boolean cannot be reinterpreted as an assertion predicate.
    #[test]
    fn assertion_rejects_non_boolean_conditions() {
        for width in [8, 32] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            let module =
                ModuleOp::new(&mut ctx, Identifier::try_from("invalid_assertion").unwrap());
            let condition = integer(&mut ctx, width);
            let ty = FunctionType::get(&ctx, vec![condition.into()], vec![]);
            let function_op = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(ty.into()));
            function.set_symbol_name(&mut ctx, Identifier::try_from("invalid_condition").unwrap());
            module.append_operation(&mut ctx, function_op, 0);
            let block = BasicBlock::new(&mut ctx, None, vec![condition.into()]);
            block.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
            let condition = block.deref(&ctx).get_argument(0);
            operation::<MirAssertOp>(&mut ctx, block, vec![], vec![condition]);
            operation::<MirReturnOp>(&mut ctx, block, vec![], vec![]);
            let error = CutlassFullCuteMlir22::new("sm_100a")
                .unwrap()
                .translate_module(&ctx, &module)
                .unwrap_err();
            assert!(error.to_string().contains("i1"), "{error}");
        }
    }

    /// Check compiler-created arrays retain scalar lane order and valid zero-length representations.
    #[test]
    fn compiler_result_array_preserves_lane_order_and_empty_arrays() {
        for count in [0, 4, 32] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            let module = ModuleOp::new(&mut ctx, Identifier::try_from("registers").unwrap());
            let word = integer(&mut ctx, 32);
            let array = MirArrayType::get(&mut ctx, word.into(), count);
            let args = vec![word.into(); count as usize];
            let ty = FunctionType::get(&ctx, args.clone(), vec![array.into()]);
            let function_op = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(ty.into()));
            function.set_symbol_name(&mut ctx, Identifier::try_from("bundle").unwrap());
            module.append_operation(&mut ctx, function_op, 0);
            let block = BasicBlock::new(&mut ctx, None, args);
            block.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
            let values = (0..count as usize)
                .map(|i| block.deref(&ctx).get_argument(i))
                .collect();
            let bundle =
                operation::<MirConstructArrayOp>(&mut ctx, block, vec![array.into()], values);
            bundle.deref_mut(&ctx).attributes.set(
                Identifier::try_from("compiler_result_bundle").unwrap(),
                CompilerResultBundleAttr(true),
            );
            let value = bundle.deref(&ctx).get_result(0);
            operation::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);
            let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
            let target = profile.translate_module(&ctx, &module).unwrap();
            let text = render_mapping_module_without_cutlass_envelope(&target, "registers");
            assert_eq!(
                text.matches("\"llvm.insertvalue\"").count(),
                count as usize,
                "{text}"
            );
            for i in 0..count {
                let position = format!("position = array<i64: {i}>");
                let insert = text.lines().find(|line| line.contains(&position)).unwrap();
                assert!(insert.contains(&format!(", %v{i})")), "{text}");
            }
            assert!(!text.contains("compiler_result_bundle"), "{text}");
            assert!(!text.contains("llvm.alloca"), "{text}");
        }
    }

    /// Check logical fields map to physical aggregate slots and dynamic array accesses retain element semantics.
    #[test]
    fn aggregate_slots_and_dynamic_array_access_map_to_llvm() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, Identifier::try_from("memory").unwrap());
        let f32_type = FP32Type::get(&ctx);
        let u64_type = integer(&mut ctx, 64);
        let unit_type = UnitType::get(&ctx);
        let array_type = MirArrayType::get(&mut ctx, f32_type.into(), 4);
        let structure_type = MirStructType::get_with_full_layout(
            &mut ctx,
            "RegisterTile".into(),
            vec!["marker".into(), "values".into()],
            vec![unit_type.into(), array_type.into()],
            vec![0, 1],
            vec![0, 0],
            16,
            4,
        );
        let structure_pointer = MirPtrType::get_generic(&mut ctx, structure_type.into(), true);
        let array_pointer = MirPtrType::get_generic(&mut ctx, array_type.into(), true);
        let element_pointer = MirPtrType::get_generic(&mut ctx, f32_type.into(), true);

        let function_type = FunctionType::get(
            &ctx,
            vec![structure_type.into(), u64_type.into()],
            vec![f32_type.into()],
        );
        let function_operation = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(
            &mut ctx,
            function_operation,
            TypeAttr::new(function_type.into()),
        );
        function.set_symbol_name(&mut ctx, Identifier::try_from("read_tile").unwrap());
        function_operation.deref_mut(&ctx).attributes.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
        module.append_operation(&mut ctx, function_operation, 0);
        let region = function_operation.deref(&ctx).get_region(0);
        let entry = BasicBlock::new(&mut ctx, None, vec![structure_type.into(), u64_type.into()]);
        entry.insert_at_back(region, &ctx);
        let arguments = entry.deref(&ctx).arguments().collect::<Vec<_>>();

        let alloca =
            operation::<MirAllocaOp>(&mut ctx, entry, vec![structure_pointer.into()], vec![]);
        let slot = alloca.deref(&ctx).get_result(0);
        operation::<MirStoreOp>(&mut ctx, entry, vec![], vec![slot, arguments[0]]);

        let field =
            operation::<MirFieldAddrOp>(&mut ctx, entry, vec![array_pointer.into()], vec![slot]);
        MirFieldAddrOp::new(field).set_attr_field_index(&ctx, FieldIndexAttr(1));
        MirFieldAddrOp::new(field).set_attr_aggregate_ty(
            &ctx,
            pliron::builtin::attributes::TypeAttr::new(structure_type.into()),
        );
        let field_result = field.deref(&ctx).get_result(0);
        let element = operation::<MirArrayElementAddrOp>(
            &mut ctx,
            entry,
            vec![element_pointer.into()],
            vec![field_result, arguments[1]],
        );
        let element_result = element.deref(&ctx).get_result(0);
        let loaded =
            operation::<MirLoadOp>(&mut ctx, entry, vec![f32_type.into()], vec![element_result]);
        let loaded_result = loaded.deref(&ctx).get_result(0);

        let array = operation::<MirExtractFieldOp>(
            &mut ctx,
            entry,
            vec![array_type.into()],
            vec![arguments[0]],
        );
        MirExtractFieldOp::new(array).set_attr_index(&ctx, FieldIndexAttr(1));
        let array_result = array.deref(&ctx).get_result(0);
        let extracted = operation::<MirExtractArrayElementOp>(
            &mut ctx,
            entry,
            vec![f32_type.into()],
            vec![array_result, arguments[1]],
        );
        let extracted_result = extracted.deref(&ctx).get_result(0);
        let sum = operation::<MirAddOp>(
            &mut ctx,
            entry,
            vec![f32_type.into()],
            vec![loaded_result, extracted_result],
        );
        let sum_result = sum.deref(&ctx).get_result(0);
        operation::<MirReturnOp>(&mut ctx, entry, vec![], vec![sum_result]);

        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let target = profile.translate_module(&ctx, &module).unwrap();
        let text = render_mapping_module_without_cutlass_envelope(&target, "memory");
        let expected = r#""builtin.module"() <{sym_name = "memory"}> ({
  ^bb0:
    "func.func"() <{function_type = (!llvm.struct<(!llvm.array<4 x f32>)>, i64) -> (f32), sym_name = "read_tile"}> ({
      ^bb1(%v0: !llvm.struct<(!llvm.array<4 x f32>)>, %v1: i64):
        %v9 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
        %v2 = "llvm.alloca"(%v9) <{alignment = 4 : i64, elem_type = !llvm.struct<(!llvm.array<4 x f32>)>}> : (i32) -> (!llvm.ptr)
        "llvm.store"(%v0, %v2) <{alignment = 4 : i64, ordering = 0 : i64}> : (!llvm.struct<(!llvm.array<4 x f32>)>, !llvm.ptr) -> ()
        %v3 = "llvm.getelementptr"(%v2) <{elem_type = !llvm.struct<(!llvm.array<4 x f32>)>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, 0>}> : (!llvm.ptr) -> (!llvm.ptr)
        %v4 = "llvm.getelementptr"(%v3, %v1) <{elem_type = !llvm.array<4 x f32>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, -2147483648>}> : (!llvm.ptr, i64) -> (!llvm.ptr)
        %v5 = "llvm.load"(%v4) <{alignment = 4 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> (f32)
        %v6 = "llvm.extractvalue"(%v0) <{position = array<i64: 0>}> : (!llvm.struct<(!llvm.array<4 x f32>)>) -> (!llvm.array<4 x f32>)
        %v10 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
        %v11 = "llvm.alloca"(%v10) <{alignment = 4 : i64, elem_type = !llvm.array<4 x f32>}> : (i32) -> (!llvm.ptr)
        "llvm.store"(%v6, %v11) <{alignment = 4 : i64, ordering = 0 : i64}> : (!llvm.array<4 x f32>, !llvm.ptr) -> ()
        %v12 = "llvm.getelementptr"(%v11, %v1) <{elem_type = !llvm.array<4 x f32>, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: 0, -2147483648>}> : (!llvm.ptr, i64) -> (!llvm.ptr)
        %v7 = "llvm.load"(%v12) <{alignment = 4 : i64, ordering = 0 : i64}> : (!llvm.ptr) -> (f32)
        %v8 = "arith.addf"(%v5, %v7) : (f32, f32) -> (f32)
        "func.return"(%v8) : (f32) -> ()
    }) {cute.kernel = unit, gpu.kernel = unit} : () -> ()
}) : () -> ()
"#;
        assert_eq!(
            crate::profile::normalize_mlir_identifiers(&text),
            crate::profile::normalize_mlir_identifiers(expected)
        );
    }
}
