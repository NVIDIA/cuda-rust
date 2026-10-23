/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Physical Rust carriers needed by checked instruction-level APIs.
//! Enum storage retains rustc's byte offsets, size and alignment. Niche encoding
//! remains unsupported rather than being approximated by a logical tag.
use crate::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};
use dialect_mir::{
    attributes::VariantIndexAttr,
    ops::*,
    types::{
        EnumCarrierKind, EnumLayoutKind, MirArrayType, MirEnumType, MirPtrType, MirStructType,
        MirTupleType,
    },
};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Register physical enum carriers, checked integer arithmetic, and unreachable mappings.
pub(crate) fn register(r: &mut TranslationRegistry) -> Result<(), TranslationError> {
    r.register_type::<MirEnumType>(EnumType)?;
    r.register_attribute::<VariantIndexAttr>(DropAttribute)?;
    r.register_operation::<MirConstructEnumOp>(EnumOp::Construct)?;
    r.register_operation::<MirGetDiscriminantOp>(EnumOp::Discriminant)?;
    r.register_operation::<MirEnumPayloadOp>(EnumOp::Payload)?;
    r.register_operation::<MirCheckedAddOp>(Checked("add"))?;
    r.register_operation::<MirCheckedMulOp>(Checked("mul"))?;
    r.register_operation::<MirUnreachableOp>(Unreachable)?;
    Ok(())
}

/// Validate reviewed enum layouts and preserve their physical size and alignment.
pub(crate) fn enum_storage_type(e: &MirEnumType) -> Result<MlirType, String> {
    // Infallible core array helpers retain their uninhabited residual type in
    // unreachable branches. It has a carrier for SSA typing, but no value can
    // be constructed and no discriminant or payload may be read from it.
    if e.layout_kind == EnumLayoutKind::Empty {
        if e.total_size != 0
            || e.abi_align != 1
            || e.tag_offset != 0
            || e.carrier_kind != EnumCarrierKind::None
            || e.carrier_width != 0
            || e.carrier_address_space != 0
            || e.variant_names.len() != e.variant_discriminants.len()
            || e.variant_names.len() != e.variant_field_counts.len()
            || e.variant_names.len() != e.variant_inhabited.len()
            || e.variant_inhabited.iter().any(|&inhabited| inhabited != 0)
            || e.variant_field_counts
                .iter()
                .map(|&n| u64::from(n))
                .sum::<u64>()
                != e.all_field_types.len() as u64
            || e.all_field_types.len() != e.all_field_offsets.len()
            || e.all_field_types.len() != e.all_field_sizes.len()
            || e.all_field_offsets.iter().any(|&offset| offset != 0)
            || e.all_field_sizes.iter().any(|&size| size != 0)
        {
            return Err(
                "empty enum storage requires a zero-sized layout with no inhabited variants".into(),
            );
        }
        return MlirType::dialect("!llvm.struct<(array<0 x i8>, array<0 x i8>)>");
    }
    if !matches!(
        e.layout_kind,
        EnumLayoutKind::Direct | EnumLayoutKind::Single
    ) {
        return Err("MLIR export supports only reviewed Direct/Single enum layouts".into());
    }
    if !matches!(e.abi_align, 1 | 2 | 4 | 8) || !e.total_size.is_multiple_of(e.abi_align) {
        return Err("enum storage needs known size and alignment of 1, 2, 4 or 8".into());
    }
    if e.layout_kind == EnumLayoutKind::Direct
        && (e.carrier_kind != EnumCarrierKind::Integer
            || !matches!(e.carrier_width, 8 | 16 | 32 | 64)
            || e.tag_offset
                .checked_add(u64::from(e.carrier_width) / 8)
                .is_none_or(|end| end > e.total_size))
    {
        return Err("direct enum needs an in-bounds integer tag".into());
    }
    if e.all_field_types.len() != e.all_field_offsets.len()
        || e.all_field_types.len() != e.all_field_sizes.len()
    {
        return Err("enum lacks complete rustc field layout".into());
    }
    for (&offset, &size) in e.all_field_offsets.iter().zip(&e.all_field_sizes) {
        if offset
            .checked_add(size)
            .is_none_or(|end| end > e.total_size)
        {
            return Err("enum field exceeds storage".into());
        }
    }
    // Independent bytes keep inactive payload/padding undef separate from the
    // initialized tag. A zero-length carrier supplies alignment without bytes.
    MlirType::dialect(format!(
        "!llvm.struct<(array<{} x i8>, array<0 x i{}>)>",
        e.total_size,
        e.abi_align * 8
    ))
}
/// Translate a reviewed enum layout without conflating inactive bytes with its initialized tag.
struct EnumType;
// These payloads require storage adapters, not merely a same-size load/store.
// Keep the initial enum pack fail-closed until those adapters are reviewed.
/// Recursively reject bool and named-space pointer payloads needing unimplemented byte adapters.
pub(super) fn check_payload_storage(ctx: &Context, ty: TypeHandle) -> Result<(), String> {
    let t = ty.deref(ctx);
    if t.downcast_ref::<IntegerType>()
        .is_some_and(|i| i.width() == 1)
    {
        return Err("enum bool payloads require canonical byte storage".into());
    }
    if t.downcast_ref::<MirPtrType>()
        .is_some_and(|p| p.address_space() != 0)
    {
        return Err("enum named-space pointers require generic storage adapters".into());
    }
    let fields: &[TypeHandle] = if let Some(e) = t.downcast_ref::<MirEnumType>() {
        &e.all_field_types
    } else if let Some(s) = t.downcast_ref::<MirStructType>() {
        s.field_types()
    } else if let Some(t) = t.downcast_ref::<MirTupleType>() {
        t.get_types()
    } else if let Some(a) = t.downcast_ref::<MirArrayType>() {
        if a.size() != 0 {
            check_payload_storage(ctx, a.element_type())?;
        }
        return Ok(());
    } else {
        return Ok(());
    };
    for &field in fields {
        check_payload_storage(ctx, field)?;
    }
    Ok(())
}
impl TypeTranslation for EnumType {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        check_payload_storage(ctx, source)?;
        enum_storage_type(
            source
                .deref(ctx)
                .downcast_ref::<MirEnumType>()
                .ok_or("expected enum")?,
        )
    }
}

/// Build an operation with the supplied result IDs, operands, and source location.
pub(super) fn op(
    name: &str,
    operands: Vec<MlirValueUse>,
    results: Vec<MlirResult>,
    loc: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut op = MlirOperation::new(name)?;
    op.operands = operands;
    op.results = results;
    op.location = loc.clone();
    Ok(op)
}
/// Allocate one SSA identifier and return matching result and operand handles.
fn fresh(s: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = s.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}
/// Append an integer constant of the requested bit width to the output sequence.
fn integer(
    value: i128,
    width: u32,
    s: &mut TranslationSession<'_>,
    loc: &MlirLocation,
    out: &mut Vec<MlirOperation>,
) -> Result<MlirValueUse, String> {
    let ty = MlirType::Integer(width);
    let (r, v) = fresh(s, ty.clone());
    let mut c = op("arith.constant", vec![], vec![r], loc)?;
    c.properties
        .insert("value".into(), MlirAttribute::Integer { value, ty });
    out.push(c);
    Ok(v)
}
// Private markers are consumed by FunctionTranslation before MLIR is emitted.
// Only the scratch allocations below qualify; Rust/MIR allocas are untouched.
/// Private marker identifying the constant allocation count for enum scratch storage.
const ENUM_SCRATCH_COUNT: &str = "cuda_oxide.enum_scratch_count";
/// Private marker identifying a scratch allocation eligible for entry-block hoisting.
const ENUM_SCRATCH_SLOT: &str = "cuda_oxide.enum_scratch_slot";

/// Make enum packing scratch visible to LLVM's entry-block scalar replacement.
///
/// Each slot is private to one enum operation: its address is only used by
/// immediate byte-GEP/store/load operations. Moving the allocation, without
/// moving those accesses, preserves conditional initialization and loop values.
/// Distinct slots are never merged, and inactive payload/padding bytes retain
/// their independent byte representation.
pub(crate) fn hoist_enum_scratch(function: &mut MlirOperation) -> Result<(), String> {
    /// Recognize either private marker used by enum scratch hoisting.
    fn marked(op: &MlirOperation) -> bool {
        op.attributes.contains_key(ENUM_SCRATCH_COUNT)
            || op.attributes.contains_key(ENUM_SCRATCH_SLOT)
    }
    /// Reject marked scratch operations nested below the function's direct block operations.
    fn reject_nested_markers(op: &MlirOperation) -> Result<(), String> {
        for region in &op.regions {
            for block in &region.blocks {
                for nested in &block.operations {
                    if marked(nested) {
                        return Err("enum scratch in a nested region is not supported".into());
                    }
                    reject_nested_markers(nested)?;
                }
            }
        }
        Ok(())
    }
    if function.regions.len() != 1 {
        return Err("enum scratch hoisting requires one function body region".into());
    }
    let region = &mut function.regions[0];
    if region.blocks.is_empty() {
        return Ok(()); // External declaration.
    }
    let mut prologue = Vec::new();
    for block in &mut region.blocks {
        let mut body = Vec::new();
        let mut operations = core::mem::take(&mut block.operations).into_iter();
        while let Some(mut count) = operations.next() {
            reject_nested_markers(&count)?;
            if !marked(&count) {
                body.push(count);
                continue;
            }
            let valid_count = count.name == "arith.constant"
                && count.attributes.len() == 1
                && count.attributes.get(ENUM_SCRATCH_COUNT) == Some(&MlirAttribute::Unit)
                && count.operands.is_empty()
                && count.results.len() == 1
                && count.results[0].ty == MlirType::Integer(32)
                && count.properties.get("value")
                    == Some(&MlirAttribute::Integer {
                        value: 1,
                        ty: MlirType::Integer(32),
                    })
                && count.regions.is_empty()
                && count.successors.is_empty();
            if !valid_count {
                return Err("malformed enum scratch count marker".into());
            }
            let mut allocation = operations
                .next()
                .ok_or("enum scratch count has no allocation")?;
            let valid_allocation = allocation.name == "llvm.alloca"
                && allocation.attributes.len() == 1
                && allocation.attributes.get(ENUM_SCRATCH_SLOT) == Some(&MlirAttribute::Unit)
                && allocation.operands
                    == vec![MlirValueUse {
                        id: count.results[0].id,
                        ty: MlirType::Integer(32),
                    }]
                && allocation.results.len() == 1
                && allocation.results[0].ty == MlirType::dialect("!llvm.ptr")?
                && allocation.properties.contains_key("elem_type")
                && allocation.regions.is_empty()
                && allocation.successors.is_empty();
            if !valid_allocation {
                return Err("malformed enum scratch allocation or count dependency".into());
            }
            count.attributes.remove(ENUM_SCRATCH_COUNT);
            allocation.attributes.remove(ENUM_SCRATCH_SLOT);
            prologue.extend([count, allocation]);
        }
        block.operations = body;
    }
    prologue.append(&mut region.blocks[0].operations);
    region.blocks[0].operations = prologue;
    Ok(())
}

/// Allocate one private, aligned enum scratch slot and mark it for entry-block hoisting.
pub(super) fn slot(
    ty: MlirType,
    align: u64,
    s: &mut TranslationSession<'_>,
    loc: &MlirLocation,
    out: &mut Vec<MlirOperation>,
) -> Result<MlirValueUse, String> {
    let one = integer(1, 32, s, loc, out)?;
    out.last_mut()
        .expect("integer appends its constant")
        .attributes
        .insert(ENUM_SCRATCH_COUNT.into(), MlirAttribute::Unit);
    let (r, p) = fresh(s, MlirType::dialect("!llvm.ptr")?);
    let mut a = op("llvm.alloca", vec![one], vec![r], loc)?;
    a.attributes
        .insert(ENUM_SCRATCH_SLOT.into(), MlirAttribute::Unit);
    a.properties
        .insert("elem_type".into(), MlirAttribute::Type(ty));
    a.properties.insert(
        "alignment".into(),
        MlirAttribute::Integer {
            value: align.into(),
            ty: MlirType::Integer(64),
        },
    );
    out.push(a);
    Ok(p)
}
/// Offset an enum scratch pointer by its rustc-provided byte position.
fn byte_pointer(
    base: MlirValueUse,
    offset: u64,
    s: &mut TranslationSession<'_>,
    loc: &MlirLocation,
    out: &mut Vec<MlirOperation>,
) -> Result<MlirValueUse, String> {
    let offset = i32::try_from(offset).map_err(|_| "enum offset exceeds supported GEP range")?;
    let (r, p) = fresh(s, base.ty.clone());
    let mut gep = op("llvm.getelementptr", vec![base], vec![r], loc)?;
    gep.properties.insert(
        "elem_type".into(),
        MlirAttribute::Type(MlirType::Integer(8)),
    );
    gep.properties.insert(
        "rawConstantIndices".into(),
        MlirAttribute::DenseI32Array(vec![offset]),
    );
    gep.properties.insert(
        "noWrapFlags".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(32),
        },
    );
    out.push(gep);
    Ok(p)
}
/// Write one enum tag or payload to potentially unaligned byte storage.
pub(super) fn store(
    value: MlirValueUse,
    pointer: MlirValueUse,
    loc: &MlirLocation,
    out: &mut Vec<MlirOperation>,
) -> Result<(), String> {
    let mut st = op("llvm.store", vec![value, pointer], vec![], loc)?;
    // Variant payloads may be packed or overlap differently typed variants.
    st.properties.insert(
        "alignment".into(),
        MlirAttribute::Integer {
            value: 1,
            ty: MlirType::Integer(64),
        },
    );
    out.push(st);
    Ok(())
}
/// Read one enum tag or payload from potentially unaligned byte storage.
pub(super) fn load(
    pointer: MlirValueUse,
    result: MlirResult,
    loc: &MlirLocation,
    out: &mut Vec<MlirOperation>,
) -> Result<(), String> {
    let mut ld = op("llvm.load", vec![pointer], vec![result], loc)?;
    ld.properties.insert(
        "alignment".into(),
        MlirAttribute::Integer {
            value: 1,
            ty: MlirType::Integer(64),
        },
    );
    out.push(ld);
    Ok(())
}
/// Select physical enum construction, tag extraction, or payload extraction.
enum EnumOp {
    /// Initialize the active variant's tag and payload at their physical byte offsets.
    Construct,
    /// Read the variant tag from its rustc-provided storage position.
    Discriminant,
    /// Read a selected variant payload field from physical enum storage.
    Payload,
}
impl OperationTranslation for EnumOp {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        s: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if input.results.len() != 1
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("invalid enum operation shape".into());
        }
        let source_ty = if matches!(self, Self::Construct) {
            source.deref(ctx).get_result(0).get_type(ctx)
        } else {
            source.deref(ctx).get_operand(0).get_type(ctx)
        };
        let e = source_ty
            .deref(ctx)
            .downcast_ref::<MirEnumType>()
            .ok_or("expected enum type")?
            .clone();
        let storage_ty = enum_storage_type(&e)?;
        if e.layout_kind == EnumLayoutKind::Empty {
            return Err("cannot construct or inspect an uninhabited enum".into());
        }
        let loc = &input.location;
        let result = input.results[0].clone();
        let mut out = vec![];
        if matches!(self, Self::Discriminant) && e.layout_kind == EnumLayoutKind::Single {
            let variant = e.single_variant as usize;
            if e.variant_is_inhabited(variant) != Some(true) {
                return Err("cannot read uninhabited enum discriminant".into());
            }
            let bits = *e
                .variant_discriminants
                .get(variant)
                .ok_or("missing single-variant discriminant")?;
            let mut c = op("arith.constant", vec![], vec![result.clone()], loc)?;
            c.properties.insert(
                "value".into(),
                MlirAttribute::Integer {
                    value: bits.into(),
                    ty: result.ty,
                },
            );
            return Ok(vec![c]);
        }
        let p = slot(storage_ty, e.abi_align, s, loc, &mut out)?;
        match self {
            Self::Construct => {
                let variant = MirConstructEnumOp::new(source)
                    .get_attr_construct_enum_variant_index(ctx)
                    .ok_or("missing variant")?
                    .0 as usize;
                if e.variant_is_inhabited(variant) != Some(true) {
                    return Err("cannot construct uninhabited enum variant".into());
                }
                let count = *e
                    .variant_field_counts
                    .get(variant)
                    .ok_or("invalid variant")? as usize;
                if count != input.operands.len() {
                    return Err("enum payload arity mismatch".into());
                }
                if e.layout_kind == EnumLayoutKind::Direct {
                    let tag = integer(
                        e.variant_discriminants[variant].into(),
                        e.carrier_width,
                        s,
                        loc,
                        &mut out,
                    )?;
                    let ptr = byte_pointer(p.clone(), e.tag_offset, s, loc, &mut out)?;
                    store(tag, ptr, loc, &mut out)?;
                } else if variant != e.single_variant as usize {
                    return Err("construction disagrees with single variant layout".into());
                }
                let start = e.variant_field_counts[..variant]
                    .iter()
                    .map(|v| *v as usize)
                    .sum::<usize>();
                for (i, value) in input.operands.into_iter().enumerate() {
                    let size =
                        crate::mir::memory::source_stored_size(ctx, e.all_field_types[start + i])
                            .ok_or("unknown enum field storage")?;
                    if size != e.all_field_sizes[start + i] {
                        return Err("enum payload storage size differs from rustc".into());
                    }
                    if size == 0 {
                        continue;
                    }
                    let ptr =
                        byte_pointer(p.clone(), e.all_field_offsets[start + i], s, loc, &mut out)?;
                    store(value, ptr, loc, &mut out)?;
                }
                load(p, result, loc, &mut out)?;
            }
            Self::Discriminant => {
                store(input.operands[0].clone(), p.clone(), loc, &mut out)?;
                if result.ty != MlirType::Integer(e.carrier_width) {
                    return Err("direct enum tag/result widths differ".into());
                }
                let ptr = byte_pointer(p, e.tag_offset, s, loc, &mut out)?;
                load(ptr, result, loc, &mut out)?;
            }
            Self::Payload => {
                let payload = MirEnumPayloadOp::new(source);
                let variant = payload
                    .get_attr_payload_variant_index(ctx)
                    .ok_or("missing payload variant")?
                    .0 as usize;
                let field = payload
                    .get_attr_payload_field_index(ctx)
                    .ok_or("missing payload field")?
                    .0 as usize;
                if field
                    >= *e
                        .variant_field_counts
                        .get(variant)
                        .ok_or("invalid payload variant")? as usize
                {
                    return Err("invalid payload field".into());
                }
                let flat = e.variant_field_counts[..variant]
                    .iter()
                    .map(|v| *v as usize)
                    .sum::<usize>()
                    + field;
                if crate::mir::memory::source_stored_size(ctx, e.all_field_types[flat])
                    != Some(e.all_field_sizes[flat])
                {
                    return Err("enum payload storage size differs from rustc".into());
                }
                if e.all_field_sizes[flat] == 0 {
                    out.push(op("llvm.mlir.poison", vec![], vec![result], loc)?);
                } else {
                    store(input.operands[0].clone(), p.clone(), loc, &mut out)?;
                    let ptr = byte_pointer(p, e.all_field_offsets[flat], s, loc, &mut out)?;
                    load(ptr, result, loc, &mut out)?;
                }
            }
        }
        Ok(out)
    }
}

/// Emit signed or unsigned overflow arithmetic and rebuild the Rust result tuple.
struct Checked(
    #[doc = "Arithmetic family, currently add or mul, used to select an overflow intrinsic."]
    &'static str,
);
impl OperationTranslation for Checked {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        s: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if input.operands.len() != 2
            || input.results.len() != 1
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("invalid checked arithmetic shape".into());
        }
        let ty = source.deref(ctx).get_operand(0).get_type(ctx);
        let ty_ref = ty.deref(ctx);
        let integer = ty_ref
            .downcast_ref::<IntegerType>()
            .ok_or("checked arithmetic needs integers")?;
        let width = integer.width();
        let sign = if integer.signedness() == Signedness::Signed {
            "s"
        } else {
            "u"
        };
        let result_ty = source.deref(ctx).get_result(0).get_type(ctx);
        let result_ref = result_ty.deref(ctx);
        let tuple = result_ref
            .downcast_ref::<MirTupleType>()
            .ok_or("checked arithmetic needs a tuple result")?;
        if tuple.get_types().len() != 2
            || tuple.get_types()[0] != ty
            || s.translate_type(tuple.get_types()[1])
                .map_err(|e| e.to_string())?
                != MlirType::Integer(1)
        {
            return Err("checked tuple must contain value and overflow flag".into());
        }
        let (r, v) = fresh(
            s,
            MlirType::dialect(format!("!llvm.struct<(i{width}, i1)>"))?,
        );
        let mut call = op(
            "llvm.call_intrinsic",
            input.operands.clone(),
            vec![r],
            &input.location,
        )?;
        call.properties.insert(
            "intrin".into(),
            MlirAttribute::String(format!("llvm.{sign}{}.with.overflow.i{width}", self.0)),
        );
        call.properties.insert(
            "op_bundle_sizes".into(),
            MlirAttribute::DenseI32Array(vec![]),
        );
        call.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![2, 0]),
        );
        let mut out = vec![call];
        let mut fields = vec![];
        for (i, ty) in [MlirType::Integer(width), MlirType::Integer(1)]
            .into_iter()
            .enumerate()
        {
            let (r, field) = fresh(s, ty);
            let mut extract = op(
                "llvm.extractvalue",
                vec![v.clone()],
                vec![r],
                &input.location,
            )?;
            extract.properties.insert(
                "position".into(),
                MlirAttribute::DenseI64Array(vec![i as i64]),
            );
            out.push(extract);
            fields.push(field);
        }
        input.operands = fields;
        out.extend(
            crate::mir::memory::ConstructStructTranslation.translate(ctx, source, input, s)?,
        );
        Ok(out)
    }
}
/// Trap before marking control flow unreachable at the device contract boundary.
struct Unreachable;
impl OperationTranslation for Unreachable {
    fn translate(
        &self,
        _: &Context,
        _: Ptr<Operation>,
        input: OperationInput,
        _: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if !input.operands.is_empty()
            || !input.results.is_empty()
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("invalid unreachable shape".into());
        }
        Ok(vec![op(
            "llvm.unreachable",
            vec![],
            vec![],
            &input.location,
        )?])
    }
}

#[cfg(test)]
mod tests {
    //! Enum byte-layout and scratch-hoisting regressions, including conditional and loop-local accesses.

    use super::*;
    use dialect_mir::types::{EnumEncoding, EnumVariant};
    /// Construct the explicit Direct enum layout used by byte-storage rejection tests.
    fn direct(ctx: &mut Context) -> MirEnumType {
        let tag = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let word = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        MirEnumType::get_with_encoding(
            ctx,
            "ResultWord".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Ok".into(), vec![word], vec![8], vec![8]),
                EnumVariant::new_with_layout("Err".into(), vec![tag], vec![1], vec![1]),
            ],
            EnumEncoding {
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 8,
                variant_inhabited: vec![1, 1],
                ..Default::default()
            },
        )
        .deref(ctx)
        .clone()
    }
    /// Reject nested bool or named-space pointer payloads until their byte-storage adapters are supported.
    #[test]
    fn enum_payloads_reject_unreviewed_byte_and_pointer_adapters_recursively() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, boolean, 4).into();
        assert!(
            check_payload_storage(&ctx, array)
                .unwrap_err()
                .contains("bool")
        );
        let pointer: TypeHandle = MirPtrType::get(&mut ctx, boolean, true, 3).into();
        let tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![pointer]).into();
        assert!(
            check_payload_storage(&ctx, tuple)
                .unwrap_err()
                .contains("named-space")
        );
        let generic: TypeHandle = MirPtrType::get(&mut ctx, boolean, true, 0).into();
        check_payload_storage(&ctx, generic).unwrap(); // pointee is not stored inline
    }
    /// Check inactive payload and padding bytes remain independent of the initialized enum tag.
    #[test]
    fn enum_storage_keeps_padding_independent_of_initialized_tag_bytes() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let e = direct(&mut ctx);
        assert_eq!(
            enum_storage_type(&e).unwrap(),
            MlirType::dialect("!llvm.struct<(array<16 x i8>, array<0 x i64>)>").unwrap()
        );
        let mut single = e.clone();
        single.layout_kind = EnumLayoutKind::Single;
        single.carrier_kind = EnumCarrierKind::None;
        single.carrier_width = 0;
        assert_eq!(
            enum_storage_type(&single).unwrap(),
            enum_storage_type(&e).unwrap()
        );
        for kind in [
            EnumLayoutKind::Niche,
            EnumLayoutKind::Unknown,
            EnumLayoutKind::Empty,
        ] {
            let mut invalid = e.clone();
            invalid.layout_kind = kind;
            assert!(enum_storage_type(&invalid).is_err());
        }
        let mut invalid = e.clone();
        invalid.tag_offset = 16;
        assert!(enum_storage_type(&invalid).is_err());
        let mut invalid = e.clone();
        invalid.all_field_offsets[0] = 12;
        assert!(enum_storage_type(&invalid).is_err());
    }
    /// Check scratch allocations move to entry while conditional construction and loop projections remain in place.
    #[test]
    fn conditional_enum_construct_and_loop_projections_use_entry_scratch() {
        use crate::{CutlassPrimitivesMlir22, MlirConsumerProfile};
        use dialect_mir::attributes::FieldIndexAttr;
        use pliron::{
            basic_block::BasicBlock,
            builtin::{
                attributes::TypeAttr,
                op_interfaces::{
                    OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface,
                },
                ops::ModuleOp,
                types::FunctionType,
            },
            op::Op,
            value::Value,
        };
        /// Append a typed source operation to the fixture block and return its operation handle.
        fn append<O: Op>(
            ctx: &mut Context,
            block: Ptr<BasicBlock>,
            types: Vec<TypeHandle>,
            values: Vec<Value>,
        ) -> Ptr<Operation> {
            let op = Operation::new(ctx, O::get_concrete_op_info(), types, values, vec![], 0);
            op.insert_at_back(block, ctx);
            op
        }
        /// Create a conditional source branch with explicit destination blocks and no successor operands.
        fn conditional(
            ctx: &mut Context,
            block: Ptr<BasicBlock>,
            flag: Value,
            left: Ptr<BasicBlock>,
            right: Ptr<BasicBlock>,
            lhs: Vec<Value>,
            rhs: Vec<Value>,
        ) {
            let (operands, segments) =
                MirCondBranchOp::compute_segment_sizes(vec![vec![flag], lhs, rhs]);
            let op = Operation::new(
                ctx,
                MirCondBranchOp::get_concrete_op_info(),
                vec![],
                operands,
                vec![left, right],
                0,
            );
            MirCondBranchOp::new(op).set_operand_segment_sizes(ctx, segments);
            op.insert_at_back(block, ctx);
        }
        /// Find the translated function recursively through the module envelope.
        fn find_function(op: &MlirOperation) -> Option<&MlirOperation> {
            if op.name == "func.func" {
                return Some(op);
            }
            op.regions
                .iter()
                .flat_map(|r| &r.blocks)
                .flat_map(|b| &b.operations)
                .find_map(find_function)
        }
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let flag: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enumeration: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Result".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Ok".into(), vec![word], vec![8], vec![8]),
                EnumVariant::new_with_layout("Err".into(), vec![tag], vec![1], vec![1]),
            ],
            0,
            16,
            8,
        )
        .into();
        let module = ModuleOp::new(&mut ctx, "enum_loop".try_into().unwrap());
        let ty = FunctionType::get(&ctx, vec![flag, word], vec![word]);
        let f = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, f, TypeAttr::new(ty.into()))
            .set_symbol_name(&mut ctx, "test".try_into().unwrap());
        module.append_operation(&mut ctx, f, 0);
        let region = f.deref(&ctx).get_region(0);
        let entry = BasicBlock::new(&mut ctx, None, vec![flag, word]);
        let construct = BasicBlock::new(&mut ctx, None, vec![]);
        let loop_body = BasicBlock::new(&mut ctx, None, vec![enumeration]);
        let exit = BasicBlock::new(&mut ctx, None, vec![word]);
        for block in [entry, construct, loop_body, exit] {
            block.insert_at_back(region, &ctx);
        }
        let condition = entry.deref(&ctx).get_argument(0);
        let input = entry.deref(&ctx).get_argument(1);
        conditional(
            &mut ctx,
            entry,
            condition,
            construct,
            exit,
            vec![],
            vec![input],
        );
        let c = append::<MirConstructEnumOp>(&mut ctx, construct, vec![enumeration], vec![input]);
        MirConstructEnumOp::new(c).set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        let value = c.deref(&ctx).get_result(0);
        let branch = Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![loop_body],
            0,
        );
        branch.insert_at_back(construct, &ctx);
        let carried = loop_body.deref(&ctx).get_argument(0);
        append::<MirGetDiscriminantOp>(&mut ctx, loop_body, vec![tag], vec![carried]);
        let payload = append::<MirEnumPayloadOp>(&mut ctx, loop_body, vec![word], vec![carried]);
        let p = MirEnumPayloadOp::new(payload);
        p.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
        p.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        let value = payload.deref(&ctx).get_result(0);
        conditional(
            &mut ctx,
            loop_body,
            condition,
            loop_body,
            exit,
            vec![carried],
            vec![value],
        );
        let result = exit.deref(&ctx).get_argument(0);
        append::<MirReturnOp>(&mut ctx, exit, vec![], vec![result]);
        let translated = CutlassPrimitivesMlir22::new("sm_100a")
            .unwrap()
            .translate_module(&ctx, &module)
            .unwrap();
        let f = find_function(&translated.root).unwrap();
        let blocks = &f.regions[0].blocks;
        let allocas: Vec<_> = blocks[0]
            .operations
            .iter()
            .filter(|o| o.name == "llvm.alloca")
            .collect();
        assert_eq!(
            allocas.len(),
            3,
            "all three enum operations need distinct entry scratch"
        );
        let ids: std::collections::HashSet<_> = allocas.iter().map(|o| o.results[0].id).collect();
        assert_eq!(ids.len(), 3);
        assert!(
            !blocks[0]
                .operations
                .iter()
                .any(|o| o.name == "llvm.store" || o.name == "llvm.load")
        );
        assert_eq!(
            blocks[1]
                .operations
                .iter()
                .filter(|o| o.name == "llvm.store")
                .count(),
            2
        );
        assert_eq!(
            blocks[2]
                .operations
                .iter()
                .filter(|o| o.name == "llvm.store")
                .count(),
            2
        );
        assert!(
            blocks[1..]
                .iter()
                .flat_map(|b| &b.operations)
                .all(|o| o.name != "llvm.alloca")
        );
        assert_eq!(blocks[0].operations.last().unwrap().name, "cf.cond_br");
        assert_eq!(
            blocks[2].operations.last().unwrap().successors,
            vec![blocks[2].id, blocks[3].id]
        );
        let text = crate::render_module(&translated).unwrap();
        assert!(!text.contains(ENUM_SCRATCH_COUNT) && !text.contains(ENUM_SCRATCH_SLOT));
    }

    /// Build a small target function with marked enum scratch for hoisting validation.
    fn scratch_fixture() -> MlirOperation {
        use crate::{MlirBlock, MlirBlockId, MlirRegion, MlirValueId};
        let mut count = MlirOperation::new("arith.constant").unwrap();
        count.results.push(MlirResult {
            id: MlirValueId(0),
            ty: MlirType::Integer(32),
        });
        count.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 1,
                ty: MlirType::Integer(32),
            },
        );
        count
            .attributes
            .insert(ENUM_SCRATCH_COUNT.into(), MlirAttribute::Unit);
        let mut allocation = MlirOperation::new("llvm.alloca").unwrap();
        allocation.results.push(MlirResult {
            id: MlirValueId(1),
            ty: MlirType::dialect("!llvm.ptr").unwrap(),
        });
        allocation.operands.push(MlirValueUse {
            id: MlirValueId(0),
            ty: MlirType::Integer(32),
        });
        allocation.properties.insert(
            "elem_type".into(),
            MlirAttribute::Type(MlirType::Integer(64)),
        );
        allocation
            .attributes
            .insert(ENUM_SCRATCH_SLOT.into(), MlirAttribute::Unit);
        let mut f = MlirOperation::new("func.func").unwrap();
        f.regions.push(MlirRegion {
            blocks: vec![
                MlirBlock {
                    id: MlirBlockId(0),
                    arguments: vec![],
                    operations: vec![MlirOperation::new("cf.br").unwrap()],
                },
                MlirBlock {
                    id: MlirBlockId(1),
                    arguments: vec![],
                    operations: vec![count, allocation],
                },
            ],
        });
        f
    }

    /// Check the hoister moves only its explicitly marked constant-size scratch allocations.
    #[test]
    fn enum_scratch_hoist_leaves_unmarked_and_dynamic_allocations_in_place() {
        let mut f = scratch_fixture();
        let mut ordinary = f.regions[0].blocks[1].operations[1].clone();
        ordinary.attributes.clear();
        ordinary.operands[0].id = crate::MlirValueId(99); // Dynamic count.
        f.regions[0].blocks[1].operations.push(ordinary.clone());
        hoist_enum_scratch(&mut f).unwrap();
        assert_eq!(f.regions[0].blocks[1].operations, vec![ordinary]);
        assert_eq!(f.regions[0].blocks[0].operations.len(), 3);
    }

    /// Reject malformed scratch markers or allocation dependencies that cannot safely move to entry.
    #[test]
    fn enum_scratch_hoist_rejects_malformed_markers_and_dependencies() {
        for case in 0..5 {
            let mut f = scratch_fixture();
            let operations = &mut f.regions[0].blocks[1].operations;
            match case {
                0 => {
                    operations[0].properties.insert(
                        "value".into(),
                        MlirAttribute::Integer {
                            value: 2,
                            ty: MlirType::Integer(32),
                        },
                    );
                }
                1 => {
                    operations[1].operands[0].id = crate::MlirValueId(99);
                }
                2 => {
                    operations[1].attributes.clear();
                }
                3 => {
                    operations.remove(0);
                }
                4 => {
                    operations[0].attributes.insert(
                        ENUM_SCRATCH_COUNT.into(),
                        MlirAttribute::String("true".into()),
                    );
                }
                _ => unreachable!(),
            }
            assert!(hoist_enum_scratch(&mut f).is_err(), "case {case}");
        }
        let mut f = scratch_fixture();
        let mut nested = MlirOperation::new("func.func").unwrap();
        nested.regions = vec![f.regions[0].clone()];
        f.regions[0].blocks[1].operations = vec![nested];
        assert!(
            hoist_enum_scratch(&mut f)
                .unwrap_err()
                .contains("nested region")
        );
    }
}
