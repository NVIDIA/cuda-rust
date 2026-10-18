/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared static handles, MIR adapters, and scalar operation builders.

use crate::{
    MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse, OperationInput,
    OperationTranslation, TranslationRegistry, TranslationSession, TypeTranslation,
};
use dialect_mir::{
    ops::{MirExternSharedOp, MirInsertFieldOp},
    types::MirUnionType,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Represent source-only verified layout contracts with an empty target tuple.
pub(in crate::cute) struct StaticContractTypeTranslation;

impl TypeTranslation for StaticContractTypeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        Ok(MlirType::Tuple(vec![]))
    }
}

/// Map MIR field insertion to llvm.insertvalue at its verified field index.
pub(in crate::cute) struct InsertFieldTranslation;

impl OperationTranslation for InsertFieldTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = MirInsertFieldOp::new(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("mir.insert_field", &input, 1, 2)?;
        let index = semantic
            .get_attr_insert_index(ctx)
            .ok_or_else(|| "mir.insert_field has no insert_index".to_owned())?
            .0 as i64;
        if source
            .deref(ctx)
            .get_operand(0)
            .get_type(ctx)
            .deref(ctx)
            .is::<MirUnionType>()
        {
            return crate::mir::storage::insert_union_field(
                ctx,
                source,
                input,
                session,
                index as usize,
            );
        }
        let mut target = operation(
            "llvm.insertvalue",
            input.results,
            input.operands,
            &input.location,
        )?;
        target
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![index]));
        Ok(vec![target])
    }
}

/// Obtain dynamic shared memory and apply the source byte offset and alignment.
pub(in crate::cute) struct ExternSharedTranslation;

impl OperationTranslation for ExternSharedTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = MirExternSharedOp::new(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("extern_byte_offset");
        input.attributes.remove("extern_alignment");
        expect_shape("mir.extern_shared", &input, 1, 0)?;
        let offset = semantic.get_byte_offset_value(ctx);
        let alignment = semantic.get_alignment_value(ctx);
        let cute_pointer = MlirType::dialect(format!("!cute.ptr<i8, smem, align<{alignment}>>"))?;
        let (raw_result, raw) = fresh(session, cute_pointer);
        let raw_op = operation(
            "cute_nvgpu.arch.get_dyn_smem",
            vec![raw_result],
            vec![],
            &input.location,
        )?;
        let (base_result, base) = if offset == 0 {
            (
                input.results[0].clone(),
                MlirValueUse {
                    id: input.results[0].id,
                    ty: input.results[0].ty.clone(),
                },
            )
        } else {
            fresh(session, shared_pointer_type()?)
        };
        let bridge = operation(
            "builtin.unrealized_conversion_cast",
            vec![base_result],
            vec![raw],
            &input.location,
        )?;
        if offset == 0 {
            return Ok(vec![raw_op, bridge]);
        }

        let index = constant_value(
            session,
            MlirType::Integer(64),
            offset as i128,
            &input.location,
        )?;
        let mut gep = operation(
            "llvm.getelementptr",
            input.results,
            vec![base, index.1],
            &input.location,
        )?;
        gep_properties(&mut gep, MlirType::Integer(8), vec![i32::MIN]);
        Ok(vec![raw_op, bridge, index.0, gep])
    }
}

/// Preserve a one-input static handle through an unrealized conversion cast.
pub(in crate::cute) struct IdentityTranslation(
    #[doc = "Source operation spelling used when its one-to-one contract is violated."]
    pub(in crate::cute) &'static str,
);

impl OperationTranslation for IdentityTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape(self.0, &input, 1, 1)?;
        Ok(vec![operation(
            "builtin.unrealized_conversion_cast",
            input.results,
            input.operands,
            &input.location,
        )?])
    }
}

/// Preserve the first runtime operand while erasing verified static operands.
pub(in crate::cute) struct IdentityFirstOperandTranslation(
    #[doc = "Source operation spelling used when its first-operand forwarding contract is violated."]
    pub(in crate::cute) &'static str,
);

impl OperationTranslation for IdentityFirstOperandTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if input.results.len() != 1 || input.operands.is_empty() {
            return Err(format!(
                "{} needs one result and at least one operand",
                self.0
            ));
        }
        if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty()
        {
            return Err(format!("{} retained source-only structure", self.0));
        }
        Ok(vec![operation(
            "builtin.unrealized_conversion_cast",
            input.results,
            vec![input.operands[0].clone()],
            &input.location,
        )?])
    }
}

// The remaining definitions in this file are kept below the scalar foundation
// so each official-parser rung can be tested independently.

/// Erase listed static attributes and forward each operand to its corresponding result.
pub(in crate::cute) struct StaticMultiIdentityTranslation {
    /// Source operation spelling used in contract diagnostics.
    pub(in crate::cute) name: &'static str,
    /// Source-only attributes consumed before forwarding runtime operands.
    pub(in crate::cute) attributes: &'static [&'static str],
}

impl OperationTranslation for StaticMultiIdentityTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        for attribute in self.attributes {
            input.attributes.remove(*attribute);
        }
        if input.results.len() != input.operands.len() {
            return Err(format!(
                "{} expected matching result and operand counts",
                self.name
            ));
        }
        expect_shape(self.name, &input, input.results.len(), input.operands.len())?;
        input
            .results
            .into_iter()
            .zip(input.operands)
            .map(|(result, operand)| {
                operation(
                    "builtin.unrealized_conversion_cast",
                    vec![result],
                    vec![operand],
                    &input.location,
                )
            })
            .collect()
    }
}

/// Append the requested integer conversion and return its typed result.
pub(in crate::cute) fn cast_int(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    target: MlirType,
    name: &str,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, target);
    operations.push(operation(name, vec![result], vec![value], location)?);
    Ok(use_)
}

/// Append an LLVM pointer offset measured in elements of the supplied type.
pub(in crate::cute) fn gep_value(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    index: MlirValueUse,
    element: MlirType,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, base.ty.clone());
    let mut gep = operation(
        "llvm.getelementptr",
        vec![result],
        vec![base, index],
        location,
    )?;
    gep_properties(&mut gep, element, vec![i32::MIN]);
    operations.push(gep);
    Ok(use_)
}

/// Set GEP element and index metadata without promising no-wrap behavior.
pub(in crate::cute) fn gep_properties(
    operation: &mut MlirOperation,
    element: MlirType,
    indices: Vec<i32>,
) {
    operation
        .properties
        .insert("elem_type".into(), MlirAttribute::Type(element));
    operation.properties.insert(
        "noWrapFlags".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(32),
        },
    );
    operation.properties.insert(
        "rawConstantIndices".into(),
        MlirAttribute::DenseI32Array(indices),
    );
}

/// Append an i64 constant while preserving the unsigned input's bit pattern.
pub(in crate::cute) fn emit_i64_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (operation_, use_) =
        constant_value(session, MlirType::Integer(64), u64_bits(value), location)?;
    operations.push(operation_);
    Ok(use_)
}

/// Append an i32 constant and return its SSA operand handle.
pub(in crate::cute) fn emit_i32_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (operation_, use_) = constant_value(session, MlirType::Integer(32), value, location)?;
    operations.push(operation_);
    Ok(use_)
}

/// Allocate a typed integer constant and return its operation and SSA use.
pub(in crate::cute) fn constant_value(
    session: &mut TranslationSession<'_>,
    ty: MlirType,
    value: i128,
    location: &MlirLocation,
) -> Result<(MlirOperation, MlirValueUse), String> {
    let (result, use_) = fresh(session, ty);
    Ok((integer_constant(result, value, location)?, use_))
}

/// Append a two-operand operation whose result has the left operand's type.
pub(in crate::cute) fn emit_binary(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    name: &str,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, left.ty.clone());
    operations.push(binary(name, result, left, right, location)?);
    Ok(use_)
}

/// Build a two-operand target operation using the supplied result.
pub(in crate::cute) fn binary(
    name: &str,
    result: MlirResult,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(name, vec![result], vec![left, right], location)
}

/// Build an arith.constant using the supplied result's integer type.
pub(in crate::cute) fn integer_constant(
    result: MlirResult,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("arith.constant", vec![result.clone()], vec![], location)?;
    operation.properties.insert(
        "value".into(),
        MlirAttribute::Integer {
            value,
            ty: result.ty,
        },
    );
    Ok(operation)
}

/// Require exact result/operand counts and no leftover attributes, regions, or successors.
pub(in crate::cute) fn expect_shape(
    name: &str,
    input: &OperationInput,
    results: usize,
    operands: usize,
) -> Result<(), String> {
    if input.results.len() != results || input.operands.len() != operands {
        return Err(format!(
            "{name} expected {results} results and {operands} operands, got {} and {}",
            input.results.len(),
            input.operands.len()
        ));
    }
    if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty() {
        return Err(format!("{name} retained source-only target structure"));
    }
    Ok(())
}

/// Build a target operation with the supplied operands, results, and source location.
pub(in crate::cute) fn operation(
    name: &str,
    results: Vec<MlirResult>,
    operands: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.results = results;
    operation.operands = operands;
    operation.location = location.clone();
    Ok(operation)
}

/// Allocate one SSA identifier and return matching result and operand handles.
pub(in crate::cute) fn fresh(
    session: &mut TranslationSession<'_>,
    ty: MlirType,
) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

/// Return the opaque LLVM pointer type for generic address space zero.
pub(in crate::cute) fn generic_pointer_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.ptr")
}

/// Return the opaque LLVM pointer type for CTA shared address space three.
pub(in crate::cute) fn shared_pointer_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.ptr<3>")
}

/// Preserve all 64 bits while spelling an unsigned value as a signed MLIR literal.
pub(in crate::cute) fn u64_bits(value: u64) -> i128 {
    i64::from_ne_bytes(value.to_ne_bytes()) as i128
}

/// Preserve a source diagnostic as the string expected by translation recipes.
pub(in crate::cute) fn stringify(error: impl std::fmt::Display) -> String {
    error.to_string()
}
