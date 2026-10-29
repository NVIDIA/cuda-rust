/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Register-MMA and shared-memory primitives for SM120, using standard NVVM.
//! Source intrinsic identities and structural selectors are checked before any
//! source-only attributes are removed. No CuTe operations or types are involved.

use crate::{
    DropAttribute, MlirAttribute, MlirFloatType, MlirLocation, MlirOperation, MlirResult, MlirType,
    MlirValueUse, OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};
use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};

/// The exact register-MMA and matrix-load variants used by the supported kernels.
pub fn register_nvvm_sm120_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<LdmatrixShapeAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixMultiplicityAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixElementAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixStateSpaceAttr>(DropAttribute)?;
    registry.register_operation::<LdmatrixOp>(LoadTranslation)?;
    registry.register_operation::<StmatrixM8n8X2Op>(StoreTranslation)?;
    registry.register_operation::<CpAsyncBulkTensorG2sTile2dOp>(TmaTranslation)?;
    registry.register_operation::<CpAsyncBulkTensorG2sCtaTile2dOp>(TmaCtaTranslation)?;
    registry.register_attribute::<RegisterMmaShapeAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaOperationAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaKindAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaAccumulatorAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaElementAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaOverflowAttr>(DropAttribute)?;
    registry.register_operation::<RegisterMmaOp>(MmaTranslation)?;
    Ok(())
}

/// Reorder reviewed register MMA operands and preserve packed lanes and scale selectors.
struct MmaTranslation;
impl OperationTranslation for MmaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mma = RegisterMmaOp::new(source);
        mma.verify(ctx).map_err(|e| e.to_string())?;
        // Both formats use packed u32 source carriers, but NVVM deliberately
        // consumes BF16 pairs as i32 and FP16 pairs as vector<2xf16>.
        let a_element = mma.get_attr_nvvm_register_mma_a_element(ctx);
        let b_element = mma.get_attr_nvvm_register_mma_b_element(ctx);
        let is_bf16 = a_element.as_deref() == Some(&RegisterMmaElementAttr::Bf16);
        let matching_half_operands = a_element.as_deref() == b_element.as_deref()
            && matches!(
                a_element.as_deref(),
                Some(RegisterMmaElementAttr::F16 | RegisterMmaElementAttr::Bf16)
            );
        if mma.get_attr_nvvm_register_mma_shape(ctx).as_deref()
            == Some(&RegisterMmaShapeAttr::M16n8k16)
            && mma.get_attr_nvvm_register_mma_kind(ctx).as_deref()
                == Some(&RegisterMmaKindAttr::Standard)
            && mma.get_attr_nvvm_register_mma_operation(ctx).as_deref()
                == Some(&RegisterMmaOperationAttr::Multiply)
            && mma.get_attr_nvvm_register_mma_accumulator(ctx).as_deref()
                == Some(&RegisterMmaAccumulatorAttr::F32)
            && matching_half_operands
            && mma.get_attr_nvvm_register_mma_a_layout(ctx).as_deref()
                == Some(&RegisterMmaLayoutAttr::Row)
            && mma.get_attr_nvvm_register_mma_b_layout(ctx).as_deref()
                == Some(&RegisterMmaLayoutAttr::Col)
            && mma.get_attr_nvvm_register_mma_overflow(ctx).as_deref()
                == Some(&RegisterMmaOverflowAttr::NotApplicable)
        {
            let marker = if is_bf16 { "v1:i0105" } else { "v1:i0106" };
            require_input(&mut input, marker, 10, 4)?;
            let mut operations = vec![];
            let mut operands = vec![];
            // Source operands are C[4], packed A[4], packed B[2]. NVVM requires
            // A[4], B[2], C[4], preserving each packed pair's original bits.
            for packed in &input.operands[4..] {
                if is_bf16 {
                    operands.push(packed.clone());
                    continue;
                }
                let (result, value) = fresh(
                    session,
                    MlirType::Vector {
                        shape: vec![2],
                        element: Box::new(MlirType::Float(MlirFloatType::F16)),
                    },
                );
                operations.push(operation(
                    "llvm.bitcast",
                    vec![packed.clone()],
                    vec![result],
                    &input.location,
                )?);
                operands.push(value);
            }
            operands.extend_from_slice(&input.operands[..4]);
            let (result, aggregate) = fresh(
                session,
                MlirType::dialect("!llvm.struct<(f32, f32, f32, f32)>")?,
            );
            let mut op = operation("nvvm.mma.sync", operands, vec![result], &input.location)?;
            property(&mut op, "shape", "#nvvm.shape<m = 16, n = 8, k = 16>")?;
            property(&mut op, "layoutA", "#nvvm.mma_layout<row>")?;
            property(&mut op, "layoutB", "#nvvm.mma_layout<col>")?;
            let element = if is_bf16 {
                "#nvvm.mma_type<bf16>"
            } else {
                "#nvvm.mma_type<f16>"
            };
            property(&mut op, "multiplicandAPtxType", element)?;
            property(&mut op, "multiplicandBPtxType", element)?;
            op.properties.insert(
                "operandSegmentSizes".into(),
                MlirAttribute::DenseI32Array(vec![4, 2, 4]),
            );
            operations.push(op);
            extract_results(&mut operations, aggregate, input.results, &input.location)?;
            return Ok(operations);
        }
        if mma.get_attr_nvvm_register_mma_shape(ctx).as_deref()
            != Some(&RegisterMmaShapeAttr::M16n8k64)
            || mma.get_attr_nvvm_register_mma_kind(ctx).as_deref()
                != Some(&RegisterMmaKindAttr::Mxf4)
            || mma.get_attr_nvvm_register_mma_operation(ctx).as_deref()
                != Some(&RegisterMmaOperationAttr::Multiply)
            || mma.get_attr_nvvm_register_mma_accumulator(ctx).as_deref()
                != Some(&RegisterMmaAccumulatorAttr::F32)
            || mma.get_attr_nvvm_register_mma_a_element(ctx).as_deref()
                != Some(&RegisterMmaElementAttr::E2m1)
            || mma.get_attr_nvvm_register_mma_b_element(ctx).as_deref()
                != Some(&RegisterMmaElementAttr::E2m1)
            || mma.get_attr_nvvm_register_mma_a_layout(ctx).as_deref()
                != Some(&RegisterMmaLayoutAttr::Row)
            || mma.get_attr_nvvm_register_mma_b_layout(ctx).as_deref()
                != Some(&RegisterMmaLayoutAttr::Col)
            || mma.get_attr_nvvm_register_mma_overflow(ctx).as_deref()
                != Some(&RegisterMmaOverflowAttr::NotApplicable)
        {
            return Err("SM120 register MMA mapping supports m16n8k64 MXF4 row/col E2M1 with F32 accumulation and UE8M0 scale_vec::2X, or m16n8k16 standard row/col matching F16 or BF16 with F32 accumulation only".into());
        }
        require_input(&mut input, "v1:i1039", 16, 4)?;
        // Canonical source order is C, A, B, scales; NVVM uses A, B, C, scales.
        let operands = input.operands[4..10]
            .iter()
            .chain(&input.operands[..4])
            .chain(&input.operands[10..])
            .cloned()
            .collect();
        let (result, aggregate) = fresh(
            session,
            MlirType::dialect("!llvm.struct<(f32, f32, f32, f32)>")?,
        );
        let mut op = operation(
            "nvvm.mma.block_scale",
            operands,
            vec![result],
            &input.location,
        )?;
        property(&mut op, "shape", "#nvvm.shape<m = 16, n = 8, k = 64>")?;
        property(&mut op, "multiplicandAPtxType", "#nvvm.mma_type<e2m1>")?;
        property(&mut op, "multiplicandBPtxType", "#nvvm.mma_type<e2m1>")?;
        property(&mut op, "scaleVecSize", "#nvvm.scale_vec_size<x2>")?;
        property(
            &mut op,
            "blockScaleFormat",
            "#nvvm.block_scale_format<ue8m0>",
        )?;
        property(&mut op, "kind", "#nvvm.block_scale_kind<mxf4>")?;
        op.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![4, 2, 4, 1, 1, 1, 1, 1, 1]),
        );
        let mut operations = vec![op];
        extract_results(&mut operations, aggregate, input.results, &input.location)?;
        Ok(operations)
    }
}

/// Emit the exact matrix-load transpose, shape, packing, and result-register contract.
struct LoadTranslation;
impl OperationTranslation for LoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let load = LdmatrixOp::new(source);
        load.verify(ctx).map_err(|e| e.to_string())?;
        if load.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() != Some(&LdmatrixShapeAttr::M8n8)
            || load.get_attr_nvvm_ldmatrix_element(ctx).as_deref()
                != Some(&LdmatrixElementAttr::B16)
            || load.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref()
                != Some(&LdmatrixStateSpaceAttr::Shared)
        {
            return Err(
                "SM120 matrix load mapping supports m8n8.x2/x4.b16 normal or transposed shared loads only"
                    .into(),
            );
        }
        let (count, marker, layout) = match (
            load.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref(),
            load.get_attr_nvvm_ldmatrix_layout(ctx).as_deref(),
        ) {
            (Some(LdmatrixMultiplicityAttr::X2), Some(LdmatrixLayoutAttr::Normal)) => (2, "v1:i0027", "#nvvm.mma_layout<row>"),
            (Some(LdmatrixMultiplicityAttr::X2), Some(LdmatrixLayoutAttr::Transposed)) => (2, "v1:i0028", "#nvvm.mma_layout<col>"),
            (Some(LdmatrixMultiplicityAttr::X4), Some(LdmatrixLayoutAttr::Normal)) => (4, "v1:i0013", "#nvvm.mma_layout<row>"),
            (Some(LdmatrixMultiplicityAttr::X4), Some(LdmatrixLayoutAttr::Transposed)) => (4, "v1:i0029", "#nvvm.mma_layout<col>"),
            _ => return Err("SM120 matrix load mapping supports m8n8.x2/x4.b16 normal or transposed shared loads only".into()),
        };
        require_input(&mut input, marker, 1, count)?;
        let mut operations = vec![];
        let pointer = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            3,
            &input.location,
        )?;
        let (result, aggregate) = fresh(
            session,
            MlirType::dialect(if count == 2 {
                "!llvm.struct<(i32, i32)>"
            } else {
                "!llvm.struct<(i32, i32, i32, i32)>"
            })?,
        );
        let mut op = operation(
            "nvvm.ldmatrix",
            vec![pointer],
            vec![result],
            &input.location,
        )?;
        property(&mut op, "layout", layout)?;
        op.properties.insert(
            "num".into(),
            MlirAttribute::Integer {
                value: count as i128,
                ty: MlirType::Integer(32),
            },
        );
        operations.push(op);
        extract_results(&mut operations, aggregate, input.results, &input.location)?;
        Ok(operations)
    }
}

/// Emit the verified matrix store without reordering its scalar register operands.
struct StoreTranslation;
impl OperationTranslation for StoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        StmatrixM8n8X2Op::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0301", 3, 0)?;
        let mut operations = vec![];
        input.operands[0] = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            3,
            &input.location,
        )?;
        let mut op = operation("nvvm.stmatrix", input.operands, vec![], &input.location)?;
        property(&mut op, "layout", "#nvvm.mma_layout<row>")?;
        operations.push(op);
        Ok(operations)
    }
}

/// Emit local shared-barrier TMA loads while rejecting hidden multicast or cache-policy changes.
struct TmaTranslation;
impl OperationTranslation for TmaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CpAsyncBulkTensorG2sTile2dOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0329", 7, 0)?;
        if input.operands[..3]
            .iter()
            .any(|operand| matches!(operand.ty, MlirType::Integer(_)))
        {
            return Err(
                "TMA requires pointer carriers for destination, barrier, and descriptor".into(),
            );
        }
        if constant_integer(ctx, source, 5)? != 0 || constant_integer(ctx, source, 6)? != 0 {
            return Err(
                "ordinary 2D TMA requires the importer's zero multicast mask and cache hint".into(),
            );
        }
        for index in [3, 4] {
            if input.operands[index].ty != MlirType::Integer(32) {
                return Err("2D TMA coordinates must be i32".into());
            }
        }
        if input.operands[5].ty != MlirType::Integer(16)
            || input.operands[6].ty != MlirType::Integer(64)
        {
            return Err("2D TMA mask/cache carriers must be i16/i64".into());
        }
        let mut operations = vec![];
        // The canonical LLVM g2s intrinsic encodes the destination in p7,
        // including for a non-multicast copy into the executing CTA's SMEM.
        let dst = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            7,
            &input.location,
        )?;
        let barrier = pointer(
            session,
            &mut operations,
            input.operands[1].clone(),
            3,
            &input.location,
        )?;
        let descriptor = pointer(
            session,
            &mut operations,
            input.operands[2].clone(),
            0,
            &input.location,
        )?;
        let mut op = operation(
            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
            vec![
                dst,
                descriptor,
                input.operands[3].clone(),
                input.operands[4].clone(),
                barrier,
            ],
            vec![],
            &input.location,
        )?;
        op.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 0, 0, 0]),
        );
        op.properties
            .insert("useIntrinsic".into(), MlirAttribute::Bool(false));
        operations.push(op);
        Ok(operations)
    }
}

/// Lower a CTA-local 2D TMA load (`cp.async.bulk.tensor.2d.shared::cta.global.tile`).
struct TmaCtaTranslation;
impl OperationTranslation for TmaCtaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CpAsyncBulkTensorG2sCtaTile2dOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i1035", 6, 0)?;
        if input.operands[..3]
            .iter()
            .any(|operand| matches!(operand.ty, MlirType::Integer(_)))
        {
            return Err(
                "TMA requires pointer carriers for destination, barrier, and descriptor".into(),
            );
        }
        if constant_integer(ctx, source, 5)? != 0 {
            return Err("ordinary 2D CTA TMA requires the importer's zero cache hint".into());
        }
        for index in [3, 4] {
            if input.operands[index].ty != MlirType::Integer(32) {
                return Err("2D TMA coordinates must be i32".into());
            }
        }
        if input.operands[5].ty != MlirType::Integer(64) {
            return Err("2D CTA TMA cache carrier must be i64".into());
        }
        let mut operations = vec![];
        let dst = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            7,
            &input.location,
        )?;
        let barrier = pointer(
            session,
            &mut operations,
            input.operands[1].clone(),
            3,
            &input.location,
        )?;
        let descriptor = pointer(
            session,
            &mut operations,
            input.operands[2].clone(),
            0,
            &input.location,
        )?;
        let mut op = operation(
            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
            vec![
                dst,
                descriptor,
                input.operands[3].clone(),
                input.operands[4].clone(),
                barrier,
            ],
            vec![],
            &input.location,
        )?;
        op.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 0, 0, 0]),
        );
        op.properties
            .insert("useIntrinsic".into(), MlirAttribute::Bool(false));
        operations.push(op);
        Ok(operations)
    }
}

/// Check operand/result counts and canonical identity, then reject leftover source structure.
pub(crate) fn require_input(
    input: &mut OperationInput,
    marker: &str,
    operands: usize,
    results: usize,
) -> Result<(), String> {
    if input.attributes.remove("cuda_oxide_intrinsic_marker")
        != Some(MlirAttribute::String(marker.into()))
    {
        return Err(format!(
            "SM120 operation expected intrinsic marker {marker}"
        ));
    }
    if input.operands.len() != operands
        || input.results.len() != results
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err(format!(
            "SM120 operation expected {operands} operands, {results} results, and no residual attributes or control flow"
        ));
    }
    Ok(())
}
/// Require an unsigned compile-time integer selector at the specified source operand.
pub(crate) fn constant_integer(
    ctx: &Context,
    source: Ptr<Operation>,
    index: usize,
) -> Result<u64, String> {
    let definition = source
        .deref(ctx)
        .get_operand(index)
        .defining_op()
        .ok_or("expected a constant integer operand")?;
    if Operation::get_opid(definition, ctx) != MirConstantOp::get_opid_static() {
        return Err("expected a constant integer operand".into());
    }
    MirConstantOp::new(definition)
        .get_attr_value(ctx)
        .ok_or("constant has no value")?
        .value()
        .to_string_decimal(false)
        .parse()
        .map_err(|_| "constant integer exceeds u64".into())
}
/// Preserve the required NVVM pointer space, including explicit integer-offset conversions.
fn pointer(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    space: u32,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let target = MlirType::dialect(if space == 0 {
        "!llvm.ptr".into()
    } else {
        format!("!llvm.ptr<{space}>")
    })?;
    if value.ty == target {
        return Ok(value);
    }
    let shared_source = value.ty == MlirType::dialect("!llvm.ptr")?
        || value.ty == MlirType::dialect("!llvm.ptr<3>")?;
    let descriptor_source = value.ty == MlirType::dialect("!llvm.ptr<1>")?
        || value.ty == MlirType::dialect("!llvm.ptr<4>")?;
    let cast = if space == 3 && value.ty == MlirType::Integer(32) {
        "llvm.inttoptr"
    } else if (space != 0 && shared_source) || (space == 0 && descriptor_source) {
        "llvm.addrspacecast"
    } else {
        return Err(format!(
            "unsupported pointer {:?} for SM120 address space {space}",
            value.ty
        ));
    };
    let (result, converted) = fresh(session, target);
    operations.push(operation(cast, vec![value], vec![result], location)?);
    Ok(converted)
}
/// Attach a parsed dialect attribute as a target operation property.
pub(crate) fn property(op: &mut MlirOperation, name: &str, value: &str) -> Result<(), String> {
    op.properties
        .insert(name.into(), MlirAttribute::dialect(value)?);
    Ok(())
}
/// Unpack a native aggregate result into source SSA results in their original register order.
fn extract_results(
    operations: &mut Vec<MlirOperation>,
    aggregate: MlirValueUse,
    results: Vec<MlirResult>,
    location: &MlirLocation,
) -> Result<(), String> {
    for (index, result) in results.into_iter().enumerate() {
        let mut extract = operation(
            "llvm.extractvalue",
            vec![aggregate.clone()],
            vec![result],
            location,
        )?;
        extract.properties.insert(
            "position".into(),
            MlirAttribute::DenseI64Array(vec![index as i64]),
        );
        operations.push(extract);
    }
    Ok(())
}
/// Allocate one SSA identifier and return matching result and operand handles.
pub(crate) fn fresh(
    session: &mut TranslationSession<'_>,
    ty: MlirType,
) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}
/// Build a target operation with the supplied operands, results, and source location.
pub(crate) fn operation(
    name: &str,
    operands: Vec<MlirValueUse>,
    results: Vec<MlirResult>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.operands = operands;
    operation.results = results;
    operation.location = location.clone();
    Ok(operation)
}
