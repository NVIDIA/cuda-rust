/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent one-dimensional tile scheduler and runtime work coordinates.

use crate::cute::common::{
    binary, constant_value, emit_binary, emit_i64_constant, expect_shape, fresh, operation,
    stringify, u64_bits,
};
use crate::{
    MlirAttribute, MlirOperation, MlirType, OperationInput, OperationTranslation,
    TranslationRegistry, TranslationSession, TypeTranslation,
};
use dialect_cute::{
    attributes::CuteTileGridAttr,
    scheduler_ops::{
        CuteSchedulerAdvanceOp, CuteSchedulerHasWorkOp, CuteSchedulerNew1dOp,
        CuteWorkTileCoordinatesOp,
    },
    types::CuteWorkTileType,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Represent a verified scheduler work tile as its runtime integer cursor.
pub(in crate::cute) struct WorkTileTypeTranslation;

impl TypeTranslation for WorkTileTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        source
            .deref(ctx)
            .downcast_ref::<CuteWorkTileType>()
            .ok_or_else(|| "expected cute.work_tile".to_owned())?
            .verify(ctx)
            .map_err(stringify)?;
        Ok(MlirType::Integer(64))
    }
}

/// Initialize the persistent scheduler cursor from the source launch coordinate.
pub(in crate::cute) struct SchedulerNewTranslation;

impl OperationTranslation for SchedulerNewTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteSchedulerNew1dOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.scheduler_new_1d", &input, 2, 0)?;
        let mut operations = vec![];
        for (name, result) in [
            ("nvvm.read.ptx.sreg.ctaid.x", input.results[0].clone()),
            ("nvvm.read.ptx.sreg.nctaid.x", input.results[1].clone()),
        ] {
            let (narrow_result, narrow) = fresh(session, MlirType::Integer(32));
            operations.push(operation(
                name,
                vec![narrow_result],
                vec![],
                &input.location,
            )?);
            operations.push(operation(
                "arith.extui",
                vec![result],
                vec![narrow],
                &input.location,
            )?);
        }
        Ok(operations)
    }
}

/// Compare the cursor with the verified total tile count.
pub(in crate::cute) struct SchedulerHasWorkTranslation;

impl OperationTranslation for SchedulerHasWorkTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteSchedulerHasWorkOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("cute.scheduler_has_work", &input, 1, 1)?;
        let total = semantic
            .tile_grid(ctx)
            .and_then(CuteTileGridAttr::total_tiles)
            .ok_or_else(|| "scheduler tile count overflowed".to_owned())?;
        let (constant, total_use) = constant_value(
            session,
            MlirType::Integer(64),
            u64_bits(total),
            &input.location,
        )?;
        let mut compare = operation(
            "arith.cmpi",
            input.results,
            vec![input.operands[0].clone(), total_use],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 6, // unsigned less-than
                ty: MlirType::Integer(64),
            },
        );
        Ok(vec![constant, compare])
    }
}

/// Decode the cursor into M/N tile coordinates using the verified traversal order.
pub(in crate::cute) struct WorkTileCoordinatesTranslation;

impl OperationTranslation for WorkTileCoordinatesTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteWorkTileCoordinatesOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.work_tile_coordinates", &input, 4, 1)?;
        let tile_ty = source.deref(ctx).get_operand(0).get_type(ctx);
        let tile_ref = tile_ty.deref(ctx);
        let grid = tile_ref
            .downcast_ref::<CuteWorkTileType>()
            .ok_or_else(|| "work_tile_coordinates operand lost its grid".to_owned())?
            .grid;
        let linear = input.operands[0].clone();
        let mut operations = vec![];
        let zero = emit_i64_constant(session, &mut operations, 0, &input.location)?;
        operations.push(binary(
            "arith.addi",
            input.results[0].clone(),
            linear.clone(),
            zero,
            &input.location,
        )?);

        let m_width = emit_i64_constant(session, &mut operations, grid.m_tiles, &input.location)?;
        let quotient_m = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            linear.clone(),
            m_width.clone(),
            &input.location,
        )?;
        let consumed_m = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            quotient_m.clone(),
            m_width,
            &input.location,
        )?;
        operations.push(binary(
            "arith.subi",
            input.results[1].clone(),
            linear,
            consumed_m,
            &input.location,
        )?);

        let n_width = emit_i64_constant(session, &mut operations, grid.n_tiles, &input.location)?;
        let batch = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            quotient_m.clone(),
            n_width.clone(),
            &input.location,
        )?;
        let consumed_n = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            batch.clone(),
            n_width,
            &input.location,
        )?;
        operations.push(binary(
            "arith.subi",
            input.results[2].clone(),
            quotient_m,
            consumed_n,
            &input.location,
        )?);
        let zero = emit_i64_constant(session, &mut operations, 0, &input.location)?;
        operations.push(binary(
            "arith.addi",
            input.results[3].clone(),
            batch,
            zero,
            &input.location,
        )?);
        Ok(operations)
    }
}

/// Advance by the persistent grid stride, saturating instead of wrapping on overflow.
pub(in crate::cute) struct SchedulerAdvanceTranslation;

impl OperationTranslation for SchedulerAdvanceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteSchedulerAdvanceOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.scheduler_advance", &input, 1, 2)?;
        let mut operations = vec![];
        let wrapped = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            input.operands[0].clone(),
            input.operands[1].clone(),
            &input.location,
        )?;
        let (overflow_result, overflow) = fresh(session, MlirType::Integer(1));
        let mut compare = operation(
            "arith.cmpi",
            vec![overflow_result],
            vec![wrapped.clone(), input.operands[0].clone()],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 6, // unsigned less-than
                ty: MlirType::Integer(64),
            },
        );
        operations.push(compare);
        let maximum = emit_i64_constant(session, &mut operations, u64::MAX, &input.location)?;
        operations.push(operation(
            "arith.select",
            input.results,
            vec![overflow, maximum, wrapped],
            &input.location,
        )?);
        Ok(operations)
    }
}
