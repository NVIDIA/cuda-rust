/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous-copy control and reviewed scalar instruction mappings.
//!
//! Selectors and canonical intrinsic identities are checked before dropping
//! source attributes. Common numeric operations share their precise mappings.

use crate::nvvm::sm120::{constant_integer, fresh, operation, property, require_input};
use crate::{
    DropAttribute, MlirAttribute, MlirOperation, MlirType, OperationInput, OperationTranslation,
    TranslationError, TranslationRegistry, TranslationSession,
};
use dialect_nvvm::ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Register the instruction variants used by the Ampere attention schedule.
pub fn register_nvvm_attention_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<ScalarMathFormatAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathOperationAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathPrecisionAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathSubnormalAttr>(DropAttribute)?;
    registry.register_operation::<ScalarMathOp>(MathTranslation)?;
    registry.register_operation::<ShflSyncBflyF32Op>(ShuffleTranslation)?;
    register_nvvm_async_control_pack(registry)
}

/// Register asynchronous-copy commit and compile-time wait-count operations.
pub(crate) fn register_nvvm_async_control_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_operation::<CpAsyncCommitGroupOp>(CommitTranslation)?;
    registry.register_operation::<CpAsyncWaitGroupOp>(WaitTranslation)?;
    Ok(())
}

/// Map reviewed attention math while preserving approximation, subnormal mode, and identity.
struct MathTranslation;
impl OperationTranslation for MathTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let math = ScalarMathOp::new(source);
        math.verify(ctx).map_err(|e| e.to_string())?;
        // Log2, reciprocal square root and sine use the common numeric mapping;
        // exp2 and reciprocal retain the SM120 attention schedule's
        // validated CUTLASS lowering below.
        if matches!(
            math.get_attr_nvvm_scalar_math_operation(ctx).as_deref(),
            Some(
                ScalarMathOperationAttr::Lg2
                    | ScalarMathOperationAttr::Rsqrt
                    | ScalarMathOperationAttr::Sin
            )
        ) {
            return crate::nvvm::numeric::MathTranslation.translate(ctx, source, input, session);
        }
        let selectors = (
            math.get_attr_nvvm_scalar_math_format(ctx),
            math.get_attr_nvvm_scalar_math_operation(ctx),
            math.get_attr_nvvm_scalar_math_precision(ctx),
            math.get_attr_nvvm_scalar_math_subnormal(ctx),
        );
        let (target, marker) =
            match (
                selectors.0.as_deref(),
                selectors.1.as_deref(),
                selectors.2.as_deref(),
                selectors.3.as_deref(),
            ) {
                (
                    Some(ScalarMathFormatAttr::F32),
                    Some(ScalarMathOperationAttr::Ex2),
                    Some(ScalarMathPrecisionAttr::Approx),
                    Some(ScalarMathSubnormalAttr::Preserve),
                ) => ("llvm.inline_asm", "v1:i0819"),
                (
                    Some(ScalarMathFormatAttr::F32),
                    Some(ScalarMathOperationAttr::Ex2),
                    Some(ScalarMathPrecisionAttr::Approx),
                    Some(ScalarMathSubnormalAttr::Ftz),
                ) => ("llvm.inline_asm", "v1:i0820"),
                (
                    Some(ScalarMathFormatAttr::F32),
                    Some(ScalarMathOperationAttr::Rcp),
                    Some(ScalarMathPrecisionAttr::Approx),
                    Some(ScalarMathSubnormalAttr::Ftz),
                ) => ("nvvm.rcp.approx.ftz.f", "v1:i0788"),
                _ => return Err(
                    "attention math mapping supports F32 approximate exp2 and FTZ reciprocal only"
                        .into(),
                ),
            };
        require_input(&mut input, marker, 1, 1)?;
        let mut op = operation(target, input.operands, input.results, &input.location)?;
        if matches!(marker, "v1:i0819" | "v1:i0820") {
            // CUTLASS 4.7 parses nvvm.ex2 and llvm.nvvm.ex2.approx.f, but its
            // bundled libNVVM rejects their lowering. Keep each typed
            // intrinsic's exact subnormal mode through this compiler fallback.
            // The FTZ form also matches CuTeDSL 4.7 math.exp2(fastmath=True).
            // There are no memory effects, so the pure asm needs no sideeffect.
            let instruction = if marker == "v1:i0820" {
                "ex2.approx.ftz.f32 $0, $1;"
            } else {
                "ex2.approx.f32 $0, $1;"
            };
            op.properties.insert(
                "asm_string".into(),
                MlirAttribute::String(instruction.into()),
            );
            op.properties
                .insert("constraints".into(), MlirAttribute::String("=f,f".into()));
        }
        Ok(vec![op])
    }
}

/// Preserve butterfly-shuffle membership, lane offset, and warp clamp.
struct ShuffleTranslation;
impl OperationTranslation for ShuffleTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        ShflSyncBflyF32Op::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0055", 3, 1)?;
        let (result, clamp) = fresh(session, MlirType::Integer(32));
        let mut constant = operation("arith.constant", vec![], vec![result], &input.location)?;
        constant.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: 31,
                ty: MlirType::Integer(32),
            },
        );
        input.operands.push(clamp);
        let mut shuffle = operation(
            "nvvm.shfl.sync",
            input.operands,
            input.results,
            &input.location,
        )?;
        property(&mut shuffle, "kind", "#nvvm<shfl_kind bfly>")?;
        Ok(vec![constant, shuffle])
    }
}

/// Commit preceding asynchronous copies into a hardware group.
struct CommitTranslation;
impl OperationTranslation for CommitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CpAsyncCommitGroupOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0094", 0, 0)?;
        Ok(vec![operation(
            "nvvm.cp.async.commit.group",
            vec![],
            vec![],
            &input.location,
        )?])
    }
}

/// Wait for a legal compile-time number of outstanding asynchronous-copy groups.
struct WaitTranslation;
impl OperationTranslation for WaitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CpAsyncWaitGroupOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0096", 1, 0)?;
        let count = constant_integer(ctx, source, 0)?;
        if count > 7 {
            return Err("cp.async.wait_group requires an immediate group count in 0..=7".into());
        }
        let mut wait = operation("nvvm.cp.async.wait.group", vec![], vec![], &input.location)?;
        wait.properties.insert(
            "n".into(),
            MlirAttribute::Integer {
                value: count.into(),
                ty: MlirType::Integer(32),
            },
        );
        Ok(vec![wait])
    }
}
