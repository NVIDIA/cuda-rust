/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Shared scalar and packed arithmetic for CuTe and instruction-only profiles.
//! Precision, rounding, lane masks, and intrinsic identity survive translation.
use crate::nvvm::tcgen05::{fresh, operation, require_input};
use crate::{
    DropAttribute, MlirAttribute, MlirFloatType, MlirOperation, MlirType, OperationInput,
    OperationTranslation, TranslationError, TranslationRegistry, TranslationSession,
};
use dialect_nvvm::ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Source-operation verifier run before discarding its intrinsic metadata.
type VerifyOperation = fn(&Context, Ptr<Operation>) -> Result<(), String>;

/// Register reviewed scalar, packed, and warp arithmetic with exact native intrinsic identities.
pub fn register_nvvm_numeric_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<ScalarMathFormatAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathOperationAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathPrecisionAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarMathSubnormalAttr>(DropAttribute)?;
    registry.register_operation::<ScalarMathOp>(MathTranslation)?;
    registry.register_operation::<ShflSyncBflyF32Op>(IntrinsicTranslation {
        marker: "v1:i0055",
        name: "llvm.nvvm.shfl.sync.bfly.f32",
        operands: 3,
        shuffle: true,
        verify: |ctx, source| {
            ShflSyncBflyF32Op::new(source)
                .verify(ctx)
                .map_err(|e| e.to_string())
        },
    })?;
    register_nvvm_numeric_common_pack(registry)
}

/// Preserve each architecture's validated CUTLASS math lowering while sharing
/// the remaining arithmetic and asynchronous-copy control registrations.
pub(crate) fn register_nvvm_profile_numeric_pack(
    registry: &mut TranslationRegistry,
    gpu_arch: &str,
) -> Result<(), TranslationError> {
    if gpu_arch == "sm_120a" {
        register_nvvm_numeric_common_pack(registry)?;
        crate::register_nvvm_attention_pack(registry)
    } else {
        register_nvvm_numeric_pack(registry)?;
        crate::nvvm::attention::register_nvvm_async_control_pack(registry)
    }
}

/// Register arithmetic and synchronization mappings shared by both architecture profiles.
fn register_nvvm_numeric_common_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_operation::<CvtF16x2F32Op>(PackTranslation)?;
    registry.register_attribute::<ScalarArithmeticFormatAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarArithmeticOperationAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarArithmeticRoundingAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarArithmeticSubnormalAttr>(DropAttribute)?;
    registry.register_attribute::<ScalarArithmeticSaturationAttr>(DropAttribute)?;
    registry.register_operation::<ScalarArithmeticOp>(ArithmeticTranslation)?;
    /// Register one packed FP32 instruction with its exact identity and source verifier.
    macro_rules! packed {
        ($op:ty, $marker:literal, $name:literal, $args:literal) => {
            registry.register_operation::<$op>(PackedArithmeticTranslation {
                marker: $marker,
                name: $name,
                operands: $args,
                verify: |ctx, source| <$op>::new(source).verify(ctx).map_err(|e| e.to_string()),
            })?;
        };
    }
    packed!(AddF32x2Op, "v1:i0995", "nvvm.add.packed.f32x2", 2);
    packed!(MulF32x2Op, "v1:i0999", "nvvm.mul.packed.f32x2", 2);
    packed!(FmaF32x2Op, "v1:i1001", "nvvm.fma.packed.f32x2", 3);

    /// Register one direct intrinsic call with its operand count and shuffle contract.
    macro_rules! direct {
        ($op:ty, $marker:literal, $name:literal, $args:literal, $shuffle:literal) => {
            registry.register_operation::<$op>(IntrinsicTranslation {
                marker: $marker,
                name: $name,
                operands: $args,
                shuffle: $shuffle,
                verify: |ctx, source| <$op>::new(source).verify(ctx).map_err(|e| e.to_string()),
            })?;
        };
    }
    registry.register_operation::<ReduxSyncFmaxOp>(SyncTranslation(0))?;
    registry.register_operation::<CpAsyncCa4Op>(SyncTranslation(1))?;
    registry.register_operation::<CpAsyncMbarrierArriveNoIncOp>(SyncTranslation(2))?;
    registry.register_operation::<VoteSyncAllOp>(SyncTranslation(3))?;
    registry.register_operation::<BarWarpSyncOp>(SyncTranslation(4))?;
    direct!(
        ShflSyncDownF32Op,
        "v1:i0056",
        "llvm.nvvm.shfl.sync.down.f32",
        3,
        true
    );
    direct!(
        ShflSyncDownI32Op,
        "v1:i0052",
        "llvm.nvvm.shfl.sync.down.i32",
        3,
        true
    );
    direct!(
        ShflSyncIdxF32Op,
        "v1:i0054",
        "llvm.nvvm.shfl.sync.idx.f32",
        3,
        true
    );
    direct!(
        ShflSyncIdxI32Op,
        "v1:i0050",
        "llvm.nvvm.shfl.sync.idx.i32",
        3,
        true
    );
    direct!(
        ShflSyncBflyI32Op,
        "v1:i0051",
        "llvm.nvvm.shfl.sync.bfly.i32",
        3,
        true
    );
    registry.register_operation::<SetMaxNRegIncOp>(RegisterTranslation(true))?;
    registry.register_operation::<SetMaxNRegDecOp>(RegisterTranslation(false))?;
    Ok(())
}

/// Build an LLVM intrinsic call retaining translated operands/results and an empty bundle list.
fn intrinsic(input: OperationInput, name: &str) -> Result<MlirOperation, String> {
    let count = input.operands.len();
    let mut call = operation(
        "llvm.call_intrinsic",
        input.operands,
        input.results,
        &input.location,
    )?;
    call.properties
        .insert("intrin".into(), MlirAttribute::String(name.into()));
    call.properties.insert(
        "op_bundle_sizes".into(),
        MlirAttribute::DenseI32Array(vec![]),
    );
    call.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![count as i32, 0]),
    );
    Ok(call)
}

/// Validate one catalog identity and emit its named intrinsic with reviewed operands.
struct IntrinsicTranslation {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// LLVM intrinsic spelling selected for this reviewed source operation.
    name: &'static str,
    /// Expected number of runtime operands at the source boundary.
    operands: usize,
    /// Add the full-warp clamp required by the canonical butterfly-shuffle contract.
    shuffle: bool,
    /// Check the source operation's selectors and operand contract.
    verify: VerifyOperation,
}
impl OperationTranslation for IntrinsicTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        require_input(&mut input, self.marker, self.operands, 1)?;
        let mut ops = vec![];
        if self.shuffle {
            // The public warp shuffle operates across the full 32-lane warp;
            // the membership mask remains a distinct caller-supplied operand.
            let (result, clamp) = fresh(session, MlirType::Integer(32));
            let mut constant = operation("arith.constant", vec![], vec![result], &input.location)?;
            constant.properties.insert(
                "value".into(),
                MlirAttribute::Integer {
                    value: 31,
                    ty: MlirType::Integer(32),
                },
            );
            ops.push(constant);
            input.operands.push(clamp);
        }
        ops.push(intrinsic(input, self.name)?);
        Ok(ops)
    }
}

/// Map reviewed scalar math to exact native intrinsics with preserved precision and subnormal mode.
pub(crate) struct MathTranslation;
impl OperationTranslation for MathTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let op = ScalarMathOp::new(source);
        op.verify(ctx).map_err(|e| e.to_string())?;
        if op.get_attr_nvvm_scalar_math_format(ctx).as_deref() != Some(&ScalarMathFormatAttr::F32)
            || op.get_attr_nvvm_scalar_math_precision(ctx).as_deref()
                != Some(&ScalarMathPrecisionAttr::Approx)
        {
            return Err(
                "numeric pack supports approximate FP32 exp2/log2/reciprocal/rsqrt/sine only"
                    .into(),
            );
        }
        let pair = (
            op.get_attr_nvvm_scalar_math_operation(ctx),
            op.get_attr_nvvm_scalar_math_subnormal(ctx),
        );
        let (marker, name) = match (pair.0.as_deref(), pair.1.as_deref()) {
            (Some(ScalarMathOperationAttr::Ex2), Some(ScalarMathSubnormalAttr::Preserve)) => {
                ("v1:i0819", "llvm.nvvm.ex2.approx.f32")
            }
            (Some(ScalarMathOperationAttr::Ex2), Some(ScalarMathSubnormalAttr::Ftz)) => {
                ("v1:i0820", "llvm.nvvm.ex2.approx.ftz.f32")
            }
            (Some(ScalarMathOperationAttr::Lg2), Some(ScalarMathSubnormalAttr::Preserve)) => {
                ("v1:i0786", "llvm.nvvm.lg2.approx.f")
            }
            (Some(ScalarMathOperationAttr::Lg2), Some(ScalarMathSubnormalAttr::Ftz)) => {
                ("v1:i0787", "llvm.nvvm.lg2.approx.ftz.f")
            }
            (Some(ScalarMathOperationAttr::Rcp), Some(ScalarMathSubnormalAttr::Ftz)) => {
                ("v1:i0788", "llvm.nvvm.rcp.approx.ftz.f")
            }
            (Some(ScalarMathOperationAttr::Sin), Some(ScalarMathSubnormalAttr::Preserve)) => {
                ("v1:i0782", "llvm.nvvm.sin.approx.f")
            }
            (Some(ScalarMathOperationAttr::Rsqrt), Some(ScalarMathSubnormalAttr::Preserve)) => {
                ("v1:i0802", "llvm.nvvm.rsqrt.approx.f")
            }
            _ => {
                return Err(
                    "numeric pack does not support this scalar math selector combination".into(),
                );
            }
        };
        require_input(&mut input, marker, 1, 1)?;
        Ok(vec![intrinsic(input, name)?])
    }
}

/// Preserve scalar arithmetic format, rounding, subnormal, and saturation selectors.
struct ArithmeticTranslation;
impl OperationTranslation for ArithmeticTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let op = ScalarArithmeticOp::new(source);
        op.verify(ctx).map_err(|e| e.to_string())?;
        if op.get_attr_nvvm_scalar_arithmetic_format(ctx).as_deref()
            != Some(&ScalarArithmeticFormatAttr::F32)
            || op.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref()
                != Some(&ScalarArithmeticOperationAttr::Fma)
            || op.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref()
                != Some(&ScalarArithmeticRoundingAttr::Rn)
            || op
                .get_attr_nvvm_scalar_arithmetic_saturation(ctx)
                .as_deref()
                != Some(&ScalarArithmeticSaturationAttr::None)
        {
            return Err("numeric pack supports unsaturated FP32 round-nearest FMA only".into());
        }
        let (marker, name) = match op.get_attr_nvvm_scalar_arithmetic_subnormal(ctx).as_deref() {
            Some(ScalarArithmeticSubnormalAttr::Preserve) => ("v1:i0418", "llvm.nvvm.fma.rn.f"),
            Some(ScalarArithmeticSubnormalAttr::Ftz) => ("v1:i0419", "llvm.nvvm.fma.rn.ftz.f"),
            _ => return Err("FP32 FMA requires a subnormal policy".into()),
        };
        require_input(&mut input, marker, 3, 1)?;
        Ok(vec![intrinsic(input, name)?])
    }
}

/// The Rust carrier holds the low FP32 lane in bits 0..31. Bitcasting to an
/// LLVM vector preserves both lanes and lets NVVM select packed instructions;
/// scalar arithmetic would lose the source instruction-family contract.
struct PackedArithmeticTranslation {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// LLVM intrinsic spelling selected for this reviewed source operation.
    name: &'static str,
    /// Expected number of runtime operands at the source boundary.
    operands: usize,
    /// Check the source operation's selectors and operand contract.
    verify: VerifyOperation,
}
impl OperationTranslation for PackedArithmeticTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        require_input(&mut input, self.marker, self.operands, 1)?;
        let pair_type = MlirType::Vector {
            shape: vec![2],
            element: Box::new(MlirType::Float(MlirFloatType::F32)),
        };
        let mut ops = Vec::with_capacity(self.operands + 2);
        let mut pairs = Vec::with_capacity(self.operands);
        for operand in input.operands {
            let (result, pair) = fresh(session, pair_type.clone());
            ops.push(operation(
                "llvm.bitcast",
                vec![operand],
                vec![result],
                &input.location,
            )?);
            pairs.push(pair);
        }
        let (result, pair) = fresh(session, pair_type);
        let mut packed = operation(self.name, pairs, vec![result], &input.location)?;
        packed.properties.insert(
            "rnd".into(),
            MlirAttribute::dialect("#nvvm.fp_rnd_mode<rn>")?,
        );
        packed
            .properties
            .insert("ftz".into(), MlirAttribute::Bool(false));
        ops.push(packed);
        ops.push(operation(
            "llvm.bitcast",
            vec![pair],
            input.results,
            &input.location,
        )?);
        Ok(ops)
    }
}

/// Emit the selected register-budget increase or decrease instruction.
struct RegisterTranslation(#[doc = "True increases the register budget; false decreases it."] bool);
impl OperationTranslation for RegisterTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let count = if self.0 {
            let op = SetMaxNRegIncOp::new(source);
            op.verify(ctx).map_err(|e| e.to_string())?;
            op.register_count(ctx)
        } else {
            let op = SetMaxNRegDecOp::new(source);
            op.verify(ctx).map_err(|e| e.to_string())?;
            op.register_count(ctx)
        }
        .ok_or("missing register count")?;
        input.attributes.remove("register_count");
        require_input(
            &mut input,
            if self.0 { "v1:i0915" } else { "v1:i0916" },
            0,
            0,
        )?;
        let mut op = operation("nvvm.setmaxregister", vec![], vec![], &input.location)?;
        op.properties.insert(
            "regCount".into(),
            MlirAttribute::Integer {
                value: count as i128,
                ty: MlirType::Integer(32),
            },
        );
        op.properties.insert(
            "action".into(),
            MlirAttribute::dialect(if self.0 {
                "#nvvm<action increase>"
            } else {
                "#nvvm<action decrease>"
            })?,
        );
        Ok(vec![op])
    }
}
/// Convert low-first FP32 operands into the exact packed FP16 register representation.
struct PackTranslation;
impl OperationTranslation for PackTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CvtF16x2F32Op::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0081", 2, 1)?;
        let (result, packed) = fresh(
            session,
            MlirType::Vector {
                shape: vec![2],
                element: Box::new(MlirType::Float(MlirFloatType::F16)),
            },
        );
        // Canonical Rust API is (low, high); LLVM's intrinsic is (high, low).
        input.operands.reverse();
        let mut call = operation(
            "llvm.call_intrinsic",
            input.operands,
            vec![result],
            &input.location,
        )?;
        call.properties.insert(
            "intrin".into(),
            MlirAttribute::String("llvm.nvvm.ff2f16x2.rn".into()),
        );
        call.properties.insert(
            "op_bundle_sizes".into(),
            MlirAttribute::DenseI32Array(vec![]),
        );
        call.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![2, 0]),
        );
        Ok(vec![
            call,
            operation("llvm.bitcast", vec![packed], input.results, &input.location)?,
        ])
    }
}

/// Native MLIR operations retain the compiler's PTX lowering for these instructions.
struct SyncTranslation(
    #[doc = "Reviewed selector: 0 reduction, 1 copy, 2 barrier arrival, 3 vote, or 4 warp sync."]
    u8,
);
impl OperationTranslation for SyncTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let (marker, args, results) = match self.0 {
            0 => {
                ReduxSyncFmaxOp::new(source)
                    .verify(ctx)
                    .map_err(|e| e.to_string())?;
                ("v1:i1007", 2, 1)
            }
            1 => {
                CpAsyncCa4Op::new(source)
                    .verify(ctx)
                    .map_err(|e| e.to_string())?;
                ("v1:i0086", 2, 0)
            }
            2 => {
                CpAsyncMbarrierArriveNoIncOp::new(source)
                    .verify(ctx)
                    .map_err(|e| e.to_string())?;
                ("v1:i0103", 1, 0)
            }
            3 => {
                VoteSyncAllOp::new(source)
                    .verify(ctx)
                    .map_err(|e| e.to_string())?;
                ("v1:i0040", 2, 1)
            }
            4 => {
                BarWarpSyncOp::new(source)
                    .verify(ctx)
                    .map_err(|e| e.to_string())?;
                ("v1:i0049", 1, 0)
            }
            _ => unreachable!(),
        };
        require_input(&mut input, marker, args, results)?;
        let mut e = crate::cute::Emit {
            session,
            location: input.location,
            ops: vec![],
        };
        let (name, operands) = match self.0 {
            0 => (
                "nvvm.redux.sync",
                vec![input.operands[1].clone(), input.operands[0].clone()],
            ),
            1 => {
                let dst = e.llvm_ptr(input.operands[0].clone(), 3)?;
                let src = e.llvm_ptr(input.operands[1].clone(), 1)?;
                ("nvvm.cp.async.shared.global", vec![dst, src])
            }
            2 => {
                let bar = e.llvm_ptr(input.operands[0].clone(), 3)?;
                ("nvvm.cp.async.mbarrier.arrive", vec![bar])
            }
            3 => ("nvvm.vote.sync", input.operands),
            4 => ("nvvm.bar.warp.sync", input.operands),
            _ => unreachable!(),
        };
        let mut op = e.operation(name, operands)?;
        op.results = input.results;
        match self.0 {
            0 => {
                crate::cute::property(&mut op, "kind", "#nvvm<reduction_kind fmax>")?;
            }
            1 => {
                crate::cute::integer_property(&mut op, "size", 32, 4);
                crate::cute::property(&mut op, "modifier", "#nvvm<load_cache_modifier ca>")?;
            }
            2 => {
                op.properties
                    .insert("noinc".into(), MlirAttribute::Bool(true));
            }
            3 => {
                crate::cute::property(&mut op, "kind", "#nvvm<vote_sync_kind all>")?;
            }
            _ => {}
        }
        e.ops.push(op);
        Ok(e.ops)
    }
}
