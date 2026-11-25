//! Attention math and asynchronous-copy semantics across both consumer profiles.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::ops::{MirConstantOp, MirFuncOp, MirReturnOp};
use dialect_nvvm::ops::*;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    utils::apint::APInt,
    value::Value,
};

/// Typed scalar and asynchronous-copy source function for comparing both consumer profiles.
struct Fixture {
    /// Source IR context owning the fixture's registered types and operations.
    ctx: Context,
    /// Source module containing the operations under test.
    module: ModuleOp,
    /// Function entry block where the fixture appends source operations.
    block: Ptr<BasicBlock>,
}

impl Fixture {
    /// Build a MIR/NVVM context and function with the operands required by attention intrinsics.
    fn new() -> Self {
        Self::with_float_width(false)
    }

    /// Select FP64 only when testing rejection of an otherwise valid scalar-math operation.
    fn with_float_width(double: bool) -> Self {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        let scalar = if double {
            FP64Type::get(&ctx).into()
        } else {
            FP32Type::get(&ctx).into()
        };
        let u32 = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let args = vec![u32, scalar, u32];
        let module = ModuleOp::new(&mut ctx, "attention_primitives".try_into().unwrap());
        let ty = FunctionType::get(&ctx, args.clone(), vec![]);
        let raw = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, raw, TypeAttr::new(ty.into()))
            .set_symbol_name(&mut ctx, "attention_primitives".try_into().unwrap());
        module.append_operation(&mut ctx, raw, 0);
        let block = BasicBlock::new(&mut ctx, None, args);
        block.insert_at_back(raw.deref(&ctx).get_region(0), &ctx);
        Self { ctx, module, block }
    }

    /// Return the indexed source function argument for constructing a test operation.
    fn arg(&self, index: usize) -> Value {
        self.block.deref(&self.ctx).get_argument(index)
    }

    /// Append a typed source operation to the fixture block and return its operation handle.
    fn append(&mut self, op: Ptr<Operation>, marker: &str) -> Ptr<Operation> {
        if !marker.is_empty() {
            op.deref_mut(&self.ctx).attributes.set(
                "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                StringAttr::new(marker.into()),
            );
        }
        op.insert_at_back(self.block, &self.ctx);
        op
    }

    /// Append an unsigned compile-time wait count to the source block.
    fn constant(&mut self, value: u64) -> Value {
        let ty = IntegerType::get(&self.ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            &mut self.ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(op).set_attr_value(
            &self.ctx,
            IntegerAttr::new(ty, APInt::from_u64(value, 32usize.try_into().unwrap())),
        );
        self.append(op, "");
        op.deref(&self.ctx).get_result(0)
    }

    /// Append scalar math with explicit precision, format, subnormal mode, and catalog identity.
    fn math(
        &mut self,
        operation: ScalarMathOperationAttr,
        subnormal: ScalarMathSubnormalAttr,
        marker: &str,
    ) -> Ptr<Operation> {
        let input = self.arg(1);
        let op = ScalarMathOp::build(
            &mut self.ctx,
            input,
            ScalarMathFormatAttr::F32,
            operation,
            ScalarMathPrecisionAttr::Approx,
            subnormal,
        );
        self.append(op, marker)
    }

    /// Append the source function return after the tested instructions.
    fn finish(&mut self) {
        let op = Operation::new(
            &mut self.ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        self.append(op, "");
    }

    /// Export the fixture using the default full-CuTe attention profile.
    fn translate(&self) -> Result<String, String> {
        self.translate_with_profile(&CutlassPrimitivesMlir22::new("sm_120a").unwrap())
    }

    /// Export and render the same source fixture with the supplied consumer profile.
    fn translate_with_profile(&self, profile: &dyn MlirConsumerProfile) -> Result<String, String> {
        profile
            .translate_module(&self.ctx, &self.module)
            .and_then(|m| render_module(&m).map_err(Into::into))
            .map_err(|e| e.to_string())
    }
}

/// RMSNorm's approximate reciprocal square root preserves subnormals in each consumer profile.
#[test]
fn rsqrt_preserves_subnormals_and_native_intrinsic_identity() {
    let mut f = Fixture::new();
    f.math(
        ScalarMathOperationAttr::Rsqrt,
        ScalarMathSubnormalAttr::Preserve,
        "v1:i0802",
    );
    f.finish();
    for arch in ["sm_100a", "sm_120a"] {
        let profiles: [&dyn MlirConsumerProfile; 2] = [
            &CutlassPrimitivesMlir22::new(arch).unwrap(),
            &CutlassFullCuteMlir22::new(arch).unwrap(),
        ];
        for profile in profiles {
            let text = f.translate_with_profile(profile).unwrap();
            assert!(text.contains("\"llvm.call_intrinsic\"(%v1)"), "{text}");
            assert!(
                text.contains("intrin = \"llvm.nvvm.rsqrt.approx.f\""),
                "{text}"
            );
            assert!(text.contains(": (f32) -> (f32)"), "{text}");
            assert!(!text.contains("rsqrt.approx.ftz"), "{text}");
            assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
        }
    }
}

/// Only the reviewed FP32, approximate, preserve-subnormal rsqrt contract is admitted.
#[test]
fn rsqrt_rejects_unsupported_variants_and_forged_identity() {
    for case in 0..6 {
        let mut f = Fixture::with_float_width(case == 2);
        let input = f.arg(if case == 3 { 0 } else { 1 });
        let op = ScalarMathOp::build(
            &mut f.ctx,
            input,
            if case == 2 {
                ScalarMathFormatAttr::F64
            } else {
                ScalarMathFormatAttr::F32
            },
            ScalarMathOperationAttr::Rsqrt,
            if case == 1 {
                ScalarMathPrecisionAttr::Rn
            } else {
                ScalarMathPrecisionAttr::Approx
            },
            if case == 0 {
                ScalarMathSubnormalAttr::Ftz
            } else {
                ScalarMathSubnormalAttr::Preserve
            },
        );
        let marker = match case {
            0 | 4 => "v1:i0803",
            2 => "v1:i0804",
            5 => "",
            _ => "v1:i0802",
        };
        f.append(op, marker);
        f.finish();
        let profiles: [&dyn MlirConsumerProfile; 2] = [
            &CutlassPrimitivesMlir22::new("sm_120a").unwrap(),
            &CutlassFullCuteMlir22::new("sm_120a").unwrap(),
        ];
        for profile in profiles {
            let error = f.translate_with_profile(profile).unwrap_err();
            if case >= 4 {
                assert!(
                    error.contains("expected intrinsic marker v1:i0802"),
                    "{error}"
                );
            }
        }
    }
}

/// SnakeBeta's approximate sine preserves subnormals in each consumer profile.
#[test]
fn sin_preserves_subnormals_and_native_intrinsic_identity() {
    let mut f = Fixture::new();
    f.math(
        ScalarMathOperationAttr::Sin,
        ScalarMathSubnormalAttr::Preserve,
        "v1:i0782",
    );
    f.finish();
    for arch in ["sm_100a", "sm_120a"] {
        let profiles: [&dyn MlirConsumerProfile; 2] = [
            &CutlassPrimitivesMlir22::new(arch).unwrap(),
            &CutlassFullCuteMlir22::new(arch).unwrap(),
        ];
        for profile in profiles {
            let text = f.translate_with_profile(profile).unwrap();
            assert!(text.contains("\"llvm.call_intrinsic\"(%v1)"), "{text}");
            assert!(
                text.contains("intrin = \"llvm.nvvm.sin.approx.f\""),
                "{text}"
            );
            assert!(text.contains(": (f32) -> (f32)"), "{text}");
            assert!(!text.contains("sin.approx.ftz"), "{text}");
            assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
        }
    }
}

/// Only the reviewed FP32, approximate, preserve-subnormal sin contract is admitted.
#[test]
fn sin_rejects_unsupported_variants_and_forged_identity() {
    for case in 0..6 {
        let mut f = Fixture::with_float_width(case == 2);
        let input = f.arg(if case == 3 { 0 } else { 1 });
        let op = ScalarMathOp::build(
            &mut f.ctx,
            input,
            if case == 2 {
                ScalarMathFormatAttr::F64
            } else {
                ScalarMathFormatAttr::F32
            },
            ScalarMathOperationAttr::Sin,
            if case == 1 {
                ScalarMathPrecisionAttr::Rn
            } else {
                ScalarMathPrecisionAttr::Approx
            },
            if case == 0 {
                ScalarMathSubnormalAttr::Ftz
            } else {
                ScalarMathSubnormalAttr::Preserve
            },
        );
        let marker = match case {
            0 | 4 => "v1:i0783",
            2 => "v1:i0784",
            5 => "",
            _ => "v1:i0782",
        };
        f.append(op, marker);
        f.finish();
        let profiles: [&dyn MlirConsumerProfile; 2] = [
            &CutlassPrimitivesMlir22::new("sm_120a").unwrap(),
            &CutlassFullCuteMlir22::new("sm_120a").unwrap(),
        ];
        for profile in profiles {
            let error = f.translate_with_profile(profile).unwrap_err();
            if case >= 4 {
                assert!(
                    error.contains("expected intrinsic marker v1:i0782"),
                    "{error}"
                );
            }
        }
    }
}

/// Check both profiles retain the exact FTZ exp2 intrinsic and mode.
#[test]
fn ftz_exp2_keeps_its_exact_subnormal_mode_in_both_profiles() {
    let profiles: [&dyn MlirConsumerProfile; 2] = [
        &CutlassPrimitivesMlir22::new("sm_120a").unwrap(),
        &CutlassFullCuteMlir22::new("sm_120a").unwrap(),
    ];
    let mut f = Fixture::new();
    f.math(
        ScalarMathOperationAttr::Ex2,
        ScalarMathSubnormalAttr::Ftz,
        "v1:i0820",
    );
    f.finish();
    for profile in profiles {
        let text = f.translate_with_profile(profile).unwrap();
        assert!(text.contains("\"llvm.inline_asm\"(%v1)"), "{text}");
        assert!(
            text.contains("asm_string = \"ex2.approx.ftz.f32 $0, $1;\""),
            "{text}"
        );
        assert!(text.contains("constraints = \"=f,f\""), "{text}");
        assert!(text.contains(": (f32) -> (f32)"), "{text}");
        assert!(!text.contains("ex2.approx.f32"), "{text}");
        assert!(!text.contains("has_side_effects"), "{text}");
        assert!(!text.contains("llvm.call"), "{text}");
        assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
    }
}

/// Reject an FTZ exp2 operation mislabeled with another canonical identity.
#[test]
fn ftz_exp2_rejects_other_canonical_identities_in_both_profiles() {
    let profiles: [&dyn MlirConsumerProfile; 2] = [
        &CutlassPrimitivesMlir22::new("sm_120a").unwrap(),
        &CutlassFullCuteMlir22::new("sm_120a").unwrap(),
    ];
    for marker in ["v1:i0819", "v1:i0788", ""] {
        let mut f = Fixture::new();
        f.math(
            ScalarMathOperationAttr::Ex2,
            ScalarMathSubnormalAttr::Ftz,
            marker,
        );
        f.finish();
        for profile in profiles {
            let error = f.translate_with_profile(profile).unwrap_err();
            assert!(
                error.contains("expected intrinsic marker v1:i0820"),
                "{error}"
            );
        }
    }
}

/// Reject FTZ exp2 operands or precision selectors outside its reviewed contract.
#[test]
fn ftz_exp2_rejects_wrong_type_and_selectors() {
    let profiles: [&dyn MlirConsumerProfile; 2] = [
        &CutlassPrimitivesMlir22::new("sm_120a").unwrap(),
        &CutlassFullCuteMlir22::new("sm_120a").unwrap(),
    ];
    for case in 0..5 {
        let mut f = Fixture::new();
        let input = f.arg(if case == 0 { 0 } else { 1 });
        let op = ScalarMathOp::build(
            &mut f.ctx,
            input,
            if case == 1 {
                ScalarMathFormatAttr::F64
            } else {
                ScalarMathFormatAttr::F32
            },
            if case == 2 {
                ScalarMathOperationAttr::Lg2
            } else {
                ScalarMathOperationAttr::Ex2
            },
            if case == 3 {
                ScalarMathPrecisionAttr::Rn
            } else {
                ScalarMathPrecisionAttr::Approx
            },
            if case == 4 {
                ScalarMathSubnormalAttr::Preserve
            } else {
                ScalarMathSubnormalAttr::Ftz
            },
        );
        f.append(op, "v1:i0820");
        f.finish();
        for profile in profiles {
            assert!(f.translate_with_profile(profile).is_err(), "case {case}");
        }
    }
}

/// Check softmax exp/reciprocal operations retain their native approximation and subnormal semantics.
#[test]
fn softmax_math_keeps_approximation_and_subnormal_modes() {
    let mut f = Fixture::new();
    f.math(
        ScalarMathOperationAttr::Ex2,
        ScalarMathSubnormalAttr::Preserve,
        "v1:i0819",
    );
    f.math(
        ScalarMathOperationAttr::Rcp,
        ScalarMathSubnormalAttr::Ftz,
        "v1:i0788",
    );
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("\"llvm.inline_asm\"(%v1)"), "{text}");
    assert!(
        text.contains("asm_string = \"ex2.approx.f32 $0, $1;\""),
        "{text}"
    );
    assert!(text.contains("constraints = \"=f,f\""), "{text}");
    assert!(!text.contains("ex2.approx.ftz"), "{text}");
    assert!(!text.contains("llvm.call"), "{text}");
    assert!(
        text.contains("\"nvvm.rcp.approx.ftz.f\"(%v1) : (f32) -> (f32)"),
        "{text}"
    );
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}

/// Reject unreviewed math modes and fabricated intrinsic identities.
#[test]
fn unsupported_math_and_forged_identity_are_rejected() {
    let mut f = Fixture::new();
    let op = f.math(
        ScalarMathOperationAttr::Ex2,
        ScalarMathSubnormalAttr::Preserve,
        "v1:i0788",
    );
    f.finish();
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0819")
    );
    ScalarMathOp::new(op).set_attr_nvvm_scalar_math_operation(&f.ctx, ScalarMathOperationAttr::Cos);
    assert!(
        f.translate()
            .unwrap_err()
            .contains("supports F32 approximate exp2 and FTZ reciprocal")
    );
}

/// Check the shuffle keeps its membership mask, lane XOR offset, and full-warp clamp.
#[test]
fn butterfly_shuffle_preserves_mask_offset_and_warp_clamp() {
    let mut f = Fixture::new();
    let (mask, value, offset) = (f.arg(0), f.arg(1), f.arg(2));
    let op = ShflSyncBflyF32Op::build(&mut f.ctx, mask, value, offset);
    f.append(op, "v1:i0055");
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("value = 31 : i32"), "{text}");
    assert!(text.contains("\"nvvm.shfl.sync\"(%v0, %v1, %v2,"), "{text}");
    assert!(text.contains("kind = #nvvm<shfl_kind bfly>"), "{text}");
    assert!(!text.contains("return_value_and_is_valid"));
    op.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0054".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0055")
    );
}

/// Check asynchronous-copy grouping preserves explicit commits and constant wait counts.
#[test]
fn cp_async_commit_and_wait_keep_the_group_count() {
    let mut f = Fixture::new();
    let commit = CpAsyncCommitGroupOp::build(&mut f.ctx);
    f.append(commit, "v1:i0094");
    let count = f.constant(3);
    let wait = CpAsyncWaitGroupOp::build(&mut f.ctx, count);
    f.append(wait, "v1:i0096");
    f.finish();
    let text = f.translate().unwrap();
    let commit_pos = text
        .find("\"nvvm.cp.async.commit.group\"() : () -> ()")
        .unwrap();
    let wait_pos = text
        .find("\"nvvm.cp.async.wait.group\"() <{n = 3 : i32}> : () -> ()")
        .unwrap();
    assert!(commit_pos < wait_pos, "{text}");
    wait.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0095".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0096")
    );
}

/// Reject wait selectors that cannot be represented by the hardware immediate contract.
#[test]
fn cp_async_wait_rejects_runtime_or_out_of_range_counts() {
    for count in [None, Some(8)] {
        let mut f = Fixture::new();
        let count = match count {
            Some(n) => f.constant(n),
            None => f.arg(0),
        };
        let wait = CpAsyncWaitGroupOp::build(&mut f.ctx, count);
        f.append(wait, "v1:i0096");
        f.finish();
        assert!(f.translate().is_err());
    }
}

/// Check scalar numeric and asynchronous-copy registrations compose without duplicate ownership.
#[test]
fn numeric_and_async_packs_coexist_on_both_architectures() {
    let mut f = Fixture::new();
    for (operation, subnormal, marker) in [
        (
            ScalarMathOperationAttr::Ex2,
            ScalarMathSubnormalAttr::Preserve,
            "v1:i0819",
        ),
        (
            ScalarMathOperationAttr::Ex2,
            ScalarMathSubnormalAttr::Ftz,
            "v1:i0820",
        ),
        (
            ScalarMathOperationAttr::Lg2,
            ScalarMathSubnormalAttr::Preserve,
            "v1:i0786",
        ),
        (
            ScalarMathOperationAttr::Lg2,
            ScalarMathSubnormalAttr::Ftz,
            "v1:i0787",
        ),
    ] {
        f.math(operation, subnormal, marker);
    }
    let (mask, value, offset) = (f.arg(0), f.arg(1), f.arg(2));
    let shuffle = ShflSyncBflyF32Op::build(&mut f.ctx, mask, value, offset);
    f.append(shuffle, "v1:i0055");
    let commit = CpAsyncCommitGroupOp::build(&mut f.ctx);
    f.append(commit, "v1:i0094");
    let count = f.constant(3);
    let wait = CpAsyncWaitGroupOp::build(&mut f.ctx, count);
    f.append(wait, "v1:i0096");
    f.finish();

    for arch in ["sm_100a", "sm_120a"] {
        let profiles: [&dyn MlirConsumerProfile; 2] = [
            &CutlassPrimitivesMlir22::new(arch).unwrap(),
            &CutlassFullCuteMlir22::new(arch).unwrap(),
        ];
        for profile in profiles {
            let text = f.translate_with_profile(profile).unwrap();
            for name in [
                "llvm.nvvm.lg2.approx.f",
                "llvm.nvvm.lg2.approx.ftz.f",
                "nvvm.cp.async.commit.group",
                "nvvm.cp.async.wait.group",
            ] {
                assert!(text.contains(name), "{arch}: {text}");
            }
            assert!(text.contains("n = 3 : i32"), "{arch}: {text}");
            assert!(text.contains("value = 31 : i32"), "{arch}: {text}");
            if arch == "sm_100a" {
                for name in [
                    "llvm.nvvm.ex2.approx.f32",
                    "llvm.nvvm.ex2.approx.ftz.f32",
                    "llvm.nvvm.shfl.sync.bfly.f32",
                ] {
                    assert!(text.contains(name), "{text}");
                }
                assert!(!text.contains("llvm.inline_asm"), "{text}");
            } else {
                for name in [
                    "ex2.approx.f32 $0, $1;",
                    "ex2.approx.ftz.f32 $0, $1;",
                    "nvvm.shfl.sync",
                ] {
                    assert!(text.contains(name), "{text}");
                }
                assert!(!text.contains("llvm.nvvm.ex2"), "{text}");
                assert!(!text.contains("llvm.nvvm.shfl.sync.bfly"), "{text}");
            }
            assert!(!text.contains("cuda_oxide_intrinsic_marker"), "{text}");
        }
    }
}
