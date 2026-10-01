/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

#[test]
fn test_fast_float_intrinsics_lower_to_explicit_fast_binops() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics;
    use llvm_export::attributes::{FastmathFlags, FastmathFlagsAttr};
    use llvm_export::op_interfaces::FastMathFlags;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::op_interfaces::CallOpInterface;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::{TypeHandle, Typed};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let f32_ty_obj: TypeHandle = f32_ty.into();
    let f64_ty_obj: TypeHandle = f64_ty.into();
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![f32_ty_obj, f32_ty_obj, f64_ty_obj, f64_ty_obj],
    );
    let f32_lhs = entry.deref(&ctx).get_argument(0);
    let f32_rhs = entry.deref(&ctx).get_argument(1);
    let f64_lhs = entry.deref(&ctx).get_argument(2);
    let f64_rhs = entry.deref(&ctx).get_argument(3);

    for (callee, lhs, rhs, result_ty) in [
        (
            rust_intrinsics::CALLEE_FADD_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FSUB_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FMUL_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FDIV_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FREM_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FADD_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FSUB_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FMUL_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FDIV_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FREM_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
    ] {
        let call_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![result_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let call = mir::MirCallOp::new(call_ptr);
        call.set_attr_callee(&ctx, StringAttr::new(callee.to_string()));
        call_ptr.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let explicit_fast_flags: FastmathFlagsAttr = FastmathFlags::FAST.into();
    assert_ne!(
        explicit_fast_flags,
        FastmathFlagsAttr::default(),
        "FastmathFlagsAttr::default() is empty; f*_fast must use explicit fast flags"
    );

    let mut fadd_counts = [0usize; 2];
    let mut fsub_counts = [0usize; 2];
    let mut fmul_counts = [0usize; 2];
    let mut fdiv_counts = [0usize; 2];
    let mut frem_counts = [0usize; 2];

    macro_rules! count_fast_binop {
        ($body_op:expr, $op_ty:ty, $counts:ident, $name:literal) => {
            if let Some(op) = Operation::get_op::<$op_ty>($body_op, &ctx) {
                assert_eq!(
                    op.fast_math_flags(&ctx),
                    explicit_fast_flags,
                    concat!($name, " must carry explicit LLVM fast-math flags")
                );
                let result_ty = op.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
                if result_ty == f32_ty_obj {
                    $counts[0] += 1;
                } else if result_ty == f64_ty_obj {
                    $counts[1] += 1;
                } else {
                    panic!(concat!($name, " lowered to an unexpected result type"));
                }
            }
        };
    }

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                assert!(
                    Operation::get_op::<mir::MirCallOp>(body_op, &ctx).is_none(),
                    "f*_fast placeholder mir.call must not survive MIR lowering"
                );
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                {
                    let callee = sym.to_string();
                    assert!(
                        !callee.starts_with(rust_intrinsics::PLACEHOLDER_PREFIX),
                        "lowered LLVM must not call unresolved Rust intrinsic placeholder `{callee}`"
                    );
                }
                count_fast_binop!(body_op, llvm::FAddOp, fadd_counts, "fadd_fast");
                count_fast_binop!(body_op, llvm::FSubOp, fsub_counts, "fsub_fast");
                count_fast_binop!(body_op, llvm::FMulOp, fmul_counts, "fmul_fast");
                count_fast_binop!(body_op, llvm::FDivOp, fdiv_counts, "fdiv_fast");
                count_fast_binop!(body_op, llvm::FRemOp, frem_counts, "frem_fast");
            }
        }
    }

    assert_eq!(fadd_counts, [1, 1], "fadd_fast must lower for f32 and f64");
    assert_eq!(fsub_counts, [1, 1], "fsub_fast must lower for f32 and f64");
    assert_eq!(fmul_counts, [1, 1], "fmul_fast must lower for f32 and f64");
    assert_eq!(fdiv_counts, [1, 1], "fdiv_fast must lower for f32 and f64");
    assert_eq!(frem_counts, [1, 1], "frem_fast must lower for f32 and f64");

    Ok(())
}

#[test]
fn test_ex2_approx_f16_uses_exact_pure_i16_inline_ptx_on_both_backends() -> Result<(), anyhow::Error>
{
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let i16_ty = IntegerType::get(&ctx, 16, Signedness::Unsigned);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i16_ty.into()]);
        let operand = entry.deref(&ctx).get_argument(0);
        nvvm::ScalarMathOp::build(
            &mut ctx,
            operand,
            nvvm::ScalarMathFormatAttr::F16,
            nvvm::ScalarMathOperationAttr::Ex2,
            nvvm::ScalarMathPrecisionAttr::Approx,
            nvvm::ScalarMathSubnormalAttr::Preserve,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .map_err(|error| anyhow::anyhow!("{error}"))?;

        let inline_asm = lowered_kernel_body(&ctx, module_ptr)
            .into_iter()
            .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
            .collect::<Vec<_>>();
        assert_eq!(inline_asm.len(), 1);
        let inline_asm = &inline_asm[0];
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("ex2.approx.f16 $0, $1;")
        );
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=h,h")
        );
        assert_eq!(llvm::asm_kind(&ctx, inline_asm), llvm::AsmKind::Pure);
        assert!(
            !inline_asm
                .get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );

        let op = inline_asm.get_operation().deref(&ctx);
        assert_eq!(op.get_num_operands(), 1);
        assert_eq!(op.get_num_results(), 1);
        for ty in [
            op.get_operand(0).get_type(&ctx),
            op.get_result(0).get_type(&ctx),
        ] {
            let ty = ty.deref(&ctx);
            let integer = ty
                .downcast_ref::<IntegerType>()
                .expect("f16 values use an i16 transport type");
            assert_eq!(integer.width(), 16);
        }

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("ex2.approx.f16 module exports to LLVM IR");
        assert!(
            ir.contains("call i16 asm \"ex2.approx.f16 $0, $1;\", \"=h,h\"(i16"),
            "{ir}"
        );
        assert!(!ir.contains("asm sideeffect"), "{ir}");
        assert!(!ir.contains("~{memory}"), "{ir}");
    }
    Ok(())
}

#[test]
fn test_ex2_approx_f16_lowering_rejects_wrong_type_and_format() {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut wrong_type_ctx = make_test_ctx();
        let f32_ty = FP32Type::get(&wrong_type_ctx);
        let (module_ptr, entry) = build_test_kernel(&mut wrong_type_ctx, vec![f32_ty.into()]);
        let operand = entry.deref(&wrong_type_ctx).get_argument(0);
        nvvm::ScalarMathOp::build(
            &mut wrong_type_ctx,
            operand,
            nvvm::ScalarMathFormatAttr::F16,
            nvvm::ScalarMathOperationAttr::Ex2,
            nvvm::ScalarMathPrecisionAttr::Approx,
            nvvm::ScalarMathSubnormalAttr::Preserve,
        )
        .insert_at_back(entry, &wrong_type_ctx);
        append_return(&mut wrong_type_ctx, entry);
        let error = mir_lower::lower_mir_to_llvm_with_options(
            &mut wrong_type_ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .expect_err("an f16 operation with an f32 operand must fail closed")
        .to_string();
        assert!(
            error.contains("nvvm.scalar_math types do not match its format"),
            "{error}"
        );

        let mut wrong_format_ctx = make_test_ctx();
        let i16_ty = IntegerType::get(&wrong_format_ctx, 16, Signedness::Unsigned);
        let (module_ptr, entry) = build_test_kernel(&mut wrong_format_ctx, vec![i16_ty.into()]);
        let operand = entry.deref(&wrong_format_ctx).get_argument(0);
        nvvm::ScalarMathOp::build(
            &mut wrong_format_ctx,
            operand,
            nvvm::ScalarMathFormatAttr::F16,
            nvvm::ScalarMathOperationAttr::Ex2,
            nvvm::ScalarMathPrecisionAttr::Approx,
            nvvm::ScalarMathSubnormalAttr::Ftz,
        )
        .insert_at_back(entry, &wrong_format_ctx);
        append_return(&mut wrong_format_ctx, entry);
        let error = mir_lower::lower_mir_to_llvm_with_options(
            &mut wrong_format_ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .expect_err("unadmitted ex2.approx.ftz.f16 must fail closed")
        .to_string();
        assert!(
            error.contains("nvvm.scalar_math variant is not admitted"),
            "{error}"
        );
    }
}

// =============================================================================
// Integer dot product (dp4a / dp2a) lowering tests
// =============================================================================

const DOT_PRODUCT_TYPED_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_idp4a_s_s",
    "llvm_nvvm_idp4a_u_u",
    "llvm_nvvm_idp2a_s_s",
    "llvm_nvvm_idp2a_u_u",
];

const DOT_PRODUCT_PTX: [&str; 4] = [
    "dp4a.s32.s32 $0, $1, $2, $3;",
    "dp4a.u32.u32 $0, $1, $2, $3;",
    "dp2a.lo.s32.s32 $0, $1, $2, $3;",
    "dp2a.lo.u32.u32 $0, $1, $2, $3;",
];

fn lower_all_dot_product_forms(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);
    let operands = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    for op_info in [
        nvvm::Dp4aS32Op::get_concrete_op_info(),
        nvvm::Dp4aU32Op::get_concrete_op_info(),
        nvvm::Dp2aS32Op::get_concrete_op_info(),
        nvvm::Dp2aU32Op::get_concrete_op_info(),
    ] {
        Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            operands.clone(),
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_dot_product_llvm_nvptx_uses_typed_intrinsics_and_low_selector() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;

    let (ctx, module_ptr) = lower_all_dot_product_forms(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut calls = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX dot products must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !callee.starts_with("llvm_nvvm_idp") {
            continue;
        }
        let expected_arity = if callee.contains("idp2a") { 4 } else { 3 };
        assert_eq!(op.deref(&ctx).get_num_operands(), expected_arity);
        if expected_arity == 4 {
            let selector = op.deref(&ctx).get_operand(2);
            let defining_op = selector.defining_op().expect("selector is a constant");
            let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                .expect("selector is an LLVM constant");
            let attribute = constant.get_value(&ctx);
            let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                .downcast_ref::<IntegerAttr>()
                .expect("selector constant is an integer");
            assert_eq!(integer.value().bw(), 1);
            assert_eq!(integer.value().to_u64(), 0, "dp2a must select `.lo`");
        }
        calls.push(callee);
    }
    calls.sort();
    let mut expected = DOT_PRODUCT_TYPED_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_dot_product_libnvvm_uses_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_dot_product_forms(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut inline_ptx = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
            "libNVVM dot products must not use typed intrinsic calls"
        );
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        inline_ptx.push(
            inline_asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
        );
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,r,r,r")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(op.deref(&ctx).get_num_operands(), 3);
        assert_eq!(op.deref(&ctx).get_num_results(), 1);
    }
    inline_ptx.sort();
    let mut expected = DOT_PRODUCT_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(inline_ptx, expected);
    Ok(())
}

// =============================================================================
// Byte permutation lowering tests
// =============================================================================

const PRMT_TYPED_INTRINSICS: [(&str, usize); 7] = [
    ("llvm_nvvm_prmt", 3),
    ("llvm_nvvm_prmt_f4e", 3),
    ("llvm_nvvm_prmt_b4e", 3),
    ("llvm_nvvm_prmt_rc8", 2),
    ("llvm_nvvm_prmt_ecl", 2),
    ("llvm_nvvm_prmt_ecr", 2),
    ("llvm_nvvm_prmt_rc16", 2),
];

const PRMT_INLINE_PTX: [(&str, &str, usize); 7] = [
    ("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.f4e $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.b4e $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.rc8 $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.ecl $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.ecr $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.rc16 $0, $1, 0, $2;", "=r,r,r", 2),
];

fn lower_all_prmt_modes(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);
    let operands = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();

    for mode in [
        nvvm::PrmtModeAttr::Generic,
        nvvm::PrmtModeAttr::F4e,
        nvvm::PrmtModeAttr::B4e,
    ] {
        nvvm::PrmtOp::build(&mut ctx, operands.clone(), mode).insert_at_back(entry, &ctx);
    }
    for mode in [
        nvvm::PrmtModeAttr::Rc8,
        nvvm::PrmtModeAttr::Ecl,
        nvvm::PrmtModeAttr::Ecr,
        nvvm::PrmtModeAttr::Rc16,
    ] {
        nvvm::PrmtOp::build(&mut ctx, vec![operands[0], operands[2]], mode)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_prmt_llvm_nvptx_uses_exact_typed_intrinsics() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_prmt_modes(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut calls = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX byte permutations must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, expected_arity)) = PRMT_TYPED_INTRINSICS
            .iter()
            .find(|(expected, _)| *expected == callee)
        else {
            continue;
        };
        assert_eq!(op.deref(&ctx).get_num_operands(), *expected_arity);
        calls.push((callee, *expected_arity));
    }
    calls.sort();
    let mut expected = PRMT_TYPED_INTRINSICS.map(|(name, arity)| (name.to_owned(), arity));
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_prmt_libnvvm_uses_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_prmt_modes(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut inline_ptx = Vec::new();
    for op in body {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !callee.to_string().starts_with("llvm_nvvm_prmt"),
                "libNVVM byte permutations must not use typed intrinsic calls"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let Some((_, expected_constraints, expected_arity)) = PRMT_INLINE_PTX
            .iter()
            .find(|(expected, _, _)| *expected == template)
        else {
            continue;
        };
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some(*expected_constraints)
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(op.deref(&ctx).get_num_operands(), *expected_arity);
        assert_eq!(op.deref(&ctx).get_num_results(), 1);
        inline_ptx.push((template, *expected_constraints, *expected_arity));
    }
    inline_ptx.sort();
    let mut expected = PRMT_INLINE_PTX
        .map(|(template, constraints, arity)| (template.to_owned(), constraints, arity));
    expected.sort();
    assert_eq!(inline_ptx, expected);
    Ok(())
}

#[test]
fn test_generated_packed_arithmetic_lowers_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, usize, &str, &str); 18] = [
        (
            nvvm::FmaBf16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::FmaReluBf16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.relu.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::AddBf16x2Op::get_concrete_op_info(),
            2,
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::SubBf16x2Op::get_concrete_op_info(),
            2,
            "sub.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MulBf16x2Op::get_concrete_op_info(),
            2,
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MinBf16x2Op::get_concrete_op_info(),
            2,
            "min.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MaxBf16x2Op::get_concrete_op_info(),
            2,
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::NegBf16x2Op::get_concrete_op_info(),
            1,
            "neg.bf16x2 $0, $1;",
            "=r,r",
        ),
        (
            nvvm::AbsBf16x2Op::get_concrete_op_info(),
            1,
            "abs.bf16x2 $0, $1;",
            "=r,r",
        ),
        (
            nvvm::FmaF16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::FmaReluF16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.relu.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::AddF16x2Op::get_concrete_op_info(),
            2,
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::SubF16x2Op::get_concrete_op_info(),
            2,
            "sub.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MulF16x2Op::get_concrete_op_info(),
            2,
            "mul.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MinF16x2Op::get_concrete_op_info(),
            2,
            "min.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MaxF16x2Op::get_concrete_op_info(),
            2,
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::NegF16x2Op::get_concrete_op_info(),
            1,
            "neg.f16x2 $0, $1;",
            "=r,r",
        ),
        (
            nvvm::AbsF16x2Op::get_concrete_op_info(),
            1,
            "abs.f16x2 $0, $1;",
            "=r,r",
        ),
    ];

    let operands = [
        entry.deref(&ctx).get_argument(0),
        entry.deref(&ctx).get_argument(1),
        entry.deref(&ctx).get_argument(2),
    ];
    for &(op_info, operand_count, _, _) in &cases {
        let op = Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            operands[..operand_count].to_vec(),
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut lowered = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let inline_asm_op = inline_asm.get_operation();
                let operand_count = inline_asm_op.deref(&ctx).operands().count();
                let result_count = inline_asm_op.deref(&ctx).get_num_results();
                lowered.push((
                    inline_asm
                        .get_attr_llvm_inline_asm_template(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("packed inline asm must have a template"),
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("packed inline asm must have constraints"),
                    llvm::asm_kind_opt(&ctx, &inline_asm),
                    inline_asm
                        .get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent")),
                    operand_count,
                    result_count,
                ));
            }
        }
    }

    assert_eq!(
        lowered.len(),
        cases.len(),
        "each packed operation must lower to exactly one inline-asm op"
    );
    for &(_, expected_operand_count, expected_template, expected_constraints) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _, _)| template == expected_template)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact `{expected_template}` lowering"
        );
        let (_, constraints, kind, convergent, operand_count, result_count) = matches[0];
        assert_eq!(constraints, expected_constraints, "{expected_template}");
        assert_eq!(*kind, Some(llvm::AsmKind::Pure), "{expected_template}");
        assert!(!*convergent, "{expected_template}");
        assert_eq!(
            *operand_count, expected_operand_count,
            "{expected_template} input arity"
        );
        assert_eq!(*result_count, 1, "{expected_template} result arity");
    }

    Ok(())
}

#[test]
fn test_generated_integer_minmax_lowers_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into()]);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, &str); 8] = [
        (
            nvvm::MinReluS32Op::get_concrete_op_info(),
            "min.relu.s32 $0, $1, $2;",
        ),
        (
            nvvm::MaxReluS32Op::get_concrete_op_info(),
            "max.relu.s32 $0, $1, $2;",
        ),
        (
            nvvm::MinS16x2Op::get_concrete_op_info(),
            "min.s16x2 $0, $1, $2;",
        ),
        (
            nvvm::MaxS16x2Op::get_concrete_op_info(),
            "max.s16x2 $0, $1, $2;",
        ),
        (
            nvvm::MinU16x2Op::get_concrete_op_info(),
            "min.u16x2 $0, $1, $2;",
        ),
        (
            nvvm::MaxU16x2Op::get_concrete_op_info(),
            "max.u16x2 $0, $1, $2;",
        ),
        (
            nvvm::MinReluS16x2Op::get_concrete_op_info(),
            "min.relu.s16x2 $0, $1, $2;",
        ),
        (
            nvvm::MaxReluS16x2Op::get_concrete_op_info(),
            "max.relu.s16x2 $0, $1, $2;",
        ),
    ];

    let operands = [
        entry.deref(&ctx).get_argument(0),
        entry.deref(&ctx).get_argument(1),
    ];
    for &(op_info, _) in &cases {
        let op = Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            operands.to_vec(),
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut lowered = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let inline_asm_op = inline_asm.get_operation();
                let operand_count = inline_asm_op.deref(&ctx).operands().count();
                let result_count = inline_asm_op.deref(&ctx).get_num_results();
                lowered.push((
                    inline_asm
                        .get_attr_llvm_inline_asm_template(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("integer min/max inline asm must have a template"),
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("integer min/max inline asm must have constraints"),
                    llvm::asm_kind_opt(&ctx, &inline_asm),
                    inline_asm
                        .get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent")),
                    operand_count,
                    result_count,
                ));
            }
        }
    }

    assert_eq!(
        lowered.len(),
        cases.len(),
        "each integer min/max operation must lower to exactly one inline-asm op"
    );
    for &(_, expected_template) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _, _)| template == expected_template)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact `{expected_template}` lowering"
        );
        let (_, constraints, kind, convergent, operand_count, result_count) = matches[0];
        assert_eq!(constraints, "=r,r,r", "{expected_template}");
        assert_eq!(*kind, Some(llvm::AsmKind::Pure), "{expected_template}");
        assert!(!*convergent, "{expected_template}");
        assert_eq!(*operand_count, 2, "{expected_template} input arity");
        assert_eq!(*result_count, 1, "{expected_template} result arity");
    }

    Ok(())
}

#[test]
fn test_generated_packed_conversions_lower_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);
    let low = entry.deref(&ctx).get_argument(0);
    let high = entry.deref(&ctx).get_argument(1);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, &str); 6] = [
        (
            nvvm::CvtF32x2Bf16x2Op::get_concrete_op_info(),
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtF16x2F32Op::get_concrete_op_info(),
            "cvt.rn.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRzF16x2F32Op::get_concrete_op_info(),
            "cvt.rz.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRnReluF16x2F32Op::get_concrete_op_info(),
            "cvt.rn.relu.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRnReluBf16x2F32Op::get_concrete_op_info(),
            "cvt.rn.relu.bf16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRzBf16x2F32Op::get_concrete_op_info(),
            "cvt.rz.bf16x2.f32 $0, $2, $1;",
        ),
    ];
    for &(op_info, _) in &cases {
        let op = Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            vec![low, high],
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut lowered = Vec::new();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let region = func.get_operation().deref(&ctx).get_region(0);
        for block in region.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .expect("packed conversion must have an asm template");
                if !template.starts_with("cvt.") {
                    continue;
                }
                lowered.push((
                    template,
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone())),
                    asm.get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent")),
                    llvm::asm_kind_opt(&ctx, &asm),
                    asm.get_operation().deref(&ctx).operands().count(),
                    asm.get_operation().deref(&ctx).get_num_results(),
                ));
            }
        }
    }

    assert_eq!(
        lowered.len(),
        cases.len(),
        "each packed conversion must lower to one inline-asm op"
    );
    for &(_, expected_template) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _, _)| template == expected_template)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact `{expected_template}` lowering"
        );
        let (_, constraints, convergent, kind, operands, results) = matches[0];
        assert_eq!(
            constraints.as_deref(),
            Some("=r,f,f"),
            "{expected_template}"
        );
        assert!(!*convergent, "{expected_template}");
        assert_eq!(*kind, Some(llvm::AsmKind::Pure), "{expected_template}");
        assert_eq!(*operands, 2, "{expected_template} input arity");
        assert_eq!(*results, 1, "{expected_template} result arity");
    }

    Ok(())
}

const FP8_CONVERSION_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_ff_to_e4m3x2_rn",
    "llvm_nvvm_ff_to_e4m3x2_rn_relu",
    "llvm_nvvm_ff_to_e5m2x2_rn",
    "llvm_nvvm_ff_to_e5m2x2_rn_relu",
];

const FP8_CONVERSION_PTX: [&str; 4] = [
    "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.relu.e4m3x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.relu.e5m2x2.f32 $0, $2, $1;",
];

fn lower_all_fp8_conversions(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);
    let low = entry.deref(&ctx).get_argument(0);
    let high = entry.deref(&ctx).get_argument(1);

    nvvm::CvtRnSatfiniteE4m3x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteReluE4m3x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteE5m2x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteReluE5m2x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_fp8_conversions_llvm_nvptx_use_exact_typed_calls() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_fp8_conversions(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX FP8 conversions must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !FP8_CONVERSION_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }
        let call_op = call.get_operation();
        assert_eq!(call_op.deref(&ctx).get_num_operands(), 2);
        assert_eq!(call_op.deref(&ctx).get_num_results(), 1);
        let block = call_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            call_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(1)
        );
        assert_eq!(
            call_op.deref(&ctx).get_operand(1),
            block.deref(&ctx).get_argument(0)
        );
        let result_ty = call_op.deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            result_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("FP8 conversion result is an integer")
                .width(),
            16
        );
        calls.push(callee);
    }
    calls.sort();
    let mut expected = FP8_CONVERSION_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_fp8_conversions_libnvvm_use_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_fp8_conversions(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut templates = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !FP8_CONVERSION_INTRINSICS.contains(&callee.as_ref()),
                "libNVVM FP8 conversions must not use typed intrinsics"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        if !FP8_CONVERSION_PTX.contains(&template.as_str()) {
            continue;
        }
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=h,f,f")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert!(
            !inline_asm
                .get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
        let asm_op = inline_asm.get_operation();
        assert_eq!(asm_op.deref(&ctx).get_num_operands(), 2);
        assert_eq!(asm_op.deref(&ctx).get_num_results(), 1);
        let result_ty = asm_op.deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            result_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("FP8 conversion result is an integer")
                .width(),
            16
        );
        templates.push(template);
    }
    templates.sort();
    let mut expected = FP8_CONVERSION_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(templates, expected);
    Ok(())
}

const SCALAR_CONVERSION_INTRINSICS: [&str; 10] = [
    "llvm_nvvm_f2tf32_rna",
    "llvm_nvvm_f2tf32_rna_satfinite",
    "llvm_nvvm_f2tf32_rn",
    "llvm_nvvm_f2tf32_rn_relu",
    "llvm_nvvm_f2tf32_rn_satfinite",
    "llvm_nvvm_f2tf32_rn_relu_satfinite",
    "llvm_nvvm_f2tf32_rz",
    "llvm_nvvm_f2tf32_rz_relu",
    "llvm_nvvm_f2tf32_rz_satfinite",
    "llvm_nvvm_f2tf32_rz_relu_satfinite",
];

const SCALAR_CONVERSION_PTX: [&str; 10] = [
    "cvt.rna.tf32.f32 $0, $1;",
    "cvt.rna.satfinite.tf32.f32 $0, $1;",
    "cvt.rn.tf32.f32 $0, $1;",
    "cvt.rn.relu.tf32.f32 $0, $1;",
    "cvt.rn.satfinite.tf32.f32 $0, $1;",
    "cvt.rn.relu.satfinite.tf32.f32 $0, $1;",
    "cvt.rz.tf32.f32 $0, $1;",
    "cvt.rz.relu.tf32.f32 $0, $1;",
    "cvt.rz.satfinite.tf32.f32 $0, $1;",
    "cvt.rz.relu.satfinite.tf32.f32 $0, $1;",
];

fn lower_all_scalar_conversions(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into()]);
    let value = entry.deref(&ctx).get_argument(0);

    let variants = [
        (
            nvvm::ScalarConversionRoundingAttr::NearestAway,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestAway,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::Relu,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::ReluSatfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::Relu,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::ReluSatfinite,
        ),
    ];
    for (rounding, saturation) in variants {
        nvvm::ScalarConversionOp::build(&mut ctx, value, rounding, saturation)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_scalar_conversions_llvm_nvptx_use_exact_typed_calls() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::builtin::types::{FP32Type, IntegerType};

    let (ctx, module_ptr) = lower_all_scalar_conversions(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX scalar conversions must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !SCALAR_CONVERSION_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }

        let call_op = call.get_operation();
        assert_eq!(call_op.deref(&ctx).get_num_operands(), 1, "{callee}");
        assert_eq!(call_op.deref(&ctx).get_num_results(), 1, "{callee}");
        let block = call_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            call_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(0),
            "{callee} source operand"
        );

        let function_ty = call.callee_type(&ctx);
        let function_ty = function_ty.deref(&ctx);
        let function_ty = function_ty
            .downcast_ref::<llvm_types::FuncType>()
            .expect("scalar conversion callee has an LLVM function type");
        assert_eq!(function_ty.arg_types().len(), 1, "{callee}");
        assert!(
            function_ty.arg_types()[0]
                .deref(&ctx)
                .downcast_ref::<FP32Type>()
                .is_some(),
            "{callee} source type"
        );
        assert_eq!(
            function_ty
                .result_type()
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("scalar conversion result is an integer")
                .width(),
            32,
            "{callee} result type"
        );
        calls.push(callee);
    }
    calls.sort();
    let mut expected = SCALAR_CONVERSION_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut declarations: Vec<_> = module_block
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
        .map(|func| func.get_symbol_name(&ctx).to_string())
        .filter(|name| SCALAR_CONVERSION_INTRINSICS.contains(&name.as_str()))
        .collect();
    declarations.sort();
    assert_eq!(declarations, expected);
    Ok(())
}

#[test]
fn test_scalar_conversions_libnvvm_use_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_scalar_conversions(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut templates = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !SCALAR_CONVERSION_INTRINSICS.contains(&callee.as_ref()),
                "libNVVM scalar conversions must not use typed intrinsics"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert!(
            SCALAR_CONVERSION_PTX.contains(&template.as_str()),
            "unexpected scalar conversion template `{template}`"
        );
        assert_eq!(
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,f"),
            "{template}"
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert!(
            !inline_asm
                .get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent")),
            "{template}"
        );

        let asm_op = inline_asm.get_operation();
        assert_eq!(asm_op.deref(&ctx).get_num_operands(), 1, "{template}");
        assert_eq!(asm_op.deref(&ctx).get_num_results(), 1, "{template}");
        let block = asm_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            asm_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(0),
            "{template} source operand"
        );
        assert_eq!(
            asm_op
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("scalar conversion result is an integer")
                .width(),
            32,
            "{template} result type"
        );
        templates.push(template);
    }
    templates.sort();
    let mut expected = SCALAR_CONVERSION_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(templates, expected);
    Ok(())
}

#[test]
fn test_scalar_conversion_invalid_variant_fails_closed() {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into()]);
    let value = entry.deref(&ctx).get_argument(0);
    nvvm::ScalarConversionOp::build(
        &mut ctx,
        value,
        nvvm::ScalarConversionRoundingAttr::NearestAway,
        nvvm::ScalarConversionSaturationAttr::Relu,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let result = mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LlvmNvptx,
            ..Default::default()
        },
    );
    let error = result.expect_err("unadmitted scalar conversion must not lower");
    assert!(
        error.to_string().contains("scalar_conversion"),
        "unexpected error: {error}"
    );
}

fn lower_representative_scalar_arithmetic(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into(), f32_ty.into()]);
    let args: Vec<_> = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    for (operation, saturation, operands) in [
        (
            nvvm::ScalarArithmeticOperationAttr::Add,
            nvvm::ScalarArithmeticSaturationAttr::None,
            vec![args[0], args[1]],
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Add,
            nvvm::ScalarArithmeticSaturationAttr::Sat,
            vec![args[0], args[1]],
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Fma,
            nvvm::ScalarArithmeticSaturationAttr::None,
            args.clone(),
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Fma,
            nvvm::ScalarArithmeticSaturationAttr::Sat,
            args,
        ),
    ] {
        nvvm::ScalarArithmeticOp::build(
            &mut ctx,
            operands,
            nvvm::ScalarArithmeticFormatAttr::F32,
            operation,
            nvvm::ScalarArithmeticRoundingAttr::Rn,
            nvvm::ScalarArithmeticSubnormalAttr::Preserve,
            saturation,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

fn is_representative_scalar_arithmetic_intrinsic(name: &str) -> bool {
    name.starts_with("llvm_nvvm_add_rn") || name.starts_with("llvm_nvvm_fma_rn")
}

#[test]
fn test_scalar_arithmetic_llvm_uses_inline_ptx_only_for_saturation() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) =
        lower_representative_scalar_arithmetic(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    let mut inline_ptx = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            && is_representative_scalar_arithmetic_intrinsic(callee.as_ref())
        {
            calls.push(callee.to_string());
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let constraints = inline_asm
            .get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert!(
            !inline_asm
                .get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
        inline_ptx.push((template, constraints));
    }

    calls.sort();
    assert_eq!(calls, ["llvm_nvvm_add_rn_f", "llvm_nvvm_fma_rn_f"]);
    inline_ptx.sort();
    assert_eq!(
        inline_ptx,
        [
            ("add.rn.sat.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("fma.rn.sat.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
        ]
    );

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut declarations: Vec<_> = module_block
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
        .map(|func| func.get_symbol_name(&ctx).to_string())
        .filter(|name| is_representative_scalar_arithmetic_intrinsic(name))
        .collect();
    declarations.sort();
    assert_eq!(declarations, calls);
    Ok(())
}

#[test]
fn test_scalar_arithmetic_libnvvm_uses_exact_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) =
        lower_representative_scalar_arithmetic(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut calls = Vec::new();
    let mut inline_ptx = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            && is_representative_scalar_arithmetic_intrinsic(callee.as_ref())
        {
            calls.push(callee.to_string());
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_llvm_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let constraints = inline_asm
            .get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        inline_ptx.push((template, constraints));
    }

    assert!(calls.is_empty());
    inline_ptx.sort();
    assert_eq!(
        inline_ptx,
        [
            ("add.rn.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("add.rn.sat.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("fma.rn.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
            ("fma.rn.sat.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
        ]
    );
    Ok(())
}
