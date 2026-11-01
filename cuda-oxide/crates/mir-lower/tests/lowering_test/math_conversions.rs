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
fn test_j1_infinity_correction_on_both_backends() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            let mut ctx = make_test_ctx();
            let ty: TypeHandle = if wide {
                FP64Type::get(&ctx).into()
            } else {
                FP32Type::get(&ctx).into()
            };
            let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ty]);
            let x = entry.deref(&ctx).get_argument(0);
            let call = Operation::new(
                &mut ctx,
                mir::MirCallOp::get_concrete_op_info(),
                vec![ty],
                vec![x],
                vec![],
                0,
            );
            let callee = if wide {
                ri::CALLEE_J1_F64
            } else {
                ri::CALLEE_J1_F32
            };
            mir::MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new(callee.to_string()));
            call.insert_at_back(entry, &ctx);
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
            pliron::operation::verify_operation(module_ptr, &ctx)
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
            let ir =
                llvm_export::export::export_module_to_string(&ctx, &module).expect("j1 IR export");
            let symbol = if wide { "__nv_j1" } else { "__nv_j1f" };
            assert!(
                ir.contains(&format!(
                    "call {} @{symbol}(",
                    if wide { "double" } else { "float" }
                )),
                "{ir}"
            );
            assert!(
                ir.contains("icmp eq") && ir.contains("select i1") && ir.contains("and i"),
                "{ir}"
            );
            assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
        }
    }
    Ok(())
}

#[test]
fn test_j1_rejects_malformed_scalar_signatures() {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::TypeHandle;

    for wide in [false, true] {
        for bad in 0..3 {
            let mut ctx = make_test_ctx();
            let ty: TypeHandle = if wide {
                FP64Type::get(&ctx).into()
            } else {
                FP32Type::get(&ctx).into()
            };
            let other: TypeHandle = if wide {
                FP32Type::get(&ctx).into()
            } else {
                FP64Type::get(&ctx).into()
            };
            let (module, entry) = build_test_kernel(&mut ctx, vec![ty, other]);
            let x = entry.deref(&ctx).get_argument(0);
            let y = entry.deref(&ctx).get_argument(1);
            let operands = match bad {
                0 => vec![],
                1 => vec![y],
                _ => vec![x],
            };
            let result = if bad == 2 { other } else { ty };
            let call = Operation::new(
                &mut ctx,
                mir::MirCallOp::get_concrete_op_info(),
                vec![result],
                operands,
                vec![],
                0,
            );
            let callee = if wide {
                ri::CALLEE_J1_F64
            } else {
                ri::CALLEE_J1_F32
            };
            mir::MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new(callee.to_string()));
            call.insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);
            assert!(mir_lower::lower_mir_to_llvm(&mut ctx, module).is_err());
        }
    }
}

#[test]
fn test_c23_minmax_uses_ieee_order_nan_and_signed_zero_on_both_backends()
-> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for op_id in 0..4 {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty, float_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let y = entry.deref(&ctx).get_argument(1);
                let placeholder = match (wide, op_id) {
                    (false, 0) => ri::CALLEE_FMAXIMUM_F32,
                    (true, 0) => ri::CALLEE_FMAXIMUM_F64,
                    (false, 1) => ri::CALLEE_FMINIMUM_F32,
                    (true, 1) => ri::CALLEE_FMINIMUM_F64,
                    (false, 2) => ri::CALLEE_FMAXIMUM_NUM_F32,
                    (true, 2) => ri::CALLEE_FMAXIMUM_NUM_F64,
                    (false, _) => ri::CALLEE_FMINIMUM_NUM_F32,
                    (true, _) => ri::CALLEE_FMINIMUM_NUM_F64,
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![float_ty],
                    vec![x, y],
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call)
                    .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                call.insert_at_back(entry, &ctx);
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
                pliron::operation::verify_operation(module_ptr, &ctx)
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("C23 min/max IR export");
                assert!(ir.contains("fcmp olt") && ir.contains("fcmp ogt"), "{ir}");
                assert_eq!(ir.matches("fcmp uno").count(), 2, "{ir}");
                assert!(ir.contains("icmp eq"), "{ir}");
                assert_eq!(ir.matches("select i1").count(), 5, "{ir}");
                assert!(
                    !ir.contains(ri::PLACEHOLDER_PREFIX) && !ir.contains("@__nv_"),
                    "{ir}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_c23_minmax_rejects_malformed_signatures() {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for invalid in 0..5 {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let other_ty: TypeHandle = if wide {
                    FP32Type::get(&ctx).into()
                } else {
                    FP64Type::get(&ctx).into()
                };
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty, other_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let other = entry.deref(&ctx).get_argument(1);
                let args = match invalid {
                    0 => vec![],
                    1 => vec![x],
                    2 => vec![x, x, x],
                    3 => vec![x, other],
                    _ => vec![x, x],
                };
                let result = if invalid == 4 { other_ty } else { float_ty };
                let placeholder = if wide {
                    ri::CALLEE_FMAXIMUM_NUM_F64
                } else {
                    ri::CALLEE_FMAXIMUM_F32
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![result],
                    args,
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call)
                    .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                call.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);
                let error = mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .expect_err("malformed C23 min/max must fail")
                .to_string();
                let expected = if invalid <= 2 {
                    "requires 2 operand(s)"
                } else {
                    "C23 min/max signature mismatch"
                };
                assert!(error.contains(expected), "{invalid}: {error}");
            }
        }
    }
}

#[test]
fn test_lgamma_r_composed_tuple_and_special_cases_on_both_backends() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for reordered in [false, true] {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let int_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                let tuple: TypeHandle = if reordered {
                    MirTupleType::get_with_layout(
                        &mut ctx,
                        vec![float_ty, int_ty],
                        vec![1, 0],
                        vec![if wide { 8 } else { 4 }, 0],
                        if wide { 16 } else { 8 },
                        if wide { 8 } else { 4 },
                    )
                    .into()
                } else {
                    MirTupleType::get(&mut ctx, vec![float_ty, int_ty]).into()
                };
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![tuple],
                    vec![x],
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call).set_attr_callee(
                    &ctx,
                    StringAttr::new(
                        if wide {
                            ri::CALLEE_LGAMMA_R_F64
                        } else {
                            ri::CALLEE_LGAMMA_R_F32
                        }
                        .to_string(),
                    ),
                );
                call.insert_at_back(entry, &ctx);
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
                pliron::operation::verify_operation(module_ptr, &ctx)
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("lgamma_r module must export");
                let scalar = if wide { "double" } else { "float" };
                let suffix = if wide { "" } else { "f" };
                for signature in [
                    format!("declare {scalar} @__nv_lgamma{suffix}({scalar})"),
                    format!("declare {scalar} @__nv_floor{suffix}({scalar})"),
                    format!("declare {scalar} @__nv_fmod{suffix}({scalar}, {scalar})"),
                ] {
                    assert!(ir.contains(&signature), "{signature}: {ir}");
                }
                assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
                assert_eq!(ir.matches("insertvalue").count(), 2, "{ir}");
                assert!(ir.matches("select i1").count() >= 5, "{ir}");
                assert!(
                    ir.contains("fcmp olt") && ir.contains("fcmp oeq") && ir.contains("icmp eq"),
                    "{ir}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_lgamma_r_rejects_malformed_tuple_and_operands() {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for invalid in 0..7 {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let other_ty: TypeHandle = if wide {
                    FP32Type::get(&ctx).into()
                } else {
                    FP64Type::get(&ctx).into()
                };
                let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty, other_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let other = entry.deref(&ctx).get_argument(1);
                let fields = match invalid {
                    3 => vec![float_ty, i64_ty],
                    4 => vec![other_ty, i32_ty],
                    5 => vec![i32_ty, float_ty],
                    _ => vec![float_ty, i32_ty],
                };
                let tuple: TypeHandle = MirTupleType::get(&mut ctx, fields).into();
                let result = if invalid == 2 { float_ty } else { tuple };
                let args = match invalid {
                    0 => vec![],
                    1 => vec![x, x],
                    6 => vec![other],
                    _ => vec![x],
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![result],
                    args,
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call).set_attr_callee(
                    &ctx,
                    StringAttr::new(
                        if wide {
                            ri::CALLEE_LGAMMA_R_F64
                        } else {
                            ri::CALLEE_LGAMMA_R_F32
                        }
                        .to_string(),
                    ),
                );
                call.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);
                let error = mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .expect_err("malformed lgamma_r must fail")
                .to_string();
                let expected = match invalid {
                    0 | 1 => "requires 1 operand(s)",
                    2 => "requires a (float, i32) tuple",
                    _ => "signature mismatch",
                };
                assert!(error.contains(expected), "case {invalid}: {error}");
            }
        }
    }
}

#[test]
fn test_modf_remquo_out_parameter_lowering_and_tuple_layout() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for remquo in [false, true] {
                for reordered in [false, true] {
                    let mut ctx = make_test_ctx();
                    let float_ty: TypeHandle = if wide {
                        FP64Type::get(&ctx).into()
                    } else {
                        FP32Type::get(&ctx).into()
                    };
                    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                    let fields = vec![float_ty, if remquo { i32_ty } else { float_ty }];
                    let tuple_ty: TypeHandle = if reordered {
                        MirTupleType::get_with_layout(
                            &mut ctx,
                            fields,
                            vec![1, 0],
                            vec![if wide { 8 } else { 4 }, 0],
                            if wide { 16 } else { 8 },
                            if wide { 8 } else { 4 },
                        )
                        .into()
                    } else {
                        MirTupleType::get(&mut ctx, fields).into()
                    };
                    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty]);
                    let x = entry.deref(&ctx).get_argument(0);
                    let placeholder = match (remquo, wide) {
                        (false, false) => ri::CALLEE_MODF_F32,
                        (false, true) => ri::CALLEE_MODF_F64,
                        (true, false) => ri::CALLEE_REMQUO_F32,
                        (true, true) => ri::CALLEE_REMQUO_F64,
                    };
                    let args = if remquo { vec![x, x] } else { vec![x] };
                    let call = Operation::new(
                        &mut ctx,
                        mir::MirCallOp::get_concrete_op_info(),
                        vec![tuple_ty],
                        args,
                        vec![],
                        0,
                    );
                    mir::MirCallOp::new(call)
                        .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                    call.insert_at_back(entry, &ctx);
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
                    pliron::operation::verify_operation(module_ptr, &ctx)
                        .map_err(|error| anyhow::anyhow!("{error}"))?;
                    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                        .expect("out-parameter module must export");
                    let name = match (remquo, wide) {
                        (false, false) => "__nv_modff",
                        (false, true) => "__nv_modf",
                        (true, false) => "__nv_remquof",
                        (true, true) => "__nv_remquo",
                    };
                    let scalar = if wide { "double" } else { "float" };
                    let signature = if remquo {
                        format!("declare {scalar} @{name}({scalar}, {scalar}, ptr)")
                    } else {
                        format!("declare {scalar} @{name}({scalar}, ptr)")
                    };
                    assert!(ir.contains(&signature), "{signature}: {ir}");
                    assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
                    assert_eq!(ir.matches("alloca").count(), 1, "{ir}");
                    assert_eq!(ir.matches("insertvalue").count(), 2, "{ir}");
                    assert_eq!(ir.matches("select i1").count(), usize::from(remquo), "{ir}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_modf_remquo_reject_malformed_calls() {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for remquo in [false, true] {
                for invalid in 0..4 {
                    let mut ctx = make_test_ctx();
                    let float_ty: TypeHandle = if wide {
                        FP64Type::get(&ctx).into()
                    } else {
                        FP32Type::get(&ctx).into()
                    };
                    let other_ty: TypeHandle = if wide {
                        FP32Type::get(&ctx).into()
                    } else {
                        FP64Type::get(&ctx).into()
                    };
                    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                    let fields = vec![float_ty, if remquo { i32_ty } else { float_ty }];
                    let tuple: TypeHandle = MirTupleType::get(
                        &mut ctx,
                        if invalid == 2 {
                            vec![other_ty, float_ty]
                        } else {
                            fields
                        },
                    )
                    .into();
                    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty, other_ty]);
                    let x = entry.deref(&ctx).get_argument(0);
                    let other = entry.deref(&ctx).get_argument(1);
                    let placeholder = match (remquo, wide) {
                        (false, false) => ri::CALLEE_MODF_F32,
                        (false, true) => ri::CALLEE_MODF_F64,
                        (true, false) => ri::CALLEE_REMQUO_F32,
                        (true, true) => ri::CALLEE_REMQUO_F64,
                    };
                    let mut args = if remquo { vec![x, x] } else { vec![x] };
                    if invalid == 0 {
                        args.pop();
                    } else if invalid == 3 {
                        *args.last_mut().unwrap() = other;
                    }
                    let call = Operation::new(
                        &mut ctx,
                        mir::MirCallOp::get_concrete_op_info(),
                        vec![if invalid == 1 { float_ty } else { tuple }],
                        args,
                        vec![],
                        0,
                    );
                    mir::MirCallOp::new(call)
                        .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                    call.insert_at_back(entry, &ctx);
                    append_return(&mut ctx, entry);
                    let error = mir_lower::lower_mir_to_llvm_with_options(
                        &mut ctx,
                        module_ptr,
                        mir_lower::LoweringOptions {
                            intrinsic_backend: backend,
                            ..Default::default()
                        },
                    )
                    .expect_err("malformed tuple math ABI must fail")
                    .to_string();
                    let expected = if invalid == 0 {
                        "requires"
                    } else if invalid == 1 {
                        "requires a tuple result"
                    } else {
                        "signature mismatch"
                    };
                    assert!(error.contains(expected), "{invalid}: {error}");
                }
            }
        }
    }
}

#[test]
fn test_frexp_out_parameter_lowering_on_both_backends() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::{TypeHandle, Typed};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            // The reordered case also carries real padding. Tuple packing
            // must use the declared-field map rather than hardcoded slots.
            for reordered in [false, true] {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let i32_mir: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                let tuple_ty: TypeHandle = if reordered {
                    MirTupleType::get_with_layout(
                        &mut ctx,
                        vec![float_ty, i32_mir],
                        vec![1, 0],
                        vec![if wide { 8 } else { 4 }, 0],
                        if wide { 16 } else { 8 },
                        if wide { 8 } else { 4 },
                    )
                    .into()
                } else {
                    MirTupleType::get(&mut ctx, vec![float_ty, i32_mir]).into()
                };
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let parent = entry.deref(&ctx).get_parent_op(&ctx).unwrap();
                let region = parent.deref(&ctx).get_region(0);
                let body = BasicBlock::new(&mut ctx, None, vec![]);
                body.insert_at_back(region, &ctx);
                Operation::new(
                    &mut ctx,
                    mir::MirGotoOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![body],
                    0,
                )
                .insert_at_back(entry, &ctx);
                let placeholder = if wide {
                    ri::CALLEE_FREXP_F64
                } else {
                    ri::CALLEE_FREXP_F32
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![tuple_ty],
                    vec![x],
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call)
                    .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                call.insert_at_back(body, &ctx);
                append_return(&mut ctx, body);
                mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .map_err(|error| anyhow::anyhow!("{error}"))?;
                pliron::operation::verify_operation(module_ptr, &ctx)
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                let module_block = module_ptr
                    .deref(&ctx)
                    .get_region(0)
                    .deref(&ctx)
                    .iter(&ctx)
                    .next()
                    .unwrap();
                let func = module_block
                    .deref(&ctx)
                    .iter(&ctx)
                    .find_map(|op| {
                        let func = Operation::get_op::<llvm::FuncOp>(op, &ctx)?;
                        (func.get_symbol_name(&ctx).to_string() == "kernel_func").then_some(func)
                    })
                    .unwrap();
                let blocks = func
                    .get_operation()
                    .deref(&ctx)
                    .get_region(0)
                    .deref(&ctx)
                    .iter(&ctx)
                    .collect::<Vec<_>>();
                let operations = blocks
                    .iter()
                    .flat_map(|block| block.deref(&ctx).iter(&ctx))
                    .collect::<Vec<_>>();
                let allocas = operations
                    .iter()
                    .filter(|&&op| Operation::get_op::<llvm::AllocaOp>(op, &ctx).is_some())
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(allocas.len(), 1);
                assert_eq!(
                    allocas[0].deref(&ctx).get_parent_block(),
                    Some(blocks[0]),
                    "out slot must be static even for a call outside the entry block"
                );
                let calls = operations
                    .iter()
                    .filter_map(|&op| Operation::get_op::<llvm::CallOp>(op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(calls.len(), 1);
                let call = &calls[0];
                let CallOpCallable::Direct(symbol) = call.callee(&ctx) else {
                    panic!("expected a direct libdevice call")
                };
                assert_eq!(
                    symbol.to_string(),
                    if wide { "__nv_frexp" } else { "__nv_frexpf" }
                );
                let call_op = call.get_operation().deref(&ctx);
                assert_eq!(call_op.get_num_operands(), 2);
                assert_eq!(call_op.get_result(0).get_type(&ctx), float_ty);
                let out_ptr = call_op.get_operand(1);
                assert_eq!(out_ptr, allocas[0].deref(&ctx).get_result(0));
                assert!(
                    operations
                        .iter()
                        .any(|&op| Operation::get_op::<llvm::StoreOp>(op, &ctx).is_some()
                            && op.deref(&ctx).get_operand(1) == out_ptr)
                );
                assert!(
                    operations
                        .iter()
                        .any(|&op| Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some()
                            && op.deref(&ctx).get_operand(0) == out_ptr)
                );
                assert_eq!(
                    operations
                        .iter()
                        .filter(|&&op| Operation::get_op::<llvm::SelectOp>(op, &ctx).is_some())
                        .count(),
                    2,
                    "non-finite fraction and exponent must both be normalized"
                );
                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("frexp module must export");
                assert!(
                    ir.contains(if wide {
                        "declare double @__nv_frexp(double, ptr)"
                    } else {
                        "declare float @__nv_frexpf(float, ptr)"
                    }),
                    "{ir}"
                );
                assert!(
                    ir.contains(if wide { "and i64" } else { "and i32" }),
                    "{ir}"
                );
                assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
            }
        }
    }
    Ok(())
}

#[test]
fn test_frexp_rejects_malformed_calls() {
    use dialect_mir::rust_intrinsics as ri;
    use dialect_mir::types::MirTupleType;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for wide in [false, true] {
            for invalid in 0..7 {
                let mut ctx = make_test_ctx();
                let float_ty: TypeHandle = if wide {
                    FP64Type::get(&ctx).into()
                } else {
                    FP32Type::get(&ctx).into()
                };
                let other_ty: TypeHandle = if wide {
                    FP32Type::get(&ctx).into()
                } else {
                    FP64Type::get(&ctx).into()
                };
                let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
                let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![float_ty, other_ty]);
                let x = entry.deref(&ctx).get_argument(0);
                let other = entry.deref(&ctx).get_argument(1);
                let fields = match invalid {
                    3 => vec![float_ty, i64_ty],
                    4 => vec![other_ty, i32_ty],
                    5 => vec![float_ty],
                    _ => vec![float_ty, i32_ty],
                };
                let tuple: TypeHandle = MirTupleType::get(&mut ctx, fields).into();
                let result = if invalid == 2 { float_ty } else { tuple };
                let args = match invalid {
                    0 => vec![],
                    1 => vec![x, x],
                    6 => vec![other],
                    _ => vec![x],
                };
                let placeholder = if wide {
                    ri::CALLEE_FREXP_F64
                } else {
                    ri::CALLEE_FREXP_F32
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![result],
                    args,
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call)
                    .set_attr_callee(&ctx, StringAttr::new(placeholder.to_string()));
                call.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);
                let error = mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .expect_err("malformed frexp ABI must fail")
                .to_string();
                let expected = match invalid {
                    0 | 1 => "requires 1 operand(s)",
                    2 => "requires a (float, i32) result",
                    _ => "signature mismatch",
                };
                assert!(error.contains(expected), "case {invalid}: {error}");
            }
        }
    }
}

#[test]
fn test_special_math_mixed_signatures_on_both_backends() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::{TypeHandle, Typed};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let i32_mir: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
        let i32_llvm: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty, f64_ty, i32_mir]);
        let x32 = entry.deref(&ctx).get_argument(0);
        let x64 = entry.deref(&ctx).get_argument(1);
        let exponent = entry.deref(&ctx).get_argument(2);
        let cases = [
            (
                ri::CALLEE_TGAMMA_F32,
                "__nv_tgammaf",
                f32_ty,
                vec![x32],
                vec![f32_ty],
            ),
            (
                ri::CALLEE_TGAMMA_F64,
                "__nv_tgamma",
                f64_ty,
                vec![x64],
                vec![f64_ty],
            ),
            (
                ri::CALLEE_ILOGB_F32,
                "__nv_ilogbf",
                i32_mir,
                vec![x32],
                vec![f32_ty],
            ),
            (
                ri::CALLEE_ILOGB_F64,
                "__nv_ilogb",
                i32_mir,
                vec![x64],
                vec![f64_ty],
            ),
            (
                ri::CALLEE_LDEXP_F32,
                "__nv_ldexpf",
                f32_ty,
                vec![x32, exponent],
                vec![f32_ty, i32_llvm],
            ),
            (
                ri::CALLEE_LDEXP_F64,
                "__nv_ldexp",
                f64_ty,
                vec![x64, exponent],
                vec![f64_ty, i32_llvm],
            ),
            // Control: the pre-existing mixed float/integer powi path.
            (
                ri::CALLEE_POWI_F32,
                "__nv_powif",
                f32_ty,
                vec![x32, exponent],
                vec![f32_ty, i32_llvm],
            ),
            (
                ri::CALLEE_POWI_F64,
                "__nv_powi",
                f64_ty,
                vec![x64, exponent],
                vec![f64_ty, i32_llvm],
            ),
        ];
        for (callee, _, result_ty, operands, _) in &cases {
            let call = Operation::new(
                &mut ctx,
                mir::MirCallOp::get_concrete_op_info(),
                vec![*result_ty],
                operands.clone(),
                vec![],
                0,
            );
            mir::MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new((*callee).to_string()));
            call.insert_at_back(entry, &ctx);
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

        let calls = lowered_kernel_body(&ctx, module_ptr)
            .into_iter()
            .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), cases.len());
        for call in calls {
            let CallOpCallable::Direct(symbol) = call.callee(&ctx) else {
                panic!("special math must lower to a direct call");
            };
            let name = symbol.to_string();
            let (_, _, mir_result, _, expected_args) = cases
                .iter()
                .find(|(_, symbol, _, _, _)| *symbol == name)
                .expect("unexpected libdevice symbol");
            let expected_result = if *mir_result == i32_mir {
                i32_llvm
            } else {
                *mir_result
            };
            let op = call.get_operation().deref(&ctx);
            assert_eq!(op.get_num_results(), 1, "{name}");
            assert_eq!(op.get_result(0).get_type(&ctx), expected_result, "{name}");
            assert_eq!(
                op.operands()
                    .map(|arg| arg.get_type(&ctx))
                    .collect::<Vec<_>>(),
                *expected_args,
                "{name}"
            );
        }
        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("mixed-signature math must export to LLVM IR");
        for declaration in [
            "declare float @__nv_tgammaf(float",
            "declare double @__nv_tgamma(double",
            "declare i32 @__nv_ilogbf(float",
            "declare i32 @__nv_ilogb(double",
            "declare float @__nv_ldexpf(float",
            "declare double @__nv_ldexp(double",
        ] {
            assert!(ir.contains(declaration), "missing {declaration}:\n{ir}");
        }
        assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
    }
    Ok(())
}

#[test]
fn test_special_math_lowering_rejects_bad_arity_and_types() {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for bad_case in 0..10 {
            let mut ctx = make_test_ctx();
            let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
            let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
            let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
            let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
            let (module_ptr, entry) =
                build_test_kernel(&mut ctx, vec![f32_ty, f64_ty, i32_ty, i64_ty]);
            let x32 = entry.deref(&ctx).get_argument(0);
            let x64 = entry.deref(&ctx).get_argument(1);
            let exponent = entry.deref(&ctx).get_argument(2);
            let wide_exponent = entry.deref(&ctx).get_argument(3);
            let (callee, results, args, diagnostic) = match bad_case {
                0 => (
                    ri::CALLEE_TGAMMA_F32,
                    vec![f32_ty],
                    vec![],
                    "requires 1 operand(s)",
                ),
                1 => (
                    ri::CALLEE_LDEXP_F64,
                    vec![f64_ty],
                    vec![x64],
                    "requires 2 operand(s)",
                ),
                2 => (
                    ri::CALLEE_ILOGB_F32,
                    vec![f32_ty],
                    vec![x32],
                    "signature mismatch",
                ),
                3 => (
                    ri::CALLEE_LDEXP_F32,
                    vec![f32_ty],
                    vec![x32, x32],
                    "signature mismatch",
                ),
                4 => (
                    ri::CALLEE_LDEXP_F64,
                    vec![f64_ty],
                    vec![x64, wide_exponent],
                    "signature mismatch",
                ),
                5 => (
                    ri::CALLEE_TGAMMA_F32,
                    vec![f32_ty],
                    vec![x64],
                    "signature mismatch",
                ),
                6 => (
                    ri::CALLEE_TGAMMA_F64,
                    vec![f32_ty],
                    vec![x64],
                    "signature mismatch",
                ),
                7 => (
                    ri::CALLEE_ILOGB_F64,
                    vec![i64_ty],
                    vec![x64],
                    "signature mismatch",
                ),
                8 => (
                    ri::CALLEE_LDEXP_F32,
                    vec![f64_ty],
                    vec![x32, exponent],
                    "signature mismatch",
                ),
                _ => (
                    ri::CALLEE_TGAMMA_F64,
                    vec![],
                    vec![x64],
                    "must have one result",
                ),
            };
            let call = Operation::new(
                &mut ctx,
                mir::MirCallOp::get_concrete_op_info(),
                results,
                args,
                vec![],
                0,
            );
            mir::MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new(callee.to_string()));
            call.insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);
            let error = mir_lower::lower_mir_to_llvm_with_options(
                &mut ctx,
                module_ptr,
                mir_lower::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            )
            .expect_err("malformed special math calls must fail")
            .to_string();
            assert!(error.contains(diagnostic), "case {bad_case}: {error}");
        }
    }
}

#[test]
fn test_plain_libm_aliases_lower_with_typed_abi_on_both_backends() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::{TypeHandle, Typed};

    // 0 = unary, 1 = binary.
    let families = [
        ("erf", 0),
        ("erfc", 0),
        ("exp10", 0),
        ("j0", 0),
        ("lgamma", 0),
        ("y0", 0),
        ("y1", 0),
        ("fdim", 1),
        ("fmod", 1),
        ("nextafter", 1),
        ("remainder", 1),
    ];
    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty, f64_ty]);
        let float_values = [
            entry.deref(&ctx).get_argument(0),
            entry.deref(&ctx).get_argument(1),
        ];
        for (wide, float_ty) in [(false, f32_ty), (true, f64_ty)] {
            let value = float_values[usize::from(wide)];
            for (family, signature) in families {
                let alias = if wide {
                    family.to_owned()
                } else {
                    format!("{family}f")
                };
                let intrinsic =
                    ri::RustFloatMathIntrinsic::from_libm_path(&format!("libm::{alias}"))
                        .expect("the alias must be registered");
                let operands = match signature {
                    0 => vec![value],
                    1 => vec![value, value],
                    _ => unreachable!(),
                };
                let call = Operation::new(
                    &mut ctx,
                    mir::MirCallOp::get_concrete_op_info(),
                    vec![float_ty],
                    operands,
                    vec![],
                    0,
                );
                mir::MirCallOp::new(call).set_attr_callee(
                    &ctx,
                    StringAttr::new(intrinsic.placeholder_callee().to_string()),
                );
                call.insert_at_back(entry, &ctx);
            }
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
        pliron::operation::verify_operation(module_ptr, &ctx)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        let calls = lowered_kernel_body(&ctx, module_ptr)
            .into_iter()
            .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), families.len() * 2);
        for call in calls {
            let CallOpCallable::Direct(symbol) = call.callee(&ctx) else {
                panic!("expected a direct libdevice call");
            };
            let name = symbol.to_string();
            // The width is taken from the call result, since __nv_erf ends in f.
            let op = call.get_operation().deref(&ctx);
            let float_ty = op.get_result(0).get_type(&ctx);
            let wide = float_ty == f64_ty;
            let suffix = if wide { "" } else { "f" };
            let family = families
                .iter()
                .find(|(family, _)| name == format!("__nv_{family}{suffix}"))
                .expect("unexpected libdevice symbol");
            let arguments = match family.1 {
                0 => vec![float_ty],
                1 => vec![float_ty, float_ty],
                _ => unreachable!(),
            };
            assert_eq!(
                op.operands()
                    .map(|arg| arg.get_type(&ctx))
                    .collect::<Vec<_>>(),
                arguments,
                "{name}"
            );
            assert_eq!(op.get_num_results(), 1, "{name}");
        }
        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("plain alias module must export to LLVM IR");
        for declaration in [
            "declare float @__nv_erff(float)",
            "declare double @__nv_erf(double)",
            "declare float @__nv_nextafterf(float, float)",
            "declare double @__nv_nextafter(double, double)",
        ] {
            assert!(ir.contains(declaration), "missing {declaration}:\n{ir}");
        }
        assert!(!ir.contains(ri::PLACEHOLDER_PREFIX), "{ir}");
    }
    Ok(())
}

#[test]
fn test_plain_libm_aliases_reject_invalid_scalar_signatures() {
    use dialect_mir::rust_intrinsics as ri;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for case in 0..8 {
            let mut ctx = make_test_ctx();
            let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
            let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
            let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
            let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty, f64_ty, i32_ty]);
            let x32 = entry.deref(&ctx).get_argument(0);
            let x64 = entry.deref(&ctx).get_argument(1);
            let n32 = entry.deref(&ctx).get_argument(2);
            let (alias, result, operands, diagnostic) = match case {
                0 => ("fmodf", f32_ty, vec![x32, n32], "signature mismatch"),
                1 => ("erf", f64_ty, vec![x32], "signature mismatch"),
                2 => ("lgammaf", f32_ty, vec![], "requires 1 operand(s)"),
                3 => ("nextafter", f64_ty, vec![x64], "requires 2 operand(s)"),
                4 => ("nextafterf", f64_ty, vec![x32, x32], "signature mismatch"),
                5 => ("y0f", f32_ty, vec![x64], "signature mismatch"),
                6 => ("fdim", f64_ty, vec![x64, x32], "signature mismatch"),
                _ => ("j0", f32_ty, vec![x64], "signature mismatch"),
            };
            let intrinsic =
                ri::RustFloatMathIntrinsic::from_libm_path(&format!("libm::{alias}")).unwrap();
            let call = Operation::new(
                &mut ctx,
                mir::MirCallOp::get_concrete_op_info(),
                vec![result],
                operands,
                vec![],
                0,
            );
            mir::MirCallOp::new(call).set_attr_callee(
                &ctx,
                StringAttr::new(intrinsic.placeholder_callee().to_string()),
            );
            call.insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);
            let error = mir_lower::lower_mir_to_llvm_with_options(
                &mut ctx,
                module_ptr,
                mir_lower::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            )
            .expect_err("malformed alias must fail")
            .to_string();
            assert!(error.contains(diagnostic), "case {case}: {error}");
        }
    }
}

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
