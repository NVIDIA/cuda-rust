//! Rust FP32 maximum semantics and rejection of incompatible intrinsic signatures.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::{
    ops::{MirCallOp, MirFuncOp, MirReturnOp},
    rust_intrinsics::CALLEE_MAXNUM_NSZ_F32,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
    },
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};

/// Build and export a scalar call with independently controlled operand and result signatures.
fn translate_call(
    profile: &dyn MlirConsumerProfile,
    callee: &str,
    operand_types: &[&str],
    result_types: &[&str],
) -> Result<String, String> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let ty = |name| -> TypeHandle {
        match name {
            "f32" => FP32Type::get(&ctx).into(),
            "f64" => FP64Type::get(&ctx).into(),
            "i32" => IntegerType::get(&ctx, 32, Signedness::Signed).into(),
            _ => unreachable!(),
        }
    };
    let operands = operand_types
        .iter()
        .map(|name| ty(*name))
        .collect::<Vec<_>>();
    let results = result_types
        .iter()
        .map(|name| ty(*name))
        .collect::<Vec<_>>();
    let function_type = FunctionType::get(&ctx, operands.clone(), results.clone());
    let module = ModuleOp::new(&mut ctx, "rust_float_max".try_into().unwrap());
    let function = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(&mut ctx, function, TypeAttr::new(function_type.into()))
        .set_symbol_name(&mut ctx, "user_max".try_into().unwrap());
    module.append_operation(&mut ctx, function, 0);
    let block = BasicBlock::new(&mut ctx, None, operands);
    block.insert_at_back(function.deref(&ctx).get_region(0), &ctx);
    let arguments = block.deref(&ctx).arguments().collect();
    let call = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        results,
        arguments,
        vec![],
        0,
    );
    MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new(callee.into()));
    call.insert_at_back(block, &ctx);
    let values = call.deref(&ctx).results().collect();
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        values,
        vec![],
        0,
    );
    ret.insert_at_back(block, &ctx);
    profile
        .translate_module(&ctx, &module)
        .and_then(|module| render_module(&module).map_err(Into::into))
        .map_err(|error| error.to_string())
}

/// Create both reviewed profiles for the same SM100 target.
fn profiles() -> [Box<dyn MlirConsumerProfile>; 2] {
    [
        Box::new(CutlassFullCuteMlir22::new("sm_120a").unwrap()),
        Box::new(CutlassPrimitivesMlir22::new("sm_120a").unwrap()),
    ]
}

/// Check both profiles preserve Rust max's NaN-ignoring and non-FTZ behavior.
#[test]
fn rust_f32_max_exports_nan_ignoring_subnormal_preserving_native_maximum() {
    for profile in profiles() {
        let text =
            translate_call(&*profile, CALLEE_MAXNUM_NSZ_F32, &["f32", "f32"], &["f32"]).unwrap();
        // CUTLASS 4.7 cute/arch/nvvm_wrappers.py::fmax uses the modifier-free
        // nvvm.fmax for maximumNumber: ignore qNaN/sNaN, preserve subnormals.
        // PTX's preference for +0 over -0 is allowed by Rust's nsz contract.
        assert!(
            text.contains("\"nvvm.fmax\"(%v0, %v1) : (f32, f32) -> (f32)"),
            "{}: {text}",
            profile.name()
        );
        assert!(
            !text.contains("\"func.call\"") && !text.contains(CALLEE_MAXNUM_NSZ_F32),
            "{text}"
        );
        assert!(
            text.contains("\"func.return\"(%v2) : (f32) -> ()"),
            "{text}"
        );
    }
}

/// Reject max calls whose argument/result arity differs from the reviewed intrinsic.
#[test]
fn rust_f32_max_rejects_malformed_operand_and_result_counts() {
    for profile in profiles() {
        for (operands, results) in [
            (vec![], vec!["f32"]),
            (vec!["f32"], vec!["f32"]),
            (vec!["f32", "f32", "f32"], vec!["f32"]),
            (vec!["f32", "f32"], vec![]),
            (vec!["f32", "f32"], vec!["f32", "f32"]),
        ] {
            let result = translate_call(&*profile, CALLEE_MAXNUM_NSZ_F32, &operands, &results);
            assert!(
                result.is_err(),
                "{} accepted {operands:?} -> {results:?}: {result:?}",
                profile.name()
            );
        }
    }
}

/// Reject max calls that try to reuse the FP32 mapping for another scalar format.
#[test]
fn rust_f32_max_rejects_non_f32_operands_and_results() {
    for profile in profiles() {
        for (operands, results) in [
            (["f64", "f32"], ["f32"]),
            (["f32", "f64"], ["f32"]),
            (["i32", "f32"], ["f32"]),
            (["f32", "i32"], ["f32"]),
            (["f32", "f32"], ["f64"]),
            (["f32", "f32"], ["i32"]),
        ] {
            let result = translate_call(&*profile, CALLEE_MAXNUM_NSZ_F32, &operands, &results);
            assert!(
                result.is_err(),
                "{} accepted {operands:?} -> {results:?}: {result:?}",
                profile.name()
            );
        }
    }
}

/// Ensure an unrelated scalar function is not mistaken for the Rust max intrinsic.
#[test]
fn ordinary_f32_function_calls_remain_calls() {
    for profile in profiles() {
        let text = translate_call(&*profile, "user_max", &["f32", "f32"], &["f32"]).unwrap();
        assert!(
            text.contains("\"func.call\"(%v0, %v1) <{callee = @user_max}> : (f32, f32) -> (f32)"),
            "{text}"
        );
        assert!(!text.contains("\"nvvm.fmax\""), "{text}");
    }
}
