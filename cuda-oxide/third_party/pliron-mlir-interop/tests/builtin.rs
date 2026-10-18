// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Translation of pliron's builtin dialect to MLIR.

mod common;

use core::num::NonZero;

use expect_test::expect;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{
        attributes::{
            BoolAttr, DictAttr, FPBFloatAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr,
            IdentifierAttr, IntegerAttr, OperandSegmentSizesAttr, StringAttr, TypeAttr, UnitAttr,
            VecAttr,
        },
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::{ConstantOp, FuncOp, ModuleOp},
        types::{BF16Type, FP16Type, FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
    },
    context::Context,
    op::Op,
    r#type::TypeHandle,
    utils::apfloat::{BFloat, Double, Half, Single},
    utils::apint::APInt,
};
use pliron_llvm::ops::ReturnOp;

fn i32_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

#[test]
fn types() {
    let ctx = &mut Context::new();

    let types: Vec<TypeHandle> = vec![
        IntegerType::get(ctx, 1, Signedness::Signless).into(),
        IntegerType::get(ctx, 32, Signedness::Signless).into(),
        IntegerType::get(ctx, 64, Signedness::Signed).into(),
        IntegerType::get(ctx, 8, Signedness::Unsigned).into(),
        FP16Type::get(ctx).into(),
        BF16Type::get(ctx).into(),
        FP32Type::get(ctx).into(),
        FP64Type::get(ctx).into(),
    ];

    let printed: Vec<String> = types.iter().map(|ty| common::to_mlir(ctx, ty)).collect();
    expect![[r#"
        i1
        i32
        si64
        ui8
        f16
        bf16
        f32
        f64"#]]
    .assert_eq(&printed.join("\n"));

    common::verify_types(&printed);
}

#[test]
fn function_type() {
    let ctx = &mut Context::new();
    let i32_ty = i32_ty(ctx);
    let f64_ty = FP64Type::get(ctx).into();

    // MLIR spells a function type the same way, but it is not an LLVM-compatible
    // type, so it only shows up in attribute position.
    let no_args: TypeHandle = FunctionType::get(ctx, vec![], vec![]).into();
    let two_and_one: TypeHandle = FunctionType::get(ctx, vec![i32_ty, f64_ty], vec![i32_ty]).into();

    let printed = vec![
        common::to_mlir(ctx, &no_args),
        common::to_mlir(ctx, &two_and_one),
    ];
    expect![[r#"
        () -> ()
        (i32, f64) -> (i32)"#]]
    .assert_eq(&printed.join("\n"));

    common::verify_attrs(&printed);
}

#[test]
fn attributes() {
    let ctx = &mut Context::new();
    let i32_int = IntegerType::get(ctx, 32, Signedness::Signless);
    let si32_int = IntegerType::get(ctx, 32, Signedness::Signed);
    let bw32 = NonZero::new(32).unwrap();

    let attrs: Vec<AttrObj> = vec![
        Box::new(IdentifierAttr::new("some_symbol".try_into().unwrap())),
        Box::new(StringAttr::new(
            "hello \"world\"\n\tand a \\ and \u{e9}".to_string(),
        )),
        Box::new(BoolAttr::new(true)),
        Box::new(BoolAttr::new(false)),
        Box::new(IntegerAttr::new(i32_int, APInt::from_u64(42, bw32))),
        Box::new(IntegerAttr::new(si32_int, APInt::from_i64(-7, bw32))),
        Box::new(FPHalfAttr("1.5".parse::<Half>().unwrap())),
        Box::new(FPBFloatAttr("-1.0".parse::<BFloat>().unwrap())),
        Box::new(FPSingleAttr("-2.25".parse::<Single>().unwrap())),
        Box::new(FPDoubleAttr(
            core::f64::consts::PI.to_string().parse::<Double>().unwrap(),
        )),
        Box::new(UnitAttr::new()),
        Box::new(TypeAttr::new(i32_ty(ctx))),
        Box::new(VecAttr::new(vec![
            Box::new(BoolAttr::new(true)),
            Box::new(StringAttr::new("x".to_string())),
        ])),
        Box::new(DictAttr::new(vec![
            (
                "a".try_into().unwrap(),
                Box::new(BoolAttr::new(false)) as AttrObj,
            ),
            (
                "b".try_into().unwrap(),
                Box::new(StringAttr::new("y".to_string())) as AttrObj,
            ),
        ])),
        Box::new(OperandSegmentSizesAttr(vec![1, 0, 2])),
    ];

    let printed: Vec<String> = attrs.iter().map(|a| common::to_mlir(ctx, &**a)).collect();
    expect![[r#"
        @some_symbol
        "hello \"world\"\n\tand a \\ and \C3\A9"
        true
        false
        42 : i32
        -7 : si32
        0x3E00 : f16
        0xBF80 : bf16
        0xC0100000 : f32
        0x400921FB54442D18 : f64
        unit
        i32
        [true, "x"]
        {a = false, b = "y"}
        array<i32: 1, 0, 2>"#]]
    .assert_eq(&printed.join("\n"));

    common::verify_attrs(&printed);
}

/// `builtin.module` / `builtin.func` / `builtin.constant`. MLIR's builtin
/// dialect has neither a function nor a constant op, so the latter two become
/// `func.func` and `arith.constant`.
#[test]
fn ops() {
    let ctx = &mut Context::new();
    let i32_ty = i32_ty(ctx);
    let bw32 = NonZero::new(32).unwrap();

    let module = ModuleOp::new(ctx, "a_module".try_into().unwrap());

    let func_ty = FunctionType::get(ctx, vec![i32_ty], vec![i32_ty]);
    let func = FuncOp::new(ctx, "answer".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);

    let body = func.get_entry_block(ctx);
    let i32_int = IntegerType::get(ctx, 32, Signedness::Signless);
    let constant = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(i32_int, APInt::from_u64(42, bw32))),
    );
    constant.get_operation().insert_at_back(body, ctx);
    // pliron's builtin dialect has no terminator op of its own.
    let ret = ReturnOp::new(ctx, Some(constant.get_result(ctx)));
    ret.get_operation().insert_at_back(body, ctx);

    // A body-less function is a declaration.
    let decl_ty = FunctionType::get(ctx, vec![], vec![]);
    let decl = FuncOp::new(ctx, "declared".try_into().unwrap(), decl_ty);
    BasicBlock::erase(decl.get_entry_block(ctx), ctx);
    module.append_operation(ctx, decl.get_operation(), 0);

    let printed = common::to_mlir(ctx, &module.get_operation());
    expect![[r#"
        "builtin.module"() <{sym_name = "a_module"}> ({
          ^block1v1:
            "func.func"() <{function_type = (i32) -> (i32), sym_name = "answer"}> ({
              ^entry_block2v1(%v0: i32):
                %v1 = "arith.constant"() <{value = 42 : i32}> : () -> (i32)
                "llvm.return"(%v1) : (i32) -> ()
            }) : () -> ()
            "func.func"() <{function_type = () -> (), sym_name = "declared", sym_visibility = "private"}> ({
            }) : () -> ()
        }) : () -> ()"#]]
    .assert_eq(&printed);

    common::expect_mlir_opt_output(
        &printed,
        expect![[r#"
        module @a_module {
          func.func @answer(%arg0: i32) -> i32 {
            %c42_i32 = arith.constant 42 : i32
            llvm.return %c42_i32 : i32
          }
          func.func private @declared()
        }

    "#]],
    );
}
