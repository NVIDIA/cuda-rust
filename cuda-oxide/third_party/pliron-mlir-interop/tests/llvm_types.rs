// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Translation of the pliron LLVM dialect's types to MLIR.

mod common;

use expect_test::expect;
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::Context,
    r#type::TypeHandle,
};
use pliron_llvm::types::{
    ArrayType, FuncType, PointerType, StructLayout, StructType, VectorType, VectorTypeKind,
    VoidType,
};

#[test]
fn types() {
    let ctx = &mut Context::new();
    let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let void_ty: TypeHandle = VoidType::get(ctx).into();

    let types: Vec<TypeHandle> = vec![
        PointerType::get(ctx, 0).into(),
        PointerType::get(ctx, 3).into(),
        ArrayType::get(ctx, i8_ty, 4).into(),
        VectorType::get(ctx, i32_ty, 4, VectorTypeKind::Fixed).into(),
        VectorType::get(ctx, f32_ty, 2, VectorTypeKind::Scalable).into(),
        // `!llvm.void` is only ever a function result.
        FuncType::get(ctx, void_ty, vec![i32_ty], false).into(),
        FuncType::get(ctx, i32_ty, vec![i8_ty], true).into(),
        FuncType::get(ctx, i32_ty, vec![], true).into(),
        StructType::get_unnamed(ctx, (vec![i32_ty, f32_ty], StructLayout::Unpacked)).into(),
        StructType::get_unnamed(ctx, (vec![i8_ty, i32_ty], StructLayout::Packed)).into(),
        StructType::get_named(ctx, "Pair".try_into().unwrap(), None)
            .unwrap()
            .into(),
    ];

    let printed: Vec<String> = types.iter().map(|ty| common::to_mlir(ctx, ty)).collect();
    expect![[r#"
        !llvm.ptr
        !llvm.ptr<3>
        !llvm.array<4 x i8>
        vector<4 x i32>
        vector<[2] x f32>
        !llvm.func<!llvm.void (i32)>
        !llvm.func<i32 (i8, ...)>
        !llvm.func<i32 (...)>
        !llvm.struct<(i32, f32)>
        !llvm.struct<packed (i8, i32)>
        !llvm.struct<"Pair", opaque>"#]]
    .assert_eq(&printed.join("\n"));

    // Function types and opaque structs are not valid parameter types,
    // so verify them separately.
    let (attr_types, value_types): (Vec<String>, Vec<String>) = printed
        .into_iter()
        .partition(|ty| ty.starts_with("!llvm.func<") || ty.ends_with(", opaque>"));
    common::verify_types(&value_types);
    common::verify_attrs(&attr_types);
}

/// A named struct whose body refers back to itself prints as `!llvm.struct<"name">`
/// on the recursive edge, exactly as MLIR does.
#[test]
fn recursive_struct() {
    let ctx = &mut Context::new();
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();

    let name: pliron::identifier::Identifier = "LinkedList".try_into().unwrap();
    let list = StructType::get_named(ctx, name.clone(), None).unwrap();
    let nested = StructType::get_unnamed(ctx, (vec![list.into(), i64_ty], StructLayout::Unpacked));
    StructType::get_named(
        ctx,
        name,
        Some((vec![i64_ty, ptr_ty, nested.into()], StructLayout::Unpacked)),
    )
    .unwrap();

    let printed = common::to_mlir(ctx, &TypeHandle::from(list));
    expect![[
        r#"!llvm.struct<"LinkedList", (i64, !llvm.ptr, !llvm.struct<(!llvm.struct<"LinkedList">, i64)>)>"#
    ]]
    .assert_eq(&printed);

    common::verify_types(&[printed]);
}
