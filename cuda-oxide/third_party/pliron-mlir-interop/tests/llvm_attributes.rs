// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Translation of the pliron LLVM dialect's attributes to MLIR.

mod common;

use core::num::NonZero;

use expect_test::expect;
use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::{FPSingleAttr, IntegerAttr},
        types::{FP32Type, IntegerType, Signedness},
    },
    context::Context,
    printable::Printable,
    r#type::TypeHandle,
    utils::{apfloat::Single, apint::APInt},
};
use pliron_llvm::{
    attributes::{
        AddressSpaceAttr, AggregateAttr, AlignmentAttr, AtomicOrderingAttr, AtomicRmwKindAttr,
        BytesAttr, CaseValuesAttr, FCmpPredicateAttr, FastmathFlags, FastmathFlagsAttr,
        GepIndexAttr, GepIndicesAttr, GepNoWrapFlags, GepNoWrapFlagsAttr, ICmpPredicateAttr,
        InsertExtractValueIndicesAttr, IntegerOverflowFlagsAttr, LinkageAttr, PoisonAttr,
        ShuffleVectorMaskAttr, SplatAttr, SymbolAddrAttr, UndefAttr, ZeroAttr,
    },
    types::{ArrayType, PointerType, StructLayout, StructType, VectorType, VectorTypeKind},
};
use pliron_mlir_interop::MlirPrinter;

#[test]
fn attributes() {
    let ctx = &mut Context::new();
    let i32_int = IntegerType::get(ctx, 32, Signedness::Signless);
    let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
    let bw32 = NonZero::new(32).unwrap();
    let i8_array = ArrayType::get(
        ctx,
        IntegerType::get(ctx, 8, Signedness::Signless).into(),
        2,
    );
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();

    let attrs: Vec<AttrObj> = vec![
        Box::new(LinkageAttr::InternalLinkage),
        Box::new(LinkageAttr::ExternalWeakLinkage),
        Box::new(ICmpPredicateAttr::SLT),
        Box::new(ICmpPredicateAttr::UGE),
        Box::new(FCmpPredicateAttr::OEQ),
        Box::new(FCmpPredicateAttr::True),
        Box::new(AtomicOrderingAttr::Monotonic),
        Box::new(AtomicOrderingAttr::SeqCst),
        Box::new(AtomicRmwKindAttr::Add),
        Box::new(AtomicRmwKindAttr::FMin),
        Box::new(IntegerOverflowFlagsAttr {
            nsw: false,
            nuw: false,
        }),
        Box::new(IntegerOverflowFlagsAttr {
            nsw: true,
            nuw: true,
        }),
        Box::new(FastmathFlagsAttr(FastmathFlags::empty())),
        Box::new(FastmathFlagsAttr(FastmathFlags::NNAN | FastmathFlags::ARCP)),
        Box::new(FastmathFlagsAttr(FastmathFlags::FAST)),
        Box::new(AlignmentAttr(8)),
        Box::new(AddressSpaceAttr(3)),
        Box::new(ZeroAttr(ptr_ty)),
        Box::new(UndefAttr(ptr_ty)),
        Box::new(PoisonAttr(ptr_ty)),
        Box::new(InsertExtractValueIndicesAttr(vec![0, 2])),
        Box::new(ShuffleVectorMaskAttr(vec![3, 1, -1, 0])),
        Box::new(GepIndicesAttr(vec![
            GepIndexAttr::Constant(0),
            GepIndexAttr::OperandIdx(1),
            GepIndexAttr::Constant(7),
        ])),
        // LLVM GEP indices are signed; pliron stores their raw bits in a `u32`.
        Box::new(GepIndicesAttr(vec![GepIndexAttr::Constant(-1i32 as u32)])),
        Box::new(GepIndicesAttr(vec![])),
        Box::new(CaseValuesAttr(vec![
            IntegerAttr::new(i32_int, APInt::from_u64(0, bw32)),
            IntegerAttr::new(i32_int, APInt::from_u64(11, bw32)),
        ])),
        Box::new(BytesAttr::new(vec![0, 1, 255])),
        Box::new(GepNoWrapFlagsAttr(GepNoWrapFlags::NUW)),
        // `inbounds` sets `nusw` too.
        Box::new(GepNoWrapFlagsAttr(
            GepNoWrapFlags::INBOUNDS | GepNoWrapFlags::NUW,
        )),
        Box::new(AggregateAttr::new(
            vec![
                Box::new(i32_attr(ctx, 1)),
                Box::new(i32_attr(ctx, 2)),
                Box::new(i32_attr(ctx, 3)),
            ],
            ArrayType::get(ctx, i32_int.into(), 3).into(),
        )),
        // An array of arrays is a `tensor` of rank 2, whichever attribute
        // holds the inner arrays.
        Box::new(AggregateAttr::new(
            vec![
                Box::new(BytesAttr::new(vec![1, 2])),
                Box::new(AggregateAttr::new(
                    vec![Box::new(i8_attr(ctx, 3)), Box::new(i8_attr(ctx, 4))],
                    i8_array.into(),
                )),
            ],
            ArrayType::get(ctx, i8_array.into(), 2).into(),
        )),
        Box::new(AggregateAttr::new(
            vec![
                Box::new(FPSingleAttr("1.5".parse::<Single>().unwrap())),
                Box::new(FPSingleAttr("2.0".parse::<Single>().unwrap())),
            ],
            VectorType::get(ctx, f32_ty, 2, VectorTypeKind::Fixed).into(),
        )),
        // An empty array of arrays must print as `dense<>`.
        Box::new(AggregateAttr::new(
            vec![],
            ArrayType::get(ctx, i8_array.into(), 0).into(),
        )),
        Box::new(BytesAttr::new(vec![])),
        Box::new(SplatAttr::new(
            Box::new(i32_attr(ctx, 7)),
            VectorType::get(ctx, i32_int.into(), 4, VectorTypeKind::Fixed),
        )),
    ];

    let printed: Vec<String> = attrs.iter().map(|a| common::to_mlir(ctx, &**a)).collect();
    expect![[r#"
        #llvm.linkage<internal>
        #llvm.linkage<extern_weak>
        2 : i64
        9 : i64
        1 : i64
        15 : i64
        2 : i64
        7 : i64
        1 : i64
        14 : i64
        #llvm.overflow<none>
        #llvm.overflow<nsw, nuw>
        #llvm.fastmath<none>
        #llvm.fastmath<nnan, arcp>
        #llvm.fastmath<nnan, ninf, nsz, arcp, contract, afn, reassoc>
        8 : i64
        3 : i32
        #llvm.zero
        #llvm.undef
        #llvm.poison
        array<i64: 0, 2>
        array<i32: 3, 1, -1, 0>
        array<i32: 0, -2147483648, 7>
        array<i32: -1>
        array<i32>
        dense<[0, 11]> : vector<2xi32>
        dense<[0, 1, -1]> : tensor<3xi8>
        4 : i32
        7 : i32
        dense<[1, 2, 3]> : tensor<3xi32>
        dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
        dense<[0x3FC00000, 0x40000000]> : vector<2xf32>
        dense<> : tensor<0x2xi8>
        dense<> : tensor<0xi8>
        dense<7> : vector<4xi32>"#]]
    .assert_eq(&printed.join("\n"));

    common::verify_attrs(&printed);
}

/// LLVM-IR has linkage kinds that MLIR's LLVM dialect does not model.
#[test]
fn unsupported_linkage_is_an_error() {
    let ctx = &Context::new();
    let attr: AttrObj = Box::new(LinkageAttr::DLLExportLinkage);
    let printer = MlirPrinter::new(ctx, &*attr as &dyn pliron::attribute::Attribute);
    let mut out = String::new();
    assert!(std::fmt::Write::write_fmt(&mut out, format_args!("{printer}")).is_err());
    let err = printer.take_error().expect("an error must be set");
    expect![[r#"
        Compilation error: invalid input program.
        Cannot translate to MLIR: DLLExportLinkage has no MLIR equivalent"#]]
    .assert_eq(&err.disp(ctx).to_string());
}

/// MLIR spends `i32::MIN` on marking a dynamic index, so a constant index of
/// that value has no MLIR encoding.
#[test]
fn gep_constant_index_colliding_with_the_dynamic_marker_is_an_error() {
    let ctx = &Context::new();
    let attr: AttrObj = Box::new(GepIndicesAttr(vec![GepIndexAttr::Constant(
        i32::MIN as u32,
    )]));
    let printer = MlirPrinter::new(ctx, &*attr as &dyn pliron::attribute::Attribute);
    let mut out = String::new();
    assert!(std::fmt::Write::write_fmt(&mut out, format_args!("{printer}")).is_err());
    let err = printer.take_error().expect("an error must be set");
    expect![[r#"
        Compilation error: invalid input program.
        Cannot translate to MLIR: llvm.gep constant index -2147483648 is indistinguishable from MLIR's marker for a dynamic index"#]].assert_eq(&err.disp(ctx).to_string());
}

/// MLIR takes a GEP's dynamic indices from its operands in order, so pliron
/// indices that name operands out of order cannot be translated.
#[test]
fn out_of_order_gep_indices_are_an_error() {
    let ctx = &Context::new();
    let attr: AttrObj = Box::new(GepIndicesAttr(vec![
        GepIndexAttr::OperandIdx(2),
        GepIndexAttr::OperandIdx(1),
    ]));
    let printer = MlirPrinter::new(ctx, &*attr as &dyn pliron::attribute::Attribute);
    let mut out = String::new();
    assert!(std::fmt::Write::write_fmt(&mut out, format_args!("{printer}")).is_err());
    let err = printer.take_error().expect("an error must be set");
    expect![[r#"
        Compilation error: invalid input program.
        Cannot translate to MLIR: llvm.gep dynamic index refers to operand 2, but MLIR consumes dynamic indices in operand order (expected operand 1)"#]]
    .assert_eq(&err.disp(ctx).to_string());
}

/// A struct constant, or one with a symbol address, has no MLIR attribute.
/// Only `llvm.mlir.constant` and `llvm.mlir.global` can build it, with operations.
#[test]
fn struct_constant_attribute_is_an_error() {
    let ctx = &Context::new();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let ptr_ty = PointerType::get(ctx, 0);
    let struct_ty =
        StructType::get_unnamed(ctx, (vec![i32_ty, ptr_ty.into()], StructLayout::Unpacked));
    let attr: AttrObj = Box::new(AggregateAttr::new(
        vec![
            Box::new(i32_attr(ctx, 1)),
            Box::new(SymbolAddrAttr::new("g".try_into().unwrap(), ptr_ty)),
        ],
        struct_ty.into(),
    ));
    let printer = MlirPrinter::new(ctx, &*attr as &dyn pliron::attribute::Attribute);
    let mut out = String::new();
    assert!(std::fmt::Write::write_fmt(&mut out, format_args!("{printer}")).is_err());
    let err = printer.take_error().expect("an error must be set");
    expect![[r#"
        Compilation error: invalid input program.
        Cannot translate to MLIR: constant `llvm.aggregate` has no MLIR attribute; only `llvm.mlir.constant` and `llvm.mlir.global` can build it"#]].assert_eq(&err.disp(ctx).to_string());
}

fn i32_attr(ctx: &Context, value: u64) -> IntegerAttr {
    IntegerAttr::new(
        IntegerType::get(ctx, 32, Signedness::Signless),
        APInt::from_u64(value, NonZero::new(32).unwrap()),
    )
}

fn i8_attr(ctx: &Context, value: u64) -> IntegerAttr {
    IntegerAttr::new(
        IntegerType::get(ctx, 8, Signedness::Signless),
        APInt::from_u64(value, NonZero::new(8).unwrap()),
    )
}
