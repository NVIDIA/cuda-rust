/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::MirPointerKindAuthorityAttr,
    types::{MirPointerKind, MirPtrType, MirTupleType},
};
use dialect_nvvm::ops::InlinePtxOp;

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{IntegerType, Signedness},
    context::Context,
    op::verify_op,
};

#[test]
fn test_inline_ptx_results_must_match_output_constraints() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let input = block.deref(&ctx).get_argument(0);

    let void = InlinePtxOp::build(&mut ctx, vec![], vec![], "membar.gl;", "", true, false);
    assert!(verify_op(&InlinePtxOp::new(void), &ctx).is_ok());

    let single = InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "add.u32 $0, $1, $1;",
        "=r,r",
        false,
        false,
    );
    assert!(verify_op(&InlinePtxOp::new(single), &ctx).is_ok());

    let multi = InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into(), i32_ty.into()],
        vec![input],
        "add.u32 $0, $2, $2; mul.lo.u32 $1, $2, $2;",
        "=r,=r,r",
        false,
        false,
    );
    assert!(verify_op(&InlinePtxOp::new(multi), &ctx).is_ok());

    let missing_result = InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "add.u32 $0, $2, $2; mul.lo.u32 $1, $2, $2;",
        "=r,=r,r",
        false,
        false,
    );
    assert!(verify_op(&InlinePtxOp::new(missing_result), &ctx).is_err());

    let extra_result = InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "prefetch.global.L1 [$0];",
        "r",
        true,
        false,
    );
    assert!(verify_op(&InlinePtxOp::new(extra_result), &ctx).is_err());
}

#[test]
fn test_inline_ptx_pointer_results_require_exact_inline_asm_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let erased_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::Erased);
    let nested_erased_ty = MirTupleType::get(&mut ctx, vec![erased_ty.into()]);

    let build = |ctx: &mut Context, result_ty| {
        InlinePtxOp::build(
            ctx,
            vec![result_ty],
            vec![],
            "mov.u64 $0, 0;",
            "=l",
            false,
            false,
        )
    };

    for (name, result_ty) in [
        ("UniqueRef", unique_ty.into()),
        ("Erased", erased_ty.into()),
        ("nested Erased", nested_erased_ty.into()),
    ] {
        let unmarked = InlinePtxOp::new(build(&mut ctx, result_ty));
        let error = verify_op(&unmarked, &ctx).unwrap_err();
        assert!(
            error
                .err
                .to_string()
                .contains("pointer-carrying results require InlineAsm pointer-kind authority"),
            "unexpected unmarked {name} error: {}",
            error.err
        );

        let marked = InlinePtxOp::new(build(&mut ctx, result_ty));
        marked.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::InlineAsm);
        assert!(
            verify_op(&marked, &ctx).is_ok(),
            "marked {name} pointer result should verify"
        );
    }

    let wrong_authority = InlinePtxOp::new(build(&mut ctx, unique_ty.into()));
    wrong_authority.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    let wrong_error = verify_op(&wrong_authority, &ctx).unwrap_err();
    assert!(
        wrong_error.err.to_string().contains("found RustCast"),
        "unexpected wrong-authority error: {}",
        wrong_error.err
    );

    let spurious = InlinePtxOp::new(build(&mut ctx, i32_ty.into()));
    spurious.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::InlineAsm);
    let spurious_error = verify_op(&spurious, &ctx).unwrap_err();
    assert!(
        spurious_error
            .err
            .to_string()
            .contains("spurious for pointer-free results"),
        "unexpected spurious-authority error: {}",
        spurious_error.err
    );
}

#[test]
fn test_inline_ptx_count_output_constraints() {
    assert_eq!(InlinePtxOp::count_output_constraints("=r,r,r"), 1);
    assert_eq!(InlinePtxOp::count_output_constraints("=r,=r,=f,=d,r,l"), 4);
    assert_eq!(InlinePtxOp::count_output_constraints("r,l,~{memory}"), 0);
    assert_eq!(InlinePtxOp::count_output_constraints(""), 0);
    // `=` only counts as an output marker at the start of a token.
    assert_eq!(InlinePtxOp::count_output_constraints("r,r=f"), 0);
}
