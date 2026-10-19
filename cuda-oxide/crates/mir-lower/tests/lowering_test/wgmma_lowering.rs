/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

pub(super) fn build_wgmma_pointer_test_kernel(
    ctx: &mut Context,
    accumulator_count: usize,
    trailing_arg_types: Vec<pliron::r#type::TypeHandle>,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    Vec<pliron::value::Value>,
    pliron::value::Value,
    pliron::value::Value,
    Vec<pliron::value::Value>,
) {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    let f32_ty = FP32Type::get(ctx);
    let accumulator_ptr_ty: TypeHandle = MirPtrType::get_generic(ctx, f32_ty.into(), true).into();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();

    let trailing_count = trailing_arg_types.len();
    let mut argument_types = vec![accumulator_ptr_ty; accumulator_count];
    argument_types.push(u64_ty);
    argument_types.push(u64_ty);
    argument_types.extend(trailing_arg_types);

    let (module_ptr, entry) = build_test_kernel(ctx, argument_types);

    let accumulators = (0..accumulator_count)
        .map(|index| entry.deref(ctx).get_argument(index))
        .collect::<Vec<_>>();
    let desc_a = entry.deref(ctx).get_argument(accumulator_count);
    let desc_b = entry.deref(ctx).get_argument(accumulator_count + 1);
    let trailing_arguments = (0..trailing_count)
        .map(|index| entry.deref(ctx).get_argument(accumulator_count + 2 + index))
        .collect::<Vec<_>>();

    (
        module_ptr,
        entry,
        accumulators,
        desc_a,
        desc_b,
        trailing_arguments,
    )
}

fn build_wgmma_canonical_pointer_test_kernel_with_rows(
    ctx: &mut Context,
    accumulator_rows: u64,
    accumulator_count: usize,
    descriptor_count: usize,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    Vec<pliron::value::Value>,
    Vec<pliron::value::Value>,
) {
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let f32_ty = FP32Type::get(ctx);
    let row_ty = MirArrayType::get(ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(ctx, row_ty.into(), accumulator_rows);
    let accumulator_ptr_ty = MirPtrType::get_generic(ctx, accumulator_ty.into(), true);
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);

    let accumulator_arg_ty: pliron::r#type::TypeHandle = accumulator_ptr_ty.into();
    let u64_arg_ty: pliron::r#type::TypeHandle = u64_ty.into();
    let mut argument_types = vec![accumulator_arg_ty; accumulator_count];
    argument_types.extend(vec![u64_arg_ty; descriptor_count]);
    let (module_ptr, entry) = build_test_kernel(ctx, argument_types);

    let accumulators = (0..accumulator_count)
        .map(|index| entry.deref(ctx).get_argument(index))
        .collect::<Vec<_>>();
    let descriptors = (0..descriptor_count)
        .map(|index| entry.deref(ctx).get_argument(accumulator_count + index))
        .collect::<Vec<_>>();

    (module_ptr, entry, accumulators, descriptors)
}

pub(super) fn build_wgmma_canonical_pointer_test_kernel(
    ctx: &mut Context,
    accumulator_count: usize,
    descriptor_count: usize,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    Vec<pliron::value::Value>,
    Vec<pliron::value::Value>,
) {
    build_wgmma_canonical_pointer_test_kernel_with_rows(ctx, 4, accumulator_count, descriptor_count)
}

fn build_wgmma_m64n128_canonical_pointer_test_kernel(
    ctx: &mut Context,
    accumulator_count: usize,
    descriptor_count: usize,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    Vec<pliron::value::Value>,
    Vec<pliron::value::Value>,
) {
    build_wgmma_canonical_pointer_test_kernel_with_rows(ctx, 8, accumulator_count, descriptor_count)
}

pub(super) fn append_pointer_wgmma_mma(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    accumulator: pliron::value::Value,
    desc_a: pliron::value::Value,
    desc_b: pliron::value::Value,
) {
    Operation::new(
        ctx,
        nvvm::WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}

pub(super) fn append_pointer_wgmma_mma_m64n128(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    accumulator: pliron::value::Value,
    desc_a: pliron::value::Value,
    desc_b: pliron::value::Value,
) {
    Operation::new(
        ctx,
        nvvm::WgmmaMmaM64N128K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}

pub(super) fn append_pointer_wgmma_mma_f16(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    accumulator: pliron::value::Value,
    desc_a: pliron::value::Value,
    desc_b: pliron::value::Value,
) {
    Operation::new(
        ctx,
        nvvm::WgmmaMmaM64N64K16F32F16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}

pub(super) fn append_pointer_wgmma_mma_tf32(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    accumulator: pliron::value::Value,
    desc_a: pliron::value::Value,
    desc_b: pliron::value::Value,
) {
    Operation::new(
        ctx,
        nvvm::WgmmaMmaM64N64K8F32Tf32Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}

pub(super) fn append_wgmma_wait_group_constant(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    pending_groups: i64,
) {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let constant_attr = IntegerAttr::new(
        u64_ty,
        APInt::from_i64(pending_groups, NonZeroUsize::new(64).unwrap()),
    );
    let constant = Operation::new(
        ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(constant).set_attr_value(ctx, constant_attr);
    constant.insert_at_back(block, ctx);

    let value = constant.deref(ctx).get_result(0);
    nvvm::WgmmaWaitGroupSyncAlignedOp::build(ctx, value).insert_at_back(block, ctx);
}

pub(super) fn append_mir_unsigned_constant(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
    ty: pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType>,
    value: u64,
) -> pliron::value::Value {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let width = usize::try_from(ty.deref(ctx).width()).expect("integer width must fit usize");
    let constant = Operation::new(
        ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(constant).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(
                value,
                NonZeroUsize::new(width).expect("nonzero integer width"),
            ),
        ),
    );
    constant.insert_at_back(block, ctx);
    constant.deref(ctx).get_result(0)
}

#[test]
fn test_deferred_wgmma_group_lowers_to_one_register_lifetime_scope() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), true);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = entry.deref(&ctx).get_argument(0);
    let desc_a = entry.deref(&ctx).get_argument(1);
    let desc_b = entry.deref(&ctx).get_argument(2);

    nvvm::WgmmaMmaGroupM64N64K16F32Bf16Op::build(&mut ctx, accumulator, vec![desc_a, desc_b])
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| {
                    template.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("WGMMA template");
    assert_eq!(template.matches("ld.f32 %acc").count(), 32);
    assert_eq!(template.matches("st.f32 [$0").count(), 32);
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    let wait = template.find("wgmma.wait_group.sync.aligned 0").unwrap();
    let first_store = template.find("st.f32 [$0").unwrap();
    assert!(
        wait < first_store,
        "accumulator stores must follow wait_group<0>"
    );
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some("l,l,l,~{memory}")
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    Ok(())
}

#[test]
fn test_value_form_wgmma_group_lowers_to_tied_register_inline_ptx() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::r#type::Typed;

    const ACCUMULATOR_LEN: usize = 32;
    const DESCRIPTOR_COUNT: usize = 4;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    let argument_types = (0..ACCUMULATOR_LEN)
        .map(|_| f32_ty.into())
        .chain((0..DESCRIPTOR_COUNT).map(|_| u64_ty.into()))
        .collect::<Vec<pliron::r#type::TypeHandle>>();

    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

    let accumulators = (0..ACCUMULATOR_LEN)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let descriptors = (0..DESCRIPTOR_COUNT)
        .map(|index| entry.deref(&ctx).get_argument(ACCUMULATOR_LEN + index))
        .collect::<Vec<_>>();

    nvvm::WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(&mut ctx, accumulators, descriptors)
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| {
                    template.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let asm_op = asm.get_operation();
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("value-form WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 2);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    for forbidden in [".reg .f32", "ld.f32", "st.f32"] {
        assert!(
            !template.contains(forbidden),
            "value-form WGMMA must not materialize accumulator memory via {forbidden:?}: {template}"
        );
    }

    let accumulator_operands = (0..ACCUMULATOR_LEN)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    assert!(
        template.contains(&format!("{{{accumulator_operands}}}")),
        "WGMMA must consume the 32 output operands as one accumulator tuple: {template}"
    );
    assert!(template.contains("$64, $65"), "{template}");
    assert!(template.contains("$66, $67"), "{template}");

    let mut expected_constraints = vec!["=f".to_owned(); ACCUMULATOR_LEN];
    expected_constraints.extend((0..ACCUMULATOR_LEN).map(|index| index.to_string()));
    expected_constraints.extend((0..DESCRIPTOR_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");

    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    let asm_ref = asm_op.deref(&ctx);
    assert_eq!(
        asm_ref.get_num_operands(),
        ACCUMULATOR_LEN + DESCRIPTOR_COUNT
    );
    assert_eq!(
        asm_ref.get_num_results(),
        1,
        "LLVM inline asm must return the 32 WGMMA accumulator values as one struct"
    );

    let aggregate = asm_ref.get_result(0);
    let aggregate_ty = aggregate.get_type(&ctx);
    let aggregate_ty = aggregate_ty.deref(&ctx);
    let struct_ty = aggregate_ty
        .downcast_ref::<llvm_types::StructType>()
        .expect("value-form WGMMA inline asm must return an LLVM struct");
    assert_eq!(struct_ty.num_fields(), ACCUMULATOR_LEN);
    for index in 0..ACCUMULATOR_LEN {
        assert!(
            struct_ty
                .field_type(index)
                .deref(&ctx)
                .downcast_ref::<FP32Type>()
                .is_some(),
            "WGMMA result field {index} must remain f32"
        );
    }

    let extract_indices = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::ExtractValueOp>(operation, &ctx))
        .filter_map(|extract| {
            let extract_op = extract.get_operation().deref(&ctx);
            (extract_op.get_operand(0) == aggregate).then(|| extract.indices(&ctx))
        })
        .collect::<Vec<_>>();

    let expected_indices = (0..ACCUMULATOR_LEN)
        .map(|index| vec![index as u32])
        .collect::<Vec<_>>();
    assert_eq!(
        extract_indices, expected_indices,
        "all 32 value-form WGMMA results must be extracted in constraint order"
    );

    Ok(())
}

#[test]
fn test_value_form_m64n128_bf16_wgmma_group_lowers_to_sixty_four_tied_registers()
-> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::r#type::Typed;

    const ACCUMULATOR_LEN: usize = 64;
    const DESCRIPTOR_COUNT: usize = 4;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let argument_types = (0..ACCUMULATOR_LEN)
        .map(|_| f32_ty.into())
        .chain((0..DESCRIPTOR_COUNT).map(|_| u64_ty.into()))
        .collect::<Vec<pliron::r#type::TypeHandle>>();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

    let accumulators = (0..ACCUMULATOR_LEN)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let descriptors = (0..DESCRIPTOR_COUNT)
        .map(|index| entry.deref(&ctx).get_argument(ACCUMULATOR_LEN + index))
        .collect::<Vec<_>>();

    nvvm::WgmmaMmaGroupValuesM64N128K16F32Bf16Op::build(&mut ctx, accumulators, descriptors)
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n128k16.f32.bf16.bf16"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("m64n128 value-form WGMMA template");
    assert_eq!(template.matches("wgmma.mma_async").count(), 2);
    assert!(template.contains("m64n128k16.f32.bf16.bf16"));
    assert!(template.contains("$128, $129"));
    assert!(template.contains("$130, $131"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));

    let constraints = asm
        .get_attr_llvm_inline_asm_constraints(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("m64n128 value-form WGMMA constraints");
    assert_eq!(
        constraints
            .split(',')
            .filter(|value| *value == "=f")
            .count(),
        64
    );
    assert_eq!(
        constraints.split(',').filter(|value| *value == "l").count(),
        4
    );
    assert!(constraints.ends_with("~{memory}"));

    let aggregate = asm.get_operation().deref(&ctx).get_result(0);
    let aggregate_ty = aggregate.get_type(&ctx);
    let aggregate_ty = aggregate_ty.deref(&ctx);
    let struct_ty = aggregate_ty
        .downcast_ref::<llvm_types::StructType>()
        .expect("m64n128 inline asm must return an LLVM struct");
    assert_eq!(struct_ty.num_fields(), ACCUMULATOR_LEN);
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_value_form_f16_wgmma_group_lowers_to_tied_register_inline_ptx() -> Result<(), anyhow::Error>
{
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const ACCUMULATOR_LEN: usize = 32;
    const DESCRIPTOR_COUNT: usize = 2;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let argument_types = (0..ACCUMULATOR_LEN)
        .map(|_| f32_ty.into())
        .chain((0..DESCRIPTOR_COUNT).map(|_| u64_ty.into()))
        .collect::<Vec<pliron::r#type::TypeHandle>>();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

    let accumulators = (0..ACCUMULATOR_LEN)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let descriptors = (0..DESCRIPTOR_COUNT)
        .map(|index| entry.deref(&ctx).get_argument(ACCUMULATOR_LEN + index))
        .collect::<Vec<_>>();

    nvvm::WgmmaMmaGroupValuesM64N64K16F32F16Op::build(&mut ctx, accumulators, descriptors)
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n64k16.f32.f16.f16"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("F16 value-form WGMMA template");
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    assert!(template.contains("m64n64k16.f32.f16.f16"));
    assert!(!template.contains(".bf16.bf16"));
    assert!(template.contains("wgmma.fence.sync.aligned"));
    assert!(template.contains("wgmma.commit_group.sync.aligned"));
    assert!(template.contains("wgmma.wait_group.sync.aligned 0"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));

    let constraints = asm
        .get_attr_llvm_inline_asm_constraints(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("F16 value-form WGMMA constraints");
    assert_eq!(
        constraints
            .split(',')
            .filter(|value| *value == "=f")
            .count(),
        32
    );
    assert_eq!(
        constraints.split(',').filter(|value| *value == "l").count(),
        2
    );
    assert!(constraints.ends_with("~{memory}"));
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_value_form_tf32_wgmma_group_lowers_to_tied_register_inline_ptx() -> Result<(), anyhow::Error>
{
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const ACCUMULATOR_LEN: usize = 32;
    const DESCRIPTOR_COUNT: usize = 2;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let argument_types = (0..ACCUMULATOR_LEN)
        .map(|_| f32_ty.into())
        .chain((0..DESCRIPTOR_COUNT).map(|_| u64_ty.into()))
        .collect::<Vec<pliron::r#type::TypeHandle>>();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

    let accumulators = (0..ACCUMULATOR_LEN)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let descriptors = (0..DESCRIPTOR_COUNT)
        .map(|index| entry.deref(&ctx).get_argument(ACCUMULATOR_LEN + index))
        .collect::<Vec<_>>();

    nvvm::WgmmaMmaGroupValuesM64N64K8F32Tf32Op::build(&mut ctx, accumulators, descriptors)
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n64k8.f32.tf32.tf32"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("TF32 value-form WGMMA template");
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    assert!(template.contains("m64n64k8.f32.tf32.tf32"));
    assert!(!template.contains(".bf16.bf16"));
    assert!(!template.contains(".f16.f16"));
    assert!(!template.contains("m64n64k16"));
    assert!(template.contains("$64, $65, 1, 1, 1;"));
    assert!(!template.contains("$64, $65, 1, 1, 1, 0, 0;"));
    assert!(template.contains("wgmma.fence.sync.aligned"));
    assert!(template.contains("wgmma.commit_group.sync.aligned"));
    assert!(template.contains("wgmma.wait_group.sync.aligned 0"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));

    let constraints = asm
        .get_attr_llvm_inline_asm_constraints(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("TF32 value-form WGMMA constraints");
    assert_eq!(
        constraints
            .split(',')
            .filter(|value| *value == "=f")
            .count(),
        32
    );
    assert_eq!(
        constraints.split(',').filter(|value| *value == "l").count(),
        2
    );
    assert!(constraints.ends_with("~{memory}"));
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_pointer_form_wgmma_sequence_preserves_deferred_fallback() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    // A direct *mut f32 is intentionally not the public [[f32; 8]; 4]
    // accumulator shape. It must therefore retain the deferred pointer path.
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), true);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = entry.deref(&ctx).get_argument(0);
    let desc_a = entry.deref(&ctx).get_argument(1);
    let desc_b = entry.deref(&ctx).get_argument(2);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    Operation::new(
        &mut ctx,
        nvvm::WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);

    let zero_attr = IntegerAttr::new(u64_ty, APInt::from_i64(0, NonZeroUsize::new(64).unwrap()));
    let zero = Operation::new(
        &mut ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(zero).set_attr_value(&ctx, zero_attr);
    zero.insert_at_back(entry, &ctx);
    let zero_value = zero.deref(&ctx).get_result(0);
    nvvm::WgmmaWaitGroupSyncAlignedOp::build(&mut ctx, zero_value).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("wgmma.mma_async"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("WGMMA template");
    assert_eq!(template.matches("ld.f32 %acc").count(), 32);
    assert_eq!(template.matches("st.f32 [$0").count(), 32);
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    assert!(template.contains("wgmma.fence.sync.aligned"));
    assert!(template.contains("wgmma.commit_group.sync.aligned"));
    assert!(template.contains("wgmma.wait_group.sync.aligned 0"));
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some("l,l,l,~{memory}")
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_pointer_form_wgmma_region_canonicalizes_reborrow_identity() -> Result<(), anyhow::Error> {
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    use dialect_mir::types::{MirArrayType, MirPointerKind, MirPtrType};
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let erased_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let unique_ptr_ty: pliron::r#type::TypeHandle = MirPtrType::get_generic_with_kind(
        &mut ctx,
        accumulator_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    )
    .into();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![erased_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = entry.deref(&ctx).get_argument(0);
    let desc_a = entry.deref(&ctx).get_argument(1);
    let desc_b = entry.deref(&ctx).get_argument(2);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);

    // The importer emits a fresh kind-only retype for each `Rvalue::Ref`.
    // Both typed operands must canonicalize to the same storage identity, while
    // the first typed operand remains available to the linear lowering plan.
    let first_retype = Operation::new(
        &mut ctx,
        mir::MirCastOp::get_concrete_op_info(),
        vec![unique_ptr_ty],
        vec![accumulator],
        vec![],
        0,
    );
    mir::MirCastOp::new(first_retype).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mir::MirCastOp::new(first_retype)
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    first_retype.insert_at_back(entry, &ctx);
    let first_reborrow = first_retype.deref(&ctx).get_result(0);

    Operation::new(
        &mut ctx,
        nvvm::WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![first_reborrow, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);

    let second_retype = Operation::new(
        &mut ctx,
        mir::MirCastOp::get_concrete_op_info(),
        vec![unique_ptr_ty],
        vec![accumulator],
        vec![],
        0,
    );
    mir::MirCastOp::new(second_retype).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mir::MirCastOp::new(second_retype)
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    second_retype.insert_at_back(entry, &ctx);
    let second_reborrow = second_retype.deref(&ctx).get_result(0);
    assert_ne!(
        first_reborrow, second_reborrow,
        "the regression requires two distinct reborrow SSA values"
    );
    let first_reborrow_ty = first_reborrow.get_type(&ctx);
    assert_eq!(
        first_reborrow_ty
            .deref(&ctx)
            .downcast_ref::<MirPtrType>()
            .expect("first reborrow must remain a typed MIR pointer")
            .pointer_kind(),
        MirPointerKind::UniqueRef,
        "the regression must provide a typed UniqueRef for the linear plan to retain"
    );

    Operation::new(
        &mut ctx,
        nvvm::WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![second_reborrow, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);

    let zero_attr = IntegerAttr::new(u64_ty, APInt::from_i64(0, NonZeroUsize::new(64).unwrap()));
    let zero = Operation::new(
        &mut ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(zero).set_attr_value(&ctx, zero_attr);
    zero.insert_at_back(entry, &ctx);
    let zero_value = zero.deref(&ctx).get_result(0);
    nvvm::WgmmaWaitGroupSyncAlignedOp::build(&mut ctx, zero_value).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| {
                    template.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "distinct reborrow SSA values for one accumulator must not break deferred fusion"
    );
    let template = matching[0]
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("WGMMA template");
    assert_eq!(
        template.matches("wgmma.mma_async").count(),
        2,
        "both reborrows must remain in one fused accumulator region"
    );
    Ok(())
}

#[test]
fn test_pointer_form_wgmma_sequence_uses_value_adapter_before_lowering() -> Result<(), anyhow::Error>
{
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    const ACCUMULATOR_LEN: usize = 32;
    const DESCRIPTOR_COUNT: usize = 4;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);

    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![
            accumulator_ptr_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
        ],
    );
    let accumulator = entry.deref(&ctx).get_argument(0);
    let descriptors = (1..=DESCRIPTOR_COUNT)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    for pair in descriptors.as_chunks::<2>().0 {
        Operation::new(
            &mut ctx,
            nvvm::WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
            vec![],
            vec![accumulator, pair[0], pair[1]],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
    }
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);

    let zero_attr = IntegerAttr::new(u64_ty, APInt::from_i64(0, NonZeroUsize::new(64).unwrap()));
    let zero = Operation::new(
        &mut ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(zero).set_attr_value(&ctx, zero_attr);
    zero.insert_at_back(entry, &ctx);
    let zero_value = zero.deref(&ctx).get_result(0);
    nvvm::WgmmaWaitGroupSyncAlignedOp::build(&mut ctx, zero_value).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("wgmma.mma_async"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 2);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "value-form WGMMA must not materialize accumulator memory inside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); ACCUMULATOR_LEN];
    expected_constraints.extend((0..ACCUMULATOR_LEN).map(|index| index.to_string()));
    expected_constraints.extend((0..DESCRIPTOR_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");

    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    let asm_ref = asm_operation.deref(&ctx);
    assert_eq!(
        asm_ref.get_num_operands(),
        ACCUMULATOR_LEN + DESCRIPTOR_COUNT
    );
    assert_eq!(asm_ref.get_num_results(), 1);

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("WGMMA asm must be in the lowered kernel body");

    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        load_positions.len(),
        ACCUMULATOR_LEN,
        "the linear adapter must load each accumulator value exactly once"
    );
    assert!(
        load_positions.iter().all(|index| *index < asm_position),
        "all accumulator loads must happen before the WGMMA region"
    );

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        store_positions.len(),
        ACCUMULATOR_LEN,
        "the linear adapter must store each accumulator value exactly once"
    );
    assert!(
        store_positions.iter().all(|index| *index > asm_position),
        "all accumulator stores must happen after the final wait"
    );

    assert_eq!(
        body.iter()
            .filter(
                |operation| Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            )
            .count(),
        ACCUMULATOR_LEN,
        "all WGMMA accumulator results must be recovered as scalar SSA values"
    );

    Ok(())
}

#[test]
fn test_pointer_form_m64n128_bf16_linear_full_drain_uses_sixty_four_value_adapter()
-> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_m64n128_canonical_pointer_test_kernel(&mut ctx, 1, 4);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_m64n128(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    append_pointer_wgmma_mma_m64n128(&mut ctx, entry, accumulator, descriptors[2], descriptors[3]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n128k16.f32.bf16.bf16"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("m64n128 pointer-form WGMMA template");
    assert_eq!(template.matches("wgmma.mma_async").count(), 2);
    assert!(template.contains("m64n128k16.f32.bf16.bf16"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));

    let constraints = asm
        .get_attr_llvm_inline_asm_constraints(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("m64n128 pointer-form WGMMA constraints");
    assert_eq!(
        constraints
            .split(',')
            .filter(|value| *value == "=f")
            .count(),
        64
    );
    assert_eq!(
        constraints.split(',').filter(|value| *value == "l").count(),
        4
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_pointer_form_f16_wgmma_linear_full_drain_uses_value_adapter() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 2);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_f16(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n64k16.f32.f16.f16"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("F16 pointer-form WGMMA template");
    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("m64n64k16.f32.f16.f16"));
    assert!(!template.contains(".bf16.bf16"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_pointer_form_tf32_wgmma_linear_full_drain_uses_value_adapter() -> Result<(), anyhow::Error>
{
    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, 1, 2);
    let accumulator = accumulators[0];

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_pointer_wgmma_mma_tf32(&mut ctx, entry, accumulator, descriptors[0], descriptors[1]);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let matching = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|operation| Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx))
        .filter(|asm| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("m64n64k8.f32.tf32.tf32"))
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);

    let asm = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("TF32 pointer-form WGMMA template");
    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 1);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("m64n64k8.f32.tf32.tf32"));
    assert!(!template.contains(".bf16.bf16"));
    assert!(!template.contains(".f16.f16"));
    assert!(!template.contains("m64n64k16"));
    assert!(template.contains("$64, $65, 1, 1, 1;"));
    assert!(!template.contains("$64, $65, 1, 1, 1, 0, 0;"));
    assert!(!template.contains("ld.f32"));
    assert!(!template.contains("st.f32"));
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);

    Ok(())
}

#[test]
fn test_pointer_form_wgmma_partial_wait_pipeline_keeps_multiple_groups_in_flight()
-> Result<(), anyhow::Error> {
    const SLOT_COUNT: usize = 2;
    const ACCUMULATOR_LEN: usize = 32;
    const GROUP_COUNT: usize = 4;
    const DESCRIPTOR_COUNT: usize = GROUP_COUNT * 2;
    const MAX_PENDING_GROUPS: i64 = 1;
    const RESULT_COUNT: usize = SLOT_COUNT * ACCUMULATOR_LEN;

    let mut ctx = make_test_ctx();
    let (module_ptr, entry, accumulators, descriptors) =
        build_wgmma_canonical_pointer_test_kernel(&mut ctx, SLOT_COUNT, DESCRIPTOR_COUNT);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
    for group in 0..GROUP_COUNT {
        append_pointer_wgmma_mma(
            &mut ctx,
            entry,
            accumulators[group % SLOT_COUNT],
            descriptors[group * 2],
            descriptors[group * 2 + 1],
        );
        nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(entry, &ctx);
        if group + 1 >= SLOT_COUNT {
            append_wgmma_wait_group_constant(&mut ctx, entry, MAX_PENDING_GROUPS);
        }
    }
    append_wgmma_wait_group_constant(&mut ctx, entry, 0);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| {
                    template.contains("wgmma.mma_async")
                        && template.contains("wgmma.wait_group.sync.aligned 1")
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "expected one fused partial-wait WGMMA pipeline"
    );

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("pipeline WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), GROUP_COUNT);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        GROUP_COUNT
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 1").count(),
        GROUP_COUNT - SLOT_COUNT + 1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(
        template.contains("{$0, $1"),
        "slot 0 must use the first accumulator tuple"
    );
    assert!(
        template.contains("{$32, $33"),
        "slot 1 must use the second accumulator tuple"
    );
    assert!(template.contains("$128, $129"));
    assert!(template.contains("$134, $135"));
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "pipeline WGMMA must not materialize accumulator memory inside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); RESULT_COUNT];
    expected_constraints.extend((0..RESULT_COUNT).map(|index| index.to_string()));
    expected_constraints.extend((0..DESCRIPTOR_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    assert_eq!(
        asm_operation.deref(&ctx).get_num_operands(),
        RESULT_COUNT + DESCRIPTOR_COUNT
    );
    assert_eq!(asm_operation.deref(&ctx).get_num_results(), 1);

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("pipeline asm must be in the lowered kernel body");
    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(load_positions.len(), RESULT_COUNT);
    assert!(load_positions.iter().all(|index| *index < asm_position));

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(store_positions.len(), RESULT_COUNT);
    assert!(store_positions.iter().all(|index| *index > asm_position));
    assert_eq!(
        body.iter()
            .filter(|operation| {
                Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            })
            .count(),
        RESULT_COUNT
    );

    Ok(())
}

pub(super) fn build_pointer_form_wgmma_counted_pipeline_case(
    ctx: &mut Context,
    slot_count: usize,
    wait_depths: &[i64],
    repeat_last_accumulator: bool,
) -> pliron::context::Ptr<Operation> {
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    assert_eq!(wait_depths.len(), slot_count);

    let f32_ty = FP32Type::get(ctx);
    let row_ty = MirArrayType::get(ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(ctx, accumulator_ty.into(), true);
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let u64_type: pliron::r#type::TypeHandle = u64_ty.into();

    let mut argument_types: Vec<pliron::r#type::TypeHandle> =
        vec![accumulator_ptr_ty.into(); slot_count];
    argument_types.extend(vec![u64_type; slot_count * 2]);
    let (module_ptr, preheader) = build_test_kernel(ctx, argument_types);

    let accumulators = (0..slot_count)
        .map(|slot| preheader.deref(ctx).get_argument(slot))
        .collect::<Vec<_>>();
    let desc_bases = (0..slot_count * 2)
        .map(|index| preheader.deref(ctx).get_argument(slot_count + index))
        .collect::<Vec<_>>();

    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
    let function = module_block.deref(ctx).iter(ctx).next().unwrap();
    let function_region = function.deref(ctx).get_region(0);

    let mut header_types: Vec<pliron::r#type::TypeHandle> = vec![u32_ty.into()];
    header_types.extend(vec![u64_type; slot_count * 2]);
    let header = BasicBlock::new(ctx, None, header_types);
    header.insert_at_back(function_region, ctx);
    let latch = BasicBlock::new(ctx, None, vec![]);
    latch.insert_at_back(function_region, ctx);
    let exit = BasicBlock::new(ctx, None, vec![]);
    exit.insert_at_back(function_region, ctx);

    nvvm::WgmmaFenceSyncAlignedOp::build(ctx).insert_at_back(preheader, ctx);
    let i0 = append_mir_unsigned_constant(ctx, preheader, u32_ty, 0);
    let mut initial_values = vec![i0];
    initial_values.extend(desc_bases.iter().copied());
    Operation::new(
        ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        initial_values,
        vec![header],
        0,
    )
    .insert_at_back(preheader, ctx);

    let i = header.deref(ctx).get_argument(0);
    let descriptors = (0..slot_count * 2)
        .map(|index| header.deref(ctx).get_argument(1 + index))
        .collect::<Vec<_>>();
    let bound = append_mir_unsigned_constant(ctx, header, u32_ty, 4);
    let lt = Operation::new(
        ctx,
        mir::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![i, bound],
        vec![],
        0,
    );
    lt.insert_at_back(header, ctx);
    let lt_value = lt.deref(ctx).get_result(0);
    let not_lt = Operation::new(
        ctx,
        mir::MirNotOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![lt_value],
        vec![],
        0,
    );
    not_lt.insert_at_back(header, ctx);
    let not_lt_value = not_lt.deref(ctx).get_result(0);
    let (branch_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
    let branch = Operation::new(
        ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        branch_operands,
        vec![exit, latch],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(ctx, segment_sizes);
    branch.insert_at_back(header, ctx);

    for slot in 0..slot_count {
        let accumulator = if repeat_last_accumulator && slot + 1 == slot_count {
            accumulators[0]
        } else {
            accumulators[slot]
        };
        append_pointer_wgmma_mma(
            ctx,
            latch,
            accumulator,
            descriptors[slot * 2],
            descriptors[slot * 2 + 1],
        );
        nvvm::WgmmaCommitGroupSyncAlignedOp::build(ctx).insert_at_back(latch, ctx);
        append_wgmma_wait_group_constant(ctx, latch, wait_depths[slot]);
    }

    let one = append_mir_unsigned_constant(ctx, latch, u32_ty, 1);
    let i_next = Operation::new(
        ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![i, one],
        vec![],
        0,
    );
    i_next.insert_at_back(latch, ctx);
    let i_next = i_next.deref(ctx).get_result(0);

    let mut next_values = vec![i_next];
    for (index, descriptor) in descriptors.iter().copied().enumerate() {
        let step = append_mir_unsigned_constant(ctx, latch, u64_ty, 16 * (index as u64 + 1));
        let next = Operation::new(
            ctx,
            mir::MirAddOp::get_concrete_op_info(),
            vec![u64_ty.into()],
            vec![descriptor, step],
            vec![],
            0,
        );
        next.insert_at_back(latch, ctx);
        next_values.push(next.deref(ctx).get_result(0));
    }

    Operation::new(
        ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        next_values,
        vec![header],
        0,
    )
    .insert_at_back(latch, ctx);

    append_wgmma_wait_group_constant(ctx, exit, 0);
    append_return(ctx, exit);

    module_ptr
}

#[test]
fn test_pointer_form_wgmma_two_slot_counted_pipeline_stays_register_resident()
-> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const SLOT_COUNT: usize = 2;
    const ACCUMULATOR_LEN: usize = 32;
    const RESULT_COUNT: usize = SLOT_COUNT * ACCUMULATOR_LEN;
    const LOOP_CONTROL_COUNT: usize = 9;
    const TRIP_COUNT: u64 = 4;
    const DESC_A0_STEP: u64 = 16;
    const DESC_B0_STEP: u64 = 32;
    const DESC_A1_STEP: u64 = 48;
    const DESC_B1_STEP: u64 = 64;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    let (module_ptr, preheader) = build_test_kernel(
        &mut ctx,
        vec![
            accumulator_ptr_ty.into(),
            accumulator_ptr_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
        ],
    );
    let accumulator0 = preheader.deref(&ctx).get_argument(0);
    let accumulator1 = preheader.deref(&ctx).get_argument(1);
    let desc_a0_base = preheader.deref(&ctx).get_argument(2);
    let desc_b0_base = preheader.deref(&ctx).get_argument(3);
    let desc_a1_base = preheader.deref(&ctx).get_argument(4);
    let desc_b1_base = preheader.deref(&ctx).get_argument(5);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let header = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            u32_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
            u64_ty.into(),
        ],
    );
    header.insert_at_back(function_region, &ctx);
    let latch = BasicBlock::new(&mut ctx, None, vec![]);
    latch.insert_at_back(function_region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![]);
    exit.insert_at_back(function_region, &ctx);

    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(preheader, &ctx);
    let i0 = append_mir_unsigned_constant(&mut ctx, preheader, u32_ty, 0);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i0, desc_a0_base, desc_b0_base, desc_a1_base, desc_b1_base],
        vec![header],
        0,
    )
    .insert_at_back(preheader, &ctx);

    let i = header.deref(&ctx).get_argument(0);
    let desc_a0 = header.deref(&ctx).get_argument(1);
    let desc_b0 = header.deref(&ctx).get_argument(2);
    let desc_a1 = header.deref(&ctx).get_argument(3);
    let desc_b1 = header.deref(&ctx).get_argument(4);
    let bound = append_mir_unsigned_constant(&mut ctx, header, u32_ty, TRIP_COUNT);
    let lt = Operation::new(
        &mut ctx,
        mir::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![i, bound],
        vec![],
        0,
    );
    lt.insert_at_back(header, &ctx);
    let lt_value = lt.deref(&ctx).get_result(0);
    let not_lt = Operation::new(
        &mut ctx,
        mir::MirNotOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![lt_value],
        vec![],
        0,
    );
    not_lt.insert_at_back(header, &ctx);
    let not_lt_value = not_lt.deref(&ctx).get_result(0);
    let (branch_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        branch_operands,
        vec![exit, latch],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(header, &ctx);

    append_pointer_wgmma_mma(&mut ctx, latch, accumulator0, desc_a0, desc_b0);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(latch, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, latch, 1);
    append_pointer_wgmma_mma(&mut ctx, latch, accumulator1, desc_a1, desc_b1);
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(latch, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, latch, 1);

    let one = append_mir_unsigned_constant(&mut ctx, latch, u32_ty, 1);
    let i_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![i, one],
        vec![],
        0,
    );
    i_next.insert_at_back(latch, &ctx);
    let i_next = i_next.deref(&ctx).get_result(0);

    let desc_a0_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_A0_STEP);
    let desc_a0_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_a0, desc_a0_step],
        vec![],
        0,
    );
    desc_a0_next.insert_at_back(latch, &ctx);
    let desc_a0_next = desc_a0_next.deref(&ctx).get_result(0);

    let desc_b0_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_B0_STEP);
    let desc_b0_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_b0, desc_b0_step],
        vec![],
        0,
    );
    desc_b0_next.insert_at_back(latch, &ctx);
    let desc_b0_next = desc_b0_next.deref(&ctx).get_result(0);

    let desc_a1_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_A1_STEP);
    let desc_a1_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_a1, desc_a1_step],
        vec![],
        0,
    );
    desc_a1_next.insert_at_back(latch, &ctx);
    let desc_a1_next = desc_a1_next.deref(&ctx).get_result(0);

    let desc_b1_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_B1_STEP);
    let desc_b1_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_b1, desc_b1_step],
        vec![],
        0,
    );
    desc_b1_next.insert_at_back(latch, &ctx);
    let desc_b1_next = desc_b1_next.deref(&ctx).get_result(0);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![
            i_next,
            desc_a0_next,
            desc_b0_next,
            desc_a1_next,
            desc_b1_next,
        ],
        vec![header],
        0,
    )
    .insert_at_back(latch, &ctx);

    append_wgmma_wait_group_constant(&mut ctx, exit, 0);
    append_return(&mut ctx, exit);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("L__wgmma_pipeline_loop_${:uid}:"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "expected one fused two-slot counted WGMMA pipeline"
    );

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("counted-pipeline WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 2);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        2
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 1").count(),
        2
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("{$0, $1"));
    assert!(template.contains("{$32, $33"));
    assert!(template.contains("mov.u64 %desc_a0, $128;"));
    assert!(template.contains("mov.u64 %desc_b0, $129;"));
    assert!(template.contains("mov.u64 %desc_a1, $130;"));
    assert!(template.contains("mov.u64 %desc_b1, $131;"));
    assert!(template.contains("add.u64 %desc_a0, %desc_a0, $132;"));
    assert!(template.contains("add.u64 %desc_b0, %desc_b0, $133;"));
    assert!(template.contains("add.u64 %desc_a1, %desc_a1, $134;"));
    assert!(template.contains("add.u64 %desc_b1, %desc_b1, $135;"));
    assert!(template.contains("mov.u64 %remaining, $136;"));
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "counted pipeline must keep accumulator memory outside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); RESULT_COUNT];
    expected_constraints.extend((0..RESULT_COUNT).map(|index| index.to_string()));
    expected_constraints.extend((0..LOOP_CONTROL_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    assert_eq!(
        asm_operation.deref(&ctx).get_num_operands(),
        RESULT_COUNT + LOOP_CONTROL_COUNT
    );

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("counted-pipeline WGMMA asm must be in the lowered kernel body");
    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(load_positions.len(), RESULT_COUNT);
    assert!(load_positions.iter().all(|index| *index < asm_position));

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(store_positions.len(), RESULT_COUNT);
    assert!(store_positions.iter().all(|index| *index > asm_position));
    assert_eq!(
        body.iter()
            .filter(|operation| {
                Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            })
            .count(),
        RESULT_COUNT
    );

    Ok(())
}

#[test]
fn test_pointer_form_wgmma_three_slot_counted_pipeline_stays_register_resident()
-> Result<(), anyhow::Error> {
    const RESULT_COUNT: usize = 96;
    const LOOP_CONTROL_COUNT: usize = 13;

    let mut ctx = make_test_ctx();
    let module_ptr = build_pointer_form_wgmma_counted_pipeline_case(&mut ctx, 3, &[2, 2, 2], false);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("L__wgmma_pipeline_loop_${:uid}:"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "expected one fused three-slot counted WGMMA pipeline"
    );

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("three-slot counted-pipeline WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(template.matches("wgmma.mma_async").count(), 3);
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        3
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 2").count(),
        3
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("{$0, $1"));
    assert!(template.contains("{$32, $33"));
    assert!(template.contains("{$64, $65"));
    assert!(template.contains("mov.u64 %desc_a0, $192;"));
    assert!(template.contains("mov.u64 %desc_b0, $193;"));
    assert!(template.contains("mov.u64 %desc_a1, $194;"));
    assert!(template.contains("mov.u64 %desc_b1, $195;"));
    assert!(template.contains("mov.u64 %desc_a2, $196;"));
    assert!(template.contains("mov.u64 %desc_b2, $197;"));
    assert!(template.contains("add.u64 %desc_a0, %desc_a0, $198;"));
    assert!(template.contains("add.u64 %desc_b0, %desc_b0, $199;"));
    assert!(template.contains("add.u64 %desc_a1, %desc_a1, $200;"));
    assert!(template.contains("add.u64 %desc_b1, %desc_b1, $201;"));
    assert!(template.contains("add.u64 %desc_a2, %desc_a2, $202;"));
    assert!(template.contains("add.u64 %desc_b2, %desc_b2, $203;"));
    assert!(template.contains("mov.u64 %remaining, $204;"));
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "three-slot counted pipeline must keep accumulator memory outside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); RESULT_COUNT];
    expected_constraints.extend((0..RESULT_COUNT).map(|index| index.to_string()));
    expected_constraints.extend((0..LOOP_CONTROL_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    assert_eq!(
        asm_operation.deref(&ctx).get_num_operands(),
        RESULT_COUNT + LOOP_CONTROL_COUNT
    );

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("three-slot counted-pipeline WGMMA asm must be in the lowered kernel body");
    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(load_positions.len(), RESULT_COUNT);
    assert!(load_positions.iter().all(|index| *index < asm_position));

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(store_positions.len(), RESULT_COUNT);
    assert!(store_positions.iter().all(|index| *index > asm_position));
    assert_eq!(
        body.iter()
            .filter(|operation| {
                Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            })
            .count(),
        RESULT_COUNT
    );

    Ok(())
}

#[test]
fn test_pointer_form_wgmma_counted_k_loop_stays_register_resident() -> Result<(), anyhow::Error> {
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    use dialect_mir::types::{MirArrayType, MirPointerKind, MirPtrType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const ACCUMULATOR_LEN: usize = 32;
    const LOOP_CONTROL_COUNT: usize = 5;
    const TRIP_COUNT: u64 = 4;
    const DESC_A_STEP: u64 = 16;
    const DESC_B_STEP: u64 = 32;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let unique_accumulator_ptr_ty: pliron::r#type::TypeHandle = MirPtrType::get_generic_with_kind(
        &mut ctx,
        accumulator_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    )
    .into();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    let (module_ptr, preheader) = build_test_kernel(
        &mut ctx,
        vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = preheader.deref(&ctx).get_argument(0);
    let desc_a_base = preheader.deref(&ctx).get_argument(1);
    let desc_b_base = preheader.deref(&ctx).get_argument(2);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let header = BasicBlock::new(
        &mut ctx,
        None,
        vec![u32_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    header.insert_at_back(function_region, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![]);
    body.insert_at_back(function_region, &ctx);
    let latch = BasicBlock::new(&mut ctx, None, vec![]);
    latch.insert_at_back(function_region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![]);
    exit.insert_at_back(function_region, &ctx);
    let wait_block = BasicBlock::new(&mut ctx, None, vec![]);
    wait_block.insert_at_back(function_region, &ctx);

    // preheader: fence; i0 = 0; goto header(i0, desc_a_base, desc_b_base)
    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(preheader, &ctx);
    let i0 = append_mir_unsigned_constant(&mut ctx, preheader, u32_ty, 0);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i0, desc_a_base, desc_b_base],
        vec![header],
        0,
    )
    .insert_at_back(preheader, &ctx);

    // header(i, desc_a, desc_b): if !(i < 4) exit else body.
    let i = header.deref(&ctx).get_argument(0);
    let desc_a = header.deref(&ctx).get_argument(1);
    let desc_b = header.deref(&ctx).get_argument(2);
    let bound = append_mir_unsigned_constant(&mut ctx, header, u32_ty, TRIP_COUNT);
    let lt = Operation::new(
        &mut ctx,
        mir::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![i, bound],
        vec![],
        0,
    );
    lt.insert_at_back(header, &ctx);
    let lt_value = lt.deref(&ctx).get_result(0);
    let not_lt = Operation::new(
        &mut ctx,
        mir::MirNotOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![lt_value],
        vec![],
        0,
    );
    not_lt.insert_at_back(header, &ctx);
    let not_lt_value = not_lt.deref(&ctx).get_result(0);
    let (branch_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        branch_operands,
        vec![exit, body],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(header, &ctx);

    // body: a fresh Rust `&mut` reborrow and one WGMMA per K iteration. Rust
    // calls end a MIR block, so the post-call arithmetic lives in the latch.
    // The reborrow is loop-local, but its canonical storage identity is the
    // preheader accumulator argument.
    let retype = Operation::new(
        &mut ctx,
        mir::MirCastOp::get_concrete_op_info(),
        vec![unique_accumulator_ptr_ty],
        vec![accumulator],
        vec![],
        0,
    );
    mir::MirCastOp::new(retype).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mir::MirCastOp::new(retype)
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    retype.insert_at_back(body, &ctx);
    let reborrowed_accumulator = retype.deref(&ctx).get_result(0);
    append_pointer_wgmma_mma(&mut ctx, body, reborrowed_accumulator, desc_a, desc_b);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![latch],
        0,
    )
    .insert_at_back(body, &ctx);

    // latch: affine induction and descriptor recurrences, then the back edge.
    let one = append_mir_unsigned_constant(&mut ctx, latch, u32_ty, 1);
    let i_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![i, one],
        vec![],
        0,
    );
    i_next.insert_at_back(latch, &ctx);
    let i_next = i_next.deref(&ctx).get_result(0);

    let desc_a_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_A_STEP);
    let desc_a_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_a, desc_a_step],
        vec![],
        0,
    );
    desc_a_next.insert_at_back(latch, &ctx);
    let desc_a_next = desc_a_next.deref(&ctx).get_result(0);

    let desc_b_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_B_STEP);
    let desc_b_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_b, desc_b_step],
        vec![],
        0,
    );
    desc_b_next.insert_at_back(latch, &ctx);
    let desc_b_next = desc_b_next.deref(&ctx).get_result(0);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i_next, desc_a_next, desc_b_next],
        vec![header],
        0,
    )
    .insert_at_back(latch, &ctx);

    // Rust calls split the exit sequence too: commit in the loop exit, then
    // wait_group<0> in its unique linear successor.
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(exit, &ctx);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![wait_block],
        0,
    )
    .insert_at_back(exit, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, wait_block, 0);
    append_return(&mut ctx, wait_block);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("L__wgmma_loop_${:uid}:"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "expected one fused counted-loop WGMMA asm"
    );

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("counted-loop WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        template.matches("wgmma.mma_async").count(),
        1,
        "the PTX template contains one MMA instruction controlled by the internal K-loop"
    );
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("mov.u64 %desc_a, $64;"));
    assert!(template.contains("mov.u64 %desc_b, $65;"));
    assert!(template.contains("add.u64 %desc_a, %desc_a, $66;"));
    assert!(template.contains("add.u64 %desc_b, %desc_b, $67;"));
    assert!(template.contains("mov.u64 %remaining, $68;"));
    assert!(template.contains("@%loop_more bra.uni L__wgmma_done_${:uid};"));
    assert!(template.contains("@%loop_more bra.uni L__wgmma_loop_${:uid};"));
    assert!(template.contains("L__wgmma_done_${:uid}:"));
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "counted-loop WGMMA must keep accumulator memory outside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); ACCUMULATOR_LEN];
    expected_constraints.extend((0..ACCUMULATOR_LEN).map(|index| index.to_string()));
    expected_constraints.extend((0..LOOP_CONTROL_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    assert_eq!(
        asm_operation.deref(&ctx).get_num_operands(),
        ACCUMULATOR_LEN + LOOP_CONTROL_COUNT
    );

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("counted-loop WGMMA asm must be in the lowered kernel body");

    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        load_positions.len(),
        ACCUMULATOR_LEN,
        "K-loop accumulator must be loaded exactly once, not once per iteration"
    );
    assert!(load_positions.iter().all(|index| *index < asm_position));

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        store_positions.len(),
        ACCUMULATOR_LEN,
        "K-loop accumulator must be stored exactly once after the final wait"
    );
    assert!(store_positions.iter().all(|index| *index > asm_position));

    assert_eq!(
        body.iter()
            .filter(
                |operation| Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            )
            .count(),
        ACCUMULATOR_LEN
    );

    Ok(())
}

#[test]
fn test_f16_wgmma_counted_k_loop_stays_register_resident() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirArrayType, MirPtrType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    const ACCUMULATOR_LEN: usize = 32;
    const LOOP_CONTROL_COUNT: usize = 5;
    const TRIP_COUNT: u64 = 4;
    const DESC_A_STEP: u64 = 16;
    const DESC_B_STEP: u64 = 32;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let row_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 8);
    let accumulator_ty = MirArrayType::get(&mut ctx, row_ty.into(), 4);
    let accumulator_ptr_ty = MirPtrType::get_generic(&mut ctx, accumulator_ty.into(), true);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    let (module_ptr, preheader) = build_test_kernel(
        &mut ctx,
        vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    let accumulator = preheader.deref(&ctx).get_argument(0);
    let desc_a_base = preheader.deref(&ctx).get_argument(1);
    let desc_b_base = preheader.deref(&ctx).get_argument(2);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let function = module_block.deref(&ctx).iter(&ctx).next().unwrap();
    let function_region = function.deref(&ctx).get_region(0);

    let header = BasicBlock::new(
        &mut ctx,
        None,
        vec![u32_ty.into(), u64_ty.into(), u64_ty.into()],
    );
    header.insert_at_back(function_region, &ctx);
    let latch = BasicBlock::new(&mut ctx, None, vec![]);
    latch.insert_at_back(function_region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![]);
    exit.insert_at_back(function_region, &ctx);

    // preheader: fence; i0 = 0; goto header(i0, desc_a_base, desc_b_base)
    nvvm::WgmmaFenceSyncAlignedOp::build(&mut ctx).insert_at_back(preheader, &ctx);
    let i0 = append_mir_unsigned_constant(&mut ctx, preheader, u32_ty, 0);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i0, desc_a_base, desc_b_base],
        vec![header],
        0,
    )
    .insert_at_back(preheader, &ctx);

    // header(i, desc_a, desc_b): if !(i < 4) exit else latch.
    let i = header.deref(&ctx).get_argument(0);
    let desc_a = header.deref(&ctx).get_argument(1);
    let desc_b = header.deref(&ctx).get_argument(2);
    let bound = append_mir_unsigned_constant(&mut ctx, header, u32_ty, TRIP_COUNT);
    let lt = Operation::new(
        &mut ctx,
        mir::MirLtOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![i, bound],
        vec![],
        0,
    );
    lt.insert_at_back(header, &ctx);
    let lt_value = lt.deref(&ctx).get_result(0);
    let not_lt = Operation::new(
        &mut ctx,
        mir::MirNotOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![lt_value],
        vec![],
        0,
    );
    not_lt.insert_at_back(header, &ctx);
    let not_lt_value = not_lt.deref(&ctx).get_result(0);
    let (branch_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        branch_operands,
        vec![exit, latch],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(header, &ctx);

    // latch: one WGMMA per K iteration and affine descriptor recurrences.
    append_pointer_wgmma_mma_f16(&mut ctx, latch, accumulator, desc_a, desc_b);

    let one = append_mir_unsigned_constant(&mut ctx, latch, u32_ty, 1);
    let i_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![i, one],
        vec![],
        0,
    );
    i_next.insert_at_back(latch, &ctx);
    let i_next = i_next.deref(&ctx).get_result(0);

    let desc_a_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_A_STEP);
    let desc_a_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_a, desc_a_step],
        vec![],
        0,
    );
    desc_a_next.insert_at_back(latch, &ctx);
    let desc_a_next = desc_a_next.deref(&ctx).get_result(0);

    let desc_b_step = append_mir_unsigned_constant(&mut ctx, latch, u64_ty, DESC_B_STEP);
    let desc_b_next = Operation::new(
        &mut ctx,
        mir::MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![desc_b, desc_b_step],
        vec![],
        0,
    );
    desc_b_next.insert_at_back(latch, &ctx);
    let desc_b_next = desc_b_next.deref(&ctx).get_result(0);

    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![i_next, desc_a_next, desc_b_next],
        vec![header],
        0,
    )
    .insert_at_back(latch, &ctx);

    // exit: the only place where the asynchronous lifetime may become visible.
    nvvm::WgmmaCommitGroupSyncAlignedOp::build(&mut ctx).insert_at_back(exit, &ctx);
    append_wgmma_wait_group_constant(&mut ctx, exit, 0);
    append_return(&mut ctx, exit);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let body = lowered_kernel_body(&ctx, module_ptr);
    let matching = body
        .iter()
        .copied()
        .filter_map(|operation| {
            Operation::get_op::<llvm::InlineAsmOp>(operation, &ctx)
                .map(|inline_asm| (operation, inline_asm))
        })
        .filter(|(_, asm)| {
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .is_some_and(|template| template.contains("L__wgmma_loop_${:uid}:"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "expected one fused F16 counted-loop WGMMA asm"
    );

    let (asm_operation, asm) = &matching[0];
    let template = asm
        .get_attr_llvm_inline_asm_template(&ctx)
        .map(|value| String::from((*value).clone()))
        .expect("F16 counted-loop WGMMA template");

    assert_eq!(template.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        template
            .matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16")
            .count(),
        1,
        "the F16 counted-loop template must contain exactly one F16 MMA"
    );
    assert!(!template.contains(".bf16.bf16"));
    assert_eq!(
        template.matches("wgmma.commit_group.sync.aligned").count(),
        1
    );
    assert_eq!(
        template.matches("wgmma.wait_group.sync.aligned 0").count(),
        1
    );
    assert!(template.contains("mov.u64 %desc_a, $64;"));
    assert!(template.contains("mov.u64 %desc_b, $65;"));
    assert!(template.contains("add.u64 %desc_a, %desc_a, $66;"));
    assert!(template.contains("add.u64 %desc_b, %desc_b, $67;"));
    assert!(template.contains("mov.u64 %remaining, $68;"));
    assert!(template.contains("@%loop_more bra.uni L__wgmma_done_${:uid};"));
    assert!(template.contains("@%loop_more bra.uni L__wgmma_loop_${:uid};"));
    assert!(template.contains("L__wgmma_done_${:uid}:"));
    assert!(
        !template.contains(".reg .f32")
            && !template.contains("ld.f32")
            && !template.contains("st.f32"),
        "F16 counted-loop WGMMA must keep accumulator memory outside asm: {template}"
    );

    let mut expected_constraints = vec!["=f".to_owned(); ACCUMULATOR_LEN];
    expected_constraints.extend((0..ACCUMULATOR_LEN).map(|index| index.to_string()));
    expected_constraints.extend((0..LOOP_CONTROL_COUNT).map(|_| "l".to_owned()));
    expected_constraints.push("~{memory}".to_owned());
    let expected_constraints = expected_constraints.join(",");
    assert_eq!(
        asm.get_attr_llvm_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .as_deref(),
        Some(expected_constraints.as_str())
    );
    assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
    assert_eq!(
        asm_operation.deref(&ctx).get_num_operands(),
        ACCUMULATOR_LEN + LOOP_CONTROL_COUNT
    );

    let asm_position = body
        .iter()
        .position(|operation| operation == asm_operation)
        .expect("F16 counted-loop WGMMA asm must be in the lowered kernel body");

    let load_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::LoadOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        load_positions.len(),
        ACCUMULATOR_LEN,
        "F16 K-loop accumulator must be loaded exactly once"
    );
    assert!(load_positions.iter().all(|index| *index < asm_position));

    let store_positions = body
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            Operation::get_op::<llvm::StoreOp>(*operation, &ctx)
                .is_some()
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        store_positions.len(),
        ACCUMULATOR_LEN,
        "F16 K-loop accumulator must be stored exactly once after the final wait"
    );
    assert!(store_positions.iter().all(|index| *index > asm_position));

    assert_eq!(
        body.iter()
            .filter(
                |operation| Operation::get_op::<llvm::ExtractValueOp>(**operation, &ctx).is_some()
            )
            .count(),
        ACCUMULATOR_LEN
    );

    Ok(())
}
