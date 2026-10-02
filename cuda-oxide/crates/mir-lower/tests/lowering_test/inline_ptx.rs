/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

#[test]
fn test_inline_ptx_op_lowers_to_inline_asm_attrs() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let input = entry.deref(&ctx).get_argument(0);

    let inline_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "add.u32 $0, $1, $1;",
        "=r,r",
        true,
        true,
    );
    inline_ptx.insert_at_back(entry, &ctx);
    let register_only_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "mul.lo.u32 $0, $1, $1;",
        "=r,r",
        false,
        true,
    );
    register_only_ptx.insert_at_back(entry, &ctx);
    let may_diverge_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "cvt.u32.u32 $0, $1;",
        "=r,r",
        false,
        false,
    );
    may_diverge_ptx.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_conservative = false;
    let mut found_register_only = false;
    let mut found_may_diverge = false;
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
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|s| String::from((*s).clone()));
                match template.as_deref() {
                    Some("add.u32 $0, $1, $1;") => {
                        found_conservative = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_llvm_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            inline_asm
                                .get_attr_llvm_inline_asm_attrs(&ctx)
                                .is_some_and(|attrs| attrs.has("convergent"))
                        );
                        assert!(llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    Some("mul.lo.u32 $0, $1, $1;") => {
                        found_register_only = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_llvm_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            inline_asm
                                .get_attr_llvm_inline_asm_attrs(&ctx)
                                .is_some_and(|attrs| attrs.has("convergent"))
                        );
                        assert!(!llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    Some("cvt.u32.u32 $0, $1;") => {
                        found_may_diverge = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_llvm_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            !inline_asm
                                .get_attr_llvm_inline_asm_attrs(&ctx)
                                .is_some_and(|attrs| attrs.has("convergent"))
                        );
                        assert!(!llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    _ => continue,
                }
            }
        }
    }

    assert!(
        found_conservative,
        "Expected conservative inline PTX asm op"
    );
    assert!(
        found_register_only,
        "Expected register-only inline PTX asm op"
    );
    assert!(found_may_diverge, "Expected may-diverge inline PTX asm op");
    Ok(())
}

#[test]
fn test_multi_result_inline_ptx_lowers_to_struct_asm_and_extractvalues() -> Result<(), anyhow::Error>
{
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let input = entry.deref(&ctx).get_argument(0);

    let inline_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into(), i32_ty.into()],
        vec![input],
        "add.u32 $0, $2, $2; mul.lo.u32 $1, $2, $2;",
        "=r,=r,r",
        true,
        false,
    );
    let expected_location = Location::Named {
        name: "source-inline-ptx".to_string(),
        child_loc: Box::new(Location::Unknown),
    };
    inline_ptx
        .deref_mut(&ctx)
        .set_loc(expected_location.clone());
    inline_ptx.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut asm_result = None;
    let mut extract_indices = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
            assert_eq!(
                inline_asm.get_operation().deref(&ctx).loc(),
                expected_location
            );
            assert_eq!(
                inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|s| String::from((*s).clone()))
                    .as_deref(),
                Some("add.u32 $0, $2, $2; mul.lo.u32 $1, $2, $2;")
            );
            assert_eq!(
                inline_asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|s| String::from((*s).clone()))
                    .as_deref(),
                Some("=r,=r,r")
            );
            let result = inline_asm.get_operation().deref(&ctx).get_result(0);
            let result_ty = result.get_type(&ctx);
            let result_ty = result_ty.deref(&ctx);
            let struct_ty = result_ty
                .downcast_ref::<llvm_types::StructType>()
                .expect("multi-output inline PTX must return an LLVM struct");
            assert_eq!(struct_ty.num_fields(), 2);
            for index in 0..2 {
                assert_eq!(
                    struct_ty
                        .field_type(index)
                        .deref(&ctx)
                        .downcast_ref::<IntegerType>()
                        .expect("multi-output inline PTX struct field must stay i32")
                        .width(),
                    32
                );
            }
            asm_result = Some(result);
        } else if let Some(extract) = Operation::get_op::<llvm::ExtractValueOp>(op, &ctx) {
            assert_eq!(extract.get_operation().deref(&ctx).loc(), expected_location);
            let aggregate = extract.get_operation().deref(&ctx).get_operand(0);
            assert_eq!(
                Some(aggregate),
                asm_result,
                "extractvalue must consume the struct-returning asm result"
            );
            extract_indices.push(extract.indices(&ctx));
        }
    }

    assert!(
        asm_result.is_some(),
        "Expected struct-returning inline PTX asm op"
    );
    assert_eq!(
        extract_indices,
        vec![vec![0], vec![1]],
        "each output must be extracted once, in constraint order"
    );
    Ok(())
}

#[test]
fn test_inline_ptx_supports_thirty_two_tied_f32_results() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::FP32Type;
    use pliron::r#type::Typed;

    const ACCUMULATOR_LEN: usize = 32;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(); ACCUMULATOR_LEN]);

    let inputs = (0..ACCUMULATOR_LEN)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();

    let template = (0..ACCUMULATOR_LEN)
        .map(|index| format!("mov.f32 ${index}, ${index};"))
        .collect::<Vec<_>>()
        .join(" ");

    let mut constraints = vec!["=f".to_string(); ACCUMULATOR_LEN];
    constraints.extend((0..ACCUMULATOR_LEN).map(|index| index.to_string()));
    let constraints = constraints.join(",");

    let inline_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![f32_ty.into(); ACCUMULATOR_LEN],
        inputs,
        &template,
        &constraints,
        true,
        true,
    );
    inline_ptx.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut asm_result = None;
    let mut extract_indices = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
            assert_eq!(
                inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .as_deref(),
                Some(template.as_str()),
            );

            assert_eq!(
                inline_asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .as_deref(),
                Some(constraints.as_str()),
            );

            let result = inline_asm.get_operation().deref(&ctx).get_result(0);
            let result_ty = result.get_type(&ctx);
            let result_ty = result_ty.deref(&ctx);

            let struct_ty = result_ty
                .downcast_ref::<llvm_types::StructType>()
                .expect("32-output inline PTX must return an LLVM struct");

            assert_eq!(
                struct_ty.num_fields(),
                ACCUMULATOR_LEN,
                "inline PTX must return exactly 32 accumulator values",
            );

            for index in 0..ACCUMULATOR_LEN {
                assert!(
                    struct_ty
                        .field_type(index)
                        .deref(&ctx)
                        .downcast_ref::<FP32Type>()
                        .is_some(),
                    "inline PTX result field {index} must remain f32",
                );
            }

            assert!(
                asm_result.replace(result).is_none(),
                "expected exactly one inline PTX operation",
            );
        } else if let Some(extract) = Operation::get_op::<llvm::ExtractValueOp>(op, &ctx) {
            let aggregate = extract.get_operation().deref(&ctx).get_operand(0);

            assert_eq!(
                Some(aggregate),
                asm_result,
                "extractvalue must consume the struct-returning asm result",
            );

            extract_indices.push(extract.indices(&ctx));
        }
    }

    assert!(
        asm_result.is_some(),
        "expected a struct-returning inline PTX asm operation",
    );

    let expected_indices = (0..ACCUMULATOR_LEN)
        .map(|index| vec![index as u32])
        .collect::<Vec<_>>();

    assert_eq!(
        extract_indices, expected_indices,
        "each accumulator output must be extracted exactly once and in constraint order",
    );

    Ok(())
}
