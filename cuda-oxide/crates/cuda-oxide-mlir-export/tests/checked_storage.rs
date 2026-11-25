//! Checked arithmetic, enum byte storage, shared globals, and Rust call contracts.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::{
    attributes::{FieldIndexAttr, VariantIndexAttr},
    ops::*,
    types::*,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
/// Append a typed source operation to the fixture block and return its operation handle.
fn append<O: Op>(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    types: Vec<TypeHandle>,
    values: Vec<Value>,
) -> Ptr<Operation> {
    let o = Operation::new(ctx, O::get_concrete_op_info(), types, values, vec![], 0);
    o.insert_at_back(b, ctx);
    o
}
/// Create a named MIR function and its argument-bearing entry block for a storage test.
fn function(
    ctx: &mut Context,
    args: Vec<TypeHandle>,
    returns: Vec<TypeHandle>,
) -> (ModuleOp, Ptr<BasicBlock>) {
    let m = ModuleOp::new(ctx, "storage".try_into().unwrap());
    let ft = FunctionType::get(ctx, args.clone(), returns);
    let f = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(ctx, f, TypeAttr::new(ft.into()))
        .set_symbol_name(ctx, "test".try_into().unwrap());
    m.append_operation(ctx, f, 0);
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(f.deref(ctx).get_region(0), ctx);
    (m, b)
}
/// Translate a prepared source module with the SM100 primitives profile and render its MLIR.
fn render(ctx: &Context, m: &ModuleOp) -> Result<String, String> {
    CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(ctx, m)
        .and_then(|m| render_module(&m).map_err(Into::into))
        .map_err(|e| e.to_string())
}
/// Check overflow intrinsics use source signedness and return the exact padded Rust tuple.
#[test]
fn checked_arithmetic_preserves_signedness_and_padded_overflow_tuple() {
    for signedness in [Signedness::Signed, Signedness::Unsigned] {
        for mul in [false, true] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            let word: TypeHandle = IntegerType::get(&ctx, 64, signedness).into();
            let flag: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
            let tuple: TypeHandle = MirTupleType::get_with_layout(
                &mut ctx,
                vec![word, flag],
                vec![0, 1],
                vec![0, 8],
                16,
                8,
            )
            .into();
            let (m, b) = function(&mut ctx, vec![word, word], vec![tuple]);
            let args = b.deref(&ctx).arguments().collect();
            let op = if mul {
                append::<MirCheckedMulOp>(&mut ctx, b, vec![tuple], args)
            } else {
                append::<MirCheckedAddOp>(&mut ctx, b, vec![tuple], args)
            };
            let v = op.deref(&ctx).get_result(0);
            append::<MirReturnOp>(&mut ctx, b, vec![], vec![v]);
            let text = render(&ctx, &m).unwrap();
            let sign = if signedness == Signedness::Signed {
                "s"
            } else {
                "u"
            };
            let name = if mul { "mul" } else { "add" };
            assert!(
                text.contains(&format!("llvm.{sign}{name}.with.overflow.i64")),
                "{text}"
            );
            assert!(
                text.contains("!llvm.struct<(i64, i1, !llvm.array<7 x i8>)>"),
                "{text}"
            );
            assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 2);
            assert_eq!(text.matches("\"llvm.insertvalue\"").count(), 2);
        }
    }
}
/// Check enum tag and payload accesses use their recorded byte offsets and storage size.
#[test]
fn direct_enum_uses_physical_offsets_for_tag_and_active_payload() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let e: TypeHandle = MirEnumType::get_with_layout(
        &mut ctx,
        "Result".into(),
        tag,
        vec![7, 19],
        vec![
            EnumVariant::new_with_layout("Ok".into(), vec![word], vec![8], vec![8]),
            EnumVariant::new_with_layout("Err".into(), vec![tag], vec![1], vec![1]),
        ],
        0,
        16,
        8,
    )
    .into();
    let (m, b) = function(&mut ctx, vec![word], vec![word]);
    let arg = b.deref(&ctx).get_argument(0);
    let c = append::<MirConstructEnumOp>(&mut ctx, b, vec![e], vec![arg]);
    MirConstructEnumOp::new(c).set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    let v = c.deref(&ctx).get_result(0);
    append::<MirGetDiscriminantOp>(&mut ctx, b, vec![tag], vec![v]);
    let p = append::<MirEnumPayloadOp>(&mut ctx, b, vec![word], vec![v]);
    let p = MirEnumPayloadOp::new(p);
    p.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    p.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    let v = p.get_operation().deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, b, vec![], vec![v]);
    let text = render(&ctx, &m).unwrap();
    assert!(text.contains("value = 7 : i8"));
    assert_eq!(
        text.matches("rawConstantIndices = array<i32: 8>").count(),
        2
    );
    assert_eq!(
        text.matches("rawConstantIndices = array<i32: 0>").count(),
        2
    );
    assert!(text.contains("!llvm.struct<(array<16 x i8>, array<0 x i64>)>"));
}
/// Build repeated shared declarations, optionally introducing a named storage-contract conflict.
fn shared_case(conflict: &str) -> Result<String, String> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let unsigned = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signed = IntegerType::get(&ctx, 32, Signedness::Signed);
    let size_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (m, b) = function(&mut ctx, vec![], vec![]);
    for i in 0..2 {
        let element: TypeHandle = if conflict == "type" && i == 1 {
            signed.into()
        } else {
            unsigned.into()
        };
        let ptr: TypeHandle = MirPtrType::get(&mut ctx, element, true, 3).into();
        let a = append::<MirSharedAllocOp>(&mut ctx, b, vec![ptr], vec![]);
        let a = MirSharedAllocOp::new(a);
        a.set_attr_elem_type(&ctx, TypeAttr::new(element));
        a.set_attr_size(
            &ctx,
            IntegerAttr::new(
                size_ty,
                APInt::from_u64(
                    if conflict == "size" && i == 1 { 8 } else { 4 },
                    64.try_into().unwrap(),
                ),
            ),
        );
        a.set_attr_alloc_key(
            &ctx,
            StringAttr::new(if conflict == "distinct" {
                format!("key{i}")
            } else {
                "key".into()
            }),
        );
        // An absent alignment means the element's natural alignment; the dialect
        // rejects an explicit zero.
        if conflict == "alignment" && i == 1 {
            a.set_alignment_value(&mut ctx, 16);
        }
    }
    append::<MirReturnOp>(&mut ctx, b, vec![], vec![]);
    render(&ctx, &m)
}
/// Check matching shared declarations merge, conflicting ones fail, and an absent alignment uses the natural value.
#[test]
fn shared_declarations_deduplicate_only_identical_storage_and_absent_alignment_means_natural() {
    let text = shared_case("same").unwrap();
    assert_eq!(text.matches("\"llvm.mlir.global\"").count(), 1);
    assert!(text.contains("alignment = 4 : i64"));
    assert!(!text.contains("__source_element_identity"));
    assert_eq!(
        shared_case("distinct")
            .unwrap()
            .matches("\"llvm.mlir.global\"")
            .count(),
        2
    );
    for conflict in ["type", "size", "alignment"] {
        assert!(
            shared_case(conflict)
                .unwrap_err()
                .contains("incompatible declarations"),
            "{conflict}"
        );
    }
}

/// Check dynamic shared storage uses one external AS3 global and byte-correct aligned offsets.
#[test]
fn primitive_dynamic_shared_uses_external_as3_storage_with_alignment_and_offsets() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let byte: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let ptr: TypeHandle = MirPtrType::get(&mut ctx, byte, true, 3).into();
    let (module, block) = function(&mut ctx, vec![], vec![ptr]);
    for (offset, alignment) in [(0, 128), (4096, 1024)] {
        let op = append::<MirExternSharedOp>(&mut ctx, block, vec![ptr], vec![]);
        let shared = MirExternSharedOp::new(op);
        shared.set_alignment_value(&mut ctx, alignment);
        shared.set_byte_offset_value(&mut ctx, offset);
        if offset != 0 {
            let value = op.deref(&ctx).get_result(0);
            append::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);
        }
    }
    let text = render(&ctx, &module).unwrap();
    assert_eq!(text.matches("\"llvm.mlir.global\"").count(), 1, "{text}");
    assert!(
        text.contains("!llvm.array<0 x i8>") && text.contains("alignment = 1024 : i64"),
        "{text}"
    );
    assert!(
        text.contains("#llvm.linkage<external>") && text.contains("addr_space = 3 : i32"),
        "{text}"
    );
    assert!(
        text.contains("4096") && text.contains("\"llvm.getelementptr\""),
        "{text}"
    );
    assert!(
        !text.contains("\"cute") && !text.contains("!cute"),
        "{text}"
    );
}

/// Check an unreachable device contract emits a trap before the LLVM terminator.
#[test]
fn primitive_contract_failure_emits_trap_before_unreachable() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let (module, block) = function(&mut ctx, vec![], vec![]);
    let trap = append::<dialect_nvvm::ops::TrapOp>(&mut ctx, block, vec![], vec![]);
    trap.deref_mut(&ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0295".into()),
    );
    append::<MirUnreachableOp>(&mut ctx, block, vec![], vec![]);
    let text = render(&ctx, &module).unwrap();
    assert!(
        text.find("\"llvm.intr.trap\"")
            .expect("trap must use the registered LLVM intrinsic")
            < text.find("\"llvm.unreachable\"").unwrap(),
        "{text}"
    );
}

/// Check unit-returning calls use a void callee signature while retaining needed empty-value carriers.
#[test]
fn logical_unit_call_uses_void_signature_and_preserves_value_uses() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let unit: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let pointer: TypeHandle = MirPtrType::get(&mut ctx, unit, true, 0).into();
    let (module, block) = function(&mut ctx, vec![pointer], vec![]);
    let destination = block.deref(&ctx).get_argument(0);
    let call = append::<MirCallOp>(&mut ctx, block, vec![unit], vec![destination]);
    MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new("test".into()));
    let value = call.deref(&ctx).get_result(0);
    append::<MirStoreOp>(&mut ctx, block, vec![], vec![destination, value]);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![]);
    let text = render(&ctx, &module).unwrap();
    let call = text
        .lines()
        .find(|line| line.contains("\"func.call\""))
        .unwrap();
    assert!(call.ends_with("-> ()"), "{text}");
    assert!(call.trim().starts_with("\"func.call\""), "{text}");
    let carrier = text
        .lines()
        .find(|line| line.contains("\"ub.poison\""))
        .unwrap();
    let value = carrier.trim().split(" = ").next().unwrap();
    assert!(text.contains(&format!("\"llvm.store\"({value},")), "{text}");
}

/// Keep explicit empty aggregate return values distinct from logical unit calls.
#[test]
fn explicitly_returned_zero_sized_aggregates_keep_their_call_result() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let structure: TypeHandle =
        MirStructType::get(&mut ctx, "Marker".into(), vec![], vec![]).into();
    for result in [tuple, structure] {
        let (module, block) = function(&mut ctx, vec![], vec![result]);
        let call = append::<MirCallOp>(&mut ctx, block, vec![result], vec![]);
        MirCallOp::new(call).set_attr_callee(&ctx, StringAttr::new("test".into()));
        let value = call.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);
        let text = render(&ctx, &module).unwrap();
        let call = text
            .lines()
            .find(|line| line.contains("\"func.call\""))
            .unwrap();
        assert!(
            call.contains(" = ") && call.ends_with("-> (!llvm.struct<()>)"),
            "{text}"
        );
        assert!(!text.contains("\"ub.poison\""), "{text}");
    }
}

/// Check reviewed scalar intrinsics become target operations without unresolved helper symbols.
#[test]
fn rust_scalar_intrinsics_are_operations_not_external_calls() {
    use dialect_mir::rust_intrinsics::{CALLEE_SATURATING_SUB, CALLEE_SELECT_UNPREDICTABLE};
    for sign in [Signedness::Signed, Signedness::Unsigned] {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let word: TypeHandle = IntegerType::get(&ctx, 32, sign).into();
        let flag: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let (m, b) = function(&mut ctx, vec![flag, word, word], vec![word]);
        let args: Vec<_> = b.deref(&ctx).arguments().collect();
        let select = append::<MirCallOp>(&mut ctx, b, vec![word], args.clone());
        MirCallOp::new(select)
            .set_attr_callee(&ctx, StringAttr::new(CALLEE_SELECT_UNPREDICTABLE.into()));
        let selected = select.deref(&ctx).get_result(0);
        let sub = append::<MirCallOp>(&mut ctx, b, vec![word], vec![selected, args[2]]);
        MirCallOp::new(sub).set_attr_callee(&ctx, StringAttr::new(CALLEE_SATURATING_SUB.into()));
        let result = sub.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, b, vec![], vec![result]);
        let text = render(&ctx, &m).unwrap();
        assert!(text.contains("\"arith.select\""), "{text}");
        let sign = if sign == Signedness::Signed { "s" } else { "u" };
        assert!(text.contains(&format!("llvm.{sign}sub.sat.i32")), "{text}");
        assert!(!text.contains("__cuda_oxide_rust_intrinsic_"), "{text}");
    }
}

/// Check constants introduced by unrolling preserve their complete source bit patterns.
#[test]
fn unrolling_builtin_integer_constants_retain_width_and_bits() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let word = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let (m, b) = function(&mut ctx, vec![], vec![word.into()]);
    let constant = pliron::builtin::ops::ConstantOp::new(
        &mut ctx,
        Box::new(IntegerAttr::new(
            word,
            APInt::from_u64(u32::MAX as u64, 32.try_into().unwrap()),
        )),
    );
    constant.get_operation().insert_at_back(b, &ctx);
    let value = constant.get_operation().deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, b, vec![], vec![value]);
    let text = render(&ctx, &m).unwrap();
    assert!(
        text.contains("\"arith.constant\"() <{value = -1 : i32}>"),
        "{text}"
    );
    assert!(!text.contains("builtin.constant"), "{text}");
}

/// Check the Rust maximum maps to native NaN-ignoring FP32 behavior without FTZ modifiers.
#[test]
fn rust_fp32_max_preserves_nan_ignoring_and_subnormal_contract() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let scalar: TypeHandle = pliron::builtin::types::FP32Type::get(&ctx).into();
    let (module, block) = function(&mut ctx, vec![scalar, scalar], vec![scalar]);
    let args = block.deref(&ctx).arguments().collect();
    let call = append::<MirCallOp>(&mut ctx, block, vec![scalar], args);
    MirCallOp::new(call).set_attr_callee(
        &ctx,
        StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_MAXNUM_NSZ_F32.into()),
    );
    let value = call.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);
    let text = render(&ctx, &module).unwrap();
    let native = text
        .lines()
        .find(|line| line.contains("\"nvvm.fmax\""))
        .unwrap();
    assert!(
        !native.contains("nan") && !native.contains("ftz") && !native.contains("abs"),
        "{text}"
    );
    assert!(
        !text.contains("llvm.maxnum") && !text.contains("__cuda_oxide_rust_intrinsic_"),
        "{text}"
    );
}

/// Reject attempts to use the FP32 maximum mapping with incompatible source types.
#[test]
fn rust_fp32_max_rejects_other_signatures() {
    for (operand_width, result_width, operand_count) in [(64, 32, 2), (32, 64, 2), (32, 32, 1)] {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let fp32: TypeHandle = pliron::builtin::types::FP32Type::get(&ctx).into();
        let fp64: TypeHandle = pliron::builtin::types::FP64Type::get(&ctx).into();
        let operand = if operand_width == 32 { fp32 } else { fp64 };
        let result = if result_width == 32 { fp32 } else { fp64 };
        let (module, block) = function(&mut ctx, vec![operand; operand_count], vec![result]);
        let args = block.deref(&ctx).arguments().collect();
        let call = append::<MirCallOp>(&mut ctx, block, vec![result], args);
        MirCallOp::new(call).set_attr_callee(
            &ctx,
            StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_MAXNUM_NSZ_F32.into()),
        );
        let value = call.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);
        let error = render(&ctx, &module).unwrap_err();
        assert!(
            error.contains("requires exactly two FP32 operands and one FP32 result"),
            "{error}"
        );
    }
}
