/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust array construction needs byte-preserving unions, slices and aggregate casts.

// These fixtures intentionally forge pointer contracts to test verifier rejection.
#![allow(clippy::disallowed_methods)]

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use cuda_oxide_mlir_export::{MlirAttribute, MlirOperation, MlirType, render_module};
use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr},
    ops::*,
    types::*,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness, UnitType},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};

/// Create a source context with the dialects used by the full CuTe profile.
fn context() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    ctx
}

/// Append an operation and return its handle for setting semantic attributes.
fn append<O: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    types: Vec<TypeHandle>,
    values: Vec<Value>,
) -> Ptr<Operation> {
    let operation = Operation::new(ctx, O::get_concrete_op_info(), types, values, vec![], 0);
    operation.insert_at_back(block, ctx);
    operation
}

/// Create one ordinary MIR function in a module.
fn function(
    ctx: &mut Context,
    args: Vec<TypeHandle>,
    results: Vec<TypeHandle>,
) -> (ModuleOp, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "register_storage".try_into().unwrap());
    let signature = FunctionType::get(ctx, args.clone(), results);
    let function = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(ctx, function, TypeAttr::new(signature.into()))
        .set_symbol_name(ctx, "test".try_into().unwrap());
    module.append_operation(ctx, function, 0);
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(function.deref(ctx).get_region(0), ctx);
    (module, block)
}

/// Translate through the production SM120 CuTe profile.
fn render(ctx: &Context, module: &ModuleOp) -> Result<String, String> {
    CutlassFullCuteMlir22::new("sm_120a")
        .unwrap()
        .translate_module(ctx, module)
        .and_then(|target| render_module(&target).map_err(Into::into))
        .map_err(|error| error.to_string())
}

/// Model the standard library's uninitialized-or-value storage layout.
fn maybe_uninit(ctx: &mut Context, value: TypeHandle, size: u64, align: u64) -> TypeHandle {
    let unit = UnitType::get(ctx).into();
    MirUnionType::get(
        ctx,
        "MaybeUninit".into(),
        vec!["uninit".into(), "value".into()],
        vec![unit, value],
        size,
        align,
    )
    .into()
}

/// Field one of a union overlaps field zero; it is not LLVM aggregate slot one.
#[test]
fn union_value_insert_and_extract_share_byte_zero() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let union = maybe_uninit(&mut ctx, scalar, 4, 4);
    let (module, block) = function(&mut ctx, vec![scalar], vec![scalar]);
    let argument = block.deref(&ctx).get_argument(0);
    let undef = append::<MirUndefOp>(&mut ctx, block, vec![union], vec![]);
    let empty = undef.deref(&ctx).get_result(0);
    let insert = append::<MirInsertFieldOp>(&mut ctx, block, vec![union], vec![empty, argument]);
    MirInsertFieldOp::new(insert).set_attr_insert_index(&ctx, FieldIndexAttr(1));
    let initialized = insert.deref(&ctx).get_result(0);
    let extract = append::<MirExtractFieldOp>(&mut ctx, block, vec![scalar], vec![initialized]);
    MirExtractFieldOp::new(extract).set_attr_index(&ctx, FieldIndexAttr(1));
    let result = extract.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
    let text = render(&ctx, &module).unwrap();
    assert!(text.contains("array<4 x i8>"), "{text}");
    assert!(text.contains("array<0 x i32>"), "{text}");
    assert!(
        text.contains("\"llvm.store\"") && text.contains("\"llvm.load\""),
        "{text}"
    );
    assert!(!text.contains("position = array<i64: 1>"), "{text}");
}

/// `array::from_fn` reinterprets initialized storage without numeric conversion.
#[test]
fn initialized_array_transmute_preserves_storage_and_alignment() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let union = maybe_uninit(&mut ctx, scalar, 4, 4);
    let source = MirArrayType::get(&mut ctx, union, 4).into();
    let target = MirArrayType::get(&mut ctx, scalar, 4).into();
    let (module, block) = function(&mut ctx, vec![source], vec![target]);
    let argument = block.deref(&ctx).get_argument(0);
    let cast = append::<MirCastOp>(&mut ctx, block, vec![target], vec![argument]);
    MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
    let result = cast.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
    let text = render(&ctx, &module).unwrap();
    assert!(text.contains("!llvm.array<4 x f32>"), "{text}");
    assert!(
        text.contains("\"llvm.store\"") && text.contains("\"llvm.load\""),
        "{text}"
    );
    assert!(
        !text.contains("arith.sitofp") && !text.contains("arith.fptosi"),
        "{text}"
    );
}

/// The mutable slice used by array initialization preserves both carrier fields.
#[test]
fn constructed_slice_keeps_pointer_and_length() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let union = maybe_uninit(&mut ctx, scalar, 4, 4);
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, union, true, 0, MirPointerKind::RawMut).into();
    let slice = MirSliceType::get_with_kind(&mut ctx, union, MirPointerKind::RawMut).into();
    let length = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let (module, block) = function(&mut ctx, vec![pointer, length], vec![slice]);
    let args = block.deref(&ctx).arguments().collect();
    let construct = append::<MirConstructSliceOp>(&mut ctx, block, vec![slice], args);
    let result = construct.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
    let text = render(&ctx, &module).unwrap();
    assert_eq!(text.matches("\"llvm.insertvalue\"").count(), 2, "{text}");
    assert!(text.contains("position = array<i64: 0>"), "{text}");
    assert!(text.contains("position = array<i64: 1>"), "{text}");
    assert!(text.contains("!llvm.struct<(!llvm.ptr, i64)>"), "{text}");
}

/// Array-to-slice coercion retains the original pointer and supplies the static extent.
#[test]
fn array_unsizing_preserves_extent_and_pointer_kind() {
    for (length, kind) in [
        (0, MirPointerKind::SharedRef),
        (7, MirPointerKind::UniqueRef),
    ] {
        let mut ctx = context();
        let element = FP32Type::get(&ctx).into();
        let array = MirArrayType::get(&mut ctx, element, length).into();
        let mutable = kind == MirPointerKind::UniqueRef;
        let pointer = MirPtrType::get_with_kind(&mut ctx, array, mutable, 0, kind).into();
        let slice = MirSliceType::get_with_kind(&mut ctx, element, kind).into();
        let (module, block) = function(&mut ctx, vec![pointer], vec![slice]);
        let argument = block.deref(&ctx).get_argument(0);
        let cast = append::<MirCastOp>(&mut ctx, block, vec![slice], vec![argument]);
        let semantic = MirCastOp::new(cast);
        semantic.set_attr_cast_kind(&ctx, MirCastKindAttr::PointerCoercionUnsize);
        semantic.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
        let result = cast.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
        let text = render(&ctx, &module).unwrap();
        assert_eq!(text.matches("\"llvm.insertvalue\"").count(), 2, "{text}");
        assert!(text.contains(&format!("value = {length} : i64")), "{text}");
        assert!(
            !text.contains("llvm.alloca") && !text.contains("llvm.load"),
            "{text}"
        );
    }
}

/// Coercion cannot erase a named address space or strengthen a borrow.
#[test]
fn array_unsizing_rejects_incompatible_pointer_contracts() {
    for (space, target_kind, wrong_element, not_array) in [
        (3, MirPointerKind::SharedRef, false, false),
        (0, MirPointerKind::UniqueRef, false, false),
        (0, MirPointerKind::SharedRef, true, false),
        (0, MirPointerKind::SharedRef, false, true),
    ] {
        let mut ctx = context();
        let element = FP32Type::get(&ctx).into();
        let pointee = if not_array {
            element
        } else {
            MirArrayType::get(&mut ctx, element, 7).into()
        };
        let pointer =
            MirPtrType::get_with_kind(&mut ctx, pointee, false, space, MirPointerKind::SharedRef)
                .into();
        let target_element = if wrong_element {
            MirFP16Type::get(&ctx).into()
        } else {
            element
        };
        let slice = MirSliceType::get_with_kind(&mut ctx, target_element, target_kind).into();
        let (module, block) = function(&mut ctx, vec![pointer], vec![slice]);
        let argument = block.deref(&ctx).get_argument(0);
        let cast = append::<MirCastOp>(&mut ctx, block, vec![slice], vec![argument]);
        let semantic = MirCastOp::new(cast);
        semantic.set_attr_cast_kind(&ctx, MirCastKindAttr::PointerCoercionUnsize);
        semantic.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
        let result = cast.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
        assert!(render(&ctx, &module).is_err());
    }
}

/// Infallible array construction retains an uninhabited residual in unreachable IR.
#[test]
fn empty_residual_has_zero_storage_but_no_discriminant() {
    let mut ctx = context();
    let tag = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let empty = MirEnumType::get_with_encoding(
        &mut ctx,
        "NeverShortCircuitResidual".into(),
        tag,
        vec![],
        vec![],
        EnumEncoding {
            abi_align: 1,
            layout_kind: EnumLayoutKind::Empty,
            ..Default::default()
        },
    )
    .into();
    let unit = UnitType::get(&ctx).into();
    let empty_flow = MirEnumType::get_with_encoding(
        &mut ctx,
        "ControlFlow".into(),
        tag,
        vec![0, 1],
        vec![
            EnumVariant::new_with_layout("Continue".into(), vec![unit], vec![0], vec![0]),
            EnumVariant::new_with_layout("Break".into(), vec![empty], vec![0], vec![0]),
        ],
        EnumEncoding {
            abi_align: 1,
            layout_kind: EnumLayoutKind::Empty,
            variant_inhabited: vec![0, 0],
            ..Default::default()
        },
    )
    .into();
    let (flow_module, flow_block) = function(&mut ctx, vec![empty_flow], vec![]);
    append::<MirUnreachableOp>(&mut ctx, flow_block, vec![], vec![]);
    assert!(render(&ctx, &flow_module).is_ok());
    let (module, block) = function(&mut ctx, vec![empty], vec![]);
    append::<MirUnreachableOp>(&mut ctx, block, vec![], vec![]);
    let text = render(&ctx, &module).unwrap();
    assert!(text.contains("array<0 x i8>"), "{text}");
    let (module, block) = function(&mut ctx, vec![empty], vec![tag]);
    let value = block.deref(&ctx).get_argument(0);
    let discriminant = append::<MirGetDiscriminantOp>(&mut ctx, block, vec![tag], vec![value]);
    let result = discriminant.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
    assert!(render(&ctx, &module).is_err());
}

/// Reject unreviewed byte representations and malformed ABI metadata.
#[test]
fn union_storage_rejects_bool_pointer_and_invalid_layouts() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let flag = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let pointer = MirPtrType::get_generic(&mut ctx, scalar, false).into();
    for (value, size, align) in [
        (flag, 1, 1),
        (pointer, 8, 8),
        (scalar, 2, 2),
        (scalar, 4, 3),
    ] {
        let union = maybe_uninit(&mut ctx, value, size, align);
        let (module, block) = function(&mut ctx, vec![union], vec![union]);
        let argument = block.deref(&ctx).get_argument(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![argument]);
        assert!(
            render(&ctx, &module).is_err(),
            "accepted invalid union size={size} align={align}"
        );
    }
}

/// Aggregate reinterpretation must not read past storage or bypass byte adapters.
#[test]
fn aggregate_transmute_rejects_size_mismatch_and_bool_storage() {
    let mut ctx = context();
    let word = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let flag = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let byte = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let words = MirArrayType::get(&mut ctx, word, 4).into();
    let short = MirArrayType::get(&mut ctx, word, 3).into();
    let flags = MirArrayType::get(&mut ctx, flag, 4).into();
    let bytes = MirArrayType::get(&mut ctx, byte, 4).into();
    for (source, target) in [(words, short), (flags, bytes), (bytes, flags)] {
        let (module, block) = function(&mut ctx, vec![source], vec![target]);
        let argument = block.deref(&ctx).get_argument(0);
        let cast = append::<MirCastOp>(&mut ctx, block, vec![target], vec![argument]);
        MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
        let result = cast.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
        assert!(render(&ctx, &module).is_err());
    }
}

/// Return the exported function so storage tests can check operand identity and order.
fn exported_function(ctx: &Context, module: &ModuleOp) -> MlirOperation {
    fn find(operation: &MlirOperation) -> Option<&MlirOperation> {
        if operation.name == "func.func" {
            return Some(operation);
        }
        operation
            .regions
            .iter()
            .flat_map(|region| &region.blocks)
            .flat_map(|block| &block.operations)
            .find_map(find)
    }
    let target = CutlassFullCuteMlir22::new("sm_120a")
        .unwrap()
        .translate_module(ctx, module)
        .unwrap();
    find(&target.root)
        .expect("fixture contains one function")
        .clone()
}

/// Taking either union field's address must retain its space and overlap at byte zero.
#[test]
fn union_field_addresses_overlap_without_changing_address_space() {
    for address_space in [0, 3] {
        let mut ctx = context();
        let wide = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let narrow = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let union = MirUnionType::get(
            &mut ctx,
            "OverlappingWords".into(),
            vec!["wide".into(), "narrow".into()],
            vec![wide, narrow],
            8,
            8,
        )
        .into();
        let pointer = MirPtrType::get(&mut ctx, union, true, address_space).into();
        let wide_pointer = MirPtrType::get(&mut ctx, wide, true, address_space).into();
        let narrow_pointer = MirPtrType::get(&mut ctx, narrow, true, address_space).into();
        let (module, block) = function(&mut ctx, vec![pointer], vec![wide_pointer, narrow_pointer]);
        let base = block.deref(&ctx).get_argument(0);
        let mut results = Vec::new();
        for (index, result_type) in [wide_pointer, narrow_pointer].into_iter().enumerate() {
            let address = MirFieldAddrOp::build(&mut ctx, base, result_type, index as u32).unwrap();
            address.insert_at_back(block, &ctx);
            results.push(address.deref(&ctx).get_result(0));
        }
        append::<MirReturnOp>(&mut ctx, block, vec![], results);

        let target = exported_function(&ctx, &module);
        let entry = &target.regions[0].blocks[0];
        let addresses: Vec<_> = entry
            .operations
            .iter()
            .filter(|operation| operation.name == "llvm.getelementptr")
            .collect();
        assert_eq!(addresses.len(), 2);
        for address in &addresses {
            assert_eq!(address.operands[0].id, entry.arguments[0].id);
            assert_ne!(address.results[0].id, entry.arguments[0].id);
            assert_eq!(address.results[0].ty, entry.arguments[0].ty);
            assert_eq!(
                address.properties.get("elem_type"),
                Some(&MlirAttribute::Type(MlirType::Integer(8))),
            );
            assert_eq!(
                address.properties.get("rawConstantIndices"),
                Some(&MlirAttribute::DenseI32Array(vec![0])),
            );
        }
        assert_ne!(addresses[0].results[0].id, addresses[1].results[0].id);
        assert!(
            !entry
                .operations
                .iter()
                .any(|operation| operation.name == "llvm.addrspacecast")
        );
    }
}

/// A short field write retains the high bytes of the existing wider union value.
#[test]
fn partial_union_overwrite_preserves_untouched_bytes() {
    let mut ctx = context();
    let wide = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let narrow = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let union = MirUnionType::get(
        &mut ctx,
        "PartialOverwrite".into(),
        vec!["wide".into(), "narrow".into()],
        vec![wide, narrow],
        8,
        8,
    )
    .into();
    let (module, block) = function(&mut ctx, vec![union, narrow], vec![wide]);
    let args: Vec<_> = block.deref(&ctx).arguments().collect();
    let insert = append::<MirInsertFieldOp>(&mut ctx, block, vec![union], args);
    MirInsertFieldOp::new(insert).set_attr_insert_index(&ctx, FieldIndexAttr(1));
    let updated = insert.deref(&ctx).get_result(0);
    let extract = append::<MirExtractFieldOp>(&mut ctx, block, vec![wide], vec![updated]);
    MirExtractFieldOp::new(extract).set_attr_index(&ctx, FieldIndexAttr(0));
    let value = extract.deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, block, vec![], vec![value]);

    let target = exported_function(&ctx, &module);
    let entry = &target.regions[0].blocks[0];
    let memory: Vec<_> = entry
        .operations
        .iter()
        .filter(|operation| matches!(operation.name.as_str(), "llvm.store" | "llvm.load"))
        .collect();
    assert_eq!(
        memory
            .iter()
            .map(|operation| operation.name.as_str())
            .collect::<Vec<_>>(),
        [
            "llvm.store",
            "llvm.store",
            "llvm.load",
            "llvm.store",
            "llvm.load"
        ],
    );
    // Preserve the complete original carrier before overwriting its low word.
    assert_eq!(memory[0].operands[0].id, entry.arguments[0].id);
    assert_eq!(memory[1].operands[0].id, entry.arguments[1].id);
    assert_eq!(memory[1].operands[0].ty, MlirType::Integer(32));
    assert_eq!(memory[0].operands[1], memory[1].operands[1]);
    assert_eq!(memory[2].operands[0], memory[1].operands[1]);
    // The wider read must observe that updated carrier, including untouched bytes.
    assert_eq!(memory[3].operands[0].id, memory[2].results[0].id);
    assert_eq!(memory[4].operands[0], memory[3].operands[1]);
    assert_eq!(memory[4].results[0].ty, MlirType::Integer(64));
}

/// Nesting unsupported storage inside an array, tuple or struct cannot hide it.
#[test]
fn byte_storage_rejects_nested_pointer_and_bool_payloads() {
    let mut ctx = context();
    let word = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let flag = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let byte = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let pointer = MirPtrType::get_generic(&mut ctx, word, false).into();
    let pointers = MirArrayType::get(&mut ctx, pointer, 1).into();
    let tuple = MirTupleType::get(&mut ctx, vec![pointers]).into();
    let structure = MirStructType::get(
        &mut ctx,
        "PointerWrapper".into(),
        vec!["pointer".into()],
        vec![pointer],
    )
    .into();
    let flags = MirArrayType::get(&mut ctx, flag, 1).into();
    let nested_flags = MirArrayType::get(&mut ctx, flags, 1).into();
    let bytes = MirArrayType::get(&mut ctx, byte, 1).into();
    let words = MirArrayType::get(&mut ctx, word, 1).into();
    for (payload, target, size, align, reason) in [
        (pointers, words, 8, 8, "pointer"),
        (tuple, words, 8, 8, "pointer"),
        (structure, words, 8, 8, "pointer"),
        (nested_flags, bytes, 1, 1, "bool"),
    ] {
        let union = maybe_uninit(&mut ctx, payload, size, align);
        let (module, block) = function(&mut ctx, vec![union], vec![union]);
        let argument = block.deref(&ctx).get_argument(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![argument]);
        let error = render(&ctx, &module).unwrap_err();
        assert!(error.contains(reason), "{error}");

        for (source, destination) in [(payload, target), (target, payload)] {
            let (module, block) = function(&mut ctx, vec![source], vec![destination]);
            let argument = block.deref(&ctx).get_argument(0);
            let cast = append::<MirCastOp>(&mut ctx, block, vec![destination], vec![argument]);
            MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
            let result = cast.deref(&ctx).get_result(0);
            append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
            let error = render(&ctx, &module).unwrap_err();
            assert!(error.contains(reason), "{error}");
        }
    }
}

/// Slice construction cannot erase address spaces or invent a compatible element/kind/length.
#[test]
fn constructed_slice_rejects_incompatible_pointer_and_length() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let other = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let length64 = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let length32 = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, scalar, true, 0, MirPointerKind::RawMut).into();
    let shared =
        MirPtrType::get_with_kind(&mut ctx, scalar, true, 3, MirPointerKind::RawMut).into();
    let wrong_element =
        MirPtrType::get_with_kind(&mut ctx, other, true, 0, MirPointerKind::RawMut).into();
    let immutable =
        MirPtrType::get_with_kind(&mut ctx, scalar, false, 0, MirPointerKind::RawConst).into();
    let slice = MirSliceType::get_with_kind(&mut ctx, scalar, MirPointerKind::RawMut).into();
    for (pointer, length, reason) in [
        (shared, length64, "address space"),
        (wrong_element, length64, "pointee mismatch"),
        (immutable, length64, "pointer kind"),
        (pointer, length32, "64-bit length"),
        (pointer, scalar, "integer type"),
    ] {
        let (module, block) = function(&mut ctx, vec![pointer, length], vec![slice]);
        let arguments = block.deref(&ctx).arguments().collect();
        let construct = append::<MirConstructSliceOp>(&mut ctx, block, vec![slice], arguments);
        let result = construct.deref(&ctx).get_result(0);
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![result]);
        let error = render(&ctx, &module).unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
}
