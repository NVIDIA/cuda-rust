/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::num::NonZero;

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_mir::{
    attributes::ReferenceParamValidityAttr,
    ops::{MirFuncOp, MirReturnOp},
    types::{MirArrayType, MirPointerKind, MirPtrType, MirSliceType, MirStructType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
};
use pliron_mlir_export::{MlirAttribute, render_module};
use reserved_oxide_symbols::{
    MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX, MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};

#[allow(clippy::disallowed_methods)] // Synthetic fixture models rustc-proven reference metadata.
fn fixture(case: &str) -> Result<(String, Vec<MlirAttribute>), String> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let byte: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let word_type = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let u64: TypeHandle = word_type.into();
    let bytes: TypeHandle =
        MirArrayType::get(&mut ctx, byte, if case == "zero" { 0 } else { 128 }).into();
    let descriptor: TypeHandle = if case == "zero" {
        bytes
    } else {
        MirStructType::get_with_full_layout(
            &mut ctx,
            "Descriptor".into(),
            vec!["bytes".into()],
            vec![bytes],
            vec![0],
            vec![0],
            128,
            64,
        )
        .into()
    };
    let pointer: TypeHandle = MirPtrType::get(
        &mut ctx,
        descriptor,
        case == "mutable",
        if case == "space" { 1 } else { 0 },
    )
    .into();
    let slice: TypeHandle =
        MirSliceType::get_with_kind(&mut ctx, byte, MirPointerKind::SharedRef).into();
    let args = vec![
        slice,
        if case == "scalar" { u64 } else { pointer },
        u64,
        pointer,
    ];
    let module = ModuleOp::new(&mut ctx, "grid_constants".try_into().unwrap());
    let function_type = FunctionType::get(&ctx, args.clone(), vec![]);
    let function_op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
    function.set_symbol_name(&mut ctx, "copy_maps".try_into().unwrap());
    if case != "nonkernel" {
        function_op.deref_mut(&ctx).attributes.set(
            "gpu_kernel".try_into().unwrap(),
            StringAttr::new("true".into()),
        );
        function.set_reference_param_validity(&mut ctx, 0, ReferenceParamValidityAttr(1));
    }
    for index in [1, 3] {
        let suffix = if case == "noncanonical" && index == 1 {
            "01".into()
        } else if case == "range" && index == 1 {
            "4".into()
        } else {
            index.to_string()
        };
        let mut op = function_op.deref_mut(&ctx);
        op.attributes.set(
            format!("{MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX}{suffix}")
                .as_str()
                .try_into()
                .unwrap(),
            TypeAttr::new(if case == "pointee" { bytes } else { descriptor }),
        );
        if case != "incomplete" || index != 1 {
            op.attributes.set(
                format!("{MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX}{suffix}")
                    .as_str()
                    .try_into()
                    .unwrap(),
                IntegerAttr::new(
                    word_type,
                    APInt::from_u64(
                        if case == "alignment" { 3 } else { 64 },
                        NonZero::new(64).unwrap(),
                    ),
                ),
            );
        }
    }
    module.append_operation(&mut ctx, function_op, 0);
    let entry = BasicBlock::new(&mut ctx, None, args);
    entry.insert_at_back(function_op.deref(&ctx).get_region(0), &ctx);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, &ctx);
    let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
    let target = profile
        .translate_module(&ctx, &module)
        .map_err(|e| e.to_string())?;
    let device = &target.root.regions[0].blocks[0].operations[0];
    let function = &device.regions[0].blocks[0].operations[0];
    let Some(MlirAttribute::Array(attributes)) = function.properties.get("arg_attrs") else {
        return Err("missing argument attributes".into());
    };
    Ok((render_module(&target), attributes.clone()))
}

#[test]
fn descriptors_remain_by_value_and_grid_constant_after_slice_flattening() {
    let (text, attrs) = fixture("valid").unwrap();
    assert_eq!(attrs.len(), 5);
    for index in [0, 1, 3] {
        assert_eq!(attrs[index], MlirAttribute::Dictionary(Default::default()));
    }
    assert_eq!(attrs[2], attrs[4]);
    assert!(
        text.contains("llvm.byval = !llvm.array<128 x i8>"),
        "{text}"
    );
    assert!(text.contains("llvm.align = 64 : i64"), "{text}");
    assert!(text.contains("nvvm.grid_constant"), "{text}");
    assert!(
        !text.contains(MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX),
        "{text}"
    );
    assert!(!text.contains("reference_param_validity"), "{text}");
    assert!(
        text.contains("llvm.insertvalue"),
        "slice reconstruction must survive: {text}"
    );
}

#[test]
fn invalid_grid_constant_contracts_fail_before_emitting_an_ordinary_pointer_abi() {
    for (case, expected) in [
        ("nonkernel", "require a kernel entry"),
        ("noncanonical", "malformed or out-of-range"),
        ("range", "malformed or out-of-range"),
        ("incomplete", "incomplete grid-constant"),
        ("alignment", "power of two"),
        ("scalar", "not a pointer"),
        ("mutable", "immutable generic pointer"),
        ("space", "immutable generic pointer"),
        ("zero", "sized, nonzero pointee"),
        ("pointee", "pointee metadata differs"),
    ] {
        let error = fixture(case).unwrap_err();
        assert!(error.contains(expected), "{case}: {error}");
    }
}
