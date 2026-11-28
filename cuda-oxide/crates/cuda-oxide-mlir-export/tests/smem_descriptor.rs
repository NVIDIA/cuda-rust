//! The tcgen05 shared-memory descriptor is exported as one NVVM operation.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use cuda_oxide_mlir_export::{MlirAttribute, MlirBlock, MlirOperation, MlirType, MlirValueId};
use dialect_mir::ops::{MirFuncOp, MirReturnOp};
use dialect_nvvm::ops::{
    Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr,
    Tcgen05MmaOp, Tcgen05SmemDescriptorOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::Context,
    identifier::Identifier,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};
use std::collections::HashMap;

/// How the fixture deviates from the descriptor a kernel produces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fixture {
    /// The importer's shape: six typed fields, one `u64` result, no attributes.
    Imported,
    /// A leftover attribute on the descriptor operation.
    ResidualAttribute,
    /// A `u32` where the `u64` shared byte address belongs.
    NarrowAddress,
}

/// Export a function that packs a descriptor from its arguments and passes it
/// to a dense FP16 MMA as the A operand. Arguments are
/// `(address, leading_dim_bytes, stride_bytes, base_offset, leading_mode,
/// swizzle, tmem, b_desc, idesc, enable_d, narrow_address)`.
fn export(fixture: Fixture) -> Result<MlirBlock, String> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    // The importer types `bool` as a signless one-bit integer.
    let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let arguments = vec![
        u64_ty, u32_ty, u32_ty, u8_ty, bool_ty, u8_ty, u32_ty, u64_ty, u32_ty, bool_ty, u32_ty,
    ];
    let module = ModuleOp::new(&mut ctx, "descriptor".try_into().unwrap());
    let fty = FunctionType::get(&ctx, arguments.clone(), vec![]);
    let f = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(&mut ctx, f, TypeAttr::new(fty.into()));
    func.set_symbol_name(&mut ctx, "descriptor".try_into().unwrap());
    module.append_operation(&mut ctx, f, 0);
    let block = BasicBlock::new(&mut ctx, None, arguments);
    block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();

    let address = if fixture == Fixture::NarrowAddress {
        a[10]
    } else {
        a[0]
    };
    let descriptor = Operation::new(
        &mut ctx,
        Tcgen05SmemDescriptorOp::get_concrete_op_info(),
        vec![u64_ty],
        vec![address, a[1], a[2], a[3], a[4], a[5]],
        vec![],
        0,
    );
    if fixture == Fixture::ResidualAttribute {
        descriptor.deref_mut(&ctx).attributes.set(
            Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
            StringAttr::new("v1:i0001".into()),
        );
    }
    descriptor.insert_at_back(block, &ctx);
    let a_desc = descriptor.deref(&ctx).get_result(0);

    let mma = Operation::new(
        &mut ctx,
        Tcgen05MmaOp::get_concrete_op_info(),
        vec![],
        vec![a[6], a_desc, a[7], a[8], a[9]],
        vec![],
        0,
    );
    mma.deref_mut(&ctx).attributes.set(
        Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
        StringAttr::new("v1:i0763".into()),
    );
    let typed = Tcgen05MmaOp::new(mma);
    typed.set_attr_nvvm_tcgen05_mma_form(&ctx, Tcgen05MmaFormAttr::Shared);
    typed.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::F16);
    typed.set_attr_nvvm_tcgen05_mma_cta_group(&ctx, Tcgen05MmaCtaGroupAttr::Cg1);
    typed.set_attr_nvvm_tcgen05_mma_collector_a(&ctx, Tcgen05MmaCollectorAAttr::Discard);
    mma.insert_at_back(block, &ctx);

    Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    )
    .insert_at_back(block, &ctx);

    let target = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .map_err(|error| error.to_string())?;
    Ok(
        target.root.regions[0].blocks[0].operations[0].regions[0].blocks[0].operations[0].regions
            [0]
        .blocks[0]
            .clone(),
    )
}

/// Find the only exported operation with this name.
fn only<'a>(block: &'a MlirBlock, name: &str) -> &'a MlirOperation {
    let mut found = block.operations.iter().filter(|op| op.name == name);
    let op = found.next().unwrap_or_else(|| panic!("no {name}"));
    assert!(found.next().is_none(), "more than one {name}");
    op
}

/// Find the operation that defines an SSA value in the function block.
fn definition(block: &MlirBlock, id: MlirValueId) -> &MlirOperation {
    block
        .operations
        .iter()
        .find(|op| op.results.iter().any(|result| result.id == id))
        .expect("value defined in the function block")
}

/// Check a constant operand of the exported arithmetic.
fn assert_constant(block: &MlirBlock, id: MlirValueId, value: i128, width: u32) {
    let constant = definition(block, id);
    assert_eq!(constant.name, "arith.constant");
    assert_eq!(
        constant.properties["value"],
        MlirAttribute::Integer {
            value,
            ty: MlirType::Integer(width),
        }
    );
}

/// Check the descriptor is one NVVM operation whose result is the MMA's A operand.
#[test]
fn descriptor_becomes_the_nvvm_operation_and_feeds_the_mma() {
    let block = export(Fixture::Imported).unwrap();
    let arguments: Vec<_> = block.arguments.iter().map(|argument| argument.id).collect();
    let descriptor = only(&block, "nvvm.tcgen05_mma_smem_desc_v2");
    let mma = only(&block, "nvvm.tcgen05.mma");

    assert_eq!(
        descriptor
            .operands
            .iter()
            .map(|operand| operand.ty.clone())
            .collect::<Vec<_>>(),
        [32, 32, 32, 8, 8, 1, 1, 8].map(MlirType::Integer)
    );
    assert_eq!(descriptor.results.len(), 1);
    assert_eq!(descriptor.results[0].ty, MlirType::Integer(64));
    assert!(descriptor.properties.is_empty());

    // The MMA reads the descriptor directly, not arithmetic derived from it.
    assert_eq!(mma.operands[1].id, descriptor.results[0].id);
    assert_eq!(mma.operands[2].id, arguments[7]);

    // Start address: the byte address shifted to 16-byte units, masked to 14
    // bits and truncated to the operand type.
    let start = definition(&block, descriptor.operands[0].id);
    assert_eq!(start.name, "arith.trunci");
    let masked = definition(&block, start.operands[0].id);
    assert_eq!(masked.name, "arith.andi");
    assert_constant(&block, masked.operands[1].id, 0x3FFF, 64);
    let units = definition(&block, masked.operands[0].id);
    assert_eq!(units.name, "arith.shrui");
    assert_eq!(units.operands[0].id, arguments[0]);
    assert_constant(&block, units.operands[1].id, 4, 64);

    // Leading-dimension offset: 16-byte units, masked to 14 bits.
    let masked = definition(&block, descriptor.operands[1].id);
    assert_eq!(masked.name, "arith.andi");
    assert_constant(&block, masked.operands[1].id, 0x3FFF, 32);
    let units = definition(&block, masked.operands[0].id);
    assert_eq!(units.name, "arith.shrui");
    assert_eq!(units.operands[0].id, arguments[1]);
    assert_constant(&block, units.operands[1].id, 4, 32);

    // Stride: 16-byte units. The field has 14 bits on both sides.
    let units = definition(&block, descriptor.operands[2].id);
    assert_eq!(units.name, "arith.shrui");
    assert_eq!(units.operands[0].id, arguments[2]);
    assert_constant(&block, units.operands[1].id, 4, 32);

    // Base offset, leading-dimension mode and swizzle are forwarded.
    for (operand, argument) in [(4, 3), (5, 4), (7, 5)] {
        assert_eq!(descriptor.operands[operand].id, arguments[argument]);
    }
    // Descriptor version 1 and a zero K-segment bit are constants.
    assert_constant(&block, descriptor.operands[3].id, 1, 8);
    let k_segment = definition(&block, descriptor.operands[6].id);
    assert_eq!(k_segment.properties["value"], MlirAttribute::Bool(false));

    // No shift-left or or of the written-out encoding is left in the function.
    for op in &block.operations {
        assert!(
            !matches!(op.name.as_str(), "arith.shli" | "arith.ori"),
            "unexpected {}",
            op.name
        );
    }
}

/// The descriptor encoding written out, as the builder computed it before the
/// descriptor operation existed. Inputs are byte values.
fn reference_smem_descriptor(
    address: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    base_offset: u8,
    leading_dim_absolute: bool,
    swizzle: u8,
) -> u64 {
    let mut raw: u64 = 0;
    raw |= (address >> 4) & 0x3FFF;
    raw |= (((leading_dim_bytes >> 4) & 0x3FFF) as u64) << 16;
    raw |= (((stride_bytes >> 4) & 0x3FFF) as u64) << 32;
    raw |= 1u64 << 46;
    raw |= ((base_offset & 0x7) as u64) << 49;
    if leading_dim_absolute {
        raw |= 1u64 << 52;
    }
    raw |= (swizzle as u64) << 61;
    raw
}

/// Evaluate an exported value. `nvvm.tcgen05_mma_smem_desc_v2` packs the low
/// 15, 15, 14, 3, 3, 1, 1 and 3 bits of its operands at bits 0, 16, 32, 46, 49,
/// 52, 53 and 61.
fn evaluate(block: &MlirBlock, inputs: &HashMap<MlirValueId, u64>, id: MlirValueId) -> u64 {
    if let Some(value) = inputs.get(&id) {
        return *value;
    }
    let op = definition(block, id);
    let operand = |index: usize| evaluate(block, inputs, op.operands[index].id);
    match op.name.as_str() {
        "arith.constant" => match &op.properties["value"] {
            MlirAttribute::Integer { value, .. } => *value as u64,
            MlirAttribute::Bool(value) => *value as u64,
            _ => panic!("non-integer constant"),
        },
        "arith.andi" => operand(0) & operand(1),
        "arith.shrui" => operand(0) >> operand(1),
        "arith.trunci" => operand(0) & 0xFFFF_FFFF,
        "nvvm.tcgen05_mma_smem_desc_v2" => [
            (15, 0),
            (15, 16),
            (14, 32),
            (3, 46),
            (3, 49),
            (1, 52),
            (1, 53),
            (3, 61),
        ]
        .into_iter()
        .enumerate()
        .fold(0, |packed, (index, (width, shift))| {
            packed | ((operand(index) & ((1u64 << width) - 1)) << shift)
        }),
        other => panic!("unexpected descriptor arithmetic: {other}"),
    }
}

/// Check the exported operation packs the bits of the written-out encoding,
/// also for values wider than a field.
#[test]
fn exported_descriptor_has_the_bits_of_the_written_out_encoding() {
    let block = export(Fixture::Imported).unwrap();
    let arguments: Vec<_> = block.arguments.iter().map(|argument| argument.id).collect();
    let descriptor = only(&block, "nvvm.tcgen05_mma_smem_desc_v2").results[0].id;
    let addresses = [0u64, 0x1000, 0x3_FFF0, 0x4_0000, 0xFFFF_FFFF_FFFF_FFFF];
    let offsets = [0u32, 16, 2048, 0x3_FFF0, 0x4_0000, 0xFFFF_FFFF];
    for address in addresses {
        for leading in offsets {
            for stride in offsets {
                for (base_offset, absolute, swizzle) in [
                    (0u8, false, 0u8),
                    (5, true, 2),
                    (0xFF, false, 6),
                    (7, true, 4),
                ] {
                    let fields = [
                        address,
                        leading as u64,
                        stride as u64,
                        base_offset as u64,
                        absolute as u64,
                        swizzle as u64,
                    ];
                    let inputs = arguments.iter().copied().zip(fields).collect();
                    assert_eq!(
                        evaluate(&block, &inputs, descriptor),
                        reference_smem_descriptor(
                            address,
                            leading,
                            stride,
                            base_offset,
                            absolute,
                            swizzle
                        ),
                        "address={address:#x} leading={leading:#x} stride={stride:#x}"
                    );
                }
            }
        }
    }
}

/// Check the rule accepts only the importer's shape of the source operation.
#[test]
fn descriptor_rule_rejects_other_shapes_of_the_source_operation() {
    let error = export(Fixture::ResidualAttribute).err().unwrap();
    assert!(
        error.contains("six operands, one result, and no attributes"),
        "{error}"
    );
    let error = export(Fixture::NarrowAddress).err().unwrap();
    assert!(
        error.contains("operands must be u64, u32, u32, u8, bool, u8"),
        "{error}"
    );
}
