/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Execute the actual exported descriptor arithmetic on the CPU. Native CuTe
//! descriptor construction is an input to this probe; its layout is unchanged.
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use cuda_oxide_mlir_export::{MlirAttribute, MlirBlock, MlirOperation, MlirType, MlirValueId};
use dialect_cute::sm100_attention_ops::*;
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirFP16Type, MirPtrType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};
use std::collections::HashMap;

/// Exported descriptor-building SSA graph evaluated independently of GPU execution.
struct Probe {
    /// Target operation defining each SSA result inspected by the probe.
    definitions: HashMap<MlirValueId, MlirOperation>,
    /// SSA argument representing the dynamic shared-memory byte offset.
    offset: MlirValueId,
    /// Descriptor result IDs whose address fields are checked.
    descriptors: Vec<MlirValueId>,
}

/// Find the translated block containing the TCGen05 MMA instruction.
fn function_block(op: &MlirOperation) -> Option<&MlirBlock> {
    for region in &op.regions {
        for block in &region.blocks {
            if block
                .operations
                .iter()
                .any(|op| op.name == "nvvm.tcgen05.mma")
            {
                return Some(block);
            }
            for child in &block.operations {
                if let Some(found) = function_block(child) {
                    return Some(found);
                }
            }
        }
    }
    None
}

impl Probe {
    /// Export an attention MMA plan for the requested shape and shared-memory major mode.
    fn new(rows: u32, k: u32, mn: bool) -> Self {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "descriptor_probe".try_into().unwrap());
        let u32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let b: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
        let f16 = MirFP16Type::get(&ctx).into();
        let shared: TypeHandle = MirPtrType::get(&mut ctx, f16, true, 3).into();
        let args = vec![u32, shared, shared, b, u32, u32];
        let ty = FunctionType::get(&ctx, args.clone(), vec![]);
        let f = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, f, TypeAttr::new(ty.into()))
            .set_symbol_name(&mut ctx, "descriptor_probe".try_into().unwrap());
        module.append_operation(&mut ctx, f, 0);
        let block = BasicBlock::new(&mut ctx, None, args);
        block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
        let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
        let group = if rows <= 128 { 2 } else { 1 };
        CuteSm100MmaSsOp::new(
            &mut ctx,
            a,
            CuteSm100MmaAttr {
                m: 128,
                n: rows * group,
                k,
                cta_group: group,
                a_mn: false,
                b_mn: mn,
                a_layout: shared_layout(128 / group, k, false).unwrap(),
                b_layout: shared_layout(rows, k, mn).unwrap(),
            },
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(block, &ctx);
        let target = CutlassFullCuteMlir22::new("sm_100a")
            .unwrap()
            .translate_module(&ctx, &module)
            .unwrap();
        let block = function_block(&target.root).unwrap();
        Self {
            offset: block.arguments[5].id,
            descriptors: block
                .operations
                .iter()
                .filter(|op| op.name == "nvvm.tcgen05.mma")
                .map(|op| op.operands[2].id)
                .collect(),
            definitions: block
                .operations
                .iter()
                .filter(|op| !op.results.is_empty())
                .map(|op| (op.results[0].id, op.clone()))
                .collect(),
        }
    }

    /// Interpret descriptor arithmetic using a chosen base descriptor and dynamic byte offset.
    fn evaluate(&self, id: MlirValueId, base: u64, byte_offset: u32) -> u64 {
        if id == self.offset {
            return byte_offset as u64;
        }
        let op = &self.definitions[&id];
        let operand = |i: usize| self.evaluate(op.operands[i].id, base, byte_offset);
        let value = match op.name.as_str() {
            "cute_nvgpu.make_umma_smem_desc" => base,
            "builtin.unrealized_conversion_cast" | "arith.extui" | "arith.trunci" => operand(0),
            "arith.constant" => match op.properties["value"] {
                MlirAttribute::Integer { value, .. } => value as u64,
                _ => panic!("non-integer constant"),
            },
            "arith.addi" => operand(0).wrapping_add(operand(1)),
            "arith.andi" => operand(0) & operand(1),
            "arith.ori" => operand(0) | operand(1),
            "arith.shrui" => operand(0) >> operand(1),
            other => panic!("unexpected descriptor arithmetic: {other}"),
        };
        match op.results[0].ty {
            MlirType::Integer(32) => value as u32 as u64,
            _ => value,
        }
    }
}

/// Check address updates leave every descriptor metadata field unchanged across boundary cases.
#[test]
fn descriptor_update_preserves_metadata_across_the_entire_address_field() {
    let probe = Probe::new(64, 64, false);
    // Both supported swizzles, stride extremes, and non-address bit patterns.
    // The last pattern deliberately fills reserved bits to catch accidental
    // masking of metadata independently of native descriptor construction.
    for metadata in [0x4000_4040_0001_0000, 0x8000_7fff_3fff_0000, !0x3fffu64] {
        for address in 0..=0x3fffu64 {
            let remaining = 0x3fff - address;
            for delta in [0, remaining / 2, remaining] {
                let base = metadata | address;
                let got = probe.evaluate(probe.descriptors[0], base, (delta * 16) as u32);
                assert_eq!(got, base + delta, "address={address:#x}, delta={delta:#x}");
                assert_eq!(got & !0x3fff, metadata);
            }
        }
    }
}

/// Check K-major and MN-major descriptor updates agree with independent byte-address arithmetic.
#[test]
fn tile_and_k_offsets_match_addition_for_both_layout_families() {
    for rows in (8..=256).step_by(8) {
        if rows > 128 && rows % 16 != 0 {
            continue;
        }
        for k in [32, 64, 128, 256] {
            for mn in [false, true] {
                if shared_layout(rows, k, mn).is_err() {
                    continue;
                }
                let probe = Probe::new(rows, k, mn);
                let tile_bytes = rows * k * 2;
                let span = if mn { rows } else { k }.min(64);
                let period = if span == 32 { 512 } else { 1024 };
                let last_base = (0x40000 - tile_bytes) / period * period;
                for base_bytes in [0, period, last_base] {
                    let last_tile = (0x40000 - base_bytes) / tile_bytes - 1;
                    for tile in [0, last_tile / 2, last_tile] {
                        let byte_offset = tile * tile_bytes;
                        let base = 0x4000_4040_0001_0000 | (base_bytes / 16) as u64;
                        for (i, descriptor) in probe.descriptors.iter().enumerate() {
                            // Physical matrix strides, expressed directly in
                            // bytes rather than the exporter's address units.
                            let col = i as u32 * 16;
                            let within_tile = if mn {
                                col * span * 2
                            } else {
                                (col / span * rows * span + col % span) * 2
                            };
                            let expected = base + (byte_offset + within_tile) as u64 / 16;
                            assert_eq!(
                                probe.evaluate(*descriptor, base, byte_offset),
                                expected,
                                "rows={rows}, k={k}, mn={mn}, base={base_bytes}, tile={tile}, col={col}"
                            );
                        }
                    }
                }
            }
        }
    }
}
