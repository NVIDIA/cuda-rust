//! Named shared offsets versus Rust generic pointer-address semantics.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use cuda_oxide_mlir_export::{MlirOperation, MlirType};
use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{MirCastOp, MirFuncOp, MirReturnOp},
    types::MirPtrType,
};
use dialect_nvvm::ops::{CvtaGenericToClusterSharedOffsetOp, CvtaGenericToSharedOffsetOp};
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

/// Build pointer-address operations for a selected source space and export through the primitives profile.
fn translate(
    space: u32,
    width: u32,
    kind: Option<MirCastKindAttr>,
    cluster: bool,
) -> Vec<MlirOperation> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let word: TypeHandle = IntegerType::get(&ctx, width, Signedness::Unsigned).into();
    let ptr: TypeHandle = MirPtrType::get(&mut ctx, word, true, space).into();
    let module = ModuleOp::new(&mut ctx, "addresses".try_into().unwrap());
    let fty = FunctionType::get(&ctx, vec![ptr], vec![word]);
    let f = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(&mut ctx, f, TypeAttr::new(fty.into()));
    func.set_symbol_name(&mut ctx, "address".try_into().unwrap());
    module.append_operation(&mut ctx, f, 0);
    let block = BasicBlock::new(&mut ctx, None, vec![ptr]);
    block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
    let arg = block.deref(&ctx).get_argument(0);
    let info = if kind.is_some() {
        MirCastOp::get_concrete_op_info()
    } else if cluster {
        CvtaGenericToClusterSharedOffsetOp::get_concrete_op_info()
    } else {
        CvtaGenericToSharedOffsetOp::get_concrete_op_info()
    };
    let cast = Operation::new(&mut ctx, info, vec![word], vec![arg], vec![], 0);
    if let Some(kind) = kind {
        MirCastOp::new(cast).set_attr_cast_kind(&ctx, kind);
    }
    cast.insert_at_back(block, &ctx);
    let value = cast.deref(&ctx).get_result(0);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![value],
        vec![],
        0,
    );
    ret.insert_at_back(block, &ctx);
    let target = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .unwrap();
    target.root.regions[0].blocks[0].operations[0].regions[0].blocks[0].operations[0].regions[0]
        .blocks[0]
        .operations
        .clone()
}

/// Check explicit shared offsets retain named spaces while Rust address casts use generic addresses.
#[test]
fn explicit_shared_offsets_preserve_cluster_rank_while_rust_addresses_genericize() {
    for space in [0, 3, 7] {
        for width in [32, 64] {
            for kind in [
                None,
                Some(MirCastKindAttr::PointerExposeAddress),
                Some(MirCastKindAttr::Transmute),
            ] {
                if kind == Some(MirCastKindAttr::Transmute) && width != 64 {
                    continue;
                }
                let expose = kind.is_some();
                let ops = translate(space, width, kind, false);
                let casts: Vec<_> = ops
                    .iter()
                    .filter(|o| o.name == "llvm.addrspacecast")
                    .collect();
                let ptrtoint = ops.iter().find(|o| o.name == "llvm.ptrtoint").unwrap();
                let needs_cast = if expose { space != 0 } else { space == 0 };
                assert_eq!(casts.len(), usize::from(needs_cast));
                let result_space = if expose {
                    0
                } else if space == 0 {
                    3
                } else {
                    space
                };
                let spelling = if result_space == 0 {
                    "!llvm.ptr".into()
                } else {
                    format!("!llvm.ptr<{result_space}>")
                };
                assert_eq!(
                    ptrtoint.operands[0].ty,
                    MlirType::dialect(spelling).unwrap()
                );
                assert_eq!(ptrtoint.results[0].ty, MlirType::Integer(width));
                if needs_cast {
                    assert_eq!(ptrtoint.operands[0].id, casts[0].results[0].id);
                }
            }
        }
    }
}

/// Check generic pointers enter cluster shared space before their integer offset is formed.
#[test]
fn explicit_cluster_offsets_target_as7_even_from_generic_pointers() {
    for space in [0, 3, 7] {
        for width in [32, 64] {
            let ops = translate(space, width, None, true);
            let casts: Vec<_> = ops
                .iter()
                .filter(|o| o.name == "llvm.addrspacecast")
                .collect();
            let int = ops.iter().find(|o| o.name == "llvm.ptrtoint").unwrap();
            assert_eq!(casts.len(), usize::from(space != 7));
            assert_eq!(
                int.operands[0].ty,
                MlirType::dialect("!llvm.ptr<7>").unwrap()
            );
            if space != 7 {
                assert_eq!(int.operands[0].id, casts[0].results[0].id);
            }
        }
    }
}
