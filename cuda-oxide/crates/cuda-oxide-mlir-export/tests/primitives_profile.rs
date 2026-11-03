/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Primitives-only profile isolation, source verification, and reproducibility metadata.

use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use pliron::builtin::op_interfaces::SingleBlockRegionInterface;
use pliron::{
    builtin::ops::ModuleOp, context::Context, identifier::Identifier, op::Op, printable::Printable,
};

/// Reject CuTe pointees even when the runtime carrier would become an opaque pointer.
#[test]
fn primitives_rejects_cute_hidden_in_pointer_type_metadata() {
    use dialect_cute::{attributes::CuteTileGridAttr, types::CuteWorkTileType};
    use dialect_mir::types::{MirPtrType, MirTupleType};
    use pliron::builtin::attributes::{TypeAttr, VecAttr};
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "nested".try_into().unwrap());
    let cute = CuteWorkTileType::get(&mut ctx, CuteTileGridAttr::new(1, 1, 1));
    let tuple = MirTupleType::get(&mut ctx, vec![cute.into()]);
    let pointer = MirPtrType::get_generic(&mut ctx, tuple.into(), false);
    module.get_operation().deref_mut(&ctx).attributes.set(
        "metadata".try_into().unwrap(),
        VecAttr(vec![Box::new(TypeAttr::new(pointer.into()))]),
    );
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .verify_source_module(&ctx, &module)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cute.work_tile"), "{error}");
}

/// Check ordinary MIR exports with a primitives-only registry.
#[test]
fn primitives_profile_translates_without_cute_registration() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, Identifier::try_from("primitives").unwrap());
    let before = module.disp(&ctx).to_string();
    let profile = CutlassPrimitivesMlir22::new("sm_100a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    assert_eq!(
        target.root.regions[0].blocks[0].operations[0].name,
        "gpu.module"
    );
    assert_eq!(module.disp(&ctx).to_string(), before);
    assert!(profile.build_registry().unwrap().is_sealed());
}

/// Check profile identity, compiler revision, and architecture are recorded in the manifest.
#[test]
fn primitives_profile_pins_consumer_and_target() {
    let profile = CutlassPrimitivesMlir22::new("sm_100a").unwrap();
    let manifest = profile.manifest();
    assert_eq!(manifest.profile, "cutlass-primitives-mlir22");
    assert_eq!(manifest.consumer_revision, "cutlass-compiler-v4.7.0");
    assert_eq!(manifest.llvm_major, 22);
    assert_eq!(
        manifest.pliron_revision,
        "435536c8611af34c48c8f73d78e77894a05d103d"
    );
    assert_eq!(
        manifest.interop_revision,
        cuda_oxide_mlir_export::INTEROP_REVISION
    );
    for arch in ["sm_103a", "sm_120a"] {
        assert_eq!(
            CutlassPrimitivesMlir22::new(arch)
                .unwrap()
                .manifest()
                .gpu_arch,
            arch
        );
    }
    for arch in ["sm_90a", "sm_120", "sm_100", "sm_103", "garbage"] {
        assert!(CutlassPrimitivesMlir22::new(arch).is_err());
    }
}

/// Reject a CuTe operation before its own verifier can obscure the profile violation.
#[test]
fn primitives_rejects_cute_operations_before_local_verification() {
    let mut ctx = Context::new();
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "cute_op".try_into().unwrap());
    let op = pliron::operation::Operation::new(
        &mut ctx,
        dialect_cute::ops::CuteCopyG2SOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    module.append_operation(&mut ctx, op, 0);
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .verify_source_module(&ctx, &module)
        .unwrap_err()
        .to_string();
    assert!(error.contains("CuTe operation"), "{error}");
}

/// Find forbidden CuTe metadata recursively inside nested attribute dictionaries.
#[test]
fn primitives_rejects_cute_attributes_in_dictionaries() {
    use pliron::{attribute::AttributeDict, builtin::attributes::DictAttr};
    let mut ctx = Context::new();
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "cute_attr".try_into().unwrap());
    let mut dictionary = AttributeDict::default();
    dictionary.set(
        "grid".try_into().unwrap(),
        dialect_cute::attributes::CuteTileGridAttr::new(1, 1, 1),
    );
    module
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set("metadata".try_into().unwrap(), DictAttr(dictionary));
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .verify_source_module(&ctx, &module)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cute.tile_grid"), "{error}");
}

/// Reject unreviewed NVVM instructions instead of translating them by name alone.
#[test]
fn primitives_rejects_unmapped_nvvm_instructions() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "unmapped".try_into().unwrap());
    let op = pliron::operation::Operation::new(
        &mut ctx,
        dialect_nvvm::ops::Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    module.append_operation(&mut ctx, op, 0);
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .unwrap_err()
        .to_string();
    assert!(error.contains("nvvm.barrier0"), "{error}");
}

/// Check the primitives profile also enforces ordinary source operation validity.
#[test]
fn primitives_still_checks_ordinary_ir_verification() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "invalid".try_into().unwrap());
    // A special-register read must produce one result.
    let op = pliron::operation::Operation::new(
        &mut ctx,
        dialect_nvvm::ops::ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    module.append_operation(&mut ctx, op, 0);
    assert!(
        CutlassPrimitivesMlir22::new("sm_100a")
            .unwrap()
            .verify_source_module(&ctx, &module)
            .is_err()
    );
}
