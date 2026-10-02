/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

use crate::model::{
    RegisterMmaAdapter, RegisterMmaCompatibilitySource, RegisterMmaKind, RuntimeValidation,
    SparseMmaAccumulator, SparseMmaAdapter, SparseMmaShape,
};
use crate::render::collector_targets::generated_intrinsic_variant;
use crate::render::common::intrinsic_marker;
use crate::render::families::{
    BLACKWELL_LDMATRIX_EFFECTIVE_FLOORS, SPARSE_MMA_ORDERED_METADATA_RULE,
    SPARSE_MMA_ORDERED_TF32_METADATA_RULE, SPARSE_MMA_STANDARD_METADATA_RULE, ldmatrix,
    register_mma_attr_variants, register_mma_constraints, register_mma_template, register_mmas,
    sparse_mma_carriers, sparse_mma_constraints, sparse_mma_fragment_counts,
    sparse_mma_metadata_rule, sparse_mma_selector_description, sparse_mma_selector_values,
    sparse_mma_template, sparse_mmas,
};
use crate::util::read_json;
use std::path::Path;

#[test]
fn stmatrix_rendering_preserves_all_four_public_and_backend_contracts() {
    let catalog = catalog_with_stmatrix();
    validate_renderable(&catalog).unwrap();
    assert_eq!(stmatrices(&catalog).count(), 4);

    let compatibility = render_compat_stmatrix(&catalog, "test-hash");
    for signature in [
        "pub unsafe fn stmatrix_m8n8_x2(smem_ptr: *mut u8, r0: u32, r1: u32)",
        "pub unsafe fn stmatrix_m8n8_x2_trans(smem_ptr: *mut u8, r0: u32, r1: u32)",
        "pub unsafe fn stmatrix_m8n8_x4(smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32)",
        "pub unsafe fn stmatrix_m8n8_x4_trans(smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32)",
    ] {
        assert!(compatibility.contains(signature));
    }
    assert!(compatibility.contains("All warp lanes must participate"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod stmatrix;"));
    assert!(dialect_mod.contains("stmatrix::register(ctx)"));
    let dialect = render_dialect_stmatrix(&catalog, "test-hash");
    for (op_type, op_name, operands) in [
        ("StmatrixM8n8X2Op", "nvvm.stmatrix_m8n8_x2", 3),
        ("StmatrixM8n8X2TransOp", "nvvm.stmatrix_m8n8_x2_trans", 3),
        ("StmatrixM8n8X4Op", "nvvm.stmatrix_m8n8_x4", 5),
        ("StmatrixM8n8X4TransOp", "nvvm.stmatrix_m8n8_x4_trans", 5),
    ] {
        assert!(dialect.contains(&format!("pub struct {op_type};")));
        assert!(dialect.contains(&format!("name = \"{op_name}\"")));
        assert!(dialect.contains(&format!("NOpdsInterface<{operands}>")));
        assert!(dialect.contains(&format!("{op_type}::register(ctx)")));
    }

    let importer = render_importer(&catalog, "test-hash");
    for (path, op_type, marker) in [
        (
            "cuda_device::tcgen05::stmatrix_m8n8_x2",
            "StmatrixM8n8X2Op",
            "v1:i0301",
        ),
        (
            "cuda_device::tcgen05::stmatrix_m8n8_x2_trans",
            "StmatrixM8n8X2TransOp",
            "v1:i0302",
        ),
        (
            "cuda_device::tcgen05::stmatrix_m8n8_x4",
            "StmatrixM8n8X4Op",
            "v1:i0303",
        ),
        (
            "cuda_device::tcgen05::stmatrix_m8n8_x4_trans",
            "StmatrixM8n8X4TransOp",
            "v1:i0304",
        ),
    ] {
        assert!(importer.contains(path));
        assert!(importer.contains(&format!("{op_type}::get_concrete_op_info()")));
        assert!(importer.contains(&format!(
            "set_generated_intrinsic_marker(ctx, store, \"{marker}\")"
        )));
    }

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for StmatrixM8n8")
            .count(),
        4
    );
    assert!(lowering.contains("IntrinsicBackend::LlvmNvptx"));
    assert!(lowering.contains("IntrinsicBackend::LibNvvm"));
    assert!(lowering.contains("cast_to_shared_addrspace"));
    assert!(lowering.contains("llvm_nvvm_stmatrix_sync_aligned_m8n8_x2_b16_p3"));
    assert!(lowering.contains("llvm_nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16_p3"));
    assert!(lowering.contains("stmatrix.sync.aligned.m8n8.x{register_count}{trans}.shared.b16"));
    assert!(lowering.contains("std::iter::once(\"~{memory}\")"));

    let x2 = stmatrices(&catalog)
        .find(|record| record.id == "stmatrix_m8n8_x2_b16")
        .unwrap();
    let probe = render_probe(&catalog, x2, "test-hash");
    assert!(probe.contains(
        "declare void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3), i32, i32)"
    ));
    assert!(probe.contains("%shared = addrspacecast ptr %generic to ptr addrspace(3)"));
    assert!(probe.contains(
            "call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %shared, i32 %r0, i32 %r1)"
        ));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/stmatrix.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/stmatrix.rs"
    )));
}

#[test]
fn ldmatrix_family_lowering_uses_one_attribute_dispatch_impl() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();

    let dialect = render_dialect_ldmatrix(&catalog, "test-hash");
    let rendered = render_lowering(&catalog, "test-hash");
    assert_eq!(
        rendered
            .matches("impl MirToLlvmConversion for LdmatrixOp")
            .count(),
        1
    );
    assert!(rendered.contains("LdmatrixMultiplicityAttr::X4"));
    assert!(rendered.contains("LdmatrixMultiplicityAttr::X2"));
    assert!(rendered.contains("LdmatrixMultiplicityAttr::X1"));
    assert!(rendered.contains("LdmatrixLayoutAttr::Transposed"));
    assert!(rendered.contains("llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3"));
    for (op_type, op_name, count, instruction_head, intrinsic) in [
        (
            "LdmatrixX1Op",
            "nvvm.ldmatrix_x1",
            1,
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_b16_p3",
        ),
        (
            "LdmatrixX1TransOp",
            "nvvm.ldmatrix_x1_trans",
            1,
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16_p3",
        ),
        (
            "LdmatrixX2Op",
            "nvvm.ldmatrix_x2",
            2,
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_b16_p3",
        ),
        (
            "LdmatrixX2TransOp",
            "nvvm.ldmatrix_x2_trans",
            2,
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3",
        ),
        (
            "LdmatrixX4Op",
            "nvvm.ldmatrix_x4",
            4,
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16_p3",
        ),
        (
            "LdmatrixX4TransOp",
            "nvvm.ldmatrix_x4_trans",
            4,
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
            "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16_p3",
        ),
    ] {
        assert!(dialect.contains(&format!("pub struct {op_type};")));
        assert!(dialect.contains(&format!("name = {op_name:?}")));
        assert!(dialect.contains(&format!("NResultsInterface<{count}>")));
        assert!(dialect.contains(&format!("{op_type}::register(ctx);")));
        assert_eq!(
            rendered
                .matches(&format!("impl MirToLlvmConversion for {op_type}"))
                .count(),
            1
        );
        assert!(rendered.contains(&format!(
            "self.get_operation(), {count}, {instruction_head:?}, {intrinsic:?}"
        )));
    }

    let importer = render_importer(&catalog, "test-hash");
    assert!(!importer.contains("LdmatrixX1Op"));
    assert!(!importer.contains("LdmatrixX4TransOp"));
    let targets = render_targets(&catalog, "test-hash");
    for op_name in [
        "nvvm.ldmatrix_x1",
        "nvvm.ldmatrix_x1_trans",
        "nvvm.ldmatrix_x2",
        "nvvm.ldmatrix_x2_trans",
        "nvvm.ldmatrix_x4",
        "nvvm.ldmatrix_x4_trans",
    ] {
        assert!(targets.contains(&format!("{op_name:?} => Some(")));
    }
    assert_eq!(ldmatrix(&catalog).count(), 18);
    assert!(dialect.contains("LdmatrixShapeAttr::M16n16"));
    assert!(dialect.contains("LdmatrixShapeAttr::M8n16"));
    assert!(dialect.contains("LdmatrixElementAttr::B8x16B4x16P64"));
    assert!(
        rendered.contains(
            "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3"
        )
    );
    assert!(rendered.contains("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains(
        "Instruction floor PTX 8.6; the selected target may require a newer PTX version."
    ));
    assert!(raw.contains("no lane may have exited"));
    assert!(raw.contains("have 32 readable shared-memory bytes"));
    assert!(raw.contains("have 16 readable shared-memory bytes"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains(BLACKWELL_LDMATRIX_EFFECTIVE_FLOORS));
}

#[test]
fn ldmatrix_x1_probe_keeps_its_pointer_operand() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let record = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == "ldmatrix_m8n8_x1_b16")
        .unwrap();

    let rendered = render_probe(&catalog, record, "test-hash");
    assert!(
        rendered.contains(
            "declare i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3))"
        )
    );
    assert!(rendered.contains("define i32 @probe_ldmatrix_m8n8_x1_b16(ptr %generic)"));
    assert!(!rendered.contains("@llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3()"));
}

#[test]
fn movmatrix_rendering_owns_dialect_import_and_lowering() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut catalog = crate::resolve::resolve(&repo_root).unwrap();
    let mut record = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == "packed_atomic_add_f16x2")
        .unwrap()
        .clone();
    record.id = "movmatrix_trans_b16".into();
    record.operation_key = "movmatrix.m8n8.trans.b16".into();
    record.family = "movmatrix".into();
    record.rust.abi_id = "i0305".into();
    record.rust.module = "matrix".into();
    record.rust.name = "movmatrix_trans_b16".into();
    record.rust.arguments = vec!["u32".into()];
    record.rust.result = "u32".into();
    record.rust.safe = false;
    record.rust.must_use = true;
    record.rust.canonical_path = "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0305".into();
    record.rust.public_path = "cuda_intrinsics::matrix::movmatrix_trans_b16".into();
    record.rust.compatibility_paths = vec!["cuda_device::wmma::movmatrix_trans_b16".into()];
    record.dialect.op_type = "MovmatrixTransB16Op".into();
    record.dialect.op_name = "nvvm.movmatrix_trans_b16".into();
    record.dialect.operands = vec!["i32".into()];
    record.dialect.results = vec!["i32".into()];
    record.semantics.pure = false;
    record.semantics.memory = "inaccessible_read_write".into();
    record.semantics.convergent = true;
    record.semantics.execution_scope = "warp".into();
    record.packed_atomic = None;
    record.movmatrix = Some(crate::model::Movmatrix {
        participation:
            crate::model::MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes,
        adapter: crate::model::MovmatrixAdapter::PackedB16x2U32ToPackedB16x2U32,
        runtime_validation: RuntimeValidation::Unexecuted,
    });
    record.lowering = "generated_movmatrix_inline_ptx".into();
    record.expected_ptx = crate::ptx::InstructionPattern::new(
        "movmatrix",
        &["sync", "aligned", "m8n8", "trans", "b16"],
        vec![
            crate::ptx::OperandPattern::Register,
            crate::ptx::OperandPattern::Register,
        ],
    );
    record.summary = "Transposes one packed b16 matrix fragment across a warp.".into();
    catalog.intrinsics.push(record);

    validate_renderable(&catalog).unwrap();
    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod movmatrix;"));
    assert!(dialect_mod.contains("movmatrix::register(ctx);"));

    let dialect = render_dialect_movmatrix(&catalog, "test-hash");
    assert!(dialect.contains("pub struct MovmatrixTransB16Op;"));
    assert!(dialect.contains("name = \"nvvm.movmatrix_trans_b16\""));
    assert!(dialect.contains("MovmatrixTransB16Op::register(ctx);"));

    let compatibility = render_compat_movmatrix(&catalog, "test-hash");
    assert!(compatibility.contains("pub unsafe fn movmatrix_trans_b16(value: u32) -> u32"));
    assert!(compatibility.contains("All 32 warp lanes must execute the same call"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::wmma::movmatrix_trans_b16"));
    assert!(importer.contains("MovmatrixTransB16Op::build(ctx, arg0)"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for MovmatrixTransB16Op"));
    assert!(lowering.contains("movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"));
    assert!(lowering.contains("\"=r,r\""));
    assert!(lowering.contains("inline_asm_convergent"));

    let probe_record = movmatrix(&catalog).next().unwrap();
    let probe = render_probe(&catalog, probe_record, "test-hash");
    assert!(probe.contains("call i32 asm \"movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;\""));
    assert!(probe.contains("attributes #0 = { convergent }"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/movmatrix.rs"
    )));
}

#[test]
fn register_mma_rendering_preserves_apis_order_convergence_and_variants() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(catalog.intrinsics.len(), 1029);
    let records: Vec<_> = register_mmas(&catalog).collect();
    assert_eq!(records.len(), 154);
    let generated_records = records
        .iter()
        .copied()
        .filter(|record| {
            record.register_mma.as_ref().unwrap().compatibility_source
                == RegisterMmaCompatibilitySource::GeneratedStub
        })
        .collect::<Vec<_>>();
    let existing_records = records
        .iter()
        .copied()
        .filter(|record| {
            record.register_mma.as_ref().unwrap().compatibility_source
                == RegisterMmaCompatibilitySource::ExistingStub
        })
        .collect::<Vec<_>>();
    assert_eq!(generated_records.len(), 149);
    assert_eq!(existing_records.len(), 5);

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    for record in &records {
        assert!(raw.contains(&format!("pub unsafe fn {}(", record.rust.abi_id)));
    }
    assert!(raw.contains("no lane may have exited"));
    assert!(raw.contains("Signed accumulator overflow wraps"));
    assert!(raw.contains("Signed accumulator overflow clamps"));

    let compatibility = render_compat_register_mma(&catalog, "test-hash");
    assert_eq!(compatibility.matches("pub unsafe fn ").count(), 149);
    for record in generated_records {
        let argument_names: &[&str] = if record.register_mma.as_ref().unwrap().adapter
            == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
        {
            &[
                "c",
                "a",
                "b",
                "scale_a",
                "byte_id_a",
                "thread_id_a",
                "scale_b",
                "byte_id_b",
                "thread_id_b",
            ]
        } else {
            &["c", "a", "b"]
        };
        let arguments = argument_names
            .iter()
            .zip(&record.rust.arguments)
            .map(|(name, ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        assert!(compatibility.contains(&format!(
            "pub unsafe fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )));
    }
    for record in existing_records {
        assert!(!compatibility.contains(&format!("pub unsafe fn {}(", record.rust.name)));
    }

    let dialect = render_dialect_register_mma(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct RegisterMmaOp").count(), 1);
    assert!(dialect.contains("let recipe: (&[MmaCarrier], &[MmaCarrier])"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M8n8k4"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M8n8k16"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M8n8k32"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M8n8k128"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M16n8k4"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M16n8k32"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M16n8k64"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M16n8k128"));
    assert!(dialect.contains("RegisterMmaShapeAttr::M16n8k256"));
    assert!(dialect.contains("RegisterMmaOperationAttr::Multiply"));
    assert!(dialect.contains("RegisterMmaOperationAttr::AndPopc"));
    assert!(dialect.contains("RegisterMmaOperationAttr::XorPopc"));
    assert!(dialect.contains("pub enum RegisterMmaKindAttr { Standard, F8f6f4, Mxf8f6f4 }"));
    assert!(dialect.contains("kind_or_inferred"));
    assert!(dialect.contains("RegisterMmaAccumulatorAttr::F16"));
    assert!(dialect.contains("operation_or_multiply"));
    assert!(dialect.contains("MmaCarrier::I32 | MmaCarrier::U16 | MmaCarrier::U32"));
    assert!(dialect.contains("if matches!(carrier, MmaCarrier::U16)"));
    assert!(dialect.contains("RegisterMmaElementAttr::B1"));
    assert!(dialect.contains("RegisterMmaElementAttr::E2m1"));
    assert!(dialect.contains("RegisterMmaElementAttr::E5m2"));
    assert!(dialect.contains("RegisterMmaElementAttr::S4"));
    assert!(dialect.contains("RegisterMmaElementAttr::U4"));
    assert!(dialect.contains("RegisterMmaElementAttr::U8"));
    assert!(dialect.contains("RegisterMmaOverflowAttr::Wrapping"));
    assert!(dialect.contains("RegisterMmaOverflowAttr::Satfinite"));
    assert!(dialect.contains(
            "&[MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]"
        ));
    for (op_type, op_name) in [
        ("MmaM16N8K16F32Bf16Op", "nvvm.mma_m16n8k16_f32_bf16"),
        ("MmaM16N8K16F32F16Op", "nvvm.mma_m16n8k16_f32_f16"),
        ("MmaM16N8K8F32Tf32Op", "nvvm.mma_m16n8k8_f32_tf32"),
        ("MmaM16N8K32S32S8Op", "nvvm.mma_m16n8k32_s32_s8"),
        ("MmaM8N8K4F64Op", "nvvm.mma_m8n8k4_f64"),
    ] {
        assert!(dialect.contains(&format!("pub struct {op_type};")));
        assert!(dialect.contains(&format!("name = {op_name:?}")));
        assert!(dialect.contains(&format!("{op_type}::register(ctx);")));
    }

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("enum GeneratedMmaImportAdapter"));
    assert!(importer.contains("C2U32A4U32B2U32ToD2U32"));
    assert!(importer.contains("C2U32A2U32B1U32ToD2U32"));
    assert!(importer.contains("C4F32A2U32B1U32ToD4F32"));
    assert!(importer.contains("C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32"));
    assert!(importer.contains("ctx, b_value, b_ty, b_count, block_ptr, last_after_b, loc.clone()"));
    assert!(importer.contains("C4I32A2U32B1U32ToD4I32"));
    assert!(importer.contains("C2I32A1U32B1U32ToD2I32"));
    assert!(importer.contains("(i32_ty, 2, u32_ty, 1, false, u32_ty, 1, false, i32_ty, 2)"));
    assert!(importer.contains("(i32_ty, 4, u32_ty, 2, true, u32_ty, 1, false, i32_ty, 4)"));
    assert!(importer.contains("(u32_ty, 2, u32_ty, 4, true, u32_ty, 2, true, u32_ty, 2)"));
    assert!(importer.contains("import_generated_mma_operands"));
    assert!(importer.contains("bundle_generated_mma_results"));
    assert!(importer.contains("set_attr_nvvm_register_mma_operation"));
    assert!(importer.contains("set_attr_nvvm_register_mma_kind"));
    for record in &records {
        assert!(importer.contains(&record.rust.canonical_path));
        assert!(importer.contains(&record.rust.compatibility_paths[0]));
        assert!(importer.contains(&format!(
            "set_generated_intrinsic_marker(ctx, mma, {:?})",
            intrinsic_marker(&catalog, record)
        )));
    }

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for RegisterMmaOp")
            .count(),
        1
    );
    assert!(lowering.contains("convert_generated_register_mma"));
    assert!(lowering.contains("operation_or_multiply"));
    assert!(lowering.contains("kind_or_inferred"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"));
    assert!(lowering.contains("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r"));
    assert!(lowering.contains("=d,=d,d,d,d,d"));
    assert!(lowering.contains("=r,=r,=r,=r,r,r,r,r,r,r,r"));
    assert!(lowering.contains("=r,=r,r,r,r,r"));
    assert!(lowering.contains("=r,=r,r,r,r,r,r,r,r,r"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.s4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc"));
    assert!(lowering.contains("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"));
    assert!(lowering.contains("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc"));
    for op_type in [
        "MmaM16N8K16F32Bf16Op",
        "MmaM16N8K16F32F16Op",
        "MmaM16N8K8F32Tf32Op",
        "MmaM16N8K32S32S8Op",
        "MmaM8N8K4F64Op",
    ] {
        assert_eq!(
            lowering
                .matches(&format!("impl MirToLlvmConversion for {op_type}"))
                .count(),
            1
        );
    }
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 4, 7, "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 {$0, $1, $2, $3}, {$8, $9}, {$10}, {$4, $5, $6, $7};", "=r,=r,=r,=r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 4, 10, "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};", "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 2, 4, "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {$0, $1}, {$4}, {$5}, {$2, $3};", "=r,=r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 4, 7, "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {$0, $1, $2, $3}, {$8, $9}, {$10}, {$4, $5, $6, $7};", "=r,=r,=r,=r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 4, 10, "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};", "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::F32, 4, 10, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};", "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::F32, 4, 10, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};", "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 2, 8, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m1.e2m1.f16 {$0, $1}, {$4, $5, $6, $7}, {$8, $9}, {$2, $3};", "=r,=r,r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 2, 8, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e5m2.e5m2.f16 {$0, $1}, {$4, $5, $6, $7}, {$8, $9}, {$2, $3};", "=r,=r,r,r,r,r,r,r,r,r")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::F32, 4, 16, "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e2m1.e2m1.f32.ue8m0 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7}, $14, {$15, $16}, $17, {$18, $19};", "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r,r,h,h,r,h,h")"#
        ));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("GeneratedIntrinsicVariant::RegisterMma"));
    assert!(targets.contains("GeneratedRegisterMmaShape::M8n8k4"));
    assert!(targets.contains("GeneratedRegisterMmaShape::M16n8k4"));
    assert!(targets.contains(
            "pub enum GeneratedRegisterMmaShape { M8n8k4, M8n8k16, M8n8k32, M8n8k128, M16n8k4, M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128, M16n8k256 }"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M8n8k16 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k16)"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M8n8k32 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k32)"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M8n8k128 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k128)"
        ));
    assert!(targets.contains("GeneratedRegisterMmaShape::M16n8k32"));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M16n8k64 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k64)"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M16n8k128 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k128)"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaShape::M16n8k256 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k256)"
        ));
    assert!(targets.contains("GeneratedRegisterMmaOperation::Multiply"));
    assert!(targets.contains("GeneratedRegisterMmaOperation::AndPopc"));
    assert!(targets.contains("GeneratedRegisterMmaOperation::XorPopc"));
    assert!(targets.contains("pub enum GeneratedRegisterMmaKind { Standard, F8f6f4, Mxf8f6f4 }"));
    assert!(targets.contains("kind: GeneratedRegisterMmaKind::Standard"));
    assert!(targets.contains("kind: GeneratedRegisterMmaKind::F8f6f4"));
    assert!(targets.contains("kind: GeneratedRegisterMmaKind::Mxf8f6f4"));
    assert!(targets.contains("kind_or_inferred"));
    assert!(targets.contains("RegisterMmaKindAttr::Standard"));
    assert!(targets.contains("RegisterMmaKindAttr::F8f6f4"));
    assert!(targets.contains("RegisterMmaKindAttr::Mxf8f6f4"));
    assert!(targets.contains("GeneratedRegisterMmaAccumulator::F16"));
    assert!(targets.contains("operation: GeneratedRegisterMmaOperation::AndPopc"));
    assert!(targets.contains("operation: GeneratedRegisterMmaOperation::XorPopc"));
    assert!(targets.contains("operation: mma_operation"));
    assert!(targets.contains("operation_or_multiply"));
    assert!(targets.contains("GeneratedRegisterMmaElement::B1"));
    assert!(targets.contains("GeneratedRegisterMmaElement::E2m1"));
    assert!(targets.contains("GeneratedRegisterMmaElement::E5m2"));
    assert!(targets.contains("GeneratedRegisterMmaElement::S4"));
    assert!(targets.contains("GeneratedRegisterMmaElement::U4"));
    assert!(targets.contains("GeneratedRegisterMmaElement::U8"));
    assert!(targets.contains(
        "a_element: GeneratedRegisterMmaElement::S4, b_element: GeneratedRegisterMmaElement::U4"
    ));
    assert!(targets.contains(
        "a_element: GeneratedRegisterMmaElement::U4, b_element: GeneratedRegisterMmaElement::S4"
    ));
    assert!(targets.contains(
        "a_element: GeneratedRegisterMmaElement::S8, b_element: GeneratedRegisterMmaElement::U8"
    ));
    assert!(targets.contains(
        "a_element: GeneratedRegisterMmaElement::U8, b_element: GeneratedRegisterMmaElement::S8"
    ));
    assert!(targets.contains("overflow: GeneratedRegisterMmaOverflow::Satfinite"));
    assert!(targets.contains("get_attr_nvvm_register_mma_overflow"));
    assert!(targets.contains("minimum_ptx: GeneratedPtxVersion::from_encoded(87)"));
    assert!(targets.contains("minimum_ptx: GeneratedPtxVersion::from_encoded(84)"));
    assert!(targets.contains("GeneratedHardwareAlternative::ExactArchitecture(120)"));
    assert!(targets.contains("GeneratedHardwareAlternative::FamilyTarget(121)"));

    let first_dense = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_f32_e2m1_e2m1")
        .unwrap();
    let last_dense = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_f32_e5m2_e5m2")
        .unwrap();
    assert_eq!(first_dense.rust.abi_id, "i0454");
    assert_eq!(last_dense.rust.abi_id, "i0478");
    assert_eq!(
        register_mma_constraints(first_dense),
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r"
    );
    let first_dense_f16 = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_f16_e2m1_e2m1")
        .unwrap();
    let last_dense_f16 = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_f16_e5m2_e5m2")
        .unwrap();
    assert_eq!(first_dense_f16.rust.abi_id, "i0479");
    assert_eq!(last_dense_f16.rust.abi_id, "i0503");
    assert_eq!(
        register_mma_constraints(first_dense_f16),
        "=r,=r,r,r,r,r,r,r,r,r"
    );
    assert!(render_probe(&catalog, first_dense_f16, "test-hash").contains("define { i32, i32 }"));

    let first_mxf8f6f4 = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_mxf8f6f4_f32_e2m1_e2m1")
        .unwrap();
    let last_mxf8f6f4 = records
        .iter()
        .find(|record| record.id == "mma_m16n8k32_mxf8f6f4_f32_e5m2_e5m2")
        .unwrap();
    assert_eq!(first_mxf8f6f4.rust.abi_id, "i0858");
    assert_eq!(last_mxf8f6f4.rust.abi_id, "i0882");
    assert_eq!(
        register_mma_constraints(first_mxf8f6f4),
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r,r,h,h,r,h,h"
    );
    assert!(
        render_probe(&catalog, first_mxf8f6f4, "test-hash").contains("kind::mxf8f6f4.block_scale")
    );
    let first_mxf8f6f4_compatibility = format!("pub unsafe fn {}(", first_mxf8f6f4.rust.name);
    let first_mxf8f6f4_compatibility = compatibility
        .find(&first_mxf8f6f4_compatibility)
        .expect("generated MXF8F6F4 compatibility wrapper");
    assert!(
        compatibility[..first_mxf8f6f4_compatibility]
            .ends_with("#[allow(clippy::too_many_arguments)]\n#[must_use]\n#[inline(never)]\n")
    );

    for record in &records {
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("asm sideeffect"));
        assert!(probe.contains("attributes #0 = { convergent }"));
        assert!(probe.contains(&register_mma_template(record)));
        assert!(probe.contains(&register_mma_constraints(record)));
    }

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Register-MMA contracts"));
    assert!(reference.contains("performs XOR, population count, and accumulate"));
    assert!(reference.contains("performs AND, population count, and accumulate"));
    assert!(reference.contains("runtime validation is not executed on a GPU"));
    for record in &records {
        assert!(reference.contains(&format!("- `{}`: runtime `unexecuted`", record.id)));
    }

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/register_mma.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/register_mma.rs"
    )));
}

#[test]
fn register_mma_kind_is_explicit_and_legacy_ir_stays_compatible() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut catalog: CatalogFile = read_json(&repo_root.join("intrinsics/catalog.json")).unwrap();
    let old = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == "mma_m16n8k32_f16_e4m3_e4m3")
        .unwrap()
        .clone();
    assert_eq!(old.register_mma.as_ref().unwrap().kind, None);
    assert_eq!(
        register_mma_attr_variants(&old).2,
        "RegisterMmaKindAttr::F8f6f4"
    );
    assert!(generated_intrinsic_variant(&old).contains("kind: GeneratedRegisterMmaKind::F8f6f4"));

    let mut standard = old.clone();
    standard.id = "mma_m16n8k32_fp8_f16_e4m3_e4m3".into();
    standard.operation_key = "matrix.mma.m16n8k32.row.col.standard_fp8.f16.e4m3.e4m3.f16".into();
    standard.rust.abi_id = "i9999".into();
    standard.rust.name = standard.id.clone();
    standard.rust.canonical_path = format!("cuda_intrinsics::matrix::{}", standard.id);
    standard.rust.public_path = standard.rust.canonical_path.clone();
    standard.rust.compatibility_paths = vec![format!("cuda_device::wmma::{}", standard.id)];
    standard.llvm.as_mut().unwrap().symbol =
        "llvm.nvvm.mma.m16n8k32.row.col.f16.e4m3.e4m3.f16".into();
    standard.register_mma.as_mut().unwrap().kind = Some(RegisterMmaKind::Standard);
    standard
        .expected_ptx
        .modifiers
        .retain(|modifier| modifier != "kind::f8f6f4");
    assert_eq!(
        register_mma_attr_variants(&standard).2,
        "RegisterMmaKindAttr::Standard"
    );
    assert!(
        generated_intrinsic_variant(&standard).contains("kind: GeneratedRegisterMmaKind::Standard")
    );
    catalog.intrinsics.push(standard);

    let dialect = render_dialect_register_mma(&catalog, "test-hash");
    assert!(dialect.contains("pub fn kind_or_inferred"));
    assert!(
        dialect.contains("let old_f8f6f4 = self.get_attr_nvvm_register_mma_shape(ctx).as_deref()")
    );
    assert!(dialect.contains("== Some(&RegisterMmaShapeAttr::M16n8k32)"));
    assert!(
        dialect
            .contains("&& low_format(self.get_attr_nvvm_register_mma_a_element(ctx).as_deref())")
    );
    assert!(
        dialect
            .contains("&& low_format(self.get_attr_nvvm_register_mma_b_element(ctx).as_deref())")
    );
    assert!(dialect.contains("RegisterMmaKindAttr::F8f6f4"));
    assert!(dialect.contains("RegisterMmaKindAttr::Standard"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("set_attr_nvvm_register_mma_kind(ctx, RegisterMmaKindAttr::F8f6f4)"));
    assert!(
        importer.contains("set_attr_nvvm_register_mma_kind(ctx, RegisterMmaKindAttr::Standard)")
    );

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("let kind = self.kind_or_inferred(ctx)"));
    assert!(lowering.contains("kind::f8f6f4.f16.e4m3.e4m3.f16"));
    assert!(lowering.contains("row.col.f16.e4m3.e4m3.f16"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("kind: GeneratedRegisterMmaKind::F8f6f4"));
    assert!(targets.contains("kind: GeneratedRegisterMmaKind::Standard"));
    assert!(targets.contains(
            "GeneratedRegisterMmaKind::Standard => op.kind_or_inferred(ctx) == RegisterMmaKindAttr::Standard"
        ));
    assert!(targets.contains(
            "GeneratedRegisterMmaKind::F8f6f4 => op.kind_or_inferred(ctx) == RegisterMmaKindAttr::F8f6f4"
        ));
    assert!(targets.contains("kind_matches && accumulator_matches"));
}

#[test]
fn sparse_mma_rendering_enforces_selector_and_keeps_family_distinct() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = sparse_mmas(&catalog).collect::<Vec<_>>();
    let k32 = records
        .iter()
        .filter(|record| record.sparse_mma.as_ref().unwrap().shape == SparseMmaShape::M16n8k32)
        .count();
    let k64 = records
        .iter()
        .filter(|record| record.sparse_mma.as_ref().unwrap().shape == SparseMmaShape::M16n8k64)
        .count();
    let k128 = records
        .iter()
        .filter(|record| record.sparse_mma.as_ref().unwrap().shape == SparseMmaShape::M16n8k128)
        .count();
    assert_eq!((records.len(), k32, k64, k128), (126, 19, 86, 16));
    let standard_k64 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k64_s32_s8")
        .unwrap();
    assert_eq!(
        standard_k64.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
    );
    assert_eq!(
        standard_k64.sparse_mma.as_ref().unwrap().llvm_adapter,
        crate::model::SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
    );
    assert_eq!(
        sparse_mma_template(standard_k64),
        "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, {$4, $5, $6, $7}, $16, $17;"
    );
    assert_eq!(
        sparse_mma_constraints(standard_k64),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n"
    );
    let standard_int4_k64 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k64_s32_s4")
        .unwrap();
    assert_eq!(
        sparse_mma_template(standard_int4_k64),
        "mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$8, $9}, {$10, $11}, {$4, $5, $6, $7}, $12, $13;"
    );
    assert_eq!(
        sparse_mma_constraints(standard_int4_k64),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n"
    );
    let ordered_int4_k64 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k64_s32_s4")
        .unwrap();
    assert_eq!(
        ordered_int4_k64.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32
    );
    assert_eq!(
        ordered_int4_k64.sparse_mma.as_ref().unwrap().llvm_adapter,
        crate::model::SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32
    );
    assert_eq!(
        sparse_mma_template(ordered_int4_k64),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$8, $9}, {$10, $11}, {$4, $5, $6, $7}, $12, $13;"
    );
    assert_eq!(
        sparse_mma_constraints(ordered_int4_k64),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n"
    );
    let ordered_int4_k128 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k128_s32_s4")
        .unwrap();
    assert_eq!(
        ordered_int4_k128.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
    );
    assert_eq!(
        ordered_int4_k128.sparse_mma.as_ref().unwrap().llvm_adapter,
        crate::model::SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
    );
    assert_eq!(
        sparse_mma_template(ordered_int4_k128),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, {$4, $5, $6, $7}, $16, $17;"
    );
    assert_eq!(
        sparse_mma_constraints(ordered_int4_k128),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n"
    );
    let standard_int4_k128 = records
        .iter()
        .copied()
        .find(|record| record.id == "mma_sp_m16n8k128_s32_s4")
        .unwrap();
    assert_eq!(
        sparse_mma_template(standard_int4_k128),
        "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, {$4, $5, $6, $7}, $16, $17;"
    );
    assert_eq!(
        sparse_mma_constraints(standard_int4_k128),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n"
    );
    let ordered_f8f6f4 = records
        .iter()
        .copied()
        .find(|record| {
            record.id == "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32"
        })
        .unwrap();
    assert_eq!(
        ordered_f8f6f4.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
    );
    assert_eq!(
        sparse_mma_template(ordered_f8f6f4),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, {$4, $5, $6, $7}, $16, $17;"
    );
    assert_eq!(
        sparse_mma_constraints(ordered_f8f6f4),
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r,r,r,r,n"
    );
    assert_eq!(
            sparse_mma_carriers(ordered_f8f6f4),
            (
                "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]".into(),
                "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32]".into(),
            )
        );
    let ordered_f8f6f4_f16 = records
        .iter()
        .copied()
        .find(|record| {
            record.id == "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16"
        })
        .unwrap();
    assert_eq!(
        ordered_f8f6f4_f16.sparse_mma.as_ref().unwrap().adapter,
        SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32
    );
    assert_eq!(sparse_mma_fragment_counts(ordered_f8f6f4_f16), (2, 4, 4, 2));
    assert_eq!(
        sparse_mma_template(ordered_f8f6f4_f16),
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f16.e2m1.e2m1.f16 {$0, $1}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$2, $3}, $12, $13;"
    );
    assert_eq!(
        sparse_mma_constraints(ordered_f8f6f4_f16),
        "=r,=r,r,r,r,r,r,r,r,r,r,r,r,n"
    );
    assert_eq!(
            sparse_mma_carriers(ordered_f8f6f4_f16),
            (
                "&[MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]".into(),
                "&[MmaCarrier::U32, MmaCarrier::U32]".into(),
            )
        );
    assert_eq!(sparse_mma_selector_values(ordered_f8f6f4_f16), &[0]);

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("must be the compile-time constant `0` or `1`"));
    assert!(raw.contains("must be the compile-time constant `0`"));
    for record in &records {
        let (c_count, a_count, b_count, d_count) = sparse_mma_fragment_counts(record);
        let scalar = match record.sparse_mma.as_ref().unwrap().accumulator {
            SparseMmaAccumulator::F16 => "u32",
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        };
        assert!(raw.contains(&format!(
                "pub unsafe fn {}(_arg0: [{scalar}; {c_count}], _arg1: [u32; {a_count}], _arg2: [u32; {b_count}], _arg3: u32, _arg4: u32) -> [{scalar}; {d_count}]",
                record.rust.abi_id,
            )));
    }

    let compatibility = render_compat_sparse_mma(&catalog, "test-hash");
    assert_eq!(compatibility.matches("pub unsafe fn ").count(), 126);
    assert!(
        compatibility
            .contains("c: [i32; 4], a: [u32; 2], b: [u32; 2], metadata: u32, selector: u32")
    );
    assert!(
        compatibility
            .contains("c: [i32; 4], a: [u32; 4], b: [u32; 4], metadata: u32, selector: u32")
    );
    assert!(
        compatibility
            .contains("c: [f32; 4], a: [u32; 4], b: [u32; 4], metadata: u32, selector: u32")
    );
    assert!(
        compatibility
            .contains("c: [u32; 2], a: [u32; 4], b: [u32; 4], metadata: u32, selector: u32")
    );

    let dialect = render_dialect_sparse_mma(&catalog, "test-hash");
    assert_eq!(dialect.matches("pub struct SparseMmaOp").count(), 1);
    assert!(dialect.contains("MmaCarrier::F32"));
    assert!(dialect.contains("MmaCarrier::I32"));
    assert!(dialect.contains("MmaCarrier::U32"));
    assert!(dialect.contains("operands.len() != expected_operands"));
    assert!(
        dialect.contains("SparseMmaShapeAttr { M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128 }")
    );
    assert!(dialect.contains("SparseMmaAccumulatorAttr { F16, F32, S32 }"));
    assert!(dialect.contains(
        "SparseMmaSelectorAttr { ImmediateZeroThroughThree, ImmediateZeroOrOne, ImmediateZero }"
    ));
    assert!(dialect.contains("SparseMmaElementAttr::E2m1"));
    assert!(dialect.contains("SparseMmaElementAttr::E2m3"));
    assert!(dialect.contains("SparseMmaElementAttr::E3m2"));
    assert!(dialect.contains("SparseMmaElementAttr::E4m3"));
    assert!(dialect.contains("SparseMmaElementAttr::E5m2"));
    assert!(dialect.contains("SparseMmaElementAttr::S4"));
    assert!(dialect.contains("SparseMmaElementAttr::U4"));
    assert!(dialect.contains("SparseMmaElementAttr::S8"));
    assert!(dialect.contains("SparseMmaElementAttr::U8"));
    assert!(dialect.contains("SparseMmaOverflowAttr::Wrapping"));
    assert!(dialect.contains("SparseMmaOverflowAttr::Satfinite"));
    assert!(dialect.contains("SparseMmaOverflowAttr::NotApplicable"));
    assert!(dialect.contains("SparseMmaMetadataAttr { Standard, Ordered }"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("sparse MMA selector must be the compile-time constant 0 or 1"));
    assert!(importer.contains("sparse MMA selector must be the compile-time constant 0"));
    assert!(importer.contains("GeneratedMmaImportAdapter::C4I32A2U32B2U32ToD4I32"));
    assert!(importer.contains("GeneratedMmaImportAdapter::C4I32A4U32B4U32ToD4I32"));
    assert!(importer.contains("GeneratedMmaImportAdapter::C4F32A4U32B4U32ToD4F32"));
    assert!(importer.contains("GeneratedMmaImportAdapter::C2U32A4U32B4U32ToD2U32"));
    assert!(importer.contains("(u32_ty, 2, u32_ty, 4, true, u32_ty, 4, true, u32_ty, 2)"));
    assert!(importer.contains("let (c_array, last_op) = rvalue::translate_operand("));
    assert!(importer.contains("ctx, body, &args[0], value_map"));
    assert!(importer.contains("let (a_value, last_after_a) = rvalue::translate_operand("));
    assert!(importer.contains("ctx, body, &args[1], value_map"));
    assert!(importer.contains("let (b_value, last_after_b) = rvalue::translate_operand("));
    assert!(importer.contains("ctx, body, &args[2], value_map"));
    assert!(importer.contains("operands.push(metadata)"));
    assert!(importer.contains("operands.push(selector_value)"));
    assert!(importer.contains("SparseMmaOp::get_concrete_op_info"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert_eq!(
        lowering
            .matches("impl MirToLlvmConversion for SparseMmaOp")
            .count(),
        1
    );
    assert!(lowering.contains("convert_generated_sparse_mma"));
    for record in &records {
        assert!(lowering.contains(&sparse_mma_template(record)));
        assert!(lowering.contains(&sparse_mma_constraints(record)));
    }

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("GeneratedIntrinsicVariant::SparseMma"));
    assert!(targets.contains("Operation::get_op::<SparseMmaOp>"));
    assert!(targets.contains("get_attr_nvvm_sparse_mma_selector"));
    assert!(targets.contains("GeneratedSparseMmaSelector::ImmediateZeroOrOne"));
    assert!(targets.contains("GeneratedSparseMmaSelector::ImmediateZero"));
    assert!(targets.contains("GeneratedSparseMmaShape::M16n8k64"));
    assert!(targets.contains("SparseMmaShapeAttr::M16n8k64"));
    assert!(targets.contains("GeneratedSparseMmaShape::M16n8k128"));
    assert!(targets.contains("SparseMmaShapeAttr::M16n8k128"));
    assert!(targets.contains("GeneratedSparseMmaElement::S4"));
    assert!(targets.contains("GeneratedSparseMmaElement::U4"));
    assert!(targets.contains("GeneratedSparseMmaElement::E2m1"));
    assert!(targets.contains("GeneratedSparseMmaElement::E5m2"));
    assert!(targets.contains("SparseMmaElementAttr::S4"));
    assert!(targets.contains("SparseMmaElementAttr::U4"));
    assert!(targets.contains("GeneratedSparseMmaMetadata::Standard"));
    assert!(targets.contains("SparseMmaMetadataAttr::Standard"));
    assert!(targets.contains("GeneratedSparseMmaMetadata::Ordered"));
    assert!(targets.contains("SparseMmaMetadataAttr::Ordered"));
    assert!(targets.contains("GeneratedSparseMmaAccumulator::F16"));
    assert!(targets.contains("GeneratedSparseMmaAccumulator::F32"));
    assert!(targets.contains("GeneratedSparseMmaOverflow::NotApplicable"));
    assert!(targets.contains("GeneratedHardwareAlternative::ExactArchitecture(120)"));

    assert_eq!(raw.matches(SPARSE_MMA_STANDARD_METADATA_RULE).count(), 36);
    assert_eq!(raw.matches(SPARSE_MMA_ORDERED_METADATA_RULE).count(), 88);
    assert_eq!(
        raw.matches(SPARSE_MMA_ORDERED_TF32_METADATA_RULE).count(),
        2
    );
    assert_eq!(
        compatibility
            .matches(SPARSE_MMA_STANDARD_METADATA_RULE)
            .count(),
        36
    );
    assert_eq!(
        compatibility
            .matches(SPARSE_MMA_ORDERED_METADATA_RULE)
            .count(),
        88
    );
    assert_eq!(
        compatibility
            .matches(SPARSE_MMA_ORDERED_TF32_METADATA_RULE)
            .count(),
        2
    );
    assert!(
        lowering.contains("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
    );
    assert!(
        lowering.contains("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32")
    );
    assert!(
        lowering.contains("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32")
    );
    assert!(
        lowering.contains("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32")
    );
    assert!(lowering.contains("mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32"));
    assert!(lowering.contains("mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"));
    assert!(lowering.contains("mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32"));
    assert!(lowering.contains(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
    ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::F32, 4, 14, "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, {$4, $5, $6, $7}, $16, $17;", "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r,r,r,r,n")"#
        ));
    assert!(lowering.contains(
            r#"(GeneratedMmaResultType::I32, 2, 12, "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f16.e2m1.e2m1.f16 {$0, $1}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$2, $3}, $12, $13;", "=r,=r,r,r,r,r,r,r,r,r,r,r,r,n")"#
        ));
    let f16_probe = render_probe(&catalog, ordered_f8f6f4_f16, "test-hash");
    assert!(f16_probe.contains(
            "define { i32, i32 } @probe_mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16_selector_0"
        ));
    assert!(!f16_probe.contains("_selector_1"));

    for record in &records {
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(&format!("probe_{}_selector_0", record.id)));
        let selectors = sparse_mma_selector_values(record);
        assert_eq!(
            probe.contains(&format!("probe_{}_selector_1", record.id)),
            selectors.contains(&1)
        );
        assert_eq!(probe.matches("asm sideeffect").count(), selectors.len());
        assert!(probe.contains(&sparse_mma_template(record)));
        assert!(probe.contains(&sparse_mma_constraints(record)));
        assert!(probe.contains("attributes #0 = { convergent }"));
    }

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Sparse-MMA contracts"));
    assert!(reference.contains("C, A, B, metadata, selector order"));
    assert!(reference.contains("LLVM source record uses A, B, C, metadata, selector order"));
    assert_eq!(
        reference.matches(SPARSE_MMA_STANDARD_METADATA_RULE).count(),
        36
    );
    assert_eq!(
        reference.matches(SPARSE_MMA_ORDERED_METADATA_RULE).count(),
        88
    );
    assert_eq!(
        reference
            .matches(SPARSE_MMA_ORDERED_TF32_METADATA_RULE)
            .count(),
        2
    );
    let tf32_k8 = records
        .iter()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k8_f32_tf32")
        .unwrap();
    let tf32_k16 = records
        .iter()
        .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k16_f32_tf32")
        .unwrap();
    assert_eq!(
        sparse_mma_metadata_rule(tf32_k8.sparse_mma.as_ref().unwrap()),
        SPARSE_MMA_ORDERED_TF32_METADATA_RULE
    );
    assert_eq!(
        sparse_mma_selector_description(tf32_k8),
        "the compile-time constant `0`, `1`, `2`, or `3`"
    );
    assert_eq!(
        sparse_mma_metadata_rule(tf32_k16.sparse_mma.as_ref().unwrap()),
        SPARSE_MMA_ORDERED_TF32_METADATA_RULE
    );
    assert_eq!(
        sparse_mma_selector_description(tf32_k16),
        "the compile-time constant `0` or `1`"
    );
    let sparse_reference = reference
        .split("## Sparse-MMA contracts")
        .nth(1)
        .unwrap()
        .split("## Packed-atomic contracts")
        .next()
        .unwrap();
    assert!(sparse_reference.contains("Overflow mode is not applicable."));
    assert!(!sparse_reference.contains("Integer overflow is not applicable"));
    for record in &records {
        let runtime = match record.sparse_mma.as_ref().unwrap().runtime_validation {
            RuntimeValidation::Unexecuted => "unexecuted",
            RuntimeValidation::Executed => "executed",
        };
        assert!(reference.contains(&format!("- `{}`: runtime `{runtime}`", record.id)));
    }

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/sparse_mma.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/sparse_mma.rs"
    )));
}
