/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    AbiLedgerEntry, AbiLedgerFile, BackendLoweringMechanism, CatalogHardwareAlternative,
    CatalogHardwareTarget, ClcAdmission, ClcOperation, ClusterBarrierAdmission, ClusterBarrierMode,
    ClusterMemoryAdmission, ClusterMemoryOperation, DebugControlAdmission, DebugControlOperation,
    DotProductOperation, DotProductSignedness, EvidenceFile, EvidenceRecord, EvidenceStage,
    EvidenceStageKind, ImportedFile, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    MaskEncoding, MbarrierExtendedAdmission, MbarrierExtendedOperation, MovmatrixAdapter,
    MovmatrixParticipation, OverlayBackendLowering, OverlayFile, OverlayIntrinsic,
    PackedAluAdapter, PackedAluFormat, PackedAluOperation, PackedConversionAdapter,
    PackedConversionDestinationFormat, PackedConversionFp8Admission, PackedConversionFp8Direction,
    PackedConversionFp8F16x2Admission, PackedConversionFp8Format, PackedConversionRounding,
    PackedConversionSaturation, PackedConversionSourceFormat, PreSm70MemberMaskRule, PrmtAdmission,
    PrmtMode, RegisterMmaAccumulator, RegisterMmaAmpereFloatAdmission, RegisterMmaF8F6F4Admission,
    RegisterMmaFp8Admission, RuntimeValidation, SparseMmaElement, SparseMmaF8F6F4Admission,
    SparseMmaF8F6F4F16Admission, SparseMmaFp8F32Admission, SpecialRegisterAdmission,
    StmatrixAdmission, StmatrixLayout, StmatrixMultiplicity, Tcgen05Admission,
    Tcgen05CpAdmissionVariant, Tcgen05CpGroup, Tcgen05LdAdmissionVariant,
    Tcgen05MmaAdmissionVariant, Tcgen05MmaForm, Tcgen05Operation, Tcgen05StAdmissionVariant,
    ThreadfenceAdmission, ThreadfenceScope, TmaAdmission, VoteAdapter, VoteMode, VoteParticipation,
    WarpShuffleAdapter, WarpShuffleMode, WarpShuffleParticipation, WarpShuffleSourceLane,
    WarpShuffleValueKind, WgmmaControlAdmission, WgmmaControlMode,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use crate::util::read_json;
use anyhow::Result;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::ImportedSelection;
use crate::resolve::abi_ledger::*;
use crate::resolve::evidence::*;
use crate::resolve::families::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::*;

pub(super) fn sreg_pattern(special_register: &str) -> InstructionPattern {
    InstructionPattern::new(
        "mov",
        &["u32"],
        vec![
            OperandPattern::Register,
            OperandPattern::Exact {
                value: special_register.into(),
            },
        ],
    )
}

pub(super) fn policy() -> OverlayIntrinsic {
    OverlayIntrinsic {
        id: "thread_idx_x".into(),
        abi_id: "i0001".into(),
        operation_key: "launch.thread_index.x".into(),
        family: "sreg".into(),
        source: None,
        source_record: Some("int_nvvm_read_ptx_sreg_tid_x".into()),
        rust_module: "sreg".into(),
        rust_name: "thread_idx_x".into(),
        rust_arguments: vec![],
        rust_result: "u32".into(),
        safe: true,
        must_use: false,
        safe_allowlist_reason: Some("no caller obligations".into()),
        public_rust_path: "cuda_intrinsics::sreg::thread_idx_x".into(),
        compatibility_rust_paths: vec!["cuda_device::thread::threadIdx_x".into()],
        dialect_op_type: "ReadPtxSregTidXOp".into(),
        dialect_op_name: "nvvm.read_ptx_sreg_tid_x".into(),
        dialect_operands: vec![],
        dialect_results: vec![],
        llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.tid.x".into()),
        resolved_llvm_symbol: None,
        llvm_arguments: vec![],
        llvm_results: vec!["i32".into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: "2.0".into(),
        minimum_sm: None,
        ptx_result: "u32".into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: "10.1 Special Registers: %tid".into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/".into(),
        lowering: "direct_nvvm".into(),
        backend_lowerings: vec![],
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        integer_minmax: None,
        packed_conversion: None,
        scalar_conversion: None,
        scalar_arithmetic: None,
        scalar_math: None,
        extended_minmax: None,
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: None,
        prmt: None,
        cluster_barrier: None,
        wgmma_control: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        tma: None,
        tcgen05: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: sreg_pattern("%tid.x"),
        summary: "thread index".into(),
    }
}

pub(super) fn distinct_policy() -> OverlayIntrinsic {
    let mut record = policy();
    record.id = "thread_idx_y".into();
    record.abi_id = "i0002".into();
    record.operation_key = "launch.thread_index.y".into();
    record.source_record = Some("int_nvvm_read_ptx_sreg_tid_y".into());
    record.rust_name = "thread_idx_y".into();
    record.public_rust_path = "cuda_intrinsics::sreg::thread_idx_y".into();
    record.compatibility_rust_paths = vec!["cuda_device::thread::threadIdx_y".into()];
    record.dialect_op_type = "ReadPtxSregTidYOp".into();
    record.dialect_op_name = "nvvm.read_ptx_sreg_tid_y".into();
    record.llvm_symbol = Some("llvm.nvvm.read.ptx.sreg.tid.y".into());
    record.expected_ptx = sreg_pattern("%tid.y");
    record
}

pub(super) fn movmatrix_policy() -> OverlayIntrinsic {
    let mut record = policy();
    record.id = "movmatrix_trans_b16".into();
    record.abi_id = "i0305".into();
    record.operation_key = "movmatrix.m8n8.trans.b16".into();
    record.family = "movmatrix".into();
    record.source = Some(IntrinsicSource::PtxNative {
        instruction: "movmatrix.sync.aligned.m8n8.trans.b16".into(),
    });
    record.source_record = None;
    record.rust_module = "matrix".into();
    record.rust_name = "movmatrix_trans_b16".into();
    record.rust_arguments = vec!["u32".into()];
    record.rust_result = "u32".into();
    record.safe = false;
    record.must_use = true;
    record.safe_allowlist_reason = None;
    record.public_rust_path = "cuda_intrinsics::matrix::movmatrix_trans_b16".into();
    record.compatibility_rust_paths = vec!["cuda_device::wmma::movmatrix_trans_b16".into()];
    record.dialect_op_type = "MovmatrixTransB16Op".into();
    record.dialect_op_name = "nvvm.movmatrix_trans_b16".into();
    record.dialect_operands = vec!["i32".into()];
    record.dialect_results = vec!["i32".into()];
    record.llvm_symbol = None;
    record.resolved_llvm_symbol = None;
    record.llvm_arguments.clear();
    record.llvm_results.clear();
    record.pure = false;
    record.memory = "inaccessible_read_write".into();
    record.convergent = true;
    record.execution_scope = "warp".into();
    record.minimum_ptx = "7.8".into();
    record.minimum_sm = Some("sm_75".into());
    record.ptx_result = "u32".into();
    record.targets = "all".into();
    record.ptx_isa_section =
        "9.7.15.5.17 Warp-level matrix transpose instruction: movmatrix".into();
    record.lowering = "generated_movmatrix_inline_ptx".into();
    record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
        .into_iter()
        .map(|backend| OverlayBackendLowering {
            backend,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: match backend {
                IntrinsicBackend::LlvmNvptx => "llvm-test",
                IntrinsicBackend::LibNvvm => "libnvvm-test",
            }
            .into(),
            targets: None,
            minimum_ptx: Some("7.8".into()),
            minimum_sm: Some("sm_75".into()),
        })
        .collect();
    record.movmatrix = Some(crate::model::Movmatrix {
        participation: MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes,
        adapter: MovmatrixAdapter::PackedB16x2U32ToPackedB16x2U32,
        runtime_validation: RuntimeValidation::Unexecuted,
    });
    record.expected_ptx = InstructionPattern::new(
        "movmatrix",
        &["sync", "aligned", "m8n8", "trans", "b16"],
        vec![OperandPattern::Register, OperandPattern::Register],
    );
    record.summary = "Transposes one packed b16 matrix fragment across a warp.".into();
    record
}

pub(super) fn declaration() -> ImportedIntrinsic {
    ImportedIntrinsic {
        source_record: "int_nvvm_read_ptx_sreg_tid_x".into(),
        llvm_name: "llvm.nvvm.read.ptx.sreg.tid.x".into(),
        arguments: vec![],
        results: vec!["i32".into()],
        classes: vec!["NVVMPureIntrinsic".into()],
        properties: vec![
            "IntrNoMem".into(),
            "IntrSpeculatable".into(),
            "NoUndef<ret>".into(),
            "Range<ret,0,1024>".into(),
        ],
        selections: vec![ImportedSelection {
            source_record: "INT_PTX_SREG_TID_x".into(),
            asm: "mov.u32 $d, %tid.x;".into(),
            predicates: vec![],
            constraints: Default::default(),
        }],
    }
}

pub(super) fn evidence() -> EvidenceRecord {
    EvidenceRecord {
        id: "thread_idx_x".into(),
        source: None,
        source_record: Some("int_nvvm_read_ptx_sreg_tid_x".into()),
        llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.tid.x".into()),
        resolved_llvm_symbol: None,
        llvm_arguments: vec![],
        llvm_results: vec!["i32".into()],
        concrete_llvm_arguments: vec![],
        concrete_llvm_results: vec![],
        target_triple: "nvptx64-nvidia-cuda".into(),
        gpu_target: "sm_70".into(),
        ptx_feature: "+ptx60".into(),
        status: "lowered".into(),
        stages: vec![],
        declaration_attributes_canonicalized: None,
        runtime_validation: None,
        expected_ptx: sreg_pattern("%tid.x"),
    }
}

pub(super) fn validate_test_evidence(
    policy: &OverlayIntrinsic,
    record: EvidenceRecord,
) -> Result<()> {
    let file = EvidenceFile {
        schema: 3,
        backend_profile: "test".into(),
        backend_kind: None,
        llvm_revision: "test".into(),
        backend_version: "LLVM version test".into(),
        backend_sha256: "0123456789abcdef".into(),
        artifact_path: None,
        build_id_prefix: None,
        nvvm_ir_version: None,
        debug_ir_version: None,
        records: vec![record],
    };
    let indexed = IndexedEvidence {
        file: &file,
        record: &file.records[0],
        backend_version: &file.backend_version,
        backend_sha256: &file.backend_sha256,
    };
    validate_evidence(policy, &indexed, None)
}

pub(super) fn shared_matrix_stage() -> EvidenceStage {
    EvidenceStage {
        targets: vec!["sm_80".into(), "ptx71".into()],
        representation: "shared fixture".into(),
        stage: EvidenceStageKind::BackendCodegen,
        mechanism: Some(BackendLoweringMechanism::InlinePtx),
        outcome: "succeeded".into(),
        detail: "$dst remains fixture text".into(),
        artifact_kind: None,
        tool_path: None,
        tool_version: None,
        tool_sha256: None,
    }
}

pub(super) fn synthetic_matrix_json() -> serde_json::Value {
    serde_json::json!({
        "schema": 6,
        "backend_profile": "matrix-test",
        "backend_kind": "llvm_nvptx",
        "llvm_revision": "test",
        "backend_version": "LLVM matrix test",
        "backend_sha256": "0123456789abcdef",
        "defaults": {
            "llvm_arguments": ["i32"],
            "llvm_results": ["i32"],
            "target_triple": "nvptx64-nvidia-cuda",
            "gpu_target": "sm_80",
            "ptx_feature": "+ptx71",
            "status": "lowered"
        },
        "fixtures": [{
            "id": "shared",
            "coverage_count": 2,
            "stages": [{
                "targets": ["sm_80", "ptx71"],
                "representation": "shared fixture",
                "stage": "backend_codegen",
                "mechanism": "inline_ptx",
                "outcome": "succeeded",
                "detail": "$dst remains fixture text"
            }]
        }],
        "matrices": [{
            "axes": [{
                "name": "element",
                "values": ["s8", "u8"]
            }],
            "product_count": 2,
            "fixtures": ["shared"],
            "template": {
                "id": "synthetic_${element}",
                "source_record": "int_synthetic_${element}",
                "llvm_symbol": "llvm.synthetic.${element}",
                "expected_ptx": {
                    "mnemonic": "mma",
                    "modifiers": ["sync", "${element}"],
                    "operands": [{"kind": "register"}]
                }
            }
        }],
        "records": [{
            "id": "synthetic_explicit",
            "source_record": "int_synthetic_explicit",
            "llvm_symbol": "llvm.synthetic.explicit",
            "llvm_arguments": ["i32"],
            "llvm_results": ["i32"],
            "target_triple": "nvptx64-nvidia-cuda",
            "gpu_target": "sm_80",
            "ptx_feature": "+ptx71",
            "status": "lowered",
            "expected_ptx": {
                "mnemonic": "mma",
                "modifiers": ["sync", "explicit"],
                "operands": [{"kind": "register"}]
            }
        }]
    })
}

pub(super) fn policy_matrix_json() -> serde_json::Value {
    serde_json::json!({
        "schema": 6,
        "backend_profile": "matrix-test",
        "llvm_revision": "test",
        "backend_version": "LLVM matrix test",
        "backend_sha256": "0123456789abcdef",
        "defaults": {
            "llvm_arguments": [],
            "llvm_results": ["i32"],
            "target_triple": "nvptx64-nvidia-cuda",
            "gpu_target": "sm_70",
            "ptx_feature": "+ptx60",
            "status": "lowered"
        },
        "fixtures": [{
            "id": "policy_fixture",
            "coverage_count": 1,
            "stages": [{
                "targets": ["sm_70", "ptx60"],
                "representation": "policy fixture",
                "stage": "backend_codegen",
                "mechanism": "typed_nvvm",
                "outcome": "succeeded",
                "detail": "shared policy fixture"
            }]
        }],
        "matrices": [{
            "axes": [{
                "name": "axis",
                "values": ["x"]
            }],
            "product_count": 1,
            "fixtures": ["policy_fixture"],
            "template": {
                "id": "thread_idx_${axis}",
                "source_record": "int_nvvm_read_ptx_sreg_tid_${axis}",
                "llvm_symbol": "llvm.nvvm.read.ptx.sreg.tid.${axis}",
                "expected_ptx": {
                    "mnemonic": "mov",
                    "modifiers": ["u32"],
                    "operands": [
                        {"kind": "register"},
                        {"kind": "exact", "value": "%tid.${axis}"}
                    ]
                }
            }
        }]
    })
}

pub(super) fn parse_synthetic_evidence(value: &serde_json::Value) -> Result<EvidenceFile> {
    parse_evidence_bytes(&serde_json::to_vec(value).unwrap(), "synthetic evidence")
}

pub(super) fn assert_synthetic_evidence_error(value: &serde_json::Value, expected: &str) {
    let error = parse_synthetic_evidence(value).unwrap_err();
    let message = format!("{error:#}");
    assert!(
        message.contains(expected),
        "expected {expected:?} in {message:?}"
    );
}

pub(super) fn overlay_file(records: Vec<OverlayIntrinsic>) -> OverlayFile {
    OverlayFile {
        schema: OVERLAY_SCHEMA,
        catalog_version: "test".into(),
        intrinsic_abi: 1,
        backend_profile: "test".into(),
        shards: vec![],
        intrinsics: records,
    }
}

pub(super) fn bind_pinned_abi_ids(repo_root: &Path, overlay: &mut OverlayFile) {
    let ledger_path = repo_root.join(format!("intrinsics/abi-v{}.toml", overlay.intrinsic_abi));
    let ledger: AbiLedgerFile =
        toml::from_str(&std::fs::read_to_string(ledger_path).unwrap()).unwrap();
    bind_generated_abi_ids(overlay, &ledger).unwrap();
}

pub(super) fn validate_imported_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let source = resolve_policy_source(policy)?;
    validate_policy(policy, &source, Some(declaration), 1)
}

pub(super) fn pinned_active_mask_and_warp_match_records()
-> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .into_iter()
        .map(|record| (record.source_record.clone(), record))
        .collect();

    overlay
        .intrinsics
        .into_iter()
        .filter(|record| matches!(record.family.as_str(), "active_mask" | "warp_match"))
        .map(|policy| {
            let declaration = declarations[policy.source_record.as_deref().unwrap()].clone();
            (policy.id.clone(), (policy, declaration))
        })
        .collect()
}

pub(super) fn pinned_mbarrier_basic_records()
-> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .into_iter()
        .map(|record| (record.source_record.clone(), record))
        .collect();

    overlay
        .intrinsics
        .into_iter()
        .filter(|record| record.family == "mbarrier_basic")
        .map(|policy| {
            let declaration = declarations[policy.source_record.as_deref().unwrap()].clone();
            (policy.id.clone(), (policy, declaration))
        })
        .collect()
}

pub(super) fn pinned_cp_async_mbarrier_records()
-> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .into_iter()
        .map(|record| (record.source_record.clone(), record))
        .collect();

    overlay
        .intrinsics
        .into_iter()
        .filter(|record| record.family == "cp_async_mbarrier")
        .map(|policy| {
            let declaration = declarations[policy.source_record.as_deref().unwrap()].clone();
            (policy.id.clone(), (policy, declaration))
        })
        .collect()
}

pub(super) fn packed_policy(id: &str) -> OverlayIntrinsic {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml"))
        .unwrap()
        .0
        .intrinsics
        .into_iter()
        .find(|record| record.id == id)
        .unwrap()
}

pub(super) fn packed_alu_policy(
    format: PackedAluFormat,
    operation: PackedAluOperation,
) -> OverlayIntrinsic {
    let recipe = packed_alu_recipe(format, operation).expect("test recipe pair");
    let (rust_module, rust_type, dialect_type, adapter) = match format {
        PackedAluFormat::Bf16x2 => ("bf16x2", "u32", "i32", PackedAluAdapter::DirectPackedU32),
        PackedAluFormat::F16x2 => ("f16x2", "u32", "i32", PackedAluAdapter::DirectPackedU32),
        PackedAluFormat::F32x2 => ("f32x2", "u64", "i64", PackedAluAdapter::DirectPackedU64),
    };
    let mut record = policy();
    record.id = recipe.id.into();
    record.abi_id = recipe.abi_id.into();
    record.operation_key = recipe.operation_key.into();
    record.family = "packed_alu".into();
    match &recipe.source {
        PackedAluRecipeSource::Imported {
            record: source_record,
            symbol,
            resolved_symbol,
            arguments,
            results,
            ..
        } => {
            record.source = None;
            record.source_record = Some((*source_record).into());
            record.llvm_symbol = Some((*symbol).into());
            record.resolved_llvm_symbol = resolved_symbol.map(str::to_owned);
            record.llvm_arguments = arguments.iter().map(|value| (*value).into()).collect();
            record.llvm_results = results.iter().map(|value| (*value).into()).collect();
        }
        PackedAluRecipeSource::PtxNative => {
            record.source = Some(IntrinsicSource::PtxNative {
                instruction: recipe.ptx_mnemonic.into(),
            });
            record.source_record = None;
            record.llvm_symbol = None;
            record.resolved_llvm_symbol = None;
            record.llvm_arguments.clear();
            record.llvm_results.clear();
        }
    }
    record.rust_module = rust_module.into();
    record.rust_name = recipe.rust_name.into();
    record.rust_arguments = vec![rust_type.into(); recipe.arity];
    record.rust_result = rust_type.into();
    record.safe = true;
    record.must_use = recipe.must_use;
    record.safe_allowlist_reason = Some("the operation has no caller obligations".into());
    record.public_rust_path = format!("cuda_intrinsics::{rust_module}::{}", recipe.rust_name);
    record.compatibility_rust_paths =
        vec![format!("cuda_device::{rust_module}::{}", recipe.rust_name)];
    record.dialect_op_type = recipe.dialect_op_type.into();
    record.dialect_op_name = recipe.dialect_op_name.into();
    record.dialect_operands = vec![dialect_type.into(); recipe.arity];
    record.dialect_results = vec![dialect_type.into()];
    record.pure = true;
    record.memory = "none".into();
    record.convergent = false;
    record.execution_scope = "thread".into();
    record.minimum_ptx = recipe.minimum_ptx.into();
    record.minimum_sm = Some(recipe.minimum_sm.into());
    record.ptx_result = rust_type.into();
    record.ptx_isa_section = recipe.ptx_isa_section.into();
    record.ptx_isa_url = recipe.ptx_isa_url.into();
    record.lowering = "generated_packed_alu_inline_ptx".into();
    record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
        .into_iter()
        .map(|backend| {
            let (minimum_ptx, minimum_sm) =
                packed_alu_backend_floor(&recipe, format, operation, backend);
            crate::model::OverlayBackendLowering {
                backend,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: format!("{backend:?}-test"),
                targets: None,
                minimum_ptx: Some(minimum_ptx.into()),
                minimum_sm: Some(minimum_sm.into()),
            }
        })
        .collect();
    record.packed_alu = Some(crate::model::PackedAlu {
        format,
        native_minimum_sm: recipe.native_minimum_sm,
        operation,
        adapter,
    });
    record.expected_ptx = InstructionPattern::new(
        recipe.ptx_mnemonic.split('.').next().unwrap(),
        recipe.modifiers,
        vec![OperandPattern::Register; recipe.arity + 1],
    );
    record.summary = format!("packed {rust_module} arithmetic");
    record
}

pub(super) fn packed_alu_declaration(
    format: PackedAluFormat,
    operation: PackedAluOperation,
) -> Option<ImportedIntrinsic> {
    let recipe = packed_alu_recipe(format, operation).expect("test recipe pair");
    let PackedAluRecipeSource::Imported {
        record,
        symbol,
        arguments,
        results,
        properties,
        selection,
        selection_asm,
        ..
    } = recipe.source
    else {
        return None;
    };
    let classes = if matches!(operation, PackedAluOperation::Min | PackedAluOperation::Max) {
        vec!["Intrinsic".into()]
    } else {
        vec!["Intrinsic".into(), "NVVMPureIntrinsic".into()]
    };
    let mut selections = vec![ImportedSelection {
        source_record: selection.into(),
        asm: selection_asm.into(),
        predicates: vec![
            format!("Subtarget->getSmVersion() >= {}", recipe.native_minimum_sm),
            format!(
                "Subtarget->getPTXVersion() >= {}",
                recipe.minimum_ptx.replace('.', "")
            ),
        ],
        constraints: Default::default(),
    }];
    if operation == PackedAluOperation::Abs {
        selections.extend((0..5).map(|index| ImportedSelection {
            source_record: format!("OTHER_ABS_{index}"),
            asm: "abs.f32 $dst, $src0;".into(),
            predicates: vec![],
            constraints: Default::default(),
        }));
    }
    Some(ImportedIntrinsic {
        source_record: record.into(),
        llvm_name: symbol.into(),
        arguments: arguments.iter().map(|value| (*value).into()).collect(),
        results: results.iter().map(|value| (*value).into()).collect(),
        classes,
        properties: properties.iter().map(|value| (*value).into()).collect(),
        selections,
    })
}

pub(super) fn packed_conversion_policy(
    destination_format: PackedConversionDestinationFormat,
    rounding: PackedConversionRounding,
    saturation: PackedConversionSaturation,
) -> OverlayIntrinsic {
    let conversion = crate::model::PackedConversion {
        source_format: PackedConversionSourceFormat::F32x2,
        destination_format,
        rounding,
        saturation,
        adapter: PackedConversionAdapter::ReverseHighLowOperands,
    };
    let recipe = packed_conversion_recipe(&conversion).expect("test packed-conversion recipe");
    let mut record = policy();
    record.id = recipe.id.into();
    record.abi_id = recipe.abi_id.into();
    record.operation_key = recipe.operation_key.into();
    record.family = "packed_conversion".into();
    record.source_record = Some(recipe.source_record.into());
    record.rust_module = "convert".into();
    record.rust_name = recipe.rust_name.into();
    record.rust_arguments = vec!["f32".into(), "f32".into()];
    let result_width = packed_conversion_result_width(&conversion);
    record.rust_result = format!("u{result_width}");
    record.safe = true;
    record.must_use = false;
    record.safe_allowlist_reason = Some("the operation has no caller obligations".into());
    record.public_rust_path = format!("cuda_intrinsics::convert::{}", recipe.rust_name);
    record.compatibility_rust_paths = vec![recipe.compatibility_path.into()];
    record.dialect_op_type = recipe.dialect_op_type.into();
    record.dialect_op_name = recipe.dialect_op_name.into();
    record.dialect_operands = vec!["f32".into(), "f32".into()];
    record.dialect_results = vec![format!("i{result_width}")];
    record.llvm_symbol = Some(recipe.llvm_symbol.into());
    record.llvm_arguments = vec!["f32".into(), "f32".into()];
    record.llvm_results = vec![recipe.llvm_result.into()];
    record.pure = true;
    record.memory = "none".into();
    record.convergent = false;
    record.execution_scope = "thread".into();
    let (minimum_ptx, minimum_sm) = packed_conversion_floor(&conversion);
    record.minimum_ptx = minimum_ptx.into();
    record.minimum_sm = Some(minimum_sm.into());
    record.ptx_result = format!("u{result_width}");
    record.ptx_isa_section = "9.7.9.22 Data Movement and Conversion Instructions: cvt".into();
    record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt".into();
    record.lowering = packed_conversion_lowering(&conversion).into();
    record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
        .into_iter()
        .map(|backend| OverlayBackendLowering {
            backend,
            mechanism: packed_conversion_backend_mechanism(&conversion, backend),
            evidence_profile: "test".into(),
            targets: None,
            minimum_ptx: Some(minimum_ptx.into()),
            minimum_sm: Some(minimum_sm.into()),
        })
        .collect();
    let modifiers = packed_conversion_ptx_modifiers(&conversion);
    record.packed_conversion = Some(conversion);
    record.expected_ptx =
        InstructionPattern::new("cvt", &modifiers, vec![OperandPattern::Register; 3]);
    record.summary = recipe.summary.into();
    record
}

pub(super) fn packed_conversion_declaration(policy: &OverlayIntrinsic) -> ImportedIntrinsic {
    ImportedIntrinsic {
        source_record: policy.source_record.clone().unwrap(),
        llvm_name: policy.llvm_symbol.clone().unwrap(),
        arguments: policy.llvm_arguments.clone(),
        results: policy.llvm_results.clone(),
        classes: vec!["Intrinsic".into(), "PureIntrinsic".into()],
        properties: vec![
            "IntrNoCreateUndefOrPoison".into(),
            "IntrNoMem".into(),
            "IntrSpeculatable".into(),
        ],
        selections: vec![],
    }
}

pub(super) fn packed_conversion_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
    let mut record = evidence();
    record.id = policy.id.clone();
    record.source_record = policy.source_record.clone();
    record.llvm_symbol = policy.llvm_symbol.clone();
    record.resolved_llvm_symbol = policy.resolved_llvm_symbol.clone();
    record.llvm_arguments = policy.llvm_arguments.clone();
    record.llvm_results = policy.llvm_results.clone();
    record.concrete_llvm_arguments = policy.llvm_arguments.clone();
    record.concrete_llvm_results = policy.llvm_results.clone();
    record.declaration_attributes_canonicalized = Some(true);
    record.expected_ptx = policy.expected_ptx.clone();
    record
}

pub(super) fn redux_policy() -> OverlayIntrinsic {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml"))
        .unwrap()
        .0
        .intrinsics
        .into_iter()
        .find(|record| record.id == "redux_sync_add")
        .unwrap()
}

pub(super) fn redux_declaration() -> ImportedIntrinsic {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
    serde_json::from_str::<ImportedFile>(&text)
        .unwrap()
        .intrinsics
        .into_iter()
        .find(|record| record.source_record == "int_nvvm_redux_sync_add")
        .unwrap()
}

pub(super) fn sync_policy() -> OverlayIntrinsic {
    packed_policy("sync_threads")
}

pub(super) fn sync_declaration() -> ImportedIntrinsic {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
    serde_json::from_str::<ImportedFile>(&text)
        .unwrap()
        .intrinsics
        .into_iter()
        .find(|record| record.source_record == "int_nvvm_barrier_cta_sync_aligned_all")
        .unwrap()
}

pub(super) fn warp_barrier_policy() -> OverlayIntrinsic {
    packed_policy("sync_mask")
}

pub(super) fn warp_barrier_declaration() -> ImportedIntrinsic {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
    serde_json::from_str::<ImportedFile>(&text)
        .unwrap()
        .intrinsics
        .into_iter()
        .find(|record| record.source_record == "int_nvvm_bar_warp_sync")
        .unwrap()
}

pub(super) fn vote_policy(mode: VoteMode) -> OverlayIntrinsic {
    let recipe = vote_recipe(mode);
    let mut record = policy();
    record.id = recipe.id.into();
    record.abi_id = recipe.abi_id.into();
    record.operation_key = recipe.operation_key.into();
    record.family = "vote".into();
    record.source_record = Some(recipe.source_record.into());
    record.rust_module = "warp".into();
    record.rust_name = recipe.rust_name.into();
    record.rust_arguments = vec!["u32".into(), "bool".into()];
    record.rust_result = recipe.rust_result.into();
    record.safe = false;
    record.must_use = true;
    record.safe_allowlist_reason = None;
    record.public_rust_path = format!("cuda_intrinsics::warp::{}", recipe.rust_name);
    record.compatibility_rust_paths = if recipe.has_compatibility_path {
        vec![format!("cuda_device::warp::{}", recipe.rust_name)]
    } else {
        vec![]
    };
    record.dialect_op_type = recipe.dialect_op_type.into();
    record.dialect_op_name = recipe.dialect_op_name.into();
    record.dialect_operands = vec!["i32".into(), "i1".into()];
    record.dialect_results = vec![recipe.llvm_result.into()];
    record.llvm_symbol = Some(recipe.llvm_symbol.into());
    record.llvm_arguments = vec!["i32".into(), "i1".into()];
    record.llvm_results = vec![recipe.llvm_result.into()];
    record.pure = false;
    record.memory = "inaccessible_read_write".into();
    record.convergent = true;
    record.execution_scope = "warp".into();
    record.minimum_ptx = "6.0".into();
    record.minimum_sm = Some("sm_30".into());
    record.ptx_result = recipe.rust_result.into();
    record.ptx_isa_section = "9.7.14.10 Warp Vote Instructions: vote.sync".into();
    record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync".into();
    record.lowering = "generated_vote".into();
    record.vote = Some(crate::model::Vote {
        mode,
        participation: VoteParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
        legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
        adapter: VoteAdapter::DirectMaskPredicate,
        mask_encoding: MaskEncoding::RegisterOrImmediate,
    });
    record.expected_ptx = InstructionPattern::new(
        "vote",
        &["sync", recipe.ptx_mode, recipe.ptx_type],
        vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::RegisterOrImmediate,
        ],
    );
    record.summary = "warp vote".into();
    record
}

pub(super) fn vote_declaration(mode: VoteMode) -> ImportedIntrinsic {
    let recipe = vote_recipe(mode);
    let selection = |source_record: &str| ImportedSelection {
        source_record: source_record.into(),
        asm: format!(
            "vote.sync.{}.{} \t$dest, $pred, $mask;",
            recipe.ptx_mode, recipe.ptx_type
        ),
        predicates: vec![
            "Subtarget->getPTXVersion() >= 60".into(),
            "Subtarget->getSmVersion() >= 30".into(),
        ],
        constraints: Default::default(),
    };
    ImportedIntrinsic {
        source_record: recipe.source_record.into(),
        llvm_name: recipe.llvm_symbol.into(),
        arguments: vec!["i32".into(), "i1".into()],
        results: vec![recipe.llvm_result.into()],
        classes: vec![
            "ClangBuiltin".into(),
            "NVVMBuiltin".into(),
            "SDPatternOperator".into(),
            "Intrinsic".into(),
        ],
        properties: vec![
            "IntrConvergent".into(),
            "IntrInaccessibleMemOnly".into(),
            "IntrNoCallback".into(),
        ],
        selections: vec![
            selection(recipe.immediate_selection),
            selection(recipe.register_selection),
        ],
    }
}

pub(super) fn warp_shuffle_policy(
    mode: WarpShuffleMode,
    value_kind: WarpShuffleValueKind,
) -> OverlayIntrinsic {
    let recipe = warp_shuffle_recipe(mode, value_kind);
    let mut record = policy();
    record.id = recipe.id.into();
    record.abi_id = recipe.abi_id.into();
    record.operation_key = recipe.operation_key.into();
    record.family = "warp_shuffle".into();
    match recipe.source {
        WarpShuffleRecipeSource::LlvmImported {
            source_record,
            llvm_symbol,
        } => {
            record.source_record = Some(source_record.into());
            record.llvm_symbol = Some(llvm_symbol.into());
            record.llvm_arguments = vec![
                "i32".into(),
                recipe.dialect_value.into(),
                "i32".into(),
                "i32".into(),
            ];
            record.llvm_results = vec![recipe.dialect_value.into()];
        }
        WarpShuffleRecipeSource::PtxNative { instruction } => {
            record.source = Some(IntrinsicSource::PtxNative {
                instruction: instruction.into(),
            });
            record.source_record = None;
            record.llvm_symbol = None;
            record.llvm_arguments.clear();
            record.llvm_results.clear();
        }
    }
    record.rust_module = "warp".into();
    record.rust_name = recipe.rust_name.into();
    record.rust_arguments = vec!["u32".into(), recipe.rust_value.into(), "u32".into()];
    record.rust_result = recipe.rust_value.into();
    record.safe = false;
    record.must_use = true;
    record.safe_allowlist_reason = None;
    record.public_rust_path = format!("cuda_intrinsics::warp::{}", recipe.rust_name);
    record.compatibility_rust_paths = vec![format!("cuda_device::warp::{}", recipe.rust_name)];
    record.dialect_op_type = recipe.dialect_op_type.into();
    record.dialect_op_name = recipe.dialect_op_name.into();
    record.dialect_operands = vec!["i32".into(), recipe.dialect_value.into(), "i32".into()];
    record.dialect_results = vec![recipe.dialect_value.into()];
    record.pure = false;
    record.memory = "inaccessible_read_write".into();
    record.convergent = true;
    record.execution_scope = "warp".into();
    record.minimum_ptx = "6.0".into();
    record.minimum_sm = Some("sm_30".into());
    record.ptx_result = recipe.rust_value.into();
    record.targets = "all".into();
    record.ptx_isa_section = "9.7.9.6 Data Movement and Conversion Instructions: shfl.sync".into();
    record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync".into();
    record.lowering = recipe.lowering.into();
    record.backend_lowerings = vec![
        crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LlvmNvptx,
            mechanism: recipe.backend_mechanism,
            evidence_profile: "llvm-test".into(),
            targets: None,
            minimum_ptx: Some("6.0".into()),
            minimum_sm: Some("sm_30".into()),
        },
        crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LibNvvm,
            mechanism: recipe.backend_mechanism,
            evidence_profile: "libnvvm-test".into(),
            targets: None,
            minimum_ptx: Some("6.0".into()),
            minimum_sm: Some("sm_75".into()),
        },
    ];
    record.warp_shuffle = Some(crate::model::WarpShuffle {
        mode,
        value_kind,
        participation:
            WarpShuffleParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
        legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
        source_lane: WarpShuffleSourceLane::InRangeSourceActiveAndNamedOutOfRangeCopiesSelf,
        adapter: recipe.adapter,
        clamp: recipe.clamp,
        lane_encoding: recipe.operand_encoding,
        mask_encoding: recipe.operand_encoding,
    });
    let operands = match recipe.adapter {
        WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp => vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::RegisterOrImmediate,
            OperandPattern::Exact {
                value: recipe.clamp.to_string(),
            },
            OperandPattern::RegisterOrImmediate,
        ],
        WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble => {
            vec![
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Exact { value: "lo".into() },
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: recipe.clamp.to_string(),
                },
                OperandPattern::Register,
            ]
        }
    };
    record.expected_ptx =
        InstructionPattern::new("shfl", &["sync", recipe.ptx_mode, "b32"], operands);
    record.summary = "synchronized warp shuffle".into();
    record
}

pub(super) fn warp_shuffle_declaration(
    mode: WarpShuffleMode,
    value_kind: WarpShuffleValueKind,
) -> ImportedIntrinsic {
    let recipe = warp_shuffle_recipe(mode, value_kind);
    let WarpShuffleRecipeSource::LlvmImported {
        source_record,
        llvm_symbol,
    } = recipe.source
    else {
        panic!("PTX-native i64 shuffles have no imported declaration");
    };
    let selections = (0..8)
        .map(|index| ImportedSelection {
            source_record: format!("anonymous_test_{index}"),
            asm: format!(
                "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
                recipe.ptx_mode
            ),
            predicates: vec![
                "Subtarget->getPTXVersion() >= 60".into(),
                "Subtarget->getSmVersion() >= 30".into(),
            ],
            constraints: Default::default(),
        })
        .collect();
    ImportedIntrinsic {
        source_record: source_record.into(),
        llvm_name: llvm_symbol.into(),
        arguments: vec![
            "i32".into(),
            recipe.dialect_value.into(),
            "i32".into(),
            "i32".into(),
        ],
        results: vec![recipe.dialect_value.into()],
        classes: vec![
            "ClangBuiltin".into(),
            "NVVMBuiltin".into(),
            "SDPatternOperator".into(),
            "Intrinsic".into(),
        ],
        properties: vec![
            "IntrConvergent".into(),
            "IntrInaccessibleMemOnly".into(),
            "IntrNoCallback".into(),
        ],
        selections,
    }
}

pub(super) fn sync_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
    let mut record = evidence();
    record.id = policy.id.clone();
    record.source_record = policy.source_record.clone();
    record.llvm_symbol = policy.llvm_symbol.clone();
    record.llvm_arguments = policy.llvm_arguments.clone();
    record.llvm_results = policy.llvm_results.clone();
    record.expected_ptx = policy.expected_ptx.clone();
    record
}

pub(super) fn dot_product_policy(
    operation: DotProductOperation,
    signedness: DotProductSignedness,
) -> OverlayIntrinsic {
    let recipe = dot_product_recipe(operation, signedness);
    let mut record = policy();
    record.id = recipe.id.into();
    record.abi_id = match (operation, signedness) {
        (DotProductOperation::Dp4a, DotProductSignedness::Signed) => "i0030",
        (DotProductOperation::Dp4a, DotProductSignedness::Unsigned) => "i0031",
        (DotProductOperation::Dp2a, DotProductSignedness::Signed) => "i0032",
        (DotProductOperation::Dp2a, DotProductSignedness::Unsigned) => "i0033",
    }
    .into();
    record.operation_key = recipe.operation_key.into();
    record.family = "dotprod".into();
    record.source = None;
    record.source_record = Some(recipe.source_record.into());
    record.rust_module = "dotprod".into();
    record.rust_name = recipe.rust_name.into();
    record.rust_arguments = vec!["u32".into(), "u32".into(), recipe.rust_value.into()];
    record.rust_result = recipe.rust_value.into();
    record.safe = true;
    record.must_use = false;
    record.safe_allowlist_reason = Some(
        "per-thread integer arithmetic has no memory, pointer, or participation obligations".into(),
    );
    record.public_rust_path = format!("cuda_intrinsics::dotprod::{}", recipe.rust_name);
    record.compatibility_rust_paths = vec![format!("cuda_device::dotprod::{}", recipe.rust_name)];
    record.dialect_op_type = recipe.dialect_op_type.into();
    record.dialect_op_name = recipe.dialect_op_name.into();
    record.dialect_operands = vec!["i32".into(), "i32".into(), "i32".into()];
    record.dialect_results = vec!["i32".into()];
    record.llvm_symbol = Some(recipe.llvm_symbol.into());
    record.resolved_llvm_symbol = None;
    record.llvm_arguments = recipe
        .llvm_arguments
        .iter()
        .map(|argument| (*argument).into())
        .collect();
    record.llvm_results = vec!["i32".into()];
    record.pure = true;
    record.memory = "none".into();
    record.convergent = false;
    record.execution_scope = "thread".into();
    record.minimum_ptx = "5.0".into();
    record.minimum_sm = Some("sm_61".into());
    record.ptx_result = recipe.rust_value.into();
    record.targets = "all".into();
    record.lowering = "generated_dotprod".into();
    record.backend_lowerings = vec![
        crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LlvmNvptx,
            mechanism: BackendLoweringMechanism::TypedNvvm,
            evidence_profile: "llvm-test".into(),
            targets: None,
            minimum_ptx: None,
            minimum_sm: None,
        },
        crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LibNvvm,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: "libnvvm-test".into(),
            targets: None,
            minimum_ptx: None,
            minimum_sm: Some("sm_75".into()),
        },
    ];
    record.dot_product = Some(crate::model::DotProduct {
        operation,
        signedness,
        adapter: recipe.adapter,
    });
    record.expected_ptx = InstructionPattern::new(
        recipe.ptx_mnemonic,
        recipe.ptx_modifiers,
        vec![OperandPattern::Register; 4],
    );
    record.summary = "packed integer dot product".into();
    record
}

pub(super) fn dot_product_declaration(
    operation: DotProductOperation,
    signedness: DotProductSignedness,
) -> ImportedIntrinsic {
    let recipe = dot_product_recipe(operation, signedness);
    let selection = |source_record: &str, half: Option<(&str, i64)>| ImportedSelection {
        source_record: source_record.into(),
        asm: format!(
            "{}.{} $dst, $a, $b, $c;",
            recipe.ptx_mnemonic,
            match half {
                Some((name, _)) => {
                    let types = &recipe.ptx_modifiers[1..];
                    format!("{name}.{}", types.join("."))
                }
                None => recipe.ptx_modifiers.join("."),
            }
        ),
        predicates: vec!["hasDotInstructions".into()],
        constraints: crate::model::ImportedSelectionConstraints {
            address_space: None,
            immediate_bindings: half
                .map(|(_, value)| {
                    vec![crate::model::ImportedImmediateBinding {
                        argument_index: 2,
                        value,
                    }]
                })
                .unwrap_or_default(),
        },
    };
    let selections = match operation {
        DotProductOperation::Dp4a => vec![selection("DOT4", None)],
        DotProductOperation::Dp2a => vec![
            selection("DOT2_hi", Some(("hi", -1))),
            selection("DOT2_lo", Some(("lo", 0))),
        ],
    };
    ImportedIntrinsic {
        source_record: recipe.source_record.into(),
        llvm_name: recipe.llvm_symbol.into(),
        arguments: recipe
            .llvm_arguments
            .iter()
            .map(|argument| (*argument).into())
            .collect(),
        results: vec!["i32".into()],
        classes: vec!["NVVMPureIntrinsic".into()],
        properties: recipe
            .llvm_properties
            .iter()
            .map(|property| (*property).into())
            .collect(),
        selections,
    }
}

pub(super) fn dot_product_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
    let mut record = evidence();
    record.id = policy.id.clone();
    record.source_record = policy.source_record.clone();
    record.llvm_symbol = policy.llvm_symbol.clone();
    record.llvm_arguments = policy.llvm_arguments.clone();
    record.llvm_results = policy.llvm_results.clone();
    record.concrete_llvm_arguments = policy.llvm_arguments.clone();
    record.concrete_llvm_results = policy.llvm_results.clone();
    record.declaration_attributes_canonicalized = Some(true);
    record.gpu_target = "sm_61".into();
    record.ptx_feature = "+ptx50".into();
    record.expected_ptx = policy.expected_ptx.clone();
    record
}

pub(super) fn validate_ptx_native_policy(policy: &OverlayIntrinsic) -> Result<()> {
    let source = resolve_policy_source(policy)?;
    validate_policy(policy, &source, None, 1)
}

pub(super) fn ledger_entry(record: &OverlayIntrinsic) -> AbiLedgerEntry {
    AbiLedgerEntry {
        abi_id: record.abi_id.clone(),
        status: "active".into(),
        catalog_id: record.id.clone(),
        operation_key: record.operation_key.clone(),
        raw_rust_signature: raw_rust_signature(record),
    }
}

pub(super) fn ledger(entries: Vec<AbiLedgerEntry>) -> AbiLedgerFile {
    AbiLedgerFile {
        schema: 1,
        intrinsic_abi: 1,
        entries,
    }
}

pub(super) fn test_f8f6f4_admission() -> SparseMmaF8F6F4Admission {
    let formats = vec![
        SparseMmaElement::E2m1,
        SparseMmaElement::E2m3,
        SparseMmaElement::E3m2,
        SparseMmaElement::E4m3,
        SparseMmaElement::E5m2,
    ];
    SparseMmaF8F6F4Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        a_elements: formats.clone(),
        b_elements: formats,
        product_count: 25,
    }
}

pub(super) fn test_sparse_mma_fp8_f32_admission() -> SparseMmaFp8F32Admission {
    SparseMmaFp8F32Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        a_elements: SPARSE_MMA_FP8_F32_ELEMENTS.into(),
        b_elements: SPARSE_MMA_FP8_F32_ELEMENTS.into(),
        product_count: 4,
    }
}

pub(super) fn test_sparse_mma_f8f6f4_f16_admission() -> SparseMmaF8F6F4F16Admission {
    SparseMmaF8F6F4F16Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        _legacy_first_abi_id: Some("i0525".into()),
        a_elements: SPARSE_MMA_F8F6F4_ELEMENTS.into(),
        b_elements: SPARSE_MMA_F8F6F4_ELEMENTS.into(),
        product_count: 25,
    }
}

pub(super) fn test_register_mma_f8f6f4_admission(
    _accumulator: RegisterMmaAccumulator,
) -> RegisterMmaF8F6F4Admission {
    let formats = REGISTER_MMA_F8F6F4_ELEMENTS.to_vec();
    RegisterMmaF8F6F4Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        _legacy_first_abi_id: None,
        a_elements: formats.clone(),
        b_elements: formats,
        product_count: 25,
        targets: ["sm_120a", "sm_120f", "sm_121a", "sm_121f"]
            .map(Into::into)
            .into(),
    }
}

pub(super) fn test_register_mma_fp8_admission() -> RegisterMmaFp8Admission {
    RegisterMmaFp8Admission {
        llvm_evidence_profile: "llvm-fp8-test".into(),
        libnvvm_evidence_profile: "libnvvm-fp8-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        _legacy_first_abi_id: Some("i0504".into()),
        shapes: REGISTER_MMA_FP8_SHAPES.into(),
        accumulators: REGISTER_MMA_FP8_ACCUMULATORS.into(),
        a_elements: REGISTER_MMA_FP8_ELEMENTS.into(),
        b_elements: REGISTER_MMA_FP8_ELEMENTS.into(),
        product_count: 16,
    }
}

pub(super) fn test_register_mma_ampere_float_admission() -> RegisterMmaAmpereFloatAdmission {
    RegisterMmaAmpereFloatAdmission {
        llvm_evidence_profile: "llvm-ampere-float-test".into(),
        libnvvm_evidence_profile: "libnvvm-ampere-float-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        _legacy_first_abi_id: Some("i0520".into()),
        product_count: 5,
        variants: REGISTER_MMA_AMPERE_FLOAT_VARIANTS.into(),
    }
}

pub(super) fn test_prmt_admission() -> PrmtAdmission {
    let variants = [
        PrmtMode::Generic,
        PrmtMode::F4e,
        PrmtMode::B4e,
        PrmtMode::Rc8,
        PrmtMode::Ecl,
        PrmtMode::Ecr,
        PrmtMode::Rc16,
    ]
    .map(|mode| crate::model::PrmtAdmissionVariant {
        abi_id: prmt_recipe(mode).abi_id.into(),
        mode,
    })
    .into();
    PrmtAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants,
    }
}

pub(super) fn test_fp8_conversion_admission() -> PackedConversionFp8Admission {
    PackedConversionFp8Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        destination_formats: vec![
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionDestinationFormat::E5m2x2,
        ],
        saturations: vec![
            PackedConversionSaturation::Satfinite,
            PackedConversionSaturation::SatfiniteRelu,
        ],
        product_count: 4,
    }
}

pub(super) fn test_fp8_f16x2_conversion_admission() -> PackedConversionFp8F16x2Admission {
    PackedConversionFp8F16x2Admission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        fp8_formats: vec![
            PackedConversionFp8Format::E4m3x2,
            PackedConversionFp8Format::E5m2x2,
        ],
        directions: vec![
            PackedConversionFp8Direction::Pack,
            PackedConversionFp8Direction::Unpack,
        ],
        relu_variants: true,
        product_count: 8,
    }
}

pub(super) fn test_cluster_barrier_admission() -> ClusterBarrierAdmission {
    let variants = [
        ClusterBarrierMode::Arrive,
        ClusterBarrierMode::ArriveAligned,
        ClusterBarrierMode::ArriveRelaxed,
        ClusterBarrierMode::ArriveRelaxedAligned,
        ClusterBarrierMode::Wait,
        ClusterBarrierMode::WaitAligned,
    ]
    .map(|mode| crate::model::ClusterBarrierAdmissionVariant {
        abi_id: cluster_barrier_recipe(mode).abi_id.into(),
        mode,
    })
    .into();
    ClusterBarrierAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants,
    }
}

pub(super) fn test_wgmma_control_admission() -> WgmmaControlAdmission {
    let variants = [
        WgmmaControlMode::Fence,
        WgmmaControlMode::CommitGroup,
        WgmmaControlMode::WaitGroup,
    ]
    .map(|mode| crate::model::WgmmaControlAdmissionVariant {
        abi_id: wgmma_control_recipe(mode).abi_id.into(),
        mode,
    })
    .into();
    WgmmaControlAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants,
    }
}

pub(super) fn test_special_register_admission() -> SpecialRegisterAdmission {
    SpecialRegisterAdmission {
        llvm_evidence_profile: "rust-llvm-23.1.0-16696adc".into(),
        libnvvm_evidence_profile: "cuda-13.3-libnvvm-13.3.33".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        registers: REVIEWED_SPECIAL_REGISTERS.into(),
        product_count: REVIEWED_SPECIAL_REGISTERS.len(),
    }
}

pub(super) fn test_debug_control_admission() -> DebugControlAdmission {
    DebugControlAdmission {
        llvm_evidence_profile: "llvm-debug-test".into(),
        libnvvm_evidence_profile: "libnvvm-debug-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        operations: vec![
            DebugControlOperation::Trap,
            DebugControlOperation::Breakpoint,
            DebugControlOperation::Pmevent,
        ],
        abi_ids: vec!["i9001".into(), "i9002".into(), "i9003".into()],
    }
}

pub(super) fn test_clc_admission() -> ClcAdmission {
    let operations = [
        ClcOperation::TryCancel,
        ClcOperation::TryCancelMulticast,
        ClcOperation::QueryIsCanceled,
        ClcOperation::QueryGetFirstCtaidX,
        ClcOperation::QueryGetFirstCtaidY,
        ClcOperation::QueryGetFirstCtaidZ,
    ];
    ClcAdmission {
        llvm_evidence_profile: "llvm-clc-test".into(),
        libnvvm_evidence_profile: "libnvvm-clc-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: operations
            .into_iter()
            .map(|operation| crate::model::ClcAdmissionVariant {
                abi_id: clc_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
    }
}

pub(super) fn test_tma_admission() -> TmaAdmission {
    TmaAdmission {
        llvm_evidence_profile: "llvm-tma-test".into(),
        libnvvm_evidence_profile: "libnvvm-tma-test".into(),
        reduce_llvm_evidence_profile: Some("llvm-tma-reduce-test".into()),
        reduce_libnvvm_evidence_profile: Some("libnvvm-tma-reduce-test".into()),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: TMA_OPERATIONS
            .into_iter()
            .map(|operation| crate::model::TmaAdmissionVariant {
                abi_id: tma_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
        reduce_variants: tma_reduction_admission_variants(),
    }
}

pub(super) fn test_tcgen05_admission() -> Tcgen05Admission {
    let operations = [
        Tcgen05Operation::Alloc,
        Tcgen05Operation::Dealloc,
        Tcgen05Operation::RelinquishAllocPermit,
        Tcgen05Operation::FenceBeforeThreadSync,
        Tcgen05Operation::FenceAfterThreadSync,
        Tcgen05Operation::Commit,
        Tcgen05Operation::CommitSharedCluster,
        Tcgen05Operation::MmaWsF16,
        Tcgen05Operation::MmaF16,
        Tcgen05Operation::MmaWsBf16,
        Tcgen05Operation::MmaWsTf32,
        Tcgen05Operation::CpSmemToTmem,
        Tcgen05Operation::Ld16x256bX8Pure,
        Tcgen05Operation::Ld16x256bPure,
        Tcgen05Operation::LoadWait,
        Tcgen05Operation::StoreWait,
        Tcgen05Operation::AllocCg2,
        Tcgen05Operation::DeallocCg2,
        Tcgen05Operation::RelinquishAllocPermitCg2,
        Tcgen05Operation::MmaF16Cg2,
        Tcgen05Operation::CommitCg2,
        Tcgen05Operation::CommitSharedClusterCg2,
        Tcgen05Operation::CommitMulticastCg2,
        Tcgen05Operation::CpSmemToTmemCg2,
        Tcgen05Operation::CommitMulticast,
        Tcgen05Operation::ShiftDown,
        Tcgen05Operation::ShiftDownCg2,
    ];
    Tcgen05Admission {
        llvm_evidence_profile: "llvm-tcgen05-test".into(),
        libnvvm_evidence_profile: "libnvvm-tcgen05-test".into(),
        cp_llvm_evidence_profile: None,
        cp_libnvvm_evidence_profile: None,
        ld_llvm_evidence_profile: None,
        ld_libnvvm_evidence_profile: None,
        st_llvm_evidence_profile: None,
        st_libnvvm_evidence_profile: None,
        offset_llvm_evidence_profile: None,
        offset_libnvvm_evidence_profile: None,
        control_llvm_evidence_profile: Some("llvm-tcgen05-control-test".into()),
        control_libnvvm_evidence_profile: Some("libnvvm-tcgen05-control-test".into()),
        mma_llvm_evidence_profile: None,
        mma_libnvvm_evidence_profile: None,
        mma_llvm_target_contracts: vec![],
        mma_libnvvm_target_contracts: vec![],
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: operations
            .into_iter()
            .map(|operation| crate::model::Tcgen05AdmissionVariant {
                abi_id: tcgen05_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
        cp_variants: vec![],
        ld_variants: vec![],
        st_variants: vec![],
        ld_offset_variants: vec![],
        st_offset_variants: vec![],
        mma_variants: vec![],
    }
}

pub(super) fn without_tcgen05_control(mut admission: Tcgen05Admission) -> Tcgen05Admission {
    admission.variants.truncate(24);
    admission.control_llvm_evidence_profile = None;
    admission.control_libnvvm_evidence_profile = None;
    admission
}

pub(super) fn test_tcgen05_cp_admission() -> Tcgen05Admission {
    let mut admission = test_tcgen05_admission();
    admission.cp_llvm_evidence_profile = Some("llvm-tcgen05-cp-test".into());
    admission.cp_libnvvm_evidence_profile = Some("libnvvm-tcgen05-cp-test".into());
    admission.cp_variants = TCGEN05_CP_MEMBERS
        .into_iter()
        .flat_map(|member| {
            [Tcgen05CpGroup::Cg1, Tcgen05CpGroup::Cg2]
                .into_iter()
                .map(move |group| (member, group))
        })
        .enumerate()
        .map(|(index, (member, group))| Tcgen05CpAdmissionVariant {
            abi_id: format!("i{:04}", 578 + index),
            member,
            group,
        })
        .collect();
    admission
}

pub(super) fn test_tcgen05_ld_admission() -> Tcgen05Admission {
    let mut admission = test_tcgen05_cp_admission();
    admission.ld_llvm_evidence_profile = Some("llvm-tcgen05-ld-test".into());
    admission.ld_libnvvm_evidence_profile = Some("libnvvm-tcgen05-ld-test".into());
    admission.ld_variants = TCGEN05_LD_VARIANTS
        .into_iter()
        .flat_map(|(shape, multiplicity)| {
            [false, true]
                .into_iter()
                .map(move |pack16| (shape, multiplicity, pack16))
        })
        .enumerate()
        .map(
            |(index, (shape, multiplicity, pack16))| Tcgen05LdAdmissionVariant {
                abi_id: format!("i{:04}", 612 + index),
                shape,
                multiplicity,
                pack16,
            },
        )
        .collect();
    admission
}

pub(super) fn test_tcgen05_st_admission() -> Tcgen05Admission {
    let mut admission = test_tcgen05_ld_admission();
    admission.st_llvm_evidence_profile = Some("llvm-tcgen05-st-test".into());
    admission.st_libnvvm_evidence_profile = Some("libnvvm-tcgen05-st-test".into());
    admission.st_variants = TCGEN05_ST_VARIANTS
        .into_iter()
        .flat_map(|(shape, multiplicity)| {
            [false, true]
                .into_iter()
                .map(move |unpack16| (shape, multiplicity, unpack16))
        })
        .enumerate()
        .map(
            |(index, (shape, multiplicity, unpack16))| Tcgen05StAdmissionVariant {
                abi_id: format!("i{:04}", 670 + index),
                shape,
                multiplicity,
                unpack16,
            },
        )
        .collect();
    admission
}

pub(super) fn test_tcgen05_offset_admission() -> Tcgen05Admission {
    let mut admission = test_tcgen05_st_admission();
    admission.offset_llvm_evidence_profile = Some("llvm-tcgen05-offset-test".into());
    admission.offset_libnvvm_evidence_profile = Some("libnvvm-tcgen05-offset-test".into());
    admission.ld_offset_variants = TCGEN05_OFFSET_LDST_VARIANTS
        .into_iter()
        .flat_map(|(shape, multiplicity)| {
            [false, true]
                .into_iter()
                .map(move |pack16| (shape, multiplicity, pack16))
        })
        .enumerate()
        .map(
            |(index, (shape, multiplicity, pack16))| Tcgen05LdAdmissionVariant {
                abi_id: format!("i{:04}", 728 + index),
                shape,
                multiplicity,
                pack16,
            },
        )
        .collect();
    admission.st_offset_variants = TCGEN05_OFFSET_LDST_VARIANTS
        .into_iter()
        .flat_map(|(shape, multiplicity)| {
            [false, true]
                .into_iter()
                .map(move |unpack16| (shape, multiplicity, unpack16))
        })
        .enumerate()
        .map(
            |(index, (shape, multiplicity, unpack16))| Tcgen05StAdmissionVariant {
                abi_id: format!("i{:04}", 744 + index),
                shape,
                multiplicity,
                unpack16,
            },
        )
        .collect();
    admission
}

pub(super) fn test_tcgen05_mma_admission() -> Tcgen05Admission {
    let mut admission = test_tcgen05_admission();
    admission.mma_llvm_evidence_profile = Some("llvm-tcgen05-mma-test".into());
    admission.mma_libnvvm_evidence_profile = Some("libnvvm-tcgen05-mma-test".into());
    admission.mma_llvm_target_contracts =
        expected_tcgen05_mma_target_contracts(IntrinsicBackend::LlvmNvptx);
    admission.mma_libnvvm_target_contracts =
        expected_tcgen05_mma_target_contracts(IntrinsicBackend::LibNvvm);
    admission.mma_variants = TCGEN05_MMA_FORMS
        .into_iter()
        .map(|form| (form, None))
        .chain(
            TCGEN05_MMA_ALIASES
                .into_iter()
                .map(|alias| (Tcgen05MmaForm::WsTensor, Some(alias))),
        )
        .enumerate()
        .map(|(index, (form, alias))| Tcgen05MmaAdmissionVariant {
            abi_id: format!("i{:04}", 763 + index),
            form,
            alias,
        })
        .chain(
            TCGEN05_MMA_ALIASES
                .into_iter()
                .enumerate()
                .map(|(index, alias)| Tcgen05MmaAdmissionVariant {
                    abi_id: format!("i{:04}", 1011 + index),
                    form: Tcgen05MmaForm::Shared,
                    alias: Some(alias),
                }),
        )
        .collect();
    admission
}

pub(super) fn assert_tcgen05_backend_target_split(record: &OverlayIntrinsic) {
    let llvm = &record.backend_lowerings[0];
    let libnvvm = &record.backend_lowerings[1];
    assert_eq!(record.targets, TCGEN05_LLVM_TARGETS);
    assert_eq!(llvm.backend, IntrinsicBackend::LlvmNvptx);
    assert_eq!(llvm.targets, None);
    assert_eq!(libnvvm.backend, IntrinsicBackend::LibNvvm);
    assert_eq!(libnvvm.targets.as_deref(), Some(TCGEN05_LIBNVVM_TARGETS));
    assert_eq!(
        backend_target_requirement(record, llvm).unwrap().hardware,
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
            ],
        }
    );
    assert_eq!(
        backend_target_requirement(record, libnvvm)
            .unwrap()
            .hardware,
        CatalogHardwareTarget::AnyOf {
            alternatives: vec![
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
            ],
        }
    );
}

pub(super) fn test_threadfence_admission() -> ThreadfenceAdmission {
    ThreadfenceAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: vec![
            crate::model::ThreadfenceAdmissionVariant {
                abi_id: "i0298".into(),
                scope: ThreadfenceScope::Cta,
            },
            crate::model::ThreadfenceAdmissionVariant {
                abi_id: "i0299".into(),
                scope: ThreadfenceScope::Device,
            },
            crate::model::ThreadfenceAdmissionVariant {
                abi_id: "i0300".into(),
                scope: ThreadfenceScope::System,
            },
        ],
    }
}

pub(super) fn test_cluster_memory_admission() -> ClusterMemoryAdmission {
    ClusterMemoryAdmission {
        llvm_evidence_profile: "llvm-cluster-memory-test".into(),
        libnvvm_evidence_profile: "libnvvm-cluster-memory-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: vec![
            crate::model::ClusterMemoryAdmissionVariant {
                abi_id: "i0320".into(),
                operation: ClusterMemoryOperation::MapSharedRank,
            },
            crate::model::ClusterMemoryAdmissionVariant {
                abi_id: "i0321".into(),
                operation: ClusterMemoryOperation::ReadU32,
            },
        ],
    }
}

pub(super) fn test_stmatrix_admission() -> StmatrixAdmission {
    let variants = [
        (StmatrixMultiplicity::X2, StmatrixLayout::Normal, "i0301"),
        (
            StmatrixMultiplicity::X2,
            StmatrixLayout::Transposed,
            "i0302",
        ),
        (StmatrixMultiplicity::X4, StmatrixLayout::Normal, "i0303"),
        (
            StmatrixMultiplicity::X4,
            StmatrixLayout::Transposed,
            "i0304",
        ),
    ]
    .map(
        |(multiplicity, layout, abi_id)| crate::model::StmatrixAdmissionVariant {
            abi_id: abi_id.into(),
            multiplicity,
            layout,
        },
    )
    .into();
    StmatrixAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants,
    }
}

pub(super) fn test_mbarrier_extended_admission() -> MbarrierExtendedAdmission {
    let variants = [
        MbarrierExtendedOperation::ArriveExpectTxCta,
        MbarrierExtendedOperation::ArriveExpectTxCluster,
        MbarrierExtendedOperation::ArriveRemoteCluster,
        MbarrierExtendedOperation::TryWaitTokenCta,
        MbarrierExtendedOperation::TryWaitParityCta,
        MbarrierExtendedOperation::TryWaitParityCluster,
        MbarrierExtendedOperation::FenceProxyAsyncSharedCta,
        MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster,
        MbarrierExtendedOperation::Nanosleep,
    ]
    .map(|operation| crate::model::MbarrierExtendedAdmissionVariant {
        abi_id: mbarrier_extended_recipe(operation).abi_id.into(),
        operation,
    })
    .into();
    MbarrierExtendedAdmission {
        llvm_evidence_profile: "llvm-test".into(),
        libnvvm_evidence_profile: "libnvvm-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants,
    }
}

pub(super) struct CandidateTestRepo(pub(super) PathBuf);

impl Drop for CandidateTestRepo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) fn repo_without_evidence() -> CandidateTestRepo {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = std::env::temp_dir().join(format!(
        "cuda-intrinsics-candidate-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let input = root.join("intrinsics");
    fs::create_dir_all(input.join("overlay")).unwrap();
    for name in [
        "upstream.lock",
        "imported.json",
        "overlay.toml",
        "abi-v1.toml",
    ] {
        fs::copy(source.join("intrinsics").join(name), input.join(name)).unwrap();
    }
    for entry in fs::read_dir(source.join("intrinsics/overlay")).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().and_then(|value| value.to_str()) == Some("toml") {
            fs::copy(entry.path(), input.join("overlay").join(entry.file_name())).unwrap();
        }
    }
    CandidateTestRepo(root)
}
