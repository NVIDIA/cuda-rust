/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::model::ImportedSelectionConstraints;
use crate::ptx::{InstructionPattern, OperandPattern};

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogHardwareTarget, CatalogSelection,
    WgmmaControlAdapter, WgmmaControlMode, WgmmaControlParticipation,
};
use std::path::Path;

mod mma;
mod scalar_misc;
mod tensor;
mod warp_memory;

fn catalog_with_debug_controls() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    assert_eq!(debug_controls(&catalog).count(), 3);
    catalog
}

fn catalog_with_stmatrix() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    assert_eq!(stmatrices(&catalog).count(), 4);
    catalog
}

fn catalog_with_clc() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::test_catalog_with_clc(&repo_root).unwrap();
    assert_eq!(clc_intrinsics(&catalog).count(), 6);
    catalog
}

fn catalog_with_tma() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::test_catalog_with_tma(&repo_root).unwrap();
    assert_eq!(tma_intrinsics(&catalog).count(), 111);
    catalog
}

fn catalog_with_wgmma_controls() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut catalog = crate::resolve::resolve(&repo_root).unwrap();
    if wgmma_controls(&catalog).count() == 3 {
        return catalog;
    }
    let template = catalog
        .intrinsics
        .iter()
        .find(|record| record.cluster_barrier.is_some())
        .expect("cluster-barrier template")
        .clone();

    let recipes = [
        (
            WgmmaControlMode::Fence,
            "i0317",
            "wgmma_fence",
            "WgmmaFenceSyncAlignedOp",
            "nvvm.wgmma_fence_sync_aligned",
            "llvm.nvvm.wgmma.fence.sync.aligned",
            "WGMMA_FENCE_SYNC_ALIGNED",
            "fence.sync.aligned",
            WgmmaControlAdapter::NoArguments,
            "cuda_device::wgmma::wgmma_fence",
        ),
        (
            WgmmaControlMode::CommitGroup,
            "i0318",
            "wgmma_commit_group",
            "WgmmaCommitGroupSyncAlignedOp",
            "nvvm.wgmma_commit_group_sync_aligned",
            "llvm.nvvm.wgmma.commit_group.sync.aligned",
            "WGMMA_COMMIT_GROUP_SYNC_ALIGNED",
            "commit_group.sync.aligned",
            WgmmaControlAdapter::NoArguments,
            "cuda_device::wgmma::wgmma_commit_group",
        ),
        (
            WgmmaControlMode::WaitGroup,
            "i0319",
            "wgmma_wait_group",
            "WgmmaWaitGroupSyncAlignedOp",
            "nvvm.wgmma_wait_group_sync_aligned",
            "llvm.nvvm.wgmma.wait_group.sync.aligned",
            "WGMMA_WAIT_GROUP_SYNC_ALIGNED",
            "wait_group.sync.aligned",
            WgmmaControlAdapter::ConstGenericU32ToI64Immediate,
            "cuda_device::wgmma::__wgmma_wait_group",
        ),
    ];

    for (
        mode,
        abi_id,
        id,
        op_type,
        op_name,
        llvm_symbol,
        selection_record,
        suffix,
        adapter,
        compatibility_path,
    ) in recipes
    {
        let wait = mode == WgmmaControlMode::WaitGroup;
        let mut record = template.clone();
        record.id = id.into();
        record.operation_key = format!("wgmma.control.{suffix}");
        record.family = "wgmma_control".into();
        record.source = crate::model::IntrinsicSource::LlvmImported {
            source_record: format!("int_nvvm_wgmma_{}", suffix.replace('.', "_")),
        };
        record.selections = vec![CatalogSelection {
            source_record: selection_record.into(),
            asm: format!("wgmma.{suffix};"),
            predicates: vec!["Subtarget->getPTXVersion() >= 80".into(), "hasSM90a".into()],
            constraints: ImportedSelectionConstraints::default(),
        }];
        record.rust.abi_id = abi_id.into();
        record.rust.module = "wgmma".into();
        record.rust.name = id.into();
        record.rust.arguments = if wait { vec!["u64".into()] } else { vec![] };
        record.rust.result = "()".into();
        record.rust.safe = false;
        record.rust.must_use = false;
        record.rust.safe_allowlist_reason = None;
        record.rust.canonical_path =
            format!("cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::{abi_id}");
        record.rust.public_path = format!("cuda_intrinsics::wgmma::{id}");
        record.rust.compatibility_paths = vec![compatibility_path.into()];
        record.dialect.op_type = op_type.into();
        record.dialect.op_name = op_name.into();
        record.dialect.operands = if wait { vec!["i64".into()] } else { vec![] };
        record.dialect.results.clear();
        let llvm = record.llvm.as_mut().unwrap();
        llvm.symbol = llvm_symbol.into();
        llvm.resolved_symbol = None;
        llvm.arguments = if wait { vec!["i64".into()] } else { vec![] };
        llvm.results.clear();
        llvm.properties = if wait {
            vec!["ImmArg<arg0>".into(), "IntrConvergent".into()]
        } else {
            vec!["IntrConvergent".into()]
        };
        record.semantics.pure = false;
        record.semantics.memory = "read_write".into();
        record.semantics.convergent = true;
        record.semantics.execution_scope = "warpgroup".into();
        record.target.minimum_ptx = "8.0".parse().unwrap();
        record.target.hardware = CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 90 }],
        };
        record.target.ptx_result = "()".into();
        record.target.targets = "sm_90a".into();
        record.target.ptx_isa_version = "9.3".into();
        record.target.ptx_isa_section = "WGMMA control".into();
        record.target.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions".into();
        for lowering in &mut record.backend_lowerings {
            lowering.target.minimum_ptx = "8.0".parse().unwrap();
            lowering.target.hardware = record.target.hardware.clone();
            lowering.mechanism = match lowering.backend {
                IntrinsicBackend::LlvmNvptx => BackendLoweringMechanism::TypedNvvm,
                IntrinsicBackend::LibNvvm => BackendLoweringMechanism::InlinePtx,
            };
        }
        record.packed_atomic = None;
        record.redux = None;
        record.vote = None;
        record.active_mask = None;
        record.warp_match = None;
        record.warp_barrier = None;
        record.warp_shuffle = None;
        record.dot_product = None;
        record.packed_alu = None;
        record.packed_conversion = None;
        record.cp_async_copy = None;
        record.cp_async_control = None;
        record.cp_async_mbarrier = None;
        record.mbarrier_basic = None;
        record.register_mma = None;
        record.sparse_mma = None;
        record.prmt = None;
        record.cluster_barrier = None;
        record.wgmma_control = Some(crate::model::WgmmaControl {
            mode,
            adapter,
            participation: WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction,
        });
        record.special_register = None;
        record.ldmatrix = None;
        record.lowering = "generated_wgmma_control".into();
        record.expected_ptx = InstructionPattern {
            mnemonic: "wgmma".into(),
            modifiers: suffix.split('.').map(str::to_owned).collect(),
            operands: if wait {
                vec![OperandPattern::Immediate]
            } else {
                vec![]
            },
        };
        record.summary = format!("Generated {id} test record.");
        catalog.intrinsics.push(record);
    }
    catalog
}

fn raw_abi_item<'a>(raw: &'a str, id: &str) -> &'a str {
    let marker = format!("Catalog ID: `{id}`");
    let marker_offset = raw.find(&marker).unwrap();
    let start = raw[..marker_offset]
        .rfind("\n\n")
        .map_or(0, |offset| offset + 2);
    let end = raw[marker_offset..]
        .find("\n}\n")
        .map_or(raw.len(), |offset| marker_offset + offset + 3);
    &raw[start..end]
}

fn raw_abi_safety_block<'a>(raw: &'a str, id: &str) -> &'a str {
    let item = raw_abi_item(raw, id);
    let marker = "/// # Safety\n";
    let start = item.find(marker).unwrap() + marker.len();
    let end = item[start..].find("#[").unwrap() + start;
    &item[start..end]
}

fn catalog_with_tcgen05() -> CatalogFile {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::test_catalog_with_tcgen05(&repo_root).unwrap();
    assert_eq!(tcgen05_intrinsics(&catalog).count(), 233);
    catalog
}
