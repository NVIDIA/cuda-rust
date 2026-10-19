/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation, TargetSelectorBinding, Tcgen05, Tcgen05Admission,
    Tcgen05CpGroup, Tcgen05MmaBUsage, Tcgen05MmaForm, Tcgen05Operation, Tcgen05SourceContract,
};
use crate::ptx::InstructionPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use super::*;
use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;
use crate::resolve::targets::*;

pub(in crate::resolve) fn expand_tcgen05_admission(
    admission: &Tcgen05Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "tcgen05 runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact tcgen05 admission requires both backend evidence profiles"
    );
    let expected = [
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
    let has_control_variants = admission.variants.iter().any(|variant| {
        matches!(
            variant.operation,
            Tcgen05Operation::CommitMulticast
                | Tcgen05Operation::ShiftDown
                | Tcgen05Operation::ShiftDownCg2
        )
    });
    if has_control_variants {
        ensure!(
            admission
                .control_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .control_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 control admission requires both backend evidence profiles"
        );
    }
    let expected = if has_control_variants {
        &expected[..]
    } else {
        &expected[..24]
    };
    ensure!(
        admission
            .variants
            .iter()
            .map(|variant| variant.operation)
            .eq(expected.iter().copied()),
        "compact tcgen05 admission must list all 24 base operations or all 27 current operations in canonical order"
    );

    let mut records = admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = tcgen05_recipe(variant.operation);
            let control = matches!(
                variant.operation,
                Tcgen05Operation::CommitMulticast
                    | Tcgen05Operation::ShiftDown
                    | Tcgen05Operation::ShiftDownCg2
            );
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "tcgen05".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "tcgen05".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: recipe.safe,
                must_use: false,
                safe_allowlist_reason: recipe.safe_reason.map(Into::into),
                public_rust_path: format!("cuda_intrinsics::tcgen05::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::tcgen05::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: recipe
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: recipe
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: recipe.memory.into(),
                convergent: true,
                execution_scope: recipe.operation.execution_scope().into(),
                minimum_ptx: "8.6".into(),
                minimum_sm: None,
                ptx_result: recipe.rust_result.into(),
                targets: TCGEN05_LLVM_TARGETS.into(),
                ptx_isa_version: "8.6".into(),
                ptx_isa_section: "Tensor Memory tcgen05 instructions".into(),
                ptx_isa_url:
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory".into(),
                lowering: "generated_tcgen05".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: if control {
                            admission
                                .control_llvm_evidence_profile
                                .as_ref()
                                .expect("validated tcgen05 control LLVM evidence profile")
                                .clone()
                        } else {
                            admission.llvm_evidence_profile.clone()
                        },
                        targets: None,
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: None,
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: if control {
                            admission
                                .control_libnvvm_evidence_profile
                                .as_ref()
                                .expect("validated tcgen05 control libNVVM evidence profile")
                                .clone()
                        } else {
                            admission.libnvvm_evidence_profile.clone()
                        },
                        targets: Some(TCGEN05_LIBNVVM_TARGETS.into()),
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: None,
                    },
                ],
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
                tcgen05: Some(Tcgen05 {
                    operation: recipe.operation,
                    cp: None,
                    ld: None,
                    st: None,
                    mma: None,
                    adapter: recipe.adapter,
                    source_contract: recipe.source_contract,
                    runtime_validation: admission.runtime_validation,
                }),
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "tcgen05".into(),
                    modifiers: recipe.modifiers,
                    operands: recipe.operands,
                },
                summary: recipe.summary.into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if admission.cp_variants.is_empty() {
        ensure!(
            admission.cp_llvm_evidence_profile.is_none()
                && admission.cp_libnvvm_evidence_profile.is_none(),
            "tcgen05 copy evidence profiles require admitted copy variants"
        );
    } else {
        ensure!(
            [
                Tcgen05MmaBUsage::Discard,
                Tcgen05MmaBUsage::LastUse,
                Tcgen05MmaBUsage::Fill,
                Tcgen05MmaBUsage::Use,
            ]
            .map(Tcgen05MmaBUsage::selector_value)
                == [0, 1, 2, 3],
            "tcgen05 MMA B-usage selector mapping changed"
        );
        ensure!(
            admission
                .cp_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .cp_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 copy admission requires both backend evidence profiles"
        );

        use Tcgen05CpGroup::{Cg1, Cg2};
        let expected_cp = TCGEN05_CP_MEMBERS
            .into_iter()
            .flat_map(|member| [Cg1, Cg2].into_iter().map(move |group| (member, group)))
            .collect::<Vec<_>>();
        ensure!(
            admission
                .cp_variants
                .iter()
                .map(|variant| (variant.member, variant.group))
                .eq(expected_cp),
            "compact tcgen05 copy admission must list all 34 variants in canonical order"
        );
        for variant in &admission.cp_variants {
            validate_abi_id(&variant.abi_id)?;
            let base_id = if variant.group == Cg1 {
                "tcgen05_cp_smem_to_tmem"
            } else {
                "tcgen05_cp_smem_to_tmem_cg2"
            };
            let base = records
                .iter()
                .find(|record| record.id == base_id)
                .unwrap()
                .clone();
            records.push(materialize_tcgen05_cp_variant(&base, admission, variant));
        }
    }

    if admission.ld_variants.is_empty() {
        ensure!(
            admission.ld_llvm_evidence_profile.is_none()
                && admission.ld_libnvvm_evidence_profile.is_none(),
            "tcgen05 load evidence profiles require admitted load variants"
        );
    } else {
        ensure!(
            admission
                .ld_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .ld_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 load admission requires both backend evidence profiles"
        );
        let expected_ld = TCGEN05_LD_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |pack16| (shape, multiplicity, pack16))
            })
            .collect::<Vec<_>>();
        ensure!(
            admission
                .ld_variants
                .iter()
                .map(|variant| (variant.shape, variant.multiplicity, variant.pack16))
                .eq(expected_ld),
            "compact tcgen05 load admission must list all 58 variants in canonical order"
        );
        let base = records
            .iter()
            .find(|record| record.id == "tcgen05_ld_16x256b_pure")
            .expect("closed tcgen05 load base")
            .clone();
        let first_load = records.len();
        for variant in &admission.ld_variants {
            validate_abi_id(&variant.abi_id)?;
            records.push(materialize_tcgen05_ld_variant(&base, admission, variant));
        }
        for pair in records[first_load..].as_chunks::<2>().0 {
            let raw = pair[0].tcgen05.as_ref().and_then(|tcgen05| tcgen05.ld);
            let packed = pair[1].tcgen05.as_ref().and_then(|tcgen05| tcgen05.ld);
            ensure!(
                raw.is_some_and(|ld| !ld.pack16)
                    && packed.is_some_and(|ld| ld.pack16)
                    && raw.map(|mut ld| {
                        ld.pack16 = true;
                        ld
                    }) == packed
                    && pair[0].source_record == pair[1].source_record
                    && pair[0].llvm_symbol == pair[1].llvm_symbol,
                "tcgen05 load source sharing must pair raw and pack16 leaves"
            );
        }
    }

    if admission.st_variants.is_empty() {
        ensure!(
            admission.st_llvm_evidence_profile.is_none()
                && admission.st_libnvvm_evidence_profile.is_none(),
            "tcgen05 store evidence profiles require admitted store variants"
        );
    } else {
        ensure!(
            admission
                .st_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .st_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 store admission requires both backend evidence profiles"
        );
        let expected_st = TCGEN05_ST_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |unpack16| (shape, multiplicity, unpack16))
            })
            .collect::<Vec<_>>();
        ensure!(
            admission
                .st_variants
                .iter()
                .map(|variant| (variant.shape, variant.multiplicity, variant.unpack16))
                .eq(expected_st),
            "compact tcgen05 store admission must list all 58 variants in canonical order"
        );
        let base = records
            .iter()
            .find(|record| record.id == "tcgen05_ld_16x256b_pure")
            .expect("closed tcgen05 store base")
            .clone();
        let first_store = records.len();
        for variant in &admission.st_variants {
            validate_abi_id(&variant.abi_id)?;
            records.push(materialize_tcgen05_st_variant(&base, admission, variant));
        }
        for pair in records[first_store..].as_chunks::<2>().0 {
            let raw = pair[0].tcgen05.as_ref().and_then(|tcgen05| tcgen05.st);
            let unpacked = pair[1].tcgen05.as_ref().and_then(|tcgen05| tcgen05.st);
            ensure!(
                raw.is_some_and(|st| !st.unpack16)
                    && unpacked.is_some_and(|st| st.unpack16)
                    && raw.map(|mut st| {
                        st.unpack16 = true;
                        st
                    }) == unpacked
                    && pair[0].source_record == pair[1].source_record
                    && pair[0].llvm_symbol == pair[1].llvm_symbol,
                "tcgen05 store source sharing must pair raw and unpack16 leaves"
            );
        }
    }

    if admission.ld_offset_variants.is_empty() && admission.st_offset_variants.is_empty() {
        ensure!(
            admission.offset_llvm_evidence_profile.is_none()
                && admission.offset_libnvvm_evidence_profile.is_none(),
            "tcgen05 offset evidence profiles require admitted offset load/store variants"
        );
    } else {
        ensure!(
            admission
                .offset_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .offset_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 offset admission requires both backend evidence profiles"
        );
        let expected_ld = TCGEN05_OFFSET_LDST_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |pack16| (shape, multiplicity, pack16))
            })
            .collect::<Vec<_>>();
        ensure!(
            admission
                .ld_offset_variants
                .iter()
                .map(|variant| (variant.shape, variant.multiplicity, variant.pack16))
                .eq(expected_ld),
            "compact tcgen05 offset load admission must list all 16 variants in canonical order"
        );
        let expected_st = TCGEN05_OFFSET_LDST_VARIANTS
            .into_iter()
            .flat_map(|(shape, multiplicity)| {
                [false, true]
                    .into_iter()
                    .map(move |unpack16| (shape, multiplicity, unpack16))
            })
            .collect::<Vec<_>>();
        ensure!(
            admission
                .st_offset_variants
                .iter()
                .map(|variant| (variant.shape, variant.multiplicity, variant.unpack16))
                .eq(expected_st),
            "compact tcgen05 offset store admission must list all 16 variants in canonical order"
        );

        let base = records
            .iter()
            .find(|record| record.id == "tcgen05_ld_16x256b_pure")
            .expect("closed tcgen05 offset load/store base")
            .clone();
        let first_load = records.len();
        for variant in &admission.ld_offset_variants {
            validate_abi_id(&variant.abi_id)?;
            records.push(materialize_tcgen05_ld_variant(&base, admission, variant));
        }
        for pair in records[first_load..].as_chunks::<2>().0 {
            let raw = pair[0].tcgen05.as_ref().and_then(|tcgen05| tcgen05.ld);
            let packed = pair[1].tcgen05.as_ref().and_then(|tcgen05| tcgen05.ld);
            ensure!(
                raw.is_some_and(|ld| !ld.pack16)
                    && packed.is_some_and(|ld| ld.pack16)
                    && raw.map(|mut ld| {
                        ld.pack16 = true;
                        ld
                    }) == packed
                    && pair[0].source_record == pair[1].source_record
                    && pair[0].llvm_symbol == pair[1].llvm_symbol,
                "tcgen05 offset load source sharing must pair raw and pack16 leaves"
            );
        }

        let first_store = records.len();
        for variant in &admission.st_offset_variants {
            validate_abi_id(&variant.abi_id)?;
            records.push(materialize_tcgen05_st_variant(&base, admission, variant));
        }
        for pair in records[first_store..].as_chunks::<2>().0 {
            let raw = pair[0].tcgen05.as_ref().and_then(|tcgen05| tcgen05.st);
            let unpacked = pair[1].tcgen05.as_ref().and_then(|tcgen05| tcgen05.st);
            ensure!(
                raw.is_some_and(|st| !st.unpack16)
                    && unpacked.is_some_and(|st| st.unpack16)
                    && raw.map(|mut st| {
                        st.unpack16 = true;
                        st
                    }) == unpacked
                    && pair[0].source_record == pair[1].source_record
                    && pair[0].llvm_symbol == pair[1].llvm_symbol,
                "tcgen05 offset store source sharing must pair raw and unpack16 leaves"
            );
        }
    }

    if admission.mma_variants.is_empty() {
        ensure!(
            admission.mma_llvm_evidence_profile.is_none()
                && admission.mma_libnvvm_evidence_profile.is_none()
                && admission.mma_llvm_target_contracts.is_empty()
                && admission.mma_libnvvm_target_contracts.is_empty(),
            "tcgen05 MMA profiles and target contracts require admitted variants"
        );
    } else {
        ensure!(
            admission
                .mma_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .mma_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty()),
            "compact tcgen05 MMA admission requires both backend evidence profiles"
        );
        let expected_variants = TCGEN05_MMA_FORMS
            .into_iter()
            .map(|form| (form, None))
            .chain(
                TCGEN05_MMA_ALIASES
                    .into_iter()
                    .map(|alias| (Tcgen05MmaForm::WsTensor, Some(alias))),
            )
            .chain(
                TCGEN05_MMA_ALIASES
                    .into_iter()
                    .map(|alias| (Tcgen05MmaForm::Shared, Some(alias))),
            )
            .collect::<Vec<_>>();
        ensure!(
            admission
                .mma_variants
                .iter()
                .map(|variant| (variant.form, variant.alias))
                .eq(expected_variants),
            "compact tcgen05 MMA admission must list all 24 APIs in canonical order"
        );
        for variant in &admission.mma_variants {
            validate_abi_id(&variant.abi_id)?;
        }
        let expected_llvm = expected_tcgen05_mma_target_contracts(IntrinsicBackend::LlvmNvptx);
        let expected_libnvvm = expected_tcgen05_mma_target_contracts(IntrinsicBackend::LibNvvm);
        ensure!(
            admission.mma_llvm_target_contracts == expected_llvm
                && admission.mma_libnvvm_target_contracts == expected_libnvvm,
            "tcgen05 MMA target contracts changed from the reviewed backend matrices"
        );
        let llvm_target =
            resolve_target_contracts("tcgen05 MMA LLVM", &admission.mma_llvm_target_contracts)?;
        let libnvvm_target = resolve_target_contracts(
            "tcgen05 MMA libNVVM",
            &admission.mma_libnvvm_target_contracts,
        )?;
        let fixed_selector = [TargetSelectorBinding {
            name: "kind".into(),
            value: "f8f6f4".into(),
        }];
        let fixed_llvm = resolve_target_contract(
            "tcgen05 MMA LLVM alias",
            &fixed_selector,
            &admission.mma_llvm_target_contracts,
        )?;
        let fixed_libnvvm = resolve_target_contract(
            "tcgen05 MMA libNVVM alias",
            &fixed_selector,
            &admission.mma_libnvvm_target_contracts,
        )?;
        for variant in &admission.mma_variants {
            let (llvm, libnvvm) = if variant.alias.is_some() {
                (&fixed_llvm, &fixed_libnvvm)
            } else {
                (&llvm_target, &libnvvm_target)
            };
            records.push(materialize_tcgen05_mma_variant(
                admission, variant, llvm, libnvvm,
            ));
        }
    }
    Ok(records)
}

pub(in crate::resolve) fn validate_tcgen05_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let tcgen05 = policy
        .tcgen05
        .as_ref()
        .with_context(|| format!("{} has no closed tcgen05 contract", policy.id))?;
    if let Some(mma) = &tcgen05.mma {
        return validate_tcgen05_mma_policy(policy, declaration, tcgen05, mma);
    }
    if let Some(ld) = tcgen05.ld {
        return validate_tcgen05_ld_policy(policy, declaration, tcgen05, ld);
    }
    if let Some(st) = tcgen05.st {
        return validate_tcgen05_st_policy(policy, declaration, tcgen05, st);
    }
    if let Some(cp) = tcgen05.cp {
        return validate_tcgen05_cp_policy(policy, declaration, tcgen05, cp);
    }
    ensure!(
        !matches!(
            tcgen05.operation,
            Tcgen05Operation::Ld | Tcgen05Operation::St | Tcgen05Operation::Mma
        ),
        "{} has a tcgen05 load/store/MMA operation without its closed identity",
        policy.id
    );
    let recipe = tcgen05_recipe(tcgen05.operation);
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} tcgen05 identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && policy.safe == recipe.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.as_deref() == recipe.safe_reason
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::tcgen05::{}", recipe.id)],
        "{} tcgen05 Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && declaration.arguments == recipe.llvm_arguments
            && declaration.results == recipe.llvm_results
            && declaration.classes == recipe.imported_classes
            && declaration.properties == recipe.imported_properties
            && policy.lowering == "generated_tcgen05",
        "{} tcgen05 carrier or imported declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent
            && policy.execution_scope == recipe.operation.execution_scope()
            && tcgen05.ld.is_none()
            && tcgen05.st.is_none()
            && tcgen05.adapter == recipe.adapter
            && tcgen05.source_contract == recipe.source_contract
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LLVM_TARGETS
            && policy.ptx_isa_version == "8.6"
            && policy.ptx_result == recipe.rust_result
            && policy.expected_ptx.mnemonic == "tcgen05"
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == recipe.operands,
        "{} tcgen05 target or PTX contract changed",
        policy.id
    );
    validate_tcgen05_backend_routes(policy, "tcgen05")?;
    validate_tcgen05_source_contract(&recipe, declaration)?;
    ensure_no_other_family_contract(policy, "tcgen05")?;
    Ok(())
}

pub(in crate::resolve) fn validate_tcgen05_backend_routes(
    policy: &OverlayIntrinsic,
    family: &str,
) -> Result<()> {
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings[0].backend == IntrinsicBackend::LlvmNvptx
            && policy.backend_lowerings[1].backend == IntrinsicBackend::LibNvvm
            && policy.backend_lowerings[0].targets.is_none()
            && policy.backend_lowerings[1].targets.as_deref() == Some(TCGEN05_LIBNVVM_TARGETS)
            && policy.backend_lowerings.iter().all(|route| {
                route.mechanism == BackendLoweringMechanism::InlinePtx
                    && route.minimum_ptx.as_deref() == Some("8.6")
                    && route.minimum_sm.is_none()
                    && !route.evidence_profile.trim().is_empty()
            }),
        "{} {family} backend route changed",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_tcgen05_source_contract(
    recipe: &Tcgen05Recipe,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    match recipe.source_contract {
        Tcgen05SourceContract::ExactTablegenSelection => {
            ensure!(
                declaration.selections.len() == 1,
                "{} must keep one exact tcgen05 selection",
                recipe.id
            );
            let selection = &declaration.selections[0];
            let expected_predicate = if matches!(
                recipe.operation,
                Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2
            ) {
                "Subtarget->hasTcgen05ShiftSupport()"
            } else {
                "Subtarget->hasTcgen05InstSupport()"
            };
            ensure!(
                recipe.selection_record == Some(selection.source_record.as_str())
                    && recipe.selection_asm == Some(selection.asm.as_str())
                    && selection.predicates == [expected_predicate]
                    && selection.constraints.is_empty(),
                "{} exact tcgen05 selection changed",
                recipe.id
            );
        }
        Tcgen05SourceContract::TablegenSelectionChangesPtx
            if matches!(
                recipe.operation,
                Tcgen05Operation::Commit | Tcgen05Operation::CommitCg2
            ) =>
        {
            ensure!(
                declaration.selections.len() == 1,
                "{} must keep one canonical commit selection",
                recipe.id
            );
            let selection = &declaration.selections[0];
            ensure!(
                recipe.selection_record == Some(selection.source_record.as_str())
                    && recipe.selection_asm == Some(selection.asm.as_str())
                    && selection.predicates == ["Subtarget->hasTcgen05InstSupport()"]
                    && selection.constraints.is_empty()
                    && !recipe.operands.is_empty(),
                "{} canonical commit selection changed",
                recipe.id
            );
        }
        Tcgen05SourceContract::TablegenSelectionChangesPtx => {
            ensure!(
                declaration.selections.len() == 64,
                "{} must keep the 64 collector selections",
                recipe.id
            );
            let actual = declaration
                .selections
                .iter()
                .map(|selection| {
                    ensure!(
                        selection.constraints.is_empty(),
                        "{} collector selection gained constraints",
                        recipe.id
                    );
                    let predicate = if selection.asm.contains(".kind::i8.") {
                        "Subtarget->hasTcgen05MMAI8Kind()"
                    } else {
                        "Subtarget->hasTcgen05InstSupport()"
                    };
                    ensure!(
                        selection.predicates == [predicate],
                        "{} collector predicate changed",
                        recipe.id
                    );
                    Ok(selection.asm.clone())
                })
                .collect::<Result<BTreeSet<_>>>()?;
            let expected = ["f16", "tf32", "f8f6f4", "i8"]
                .into_iter()
                .flat_map(|kind| {
                    ["b0", "b1", "b2", "b3"].into_iter().flat_map(move |collector| {
                        ["discard", "fill", "use", "lastuse"].into_iter().map(move |action| {
                            format!("tcgen05.mma.ws.cta_group::1.kind::{kind}.collector::{collector}::{action} [$dtmem], [$a], $b, $idesc, $enable_inp_d;")
                        })
                    })
                })
                .collect::<BTreeSet<_>>();
            ensure!(
                actual == expected,
                "{} collector selection matrix changed",
                recipe.id
            );
        }
        Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection => ensure!(
            declaration.selections.is_empty(),
            "{} unexpectedly gained a TableGen selection; review the backend route",
            recipe.id
        ),
    }
    Ok(())
}
