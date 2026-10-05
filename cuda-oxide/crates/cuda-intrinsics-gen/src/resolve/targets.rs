/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogHardwareAlternative, CatalogHardwareTarget, CatalogTargetAlternative,
    CatalogTargetContract, CatalogTargetRequirement, ClcOperation, ExecutionControlOperation,
    IntrinsicBackend, LdmatrixElement, LdmatrixShape, OverlayIntrinsic, PtxVersion,
    SparseMmaAccumulator, TargetContract, TargetSelectorBinding, Tcgen05Operation, TmaOperation,
};
use anyhow::{Context, Result, bail, ensure};

use super::families::*;

/// Check authored instruction floors, never backend-profile evidence floors.
pub(super) fn validate_arch_introduction_floor(policy: &OverlayIntrinsic) -> Result<()> {
    let native_sm = policy
        .packed_alu
        .as_ref()
        .map(|packed| packed.native_minimum_sm)
        .or_else(|| {
            policy
                .packed_atomic
                .as_ref()
                .map(|packed| packed.native_minimum_sm)
        })
        .or_else(|| {
            policy
                .integer_minmax
                .as_ref()
                .map(|integer| integer.native_minimum_sm)
        });
    let sm = match native_sm {
        Some(sm) => sm,
        None => match policy.minimum_sm.as_deref() {
            Some(sm) => parse_sm_spelling(&policy.id, "minimum_sm", sm, None)?,
            None => return Ok(()),
        },
    };
    let arch = cuda_target_spec::CudaArch::new(u32::from(sm), None)?;
    let introduction = cuda_target_spec::arch_introduced_ptx(&arch).with_context(|| {
        format!(
            "{} has no recorded PTX introduction for hardware floor {arch}",
            policy.id
        )
    })?;
    let floor = parse_ptx_version(&policy.minimum_ptx, &policy.id)?;
    ensure!(
        floor.encoded() >= introduction,
        "{} minimum_ptx {} is below PTX {}.{}, which first named hardware floor {arch}",
        policy.id,
        policy.minimum_ptx,
        introduction / 10,
        introduction % 10,
    );
    Ok(())
}

pub(super) fn validate_selected_target_predicates(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> Result<()> {
    if policy.family == "wgmma_control" {
        ensure!(
            selection.predicates == ["Subtarget->getPTXVersion() >= 80", "hasSM90a"]
                && parse_ptx_version(&policy.minimum_ptx, &policy.id)?.encoded() == 80
                && parse_hardware_target(policy)?
                    == CatalogHardwareTarget::AnyOf {
                        alternatives: vec![CatalogHardwareAlternative::ExactArchitecture {
                            sm: 90,
                        }],
                    },
            "{} WGMMA-control selection must retain the PTX 8.0 and sm_90a gates",
            policy.id
        );
        return Ok(());
    }
    if policy.family == "register_control" {
        let operation = ExecutionControlOperation::from_catalog_id(&policy.id)
            .with_context(|| format!("{} has no closed register-control operation", policy.id))?;
        ensure!(
            matches!(
                operation,
                ExecutionControlOperation::SetMaxNRegInc | ExecutionControlOperation::SetMaxNRegDec
            ) && selection.predicates == ["Subtarget->hasSetMaxNRegSupport()"]
                && parse_ptx_version(&policy.minimum_ptx, &policy.id)?.encoded() == 80
                && policy.minimum_sm.is_none()
                && policy.targets == TENSOR_MAP_REPLACE_TARGETS,
            "{} setmaxnreg selection must retain the helper predicate and reviewed PTX 8.0 target matrix",
            policy.id
        );
        return Ok(());
    }

    let mma_family = matches!(policy.family.as_str(), "register_mma" | "sparse_mma");
    let tcgen05_mma = policy
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref());
    let mut imported_ptx: Option<u16> = None;
    let mut imported_sm: Option<u16> = None;
    let mut has_dot_instructions = false;
    let mut has_clc_multicast_support = false;
    let mut has_tma_blackwell_support = false;
    let mut has_tcgen05_support = false;
    let mut has_tcgen05_shift_support = false;
    let mut has_tcgen05_mma_i8_support = false;
    let mut has_ldstmatrix_blackwell_support = false;
    let mut has_mma_block_scale_support = false;
    let mut has_redux_sync_f32_support = false;
    for predicate in &selection.predicates {
        if let Some(value) = predicate.strip_prefix("Subtarget->getPTXVersion() >= ") {
            let value = value.parse::<u16>().with_context(|| {
                format!("{} has malformed PTX predicate {predicate:?}", policy.id)
            })?;
            ensure!(
                imported_ptx.is_none() || mma_family,
                "{} has duplicate PTX predicates",
                policy.id
            );
            imported_ptx = Some(imported_ptx.unwrap_or_default().max(value));
        } else if let Some(value) = predicate.strip_prefix("Subtarget->getSmVersion() >= ") {
            let value = value.parse::<u16>().with_context(|| {
                format!("{} has malformed SM predicate {predicate:?}", policy.id)
            })?;
            ensure!(
                imported_sm.is_none() || mma_family,
                "{} has duplicate SM predicates",
                policy.id
            );
            imported_sm = Some(imported_sm.unwrap_or_default().max(value));
        } else if predicate == "hasDotInstructions" {
            ensure!(
                policy.family == "dotprod",
                "{} selected instruction uses dot-product target gating outside the dotprod family",
                policy.id
            );
            ensure!(
                !has_dot_instructions && imported_ptx.is_none() && imported_sm.is_none(),
                "{} has duplicate or conflicting dot-product target predicates",
                policy.id
            );
            has_dot_instructions = true;
            imported_ptx = Some(50);
            imported_sm = Some(61);
        } else if predicate == "Subtarget->hasClusterLaunchControlTryCancelMulticastSupport()" {
            ensure!(
                policy.family == "clc"
                    && policy
                        .clc
                        .as_ref()
                        .is_some_and(|clc| { clc.operation == ClcOperation::TryCancelMulticast }),
                "{} uses the CLC multicast target predicate outside that operation",
                policy.id
            );
            ensure!(
                !has_clc_multicast_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions,
                "{} has duplicate or conflicting CLC multicast target predicates",
                policy.id
            );
            has_clc_multicast_support = true;
        } else if predicate == "Subtarget->hasTMABlackwellSupport()" {
            ensure!(
                policy.family == "tma"
                    && policy.tma.as_ref().is_some_and(|tma| {
                        matches!(
                            tma.operation,
                            TmaOperation::G2sTile2dMulticastCg2
                                | TmaOperation::PrefetchTileGather4TwoDimensional
                                | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint
                        )
                    }),
                "{} uses the Blackwell TMA target predicate outside the cta_group::2 operation",
                policy.id
            );
            ensure!(
                !has_tma_blackwell_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support,
                "{} has duplicate or conflicting Blackwell TMA target predicates",
                policy.id
            );
            has_tma_blackwell_support = true;
        } else if predicate == "Subtarget->hasTcgen05InstSupport()" {
            ensure!(
                policy.family == "tcgen05" && policy.tcgen05.is_some(),
                "{} uses the tcgen05 target predicate outside that family",
                policy.id
            );
            ensure!(
                !has_tcgen05_support
                    && !has_tcgen05_shift_support
                    && !has_tcgen05_mma_i8_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support
                    && !has_tma_blackwell_support,
                "{} has duplicate or conflicting tcgen05 target predicates",
                policy.id
            );
            has_tcgen05_support = true;
        } else if predicate == "Subtarget->hasTcgen05MMAI8Kind()" {
            ensure!(
                tcgen05_mma.is_some() && selection.asm.contains(".kind::i8."),
                "{} uses the tcgen05 I8 predicate outside an I8 MMA selection",
                policy.id
            );
            ensure!(
                !has_tcgen05_mma_i8_support
                    && !has_tcgen05_support
                    && !has_tcgen05_shift_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support
                    && !has_tma_blackwell_support,
                "{} has duplicate or conflicting tcgen05 I8 predicates",
                policy.id
            );
            has_tcgen05_mma_i8_support = true;
        } else if predicate == "Subtarget->hasTcgen05ShiftSupport()" {
            ensure!(
                policy.family == "tcgen05"
                    && policy.tcgen05.as_ref().is_some_and(|tcgen05| {
                        matches!(
                            tcgen05.operation,
                            Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2
                        )
                    }),
                "{} uses the tcgen05 shift target predicate outside a shift operation",
                policy.id
            );
            ensure!(
                !has_tcgen05_shift_support
                    && !has_tcgen05_support
                    && !has_tcgen05_mma_i8_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support
                    && !has_tma_blackwell_support,
                "{} has duplicate or conflicting tcgen05 shift target predicates",
                policy.id
            );
            has_tcgen05_shift_support = true;
        } else if predicate == "Subtarget->hasLdStmatrixBlackwellSupport()" {
            ensure!(
                policy.family == "ldmatrix"
                    && policy.ldmatrix_variant.as_ref().is_some_and(|variant| {
                        variant.shape != LdmatrixShape::M8n8
                            && variant.element != LdmatrixElement::B16
                    }),
                "{} uses the Blackwell ld/stmatrix predicate outside a Blackwell ldmatrix variant",
                policy.id
            );
            ensure!(
                !has_ldstmatrix_blackwell_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support
                    && !has_tma_blackwell_support
                    && !has_tcgen05_support
                    && !has_tcgen05_shift_support,
                "{} has duplicate or conflicting Blackwell ldmatrix target predicates",
                policy.id
            );
            has_ldstmatrix_blackwell_support = true;
        } else if predicate == "Subtarget->hasMMABlockScale()" {
            ensure!(
                matches!(policy.family.as_str(), "register_mma" | "sparse_mma")
                    && (policy.register_mma.is_some() || policy.sparse_mma.is_some()),
                "{} uses the MMA block-scale target predicate outside a closed MMA family",
                policy.id
            );
            ensure!(
                !has_mma_block_scale_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions
                    && !has_clc_multicast_support
                    && !has_tma_blackwell_support
                    && !has_tcgen05_support
                    && !has_tcgen05_shift_support
                    && !has_ldstmatrix_blackwell_support,
                "{} has duplicate or conflicting MMA block-scale target predicates",
                policy.id
            );
            has_mma_block_scale_support = true;
        } else if predicate == "Subtarget->hasReduxSyncF32()" {
            ensure!(
                policy.family == "redux"
                    && policy
                        .redux
                        .as_ref()
                        .is_some_and(|redux| redux_recipe(redux.operation).value_type == "f32"),
                "{} uses the f32 redux target predicate outside an f32 redux operation",
                policy.id
            );
            ensure!(
                !has_redux_sync_f32_support && imported_ptx.is_none() && imported_sm.is_none(),
                "{} has duplicate or conflicting f32 redux target predicates",
                policy.id
            );
            has_redux_sync_f32_support = true;
        } else {
            bail!(
                "{} selected instruction has unsupported target predicate {predicate:?}; target gates must fail closed",
                policy.id
            );
        }
    }
    let overlay_ptx = parse_ptx_version(&policy.minimum_ptx, &policy.id)?.encoded();
    if let Some(mma) = tcgen05_mma {
        let i8_selection = selection.asm.contains(".kind::i8.");
        ensure!(
            selection.predicates.len() == 1
                && ((i8_selection && has_tcgen05_mma_i8_support && !has_tcgen05_support)
                    || (!i8_selection && has_tcgen05_support && !has_tcgen05_mma_i8_support))
                && !has_tcgen05_shift_support,
            "{} tcgen05 MMA selection does not retain its exact kind predicate",
            policy.id
        );
        let contracts = expected_tcgen05_mma_target_contracts(IntrinsicBackend::LlvmNvptx);
        let expected_target = if let Some(fixed) = mma.fixed_selectors {
            resolve_target_contract(
                "tcgen05 MMA selected predicate",
                &[TargetSelectorBinding {
                    name: "kind".into(),
                    value: tcgen05_mma_kind_name(fixed.kind).into(),
                }],
                &contracts,
            )?
        } else {
            resolve_target_contracts("tcgen05 MMA selected predicate", &contracts)?
        };
        ensure!(
            mma.llvm_target == expected_target,
            "{} tcgen05 MMA predicate does not map to its closed LLVM target matrix",
            policy.id
        );
        return Ok(());
    }
    if has_mma_block_scale_support {
        let (minimum_ptx_matches, target_matches) = match policy.family.as_str() {
            "register_mma" => (
                overlay_ptx == 87,
                policy.targets == "sm_120a|sm_120f|sm_121a|sm_121f" && policy.minimum_sm.is_none(),
            ),
            "sparse_mma" => (
                overlay_ptx == 87,
                policy.minimum_sm.is_none()
                    && policy.sparse_mma.as_ref().is_some_and(|mma| {
                        policy.targets
                            == match mma.accumulator {
                                SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32 => {
                                    SPARSE_MMA_F8F6F4_TARGETS
                                }
                                SparseMmaAccumulator::S32 => return false,
                            }
                    }),
            ),
            _ => (false, false),
        };
        ensure!(
            minimum_ptx_matches && target_matches,
            "{} MMA block-scale predicate requires its reviewed Blackwell target matrix",
            policy.id
        );
        return Ok(());
    }
    if has_redux_sync_f32_support {
        ensure!(
            selection.predicates.len() == 1
                && overlay_ptx == 86
                && policy.targets == REDUX_F32_TARGETS
                && policy.minimum_sm.is_none(),
            "{} f32 redux selection must carry only the hasReduxSyncF32 predicate and its reviewed Blackwell target matrix",
            policy.id
        );
        return Ok(());
    }
    // MMA uses reviewed inline PTX. Its imported predicates gate LLVM's typed
    // selection, while the closed recipe and terminal evidence set the native
    // PTX floor.
    if mma_family {
        return Ok(());
    }
    if let Some(imported_ptx) = imported_ptx {
        ensure!(
            overlay_ptx == imported_ptx,
            "{} minimum PTX {} disagrees with selected instruction predicate PTX {}",
            policy.id,
            policy.minimum_ptx,
            format_args!("{}.{}", imported_ptx / 10, imported_ptx % 10)
        );
    }
    if let Some(imported_sm) = imported_sm {
        if let Some(packed) = &policy.packed_alu {
            ensure!(
                packed.native_minimum_sm == imported_sm,
                "{} native minimum SM {} disagrees with selected instruction predicate sm_{}",
                policy.id,
                packed.native_minimum_sm,
                imported_sm
            );
        } else {
            let overlay_target = parse_hardware_target(policy)?;
            let target_matches = overlay_target
                == CatalogHardwareTarget::AnyOf {
                    alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: imported_sm }],
                };
            ensure!(
                target_matches,
                "{} minimum SM {:?} disagrees with selected instruction predicate sm_{}",
                policy.id,
                policy.minimum_sm,
                imported_sm
            );
        }
    }
    if policy.family == "ldmatrix" {
        if policy
            .ldmatrix_variant
            .as_ref()
            .is_some_and(|variant| variant.shape == LdmatrixShape::M8n8)
        {
            ensure!(
                imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
                "{} classic ldmatrix selection must carry exactly its PTX and SM predicates",
                policy.id
            );
        } else {
            ensure!(
                has_ldstmatrix_blackwell_support
                    && selection.predicates.len() == 1
                    && policy.targets == BLACKWELL_LDMATRIX_LLVM_TARGETS,
                "{} Blackwell ldmatrix selection must retain its helper predicate and reviewed exact targets",
                policy.id
            );
        }
    } else if policy.family == "dotprod" {
        ensure!(
            has_dot_instructions && selection.predicates.len() == 1,
            "{} dotprod selection must carry only the hasDotInstructions predicate",
            policy.id
        );
    } else if policy.family == "clc" {
        match policy.clc.as_ref().map(|clc| clc.operation) {
            Some(ClcOperation::TryCancel) => ensure!(
                imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
                "{} selection must carry exactly its PTX and SM predicates",
                policy.id
            ),
            Some(ClcOperation::TryCancelMulticast) => ensure!(
                has_clc_multicast_support
                    && selection.predicates.len() == 1
                    && parse_hardware_target(policy)?
                        == CatalogHardwareTarget::AnyOf {
                            alternatives: vec![
                                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 121 },
                            ],
                        },
                "{} multicast target predicate must map to the reviewed exact architectures",
                policy.id
            ),
            _ => bail!(
                "{} query operation unexpectedly has an instruction selection",
                policy.id
            ),
        }
    } else if policy.family == "tma" {
        if policy.tma.as_ref().is_some_and(|tma| {
            matches!(
                tma.operation,
                TmaOperation::G2sTile2dMulticastCg2
                    | TmaOperation::PrefetchTileGather4TwoDimensional
                    | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint
            )
        }) {
            ensure!(
                has_tma_blackwell_support
                    && selection.predicates.len() == 1
                    && parse_hardware_target(policy)?
                        == CatalogHardwareTarget::AnyOf {
                            alternatives: vec![
                                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                            ],
                        },
                "{} Blackwell TMA predicate must map to the reviewed exact architectures",
                policy.id
            );
        } else {
            ensure!(
                imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
                "{} TMA selection must carry exactly its PTX and SM predicates",
                policy.id
            );
        }
    } else if policy.family == "tcgen05" {
        let shift = policy.tcgen05.as_ref().is_some_and(|tcgen05| {
            matches!(
                tcgen05.operation,
                Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2
            )
        });
        ensure!(
            ((shift && has_tcgen05_shift_support && !has_tcgen05_support)
                || (!shift && has_tcgen05_support && !has_tcgen05_shift_support))
                && selection.predicates.len() == 1
                && parse_hardware_target(policy)?
                    == CatalogHardwareTarget::AnyOf {
                        alternatives: vec![
                            CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                            CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                            CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                            CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                        ],
                    },
            "{} tcgen05 selection must use the reviewed accelerated architectures",
            policy.id
        );
    } else if matches!(
        policy.family.as_str(),
        "vote"
            | "active_mask"
            | "warp_match"
            | "elect"
            | "warp_barrier"
            | "warp_shuffle"
            | "cp_async_copy"
            | "cp_async_control"
            | "mbarrier_basic"
            | "cluster_barrier"
            | "scalar_conversion"
            | "grid_dependency"
    ) {
        ensure!(
            imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
            "{} selection must carry exactly its PTX and SM predicates",
            policy.id
        );
    }
    Ok(())
}

pub(super) fn parse_ptx_version(value: &str, intrinsic_id: &str) -> Result<PtxVersion> {
    value
        .parse()
        .map_err(|reason: String| anyhow::anyhow!("{intrinsic_id} minimum_ptx {value:?}: {reason}"))
}

pub(super) fn parse_hardware_target(policy: &OverlayIntrinsic) -> Result<CatalogHardwareTarget> {
    parse_hardware_target_fields(&policy.id, &policy.targets, policy.minimum_sm.as_deref())
}

/// Resolve every selector contract without merging target sets.
pub(crate) fn resolve_target_contracts(
    intrinsic_id: &str,
    contracts: &[TargetContract],
) -> Result<CatalogTargetRequirement> {
    ensure!(
        !contracts.is_empty(),
        "{intrinsic_id} has no reviewed target contracts"
    );

    let expected_selector_names = contracts[0]
        .selectors
        .iter()
        .map(|selector| selector.name.as_str())
        .collect::<Vec<_>>();
    let mut resolved_contracts = Vec::with_capacity(contracts.len());
    for contract in contracts {
        validate_selector_bindings(
            intrinsic_id,
            "target-contract selectors",
            &contract.selectors,
        )?;
        ensure!(
            contract
                .selectors
                .iter()
                .map(|selector| selector.name.as_str())
                .eq(expected_selector_names.iter().copied()),
            "{intrinsic_id} target contracts must use one closed selector schema"
        );
        ensure!(
            !contract.alternatives.is_empty(),
            "{intrinsic_id} target contract {:?} has no alternatives",
            contract.selectors
        );

        let alternatives = contract
            .alternatives
            .iter()
            .map(|alternative| {
                Ok(CatalogTargetAlternative {
                    minimum_ptx: parse_ptx_version(&alternative.minimum_ptx, intrinsic_id)?,
                    hardware: parse_target_contract_hardware(intrinsic_id, &alternative.target)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            alternatives
                .windows(2)
                .all(|pair| target_hardware_sort_key(pair[0].hardware)
                    < target_hardware_sort_key(pair[1].hardware)),
            "{intrinsic_id} target contract {:?} alternatives must have unique, sorted hardware targets",
            contract.selectors
        );
        let minimum_count = alternatives
            .iter()
            .filter(|alternative| {
                matches!(
                    alternative.hardware,
                    CatalogHardwareAlternative::MinimumSm { .. }
                )
            })
            .count();
        ensure!(
            minimum_count == 0 || alternatives.len() == 1,
            "{intrinsic_id} target contract {:?} cannot mix a minimum-SM range with other alternatives",
            contract.selectors
        );
        resolved_contracts.push(CatalogTargetContract {
            selectors: contract.selectors.clone(),
            alternatives,
        });
    }
    ensure!(
        resolved_contracts
            .windows(2)
            .all(|pair| pair[0].selectors < pair[1].selectors),
        "{intrinsic_id} target contracts must have unique, sorted selector bindings"
    );

    let minimum_ptx = resolved_contracts
        .iter()
        .flat_map(|contract| contract.alternatives.iter())
        .map(|alternative| alternative.minimum_ptx)
        .min()
        .unwrap();
    Ok(CatalogTargetRequirement {
        minimum_ptx,
        hardware: CatalogHardwareTarget::TargetMatrix {
            contracts: resolved_contracts,
        },
    })
}

/// Resolve one fixed selector tuple from a closed target matrix.
pub(crate) fn resolve_target_contract(
    intrinsic_id: &str,
    selected: &[TargetSelectorBinding],
    contracts: &[TargetContract],
) -> Result<CatalogTargetRequirement> {
    validate_selector_bindings(intrinsic_id, "selected target selectors", selected)?;
    let resolved = resolve_target_contracts(intrinsic_id, contracts)?;
    let CatalogHardwareTarget::TargetMatrix { contracts } = resolved.hardware else {
        unreachable!("target-contract resolution always returns a matrix")
    };

    let matching = contracts
        .into_iter()
        .filter(|contract| contract.selectors == selected)
        .collect::<Vec<_>>();
    ensure!(
        matching.len() == 1,
        "{intrinsic_id} selected target selectors {:?} must match exactly one reviewed contract",
        selected
    );
    let contract = matching.into_iter().next().unwrap();
    let minimum_ptx = contract
        .alternatives
        .iter()
        .map(|alternative| alternative.minimum_ptx)
        .min()
        .unwrap();
    Ok(CatalogTargetRequirement {
        minimum_ptx,
        hardware: CatalogHardwareTarget::TargetMatrix {
            contracts: vec![contract],
        },
    })
}

pub(super) fn validate_selector_bindings(
    intrinsic_id: &str,
    label: &str,
    selectors: &[TargetSelectorBinding],
) -> Result<()> {
    for selector in selectors {
        ensure!(
            is_target_selector_token(&selector.name) && is_target_selector_token(&selector.value),
            "{intrinsic_id} {label} must use lowercase snake-case names and values"
        );
    }
    ensure!(
        selectors.windows(2).all(|pair| pair[0].name < pair[1].name),
        "{intrinsic_id} {label} must have unique, sorted names"
    );
    Ok(())
}

pub(super) fn is_target_selector_token(value: &str) -> bool {
    value
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_lowercase())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        && !value.ends_with('_')
        && !value.contains("__")
}

pub(super) fn parse_target_contract_hardware(
    intrinsic_id: &str,
    target: &str,
) -> Result<CatalogHardwareAlternative> {
    if let Some(minimum) = target.strip_suffix('+') {
        let sm = parse_sm_spelling(intrinsic_id, "target contract", minimum, None)?;
        return Ok(CatalogHardwareAlternative::MinimumSm { sm });
    }
    parse_exact_hardware_alternative(intrinsic_id, target)
}

pub(super) fn target_hardware_sort_key(hardware: CatalogHardwareAlternative) -> (u16, u8) {
    match hardware {
        CatalogHardwareAlternative::MinimumSm { sm } => (sm, 0),
        CatalogHardwareAlternative::ExactArchitecture { sm } => (sm, 1),
        CatalogHardwareAlternative::FamilyTarget { sm } => (sm, 2),
    }
}

pub(super) fn parse_hardware_target_fields(
    intrinsic_id: &str,
    targets: &str,
    minimum_sm: Option<&str>,
) -> Result<CatalogHardwareTarget> {
    if targets == "all" {
        let Some(minimum_sm) = minimum_sm else {
            return Ok(CatalogHardwareTarget::All);
        };
        let sm = parse_sm_spelling(intrinsic_id, "minimum_sm", minimum_sm, None)?;
        return Ok(CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm }],
        });
    }

    if targets.contains('|') {
        ensure!(
            minimum_sm.is_none(),
            "{} target alternatives {:?} cannot be combined with minimum_sm",
            intrinsic_id,
            targets
        );
        let spellings = targets.split('|').collect::<Vec<_>>();
        ensure!(
            spellings.len() >= 2,
            "{} target alternatives must contain at least two targets",
            intrinsic_id
        );
        ensure!(
            spellings.windows(2).all(|pair| pair[0] < pair[1]),
            "{} target alternatives must be unique and sorted",
            intrinsic_id
        );
        return Ok(CatalogHardwareTarget::AnyOf {
            alternatives: spellings
                .into_iter()
                .map(|spelling| parse_exact_hardware_alternative(intrinsic_id, spelling))
                .collect::<Result<Vec<_>>>()?,
        });
    }

    ensure!(
        minimum_sm.is_none(),
        "{} exact targets {:?} cannot be combined with minimum_sm",
        intrinsic_id,
        targets
    );
    Ok(CatalogHardwareTarget::AnyOf {
        alternatives: vec![parse_exact_hardware_alternative(intrinsic_id, targets)?],
    })
}

pub(super) fn parse_exact_hardware_alternative(
    intrinsic_id: &str,
    target: &str,
) -> Result<CatalogHardwareAlternative> {
    let suffix = target
        .chars()
        .last()
        .filter(|suffix| matches!(suffix, 'a' | 'f'));
    let Some(suffix) = suffix else {
        bail!(
            "{} targets {:?} must be `all`, exact `sm_Na`, or family `sm_Nf`",
            intrinsic_id,
            target
        );
    };
    let sm = parse_sm_spelling(intrinsic_id, "targets", target, Some(suffix))?;
    Ok(match suffix {
        'a' => CatalogHardwareAlternative::ExactArchitecture { sm },
        'f' => CatalogHardwareAlternative::FamilyTarget { sm },
        _ => unreachable!(),
    })
}

pub(super) fn parse_sm_spelling(
    intrinsic_id: &str,
    field: &str,
    value: &str,
    suffix: Option<char>,
) -> Result<u16> {
    let body = value.strip_prefix("sm_").with_context(|| {
        format!("{intrinsic_id} {field} {value:?} must use canonical sm_NN spelling")
    })?;
    let digits = match suffix {
        Some(suffix) => body.strip_suffix(suffix).with_context(|| {
            format!("{intrinsic_id} {field} {value:?} has the wrong target suffix")
        })?,
        None => body,
    };
    ensure!(
        matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit()),
        "{} {} {:?} must use canonical sm_NN{} spelling",
        intrinsic_id,
        field,
        value,
        suffix.map_or("", |suffix| if suffix == 'a' { "a" } else { "f" })
    );
    let sm: u16 = digits
        .parse()
        .with_context(|| format!("{intrinsic_id} {field} target is too large"))?;
    let canonical = match suffix {
        Some(suffix) => format!("sm_{sm}{suffix}"),
        None => format!("sm_{sm}"),
    };
    ensure!(
        sm > 0 && canonical == value,
        "{} {} {:?} is not canonical",
        intrinsic_id,
        field,
        value
    );
    Ok(sm)
}

pub(super) fn backend_target_requirement(
    policy: &OverlayIntrinsic,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<CatalogTargetRequirement> {
    if let Some(mma) = policy
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref())
    {
        return Ok(match lowering.backend {
            IntrinsicBackend::LlvmNvptx => mma.llvm_target.clone(),
            IntrinsicBackend::LibNvvm => mma.libnvvm_target.clone(),
        });
    }
    let minimum_ptx = lowering
        .minimum_ptx
        .as_deref()
        .unwrap_or(&policy.minimum_ptx);
    let targets = lowering.targets.as_deref().unwrap_or(&policy.targets);
    let minimum_sm = lowering.minimum_sm.as_deref().or_else(|| {
        if lowering.targets.is_none() {
            policy.minimum_sm.as_deref()
        } else {
            None
        }
    });
    Ok(CatalogTargetRequirement {
        minimum_ptx: parse_ptx_version(minimum_ptx, &policy.id)?,
        hardware: parse_hardware_target_fields(&policy.id, targets, minimum_sm)?,
    })
}
