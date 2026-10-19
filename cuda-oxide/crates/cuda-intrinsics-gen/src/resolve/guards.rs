/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CpAsyncControlOperation, DotProductAdapter,
    ExecutionControlOperation, IntrinsicBackend, OverlayIntrinsic, Tcgen05CpGroup,
    Tcgen05MmaBUsage, Tcgen05MmaKind, Tcgen05SourceContract, TmaOperation, WgmmaControlMode,
};
use crate::ptx::OperandPattern;
use anyhow::{Result, ensure};
use ptx_parse::ParseError;
use std::collections::{BTreeMap, BTreeSet};

use super::families::*;

pub(super) fn insert_unique(set: &mut BTreeSet<String>, value: &str, kind: &str) -> Result<()> {
    ensure!(set.insert(value.to_owned()), "duplicate {kind}: {value}");
    Ok(())
}

pub(super) fn selection_matches_policy(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> Result<bool, ParseError> {
    if policy.family == "wgmma_control" {
        let Some(control) = &policy.wgmma_control else {
            return Ok(false);
        };
        let recipe = wgmma_control_recipe(control.mode);
        let selection_shape_matches = if control.mode == WgmmaControlMode::WaitGroup {
            selection.asm == "wgmma.wait_group.sync.aligned \t$n;"
        } else {
            policy.expected_ptx.matches(&selection.asm)?
        };
        return Ok(selection.source_record == recipe.selection_record
            && selection_shape_matches
            && selection.predicates == ["Subtarget->getPTXVersion() >= 80", "hasSM90a"]
            && selection.constraints.is_empty());
    }

    if policy.family == "tma" {
        return Ok(selection_matches_tma_policy(policy, selection));
    }
    if matches!(
        policy.family.as_str(),
        "counted_barrier" | "grid_dependency" | "register_control"
    ) {
        let Some(operation) = ExecutionControlOperation::from_catalog_id(&policy.id) else {
            return Ok(false);
        };
        let recipe = execution_control_recipe(operation);
        return Ok(recipe
            .selection_records
            .contains(&selection.source_record.as_str())
            && selection.asm == recipe.selection_asm
            && selection.predicates == recipe.selection_predicates
            && selection.constraints.is_empty());
    }
    if policy.family == "tcgen05" {
        let Some(tcgen05) = &policy.tcgen05 else {
            return Ok(false);
        };
        if let Some(mma) = &tcgen05.mma {
            let expected = if mma.alias.is_some() {
                BTreeSet::from([if tcgen05_mma_is_ws(mma.form) {
                    tcgen05_mma_declaration_asm(
                        mma.form,
                        Tcgen05MmaKind::F8f6f4,
                        1,
                        None,
                        Some(0),
                        Some(Tcgen05MmaBUsage::Discard),
                    )
                } else {
                    tcgen05_mma_declaration_asm(
                        mma.form,
                        Tcgen05MmaKind::F8f6f4,
                        1,
                        Some("discard"),
                        None,
                        None,
                    )
                }])
            } else {
                tcgen05_mma_valid_selection_asms(mma.form)
            };
            return Ok(expected.contains(&selection.asm)
                && selection.predicates
                    == [if selection.asm.contains(".kind::i8.") {
                        "Subtarget->hasTcgen05MMAI8Kind()"
                    } else {
                        "Subtarget->hasTcgen05InstSupport()"
                    }]
                && selection.constraints.is_empty());
        }
        if tcgen05.source_contract != Tcgen05SourceContract::ExactTablegenSelection {
            return Ok(false);
        }
        if let Some(cp) = tcgen05.cp {
            let recipe = tcgen05_cp_member_recipe(cp.member);
            let group = match cp.group {
                Tcgen05CpGroup::Cg1 => 1,
                Tcgen05CpGroup::Cg2 => 2,
            };
            return Ok(selection.source_record
                == format!("TCGEN05_CP_{}_cg{group}", recipe.selection_stem)
                && selection.asm
                    == format!(
                        "tcgen05.cp.cta_group::{group}.{} \t[$tmem_addr], $sdesc;",
                        recipe.ptx_suffix
                    )
                && selection.predicates == ["Subtarget->hasTcgen05InstSupport()"]
                && selection.constraints.is_empty());
        }
        let recipe = tcgen05_recipe(tcgen05.operation);
        return Ok(
            recipe.selection_record == Some(selection.source_record.as_str())
                && recipe.selection_asm == Some(selection.asm.as_str())
                && selection.constraints.is_empty(),
        );
    }
    if policy.family == "sparse_mma" {
        let Some(last) = policy.expected_ptx.operands.last() else {
            return Ok(false);
        };
        if *last != OperandPattern::Immediate {
            return Ok(false);
        }
        let mut selection_shape = policy.expected_ptx.clone();
        *selection_shape.operands.last_mut().unwrap() = OperandPattern::RegisterOrImmediate;
        return Ok(selection_shape.matches(&selection.asm)? && selection.constraints.is_empty());
    }
    if policy.family == "sync" {
        if policy.id == "sync_threads" {
            return Ok(selection.source_record == "BARRIER_CTA_SYNC_ALIGNED_ALL_i"
                && selection.asm == "bar.sync \t$i;"
                && selection.predicates.is_empty()
                && selection.constraints.is_empty());
        }
        let Some(scope) = threadfence_scope_for_id(&policy.id) else {
            return Ok(false);
        };
        let recipe = threadfence_recipe(scope);
        return Ok(selection.source_record == recipe.selection_record
            && selection.asm == format!("membar.{};", recipe.ptx_level)
            && selection.predicates.is_empty()
            && selection.constraints.is_empty());
    }

    if policy.family == "vote" {
        let Some(vote) = &policy.vote else {
            return Ok(false);
        };
        let recipe = vote_recipe(vote.mode);
        return Ok([recipe.immediate_selection, recipe.register_selection]
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)?
            && selection.constraints.address_space.is_none()
            && selection.constraints.immediate_bindings.is_empty());
    }

    if policy.family == "warp_match" {
        let Some(warp_match) = &policy.warp_match else {
            return Ok(false);
        };
        let recipe = warp_match_recipe(warp_match.mode, warp_match.value_width);
        return Ok(recipe
            .selections
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)?
            && selection.constraints.is_empty());
    }

    if policy.family == "elect" {
        return Ok(["INT_ELECT_SYNC_I", "INT_ELECT_SYNC_R"]
            .contains(&selection.source_record.as_str())
            && selection.asm == "elect.sync \t$dest|$pred, $mask;"
            && selection.constraints.is_empty());
    }

    if policy.family == "warp_barrier" {
        return Ok(policy.warp_barrier.is_some()
            && ["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"]
                .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)?
            && selection.constraints.is_empty());
    }

    if policy.family == "warp_shuffle" {
        let Some(shuffle) = &policy.warp_shuffle else {
            return Ok(false);
        };
        let recipe = warp_shuffle_recipe(shuffle.mode, shuffle.value_kind);
        return Ok(selection.asm
            == format!(
                "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
                recipe.ptx_mode
            )
            && selection.constraints.is_empty());
    }

    if policy.family == "cp_async_copy" {
        let Some(copy) = &policy.cp_async_copy else {
            return Ok(false);
        };
        let Some(recipe) = cp_async_copy_recipe(copy) else {
            return Ok(false);
        };
        return Ok(recipe
            .selections
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)?
            && selection.constraints.is_empty());
    }

    if policy.family == "cp_async_control" {
        let Some(control) = &policy.cp_async_control else {
            return Ok(false);
        };
        let recipe = cp_async_control_recipe(control.operation);
        let instruction_matches = if control.operation == CpAsyncControlOperation::WaitGroup {
            selection.asm == "cp.async.wait_group \t$n;"
        } else {
            policy.expected_ptx.matches(&selection.asm)?
        };
        return Ok(selection.source_record == recipe.selection
            && instruction_matches
            && selection.constraints.is_empty());
    }

    if policy.family == "cp_async_mbarrier" {
        let Some(bridge) = &policy.cp_async_mbarrier else {
            return Ok(false);
        };
        let recipe = cp_async_mbarrier_recipe(bridge.operation, bridge.state_space);
        return Ok(selection.source_record == recipe.selection
            && selection.asm == recipe.selection_asm
            && selection.constraints.is_empty());
    }

    if policy.family == "mbarrier_basic" {
        let Some(mbarrier) = &policy.mbarrier_basic else {
            return Ok(false);
        };
        let recipe = mbarrier_basic_recipe(mbarrier.operation);
        return Ok(selection.source_record == recipe.selection
            && policy.expected_ptx.matches(&selection.asm)?
            && selection.constraints.is_empty());
    }

    if !policy.expected_ptx.matches(&selection.asm)?
        || policy
            .selected_address_space
            .is_some_and(|address_space| selection.constraints.address_space != Some(address_space))
    {
        return Ok(false);
    }

    let Some(dot_product) = &policy.dot_product else {
        return Ok(true);
    };
    if selection.constraints.address_space.is_some() {
        return Ok(false);
    }
    Ok(match dot_product.adapter {
        DotProductAdapter::DirectThreeOperands => {
            selection.constraints.immediate_bindings.is_empty()
        }
        DotProductAdapter::InsertLowHalfFalse => {
            selection.constraints.immediate_bindings.len() == 1
                && selection.constraints.immediate_bindings[0].argument_index == 2
                && selection.constraints.immediate_bindings[0].value == 0
        }
    })
}

pub(super) fn selection_matches_tma_policy(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> bool {
    let Some(tma) = &policy.tma else {
        return false;
    };
    let operation = tma.operation;
    if matches!(
        operation,
        TmaOperation::Reduce
            | TmaOperation::PrefetchTensorMap
            | TmaOperation::ReplaceBoxDim
            | TmaOperation::ReplaceElementStride
            | TmaOperation::ReplaceElementType
            | TmaOperation::ReplaceFillMode
            | TmaOperation::ReplaceGlobalAddress
            | TmaOperation::ReplaceGlobalDim
            | TmaOperation::ReplaceGlobalStride
            | TmaOperation::ReplaceInterleaveLayout
            | TmaOperation::ReplaceRank
            | TmaOperation::ReplaceSwizzleAtomicity
            | TmaOperation::ReplaceSwizzleMode
    ) {
        return false;
    }
    let (source_record, asm) = match operation {
        TmaOperation::G2sTile1d => (
            "TMA_G2S_TILE_CG0_1D",
            "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes$cg [$dst], [$tmap, {{$d0}}], [$mbar];",
        ),
        TmaOperation::G2sTile2d => (
            "TMA_G2S_TILE_CG0_2D",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes$cg [$dst], [$tmap, {{$d0, $d1}}], [$mbar];",
        ),
        TmaOperation::G2sTile2dMulticast => (
            "TMA_G2S_TILE_CG0_2D_MC",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster$cg [$dst], [$tmap, {{$d0, $d1}}], [$mbar], $mc;",
        ),
        TmaOperation::G2sTile2dMulticastCg2 => (
            "TMA_G2S_TILE_2D_MC",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster$cg [$dst], [$tmap, {{$d0, $d1}}], [$mbar], $mc;",
        ),
        TmaOperation::G2sTile3d => (
            "TMA_G2S_TILE_CG0_3D",
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes$cg [$dst], [$tmap, {{$d0, $d1, $d2}}], [$mbar];",
        ),
        TmaOperation::G2sTile4d => (
            "TMA_G2S_TILE_CG0_4D",
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes$cg [$dst], [$tmap, {{$d0, $d1, $d2, $d3}}], [$mbar];",
        ),
        TmaOperation::G2sTile5d => (
            "TMA_G2S_TILE_CG0_5D",
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes$cg [$dst], [$tmap, {{$d0, $d1, $d2, $d3, $d4}}], [$mbar];",
        ),
        TmaOperation::S2gTile1d => (
            "TMA_TENSOR_S2G_TILE_1D",
            "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [$tmap, {{$d0}}], [$src];",
        ),
        TmaOperation::S2gTile2d => (
            "TMA_TENSOR_S2G_TILE_2D",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [$tmap, {{$d0, $d1}}], [$src];",
        ),
        TmaOperation::S2gTile3d => (
            "TMA_TENSOR_S2G_TILE_3D",
            "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [$tmap, {{$d0, $d1, $d2}}], [$src];",
        ),
        TmaOperation::S2gTile4d => (
            "TMA_TENSOR_S2G_TILE_4D",
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [$tmap, {{$d0, $d1, $d2, $d3}}], [$src];",
        ),
        TmaOperation::S2gTile5d => (
            "TMA_TENSOR_S2G_TILE_5D",
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [$tmap, {{$d0, $d1, $d2, $d3, $d4}}], [$src];",
        ),
        TmaOperation::CommitGroup => ("CP_ASYNC_BULK_COMMIT_GROUP", "cp.async.bulk.commit_group;"),
        TmaOperation::WaitGroup => ("CP_ASYNC_BULK_WAIT_GROUP", "cp.async.bulk.wait_group \t$n;"),
        TmaOperation::WaitGroupRead => (
            "CP_ASYNC_BULK_WAIT_GROUP_READ",
            "cp.async.bulk.wait_group.read \t$n;",
        ),
        TmaOperation::PrefetchTile1d => (
            "TMA_TENSOR_PF_TILE_1D",
            "cp.async.bulk.prefetch.tensor.1d.L2.global.tile [$tmap, {{$d0}}];",
        ),
        TmaOperation::PrefetchTile2d => (
            "TMA_TENSOR_PF_TILE_2D",
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [$tmap, {{$d0, $d1}}];",
        ),
        TmaOperation::PrefetchTile3d => (
            "TMA_TENSOR_PF_TILE_3D",
            "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [$tmap, {{$d0, $d1, $d2}}];",
        ),
        TmaOperation::PrefetchTile4d => (
            "TMA_TENSOR_PF_TILE_4D",
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile [$tmap, {{$d0, $d1, $d2, $d3}}];",
        ),
        TmaOperation::PrefetchTile5d => (
            "TMA_TENSOR_PF_TILE_5D",
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile [$tmap, {{$d0, $d1, $d2, $d3, $d4}}];",
        ),
        TmaOperation::PrefetchTileGather4TwoDimensional => (
            "TMA_TENSOR_PF_TILE_GATHER4_2D",
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [$tmap, {{$d0, $d1, $d2, $d3, $d4}}];",
        ),
        TmaOperation::PrefetchTile1dCacheHint => (
            "TMA_TENSOR_PF_TILE_1D_CH",
            "cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint [$tmap, {{$d0}}], $ch;",
        ),
        TmaOperation::PrefetchTile2dCacheHint => (
            "TMA_TENSOR_PF_TILE_2D_CH",
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [$tmap, {{$d0, $d1}}], $ch;",
        ),
        TmaOperation::PrefetchTile3dCacheHint => (
            "TMA_TENSOR_PF_TILE_3D_CH",
            "cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [$tmap, {{$d0, $d1, $d2}}], $ch;",
        ),
        TmaOperation::PrefetchTile4dCacheHint => (
            "TMA_TENSOR_PF_TILE_4D_CH",
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint [$tmap, {{$d0, $d1, $d2, $d3}}], $ch;",
        ),
        TmaOperation::PrefetchTile5dCacheHint => (
            "TMA_TENSOR_PF_TILE_5D_CH",
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint [$tmap, {{$d0, $d1, $d2, $d3, $d4}}], $ch;",
        ),
        TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint => (
            "TMA_TENSOR_PF_TILE_GATHER4_2D_CH",
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [$tmap, {{$d0, $d1, $d2, $d3, $d4}}], $ch;",
        ),
        TmaOperation::FenceProxyTensorMapAcquireCluster => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_CLUSTER",
            "fence.proxy.tensormap::generic.acquire.cluster [$addr], 128;",
        ),
        TmaOperation::FenceProxyTensorMapAcquireCta => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_CTA",
            "fence.proxy.tensormap::generic.acquire.cta [$addr], 128;",
        ),
        TmaOperation::FenceProxyTensorMapAcquireGpu => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_GPU",
            "fence.proxy.tensormap::generic.acquire.gpu [$addr], 128;",
        ),
        TmaOperation::FenceProxyTensorMapAcquireSystem => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_ACQUIRE_SYS",
            "fence.proxy.tensormap::generic.acquire.sys [$addr], 128;",
        ),
        TmaOperation::FenceProxyTensorMapReleaseCluster => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_CLUSTER",
            "fence.proxy.tensormap::generic.release.cluster;",
        ),
        TmaOperation::FenceProxyTensorMapReleaseCta => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_CTA",
            "fence.proxy.tensormap::generic.release.cta;",
        ),
        TmaOperation::FenceProxyTensorMapReleaseGpu => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_GPU",
            "fence.proxy.tensormap::generic.release.gpu;",
        ),
        TmaOperation::FenceProxyTensorMapReleaseSystem => (
            "INT_FENCE_PROXY_TENSORMAP_GENERIC_RELEASE_SYS",
            "fence.proxy.tensormap::generic.release.sys;",
        ),
        TmaOperation::Reduce
        | TmaOperation::PrefetchTensorMap
        | TmaOperation::ReplaceBoxDim
        | TmaOperation::ReplaceElementStride
        | TmaOperation::ReplaceElementType
        | TmaOperation::ReplaceFillMode
        | TmaOperation::ReplaceGlobalAddress
        | TmaOperation::ReplaceGlobalDim
        | TmaOperation::ReplaceGlobalStride
        | TmaOperation::ReplaceInterleaveLayout
        | TmaOperation::ReplaceRank
        | TmaOperation::ReplaceSwizzleAtomicity
        | TmaOperation::ReplaceSwizzleMode => unreachable!("selectionless TMA operation"),
    };

    let mut immediate_bindings = Vec::new();
    if matches!(
        operation,
        TmaOperation::G2sTile1d
            | TmaOperation::G2sTile2d
            | TmaOperation::G2sTile2dMulticast
            | TmaOperation::G2sTile2dMulticastCg2
            | TmaOperation::G2sTile3d
            | TmaOperation::G2sTile4d
            | TmaOperation::G2sTile5d
    ) {
        let dimensions = operation.dimensions().unwrap();
        immediate_bindings.push(crate::model::ImportedImmediateBinding {
            argument_index: dimensions + 5,
            value: if matches!(
                operation,
                TmaOperation::G2sTile2dMulticast | TmaOperation::G2sTile2dMulticastCg2
            ) {
                -1
            } else {
                0
            },
        });
        immediate_bindings.push(crate::model::ImportedImmediateBinding {
            argument_index: dimensions + 6,
            value: 0,
        });
    } else if matches!(
        operation,
        TmaOperation::S2gTile1d
            | TmaOperation::S2gTile2d
            | TmaOperation::S2gTile3d
            | TmaOperation::S2gTile4d
            | TmaOperation::S2gTile5d
    ) {
        immediate_bindings.push(crate::model::ImportedImmediateBinding {
            argument_index: operation.dimensions().unwrap() + 3,
            value: 0,
        });
    } else if let Some(coordinate_count) = operation.prefetch_coordinate_count() {
        immediate_bindings.push(crate::model::ImportedImmediateBinding {
            argument_index: coordinate_count + 2,
            value: if operation.uses_prefetch_cache_hint() {
                -1
            } else {
                0
            },
        });
    }

    selection.source_record == source_record
        && selection.asm == asm
        && selection.constraints
            == crate::model::ImportedSelectionConstraints {
                address_space: None,
                immediate_bindings,
            }
}

pub(super) fn ensure_exact_inline_ptx_backends(
    policy: &OverlayIntrinsic,
    requirements: [(IntrinsicBackend, &str, Option<&str>); 2],
    family: &str,
) -> Result<()> {
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly two reviewed {family} inline-PTX routes",
        policy.id
    );
    let requirements: BTreeMap<_, _> = requirements
        .into_iter()
        .map(|(backend, ptx, minimum_sm)| (backend, (ptx, minimum_sm)))
        .collect();
    for lowering in &policy.backend_lowerings {
        let (minimum_ptx, minimum_sm) = requirements[&lowering.backend];
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == minimum_sm
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact {family} floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

pub(super) fn ensure_no_other_family_contract(
    policy: &OverlayIntrinsic,
    family: &str,
) -> Result<()> {
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && (policy.family == "tma" || policy.selected_address_space.is_none())
            && (policy.family == "packed_alu") == policy.packed_alu.is_some()
            && (policy.family == "integer_minmax") == policy.integer_minmax.is_some()
            && (policy.family == "packed_conversion") == policy.packed_conversion.is_some()
            && (policy.family == "scalar_conversion") == policy.scalar_conversion.is_some()
            && (policy.family == "scalar_arithmetic") == policy.scalar_arithmetic.is_some()
            && (policy.family == "scalar_math") == policy.scalar_math.is_some()
            && (policy.family == "extended_minmax") == policy.extended_minmax.is_some()
            && (policy.family == "cp_async_copy") == policy.cp_async_copy.is_some()
            && (policy.family == "cp_async_control") == policy.cp_async_control.is_some()
            && (policy.family == "cp_async_mbarrier") == policy.cp_async_mbarrier.is_some()
            && (policy.family == "mbarrier_basic") == policy.mbarrier_basic.is_some()
            && (policy.family == "movmatrix") == policy.movmatrix.is_some()
            && (policy.family == "mbarrier_extended") == policy.mbarrier_extended.is_some()
            && (policy.family == "register_mma") == policy.register_mma.is_some()
            && (policy.family == "sparse_mma") == policy.sparse_mma.is_some()
            && (policy.family == "prmt") == policy.prmt.is_some()
            && (policy.family == "cluster_barrier") == policy.cluster_barrier.is_some()
            && (policy.family == "wgmma_control") == policy.wgmma_control.is_some()
            && (policy.family == "debug_control") == policy.debug_control.is_some()
            && (policy.family == "cluster_memory") == policy.cluster_memory.is_some()
            && (policy.family == "clc") == policy.clc.is_some()
            && (policy.family == "tma") == policy.tma.is_some()
            && (policy.family == "tcgen05") == policy.tcgen05.is_some(),
        "{} mixes another generated-family contract with {family}",
        policy.id
    );
    Ok(())
}
