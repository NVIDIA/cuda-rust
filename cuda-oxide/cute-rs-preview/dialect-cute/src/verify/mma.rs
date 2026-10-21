/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared tensor placement, register carriers, and tiled matrix-instruction contracts.

use std::collections::HashSet;

use dialect_mir::ops::{
    MirCastOp, MirConstructStructOp, MirConstructTupleOp, MirExtractFieldOp, MirFieldAddrOp,
    MirInsertFieldOp, MirLoadOp, MirStoreOp, control_flow::MirCondBranchOp,
};
use pliron::{
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    value::Value,
};

use super::{
    Result,
    ghosts::smem_mma_semantic_ids,
    invalid,
    provenance::{
        CarrierKey, OrdinaryFacts, branch_successor_operands, is_semantic_pointer_carrier_cast,
        local_cell, local_cell_is_closed, poison_missing_forwarding_sources,
        propagate_ordinary_plumbing, seed_unknown_integer_carriers,
    },
};
use crate::{
    attributes::CuteTiledMmaPlanAttr,
    ops::epilogue::CuteEpilogueStoreFragmentOp,
    ops::mma::{
        CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
        CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
    },
};

/// Read the tiled-MMA plan from a recognized compute operation, or return `None`.
fn mma_plan_for_op(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Option<CuteTiledMmaPlanAttr> {
    if *opid == CuteTiledMmaSliceOp::get_opid_static() {
        CuteTiledMmaSliceOp::wrap(operation).plan(ctx)
    } else if *opid == CuteFragmentFillOp::get_opid_static() {
        CuteFragmentFillOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaLoadScalesOp::get_opid_static() {
        CuteMmaLoadScalesOp::wrap(operation).plan(ctx)
    } else if *opid == CuteFragmentSliceKOp::get_opid_static() {
        CuteFragmentSliceKOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaLoadAOp::get_opid_static() {
        CuteMmaLoadAOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaPartitionBOp::get_opid_static() {
        CuteMmaPartitionBOp::wrap(operation).plan(ctx)
    } else if *opid == CuteTiledGemmOp::get_opid_static() {
        CuteTiledGemmOp::wrap(operation).plan(ctx)
    } else {
        None
    }
}

/// One possible semantic origin of an ordinary MMA carrier or aggregate field.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MmaFact {
    /// Lane selection produced by a tiled-MMA slice.
    Lane(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Shared allocation base produced by a tensor overlay.
    SmemBase(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Element capacity paired with a tensor overlay's shared base.
    SmemCapacity(
        #[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>,
    ),
    /// A complete stage of register scales produced by a scale load.
    StageScales(
        #[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>,
    ),
    /// A K=64 scale selection produced from one stage.
    SlicedScales(
        #[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>,
    ),
    /// Packed A registers produced by a semantic shared load.
    AFragment(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Shared B base preserved by lazy partitioning.
    BBase(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Element capacity paired with a lazy B partition.
    BCapacity(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Warp-N coordinate of a lazy B partition.
    BWarpN(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// K-half coordinate of a lazy B partition.
    BKHalf(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// FP32 accumulator initialized or updated by the same compute chain.
    Accumulator(
        #[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>,
    ),
    /// An ordinary scalar whose exact SSA identity must remain consistent.
    Scalar(#[doc = "Original SSA value lacking semantic provenance."] Value),
    /// An input without a proven semantic origin; merging it poisons provenance.
    Unknown(#[doc = "Original SSA value lacking semantic provenance."] Value),
}

/// Return the sole carrier fact, rejecting absent or conflicting facts; callers validate its kind.
pub(super) fn one_mma_fact(
    facts: &OrdinaryFacts<MmaFact>,
    value: Value,
    what: &str,
) -> Result<MmaFact> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no shared-MMA provenance after ordinary carrier forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges carriers from different shared-MMA stories"
        )));
    }
    Ok(*found.iter().next().expect("one fact"))
}

/// Require the lane carrier to originate from the expected tiled-MMA slice.
fn require_mma_lane(
    facts: &OrdinaryFacts<MmaFact>,
    value: Value,
    what: &str,
) -> Result<Ptr<Operation>> {
    match one_mma_fact(facts, value, what)? {
        MmaFact::Lane(root) => Ok(root),
        _ => Err(invalid(format!(
            "{what} must come from cute.tiled_mma_slice"
        ))),
    }
}

/// Require the shared base and capacity to come from the same typed overlay.
pub(super) fn require_smem_overlay_pair(
    facts: &OrdinaryFacts<MmaFact>,
    base: Value,
    capacity: Value,
    what: &str,
) -> Result<Ptr<Operation>> {
    let base_root = match one_mma_fact(facts, base, &format!("{what} base"))? {
        MmaFact::SmemBase(root) => root,
        _ => {
            return Err(invalid(format!(
                "{what} base must come from cute.smem_tensor_overlay"
            )));
        }
    };
    let capacity_root = match one_mma_fact(facts, capacity, &format!("{what} capacity"))? {
        MmaFact::SmemCapacity(root) => root,
        _ => {
            return Err(invalid(format!(
                "{what} capacity must come from cute.smem_tensor_overlay"
            )));
        }
    };
    if base_root != capacity_root {
        return Err(invalid(format!(
            "{what} mixes base and capacity from different shared overlays"
        )));
    }
    Ok(base_root)
}

/// Require two scalar carriers to preserve the same original SSA identity.
fn require_same_mma_scalar(
    facts: &OrdinaryFacts<MmaFact>,
    left: Value,
    right: Value,
    what: &str,
) -> Result {
    if left == right {
        return Ok(());
    }
    match (
        one_mma_fact(facts, left, what)?,
        one_mma_fact(facts, right, what)?,
    ) {
        (MmaFact::Scalar(left), MmaFact::Scalar(right)) if left == right => Ok(()),
        _ => Err(invalid(format!(
            "{what} must be one identical ordinary SSA carrier"
        ))),
    }
}

/// Propagate compute origins through semantic operations and ordinary carriers, then poison unknown merges.
pub(super) fn build_mma_facts(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> OrdinaryFacts<MmaFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        let opid = Operation::get_opid(*operation, ctx);
        let operation_ref = operation.deref(ctx);
        if opid == CuteSmemTensorOverlayOp::get_opid_static() {
            let overlay = CuteSmemTensorOverlayOp::wrap(*operation);
            facts.insert(overlay.base(ctx), MmaFact::SmemBase(*operation));
            facts.insert(overlay.capacity(ctx), MmaFact::SmemCapacity(*operation));
        } else if opid == CuteTiledMmaSliceOp::get_opid_static() {
            facts.insert(
                CuteTiledMmaSliceOp::wrap(*operation).sliced_lane(ctx),
                MmaFact::Lane(*operation),
            );
        } else if opid == CuteFragmentFillOp::get_opid_static() {
            facts.insert(
                CuteFragmentFillOp::wrap(*operation).fragment(ctx),
                MmaFact::Accumulator(*operation),
            );
        } else if opid == CuteMmaLoadScalesOp::get_opid_static() {
            facts.insert(
                CuteMmaLoadScalesOp::wrap(*operation).scales(ctx),
                MmaFact::StageScales(*operation),
            );
        } else if opid == CuteFragmentSliceKOp::get_opid_static() {
            facts.insert(
                CuteFragmentSliceKOp::wrap(*operation).scales(ctx),
                MmaFact::SlicedScales(*operation),
            );
        } else if opid == CuteMmaLoadAOp::get_opid_static() {
            facts.insert(
                CuteMmaLoadAOp::wrap(*operation).fragment(ctx),
                MmaFact::AFragment(*operation),
            );
        } else if opid == CuteMmaPartitionBOp::get_opid_static() {
            facts.insert(operation_ref.get_result(0), MmaFact::BBase(*operation));
            facts.insert(operation_ref.get_result(1), MmaFact::BCapacity(*operation));
            facts.insert(operation_ref.get_result(2), MmaFact::BWarpN(*operation));
            facts.insert(operation_ref.get_result(3), MmaFact::BKHalf(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CuteTiledGemmOp::get_opid_static() {
                continue;
            }
            let operation_ref = operation.deref(ctx);
            let Some(accumulator_facts) = facts.at(operation_ref.get_operand(7)).cloned() else {
                continue;
            };
            for accumulator in accumulator_facts {
                if let MmaFact::Accumulator(root) = accumulator {
                    changed |= facts.insert(
                        CuteTiledGemmOp::wrap(*operation).accumulator(ctx),
                        MmaFact::Accumulator(root),
                    );
                }
            }
        }
        if !changed {
            break;
        }
    }

    seed_unknown_integer_carriers(ctx, operations, &mut facts, 32, MmaFact::Scalar);
    seed_unknown_integer_carriers(ctx, operations, &mut facts, 64, MmaFact::Scalar);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, MmaFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

/// Look up an exact semantic origin inside a carrier's aggregate field path.
fn mma_fact_at_path(
    facts: &OrdinaryFacts<MmaFact>,
    value: Value,
    path: &[u32],
    fact: MmaFact,
) -> bool {
    facts
        .at_path(value, path)
        .is_some_and(|found| found.contains(&fact))
}

/// Check whether an operation consumes the fact in its designated semantic operand position.
fn mma_semantic_consumer(
    ctx: &Context,
    user: Ptr<Operation>,
    operand_index: usize,
    fact: MmaFact,
) -> bool {
    let opid = Operation::get_opid(user, ctx);
    match fact {
        MmaFact::Lane(_) => {
            operand_index == 0
                && (opid == CuteMmaLoadScalesOp::get_opid_static()
                    || opid == CuteMmaLoadAOp::get_opid_static()
                    || opid == CuteTiledGemmOp::get_opid_static())
        }
        MmaFact::SmemBase(_) => {
            (opid == CuteMmaLoadScalesOp::get_opid_static() && matches!(operand_index, 1 | 3))
                || (opid == CuteMmaLoadAOp::get_opid_static() && operand_index == 1)
                || (opid == CuteMmaPartitionBOp::get_opid_static() && operand_index == 0)
        }
        MmaFact::SmemCapacity(_) => {
            (opid == CuteMmaLoadScalesOp::get_opid_static() && matches!(operand_index, 2 | 4))
                || (opid == CuteMmaLoadAOp::get_opid_static() && operand_index == 2)
                || (opid == CuteMmaPartitionBOp::get_opid_static() && operand_index == 1)
        }
        MmaFact::StageScales(_) => {
            opid == CuteFragmentSliceKOp::get_opid_static() && operand_index == 0
        }
        MmaFact::SlicedScales(_) => {
            opid == CuteTiledGemmOp::get_opid_static() && operand_index == 6
        }
        MmaFact::AFragment(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 1,
        MmaFact::BBase(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 2,
        MmaFact::BCapacity(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 3,
        MmaFact::BWarpN(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 4,
        MmaFact::BKHalf(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 5,
        MmaFact::Accumulator(_) => {
            (opid == CuteTiledGemmOp::get_opid_static() && operand_index == 7)
                || (opid == CuteEpilogueStoreFragmentOp::get_opid_static() && operand_index == 3)
        }
        MmaFact::Scalar(_) | MmaFact::Unknown(_) => true,
    }
}

/// Check that a branch forwards the carrier fact to the matching successor argument.
fn mma_branch_forwards_fact(
    ctx: &Context,
    facts: &OrdinaryFacts<MmaFact>,
    user: Ptr<Operation>,
    value: Value,
    path: &[u32],
    fact: MmaFact,
) -> bool {
    let operation = user.deref(ctx);
    (0..operation.get_num_successors()).any(|successor_index| {
        let successor = operation.get_successor(successor_index);
        branch_successor_operands(ctx, user, successor_index).is_some_and(|operands| {
            operands
                .iter()
                .copied()
                .enumerate()
                .any(|(index, incoming)| {
                    incoming == value
                        && index < successor.deref(ctx).get_num_arguments()
                        && mma_fact_at_path(
                            facts,
                            successor.deref(ctx).get_argument(index),
                            path,
                            fact,
                        )
                })
        })
    })
}

/// Check that ordinary casts, aggregates, or closed local cells forward the tracked fact without escape.
fn mma_ordinary_use_forwards_fact(
    ctx: &Context,
    facts: &OrdinaryFacts<MmaFact>,
    key: &CarrierKey,
    fact: MmaFact,
    user: Ptr<Operation>,
    operand_index: usize,
) -> bool {
    let opid = Operation::get_opid(user, ctx);
    let operation = user.deref(ctx);
    if opid == MirCastOp::get_opid_static()
        && operand_index == 0
        && is_semantic_pointer_carrier_cast(ctx, user)
        && operation.get_num_results() == 1
    {
        return mma_fact_at_path(facts, operation.get_result(0), &key.path, fact);
    }
    if (opid == MirConstructStructOp::get_opid_static()
        || opid == MirConstructTupleOp::get_opid_static())
        && operation.get_num_results() == 1
    {
        let mut destination_path = vec![operand_index as u32];
        destination_path.extend_from_slice(&key.path);
        return mma_fact_at_path(facts, operation.get_result(0), &destination_path, fact);
    }
    if opid == MirInsertFieldOp::get_opid_static()
        && operation.get_num_operands() == 2
        && operation.get_num_results() == 1
    {
        let Some(index) = MirInsertFieldOp::new(user).get_attr_insert_index(ctx) else {
            return false;
        };
        if operand_index == 0 {
            if key.path.first().copied() == Some(index.0) {
                return true;
            }
            return mma_fact_at_path(facts, operation.get_result(0), &key.path, fact);
        }
        if operand_index == 1 {
            let mut destination_path = vec![index.0];
            destination_path.extend_from_slice(&key.path);
            return mma_fact_at_path(facts, operation.get_result(0), &destination_path, fact);
        }
        return false;
    }
    if opid == MirExtractFieldOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
    {
        let Some(index) = MirExtractFieldOp::new(user).get_attr_index(ctx) else {
            return false;
        };
        if key.path.first().copied() != Some(index.0) {
            return true;
        }
        return mma_fact_at_path(facts, operation.get_result(0), &key.path[1..], fact);
    }
    if opid == MirStoreOp::get_opid_static() && operation.get_num_operands() == 2 {
        let store = MirStoreOp::new(user);
        if operand_index == 0 {
            return local_cell(ctx, store.address_opd(ctx))
                .is_some_and(|(root, _)| local_cell_is_closed(ctx, root));
        }
        if operand_index == 1
            && let Some((root, mut destination_path)) = local_cell(ctx, store.address_opd(ctx))
            && local_cell_is_closed(ctx, root)
        {
            destination_path.extend_from_slice(&key.path);
            return mma_fact_at_path(facts, root, &destination_path, fact);
        }
        return false;
    }
    if opid == MirLoadOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
    {
        let load = MirLoadOp::new(user);
        let Some((root, cell_path)) = local_cell(ctx, load.address_opd(ctx)) else {
            return false;
        };
        if !local_cell_is_closed(ctx, root) {
            return false;
        }
        let Some(result_path) = key.path.strip_prefix(cell_path.as_slice()) else {
            return true;
        };
        return mma_fact_at_path(facts, operation.get_result(0), result_path, fact);
    }
    if opid == MirFieldAddrOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
        && MirFieldAddrOp::new(user)
            .get_attr_field_index(ctx)
            .is_some()
    {
        return local_cell(ctx, operation.get_result(0))
            .is_some_and(|(root, _)| local_cell_is_closed(ctx, root));
    }
    if opid == MirCondBranchOp::get_opid_static() && operand_index == 0 {
        return false;
    }
    mma_branch_forwards_fact(ctx, facts, user, key.value, &key.path, fact)
}

/// Reject ordinary uses that discard or expose semantic MMA carriers outside recognized forwarding.
pub(super) fn verify_mma_use_closure(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &OrdinaryFacts<MmaFact>,
) -> Result {
    for (key, found) in &facts.facts {
        for fact in found.iter().copied() {
            if matches!(fact, MmaFact::Scalar(_) | MmaFact::Unknown(_)) {
                continue;
            }
            for r#use in key.value.uses(ctx) {
                let user = r#use.user_op();
                let operand_index = r#use.find_index(ctx);
                if operations.contains(&user)
                    && (mma_semantic_consumer(ctx, user, operand_index, fact)
                        || mma_ordinary_use_forwards_fact(
                            ctx,
                            facts,
                            key,
                            fact,
                            user,
                            operand_index,
                        ))
                {
                    continue;
                }
                return Err(invalid(format!(
                    "shared-MMA carrier has an unsupported outgoing use by `{}` operand {operand_index}",
                    Operation::get_opid(user, ctx)
                )));
            }
        }
    }
    Ok(())
}

/// Require a compute operation's static plan to match the chain's tiled-MMA root.
fn require_same_mma_plan(
    ctx: &Context,
    left: Ptr<Operation>,
    right: Ptr<Operation>,
    what: &str,
) -> Result {
    let left_id = Operation::get_opid(left, ctx);
    let right_id = Operation::get_opid(right, ctx);
    if mma_plan_for_op(ctx, left, &left_id) != mma_plan_for_op(ctx, right, &right_id) {
        return Err(invalid(format!(
            "{what} connects shared-MMA operations with different tiled-MMA plans"
        )));
    }
    Ok(())
}

/// Validate shared overlays, lazy B partitions, scale/A fragments, and accumulator provenance together.
pub(super) fn verify_smem_mma_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = smem_mma_semantic_ids();
    let facts = build_mma_facts(ctx, all_ops);
    let mut overlays = HashSet::new();
    let mut used_overlays = HashSet::new();
    let mut slices = HashSet::new();
    let mut used_slices = HashSet::new();
    let mut fills = HashSet::new();
    let mut used_fills = HashSet::new();
    let mut scale_loads = HashSet::new();
    let mut used_scale_loads = HashSet::new();
    let mut scale_slices = HashSet::new();
    let mut used_scale_slices = HashSet::new();
    let mut a_loads = HashSet::new();
    let mut used_a_loads = HashSet::new();
    let mut b_partitions = HashSet::new();
    let mut used_b_partitions = HashSet::new();
    let mut gemms = 0usize;

    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "shared/MMA operation `{opid}` is not attached to a block"
            )));
        }
        let operation_ref = operation.deref(ctx);
        if opid == CuteSmemTensorOverlayOp::get_opid_static() {
            overlays.insert(*operation);
        } else if opid == CuteTiledMmaSliceOp::get_opid_static() {
            slices.insert(*operation);
        } else if opid == CuteFragmentFillOp::get_opid_static() {
            fills.insert(*operation);
        } else if opid == CuteMmaLoadScalesOp::get_opid_static() {
            scale_loads.insert(*operation);
            let semantic = CuteMmaLoadScalesOp::wrap(*operation);
            let lane = require_mma_lane(
                &facts,
                operation_ref.get_operand(0),
                "cute.mma_load_scales lane",
            )?;
            require_same_mma_plan(ctx, lane, *operation, "mma scale load")?;
            used_slices.insert(lane);
            let scale_a = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(1),
                operation_ref.get_operand(2),
                "cute.mma_load_scales A",
            )?;
            let scale_b = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(3),
                operation_ref.get_operand(4),
                "cute.mma_load_scales B",
            )?;
            if semantic.scale_a_view(ctx) != CuteSmemTensorOverlayOp::wrap(scale_a).view(ctx)
                || semantic.scale_b_view(ctx) != CuteSmemTensorOverlayOp::wrap(scale_b).view(ctx)
            {
                return Err(invalid(
                    "cute.mma_load_scales typed views do not match its exact shared overlays",
                ));
            }
            used_overlays.extend([scale_a, scale_b]);
        } else if opid == CuteFragmentSliceKOp::get_opid_static() {
            scale_slices.insert(*operation);
            let load = match one_mma_fact(
                &facts,
                operation_ref.get_operand(0),
                "cute.fragment_slice_k stage scales",
            )? {
                MmaFact::StageScales(load) => load,
                _ => {
                    return Err(invalid(
                        "cute.fragment_slice_k must consume cute.mma_load_scales",
                    ));
                }
            };
            require_same_mma_plan(ctx, load, *operation, "scale K slice")?;
            used_scale_loads.insert(load);
        } else if opid == CuteMmaLoadAOp::get_opid_static() {
            a_loads.insert(*operation);
            let semantic = CuteMmaLoadAOp::wrap(*operation);
            let lane =
                require_mma_lane(&facts, operation_ref.get_operand(0), "cute.mma_load_a lane")?;
            require_same_mma_plan(ctx, lane, *operation, "MMA A load")?;
            used_slices.insert(lane);
            let overlay = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(1),
                operation_ref.get_operand(2),
                "cute.mma_load_a",
            )?;
            if semantic.view(ctx) != CuteSmemTensorOverlayOp::wrap(overlay).view(ctx) {
                return Err(invalid(
                    "cute.mma_load_a typed view does not match its exact shared overlay",
                ));
            }
            used_overlays.insert(overlay);
        } else if opid == CuteMmaPartitionBOp::get_opid_static() {
            b_partitions.insert(*operation);
            let semantic = CuteMmaPartitionBOp::wrap(*operation);
            let overlay = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(0),
                operation_ref.get_operand(1),
                "cute.mma_partition_b",
            )?;
            if semantic.view(ctx) != CuteSmemTensorOverlayOp::wrap(overlay).view(ctx) {
                return Err(invalid(
                    "cute.mma_partition_b typed view does not match its exact shared overlay",
                ));
            }
            used_overlays.insert(overlay);
        } else if opid == CuteTiledGemmOp::get_opid_static() {
            gemms += 1;
            let semantic = CuteTiledGemmOp::wrap(*operation);
            let lane =
                require_mma_lane(&facts, operation_ref.get_operand(0), "cute.tiled_gemm lane")?;
            let a = match one_mma_fact(
                &facts,
                operation_ref.get_operand(1),
                "cute.tiled_gemm A fragment",
            )? {
                MmaFact::AFragment(load) => load,
                _ => return Err(invalid("cute.tiled_gemm A must come from cute.mma_load_a")),
            };
            let b = match (
                one_mma_fact(&facts, operation_ref.get_operand(2), "tiled_gemm B base")?,
                one_mma_fact(
                    &facts,
                    operation_ref.get_operand(3),
                    "tiled_gemm B capacity",
                )?,
                one_mma_fact(&facts, operation_ref.get_operand(4), "tiled_gemm B warp-N")?,
                one_mma_fact(&facts, operation_ref.get_operand(5), "tiled_gemm B K-half")?,
            ) {
                (
                    MmaFact::BBase(base),
                    MmaFact::BCapacity(capacity),
                    MmaFact::BWarpN(warp_n),
                    MmaFact::BKHalf(k_half),
                ) if base == capacity && base == warp_n && base == k_half => base,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm B carriers must be the four matching results of one mma_partition_b",
                    ));
                }
            };
            let scales = match one_mma_fact(
                &facts,
                operation_ref.get_operand(6),
                "cute.tiled_gemm scales",
            )? {
                MmaFact::SlicedScales(slice) => slice,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm scales must come from cute.fragment_slice_k",
                    ));
                }
            };
            let fill = match one_mma_fact(
                &facts,
                operation_ref.get_operand(7),
                "cute.tiled_gemm accumulator",
            )? {
                MmaFact::Accumulator(fill) => fill,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm accumulator must be the closed recurrence seeded by fragment_fill",
                    ));
                }
            };
            for producer in [lane, a, b, scales, fill] {
                require_same_mma_plan(ctx, producer, *operation, "tiled GEMM spine")?;
            }
            let scale_load = match one_mma_fact(
                &facts,
                scales.deref(ctx).get_operand(0),
                "tiled GEMM scale-load origin",
            )? {
                MmaFact::StageScales(load) => load,
                _ => unreachable!("fragment_slice_k was verified above"),
            };
            let scale_load_ref = scale_load.deref(ctx);
            let a_ref = a.deref(ctx);
            let b_ref = b.deref(ctx);
            let scale_lane = require_mma_lane(
                &facts,
                scale_load_ref.get_operand(0),
                "mma_load_scales lane",
            )?;
            let a_lane = require_mma_lane(&facts, a_ref.get_operand(0), "mma_load_a lane")?;
            if scale_lane != lane || a_lane != lane {
                return Err(invalid(
                    "tiled GEMM, scale load, and A load must use one tiled-MMA lane slice",
                ));
            }
            require_same_mma_scalar(
                &facts,
                scale_load_ref.get_operand(5),
                a_ref.get_operand(3),
                "shared-MMA warp-M",
            )?;
            require_same_mma_scalar(
                &facts,
                scale_load_ref.get_operand(6),
                b_ref.get_operand(2),
                "shared-MMA warp-N",
            )?;
            require_same_mma_scalar(
                &facts,
                scales.deref(ctx).get_operand(1),
                a_ref.get_operand(4),
                "shared-MMA K-half",
            )?;
            require_same_mma_scalar(
                &facts,
                a_ref.get_operand(4),
                b_ref.get_operand(3),
                "shared-MMA K-half",
            )?;
            if semantic.b_view(ctx) != CuteMmaPartitionBOp::wrap(b).view(ctx) {
                return Err(invalid(
                    "cute.tiled_gemm B view does not match its mma_partition_b producer",
                ));
            }
            used_slices.insert(lane);
            used_a_loads.insert(a);
            used_b_partitions.insert(b);
            used_scale_slices.insert(scales);
            used_fills.insert(fill);
        }
    }

    let found = overlays.len()
        + slices.len()
        + fills.len()
        + scale_loads.len()
        + scale_slices.len()
        + a_loads.len()
        + b_partitions.len()
        + gemms;
    if found == 0 {
        return Ok(());
    }
    verify_mma_use_closure(ctx, all_ops, &facts)?;
    if gemms == 0
        || overlays != used_overlays
        || slices != used_slices
        || fills != used_fills
        || scale_loads != used_scale_loads
        || scale_slices != used_scale_slices
        || a_loads != used_a_loads
        || b_partitions != used_b_partitions
    {
        return Err(invalid(
            "shared-MMA story is incomplete or has an unconsumed overlay/slice/load/partition/fill producer",
        ));
    }
    Ok(())
}
