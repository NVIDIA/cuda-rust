/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared output placement and asynchronous store synchronization contracts.

use dialect_mir::{
    ops::{
        MirAddOp, MirCallOp, MirCastOp, MirConstantOp, MirConstructStructOp, MirConstructTupleOp,
        MirExtractFieldOp, MirGotoOp, MirInsertFieldOp, MirLoadOp, MirRefOp, MirStorageDeadOp,
        MirStorageLiveOp, MirUndefOp,
    },
    types::{MirPtrType, MirStructType, address_space},
};
use pliron::{
    builtin::types::IntegerType,
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{
    Result,
    cfg::{operation_reaches, require_linear_protocol_order},
    ghosts::epilogue_semantic_ids,
    invalid,
    mma::{MmaFact, build_mma_facts, one_mma_fact},
    provenance::{is_semantic_pointer_carrier_cast, local_cell, local_cell_is_closed},
    tma::direct_tma_view_producer,
};
use crate::{
    attributes::CuteEpilogueSyncPhaseAttr,
    ops::epilogue::{
        CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp,
        CuteEpilogueSyncOp, CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp,
        CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
    },
    ops::gemm_tma::{CuteTmaGmemViewOp, CuteTmaSmemViewOp},
    ops::mma::CuteFragmentFillOp,
};

/// Accept only pointer casts that preserve shared carrier provenance.
fn is_epilogue_pointer_identity_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_semantic_pointer_carrier_cast(ctx, operation)
}

/// Recognize carrier-only operations allowed between writer synchronization anchors.
fn is_epilogue_writer_plumbing(ctx: &Context, operation: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(operation, ctx);
    opid == MirGotoOp::get_opid_static()
        || opid == MirStorageLiveOp::get_opid_static()
        || opid == MirStorageDeadOp::get_opid_static()
        || opid == MirExtractFieldOp::get_opid_static()
        || is_epilogue_pointer_identity_cast(ctx, operation)
        || is_closed_local_cell_load(ctx, operation)
}

/// A non-volatile `mir.load` from a closed compiler-created local cell only
/// re-materializes carrier state (e.g. the epilogue handle struct spilled to
/// an alloca slot); it cannot touch tile memory, so it is admissible plumbing
/// between the writer-protocol anchor operations.
fn is_closed_local_cell_load(ctx: &Context, operation: Ptr<Operation>) -> bool {
    if Operation::get_opid(operation, ctx) != MirLoadOp::get_opid_static() {
        return false;
    }
    let load = MirLoadOp::new(operation);
    {
        let operation_ref = operation.deref(ctx);
        if operation_ref.get_num_operands() != 1 || operation_ref.get_num_results() != 1 {
            return false;
        }
    }
    if load.is_volatile(ctx) {
        return false;
    }
    let Some((root, _path)) = local_cell(ctx, load.address_opd(ctx)) else {
        return false;
    };
    local_cell_is_closed(ctx, root)
}

/// Accept only the dead immutable reference Rust leaves around a no-inline
/// call on the fieldless, zero-sized TMA store-pipeline token.
fn is_dead_epilogue_store_pipeline_ref(ctx: &Context, operation: Ptr<Operation>) -> bool {
    if Operation::get_opid(operation, ctx) != MirRefOp::get_opid_static() {
        return false;
    }
    let reference = MirRefOp::new(operation);
    if reference.verify(ctx).is_err()
        || reference
            .get_attr_mutable(ctx)
            .is_none_or(|mutable| mutable.0)
    {
        return false;
    }
    let operation_ref = operation.deref(ctx);
    let referent = operation_ref.get_operand(0);
    let result = operation_ref.get_result(0);
    if result.is_used(ctx) {
        return false;
    }

    let referent_ty = referent.get_type(ctx);
    let referent_ty_ref = referent_ty.deref(ctx);
    let Some(store_pipeline) = referent_ty_ref.downcast_ref::<MirStructType>() else {
        return false;
    };
    if store_pipeline.name != "TmaStorePipeline"
        || !store_pipeline.field_names.is_empty()
        || !store_pipeline.field_types.is_empty()
        || !store_pipeline.mem_to_decl.is_empty()
        || !store_pipeline.field_offsets.is_empty()
        || store_pipeline.total_size != 0
        || store_pipeline.abi_align != 1
    {
        return false;
    }
    let result_ty = result.get_type(ctx);
    let result_ty_ref = result_ty.deref(ctx);
    let Some(pointer) = result_ty_ref.downcast_ref::<MirPtrType>() else {
        return false;
    };
    if pointer.pointee != referent_ty
        || pointer.is_mutable
        || pointer.address_space != address_space::GENERIC
    {
        return false;
    }

    let Some(construct) = referent.defining_op() else {
        return false;
    };
    if Operation::get_opid(construct, ctx) != MirConstructStructOp::get_opid_static() {
        return false;
    }
    let construct_ref = construct.deref(ctx);
    if construct_ref.get_num_operands() != 0
        || construct_ref.get_num_results() != 1
        || construct_ref.get_result(0) != referent
    {
        return false;
    }
    let uses = referent.uses(ctx);
    uses.len() == 1 && uses[0].user_op() == operation && uses[0].find_index(ctx) == 0
}

/// Recognize harmless carrier assembly and indexing between TMA issuer anchors.
fn is_epilogue_issuer_plumbing(ctx: &Context, operation: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(operation, ctx);
    is_epilogue_writer_plumbing(ctx, operation)
        || opid == MirUndefOp::get_opid_static()
        || opid == MirInsertFieldOp::get_opid_static()
        || opid == MirConstructStructOp::get_opid_static()
        || opid == MirConstructTupleOp::get_opid_static()
        || opid == MirConstantOp::get_opid_static()
        || opid == MirAddOp::get_opid_static()
        || is_dead_epilogue_store_pipeline_ref(ctx, operation)
}

/// Follow constant field insertions to the requested carrier, rejecting overly deep chains.
fn resolve_inserted_epilogue_field(
    ctx: &Context,
    mut aggregate: Value,
    wanted_index: u32,
) -> Result<Option<Value>> {
    for _ in 0..16 {
        let Some(definer) = aggregate.defining_op() else {
            return Ok(None);
        };
        if Operation::get_opid(definer, ctx) != MirInsertFieldOp::get_opid_static() {
            return Ok(None);
        }
        let insert = MirInsertFieldOp::new(definer);
        let Some(index) = insert.get_attr_insert_index(ctx) else {
            return Err(invalid(
                "epilogue carrier has an insert_field without a constant index",
            ));
        };
        let operation = definer.deref(ctx);
        if index.0 == wanted_index {
            return Ok(Some(operation.get_operand(1)));
        }
        aggregate = operation.get_operand(0);
    }
    Err(invalid(
        "epilogue carrier insert/extract chain is unexpectedly deep",
    ))
}

/// Peel only lossless SSA carrier plumbing. Memory, arithmetic, control flow,
/// and arbitrary aliases deliberately stop provenance resolution.
fn resolve_epilogue_ssa_carrier(ctx: &Context, mut value: Value) -> Result<Value> {
    for _ in 0..16 {
        let Some(definer) = value.defining_op() else {
            return Ok(value);
        };
        let opid = Operation::get_opid(definer, ctx);
        if opid == MirCastOp::get_opid_static() {
            if is_semantic_pointer_carrier_cast(ctx, definer) {
                value = definer.deref(ctx).get_operand(0);
                continue;
            }
            return Ok(value);
        }
        if opid == MirExtractFieldOp::get_opid_static() {
            let extract = MirExtractFieldOp::new(definer);
            let Some(index) = extract.get_attr_index(ctx) else {
                return Err(invalid(
                    "epilogue carrier has an extract_field without a constant index",
                ));
            };
            let aggregate = definer.deref(ctx).get_operand(0);
            if let Some(aggregate_definer) = aggregate.defining_op() {
                let aggregate_id = Operation::get_opid(aggregate_definer, ctx);
                if aggregate_id == MirConstructStructOp::get_opid_static()
                    || aggregate_id == MirConstructTupleOp::get_opid_static()
                {
                    let aggregate_operation = aggregate_definer.deref(ctx);
                    if usize::try_from(index.0)
                        .ok()
                        .is_some_and(|index| index < aggregate_operation.get_num_operands())
                    {
                        value = aggregate_operation.get_operand(index.0 as usize);
                        continue;
                    }
                }
            }
            let Some(inserted) = resolve_inserted_epilogue_field(ctx, aggregate, index.0)? else {
                return Ok(value);
            };
            value = inserted;
            continue;
        }
        return Ok(value);
    }
    Err(invalid(
        "epilogue carrier cast/insert/extract chain is unexpectedly deep",
    ))
}

/// Reject carriers that differ after peeling lossless SSA forwarding.
fn require_same_epilogue_carrier(
    ctx: &Context,
    actual: Value,
    expected: Value,
    message: &str,
) -> Result {
    if resolve_epilogue_ssa_carrier(ctx, actual)? != resolve_epilogue_ssa_carrier(ctx, expected)? {
        return Err(invalid(message));
    }
    Ok(())
}

/// Read a direct unsigned 64-bit MIR literal used for epilogue coordinates.
fn epilogue_u64_constant(ctx: &Context, value: Value) -> Option<u64> {
    let integer = value.get_type(ctx);
    let integer = integer.deref(ctx);
    if !integer
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
    {
        return None;
    }
    let constant = value.defining_op()?;
    if Operation::get_opid(constant, ctx) != MirConstantOp::get_opid_static() {
        return None;
    }
    let attribute = MirConstantOp::new(constant).get_attr_value(ctx)?;
    (attribute.value().bw() <= 64).then(|| attribute.value().to_u64())
}

/// Require the right output half's column to be the left column plus one.
fn require_next_epilogue_column(ctx: &Context, left: Value, right: Value) -> Result {
    let left = resolve_epilogue_ssa_carrier(ctx, left)?;
    let right = resolve_epilogue_ssa_carrier(ctx, right)?;
    let Some(add) = right.defining_op() else {
        return Err(invalid(
            "epilogue half 1 must target the tile column immediately after half 0",
        ));
    };
    if Operation::get_opid(add, ctx) != MirAddOp::get_opid_static() {
        return Err(invalid("epilogue half 1 must target `half_0_column + 1`"));
    }
    let add = add.deref(ctx);
    let lhs = resolve_epilogue_ssa_carrier(ctx, add.get_operand(0))?;
    let rhs = resolve_epilogue_ssa_carrier(ctx, add.get_operand(1))?;
    if !((lhs == left && epilogue_u64_constant(ctx, rhs) == Some(1))
        || (rhs == left && epilogue_u64_constant(ctx, lhs) == Some(1)))
    {
        return Err(invalid("epilogue half 1 must target `half_0_column + 1`"));
    }
    Ok(())
}

/// Require reachable hand-offs from acquire through writers, issuer, commit, and tail.
pub(super) fn require_epilogue_cfg_links(
    ctx: &Context,
    acquire: Ptr<Operation>,
    reusable_sync: Ptr<Operation>,
    ready_sync: Ptr<Operation>,
    half_zero: Ptr<Operation>,
    commit: Ptr<Operation>,
    tail: Ptr<Operation>,
) -> Result {
    if operation_reaches(ctx, acquire, reusable_sync)
        && operation_reaches(ctx, ready_sync, half_zero)
        && operation_reaches(ctx, commit, tail)
    {
        Ok(())
    } else {
        Err(invalid(
            "epilogue CFG must connect acquire -> writer hand-off -> TMA issuer -> commit -> tail inside one role-anchored story",
        ))
    }
}

/// A semantic output copy and the direct views supplying its storage.
#[derive(Clone, Copy)]
struct EpilogueStoreFacts {
    /// The semantic shared-to-global copy being checked.
    operation: Ptr<Operation>,
    /// The direct shared view producer carrying the output half.
    source_view: Ptr<Operation>,
    /// The direct descriptor-backed global view producer.
    destination_view: Ptr<Operation>,
}

/// Resolve both copy operands to direct TMA views inside the current function scope.
fn epilogue_store_facts(
    ctx: &Context,
    operation: Ptr<Operation>,
    scope: &[Ptr<Operation>],
) -> Result<EpilogueStoreFacts> {
    let store = CuteTmaStore2dSemanticOp::wrap(operation);
    let source_view = direct_tma_view_producer(
        ctx,
        store.source(ctx),
        &CuteTmaSmemViewOp::get_opid_static(),
        "epilogue store source",
        scope,
    )?;
    let destination_view = direct_tma_view_producer(
        ctx,
        store.destination(ctx),
        &CuteTmaGmemViewOp::get_opid_static(),
        "epilogue store destination",
        scope,
    )?;
    Ok(EpilogueStoreFacts {
        operation,
        source_view,
        destination_view,
    })
}

/// Validate the single two-half output protocol, carrier origins, and role-local ordering.
pub(super) fn verify_epilogue_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = epilogue_semantic_ids();
    let mut overlays = Vec::new();
    let mut slices = Vec::new();
    let mut fragments = Vec::new();
    let mut syncs = Vec::new();
    let mut halves = Vec::new();
    let mut acquires = Vec::new();
    let mut commits = Vec::new();
    let mut tails = Vec::new();
    let mut stores = Vec::new();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if opid == ids[0] {
            overlays.push(*operation);
        } else if opid == ids[1] {
            slices.push(*operation);
        } else if opid == ids[2] {
            fragments.push(*operation);
        } else if opid == ids[3] {
            syncs.push(*operation);
        } else if opid == ids[4] {
            halves.push(*operation);
        } else if opid == ids[5] {
            acquires.push(*operation);
        } else if opid == ids[6] {
            commits.push(*operation);
        } else if opid == ids[7] {
            tails.push(*operation);
        } else if opid == ids[8] {
            stores.push(*operation);
        }
    }

    let found = overlays.len()
        + slices.len()
        + fragments.len()
        + syncs.len()
        + halves.len()
        + acquires.len()
        + commits.len()
        + tails.len()
        + stores.len();
    if found == 0 {
        return Ok(());
    }
    if overlays.len() != 1
        || slices.len() != 1
        || fragments.len() != 1
        || syncs.len() != 2
        || halves.len() != 2
        || acquires.len() != 1
        || commits.len() != 1
        || tails.len() != 1
        || stores.len() != 2
    {
        return Err(invalid(format!(
            "epilogue story needs exactly 1 overlay, 1 warp slice, 1 fragment store, 2 syncs, 2 halves, 1 acquire, 2 TMA stores, 1 commit, and 1 tail; found {}, {}, {}, {}, {}, {}, {}, {}, {}",
            overlays.len(),
            slices.len(),
            fragments.len(),
            syncs.len(),
            halves.len(),
            acquires.len(),
            stores.len(),
            commits.len(),
            tails.len()
        )));
    }

    let overlay = overlays[0];
    let warp_slice = slices[0];
    let store_fragment = fragments[0];
    let acquire = acquires[0];
    let commit = commits[0];
    let tail = tails[0];
    let overlay_view = CuteEpilogueSmemOverlayOp::wrap(overlay);
    let warp_view = CuteEpilogueWarpSliceOp::wrap(warp_slice);
    let fragment = CuteEpilogueStoreFragmentOp::wrap(store_fragment);
    let overlay_tile = overlay_view
        .tile(ctx)
        .expect("locally verified epilogue overlay has a tile");
    if warp_view.tile(ctx).as_ref() != Some(&overlay_tile)
        || fragment.tile(ctx).as_ref() != Some(&overlay_tile)
    {
        return Err(invalid(
            "epilogue overlay, warp slice, and fragment store must carry the same tile plan",
        ));
    }
    let mma_facts = build_mma_facts(ctx, all_ops);
    let accumulator_seed = match one_mma_fact(
        &mma_facts,
        fragment.accumulator(ctx),
        "epilogue fragment accumulator",
    )? {
        MmaFact::Accumulator(seed) => seed,
        _ => {
            return Err(invalid(
                "epilogue fragment accumulator must be the closed tiled-GEMM recurrence",
            ));
        }
    };
    if CuteFragmentFillOp::wrap(accumulator_seed).plan(ctx)
        != Some(overlay_tile.plan.tiled_mma.clone())
    {
        return Err(invalid(
            "epilogue accumulator plan differs from its shared-MMA story",
        ));
    }
    require_same_epilogue_carrier(
        ctx,
        warp_view.input_base(ctx),
        overlay_view.base(ctx),
        "epilogue warp slice must use the base produced by the epilogue overlay",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.base(ctx),
        warp_view.base(ctx),
        "epilogue fragment store must use the base produced by the warp slice",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.warp_id(ctx),
        warp_view.warp_id(ctx),
        "epilogue fragment store must use the warp ID produced by the warp slice",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.lane(ctx),
        warp_view.lane(ctx),
        "epilogue fragment store must use the lane produced by the warp slice",
    )?;

    let reusable_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::Reusable)
        })
        .ok_or_else(|| invalid("epilogue story is missing the Reusable sync"))?;
    let ready_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::ReadyForTma)
        })
        .ok_or_else(|| invalid("epilogue story is missing the ReadyForTma sync"))?;
    if reusable_sync == ready_sync {
        return Err(invalid(
            "epilogue story needs one distinct sync for each phase",
        ));
    }
    for sync in [reusable_sync, ready_sync] {
        let sync = CuteEpilogueSyncOp::wrap(sync);
        if sync.tile(ctx).as_ref() != Some(&overlay_tile) {
            return Err(invalid(
                "both epilogue syncs must carry the overlay tile plan",
            ));
        }
        require_same_epilogue_carrier(
            ctx,
            sync.base(ctx),
            overlay_view.base(ctx),
            "both epilogue syncs must use the base produced by the epilogue overlay",
        )?;
    }

    let writer_range = require_linear_protocol_order(
        ctx,
        "epilogue writer protocol",
        &[reusable_sync, store_fragment, ready_sync],
    )?;
    for operation in &writer_range {
        let expected = [reusable_sync, store_fragment, ready_sync].contains(operation);
        if !expected && !is_epilogue_writer_plumbing(ctx, *operation) {
            return Err(invalid(format!(
                "epilogue writer protocol has interposed non-carrier operation `{}`",
                Operation::get_opid(*operation, ctx)
            )));
        }
    }
    let writer_events: Vec<_> = writer_range
        .iter()
        .copied()
        .filter(|operation| {
            let opid = Operation::get_opid(*operation, ctx);
            opid.dialect.to_string() == crate::CUTE_DIALECT_NAME
                || opid == MirCallOp::get_opid_static()
        })
        .collect();
    if writer_events != [reusable_sync, store_fragment, ready_sync] {
        return Err(invalid(
            "epilogue writer protocol must be exactly Reusable -> fragment store -> ReadyForTma, whose semantic boundary includes publication to the TMA proxy",
        ));
    }

    let half_zero = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|half| half.0 == 0)
        })
        .ok_or_else(|| invalid("epilogue story is missing half 0"))?;
    let half_one = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|half| half.0 == 1)
        })
        .ok_or_else(|| invalid("epilogue story is missing half 1"))?;
    if half_zero == half_one {
        return Err(invalid("epilogue story needs two distinct shared halves"));
    }
    for half in [half_zero, half_one] {
        let half = CuteEpilogueHalfOp::wrap(half);
        if half.tile(ctx).as_ref() != Some(&overlay_tile) {
            return Err(invalid(
                "both epilogue halves must carry the overlay tile plan",
            ));
        }
        require_same_epilogue_carrier(
            ctx,
            half.full_base(ctx),
            overlay_view.base(ctx),
            "both epilogue halves must use the base produced by the epilogue overlay",
        )?;
    }

    let store_zero = epilogue_store_facts(ctx, stores[0], all_ops)?;
    let store_one = epilogue_store_facts(ctx, stores[1], all_ops)?;
    for (half, store) in [(half_zero, store_zero), (half_one, store_one)] {
        let half = CuteEpilogueHalfOp::wrap(half);
        let source = CuteTmaSmemViewOp::wrap(store.source_view);
        require_same_epilogue_carrier(
            ctx,
            source.base(ctx),
            half.half_base(ctx),
            "each semantic TMA store source base must come from the matching epilogue half through only the compiler's SSA carrier adapter",
        )?;
        require_same_epilogue_carrier(
            ctx,
            source.capacity(ctx),
            half.capacity(ctx),
            "each semantic TMA store source capacity must come from the matching epilogue half through only the compiler's SSA carrier adapter",
        )?;
    }
    if store_zero.operation.deref(ctx).get_operand(1).get_type(ctx)
        != store_one.operation.deref(ctx).get_operand(1).get_type(ctx)
    {
        return Err(invalid(
            "both epilogue stores must target the same typed output tile",
        ));
    }
    let destination_zero = CuteTmaGmemViewOp::wrap(store_zero.destination_view);
    let destination_one = CuteTmaGmemViewOp::wrap(store_one.destination_view);
    require_same_epilogue_carrier(
        ctx,
        destination_one.descriptor(ctx),
        destination_zero.descriptor(ctx),
        "both epilogue halves must target the same TMA destination descriptor",
    )?;
    let semantic_zero = CuteTmaStore2dSemanticOp::wrap(store_zero.operation);
    let semantic_one = CuteTmaStore2dSemanticOp::wrap(store_one.operation);
    require_same_epilogue_carrier(
        ctx,
        semantic_one.tile_row(ctx),
        semantic_zero.tile_row(ctx),
        "both epilogue halves must target the same output tile row",
    )?;
    require_next_epilogue_column(
        ctx,
        semantic_zero.tile_column(ctx),
        semantic_one.tile_column(ctx),
    )?;

    let issuer_range = require_linear_protocol_order(
        ctx,
        "epilogue TMA issuer protocol",
        &[
            half_zero,
            store_zero.operation,
            half_one,
            store_one.operation,
            commit,
        ],
    )?;
    for operation in issuer_range {
        let opid = Operation::get_opid(operation, ctx);
        let allowed_tma_view = opid == CuteTmaGmemViewOp::get_opid_static()
            || opid == CuteTmaSmemViewOp::get_opid_static();
        let expected = [
            half_zero,
            store_zero.operation,
            half_one,
            store_one.operation,
            commit,
        ]
        .contains(&operation);
        if !allowed_tma_view && !expected && !is_epilogue_issuer_plumbing(ctx, operation) {
            return Err(invalid(format!(
                "epilogue TMA issuer protocol has interposed operation `{opid}`"
            )));
        }
    }

    let acquire_pipeline = CuteTmaStoreAcquireOp::wrap(acquire).pipeline(ctx);
    let commit_pipeline = CuteTmaStoreCommitOp::wrap(commit).pipeline(ctx);
    let tail_pipeline = CuteTmaStoreTailOp::wrap(tail).pipeline(ctx);
    if acquire_pipeline != commit_pipeline || acquire_pipeline != tail_pipeline {
        return Err(invalid(
            "epilogue acquire, commit, and tail must carry the same TMA store-pipeline facts",
        ));
    }

    require_epilogue_cfg_links(
        ctx,
        acquire,
        reusable_sync,
        ready_sync,
        half_zero,
        commit,
        tail,
    )
}
