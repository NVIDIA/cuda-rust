/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Access-safety checks for partition loads and stores: the one place a
//! coordinate is proven inside its partition grid.
//!
//! Per axis, safety is the pair of linear goals `0 <= i` and `i * t < e`
//! (see [`goals`]). Every access check forms that pair through one path —
//! [`goals::AxisGoals`], built by `form_axis_goals` — and then tries the
//! same ladder of proof sources, top-down, stopping at the first rung that
//! decides it:
//!
//! 1. **Provenance match** — the index is branded by the axis it indexes
//!    (a `with_bounds` binding, a partition-axis iterand), so both goals
//!    hold by construction (`brand_of`, `discharge_by_axis_provenance`).
//! 2. **Static fold** — a statically-known index range against a
//!    statically-known tile grid decides both goals; a violation is a
//!    compile error (`fold_static`). A `tile_block_id(k)` coordinate is
//!    decided here too, by the grid axiom: the id is grid-bounded by the
//!    execution model, and the grid-vs-tile-count claim moves to a launch
//!    check (`discharge_by_block_id_axiom`).
//! 3. **Declared-fact entailment** — the goal follows at JIT time from
//!    launch-verified preconditions ([`lower_obligation`],
//!    [`resolve_dim_eq`]).
//! 4. **Launch hoist** — the goal's operands are launch-known, so it
//!    evacuates to a host-side check before the kernel runs (also via
//!    [`lower_obligation`]).
//! 5. **Residual device placement** — an in-kernel assert, hoisted to the
//!    outermost loop preheader whose frame decides it; gated by
//!    [`deny_residual_check`] ([`placement`]).
//!
//! The two access families are two *policies* over that one ladder, not
//! two frameworks. The plain family ([`compile_check_partition_access`],
//! behind `Partition::load`) draws provenance from partition-axis
//! iteration and declared root-dimension equalities, and lets an undecided
//! goal fall to rung 5. The bounded family
//! ([`compile_check_bounded_partition_access`], behind
//! `BoundedPartition{,Mut}` accesses) draws provenance from `with_bounds`
//! brands, reaches rung 4 through the zero-coordinate extent obligation,
//! and treats an undecided goal as a compile error — never a device check.
//!
//! [`lower_obligation`]: CUDATileFunctionCompiler::lower_obligation
//! [`resolve_dim_eq`]: CUDATileFunctionCompiler::resolve_dim_eq
//! [`deny_residual_check`]: CUDATileFunctionCompiler::deny_residual_check
//! [`compile_check_bounded_partition_access`]: CUDATileFunctionCompiler::compile_check_bounded_partition_access
//! [`compile_check_partition_access`]: CUDATileFunctionCompiler::compile_check_partition_access

mod goals;
mod placement;

use syn::spanned::Spanned;
use syn::ExprCall;

use quote::ToTokens;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue};
use super::shared_types::Kind;
use crate::error::JITError;
use crate::generics::GenericVars;

use cutile_ir::ir::{BlockId, Module};

impl<'m> CUDATileFunctionCompiler<'m> {
    /// One goal pair decided without any in-kernel cost.
    fn count_discharged(&self) {
        self.check_stats
            .discharged
            .set(self.check_stats.discharged.get() + 1);
    }

    /// Compiles a `check_partition_access` compiler_op call: the per-axis
    /// access check guarding `Partition::load`. Walks the ladder per
    /// coordinate axis; what no proof source decides is placed as a
    /// residual device check.
    pub(super) fn compile_check_partition_access(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let mut args =
            self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
        let partition_value = args.remove(0);
        let index_value = args.remove(0);
        if partition_value.kind != Kind::StructuredType {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "expected a structured or primitive type for first argument of `{}`, got {:?}",
                    call_expr.to_token_stream(),
                    partition_value.kind
                ),
            );
        }
        if index_value.kind != Kind::Compound {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "Unexpected kind for arg 1 in {}",
                    call_expr.to_token_stream()
                ),
            );
        }
        let (static_tile, static_shape, dim_map) =
            self.partition_static_geometry(&partition_value, &call_expr.span())?;

        // The runtime shape operands, for residual checks over dynamic
        // extents.
        let tensor_shape_value = partition_value
            .get_type_meta_field("tensor_view.shape()")
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    "Failed to obtain type meta field tensor_view.shape().",
                )
            })?;
        let Some(tensor_shape_values) = tensor_shape_value.fields.as_ref() else {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected fields for tensor shape expression.",
            );
        };
        let Some(shape_dims) = tensor_shape_values.get("dims") else {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected dims field for shape expression.",
            );
        };
        let Some(dynamic_shape) = shape_dims.values.as_ref() else {
            return self.jit_error_result(&call_expr.span(), "expected a compound (tuple) value");
        };

        let Some(indexes) = index_value.values.as_ref() else {
            return self.jit_error_result(&call_expr.span(), "expected a compound (tuple) value");
        };
        let len = static_tile.len();
        if len != indexes.len() || len != static_shape.len() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "Unexpected tile ({}), shape ({}), or index ({}) length mismatch.",
                    len,
                    static_shape.len(),
                    indexes.len()
                ),
            );
        }
        for (axis, index) in indexes.iter().enumerate() {
            let axis_goals = self.form_axis_goals(
                &partition_value,
                &static_tile,
                &static_shape,
                &dim_map,
                axis,
                index,
            );
            // Rungs 1 and 2 are proofs, not placements — but the disabled
            // policy skips them anyway: that build is the differential
            // harness's semantic reference, and a reference that inherits
            // the very proofs under audit cannot catch one that is wrong
            // (2026-08-12 review, S2). The bounded family keeps its rungs —
            // its undischarged checks are compile errors, so it has no
            // device placement to fall back to.
            if self.check_opts.discharge_proofs {
                // Rung 1/3: axis provenance — a same-view iterand, or a
                // foreign iterand whose axis a declared root-dimension
                // equality relates to this one.
                if self.discharge_by_axis_provenance(&axis_goals, &partition_value) {
                    self.count_discharged();
                    continue;
                }
                // Rung 2: static fold.
                match goals::fold_static(&axis_goals) {
                    Some(Ok(())) => {
                        self.count_discharged();
                        continue;
                    }
                    Some(Err(violation)) => {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!(
                                "Bounds check failed: 0 <= {} && {} < {}",
                                violation.min, violation.max, violation.num_tiles
                            ),
                        );
                    }
                    None => {}
                }
                // Rung 2.5: the block-id axiom — `tile_block_id(k)` is
                // grid-bounded by the execution model; the residual
                // grid-vs-tile-count claim moves to a launch check
                // (goals.rs, discharge_by_block_id_axiom).
                if self.discharge_by_block_id_axiom(&axis_goals, &partition_value) {
                    self.count_discharged();
                    continue;
                }
            }
            // Rung 3/4: a constant `[0, 0]` coordinate against a dynamic
            // extent reduces to `extent > 0`, a launch-known predicate.
            // Gated on launch relocation because the check leaves the
            // kernel. Gated here, not inside the rung: the bounded family
            // shares it and MUST keep it under every policy, since its
            // undischarged checks are compile errors rather than device
            // placements.
            if self.check_opts.relocate_to_launch
                && self.hoist_zero_coordinate_nonempty_extent(&axis_goals, &partition_value)
            {
                self.count_discharged();
                continue;
            }
            // Rung 5: residual device placement.
            let dynamic_extent = if axis_goals.static_extent.is_none() {
                Some(self.resolve_dynamic_extent(
                    &axis_goals,
                    &static_shape,
                    dynamic_shape,
                    call_expr,
                )?)
            } else {
                None
            };
            self.place_residual_check(
                module,
                block_id,
                &axis_goals,
                dynamic_extent,
                generic_vars,
                ctx,
                &call_expr.span(),
            )?;
        }
        Ok(None)
    }

    /// Compiles a `check_bounded_partition_access` compiler_op call: the
    /// per-axis access check for a `coord(...)` coordinate into a bounded
    /// partition (`BoundedPartition{,Mut}`). Each axis discharges by brand
    /// provenance, the static fold, or the launch-hoisted non-empty-extent
    /// obligation; an unresolved coordinate is a compile error, never a
    /// device check.
    pub(super) fn compile_check_bounded_partition_access(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 2 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`check_bounded_partition_access` expects 2 arguments, got {}",
                    call_expr.args.len()
                ),
            );
        }
        let mut args =
            self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
        let partition = args.remove(0);
        let coord = args.remove(0);
        let Some(bound_axes) = partition.bounded_axes.as_ref() else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "bounded partition load requires bounds established by `Partition::with_bounds`",
            );
        };
        let Some(fields) = coord.fields.as_ref() else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "bounded partition load requires a coordinate created by `coord(...)`",
            );
        };
        let Some(coords) = fields.get("coords") else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "coordinate is missing its metadata",
            );
        };
        let Some(coord_values) = coords.values.as_ref() else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "coordinates must be a compound value",
            );
        };
        if coord_values.len() != bound_axes.len() {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                &format!(
                    "coordinate rank {} does not match bounded partition rank {}",
                    coord_values.len(),
                    bound_axes.len()
                ),
            );
        }
        let (static_tile, static_shape, dim_map) =
            self.partition_static_geometry(&partition, &call_expr.args[0].span())?;
        for (axis, (coord_value, bound_origin)) in
            coord_values.iter().zip(bound_axes.iter()).enumerate()
        {
            let axis_goals = self.form_axis_goals(
                &partition,
                &static_tile,
                &static_shape,
                &dim_map,
                axis,
                coord_value,
            );
            // Rung 1: brand provenance.
            match goals::brand_of(&axis_goals, bound_origin) {
                Some(goals::Brand::Matched) => {
                    self.count_discharged();
                    continue;
                }
                Some(goals::Brand::Foreign) => {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!(
                            "bounded partition coordinate axis {axis} was produced by a different dimension"
                        ),
                    );
                }
                None => {}
            }
            // Rung 2: static fold.
            match goals::fold_static(&axis_goals) {
                Some(Ok(())) => {
                    self.count_discharged();
                    continue;
                }
                Some(Err(violation)) => {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!(
                            "bounded partition coordinate axis {axis}: constant range [{}, {}] is not within the {}-tile grid",
                            violation.min, violation.max, violation.num_tiles
                        ),
                    );
                }
                None => {}
            }
            // Rung 3/4: a constant `[0, 0]` coordinate against a dynamic
            // extent reduces to `extent > 0`, a launch-known predicate.
            if self.hoist_zero_coordinate_nonempty_extent(&axis_goals, &partition) {
                self.count_discharged();
                continue;
            }
            // NOTE: there is deliberately NO tile-block-id discharge rung
            // here. Two different quantities have both been called "the
            // partition grid", and conflating them is unsound:
            //
            //   * `NumTileBlocks(k)` (the launch grid) counts CTA *slabs*:
            //     `div_ceil(root_shape, slab_shape)` (`Partition::grid`),
            //     which is what `validate_grids` checks at launch.
            //   * the bound this access needs is the tile count *within*
            //     the kernel-visible view: `ceil(static_shape / static_tile)`
            //     computed above — for a `&mut Tensor` param that view is
            //     one slab, so this counts tiles inside a slab.
            //
            // These are unrelated, so the hardware axiom `TileBlockId(k) <
            // NumTileBlocks(k)` does NOT bound this access. A previous rung
            // (reverted) formed that axiom *as* its obligation and proved it
            // against itself — a tautology that discharged every block-id
            // access and admitted out-of-bounds stores (reproduced on
            // hardware: CTAs writing other CTAs' rows).
            //
            // The real fix must form the actual proposition
            // `TileBlockId(axis) < num_partitions(view, axis)` and obtain
            // `NumTileBlocks(axis) == num_partitions(view, axis)` as a
            // *verified* `Launch` obligation, never as an assumption.
            // See `.internal/tasks/in_progress/check_hoisting/OBLIGATION_SYSTEM_DESIGN.md`.
            return self.jit_error_result(
                &call_expr.args[1].span(),
                &format!(
                    "bounded partition coordinate axis {axis} must come from iterating the matching dimension or be a constant within the axis's static tile grid"
                ),
            );
        }
        Ok(None)
    }
}
