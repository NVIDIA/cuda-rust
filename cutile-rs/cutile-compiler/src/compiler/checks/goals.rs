/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-axis goal formation, and the proof sources that decide goals at JIT
//! time.
//!
//! For a coordinate `i` into logical axis `axis` of a partition whose tile
//! extent there is `t` and whose (remapped) tensor extent is `e`, access
//! safety is exactly the pair of linear goals
//!
//! ```text
//!     0 <= i          (lower)
//!     i * t < e       (upper)
//! ```
//!
//! The upper goal is the linearization `i < ceil(e/t)  <=>  i*t < e` for
//! `t >= 1`, which keeps `ceil` out of the predicate term language: every
//! goal stays linear, so static ranges, declared facts, and launch metadata
//! can all decide it. [`AxisGoals`] is the formed pair — the index, the
//! tile extent, and the identity of the extent bounding it. The functions
//! here are proof *sources* over that one representation: each either
//! decides both goals or leaves them open for the next rung, in the ladder
//! order documented in [`super`]. What no source decides falls to residual
//! placement (`super::placement`).

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::ExprCall;

use super::super::_function::CUDATileFunctionCompiler;
use super::super::_value::{DimOrigin, TileRustValue};
use crate::error::JITError;

/// One coordinate axis of a partition access, resolved: the two linear
/// goals `0 <= i` and `i * t < e` in data form.
///
/// The extent identity is resolved through the partition's dim-map at
/// formation time: `extent_axis = dim_map[axis]` is the tensor axis whose
/// extent bounds this coordinate, and every proof source must speak about
/// that axis — a permuted partition's logical axis `axis` is bounded by
/// root axis `dim_map[axis]`, never by `axis` itself (2026-08-04 review,
/// finding Ask3-2).
pub(super) struct AxisGoals<'v> {
    /// Logical coordinate position; diagnostics name this axis.
    pub(super) axis: usize,
    /// `dim_map[axis]`: the tensor axis whose extent `e` bounds the access.
    pub(super) extent_axis: usize,
    /// Static tile extent `t >= 1`.
    pub(super) tile: i32,
    /// `e` when a static source pins it: the partition view type when it
    /// kept the extent, otherwise the declared parameter shape — which the
    /// generated launcher verifies against the real tensor before the
    /// kernel runs (SC2), so it is evidence, not trust. `None` means the
    /// extent is genuinely runtime.
    pub(super) static_extent: Option<i32>,
    /// The coordinate `i`, with whatever facts it carries: constant bounds,
    /// a symbolic term, and axis provenance.
    pub(super) index: &'v TileRustValue,
}

/// A statically provable goal violation, produced by [`fold_static`]. The
/// calling family turns it into its own compile error.
pub(super) struct StaticViolation {
    /// The index's smallest possible value (decides the lower goal).
    pub(super) min: i64,
    /// The index's largest possible value (decides the upper goal).
    pub(super) max: i64,
    /// `ceil(e / t)`: the number of tiles the axis actually has.
    pub(super) num_tiles: i64,
}

/// What the brand rung concludes about a coordinate (bounded family).
pub(super) enum Brand {
    /// Branded by exactly the `Dim` that `with_bounds` bound to this axis:
    /// the binding check already related that `Dim` to the axis's tile
    /// count, so both goals hold by construction.
    Matched,
    /// Branded by a *different* dimension. A foreign brand is a compile
    /// error in the bounded family: the coordinate provably iterates some
    /// other axis's domain.
    Foreign,
}

/// The brand-provenance rung: compare the coordinate's origin against the
/// `Dim` recorded for this axis by `with_bounds`. `None` means the
/// coordinate is unbranded and the rung is open.
pub(super) fn brand_of(goals: &AxisGoals<'_>, bound_origin: &DimOrigin) -> Option<Brand> {
    match goals.index.index_origin.as_ref() {
        Some(origin) if origin == bound_origin => Some(Brand::Matched),
        Some(_) => Some(Brand::Foreign),
        None => None,
    }
}

/// The static-fold rung: a statically ranged index against a statically
/// pinned extent decides both goals outright. `None` leaves the rung open
/// (runtime index range or runtime extent); `Some(Err(_))` is a provable
/// violation the family must surface as a compile error, mirroring how a
/// constant out-of-range subscript is rejected rather than checked.
pub(super) fn fold_static(goals: &AxisGoals<'_>) -> Option<Result<(), StaticViolation>> {
    let bounds = goals.index.bounds?;
    let extent = goals.static_extent?;
    let num_tiles = (extent as i64 + goals.tile as i64 - 1) / goals.tile as i64;
    Some(if 0 <= bounds.start && bounds.end < num_tiles {
        Ok(())
    } else {
        Err(StaticViolation {
            min: bounds.start,
            max: bounds.end,
            num_tiles,
        })
    })
}

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Form the goal pair for one coordinate axis: resolve the extent axis
    /// through the dim-map, then attach the static extent when the view
    /// type or the declared (launch-verified) parameter shape pins it.
    ///
    /// This is the one place the goals are formed; every proof source and
    /// the residual placement consume what it returns. In particular the
    /// extent-frame question (root extent vs per-CTA slab) never reaches
    /// the rungs: the static sources used here coincide with the kernel's
    /// view by construction, and the predicate-language rungs name the
    /// extent through [`Self::extent_atom`], the single place that frame
    /// decision lives.
    pub(super) fn form_axis_goals<'v>(
        &self,
        partition: &TileRustValue,
        static_tile: &[i32],
        static_shape: &[i32],
        dim_map: &[i32],
        axis: usize,
        index: &'v TileRustValue,
    ) -> AxisGoals<'v> {
        let extent_axis = dim_map[axis] as usize;
        let static_extent = match static_shape[extent_axis] {
            -1 => self.declared_view_extent(partition, dim_map, axis),
            extent => Some(extent),
        };
        AxisGoals {
            axis,
            extent_axis,
            tile: static_tile[axis],
            static_extent,
            index,
        }
    }

    /// The axis-provenance rung (plain family): the coordinate's origin
    /// proves it iterates the very axis it indexes.
    ///
    /// Two forms of the same evidence:
    ///
    /// * Cross-tensor: the coordinate iterates axis `a` of some *other*
    ///   tensor with the same tile extent, so it is bounded by that axis's
    ///   tile count, and `dim(other, a) <= dim(target, extent_axis)` carries
    ///   the bound across — settled at JIT by a declared equality, or checked
    ///   on the host at launch. The goal is the INEQUALITY, source at most
    ///   target: a target strictly larger than the source is safe, and
    ///   stating equality instead rejected it (found 2026-08-05, in review of
    ///   `2ca6e3d`). The target side of the query is the REMAPPED axis
    ///   `extent_axis`, never the logical coordinate `axis`: for a permuted
    ///   partition the extent bounding logical axis `axis` is root axis
    ///   `dim_map[axis]` (2026-08-04 review, finding Ask3-2; regression
    ///   pinned in `partition_access_soundness.rs`).
    /// * Same-view: the coordinate came from iterating this partition's own
    ///   axis (`num_tiles(&p, axis)` provenance), so `i < ceil(e/t)` and
    ///   `0 <= i` hold by the iteration domain.
    pub(super) fn discharge_by_axis_provenance(
        &self,
        goals: &AxisGoals<'_>,
        partition: &TileRustValue,
    ) -> bool {
        if let (Some(origin), Some(target)) = (
            goals.index.partition_axis_origin.as_ref(),
            partition.tensor_origin.as_ref(),
        ) {
            // Both sides must be root-framed: the equality is stated over
            // `dim(t, a)`, which names whole tensors. For a slabbed `&mut`
            // target the access is bounded by its per-CTA extent, which a
            // declared root fact does not describe — so a match there would
            // discharge against evidence about a different quantity. The
            // source side is guarded where the provenance is minted.
            if origin.tile_dim == goals.tile
                && self.root_framed_param(target).is_some()
                && self.resolve_dim_le(
                    &origin.tensor,
                    origin.axis,
                    target,
                    goals.extent_axis,
                    goals.tile,
                )
            {
                return true;
            }
        }
        if let (
            Some(DimOrigin::PartitionAxis {
                view,
                axis,
                tile_dim,
            }),
            Some(partition_view),
        ) = (goals.index.index_origin.as_ref(), partition.value)
        {
            if *view == partition_view && *axis == goals.axis && *tile_dim == goals.tile {
                return true;
            }
        }
        false
    }

    /// The block-id axiom rung (plain family): a coordinate that IS the
    /// special register `tile_block_id(k)` is bounded by the launch grid —
    /// `0 <= bid < num_tile_blocks(k)` is the CUDA execution model, the one
    /// hardware axiom this module admits. What remains of the goal pair is
    /// `num_tile_blocks(k) <= ceil(e/t)`, whose operands are all
    /// launch-known: stated linearized (`ntb * t <= e + t - 1`, sound for
    /// `t >= 1`) over the extent atom so the frame decision stays in
    /// [`Self::extent_atom`], and routed through the shared obligation
    /// path — verified against the actual launch grid before the kernel
    /// runs. This is exactly the "verified `Launch` obligation" the
    /// `NumTileBlocks` vocabulary doc requires before the grid may be
    /// related to a tile count; the unverified form of this rung once
    /// admitted out-of-bounds stores and was reverted (2026-08-04 review).
    ///
    /// The rung accepts only the canonical bare register — one atom,
    /// coefficient 1, no constant. Arithmetic that changes the value
    /// forfeits the term and falls to the later rungs; identities the term
    /// algebra normalizes away (`+ c - c`, `* 1`) keep it, and are
    /// value-equal to the register by the ring semantics of wrapping
    /// arithmetic, so the substitution stays sound (2026-08-18 review, H1;
    /// the wrapping identity is executed on-GPU by the review's probe).
    pub(super) fn discharge_by_block_id_axiom(
        &self,
        goals: &AxisGoals<'_>,
        partition: &TileRustValue,
    ) -> bool {
        use cuda_async::predicate::{Atom, Predicate, Term};
        let Some(term) = goals.index.term.as_ref() else {
            return false;
        };
        if term.constant_part() != 0 || term.coeffs().len() != 1 {
            return false;
        }
        let Some((atom, &coeff)) = term.coeffs().iter().next() else {
            return false;
        };
        let k = match (atom, coeff) {
            (Atom::TileBlockId(k), 1) => *k,
            _ => return false,
        };
        let Some(tensor) = partition.tensor_origin.as_ref() else {
            return false;
        };
        let Some(&param) = self.param_index.get(tensor) else {
            return false;
        };
        let tile = goals.tile as i64;
        let Some(lhs) = Term::atom(Atom::NumTileBlocks(k)).mul_const(tile) else {
            return false;
        };
        let Some(rhs) =
            Term::atom(self.extent_atom(param, goals.extent_axis)).add(&Term::constant(tile - 1))
        else {
            return false;
        };
        let Some(le) = Predicate::le(&lhs, &rhs) else {
            return false;
        };
        let cause = format!(
            "num_tile_blocks({k}) <= ceil(extent({tensor}, {})/{})",
            goals.extent_axis, goals.tile
        );
        self.lower_obligation(le, cause)
    }

    /// The entailment/launch rung for a constant zero coordinate against a
    /// runtime extent: `i ∈ [0, 0]` reduces the goal pair to
    /// `extent(tensor, extent_axis) > 0`, a launch-known predicate. Routed
    /// through the shared assert-to-obligation path, it is discharged at
    /// JIT or hoisted to a host check at launch — zero in-kernel cost
    /// either way. The extent atom picks the frame (SC1): the check states
    /// the *kernel-visible* extent is non-empty, which for a `&mut` param
    /// is the slab, not the root.
    ///
    /// Both families use this rung. It costs one thing: a launch whose
    /// tensor is empty on this axis is rejected even if the access would
    /// have sat behind a false runtime condition and never executed. That
    /// trade was already accepted for branded accesses (pinned by
    /// `hoisted_check_rejects_zero_extent_at_launch`), and the two families
    /// must be able to prove the same goals from the same evidence — a
    /// discharge reachable only through `with_bounds` would become a
    /// capability lost when that annotation goes away, not a capability
    /// kept.
    pub(super) fn hoist_zero_coordinate_nonempty_extent(
        &self,
        goals: &AxisGoals<'_>,
        partition: &TileRustValue,
    ) -> bool {
        use cuda_async::predicate::{Predicate, Term};
        if goals.static_extent.is_some() {
            return false;
        }
        let Some(bounds) = goals.index.bounds else {
            return false;
        };
        if !(bounds.start == 0 && bounds.end == 0) {
            return false;
        }
        let Some(tensor) = partition.tensor_origin.as_ref() else {
            return false;
        };
        let Some(&param) = self.param_index.get(tensor) else {
            return false;
        };
        let predicate = Predicate::nonzero(Term::atom(self.extent_atom(param, goals.extent_axis)));
        let cause = format!(
            "partition access on axis {} of `{tensor}` requires a non-empty extent",
            goals.extent_axis
        );
        self.lower_obligation(predicate, cause)
    }

    /// The runtime scalar holding the extent for a goal whose static
    /// sources all came up empty: the per-axis value of the view's shape
    /// operands. Only the residual path needs it — a discharged goal never
    /// touches the shape values — so callers resolve it after the ladder,
    /// not at formation.
    pub(super) fn resolve_dynamic_extent(
        &self,
        goals: &AxisGoals<'_>,
        static_shape: &[i32],
        shape_dims: &[TileRustValue],
        call_expr: &ExprCall,
    ) -> Result<TileRustValue, JITError> {
        let span: Span = call_expr.span();
        // The shape operands hold values only for the view type's dynamic
        // dims, in axis order: count the dynamic dims through the extent
        // axis to find this axis's position among them.
        let dynamic_shape_index = static_shape
            .iter()
            .take(goals.extent_axis + 1)
            .filter(|&&dim| dim == -1)
            .count()
            .checked_sub(1)
            .ok_or_else(|| {
                self.jit_error(
                    &span,
                    "internal: dynamic partition dimension was not found in tensor shape metadata",
                )
            })?;
        shape_dims
            .get(dynamic_shape_index)
            .cloned()
            .ok_or_else(|| {
                self.jit_error(
                    &span,
                    &format!(
                        "internal: tensor shape metadata is missing dynamic dimension {dynamic_shape_index}"
                    ),
                )
            })
    }
}
