/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static scheduling kept as five small CuTe operations.
//!
//! A physical CUDA block can process more than one logical output tile:
//!
//! ```text
//! new_1d
//!   │ current = blockIdx.x, stride = gridDim.x
//!   ▼
//! current ── has_work? ── yes ──► current_tile ──► (linear, m, n, batch)
//!   ▲                                                        │
//!   └──────────── advance(current, stride) ◄──────────────────┘
//! ```
//!
//! The loop carries only ordinary `u64` values. `work_tile` is a short-lived
//! compiler handle, so this layer adds no runtime scheduler object. A backend
//! continuation materializes the required special-register reads and integer
//! arithmetic.

use pliron::builtin::{
    op_interfaces::{NOpdsInterface, NResultsInterface},
    types::{IntegerType, Signedness},
};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::CuteTileGridAttr;
use crate::types::CuteWorkTileType;

fn u64_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Unsigned).into()
}

fn bool_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 1, Signedness::Signless).into()
}

fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

fn is_bool(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 1)
}

fn work_tile_of(ctx: &Context, value: Value) -> Option<CuteWorkTileType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteWorkTileType>()
        .cloned()
}

/// Check whether the current linear number is inside the logical tile grid.
///
/// Keeping this separate from [`CuteSchedulerCurrentOp`] matters: the tile's
/// M/N/batch arithmetic remains inside the true loop path. A finished block
/// performs only the bounds check.
#[pliron_op(
    name = "cute.scheduler_has_work",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (has_work_grid: CuteTileGridAttr)
)]
pub struct CuteSchedulerHasWorkOp;

impl CuteSchedulerHasWorkOp {
    pub fn new(ctx: &mut Context, current: Value, grid: CuteTileGridAttr) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![bool_type(ctx)],
                vec![current],
                vec![],
                0,
            ),
        };
        operation.set_attr_has_work_grid(ctx, grid);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Current linear tile number carried by the loop.
    #[must_use]
    pub fn current(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// True while `current` is smaller than the static tile count.
    #[must_use]
    pub fn has_work(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Static logical output grid used by the bounds check.
    #[must_use]
    pub fn tile_grid(&self, ctx: &Context) -> Option<CuteTileGridAttr> {
        self.get_attr_has_work_grid(ctx).map(|grid| *grid)
    }
}

impl Verify for CuteSchedulerHasWorkOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scheduler_has_work needs 1 operand and 1 result"
            );
        }
        if !is_u64(ctx, self.current(ctx)) || !is_bool(ctx, self.has_work(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.scheduler_has_work needs a u64 current value and an i1 result"
            );
        }
        let Some(grid) = self.tile_grid(ctx) else {
            return verify_err!(op.loc(), "cute.scheduler_has_work must carry a tile grid");
        };
        grid.verify(ctx)
    }
}

/// Start this block's one-dimensional persistent schedule.
///
/// Backend lowering reads `blockIdx.x` into `current` and `gridDim.x` into
/// `stride`.
/// Keeping those values as `u64` means an ordinary loop can carry them after
/// every semantic scheduler operation has been erased.
#[pliron_op(
    name = "cute.scheduler_new_1d",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<2>],
    attributes = (new_1d_grid: CuteTileGridAttr)
)]
pub struct CuteSchedulerNew1dOp;

impl CuteSchedulerNew1dOp {
    pub fn new(ctx: &mut Context, grid: CuteTileGridAttr) -> Self {
        let ty = u64_type(ctx);
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty, ty],
                vec![],
                vec![],
                0,
            ),
        };
        operation.set_attr_new_1d_grid(ctx, grid);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Linear tile number assigned to this physical block first.
    #[must_use]
    pub fn current(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Number of physical blocks between this block's logical tiles.
    #[must_use]
    pub fn stride(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Static logical output grid described by this schedule.
    #[must_use]
    pub fn tile_grid(&self, ctx: &Context) -> Option<CuteTileGridAttr> {
        self.get_attr_new_1d_grid(ctx).map(|grid| *grid)
    }
}

impl Verify for CuteSchedulerNew1dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "cute.scheduler_new_1d needs 0 operands and 2 results"
            );
        }
        if !is_u64(ctx, self.current(ctx)) || !is_u64(ctx, self.stride(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.scheduler_new_1d current and stride must be unsigned 64-bit integers"
            );
        }
        let Some(grid) = self.tile_grid(ctx) else {
            return verify_err!(op.loc(), "cute.scheduler_new_1d must carry a tile grid");
        };
        grid.verify(ctx)
    }
}

/// Expose the current linear number as a semantic work-tile handle.
///
/// Call this only on the true result of [`CuteSchedulerHasWorkOp`]. The handle
/// carries no runtime bits; backend lowering resolves it back to `current`.
#[pliron_op(
    name = "cute.scheduler_current",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (current_grid: CuteTileGridAttr)
)]
pub struct CuteSchedulerCurrentOp;

impl CuteSchedulerCurrentOp {
    pub fn new(ctx: &mut Context, current: Value, grid: CuteTileGridAttr) -> Self {
        let tile: TypeHandle = CuteWorkTileType::get(ctx, grid).into();
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![tile],
                vec![current],
                vec![],
                0,
            ),
        };
        operation.set_attr_current_grid(ctx, grid);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Current linear tile number carried by the loop.
    #[must_use]
    pub fn current(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Semantic tile handle created on the `has_work` path.
    #[must_use]
    pub fn work_tile(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Static logical output grid used by the bounds check and tile mapping.
    #[must_use]
    pub fn tile_grid(&self, ctx: &Context) -> Option<CuteTileGridAttr> {
        self.get_attr_current_grid(ctx).map(|grid| *grid)
    }
}

impl Verify for CuteSchedulerCurrentOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scheduler_current needs 1 operand and 1 result"
            );
        }
        if !is_u64(ctx, self.current(ctx)) {
            return verify_err!(op.loc(), "cute.scheduler_current input must be a u64 value");
        }
        let Some(grid) = self.tile_grid(ctx) else {
            return verify_err!(op.loc(), "cute.scheduler_current must carry a tile grid");
        };
        grid.verify(ctx)?;
        let Some(tile) = work_tile_of(ctx, self.work_tile(ctx)) else {
            return verify_err!(
                op.loc(),
                "cute.scheduler_current result must be a work tile"
            );
        };
        if tile.grid != grid {
            return verify_err!(
                op.loc(),
                "cute.scheduler_current work tile must use the operation's tile grid"
            );
        }
        tile.verify(ctx)
    }
}

/// Map one valid linear tile to M-fastest `(m, n, batch)` coordinates.
///
/// Backend lowering uses the same two divide and multiply/subtract pairs as
/// the Rust scheduler. `linear` is returned unchanged.
#[pliron_op(
    name = "cute.work_tile_coordinates",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>]
)]
pub struct CuteWorkTileCoordinatesOp;

impl CuteWorkTileCoordinatesOp {
    pub fn new(ctx: &mut Context, tile: Value) -> Self {
        let ty = u64_type(ctx);
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty, ty, ty, ty],
                vec![tile],
                vec![],
                0,
            ),
        }
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn work_tile(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn linear(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn m_tile(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    #[must_use]
    pub fn n_tile(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(2)
    }

    #[must_use]
    pub fn batch(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(3)
    }
}

impl Verify for CuteWorkTileCoordinatesOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 4 {
            return verify_err!(
                op.loc(),
                "cute.work_tile_coordinates needs 1 operand and 4 results"
            );
        }
        let Some(tile) = work_tile_of(ctx, self.work_tile(ctx)) else {
            return verify_err!(
                op.loc(),
                "cute.work_tile_coordinates input must be a work tile"
            );
        };
        tile.verify(ctx)?;
        if ![
            self.linear(ctx),
            self.m_tile(ctx),
            self.n_tile(ctx),
            self.batch(ctx),
        ]
        .into_iter()
        .all(|value| is_u64(ctx, value))
        {
            return verify_err!(
                op.loc(),
                "cute.work_tile_coordinates results must be unsigned 64-bit integers"
            );
        }
        Ok(())
    }
}

/// Move this block to its next logical tile.
///
/// The result is `current.saturating_add(stride)`. Saturation prevents a
/// wrapped tile number from looking valid again.
#[pliron_op(
    name = "cute.scheduler_advance",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
    attributes = (advance_grid: CuteTileGridAttr)
)]
pub struct CuteSchedulerAdvanceOp;

impl CuteSchedulerAdvanceOp {
    pub fn new(ctx: &mut Context, current: Value, stride: Value, grid: CuteTileGridAttr) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![u64_type(ctx)],
                vec![current, stride],
                vec![],
                0,
            ),
        };
        operation.set_attr_advance_grid(ctx, grid);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn current(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn stride(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn next(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Static logical output grid described by this schedule.
    #[must_use]
    pub fn tile_grid(&self, ctx: &Context) -> Option<CuteTileGridAttr> {
        self.get_attr_advance_grid(ctx).map(|grid| *grid)
    }
}

impl Verify for CuteSchedulerAdvanceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scheduler_advance needs 2 operands and 1 result"
            );
        }
        if !is_u64(ctx, self.current(ctx))
            || !is_u64(ctx, self.stride(ctx))
            || !is_u64(ctx, self.next(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.scheduler_advance current, stride, and next must be unsigned 64-bit integers"
            );
        }
        let Some(grid) = self.tile_grid(ctx) else {
            return verify_err!(op.loc(), "cute.scheduler_advance must carry a tile grid");
        };
        grid.verify(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirUndefOp;

    fn result(ctx: &Context, op: Ptr<Operation>, index: usize) -> Value {
        op.deref(ctx).get_result(index)
    }

    fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
        MirUndefOp::new(ctx, ty)
            .get_operation()
            .deref(ctx)
            .get_result(0)
    }

    #[test]
    fn persistent_schedule_stays_five_composable_steps() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let grid = CuteTileGridAttr::new(16, 16, 1);

        let start = CuteSchedulerNew1dOp::new(&mut ctx, grid);
        assert!(start.verify(&ctx).is_ok());
        let current = start.current(&ctx);
        let stride = start.stride(&ctx);

        let has_work = CuteSchedulerHasWorkOp::new(&mut ctx, current, grid);
        assert!(has_work.verify(&ctx).is_ok());

        let selected = CuteSchedulerCurrentOp::new(&mut ctx, current, grid);
        assert!(selected.verify(&ctx).is_ok());
        let tile = selected.work_tile(&ctx);

        let coordinates = CuteWorkTileCoordinatesOp::new(&mut ctx, tile);
        assert!(coordinates.verify(&ctx).is_ok());

        let advance = CuteSchedulerAdvanceOp::new(&mut ctx, current, stride, grid);
        assert!(advance.verify(&ctx).is_ok());
        assert!(is_u64(&ctx, advance.next(&ctx)));
    }

    #[test]
    fn current_rejects_a_work_tile_for_a_different_grid() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let u64_ty = u64_type(&ctx);
        let current = undef(&mut ctx, u64_ty);
        let grid = CuteTileGridAttr::new(16, 16, 1);
        let selected = CuteSchedulerCurrentOp::new(&mut ctx, current, grid);
        selected.set_attr_current_grid(&ctx, CuteTileGridAttr::new(8, 32, 1));

        assert!(
            selected
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("operation's tile grid")
        );
    }

    #[test]
    fn coordinates_reject_an_ordinary_runtime_value() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let u64_ty = u64_type(&ctx);
        let scalar = undef(&mut ctx, u64_ty);
        let coordinates = CuteWorkTileCoordinatesOp::new(&mut ctx, scalar);

        assert!(
            coordinates
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("input must be a work tile")
        );
    }

    #[test]
    fn grid_checks_zero_and_overflow() {
        let ctx = Context::new();
        assert!(CuteTileGridAttr::new(3, 2, 1).verify(&ctx).is_ok());
        assert_eq!(CuteTileGridAttr::new(3, 2, 1).total_tiles(), Some(6));
        assert!(CuteTileGridAttr::new(0, 2, 1).verify(&ctx).is_err());
        assert!(CuteTileGridAttr::new(u64::MAX, 2, 1).verify(&ctx).is_err());
    }

    #[test]
    fn result_order_matches_the_visual_contract() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let grid = CuteTileGridAttr::new(3, 2, 1);
        let start = CuteSchedulerNew1dOp::new(&mut ctx, grid);
        let current = start.current(&ctx);
        let has_work = CuteSchedulerHasWorkOp::new(&mut ctx, current, grid);
        let selected = CuteSchedulerCurrentOp::new(&mut ctx, current, grid);
        let tile = selected.work_tile(&ctx);
        let coordinates = CuteWorkTileCoordinatesOp::new(&mut ctx, tile);

        assert_eq!(
            coordinates.linear(&ctx),
            result(&ctx, coordinates.get_operation(), 0)
        );
        assert_eq!(
            coordinates.m_tile(&ctx),
            result(&ctx, coordinates.get_operation(), 1)
        );
        assert_eq!(
            coordinates.n_tile(&ctx),
            result(&ctx, coordinates.get_operation(), 2)
        );
        assert_eq!(
            coordinates.batch(&ctx),
            result(&ctx, coordinates.get_operation(), 3)
        );
        assert!(is_bool(&ctx, has_work.has_work(&ctx)));
    }
}
