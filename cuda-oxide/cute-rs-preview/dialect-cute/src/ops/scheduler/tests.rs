/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use dialect_mir::ops::MirUndefOp;

/// Return result zero from the fixture operation for connecting a semantic chain.
fn result(ctx: &Context, op: Ptr<Operation>, index: usize) -> Value {
    op.deref(ctx).get_result(index)
}

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Scheduler construction, work test, tile selection, coordinates, and advance remain explicit operations.
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

/// Selected work-tile metadata must agree with the scheduler's static grid.
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

/// Coordinate extraction requires a semantic work-tile ghost, not an untyped scalar.
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

/// Grid dimensions must be positive and their product must fit the supported integer size.
#[test]
fn grid_checks_zero_and_overflow() {
    let ctx = Context::new();
    assert!(CuteTileGridAttr::new(3, 2, 1).verify(&ctx).is_ok());
    assert_eq!(CuteTileGridAttr::new(3, 2, 1).total_tiles(), Some(6));
    assert!(CuteTileGridAttr::new(0, 2, 1).verify(&ctx).is_err());
    assert!(CuteTileGridAttr::new(u64::MAX, 2, 1).verify(&ctx).is_err());
}

/// The coordinate ABI exposes linear, M, N, and batch results in that exact order.
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
