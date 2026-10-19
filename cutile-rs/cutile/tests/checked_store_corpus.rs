/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adversarial corpus for the checked store surface (`PartitionMut::store`).
//!
//! Making stores checked moved a whole family of accesses from "unverified"
//! to "must be proven or checked", so each way of writing an index gets a
//! case here: what must be rejected outright, what must keep a real check,
//! what must leave for launch, and what must cost nothing. Together they say
//! that the safe store is neither too weak (a bad index slipping through) nor
//! too strong (a provable index paying for a check).
//!
//! Two members of the corpus live elsewhere because they are not specific to
//! stores, and are not duplicated here: permuted-axis discharge against the
//! remapped root axis, and lower-bound guards on dynamic indices, both in
//! `partition_access_soundness.rs`; cross-tensor shared-dimension discharge
//! in `inferred_axis_bounds.rs`. There is no `partition_permuted_mut`, so the
//! permuted case is only reachable through loads.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod checked_store_module {
    use cutile::core::*;

    /// A constant tile index past the end of the axis. The slab has one tile
    /// on axis 0; this writes the second.
    #[cutile::entry]
    fn constant_out_of_range<const N: i32, const BLOCK: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        p.store(t, [1i32, 0i32]);
    }

    /// A negative constant tile index. The upper comparison alone would admit
    /// it, so this exercises the lower goal on the store path.
    #[cutile::entry]
    fn constant_negative<const N: i32, const BLOCK: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        p.store(t, [0i32, -1i32]);
    }

    /// A runtime scalar index: unknown in both directions, so the store keeps
    /// a real in-kernel check.
    #[cutile::entry]
    fn runtime_scalar_index<const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
        idx: i32,
    ) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        p.store(t, [0i32, idx]);
    }

    /// A tile-block id as the index. The hardware bound on a block id counts
    /// CTA slabs, which is NOT this partition's tile count, so it proves
    /// nothing here and the store must keep its check. (The rung that once
    /// conflated the two admitted out-of-bounds stores; see the note in
    /// `checks/mod.rs`.)
    #[cutile::entry]
    fn block_id_index<const N: i32, const BLOCK: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        p.store(t, [0i32, pid.0]);
    }

    /// Mixed coordinates: a constant on one axis, an inferred loop index on
    /// the other. Each axis is decided on its own evidence.
    #[cutile::entry]
    fn mixed_constant_and_inferred<const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
    ) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles(&p, 1) {
            let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
            p.store(t, [0i32, j]);
        }
    }

    /// The same store loop against a genuinely dynamic row extent: nothing at
    /// compile time can decide the constant row `0`, so it becomes a
    /// non-empty-extent check at launch instead of an in-kernel one.
    #[cutile::entry]
    fn dynamic_row_extent<const N: i32, const BLOCK: i32>(out: &mut Tensor<f32, { [-1, N] }>) {
        let tile_shape = shape![1, BLOCK];
        let mut p = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles(&p, 1) {
            let t: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
            p.store(t, [0i32, j]);
        }
    }
}

use checked_store_module::__module_ast_self;
use cutile_compiler::compile_api::CheckPlacementCounts;

fn compile_mlir(
    name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> Result<String, String> {
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    common::compile_to_ir(
        __module_ast_self,
        "checked_store_module",
        name,
        &generics,
        strides,
        &[],
        &[],
        None,
        &CompileOptions::default().device_debug(false),
    )
    .map_err(|err| err.to_string())
}

fn artifacts(
    name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> cutile_compiler::compile_api::CompileArtifacts {
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    cutile_compiler::compile_api::KernelCompiler::new(
        __module_ast_self,
        "checked_store_module",
        name,
    )
    .target("sm_120")
    .generics(generics)
    .strides(strides)
    .options(CompileOptions::default().device_debug(false))
    .compile()
    .unwrap_or_else(|e| panic!("compile {name}: {e}"))
}

/// `(discharged, hoisted, in_place)` for one kernel.
fn counts(name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> (u32, u32, u32) {
    let CheckPlacementCounts {
        discharged,
        hoisted,
        in_place,
    } = artifacts(name, generics, strides).check_counts();
    (discharged, hoisted, in_place)
}

const STRIDES: &[(&str, &[i32])] = &[("out", &[256, 1])];

// An index the compiler can prove is out of range is a compile error, not a
// runtime trap — the same treatment a constant out-of-range subscript gets.
#[test]
fn constant_past_the_end_is_rejected() {
    common::with_test_stack(|| {
        let err = compile_mlir("constant_out_of_range", &["256", "64"], STRIDES)
            .expect_err("a constant past the end of the axis must not compile");
        assert!(
            err.contains("Bounds check failed") || err.contains("out of bounds"),
            "expected an out-of-range diagnostic, got: {err}"
        );
    });
}

// The lower goal has to be enforced on the store path too: a negative index
// passes any upper comparison.
#[test]
fn constant_negative_index_is_rejected() {
    common::with_test_stack(|| {
        let err = compile_mlir("constant_negative", &["256", "64"], STRIDES)
            .expect_err("a negative constant index must not compile");
        assert!(
            err.contains("Bounds check failed")
                || err.contains("0 <=")
                || err.to_lowercase().contains("negative"),
            "expected a lower-bound diagnostic, got: {err}"
        );
    });
}

// Nothing bounds a runtime scalar, so the store must keep a real check —
// guarded in both directions.
#[test]
fn runtime_scalar_index_keeps_a_two_sided_check() {
    common::with_test_stack(|| {
        let mlir = compile_mlir("runtime_scalar_index", &["256", "64"], STRIDES)
            .expect("compile runtime_scalar_index");
        assert!(
            mlir.contains("partition access out of bounds"),
            "an unprovable store index must keep its check:\n{mlir}"
        );
        assert!(
            mlir.contains("greater_than_or_equal"),
            "a runtime store index needs a lower-bound guard, found none:\n{mlir}"
        );
        assert_eq!(
            counts("runtime_scalar_index", &["256", "64"], STRIDES),
            (1, 0, 1),
            "expected the constant row to discharge and the runtime column to \
             stay in the kernel"
        );
    });
}

// A block id bounds the launch grid, not this partition's tiles — the id
// alone is still not a proof. The axiom rung may discharge the in-kernel
// check ONLY by staking a launch check that the grid fits the view's tile
// count, verified against the exact grid the kernel launches with; a grid
// wider than the tiles is rejected before any GPU work. (This pin's earlier
// form asserted the check stayed in the kernel, the sound-but-conservative
// posture before the rung existed.)
#[test]
fn block_id_index_discharges_by_staking_a_grid_check() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;
        let mlir = compile_mlir("block_id_index", &["256", "64"], STRIDES)
            .expect("compile block_id_index");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "the block-id store check should leave the kernel:\n{mlir}"
        );
        let compiled = artifacts("block_id_index", &["256", "64"], STRIDES);
        let checks = compiled.launch_checks();
        assert!(
            format!("{checks:?}").contains("num_tile_blocks(0)"),
            "the discharge must stake a claim on the launch grid: {checks:?}"
        );
        // The view has ceil(256/64) = 4 column tiles; the indexed axis is the
        // grid's x axis. Four blocks launch; five must not.
        let roots = [vec![256i32, 256]];
        let views = [vec![1i32, 256]];
        assert!(
            validate_launch_checks(checks, &roots, &views, (4, 1, 1)).is_ok(),
            "a grid inside the view's tile count must launch: {checks:?}"
        );
        assert!(
            validate_launch_checks(checks, &roots, &views, (5, 1, 1)).is_err(),
            "a grid wider than the view's tile count must be rejected: {checks:?}"
        );
    });
}

// Constant and inferred coordinates are decided independently, and both are
// provable here: a checked store costs nothing.
#[test]
fn mixed_constant_and_inferred_coordinates_fully_discharge() {
    common::with_test_stack(|| {
        assert_eq!(
            counts("mixed_constant_and_inferred", &["256", "64"], STRIDES),
            (2, 0, 0),
            "expected the constant row and the inferred column to discharge"
        );
        let mlir = compile_mlir("mixed_constant_and_inferred", &["256", "64"], STRIDES)
            .expect("compile mixed_constant_and_inferred");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "a fully provable checked store must emit no check:\n{mlir}"
        );
    });
}

// With a dynamic row extent the constant row cannot be decided at compile
// time, but it reduces to "this axis is non-empty" — a launch-known fact. It
// leaves the kernel entirely, and rejects an empty slab at launch.
#[test]
fn dynamic_row_extent_moves_the_check_to_launch() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;
        let mlir = compile_mlir("dynamic_row_extent", &["256", "64"], STRIDES)
            .expect("compile dynamic_row_extent");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "the non-empty-extent obligation should leave the kernel:\n{mlir}"
        );
        let compiled = artifacts("dynamic_row_extent", &["256", "64"], STRIDES);
        let checks = compiled.launch_checks();
        assert!(
            !checks.is_empty(),
            "expected a launch check for the dynamic row extent, got none"
        );
        // A `&mut` parameter is slabbed, so the check is over the per-CTA view,
        // not the root — an empty slab must be rejected even though the root
        // tensor is non-empty.
        assert!(
            format!("{checks:?}").contains("ViewExtent"),
            "a &mut param's extent check must be in the view (slab) frame: {checks:?}"
        );
        let roots = [vec![256i32, 256]];
        assert!(
            validate_launch_checks(checks, &roots, &[vec![0i32, 256]], (1, 1, 1)).is_err(),
            "an empty slab must be rejected at launch"
        );
        assert!(
            validate_launch_checks(checks, &roots, &[vec![1i32, 256]], (1, 1, 1)).is_ok(),
            "a non-empty slab must be accepted at launch"
        );
    });
}
