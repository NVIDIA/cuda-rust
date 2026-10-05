/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Half of each twin pair is deliberately the deprecated spelling: the tests
// exist to show the two place identically.
#![allow(deprecated)]

//! Ordinary `for i in 0..num_tiles(&p, a)` loops must prove their accesses
//! safe without any annotation.
//!
//! `num_tiles(&p, a)` already tells the compiler which partition axis its
//! result counts. These tests pin that an ordinary integer loop over that
//! bound carries the same information as the annotated forms — iterating a
//! `Dim`, or binding the axis with `with_bounds` — so the unannotated
//! kernel places its checks exactly where its annotated twin does. That
//! equality is the precondition for removing the annotations entirely.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod inferred_module {
    use cutile::core::*;

    /// Unannotated: an ordinary range loop over `num_tiles`, dynamic extent.
    #[cutile::entry]
    fn plain_loop<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        for i in 0i32..num_tiles(&p, 0) {
            let t = p.load([i]);
            z.store(t);
        }
    }

    /// Annotated twin: the same loop over an explicit `Dim`.
    #[cutile::entry]
    fn dim_loop<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let d = num_tiles(&p, 0).into_dim();
        for i in d {
            let t = p.load([i]);
            z.store(t);
        }
    }

    /// Unannotated, rank 2: both axes iterate their own `num_tiles`.
    #[cutile::entry]
    fn plain_loop_2d<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let p = x.partition(shape![BM, BN]);
        for i in 0i32..num_tiles(&p, 0) {
            for j in 0i32..num_tiles(&p, 1) {
                let t = p.load([i, j]);
                z.store(t);
            }
        }
    }

    /// Annotated twin of `plain_loop_2d`, using `with_bounds` branding.
    #[cutile::entry]
    fn bounded_loop_2d<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let p = x.partition(shape![BM, BN]);
        let rows = num_tiles(&p, 0).into_dim();
        let cols = num_tiles(&p, 1).into_dim();
        let bp = p.with_bounds((rows, cols));
        for i in rows {
            for j in cols {
                let t: Tile<f32, { [BM, BN] }> = bp.load(coord((i, j)));
                z.store(t);
            }
        }
    }

    /// Unannotated shared-dimension GEMM: the `k` loop iterates `px`'s axis 1,
    /// and `py.load([k, _])` indexes a DIFFERENT tensor's axis 0. A declared
    /// equality relates the two extents.
    #[cutile::entry(
        preconditions = (dim(x, 1) == dim(y, 0),)
    )]
    fn plain_shared_dim_gemm<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(shape![BM, BK]);
        let py = y.partition(shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
        for k in 0i32..num_tiles(&px, 1) {
            let tx = px.load([0i32, k]);
            let ty = py.load([k, 0i32]);
            acc = mma(tx, ty, acc);
        }
        z.store(acc);
    }

    /// Identical to `plain_shared_dim_gemm` but with NO declared equality.
    /// Iterating `x`'s axis says nothing about `y`'s extent, so the foreign
    /// access must keep a real check.
    #[cutile::entry]
    fn plain_shared_dim_gemm_undeclared<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(shape![BM, BK]);
        let py = y.partition(shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
        for k in 0i32..num_tiles(&px, 1) {
            let tx = px.load([0i32, k]);
            let ty = py.load([k, 0i32]);
            acc = mma(tx, ty, acc);
        }
        z.store(acc);
    }

    /// The dominant shape in grout's safe kernels: components minted by a
    /// mapped partition's schedule index a plain partition of ANOTHER tensor,
    /// with `with_bounds` tying that tensor's axes to the mapped grid.
    #[cutile::entry]
    fn mapped_components_bounded<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let xp = x
            .partition(shape![BM, BN])
            .with_bounds((num_tiles(&z, 0), num_tiles(&z, 1)));
        for index in z.iter_indices() {
            let (m, n) = index.components();
            let t = xp.load(coord((m, n)));
            z.store(t, index);
        }
    }

    /// The same kernel with the binding stated as a signature contract instead
    /// of a body annotation, and plain indexing.
    #[cutile::entry(
        preconditions = (
            dim(z, 0) == dim(x, 0),
            dim(z, 1) == dim(x, 1),
        )
    )]
    fn mapped_components_declared<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let xp = x.partition(shape![BM, BN]);
        for index in z.iter_indices() {
            let (m, n) = index.components();
            let t = xp.load([m, n]);
            z.store(t, index);
        }
    }
    /// KNOWN LIMITATION, pinned deliberately: the foreign access sits behind a
    /// runtime condition, but the fact it needs is still hoisted, so the check
    /// applies to every launch — including those where `flag <= 0` and the load
    /// never runs. See LIMITATIONS in `HOISTING_COVERAGE.md`.
    #[cutile::entry]
    fn conditional_foreign_access<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BK, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        flag: i32,
    ) {
        let px = x.partition(shape![BM, BK]);
        let py = y.partition(shape![BK, BN]);
        for k in 0i32..num_tiles(&px, 1) {
            if flag > 0i32 {
                let ty = py.load([k, 0i32]);
                z.store(ty);
            }
        }
    }

    /// The wide-tile idiom: a tile-block id indexes a partition directly —
    /// no loop, no annotation. The block-id axiom rung must discharge the
    /// access and stake a launch check on the grid instead. Root frame.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn block_id_indexed_load<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t = p.load([pid.0]);
        z.store(t);
    }

    /// The store side of the same idiom, through a mutable (view-framed)
    /// partition: the staked launch check must be stated over the
    /// kernel-visible view.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn block_id_indexed_store<const B: i32>(out: &mut Tensor<f32, { [-1] }>) {
        let mut p = out.partition_mut(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t: Tile<f32, { [B] }> = constant(1.0, shape![B]);
        p.store(t, [pid.0]);
    }
}

use cutile_compiler::compile_api::CheckPlacementCounts;
use inferred_module::__module_ast_self;

fn artifacts(
    name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> cutile_compiler::compile_api::CompileArtifacts {
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    cutile_compiler::compile_api::KernelCompiler::new(__module_ast_self, "inferred_module", name)
        .target("sm_120")
        .generics(generics)
        .strides(strides)
        .options(CompileOptions::default())
        .compile()
        .unwrap_or_else(|e| panic!("compile {name}: {e}"))
}

/// `(discharged, hoisted, in_place, launch_checks)` for one kernel.
fn counts(name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> (u32, u32, u32, usize) {
    let generics: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    let strides: Vec<(&str, &[i32])> = strides.to_vec();
    let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
        __module_ast_self,
        "inferred_module",
        name,
    )
    .target("sm_120")
    .generics(generics)
    .strides(&strides)
    .options(CompileOptions::default())
    .compile()
    .unwrap_or_else(|e| panic!("compile {name}: {e}"));
    let CheckPlacementCounts {
        discharged,
        hoisted,
        in_place,
    } = artifacts.check_counts();
    (
        discharged,
        hoisted,
        in_place,
        artifacts.launch_checks().len(),
    )
}

// An ordinary `0..num_tiles(&p, 0)` loop must place its check exactly where
// the `Dim`-iterating twin does: fully discharged, nothing left for the
// device.
#[test]
fn plain_loop_matches_its_dim_loop_twin() {
    common::with_test_stack(|| {
        let strides: &[(&str, &[i32])] = &[("z", &[1]), ("x", &[1])];
        let plain = counts("plain_loop", &["64"], strides);
        let annotated = counts("dim_loop", &["64"], strides);
        assert_eq!(
            plain, annotated,
            "unannotated loop placed its checks differently from the `Dim` twin \
             (discharged, hoisted, in_place, launch)"
        );
        assert_eq!(
            plain,
            (1, 0, 0, 0),
            "expected the access to discharge at compile time with no residual check"
        );
    });
}

// The same equality on a rank-2 partition, against the `with_bounds` twin.
#[test]
fn plain_loop_2d_matches_its_bounded_twin() {
    common::with_test_stack(|| {
        let strides: &[(&str, &[i32])] = &[("z", &[64, 1]), ("x", &[64, 1])];
        let plain = counts("plain_loop_2d", &["64", "64"], strides);
        let annotated = counts("bounded_loop_2d", &["64", "64"], strides);
        assert_eq!(
            plain, annotated,
            "unannotated rank-2 loops placed their checks differently from the \
             `with_bounds` twin (discharged, hoisted, in_place, launch)"
        );
        assert_eq!(
            plain,
            (2, 0, 0, 0),
            "expected both axes to discharge at compile time"
        );
    });
}

// A shared-dimension GEMM: the loop axis belongs to `x`, but one access
// indexes `y`. A declared `dim(x, 1) == dim(y, 0)` must carry the inferred
// provenance across tensors, exactly as it does for `with_bounds` today.
#[test]
fn shared_dimension_gemm_discharges_across_tensors() {
    common::with_test_stack(|| {
        let got = counts(
            "plain_shared_dim_gemm",
            &["32", "32", "32"],
            &[("z", &[32, 1]), ("x", &[32, 1]), ("y", &[32, 1])],
        );
        assert_eq!(
            got,
            (4, 0, 0, 2),
            "expected all four coordinates to leave the kernel: `k` on `x`'s own \
             axis and `k` on `y`'s axis via the declared equality, plus the two \
             constant-0 coordinates as non-empty-extent checks at launch"
        );
    });
}

// Without a declared equality, iterating `x`'s axis still says nothing about
// `y`'s extent — but the compiler can now DERIVE the fact it needs and have
// the host check it, rather than demanding the author declare it. The access
// leaves the kernel; the obligation does not disappear, it relocates. Both
// halves matter, so both are asserted: the placement, and that the emitted
// check actually rejects the shapes it exists to reject.
#[test]
fn undeclared_cross_tensor_access_relocates_to_launch() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;
        let strides: &[(&str, &[i32])] = &[("z", &[32, 1]), ("x", &[32, 1]), ("y", &[32, 1])];
        let (discharged, hoisted, in_place, _) = counts(
            "plain_shared_dim_gemm_undeclared",
            &["32", "32", "32"],
            strides,
        );
        assert_eq!(
            (discharged, hoisted + in_place),
            (4, 0),
            "with the fact derived rather than declared, nothing should remain \
             in the kernel"
        );

        let artifacts = artifacts(
            "plain_shared_dim_gemm_undeclared",
            &["32", "32", "32"],
            strides,
        );
        let checks = artifacts.launch_checks();
        // Params in signature order: z, x, y. The derived fact is
        // `dim(x, 1) <= dim(y, 0)`: the kernel walks x's k tiles into y's
        // axis 0, so y short there is unsafe, and y LARGER is safe — the
        // check must be an inequality, not an equality. Stating equality
        // rejected the larger case (2026-08-05 review of `2ca6e3d`).
        let matching = [vec![32i32, 32], vec![64, 128], vec![128, 64]];
        assert!(
            validate_launch_checks(checks, &matching, &matching, (1, 1, 1)).is_ok(),
            "a launch whose shared dimension agrees must be accepted: {checks:?}"
        );
        let larger_y = [vec![32i32, 32], vec![64, 128], vec![256, 64]];
        assert!(
            validate_launch_checks(checks, &larger_y, &larger_y, (1, 1, 1)).is_ok(),
            "a `y` larger than the walk needs is safe and must be accepted; \
             rejecting it means the derived fact is an equality again: {checks:?}"
        );
        let short_y = [vec![32i32, 32], vec![64, 128], vec![96, 64]];
        assert!(
            validate_launch_checks(checks, &short_y, &short_y, (1, 1, 1)).is_err(),
            "a launch whose `y` is short on the shared dimension must be \
             rejected at launch: {checks:?}"
        );
        // Shorter in ELEMENTS but equal in TILES: ceil(100/32) == ceil(128/32)
        // == 4, so every walked index is in range. The extent-form check
        // rejected this band (issue #216); the tile-count form must accept.
        let band_y = [vec![32i32, 32], vec![64, 128], vec![100, 64]];
        assert!(
            validate_launch_checks(checks, &band_y, &band_y, (1, 1, 1)).is_ok(),
            "equal tile counts must be accepted even when the target is \
             shorter in elements: {checks:?}"
        );
    });
}

// The grout shape, both spellings. `with_bounds` states the tie between the
// mapped grid and `x`'s axes in the body and has the compiler verify it;
// declared preconditions state the same tie in the signature and have the
// launcher verify it. If the two place identically, the annotation is a
// second way to say something the signature already says — which is what
// deciding the annotation's future turns on.
#[test]
fn mapped_component_access_places_the_same_either_way() {
    common::with_test_stack(|| {
        let strides: &[(&str, &[i32])] = &[("z", &[256, 1]), ("x", &[256, 1])];
        let bounded = counts(
            "mapped_components_bounded",
            &["32", "32", "4", "4"],
            strides,
        );
        let declared = counts(
            "mapped_components_declared",
            &["32", "32", "4", "4"],
            strides,
        );
        assert_eq!(
            bounded, declared,
            "the `with_bounds` and declared-precondition spellings placed their \
             checks differently (discharged, hoisted, in_place, launch)"
        );
        assert_eq!(
            declared.1 + declared.2,
            0,
            "neither spelling should leave a check in the kernel: {declared:?}"
        );
    });
}

// KNOWN LIMITATION — pinned so it cannot change silently, not because it is
// desirable. A launch check is unconditional: it constrains every launch, even
// ones where the access it guards never executes. Here the foreign load sits
// behind `flag > 0`, yet the derived tile-count fact is still enforced at launch,
// so a caller passing mismatched extents and `flag = 0` is rejected despite
// running no offending access. Before cross-tensor facts relocated, that
// caller got a device check inside the branch, which simply never fired.
//
// Fixing it means not lowering an obligation whose access is control-dependent
// — trading hoisting coverage for not rejecting valid launches. That is a
// design decision, not an oversight.
// Name reviewed and approved by hme (2026-08-05): the length is fantastic.
// It states the limitation being pinned, so whoever trips this failure knows
// what changed before opening anything.
#[test]
fn control_dependent_access_still_imposes_an_unconditional_launch_check() {
    common::with_test_stack(|| {
        let artifacts = artifacts(
            "conditional_foreign_access",
            &["32", "32", "32"],
            &[("z", &[32, 1]), ("x", &[32, 1]), ("y", &[32, 1])],
        );
        let causes: Vec<&str> = artifacts
            .launch_checks()
            .iter()
            .map(|c| c.cause.as_str())
            .collect();
        assert!(
            causes
                .iter()
                .any(|c| c.contains("ceil(dim(x, 1)/32) <= ceil(dim(y, 0)/32)")),
            "the guarded access's extent fact is hoisted regardless of the \
             condition guarding it; if this ever stops being true, the \
             limitation documented here has been addressed: {causes:?}"
        );
    });
}

// A tile-block id is grid-bounded by the execution model — the one hardware
// axiom the checker admits. It is NOT partition-bounded: the rung discharges
// the in-kernel check only by staking a launch check that the grid fits the
// axis's tile count, verified against the exact grid the kernel launches
// with. Pinned per frame — a root-framed load and a view-framed store, both
// compiled under deny_in_kernel_checks (the wide-tile kernel class from the
// 2026-08 B200 evaluation carried 8-16 in-kernel checks for exactly this).
#[test]
fn block_id_index_discharges_against_the_launch_grid() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;

        // Load, root frame. Compiling under deny already proves nothing
        // stayed in the kernel; the counts and the staked check are pinned
        // on top of that.
        let strides: &[(&str, &[i32])] = &[("z", &[1]), ("x", &[1])];
        let (_, hoisted, in_place, _) = counts("block_id_indexed_load", &["16"], strides);
        assert_eq!(
            (hoisted, in_place),
            (0, 0),
            "the block-id access must leave the kernel entirely"
        );
        let art = artifacts("block_id_indexed_load", &["16"], strides);
        let checks = art.launch_checks();
        assert!(
            format!("{checks:?}").contains("num_tile_blocks(0)"),
            "the discharge must stake a claim on the launch grid: {checks:?}"
        );
        // x has ceil(48/16) = 3 tiles: a grid of 3 launches, 4 must not.
        let shapes = [vec![16i32], vec![48]];
        assert!(
            validate_launch_checks(checks, &shapes, &shapes, (3, 1, 1)).is_ok(),
            "a grid inside the tile count must launch: {checks:?}"
        );
        assert!(
            validate_launch_checks(checks, &shapes, &shapes, (4, 1, 1)).is_err(),
            "a grid wider than the tile count must be rejected at launch: {checks:?}"
        );
        // A ragged final tile still counts: ceil(40/16) = 3.
        let ragged = [vec![16i32], vec![40]];
        assert!(
            validate_launch_checks(checks, &ragged, &ragged, (3, 1, 1)).is_ok(),
            "a partial final tile is still a tile: {checks:?}"
        );

        // Store, view frame.
        let strides: &[(&str, &[i32])] = &[("out", &[1])];
        let (_, hoisted, in_place, _) = counts("block_id_indexed_store", &["16"], strides);
        assert_eq!((hoisted, in_place), (0, 0));
        let art = artifacts("block_id_indexed_store", &["16"], strides);
        let checks = art.launch_checks();
        let shapes = [vec![48i32]];
        assert!(validate_launch_checks(checks, &shapes, &shapes, (3, 1, 1)).is_ok());
        assert!(
            validate_launch_checks(checks, &shapes, &shapes, (4, 1, 1)).is_err(),
            "the view-framed store must also be grid-gated: {checks:?}"
        );
    });
}
