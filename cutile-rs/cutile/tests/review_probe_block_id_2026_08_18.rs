/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Audit probes for the 2026-08-18 block-id bounds-check review.
//!
//! These kernels deliberately combine the block-id proof with symbolic
//! arithmetic, control-flow joins, loop iterands, crossed grid/tensor axes,
//! and a permuted partition. The compile-only test runs everywhere; the
//! ignored execution test additionally exercises the generated launcher and
//! JIT cache on a GPU.

use std::process::Command;
use std::sync::Arc;

use cutile::api;
use cutile::compile_api::{CheckPlacementCounts, KernelCompiler};
use cutile::prelude::*;
use cutile::tensor::{IntoPartition, ToHostVec};

mod common;

#[cutile::module]
mod block_id_review_module {
    use cutile::core::*;

    /// The two additions cancel modulo 2^32, so the value remains exactly
    /// `pid.1` even on the wrapping path. Tuple repacking must preserve that
    /// fact, while `dim_map = [1, 0]` must send logical axis 0 to root axis 1.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn permuted_wrapping_identity<const B: i32>(
        z: &mut Tensor<f32, { [B, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let p = x.partition_permuted(shape![B, B], const_array![1, 0]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let repacked = (pid.1, pid.0);
        let (y, _x) = repacked;
        let index = (y + 2_147_483_647i32) - 2_147_483_647i32;
        let t = p.load([index, 0i32]);
        z.store(t);
    }

    /// A runtime join can replace the block id with an arbitrary scalar. The
    /// joined value must lose the block-id term and retain an in-kernel check.
    #[cutile::entry]
    fn joined_reassignment<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        replacement: i32,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut index = pid.0;
        if replacement >= 0i32 {
            index = replacement;
        }
        let t = p.load([index]);
        z.store(t);
    }

    /// Compile-time branch adoption must publish the value from the sole
    /// feasible branch, not retain the pre-branch block-id term.
    #[cutile::entry]
    fn exact_if_replacement<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        replacement: i32,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut index = pid.0;
        if true {
            index = replacement;
        }
        let t = p.load([index]);
        z.store(t);
    }

    /// Conversely, an assignment in the unreachable compile-time branch must
    /// not erase the exact block-id value that reaches the access.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn exact_if_unreachable_replacement<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        replacement: i32,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut index = pid.0;
        if false {
            index = replacement;
        }
        let t = p.load([index]);
        z.store(t);
    }

    /// A variable carried out of a loop must not keep the block-id term it had
    /// before the loop once the body can replace its runtime value.
    #[cutile::entry]
    fn loop_carried_replacement<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        replacement: i32,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut index = pid.0;
        for _i in 0i32..1i32 {
            index = replacement;
        }
        let t = p.load([index]);
        z.store(t);
    }

    /// The access uses the loop iterand, not the bare special register. Its
    /// proof is the loop bound / same-view provenance; the block-id rung must
    /// not stake a launch check for it.
    #[cutile::entry]
    fn persistent_walk<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ntb: (i32, i32, i32) = get_num_tile_blocks();
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in (pid.0..num_tiles(&p, 0)).step_by(ntb.0 as usize) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// No mutable argument contributes an inferred grid. This lets the same
    /// JIT specialization be launched repeatedly with different explicit
    /// grids; every invocation must validate its own grid before launch.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn explicit_grid_revalidation<const B: i32>(x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let _t = p.load([pid.0]);
    }

    /// Same access without the deny policy, used to inspect forced-device
    /// placement in a subprocess with isolated environment variables.
    #[cutile::entry]
    fn block_id_ablation<const B: i32>(x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let _t = p.load([pid.0]);
    }

    /// The symbolic cancellation is valid only if the runtime integer ops
    /// obey the same modulo-2^32 ring identity on their wrapping path.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn wrapping_identity_execution<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let index = (pid.0 + 2_147_483_647i32) - 2_147_483_647i32;
        z.store(p.load([index]));
    }

    /// Frame attack: `partition_mut` adds `pid * ceil(OUTER/INNER)` after the
    /// safe check. Using pid again as the local coordinate can therefore make
    /// the effective coordinate much larger than the one the rung proves.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn nested_nondivisible_block_id<const OUTER: i32, const INNER: i32>(
        out: &mut Tensor<f32, { [1, OUTER] }>,
    ) {
        let mut p: PartitionMut<f32, { [1, INNER] }> = out.partition_mut(shape![1, INNER]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, { [1, INNER] }> = constant(7.0, shape![1, INNER]);
        p.store(tile, [0i32, pid.1]);
    }
}

use block_id_review_module::__module_ast_self;

const B: usize = 16;

fn compile(name: &str, strides: &[(&str, &[i32])]) -> cutile::compile_api::CompileArtifacts {
    KernelCompiler::new(__module_ast_self, "block_id_review_module", name)
        .target("sm_120")
        .generics(vec![B.to_string()])
        .strides(strides)
        .compile()
        .unwrap_or_else(|err| panic!("compile {name}: {err}"))
}

#[test]
#[ignore = "subprocess entry point for isolated force-device compilation"]
fn block_id_ablation_compile_runner() {
    let art = compile("block_id_ablation", &[("x", &[1])]);
    let counts = art.check_counts();
    println!(
        "COUNTS:{}:{}:{}:{}",
        counts.discharged,
        counts.hoisted,
        counts.in_place,
        art.launch_checks().len()
    );
}

fn ablation_counts(force_device: bool) -> (u32, u32, u32, usize) {
    let exe = std::env::current_exe().expect("current test executable");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--exact",
        "block_id_ablation_compile_runner",
        "--ignored",
        "--nocapture",
    ])
    .env_remove("CUTILE_FORCE_DEVICE_CHECKS")
    .env_remove("CUTILE_DISABLE_CHECK_HOISTING");
    if force_device {
        cmd.env("CUTILE_FORCE_DEVICE_CHECKS", "1");
    }
    let output = cmd.output().expect("ablation compile subprocess");
    assert!(
        output.status.success(),
        "ablation compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("COUNTS:"))
        .unwrap_or_else(|| panic!("missing placement counts in:\n{stdout}"));
    let values = line["COUNTS:".len()..]
        .split(':')
        .map(|value| value.parse::<usize>().expect("numeric placement count"))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 4);
    (
        values[0] as u32,
        values[1] as u32,
        values[2] as u32,
        values[3],
    )
}

#[test]
fn adversarial_block_id_placements_are_framed_by_the_staked_predicate() {
    common::with_test_stack(|| {
        use cutile::tile_kernel::validate_launch_checks;

        let art = compile(
            "permuted_wrapping_identity",
            &[("z", &[B as i32, 1]), ("x", &[B as i32, 1])],
        );
        let counts = art.check_counts();
        assert_eq!(
            counts,
            CheckPlacementCounts {
                discharged: 2,
                hoisted: 0,
                in_place: 0,
            },
            "the block-id axis and constant-zero axis should both discharge"
        );
        let checks = art.launch_checks();
        let debug = format!("{checks:?}");
        assert!(
            debug.contains("num_tile_blocks(1)") && debug.contains("extent(x, 1)"),
            "pid.1 must be checked against x's remapped root axis 1: {debug}"
        );

        // The grid's y extent is four. Four target tiles pass; three reject.
        let roots_matching = [vec![B as i32, 4 * B as i32], vec![B as i32, 4 * B as i32]];
        let views_matching = [vec![B as i32, B as i32], roots_matching[1].clone()];
        assert!(
            validate_launch_checks(checks, &roots_matching, &views_matching, (1, 4, 1)).is_ok()
        );
        let roots_short = [vec![B as i32, 4 * B as i32], vec![B as i32, 3 * B as i32]];
        let views_short = [vec![B as i32, B as i32], roots_short[1].clone()];
        assert!(
            validate_launch_checks(checks, &roots_short, &views_short, (1, 4, 1)).is_err(),
            "the remapped target has only three tiles for four y-blocks"
        );

        let joined = compile("joined_reassignment", &[("z", &[1]), ("x", &[1])]);
        assert_eq!(
            joined.check_counts(),
            CheckPlacementCounts {
                discharged: 0,
                hoisted: 0,
                in_place: 1,
            },
            "a runtime join must invalidate the stale TileBlockId term"
        );
        assert!(
            joined
                .launch_checks()
                .iter()
                .all(|check| !check.cause.contains("num_tile_blocks")),
            "the joined value must not reach the block-id rung"
        );

        let exact_taken = compile("exact_if_replacement", &[("z", &[1]), ("x", &[1])]);
        assert_eq!(
            exact_taken.check_counts(),
            CheckPlacementCounts {
                discharged: 0,
                hoisted: 0,
                in_place: 1,
            },
            "the taken compile-time branch must publish its unproven replacement"
        );
        assert!(
            exact_taken.launch_checks().is_empty(),
            "the replaced value must not retain a stale block-id launch proof"
        );

        let exact_untaken = compile(
            "exact_if_unreachable_replacement",
            &[("z", &[1]), ("x", &[1])],
        );
        assert_eq!(
            exact_untaken.check_counts(),
            CheckPlacementCounts {
                discharged: 1,
                hoisted: 0,
                in_place: 0,
            },
            "the unreachable branch must leave the actual block-id value intact"
        );
        assert_eq!(exact_untaken.launch_checks().len(), 1);

        let loop_carried = compile("loop_carried_replacement", &[("z", &[1]), ("x", &[1])]);
        assert_eq!(loop_carried.check_counts().in_place, 1);
        assert!(
            loop_carried.launch_checks().is_empty(),
            "a loop-carried replacement must not retain a stale block-id term"
        );

        let persistent = compile("persistent_walk", &[("z", &[1]), ("x", &[1])]);
        assert_eq!(
            persistent.check_counts().in_place,
            1,
            "the current non-unit-step loop proof remains a residual device check"
        );
        assert!(
            persistent
                .launch_checks()
                .iter()
                .all(|check| !check.cause.contains("num_tile_blocks")),
            "the loop iterand must discharge from its loop proof, not the block-id rung"
        );

        assert_eq!(
            ablation_counts(false),
            (1, 0, 0, 1),
            "normal mode should discharge and stake one grid check"
        );
        assert_eq!(
            ablation_counts(true),
            (0, 0, 1, 0),
            "full ablation must skip the rung and check the actual id in place"
        );
    });
}

#[test]
#[ignore = "audit probe requires a CUDA GPU"]
fn generated_launcher_revalidates_explicit_grid_on_every_cached_launch() {
    common::with_test_stack(|| {
        // Blocks 1 through 3 overflow the first add. Their loaded tiles still
        // have to match pid.0 exactly after the cancelling subtraction.
        let host = Arc::new((0..4 * B).map(|i| i as f32).collect::<Vec<_>>());
        let input: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&host)
            .sync()
            .expect("identity input")
            .into();
        let output = api::zeros::<f32>(&[4 * B]).sync().expect("identity output");
        let (output, _input) =
            block_id_review_module::wrapping_identity_execution(output.partition([B]), input)
                .generics(vec![B.to_string()])
                .sync()
                .expect("wrapping identity launch");
        let got = output
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("identity output copy");
        assert_eq!(
            got, *host,
            "the canonical term must equal the runtime value"
        );

        let x: Arc<Tensor<f32>> = api::zeros::<f32>(&[3 * B])
            .sync()
            .expect("x allocation")
            .into();

        // First launch compiles/populates the cache and fits the three-tile x.
        let (_x,) = block_id_review_module::explicit_grid_revalidation(x.clone())
            .generics(vec![B.to_string()])
            .grid((3, 1, 1))
            .sync()
            .expect("three-block launch");

        // Same specialization and input, different runtime grid. A cache hit
        // must not bypass the per-invocation launch validator.
        let err = block_id_review_module::explicit_grid_revalidation(x)
            .generics(vec![B.to_string()])
            .grid((4, 1, 1))
            .sync()
            .expect_err("four blocks must not index a three-tile tensor");
        assert!(
            err.to_string().contains("num_tile_blocks(0)"),
            "expected the block-id launch check, got: {err}"
        );
    });
}

#[test]
#[ignore = "audit soundness probe requires a CUDA GPU"]
fn nested_nondivisible_frame_does_not_spill_into_the_next_row() {
    common::with_test_stack(|| {
        const ROWS: usize = 2;
        const EXTENT: usize = 200;
        const OUTER: usize = 50;
        const INNER: usize = 16;

        let output = api::zeros::<f32>(&[ROWS, EXTENT])
            .sync()
            .expect("nested frame output");
        let (output,) =
            block_id_review_module::nested_nondivisible_block_id(output.partition([1, OUTER]))
                .generics(vec![OUTER.to_string(), INNER.to_string()])
                .sync()
                .expect("the staked grid check accepts grid (2, 4)");
        let got = output
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("nested frame output copy");

        for row in 0..ROWS {
            for col in [0usize, 80, 160] {
                assert_eq!(
                    got[row * EXTENT + col],
                    7.0,
                    "the in-range CTAs must execute their stores"
                );
            }
        }
        // For y=3 the adjusted nested index is 3*ceil(50/16)+3 = 15,
        // whose element offset is 240. If the unchecked store linearizes that
        // out-of-axis coordinate, row 0 spills into row 1 columns 40..55.
        assert!(
            got[EXTENT + 40..EXTENT + 56]
                .iter()
                .all(|&value| value == 0.0),
            "out-of-axis nested store spilled into the next row: {:?}",
            &got[EXTENT + 40..EXTENT + 56]
        );
    });
}
