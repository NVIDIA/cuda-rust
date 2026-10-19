/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounds-check placement soundness (2026-08 audit): a loop body with a
//! `continue` keeps its checks in place and an in-place check tests the
//! actual index (not the range's extreme); a `step_by` loop is checked at the
//! last *attained* index; and the tile-count arithmetic
//! `ceil(extent / tile)` does not wrap for extents near `i32::MAX`.

use std::sync::Arc;

use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;

use crate::audit_common::{self, host, report_outcome, run_in_subprocess, upload, Outcome};
use crate::common;

#[cutile::module]
mod check_hoisting_module {
    use cutile::core::*;

    /// `continue` skips the access for `k >= limit`: the attained index set
    /// is `[0, limit)`, so the check must not be hoisted to `k = 63` (nor
    /// test `63` in place).
    #[cutile::entry()]
    fn continue_before<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        limit: i32,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for k in 0i32..64i32 {
            if k >= limit {
                continue;
            }
            acc = acc + p.load([k]);
        }
        z.store(acc);
    }

    /// `(0..10).step_by(4)` attains {0, 4, 8}: nine tiles suffice.
    #[cutile::entry()]
    fn stepped_static<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [144] }>) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in (0i32..10i32).step_by(4) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// Eight tiles do not: index 8 is attained and out of range.
    #[cutile::entry()]
    fn stepped_static_short<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [128] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in (0i32..10i32).step_by(4) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// The runtime-extent twin: hoisted, tested at the last attained index.
    #[cutile::entry()]
    fn stepped_dynamic<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in (0i32..10i32).step_by(4) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// A static extent near `i32::MAX`: `ceil(extent / 16)` must fold to
    /// 134217728, not wrap.
    #[cutile::entry()]
    fn huge_static_extent(
        z: &mut Tensor<u8, { [16] }>,
        x: &Tensor<u8, { [2147483647] }>,
        idx: i32,
    ) {
        let p = x.partition(shape![16]);
        z.store(p.load([idx]));
    }

    /// The runtime-extent twin, exercised on the device with a 2 GiB tensor.
    #[cutile::entry()]
    fn huge_dynamic_extent(z: &mut Tensor<u8, { [16] }>, x: &Tensor<u8, { [-1] }>, idx: i32) {
        let p = x.partition(shape![16]);
        z.store(p.load([idx]));
    }
}

use check_hoisting_module::__module_ast_self;

const B: usize = 16;

fn compile(
    function_name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> Result<(String, cutile::compile_api::CheckPlacementCounts), cutile_compiler::error::JITError> {
    audit_common::compile(
        __module_ast_self,
        "check_hoisting_module",
        function_name,
        generics,
        strides,
    )
}

#[test]
fn a_loop_body_with_continue_keeps_its_check_in_place() {
    common::with_test_stack(|| {
        let (ir, counts) = compile("continue_before", &["16"], &[("z", &[1]), ("x", &[1])])
            .expect("continue_before should compile");
        assert_eq!(
            (counts.hoisted, counts.in_place),
            (0, 1),
            "the check must not leave the body: {counts:?}\n{ir}"
        );
        // Ten tiles, loop range `0..64`, `limit = 4`: every attained access
        // is in bounds, so the launch must succeed (the hoisted check tested
        // `k = 63` and trapped — differential harness defect D2).
        let x = upload((0..(10 * B) as i32).map(|v| v as f32).collect());
        let (z, _x, _limit) =
            check_hoisting_module::continue_before(api::zeros::<f32>(&[B]).partition([B]), x, 4)
                .generics(vec![B.to_string()])
                .compile_options(CompileOptions::new().device_debug(false))
                .sync()
                .expect("continue_before with every attained access in bounds");
        let expected: Vec<f32> = (0..B)
            .map(|j| (0..4).map(|k| (k * B + j) as f32).sum())
            .collect();
        assert_eq!(host(&z.unpartition()), expected);
    });
}

#[test]
fn stepped_loops_are_checked_at_the_last_attained_index() {
    common::with_test_stack(|| {
        // Nine static tiles: {0, 4, 8} all fit; the former `[0, 9]` interval
        // failed the static fold with "9 < 9".
        let (_ir, counts) = compile("stepped_static", &["16"], &[("z", &[1]), ("x", &[1])])
            .expect("stepped_static should compile: index 8 is the last attained");
        assert_eq!(counts.discharged, 1, "{counts:?}");
        // Eight static tiles: index 8 is attained and out of range.
        let err = compile("stepped_static_short", &["16"], &[("z", &[1]), ("x", &[1])])
            .err()
            .map(|err| err.to_string())
            .expect("stepped_static_short must be rejected");
        assert!(
            err.contains("Bounds check failed") && err.contains("8 < 8"),
            "expected the static fold to reject index 8: {err}"
        );
        // Runtime extent: hoisted, tested at 8.
        let (ir, counts) = compile("stepped_dynamic", &["16"], &[("z", &[1]), ("x", &[1])])
            .expect("stepped_dynamic should compile");
        assert_eq!(counts.hoisted, 1, "{counts:?}\n{ir}");
        let x = upload((0..(9 * B) as i32).map(|v| v as f32).collect());
        let (z, _x) =
            check_hoisting_module::stepped_dynamic(api::zeros::<f32>(&[B]).partition([B]), x)
                .generics(vec![B.to_string()])
                .compile_options(CompileOptions::new().device_debug(false))
                .sync()
                .expect("nine tiles cover the attained indices {0, 4, 8}");
        let expected: Vec<f32> = (0..B)
            .map(|j| [0usize, 4, 8].iter().map(|k| (k * B + j) as f32).sum())
            .collect();
        assert_eq!(host(&z.unpartition()), expected);
    });
}

#[test]
fn tile_count_arithmetic_does_not_wrap_near_i32_max() {
    common::with_test_stack(|| {
        let (ir, _) = compile("huge_static_extent", &[], &[("z", &[1]), ("x", &[1])])
            .expect("huge_static_extent should compile");
        assert!(
            ir.contains("134217728"),
            "ceil(2147483647 / 16) must fold to 134217728:\n{ir}"
        );
        // The runtime twin, against a real 2 GiB tensor: the last tile is
        // 134217727, and `(extent + 15) / 16` wrapped negative.
        let len = i32::MAX as usize;
        let x = match api::zeros::<u8>(&[len]).sync() {
            Ok(x) => Arc::new(x),
            Err(err) => {
                eprintln!("skipping the 2 GiB runtime case: allocation failed: {err}");
                return;
            }
        };
        let last_tile = (len / 16) as i32;
        let (z, _x, _idx) = check_hoisting_module::huge_dynamic_extent(
            api::zeros::<u8>(&[16]).partition([16]),
            x,
            last_tile,
        )
        .sync()
        .expect("the last tile of a near-i32::MAX extent is in bounds");
        assert_eq!(host(&z.unpartition()), vec![0u8; 16]);
    });
}

// ---------------------------------------------------------------------------
// The out-of-range twin must still stop (subprocess: a device trap poisons
// the CUDA context).
// ---------------------------------------------------------------------------

const CASE_ENV: &str = "CUTILE_AUDIT_CHECK_HOISTING_CASE";

fn execute_trap_case(case: &str) -> Result<(), String> {
    match case {
        // Eight tiles, attained indices {0, 4, 8}: index 8 is out of range.
        "stepped_dynamic_short" => {
            let x = upload((0..(8 * B) as i32).map(|v| v as f32).collect());
            check_hoisting_module::stepped_dynamic(api::zeros::<f32>(&[B]).partition([B]), x)
                .generics(vec![B.to_string()])
                .compile_options(CompileOptions::new().device_debug(false))
                .sync()
                .map_err(|err| err.to_string())?;
            Ok(())
        }
        other => Err(format!("unknown case {other}")),
    }
}

/// Subprocess entry point; see `audit_common::run_in_subprocess`.
#[test]
#[ignore]
fn check_hoisting_case_runner() {
    let case = std::env::var(CASE_ENV).expect("case env var not set");
    common::with_test_stack(move || {
        report_outcome(std::panic::catch_unwind(|| execute_trap_case(&case)));
    });
}

#[test]
fn stepped_loop_past_the_last_tile_still_stops() {
    let case = "stepped_dynamic_short";
    match run_in_subprocess(
        "audit_check_hoisting::check_hoisting_case_runner",
        CASE_ENV,
        case,
    ) {
        Outcome::Stop(msg) => eprintln!("{case}: stopped as expected: {msg}"),
        Outcome::Ok => panic!("{case}: an out-of-range access ran to completion"),
    }
}
