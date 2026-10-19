/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Differential placement testing: the checker's output, compiled two ways,
//! must produce outcome-refining executions.
//!
//! Every case runs twice — normally, and under `CUTILE_FORCE_DEVICE_CHECKS=1`
//! (full placement ablation: every plain-family check at its access site,
//! two-sided, over the actual runtime values — no discharge by provenance,
//! fold, entailment, or inferred bounds, so the reference build inherits
//! none of the proofs it referees; 2026-08-12 review, S2). Placement is
//! allowed to move a stop earlier and cleaner (an in-loop trap may become a
//! preheader trap may become a launch rejection) and must change nothing
//! else:
//!
//! - ablated `ok` ⟹ normal `ok` with an identical result (else placement
//!   manufactured a failure — a precision violation);
//! - ablated `stop` ⟹ normal `stop` of any kind (else placement erased a
//!   failure — a soundness violation, never tolerable);
//! - equal `ok` outcomes must hash equal (else a miscompile).
//!
//! Each case runs in a subprocess because a device-side assert poisons the
//! CUDA context; the runner executes one case per process and reports one
//! `OUTCOME:` line.
//!
//! Known precision violations are pinned in `KNOWN_P_VIOLATIONS` with the
//! defect they trace to, so the harness both proves it can detect them and
//! fails the moment one appears or disappears unannounced. Fixing a defect
//! moves its rows out of the ledger in the same commit. Soundness violations
//! have no ledger: one observed is a hard failure.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::sync::Arc;

use cutile::api;
use cutile::compile_api::KernelCompiler;
use cutile::prelude::*;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod diff_module {
    use cutile::core::*;

    /// A foreign access behind a runtime flag. The `k` walk is bounded by
    /// `x`'s axis 1; using it on `y` derives `dim(x,1) <= dim(y,0)` plus a
    /// non-empty-extent fact — both currently enforced at launch regardless
    /// of `flag` (defect D1 in `CHECK_PLACEMENT_CONTROL_FLOW.md`).
    #[cutile::entry]
    fn guarded_foreign<const B: i32>(
        z: &mut Tensor<f32, { [B, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        flag: i32,
    ) {
        let px = x.partition(shape![B, B]);
        let py = y.partition(shape![B, B]);
        let mut acc: Tile<f32, { [B, B] }> = constant(0.0, shape![B, B]);
        for k in 0i32..num_tiles(&px, 1) {
            if flag > 0i32 {
                let t = py.load([k, 0i32]);
                acc = acc + t;
            }
        }
        z.store(acc);
    }

    /// An access skipped by `continue` for `k >= limit`. The attained index
    /// set is `[0, limit)`, a strict subset of the loop's range; the check
    /// must stay in place, since a hoisted one would test the range's
    /// extreme (the former defect D2).
    #[cutile::entry]
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

    // NOTE: the H5b row (`break` after the access) is unrepresentable today:
    // Tile IR rejects `cuda_tile.break` inside `cuda_tile.for` at verification
    // ("can only be nested within ... 'cuda_tile.loop', 'cuda_tile.if'"), and
    // the DSL's `while`/`loop` forms push no frame, so nothing hoists across
    // them. The hypothesis guards the future generalization, not current
    // reachability — see CHECK_PLACEMENT_CONTROL_FLOW.md.

    /// A runtime scalar index: in place under both modes, so outcomes must
    /// match exactly — the control case that keeps the harness honest.
    #[cutile::entry]
    fn runtime_scalar<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        idx: i32,
    ) {
        let p = x.partition(shape![B]);
        let t = p.load([idx]);
        z.store(t);
    }

    /// A tile-block id indexing a foreign partition directly (the wide-tile
    /// idiom): normal placement discharges via the block-id axiom rung and
    /// stakes a launch check on the grid; the reference build checks the
    /// actual id at the access site. A grid wider than the target's tile
    /// count must stop in both builds.
    #[cutile::entry]
    fn block_id_foreign<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t = p.load([pid.0]);
        z.store(t);
    }

    /// A same-view walk: fully discharged by provenance normally, so the
    /// ONLY thing standing between it and silence under ablation is the
    /// reference build refusing to inherit that proof (S2's count pin).
    #[cutile::entry]
    fn same_view_walk<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..num_tiles(&p, 0) {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// A safe static walk, used to pin that full ablation bypasses the static
    /// fold and emits an actual-value check at the access.
    #[cutile::entry]
    fn static_fold_walk<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [48] }>) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// Mathematical range `[0, 2]`, machine value 294,967,296 at `i = 2`
    /// (the product wraps; `max` keeps the wreck). Both builds must stop
    /// (2026-08-12 review, S1 scenario 1).
    #[cutile::entry]
    fn wrapped_product_masked_by_max<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = max(i * -2_000_000_000i32, i);
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }

    /// Mathematically nonnegative remainder, machine value `-296` at
    /// `i = 2` (the dividend wraps). The lower guard must exist and fire in
    /// both builds (2026-08-12 review, S1 scenario 2 — the row the old
    /// reference silently passed).
    #[cutile::entry]
    fn wrapped_dividend_negative_remainder<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = (i * 2_000_000_000i32) % 1_000i32;
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }

    /// Static-extent twin of the wrapped product: the row the old reference
    /// passed by inheriting the static fold (2026-08-12 review, S1/S2).
    #[cutile::entry]
    fn wrapped_product_static_extent<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [48] }>,
    ) {
        let p = x.partition(shape![B]);
        let mut acc: Tile<f32, { [B] }> = constant(0.0, shape![B]);
        for i in 0i32..3i32 {
            let index = max(i * -2_000_000_000i32, i);
            acc = acc + p.load([index]);
        }
        z.store(acc);
    }
}

use diff_module::__module_ast_self;

const B: usize = 16;

/// One corpus row: a kernel, its launch data, and what the differential is
/// expected to show today.
struct Case {
    name: &'static str,
    /// `Some(reason)`: this row currently VIOLATES precision (ablated ok,
    /// normal stop) for the stated defect, and the harness asserts the
    /// violation is still present. `None`: the row must refine.
    known_p_violation: Option<&'static str>,
    /// Some rows are deliberately out of bounds. Their ablated execution
    /// must stop; allowing `Ok`/`Ok` would recreate the S2 oracle blind spot.
    reference_must_stop: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "guarded_foreign_flag_off_mismatched",
        known_p_violation: Some(
            "D1: launch checks ignore control dependence; enforced with flag = 0",
        ),
        reference_must_stop: false,
    },
    Case {
        name: "guarded_foreign_flag_on_matching",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "guarded_foreign_flag_on_larger_target",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "guarded_foreign_flag_on_short_target",
        known_p_violation: None,
        reference_must_stop: true,
    },
    Case {
        name: "continue_before_limit_in_bounds",
        // D2 ("hoisted check tests an index `continue` never attains") is
        // fixed: a body with an early exit keeps its check in place.
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "block_id_foreign_matching",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "block_id_foreign_short_target",
        known_p_violation: None,
        reference_must_stop: true,
    },
    Case {
        name: "runtime_scalar_in_bounds",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "runtime_scalar_out_of_bounds",
        known_p_violation: None,
        reference_must_stop: true,
    },
    Case {
        name: "same_view_walk",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "static_fold_walk",
        known_p_violation: None,
        reference_must_stop: false,
    },
    Case {
        name: "wrapped_product_masked_by_max",
        known_p_violation: None,
        reference_must_stop: true,
    },
    Case {
        name: "wrapped_dividend_negative_remainder",
        known_p_violation: None,
        reference_must_stop: true,
    },
    Case {
        name: "wrapped_product_static_extent",
        known_p_violation: None,
        reference_must_stop: true,
    },
];

fn patterned(len: usize) -> Arc<Vec<f32>> {
    Arc::new((0..len).map(|i| ((i % 23) as f32) * 0.25 - 2.0).collect())
}

fn hash_outputs(v: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    for x in v {
        x.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Runs one corpus row to completion in THIS process and returns
/// `Ok(output)` or `Err(stop message)`. Every launch error and device trap
/// surfaces as an `Err` from a `sync()`.
fn execute_case(name: &str) -> Result<Vec<f32>, String> {
    // Exercise placement optimizations even when Cargo enables device debug.
    let options = CompileOptions::new().device_debug(false);
    fn e<E: std::fmt::Display>(err: E) -> String {
        err.to_string()
    }
    match name {
        // x is [B, 4B] (4 k-tiles); y's row count is the variable.
        n if n.starts_with("guarded_foreign") => {
            let (flag, y_rows): (i32, usize) = match n {
                "guarded_foreign_flag_off_mismatched" => (0, 3 * B),
                "guarded_foreign_flag_on_matching" => (1, 4 * B),
                "guarded_foreign_flag_on_larger_target" => (1, 8 * B),
                "guarded_foreign_flag_on_short_target" => (1, 3 * B),
                other => return Err(format!("unknown case {other}")),
            };
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(B * 4 * B))
                .reshape(&[B, 4 * B])
                .sync()
                .map_err(e)?
                .into();
            // NOTE: a zero-extent row is untestable here — the host API
            // refuses zero-length tensors — so the accepted non-empty-extent
            // trade is covered at the validate level by
            // `hoisted_check_rejects_zero_extent_at_launch` instead.
            let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(y_rows * B))
                .reshape(&[y_rows, B])
                .sync()
                .map_err(e)?
                .into();
            let z = api::zeros::<f32>(&[B, B]).sync().map_err(e)?;
            let (z, _x, _y, _flag) = diff_module::guarded_foreign(z.partition([B, B]), x, y, flag)
                .generics(vec![B.to_string()])
                .compile_options(options)
                .sync()
                .map_err(e)?;
            z.unpartition().to_host_vec().sync().map_err(e)
        }
        "continue_before_limit_in_bounds" => {
            // 10 tiles; the loop range is 0..64; limit keeps every attained
            // access inside the 10 tiles.
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(10 * B))
                .sync()
                .map_err(e)?
                .into();
            let z = api::zeros::<f32>(&[B]).sync().map_err(e)?;
            let limit = 4i32;
            let (z, _x, _limit) = diff_module::continue_before(z.partition([B]), x, limit)
                .generics(vec![B.to_string()])
                .compile_options(options)
                .sync()
                .map_err(e)?;
            z.unpartition().to_host_vec().sync().map_err(e)
        }
        n if n.starts_with("block_id_foreign") => {
            // z is 4 tiles of B, so the inferred grid is 4; x's tile count is
            // the variable: 4 (grid fits) or 3 (grid one tile too wide).
            let x_tiles = if n.ends_with("matching") { 4 } else { 3 };
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(x_tiles * B))
                .sync()
                .map_err(e)?
                .into();
            let z = api::zeros::<f32>(&[4 * B]).sync().map_err(e)?;
            let (z, _x) = diff_module::block_id_foreign(z.partition([B]), x)
                .generics(vec![B.to_string()])
                .compile_options(options)
                .sync()
                .map_err(e)?;
            z.unpartition().to_host_vec().sync().map_err(e)
        }
        n if n.starts_with("runtime_scalar") => {
            let idx: i32 = if n.ends_with("in_bounds") { 3 } else { 12 };
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(10 * B))
                .sync()
                .map_err(e)?
                .into();
            let z = api::zeros::<f32>(&[B]).sync().map_err(e)?;
            let (z, _x, _idx) = diff_module::runtime_scalar(z.partition([B]), x, idx)
                .generics(vec![B.to_string()])
                .compile_options(options)
                .sync()
                .map_err(e)?;
            z.unpartition().to_host_vec().sync().map_err(e)
        }
        "same_view_walk"
        | "static_fold_walk"
        | "wrapped_product_masked_by_max"
        | "wrapped_dividend_negative_remainder"
        | "wrapped_product_static_extent" => {
            let x_len = if name == "wrapped_dividend_negative_remainder" {
                1_000 * B
            } else {
                3 * B
            };
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&patterned(x_len))
                .sync()
                .map_err(e)?
                .into();
            let z = api::zeros::<f32>(&[B]).sync().map_err(e)?;
            let z = match name {
                "same_view_walk" => {
                    let (z, _x) = diff_module::same_view_walk(z.partition([B]), x)
                        .generics(vec![B.to_string()])
                        .compile_options(options)
                        .sync()
                        .map_err(e)?;
                    z
                }
                "static_fold_walk" => {
                    let (z, _x) = diff_module::static_fold_walk(z.partition([B]), x)
                        .generics(vec![B.to_string()])
                        .compile_options(options)
                        .sync()
                        .map_err(e)?;
                    z
                }
                "wrapped_product_masked_by_max" => {
                    let (z, _x) = diff_module::wrapped_product_masked_by_max(z.partition([B]), x)
                        .generics(vec![B.to_string()])
                        .compile_options(options)
                        .sync()
                        .map_err(e)?;
                    z
                }
                "wrapped_dividend_negative_remainder" => {
                    let (z, _x) =
                        diff_module::wrapped_dividend_negative_remainder(z.partition([B]), x)
                            .generics(vec![B.to_string()])
                            .compile_options(options)
                            .sync()
                            .map_err(e)?;
                    z
                }
                "wrapped_product_static_extent" => {
                    let (z, _x) = diff_module::wrapped_product_static_extent(z.partition([B]), x)
                        .generics(vec![B.to_string()])
                        .compile_options(options)
                        .sync()
                        .map_err(e)?;
                    z
                }
                _ => unreachable!(),
            };
            z.unpartition().to_host_vec().sync().map_err(e)
        }
        other => Err(format!("unknown case {other}")),
    }
}

/// Subprocess entry point: runs the case named by `CUTILE_DIFF_CASE` and
/// prints exactly one `OUTCOME:` line. Ignored so it only runs when the
/// parent test invokes it by name.
#[test]
#[ignore]
fn differential_case_runner() {
    let case = std::env::var("CUTILE_DIFF_CASE").expect("CUTILE_DIFF_CASE not set");
    common::with_test_stack(move || {
        let result = std::panic::catch_unwind(|| execute_case(&case));
        match result {
            Ok(Ok(out)) => println!("OUTCOME:OK:{:016x}", hash_outputs(&out)),
            Ok(Err(msg)) => println!("OUTCOME:STOP:{}", msg.replace('\n', " | ")),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "opaque panic".to_string());
                println!("OUTCOME:PANIC:{}", msg.replace('\n', " | "));
            }
        }
    });
}

/// Compile-only subprocess entry point for pinning proof ablation without
/// mutating process-global environment variables in the parent test.
#[test]
#[ignore]
fn differential_compile_runner() {
    let case = std::env::var("CUTILE_DIFF_COMPILE_CASE").expect("CUTILE_DIFF_COMPILE_CASE not set");
    common::with_test_stack(move || {
        let artifacts = KernelCompiler::new(__module_ast_self, "diff_module", &case)
            .target("sm_120")
            .generics(vec![B.to_string()])
            .strides(&[("z", &[1]), ("x", &[1])])
            .options(CompileOptions::default().device_debug(false))
            .compile()
            .unwrap_or_else(|err| panic!("compile {case}: {err}"));
        let counts = artifacts.check_counts();
        println!(
            "COUNTS:{}:{}:{}:{}",
            counts.discharged,
            counts.hoisted,
            counts.in_place,
            artifacts.launch_checks().len()
        );
    });
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Ok(String),
    Stop(String),
}

fn run_subprocess(case: &str, ablate: bool) -> Outcome {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--exact",
        "differential_case_runner",
        "--ignored",
        "--nocapture",
    ])
    .env("CUTILE_DIFF_CASE", case)
    .env_remove("CUTILE_FORCE_DEVICE_CHECKS")
    .env_remove("CUTILE_DISABLE_CHECK_HOISTING");
    if ablate {
        cmd.env("CUTILE_FORCE_DEVICE_CHECKS", "1");
    }
    let out = cmd.output().expect("spawn case subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let Some(line) = stdout.lines().find(|l| l.starts_with("OUTCOME:")) else {
        // A device trap poisons the CUDA context; the destructors that run
        // while the error propagates then panic themselves, which aborts the
        // process before the runner can print. That IS a stop — classify it
        // as one when the wreckage says so, and fail loudly otherwise.
        if out.status.code() != Some(0) && stderr.contains("panicked") {
            let first = stderr
                .lines()
                .find(|l| l.contains("panicked"))
                .unwrap_or("aborted");
            return Outcome::Stop(format!("aborted during cleanup: {first}"));
        }
        panic!(
            "case {case} (ablate={ablate}) produced no OUTCOME line.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    };
    if let Some(hash) = line.strip_prefix("OUTCOME:OK:") {
        Outcome::Ok(hash.to_string())
    } else if let Some(msg) = line.strip_prefix("OUTCOME:STOP:") {
        Outcome::Stop(msg.to_string())
    } else {
        panic!("case {case} (ablate={ablate}) infrastructure failure: {line}")
    }
}

fn run_compile_subprocess(case: &str, ablate: bool) -> (u32, u32, u32, usize) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--exact",
        "differential_compile_runner",
        "--ignored",
        "--nocapture",
    ])
    .env("CUTILE_DIFF_COMPILE_CASE", case)
    .env_remove("CUTILE_FORCE_DEVICE_CHECKS")
    .env_remove("CUTILE_DISABLE_CHECK_HOISTING");
    if ablate {
        cmd.env("CUTILE_FORCE_DEVICE_CHECKS", "1");
    }
    let out = cmd.output().expect("spawn compile subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "compile case {case} (ablate={ablate}) failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let line = stdout
        .lines()
        .find(|line| line.starts_with("COUNTS:"))
        .unwrap_or_else(|| panic!("compile case {case} produced no COUNTS line:\n{stdout}"));
    let values = line["COUNTS:".len()..]
        .split(':')
        .map(|value| value.parse::<usize>().expect("numeric placement count"))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 4, "malformed placement counts: {line}");
    (
        values[0] as u32,
        values[1] as u32,
        values[2] as u32,
        values[3],
    )
}

// The differential proper. For each row: ablated is the reference semantics
// (every check at its access site); normal placement must refine it.
#[test]
fn placement_is_outcome_refining() {
    let mut failures: Vec<String> = vec![];
    for case in CASES {
        let ablated = run_subprocess(case.name, true);
        let normal = run_subprocess(case.name, false);
        if case.reference_must_stop && !matches!(ablated, Outcome::Stop(_)) {
            failures.push(format!(
                "{}: BROKEN REFERENCE — deliberately out-of-bounds case did not stop under full ablation ({ablated:?})",
                case.name
            ));
        }
        match (&ablated, &normal) {
            // Soundness: a stop may never be erased. No ledger for these.
            (Outcome::Stop(_), Outcome::Ok(_)) => failures.push(format!(
                "{}: SOUNDNESS VIOLATION — ablated stopped ({ablated:?}) but placement ran to \
                 completion",
                case.name
            )),
            // Precision: placement may not manufacture a stop, unless the row
            // is pinned to a known defect — in which case it MUST still fail,
            // so the ledger cannot go stale.
            (Outcome::Ok(_), Outcome::Stop(stop)) => match case.known_p_violation {
                Some(reason) => println!(
                    "{}: pinned precision violation present as expected ({reason}): {stop}",
                    case.name
                ),
                None => failures.push(format!(
                    "{}: PRECISION VIOLATION — ablated ok, placement stopped: {stop}",
                    case.name
                )),
            },
            (Outcome::Ok(a), Outcome::Ok(b)) => {
                if case.known_p_violation.is_some() {
                    failures.push(format!(
                        "{}: pinned as a known precision violation but refines now — the \
                         defect was fixed; move the row out of the ledger",
                        case.name
                    ));
                } else if a != b {
                    failures.push(format!(
                        "{}: MISCOMPILE — both ran but outputs differ ({a} vs {b})",
                        case.name
                    ));
                }
            }
            (Outcome::Stop(_), Outcome::Stop(_)) => {
                if case.known_p_violation.is_some() {
                    failures.push(format!(
                        "{}: pinned as a precision violation but the ablated run also \
                         stopped — the pin is mischaracterised",
                        case.name
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "differential placement violations:\n{}",
        failures.join("\n")
    );
}

#[test]
fn force_device_checks_ablate_provenance_and_static_folds() {
    for case in ["same_view_walk", "static_fold_walk"] {
        assert_eq!(
            run_compile_subprocess(case, false),
            (1, 0, 0, 0),
            "{case}: normal compilation should discharge its proof"
        );
        assert_eq!(
            run_compile_subprocess(case, true),
            (0, 0, 1, 0),
            "{case}: full ablation must emit one actual-value in-place check"
        );
    }
}

// The launch checks the compiler discharges against must actually run. The
// validator snapshot used by the generated launcher was taken BEFORE
// compilation — the accumulator fills during it — so every hoisted check was
// silently dropped at launch and the discharged accesses ran unchecked. A
// launch rejection happens before any kernel starts, so this is safe to
// assert in-process.
#[test]
fn launch_checks_are_enforced_at_launch() {
    common::with_test_stack(|| {
        let result = execute_case("guarded_foreign_flag_on_short_target");
        let err = result.expect_err(
            "a launch violating dim(x, 1) <= dim(y, 0) must be rejected before the kernel runs",
        );
        assert!(
            err.contains("<=") || err.to_lowercase().contains("launch"),
            "expected a launch-check rejection, got: {err}"
        );
    });
}
