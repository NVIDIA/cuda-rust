/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Audit-only regression probe for the 2026-08-12 bounds-check placement review.
//!
//! This test is deliberately ignored because every case intentionally trips a
//! device assert. It began as the review's executable repro and now verifies
//! that normal and fully ablated builds both stop on the original bad inputs.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::sync::Arc;

use cutile::api;
use cutile::compile_api::{CheckPlacementCounts, KernelCompiler};
use cutile::prelude::*;
use cutile::tensor::{IntoPartition, ToHostVec};

mod common;

#[cutile::module]
mod review_probe_module {
    use cutile::core::*;

    /// Range analysis concludes that `index` is in `[0, 2]`, but the `i = 2`
    /// multiplication wraps in `i32` and produces index 294,967,296.
    #[cutile::entry]
    fn final_interval_fits_dynamic<const B: i32>(
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

    /// The inferred remainder is nonnegative, but wrapping in the dividend
    /// makes the runtime remainder negative. Force mode uses the actual value
    /// for the upper goal while still suppressing the lower guard from the
    /// inferred interval.
    #[cutile::entry]
    fn final_nonnegative_after_wrap_dynamic<const B: i32>(
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

    /// Static-extent twin: a wrap-tainted interval must not reach the static
    /// fold, and full ablation must independently test the actual value.
    #[cutile::entry]
    fn final_interval_fits_static<const B: i32>(
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

use review_probe_module::__module_ast_self;

const B: usize = 16;

fn hash_outputs(v: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    for x in v {
        x.to_bits().hash(&mut h);
    }
    h.finish()
}

fn execute_case(case: &str) -> Result<Vec<f32>, String> {
    fn e<E: std::fmt::Display>(err: E) -> String {
        err.to_string()
    }

    let x_len = if case == "dynamic_lower" {
        1_000 * B
    } else {
        3 * B
    };
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; x_len]))
        .sync()
        .map_err(e)?
        .into();
    let z = api::zeros::<f32>(&[B]).sync().map_err(e)?;
    let z = match case {
        "dynamic" => {
            let (z, _x) = review_probe_module::final_interval_fits_dynamic(z.partition([B]), x)
                .generics(vec![B.to_string()])
                .sync()
                .map_err(e)?;
            z
        }
        "dynamic_lower" => {
            let (z, _x) =
                review_probe_module::final_nonnegative_after_wrap_dynamic(z.partition([B]), x)
                    .generics(vec![B.to_string()])
                    .sync()
                    .map_err(e)?;
            z
        }
        "static" => {
            let (z, _x) = review_probe_module::final_interval_fits_static(z.partition([B]), x)
                .generics(vec![B.to_string()])
                .sync()
                .map_err(e)?;
            z
        }
        other => return Err(format!("unknown case {other}")),
    };
    z.unpartition().to_host_vec().sync().map_err(e)
}

#[test]
#[ignore = "subprocess entry point for the audit-only probe"]
fn review_probe_case_runner() {
    let Ok(case) = std::env::var("CUTILE_REVIEW_PROBE_CASE") else {
        return;
    };
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

#[derive(Debug, PartialEq)]
enum Outcome {
    Ok(String),
    Stop(String),
}

fn run_subprocess(case: &str, force_device_checks: bool) -> Outcome {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args([
        "--exact",
        "review_probe_case_runner",
        "--ignored",
        "--nocapture",
    ])
    .env("CUTILE_REVIEW_PROBE_CASE", case)
    .env_remove("CUTILE_FORCE_DEVICE_CHECKS")
    .env_remove("CUTILE_DISABLE_CHECK_HOISTING");
    if force_device_checks {
        cmd.env("CUTILE_FORCE_DEVICE_CHECKS", "1");
    }

    let out = cmd.output().expect("spawn probe subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let Some(line) = stdout.lines().find(|line| line.starts_with("OUTCOME:")) else {
        if out.status.code() != Some(0) && stderr.contains("panicked") {
            let first = stderr
                .lines()
                .find(|line| line.contains("panicked"))
                .unwrap_or("aborted");
            return Outcome::Stop(format!("aborted during cleanup: {first}"));
        }
        panic!(
            "case {case} (force={force_device_checks}) produced no OUTCOME line.\n\
             stdout:\n{stdout}\nstderr:\n{stderr}"
        );
    };

    if let Some(hash) = line.strip_prefix("OUTCOME:OK:") {
        Outcome::Ok(hash.to_string())
    } else if let Some(msg) = line.strip_prefix("OUTCOME:STOP:") {
        Outcome::Stop(msg.to_string())
    } else {
        panic!("probe infrastructure failure: {line}")
    }
}

fn compile_counts(function_name: &str) -> CheckPlacementCounts {
    KernelCompiler::new(__module_ast_self, "review_probe_module", function_name)
        .target("sm_120")
        .generics(vec![B.to_string()])
        .strides(&[("z", &[1]), ("x", &[1])])
        .compile()
        .unwrap_or_else(|err| panic!("compile {function_name}: {err}"))
        .check_counts()
}

#[test]
#[ignore = "audit regression probe intentionally trips device asserts"]
fn fixes_final_interval_overflow_and_ablation_blind_spot() {
    let (dynamic_counts, dynamic_lower_counts, static_counts) = common::with_test_stack(|| {
        (
            compile_counts("final_interval_fits_dynamic"),
            compile_counts("final_nonnegative_after_wrap_dynamic"),
            compile_counts("final_interval_fits_static"),
        )
    });
    assert_eq!(
        dynamic_counts,
        CheckPlacementCounts {
            discharged: 0,
            hoisted: 0,
            in_place: 1,
        }
    );
    assert_eq!(
        dynamic_lower_counts,
        CheckPlacementCounts {
            discharged: 0,
            hoisted: 0,
            in_place: 1,
        }
    );
    assert_eq!(
        static_counts,
        CheckPlacementCounts {
            discharged: 0,
            hoisted: 0,
            in_place: 1,
        }
    );

    for case in ["dynamic", "dynamic_lower", "static"] {
        let normal = run_subprocess(case, false);
        let forced = run_subprocess(case, true);
        assert!(
            matches!(normal, Outcome::Stop(_)),
            "normal compilation must stop on the original {case} repro: {normal:?}"
        );
        assert!(
            matches!(forced, Outcome::Stop(_)),
            "full ablation must independently stop on {case}: {forced:?}"
        );
    }
}
