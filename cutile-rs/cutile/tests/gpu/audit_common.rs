/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared helpers for the `audit_*` regression modules (2026-08 codegen
//! audit): compile-only access to a kernel's IR and placement counts, host
//! transfers, and a subprocess runner for cases that are *expected* to stop
//! on the device — a device trap poisons the CUDA context of the process it
//! happens in, so such cases never run in the test process itself.

use std::process::Command;
use std::sync::Arc;

use cutile::compile_api::{CheckPlacementCounts, KernelCompiler};
use cutile::prelude::*;
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::error::JITError;

/// Compiles one kernel of `module_ast_fn`'s module for `sm_120` and returns
/// its IR text and placement counts.
pub fn compile<F: Fn() -> Module>(
    module_ast_fn: F,
    module_name: &str,
    function_name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> Result<(String, CheckPlacementCounts), JITError> {
    KernelCompiler::new(module_ast_fn, module_name, function_name)
        .target("sm_120")
        .generics(generics.iter().map(|g| g.to_string()).collect())
        .strides(strides)
        // These audits pin optimized placement, independent of Cargo's
        // development-profile device-debug default.
        .options(CompileOptions::default().device_debug(false))
        .compile()
        .map(|artifacts| (artifacts.ir_text(), artifacts.check_counts()))
}

/// Uploads host values as a rank-1 device tensor.
pub fn upload<T: DType>(values: Vec<T>) -> Arc<Tensor<T>> {
    Arc::new(
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("upload"),
    )
}

/// Copies a device tensor back to the host.
pub fn host<T: DType>(tensor: &Tensor<T>) -> Vec<T> {
    tensor.dup().to_host_vec().sync().expect("to_host")
}

/// What a subprocess case did.
#[derive(Debug)]
pub enum Outcome {
    /// The launch ran to completion.
    Ok,
    /// The launch was refused or the device trapped; carries the message.
    Stop(String),
}

/// Runs `runner_test` (an `#[ignore]`d test that reads `case` from the
/// environment variable `env_var` and prints one `OUTCOME:` line) in a fresh
/// copy of this test binary.
pub fn run_in_subprocess(runner_test: &str, env_var: &str, case: &str) -> Outcome {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args(["--exact", runner_test, "--ignored", "--nocapture"])
        .env(env_var, case)
        .output()
        .expect("spawn case subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let Some(line) = stdout.lines().find(|l| l.starts_with("OUTCOME:")) else {
        // A device trap poisons the CUDA context; the destructors that run
        // while the error propagates can panic and abort the process before
        // the runner prints. That IS a stop.
        if out.status.code() != Some(0) && stderr.contains("panicked") {
            return Outcome::Stop("aborted during cleanup".to_string());
        }
        panic!("case {case} produced no OUTCOME line.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    };
    if line == "OUTCOME:OK" {
        Outcome::Ok
    } else if let Some(msg) = line.strip_prefix("OUTCOME:STOP:") {
        Outcome::Stop(msg.to_string())
    } else {
        panic!("case {case} infrastructure failure: {line}")
    }
}

/// The runner side of [`run_in_subprocess`]: prints the `OUTCOME:` line for
/// `result`, mapping a panic to a stop as well.
pub fn report_outcome(result: std::thread::Result<Result<(), String>>) {
    match result {
        Ok(Ok(())) => println!("OUTCOME:OK"),
        Ok(Err(msg)) => println!("OUTCOME:STOP:{}", msg.replace('\n', " | ")),
        Err(_) => println!("OUTCOME:STOP:panicked"),
    }
}
