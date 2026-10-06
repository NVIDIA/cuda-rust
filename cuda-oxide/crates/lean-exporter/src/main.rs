/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    env, fs,
    io::{self, Write},
    process::ExitCode,
};

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let inspect = args.first().is_some_and(|arg| arg == "--inspect");
    let args = &args[usize::from(inspect)..];
    if args.len() != 2 {
        return Err("usage: lean-exporter [--inspect] <input.mir> <function>".into());
    }
    let name = args[1].to_str().ok_or("function name must be UTF-8")?;
    let source =
        fs::read_to_string(&args[0]).map_err(|error| format!("cannot read input: {error}"))?;
    let lean =
        lean_exporter::export_named_source(&source, name).map_err(|error| error.to_string())?;
    let output = if inspect {
        format!("=== MIR input ===\n{source}\n=== Generated Lean ===\n{lean}")
    } else {
        lean
    };
    // Nothing is written until every input check and the full translation pass.
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|error| error.to_string())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("lean-exporter: {error}");
            ExitCode::FAILURE
        }
    }
}
