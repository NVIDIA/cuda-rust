/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    dir: PathBuf,
    input: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("ptx-schedule-cli-{}-{id}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.ptx");
        fs::write(
            &input,
            ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() {\n  bar.sync 0;\n  ret;\n}\n",
        )
        .unwrap();
        Self { dir, input }
    }

    fn output(&self) -> PathBuf {
        self.dir.join("output.ptx")
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_ptx-schedule"))
            .arg(&self.input)
            .args(args)
            .output()
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn run_without_input(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ptx-schedule"))
        .args(args)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_usage_error(output: Output, fragment: &str, absent: Option<&Path>) {
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(stderr(&output).contains(fragment), "{}", stderr(&output));
    if let Some(path) = absent {
        assert!(!path.exists(), "unexpected output at {}", path.display());
    }
}

#[test]
fn help_shows_the_two_exclusive_forms() {
    let output = run_without_input(&["--help"]);
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ptx-schedule <INPUT.ptx> --list-sites"));
    assert!(stdout.contains("ptx-schedule <INPUT.ptx> -o <OUTPUT> [OPTIONS]"));
}

#[test]
fn malformed_commands_are_named_usage_errors() {
    assert_usage_error(run_without_input(&[]), "<INPUT.ptx>", None);

    let fixture = Fixture::new();
    let output_path = fixture.output();
    for (args, fragment) in [
        (vec!["--seed"], "value is required"),
        (vec!["--focus", "--list-sites"], "value is required"),
        (vec!["--seed", "abc", "-o", "unused.ptx"], "invalid value"),
        (vec!["--bogus"], "unexpected argument"),
        (vec!["--list-sites", "--seed", "1"], "cannot be used with"),
        (vec!["--seed", "1"], "required"),
    ] {
        assert_usage_error(fixture.run(&args), fragment, Some(&output_path));
    }
}

#[test]
fn both_valid_modes_execute() {
    let fixture = Fixture::new();
    let listed = fixture.run(&["--list-sites"]);
    assert!(listed.status.success(), "{listed:?}");
    assert!(String::from_utf8_lossy(&listed.stdout).contains("bar.sync 0"));

    let output_path = fixture.output();
    let rewritten = fixture.run(&["--seed", "7", "--output", output_path.to_str().unwrap()]);
    assert!(rewritten.status.success(), "{rewritten:?}");
    assert!(output_path.is_file());
}
