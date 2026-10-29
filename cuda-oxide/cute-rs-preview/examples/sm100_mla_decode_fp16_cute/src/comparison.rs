/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Default CuTeDSL comparison dispatch and the serialized-input adapter boundary.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

/// Run help or the Python comparison; return false only for serialized-input adapter mode.
pub fn dispatch(arguments: &[OsString]) -> super::Result<bool> {
    // compare.py re-enters this executable with serialized inputs for the Rust run.
    if arguments.iter().any(|argument| argument == "--input-dir") {
        return Ok(false);
    }
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(
            "sm100_mla_decode_fp16_cute [--batch 1 --kv-length 1024 --splits 2] [--clusters N]\n  [--warmup 10 --iters 100 --windows 3] [--check-only] [--json RESULT.json]\n  [--softmax-scale 0.0416 --output-scale 1.0] [--seed 1111]\n  [--pattern random|zero-query] [--timing-order cutedsl-first|rust-first] [--timeout 180]\nDefault: validate and compare CuTeDSL and CuTe-RS, then print two timing lines.\n--check-only validates both implementations without timing.\nAliases: --iterations, --rounds, --validation-only.\nPython selection: CUDA_OXIDE_CUTE_PYTHON, example .venv, cute-rs-preview/.venv, then python3.\nRequires local compare.py and reference/ assets.\n\nAdapter mode: --input-dir DIR --output-dir DIR [--clusters N]\n  [--validation-only] [--warmup 10 --iterations 100 --rounds 3]\nRuns CuTe-RS on serialized inputs and emits JSON for compare.py.\nUse --input-dir DIR --help for the input format."
        );
        return Ok(true);
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var_os("CUDA_OXIDE_CUTE_PYTHON")
        .map(PathBuf::from)
        .or_else(|| {
            [
                directory.join(".venv/bin/python"),
                directory.join("../../.venv/bin/python"),
            ]
            .into_iter()
            .find(|path| path.is_file())
        })
        .unwrap_or_else(|| PathBuf::from("python3"));
    let mut command = comparison_command(directory, &python, &std::env::current_exe()?, arguments)?;
    run(&mut command)?;
    Ok(true)
}

/// Check local reference assets and forward arguments, pinning the currently running Rust binary.
fn comparison_command(
    directory: &Path,
    python: &Path,
    binary: &Path,
    arguments: &[OsString],
) -> super::Result<Command> {
    let script = directory.join("compare.py");
    if !script.is_file() || !directory.join("reference").is_dir() {
        return Err(format!(
            "CuTeDSL comparison requires compare.py and reference/ in {}",
            directory.display()
        )
        .into());
    }
    let mut command = Command::new(python);
    // The runner must benchmark this executable even if a forwarded argument supplies another path.
    command
        .arg(script)
        .args(arguments)
        .arg("--rust-bin")
        .arg(binary);
    Ok(command)
}

/// Run the comparison subprocess and report startup or nonzero-exit errors.
fn run(command: &mut Command) -> super::Result<()> {
    let status = command.status().map_err(|error| {
        format!(
            "could not start CuTeDSL comparison with {}: {error}; create cute-rs-preview/.venv as the README shows, or set CUDA_OXIDE_CUTE_PYTHON",
            command.get_program().to_string_lossy()
        )
    })?;
    if !status.success() {
        return Err(format!("CuTeDSL comparison failed ({status})").into());
    }
    Ok(())
}

/// Check argument forwarding and adapter dispatch without launching Python or CUDA.
#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};

    /// Temporary executable that records arguments and simulates a comparison exit status.
    struct FakePython {
        /// Private temporary directory holding reference placeholders, the executable and captured arguments.
        directory: PathBuf,
    }

    impl FakePython {
        /// Create an isolated recording executable and minimal comparison assets.
        fn new() -> Self {
            /// Per-process sequence preventing temporary-directory collisions between concurrent tests.
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let directory = std::env::temp_dir().join(format!(
                "decode-cli-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            fs::create_dir(directory.join("reference")).unwrap();
            fs::write(directory.join("compare.py"), b"").unwrap();
            let python = directory.join("python");
            fs::write(
                &python,
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$COMPARISON_ARGUMENTS\"\nexit \"${COMPARISON_EXIT:-0}\"\n",
            )
            .unwrap();
            fs::set_permissions(python, fs::Permissions::from_mode(0o755)).unwrap();
            Self { directory }
        }

        /// Build a comparison command whose simulated interpreter records every argument.
        fn command(&self, arguments: &[OsString]) -> Command {
            let mut command = comparison_command(
                &self.directory,
                &self.directory.join("python"),
                Path::new("/path to/running-binary"),
                arguments,
            )
            .unwrap();
            command.env("COMPARISON_ARGUMENTS", self.directory.join("arguments"));
            command
        }
    }

    impl Drop for FakePython {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    /// Forward every option unchanged and keep a binary path containing spaces as one argument.
    #[test]
    fn comparison_forwards_options_and_preserves_binary_paths_with_spaces() {
        let python = FakePython::new();
        let arguments = [
            "--batch",
            "2",
            "--kv-length",
            "129",
            "--splits",
            "3",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--windows",
            "2",
            "--check-only",
        ]
        .map(OsString::from);
        run(&mut python.command(&arguments)).unwrap();
        let passed = fs::read_to_string(python.directory.join("arguments")).unwrap();
        let passed: Vec<_> = passed.lines().collect();
        assert!(passed[0].ends_with("/compare.py"));
        assert_eq!(
            &passed[1..],
            &[
                "--batch",
                "2",
                "--kv-length",
                "129",
                "--splits",
                "3",
                "--warmup",
                "0",
                "--iters",
                "1",
                "--windows",
                "2",
                "--check-only",
                "--rust-bin",
                "/path to/running-binary",
            ]
        );
    }

    /// Propagate a nonzero comparison exit while preserving evidence that it was launched.
    #[test]
    fn failed_comparison_is_an_error() {
        let python = FakePython::new();
        let mut command = python.command(&[]);
        command.env("COMPARISON_EXIT", "7");
        assert!(run(&mut command).is_err());
        assert!(python.directory.join("arguments").is_file());
    }

    /// Serialized input arguments must bypass the Python comparison dispatcher.
    #[test]
    fn adapter_input_does_not_reenter_comparison() {
        let arguments = ["--input-dir", "/inputs", "--output-dir", "/outputs"].map(OsString::from);
        assert!(!dispatch(&arguments).unwrap());
    }
}
