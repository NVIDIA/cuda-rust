/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    ffi::OsStr,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Output},
};

use crate::{backend, backend_source};

use super::{Context, context::example_roots};

const DEMO_INPUT: &str = "crates/lean-exporter/fixtures/arithmetic.mir";
const DEMO_FUNCTION: &str = "global_index";
const DEMO_NOTICE: &str = "Demo: hand-authored scalar MIR, not extracted from a Rust kernel.\n\n";

fn resolve_verified_example_functions(
    ctx: &Context,
    workspace: &Path,
    example_name: &str,
    explicit_fn: Option<&str>,
) -> Result<(PathBuf, Vec<String>), String> {
    let canonical_name = match example_name {
        "verified_vecadd" => "safe_vecadd",
        "verified_gemm_views" => "safe_gemm_views",
        "verified_warp_reduce" => "safe_warp_reduce",
        "verified_subwarp_stage" => "safe_subwarp_stage",
        "verified_async_copy" => "safe_async_copy",
        "verified_tcgen05_matmul" => "safe_tcgen05_matmul",
        other => other,
    };
    let roots = example_roots(ctx);
    let matches: Vec<_> = roots
        .iter()
        .map(|root| root.join(canonical_name))
        .filter(|path| path.is_dir())
        .collect();
    let example_dir = match matches.as_slice() {
        [dir] => dir.clone(),
        [] => return Err("custom input and --function must be supplied together".into()),
        _ => return Err(format!("example name is ambiguous: {example_name}")),
    };

    let main_rs = example_dir.join("src/main.rs");
    let source = fs::read_to_string(&main_rs).unwrap_or_default();
    if !source.contains("#[verify(") {
        return Err(format!(
            "example `{example_name}` does not use `#[verify(...)]`; only provably safe examples can be exported with `cargo oxide lean-export <example>`"
        ));
    }

    let (fixture_rel, default_fns): (&str, &[&str]) = match canonical_name {
        "safe_vecadd" => ("crates/lean-exporter/fixtures/vecadd.mir", &["vecadd"]),
        "safe_gemm_views" => (
            "crates/lean-exporter/fixtures/sgemm_tiled.mir",
            &["sgemm_tiled_views", "sgemm_tiled_raw"],
        ),
        "safe_warp_reduce" => (
            "crates/lean-exporter/fixtures/warp_reduce.mir",
            &["warp_reduce_sum", "block_reduce_sum"],
        ),
        "safe_subwarp_stage" => (
            "crates/lean-exporter/fixtures/subwarp_multibuf.mir",
            &["subwarp_bit_split_stage", "multibuf_subwarp_stage"],
        ),
        "safe_async_copy" => (
            "crates/lean-exporter/fixtures/async_copy_stage.mir",
            &["async_copy_stage"],
        ),
        "safe_tcgen05_matmul" => (
            "crates/lean-exporter/fixtures/tcgen05_matmul.mir",
            &["tcgen05_matmul_verified"],
        ),
        _ => {
            return Err(format!(
                "no PostMem2Reg MIR fixture registered for example `{example_name}`"
            ));
        }
    };
    let fns = match explicit_fn {
        Some(f) => vec![f.to_string()],
        None => default_fns.iter().map(|s| (*s).to_string()).collect(),
    };
    Ok((workspace.join(fixture_rel), fns))
}

/// Display input MIR and the generated scalar Lean model without building the
/// device compiler, loading CUDA, or running Lean. The exporter remains a
/// separate workspace tool instead of a dependency of the installed CLI.
pub fn lean_export(ctx: &Context, input: Option<&Path>, function: Option<&str>) {
    let result = std::env::current_dir()
        .map_err(|error| format!("cannot determine current directory: {error}"))
        .and_then(|invocation_dir| {
            let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            let commands = exporter_commands(ctx, &invocation_dir, input, function, &cargo)?;
            let mut combined = String::new();
            for mut command in commands {
                let output = command
                    .output()
                    .map_err(|error| format!("could not start the scalar MIR exporter: {error}"))?;
                let rendered = inspection_output(output, input.is_none())?;
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&rendered);
            }
            io::stdout()
                .lock()
                .write_all(combined.as_bytes())
                .map_err(|error| format!("cannot write inspection output: {error}"))
        });
    if let Err(error) = result {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn exporter_commands(
    ctx: &Context,
    invocation_dir: &Path,
    input: Option<&Path>,
    function: Option<&str>,
    cargo: &OsStr,
) -> Result<Vec<Command>, String> {
    let exporter = backend::simt_crate_path(&ctx.workspace_root, "lean-exporter");
    if !ctx.is_workspace || !exporter.join("Cargo.toml").is_file() {
        return Err("`cargo oxide lean-export` must be run inside a cuda-oxide source checkout containing crates/lean-exporter".into());
    }
    let workspace = exporter
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot locate the CUDA Oxide workspace for lean-exporter")?;
    let (input_path, functions): (PathBuf, Vec<String>) = match (input, function) {
        (None, None) => (workspace.join(DEMO_INPUT), vec![DEMO_FUNCTION.to_string()]),
        (Some(path), Some(function)) => {
            if path.extension().and_then(|e| e.to_str()) != Some("mir")
                && let Some(name) = path.to_str()
                && !name.contains('/')
                && !name.contains('\\')
                && let Ok(resolved) =
                    resolve_verified_example_functions(ctx, workspace, name, Some(function))
            {
                resolved
            } else {
                let absolute = if path.is_absolute() {
                    path.to_owned()
                } else {
                    invocation_dir.join(path)
                };
                (absolute, vec![function.to_string()])
            }
        }
        (Some(path), None) => {
            if let Some(name) = path.to_str()
                && path.extension().and_then(|e| e.to_str()) != Some("mir")
                && !name.contains('/')
                && !name.contains('\\')
            {
                resolve_verified_example_functions(ctx, workspace, name, None)?
            } else {
                return Err("custom input and --function must be supplied together".into());
            }
        }
        (None, Some(_)) => {
            return Err("custom input and --function must be supplied together".into());
        }
    };
    let mut commands = Vec::new();
    for fn_name in functions {
        let mut command = Command::new(cargo);
        // A root-level Cargo invocation can pass the stable host toolchain through
        // RUSTUP_TOOLCHAIN. This source tool belongs to the nested SIMT workspace.
        if let Some(channel) = backend_source::pinned_channel(workspace) {
            command.env("RUSTUP_TOOLCHAIN", channel);
        }
        command
            .current_dir(workspace)
            .args(["run", "--quiet", "--manifest-path"])
            .arg(workspace.join("Cargo.toml"))
            .args(["-p", "lean-exporter", "--", "--inspect"])
            .arg(&input_path)
            .arg(fn_name);
        commands.push(command);
    }
    Ok(commands)
}

#[cfg(test)]
fn exporter_command(
    ctx: &Context,
    invocation_dir: &Path,
    input: Option<&Path>,
    function: Option<&str>,
    cargo: &OsStr,
) -> Result<Command, String> {
    exporter_commands(ctx, invocation_dir, input, function, cargo).map(|mut cmds| cmds.remove(0))
}

fn inspection_output(output: Output, is_demo: bool) -> Result<String, String> {
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        return Err(if detail.is_empty() {
            format!("scalar MIR export failed ({})", output.status)
        } else {
            format!("scalar MIR export failed:\n{detail}")
        });
    }
    let output = String::from_utf8(output.stdout)
        .map_err(|error| format!("exporter returned invalid UTF-8: {error}"))?;
    Ok(if is_demo {
        format!("{DEMO_NOTICE}{output}")
    } else {
        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf};

    struct Workspace(PathBuf);

    impl Workspace {
        fn new() -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "cargo oxide lean export {} {unique}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("crates/lean-exporter")).unwrap();
            fs::write(root.join("crates/lean-exporter/Cargo.toml"), "").unwrap();
            Self(root)
        }

        fn context(&self) -> Context {
            Context {
                workspace_root: self.0.clone(),
                codegen_crate: self.0.join("missing-backend"),
                examples_dir: self.0.join("missing-examples"),
                backend_so: self.0.join("missing-backend.so"),
                is_workspace: true,
                config: Default::default(),
            }
        }
    }

    impl Drop for Workspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn lean_export_command_selects_demo_without_backend_or_prover() {
        let workspace = Workspace::new();
        let ctx = workspace.context();
        let command =
            exporter_command(&ctx, &workspace.0, None, None, OsStr::new("cargo")).unwrap();
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(command.get_program(), "cargo");
        assert_eq!(command.get_current_dir(), Some(workspace.0.as_path()));
        assert_eq!(args[0..3], ["run", "--quiet", "--manifest-path"]);
        assert_eq!(args[3], workspace.0.join("Cargo.toml"));
        assert_eq!(args[4..8], ["-p", "lean-exporter", "--", "--inspect"]);
        assert_eq!(args[8], workspace.0.join(DEMO_INPUT));
        assert_eq!(args[9], DEMO_FUNCTION);
        assert_eq!(
            command.get_envs().count(),
            0,
            "no backend configuration is applied"
        );
    }

    #[test]
    fn lean_export_resolves_relative_paths_from_invocation_directory() {
        let workspace = Workspace::new();
        let caller = workspace.0.join("nested caller");
        let input = Path::new("input with spaces.mir");
        let command = exporter_command(
            &workspace.context(),
            &caller,
            Some(input),
            Some("chosen"),
            OsStr::new("cargo"),
        )
        .unwrap();
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(args[8], caller.join(input));
        assert_eq!(args[9], "chosen");

        let absolute = workspace.0.join("absolute file.mir");
        let command = exporter_command(
            &workspace.context(),
            &caller,
            Some(&absolute),
            Some("chosen"),
            OsStr::new("cargo"),
        )
        .unwrap();
        assert_eq!(command.get_args().nth(8), Some(absolute.as_os_str()));
    }

    #[test]
    fn lean_export_uses_simt_workspace_from_combined_checkout() {
        let workspace = Workspace::new();
        let simt = workspace.0.join("cuda-oxide");
        fs::create_dir(&simt).unwrap();
        fs::rename(workspace.0.join("crates"), simt.join("crates")).unwrap();
        fs::create_dir_all(simt.join("crates/rustc-codegen-cuda")).unwrap();
        fs::write(simt.join("crates/rustc-codegen-cuda/Cargo.toml"), "").unwrap();
        fs::write(
            simt.join("rust-toolchain.toml"),
            r#"[toolchain]
channel = "nightly-test"
"#,
        )
        .unwrap();
        let caller = workspace.0.join("caller");
        for root in [&workspace.0, &simt] {
            let mut ctx = workspace.context();
            ctx.workspace_root = root.clone();
            let demo = exporter_command(&ctx, &caller, None, None, OsStr::new("cargo"))
                .expect("export from either repository root");
            assert_eq!(demo.get_current_dir(), Some(simt.as_path()));
            assert_eq!(
                demo.get_args().nth(3),
                Some(simt.join("Cargo.toml").as_os_str())
            );
            assert_eq!(
                demo.get_args().nth(8),
                Some(simt.join(DEMO_INPUT).as_os_str())
            );
            assert!(demo.get_envs().any(|(name, value)| {
                name == "RUSTUP_TOOLCHAIN" && value == Some(OsStr::new("nightly-test"))
            }));
            let custom = exporter_command(
                &ctx,
                &caller,
                Some(Path::new("input.mir")),
                Some("chosen"),
                OsStr::new("cargo"),
            )
            .unwrap();
            assert_eq!(
                custom.get_args().nth(8),
                Some(caller.join("input.mir").as_os_str())
            );
        }
    }

    #[test]
    fn lean_export_rejects_non_checkouts_and_partial_options() {
        let workspace = Workspace::new();
        let mut ctx = workspace.context();
        for (input, function) in [(Some(Path::new("x.mir")), None), (None, Some("x"))] {
            assert!(
                exporter_command(&ctx, &workspace.0, input, function, OsStr::new("cargo")).is_err()
            );
        }
        ctx.is_workspace = false;
        assert!(
            exporter_command(&ctx, &workspace.0, None, None, OsStr::new("cargo"))
                .unwrap_err()
                .contains("source checkout")
        );
    }

    #[cfg(unix)]
    #[test]
    fn lean_export_discards_partial_stdout_on_subprocess_failure() {
        let output = Command::new("sh")
            .args([
                "-c",
                "printf 'partial Lean'; printf 'unsupported operation' >&2; exit 7",
            ])
            .output()
            .unwrap();
        let error = inspection_output(output, true).unwrap_err();
        assert!(error.contains("unsupported operation"));
        assert!(!error.contains("partial Lean"));
        assert!(!error.contains("Demo:"));
    }

    #[cfg(unix)]
    #[test]
    fn lean_export_adds_demo_provenance_only_after_success() {
        let output = Command::new("sh")
            .args([
                "-c",
                "printf '=== MIR input ===\\nsource\\n=== Generated Lean ===\\nmodel\\n'",
            ])
            .output()
            .unwrap();
        let text = inspection_output(output, true).unwrap();
        assert!(text.starts_with(DEMO_NOTICE));
        assert!(text.contains("=== MIR input ===\nsource"));
        assert!(text.contains("=== Generated Lean ===\nmodel"));
    }
}
