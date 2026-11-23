/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    ffi::OsStr,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use crate::{backend, backend_source};

use super::*;

const DEMO_INPUT: &str = "crates/lean-exporter/fixtures/arithmetic.mir";
const DEMO_FUNCTION: &str = "global_index";
const DEMO_NOTICE: &str = "Demo: hand-authored scalar MIR, not extracted from a Rust kernel.\n\n";

const LEAN_EXPORT_FILE_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_FILE";
const LEAN_EXPORT_FUNCTION_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_FUNCTION";
const LEAN_EXPORT_DISCOVERED_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_DISCOVERED_FILE";
const LEAN_EXPORT_ONLY_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_ONLY";

/// Command-line options forwarded to `cargo oxide lean-export`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeanExportOptions<'a> {
    /// Example/crate name or raw `.mir` file path.
    pub input: Option<&'a Path>,
    /// Optional kernel function filter (required for raw `.mir` inputs).
    pub function: Option<&'a str>,
    /// Optional target GPU architecture override (`sm_90`, `sm_100a`, `sm_120`).
    pub arch: Option<&'a str>,
    /// Optional comma-separated Cargo feature list.
    pub features: Option<&'a str>,
    /// Whether to stream compiler progress to stderr.
    pub verbose: bool,
}

enum LeanExportTarget {
    /// Zero-argument scalar MIR demo inside a `cuda-oxide` source checkout.
    WorkspaceDemo,
    /// Explicit `.mir` file + `--function <name>` routed to `lean-exporter --inspect`.
    RawMirFile { path: PathBuf, function: String },
    /// Live Rust example, model, or standalone project compiled to `PostMem2Reg`.
    RustCrate {
        display_name: String,
        crate_dir: PathBuf,
    },
}

fn canonicalize_legacy_example_name(name: &str) -> &str {
    match name {
        "verified_vecadd" => "safe_vecadd",
        "verified_gemm_views" => "safe_gemm_views",
        "verified_warp_reduce" => "safe_warp_reduce",
        "verified_subwarp_stage" => "safe_subwarp_stage",
        "verified_async_copy" => "safe_async_copy",
        "verified_tcgen05_matmul" => "safe_tcgen05_matmul",
        other => other,
    }
}

fn is_raw_mir_path(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("mir")
        || path
            .to_str()
            .is_some_and(|text| text.contains('/') || text.contains('\\'))
}

fn classify_target(
    ctx: &Context,
    invocation_dir: &Path,
    opts: &LeanExportOptions<'_>,
) -> Result<LeanExportTarget, String> {
    match (opts.input, opts.function) {
        (None, Some(_)) if ctx.is_workspace => {
            Err("custom input and --function must be supplied together".into())
        }
        (None, None) if ctx.is_workspace => {
            if opts.arch.is_some() || opts.features.is_some() {
                return Err("--arch and --features require an example or crate target".into());
            }
            Ok(LeanExportTarget::WorkspaceDemo)
        }
        (None, _) => {
            let display_name = ctx
                .workspace_root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project")
                .to_string();
            Ok(LeanExportTarget::RustCrate {
                display_name,
                crate_dir: ctx.workspace_root.clone(),
            })
        }
        (Some(path), fn_opt) if is_raw_mir_path(path) => {
            if opts.arch.is_some() || opts.features.is_some() {
                return Err(
                    "--arch and --features apply to Rust crates, not raw `.mir` files".into(),
                );
            }
            let Some(function) = fn_opt else {
                return Err("custom input and --function must be supplied together".into());
            };
            let absolute = if path.is_absolute() {
                path.to_owned()
            } else {
                invocation_dir.join(path)
            };
            Ok(LeanExportTarget::RawMirFile {
                path: absolute,
                function: function.to_string(),
            })
        }
        (Some(path), _) => {
            let raw_name = path
                .to_str()
                .ok_or("target crate or example name must be valid UTF-8")?;
            let canonical = canonicalize_legacy_example_name(raw_name);
            if ctx.is_workspace {
                let roots = example_roots(ctx);
                let matches: Vec<_> = roots
                    .iter()
                    .map(|root| root.join(canonical))
                    .filter(|dir| dir.is_dir())
                    .collect();
                match matches.as_slice() {
                    [dir] => Ok(LeanExportTarget::RustCrate {
                        display_name: canonical.to_string(),
                        crate_dir: dir.clone(),
                    }),
                    [] => Err(format!(
                        "unknown example or model `{raw_name}` (pass `<file.mir> --function <name>` to inspect a raw MIR file)"
                    )),
                    _ => Err(format!("example name is ambiguous: {raw_name}")),
                }
            } else {
                let standalone_name = ctx
                    .workspace_root
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if raw_name == standalone_name || canonical == standalone_name {
                    Ok(LeanExportTarget::RustCrate {
                        display_name: standalone_name.to_string(),
                        crate_dir: ctx.workspace_root.clone(),
                    })
                } else {
                    Err(format!(
                        "standalone project is `{standalone_name}`, not `{raw_name}` (run `cargo oxide lean-export` without a crate name, or pass `<file.mir> --function <name>`)"
                    ))
                }
            }
        }
    }
}

/// Recursively scan all `.rs` files under `dir` for non-comment `#[verify` attributes.
fn dir_contains_verify_attr(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if dir_contains_verify_attr(&path) {
                return true;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && let Ok(source) = fs::read_to_string(&path)
            && source_has_verify_attr(&source)
        {
            return true;
        }
    }
    false
}

fn source_has_verify_attr(source: &str) -> bool {
    source.lines().any(|line| {
        let code = line.split("//").next().unwrap_or("");
        code.contains("#[verify(") || code.contains("#[cuda_device::verify(")
    })
}

fn crate_uses_verify(crate_dir: &Path) -> bool {
    if dir_contains_verify_attr(&crate_dir.join("src")) {
        return true;
    }
    if crate_dir.join("Cargo.toml").is_file()
        && let Some(interop) = load_interop_config(crate_dir)
    {
        for dev in &interop.device_crates {
            if let Some(dev_dir) = crate_dir.join(&dev.manifest_path).parent()
                && dir_contains_verify_attr(&dev_dir.join("src"))
            {
                return true;
            }
        }
    }
    false
}

/// Snapshot existing `.ptx` files before an export-only compile so they can be
/// restored (or newly created ones removed) once capture completes.
struct PtxSnapshot {
    entries: Vec<(PathBuf, Option<Vec<u8>>)>,
}

impl PtxSnapshot {
    fn capture(crate_dir: &Path, display_name: &str) -> Self {
        let entries = ptx_artifact_paths(crate_dir, display_name)
            .into_iter()
            .map(|path| {
                let prev = fs::read(&path).ok();
                (path, prev)
            })
            .collect();
        Self { entries }
    }
}

impl Drop for PtxSnapshot {
    fn drop(&mut self) {
        for (path, prev) in &self.entries {
            match prev {
                Some(bytes) => {
                    let _ = fs::write(path, bytes);
                }
                None => {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
}

/// Export `PostMem2Reg` MIR and generated Lean 4 proof modules for a verified
/// Rust crate/example, a raw `.mir` file, or the workspace scalar demo.
pub fn lean_export(passive_ctx: &Context, opts: LeanExportOptions<'_>) {
    let result = std::env::current_dir()
        .map_err(|error| format!("cannot determine current directory: {error}"))
        .and_then(
            |invocation_dir| match classify_target(passive_ctx, &invocation_dir, &opts)? {
                LeanExportTarget::WorkspaceDemo => {
                    run_mir_exporter_cli(passive_ctx, &invocation_dir, None, None)
                }
                LeanExportTarget::RawMirFile { path, function } => {
                    run_mir_exporter_cli(passive_ctx, &invocation_dir, Some(&path), Some(&function))
                }
                LeanExportTarget::RustCrate {
                    display_name,
                    crate_dir,
                } => export_live_rust_crate(&display_name, &crate_dir, &opts),
            },
        );
    if let Err(error) = result {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn export_live_rust_crate(
    display_name: &str,
    crate_dir: &Path,
    opts: &LeanExportOptions<'_>,
) -> Result<(), String> {
    if !crate_uses_verify(crate_dir) {
        return Err(format!(
            "example `{display_name}` does not use `#[verify(...)]`; only provably safe crates can be exported with `cargo oxide lean-export`"
        ));
    }

    let ctx = if opts.verbose {
        resolve_context()
    } else {
        resolve_context_quiet_stdout()
    };
    let target_arch = configured_arch(&ctx, opts.arch);
    let materialization = MaterializationMode::default();
    let _ptx_guard = PtxSnapshot::capture(crate_dir, display_name);

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let export_path =
        std::env::temp_dir().join(format!("cuda_oxide_lean_export_{pid}_{unique}.txt"));
    let discovered_path =
        std::env::temp_dir().join(format!("cuda_oxide_lean_discovered_{pid}_{unique}.txt"));

    touch_main_rs(crate_dir);
    if let Some(interop) = load_interop_config(crate_dir) {
        for dev in &interop.device_crates {
            if let Some(dev_dir) = crate_dir.join(&dev.manifest_path).parent() {
                touch_main_rs(dev_dir);
            }
        }
    }

    let build_dirs: Vec<PathBuf> =
        match load_interop_config(crate_dir).filter(|cfg| !cfg.device_crates.is_empty()) {
            Some(interop) => interop
                .device_crates
                .iter()
                .filter_map(|dev| {
                    crate_dir
                        .join(&dev.manifest_path)
                        .parent()
                        .map(Path::to_path_buf)
                })
                .collect(),
            None => vec![crate_dir.to_path_buf()],
        };

    let fingerprint =
        lean_export_codegen_fingerprint(&ctx, opts.verbose, target_arch, &materialization);
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());

    for build_dir in &build_dirs {
        let mut cmd = Command::new(&cargo);
        cmd.args(["build", "--release"]).current_dir(build_dir);
        if let Some(features) = opts.features {
            cmd.args(["--features", features]);
        }
        apply_common_codegen_env(&mut cmd, &ctx, opts.verbose, false, false, DeviceDebug::Off);
        apply_codegen_configuration_or_exit(
            &mut cmd,
            &ctx,
            CodegenProfilePolicy::ReleaseLike,
            &[],
            &fingerprint,
        );
        apply_output_mode(&mut cmd, false, target_arch, &materialization);
        cmd.env(LEAN_EXPORT_ONLY_ENV, "1");
        cmd.env(LEAN_EXPORT_FILE_ENV, &export_path);
        cmd.env(LEAN_EXPORT_DISCOVERED_ENV, &discovered_path);
        if let Some(fn_name) = opts.function {
            cmd.env(LEAN_EXPORT_FUNCTION_ENV, fn_name);
        }

        if opts.verbose {
            let status = cmd
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(|e| format!("failed to run cargo build: {e}"))?;
            if !status.success() {
                let _ = fs::remove_file(&export_path);
                let _ = fs::remove_file(&discovered_path);
                return Err(format!(
                    "device compilation failed for `{display_name}` (exit {})",
                    status
                ));
            }
        } else {
            let output = cmd
                .output()
                .map_err(|e| format!("failed to run cargo build: {e}"))?;
            if !output.status.success() {
                let _ = fs::remove_file(&export_path);
                let _ = fs::remove_file(&discovered_path);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = stderr.trim();
                return Err(if detail.is_empty() {
                    format!(
                        "device compilation failed for `{display_name}` ({})",
                        output.status
                    )
                } else {
                    format!("device compilation failed for `{display_name}`:\n{detail}")
                });
            }
        }
    }

    let exported = fs::read_to_string(&export_path).unwrap_or_default();
    let discovered = fs::read_to_string(&discovered_path).unwrap_or_default();
    let _ = fs::remove_file(&export_path);
    let _ = fs::remove_file(&discovered_path);

    if exported.trim().is_empty() {
        let mut names: Vec<&str> = discovered
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        names.dedup();
        if let Some(wanted) = opts.function {
            if names.is_empty() {
                return Err(format!(
                    "no `#[verify(...)]` kernels were compiled in `{display_name}`"
                ));
            }
            return Err(format!(
                "verified kernel `{wanted}` not found in `{display_name}`; available `#[verify]` kernels: {}",
                names.join(", ")
            ));
        }
        return Err(format!(
            "no `#[verify(...)]` kernels were compiled in `{display_name}`"
        ));
    }

    write_stdout(exported.as_bytes())
}

fn write_stdout(bytes: &[u8]) -> Result<(), String> {
    match io::stdout().lock().write_all(bytes) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(err) => Err(format!("cannot write inspection output: {err}")),
    }
}

fn run_mir_exporter_cli(
    ctx: &Context,
    invocation_dir: &Path,
    input: Option<&Path>,
    function: Option<&str>,
) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = exporter_command(ctx, invocation_dir, input, function, &cargo)?;
    let output = command
        .output()
        .map_err(|error| format!("could not start the MIR exporter: {error}"))?;
    let rendered = inspection_output(output, input.is_none())?;
    write_stdout(rendered.as_bytes())
}

fn exporter_command(
    ctx: &Context,
    invocation_dir: &Path,
    input: Option<&Path>,
    function: Option<&str>,
    cargo: &OsStr,
) -> Result<Command, String> {
    let exporter = backend::simt_crate_path(&ctx.workspace_root, "lean-exporter");
    if !ctx.is_workspace || !exporter.join("Cargo.toml").is_file() {
        return Err("`cargo oxide lean-export` on `.mir` files or the workspace demo must be run inside a cuda-oxide source checkout containing crates/lean-exporter".into());
    }
    let workspace = exporter
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot locate the CUDA Oxide workspace for lean-exporter")?;
    let (input_path, fn_name): (PathBuf, String) = match (input, function) {
        (None, None) => (workspace.join(DEMO_INPUT), DEMO_FUNCTION.to_string()),
        (Some(path), Some(function)) => {
            let absolute = if path.is_absolute() {
                path.to_owned()
            } else {
                invocation_dir.join(path)
            };
            (absolute, function.to_string())
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err("custom input and --function must be supplied together".into());
        }
    };
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
    Ok(command)
}

fn inspection_output(output: Output, is_demo: bool) -> Result<String, String> {
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        return Err(if detail.is_empty() {
            format!("MIR export failed ({})", output.status)
        } else {
            format!("MIR export failed:\n{detail}")
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

    #[test]
    fn crate_uses_verify_scans_nested_submodules_and_ignores_comments() {
        let workspace = Workspace::new();
        let example_dir = workspace.0.join("examples/safe_gemm_like");
        fs::create_dir_all(example_dir.join("src")).unwrap();
        fs::write(
            example_dir.join("src/main.rs"),
            "mod kernels;\n// #[verify(no_oob_idx(a))]\nfn main() {}\n",
        )
        .unwrap();
        assert!(
            !crate_uses_verify(&example_dir),
            "commented-out #[verify] must not count"
        );
        fs::write(
            example_dir.join("src/kernels.rs"),
            "#[verify(no_oob_idx(SMEM), uniform_barriers)]\npub fn gemm() {}\n",
        )
        .unwrap();
        assert!(
            crate_uses_verify(&example_dir),
            "#[verify] inside src/kernels.rs must be discovered"
        );
    }

    #[test]
    fn classify_target_distinguishes_demo_mir_examples_and_standalone() {
        let workspace = Workspace::new();
        let mut ctx = workspace.context();
        ctx.codegen_crate = workspace.0.join("crates/rustc-codegen-cuda");
        let ex_root = workspace.0.join("examples/safe_gemm_sol_final/src");
        fs::create_dir_all(&ex_root).unwrap();

        assert!(matches!(
            classify_target(&ctx, &workspace.0, &LeanExportOptions::default()).unwrap(),
            LeanExportTarget::WorkspaceDemo
        ));
        assert!(
            classify_target(
                &ctx,
                &workspace.0,
                &LeanExportOptions {
                    function: Some("k"),
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(matches!(
            classify_target(
                &ctx,
                &workspace.0,
                &LeanExportOptions {
                    input: Some(Path::new("foo.mir")),
                    function: Some("k"),
                    ..Default::default()
                }
            )
            .unwrap(),
            LeanExportTarget::RawMirFile { .. }
        ));
        match classify_target(
            &ctx,
            &workspace.0,
            &LeanExportOptions {
                input: Some(Path::new("safe_gemm_sol_final")),
                ..Default::default()
            },
        )
        .unwrap()
        {
            LeanExportTarget::RustCrate {
                display_name,
                crate_dir,
            } => {
                assert_eq!(display_name, "safe_gemm_sol_final");
                assert_eq!(crate_dir, workspace.0.join("examples/safe_gemm_sol_final"));
            }
            _ => panic!("expected RustCrate target"),
        }

        // Standalone mode defaults to the current project even with `--function`.
        ctx.is_workspace = false;
        assert!(matches!(
            classify_target(
                &ctx,
                &workspace.0,
                &LeanExportOptions {
                    function: Some("my_kernel"),
                    ..Default::default()
                }
            )
            .unwrap(),
            LeanExportTarget::RustCrate { .. }
        ));
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
