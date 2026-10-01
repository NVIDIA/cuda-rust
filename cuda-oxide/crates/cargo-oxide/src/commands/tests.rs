/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;
use std::ffi::OsStr;

/// The concatenated sources of every module under `commands/`, standing in
/// for the old single-file `commands.rs` that the source-scanning tests
/// below used to read with `include_str!`.
const COMMANDS_SOURCE: &str = concat!(
    include_str!("mod.rs"),
    include_str!("artifacts.rs"),
    include_str!("build_run.rs"),
    include_str!("clean.rs"),
    include_str!("codegen_env.rs"),
    include_str!("context.rs"),
    include_str!("doctor.rs"),
    include_str!("examples_list.rs"),
    include_str!("fingerprint.rs"),
    include_str!("fmt.rs"),
    include_str!("host_cargo.rs"),
    include_str!("interop.rs"),
    include_str!("ltoir.rs"),
    include_str!("materialize.rs"),
    include_str!("passthrough.rs"),
    include_str!("pipeline_debug.rs"),
    include_str!("scaffold.rs"),
    include_str!("setup_update.rs"),
    include_str!("tests.rs"),
    include_str!("toolkit.rs"),
);

fn command_env(cmd: &Command, key: &str) -> Option<String> {
    cmd.get_envs()
        .find(|(name, _)| *name == OsStr::new(key))
        .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
}

fn decoded_rustflags(encoded: &str) -> Vec<&str> {
    encoded.split(ENCODED_RUSTFLAGS_SEPARATOR).collect()
}

fn has_backend_identity_cfg(flags: &[&str]) -> bool {
    flags.windows(2).any(|pair| {
        pair[0] == "--cfg"
            && pair[1].starts_with("cuda_oxide_internal_backend_identity=\"")
            && pair[1].ends_with('"')
    })
}

fn has_full_debug_cfg(flags: &[&str]) -> bool {
    flags
        .windows(2)
        .any(|pair| pair == ["--cfg", FULL_DEBUG_GET_MUT_OUTLINE_CFG])
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

/// `cargo_passthrough_command` with an empty ambient
/// `CUDA_OXIDE_MATERIALIZE_CUBIN`.
///
/// Every test builds the command through this. Reading the real variable
/// would let an exported value override `opts.materialize_cubin` and drive
/// the test into materializer discovery, which re-executes the libtest
/// binary and then exits the process -- aborting the whole suite instead of
/// failing one case.
fn passthrough_command_for_test(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) -> Result<Command, String> {
    cargo_passthrough_command_with_env(ctx, cargo_subcommand, opts, cargo_args, None)
}

fn cargo_artifact_freshness(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    materializer_provenance: Option<&str>,
) -> BTreeMap<String, bool> {
    let mut cmd = passthrough_command_for_test(
        ctx,
        CargoPassthroughSubcommand::Build,
        opts,
        &["--message-format=json-render-diagnostics".to_string()],
    )
    .unwrap();
    if let Some(provenance) = materializer_provenance {
        // Exercise a non-canonical spelling accepted by the backend. The
        // macro must still track exact provenance rather than keying that
        // dependency on the wrapper's canonical `1` spelling.
        cmd.env(MATERIALIZE_ENV, "true");
        cmd.env(EXPECTED_PROVENANCE_ENV, provenance);
    }
    let output = cmd.output().expect("failed to run Cargo cache probe");
    assert!(
        output.status.success(),
        "Cargo cache probe failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("Cargo JSON must be UTF-8")
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact")
        .filter_map(|message| {
            Some((
                message["target"]["name"].as_str()?.to_string(),
                message["fresh"].as_bool()?,
            ))
        })
        .collect()
}

fn test_context(config: OxideConfig) -> Context {
    Context {
        workspace_root: PathBuf::from("/tmp/cargo-oxide-test-workspace"),
        codegen_crate: PathBuf::from("/tmp/cargo-oxide-test-codegen"),
        examples_dir: PathBuf::from("/tmp/cargo-oxide-test-examples"),
        backend_so: PathBuf::from("llvm"),
        is_workspace: false,
        config,
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), unique))
}

#[cfg(unix)]
fn doctor_clang_fixture(root: &Path, name: &str, resource_response: &str) -> String {
    use std::os::unix::fs::PermissionsExt;

    let path = root.join(name);
    fs::write(
        &path,
        format!(
            "#!/bin/sh\ncase \"$1\" in\n--version) echo clang; exit 0;;\n-print-resource-dir) {resource_response};;\nesac\nexit 1\n"
        ),
    )
    .unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path.to_string_lossy().into_owned()
}

#[cfg(unix)]
#[test]
fn doctor_clang_resource_falls_back_after_absent_failed_and_empty_probes() {
    let root = unique_temp_dir("cargo_oxide_doctor_clang_fallback");
    fs::create_dir_all(&root).unwrap();
    let absent = root.join("clang").to_string_lossy().into_owned();
    let failed = doctor_clang_fixture(&root, "clang-22", "exit 1");
    let empty = doctor_clang_fixture(&root, "clang-21", "printf '   \n'; exit 0");
    let working = doctor_clang_fixture(
        &root,
        "clang-20",
        "printf ' /clang/resource dir \n'; exit 0",
    );
    assert_eq!(
        clang_resource_dir(&[&absent, &failed, &empty, &working]),
        Some((working.as_str(), "/clang/resource dir".to_string()))
    );
    assert_eq!(clang_resource_dir(&[&absent, &failed, &empty]), None);
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn doctor_clang_resource_prefers_the_first_successful_candidate() {
    let root = unique_temp_dir("cargo_oxide_doctor_clang_precedence");
    fs::create_dir_all(&root).unwrap();
    let bare = doctor_clang_fixture(&root, "clang", "printf '/bare/resource\n'; exit 0");
    let versioned =
        doctor_clang_fixture(&root, "clang-22", "printf '/versioned/resource\n'; exit 0");
    assert_eq!(
        clang_resource_dir(&[&bare, &versioned]),
        Some((bare.as_str(), "/bare/resource".to_string()))
    );
    fs::remove_dir_all(root).unwrap();
}

/// The examples walk backing `cargo oxide fmt` must reach nested manifests
/// and skip build directories.
///
/// The gate's glob is `examples/**/Cargo.toml`, so a nested manifest is a
/// scope of its own; the loop this replaced read only the first level, which
/// is how `cutile_inter_kernel/simt` and `interop_cubin_identity/device`
/// went unformatted. `target` is skipped because a working tree has one and
/// a packaged manifest under it is not a crate this repository formats.
#[test]
fn collect_example_manifests_reaches_nested_and_skips_target() {
    let root = unique_temp_dir("cargo_oxide_fmt_walk");
    let write = |rel: &str| {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "[package]\nname = \"x\"\n").unwrap();
    };

    write("plain/Cargo.toml");
    write("with_member/Cargo.toml");
    write("with_member/kernel-lib/Cargo.toml");
    write("own_workspace/Cargo.toml");
    write("own_workspace/simt/Cargo.toml");
    write("built/Cargo.toml");
    write("built/target/package/vendored/Cargo.toml");
    write(".hidden/Cargo.toml");
    std::fs::create_dir_all(root.join("no_manifest_here")).unwrap();

    let mut found = Vec::new();
    collect_example_manifests(&root, &mut found);
    let mut got: Vec<String> = found
        .iter()
        .map(|p| {
            p.strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/")
        })
        .collect();
    got.sort();

    assert_eq!(
        got,
        vec![
            // `built` itself is a real example; only the manifest inside
            // its `target/` is skipped.
            "built/Cargo.toml",
            "own_workspace/Cargo.toml",
            "own_workspace/simt/Cargo.toml",
            "plain/Cargo.toml",
            "with_member/Cargo.toml",
            "with_member/kernel-lib/Cargo.toml",
        ],
        "expected both nested manifests, and neither the one under target/ \
             nor the one in a dot directory"
    );

    let _ = std::fs::remove_dir_all(&root);
}

fn test_materializer_handshake() -> cuda_artifact_finalizer::MaterializerHandshakeV1 {
    let file = cuda_artifact_finalizer::ToolFileIdentity {
        length: 123,
        modified_seconds: 456,
        modified_nanoseconds: 789,
        device: Some(10),
        inode: Some(11),
        change_time_seconds: Some(12),
        change_time_nanoseconds: Some(13),
    };
    cuda_artifact_finalizer::MaterializerHandshakeV1::new(
        cuda_artifact_finalizer::PinnedToolProvenance {
            sha256: [1; 32],
            file,
        },
        cuda_artifact_finalizer::PinnedToolProvenance {
            sha256: [2; 32],
            file,
        },
        [3; 32],
    )
}

#[test]
fn strict_materialization_boolean_rejects_presence_only_values() {
    for value in ["1", "true", " YES ", "on"] {
        assert!(parse_strict_bool(MATERIALIZE_ENV, value).unwrap());
    }
    for value in ["0", "false", " NO ", "off"] {
        assert!(!parse_strict_bool(MATERIALIZE_ENV, value).unwrap());
    }
    for value in ["", "enabled", "2"] {
        let error = parse_strict_bool(MATERIALIZE_ENV, value).unwrap_err();
        assert!(error.contains("must be a boolean"), "{error}");
    }
}

#[test]
fn materialization_rejects_nvvm_ir_as_a_competing_final_output() {
    let error = prepare_materialization_result(
        &test_context(OxideConfig::default()),
        true,
        Some("sm_90"),
        true,
    )
    .expect_err("the two user-facing final output modes must conflict");

    assert!(error.contains("cannot be combined with --emit-nvvm-ir"));
}

#[test]
fn materializer_discovery_uses_the_same_project_tool_environment_as_rustc() {
    let configured_libdevice = "/configured/cuda/nvvm/libdevice/libdevice.10.bc";
    let ctx = test_context(OxideConfig {
        env: vec![
            (
                "CUDA_OXIDE_LIBDEVICE".to_string(),
                configured_libdevice.to_string(),
            ),
            (
                "CUDA_TOOLKIT_PATH".to_string(),
                "/configured/cuda".to_string(),
            ),
            (
                "LD_LIBRARY_PATH".to_string(),
                "/configured/cuda/lib64".to_string(),
            ),
            (
                MATERIALIZER_HANDSHAKE_ENV.to_string(),
                "ambient-handshake-must-not-be-used".to_string(),
            ),
        ],
        ..OxideConfig::default()
    });
    let discovery = materializer_discovery_command(&ctx, Path::new("/fake/cargo-oxide"));
    let mut rustc_child = Command::new("cargo");
    apply_common_codegen_env(
        &mut rustc_child,
        &ctx,
        false,
        false,
        false,
        DeviceDebug::Off,
    );

    for key in [
        "CUDA_OXIDE_LIBDEVICE",
        "CUDA_TOOLKIT_PATH",
        "LD_LIBRARY_PATH",
    ] {
        assert_eq!(
            command_env(&discovery, key),
            command_env(&rustc_child, key),
            "discovery and rustc must see the same {key}"
        );
    }
    if std::env::var_os("CUDA_OXIDE_LIBDEVICE").is_none() {
        assert_eq!(
            command_env(&discovery, "CUDA_OXIDE_LIBDEVICE").as_deref(),
            Some(configured_libdevice)
        );
    }
    assert_eq!(command_env(&discovery, MATERIALIZER_HANDSHAKE_ENV), None);
}

#[test]
fn materializer_handshake_cache_accepts_only_consistent_v1_records() {
    let root = unique_temp_dir("cargo_oxide_materializer_handshake");
    fs::create_dir(&root).unwrap();
    let mut ctx = test_context(OxideConfig::default());
    ctx.workspace_root = root.clone();
    let handshake = test_materializer_handshake();

    write_materializer_handshake_cache(&ctx, &handshake);
    let cached = read_materializer_handshake_cache(&ctx).unwrap();
    assert_eq!(
        serde_json::from_str::<cuda_artifact_finalizer::MaterializerHandshakeV1>(&cached).unwrap(),
        handshake,
    );

    let mut inconsistent = handshake;
    inconsistent.libnvvm.sha256[0] ^= 1;
    fs::write(
        materializer_handshake_cache_path(&ctx),
        serde_json::to_string(&inconsistent).unwrap(),
    )
    .unwrap();
    assert!(read_materializer_handshake_cache(&ctx).is_none());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_stem_normalizes_hyphens_like_cargo() {
    assert_eq!(artifact_stem("rustlantis-smoke"), "rustlantis_smoke");
    assert_eq!(artifact_stem("vecadd"), "vecadd");
}

#[test]
fn emit_ltoir_paths_use_normalized_crate_stem() {
    // Regression for the emit-ltoir read/write mismatch on hyphenated
    // crates: the backend writes `rustlantis_smoke.{ll,ltoir}`, so both the
    // NVVM IR read and the default LTOIR write must resolve to the
    // underscore stem rather than the raw example name.
    let dir = Path::new("/tmp/cargo-oxide-emit-ltoir");
    assert_eq!(
        emitted_ll_path(dir, "rustlantis-smoke"),
        dir.join("rustlantis_smoke.ll")
    );
    assert_eq!(
        default_ltoir_path(dir, "rustlantis-smoke"),
        dir.join("rustlantis_smoke.ltoir")
    );
    // A non-hyphenated example is unaffected.
    assert_eq!(emitted_ll_path(dir, "vecadd"), dir.join("vecadd.ll"));
    assert_eq!(default_ltoir_path(dir, "vecadd"), dir.join("vecadd.ltoir"));
}

#[test]
fn generated_file_cleanup_preserves_ltoir_cubin_cache() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cuda_oxide_clean_cache_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&root).unwrap();
    for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
        std::fs::write(root.join(format!("my_kernel.{extension}")), b"stale").unwrap();
    }
    let cached_cubin = root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
    std::fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
    std::fs::write(&cached_cubin, b"persistent cache entry").unwrap();

    clean_generated_files(&root, "my-kernel");

    for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
        assert!(!root.join(format!("my_kernel.{extension}")).exists());
    }
    assert_eq!(
        std::fs::read(&cached_cubin).unwrap(),
        b"persistent cache entry"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn clean_removes_only_local_target_and_matching_artifacts() {
    let root = unique_temp_dir("cargo_oxide_clean_standalone");
    std::fs::create_dir_all(root.join("target/debug")).unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "my-kernel"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    for suffix in GENERATED_ARTIFACT_SUFFIXES {
        std::fs::write(root.join(format!("my_kernel.{suffix}")), b"generated").unwrap();
    }

    let unrelated_artifact = root.join("other_kernel.ptx");
    std::fs::write(&unrelated_artifact, b"preserve").unwrap();

    let cached_cubin = root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
    std::fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
    std::fs::write(&cached_cubin, b"persistent cache").unwrap();

    let ctx = Context {
        workspace_root: root.clone(),
        codegen_crate: root.clone(),
        examples_dir: root.clone(),
        backend_so: root.join("unused-backend.so"),
        is_workspace: false,
        config: OxideConfig::default(),
    };

    let summary = clean_context(&ctx).unwrap();

    assert_eq!(summary.removed_directories, 1);
    assert_eq!(summary.removed_files, GENERATED_ARTIFACT_SUFFIXES.len());
    assert!(!root.join("target").exists());

    for suffix in GENERATED_ARTIFACT_SUFFIXES {
        assert!(!root.join(format!("my_kernel.{suffix}")).exists());
    }

    assert_eq!(std::fs::read(&unrelated_artifact).unwrap(), b"preserve");
    assert_eq!(std::fs::read(&cached_cubin).unwrap(), b"persistent cache");

    let second_summary = clean_context(&ctx).unwrap();

    assert_eq!(second_summary, CleanSummary::default());

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn workspace_clean_removes_root_backend_and_example_targets() {
    let root = unique_temp_dir("cargo_oxide_clean_workspace");
    let codegen_crate = root.join("crates/rustc-codegen-cuda");
    let examples_dir = codegen_crate.join("examples");
    let example_dir = examples_dir.join("demo");

    std::fs::create_dir_all(root.join("target/debug")).unwrap();
    std::fs::create_dir_all(codegen_crate.join("target/debug")).unwrap();
    std::fs::create_dir_all(example_dir.join("target/debug")).unwrap();

    std::fs::write(
        example_dir.join("Cargo.toml"),
        r#"
[package]
name = "demo"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    std::fs::write(example_dir.join("demo.ptx"), b"generated").unwrap();

    let ctx = Context {
        workspace_root: root.clone(),
        codegen_crate: codegen_crate.clone(),
        examples_dir,
        backend_so: root.join("unused-backend.so"),
        is_workspace: true,
        config: OxideConfig::default(),
    };

    let summary = clean_context(&ctx).unwrap();

    assert_eq!(summary.removed_directories, 3);
    assert_eq!(summary.removed_files, 1);
    assert!(!root.join("target").exists());
    assert!(!codegen_crate.join("target").exists());
    assert!(!example_dir.join("target").exists());
    assert!(!example_dir.join("demo.ptx").exists());

    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn clean_refuses_symlinked_target_directory() {
    use std::os::unix::fs::symlink;

    let root = unique_temp_dir("cargo_oxide_clean_symlink");
    let external = unique_temp_dir("cargo_oxide_clean_external");

    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&external).unwrap();
    std::fs::write(external.join("sentinel"), b"preserve").unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "symlink-test"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    symlink(&external, root.join("target")).unwrap();

    let ctx = Context {
        workspace_root: root.clone(),
        codegen_crate: root.clone(),
        examples_dir: root.clone(),
        backend_so: root.join("unused-backend.so"),
        is_workspace: false,
        config: OxideConfig::default(),
    };

    let error = clean_context(&ctx).unwrap_err();

    assert!(error.contains("symlinked target directory"), "{error}");
    assert_eq!(
        std::fs::read(external.join("sentinel")).unwrap(),
        b"preserve"
    );

    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(external).unwrap();
}

#[test]
fn cargo_metadata_selection_prefers_default_run() {
    let root = unique_temp_dir("cargo_oxide_default_run");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "multi-bin-package"
default-run = "main_bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "main_bin"
path = "src/main.rs"

[[bin]]
name = "other_bin"
path = "src/other.rs"
"#,
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

    let selection = cargo_executable_selection(&root, None).unwrap();
    assert_eq!(selection.packages.len(), 1);
    let package = &selection.packages[0];
    assert!(package.package_id.starts_with("path+file://"));
    assert!(package.package_id.contains("multi-bin-package@0.1.0"));
    assert_eq!(package.package_name, "multi-bin-package");
    assert_eq!(package.default_run.as_deref(), Some("main_bin"));
    assert_eq!(selection.explicit_bin, None);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_ignores_bins_disabled_by_required_features() {
    let root = unique_temp_dir("cargo_oxide_artifact_required_features");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "feature-gated-bins"
version = "0.1.0"
edition = "2024"

[features]
extra = []

[[bin]]
name = "always"
path = "src/always.rs"

[[bin]]
name = "gated"
path = "src/gated.rs"
required-features = ["extra"]
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/always.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("src/gated.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

    let expected_name = format!("always{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_selects_custom_bin_in_configured_target_dir() {
    let root = unique_temp_dir("cargo_oxide_artifact_binary");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(root.join(".cargo")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "package-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "actual-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        root.join(".cargo/config.toml"),
        "[build]\ntarget-dir = \"configured-target\"\n",
    )
    .unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

    assert!(binary.exists());
    let expected_name = format!("actual-bin{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    assert!(
        binary
            .components()
            .any(|part| part.as_os_str() == "configured-target")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_selects_single_binary_from_virtual_workspace() {
    let root = unique_temp_dir("cargo_oxide_artifact_workspace");
    let member = root.join("member");
    std::fs::create_dir_all(member.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"member\"]\nresolver = \"2\"\n",
    )
    .unwrap();
    std::fs::write(
        member.join("Cargo.toml"),
        r#"
[package]
name = "workspace-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "workspace-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(member.join("src/main.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

    let expected_name = format!("workspace-bin{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_honors_virtual_workspace_default_member_default_run() {
    let root = unique_temp_dir("cargo_oxide_artifact_default_member");
    let app = root.join("app");
    let ignored = root.join("ignored");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::create_dir_all(ignored.join("src")).unwrap();
    std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"app\", \"ignored\"]\ndefault-members = [\"app\"]\nresolver = \"2\"\n",
        )
        .unwrap();
    std::fs::write(
        app.join("Cargo.toml"),
        r#"
[package]
name = "selected-package"
default-run = "chosen-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "chosen-bin"
path = "src/chosen.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
    )
    .unwrap();
    std::fs::write(app.join("src/chosen.rs"), "fn main() {}\n").unwrap();
    std::fs::write(app.join("src/other.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        ignored.join("Cargo.toml"),
        r#"
[package]
name = "ignored-package"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();
    std::fs::write(ignored.join("src/main.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

    let expected_name = format!("chosen-bin{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_honors_nonvirtual_workspace_default_member() {
    let root = unique_temp_dir("cargo_oxide_artifact_nonvirtual_default_member");
    let member = root.join("member");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(member.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "workspace-root-package"
version = "0.1.0"
edition = "2024"

[workspace]
members = ["member"]
default-members = ["member"]
resolver = "2"

[[bin]]
name = "root-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        member.join("Cargo.toml"),
        r#"
[package]
name = "selected-member"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "member-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(member.join("src/main.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

    let expected_name = format!("member-bin{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_explicit_bin_selects_one_of_multiple_default_members() {
    let root = unique_temp_dir("cargo_oxide_artifact_multiple_default_members");
    let first = root.join("first");
    let second = root.join("second");
    std::fs::create_dir_all(first.join("src")).unwrap();
    std::fs::create_dir_all(second.join("src")).unwrap();
    std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"first\", \"second\"]\ndefault-members = [\"first\", \"second\"]\nresolver = \"2\"\n",
        )
        .unwrap();
    std::fs::write(
        first.join("Cargo.toml"),
        r#"
[package]
name = "first-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "first-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(first.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        second.join("Cargo.toml"),
        r#"
[package]
name = "second-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "chosen-bin"
path = "src/main.rs"
"#,
    )
    .unwrap();
    std::fs::write(second.join("src/main.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--bin", "chosen-bin"])
        .current_dir(&root);
    let binary = run_cargo_build_for_executable(&mut cmd, &root, Some("chosen-bin")).unwrap();

    let expected_name = format!("chosen-bin{}", std::env::consts::EXE_SUFFIX);
    assert_eq!(
        binary.file_name().and_then(OsStr::to_str),
        Some(expected_name.as_str())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_bin_must_be_unique_across_selected_packages() {
    let selection = CargoExecutableSelection {
        packages: vec![
            CargoSelectedPackage {
                package_id: "first-package 0.1.0".to_string(),
                package_name: "first-package".to_string(),
                default_run: None,
            },
            CargoSelectedPackage {
                package_id: "second-package 0.1.0".to_string(),
                package_name: "second-package".to_string(),
                default_run: None,
            },
        ],
        explicit_bin: Some("shared-bin".to_string()),
    };
    let artifacts = vec![
        CargoExecutableArtifact {
            package_id: "first-package 0.1.0".to_string(),
            target_name: "shared-bin".to_string(),
            path: PathBuf::from("/tmp/first/shared-bin"),
        },
        CargoExecutableArtifact {
            package_id: "second-package 0.1.0".to_string(),
            target_name: "shared-bin".to_string(),
            path: PathBuf::from("/tmp/second/shared-bin"),
        },
    ];

    let error = select_cargo_executable_artifact(&selection, &artifacts)
        .expect_err("the binary name does not uniquely identify an artifact");

    assert!(error.contains("multiple selected packages"), "{error}");
    assert!(error.contains("first-package"), "{error}");
    assert!(error.contains("second-package"), "{error}");
}

#[test]
fn one_executable_package_is_selected_alongside_library_only_defaults() {
    let selection = CargoExecutableSelection {
        packages: vec![
            CargoSelectedPackage {
                package_id: "library-package 0.1.0".to_string(),
                package_name: "library-package".to_string(),
                default_run: None,
            },
            CargoSelectedPackage {
                package_id: "application-package 0.1.0".to_string(),
                package_name: "application-package".to_string(),
                default_run: None,
            },
        ],
        explicit_bin: None,
    };
    let artifact = CargoExecutableArtifact {
        package_id: "application-package 0.1.0".to_string(),
        target_name: "application-bin".to_string(),
        path: PathBuf::from("/tmp/application/application-bin"),
    };

    assert_eq!(
        select_cargo_executable_artifact(&selection, &[artifact]).unwrap(),
        PathBuf::from("/tmp/application/application-bin")
    );
}

#[test]
fn unbuilt_default_run_is_not_skipped_for_another_selected_package() {
    let selection = CargoExecutableSelection {
        packages: vec![
            CargoSelectedPackage {
                package_id: "first-package 0.1.0".to_string(),
                package_name: "first-package".to_string(),
                default_run: Some("gated-bin".to_string()),
            },
            CargoSelectedPackage {
                package_id: "second-package 0.1.0".to_string(),
                package_name: "second-package".to_string(),
                default_run: None,
            },
        ],
        explicit_bin: None,
    };
    let artifacts = [CargoExecutableArtifact {
        package_id: "second-package 0.1.0".to_string(),
        target_name: "other-bin".to_string(),
        path: PathBuf::from("/tmp/second/other-bin"),
    }];

    let error = select_cargo_executable_artifact(&selection, &artifacts)
        .expect_err("a missing default-run must not fall back to another package");

    assert!(error.contains("first-package"), "{error}");
    assert!(error.contains("target `gated-bin`"), "{error}");
}

#[test]
fn cargo_json_errors_when_requested_bin_was_not_built() {
    let root = unique_temp_dir("cargo_oxide_artifact_missing_bin");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "package-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "actual-bin"
path = "src/actual.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/actual.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--bin", "actual-bin"])
        .current_dir(&root);
    let error = run_cargo_build_for_executable(&mut cmd, &root, Some("other-bin"))
        .expect_err("requested but unbuilt binary should be rejected");

    assert!(error.contains("target `other-bin`"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cargo_json_errors_when_default_run_was_not_built() {
    let root = unique_temp_dir("cargo_oxide_artifact_missing_default_run");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "package-bin"
default-run = "default-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "default-bin"
path = "src/default.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/default.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--bin", "other-bin"])
        .current_dir(&root);
    let error = run_cargo_build_for_executable(&mut cmd, &root, None)
        .expect_err("unbuilt default-run binary should be rejected");

    assert!(error.contains("target `default-bin`"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_selection_ignores_executable_artifacts_from_other_packages() {
    let selection = CargoExecutableSelection {
        packages: vec![CargoSelectedPackage {
            package_id: "app 0.1.0".to_string(),
            package_name: "app".to_string(),
            default_run: None,
        }],
        explicit_bin: Some("app-bin".to_string()),
    };
    let artifacts = vec![
        CargoExecutableArtifact {
            package_id: "build-tool 0.1.0".to_string(),
            target_name: "app-bin".to_string(),
            path: PathBuf::from("/tmp/build-tool/app-bin"),
        },
        CargoExecutableArtifact {
            package_id: "app 0.1.0".to_string(),
            target_name: "helper-bin".to_string(),
            path: PathBuf::from("/tmp/app/helper-bin"),
        },
    ];

    let error = select_cargo_executable_artifact(&selection, &artifacts)
        .expect_err("foreign package artifacts must not be selected");
    assert!(error.contains("target `app-bin`"), "{error}");
    assert!(error.contains("selected packages app"), "{error}");
}

#[test]
fn sanitizer_adds_nonzero_error_exitcode_by_default() {
    let invocation = sanitizer_invocation_args(&["--leak-check".to_string(), "full".to_string()]);

    assert_eq!(
        invocation.args,
        ["--error-exitcode", "86", "--leak-check", "full"]
    );
    assert!(invocation.uses_default_error_exitcode);
    assert!(!invocation.status_checks_weakened);
}

#[test]
fn sanitizer_preserves_explicit_zero_error_exitcode_without_claiming_detection() {
    let separated = sanitizer_invocation_args(&[
        "--error-exitcode".to_string(),
        "0".to_string(),
        "--leak-check".to_string(),
    ]);
    let equals = sanitizer_invocation_args(&["--error-exitcode=0".to_string()]);
    let repeated = sanitizer_invocation_args(&[
        "--error-exitcode=86".to_string(),
        "--error-exitcode=0".to_string(),
    ]);

    assert_eq!(separated.args, ["--error-exitcode", "0", "--leak-check"]);
    assert!(!separated.uses_default_error_exitcode);
    assert!(!separated.status_checks_weakened);
    assert_eq!(equals.args, ["--error-exitcode=0"]);
    assert!(!equals.uses_default_error_exitcode);
    assert_eq!(repeated.args, ["--error-exitcode=86", "--error-exitcode=0"]);
    assert!(!repeated.uses_default_error_exitcode);
}

#[test]
fn sanitizer_detects_options_that_weaken_success_status() {
    for args in [
        vec!["--check-exit-code=no".to_string()],
        vec!["--check-exit-code".to_string(), "no".to_string()],
        vec!["--require-cuda-init=no".to_string()],
        vec!["--require-cuda-init".to_string(), "NO".to_string()],
    ] {
        let invocation = sanitizer_invocation_args(&args);
        assert!(invocation.status_checks_weakened, "{args:?}");
    }
}

#[test]
fn interop_device_build_debug_assertions_select_the_release_like_assertion_profile() {
    assert_eq!(
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Build,
            false,
            false,
            DeviceDebug::Off,
            false
        )
        .codegen_profile(),
        CodegenProfilePolicy::ReleaseLike
    );
    assert_eq!(
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Build,
            false,
            false,
            DeviceDebug::Off,
            true
        )
        .codegen_profile(),
        CodegenProfilePolicy::ReleaseLikeWithDebugAssertions
    );
}

#[test]
fn interop_build_keeps_debug_assertions_and_device_debug_independent() {
    let ctx = test_context(OxideConfig::default());
    for debug in [DeviceDebug::Off, DeviceDebug::LineTables, DeviceDebug::Full] {
        for assertions in [false, true] {
            let options = InteropDeviceBuildOptions::for_route(
                InteropDeviceBuildRoute::Build,
                false,
                false,
                debug,
                assertions,
            );
            let mut cmd = Command::new("cargo");
            apply_interop_device_codegen_options_with_env(&mut cmd, &ctx, false, options, None);
            let mut encoded = build_encoded_rustflags_with_existing(
                Path::new("/tmp/librustc_codegen_cuda.so"),
                options.codegen_profile(),
                &[],
                &[],
                None,
                None,
            );
            append_full_debug_rustflags(&mut encoded, &cmd, None);
            let flags = decoded_rustflags(&encoded);

            assert!(flags.contains(&"-Copt-level=3"));
            assert!(flags.contains(&if assertions {
                "-Cdebug-assertions=on"
            } else {
                "-Cdebug-assertions=off"
            }));
            if assertions {
                assert!(flags.contains(&"-Coverflow-checks=off"));
            }
            assert_eq!(has_full_debug_cfg(&flags), debug == DeviceDebug::Full);
            for required in FULL_DEBUG_MIR_RUSTFLAGS {
                assert_eq!(flags.contains(required), debug == DeviceDebug::Full);
            }
        }
    }
}

#[test]
fn sanitize_interop_codegen_defaults_to_line_tables_and_forwards_no_fmad() {
    let ctx = test_context(OxideConfig::default());
    let mut cmd = Command::new("cargo");
    let options = InteropDeviceBuildOptions::for_route(
        InteropDeviceBuildRoute::Sanitize,
        true,
        false,
        DeviceDebug::Off,
        false,
    );

    apply_interop_device_codegen_options_with_env(&mut cmd, &ctx, false, options, None);

    assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
        Some("line-tables")
    );

    let fingerprint = interop_codegen_fingerprint_with_env(
        &ctx,
        false,
        options,
        Some("sm_80"),
        None,
        Path::new("/tmp/generated-ptx"),
        false,
        None,
        &MaterializationMode::default(),
        &BTreeMap::new(),
    );
    apply_codegen_configuration(
        &mut cmd,
        &ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    )
    .unwrap();
    let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
    assert!(has_backend_identity_cfg(&decoded_rustflags(&encoded)));
    assert_eq!(
        command_env(&cmd, CODEGEN_FINGERPRINT_ENV).as_deref(),
        Some(fingerprint.as_str())
    );
}

#[test]
fn sanitize_device_debug_flag_overrides_the_line_tables_default() {
    let ctx = test_context(OxideConfig::default());
    let mut cmd = Command::new("cargo");

    // Mirror codegen_build_host_binary's ordering: the flag's level lands
    // on `cmd` first, then the sanitizer default runs. `env_debug_set` is
    // injected as false, so with no ambient CUDA_OXIDE_DEBUG the explicit
    // flag alone must suppress the line-tables default.
    apply_common_codegen_env(&mut cmd, &ctx, false, false, false, DeviceDebug::Full);
    apply_default_sanitizer_line_tables_with_env(&mut cmd, &ctx, false, DeviceDebug::Full);

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
        Some("full")
    );
}

#[test]
fn standard_interop_codegen_forwards_no_fmad_without_debug_override() {
    let ctx = test_context(OxideConfig::default());
    let mut cmd = Command::new("cargo");

    apply_interop_device_codegen_options_with_env(
        &mut cmd,
        &ctx,
        false,
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Build,
            true,
            false,
            DeviceDebug::Off,
            false,
        ),
        None,
    );

    assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEBUG"), None);
}

#[test]
fn interop_build_run_and_sanitize_routes_forward_device_debug() {
    let ctx = test_context(OxideConfig::default());

    for (route, requested, expected, sanitizer_line_tables) in [
        (
            InteropDeviceBuildRoute::Build,
            DeviceDebug::Full,
            "full",
            false,
        ),
        (
            InteropDeviceBuildRoute::Run,
            DeviceDebug::LineTables,
            "line",
            false,
        ),
        (
            InteropDeviceBuildRoute::Sanitize,
            DeviceDebug::Full,
            "full",
            true,
        ),
    ] {
        let options = InteropDeviceBuildOptions::for_route(route, false, false, requested, false);
        assert_eq!(options.device_debug, requested);
        assert_eq!(options.sanitizer_line_tables, sanitizer_line_tables);

        let mut cmd = Command::new("cargo");
        apply_interop_device_codegen_options_with_env(&mut cmd, &ctx, false, options, None);
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
            Some(expected),
            "{route:?} must forward its CLI debug policy"
        );

        let mut encoded = "base".to_string();
        append_full_debug_rustflags(&mut encoded, &cmd, None);
        let flags = decoded_rustflags(&encoded);
        if requested == DeviceDebug::Full {
            assert!(
                has_full_debug_cfg(&flags),
                "{route:?} omitted full-debug cfg"
            );
            for required in FULL_DEBUG_MIR_RUSTFLAGS {
                assert!(flags.contains(required), "{route:?} omitted {required}");
            }
        } else {
            assert_eq!(flags, ["base"]);
        }
    }
}

#[test]
fn sanitize_interop_debug_precedence_is_cli_then_ambient_then_project_then_default() {
    struct Case {
        label: &'static str,
        cli: DeviceDebug,
        inherited: Option<&'static str>,
        project: Option<&'static str>,
        expected: &'static str,
    }

    let cases = [
        Case {
            label: "CLI full outranks ambient and project",
            cli: DeviceDebug::Full,
            inherited: Some("line"),
            project: Some("off"),
            expected: "full",
        },
        Case {
            label: "CLI line tables outrank ambient and project",
            cli: DeviceDebug::LineTables,
            inherited: Some("full"),
            project: Some("off"),
            expected: "line",
        },
        Case {
            label: "ambient opt-out outranks project and sanitizer default",
            cli: DeviceDebug::Off,
            inherited: Some("off"),
            project: Some("full"),
            expected: "off",
        },
        Case {
            label: "project setting outranks sanitizer default",
            cli: DeviceDebug::Off,
            inherited: None,
            project: Some("full"),
            expected: "full",
        },
        Case {
            label: "sanitizer supplies line tables only as the final default",
            cli: DeviceDebug::Off,
            inherited: None,
            project: None,
            expected: "line-tables",
        },
    ];

    for case in cases {
        let ctx = test_context(OxideConfig {
            env: case
                .project
                .map(|value| vec![("CUDA_OXIDE_DEBUG".to_string(), value.to_string())])
                .unwrap_or_default(),
            ..OxideConfig::default()
        });
        let options = InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Sanitize,
            false,
            false,
            case.cli,
            false,
        );
        let mut cmd = Command::new("cargo");
        apply_interop_device_codegen_options_with_env(
            &mut cmd,
            &ctx,
            false,
            options,
            case.inherited.map(OsStr::new),
        );

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
            Some(case.expected),
            "{}",
            case.label
        );
    }
}

#[test]
fn interop_fingerprint_tracks_effective_device_debug_policy() {
    let ctx = test_context(OxideConfig::default());
    let project_full = test_context(OxideConfig {
        env: vec![("CUDA_OXIDE_DEBUG".to_string(), "full".to_string())],
        ..OxideConfig::default()
    });
    let materialization = MaterializationMode::default();
    let empty_env = BTreeMap::new();
    let mut inherited_full = BTreeMap::new();
    inherited_full.insert("CUDA_OXIDE_DEBUG".to_string(), b"full".to_vec());
    let mut inherited_off = BTreeMap::new();
    inherited_off.insert("CUDA_OXIDE_DEBUG".to_string(), b"off".to_vec());
    let options =
        |route, debug| InteropDeviceBuildOptions::for_route(route, false, false, debug, false);
    let fingerprint = |ctx: &Context, options, inherited_env: &BTreeMap<String, Vec<u8>>| {
        interop_codegen_fingerprint_with_env(
            ctx,
            false,
            options,
            Some("sm_80"),
            None,
            Path::new("/tmp/cuda-oxide-artifacts"),
            false,
            None,
            &materialization,
            inherited_env,
        )
    };

    let off = options(InteropDeviceBuildRoute::Build, DeviceDebug::Off);
    let line = options(InteropDeviceBuildRoute::Build, DeviceDebug::LineTables);
    let full = options(InteropDeviceBuildRoute::Build, DeviceDebug::Full);
    assert_ne!(
        fingerprint(&ctx, off, &empty_env),
        fingerprint(&ctx, line, &empty_env)
    );
    assert_ne!(
        fingerprint(&ctx, off, &empty_env),
        fingerprint(&ctx, full, &empty_env)
    );
    assert_ne!(
        fingerprint(&ctx, line, &empty_env),
        fingerprint(&ctx, full, &empty_env)
    );

    // A CLI policy replaces inherited and project values in both the command
    // and its cache identity, rather than producing irrelevant identities.
    assert_eq!(
        fingerprint(&ctx, line, &empty_env),
        fingerprint(&ctx, line, &inherited_full)
    );
    assert_eq!(
        fingerprint(&ctx, line, &empty_env),
        fingerprint(&project_full, line, &empty_env)
    );
    // Without a CLI level, project configuration participates in the key and
    // an inherited value replaces it, mirroring command-environment order.
    assert_eq!(
        fingerprint(&project_full, off, &empty_env),
        fingerprint(&ctx, full, &empty_env)
    );
    assert_eq!(
        fingerprint(&project_full, off, &inherited_off),
        fingerprint(&ctx, off, &inherited_off)
    );

    // Sanitizer's implicit line-table request is output-affecting even though
    // it is not represented by a CLI DeviceDebug variant.
    let sanitize_default = options(InteropDeviceBuildRoute::Sanitize, DeviceDebug::Off);
    assert_ne!(
        fingerprint(&ctx, off, &empty_env),
        fingerprint(&ctx, sanitize_default, &empty_env)
    );
    assert_eq!(
        fingerprint(&ctx, line, &empty_env),
        fingerprint(&ctx, sanitize_default, &empty_env),
        "implicit sanitizer line tables and explicit line tables have identical device semantics"
    );
    assert_eq!(
        fingerprint(&ctx, full, &empty_env),
        fingerprint(
            &ctx,
            options(InteropDeviceBuildRoute::Sanitize, DeviceDebug::Full),
            &empty_env,
        ),
        "the sanitize route must not fork the cache when CLI full already fixes the policy"
    );
    assert_eq!(
        fingerprint(&ctx, off, &inherited_full),
        fingerprint(&ctx, sanitize_default, &inherited_full),
        "ambient full suppresses the sanitizer default in both command and cache identity"
    );
}

#[test]
fn interop_fingerprint_tracks_artifact_mode_and_device_features() {
    let ctx = test_context(OxideConfig::default());
    let options = InteropDeviceBuildOptions::for_route(
        InteropDeviceBuildRoute::Build,
        false,
        false,
        DeviceDebug::Off,
        false,
    );
    let fingerprint = |emit_nvvm_ir: bool, device_features: Option<&str>| {
        interop_codegen_fingerprint(
            &ctx,
            false,
            options,
            Some("sm_120a"),
            None,
            Path::new("/tmp/cuda-oxide-artifacts"),
            emit_nvvm_ir,
            device_features,
            &MaterializationMode::default(),
        )
    };

    assert_ne!(fingerprint(false, None), fingerprint(true, None));
    assert_ne!(
        fingerprint(true, None),
        fingerprint(true, Some("tensor-cores"))
    );
}

#[test]
fn interop_codegen_forwards_unchecked_indexing() {
    let ctx = test_context(OxideConfig::default());
    let mut cmd = Command::new("cargo");

    apply_interop_device_codegen_options_with_env(
        &mut cmd,
        &ctx,
        false,
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Run,
            false,
            true,
            DeviceDebug::Off,
            false,
        ),
        None,
    );

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_UNCHECKED_INDEXING").as_deref(),
        Some("1")
    );
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA"), None);
}

#[test]
fn sanitize_fingerprint_tracks_output_affecting_settings() {
    let ctx = test_context(OxideConfig::default());
    // Empty inherited environment, matching
    // `passthrough_fingerprint_tracks_output_affecting_settings`. An
    // ambient CUDA_OXIDE_NO_FMA / CUDA_OXIDE_UNCHECKED_INDEXING is folded
    // into the digest on its own, so reading the real environment would
    // make toggling the corresponding argument a no-op and collapse these
    // fingerprints onto the base.
    let inherited_env = BTreeMap::new();
    let fingerprint = |no_fmad: bool,
                       unchecked_indexing: bool,
                       target_arch: Option<&str>,
                       detected_device_arch: Option<&str>,
                       ptx_dir: Option<&Path>| {
        sanitize_codegen_fingerprint_with_env(
            &ctx,
            false,
            no_fmad,
            unchecked_indexing,
            DeviceDebug::Off,
            target_arch,
            detected_device_arch,
            ptx_dir,
            &MaterializationMode::default(),
            &inherited_env,
        )
    };

    let base = fingerprint(false, false, None, Some("sm_80"), None);

    for changed in [
        fingerprint(true, false, None, Some("sm_80"), None),
        fingerprint(false, true, None, Some("sm_80"), None),
        fingerprint(false, false, None, Some("sm_90"), None),
        fingerprint(false, false, Some("sm_80"), None, None),
        fingerprint(
            false,
            false,
            None,
            Some("sm_80"),
            Some(Path::new("/tmp/generated-ptx")),
        ),
    ] {
        assert_ne!(base, changed);
    }
}

#[test]
fn pipeline_diagnostics_have_a_distinct_device_fingerprint() {
    let ctx = test_context(OxideConfig::default());
    let materialization = MaterializationMode::default();
    let standard = standard_codegen_fingerprint(
        &ctx,
        true,
        false,
        false,
        DeviceDebug::Off,
        false,
        Some("sm_86"),
        None,
        &materialization,
    );
    let pipeline = pipeline_codegen_fingerprint(
        &ctx,
        false,
        false,
        DeviceDebug::Off,
        false,
        Some("sm_86"),
        &materialization,
    );

    assert_ne!(standard, pipeline);
}

/// A `Context` whose `cuda-oxide.toml` points `CUDA_TOOLKIT_PATH` at
/// `root`, alongside a fake executable named `name` under `root/bin`.
fn toolkit_context_with_tool(root: &Path, name: &str) -> (Context, PathBuf) {
    let tool = root.join("bin").join(name);
    std::fs::create_dir_all(tool.parent().unwrap()).unwrap();
    std::fs::write(&tool, b"fake tool").unwrap();
    let ctx = test_context(OxideConfig {
        env: vec![(
            "CUDA_TOOLKIT_PATH".to_string(),
            root.to_string_lossy().into_owned(),
        )],
        ..OxideConfig::default()
    });
    (ctx, tool)
}

/// Every table entry pairs a tool with its own fallback list. Pure, so it
/// covers all three tools on every host: pairing `nvcc` with
/// `CUDA_GDB_FALLBACK_PATHS` is a one-token slip that no filesystem test
/// using a fabricated name could see.
#[test]
fn each_toolkit_tool_is_paired_with_its_own_fallback_paths() {
    assert_eq!(DOCTOR_TOOLKIT_TOOLS.len(), 3);
    for (name, fallbacks) in DOCTOR_TOOLKIT_TOOLS {
        assert!(!fallbacks.is_empty(), "{name} has no fallback paths");
        for path in fallbacks {
            assert_eq!(
                Path::new(path).file_name().and_then(|n| n.to_str()),
                Some(name),
                "{name} lists a fallback for a different tool: {path}"
            );
        }
    }
}

/// The lookup itself: each tool gets its own list, and an unknown name gets
/// none rather than another tool's. Pure, so it holds on every host.
#[test]
fn the_fallback_lookup_returns_each_tools_own_list() {
    assert_eq!(toolkit_tool_fallbacks("nvcc"), NVCC_FALLBACK_PATHS);
    assert_eq!(toolkit_tool_fallbacks("cuda-gdb"), CUDA_GDB_FALLBACK_PATHS);
    assert_eq!(
        toolkit_tool_fallbacks("compute-sanitizer"),
        COMPUTE_SANITIZER_FALLBACK_PATHS
    );
    assert!(toolkit_tool_fallbacks("ptxas").is_empty());
}

/// Every call site names a tool the table knows. `doctor_toolkit_tool`
/// falls back to an empty list for an unknown name, so a typo would
/// silently drop the standard install roots rather than fail.
#[test]
fn every_doctor_toolkit_tool_call_names_a_tool_in_the_table() {
    let source = COMMANDS_SOURCE;
    let mut calls = 0;
    for (index, _) in source.match_indices("doctor_toolkit_tool(ctx, \"") {
        let rest = &source[index + "doctor_toolkit_tool(ctx, \"".len()..];
        let name = &rest[..rest.find('"').expect("a closing quote")];
        assert!(
            DOCTOR_TOOLKIT_TOOLS.iter().any(|(tool, _)| *tool == name),
            "{name} is probed but absent from DOCTOR_TOOLKIT_TOOLS, so it \
                 would silently lose its fallback paths"
        );
        calls += 1;
    }
    assert_eq!(
        calls,
        DOCTOR_TOOLKIT_TOOLS.len(),
        "every table entry is probed once"
    );
}

/// Unaffected control: the configured toolkit root is preferred over the
/// standard install roots, so pointing `CUDA_TOOLKIT_PATH` at a toolkit does
/// not start picking up `/usr/local/cuda`. Discovery order is unchanged by
/// this fix; only which function `doctor` calls changed.
///
/// PATH precedence is deliberately *not* asserted here. `find_executable`
/// shells out to `which`, so testing it would mean mutating the process
/// `PATH` -- `unsafe` in edition 2024 and racy across the test threads. The
/// end-to-end evidence in the pull request covers that case against the real
/// binary instead, which is the stronger check anyway.
#[test]
fn the_configured_root_is_preferred_over_the_standard_install_roots() {
    let root = unique_temp_dir("cargo_oxide_doctor_order");
    let name = "cuda-oxide-test-doctor-order";
    let under_toolkit = root.join("bin").join(name);
    std::fs::create_dir_all(under_toolkit.parent().unwrap()).unwrap();
    std::fs::write(&under_toolkit, b"fake tool").unwrap();
    let fallback = root.join("fallback").join(name);
    std::fs::create_dir_all(fallback.parent().unwrap()).unwrap();
    std::fs::write(&fallback, b"fake tool").unwrap();
    let ctx = test_context(OxideConfig {
        env: vec![(
            "CUDA_TOOLKIT_PATH".to_string(),
            root.to_string_lossy().into_owned(),
        )],
        ..OxideConfig::default()
    });

    let fallback_arg = fallback.to_string_lossy().into_owned();
    assert_eq!(
        find_cuda_toolkit_executable_with_env(&ctx, name, &[&fallback_arg], |_| None),
        Some(under_toolkit),
        "the configured toolkit root must win over a fallback path"
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// Negative: absent from PATH, from the configured root, and from the
/// fallbacks is still `None` -- the fix must not invent a path.
#[test]
fn a_missing_tool_is_still_missing() {
    let root = unique_temp_dir("cargo_oxide_doctor_absent");
    std::fs::create_dir_all(&root).unwrap();
    let ctx = test_context(OxideConfig {
        env: vec![(
            "CUDA_TOOLKIT_PATH".to_string(),
            root.to_string_lossy().into_owned(),
        )],
        ..OxideConfig::default()
    });
    assert_eq!(
        find_cuda_toolkit_executable_with_env(
            &ctx,
            "cuda-oxide-test-absent-tool",
            &["/nonexistent/bin/cuda-oxide-test-absent-tool"],
            |_| None,
        ),
        None
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// The invariant a table alone cannot hold: no CUDA toolkit executable may
/// be probed with a bare `Command::new`, which is how `nvcc` and `cuda-gdb`
/// came to disagree with the commands they predict. Read out of the source
/// rather than from `DOCTOR_TOOLKIT_TOOLS`, so a tool left out of the table
/// is still caught.
#[test]
fn no_toolkit_executable_is_probed_by_bare_path() {
    let source = COMMANDS_SOURCE;
    for tool in [
        "nvcc",
        "cuda-gdb",
        "compute-sanitizer",
        "ptxas",
        "nvlink",
        "fatbinary",
        "nvdisasm",
        "cuobjdump",
    ] {
        let bare = format!("Command::new(\"{tool}\")");
        assert!(
            !source.contains(&bare),
            "{tool} is probed with {bare}, bypassing the toolkit root; \
                 route it through doctor_toolkit_tool / find_cuda_toolkit_executable"
        );
    }
}

/// `doctor` predicts whether `debug` will work, so both must consult one
/// list. A second inline copy is how they drifted apart.
#[test]
fn doctor_and_debug_share_one_cuda_gdb_fallback_list() {
    let source = COMMANDS_SOURCE;
    assert!(
        source.matches("CUDA_GDB_FALLBACK_PATHS").count() >= 3,
        "the shared list is not referenced by both call sites"
    );
    assert_eq!(
        source.matches("\"/usr/local/cuda/bin/cuda-gdb\"").count(),
        1,
        "a second inline cuda-gdb path list has appeared"
    );
}

#[test]
fn sanitizer_tool_lookup_uses_project_cuda_toolkit_root() {
    let root = unique_temp_dir("cargo_oxide_sanitizer_tool");
    let (ctx, tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-sanitizer");

    // `|_| None` stands in for an empty ambient environment. Reading the
    // real one would let an exported CUDA_TOOLKIT_PATH/CUDA_HOME shadow the
    // configured root this test asserts on.
    assert_eq!(
        find_cuda_toolkit_executable_with_env(&ctx, "cuda-oxide-test-sanitizer", &[], |_| None),
        Some(tool)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn doctor_compute_sanitizer_lookup_matches_sanitize_discovery() {
    // Hermetic: a fake tool name keeps the user's real PATH (and any
    // installed compute-sanitizer) out of the lookup, the injected empty
    // environment keeps an exported toolkit root out of it, and the shared
    // fallback const exercises the exact argument both `doctor` and
    // `sanitize` pass. The configured toolkit root wins before any
    // fallback path is consulted.
    let root = unique_temp_dir("cargo_oxide_doctor_sanitizer");
    let (ctx, tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-doctor-sanitizer");

    assert_eq!(
        find_cuda_toolkit_executable_with_env(
            &ctx,
            "cuda-oxide-test-doctor-sanitizer",
            COMPUTE_SANITIZER_FALLBACK_PATHS,
            |_| None,
        ),
        Some(tool)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn ambient_cuda_toolkit_path_shadows_the_project_configured_root() {
    // The precedence the two lookups above have to be insulated from: an
    // exported CUDA_TOOLKIT_PATH outranks `cuda-oxide.toml`, so a tool
    // present only under the configured root is not found.
    let root = unique_temp_dir("cargo_oxide_ambient_shadow");
    let (ctx, _tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-shadowed-sanitizer");
    let ambient = unique_temp_dir("cargo_oxide_ambient_root");

    assert_eq!(
        find_cuda_toolkit_executable_with_env(
            &ctx,
            "cuda-oxide-test-shadowed-sanitizer",
            &[],
            |key| (key == "CUDA_TOOLKIT_PATH").then(|| ambient.to_string_lossy().into_owned()),
        ),
        None
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_passthrough_defers_profile_flags_to_cargo_and_keeps_invariants() {
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/librustc_codegen_cuda.so"),
        CargoPassthroughSubcommand::Test.codegen_profile(false),
        &[],
        &["--cfg".to_string(), "device_test".to_string()],
        None,
        None,
    );
    let flags = decoded_rustflags(&rustflags);

    assert_eq!(
        flags,
        [
            "--cfg",
            "device_test",
            "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
            "-Zmir-enable-passes=-JumpThreading",
            "-Zalways-encode-mir",
            "-Csymbol-mangling-version=v0",
        ]
    );
    assert!(!flags.iter().any(|flag| flag.starts_with("-Copt-level")));
    assert!(
        !flags
            .iter()
            .any(|flag| flag.starts_with("-Cdebug-assertions"))
    );
    assert!(!flags.iter().any(|flag| flag.starts_with("-Cdebuginfo")));

    let ctx = test_context(OxideConfig::default());
    let opts = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: None,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    for cargo_args in [
        vec!["--release".to_string()],
        vec!["--profile".to_string(), "ci".to_string()],
    ] {
        let cmd = passthrough_command_for_test(
            &ctx,
            CargoPassthroughSubcommand::Test,
            &opts,
            &cargo_args,
        )
        .unwrap();
        let mut expected = vec!["test".to_string()];
        expected.extend(cargo_args);
        assert_eq!(
            cmd.get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn build_passthrough_retains_release_profile_and_required_flags() {
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/librustc_codegen_cuda.so"),
        CargoPassthroughSubcommand::Build.codegen_profile(false),
        &[],
        &[],
        Some("-Lnative=/nix/store/cuda-cudart/lib\u{1f}-Copt-level=0\u{1f}-Zcodegen-backend=llvm"),
        Some("-L native=/nix/store/cuda-cudart/lib"),
    );
    let flags = decoded_rustflags(&rustflags);

    assert_eq!(flags[0], "-Lnative=/nix/store/cuda-cudart/lib");
    assert!(flags.contains(&"-Copt-level=0"));
    assert!(flags.contains(&"-Zcodegen-backend=llvm"));
    assert_eq!(
        &flags[flags.len() - 6..],
        [
            "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
            "-Copt-level=3",
            "-Cdebug-assertions=off",
            "-Zmir-enable-passes=-JumpThreading",
            "-Zalways-encode-mir",
            "-Csymbol-mangling-version=v0",
        ]
    );
    assert!(!flags.contains(&"native=/nix/store/cuda-cudart/lib"));
}

#[test]
fn build_debug_assertions_keep_release_optimization_without_overflow_checks() {
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/librustc_codegen_cuda.so"),
        CargoPassthroughSubcommand::Build.codegen_profile(true),
        &[],
        &[],
        Some("-Cdebug-assertions=off\u{1f}-Coverflow-checks=on\u{1f}-Copt-level=0"),
        None,
    );
    let flags = decoded_rustflags(&rustflags);

    assert_eq!(
        &flags[flags.len() - 7..],
        [
            "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
            "-Copt-level=3",
            "-Cdebug-assertions=on",
            "-Coverflow-checks=off",
            "-Zmir-enable-passes=-JumpThreading",
            "-Zalways-encode-mir",
            "-Csymbol-mangling-version=v0",
        ]
    );
}

#[test]
fn test_profile_ignores_the_build_only_debug_assertions_toggle() {
    assert_eq!(
        CargoPassthroughSubcommand::Test.codegen_profile(true),
        CodegenProfilePolicy::CargoSelected
    );
}

#[test]
fn encoded_rustflags_preserve_configured_flag_boundaries_and_spaces() {
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/backend path/librustc_codegen_cuda.so"),
        CodegenProfilePolicy::ReleaseLike,
        &["--cfg".to_string(), "model=\"alpha beta\"".to_string()],
        &[],
        None,
        Some("-L native=/nix/store/cuda-cudart/lib"),
    );
    let flags = decoded_rustflags(&rustflags);

    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
    );
    assert_eq!(&flags[2..4], ["-L", "native=/nix/store/cuda-cudart/lib"]);
    assert_eq!(
        flags[flags.len() - 6],
        "-Zcodegen-backend=/tmp/backend path/librustc_codegen_cuda.so"
    );
}

#[test]
fn encoded_rustflags_remove_wrapper_owned_codegen_cfgs() {
    let encoded = [
        "--cfg",
        "cuda_oxide_internal_codegen_env=\"inherited\"",
        "--cfg=cuda_oxide_internal_materializer_provenance=\"inherited\"",
        "--cfg",
        FULL_DEBUG_GET_MUT_OUTLINE_CFG,
        "--cfg",
        "keep_inherited",
    ]
    .join(&ENCODED_RUSTFLAGS_SEPARATOR.to_string());
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/librustc_codegen_cuda.so"),
        CodegenProfilePolicy::ReleaseLike,
        &[
            "--cfg".to_string(),
            "cuda_oxide_internal_codegen_env=\"configured\"".to_string(),
            "--cfg=cuda_oxide_internal_outline_disjoint_get_mut_v1".to_string(),
            "--cfg".to_string(),
            "cuda_oxide_internal_outline_disjoint_get_mut_v1_extra".to_string(),
            "--cfg".to_string(),
            "keep_configured".to_string(),
        ],
        &[
            "--cfg".to_string(),
            "cuda_oxide_internal_materializer_provenance=\"explicit\"".to_string(),
            "--cfg".to_string(),
            "cuda_oxide_internal_outline_disjoint_get_mut_v1=\"spoofed\"".to_string(),
            "--cfg".to_string(),
            "keep_explicit".to_string(),
        ],
        Some(&encoded),
        None,
    );
    let flags = decoded_rustflags(&rustflags);

    let is_wrapper_owned_cfg = |flag: &&str| {
        let value = flag.strip_prefix("--cfg=").unwrap_or(flag);
        [
            LEGACY_CODEGEN_FINGERPRINT_CFG,
            LEGACY_MATERIALIZER_PROVENANCE_CFG,
            FULL_DEBUG_GET_MUT_OUTLINE_CFG,
        ]
        .iter()
        .any(|name| {
            value
                .strip_prefix(name)
                .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with('='))
        })
    };
    assert!(!flags.iter().any(is_wrapper_owned_cfg));
    for retained in [
        "cuda_oxide_internal_outline_disjoint_get_mut_v1_extra",
        "keep_configured",
        "keep_inherited",
        "keep_explicit",
    ] {
        assert!(flags.contains(&retained));
    }
}

#[test]
fn debug_profile_retains_release_defaults_and_adds_debuginfo() {
    let rustflags = build_encoded_rustflags_with_existing(
        Path::new("/tmp/librustc_codegen_cuda.so"),
        CodegenProfilePolicy::ReleaseLikeWithDebugInfo,
        &[],
        &[],
        None,
        Some(""),
    );
    let flags = decoded_rustflags(&rustflags);

    assert!(flags.contains(&"-Copt-level=3"));
    assert!(flags.contains(&"-Cdebug-assertions=off"));
    assert!(flags.contains(&"-Cdebuginfo=2"));
    assert!(flags.contains(&"-Zmir-enable-passes=-JumpThreading"));
    assert!(flags.contains(&"-Zalways-encode-mir"));
    assert!(flags.contains(&"-Csymbol-mangling-version=v0"));
    assert!(!flags.contains(&""));
}

#[test]
fn project_config_parser_loads_backend_arch_flags_and_env() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_config_test_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    std::fs::write(
        cargo_dir.join("cuda-oxide.toml"),
        r#"
backend = "../backend/librustc_codegen_cuda.so"
default-arch = "sm_90"
extra-rustflags = ["--cfg", "model=\"alpha beta\""]

[env]
MY_BUILD_FLAG = "configured"
"#,
    )
    .unwrap();

    let config = load_oxide_config(&root);
    assert_eq!(
        config.backend,
        Some(cargo_dir.join("../backend/librustc_codegen_cuda.so"))
    );
    assert_eq!(config.default_arch.as_deref(), Some("sm_90"));
    assert_eq!(config.extra_rustflags, ["--cfg", "model=\"alpha beta\""]);
    assert_eq!(
        config.env,
        vec![("MY_BUILD_FLAG".to_string(), "configured".to_string())]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn inspect_oxide_config_missing_is_informational() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_config_missing_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&root).unwrap();
    assert!(matches!(
        inspect_oxide_config(&root),
        OxideConfigInspection::Missing
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn inspect_oxide_config_rejects_bad_toml_and_arch() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_config_bad_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    std::fs::write(cargo_dir.join("cuda-oxide.toml"), "default-arch = [\n").unwrap();
    match inspect_oxide_config(&root) {
        OxideConfigInspection::Invalid { errors, .. } => {
            assert!(errors.iter().any(|e| e.contains("could not parse")));
        }
        other => panic!("expected Invalid, got {other:?}"),
    }

    std::fs::write(
        cargo_dir.join("cuda-oxide.toml"),
        "default-arch = \"sm_9x\"\n",
    )
    .unwrap();
    match inspect_oxide_config(&root) {
        OxideConfigInspection::Invalid { errors, .. } => {
            assert!(errors.iter().any(|e| e.contains("default-arch")));
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// `default-arch` load-time validation must be exactly as permissive as
/// the consumers: `parse_nvvm_arch` (NVVM path) accepts `sm_XX`,
/// `compute_XX`, and bare `XX`, so none of those may fail the load.
/// Non-`sm_XX` spellings only earn an advisory warning.
#[test]
fn default_arch_validation_matches_the_real_arch_parser() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_config_arch_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    let config_path = cargo_dir.join("cuda-oxide.toml");

    for accepted in ["sm_80", "sm_90a", "sm_100f", "sm_120"] {
        std::fs::write(&config_path, format!("default-arch = \"{accepted}\"\n")).unwrap();
        match inspect_oxide_config(&root) {
            OxideConfigInspection::Valid { warnings, .. } => {
                assert!(warnings.is_empty(), "unexpected warnings for {accepted}");
            }
            other => panic!("expected {accepted} to be Valid, got {other:?}"),
        }
    }

    // Spellings that genuinely work today (the NVVM path normalizes
    // them) load fine but get the preferred-spelling advice.
    for (works_with_warning, preferred) in [("compute_90", "sm_90"), ("90", "sm_90")] {
        std::fs::write(
            &config_path,
            format!("default-arch = \"{works_with_warning}\"\n"),
        )
        .unwrap();
        match inspect_oxide_config(&root) {
            OxideConfigInspection::Valid { config, warnings } => {
                assert_eq!(config.default_arch.as_deref(), Some(works_with_warning));
                assert!(
                    warnings.iter().any(|w| w.contains(preferred)),
                    "expected a `{preferred}` spelling advisory for \
                         {works_with_warning}, got {warnings:?}"
                );
            }
            other => panic!("expected {works_with_warning} to be Valid, got {other:?}"),
        }
    }

    for rejected in ["sm_9", "sm_90x", "hopper"] {
        std::fs::write(&config_path, format!("default-arch = \"{rejected}\"\n")).unwrap();
        match inspect_oxide_config(&root) {
            OxideConfigInspection::Invalid { errors, .. } => {
                assert!(
                    errors.iter().any(|e| e.contains("default-arch")),
                    "expected a default-arch error for {rejected}, got {errors:?}"
                );
            }
            other => panic!("expected {rejected} to be Invalid, got {other:?}"),
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn inspect_oxide_config_warns_on_forbidden_env_keys() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_config_warn_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    std::fs::write(
        cargo_dir.join("cuda-oxide.toml"),
        r#"
default-arch = "sm_90a"

[env]
RUSTFLAGS = "-C opt-level=3"
CARGO_ENCODED_RUSTFLAGS = "legacy"
MY_OK = "1"
"#,
    )
    .unwrap();

    match inspect_oxide_config(&root) {
        OxideConfigInspection::Valid { config, warnings } => {
            assert_eq!(config.default_arch.as_deref(), Some("sm_90a"));
            assert!(
                warnings
                    .iter()
                    .any(|w| w.contains("RUSTFLAGS") && w.contains("ignored"))
            );
            assert!(
                warnings
                    .iter()
                    .any(|w| w.contains("CARGO_ENCODED_RUSTFLAGS") && w.contains("ignored"))
            );
        }
        other => panic!("expected Valid with warnings, got {other:?}"),
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn doctor_survives_malformed_config_and_reports_the_failed_check() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_doctor_config_bad_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    std::fs::write(cargo_dir.join("cuda-oxide.toml"), "default-arch = [\n").unwrap();

    // Passive context resolution must not exit: it degrades to defaults
    // so the doctor scan can start at all.
    assert_eq!(load_oxide_config_lenient(&root), OxideConfig::default());

    // Doctor's own check re-inspects the file and fails.
    let check = check_oxide_config(&root);
    assert!(check.failed);
    assert!(check.headline.starts_with('✗'), "{}", check.headline);
    assert!(
        check
            .details
            .iter()
            .any(|line| line.contains("could not parse")),
        "{:?}",
        check.details
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn doctor_reports_env_rustflags_warning_without_failing_the_check() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_doctor_config_warn_{}_{}",
        std::process::id(),
        unique
    ));
    let cargo_dir = root.join(".cargo");
    std::fs::create_dir_all(&cargo_dir).unwrap();
    std::fs::write(
        cargo_dir.join("cuda-oxide.toml"),
        "default-arch = \"sm_90a\"\n\n[env]\nRUSTFLAGS = \"-C opt-level=3\"\n",
    )
    .unwrap();

    let check = check_oxide_config(&root);
    assert!(!check.failed);
    assert!(check.headline.contains("default-arch = sm_90a"));
    assert!(
        check
            .details
            .iter()
            .any(|line| line.contains("RUSTFLAGS") && line.contains("ignored")),
        "{:?}",
        check.details
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn doctor_reports_missing_config_as_informational() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_doctor_config_missing_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&root).unwrap();

    let check = check_oxide_config(&root);
    assert!(!check.failed);
    assert_eq!(check.headline, "- not present (using defaults)");
    assert!(check.details.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn passthrough_command_preserves_argv_and_cli_overrides_config_defaults() {
    let config = OxideConfig {
        extra_rustflags: vec!["--cfg".to_string(), "from_config".to_string()],
        env: vec![
            ("CARGO_TARGET_DIR".to_string(), "config-target".to_string()),
            (
                "CUDA_OXIDE_DEVICE_CODEGEN_CRATE".to_string(),
                "config_owner".to_string(),
            ),
            ("CUDA_OXIDE_VERBOSE".to_string(), "configured".to_string()),
        ],
        ..OxideConfig::default()
    };
    let ctx = test_context(config);
    let device_cfgs = vec!["model=\"alpha beta\"".to_string()];
    let opts = CargoPassthroughOptions {
        verbose: true,
        emit_nvvm_ir: false,
        arch: Some("sm_90"),
        features: Some("wrapper_feature"),
        cargo_target_dir: Some(Path::new("cli-target")),
        device_codegen_crate: Some("gpu-kernels, math_gpu"),
        device_cfgs: &device_cfgs,
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    let cargo_args = vec![
        "-p".to_string(),
        "gpu-app".to_string(),
        "--".to_string(),
        "--nocapture".to_string(),
    ];

    let cmd =
        passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Test, &opts, &cargo_args)
            .unwrap();
    assert_eq!(
        cmd.get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        [
            "test",
            "--features",
            "wrapper_feature",
            "-p",
            "gpu-app",
            "--",
            "--nocapture",
        ]
    );
    assert_eq!(
        command_env(&cmd, "CARGO_TARGET_DIR").as_deref(),
        Some("cli-target")
    );
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_DEVICE_CODEGEN_CRATE").as_deref(),
        Some("gpu_kernels,math_gpu")
    );
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
        Some("sm_90")
    );
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_VERBOSE").as_deref(),
        Some("1")
    );

    let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
    let flags = decoded_rustflags(&encoded);
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--cfg", "from_config"])
    );
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
    );
    assert!(has_backend_identity_cfg(&flags));
    assert!(!flags.iter().any(|flag| {
        flag.contains("cuda_oxide_internal_codegen_env")
            || flag.contains("cuda_oxide_internal_materializer_provenance")
    }));
    assert!(is_sha256(
        &command_env(&cmd, CODEGEN_FINGERPRINT_ENV).unwrap()
    ));
    assert!(
        cmd.get_envs()
            .any(|(key, value)| key == OsStr::new("RUSTFLAGS") && value.is_none())
    );
}

#[test]
fn build_passthrough_debug_assertions_select_the_wrapper_profile() {
    let ctx = test_context(OxideConfig::default());
    let opts = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: None,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: true,
    };

    let cmd =
        passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Build, &opts, &[]).unwrap();
    let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
    let flags = decoded_rustflags(&encoded);

    assert_eq!(
        &flags[flags.len() - 6..],
        [
            "-Copt-level=3",
            "-Cdebug-assertions=on",
            "-Coverflow-checks=off",
            "-Zmir-enable-passes=-JumpThreading",
            "-Zalways-encode-mir",
            "-Csymbol-mangling-version=v0",
        ]
    );
}

#[test]
fn passthrough_command_accepts_empty_cargo_args() {
    let ctx = test_context(OxideConfig::default());
    let opts = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: None,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };

    let cmd =
        passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Test, &opts, &[]).unwrap();
    assert_eq!(
        cmd.get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        ["test"]
    );
}

#[test]
fn architecture_and_output_mode_do_not_change_global_rustflags() {
    let ctx = test_context(OxideConfig::default());
    let base = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: Some("sm_80"),
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    let base_cmd =
        passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Build, &base, &[]).unwrap();
    let different_mode = CargoPassthroughOptions {
        emit_nvvm_ir: true,
        arch: Some("sm_90"),
        ..base
    };
    let different_cmd = passthrough_command_for_test(
        &ctx,
        CargoPassthroughSubcommand::Build,
        &different_mode,
        &[],
    )
    .unwrap();

    assert_eq!(
        command_env(&base_cmd, "CARGO_ENCODED_RUSTFLAGS"),
        command_env(&different_cmd, "CARGO_ENCODED_RUSTFLAGS"),
        "architecture/output switches must not invalidate every dependency"
    );
    assert_ne!(
        command_env(&base_cmd, CODEGEN_FINGERPRINT_ENV),
        command_env(&different_cmd, CODEGEN_FINGERPRINT_ENV),
        "device owners still need a distinct Cargo identity"
    );
}

#[test]
fn codegen_mode_changes_rebuild_only_the_tracked_device_owner() {
    let root = unique_temp_dir("cargo_oxide_scoped_codegen_fingerprint");
    let target = root.join("target");
    for path in [
        root.join("shared-dep/src"),
        root.join("tracked-macro/src"),
        root.join("device-owner/src"),
        root.join("device-consumer/src"),
    ] {
        std::fs::create_dir_all(path).unwrap();
    }
    std::fs::write(
        root.join("Cargo.toml"),
        r#"[workspace]
resolver = "3"
members = ["shared-dep", "tracked-macro", "device-owner", "device-consumer"]
"#,
    )
    .unwrap();
    std::fs::write(
        root.join("shared-dep/Cargo.toml"),
        r#"[package]
name = "shared-dep"
version = "0.0.0"
edition = "2024"
"#,
    )
    .unwrap();
    std::fs::write(
        root.join("shared-dep/src/lib.rs"),
        "pub fn shared_value() -> u32 { 42 }\n",
    )
    .unwrap();
    std::fs::write(
        root.join("tracked-macro/Cargo.toml"),
        r#"[package]
name = "tracked-macro"
version = "0.0.0"
edition = "2024"

[lib]
proc-macro = true
"#,
    )
    .unwrap();
    std::fs::write(
        root.join("tracked-macro/src/lib.rs"),
        format!(
            r#"#![feature(proc_macro_tracked_env)]
extern crate proc_macro;

#[proc_macro]
pub fn track_codegen(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {{
    let _ = proc_macro::tracked::env_var({CODEGEN_FINGERPRINT_ENV:?});
    let _ = proc_macro::tracked::env_var({MATERIALIZE_ENV:?});
    let _ = proc_macro::tracked::env_var({EXPECTED_PROVENANCE_ENV:?});
    "()".parse().unwrap()
}}
"#
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("device-owner/Cargo.toml"),
        r#"[package]
name = "device-owner"
version = "0.0.0"
edition = "2024"

[dependencies]
shared-dep = { path = "../shared-dep" }
tracked-macro = { path = "../tracked-macro" }
"#,
    )
    .unwrap();
    std::fs::write(
            root.join("device-owner/src/lib.rs"),
            "const _: () = tracked_macro::track_codegen!();\npub fn device_value() -> u32 { shared_dep::shared_value() }\n",
        )
        .unwrap();
    std::fs::write(
        root.join("device-consumer/Cargo.toml"),
        r#"[package]
name = "device-consumer"
version = "0.0.0"
edition = "2024"

[dependencies]
device-owner = { path = "../device-owner" }
"#,
    )
    .unwrap();
    std::fs::write(
        root.join("device-consumer/src/main.rs"),
        "fn main() { assert_eq!(device_owner::device_value(), 42); }\n",
    )
    .unwrap();

    let ctx = Context {
        workspace_root: root.clone(),
        codegen_crate: root.join("unused-codegen-source"),
        examples_dir: root.join("unused-examples"),
        backend_so: PathBuf::from("llvm"),
        is_workspace: false,
        config: OxideConfig::default(),
    };
    let base = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: Some("sm_80"),
        features: None,
        cargo_target_dir: Some(&target),
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };

    let cold = cargo_artifact_freshness(&ctx, &base, None);
    assert_eq!(cold.get("shared_dep"), Some(&false));
    assert_eq!(cold.get("tracked_macro"), Some(&false));
    assert_eq!(cold.get("device_owner"), Some(&false));
    assert_eq!(cold.get("device-consumer"), Some(&false));

    let warm = cargo_artifact_freshness(&ctx, &base, None);
    assert_eq!(warm.get("shared_dep"), Some(&true));
    assert_eq!(warm.get("tracked_macro"), Some(&true));
    assert_eq!(warm.get("device_owner"), Some(&true));
    assert_eq!(warm.get("device-consumer"), Some(&true));

    let different_arch = CargoPassthroughOptions {
        arch: Some("sm_90"),
        ..base
    };
    let arch_switch = cargo_artifact_freshness(&ctx, &different_arch, None);
    assert_eq!(arch_switch.get("shared_dep"), Some(&true));
    assert_eq!(arch_switch.get("tracked_macro"), Some(&true));
    assert_eq!(arch_switch.get("device_owner"), Some(&false));
    assert_eq!(arch_switch.get("device-consumer"), Some(&false));

    let different_output = CargoPassthroughOptions {
        emit_nvvm_ir: true,
        ..different_arch
    };
    let output_switch = cargo_artifact_freshness(&ctx, &different_output, None);
    assert_eq!(output_switch.get("shared_dep"), Some(&true));
    assert_eq!(output_switch.get("tracked_macro"), Some(&true));
    assert_eq!(output_switch.get("device_owner"), Some(&false));
    assert_eq!(output_switch.get("device-consumer"), Some(&false));

    let repeated_output = cargo_artifact_freshness(&ctx, &different_output, None);
    assert_eq!(repeated_output.get("shared_dep"), Some(&true));
    assert_eq!(repeated_output.get("tracked_macro"), Some(&true));
    assert_eq!(repeated_output.get("device_owner"), Some(&true));
    assert_eq!(repeated_output.get("device-consumer"), Some(&true));

    let provenance_switch = cargo_artifact_freshness(
        &ctx,
        &different_output,
        Some("11d91fbe164094f6242d44103d0fb01968b96c6d8f48f124eac8fa73a307a657"),
    );
    assert_eq!(provenance_switch.get("shared_dep"), Some(&true));
    assert_eq!(provenance_switch.get("tracked_macro"), Some(&true));
    assert_eq!(provenance_switch.get("device_owner"), Some(&false));
    assert_eq!(provenance_switch.get("device-consumer"), Some(&false));

    let changed_provenance = cargo_artifact_freshness(
        &ctx,
        &different_output,
        Some("5b11618c2e44027877d0cd4d0cfd10afed5ef262876791e483ec58f4c5569139"),
    );
    assert_eq!(changed_provenance.get("shared_dep"), Some(&true));
    assert_eq!(changed_provenance.get("tracked_macro"), Some(&true));
    assert_eq!(changed_provenance.get("device_owner"), Some(&false));
    assert_eq!(changed_provenance.get("device-consumer"), Some(&false));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn owner_filter_resolution_is_normalized_and_has_explicit_precedence() {
    assert_eq!(
        resolve_device_codegen_crates(None, None, Some("gpu-kernels, math_gpu"))
            .unwrap()
            .as_deref(),
        Some("gpu_kernels,math_gpu"),
    );
    assert_eq!(
        resolve_device_codegen_crates(None, Some("parent-owner"), Some("config-owner"))
            .unwrap()
            .as_deref(),
        Some("parent_owner"),
    );
    assert!(
        resolve_device_codegen_crates(Some(""), Some("parent-owner"), Some("config-owner"))
            .is_err()
    );
}

#[test]
fn passthrough_fingerprint_tracks_output_affecting_settings() {
    let ctx = test_context(OxideConfig::default());
    let base = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: Some("sm_80"),
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    let inherited_env = BTreeMap::new();
    let base_hash = passthrough_codegen_fingerprint_with_env(
        &ctx,
        &base,
        None,
        Some("sm_80"),
        &MaterializationMode::default(),
        &inherited_env,
    );

    let arch = CargoPassthroughOptions {
        arch: Some("sm_90"),
        ..base
    };
    let emit = CargoPassthroughOptions {
        emit_nvvm_ir: true,
        ..base
    };
    let no_fmad = CargoPassthroughOptions {
        no_fmad: true,
        ..base
    };
    let unchecked_indexing = CargoPassthroughOptions {
        unchecked_indexing: true,
        ..base
    };
    let configured_ptx = test_context(OxideConfig {
        env: vec![(
            "CUDA_OXIDE_PTX_DIR".to_string(),
            "configured-ptx".to_string(),
        )],
        ..OxideConfig::default()
    });

    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &arch,
            None,
            Some("sm_90"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &emit,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &no_fmad,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &unchecked_indexing,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &base,
            Some("gpu_kernel"),
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &configured_ptx,
            &base,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        )
    );
    let materialized = MaterializationMode {
        prepared: Some(PreparedMaterialization {
            provenance_sha256_hex: "ab".repeat(32),
            tool_identity_handshake_json: "{\"version\":1}".to_string(),
        }),
    };
    assert_ne!(
        base_hash,
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &base,
            None,
            Some("sm_80"),
            &materialized,
            &inherited_env,
        ),
        "exact CUDA-tool provenance must change Cargo's rustc fingerprint"
    );
}

#[test]
fn passthrough_fingerprint_tracks_non_unicode_presence_switch_bytes() {
    let ctx = test_context(OxideConfig::default());
    let opts = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: Some("sm_80"),
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    let fingerprint = |inherited_env: &BTreeMap<String, Vec<u8>>| {
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            &opts,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            inherited_env,
        )
    };
    let absent = BTreeMap::new();
    let first = BTreeMap::from([("CUDA_OXIDE_NO_FMA".to_string(), vec![0xff])]);
    let second = BTreeMap::from([("CUDA_OXIDE_NO_FMA".to_string(), vec![0xfe])]);

    assert_ne!(fingerprint(&absent), fingerprint(&first));
    assert_ne!(fingerprint(&first), fingerprint(&second));
}

#[test]
fn global_backend_identity_tracks_rebuild_at_same_path() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "cargo_oxide_backend_fingerprint_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&root).unwrap();
    let backend = root.join("librustc_codegen_cuda.so");
    std::fs::write(&backend, b"first").unwrap();
    let original = std::fs::metadata(&backend).unwrap();
    let original_modified = original.modified().unwrap();

    let mut ctx = test_context(OxideConfig::default());
    ctx.backend_so = backend.clone();
    let fingerprint = "42".repeat(32);
    let mut before_cmd = Command::new("cargo");
    apply_codegen_configuration(
        &mut before_cmd,
        &ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    )
    .unwrap();
    let before = command_env(&before_cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
    // Preserve the weak metadata identity that used to be fingerprinted:
    // only the bytes differ.
    std::fs::write(&backend, b"other").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&backend)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(original_modified))
        .unwrap();
    let replacement = std::fs::metadata(&backend).unwrap();
    assert_eq!(replacement.len(), original.len());
    assert_eq!(replacement.modified().unwrap(), original_modified);
    let mut after_cmd = Command::new("cargo");
    apply_codegen_configuration(
        &mut after_cmd,
        &ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    )
    .unwrap();
    let after = command_env(&after_cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();

    assert_ne!(before, after);
    assert_eq!(
        command_env(&before_cmd, CODEGEN_FINGERPRINT_ENV),
        command_env(&after_cmd, CODEGEN_FINGERPRINT_ENV)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn owner_filter_rejects_empty_or_invalid_entries() {
    assert_eq!(
        normalize_device_codegen_crates("gpu-kernels, math_gpu").unwrap(),
        "gpu_kernels,math_gpu"
    );
    assert!(normalize_device_codegen_crates("").is_err());
    assert!(normalize_device_codegen_crates("   ").is_err());
    assert!(normalize_device_codegen_crates("gpu,").is_err());
    assert!(normalize_device_codegen_crates("gpu,not a crate").is_err());
}

#[test]
fn internal_ptx_directory_overrides_project_env_default() {
    let ctx = test_context(OxideConfig {
        env: vec![(
            "CUDA_OXIDE_PTX_DIR".to_string(),
            "configured-ptx".to_string(),
        )],
        ..OxideConfig::default()
    });
    let mut cmd = Command::new("cargo");
    apply_common_codegen_env(&mut cmd, &ctx, false, false, false, DeviceDebug::Off);
    cmd.env("CUDA_OXIDE_PTX_DIR", "internal-ptx");
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_PTX_DIR").as_deref(),
        Some("internal-ptx")
    );
}

#[test]
fn nvvm_arch_normalizes_all_accepted_forms() {
    // `sm_XX` is the form `--arch` and the rest of cargo-oxide use.
    assert_eq!(parse_nvvm_arch("sm_120").unwrap().compute(), "compute_120");
    assert_eq!(parse_nvvm_arch("sm_90").unwrap().compute(), "compute_90");
    // `compute_XX` passes through unchanged.
    assert_eq!(
        parse_nvvm_arch("compute_100").unwrap().compute(),
        "compute_100"
    );
    // A bare capability is accepted too.
    assert_eq!(parse_nvvm_arch("120").unwrap().compute(), "compute_120");
    assert!(parse_nvvm_arch("sm_90x").is_err());
}

#[test]
fn emit_ltoir_preserves_fma_and_debug_policy_for_libnvvm() {
    let arch = parse_nvvm_arch("sm_90").unwrap();
    for (artifact_debug, finalizer_debug) in [
        (
            oxide_artifacts::ArtifactDebugPolicy::None,
            cuda_artifact_finalizer::DebugPolicy::None,
        ),
        (
            oxide_artifacts::ArtifactDebugPolicy::LineTables,
            cuda_artifact_finalizer::DebugPolicy::LineTables,
        ),
        (
            oxide_artifacts::ArtifactDebugPolicy::Full,
            cuda_artifact_finalizer::DebugPolicy::Full,
        ),
    ] {
        let artifact_options = oxide_artifacts::ArtifactCompileOptions::new()
            .with_fma_contraction(false)
            .with_debug_policy(artifact_debug);
        let finalizer_options = finalization_options_from_artifact(&arch, artifact_options);

        assert_eq!(finalizer_options.target(), &arch);
        assert!(!finalizer_options.allow_fma_contraction());
        assert_eq!(finalizer_options.debug_policy(), finalizer_debug);
    }
}

#[test]
fn apply_output_mode_sets_target_for_arch_override() {
    let mut cmd = Command::new("cargo");

    apply_output_mode(
        &mut cmd,
        false,
        Some("sm_120"),
        &MaterializationMode::default(),
    );

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
        Some("sm_120")
    );
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
}

#[test]
fn apply_output_mode_sets_nvvm_ir_flag_and_target() {
    let mut cmd = Command::new("cargo");

    apply_output_mode(
        &mut cmd,
        true,
        Some("sm_100a"),
        &MaterializationMode::default(),
    );

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
        Some("sm_100a")
    );
    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR").as_deref(),
        Some("1")
    );
}

#[test]
fn materialization_forces_nvvm_ir_and_exact_provenance_handshake() {
    let mut cmd = Command::new("cargo");
    let materialization = MaterializationMode {
        prepared: Some(PreparedMaterialization {
            provenance_sha256_hex: "42".repeat(32),
            tool_identity_handshake_json: "{\"version\":1}".to_string(),
        }),
    };

    apply_output_mode(&mut cmd, false, Some("sm_90"), &materialization);

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR").as_deref(),
        Some("1")
    );
    assert_eq!(command_env(&cmd, MATERIALIZE_ENV).as_deref(), Some("1"));
    assert_eq!(
        command_env(&cmd, MATERIALIZER_HANDSHAKE_ENV).as_deref(),
        Some("{\"version\":1}")
    );
    assert_eq!(
        command_env(&cmd, EXPECTED_PROVENANCE_ENV).as_deref(),
        Some("4242424242424242424242424242424242424242424242424242424242424242")
    );
}

#[test]
fn apply_output_mode_leaves_auto_detect_ptx_unset() {
    let mut cmd = Command::new("cargo");

    apply_output_mode(&mut cmd, false, None, &MaterializationMode::default());

    assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
}

#[test]
fn apply_device_arch_hint_sets_hint_when_no_explicit_arch() {
    let mut cmd = Command::new("cargo");

    apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
        Some("sm_120a")
    );
    // The hint must never masquerade as the hard override.
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
}

#[test]
fn apply_device_arch_hint_skipped_when_arch_explicit() {
    // An explicit --arch already went to CUDA_OXIDE_TARGET; don't also
    // emit a competing device hint.
    let mut cmd = Command::new("cargo");

    apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

    assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
}

#[test]
fn apply_device_arch_hint_noop_without_detection() {
    let mut cmd = Command::new("cargo");

    apply_device_arch_hint(&mut cmd, None, None);

    assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
}

#[test]
fn debug_output_mode_forwards_detected_gpu_hint() {
    let mut cmd = Command::new("cargo");

    apply_output_mode(&mut cmd, false, None, &MaterializationMode::default());
    apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
        Some("sm_120a")
    );
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
}

#[test]
fn debug_output_mode_honors_explicit_arch_override() {
    let mut cmd = Command::new("cargo");

    apply_output_mode(
        &mut cmd,
        false,
        Some("sm_90"),
        &MaterializationMode::default(),
    );
    apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

    assert_eq!(
        command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
        Some("sm_90")
    );
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
    assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
}

#[test]
fn format_sm_arch_uses_cuda_target_spelling() {
    // cc < 9.0 — no arch-specific target exists in the PTX ISA, so we
    // emit the plain `sm_XY` form. Confirms we do not produce false
    // positives like `sm_75a` / `sm_80a` / `sm_89a`.
    assert_eq!(format_sm_arch((7, 0)), "sm_70");
    assert_eq!(format_sm_arch((7, 5)), "sm_75");
    assert_eq!(format_sm_arch((8, 0)), "sm_80");
    assert_eq!(format_sm_arch((8, 6)), "sm_86");
    assert_eq!(format_sm_arch((8, 9)), "sm_89");

    // cc ≥ 9.0 — every chip that reports this CC is an arch-specific
    // (`a`) variant. Auto-detect emits the `a` form so the codegen
    // backend can lower WGMMA / tcgen05 / TMA-multicast / cta_group
    // intrinsics without falling through to a plain target that ptxas
    // would reject. Confirms we do not produce false negatives.
    assert_eq!(format_sm_arch((9, 0)), "sm_90a"); // Hopper (H100/H200)
    assert_eq!(format_sm_arch((10, 0)), "sm_100a"); // Blackwell DC
    assert_eq!(format_sm_arch((10, 1)), "sm_101a");
    assert_eq!(format_sm_arch((10, 3)), "sm_103a");
    assert_eq!(format_sm_arch((12, 0)), "sm_120a"); // consumer Blackwell
}

#[test]
fn parse_compute_cap_accepts_real_nvidia_smi_output() {
    assert_eq!(parse_compute_cap("12.0\n"), Some((12, 0)));
    assert_eq!(parse_compute_cap("7.5\n"), Some((7, 5)));
    assert_eq!(parse_compute_cap("10.3"), Some((10, 3)));
    // End-to-end with format_sm_arch: the values the backend sees.
    assert_eq!(
        format_sm_arch(parse_compute_cap("12.0\n").unwrap()),
        "sm_120a"
    );
    assert_eq!(format_sm_arch(parse_compute_cap("7.5\n").unwrap()), "sm_75");
}

#[test]
fn parse_compute_cap_takes_first_gpu_on_multi_gpu_machines() {
    assert_eq!(parse_compute_cap("9.0\n12.0\n"), Some((9, 0)));
}

#[test]
fn parse_gpu_name_cap_and_driver_splits_on_last_two_commas() {
    assert_eq!(
        parse_gpu_name_cap_and_driver("NVIDIA GeForce RTX 5090, 12.0, 580.65.06\n"),
        Some((
            "NVIDIA GeForce RTX 5090".to_string(),
            (12, 0),
            "580.65.06".to_string()
        ))
    );
    // Failure banner: no comma-separated cc/driver fields.
    assert_eq!(
        parse_gpu_name_cap_and_driver("NVIDIA-SMI has failed.\n"),
        None
    );
    assert_eq!(parse_gpu_name_cap_and_driver(""), None);
}

#[test]
fn cuda_toolkit_resolution_prefers_toolkit_path_then_home() {
    let accept_all = |_: &Path| Ok(13_030);

    let both = resolve_cuda_toolkit(
        |var| match var {
            "CUDA_TOOLKIT_PATH" => Some("/cuda/toolkit".into()),
            "CUDA_HOME" => Some("/cuda/home".into()),
            _ => None,
        },
        accept_all,
    );
    assert_eq!(both.root, Path::new("/cuda/toolkit"));
    assert_eq!(both.source, ToolkitSource::Explicit("CUDA_TOOLKIT_PATH"));
    assert!(both.rejected.is_empty());

    let home_only = resolve_cuda_toolkit(
        |var| (var == "CUDA_HOME").then(|| "/cuda/home".into()),
        accept_all,
    );
    assert_eq!(home_only.root, Path::new("/cuda/home"));
    assert_eq!(home_only.source, ToolkitSource::Explicit("CUDA_HOME"));
}

#[test]
fn an_explicit_toolkit_that_does_not_validate_is_answered_not_stepped_over() {
    // The build script fails on a variable that is set but does not validate,
    // rather than falling through to discovery. A typo in CUDA_HOME must not
    // read here as a healthy default install (#1265).
    let typo = resolve_cuda_toolkit(
        |var| (var == "CUDA_HOME").then(|| "/cuda/typo".into()),
        |root| Err(format!("{} does not contain cuda.h", root.display())),
    );
    assert_eq!(typo.source, ToolkitSource::Unresolved);
    assert_eq!(typo.root, Path::new("/cuda/typo"));
    assert_eq!(
        typo.rejected,
        vec!["CUDA_HOME=/cuda/typo is invalid: /cuda/typo does not contain cuda.h"]
    );

    // An empty value is its own error there, not an unset variable.
    let empty = resolve_cuda_toolkit(
        |var| (var == "CUDA_TOOLKIT_PATH").then(std::ffi::OsString::new),
        |_| Ok(13_030),
    );
    assert_eq!(empty.source, ToolkitSource::Unresolved);
    assert_eq!(
        empty.rejected,
        vec!["CUDA_TOOLKIT_PATH is set to an empty string"]
    );
}

#[cfg(unix)]
#[test]
fn toolkit_resolution_preserves_non_unicode_explicit_paths() {
    use std::os::unix::ffi::OsStringExt;

    let parent = unique_temp_dir("cargo_oxide_toolkit_non_unicode");
    let root = parent.join(std::ffi::OsString::from_vec(b"cuda-\xff".to_vec()));
    fs::create_dir_all(root.join("include")).unwrap();
    fs::write(root.join("include/cuda.h"), "#define CUDA_VERSION 13000\n").unwrap();

    let resolve = || {
        resolve_cuda_toolkit(
            |key| (key == "CUDA_TOOLKIT_PATH").then(|| root.clone().into_os_string()),
            |path| validate_cuda_toolkit(path, None),
        )
    };
    let choice = resolve();
    assert_eq!(choice.root, root);
    assert_eq!(choice.source, ToolkitSource::Explicit("CUDA_TOOLKIT_PATH"));

    fs::remove_file(root.join("include/cuda.h")).unwrap();
    let rejected = resolve();
    assert_eq!(rejected.root, root);
    assert_eq!(rejected.source, ToolkitSource::Unresolved);
    assert_eq!(rejected.rejected.len(), 1);
    fs::remove_dir_all(parent).unwrap();
}

#[test]
fn discovery_takes_the_first_candidate_that_validates() {
    // The drift this fixes: the old mirror answered `/usr/local/cuda` for an
    // unset environment, while the build script walks a fixed list and stops
    // at the first root carrying a new-enough cuda.h. Indices rather than
    // literals, so the assertion is about the order and not about the paths.
    let chosen = DEFAULT_TOOLKIT_CANDIDATES[1];
    let skipped = DEFAULT_TOOLKIT_CANDIDATES[0];
    let choice = resolve_cuda_toolkit(
        |_| None,
        |root| {
            if root == Path::new(chosen) {
                Ok(13_020)
            } else {
                Err(format!("{} does not contain cuda.h", root.display()))
            }
        },
    );
    assert_eq!(choice.source, ToolkitSource::Discovered);
    assert_eq!(choice.root, Path::new(chosen));
    assert_eq!(
        choice.rejected,
        vec![format!("{skipped}: {skipped} does not contain cuda.h")]
    );

    // Nothing validates: every candidate is reported, and the root is only
    // somewhere to point the message.
    let nothing = resolve_cuda_toolkit(
        |_| None,
        |root| Err(format!("{} is not a directory", root.display())),
    );
    assert_eq!(nothing.source, ToolkitSource::Unresolved);
    assert_eq!(
        nothing.root,
        Path::new(DEFAULT_TOOLKIT_CANDIDATES.last().unwrap())
    );
    assert_eq!(nothing.rejected.len(), DEFAULT_TOOLKIT_CANDIDATES.len());
}

#[test]
fn validate_cuda_toolkit_applies_the_build_scripts_acceptance() {
    let root = unique_temp_dir("cargo_oxide_toolkit_validate");
    fs::create_dir_all(root.join("include")).unwrap();
    let cuda_h = root.join("include/cuda.h");
    let root_str = root.to_str().unwrap().to_string();

    fs::write(&cuda_h, "#define CUDA_VERSION 13030\n").unwrap();
    assert_eq!(validate_cuda_toolkit(&root_str, None), Ok(13_030));

    // A host include directory must never shadow an explicit target tree.
    assert!(validate_cuda_toolkit(&root_str, Some("missing-target")).is_err());
    let target = root.join("targets/test-target/include");
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("cuda.h"), "#define CUDA_VERSION 13000\n").unwrap();
    assert_eq!(
        validate_cuda_toolkit(&root_str, Some("test-target")),
        Ok(13_000)
    );
    fs::write(target.join("cuda.h"), "#define CUDA_VERSION 12080\n").unwrap();
    assert!(validate_cuda_toolkit(&root_str, Some("test-target")).is_err());

    // The floor is the build script's 13.0, not a guess: a toolkit below it
    // is rejected even though cuda.h is right there.
    fs::write(&cuda_h, "#define CUDA_VERSION 12080\n").unwrap();
    let too_old = validate_cuda_toolkit(&root_str, None).unwrap_err();
    assert!(too_old.contains("too old"), "{too_old}");
    assert!(too_old.contains("12.8"), "{too_old}");

    fs::remove_file(&cuda_h).unwrap();
    let headerless = validate_cuda_toolkit(&root_str, None).unwrap_err();
    assert!(
        headerless.contains("does not contain cuda.h"),
        "{headerless}"
    );

    let absent = validate_cuda_toolkit(format!("{root_str}/nope"), None).unwrap_err();
    assert!(absent.contains("is not a directory"), "{absent}");

    fs::remove_dir_all(&root).ok();
}

#[test]
fn cuda_version_from_header_reads_the_define_the_build_script_reads() {
    assert_eq!(
        cuda_version_from_header("#define CUDA_VERSION 13030"),
        Some(13_030)
    );
    assert_eq!(
        cuda_version_from_header("/* preamble */\n#define CUDA_VERSION   13000  \n"),
        Some(13_000)
    );
    // A neighbouring define is not the one, and neither is prose.
    assert_eq!(
        cuda_version_from_header("#define CUDA_VERSION_MAJOR 13"),
        None
    );
    assert_eq!(cuda_version_from_header("no defines here"), None);

    assert_eq!(format_cuda_version(13_030), "13.3");
    assert_eq!(format_cuda_version(13_000), "13.0");
    assert_eq!(format_cuda_version(12_080), "12.8");
}

#[test]
fn cuda_header_candidates_cover_standard_and_redistributable_layouts() {
    // Standard install layout first, then the matching targets/ layout.
    assert_eq!(
        cuda_header_candidates("/usr/local/cuda", None, "x86_64", "linux"),
        vec![
            PathBuf::from("/usr/local/cuda/include/cuda.h"),
            PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include/cuda.h"),
        ]
    );
    // aarch64 Linux is ambiguous between servers (sbsa-linux) and Tegra
    // (aarch64-linux), so both are probed, servers first.
    assert_eq!(
        cuda_header_candidates("/opt/ctk", None, "aarch64", "linux"),
        vec![
            PathBuf::from("/opt/ctk/include/cuda.h"),
            PathBuf::from("/opt/ctk/targets/sbsa-linux/include/cuda.h"),
            PathBuf::from("/opt/ctk/targets/aarch64-linux/include/cuda.h"),
        ]
    );
    // Unknown host arch or non-Linux OS: only the standard layout.
    assert_eq!(
        cuda_header_candidates("/opt/ctk", None, "riscv64", "linux"),
        vec![PathBuf::from("/opt/ctk/include/cuda.h")]
    );
    assert_eq!(
        cuda_header_candidates("/opt/ctk", None, "aarch64", "macos"),
        vec![PathBuf::from("/opt/ctk/include/cuda.h")]
    );
    // CUDA_TOOLKIT_TARGET_DIR replaces the table with one directory;
    // a blank value means "unset".
    assert_eq!(
        cuda_header_candidates("/opt/ctk", Some("aarch64-linux"), "aarch64", "linux"),
        vec![PathBuf::from(
            "/opt/ctk/targets/aarch64-linux/include/cuda.h"
        )]
    );
    assert_eq!(
        cuda_header_candidates("/opt/ctk", Some("  "), "x86_64", "linux"),
        vec![
            PathBuf::from("/opt/ctk/include/cuda.h"),
            PathBuf::from("/opt/ctk/targets/x86_64-linux/include/cuda.h"),
        ]
    );
}

#[test]
fn parse_rust_toolchain_toml_reads_channel_and_components() {
    let pin = parse_rust_toolchain_toml(
        r#"[toolchain]
channel = "nightly-2026-08-28"
components = ["rust-src", "rustc-dev", "llvm-tools"]
"#,
    )
    .expect("pin should parse");
    assert_eq!(pin.channel, "nightly-2026-08-28");
    assert_eq!(
        pin.components,
        vec![
            "rust-src".to_string(),
            "rustc-dev".to_string(),
            "llvm-tools".to_string()
        ]
    );
}

#[test]
fn parse_rust_toolchain_toml_rejects_missing_channel() {
    let error = parse_rust_toolchain_toml("[toolchain]\ncomponents = [\"rust-src\"]\n")
        .expect_err("channel is required");
    assert!(error.contains("channel"), "{error}");
}

#[test]
fn active_toolchain_matches_channel_accepts_target_triple_suffix() {
    assert!(active_toolchain_matches_channel(
        "nightly-2026-08-28-aarch64-apple-darwin (default)",
        "nightly-2026-08-28"
    ));
    assert!(active_toolchain_matches_channel(
        "nightly-2026-08-28",
        "nightly-2026-08-28"
    ));
    assert!(!active_toolchain_matches_channel(
        "nightly-2026-01-01-x86_64-unknown-linux-gnu (default)",
        "nightly-2026-08-28"
    ));
}

#[test]
fn active_toolchain_matches_channel_accepts_rustup_128_and_129_formats() {
    // rustup 1.29 single-line form with an override annotation, as
    // observed verbatim on a workspace with rust-toolchain.toml.
    assert!(active_toolchain_matches_channel(
        "nightly-2026-08-28-x86_64-unknown-linux-gnu (overridden by \
             '/home/user/cuda-oxide/rust-toolchain.toml')",
        "nightly-2026-08-28"
    ));
    // rustup 1.28 two-line form: bare name, then the reason line.
    assert!(active_toolchain_matches_channel(
        "nightly-2026-08-28-x86_64-unknown-linux-gnu\nactive because: \
             overridden by '/home/user/cuda-oxide/rust-toolchain.toml'",
        "nightly-2026-08-28"
    ));
    // A mismatched pin must not be rescued by later lines.
    assert!(!active_toolchain_matches_channel(
        "stable-x86_64-unknown-linux-gnu\nactive because: default",
        "nightly-2026-08-28"
    ));
}

#[test]
fn plan_update_selects_advise_setup_or_cache_refresh() {
    assert_eq!(plan_update(true, false), UpdatePlan::AdviseSetup);
    assert_eq!(plan_update(true, true), UpdatePlan::RunSetup);
    assert_eq!(plan_update(false, false), UpdatePlan::RefreshCache);
    assert_eq!(plan_update(false, true), UpdatePlan::RefreshCache);
}

/// A `.cargo/cuda-oxide.toml` backend pin outranks the shared cache, so
/// `update` must refuse just like it does for `CUDA_OXIDE_BACKEND`.
#[test]
fn update_refuses_when_the_config_pins_a_backend() {
    let pinned = test_context(OxideConfig {
        backend: Some(PathBuf::from("/tmp/pinned-backend.so")),
        ..OxideConfig::default()
    });
    // `None` stands in for an unset ambient `CUDA_OXIDE_BACKEND`. Reading
    // the real one would let an exported value produce the env refusal for
    // both inputs, including the unpinned case asserted to be `None`.
    let refusal =
        update_pin_refusal_with_env(&pinned, None).expect("config pin must refuse update");
    assert!(refusal.contains("pins the backend"), "{refusal}");
    assert!(refusal.contains("/tmp/pinned-backend.so"), "{refusal}");

    let unpinned = test_context(OxideConfig::default());
    assert_eq!(update_pin_refusal_with_env(&unpinned, None), None);

    // The env var outranks the project pin: set, it refuses even unpinned.
    let from_env = update_pin_refusal_with_env(&unpinned, Some("/tmp/env-backend.so".into()))
        .expect("exported CUDA_OXIDE_BACKEND must refuse update");
    assert!(from_env.contains("CUDA_OXIDE_BACKEND is set"), "{from_env}");
}

#[test]
fn doctor_verified_components_unions_pin_list_with_required_floor() {
    // Pin lists everything: order preserved, no duplicates appended.
    let pin = RustToolchainPin {
        channel: "nightly-2026-08-28".to_string(),
        components: vec![
            "rust-src".to_string(),
            "rustc-dev".to_string(),
            "rust-analyzer".to_string(),
            "clippy".to_string(),
            "llvm-tools".to_string(),
        ],
    };
    assert_eq!(
        doctor_verified_components(&pin),
        vec![
            "rust-src",
            "rustc-dev",
            "rust-analyzer",
            "clippy",
            "llvm-tools"
        ]
    );

    // A trimmed pin still gets the hard floor appended.
    let trimmed = RustToolchainPin {
        channel: "nightly-2026-08-28".to_string(),
        components: vec!["clippy".to_string()],
    };
    assert_eq!(
        doctor_verified_components(&trimmed),
        vec!["clippy", "rust-src", "rustc-dev", "llvm-tools"]
    );
}

#[test]
fn missing_rustup_components_detects_host_triple_suffixes() {
    let installed = "\
rust-src-aarch64-apple-darwin
clippy-aarch64-apple-darwin
";
    assert_eq!(
        missing_rustup_components(installed, &["rust-src", "llvm-tools"]),
        vec!["llvm-tools".to_string()]
    );
    assert!(missing_rustup_components(installed, &["rust-src"]).is_empty());
}

#[test]
fn parse_compute_cap_rejects_failure_banners_and_garbage() {
    // nvidia-smi prints failure text to STDOUT, not stderr.
    assert_eq!(
        parse_compute_cap(
            "NVIDIA-SMI has failed because it couldn't communicate \
                 with the NVIDIA driver.\n"
        ),
        None
    );
    assert_eq!(parse_compute_cap(""), None);
    assert_eq!(parse_compute_cap("\n"), None);
    assert_eq!(parse_compute_cap("N/A\n"), None);
    assert_eq!(parse_compute_cap("12\n"), None);
    assert_eq!(parse_compute_cap("12.\n"), None);
    assert_eq!(parse_compute_cap(".5\n"), None);
    assert_eq!(parse_compute_cap("12.0.1\n"), None);
}

// All three skip cases inject the `CUDA_OXIDE_TARGET` probe rather than
// reading the ambient one, so each asserts the slot it names instead of
// passing because the developer happens to have the variable exported.

#[test]
fn detect_run_target_arch_skips_when_arch_explicit() {
    // --arch wins; never query the GPU.
    assert_eq!(
        detect_run_target_arch_with_env(Some("sm_120"), false, false),
        None
    );
}

#[test]
fn detect_run_target_arch_skips_when_emit_nvvm_ir() {
    // NVVM IR mode requires explicit --arch; auto-detect must not run.
    assert_eq!(detect_run_target_arch_with_env(None, true, false), None);
}

#[test]
fn detect_run_target_arch_skips_when_env_target_set() {
    // Slot 2 wins; never query the GPU. Injected rather than exported:
    // `set_var` is a data race against the `vars_os` reads the fingerprint
    // helpers perform on other test threads, which the cargo test harness
    // runs concurrently by default.
    assert_eq!(detect_run_target_arch_with_env(None, false, true), None);
}

#[test]
fn build_arch_warning_is_emitted_when_arch_is_unconfigured_and_local_gpu_exists() {
    let warning = build_arch_warning(false, Some("sm_121a"))
        .expect("an unconfigured build with a local GPU should warn");

    assert!(warning.contains("backend default"));
    assert!(warning.contains("first GPU reported by `nvidia-smi`"));
    assert!(warning.contains("sm_121a"));
    assert!(warning.contains("--arch <sm_XX>"));
}

#[test]
fn build_arch_warning_is_suppressed_when_arch_is_configured() {
    assert_eq!(build_arch_warning(true, Some("sm_121a")), None);
}

#[test]
fn build_arch_warning_is_suppressed_when_no_local_gpu_is_detected() {
    assert_eq!(build_arch_warning(false, None), None);
}

fn write_list_example(
    examples_dir: &Path,
    name: &str,
    manifest_description: Option<&str>,
    readme: Option<&str>,
) {
    let example_dir = examples_dir.join(name);
    std::fs::create_dir_all(&example_dir).unwrap();

    let description = manifest_description
        .map(|value| format!("description = {value:?}\n"))
        .unwrap_or_default();

    std::fs::write(
        example_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = {name:?}\nversion = \"0.1.0\"\nedition = \"2024\"\n{description}"
        ),
    )
    .unwrap();

    if let Some(readme) = readme {
        std::fs::write(example_dir.join("README.md"), readme).unwrap();
    }
}

#[test]
fn readme_parser_extracts_title_description_and_requirements() {
    let parsed = parse_example_readme(
        "vecadd",
        r#"
# vecadd

## Vector Addition

Adds two vectors using one CUDA thread per element.

## Hardware Requirements

- **Minimum GPU**: sm_70+
- **CUDA Toolkit**: 12.x+
"#,
    );

    assert_eq!(parsed.title.as_deref(), Some("Vector Addition"));
    assert_eq!(
        parsed.description.as_deref(),
        Some("Adds two vectors using one CUDA thread per element.")
    );
    assert_eq!(
        parsed.requirements,
        ["Minimum GPU: sm_70+", "CUDA Toolkit: 12.x+"]
    );
}

#[test]
fn readme_parser_does_not_use_run_as_title() {
    let parsed = parse_example_readme(
        "cuda_module_nested",
        r#"
# cuda_module_nested

## Run

Expected output:

```text
PASS
```

"#,
    );

    assert_eq!(parsed.title.as_deref(), Some("cuda_module_nested"));
    assert_eq!(parsed.description, None);
}

#[test]
fn readme_parser_does_not_scan_later_headings_for_title() {
    let parsed = parse_example_readme(
        "example",
        r#"

# example

Introductory description.

## Build

Build instructions.

## Advanced Implementation Details

Internal details.
"#,
    );

    assert_eq!(parsed.title.as_deref(), Some("example"));
    assert_eq!(
        parsed.description.as_deref(),
        Some("Introductory description.")
    );
}

#[test]
fn readme_parser_stops_description_at_next_heading() {
    let parsed = parse_example_readme(
        "vecadd",
        r#"

# vecadd

## Vector Addition

Adds two vectors on the GPU.

## Run

Run the example with cargo oxide.
"#,
    );

    assert_eq!(parsed.title.as_deref(), Some("Vector Addition"));
    assert_eq!(
        parsed.description.as_deref(),
        Some("Adds two vectors on the GPU.")
    );
}

#[test]
fn requirement_parser_joins_wrapped_list_items() {
    let parsed = parse_example_readme(
        "example",
        r#"

# example

## Requirements

* CUDA Toolkit 13.1+ with nvcc and tileiras available. This example
  also requires the CUDA development libraries.
* Blackwell GPU with sm_100+ support.
  "#,
    );

    assert_eq!(
        parsed.requirements,
        [
            "CUDA Toolkit 13.1+ with nvcc and tileiras available. This example also requires the CUDA development libraries.",
            "Blackwell GPU with sm_100+ support.",
        ]
    );
}

#[test]
fn requirement_parser_does_not_absorb_paragraph_after_blank_line() {
    // Modeled on the cpp_consumes_rust_device README: a bullet list under
    // the requirements heading, then a blank line, then a follow-up
    // paragraph and a code fence. The paragraph is a new paragraph, not a
    // wrapped continuation of the last bullet.
    let parsed = parse_example_readme(
        "cpp_consumes_rust_device",
        r#"
# cpp_consumes_rust_device

## Prerequisites

- CUDA Toolkit (nvcc, libNVVM, nvJitLink)
- Blackwell+ GPU (sm_100+) — LTOIR requires NVVM 20 dialect

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run cpp_consumes_rust_device
```
"#,
    );

    assert_eq!(
        parsed.requirements,
        [
            "CUDA Toolkit (nvcc, libNVVM, nvJitLink)",
            "Blackwell+ GPU (sm_100+) — LTOIR requires NVVM 20 dialect",
        ]
    );
}

#[test]
fn requirement_parser_joins_wrapped_items_but_not_following_paragraphs() {
    // Modeled on the cutile_inter_kernel README: the last bullet wraps
    // across indented lines (joined), and the paragraph after the blank
    // line must not be glued onto it.
    let parsed = parse_example_readme(
        "cutile_inter_kernel",
        r#"
# cutile_inter_kernel

## Requirements

- cuda-oxide from this repository.
- CUDA Toolkit 13.1+ with `nvcc` and `tileiras` available. This example
  defaults `CUDA_TOOLKIT_PATH` to `/usr/local/cuda` through its local Cargo
  config; set `CUDA_TOOLKIT_PATH` yourself if your toolkit lives elsewhere.

`cargo oxide run` targets explicit `--arch` first, then `CUDA_OXIDE_TARGET`,
then auto-detects the local GPU.

## Run

Run instructions.
"#,
    );

    assert_eq!(
        parsed.requirements,
        [
            "cuda-oxide from this repository.",
            "CUDA Toolkit 13.1+ with nvcc and tileiras available. This example \
                 defaults CUDA_TOOLKIT_PATH to /usr/local/cuda through its local Cargo \
                 config; set CUDA_TOOLKIT_PATH yourself if your toolkit lives elsewhere.",
        ]
    );
}

#[test]
fn requirement_parser_captures_ordered_list_items() {
    // Modeled on the mathdx_ffi_test README: prerequisites written as an
    // ordered list, followed by a paragraph that is not part of the list.
    let parsed = parse_example_readme(
        "mathdx_ffi_test",
        r#"
# mathdx_ffi_test

## Prerequisites

1. **CUDA Toolkit 12.x+** with nvcc
2. **MathDx Library** - Download from: https://developer.nvidia.com/cublasdx-downloads
3. **cuda-oxide compiler** toolchain

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example.
"#,
    );

    assert_eq!(
        parsed.requirements,
        [
            "CUDA Toolkit 12.x+ with nvcc",
            "MathDx Library - Download from: https://developer.nvidia.com/cublasdx-downloads",
            "cuda-oxide compiler toolchain",
        ]
    );
}

#[test]
fn requirement_parser_recognizes_build_requirements_heading() {
    let parsed = parse_example_readme(
        "example",
        r#"
# example

## Build Requirements

- nvcc with `--expt-relaxed-constexpr`
"#,
    );

    assert_eq!(parsed.requirements, ["nvcc with --expt-relaxed-constexpr"]);
}

#[test]
fn requirement_parser_parses_two_column_requirement_tables() {
    // Modeled on the abi_hmm README: requirements in a two-column table,
    // including an escaped pipe inside a cell.
    let parsed = parse_example_readme(
        "abi_hmm",
        r#"
# abi_hmm

## Requirements

| Requirement   | Minimum                                           |
|---------------|---------------------------------------------------|
| GPU           | Turing or newer (RTX 20xx+)                       |
| Linux Kernel  | 6.1.24+                                           |
| HMM Support   | `nvidia-smi -q \| grep Addressing` shows "HMM"    |

## Build and Run

Instructions.
"#,
    );

    assert_eq!(
        parsed.requirements,
        [
            "GPU: Turing or newer (RTX 20xx+)",
            "Linux Kernel: 6.1.24+",
            "HMM Support: nvidia-smi -q | grep Addressing shows \"HMM\"",
        ]
    );
}

#[test]
fn requirement_parser_skips_tables_that_are_not_two_columns() {
    // A three-column table has no unambiguous name/value mapping, so it
    // must be skipped whole instead of half-parsed.
    let parsed = parse_example_readme(
        "example",
        r#"
# example

## Requirements

| Test  | Status | Description |
|-------|--------|-------------|
| alpha | Pass   | First test  |
| beta  | Pass   | Second test |
"#,
    );

    assert_eq!(parsed.requirements, Vec::<String>::new());
}

#[test]
fn example_discovery_is_sorted_and_uses_manifest_fallback() {
    let root = unique_temp_dir("cargo_oxide_list_examples");
    std::fs::create_dir_all(&root).unwrap();

    write_list_example(&root, "zeta", Some("Manifest fallback description"), None);

    write_list_example(
        &root,
        "alpha",
        None,
        Some("# alpha\n\n## Alpha Example\n\nREADME description.\n"),
    );

    let examples = discover_examples(&root).unwrap();

    assert_eq!(
        examples
            .iter()
            .map(|example| example.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "zeta"]
    );
    assert_eq!(examples[0].description, "README description.");
    assert_eq!(examples[1].description, "Manifest fallback description");

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn example_discovery_keeps_examples_without_readmes() {
    let root = unique_temp_dir("cargo_oxide_list_missing_readme");
    std::fs::create_dir_all(&root).unwrap();

    write_list_example(&root, "minimal", None, None);

    let examples = discover_examples(&root).unwrap();

    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].name, "minimal");
    assert_eq!(examples[0].description, "No description documented.");

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn example_discovery_skips_directory_without_manifest() {
    let root = unique_temp_dir("cargo_oxide_list_missing_manifest");
    std::fs::create_dir_all(root.join("scratch")).unwrap();

    write_list_example(&root, "real", Some("A real example"), None);

    let examples =
        discover_examples(&root).expect("manifest-less directories must not abort listing");

    assert_eq!(
        examples
            .iter()
            .map(|example| example.name.as_str())
            .collect::<Vec<_>>(),
        ["real"]
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn interop_metadata_selects_named_binary_artifact() {
    let root = unique_temp_dir("cargo_oxide_named_bin_interop");
    let device_dir = root.join("device");
    std::fs::create_dir_all(&device_dir).unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "host-app"
version = "0.1.0"
edition = "2024"

[[package.metadata.cuda-oxide.device-crates]]
manifest-path = "device/Cargo.toml"
bin = "secondary-device"
"#,
    )
    .unwrap();
    std::fs::write(
        device_dir.join("Cargo.toml"),
        r#"[package]
name = "kernel-package"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    let config = load_interop_config(&root).expect("interop metadata should load");
    let device = &config.device_crates[0];
    assert_eq!(device.bin.as_deref(), Some("secondary-device"));
    assert_eq!(
        interop_device_cargo_target_name(&device_dir.join("Cargo.toml"), device),
        "secondary-device"
    );
    assert_eq!(
        interop_device_artifact_name(&device_dir.join("Cargo.toml"), device),
        "secondary_device"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn interop_binary_target_uses_cargo_source_path() {
    let manifest = Path::new("/workspace/kernels/Cargo.toml");
    let metadata = serde_json::json!({
        "packages": [{
            "manifest_path": "/workspace/kernels/Cargo.toml",
            "targets": [
                {
                    "name": "kernel-package",
                    "kind": ["lib"],
                    "src_path": "/workspace/kernels/src/lib.rs"
                },
                {
                    "name": "secondary-device",
                    "kind": ["bin"],
                    "src_path": "/workspace/kernels/src/device_secondary.rs"
                }
            ]
        }]
    });

    assert_eq!(
        interop_binary_target_from_metadata(&metadata, manifest, "secondary-device").unwrap(),
        InteropBinaryTarget {
            source_path: PathBuf::from("/workspace/kernels/src/device_secondary.rs"),
        }
    );
}

#[test]
fn interop_binary_target_rejects_unknown_name_with_available_targets() {
    let manifest = Path::new("/workspace/kernels/Cargo.toml");
    let metadata = serde_json::json!({
        "packages": [{
            "manifest_path": "/workspace/kernels/Cargo.toml",
            "targets": [{
                "name": "main-device",
                "kind": ["bin"],
                "src_path": "/workspace/kernels/src/device_main.rs"
            }]
        }]
    });

    let error =
        interop_binary_target_from_metadata(&metadata, manifest, "missing-device").unwrap_err();
    assert!(error.contains("no binary target \"missing-device\""));
    assert!(error.contains("available binary targets: main-device"));
}

#[test]
fn release_depfile_stem_preserves_hyphens_like_cargo_uplift() {
    let target_dir = Path::new("/workspace/device/target");
    // Regression: the stem was normalize_crate_name'd (hyphen -> underscore),
    // but cargo uplifts dep-info named after the bin target verbatim, so
    // every hyphenated bin/package with source-identity aborted with
    // "did not produce dependency file".
    assert_eq!(
        release_depfile_path(target_dir, "simt-device"),
        PathBuf::from("/workspace/device/target/release/simt-device.d")
    );
    assert_eq!(
        release_depfile_path(target_dir, "kernels"),
        PathBuf::from("/workspace/device/target/release/kernels.d")
    );
}

/// The load-bearing claim behind `release_depfile_path` is cargo's own
/// uplift naming, so assert it against a real `cargo build` of a
/// hyphenated package with a hyphenated bin target rather than against
/// our expectations of it.
#[test]
fn release_depfile_path_matches_real_cargo_uplift_for_hyphenated_bin() {
    let root = unique_temp_dir("cargo_oxide_hyphen_depfile");
    std::fs::create_dir_all(root.join("src/bin")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "probe-device"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "hyphen-device"
path = "src/bin/hyphen_device.rs"
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/bin/hyphen_device.rs"), "fn main() {}\n").unwrap();
    // Pin the probe's target dir inside the temp root so an ambient
    // CARGO_TARGET_DIR (shared CI caches) cannot collide across tests.
    let target_dir = root.join("target");

    let build = Command::new("cargo")
        .args(["build", "--release", "--bin", "hyphen-device"])
        .env("CARGO_TARGET_DIR", &target_dir)
        .current_dir(&root)
        .output()
        .expect("failed to run cargo build for the depfile probe");
    assert!(
        build.status.success(),
        "depfile probe build failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let metadata = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .env("CARGO_TARGET_DIR", &target_dir)
        .current_dir(&root)
        .output()
        .expect("failed to run cargo metadata for the depfile probe");
    assert!(
        metadata.status.success(),
        "depfile probe metadata failed:\n{}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&metadata.stdout).unwrap();

    let depfile =
        release_depfile_path(&cargo_target_directory(&metadata).unwrap(), "hyphen-device");
    assert!(
        depfile.is_file(),
        "cargo did not uplift the dep-info where we derive it: {}",
        depfile.display()
    );
    // The underscore-normalized twin must NOT be where we look.
    assert!(
        !depfile.with_file_name("hyphen_device.d").exists(),
        "cargo unexpectedly uplifted an underscore-normalized dep-info file"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn ptx_artifact_paths_normalize_hyphenated_example_names() {
    let root = unique_temp_dir("cargo_oxide_inspect_regular");
    std::fs::create_dir_all(&root).unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "demo-app"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    assert_eq!(
        ptx_artifact_paths(&root, "demo-app"),
        vec![root.join("demo_app.ptx")]
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn ptx_artifact_paths_resolve_interop_device_artifacts() {
    let root = unique_temp_dir("cargo_oxide_inspect_interop");
    let device_dir = root.join("device");
    std::fs::create_dir_all(&device_dir).unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "host-app"
version = "0.1.0"
edition = "2024"

[package.metadata.cuda-oxide]
interop = "device"

[[package.metadata.cuda-oxide.device-crates]]
manifest-path = "device/Cargo.toml"
ptx-dir = "generated"
artifact-name = "custom-device"
"#,
    )
    .unwrap();

    std::fs::write(
        device_dir.join("Cargo.toml"),
        r#"[package]
name = "device-app"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    assert_eq!(
        ptx_artifact_paths(&root, "host-app"),
        vec![root.join("generated/custom_device.ptx")]
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn interop_metadata_declares_cubin_and_source_identity() {
    let root = unique_temp_dir("cargo_oxide_cubin_interop");
    let device_dir = root.join("device");
    std::fs::create_dir_all(&device_dir).unwrap();

    std::fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "host-app"
version = "0.1.0"
edition = "2024"

[package.metadata.cuda-oxide]
interop = "device"

[[package.metadata.cuda-oxide.device-crates]]
manifest-path = "device/Cargo.toml"
artifact-dir = "device"
artifact-name = "custom-device"
artifact-kind = "cubin"
source-identity = true
"#,
    )
    .unwrap();
    std::fs::write(
        device_dir.join("Cargo.toml"),
        r#"[package]
name = "device-app"
version = "0.1.0"
edition = "2024"
"#,
    )
    .unwrap();

    let config = load_interop_config(&root).expect("interop metadata should load");
    assert_eq!(config.device_crates.len(), 1);
    let device = &config.device_crates[0];
    assert_eq!(device.artifact_kind, InteropArtifactKind::Cubin);
    assert!(device.source_identity);
    assert_eq!(
        interop_device_artifact_path(&root, device, "custom-device"),
        root.join("device/custom_device.cubin")
    );
    assert_eq!(
        interop_cubin_target(Some("sm_120a"), None).unwrap().sm(),
        "sm_120a"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn recorded_target_sidecar_is_the_completion_marker() {
    let root = unique_temp_dir("cargo_oxide_recorded_target");
    std::fs::create_dir_all(&root).unwrap();
    let ir_path = root.join("device_kernels.ll");

    // No sidecar: the backend never completed its artifact contract.
    let error = read_interop_recorded_target(&ir_path).unwrap_err();
    assert!(error.contains("completion marker"), "{error}");

    // Bare target line (pre-versioned contract) is accepted.
    std::fs::write(root.join("device_kernels.target"), "sm_90a\n").unwrap();
    assert_eq!(read_interop_recorded_target(&ir_path).unwrap(), "sm_90a");

    // The versioned marker says the sibling .options file is required...
    std::fs::write(
        root.join("device_kernels.target"),
        format!(
            "sm_120a\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .unwrap();
    let error = read_interop_recorded_target(&ir_path).unwrap_err();
    assert!(error.contains("compile options"), "{error}");

    // ...and the record is trusted once it exists.
    std::fs::write(root.join("device_kernels.options"), "fma=on\ndebug=none\n").unwrap();
    assert_eq!(read_interop_recorded_target(&ir_path).unwrap(), "sm_120a");

    // Unknown trailing content is rejected, not half-trusted.
    std::fs::write(
        root.join("device_kernels.target"),
        "sm_120a\nmystery-marker\nrest\n",
    )
    .unwrap();
    assert!(read_interop_recorded_target(&ir_path).is_err());
    std::fs::write(root.join("device_kernels.target"), "\n").unwrap();
    assert!(read_interop_recorded_target(&ir_path).is_err());

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn ptx_recorded_target_reads_the_target_directive() {
    let root = unique_temp_dir("cargo_oxide_ptx_target");
    std::fs::create_dir_all(&root).unwrap();
    let ptx_path = root.join("kernels.ptx");

    std::fs::write(
        &ptx_path,
        "// comment\n.version 8.7\n.target sm_120a\n.address_size 64\n",
    )
    .unwrap();
    assert_eq!(ptx_recorded_target(&ptx_path).unwrap(), "sm_120a");

    // Device-debug builds record `.target sm_90, debug`.
    std::fs::write(&ptx_path, ".version 8.3\n.target sm_90, debug\n").unwrap();
    assert_eq!(ptx_recorded_target(&ptx_path).unwrap(), "sm_90");

    std::fs::write(&ptx_path, ".version 8.3\n").unwrap();
    assert!(ptx_recorded_target(&ptx_path).is_err());

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn interop_identity_target_is_read_from_the_emitted_artifact() {
    let root = unique_temp_dir("cargo_oxide_identity_target");
    std::fs::create_dir_all(root.join("device")).unwrap();

    let ptx_crate = DeviceCrateConfig {
        manifest_path: PathBuf::from("device/Cargo.toml"),
        artifact_dir: PathBuf::from("device"),
        artifact_name: Some("kernels".to_string()),
        artifact_kind: InteropArtifactKind::Ptx,
        source_identity: true,
        bin: None,
    };
    // Whatever the request hint said, the emitted PTX is the record.
    std::fs::write(
        root.join("device/kernels.ptx"),
        ".version 8.7\n.target sm_90a\n",
    )
    .unwrap();
    assert_eq!(
        interop_artifact_recorded_target(&root, &ptx_crate, "kernels").unwrap(),
        "sm_90a"
    );

    // Cubin identity reads the backend sidecar the finalizer compiled
    // with, not the PTX and not any hint.
    let cubin_crate = DeviceCrateConfig {
        artifact_kind: InteropArtifactKind::Cubin,
        ..ptx_crate
    };
    std::fs::write(root.join("device/kernels.target"), "sm_100a\n").unwrap();
    assert_eq!(
        interop_artifact_recorded_target(&root, &cubin_crate, "kernels").unwrap(),
        "sm_100a"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn read_ptx_artifact_returns_exact_contents() {
    let root = unique_temp_dir("cargo_oxide_read_ptx");
    std::fs::create_dir_all(&root).unwrap();

    let path = root.join("demo.ptx");
    std::fs::write(&path, ".version 8.0\n.target sm_90\n").unwrap();

    assert_eq!(
        read_ptx_artifact(&path).unwrap(),
        ".version 8.0\n.target sm_90\n"
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn list_json_has_versioned_stable_shape() {
    let examples = vec![ExampleInfo {
        name: "vecadd".to_string(),
        title: "Vector Addition".to_string(),
        description: "Adds two vectors.".to_string(),
        requirements: vec!["Minimum GPU: sm_70+".to_string()],
    }];

    let output = format_examples_json(&examples).unwrap();
    let value: serde_json::Value = serde_json::from_str(&output).unwrap();

    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["examples"][0]["name"], "vecadd");
    assert_eq!(
        value["examples"][0]["requirements"][0],
        "Minimum GPU: sm_70+"
    );
    assert!(output.ends_with('\n'));
}

#[test]
fn read_ptx_artifact_reports_missing_file() {
    let root = unique_temp_dir("cargo_oxide_missing_ptx");
    let path = root.join("missing.ptx");

    let error = read_ptx_artifact(&path).unwrap_err();

    assert!(error.contains("could not read generated PTX"));
    assert!(error.contains("missing.ptx"));
}

#[test]
fn nvvm_ir_requested_reads_project_configuration() {
    let ctx = Context {
        workspace_root: PathBuf::from("/tmp/project"),
        codegen_crate: PathBuf::from("/tmp/project"),
        examples_dir: PathBuf::from("/tmp/project"),
        backend_so: PathBuf::from("/tmp/backend.so"),
        is_workspace: false,
        config: OxideConfig {
            env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "true".to_string())],
            ..OxideConfig::default()
        },
    };

    assert_eq!(nvvm_ir_requested_with_env(&ctx, None), Ok(true));
}

#[test]
fn nvvm_ir_requested_accepts_disabled_project_configuration() {
    let ctx = Context {
        workspace_root: PathBuf::from("/tmp/project"),
        codegen_crate: PathBuf::from("/tmp/project"),
        examples_dir: PathBuf::from("/tmp/project"),
        backend_so: PathBuf::from("/tmp/backend.so"),
        is_workspace: false,
        config: OxideConfig {
            env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "false".to_string())],
            ..OxideConfig::default()
        },
    };

    assert_eq!(nvvm_ir_requested_with_env(&ctx, None), Ok(false));
}

#[test]
fn nvvm_ir_requested_env_disable_overrides_enabled_project_configuration() {
    let ctx = Context {
        workspace_root: PathBuf::from("/tmp/project"),
        codegen_crate: PathBuf::from("/tmp/project"),
        examples_dir: PathBuf::from("/tmp/project"),
        backend_so: PathBuf::from("/tmp/backend.so"),
        is_workspace: false,
        config: OxideConfig {
            env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "true".to_string())],
            ..OxideConfig::default()
        },
    };

    // The process environment outranks `cuda-oxide.toml`: an explicit
    // false in the environment wins over the project's `true`, in either
    // accepted spelling.
    for disabled in ["false", "0"] {
        assert_eq!(
            nvvm_ir_requested_with_env(&ctx, Some(disabled.into())),
            Ok(false)
        );
    }
}

#[test]
fn debug_policy_token_reports_what_the_shared_parser_decides() {
    // `scripts/smoketest.sh` asks for this token to decide whether its
    // optimized code-shape gates apply. The spellings below are the ones a
    // second implementation gets wrong: `2` is an alias, the comparison is
    // case-insensitive, and the value is trimmed with `str::trim`, which
    // takes the Unicode White_Space set rather than the POSIX class.
    for full in [
        "full",
        "2",
        "FULL",
        "Full",
        "  full  ",
        "\u{a0}full",
        "\tfull\n",
    ] {
        assert_eq!(debug_policy_token(Some(full)), "full", "{full:?}");
    }
    for lines in [
        "1",
        "line",
        "lines",
        "line-tables",
        "line-tables-only",
        " Line ",
    ] {
        assert_eq!(debug_policy_token(Some(lines)), "line-tables", "{lines:?}");
    }
    for none in ["0", "off", "none", "OFF"] {
        assert_eq!(debug_policy_token(Some(none)), "none", "{none:?}");
    }
    // An unrecognized value is distinct from an unset one: the first is
    // someone writing a value that means nothing, the second leaves the
    // caller's own default in place. Neither is full debug.
    for other in ["", "   ", "fullx", "full full", "3", "yes", "debug"] {
        assert_eq!(debug_policy_token(Some(other)), "unrecognized", "{other:?}");
    }
    assert_eq!(debug_policy_token(None), "unset");
}

#[test]
fn scaffold_sync_template_uses_launch_contract_and_docs() {
    let files = scaffold_files("demo_kernel", false);
    assert!(files.cargo_toml.contains("name = \"demo_kernel\""));
    assert!(files.readme.contains("cargo oxide doctor"));
    assert!(files.readme.contains("cargo oxide run"));
    assert!(files.gitignore.contains("/target/"));
    // The template uses the launch_bounds / launch_contract attribute
    // macros, so the cuda_device import must bring them in; a scaffolded
    // project fails to compile without these names. The order is rustfmt's,
    // pinned by `scaffold_templates_are_rustfmt_clean`.
    assert!(files.main_rs.contains(
        "use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};"
    ));
    assert!(
        files
            .main_rs
            .contains("#[launch_contract(domain = 1, block = (256, 1, 1))]")
    );
    assert!(files.main_rs.contains("prepare_vecadd"));
    assert!(files.main_rs.contains("LaunchConfig1D"));
    assert!(!files.main_rs.contains("LaunchConfig::for_num_elems"));
    // The host runtime is the crates.io release shared with cutile-rs. A git
    // pin at this repository would resolve to nothing (the local copies are
    // retired) or, worse, to a stale copy on an old revision.
    assert!(
        files
            .cargo_toml
            .contains(&format!("cuda-core = \"{SHARED_HOST_CRATES_VERSION}\""))
    );
    assert!(!files.cargo_toml.contains("cuda-core = {"));
}

#[test]
fn scaffold_async_template_keeps_async_deps_and_docs() {
    let files = scaffold_files("async_demo", true);
    assert!(files.cargo_toml.contains("cuda-async"));
    assert!(files.cargo_toml.contains("tokio"));
    assert!(files.readme.contains("async cuda-oxide"));
    assert!(files.readme.contains("cargo oxide doctor"));
    // The async README must stand alone: it describes the async launch
    // path and never talks about "the sync template".
    assert!(files.readme.contains("DeviceOperation"));
    assert!(!files.readme.contains("sync template"));
    assert!(files.gitignore.contains("**/*.ptx"));
    assert!(files.main_rs.contains("vecadd_async"));
    assert!(files.main_rs.contains("use cuda_host::cuda_module;"));
    assert!(!files.main_rs.contains("use cuda_device::{cuda_module"));
    // Shared host runtime from crates.io; the SIMT surface lives under simt::.
    assert!(
        files
            .cargo_toml
            .contains(&format!("cuda-async = \"{SHARED_HOST_CRATES_VERSION}\""))
    );
    assert!(!files.cargo_toml.contains("cuda-async = {"));
    assert!(!files.cargo_toml.contains("cuda-bindings"));
    assert!(files.main_rs.contains("use cuda_core::simt::LaunchConfig;"));
    assert!(
        files
            .main_rs
            .contains("use cuda_async::simt::device_operation::DeviceOperation;")
    );
}

/// A scaffolded project must already be `rustfmt`-clean.
///
/// CONTRIBUTING points contributors at `cargo oxide fmt`, so a template that is
/// not canonical makes a brand-new project fail its very first format check on
/// code the user never wrote.
///
/// This runs the real `rustfmt` over the rendered template instead of
/// re-deriving its import ordering here. A hand-written ordering check would
/// agree with a stale template as readily as with a correct one, and would not
/// notice drift anywhere else in the file --
/// `release_depfile_path_matches_real_cargo_uplift_for_hyphenated_bin` shells
/// out to real `cargo` for the same reason.
///
/// The edition is read back out of the scaffolded `Cargo.toml` so this cannot
/// drift from the manifest it is meant to describe.
#[test]
fn scaffold_templates_are_rustfmt_clean() {
    for (async_mode, label) in [(false, "sync"), (true, "async")] {
        let files = scaffold_files("demo_kernel", async_mode);

        let edition = files
            .cargo_toml
            .lines()
            .find_map(|line| line.trim().strip_prefix("edition = "))
            .map(|value| value.trim().trim_matches('"').to_string())
            .unwrap_or_else(|| panic!("{label} scaffold Cargo.toml declares no edition"));

        let root = unique_temp_dir(&format!("cargo_oxide_scaffold_fmt_{label}"));
        std::fs::create_dir_all(&root).unwrap();
        let main_rs = root.join("main.rs");
        std::fs::write(&main_rs, &files.main_rs).unwrap();

        // A scaffolded project carries no rustfmt.toml, so plain defaults for
        // its edition are exactly what `cargo fmt` applies there.
        let check = Command::new("rustfmt")
            .args(["--edition", edition.as_str(), "--check"])
            .arg(&main_rs)
            .output()
            .expect("failed to run rustfmt; it is in rust-toolchain.toml's component list");

        let diff = String::from_utf8_lossy(&check.stdout).into_owned();
        std::fs::remove_dir_all(&root).unwrap();

        assert!(
            check.status.success() && diff.is_empty(),
            "the {label} scaffold template is not rustfmt-clean, so `cargo oxide fmt \
             --check` fails in a freshly scaffolded project:\n{diff}"
        );
    }
}

#[test]
fn scaffold_gitignore_covers_every_clean_artifact_suffix() {
    let gitignore = scaffold_gitignore();
    assert!(gitignore.contains("/target/"));
    assert!(gitignore.contains("**/*.bc"));
    for suffix in GENERATED_ARTIFACT_SUFFIXES {
        // Match whole lines, not substrings: `**/*.cubin.target` contains
        // `**/*.cubin` as a substring, so `contains()` would keep passing
        // even if the `cubin` pattern itself were dropped.
        let pattern = format!("**/*.{suffix}");
        assert!(
            gitignore.lines().any(|line| line == pattern),
            "scaffold .gitignore must ignore clean suffix `{suffix}`"
        );
    }
}

#[test]
fn device_debug_env_value_matches_the_backend_parser() {
    // The exported strings must round-trip through the shared
    // `CUDA_OXIDE_DEBUG` parser (the same one the codegen backend uses);
    // a typo would silently fall through to the profile-derived default
    // instead of failing, so check the actual parse.
    assert_eq!(DeviceDebug::Off.env_value(), None);
    assert_eq!(DeviceDebug::LineTables.env_value(), Some("line"));
    assert_eq!(DeviceDebug::Full.env_value(), Some("full"));
    assert_eq!(
        cuda_artifact_finalizer::DebugPolicy::parse_env_override("line"),
        Some(cuda_artifact_finalizer::DebugPolicy::LineTables)
    );
    assert_eq!(
        cuda_artifact_finalizer::DebugPolicy::parse_env_override("full"),
        Some(cuda_artifact_finalizer::DebugPolicy::Full)
    );
}

#[test]
fn passthrough_fingerprint_separates_the_device_debug_policies() {
    let ctx = test_context(OxideConfig::default());
    let base = CargoPassthroughOptions {
        verbose: false,
        emit_nvvm_ir: false,
        arch: None,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad: false,
        unchecked_indexing: false,
        materialize_cubin: false,
        device_debug: DeviceDebug::Off,
        debug_assertions: false,
    };
    let line_tables = CargoPassthroughOptions {
        device_debug: DeviceDebug::LineTables,
        ..base
    };
    let full = CargoPassthroughOptions {
        device_debug: DeviceDebug::Full,
        ..base
    };
    let materialization = MaterializationMode::default();
    // Empty inherited env, for the same reason as the sibling fingerprint
    // tests: an ambient CUDA_OXIDE_DEBUG is folded in on its own, which would
    // collapse these onto the base.
    let inherited_env = BTreeMap::new();
    let fp = |opts: &CargoPassthroughOptions<'_>| {
        passthrough_codegen_fingerprint_with_env(
            &ctx,
            opts,
            None,
            None,
            &materialization,
            &inherited_env,
        )
    };
    // The policy changes what libNVVM and nvJitLink are asked to do (`-g`,
    // `-opt=0`, `-lineinfo`), so it must not share a fingerprint with the
    // default -- otherwise Cargo reuses artifacts built without it.
    let off = fp(&base);
    assert_ne!(off, fp(&line_tables));
    assert_ne!(off, fp(&full));
    assert_ne!(fp(&line_tables), fp(&full));
}

/// Full debug excludes only SROA and `SingleUseConsts`.
/// `ReferencePropagation` and MIR inlining stay enabled; a private cfg outlines
/// only `DisjointSlice::get_mut` to keep CUDA-GDB frame ranges accurate.
#[test]
fn full_device_debug_applies_only_the_targeted_debug_controls() {
    let cmd = Command::new("cargo");
    let mut encoded = "base".to_string();

    append_full_debug_rustflags(&mut encoded, &cmd, Some("full"));

    assert_eq!(
        decoded_rustflags(&encoded),
        [
            "base",
            "--cfg",
            FULL_DEBUG_GET_MUT_OUTLINE_CFG,
            "-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts"
        ]
    );
    assert!(!encoded.contains("mir-opt-level"), "{encoded}");
    assert!(!encoded.contains("inline-mir"), "{encoded}");
}

#[test]
fn numeric_full_debug_alias_selects_the_same_rustc_controls() {
    // The shared parser makes `CUDA_OXIDE_DEBUG=2` select the same rustc
    // controls as `full`.
    let mut cmd = Command::new("cargo");
    cmd.env("CUDA_OXIDE_DEBUG", "2");
    let mut encoded = "base".to_string();

    append_full_debug_rustflags(&mut encoded, &cmd, None);

    assert_eq!(
        decoded_rustflags(&encoded),
        [
            "base",
            "--cfg",
            FULL_DEBUG_GET_MUT_OUTLINE_CFG,
            "-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts"
        ]
    );
}

#[test]
fn line_tables_keep_normal_mir_optimization() {
    let mut cmd = Command::new("cargo");
    cmd.env("CUDA_OXIDE_DEBUG", "line");
    let mut encoded = "base".to_string();

    append_full_debug_rustflags(&mut encoded, &cmd, None);

    assert_eq!(decoded_rustflags(&encoded), ["base"]);
}

#[test]
fn explicit_line_tables_override_inherited_full_debug_for_mir_optimization() {
    let mut cmd = Command::new("cargo");
    cmd.env("CUDA_OXIDE_DEBUG", "line");
    let mut encoded = "base".to_string();

    append_full_debug_rustflags(&mut encoded, &cmd, Some("full"));

    assert_eq!(decoded_rustflags(&encoded), ["base"]);
}

#[test]
fn non_full_codegen_profiles_do_not_activate_the_targeted_debug_cfg() {
    for profile in [
        CodegenProfilePolicy::CargoSelected,
        CodegenProfilePolicy::ReleaseLike,
        CodegenProfilePolicy::ReleaseLikeWithDebugInfo,
    ] {
        for (debug, inherited) in [
            (None, None),
            (Some("off"), Some("full")),
            (Some("line"), Some("full")),
        ] {
            let mut encoded = build_encoded_rustflags_with_existing(
                Path::new("/tmp/librustc_codegen_cuda.so"),
                profile,
                &[],
                &[],
                None,
                None,
            );
            let mut cmd = Command::new("cargo");
            if let Some(debug) = debug {
                cmd.env("CUDA_OXIDE_DEBUG", debug);
            }

            append_full_debug_rustflags(&mut encoded, &cmd, inherited);

            let flags = decoded_rustflags(&encoded);
            assert!(
                !has_full_debug_cfg(&flags),
                "{profile:?}/{debug:?}: {flags:?}"
            );
            assert!(
                !flags
                    .iter()
                    .any(|flag| FULL_DEBUG_MIR_RUSTFLAGS.contains(flag)),
                "{profile:?}/{debug:?}: {flags:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// doctor: "Backend source (cuda-oxide commit)" line
// ---------------------------------------------------------------------------

/// Every branch of the backend-source line, from the resolved dependency and
/// the commit recorded in the shared cache. The line is informational, so no
/// branch may flip doctor's exit status.
#[test]
fn doctor_backend_source_check_covers_every_branch() {
    use crate::backend_source::DependencySource;
    let sha_a = "a1b4f11882592fae9d022c86a9d8d1a4c9426980";
    let sha_b = "596a6353de480662bbc03cb2d3c0f92e88f1e6c6";
    let git = |rev: &str| {
        Some(DependencySource::Git {
            checkout: PathBuf::from("/checkouts/x"),
            rev: rev.to_string(),
        })
    };

    let check = backend_source_check(
        Err("`cargo metadata` failed (exit status: 101)".into()),
        None,
    );
    assert!(
        check
            .headline
            .starts_with("- could not resolve the cuda-oxide dependency offline ("),
        "{}",
        check.headline
    );
    assert!(
        check.headline.contains("exit status: 101"),
        "{}",
        check.headline
    );

    let check = backend_source_check(Ok(None), Some(sha_a.into()));
    assert!(
        check
            .headline
            .starts_with("- no cuda-device/cuda-host dependency"),
        "{}",
        check.headline
    );

    let check = backend_source_check(
        Ok(Some(DependencySource::Path {
            checkout: PathBuf::from("/dev/cuda-oxide"),
        })),
        None,
    );
    assert_eq!(
        check.headline,
        "✓ cuda-oxide checkout /dev/cuda-oxide (path dependency) builds in place"
    );

    let check = backend_source_check(Ok(git(sha_a)), Some(sha_a.into()));
    assert_eq!(
        check.headline,
        "✓ cache built from a1b4f11882, matching this project's dependency"
    );
    assert!(check.details.is_empty());

    // The mismatch line must put the CACHE's commit first and the PROJECT's
    // second; swapped, it would send the user chasing the wrong commit.
    let check = backend_source_check(Ok(git(sha_a)), Some(sha_b.into()));
    assert_eq!(
        check.headline,
        "⚠ cache built from 596a6353de but this project depends on a1b4f11882"
    );
    assert_eq!(check.details.len(), 1, "{:?}", check.details);
    assert!(
        check.details[0].contains("rebuilds the cache"),
        "{:?}",
        check.details
    );

    let check = backend_source_check(Ok(git(sha_a)), None);
    assert!(
        check.headline.starts_with("⚠ cache records no commit"),
        "{}",
        check.headline
    );
    assert!(check.headline.ends_with("a1b4f11882"), "{}", check.headline);
    assert_eq!(check.details.len(), 1, "{:?}", check.details);

    for source in [Err("x".to_string()), Ok(None), Ok(git(sha_a))] {
        assert!(
            !backend_source_check(source, Some(sha_b.into())).failed,
            "the backend-source line is informational and must never fail doctor"
        );
    }
}
