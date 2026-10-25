// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordinary Cargo freshness across a dependency registration and its consumer.
//! The fake provider is a shell process; no CUDA toolkit or GPU is used.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn build(root: &Path, materialize: bool, option: Option<&str>, expected: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO"));
    command
        .args(["build", "-vv", "--offline"])
        .current_dir(root)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("CUDA_HOME", root.join("toolkit"))
        .env("CUDA_OXIDE_TARGET", "sm_120")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTFLAGS");
    if let Some(option) = option {
        command.env("NATIVE_INPUT_OPTION", option);
    } else {
        command.env_remove("NATIVE_INPUT_OPTION");
    }
    if materialize {
        command.env("CUDA_OXIDE_MATERIALIZE_CUBIN", "1");
    } else {
        command.env_remove("CUDA_OXIDE_MATERIALIZE_CUBIN");
    }
    let output = command.output().unwrap();
    let log = String::from_utf8_lossy(&output.stderr).into_owned();
    if let Some(error) = expected {
        assert!(!output.status.success(), "{log}");
        assert!(log.contains(error), "{log}");
    } else {
        assert!(output.status.success(), "{log}");
    }
    log
}

fn provider(root: &Path, response: &str) {
    fs::write(
        root.join("provider"),
        format!("#!/bin/sh\ncat >/dev/null\nprintf '%s' '{response}'\n"),
    )
    .unwrap();
    fs::set_permissions(root.join("provider"), fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn tracks_native_inputs_before_cargo_dep_info() {
    let root = std::env::temp_dir().join(format!(
        "oxide-library-inputs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let device = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cuda-device")
        .canonicalize()
        .unwrap();
    fs::create_dir_all(root.join("api/src")).unwrap();
    fs::create_dir_all(root.join("consumer/src")).unwrap();
    fs::create_dir_all(root.join("headers")).unwrap();
    fs::write(root.join("headers/value.h"), "initial").unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers=['api','consumer']\ndefault-members=['consumer']\nresolver='3'\n",
    )
    .unwrap();
    fs::write(root.join("api/Cargo.toml"), format!(
        "[package]\nname='native-input-api'\nversion='0.0.0'\nedition='2024'\n[dependencies]\ncuda-device={{path={:?}}}\n", device.to_str().unwrap()
    )).unwrap();
    fs::write(root.join("consumer/Cargo.toml"),
        "[package]\nname='native-input-consumer'\nversion='0.0.0'\nedition='2024'\n[dependencies]\nnative-input-api={path='../api'}\n").unwrap();
    let registration = serde_json::json!({
        "version":1,"provider":"fixture","executable":root.join("provider"),
        "operation":"test","algorithm":"direct"
    })
    .to_string();
    fs::write(root.join("api/build.rs"), format!(
        "fn main() {{ println!(\"cargo::rerun-if-changed=build.rs\"); println!(\"cargo::rustc-env=REG={{}}\", {registration:?}); }}"
    )).unwrap();
    fs::write(
        root.join("api/src/lib.rs"),
        "pub fn registration() -> &'static str { cuda_device::library::registration!(\"REG\") }\n",
    )
    .unwrap();
    fs::write(
        root.join("consumer/src/main.rs"),
        "fn main() { assert!(native_input_api::registration().contains(\"fixture\")); }\n",
    )
    .unwrap();
    let response = serde_json::json!({
        "version":1,"paths":[root.join("headers")],"environment":["NATIVE_INPUT_OPTION"]
    })
    .to_string();
    provider(&root, &response);
    build(&root, true, None, None);
    let warm = build(&root, true, None, None);
    assert!(warm.contains("Fresh native-input-api"), "{warm}");
    assert!(warm.contains("Fresh native-input-consumer"), "{warm}");

    fs::write(root.join("headers/value.h"), "changed").unwrap();
    let changed = build(&root, true, None, None);
    assert!(changed.contains("Compiling native-input-api"), "{changed}");
    assert!(
        changed.contains("Compiling native-input-consumer"),
        "{changed}"
    );
    fs::write(root.join("headers/new.h"), "added").unwrap();
    let added = build(&root, true, None, None);
    assert!(added.contains("Compiling native-input-api"), "{added}");

    let option = build(&root, true, Some("changed"), None);
    assert!(option.contains("Compiling native-input-api"), "{option}");
    assert!(
        option.contains("Compiling native-input-consumer"),
        "{option}"
    );
    let warm = build(&root, true, Some("changed"), None);
    assert!(warm.contains("Fresh native-input-api"), "{warm}");

    provider(&root, r#"{"version":2,"paths":[],"environment":[]}"#);
    build(
        &root,
        true,
        Some("changed"),
        Some("input response requires version 1"),
    );
    provider(
        &root,
        r#"{"version":1,"paths":["relative"],"environment":[]}"#,
    );
    build(
        &root,
        true,
        Some("changed"),
        Some("input paths must be absolute"),
    );
    provider(&root, "malformed");
    build(
        &root,
        true,
        Some("changed"),
        Some("invalid native library input response"),
    );
    // A normal host build expands the registration without querying CUDA or
    // attempting to consume the deliberately invalid native response.
    build(&root, false, None, None);
    fs::remove_dir_all(root).unwrap();
}
