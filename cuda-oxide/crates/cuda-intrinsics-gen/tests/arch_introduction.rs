/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check(root: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cuda-intrinsics-gen"))
        .arg("check")
        .arg("--repo-root")
        .arg(root)
        .output()
        .expect("run cuda-intrinsics-gen check")
}

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn copy_inputs(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_inputs(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn check_rejects_ptx_1_0_on_sm_80() {
    let fixture = Fixture(std::env::temp_dir().join(format!(
        "cuda-intrinsics-arch-introduction-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    )));
    copy_inputs(
        &repo_root().join("intrinsics"),
        &fixture.0.join("intrinsics"),
    );
    let path = fixture.0.join("intrinsics/overlay/redux.toml");
    let original = fs::read_to_string(&path).unwrap();
    assert!(original.contains("minimum_sm = \"sm_80\""));
    let mutated = original.replacen("minimum_ptx = \"7.0\"", "minimum_ptx = \"1.0\"", 1);
    assert_ne!(original, mutated);
    fs::write(path, mutated).unwrap();

    let output = check(&fixture.0);
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{diagnostic}");
    assert!(
        diagnostic
            .contains("minimum_ptx 1.0 is below PTX 7.0, which first named hardware floor sm_80"),
        "expected the introduction check, not a recipe or stale-output failure: {diagnostic}",
    );
}

#[test]
fn check_accepts_authored_floors_and_native_hardware_control() {
    // Includes sm_80 at PTX 7.0, later instructions above that floor, packed
    // f16x2 with native sm_53, and libNVVM profile floors below sm_75's 6.3.
    let output = check(&repo_root());
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
