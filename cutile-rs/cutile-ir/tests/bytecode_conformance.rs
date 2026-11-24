/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reader conformance against NVIDIA's reference bytecode, read from the
//! pinned `cuda-tile` submodule.

use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion};
use cutile_ir::read_bytecode_versioned;

fn fixture_root() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cuda-tile-rs/cuda-tile/test/Bytecode");
    p.is_dir().then_some(p)
}

fn fixtures() -> Vec<(String, std::path::PathBuf)> {
    let Some(root) = fixture_root() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read fixture dir") {
            let path = entry.expect("fixture entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "tileirbc") {
                let rel = path
                    .strip_prefix(&root)
                    .expect("fixture under root")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((rel, path));
            }
        }
    }
    out.sort();
    out
}

/// Skipping and passing look identical in a log. Local runs skip. CI must have the fixtures.
fn skip(reason: &str) {
    let required = std::env::var_os("CI").is_some()
        || std::env::var_os("CUTILE_REQUIRE_REFERENCE_FIXTURES").is_some();
    assert!(
        !required,
        "{reason}: cuda-tile submodule is not checked out; run \
         `git submodule update --init cutile-rs/cuda-tile-rs/cuda-tile`"
    );
    eprintln!("skipping {reason}: cuda-tile submodule is not checked out");
}

/// Fixtures expected to fail. Typical reasons for being in this list is:
/// * Bytecode version < `BytecodeVersion::MIN_SUPPORTED`.
/// * Header declares older version than features used.
const EXPECTED_READER_ERRORS: &[&str] = &[
    "versioning/Inputs/13.1/f4e2m1fn-in-13.1.tileirbc",
    "versioning/Inputs/13.4/unicast-ptr-attr-in-13.4.tileirbc",
];

fn expected_to_fail(rel: &str) -> bool {
    rel.starts_with("invalid/")
        || rel.starts_with("versioning/Inputs/13.1/")
        || EXPECTED_READER_ERRORS.contains(&rel)
}

#[test]
fn every_valid_reference_fixture_decodes_and_is_idempotent() {
    let all = fixtures();
    if all.is_empty() {
        return skip("reference decode");
    }
    let mut checked = 0;
    for (rel, path) in &all {
        if expected_to_fail(rel) {
            continue;
        }
        let data = std::fs::read(path).expect("read fixture");
        let (module, version) =
            read_bytecode_versioned(&data).unwrap_or_else(|e| panic!("{rel}: decode failed: {e}"));
        let first = write_bytecode_version(&module, version)
            .unwrap_or_else(|e| panic!("{rel}: re-encode failed: {e}"));
        let (again, version2) = read_bytecode_versioned(&first)
            .unwrap_or_else(|e| panic!("{rel}: re-decode of our own bytes failed: {e}"));
        assert_eq!(version2, version, "{rel}: version changed across re-encode");
        let second = write_bytecode_version(&again, version2)
            .unwrap_or_else(|e| panic!("{rel}: second re-encode failed: {e}"));
        assert_eq!(
            first,
            second,
            "{rel}: decode/re-encode is not idempotent ({} then {} bytes)",
            first.len(),
            second.len()
        );
        checked += 1;
    }
    assert!(
        checked >= 20,
        "expected at least 20 valid fixtures, got {checked}"
    );
}

#[test]
fn no_reference_fixture_panics_or_decodes_unstably() {
    let all = fixtures();
    if all.is_empty() {
        return skip("reference stability");
    }
    for (rel, path) in &all {
        let data = std::fs::read(path).expect("read fixture");
        let Ok((module, version)) = read_bytecode_versioned(&data) else {
            continue;
        };
        let Ok(first) = write_bytecode_version(&module, version) else {
            continue;
        };
        let Ok((again, version2)) = read_bytecode_versioned(&first) else {
            panic!("{rel}: our own re-encode does not read back");
        };
        let second = write_bytecode_version(&again, version2)
            .unwrap_or_else(|e| panic!("{rel}: second re-encode failed: {e}"));
        assert_eq!(first, second, "{rel}: decode/re-encode is not idempotent");
    }
}

#[test]
fn malformed_reference_fixtures_are_rejected() {
    let all = fixtures();
    if all.is_empty() {
        return skip("reference rejection");
    }
    let mut checked = 0;
    for (rel, path) in &all {
        if !rel.starts_with("invalid/") {
            continue;
        }
        let data = std::fs::read(path).expect("read fixture");
        assert!(
            read_bytecode_versioned(&data).is_err(),
            "{rel}: malformed bytecode was accepted"
        );
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected at least 6 invalid fixtures, got {checked}"
    );
}

#[test]
fn version_13_1_is_out_of_the_supported_range() {
    let all = fixtures();
    if all.is_empty() {
        return skip("13.1 range");
    }
    assert_eq!(BytecodeVersion::MIN_SUPPORTED, BytecodeVersion::V13_2);
    for (rel, path) in &all {
        if !rel.starts_with("versioning/Inputs/13.1/") {
            continue;
        }
        let data = std::fs::read(path).expect("read fixture");
        let err = read_bytecode_versioned(&data)
            .err()
            .unwrap_or_else(|| panic!("{rel}: 13.1 bytecode decoded unexpectedly"))
            .to_string();
        assert!(err.contains("13.1"), "{rel}: unexpected diagnostic: {err}");
    }
}

#[test]
fn i1_dense_constants_decode_bit_packed() {
    let Some(root) = fixture_root() else {
        return skip("i1 payloads");
    };
    let data = std::fs::read(root.join("versioning/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc"))
        .expect("read fixture");
    let (module, _) = read_bytecode_versioned(&data).expect("decode");
    let text = module.to_mlir_text();
    for expected in [
        "<i1: true, false, true, false>",
        "<i1: false, false, true, true>",
        "<i1: true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true>",
    ] {
        assert!(text.contains(expected), "missing {expected} in:\n{text}");
    }
}
