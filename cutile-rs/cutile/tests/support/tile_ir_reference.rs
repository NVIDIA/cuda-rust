/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional independent verification with CUDA Tile IR's C++ bytecode reader/verifier.
pub fn verify_if_configured(bytes: &[u8]) {
    let Some(translator) = std::env::var_os("CUTILE_REFERENCE_TRANSLATE") else {
        return;
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("cutile-reference-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let input = dir.join("kernel.bc");
    let output = dir.join("kernel.mlir");
    std::fs::write(&input, bytes).unwrap();
    let result = std::process::Command::new(translator)
        .arg("--cudatilebc-to-mlir")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("launch CUTILE_REFERENCE_TRANSLATE");
    assert!(
        result.status.success(),
        "C++ Tile IR reader/verifier rejected {}: {}\n{}",
        input.display(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(std::fs::read_to_string(output)
        .unwrap()
        .contains("cuda_tile.module"));
    std::fs::remove_dir_all(dir).unwrap();
}
