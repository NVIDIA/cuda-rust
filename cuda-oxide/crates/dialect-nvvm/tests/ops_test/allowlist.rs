/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[test]
fn handwritten_ops_match_reviewed_allowlist() {
    use std::{fs, path::Path};

    let ops_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ops");
    let mut found = Vec::new();

    for entry in fs::read_dir(&ops_dir).expect("read top-level NVVM ops directory") {
        let path = entry.expect("read NVVM ops entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("read handwritten NVVM ops source");
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("NVVM ops file name");

        for (marker, _) in source.match_indices("#[pliron_op(") {
            let declaration = source[marker..]
                .lines()
                .find_map(|line| line.trim().strip_prefix("pub struct "))
                .expect("pliron_op must be followed by a public struct");
            let name = declaration
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .next()
                .expect("NVVM op struct name");
            found.push((file.to_owned(), name.to_owned()));
        }
    }

    let mut expected = [
        ("asm.rs", "InlinePtxOp"),
        ("atomic.rs", "NvvmAtomicLoadOp"),
        ("atomic.rs", "NvvmAtomicStoreOp"),
        ("atomic.rs", "NvvmAtomicFenceOp"),
        ("atomic.rs", "NvvmAtomicRmwOp"),
        ("atomic.rs", "NvvmAtomicCmpxchgOp"),
        ("cluster.rs", "ReadPtxSregClusterIdxOp"),
        ("cluster.rs", "ReadPtxSregNclusterIdOp"),
        ("debug.rs", "AssertFailOp"),
        ("debug.rs", "VprintfOp"),
        ("grid.rs", "GridSyncOp"),
        ("memory.rs", "CvtaGenericToSharedOffsetOp"),
        ("wgmma.rs", "WgmmaMakeSmemDescOp"),
        ("wgmma.rs", "WgmmaMmaM64N64K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaM64N128K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaM64N64K16F32F16Op"),
        ("wgmma.rs", "WgmmaMmaM64N64K8F32Tf32Op"),
        ("wgmma.rs", "WgmmaMmaGroupM64N64K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaGroupValuesM64N64K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaGroupValuesM64N128K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaGroupValuesM64N64K16F32F16Op"),
        ("wgmma.rs", "WgmmaMmaGroupValuesM64N64K8F32Tf32Op"),
        ("wgmma.rs", "WgmmaMmaLoopValuesM64N64K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaLoopValuesM64N64K16F32F16Op"),
        ("wgmma.rs", "WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op"),
        ("wgmma.rs", "WgmmaMmaPipelineValuesM64N64K16F32Bf16Op"),
    ];
    expected.sort_unstable();
    found.sort_unstable();

    assert_eq!(
        found,
        expected.map(|(file, op)| (file.to_owned(), op.to_owned())),
        "top-level handwritten NVVM ops changed; generate leaf ops or review this allowlist"
    );
}
