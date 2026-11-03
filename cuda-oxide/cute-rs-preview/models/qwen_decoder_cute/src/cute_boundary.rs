/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Keep hardware instructions below the Qwen cute-rs kernel boundary.
//!
//! This guard rejects direct instruction calls/imports. Layout expressiveness,
//! correct composition and safety still require code review and numerical tests.

/// Named source snapshots checked for instruction calls outside cute-rs abstractions.
const KERNELS: &[(&str, &str)] = &[
    (
        "projections",
        include_str!("../../../examples/ffn_cute/src/kernel.rs"),
    ),
    (
        "normalization",
        include_str!("../../../examples/common/rmsnorm.rs"),
    ),
    (
        "rotary/cache",
        include_str!("../../../examples/rope_cache_cute/src/kernel.rs"),
    ),
    (
        "attention",
        include_str!("../../../examples/rope_cache_cute/src/attention_kernel.rs"),
    ),
    ("codec", include_str!("../src/codec_kernel.rs")),
    ("codec attention", include_str!("../src/codec_attention.rs")),
    ("elementwise", include_str!("../src/element_kernel.rs")),
    (
        "predictor",
        include_str!("../src/predictor_graph_kernel.rs"),
    ),
    (
        "reference linear",
        include_str!("../src/reference_linear_kernel.rs"),
    ),
];

/// Return the first forbidden CTM import or direct-instruction spelling.
///
/// This lexical guard strips whole-line comments and allows only kernel ABI,
/// thread-index and storage imports. It is not a Rust parser or safety proof.
fn boundary_error(source: &str) -> Option<String> {
    let mut source = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    while let Some(start) = source.find("use ctm_rs::") {
        let end = start + source[start..].find(';')? + 1;
        for token in source[start..end]
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
        {
            if !matches!(
                token,
                "use"
                    | "ctm_rs"
                    | "memory"
                    | "DisjointSlice"
                    | "DynamicSharedArray"
                    | "cuda_module"
                    | "kernel"
                    | "launch_bounds"
                    | "launch_contract"
                    | "thread"
            ) {
                return Some(format!("instruction import {token} belongs below cute-rs"));
            }
        }
        source.replace_range(start..end, "");
    }
    for forbidden in [
        "ctm_rs::",
        "cuda_device::",
        "acc_coords(",
        "::from_registers(",
    ] {
        if source.contains(forbidden) {
            return Some(format!(
                "direct hardware operation {forbidden} belongs below cute-rs"
            ));
        }
    }
    None
}

/// Require every listed workload kernel source to pass the abstraction boundary guard.
#[test]
fn qwen_kernels_keep_instructions_below_cute_rs() {
    let errors: Vec<_> = KERNELS
        .iter()
        .filter_map(|(name, source)| boundary_error(source).map(|error| format!("{name}: {error}")))
        .collect();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// Exercise allowed ABI imports and rejected direct or imported hardware operations.
#[test]
fn boundary_guard_rejects_direct_and_imported_instructions() {
    assert!(boundary_error("use ctm_rs::{cuda_module, kernel, thread};").is_none());
    assert!(boundary_error("use ctm_rs::memory::DisjointSlice;").is_none());
    for code in [
        "use ctm_rs::mma::{Bf16A, mma_bf16};",
        "use ctm_rs::{kernel, float, thread};",
        "ctm_rs::warp::shuffle_xor_f32(value, 1)",
        "cuda_device::float::ex2_approx_f32(value)",
        "Bf16A::from_registers([0; 4])",
    ] {
        assert!(boundary_error(code).is_some(), "accepted {code}");
    }
}
