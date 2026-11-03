/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for in-memory cache key correctness.
//!
//! These are CPU-only (no GPU required) and assert that [`TileFunctionKey`]
//! distinguishes every input that can change the generated GPU code, so kernels
//! are never falsely reused.
//!
//! The key *is* the cache identity: `KERNEL_CACHE` is a `DashMap` keyed on the
//! whole struct. `display_hash()` is a 64-bit digest for log lines only, so these
//! tests compare keys, not digests.

use cutile::tile_kernel::{CompileOptions, FunctionKey, TileFunctionKey};
use cutile_compiler::specialization::{DivHint, SpecializationBits};

fn default_key() -> cutile::tile_kernel::TileFunctionKeyBuilder {
    TileFunctionKey::builder("m", "f")
        .source_hash("hash")
        .gpu_name("sm_90")
        .compiler_version("0.0.1")
        .tileiras_fingerprint("release 13.3, V13.3.36")
}

// TileFunctionKey identity properties
#[test]
fn cache_key_deterministic() {
    let key1 = TileFunctionKey::builder("mod", "fn")
        .generics(vec!["f32".into()])
        .source_hash("abc123")
        .gpu_name("sm_90")
        .compiler_version("0.0.1-alpha")
        .tileiras_fingerprint("release 13.3, V13.3.36")
        .build();
    let key2 = key1.clone();
    assert_eq!(key1, key2);
    assert_eq!(key1.display_hash(), key2.display_hash());
}

#[test]
fn cache_key_different_source_hash() {
    let key_a = default_key().source_hash("hash_v1").build();
    let key_b = default_key().source_hash("hash_v2").build();
    assert_ne!(key_a, key_b);
}

#[test]
fn cache_key_different_gpu_name() {
    let key_a = default_key().gpu_name("sm_80").build();
    let key_b = default_key().gpu_name("sm_90").build();
    assert_ne!(key_a, key_b);
}

#[test]
fn cache_key_different_device_id() {
    let key_a = default_key().device_id(0).build();
    let key_b = default_key().device_id(1).build();
    assert_ne!(
        key_a, key_b,
        "same kernel on different devices must not share a cache entry"
    );
}

#[test]
fn cache_key_different_compiler_version() {
    let key_a = default_key().compiler_version("0.0.1").build();
    let key_b = default_key().compiler_version("0.0.2").build();
    assert_ne!(key_a, key_b);
}

// The key names the resolved `tileiras`, so pointing `CUTILE_TILEIRAS_PATH`
// at a different binary must not reuse the previous binary's cubin.
#[test]
fn cache_key_different_tileiras_fingerprint() {
    let key_a = default_key()
        .tileiras_fingerprint("release 13.2, V13.2.55")
        .build();
    let key_b = default_key()
        .tileiras_fingerprint("release 13.3, V13.3.36")
        .build();
    assert_ne!(key_a, key_b);
}

// `tileiras_fingerprint()` falls back to `stat\0<path>\0<len>\0<mtime>` when
// `--version` fails. That form must not collide with any version string.
#[test]
fn cache_key_tileiras_stat_fallback_is_distinct() {
    let key_stat = default_key()
        .tileiras_fingerprint("stat\x00/usr/local/cuda/bin/tileiras\x0094855128\x00170000000")
        .build();
    let key_version = default_key()
        .tileiras_fingerprint("release 13.3, V13.3.36")
        .build();
    assert_ne!(key_stat, key_version);
}

#[test]
fn cache_key_different_generics() {
    let key_a = default_key().generics(vec!["f32".into()]).build();
    let key_b = default_key().generics(vec!["f16".into()]).build();
    assert_ne!(key_a, key_b);
}

// Cache keys must distinguish data alignments to prevent incorrect kernel reuse.
#[test]
fn cache_key_different_spec_args() {
    let spec_aligned = SpecializationBits {
        shape_div: vec![DivHint::from_value(16), DivHint::from_value(16)],
        stride_div: vec![DivHint::from_value(16), DivHint::from_value(16)],
        stride_one: vec![false, true],
        base_ptr_div: DivHint::from_ptr(16),
        elements_disjoint: true,
    };
    let spec_misaligned = SpecializationBits {
        shape_div: vec![DivHint::from_value(4), DivHint::from_value(4)],
        stride_div: vec![DivHint::from_value(4), DivHint::from_value(4)],
        stride_one: vec![false, true],
        base_ptr_div: DivHint::from_ptr(4),
        elements_disjoint: true,
    };
    let key_a = default_key()
        .spec_args(vec![("x".into(), spec_aligned)])
        .build();
    let key_b = default_key()
        .spec_args(vec![("x".into(), spec_misaligned)])
        .build();
    assert_ne!(
        key_a, key_b,
        "different SpecializationBits must produce distinct memory keys"
    );
}

#[test]
fn cache_key_different_compile_options() {
    let key_a = default_key()
        .compile_options(CompileOptions::default().max_divisibility(8))
        .build();
    let key_b = default_key()
        .compile_options(CompileOptions::default().max_divisibility(16))
        .build();
    assert_ne!(
        key_a, key_b,
        "different CompileOptions must produce distinct memory keys"
    );

    let key_c = default_key()
        .compile_options(CompileOptions::default().occupancy(2))
        .build();
    let key_d = default_key()
        .compile_options(CompileOptions::default().occupancy(4))
        .build();
    assert_ne!(key_c, key_d);

    let key_e = default_key()
        .compile_options(CompileOptions::default().num_worker_warps_per_cta(4))
        .build();
    let key_f = default_key()
        .compile_options(CompileOptions::default().num_worker_warps_per_cta(8))
        .build();
    assert_ne!(key_e, key_f);
}

#[test]
fn cache_key_no_field_boundary_ambiguity() {
    let key_a = default_key().gpu_name("sm_9").compiler_version("0").build();
    let key_b = default_key().gpu_name("sm_90").compiler_version("").build();
    assert_ne!(key_a, key_b);
}
