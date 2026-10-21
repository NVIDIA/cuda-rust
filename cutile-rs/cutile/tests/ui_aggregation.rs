/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail tests for the aggregation capabilities.
//!
//! A lane of its own so CPU CI can run it. The whole `tests/ui` suite cannot:
//! `graph_scope_rejects_allocating_input.stderr` is a stale snapshot holding a
//! `$WORKSPACE/cuda-async/...` path the merged-repository layout cannot
//! reproduce, so `--test ui` fails on unmodified `main`. Rewriting that
//! snapshot would bless whatever the diagnostic says today, which is a
//! separate decision from this capability's; these four cases are moved here
//! instead, where their expectations are checked on every CPU CI run.
//!
//! Each case in `tests/ui/*.rs` is a program that must NOT compile, paired
//! with the expected diagnostics in the `.stderr` file next to it. These need
//! no GPU; they only build against `cutile`. Regenerate the expectations with
//! `TRYBUILD=overwrite cargo test -p cutile --test ui_aggregation` after an
//! intentional diagnostic change — and read the regenerated file, because the
//! same command will also bless a real regression.

/// An aggregation capability is a narrow capability, not a tensor. `partition` is
/// its only view constructor and neither it nor `AggregationPartition`
/// exposes ordinary load or store access; `atomic_add` is restricted to the
/// float elements the operation matrix admits; the reference is shared by
/// construction; and binding one needs exclusive host authority over the
/// backing tensor.
#[test]
fn ui_aggregation() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/aggregation_tensor_withholds_ordinary_access.rs");
    t.compile_fail("tests/ui/aggregation_tensor_requires_a_supported_element.rs");
    t.compile_fail("tests/ui/aggregation_withholds_other_operations.rs");
    t.compile_fail("tests/ui/max_tensor_requires_an_ordered_element.rs");
    t.compile_fail("tests/ui/aggregation_binding_cannot_be_reimplemented.rs");
    t.compile_fail("tests/ui/aggregation_tensor_rejects_mutable_reference.rs");
    t.compile_fail("tests/ui/aggregation_tensor_needs_exclusive_host_authority.rs");
}
