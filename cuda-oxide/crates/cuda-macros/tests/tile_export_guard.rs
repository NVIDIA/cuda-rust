// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `#[tile_export]` signature and argument contracts: every accepted form
//! expands against the real `cuda_device::device`, and each rejected form
//! fails with a macro diagnostic that says what is accepted instead.

#[test]
fn tile_export_contracts() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/tile_export_forms.rs");
    t.compile_fail("tests/compile_fail/tile_export_generic.rs");
    t.compile_fail("tests/compile_fail/tile_export_missing_shape.rs");
    t.compile_fail("tests/compile_fail/tile_export_unsupported_param.rs");
    t.compile_fail("tests/compile_fail/tile_export_no_outputs.rs");
    t.compile_fail("tests/compile_fail/tile_export_warp_shape.rs");
    t.compile_fail("tests/compile_fail/tile_export_warp_footprint.rs");
    t.compile_fail("tests/compile_fail/tile_export_mixed_footprints.rs");
    t.compile_fail("tests/compile_fail/tile_export_unknown_argument.rs");
    t.compile_fail("tests/compile_fail/tile_export_outputs_split.rs");
}
