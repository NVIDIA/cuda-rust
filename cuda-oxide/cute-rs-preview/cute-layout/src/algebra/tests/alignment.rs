/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Contract tests for static alignment analysis.

use super::*;

/// Static alignment factors must agree with the pinned cutegen examples.
#[test]
fn max_alignment_matches_cutegen_static_examples() {
    let cases = [
        ("(1,8):(1,1)", 8),
        ("(4,2,(8,2)):(512,1,(8,8192))", 2),
        ("((1,(2))):((1,(0)))", 1),
        ("(8,16):(1,4)", 4),
        ("(8,16):(1,6)", 2),
        ("(8,16):(1,8)", 128),
    ];
    for (text, expected) in cases {
        assert_eq!(max_alignment(&l(text)).unwrap(), expected, "{text}");
    }
}

/// Negative strides must not overstate the guaranteed access alignment.
#[test]
fn max_alignment_is_conservative_for_negative_strides() {
    assert_eq!(max_alignment(&l("8:-1")).unwrap(), 1);
}
