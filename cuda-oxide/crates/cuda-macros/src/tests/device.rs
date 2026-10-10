/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device::sets_inline_policy;
use syn::{Attribute, parse_quote};

#[test]
fn inline_policy_detection_covers_cfg_attr() {
    let policies: [Attribute; 5] = [
        parse_quote!(#[inline]),
        parse_quote!(#[inline(always)]),
        parse_quote!(#[inline(never)]),
        parse_quote!(#[cfg_attr(any(), inline(always))]),
        parse_quote!(#[cfg_attr(all(), cold, cfg_attr(all(), inline))]),
    ];
    for attr in &policies {
        assert!(sets_inline_policy(&attr.meta), "{}", quote::quote!(#attr));
    }

    // The first item of `cfg_attr` is a condition, not an attribute.
    let others: [Attribute; 2] = [
        parse_quote!(#[cold]),
        parse_quote!(#[cfg_attr(inline, cold)]),
    ];
    for attr in &others {
        assert!(!sets_inline_policy(&attr.meta), "{}", quote::quote!(#attr));
    }
}
