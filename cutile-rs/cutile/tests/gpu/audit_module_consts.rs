/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-level `const`s resolve per module. They used to live in one flat,
//! last-indexed-wins map, so a same-named const in a linked helper module
//! replaced the kernel's own value in its tile shapes: the compiler emitted
//! `tile<64xf32>` while the generated launcher believed 128 (2026-08 audit).

use cutile::prelude::*;

use crate::audit_common::{self, host};
use crate::common;

/// The helper module defines its own `N`, different from the kernel's, and
/// uses it in a value position.
#[cutile::module]
mod const_helper_module {

    const N: i32 = 64;

    /// This module's `N`, as seen from inside the inlined body.
    pub fn helper_n() -> i32 {
        N
    }
}

/// The kernel module defines `N = 128` and uses it both in a type (the tile
/// shape, which the launcher also sees) and in a value.
#[cutile::module]
mod const_kernel_module {
    use super::const_helper_module::helper_n;
    use cutile::core::*;

    const N: i32 = 128;

    /// Stores `[N]` elements (this module's `N`) whose values encode both
    /// modules' `N`: `1000 * kernel_n + helper_n`.
    #[cutile::entry()]
    fn fill_with_both(out: &mut Tensor<f32, { [N] }>) {
        let mine: i32 = N;
        let theirs: i32 = helper_n();
        let encoded: Tile<f32, { [N] }> =
            broadcast_scalar(convert_scalar::<f32>(mine * 1000i32 + theirs), out.shape());
        out.store(encoded);
    }
}

use const_kernel_module::__module_ast_self;

#[test]
fn same_named_consts_keep_their_own_module_scope() {
    common::with_test_stack(|| {
        let (ir, _) = audit_common::compile(
            __module_ast_self,
            "const_kernel_module",
            "fill_with_both",
            &[],
            &[("out", &[1])],
        )
        .expect("fill_with_both should compile");
        assert!(
            ir.contains("tile<128xf32>"),
            "the kernel's `N = 128` must shape the tile, not the helper's 64:\n{ir}"
        );
        assert!(
            !ir.contains("tile<64xf32>"),
            "the helper module's `N` leaked into the kernel's shape:\n{ir}"
        );
        // On the device: 128 elements (the launcher's partition and the
        // kernel's shape agree), each `1000 * 128 + 64`.
        let (out,) =
            const_kernel_module::fill_with_both(api::zeros::<f32>(&[128]).partition([128]))
                .sync()
                .expect("fill_with_both");
        let out = host(&out.unpartition());
        assert_eq!(out.len(), 128);
        assert!(
            out.iter().all(|&v| v == 128_064.0),
            "each module must see its own `N`: {:?}",
            &out[..4]
        );
    });
}
