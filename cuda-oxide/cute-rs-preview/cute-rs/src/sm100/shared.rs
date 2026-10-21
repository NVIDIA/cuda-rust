/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-memory tiles for paired two-CTA GEMM.

use crate::markers::ReifySmem2D;
use core::marker::PhantomData;

/// A non-owning FP16 shared tile with a layout and an operand role.
pub struct Sm100SharedTile<Layout, Role> {
    /// Non-owning shared FP16 allocation base.
    pub(super) base: *mut f16,
    /// Records the shared layout and A/B operand role.
    marker: PhantomData<(Layout, Role)>,
}

impl<Layout: ReifySmem2D, Role> Sm100SharedTile<Layout, Role> {
    /// Interpret existing shared storage using `Layout`.
    ///
    /// # Safety
    ///
    /// `base` must remain live and contain every element addressed by `Layout`.
    /// TMA input bases must be aligned to their full swizzle period; the
    /// supported B128 MMA layouts require 1024-byte alignment.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        Self {
            base,
            marker: PhantomData,
        }
    }
}
