/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The cuda-oxide async surface.
//!
//! Everything under this module is the cuda-oxide `cuda-async` API, brought
//! over as-is for the shared host-crate migration. The two async stacks fork
//! the same ancestor, so they share most type names with diverged
//! definitions; nothing here is re-exported at the crate root, and consumers
//! address it as `cuda_async::simt::<module>::<item>`, mirroring oxide's
//! `cuda_async::<module>::<item>`. Reconciliation happens after the
//! repository migration, and this module is deletable as a unit.
//!
//! One departure, recorded for the reconciliation: oxide's `zip!` and
//! `unzip!` macros are present but not exported, since this crate already
//! exports macros with those names.

pub mod device_box;
pub mod device_context;
pub mod device_future;
pub mod device_operation;
pub mod error;
pub mod launch;
pub mod reclaim;
pub mod scheduling_policies;
