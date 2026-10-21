/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attention metadata and architecture-specific device operations.
//!
//! [`sm100`] is available to device-only consumers. Checked BF16 shape and
//! cache metadata requires the `host` feature and is reexported here to
//! preserve the existing `attention::AttentionShape` and related imports.

#[cfg(feature = "host")]
pub mod metadata;
pub mod sm100;

#[cfg(feature = "host")]
pub use metadata::{AttentionError, AttentionShape, AttentionWindow, CacheShape};
