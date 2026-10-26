/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translate `cute-rs` calls into CuTe operations.
//!
//! CuTe-specific call handling lives here. The shared translator calls this
//! module from `translate_call`, as it does for CUDA intrinsics.
//!
//! Calls are matched by their definition path, found by walking the `DefId`
//! parent chain. This keeps matching stable when a function is re-exported or
//! a trait method is printed with a different name:
//!
//! ```text
//! public name:      cute_rs::load_tile
//! definition path:  cute_rs::copy::vector::load_tile -> CUTE_FNS entry
//! ```
//!
//! There are three routes through this boundary:
//!
//! - Registered primitives use [`registry::CUTE_FNS`] and validate generic
//!   argument kinds before their family decoder reads static configuration.
//! - Ergonomic methods in approved source modules are recognized by exact
//!   defining module and receiver/result ADTs in [`classify`]. Their family
//!   decoders validate those types without a registry entry.
//! - Other helpers in approved source modules use ordinary body translation.
//!
//! The first two routes emit CuTe operations without translating the Rust body.
//! Unknown `cute_rs` calls outside approved modules are errors: translating a
//! compiler stub's body would lose the operation's meaning. [`identity`] owns
//! the module allowlist; [`dispatch`] selects the decoder and emitter.

mod classify;
mod dispatch;
mod identity;
mod registry;

pub(crate) mod block_scaled;
pub(crate) mod emit;
pub(crate) mod epilogue;
pub(crate) mod layout;
pub(crate) mod pipeline;
pub(crate) mod scheduler;
pub(crate) mod sm100;
pub(crate) mod sm100_attention;
pub(crate) mod smem_mma;
pub(crate) mod static_config;
pub(crate) mod tensor;

pub(crate) use dispatch::try_translate_cute_call;
pub(super) use identity::function_item_requires_direct_dispatch;
pub use registry::is_cute_compiler_hook_path;
