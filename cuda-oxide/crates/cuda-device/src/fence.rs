/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory fence intrinsics for CUDA device code.
//!
//! These are the device-side visibility primitives used to order ordinary global
//! stores before signaling via atomics:
//!
//! - [`threadfence_block()`] -> PTX `membar.cta`
//! - [`threadfence()`] -> PTX `membar.gl`
//! - [`threadfence_system()`] -> PTX `membar.sys`
//!
//! The functions are compiler-recognized stubs. Their bodies never execute; the
//! cuda-oxide compiler replaces each call with the corresponding NVVM/PTX fence.

include!("generated/fence.rs");
