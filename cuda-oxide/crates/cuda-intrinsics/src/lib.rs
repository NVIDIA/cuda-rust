/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
//! Low-level CUDA intrinsic declarations generated for cuda-oxide.
//!
//! Most applications should use `cuda-device`. This crate is the raw compiler
//! contract: cuda-oxide recognizes these functions by their generated paths and
//! replaces calls with GPU operations. Their placeholder bodies are never meant
//! to execute.

include!("generated/mod.rs");
