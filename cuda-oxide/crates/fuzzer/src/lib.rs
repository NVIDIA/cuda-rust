/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Differential codegen fuzzer support crate for cuda-oxide.
//!
//! This crate hosts the reusable pieces shared by the rustlantis-style
//! differential tester:
//!
//! - A device-compatible single-`u64` trace state, exposed via
//!   [`trace_reset`], [`trace_finish`], and the generic [`dump_var`].
//! - A [`TraceValue`] / [`TraceDump`] pair so generated programs can fold
//!   intermediate values of arbitrary supported scalar types into the trace.
//!
//! The runtime layout (vendored rustlantis source under `rustlantis/`,
//! `tools/mir_generator.py`, etc.) lives next to this library. The library
//! itself is `no_std` and contains nothing CUDA- or std-specific.

// The library is `no_std`. The test harness is not, so a test build links std
// on the host while every real build, including the device build, does not.
#![cfg_attr(not(test), no_std)]

pub mod trace;

pub use trace::{TraceDump, TraceValue, dump_var, trace_finish, trace_reset};
