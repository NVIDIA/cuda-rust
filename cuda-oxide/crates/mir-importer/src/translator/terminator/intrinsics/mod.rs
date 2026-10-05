/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU intrinsic dispatch and expansion.
//!
//! This module handles the translation of `cuda_device` intrinsic calls into
//! `dialect-nvvm` operations. Intrinsics are organized by functional category:
//!
//! | Module       | Intrinsics                                                                   |
//! |--------------|------------------------------------------------------------------------------|
//! | `generated`  | Every catalog intrinsic, dispatched by canonical and compatibility path      |
//! | `indexing`   | `threadIdx_*`, `blockIdx_*`, `index_1d`, `index_2d::<S>`, `index_2d_runtime` |
//! | `memory`     | `SharedArray`, `stmatrix_*`, type conversions                                |
//! | `atomic`     | Atomic read-modify-write and compare-exchange                                |
//! | `wgmma`      | Hopper WGMMA matrix operations                                               |
//! | `tma`        | Tensor Memory Accelerator (TMA) operations                                   |
//! | `debug`      | `clock`, `clock64`, `globaltimer`, `trap`, `breakpoint`                      |
//! | `asm`        | Inline PTX marker calls                                                      |
//! | `iket`       | `cuda_device::iket` compiler markers                                         |
//! | `layout`     | Rust DST layout intrinsics: `size_of_val`, `align_of_val`                    |
//! | `bigint`     | Rust compiler bigint helpers                                                 |
//! | `bitops`     | Rust compiler bit-manipulation intrinsics                                    |
//! | `exact_div`  | Rust compiler `exact_div`                                                    |
//! | `float_math` | Rust compiler floating-point math intrinsics                                 |
//! | `saturating` | Rust compiler saturating integer intrinsics                                  |
//!
//! # Architecture
//!
//! Each intrinsic module exports `emit_*` functions that:
//! 1. Take MIR operands and translate them to pliron IR values
//! 2. Create the appropriate `dialect-nvvm` operations
//! 3. Store results in the value map
//! 4. Emit a zero-operand `mir.goto` to the call's single successor block
//!
//! # Note
//!
//! Currently, all emit functions remain in `terminator/mod.rs` for compilation
//! stability. This module structure is prepared for gradual migration of
//! functions to their respective category modules.

// Submodules for intrinsic categories (to be populated incrementally)
pub mod asm;
pub mod atomic;
pub mod bigint;
pub mod bitops;
pub mod debug;
pub mod exact_div;
pub mod float_math;
pub mod generated;
pub mod iket;
pub mod indexing;
pub mod layout;
pub mod memory;
pub mod saturating;
pub mod tma;
pub mod wgmma;
