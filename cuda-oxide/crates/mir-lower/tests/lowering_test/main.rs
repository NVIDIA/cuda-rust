/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests build kinded fixture types directly; production minting lives in
// mir-importer's facts.rs (see the workspace clippy.toml disallowed-methods).
#![allow(clippy::disallowed_methods)]

mod atomic_storage;
mod atomics_fences;
mod barriers_sync;
mod calls_and_values;
mod common;
mod cp_async;
mod inline_ptx;
mod math_conversions;
mod matrix_memory;
mod mma;
mod shift_representation;
mod sregs_and_warp;
mod tma;
mod wgmma_lowering;
mod wgmma_rejections;
