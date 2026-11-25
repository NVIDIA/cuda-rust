/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Copy and MMA atoms (`cutlass.cute.atom`).
//!
//! [`cooperative`] moves shared tiles with block participation; [`tiled`]
//! describes typed copy atoms and operand roles; [`vector`] moves one thread's
//! short vectors. TMA descriptor and bulk tile transfers live in [`crate::nvgpu::cpasync`].

pub mod cooperative;
pub mod tiled;
pub mod vector;
