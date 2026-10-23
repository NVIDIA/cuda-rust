/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Data movement between global memory, shared memory and thread registers.
//!
//! [`cooperative`] moves shared tiles with block participation; [`tiled`]
//! describes typed copy atoms and operand roles. [`vector`] moves one thread's
//! short vectors, while [`tma`] describes asynchronous whole-tile transfers.
//! Each family retains its own layout and participation requirements.

pub mod cooperative;
pub mod tiled;
pub mod tma;
pub mod vector;
