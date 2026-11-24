/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The cuTile Rust DSL and Tile IR specifications.
//!
//! [`_core`] is the DSL itself: `Tile`, `Tensor`, ops, and their static parameter modules.
//! [`_tileir`] is the 1:1 Tile IR op mirror.

pub mod _core;
pub mod _tileir;
pub use _core::core;
pub use _tileir::tileir;
pub use cutile_frontend;
pub use linkme;
