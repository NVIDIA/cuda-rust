/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Composable FP16 CuTe operations for data-center Blackwell (SM100).
//!
//! QK and PV retain separate layout and MMA contracts. Control flow, online
//! softmax, buffer phases and participating warps remain explicit in Rust.
//! Layout-aware MMA/TMA use the CuTe compiler path; pipeline synchronization
//! and TMEM register transfers use CTM instruction primitives.
pub mod pipeline;

mod mma;
mod shared;
mod tma;
mod tmem;

pub use mma::{MmaSs, MmaTs};
pub use shared::{SharedLayout, SharedTile, SharedTileArray, SmemLayout};
pub use tma::TmaCopy;
pub use tmem::{TmemAllocator, TmemFragment, tmem_row_address};

#[doc(hidden)]
pub mod __compiler;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::create_tensor_map_tiled;
