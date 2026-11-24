/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! In-memory Tile IR representation.
//!
//! Types here mirror the CUDA Tile dialect. Ownership is index-based
//! (no lifetime parameters) so IR fragments can be built out of order
//! and moved freely.

mod attr;
mod block;
pub mod fmt;
mod location;
mod module;
mod op;
mod region;
mod types;
mod value;

pub use attr::*;
pub use block::*;
pub use fmt::format_type;
pub use location::*;
pub use module::*;
pub use op::*;
pub use region::*;
pub use types::*;
pub use value::*;
