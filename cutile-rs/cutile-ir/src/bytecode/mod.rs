/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR bytecode writer.
//!
//! Serializes an in-memory [`Module`](crate::ir::Module) into the binary bytecode format
//! consumed by `tileiras`. Format reference: `BytecodeWriter.cpp` in the
//! `cuda-tile` submodule.

mod debug_info;
pub mod decoder;
pub mod encoding;
mod enums;
mod op_writer;
mod opcode;
mod writer;

pub use enums::*;
pub use opcode::*;
pub use writer::*;
