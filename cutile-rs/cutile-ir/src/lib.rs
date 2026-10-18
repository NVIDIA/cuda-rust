/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure Rust Tile IR representation and bytecode writer.
//!
//! This crate provides an in-memory IR that mirrors the CUDA Tile dialect,
//! a builder API for constructing IR programs, and a bytecode serializer
//! that emits the Tile IR bytecode format consumed by `tileiras`.
//!
//! No LLVM or MLIR dependency is required.

pub mod builder;
pub mod bytecode;
pub mod capabilities;
pub mod ir;
pub mod tilelib;

// Shared implementation details; not re-exported by the cuTile user API.
#[doc(hidden)]
pub mod requirements;
#[doc(hidden)]
pub mod toolchain;

mod error;
mod verify_target;

pub use error::{Error, Result};

// Re-export the most commonly used entry points at crate root.
pub use builder::OpBuilder;
pub use bytecode::decoder::{decode_bytecode, decode_bytecode_file};
pub use bytecode::{
    module_preview_content, write_bytecode, write_bytecode_to_file, write_bytecode_version,
};
pub use ir::Module;
pub use tilelib::{
    add_extern_func, add_extern_impl, implementation_module, interface_module, preview_call,
    verify_library, ExternFunc, ExternImpl, FragmentArg, TileArg,
};
