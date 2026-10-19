/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler backend v2: translates Rust syn ASTs into Tile IR using the
//! `tile-ir` crate. Emits bytecode directly — no LLVM/MLIR dependency.
//!
//! Produces `cutile_ir::Module`, which is serialized to bytecode for GPU
//! compilation via the `tileiras` tool.

pub mod _function;
pub mod _module;
pub mod _type;
pub(crate) mod _value;
mod checks;
mod compile_assume;
mod compile_binary_op;
mod compile_block;
mod compile_cuda_tile_op;
mod compile_expression;
mod compile_global;
mod compile_inline;
mod compile_intrinsic;
mod compile_type;
pub mod optimization_hints;
pub mod shared_types;
pub mod shared_utils;
pub mod tile_rust_type;
pub mod utils;

pub use _function::CUDATileFunctionCompiler;
pub use _module::CUDATileModules;
