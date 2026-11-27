/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT compiler that translates Rust DSL modules into Tile IR and compiles them to GPU cubins.

#![allow(non_snake_case)]
extern crate core;

pub mod compile_api;
pub mod compile_control;
pub mod cuda_tile_runtime_utils;
pub mod jit_cache;
pub mod jit_env;
pub use cutile_frontend;
pub use cutile_frontend::compiler::utils;
pub use cutile_frontend::{
    ast, compiler, dump, error, foreign_decl, generics, hints, passes, registry, specialization,
    tile_library, types,
};
