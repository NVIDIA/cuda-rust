/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler frontend. Translates Rust DSL modules into Tile IR.
//!
//! The DSL is a subset of Rust, so the grammar is mostly identical. `syn` produces the AST
//! with [`ast`] and adds source locations for it. Then [`passes`] resolves and annotates the
//! AST, and finally [`compiler`] lowers it to a [`cutile_ir::Module`].
//!
//! See `cutile-compiler` for lowering a [`cutile_ir::Module`] to GPU cubins.

#![allow(non_snake_case)]

pub mod ast;
mod bounds;
pub mod check_optimizations;
pub mod compiler;
pub mod dump;
pub mod env;
pub mod error;
pub mod foreign_decl;
pub mod generics;
pub mod hints;
mod kernel_entry_generator;
pub mod kernel_naming;
pub mod passes;
pub mod registry;
pub mod specialization;
pub mod syn_utils;
pub mod tile_library;
pub mod train_map;
pub mod type_aliases;
pub mod types;
pub mod use_classifier;
mod value_facts;
pub use compiler::utils;
