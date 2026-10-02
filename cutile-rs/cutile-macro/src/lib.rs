/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Procedural macros for the cuTile Rust GPU kernel framework.
//!
//! Provides the `#[cutile::module]` attribute that turns a Rust module of
//! kernel code into a compilable Rust module + a runtime hook that hands
//! the original (pre-expansion) source AST to the JIT for CUDA codegen.
//!
//! ## Pipeline (per item inside the module)
//!
//! ```text
//! Rust source AST
//!       ↓
//! [validate_dsl_syntax]  ← entry-point parameter checks
//!       ↓
//! [_module]              ← orchestration: route each item by attribute
//!       ↓ ↓ ↓
//!   [rank_instantiation]     [shadow_dispatch]      [kernel_launcher_generator]
//!   rank-instance specialize  CGA-erased trait      emit `#[entry]` launchers
//!   structs / impls /    + rank-instance impls      (host-side glue)
//!   inherent fn bodies   + free-fn wrappers
//!       ↓
//! Emitted Rust (consumed only by rustc)
//!  + `_module_asts()`    ← runtime hook returning the original source AST
//!                          for the JIT compiler
//! ```
//!
//! Two emitters carry the rank-polymorphism:
//!
//! - `rank_instantiation` handles items whose generics include a CGA
//!   (`const X: [i32; N]`) by producing one concrete copy per rank instance.
//! - `shadow_dispatch` handles `#[variadic_op]` fns and `#[variadic_trait]`
//!   declarations by emitting a single CGA-erased shadow trait plus
//!   rank-instance impls; user call sites resolve through rustc's normal trait
//!   dispatch.
//!
//! The JIT path is independent: at proc-macro time `_module` captures the
//! verbatim source text via `Span::source_text()`, and `_module_asts()`
//! re-parses that string at runtime so the JIT works from the user's
//! original CGA-generic definitions, not the macro-emitted specializations.
//!
//! ## Example
//!
//! ```rust,ignore
//! #[cutile::module]
//! mod my_kernels {
//!     use cutile::core::*;
//!
//!     #[cutile::entry]
//!     fn vector_add<T: ElementType, const N: i32>(
//!         z: &mut Tensor<T, {[N]}>,
//!         x: &Tensor<T, {[-1]}>,
//!         y: &Tensor<T, {[-1]}>,
//!     ) {
//!         let tile_x = load_tile_like(x, z);
//!         let tile_y = load_tile_like(y, z);
//!         z.store(tile_x + tile_y);
//!     }
//! }
//! ```
//!
//! Expands to: rank-instance specializations of `vector_add` for rustc, a
//! launcher (`VectorAdd { … }` + `vector_add(…)`) for host code, and
//! `_module_asts()` returning the captured source AST for JIT compilation.
//!
//! ## See also
//!
//! - `cutile` — runtime library and core types.
//! - `cutile_compiler` — MLIR/PTX backend that consumes `_module_asts()`.

#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(non_snake_case)]

use proc_macro::TokenStream;

// Note: These modules are private because proc-macro crates can only export proc-macro functions.
// Use `cargo doc --document-private-items` to generate documentation for these modules.
mod _module;
mod error;
mod kernel_launcher_generator;
mod rank_instantiation;
mod shadow_dispatch;
mod validate_dsl_syntax;

/// Transforms a Rust module into GPU kernel code with kernel launchers.
///
/// This procedural macro is the main entry point for writing GPU kernels in cuTile Rust.
/// It processes a module containing kernel functions marked with `#[entry]` and generates:
///
/// - MLIR AST builder functions for compilation to CUDA
/// - Direct launcher functions for host-side execution
/// - `Unified launchers accepting `IntoDeviceOp` for each parameter
/// - Type metadata for shape inference and validation
///
/// ## Basic Usage
///
/// ```rust,ignore
/// #[cutile::module]
/// mod kernels {
///     use cutile::core::*;
///     
///     #[cutile::entry]
///     fn my_kernel<const N: i32>(data: &mut Tensor<f32, {[N]}>) {
///         let tile = data.load();
///         data.store(tile * 2.0);
///     }
/// }
///
/// // Generated: kernels::my_kernel() unified launcher (accepts IntoDeviceOp args)
/// ```
///
/// ## Attributes
///
/// - `tile_rust_crate=true` - Indicates this is within the cutile crate
///
/// ## Generated Code
///
/// For each `#[entry]` function, the macro generates:
///
/// 1. **AST Builder** - `<function>_ast()` - Builds MLIR representation
/// 2. **Direct Launcher** - `<function>()` - Wraps materialized values as device operations
/// 4. **Metadata** - Type information for shape inference
///
/// ## See Also
///
/// - Main crate documentation for usage examples
/// - `_module::module` for implementation details
#[proc_macro_attribute]
pub fn module(attr: TokenStream, input: TokenStream) -> TokenStream {
    _module::module(attr, input)
}
