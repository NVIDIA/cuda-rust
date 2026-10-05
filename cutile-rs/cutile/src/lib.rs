/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! # cuTile Rust
//!
//! A high-performance GPU computing library that enables you to write Rust code that compiles
//! directly to CUDA kernels. cuTile Rust combines the safety and ergonomics of Rust with the
//! performance of hand-tuned GPU code.
//!
//! ## Overview
//!
//! cuTile Rust provides a complete stack for GPU programming in Rust:
//!
//! - **Core types**: GPU tensors, tiles, and partitions for structured data access
//! - **Async execution**: Modern async/await syntax for GPU operations with automatic scheduling
//! - **Kernel compilation**: Rust → Tile IR bytecode → cubin compilation pipeline with caching
//! - **High-level API**: Familiar NumPy-like operations for tensor creation and manipulation
//! - **Built-in kernels**: The creation and conversion kernels behind [`api`] (`zeros`, `arange`,
//!   `eye`, dtype conversion); reusable compute kernels live in the `cutile-kernels` crate
//!
//! ## Quick Start
//!
//! ### Creating and manipulating tensors
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Create tensors on GPU
//! let x = api::ones::<f32>(&[1024]).await;
//! let y = api::zeros::<f32>(&[1024]).await;
//! let z = api::arange::<f32>(256).await;
//!
//! // All operations are async and execute on GPU
//! ```
//!
//! ### Writing custom GPU kernels
//!
//! ```rust,ignore
//! use cutile::prelude::*;
//!
//! #[cutile::module]
//! mod kernels {
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
//!
//! // Output-first: &mut param is first. Pass DeviceOps directly.
//! let result = kernels::vector_add(
//!     api::zeros(&[256]).partition([64]),
//!     api::ones(&[256]),
//!     api::ones(&[256]),
//! )
//! .first()
//! .unpartition()
//! .to_host_vec()
//! .await?;
//! ```
//!
//! ### Async GPU pipelines
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // All operations are lazy — no GPU work until .await
//! let x = api::randn(0.0f32, 1.0, [1024, 1024], None).await?;
//! let y = api::randn(0.0f32, 1.0, [1024, 1024], None).await?;
//!
//! let z = gemm(
//!     api::zeros(&[1024, 1024]).partition([128, 128]),
//!     x, y,
//! )
//! .generics(vec!["128".into(), "128".into(), "32".into(), "1024".into()])
//! .first()
//! .unpartition()
//! .await?;
//! ```
//!
//! ## Module Organization
//!
//! - [`core`] - Core GPU types: `Tile`, `Tensor`, `Partition`, and tile operations
//! - [`api`] - High-level tensor creation and manipulation functions
//! - [`tensor`] - GPU tensor type, partitioning, and memory management
//! - [`tile_kernel`] - Async execution primitives and kernel compilation infrastructure
//! - [`kernels`] - The creation and conversion kernels that back [`api`]
//!
//! ## Key Concepts
//!
//! ### Tensors and Partitions
//!
//! A [`Tensor`](tensor::Tensor) represents data on the GPU. It can be partitioned into tiles
//! that map to CUDA thread blocks:
//!
//! ```rust,ignore
//! // Create a tensor and partition it into 64-element tiles
//! let x = api::ones(&[256]).partition([64]); // Creates 4 partitions
//! ```
//!
//! ### Tiles
//!
//! Within kernels, you work with `Tile` values that represent data in registers
//! or shared memory. Tiles support efficient element-wise operations and broadcasting:
//!
//! ```rust,ignore
//! let tile_a = load_tile_mut(tensor_a);
//! let tile_b = load_tile_mut(tensor_b);
//! let result = tile_a * tile_b + tile_a; // All operations on GPU registers
//! ```
//!
//! ### Async Execution
//!
//! All GPU operations return futures that can be awaited. The runtime automatically manages
//! dependencies, kernel launches, and memory transfers:
//!
//! ```rust,ignore
//! // Operations compose naturally with async/await
//! let x = api::zeros(&[1024]).await;
//! let y = my_kernel(x).await;
//! let z = another_kernel(y).await;
//! ```
//!
//! ## Feature Flags
//!
//! - `experimental-tune` (off by default) — enables the `cutile::tune` module: declared
//!   search spaces, pluggable searchers, and persisted, provenance-checked autotuning
//!   results. Experimental: its API may change in breaking ways between releases.
//!
//! ## Safety
//!
//! cuTile Rust uses unsafe code internally for FFI with CUDA and for performance-critical operations.
//! The public API is designed to be safe: kernel launches are validated against the compiled
//! specialization (element types, partition shapes, declared preconditions) before any GPU work,
//! and partitions give each thread block a disjoint tile. Partition shapes need not divide the
//! tensor shape — the launch grid is the ceiling division, and a partial edge tile is loaded and
//! stored with bounds checks (or padding) rather than rejected.
//!
//! ## Examples
//!
//! See the `cutile-examples` crate for more comprehensive examples including:
//! - Matrix multiplication (GEMM)
//! - Async execution patterns
//! - Custom kernel development
//!
//! ## Performance
//!
//! cuTile Rust kernels can achieve performance competitive with hand-written CUDA:
//! - Zero-cost abstractions: Rust compiles to Tile IR, then optimized GPU code
//! - Compile-time specialization: Tile shapes and types are compile-time constants
//! - Kernel caching: Compiled kernels are cached per-device for reuse
//!
//! ## Learning Resources
//!
//! For tutorials, the DSL reference, and the execution model, see the
//! [cuTile Rust Book](https://nvlabs.github.io/cutile-rs/).

pub mod _core;
pub mod _tileir;
pub mod error;
pub use _core::core;
pub use _tileir::tileir;

// LINKING Phase B: register an additional registry entry at the public
// `cutile::core` path so kernel `use cutile::core::*` statements resolve
// to the same module that `cutile::_core::core::__module_ast_self()`
// builds. Without this alias, the registry-driven JIT path misses the
// cutile DSL core module when kernels use the public re-export name.
//
// Per-`pub use ...::module` re-exports of cuTile modules will need a
// similar alias entry. A future macro could generate these automatically;
// for now the cutile crate's single re-export is hand-registered here.
#[linkme::distributed_slice(cutile_compiler::registry::CUTILE_MODULES)]
static __CUTILE_REEXPORT_CORE: cutile_compiler::registry::CutileModuleEntry =
    cutile_compiler::registry::CutileModuleEntry {
        absolute_path: "cutile::core",
        build: _core::core::__module_ast_self,
    };
#[linkme::distributed_slice(cutile_compiler::registry::CUTILE_MODULES)]
static __CUTILE_REEXPORT_TILEIR: cutile_compiler::registry::CutileModuleEntry =
    cutile_compiler::registry::CutileModuleEntry {
        absolute_path: "cutile::tileir",
        build: _tileir::tileir::__module_ast_self,
    };
pub mod api;
pub mod bench;
pub mod kernels;
pub mod prelude;
pub mod tensor;
pub mod tile_kernel;
#[cfg(feature = "experimental-tune")]
pub mod tune;
pub mod utils;

pub use cuda_async;
pub use cuda_core;
pub use cuda_core::{DType, DTypeId};
pub use cutile_compiler;
pub use cutile_compiler::compile_api;
/// Opt-in persistent on-disk cubin cache. Off by default; enable
/// explicitly via [`jit_cache::enable_default`] or [`jit_cache::enable`].
pub use cutile_compiler::jit_cache;
pub use cutile_macro::module;
pub use half;
pub use num_traits;
