/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIR translation functions often have many parameters to pass context
#![allow(clippy::too_many_arguments)]
// Complex types are unavoidable when working with rustc internals
#![allow(clippy::type_complexity)]

//! Rust MIR to `dialect-mir` translator for cuda-oxide.
//!
//! This crate translates Rust's Mid-level Intermediate Representation (MIR)
//! into [`dialect-mir`][dialect_mir] — a pliron dialect (MLIR-like) that
//! preserves Rust semantics — then hands that module to the shared
//! `cuda-oxide-codegen` backend.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────── mir-importer ──────────────────────────────────┐
//! │                                                                       │
//! │  ┌──────────────┐   ┌─────────────────────────────────────────────┐   │
//! │  │  translator  │──▶│          cuda-oxide-codegen                 │   │
//! │  │              │   │                                             │   │
//! │  │     MIR      │   │  dialect-mir (alloca)                       │   │
//! │  │      ──▶     │   │    ──▶ mem2reg                              │   │
//! │  │  dialect-mir │   │    ──▶ dialect-mir (SSA)                    │   │
//! │  │   (alloca)   │   │    ──▶ annotated loop unroll                │   │
//! │  │              │   │    ──▶ LLVM dialect  (via mir-lower)        │   │
//! │  │              │   │    ──▶ LLVM IR ──▶ PTX  (via llc)           │   │
//! │  └──────────────┘   └─────────────────────────────────────────────┘   │
//! │                                                                       │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Modules
//!
//! | Module         | Purpose                                                     |
//! |----------------|-------------------------------------------------------------|
//! | `translator`   | MIR → `dialect-mir` (alloca + load/store); crate-internal   |
//! | [`pipeline`]   | Translate a module, then call the shared codegen backend    |
//! | [`error`]      | Error types integrated with pliron's error system           |
//!
//! Note: Function collection is handled by `rustc-codegen-cuda/src/collector.rs`
//! which uses rustc internals for efficient traversal.
//!
//! # Example
//!
//! ```rust,ignore
//! use mir_importer::{CollectedFunction, PipelineConfig, run_pipeline};
//!
//! // Inside a rustc callback, once collection has the monomorphized set:
//! let functions: Vec<CollectedFunction> = collect_device_functions();
//!
//! let result = run_pipeline(
//!     &functions,
//!     &[], // device externs
//!     &PipelineConfig {
//!         output_dir: out.to_path_buf(),
//!         output_name: "kernel".to_string(),
//!         ..PipelineConfig::default()
//!     },
//!     known_defs, // lang-item DefIds resolved by the driver (KnownDefs)
//! )?;
//! ```
//!
//! `run_pipeline` owns the whole run: it registers the dialects, translates
//! every collected body into one module, and hands that module to the shared
//! backend. The translator is reached through it rather than called directly.
//!
//! # Alloca + load/store model
//!
//! Every non-ZST MIR local is materialised as a single `mir.alloca` emitted
//! at the top of the function's entry block. Defs lower to `mir.store`, uses
//! lower to `mir.load`. Cross-block data flow happens through the slots, so
//! blocks (other than the entry) take no arguments. Pliron's `mem2reg` pass
//! promotes the slots back into SSA before the `dialect-mir` → LLVM dialect
//! lowering runs.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_span;

/// Value used for every rustc `Operand::RuntimeChecks` query in device MIR.
///
/// Collection imports this same constant when choosing monomorphized switch
/// successors, so call discovery and emitted control flow cannot disagree.
pub const DEVICE_RUNTIME_CHECKS_VALUE: bool = false;

pub mod error;
pub mod pipeline;
// Crate-internal: every caller reaches the translator through `pipeline`, and
// the handful of predicates outsiders need are re-exported below. Keeping the
// module public would also keep `dead_code` switched off for the whole tree
// under it, since every item in it would count as reachable API.
pub(crate) mod translator;

pub use error::{TranslationErr, TranslationResult};
pub use pipeline::{
    CollectedFunction, CompilationArtifactKind, CompilationResult, DebugGlobalVariableIdentity,
    DeviceExternAttrs, DeviceExternDecl, DeviceExternType, KernelLaunchBounds, PipelineConfig,
    PipelineError, StatementDebugInfo, StatementDebugInfoBlock, StatementDebugInfoMap,
    build_debug_global_variable_info, build_debug_shared_array_variable_info,
    device_static_global_key, run_pipeline,
};
pub use translator::facts::KnownDefs;
pub use translator::terminator::drop_glue::{drop_glue_is_noop, drop_instance_is_noop};
pub use translator::terminator::is_panic_entry_path;
