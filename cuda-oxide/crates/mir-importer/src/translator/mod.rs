/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust MIR to `dialect-mir` translator.
//!
//! Converts Rust's MIR (from rustc) into [`dialect-mir`][dialect_mir] ops.
//! This is the core of cuda-oxide's ability to compile Rust to GPU code.
//!
//! # Module Structure
//!
//! | Module         | Purpose                                           |
//! |----------------|---------------------------------------------------|
//! | [`body`]       | Function-level translation, alloca setup          |
//! | [`block`]      | Basic block translation coordinator               |
//! | [`statement`]  | Statement translation (assignments, storage)      |
//! | [`terminator`] | Terminator translation (goto, call, return)       |
//! | [`rvalue`]     | Expression translation (binops, casts, etc.)      |
//! | [`types`]      | Rust type → `dialect-mir` type conversion         |
//! | [`values`]     | MIR local → alloca slot mapping                   |
//! | [`facts`]      | Typed fact reads from rustc (the oracle)          |
//! | [`layout`]     | Shared readers over rustc's aggregate layout       |
//! | [`location`]   | Source-location helpers for MIR translation       |
//! | [`payload_store`] | Enum-payload stores whose storage type differs  |
//!
//! # Translation Flow
//!
//! ```text
//! pipeline::run_pipeline()
//!   └─▶ register_dialects()
//!   └─▶ body::translate_body()          // once per collected function
//!         ├─▶ emit_entry_allocas()        // one alloca per non-ZST local
//!         └─▶ For each reachable block:
//!               └─▶ block::translate_block()
//!                     ├─▶ statement::translate_statement()
//!                     │     └─▶ rvalue::translate_rvalue()
//!                     └─▶ terminator::translate_terminator()
//! ```
//!
//! # Alloca + load/store model
//!
//! Every non-ZST MIR local is backed by a single `mir.alloca` emitted at the
//! top of the function's entry block. Defs lower to `mir.store`, uses lower
//! to `mir.load`. Cross-block data flow happens via these slots — no block
//! arguments other than the entry block's function parameters.
//!
//! The `mem2reg` pass in [`crate::pipeline`] promotes the scalar slots back
//! into SSA before the `dialect-mir` → LLVM dialect lowering runs.

pub mod block;
pub mod body;
pub(crate) mod facts;
pub(crate) mod layout;
pub(crate) mod location;
pub(crate) mod payload_store;
pub mod rvalue;
pub mod statement;
pub(crate) mod statement_debug;
pub mod terminator;
pub mod types;
pub mod values;

use pliron::context::Context;

/// Public `SharedArray` methods whose compiler expansion returns a generic
/// pointer to the underlying shared allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SharedArrayPointerMethod {
    /// `SharedArray::as_ptr(&self)`.
    BorrowedConst,
    /// `SharedArray::as_mut_ptr(&mut self)`.
    BorrowedMut,
    /// `SharedArray::as_raw_mut_ptr(*mut Self)`.
    RawMut,
}

/// Recognize exactly the three public `SharedArray` pointer conversions.
///
/// Intrinsic dispatch and destination address-space classification both use
/// this helper so adding a method cannot update one compiler path without the
/// other.
pub(crate) fn shared_array_pointer_method(path: &str) -> Option<SharedArrayPointerMethod> {
    if !path.starts_with("cuda_device::")
        || !path.split("::").any(|component| component == "SharedArray")
    {
        return None;
    }

    match path.rsplit("::").next() {
        Some("as_ptr") => Some(SharedArrayPointerMethod::BorrowedConst),
        Some("as_mut_ptr") => Some(SharedArrayPointerMethod::BorrowedMut),
        Some("as_raw_mut_ptr") => Some(SharedArrayPointerMethod::RawMut),
        _ => None,
    }
}

/// Registers all dialects needed for translation.
///
/// Registers `dialect-mir` (our MIR modelling dialect), `dialect-nvvm`
/// (GPU intrinsics), and the `builtin` dialect (`ModuleOp`, `FunctionType`).
/// Note: Each dialect's `register()` function uses `entry().or_insert()`,
/// so it's safe to call even if already registered.
pub fn register_dialects(ctx: &mut Context) {
    dialect_mir::register(ctx);
    dialect_iket::register(ctx);

    // dialect-nvvm is required for thread / block / warp intrinsics.
    dialect_nvvm::register(ctx);

    // The builtin dialect (ModuleOp etc.) is auto-registered by pliron 0.14.
}

#[cfg(test)]
mod tests {
    use super::{SharedArrayPointerMethod, shared_array_pointer_method};

    #[test]
    fn shared_array_pointer_recognition_is_exact_and_centralized() {
        for (path, expected) in [
            (
                "cuda_device::shared::SharedArray::as_ptr",
                SharedArrayPointerMethod::BorrowedConst,
            ),
            (
                "cuda_device::shared::SharedArray::as_mut_ptr",
                SharedArrayPointerMethod::BorrowedMut,
            ),
            (
                "cuda_device::shared::SharedArray::as_raw_mut_ptr",
                SharedArrayPointerMethod::RawMut,
            ),
        ] {
            assert_eq!(shared_array_pointer_method(path), Some(expected), "{path}");
        }

        for near_match in [
            "cuda_device::shared::DynamicSharedArray::as_ptr",
            "cuda_device::shared::SharedArrayHelper::as_ptr",
            "cuda_device::shared::SharedArray::as_raw_mut_ptr_extra",
            "other_crate::SharedArray::as_raw_mut_ptr",
        ] {
            assert_eq!(
                shared_array_pointer_method(near_match),
                None,
                "{near_match}"
            );
        }
    }
}
