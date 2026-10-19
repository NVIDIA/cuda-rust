/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw Rust FFI bindings to the MLIR C-API and the cuda-tile C-API.
//!
//! Bindings are generated at build time via `bindgen` against the LLVM+MLIR
//! sources the crate's `build.rs` downloads and builds (no reliance on
//! `mlir-sys` / `melior`).
//!
//! Usage is pure `unsafe` FFI — see the cuda-tile and MLIR C-API documentation
//! upstream. The helpers below wrap a handful of common patterns (parsing
//! modules / operations / attributes, running pass managers) so callers don't
//! have to reimplement the `CString` + `MlirStringRef` dance every time.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(dead_code)]

/// Raw FFI bindings for `mlir-c/*` and `cuda_tile-c/*`.
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use ffi::*;
use std::ffi::CString;
use std::os::raw::c_char;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Registers all cuda-tile dialects into the given `MlirDialectRegistry`.
///
/// # Safety
/// `registry` must be a valid, non-null `MlirDialectRegistry` produced by
/// `mlirDialectRegistryCreate`.
pub unsafe fn register_all_dialects(registry: MlirDialectRegistry) {
    mlirCudaTileRegisterAllDialects(registry);
}

/// Registers all cuda-tile passes.
pub unsafe fn register_all_passes() {
    mlirCudaTileRegisterAllPasses();
}

// ---------------------------------------------------------------------------
// String-ref / CString plumbing
// ---------------------------------------------------------------------------

/// Build a transient `MlirStringRef` from a Rust byte slice.
///
/// MLIR's `MlirStringRef` is non-owning — the returned view borrows from the
/// input slice, so callers must keep the slice alive for the duration of the
/// FFI call.
pub fn string_ref(bytes: &[u8]) -> MlirStringRef {
    MlirStringRef {
        data: bytes.as_ptr() as *const c_char,
        length: bytes.len(),
    }
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

/// Parse an MLIR module from text.
///
/// Returns `Some(MlirModule)` on success, `None` if parsing failed (e.g.,
/// invalid syntax, unloaded dialect).
///
/// # Safety
/// `ctx` must be a valid `MlirContext` with the dialects used by `source`
/// already loaded.
pub unsafe fn module_parse(ctx: MlirContext, source: &str) -> Option<MlirModule> {
    let module = mlirModuleCreateParse(ctx, string_ref(source.as_bytes()));
    // `MlirModule` is `struct { void *ptr; }`; null ptr = parse failure.
    if (module.ptr as *const ()).is_null() {
        None
    } else {
        Some(module)
    }
}

/// Parse a standalone MLIR operation from text.
///
/// Equivalent to `mlir::OperationCreateParse`. `source_name` is used for
/// diagnostics; pass `None` to use a default name.
///
/// # Safety
/// `ctx` must be a valid `MlirContext`.
pub unsafe fn operation_parse(
    ctx: MlirContext,
    source: &str,
    source_name: Option<&str>,
) -> Option<MlirOperation> {
    let source_c = CString::new(source).ok()?;
    let name_c = CString::new(source_name.unwrap_or("sourceName")).ok()?;
    let op = mlirOperationCreateParse(
        ctx,
        string_ref(source_c.as_bytes()),
        string_ref(name_c.as_bytes()),
    );
    if (op.ptr as *const ()).is_null() {
        None
    } else {
        Some(op)
    }
}

/// Parse an MLIR attribute from its textual form (e.g., `"42 : i32"`).
///
/// # Safety
/// `ctx` must be a valid `MlirContext`.
pub unsafe fn attribute_parse(ctx: MlirContext, source: &str) -> Option<MlirAttribute> {
    let source_c = CString::new(source).ok()?;
    let attr = mlirAttributeParseGet(ctx, string_ref(source_c.as_bytes()));
    if (attr.ptr as *const ()).is_null() {
        None
    } else {
        Some(attr)
    }
}

/// Parse an MLIR type from its textual form (e.g., `"i32"`, `"!cuda_tile.tile<4xf16>"`).
///
/// # Safety
/// `ctx` must be a valid `MlirContext` with the relevant dialects loaded.
pub unsafe fn type_parse(ctx: MlirContext, source: &str) -> Option<MlirType> {
    let source_c = CString::new(source).ok()?;
    let ty = mlirTypeParseGet(ctx, string_ref(source_c.as_bytes()));
    if (ty.ptr as *const ()).is_null() {
        None
    } else {
        Some(ty)
    }
}
