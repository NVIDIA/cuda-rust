/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type interfaces for MIR → LLVM type conversion.
//!
//! Defined here (in `mir-lower`) rather than in `dialect-mir` so that the
//! `#[type_interface_impl]` blocks can live in this crate without violating
//! Rust's orphan rules: the traits are local, so we can implement them for
//! foreign types (`MirPtrType`, `IntegerType`, etc.).
//!
//! Two interfaces:
//!
//! - [`MirConvertibleType`] — marker badge for `can_convert_type` queries.
//! - [`MirTypeConversion`] — returns a function pointer that performs the
//!   actual conversion (function-pointer indirection is required because
//!   `type_cast` borrows `ctx` immutably, but conversion needs `&mut ctx`).

use pliron::context::Context;
use pliron::derive::type_interface;
use pliron::result::Result;
use pliron::r#type::{Type, TypeHandle};

/// Function pointer type for MIR → LLVM type conversion.
///
/// The indirection lets us extract a `Copy` value from the immutable borrow,
/// drop the borrow, then call with `&mut Context`.
pub type ConvertMirTypeFn = fn(TypeHandle, &mut Context) -> anyhow::Result<TypeHandle>;

/// Marker interface: "this type has an LLVM equivalent".
///
/// Used by `can_convert_type` to decide whether the DialectConversion
/// framework should attempt automatic block-argument type conversion.
#[type_interface]
pub trait MirConvertibleType {
    /// No-op — the underlying type verifiers are sufficient.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Type interface for MIR → LLVM type conversion dispatch.
///
/// Each MIR / builtin / LLVM type that `convert_type` handles implements
/// this interface. The lowering code calls `type_cast`, extracts the
/// function pointer, drops the borrow, then invokes the converter.
#[type_interface]
pub trait MirTypeConversion {
    /// Return the conversion function for this type.
    fn converter(&self) -> ConvertMirTypeFn;

    /// No-op — the underlying type verifiers are sufficient.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
