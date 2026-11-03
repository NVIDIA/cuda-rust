/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Centralized naming for user-facing kernel names and generated symbols.
///
/// `public_name` is always the kernel name written by the user in `#[cutile::entry]`.
/// All other names are derived from it so the mapping between:
///
/// - the user-visible kernel API
/// - the hidden Rust implementation symbol
/// - the compiler-generated entry point
/// - helper functions used by launcher codegen
///
/// lives in exactly one place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelNaming {
    public_name: String,
}

impl KernelNaming {
    const USER_IMPL_PREFIX: &'static str = "__cutile_user_impl_";

    /// Creates naming metadata from the user-defined kernel function name.
    pub fn new(public_name: &str) -> Self {
        Self {
            public_name: public_name.to_string(),
        }
    }

    /// Returns the user-visible kernel name for either a public kernel name or
    /// a generated user-implementation symbol.
    pub fn canonical_public_name(name: &str) -> String {
        match Self::public_name_from_user_impl_name(name) {
            Some(public_name) => public_name,
            None => name.to_string(),
        }
    }

    /// If `name` is a generated user-implementation symbol, returns the
    /// original user-visible kernel name.
    pub fn public_name_from_user_impl_name(name: &str) -> Option<String> {
        name.strip_prefix(Self::USER_IMPL_PREFIX)
            .map(|s| s.to_string())
    }

    /// If `name` is a generated `_entry` symbol (see
    /// [`entry_name`](Self::entry_name)), returns the user-visible kernel
    /// name. Debug info uses this as the subprogram name so debuggers and
    /// profilers show the name the user wrote, with the symbol kept as the
    /// linkage name.
    pub fn public_name_from_entry_name(name: &str) -> Option<String> {
        name.strip_suffix("_entry").map(|s| s.to_string())
    }

    /// Returns the user-visible kernel name.
    ///
    /// This is the name callers pass to `CUDATileFunctionCompiler::new(...)`
    /// and the unsuffixed launcher name exposed from `#[cutile::module]`.
    pub fn public_name(&self) -> &str {
        &self.public_name
    }

    /// Returns the `_apply` variant name (legacy, still generated).
    ///
    /// Prefer the unified launcher `<kernel>(...)` which accepts `IntoDeviceOp` args directly.
    pub fn apply_name(&self) -> String {
        format!("{}_apply", self.public_name)
    }

    /// Returns the `_op` variant name (legacy, still generated).
    ///
    /// Prefer the unified launcher `<kernel>(...)` which accepts `IntoDeviceOp` args directly.
    pub fn device_op_helper_name(&self) -> String {
        format!("{}_op", self.public_name)
    }

    /// Returns the hidden Rust symbol for the user-authored kernel implementation.
    ///
    /// This avoids colliding with the public unsuffixed launcher while still
    /// letting the compiler-generated entry point call the real kernel body.
    pub fn user_impl_name(&self) -> String {
        format!("{}{}", Self::USER_IMPL_PREFIX, self.public_name)
    }

    /// Returns the compiler-facing entry point name.
    ///
    /// This is the symbol emitted into the generated CUDA Tile module and later
    /// loaded as the compiled kernel entry.
    pub fn entry_name(&self) -> String {
        format!("{}_entry", self.public_name)
    }
}
