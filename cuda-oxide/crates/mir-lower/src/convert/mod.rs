/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `dialect-mir` → LLVM dialect operation conversion.
//!
//! Each MIR/NVVM op implements `MirToLlvmConversion` (see
//! [`crate::conversion_interface`]) via `#[op_interface_impl]` blocks in
//! [`interface_impls`]. The lowering pass dispatches through `op_cast`, which
//! resolves to the correct per-op converter in O(1) via vtable lookup.
//!
//! Converter logic lives in submodules organised by category:
//! - [`ops::arithmetic`] — arithmetic, bitwise, and comparison ops
//! - [`ops::memory`] — load, store, ref, pointer offset, shared alloc
//! - [`ops::constants`] — integer and float constants
//! - [`ops::cast`] — type casts (int↔float, ptr↔int, transmute, etc.)
//! - [`ops::aggregate`] — struct, tuple, array, and enum ops
//! - [`ops::control_flow`] — return, goto, branches, assert, unreachable
//! - [`ops::call`] — function calls
//! - [`intrinsics`] — GPU intrinsics (thread/block queries, TMA, WGMMA, etc.)
//!
//! # Adding New Operations
//!
//! 1. Add the op type to the appropriate dialect crate.
//! 2. Write a `pub(crate) fn convert_*` function in the relevant submodule.
//! 3. Add an `#[op_interface_impl]` block in [`interface_impls`].

use pliron::{
    context::{Context, Ptr},
    location::Located,
    operation::Operation,
};

/// Copy the source operation's location to an operation created while lowering it.
pub(crate) fn preserve_location(
    ctx: &mut Context,
    source: Ptr<Operation>,
    lowered: Ptr<Operation>,
) -> Ptr<Operation> {
    lowered.deref_mut(ctx).set_loc(source.deref(ctx).loc());
    lowered
}

pub(crate) mod enum_payload_storage;
mod generated_intrinsics;
pub mod interface_impls;
pub mod intrinsics;
pub mod ops;
pub(crate) mod target_stable_storage;
pub mod type_interface_impls;
pub mod types;
