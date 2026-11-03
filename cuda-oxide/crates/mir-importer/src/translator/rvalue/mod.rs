/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rvalue translation: MIR expressions → `dialect-mir` operations.
//!
//! Translates the right-hand side of MIR assignments into `dialect-mir` ops.
//!
//! # Supported Rvalues
//!
//! | MIR Rvalue          | `dialect-mir` Op                                      |
//! |---------------------|-------------------------------------------------------|
//! | `BinaryOp(+,-,*,/)` | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`            |
//! | `BinaryOp(<,<=,>)`  | `mir.lt`, `mir.le`, `mir.gt`, etc.                    |
//! | `CheckedBinaryOp`   | `mir.checked_add`, etc. (returns tuple)               |
//! | `UnaryOp(Not,Neg)`  | `mir.not`, `mir.neg`                                  |
//! | `Cast`              | `mir.cast`                                            |
//! | `Ref`               | Slot pointer for locals; `mir.ref` for SSA values     |
//! | `Use(operand)`      | `mir.load` of the source slot (no op for constants)   |
//! | `Aggregate`         | `mir.construct_tuple/struct/enum/array`               |
//! | `Repeat`            | `mir.construct_array` (array repeat syntax)           |
//! | `CopyForDeref`      | Same place-read lowering as `Copy`/`Move`             |
//!
//! # Key Functions
//!
//! - [`translate_rvalue`]: Main entry point for rvalue translation
//! - [`translate_operand`]: Translates operands (Copy/Move/Constant/RuntimeChecks)
//! - [`translate_place`]: Translates places to their SSA values (handles ghost locals)
//! - `translate_constant`: Translates MIR constants to `dialect-mir`
//! - `create_ghost_enum_default`: Synthesises a placeholder for never-assigned enum locals

mod aggregate;
mod coerce;
mod const_alloc;
mod const_bytes;
mod const_enum;
mod const_union;
mod expr;
mod fn_ptr;
mod operand;
mod place_addr;
mod place_iter;
mod place_read;
mod pointee;
mod promoted;
mod static_global;
mod statics;

pub use expr::translate_rvalue;
pub use operand::translate_operand;
pub use place_read::translate_place;

pub(crate) use const_bytes::constant_bytes;
pub(crate) use place_addr::{enum_payload_needs_storage_coercion_pub, translate_place_address};
pub(crate) use place_read::apply_enum_field_projection_pub;
pub(crate) use promoted::translate_array_constant_into_alloca;
