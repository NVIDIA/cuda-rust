/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` aggregate operations (structs, tuples, enums) to
//! their LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation            | LLVM Operation(s)                    | Description            |
//! |--------------------------|--------------------------------------|------------------------|
//! | `mir.extract_field`      | `llvm.extractvalue`                  | Get struct/tuple field |
//! | `mir.insert_field`       | `llvm.insertvalue`                   | Set struct/tuple field |
//! | `mir.construct_struct`   | `llvm.undef` + `llvm.insertvalue`    | Build struct           |
//! | `mir.construct_tuple`    | `llvm.undef` + `llvm.insertvalue`    | Build tuple            |
//! | `mir.construct_slice`    | `llvm.undef` + `llvm.insertvalue`    | Build slice fat ptr    |
//! | `mir.construct_enum`     | `llvm.undef` + `llvm.insertvalue`    | Build enum             |
//! | `mir.get_discriminant`   | `llvm.extractvalue`                  | Get enum tag           |
//! | `mir.set_discriminant`   | `llvm.getelementptr` + `llvm.store`  | Write enum tag         |
//! | `mir.enum_payload`       | `llvm.extractvalue`                  | Get enum payload       |
//!
//! # Enum Representation
//!
//! Enums use rustc's physical layout, not a cuda-oxide-only tagged struct.
//! `build_enum_slot_map` places the direct tag or niche carrier and every
//! payload at rustc's byte offsets, reuses identical overlapping storage, and
//! routes differently typed overlaps through byte-addressed memory. `Single`
//! and `Empty` layouts have no carrier at all.
//!
//! A direct tag holds the variant's DECLARED discriminant value, not its
//! position. For `core::cmp::Ordering`, `Less` therefore stores -1 (the i8 bit
//! pattern 255), `Equal` stores 0, and `Greater` stores 1. A niche layout
//! instead uses rustc's wrapping `niche_start + variant_offset` encoding and
//! introduces no extra tag.

mod addressing;
mod array_extract;
mod carrier_field_addr;
mod common;
mod construct;
mod enum_layout;
mod enums;
mod fields;
#[cfg(test)]
mod test_support;

pub(crate) use addressing::convert_array_element_addr;
pub(crate) use array_extract::convert_extract_array_element;
pub(crate) use carrier_field_addr::convert_field_addr;
pub(crate) use construct::{
    convert_construct_array, convert_construct_disjoint_slice, convert_construct_slice,
    convert_construct_struct, convert_construct_tuple,
};
pub(crate) use enums::{
    convert_construct_enum, convert_enum_payload, convert_get_discriminant,
    convert_set_discriminant,
};
pub(crate) use fields::{convert_extract_field, convert_insert_field};
