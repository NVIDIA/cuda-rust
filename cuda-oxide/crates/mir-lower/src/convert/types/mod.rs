/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion from `dialect-mir` types to LLVM dialect types.
//!
//! This module handles the translation of `dialect-mir` type representations
//! to their LLVM dialect equivalents. Type conversion is foundational to
//! the lowering pass—most operation converters depend on it.
//!
//! # Overview
//!
//! `dialect-mir` types are high-level, Rust-like types that preserve semantic
//! information (signedness, slice semantics, etc.). LLVM dialect types are
//! lower-level and match LLVM IR types directly.
//!
//! # Type Mapping Table
//!
//! | `dialect-mir` Type              | LLVM dialect Type                 | Notes                       |
//! |---------------------------------|-----------------------------------|-----------------------------|
//! | `IntegerType` (signed/unsigned) | `IntegerType` (signless)          | Width preserved             |
//! | `MirFP16Type`                   | `HalfType`                        | Rust `f16` → LLVM `half`    |
//! | `FP32Type`, `FP64Type`          | Same (builtin)                    | Pass-through                |
//! | `MirPtrType`                    | `PointerType`                     | Address space preserved     |
//! | `MirSliceType`                  | `StructType { ptr, i64 }`         | Fat pointer                 |
//! | `MirDisjointSliceType`          | `StructType { ptr, i64 }`         | Same as slice               |
//! | `MirTupleType`                  | `StructType`                      | Empty tuple → empty struct  |
//! | `MirStructType`                 | `StructType`                      | Fields recursively converted|
//! | `MirUnionType`                  | Aligned shared-storage struct    | All fields start at byte zero|
//! | `MirEnumType`                   | `StructType` (rustc byte layout)  | See "Enum Type Representation" |
//! | `ArrayType`                     | `ArrayType`                       | Element type converted      |
//! | `VectorType`                    | `VectorType`                      | Element type converted      |
//!
//! # Signedness Handling
//!
//! LLVM IR integers are signless—the signedness is encoded in the operations
//! that use them (e.g., `sdiv` vs `udiv`). During type conversion:
//!
//! - Signed/unsigned MIR integers → signless LLVM integers
//! - The original signedness is preserved in operations (see `arithmetic.rs`)
//!
//! # Address Space Handling
//!
//! GPU memory uses address spaces to distinguish memory types:
//!
//! | Address Space | Memory Type | Usage                     |
//! |---------------|-------------|---------------------------|
//! | 0             | Generic     | Can point to any memory   |
//! | 1             | Global      | Device memory (VRAM)      |
//! | 3             | Shared      | Per-block shared memory   |
//! | 4             | Constant    | Read-only device memory   |
//! | 5             | Local       | Per-thread stack/spill    |
//!
//! Pointer address spaces are preserved through conversion. Slice types use
//! generic address space (0) because they can point to any memory type.
//!
//! # Slice Type Representation
//!
//! Rust slices (`&[T]`) are represented as fat pointers in LLVM:
//!
//! ```text
//! MIR: MirSliceType<f32>
//! LLVM: struct { ptr, i64 }  ; pointer + length
//! ```
//!
//! This matches the Rust ABI for slices passed by value.
//!
//! # Enum Type Representation
//!
//! A Rust enum is one tag plus the payload of whichever variant is
//! alive; all variants share the same bytes. We build an LLVM struct
//! that puts the tag and every payload field at the exact byte position
//! rustc chose, inserting `[N x i8]` filler for the gaps:
//!
//! ```text
//! #[repr(u32)] enum E { A(u32), B(f32), C }   // rustc: 8 bytes,
//!                                             // tag at 0, payloads at 4
//! LLVM: { i32, i32 }   ; slot 0 = tag, slot 1 = A's payload
//!                      ; B's f32 also lives at byte 4 but has a
//!                      ; different type, so it is read/written through
//!                      ; memory instead of owning a slot
//! ```
//!
//! Because the bytes match rustc exactly, enum data can cross the
//! host/device boundary safely. The tag slot stores the variant's
//! DECLARED discriminant value (`enum E { A = 7 }` stores 7), not its
//! position. See `build_enum_slot_map` in this module for the full
//! story.
//!
//! # Function Type Conversion
//!
//! Function types undergo ABI transformations:
//!
//! - Slice arguments are flattened to `(ptr, len)` pairs
//! - Struct arguments are flattened to individual fields
//! - Empty tuple return type becomes void
//!
//! This matches the C ABI for GPU kernels.

mod conversion;
mod enum_layout;
mod func_abi;
mod global_layout;
mod layout;
mod pointer_storage;
mod struct_layout;
#[cfg(test)]
mod test_support;
mod union_storage;

pub use conversion::{convert_type, is_zero_sized_type};
pub(crate) use conversion::{make_disjoint_slice_struct, make_slice_struct};
pub(crate) use enum_layout::{
    EnumSlotMap, build_enum_slot_map, convert_enum_to_llvm, enum_unmodeled_in_memory,
    find_unmodeled_enum_in_abi,
};
pub(crate) use func_abi::{
    TransparentScalarAbiInfo, convert_grid_constant_storage_type, packed_shared_internal_abi_info,
    transparent_scalar_abi_info, transparent_scalar_field,
};
pub use func_abi::{convert_function_type, is_kernel_func};
// Surface parity with the pre-split `types.rs`: these four are currently
// consumed only inside `func_abi` itself or under `cfg(test)` (e.g.
// `transparent_scalar_llvm_type` from the `lowering.rs` tests), so the
// re-export trips `unused_imports` in non-test builds.
#[allow(unused_imports)]
pub(crate) use func_abi::{
    MAX_PACKED_SHARED_INTERNAL_ABI_ARRAY_REWRITE_LEAVES, PackedSharedInternalAbiInfo,
    TransparentScalarLayer, transparent_scalar_llvm_type,
};
pub(crate) use global_layout::{
    validate_initialized_global_layout, validate_relocated_initialized_global_layout,
};
pub(crate) use layout::{
    llvm_byte_faithful_twin, llvm_type_contains_i1, llvm_type_is_byte_faithful,
    llvm_type_size_align, mir_element_stride, mir_type_abi_align, natural_struct_layout,
};
pub(crate) use pointer_storage::{
    llvm_packed_struct_contains_pointer_in_address_space,
    llvm_type_contains_pointer_in_address_space,
};
pub use struct_layout::struct_value_lowering_is_byte_faithful;
pub(crate) use struct_layout::{StructLayoutInfo, StructSlotMap, build_struct_slot_map};
pub(crate) use union_storage::build_union_storage_type;
