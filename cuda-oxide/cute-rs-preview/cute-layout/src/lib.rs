/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Describes where each logical tensor cell lives in linear memory.
//!
//! Parse `shape:stride`, then evaluate a coordinate:
//!
//! ```
//! use cute_layout::{IntTuple, Layout};
//! let layout: Layout = "(2,3):(3,1)".parse().unwrap();
//! assert_eq!(layout.checked_call(&IntTuple::Leaf(1)), Some(3));
//! ```
//!
//! Here flat coordinate `1` selects `(row=1, column=0)`; the walk is explained below.
//!
//! # Modes
//!
//! GPU memory is a one-dimensional list of bytes, while kernels usually think
//! in coordinates such as `(row, column)`. A [`Layout`] is the bridge between
//! those two views. It pairs a **shape** with a **stride**:
//!
//! ```text
//! layout  (2,3):(3,1)
//!          shape  stride
//!
//! offset(row, column) = row * 3 + column * 1
//!
//!                    column
//!                 0    1    2
//!               +----+----+----+
//!       row 0   |  0 |  1 |  2 |   numbers are linear element offsets
//!               +----+----+----+
//!       row 1   |  3 |  4 |  5 |
//!               +----+----+----+
//! ```
//!
//! A coordinate identifies one logical position. In this flat 2D example it
//! has two numbers, `(row, column)`. Each number needed to identify a position
//! is called a **mode**. For this flat 2D layout, those numbers are the row and
//! column.
//!
//! Put another way, a mode is one number in the coordinate that you can change
//! while keeping the other numbers fixed:
//!
//! ```text
//!                     change column
//!                (0,1) ------------> (0,2)
//!                  |
//!                  | change row
//!                  v
//!                (1,1)
//! ```
//!
//! In this flat example, each mode has one shape value and one matching stride:
//!
//! ```text
//!              row mode   column mode
//! shape            2           3
//! stride           3           1
//! ```
//!
//! The row mode is described by shape `2` and stride `3`; the column mode is
//! described by shape `3` and stride `1`. Shape and stride entries describe
//! the modes; they are not extra modes. The row shape allows values `0` and
//! `1`, while the column shape allows values `0`, `1`, and `2`. The matching
//! stride says how far the linear offset moves when that coordinate number
//! increases by one. Here, moving to the next row adds `3` to the offset,
//! while moving to the next column adds `1`.
//!
//! Nested parentheses group coordinate parts without changing this rule. For
//! example, `((2,4),3):((1,2),8)` takes a coordinate `((a,b),c)`. It has two
//! top-level modes:
//!
//! ```text
//! mode 0: (a,b), described by (2,4):(1,2)
//! mode 1: c,     described by       3:8
//! ```
//!
//! ## Flat-coordinate walk
//!
//! A layout can also be called with one flat coordinate instead of a tuple.
//! CuTe unpacks flat coordinates by changing the **leftmost mode first**. This
//! fixed order is called the flat-coordinate walk:
//!
//! ```text
//! shape                   (2,3)
//! flat coordinate          0     1     2     3     4     5
//! unpacked coordinate    (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)
//! ```
//!
//! The shape decides this walk; the strides decide the resulting offsets. For
//! `(2,3):(3,1)`, the coordinates above map to offsets `0, 3, 1, 4, 2, 5`.
//! This order is sometimes called *colexicographic*: in plain English, the
//! leftmost coordinate number changes fastest.
//!
//! Layout algebra often preserves the mapping from each **flat coordinate**
//! to its offset while changing how tuple coordinates are grouped. For
//! example, these layouts have the same flat-coordinate mapping:
//!
//! ```text
//! flat coordinate       0  1  2  3  4  5  6  7
//! (2,4):(1,2) offset    0  1  2  3  4  5  6  7
//!       8:1 offset      0  1  2  3  4  5  6  7
//! ```
//!
//! Their native tuple-shaped interfaces differ: the first names a position as
//! `(a,b)`, while the second names it with one number. For example, `(1,3)` in
//! the first layout and `7` in the second both reach offset `7`. Both layouts
//! still accept a flat coordinate through [`Layout::call`].
//!
//! Layout offsets do not carry a unit. Most layouts count elements; layouts
//! used at a memory boundary may count bytes. Code that converts between the
//! two must state the unit explicitly.
//!
//! # Tiles and operations
//!
//! This crate also contains the operations needed to build GPU tiles:
//!
//! ```text
//! tensor layout
//!      |
//!      +-- divide/product --> repeated tiles
//!      +-- thread/value   --> cells owned by each thread
//!      +-- compose        --> final memory addresses
//!      +-- swizzle        --> shared-memory bank-friendly addresses
//! ```
//!
//! A **tile** is a small rectangular part of a tensor handled as one unit. A
//! **thread-value (TV) layout** assigns every tile cell to a thread and to one
//! of that thread's values. The validation code checks that these pieces cover
//! the tile exactly before the compiler emits memory operations.
//!
//! # Atoms
//!
//! An **atom** is a compile-time description of one indivisible operation used
//! as a building block for a larger operation. It is not itself an intrinsic
//! and does not perform the operation. "Indivisible" means the threads and
//! values described by the atom must participate together as one unit.
//!
//! A **copy atom** is one indivisible copy transaction used as a building block
//! for a larger cooperative copy. `CopyG2S<16>` is a compile-time description
//! of one thread copying 16 bytes with `cp.async`; it does not perform the copy
//! itself. More generally, an atom can also describe a compute operation such
//! as a matrix multiply-accumulate (MMA).
//!
//! ```text
//! CopyG2S<16> marker -- compiler turns it into --> one 16-byte cp.async transaction
//! many such transactions ------------------> one cooperative tile copy
//! ```
//!
//! The crate is `no_std` and shared across the compiler boundary:
//!
//! - `cute-rs` uses the types to describe a kernel's static layout; and
//! - the compiler evaluates the same layouts when it creates addresses.
//!
//! Keeping one implementation on both sides prevents the Rust API and emitted
//! GPU code from disagreeing. Tests compare complete mappings with a pinned
//! cutegen revision as an independent reference.
//!
//! # Module navigation
//!
//! [`mod@tuple`] owns nested integer tuples, [`layout`] constructs and evaluates
//! layouts, and [`parse`] supplies their text representation. [`algebra`]
//! groups mapping transformations, tiling, thread/value ownership and alignment.
//! [`composed`], [`swizzle`], [`predication`] and [`validation`] handle their
//! corresponding memory-mapping and safety checks. Foundational types remain
//! available at the crate root through reexports of the same definitions.

#![no_std]

extern crate alloc;

pub mod algebra;
pub mod composed;
pub mod layout;
pub mod parse;
pub mod predication;
pub mod swizzle;
pub mod tuple;
pub mod validation;

pub use composed::{ComposedLayout, ComposedLayoutError, OffsetUnit};
pub use layout::{Layout, crd2idx};
pub use parse::ParseLayoutError;
pub use predication::{
    AtomCoord, GmemMatrixBounds, PredicatedSource, PredicationError, PredicationResult,
    SafeSourceOffset, plan_uniform_cp_async_source,
};
pub use swizzle::Swizzle;
pub use tuple::IntTuple;
pub use validation::{
    CooperativeCopyPlan, CooperativePlanError, MAX_COOPERATIVE_THREADS,
    MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE, MAX_STATIC_VALIDATION_BYTES,
    MAX_STATIC_VALIDATION_ELEMENTS, ValidationError, ValidationResult, tma_phase_alignment_bytes,
    validate_atom_compatibility, validate_compact_thread_layout, validate_cooperative_copy_plan,
    validate_ldmatrix_source, validate_per_thread_atom_contiguity, validate_tma_encodable,
    validate_tv_exact_coverage, validate_tv_layout,
};

/// Layout evaluation, representation and parsing contracts.
#[cfg(test)]
mod tests;
