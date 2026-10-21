/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builds new layouts from existing ones without generating runtime index math.
//!
//! Start with the basic idea from the crate root: a [`Layout`] maps a logical
//! coordinate to a linear offset. This module changes *descriptions* of those
//! maps at compile time. It never moves memory itself.
//!
//! The main operations answer practical tiling questions:
//!
//! ```text
//! coalesce     Can adjacent modes be written as one simpler mode?
//! composition  What is the final map after passing through two maps?
//! complement   Which offsets does a layout leave unused?
//! inverse      Which coordinate reaches a requested offset?
//! divide       Where am I inside a tile, and which tile am I in?
//! product      How do I repeat this small layout?
//! TV layout    Which thread owns each value in a tile?
//! ```
//!
//! A [**mode**](crate#modes) is one number in a coordinate that can change
//! while the other numbers stay fixed. In the examples below, `4:2` means
//! "four coordinate values, with an offset step of two":
//!
//! ```text
//! coordinate       0  1  2  3
//! offset            0  2  4  6
//! ```
//!
//! `(2,3):(3,1)` has two modes: the pairs `2:3` and `3:1`. A **flat walk**
//! numbers all tuple coordinates with the left-hand mode changing fastest:
//!
//! ```text
//! flat coordinate   0      1      2      3      4      5
//! tuple coordinate (0,0)  (1,0)  (0,1)  (1,1)  (0,2)  (1,2)
//! offset             0      3      1      4      2      5
//! ```
//!
//! A **tile** is a fixed-size part of a larger tensor. A **TV layout** maps
//! `(thread index, value index)` to a cell inside that tile.
//!
//! Why the compiler needs algebra rather than recovered scalar arithmetic:
//!
//! ```text
//! Rust const-generic layouts
//!            |
//!            v
//!     these operations
//!            |
//!            v
//! exact per-thread addresses + proof of contiguous vector accesses
//! ```
//!
//! Some shape-and-stride combinations cannot be represented by the supported
//! layout form; those operations return [`AlgebraError`]. A tile size that does
//! not divide evenly is different: the operation may describe a rounded-up
//! final tile, and [`crate::predication`] handles its extra edge positions.
//! Untrusted compiler input is additionally checked and bounded by
//! [`crate::validation`] before it reaches expensive enumeration.
//!
//! The implementation follows cutegen, the C++ reference implementation.
//! Exhaustive small-layout tests check the defining equations, and a pinned
//! cutegen oracle compares full address maps so that formatting or
//! implementation details cannot hide a semantic difference.

/// Static vector-access alignment derived from layout strides.
pub mod alignment;
/// Simplification, composition, inverse maps, and reshaping.
pub mod mapping;
/// Thread/value ownership for static tiles.
pub mod thread_value;
/// Tile division and repeated-layout products.
pub mod tiling;

pub use alignment::max_alignment;
pub use mapping::{coalesce, complement, composition, left_inverse, right_inverse, with_shape};
pub use thread_value::make_layout_tv;
pub use tiling::{
    blocked_product, logical_divide, logical_product, raked_product, zipped_divide_by_shape,
};

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::{IntTuple, Layout};

/// Explains why a requested layout transformation is not mathematically valid.
///
/// For example, splitting six cells into tiles of four leaves an incomplete
/// tile. Operations that require exact division report an error instead of
/// silently constructing a wrong address map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraError(
    #[doc = "Explanation of the unsupported or inconsistent layout transformation."] pub String,
);

impl fmt::Display for AlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layout algebra error: {}", self.0)
    }
}

impl core::error::Error for AlgebraError {}

/// Layout-algebra result with a descriptive transformation error.
type Result<T> = core::result::Result<T, AlgebraError>;

/// Flattens nested modes into `(shape extent, stride)` pairs from left to right.
///
/// `((2,3),4):((12,4),1)` becomes `[(2,12), (3,4), (4,1)]`.
fn flat_modes(l: &Layout) -> Vec<(i64, i64)> {
    l.shape
        .leaves()
        .into_iter()
        .zip(l.stride.leaves())
        .collect()
}

/// Builds the simplest tuple structure that can hold a flat mode list.
///
/// No modes produce the neutral one-cell map `1:0`; one mode does not need
/// parentheses; two or more modes become matching shape and stride tuples.
fn layout_from_flat(modes: Vec<(i64, i64)>) -> Layout {
    match modes.len() {
        0 => Layout {
            shape: IntTuple::Leaf(1),
            stride: IntTuple::Leaf(0),
        },
        1 => Layout {
            shape: IntTuple::Leaf(modes[0].0),
            stride: IntTuple::Leaf(modes[0].1),
        },
        _ => Layout {
            shape: IntTuple::Tuple(modes.iter().map(|(s, _)| IntTuple::Leaf(*s)).collect()),
            stride: IntTuple::Tuple(modes.iter().map(|(_, d)| IntTuple::Leaf(*d)).collect()),
        },
    }
}

/// Definitional equations and known layout maps, grouped by algebra family.
#[cfg(test)]
mod tests;
