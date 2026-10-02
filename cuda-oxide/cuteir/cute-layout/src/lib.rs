/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Describes where each logical tensor cell lives in linear memory.
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
//! for a larger cooperative copy. `CpAsync<16>` is a compile-time description
//! of one thread copying 16 bytes with `cp.async`; it does not perform the copy
//! itself. More generally, an atom can also describe a compute operation such
//! as a matrix multiply-accumulate (MMA).
//!
//! ```text
//! CpAsync<16> marker -- compiler turns it into --> one 16-byte cp.async transaction
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

#![no_std]

extern crate alloc;

pub mod algebra;
pub mod composed;
pub mod predication;
pub mod swizzle;
pub mod validation;

pub use composed::{ComposedLayout, ComposedLayoutError, OffsetUnit};
pub use predication::{
    AtomCoord, GmemMatrixBounds, PredicatedSource, PredicationError, PredicationResult,
    SafeSourceOffset, plan_uniform_cp_async_source,
};
pub use swizzle::Swizzle;
pub use validation::{
    CooperativeCopyPlan, CooperativePlanError, MAX_COOPERATIVE_THREADS,
    MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE, MAX_STATIC_VALIDATION_BYTES,
    MAX_STATIC_VALIDATION_ELEMENTS, ValidationError, ValidationResult, tma_phase_alignment_bytes,
    validate_atom_compatibility, validate_compact_thread_layout, validate_cooperative_copy_plan,
    validate_ldmatrix_source, validate_per_thread_atom_contiguity, validate_tma_encodable,
    validate_tv_exact_coverage, validate_tv_layout,
};

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

/// One integer, or an ordered group of integers with preserved parentheses.
///
/// Shapes, strides, and coordinates all use this type so their structures can
/// match:
///
/// ```text
/// value                                      printed form
/// Leaf(4)                                    4
/// Tuple([Leaf(2), Leaf(3)])                  (2,3)
/// Tuple([Tuple([Leaf(2), Leaf(3)]), Leaf(4)]) ((2,3),4)
/// ```
///
/// When used as a layout's shape or stride, each top-level item describes one
/// [mode](crate#modes). Thus `((2,3),4):((12,4),1)` has two top-level modes:
/// `(2,3):(12,4)` and `4:1`. The first happens to contain two nested
/// coordinate parts. Keeping the parentheses lets layout operations preserve
/// that grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntTuple {
    /// A single integer with no nested structure.
    Leaf(i64),
    /// An ordered group of integers or further groups.
    Tuple(Vec<IntTuple>),
}

impl IntTuple {
    /// Multiplies all contained integers.
    ///
    /// When this tuple is a shape, that product is its logical cell count:
    /// `(2,(3,4))` contains `2 * 3 * 4 = 24` cells. This low-level method does
    /// not reject zero or negative values and does not protect against
    /// overflow; [`Layout::checked_size`] does.
    pub fn size(&self) -> i64 {
        match self {
            IntTuple::Leaf(v) => *v,
            IntTuple::Tuple(ts) => ts.iter().map(IntTuple::size).product(),
        }
    }

    /// Reports whether two tuples have the same parenthesis structure.
    ///
    /// The integer values do not matter:
    ///
    /// ```text
    /// (2,(3,4)) and (8,(5,6))  -> congruent: both look like (_,(_,_))
    /// (2,(3,4)) and ((8,5),6)  -> not congruent: the grouping differs
    /// ```
    ///
    /// A [`Layout`] requires congruent shape and stride tuples so every shape
    /// number has a matching stride number in the same position.
    pub fn congruent(&self, other: &IntTuple) -> bool {
        match (self, other) {
            (IntTuple::Leaf(_), IntTuple::Leaf(_)) => true,
            (IntTuple::Tuple(a), IntTuple::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.congruent(y))
            }
            _ => false,
        }
    }

    /// Returns every integer from left to right and discards parentheses.
    ///
    /// ```text
    /// input:  ((2,3),4)
    /// output: [2, 3, 4]    parentheses are gone
    /// ```
    ///
    /// Algebra operations use this when they need the numbers but not the
    /// nested [mode](crate#modes) structure.
    pub fn leaves(&self) -> Vec<i64> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves(&self, out: &mut Vec<i64>) {
        match self {
            IntTuple::Leaf(v) => out.push(*v),
            IntTuple::Tuple(ts) => ts.iter().for_each(|t| t.collect_leaves(out)),
        }
    }

    /// Counts top-level [modes](crate#modes); one ungrouped integer has rank 1.
    ///
    /// ```text
    /// 4          -> [ 4 ]          rank 1
    /// (2,3)      -> [ 2 | 3 ]      rank 2
    /// ((2,3),4)  -> [ (2,3) | 4 ]  rank 2
    /// ```
    ///
    /// The inner `2` and `3` in the last example are nested inside its first
    /// top-level mode, so they do not increase the top-level rank.
    pub fn rank(&self) -> usize {
        match self {
            IntTuple::Leaf(_) => 1,
            IntTuple::Tuple(ts) => ts.len(),
        }
    }

    /// Puts a flat leaf list back into this tuple's tree structure.
    ///
    /// This is an internal helper. It panics when the list does not contain
    /// exactly enough values for the tree.
    fn restructure(&self, leaves: &mut alloc::slice::Iter<'_, i64>) -> IntTuple {
        match self {
            IntTuple::Leaf(_) => IntTuple::Leaf(*leaves.next().expect("leaf count mismatch")),
            IntTuple::Tuple(ts) => {
                IntTuple::Tuple(ts.iter().map(|t| t.restructure(leaves)).collect())
            }
        }
    }
}

/// Converts a logical coordinate into a linear offset using `shape:stride`.
///
/// With a tuple coordinate, multiply each coordinate number by the matching
/// stride and add the results:
///
/// ```text
/// layout = (2,3):(3,1)
/// coord  = (1,2)            row 1, column 2
///
/// offset = 1*3 + 2*1 = 5
/// ```
///
/// A single integer is also accepted for a tuple shape. It follows the
/// [flat-coordinate walk](crate#flat-coordinate-walk): unpack it by changing
/// the leftmost [mode](crate#modes) fastest, then apply the strides.
///
/// ```text
/// layout                    (2,3):(3,1)
/// flat coordinate           0     1     2     3     4     5
/// unpacked coordinate     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)
/// returned offset            0     3     1     4     2     5
/// ```
///
/// Notice that flat coordinate `1` returns offset `3`, not offset `1`: the
/// shape controls how the flat number is unpacked, while the strides control
/// where that coordinate lives.
///
/// # Panics and unchecked input
///
/// This low-level function assumes that shape and stride have the same
/// structure, that a tuple coordinate matches them, and that coordinates and
/// arithmetic are valid. It may panic when those assumptions are broken.
/// Compiler attributes from outside this crate must go through the checked
/// validators instead. [`Layout::checked_call`] is the checked evaluator.
pub fn crd2idx(coord: &IntTuple, shape: &IntTuple, stride: &IntTuple) -> i64 {
    match (coord, shape, stride) {
        (IntTuple::Leaf(c), IntTuple::Leaf(_s), IntTuple::Leaf(d)) => {
            debug_assert!(*c < *_s, "coordinate out of bounds");
            c * d
        }
        (IntTuple::Tuple(cs), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            debug_assert!(cs.len() == ss.len() && ss.len() == ds.len());
            cs.iter()
                .zip(ss.iter().zip(ds))
                .map(|(c, (s, d))| crd2idx(c, s, d))
                .sum()
        }
        (IntTuple::Leaf(c), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            debug_assert!(ss.len() == ds.len());
            let mut rem = *c;
            let mut idx = 0;
            for (i, (s, d)) in ss.iter().zip(ds).enumerate() {
                let sz = s.size();
                // The final mode receives the whole remaining quotient. For
                // an out-of-range flat coordinate this extends that mode
                // instead of silently wrapping it back into the shape.
                let c_i = if i + 1 == ss.len() { rem } else { rem % sz };
                idx += crd2idx(&IntTuple::Leaf(c_i), s, d);
                rem /= sz;
            }
            idx
        }
        _ => panic!("crd2idx: coord/shape/stride shapes are incompatible"),
    }
}

/// Safe evaluator used after the compiler decodes an untrusted layout.
///
/// Unlike [`crd2idx`], this returns `None` for an out-of-range coordinate, an
/// incompatible tuple tree, an invalid shape, or overflowing arithmetic.
fn checked_crd2idx(coord: &IntTuple, shape: &IntTuple, stride: &IntTuple) -> Option<i64> {
    match (coord, shape, stride) {
        (IntTuple::Leaf(c), IntTuple::Leaf(s), IntTuple::Leaf(d)) => {
            (*c >= 0 && *c < *s).then_some(())?;
            c.checked_mul(*d)
        }
        (IntTuple::Tuple(cs), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            (cs.len() == ss.len() && ss.len() == ds.len()).then_some(())?;
            cs.iter()
                .zip(ss.iter().zip(ds))
                .try_fold(0i64, |sum, (c, (s, d))| {
                    sum.checked_add(checked_crd2idx(c, s, d)?)
                })
        }
        (IntTuple::Leaf(c), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            (ss.len() == ds.len() && *c >= 0).then_some(())?;
            let total = checked_tuple_size(shape)?;
            (*c < total).then_some(())?;
            let mut rem = *c;
            let mut idx = 0i64;
            for (i, (s, d)) in ss.iter().zip(ds).enumerate() {
                let size = checked_tuple_size(s)?;
                (size > 0).then_some(())?;
                let mode_coord = if i + 1 == ss.len() { rem } else { rem % size };
                idx = idx.checked_add(checked_crd2idx(&IntTuple::Leaf(mode_coord), s, d)?)?;
                rem /= size;
            }
            Some(idx)
        }
        _ => None,
    }
}

fn checked_tuple_size(tuple: &IntTuple) -> Option<i64> {
    match tuple {
        IntTuple::Leaf(value) => (*value > 0).then_some(*value),
        IntTuple::Tuple(items) => items.iter().try_fold(1i64, |size, item| {
            size.checked_mul(checked_tuple_size(item)?)
        }),
    }
}

/// Maps logical coordinates to linear offsets.
///
/// Write a layout as `shape:stride`. In a flat layout, each shape number and
/// the stride in the same position describe one [mode](crate#modes):
///
/// ```text
/// Layout (2,3):(3,1)
///
/// mode       shape says                 stride says
/// row        row can be 0 or 1          changing row adds 3
/// column     column can be 0, 1, or 2   changing column adds 1
///
/// offset(row, column) = row*3 + column*1
/// ```
///
/// Shape and stride must have the same parenthesis structure. In
/// `((2,4),3):((1,2),8)`, the nested pair `(2,4):(1,2)` is one grouped
/// top-level mode. The offset is usually an element index, but it can be a byte
/// index when the caller explicitly treats the strides as bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    /// How many coordinate values each [mode](crate#modes) accepts.
    ///
    /// In `(2,3):(3,1)`, shape `(2,3)` means two rows and three columns.
    pub shape: IntTuple,
    /// How much the offset changes when each matching coordinate increases by
    /// one.
    ///
    /// In `(2,3):(3,1)`, stride `(3,1)` means “add 3 for the next row” and
    /// “add 1 for the next column.” Strides describe modes; they are not
    /// additional [modes](crate#modes).
    pub stride: IntTuple,
}

impl Layout {
    /// Creates `shape:stride` after checking that their parentheses match.
    ///
    /// ```text
    /// shape (2,3) + stride (3,1) -> (2,3):(3,1)  valid
    /// shape (2,3) + stride 1     -> (2,3):1      invalid
    /// ```
    ///
    /// The invalid case has two shape entries but no two-entry stride tuple.
    /// This constructor checks structure only; it does not reject zero or
    /// negative extents. Use [`Layout::checked_size`] or the crate validators
    /// when accepting untrusted layouts.
    ///
    /// # Panics
    ///
    /// Panics when `shape` and `stride` do not have identical nesting.
    pub fn new(shape: IntTuple, stride: IntTuple) -> Self {
        assert!(
            shape.congruent(&stride),
            "shape and stride must be congruent"
        );
        Layout { shape, stride }
    }

    /// Returns the logical cell count: the product of all shape numbers.
    ///
    /// `(2,3):(3,1)` has `2 * 3 = 6` cells. Strides do not affect this count:
    /// `4:2` has four logical cells even though they occupy offsets
    /// `0, 2, 4, 6`. This unchecked form may return a non-positive value or
    /// overflow for an invalid shape; [`Layout::checked_size`] rejects those
    /// cases.
    pub fn size(&self) -> i64 {
        self.shape.size()
    }

    /// Returns the positive logical cell count, or `None` for an invalid size.
    ///
    /// ```text
    /// shape (2,3)       -> Some(6)
    /// shape (2,0)       -> None       zero extent
    /// shape (i64::MAX,2) -> None      product does not fit in i64
    /// ```
    ///
    /// Strides are irrelevant to the logical cell count.
    pub fn checked_size(&self) -> Option<i64> {
        checked_tuple_size(&self.shape)
    }

    /// Maps one tuple or flat coordinate to its linear offset.
    ///
    /// ```text
    /// layout (2,3):(3,1)
    ///
    /// tuple coordinate (1,2) -> 1*3 + 2*1 -> offset 5
    /// flat coordinate  4     -> unpacked as (0,2) -> offset 2
    /// ```
    ///
    /// Flat coordinates use the [leftmost-mode-fastest walk](crate#flat-coordinate-walk).
    /// This unchecked convenience method has the same input requirements as
    /// [`crd2idx`].
    pub fn call(&self, coord: &IntTuple) -> i64 {
        crd2idx(coord, &self.shape, &self.stride)
    }

    /// Safely maps one tuple or flat coordinate to its linear offset.
    ///
    /// ```text
    /// layout (2,3):(3,1)
    /// coordinate (1,2) -> Some(5)
    /// coordinate (2,0) -> None     row 2 is outside a two-row shape
    /// flat coordinate 6 -> None    the six cells are numbered 0 through 5
    /// ```
    ///
    /// `None` also means a tuple has the wrong parenthesis structure, the
    /// shape is invalid, or the computed offset does not fit in `i64`.
    pub fn checked_call(&self, coord: &IntTuple) -> Option<i64> {
        checked_crd2idx(coord, &self.shape, &self.stride)
    }

    /// Creates a one-dimensional layout of `n` adjacent cells: `n:1`.
    ///
    /// ```text
    /// contiguous(4) = 4:1
    /// coordinate          0  1  2  3
    /// offset              0  1  2  3
    /// ```
    ///
    /// This constructor does not validate `n`; [`Layout::checked_size`]
    /// rejects zero or negative extents.
    pub fn contiguous(n: i64) -> Self {
        Layout {
            shape: IntTuple::Leaf(n),
            stride: IntTuple::Leaf(1),
        }
    }

    /// Reports whether every flat coordinate maps to the same-numbered offset.
    ///
    /// ```text
    /// 4:1                    4:2
    /// coord   0 1 2 3        coord   0 1 2 3
    /// offset  0 1 2 3        offset  0 2 4 6
    ///         identity               not identity
    /// ```
    ///
    /// A multi-mode layout can also be an identity map. Its flat-coordinate
    /// walk must visit adjacent offsets in order:
    ///
    /// ```text
    /// layout                 (2,3):(1,2)
    /// flat coordinate         0     1     2     3     4     5
    /// tuple coordinate      (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)
    /// offset                  0     1     2     3     4     5
    /// ```
    ///
    /// This checks the flat-coordinate mapping; it does not say that the
    /// original tuple coordinate interface is one-dimensional. An identity
    /// mapping describes one adjacent memory interval, so the compiler may be
    /// able to replace scalar accesses with one vector access. The method
    /// checks every coordinate and is intended for small compile-time tiles.
    pub fn is_identity_map(&self) -> bool {
        let Some(size) = self.checked_size() else {
            return false;
        };
        (0..size)
            .all(|coordinate| self.checked_call(&IntTuple::Leaf(coordinate)) == Some(coordinate))
    }

    /// Returns the length of the offset interval needed to contain the layout.
    ///
    /// This is one more than the greatest reached offset, so skipped offsets
    /// still count:
    ///
    /// ```text
    /// layout 4:2
    /// coordinate        0  1  2  3
    /// reached offset    0  2  4  6
    /// needed positions   0  1  2  3  4  5  6  -> cosize 7
    ///
    /// size = 4 logical cells, but cosize = 7 offset positions
    /// ```
    ///
    /// You may see the reached offsets called the *image* and their containing
    /// interval called the *codomain* in layout-algebra material. The plain
    /// rule is the one above: `cosize` is greatest reached offset plus one.
    /// This calculation assumes non-negative strides; do not use it to bound
    /// a negative-stride layout.
    pub fn cosize(&self) -> i64 {
        let n = self.size();
        if n == 0 {
            return 0;
        }
        self.call(&IntTuple::Leaf(n - 1)) + 1
    }

    /// Builds a compact layout with the smallest stride on the leftmost
    /// coordinate number.
    ///
    /// For a `(row,column)` shape, this is column-major: rows within one column
    /// are adjacent, then the next column begins.
    ///
    /// ```text
    /// shape (2,3), init 1  ->  (2,3):(1,2)
    ///
    ///                              column
    ///                            0   1   2
    ///                          +---+---+---+
    ///                    row 0 | 0 | 2 | 4 |  numbers are offsets
    ///                          +---+---+---+
    ///                    row 1 | 1 | 3 | 5 |
    ///                          +---+---+---+
    ///
    /// flat-coordinate walk  (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
    /// returned offsets        0,     1,     2,     3,     4,     5
    /// ```
    ///
    /// `init` is the distance between adjacent cells in that walk. With
    /// `init = 4`, the result is `(2,3):(4,8)` and the flat coordinates map to
    /// offsets `0, 4, 8, 12, 16, 20`. Shape extents and stride arithmetic are
    /// unchecked; validate externally supplied layouts before using them.
    pub fn col_major_with(shape: IntTuple, init: i64) -> Layout {
        let leaves = shape.leaves();
        let mut strides = Vec::with_capacity(leaves.len());
        let mut running = init;
        for s in &leaves {
            strides.push(running);
            running *= s;
        }
        let stride = shape.restructure(&mut strides.iter());
        Layout { shape, stride }
    }

    /// Builds a compact column-major layout with unit spacing.
    ///
    /// This is [`Layout::col_major_with`] with `init = 1`. For example, shape
    /// `(2,3)` becomes `(2,3):(1,2)`.
    pub fn col_major(shape: IntTuple) -> Layout {
        Layout::col_major_with(shape, 1)
    }

    /// Splits one layout into one sublayout per top-level [mode](crate#modes).
    ///
    /// ```text
    /// original layout            (2,3):(3,1)
    ///
    /// top-level position       0             1
    /// shape entry              2             3
    /// matching stride          3             1
    /// returned sublayout       2:3           3:1
    ///
    /// result                  [2:3, 3:1]
    /// ```
    ///
    /// Thus “neighboring modes” means adjacent shape/stride pairs inside the
    /// same original layout—not two separate shapes such as `(2,3)` and
    /// `(3,4)`. For a nested layout,
    /// `((2,4),3):((1,2),8)` becomes `[(2,4):(1,2), 3:8]`; the parentheses
    /// inside the first mode stay intact. A rank-1 layout returns a one-item
    /// vector containing itself.
    pub fn modes(&self) -> Vec<Layout> {
        match (&self.shape, &self.stride) {
            (IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => ss
                .iter()
                .zip(ds)
                .map(|(s, d)| Layout {
                    shape: s.clone(),
                    stride: d.clone(),
                })
                .collect(),
            _ => vec![self.clone()],
        }
    }

    /// Joins sublayouts as adjacent top-level [modes](crate#modes) of one layout.
    ///
    /// ```text
    /// input sublayouts       [2:1, 4:2]
    /// shape entries          [ 2,   4 ] -> (2,4)
    /// stride entries         [ 1,   2 ] -> (1,2)
    /// result                              (2,4):(1,2)
    /// ```
    ///
    /// This reverses [`Layout::modes`]. One input remains unwrapped. No inputs
    /// produce `1:0`: a one-cell mode has shape extent `1`, so its only valid
    /// coordinate is `0`, and `0 * 0` gives offset `0`.
    pub fn from_modes(modes: Vec<Layout>) -> Layout {
        match modes.len() {
            0 => Layout {
                shape: IntTuple::Leaf(1),
                stride: IntTuple::Leaf(0),
            },
            1 => modes.into_iter().next().unwrap(),
            _ => Layout {
                shape: IntTuple::Tuple(modes.iter().map(|m| m.shape.clone()).collect()),
                stride: IntTuple::Tuple(modes.into_iter().map(|m| m.stride).collect()),
            },
        }
    }
}

// Text form used in diagnostics and in the cutegen test oracle:
//
//   shape:stride
//   (8,32):(32,1)
//   ((2,4),32):((128,32),1)  // nested mode on the left
//
// Parentheses preserve the tuple tree; they are not merely decoration.

impl fmt::Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntTuple::Leaf(v) => write!(f, "{v}"),
            IntTuple::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// Indicates that text was not a complete, structurally valid layout or tuple.
///
/// Examples include a missing `:`, an unclosed tuple, or shape and stride
/// trees that do not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLayoutError;

impl fmt::Display for ParseLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CuTe layout/tuple syntax")
    }
}

impl core::error::Error for ParseLayoutError {}

fn parse_int_tuple(s: &str) -> Result<(IntTuple, &str), ParseLayoutError> {
    let s = s.trim_start();
    if let Some(mut rest) = s.strip_prefix('(') {
        let mut items = Vec::new();
        loop {
            let (item, r) = parse_int_tuple(rest)?;
            items.push(item);
            let r = r.trim_start();
            if let Some(r) = r.strip_prefix(',') {
                rest = r;
            } else if let Some(r) = r.strip_prefix(')') {
                return Ok((IntTuple::Tuple(items), r));
            } else {
                return Err(ParseLayoutError);
            }
        }
    } else {
        let end = s
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '-'))
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let value: i64 = s[..end].parse().map_err(|_| ParseLayoutError)?;
        Ok((IntTuple::Leaf(value), &s[end..]))
    }
}

impl FromStr for IntTuple {
    type Err = ParseLayoutError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (t, rest) = parse_int_tuple(s)?;
        if rest.trim().is_empty() {
            Ok(t)
        } else {
            Err(ParseLayoutError)
        }
    }
}

impl FromStr for Layout {
    type Err = ParseLayoutError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (shape, rest) = parse_int_tuple(s)?;
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':').ok_or(ParseLayoutError)?;
        let (stride, rest) = parse_int_tuple(rest)?;
        if !rest.trim().is_empty() {
            return Err(ParseLayoutError);
        }
        if !shape.congruent(&stride) {
            return Err(ParseLayoutError);
        }
        Ok(Layout { shape, stride })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn t(v: &[i64]) -> IntTuple {
        IntTuple::Tuple(v.iter().map(|x| IntTuple::Leaf(*x)).collect())
    }

    #[test]
    fn row_major_8x32() {
        // Eight rows of 32 adjacent values: moving one row adds 32 and moving
        // one column adds 1.
        let l = Layout::new(t(&[8, 32]), t(&[32, 1]));
        assert_eq!(l.size(), 256);
        assert_eq!(l.call(&t(&[0, 0])), 0);
        assert_eq!(l.call(&t(&[1, 0])), 32);
        assert_eq!(l.call(&t(&[3, 7])), 3 * 32 + 7);
    }

    #[test]
    fn integer_coord_is_colexicographic() {
        // The leftmost mode changes first, so flat coordinate 5 becomes
        // `(row 5, column 0)` and maps to `5 * 32`.
        let l = Layout::new(t(&[8, 32]), t(&[32, 1]));
        assert_eq!(l.call(&IntTuple::Leaf(5)), 160);
        // After all eight rows, flat coordinate 9 becomes `(1,1)`.
        assert_eq!(l.call(&IntTuple::Leaf(9)), 33);
    }

    #[test]
    fn display_and_parse_roundtrip() {
        use alloc::string::ToString;
        for text in ["(8,32):(32,1)", "4:1", "((2,4),32):((128,32),1)"] {
            let l: Layout = text.parse().unwrap();
            assert_eq!(l.to_string(), text);
        }
        assert!("(8,32):(32)".parse::<Layout>().is_err()); // not congruent
        assert!("(8,32)".parse::<Layout>().is_err()); // missing stride
    }

    #[test]
    fn identity_map_detection() {
        assert!(Layout::contiguous(4).is_identity_map());
        // Parentheses around a single mode do not change its mapping.
        let l: Layout = "(4):(1)".parse().unwrap();
        assert!(l.is_identity_map());
        // This shape/stride order makes its colexicographic flat coordinates
        // walk through adjacent offsets.
        let rm: Layout = "(32,8):(1,32)".parse().unwrap();
        assert!(rm.is_identity_map());
        // A stride of two skips every other offset.
        let s2: Layout = "4:2".parse().unwrap();
        assert!(!s2.is_identity_map());

        let overflowing_size = Layout::new(t(&[i64::MAX, 2]), t(&[1, 1]));
        assert!(!overflowing_size.is_identity_map());
        let overflowing_address = Layout::new(IntTuple::Leaf(3), IntTuple::Leaf(i64::MAX));
        assert_eq!(overflowing_address.checked_call(&IntTuple::Leaf(2)), None);
    }

    #[test]
    fn nested_shape() {
        // The first top-level mode contains two nested coordinate parts; the
        // second top-level mode is a single part.
        let shape = IntTuple::Tuple(vec![t(&[2, 4]), IntTuple::Leaf(32)]);
        let stride = IntTuple::Tuple(vec![t(&[128, 32]), IntTuple::Leaf(1)]);
        let l = Layout::new(shape, stride);
        assert_eq!(l.size(), 256);
        // Every nested coordinate is multiplied by its matching nested stride.
        let c = IntTuple::Tuple(vec![t(&[1, 2]), IntTuple::Leaf(5)]);
        assert_eq!(l.call(&c), 128 + 64 + 5);
    }
}
