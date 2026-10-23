/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Layout construction and checked or unchecked coordinate evaluation.

use alloc::vec;
use alloc::vec::Vec;

use crate::IntTuple;

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

/// Multiply positive shape leaves, rejecting empty, nonpositive or overflowing shapes.
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
            shape.is_congruent(&stride),
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
        let leaves = shape.flatten();
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
    /// vector containing itself. [`Layout::get`] returns one mode on its own.
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

    /// Returns top-level [mode](crate#modes) `mode` as its own layout, as
    /// CuTe's `get<I>(layout)` and CuTeDSL's `cute.get(layout, mode=[i])`.
    ///
    /// ```text
    /// layout     ((2,4),3):((1,2),8)
    /// get(0)     (2,4):(1,2)
    /// get(1)     3:8
    /// ```
    ///
    /// The result is `modes()[mode]` without building the other modes. A
    /// rank-1 layout returns itself for `mode == 0`.
    ///
    /// # Panics
    ///
    /// Panics when `mode` is not below [`Layout::rank`].
    pub fn get(&self, mode: usize) -> Layout {
        match (&self.shape, &self.stride) {
            (IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
                assert!(
                    mode < ss.len(),
                    "layout mode {mode} is out of range for rank {}",
                    ss.len()
                );
                Layout {
                    shape: ss[mode].clone(),
                    stride: ds[mode].clone(),
                }
            }
            _ => {
                assert!(mode == 0, "layout mode {mode} is out of range for rank 1");
                self.clone()
            }
        }
    }

    /// Counts the top-level [modes](crate#modes), as CuTe's `rank(layout)`.
    ///
    /// ```text
    /// 4:1                    rank 1
    /// (2,3):(3,1)            rank 2
    /// ((2,4),3):((1,2),8)    rank 2    nesting inside a mode does not count
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.rank()
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
