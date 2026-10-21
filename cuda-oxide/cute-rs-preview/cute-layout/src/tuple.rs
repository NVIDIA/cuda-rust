/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Nested integer tuples shared by shapes, strides, and coordinates.

use alloc::vec::Vec;

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
    Leaf(#[doc = "One shape, stride or coordinate integer."] i64),
    /// An ordered group of integers or further groups.
    Tuple(#[doc = "Ordered child modes whose nested grouping is preserved."] Vec<IntTuple>),
}

impl IntTuple {
    /// Multiplies all contained integers.
    ///
    /// When this tuple is a shape, that product is its logical cell count:
    /// `(2,(3,4))` contains `2 * 3 * 4 = 24` cells. This low-level method does
    /// not reject zero or negative values and does not protect against
    /// overflow; [`Layout::checked_size`](crate::Layout::checked_size) does.
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
    /// A [`Layout`](crate::Layout) requires congruent shape and stride tuples so every shape
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

    /// Append leaf integers left to right without retaining tuple parentheses.
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
    pub(crate) fn restructure(&self, leaves: &mut alloc::slice::Iter<'_, i64>) -> IntTuple {
        match self {
            IntTuple::Leaf(_) => IntTuple::Leaf(*leaves.next().expect("leaf count mismatch")),
            IntTuple::Tuple(ts) => {
                IntTuple::Tuple(ts.iter().map(|t| t.restructure(leaves)).collect())
            }
        }
    }
}
