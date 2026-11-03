/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread/value ownership maps derived from thread and per-thread value layouts.

use alloc::vec;
use alloc::vec::Vec;

use crate::{IntTuple, Layout};

use super::Result;
use super::mapping::{right_inverse, with_shape};
use super::tiling::raked_product;

/// Builds a direct map from `(thread, value_number)` to a cell in one tile.
///
/// `thr` says how thread numbers are arranged; `val` says how one thread's
/// value numbers are arranged. For this example their input mappings are:
///
/// ```text
/// thr = (2,3):(3,1)       val = (2,2):(2,1)
///
/// thread grid             one thread's value pattern
///   0  1  2                 0  1
///   3  4  5                 2  3
/// ```
///
/// Combining the two patterns makes a `4 x 6` tile. `TnVm` means value `m`
/// owned by thread `n`:
///
/// ```text
///                         tile column
///                0      1      2      3      4      5
///            +------+------+------+------+------+------+
/// tile row 0 | T0V0 | T0V1 | T1V0 | T1V1 | T2V0 | T2V1 |
///            +------+------+------+------+------+------+
///          1 | T0V2 | T0V3 | T1V2 | T1V3 | T2V2 | T2V3 |
///            +------+------+------+------+------+------+
///          2 | T3V0 | T3V1 | T4V0 | T4V1 | T5V0 | T5V1 |
///            +------+------+------+------+------+------+
///          3 | T3V2 | T3V3 | T4V2 | T4V3 | T5V2 | T5V3 |
///            +------+------+------+------+------+------+
/// ```
///
/// The result is `(tiler, tv)`:
///
/// ```text
/// tiler = [4, 6]                              tile shape
/// tv = ((3,2),(2,2)):((8,2),(4,1))           direct mapping
///
/// input (thread 4, value 3) -> tv offset 15 -> tile cell (row 3, column 3)
/// ```
///
/// Here the tile is flattened with four rows changing fastest, so cell
/// `(3,3)` has flat offset `3 + 3 * 4 = 15`. The construction first makes a
/// "tile cell -> (thread, value)" map with [`raked_product`], then reverses it
/// with [`right_inverse`] and [`with_shape`].
///
/// A [copy atom](crate#atoms) is not part of this function. Later validation
/// groups the returned values into copy transactions and checks their memory
/// addresses.
///
/// This algebra assumes the thread and value patterns combine into one
/// complete, one-owner-per-cell tile; it does not prove that precondition.
/// Pass the result to
/// [validate_tv_layout](crate::validation::validate_tv_layout) before using a
/// layout decoded from external compiler input.
pub fn make_layout_tv(thr: &Layout, val: &Layout) -> Result<(Vec<i64>, Layout)> {
    let layout_mn = raked_product(thr, val)?;
    let tiler_mn: Vec<i64> = layout_mn.modes().iter().map(|m| m.size()).collect();
    let inv = right_inverse(&layout_mn)?;
    let tv = with_shape(
        &inv,
        IntTuple::Tuple(vec![IntTuple::Leaf(thr.size()), IntTuple::Leaf(val.size())]),
    )?;
    Ok((tiler_mn, tv))
}
