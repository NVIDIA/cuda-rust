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

/// Unpacks a flat 1D coordinate `idx` into a logical coordinate tuple matching
/// `shape` under CuTe's colexicographic (leftmost-mode-fastest) walk
/// (CuTeDSL `cute.idx2crd`).
///
/// This is the inverse of the flat-coordinate decomposition in [`crd2idx`]:
/// for any `0 <= idx < shape.size()`, evaluating `crd2idx(&idx2crd(idx, shape), shape, stride)`
/// produces the exact same offset as `crd2idx(&IntTuple::Leaf(idx), shape, stride)`.
pub fn idx2crd(idx: i64, shape: &IntTuple) -> IntTuple {
    match shape {
        IntTuple::Leaf(_s) => IntTuple::Leaf(idx),
        IntTuple::Tuple(ss) => {
            let mut rem = idx;
            let mut coords = Vec::with_capacity(ss.len());
            for (i, s) in ss.iter().enumerate() {
                let sz = s.size();
                let c_i = if i + 1 == ss.len() { rem } else { rem % sz };
                coords.push(idx2crd(c_i, s));
                rem /= sz;
            }
            IntTuple::Tuple(coords)
        }
    }
}

/// Allocation-free 2D column-major `idx2crd` (`(rows, cols) : (1, rows)`).
///
/// Unpacks a flat coordinate `idx` into `(row, col) = (idx % rows, idx / rows)`
/// with the row mode (`0..rows`) changing fastest (CuTeDSL `cute.idx2crd`).
#[must_use]
#[inline(always)]
pub const fn idx2crd_col_major_2d(idx: u32, rows: u32) -> (u32, u32) {
    (idx % rows, idx / rows)
}

/// Allocation-free 2D row-major `idx2crd` (`(rows, cols) : (cols, 1)`).
///
/// Unpacks a flat coordinate `idx` into `(row, col) = (idx / cols, idx % cols)`
/// with the column mode (`0..cols`) changing fastest (CuTeDSL `cute.idx2crd`).
#[must_use]
#[inline(always)]
pub const fn idx2crd_row_major_2d(idx: u32, cols: u32) -> (u32, u32) {
    (idx / cols, idx % cols)
}

/// Allocation-free grouped-M 2D tile coordinate mapping (`logical_divide` on M by `GROUP_M`
/// followed by `idx2crd`).
///
/// Divides an `(m_tiles, n_tiles)` tile grid into L2-cache-friendly groups of up to
/// `GROUP_M` M-tiles (`((actual_m, n_tiles), m_groups)`), then unpacks the flat
/// persistent tile index `tile` into `(m_tile, n_tile)` via [`idx2crd_col_major_2d`].
#[must_use]
#[inline(always)]
pub const fn idx2crd_grouped_2d<const GROUP_M: u32>(
    tile: u32,
    m_tiles: u32,
    n_tiles: u32,
) -> (u32, u32) {
    const { assert!(GROUP_M > 0) };
    let group_m = if m_tiles >= GROUP_M { GROUP_M } else { m_tiles };
    let group_tiles = group_m * n_tiles;
    let (in_group, group_id) = idx2crd_col_major_2d(tile, group_tiles);
    let first_m = group_id * group_m;
    let rem_m = m_tiles - first_m;
    let actual_m = if rem_m >= group_m { group_m } else { rem_m };
    let (m_in_group, n_tile) = idx2crd_col_major_2d(in_group, actual_m);
    (first_m + m_in_group, n_tile)
}

/// Allocation-free 2D column-major `crd2idx` (`(rows, cols) : (1, rows)`).
///
/// Maps `(row, col)` to its linear offset `row + col * rows` (CuTeDSL `cute.crd2idx`).
#[must_use]
#[inline(always)]
pub const fn crd2idx_col_major_2d(row: u32, col: u32, rows: u32) -> u32 {
    row + col * rows
}

/// Allocation-free 3D column-major `crd2idx` (`(d0, d1, d2) : (1, d0, d0 * d1)`).
///
/// Maps `(i0, i1, i2)` to its linear offset `i0 + (i1 + i2 * d1) * d0` (CuTeDSL `cute.crd2idx`).
#[must_use]
#[inline(always)]
pub const fn crd2idx_col_major_3d(i0: u32, i1: u32, i2: u32, d0: u32, d1: u32) -> u32 {
    i0 + (i1 + i2 * d1) * d0
}

/// Allocation-free 2D row-major `crd2idx` (`(rows, cols) : (cols, 1)`).
///
/// Maps `(row, col)` to its linear offset `row * cols + col` (CuTeDSL `cute.crd2idx`).
#[must_use]
#[inline(always)]
pub const fn crd2idx_row_major_2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

/// Allocation-free 3D row-major `crd2idx` (`(d0, d1, d2) : (d1 * d2, d2, 1)`).
///
/// Maps `(i0, i1, i2)` to its linear offset `(i0 * d1 + i1) * d2 + i2` (CuTeDSL `cute.crd2idx`).
#[must_use]
#[inline(always)]
pub const fn crd2idx_row_major_3d(i0: usize, i1: usize, i2: usize, d1: usize, d2: usize) -> usize {
    (i0 * d1 + i1) * d2 + i2
}

/// Allocation-free `crd2idx` helper for a canonical SM100 tiled FP16 shared-memory layout
/// given explicit `(rows, cols, mn_major)` extents.
#[must_use]
#[inline(always)]
pub const fn sm100_smem_crd2idx_with_extents(
    row: u32,
    col: u32,
    rows: u32,
    cols: u32,
    mn_major: bool,
) -> usize {
    let contiguous = if mn_major { rows } else { cols };
    let outer_extent = if mn_major { cols } else { rows };
    assert!(contiguous >= 32 && contiguous.is_multiple_of(32));
    assert!(outer_extent > 0 && outer_extent.is_multiple_of(8));
    assert!(contiguous == 32 || contiguous.is_multiple_of(64));
    let span = if contiguous == 32 { 32 } else { 64 };
    let swizzle_bits = if span == 32 { 2 } else { 3 };
    let (inner, outer) = if mn_major { (row, col) } else { (col, row) };
    let (inner_in_span, span_idx) = idx2crd_col_major_2d(inner, span);
    let plain = (span_idx * (span * outer_extent) + outer * span + inner_in_span) as i64;
    crate::Swizzle::new(swizzle_bits, 3, 3).apply(plain) as usize
}

/// Allocation-free `crd2idx` for a canonical SM100 tiled FP16 shared-memory layout
/// (`tile_to_shape` of the `8 × SPAN` swizzle atom composed with `Swizzle<B, 3, 3>`).
///
/// Evaluates the outer tiled layout `((8, ROWS/8), (SPAN, COLS/SPAN))` (for K-major) or
/// `((SPAN, ROWS/SPAN), (8, COLS/8))` (for MN-major) and applies [`crate::Swizzle::apply`].
#[must_use]
#[inline(always)]
pub const fn sm100_smem_crd2idx<const ROWS: u32, const COLS: u32, const MN_MAJOR: bool>(
    row: u32,
    col: u32,
) -> usize {
    sm100_smem_crd2idx_with_extents(row, col, ROWS, COLS, MN_MAJOR)
}

/// Allocation-free `u32` word offset within one 128-row × 4-group Blackwell scale atom
/// (`((32, 4)) : ((4, 1))`, i.e. `idx2crd_col_major_2d(row % 128, 32)` composed with `RowMajor<32, 4>`).
#[must_use]
#[inline(always)]
pub const fn sm1xx_scale_atom_word_offset(row: usize) -> usize {
    let (r_in_32, quarter) = idx2crd_col_major_2d((row % 128) as u32, 32);
    crd2idx_row_major_2d(r_in_32 as usize, quarter as usize, 4)
}

/// Allocation-free byte offset for Blackwell's canonical `(32, 4, 4)` block-scale layout
/// (`Sm1xxBlockScaleKMajor` `crd2idx`).
///
/// Decomposes `row` into `(row_in_atom, m_block)` via [`idx2crd_col_major_2d`] and `group` into
/// `(group_in_atom, k_block)` via [`idx2crd_col_major_2d`], then combines the 512-byte atom index
/// (`crd2idx_row_major_3d(batch, m_block, k_block, rest_m, rest_k)`) with the in-atom byte offset.
#[must_use]
#[inline(always)]
pub const fn sm1xx_block_scale_crd2idx(
    batch: usize,
    row: usize,
    group: usize,
    rest_m: usize,
    rest_k: usize,
) -> usize {
    let (row_in_atom, m_block) = idx2crd_col_major_2d(row as u32, 128);
    let (group_in_atom, k_block) = idx2crd_col_major_2d(group as u32, 4);
    let atom_idx = crd2idx_row_major_3d(batch, m_block as usize, k_block as usize, rest_m, rest_k);
    let word_in_atom = sm1xx_scale_atom_word_offset(row_in_atom as usize);
    let byte_in_atom = crd2idx_row_major_2d(word_in_atom, group_in_atom as usize, 4);
    crd2idx_row_major_2d(atom_idx, byte_in_atom, 512)
}

/// Allocation-free `u64` offset for a `tcgen05` `SWIZZLE_NONE` (`LBO = rows * 16, SBO = 128`)
/// `K = 512` (`32` scale groups of `16` FP4 elements) burst tile with `ROWS` (`128` or `256`) rows:
///
/// ```text
/// Hierarchical layout over 8-byte (16-FP4) `u64` elements per burst:
///   Shape  = (ROWS, (2_u64s_per_16B, 2_halves_per_K64, 8_K64_steps_per_burst))
///   Stride = (   2, (             1,         ROWS * 2,              ROWS * 4))
/// ```
///
/// Decomposes `scale_group` into `(grp_in_burst 0..32, burst)` and `(u64_in_16b, half_k64, step_k64)`
/// using [`idx2crd_col_major_2d`] and evaluates the hierarchical stride via [`crd2idx_row_major_2d`].
#[must_use]
#[inline(always)]
pub const fn sm100_noswizzle_burst_u64_crd2idx<const ROWS: usize>(
    group_batch: usize,
    row_in_tile: usize,
    scale_group: usize,
    k_bursts: usize,
) -> usize {
    let (grp_in_burst, burst) = idx2crd_col_major_2d(scale_group as u32, 32);
    let (u64_in_16b, half_and_step) = idx2crd_col_major_2d(grp_in_burst, 2);
    let (half_k64, step_k64) = idx2crd_col_major_2d(half_and_step, 2);
    let burst_idx = crd2idx_row_major_2d(group_batch, burst as usize, k_bursts);
    let sub_u64 = (step_k64 as usize) * (ROWS * 4)
        + (half_k64 as usize) * (ROWS * 2)
        + (row_in_tile * 2)
        + (u64_in_16b as usize);
    crd2idx_row_major_2d(burst_idx, sub_u64, ROWS * 32)
}

/// Sealed supertrait restricting [`SmemLayout`] to canonical `cute-layout` definitions.
mod sealed_smem {
    /// Marks only shared layouts recognized by the compiler.
    pub trait Layout {}
}

/// Static contract for a canonical SM100 tiled FP16 shared-memory layout.
///
/// Implemented by [`SharedLayout`]; provides compile-time dimensions, swizzle
/// parameters, byte footprint, and element-level `crd2idx` evaluation.
pub trait SmemLayout: sealed_smem::Layout {
    /// Evaluating this constant rejects an unsupported physical layout at compile time.
    const VALID: ();
    /// Logical MN extent in FP16 elements.
    const ROWS: u32;
    /// Logical K extent in FP16 elements.
    const COLS: u32;
    /// Whether the MN mode (`true`) or the K mode (`false`) is contiguous.
    const MN_MAJOR: bool;
    /// Contiguous FP16 values in one swizzle atom (`32` for B64, `64` for B128).
    const SPAN: u32;
    /// Number of XOR bits in the shared-memory swizzle (`2` for B64, `3` for B128).
    const SWIZZLE_BITS: u32;
    /// Byte footprint of one complete `ROWS × COLS` FP16 shared tile.
    const BYTES: usize;
    /// Map a logical `(row, col)` coordinate into shared storage in FP16 elements
    /// (CuTeDSL `cute.crd2idx`).
    fn crd2idx(row: u32, col: u32) -> usize;
}

/// Canonical SM100 `(MN, K)` tiled FP16 shared-memory layout marker.
///
/// `MN_MAJOR` selects the contiguous logical mode (`false` = K-major, `true` = MN-major).
/// Contiguous dimensions wider than 64 FP16 values are tiled in 128-byte chunks via
/// [`crate::algebra::tile_to_shape`] and swizzled with `Swizzle<B, 3, 3>`.
pub struct SharedLayout<const ROWS: u32, const COLS: u32, const MN_MAJOR: bool = false>;

impl<const R: u32, const C: u32, const MN: bool> SharedLayout<R, C, MN> {
    /// Reject unsupported contiguous swizzle spans and outer tile extents at compile time.
    pub const VALID: () = {
        let contiguous = if MN { R } else { C };
        let outer = if MN { C } else { R };
        assert!(contiguous >= 32 && contiguous.is_multiple_of(32));
        assert!(outer > 0 && outer.is_multiple_of(8));
        assert!(contiguous == 32 || contiguous.is_multiple_of(64));
    };

    /// Byte footprint of one complete `R × C` FP16 shared tile (`R * C * 2` bytes).
    pub const BYTES: usize = {
        let () = Self::VALID;
        (R as usize) * (C as usize) * 2
    };

    /// Offset in FP16 elements from the start of this shared tile (CuTeDSL `cute.crd2idx`).
    #[must_use]
    #[inline(always)]
    pub const fn crd2idx(row: u32, col: u32) -> usize {
        let () = Self::VALID;
        sm100_smem_crd2idx::<R, C, MN>(row, col)
    }
}

impl<const R: u32, const C: u32, const MN: bool> sealed_smem::Layout for SharedLayout<R, C, MN> {}

impl<const R: u32, const C: u32, const MN: bool> SmemLayout for SharedLayout<R, C, MN> {
    const VALID: () = SharedLayout::<R, C, MN>::VALID;
    const ROWS: u32 = R;
    const COLS: u32 = C;
    const MN_MAJOR: bool = MN;
    const SPAN: u32 = if (if MN { R } else { C }) == 32 {
        32
    } else {
        64
    };
    const SWIZZLE_BITS: u32 = if Self::SPAN == 32 { 2 } else { 3 };
    const BYTES: usize = SharedLayout::<R, C, MN>::BYTES;

    #[inline(always)]
    fn crd2idx(row: u32, col: u32) -> usize {
        Self::crd2idx(row, col)
    }
}

impl<const B: u32, const R: i64, const C: i64>
    crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::RowMajor<R, C>,
    >
{
    /// Offset in FP16 elements from the start of this K-major SM100 shared tile
    /// (CuTeDSL `cute.crd2idx`).
    #[must_use]
    #[inline(always)]
    pub const fn crd2idx(row: u32, col: u32) -> usize {
        let () = <Self as SmemLayout>::VALID;
        sm100_smem_crd2idx_with_extents(row, col, R as u32, C as u32, false)
    }
}

impl<const B: u32, const R: i64, const C: i64> sealed_smem::Layout
    for crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::RowMajor<R, C>,
    >
{
}

impl<const B: u32, const R: i64, const C: i64> SmemLayout
    for crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::RowMajor<R, C>,
    >
{
    const VALID: () = {
        assert!(R > 0 && R <= u32::MAX as i64 && C > 0 && C <= u32::MAX as i64);
        let r = R as u32;
        let c = C as u32;
        assert!(c >= 32 && c.is_multiple_of(32));
        assert!(r.is_multiple_of(8));
        assert!(c == 32 || c.is_multiple_of(64));
        let expected_bits = if c == 32 { 2 } else { 3 };
        assert!(B == expected_bits);
    };
    const ROWS: u32 = {
        let () = Self::VALID;
        R as u32
    };
    const COLS: u32 = {
        let () = Self::VALID;
        C as u32
    };
    const MN_MAJOR: bool = false;
    const SPAN: u32 = if Self::COLS == 32 { 32 } else { 64 };
    const SWIZZLE_BITS: u32 = B;
    const BYTES: usize = {
        let () = Self::VALID;
        (R as usize) * (C as usize) * 2
    };

    #[inline(always)]
    fn crd2idx(row: u32, col: u32) -> usize {
        Self::crd2idx(row, col)
    }
}

impl<const B: u32, const R: i64, const C: i64>
    crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::ColMajor<R, C>,
    >
{
    /// Offset in FP16 elements from the start of this MN-major SM100 shared tile
    /// (CuTeDSL `cute.crd2idx`).
    #[must_use]
    #[inline(always)]
    pub const fn crd2idx(row: u32, col: u32) -> usize {
        let () = <Self as SmemLayout>::VALID;
        sm100_smem_crd2idx_with_extents(row, col, R as u32, C as u32, true)
    }
}

impl<const B: u32, const R: i64, const C: i64> sealed_smem::Layout
    for crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::ColMajor<R, C>,
    >
{
}

impl<const B: u32, const R: i64, const C: i64> SmemLayout
    for crate::markers::ComposedLayout<
        crate::markers::Swizzle<B, 3, 3>,
        0,
        crate::markers::ColMajor<R, C>,
    >
{
    const VALID: () = {
        assert!(R > 0 && R <= u32::MAX as i64 && C > 0 && C <= u32::MAX as i64);
        let r = R as u32;
        let c = C as u32;
        assert!(r >= 32 && r.is_multiple_of(32));
        assert!(c.is_multiple_of(8));
        assert!(r == 32 || r.is_multiple_of(64));
        let expected_bits = if r == 32 { 2 } else { 3 };
        assert!(B == expected_bits);
    };
    const ROWS: u32 = {
        let () = Self::VALID;
        R as u32
    };
    const COLS: u32 = {
        let () = Self::VALID;
        C as u32
    };
    const MN_MAJOR: bool = true;
    const SPAN: u32 = if Self::ROWS == 32 { 32 } else { 64 };
    const SWIZZLE_BITS: u32 = B;
    const BYTES: usize = {
        let () = Self::VALID;
        (R as usize) * (C as usize) * 2
    };

    #[inline(always)]
    fn crd2idx(row: u32, col: u32) -> usize {
        Self::crd2idx(row, col)
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

    /// Builds a compact layout with the smallest stride on the rightmost
    /// coordinate number (row-major).
    ///
    /// For a `(row, column)` shape with `init = 1`, shape `(2,3)` becomes
    /// `(2,3):(3,1)`.
    pub fn row_major_with(shape: IntTuple, init: i64) -> Layout {
        let leaves = shape.flatten();
        let mut strides = vec![0; leaves.len()];
        let mut running = init;
        for (s, dst) in leaves.iter().rev().zip(strides.iter_mut().rev()) {
            *dst = running;
            running *= s;
        }
        let stride = shape.restructure(&mut strides.iter());
        Layout { shape, stride }
    }

    /// Builds a compact row-major layout with unit spacing on the rightmost mode.
    ///
    /// This is [`Layout::row_major_with`] with `init = 1`. For example, shape
    /// `(2,3)` becomes `(2,3):(3,1)`.
    pub fn row_major(shape: IntTuple) -> Layout {
        Layout::row_major_with(shape, 1)
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
