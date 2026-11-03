/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Logical division and products that partition or repeat layouts.

use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::{IntTuple, Layout};

use super::mapping::{coalesce, complement, composition};
use super::{AlgebraError, Result};

/// Groups a coordinate space into tiles without moving any data.
///
/// Think of eight numbered items packed into boxes that hold four items each.
/// One number from `0` through `7` can then be replaced by two smaller numbers:
/// the item's slot inside its box and the box number.
///
/// ```text
/// original coordinate i     0  1  2  3 | 4  5  6  7
/// slot inside box           0  1  2  3 | 0  1  2  3
/// box number                0  0  0  0 | 1  1  1  1
/// new coordinate          (0,0) ... (3,0) | (0,1) ... (3,1)
/// result offset             0  1  2  3 | 4  5  6  7
/// ```
///
/// In layout notation, the whole strip is `a = 8:1`, one box is `b = 4:1`,
/// and `logical_divide(a, b)` returns `(4,2):(1,4)`. For example, original
/// coordinate `6` becomes `(slot 2, box 1)`, and the result still reaches
/// offset `2 + 4 * 1 = 6`.
///
/// The first top-level [mode](crate#modes) describes a position inside the
/// `b` pattern. The second chooses a translated copy of that pattern. `b` is a
/// layout, not just a tile size, so its strides also control the walk within a
/// tile. The final offsets still come from `a`.
///
/// If the items do not fill the last box, the result still describes a whole
/// box. For example, `logical_divide(6:1, 4:1)` also returns
/// `(4,2):(1,4)`. Its last box includes offsets `6` and `7`, so later
/// predication must disable those two extra positions.
pub fn logical_divide(a: &Layout, b: &Layout) -> Result<Layout> {
    let rest = complement(b, a.size())?;
    composition(a, &Layout::from_modes(vec![b.clone(), rest]))
}

/// Divides every top-level coordinate into tiles, then groups the answers into
/// "position inside the tile" and "which tile".
///
/// For example, divide a four-row, six-column grid into `2 x 3` tiles:
///
/// ```text
/// a = (4,6):(6,1)          tile_shape = [2, 3]
///
///                              column
///                         0  1  2 | 3  4  5
///                       +---------+---------+
///                 row 0 | A  A  A | B  B  B |
///                     1 | A  A  A | B  B  B |
///                       +---------+---------+
///                     2 | C  C  C | D  D  D |
///                     3 | C  C  C | D  D  D |
///                       +---------+---------+
///
/// A = tile (0,0)   B = tile (0,1)   C = tile (1,0)   D = tile (1,1)
/// ```
///
/// Row `3` is row `1` inside tile row `1`. Column `5` is column `2` inside
/// tile column `1`. The original coordinate `(3,5)` is therefore renamed as
/// `((1,2), (1,1))`:
///
/// ```text
/// ((inside_row, inside_column), (tile_row, tile_column))
///                    ((1, 2),             (1, 1))
///
/// original offset = 3 * 6 + 5 = 23
/// result offset                         = 23
/// result = ((2,3),(2,2)):((6,1),(12,3))
/// ```
///
/// The name **zipped** describes how the coordinate pieces are regrouped, like
/// matching the two sides of a zipper:
///
/// ```text
/// split each axis: ((inside_row, tile_row), (inside_column, tile_column))
/// zip the answers: ((inside_row, inside_column), (tile_row, tile_column))
/// ```
///
/// This changes how a coordinate is named, not the offset it reaches. Each
/// entry in `tile_shape` matches one top-level [mode](crate#modes) of `a`, so
/// the two ranks must be equal. If a tile extent does not divide its matching
/// input extent, the final tile is rounded up and later predication must
/// disable its extra edge positions.
///
/// This is CuTe's `zipped_divide` with the tiler given as a shape, so every
/// tile mode is contiguous (stride 1). A tiler layout with its own strides
/// is not accepted.
pub fn zipped_divide(a: &Layout, tile_shape: &[i64]) -> Result<Layout> {
    let modes = a.modes();
    if modes.len() != tile_shape.len() {
        return Err(AlgebraError(format!(
            "zipped_divide: tiler rank {} does not match layout rank {}",
            tile_shape.len(),
            modes.len()
        )));
    }
    let mut tiles = Vec::with_capacity(modes.len());
    let mut rests = Vec::with_capacity(modes.len());
    for (mode, &t) in modes.iter().zip(tile_shape) {
        let divided = logical_divide(mode, &Layout::contiguous(t))?;
        let mut parts = divided.modes();
        if parts.len() != 2 {
            return Err(AlgebraError(
                "zipped_divide: mode divide was not rank-2".into(),
            ));
        }
        rests.push(parts.pop().unwrap());
        tiles.push(parts.pop().unwrap());
    }
    Ok(Layout::from_modes(vec![
        Layout::from_modes(tiles),
        Layout::from_modes(rests),
    ]))
}

/// Builds a larger layout by placing translated copies of `a` according to
/// `b`. It describes the copies; it does not copy any data.
///
/// Think of `a = 2:1` as a tray with two positions. With `b = 3:1`, the product
/// has three trays:
///
/// ```text
/// copy                       A       B       C
/// copy number                0       1       2
/// positions in each copy   [0 1]   [0 1]   [0 1]
/// result offsets           [0 1]   [2 3]   [4 5]
///
/// product coordinate      (0,0) (1,0) | (0,1) (1,1) | (0,2) (1,2)
/// result offset              0     1   |   2     3   |   4     5
/// result = (2,3):(1,2)
/// ```
///
/// The result keeps two logical groups: `(position in a, copy selected by b)`.
/// If both inputs describe rows and columns, that grouping looks like:
///
/// ```text
/// ((position_row, position_column), (copy_row, copy_column))
/// ```
///
/// This is the grouping that [`blocked_product`] and [`raked_product`] later
/// pair up axis by axis. `b` is a layout, not merely a number of copies, so it
/// can also add spacing or change their order. For example,
/// `logical_product(2:1, 3:2)` is `(2,3):(1,4)` and places the copies at
/// offsets `[0,1]`, `[4,5]`, and `[8,9]`.
pub fn logical_product(a: &Layout, b: &Layout) -> Result<Layout> {
    let comp = complement(a, a.size() * b.cosize())?;
    let repeated = composition(&comp, b)?;
    Ok(Layout::from_modes(vec![a.clone(), repeated]))
}

/// Adds non-varying one-cell modes until two product inputs have equal rank.
///
/// A `1:0` mode contributes only offset zero, so padding changes structure but
/// not the original mapping.
fn pad_to_rank(l: &Layout, n: usize) -> Layout {
    let mut modes = l.modes();
    while modes.len() < n {
        modes.push(Layout {
            shape: IntTuple::Leaf(1),
            stride: IntTuple::Leaf(0),
        });
    }
    Layout::from_modes(modes)
}

/// Pairs matching axes of a logical product, like closing a zipper.
///
/// [`logical_product`] first keeps all `a` coordinates separate from all copy
/// coordinates. This helper pairs the matching coordinates instead:
///
/// ```text
/// logical product: ((a_row, a_column), (copy_row, copy_column))
///
/// block first:     ((a_row, copy_row), (a_column, copy_column))
/// copy first:      ((copy_row, a_row), (copy_column, a_column))
/// ```
///
/// `block_first` selects the second or third line. Each pair is simplified
/// afterward, so a pair may print as one mode when it is one constant-stride
/// walk. If the inputs have different ranks, missing right-hand axes act like
/// one-choice `1:0` modes.
fn zipped_product(a: &Layout, b: &Layout, block_first: bool) -> Result<Layout> {
    let n = a.modes().len().max(b.modes().len());
    let ap = pad_to_rank(a, n);
    let bp = pad_to_rank(b, n);
    let lp = logical_product(&ap, &bp)?;
    let parts = lp.modes();
    let (block, reps) = (&parts[0], &parts[1]);
    let block_modes = block.modes();
    let rep_modes = reps.modes();
    if block_modes.len() != n || rep_modes.len() != n {
        return Err(AlgebraError("product: unexpected mode structure".into()));
    }
    let zipped = block_modes
        .into_iter()
        .zip(rep_modes)
        .map(|(bm, rm)| {
            let pair = if block_first {
                Layout::from_modes(vec![bm, rm])
            } else {
                Layout::from_modes(vec![rm, bm])
            };
            coalesce(&pair)
        })
        .collect();
    Ok(Layout::from_modes(zipped))
}

/// Repeats `a` while keeping each copy together as a block.
///
/// For one-axis inputs, `a = 2:1` gives two positions per copy and `b = 3:1`
/// makes three copies. Letters name the copies; digits name positions inside a
/// copy:
///
/// ```text
/// copies                  A = [0 1]   B = [2 3]   C = [4 5]
/// blocked walk            A0 A1     | B0 B1     | C0 C1
/// result offset            0  1     |  2  3     |  4  5
/// result = 6:1
/// ```
///
/// In plain English: finish copy A before starting B, then finish B before
/// starting C. [`raked_product`] visits the same six positions in a different
/// order.
///
/// More precisely, this pairs each [mode](crate#modes) of `a` with the matching
/// repetition mode derived from `b`, putting the `a` coordinate first. For
/// layouts with several modes, "finish one whole copy" need not describe the
/// global flat walk; the blocked order applies inside each matching mode pair.
/// Strides still decide whether successive offsets are adjacent in memory.
pub fn blocked_product(a: &Layout, b: &Layout) -> Result<Layout> {
    zipped_product(a, b, true)
}

/// Repeats `a` while visiting the same position across all copies first.
///
/// Using the same one-axis example as [`blocked_product`], imagine taking the
/// first item from copies A, B, and C before coming back for their second item:
///
/// ```text
/// copies                  A = [0 1]   B = [2 3]   C = [4 5]
/// raked walk              A0 B0 C0  | A1 B1 C1
/// result offset            0  2  4  |  1  3  5
/// result = (3,2):(2,1)
/// ```
///
/// Both products cover the same six positions. Only their coordinate grouping
/// and flat-coordinate walk differ. More precisely, this pairs each
/// [mode](crate#modes) of `a` with the matching repetition mode derived from
/// `b`, putting the repetition coordinate first. With several modes, the raked
/// order applies inside each matching pair rather than across one global list
/// of copies.
///
/// [`make_layout_tv`](super::thread_value::make_layout_tv) uses this interleaving to combine thread and per-thread
/// value patterns without assigning the same tile cell twice.
pub fn raked_product(a: &Layout, b: &Layout) -> Result<Layout> {
    zipped_product(a, b, false)
}

/// Tiles an `atom` layout to cover `target_shape`, stepping through the repetition
/// grid in `order` (CuTeDSL `cute.tile_to_shape`).
///
/// Unlike [`blocked_product`], which coalesces each `(atom_mode, rep_mode)` pair,
/// `tile_to_shape` preserves the two-level `(atom_mode, rep_mode)` hierarchy in each
/// axis so hardware descriptor generators and TMA verifiers can inspect both the
/// inner swizzle atom and the outer tile grid.
///
/// - `order = &[0, 1]` tiles in column-major order (mode 0 repeats fastest).
/// - `order = &[1, 0]` tiles in row-major order (mode 1 repeats fastest).
pub fn tile_to_shape(atom: &Layout, target_shape: &[i64], order: &[usize]) -> Result<Layout> {
    let atom_modes = atom.modes();
    let n = atom_modes.len();
    if target_shape.len() != n || order.len() != n {
        return Err(AlgebraError(format!(
            "tile_to_shape: atom rank {n}, target_shape len {}, order len {} must match",
            target_shape.len(),
            order.len()
        )));
    }
    let mut reps = Vec::with_capacity(n);
    for (i, mode) in atom_modes.iter().enumerate() {
        let atom_extent = mode
            .checked_size()
            .ok_or_else(|| AlgebraError("tile_to_shape: invalid atom mode size".into()))?;
        let target = target_shape[i];
        if target <= 0 || target % atom_extent != 0 {
            return Err(AlgebraError(format!(
                "tile_to_shape: target extent {target} is not a positive multiple of atom extent {atom_extent}"
            )));
        }
        reps.push(target / atom_extent);
    }
    let mut seen = vec![false; n];
    let mut rep_strides = vec![0i64; n];
    let mut running = 1i64;
    for &axis in order {
        if axis >= n || seen[axis] {
            return Err(AlgebraError(
                "tile_to_shape: order must be a permutation of 0..rank".into(),
            ));
        }
        seen[axis] = true;
        rep_strides[axis] = running;
        running = running
            .checked_mul(reps[axis])
            .ok_or_else(|| AlgebraError("tile_to_shape: repetition product overflow".into()))?;
    }
    let rep_layout = Layout::new(
        IntTuple::Tuple(reps.into_iter().map(IntTuple::Leaf).collect()),
        IntTuple::Tuple(rep_strides.into_iter().map(IntTuple::Leaf).collect()),
    );
    let lp = logical_product(atom, &rep_layout)?;
    let parts = lp.modes();
    let block_modes = parts[0].modes();
    let rep_modes = parts[1].modes();
    let paired = block_modes
        .into_iter()
        .zip(rep_modes)
        .map(|(bm, rm)| Layout::from_modes(vec![bm, rm]))
        .collect();
    Ok(Layout::from_modes(paired))
}

/// Construct the canonical SM100 tiled FP16 shared-memory [` crate::ComposedLayout`]
/// using [`tile_to_shape`] and [`crate::Swizzle`] (CuTeDSL `make_smem_layout_k` /
/// `make_smem_layout_mn` + `cute.tile_to_shape`).
///
/// - `mn = false` (K-major): tiles the `(8, span) : (span, 1)` swizzle atom to
///   `(rows, cols)` with `order = &[0, 1]`.
/// - `mn = true` (MN-major): tiles the `(span, 8) : (1, span)` swizzle atom to
///   `(rows, cols)` with `order = &[1, 0]`.
pub fn sm100_smem_layout(
    rows: u32,
    cols: u32,
    mn: bool,
) -> core::result::Result<crate::ComposedLayout, alloc::string::String> {
    let (contiguous, outer) = if mn { (rows, cols) } else { (cols, rows) };
    if contiguous < 32
        || (contiguous != 32 && !contiguous.is_multiple_of(64))
        || outer == 0
        || !outer.is_multiple_of(8)
    {
        return Err(
            "SM100 shared layout needs contiguous 32 or a multiple of 64 and outer multiple of 8"
                .into(),
        );
    }
    let span = i64::from(contiguous.min(64));
    let (atom, order): (Layout, [usize; 2]) = if mn {
        (
            Layout::col_major(IntTuple::Tuple(vec![
                IntTuple::Leaf(span),
                IntTuple::Leaf(8),
            ])),
            [1, 0],
        )
    } else {
        (
            Layout::row_major(IntTuple::Tuple(vec![
                IntTuple::Leaf(8),
                IntTuple::Leaf(span),
            ])),
            [0, 1],
        )
    };
    let outer_layout = tile_to_shape(&atom, &[i64::from(rows), i64::from(cols)], &order)
        .map_err(|e| e.to_string())?;
    crate::ComposedLayout::new(
        crate::Swizzle::new(if span == 32 { 2 } else { 3 }, 3, 3),
        0,
        outer_layout,
        crate::OffsetUnit::Elements,
    )
    .map_err(|e| e.to_string())
}
