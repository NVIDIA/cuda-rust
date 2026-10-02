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

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::{IntTuple, Layout};

/// Explains why a requested layout transformation is not mathematically valid.
///
/// For example, splitting six cells into tiles of four leaves an incomplete
/// tile. Operations that require exact division report an error instead of
/// silently constructing a wrong address map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraError(pub String);

impl fmt::Display for AlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layout algebra error: {}", self.0)
    }
}

impl core::error::Error for AlgebraError {}

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

/// Rewrites a layout as fewer [modes](crate#modes) while preserving its flat
/// coordinate-to-offset sequence.
///
/// A **one-cell mode** means an `extent:stride` pair whose extent is `1`. Its
/// coordinate can only be zero, so it never changes the offset:
///
/// ```text
/// before = (1,4):(99,1)
///
/// mode pair             1:99                  4:1
/// allowed coordinate    only 0                0, 1, 2, 3
/// offset contribution   0 * 99 = 0            0, 1, 2, 3
///
/// tuple coordinate      (0,0)  (0,1)  (0,2)  (0,3)
/// before offset           0      1      2      3
/// after  = 4:1            0      1      2      3
/// ```
///
/// **Adjacent modes** are consecutive shape/stride pairs inside the same
/// layout, not two separate shapes. In `(2,4):(1,2)`, they are `2:1` and
/// `4:2`:
///
/// ```text
/// tuple coordinate     (0,0) (1,0) | (0,1) (1,1) | (0,2) (1,2) | (0,3) (1,3)
/// before offset           0     1   |   2     3   |   4     5   |   6     7
/// flat coordinate         0     1       2     3       4     5       6     7
/// after = 8:1             0     1       2     3       4     5       6     7
/// ```
///
/// The pairs merge because the second stride is `2 = 2 * 1`: its first step
/// starts immediately after visiting both positions of the first pair. In
/// general, `s0:d0` and `s1:d1` merge when `d1 == s0 * d0`.
///
/// Coalescing means the same flat sequence can be written as one
/// constant-stride walk. It does **not** require adjacent memory:
/// `(2,4):(3,6)` maps to `0,3,6,...,21` and coalesces to `8:3`. Successive
/// offsets are physically adjacent only when the resulting stride has
/// magnitude `1`.
///
/// Tuple coordinate structure may change during this rewrite. For example,
/// `(2,4):(1,2)` accepts `(first, second)`, while `8:1` accepts one flat
/// coordinate; their flat walks are the same.
pub fn coalesce(l: &Layout) -> Layout {
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (s, d) in flat_modes(l) {
        if s == 1 {
            continue;
        }
        if let Some(last) = out.last_mut()
            && d == last.0 * last.1
        {
            last.0 *= s;
            continue;
        }
        out.push((s, d));
    }
    layout_from_flat(out)
}

/// Removes broadcast modes before simplifying a layout.
///
/// A stride-zero mode repeats the same offset for every coordinate:
/// `3:0` maps `0,1,2 -> 0,0,0`. It therefore occupies no additional offsets
/// when [`complement`] asks which locations remain unused.
fn filter(l: &Layout) -> Layout {
    let kept: Vec<(i64, i64)> = flat_modes(l)
        .into_iter()
        .map(|(s, d)| if d == 0 { (1, 0) } else { (s, d) })
        .collect();
    coalesce(&layout_from_flat(kept))
}

/// Replaces two consecutive mappings with one direct mapping.
///
/// `b` first turns the input coordinate into a coordinate for `a`; `a` then
/// turns that coordinate into the final offset. For example:
///
/// ```text
/// a = 8:2      coordinate  0  1  2  3  4  5  6  7
///              offset      0  2  4  6  8 10 12 14
///
/// b = 4:2      input c     0  1  2  3
///              b(c)        0  2  4  6       coordinates selected in a
///              a(b(c))     0  4  8 12       final offsets
///
/// result = 4:4
///              input c     0  1  2  3
///              offset      0  4  8 12
/// ```
///
/// In symbols, `result(c) = a(b(c))`, also written `a ∘ b`. The compiler
/// uses this to replace chains such as "thread and value -> tile cell ->
/// memory offset" with one direct offset calculation.
///
/// If `b` has several [modes](crate#modes), the result keeps their nesting.
/// Some input walks cannot be represented by the supported constant-stride
/// mode structure; those cases return [`AlgebraError`]. An imperfect final
/// tile may be rounded up, so later predication must disable coordinates past
/// the real tensor boundary.
pub fn composition(a: &Layout, b: &Layout) -> Result<Layout> {
    if let (IntTuple::Tuple(bs), IntTuple::Tuple(bd)) = (&b.shape, &b.stride) {
        let modes = bs
            .iter()
            .zip(bd)
            .map(|(s, d)| {
                composition(
                    a,
                    &Layout {
                        shape: s.clone(),
                        stride: d.clone(),
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Each output mode corresponds to the same input mode in `b`; only
        // that mode's mapping has passed through `a`.
        return Ok(Layout {
            shape: IntTuple::Tuple(modes.iter().map(|m| m.shape.clone()).collect()),
            stride: IntTuple::Tuple(modes.into_iter().map(|m| m.stride).collect()),
        });
    }

    let (IntTuple::Leaf(b_shape), IntTuple::Leaf(b_stride)) = (&b.shape, &b.stride) else {
        return Err(AlgebraError(
            "composition: rhs shape/stride not congruent".into(),
        ));
    };
    compose_int(a, *b_shape, *b_stride)
}

/// Composes `a` with one input mode described by `b_shape:b_stride`.
fn compose_int(a: &Layout, b_shape: i64, b_stride: i64) -> Result<Layout> {
    // A stride-zero input always selects coordinate zero in `a`, so the
    // result is the same broadcast regardless of `a`'s strides.
    if b_stride == 0 {
        return Ok(layout_from_flat(vec![(b_shape, 0)]));
    }
    let ac = coalesce(a);
    let modes = flat_modes(&ac);
    // With one mode in `a`, each step of `b` advances `b_stride` coordinates,
    // and each of those advances `a` by its stride. Multiplication combines
    // the two steps.
    if modes.len() == 1 {
        return Ok(layout_from_flat(vec![(b_shape, b_stride * modes[0].1)]));
    }

    let mut result: Vec<(i64, i64)> = Vec::new();
    let mut rest_shape = b_shape;
    let mut rest_stride = b_stride;
    for &(curr_shape, curr_stride) in &modes[..modes.len() - 1] {
        // A step through `b` must align with a digit boundary in `a`, or be
        // small enough to live entirely inside the current mode. Otherwise a
        // simple affine mode cannot represent the composed walk.
        if rest_stride % curr_shape != 0 && rest_stride.abs() >= curr_shape {
            return Err(AlgebraError(format!(
                "composition: stride {rest_stride} does not divide lhs mode of shape {curr_shape} \
                 (stride divisibility condition)"
            )));
        }
        let abs_rest_stride = rest_stride.abs();
        let sign = if rest_stride < 0 { -1 } else { 1 };
        let next_shape = ceil_div(curr_shape, abs_rest_stride);
        let next_stride = ceil_div(abs_rest_stride, curr_shape) * sign;

        if next_shape == 1 || rest_shape == 1 {
            rest_stride = next_stride;
            continue;
        }
        let new_shape = next_shape.min(rest_shape);
        // Every non-final piece must split the remaining coordinates into
        // equal groups. The final piece is allowed to describe a partial
        // overhang, which higher-level predication will mask.
        if rest_shape % new_shape != 0 {
            return Err(AlgebraError(format!(
                "composition: shape {rest_shape} is not divisible by {new_shape} \
                 (shape divisibility condition)"
            )));
        }
        result.push((new_shape, rest_stride * curr_stride));
        rest_shape /= new_shape;
        rest_stride = next_stride;
    }

    let last_stride = modes[modes.len() - 1].1;
    if result.is_empty() {
        return Ok(layout_from_flat(vec![(
            rest_shape,
            rest_stride * last_stride,
        )]));
    }
    if rest_shape != 1 {
        result.push((rest_shape, rest_stride * last_stride));
    }
    Ok(coalesce(&layout_from_flat(result)))
}

fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Builds the extra coordinate needed to fill the offsets that `l` skips.
///
/// Take `l = 4:2`, which reaches only the even offsets below `8`:
///
/// ```text
/// l coordinate          0  1  2  3
/// l offset              0  2  4  6
/// ```
///
/// `complement(l, 8)` returns `2:1`. Its coordinate chooses an extra offset of
/// zero or one. Joining the two layouts means adding their contributions:
///
/// ```text
/// combined coordinate  (0,0) (1,0) (2,0) (3,0) | (0,1) (1,1) (2,1) (3,1)
/// l contribution          0     2     4     6   |   0     2     4     6
/// extra contribution      0     0     0     0   |   1     1     1     1
/// combined offset         0     2     4     6   |   1     3     5     7
/// ```
///
/// Together they reach every offset from `0` through `7` exactly once. The
/// complement is therefore an *extra choice to add*, not a list of all unused
/// offsets by itself.
///
/// `cosize_hi` is one past the last requested offset: a value of `8` requests
/// offsets `0` through `7`. It need not equal [`Layout::cosize`] of `l`.
/// [`logical_divide`] uses the returned [modes](crate#modes) for its “which
/// tile?” coordinate.
///
/// When the requested endpoint does not fit a whole repetition, the result may
/// cover a rounded-up superset. For example, `complement(4:1, 6)` is `2:4`;
/// joining them as `(4,2):(1,4)` reaches offsets `0` through `7`, although only
/// `0` through `5` were requested. Later predication must disable `6` and `7`.
pub fn complement(l: &Layout, cosize_hi: i64) -> Result<Layout> {
    let f = filter(l);
    let modes = flat_modes(&f);
    // A broadcast or one-cell layout contributes no varying offset. The new
    // coordinate must therefore span the entire requested interval.
    if modes.len() == 1 && (modes[0].1 == 0 || modes[0].0 == 1) {
        return Ok(coalesce(&Layout::col_major(IntTuple::Leaf(cosize_hi))));
    }

    let mut sd = modes;
    sd.sort_by_key(|&(_, d)| d);

    let mut out_shapes: Vec<i64> = Vec::new();
    let mut out_strides: Vec<i64> = vec![1];
    for &(curr_shape, curr_stride) in &sd[..sd.len() - 1] {
        let prev_stride = *out_strides.last().unwrap();
        if curr_stride % prev_stride != 0 {
            return Err(AlgebraError(format!(
                "complement: stride {curr_stride} not divisible by {prev_stride}"
            )));
        }
        out_shapes.push(curr_stride / prev_stride);
        out_strides.push(curr_shape * curr_stride);
    }
    let (last_shape, last_stride) = *sd.last().unwrap();
    let prev_stride = *out_strides.last().unwrap();
    if last_stride % prev_stride != 0 {
        return Err(AlgebraError(format!(
            "complement: stride {last_stride} not divisible by {prev_stride}"
        )));
    }
    out_shapes.push(last_stride / prev_stride);
    let new_stride = last_stride * last_shape;
    let rest_shape = ceil_div(cosize_hi, new_stride);
    out_shapes.push(rest_shape);
    // The final coordinate advances beyond everything covered so far. Its
    // stride is paired with the final shape appended just above.
    out_strides.push(new_stride);

    let flat: Vec<(i64, i64)> = out_shapes.into_iter().zip(out_strides).collect();
    Ok(coalesce(&layout_from_flat(flat)))
}

/// Builds the reverse lookup for a layout: given an output value, finds a flat
/// input coordinate that produces it.
///
/// The main use is assigning work inside a GPU tile. While planning a tile, it
/// is natural to describe who owns each cell:
///
/// ```text
/// tile cell -> (thread, value number)
/// ```
///
/// A running GPU thread needs the opposite answer. It knows its thread number
/// and which of its values it is processing, and must find the corresponding
/// tile cell:
///
/// ```text
/// (thread, value number) -> tile cell
/// ```
///
/// Consider two threads, each responsible for three values. `T0V2` means
/// "thread 0, value number 2":
///
/// ```text
/// tile cell c       0     1     2   |   3     4     5
/// owner           T0V0  T0V1  T0V2  | T1V0  T1V1  T1V2
/// ```
///
/// A [`Layout`] returns one integer, so the pair `(thread, value number)` is
/// temporarily encoded as an **owner ID**. Thread number changes fastest:
///
/// ```text
/// owner ID = thread + 2 * value_number
///
/// owner          T0V0  T1V0 | T0V1  T1V1 | T0V2  T1V2
/// owner ID         0     1   |   2     3   |   4     5
/// ```
///
/// In layout notation, `thr = 2:1` describes the two threads and `val = 3:1`
/// describes their three value numbers. [`raked_product`] combines them into
/// the forward ownership layout:
///
/// ```text
/// a = raked_product(thr, val) = (3,2):(2,1)
///
/// tile cell c          0     1     2   |   3     4     5
/// owner              T0V0  T0V1  T0V2  | T1V0  T1V1  T1V2
/// a(c), owner ID       0     2     4   |   1     3     5
/// ```
///
/// Here the integer produced by `a` is an owner ID, not a memory address. The
/// right inverse `r` reverses the lookup:
///
/// ```text
/// r = right_inverse(a) = (2,3):(3,1)
///
/// owner ID x               0  1  2  3  4  5
/// r(x), tile cell          0  3  1  4  2  5
/// a(r(x)), owner ID        0  1  2  3  4  5
/// ```
///
/// For example, thread 1 processing its value number 0 has owner ID
/// `1 + 2 * 0 = 1`, and `r(1) = 3` tells it to use tile cell 3.
/// [`make_layout_tv`] performs this construction in three steps:
///
/// ```text
/// raked_product   tile cell -> owner ID
/// right_inverse  owner ID  -> tile cell
/// with_shape      (thread, value number) -> tile cell
/// ```
///
/// In general, the returned layout guarantees `a(r(x)) = x` for every `x` that
/// `r` covers. It is called a **right** inverse because, in `a(r(x))`, `r` is
/// written to the right of `a` and runs first.
///
/// This is not a universal lookup for every layout. If `a` skips an output,
/// there is no coordinate to return for it. If several coordinates produce the
/// same output, `r` may choose one but cannot recover the others. The compact
/// shape-and-stride format can also represent only certain reverse patterns,
/// so the returned range may be shorter than all reachable outputs. The
/// thread/value use requires every owner ID and tile cell exactly once; callers
/// validate that condition with
/// [validate_tv_layout](crate::validation::validate_tv_layout).
///
/// Compare [`left_inverse`], which instead starts with an input, sends it
/// through `a`, and must recover that same input.
pub fn right_inverse(a: &Layout) -> Result<Layout> {
    let ac = coalesce(a);
    let modes = flat_modes(&ac);
    if modes.len() == 1 && modes[0].0 == 1 {
        return Ok(layout_from_flat(vec![(1, 0)]));
    }
    // Before sorting modes by the offsets they reach, remember how each mode
    // contributes to the original flat coordinate. These prefix products
    // become strides in the inverse map.
    let mut rstrides = Vec::with_capacity(modes.len());
    let mut running = 1i64;
    for &(s, _) in &modes {
        rstrides.push(running);
        running *= s;
    }
    let mut dsa: Vec<(i64, i64, i64)> = modes
        .iter()
        .zip(&rstrides)
        .map(|(&(s, d), &r)| (d, s, r))
        .collect();
    dsa.sort();

    let mut current = 1i64;
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (stride, shape, rstride) in dsa {
        if stride == 0 {
            continue;
        }
        if current != stride {
            break;
        }
        out.push((shape, rstride));
        current = shape * stride;
    }
    if out.is_empty() {
        out.push((1, 0));
    }
    Ok(coalesce(&layout_from_flat(out)))
}

/// Recovers the original input after that input has passed through `a`.
///
/// This round trip starts with an input, unlike [`right_inverse`]:
///
/// ```text
/// left-inverse trip: input --a--> offset --r--> the same input
/// ```
///
/// For example, `a = 3:2` stores three logical positions in every other
/// offset. The gaps do not matter because the trip only visits offsets that
/// `a` actually produces:
///
/// ```text
/// original input i       0  1  2
/// a(i), the offset       0  2  4
/// r(a(i))                0  1  2
///
/// left_inverse(a) = (2,3):(3,1)
/// relevant lookups: r(0) = 0, r(2) = 1, r(4) = 2
/// ```
///
/// The guarantee is `r(a(i)) = i` for every input `i`. It is called a
/// **left** inverse because `r` appears on the left in `r ∘ a` and runs
/// second. Values of `r` at unused offsets such as `1` and `3` have no meaning
/// for this guarantee.
///
/// Every input must have its own offset. If two inputs share one, the offset
/// cannot tell them apart on the way back. In other words, `a` must be
/// **one-to-one** (also called *injective*). The duplicate-offset example from
/// [`right_inverse`], `(2,2):(1,0)`, therefore has no valid left inverse.
/// This function does not exhaustively check that requirement; callers must
/// validate externally supplied layouts before relying on the result.
///
/// The implementation fills `a`'s gaps with [`complement`], then uses
/// [`right_inverse`].
pub fn left_inverse(a: &Layout) -> Result<Layout> {
    let comp = complement(a, a.cosize())?;
    let padded = Layout::from_modes(vec![a.clone(), comp]);
    right_inverse(&padded)
}

/// Replaces a flat input coordinate with a tuple-shaped coordinate, preserving
/// the same flat offset walk.
///
/// When both have six cells, as in `l = 6:1` and `shape = (2,3)`, the left
/// coordinate changes fastest:
///
/// ```text
/// old flat coordinate     0      1      2      3      4      5
/// new tuple coordinate   (0,0)  (1,0)  (0,1)  (1,1)  (0,2)  (1,2)
/// offset                   0      1      2      3      4      5
///
/// result = (2,3):(1,2)
/// ```
///
/// Only the way callers name a coordinate changes; the flattened
/// coordinate-to-offset sequence does not. This is useful when a flat inverse
/// or product must expose named `(thread, value)` [modes](crate#modes).
/// Preserving the complete walk requires `shape.size() == l.size()`. This
/// function does not enforce equality: a smaller shape selects a prefix, while
/// a larger shape may extend beyond `l`'s original coordinate range.
pub fn with_shape(l: &Layout, shape: IntTuple) -> Result<Layout> {
    composition(l, &Layout::col_major(shape))
}

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
pub fn zipped_divide_by_shape(a: &Layout, tile_shape: &[i64]) -> Result<Layout> {
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
/// [`make_layout_tv`] uses this interleaving to combine thread and per-thread
/// value patterns without assigning the same tile cell twice.
pub fn raked_product(a: &Layout, b: &Layout) -> Result<Layout> {
    zipped_product(a, b, false)
}

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

/// Returns the alignment factor guaranteed by the layout's offset pattern.
///
/// This examines only the static strides; it does not inspect a memory
/// pointer. For `(8,16):(1,4)`, the first coordinate gives an eight-offset
/// unit-stride run, while changing the second coordinate shifts that run by
/// four:
///
/// ```text
/// layout coordinate       offset walk
/// (0 through 7, 0)        0 1 2 3 4 5 6 7       run length 8
/// (0 through 7, 1)        4 5 6 7 8 9 10 11     shifted by 4
///
/// greatest common divisor of 8 and 4 = 4 offset units
/// ```
///
/// The result is `4` because width 4 divides both the eight-unit run and its
/// four-unit shift. Four-unit vector chunks therefore have compatible starts.
/// This does **not** prove that the base pointer is aligned; the compiler must
/// check the pointer separately before selecting a vector instruction.
///
/// The unit is whatever the layout's strides count: elements for an
/// element-stride layout or bytes for a byte-stride layout. A negative stride
/// returns `1` because this analysis cannot prove a larger factor for a
/// backwards walk.
pub fn max_alignment(layout: &Layout) -> Result<i64> {
    let shapes = layout.shape.leaves();
    let strides = layout.stride.leaves();
    if shapes.iter().any(|&shape| shape <= 0) {
        return Err(AlgebraError(
            "max_alignment: every shape extent must be positive".into(),
        ));
    }
    if strides.iter().any(|&stride| stride < 0) {
        return Ok(1);
    }

    let flat = coalesce(layout);
    let inverse = right_inverse(&flat)?;
    let permuted = logical_divide(&flat, &inverse)?;
    let modes = permuted.modes();
    if modes.len() != 2 {
        return Err(AlgebraError(
            "max_alignment: expected vector and remainder modes".into(),
        ));
    }

    let vector_size = modes[0].size();
    let rest_stride_gcd = modes[1]
        .stride
        .leaves()
        .into_iter()
        .fold(0, gcd_nonnegative);
    Ok(gcd_nonnegative(vector_size, rest_stride_gcd).max(1))
}

fn gcd_nonnegative(a: i64, b: i64) -> i64 {
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    // Both inputs came from i64 layout values, so their greatest common
    // divisor also fits in i64.
    a as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn l(text: &str) -> Layout {
        text.parse().unwrap()
    }

    /// Small layouts whose complete coordinate maps are cheap to enumerate.
    fn corpus() -> Vec<Layout> {
        [
            "4:1",
            "8:1",
            "4:2",
            "6:4",
            "(2,3):(3,1)",
            "(2,3):(1,2)",
            "(4,4):(4,1)",
            "(4,4):(1,4)",
            "(2,2,2):(4,2,1)",
            "(2,2,2):(1,2,4)",
            "(8,4):(4,1)",
            "(2,4):(1,8)",
        ]
        .iter()
        .map(|t| l(t))
        .collect()
    }

    #[test]
    fn coalesce_examples() {
        assert_eq!(coalesce(&l("(2,4):(1,2)")).to_string(), "8:1");
        assert_eq!(coalesce(&l("(2,1,4):(1,7,2)")).to_string(), "8:1");
        assert_eq!(coalesce(&l("(2,4):(4,1)")).to_string(), "(2,4):(4,1)");
        assert_eq!(coalesce(&l("1:0")).to_string(), "1:0");
    }

    /// Checks the definition of composition at every coordinate:
    /// `result(i) == a(b(i))`.
    ///
    /// This test uses a one-mode `b`. Its stride must line up with a boundary
    /// in flattened `a`. For shape modes `[s0, s1, ...]`, those boundaries are
    /// `[1, s0, s0*s1, ...]`: the number of coordinates covered before the
    /// next mode changes. The end of `b` must also divide one of those
    /// boundaries exactly.
    ///
    /// Other tests cover multi-mode composition through divide, product, and
    /// TV construction. Imperfect final tiles are deliberately excluded here
    /// because composition may describe their overhang; predication, rather
    /// than the pointwise identity, makes those tiles safe.
    #[test]
    fn composition_is_definitional() {
        let mut checked = 0;
        for a in corpus() {
            let prefix: Vec<i64> = {
                let mut p = vec![1i64];
                for (s, _) in super::flat_modes(&super::coalesce(&a)) {
                    p.push(p.last().unwrap() * s);
                }
                p
            };
            let total = a.size();
            for n in [1i64, 2, 3, 4, 6, 8] {
                for d in [0i64, 1, 2, 3, 4, 6, 8] {
                    let b = Layout {
                        shape: IntTuple::Leaf(n),
                        stride: IntTuple::Leaf(d),
                    };
                    if b.cosize() > total {
                        continue;
                    }
                    let admissible = d == 0
                        || (prefix.contains(&d)
                            && prefix.iter().any(|&p| p >= n * d && p % (n * d) == 0));
                    if !admissible {
                        continue;
                    }
                    let Ok(r) = composition(&a, &b) else {
                        continue; // divisibility precondition not met: fine
                    };
                    assert_eq!(r.size(), b.size(), "size mismatch for {a} o {b}");
                    for i in 0..b.size() {
                        let expect = a.call(&IntTuple::Leaf(b.call(&IntTuple::Leaf(i))));
                        let got = r.call(&IntTuple::Leaf(i));
                        assert_eq!(got, expect, "({a}) o ({b}) at {i}");
                    }
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 60,
            "corpus too weak: only {checked} pairs checked"
        );
    }

    #[test]
    fn composition_known_examples() {
        // A documented nested example: select a column-like walk through a
        // row-major matrix, then preserve the resulting submode structure.
        assert_eq!(
            composition(&l("(6,2):(8,2)"), &l("(4,3):(3,1)"))
                .unwrap()
                .to_string(),
            "((2,2),3):((24,2),8)"
        );
        // Every top-level mode on the right remains a top-level result mode.
        let r = composition(&l("(8,8):(8,1)"), &l("(4,2):(2,1)")).unwrap();
        for i in 0..8 {
            assert_eq!(
                r.call(&IntTuple::Leaf(i)),
                l("(8,8):(8,1)").call(&IntTuple::Leaf(l("(4,2):(2,1)").call(&IntTuple::Leaf(i))))
            );
        }
    }

    /// Joining a layout and its complement must reach each offset in `[0, M)`
    /// exactly once.
    #[test]
    fn complement_is_definitional() {
        let cases = [
            ("4:1", 24),
            ("4:2", 24),
            ("6:4", 24),
            ("(2,3):(3,1)", 12),
            ("(2,2):(1,8)", 16),
            ("(4,4):(4,1)", 32),
        ];
        for (text, m) in cases {
            let a = l(text);
            let c = complement(&a, m).unwrap();
            let combined = Layout::from_modes(vec![a.clone(), c.clone()]);
            let total = combined.size();
            assert_eq!(total, m, "complement of {text} in {m} has wrong total size");
            let mut seen: Vec<i64> = (0..total)
                .map(|i| combined.call(&IntTuple::Leaf(i)))
                .collect();
            seen.sort_unstable();
            let expect: Vec<i64> = (0..m).collect();
            assert_eq!(
                seen, expect,
                "complement of {text} in {m} is not a bijection"
            );
        }
        // Even offsets plus a parity coordinate cover all 24 offsets.
        assert_eq!(
            complement(&l("4:2"), 24).unwrap().to_string(),
            "(2,3):(1,8)"
        );
    }

    /// A right inverse must turn every reachable consecutive offset back into
    /// a coordinate that produces that offset.
    #[test]
    fn right_inverse_is_definitional() {
        for a in corpus() {
            let r = right_inverse(&a).unwrap();
            for i in 0..r.size() {
                assert_eq!(
                    a.call(&IntTuple::Leaf(r.call(&IntTuple::Leaf(i)))),
                    i,
                    "right_inverse of {a} fails at {i}"
                );
            }
            // A compact one-to-one layout has no gap, so its inverse covers
            // the full logical cell count.
            if a.cosize() == a.size() {
                assert_eq!(r.size(), a.size(), "right_inverse of compact {a} not full");
            }
        }
    }

    /// A left inverse must recover every original coordinate for a layout with
    /// no duplicate offsets.
    #[test]
    fn left_inverse_is_definitional() {
        for a in corpus() {
            // Keep only layouts where every coordinate reaches a distinct
            // offset; duplicate offsets cannot be undone.
            let mut image: Vec<i64> = (0..a.size()).map(|i| a.call(&IntTuple::Leaf(i))).collect();
            image.sort_unstable();
            image.dedup();
            if image.len() as i64 != a.size() {
                continue;
            }
            let r = left_inverse(&a).unwrap();
            for i in 0..a.size() {
                assert_eq!(
                    r.call(&IntTuple::Leaf(a.call(&IntTuple::Leaf(i)))),
                    i,
                    "left_inverse of {a} fails at {i}"
                );
            }
        }
    }

    /// Zipped division changes the coordinate structure, not the address. The
    /// inside-tile and tile-number coordinates are recombined mode by mode and
    /// compared with the original layout.
    #[test]
    fn zipped_divide_is_definitional() {
        let a = l("(8,6):(6,1)");
        let r = zipped_divide_by_shape(&a, &[4, 3]).unwrap();
        for t0 in 0..4 {
            for t1 in 0..3 {
                for r0 in 0..2 {
                    for r1 in 0..2 {
                        let coord = IntTuple::Tuple(vec![
                            IntTuple::Tuple(vec![IntTuple::Leaf(t0), IntTuple::Leaf(t1)]),
                            IntTuple::Tuple(vec![IntTuple::Leaf(r0), IntTuple::Leaf(r1)]),
                        ]);
                        let orig = IntTuple::Tuple(vec![
                            IntTuple::Leaf(t0 + 4 * r0),
                            IntTuple::Leaf(t1 + 3 * r1),
                        ]);
                        assert_eq!(
                            r.call(&coord),
                            a.call(&orig),
                            "at t=({t0},{t1}) r=({r0},{r1})"
                        );
                    }
                }
            }
        }
    }

    /// Checks the documented six-thread, four-values-per-thread example and
    /// then verifies that its 24 assignments cover the `4 x 6` tile exactly.
    #[test]
    fn make_layout_tv_matches_dsl_docstring() {
        let thr = l("(2,3):(3,1)");
        let val = l("(2,2):(2,1)");
        let (tiler, tv) = make_layout_tv(&thr, &val).unwrap();
        assert_eq!(tiler, vec![4, 6]);
        assert_eq!(tv.to_string(), "((3,2),(2,2)):((8,2),(4,1))");
        // No two `(thread, value)` pairs may own the same tile cell, and no
        // cell may be left unowned.
        let mut seen: Vec<i64> = (0..tv.size())
            .map(|i| tv.call(&IntTuple::Leaf(i)))
            .collect();
        seen.sort_unstable();
        let expect: Vec<i64> = (0..24).collect();
        assert_eq!(seen, expect);
    }

    #[test]
    fn logical_product_repeats_the_block() {
        // Six copies of a four-cell block must reach all 24 offsets exactly
        // once.
        let r = logical_product(&l("4:1"), &l("6:1")).unwrap();
        let mut seen: Vec<i64> = (0..r.size()).map(|i| r.call(&IntTuple::Leaf(i))).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn max_alignment_matches_cutegen_static_examples() {
        let cases = [
            ("(1,8):(1,1)", 8),
            ("(4,2,(8,2)):(512,1,(8,8192))", 2),
            ("((1,(2))):((1,(0)))", 1),
            ("(8,16):(1,4)", 4),
            ("(8,16):(1,6)", 2),
            ("(8,16):(1,8)", 128),
        ];
        for (text, expected) in cases {
            assert_eq!(max_alignment(&l(text)).unwrap(), expected, "{text}");
        }
    }

    #[test]
    fn max_alignment_is_conservative_for_negative_strides() {
        assert_eq!(max_alignment(&l("8:-1")).unwrap(), 1);
    }
}
