/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Simplification, composition, inverse maps, and reshaping.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{IntTuple, Layout};

use super::{AlgebraError, Result, flat_modes, layout_from_flat};

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

/// Round a nonnegative extent upward to a whole positive-divisor tile count.
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
/// [`logical_divide`](super::tiling::logical_divide) uses the returned [modes](crate#modes) for its “which
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
/// describes their three value numbers. [`raked_product`](super::tiling::raked_product) combines them into
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
/// [`make_layout_tv`](super::thread_value::make_layout_tv) performs this construction in three steps:
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
