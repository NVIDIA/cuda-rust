/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Choosing a tile shape from a target memory transaction width.
//!
//! Transaction width is set by the alignment of the element type, so it is a
//! consequence of the data layout rather than something a kernel can request.
//! That makes the useful question the inverted one: *given* the width I want,
//! what does the tile shape have to be?
//!
//! The `gemm_sol` example already reasons in that direction, by hand, in a
//! comment:
//!
//! ```text
//! Each K-group: 128 rows x 8 elements x 2 bytes = 2048 bytes
//! ```
//!
//! Read as a derivation: a 128-bit transaction of f16 is 8 elements, 128 rows
//! then move 1024 elements per group. Run the same arithmetic against a `K = 32`
//! tile and it forces `M >= 1024 / 32 = 32` - which is how CUTLASS states the
//! constraint for a row-major A operand at 128 threads and 8-element access.
//! Every step is arithmetic on the width, the element size, and the launch
//! shape - which is what this module computes, so it does not have to be
//! re-derived in a comment per kernel.
//!
//! Everything here is `const`, so a plan can drive a `const` generic or sit in a
//! `const {}` assertion and turn a tile shape that cannot reach the intended
//! width into a compile error.
//!
//! # Scope
//!
//! This answers "what shape does this width need". It does not decide whether
//! the width is *achievable*, which depends on the element type and, for loads,
//! on how the value is used.
//!
//! Measured on sm_86, with the element type 16-byte aligned throughout:
//!
//! ```text
//! how the element is used                     LDG          STG
//! never decomposed                            LDG.E.128    STG.E.128
//! lanes read, whole value assembled+stored    LDG.E x4     STG.E.128
//! lanes read, scalar stored                   LDG.E x4     STG.E
//! ```
//!
//! So `align` is a necessary condition, and for loads not a sufficient one:
//! touching individual lanes lets the optimiser split the local apart and load
//! the lanes separately whatever the alignment. Stores have no such condition.
//!
//! That limit is specific to a *direct blocked* load, where the wide value
//! itself feeds the per-element arithmetic. CUB's `BLOCK_LOAD_VECTORIZE`
//! reaches the width for such kernels anyway, by splitting the layouts: load
//! *striped* with wide loads (where nothing decomposes the value), then
//! exchange through shared memory into the blocked arrangement the arithmetic
//! wants. Decomposition still happens, but on registers that never came from
//! a wide load. cuda-oxide has no striped/blocked exchange primitive yet, so
//! a decomposing kernel currently cannot reach the width - the gap is the
//! missing layout change, not anything about alignment.
//!
//! A plan therefore describes the shape a width *needs*, not a width the kernel
//! is guaranteed to get - the data has to move as whole elements as well.
//! See the `vectorization` example for the alignment half.
//!
//! # Coalescing is the same kind of claim
//!
//! [`lines_touched`] and friends answer the global-memory question that
//! [`crate::swizzle::conflict_degree`] answers for shared memory: how many
//! hardware transactions a static access pattern costs. They carry the identical
//! caveat, and it is worth stating rather than implying.
//!
//! **These are necessary-condition checkers.** They bound what the *layout*
//! permits; they cannot report what codegen delivers. This module already
//! contains the proof: it reports `TXN_128` for a tile shape whose measured SASS
//! is four scalar `LDG.E`, because SROA splits any decomposed value. Likewise
//! `wasted_lines == 0` proves the pattern does not forbid a coalesced access; it
//! never proves one happened. A green assertion that is read as a guarantee is
//! worse than no assertion.

/// One 32-bit transaction (`LDG.E` / `STG.E`).
pub const TXN_32: usize = 4;
/// One 64-bit transaction (`LDG.E.64` / `STG.E.64`).
pub const TXN_64: usize = 8;
/// One 128-bit transaction (`LDG.E.128` / `STG.E.128`), the widest available.
pub const TXN_128: usize = 16;

/// What a target transaction width implies for a tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AccessPlan {
    /// Bytes moved per thread per instruction.
    pub txn_bytes: usize,
    /// Element type alignment required to reach `txn_bytes`.
    ///
    /// Necessary, and for loads not sufficient - see the module's *Scope*
    /// section.
    pub align: usize,
    /// Elements each thread moves per instruction.
    pub elems_per_thread: usize,
    /// Threads participating.
    pub threads: usize,
    /// Elements the whole block moves per instruction.
    pub elems_per_block: usize,
}

impl AccessPlan {
    /// Smallest extent of one tile axis, given the contiguous extent of the
    /// other.
    ///
    /// This models a strided operand, the CUTLASS shape rule: `other` is the
    /// contiguous extent (the row length of a row-major tile), and rows are
    /// not assumed adjacent in memory, so no access may cross a row boundary.
    /// Two divisibilities follow: each thread's access must fit a whole
    /// number of times in a row (`elems_per_thread` divides `other`), and a
    /// block-wide access must cover whole rows (`other` divides
    /// `elems_per_block`). Returns `None` when either fails, because rounding
    /// here would silently change the tile.
    ///
    /// This is the CUTLASS derivation: with 1024 elements per block-wide access
    /// and `K = 32`, the minimum `M` is 32.
    #[must_use]
    pub const fn min_extent(&self, other: usize) -> Option<usize> {
        if other == 0
            || !other.is_multiple_of(self.elems_per_thread)
            || !self.elems_per_block.is_multiple_of(other)
        {
            return None;
        }
        Some(self.elems_per_block / other)
    }

    /// Whether a tile of `tile_elems` elements can be moved by whole block-wide
    /// accesses at this width.
    #[must_use]
    pub const fn fits_tile(&self, tile_elems: usize) -> bool {
        self.elems_per_block != 0
            && tile_elems != 0
            && tile_elems.is_multiple_of(self.elems_per_block)
    }

    /// Block-wide accesses needed to move a tile of `tile_elems`.
    #[must_use]
    pub const fn passes_for_tile(&self, tile_elems: usize) -> Option<usize> {
        if !self.fits_tile(tile_elems) {
            return None;
        }
        Some(tile_elems / self.elems_per_block)
    }
}

/// Plan a tile for a target transaction width.
///
/// Returns `None` when the width is not one the hardware offers, when it is not
/// a whole number of elements, or when `threads` is zero.
///
/// # Example
///
/// ```rust
/// #![feature(f16)]
/// use cuda_device::access::{self, AccessPlan};
///
/// // 128-bit transactions of f16, across 128 threads.
/// const PLAN: AccessPlan = match access::plan::<f16>(access::TXN_128, 128) {
///     Some(plan) => plan,
///     None => panic!("128 bits is a whole number of f16"),
/// };
/// // 8 elements per thread, 1024 per block, so K = 32 forces M >= 32.
/// const MIN_M: usize = match PLAN.min_extent(32) {
///     Some(extent) => extent,
///     None => panic!("32 divides 1024"),
/// };
/// assert_eq!((PLAN.elems_per_thread, PLAN.elems_per_block, MIN_M), (8, 1024, 32));
/// ```
#[must_use]
pub const fn plan<T>(txn_bytes: usize, threads: usize) -> Option<AccessPlan> {
    plan_for_elem_size(core::mem::size_of::<T>(), txn_bytes, threads)
}

/// [`plan`] with the element size supplied directly.
///
/// Useful when the element type is not nameable in the calling context, such as
/// a packed pair whose Rust type lives in another crate.
#[must_use]
pub const fn plan_for_elem_size(
    elem_size: usize,
    txn_bytes: usize,
    threads: usize,
) -> Option<AccessPlan> {
    if threads == 0 || elem_size == 0 {
        return None;
    }
    // Only these three widths exist; a "96-bit" request is a mistake, not
    // something to round down silently.
    if txn_bytes != TXN_32 && txn_bytes != TXN_64 && txn_bytes != TXN_128 {
        return None;
    }
    // A transaction has to be a whole number of elements. An f32 cannot do a
    // 6-byte access, and an f64 cannot do a 32-bit one.
    if !txn_bytes.is_multiple_of(elem_size) {
        return None;
    }
    let elems_per_thread = txn_bytes / elem_size;
    let Some(elems_per_block) = threads.checked_mul(elems_per_thread) else {
        return None;
    };
    Some(AccessPlan {
        txn_bytes,
        align: txn_bytes,
        elems_per_thread,
        threads,
        elems_per_block,
    })
}

/// The widest transaction a tile of `tile_elems` can be moved by, across
/// `threads` threads, without a partial access.
///
/// The auditing direction, kept for comparing an existing kernel against the
/// width it could have had.
#[must_use]
pub const fn widest<T>(tile_elems: usize, threads: usize) -> Option<AccessPlan> {
    let elem_size = core::mem::size_of::<T>();
    // Widest first, so the first whole fit wins.
    let candidates = [TXN_128, TXN_64, TXN_32];
    let mut i = 0;
    while i < candidates.len() {
        if let Some(p) = plan_for_elem_size(elem_size, candidates[i], threads)
            && p.fits_tile(tile_elems)
        {
            return Some(p);
        }
        i += 1;
    }
    None
}

/// Bytes in one L1/L2 cache line. A warp's global accesses are serviced in
/// whole lines, so the count of distinct lines touched -- not the bytes
/// requested -- is what the memory system pays for.
pub const LINE_BYTES: usize = 128;

/// Bytes in one L2 sector. A line is four sectors, and a partially-used line
/// still costs whole sectors, so sector counting is the finer-grained metric.
pub const SECTOR_BYTES: usize = 32;

/// Distinct cache lines a warp's accesses touch.
///
/// The global-memory counterpart of [`swizzle::conflict_degree`], and takes the
/// same shape: element offsets per lane plus the element size. Lane `i` reads
/// `elem_size` bytes at element `offsets[i]`, so its byte span is
/// `offsets[i] * elem_size .. + elem_size`, and a lane whose span crosses a line
/// boundary is counted against both lines.
///
/// [`swizzle::conflict_degree`]: crate::swizzle::conflict_degree
///
/// # Base alignment
///
/// Line indices are computed from byte offset 0, so element offset 0 is assumed
/// to sit at the start of a 128-byte cache line. That holds for allocation
/// bases -- device allocations are at least 256-byte aligned -- but not for an
/// arbitrary sub-slice base, which can start mid-line and straddle one more
/// line than reported.
///
/// # Interpretation
///
/// Compare against [`minimum_lines`]. Equality means the pattern is as coalesced
/// as the requested bytes allow; any excess is lines fetched and partly discarded.
///
/// Returns 0 when `elem_size` is 0 -- a zero-sized element touches no memory.
///
/// # Example
///
/// ```rust
/// use cuda_device::access::{lines_touched, minimum_lines, LINE_BYTES};
///
/// // Each lane takes 4 contiguous f32, i.e. one 16-byte access per lane, and
/// // lane `i` starts where lane `i - 1` ended: element offsets 0, 1, 2, ...
/// let mut contiguous = [0usize; 32];
/// let mut lane = 0;
/// while lane < 32 {
///     contiguous[lane] = lane;
///     lane += 1;
/// }
/// // 32 lanes x 16 bytes = 512 contiguous bytes = exactly 4 lines, the floor.
/// assert_eq!(lines_touched(&contiguous, 16), 4);
/// assert_eq!(minimum_lines(32 * 16), 4);
/// assert_eq!(LINE_BYTES, 128);
/// ```
#[must_use]
pub const fn lines_touched(
    offsets: &[usize; crate::swizzle::WARP_LANES],
    elem_size: usize,
) -> usize {
    if elem_size == 0 {
        return 0;
    }
    let lanes = crate::swizzle::WARP_LANES;
    let mut count = 0;
    let mut lane = 0;
    while lane < lanes {
        let (first, last) = line_span(offsets[lane], elem_size);
        let mut line = first;
        while line <= last {
            // A line is new unless some earlier lane already spanned it. Lines
            // within one lane's own span are distinct by construction, so only
            // earlier lanes need checking -- which keeps this O(lanes^2) with no
            // scratch buffer, and therefore `const`-evaluable.
            let mut seen = false;
            let mut earlier = 0;
            while earlier < lane {
                let (f, l) = line_span(offsets[earlier], elem_size);
                if f <= line && line <= l {
                    seen = true;
                }
                earlier += 1;
            }
            if !seen {
                count += 1;
            }
            if line == usize::MAX {
                break;
            }
            line += 1;
        }
        lane += 1;
    }
    count
}

/// First and last line index a lane's access spans, saturating rather than
/// wrapping so a pathological offset cannot fold back onto a low line.
const fn line_span(elem_offset: usize, elem_size: usize) -> (usize, usize) {
    let start = elem_offset.saturating_mul(elem_size);
    let end = start.saturating_add(elem_size - 1);
    (start / LINE_BYTES, end / LINE_BYTES)
}

/// Fewest lines that could service `total_bytes`, i.e. the floor
/// [`lines_touched`] is measured against.
///
/// This is the count for a perfectly contiguous, line-aligned access. A real
/// pattern can equal it but never beat it.
#[must_use]
pub const fn minimum_lines(total_bytes: usize) -> usize {
    total_bytes.div_ceil(LINE_BYTES)
}

/// Lines fetched beyond the minimum -- bandwidth requested and discarded.
///
/// Zero means fully coalesced.
#[must_use]
pub const fn wasted_lines(
    offsets: &[usize; crate::swizzle::WARP_LANES],
    elem_size: usize,
) -> usize {
    let touched = lines_touched(offsets, elem_size);
    let floor = minimum_lines(crate::swizzle::WARP_LANES.saturating_mul(elem_size));
    touched.saturating_sub(floor)
}

/// Whether the warp touches no more lines than the requested bytes require.
///
/// Suitable for a `const {}` assertion, which turns a layout that cannot
/// coalesce into a build error:
///
/// ```rust
/// use cuda_device::access::is_fully_coalesced;
///
/// const LANES: usize = 32;
/// const CONTIGUOUS: [usize; LANES] = {
///     let mut o = [0usize; LANES];
///     let mut i = 0;
///     while i < LANES {
///         o[i] = i;
///         i += 1;
///     }
///     o
/// };
/// const _: () = assert!(is_fully_coalesced(&CONTIGUOUS, 4));
/// ```
#[must_use]
pub const fn is_fully_coalesced(
    offsets: &[usize; crate::swizzle::WARP_LANES],
    elem_size: usize,
) -> bool {
    wasted_lines(offsets, elem_size) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduces the CUTLASS threadblock-shape derivation quoted in the module
    /// docs. If this drifts, the module's stated justification is wrong.
    #[test]
    fn reproduces_the_cutlass_threadblock_derivation() {
        // f16 is 2 bytes; a 128-bit transaction is therefore 8 elements.
        let p = plan_for_elem_size(2, TXN_128, 128).unwrap();
        assert_eq!(
            p.elems_per_thread, 8,
            "128-bit of f16 is an 8-element access"
        );
        assert_eq!(p.elems_per_block, 1024, "128 threads x 8 elements");
        assert_eq!(p.align, 16, "128-bit needs 16-byte alignment");
        // "ThreadblockShape M=32 is the minimum ... with 128 threads x
        // 8-element access", against K = 32.
        assert_eq!(p.min_extent(32), Some(32));
    }

    #[test]
    fn element_size_scales_elements_per_thread() {
        // Same width, three element sizes.
        assert_eq!(
            plan_for_elem_size(4, TXN_128, 32).unwrap().elems_per_thread,
            4
        );
        assert_eq!(
            plan_for_elem_size(2, TXN_128, 32).unwrap().elems_per_thread,
            8
        );
        assert_eq!(
            plan_for_elem_size(8, TXN_128, 32).unwrap().elems_per_thread,
            2
        );
        // f64 cannot do a 32-bit access: it is not a whole number of elements.
        assert_eq!(plan_for_elem_size(8, TXN_32, 32), None);
    }

    /// A width the hardware does not offer must be refused rather than rounded,
    /// otherwise the plan would describe an access that is never emitted.
    #[test]
    fn rejects_widths_that_do_not_exist() {
        for bad in [1usize, 2, 3, 6, 12, 24, 32, 0] {
            assert_eq!(
                plan_for_elem_size(4, bad, 32),
                None,
                "{bad}-byte transaction must be refused"
            );
        }
    }

    /// An extent that does not divide the block-wide access is refused, because
    /// rounding it would quietly change the tile the caller asked for.
    #[test]
    fn refuses_extents_that_do_not_divide() {
        let p = plan_for_elem_size(4, TXN_128, 128).unwrap(); // 512 elems/block
        assert_eq!(p.min_extent(32), Some(16));
        assert_eq!(p.min_extent(64), Some(8));
        assert_eq!(p.min_extent(48), None, "48 does not divide 512");
        assert_eq!(p.min_extent(0), None);
    }

    /// A row shorter than one thread's access is shape-impossible on a
    /// strided operand: the access would cross the row boundary. So `other`
    /// must be a multiple of `elems_per_thread`, not merely a divisor of
    /// `elems_per_block`.
    #[test]
    fn refuses_extents_shorter_than_one_thread_access() {
        // f16 at 128-bit across 128 threads: 8 elements per thread, 1024 per
        // block.
        let p = plan_for_elem_size(2, TXN_128, 128).unwrap();
        // 4 divides 1024, but a row of 4 f16 (8 bytes) cannot hold one
        // thread's 16-byte access.
        assert_eq!(p.min_extent(4), None, "4 f16 cannot hold a 128-bit access");
        // The smallest legal extent is exactly one thread access: 8 elements,
        // for 1024 / 8 = 128 rows.
        assert_eq!(p.min_extent(8), Some(128));
    }

    #[test]
    fn counts_passes_over_a_tile() {
        let p = plan_for_elem_size(4, TXN_128, 128).unwrap(); // 512 elems/block
        assert_eq!(p.passes_for_tile(512), Some(1));
        assert_eq!(p.passes_for_tile(2048), Some(4));
        assert_eq!(p.passes_for_tile(500), None, "partial access");
        assert!(p.fits_tile(1024) && !p.fits_tile(1023));
    }

    /// The auditing direction: the widest whole fit, not merely a legal one.
    #[test]
    fn widest_picks_the_largest_whole_fit() {
        // 32 threads of f32: 128-bit needs 128 elements per access.
        assert_eq!(widest::<f32>(128, 32).unwrap().txn_bytes, TXN_128);
        // 64 elements cannot take a 128-bit block access, but fits 64-bit.
        assert_eq!(widest::<f32>(64, 32).unwrap().txn_bytes, TXN_64);
        // 32 elements: one f32 each.
        assert_eq!(widest::<f32>(32, 32).unwrap().txn_bytes, TXN_32);
        // Nothing divides 33.
        assert_eq!(widest::<f32>(33, 32), None);
    }

    /// Usable in const position, which is the point: a tile shape that cannot
    /// reach the intended width should fail to compile, not to benchmark.
    #[test]
    fn plan_is_usable_at_compile_time() {
        const P: AccessPlan = match plan_for_elem_size(4, TXN_128, 256) {
            Some(p) => p,
            None => panic!("128-bit f32 across 256 threads is representable"),
        };
        const MIN_M: usize = match P.min_extent(64) {
            Some(m) => m,
            None => panic!("64 divides 1024"),
        };
        const _: () = assert!(MIN_M == 16);
        assert_eq!(P.elems_per_block, 1024);
        assert_eq!(MIN_M, 16);
    }

    const fn ramp(step: usize) -> [usize; crate::swizzle::WARP_LANES] {
        let mut o = [0usize; crate::swizzle::WARP_LANES];
        let mut i = 0;
        while i < crate::swizzle::WARP_LANES {
            o[i] = i * step;
            i += 1;
        }
        o
    }

    /// The measured case this was built for: each lane takes 4 contiguous f32,
    /// i.e. one 16-byte access per lane laid end to end, so a warp covers 512
    /// contiguous bytes -- exactly 4 lines, the floor.
    #[test]
    fn a_contiguous_vector_access_is_minimal() {
        let offsets = ramp(1);
        assert_eq!(lines_touched(&offsets, 16), 4);
        assert_eq!(minimum_lines(crate::swizzle::WARP_LANES * 16), 4);
        assert_eq!(wasted_lines(&offsets, 16), 0);
        assert!(is_fully_coalesced(&offsets, 16));
    }

    /// The distinction that makes the metric worth having. Striding by 4 elements
    /// while reading only one of them requests the same 128 bytes as a
    /// contiguous single-element access, but spreads them over 4 lines -- so 3
    /// lines are fetched and discarded. `AccessPlan` cannot see this: the tile
    /// shape is identical either way.
    #[test]
    fn a_strided_scalar_access_wastes_lines() {
        let offsets = ramp(4);
        assert_eq!(lines_touched(&offsets, 4), 4);
        assert_eq!(minimum_lines(crate::swizzle::WARP_LANES * 4), 1);
        assert_eq!(wasted_lines(&offsets, 4), 3);
        assert!(!is_fully_coalesced(&offsets, 4));
    }

    /// One f32 per lane: 128 contiguous bytes, one line.
    #[test]
    fn one_element_per_lane_is_a_single_line() {
        let offsets = ramp(1);
        assert_eq!(lines_touched(&offsets, 4), 1);
        assert!(is_fully_coalesced(&offsets, 4));
    }

    /// A stride that puts every lane on its own line is the worst case: 32 lines
    /// fetched to deliver 128 bytes, so 31 are wasted.
    #[test]
    fn a_line_sized_stride_wastes_every_line_but_one() {
        let offsets = ramp(LINE_BYTES / 4); // 32 f32 apart = 128 B apart
        assert_eq!(lines_touched(&offsets, 4), 32);
        assert_eq!(minimum_lines(crate::swizzle::WARP_LANES * 4), 1);
        assert_eq!(wasted_lines(&offsets, 4), 31);
        assert!(!is_fully_coalesced(&offsets, 4));
    }

    /// Every lane at the same offset is a broadcast: one line, and the floor for
    /// 32 lanes of f32 is also one, so it counts as coalesced.
    #[test]
    fn a_broadcast_touches_one_line() {
        let offsets = [7usize; crate::swizzle::WARP_LANES];
        assert_eq!(lines_touched(&offsets, 4), 1);
        assert!(is_fully_coalesced(&offsets, 4));
    }

    /// A 16-byte element straddling a line boundary is charged to both lines.
    #[test]
    fn an_access_crossing_a_boundary_counts_both_lines() {
        // A 16-byte element at element offset 7 occupies bytes 112..=127, which
        // ends exactly on the line edge -- one line.
        let aligned = [7usize; crate::swizzle::WARP_LANES];
        assert_eq!(lines_touched(&aligned, 16), 1, "112..128 stays in line 0");

        // A 12-byte element at element offset 10 occupies 120..=131, so it is
        // charged to both lines. Element units cannot express a 1-byte shift,
        // hence the odd element size.
        let straddle = [10usize; crate::swizzle::WARP_LANES];
        assert_eq!(
            lines_touched(&straddle, 12),
            2,
            "120..132 spans lines 0 and 1"
        );
    }

    /// A zero-sized element touches no memory.
    #[test]
    fn zero_sized_elements_touch_nothing() {
        let offsets = ramp(1);
        assert_eq!(lines_touched(&offsets, 0), 0);
    }

    /// The whole point is `const` evaluation, so prove it happens.
    #[test]
    fn the_metrics_are_const_evaluable() {
        const OFFSETS: [usize; crate::swizzle::WARP_LANES] = ramp(1);
        const TOUCHED: usize = lines_touched(&OFFSETS, 16);
        const WASTED: usize = wasted_lines(&OFFSETS, 16);
        const _: () = assert!(is_fully_coalesced(&OFFSETS, 16));
        assert_eq!(TOUCHED, 4);
        assert_eq!(WASTED, 0);
    }
}
