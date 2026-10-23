/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread ownership, exact coverage and copy-atom contiguity.

use super::{MAX_STATIC_VALIDATION_ELEMENTS, ValidationError, ValidationResult};
use crate::{ComposedLayout, IntTuple, Layout, OffsetUnit};
use alloc::{vec, vec::Vec};

/// Require a compact thread layout.
///
/// “Compact” means the layout produces every thread ID from zero through
/// `size - 1` exactly once. It describes membership, not physical ordering, so
/// a permutation is valid while a gap or duplicate is not:
///
/// ```text
/// flat coordinate                0  1  2  3
/// unpacked pair for tuple shapes (0,0) (1,0) (0,1) (1,1)
///
/// layout (2,2):(2,1) -> thread ID 0  2  1  3   valid: 0,1,2,3 appear once
/// layout       4:2   -> thread ID 0  2  4  6   invalid: 1 and 3 are missing
/// layout (2,2):(1,1) -> thread ID 0  1  1  2   invalid: ID 1 appears twice
/// ```
pub fn validate_compact_thread_layout(thread_layout: &Layout) -> ValidationResult<()> {
    let thread_count = thread_layout
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("thread layout size"))?;
    let thread_count_usize = host_count("thread count", thread_count)?;
    let mut owners: Vec<Option<i64>> = vec![None; thread_count_usize];

    for coordinate in 0..thread_count {
        let thread = thread_layout
            .checked_call(&IntTuple::Leaf(coordinate))
            .ok_or(ValidationError::ArithmeticOverflow("thread-layout address"))?;
        if !(0..thread_count).contains(&thread) {
            return Err(ValidationError::ThreadOutOfRange {
                coordinate,
                thread,
                thread_count,
            });
        }
        let slot = &mut owners[thread as usize];
        if let Some(first_coordinate) = *slot {
            return Err(ValidationError::DuplicateThread {
                thread,
                first_coordinate,
                second_coordinate: coordinate,
            });
        }
        *slot = Some(coordinate);
    }
    Ok(())
}

/// Prove exact coverage of a tile by a thread-value (TV) assignment.
///
/// A TV layout has two top-level [modes](crate#modes): thread ID and that
/// thread's local value number. Exact coverage requires a one-to-one match:
///
/// ```text
/// tv = (2,2):(2,1)
///
///                 value 0   value 1
/// thread 0           0         1
/// thread 1           2         3
///                     ^ numbers are tile cells; every cell appears once
/// ```
///
/// Wrong mode sizes, a mismatched assignment count, an out-of-range cell, or a
/// duplicate owner produces a specific [`ValidationError`]. When assignment
/// count equals cell count, an unowned cell necessarily comes with a duplicate
/// or out-of-range assignment and is rejected by that earlier check.
pub fn validate_tv_exact_coverage(
    tv: &Layout,
    thread_count: i64,
    values_per_thread: i64,
    tile_elements: i64,
) -> ValidationResult<()> {
    let thread_count = positive_count("thread count", thread_count)?;
    let values_per_thread = positive_count("values per thread", values_per_thread)?;
    let tile_elements = positive_count("tile elements", tile_elements)?;

    let modes = tv.modes();
    if modes.len() != 2 {
        return Err(ValidationError::TvRank {
            actual: modes.len(),
        });
    }
    let thread_mode_size = modes[0]
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("TV thread-mode size"))?;
    if thread_mode_size != thread_count {
        return Err(ValidationError::TvModeSize {
            mode: "thread",
            expected: thread_count,
            actual: thread_mode_size,
        });
    }
    let value_mode_size = modes[1]
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("TV value-mode size"))?;
    if value_mode_size != values_per_thread {
        return Err(ValidationError::TvModeSize {
            mode: "value",
            expected: values_per_thread,
            actual: value_mode_size,
        });
    }

    let assignments = thread_count
        .checked_mul(values_per_thread)
        .ok_or(ValidationError::ArithmeticOverflow("TV assignment count"))?;
    if assignments != tile_elements {
        return Err(ValidationError::TvCardinality {
            assignments,
            tile_elements,
        });
    }

    let mut owners: Vec<Option<(i64, i64)>> =
        vec![None; host_count("tile elements", tile_elements)?];
    for thread in 0..thread_count {
        for value in 0..values_per_thread {
            let cell = tv
                .checked_call(&tv_coord(thread, value))
                .ok_or(ValidationError::ArithmeticOverflow("TV-layout address"))?;
            if !(0..tile_elements).contains(&cell) {
                return Err(ValidationError::TvOutOfRange {
                    thread,
                    value,
                    cell,
                    tile_elements,
                });
            }
            let slot = &mut owners[cell as usize];
            if let Some(first_owner) = *slot {
                return Err(ValidationError::DuplicateCell {
                    cell,
                    first_owner,
                    second_owner: (thread, value),
                });
            }
            *slot = Some((thread, value));
        }
    }
    if let Some(cell) = owners.iter().position(Option::is_none) {
        return Err(ValidationError::MissingCell { cell: cell as i64 });
    }
    Ok(())
}

/// Check a TV layout produced by [`make_layout_tv`](crate::algebra::make_layout_tv) against its source
/// layouts.
///
/// `tiler_shape` contains one extent per tile [mode](crate#modes). Their product
/// is the tile cell count. This function first requires compact thread IDs,
/// then requires the returned `(thread, local value) -> cell` map to cover that
/// count exactly.
pub fn validate_tv_layout(
    thread_layout: &Layout,
    value_layout: &Layout,
    tiler_shape: &[i64],
    tv: &Layout,
) -> ValidationResult<()> {
    validate_compact_thread_layout(thread_layout)?;
    let tile_elements = tiler_shape.iter().try_fold(1i64, |product, &extent| {
        let extent = positive_count("tiler extent", extent)?;
        product
            .checked_mul(extent)
            .ok_or(ValidationError::ArithmeticOverflow("tiler element count"))
    })?;
    let thread_count = thread_layout
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("thread layout size"))?;
    let values_per_thread = value_layout
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("value layout size"))?;
    validate_tv_exact_coverage(tv, thread_count, values_per_thread, tile_elements)
}

/// Require the cooperative copy to contain only whole groups described by its
/// [copy atom](crate#atoms).
///
/// A [copy atom](crate#atoms) describes how many threads participate and how
/// many local values each contributes to one hardware transaction. Both copy
/// counts must divide evenly:
///
/// ```text
/// 256 copy threads / 8 atom threads = 32 whole groups   valid
/// 250 copy threads / 8 atom threads = 31 + remainder   invalid
/// ```
///
/// `CopyG2S<N>` uses one thread per atom, but this helper also supports later
/// atom kinds that group several threads.
pub fn validate_atom_compatibility(
    copy_threads: i64,
    copy_values_per_thread: i64,
    atom_threads: i64,
    atom_values_per_thread: i64,
) -> ValidationResult<()> {
    let copy_threads = positive_count("copy thread count", copy_threads)?;
    let copy_values_per_thread = positive_count("copy values per thread", copy_values_per_thread)?;
    let atom_threads = positive_count("atom thread count", atom_threads)?;
    let atom_values_per_thread = positive_count("atom values per thread", atom_values_per_thread)?;

    if copy_threads % atom_threads != 0 {
        return Err(ValidationError::AtomDoesNotDivide {
            axis: "thread",
            copy_count: copy_threads,
            atom_count: atom_threads,
        });
    }
    if copy_values_per_thread % atom_values_per_thread != 0 {
        return Err(ValidationError::AtomDoesNotDivide {
            axis: "value",
            copy_count: copy_values_per_thread,
            atom_count: atom_values_per_thread,
        });
    }
    Ok(())
}

/// Prove that the transaction described by each one-thread
/// [copy atom](crate#atoms) maps to one aligned, consecutive byte range.
///
/// The transaction described by a 16-byte copy atom must map like this:
///
/// ```text
/// byte in atom:      0   1   2   ... 15
/// physical offset:  P  P+1 P+2  ... P+15
///                    ^ P must also be 16-byte aligned
/// ```
///
/// For a smaller concrete example, let two threads each own two four-byte
/// values:
///
/// ```text
/// tv     = (2,2):(2,1)     memory = 4:4 byte offsets
///
/// owner       tile cells   value-start bytes   complete 8-byte atom
/// thread 0       0, 1           0, 4             0 through 7
/// thread 1       2, 3           8, 12            8 through 15
/// ```
///
/// Both atom starts are 8-byte aligned, and every byte through the end of each
/// interval follows consecutively. An interleaved TV map such as
/// `(2,2):(1,2)` would instead give each thread cells `0,2` or `1,3`, so the
/// check would reject it for this simple memory layout.
///
/// The proof evaluates the complete byte-addressed memory map, including its
/// fixed offset and swizzle. It does not reject solely because
/// [the swizzle's general alignment estimate](crate::Swizzle::max_alignment)
/// is smaller; evaluating the exact map can prove a wider interval in a
/// special case.
///
/// This function is for `CopyG2S<N>`, where one thread owns each atom.
/// Multi-thread atom kinds also need [`validate_atom_compatibility`] plus a
/// compiler check specific to that atom's ownership rule.
pub fn validate_per_thread_atom_contiguity(
    tv: &Layout,
    thread_count: i64,
    values_per_thread: i64,
    atom_bytes: i64,
    element_bytes: i64,
    memory: &ComposedLayout,
    base_alignment_bytes: i64,
) -> ValidationResult<()> {
    let atom_bytes = positive_count("atom bytes", atom_bytes)?;
    let element_bytes = positive_count("element bytes", element_bytes)?;
    if atom_bytes % element_bytes != 0 {
        return Err(ValidationError::AtomSize {
            atom_bytes,
            element_bytes,
        });
    }
    if memory.unit() != OffsetUnit::Bytes {
        return Err(ValidationError::AtomNeedsByteLayout);
    }
    let memory_size = memory
        .outer()
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("memory layout size"))?;
    validate_tv_exact_coverage(tv, thread_count, values_per_thread, memory_size)?;

    let values_per_atom = atom_bytes / element_bytes;
    let values_per_atom_usize = host_count("atom values per thread", values_per_atom)?;
    validate_atom_compatibility(thread_count, values_per_thread, 1, values_per_atom)?;

    let base_alignment_bytes = positive_count("base alignment", base_alignment_bytes)?;
    if base_alignment_bytes % atom_bytes != 0 {
        return Err(ValidationError::BaseAlignment {
            required: atom_bytes,
            guaranteed: base_alignment_bytes,
        });
    }

    for thread in 0..thread_count {
        for first_value in (0..values_per_thread).step_by(values_per_atom_usize) {
            let first_cell = tv
                .checked_call(&tv_coord(thread, first_value))
                .ok_or(ValidationError::ArithmeticOverflow("TV atom start"))?;
            let first_offset = memory
                .checked_call(&IntTuple::Leaf(first_cell))
                .ok_or(ValidationError::ArithmeticOverflow("memory atom start"))?;
            if first_offset.rem_euclid(atom_bytes) != 0 {
                return Err(ValidationError::AtomStartMisaligned {
                    thread,
                    value: first_value,
                    offset: first_offset,
                    atom_bytes,
                });
            }

            for byte_in_atom in 0..atom_bytes {
                let value_delta = byte_in_atom / element_bytes;
                let byte_in_element = byte_in_atom % element_bytes;
                let value = first_value
                    .checked_add(value_delta)
                    .ok_or(ValidationError::ArithmeticOverflow("TV atom value"))?;
                let cell = tv
                    .checked_call(&tv_coord(thread, value))
                    .ok_or(ValidationError::ArithmeticOverflow("TV atom address"))?;
                let actual = memory
                    .checked_call_with_outer_delta(&IntTuple::Leaf(cell), byte_in_element)
                    .ok_or(ValidationError::ArithmeticOverflow("memory atom address"))?;
                let expected = first_offset.checked_add(byte_in_atom).ok_or(
                    ValidationError::ArithmeticOverflow("expected atom byte address"),
                )?;
                if actual != expected {
                    return Err(ValidationError::AtomNotContiguous {
                        thread,
                        value: first_value,
                        byte_in_atom,
                        expected,
                        actual,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Build the tuple coordinate selecting one thread and its local value.
pub(super) fn tv_coord(thread: i64, value: i64) -> IntTuple {
    IntTuple::Tuple(vec![IntTuple::Leaf(thread), IntTuple::Leaf(value)])
}

/// Require a positive bounded count before exhaustive layout enumeration.
pub(super) fn positive_count(name: &'static str, value: i64) -> ValidationResult<i64> {
    if value <= 0 {
        Err(ValidationError::InvalidCount { name, value })
    } else {
        Ok(value)
    }
}

/// Convert a validated count to the host indexing type without truncation.
pub(super) fn host_count(name: &'static str, value: i64) -> ValidationResult<usize> {
    if value > MAX_STATIC_VALIDATION_ELEMENTS {
        return Err(ValidationError::CountTooLarge { name, value });
    }
    usize::try_from(value).map_err(|_| ValidationError::CountTooLarge { name, value })
}
