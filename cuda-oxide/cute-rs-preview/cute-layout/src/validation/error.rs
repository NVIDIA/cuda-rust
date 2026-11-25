/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Errors returned by individual static layout proofs.

use core::fmt;

/// Failure reported by one reusable, low-level layout check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A count such as threads, values, or bytes was zero or negative.
    InvalidCount {
        /// Name of the invalid quantity.
        name: &'static str,
        /// Invalid value supplied by the caller.
        value: i64,
    },
    /// A count exceeds the safe exhaustive-enumeration limit.
    CountTooLarge {
        /// Name of the oversized quantity.
        name: &'static str,
        /// Count that would have been enumerated.
        value: i64,
    },
    /// A thread-layout coordinate produced an ID outside the block's range.
    ThreadOutOfRange {
        /// Logical coordinate evaluated in the thread layout.
        coordinate: i64,
        /// Thread ID produced by that coordinate.
        thread: i64,
        /// Number of threads, whose valid IDs are `0..thread_count`.
        thread_count: i64,
    },
    /// Two thread-layout coordinates produced the same thread ID.
    DuplicateThread {
        /// Repeated thread ID.
        thread: i64,
        /// First coordinate that produced the ID.
        first_coordinate: i64,
        /// Second coordinate that produced the ID.
        second_coordinate: i64,
    },
    /// A TV layout did not have exactly two [modes](crate#modes): thread and
    /// local value.
    TvRank {
        /// Number of top-level modes found.
        actual: usize,
    },
    /// A TV mode did not have the expected number of entries.
    TvModeSize {
        /// Either `"thread"` or `"value"`.
        mode: &'static str,
        /// Required number of entries.
        expected: i64,
        /// Number of entries found.
        actual: i64,
    },
    /// The number of TV pairs differs from the number of tile cells.
    TvCardinality {
        /// `thread_count * values_per_thread`.
        assignments: i64,
        /// Number of cells in the tile.
        tile_elements: i64,
    },
    /// A TV pair mapped outside the tile.
    TvOutOfRange {
        /// Thread ID in the bad TV pair.
        thread: i64,
        /// Local value number in the bad TV pair.
        value: i64,
        /// Tile cell produced by the TV map.
        cell: i64,
        /// Number of valid tile cells.
        tile_elements: i64,
    },
    /// Two TV pairs claimed the same tile cell.
    DuplicateCell {
        /// Tile cell with two owners.
        cell: i64,
        /// First `(thread, local value)` owner.
        first_owner: (i64, i64),
        /// Second `(thread, local value)` owner.
        second_owner: (i64, i64),
    },
    /// No TV pair claimed one tile cell.
    MissingCell {
        /// Unowned tile cell.
        cell: i64,
    },
    /// The copy cannot be split into whole thread/value groups required by its
    /// [copy atom](crate#atoms).
    AtomDoesNotDivide {
        /// Grouped dimension: `"thread"` or `"value"`.
        axis: &'static str,
        /// Total count in the cooperative copy.
        copy_count: i64,
        /// Count consumed by one atom.
        atom_count: i64,
    },
    /// A transaction width is not a whole number of elements.
    AtomSize {
        /// Transaction width in bytes.
        atom_bytes: i64,
        /// Width of one element in bytes.
        element_bytes: i64,
    },
    /// Physical contiguity was requested from an element-offset layout.
    AtomNeedsByteLayout,
    /// The memory base does not guarantee the transaction's alignment.
    BaseAlignment {
        /// Required byte alignment.
        required: i64,
        /// Alignment promised by the caller.
        guaranteed: i64,
    },
    /// The first byte of a transaction is not atom-aligned.
    AtomStartMisaligned {
        /// Thread that owns the atom.
        thread: i64,
        /// First local value in the atom.
        value: i64,
        /// Physical start offset in bytes.
        offset: i64,
        /// Required transaction alignment in bytes.
        atom_bytes: i64,
    },
    /// Bytes assigned to one atom do not form one consecutive interval.
    AtomNotContiguous {
        /// Thread that owns the atom.
        thread: i64,
        /// First local value in the atom.
        value: i64,
        /// Byte position within the atom where continuity breaks.
        byte_in_atom: i64,
        /// Consecutive byte offset required at that position.
        expected: i64,
        /// Byte offset produced by the composed memory map.
        actual: i64,
    },
    /// Checked integer arithmetic failed while validating the named fact.
    ArithmeticOverflow(
        #[doc = "Name of the layout calculation whose checked arithmetic overflowed."] &'static str,
    ),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidCount { name, value } => {
                write!(f, "{name} must be positive, got {value}")
            }
            ValidationError::CountTooLarge { name, value } => {
                write!(f, "{name}={value} cannot be enumerated on this host")
            }
            ValidationError::ThreadOutOfRange {
                coordinate,
                thread,
                thread_count,
            } => write!(
                f,
                "thread layout coordinate {coordinate} maps to thread {thread}, outside 0..{thread_count}"
            ),
            ValidationError::DuplicateThread {
                thread,
                first_coordinate,
                second_coordinate,
            } => write!(
                f,
                "thread {thread} is produced twice, by coordinates {first_coordinate} and {second_coordinate}"
            ),
            ValidationError::TvRank { actual } => {
                write!(
                    f,
                    "TV layout must have (thread,value) modes, got rank {actual}"
                )
            }
            ValidationError::TvModeSize {
                mode,
                expected,
                actual,
            } => write!(f, "TV {mode} mode has size {actual}, expected {expected}"),
            ValidationError::TvCardinality {
                assignments,
                tile_elements,
            } => write!(
                f,
                "TV layout has {assignments} assignments for a {tile_elements}-element tile"
            ),
            ValidationError::TvOutOfRange {
                thread,
                value,
                cell,
                tile_elements,
            } => write!(
                f,
                "TV ({thread},{value}) maps to cell {cell}, outside 0..{tile_elements}"
            ),
            ValidationError::DuplicateCell {
                cell,
                first_owner,
                second_owner,
            } => write!(
                f,
                "tile cell {cell} has two owners: TV ({},{}) and TV ({},{})",
                first_owner.0, first_owner.1, second_owner.0, second_owner.1
            ),
            ValidationError::MissingCell { cell } => {
                write!(f, "tile cell {cell} has no TV owner")
            }
            ValidationError::AtomDoesNotDivide {
                axis,
                copy_count,
                atom_count,
            } => write!(
                f,
                "copy {axis} count {copy_count} is not divisible by atom {axis} count {atom_count}"
            ),
            ValidationError::AtomSize {
                atom_bytes,
                element_bytes,
            } => write!(
                f,
                "a {atom_bytes}-byte atom cannot be split into {element_bytes}-byte elements"
            ),
            ValidationError::AtomNeedsByteLayout => {
                write!(
                    f,
                    "atom contiguity must be checked with a byte-offset layout"
                )
            }
            ValidationError::BaseAlignment {
                required,
                guaranteed,
            } => write!(
                f,
                "atom needs {required}-byte base alignment, but only {guaranteed} bytes are guaranteed"
            ),
            ValidationError::AtomStartMisaligned {
                thread,
                value,
                offset,
                atom_bytes,
            } => write!(
                f,
                "TV ({thread},{value}) starts at byte {offset}, not a {atom_bytes}-byte boundary"
            ),
            ValidationError::AtomNotContiguous {
                thread,
                value,
                byte_in_atom,
                expected,
                actual,
            } => write!(
                f,
                "TV atom ({thread},{value}) breaks at byte {byte_in_atom}: expected {expected}, got {actual}"
            ),
            ValidationError::ArithmeticOverflow(context) => {
                write!(f, "integer overflow while checking {context}")
            }
        }
    }
}

impl core::error::Error for ValidationError {}
