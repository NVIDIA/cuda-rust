/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check a cooperative copy's compile-time layout facts before the compiler
//! turns it into GPU copy instructions.
//!
//! A **tile** is the small, fixed rectangle copied by one CUDA block. A
//! [**mode**](crate#modes) is one part of a coordinate that you can change while
//! keeping the other parts fixed. The first supported form of `copy_g2s`
//! requires two top-level tile modes: rows and columns.
//!
//! ```text
//! 2-row x 3-column tile
//!
//!          column 0  column 1  column 2
//! row 0       A         B         C
//! row 1       D         E         F
//! ```
//!
//! A **thread-value (TV) assignment** answers: “which tile cell does local
//! value `v` of thread `t` own?” For example:
//!
//! ```text
//! TV(0,0) -> A    TV(0,1) -> B    TV(0,2) -> C
//! TV(1,0) -> D    TV(1,1) -> E    TV(1,2) -> F
//! ```
//!
//! **Exact coverage** means every cell has exactly one TV owner. A hole means
//! an element is never copied; two owners mean two threads can write the same
//! destination. Both are rejected.
//!
//! A **compact thread layout** produces every thread ID from `0` through
//! `thread_count - 1` exactly once. The IDs may be reordered, but none may be
//! missing or repeated. A [**copy atom**](crate#atoms) describes one indivisible
//! transaction such as a 16-byte `cp.async`; all values assigned to that atom
//! must form one aligned, consecutive byte range.
//!
//! Global memory uses **row-major** order: columns are adjacent, and
//! `offset(row, column) = row * column_count + column`. Shared memory may use
//! a composed swizzle, so its complete byte map is checked separately.
//!
//! These validators exhaustively inspect small compile-time maps and return the
//! minimum shared-memory capacity. When the compiler emits the real copy
//! instructions, it must still check facts known only while the kernel runs:
//! valid pointers, matrix bounds, pitch (the element distance between row
//! starts), edge handling, the actual shared allocation, and the current
//! thread index.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algebra::make_layout_tv;
use crate::{ComposedLayout, IntTuple, Layout, OffsetUnit};

/// Result returned by the reusable low-level layout checks in this module.
pub type ValidationResult<T> = core::result::Result<T, ValidationError>;

/// Largest element count that an exhaustive static proof may enumerate.
///
/// One million elements is well above a practical shared-memory tile. The
/// limit prevents a malformed compiler attribute from causing an OOM or hang.
pub const MAX_STATIC_VALIDATION_ELEMENTS: i64 = 1 << 20;

/// Largest byte map that the exact shared-memory proof may enumerate.
pub const MAX_STATIC_VALIDATION_BYTES: i64 = 1 << 20;

/// CUDA's maximum number of threads in one cooperative block.
pub const MAX_COOPERATIVE_THREADS: i64 = 1024;

/// Largest absolute static stride or offset accepted before layout algebra.
///
/// This bound also catches huge strides hidden under a size-one
/// [mode](crate#modes).
pub const MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE: u64 = 1 << 20;

/// Checked, reusable compile-time map for one block-wide global-to-shared copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativeCopyPlan {
    /// Tile shape with exactly two [modes](crate#modes):
    /// `[row_count, column_count]`.
    pub tiler_shape: Vec<i64>,
    /// Map from `(thread ID, local value number)` to a logical tile cell.
    pub tv_layout: Layout,
    /// Canonical row-major global-memory tile map, measured in bytes.
    pub gmem_byte_layout: ComposedLayout,
    /// Complete ordinary-layout-plus-swizzle shared-memory map, in bytes.
    pub smem_byte_layout: ComposedLayout,
    /// Number of participating block threads: valid IDs are `0..thread_count`.
    pub thread_count: i64,
    /// Number of logical tile elements owned by each thread.
    pub values_per_thread: i64,
    /// Total cells in the tile: rows multiplied by columns.
    pub tile_elements: i64,
    /// Minimum `SmemTile<T>::capacity`, measured in `T` elements.
    ///
    /// Before issuing the copy, the compiler must prove that the runtime
    /// capacity is at least this value. It includes fixed offsets and holes
    /// introduced by the shared-memory mapping, not only `tile_elements`.
    pub minimum_smem_capacity: i64,
}

/// Explanation of why a complete cooperative copy plan is unsafe to lower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativePlanError(String);

impl fmt::Display for CooperativePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for CooperativePlanError {}

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
    ArithmeticOverflow(&'static str),
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

fn plan_error(message: impl Into<String>) -> CooperativePlanError {
    CooperativePlanError(message.into())
}

fn checked_plan_layout_size(layout: &Layout, role: &str) -> Result<i64, CooperativePlanError> {
    let size = layout
        .checked_size()
        .ok_or_else(|| plan_error(format!("{role} layout size is invalid or overflows i64")))?;
    if size > MAX_STATIC_VALIDATION_ELEMENTS {
        return Err(plan_error(format!(
            "{role} layout has {size} elements; static validation limit is \
             {MAX_STATIC_VALIDATION_ELEMENTS}"
        )));
    }
    Ok(size)
}

/// Reject unsafe raw strides before CuTe algebra can multiply them.
///
/// A mode with extent one always evaluates coordinate zero, so ordinary map
/// evaluation cannot reveal its stride:
///
/// ```text
/// shape:   1
/// coord:   0
/// offset:  0 * huge_stride = 0   <- huge stride stays hidden
/// ```
///
/// Reading every raw stride first prevents later algebra from overflowing on
/// such a hidden value.
fn validate_plan_layout_arithmetic(
    layout: &Layout,
    role: &str,
) -> Result<(), CooperativePlanError> {
    for stride in layout.stride.leaves() {
        if stride.unsigned_abs() > MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE {
            return Err(plan_error(format!(
                "{role} layout stride {stride} exceeds static magnitude limit \
                 {MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE}"
            )));
        }
    }

    let size = checked_plan_layout_size(layout, role)?;
    for coordinate in 0..size {
        let offset = layout
            .checked_call(&IntTuple::Leaf(coordinate))
            .ok_or_else(|| {
                plan_error(format!(
                    "{role} layout coordinate {coordinate} overflows i64"
                ))
            })?;
        if offset.unsigned_abs() > MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE {
            return Err(plan_error(format!(
                "{role} layout coordinate {coordinate} maps to {offset}; static magnitude limit \
                 is {MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE}"
            )));
        }
    }
    Ok(())
}

/// Require the exact dense row-major map used by `GmemMatrix`.
///
/// CuTe flattens a two-mode logical coordinate with the row mode changing
/// first. Row-major memory instead makes the column physically adjacent:
///
/// ```text
/// 2 rows x 3 columns
/// flat cell:  0     1   |  2     3   |  4     5
/// (row,col): (0,0) (1,0) | (0,1) (1,1) | (0,2) (1,2)
/// memory off:   0     3   |  1     4   |  2     5
///
/// offset = row * column_count + column
/// ```
fn validate_canonical_row_major_tile_layout(
    tile_layout: &Layout,
    rows: i64,
    columns: i64,
) -> Result<(), CooperativePlanError> {
    let size = checked_plan_layout_size(tile_layout, "tile")?;
    for cell in 0..size {
        let row = cell % rows;
        let column = cell / rows;
        let expected = row
            .checked_mul(columns)
            .and_then(|offset| offset.checked_add(column))
            .ok_or_else(|| plan_error("canonical row-major tile offset overflows i64"))?;
        let actual = tile_layout
            .checked_call(&IntTuple::Leaf(cell))
            .ok_or_else(|| plan_error(format!("tile coordinate {cell} overflows i64")))?;
        if actual != expected {
            return Err(plan_error(format!(
                "tile layout is not canonical row-major: logical cell {cell} must map to \
                 {expected}, got {actual}"
            )));
        }
    }
    Ok(())
}

fn validate_source_atom_rows(
    tv: &Layout,
    thread_count: i64,
    values_per_thread: i64,
    values_per_atom: i64,
    tile_rows: i64,
) -> Result<(), CooperativePlanError> {
    let step = usize::try_from(values_per_atom)
        .map_err(|_| plan_error("copy atom value count does not fit this host"))?;
    for thread in 0..thread_count {
        for first_value in (0..values_per_thread).step_by(step) {
            let mut atom_row = None;
            let atom_end = first_value
                .checked_add(values_per_atom)
                .ok_or_else(|| plan_error("copy atom value range overflows i64"))?;
            for value in first_value..atom_end {
                let cell = tv
                    .checked_call(&tv_coord(thread, value))
                    .ok_or_else(|| plan_error(format!("TV ({thread},{value}) overflows i64")))?;
                // Decode the flat CuTe cell before checking the atom's row:
                //
                // cell = row + column * tile_rows
                // row  = cell % tile_rows
                //
                // Example for two rows: cells 0,2,4 are row 0; cells 1,3,5
                // are row 1. TileL applies the physical row-major stride later.
                let row = cell.rem_euclid(tile_rows);
                if let Some(first_row) = atom_row
                    && first_row != row
                {
                    return Err(plan_error(format!(
                        "thread {thread}'s atom at value {first_value} crosses from row \
                         {first_row} to row {row}"
                    )));
                }
                atom_row = Some(row);
            }
        }
    }
    Ok(())
}

/// Prove that every shared byte has at most one owner and compute capacity.
///
/// `minimum_smem_capacity` is one past the highest referenced byte, rounded
/// up to whole elements:
///
/// ```text
/// referenced shared bytes:  8 .. 31
/// required byte range:      0 .. 32
/// f32 capacity:             ceil(32 / 4) = 8 elements
/// ```
fn validate_physical_smem_map(
    tv: &Layout,
    thread_count: i64,
    values_per_thread: i64,
    element_bytes: i64,
    smem_bytes: &ComposedLayout,
) -> Result<i64, CooperativePlanError> {
    let mut owners = BTreeMap::new();
    let mut last_byte = -1i64;
    for thread in 0..thread_count {
        for value in 0..values_per_thread {
            let cell = tv
                .checked_call(&tv_coord(thread, value))
                .ok_or_else(|| plan_error(format!("TV ({thread},{value}) overflows i64")))?;
            for byte_in_element in 0..element_bytes {
                let offset = smem_bytes
                    .checked_call_with_inner_delta(&IntTuple::Leaf(cell), byte_in_element)
                    .ok_or_else(|| {
                        plan_error(format!(
                            "TV ({thread},{value}) byte {byte_in_element} overflows its smem offset"
                        ))
                    })?;
                if offset < 0 {
                    return Err(plan_error(format!(
                        "TV ({thread},{value}) maps byte {byte_in_element} to negative smem offset \
                         {offset}"
                    )));
                }
                if offset >= MAX_STATIC_VALIDATION_BYTES {
                    return Err(plan_error(format!(
                        "smem byte offset {offset} exceeds static physical limit \
                         {MAX_STATIC_VALIDATION_BYTES}"
                    )));
                }
                if let Some((first_thread, first_value, first_byte)) =
                    owners.insert(offset, (thread, value, byte_in_element))
                {
                    return Err(plan_error(format!(
                        "smem byte {offset} has two owners: TV ({first_thread},{first_value}) byte \
                         {first_byte} and TV ({thread},{value}) byte {byte_in_element}"
                    )));
                }
                last_byte = last_byte.max(offset);
            }
        }
    }
    let required_bytes = last_byte
        .checked_add(1)
        .ok_or_else(|| plan_error("required smem byte range overflows i64"))?;
    required_bytes
        .checked_add(element_bytes - 1)
        .map(|rounded| rounded / element_bytes)
        .ok_or_else(|| plan_error("required smem capacity overflows i64"))
}

/// Validate all compile-time facts for one block-wide global-to-shared copy.
///
/// Inputs describe one fixed tile:
///
/// - `atom_bytes`: transaction width described by the [copy atom](crate#atoms),
///   currently 4, 8, or 16 bytes;
/// - `thread_layout`: tile coordinates to compact block thread IDs;
/// - `value_layout`: per-thread local value arrangement;
/// - `tile_layout`: exactly two row/column [modes](crate#modes) in canonical
///   row-major order;
/// - `smem_layout`: destination element map, including any composed swizzle;
/// - `element_bytes`: byte width of one copied value.
///
/// The validator constructs the TV assignment and proves:
///
/// ```text
/// every (thread, local value) -> one in-range tile cell
/// every tile cell             -> one owner
/// transaction described by each copy atom -> one aligned consecutive byte interval
/// every shared byte           -> at most one writer
/// ```
///
/// Here is a complete four-cell example with two threads and two `f32` values
/// per thread:
///
/// ```text
/// thread_layout = (2,1):(1,0)   -> thread IDs 0 and 1 down the rows
/// value_layout  = (1,2):(0,1)   -> local values 0 and 1 across the columns
/// tile_layout   = (2,2):(2,1)   -> row-major element offsets 0 through 3
/// smem_layout   = identity composition of tile_layout
/// atom_bytes    = 8             -> each thread copies two adjacent f32 values
/// element_bytes = 4             -> each value is four bytes
///
/// TV map produced by make_layout_tv = (2,2):(1,2)
///
///                 value 0       value 1
/// thread 0         cell 0/off 0   cell 2/off 1   bytes 0 through 7
/// thread 1         cell 1/off 2   cell 3/off 3   bytes 8 through 15
/// ```
///
/// `cell` is the flat tile-cell number; `off` is its row-major element offset.
/// Every cell has one owner, and each thread's two physical offsets are
/// adjacent, forming one aligned 8-byte transaction.
///
/// The returned plan contains byte-addressed maps and the minimum shared
/// capacity. Before using it, the compiler must still check facts known only
/// while the kernel runs: enough shared storage, matrix bounds and row spacing,
/// valid and aligned pointers, the current thread index, and how many bytes an
/// edge copy should read before zero-filling the rest.
pub fn validate_cooperative_copy_plan(
    atom_bytes: u32,
    thread_layout: &Layout,
    value_layout: &Layout,
    tile_layout: &Layout,
    smem_layout: &ComposedLayout,
    element_bytes: i64,
) -> Result<CooperativeCopyPlan, CooperativePlanError> {
    if !matches!(atom_bytes, 4 | 8 | 16) {
        return Err(plan_error(format!(
            "copy atom must be 4, 8, or 16 bytes, got {atom_bytes}"
        )));
    }
    if element_bytes <= 0 || !(element_bytes as u64).is_power_of_two() {
        return Err(plan_error(format!(
            "element size must be a positive power of two, got {element_bytes} bytes"
        )));
    }
    if i64::from(atom_bytes) % element_bytes != 0 {
        return Err(plan_error(
            "copy atom does not contain a whole number of elements",
        ));
    }

    let thread_count = checked_plan_layout_size(thread_layout, "thread")?;
    let values_per_thread = checked_plan_layout_size(value_layout, "value")?;
    let tile_elements = checked_plan_layout_size(tile_layout, "tile")?;
    let smem_elements = checked_plan_layout_size(smem_layout.inner(), "shared-memory inner")?;
    let assignments = thread_count
        .checked_mul(values_per_thread)
        .ok_or_else(|| plan_error("TV assignment count overflows i64"))?;
    if assignments != tile_elements {
        return Err(plan_error(format!(
            "TV layout has {assignments} assignments for a {tile_elements}-element tile"
        )));
    }
    let tile_bytes = tile_elements
        .checked_mul(element_bytes)
        .ok_or_else(|| plan_error("copy tile byte count overflows i64"))?;
    if tile_bytes > MAX_STATIC_VALIDATION_BYTES {
        return Err(plan_error(format!(
            "copy tile has {tile_bytes} bytes; static byte-map limit is \
             {MAX_STATIC_VALIDATION_BYTES}"
        )));
    }
    if smem_layout.offset().unsigned_abs() > MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE {
        return Err(plan_error(format!(
            "shared-memory composed offset {} exceeds static magnitude limit \
             {MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE}",
            smem_layout.offset()
        )));
    }
    for (name, layout) in [
        ("thread", thread_layout),
        ("value", value_layout),
        ("tile", tile_layout),
        ("shared-memory inner", smem_layout.inner()),
    ] {
        validate_plan_layout_arithmetic(layout, name)?;
    }

    validate_compact_thread_layout(thread_layout)
        .map_err(|error| plan_error(format!("invalid thread layout: {error}")))?;
    if thread_count > MAX_COOPERATIVE_THREADS {
        return Err(plan_error(format!(
            "thread count {thread_count} exceeds CUDA block limit {MAX_COOPERATIVE_THREADS}"
        )));
    }

    let (tiler_shape, tv_layout) = make_layout_tv(thread_layout, value_layout)
        .map_err(|error| plan_error(format!("invalid TV algebra: {error}")))?;
    validate_tv_layout(thread_layout, value_layout, &tiler_shape, &tv_layout)
        .map_err(|error| plan_error(format!("invalid TV layout: {error}")))?;

    let tile_modes = tile_layout.modes();
    let tile_shape = tile_modes
        .iter()
        .map(|mode| {
            mode.checked_size()
                .ok_or_else(|| plan_error("tile mode size is invalid or overflows i64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tiler_shape.len() != 2 || tile_shape.len() != 2 {
        return Err(plan_error(
            "copy_g2s v0 requires exactly two row/column tile modes",
        ));
    }
    if tile_shape != tiler_shape {
        return Err(plan_error(format!(
            "tile shape {tile_shape:?} does not match TV tiler {tiler_shape:?}"
        )));
    }

    validate_tv_exact_coverage(&tv_layout, thread_count, values_per_thread, tile_elements)
        .map_err(|error| plan_error(format!("invalid tile coverage: {error}")))?;
    validate_canonical_row_major_tile_layout(tile_layout, tile_shape[0], tile_shape[1])
        .map_err(|error| plan_error(format!("invalid tile layout: {error}")))?;
    if smem_elements != tile_elements {
        return Err(plan_error(format!(
            "shared-memory layout covers {smem_elements} elements but tile layout covers \
             {tile_elements}"
        )));
    }

    let values_per_atom = i64::from(atom_bytes) / element_bytes;
    validate_atom_compatibility(thread_count, values_per_thread, 1, values_per_atom)
        .map_err(|error| plan_error(format!("invalid copy atom: {error}")))?;
    validate_source_atom_rows(
        &tv_layout,
        thread_count,
        values_per_thread,
        values_per_atom,
        tile_shape[0],
    )
    .map_err(|error| plan_error(format!("global copy atom crosses a tile row: {error}")))?;

    let gmem_byte_layout = ComposedLayout::from_layout(tile_layout.clone(), OffsetUnit::Elements)
        .to_byte_offsets(element_bytes)
        .map_err(|error| plan_error(format!("invalid gmem byte layout: {error}")))?;
    validate_per_thread_atom_contiguity(
        &tv_layout,
        thread_count,
        values_per_thread,
        i64::from(atom_bytes),
        element_bytes,
        &gmem_byte_layout,
        i64::from(atom_bytes),
    )
    .map_err(|error| {
        plan_error(format!(
            "global copy atom is not physically contiguous: {error}"
        ))
    })?;

    let smem_byte_layout = smem_layout
        .to_byte_offsets(element_bytes)
        .map_err(|error| plan_error(format!("invalid smem byte layout: {error}")))?;
    validate_per_thread_atom_contiguity(
        &tv_layout,
        thread_count,
        values_per_thread,
        i64::from(atom_bytes),
        element_bytes,
        &smem_byte_layout,
        i64::from(atom_bytes),
    )
    .map_err(|error| {
        plan_error(format!(
            "shared-memory copy atom is not physically contiguous: {error}"
        ))
    })?;
    let minimum_smem_capacity = validate_physical_smem_map(
        &tv_layout,
        thread_count,
        values_per_thread,
        element_bytes,
        &smem_byte_layout,
    )
    .map_err(|error| plan_error(format!("invalid physical shared-memory map: {error}")))?;

    Ok(CooperativeCopyPlan {
        tiler_shape,
        tv_layout,
        gmem_byte_layout,
        smem_byte_layout,
        thread_count,
        values_per_thread,
        tile_elements,
        minimum_smem_capacity,
    })
}

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

/// Check a TV layout produced by [`make_layout_tv`] against its source
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
/// `CpAsync<N>` uses one thread per atom, but this helper also supports later
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
/// This function is for `CpAsync<N>`, where one thread owns each atom.
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
        .inner()
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
                    .checked_call_with_inner_delta(&IntTuple::Leaf(cell), byte_in_element)
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

fn tv_coord(thread: i64, value: i64) -> IntTuple {
    IntTuple::Tuple(vec![IntTuple::Leaf(thread), IntTuple::Leaf(value)])
}

fn positive_count(name: &'static str, value: i64) -> ValidationResult<i64> {
    if value <= 0 {
        Err(ValidationError::InvalidCount { name, value })
    } else {
        Ok(value)
    }
}

fn host_count(name: &'static str, value: i64) -> ValidationResult<usize> {
    if value > MAX_STATIC_VALIDATION_ELEMENTS {
        return Err(ValidationError::CountTooLarge { name, value });
    }
    usize::try_from(value).map_err(|_| ValidationError::CountTooLarge { name, value })
}

/// Prove a shared tile can feed `ldmatrix` for `f16` operands.
///
/// `ldmatrix` reads one 8x8 `f16` matrix as eight row segments of 16 bytes,
/// each supplied by one lane's address. The hardware requires every segment
/// to be 16 consecutive, 16-byte-aligned shared bytes. This check enumerates
/// every 8-column-aligned segment of the whole tile through the composed
/// byte map (swizzle included), so any warp-tile window the compiler later
/// addresses is covered:
///
/// ```text
/// for every row r, every column block c8 (multiples of 8):
///     bytes of elements (r, c8..c8+8)  must be  base, base+2, ..., base+14
///     and base % 16 == 0
/// ```
///
/// Layouts that scatter a row (column-major smem, byte-hostile swizzles)
/// fail here at compile time instead of producing garbage fragments.
pub fn validate_ldmatrix_source(
    smem_byte_layout: &ComposedLayout,
    rows: i64,
    columns: i64,
) -> ValidationResult<()> {
    let rows = positive_count("ldmatrix rows", rows)?;
    let columns = positive_count("ldmatrix columns", columns)?;
    if rows % 8 != 0 || columns % 8 != 0 {
        return Err(ValidationError::InvalidCount {
            name: "ldmatrix tile extents must be multiples of 8",
            value: rows * columns,
        });
    }
    let total = rows
        .checked_mul(columns)
        .ok_or(ValidationError::CountTooLarge {
            name: "ldmatrix tile elements",
            value: i64::MAX,
        })?;
    host_count("ldmatrix tile elements", total)?;

    for row in 0..rows {
        for block in 0..(columns / 8) {
            let first_cell = row + rows * (block * 8);
            let base = smem_byte_layout
                .checked_call(&IntTuple::Leaf(first_cell))
                .ok_or(ValidationError::InvalidCount {
                    name: "ldmatrix segment base is unmappable",
                    value: first_cell,
                })?;
            if base % 16 != 0 {
                return Err(ValidationError::AtomStartMisaligned {
                    thread: row,
                    value: block,
                    offset: base,
                    atom_bytes: 16,
                });
            }
            for element in 1..8 {
                let cell = row + rows * (block * 8 + element);
                let byte = smem_byte_layout.checked_call(&IntTuple::Leaf(cell)).ok_or(
                    ValidationError::InvalidCount {
                        name: "ldmatrix segment element is unmappable",
                        value: cell,
                    },
                )?;
                if byte != base + element * 2 {
                    return Err(ValidationError::AtomNotContiguous {
                        thread: row,
                        value: block,
                        byte_in_atom: element * 2,
                        expected: base + element * 2,
                        actual: byte,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Prove a shared tile is one of the four arrangements TMA can encode.
///
/// TMA writes row-major boxes scrambled by at most one of the fixed
/// byte-unit patterns `S<B,4,3>` (B in 0..=3), with no composed offset.
/// Swizzled boxes must span the whole row. This is the compiler-side twin
/// of the host encoder's checks, so an unencodable `SmemL` fails at KERNEL
/// compile time instead of at host run time.
pub fn validate_tma_encodable(
    smem_layout: &ComposedLayout,
    element_bytes: i64,
) -> ValidationResult<()> {
    positive_count("TMA element bytes", element_bytes)?;
    if element_bytes & (element_bytes - 1) != 0 {
        return Err(ValidationError::InvalidCount {
            name: "TMA element byte width must be a power of two",
            value: element_bytes,
        });
    }
    if smem_layout.unit() != OffsetUnit::Elements {
        return Err(ValidationError::InvalidCount {
            name: "TMA shared layout offsets must count elements",
            value: 0,
        });
    }
    let modes = smem_layout.inner().modes();
    if modes.len() != 2 {
        return Err(ValidationError::TvRank {
            actual: modes.len(),
        });
    }
    let rows = modes[0]
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("TMA tile rows"))?;
    let columns = modes[1]
        .checked_size()
        .ok_or(ValidationError::ArithmeticOverflow("TMA tile columns"))?;
    positive_count("TMA tile rows", rows)?;
    positive_count("TMA tile columns", columns)?;
    let row_leaves = modes[0].stride.leaves();
    let col_leaves = modes[1].stride.leaves();
    if row_leaves != [columns] || col_leaves != [1] {
        return Err(ValidationError::InvalidCount {
            name: "TMA tile must be row-major (row stride = columns, column stride = 1)",
            value: rows * columns,
        });
    }
    if smem_layout.offset() != 0 {
        return Err(ValidationError::InvalidCount {
            name: "TMA cannot express a composed offset",
            value: smem_layout.offset(),
        });
    }
    let swizzle = smem_layout.outer();
    if swizzle.bits == 0 {
        return Ok(());
    }
    let m_bytes = i64::from(swizzle.base) + element_bytes.trailing_zeros() as i64;
    if swizzle.bits > 3 || m_bytes != 4 || swizzle.shift != 3 {
        return Err(ValidationError::InvalidCount {
            name: "swizzle is not one of TMA's encodable byte-unit S<B,4,3> modes",
            value: i64::from(swizzle.bits),
        });
    }
    let pitch_bytes = columns * element_bytes;
    if pitch_bytes != (16 << swizzle.bits) {
        return Err(ValidationError::InvalidCount {
            name: "swizzled TMA tile pitch must equal the swizzle span",
            value: pitch_bytes,
        });
    }
    Ok(())
}

/// Return the shared-base alignment required to keep one TMA swizzle phase.
///
/// TMA always needs at least a 16-byte shared address. A swizzle adds higher
/// address bits that choose the phase of its XOR mapping:
///
/// ```text
/// byte-unit S<B,4,3>
///
/// low 4 bits       target B bits             source B bits
/// [ unchanged ]    [ XOR target ]   +3 start [ XOR source ]
/// 0 ........ 3     4 .... 4+B-1              7 .... 7+B-1
///
/// highest participating bit + 1 = 4 + B + 3
/// required alignment             = 2^(4 + B + 3) bytes
/// ```
///
/// That gives the layouts used by the GEMM transport path:
///
/// ```text
/// no swizzle       16 bytes
/// B64,  B = 2     512 bytes
/// B128, B = 3    1024 bytes
/// ```
///
/// This is the minimum alignment promised by the unsafe TMA copy. It is not
/// a claim about the descriptor pointer or the hidden global-memory address.
/// The layout is checked with [`validate_tma_encodable`] first, so callers
/// cannot obtain an alignment for a layout the hardware cannot represent.
pub fn tma_phase_alignment_bytes(
    smem_layout: &ComposedLayout,
    element_bytes: i64,
) -> ValidationResult<i64> {
    validate_tma_encodable(smem_layout, element_bytes)?;
    let swizzle = smem_layout.outer();
    if swizzle.bits == 0 {
        return Ok(16);
    }

    // Validation above normalized the element-based swizzle to byte-unit
    // S<B,4,3>. The highest participating source bit is therefore 4+B+3-1.
    let phase_bits = 4_u32
        .checked_add(swizzle.bits)
        .and_then(|bits| bits.checked_add(3))
        .ok_or(ValidationError::ArithmeticOverflow(
            "TMA swizzle phase alignment exponent",
        ))?;
    1_i64
        .checked_shl(phase_bits)
        .ok_or(ValidationError::ArithmeticOverflow(
            "TMA swizzle phase alignment",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Swizzle;
    use alloc::string::ToString;

    fn l(text: &str) -> Layout {
        text.parse().unwrap()
    }

    fn canary_plan() -> Result<CooperativeCopyPlan, CooperativePlanError> {
        let thread = l("(6,1):(1,0)");
        let value = l("(1,4):(0,1)");
        let tile = l("(6,4):(4,1)");
        let smem =
            ComposedLayout::new(Swizzle::new(3, 4, 3), 8, tile.clone(), OffsetUnit::Elements)
                .unwrap();
        validate_cooperative_copy_plan(16, &thread, &value, &tile, &smem, 4)
    }

    #[test]
    fn high_level_plan_matches_the_live_decode_canary() {
        let plan = canary_plan().unwrap();
        assert_eq!(plan.tiler_shape, vec![6, 4]);
        assert_eq!(plan.thread_count, 6);
        assert_eq!(plan.values_per_thread, 4);
        assert_eq!(plan.tile_elements, 24);
        assert_eq!(plan.minimum_smem_capacity, 32);
    }

    #[test]
    fn high_level_plan_rejects_a_digit_permuted_row_major_tile() {
        let thread = l("(1,1):(1,0)");
        // The logical columns are 0,1,2,3, but this nested digit mode visits
        // them as 0,2,1,3. Giving TV and TileL the same permutation makes the
        // two mistakes cancel and appear contiguous:
        //
        // TV order:       0 2 1 3
        // TileL maps to:  0 1 2 3
        //
        // A real row-major GmemMatrix still uses logical order 0,1,2,3, so
        // the high-level validator must reject the permuted TileL itself.
        let digit_permutation = l("(1,(2,2)):(0,(2,1))");
        let (_, tv) = make_layout_tv(&thread, &digit_permutation).unwrap();
        let apparent_offsets: Vec<_> = (0..4)
            .map(|value| {
                let cell = tv.checked_call(&tv_coord(0, value)).unwrap();
                digit_permutation
                    .checked_call(&IntTuple::Leaf(cell))
                    .unwrap()
            })
            .collect();
        assert_eq!(apparent_offsets, vec![0, 1, 2, 3]);

        let smem = ComposedLayout::from_layout(digit_permutation.clone(), OffsetUnit::Elements);
        let error = validate_cooperative_copy_plan(
            16,
            &thread,
            &digit_permutation,
            &digit_permutation,
            &smem,
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not canonical row-major"), "{error}");
    }

    #[test]
    fn high_level_plan_bounds_threads_and_hidden_raw_strides() {
        let thread = l("(1025,1):(1,0)");
        let value = l("(1,1):(0,0)");
        let tile = l("(1025,1):(1,0)");
        let smem = ComposedLayout::from_layout(tile.clone(), OffsetUnit::Elements);
        let error = validate_cooperative_copy_plan(4, &thread, &value, &tile, &smem, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("CUDA block limit 1024"), "{error}");

        // A size-one mode has only coordinate zero, which hides its stride:
        //
        // 0 * i64::MAX = 0
        //
        // Raw-stride validation sees i64::MAX before algebra can multiply it.
        let hidden_stride = l("(6,1):(1,9223372036854775807)");
        let tile = l("(6,4):(4,1)");
        let smem = ComposedLayout::from_layout(tile.clone(), OffsetUnit::Elements);
        let error =
            validate_cooperative_copy_plan(16, &hidden_stride, &l("(1,4):(0,1)"), &tile, &smem, 4)
                .unwrap_err()
                .to_string();
        assert!(error.contains("layout stride"), "{error}");
    }

    #[test]
    fn compact_threads_allow_permutation_but_reject_gaps() {
        validate_compact_thread_layout(&l("(2,2):(2,1)")).unwrap();
        validate_compact_thread_layout(&l("(2,2):(1,2)")).unwrap();

        assert!(matches!(
            validate_compact_thread_layout(&l("4:2")),
            Err(ValidationError::ThreadOutOfRange { .. })
        ));
        assert!(matches!(
            validate_compact_thread_layout(&l("(2,2):(0,1)")),
            Err(ValidationError::DuplicateThread { .. })
        ));

        let too_large = Layout::contiguous(MAX_STATIC_VALIDATION_ELEMENTS + 1);
        assert!(matches!(
            validate_compact_thread_layout(&too_large),
            Err(ValidationError::CountTooLarge { .. })
        ));
    }

    #[test]
    fn tv_layout_must_be_an_exact_partition() {
        let valid = l("(4,2):(2,1)");
        validate_tv_exact_coverage(&valid, 4, 2, 8).unwrap();

        let duplicate = l("(4,2):(1,1)");
        assert!(matches!(
            validate_tv_exact_coverage(&duplicate, 4, 2, 8),
            Err(ValidationError::DuplicateCell { .. })
        ));

        let out_of_range = l("(4,2):(3,1)");
        assert!(matches!(
            validate_tv_exact_coverage(&out_of_range, 4, 2, 8),
            Err(ValidationError::TvOutOfRange { .. })
        ));
    }

    #[test]
    fn make_layout_tv_result_passes_all_static_checks() {
        let threads = l("(2,3):(3,1)");
        let values = l("(2,2):(2,1)");
        let (tiler, tv) = crate::algebra::make_layout_tv(&threads, &values).unwrap();
        validate_tv_layout(&threads, &values, &tiler, &tv).unwrap();
    }

    #[test]
    fn atom_counts_must_divide() {
        validate_atom_compatibility(256, 4, 8, 2).unwrap();
        assert_eq!(
            validate_atom_compatibility(250, 4, 8, 2),
            Err(ValidationError::AtomDoesNotDivide {
                axis: "thread",
                copy_count: 250,
                atom_count: 8,
            })
        );
        assert!(matches!(
            validate_atom_compatibility(256, 3, 8, 2),
            Err(ValidationError::AtomDoesNotDivide { axis: "value", .. })
        ));
    }

    #[test]
    fn contiguous_aligned_atoms_pass() {
        // Four threads each own four f32 values: one 16-byte atom.
        //
        // cell:  0  1  2  3 | 4  5  6  7 | ...
        // owner: T0 T0 T0 T0 |T1 T1 T1 T1 | ...
        let tv = l("(4,4):(4,1)");
        let elements =
            ComposedLayout::new(Swizzle::new(2, 2, 3), 0, l("16:1"), OffsetUnit::Elements).unwrap();
        let bytes = elements.to_byte_offsets(4).unwrap();
        validate_per_thread_atom_contiguity(&tv, 4, 4, 16, 4, &bytes, 16).unwrap();
    }

    #[test]
    fn m_four_swizzle_moves_whole_sixteen_byte_atoms() {
        // The map spans enough bytes to reach the high bits that feed the
        // XOR. A smaller map would leave every source bit zero and make the
        // swizzle look like the identity by accident.
        let tv = l("(32,4):(4,1)");
        let memory =
            ComposedLayout::new(Swizzle::new(3, 4, 3), 0, l("128:4"), OffsetUnit::Bytes).unwrap();
        validate_per_thread_atom_contiguity(&tv, 32, 4, 16, 4, &memory, 16).unwrap();
    }

    #[test]
    fn scattered_values_are_not_one_atom() {
        // Coverage is complete, but each thread is raked through the tile:
        //
        // cell:  0  1  2  3 | 4  5  6  7 | ...
        // owner: T0 T1 T2 T3 |T0 T1 T2 T3 | ...
        let tv = l("(4,4):(1,4)");
        let memory = ComposedLayout::from_layout(l("16:4"), OffsetUnit::Bytes);
        assert!(matches!(
            validate_per_thread_atom_contiguity(&tv, 4, 4, 16, 4, &memory, 16),
            Err(ValidationError::AtomNotContiguous { .. })
        ));
    }

    #[test]
    fn swizzle_must_preserve_the_whole_atom() {
        let tv = l("(2,4):(4,1)");
        let memory = ComposedLayout::new(
            // M=3 leaves the lowest three address bits unchanged, which
            // guarantees only 2^3 = 8 adjacent bytes. Here the XOR splits a
            // 16-byte transaction, and the exact byte proof catches it.
            Swizzle::new(1, 3, 3),
            64,
            l("8:4"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert!(matches!(
            validate_per_thread_atom_contiguity(&tv, 2, 4, 16, 4, &memory, 16),
            Err(ValidationError::AtomStartMisaligned { .. })
                | Err(ValidationError::AtomNotContiguous { .. })
        ));
    }

    #[test]
    fn expected_atom_address_overflow_is_a_validation_error() {
        // The swizzle maps MAX-1 to MAX, while the next input remains valid:
        //
        // input:   MAX-1  MAX
        // output:  MAX    MAX-1
        let tv = l("(1,7):(7,1)");
        let memory = ComposedLayout::new(
            Swizzle::new(1, 0, 1),
            i64::MAX - 1,
            l("7:1"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert_eq!(
            validate_per_thread_atom_contiguity(&tv, 1, 7, 7, 1, &memory, 7),
            Err(ValidationError::ArithmeticOverflow(
                "expected atom byte address"
            ))
        );
    }

    #[test]
    fn tma_identity_u16_keeps_the_hardware_minimum_alignment() {
        let layout = ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            l("(1,256):(256,1)"),
            OffsetUnit::Elements,
        )
        .unwrap();

        assert_eq!(tma_phase_alignment_bytes(&layout, 2), Ok(16));
    }

    #[test]
    fn tma_b64_u8_needs_one_complete_512_byte_phase() {
        let layout = ComposedLayout::new(
            Swizzle::new(2, 4, 3),
            0,
            l("(128,64):(64,1)"),
            OffsetUnit::Elements,
        )
        .unwrap();

        assert_eq!(tma_phase_alignment_bytes(&layout, 1), Ok(512));
    }

    #[test]
    fn tma_b128_f16_needs_one_complete_1024_byte_phase() {
        let layout = ComposedLayout::new(
            Swizzle::new(3, 3, 3),
            0,
            l("(128,64):(64,1)"),
            OffsetUnit::Elements,
        )
        .unwrap();

        assert_eq!(tma_phase_alignment_bytes(&layout, 2), Ok(1024));
    }

    #[test]
    fn tma_phase_alignment_rejects_unencodable_inputs() {
        let column_major =
            ComposedLayout::new(Swizzle::IDENTITY, 0, l("(2,4):(1,2)"), OffsetUnit::Elements)
                .unwrap();
        assert!(tma_phase_alignment_bytes(&column_major, 2).is_err());

        let b64_u8 = ComposedLayout::new(
            Swizzle::new(2, 4, 3),
            0,
            l("(128,64):(64,1)"),
            OffsetUnit::Elements,
        )
        .unwrap();
        assert!(
            tma_phase_alignment_bytes(&b64_u8, 2).is_err(),
            "the element width changes the byte-unit swizzle and row pitch"
        );
        assert!(tma_phase_alignment_bytes(&b64_u8, 0).is_err());

        let byte_offsets = ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            l("(1,256):(256,1)"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert!(tma_phase_alignment_bytes(&byte_offsets, 2).is_err());
    }
}
