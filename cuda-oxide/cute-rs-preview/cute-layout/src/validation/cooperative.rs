/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Validate the complete global-to-shared cooperative copy plan.

use super::thread_value::tv_coord;
use super::{
    MAX_COOPERATIVE_THREADS, MAX_STATIC_LAYOUT_OFFSET_MAGNITUDE, MAX_STATIC_VALIDATION_BYTES,
    MAX_STATIC_VALIDATION_ELEMENTS, validate_atom_compatibility, validate_compact_thread_layout,
    validate_per_thread_atom_contiguity, validate_tv_exact_coverage, validate_tv_layout,
};
use crate::algebra::make_layout_tv;
use crate::{ComposedLayout, IntTuple, Layout, OffsetUnit};
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
use core::fmt;

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
    /// The unsafe `copy_g2s` caller must provide at least this capacity.
    /// Static validation computes the requirement; it does not insert runtime
    /// storage checks. The value includes fixed offsets and holes introduced
    /// by the shared-memory mapping, not only `tile_elements`.
    pub minimum_smem_capacity: i64,
}

/// Explanation of why a complete cooperative copy plan is unsafe to lower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativePlanError(
    #[doc = "Explanation of the static copy-plan constraint that failed."] String,
);

impl fmt::Display for CooperativePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for CooperativePlanError {}

/// Wrap a formatted static-copy diagnostic in the plan error type.
fn plan_error(message: impl Into<String>) -> CooperativePlanError {
    CooperativePlanError(message.into())
}

/// Require positive bounded extents before exhaustive static map validation.
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

/// Reject source atoms crossing a logical row or violating the promised pitch alignment.
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
/// capacity. The unsafe caller guarantees enough shared storage, valid and
/// aligned pointers, the promised row spacing, and participating thread IDs.
/// The compiler computes valid source bytes from runtime matrix bounds and
/// zero-fills the remainder of each edge copy.
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
