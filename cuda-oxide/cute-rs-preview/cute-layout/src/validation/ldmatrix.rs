/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check shared-memory matrix-load layout restrictions.

use super::thread_value::{host_count, positive_count};
use super::{ValidationError, ValidationResult};
use crate::{ComposedLayout, IntTuple};

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
