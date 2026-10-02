/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Plan a safe global-memory read when a copy partly crosses a matrix edge.
//!
//! A [**copy atom**](crate#atoms) describes one indivisible, fixed-width
//! `cp.async` transaction, such as a 16-byte copy. A **predicate** is just a
//! boolean, such as `col < cols`. Branching uses that boolean to pick a
//! path; **predication** runs the same instruction on every thread and uses
//! the boolean to mask its effect:
//!
//! ```text
//! BRANCHING                              PREDICATION
//!
//!   p = (col < cols)?                      p = (col < cols)?
//!        │                                 copy 16 bytes if p, else 0 bytes
//!    ┌───┴────┐                                 │
//!   yes       no                           one path, all threads together
//!    │         │
//!  copy      skip
//! ```
//!
//! A cooperative copy must use the right-hand form, and its predicate
//! generalizes from a boolean to the instruction's `source_bytes` operand:
//!
//! ```text
//! 16-byte atom of four f32 values
//!
//! inside a row: [ A B C D ]       source_bytes = 16
//! at row edge:  [ A B | outside ] source_bytes =  8; remaining bytes become 0
//! fully outside:[ outside       ] source_bytes =  0; all 16 destination bytes become 0
//! ```
//!
//! `source_bytes = 0` does **not** skip the instruction. Every thread still
//! issues its cooperative copy. That copy reads zero source bytes and
//! zero-fills the complete destination atom; it is not a no-op.
//!
//! One path is required because branching around the copy fails two ways at
//! once. The op's lifecycle synchronizes inside the region a boundary branch
//! would skip, so a thread on the false path never reaches that barrier
//! (deadlock or undefined behavior). And the TV assignment gives every
//! shared-memory cell one owning thread, so a thread that skips leaves its
//! cells unwritten: a silent hole, wrong data, no error. Masking the effect
//! instead of the path fixes both.
//!
//! A row-major matrix stores each logical row left to right. The **leading
//! dimension**, also called the **pitch**, is the element distance between
//! two row starts. It can be larger than the logical column count because a
//! row may end with padding:
//!
//! ```text
//! columns = 5, leading_dim = 8
//! row 0 offsets: [ 0  1  2  3  4 |  5  6  7 padding ]
//! row 1 offsets: [ 8  9 10 11 12 | 13 14 15 padding ]
//!                  ^ row starts are 8 elements apart
//! ```
//!
//! Even a zero-byte instruction needs a source pointer operand. For an
//! out-of-range logical coordinate, the plan therefore selects `base + 0` as
//! a **safe fallback**. It never forms the imaginary out-of-range address:
//!
//! ```text
//! in bounds: base + checked(row * leading_dim + column)
//! zero read: base + 0  (aligned fallback; source_bytes = 0)
//! ```
//!
//! This module computes data only: the compile-time half of the split. The
//! predicate's value depends on runtime numbers, so the emitted kernel
//! evaluates it; what is settled here is which comparisons exist and why
//! every branch-free outcome is memory-safe:
//!
//! ```text
//! this module (compile time)              emitted kernel (GPU runtime)
//! ──────────────────────────              ────────────────────────────
//! which comparisons must exist            evaluate them: row < rows?
//! prove base + 0 is a safe fallback       pick the offset with select
//! reject pitch < cols, overflow,          issue cp.async with the chosen
//!   bad atom widths, loudly                 source_bytes (16 / 8 / 0)
//! ```
//!
//! Later, the compiler must use the plan to choose the offset and source
//! size consistently for every participating thread, and then issue
//! `cp.async` for each one.

use core::fmt;

/// Runtime facts promised by one row-major global-memory matrix view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GmemMatrixBounds {
    /// Number of logical rows, excluding any allocation beyond the matrix.
    pub rows: u64,
    /// Number of logical elements in each row, excluding row padding.
    pub columns: u64,
    /// Pitch: element distance from the start of one row to the next.
    pub leading_dim: u64,
    /// Byte alignment guaranteed for the matrix's base pointer.
    pub base_alignment_bytes: u32,
}

/// Logical `(row, column)` of the first element in one
/// [copy atom](crate#atoms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomCoord {
    /// Zero-based logical row.
    pub row: u64,
    /// Zero-based logical column where the transaction begins.
    pub column: u64,
}

/// Element offset that the compiler-generated cooperative copy may safely add
/// to the base pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeSourceOffset {
    /// The [copy atom](crate#atoms) starts inside the logical matrix, so this is
    /// its real offset.
    InBounds {
        /// Checked `row * leading_dim + column`, measured in elements.
        element_offset: u64,
    },
    /// The atom reads zero bytes, so use an aligned, valid pointer operand.
    AlignedFallback {
        /// Always zero today: `base + 0` is safe and aligned.
        element_offset: u64,
    },
}

impl SafeSourceOffset {
    /// Return the element offset to add to the base pointer.
    pub const fn element_offset(self) -> u64 {
        match self {
            Self::InBounds { element_offset } | Self::AlignedFallback { element_offset } => {
                element_offset
            }
        }
    }
}

/// Source operands for the uniformly issued transaction described by one
/// [copy atom](crate#atoms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredicatedSource {
    /// Number of bytes read from global memory, from zero through atom width.
    /// `cp.async` zero-fills every remaining destination byte. A value of zero
    /// still means “issue the instruction, read nothing, zero-fill the atom.”
    pub source_bytes: u32,
    /// Safe pointer offset used by that instruction.
    pub source_offset: SafeSourceOffset,
}

/// Result type returned by the predication planner.
pub type PredicationResult<T> = core::result::Result<T, PredicationError>;

/// Reason a [copy atom](crate#atoms) cannot be planned safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicationError {
    /// `cp.async` supports only 4-, 8-, or 16-byte transaction widths here.
    UnsupportedAtomBytes(u32),
    /// An element cannot have a zero-byte representation.
    ZeroElementBytes,
    /// The transaction width is not a whole number of elements.
    AtomSplitsElement {
        /// Requested transaction width in bytes.
        atom_bytes: u32,
        /// Width of one matrix element in bytes.
        element_bytes: u32,
    },
    /// The pitch would place the next row before the current logical row ends.
    LeadingDimensionTooSmall {
        /// Logical elements in one row.
        columns: u64,
        /// Promised distance between row starts, in elements.
        leading_dim: u64,
    },
    /// The base pointer is not aligned for the transaction width.
    InsufficientBaseAlignment {
        /// Required base alignment in bytes.
        required: u32,
        /// Alignment promised by the matrix view, in bytes.
        guaranteed: u32,
    },
    /// Computing `row * leading_dim + column` overflowed.
    ElementOffsetOverflow,
    /// Converting a valid element offset into bytes overflowed.
    ByteOffsetOverflow {
        /// Checked source position measured in elements.
        element_offset: u64,
        /// Width of one element in bytes.
        element_bytes: u32,
    },
    /// The atom begins at an address that is not transaction-aligned.
    MisalignedAtomStart {
        /// Source offset from the base pointer, measured in bytes.
        byte_offset: u64,
        /// Required transaction alignment in bytes.
        atom_bytes: u32,
    },
}

impl fmt::Display for PredicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedAtomBytes(bytes) => {
                write!(f, "cp.async atom must be 4, 8, or 16 bytes, got {bytes}")
            }
            Self::ZeroElementBytes => write!(f, "element width must be positive"),
            Self::AtomSplitsElement {
                atom_bytes,
                element_bytes,
            } => write!(
                f,
                "a {atom_bytes}-byte atom cannot be split into {element_bytes}-byte elements"
            ),
            Self::LeadingDimensionTooSmall {
                columns,
                leading_dim,
            } => write!(
                f,
                "leading dimension {leading_dim} is smaller than the {columns} matrix columns"
            ),
            Self::InsufficientBaseAlignment {
                required,
                guaranteed,
            } => write!(
                f,
                "a {required}-byte atom needs matching base alignment, but only {guaranteed} bytes are guaranteed"
            ),
            Self::ElementOffsetOverflow => {
                write!(f, "row * leading_dim + column overflows u64")
            }
            Self::ByteOffsetOverflow {
                element_offset,
                element_bytes,
            } => write!(
                f,
                "element offset {element_offset} times element width {element_bytes} overflows u64"
            ),
            Self::MisalignedAtomStart {
                byte_offset,
                atom_bytes,
            } => write!(
                f,
                "source byte offset {byte_offset} is not aligned for a {atom_bytes}-byte atom"
            ),
        }
    }
}

impl core::error::Error for PredicationError {}

/// Plan the source pointer and source size for the transaction described by one
/// [copy atom](crate#atoms).
///
/// `atom_bytes` is the fixed destination transaction width. The returned
/// `source_bytes` is in `0..=atom_bytes` and always contains whole elements.
/// At a right edge it stops at the logical column count, never entering row
/// padding or the next row.
///
/// For four-byte elements, a 16-byte atom, and a 12-element row pitch:
///
/// ```text
/// atom starts at row 0, column 8; columns = 10
/// logical row: [ 0 1 2 3 4 5 6 7 | 8 9 | padding ]
///                                  ^----- 8 source bytes
/// start byte: (0 * 12 + 8) * 4 = 32, which is 16-byte aligned
/// remaining 8 destination bytes are zero-filled
/// ```
///
/// If the starting coordinate is outside the matrix, the result uses the
/// aligned base fallback and `source_bytes = 0`. Lowering must still issue the
/// copy instruction; it reads nothing and zero-fills the entire atom.
pub fn plan_uniform_cp_async_source(
    matrix: GmemMatrixBounds,
    atom: AtomCoord,
    atom_bytes: u32,
    element_bytes: u32,
) -> PredicationResult<PredicatedSource> {
    if !matches!(atom_bytes, 4 | 8 | 16) {
        return Err(PredicationError::UnsupportedAtomBytes(atom_bytes));
    }
    if element_bytes == 0 {
        return Err(PredicationError::ZeroElementBytes);
    }
    if !atom_bytes.is_multiple_of(element_bytes) {
        return Err(PredicationError::AtomSplitsElement {
            atom_bytes,
            element_bytes,
        });
    }
    if matrix.leading_dim < matrix.columns {
        return Err(PredicationError::LeadingDimensionTooSmall {
            columns: matrix.columns,
            leading_dim: matrix.leading_dim,
        });
    }
    if matrix.base_alignment_bytes == 0 || !matrix.base_alignment_bytes.is_multiple_of(atom_bytes) {
        return Err(PredicationError::InsufficientBaseAlignment {
            required: atom_bytes,
            guaranteed: matrix.base_alignment_bytes,
        });
    }

    // Check logical bounds before doing address arithmetic:
    //
    // outside coordinate ──► base + 0, source_bytes = 0
    //                       (never compute its imaginary row * pitch)
    //
    // Even `(u64::MAX, u64::MAX)` therefore selects the safe base pointer.
    if atom.row >= matrix.rows || atom.column >= matrix.columns {
        return Ok(PredicatedSource {
            source_bytes: 0,
            source_offset: SafeSourceOffset::AlignedFallback { element_offset: 0 },
        });
    }

    let element_offset = atom
        .row
        .checked_mul(matrix.leading_dim)
        .and_then(|row_start| row_start.checked_add(atom.column))
        .ok_or(PredicationError::ElementOffsetOverflow)?;
    let byte_offset = element_offset.checked_mul(u64::from(element_bytes)).ok_or(
        PredicationError::ByteOffsetOverflow {
            element_offset,
            element_bytes,
        },
    )?;
    if !byte_offset.is_multiple_of(u64::from(atom_bytes)) {
        return Err(PredicationError::MisalignedAtomStart {
            byte_offset,
            atom_bytes,
        });
    }

    let atom_elements = u64::from(atom_bytes / element_bytes);
    let available_elements = matrix.columns - atom.column;
    let copied_elements = available_elements.min(atom_elements);
    let source_bytes = u32::try_from(copied_elements * u64::from(element_bytes))
        .expect("source size is bounded by a 16-byte atom");

    Ok(PredicatedSource {
        source_bytes,
        source_offset: SafeSourceOffset::InBounds { element_offset },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATRIX: GmemMatrixBounds = GmemMatrixBounds {
        rows: 3,
        columns: 10,
        leading_dim: 12,
        base_alignment_bytes: 16,
    };

    #[test]
    fn full_atom_reads_the_whole_width() {
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 1, column: 4 }, 16, 4,),
            Ok(PredicatedSource {
                source_bytes: 16,
                source_offset: SafeSourceOffset::InBounds { element_offset: 16 },
            })
        );
    }

    #[test]
    fn partial_atom_stops_at_the_logical_row_edge() {
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 1, column: 8 }, 16, 4,),
            Ok(PredicatedSource {
                source_bytes: 8,
                source_offset: SafeSourceOffset::InBounds { element_offset: 20 },
            })
        );
    }

    #[test]
    fn zero_atom_uses_the_aligned_base_fallback() {
        for atom in [
            AtomCoord { row: 1, column: 12 },
            AtomCoord {
                row: u64::MAX,
                column: u64::MAX,
            },
        ] {
            assert_eq!(
                plan_uniform_cp_async_source(MATRIX, atom, 16, 4),
                Ok(PredicatedSource {
                    source_bytes: 0,
                    source_offset: SafeSourceOffset::AlignedFallback { element_offset: 0 },
                })
            );
        }
    }

    #[test]
    fn valid_offset_arithmetic_is_checked() {
        let huge_pitch = GmemMatrixBounds {
            rows: 3,
            columns: 2,
            leading_dim: u64::MAX,
            base_alignment_bytes: 4,
        };
        assert_eq!(
            plan_uniform_cp_async_source(huge_pitch, AtomCoord { row: 1, column: 1 }, 4, 4,),
            Err(PredicationError::ElementOffsetOverflow)
        );

        let byte_overflow = GmemMatrixBounds {
            rows: 2,
            columns: 1,
            leading_dim: u64::MAX,
            base_alignment_bytes: 8,
        };
        assert_eq!(
            plan_uniform_cp_async_source(byte_overflow, AtomCoord { row: 1, column: 0 }, 8, 8,),
            Err(PredicationError::ByteOffsetOverflow {
                element_offset: u64::MAX,
                element_bytes: 8,
            })
        );
    }

    #[test]
    fn malformed_widths_and_matrix_pitch_are_loud_errors() {
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 0, column: 0 }, 12, 4),
            Err(PredicationError::UnsupportedAtomBytes(12))
        );
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 0, column: 0 }, 16, 0),
            Err(PredicationError::ZeroElementBytes)
        );
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 0, column: 0 }, 16, 3),
            Err(PredicationError::AtomSplitsElement {
                atom_bytes: 16,
                element_bytes: 3,
            })
        );

        let bad_pitch = GmemMatrixBounds {
            leading_dim: 9,
            ..MATRIX
        };
        assert_eq!(
            plan_uniform_cp_async_source(bad_pitch, AtomCoord { row: 0, column: 0 }, 16, 4),
            Err(PredicationError::LeadingDimensionTooSmall {
                columns: 10,
                leading_dim: 9,
            })
        );
    }

    #[test]
    fn base_and_atom_start_alignment_are_checked() {
        let weak_base = GmemMatrixBounds {
            base_alignment_bytes: 8,
            ..MATRIX
        };
        assert_eq!(
            plan_uniform_cp_async_source(weak_base, AtomCoord { row: 0, column: 0 }, 16, 4),
            Err(PredicationError::InsufficientBaseAlignment {
                required: 16,
                guaranteed: 8,
            })
        );
        assert_eq!(
            plan_uniform_cp_async_source(MATRIX, AtomCoord { row: 0, column: 1 }, 16, 4),
            Err(PredicationError::MisalignedAtomStart {
                byte_offset: 4,
                atom_bytes: 16,
            })
        );
    }
}
