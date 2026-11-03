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
//! minimum shared-memory capacity. The unsafe copy caller supplies valid and
//! aligned pointers, the promised row pitch, sufficient shared storage, and
//! the required participating threads. The compiler uses runtime matrix
//! bounds to zero-fill the unread portion of each edge copy.

mod cooperative;
mod error;
mod ldmatrix;
#[cfg(test)]
mod tests;
mod thread_value;
mod tma;

pub use cooperative::{CooperativeCopyPlan, CooperativePlanError, validate_cooperative_copy_plan};
pub use error::ValidationError;
pub use ldmatrix::validate_ldmatrix_source;
pub use thread_value::{
    validate_atom_compatibility, validate_compact_thread_layout,
    validate_per_thread_atom_contiguity, validate_tv_exact_coverage, validate_tv_layout,
};
pub use tma::{tma_phase_alignment_bytes, validate_tma_encodable};

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
