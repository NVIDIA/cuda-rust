/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check tensor-memory-accelerator encoding and phase alignment.

use super::thread_value::positive_count;
use super::{ValidationError, ValidationResult};
use crate::{ComposedLayout, OffsetUnit};

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
    let modes = smem_layout.outer().modes();
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
    let row_leaves = modes[0].stride.flatten();
    let col_leaves = modes[1].stride.flatten();
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
    let swizzle = smem_layout.inner();
    if swizzle.num_bits == 0 {
        return Ok(());
    }
    let m_bytes = i64::from(swizzle.num_base) + element_bytes.trailing_zeros() as i64;
    if swizzle.num_bits > 3 || m_bytes != 4 || swizzle.num_shift != 3 {
        return Err(ValidationError::InvalidCount {
            name: "swizzle is not one of TMA's encodable byte-unit S<B,4,3> modes",
            value: i64::from(swizzle.num_bits),
        });
    }
    let pitch_bytes = columns * element_bytes;
    if pitch_bytes != (16 << swizzle.num_bits) {
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
    let swizzle = smem_layout.inner();
    if swizzle.num_bits == 0 {
        return Ok(16);
    }

    // Validation above normalized the element-based swizzle to byte-unit
    // S<B,4,3>. The highest participating source bit is therefore 4+B+3-1.
    let phase_bits = 4_u32
        .checked_add(swizzle.num_bits)
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
