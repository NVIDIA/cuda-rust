/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed copy widths, divisibility promises, and warp matrix operand roles.

use pliron_derive::pliron_attr;

/// A promise that an integer is divisible by this positive number.
///
/// This is deliberately a typed attribute instead of an unstructured string:
/// A later direct SSA consumer can ask the defining `cute.assume_div` op for
/// the exact divisor without rediscovering it from surrounding arithmetic.
#[pliron_attr(name = "cute.divisibility", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteDivisibilityAttr(#[doc = "Positive divisor promised by the caller."] pub u64);

/// Width of one copy atom in bytes.
#[pliron_attr(name = "cute.copy_atom_bytes", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteCopyAtomAttr(#[doc = "Bytes transferred by one copy instruction."] pub u32);

/// Which `mma.sync` operand a warp-cooperative matrix load feeds.
///
/// The role fixes the window shape and the `ldmatrix` variant:
///
/// ```text
/// A: 16x16 window, four 8x8 matrices (.x4), row-major fragment
/// B: 16x8  window, two  8x8 matrices (.x2), transposed in flight
/// BPair: 16x16 window, four 8x8 matrices (.x4), transposed in flight
/// BPairKMajor: 16x16 window, four 8x8 matrices (.x4), K-major B pair without transpose
/// ```
#[pliron_attr(name = "cute.matrix_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMatrixRoleAttr {
    /// A-operand fragment loaded from four 8-by-8 matrices without transpose.
    A,
    /// B-operand fragment loaded from two 8-by-8 matrices with transpose.
    B,
    /// Two B fragments loaded from four 8-by-8 matrices with transpose.
    BPair,
    /// Two K-major B fragments loaded from four matrices without transpose.
    BPairKMajor,
}
