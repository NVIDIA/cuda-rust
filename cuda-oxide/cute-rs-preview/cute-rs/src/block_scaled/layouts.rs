/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value and scale coordinate layouts for block-scaled tensors.

use crate::copy::tiled::SharedTensor;
use crate::numeric::UE8M0x4;
use core::marker::PhantomData;

/// Rows whose K values are adjacent in memory.
///
/// ```text
/// row 0: [k0 k1 k2 ...]
/// row 1: [k0 k1 k2 ...]
/// ```
///
/// `Mode` is either [`Mkl`](super::Mkl) or [`Nkl`](super::Nkl). It prevents A rows and B rows from
/// being swapped even though both have the same memory layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMajor<Mode> {
    /// Logical M or N rows in each batch.
    pub(super) rows: usize,
    /// Logical FP4 values along one row’s K dimension.
    pub(super) k: usize,
    /// Selects the M/K/L or N/K/L coordinate convention.
    pub(super) mode: PhantomData<Mode>,
}

impl<Mode> KMajor<Mode> {
    /// Describe `rows` rows, each containing `k` adjacent logical values.
    #[must_use]
    #[inline(always)]
    pub const fn new(rows: usize, k: usize) -> Self {
        Self {
            rows,
            k,
            mode: PhantomData,
        }
    }

    /// Return the number of rows in each batch: `M` or `N`.
    #[must_use]
    #[inline(always)]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Return the number of logical K values in each row.
    #[must_use]
    #[inline(always)]
    pub const fn k(self) -> usize {
        self.k
    }
}

/// Scales stored one row after another with no padding.
///
/// One scale covers `SF_VEC` neighboring K values. For `SF_VEC = 4`:
///
/// ```text
/// values: [0 1 2 3] [4 5 6 7]
/// scales: [   s0  ] [   s1  ]
/// memory: row 0 scales, then row 1 scales, ...
/// ```
///
/// [`Sm1xxBlockScaleKMajor`] stores the same logical scales in Blackwell's
/// padded hardware layout instead.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseBlockScaleKMajor<Mode, const SF_VEC: usize> {
    /// Logical M or N rows in the scale matrix.
    pub(super) rows: usize,
    /// Scale groups per row, rounded up to include a partial final group.
    pub(super) groups: usize,
    /// Carries the operand’s row-coordinate convention.
    mode: PhantomData<Mode>,
}

impl<Mode, const SF_VEC: usize> DenseBlockScaleKMajor<Mode, SF_VEC> {
    /// Describe scales for `rows` rows of `k` logical values.
    #[must_use]
    #[inline(always)]
    pub const fn new(rows: usize, k: usize) -> Self {
        const { assert!(SF_VEC > 0) };
        Self {
            rows,
            groups: k.div_ceil(SF_VEC),
            mode: PhantomData,
        }
    }

    /// Map the coordinate `(row, K group)` to its byte index (CuTeDSL
    /// `cute.crd2idx` on this layout).
    #[must_use]
    #[inline(always)]
    pub const fn crd2idx(&self, row: usize, group: usize) -> usize {
        row * self.groups + group
    }

    /// Return the number of scale groups in each row.
    #[must_use]
    #[inline(always)]
    pub const fn groups(self) -> usize {
        self.groups
    }

    /// Return the total number of scale bytes.
    #[must_use]
    #[inline(always)]
    pub const fn storage_len(self) -> usize {
        self.rows * self.groups
    }
}

/// Number of rows in one Blackwell scale block.
pub const CANONICAL_SCALE_M: usize = 128;
/// Number of neighboring K scale groups in one Blackwell scale block.
pub const CANONICAL_SCALE_K_GROUPS: usize = 4;
/// Bytes occupied by one Blackwell scale block.
pub const CANONICAL_SCALE_ATOM_BYTES: usize = 512;

/// Shared-memory layout for one Blackwell scale block with `SF = 32`.
///
/// The block contains 128 rows × 4 scale bytes = 512 bytes. TMA moves all 512
/// bytes without interpreting them. Each row's four scales form one `u32`:
///
/// ```text
/// one row: [scale 0][scale 1][scale 2][scale 3]
///
/// physical word order:
/// row 0, row 32, row 64, row 96,
/// row 1, row 33, row 65, row 97, ...
/// ```
///
/// This hardware order differs from the simple row-after-row order in
/// [`DenseBlockScaleKMajor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120ScaleAtom;

impl Sm120ScaleAtom {
    /// Number of logical rows in the block.
    pub const ROWS: usize = CANONICAL_SCALE_M;
    /// Number of `u32` words in the block; one word per row.
    pub const WORDS: usize = CANONICAL_SCALE_ATOM_BYTES / size_of::<u32>();
    /// Number of bytes copied by TMA.
    pub const BYTES: usize = CANONICAL_SCALE_ATOM_BYTES;

    /// Return the `u32` word holding one logical row's scales.
    ///
    /// The layout repeats every 128 rows, so only the lowest seven row bits are
    /// used.
    #[must_use]
    #[inline(always)]
    pub const fn word_offset(row: usize) -> usize {
        (row & 31) * 4 + ((row >> 5) & 3)
    }
}

/// A typed view of one Blackwell scale block in shared memory.
///
/// Creating this view with [`SharedTensor::from_raw_parts`] does not allocate,
/// convert, or copy memory. Its address must be aligned for `u32`, even when
/// TMA wrote the same bytes using `u16` storage boxes. `Role` keeps A and B
/// scale blocks distinct.
pub type SharedScaleAtom<Role> = SharedTensor<UE8M0x4, u32, Sm120ScaleAtom, Role>;

/// Blackwell's required memory layout for block scales.
///
/// Logically, each `(batch, row, K group)` points to one byte. Physically,
/// scales are arranged in 512-byte blocks:
///
/// ```text
/// logical scales
///      │
///      ├─ rows in groups of 128
///      └─ K groups in groups of 4
///              │
///              ▼
///       512-byte scale blocks
/// ```
///
/// Partial row or K groups still reserve a complete block.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm1xxBlockScaleKMajor<const SF_VEC: usize> {
    /// Number of complete or padded 128-row blocks per batch.
    pub(super) rest_m: usize,
    /// Number of complete or padded four-scale-group blocks along K.
    pub(super) rest_k: usize,
}

impl<const SF_VEC: usize> Sm1xxBlockScaleKMajor<SF_VEC> {
    /// Describe Blackwell-formatted scales for `rows × k` logical values.
    #[must_use]
    #[inline(always)]
    pub const fn new(rows: usize, k: usize) -> Self {
        assert!(SF_VEC > 0, "scale-vector size must be positive");
        Self {
            rest_m: rows.div_ceil(CANONICAL_SCALE_M),
            rest_k: k.div_ceil(SF_VEC * CANONICAL_SCALE_K_GROUPS),
        }
    }

    /// Map the coordinate `(batch, row, K group)` to its byte index in the
    /// 512-byte block order (CuTeDSL `cute.crd2idx` on this layout).
    #[must_use]
    #[inline(always)]
    pub const fn crd2idx(self, batch: usize, row: usize, group: usize) -> usize {
        sm1xx_scale_offset!(self, batch, row, group)
    }

    /// Return the required byte count, including complete padded blocks.
    #[must_use]
    #[inline(always)]
    pub const fn storage_len(self, batches: usize) -> usize {
        batches * self.rest_m * self.rest_k * CANONICAL_SCALE_ATOM_BYTES
    }
}
