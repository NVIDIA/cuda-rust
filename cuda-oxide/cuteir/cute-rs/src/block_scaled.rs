/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor views for values that share scales in small blocks.
//!
//! Block scaling stores narrow values, such as FP4, plus one scale for each
//! short K group:
//!
//! ```text
//! values: [v0 v1 ... v15] [v16 ... v31]
//! scales: [      s0      ] [      s1      ]
//! result:  value × its group's scale
//! ```
//!
//! The views connect the value memory, scale memory, layout, and group size in
//! one Rust type:
//!
//! ```text
//! value slice + scale slice
//!            │
//!            ▼
//!   BlockScaledTensor
//!       ├── GEMV: row ─► K64 tile ─► registers
//!       └── GEMM: global tile ─► shared tile ─► MMA
//! ```
//!
//! Creating or reshaping a view does not allocate, copy, or convert data. The
//! small wrapper methods are inlined during device compilation.

use core::marker::PhantomData;

use crate::cooperative::GmemMatrix;
use crate::markers::LeadingDim;
use crate::numeric::{E2M1, Mxf4E2M1, PackedE2M1x2, UE8M0, UE8M0x2, UE8M0x4};
use crate::tiled_copy::{GlobalCopyTensor, SharedTensor};
use crate::{Tensor, TensorElement, assume_div, load_tile};

/// Marks an operand whose logical coordinates are `(M, K, L)`.
///
/// `M` is the matrix row, `K` is the dot-product position, and `L` is batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mkl;

/// Marks an operand whose logical coordinates are `(N, K, L)`.
///
/// `N` is the output column, `K` is the dot-product position, and `L` is batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nkl;

/// Rows whose K values are adjacent in memory.
///
/// ```text
/// row 0: [k0 k1 k2 ...]
/// row 1: [k0 k1 k2 ...]
/// ```
///
/// `Mode` is either [`Mkl`] or [`Nkl`]. It prevents A rows and B rows from
/// being swapped even though both have the same memory layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMajor<Mode> {
    rows: usize,
    k: usize,
    mode: PhantomData<Mode>,
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
    rows: usize,
    groups: usize,
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

    /// Return the byte offset for `(row, K group)`.
    #[must_use]
    #[inline(always)]
    pub const fn offset(&self, row: usize, group: usize) -> usize {
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

/// The two [`UE8M0`] scales used by one K=64 MXFP4 MMA.
///
/// Each scale covers 32 K values:
///
/// ```text
/// K:      0 ................ 31 | 32 ............... 63
/// scale:              s0       |              s1
/// ```
///
/// `Mode` keeps A scales ([`Mkl`]) separate from B scales ([`Nkl`]) so the MMA
/// call cannot swap them accidentally.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaScalePair<Mode> {
    bits: u32,
    mode: PhantomData<Mode>,
}

impl<Mode> MmaScalePair<Mode> {
    /// Return the two packed bytes passed to the MMA instruction.
    ///
    /// Bits 7..0 scale K=0..31. Bits 15..8 scale K=32..63.
    #[must_use]
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }
}

/// Four neighboring K-group scales packed in one register.
///
/// ```text
/// [s0][s1][s2][s3]
///  └ pair 0 ┘└ pair 1 ┘
/// ```
///
/// [`Self::pair`] selects the two scales needed by one K=64 MMA. `Mode` keeps
/// A and B scale packs distinct.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalePack4<Mode> {
    bits: u32,
    mode: PhantomData<Mode>,
}

impl<Mode> ScalePack4<Mode> {
    /// Wrap four packed scale bytes without changing them.
    #[inline(always)]
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self {
            bits,
            mode: PhantomData,
        }
    }

    #[cfg(test)]
    /// Return the packed bytes for host-side layout tests.
    #[inline(always)]
    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }

    /// Select scales 0 and 1 for `HALF = 0`, or 2 and 3 for `HALF = 1`.
    #[must_use]
    #[inline(always)]
    pub const fn pair<const HALF: usize>(self) -> MmaScalePair<Mode> {
        const { assert!(HALF < 2) };
        MmaScalePair {
            bits: (self.bits >> (HALF * 16)) & 0xffff,
            mode: PhantomData,
        }
    }

    /// Select either two-scale pair using a runtime index and no bounds check.
    ///
    /// # Safety
    ///
    /// `half` must be exactly `0` or `1`.
    #[inline(always)]
    pub(crate) const unsafe fn pair_at_unchecked(self, half: usize) -> MmaScalePair<Mode> {
        MmaScalePair {
            bits: (self.bits >> (half * 16)) & 0xffff,
            mode: PhantomData,
        }
    }
}

/// Packed MXFP4 values and their dense UE8M0 scales for one batch (`L = 1`).
///
/// ```text
/// logical K values: [fp4 fp4 fp4 fp4] ... 32 values ...
/// value storage:    [   one f16 box   ]
/// scale storage:    [             one UE8M0 scale      ]
/// ```
///
/// The `f16` is only a 16-bit storage box for four FP4 nibbles; it is not an
/// FP16 numeric value. Scales are stored densely, one byte per 32 K values.
#[derive(Clone, Copy, Debug)]
pub struct Mxfp4BlockScaledTensor<'a, Mode: Copy> {
    values: Tensor<'a, Mxf4E2M1, KMajor<Mode>, f16>,
    scales: Tensor<'a, UE8M0, DenseBlockScaleKMajor<Mode, 32>, u8>,
}

impl<'a, Mode: Copy> Mxfp4BlockScaledTensor<'a, Mode> {
    /// View packed FP4 values and dense scale bytes as one block-scaled tensor.
    ///
    /// This does not allocate, copy, or validate memory. Later copy and load
    /// operations still require valid buffer sizes and alignment.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(
        values: &'a [f16],
        scales: &'a [u8],
        rows: usize,
        logical_k: usize,
    ) -> Self {
        Self {
            values: Tensor::from_storage(values, KMajor::new(rows, logical_k)),
            scales: Tensor::from_storage(scales, DenseBlockScaleKMajor::new(rows, logical_k)),
        }
    }

    /// Create the global-memory view used by a block-wide value copy.
    ///
    /// Logical K counts FP4 values. The returned physical view counts `f16`
    /// storage boxes, so its width is `K / 4`:
    ///
    /// ```text
    /// 128 FP4 values ──► 32 f16 storage boxes
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn values_for_copy(&self) -> GlobalCopyTensor<Mxf4E2M1, f16, LeadingDim<8>, Mode> {
        let carrier_k = self.values.layout.k / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.values.storage.as_ptr(),
            rows: self.values.layout.rows,
            cols: carrier_k,
            leading_dim: LeadingDim::<8> {
                elements: carrier_k,
            },
        })
    }

    /// View each four neighboring scale bytes as one `u32` copy word.
    ///
    /// ```text
    /// [s0][s1][s2][s3] ──► one u32
    /// ```
    ///
    /// This does not copy or convert the bytes. The tiled-copy layout decides
    /// which rows and K=128 sections are moved together.
    ///
    /// # Safety
    ///
    /// - The scale slice must contain at least `rows * groups` bytes.
    /// - Its first address must be aligned for `u32`.
    /// - Each row's number of scale groups must be divisible by four.
    /// - The returned view must not outlive this scale slice.
    #[must_use]
    #[inline(always)]
    pub unsafe fn scale_words_for_copy(
        &self,
    ) -> GlobalCopyTensor<UE8M0x4, u32, LeadingDim<1>, Mode> {
        let words = self.scales.layout.groups / 4;
        GlobalCopyTensor::from_carrier(GmemMatrix {
            base: self.scales.storage.as_ptr().cast::<u32>(),
            rows: self.scales.layout.rows,
            cols: words,
            leading_dim: LeadingDim::<1> { elements: words },
        })
    }

    /// Load the two neighboring scale bytes used by one K=64 MMA.
    ///
    /// The result keeps `Mode`, so the MMA API can distinguish A scales from B
    /// scales.
    #[must_use]
    #[inline(always)]
    pub fn load_scale_pair(&self, row: usize, pair: usize) -> MmaScalePair<Mode> {
        let offset = self.scales.layout.offset(row, pair * 2);
        MmaScalePair {
            bits: (self.scales.storage[offset] as u32)
                | ((self.scales.storage[offset + 1] as u32) << 8),
            mode: PhantomData,
        }
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
    rest_m: usize,
    rest_k: usize,
}

// Map one logical scale coordinate into Blackwell's 512-byte block order.
macro_rules! sm1xx_scale_offset {
    ($layout:expr, $batch:expr, $row:expr, $group:expr) => {{
        let layout = $layout;
        let batch = $batch;
        let row = $row;
        let group = $group;
        batch * layout.rest_m * layout.rest_k * CANONICAL_SCALE_ATOM_BYTES
            + (row >> 7) * layout.rest_k * CANONICAL_SCALE_ATOM_BYTES
            + (group >> 2) * CANONICAL_SCALE_ATOM_BYTES
            + ((row & 31) << 4)
            + (((row >> 5) & 3) << 2)
            + (group & 3)
    }};
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

    /// Return the byte offset for `(batch, row, K group)`.
    #[must_use]
    #[inline(always)]
    pub const fn offset(self, batch: usize, row: usize, group: usize) -> usize {
        sm1xx_scale_offset!(self, batch, row, group)
    }

    /// Return the required byte count, including complete padded blocks.
    #[must_use]
    #[inline(always)]
    pub const fn storage_len(self, batches: usize) -> usize {
        batches * self.rest_m * self.rest_k * CANONICAL_SCALE_ATOM_BYTES
    }
}

/// One view that keeps packed values and their scales together.
///
/// The Rust type records the data format, scale format, number of values per
/// scale, and value layout. This prevents mixing incompatible pieces later.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledTensor<
    'a,
    Data: TensorElement<u8>,
    Scale: TensorElement<u8>,
    const SF_VEC: usize,
    DataLayout,
> {
    values: Tensor<'a, Data, DataLayout, u8>,
    scales: Tensor<'a, Scale, Sm1xxBlockScaleKMajor<SF_VEC>, u8>,
}

impl<'a, Mode: Copy> BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mode>> {
    /// View packed values and Blackwell-formatted scales as one tensor.
    ///
    /// This does not validate, allocate, convert, or copy memory. The later
    /// unsafe load still requires correctly sized and aligned slices.
    #[must_use]
    #[inline(always)]
    pub const fn from_slices(values: &'a [u8], scales: &'a [u8], rows: usize, k: usize) -> Self {
        __compiler::block_scaled_make::<Mode>(values, scales, rows, k)
    }

    /// Return the number of complete K=64 tiles.
    ///
    /// Each tile has four groups of 16 values. `k` must be divisible by 64
    /// before any tile is loaded.
    #[must_use]
    #[inline(always)]
    pub fn k_tile_count(self) -> usize {
        self.values.layout.k / (16 * CANONICAL_SCALE_K_GROUPS)
    }

    /// Select one matrix or vector row in `batch` for the current thread.
    ///
    /// This calculates starting offsets only; it does not read memory.
    #[must_use]
    #[inline(always)]
    pub fn thread_row(self, batch: usize, row: usize) -> BlockScaledThreadRow<'a, Mode> {
        __compiler::block_scaled_thread_row(self, batch, row)
    }
}

/// One thread's view of a row whose K values are adjacent.
///
/// It stores slice references plus the first value and scale offsets. It does
/// not load the row.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledThreadRow<'a, Mode> {
    values: &'a [u8],
    scales: &'a [u8],
    value_row_base: usize,
    scale_row_base: usize,
    mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> BlockScaledThreadRow<'a, Mode> {
    /// Select 64 neighboring K values and their four scales.
    ///
    /// ```text
    /// row ── k_tile(0) ─► K 0..63
    ///     └─ k_tile(1) ─► K 64..127
    /// ```
    ///
    /// This calculates offsets only. Call [`BlockScaledTile64::load`] to read
    /// global memory.
    #[must_use]
    #[inline(always)]
    pub fn k_tile(self, tile: usize) -> BlockScaledTile64<'a, Mode> {
        __compiler::block_scaled_k_tile(self, tile)
    }
}

/// A view of 64 packed FP4 values and four UE8M0 scales.
///
/// Each scale covers 16 values. The view stores offsets, not loaded values.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledTile64<'a, Mode> {
    values: &'a [u8],
    scales: &'a [u8],
    value_base: usize,
    scale_base: usize,
    mode: PhantomData<Mode>,
}

impl<'a, Mode: Copy> BlockScaledTile64<'a, Mode> {
    /// Load the packed values and scales into this thread's registers.
    ///
    /// Each scale is converted to `f32` once and then reused by its group of 16
    /// FP4 values.
    ///
    /// # Safety
    ///
    /// - `values[value_base..value_base + 32]` must be readable.
    /// - `values[value_base]` must start at a 16-byte-aligned address.
    /// - `scales[scale_base..scale_base + 4]` must be readable.
    /// - `scales[scale_base]` must start at a 4-byte-aligned address.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load(self) -> LoadedBlockScaledTile64<Mode> {
        unsafe { __compiler::block_scaled_load_k64(self) }
    }
}

/// One loaded K=64 tile held in a thread's registers.
///
/// The 64 FP4 values remain packed as 32 bytes. The four scales have already
/// been converted to `f32`.
#[derive(Clone, Copy, Debug)]
pub struct LoadedBlockScaledTile64<Mode> {
    values_lo: [u8; 16],
    values_hi: [u8; 16],
    scale0: f32,
    scale1: f32,
    scale2: f32,
    scale3: f32,
    mode: PhantomData<Mode>,
}

impl<Mode: Copy> LoadedBlockScaledTile64<Mode> {
    /// Return packed FP4 pair `PAIR` in increasing K order.
    #[must_use]
    #[inline(always)]
    pub fn value_pair<const PAIR: usize>(self) -> PackedE2M1x2 {
        const { assert!(PAIR < 32) };
        let bits = if PAIR < 16 {
            self.values_lo[PAIR]
        } else {
            self.values_hi[PAIR - 16]
        };
        PackedE2M1x2::from_bits(bits)
    }

    /// Return the `f32` scale for one of the four 16-value groups.
    #[must_use]
    #[inline(always)]
    pub fn scale<const GROUP: usize>(self) -> f32 {
        const { assert!(GROUP < 4) };
        match GROUP {
            0 => self.scale0,
            1 => self.scale1,
            2 => self.scale2,
            _ => self.scale3,
        }
    }
}

impl LoadedBlockScaledTile64<Mkl> {
    /// Add the dot product of two K=64 tiles to `acc`.
    ///
    /// The types require matrix data ([`Mkl`]) on the left and vector data
    /// ([`Nkl`]) on the right:
    ///
    /// ```text
    /// acc + sum((A[k] × A_scale[k]) × B[k] × B_scale[k])
    /// ```
    ///
    /// Each scale is reused for 16 values. Each packed byte contributes two
    /// FP4 values in increasing K order.
    #[must_use]
    #[inline(always)]
    pub fn dot_accumulate(self, rhs: LoadedBlockScaledTile64<Nkl>, acc: f32) -> f32 {
        __compiler::block_scaled_dot_k64(self, rhs, acc)
    }
}

/// Stable calls recognized by the CuTe importer.
///
/// Kernel authors use the methods above. Keeping these functions in one
/// private module gives the compiler an exact boundary that does not depend
/// on whether rustc inlines an ergonomic method wrapper.
mod __compiler {
    use super::*;

    #[inline(never)]
    pub(super) const fn block_scaled_make<'a, Mode: Copy>(
        values: &'a [u8],
        scales: &'a [u8],
        rows: usize,
        k: usize,
    ) -> BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mode>> {
        let scale_layout = Sm1xxBlockScaleKMajor {
            rest_m: rows.div_ceil(CANONICAL_SCALE_M),
            rest_k: k.div_ceil(16 * CANONICAL_SCALE_K_GROUPS),
        };
        BlockScaledTensor {
            values: Tensor::from_storage(
                values,
                KMajor {
                    rows,
                    k,
                    mode: PhantomData,
                },
            ),
            scales: Tensor::from_storage(scales, scale_layout),
        }
    }

    #[inline(never)]
    pub(super) fn block_scaled_thread_row<'a, Mode: Copy>(
        tensor: BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mode>>,
        batch: usize,
        row: usize,
    ) -> BlockScaledThreadRow<'a, Mode> {
        let packed_k = tensor.values.layout.k / 2;
        let value_row_base = (batch * tensor.values.layout.rows + row) * packed_k;
        let scale_row_base = sm1xx_scale_offset!(tensor.scales.layout, batch, row, 0);

        BlockScaledThreadRow {
            values: tensor.values.storage,
            scales: tensor.scales.storage,
            value_row_base,
            scale_row_base,
            mode: PhantomData,
        }
    }

    #[inline(never)]
    pub(super) fn block_scaled_k_tile<'a, Mode: Copy>(
        row: BlockScaledThreadRow<'a, Mode>,
        tile: usize,
    ) -> BlockScaledTile64<'a, Mode> {
        BlockScaledTile64 {
            values: row.values,
            scales: row.scales,
            value_base: row.value_row_base + tile * 32,
            scale_base: row.scale_row_base + tile * CANONICAL_SCALE_ATOM_BYTES,
            mode: PhantomData,
        }
    }

    #[inline(never)]
    pub(super) unsafe fn block_scaled_load_k64<'a, Mode: Copy>(
        tile: BlockScaledTile64<'a, Mode>,
    ) -> LoadedBlockScaledTile64<Mode> {
        // State each alignment fact immediately before its load. The device
        // compiler reads that direct link; putting the value in a struct would
        // hide it.
        let value_base = unsafe { assume_div::<16>(tile.value_base) };
        let scale_base = unsafe { assume_div::<4>(tile.scale_base) };
        let values_lo = unsafe { load_tile::<u8, 16>(tile.values, value_base) };
        let values_hi = unsafe { load_tile::<u8, 16>(tile.values, value_base + 16) };
        let scale_bytes = unsafe { load_tile::<u8, 4>(tile.scales, scale_base) };

        let (scale0, scale1) = UE8M0x2::from_bytes(scale_bytes[0], scale_bytes[1]).to_f32x2();
        let (scale2, scale3) = UE8M0x2::from_bytes(scale_bytes[2], scale_bytes[3]).to_f32x2();

        LoadedBlockScaledTile64 {
            values_lo,
            values_hi,
            scale0,
            scale1,
            scale2,
            scale3,
            mode: PhantomData,
        }
    }

    #[inline(never)]
    pub(super) fn block_scaled_dot_k64(
        lhs: LoadedBlockScaledTile64<Mkl>,
        rhs: LoadedBlockScaledTile64<Nkl>,
        mut acc: f32,
    ) -> f32 {
        macro_rules! accumulate_group {
            ($group:literal; $($pair:literal),+ $(,)?) => {{
                let scale_a = lhs.scale::<$group>();
                let scale_b = rhs.scale::<$group>();
                $(
                    {
                        let (a_low, a_high) = lhs.value_pair::<$pair>().to_f32x2();
                        let (b_low, b_high) = rhs.value_pair::<$pair>().to_f32x2();
                        acc += ((a_low * scale_a) * b_low) * scale_b;
                        acc += ((a_high * scale_a) * b_high) * scale_b;
                    }
                )+
            }};
        }

        accumulate_group!(0; 0, 1, 2, 3, 4, 5, 6, 7);
        accumulate_group!(1; 8, 9, 10, 11, 12, 13, 14, 15);
        accumulate_group!(2; 16, 17, 18, 19, 20, 21, 22, 23);
        accumulate_group!(3; 24, 25, 26, 27, 28, 29, 30, 31);
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_scale_layout_matches_blackwell_golden_offsets() {
        let layout = Sm1xxBlockScaleKMajor::<16>::new(256, 128);

        assert_eq!(layout.offset(0, 0, 0), 0);
        assert_eq!(layout.offset(0, 31, 0), 496);
        assert_eq!(layout.offset(0, 32, 0), 4);
        assert_eq!(layout.offset(0, 127, 3), 511);
        assert_eq!(layout.offset(0, 0, 4), 512);
        assert_eq!(layout.offset(0, 128, 0), 1_024);
        assert_eq!(layout.offset(1, 0, 0), 2_048);
        assert_eq!(layout.storage_len(2), 4_096);
    }

    #[test]
    fn matrix_and_vector_modes_remain_distinct_types() {
        let values = [0u8; 128];
        let scales = [0u8; CANONICAL_SCALE_ATOM_BYTES];

        let matrix =
            BlockScaledTensor::<E2M1, UE8M0, 16, KMajor<Mkl>>::from_slices(&values, &scales, 4, 64);
        let vector =
            BlockScaledTensor::<E2M1, UE8M0, 16, KMajor<Nkl>>::from_slices(&values, &scales, 1, 64);

        assert_eq!(matrix.k_tile_count(), 1);
        assert_eq!(vector.k_tile_count(), 1);
        assert_eq!(matrix.thread_row(0, 3).k_tile(0).value_base, 96);
        assert_eq!(vector.thread_row(0, 0).k_tile(0).value_base, 0);
    }

    #[test]
    fn mxf4_tensor_keeps_logical_k_and_materializes_packed_carriers() {
        let values = [f16::from_bits(0); 128];
        let scales = [127u8; 16];
        let matrix = Mxfp4BlockScaledTensor::<Mkl>::from_slices(&values, &scales, 4, 64);

        let copy = matrix.values_for_copy();
        assert_eq!(copy.carrier().rows, 4);
        assert_eq!(copy.carrier().cols, 16);
        assert_eq!(copy.carrier().leading_dim.elements, 16);
        assert_eq!(matrix.scales.layout.groups(), 2);
        assert_eq!(matrix.scales.layout.storage_len(), 8);
        assert_eq!(matrix.load_scale_pair(3, 0).bits(), 0x7f7f);
    }

    #[test]
    fn staged_scale_pack_preserves_little_endian_k_order() {
        let pack = ScalePack4::<Mkl>::from_bits(0x8382_8180);
        assert_eq!(pack.pair::<0>().bits(), 0x8180);
        assert_eq!(pack.pair::<1>().bits(), 0x8382);

        #[repr(align(4))]
        struct AlignedScales([u8; 32]);
        let values = [f16::from_bits(0); 256];
        let scales = AlignedScales([127; 32]);
        let matrix = Mxfp4BlockScaledTensor::<Mkl>::from_slices(&values, &scales.0, 4, 256);
        // Safe because the wrapper gives the bytes u32 alignment and each row
        // has 8 groups, which can be split into groups of 4.
        let copy = unsafe { matrix.scale_words_for_copy() };
        assert_eq!(copy.carrier().rows, 4);
        assert_eq!(copy.carrier().cols, 2);
        assert_eq!(copy.carrier().leading_dim.elements, 2);
    }

    #[test]
    fn dense_and_canonical_scale_layouts_are_distinct() {
        let dense = DenseBlockScaleKMajor::<Mkl, 32>::new(128, 128);
        let canonical = Sm1xxBlockScaleKMajor::<32>::new(128, 128);

        assert_eq!(dense.offset(32, 0), 128);
        assert_eq!(dense.storage_len(), 512);
        assert_eq!(canonical.offset(0, 32, 0), 4);
        assert_eq!(canonical.storage_len(1), 512);
    }

    #[test]
    fn shared_scale_atom_words_match_blackwell_golden_layout() {
        let canonical = Sm1xxBlockScaleKMajor::<32>::new(128, 128);
        let mut seen = [false; Sm120ScaleAtom::WORDS];

        for row in 0..Sm120ScaleAtom::ROWS {
            let word = Sm120ScaleAtom::word_offset(row);
            assert_eq!(word * 4, canonical.offset(0, row, 0));
            for group in 0..CANONICAL_SCALE_K_GROUPS {
                assert_eq!(word * 4 + group, canonical.offset(0, row, group));
            }
            assert!(!seen[word]);
            seen[word] = true;
        }

        assert!(seen.into_iter().all(|present| present));
        assert_eq!(Sm120ScaleAtom::BYTES, 512);
        assert_eq!(
            [
                Sm120ScaleAtom::word_offset(0),
                Sm120ScaleAtom::word_offset(31),
                Sm120ScaleAtom::word_offset(32),
                Sm120ScaleAtom::word_offset(63),
                Sm120ScaleAtom::word_offset(64),
                Sm120ScaleAtom::word_offset(95),
                Sm120ScaleAtom::word_offset(96),
                Sm120ScaleAtom::word_offset(127),
            ],
            [0, 124, 1, 125, 2, 126, 3, 127]
        );
    }
}
