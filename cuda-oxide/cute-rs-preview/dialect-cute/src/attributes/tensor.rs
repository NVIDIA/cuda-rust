/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor storage, access, format, and selection metadata.

use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

/// Where a tensor view's storage lives.
///
/// The first tensor flow starts in global memory. TMA adds a shared-memory
/// destination without changing the meaning of the existing global variant:
///
/// ```text
/// host buffer ── kernel argument ──► global GPU memory
///                                      │ TMA
///                                      ▼
///                              CTA shared memory
/// ```
///
/// Keeping the choice typed avoids turning address-space numbers such as `1`
/// or `3` into a hidden contract.
#[pliron_attr(name = "cute.tensor_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorAddressSpaceAttr {
    /// Device global memory used by tensor inputs and outputs.
    Gmem,
    /// CTA-local shared memory, visible to every thread in one block.
    Smem,
}

/// Whether a tensor view may only read or may also write.
///
/// ```text
/// ReadOnly   storage ──► registers
/// ReadWrite  storage ◄── registers
/// ```
#[pliron_attr(name = "cute.tensor_access", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorAccessAttr {
    /// Storage may be read but must not be written through this view.
    ReadOnly,
    /// Storage may be both read and written through this view.
    ReadWrite,
}

/// How to read the bits stored by a tensor view.
///
/// `Plain` is the elementwise path: one `f16` or `f32` storage value already
/// is the logical value. The packed GEMV path keeps the other two meanings
/// visible:
///
/// ```text
/// E2M1   two four-bit values in global u8, or four in shared f16
/// UE8M0  one scale in global u8, or four packed scales in shared u32
/// ```
#[pliron_attr(name = "cute.tensor_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorFormatAttr {
    /// Storage elements already have the tensor's logical scalar meaning.
    Plain,
    /// Packed four-bit floating-point values requiring decoding for scalar arithmetic.
    E2M1,
    /// Unsigned exponent-only scale values for block-scaled arithmetic.
    UE8M0,
    /// Eight-bit E4M3FN floating-point scale values for NVFP4 block-scaled arithmetic.
    E4M3,
}

/// Which logical row coordinate a tensor uses.
///
/// ```text
/// Mkl  rows follow M: A[m, k, l]
/// Nkl  rows follow N: B[n, k, l]
/// Generic             elementwise data
/// ```
///
/// `Nkl` is a single vector row when GEMV has `N = 1`, and a full B matrix
/// when GEMM has many N rows. Keeping the coordinate name avoids baking one
/// kernel profile into the tensor type.
///
/// The role is a compile-time fact. It does not change the stored bytes.
#[pliron_attr(name = "cute.tensor_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorRoleAttr {
    /// No matrix-operand coordinate role is assigned to the storage.
    Generic,
    /// Rows address A's M coordinate, with reduction K and batch L.
    Mkl,
    /// Rows address B's N coordinate, with reduction K and batch L.
    Nkl,
}

/// The one-dimensional layout carried by a v0 tensor view.
///
/// The tile width stays inside the layout instead of living in a nearby
/// untyped integer:
///
/// ```text
/// Contiguous1D       [0 1 2 3 4 5 6 7]
/// Zipped1D<4>        [0 1 2 3] [4 5 6 7]
/// Tile1D<4>           ^ one selected group ^
/// ```
#[pliron_attr(name = "cute.tensor_layout", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorLayoutAttr {
    /// A complete contiguous one-dimensional tensor before tiling.
    Contiguous1D,
    /// The tensor divided into groups of the stored element width.
    Zipped1D(#[doc = "Elements in each logical tile."] u64),
    /// One selected tile whose width is stored in this variant.
    Tile1D(#[doc = "Elements in the selected logical tile."] u64),
    /// Logical K values are adjacent. Packed formats may place more than one
    /// logical value in each storage element.
    KMajor,
    /// Blackwell's canonical scale order. The number says how many logical
    /// values share one scale. The fixed atom geometry is named by the
    /// `SM1XX_BLOCK_SCALE_*` constants in this module.
    BlockScaleKMajor(#[doc = "Logical K values sharing one scale."] u64),
    /// A two-dimensional tile moved as raw carrier values by TMA.
    ///
    /// The exact row/column shape, stride, and swizzle live in the enclosing
    /// `cute.tma_view` type as a composed layout.
    Tma2D,
}

impl CuteTensorLayoutAttr {
    /// Return the tile width for a zipped or selected-tile layout.
    #[must_use]
    pub const fn tile_size(self) -> Option<u64> {
        match self {
            Self::Contiguous1D | Self::KMajor | Self::BlockScaleKMajor(_) | Self::Tma2D => None,
            Self::Zipped1D(size) | Self::Tile1D(size) => Some(size),
        }
    }

    /// Return how many logical K values share one scale, when this is the
    /// canonical block-scale layout.
    #[must_use]
    pub const fn values_per_scale(self) -> Option<u64> {
        match self {
            Self::BlockScaleKMajor(size) => Some(size),
            _ => None,
        }
    }
}

/// Logical rows inside one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_ROWS_PER_ATOM: u64 = 128;
/// Consecutive K-scale groups inside one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_GROUPS_PER_ATOM: u64 = 4;
/// Physical bytes reserved by one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_ATOM_BYTES: u64 = 512;

impl Verify for CuteTensorLayoutAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.tile_size().is_some_and(|size| size == 0) {
            return verify_err_noloc!("cute.tensor_layout tile size must be greater than zero");
        }
        if matches!(self, Self::BlockScaleKMajor(0)) {
            return verify_err_noloc!(
                "cute.tensor_layout values per scale must be greater than zero"
            );
        }
        Ok(())
    }
}

/// A byte-alignment promise attached to a tensor view or vector transfer.
///
/// This is a promise, not padding owned by the IR. For example, `16` says
/// the address used by a full tile is safe for one 16-byte transaction.
#[pliron_attr(name = "cute.alignment_bytes", format = "$0")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteAlignmentAttr(#[doc = "Promised alignment in bytes."] pub u64);

impl Verify for CuteAlignmentAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.0 == 0 || !self.0.is_power_of_two() {
            return verify_err_noloc!(
                "cute.alignment_bytes must be a positive power of two, got {}",
                self.0
            );
        }
        Ok(())
    }
}
