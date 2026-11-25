/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler boundaries for block-scaled GEMV.

use super::*;
use crate::numeric::{Float4E2M1FN, Float8E4M3FN, Float8E8M0FNU};
use crate::tensor::Tensor;
use core::marker::PhantomData;

#[inline(never)]
/// Compiler boundary constructing packed value and canonical scale views for K16 scaling.
pub(super) const fn block_scaled_make<'a, Mode: Copy>(
    values: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    k: usize,
) -> BlockScaledTensor<'a, Float4E2M1FN, Float8E8M0FNU, 16, KMajor<Mode>> {
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
/// Compiler boundary selecting a batch row and computing its value/scale byte bases.
pub(super) fn block_scaled_thread_row<'a, Mode: Copy>(
    tensor: BlockScaledTensor<'a, Float4E2M1FN, Float8E8M0FNU, 16, KMajor<Mode>>,
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
/// Compiler boundary selecting one K64 tile: 32 packed bytes and four scale bytes.
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
/// Load one K64 tile and decode its four scales.
///
/// # Safety
/// The value and scale ranges must be live, in bounds and aligned as required by the tile’s public load operation.
pub(super) unsafe fn block_scaled_load_k64<Mode: Copy>(
    tile: BlockScaledTile64<'_, Mode>,
) -> LoadedBlockScaledTile64<Mode> {
    let _ = (
        tile.values,
        tile.scales,
        tile.value_base,
        tile.scale_base,
        tile.mode,
    );
    unreachable!("CuTe block-scaled load marker requires device compilation")
}

#[inline(never)]
/// Accumulate all 64 scaled FP4 products in fixed K-group order.
pub(super) fn block_scaled_dot_k64(
    lhs: LoadedBlockScaledTile64<Mkl>,
    rhs: LoadedBlockScaledTile64<Nkl>,
    acc: f32,
) -> f32 {
    let _ = (lhs, rhs, acc);
    unreachable!("CuTe block-scaled dot marker requires device compilation")
}

#[inline(never)]
/// Compiler boundary constructing packed E2M1 value and canonical E4M3 scale views for K16 scaling.
pub(super) const fn nvfp4_make<'a, Mode: Copy>(
    values: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    k: usize,
) -> BlockScaledTensor<'a, Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mode>> {
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
/// Compiler boundary selecting an NVFP4 batch row and computing its value/scale byte bases.
pub(super) const fn nvfp4_thread_row<'a, Mode: Copy>(
    tensor: BlockScaledTensor<'a, Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mode>>,
    batch: usize,
    row: usize,
) -> Nvfp4ThreadRow<'a, Mode> {
    let packed_k = tensor.values.layout.k / 2;
    let value_row_base = (batch * tensor.values.layout.rows + row) * packed_k;
    let scale_row_base = sm1xx_scale_offset!(tensor.scales.layout, batch, row, 0);
    Nvfp4ThreadRow {
        values: tensor.values.storage,
        scales: tensor.scales.storage,
        value_row_base,
        scale_row_base,
        mode: PhantomData,
    }
}

#[inline(never)]
/// Compiler boundary selecting one K64 NVFP4 tile: 32 packed bytes and four E4M3 scale bytes.
pub(super) const fn nvfp4_k_tile<'a, Mode: Copy>(
    row: Nvfp4ThreadRow<'a, Mode>,
    tile: usize,
) -> Nvfp4Tile64<'a, Mode> {
    Nvfp4Tile64 {
        values: row.values,
        scales: row.scales,
        value_base: row.value_row_base + tile * 32,
        scale_base: row.scale_row_base + tile * CANONICAL_SCALE_ATOM_BYTES,
        mode: PhantomData,
    }
}

#[inline(never)]
/// Load one K64 NVFP4 tile and decode its four E4M3 scales.
pub(super) unsafe fn nvfp4_load_k64<Mode: Copy>(
    tile: Nvfp4Tile64<'_, Mode>,
) -> LoadedNvfp4Tile64<Mode> {
    let _ = (
        tile.values,
        tile.scales,
        tile.value_base,
        tile.scale_base,
        tile.mode,
    );
    unreachable!("CuTe NVFP4 load marker requires device compilation")
}

#[inline(never)]
/// Accumulate all 64 E4M3-scaled FP4 products in fixed K-group order.
pub(super) fn nvfp4_dot_k64(
    lhs: LoadedNvfp4Tile64<Mkl>,
    rhs: LoadedNvfp4Tile64<Nkl>,
    acc: f32,
) -> f32 {
    let _ = (
        lhs.values_lo,
        lhs.values_hi,
        lhs.scale0,
        lhs.scale1,
        lhs.scale2,
        lhs.scale3,
        rhs,
        acc,
    );
    unreachable!("CuTe NVFP4 dot marker requires device compilation")
}
