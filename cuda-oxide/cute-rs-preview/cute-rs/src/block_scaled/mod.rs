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

// Map one logical scale coordinate into Blackwell's 512-byte block order.
/// Map batch, row and K-group coordinates into the canonical 512-byte scale-block order.
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

// Compiler boundaries keep their original module paths. Storage fields are
// visible only within this family, where the boundary builds typed views.
mod gemv;
mod layouts;
mod scales;
mod tensor;

pub use gemv::{
    BlockScaledThreadRow, BlockScaledTile64, LoadedBlockScaledTile64, LoadedNvfp4Tile64,
    Nvfp4ThreadRow, Nvfp4Tile64,
};
pub use layouts::{
    CANONICAL_SCALE_ATOM_BYTES, CANONICAL_SCALE_K_GROUPS, CANONICAL_SCALE_M, DenseBlockScaleKMajor,
    KMajor, SharedScaleAtom, Sm1xxBlockScaleKMajor, Sm120ScaleAtom,
};
pub use scales::{MmaScalePair, MmaScaleQuad, ScalePack4};
pub use tensor::{BlockScaledTensor, Mxf4BlockScaledTensor, Nvfp4BlockScaledTensor};

/// Stable calls recognized by the CuTe importer.
mod __compiler;
/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests;
