/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe-style building blocks for GPU kernels written in Rust.
//!
//! The types describe what a kernel means. The compiler turns a small set of
//! recognized functions into GPU operations:
//!
//! ```text
//! Tensor + Layout + TiledCopy
//!             │
//!             ▼
//!       Rust kernel code
//!             │
//!             ▼
//!    load / copy / MMA / store
//!             │
//!             ▼
//!      selected GPU backend
//! ```
//!
//! The typed wrappers are inlined, so they add no device-side storage or call
//! overhead. Functions such as [`load_tile`] and [`copy_g2s`] are compiler
//! entry points: during device compilation, the compiler replaces each call
//! with the matching CuTe operation.
//!
//! Copy operations come in two sizes:
//!
//! - [`load_tile`] and [`store_tile`] move one thread's short vector.
//! - [`copy_g2s`] lets every thread in a block help move one larger tile from
//!   global memory to shared memory.
//!
//! [`assume_div`] tells the compiler that an index has the alignment needed by
//! a later operation.
//!
//! # Small glossary
//!
//! ```text
//! thread       one running lane
//! warp         32 threads that run together
//! block / CTA  a group of threads that can share shared memory
//! tile         a small rectangular part of a tensor
//! fragment     one thread's register-held part of a tile
//! MMA          a tensor-core matrix multiply-and-add operation
//! TMA          hardware that copies complete multidimensional tiles
//!
//! global memory ──► shared memory ──► registers
//! whole GPU         one block           one thread
//! ```

#![no_std]
#![feature(f16)]
#![deny(missing_docs)]

#[cfg(feature = "host")]
extern crate alloc;

pub mod block_scaled;
pub mod block_scaled_mma;
pub mod cooperative;
pub mod epilogue;
pub mod markers;
pub mod mma;
pub mod numeric;
pub mod pipeline;
pub mod scheduler;
pub mod sm100;
pub mod tensor;
pub mod tile;
pub mod tiled_copy;
pub mod tma;

pub use block_scaled::{
    BlockScaledTensor, BlockScaledThreadRow, BlockScaledTile64, DenseBlockScaleKMajor, KMajor,
    LoadedBlockScaledTile64, Mkl, MmaScalePair, Mxfp4BlockScaledTensor, Nkl, ScalePack4,
    SharedScaleAtom, Sm1xxBlockScaleKMajor, Sm120ScaleAtom,
};
pub use block_scaled_mma::{
    Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxf4ScalePairs, Mxf4ScalePairs128, Mxf4ScaleStage,
    Mxf4ScaleTile128, Mxfp4TiledMma,
};
pub use cooperative::{GmemMatrix, SmemTile, copy_g2s};
pub use cute_layout as layout;
pub use epilogue::{
    SM120_EPILOGUE_BYTES, SM120_EPILOGUE_COLS, SM120_EPILOGUE_ELEMENTS, SM120_EPILOGUE_HALF_BYTES,
    SM120_EPILOGUE_HALF_COLS, SM120_EPILOGUE_HALF_ELEMENTS, SM120_EPILOGUE_ROWS,
    Sm120Epilogue128x128, Sm120EpilogueHalfLayout, Sm120EpilogueWarp128x128,
    sm120_epilogue_atom_origin, sm120_epilogue_physical_offset,
};
pub use markers::{C, ColMajor, Composed, CpAsync, L, LeadingDim, RowMajor, Swizzle, T2};
pub use mma::{AccC, FragA, FragB, load_matrix_a, load_matrix_b};
pub use numeric::{E2M1, Mxf4E2M1, PackedE2M1x2, UE8M0, UE8M0x2, UE8M0x4};
pub use pipeline::{Consumer, PipelineState, Producer, TmaLoadPipeline, TmaStorePipeline};
pub use scheduler::{StaticPersistentTileScheduler, WorkTile};
pub use sm100::{
    OperandA, OperandB, Sm100AccumulatorPipeline, Sm100ClusterTmaCopy, Sm100SharedTile,
    Sm100TiledMma, Sm100TmaMmaPipeline, Sm100TmaStorePipeline, Sm100Tmem, Sm100TmemEpilogue,
};
pub use tensor::{Contiguous1D, RegisterTile, Tensor, TensorElement, TensorMut, Tile1D, Zipped1D};
pub use tile::{assume_div, load_tile, store_tile};
pub use tiled_copy::{GlobalCopyTensor, SharedTensor, TiledCopy};
pub use tma::{TmaDesc, copy_tma_2d, copy_tma_s2g_2d, tile_bytes};
