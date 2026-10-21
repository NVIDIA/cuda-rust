/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe-style building blocks for GPU kernels written in Rust.
//!
//! A tensor view describes storage without loading it. Split it into tiles,
//! then select the part one thread will process:
//!
//! ```
//! use cute_rs::Tensor;
//! let values = [0.0f32; 10];
//! let last = Tensor::from_slice(&values).zipped_divide::<4>().slice(2);
//! assert_eq!(last.base(), 8);
//! assert!(!last.is_full()); // Two values remain; use scalar loads for this tile.
//! ```
//!
//! See `cuda-oxide/cute-rs-preview/examples/elementwise_cute` in the repository for a
//! complete load/add/store kernel and launch.
//! Host descriptor constructors require the `host` feature; device operations
//! require cuda-oxide's device compiler.
//!
//! # Modules
//!
//! - [`tensor`] describes storage, register tiles and thread partitions.
//! - [`copy`] groups cooperative, tiled, vector and TMA transfers.
//! - [`mma`] contains common fragments and the BF16 and block-scaled families.
//! - [`attention`] groups host metadata and device-available SM100 operations.
//! - [`epilogue`] contains SM120 output staging.
//!
//! Existing root exports and module aliases remain available. For example,
//! `cute_rs::tma::TmaDesc` and `cute_rs::copy::tma::TmaDesc` are the same type.
//! Architecture ownership and pipeline modules retain their existing roles.
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

pub mod arch;
pub mod attention;
pub mod block_scaled;
pub mod copy;
#[cfg(feature = "host")]
pub mod decoder;
pub mod epilogue;
#[cfg(feature = "host")]
pub mod ffn;
pub mod markers;
pub mod mma;
pub mod numeric;
pub mod pipeline;
pub mod reduction;
pub mod scheduler;
pub mod sm100;
pub mod static_layout;
pub mod tensor;

/// Compatibility alias for SM100 attention operations in [`attention::sm100`].
pub use attention::sm100 as sm100_attention;
/// Compatibility alias for cooperative transfers in [`copy::cooperative`].
pub use copy::cooperative;
/// Compatibility alias for tiled transfers in [`copy::tiled`].
pub use copy::tiled as tiled_copy;
/// Compatibility alias for Tensor Memory Accelerator transfers in [`copy::tma`].
pub use copy::tma;
/// Compatibility alias for vector transfers in [`copy::vector`].
pub use copy::vector as tile;
/// Compatibility alias for BF16 matrix multiply in [`mma::bf16`].
pub use mma::bf16 as bf16_mma;
/// Compatibility alias for block-scaled matrix multiply in [`mma::block_scaled`].
pub use mma::block_scaled as block_scaled_mma;
/// Compatibility alias for thread partitions in [`tensor::partition`].
pub use tensor::partition;

pub use block_scaled::{
    BlockScaledTensor, BlockScaledThreadRow, BlockScaledTile64, DenseBlockScaleKMajor, KMajor,
    LoadedBlockScaledTile64, LoadedNvfp4Tile64, Mkl, MmaScalePair, MmaScaleQuad,
    Mxfp4BlockScaledTensor, Nkl, Nvfp4BlockScaledTensor, Nvfp4ThreadRow, Nvfp4Tile64, ScalePack4,
    SharedScaleAtom, Sm1xxBlockScaleKMajor, Sm120ScaleAtom,
};
pub use copy::cooperative::{GmemMatrix, SmemTile, copy_g2s};
pub use copy::tiled::{GlobalCopyTensor, SharedTensor, TiledCopy};
pub use copy::tma::{TmaDesc, copy_tma_2d, copy_tma_s2g_2d, tile_bytes};
pub use copy::vector::{
    GlobalStoreElement, GlobalStoreTile, assume_div, load_tile, store_global_no_allocate,
    store_tile,
};
pub use cute_layout as layout;
pub use epilogue::{
    SM120_EPILOGUE_BYTES, SM120_EPILOGUE_COLS, SM120_EPILOGUE_ELEMENTS, SM120_EPILOGUE_HALF_BYTES,
    SM120_EPILOGUE_HALF_COLS, SM120_EPILOGUE_HALF_ELEMENTS, SM120_EPILOGUE_ROWS,
    Sm120Epilogue128x128, Sm120EpilogueHalfLayout, Sm120EpilogueWarp128x128,
    sm120_epilogue_atom_origin, sm120_epilogue_physical_offset,
};
pub use markers::{C, ColMajor, Composed, CpAsync, L, LeadingDim, RowMajor, Swizzle, T2};
pub use mma::block_scaled::{
    Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxf4ScalePairs, Mxf4ScalePairs128, Mxf4ScaleStage,
    Mxf4ScaleTile128, Mxfp4TiledMma,
};
pub use mma::{
    AccC, FragA, FragB, FragBPair, load_matrix_a, load_matrix_b, load_matrix_b_pair,
    load_matrix_b_pair_k_major,
};
pub use numeric::{
    Bf16, E2M1, E4M3, E4M3x2, E4M3x4, Mxf4E2M1, PackedE2M1x2, UE8M0, UE8M0x2, UE8M0x4, e4m3_to_f32,
};
pub use pipeline::{Consumer, PipelineState, Producer, TmaLoadPipeline, TmaStorePipeline};
pub use scheduler::{StaticPersistentTileScheduler, WorkTile};
pub use sm100::{
    OperandA, OperandB, Sm100AccumulatorPipeline, Sm100ClusterTmaCopy, Sm100SharedTile,
    Sm100TiledMma, Sm100TmaMmaPipeline, Sm100TmaStorePipeline, Sm100Tmem, Sm100TmemEpilogue,
};
pub use tensor::{Contiguous1D, RegisterTile, Tensor, TensorElement, TensorMut, Tile1D, Zipped1D};
