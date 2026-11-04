/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe-style building blocks for GPU kernels written in Rust (`cutlass.cute`, `cutlass.pipeline`, `cutlass.utils`).
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
//! # Modules (1-to-1 with CuTeDSL)
//!
//! - [`layout`] — compile-time layouts, swizzles, and copy-atom markers (`cutlass.cute.core`).
//! - [`tensor`] — `Tensor`, `TensorMut`, `RegisterTile`, and `StridedTensor` partitions (`cutlass.cute.tensor`).
//! - [`atom`] — `TiledCopy`, `ThrCopy`, cooperative G2S copies, and vector copies (`cutlass.cute.atom`).
//! - [`numeric`] — `BFloat16`, `Float4E2M1FN`, `Float8E4M3FN`, `Float8E8M0FNU` (`cutlass.cute.typing`).
//! - [`arch`] — thread/warp/cluster indices, `NamedBarrier`, fences, and warp reductions (`cutlass.cute.arch`).
//! - [`nvgpu`] — GPU hardware atoms: [`nvgpu::common`], [`nvgpu::cpasync`], [`nvgpu::warp`], [`nvgpu::tcgen05`] (`cutlass.cute.nvgpu`).
//! - [`pipeline`] — `PipelineTmaAsync`, `PipelineTmaUmma`, `PipelineUmmaAsync`, `PipelineTmaStore` (`cutlass.pipeline`).
//! - [`utils`] — `TmemAllocator`, `StaticPersistentTileScheduler`, `Sm1xxBlockScaleKMajor` (`cutlass.utils`).

#![no_std]
#![feature(f16)]
#![deny(missing_docs, clippy::missing_docs_in_private_items)]

#[cfg(feature = "host")]
extern crate alloc;

pub mod arch;
pub mod atom;
pub mod layout;
pub mod numeric;
pub mod nvgpu;
pub mod pipeline;
pub mod tensor;
pub mod utils;

pub use cute_layout::static_layout;

pub use atom::cooperative::{GmemMatrix, SmemTile, copy_g2s};
pub use atom::tiled::{GlobalCopyTensor, SharedTensor, ThrCopy, TiledCopy};
pub use atom::vector::{
    GlobalStoreElement, GlobalStoreTile, assume, load_tile, store_global_no_allocate, store_tile,
};
pub use cute_layout::{
    crd2idx_col_major_2d, crd2idx_col_major_3d, crd2idx_row_major_2d, crd2idx_row_major_3d,
    idx2crd_col_major_2d, idx2crd_grouped_2d, idx2crd_row_major_2d, sm1xx_block_scale_crd2idx,
    sm1xx_scale_atom_word_offset, sm100_noswizzle_burst_u64_crd2idx,
};
pub use layout::{
    C, ColMajor, ComposedLayout, CopyG2S, CopyUniversal, Layout, LeadingDim, RowMajor, Swizzle,
};
pub use numeric::{
    BFloat16, E4M3x2, E4M3x4, Float4E2M1FN, Float4E2M1FNx2, Float4E2M1FNx8, Float8E4M3FN,
    Float8E8M0FNU, Mxf4E2M1, UE8M0x2, UE8M0x4,
};
pub use nvgpu::common::{Mkl, Nkl, OperandA, OperandB};
pub use nvgpu::cpasync::{
    Sm100ClusterTmaCopy, TensorMap, TmaCopy, copy_tma_2d, copy_tma_s2g_2d, size_in_bytes,
};
pub use nvgpu::tcgen05::{
    MmaSs, MmaTs, SharedLayout, SharedTile, SharedTileArray, Sm100BlockScaledTiledMma,
    Sm100ScaleTile, Sm100SharedTile, Sm100TiledMma, Sm100TmemEpilogue, SmemLayout, TmemFragment,
    tmem_row_address,
};
pub use nvgpu::warp::block_scaled::{
    Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxf4Nvf4TiledMma, Mxf4ScalePairs, Mxf4ScalePairs128,
    Mxf4ScaleStage, Mxf4ScaleTile128, Mxf4TiledMma, Nvfp4TiledMma,
};
pub use nvgpu::warp::epilogue::{
    SM120_EPILOGUE_BYTES, SM120_EPILOGUE_COLS, SM120_EPILOGUE_ELEMENTS, SM120_EPILOGUE_HALF_BYTES,
    SM120_EPILOGUE_HALF_COLS, SM120_EPILOGUE_HALF_ELEMENTS, SM120_EPILOGUE_ROWS,
    Sm120Epilogue128x128, Sm120EpilogueHalfLayout, Sm120EpilogueWarp128x128,
    sm120_epilogue_atom_origin, sm120_epilogue_physical_offset,
};
pub use nvgpu::warp::{
    FragmentA, FragmentB, FragmentBPair, FragmentC, LdmatrixElement, load_matrix_a, load_matrix_b,
    load_matrix_b_pair, load_matrix_b_pair_k_major,
};
pub use pipeline::{
    Consumer, PipelineState, PipelineTmaAsync, PipelineTmaStore, PipelineTmaStoreSm100,
    PipelineTmaUmma, PipelineUmmaAsync, Producer,
};
pub use static_layout::{crd2idx_2d, idx2crd_2d};
pub use tensor::{
    BitcastNumeric, Contiguous1D, ReductionOp, RegisterTile, Tensor, TensorElement, TensorMut,
    TensorSSA, Tile1D, ToNumeric, Zipped1D,
};
pub use utils::blockscaled::{
    BlockScaledTensor, BlockScaledThreadRow, BlockScaledTile64, DenseBlockScaleKMajor, KMajor,
    LoadedBlockScaledTile64, LoadedNvfp4Tile64, MmaScalePair, MmaScaleQuad, Mxf4BlockScaledTensor,
    Nvfp4BlockScaledTensor, Nvfp4ThreadRow, Nvfp4Tile64, ScalePack4, SharedNvfp4ScaleAtom,
    SharedScaleAtom, Sm1xxBlockScaleKMajor, Sm120ScaleAtom,
};
pub use utils::scheduler::{
    PersistentTileSchedulerParams, StaticPersistentTileScheduler, WorkTileInfo,
};
pub use utils::tmem_allocator::{Sm100DeallocBarrier, Sm100Tmem, TmemAllocator};
