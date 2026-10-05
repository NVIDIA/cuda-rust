/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]
#![no_std]

// Proc-macro expansions use the public crate path so the same expansion works
// both here and in downstream device crates.
extern crate self as cuda_device;

pub use cuda_macros::{
    cluster_launch, constant, convergent, cooperative_launch, cuda_module, device, gpu_printf,
    kernel, launch_bounds, launch_contract, ptx_asm, pure, readonly,
};

// Re-export for convenience
pub mod access;
pub mod async_copy;
pub mod atomic;
pub mod barrier;
pub mod bf16;
pub mod bf16x2;
pub mod clc;
pub mod cluster;
pub mod config;
pub mod constant;
pub mod convert;
pub mod cooperative_groups;
pub mod cusimd;
pub mod debug;
pub mod disjoint;
pub mod dotprod;
pub mod f16;
pub mod f16x2;
pub mod f32x2;
pub mod fence;
pub mod float;
pub mod grid;
pub mod i16x2;
pub mod iket;
pub mod int;
pub mod mma_frag;
pub mod prmt;
pub mod ptx;
pub mod shared;
pub mod swizzle;
pub mod tcgen05;
pub mod thread;
pub mod tma;
pub mod uniform;
pub mod vector;
pub mod view;
pub mod warp;
pub mod wgmma;
pub mod wmma;

pub use barrier::{
    // Core type
    Barrier,
    BarrierToken,
    GeneralBarrier,
    Invalidated,
    // Typestate managed barrier
    ManagedBarrier,
    MmaBarrier,
    MmaBarrierHandle,
    Ready,
    // Kind markers
    TmaBarrier,
    TmaBarrier0,
    TmaBarrier1,
    // Type aliases
    TmaBarrierHandle,
    // State markers
    Uninit,
};
pub use constant::{ConstantMemory, ConstantMemoryValue};
pub use cusimd::{CuSimd, Float2, Float4, TmemRegs4, TmemRegs32};
#[doc(hidden)]
pub use disjoint::{__LaunchContractDisjointSlice, __LaunchContractDisjointSliceAbi};
pub use disjoint::{DisjointSlice, SpaceLayout};
pub use fence::*;
pub use shared::{DynamicSharedArray, SharedArray};
pub use tcgen05::{
    TensorMemoryHandle, TmemAddress, TmemDeallocated, TmemF32x4, TmemF32x32, TmemGuard, TmemReady,
    TmemUninit,
};
pub use thread::*;
pub use tma::TmaDescriptor;
#[doc(hidden)]
pub use uniform::__LaunchContractUniform;
pub use uniform::Uniform;
pub use view::{
    ColView32, ColViewIter32, GridStrideRuns32, InBounds32, InBoundsMut32, LinearTiles,
    LocalIndex32, MatrixView32, RowMajorTiles, RowView32, RowViewIter32, RuntimeRowMajorTiles,
    RuntimeTileMut32, RuntimeViewMut32, StaticTileMut32, StaticView32, StaticViewMut32,
    ThreadRunMut32, ZipView32,
};
