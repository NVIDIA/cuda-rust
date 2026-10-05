/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) for Hopper `sm_90a`.
//!
//! WGMMA operates at the warpgroup level (128 threads = 4 warps) to perform
//! efficient tensor core matrix multiplication. Unlike WMMA which operates
//! per-warp (32 threads), WGMMA leverages the full warpgroup for larger tiles.
//!
//! The control and descriptor helpers are available. The
//! `m64n64k16.f32.bf16.bf16` MMA variant has value-threaded lowering for
//! several deliberately restricted, provably safe region shapes:
//!
//! - linear full-drain regions ending in `wgmma_wait_group::<0>()`;
//! - a canonical counted K-loop with affine descriptor recurrences;
//! - straight-line partial-wait pipelines with a static `wait_group<N>` and
//!   `N + 1` independent accumulator slots.
//!
//! The `m64n64k16.f32.f16.f16` variant supports canonical linear full-drain
//! regions and the canonical counted K-loop. The `m64n64k8.f32.tf32.tf32`
//! variant uses the canonical linear full-drain carrier only. The
//! `m64n128k16.f32.bf16.bf16` variant supports canonical linear full-drain
//! regions with a 64-value `[[f32; 8]; 8]` accumulator. Every accepted
//! asynchronous lifetime is fused into one convergent inline-PTX scope and ends
//! in `wait_group<0>` before accumulator values become visible to LLVM again.
//! Unsupported m64n64 BF16 full-drain pointer shapes retain the deferred
//! pointer-form fallback. m64n128, F16, and TF32 have no pointer-form fallback;
//! the legacy K=16 TF32 entry point remains unsupported.
//!
//! # Architecture
//!
//! ```text
//! WGMMA m64n64k16 Operation:
//!
//!     A (64×16)         B (16×64)           D (64×64)
//!   ┌──────────┐      ┌───────────────┐    ┌───────────────┐
//!   │          │      │               │    │               │
//!   │  64 rows │  ×   │   16 rows     │ =  │   64×64       │
//!   │  16 cols │      │   64 cols     │    │ accumulator   │
//!   │          │      │               │    │               │
//!   └──────────┘      └───────────────┘    └───────────────┘
//!   row-major         col-major            distributed across
//!   in SMEM           in SMEM              128 threads
//! ```
//!
//! # Per-Thread Accumulator
//!
//! Each thread in the 128-thread warpgroup holds 32 floats:
//! ```rust,ignore
//! let mut acc: [[f32; 8]; 4] = [[0.0; 8]; 4];
//! // Total: 128 threads × 32 = 4096 floats = 64×64 tile
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::wgmma::*;
//!
//! let mut acc: [[f32; 8]; 4] = [[0.0; 8]; 4];
//! let desc_a = make_smem_desc(a_smem_ptr);
//! let desc_b = make_smem_desc(b_smem_ptr);
//!
//! wgmma_fence();
//! wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
//! wgmma_commit_group();
//! wgmma_wait_group::<0>();
//!
//! // `acc` may be read only after the wait.
//! ```
//!
//! # Current Lowering Contract
//!
//! The compiler selects a value-threaded path only when it can prove the whole
//! asynchronous accumulator lifetime.
//!
//! - **BF16 linear full drain:** one canonical `[[f32; 8]; 4]` accumulator,
//!   one commit, and a final `wgmma_wait_group::<0>()`.
//! - **BF16 m64n128 linear full drain:** one canonical `[[f32; 8]; 8]`
//!   accumulator carries 64 `f32` values through one or more homogeneous
//!   m64n128 MMAs, one commit, and a final `wgmma_wait_group::<0>()`. There is
//!   no pointer fallback, counted-loop lowering, or partial-wait pipeline for
//!   this shape.
//! - **F16 linear full drain:** the m64n64 canonical accumulator and full-drain
//!   lifetime are supported, but there is no pointer fallback or partial-wait
//!   pipeline for F16.
//! - **TF32 linear full drain:** `m64n64k8.f32.tf32.tf32` uses the same
//!   canonical accumulator and full-drain lifetime. TF32 has no pointer
//!   fallback, counted-loop lowering, or partial-wait pipeline.
//! - **Canonical counted K-loop:** one BF16 or F16 m64n64 MMA per iteration,
//!   compile-time trip count, and `u64` descriptor recurrences of the form `desc + const`. The
//!   fence-to-final-wait lifetime is fused so the accumulator is loaded and
//!   stored once for the whole loop rather than once per iteration.
//! - **Partial-wait pipeline:** a static `wgmma_wait_group::<N>()` with
//!   `1 <= N <= 7`, exactly `N + 1` distinct canonical accumulator slots, one
//!   MMA per committed group, round-robin slot reuse only after the matching
//!   partial wait, and a mandatory final `wgmma_wait_group::<0>()`.
//! - Accumulator values must not be read or written while their asynchronous
//!   lifetime is pending.
//! - Unsupported BF16 full-drain accumulator pointer shapes retain the deferred
//!   pointer-form lowering. Dynamic partial waits, unsupported control flow,
//!   malformed pipeline schedules, F16 partial-wait/pipelined shapes, TF32
//!   counted-loop or partial-wait shapes, and the legacy K=16 TF32
//!   compatibility entry point are rejected.
//!
//! # Hardware Support
//!
//! - **sm_90a (Hopper)**: H100, H200

// =============================================================================
// WGMMA Synchronization Primitives
// =============================================================================

include!("generated/wgmma_control.rs");

// =============================================================================
// SMEM Descriptor Creation
// =============================================================================

/// Create a 64-bit shared memory descriptor for WGMMA input matrices.
///
/// This helper creates the fixed-layout descriptor used by the current
/// lowering. It combines the shared-memory address with fixed stride and
/// swizzle fields.
///
/// # Parameters
///
/// - `ptr`: Pointer to matrix data in shared memory
///
/// # Returns
///
/// A 64-bit descriptor suitable for WGMMA instructions.
///
/// # Encoding
///
/// ```rust,ignore
/// ((shared_address >> 4) & 0x3fff) | 0xC000000800080000
/// ```
///
/// # Safety
///
/// - `ptr` must point to valid shared memory
/// - The memory layout must match WGMMA requirements (proper alignment, swizzling)
///
/// # PTX
///
/// Uses `cvta.to.shared.u64` to convert the generic pointer.
#[inline(never)]
pub unsafe fn make_smem_desc(ptr: *const u8) -> u64 {
    let _ = ptr;
    // Lowered to inline PTX:
    // {
    //   .reg .u64 addr;
    //   cvta.to.shared.u64 addr, %ptr;
    //   shr.u64 addr, addr, 4;
    //   and.b64 addr, addr, 0x3fff;
    //   or.b64 %result, addr, 0xC000000800080000;
    // }
    unreachable!("make_smem_desc called outside CUDA kernel context")
}

/// Compatibility entry point for a custom SMEM descriptor.
///
/// This function does not have an importer or lowering path yet. It remains
/// public to avoid breaking existing source code.
///
/// # Parameters
///
/// - `ptr`: Pointer to matrix data in shared memory
/// - `leading_dim`: Leading dimension in bytes (divided by 16 internally)
/// - `stride`: Stride in bytes (divided by 16 internally)
/// - `swizzle_128b`: Enable 128-byte swizzling
///
/// # Safety
///
/// - `ptr` must be a valid pointer to matrix data in shared memory
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn make_smem_desc_custom(
    ptr: *const u8,
    leading_dim: u32,
    stride: u32,
    swizzle_128b: bool,
) -> u64 {
    let _ = (ptr, leading_dim, stride, swizzle_128b);
    unreachable!("make_smem_desc_custom called outside CUDA kernel context")
}

// =============================================================================
// WGMMA Matrix Multiply-Accumulate Instructions
// =============================================================================

/// Warpgroup matrix multiply-accumulate: D += A × B.
///
/// Performs a 64×64×16 matrix multiplication using tensor cores at the
/// warpgroup level. All 128 threads in the warpgroup participate.
///
/// # Matrix Dimensions
///
/// - **A**: 64×16 (M=64 rows, K=16 cols), row-major in shared memory
/// - **B**: 16×64 (K=16 rows, N=64 cols), column-major in shared memory
/// - **D**: 64×64 output, accumulated in registers
///
/// # Accumulator Layout
///
/// The 64×64 output is distributed across 128 threads:
/// - Each thread holds 32 floats in `[[f32; 8]; 4]`
/// - 128 threads × 32 = 4096 floats = 64×64
///
/// # Parameters
///
/// - `acc`: Mutable reference to the accumulator (32 floats per thread)
/// - `desc_a`: SMEM descriptor for matrix A (from `make_smem_desc`)
/// - `desc_b`: SMEM descriptor for matrix B (from `make_smem_desc`)
///
/// # PTX
///
/// ```ptx
/// wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
///     {%f0, %f1, ..., %f31}, %rd_desc_a, %rd_desc_b,
///     1, 1, 1, 0, 0;
/// ```
///
/// # Lowering Contract
///
/// This function must participate in one compiler-supported WGMMA region.
/// Linear full-drain regions use one commit and `wgmma_wait_group::<0>()`;
/// proven partial-wait pipelines may use multiple commits and a static
/// `wgmma_wait_group::<N>()` before the mandatory final full wait. Canonical
/// counted K-loops may also be fused across iterations. The compiler rejects
/// regions whose accumulator lifetime it cannot prove safe.
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
/// - `acc` must not be read or written until the region's final
///   `wgmma_wait_group::<0>()` returns
///
/// # Example
///
/// ```rust,ignore
/// wgmma_fence();
/// wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
/// wgmma_commit_group();
/// wgmma_wait_group::<0>();
/// ```
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_bf16(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k16_f32_bf16 called outside CUDA kernel context")
}

/// Warpgroup matrix multiply-accumulate: 64x128x16 BF16 with f32 accumulation.
///
/// The 64x128 output tile is distributed as 64 `f32` accumulator values per
/// thread, represented by `[[f32; 8]; 8]`. This variant currently supports
/// canonical linear full-drain regions only.
///
/// # PTX
///
/// ```ptx
/// wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
///     {%f0, %f1, ..., %f63}, %rd_desc_a, %rd_desc_b,
///     1, 1, 1, 0, 0;
/// ```
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
/// - `acc` must not be read or written until the region's final
///   `wgmma_wait_group::<0>()` returns
#[inline(never)]
pub unsafe fn wgmma_mma_m64n128k16_f32_bf16(acc: &mut [[f32; 8]; 8], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n128k16_f32_bf16 called outside CUDA kernel context")
}

/// WGMMA with f32 accumulator and f16 inputs.
///
/// This variant supports canonical linear full-drain regions and canonical
/// counted K-loops with a compile-time trip count and affine `u64` descriptor
/// recurrences. The accumulator must have the public `[[f32; 8]; 4]` shape.
/// Partial waits and non-canonical pointer fallback remain unsupported for F16.
///
/// # PTX
///
/// ```ptx
/// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
///     {%f0, %f1, ..., %f31}, %rd_desc_a, %rd_desc_b,
///     1, 1, 1, 0, 0;
/// ```
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
/// - `acc` must not be read or written until the region's final
///   `wgmma_wait_group::<0>()` returns
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_f16(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k16_f32_f16 called outside CUDA kernel context")
}

/// WGMMA with f32 accumulator and TF32 inputs.
///
/// This variant supports only the canonical linear full-drain region:
/// `fence -> one or more MMA -> commit_group -> wait_group<0>`. TF32 uses the
/// hardware K=8 shape; counted loops, partial waits, and non-canonical pointer
/// fallback remain unsupported.
///
/// # PTX
///
/// ```ptx
/// wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32
///     {%f0, %f1, ..., %f31}, %rd_desc_a, %rd_desc_b,
///     1, 1, 1;
/// ```
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
/// - `acc` must not be read or written until the region's final
///   `wgmma_wait_group::<0>()` returns
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k8_f32_tf32(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k8_f32_tf32 called outside CUDA kernel context")
}

/// Compatibility entry point for TF32 WGMMA.
///
/// This variant is public for source compatibility but remains unsupported by
/// the importer and lowering pipeline. PTX hardware shapes for TF32 use K=8,
/// so this legacy K=16 API also requires a separate compatibility decision.
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_tf32(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k16_f32_tf32 called outside CUDA kernel context")
}

// =============================================================================
// Accumulator Utilities
// =============================================================================

/// Type alias for the m64n64 WGMMA accumulator (32 floats per thread).
pub type Acc64x64 = [[f32; 8]; 4];

/// Type alias for the m64n128 WGMMA accumulator (64 floats per thread).
pub type Acc64x128 = [[f32; 8]; 8];

/// Initialize an accumulator to zero.
///
/// # Returns
///
/// A zeroed accumulator suitable for WGMMA operations.
#[inline(always)]
pub const fn zero_accumulator() -> Acc64x64 {
    [[0.0f32; 8]; 4]
}
