/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations.
//!
//! This module provides register-only matrix operations (`movmatrix` and
//! `mma.sync`) plus warp-cooperative shared-memory loads (`ldmatrix`).
//!
//! For ldmatrix, each group of four lanes loads one naturally aligned
//! 16-byte row:
//!
//! ```text
//! x1: lanes  0..7  provide addresses
//! x2: lanes  0..15 provide addresses
//! x4: lanes  0..31 provide addresses
//! ```
//!
//! On sm_75, x1 and x2 still require valid addresses in all 32 lanes. A
//! common choice is to copy the lower-lane addresses into the upper lanes.
//! The .trans forms use column-major rather than row-major layout.
//!
//! Ldmatrix is a weak memory operation: .sync converges the warp but does not
//! order memory. Callers need an appropriate barrier or fence around dependent
//! memory accesses. Movmatrix and MMA are register-only and have no memory
//! effect.

include!("generated/movmatrix.rs");

// =============================================================================
// Shared-memory matrix loads
// =============================================================================

/// Load one 8×8 matrix tile from shared memory.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// - Lanes 0-7 must each provide a valid, naturally aligned 16-byte shared-memory row
/// - On sm_75, all 32 lanes must provide a valid address
/// - Must be called by all threads in a warp (warp-synchronous)
/// - Callers must use a suitable barrier or fence to order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x1(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1 called outside CUDA kernel context")
}

/// Load one 8×8 matrix tile from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x1`].
#[inline(never)]
pub unsafe fn ldmatrix_x1_trans(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1_trans called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 2]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// - Lanes 0-15 provide the 16 row addresses
/// - On sm_75, all 32 lanes must provide a valid address
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x2(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2 called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x2`].
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2_trans called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 4]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// - All 32 lanes provide valid, naturally aligned 16-byte row addresses
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x4(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4 called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x4`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_trans called outside CUDA kernel context")
}

/// Load one 8×8 matrix from a 32-bit `.shared::cta` state-space address.
///
/// This has the same participation, alignment, and ordering requirements as
/// [`ldmatrix_x1`]. Obtain `shared_addr` with
/// [`crate::shared::cvta_generic_to_shared_u32`] and keep subsequent byte
/// arithmetic in `u32`.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x1`].
#[inline(never)]
pub unsafe fn ldmatrix_x1_shared_u32(shared_addr: u32) -> u32 {
    let _ = shared_addr;
    unreachable!("ldmatrix_x1_shared_u32 called outside CUDA kernel context")
}

/// Load one transposed 8×8 matrix from a 32-bit `.shared::cta` state-space address.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x1_trans`].
#[inline(never)]
pub unsafe fn ldmatrix_x1_trans_shared_u32(shared_addr: u32) -> u32 {
    let _ = shared_addr;
    unreachable!("ldmatrix_x1_trans_shared_u32 called outside CUDA kernel context")
}

/// Load two 8×8 matrices from a 32-bit `.shared::cta` state-space address.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x2`].
#[inline(never)]
pub unsafe fn ldmatrix_x2_shared_u32(shared_addr: u32) -> [u32; 2] {
    let _ = shared_addr;
    unreachable!("ldmatrix_x2_shared_u32 called outside CUDA kernel context")
}

/// Load two transposed 8×8 matrices from a 32-bit `.shared::cta` state-space address.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x2_trans`].
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans_shared_u32(shared_addr: u32) -> [u32; 2] {
    let _ = shared_addr;
    unreachable!("ldmatrix_x2_trans_shared_u32 called outside CUDA kernel context")
}

/// Load four 8×8 matrices from a 32-bit `.shared::cta` state-space address.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x4`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_shared_u32(shared_addr: u32) -> [u32; 4] {
    let _ = shared_addr;
    unreachable!("ldmatrix_x4_shared_u32 called outside CUDA kernel context")
}

/// Load four transposed 8×8 matrices from a 32-bit `.shared::cta` state-space address.
///
/// # Safety
///
/// Same safety contract as [`ldmatrix_x4_trans`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans_shared_u32(shared_addr: u32) -> [u32; 4] {
    let _ = shared_addr;
    unreachable!("ldmatrix_x4_trans_shared_u32 called outside CUDA kernel context")
}

/// Multiply one warp-distributed BF16 tile and add an f32 accumulator.
///
/// Together, the 32 lanes compute `D = A × B + C` for row-major `A` with
/// shape 16×16, column-major `B` with shape 16×8, and `C`/`D` with shape
/// 16×8. Each lane supplies its fragments in registers and receives four f32
/// result registers. The call itself does not access memory or act as a fence.
///
/// `a[j / 2]` and `b[j / 2]` pack logical element `j` in low-to-high 16-bit
/// order. For lane `lane`, let `group = lane / 4` and `thread = lane % 4`:
///
/// ```text
/// A element j=0..7:
///   row = group       for j in {0,1,4,5}; otherwise group + 8
///   col = thread*2 + (j&1) + (if j >= 4 { 8 } else { 0 })
///
/// B element j=0..3:
///   row = thread*2 + (j&1) + (if j >= 2 { 8 } else { 0 })
///   col = group
///
/// C/D register j=0..3:
///   row = group + (if j >= 2 { 8 } else { 0 })
///   col = thread*2 + (j&1)
/// ```
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - All 32 lanes must execute the same call with the same qualifiers. Calling
///   from divergent control flow, or after any lane has exited, is undefined
///   behavior.
/// - `c`, `a`, and `b` must contain the calling lane's fragments in the layout
///   above. A different layout computes a different matrix operation.
/// - Requires `sm_80+` and PTX ISA 7.0+. cuda-oxide selects both floors
///   automatically and rejects an explicit lower target.
#[inline(never)]
#[must_use]
pub unsafe fn mma_m16n8k16_f32_bf16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    let _ = (c, a, b);
    unreachable!("mma_m16n8k16_f32_bf16 called outside CUDA kernel context")
}

/// Multiply one warp-distributed F16 tile and add an f32 accumulator.
///
/// Together, the 32 lanes compute `D = A × B + C` for row-major `A` with
/// shape 16×16, column-major `B` with shape 16×8, and `C`/`D` with shape
/// 16×8. Each lane supplies its fragments in registers and receives four f32
/// result registers. The call itself does not access memory or act as a fence.
///
/// The Rust arrays use the same order as the PTX register lists below:
///
/// - `c[0..4]` contains `%c0..%c3`, one `.f32` accumulator per element. The
///   returned array contains `%d0..%d3` in the same order.
/// - `a[0..4]` contains `%a0..%a3`, and `b[0..2]` contains `%b0..%b1`.
///   Each `u32` is a raw `.b32` carrier holding two `.f16` values; element
///   `j` is in `a[j / 2]` or `b[j / 2]`, low 16 bits before high 16 bits.
///
/// For lane `lane`, let `group = lane / 4` and `thread = lane % 4`:
///
/// ```text
/// A element j=0..7:
///   row = group       for j in {0,1,4,5}; otherwise group + 8
///   col = thread*2 + (j&1) + (if j >= 4 { 8 } else { 0 })
///
/// B element j=0..3:
///   row = thread*2 + (j&1) + (if j >= 2 { 8 } else { 0 })
///   col = group
///
/// C/D register j=0..3:
///   row = group + (if j >= 2 { 8 } else { 0 })
///   col = thread*2 + (j&1)
/// ```
///
/// These coordinates are the `.row.col` layout: A is row-major and B is
/// column-major.
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - All 32 lanes must execute the same `mma.sync.aligned` instruction with
///   the same qualifiers. Calling from divergent control flow, or after any
///   lane has exited, is undefined behavior.
/// - `c`, `a`, and `b` must contain the calling lane's fragments in the layout
///   above. A different array order or layout computes a different matrix
///   operation.
/// - Requires `sm_80+` and PTX ISA 7.0+. cuda-oxide selects both floors
///   automatically and rejects an explicit lower target.
#[inline(never)]
#[must_use]
pub unsafe fn mma_m16n8k16_f32_f16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    let _ = (c, a, b);
    unreachable!("mma_m16n8k16_f32_f16 called outside CUDA kernel context")
}

/// Multiply one warp-distributed TF32 tile and add an f32 accumulator.
///
/// Together, the 32 lanes compute `D = A × B + C` for row-major `A` with
/// shape 16×8, column-major `B` with shape 8×8, and `C`/`D` with shape
/// 16×8. Each lane supplies its fragments in registers and receives four f32
/// result registers. The call itself does not access memory or act as a fence.
///
/// The Rust arrays use the same order as the PTX register lists below:
///
/// - `c[0..4]` contains `%c0..%c3`, one `.f32` accumulator per element. The
///   returned array contains `%d0..%d3` in the same order.
/// - `a[0..4]` contains `%a0..%a3`, and `b[0..2]` contains `%b0..%b1`.
///   Each `u32` is a raw `.b32` register containing one TF32 value.
///
/// For lane `lane`, let `group = lane / 4` and `thread = lane % 4`:
///
/// ```text
/// A register j=0..3:
///   row = group + (if j in {1,3} { 8 } else { 0 })
///   col = thread + (if j >= 2 { 4 } else { 0 })
///
/// B register j=0..1:
///   row = thread + (if j == 1 { 4 } else { 0 })
///   col = group
///
/// C/D register j=0..3:
///   row = group + (if j >= 2 { 8 } else { 0 })
///   col = thread*2 + (j&1)
/// ```
///
/// These coordinates are the `.row.col` layout: A is row-major and B is
/// column-major. The raw A and B registers must already contain valid TF32
/// values produced by `cvt.rna.tf32.f32` or an equivalent TF32-producing
/// operation. Passing arbitrary `f32::to_bits()` values is not a conversion to
/// TF32 and does not satisfy this contract.
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - All 32 lanes must execute the same `mma.sync.aligned` instruction with
///   the same qualifiers. Calling from divergent control flow, or after any
///   lane has exited, is undefined behavior.
/// - `c`, `a`, and `b` must contain the calling lane's fragments in the array
///   order and layout above. A different order computes a different matrix
///   operation; invalid TF32 bit patterns do not become valid merely because
///   they are carried in `u32`.
/// - Requires `sm_80+` and PTX ISA 7.0+. cuda-oxide selects both floors
///   automatically and rejects an explicit lower target.
#[inline(never)]
#[must_use]
pub unsafe fn mma_m16n8k8_f32_tf32(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    let _ = (c, a, b);
    unreachable!("mma_m16n8k8_f32_tf32 called outside CUDA kernel context")
}

/// Warp MMA: D = A x B + C (m8n8k4, f64 output, f64 inputs).
///
/// Performs an 8x8x4 double-precision matrix multiplication using tensor cores.
/// All 32 threads in the warp participate.
///
/// # Matrix Dimensions
///
/// - **A**: 8x4 (row-major, f64), distributed as 1 x f64 per thread
/// - **B**: 4x8 (col-major, f64), distributed as 1 x f64 per thread
/// - **D/C**: 8x8 (f64 accumulator), distributed as 2 x f64 per thread
///
/// Each lane owns these matrix elements:
///
/// ```text
/// A:   row = lane / 4, column = lane % 4
/// B:   row = lane % 4, column = lane / 4
/// C/D: row = lane / 4, columns = 2 * (lane % 4) + {0, 1}
/// ```
///
/// `acc` contains the lane's two C-fragment registers. The return value is
/// the lane's two D-fragment registers. `a` and `b` are the lane's single
/// multiplicand registers.
///
/// The multiply-add uses round-to-nearest-even (`.rn`) rounding.
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
///     {%d0, %d1},
///     {%a0},
///     {%b0},
///     {%c0, %c1};
/// ```
///
/// # Safety
///
/// - All 32 lanes in the warp must execute the same call together.
/// - Calling from divergent control flow is undefined behavior.
/// - No participating lane may have exited before the call.
/// - Requires `sm_80+` and PTX ISA 7.0+. cuda-oxide selects both floors
///   automatically.
#[inline(never)]
#[must_use]
pub unsafe fn mma_m8n8k4_f64(acc: [f64; 2], a: f64, b: f64) -> [f64; 2] {
    let _ = (acc, a, b);
    unreachable!("mma_m8n8k4_f64 called outside CUDA kernel context")
}

/// Multiply one warp-distributed INT8 tile and add an i32 accumulator.
///
/// Together, the 32 lanes compute `D = A × B + C` for row-major `A` with
/// shape 16×32, column-major `B` with shape 32×8, and `C`/`D` with shape
/// 16×8. Each lane supplies its fragments in registers and receives four i32
/// result registers. The call itself does not access memory or act as a fence.
///
/// Each `a` and `b` register packs four two's-complement `s8` values from low
/// byte to high byte. In other words, logical element `i` is stored in bits
/// `(i % 4) * 8..(i % 4 + 1) * 8` of register `[i / 4]`.
///
/// For lane `lane`, let `group = lane / 4` and `thread = lane % 4`:
///
/// ```text
/// A element i=0,...,15:
///   row = group       for 0 <= i < 4 or 8 <= i < 12; otherwise group + 8
///   col = thread*4 + (i&3) + (if i >= 8 { 16 } else { 0 })
///
/// B element i=0,...,7:
///   row = thread*4 + (i&3) + (if i >= 4 { 16 } else { 0 })
///   col = group
///
/// C/D register i=0,...,3:
///   row = group + (if i >= 2 { 8 } else { 0 })
///   col = thread*2 + (i&1)
/// ```
///
/// This is the non-`.satfinite` form. Signed accumulator overflow wraps rather
/// than clamping to `i32::MIN` or `i32::MAX`.
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - All 32 lanes must execute the same instruction with the same qualifiers.
///   Calling from divergent control flow, or after any lane has exited, is
///   undefined behavior.
/// - `c`, `a`, and `b` must contain the calling lane's fragments in the layout
///   above. A different layout computes a different matrix operation.
/// - Requires `sm_80+` and PTX ISA 7.0+. cuda-oxide selects both floors
///   automatically and rejects an explicit lower target.
#[inline(never)]
#[must_use]
pub unsafe fn mma_m16n8k32_s32_s8(c: [i32; 4], a: [u32; 4], b: [u32; 2]) -> [i32; 4] {
    let _ = (c, a, b);
    unreachable!("mma_m16n8k32_s32_s8 called outside CUDA kernel context")
}

include!("generated/register_mma.rs");
include!("generated/sparse_mma.rs");
include!("generated/ldmatrix.rs");
