/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Core Gen 5 (tcgen05) for datacenter Blackwell architectures (sm_100a family).
//!
//! tcgen05 is Blackwell's tensor core instruction set, replacing WGMMA from Hopper.
//! The key architectural change is **single-thread MMA semantics** - one thread can
//! issue an entire matrix multiply operation.
//!
//! # Key Differences from WGMMA
//!
//! | Aspect | WGMMA (Hopper) | tcgen05 (Blackwell) |
//! |--------|---------------|---------------------|
//! | MMA issue | 128 threads collectively | **1 thread** |
//! | Matrix A/D storage | Registers/SMEM | **Tensor Memory (TMEM)** |
//! | Allocation | Implicit | **Dynamic (tcgen05.alloc)** |
//! | Wait mechanism | `wgmma.wait_group` | **mbarrier.try_wait** |
//!
//! # Tensor Memory (TMEM)
//!
//! TMEM is a new per-SM memory type for tensor operands:
//! - Dynamically allocated at runtime
//! - Allocation unit: 32 columns (range: 32-512)
//! - One WARP must perform allocation (warp-synchronous)
//! - Must be explicitly deallocated before kernel exits
//!
//! # Thread Semantics
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              Thread Requirements by Operation                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  tcgen05_mma_ws:    █         (1 thread)                            │
//! │  tcgen05_commit:    █         (1 thread)                            │
//! │  tcgen05_fence:     █         (1 thread)                            │
//! │  tcgen05_alloc:     ████████  (32 threads / 1 warp)                 │
//! │  tcgen05_dealloc:   ████████  (32 threads / 1 warp)                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::tcgen05::*;
//! use cuda_device::barrier::*;
//!
//! // 1. Allocate TMEM (warp-synchronous - all 32 threads in warp)
//! if warp_id() == 0 {
//!     tcgen05_alloc(tmem_addr_ptr, 64);  // 64 columns
//! }
//! sync_threads();
//!
//! // 2. Single thread issues MMA
//! if thread_id() == 0 {
//!     tcgen05_fence_before_thread_sync();
//!     tcgen05_mma_ws_f16(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
//!     tcgen05_commit(&mbar);
//! }
//!
//! // 3. All threads wait via mbarrier
//! mbarrier_try_wait(&mbar, 0);
//!
//! // 4. Deallocate TMEM (warp-synchronous)
//! if warp_id() == 0 {
//!     tcgen05_dealloc(tmem_addr, 64);
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_100/sm_100a**: B100, B200 (Datacenter Blackwell)
//! - **sm_101a, sm_103a, sm_110a**: B300 (Blackwell Ultra), Jetson Thor
//!   (robotics/embedded); rest of the TMEM-capable `compute_100` family
//!   (same set the tcgen05 examples gate execution on: cc 10.0 / 10.1 /
//!   10.3 / 11.0)
//!
//! Note: Consumer Blackwell (sm_120, RTX 50 series) does not implement TMEM or
//! the tcgen05 instruction set; tensor operations on sm_120 use register-based `mma.sync`.

use core::marker::PhantomData;

// =============================================================================
// Tensor Memory (TMEM) Types
// =============================================================================

/// Handle to allocated Tensor Memory.
///
/// TMEM addresses are 32-bit values returned by `tcgen05_alloc`. The address
/// is written to shared memory by the allocation instruction.
///
/// # Note
///
/// This is NOT a Rust-managed allocation. You must explicitly call
/// `tcgen05_dealloc` before the kernel exits.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TensorMemoryHandle {
    /// The 32-bit TMEM address
    pub addr: u32,
}

impl TensorMemoryHandle {
    /// Create a handle from a raw TMEM address.
    ///
    /// # Safety
    ///
    /// The address must have been returned by `tcgen05_alloc`.
    #[inline(always)]
    pub const unsafe fn from_raw(addr: u32) -> Self {
        Self { addr }
    }

    /// Get the raw TMEM address.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.addr
    }
}

// =============================================================================
// TMEM Guard with Typestate (Managed TMEM Lifecycle)
// =============================================================================

/// State marker: TMEM slot not yet allocated.
pub struct TmemUninit;

/// State marker: TMEM is allocated and ready to use.
pub struct TmemReady;

/// State marker: TMEM has been deallocated.
pub struct TmemDeallocated;

/// Newtype for TMEM addresses with type safety.
///
/// Prevents accidentally using raw u32 values as TMEM addresses.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TmemAddress(u32);

impl TmemAddress {
    /// Get the raw TMEM address value.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw value (for internal/advanced use).
    ///
    /// # Safety
    ///
    /// The value must be a valid TMEM address from `tcgen05_alloc`.
    #[inline(always)]
    pub const unsafe fn from_raw(addr: u32) -> Self {
        Self(addr)
    }
}

/// Managed TMEM allocation with typestate lifecycle.
///
/// This type provides compile-time safety for TMEM allocation/deallocation:
/// - Cannot use TMEM before allocation
/// - Cannot allocate twice
/// - Cannot use after deallocation
///
/// # Type Parameters
///
/// - `State`: Lifecycle state (`TmemUninit`, `TmemReady`, `TmemDeallocated`)
/// - `N_COLS`: Number of columns (must be power of 2: 32, 64, 128, 256, 512)
///
/// # Thread Requirements
///
/// TMEM allocation/deallocation is **warp-synchronous**:
/// - All 32 threads in the allocating warp must participate
/// - Other warps do nothing during alloc/dealloc
///
/// The `alloc()` and `dealloc()` methods handle this automatically using the
/// "all threads call, designated warp executes" pattern.
///
/// # Example
///
/// ```rust,ignore
/// // Declare shared memory slot for TMEM address storage
/// static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
///
/// // ALL threads create Uninit handle
/// let tmem = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
///
/// // ALL threads call alloc - only warp 0 actually allocates
/// let tmem = unsafe { tmem.alloc() };  // Returns TmemReady
///
/// // Get the address for use in MMA operations
/// let addr = tmem.address();
///
/// // ... use TMEM in MMA operations ...
///
/// // ALL threads call dealloc - only warp 0 actually deallocates
/// let _dead = unsafe { tmem.dealloc() };  // Returns TmemDeallocated
/// ```
pub struct TmemGuard<State, const N_COLS: u32> {
    /// Pointer to shared memory where the TMEM address is stored
    smem_ptr: *mut u32,
    _state: PhantomData<State>,
}

// Safety: Pointer is only accessed through synchronized operations
unsafe impl<S, const N: u32> Send for TmemGuard<S, N> {}

impl<const N_COLS: u32> TmemGuard<TmemUninit, N_COLS> {
    /// Create an uninitialized TMEM guard from an explicit static declaration.
    ///
    /// The shared memory slot will be used by `tcgen05_alloc` to store the
    /// allocated TMEM address.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    ///
    /// let tmem = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
    /// ```
    #[inline(always)]
    pub fn from_static(smem_ptr: *mut u32) -> Self {
        TmemGuard {
            smem_ptr,
            _state: PhantomData,
        }
    }

    /// Allocate TMEM columns.
    ///
    /// **All threads in the block should call this.** Only warp 0 performs
    /// the actual allocation; all threads synchronize and receive a `TmemReady` handle.
    ///
    /// This is a convenience wrapper for `alloc_by(0)`.
    ///
    /// # Safety
    ///
    /// - Must be called before any TMEM operations
    /// - All participating threads in the block must call this together
    /// - N_COLS must be a power of 2 in range [32, 512]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call alloc - only warp 0 actually allocates
    /// let tmem = unsafe { tmem.alloc() };
    /// ```
    #[inline(always)]
    pub unsafe fn alloc(self) -> TmemGuard<TmemReady, N_COLS> {
        unsafe { self.alloc_by(0) }
    }

    /// Allocate TMEM columns with a specific warp performing allocation.
    ///
    /// **All threads in the block should call this.** Only threads in the
    /// designated warp perform the actual allocation; all threads synchronize
    /// and receive a `TmemReady` handle.
    ///
    /// # Parameters
    ///
    /// - `alloc_warp`: Warp ID that performs allocation (all 32 threads in this warp)
    ///
    /// # Safety
    ///
    /// - Must be called before any TMEM operations
    /// - All participating threads in the block must call this together
    /// - `alloc_warp` must be a valid warp ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use warp 1 for allocation
    /// let tmem = unsafe { tmem.alloc_by(1) };
    /// ```
    #[inline(always)]
    pub unsafe fn alloc_by(self, alloc_warp: u32) -> TmemGuard<TmemReady, N_COLS> {
        let warp_id = crate::warp::warp_id();

        if warp_id == alloc_warp {
            unsafe { tcgen05_alloc(self.smem_ptr, N_COLS) };
        }

        // All threads synchronize - ensures allocation is complete
        crate::thread::sync_threads();

        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}

impl<const N_COLS: u32> TmemGuard<TmemReady, N_COLS> {
    /// Get the allocated TMEM address.
    ///
    /// This reads the address that `tcgen05_alloc` wrote to shared memory.
    ///
    /// # Note
    ///
    /// All threads in the block can call this - they all read the same value.
    #[inline(always)]
    pub fn address(&self) -> TmemAddress {
        unsafe { TmemAddress(*self.smem_ptr) }
    }

    /// Get the raw TMEM address value.
    ///
    /// Convenience method for passing to intrinsics that take raw u32.
    #[inline(always)]
    pub fn raw_address(&self) -> u32 {
        unsafe { *self.smem_ptr }
    }

    /// Get the number of allocated columns.
    #[inline(always)]
    pub const fn n_cols(&self) -> u32 {
        N_COLS
    }

    /// Deallocate the TMEM.
    ///
    /// **All threads in the block should call this.** Only warp 0 performs
    /// the actual deallocation; all threads synchronize before returning.
    ///
    /// This is a convenience wrapper for `dealloc_by(0)`.
    ///
    /// Consumes the `TmemReady` guard and returns a `TmemDeallocated` guard.
    ///
    /// # CRITICAL
    ///
    /// ALL allocated TMEM MUST be deallocated before the kernel exits.
    /// Failure to do so results in `CUDA_ERROR_TENSOR_MEMORY_LEAK`.
    ///
    /// # Safety
    ///
    /// - All threads must have completed their TMEM operations before calling
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call dealloc - only warp 0 actually deallocates
    /// let _dead = unsafe { tmem.dealloc() };
    /// ```
    #[inline(always)]
    pub unsafe fn dealloc(self) -> TmemGuard<TmemDeallocated, N_COLS> {
        unsafe { self.dealloc_by(0) }
    }

    /// Deallocate TMEM with a specific warp performing deallocation.
    ///
    /// **All threads in the block should call this.** Only threads in the
    /// designated warp perform the actual deallocation; all threads synchronize
    /// before returning.
    ///
    /// # Parameters
    ///
    /// - `dealloc_warp`: Warp ID that performs deallocation (must match alloc warp)
    ///
    /// # Safety
    ///
    /// - All threads must have completed their TMEM operations before calling
    /// - All participating threads in the block must call this together
    /// - `dealloc_warp` should typically match the warp that allocated
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use warp 1 for deallocation (should match allocation warp)
    /// let _dead = unsafe { tmem.dealloc_by(1) };
    /// ```
    #[inline(always)]
    pub unsafe fn dealloc_by(self, dealloc_warp: u32) -> TmemGuard<TmemDeallocated, N_COLS> {
        // Ensure all threads are done with TMEM before deallocating
        crate::thread::sync_threads();

        let warp_id = crate::warp::warp_id();

        if warp_id == dealloc_warp {
            unsafe {
                let tmem_addr = *self.smem_ptr;
                tcgen05_dealloc(tmem_addr, N_COLS);
            }
        }

        // All threads synchronize - ensures deallocation is complete
        crate::thread::sync_threads();

        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}

// TmemGuard<TmemDeallocated, _> has NO methods - can't use after dealloc!

// =============================================================================
// TMEM Allocation / Deallocation
// =============================================================================
// =============================================================================
// Synchronization Primitives
// =============================================================================
// NOTE: tcgen05_make_smem_desc and tcgen05_make_smem_desc_strided were removed.
// Use Tcgen05SmemDescriptor::builder() instead (see below).
// =============================================================================
// Instruction Descriptor
// =============================================================================

/// Matrix element types for tcgen05 MMA operations.
///
/// These correspond to the `atype` and `btype` fields in the instruction descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05ElementType {
    /// FP16 (IEEE half precision)
    F16 = 0,
    /// BF16 (Brain floating point)
    BF16 = 1,
    /// TF32 (TensorFloat-32) - only valid for .kind::tf32
    TF32 = 2,
}

/// Eight-bit floating-point input formats for `.kind::f8f6f4` MMA.
///
/// These values occupy the same instruction-descriptor fields as
/// [`Tcgen05ElementType`], but their meaning is selected by the MMA kind. A
/// distinct type prevents an FP8 call site from spelling E4M3 as `F16` merely
/// because both formats use encoding zero in their respective kind domains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05F8ElementType {
    /// E4M3 floating point.
    E4M3 = 0,
    /// E5M2 floating point.
    E5M2 = 1,
}

/// Six-bit floating-point input formats for `.kind::f8f6f4` MMA.
///
/// The discriminants are the hardware encodings used in the instruction
/// descriptor `atype` and `btype` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05F6ElementType {
    /// E2M3 floating point.
    E2M3 = 3,
    /// E3M2 floating point.
    E3M2 = 4,
}

/// Four-bit floating-point input format for `.kind::f8f6f4` MMA.
///
/// The discriminant is the hardware encoding used in the instruction
/// descriptor `atype` and `btype` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05F4ElementType {
    /// E2M1 floating point.
    E2M1 = 5,
}

/// Accumulator (output D) data type for tcgen05 MMA operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05AccumulatorType {
    /// FP16 accumulator (only for .kind::f16)
    F16 = 0,
    /// FP32 accumulator (default for most operations)
    F32 = 1,
}

/// MMA shape for tcgen05.mma.ws operations.
///
/// For `.ws` (warp-sync) with `cta_group::1`:
/// - **f16/bf16**: M ∈ {32, 64, 128}, N ∈ {64, 128, 256}, K = 16
/// - **tf32**: M ∈ {32, 64, 128}, N ∈ {64, 128, 256}, K = 8
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tcgen05MmaShape {
    /// M dimension (rows of A and D). Valid: 32, 64, 128
    pub m: u32,
    /// N dimension (columns of B and D). Valid: 64, 128, 256
    pub n: u32,
}

impl Tcgen05MmaShape {
    /// 32×64 MMA shape (smallest)
    pub const M32_N64: Self = Self { m: 32, n: 64 };
    /// 32×128 MMA shape
    pub const M32_N128: Self = Self { m: 32, n: 128 };
    /// 32×256 MMA shape
    pub const M32_N256: Self = Self { m: 32, n: 256 };
    /// 64×64 MMA shape
    pub const M64_N64: Self = Self { m: 64, n: 64 };
    /// 64×128 MMA shape
    pub const M64_N128: Self = Self { m: 64, n: 128 };
    /// 64×256 MMA shape
    pub const M64_N256: Self = Self { m: 64, n: 256 };
    /// 128×64 MMA shape
    pub const M128_N64: Self = Self { m: 128, n: 64 };
    /// 128×128 MMA shape
    pub const M128_N128: Self = Self { m: 128, n: 128 };
    /// 128×256 MMA shape (largest, best throughput)
    pub const M128_N256: Self = Self { m: 128, n: 256 };
    /// 256×64 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N64: Self = Self { m: 256, n: 64 };
    /// 256×128 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N128: Self = Self { m: 256, n: 128 };
    /// 256×256 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N256: Self = Self { m: 256, n: 256 };
}

/// Instruction descriptor for tcgen05 MMA operations.
///
/// The 32-bit descriptor encodes the matrix shapes, data types, sparsity mode,
/// and other operation details according to the PTX ISA specification.
///
/// # Bit Layout for .kind::tf32, .kind::f16 (from PTX ISA 8.6+)
///
/// ```text
/// Bits  | Field              | Description
/// ------+--------------------+------------------------------------------
/// 0-1   | Sparsity selector  | 0-3 (if sparsity enabled)
/// 2     | Sparsity           | 0 = Dense, 1 = Sparse
/// 3     | Reserved           | 0
/// 4-5   | dtype              | D type: F16=0, F32=1
/// 6     | Reserved           | 0
/// 7-9   | atype              | A type: F16=0, BF16=1, TF32=2
/// 10-12 | btype              | B type: F16=0, BF16=1, TF32=2
/// 13    | Negate A           | 0 = No, 1 = Yes
/// 14    | Negate B           | 0 = No, 1 = Yes
/// 15    | Transpose A        | 0 = No, 1 = Yes
/// 16    | Transpose B        | 0 = No, 1 = Yes
/// 17-22 | N dimension        | N >> 3 (N/8)
/// 23    | Reserved           | 0
/// 24-28 | M dimension        | M >> 4 (M/16)
/// 29    | Reserved           | 0
/// 30-31 | B reuse shift      | 0=none, 1=8, 2=16, 3=32
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // Create descriptor for 128×256 f16 MMA with f32 accumulator
/// let idesc = Tcgen05InstructionDescriptor::builder()
///     .shape(Tcgen05MmaShape::M128_N256)
///     .element_type(Tcgen05ElementType::F16)
///     .accumulator_type(Tcgen05AccumulatorType::F32)
///     .build();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05InstructionDescriptor {
    raw: u32,
}

impl Tcgen05InstructionDescriptor {
    /// Create a new instruction descriptor builder.
    #[inline(always)]
    pub const fn builder() -> Tcgen05InstructionDescriptorBuilder {
        Tcgen05InstructionDescriptorBuilder::new()
    }

    /// Create descriptor for f16 MMA with default settings.
    ///
    /// Default: 128×256 shape, F16 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_f16() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::F16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for bf16 MMA with default settings.
    ///
    /// Default: 128×256 shape, BF16 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_bf16() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for tf32 MMA with default settings.
    ///
    /// Default: 128×256 shape, TF32 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_tf32() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::TF32)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create a dense `.kind::f8f6f4` instruction descriptor.
    ///
    /// The caller must issue the descriptor through
    /// `tcgen05_mma_shared::<2, ..>`; descriptor format values are interpreted
    /// in the FP8 domain only when the instruction kind is `f8f6f4`.
    #[inline(always)]
    pub const fn new_f8(
        shape: Tcgen05MmaShape,
        a_type: Tcgen05F8ElementType,
        b_type: Tcgen05F8ElementType,
        accumulator_type: Tcgen05AccumulatorType,
    ) -> Self {
        Self::builder()
            .shape(shape)
            .a_type_f8(a_type)
            .b_type_f8(b_type)
            .accumulator_type(accumulator_type)
            .build()
    }

    /// Create a dense `.kind::f8f6f4` instruction descriptor with FP6 inputs.
    ///
    /// The caller must issue the descriptor through an MMA instruction using
    /// `.kind::f8f6f4` so the format codes are interpreted in the FP6 domain.
    #[inline(always)]
    pub const fn new_f6(
        shape: Tcgen05MmaShape,
        a_type: Tcgen05F6ElementType,
        b_type: Tcgen05F6ElementType,
        accumulator_type: Tcgen05AccumulatorType,
    ) -> Self {
        Self::builder()
            .shape(shape)
            .a_type_f6(a_type)
            .b_type_f6(b_type)
            .accumulator_type(accumulator_type)
            .build()
    }

    /// Create a dense `.kind::f8f6f4` instruction descriptor with FP4 inputs.
    ///
    /// The caller must issue the descriptor through an MMA instruction using
    /// `.kind::f8f6f4` so the format code is interpreted in the FP4 domain.
    #[inline(always)]
    pub const fn new_f4(
        shape: Tcgen05MmaShape,
        a_type: Tcgen05F4ElementType,
        b_type: Tcgen05F4ElementType,
        accumulator_type: Tcgen05AccumulatorType,
    ) -> Self {
        Self::builder()
            .shape(shape)
            .a_type_f4(a_type)
            .b_type_f4(b_type)
            .accumulator_type(accumulator_type)
            .build()
    }

    /// Create descriptor from raw 32-bit value.
    #[inline(always)]
    pub const fn from_raw(raw: u32) -> Self {
        Self { raw }
    }

    /// Get the raw 32-bit descriptor value.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.raw
    }
}

/// Builder for `Tcgen05InstructionDescriptor`.
///
/// Allows fluent construction of instruction descriptors with compile-time
/// validation of common configurations.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05InstructionDescriptorBuilder {
    // Bit 2: Sparsity (0 = dense)
    sparse: bool,
    // Bits 4-5: dtype
    dtype: Tcgen05AccumulatorType,
    // Bits 7-9: atype format code. The 3-bit code is stored raw because its
    // meaning depends on the MMA kind: `Tcgen05ElementType` for f16-family
    // kinds, or the F8/F6/F4 element types for `.kind::f8f6f4`.
    atype: u32,
    // Bits 10-12: btype format code (same domain rules as atype)
    btype: u32,
    // Bit 13: negate A
    negate_a: bool,
    // Bit 14: negate B
    negate_b: bool,
    // Bit 15: transpose A
    transpose_a: bool,
    // Bit 16: transpose B
    transpose_b: bool,
    // Bits 17-22: N >> 3
    n_dim: u32,
    // Bits 24-28: M >> 4
    m_dim: u32,
    // Bits 30-31: B reuse shift
    b_reuse_shift: u8,
}

impl Tcgen05InstructionDescriptorBuilder {
    /// Create a new builder with default settings.
    ///
    /// Defaults:
    /// - Shape: 128×256
    /// - Element type: F16
    /// - Accumulator: F32
    /// - Dense (not sparse)
    /// - No negation or transposition
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            sparse: false,
            dtype: Tcgen05AccumulatorType::F32,
            atype: Tcgen05ElementType::F16 as u32,
            btype: Tcgen05ElementType::F16 as u32,
            negate_a: false,
            negate_b: false,
            transpose_a: false,
            transpose_b: false,
            n_dim: 256,
            m_dim: 128,
            b_reuse_shift: 0,
        }
    }

    /// Set the MMA shape (M and N dimensions).
    ///
    /// For `.ws` operations:
    /// - M ∈ {32, 64, 128}
    /// - N ∈ {64, 128, 256}
    #[inline(always)]
    pub const fn shape(mut self, shape: Tcgen05MmaShape) -> Self {
        self.m_dim = shape.m;
        self.n_dim = shape.n;
        self
    }

    /// Set M dimension directly.
    ///
    /// Valid values: 32, 64, 128
    #[inline(always)]
    pub const fn m(mut self, m: u32) -> Self {
        self.m_dim = m;
        self
    }

    /// Set N dimension directly.
    ///
    /// Valid values: 64, 128, 256
    #[inline(always)]
    pub const fn n(mut self, n: u32) -> Self {
        self.n_dim = n;
        self
    }

    /// Set the element type for both A and B matrices.
    #[inline(always)]
    pub const fn element_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.atype = ty as u32;
        self.btype = ty as u32;
        self
    }

    /// Set element type for matrix A only.
    #[inline(always)]
    pub const fn a_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.atype = ty as u32;
        self
    }

    /// Set element type for matrix B only.
    #[inline(always)]
    pub const fn b_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.btype = ty as u32;
        self
    }

    /// Set the FP8 element type for both A and B matrices.
    ///
    /// Only meaningful for `.kind::f8f6f4` MMA; the format code shares the
    /// atype/btype fields with [`Tcgen05ElementType`].
    #[inline(always)]
    pub const fn element_type_f8(mut self, ty: Tcgen05F8ElementType) -> Self {
        self.atype = ty as u32;
        self.btype = ty as u32;
        self
    }

    /// Set FP8 element type for matrix A only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn a_type_f8(mut self, ty: Tcgen05F8ElementType) -> Self {
        self.atype = ty as u32;
        self
    }

    /// Set FP8 element type for matrix B only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn b_type_f8(mut self, ty: Tcgen05F8ElementType) -> Self {
        self.btype = ty as u32;
        self
    }

    /// Set the FP6 element type for both A and B matrices.
    ///
    /// Only meaningful for `.kind::f8f6f4` MMA.
    #[inline(always)]
    pub const fn element_type_f6(mut self, ty: Tcgen05F6ElementType) -> Self {
        self.atype = ty as u32;
        self.btype = ty as u32;
        self
    }

    /// Set FP6 element type for matrix A only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn a_type_f6(mut self, ty: Tcgen05F6ElementType) -> Self {
        self.atype = ty as u32;
        self
    }

    /// Set FP6 element type for matrix B only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn b_type_f6(mut self, ty: Tcgen05F6ElementType) -> Self {
        self.btype = ty as u32;
        self
    }

    /// Set the FP4 element type for both A and B matrices.
    ///
    /// Only meaningful for `.kind::f8f6f4` MMA.
    #[inline(always)]
    pub const fn element_type_f4(mut self, ty: Tcgen05F4ElementType) -> Self {
        self.atype = ty as u32;
        self.btype = ty as u32;
        self
    }

    /// Set FP4 element type for matrix A only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn a_type_f4(mut self, ty: Tcgen05F4ElementType) -> Self {
        self.atype = ty as u32;
        self
    }

    /// Set FP4 element type for matrix B only (`.kind::f8f6f4` MMA).
    #[inline(always)]
    pub const fn b_type_f4(mut self, ty: Tcgen05F4ElementType) -> Self {
        self.btype = ty as u32;
        self
    }

    /// Set the accumulator (output D) type.
    #[inline(always)]
    pub const fn accumulator_type(mut self, ty: Tcgen05AccumulatorType) -> Self {
        self.dtype = ty;
        self
    }

    /// Enable sparse matrix A.
    ///
    /// When enabled, K dimension is doubled (K=32 for f16, K=16 for tf32).
    #[inline(always)]
    pub const fn sparse(mut self, enable: bool) -> Self {
        self.sparse = enable;
        self
    }

    /// Enable negation of matrix A values.
    #[inline(always)]
    pub const fn negate_a(mut self, enable: bool) -> Self {
        self.negate_a = enable;
        self
    }

    /// Enable negation of matrix B values.
    #[inline(always)]
    pub const fn negate_b(mut self, enable: bool) -> Self {
        self.negate_b = enable;
        self
    }

    /// Enable transposition of matrix A.
    #[inline(always)]
    pub const fn transpose_a(mut self, enable: bool) -> Self {
        self.transpose_a = enable;
        self
    }

    /// Enable transposition of matrix B.
    #[inline(always)]
    pub const fn transpose_b(mut self, enable: bool) -> Self {
        self.transpose_b = enable;
        self
    }

    /// Set maximum B matrix reuse shift.
    ///
    /// - 0: No shift (default)
    /// - 1: Maximum shift of 8
    /// - 2: Maximum shift of 16
    /// - 3: Maximum shift of 32
    #[inline(always)]
    pub const fn b_reuse_shift(mut self, shift: u8) -> Self {
        self.b_reuse_shift = shift;
        self
    }

    /// Build the instruction descriptor.
    ///
    /// Encodes all settings into the 32-bit descriptor format.
    #[inline(always)]
    pub const fn build(self) -> Tcgen05InstructionDescriptor {
        let mut raw: u32 = 0;

        // Bits 0-1: Sparsity selector (always 0 for now)
        // raw |= 0;

        // Bit 2: Sparsity
        if self.sparse {
            raw |= 1 << 2;
        }

        // Bit 3: Reserved (0)

        // Bits 4-5: dtype (accumulator type)
        raw |= (self.dtype as u32) << 4;

        // Bit 6: Reserved (0)

        // Bits 7-9: atype (3 bits)
        raw |= (self.atype & 0x7) << 7;

        // Bits 10-12: btype (3 bits)
        raw |= (self.btype & 0x7) << 10;

        // Bit 13: Negate A
        if self.negate_a {
            raw |= 1 << 13;
        }

        // Bit 14: Negate B
        if self.negate_b {
            raw |= 1 << 14;
        }

        // Bit 15: Transpose A
        if self.transpose_a {
            raw |= 1 << 15;
        }

        // Bit 16: Transpose B
        if self.transpose_b {
            raw |= 1 << 16;
        }

        // Bits 17-22: N >> 3 (6 bits)
        raw |= ((self.n_dim >> 3) & 0x3F) << 17;

        // Bit 23: Reserved (0)

        // Bits 24-28: M >> 4 (5 bits)
        raw |= ((self.m_dim >> 4) & 0x1F) << 24;

        // Bit 29: Reserved (0)

        // Bits 30-31: B reuse shift (2 bits)
        raw |= ((self.b_reuse_shift as u32) & 0x3) << 30;

        Tcgen05InstructionDescriptor { raw }
    }
}

impl Default for Tcgen05InstructionDescriptorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SMEM Descriptor
// =============================================================================

/// Swizzle modes for tcgen05 SMEM descriptors.
///
/// Swizzling reorders data in shared memory to avoid bank conflicts during
/// tensor core operations. The mode determines the XOR pattern applied to
/// addresses.
///
/// # Encoding (bits 61-63 of SMEM descriptor)
///
/// | Mode | Value | Pattern Start |
/// |------|-------|---------------|
/// | None | 0 | - |
/// | 128B + 32B atomicity | 1 | 1024B boundary |
/// | 128B | 2 | 1024B boundary |
/// | 64B | 4 | 512B boundary |
/// | 32B | 6 | 256B boundary |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Tcgen05SwizzleMode {
    /// No swizzling
    #[default]
    None = 0,
    /// 128-byte swizzle with 32-byte atomicity (best for tensor cores)
    Swizzle128B32BAtom = 1,
    /// 128-byte swizzle (most common for tensor cores)
    Swizzle128B = 2,
    /// 64-byte swizzle
    Swizzle64B = 4,
    /// 32-byte swizzle
    Swizzle32B = 6,
}

/// SMEM descriptor for tcgen05 MMA operations (64-bit).
///
/// Describes a matrix tile in shared memory for tcgen05.mma instructions.
/// The descriptor encodes the start address, stride dimensions, and swizzle mode.
///
/// # Bit Layout (from PTX ISA)
///
/// ```text
/// Bits  | Field
/// ------+----------------------------------------------------------
/// 0-13  | (start_address >> 4) & 0x3FFF
/// 14-15 | Reserved
/// 16-29 | (leading_dim_bytes >> 4) & 0x3FFF
/// 30-31 | Reserved
/// 32-45 | (stride_bytes >> 4) & 0x3FFF
/// 46-48 | Fixed: 0b001
/// 49-51 | Base offset (for non-aligned swizzle patterns)
/// 52    | Leading dim mode: 0=relative, 1=absolute
/// 53-60 | Fixed: 0x00
/// 61-63 | Swizzle mode
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // For 128×16 f16 matrix A in K-major tiled layout. The descriptor's
/// // address bits encode the raw .shared offset, so derive it with
/// // cvta_generic_to_shared_offset rather than `ptr as u64` (which is the
/// // CUDA generic address).
/// let smem_a_addr = cvta_generic_to_shared_offset(smem_a_ptr as *const u8);
/// let desc = Tcgen05SmemDescriptor::for_k_major(
///     smem_a_addr,
///     128,  // M dimension
///     16,   // K dimension
///     2,    // f16 = 2 bytes per element
///     Tcgen05SwizzleMode::None,
/// );
///
/// // Use in MMA
/// tcgen05_mma_f16(d_tmem, desc.raw(), b_desc, idesc, true);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05SmemDescriptor {
    raw: u64,
}

impl Tcgen05SmemDescriptor {
    /// Create a new SMEM descriptor builder.
    #[inline(always)]
    pub const fn builder() -> Tcgen05SmemDescriptorBuilder {
        Tcgen05SmemDescriptorBuilder::new()
    }

    /// Build descriptor for K-major tiled layout (matrix A).
    ///
    /// K-major layout is required for the A matrix in tcgen05.mma:
    /// - Matrix divided into 8×8 tiles
    /// - Tiles arranged column-major
    /// - Within each tile: row-major (K dimension contiguous)
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (must be 16-byte aligned)
    /// - `m`: M dimension (rows of the matrix)
    /// - `k`: K dimension (columns of the matrix)
    /// - `elem_bytes`: Element size in bytes (2 for f16/bf16, 4 for tf32/f32)
    /// - `swizzle`: Swizzle mode for bank conflict avoidance
    ///
    /// # Layout Computation
    ///
    /// For an M×K matrix with 8×8 tiles:
    /// - **SBO** (stride byte offset) = 8 × 8 × elem_bytes = 64 × elem_bytes
    /// - **LBO** (leading dim byte offset) = (M/8) × 8 × 8 × elem_bytes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 128×16 f16 matrix (smem_addr from cvta_generic_to_shared_offset)
    /// let desc = Tcgen05SmemDescriptor::for_k_major(
    ///     smem_addr, 128, 16, 2, Tcgen05SwizzleMode::None
    /// );
    /// // SBO = 8*8*2 = 128 bytes
    /// // LBO = (128/8)*8*8*2 = 16*64*2 = 2048 bytes
    /// ```
    #[inline(always)]
    pub const fn for_k_major(
        smem_addr: u64,
        m: usize,
        k: usize,
        elem_bytes: usize,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        let _ = k; // K determines number of tiles horizontally, but not SBO/LBO directly

        // SBO = one 8×8 tile size in bytes
        let sbo = (8 * 8 * elem_bytes) as u32;

        // LBO = column of tiles = (M/8) tiles × 64 elements × elem_bytes
        let lbo = ((m / 8) * 8 * 8 * elem_bytes) as u32;

        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(lbo)
            .stride_bytes(sbo)
            .swizzle(swizzle)
            .build()
    }

    /// Build descriptor for MN-major tiled layout (matrix B).
    ///
    /// MN-major layout is required for the B matrix in tcgen05.mma:
    /// - Matrix divided into 8×8 tiles
    /// - Tiles arranged column-major
    /// - Within each tile: column-major (MN dimension contiguous)
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (must be 16-byte aligned)
    /// - `n`: N dimension (rows of B, which is K×N transposed view)
    /// - `k`: K dimension
    /// - `elem_bytes`: Element size in bytes
    /// - `swizzle`: Swizzle mode
    #[inline(always)]
    pub const fn for_mn_major(
        smem_addr: u64,
        n: usize,
        k: usize,
        elem_bytes: usize,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        let _ = k;

        let sbo = (8 * 8 * elem_bytes) as u32;
        let lbo = ((n / 8) * 8 * 8 * elem_bytes) as u32;

        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(lbo)
            .stride_bytes(sbo)
            .swizzle(swizzle)
            .build()
    }

    /// Build descriptor from explicit byte offsets.
    ///
    /// Use this when you need full control over the descriptor parameters.
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (16-byte aligned)
    /// - `leading_dim_bytes`: Leading dimension stride in bytes (LBO)
    /// - `stride_bytes`: Stride dimension offset in bytes (SBO)
    /// - `swizzle`: Swizzle mode
    #[inline(always)]
    pub const fn from_bytes(
        smem_addr: u64,
        leading_dim_bytes: u32,
        stride_bytes: u32,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(leading_dim_bytes)
            .stride_bytes(stride_bytes)
            .swizzle(swizzle)
            .build()
    }

    /// Create descriptor from raw 64-bit value.
    #[inline(always)]
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Get the raw 64-bit descriptor value.
    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.raw
    }
}

/// Builder for `Tcgen05SmemDescriptor`.
///
/// Provides fluent API for constructing SMEM descriptors with compile-time
/// validation of common configurations.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05SmemDescriptorBuilder {
    address: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    base_offset: u8,
    leading_dim_absolute: bool,
    swizzle: Tcgen05SwizzleMode,
}

impl Tcgen05SmemDescriptorBuilder {
    /// Create a new builder with default settings.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            address: 0,
            leading_dim_bytes: 0,
            stride_bytes: 0,
            base_offset: 0,
            leading_dim_absolute: false,
            swizzle: Tcgen05SwizzleMode::None,
        }
    }

    /// Set the shared memory start address.
    ///
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn address(mut self, addr: u64) -> Self {
        self.address = addr;
        self
    }

    /// Set the leading dimension byte offset (LBO).
    ///
    /// This is the stride to move to the next "column" of tiles.
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn leading_dim_bytes(mut self, lbo: u32) -> Self {
        self.leading_dim_bytes = lbo;
        self
    }

    /// Set the stride dimension byte offset (SBO).
    ///
    /// This is typically the size of one tile (8×8 × element_size).
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn stride_bytes(mut self, sbo: u32) -> Self {
        self.stride_bytes = sbo;
        self
    }

    /// Set the swizzle mode.
    #[inline(always)]
    pub const fn swizzle(mut self, mode: Tcgen05SwizzleMode) -> Self {
        self.swizzle = mode;
        self
    }

    /// Set the base offset for non-aligned swizzle patterns.
    ///
    /// Only needed when the swizzle pattern doesn't start at a natural boundary.
    /// Formula: `base_offset = (pattern_start_addr >> 7) & 0x7`
    #[inline(always)]
    pub const fn base_offset(mut self, offset: u8) -> Self {
        self.base_offset = offset;
        self
    }

    /// Use absolute addressing for leading dimension.
    ///
    /// When true, leading_dim_bytes is an absolute address rather than
    /// a relative offset. Only supported on sm_103a+.
    #[inline(always)]
    pub const fn leading_dim_absolute(mut self, absolute: bool) -> Self {
        self.leading_dim_absolute = absolute;
        self
    }

    /// Build the SMEM descriptor.
    ///
    /// Encodes all settings into the 64-bit descriptor format.
    #[inline(always)]
    pub const fn build(self) -> Tcgen05SmemDescriptor {
        let mut raw: u64 = 0;

        // Bits 0-13: (address >> 4) & 0x3FFF
        raw |= (self.address >> 4) & 0x3FFF;

        // Bits 16-29: (leading_dim_bytes >> 4) & 0x3FFF
        raw |= (((self.leading_dim_bytes >> 4) & 0x3FFF) as u64) << 16;

        // Bits 32-45: (stride_bytes >> 4) & 0x3FFF
        raw |= (((self.stride_bytes >> 4) & 0x3FFF) as u64) << 32;

        // Bits 46-48: Fixed = 0b001
        raw |= 1u64 << 46;

        // Bits 49-51: Base offset (3 bits)
        raw |= ((self.base_offset & 0x7) as u64) << 49;

        // Bit 52: Leading dim mode (0 = relative, 1 = absolute)
        if self.leading_dim_absolute {
            raw |= 1u64 << 52;
        }

        // Bits 53-60: Fixed = 0 (already zero)

        // Bits 61-63: Swizzle mode (3 bits)
        raw |= (self.swizzle as u64) << 61;

        Tcgen05SmemDescriptor { raw }
    }
}

impl Default for Tcgen05SmemDescriptorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Collector Buffer Usage
// =============================================================================

/// Collector buffer usage hints for matrix B caching.
///
/// Collector buffers are internal to the Tensor Core (not addressable).
/// They cache matrix B data to avoid repeated shared memory reads when
/// using the same B matrix with different A matrices.
///
/// # Example
///
/// ```rust,ignore
/// // First MMA: Load B into collector b0
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Fill(0));
///
/// // Subsequent MMAs: Reuse B from collector
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Use(0));
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Use(0));
///
/// // Last use: Discard after using
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::LastUse(0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectorUsage {
    /// Read B from SMEM and cache in buffer N
    Fill(u8),
    /// Use B from buffer N (must have been filled)
    Use(u8),
    /// Use B from buffer N, then discard
    LastUse(u8),
    /// Discard buffer N without using
    Discard(u8),
}

// =============================================================================
// tcgen05 MMA Instructions
// =============================================================================
/// tcgen05 MMA with f16 inputs and collector buffer hint.
///
/// Use collector buffers to cache matrix B when reusing the same B matrix
/// across multiple MMA operations.
///
/// # Safety
///
/// - Tensor-memory addresses, `b_desc`, and `idesc` must be valid
/// - Must be called from a CUDA kernel on a target supported by this intrinsic
/// - The collector buffer must be in `0..=3`
///
/// `a_desc` is kept for compatibility; tensor A is read from `a_tmem`.
#[inline(always)]
pub unsafe fn tcgen05_mma_ws_f16_with_collector(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
    collector: CollectorUsage,
) {
    let _ = a_desc;
    macro_rules! dispatch {
        ($buffer:expr, $usage:expr) => {
            match $buffer {
                0 => unsafe {
                    tcgen05_mma_ws_tensor::<0, 0, $usage>(d_tmem, a_tmem, b_desc, idesc, enable_d)
                },
                1 => unsafe {
                    tcgen05_mma_ws_tensor::<0, 1, $usage>(d_tmem, a_tmem, b_desc, idesc, enable_d)
                },
                2 => unsafe {
                    tcgen05_mma_ws_tensor::<0, 2, $usage>(d_tmem, a_tmem, b_desc, idesc, enable_d)
                },
                3 => unsafe {
                    tcgen05_mma_ws_tensor::<0, 3, $usage>(d_tmem, a_tmem, b_desc, idesc, enable_d)
                },
                _ => panic!("collector buffer must be in 0..=3"),
            }
        };
    }
    match collector {
        CollectorUsage::Discard(buffer) => dispatch!(buffer, 0),
        CollectorUsage::LastUse(buffer) => dispatch!(buffer, 1),
        CollectorUsage::Fill(buffer) => dispatch!(buffer, 2),
        CollectorUsage::Use(buffer) => dispatch!(buffer, 3),
    }
}

// =============================================================================
// TMEM Load/Store (for results)
// =============================================================================
// NOTE: The following deprecated intrinsics were removed:
// - tcgen05_st_tmem_to_smem (wrong approach - no direct TMEM→SMEM instruction)
// - tcgen05_st_tmem_to_smem_offset (wrong approach)
// - tcgen05_ld_16x256b_x4/x8/x16/x32 (wrong - stored to SMEM instead of returning registers)
// - tcgen05_ld_32x32b_x64 (wrong - stored to SMEM instead of returning registers)
//
// Use tcgen05_ld_16x256b_pure or tcgen05_ld_16x256b_x8_pure instead.
// These return values in registers, allowing proper epilogue processing:
//   1. Load from TMEM → registers
//   2. Convert f32 → bf16/f16 in registers
//   3. Store to SMEM via stmatrix

// =============================================================================
// Pure TMEM Load (Returns Registers - No SMEM Store)
// =============================================================================

use crate::cusimd::CuSimd;

/// Result of `tcgen05_ld_16x256b_x8_pure` - 32 f32 values per thread.
///
/// This is a type alias for `CuSimd<f32, 32>` which provides:
/// - Array-style indexing: `regs[i]` for runtime access
/// - Const generic access: `regs.get::<0>()` for compile-time access
/// - Shorthand accessors: `regs.x()`, `regs.y()`, etc.
///
/// # Usage
///
/// ```rust,ignore
/// let regs: TmemF32x32 = tcgen05_ld_16x256b_x8_pure(tmem_addr);
/// tcgen05_load_wait();
///
/// // Access via index
/// let val0 = regs[0];
/// let val1 = regs[1];
///
/// // Convert and pack for stmatrix
/// let p0 = cuda_device::convert::cvt_bf16x2_f32(regs[0], regs[1]);
/// let p1 = cuda_device::convert::cvt_bf16x2_f32(regs[2], regs[3]);
/// // ...
/// stmatrix_m8n8_x4_trans(smem_ptr, p0, p1, p2, p3);
/// ```
pub type TmemF32x32 = CuSimd<f32, 32>;
// =============================================================================
// Base LDTM (16dp256bit without .x multiplier)
// =============================================================================

/// Result of `tcgen05_ld_16x256b_pure` (base LDTM) - 4 f32 values per thread.
///
/// This is a type alias for `CuSimd<f32, 4>` which provides:
/// - Array-style indexing: `regs[i]` for runtime access
/// - Const generic access: `regs.get::<0>()` for compile-time access
/// - Shorthand accessors: `regs.x()`, `regs.y()`, `regs.z()`, `regs.w()`
///
/// # Thread Data Layout
///
/// For base LDTM.16dp256bit with 32dp (data parallelism = 32 threads):
/// - Each thread receives 4 f32 values = 16 bytes = 128 bits
/// - Total per warp: 32 threads × 4 values = 128 f32 values
/// - This corresponds to an 8×16 matrix tile (8 rows × 16 columns)
///
/// # Relationship to stmatrix.x2
///
/// The data layout from base LDTM matches perfectly with `stmatrix.m8n8.x2`:
/// - 128 bf16 (after conversion) = 2 matrices × 8×8 = 128 elements ✓
/// - Thread ownership aligns: each thread owns data for specific positions
///
/// # Usage
///
/// ```rust,ignore
/// let regs: TmemF32x4 = tcgen05_ld_16x256b_pure(tmem_addr);
/// tcgen05_load_wait();
///
/// // Access via index or shorthand
/// let p0 = cuda_device::convert::cvt_bf16x2_f32(regs[0], regs[1]);
/// let p1 = cuda_device::convert::cvt_bf16x2_f32(regs[2], regs[3]);
/// // Or: cuda_device::convert::cvt_bf16x2_f32(regs.x(), regs.y())
///
/// // Store via stmatrix.x2 (non-trans)
/// stmatrix_m8n8_x2(smem_ptr, p0, p1);
/// ```
pub type TmemF32x4 = CuSimd<f32, 4>;
// =============================================================================
// TMEM Load/Store Synchronization
// =============================================================================
// =============================================================================
// Matrix Store Instructions
// =============================================================================
include!("generated/stmatrix.rs");

// Generated tcgen05 leaf intrinsics.
include!("generated/tcgen05.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_m128_n128_f32_descriptor_matches_sm100_layout() {
        let descriptor = Tcgen05InstructionDescriptor::new_f8(
            Tcgen05MmaShape::M128_N128,
            Tcgen05F8ElementType::E4M3,
            Tcgen05F8ElementType::E4M3,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!(descriptor.raw(), 0x0820_0010);

        // new_f8 delegates to the builder; the F8 setters must land on the
        // same encoding.
        let via_builder = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type_f8(Tcgen05F8ElementType::E4M3)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();
        assert_eq!(via_builder.raw(), descriptor.raw());
    }

    #[test]
    fn fp8_a_and_b_formats_have_independent_fields() {
        let e5m2_e4m3 = Tcgen05InstructionDescriptor::new_f8(
            Tcgen05MmaShape::M64_N64,
            Tcgen05F8ElementType::E5M2,
            Tcgen05F8ElementType::E4M3,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!((e5m2_e4m3.raw() >> 7) & 0x7, 1);
        assert_eq!((e5m2_e4m3.raw() >> 10) & 0x7, 0);

        let e4m3_e5m2 = Tcgen05InstructionDescriptor::new_f8(
            Tcgen05MmaShape::M64_N64,
            Tcgen05F8ElementType::E4M3,
            Tcgen05F8ElementType::E5M2,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!((e4m3_e5m2.raw() >> 7) & 0x7, 0);
        assert_eq!((e4m3_e5m2.raw() >> 10) & 0x7, 1);
    }

    #[test]
    fn fp6_fp4_descriptors_match_sm100_layout() {
        let e2m3 = Tcgen05InstructionDescriptor::new_f6(
            Tcgen05MmaShape::M128_N128,
            Tcgen05F6ElementType::E2M3,
            Tcgen05F6ElementType::E2M3,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!(e2m3.raw(), 0x0820_0d90);
        let e2m3_via_builder = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type_f6(Tcgen05F6ElementType::E2M3)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();
        assert_eq!(e2m3_via_builder.raw(), e2m3.raw());

        let e3m2 = Tcgen05InstructionDescriptor::new_f6(
            Tcgen05MmaShape::M128_N128,
            Tcgen05F6ElementType::E3M2,
            Tcgen05F6ElementType::E3M2,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!(e3m2.raw(), 0x0820_1210);

        let e2m1 = Tcgen05InstructionDescriptor::new_f4(
            Tcgen05MmaShape::M128_N128,
            Tcgen05F4ElementType::E2M1,
            Tcgen05F4ElementType::E2M1,
            Tcgen05AccumulatorType::F32,
        );
        assert_eq!(e2m1.raw(), 0x0820_1690);
        let e2m1_via_builder = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)
            .element_type_f4(Tcgen05F4ElementType::E2M1)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();
        assert_eq!(e2m1_via_builder.raw(), e2m1.raw());
    }

    #[test]
    fn f8f6f4_builder_supports_mixed_narrow_formats() {
        let descriptor = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M64_N64)
            .a_type_f8(Tcgen05F8ElementType::E4M3)
            .b_type_f6(Tcgen05F6ElementType::E3M2)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();
        assert_eq!((descriptor.raw() >> 7) & 0x7, 0);
        assert_eq!((descriptor.raw() >> 10) & 0x7, 4);

        let descriptor = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M64_N64)
            .a_type_f6(Tcgen05F6ElementType::E3M2)
            .b_type_f4(Tcgen05F4ElementType::E2M1)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build();
        assert_eq!((descriptor.raw() >> 7) & 0x7, 4);
        assert_eq!((descriptor.raw() >> 10) & 0x7, 5);
    }
}
