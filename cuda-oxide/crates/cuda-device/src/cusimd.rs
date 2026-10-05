/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuSimd<T, N>: Generic SIMD type for multi-register GPU values
//!
//! This module provides a type-safe abstraction for operations that produce
//! or consume multiple register values, such as:
//! - `tcgen05.ld` operations returning 4-32 f32 values
//! - Packed bf16×2 pairs
//! - Vector load/store operations
//!
//! # Design Philosophy
//!
//! GPU registers are values, not memory locations. `CuSimd<T, N>` represents
//! N values of type T that travel together through registers. All access
//! methods use **copy semantics** - we copy values out, not reference them.
//!
//! For the *memory* counterpart - over-aligned element types whose alignment
//! makes loads and stores single wide transactions - see [`crate::vector`]
//! (e.g. [`crate::vector::F32x4`]). For packed 16-bit ALU pairs, see
//! [`crate::f16x2`] / [`crate::bf16x2`].
//!
//! # Example
//!
//! ```rust,ignore
//! // Construct from array
//! let regs = CuSimd::<f32, 4>::new([1.0, 2.0, 3.0, 4.0]);
//!
//! // Access via index operator (copy semantics)
//! let val = regs[2];  // 3.0
//!
//! // Shorthand accessors for common sizes
//! let x = regs.x();  // 1.0
//! let w = regs.w();  // 4.0
//!
//! // Compile-time indexed access
//! let first = regs.get::<0>();
//! ```

use core::ops::Index;

// =============================================================================
// SimdElement Trait
// =============================================================================

/// Marker trait for types that can be elements of `CuSimd`.
///
/// All valid SIMD element types must be `Copy` (GPU register values are copied,
/// not referenced) and `Sized`.
pub trait SimdElement: Copy + Sized {
    /// Maximum supported N for this element type.
    ///
    /// This reflects hardware constraints:
    /// - `f32`: up to 32 (tcgen05.ld.x8 returns 32 f32 per thread)
    /// - `f16`/`bf16`: typically 2 (packed pairs)
    /// - `u32`/`i32`: up to 128 (tcgen05 register tiles)
    const MAX_N: usize;
}

impl SimdElement for f32 {
    const MAX_N: usize = 32; // tcgen05.ld.x8 returns 32 f32
}

impl SimdElement for f64 {
    const MAX_N: usize = 16;
}

impl SimdElement for u32 {
    const MAX_N: usize = 128;
}

impl SimdElement for i32 {
    const MAX_N: usize = 128;
}

impl SimdElement for u64 {
    const MAX_N: usize = 16;
}

impl SimdElement for i64 {
    const MAX_N: usize = 16;
}

impl SimdElement for u16 {
    const MAX_N: usize = 32;
}

impl SimdElement for i16 {
    const MAX_N: usize = 32;
}

// =============================================================================
// CuSimd<T, N> Core Type
// =============================================================================

/// CUDA SIMD type - N values of type T that travel together through registers.
///
/// # Memory Layout
///
/// Values are stored in a contiguous array. For most types, this maps to
/// N separate GPU registers. The exception is packed 16-bit types where
/// two values share a single 32-bit register.
///
/// # Copy Semantics
///
/// All access methods return values by **copy**, not reference. This matches
/// the semantic model of GPU registers - they're values, not addressable
/// memory locations.
///
/// # Valid Configurations
///
/// | Element Type | Common N values | Hardware Representation |
/// |--------------|-----------------|-------------------------|
/// | `f32`        | 2, 4, 8, 16, 32 | N × 32-bit registers    |
/// | `u32`/`i32`  | 2 to 128        | N × 32-bit registers    |
/// | `f64`        | 2, 4            | N × 64-bit registers    |
///
/// # Example
///
/// ```rust,ignore
/// // From tcgen05.ld (future API)
/// let regs: CuSimd<f32, 4> = tcgen05_ld_16x256b_pure(addr);
///
/// // Access individual values
/// let val = regs[2];           // Runtime index
/// let first = regs.get::<0>(); // Compile-time index
///
/// // Shorthand for common patterns
/// let (a, b) = (regs.x(), regs.y());
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CuSimd<T: SimdElement, const N: usize> {
    data: [T; N],
}

// =============================================================================
// Core Implementation
// =============================================================================

impl<T: SimdElement, const N: usize> CuSimd<T, N> {
    /// Create a new `CuSimd` from an array of N values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let simd = CuSimd::<f32, 4>::new([1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline(always)]
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Access element by compile-time index.
    ///
    /// This is preferred when the index is known at compile time, as it
    /// enables the compiler to optimize and catches out-of-bounds errors
    /// at compile time.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let simd = CuSimd::<f32, 4>::new([1.0, 2.0, 3.0, 4.0]);
    /// let third = simd.get::<2>();  // 3.0
    /// ```
    #[inline(always)]
    pub const fn get<const I: usize>(&self) -> T {
        self.data[I]
    }

    /// Access element by runtime index (returns by value/copy).
    ///
    /// For GPU code, copy semantics are natural - register values don't
    /// have addressable memory locations.
    ///
    /// Note: You can also use `simd[i]` syntax via the `Index` trait.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let simd = CuSimd::<f32, 4>::new([1.0, 2.0, 3.0, 4.0]);
    /// let idx = thread::threadIdx_x() % 4;
    /// let val = simd.at(idx as usize);  // or: simd[idx as usize]
    /// ```
    #[inline(always)]
    pub fn at(&self, i: usize) -> T {
        self.data[i]
    }

    /// Convert to array, consuming self.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let simd = CuSimd::<f32, 4>::new([1.0, 2.0, 3.0, 4.0]);
    /// let arr: [f32; 4] = simd.to_array();
    /// ```
    #[inline(always)]
    pub const fn to_array(self) -> [T; N] {
        self.data
    }

    /// Get the number of elements.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        N
    }

    /// Check if empty (always false for valid CuSimd).
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }
}

// =============================================================================
// Index Trait Implementation
// =============================================================================

/// Enables `simd[i]` syntax for runtime indexing.
///
/// Note: `Index` returns `&T` per Rust's trait definition, but for `Copy`
/// types the value is immediately copied at the use site. This is semantically
/// equivalent to returning by value for our use case.
impl<T: SimdElement, const N: usize> Index<usize> for CuSimd<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

// =============================================================================
// Shorthand Accessors for N=2
// =============================================================================

impl<T: SimdElement> CuSimd<T, 2> {
    /// Get first element (index 0).
    #[inline(always)]
    pub const fn x(&self) -> T {
        self.data[0]
    }

    /// Get second element (index 1).
    #[inline(always)]
    pub const fn y(&self) -> T {
        self.data[1]
    }

    /// Get both elements as a tuple.
    #[inline(always)]
    pub const fn xy(self) -> (T, T) {
        (self.data[0], self.data[1])
    }
}

// =============================================================================
// Shorthand Accessors for N=4
// =============================================================================

impl<T: SimdElement> CuSimd<T, 4> {
    /// Get first element (index 0).
    #[inline(always)]
    pub const fn x(&self) -> T {
        self.data[0]
    }

    /// Get second element (index 1).
    #[inline(always)]
    pub const fn y(&self) -> T {
        self.data[1]
    }

    /// Get third element (index 2).
    #[inline(always)]
    pub const fn z(&self) -> T {
        self.data[2]
    }

    /// Get fourth element (index 3).
    #[inline(always)]
    pub const fn w(&self) -> T {
        self.data[3]
    }

    /// Get all four elements as a tuple.
    #[inline(always)]
    pub const fn xyzw(self) -> (T, T, T, T) {
        (self.data[0], self.data[1], self.data[2], self.data[3])
    }

    /// Get lower half (elements 0, 1) as CuSimd<T, 2>.
    #[inline(always)]
    pub const fn lo(&self) -> CuSimd<T, 2> {
        CuSimd::new([self.data[0], self.data[1]])
    }

    /// Get upper half (elements 2, 3) as CuSimd<T, 2>.
    #[inline(always)]
    pub const fn hi(&self) -> CuSimd<T, 2> {
        CuSimd::new([self.data[2], self.data[3]])
    }
}

// =============================================================================
// Type Aliases for Common Configurations
// =============================================================================

/// 2 f32 values (vector float2).
pub type Float2 = CuSimd<f32, 2>;

/// 4 f32 values (vector float4).
pub type Float4 = CuSimd<f32, 4>;

/// 2 f64 values (vector double2).
pub type Double2 = CuSimd<f64, 2>;

/// 2 i32 values (vector int2).
pub type Int2 = CuSimd<i32, 2>;

/// 4 i32 values (vector int4).
pub type Int4 = CuSimd<i32, 4>;

/// 2 u32 values (vector uint2).
pub type Uint2 = CuSimd<u32, 2>;

/// 4 u32 values (vector uint4).
pub type Uint4 = CuSimd<u32, 4>;

// =============================================================================
// TMEM Load Result Types (for tcgen05 operations)
// =============================================================================

/// 4 f32 registers from tcgen05.ld (base LDTM).
///
/// Replaces the old `TmemF32x4` struct with a generic type.
pub type TmemRegs4 = CuSimd<f32, 4>;

/// 32 f32 registers from tcgen05.ld.x8.
///
/// Replaces the old `TmemF32x32` struct with a generic type.
pub type TmemRegs32 = CuSimd<f32, 32>;

/// 16 f32 registers from tcgen05.ld.x4.
pub type TmemRegs16 = CuSimd<f32, 16>;

/// 8 f32 registers from tcgen05.ld.x2.
pub type TmemRegs8 = CuSimd<f32, 8>;

// =============================================================================
// Default Implementation
// =============================================================================

impl<T: SimdElement + Default, const N: usize> Default for CuSimd<T, N> {
    fn default() -> Self {
        // Note: This requires T: Default which f32/u32 etc. implement
        // Using array initialization with default values
        Self {
            data: core::array::from_fn(|_| T::default()),
        }
    }
}

// =============================================================================
// PartialEq Implementation
// =============================================================================

impl<T: SimdElement + PartialEq, const N: usize> PartialEq for CuSimd<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: SimdElement + Eq, const N: usize> Eq for CuSimd<T, N> {}
