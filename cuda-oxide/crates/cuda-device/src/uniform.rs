/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launch-uniform scalar witnesses.
//!
//! Some device APIs need a value that is identical in every thread of a launch.
//! A kernel scalar parameter is uniform by construction: the host marshals one
//! value into the launch packet, and every thread of every block reads those
//! same bytes. [`Uniform<T>`] captures that fact in the type system:
//!
//! ```rust,ignore
//! #[kernel]
//! #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
//! pub fn scale(rows: Uniform<u32>, mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
//!     // `rows` came from the launch packet, so it is the same in every thread.
//!     if coord.row() >= rows.get() { return; }
//!     if let Some(mut tile) = c.tile_2d32_rt(coord) { /* ... */ }
//! }
//! ```
//!
//! The host method still takes a plain `u32`; `#[cuda_module]` wraps it. There
//! is no constructor, so device code cannot manufacture a witness from a
//! thread-dependent value. The one escape hatch is
//! [`Uniform::new_unchecked`], for a value proven uniform some other way.
//!
//! # What this witness cannot carry
//!
//! Uniformity is closed under arithmetic whose operands are all uniform, and it
//! is not closed under selection. Choosing between two `Uniform<T>` values on a
//! thread-varying condition yields a value that differs between threads.
//!
//! So a fact that has to hold across a whole slice, rather than at one call,
//! cannot be carried here. A row width is the example: it belongs to the
//! slice's index space, bound by the host, and
//! [`crate::DisjointSlice::tile_2d32_rt`] reads it from there rather than
//! taking it as an argument.

/// A scalar proven identical in every thread of the launch.
///
/// The only safe source is a kernel parameter. `#[repr(transparent)]` makes the
/// ABI identical to `T`, so the kernel entry receives the value directly and no
/// constructor runs on the device.
///
/// The private field is what carries the proof: no code outside this module
/// can name it, so no code outside this module can build a witness. Adding a
/// public constructor, or deriving anything that rebuilds the value from parts,
/// would defeat it.
#[repr(transparent)]
#[derive(Debug)]
pub struct Uniform<T> {
    value: T,
}

impl<T: Copy> Clone for Uniform<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Copying a launch-uniform value leaves it launch-uniform.
impl<T: Copy> Copy for Uniform<T> {}

impl<T> Uniform<T> {
    /// Assert that `value` is identical in every thread of this launch.
    ///
    /// Prefer declaring the kernel parameter as `Uniform<T>`, which carries the
    /// same proof with no obligation.
    ///
    /// # Safety
    ///
    /// `value` must be the same in every thread of the launch. A value derived
    /// from `threadIdx`, `blockIdx`, a per-thread load, or any branch on those
    /// does not qualify. Block and grid dimensions do qualify; see
    /// [`block_dim_x`] and its siblings for witnesses that need no `unsafe`.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: T) -> Self {
        Self { value }
    }

    /// Return the underlying value.
    ///
    /// The result is an ordinary scalar. Arithmetic on it loses the proof
    /// unless every operand is itself uniform, which is what the operator
    /// implementations below preserve.
    #[inline(always)]
    pub const fn get(&self) -> T
    where
        T: Copy,
    {
        self.value
    }
}

/// Uniformity is closed under arithmetic whose operands are all uniform.
///
/// Every thread evaluates the same operator over the same inputs, so the result
/// is the same in every thread. Wrapping and saturating forms are used because
/// a panic on overflow would be a divergent side effect in device code.
macro_rules! uniform_binop {
    ($($method:ident: $doc:literal),* $(,)?) => {
        impl Uniform<u32> {
            $(
                #[doc = $doc]
                #[inline(always)]
                pub fn $method(self, other: Self) -> Self {
                    Self {
                        value: self.value.$method(other.value),
                    }
                }
            )*
        }
    };
}

uniform_binop! {
    wrapping_add: "Wrapping addition of two uniform values.",
    wrapping_sub: "Wrapping subtraction of two uniform values.",
    wrapping_mul: "Wrapping multiplication of two uniform values.",
    saturating_add: "Saturating addition of two uniform values.",
    saturating_sub: "Saturating subtraction of two uniform values.",
    min: "The smaller of two uniform values.",
    max: "The larger of two uniform values.",
}

impl Uniform<u32> {
    /// Multiply by a compile-time constant.
    ///
    /// A constant is the same in every thread, so the product stays uniform.
    /// This is a const parameter rather than an ordinary argument precisely
    /// because an argument could carry a per-thread value.
    #[inline(always)]
    pub const fn wrapping_mul_const<const C: u32>(self) -> Self {
        Self {
            value: self.value.wrapping_mul(C),
        }
    }

    /// Add a compile-time constant.
    #[inline(always)]
    pub const fn wrapping_add_const<const C: u32>(self) -> Self {
        Self {
            value: self.value.wrapping_add(C),
        }
    }
}

mod uniform_sealed {
    pub trait Sealed {}
}

impl<T> uniform_sealed::Sealed for Uniform<T> {}

/// Compiler-facing proof that a kernel parameter is cuda-device's own
/// [`Uniform<T>`] over the expected scalar.
///
/// `#[cuda_module]` selects a parameter's host ABI from the spelling of its
/// type, so a local type also named `Uniform` would otherwise choose the
/// scalar host signature while presenting a different device layout. This
/// trait is sealed, and `#[cuda_module]` bounds every `Uniform` parameter by
/// it, so Rust resolves aliases and rejects a look-alike before any launch is
/// generated.
#[doc(hidden)]
pub trait __LaunchContractUniform<Scalar>: uniform_sealed::Sealed {}

impl<T> __LaunchContractUniform<T> for Uniform<T> {}

/// Threads per block on X, which the launch fixes for every thread.
///
/// A launch has one block shape, so every thread reads the same value. These
/// are witnesses that need no `unsafe`, unlike a value the kernel computes.
#[inline(always)]
pub fn block_dim_x() -> Uniform<u32> {
    // SAFETY: `%ntid.x` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::blockDim_x()) }
}

/// Threads per block on Y.
#[inline(always)]
pub fn block_dim_y() -> Uniform<u32> {
    // SAFETY: `%ntid.y` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::blockDim_y()) }
}

/// Threads per block on Z.
#[inline(always)]
pub fn block_dim_z() -> Uniform<u32> {
    // SAFETY: `%ntid.z` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::blockDim_z()) }
}

/// Blocks per grid on X.
#[inline(always)]
pub fn grid_dim_x() -> Uniform<u32> {
    // SAFETY: `%nctaid.x` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::gridDim_x()) }
}

/// Blocks per grid on Y.
#[inline(always)]
pub fn grid_dim_y() -> Uniform<u32> {
    // SAFETY: `%nctaid.y` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::gridDim_y()) }
}

/// Blocks per grid on Z.
#[inline(always)]
pub fn grid_dim_z() -> Uniform<u32> {
    // SAFETY: `%nctaid.z` is a launch-wide constant.
    unsafe { Uniform::new_unchecked(crate::thread::gridDim_z()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn uniform_is_abi_identical_to_its_scalar() {
        assert_eq!(size_of::<Uniform<u32>>(), size_of::<u32>());
        assert_eq!(align_of::<Uniform<u32>>(), align_of::<u32>());
        assert_eq!(size_of::<Uniform<u64>>(), size_of::<u64>());
    }

    #[test]
    fn arithmetic_over_uniform_operands_stays_uniform() {
        // SAFETY: test-local constants are trivially uniform.
        let a = unsafe { Uniform::new_unchecked(6u32) };
        let b = unsafe { Uniform::new_unchecked(7u32) };
        assert_eq!(a.wrapping_mul(b).get(), 42);
        assert_eq!(a.wrapping_add(b).get(), 13);
        assert_eq!(b.wrapping_sub(a).get(), 1);
        assert_eq!(a.min(b).get(), 6);
        assert_eq!(a.max(b).get(), 7);
        assert_eq!(a.wrapping_mul_const::<4>().get(), 24);
        assert_eq!(a.wrapping_add_const::<4>().get(), 10);
    }

    #[test]
    fn wrapping_arithmetic_does_not_panic_at_the_boundary() {
        // SAFETY: test-local constants are trivially uniform.
        let max = unsafe { Uniform::new_unchecked(u32::MAX) };
        let one = unsafe { Uniform::new_unchecked(1u32) };
        assert_eq!(max.wrapping_add(one).get(), 0);
        assert_eq!(max.saturating_add(one).get(), u32::MAX);
        let zero = unsafe { Uniform::new_unchecked(0u32) };
        assert_eq!(zero.saturating_sub(one).get(), 0);
    }
}
