/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Over-aligned element types, and a checked way to view a flat slice as them.
//!
//! Memory transaction width is set by the alignment of the *element type*, not by
//! how many elements a loop touches. Measured on sm_86, a `#[repr(C, align(16))]`
//! quad of `f32` compiles to one `LDG.E.128` plus one `STG.E.128`, while the same
//! four values at natural alignment compile to four scalar accesses. Static
//! length alone changes nothing: reading four adjacent elements out of a
//! `&[f32]`, however it is spelled, stays scalar because the element type is
//! still 4-byte aligned.
//!
//! So reaching a wide access is a question of *choosing an element type*, and the
//! obstacle is that buffers usually arrive flat. Copying into an aligned buffer
//! defeats the point. What is needed is a view: check the alignment and length
//! once at the boundary, then every access through the view is wide.
//!
//! ```rust,ignore
//! // Flat input, checked once.
//! let quads: &[F32x4] = vector::as_vectors(input).ok_or(Unaligned)?;
//! // Every read through `quads` is a 128-bit transaction.
//! let q = quads[i];
//! ```
//!
//! # Alignment is necessary but not sufficient for a wide *load*
//!
//! Measured on sm_86, with the element type 16-byte aligned throughout, only the
//! way the value is *consumed* changes the load:
//!
//! ```text
//! how the element is used                     LDG          STG
//! `*out = input[i]`, never decomposed         LDG.E.128    STG.E.128
//! lanes read, whole value assembled+stored    LDG.E x4     STG.E.128
//! lanes read, scalar stored                   LDG.E x4     STG.E
//! ```
//!
//! Touching individual lanes lets the optimiser split the local apart, and it
//! then loads the lanes separately no matter how the type is aligned. The store
//! side has no such condition: it widens whenever a whole value is assembled.
//!
//! So these types buy width on the paths that **move** data - staging global to
//! shared memory, copying, packing - and not on paths that immediately do
//! elementwise arithmetic. That is the same split CUTLASS and cuTile arrive at:
//! wide loads feed shared memory, and compute reads from there. If a kernel
//! needs both, stage through shared memory rather than expecting one loop to get
//! a wide load and lane access at once.
//!
//! # Why the check nearly always passes
//!
//! Device allocations are 256-byte aligned, so a buffer's base satisfies any of
//! these types. The check exists to make that a guarantee rather than an
//! assumption, and to fail loudly on the cases where it does not hold: a slice
//! taken at a hand-computed offset, a sub-slice starting at an odd index, or a
//! length that is not a whole number of vectors.
//!
//! # Host-side story
//!
//! `cuda-core` does not depend on `cuda-device`, so these types carry no
//! `DeviceCopy` impl, and the orphan rule stops downstream crates from adding
//! one: a `DeviceBuffer<F32x4>` cannot be filled from host values of these
//! types. That is fine, because the intended flow keeps host buffers *flat*:
//!
//! - Allocate and fill a `DeviceBuffer<f32>` as usual, pass it to a kernel
//!   taking `&[f32]`, and take the view **inside the kernel** with
//!   [`as_vectors`] / [`as_vectors_mut`]. This needs no host-side type at all.
//! - Or reinterpret the flat buffer once at the boundary with `cuda-core`'s
//!   `DeviceBuffer::cast_chunks`, using a host-defined `#[repr(C, align(16))]`
//!   element that implements `DeviceCopy`, and launch a kernel that takes that
//!   element type directly.
//!
//! # Related "N x scalar" types
//!
//! cuda-device spells grouped scalars three ways, for three different jobs:
//!
//! - **This module**: over-aligned *memory element* types. The grouping lives
//!   in the type's alignment so that loads and stores are wide transactions.
//! - [`crate::cusimd::CuSimd`]: *register* groups. N values travelling
//!   together through registers (e.g. `tcgen05.ld` results), with copy
//!   semantics and no memory-layout claim at all.
//! - [`crate::f16x2`] / [`crate::bf16x2`]: *packed arithmetic* pairs. Two
//!   16-bit values in one `u32` operated on by a single SIMD instruction;
//!   the packing is the ALU format, not an alignment trick.

use core::mem::{align_of, size_of};

/// An over-aligned group of scalars that moves in one memory transaction.
///
/// Implemented only by the types in this module. The bound exists so
/// [`as_vectors`] can relate a vector type to its scalar element and check both
/// invariants; it is deliberately not something callers implement, because a
/// wrong `LANES` would make the length check admit an out-of-bounds view.
///
/// # Safety
///
/// An implementor must be `#[repr(C)]`, contain exactly `LANES` values of
/// `Elem` and nothing else, and carry an alignment of at least
/// `size_of::<Elem>() * LANES`.
pub unsafe trait Vector: Copy {
    /// Scalar type of each lane.
    type Elem: Copy;
    /// Lanes in the vector.
    const LANES: usize;
}

/// Define an over-aligned vector type plus its `Vector` impl and accessors.
macro_rules! define_vector {
    (
        $(#[$doc:meta])*
        $name:ident, $elem:ty, $lanes:literal, $align:literal
    ) => {
        $(#[$doc])*
        #[repr(C, align($align))]
        #[derive(Clone, Copy, PartialEq, Debug)]
        pub struct $name(pub [$elem; $lanes]);

        // SAFETY: `repr(C)` over exactly `$lanes` values of `$elem`, with
        // `align($align)` and `$align == size_of::<$elem>() * $lanes` checked by
        // `layout_matches_the_declared_contract` below.
        unsafe impl Vector for $name {
            type Elem = $elem;
            const LANES: usize = $lanes;
        }

        impl $name {
            /// Build a vector from its lanes.
            #[must_use]
            #[inline(always)]
            pub const fn new(lanes: [$elem; $lanes]) -> Self {
                Self(lanes)
            }

            /// Every lane equal to `value`.
            #[must_use]
            #[inline(always)]
            pub const fn splat(value: $elem) -> Self {
                Self([value; $lanes])
            }

            /// The lanes, by value.
            #[must_use]
            #[inline(always)]
            pub const fn to_array(self) -> [$elem; $lanes] {
                self.0
            }

            /// The lanes, as a slice.
            #[must_use]
            #[inline(always)]
            pub fn as_slice(&self) -> &[$elem] {
                &self.0
            }

            /// The lanes, mutably.
            #[must_use]
            #[inline(always)]
            pub fn as_mut_slice(&mut self) -> &mut [$elem] {
                &mut self.0
            }
        }
    };
}

define_vector!(
    /// Two `f32` in one 64-bit transaction.
    F32x2, f32, 2, 8
);
define_vector!(
    /// Four `f32` in one 128-bit transaction. The `float4` equivalent.
    F32x4, f32, 4, 16
);
define_vector!(
    /// Two `f64` in one 128-bit transaction.
    F64x2, f64, 2, 16
);
define_vector!(
    /// Two `u32` in one 64-bit transaction.
    U32x2, u32, 2, 8
);
define_vector!(
    /// Four `u32` in one 128-bit transaction.
    U32x4, u32, 4, 16
);
define_vector!(
    /// Four `u16` in one 64-bit transaction.
    U16x4, u16, 4, 8
);
define_vector!(
    /// Eight `u16` in one 128-bit transaction. The width packed f16 pairs use.
    U16x8, u16, 8, 16
);

/// View a flat slice as over-aligned vectors, or `None` if it does not divide.
///
/// Fails when the length is not a multiple of `V::LANES`, or when the base
/// pointer is not aligned for `V`. Both are checked once here; every access
/// through the returned slice is then a full-width transaction.
///
/// The tail is not silently dropped. A slice of 10 `f32` viewed as `F32x4` is an
/// error rather than two vectors and a discarded remainder, because discarding
/// data is the kind of thing that should be written down at the call site.
///
/// # Empty slices
///
/// An empty input always yields `Some` empty view, regardless of its base
/// address: there is nothing to access, so there is no alignment to violate.
/// This makes the literal `&[]` (whose dangling base is only aligned for the
/// scalar) and an aligned empty sub-slice of a real buffer behave identically.
#[must_use]
pub fn as_vectors<V: Vector>(slice: &[V::Elem]) -> Option<&[V]> {
    let len = vector_len::<V>(slice.len(), slice.as_ptr() as usize)?;
    if len == 0 {
        // `from_raw_parts` requires a base aligned for `V` even at length 0,
        // and an empty input's base may only be aligned for the scalar. The
        // empty view does not need the base at all.
        return Some(&[]);
    }
    // SAFETY: length divides exactly and the base is aligned for `V`, both
    // checked above. `V` is `repr(C)` over `LANES` values of `Elem`, so `len`
    // vectors cover exactly `slice.len()` elements.
    Some(unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<V>(), len) })
}

/// Mutable [`as_vectors`]. Shares its semantics, including the empty-slice
/// case: any empty input yields `Some` empty view.
#[must_use]
pub fn as_vectors_mut<V: Vector>(slice: &mut [V::Elem]) -> Option<&mut [V]> {
    let len = vector_len::<V>(slice.len(), slice.as_ptr() as usize)?;
    if len == 0 {
        // As in `as_vectors`: no access will happen, and the possibly
        // scalar-aligned base must not reach `from_raw_parts_mut`.
        return Some(&mut []);
    }
    // SAFETY: as `as_vectors`, and the exclusive borrow of `slice` is consumed
    // for the lifetime of the result, so no aliasing view coexists.
    Some(unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<V>(), len) })
}

/// Shared checks: exact division and alignment.
#[inline]
fn vector_len<V: Vector>(elems: usize, addr: usize) -> Option<usize> {
    // Compile-time check of the `Vector` contract every view relies on: a `V`
    // whose size is not exactly `LANES * size_of::<Elem>()` would make the
    // element count and the byte extent disagree, and every index through the
    // view would then read the wrong bytes. Evaluated when `vector_len` is
    // instantiated for a `V`, so a malformed impl fails the build instead of
    // silently misindexing.
    const {
        assert!(
            size_of::<V>() == V::LANES * size_of::<V::Elem>(),
            "Vector impl violates its contract: size_of::<V>() must equal LANES * size_of::<V::Elem>()"
        );
    }
    if elems == 0 {
        // Uniform empty-slice semantics: nothing will be accessed, so the
        // base address (which for a literal `&[]` is a dangling pointer with
        // only the scalar's alignment) is irrelevant.
        return Some(0);
    }
    if V::LANES == 0 || size_of::<V::Elem>() == 0 {
        return None;
    }
    if !elems.is_multiple_of(V::LANES) {
        return None;
    }
    if !addr.is_multiple_of(align_of::<V>()) {
        return None;
    }
    Some(elems / V::LANES)
}

/// Whether a slice could be viewed as `V`, without building the view.
///
/// For deciding between a wide path and a scalar fallback before committing to
/// either.
#[must_use]
pub fn is_viewable<V: Vector>(slice: &[V::Elem]) -> bool {
    vector_len::<V>(slice.len(), slice.as_ptr() as usize).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `unsafe impl Vector` blocks assert a layout; this checks it, since a
    /// wrong `LANES` or a too-small alignment would make the view unsound.
    #[test]
    fn layout_matches_the_declared_contract() {
        macro_rules! check {
            ($t:ty) => {{
                type V = $t;
                let payload = size_of::<<V as Vector>::Elem>() * <V as Vector>::LANES;
                assert_eq!(
                    size_of::<V>(),
                    payload,
                    "{} must be exactly its lanes, with no padding",
                    stringify!($t)
                );
                assert!(
                    align_of::<V>() >= payload,
                    "{} alignment {} is below its {payload}-byte payload",
                    stringify!($t),
                    align_of::<V>()
                );
                // Only 64- and 128-bit transactions are worth over-aligning for.
                assert!(
                    align_of::<V>() == 8 || align_of::<V>() == 16,
                    "{} has alignment {}, which is not a transaction width",
                    stringify!($t),
                    align_of::<V>()
                );
            }};
        }
        check!(F32x2);
        check!(F32x4);
        check!(F64x2);
        check!(U32x2);
        check!(U32x4);
        check!(U16x4);
        check!(U16x8);
    }

    #[test]
    fn views_a_divisible_aligned_slice() {
        // Over-aligned backing store, so the base is suitable for F32x4.
        let backing = [
            F32x4::new([1.0, 2.0, 3.0, 4.0]),
            F32x4::new([5.0, 6.0, 7.0, 8.0]),
        ];
        let flat: &[f32] =
            unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<f32>(), 8) };
        let quads = as_vectors::<F32x4>(flat).expect("aligned and divisible");
        assert_eq!(quads.len(), 2);
        assert_eq!(quads[0].to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(quads[1].to_array(), [5.0, 6.0, 7.0, 8.0]);
    }

    /// A length that does not divide is refused rather than truncated, so a
    /// dropped tail cannot go unnoticed.
    #[test]
    fn refuses_a_length_that_does_not_divide() {
        let backing = [F32x4::splat(0.0); 4];
        let base = backing.as_ptr().cast::<f32>();
        for len in [1usize, 2, 3, 5, 6, 7, 9, 10, 15] {
            let flat: &[f32] = unsafe { core::slice::from_raw_parts(base, len) };
            assert!(
                as_vectors::<F32x4>(flat).is_none(),
                "{len} f32 is not a whole number of F32x4"
            );
        }
        for len in [0usize, 4, 8, 12, 16] {
            let flat: &[f32] = unsafe { core::slice::from_raw_parts(base, len) };
            assert_eq!(as_vectors::<F32x4>(flat).map(<[F32x4]>::len), Some(len / 4));
        }
    }

    /// A misaligned base is refused. This is the case the check exists for: a
    /// sub-slice starting at an odd element has the right length and the wrong
    /// address, and would otherwise fault or read the wrong data.
    #[test]
    fn refuses_a_misaligned_base() {
        let backing = [F32x4::splat(0.0); 8];
        let flat: &[f32] =
            unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<f32>(), 32) };
        // Offsets 4, 8, 12 stay 16-byte aligned; 1, 2, 3, 5, 6, 7 do not.
        for off in [4usize, 8, 12] {
            assert!(
                as_vectors::<F32x4>(&flat[off..]).is_some(),
                "offset {off} is still 16-byte aligned"
            );
        }
        for off in [1usize, 2, 3, 5, 6, 7] {
            // Take exactly 12 elements so the length divides by 4 and the
            // only possible ground for rejection is the misaligned base.
            let sub = &flat[off..off + 12];
            assert!(sub.len().is_multiple_of(4), "length must not be the reason");
            assert!(
                as_vectors::<F32x4>(sub).is_none(),
                "offset {off} is not 16-byte aligned and must be refused"
            );
        }
    }

    /// A narrower vector accepts bases a wider one rejects, which is the useful
    /// fallback: 64-bit where 128-bit will not fit.
    #[test]
    fn narrower_vectors_accept_more_bases() {
        let backing = [F32x4::splat(0.0); 4];
        let flat: &[f32] =
            unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<f32>(), 16) };
        let off2 = &flat[2..]; // 8-byte aligned, 14 elements
        assert!(
            as_vectors::<F32x4>(off2).is_none(),
            "not 16-byte aligned, and 14 does not divide by 4"
        );
        assert!(
            as_vectors::<F32x2>(off2).is_some(),
            "8-byte aligned and 14 divides by 2"
        );
    }

    /// Every empty input views as an empty vector slice, whatever its base
    /// address: the literal `&[]` (dangling, scalar-aligned base) and an
    /// empty sub-slice at a misaligned offset behave identically.
    #[test]
    fn empty_slices_always_view() {
        let empty = as_vectors::<F32x4>(&[]).expect("literal empty slice");
        assert!(empty.is_empty());
        assert!(is_viewable::<F32x4>(&[]));
        let mut none: [f32; 0] = [];
        assert!(
            as_vectors_mut::<F32x4>(&mut none)
                .expect("empty mut")
                .is_empty()
        );

        let backing = [F32x4::splat(0.0); 2];
        let flat: &[f32] =
            unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<f32>(), 8) };
        // Zero-length sub-slice at a misaligned offset: still an empty view.
        let sub = &flat[1..1];
        assert!(
            as_vectors::<F32x4>(sub)
                .expect("empty sub-slice")
                .is_empty()
        );
        assert!(is_viewable::<F32x4>(sub));
    }

    #[test]
    fn mutable_view_writes_through() {
        let mut backing = [F32x4::splat(0.0); 2];
        {
            let flat: &mut [f32] =
                unsafe { core::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<f32>(), 8) };
            let quads = as_vectors_mut::<F32x4>(flat).unwrap();
            quads[1] = F32x4::new([9.0, 8.0, 7.0, 6.0]);
        }
        assert_eq!(backing[1].to_array(), [9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn is_viewable_agrees_with_as_vectors() {
        let backing = [F32x4::splat(0.0); 4];
        let flat: &[f32] =
            unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<f32>(), 16) };
        for off in 0..8usize {
            let tail = &flat[off..];
            assert_eq!(
                is_viewable::<F32x4>(tail),
                as_vectors::<F32x4>(tail).is_some(),
                "disagreement at offset {off}"
            );
        }
    }

    #[test]
    fn lanes_round_trip() {
        let v = F32x4::new([1.5, -2.5, 3.5, -4.5]);
        assert_eq!(v.to_array(), [1.5, -2.5, 3.5, -4.5]);
        assert_eq!(v.as_slice(), &[1.5, -2.5, 3.5, -4.5]);
        assert_eq!(U16x8::splat(7).to_array(), [7u16; 8]);
        let mut m = F32x2::splat(0.0);
        m.as_mut_slice()[1] = 4.0;
        assert_eq!(m.to_array(), [0.0, 4.0]);
    }
}
