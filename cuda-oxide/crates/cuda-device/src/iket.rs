/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic annotations for In-Kernel Event Tracing (IKET).
//!
//! The functions behind these macros are compiler markers. `mir-importer`
//! replaces them with `iket.*` operations; they must never execute as ordinary
//! host or device functions.

use core::marker::PhantomData;

#[doc(inline)]
pub use crate::__cuda_oxide_iket_mark as mark;
#[doc(inline)]
pub use crate::__cuda_oxide_iket_range_end as range_end;
#[doc(inline)]
pub use crate::__cuda_oxide_iket_range_pop as range_pop;
#[doc(inline)]
pub use crate::__cuda_oxide_iket_range_push as range_push;
#[doc(inline)]
pub use crate::__cuda_oxide_iket_range_start as range_start;

/// Scalar representation carried by an IKET event.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadKind {
    None = 0,
    I8 = 1,
    U8 = 2,
    I16 = 3,
    U16 = 4,
    I32 = 5,
    U32 = 6,
    I64 = 7,
    U64 = 8,
    F32 = 9,
    F64 = 10,
    Pointer = 11,
}

/// Values accepted as IKET payloads.
#[doc(hidden)]
pub trait PayloadValue {
    const KIND: PayloadKind;
    fn __iket_bits(self) -> u64;
}

macro_rules! integer_payload {
    ($ty:ty, $kind:ident) => {
        impl PayloadValue for $ty {
            const KIND: PayloadKind = PayloadKind::$kind;

            #[inline(always)]
            fn __iket_bits(self) -> u64 {
                self as u64
            }
        }
    };
}

integer_payload!(u8, U8);
integer_payload!(u16, U16);
integer_payload!(u32, U32);
integer_payload!(u64, U64);

macro_rules! signed_payload {
    ($ty:ty, $unsigned:ty, $kind:ident) => {
        impl PayloadValue for $ty {
            const KIND: PayloadKind = PayloadKind::$kind;

            #[inline(always)]
            fn __iket_bits(self) -> u64 {
                self as $unsigned as u64
            }
        }
    };
}

signed_payload!(i8, u8, I8);
signed_payload!(i16, u16, I16);
signed_payload!(i32, u32, I32);
signed_payload!(i64, u64, I64);

impl PayloadValue for f32 {
    const KIND: PayloadKind = PayloadKind::F32;

    #[inline(always)]
    fn __iket_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl PayloadValue for f64 {
    const KIND: PayloadKind = PayloadKind::F64;

    #[inline(always)]
    fn __iket_bits(self) -> u64 {
        self.to_bits()
    }
}

impl<T> PayloadValue for *const T {
    const KIND: PayloadKind = PayloadKind::Pointer;

    #[inline(always)]
    fn __iket_bits(self) -> u64 {
        self as usize as u64
    }
}

impl<T> PayloadValue for *mut T {
    const KIND: PayloadKind = PayloadKind::Pointer;

    #[inline(always)]
    fn __iket_bits(self) -> u64 {
        self as usize as u64
    }
}

/// Compile-time identity of one token-paired range.
#[doc(hidden)]
pub trait RangeDescriptor {}

/// Linear token returned by [`range_start!`](range_start).
#[must_use = "an IKET range token must be consumed by iket::range_end!"]
pub struct RangeToken<R: RangeDescriptor> {
    range: PhantomData<R>,
}

/// Record a point event, optionally with a scalar payload.
#[doc(hidden)]
#[macro_export]
macro_rules! __cuda_oxide_iket_mark {
    ($name:literal) => {{ $crate::iket::__iket_mark($name) }};
    ($name:literal, $payload:expr) => {{ $crate::iket::__iket_mark_payload($name, $payload) }};
    ($name:expr $(, $payload:expr)?) => {
        compile_error!("IKET event names must be string literals")
    };
}

/// Begin a token-paired range, optionally with a scalar payload.
#[doc(hidden)]
#[macro_export]
macro_rules! __cuda_oxide_iket_range_start {
    ($name:literal) => {{
        struct __CudaOxideIketRange;
        impl $crate::iket::RangeDescriptor for __CudaOxideIketRange {}
        $crate::iket::__iket_range_start::<__CudaOxideIketRange>($name)
    }};
    ($name:literal, $payload:expr) => {{
        struct __CudaOxideIketRange;
        impl $crate::iket::RangeDescriptor for __CudaOxideIketRange {}
        $crate::iket::__iket_range_start_payload::<__CudaOxideIketRange, _>($name, $payload)
    }};
    ($name:expr $(, $payload:expr)?) => {
        compile_error!("IKET range names must be string literals")
    };
}

/// End a token-paired range, optionally with a scalar payload.
#[doc(hidden)]
#[macro_export]
macro_rules! __cuda_oxide_iket_range_end {
    ($token:expr) => {{ $crate::iket::__iket_range_end($token) }};
    ($token:expr, $payload:expr) => {{ $crate::iket::__iket_range_end_payload($token, $payload) }};
}

/// Push a LIFO range, optionally with a scalar payload.
#[doc(hidden)]
#[macro_export]
macro_rules! __cuda_oxide_iket_range_push {
    ($name:literal) => {{ $crate::iket::__iket_range_push($name) }};
    ($name:literal, $payload:expr) => {{ $crate::iket::__iket_range_push_payload($name, $payload) }};
    ($name:expr $(, $payload:expr)?) => {
        compile_error!("IKET range names must be string literals")
    };
}

/// Pop the most recently pushed LIFO range.
#[doc(hidden)]
#[macro_export]
macro_rules! __cuda_oxide_iket_range_pop {
    () => {{ $crate::iket::__iket_range_pop() }};
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_mark(_event_name: &'static str) {
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_mark_payload<T: PayloadValue>(_event_name: &'static str, payload: T) {
    let _ = (T::KIND, payload.__iket_bits());
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_start<R: RangeDescriptor>(_event_name: &'static str) -> RangeToken<R> {
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_start_payload<R: RangeDescriptor, T: PayloadValue>(
    _event_name: &'static str,
    payload: T,
) -> RangeToken<R> {
    let _ = (T::KIND, payload.__iket_bits());
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_end<R: RangeDescriptor>(_token: RangeToken<R>) {
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_end_payload<R: RangeDescriptor, T: PayloadValue>(
    _token: RangeToken<R>,
    payload: T,
) {
    let _ = (T::KIND, payload.__iket_bits());
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_push(_event_name: &'static str) {
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_push_payload<T: PayloadValue>(_event_name: &'static str, payload: T) {
    let _ = (T::KIND, payload.__iket_bits());
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[doc(hidden)]
#[inline(never)]
pub fn __iket_range_pop() {
    unreachable!("IKET compiler marker executed outside CUDA compilation")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_bits_preserve_source_width_and_kind() {
        assert_eq!((-7i8).__iket_bits(), 249);
        assert_eq!((-7i16).__iket_bits(), 65_529);
        assert_eq!(1.25f32.__iket_bits(), 1.25f32.to_bits() as u64);
        assert_eq!(<u64 as PayloadValue>::KIND, PayloadKind::U64);
        assert_eq!(<*const u8 as PayloadValue>::KIND, PayloadKind::Pointer);
    }
}
