/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 Notes:
 - Parameter names should not be changed without careful refactoring.
   They are sometimes used to generate corresponding Rust functions for cuda_tile operations.
*/

#![allow(nonstandard_style)]
#![allow(unused_variables)]

//! Core GPU kernel programming types and operations.
//!
//! Sections mirror the Tile IR documentation
//! (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/>).
//! Doc comments here are intentionally terse — see Tile IR for op semantics.

// ---------------------------------------------------------------
// Static operation parameter modules
//
// Defined outside the proc-macro-processed `core` module because the
// module macro doesn't support nested `mod` items. Re-exported from
// `core` via `pub use`.
//
// Each module defines a trait (`Mode`) and zero-sized marker structs.
// Binary switches use Enabled/Disabled. Multi-valued parameters use
// descriptive names.
// ---------------------------------------------------------------

/// Flush-to-zero modifier. Flushes denormal inputs and results to
/// sign-preserving zero. Only supported for f32.
pub mod ftz {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Rounding mode for floating-point operations. Mirrors Tile IR `RoundingMode`.
pub mod rounding {
    pub trait Mode {}
    pub struct NearestEven;
    pub struct PositiveInf;
    pub struct NegativeInf;
    pub struct Zero;
    pub struct Approx;
    pub struct Full;
    pub struct NearestIntToZero;
    pub struct NearestAway;
    impl Mode for NearestEven {}
    impl Mode for PositiveInf {}
    impl Mode for NegativeInf {}
    impl Mode for Zero {}
    impl Mode for Approx {}
    impl Mode for Full {}
    impl Mode for NearestIntToZero {}
    impl Mode for NearestAway {}
}

/// Whether an automatic allocation's address may be shared across tile threads.
pub mod allocation {
    pub trait Mode {}
    pub struct Local;
    pub struct Global;
    impl Mode for Local {}
    impl Mode for Global {}
}

/// Accumulation mode for the raw floating-point matrix multiply operation.
pub mod fast_acc {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Saturation for raw float-to-integer conversion.
pub mod saturating {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// NaN propagation for maxf/minf operations.
pub mod nan {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Atomic read-modify-write op selector. Mirrors Tile IR `AtomicRMWMode`.
pub mod atomic {
    pub trait Mode {}
    pub struct Add;
    pub struct AddF;
    pub struct And;
    pub struct Or;
    pub struct Xor;
    pub struct Max;
    pub struct Min;
    pub struct Umax;
    pub struct Umin;
    pub struct Xchg;
    impl Mode for Add {}
    impl Mode for AddF {}
    impl Mode for And {}
    impl Mode for Or {}
    impl Mode for Xor {}
    impl Mode for Max {}
    impl Mode for Min {}
    impl Mode for Umax {}
    impl Mode for Umin {}
    impl Mode for Xchg {}
}

/// Memory ordering, with per-op-family sub-traits restricting which variants
/// each op accepts (mirrors Tile IR `OnlyVariants` constraints).
pub mod ordering {
    mod sealed {
        pub trait AtomicOrdering {}
    }
    pub trait Mode {}
    /// `load_*_tko` ops: `Weak`, `Relaxed`, `Acquire`.
    pub trait LoadMode: Mode {}
    /// `store_*_tko` ops: `Weak`, `Relaxed`, `Release`.
    pub trait StoreMode: Mode {}
    /// `atomic_*_tko` ops: `Relaxed`, `Acquire`, `Release`, `AcqRel`.
    pub trait AtomicMode: Mode {}
    /// Atomic global loads: `Relaxed` or `Acquire`, never `Weak`.
    pub trait AtomicLoadMode: LoadMode + sealed::AtomicOrdering {}
    /// Atomic global stores: `Relaxed` or `Release`, never `Weak`.
    pub trait AtomicStoreMode: StoreMode + sealed::AtomicOrdering {}
    /// Atomic global read-modify-write orderings.
    pub trait GlobalAtomicMode: AtomicMode + sealed::AtomicOrdering {}

    pub struct Weak;
    pub struct Relaxed;
    pub struct Acquire;
    pub struct Release;
    pub struct AcqRel;

    impl Mode for Weak {}
    impl LoadMode for Weak {}
    impl StoreMode for Weak {}

    impl Mode for Relaxed {}
    impl LoadMode for Relaxed {}
    impl StoreMode for Relaxed {}
    impl AtomicMode for Relaxed {}

    impl Mode for Acquire {}
    impl LoadMode for Acquire {}
    impl AtomicMode for Acquire {}

    impl Mode for Release {}
    impl StoreMode for Release {}
    impl AtomicMode for Release {}

    impl Mode for AcqRel {}
    impl AtomicMode for AcqRel {}

    impl sealed::AtomicOrdering for Relaxed {}
    impl sealed::AtomicOrdering for Acquire {}
    impl sealed::AtomicOrdering for Release {}
    impl sealed::AtomicOrdering for AcqRel {}
    impl AtomicLoadMode for Relaxed {}
    impl AtomicLoadMode for Acquire {}
    impl AtomicStoreMode for Relaxed {}
    impl AtomicStoreMode for Release {}
    impl GlobalAtomicMode for Relaxed {}
    impl GlobalAtomicMode for Acquire {}
    impl GlobalAtomicMode for Release {}
    impl GlobalAtomicMode for AcqRel {}
}

/// Memory scope for atomics and load/store ordering. Mirrors Tile IR
/// `MemoryScope`.
pub mod scope {
    mod sealed {
        pub trait GlobalScope {}
    }
    pub trait Mode {}
    /// Scopes covering all tile programs that can safely share a global.
    pub trait GlobalScope: Mode + sealed::GlobalScope {}
    pub struct TileBlock;
    pub struct Device;
    pub struct System;
    impl Mode for TileBlock {}
    impl Mode for Device {}
    impl Mode for System {}
    impl sealed::GlobalScope for Device {}
    impl sealed::GlobalScope for System {}
    impl GlobalScope for Device {}
    impl GlobalScope for System {}
}

mod global_sealed {
    pub trait Atomic {}
}

/// Whether an op may use TMA (Tensor Memory Accelerator) when the hardware
/// supports it. Encodes Tile IR's `optimization_hints.allow_tma` hint.
pub mod tma {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Latency hint (Tile IR `optimization_hints.latency`). Single value applied
/// across SM archs; per-arch dictionary form is deferred.
pub struct Latency<const CYCLES: i32>;

/// Integer-overflow behavior. Mirrors Tile IR `IntegerOverflow`.
pub mod overflow {
    pub trait Mode {}
    pub struct None;
    pub struct NoSignedWrap;
    pub struct NoUnsignedWrap;
    pub struct NoWrap;
    impl Mode for None {}
    impl Mode for NoSignedWrap {}
    impl Mode for NoUnsignedWrap {}
    impl Mode for NoWrap {}
}

/// Comparison predicate for `cmpi` / `cmpf`. Mirrors Tile IR
/// `ComparisonPredicate`.
pub mod predicate {
    pub trait Mode {}
    pub struct Equal;
    pub struct NotEqual;
    pub struct LessThan;
    pub struct LessThanOrEqual;
    pub struct GreaterThan;
    pub struct GreaterThanOrEqual;
    impl Mode for Equal {}
    impl Mode for NotEqual {}
    impl Mode for LessThan {}
    impl Mode for LessThanOrEqual {}
    impl Mode for GreaterThan {}
    impl Mode for GreaterThanOrEqual {}
}

/// Floating-point comparison ordering for `cmpf` (`Ordered` requires both
/// operands to be non-NaN; `Unordered` succeeds if either is NaN). Mirrors
/// Tile IR `ComparisonOrdering`. Distinct from memory `ordering`.
pub mod cmp_ordering {
    pub trait Mode {}
    pub struct Unordered;
    pub struct Ordered;
    impl Mode for Unordered {}
    impl Mode for Ordered {}
}

/// Padding marker for partition-view construction.
///
/// `None` omits the Tile IR `padding_value` type parameter. The other markers
/// map to Tile IR `PaddingValue` values for out-of-bounds reads through views.
pub mod padding {
    pub trait Mode {}
    /// Omit the partition-view `padding_value` type parameter.
    pub struct None;
    /// Pad out-of-bounds lanes with zero.
    pub struct Zero;
    /// Pad out-of-bounds lanes with negative zero.
    pub struct NegZero;
    /// Pad out-of-bounds lanes with NaN.
    pub struct Nan;
    /// Pad out-of-bounds lanes with positive infinity.
    pub struct PosInf;
    /// Pad out-of-bounds lanes with negative infinity.
    pub struct NegInf;
    impl Mode for None {}
    impl Mode for Zero {}
    impl Mode for NegZero {}
    impl Mode for Nan {}
    impl Mode for PosInf {}
    impl Mode for NegInf {}
}

/// Optional dimension map for partition views.
pub mod dim_map {
    pub trait Mode {}
    /// Omit the partition-view `dim_map` type parameter.
    pub struct Identity;
    impl Mode for Identity {}
}

/// Direction for `scan`. Encodes Tile IR's `reverse: BoolAttr`.
pub mod reverse {
    pub trait Mode {}
    pub struct Forward;
    pub struct Reverse;
    impl Mode for Forward {}
    impl Mode for Reverse {}
}

/// Signedness for ops where it's user-facing (`mmai`, `ftoi`, `itof`).
/// For ops like `divi`/`remi`/`maxi` the compiler infers signedness from
/// the operand element type.
pub mod signedness {
    pub trait Mode {}
    pub struct Signed;
    pub struct Unsigned;
    impl Mode for Signed {}
    impl Mode for Unsigned {}
}

/// cuTile core DSL surface.
///
/// Sections mirror the Tile IR documentation
/// (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/>) and are
/// marked with banner comments below. See the linked Tile IR pages for
/// op semantics; doc comments here are intentionally terse.
#[cutile_macro::module(tile_rust_crate = true)]
pub mod core {

    pub use super::allocation;
    pub use super::atomic;
    pub use super::cmp_ordering;
    pub use super::dim_map;
    pub use super::fast_acc;
    pub use super::ftz;
    pub use super::nan;
    pub use super::ordering;
    pub use super::overflow;
    pub use super::padding;
    pub use super::predicate;
    pub use super::reverse;
    pub use super::rounding;
    pub use super::saturating;
    pub use super::scope;
    pub use super::signedness;
    pub use super::tma;
    pub use super::Latency;
    pub use half::{bf16, f16};
    use std::marker::PhantomData;
    use std::ops;

    // ========================================================================
    // TYPES — Tile IR §5
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html
    // ========================================================================

    // ---- §5.1 ELEMENT TYPES ------------------------------------------------

    /// Marker trait for valid `Tile<E, …>` / `Tensor<E, …>` element types.
    pub trait ElementType: Copy + Clone {
        const ZERO: Self;
    }

    /// Rust byte marker for packed Tile IR sub-byte values.
    ///
    /// `u8` is the low-level interop path for raw bytes. `f4e2m1fnx2` is the
    /// typed packed storage helper for two `f4e2m1fn` values in one byte.
    pub trait ByteElement: ElementType {}

    #[cuda_tile::ty(name = "bf16")]
    impl ElementType for bf16 {
        const ZERO: Self = bf16::ZERO;
    }
    #[cuda_tile::ty(name = "f16")]
    impl ElementType for f16 {
        const ZERO: Self = f16::ZERO;
    }
    #[cuda_tile::ty(name = "f32")]
    impl ElementType for f32 {
        const ZERO: Self = 0.0;
    }
    #[cuda_tile::ty(name = "i8")]
    impl ElementType for i8 {
        const ZERO: Self = 0;
    }
    impl ByteElement for i8 {}

    #[cuda_tile::ty(name = "i8")]
    impl ElementType for u8 {
        const ZERO: Self = 0;
    }
    impl ByteElement for u8 {}

    #[cuda_tile::ty(name = "i32")]
    impl ElementType for i32 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i32")]
    impl ElementType for u32 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i64")]
    impl ElementType for i64 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i64")]
    impl ElementType for u64 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "f64")]
    impl ElementType for f64 {
        const ZERO: Self = 0.0;
    }
    #[cuda_tile::ty(name = "i16")]
    impl ElementType for i16 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i16")]
    impl ElementType for u16 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i1")]
    impl ElementType for bool {
        const ZERO: Self = false;
    }

    // GPU-specific types: re-exported from cuda-core.
    pub use cuda_core::f4e2m1fn;
    pub use cuda_core::f4e2m1fnx2;
    pub use cuda_core::f8e4m3fn;
    pub use cuda_core::f8e5m2;
    pub use cuda_core::f8e5m3fnu;
    pub use cuda_core::f8e8m0fnu;
    pub use cuda_core::i4;
    pub use cuda_core::tf32;

    #[cuda_tile::ty(name = "tf32")]
    impl ElementType for tf32 {
        const ZERO: Self = tf32(0);
    }
    #[cuda_tile::ty(name = "f8e4m3fn")]
    impl ElementType for f8e4m3fn {
        const ZERO: Self = f8e4m3fn(0);
    }
    #[cuda_tile::ty(name = "f8e5m2")]
    impl ElementType for f8e5m2 {
        const ZERO: Self = f8e5m2(0);
    }
    #[cuda_tile::ty(name = "f8e8m0fnu")]
    impl ElementType for f8e8m0fnu {
        const ZERO: Self = f8e8m0fnu(0);
    }
    #[cuda_tile::ty(name = "f8e5m3fnu")]
    impl ElementType for f8e5m3fnu {
        const ZERO: Self = f8e5m3fnu(0);
    }
    #[cuda_tile::ty(name = "f4e2m1fn")]
    impl ElementType for f4e2m1fn {
        const ZERO: Self = f4e2m1fn(0);
    }
    #[cuda_tile::ty(name = "i8")]
    impl ElementType for f4e2m1fnx2 {
        const ZERO: Self = f4e2m1fnx2::from_bits(0);
    }
    impl ByteElement for f4e2m1fnx2 {}
    #[cuda_tile::ty(name = "i4")]
    impl ElementType for i4 {
        const ZERO: Self = i4(0);
    }

    /// Marker trait for scalar values that can be broadcast to tiles.
    /// Auto-implemented for every `ElementType`.
    pub trait Scalar {}
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["E"])]
    impl<E: ElementType> Scalar for E {}

    /// Method-form `scalar.broadcast(shape)` — auto-impl'd for every `ElementType`.
    #[cuda_tile::variadic_trait(N = 6)]
    pub trait BroadcastScalar<E: ElementType, const D: [i32; N]>
    where
        Self: ElementType,
    {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D>;
    }

    #[cuda_tile::variadic_trait_impl()]
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> BroadcastScalar<E, D> for E {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D> {
            broadcast_scalar(self, shape)
        }
    }

    // ---- §5.2 POINTERS -----------------------------------------------------

    /// Marker for GPU pointer types. Impl'd for `*mut E` and `*const E`
    /// where `E: ElementType`. `Pointer` is the read capability: every op
    /// that only loads is bound by it, so `*mut` works anywhere `*const`
    /// does — the same containment Rust's raw pointers have.
    pub trait Pointer {}
    #[cuda_tile::ty(name="!cuda_tile.tile", pointer_type="!cuda_tile.ptr", type_params=["!cuda_tile.ptr<E>"])]
    impl<E: ElementType> Pointer for *mut E {}
    #[cuda_tile::ty(name="!cuda_tile.tile", pointer_type="!cuda_tile.ptr", type_params=["!cuda_tile.ptr<E>"])]
    impl<E: ElementType> Pointer for *const E {}

    /// The write capability: only `*mut E`. Ops that store are bound by
    /// this, so a `*const E` can never reach a store without an explicit
    /// [`cast_mut`].
    pub trait PointerMut: Pointer {}
    impl<E: ElementType> PointerMut for *mut E {}

    /// Relates a pointer type to its pointee: `*mut E` and `*const E` are
    /// both `PointerTo<E>`. Generic memory ops use this so the element type
    /// is inferred from the pointer at the call site.
    pub trait PointerTo<E: ElementType>: Pointer {}
    impl<E: ElementType> PointerTo<E> for *mut E {}
    impl<E: ElementType> PointerTo<E> for *const E {}

    /// Tile of pointers — enables gather/scatter and indirect access.
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xP"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct PointerTile<P: Pointer, const D: [i32; N]> {
        _type: PhantomData<P>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<P: Pointer, const D: [i32; N]> PointerTile<P, D> {
        pub fn offset_tile<I: ElementType>(self, offset: Tile<I, D>) -> PointerTile<P, D> {
            addptr_tile(self, offset)
        }
        pub fn offset(self, offset: i32) -> PointerTile<P, D> {
            addptr(self, offset)
        }
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            broadcast_ptr(self, shape)
        }
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            reshape_ptr(self, shape)
        }
    }

    /// A supported device atomic storage type. Sealed to the supplied markers.
    ///
    /// These are device-DSL types, not host `std::sync::atomic` objects.
    /// `Value` is the scalar stored in device memory and returned by loads.
    pub trait Atomic: super::global_sealed::Atomic + Copy + Clone {
        type Value: ElementType;
    }

    /// Device atomic storage for `i32`.
    #[derive(Copy, Clone)]
    pub struct AtomicI32;
    impl super::global_sealed::Atomic for AtomicI32 {}
    impl Atomic for AtomicI32 {
        type Value = i32;
    }

    /// Device atomic storage for `u32`.
    #[derive(Copy, Clone)]
    pub struct AtomicU32;
    impl super::global_sealed::Atomic for AtomicU32 {}
    impl Atomic for AtomicU32 {
        type Value = u32;
    }

    /// Device atomic storage for `i64`.
    #[derive(Copy, Clone)]
    pub struct AtomicI64;
    impl super::global_sealed::Atomic for AtomicI64 {}
    impl Atomic for AtomicI64 {
        type Value = i64;
    }

    /// Device atomic storage for `u64`.
    #[derive(Copy, Clone)]
    pub struct AtomicU64;
    impl super::global_sealed::Atomic for AtomicU64 {}
    impl Atomic for AtomicU64 {
        type Value = u64;
    }

    /// Device atomic storage for `f32`.
    #[derive(Copy, Clone)]
    pub struct AtomicF32;
    impl super::global_sealed::Atomic for AtomicF32 {}
    impl Atomic for AtomicF32 {
        type Value = f32;
    }

    /// Device atomic storage for `f64`.
    #[derive(Copy, Clone)]
    pub struct AtomicF64;
    impl super::global_sealed::Atomic for AtomicF64 {}
    impl Atomic for AtomicF64 {
        type Value = f64;
    }

    /// Module-scope atomic global memory.
    ///
    /// `Global` is declared as a Rust `static` inside a `#[cutile::module]`.
    /// The Rust value is an immutable descriptor; mutability lives in the
    /// device storage and is exposed only through scoped atomic accesses.
    /// Use `Global<AtomicI32, { [] }>`, for example, rather than `Global<i32, { [] }>`.
    /// Weak accesses and tile-block-only scope are not available on this API.
    /// Tile IR 13.3+ also accepts `#[cuda_tile::global(constant=true,
    /// visibility="private")]` on the static. Constant globals reject stores
    /// and atomic updates; omitted options retain mutable/public behavior.
    ///
    /// Ordering follows Tile IR, not `std::sync::atomic`: these operations return
    /// completion tokens but accept no input token. They do not automatically
    /// order surrounding tensor or raw-pointer accesses; do not use them alone
    /// for acquire/release publication protocols, which also require explicit
    /// token dependencies. A load followed by a store is not an atomic RMW.
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct Global<A: Atomic, const D: [i32; N]> {
        _type: PhantomData<A>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    unsafe impl<A: Atomic, const D: [i32; N]> Sync for Global<A, D> {}

    #[cuda_tile::variadic_impl(N = 6)]
    impl<A: Atomic, const D: [i32; N]> Global<A, D> {
        /// Declare a global with a scalar initializer.
        ///
        /// Today the JIT compiler only lowers scalar globals
        /// (`Global<A, { [] }>`). Shaped globals are reserved for a follow-up pass.
        pub const fn new(_value: A::Value) -> Self {
            Self { _type: PhantomData }
        }

        /// Load the scalar global. Returns the loaded value and completion token.
        pub fn load<O: ordering::AtomicLoadMode, Sc: scope::GlobalScope>(
            &self,
            memory_ordering: O,
            memory_scope: Sc,
        ) -> (Tile<A::Value, D>, Token) {
            unreachable!()
        }

        /// Store to the scalar global. Returns the completion token.
        pub fn store<O: ordering::AtomicStoreMode, Sc: scope::GlobalScope>(
            &self,
            value: Tile<A::Value, D>,
            memory_ordering: O,
            memory_scope: Sc,
        ) -> Token {
            unreachable!()
        }

        /// Atomic add on the scalar global. Returns the old value and token.
        pub fn atomic_add<O: ordering::GlobalAtomicMode, Sc: scope::GlobalScope>(
            &self,
            value: Tile<A::Value, D>,
            memory_ordering: O,
            memory_scope: Sc,
        ) -> (Tile<A::Value, D>, Token) {
            unreachable!()
        }
    }

    // ---- §5.3 TENSOR TYPES (Tile, Tensor, Partition, PartitionMut) ---------

    /// Multi-dimensional array stored in registers / shared memory. The unit
    /// of compute inside a kernel.
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xE"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct Tile<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> Tile<E, D> {
        pub fn shape(&self) -> Shape<'_, D> {
            unreachable!()
        }
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> Tile<E, R> {
            broadcast(self, shape)
        }
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> Tile<E, R> {
            reshape(self, shape)
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<const D: [i32; N]> Tile<f4e2m1fnx2, D> {
        /// Unpack byte-addressable NVFP4 storage into logical `f4e2m1fn`
        /// values with the requested tile shape.
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn unpack<const R: [i32; M]>(self, shape: Shape<R>) -> Tile<f4e2m1fn, R> {
            __unpack_f4e2m1fnx2_tile(self, shape)
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<const D: [i32; N]> Tile<f4e2m1fn, D> {
        /// Pack logical `f4e2m1fn` values into byte-addressable NVFP4 storage
        /// with the requested tile shape.
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn pack<const R: [i32; M]>(self, shape: Shape<R>) -> Tile<f4e2m1fnx2, R> {
            __pack_f4e2m1fnx2_tile(self, shape)
        }
    }

    impl<E: ElementType, const M: i32, const N: i32> Tile<E, { [M, N] }> {
        /// Transpose a rank-2 tile.
        pub fn transpose(self) -> Tile<E, { [N, M] }> {
            permute(self, const_array![1, 0])
        }
    }

    // Operator overloads on `Tile`. Element-wise; `*` is the Hadamard product
    // — for matrix multiply use `mma`.
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Add<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn add(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Sub<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn sub(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Mul<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn mul(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Div<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn div(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Rem<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn rem(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitAnd<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitand(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitOr<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitor(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitXor<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitxor(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Shl<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn shl(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Shr<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn shr(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    /// Kernel-side view into GPU global memory. `-1` in `D` marks a dynamic dim.
    #[cuda_tile::ty(name="!cuda_tile.tensor_view",
                    type_params=["{D}xE", "strides"],
                    type_meta=["base", "shape", "strides", "token"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct Tensor<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const S: [i32; N]> Tensor<E, S> {
        /// This tensor's current ordering token: the completion token of the
        /// last load or store through it (or its entry token). Views created
        /// from the tensor start from this token.
        pub fn token(&self) -> Token {
            get_tensor_token(self)
        }
        /// Replace this tensor's ordering token; see [`set_tensor_token`].
        ///
        /// # Safety
        /// As for [`set_tensor_token`]: `token` must order every later
        /// access through this tensor after the writes it may observe.
        pub unsafe fn set_token(&self, token: Token) {
            set_tensor_token(self, token)
        }
        // TODO (hme): forward params/output_type_params like make_partition_view.
        pub fn partition<'a, const R: [i32; N]>(&'a self, tile: Shape<R>) -> Partition<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            // SAFETY: the view starts from this tensor's current ordering token.
            let p: Partition<E, R> = unsafe {
                make_partition_view(self, tile, padding::Zero, dim_map::Identity, tensor_token)
            };
            p
        }
        pub fn partition_permuted<'a, const R: [i32; N], const I: [i32; N]>(
            &'a self,
            tile: Shape<R>,
            dim_map: Array<I>,
        ) -> Partition<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            // SAFETY: the view starts from this tensor's current ordering token.
            let p: Partition<E, R> =
                unsafe { make_partition_view(self, tile, padding::Zero, dim_map, tensor_token) };
            p
        }
        /// Build a mutable tiled view of this tensor.
        ///
        /// Safe: constructing the view is not itself UB — the `&mut self` borrow
        /// gives exclusive access for the view's lifetime, and *every* store
        /// through the view is independently gated. The raw unchecked store
        /// ([`PartitionMut::store`]) is `unsafe`; the proof-carrying stores
        /// ([`PartitionMut::with_bounds`] → branded/block-id `coord`, and
        /// [`PartitionMut::store_index`] via a minted `PartitionIndex`) are safe.
        /// So no *unproven* store is reachable without `unsafe`. Mirrors the
        /// immutable [`Tensor::partition`], which is likewise safe.
        pub fn partition_mut<'a, const R: [i32; N]>(
            &'a mut self,
            tile: Shape<R>,
        ) -> PartitionMut<'a, E, R> {
            let tensor_token: Token = get_tensor_token(self);
            let outer_tile: Shape<S> = Shape::<S> { dims: &[] };
            // TODO (hme): document safety
            let mut p: PartitionMut<E, R> =
                unsafe { make_nested_partition_view_mut(self, tile, padding::None, tensor_token) };
            set_nested_mutable_partition_access_offset(&mut p, outer_tile);
            p
        }

        /// Build a mutable partition over the full tensor view.
        ///
        /// Unlike `Tensor::partition_mut`, this does not offset accesses by
        /// the current tile-block id. It is intended for schedule-driven
        /// kernels that store through private `PartitionIndex` values.
        pub unsafe fn partition_full_mut<'a, const R: [i32; N]>(
            &'a self,
            tile: Shape<R>,
        ) -> PartitionMut<'a, E, R> {
            let tensor_token: Token = get_tensor_token(self);
            unsafe { make_partition_view_mut(self, tile, padding::None, tensor_token) }
        }

        /// Returns the shape of this tensor.
        pub fn shape<'b>(&self) -> Shape<'b, S> {
            get_tensor_shape_meta(self)
        }

        /// The base device address of this tensor view.
        ///
        /// Safe, mirroring `<[T]>::as_ptr`: obtaining the address never
        /// touches memory. Dereferencing it — a gather load, or building a
        /// view over it — is `unsafe` and carries the safety conditions at
        /// those sites, exactly like dereferencing a raw pointer in Rust.
        pub fn as_ptr(&self) -> *const E {
            cast_const(tile_to_pointer(get_tensor_base(self)))
        }
        pub fn load_tile<const R: [i32; N]>(&self, shape: Shape<R>, idx: [i32; N]) -> Tile<E, R> {
            load_tile(self, shape, idx)
        }
    }

    /// Private logical index into a tensor partition grid.
    ///
    /// `PartitionIndex` values are produced by cutile schedule helpers. Safe
    /// mutable partition stores accept this type instead of raw coordinates so
    /// schedule construction is the boundary where bounds and disjointness are
    /// established.
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct PartitionIndex<const D: [i32; N]> {
        _type: PhantomData<()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<const D: [i32; N]> PartitionIndex<D> {
        pub fn coords(self) -> [i32; N] {
            partition_index_coords(self)
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<const D: [i32; N]> StoreIndex for PartitionIndex<D> {}

    impl<const D: [i32; 2]> PartitionIndex<D> {
        pub fn components(self) -> (i32, i32) {
            let coords = self.coords();
            (coords[0], coords[1])
        }
    }

    impl<const D: [i32; 3]> PartitionIndex<D> {
        pub fn components(self) -> (i32, i32, i32) {
            let coords = self.coords();
            (coords[0], coords[1], coords[2])
        }
    }

    /// Bounded partition dimension.
    ///
    /// Iterating a `Dim` produces index values branded by the JIT as bounded by
    /// that dimension.
    #[derive(Copy, Clone)]
    pub struct Dim {}

    impl Dim {
        #[deprecated(
            since = "0.3.0",
            note = "a hand-computed tile count carries no axis provenance. Use \
                    `num_tiles(&partition, axis)`, which names the axis it counts, \
                    and iterate it directly as `0..num_tiles(&p, axis)`."
        )]
        pub fn new(size: i32) -> Dim {
            dim_new(size)
        }

        pub fn value(self) -> i32 {
            dim_value(self)
        }
    }

    pub trait IntoDim {
        fn into_dim(self) -> Dim;
    }

    impl IntoDim for Dim {
        fn into_dim(self) -> Dim {
            self
        }
    }

    impl IntoDim for i32 {
        fn into_dim(self) -> Dim {
            dim_from_i32(self)
        }
    }

    impl From<i32> for Dim {
        fn from(value: i32) -> Self {
            dim_from_i32(value)
        }
    }

    pub struct DimIter {}

    impl Iterator for DimIter {
        type Item = i32;

        fn next(&mut self) -> Option<Self::Item> {
            unreachable!()
        }
    }

    impl IntoIterator for Dim {
        type Item = i32;
        type IntoIter = DimIter;

        fn into_iter(self) -> Self::IntoIter {
            unreachable!()
        }
    }

    /// Proof-carrying 2D coordinate produced from branded dimension indices.
    #[derive(Copy, Clone)]
    pub struct Coord2 {
        _type: PhantomData<()>,
    }

    /// Proof-carrying 3D coordinate produced from branded dimension indices.
    #[derive(Copy, Clone)]
    pub struct Coord3 {
        _type: PhantomData<()>,
    }

    /// Tuple of indices convertible into a proof-carrying coordinate.
    ///
    /// Each component must either be produced by iterating the `Dim` bound to
    /// the matching partition axis, or be a constant provably inside the
    /// axis's statically-known tile grid.
    pub trait CoordTuple {
        type Coord;
    }

    impl CoordTuple for (i32, i32) {
        type Coord = Coord2;
    }

    impl CoordTuple for (i32, i32, i32) {
        type Coord = Coord3;
    }

    #[cuda_tile::compiler_op(name = "coord")]
    pub fn coord<C: CoordTuple>(index: C) -> C::Coord {
        unreachable!()
    }

    /// Index forms accepted by [`MappedPartitionMut::store`].
    ///
    /// Either the whole minted [`PartitionIndex`] (every axis proven by the
    /// stream), or a [`coord`]-built composite whose streamed-axis components
    /// were minted by this partition's `iter_indices()` and whose owned-axis
    /// (`OWNED` map dim) components carry a `Dim` or constant bounds proof.
    pub trait StoreIndex {}

    impl StoreIndex for Coord2 {}

    impl StoreIndex for Coord3 {}

    /// Iterator marker for mapped partition indices.
    ///
    /// This is a zero-sized Rust/shadow-typing surface. The JIT compiler
    /// special-cases `for idx in mapped_partition.iter_indices()` and lowers it to
    /// a persistent tile-block loop that mints private `PartitionIndex`
    /// values.
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct PartitionIndices<const D: [i32; N], const M: [i32; N]> {
        _type: PhantomData<()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<const D: [i32; N], const M: [i32; N]> Iterator for PartitionIndices<D, M> {
        type Item = PartitionIndex<D>;

        fn next(&mut self) -> Option<Self::Item> {
            unreachable!()
        }
    }

    /// Mutable partition view whose valid indices are produced by a partition map.
    ///
    /// A `MappedPartitionMut` lowers to the same Tile IR partition-view type as
    /// `PartitionMut`, but its safe stores require a private `PartitionIndex`
    /// generated by the matching partition map.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view"],
                    type_meta=["token"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct MappedPartitionMut<E: ElementType, const D: [i32; N], const M: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N], const M: [i32; N]> MappedPartitionMut<E, D, M> {
        /// Iterate the private disjoint indices generated by this partition map.
        pub fn iter_indices(&self) -> PartitionIndices<D, M> {
            unreachable!()
        }

        /// Iterate indices accepted by stores into both this partition and `other`.
        ///
        /// Multi-output kernels write several tensors at the same logical tile
        /// index. Both partitions must share the tile and map shapes (enforced
        /// by the type), and their logical partition grids must match: static
        /// mismatches are compile errors, dynamic grids are checked at runtime.
        pub fn iter_indices_with<E2: ElementType>(
            &self,
            other: &MappedPartitionMut<E2, D, M>,
        ) -> PartitionIndices<D, M> {
            unreachable!()
        }

        /// Iterate a sub-range of the logical partition grid.
        ///
        /// `ranges` gives one `(start_tile, num_tiles)` pair per axis in tile
        /// units; a **literal** `num_tiles = -1` means "the rest of the axis
        /// from `start_tile`" — the spelling is resolved at compile time, so
        /// a runtime value of -1 is a negative length and fails the range
        /// assert. Only the covered tiles are visited, so a kernel
        /// updating `seq_len` rows of a `max_seq` cache iterates exactly the
        /// written tiles instead of the whole grid. Starts and lengths are
        /// runtime values (a start read from device memory works); each range
        /// is checked against the grid — statically when provable, otherwise
        /// with a runtime assert — so minted indices keep the in-bounds store
        /// proof.
        pub fn iter_indices_within(&self, ranges: [(i32, i32); N]) -> PartitionIndices<D, M> {
            unreachable!()
        }

        /// Sub-range iteration whose indices are also accepted by `other`:
        /// [`Self::iter_indices_within`] composed with
        /// [`Self::iter_indices_with`].
        pub fn iter_indices_within_with<E2: ElementType>(
            &self,
            ranges: [(i32, i32); N],
            other: &MappedPartitionMut<E2, D, M>,
        ) -> PartitionIndices<D, M> {
            unreachable!()
        }

        /// Store `tile` at a proven-disjoint partition index.
        ///
        /// Accepts the whole minted [`PartitionIndex`], or — on partitions
        /// with `OWNED` map axes — a [`coord`]-built composite: streamed-axis
        /// components must be minted by this partition's `iter_indices()`
        /// stream; owned-axis components carry a `Dim` or constant bounds
        /// proof (the stream item owns the whole axis, so bounds suffice).
        pub fn store<I: StoreIndex>(&mut self, tile: Tile<E, D>, index: I) -> Token {
            let coords: [i32; N] = validate_partition_store(self, index);
            unsafe {
                store_view_tko_mapped_mut(
                    self,
                    tile,
                    coords,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            }
        }
    }

    impl<E: ElementType, const D: [i32; 2], const M: [i32; 2]> MappedPartitionMut<E, D, M> {
        /// Map a flat persistent tile id into this partition's swizzled index.
        ///
        /// # Safety
        ///
        /// The caller must guarantee `tile_id` is in `0..num_bid_m*num_bid_n`
        /// and both partition-grid dimensions are positive. Prefer
        /// `MappedPartitionMut::iter_indices` when possible.
        pub unsafe fn index(
            &self,
            tile_id: i32,
            num_bid_m: i32,
            num_bid_n: i32,
        ) -> PartitionIndex<D> {
            unsafe { swizzle_partition_index_2d::<D, M>(tile_id, num_bid_m, num_bid_n) }
        }
    }

    /// Read-only tiled view of a `Tensor`. Index a tile via `partition.load([i, j, …])`.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view", "dim_map"],
                    type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct Partition<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a ()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, E: ElementType, const D: [i32; N]> Partition<'a, E, D> {
        pub fn load(&self, index: [i32; N]) -> Tile<E, D> {
            check_partition_access(self, index);
            let result: Tile<E, D> = load_view_tko(
                self,
                index,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            result
        }

        /// Checked load with a software-pipelining latency hint.
        ///
        /// Identical bounds checking to [`Self::load`]; `LATENCY` only feeds
        /// the Tile IR `optimization_hints.latency` knob and has no safety
        /// interaction. Use for loads on the critical path of pipelined
        /// loops, e.g. `part_k.load_pipelined::<4>([i, j])`.
        ///
        /// The cost of the bounds check depends on where the compiler can
        /// place it: see the book's "Bounds-Check Placement" chapter for the
        /// placement rules and how to inspect them (`CUTILE_JIT_LOG=1`).
        pub fn load_pipelined<const LATENCY: i32>(&self, index: [i32; N]) -> Tile<E, D> {
            check_partition_access(self, index);
            let result: Tile<E, D> = load_view_tko(
                self,
                index,
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            result
        }
    }

    impl<'a, E: ElementType, const D: [i32; 2]> Partition<'a, E, D> {
        #[deprecated(
            since = "0.3.0",
            note = "iterate `0..num_tiles(&p, axis)` and index with plain arrays; \
                    the compiler infers the same bounds, and an index walked from \
                    another tensor's axis is verified against this one at launch \
                    automatically. Migrate `Dim::new(expr)` loops to `num_tiles` \
                    rather than just dropping this call: a hand-computed count \
                    carries no axis provenance, so the access would fall back to \
                    a runtime check."
        )]
        pub fn with_bounds<A: IntoDim, B: IntoDim>(
            self,
            bounds: (A, B),
        ) -> BoundedPartition<'a, E, D> {
            partition_with_bounds(self, (bounds.0.into_dim(), bounds.1.into_dim()))
        }

        pub fn load_index(&self, index: PartitionIndex<D>) -> Tile<E, D> {
            self.load(index.coords())
        }
    }

    impl<'a, E: ElementType, const D: [i32; 3]> Partition<'a, E, D> {
        #[deprecated(
            since = "0.3.0",
            note = "iterate `0..num_tiles(&p, axis)` and index with plain arrays; \
                    the compiler infers the same bounds, and an index walked from \
                    another tensor's axis is verified against this one at launch \
                    automatically. Migrate `Dim::new(expr)` loops to `num_tiles` \
                    rather than just dropping this call: a hand-computed count \
                    carries no axis provenance, so the access would fall back to \
                    a runtime check."
        )]
        pub fn with_bounds<A: IntoDim, B: IntoDim, C: IntoDim>(
            self,
            bounds: (A, B, C),
        ) -> BoundedPartition<'a, E, D> {
            partition_with_bounds3(
                self,
                (
                    bounds.0.into_dim(),
                    bounds.1.into_dim(),
                    bounds.2.into_dim(),
                ),
            )
        }
    }

    /// Read-only partition whose valid axes have been tied to `Dim` values.
    ///
    /// Safe loads require `coord((...))` built from indices produced by those
    /// dimensions.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view", "dim_map"],
                    type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct BoundedPartition<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a ()>,
    }

    impl<'a, E: ElementType, const D: [i32; 2]> BoundedPartition<'a, E, D> {
        pub fn load(&self, index: Coord2) -> Tile<E, D> {
            check_bounded_partition_access(self, index);
            let result: Tile<E, D> = load_view_tko_bounded(
                self,
                coord2_as_array(index),
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            result
        }

        /// Checked load with a software-pipelining latency hint.
        ///
        /// Identical proof requirements to [`Self::load`]; `LATENCY` only
        /// feeds the Tile IR `optimization_hints.latency` knob and has no
        /// safety interaction.
        pub fn load_pipelined<const LATENCY: i32>(&self, index: Coord2) -> Tile<E, D> {
            check_bounded_partition_access(self, index);
            let result: Tile<E, D> = load_view_tko_bounded(
                self,
                coord2_as_array(index),
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            result
        }
    }

    impl<'a, E: ElementType, const D: [i32; 3]> BoundedPartition<'a, E, D> {
        pub fn load(&self, index: Coord3) -> Tile<E, D> {
            check_bounded_partition_access3(self, index);
            let result: Tile<E, D> = load_view_tko_bounded(
                self,
                coord3_as_array(index),
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            result
        }

        /// Checked load with a software-pipelining latency hint.
        ///
        /// Identical proof requirements to [`Self::load`]; `LATENCY` only
        /// feeds the Tile IR `optimization_hints.latency` knob and has no
        /// safety interaction.
        pub fn load_pipelined<const LATENCY: i32>(&self, index: Coord3) -> Tile<E, D> {
            check_bounded_partition_access3(self, index);
            let result: Tile<E, D> = load_view_tko_bounded(
                self,
                coord3_as_array(index),
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            result
        }
    }

    /// Mutable partition view. `store` is bounds-checked and safe; `load` is
    /// unordered and stays `unsafe` (a load may race a store to the same
    /// region, which bounds checking does not address).
    // TODO (hme): consolidate Partition + PartitionMut into a single type.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view"],
                    type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct PartitionMut<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a mut ()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, E: ElementType, const D: [i32; N]> PartitionMut<'a, E, D> {
        // Unordered load — caller must provide ordering via tokens or other sync.
        pub unsafe fn load(&self, index: [i32; N]) -> Tile<E, D> {
            let result: Tile<E, D> = unsafe {
                load_view_tko_mut(
                    self,
                    index,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            result
        }

        /// Stores a tile to this mutable partition at the specified index.
        ///
        /// The index is bounds-checked: the compiler proves it lies inside the
        /// partition grid, or emits a check that traps if it does not. An
        /// index the compiler can relate to the axis it indexes — one iterating
        /// `num_tiles(&p, a)`, or a constant inside a known extent — costs
        /// nothing at runtime. See the book's "Bounds-Check Placement" chapter
        /// for the placement rules and how to inspect them (`CUTILE_JIT_LOG=1`).
        ///
        /// Returns a token representing the completion of the store operation.
        pub fn store(&mut self, tile: Tile<E, D>, index: [i32; N]) -> Token {
            check_partition_access_mut(self, index);
            let token: Token = unsafe {
                store_view_tko_mut(
                    self,
                    tile,
                    index,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            token
        }

        /// [`Self::store`] without the bounds check.
        ///
        /// ## Safety
        ///
        /// The caller must guarantee `index` is within this partition's grid
        /// on every axis. An out-of-range index writes memory outside the
        /// partition. Prefer [`Self::store`], which proves the same property
        /// and costs nothing when the compiler can discharge it.
        pub unsafe fn store_unchecked(&mut self, tile: Tile<E, D>, index: [i32; N]) -> Token {
            let token: Token = unsafe {
                store_view_tko_mut(
                    self,
                    tile,
                    index,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            token
        }
    }

    impl<'a, E: ElementType, const D: [i32; 2]> PartitionMut<'a, E, D> {
        pub fn store_index(&mut self, tile: Tile<E, D>, index: PartitionIndex<D>) -> Token {
            self.store(tile, index.coords())
        }
    }

    /// Mutable partition whose valid axes have been tied to `Dim` values.
    ///
    /// The mutable mirror of [`BoundedPartition`]: safe stores require
    /// `coord((...))` built from indices produced by those dimensions.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view", "dim_map"],
                    type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct BoundedPartitionMut<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a mut ()>,
    }

    impl<'a, E: ElementType, const D: [i32; 2]> PartitionMut<'a, E, D> {
        /// Tie this mutable partition's axes to `Dim` bounds, mirroring
        /// [`Partition::with_bounds`]. Stores then require branded coords.
        #[deprecated(
            since = "0.3.0",
            note = "iterate `0..num_tiles(&p, axis)` and index with plain arrays; \
                    the compiler infers the same bounds, and an index walked from \
                    another tensor's axis is verified against this one at launch \
                    automatically. Migrate `Dim::new(expr)` loops to `num_tiles` \
                    rather than just dropping this call: a hand-computed count \
                    carries no axis provenance, so the access would fall back to \
                    a runtime check."
        )]
        pub fn with_bounds<A: IntoDim, B: IntoDim>(
            self,
            bounds: (A, B),
        ) -> BoundedPartitionMut<'a, E, D> {
            partition_with_bounds_mut(self, (bounds.0.into_dim(), bounds.1.into_dim()))
        }
    }

    impl<'a, E: ElementType, const D: [i32; 3]> PartitionMut<'a, E, D> {
        #[deprecated(
            since = "0.3.0",
            note = "iterate `0..num_tiles(&p, axis)` and index with plain arrays; \
                    the compiler infers the same bounds, and an index walked from \
                    another tensor's axis is verified against this one at launch \
                    automatically. Migrate `Dim::new(expr)` loops to `num_tiles` \
                    rather than just dropping this call: a hand-computed count \
                    carries no axis provenance, so the access would fall back to \
                    a runtime check."
        )]
        pub fn with_bounds<A: IntoDim, B: IntoDim, C: IntoDim>(
            self,
            bounds: (A, B, C),
        ) -> BoundedPartitionMut<'a, E, D> {
            partition_with_bounds_mut3(
                self,
                (
                    bounds.0.into_dim(),
                    bounds.1.into_dim(),
                    bounds.2.into_dim(),
                ),
            )
        }
    }

    impl<'a, E: ElementType, const D: [i32; 2]> BoundedPartitionMut<'a, E, D> {
        /// Checked store. Mirrors [`BoundedPartition::load`]: the index must be
        /// a `coord((...))` branded by this partition's bound `Dim`s.
        pub fn store(&mut self, tile: Tile<E, D>, index: Coord2) -> Token {
            check_bounded_partition_access_mut(self, index);
            let token: Token = unsafe {
                store_view_tko_bounded(
                    self,
                    tile,
                    coord2_as_array(index),
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            token
        }
    }

    impl<'a, E: ElementType, const D: [i32; 3]> BoundedPartitionMut<'a, E, D> {
        pub fn store(&mut self, tile: Tile<E, D>, index: Coord3) -> Token {
            check_bounded_partition_access_mut3(self, index);
            let token: Token = unsafe {
                store_view_tko_bounded(
                    self,
                    tile,
                    coord3_as_array(index),
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            token
        }
    }

    /// Memory-ordering token. Threaded through async memory ops to express
    /// dependencies; managed automatically by the load/store/partition APIs.
    #[cuda_tile::ty(name="!cuda_tile.token", params=[])]
    #[derive(Copy, Clone)]
    pub struct Token {}

    // ========================================================================
    // UTILITIES — compile-time descriptors, kernel intrinsics
    // ========================================================================

    /// Compile-time shape descriptor for tensors and tiles. Construct via
    /// [`shape!`].
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Shape<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, const D: [i32; N]> ops::Index<usize> for Shape<'a, D> {
        type Output = i32;

        fn index(&self, index: usize) -> &Self::Output {
            &self.dims[index]
        }
    }

    /// Construct a compile-time `Shape` from literal dims (0–4D supported).
    ///
    /// For the rare runtime-extent case, construct `Shape` directly:
    /// `Shape::<{ [-1] }> { dims: &[n] }`.
    #[macro_export]
    macro_rules! shape {
        () => {
            Shape_0::const_new()
        };
        ($x1:literal) => {
            Shape_1::<$x1>::const_new()
        };
        ($x1:literal, $x2:literal) => {
            Shape_2::<$x1, $x2>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal) => {
            Shape_3::<$x1, $x2, $x3>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal, $x4:literal) => {
            Shape_4::<$x1, $x2, $x3, $x4>::const_new()
        };
    }
    pub use shape;

    /// Construct a compile-time `Shape` from literal dims (0–4D supported).
    #[deprecated(note = "use shape![..]")]
    #[macro_export]
    macro_rules! const_shape {
        () => {
            Shape_0::const_new()
        };
        ($x1:literal) => {
            Shape_1::<$x1>::const_new()
        };
        ($x1:literal, $x2:literal) => {
            Shape_2::<$x1, $x2>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal) => {
            Shape_3::<$x1, $x2, $x3>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal, $x4:literal) => {
            Shape_4::<$x1, $x2, $x3, $x4>::const_new()
        };
    }
    #[allow(deprecated)]
    pub use const_shape;

    /// Compile-time index/metadata array — used for permutations and dim maps.
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Array<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, const D: [i32; N]> dim_map::Mode for Array<'a, D> {}

    /// Construct a compile-time `Array` from literal values (0–4D supported).
    #[macro_export]
    macro_rules! const_array {
        () => {
            Array_0::const_new()
        };
        ($x1:literal) => {
            Array_1::<$x1>::const_new()
        };
        ($x1:literal, $x2:literal) => {
            Array_2::<$x1, $x2>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal) => {
            Array_3::<$x1, $x2, $x3>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal, $x4:literal) => {
            Array_4::<$x1, $x2, $x3, $x4>::const_new()
        };
    }
    pub use const_array;

    /// Printf-style debug output from inside a kernel. `{}` becomes `%` at lowering.
    #[macro_export]
    macro_rules! cuda_tile_print {
        ($s:literal $(,$args:expr)*) => {
            unreachable!();
        };
    }
    pub use cuda_tile_print;

    /// Runtime assertion inside a kernel.
    #[macro_export]
    macro_rules! cuda_tile_assert {
        ($args:expr, $s:literal) => {
            unreachable!();
        };
    }
    pub use cuda_tile_assert;

    /// Grid dimensions `(gridDim.x, gridDim.y, gridDim.z)`.
    #[cuda_tile::op(name="cuda_tile.get_num_tile_blocks", since="V13_2", params=[])]
    pub fn get_num_tile_blocks() -> (i32, i32, i32) {
        unreachable!()
    }

    /// Current block id `(blockIdx.x, blockIdx.y, blockIdx.z)`.
    #[cuda_tile::op(name="cuda_tile.get_tile_block_id", since="V13_2", params=[])]
    pub fn get_tile_block_id() -> (i32, i32, i32) {
        unreachable!()
    }

    /// The index of the current tile program along `axis` of the launch
    /// grid, following Triton's `tl.program_id(axis)`.
    ///
    /// `axis` selects the grid dimension: 0, 1, or 2. The host derives a
    /// launch grid from a partition shape with partition axis `k` mapping
    /// to grid axis `k`, so the idiom
    /// `tensor.partition(shape).load([program_id(0)])` loads exactly the
    /// sub-tensor this tile program owns. Equivalent to the matching
    /// component of [`get_tile_block_id`]; a constant `axis` folds to
    /// that component at compile time. Axes above 2 fail a device
    /// assertion.
    // cuda_tile_assert! expands to unreachable!() for the host build.
    #[allow(unreachable_code)]
    pub fn program_id(axis: i32) -> i32 {
        let pid: (i32, i32, i32) = get_tile_block_id();
        if axis == 0 {
            pid.0
        } else if axis == 1 {
            pid.1
        } else {
            cuda_tile_assert!(axis == 2, "program_id: axis must be 0, 1, or 2");
            pid.2
        }
    }

    /// The number of tile programs along `axis` of the launch grid,
    /// following Triton's `tl.num_programs(axis)`.
    ///
    /// `axis` selects the grid dimension: 0, 1, or 2. Together with
    /// [`program_id`] this bounds the grid:
    /// `0 <= program_id(k) < num_programs(k)` for each axis `k`.
    /// Equivalent to the matching component of [`get_num_tile_blocks`];
    /// a constant `axis` folds to that component at compile time. Axes
    /// above 2 fail a device assertion.
    // cuda_tile_assert! expands to unreachable!() for the host build.
    #[allow(unreachable_code)]
    pub fn num_programs(axis: i32) -> i32 {
        let nb: (i32, i32, i32) = get_num_tile_blocks();
        if axis == 0 {
            nb.0
        } else if axis == 1 {
            nb.1
        } else {
            cuda_tile_assert!(axis == 2, "num_programs: axis must be 0, 1, or 2");
            nb.2
        }
    }

    /// Wrap a scalar in a 0-dim tile.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn scalar_to_tile<E: ElementType>(scalar: impl Scalar) -> Tile<E, { [] }> {
        unreachable!()
    }

    /// Unwrap a 0-dim tile back into a scalar.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_scalar<E: ElementType, S: Scalar>(tile: Tile<E, { [] }>) -> S {
        unreachable!()
    }

    /// Convert a scalar between element types.
    #[cuda_tile::compiler_op(name = "convert")]
    pub fn convert_scalar<S: Scalar>(x: impl Scalar) -> S {
        unreachable!()
    }

    /// Convert every element of a tile between element types.
    #[cuda_tile::compiler_op(name = "convert")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn convert_tile<TO: ElementType, FROM: ElementType, const S: [i32; N]>(
        x: Tile<FROM, S>,
    ) -> Tile<TO, S> {
        unreachable!()
    }

    /// Wrap a raw pointer as a 0-dim `PointerTile`.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn pointer_to_tile<P: Pointer>(ptr: P) -> PointerTile<P, { [] }> {
        unreachable!()
    }

    /// Unwrap a 0-dim `PointerTile` back into a raw pointer.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_pointer<P: Pointer>(tile: PointerTile<P, { [] }>) -> P {
        unreachable!()
    }

    /// Convert `*mut E` to `*const E`, mirroring `<*mut T>::cast_const`.
    /// Safe in both Rust and the DSL: the cast never changes what the
    /// pointer may do — capability is checked where memory is touched.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn cast_const<E: ElementType>(ptr: *mut E) -> *const E {
        unreachable!()
    }

    /// Convert `*const E` to `*mut E`, mirroring `<*const T>::cast_mut`.
    /// Safe like its Rust counterpart: obtaining the `*mut` is harmless;
    /// storing through it is the gated act — store ops are `unsafe` and
    /// require [`PointerMut`], carrying their own safety conditions.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn cast_mut<E: ElementType>(ptr: *const E) -> *mut E {
        unreachable!()
    }

    /// Elementwise [`cast_const`] over a pointer tile.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn cast_tile_const<E: ElementType, const D: [i32; N]>(
        tile: PointerTile<*mut E, D>,
    ) -> PointerTile<*const E, D> {
        unreachable!()
    }

    /// Elementwise [`cast_mut`] over a pointer tile.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn cast_tile_mut<E: ElementType, const D: [i32; N]>(
        tile: PointerTile<*const E, D>,
    ) -> PointerTile<*mut E, D> {
        unreachable!()
    }

    /// Bounds-check a partition access. Optimizer drops it when provably safe;
    /// otherwise emits an assertion at runtime.
    #[cuda_tile::compiler_op(name = "check")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn check_partition_access<E: ElementType, const S: [i32; N]>(
        part: &Partition<E, S>,
        index: [i32; N],
    ) {
        unreachable!()
    }

    /// [`check_partition_access`] for a mutable partition. Same check, same
    /// compiler handler; a separate declaration only because the view type
    /// differs.
    #[cuda_tile::compiler_op(name = "check")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn check_partition_access_mut<E: ElementType, const S: [i32; N]>(
        part: &PartitionMut<E, S>,
        index: [i32; N],
    ) {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "dim_new")]
    pub fn dim_new(size: i32) -> Dim {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "dim_from_i32")]
    pub fn dim_from_i32(size: i32) -> Dim {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "dim_value")]
    pub fn dim_value(dim: Dim) -> i32 {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "coord_as_array")]
    pub fn coord2_as_array(index: Coord2) -> [i32; 2] {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "coord_as_array")]
    pub fn coord3_as_array(index: Coord3) -> [i32; 3] {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "partition_with_bounds")]
    pub fn partition_with_bounds<'a, E: ElementType, const S: [i32; 2]>(
        part: Partition<'a, E, S>,
        bounds: (Dim, Dim),
    ) -> BoundedPartition<'a, E, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "partition_with_bounds")]
    pub fn partition_with_bounds3<'a, E: ElementType, const S: [i32; 3]>(
        part: Partition<'a, E, S>,
        bounds: (Dim, Dim, Dim),
    ) -> BoundedPartition<'a, E, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "check_bounded_partition_access")]
    pub fn check_bounded_partition_access<E: ElementType, const S: [i32; 2]>(
        part: &BoundedPartition<E, S>,
        index: Coord2,
    ) {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "check_bounded_partition_access")]
    pub fn check_bounded_partition_access3<E: ElementType, const S: [i32; 3]>(
        part: &BoundedPartition<E, S>,
        index: Coord3,
    ) {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "partition_with_bounds")]
    pub fn partition_with_bounds_mut<'a, E: ElementType, const S: [i32; 2]>(
        part: PartitionMut<'a, E, S>,
        bounds: (Dim, Dim),
    ) -> BoundedPartitionMut<'a, E, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "partition_with_bounds")]
    pub fn partition_with_bounds_mut3<'a, E: ElementType, const S: [i32; 3]>(
        part: PartitionMut<'a, E, S>,
        bounds: (Dim, Dim, Dim),
    ) -> BoundedPartitionMut<'a, E, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "check_bounded_partition_access")]
    pub fn check_bounded_partition_access_mut<E: ElementType, const S: [i32; 2]>(
        part: &BoundedPartitionMut<E, S>,
        index: Coord2,
    ) {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "check_bounded_partition_access")]
    pub fn check_bounded_partition_access_mut3<E: ElementType, const S: [i32; 3]>(
        part: &BoundedPartitionMut<E, S>,
        index: Coord3,
    ) {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "partition_index_coords")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn partition_index_coords<const D: [i32; N]>(index: PartitionIndex<D>) -> [i32; N] {
        unreachable!()
    }

    /// Validate a store index against a mapped partition and return its
    /// coordinates. Whole minted `PartitionIndex` values pass on stream
    /// provenance; `coord(...)` composites are checked per axis (streamed
    /// axes need minted components, owned axes need `Dim`/constant bounds
    /// proofs).
    #[cuda_tile::compiler_op(name = "validate_partition_store")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn validate_partition_store<
        E: ElementType,
        const D: [i32; N],
        const M: [i32; N],
        I: StoreIndex,
    >(
        view: &MappedPartitionMut<E, D, M>,
        index: I,
    ) -> [i32; N] {
        unreachable!()
    }

    /// Map a flat tile id to a swizzled 2D partition index without checks.
    ///
    /// This is the map primitive used by persistent GEMM-style kernels:
    /// CTAs iterate a flat tile-id stream, then this helper shapes each tile id
    /// into an in-bounds `[m, n]` partition index with swizzled ordering.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `tile_id` is in `0..num_bid_m*num_bid_n` and
    /// both partition-grid dimensions are positive. The returned
    /// `PartitionIndex` is a proof object, so constructing it with invalid
    /// inputs can make later safe partition stores out of bounds.
    #[cuda_tile::compiler_op(name = "swizzle_partition_index_2d")]
    pub unsafe fn swizzle_partition_index_2d<const D: [i32; 2], const M: [i32; 2]>(
        tile_id: i32,
        num_bid_m: i32,
        num_bid_n: i32,
    ) -> PartitionIndex<D> {
        unreachable!()
    }

    // ========================================================================
    // OPS § CORE — Tile IR §8.3
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/core.html
    // ========================================================================

    /// Broadcast a tile to a new shape (size-1 dims expand).
    #[cuda_tile::op(name="cuda_tile.broadcast", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn broadcast<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Concatenate `lhs` and `rhs` along `dim`. All other dims must match.
    #[cuda_tile::op(name="cuda_tile.cat", since="V13_2", params=["lhs", "rhs"], attribute_params=["dim:integer"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cat<E: ElementType, const SLhs: [i32; N], const SRhs: [i32; N], const SOut: [i32; N]>(
        lhs: Tile<E, SLhs>,
        rhs: Tile<E, SRhs>,
        dim: i32,
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// Tile filled with a compile-time `value`.
    #[cuda_tile::op(name="cuda_tile.constant", since="V13_2", params=[], attribute_params=["value:dense"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn constant<E: ElementType, const S: [i32; N]>(value: E, shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Extract a subtile. Result shape must evenly divide source shape.
    /// `indices` are slice indices (not byte offsets).
    #[cuda_tile::op(name="cuda_tile.extract", since="V13_2", params=["source", "...indices"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn extract<E: ElementType, const SIn: [i32; N], const SOut: [i32; N]>(
        source: Tile<E, SIn>,
        indices: [Tile<i32, { [] }>; N],
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// Insert a subtile at slice indices, not element offsets. Tile IR 13.4+.
    ///
    /// # Safety
    /// For each axis, `indices[axis]` must be in
    /// `0..(SOut[axis] / SIn[axis])`. Source shape must divide destination
    /// shape evenly. Out-of-bounds indices are undefined behavior.
    #[cuda_tile::op(name="cuda_tile.insert", since="V13_4", params=["source", "destination", "...indices"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn insert<E: ElementType, const SIn: [i32; N], const SOut: [i32; N]>(
        source: Tile<E, SIn>,
        destination: Tile<E, SOut>,
        indices: [Tile<i32, { [] }>; N],
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// 1D sequence `[0, 1, …, N-1]`.
    #[cuda_tile::op(name = "cuda_tile.iota", since = "V13_2")]
    pub fn iota<E: ElementType, const S: [i32; 1]>(shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Tensor-Core matrix multiply-accumulate: `result = lhs × rhs + acc`.
    #[cuda_tile::compiler_op(name = "mma")]
    pub fn mma<E1: ElementType, E2: ElementType, const M: i32, const N: i32, const K: i32>(
        lhs: Tile<E1, { [M, K] }>,
        rhs: Tile<E1, { [K, N] }>,
        acc: Tile<E2, { [M, N] }>,
    ) -> Tile<E2, { [M, N] }> {
        unreachable!()
    }

    /// Permute dimensions per the index array (e.g. `[1, 0]` = transpose).
    #[cuda_tile::op(name="cuda_tile.permute", since="V13_2", params=["source"], attribute_params=["permutation:array"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute<E: ElementType, const A: [i32; N], const I: [i32; N], const R: [i32; N]>(
        source: Tile<E, A>,
        permutation: Array<I>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reshape a tile (element count must match).
    #[cuda_tile::op(name="cuda_tile.reshape", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reshape<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Generic reduce along `dim` with closure `f` and `identity`.
    /// Closure body lowers to a Tile IR region at compile time.
    #[cuda_tile::op(name="cuda_tile.reduce", since="V13_2", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn reduce<E: ElementType, const S: [i32; N], F>(
        operand: Tile<E, S>,
        dim: i32,
        identity: E,
        f: F,
    ) -> Tile<E, S>
    where
        F: Fn(E, E) -> E,
    {
        unreachable!()
    }

    /// Min along `dim` (collapses that axis).
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_min<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Max along `dim`.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_max<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Sum along `dim`.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_sum<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Product along `dim`. Watch for overflow with large tiles.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_prod<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Prefix sum along `dim`. The compiler emits the addf/addi region.
    #[cuda_tile::op(name="cuda_tile.scan", since="V13_2", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan_sum<E: ElementType, const S: [i32; N], R: reverse::Mode>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: R,
        identity: E,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Generic prefix scan along `dim`. Closure body lowers to a Tile IR region.
    #[cuda_tile::op(name="cuda_tile.scan", since="V13_2", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan<E: ElementType, const S: [i32; N], R: reverse::Mode, F>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: R,
        identity: E,
        f: F,
    ) -> Tile<E, S>
    where
        F: Fn(E, E) -> E,
    {
        unreachable!()
    }

    /// `cond ? val_if_true : val_if_false` (element-wise).
    #[cuda_tile::op(name="cuda_tile.select", since="V13_2", params=["cond", "val_if_true", "val_if_false"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn select<E: ElementType, const S: [i32; N]>(
        cond: Tile<bool, S>,
        val_if_true: Tile<E, S>,
        val_if_false: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Creates a new unordered token.
    #[cuda_tile::op(name="cuda_tile.make_token", since="V13_2", params=[])]
    pub fn new_token_unordered() -> Token {
        unreachable!()
    }

    /// Allow PDL-dependent kernels to be scheduled. Tile IR 13.4+.
    /// Every CTA must signal or exit. This is not a memory fence; it is a
    /// no-op below sm_90 or without an applicable PDL launch relationship.
    ///
    /// # Safety
    /// Dependent kernels must wait before conflicting accesses. Neither
    /// kernel may rely on concurrent execution, and all allocations must
    /// remain alive until their last use, not just until this signal.
    #[cuda_tile::op(name="cuda_tile.gdc_launch_dependents_tko", since="V13_4", params=["token"])]
    pub unsafe fn gdc_launch_dependents_tko(token: Option<Token>) -> Token {
        unreachable!()
    }

    /// Wait for PDL predecessor completion and visibility of its stores.
    /// Tile IR 13.4+; a no-op below sm_90 or without a PDL launch relationship.
    ///
    /// # Safety
    /// Token-order every predecessor-dependent access after the returned
    /// token. Merely putting a load after this call in source is insufficient.
    /// Work before the wait must not race unfinished predecessor work.
    #[cuda_tile::op(name="cuda_tile.gdc_wait_tko", since="V13_4", params=["token"])]
    pub unsafe fn gdc_wait_tko(token: Option<Token>) -> Token {
        unreachable!()
    }

    /// Token-ordered proxy alias fence. Tile IR 13.4+.
    ///
    /// # Safety
    /// Chain earlier and later accesses through the input/output tokens.
    /// The fence does not establish allocation validity or cross-thread
    /// synchronization by itself; callers must uphold those contracts.
    #[cuda_tile::op(name="cuda_tile.memory_fence_alias_tko", since="V13_4", params=["token"])]
    pub unsafe fn memory_fence_alias_tko(token: Token) -> Token {
        unreachable!()
    }

    /// Allocate `num_elem` elements for the current IR block. Tile IR 13.3+.
    /// Size and nonzero power-of-two byte alignment must be compile-time constants.
    ///
    /// # Safety
    /// The allocation is uninitialized and cannot outlive its IR block.
    /// Local addresses must not escape the current tile thread. Global
    /// addresses may be shared only with proper synchronization, and no
    /// access may outlive the allocating block even in global mode.
    #[cuda_tile::op(name="cuda_tile.alloca", since="V13_3", params=[], attribute_params=["num_elem:integer", "alignment:integer"], static_params=["visibility={Global: global=unit}"])]
    pub unsafe fn alloca<E: ElementType, G: allocation::Mode>(
        num_elem: i64,
        alignment: i64,
        visibility: G,
    ) -> PointerTile<*mut E, { [] }> {
        unreachable!()
    }

    /// Combine independent tokens into one that depends on all of them.
    #[cuda_tile::op(name="cuda_tile.join_tokens", since="V13_2", params=["tokens"])]
    pub fn join_tokens(tokens: &[Token]) -> Token {
        unreachable!()
    }

    #[doc(hidden)]
    /// Low-level compiler helper; prefer `shape[index]`.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "shape")]
    pub fn get_shape_dim<const S: [i32; N]>(shape: Shape<S>, dim_idx: i32) -> i32 {
        unreachable!()
    }

    /// Extract a `Tensor`'s shape from its type metadata.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "shape")]
    pub fn get_tensor_shape_meta<'s, E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
    ) -> Shape<'s, S> {
        unreachable!()
    }

    /// Extract a `Tensor`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_tensor_token<E: ElementType, const S: [i32; N]>(tensor: &Tensor<E, S>) -> Token {
        unreachable!()
    }

    /// Extract a `Tensor`'s base pointer from its type metadata.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "base")]
    pub fn get_tensor_base<E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
    ) -> PointerTile<*mut E, { [] }> {
        unreachable!()
    }

    /// Replace a `Tensor`'s ordering token. Views created from the tensor
    /// afterwards start from `token`; views created before keep the old one.
    /// Install it on a function-level binding before creating any views.
    /// Explicit installation in conditionals, loops, or on block-local
    /// bindings is not supported. Do not rebind/shadow the tensor afterwards;
    /// the JIT rejects these cases until token updates use control-flow carries.
    ///
    /// # Safety
    /// The token must order every access the tensor's later views make
    /// after all writes they may observe. Installing an unrelated or older
    /// token lets a later load or store be scheduled before a write it
    /// depends on. The load/store APIs maintain this token themselves; set
    /// it only to splice in an external ordering point, such as a PDL wait.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_type_meta_field", type_meta_field = "token")]
    pub unsafe fn set_tensor_token<E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
        token: Token,
    ) {
        unreachable!()
    }

    // Internal completion-token update. Unlike explicit token installation,
    // loads/stores maintain their own dependencies inside control flow too.
    // SAFETY: callers pass the completion token of an access through `tensor`.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_access_token", type_meta_field = "token")]
    unsafe fn set_tensor_access_token<E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
        token: Token,
    ) {
        unreachable!()
    }

    /// Attach CTA-local index offset metadata to a nested mutable partition.
    /// Internal to [`Tensor::partition_mut`]: it rewrites the offsets every
    /// later store through the view uses, so it must not be reachable from
    /// kernel code.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_nested_mutable_partition_access_offset")]
    fn set_nested_mutable_partition_access_offset<
        'a,
        E: ElementType,
        const OUTER_TILE: [i32; N],
        const NESTED_TILE: [i32; N],
    >(
        partition: &mut PartitionMut<'a, E, NESTED_TILE>,
        outer_tile: Shape<OUTER_TILE>,
    ) {
        unreachable!()
    }

    /// Extract a `Partition`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token<E: ElementType, const D: [i32; N]>(view: &Partition<E, D>) -> Token {
        unreachable!()
    }

    /// Extract a `PartitionMut`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token_mut<E: ElementType, const D: [i32; N]>(
        view: &PartitionMut<E, D>,
    ) -> Token {
        unreachable!()
    }

    /// Number of tiles along `axis` (`cdiv(tensor_dim, tile_dim)`).
    /// `axis` must be a compile-time constant in `0..N`. Lowers to
    /// `cuda_tile.get_index_space_shape` with axis-th result extracted.
    #[cuda_tile::compiler_op(name = "num_tiles")]
    pub fn num_tiles<V>(view: &V, axis: i32) -> i32 {
        unreachable!()
    }

    // ========================================================================
    // OPS § CONVERSIONS — Tile IR §8.4
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/conversions.html
    // ========================================================================

    /// Float-to-float conversion (e.g. `f32` → `f16`).
    #[cuda_tile::op(name = "cuda_tile.ftof", since="V13_2", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>, NearestAway: rounding_mode=#cuda_tile.rounding<nearest_away>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ftof<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Float-to-integer conversion. Destination signedness is inferred from `EOut`.
    #[cuda_tile::op(name = "cuda_tile.ftoi", since="V13_2", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>, NearestIntToZero: rounding_mode=#cuda_tile.rounding<nearest_int_to_zero>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ftoi<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Float-to-integer conversion with explicit saturation. Enabling
    /// saturation requires Tile IR 13.4; disabled retains the older behavior.
    ///
    /// # Safety
    /// The operand and destination types must support the requested conversion.
    /// With saturation disabled, out-of-range inputs follow Tile IR semantics.
    #[cuda_tile::op(name="cuda_tile.ftoi", since="V13_2", params=["x"], named_attributes=["rounding_mode=nearest_int_to_zero"], static_params=["saturation={Enabled: saturating=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn ftoi_saturating<
        EIn: ElementType,
        EOut: ElementType,
        const S: [i32; N],
        SMode: saturating::Mode,
    >(
        x: Tile<EIn, S>,
        saturation: SMode,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer-to-float conversion. Source signedness is inferred from `EIn`.
    #[cuda_tile::op(name = "cuda_tile.itof", since="V13_2", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn itof<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer extension. Signedness is inferred from `EIn` by the JIT.
    #[cuda_tile::op(name="cuda_tile.exti", since="V13_2", params=["from"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exti<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        from: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer truncation.
    #[cuda_tile::op(name = "cuda_tile.trunci", since="V13_2", params = ["from"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn trunci<EIn: ElementType, EOut: ElementType, const S: [i32; N], O: overflow::Mode>(
        from: Tile<EIn, S>,
        overflow: O,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Bit-reinterpretation between same-size types.
    #[cuda_tile::op(name="cuda_tile.bitcast", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn bitcast<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Convert integer to pointer.
    #[cuda_tile::op(name="cuda_tile.int_to_ptr", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn int_to_ptr<SRC_T: ElementType, PTR_T: ElementType, const S: [i32; N]>(
        source: Tile<SRC_T, S>,
    ) -> PointerTile<*mut PTR_T, S> {
        unreachable!()
    }

    /// Convert pointer to integer.
    #[cuda_tile::op(name="cuda_tile.ptr_to_int", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_int<E: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Cast pointer type — reinterpret pointers as pointing to a different type.
    #[cuda_tile::op(name="cuda_tile.ptr_to_ptr", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_ptr<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut EIn, S>,
    ) -> PointerTile<*mut EOut, S> {
        unreachable!()
    }

    /// Attach the explicit `ptr_attr<none>` classification. Tile IR 13.4+.
    /// Tensor/view constructors preserve its presence; an unclassified pointer
    /// and an explicitly classified pointer are distinct Tile IR types.
    ///
    /// # Safety
    /// This does not create storage or establish access permissions. All
    /// subsequent accesses must satisfy the original pointer's invariants.
    #[cuda_tile::op(name="cuda_tile.ptr_to_ptr", since="V13_4", params=["source"], pointer_attribute="none")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn ptr_with_attr_none<E: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut E, S>,
    ) -> PointerTile<*mut E, S> {
        unreachable!()
    }

    // ========================================================================
    // OPS § NUMERIC / BITWISE — Tile IR §§8.7-8.9
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#floating-point
    // ========================================================================

    // ---- Integer arithmetic ------------------------------------------------

    /// Element-wise integer add. `overflow` lets the compiler assume no
    /// signed/unsigned/both wrap.
    #[cuda_tile::op(name = "cuda_tile.addi", since="V13_2", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer subtract.
    #[cuda_tile::op(name = "cuda_tile.subi", since="V13_2", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn subi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer multiply.
    #[cuda_tile::op(name = "cuda_tile.muli", since="V13_2", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn muli<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer divide. `Zero` rounding = truncation;
    /// `PositiveInf` = ceiling div; `NegativeInf` = floor div (signed only).
    #[cuda_tile::op(name = "cuda_tile.divi", since="V13_2", params = ["lhs", "rhs"], static_params = ["rounding={Zero: rounding=#cuda_tile.rounding<zero>, NearestEven: rounding=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding=#cuda_tile.rounding<negative_inf>, Approx: rounding=#cuda_tile.rounding<approx>, Full: rounding=#cuda_tile.rounding<full>}"], named_attributes = ["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn divi<E: ElementType, const S: [i32; N], R: rounding::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer remainder. Result sign matches dividend.
    #[cuda_tile::op(name = "cuda_tile.remi", since="V13_2", params = ["lhs", "rhs"], named_attributes = ["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn remi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer negation.
    #[cuda_tile::op(name = "cuda_tile.negi", since="V13_2", params = ["x"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        x: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer absolute value. Note: cuTile maps signed ints to `i64` (not `i32`).
    #[cuda_tile::op(name="cuda_tile.absi", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absi<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Multiply high — upper N bits of `x * y` for N-bit integer `E`.
    #[cuda_tile::op(name="cuda_tile.mulhii", since="V13_2", params=["x", "y"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mulhii<E: ElementType, const S: [i32; N]>(x: Tile<E, S>, y: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer maximum. Signedness inferred from `E`.
    #[cuda_tile::op(name="cuda_tile.maxi", since="V13_2", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn maxi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer minimum. Signedness inferred from `E`.
    #[cuda_tile::op(name="cuda_tile.mini", since="V13_2", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mini<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Integer Tensor-Core matrix multiply-accumulate.
    #[cuda_tile::op(
        name = "cuda_tile.mmai", since="V13_2",
        params = ["lhs", "rhs", "acc"],
        static_params = [
            "signedness_lhs={Signed: signedness_lhs=#cuda_tile.signedness<signed>, Unsigned: signedness_lhs=#cuda_tile.signedness<unsigned>}",
            "signedness_rhs={Signed: signedness_rhs=#cuda_tile.signedness<signed>, Unsigned: signedness_rhs=#cuda_tile.signedness<unsigned>}"
        ]
    )]
    #[cuda_tile::variadic_op(N = 3)]
    pub fn mmai<
        EIn: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
        SL: signedness::Mode,
        SR: signedness::Mode,
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<i32, ACC>,
        signedness_lhs: SL,
        signedness_rhs: SR,
    ) -> Tile<i32, ACC> {
        unreachable!()
    }

    // ---- Float arithmetic --------------------------------------------------

    /// Element-wise float add.
    #[cuda_tile::op(name="cuda_tile.addf", since="V13_2", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float subtract.
    #[cuda_tile::op(name="cuda_tile.subf", since="V13_2", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn subf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float multiply.
    #[cuda_tile::op(name="cuda_tile.mulf", since="V13_2", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mulf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float divide.
    #[cuda_tile::op(name="cuda_tile.divf", since="V13_2", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn divf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float remainder.
    #[cuda_tile::op(name = "cuda_tile.remf", since="V13_2", params = ["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn remf<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float negation.
    #[cuda_tile::op(name="cuda_tile.negf", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float absolute value.
    #[cuda_tile::op(name="cuda_tile.absf", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `atan2(x, y)`.
    #[cuda_tile::op(name = "cuda_tile.atan2", since="V13_2", params = ["x", "y"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atan2<E: ElementType, const S: [i32; N]>(x: Tile<E, S>, y: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Fused multiply-add: `lhs * rhs + acc` with one rounding step.
    #[cuda_tile::op(name="cuda_tile.fma", since="V13_2", params=["lhs", "rhs", "acc"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn fma<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        acc: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Floating-point Tensor-Core matrix multiply-accumulate.
    #[cuda_tile::op(name = "cuda_tile.mmaf", since="V13_2", params = ["lhs", "rhs", "acc"])]
    #[cuda_tile::variadic_op(N = 3)]
    pub fn mmaf<
        EIn: ElementType,
        EOut: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<EOut, ACC>,
    ) -> Tile<EOut, ACC> {
        unreachable!()
    }

    /// Tensor-Core multiply-accumulate with explicit fast accumulation.
    /// Tile IR 13.3+ when fast accumulation is enabled.
    ///
    /// # Safety
    /// The selected input/accumulator types must support fast accumulation;
    /// enabling it permits the reduced-precision accumulation defined by Tile IR.
    #[cuda_tile::op(name="cuda_tile.mmaf", since="V13_2", params=["lhs", "rhs", "acc"], static_params=["fast={Enabled: fast_acc=unit}"])]
    #[cuda_tile::variadic_op(N = 3)]
    pub unsafe fn mmaf_with_fast_acc<
        EIn: ElementType,
        EOut: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
        F: fast_acc::Mode,
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<EOut, ACC>,
        fast: F,
    ) -> Tile<EOut, ACC> {
        unreachable!()
    }

    /// Pack a numeric tile into bytes.
    ///
    /// This is the generic Tile IR pack operation for rank-1 sub-byte values.
    /// Reshape multi-dimensional tiles to rank 1 before packing, or use
    /// `Tile<f4e2m1fn, ...>::pack(shape)` for shaped NVFP4 tiles.
    #[cuda_tile::op(name = "cuda_tile.pack", since="V13_3", params = ["source"])]
    pub fn pack<EIn: ElementType, EByte: ByteElement, const S: [i32; 1], const R: [i32; 1]>(
        source: Tile<EIn, S>,
    ) -> Tile<EByte, R> {
        unreachable!()
    }

    /// Unpack a byte tile into a numeric tile.
    ///
    /// This is the generic Tile IR unpack operation for rank-1 sub-byte values.
    /// It can be used with `Tile<u8, ...>` when raw byte interop is needed.
    /// Reshape the result after unpacking if a matrix tile is needed, or use
    /// `Tile<f4e2m1fnx2, ...>::unpack(shape)` for shaped NVFP4 storage tiles.
    #[cuda_tile::op(name = "cuda_tile.unpack", since="V13_3", params = ["source"])]
    pub fn unpack<EOut: ElementType, EByte: ByteElement, const S: [i32; 1], const R: [i32; 1]>(
        source: Tile<EByte, S>,
    ) -> Tile<EOut, R> {
        unreachable!()
    }

    #[doc(hidden)]
    #[cuda_tile::compiler_op(name = "fp4_pack_unpack")]
    #[cuda_tile::variadic_op(N = 6, M = 6, method = "pack")]
    pub fn __pack_f4e2m1fnx2_tile<const S: [i32; N], const R: [i32; M]>(
        source: Tile<f4e2m1fn, S>,
        shape: Shape<R>,
    ) -> Tile<f4e2m1fnx2, R> {
        unreachable!()
    }

    #[doc(hidden)]
    #[cuda_tile::compiler_op(name = "fp4_pack_unpack")]
    #[cuda_tile::variadic_op(N = 6, M = 6, method = "unpack")]
    pub fn __unpack_f4e2m1fnx2_tile<const S: [i32; N], const R: [i32; M]>(
        source: Tile<f4e2m1fnx2, S>,
        shape: Shape<R>,
    ) -> Tile<f4e2m1fn, R> {
        unreachable!()
    }

    /// Floating-point Tensor-Core matrix multiply-accumulate with block-scaled inputs.
    ///
    /// For NVFP4, pass logical `Tile<f4e2m1fn, ...>` operands produced by
    /// unpacking packed `f4e2m1fnx2` or `u8` byte tiles. `lhs_scale` and
    /// `rhs_scale` provide the per-block scale values, typically with the K
    /// dimension reduced by the block size used by the quantized data layout.
    /// Accumulation is always `f32`.
    #[cuda_tile::op(
        name = "cuda_tile.mmaf_scaled", since="V13_3", min_arch="sm_100",
        params = ["lhs", "rhs", "acc", "lhs_scale", "rhs_scale"]
    )]
    #[cuda_tile::variadic_op(N = 3)]
    pub fn mmaf_scaled<
        EIn: ElementType,
        EScale: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
        const LHS_SCALE: [i32; N],
        const RHS_SCALE: [i32; N],
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<f32, ACC>,
        lhs_scale: Tile<EScale, LHS_SCALE>,
        rhs_scale: Tile<EScale, RHS_SCALE>,
    ) -> Tile<f32, ACC> {
        unreachable!()
    }

    /// Element-wise `source ^ exponent`.
    #[cuda_tile::op(name="cuda_tile.pow", since="V13_2", params=["source", "exponent"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn pow<E: ElementType, const S: [i32; N]>(
        source: Tile<E, S>,
        exponent: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Tile IR's floating-exponent spelling; `pow` remains available.
    ///
    /// # Safety
    /// Operands must satisfy the Tile IR floating-point power domain.
    #[cuda_tile::op(name="cuda_tile.fpowf", since="V13_2", params=["source", "exponent"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn fpowf<E: ElementType, const S: [i32; N]>(
        source: Tile<E, S>,
        exponent: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Floating-point power with a signed integer exponent. Tile IR 13.4+.
    ///
    /// # Safety
    /// The exponent element type must be i1, i8, i16, or i32 and is always
    /// interpreted as signed. Operands must satisfy the power domain.
    #[cuda_tile::op(name="cuda_tile.fpowi", since="V13_4", params=["source", "exponent"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn fpowi<E: ElementType, I: ElementType, const S: [i32; N]>(
        source: Tile<E, S>,
        exponent: Tile<I, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float maximum. `-0.0 < +0.0`.
    #[cuda_tile::op(name="cuda_tile.maxf", since="V13_2", params=["lhs", "rhs"], static_params=["nan={Enabled: propagate_nan=unit}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn maxf<E: ElementType, const S: [i32; N], P: nan::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        nan: P,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float minimum.
    #[cuda_tile::op(name="cuda_tile.minf", since="V13_2", params=["lhs", "rhs"], static_params=["nan={Enabled: propagate_nan=unit}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn minf<E: ElementType, const S: [i32; N], P: nan::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        nan: P,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    // ---- Min/Max + arithmetic helpers --------------------------------------

    /// Scalar minimum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn min<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Scalar maximum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn max<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise tile minimum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn min_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise tile maximum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn max_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Ceiling division `⌈a / b⌉` (scalar).
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn ceil_div<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise true (floating-point) division.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn true_div<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // ---- Math --------------------------------------------------------------

    /// Element-wise ceiling.
    #[cuda_tile::op(name="cuda_tile.ceil", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ceil<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floor.
    #[cuda_tile::op(name="cuda_tile.floor", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn floor<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise sine (radians).
    #[cuda_tile::op(name="cuda_tile.sin", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sin<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise cosine (radians).
    #[cuda_tile::op(name="cuda_tile.cos", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cos<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise tangent (radians).
    #[cuda_tile::op(name="cuda_tile.tan", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tan<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic sine.
    #[cuda_tile::op(name="cuda_tile.sinh", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sinh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic cosine.
    #[cuda_tile::op(name="cuda_tile.cosh", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cosh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic tangent.
    #[cuda_tile::op(name="cuda_tile.tanh", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tanh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Hyperbolic tangent with explicit Tile IR rounding.
    ///
    /// # Safety
    /// The element type and rounding mode must form a supported Tile IR pair.
    #[cuda_tile::op(name="cuda_tile.tanh", since="V13_2", params=["x"], static_params=["rounding={Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn tanh_with_rounding<E: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<E, S>,
        rounding: R,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `e^x`.
    #[cuda_tile::op(name="cuda_tile.exp", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Exponential with explicit Tile IR rounding. Nondefault modes need 13.3+.
    ///
    /// # Safety
    /// The element type and rounding mode must form a supported Tile IR pair.
    #[cuda_tile::op(name="cuda_tile.exp", since="V13_2", params=["x"], static_params=["rounding={Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn exp_with_rounding<E: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<E, S>,
        rounding: R,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `2^x`.
    #[cuda_tile::op(name="cuda_tile.exp2", since="V13_2", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp2<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise natural log.
    #[cuda_tile::op(name="cuda_tile.log", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise base-2 logarithm.
    #[cuda_tile::op(name="cuda_tile.log2", since="V13_2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log2<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise square root.
    #[cuda_tile::op(name="cuda_tile.sqrt", since="V13_2", params=["x"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sqrt<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        x: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `1 / sqrt(x)` (faster than `1.0 / sqrt(x)`).
    #[cuda_tile::op(name="cuda_tile.rsqrt", since="V13_2", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn rsqrt<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    // ---- Comparison --------------------------------------------------------

    /// Element-wise integer compare with explicit predicate.
    #[cuda_tile::op(
        name = "cuda_tile.cmpi", since="V13_2",
        params = ["lhs", "rhs"],
        static_params = ["predicate={Equal: comparison_predicate=#cuda_tile.cmp_predicate<equal>, NotEqual: comparison_predicate=#cuda_tile.cmp_predicate<not_equal>, LessThan: comparison_predicate=#cuda_tile.cmp_predicate<less_than>, LessThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<less_than_or_equal>, GreaterThan: comparison_predicate=#cuda_tile.cmp_predicate<greater_than>, GreaterThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<greater_than_or_equal>}"],
        named_attributes = ["signedness=inferred_signedness"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cmpi<E: ElementType, const S: [i32; N], P: predicate::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        predicate: P,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise float compare. `Ordered` requires both operands non-NaN;
    /// `Unordered` returns `true` if either is NaN.
    #[cuda_tile::op(
        name = "cuda_tile.cmpf", since="V13_2",
        params = ["lhs", "rhs"],
        static_params = [
            "predicate={Equal: comparison_predicate=#cuda_tile.cmp_predicate<equal>, NotEqual: comparison_predicate=#cuda_tile.cmp_predicate<not_equal>, LessThan: comparison_predicate=#cuda_tile.cmp_predicate<less_than>, LessThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<less_than_or_equal>, GreaterThan: comparison_predicate=#cuda_tile.cmp_predicate<greater_than>, GreaterThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<greater_than_or_equal>}",
            "ordering={Unordered: comparison_ordering=#cuda_tile.comparison_ordering<unordered>, Ordered: comparison_ordering=#cuda_tile.comparison_ordering<ordered>}"
        ]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cmpf<E: ElementType, const S: [i32; N], P: predicate::Mode, Ord: cmp_ordering::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        predicate: P,
        ordering: Ord,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    // `_tile` suffix is required by the compiler — bare names collide with
    // Rust's PartialEq/PartialOrd which can't be variadic-impl'd over CGAs.

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn eq_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ne_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn gt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ge_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn lt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn le_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    // ---- Bitwise -----------------------------------------------------------

    /// Element-wise bitwise AND.
    #[cuda_tile::op(name="cuda_tile.andi", since="V13_2", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn andi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise OR.
    #[cuda_tile::op(name="cuda_tile.ori", since="V13_2", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise XOR.
    #[cuda_tile::op(name="cuda_tile.xori", since="V13_2", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn xori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise NOT.
    #[cuda_tile::op(name = "cuda_tile.noti", since="V13_2", params = ["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn noti<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise left shift.
    #[cuda_tile::op(name = "cuda_tile.shli", since="V13_2", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shli<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise right shift. Arithmetic for signed `E`, logical for unsigned.
    #[cuda_tile::op(name="cuda_tile.shri", since="V13_2", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shri<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // ========================================================================
    // OPS § ATOMICS — Tile IR §8.10
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#atomics
    // ========================================================================

    /// Atomic read-modify-write. `mode` selects the op (`atomic::*`),
    /// `memory_ordering` excludes `Weak`. Returns `(old_values, token)`.
    ///
    /// # Safety
    /// Dereferences raw device pointers, like [`load_ptr_tko`] and
    /// [`store_ptr_tko`]: forming and offsetting the pointer tile is safe, but
    /// the read-modify-write is not. Every lane's address (masked lanes
    /// excepted — they are never dereferenced) must point to a valid,
    /// correctly-aligned `E` the kernel is allowed to write for the launch's
    /// lifetime, and the caller upholds the memory-model contract of the
    /// chosen `memory_ordering`/`memory_scope` with respect to every other
    /// access to that location.
    #[doc(hidden)]
    #[cuda_tile::op(name="cuda_tile.atomic_rmw_tko", since="V13_2", params=["pointers", "arg"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn atomic_rmw_tko<
        E: ElementType,
        const S: [i32; N],
        M: atomic::Mode,
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        pointers: PointerTile<*mut E, S>,
        arg: Tile<E, S>,
        mode: M,
        memory_ordering: O,
        memory_scope: Sc,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Atomic compare-and-swap. Bitwise comparison: NaN ≠ NaN, ±0.0 distinct
    /// if their bit patterns differ. Returns `(old_values, token)`.
    ///
    /// # Safety
    /// Dereferences raw device pointers, like [`load_ptr_tko`] and
    /// [`store_ptr_tko`]: forming and offsetting the pointer tile is safe, but
    /// the compare-and-swap is not. Every lane's address (masked lanes
    /// excepted — they are never dereferenced) must point to a valid,
    /// correctly-aligned `E` the kernel is allowed to write for the launch's
    /// lifetime, and the caller upholds the memory-model contract of the
    /// chosen `memory_ordering`/`memory_scope` with respect to every other
    /// access to that location.
    #[cuda_tile::op(name="cuda_tile.atomic_cas_tko", since="V13_2", params=["pointers", "cmp", "val"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn atomic_cas_tko<
        E: ElementType,
        const S: [i32; N],
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        pointers: PointerTile<*mut E, S>,
        cmp: Tile<E, S>,
        val: Tile<E, S>,
        memory_ordering: O,
        memory_scope: Sc,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    // ========================================================================
    // OPS § MEMORY — Tile IR §8.6
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#memory
    // ========================================================================

    /// Gather from a pointer tile. Returns `(loaded_values, token)`.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Acquire}; `Weak` drops scope.
    ///
    /// # Safety
    /// Dereferences raw device pointers. Like `core::ptr::read`, forming and
    /// offsetting the pointer tile is safe, but the load is not: every lane's
    /// address (masked lanes excepted) must point to a valid, correctly-aligned
    /// `E` the kernel is allowed to read for the launch's lifetime.
    #[cuda_tile::op(name="cuda_tile.load_ptr_tko", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_ptr_tko<
        E: ElementType,
        P: PointerTo<E>,
        const S: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        const CYCLES: i32,
    >(
        source: PointerTile<P, S>,
        memory_ordering: O,
        memory_scope: Option<Sc>,
        mask: Option<Tile<bool, S>>,
        padding_value: Option<E>,
        token: Option<Token>,
        latency: Latency<CYCLES>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Scatter to a pointer tile. Returns the completion token.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Release}; `Weak` drops scope.
    ///
    /// # Safety
    /// Dereferences raw device pointers. Like `core::ptr::write`, forming and
    /// offsetting the pointer tile is safe, but the store is not: every lane's
    /// address (masked lanes excepted) must point to a valid, correctly-aligned
    /// `E` the kernel is allowed to write, with no aliasing that would race
    /// another access, for the launch's lifetime.
    #[cuda_tile::op(name="cuda_tile.store_ptr_tko", since="V13_2", params=["destination", "value"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_ptr_tko<
        E: ElementType,
        P: PointerTo<E> + PointerMut,
        const S: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        const CYCLES: i32,
    >(
        destination: PointerTile<P, S>,
        value: Tile<E, S>,
        memory_ordering: O,
        memory_scope: Option<Sc>,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
        latency: Latency<CYCLES>,
    ) -> Token {
        unreachable!()
    }

    // ========================================================================
    // OPS § VIEWS — Tile IR §8.11
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#views
    // ========================================================================

    /// Raw load from a partition, strided, or gather/scatter view.
    /// `index` is an array or tuple in tensor-axis order; a gather's sparse
    /// axis takes a 1D integer tile, and the other axes take scalar indices.
    /// `inbounds` has one compile-time boolean per index (true needs 13.4+).
    ///
    /// # Safety
    /// All unpadded accesses must be valid. Each true inbounds entry promises
    /// that axis is in bounds. Supply a token ordering all conflicting accesses
    /// and keep the view's storage alive. Weak ordering ignores memory_scope.
    #[cuda_tile::op(name = "cuda_tile.load_view_tko", since = "V13_2")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_view_raw<
        E: ElementType,
        const S: [i32; N],
        V,
        I,
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &V,
        index: I,
        token: Option<Token>,
        memory_ordering: O,
        memory_scope: Sc,
        tma: T,
        latency: Option<i32>,
        inbounds: [bool; N],
        shape: Shape<S>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Raw store through a partition, strided, or gather/scatter view.
    /// Index, hint and inbounds conventions match [`load_view_raw`].
    ///
    /// # Safety
    /// The addressed storage must be writable, alive, and free of conflicting
    /// unordered accesses. True inbounds entries must hold for every lane.
    /// In particular, repeated scatter indices must not create a write race.
    #[cuda_tile::op(name = "cuda_tile.store_view_tko", since = "V13_2")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_view_raw<
        E: ElementType,
        const S: [i32; N],
        V,
        I,
        O: ordering::StoreMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &V,
        value: Tile<E, S>,
        index: I,
        token: Option<Token>,
        memory_ordering: O,
        memory_scope: Sc,
        tma: T,
        latency: Option<i32>,
        inbounds: [bool; N],
    ) -> Token {
        unreachable!()
    }

    /// Raw atomic reduction through an unpadded partition or strided view.
    /// Gather/scatter views and exchange mode are not supported by Tile IR.
    ///
    /// # Safety
    /// Every addressed location must be valid, writable and correctly aligned
    /// for the selected atomic operation. Order other conflicting accesses
    /// with the supplied token and keep storage alive through completion.
    #[cuda_tile::op(name = "cuda_tile.atomic_red_view_tko", since = "V13_3")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn atomic_red_view_raw<
        E: ElementType,
        const S: [i32; N],
        V,
        I,
        M: atomic::Mode,
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        view: &V,
        value: Tile<E, S>,
        index: I,
        mode: M,
        memory_ordering: O,
        memory_scope: Sc,
        token: Option<Token>,
    ) -> Token {
        unreachable!()
    }

    /// Query a partition view's index-space shape as scalar tile values.
    #[cuda_tile::op(name = "cuda_tile.get_index_space_shape", since="V13_2", params = ["src"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn get_index_space_shape<E: ElementType, const S: [i32; N]>(
        src: &Partition<E, S>,
    ) -> [i32; N] {
        unreachable!()
    }

    /// Query a tensor view's dynamic shape as scalar tile values.
    #[cuda_tile::op(name = "cuda_tile.get_tensor_shape", since="V13_2", params = ["src"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn get_tensor_shape<E: ElementType, const S: [i32; N]>(src: &Tensor<E, S>) -> [i32; N] {
        unreachable!()
    }

    /// Load a tile from a partition view at `index`. `tma::Disabled` suppresses
    /// TMA lowering. `memory_ordering` ⊆ {Weak, Relaxed, Acquire}.
    // TODO (hme): Mark loads from shared refs as unsafe and add `_unchecked` suffix.
    #[cuda_tile::op(name = "load_view_tko", since="V13_2", params = ["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_view_tko<
        E: ElementType,
        const D: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &Partition<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// `load_view_tko` for a proof-bounded read-only partition.
    #[cuda_tile::op(name = "load_view_tko", since="V13_2", params = ["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_view_tko_bounded<
        E: ElementType,
        const D: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &BoundedPartition<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// `load_view_tko` for `PartitionMut`. Caller must ensure no aliasing.
    // TODO (hme): document safety
    #[cuda_tile::op(name = "load_view_tko", since="V13_2", params = ["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_view_tko_mut<
        E: ElementType,
        const D: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &PartitionMut<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// Store a tile into a `BoundedPartitionMut` at a branded `index`. The
    /// mutable mirror of `load_view_tko_bounded`; lowers via the same
    /// `store_view_tko` op as `store_view_tko_mut`.
    #[cuda_tile::op(name = "store_view_tko", since="V13_2", params = ["view", "tile", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_view_tko_bounded<
        E: ElementType,
        const D: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &mut BoundedPartitionMut<E, D>,
        tile: Tile<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Token {
        unreachable!()
    }

    /// Store a tile into a `PartitionMut` at `index`. Returns the completion token.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Release}.
    // TODO (hme): document safety
    #[cuda_tile::op(name = "store_view_tko", since="V13_2", params = ["view", "tile", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_view_tko_mut<
        E: ElementType,
        const D: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &mut PartitionMut<E, D>,
        tile: Tile<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Token {
        unreachable!()
    }

    /// Store a tile into a mapped mutable partition at `index`.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Release}.
    #[cuda_tile::op(name = "store_view_tko", since="V13_2", params = ["view", "tile", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_view_tko_mapped_mut<
        E: ElementType,
        const D: [i32; N],
        const M: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &mut MappedPartitionMut<E, D, M>,
        tile: Tile<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Token {
        unreachable!()
    }

    /// Build a `Tensor` view from a base pointer, shape, strides, and a token.
    /// Caller must guarantee the layout is valid.
    #[cuda_tile::op(name="cuda_tile.make_tensor_view", since="V13_2",
                    params=["base", "shape.dims", "strides.dims"],
                    has_variadic_params=true,
                    output_type_params=["strides"],
                    output_type_meta=["base", "shape", "strides", "token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_tensor_view<E: ElementType, const D: [i32; N], const C: [i32; N]>(
        base: PointerTile<*mut E, { [] }>,
        shape: Shape<D>,
        strides: Array<C>,
        token: Token,
    ) -> Tensor<E, D> {
        unreachable!()
    }

    /// Build a read-only partition view. Prefer `tensor.partition(tile)`.
    ///
    /// Pass `padding::None` to omit the Tile IR `padding_value` type
    /// parameter, or a real padding marker such as `padding::Zero` to include
    /// it. Pass `dim_map::Identity` to omit `dim_map`, or an `Array` dim map
    /// to include it.
    ///
    /// # Safety
    /// `token` must order the view's loads after all writes they may observe.
    /// Passing a fresh unordered token after a tensor store would let safe
    /// `Partition::load` race that store. Normally pass `tensor_view.token()`.
    /// The returned lifetime is not tied to `tensor_view`; its storage must
    /// remain valid and no conflicting writes may overlap any view access.
    #[cuda_tile::op(name="cuda_tile.make_partition_view", since="V13_2",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value", "dim_map"],
                    output_type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_partition_view<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
        M: dim_map::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        tile: Shape<TILE_SHAPE>,
        padding_value: P,
        dim_map: M,
        token: Token,
    ) -> Partition<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Build a mutable partition view. Prefer `tensor.partition_mut(tile)`.
    ///
    /// Pass `padding::None` to omit the Tile IR `padding_value` type parameter,
    /// or a real padding marker such as `padding::Zero` to include it.
    ///
    /// # Safety
    ///
    /// The returned partition lifetime is not tied to the tensor argument, and
    /// callers must preserve the aliasing guarantees required by mutable view
    /// access.
    #[cuda_tile::op(name="cuda_tile.make_partition_view", since="V13_2",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token", "tensor_view.shape()"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_partition_view_mut<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: P,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Build a mapped mutable partition view.
    ///
    /// This is used by generated entry wrappers for
    /// `MappedPartitionMut<_, _, _>` parameters.
    #[cuda_tile::op(name="cuda_tile.make_partition_view", since="V13_2",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_mapped_partition_view<
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        const MAP_SHAPE: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: P,
        token: Token,
    ) -> MappedPartitionMut<E, TILE_SHAPE, MAP_SHAPE> {
        unreachable!()
    }

    /// Build a nested mutable partition view from an already partitioned
    /// mutable tensor. This still partitions the full tensor view; the compiler
    /// attaches metadata so tile accesses are offset by the enclosing CTA tile.
    #[cuda_tile::op(name="cuda_tile.make_partition_view", since="V13_2",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token", "tensor_view.shape()"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_nested_partition_view_mut<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: P,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    // ---- Core pointer helpers ---------------------------------------------

    /// Pointer to a kernel-module global declared via the `global` op.
    #[cuda_tile::op(name="cuda_tile.get_global", since="V13_2", params=[], named_attributes=["name:symbol_ref"])]
    pub fn get_global<E: ElementType>() -> PointerTile<*mut E, { [] }> {
        unreachable!()
    }

    /// `result[i] = ptr[i] + offset`.
    #[cuda_tile::op(name="cuda_tile.offset", since="V13_2", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr<P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: i32,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// `result[i] = ptr[i] + offset[i]`.
    #[cuda_tile::op(name="cuda_tile.offset", since="V13_2", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr_tile<I: ElementType, P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: Tile<I, D>,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// Broadcast a `PointerTile` to a new shape (size-1 dims expand).
    #[cuda_tile::op(name="cuda_tile.broadcast", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, method = "broadcast")]
    pub fn broadcast_ptr<P: Pointer, const S: [i32; N], const R: [i32; N]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    /// Reshape a `PointerTile` (element count must match).
    #[cuda_tile::op(name="cuda_tile.reshape", since="V13_2", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6, method = "reshape")]
    pub fn reshape_ptr<P: Pointer, const S: [i32; N], const R: [i32; M]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    // ========================================================================
    // OPS § MISC — Tile IR §8.12
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/misc.html
    // ========================================================================

    /// Token-ordered debug print for a single tile argument.
    #[cuda_tile::op(name = "cuda_tile.print_tko", since="V13_2", params = ["arg"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn print_tko<E: ElementType, const S: [i32; N]>(
        str: &str,
        arg: Tile<E, S>,
        token: Option<Token>,
    ) -> Token {
        unreachable!()
    }

    // ========================================================================
    // OPTIMIZATION ASSUMPTIONS
    // ========================================================================
    // Compile-time hints the optimizer treats as facts. UB if the property
    // doesn't hold at runtime — hence `unsafe`.

    /// Assert `x` is divisible by `DIVISOR`. UB if it isn't.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_div_by<T, const DIVISOR: i32>(x: T) -> T {
        unreachable!()
    }

    /// Full-width divisibility predicate; all predicate arguments are JIT constants.
    ///
    /// # Safety
    /// The promised divisibility must hold for every selected lane. `every`
    /// and `along` must both be present or both absent, as in Tile IR div_by.
    #[cuda_tile::op(name = "cuda_tile.assume", since = "V13_2")]
    pub unsafe fn assume_div_by_raw<T>(
        x: T,
        divisor: u64,
        every: Option<i64>,
        along: Option<i64>,
    ) -> T {
        unreachable!()
    }

    /// Full-width inclusive signed bounds; bounds are JIT constants.
    ///
    /// # Safety
    /// Every integer lane must lie within the provided signed bounds.
    #[cuda_tile::op(name = "cuda_tile.assume", since = "V13_2")]
    pub unsafe fn assume_bounds_raw<T>(x: T, lower: Option<i64>, upper: Option<i64>) -> T {
        unreachable!()
    }

    /// One same-elements group size per tile axis, including ranks above four.
    /// Group sizes are JIT constants.
    ///
    /// # Safety
    /// The integer/pointer lanes must satisfy the Tile IR same_elements
    /// predicate for every specified group.
    #[cuda_tile::op(name = "cuda_tile.assume", since = "V13_2")]
    pub unsafe fn assume_same_elements_raw<T, const N: usize>(x: T, groups: [i64; N]) -> T {
        unreachable!()
    }

    /// Assert every `every`-th element along dimension `along` is divisible by `divisor`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_div_by_every_along<
        T,
        const divisor: i32,
        const every: i32,
        const along: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    /// Assert `x >= LOWER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_lower<T, const LOWER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert `x <= UPPER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_upper<T, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert `LOWER <= x <= UPPER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds<T, const LOWER: i32, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each consecutive group of `GROUP0` are equal (1D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_1d<T, const GROUP0: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each `GROUP0 × GROUP1` block are equal (2D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_2d<T, const GROUP0: i32, const GROUP1: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each `GROUP0 × GROUP1 × GROUP2` block are equal (3D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_3d<
        T,
        const GROUP0: i32,
        const GROUP1: i32,
        const GROUP2: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    /// Assert elements within each `GROUP0 × … × GROUP3` block are equal (4D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_4d<
        T,
        const GROUP0: i32,
        const GROUP1: i32,
        const GROUP2: i32,
        const GROUP3: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    // ========================================================================
    // HIGH-LEVEL HELPERS
    // ========================================================================
    // Convenience wrappers built on the primitives above. Not Tile IR ops.

    /// Broadcast a scalar to a tile of the given shape.
    // `trait_name = "BroadcastScalarFn"` avoids colliding with the user
    // `BroadcastScalar` trait (PascalCase(broadcast_scalar) = BroadcastScalar).
    #[cuda_tile::variadic_op(N = 6, trait_name = "BroadcastScalarFn")]
    pub fn broadcast_scalar<E: ElementType, const S: [i32; N]>(
        x: E,
        shape: Shape<S>,
    ) -> Tile<E, S> {
        let ones_shape: Shape<{ [1; N] }> = Shape::<{ [1; N] }> { dims: &[1i32; N] };
        let tile_x: Tile<E, { [] }> = scalar_to_tile(x);
        tile_x.reshape(ones_shape).broadcast(shape)
    }

    /// Load a tile from `x` at `idx` with zero padding.
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_tile<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        x: &Tensor<E, S>,
        tile_shape: Shape<R>,
        idx: [i32; N],
    ) -> Tile<E, R> {
        let tensor_token: Token = get_tensor_token(x);
        // SAFETY: use the source tensor's own token and keep the view local.
        let x_partition: Partition<E, R> = unsafe {
            make_partition_view(
                x,
                tile_shape,
                padding::Zero,
                dim_map::Identity,
                tensor_token,
            )
        };
        let tile_x: Tile<E, R> = load_view_tko(
            &x_partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        tile_x
    }

    fn load_tile_mut_1d<E: ElementType, const S: [i32; 1]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        tile_y
    }

    fn load_tile_mut_2d<E: ElementType, const S: [i32; 2]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        tile_y
    }

    fn load_tile_mut_3d<E: ElementType, const S: [i32; 3]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        tile_y
    }

    fn store_tile_1d<E: ElementType, const S: [i32; 1]>(
        y: &mut Tensor<E, S>,
        result: Tile<E, S>,
    ) -> Token {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        new_token
    }

    fn store_tile_2d<E: ElementType, const S: [i32; 2]>(
        y: &mut Tensor<E, S>,
        result: Tile<E, S>,
    ) -> Token {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        new_token
    }

    fn store_tile_3d<E: ElementType, const S: [i32; 3]>(
        y: &mut Tensor<E, S>,
        result: Tile<E, S>,
    ) -> Token {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        // SAFETY: `new_token` is the completion token of the access this
        // function just made through `y`'s own view.
        unsafe { set_tensor_access_token(y, new_token) };
        new_token
    }

    /// Type-level dispatch for `load_tile_mut`.
    pub trait LoadTileMutAtCurrentBlock {
        type Out;

        fn load(&mut self) -> Self::Out;
    }

    impl<E: ElementType, const S: [i32; 1]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_1d(self)
        }
    }

    impl<E: ElementType, const S: [i32; 2]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_2d(self)
        }
    }

    impl<E: ElementType, const S: [i32; 3]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_3d(self)
        }
    }

    /// Load a mutable tensor tile at the current tile-block id.
    pub fn load_tile_mut<Y>(y: &mut Y) -> <Y as LoadTileMutAtCurrentBlock>::Out
    where
        Y: LoadTileMutAtCurrentBlock,
    {
        y.load()
    }

    /// Type-level dispatch for `store_tile`.
    pub trait StoreTileAtCurrentBlock<Result> {
        /// Store `result` at the current tile-block id; returns the store's
        /// completion token (also installed as the tensor's ordering token).
        fn store(&mut self, result: Result) -> Token;
    }

    impl<E: ElementType, const S: [i32; 1]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) -> Token {
            store_tile_1d(self, result)
        }
    }

    impl<E: ElementType, const S: [i32; 2]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) -> Token {
            store_tile_2d(self, result)
        }
    }

    impl<E: ElementType, const S: [i32; 3]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) -> Token {
            store_tile_3d(self, result)
        }
    }

    /// Store `result` into a mutable tensor tile at the current tile-block id.
    /// Returns the store's completion token.
    pub fn store_tile<Y, Result>(y: &mut Y, result: Result) -> Token
    where
        Y: StoreTileAtCurrentBlock<Result>,
    {
        StoreTileAtCurrentBlock::store(y, result)
    }

    /// Type-level dispatch for `load_tile_like`.
    /// Load from `self` the tile at the partition index this program owns
    /// in `y`.
    ///
    /// Contract: `self` must have the same shape as the whole tensor behind
    /// `y` — the full tensor the host partitioned, not `y`'s declared
    /// per-program slab — and `y` must be a mutable output tensor the host
    /// passed as a partition (`tensor.partition([..])` at the call site).
    /// The launch validator relates `self`'s tile count to the grid derived
    /// from `y`'s partition, so an input too small for the grid is rejected
    /// at launch.
    pub trait LoadTileLike<Y> {
        type Out;

        fn load_tile_like(&self, y: &Y) -> Self::Out;

        /// Method-form spelling of [`load_tile_like`]; same contract as the
        /// trait.
        fn load_like(&self, y: &Y) -> Self::Out;
    }

    impl<E1: ElementType, E2: ElementType, const X: [i32; 1], const S: [i32; 1]>
        LoadTileLike<Tensor<E2, S>> for Tensor<E1, X>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            // SAFETY: use the source tensor's own token and keep the view local.
            let x_partition: Partition<E1, S> = unsafe {
                make_partition_view(
                    x,
                    tile_shape,
                    padding::Zero,
                    dim_map::Identity,
                    tensor_token,
                )
            };
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }

        fn load_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            self.load_tile_like(y)
        }
    }

    impl<E1: ElementType, E2: ElementType, const X: [i32; 2], const S: [i32; 2]>
        LoadTileLike<Tensor<E2, S>> for Tensor<E1, X>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            // SAFETY: use the source tensor's own token and keep the view local.
            let x_partition: Partition<E1, S> = unsafe {
                make_partition_view(
                    x,
                    tile_shape,
                    padding::Zero,
                    dim_map::Identity,
                    tensor_token,
                )
            };
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }

        fn load_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            self.load_tile_like(y)
        }
    }

    impl<E1: ElementType, E2: ElementType, const X: [i32; 3], const S: [i32; 3]>
        LoadTileLike<Tensor<E2, S>> for Tensor<E1, X>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            // SAFETY: use the source tensor's own token and keep the view local.
            let x_partition: Partition<E1, S> = unsafe {
                make_partition_view(
                    x,
                    tile_shape,
                    padding::Zero,
                    dim_map::Identity,
                    tensor_token,
                )
            };
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }

        fn load_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            self.load_tile_like(y)
        }
    }

    /// Load a tile of `x` matching `y`'s shape, indexed by the current
    /// tile-block id. `x` must have the same shape as the whole tensor
    /// behind `y` (not `y`'s per-program slab), and `y` must be a mutable
    /// output tensor the host passed as a partition — see [`LoadTileLike`].
    pub fn load_tile_like<X, Y>(x: &X, y: &Y) -> <X as LoadTileLike<Y>>::Out
    where
        X: LoadTileLike<Y>,
    {
        x.load_tile_like(y)
    }

    /* TensorArray */
    // TODO (hme): Add a TensorArray type.
    //   #[cuda_tile::variadic_struct(N = 6)]
    //   struct TensorArray<E: ElementType, const D: [i32; N]> { _type: PhantomData<E> }

    /// Build a Tensor view from a strided pointer stored in `dst[idx]`.
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub unsafe fn load_tensor<T: ElementType, const S: [i32; N], const R: [i32; M]>(
        dst: &Tensor<i64, S>,
        idx: [i32; N],
        shape: Shape<R>,
        strides: Array<{ [-1; M] }>,
    ) -> Tensor<T, R> {
        let dims: &[i32] = &[];
        let ones_shape: Shape<{ [1; N] }> = Shape::<{ [1; N] }> { dims: dims };
        let dst_part: Partition<i64, { [1; N] }> = dst.partition(ones_shape);
        let dst_ptr_int: Tile<i64, { [1; N] }> = dst_part.load(idx);
        let dst_ptr_int: Tile<i64, { [] }> = dst_ptr_int.reshape(shape![]);
        let dst_ptr: PointerTile<*mut T, { [] }> = int_to_ptr(dst_ptr_int);
        let dst_tensor: Tensor<T, R> =
            unsafe { make_tensor_view(dst_ptr, shape, strides, new_token_unordered()) };
        dst_tensor
    }

    /// Permute a `[i32; N]` index array per `permutation`.
    #[cuda_tile::compiler_op(name = "shape")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute_array<const I: [i32; N]>(source: [i32; N], permutation: Array<I>) -> [i32; N] {
        unreachable!()
    }
}
